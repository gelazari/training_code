from __future__ import print_function
import os
import io
import sys
import cv2
import imageio
import torch
import json
import numpy as np
from scipy.interpolate import PchipInterpolator
from PIL import Image

# constants
SCALE_TOL = .1  # tolerance
TRAINEDW_DEG = 20
TRAINEDW_PX = 512
TRAINEDH_PX = 496
TRAIN_HMM = 1.9157
OUT_H = 512
OUT_W = 512


def get_boundaries(mask):
    mask = Image.fromarray(mask).convert('L')
    mask = np.array(mask)
    num_layers = len(np.unique(mask))
    boundaries = dict.fromkeys(np.arange(0, num_layers + 1))

    def bwareafilt(mask, n=1, area_range=(0, np.inf)):
        """Extract objects from binary image by size """
        areas_num, labels = cv2.connectedComponents(mask.astype(np.uint8))
        area_idx = np.arange(1, np.max(labels) + 1)
        areas = np.array([np.sum(labels == i) for i in area_idx])
        inside_range_idx = np.logical_and(areas >= area_range[0], areas <= area_range[1])
        area_idx = area_idx[inside_range_idx]
        areas = areas[inside_range_idx]
        keep_idx = area_idx[np.argsort(areas)[::-1][0:n]]
        kept_areas = areas[np.argsort(areas)[::-1][0:n]]
        if np.size(kept_areas) == 0:
            kept_areas = np.array([0])
        if n == 1:
            kept_areas = kept_areas[0]
        kept_mask = np.isin(labels, keep_idx)
        return kept_mask, kept_areas

    def fillHoles(mask):
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        len_contour = len(contours)
        contour_list = []
        for i in range(len_contour):
            drawing = np.zeros_like(mask, np.uint8)  # create a black image
            img_contour = cv2.drawContours(drawing, contours, i, (255, 255, 255), -1)
            contour_list.append(img_contour)

        out = sum(contour_list)
        return out

    def contiguous_regions(condition):
        """Finds contiguous True regions of the boolean array "condition". Returns
        a 2D array where the first column is the start index of the region and the
        second column is the end index."""

        # Find the indicies of changes in "condition"
        d = np.diff(condition)
        idx, = d.nonzero()
        # We need to start things after the change in "condition". Therefore,
        # we'll shift the index by 1 to the right.
        idx += 1
        if condition[0]:
            # If the start of condition is True prepend a 0
            idx = np.r_[0, idx]
        if condition[-1]:
            # If the end of condition is True, append the length of the array
            idx = np.r_[idx, condition.size]  # Edit
        # Reshape the result into two columns
        idx.shape = (-1, 2)
        return idx

    def get_boundary(boundary_present, boundary):
        diffbound = np.diff(boundary)
        flatBoundary = (diffbound == 0)
        segments = []
        for start, stop in contiguous_regions(flatBoundary == True):
            segment = flatBoundary[start:stop]
            segments.append((start, stop))
        segNodalPtsX = []
        segNodalPtsY = []
        x = 0
        while x < len(boundary) - 1:
            if flatBoundary[x] and boundary_present[x]:
                x_flat_ends = [item for item in segments if x in np.arange(item[0],
                                                                           item[1] + 1)][0]
                if not x_flat_ends:
                    x_flat_ends = 1
                mid_point_of_segment = (x_flat_ends[1] + x_flat_ends[0]) / 2
                diff = x_flat_ends[1] - x_flat_ends[0]
                segNodalPtsX.append(mid_point_of_segment)
                segNodalPtsY.append((boundary[x] - 0.5))
                x = x + diff
            else:
                x += 1
        ix = np.arange(len(boundary))
        iy = PchipInterpolator(segNodalPtsX, segNodalPtsY)(ix)
        # set to nan those before the first and after the last column where the layer is present
        itemindex = np.where(boundary_present == 1)
        boundary_present[np.min(itemindex):np.max(itemindex)] = 1
        iy[~boundary_present] = np.nan
        return iy

    # MASK PREPROCESSING (cleaning operations)

    # retain only main structures i.e. retina + choroid, the larger (> 10000 px) segmented areas (largest 10 of them,
    # effectively all)
    structures, kept_areas = bwareafilt(mask > 0, 10, (10000, np.inf))
    non_structures = structures == 0
    mask[non_structures] = 0

    # set all retina areas in the choroid to choroid (set to choroid anything that is not in main structure and not
    # in the main retina)
    retina, kept_areas = bwareafilt(np.logical_and(mask != 11, structures > 0), 10, (10000, np.inf))
    mask[~np.logical_or(retina, non_structures)] = 11;
    choroid = mask == 11;

    # Generous mask of main structures (retina + choroid) to eliminate spurious vitreous strands
    retina_dil = cv2.dilate(
        cv2.morphologyEx(fillHoles(np.uint8(retina)), cv2.MORPH_OPEN, cv2.getStructuringElement(2, [20, 20])),
        cv2.getStructuringElement(2, [10, 10]))
    choroid_dil = cv2.dilate(
        cv2.morphologyEx(np.uint8(choroid), cv2.MORPH_CLOSE, cv2.getStructuringElement(2, [100, 100])),
        cv2.getStructuringElement(2, [70, 70]))
    maskGenerous = np.uint8(np.logical_or(retina_dil, choroid_dil))

    # Take retina's top boundary as first white to black transition from the bottom of the flipud generous mask
    maskGenerous = np.uint8(np.logical_or(retina_dil, choroid_dil))
    retinaBoundary = mask.shape[0] - np.diff(np.flipud(maskGenerous), axis=0).argmax(axis=0)
    # and delete anything above it
    for col in range(0, mask.shape[1]):
        mask[0:retinaBoundary[col], col] = 0

    cleanMask = np.zeros(mask.shape)

    for j in range(1, num_layers):
        biggest, kept_areas = (bwareafilt(mask == j, 100, (200, np.inf)))
        if biggest.any():
            biggest = fillHoles(np.uint8(biggest))
            cleanMask[biggest > 0] = j
        else:
            message = "One layer not present, cannot continue"
            raise ValueError(message)

    mask = cleanMask

    for i in range(1, num_layers):

        if i == 1:  # ILM: top boundary is the top boundary of the whole inner retina (1 to 4)
            mk = np.logical_and(mask > 0, mask < 5)
            boundary_present = mk.max(axis=0)
            boundary = mk.argmax(axis=0) + 1
            boundaries[i] = get_boundary(boundary_present, boundary)
        elif 1 < i < 6:  # Inner layers: top boundary is the top boundary of inner retina  below (i to 5)
            mk = np.logical_and(mask > i - 1, mask < 6)
            boundary_present = mk.max(axis=0)
            boundary = mk.argmax(axis=0) + 1
            boundaries[i] = get_boundary(boundary_present, boundary)
        elif 5 < i < 11:  # Outer layers: top boundary is the top boundary of outer retina  below (i to 10)
            mk = np.logical_and(mask > i - 1, mask < 11)
            boundary_present = mk.max(axis=0)
            boundary = mk.argmax(axis=0) + 1
            boundaries[i] = get_boundary(boundary_present, boundary)
        elif i == 11:  # BM and Choroid:
            # top boundary (i.e. Bruch's Membrane - BM) obtained after a vertical flip of the 4<mask<11
            mk = np.flipud(np.logical_and(mask > 4, mask < 11))
            boundary_present = mk.max(axis=0)
            boundary = mk.argmax(axis=0)
            boundaries[i] = mask.shape[0] - get_boundary(boundary_present, boundary)
            # bottom boundary (Choroid) is obtained after a vertical flip of the mask == 11 in boundary 12
            mk = np.flipud(mask == 11)
            boundary = mk.argmax(axis=0)
            boundaries[12] = mask.shape[0] - get_boundary(boundary_present, boundary)
    boundaries[0] = np.zeros(mask.shape[1])  # Boundary 0 is empty (would be the top of the image)
    return boundaries


def get_2mm_y_range(image, hmm):
    h = np.shape(image)[0]
    profile = np.sum(image, axis=1).tolist()
    yMax = profile.index(max(profile))
    train_mm_inPixels = round(TRAIN_HMM * np.shape(image)[0] / hmm)
    halfH = round(train_mm_inPixels / 2)
    if (yMax - halfH) < 0:
        yrange = range(0, 2 * halfH + 1)
    elif (yMax + halfH + 1) > h - 1:
        yrange = range(h - 2 * halfH - 1, h)
    else:
        yrange = range(yMax - halfH, yMax + halfH + 1)
    return yrange


def get_hmm(machinecode):
    if type(machinecode) is str:
        try:
            # Heidelberg Engineering Spectralis (1 or 2)
            if machinecode.lower() == 'hespe1':
                return TRAIN_HMM  # 1.9157mm (3.87 microns * 496 px)
            # Heidelberg Engineering Spectralis (1 or 2)
            elif machinecode.lower() == 'hespe2':
                return TRAIN_HMM  # 1.9157mm (3.87 microns * 496 px)
            # Topcon Maestro
            elif machinecode.lower() == 'tcmaes':
                return 2.3
            # Topcon OCT-1000
            elif machinecode.lower() == 'tc1000':
                return 2.3
            # Topcon OCT-2000
            elif machinecode.lower() == 'tc2000':
                return 2.3
            # Topcon Triton
            elif machinecode.lower() == 'tctrit':
                return 2.54
            # Zeiss Cirrus
            elif machinecode.lower() == 'zscirr':
                return 2.0
            # Optovue iVue
            elif machinecode.lower() == 'ovivue':
                return 2.4
            # Optovue iScan
            elif machinecode.lower() == 'ovisca':
                return 2.3
        except ValueError:
            message = "Machine Code not found."
            print(message)
            return
    # leaving the option of a custom axial resolution in case machine is not there
    else:
        try:
            custom_hmm = float(machinecode)
            # range check
            if custom_hmm > 0 and custom_hmm < 10:
                return custom_hmm
            else:
                message = "Axial res is not valid."
                raise ValueError(message)
                return
        # type exception
        except ValueError:
            message = "Axial res is not valid."
            print(message)
            return


def predictScanForUser(filename,
                       model,
                       w_deg,
                       machinecode,
                       circular):
    image = imageio.imread(filename)
    image = np.array(image, dtype='float32')
    image_dim_total = len(image.shape)
    # get the input image(s) size, with "c" storing the number of images, if multiple images
    if image_dim_total == 2:
        (height, width) = image.shape
    elif image_dim_total == 3:
        (height, width, c) = image.shape
        image = image[:, :, 0]

    # if circular scan, the w_deg given is the radius and the max width in degrees is 70
    if circular:
        w_original = np.shape(image)[1]
        w_deg_actual = 2 * w_deg * np.pi
        w_deg = 20 + w_deg_actual
        w10deg_px = round(w_original / w_deg_actual * 10)
        first10deg = image[:, 0:w10deg_px]
        last10deg = image[:, -w10deg_px:]
        image = np.concatenate((last10deg, image, first10deg), 1)
        maxw_deg = 70
    # otherwise, the max width in degrees is 60
    else:
        maxw_deg = 60

    # get the image size and axial resolution
    h = np.shape(image)[0]
    w = np.shape(image)[1]
    hmm = get_hmm(machinecode)

    # if wdeg unsupported, display message
    if w_deg > maxw_deg:
        message = "Max field of view is 60 degrees (70 for circular scans), min is 10 degrees. Cannot segment the scan."
        print(message)
        return

    # if wdeg is smaller than the trained wdeg, add symmetric padding and process one scan
    elif w_deg < TRAINEDW_DEG:
        # initialize images
        bscan512x512 = np.empty((OUT_H, OUT_W))
        b = image

        # if axial window of the scan is > 2mm, crop a horizontal band of in the scan equal to 2mm and centred around
        # the peak of the scan's vertical, intensity-profile
        if hmm > 2:
            yrange = get_2mm_y_range(b, hmm)
            b = b[yrange, :]
            ystart = yrange[0]
            yend = yrange[len(yrange) - 1]
        w_px = int(TRAINEDW_PX / TRAINEDW_DEG * w_deg)

        # prepare the symmetrical, horizontally padded scan
        if w_px % 2 == 1:
            w_px = w_px + 1
        # scale width so that it is the same lateral resolution as the training
        trainingScaleBscan = np.array(Image.fromarray(b).convert('L').resize((w_px, OUT_H)))
        # add the flipped scan before and after (symm padding)
        f_trainingScaleBscan = np.flip(trainingScaleBscan, 1)
        bscan512x512 = \
            np.concatenate((f_trainingScaleBscan, trainingScaleBscan, f_trainingScaleBscan), 1)
        # make 512 px wide
        # cropping, if larger
        if np.shape(bscan512x512)[1] > TRAINEDW_PX:
            ignoreX = round((np.shape(bscan512x512)[1] - TRAINEDW_PX) / 2)
            bscan512x512 = bscan512x512[:, ignoreX:ignoreX + TRAINEDW_PX]
        # adding additional, zero-padding if still not 512 px wide after symm padding
        else:
            padWidth = round((TRAINEDW_PX - w_px) / 2)
            pad = np.zeros((OUT_H, padWidth))
            bscan512x512 = np.concatenate((pad, bscan512x512, pad), 1)

    # if wdeg is equal to the training wdeg (within scale tolerance defined)
    elif w_deg <= TRAINEDW_DEG * (1 + SCALE_TOL):
        b = image
        # if axial window of the scan is > 2mm, crop a horizontal band of in the scan equal to 2mm and centred around
        # the peak of the scan's vertical, intensity-profile
        if hmm > 2:
            yrange = get_2mm_y_range(b, hmm)
            b = b[yrange, :]
            ystart = yrange[0]
            yend = yrange[len(yrange) - 1]
        # just rescale to 512px width
        bscan512x512 = np.array(Image.fromarray(b).convert('L').resize((OUT_W, OUT_H), Image.BICUBIC))

    # if wdeg higher than the training wdeg (within scale tolerance defined), calculate the number of necessary
    # horizontal subdivisions, so that each subdivided scan has enough overlap to the following, and each subdivided
    # wdeg is equal to the training wdeg (within scale tolerance defined)
    else:
        N = int(np.ceil(w_deg / (10 * (1 + SCALE_TOL))))
        w_px = int(w / w_deg * 20)
        # make width in pixel even
        if w_px % 2 == 1:
            w_px = w_px + 1
        bscan512x512 = np.zeros((OUT_H, OUT_W, N)).astype("float32")
        yrange = [[] for i in range(N)]

        # FIRST SUBDIVISION: get the first subdivision i.e. left part of the original scan and w_px wide
        b = image[:, 0: w_px]
        if hmm > 2:  # if axial window of the scan is > 2mm, crop a horizontal band of in the scan equal to 2mm and
            # centred around the peak of the scan's vertical, intensity-profile
            yrange[0] = get_2mm_y_range(b, hmm)
            b = b[yrange[0], :]
        # add to the array of scans to be segmented in the first position (0)
        bscan512x512[:, :, 0] = np.array(Image.fromarray(b).convert('L').resize((OUT_W, OUT_H), Image.BICUBIC))

        # LAST SUBDIVISION: get the last (Nth) subdivision i.e. right part of the original scan and w_px wide
        b = image[:, np.shape(image)[1] - w_px:]
        if hmm > 2:  # if axial window of the scan is > 2mm, crop a horizontal band of in the scan equal to 2mm and
            # centred around the peak of the scan's vertical, intensity-profile
            yrange[N - 1] = get_2mm_y_range(b, hmm)
            b = b[yrange[N - 1], :]
        # add to the array of scans to be segmented in the last position (N-1)
        bscan512x512[:, :, N - 1] = np.array(Image.fromarray(b).convert('L').resize((OUT_W, OUT_H), Image.BICUBIC))

        # MIDDLE SUBDIVISIONS: get the middle subdivisions (i+1th centred between middle of point of first and last)

        # distanceBetweenFL =
        # distance between scans' middle: w of input image - 2 * half-width of 20deg-equivalent (w_px, i.e. half of first
        # and last), divided by the number of necessary scans-1
        dBSM = round((w - w_px) / (N - 1))
        # distance between scans' middle in training scale (px) =
        dBSM_train = round(512/w_px*dBSM)

        for i in range(1, N - 1):
            startX = dBSM * i  # round(w_px / 2 * (i - 1 + .5))
            b = image[:, startX: startX + w_px]
            if hmm > 2:  # if axial window of the scan is > 2mm, crop a horizontal band of in the scan equal to 2mm and
                # centred around the peak of the scan's vertical, intensity-profile
                yrange[i] = get_2mm_y_range(b, hmm)
                b = b[yrange[i], :]
            # add to the array of scans to be segmented in the relative position
            bscan512x512[:, :, i] = np.array(
                Image.fromarray(b).convert('L').resize((OUT_W, OUT_H), Image.BICUBIC))


    # SEGMENT THE SCAN

    seg512x512 = np.zeros_like(bscan512x512)
    bscan_dim_total = len(bscan512x512.shape)

    # if single scan
    if bscan_dim_total == 2:
        num_of_images = 1
        seg512x512 = segment(bscan512x512, model)[:, :, 0]

    # else if subdivisions
    elif bscan_dim_total == 3:
        (_, _, num_of_images) = bscan512x512.shape
        for num in range(num_of_images):
            res = segment(bscan512x512[:, :, num], model)
            seg512x512[:, :, num] = res[:, :, 0]


    # SEGMENTATION(S) BACK TO ORIGINAL image SIZE

    # if single scan
    if not ('N' in locals()):

        # if pad was added, remove padded width from the segmentation
        if w_deg < TRAINEDW_DEG:
            padWidth: int = round((512 - w_px) / 2)
            seg512x512 = seg512x512[:, padWidth: padWidth + w_px]
        # if axial cropping was necessary, restore in the right position
        if hmm > 2:
            hcrop = len(yrange)
            seg = np.zeros((h, w)).astype("float32")
            seg_backscaled = np.array(
                Image.fromarray(seg512x512).convert('L').resize((w, hcrop), resample=Image.NEAREST))
            seg[yrange, :] = seg_backscaled
        else:
            seg = np.array(Image.fromarray(seg512x512).convert('L').resize((w, h), resample=Image.NEAREST))

    # else if subdivision:
    else:
        for i in range(N):
            # if FIRST SUBDIVISIONS
            if i == 0:
                # take the first half-width + dBSM_train/2
                temp = seg512x512[:, 0:round(512/2+dBSM_train/2), i]
                if hmm > 2:  # if axial cropping was necessary
                    hcrop = len(yrange[i])
                    seg = np.zeros((h, temp.shape[1])).astype("float32")
                    seg_backscaled = np.array(Image.fromarray(temp).resize((temp.shape[1], hcrop), resample=Image.NEAREST))
                    seg[yrange[i], :] = seg_backscaled
                else:  # otherwise, just rescale height to image height
                    seg = np.array(Image.fromarray(temp).resize((temp.shape[1], h), resample=Image.NEAREST))
            # if MIDDLE SUBDIVISIONS
            elif i in range(1, N - 1):
                # take the middle half-width
                tempMid = seg512x512[:, int(512/2)-(round(dBSM_train/2)+(dBSM_train % 2)):
                                        int(512/2)+(round(dBSM_train/2)), i]
                if hmm > 2:  # if axial cropping was necessary
                    temp = np.zeros((h, tempMid.shape[1])).astype("float32")
                    seg_backscaled = np.array(Image.fromarray(tempMid).resize((tempMid.shape[1], hcrop), resample=Image.NEAREST))
                    temp[yrange[i], :] = seg_backscaled
                else:  # otherwise, just rescale height to image height
                    temp = np.array(Image.fromarray(tempMid).resize((tempMid.shape[1], h), resample=Image.NEAREST))

                seg = np.concatenate((seg, temp), 1)
            # if LAST SUBDIVISION
            else:
                # take the rest of from the last subdivision (this is not 256 if w_deg is not a multiple of 10)
                remaining_px = int(512/2)+(round(dBSM_train/2)+(dBSM_train % 2)) # round(512 / 20 * (w_deg - (N - 1) * 10))
                tempLast = seg512x512[:, 512 - remaining_px:, i]
                if hmm > 2:  # if axial cropping was necessary
                    temp = np.zeros((h, remaining_px)).astype("float32")
                    seg_backscaled = np.array(
                        Image.fromarray(tempLast).resize((remaining_px, hcrop), resample=Image.NEAREST))
                    temp[yrange[i], :] = seg_backscaled
                else:  # otherwise, just rescale height to image height
                    temp = np.array(Image.fromarray(tempLast).resize((remaining_px, h), resample=Image.NEAREST))
                seg = np.concatenate((seg, temp), 1)

        # Finally, rescale width to that of the original image
        seg = np.array(Image.fromarray(seg).resize((w, h), resample=Image.NEAREST))

    if circular:
        seg = seg[:, w10deg_px: w10deg_px + w_original]

    return seg


def segment(image, model):
    with torch.no_grad():
        device = 'cpu'
        image = np.array(image, dtype='float32')
        image_dim_total = len(image.shape)
        if image_dim_total == 2:
            (height, width) = image.shape
        elif image_dim_total == 3:
            (height, width, c) = image.shape
            image = image[:, :, 0]
        testimg = image.reshape(1, height, width)
        testimg = torch.from_numpy(testimg).to(device=device, dtype=torch.float32)
        c, h, w = testimg.size()
        testimg = testimg.expand(1, c, h, w)
        testoutput_original = model(testimg)
        _, testoutput_original = torch.max(testoutput_original, dim=1)
        testoutput_original = np.asarray(testoutput_original.cpu().detach().numpy(),
                                         dtype=np.uint8)
        testoutput_original = np.squeeze(testoutput_original, axis=0)
        testoutput_original = np.repeat(testoutput_original[:, :, np.newaxis], 3, axis=2)

        segmentation_map = np.zeros((h, w, 3), dtype=np.uint8)
        segmentation_map_view = np.zeros((h, w, 3), dtype=np.uint8)

        segmentation_map[:, :, 0][np.logical_and(testoutput_original[:, :, 0] == 0,
                                                 testoutput_original[:, :, 1] == 0,
                                                 testoutput_original[:, :, 2] == 0)] = 0
        segmentation_map[:, :, 1][np.logical_and(testoutput_original[:, :, 0] == 0,
                                                 testoutput_original[:, :, 1] == 0,
                                                 testoutput_original[:, :, 2] == 0)] = 0
        segmentation_map[:, :, 2][np.logical_and(testoutput_original[:, :, 0] == 0,
                                                 testoutput_original[:, :, 1] == 0,
                                                 testoutput_original[:, :, 2] == 0)] = 0
        #
        segmentation_map[:, :, 0][np.logical_and(testoutput_original[:, :, 0] == 1,
                                                 testoutput_original[:, :, 1] == 1,
                                                 testoutput_original[:, :, 2] == 1)] = 1
        segmentation_map[:, :, 1][np.logical_and(testoutput_original[:, :, 0] == 1,
                                                 testoutput_original[:, :, 1] == 1,
                                                 testoutput_original[:, :, 2] == 1)] = 1
        segmentation_map[:, :, 2][np.logical_and(testoutput_original[:, :, 0] == 1,
                                                 testoutput_original[:, :, 1] == 1,
                                                 testoutput_original[:, :, 2] == 1)] = 1
        #
        segmentation_map[:, :, 0][np.logical_and(testoutput_original[:, :, 0] == 2,
                                                 testoutput_original[:, :, 1] == 2,
                                                 testoutput_original[:, :, 2] == 2)] = 2
        segmentation_map[:, :, 1][np.logical_and(testoutput_original[:, :, 0] == 2,
                                                 testoutput_original[:, :, 1] == 2,
                                                 testoutput_original[:, :, 2] == 2)] = 2
        segmentation_map[:, :, 2][np.logical_and(testoutput_original[:, :, 0] == 2,
                                                 testoutput_original[:, :, 1] == 2,
                                                 testoutput_original[:, :, 2] == 2)] = 2
        #
        segmentation_map[:, :, 0][np.logical_and(testoutput_original[:, :, 0] == 3,
                                                 testoutput_original[:, :, 1] == 3,
                                                 testoutput_original[:, :, 2] == 3)] = 3
        segmentation_map[:, :, 1][np.logical_and(testoutput_original[:, :, 0] == 3,
                                                 testoutput_original[:, :, 1] == 3,
                                                 testoutput_original[:, :, 2] == 3)] = 3
        segmentation_map[:, :, 2][np.logical_and(testoutput_original[:, :, 0] == 3,
                                                 testoutput_original[:, :, 1] == 3,
                                                 testoutput_original[:, :, 2] == 3)] = 3
        #
        segmentation_map[:, :, 0][np.logical_and(testoutput_original[:, :, 0] == 4,
                                                 testoutput_original[:, :, 1] == 4,
                                                 testoutput_original[:, :, 2] == 4)] = 4
        segmentation_map[:, :, 1][np.logical_and(testoutput_original[:, :, 0] == 4,
                                                 testoutput_original[:, :, 1] == 4,
                                                 testoutput_original[:, :, 2] == 4)] = 4
        segmentation_map[:, :, 2][np.logical_and(testoutput_original[:, :, 0] == 4,
                                                 testoutput_original[:, :, 1] == 4,
                                                 testoutput_original[:, :, 2] == 4)] = 4
        #
        segmentation_map[:, :, 0][np.logical_and(testoutput_original[:, :, 0] == 5,
                                                 testoutput_original[:, :, 1] == 5,
                                                 testoutput_original[:, :, 2] == 5)] = 5
        segmentation_map[:, :, 1][np.logical_and(testoutput_original[:, :, 0] == 5,
                                                 testoutput_original[:, :, 1] == 5,
                                                 testoutput_original[:, :, 2] == 5)] = 5
        segmentation_map[:, :, 2][np.logical_and(testoutput_original[:, :, 0] == 5,
                                                 testoutput_original[:, :, 1] == 5,
                                                 testoutput_original[:, :, 2] == 5)] = 5
        #
        segmentation_map[:, :, 0][np.logical_and(testoutput_original[:, :, 0] == 6,
                                                 testoutput_original[:, :, 1] == 6,
                                                 testoutput_original[:, :, 2] == 6)] = 6
        segmentation_map[:, :, 1][np.logical_and(testoutput_original[:, :, 0] == 6,
                                                 testoutput_original[:, :, 1] == 6,
                                                 testoutput_original[:, :, 2] == 6)] = 6
        segmentation_map[:, :, 2][np.logical_and(testoutput_original[:, :, 0] == 6,
                                                 testoutput_original[:, :, 1] == 6,
                                                 testoutput_original[:, :, 2] == 6)] = 6
        #
        segmentation_map[:, :, 0][np.logical_and(testoutput_original[:, :, 0] == 7,
                                                 testoutput_original[:, :, 1] == 7,
                                                 testoutput_original[:, :, 2] == 1)] = 7
        segmentation_map[:, :, 1][np.logical_and(testoutput_original[:, :, 0] == 7,
                                                 testoutput_original[:, :, 1] == 7,
                                                 testoutput_original[:, :, 2] == 1)] = 7
        segmentation_map[:, :, 2][np.logical_and(testoutput_original[:, :, 0] == 7,
                                                 testoutput_original[:, :, 1] == 7,
                                                 testoutput_original[:, :, 2] == 1)] = 7
        #
        segmentation_map[:, :, 0][np.logical_and(testoutput_original[:, :, 0] == 8,
                                                 testoutput_original[:, :, 1] == 8,
                                                 testoutput_original[:, :, 2] == 2)] = 8
        segmentation_map[:, :, 1][np.logical_and(testoutput_original[:, :, 0] == 8,
                                                 testoutput_original[:, :, 1] == 8,
                                                 testoutput_original[:, :, 2] == 2)] = 8
        segmentation_map[:, :, 2][np.logical_and(testoutput_original[:, :, 0] == 8,
                                                 testoutput_original[:, :, 1] == 8,
                                                 testoutput_original[:, :, 2] == 2)] = 8
        #
        segmentation_map[:, :, 0][np.logical_and(testoutput_original[:, :, 0] == 9,
                                                 testoutput_original[:, :, 1] == 9,
                                                 testoutput_original[:, :, 2] == 3)] = 9
        segmentation_map[:, :, 1][np.logical_and(testoutput_original[:, :, 0] == 9,
                                                 testoutput_original[:, :, 1] == 9,
                                                 testoutput_original[:, :, 2] == 3)] = 9
        segmentation_map[:, :, 2][np.logical_and(testoutput_original[:, :, 0] == 9,
                                                 testoutput_original[:, :, 1] == 9,
                                                 testoutput_original[:, :, 2] == 3)] = 9
        #
        segmentation_map[:, :, 0][np.logical_and(testoutput_original[:, :, 0] == 10,
                                                 testoutput_original[:, :, 1] == 10,
                                                 testoutput_original[:, :, 2] == 4)] = 10
        segmentation_map[:, :, 1][np.logical_and(testoutput_original[:, :, 0] == 10,
                                                 testoutput_original[:, :, 1] == 10,
                                                 testoutput_original[:, :, 2] == 4)] = 10
        segmentation_map[:, :, 2][np.logical_and(testoutput_original[:, :, 0] == 10,
                                                 testoutput_original[:, :, 1] == 10,
                                                 testoutput_original[:, :, 2] == 4)] = 10
        #
        segmentation_map[:, :, 0][np.logical_and(testoutput_original[:, :, 0] == 11,
                                                 testoutput_original[:, :, 1] == 11,
                                                 testoutput_original[:, :, 2] == 5)] = 11
        segmentation_map[:, :, 1][np.logical_and(testoutput_original[:, :, 0] == 11,
                                                 testoutput_original[:, :, 1] == 11,
                                                 testoutput_original[:, :, 2] == 5)] = 11
        segmentation_map[:, :, 2][np.logical_and(testoutput_original[:, :, 0] == 11,
                                                 testoutput_original[:, :, 1] == 11,
                                                 testoutput_original[:, :, 2] == 5)] = 11
        #

        segmentation_map_view[:, :, 0][np.logical_and(testoutput_original[:, :, 0] == 0,
                                                      testoutput_original[:, :, 1] == 0,
                                                      testoutput_original[:, :, 2] == 0)] = 0
        segmentation_map_view[:, :, 1][np.logical_and(testoutput_original[:, :, 0] == 0,
                                                      testoutput_original[:, :, 1] == 0,
                                                      testoutput_original[:, :, 2] == 0)] = 0
        segmentation_map_view[:, :, 2][np.logical_and(testoutput_original[:, :, 0] == 0,
                                                      testoutput_original[:, :, 1] == 0,
                                                      testoutput_original[:, :, 2] == 0)] = 0
        #
        segmentation_map_view[:, :, 0][np.logical_and(testoutput_original[:, :, 0] == 1,
                                                      testoutput_original[:, :, 1] == 1,
                                                      testoutput_original[:, :, 2] == 1)] = 0
        segmentation_map_view[:, :, 1][np.logical_and(testoutput_original[:, :, 0] == 1,
                                                      testoutput_original[:, :, 1] == 1,
                                                      testoutput_original[:, :, 2] == 1)] = 255
        segmentation_map_view[:, :, 2][np.logical_and(testoutput_original[:, :, 0] == 1,
                                                      testoutput_original[:, :, 1] == 1,
                                                      testoutput_original[:, :, 2] == 1)] = 0
        #
        segmentation_map_view[:, :, 0][np.logical_and(testoutput_original[:, :, 0] == 2,
                                                      testoutput_original[:, :, 1] == 2,
                                                      testoutput_original[:, :, 2] == 2)] = 0
        segmentation_map_view[:, :, 1][np.logical_and(testoutput_original[:, :, 0] == 2,
                                                      testoutput_original[:, :, 1] == 2,
                                                      testoutput_original[:, :, 2] == 2)] = 0
        segmentation_map_view[:, :, 2][np.logical_and(testoutput_original[:, :, 0] == 2,
                                                      testoutput_original[:, :, 1] == 2,
                                                      testoutput_original[:, :, 2] == 2)] = 255
        #
        segmentation_map_view[:, :, 0][np.logical_and(testoutput_original[:, :, 0] == 3,
                                                      testoutput_original[:, :, 1] == 3,
                                                      testoutput_original[:, :, 2] == 3)] = 255
        segmentation_map_view[:, :, 1][np.logical_and(testoutput_original[:, :, 0] == 3,
                                                      testoutput_original[:, :, 1] == 3,
                                                      testoutput_original[:, :, 2] == 3)] = 255
        segmentation_map_view[:, :, 2][np.logical_and(testoutput_original[:, :, 0] == 3,
                                                      testoutput_original[:, :, 1] == 3,
                                                      testoutput_original[:, :, 2] == 3)] = 0
        #
        segmentation_map_view[:, :, 0][np.logical_and(testoutput_original[:, :, 0] == 4,
                                                      testoutput_original[:, :, 1] == 4,
                                                      testoutput_original[:, :, 2] == 4)] = 153
        segmentation_map_view[:, :, 1][np.logical_and(testoutput_original[:, :, 0] == 4,
                                                      testoutput_original[:, :, 1] == 4,
                                                      testoutput_original[:, :, 2] == 4)] = 51
        segmentation_map_view[:, :, 2][np.logical_and(testoutput_original[:, :, 0] == 4,
                                                      testoutput_original[:, :, 1] == 4,
                                                      testoutput_original[:, :, 2] == 4)] = 255
        #
        segmentation_map_view[:, :, 0][np.logical_and(testoutput_original[:, :, 0] == 5,
                                                      testoutput_original[:, :, 1] == 5,
                                                      testoutput_original[:, :, 2] == 5)] = 255
        segmentation_map_view[:, :, 1][np.logical_and(testoutput_original[:, :, 0] == 5,
                                                      testoutput_original[:, :, 1] == 5,
                                                      testoutput_original[:, :, 2] == 5)] = 102
        segmentation_map_view[:, :, 2][np.logical_and(testoutput_original[:, :, 0] == 5,
                                                      testoutput_original[:, :, 1] == 5,
                                                      testoutput_original[:, :, 2] == 5)] = 178
        #
        segmentation_map_view[:, :, 0][np.logical_and(testoutput_original[:, :, 0] == 6,
                                                      testoutput_original[:, :, 1] == 6,
                                                      testoutput_original[:, :, 2] == 6)] = 102
        segmentation_map_view[:, :, 1][np.logical_and(testoutput_original[:, :, 0] == 6,
                                                      testoutput_original[:, :, 1] == 6,
                                                      testoutput_original[:, :, 2] == 6)] = 255
        segmentation_map_view[:, :, 2][np.logical_and(testoutput_original[:, :, 0] == 6,
                                                      testoutput_original[:, :, 1] == 6,
                                                      testoutput_original[:, :, 2] == 6)] = 102
        #
        segmentation_map_view[:, :, 0][np.logical_and(testoutput_original[:, :, 0] == 7,
                                                      testoutput_original[:, :, 1] == 7,
                                                      testoutput_original[:, :, 2] == 1)] = 255
        segmentation_map_view[:, :, 1][np.logical_and(testoutput_original[:, :, 0] == 7,
                                                      testoutput_original[:, :, 1] == 7,
                                                      testoutput_original[:, :, 2] == 1)] = 0
        segmentation_map_view[:, :, 2][np.logical_and(testoutput_original[:, :, 0] == 7,
                                                      testoutput_original[:, :, 1] == 7,
                                                      testoutput_original[:, :, 2] == 1)] = 0
        #
        segmentation_map_view[:, :, 0][np.logical_and(testoutput_original[:, :, 0] == 8,
                                                      testoutput_original[:, :, 1] == 8,
                                                      testoutput_original[:, :, 2] == 2)] = 0
        segmentation_map_view[:, :, 1][np.logical_and(testoutput_original[:, :, 0] == 8,
                                                      testoutput_original[:, :, 1] == 8,
                                                      testoutput_original[:, :, 2] == 2)] = 255
        segmentation_map_view[:, :, 2][np.logical_and(testoutput_original[:, :, 0] == 8,
                                                      testoutput_original[:, :, 1] == 8,
                                                      testoutput_original[:, :, 2] == 2)] = 255
        #
        segmentation_map_view[:, :, 0][np.logical_and(testoutput_original[:, :, 0] == 9,
                                                      testoutput_original[:, :, 1] == 9,
                                                      testoutput_original[:, :, 2] == 3)] = 93
        segmentation_map_view[:, :, 1][np.logical_and(testoutput_original[:, :, 0] == 9,
                                                      testoutput_original[:, :, 1] == 9,
                                                      testoutput_original[:, :, 2] == 3)] = 63
        segmentation_map_view[:, :, 2][np.logical_and(testoutput_original[:, :, 0] == 9,
                                                      testoutput_original[:, :, 1] == 9,
                                                      testoutput_original[:, :, 2] == 3)] = 106
        #
        segmentation_map_view[:, :, 0][np.logical_and(testoutput_original[:, :, 0] == 10,
                                                      testoutput_original[:, :, 1] == 10,
                                                      testoutput_original[:, :, 2] == 4)] = 240
        segmentation_map_view[:, :, 1][np.logical_and(testoutput_original[:, :, 0] == 10,
                                                      testoutput_original[:, :, 1] == 10,
                                                      testoutput_original[:, :, 2] == 4)] = 163
        segmentation_map_view[:, :, 2][np.logical_and(testoutput_original[:, :, 0] == 10,
                                                      testoutput_original[:, :, 1] == 10,
                                                      testoutput_original[:, :, 2] == 4)] = 255
        #
        segmentation_map_view[:, :, 0][np.logical_and(testoutput_original[:, :, 0] == 11,
                                                      testoutput_original[:, :, 1] == 11,
                                                      testoutput_original[:, :, 2] == 5)] = 43
        segmentation_map_view[:, :, 1][np.logical_and(testoutput_original[:, :, 0] == 11,
                                                      testoutput_original[:, :, 1] == 11,
                                                      testoutput_original[:, :, 2] == 5)] = 206
        segmentation_map_view[:, :, 2][np.logical_and(testoutput_original[:, :, 0] == 11,
                                                      testoutput_original[:, :, 1] == 11,
                                                      testoutput_original[:, :, 2] == 5)] = 72

        return segmentation_map


def fullpipeline(folder_with_images="E:/bscans/",
                 w_deg=30,
                 machinecode='hespe1',
                 circular=False
                 ):

    # here we load the jit file to be used for predictions since the model class is unknown as if we're on the cloud
    model_file = "C:/Users/sbng352/Desktop/Low_rank_attention_OCT/jit_model_fluid.pth"

    # load
    model = torch.jit.load(model_file, map_location=torch.device('cpu'))

    output_folder = folder_with_images;
    # get all files in folder
    files = [os.path.join(folder_with_images, f)
             for f in os.listdir(folder_with_images) if os.path.isfile(os.path.join(folder_with_images, f))]
    filenames = [f for f in os.listdir(folder_with_images)
                 if os.path.isfile(os.path.join(folder_with_images, f))]

    for filename in files:
        # if there is a tif file and there is NOT a png file with the same name i.e. it has not already been processed
        if filename.endswith('tif') or filename.endswith('tiff'):
            print('File ' + filename + ':')
            if not os.path.isfile(filename[:-3] + 'png') and not os.path.isfile(filename[:-4] + 'png'):
                print('Processing...')
                prediction = predictScanForUser(filename,
                                                model,
                                                w_deg,
                                                machinecode,
                                                circular)

                img = Image.fromarray(prediction).convert('RGB')
                print('Saving results...')
                img.save(os.path.splitext(os.path.join(output_folder, filename))[0] + '.png', format='png')
                # BOUNDARIES
                try:
                    subpixel_boundaries = get_boundaries(prediction)
                    boundaries_json = json.dumps({int(k): v.tolist() for k, v in subpixel_boundaries.items()})
                    with open(os.path.splitext(os.path.join(output_folder, filename))[0] + '.json', 'w') as fp:
                        json.dump(boundaries_json, fp)
                except:
                    print('Error reading the boundaries (are all layers present?)')
            else:
                print('File already processed, skipping.')


if __name__ == '__main__':
    args = sys.argv

    if args.__len__() == 5:
        print('calling 2fullpipeline()" with specified wdeg, machinecode and circular')
        fullpipeline(args[1], float(args[2]), args[3], bool(args[4]))
    elif args.__len__() == 4:
        print('calling "fullpipeline()" with specified wdeg and machinecode')
        fullpipeline(args[1], float(args[2]), args[3])
    elif args.__len__() == 3:
        print('calling "fullpipeline()" with specified wdeg')
        fullpipeline(args[1], float(args[2]))
    elif args.__len__() == 2:
        print('calling "fullpipeline()" on ' + args[1] +
              ' with default values')
        #fullpipeline(args[1])
        for subdir, dirs, files in os.walk(args[1]):
            print(
                'calling "fullpipeline()" on test folder (' + subdir + ') with default values')
            fullpipeline(subdir)
    elif args.__len__() == 1:
        test_folder = 'E:/bscans/images/' #'F:/Backup_AngioOCT/GLAUCOMA/RETINA_HD/gregori_giuliana_OS_HD_Angio_Retina_159_201_5_1_M_FX/' #'F:/Topcon_TestingAllScanPatterns/Macula_5Lines'
        for subdir, dirs, files in os.walk(test_folder):
            print(
                'calling "fullpipeline()" on test folder (' + subdir + ') with default values')
            fullpipeline(subdir)
    else:
        print('Too many input args.')
