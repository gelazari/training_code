from __future__ import print_function
import os
import io
import imageio
import torch
import json
import numpy as np
from scipy.interpolate import PchipInterpolator
from PIL import Image

# constants
SCALE_TOL = .1
TRAINEDW_DEG = 20
TRAINEDW_PX = 512
TRAINEDH_PX = 496
TRAIN_HMM = 1.9157
OUT_H = 512
OUT_W = 512

########################## BITS TO EDIT ###########################################
model = torch.jit.load("C:/Users/sbng352/Desktop/Low_rank_attention_OCT/jit_model.pth",
                       map_location=torch.device('cpu'))

output_folder = "F:/Test_Optovue/ordered_scans/"
folder_with_images = "F:/Test_Optovue/ordered_scans/"

w_deg = 20
machinecode = 'ovivue'
circular = False
###################################################################################

def get_boundaries(mask):
    mask = Image.fromarray(mask).convert('L')
    mask = np.array(mask)
    num_layers = len(np.unique(mask))
    boundaries = dict.fromkeys(np.arange(0, num_layers + 1))

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

    for i in range(1, num_layers):
        print(i)
        boundary_present = (mask == i).max(axis=0)
        boundary = (mask == i).argmax(axis=0)
        boundaries[i] = get_boundary(boundary_present, boundary)
        if i == 11:
            # treat choroid in the opposite way when in layer 11
            boundary_present = (np.flipud(mask) == 11).max(axis=0)
            boundary = (np.flipud(mask) == i).argmax(axis=0)
            boundaries[12] = get_boundary(boundary_present, boundary)
            boundaries[12] = mask.shape[0] - boundaries[12] - 1
    boundaries[0] = np.zeros(mask.shape[1])
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
            return 2.3
        # Optovue iScan
        elif machinecode.lower() == 'ovisca':
            return 2.3
        # leaving the option of a custom axial resolution in case machine is not there
        else:
            try:
                return float(machinecode)
            except ValueError:
                message = "Machine Code not one of those in DB nor a float value."
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
    if image_dim_total == 2:
        (height, width) = image.shape
    elif image_dim_total == 3:
        (height, width, c) = image.shape
        image = image[:, :, 0]

    if circular:
        w_original = np.shape(image)[1]
        w_deg_actual = 2 * w_deg * np.pi
        w_deg = 20 + w_deg_actual
        w10deg_px = w_original / w_deg_actual * 10
        first10deg = image[:, 0:w10deg_px]
        last10deg = image[:, -w10deg_px:]
        image = np.concatenate((last10deg, image, first10deg), 1)
        maxw_deg = 70
    else:
        maxw_deg = 60

    h = np.shape(image)[0]
    w = np.shape(image)[1]
    hmm = get_hmm(machinecode)

    if w_deg > maxw_deg:
        message = "Max field of view is 60 degrees, min is 10 degrees. Cannot segment the scan."
        print(message)
        return
    elif w_deg < TRAINEDW_DEG:
        bscan512x512 = np.empty((OUT_H, OUT_W))
        b = image
        if hmm > 2:
            yrange = get_2mm_y_range(b, hmm)
            b = b[yrange, :]
            ystart = yrange[0]
            yend = yrange[len(yrange) - 1]
        w_px = int(TRAINEDW_PX / TRAINEDW_DEG * w_deg)
        if w_px % 2 == 1:
            w_px = w_px + 1
        trainingScaleBscan = np.array(Image.fromarray(b).convert('L').resize((w_px, OUT_H)))
        f_trainingScaleBscan = np.flip(trainingScaleBscan, 1)
        bscan512x512 = \
            np.concatenate((f_trainingScaleBscan, trainingScaleBscan, f_trainingScaleBscan), 1)

        if np.shape(bscan512x512)[1] > TRAINEDW_PX:
            ignoreX = round((np.shape(bscan512x512)[1] - TRAINEDW_PX) / 2)
            bscan512x512 = bscan512x512[ignoreX:ignoreX + TRAINEDW_PX, :]
        else:
            padWidth = round((TRAINEDW_PX - w_px) / 2)
            pad = np.zeros((OUT_H, padWidth))
            bscan512x512 = np.concatenate((pad, bscan512x512, pad), 1)
    elif w_deg <= TRAINEDW_DEG * (1 + SCALE_TOL):
        b = image
        if hmm > 2:
            yrange = get_2mm_y_range(b, hmm)
            b = b[yrange, :]
            ystart = yrange[0]
            yend = yrange[len(yrange) - 1]
        bscan512x512 = np.array(Image.fromarray(b).convert('L').resize((OUT_W, OUT_H)))
    else:
        N = int(np.ceil(w_deg / (10 * (1 + SCALE_TOL))))
        w_px = int(w / w_deg * 20)
        if w_px % 2 == 1:
            w_px = w_px + 1
        bscan512x512 = np.zeros((OUT_H, OUT_W, N)).astype("float32")
        yrange = [[] for i in range(N)]
        b = image[:, 0: w_px + 1]
        if hmm > 2:
            yrange[0] = get_2mm_y_range(b, hmm)
            b = b[yrange[0], :]
        bscan512x512[:, :, 0] = np.array(Image.fromarray(b).convert('L').resize((OUT_W, OUT_H)))
        b = image[:, np.shape(image)[1] - w_px + 1:-1]
        if hmm > 2:
            yrange[N - 1] = get_2mm_y_range(b, hmm)
            b = b[yrange[N - 1], :]
        bscan512x512[:, :, N - 1] = np.array(
            Image.fromarray(b).convert('L').resize((OUT_W, OUT_H)))
        for i in range(1, N - 1):
            startX = round(w_px / 2 * (i - 1 + .5))
            b = image[:, startX: startX + w_px - 1]
            if hmm > 2:
                yrange[i] = get_2mm_y_range(b, hmm)
                b = b[yrange[i], :]
            bscan512x512[:, :, i] = np.array(
                Image.fromarray(b).convert('L').resize((OUT_W, OUT_H)))
    # SEGMENT THE SCAN
    seg512x512 = np.zeros_like(bscan512x512)
    bscan_dim_total = len(bscan512x512.shape)
    if bscan_dim_total == 2:
        num_of_images = 1
        seg512x512 = segment(bscan512x512, model)[:, :, 0]
    elif bscan_dim_total == 3:
        (_, _, num_of_images) = bscan512x512.shape
        for num in range(num_of_images):
            res = segment(bscan512x512[:, :, num], model)
            seg512x512[:, :, num] = res[:, :, 0]
    # SEGMENTATION(S) BACK TO ORIGINAL image SIZE
    # if the image was NOT split:
    if not ('N' in locals()):
        # if axial cropping was necessary
        try:
            if any(yrange):
                hcrop = len(yrange)
                seg = np.zeros((h, w)).astype("float32")
                seg_backscaled = np.array(
                    Image.fromarray(seg512x512).convert('L').resize((w, hcrop)))
                seg[yrange, :] = seg_backscaled
        except:
            seg = np.array(Image.fromarray(seg512x512).convert('L').resize((w, h)))
    # if the image was split into 3 or more:
    else:
        for i in range(N):
            # if first image-split
            if i == 0:
                # take the first half-width
                temp = seg512x512[:, 0:int(512 / 2), i]
                # if axial cropping was necessary
                if any(yrange):
                    hcrop = len(yrange[i])
                    seg = np.zeros((h, 256)).astype("float32")
                    seg_backscaled = np.array(Image.fromarray(temp).resize((256, hcrop),
                                                                           resample=Image.NEAREST))
                    seg[yrange[i], :] = seg_backscaled
                else:
                    seg = np.array(Image.fromarray(temp).resize((256, h),
                                                                resample=Image.NEAREST))
            # if one of the middle scan-splits
            elif i in range(1, N - 1):
                # take the middle half-width
                tempMid = seg512x512[:, int(512 / 4): int(512 / 4 * 3), i]
                # if axial cropping was necessary
                if 'hcrop' in locals():
                    temp = np.zeros((h, 256)).astype("float32")
                    seg_backscaled = np.array(Image.fromarray(tempMid).resize((256, hcrop),
                                                                              resample=Image.NEAREST))
                    temp[yrange[i], :] = seg_backscaled
                # otherwise, just rescale height to image height
                else:
                    temp = np.array(Image.fromarray(tempMid).resize((256, h),
                                                                    resample=Image.NEAREST))

                seg = np.concatenate((seg, temp), 1)
            # if last scan split
            else:
                # take the last half-width
                tempLast = seg512x512[:, 512 - int(512 / 2):, i]
                # if axial cropping was necessary
                if 'hcrop' in locals():
                    temp = np.zeros((h, 256)).astype("float32")
                    seg_backscaled = np.array(Image.fromarray(tempLast).resize((256, hcrop),
                                                                               resample=Image.NEAREST))
                    temp[yrange[i], :] = seg_backscaled
                # otherwise, just rescale height to image height
                else:
                    temp = np.array(Image.fromarray(tempLast).resize((256, h),
                                                                     resample=Image.NEAREST))

                seg = np.concatenate((seg, temp), 1)
        # rescale width to that of the original image
        seg = np.array(Image.fromarray(seg).resize((w, h),
                                                   resample=Image.NEAREST))
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



files = [os.path.join(folder_with_images, f)
         for f in os.listdir(folder_with_images) if os.path.isfile(os.path.join(folder_with_images, f))]
filenames = [f for f in os.listdir(folder_with_images)
             if os.path.isfile(os.path.join(folder_with_images, f))]

for filename in files:
    if filename.endswith('tif') or filename.endswith('png') or filename.endswith('jpg'):
        prediction = predictScanForUser(filename,
                                        model,
                                        w_deg,
                                        machinecode,
                                        circular)
        subpixel_boundaries = get_boundaries(prediction)

        img = Image.fromarray(prediction).convert('RGB')
        img.save(os.path.splitext(os.path.join(output_folder, filename))[0]+'.png', format='png')

        boundaries_json = json.dumps({int(k): v.tolist() for k, v in subpixel_boundaries.items()})
        with open(os.path.splitext(os.path.join(output_folder, filename))[0]+'.json', 'w') as fp:
            json.dump(boundaries_json, fp)