from __future__ import print_function
from pathlib import Path
import gzip
import errno
import shutil
import collections

from PIL import Image

import tifffile as tiff
import imageio

import matplotlib.pyplot as plt

from torch.utils import data
#
import torch
import random
import numpy as np
import torch.nn as nn
import glob
import tifffile as tiff
import torch.nn.functional as F
from NNUtils import CustomDataset_OCT
from NNMetrics import segmentation_scores, f1_score, hd95, preprocessing_accuracy, intersectionAndUnion
import sys
import os
import imagecodecs


device = torch.device('cpu')
MODEL_PATH = 'C:/Users/sbng352/Desktop/OCT_CNNtraining/Results/final.pt'
model = torch.load(MODEL_PATH, map_location='cpu')
class_no = 12


def predict(file, output_folder):
    model.eval()
    prediction_map_path = output_folder
    with torch.no_grad():
        image = imageio.imread(file)
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
        with torch.no_grad():
            traced_cell = torch.jit.trace(model, (testimg))
        torch.jit.save(traced_cell, "jit_model.pth")
        testoutput_original = model(testimg)
        _, testoutput_original = torch.max(testoutput_original, dim=1)
        testoutput_original = np.asarray(testoutput_original.cpu().detach().numpy(), dtype=np.uint8)
        testoutput_original = np.squeeze(testoutput_original, axis=0)
        testoutput_original = np.repeat(testoutput_original[:, :, np.newaxis], 3, axis=2)

        segmentation_map = np.zeros((h, w, 3), dtype=np.uint8)
        segmentation_map_view = np.zeros((h, w, 3), dtype=np.uint8)

        segmentation_map[:, :, 0][np.logical_and(testoutput_original[:, :, 0] == 0, testoutput_original[:, :, 1] == 0,
                                                 testoutput_original[:, :, 2] == 0)] = 0
        segmentation_map[:, :, 1][np.logical_and(testoutput_original[:, :, 0] == 0, testoutput_original[:, :, 1] == 0,
                                                 testoutput_original[:, :, 2] == 0)] = 0
        segmentation_map[:, :, 2][np.logical_and(testoutput_original[:, :, 0] == 0, testoutput_original[:, :, 1] == 0,
                                                 testoutput_original[:, :, 2] == 0)] = 0
        #
        segmentation_map[:, :, 0][np.logical_and(testoutput_original[:, :, 0] == 1, testoutput_original[:, :, 1] == 1,
                                                 testoutput_original[:, :, 2] == 1)] = 1
        segmentation_map[:, :, 1][np.logical_and(testoutput_original[:, :, 0] == 1, testoutput_original[:, :, 1] == 1,
                                                 testoutput_original[:, :, 2] == 1)] = 1
        segmentation_map[:, :, 2][np.logical_and(testoutput_original[:, :, 0] == 1, testoutput_original[:, :, 1] == 1,
                                                 testoutput_original[:, :, 2] == 1)] = 1
        #
        segmentation_map[:, :, 0][np.logical_and(testoutput_original[:, :, 0] == 2, testoutput_original[:, :, 1] == 2,
                                                 testoutput_original[:, :, 2] == 2)] = 2
        segmentation_map[:, :, 1][np.logical_and(testoutput_original[:, :, 0] == 2, testoutput_original[:, :, 1] == 2,
                                                 testoutput_original[:, :, 2] == 2)] = 2
        segmentation_map[:, :, 2][np.logical_and(testoutput_original[:, :, 0] == 2, testoutput_original[:, :, 1] == 2,
                                                 testoutput_original[:, :, 2] == 2)] = 2
        #
        segmentation_map[:, :, 0][np.logical_and(testoutput_original[:, :, 0] == 3, testoutput_original[:, :, 1] == 3,
                                                 testoutput_original[:, :, 2] == 3)] = 3
        segmentation_map[:, :, 1][np.logical_and(testoutput_original[:, :, 0] == 3, testoutput_original[:, :, 1] == 3,
                                                 testoutput_original[:, :, 2] == 3)] = 3
        segmentation_map[:, :, 2][np.logical_and(testoutput_original[:, :, 0] == 3, testoutput_original[:, :, 1] == 3,
                                                 testoutput_original[:, :, 2] == 3)] = 3
        #
        segmentation_map[:, :, 0][np.logical_and(testoutput_original[:, :, 0] == 4, testoutput_original[:, :, 1] == 4,
                                                 testoutput_original[:, :, 2] == 4)] = 4
        segmentation_map[:, :, 1][np.logical_and(testoutput_original[:, :, 0] == 4, testoutput_original[:, :, 1] == 4,
                                                 testoutput_original[:, :, 2] == 4)] = 4
        segmentation_map[:, :, 2][np.logical_and(testoutput_original[:, :, 0] == 4, testoutput_original[:, :, 1] == 4,
                                                 testoutput_original[:, :, 2] == 4)] = 4
        #
        segmentation_map[:, :, 0][np.logical_and(testoutput_original[:, :, 0] == 5, testoutput_original[:, :, 1] == 5,
                                                 testoutput_original[:, :, 2] == 5)] = 5
        segmentation_map[:, :, 1][np.logical_and(testoutput_original[:, :, 0] == 5, testoutput_original[:, :, 1] == 5,
                                                 testoutput_original[:, :, 2] == 5)] = 5
        segmentation_map[:, :, 2][np.logical_and(testoutput_original[:, :, 0] == 5, testoutput_original[:, :, 1] == 5,
                                                 testoutput_original[:, :, 2] == 5)] = 5
        #
        segmentation_map[:, :, 0][np.logical_and(testoutput_original[:, :, 0] == 6, testoutput_original[:, :, 1] == 6,
                                                 testoutput_original[:, :, 2] == 6)] = 6
        segmentation_map[:, :, 1][np.logical_and(testoutput_original[:, :, 0] == 6, testoutput_original[:, :, 1] == 6,
                                                 testoutput_original[:, :, 2] == 6)] = 6
        segmentation_map[:, :, 2][np.logical_and(testoutput_original[:, :, 0] == 6, testoutput_original[:, :, 1] == 6,
                                                 testoutput_original[:, :, 2] == 6)] = 6
        #
        segmentation_map[:, :, 0][np.logical_and(testoutput_original[:, :, 0] == 7, testoutput_original[:, :, 1] == 7,
                                                 testoutput_original[:, :, 2] == 1)] = 7
        segmentation_map[:, :, 1][np.logical_and(testoutput_original[:, :, 0] == 7, testoutput_original[:, :, 1] == 7,
                                                 testoutput_original[:, :, 2] == 1)] = 7
        segmentation_map[:, :, 2][np.logical_and(testoutput_original[:, :, 0] == 7, testoutput_original[:, :, 1] == 7,
                                                 testoutput_original[:, :, 2] == 1)] = 7
        #
        segmentation_map[:, :, 0][np.logical_and(testoutput_original[:, :, 0] == 8, testoutput_original[:, :, 1] == 8,
                                                 testoutput_original[:, :, 2] == 2)] = 8
        segmentation_map[:, :, 1][np.logical_and(testoutput_original[:, :, 0] == 8, testoutput_original[:, :, 1] == 8,
                                                 testoutput_original[:, :, 2] == 2)] = 8
        segmentation_map[:, :, 2][np.logical_and(testoutput_original[:, :, 0] == 8, testoutput_original[:, :, 1] == 8,
                                                 testoutput_original[:, :, 2] == 2)] = 8
        #
        segmentation_map[:, :, 0][np.logical_and(testoutput_original[:, :, 0] == 9, testoutput_original[:, :, 1] == 9,
                                                 testoutput_original[:, :, 2] == 3)] = 9
        segmentation_map[:, :, 1][np.logical_and(testoutput_original[:, :, 0] == 9, testoutput_original[:, :, 1] == 9,
                                                 testoutput_original[:, :, 2] == 3)] = 9
        segmentation_map[:, :, 2][np.logical_and(testoutput_original[:, :, 0] == 9, testoutput_original[:, :, 1] == 9,
                                                 testoutput_original[:, :, 2] == 3)] = 9
        #
        segmentation_map[:, :, 0][np.logical_and(testoutput_original[:, :, 0] == 10, testoutput_original[:, :, 1] == 10,
                                                 testoutput_original[:, :, 2] == 4)] = 10
        segmentation_map[:, :, 1][np.logical_and(testoutput_original[:, :, 0] == 10, testoutput_original[:, :, 1] == 10,
                                                 testoutput_original[:, :, 2] == 4)] = 10
        segmentation_map[:, :, 2][np.logical_and(testoutput_original[:, :, 0] == 10, testoutput_original[:, :, 1] == 10,
                                                 testoutput_original[:, :, 2] == 4)] = 10
        #
        segmentation_map[:, :, 0][np.logical_and(testoutput_original[:, :, 0] == 11, testoutput_original[:, :, 1] == 11,
                                                 testoutput_original[:, :, 2] == 5)] = 11
        segmentation_map[:, :, 1][np.logical_and(testoutput_original[:, :, 0] == 11, testoutput_original[:, :, 1] == 11,
                                                 testoutput_original[:, :, 2] == 5)] = 11
        segmentation_map[:, :, 2][np.logical_and(testoutput_original[:, :, 0] == 11, testoutput_original[:, :, 1] == 11,
                                                 testoutput_original[:, :, 2] == 5)] = 11
        #

        segmentation_map_view[:, :, 0][np.logical_and(testoutput_original[:, :, 0] == 0, testoutput_original[:, :, 1] == 0,
                                                 testoutput_original[:, :, 2] == 0)] = 0
        segmentation_map_view[:, :, 1][np.logical_and(testoutput_original[:, :, 0] == 0, testoutput_original[:, :, 1] == 0,
                                                 testoutput_original[:, :, 2] == 0)] = 0
        segmentation_map_view[:, :, 2][np.logical_and(testoutput_original[:, :, 0] == 0, testoutput_original[:, :, 1] == 0,
                                                 testoutput_original[:, :, 2] == 0)] = 0
        #
        segmentation_map_view[:, :, 0][np.logical_and(testoutput_original[:, :, 0] == 1, testoutput_original[:, :, 1] == 1,
                                                 testoutput_original[:, :, 2] == 1)] = 0
        segmentation_map_view[:, :, 1][np.logical_and(testoutput_original[:, :, 0] == 1, testoutput_original[:, :, 1] == 1,
                                                 testoutput_original[:, :, 2] == 1)] = 255
        segmentation_map_view[:, :, 2][np.logical_and(testoutput_original[:, :, 0] == 1, testoutput_original[:, :, 1] == 1,
                                                 testoutput_original[:, :, 2] == 1)] = 0
        #
        segmentation_map_view[:, :, 0][np.logical_and(testoutput_original[:, :, 0] == 2, testoutput_original[:, :, 1] == 2,
                                                 testoutput_original[:, :, 2] == 2)] = 0
        segmentation_map_view[:, :, 1][np.logical_and(testoutput_original[:, :, 0] == 2, testoutput_original[:, :, 1] == 2,
                                                 testoutput_original[:, :, 2] == 2)] = 0
        segmentation_map_view[:, :, 2][np.logical_and(testoutput_original[:, :, 0] == 2, testoutput_original[:, :, 1] == 2,
                                                 testoutput_original[:, :, 2] == 2)] = 255
        #
        segmentation_map_view[:, :, 0][np.logical_and(testoutput_original[:, :, 0] == 3, testoutput_original[:, :, 1] == 3,
                                                 testoutput_original[:, :, 2] == 3)] = 255
        segmentation_map_view[:, :, 1][np.logical_and(testoutput_original[:, :, 0] == 3, testoutput_original[:, :, 1] == 3,
                                                 testoutput_original[:, :, 2] == 3)] = 255
        segmentation_map_view[:, :, 2][np.logical_and(testoutput_original[:, :, 0] == 3, testoutput_original[:, :, 1] == 3,
                                                 testoutput_original[:, :, 2] == 3)] = 0
        #
        segmentation_map_view[:, :, 0][np.logical_and(testoutput_original[:, :, 0] == 4, testoutput_original[:, :, 1] == 4,
                                                 testoutput_original[:, :, 2] == 4)] = 153
        segmentation_map_view[:, :, 1][np.logical_and(testoutput_original[:, :, 0] == 4, testoutput_original[:, :, 1] == 4,
                                                 testoutput_original[:, :, 2] == 4)] = 51
        segmentation_map_view[:, :, 2][np.logical_and(testoutput_original[:, :, 0] == 4, testoutput_original[:, :, 1] == 4,
                                                 testoutput_original[:, :, 2] == 4)] = 255
        #
        segmentation_map_view[:, :, 0][np.logical_and(testoutput_original[:, :, 0] == 5, testoutput_original[:, :, 1] == 5,
                                                 testoutput_original[:, :, 2] == 5)] = 255
        segmentation_map_view[:, :, 1][np.logical_and(testoutput_original[:, :, 0] == 5, testoutput_original[:, :, 1] == 5,
                                                 testoutput_original[:, :, 2] == 5)] = 102
        segmentation_map_view[:, :, 2][np.logical_and(testoutput_original[:, :, 0] == 5, testoutput_original[:, :, 1] == 5,
                                                 testoutput_original[:, :, 2] == 5)] = 178
        #
        segmentation_map_view[:, :, 0][np.logical_and(testoutput_original[:, :, 0] == 6, testoutput_original[:, :, 1] == 6,
                                                 testoutput_original[:, :, 2] == 6)] = 102
        segmentation_map_view[:, :, 1][np.logical_and(testoutput_original[:, :, 0] == 6, testoutput_original[:, :, 1] == 6,
                                                 testoutput_original[:, :, 2] == 6)] = 255
        segmentation_map_view[:, :, 2][np.logical_and(testoutput_original[:, :, 0] == 6, testoutput_original[:, :, 1] == 6,
                                                 testoutput_original[:, :, 2] == 6)] = 102
        #
        segmentation_map_view[:, :, 0][np.logical_and(testoutput_original[:, :, 0] == 7, testoutput_original[:, :, 1] == 7,
                                                 testoutput_original[:, :, 2] == 1)] = 255
        segmentation_map_view[:, :, 1][np.logical_and(testoutput_original[:, :, 0] == 7, testoutput_original[:, :, 1] == 7,
                                                 testoutput_original[:, :, 2] == 1)] = 0
        segmentation_map_view[:, :, 2][np.logical_and(testoutput_original[:, :, 0] == 7, testoutput_original[:, :, 1] == 7,
                                                 testoutput_original[:, :, 2] == 1)] = 0
        #
        segmentation_map_view[:, :, 0][np.logical_and(testoutput_original[:, :, 0] == 8, testoutput_original[:, :, 1] == 8,
                                                 testoutput_original[:, :, 2] == 2)] = 0
        segmentation_map_view[:, :, 1][np.logical_and(testoutput_original[:, :, 0] == 8, testoutput_original[:, :, 1] == 8,
                                                 testoutput_original[:, :, 2] == 2)] = 255
        segmentation_map_view[:, :, 2][np.logical_and(testoutput_original[:, :, 0] == 8, testoutput_original[:, :, 1] == 8,
                                                 testoutput_original[:, :, 2] == 2)] = 255
        #
        segmentation_map_view[:, :, 0][np.logical_and(testoutput_original[:, :, 0] == 9, testoutput_original[:, :, 1] == 9,
                                                 testoutput_original[:, :, 2] == 3)] = 93
        segmentation_map_view[:, :, 1][np.logical_and(testoutput_original[:, :, 0] == 9, testoutput_original[:, :, 1] == 9,
                                                 testoutput_original[:, :, 2] == 3)] = 63
        segmentation_map_view[:, :, 2][np.logical_and(testoutput_original[:, :, 0] == 9, testoutput_original[:, :, 1] == 9,
                                                 testoutput_original[:, :, 2] == 3)] = 106
        #
        segmentation_map_view[:, :, 0][np.logical_and(testoutput_original[:, :, 0] == 10, testoutput_original[:, :, 1] == 10,
                                                 testoutput_original[:, :, 2] == 4)] = 240
        segmentation_map_view[:, :, 1][np.logical_and(testoutput_original[:, :, 0] == 10, testoutput_original[:, :, 1] == 10,
                                                 testoutput_original[:, :, 2] == 4)] = 163
        segmentation_map_view[:, :, 2][np.logical_and(testoutput_original[:, :, 0] == 10, testoutput_original[:, :, 1] == 10,
                                                 testoutput_original[:, :, 2] == 4)] = 255
        #
        segmentation_map_view[:, :, 0][np.logical_and(testoutput_original[:, :, 0] == 11, testoutput_original[:, :, 1] == 11,
                                                 testoutput_original[:, :, 2] == 5)] = 43
        segmentation_map_view[:, :, 1][np.logical_and(testoutput_original[:, :, 0] == 11, testoutput_original[:, :, 1] == 11,
                                                 testoutput_original[:, :, 2] == 5)] = 206
        segmentation_map_view[:, :, 2][np.logical_and(testoutput_original[:, :, 0] == 11, testoutput_original[:, :, 1] == 11,
                                                 testoutput_original[:, :, 2] == 5)] = 72
        #

        basename = Path(file).stem #os.path.splitext(os.path.basename(input))[1]
        print(basename)
        prediction_name = output_folder + basename + '_pred.png'
        predictionView_name = output_folder + basename + '_predView.png'
        imageio.imsave(prediction_name, segmentation_map)
        imageio.imsave(predictionView_name, segmentation_map_view)


args = sys.argv

# check if string one is file or folder
# input = r'C:\Users\sbng352\Desktop\OCT_CNNtraining\train\images\DMO-DMO007_9379_14_x165_A2.tiff'
# output_folder = r'./'
input = args[1]
output_folder = args[2]
if os.path.isfile(input):  # Does input exist?  Is it a file, or a directory?
    predict(input, output_folder)
elif os.path.isdir(input):
    files = os.listdir(input)
    for file in files:
        predict(input+file, output_folder)
else:
    raise ValueError('Not a file or folder')


