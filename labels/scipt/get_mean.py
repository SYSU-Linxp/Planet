#!/usr/bin/env python
# coding=utf-8

import re
import os
import sys
import cv2
import numpy as np

filenames = list()
filename = ""
img_list = list()
with open("./train_labels.txt", 'rb') as f:
    for line in f:
        tmp = re.split(' ', line.strip())
        filename = tmp[0]
        filenames.append(filename)
#print filenames
for file in filenames:
    img = cv2.imread(os.path.join("../data/ALLIMAGE/", file), 1)
    img_list.append(img)

img_list = np.reshape(np.stack(img_list), [-1, 256, 256, 3])

mean = np.mean(np.mean(np.mean(img_list, axis=0), axis=0), axis=0)

print mean
