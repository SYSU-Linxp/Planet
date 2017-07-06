#!/usr/bin/env python
# coding=utf-8
import re
import csv

with open("../weather_test_labels.txt", 'w') as f:
    for i in range(40669 + 20522):
        if i < 40669:
            s = "test_" + str(i) + ".jpg"
        else:
            s = "file_" + str(i - 40669) + ".jpg"
        for j in range(4):
            s += " 0.0"
        s += '\n'
        f.write(s)
