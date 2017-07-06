#!/usr/bin/env python
# coding=utf-8
import re
import csv

with open("../test_labels.csv", 'w') as f:
    for i in range(40669 + 20522):
        if i < 40669:
            s = "test_" + str(i) + ",0"
        else:
            s = "file_" + str(i - 40669) + ",0"
        #for j in range(4):
            #s += " 0.0"
        s += '\n'
        f.write(s)
