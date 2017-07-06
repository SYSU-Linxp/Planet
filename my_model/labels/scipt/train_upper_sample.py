#!/usr/bin/env python
# coding=utf-8

import re
import csv
import random

lines = list()
with open("../train_labels.txt", 'rb') as f:
    for line in f:
        tmp = re.split(' ', line.strip())
        for i in range(1, len(tmp)):
            if float(tmp[i]) == 1.0:
                if i == 8:
                    for i in range(2):
                        lines.append(line)
                elif i == 9:
                    for i in range(3):
                        lines.append(line)
                elif i == 11:
                    for i in range(15):
                        lines.append(line)
                elif i == 12:
                    for i in range(17):
                        lines.append(line)
                elif i == 13:
                    for i in range(8):
                        lines.append(line)
                elif i == 14:
                    for i in range(10):
                        lines.append(line)
                elif i == 15:
                    for i in range(10):
                        lines.append(line)
                elif i == 16:
                    for i in range(10):
                        lines.append(line)
                elif i == 17:
                    for i in range(17):
                        lines.append(line)
                else:
                    lines.append(line)

random_sample = random.sample(lines, len(lines))
with open("../new_train_labels.txt", 'w') as f:
    for i in range(len(random_sample)):
        f.write(random_sample[i])

