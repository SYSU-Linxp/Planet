#!/usr/bin/env python
# coding=utf-8

import re
import csv
import random

lines = list()
with open("../weather_train_labels.txt", 'rb') as f:
    for line in f:
        tmp = re.split(' ', line.strip())
        for i in range(1, len(tmp)):
            if float(tmp[i]) == 1.0:
                if i == 2:
                    for i in range(5):
                        lines.append(line)
                elif i == 3:
                    for i in range(2):
                        lines.append(line)
                elif i == 4:
                    for i in range(6):
                        lines.append(line)
                else:
                    lines.append(line)

random_sample = random.sample(lines, len(lines))
with open("../new_weather_train_labels.txt", 'w') as f:
    for i in range(len(random_sample)):
        f.write(random_sample[i])

