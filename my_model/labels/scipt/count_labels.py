#!/usr/bin/env python
# coding=utf-8

import re

counter = [0] * 17

with open("../new_train_labels.txt", 'rb') as f:
    for line in f:
        tmp = re.split(' ', line.strip())
        for i in range(1, len(tmp)):
            if float(tmp[i]) == 1.0:
                counter[i - 1] += 1

print counter
