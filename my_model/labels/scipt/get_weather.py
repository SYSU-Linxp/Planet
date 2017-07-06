#!/usr/bin/env python
# coding=utf-8

import csv
import re
import random
import numpy as np

train_list = list()
val_list = list()
train_size = 37000

with open("../train_labels.txt", 'rb') as fr:
    for line in fr:
        tmp = re.split(' ', line.strip())
        s = tmp[0]
        for i in range(1, 5):
            s += " " + str(tmp[i])
        train_list.append(s)

with open("./train_labels.csv", 'w') as f:
    f_csv = csv.writer(f)
    for i in range(len(train_list)):
        row = [train_list[i]]
        f_csv.writerow(row)

with open("../val_labels.txt", 'rb') as fr:
    for line in fr:
        tmp = re.split(' ', line.strip())
        s = tmp[0]
        for i in range(1, 5):
            s += " " + str(tmp[i])
        val_list.append(s)

with open("./val_labels.csv", 'w') as f:
    f_csv = csv.writer(f)
    for i in range(len(val_list)):
        row = [val_list[i]]
        f_csv.writerow(row)






    
