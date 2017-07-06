#!/usr/bin/env python
# coding=utf-8

import csv
import re
import random
import numpy as np

label_list = list()
train_size = 37000

with open("../txt/all_labels.txt", 'rb') as fr:
    for line in fr:
        tmp = re.split(' ', line.strip())
        for i in range(len(tmp)):
            label_list.append(tmp[i])

print label_list

label_in_pro_list_for_train = list()
label_in_pro_list_for_val = list()

cnt = 0
with open("../csv/train_labels_meaning.csv", 'rb') as fs:
    for line in fs:
        label_in_pro = list()
        for i in range(len(label_list)):
            label_in_pro.append(np.float32(0.0))
        tmp = re.split(' ', line.strip())
        for i in range(len(tmp)):
            for j in range(len(label_list)):
                if label_list[j] == tmp[i]:
                    label_in_pro[j] = np.float32(1.0)
                    break
        if cnt < train_size:
            label_in_pro_list_for_train.append(label_in_pro)
        else:
            label_in_pro_list_for_val.append(label_in_pro)
        cnt = cnt + 1

with open("./train_labels.csv", 'w') as f:
    f_csv = csv.writer(f)
    for i in range(len(label_in_pro_list_for_train)):
        s = "train_" + str(i) + ".jpg"
        for j in range(len(label_list)):
            s += " " + str(label_in_pro_list_for_train[i][j])
        row = [s]
        f_csv.writerow(row)

with open("./val_labels.csv", 'w') as f:
    f_csv = csv.writer(f)
    for i in range(len(label_in_pro_list_for_val)):
        s = "train_" + str(i+train_size) + ".jpg"
        for j in range(len(label_list)):
            s += " " +  str(label_in_pro_list_for_val[i][j])
        row = [s]
        f_csv.writerow(row)






    
