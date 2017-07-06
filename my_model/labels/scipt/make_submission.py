#!/usr/bin/env python
# coding=utf-8

import re

label_lists = list()
all_labels = list()

with open("../txt/all_labels.txt", 'rb') as f:
    for line in f:
        tmp = re.split(' ', line.strip())
        for i in range(len(tmp)):
            all_labels.append(tmp[i])

print all_labels

with open("../test_labels_pro.csv", 'rb') as f:
    for line in f:
        label_list = list()
        tmp = re.split(' ', line.strip())
        is_none = False
        for i in range(1, len(tmp)):
            if float(tmp[i]) == 1.0:
                is_none = True
                label_list.append(all_labels[i - 1])
        if is_none == False:
            label_list.append(all_labels[7])
        label_lists.append(label_list)

with open("../submission.csv", 'w') as f:
    f.write("image_name,tags\n")
    for i in range (len(label_lists)):
        if i < 40669:
            s = "test_" + str(i) + ","
        else:
            s = "file_" + str(i - 40669) + ","
        for j in range(len(label_lists[i])):
            s += label_lists[i][j]
            if j != len(label_lists[i]) - 1:
                s += " "
            else:
                s += '\n'
        f.write(s)





