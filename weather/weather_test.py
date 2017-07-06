#!/usr/bin/env python
# encoding: utf-8

import os
from datetime import datetime
import numpy as np
import read_data
import csv
#import model
#import keras.applications.resnet50 as model
import keras.applications.vgg16 as model
from keras.layers import Flatten
from keras.layers import Dense
#from keras.optimizers import SGD
from keras.layers import Input
from keras.models import Model
from sklearn.metrics import fbeta_score
import re

class Config():
    batch_size = 1
    max_step = 400

    img_width = 224
    img_height = 224
    img_channel = 3

    learning_rate = 0.005

    steps = '-1'
    test_num = '35000'
    val_mode = True

    label_path = './labels/weather_val_labels.txt'
    #label_path = './labels/weather_test_labels.txt'
    params_dir = './params/' 
    data_path = './data/ALLIMAGE/'
    #net_name = 'resnet50'
    net_name = 'vgg16'

    degree = 10
    type_size = 4
    record_len = 5
    #test_size = 40669 + 20522
    test_size = 3479

def main():
    config = Config()
    #modeler = model.VGG(config)
    modeler = model.VGG16(include_top=False)
    #modeler = model.ResNet50(include_top=False, input_shape=[config.img_height,config.img_width, 3], pooling='max')
    inputs = Input(shape=[config.img_height,config.img_width,3])
    y = modeler(inputs)

    y = Flatten()(y)
    y = Dense(4096, activation='relu', name='fc1')(y)
    y = Dense(4096, activation='relu', name='fc2')(y)
    y = Dense(config.type_size, activation='softmax', name='predictions')(y)
    modeler = Model(inputs, y, name='vgg16')

    '''
    y = Dense(config.type_size, activation='softmax', name='fc17')(y)
    modeler = Model(inputs, y, name='resnet50')
    '''
    
    modeler.load_weights(config.params_dir + config.net_name + "-weather-params-" + str(config.test_num) + ".h5")

    #modeler.compile(loss='categorical_crossentropy', optimizer=SGD(lr=config.learning_rate, momentum=0.9,nesterov=True))

    # read data to test("data/train")
    test_reader = read_data.VGGReader(config)

    pre_prob = list()
    with open("./thresholds/" + config.net_name + "-weather-thresholds-" + config.test_num + ".txt", 'rb')  as fr:
        for line in fr:
            tmp = re.split(' ', line.strip())
            for i in range(config.type_size):
                pre_prob.append(float(tmp[i]))
    print "thresholds: ", pre_prob

    test_labels = list()
    pre_labels = list()
    val_labels = list()
    min_true = [1.0, 1.0, 1.0, 1.0]
    max_false = [0.0, 0.0, 0.0, 0.0]
    print "start testing..."
    for step in range(config.test_size):
        if step % (config.test_size // 10) == 0:
            print 100 * step // config.test_size, "%"

        images_test, labels_test, filesname_test = test_reader.batch()
        prob = modeler.predict(images_test)
        test_index = list()
        for i in range(config.type_size):
            val_labels.append(labels_test[0][i])

            if prob[0][i] > pre_prob[i]:
                test_index.append(i)

            # get min_true
            if labels_test[0][i] == 1.0 and prob[0][i] > 0.1 and prob[0][i] < min_true[i]:
                min_true[i] = prob[0][i]
            if labels_test[0][i] == 0.0 and prob[0][i] > max_false[i]:
                max_false[i] = prob[0][i]

        '''
        if step % 10 == 0 and config.val_mode == True:
            print labels_test[0]
            print prob[0]
        '''
    
        s = filesname_test[0]
        for n in range(config.type_size):
            is_in = False
            for m in range(len(test_index)):
                if n == test_index[m]:
                    is_in = True
            if is_in:
                s += " 1.0"
                pre_labels.append(1.0)
            else:
                s += " 0.0"
                pre_labels.append(0.0)

        test_labels.append(s)

    print "scores: ", fbeta_score(val_labels, pre_labels, beta=2)
    print "min_true: ", min_true
    print "max_false, ", max_false
   
    if config.val_mode == False:
        with open("./labels/weather_test_results.csv", 'w') as fr:
            fcsv = csv.writer(fr)
            for i in range(len(test_labels)):
                fcsv.writerow([test_labels[i]])
    

if __name__ == '__main__':
    main()


