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

class Config():
    batch_size = 3479
    max_step = 1 

    img_width = 224
    img_height = 224
    img_channel = 3

    learning_rate = 0.001
    test_num = '40000'

    label_path = './labels/weather_val_labels.txt'
    params_dir = './params/' 
    data_path = './data/ALLIMAGE/'
    #net_name = 'resnet50'
    net_name = 'vgg16'

    degree = 10
    record_len = 5
    type_size = 4
    test_size = 40669 + 20522
    #test_size = 1000

def find_threshold1(probs, labels, thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0,1,0.005)
    N=len(thresholds)
    scores = np.zeros(N,np.float32)
    for n in range(N):
        t = thresholds[n]
        score = fbeta_score(labels, probs>t, beta=2, average='samples')
        scores[n] = score

    return thresholds, scores

def find_threshold2(probs, labels, num_iters=200, seed=0.235):
    batch_size, num_classes = labels.shape[0:2]
    
    best_thresholds = [seed]*num_classes
    best_scores = [0]*num_classes
    for t in range(num_classes):

        thresholds = [seed]*num_classes
        for i in range(num_iters):
            th = i / float(num_iters)
            thresholds[t] = th
            f2 = fbeta_score(labels, probs > thresholds, beta=2, average='samples')
            if f2 > best_scores[t]:
                best_scores[t] = f2
                best_thresholds[t] = th
        print "t, best_thresholds[t], best_scores[t]= ", t, best_thresholds[t], best_scores[t]

    return best_thresholds, best_scores


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

    modeler.load_weights(config.params_dir + config.net_name + "-params-" + str(config.test_num) + ".h5")

    # read data to test("data/train")
    val_reader = read_data.VGGReader(config)

    print "finding thresholds..."
    for step in range(config.max_step):
        if step % (config.test_size // 10) == 0:
            print 100 * step // config.test_size, "%"

        images_val, labels_val, filesname_val = val_reader.get_batch()
        probs = modeler.predict(images_val)

        thresholds, scores = find_threshold1(probs, labels_val)
        i = np.argmax(scores)
        best_threshold, best_score = thresholds[i], scores[i]
        
        best_thresholds, best_scores = find_threshold2(probs, labels_val, num_iters=500, seed=best_threshold)

    with open("./thresholds/" + config.net_name + "-weather-thresholds-" + config.test_num + ".txt", 'w') as fr:
        for i in range(len(best_thresholds)):
            fr.write(str(best_thresholds[i]) + " ")
    

if __name__ == '__main__':
    main()


