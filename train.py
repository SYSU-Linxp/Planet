#!/usr/bin/env python
# encoding: utf-8

import os
from datetime import datetime
import numpy as np
import read_data
import tensorflow as tf
#from keras.applications.imagenet_utils import preprocess_input
#import model
import keras.applications.resnet50 as model
#import keras.applications.vgg16 as model
from keras.layers import Flatten
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.layers import Input
from keras.models import Model
#from keras.applications.vgg16 import preprocess_input
#from keras.applications.resnet50 import preprocess_input

class Config():
    batch_size = 32
    max_step = 2000
    save_iter = 500
    restore_steps = 0

    img_width = 224
    img_height = 224
    img_channel = 3

    learning_rate = 0.0001

    label_path = './labels/train_labels.txt'
    params_dir = './params/' 
    data_path = './data/ALLIMAGE/'
    #net_name = 'vgg16'
    net_name = 'resnet50'

    degree = 10
    record_len = 18
    type_size = 17
    #test_size = 40669 + 20522

def main():
    config = Config()
    #modeler = model.VGG(config)
    #base_modeler = model.VGG16(include_top=False, weights='imagenet')
    base_modeler = model.ResNet50(include_top=False, weights='imagenet', input_shape=[config.img_height,config.img_width, 3], pooling='max')
    inputs = Input(shape=[config.img_height,config.img_width,3])
    y = base_modeler(inputs)
    
    # fine tune the model
    y = Dense(2048, activation='relu')(y)
    y = Dropout(0.5)(y)
    y = Dense(2048, activation='relu')(y)
    y = Dropout(0.5)(y)
    #y = Dense(512, activation='relu')(y)
    y = Dense(config.type_size, activation='sigmoid', name='predictions')(y)
    modeler = Model(inputs, y, name='resnet50')

    '''
    y = Flatten()(y)
    y = Dense(4096, activation='relu', name='fc1')(y)
    y = Dense(4096, activation='relu', name='fc2')(y)
    y = Dense(config.type_size, activation='softmax', name='predictions')(y)
    modeler = Model(inputs, y, name='vgg16')
    '''
    for layer in base_modeler.layers:
        layer.trainable = False

    #print "restore params" + str(config.restore_steps)
    #modeler.load_weights(config.params_dir + config.net_name + "-params-" + str(config.restore_steps) + ".h5")

    modeler.compile(loss='binary_crossentropy', optimizer=SGD(lr=config.learning_rate, momentum=0.9,nesterov=True))

    # read data to train("data/train")
    train_reader = read_data.VGGReader(config)

    init = tf.global_variables_initializer()
    
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True

    with tf.Session(config=sess_config) as sess:
        sess.run(init)
        print 'start training'
        for step in range(config.max_step):

            with tf.device('/cpu:0'):
                images_train, labels_train, filesname_train = train_reader.get_random_batch(True)

            #images_train = preprocess_input(images_train)
            if step == 0:
                print labels_train
            modeler.train_on_batch(images_train, labels_train)
            loss_value = modeler.test_on_batch(images_train, labels_train)

            if step%10 == 0:
                print 'step %d, loss = %.3f' % (step, loss_value)
                #print prediction
            if (step+1) % config.save_iter == 0:
                # save weights
                modeler.save_weights(config.params_dir + config.net_name + "-params-" + str(step+1+config.restore_steps) + ".h5")

                pre = modeler.predict(images_train)
                for i in range(config.batch_size):
                    print labels_train[i]
                    print pre[i]
    

if __name__ == '__main__':
    main()


