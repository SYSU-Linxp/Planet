#!/usr/bin/env python
# encoding: utf-8

import os
from datetime import datetime
import numpy as np
import read_data
import csv
import tensorflow as tf
import model
from sklearn.metrics import fbeta_score

class Config():
    batch_size = 84
    max_step = 1

    img_width = 224
    img_height = 224
    img_channel = 3

    learning_rate = 0.001
    test_num = '4000'

    label_path = './labels/val_labels.txt'
    params_dir = './params/' 
    net_name = 'vgg16'
    vgg_path = params_dir + net_name + "-" + test_num + '.npy'
    data_path = '../data/ALLIMAGE/'

    degree = 10
    record_len = 18
    type_size = 17
    #test_size = 40669 + 20522
    test_size = 3479

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
    modeler = model.VGG(config)
    modeler.inference()
    
    # read data to test("data/train")
    val_reader = read_data.VGGReader(config)

    init = tf.global_variables_initializer()
    
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True

    with tf.Session(config=sess_config) as sess:
        sess.run(init)
        print 'restore params' + config.test_num

        print "finding thresholds..."
        for step in range(config.max_step):
            if step % (config.test_size // 10) == 0:
                print 100 * step // config.test_size, "%"
            with tf.device("/cpu:0"):
                images_val, labels_val, filesname_val = val_reader.batch()

            with tf.device("/gpu:0"):
                _, probs = sess.run([modeler.pred, modeler.prob], feed_dict={modeler.image_holder:images_val, modeler.is_train:False})

            thresholds, scores = find_threshold1(probs, labels_val)
            i = np.argmax(scores)
            best_threshold, best_score = thresholds[i], scores[i]
        
            best_thresholds, best_scores = find_threshold2(probs, labels_val, num_iters=500, seed=best_threshold)

        with open("./thresholds/" + config.net_name + "-thresholds-" + config.test_num + ".txt", 'w') as fr:
            for i in range(len(best_thresholds)):
                fr.write(str(best_thresholds[i]) + " ")
    

if __name__ == '__main__':
    main()


