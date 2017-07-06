#!/usr/bin/env python
# encoding: utf-8
import csv
import tensorflow as tf
import model
import read_data


class Config():
    batch_size = 1
    max_step = 6000

    img_width = 224
    img_height = 224
    img_channel = 3

    steps = '1000'
    param_dir = './params/'
    load_filename = 'vgg16-' + steps
    vgg_path = param_dir + load_filename +  '.npy'
    checkpointer_iter = 200

    label_path='./labels/test_labels.txt'
    data_path = '../data/ALLIMAGE/'

    log_dir = './log/'
    summary_iter = 200

    degree = 10
    #test_size = 37000
    test_size = 40669 + 20522
    #test_size = 40479
    #test_size = 3479
    type_size = 17


def main():
    config = Config()

    modeler = model.VGG(config)

    #read data to val("data/val")
    val_reader = read_data.VGGReader(config)

    modeler.inference()
    accuracy = modeler.accuracy()

    init = tf.global_variables_initializer()
    #saver = tf.train.Saver(max_to_keep=200)

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True

    label_indexs = list()
    test_labels = list()

    with tf.Session(config=sess_config) as sess:
        sess.run(init)
        #saver.restore(sess, Config.vgg_path)
        print 'restore params' + config.steps

        #testing
        count = 0
        num_iter = config.test_size // config.batch_size
        max_false_probs = [0.0] * config.type_size
        max_true_probs = [0.0] * config.type_size
        pre_prob = [0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.6, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13]
        for i in range(num_iter):  
            label_index = list()
            with tf.device('/cpu:0'):
                images_val, labels_val, filenames_val = val_reader.batch()

            with tf.device("/gpu:0"):
                
                predict, prob = sess.run([modeler.pred, modeler.prob], feed_dict={modeler.image_holder:images_val, modeler.is_train:False})
                if i < 20:
                    print predict, labels_val
                    print prob
                if i % (num_iter // 10) == 0:
                    print 100 * i / num_iter, "%" 
                for j in range(config.type_size):
                    if (labels_val[0][j] == 1.0 and prob[0][j] > max_true_probs[j]):
                        max_true_probs[j] = prob[0][j]

                    if (labels_val[0][j] == 0.0 and prob[0][j] > max_false_probs[j]):
                        max_false_probs[j] = prob[0][j]

                    if prob[0][j] > pre_prob[j]:
                        label_index.append(j)

                label_indexs.append(label_index)
                
                is_correct = True
                for k in range(len(label_index)):
                    if labels_val[0][label_index[k]] == 0.0:
                        is_correct = False
                        break

                count_in = 0
                count_in1 = 0
                for k in range(len(labels_val[0])):
                    if labels_val[0][k] > 0.0:
                        count_in1  += 1
                        for n in range(len(label_index)):
                            if label_index[n] == k:
                                count_in += 1
                                break
                if count_in != count_in1:
                    is_correct = False
                
                s = filenames_val[0]
                for n in range(17):
                    is_in = False
                    for m in range(len(label_index)):
                        if n == label_index[m]:
                            is_in = True
                    if is_in:
                        s += " 1.0"
                    else:
                        s += " 0.0"
                test_labels.append(s)

                if is_correct :
                    count += 1
        with open("./labels/test_labels_pro.csv", 'w') as fr:
            fcsv = csv.writer(fr)
            for i in range(len(test_labels)):
                fcsv.writerow([test_labels[i]])
        '''
        print "max prob of true sample: ", max_true_probs
        print '---------------------------------------'
        print "max prob of false sample: ", max_false_probs
        print 'AP: ',  count * 1.0 / config.test_size
        '''


if __name__ == '__main__':
    main()


