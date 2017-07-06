#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import cv2
import sys
import os
import re
import random
import math

class VGGReader():

    def __init__(self, config):
        self.records = list()
        self.batch_size = config.batch_size
        self.data_path = config.data_path

        self.img_width = config.img_width
        self.img_height = config.img_height
        #self.label_path = label_path
        self.img_channel = config.img_channel
        self.degree = config.degree
        #self.record_len = 18
        self.record_len = config.record_len
	self.line_idx = 0
        self.mean = [77.28288, 87.92146, 80.51814] 
        #self.mean = [103.939, 116.779, 123.68]
        if (self.img_channel == 3) :
            self.color_mode = 1
        else:
            self.color_mode = 0
        
        with open(config.label_path, 'rb') as fr:
            for line in fr:
                tmp = re.split(' ', line.strip())
                if(len(tmp) != self.record_len):
                    print "Length Error: ", len(tmp)
                    sys.exit(0)
                filename = tmp[0]
                #begin = int(float(tmp[1]))
                begin = []
                for i in range(1, self.record_len):
                    begin.append(float(tmp[i]))

                #for i in range(1000-17):
                    #begin.append(0.0)

                self.records.append((filename, begin))
        self.size = len(self.records)
                
                #img = cv2.imread(os.path.join(self.data_path, filename), self.color_mode)
                #img = cv2.resize(img, (self.img_width, self.img_height))
                #all_img_list.append(img)
        #imgs = np.stack(all_img_list)
        #self.mean = np.mean(np.mean(np.mean(imgs, axis=0), axis=0), axis=0)
        #print "The mean is: ", self.mean
        #print label_path, "count is :", self.size


    def random_batch(self):
        rand = random.sample(xrange(self.size), self.batch_size)
        filename_list = list()
        begins_list = list()
        for idx in rand:
            filename_list.append(self.records[idx][0])
            begins_list.append(self.records[idx][1])

        img_list = list()
        for filename in filename_list:
            img = cv2.imread(os.path.join(self.data_path, filename),
                    self.color_mode)
            img = self._random_generate_img(img)
            img_list.append(img)

        out_imgs = self._img_preprocess(np.stack(img_list))
        out_begins = np.stack(begins_list)
        return out_imgs, out_begins, filename_list


    def batch(self, line_idx=None):
        if line_idx is not None:
            self.line_idx = line_idx
        end_idx = self.line_idx + self.batch_size
        idxs = range(self.line_idx, end_idx)
        for idx in xrange(len(idxs)):
            if idxs[idx] >= self.size:
                idxs[idx] %= self.size
        if end_idx < self.size:
            self.line_idx = end_idx
        else:
            self.line_idx = end_idx % self.size

        filename_list = list()
        begins_list = list()
        for idx in idxs:
            filename_list.append(self.records[idx][0])
            begins_list.append(self.records[idx][1])

        img_list = list()
        for filename in filename_list:
            img = cv2.imread(os.path.join(self.data_path, filename),
                    self.color_mode)
            img = self._generate_img(img)
            img_list.append(img)

        out_imgs = self._img_preprocess(np.stack(img_list))
        out_begins = np.stack(begins_list)
        #out_begins = np.int32(out_begins)
        return out_imgs, out_begins, filename_list
    
    def _random_generate_img(self, img):
        '''
        scale = random.randint(256, 512)
        img = cv2.resize(img, (scale, scale))
        row_begin = random.randint(0, scale - self.img_height)
        col_begin = random.randint(0, scale - self.img_width)
        return  img[row_begin: (self.img_height + row_begin), col_begin: (col_begin + self.img_width)]
        '''
        return cv2.resize(img, (self.img_height, self.img_width))


    def _generate_img(self, img):
        #img = cv2.resize(img, (384, 384))
        img = cv2.resize(img, (self.img_height, self.img_width))
        return img 

    def _img_preprocess(self, imgs):
        imgs = np.subtract(imgs, self.mean)
        #imgs = np.float64(imgs)
        
        #print imgs
        if self.color_mode == 0:
            output = np.reshape(imgs, [-1, self.img_height, self.img_width, 1])
        elif self.color_mode == 1:
            output = np.reshape(imgs, [-1, self.img_height, self.img_width, 3])
        else:
            raise Exception ("color_mode error.")

        #output = output.astype(np.float32) * (1. / 255) - 0.5
        return output


    def _random_roate(self, images, degree):
        #degree = degree * math.pi / 180
        degree = [-90, 0, 90, 180] * images.shape[0]
        #rand_degree = np.random.uniform(-degree, degree, images.shape[0])
        rand_degree = random.sample(degree, images.shape[0])

        o_images = np.zeros_like(images)
        for idx in xrange(images.shape[0]):
            theta = rand_degree[idx]
            # image
            M = cv2.getRotationMatrix2D((self.img_width/2,self.img_height/2), theta, 1)
            warp_image = cv2.warpAffine(images[idx],M,(self.img_width,self.img_height))
            o_images[idx] = warp_image

        return o_images

    def _batch_random_roate(self, images, degree):
        degree = degree * math.pi / 180
        rand_degree = np.random.uniform(-degree, degree)

        o_images = np.zeros_like(images)
        for idx in xrange(images.shape[0]):
            theta = rand_degree
            # image
            M = cv2.getRotationMatrix2D((self.img_width/2,self.img_height/2),-theta*180/math.pi,1)
            o_images[idx] = np.expand_dims(cv2.warpAffine(images[idx],M,(self.img_width,self.img_height)), axis=2)

        return o_images

    def _random_flip_lr(self, images):
        rand_u = np.random.uniform(0.0, 1.0, images.shape[0])
        rand_cond = rand_u > 0.5

        o_images = np.zeros_like(images)
        for idx in xrange(images.shape[0]):
            condition = rand_cond[idx]
            if condition:
                # "flip"
                o_images[idx] = np.fliplr(images[idx])
            else:
                # "origin"
                o_images[idx] = images[idx]

        return o_images

    def _flip_lr(self, images):
        # if(images.shape[0] != labels.shape[0]):
        #     raise Exception("Batch size Error.")

        o_images = np.zeros_like(images)

        for idx in xrange(images.shape[0]):
            # "flip"
            o_images[idx] = np.fliplr(images[idx])
        return o_images

    def get_random_batch(self, distort=True):
        imgs, begins, filename_list = self.random_batch()
        if distort:
            #imgs = self._flip_lr(imgs)
            imgs = self._random_roate(imgs, self.degree)

        return (imgs.reshape([self.batch_size, self.img_height, self.img_width, self.img_channel]), begins, filename_list)

    def get_batch(self, distort=False, line_idx=None):

        imgs, begins, filename_list = self.batch(
                line_idx=line_idx)
        if distort:
            imgs = self._flip_lr(imgs)
            #imgs = self._batch_random_roate(imgs, self.degree)

        return (imgs.reshape([self.batch_size, self.img_height, self.img_width, self.img_channel]), begins, filename_list)



def main():
    pass


if __name__ == "__main__":
    main()

