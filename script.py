import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import cv2
import csv
import keras as k
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import Input
from sklearn.metrics import fbeta_score
#import keras.applications.vgg16 as VggModel
import keras.applications.resnet50 as ResModel
from tqdm import tqdm


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

x_train = []
x_test = []
y_train = []
test_name = []

df_train = pd.read_csv('./labels/train_v2.csv')
df_test = pd.read_csv('./labels/test_labels.csv')

flatten = lambda l: [item for sublist in l for item in sublist]
labels = list(set(flatten([l.split(' ') for l in df_train['tags'].values])))

label_map = {l: i for i, l in enumerate(labels)}
inv_label_map = {i: l for l, i in label_map.items()}

print labels

for f, tags in tqdm(df_train.values, miniters=1000):
    img = cv2.imread('./data/train_jpg/{}.jpg'.format(f))
    targets = np.zeros(17)
    for t in tags.split(' '):
        targets[label_map[t]] = 1 
    
    x_train.append(cv2.resize(img, (224, 224)))
    y_train.append(targets)

for f, tags in tqdm(df_test.values, miniters=1000):
    test_name.append('{}.jpg'.format(f))
    test_img = cv2.imread('./data/test_jpg/{}.jpg'.format(f))
    
    x_test.append(cv2.resize(test_img,(224, 224)))

print "test_name len: ", len(test_name)
print test_name[0:2]
    
y_train = np.array(y_train, np.uint8)
x_train = np.array(x_train, np.float16) / 255.
x_test = np.array(x_test, np.float16) / 255.

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)

split = 37000
x_train, x_valid, y_train, y_valid = x_train[:split], x_train[split:], y_train[:split], y_train[split:]

# model
inputs = Input(shape=[224,224,3])

base_model = ResModel.ResNet50(include_top=False, weights='imagenet')
y = base_model(inputs)
y = Flatten()(y)
y = Dense(1024, activation='relu')(y)
y = Dropout(0.5)(y)
#y = Dense(256, activation='relu')(y)
y = Dense(17, activation='sigmoid', name='predictions')(y)
model = Model(inputs, y)

'''
base_model = VggModel.VGG16(include_top=False, weights='imagenet')
y = base_model(inputs)
y = Flatten()(y)
y = Dense(4096, activation='relu')(y)
y = Dropout(0.5)(y)
y = Dense(4096, activation='relu')(y)
y = Dropout(0.5)(y)
y = Dense(4096, activation='relu')(y)
y = Dropout(0.5)(y)
y = Dense(17, activation='sigmoid', name='predictions')(y)
model = Model(inputs, y)
'''
for layer in base_model.layers:
    layer.trainable = False


'''
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(32, 32, 3)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(17, activation='sigmoid'))
'''
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print "start training..."
              
model.fit(x_train, y_train,
          batch_size=32,
          epochs=15,
          verbose=1,
          validation_data=(x_valid, y_valid))

p_val = model.predict(x_valid, batch_size=32)
print "p_val: ", p_val[0:2]
print "y_valid: ", y_valid[0:2]

thresholds, scores = find_threshold1(p_val, y_valid)
i = np.argmax(scores)
best_threshold, best_score = thresholds[i], scores[i]

best_thresholds, best_scores = find_threshold2(p_val, y_valid, num_iters=500, seed=best_threshold)


p_valid = model.predict(x_valid, batch_size=32)
print(y_valid)
print(p_valid)
print(fbeta_score(y_valid, np.array(p_valid) > best_thresholds, beta=2, average='samples'))

p_test = model.predict(x_test, batch_size=32)
print "p_test.shape: ", p_test.shape

# write result
pre_labels = list()
for i in range(len(test_name)):
    s = test_name[i]
    for j in range(17):
        if p_test[i][j] > best_thresholds[j]:
            s += " 1.0"
        else:
            s += " 0.0"
    pre_labels.append(s)

with open("./labels/test_result.csv", 'w') as f:
    fcsv = csv.writer(f)
    for i in range(len(pre_labels)):
        fcsv.writerow([pre_labels[i]])


