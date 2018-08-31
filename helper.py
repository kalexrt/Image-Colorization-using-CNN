from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from preprocessing import read_images, rgb_to_lab, extract_channels, create_train_data


path = "C:/Users/Arghyadeep/Desktop/image colorization/new process/train2017"
images = read_images(path)
lab_images = rgb_to_lab(images)
l,a,b = extract_channels(lab_images)
tr, lbla, lblb = create_train_data(l,a,b)
train_size = 350
train_data = np.array(tr[:train_size])
eval_data = np.array(tr[train_size:])
train_labels_a = np.array(lbla[:train_size])
train_labels_b = np.array(lblb[:train_size])
train_labels = []
for i in range(train_size):
    temp = np.concatenate((train_labels_a[i],train_labels_b[i]),axis=0)
    train_labels.append(temp)
train_labels = np.array(train_labels,dtype='float32')
print(train_labels.shape)
    
