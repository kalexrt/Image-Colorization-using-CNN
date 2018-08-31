#dependencies
import numpy as np
import cv2
import os

def read_images(path):
    images = []
    all_paths = os.listdir(path)
    mini_set = all_paths[:400]
    for i in mini_set:
        file = path+"/"+i
        image = cv2.imread(file)
        image = cv2.resize(image,(128,128))
        images.append(image)

    return images

x = read_images("C:/Users/Arghyadeep/Desktop/image colorization/new process/val2017")
#cv2.imshow('image',x[1])

def extract_channels(lab_images):
    l_channels = []
    a_channels = []
    b_channels = []
    for i in lab_images:
        l,a,b = cv2.split(i)
        l_channels.append(l)
        a_channels.append(a)
        b_channels.append(b)

    return np.array(l_channels), np.array(a_channels), np.array(b_channels)

l,a,b = cv2.split(x[1])
l = np.array(l)
l = l.reshape(128,128)
l = np.array(l)
print(l)
cv2.imshow('img',l)
