#dependencies
import numpy as np
import cv2
import os


#image  = cv2.imread("C:/Users/Arghyadeep/Desktop/image colorization/new process/train2017/000000000025.jpg")
#print(image)

#function to read the images and store them as rgb values in an array
def read_images(path):
    images = []
    all_paths = os.listdir(path)
    mini_set = all_paths[:400]
    for i in mini_set:
        file = path+"/"+i
        image = cv2.imread(file)
        image = cv2.resize(image,(320,240))
        images.append(image)

    return images
    
#function to convert rgb to lab 
def rgb_to_lab(images):
    lab_images = []
    for i in images:
        lab_image= cv2.cvtColor(i, cv2.COLOR_BGR2LAB)
        lab_images.append(lab_image)

    return lab_images


#function to extract the l channels and ab for training
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

#function to create train and test data
def create_train_data(l,a,b):
    train_data = []
    for i in l:
        train_data.append(np.array(i.flatten(),dtype= 'float32'))
    train_labels_a = []
    train_labels_b = []
    for i in a:
        train_labels_a.append(np.array(i.flatten(),dtype='float32'))
    for i in b:
        train_labels_b.append(np.array(i.flatten(),dtype='float32'))

    return train_data, train_labels_a, train_labels_b
        
##path = "C:/Users/Arghyadeep/Desktop/image colorization/new process/train2017"
##images = read_images(path)
##lab_images = rgb_to_lab(images)
##l,a,b = extract_channels(lab_images)
##
##
##tr,lba,lbb = create_train_data(l,a,b)
##
##print(lbb[0])





