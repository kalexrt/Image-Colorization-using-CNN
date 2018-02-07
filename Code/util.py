#Import the necessary packages
#Contains preprocessing and post processing for LAB and RGB space
#Written by Alex Aranburu and Steven Reeves for CMPS 242 final project
import numpy as np
import argparse
import glob
import cv2
import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
#from pyimagesearch import imutils
#from skimage import exposure

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict

def load_databatch(data_path, idx, img_size):
	data_file = os.path.join(data_path, 'train_data_batch_')
	
	d = unpickle(data_file + str(idx))
	x = d['data']
	y = d['labels']
	mean_image = d['mean']

	x = x/np.float32(255)	
	mean_image = mean_image/np.float32(255)

    # Labels are indexed from 1, shift it so that indexes start at 0
	y = [i-1 for i in y]
	data_size = x.shape[0]

	#x -= mean_image

	img_size2 = img_size * img_size

	x = np.dstack((x[:, :img_size2], x[:, img_size2:2*img_size2], x[:, 2*img_size2:]))
	x = x.reshape((x.shape[0], img_size, img_size, 3)).transpose(0, 3, 1, 2)

    # create mirrored images
	X_train_rgb = x[0:data_size, :, :, :]
	Y_train = y[0:data_size]
	X_train_rgb_flip = X_train_rgb[:, :, :, ::-1]
	Y_train_flip = Y_train
	X_train_rgb = np.concatenate((X_train_rgb, X_train_rgb_flip), axis=0)
	Y_train = np.concatenate((Y_train, Y_train_flip), axis=0)
	return np.swapaxes(np.swapaxes(X_train_rgb,1,2),2,3), Y_train, mean_image
	'''return dict(
        X_train_rgb=X_train_rgb,#=lasagne.utils.floatX(X_train_rgb),
        Y_train=Y_train,#=Y_train.astype('int32'),
        mean=mean_image)'''
    
def preprocess_image(pic): #Input picture must be in RGB
	#pic_grey = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY) #grayscale
	pic_lab = cv2.cvtColor(pic, cv2.COLOR_BGR2LAB) #LAB
	#pic_l = pic_l - np.mean(pic_l) #Remove the mean
	return pic_lab
	
def postprocess_image(pic): #Input must be LUV
	pic_rgb = cv2.cvtColor(pic, cv2.COLOR_LAB2BGR)	
	return pic_rgb

