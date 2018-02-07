#Import the necessary packages
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

def load_databatch(data_path, idx):
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
	pic_l = pic_lab[:,:,0] #L
	#pic_l = pic_l - np.mean(pic_l) #Remove the mean
	
	cv2.namedWindow('res', cv2.WINDOW_NORMAL)
	cv2.imshow('res',pic)
	cv2.waitKey()
	return pic_l, pic_lab
	
	

#MAIN EXECUTION

img_size = 32
num_batches = 10
num_dims = 3 #RGB
data_path = '.\\'+str(img_size)+'x'+str(img_size)

for i in range(num_batches):

	X_train_rgb, Y_train_rgb, mean_image = load_databatch(data_path, i+1)
	#X_train_lab = np.zeros((num_batches, X_train_rgb.shape[0], X_train_rgb.shape[1], X_train_rgb.shape[2], num_dims))
	#X_train_grey = np.zeros((num_batches, X_train_rgb.shape[0], X_train_rgb.shape[1], X_train_rgb.shape[2]))
	print(X_train_rgb.shape)
	for j in range(X_train_rgb.shape[0]):
		#X_train_grey[i,j], X_train_lab[i,j] = preprocess_image(X_train_rgb[j])
		X_train_grey, X_train_lab = preprocess_image(X_train_rgb[j])
	
cv2.destroyAllWindows()
