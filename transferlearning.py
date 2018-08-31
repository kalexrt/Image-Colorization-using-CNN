#dependencies
import numpy as np
import cv2
import os
import tensorflow as tf
import numpy as np
import keras

#keras dependencies
from keras.preprocessing import image
from keras.applications import resnet50, inception_v3, vgg16
from keras.models import Model,Sequential
from keras.layers import Dense, GlobalAveragePooling2D, Input
from keras.optimizers import Adam


#preprocessing
def read_images(path):
    images = []
    all_paths = os.listdir(path)
    mini_set = all_paths[:400]
    for i in mini_set:
        file = path+"/"+i
        image = cv2.imread(file)
        image = cv2.resize(image,(224,224))
        images.append(image)

    return images

def rgb_to_lab(images):
    lab_images = []
    for i in images:
        lab_image= cv2.cvtColor(i, cv2.COLOR_BGR2LAB)
        lab_images.append(lab_image)

    return lab_images

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


#helper
path = "C:/Users/Arghyadeep/Desktop/image colorization/new process/val2017"
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

eval_labels_a = np.array(lbla[train_size:])
eval_labels_b = np.array(lblb[train_size:])
eval_labels = []
for i in range(50):
    temp = np.concatenate((eval_labels_a[i],eval_labels_b[i]),axis=0)
    eval_labels.append(temp)
eval_labels = np.array(eval_labels)
print(eval_labels.shape)



#training
num_outputs = 224*224*2
batch_size = 20

model = Sequential()
model.add(Conv2D(3,kernel_size=(3,3),activation='relu'))
base_model = vgg16.VGG16

base_model = base_model(weights = 'imagenet', include_top = False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024,activation='relu')(x)
preds = Dense(num_outputs,activation='relu')(x)

model.add(Model(inputs=base_model.input, outputs=predictions))
for layer in base_model.layers:
    layer.trainable = False

model.compile(loss='mean_squared_error',
              optimizer=Adam(lr=0.01),
              metrics=['acc'])


model.fit(x_train, y_train,
          epochs=100,
          batch_size=batch_size,
          shuffle=False,
          validation_data=(train_data,train_labels))


#create colorized image
n_images = 1
new_image = train_data[:n_images]
l_channel_predict = new_image.reshape(224,224)
ab_channel_predict = model.predict(new_image)
a_channel_predict = ab_channel_predict[:224*224].reshape(224,224)
b_channel_predict = ab_channel_predict[224*224:].reshape(224,224)

predict_image = cv2.merge((l_channel_predict,a_channel_predict,b_channel_predict))
cv2.imshow('predict',predict_image)


