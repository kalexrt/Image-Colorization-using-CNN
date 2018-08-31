from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from preprocessing import read_images, rgb_to_lab, extract_channels, create_train_data

tf.logging.set_verbosity(tf.logging.INFO)



def cnn_model(features, labels, mode):
    #input layer
    input_layer = tf.reshape(features["x"],[-1,240,320,1])

    #convolution and pool layer 1
    conv1 = tf.layers.conv2d(inputs=input_layer,
                             filters=32,
                             kernel_size=[5, 5],
                             padding="same",
                             activation=tf.nn.relu)

    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)


    #convolution and pool layer 2
    conv2 = tf.layers.conv2d(inputs=pool1,
                             filters=64,
                             kernel_size=[5, 5],
                             padding="same",
                             activation=tf.nn.relu)

    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    #convolution and pool layer 3
    conv3 = tf.layers.conv2d(inputs=pool2,
                             filters=128,
                             kernel_size=[5, 5],
                             padding="same",
                             activation=tf.nn.relu)
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)


    #convolution and pool layer 4
    conv4 = tf.layers.conv2d(inputs=input_layer,
                             filters=256,
                             kernel_size=[5, 5],
                             padding="same",
                             activation=tf.nn.relu)
    pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

    flat = tf.contrib.layers.flatten(inputs = pool4)

    fc1 = tf.layers.dense(inputs = flat,units = 1024)

    dropped = tf.layers.dropout(fc1, rate=0.2,
                                training = mode == tf.estimator.ModeKeys.TRAIN)

    fc2 = tf.layers.dense(inputs = dropped, units = 2*320*240)

    

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode = mode, predictions = fc2)

    loss = tf.losses.mean_squared_error(labels = labels, predictions = fc2)


    logging_hook = tf.train.LoggingTensorHook({"loss" : loss}, every_n_iter=1)

    
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
        train_op = optimizer.minimize(loss=loss,
                                      global_step = tf.train.get_global_step()
                                      )
        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          train_op = train_op,
                                          training_hooks = [logging_hook])
    
    eval_metric_ops = {"accuracy":tf.metrics.accuracy(labels=labels,
                                                      predictions = fc2)
        }

    
    return tf.estimator.EstimatorSpec(mode=mode,
                                      loss = loss,
                                      eval_metric_ops = eval_metric_ops)

def main(unused_argv):
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
    train_labels = np.array(train_labels)
    #print(train_labels.shape)
        
    eval_labels_a = np.array(lbla[train_size:])
    eval_labels_b = np.array(lblb[train_size:])
    eval_labels = []
    for i in range(50):
        temp = np.concatenate((eval_labels_a[i],eval_labels_b[i]),axis=0)
        eval_labels.append(temp)
    eval_labels = np.array(eval_labels)

    model = tf.estimator.Estimator(model_fn = cnn_model,
                                   model_dir = "/tmp/colorization")


    train_input_fn = tf.estimator.inputs.numpy_input_fn(x = {"x":train_data},
                                                        y = train_labels,
                                                        batch_size = 50,
                                                        num_epochs = None,
                                                        shuffle = True)
    
    
    model.train(input_fn = train_input_fn,
                steps = 20)

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x":eval_data},
                                                       y = eval_labels,
                                                       num_epochs=1,
                                                       shuffle=False)

    eval_results = model.evaluate(input_fn = eval_input_fn)
    print(eval_results)

if __name__ == "__main__":
    tf.app.run()

    
    
        
