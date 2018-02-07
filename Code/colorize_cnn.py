## Convolutional Neural Network for Image Colorization python script using Tensorflow for final project in CMPS 242
## University of California, Santa Cruz 
## Authors: Alex Aranburu, Arghyadeep Giri, Rene Gutierrez, Steven Reeves
import tensorflow as tf
import util

# Create the neural network
def conv_net(x_dict, height, width, reuse):
    # Define a scope for reusing the variables
	with tf.variable_scope('ConvNet', reuse=reuse):
		# Tensor input become 4-D: [Batch Size, Height, Width, Channel]
		x = x_dict['images']
		x = tf.reshape(x, shape=[-1, height, width, 1])
        # Convolution Layer 1 with 16 filters and a kernel size of 4 repeated twice
		conv1_1 = tf.layers.conv2d(x, 16, 5, activation=tf.nn.relu)
		conv1_p = tf.layers.max_pooling2d(conv1_1, 2, 2)
		conv1_2 = tf.layers.conv2d(conv1_p, 16, 5, activation=tf.nn.relu)
		conv1_3 = tf.layers.batch_normalization(conv1_2, axis=-1, momentum=0.99, epsilon = 0.001,
		center=True, scale=True, training=True)
		conv1_4 = tf.layers.max_pooling2d(conv1_3, 2, 2)
        # Convolution Layer 2 with 32 filters and a kernel size of 4
		conv2_1 = tf.layers.conv2d(conv1_3, 32, 3, activation=tf.nn.relu)
		conv2_p = tf.layers.max_pooling2d(conv2_1, 2, 2)
		conv2_2 = tf.layers.conv2d(conv2_p, 32, 3, activation=tf.nn.relu)
		conv2_3 = tf.layers.batch_normalization(conv2_2, axis=-1, momentum=0.99, epsilon = 0.001,
		center=True, scale=True, training=True)
		conv2_4 = tf.layers.max_pooling2d(conv2_3, 2, 2)
        # Convolution Layer 3 with 64 filters and a kernel size of 2
		conv3_1 = tf.layers.conv2d(conv2_3, 64, 2, activation=tf.nn.relu)
		conv3_p = tf.layers.max_pooling2d(conv3_1, 2, 2)
		conv3_2 = tf.layers.conv2d(conv3_1, 64, 2, activation=tf.nn.relu)                
		conv3_3 = tf.layers.batch_normalization(conv3_2, axis=-1, momentum=0.99, epsilon = 0.001,
		center=True, scale=True, training=True)        
		conv3_4 = tf.layers.max_pooling2d(conv3_3, 2, 2)
        # Output layer, a, b distribution 
		out = tf.layers.conv2d(conv3_3, 128, 1, activation=tf.nn.softmax)
		out = tf.contrib.layers.flatten(out)
	return out
		
def model_fn(features, labels, mode):
	#Build the CNN
	logits_train = conv_net(features,64,64,reuse=False)
	#If in prediction mode end early
	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode, logits_train)

	#Define Loss and Optimizer
	labels = tf.contrib.layers.flatten(labels)
	loss_op = tf.contrib.losses.mean_squared_error(logits_train, labels)	
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
	train_op = optimizer.minimize(loss_op)
	
	# Evaluate the accuracy of the model 
	acc_op = tf.metrics.accuracy(labels,logits_train)

	estim_specs = tf.estimator.EstimatorSpec(mode=mode,loss=loss_op, train_op=train_op, eval_metric_ops={'accuracy': acc_op})

	return estim_specs
#--------------------------- Train Vars --------------------------------------------------------
# Training Parameters
learning_rate = 0.001
num_steps = 2000
batch_size = 2500 #?
img_size = 64
data_path = '/home/steven/CMPS/Images/pj_prepro/64x64'
X_train, Y_train_rgb, mean_image = util.load_databatch(data_path, 1, img_size)
num_input = X_train.shape[0]
for j in range(num_input):
		X_train[j] = util.preprocess_image(X_train[j])
L_train = X_train[:num_input-batch_size,:,:,0] #L-channel only
ab_train = X_train[:num_input-batch_size,:,:,1:] #ab-channels
L_test = X_train[num_input-batch_size:,:,:,0] #L-channel only
ab_test = X_train[num_input-batch_size:,:,:,1:] #ab-channels

#Build the estimator
model = tf.estimator.Estimator(model_fn)

#Define the input function for training 
input_fn = tf.estimator.inputs.numpy_input_fn(x={'images':L_train}, y=ab_train,
batch_size=batch_size, num_epochs=None, shuffle=True)

# Train the Model
model.train(input_fn, steps=num_steps)


#Evaluate Model
#Input test  
input_fn = tf.estimator.inputs.numpy_input_fn(x={'images':L_test}, y=ab_test,
batch_size=batch_size, num_epochs=None, shuffle=False)
e = model.evaluate(input_fn)

print("Testing Accuracy:", e['accuracy'])
	
