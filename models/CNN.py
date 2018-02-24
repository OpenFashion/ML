import numpy as np
import tensorflow as tf

from time import time

flags = tf.app.flags

def fit(X, Y, model):
	'''
		train the tf graph model here 
		@param X input tensor 
		@param Y ouput tensor
		@param model pretrained or loaded model that we are willing to update
	'''
	pass


def configurate():
	'''
		sets up global variables and seed for tf environment
	'''
	pass

def create_graph_segmentation():
	'''
		creates tf graph for segmentation analysis 
	'''
	# get dimensional variables
	ch = flags.num_channels
	w = flags.width
	h = flags.height
	batch_size = flags.batch_size
	# create tuplex with input image size
	input_shape = (batch_size, h, w, ch)
	# define placeholders for input and output tensors
	X = tf.placeholder(tf.float64, shape=input_shape, name='X')
	Y = tf.placeholder(tf.float64, shape=input_shape, name='Y')
	# define fisrt convolusion
	W1_shape = (5, 5, ch, 30)
	W1_init = init_filter_weight(W1_shape)
	b1_init = init_bias(W1_shape[-1])
	# define variable in tf
	W1 = tf.Variable(W1_init)
	b1 = tf.Variable(b1_init)
	Z1 = convpool(X, W1, b1)
	# define second convolusion
	W2_shape = (5, 5 W1_shape[-1], 50)
	W2_init = init_filter_weight(W2_shape)
	b2_init = init_bias(W2_shape[-1])
	# define variable in tf
	W2 = tf.Variable(W2_init)
	b2 = tf.Variable(b2_init)
	Z2 = convpool(Z1, W2, b2)
	# define fc layer TODO: calculate the input size
	W3_init = init_weight(input, 1024)
	b3_init = init_bias(1024)
	# define in tf
	W3 = tf.Variable(W3_init)
	b3 = tf.Variable(b3_init)
	# flatter the current flow
	Z2_shape = Z2.get_shape().as_list()
	# Z2_shape[0] is number of samples in tensor4D, the rest are flatten
	Z2f = tf.reshape(Z2, [Z2_shape[0], np.prod(Z2_shape[1:])
	Z3 = tf.nn.relu(tf.matmul(Z2f, W3) + b3)
	# define second fc layer
	# TODO FINISH, time to get sleep
	pass

def create_graph_degrees():
	'''
		create tf graph for body degrees of freedom identification
	'''
	pass

def predict(X, model):
	'''
		performs inference
		@param X input tesor
		@param model pretrained model
		@return Y predicting output
	'''
	pass
	return None

def convpool(X, W, b):
    '''
		default pool size is (2,2)
		@param X input tensor
		@param W convolusion weight matrix
		@param b bias for convolusion result
		@return biased convolusion result
	'''
    conv_out = tf.nn.conv2d(X, W, strides=[1, 1, 1, 1], padding='SAME')
    conv_out = tf.nn.bias_add(conv_out, b)
    pool_out = tf.nn.max_pool(conv_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    return tf.nn.relu(pool_out)

def init_filter_weight(shape):
	'''
		@param shape the shape of the filter
		@return normal random filter with shape shape
	'''
	returnn np.random.randn(*shape)/ np.sqrt(2 / np.prod(shape[:-1]),dtype=np.float64)

def init_weight(M, K):
	'''
		@param M input size
		@param K output size
		@return normal random matrix (M,K)
	'''
	return (np.random.randn(M, K,dtype=float64) / np.sqrt(M + K)).astype(np.float64)

def init_bias(size):
	'''
		@param size of the bias vector
		@return zero vector of size size
	'''
	return np.zeros(size, dtype=np.float64)