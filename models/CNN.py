import numpy as np
import tensorflow as tf

from time import time

flags = tf.app.flags

session = tf.Session()
predict = None
loss = None
optimazer = None
saver = tf.train.Saver()


def savemodel(model_file):
    '''
		@param model_file destination path
	'''
    saver.save(session, model_file)


def loadmodel(model_file):
    '''
		@param model_file source of model path
	'''
    saver.restore(session, model_file)


def fit(X, Y):
    '''
		train the tf graph model here 
		@param X input tensor 
		@param Y ouput tensor
	'''
    for i in range(flags.num_steps):
        session.run(optimazer, feed_dict={X: X, Y: Y})


def configurate():
    '''
		sets up global variables and seed for tf environment
	'''
    init = tf.global_variables_initializer()
    session.run(init)


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
    # define fisrt convolution
    W1_shape = (20, 20, ch, 40)
    W1_init = init_filter_weight(W1_shape)
    b1_init = init_bias(W1_shape[-1])
    # define variable in tf
    W1 = tf.Variable(W1_init)
    b1 = tf.Variable(b1_init)
    Z1 = convpool(X, W1, b1)
    # define second convolution
    W2_shape = (20, 20, W1_shape[-1], 80)
    W2_init = init_filter_weight(W2_shape)
    b2_init = init_bias(W2_shape[-1])
    # define variable in tf
    W2 = tf.Variable(W2_init)
    b2 = tf.Variable(b2_init)
    Z2 = convpool(Z1, W2, b2)
    # define 3rd convolution
    W3_shape = (10, 10, W2_shape[-1], 70)
    W3_init = init_filter_weight(W3_shape)
    b3_init = init_bias(W3_shape[-1])
    # define variable in tf
    W3 = tf.Variable(W3_init)
    b3 = tf.Variable(b3_init)
    Z3 = convpool(Z2, W3, b3)
    # define fc layer TODO: calculate the input size
    W4_init = init_weight(input, 1024)
    b4_init = init_bias(1024)
    # define in tf
    W4 = tf.Variable(W4_init)
    b4 = tf.Variable(b4_init)
    # flatter the current flow
    Z3_shape = Z3.get_shape().as_list()
    # Z2_shape[0] is number of samples in tensor4D, the rest are flatten
    Z3f = tf.reshape(Z3, [Z3_shape[0], np.prod(Z3_shape[1:])])
    Z4 = tf.nn.relu(tf.matmul(Z3f, W4) + b4)
    # define second fc layer

    # prediction, cost,  optimazer
    pred = tf.nn.softmax(Z4)
    cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(pred), reduction_indices=1))
    optimazer = tf.train.MomentumOptimizer(learning_rate=flags.learning_rate, momentum=flags.momentum).minimize(cost)
    predict = tf.cast(pred + 0.5, tf.int32)


def create_graph_degrees():
    '''
		create tf graph for body degrees of freedom identification
	'''
    pass


def predict_label(X):
    '''
		performs inference
		@param X input tensor
		@return Y predicting output
	'''
    return session.run(predict, feed_dict={X: X})


def convpool(X, W, b):
    '''
		default maxpool size is (8,8)
		@param X input tensor
		@param W convolution weight matrix
		@param b bias for convolution result
		@return biased convolution result
	'''
    conv_out = tf.nn.conv2d(X, W, strides=[1, 1, 1, 1], padding='SAME')
    conv_out = tf.nn.bias_add(conv_out, b)
    pool_out = tf.nn.max_pool(conv_out, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], padding='SAME')
    return tf.nn.relu(pool_out)


def init_filter_weight(shape):
    '''
		@param shape the shape of the filter
		@return normal random filter with shape shape
	'''
    return np.random.randn(*shape) / np.sqrt(2 / np.prod(shape[:-1]), dtype=np.float64)


def init_weight(M, K):
    '''
		@param M input size
		@param K output size
		@return normal random matrix (M,K)
	'''
    return (np.random.randn(M, K) / np.sqrt(M + K)).astype(np.float64)


def init_bias(size):
    '''
		@param size of the bias vector
		@return zero vector of size size
	'''
    return np.zeros(size, dtype=np.float64)
