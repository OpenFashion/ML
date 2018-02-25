import numpy as np
import tensorflow as tf

from time import time

flags = tf.app.flags

session = tf.Session()
predict = None
loss = None
optimazer = None


def savemodel(model_file):
	'''
		@param model_file destination path
	'''
	saver = tf.train.Saver()
	saver.save(session, model_file)


def loadmodel(model_file):
	'''
		@param model_file source of model path
	'''
	saver = tf.train.Saver()
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


def create_graph_body_detection():
	'''
		creates tf graph for detection analysis 
	'''
	# get dimensional variables
	ch = flags.num_channels
	w = flags.width
	h = flags.height
	batch_size = flags.batch_size
	
def predict_label(X):
	'''
		performs inference
		@param X input tensor
		@return Y predicting output
	'''
	return session.run(predict, feed_dict={X: X})

def init_weight(M, K):
	'''
		@param M input size
		@param K output size
		@return normal random matrix (M,K)
	'''
	return (np.random.randn(M, K) / np.sqrt(M + K)).astype(np.float32)


def init_bias(size):
	'''
		@param size of the bias vector
		@return zero vector of size size
	'''
	return np.zeros(size, dtype=np.float32)
