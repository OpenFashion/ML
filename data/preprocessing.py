import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from scipy.misc import imread, imsave, imresize
from skimage.color import rgb2gray
from models.edge_detection import get_edge_detected_images

flags = tf.app.flags.FLAGS

def get_data_info(path, filename):
	'''
		the data_info_filename should follow the next patter
		"
		im1name im1segname img1bodyname
		.
		.
		.
		"
		image name includes path to file
		@param data_info_filename name of a file with names of all images
		@return X, Yseg, Ybody files name arrays 
	'''
	data = pd.read_csv(path+filename, delimiter=',', names=['original', 'parts', 'segmented'])
	X = path + data['original']
	Yseg = path + data['segmented']
	Ybody = path + data['parts']
	return X, Yseg, Ybody


def get_tensor(X):
	'''
		@param X array of image files name to read 
		@return tensor of size hxwxch from X with edge stacking
	'''
	h = tf.app.flags.FLAGS.height
	w = tf.app.flags.FLAGS.width
	N = X.shape[0]
	ch = tf.app.flags.FLAGS.num_channels
	tensor = np.zeros((N, h, w, ch))
	for i, x in enumerate(X):
		tensor[i] = process_image(x)
	return tensor


def get_output_tensor(Y, label):
	'''
		@param Y array of image files name to read
		@param label the part of the body to detect
		@return tensor of size hxw from Y
	'''
	# TODO: for dictionary define values later
	body_dictionary = {'tshirts': np.array([0.0,0.0,1.0]),
						'pants': np.array([0.0,1.0,0.0]),
						'shirts': np.array([1.0,0.0,0.0])
						}

	mapper = body_dictionary[label]
	img = np.array([imread(y, mode='RGB') for y in Y], dtype=np.float32) / 255.
	gray = (img * mapper).astype(np.float32).sum(axis=3)
	gray[gray > 0.95] = -1
	gray[gray != -1] = 1
	gray[gray == -1] = 0
	return gray.reshape(gray.shape[0], -1)

def body_detection_preprocessing(X):
	'''
		@param X array of original images
		@return tensor, preprocessed images
	'''
	w = flags.body_width
	h = flags.body_height
	N = X.shape[0]

	tensor = np.zeros_like((N,w,h,1))
	for i,x in enumerate(X):
		tensor[i] = preprocess_body(x)
	return tensor 


def preprocess_body(image_file):
	'''
		@param image_file single file name of orignal image
		@return resized edge detector
	'''
	w = flags.body_width
	h = flags.body_height
	image = np.array(imread(image_file, mode='RGB'),dtype=np.float32)
	gray = get_edge_detected_images(rgb2gray(image))
	return imresize(gray, (h,w), mode='P')


def read_body_labels(Y):
	'''
		@param Y file names with body part locations
		@return tensor2D with all resized locations
	'''
	size = flags.num_body_part * 2
	N = Y.shape[0]
	x_ration = flags.width / flags.body_width
	y_ration = flags.height / flags.body_height
	ration_fix = np.ones(size)
	ration_fix[:size//2] /= x_ration
	ration_fix[size//2:] /= y_ration 
	tensor = np.zeros_like((N, size))
	for i, y in enumerate(Y):
		tensor[i] = np.array(open(y,'r').split(), dtype=float32) * ration_fix
	
	return tensor


def process_image(image_file):
	'''
		@param image_file name of the file with image
		@return tensor3D of image with 4 channels RGB + edge
	'''
	image = np.array(imread(image_file, mode='RGB'), dtype=np.float32) / 255.
	image = imresize(image, (flags.height, flags.width))
	gray = get_edge_detected_images(rgb2gray(image))
	return np.concatenate((image, gray),axis=2)
