import numpy as np
import models.CNN as CNN
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.misc import imread, imresize
from data.preprocessing import *


FLAGS = tf.app.flags.FLAGS

def plug_texture(texture, cloath, original_image, segment):
	# repetate texture over the image size
	if(texture.shape[0] > 400):
		factor = 400. / texture.shape[0]
		texture = imresize(texture, (200, 400))
		resized_texture = texture
	else:
		sizes = (original_image.shape[0] // texture.shape[0] +1 , original_image.shape[1] // texture.shape[1] + 1)
		resized_texture = np.repeat(texture, sizes[1], axis=1)
		resized_texture = np.repeat(resized_texture, sizes[0], axis=0)
		resized_texture = resized_texture[:original_image.shape[0], :original_image.shape[1]]
	# direct 1 to 1 mapping with no surface analysis
	resized_texture[segment[:,:] == 0] = [0,0,0]

	# create new image
	image = original_image
	image[segment[:,:] == 1] = resized_texture[segment[:,:] == 1]
	
	plt.imshow(image)
	plt.show()
	exit()
	return image

def get_segmented_cloath(model, image):
	# model is always loaded
	# perform prediction
	def concat(image):
		image = imresize(image, (FLAGS.height, FLAGS.width))
		gray = get_edge_detected_images(rgb2gray(image))
		return np.concatenate((image, gray),axis=2)
	return model.predict_label(concat(image))

def mapping_example():
	X, Yseg, Ybody = get_data_info(FLAGS.data_path, FLAGS.data_labels)
	image = np.array(imread(X[0], mode='RGB')) /255.
	
	cloath = 'pants'
	seg = get_output_tensor(Yseg, cloath)[0].reshape((200,400))
	texture = np.array([
		[[1,0,0], [1,0,0], [1,0,0],[1,0,0], [1,0,0], [1,0,0],[1,0,0], [1,0,0], [1,0,0], [1,0,0]],
		[[1,0,0], [1,0,0], [1,0,0],[1,0,0], [1,0,0], [1,0,0],[1,0,0], [1,0,0], [1,0,0], [1,0,0]],
		[[1,0,0], [1,0,0], [1,0,0],[1,0,0], [1,0,0], [1,0,0],[1,0,0], [1,0,0], [1,0,0], [1,0,0]],
		[[1,0,0], [1,0,0], [1,0,0],[1,0,0], [1,0,0], [1,0,0],[1,0,0], [1,0,0], [1,0,0], [1,0,0]],
		[[1,0,0], [1,0,0], [1,0,0],[1,0,0], [1,0,0], [1,0,0],[1,0,0], [1,0,0], [1,0,0], [1,0,0]],
		[[1,0,0], [1,0,0], [1,0,0],[1,0,0], [1,0,0], [1,0,0],[1,0,0], [1,0,0], [1,0,0], [1,0,0]],
		[[1,0,0], [1,0,0], [1,0,0],[1,0,0], [1,0,0], [1,0,0],[1,0,0], [1,0,0], [1,0,0], [1,0,0]],
		[[1,0,0], [1,0,0], [1,0,0],[1,0,0], [1,0,0], [1,0,0],[1,0,0], [1,0,0], [1,0,0], [1,0,0]],
		[[0,0,1], [0,0,1], [0,0,1],[0,0,1], [0,0,1], [0,0,1],[0,0,1], [0,0,1], [0,0,1], [0,0,1]],
		[[0,0,1], [0,0,1], [0,0,1],[0,0,1], [0,0,1], [0,0,1],[0,0,1], [0,0,1], [0,0,1], [0,0,1]],
		[[0,0,1], [0,0,1], [0,0,1],[0,0,1], [0,0,1], [0,0,1],[0,0,1], [0,0,1], [0,0,1], [0,0,1]],
		[[0,0,1], [0,0,1], [0,0,1],[0,0,1], [0,0,1], [0,0,1],[0,0,1], [0,0,1], [0,0,1], [0,0,1]],
		[[0,0,1], [0,0,1], [0,0,1],[0,0,1], [0,0,1], [0,0,1],[0,0,1], [0,0,1], [0,0,1], [0,0,1]],
		[[0,0,1], [0,0,1], [0,0,1],[0,0,1], [0,0,1], [0,0,1],[0,0,1], [0,0,1], [0,0,1], [0,0,1]],
		[[0,0,1], [0,0,1], [0,0,1],[0,0,1], [0,0,1], [0,0,1],[0,0,1], [0,0,1], [0,0,1], [0,0,1]]
	],dtype=np.float32)

	paper = np.array(imread('images/texture.jpeg'))
	n_image = plug_texture(texture, cloath, image)
	plt.imshow(n_image)
	plt.show()
	