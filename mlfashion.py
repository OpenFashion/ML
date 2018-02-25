import numpy as np
import tensorflow as tf

from models.CNN import CNN
from open_fashion import train_segmentation
from texture_mapper import plug_texture, get_segmented_cloath
from data.preprocessing import *


def labels():
	return ['pants', 'tshirts', 'shirts']

def train(datafolder, datafile, modelfolder):
	'''
		@param datafolder {string} the path to the folder with dataset labels i.e. db/data/
		@param datafile  {string} the name of the file with dataset labels, no path
		@param modelfolder {string} the path to folder where models should be stored
		@return dict{} dictionary of models for each label in mlfashion.labels()
	'''
	models = {}
	X,Y,_ = get_data_info(datafolder, datafile)
	
	# read input tensor
	X = get_tensor(X).astype(np.float32)
	for label in labels():
		models[label] = CNN(label)
		# create separate thread or process
		
		# get output tensor
		print('Training for ' + label)
		T = get_output_tensor(Y, label).astype(np.float32)
		train_segmentation(models[label], X, T, modelfolder)
		models[label].save(modelfolder + 'safesave/' + label + '.ckpt')
		models[label].save(modelfolder + label + '.ckpt')
		del T

		# end of part in parallel

	return models
	
def texture_swap(texture, cloath, models, original_image):
	'''
		@param texture {image\tensor3d\numpy3d} texture that we want to inject
		@param cloath {string} the cloath that we want to replace; one of the in mlfashion.labels()
		@param models {dict} dictionary of each model for each texture
		@param original_image {image\tensor3d\numpy3d`} the original image that we want to change
		@return {image} image with replaced cloath with texture
	'''
	segment = get_segmented_cloath(models[cloath], original_image)
	return plug_texture(texture, cloath, original_image, segment)

def load_models(modelsfolder):
	'''
		@param modelfolder the path to the folder where models are saved
		@return dict{} dictionary of the models for each label in mlfashion.labels()
	'''
	models = {}
	for label in labels():
		models[label] = CNN.load_model(modelsfolder + label + '0.ckpt', label)
	return models


def main(_):
	#train('data/', 'data.csv','trained_models/')
	X,Y,_ = get_data_info('data/', 'data.csv')
	
	# read input tensor
	X = get_tensor(X).astype(np.float32)
	models = {}
	models['pants'] = CNN('pants')
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
	texture_swap(texture, labels()[0], models, X[0][:,:,:3])
if __name__ == '__main__':
	tf.app.run()