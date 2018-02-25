import numpy as np
import tensorflow as tf

from open_fashion import train_segmentation
from texture_mapper import plug_texture


def labels():
	return ['pants', 'tshirts', 'shirts']

def train(datafolder, datafile, modelfolder):
	'''
		@param datafolder {string} the path to the folder with dataset labels i.e. db/data/
		@param datafile  {string} the name of the file with dataset labels, no path
		@param modelfolder {string} the path to folder where models should be stored
		@return dict{} dictionary of models for each label in mlfashion.labels()
	'''
	pass
	
def texture_swap(texture, cloath, models, original_image):
	'''
		@param texture {image\tensor3d\numpy3d} texture that we want to inject
		@param cloath {string} the cloath that we want to replace; one of the in mlfashion.labels()
		@param models {dict} dictionary of each model for each texture
		@param original_image {image\tensor3d\numpy3d`} the original image that we want to change
		@return {image} image with replaced cloath with texture
	'''
	pass	

def loadmodels(modelsfolder):
	'''
		@param modelfolder the path to the folder where models are saved
		@return dict{} dictionary of the models for each label in mlfashion.labels()
	'''
	pass