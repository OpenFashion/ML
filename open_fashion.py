import tensorflow as tf
import numpy as np
import models.CNN as CNN
import models.body_part_detection as BPD
from sklearn.utils import shuffle
from models.edge_detection import get_edge_detected_images
from data.preprocessing import *
from texture_mapper import *

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('random_seed', 1, 'Random seed for numpy random')
flags.DEFINE_float('learning_rate', 1e-4, 'Learning rate')
flags.DEFINE_float('momentum', 0.95, 'Momentum')
flags.DEFINE_float('decay', 0.9999, 'Decay factor for learing rate')
flags.DEFINE_integer('body_height', 50, 'Height of the image for body part detection')
flags.DEFINE_integer('body_width', 50, 'Width of the image for body part detection')
flags.DEFINE_integer('num_steps', 1, 'Number of update steps per batch')
flags.DEFINE_integer('num_mix', 4, 'Number of time to mix data set before getting batch')
flags.DEFINE_integer('batch_size', 13, 'Size of single batch')
flags.DEFINE_integer('width', 400, 'Width of the input image')
flags.DEFINE_integer('height', 200, 'Height of the input image')
flags.DEFINE_integer('num_channels', 4, 'Number of channels per input')
flags.DEFINE_integer('num_body_part', 14, 'Number points in body part detection')
flags.DEFINE_string('label_detection',
					'pants',
					'Feature type, supports [tshirts, pants, shirts]')
flags.DEFINE_string('data_labels', 'data.csv', 'File name with  all labels references')
flags.DEFINE_string('data_path', 'data/', 'Path to all datafiles')
flags.DEFINE_string('file_bpd', 'trained_models/bpdmodel.ckpt', 'File name to store the model for part detecitons')
flags.DEFINE_string('file_cnn', 'trainwd_models/cnnmodel.ckpt', 'File name to storethe model for segmentation')


def train_parts():
	BPD.create_graph_body_detection()
	BPD.configurate()
	X, Yseg, Ybody = get_data_info(FLAGS.data_path, FLAGS.data_labels)

	for i in range(FLAGS.num_mix):
		np.random.shuffle(X)
		Xbatch = body_detection_preprocessing(X[:FLAGS.batch_size])
		Ybatch = read_body_labels(Ybody[:FLAGS.batch_size])
		BPD.fit(Xbatch, Ybatch)
	BPD.save(FLAGS.file_bpd)

def train_segmentation(cnn, X, Y, modelfolder):
	label = cnn.label
	X, Y= shuffle(X,Y)
	X_train = X[:-FLAGS.batch_size]
	Y_train = Y[:-FLAGS.batch_size]
	X_test = X[FLAGS.batch_size:]
	Y_test = Y[FLAGS.batch_size:]
	costs = []
	c = cnn.calculate_cost(X_test, Y_test)
	print("initial cost = " + str(c))
	costs.append(c)
	for i in range(FLAGS.num_mix):
		X_train, Y_train = shuffle(X_train,Y_train)
		Xbatch = X_train[:FLAGS.batch_size]
		Ybatch = Y_train[:FLAGS.batch_size]
		cnn.fit(Xbatch, Ybatch)
		# print out current model cost
		c = cnn.calculate_cost(X_test, Y_test)
		print("iter = " + str(i) + ", cost = " + str(c))
		costs.append(c)

		if i % 30 == 0:
			cnn.save(modelfolder + 'intermediate/' + label + str(i) + '.ckpt')
		
def texture_mapping():
	mapping_example()
