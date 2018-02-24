import tensorflow as tf
from scipy.misc import imread, imsave
from scimage.color import rgb2gray
from models.edge_detection import get_edge_detected_images


def get_data_info(data_info_filename):
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

    data = np.array(open(data_info_filename, 'r').read().split('\n'))
    X = data[0, :]
    Yseg = data[1, :]
    Ybody = data[2, :]
    return X, Yseg, Ybody


def get_tensor(X):
    '''
		@param X array of image files name to read 
		@return tensor of size hxwxch from X with edge stacking
	'''
    h = tf.app.flags.height
    w = tf.app.flags.width
    N = X.shape[0]
    ch = tf.app.flags.num_channels
    tensor = np.zeros((N, h, w, ch))
    for x, i in enumerate(X):
        tensor[i] = process_image(x)

    return tensor


def get_output_tensor(Y):
    '''
		@param Y array of image files name to read
		@return tensor of size hxw from Y
	'''
    return np.array(imread(Y, mode='P'), dtype=float)


def process_image(image_file):
    '''
		@param image_file name of the file with image
		@return tensor3D of image with 4 channels RGB + edge
	'''
    image = np.array(imread(image_file, mode='RGB'), dtype=float)
    gray = get_edge_detected_images(rgb2gray(image))
    return np.hstack((image, gray))
