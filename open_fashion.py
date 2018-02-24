import tensorflow as tf
import numpy as np
import models.CNN as CNN
from models.edge_detection import get_edge_detected_images
from data.preprocessing import *

flags = tf.app.flags

flags.DEFINE_float('random_seed', 1, 'Random seed for numpy random')
flags.DEFINE_float('learing_rate', 1e-6, 'Learning rate')
flags.DEFINE_float('momentum', 0.99, 'Momentum')
flags.DEFINE_float('decay', 1e-3, 'Decay factor for learing rate')
flags.DEFINE_integer('num_steps', 100, 'Number of update steps per batch')
flags.DEFINE_integer('num_mix', 1000, 'Number of time to mix data set before getting batch')
flags.DEFINE_integer('batch_size', 500, 'Size of single batch')
flags.DEFINE_integer('width', 600, 'Width of the input image')
flags.DEFINE_integer('height', 400, 'Height of the input image')
flags.DEFINE_integer('num_channels', 4, 'Number of channels per input')
flags.DEFINE_integer('num_body_part', 14, 'Number points in body part detection')
flags.DEFINE_string('label_detection',
                    'default',
                    'Feature type, supports [shorts, jeans, longsleeves, shortsleeves]')


def main(_):
    '''
        main function to get data, preprocess data, train model
    '''
    np.random.seed(flags.random_seed)
    train_segmentation()


def train_segmentation():
    CNN.create_graph_segmentation()
    CNN.configurate()
    X, Yseg, Ybody = get_data_info('data/labels.txt')
    for i in range(flags.num_mix):
        np.shuffle(X)
        Xbatch = get_tensor(X[:flags.batch_size])
        Ybatch = get_output_tensor(Y[:flags.batch_size])
        CNN.fit(Xbatch, Ybatch)
    CNN.save('destination_file.ckpt')


if __name__ == '__main__':
    tf.app.run()
