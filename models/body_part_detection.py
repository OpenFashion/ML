import numpy as np
import sklearn.svm as svm
import tensorflow as tf

def get_support_vectors(X, Y, model):
    '''
    @param X input tensor
    @param Y output tesor
    @param model pretrained model to update training
    @return support vectors
    '''
    model.fit(X,Y)
    return model.support_vectors_