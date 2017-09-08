import numpy as np 
from keras import backend as K 

'''
Module with custom metrics 
'''

smooth = 0

def dice(y_true, y_pred):
    # Should move this to its own function, not in here. 
    # Dice = 2*TP/(2*TP+FP+FN)
    # Calculate the negative of the index (in case we want to minimize it)
    
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    y_int = y_true*y_pred
    return (2.*K.sum(y_int) + smooth)/ (K.sum(y_true) + K.sum(y_pred)+smooth)

def jaccard(y_true, y_pred):
    # Jaccard metric 
    # TP / (TP + FP + FN)
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    y_int = y_true*y_pred 
    return (K.sum(y_int)+smooth) / (K.sum(y_true) + K.sum(y_pred) - K.sum(y_int)+smooth)

def genDice(y_true, y_pred):
    # Generalized DICE metric for multi-categorical classification
    y_true = K.reshape(y_true, (y_true.shape[0]*y_true.shape[1], y_true.shape[2]))
    y_pred = K.reshape(y_pred, (y_pred.shape[0]*y_pred.shape[1], y_pred.shape[2]))
    y_int = y_pred*y_true 
    
    weights = 1/K.square(K.sum(y_true, axis=0)) # square over the flattened axis 
    numerator = 2*K.sum(weights * K.sum(y_int, axis=0), axis=-1)
    denominator = K.sum(weights * (K.sum(y_true,axis=0) + K.sum(y_pred,axis=0)), axis=-1)
    return numerator/denominator