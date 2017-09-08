import numpy as np 
from keras import backend as K 
from metrics import dice, jaccard 
'''
Module with custom loss functions 
'''

def diceLoss(y_true, y_pred):
    # Loss function for dice coefficient. Negative so we can minimize 
    return -dice(y_true, y_pred)

def modifiedDiceLoss(y_true, y_pred):
    # Reference: Milletari 2016, ArXiv: V-Net
    # Squares the denominator of the standard DICE coefficient to calculate 
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    y_int = y_true*y_pred
    return -(2.*K.sum(y_int))/ (K.sum(K.square(y_true)) + K.sum(K.square(y_pred)))

def generalizedDiceLoss(y_true, y_pred):
    # GDL, reference: https://arxiv.org/pdf/1707.03237.pdf 
    # Sudre et al, 2017, ArXiv (Generalised DICE overlap as a deep learning loss function for highly unbalanced segmentations)
    # DICE Loss for multi-category classification
    
    _EPSILON = K.epsilon()
    y_pred = K.clip(y_pred, _EPSILON, 1.0-_EPSILON)
    y_true = K.clip(y_true, _EPSILON, 1.0-_EPSILON)
    # First flatten the matrices for each channel (category) 
    ypredshape = y_pred.get_shape().as_list()
    ytrueshape = y_true.get_shape().as_list()
    
    dimp = K.prod(K.shape(y_pred)[:-1])
    dimt = K.prod(K.shape(y_true)[:-1])
    y_pred = K.reshape(y_pred, (dimp, ypredshape[-1]))
    y_true = K.reshape(y_true, (dimt, -1))
    
    y_int = y_pred*y_true 
    
    # Prevent dividing by 0 
    
    weights = 1/(K.square(K.sum(y_true, axis=0))) # square over the flattened axis

    numerator = 2*K.sum(weights*K.sum(y_int, axis=0), axis=-1)
    denominator = K.sum(weights*(K.sum(K.square(y_true), axis=0) + K.sum(K.square(y_pred),axis=0)), axis=-1)
    loss = - numerator/denominator 
    return loss
    
    
    
def jaccardLoss(y_true, y_pred):
    # Loss function is negative 
    return -jaccard(y_true,y_pred)

     
def weighted_pixelwise_crossentropy(class_weights):
    '''
    Custom loss function to do weighted pixelwise cross entropy loss 
    Done for however many classes are contained within class_weights
    '''
    _EPSILON = K.epsilon()
    def loss(y_true, y_pred):
        y_pred = K.clip(y_pred, _EPSILON, 1.0-_EPSILON)
        out = K.mean(-K.sum(class_weights*y_true*K.log(y_pred), axis=-1))
        return out 
    return loss   
    

def weighted_pixelwise_binary_crossentropy(class_weights):
    '''
    Custom loss function to do weighted pixelwise binary cross entropy loss.
    Assumes a final sigmoid activation 
    '''
    _EPSILON = K.epsilon()
    def loss(y_true, y_pred):
        y_pred = K.clip(y_pred, _EPSILON, 1.0-_EPSILON)
        out = -K.mean((y_true *class_weights[1]* K.log(y_pred) + (1.0 - y_true) * class_weights[0]* K.log(1.0 - y_pred)))
        return out 
    return loss 
    
def generic_unsupervised_loss(y_true, y_pred):
    # loss function that just returns the predicted value (which is actually the loss)  
    return y_pred 