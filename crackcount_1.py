# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 13:53:29 2020

@author: Alex Watson

Basic machine learning algorithm code for determining cracks on a rock face
"""

import matplotlib.pyplot as plt
import numpy as np

# Import datasets, classifiers and performance metrics
#from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

# Import Image importing functions
#from PIL import Image


#===============================Functions======================================

## function for importing image data
#new_resolution = [258,258]

#def down_sample_pictures(image,nx,ny):
#    # blah
#    return small_image

def loadimgdata(dPar):
    """
    function for loading the image data into python and combining it into a single list
    Params:
        dPar: dictionary of parameters (# images, image width, image height)
    NOTE: image dimensions must be the same for all photographs used in this function.
        Traces must be in Greyscale, while bases must be in RGB.
        """
    images = np.zeros((dPar['n'], dPar['h'], dPar['w'], 3)) #4d array w/ RGB data
    traces = np.zeros((dPar['n'], dPar['h'], dPar['w'])) #3d array, no RGB data      
    for i in np.arange(dPar['n']):
        images[i] = plt.imread("Data/Base/base%i.jpg"%i)
        traces[i] = plt.imread("Data/Trace/trace%i.jpg"%i)
    return images, traces


def threshold_model(x,threshold):
    """
    function that tries to label cracks based off of a simple logical
    Params:
        x: 3d array of a single base image (width, height, RGB)
        threshold: number from 0 to 255, color value/intensity of a single pixel of an image
    """
    predict = np.logical_and(x[:,:,0] <=threshold, x[:,:,1] <=threshold, x[:,:,2] <=threshold)
    return predict


# challenge -> find best threshold automaticaly
def train_model(x, y):
    """
    function that trains the threshold model using the training data by altering the threshold
    Parameters:
        x: training base images
        y: crack traces (labels) pertaining to the training images x
        dPar: dictionary of parameters (# images, image width, image height)
        NOTE: originally I wanted to replace np.arange(5) in the nested for loop with np.arange(0.5*dPar['n']),
        but this was breaking the code somehow. any idea how I could possibly fix that?
        (same as with np.arange(3) in test_model)
    """
    bestThreshold = 255
    for i in np.arange(10):
        predict = np.zeros(y.shape)
        trained_threshold_model = lambda x: threshold_model(x,bestThreshold)
        for j in np.arange(5):
            predict[j] = trained_threshold_model(x[j])
        if np.sum(predict) >= np.sum(y):
            bestThreshold = 0.5*bestThreshold
        else:
            bestThreshold = bestThreshold + 0.5*bestThreshold
    trained_threshold_model = lambda x: threshold_model(x,bestThreshold)        
    return trained_threshold_model, bestThreshold
    

def test_model(x,y,train_model,dPar,use_weights):
    """
    function that tests the trained model against the test data, and computes the error of the predictions
    Parameters:
        x: testing/validation base images
        y: crack traces (labels) pertaining to the images x
        train_model: the trained model from the train_model function
        dPar: dictionary of parameters (# images, image width, image height)
        use_weights: if 'True', then will calculate different weights for noncracks and cracks
    """
    cost = np.zeros(y.shape)
    predict = np.zeros(y.shape)
    for k in np.arange(3):
        ni = np.sum(y[k])
        N  = dPar['h']*dPar['w']
        n0 = N-ni
        if use_weights == 'True':
            w = np.ones(y[k].shape)
            w1 =((ni/N)**-1)
            w2 = ((n0/N)**-1)
            w[y[k]==1] = w1
            w[y[k]==0] = w2
        else:
            w = np.ones([y[k].shape])
            w = (ni/N)**-1
        predict[k] = train_model(x[k])
        cost[k] = (predict[k]-y[k])*w
    total_cost = np.sum(cost)
    return total_cost, predict


def confmatrix(predict,y):
    """
    function that creates a simple confusion matrix of tested predictions versus the actual crack traces,
    and also computes the percent of guesses correct
    Parameters:
        predict: tested predictions from the test_model function
        y: the crack traces (labels) pertaining to the testing/validation images
    """
    pospos = np.sum(np.logical_and(predict == 1, y ==1))
    negneg = np.sum(np.logical_and(predict == 0, y==0))
    posneg = np.sum(np.logical_and(predict == 1, y==0))
    negpos = np.sum(np.logical_and(predict == 0, y == 1))
    confmatrix = [[pospos, posneg], [negpos, negneg]]
    percent_correct = (pospos + negneg)/(posneg + negpos + pospos + negneg)
    return confmatrix, percent_correct

#================================Parameters====================================

dPar = { 'n'   : 10,    # number of total photos
         'h'   : 4000,  # pixel height of photos
         'w'   : 6000  # pixel width of photos
        }

#===========================Run the Functions==================================
images, traces = loadimgdata(dPar)

for i in np.arange(10):
   traces[i] = np.logical_and(traces[i] <=2, traces[i] >= 0)

X_train, X_test, y_train, y_test = train_test_split(
    images, traces, test_size=0.5, shuffle=False)

X_vali, X_test, y_vali, y_test = train_test_split(
    X_test, y_test, test_size=0.4, shuffle=False)

model, threshold = train_model(X_train, y_train)

t_cost, predict = test_model(X_vali, y_vali, model, dPar, 'True')

cm, correct = confmatrix(predict, y_vali)
print cm, correct