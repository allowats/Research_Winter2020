# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 13:53:29 2020

@author: Alex Watson

Basic machine learning algorithm code for determining cracks on a rock face
"""

import matplotlib.pyplot as plt
import numpy as np

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

# Import Image importing functions
from PIL import Image

## function for importing image data

def loadimgdata(n, w, h):
    """
    function for loading the image data into python and combining it into a single list
    Params:
        n: # of total photographs
        w: pixel width of photographs
        h: pixel height of photographs
    NOTE: image dimensions must be the same for all photographs used in this function.
        Traces must be in Greyscale, while bases must be in RGB.
        """
    images = np.zeros((n, h, w, 3)) #4d array w/ RGB data
    traces = np.zeros((n, h, w)) #3d array, no RGB data      
    for i in np.arange(n):
        images[i] = plt.imread("Data/Base/base%i.jpg"%i)
        traces[i] = plt.imread("Data/Trace/trace%i.jpg"%i)
    base_and_trace = list(zip(images, traces))
    return base_and_trace, images, traces

base_and_trace, images, traces = loadimgdata(10,6000,4000)

n_samples = len(images)
data = images.reshape((n_samples, -1))
classifier = svm.SVC(gamma=0.001)

X_train, X_test, y_train, y_test = train_test_split(
    data, traces, test_size=0.2, shuffle=False)
classifier.fit(X_train, y_train)
predicted = classifier.predict(X_test)