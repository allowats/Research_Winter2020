# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 13:53:29 2020

@author: Alex Watson

Basic machine learning algorithm code for determining cracks on a rock face
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d
# Import datasets, classifiers and performance metrics
#from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc
# Import Image importing functions
#from PIL import Image


#===============================Functions======================================

## function for importing image data
#new_resolution = [258,258]

#def down_sample_pictures(image,nx,ny):
#    # blah
#    return small_image

def subsample(x, y, sz, w, h, n, ntot):
    n1 = w/sz
    n1 = int(n1)
    n2 = h/sz
    n2 = int(n2)
    n_f = n*ntot
    n_f = int(n_f)
    bases = np.zeros([n_f,sz,sz,3])
    traces = np.zeros([n_f,sz,sz])
    base_temp = np.zeros([ntot,sz,sz,3])
    trace_temp = np.zeros([ntot,sz,sz])
    img = np.zeros([n2,sz,sz,3])
    img2 = np.zeros([n2,sz,sz])
    for k in np.arange(n):
        for i in np.arange(n1):
            for j in np.arange(n2):
                img[j] = x[k,sz*j:sz*j+sz,sz*i:sz*i+sz]
                img2[j] = y[k,sz*j:sz*j+sz,sz*i:sz*i+sz]
            base_temp[n2*i:n2*i+n2] = img
            trace_temp[n2*i:n2*i+n2] = img2
        bases[ntot*k:ntot*k+ntot] = base_temp
        traces[ntot*k:ntot*k+ntot]= trace_temp
    return bases,traces
    

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
    images, traces = subsample(images, traces, dPar['sz'], dPar['w'], dPar['h'], dPar['n'], dPar['ntot'])
    for i in np.arange(dPar['ntot']*dPar['n']):
        traces[i] = np.logical_and(traces[i] <=2, traces[i] >= 0)
        traces[i] = gaussian_filter(traces[i], sigma=3)
    return images, traces


def threshold_model(x, sobel_x, x_a, threshold1, threshold2, threshold3):
    """
    function that tries to label cracks based off of a simple logical
    Params:
        x: 3d array of a single base image (width, height, RGB)
        threshold: number from 0 to 255, color value/intensity of a single pixel of an image
    """
    predict1 = np.logical_and(x[:,:,0] <=threshold1, x[:,:,1] <=threshold1, x[:,:,2] <=threshold1)
    predict2 = x_a<=threshold3
    predict3 = sobel_x>=threshold2
    predict = (predict1+predict2+predict3)/3
    predict = gaussian_filter(predict, sigma = 1)
    return predict


# challenge -> find best threshold automaticaly
def train_model(x, y, ntot, n):
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
    bestThreshold1 = 255
    bestThreshold2 = 5
    bestThreshold3 = 100
    ntrain = ntot*n/2
    ntrain = int(ntrain)
    for i in np.arange(10):
        predict = np.zeros(y.shape)
        trained_threshold_model = lambda x,x_c,x_a: threshold_model(x,x_c,x_a,bestThreshold1,bestThreshold2,bestThreshold3)
        for j in np.arange(ntrain):
            x_c = sobel(x[j])
            x_avg = avg_conv(x[j])
            predict[j] = trained_threshold_model(x[j],x_c,x_avg)
        if np.sum(predict) >= np.sum(y):
            bestThreshold1 = bestThreshold1/(2+(1/(i+1)))
            bestThreshold3 = bestThreshold3/(2+(1/(i+1)))
            bestThreshold2 = bestThreshold2 + (1/(i+2))*bestThreshold2
        else:
            bestThreshold1 = bestThreshold1 + (1/(i+2))*bestThreshold1
            bestThreshold3 = bestThreshold3 + (1/(i+2))*bestThreshold3
            bestThreshold2 = bestThreshold2/(2+(1/(i+1)))
    trained_threshold_model = lambda x,x_c,x_a: threshold_model(x,x_c,x_a,bestThreshold1,bestThreshold2,bestThreshold3)        
    return trained_threshold_model, bestThreshold1, bestThreshold2
    

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
    ntest = 0.3*dPar['ntot']*dPar['n']
    ntest = int(ntest)
    for k in np.arange(ntest):
        ni = np.sum(y[k]) + 1**-5
        N  = dPar['sz']*dPar['sz']
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
        x_c = sobel(x[k])
        x_a = avg_conv(x[k])
        predict[k] = train_model(x[k],x_c, x_a)
        cost[k] = (predict[k]-y[k])*w
    total_cost = np.sum(cost)
    return total_cost, predict


def confmatrix(predict,y, thresh):
    """
    function that creates a simple confusion matrix of tested predictions versus the actual crack traces,
    and also computes the percent of guesses correct
    Parameters:
        predict: tested predictions from the test_model function
        y: the crack traces (labels) pertaining to the testing/validation images
    """
    pospos = (np.sum(np.logical_and(predict>=thresh,y>0.15)))
    negneg = (np.sum(np.logical_and(predict<thresh,y<=0.15)))
    posneg = (np.sum(np.logical_and(predict>=thresh,y<=0.15)))
    negpos = (np.sum(np.logical_and(predict<thresh,y>0.15)))
    
    confmatrix = np.array([[pospos/(pospos+negpos),posneg/(posneg+negneg)],
                           [negpos/(pospos+negpos),negneg/(posneg+negneg)]])
    return confmatrix

def sobel(x):
    s_filter = np.array([[-1,0,1],
                         [-2,0,2],
                         [-1,0,1]])
    x_grey = (x[:,:,0]+x[:,:,1]+x[:,:,2])/3
    x_x = convolve2d(x_grey, s_filter)
    x_y = convolve2d(x_grey, np.flip(s_filter.T, axis=0))
    x_sobel = np.sqrt(x_x**2 + x_y**2)
    x_sobel *= 255/ x_sobel.max()
    x_sobel = x_sobel[0:250, 0:250]
    return x_sobel    

def avg_conv(x):
    avg_filter = np.ones([10,10])
    x_grey = (x[:,:,0]+x[:,:,1]+x[:,:,2])/3
    x_avg = convolve2d(x_grey, avg_filter)
    x_avg *= 255/x_avg.max()
    x_avg = x_avg[0:250, 0:250]
    return x_avg
    
def roc(predict, y):
    nit=100
    tpr = np.zeros(nit)
    fpr = np.zeros(nit)
    thresh = 0   
    for i in np.arange(nit-1):
        tpr[i] = np.sum(np.logical_and(predict>=thresh,y>0.15))/np.sum(y>0.15)
        fpr[i] = np.sum(np.logical_and(predict>=thresh,y<=0.15))/np.sum(y<=0.15)
        thresh = thresh+(i*1/nit)
    area = auc(fpr,tpr)
    return tpr, fpr, area
#================================Parameters====================================

dPar = { 'n'   : 10,    # number of total photos
         'h'   : 4000,  # pixel height of photos
         'w'   : 6000,  # pixel width of photos
         'sz'  : 250,
         'ntot': 384,
        }

vali_thresh = 0.05
#===========================Run the Functions==================================

images, traces = loadimgdata(dPar)

X_train, X_test, y_train, y_test = train_test_split(
    images, traces, test_size=0.5, shuffle=False)

X_vali, X_test, y_vali, y_test = train_test_split(
    X_test, y_test, test_size=0.4, shuffle=False)

model, threshold_color, threshold_edge = train_model(X_train, y_train, dPar['ntot'], dPar['n'])

t_cost, predict = test_model(X_vali, y_vali, model, dPar, 'True')

cm = confmatrix(predict, y_vali, vali_thresh)
print(np.sum(y_vali>0)/(np.sum(y_vali>0)+np.sum(y_vali==0)))

tpr, fpr, roc_auc = roc(predict, y_vali)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

tileNum = 665;

x_vali_grey = (X_vali[tileNum,:,:,0]+X_vali[tileNum,:,:,1]+X_vali[tileNum,:,:,2])/3    
plt.subplot(1,3,1)
plt.imshow(x_vali_grey, cmap='gray')
plt.subplot(1,3,2)    
plt.imshow(predict[tileNum] >vali_thresh, cmap='gray')
plt.subplot(1,3,3)
plt.imshow(y_vali[tileNum]>0.15, cmap='gray')


