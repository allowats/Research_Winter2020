# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 17:25:46 2020

@author: awhip
"""
import img_seg.train as train
import img_seg.inference as inf
from sklearn.metrics import plot_confusion_matrix
import pickle as pkl
import matplotlib as plt

x_train, x_test, y_train, y_test, pred = train.main("Data/Base/","Data/Trace", "RF", "model.p")

model = pkl.load(open( "model.p", "rb" ) )

plot_confusion_matrix(model, x_test, y_test,
                                 cmap=plt.cm.Blues,
                                 normalize='true')
plt.savefig('Data/Output/ConfMatrix_RT.png', dpi=150)

#inf.main("Data/Base/","model.p","Data/Output")

