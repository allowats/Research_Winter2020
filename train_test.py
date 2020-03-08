# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 17:25:46 2020

@author: awhip
"""
import img_seg.train as train
import img_seg.inference as inf


x_train, x_test, y_train, y_test, pred = train.main("Data/Base/","Data/Trace", "RF", "model.p")

inf.main("Data/Base/","model.p","Data/Output")
