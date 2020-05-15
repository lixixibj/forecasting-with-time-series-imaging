# -*- coding: utf-8 -*-


#import pylab as plt
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
#plt.rcParams.update({'figure.max_open_warning':1})

def rec_plot(s, eps=None, steps=None):
    r"""
        transform ts to image with recurrence plot.

        Parameters
        ----------
        s
           description: time series 
           shape: list
    

        Returns
        -------
        Z
           description: rp matrix
           shape: array
        """ 
    if eps==None: eps=0.1
    if steps==None: steps=5
    N = s.size

    S = np.repeat(s[None,:], N, axis=0)
    
    Z = np.abs(S-S.T)/eps
    Z[Z>steps] = steps
 
    return Z
 



        