import scipy.io
import numpy.matlib
import numpy as np
import numpy.linalg
import math
import os.path

def LLC_coding_appr(B,X,knn):
    r"""
        LLC coding for one image.

        Parameters
        ----------
        B
           description: basic descriptors got by k-means 
           shape: (200*128) .
        X
           description: descriptors of one image
           shape: (n*128), n is the number of descriptors of one image

        Returns
        -------
        Coeff
           description: linear coefficients of the descriptors 
           shape: (n*200). n is the number of descriptors of one image
        """
    beta = 1e-4

    nframe=X.shape[0]
    nbase=B.shape[0]
    
    XX = np.sum(np.multiply(X,X),axis=1)
    BB = np.sum(np.multiply(B,B),axis=1)
    
    D = np.matrix(np.transpose(numpy.matlib.repmat(XX,nbase,1))) - 2*np.matrix(X)*np.transpose(np.matrix(B)) + np.transpose(np.matrix(np.transpose(np.matlib.repmat(BB,nframe,1))))
    IDX = np.zeros(shape=(nframe, knn))
    
    for i in range(0,nframe):
        d = D[i,]
        d = np.array(d)[0]
        idx = np.argsort(d)
        IDX[i,] = idx[:knn]
    
    II = np.identity(knn)
    Coeff = np.zeros(shape=(nframe, nbase))
    for i in range(0,nframe):
        idx = IDX[i,]
        idx = idx.astype(int)
        z = B[idx,] - np.matlib.repmat(X[i,],knn,1)
        C = np.matrix(z)*np.transpose(np.matrix(z))
        C = C + np.matrix(II)*beta*np.trace(C)
        w = numpy.linalg.solve(C,np.ones((knn,1)))
        w = w/sum(w)
        Coeff[i,idx] = np.transpose(w)
    return Coeff