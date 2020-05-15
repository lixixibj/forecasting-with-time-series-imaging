#这边实现将codes金字塔max池化的功能
import scipy.io
import numpy.matlib
import numpy as np
import numpy.linalg
import math
import os.path
import LLC_coding

def LLC_pooling(B,X,pyramid,knn,img_width,img_height,X1,Y):
    r"""
        LLC pooling for one image.

        Parameters
        ----------
        B
           description: basic descriptors got by k-means 
           shape: (128*200) .
        X
           description: descriptors of one image
           shape: (128*n), n is the number of descriptors of one image
        pyramid
           description: spatial levels
           shape: list, such as [1,2,4]
        knn
           description: number of k-nearest descriptors 
           shape: int .
        img_width
           description: width of the image
           shape: int
        img_height
           description: height of the image
           shape: int 
        X1
           description: The abscissa position of all descriptors in each image
           shape: int
        Y
           description: The ordinate position of all descriptors in each image
           shape: int 
    

        Returns
        -------
        beta
           description: coefficients of the descriptors 
           shape: (200*21)
        """
    dSize = B.shape[1]
    nSmp = X.shape[1]
    
    idxBin = np.transpose(np.zeros(nSmp))
    
    # llc coding
    llc_codes = LLC_coding.LLC_coding_appr(np.transpose(B),np.transpose(X),knn)    
    llc_codes = np.transpose(llc_codes)
    
    pLevels = len(pyramid) #spatial levels
    pBins = np.array(pyramid)*np.array(pyramid) #spatial bins on each level
    tBins = sum(pBins)
    
    beta = np.matrix(np.zeros((dSize,tBins)))
    bId = -1
    
    for i in range(pLevels):
        nBins = pBins[i]
        
        wUnit = img_width / pyramid[i]
        hUnit = img_height / pyramid[i]
        
        # find to which spatial bin each local descriptor belongs
        xBin = np.ceil(X1 / wUnit)
        yBin = np.ceil(Y / hUnit)
        idxBin = (yBin - 1)*pyramid[i] + xBin
        
        for j in range(nBins):
            bId = bId + 1
            sidxBin = np.where(idxBin == j+1)[0]
            if len(sidxBin) == 0:
                continue
            beta[:,bId] = np.transpose(np.matrix(np.amax(llc_codes[:,sidxBin],axis=1)))
    
    beta = np.transpose(np.transpose(beta).ravel())
    beta = beta/math.sqrt(sum(np.square(beta)))
    return beta