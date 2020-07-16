# -*- coding: utf-8 -*-

import cv2
import numpy as np
#sys.path.append('../LLC/')
import LLC_pooling
import pandas as pd
import datetime
starttime = datetime.datetime.now()
def getSift(file_path):  
    r"""
        get key point of one image.

        Parameters
        ----------
        file_path
           description: path of the image 
           str: 
    

        Returns
        -------
        des[1]
           description: features of all key points(descriptors) 
           shape: (n*128)
         X_array
           description: The abscissa position of all descriptors in each image
           shape: array
         Y_array
           description: The ordinate position of all descriptors in each image
           shape: array
        """ 
    img_path1 =file_path  
    img = cv2.imread(img_path1)  
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #sift = cv2.SIFT()
    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(gray,None)  
    des = sift.compute(gray,kp)  
    pt_list=[]
    x_axis_list=[]
    y_axis_list=[]
    for x_y in des[0]:
        x_axis_list.append(x_y.pt[0])
        y_axis_list.append(x_y.pt[1])
    X_array=np.array(x_axis_list)
    Y_array=np.array(y_axis_list)
    return des[1],X_array,Y_array

def get_resolution_rate_of_image(file_path):
    r"""
        get width and height of one image.

        Parameters
        ----------
        file_path
           description: path of the image 
           str: 
    
        Returns
        -------
        img_width
           description: width of the image
           shape: int
        img_height
           description: height of the image
           shape: int 
        """
    img = cv2.imread(file_path)
    sp = img.shape
    height = sp[0]  # height(rows) of image
    width = sp[1]  # width(colums) of image
    chanael = sp[2]  # the pixels value is made up of three primary colors
    #print ( 'width: %d \nheight: %d \nnumber: %d' % (width, height, chanael))
    return width,height


#get codes for one time series
#parm1 file_path_of_ts_image
#parm4 pyramid_list：pyramid_ for example:[1,2,4] 1*1 2*2 4*4
def get_codes_for_one_time_series(file_path_of_ts_image,pyramid_list):
    r"""
        get codes for one ts.

        Parameters
        ----------
        file_path_of_ts_image
           description: path of the ts 
           shape: str
        pyramid_list
           description: spatial level
           shape: list    pyramid_ for example:[1,2,4] 1*1 2*2 4*4

        Returns
        -------
        beta_list
           description: codes for one ts
           shape:list
        """ 
    import os
    root = os.getcwd() 
    #print (root)
    #print ("===start121===")
    #read basic descriptors, here we get 200 basic descriptors from M4 dataset
    name = "centroid_200.csv"
    path_of_basic_descriptors=os.path.join(root, name)

    base_df=pd.read_csv(path_of_basic_descriptors)
    #numpy_matrix = base_df.as_matrix()
    numpy_matrix = base_df.values
    #codebook B
    B=np.transpose(numpy_matrix)
    pyramid=pyramid_list
    total=0
    for pl in pyramid_list:
        total=total+pl*pl
    knn=5
    dime=total*base_df.shape[0]
    beta_all=np.empty(shape=[0, dime])
    i=0
    j=0
    des_feature,X_array,Y_array=getSift(file_path_of_ts_image)
    if isinstance(des_feature,(np.ndarray)):
        img_width,img_height=get_resolution_rate_of_image(file_path_of_ts_image)
        X=np.transpose(des_feature)
        X1=X_array
        Y=Y_array
        beta=LLC_pooling.LLC_pooling(B,X,pyramid,knn,img_width,img_height,X1,Y)
        beta_array=np.transpose(np.array(beta))
        beta_list=beta_array.tolist()
    else:
        beta_list=''
    return beta_list


             
            