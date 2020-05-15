'''
   File that computes features for a set of images

   ex. python compute_features.py --data_dir=/mnt/images/ --model=vgg19 --model_path=./vgg_19.ckpt

'''

import scipy.misc as misc
import pickle
import tensorflow as tf
import numpy as np
import argparse
import fnmatch
import sys
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pylab as plt
from ts_to_image import recurrence_plot
import gc

slim = tf.contrib.slim

'''

   Recursively obtains all images in the directory specified

'''


def min_max_transform(data):
    r"""
        min_max_scaler `data`.

        Parameters
        ----------
        data
          description: time series
          shape: list

        Returns
        -------
        new_data
           description:time series
           shape: list.
        """
    data = np.array(data)
    data_array = data.reshape((len(data), 1))
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(data_array)
    normalized_data = scaler.transform(data_array)
    new_data = []
    for i in range(normalized_data.shape[0]):
        new_data.append(normalized_data[i][0])
    return new_data


def compute_feature_of_batch_ts_with_cnn(file_path_of_ts,file_path_of_feature,cnn_model_name,file_path_of_pretrained_model):
    r'''
    compute feature of somme time series with pretrained CNN
    :param file_path_of_ts: file path of time series
    :param file_path_of_feature: file path of saving feature
    :param cnn_model_name: name of CNN model
    :param file_path_of_pretrained_model: file path of pretrained CNN
    :return: ''
    '''
    #tf.reset_default_graph()
        #read data
    data=pd.read_csv(file_path_of_ts)
    #data=data.sample(20)
    #change dataframe to list
    id_list=data.iloc[:,0].tolist()
    data_list=change_dataframe_to_dict_(data)

    model = cnn_model_name
    checkpoint_file =file_path_of_pretrained_model


    # I only have these because I thought some take in size of (299,299), but maybe not
    if 'inception' in model: height, width, channels = 224, 224, 3
    if 'resnet' in model:    height, width, channels = 224, 224, 3
    if 'vgg' in model:       height, width, channels = 224, 224, 3

    if model == 'inception_resnet_v2': height, width, channels = 299, 299, 3

    x = tf.placeholder(tf.float32, shape=(1, height, width, channels))

    # load up model specific stuff
    if model == 'inception_v1':
        #from inception_v1 import *
        from nets import inception_v1


        arg_scope = inception_v1.inception_v1_arg_scope()
        with slim.arg_scope(arg_scope):
            logits, end_points = inception_v1.inception_v1(x, is_training=False, num_classes=None)
            features = end_points['AvgPool_0a_7x7']
            # print('logits')
            # print(logits.shape)
            # print('features')
            # print(features.shape)
    elif model == 'inception_v2':
        #from inception_v2 import *
        from nets import inception_v2

        arg_scope = inception_v2.inception_v2_arg_scope()
        with slim.arg_scope(arg_scope):
            logits, end_points = inception_v2(x, is_training=False, num_classes=None)
            features = end_points['AvgPool_1a']
    elif model == 'inception_v3':
        #from inception_v3 import *
        from nets import inception_v3

        arg_scope = inception_v3.inception_v3_arg_scope()
        with slim.arg_scope(arg_scope):
            logits, end_points = inception_v3(x, is_training=False, num_classes=None)
            features = end_points['AvgPool_1a']
    elif model == 'inception_resnet_v2':
        #from inception_resnet_v2 import *
        from nets import inception_resnet_v2

        arg_scope = inception_resnet_v2.inception_resnet_v2_arg_scope()
        with slim.arg_scope(arg_scope):
            logits, end_points = inception_resnet_v2(x, is_training=False, num_classes=1001)
            features = end_points['PreLogitsFlatten']
    elif model == 'resnet_v1_50':
        #from resnet_v1 import *

        from nets import resnet_v1

        arg_scope = resnet_v1.resnet_arg_scope()
        with slim.arg_scope(arg_scope):
            logits, end_points = resnet_v1.resnet_v1_50(x, is_training=False, num_classes=1000)
            features = end_points['global_pool']
    elif model == 'resnet_v1_101':
        #from resnet_v1 import *
        from nets import resnet_v1

        arg_scope = resnet_v1.resnet_arg_scope()
        with slim.arg_scope(arg_scope):
            logits, end_points = resnet_v1.resnet_v1_101(x, is_training=False, num_classes=1000)
            features = end_points['global_pool']
    elif model == 'vgg_16':
        #from vgg import *
        from nets import vgg

        arg_scope = vgg.vgg_arg_scope()
        with slim.arg_scope(arg_scope):
            logits, end_points = vgg.vgg_16(x, is_training=False)
            features = end_points['vgg_16/fc8']
    elif model == 'vgg_19':
        #from vgg import *
        from nets import vgg

        arg_scope = vgg.vgg_arg_scope()
        with slim.arg_scope(arg_scope):
            logits, end_points = vgg.vgg_19(x, is_training=False)
            features = end_points['vgg_19/fc8']
    #cpu_config = tf.ConfigProto(intra_op_parallelism_threads = 8, inter_op_parallelism_threads = 8, device_count = {'CPU': 3})
    #sess = tf.Session(config = cpu_config)
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, checkpoint_file)
    feature_list=[]
    count_temp=0

    for i in range(len(data_list)):
        count_temp=count_temp+1
        #imaging ts
        ts_dict=data_list[i]
        ts=ts_dict['ts']
        id=ts_dict['id']
        new_ts = min_max_transform(ts)
        normalized = np.array(new_ts)
        fig, ax = plt.subplots()
        #plt.imshow(recurrence_plot.rec_plot(normalized), cmap=plt.cm.gray)
        plt.imshow(recurrence_plot.rec_plot(normalized))
        ax.set_xticks([])
        ax.set_yticks([])
        #print(id)
        path="inception-v1/"+id+".jpg"
        plt.savefig(path)
        plt.close(fig)
        #compute feature
        # #begin to compute features
        image = misc.imread(path)
        #from matplotlib.pyplot import imread
        #image=imread(path)
        # print('image')
        # print(image.size)
        image = misc.imresize(image, (height, width))
        image = np.expand_dims(image, 0)
        feature = np.squeeze(sess.run(features, feed_dict={x: image}))
        feature_list.append(feature)
        # print('feature-test')
        # print(feature)
        os.remove(path)
        if count_temp%100==0:
            print (count_temp)
        #begin to process parellel result and write_to_csv
    feature_array=np.array(feature_list)

    feature_df=pd.DataFrame(feature_array)
    # print(feature_df.shape)
    # print(len(id_list))
    #add id
    feature_df.insert(loc=0, column='id', value=id_list)
    # print(feature_final_df.shape)
    # print(feature_final_df.head())
    feature_df.to_csv(file_path_of_feature,index=False)
    gc.collect()
    # sess.close()

def remove_none_elements_from_list_(list):
    r'''
    remove none elements from list
    :param list: list
    :return: list
    '''
    return [e for e in list if(pd.notnull(e))]


def change_dataframe_to_dict_(df):
    r'''
    change dataframe to dict
    :param df: dataframe
    :return: list
    '''
    data_list=[]
    for i in range(df.shape[0]):
        ts_list=remove_none_elements_from_list_(df.iloc[i,1:].tolist())
        id=df.iloc[i,0]
        ts_dict={'ts':ts_list,'id':id}
        data_list.append(ts_dict)
    return data_list

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    warnings.simplefilter(action='ignore', category=FutureWarning)
    # #test
    ##
    num_cores=2
    #file_path_of_ts='../ts-data/M4/M4DataSet/Yearly-train.csv'
    file_path_of_ts='../ts-data/Tourism/tourism-yearly-train.csv'
    file_path_of_feature='cnn-features/Tourism/tourism-yearly-train-feature-inception_v1.csv'
    #file_path_of_feature='cnn-features/M4/Yearly-train-feature-inception_v1.csv'
    #cnn_model_name='inception_v1'
    cnn_model_name='inception_v1'
    file_path_of_pretrained_model='pretrained-models/inception_v1.ckpt'
    compute_feature_of_batch_ts_with_cnn(file_path_of_ts,file_path_of_feature,cnn_model_name,file_path_of_pretrained_model)


