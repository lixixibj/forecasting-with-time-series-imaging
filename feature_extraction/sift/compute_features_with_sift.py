#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 16:40:50 2019

@author: xixi li
"""

# ts image feature extraction
from ts_to_image import recurrence_plot
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import sys
import os
import pandas as pd

path = os.getcwd()
import LLC_main
import pylab as plt

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


def image_based_one_ts_feature_extraction_with_sift(ts_dict):
    r"""
        extract one ts imaging feature with pretrained 200 basic descriptors from M4 dataset.

        Parameters
        ----------
        ts
           description: time series
           shape: list .

        Returns
        -------
        feature
           description: ts imaging feature with 4200 dimension 200*(1*1+2*2+4*4)
            shape: list.
        """
    #if (retrain == False):
    ts=ts_dict['ts']
    id=ts_dict['id']
    new_ts = min_max_transform(ts)
    normalized = np.array(new_ts)
    fig, ax = plt.subplots()
    plt.imshow(recurrence_plot.rec_plot(normalized), cmap=plt.cm.gray)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(id+".png")
    plt.close(fig)
    feature = LLC_main.get_codes_for_one_time_series(id+".png", [1, 2, 4])
    os.remove(id+".png")
    #use CNN
    return feature[0]

    # retrain to get basic descriptors
    # else:

def remove_none_elements_from_list(list):
    r'''
    remove none elements fromm list
    :param list:
    :return: list
    '''
    return [e for e in list if(pd.notnull(e))]


def change_dataframe_to_dict(df):
    r'''
    change dataframe to dict
    :param df:
    :return: dict
    '''
    data_list=[]
    for i in range(df.shape[0]):
        ts_list=remove_none_elements_from_list(df.iloc[i,1:].tolist())
        id=df.iloc[i,0]
        ts_dict={'ts':ts_list,'id':id}
        data_list.append(ts_dict)
    return data_list




def image_based_batch_ts_feature_extraction_with_sift(file_path_of_ts,file_path_of_feature,num_cores):
    r'''
    feature extraction of some time series with sift
    :param file_path_of_ts: file path of time series
    :param file_path_of_feature: file path for saving extracted features
    :param num_cores: number of cores for parallell computing
    :return:
    '''
    #read data
    data=pd.read_csv(file_path_of_ts)
    #data=data.sample(20)
    #change dataframe to list
    id_list=data.iloc[:,0].tolist()
    data_list=change_dataframe_to_dict(data)
    #print(data_list)

    import multiprocessing
    from functools import partial
    #cores = multiprocessing.cpu_count()
    cores=num_cores
    # print(cores)
    pool = multiprocessing.Pool(processes=cores)
    func = partial(image_based_one_ts_feature_extraction_with_sift)
    multivarite_ts_list=pool.map(func, data_list)
    pool.close()
    pool.join()
    #begin to process parellel result and write_to_csv
    feature_array=np.array(multivarite_ts_list)

    feature_df=pd.DataFrame(feature_array)
    #add id
    feature_df.insert(loc=0, column='id', value=id_list)
    # print(feature_final_df.shape)
    # print(feature_final_df.head())
    feature_df.to_csv(file_path_of_feature,index=False)

#test
if __name__ == "__main__":
    num_cores=3
    file_path_of_ts='../ts-data/Tourism/tourism-quarterly-train.csv'
    file_path_of_feature='sift-features/Tourism/tourism-quarterly-train-feature-sift.csv'
    #feature extraction with sift
    image_based_batch_ts_feature_extraction_with_sift(file_path_of_ts,file_path_of_feature,num_cores)
