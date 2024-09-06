# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 22:19:55 2023

@author: Vijaya
"""

import librosa, librosa.display
import matplotlib.pyplot as plt
import scipy
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import r2_score
from sklearn import linear_model
import os
import scipy.signal as signal
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator

from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
from numpy import concatenate
from math import sqrt
from scipy import stats
import keras.backend as K
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
     
n_mfcc = 13
n_mels = 13

standardRoom_mic_locs = [
    [1.5,3.5, 0.9], [6.0,3.5,0.9],  # mic 1  # mic 2
]

        
def update_dict(dict_key, count, indata ):
    new_df = pd.DataFrame()
    for i in range(count):  
        ss =  f'{i}'
        dict_key_updated = dict_key + "_" + ss
        new_df.loc[:, dict_key_updated] = indata[i]
    return new_df

def get_item(test_list, search_term):
    items = []

    for index in range(len(test_list)):
        items.append(test_list[index][search_term])
    return items

def flat_list(test_list):
    return [item for sublist in test_list for item in sublist]


def get_items(dict_key, count, indata):
    items = [[]] * count
    for i in range(count):
        ss =  f'{i}'
        dict_key_updated = dict_key + "_" + ss
        items[i] = get_item(indata, dict_key_updated )
    return items
 
def flat_list_n_items(indata):
    xx = []
    for n in range(len(indata)):
        xx.append(flat_list(indata[0]))
    return xx


def double_exponential_smoothing(series, alpha, beta, n_preds=2):
    """
    Given a series, alpha, beta and n_preds (number of
    forecast/prediction steps), perform the prediction.
    """
    n_record = len(series)
    results = np.zeros(n_record + n_preds)

    # first value remains the same as series,
    # as there is no history to learn from;
    # and the initial trend is the slope/difference
    # between the first two value of the series
    level = series[0]
    #print(series[0])
    results[0] = series[0]
   
    if n_record == 1:
        return series[0]
    
    trend = beta * (series[1] - series[0])
    for t in range(1, n_record + 1):
        if t >= n_record:
            # forecasting new points
            value = results[t - 1]
        else:
            value = series[t]

        previous_level = level
        level = alpha * value + (1 - alpha) * (level + trend)
        trend = beta * (level - previous_level) + (1 - beta) * trend 
        results[t] = level + trend

    # for forecasting beyond the first new point,
    # the level and trend is all fixed
    if n_preds > 1:
        results[n_record + 1:] = level + np.arange(2, n_preds + 1) * trend

    return round(results[-1],2)     
# =============================================================================
# 
# with open ('datasetFinal_random_locus_dev-clean1-50-200-50-1.5.pickle', 'rb') as f:
#      audio_features_dataset_training_10_locus_1 = pickle.load(f)                
# 
# with open ('distanceDataset_random_locus_dev-clean1-50-200-50-1.5.pickle', 'rb') as f:
#      distance_dataset_10_locus_1 = pickle.load(f)
# =============================================================================
   
with open ('datasetFinal_random_locus_dev-clean1-50-200-50-1.5.pickle', 'rb') as f:
     audio_features_dataset_training_10_locus_1 = pickle.load(f)                

with open ('distanceDataset_random_locus_dev-clean1-50-200-50-1.5.pickle', 'rb') as f:
     distance_dataset_10_locus_1 = pickle.load(f)  
     
# =============================================================================
# with open ('C:/Users/Vijaya/PhD/chapter5/SignalDataset/datasetFinal_randum_locus_dev-clean1-10-1000.pickle', 'rb') as f:
#      audio_features_dataset_training_10_locus_1 = pickle.load(f)                
# 
# with open ('C:/Users/Vijaya/PhD/chapter5/SignalDataset/distanceDataset_randum_locus_dev-clean1-10-1000.pickle', 'rb') as f:
#      distance_dataset_10_locus_1 = pickle.load(f)
# 
# =============================================================================
# =============================================================================
# with open ('datasetFinal_randum_locus_dev-clean1-10-linear.pickle', 'rb') as f:
#      audio_features_dataset_training_10_locus_1 = pickle.load(f)                
# 
# with open ('distanceDataset_randum_locus_dev-clean1-10-linear.pickle', 'rb') as f:
#      distance_dataset_10_locus_1 = pickle.load(f)  
# =============================================================================
     
audio_features_dataset_training = audio_features_dataset_training_10_locus_1

audio_features_dataset_training = flat_list(audio_features_dataset_training)
audio_features_dataset_training = np.expand_dims(np.array(audio_features_dataset_training), axis=1).tolist()
    
distance_dataset_training = distance_dataset_10_locus_1
distance_dataset_training = flat_list(distance_dataset_training)
distance_dataset_training = np.expand_dims(np.array(distance_dataset_training), axis=1).tolist() 



############# extract audio data and distance data for training #############

audio_features_dataset_training_list = []

for i in range(len(audio_features_dataset_training)):
    audio_features_dataset_training_list.append(audio_features_dataset_training[i])
    
    
dataset_list = []

for i in range(len(audio_features_dataset_training_list)):
    dataset_list.append(audio_features_dataset_training_list[i])


mic1_coh_flat_list = [[]] * len(dataset_list)
mic1_abs_l_flat_list = [[]] * len(dataset_list)
mic1_abs_r_flat_list = [[]] * len(dataset_list)

abs_diff_flat_list = [[]] * len(dataset_list)

mic1_mfccs = [[]] * len(dataset_list)
mic1_melss = [[]] * len(dataset_list)

mic1_mfcc_l = [[]] * len(dataset_list)
mic1_mfcc_r = [[]] * len(dataset_list)
mic1_mels_l =  [[]] * len(dataset_list)
mic1_mels_r =  [[]] * len(dataset_list) 

mic2_coh_flat_list = [[]] * len(dataset_list)
mic2_abs_l_flat_list = [[]] * len(dataset_list)
mic2_abs_r_flat_list =[[]] * len(dataset_list)

mic2_mfccs = [[]] * len(dataset_list)
mic2_melss = [[]] * len(dataset_list)

mic2_mfcc_l = [[]] * len(dataset_list)
mic2_mfcc_r = [[]] * len(dataset_list)
mic2_mels_l = [[]] * len(dataset_list)
mic2_mels_r = [[]] * len(dataset_list)  
df_audio_features = [[]] * len(dataset_list)
    
for i in range(len(dataset_list)):
    dataset = dataset_list[i]        

    mic1_coh_flat_list[i] = flat_list(get_item(dataset,"mic1_coh" ))  

    mic1_abs_l_flat_list[i] = (flat_list(get_item(dataset,"mic1_abs_l" )))   
    mic1_abs_r_flat_list[i] = flat_list(get_item(dataset,"mic1_abs_r" ))
    
    mic1_mfcc_l[i] = flat_list_n_items(get_items("mic1_mfcc_l", n_mfcc, dataset))    
    mic1_mfcc_r[i] = flat_list_n_items(get_items("mic1_mfcc_r", n_mfcc, dataset))
    mic1_mels_l[i] = flat_list_n_items(get_items("mic1_mels_l", n_mels, dataset))
    mic1_mels_r[i] = flat_list_n_items(get_items("mic1_mels_r", n_mels, dataset))
    
    abs_diff_flat_list[i] = (flat_list(get_item(dataset,"mic1_abs_l" )))
    mic1_mfccs[i] = flat_list(get_item( dataset, "mic1_mfcc_full")) 
    mic1_melss[i] = flat_list(get_item( dataset, "mic1_mels_full")) 
    
      
    mic2_coh_flat_list[i] = flat_list(get_item(dataset,"mic2_coh" ))
    mic2_abs_l_flat_list[i] = flat_list(get_item(dataset,"mic2_abs_l" ))
    
    mic2_abs_r_flat_list[i] = flat_list(get_item(dataset,"mic2_abs_r" ))
    
    mic2_mfcc_l[i] = flat_list_n_items(get_items("mic2_mfcc_l", n_mfcc, dataset))
    mic2_mfcc_r[i] = flat_list_n_items(get_items("mic2_mfcc_r", n_mfcc, dataset))
    mic2_mels_l[i] = flat_list_n_items(get_items("mic2_mels_l", n_mels, dataset))
    mic2_mels_r[i] = flat_list_n_items(get_items("mic2_mels_r", n_mels, dataset))    
    
    mic2_mfccs[i] = flat_list(get_item(dataset, "mic2_mfcc_full")) 
    mic2_melss[i] = flat_list(get_item(dataset, "mic2_mels_full")) 
 
   
    df_audio_features[i] = pd.DataFrame(list(zip(mic1_coh_flat_list[i], mic1_abs_l_flat_list[i], mic1_abs_r_flat_list[i], mic2_coh_flat_list[i], mic2_abs_l_flat_list[i], mic2_abs_r_flat_list[i], 
                                                 abs_diff_flat_list[i] )),
                   columns =['mic1_coh', 'mic1_abs_l', "mic1_abs_r", "mic2_coh", "mic2_abs_l", "mic2_abs_r", "abs_diff"])
    
    df_audio_features[i] = df_audio_features[i].add(update_dict("mic1_mfcc_l", len(mic1_mfcc_l[i]), mic1_mfcc_l[i]) , fill_value=0)
    df_audio_features[i] = df_audio_features[i].add(update_dict("mic1_mfcc_r", len(mic1_mfcc_r[i]), mic1_mfcc_r[i]) , fill_value=0)
    df_audio_features[i] = df_audio_features[i].add(update_dict("mic1_mels_l", len(mic1_mels_l[i]), mic1_mels_l[i]) , fill_value=0)
    df_audio_features[i] = df_audio_features[i].add(update_dict("mic1_mels_r", len(mic1_mels_r[i]), mic1_mels_r[i]) , fill_value=0)
    
    
    df_audio_features[i] = df_audio_features[i].add(update_dict("mic2_mfcc_l", len(mic2_mfcc_l[i]), mic2_mfcc_l[i]) , fill_value=0)
    df_audio_features[i] = df_audio_features[i].add(update_dict("mic2_mfcc_r", len(mic2_mfcc_r[i]), mic2_mfcc_r[i]) , fill_value=0)
    df_audio_features[i] = df_audio_features[i].add(update_dict("mic2_mels_l", len(mic2_mels_l[i]), mic2_mels_l[i]) , fill_value=0)
    df_audio_features[i] = df_audio_features[i].add(update_dict("mic2_mels_r", len(mic2_mels_r[i]), mic2_mels_r[i]) , fill_value=0)

scaler = MinMaxScaler()
 

    
##### train data ################### 

lables_flat_list_train = []
for i in range(len( distance_dataset_training)):
    labels = flat_list(get_item(distance_dataset_training[i],"mic" ))
    lables_flat_list_train.append(labels)

mic1_distance_train = flat_list(get_item(distance_dataset_training[0], 'mic1-distance'))
mic2_distance_train = flat_list(get_item(distance_dataset_training[0], 'mic2-distance'))

norm_dist_diff_train = []
for i in range(len(mic1_distance_train)):
    n_d = np.subtract(mic1_distance_train[i], mic2_distance_train[i]) / math.dist(standardRoom_mic_locs[0], standardRoom_mic_locs[1])
    norm_dist_diff_train.append(n_d)  

   
train_X_list = []

for i in range(len(df_audio_features)):
    features = df_audio_features[i][['mic1_coh' ,"mic2_coh"]]  # df_audio_features[0] #df_audio_features[0][['mic1_coh',"mic2_coh"]] 
    features = scaler.fit_transform(features)
    train_X_list.append(features)
    #train_X_list.append(df_audio_features[i][['mic1_coh' ,"mic2_coh"]])

df_for_training_X_list = []

for i in range(len(train_X_list)):
    df_for_training_X_list.append(pd.DataFrame(train_X_list[i]))

train_Y_list = []

for i in range(len(lables_flat_list_train)):
    labels = pd.DataFrame(lables_flat_list_train[i], columns=['mic'])
    # labels = scaler.fit_transform(labels)
    train_Y_list.append(labels)

df_for_training_Y_list = []

for i in range(len(train_Y_list)):
    df_for_training_Y_list.append(pd.DataFrame(train_Y_list[i]))



X_train = np.concatenate([df.values[np.newaxis, :] for df in df_for_training_X_list], axis=0)
#x_test = x_test.reshape(1, x_test.shape[0],x_test.shape[1])

y_train = np.concatenate([df.values[np.newaxis, :] for df in df_for_training_Y_list], axis=0) #df_for_training_Y
#y_test =  y_test.reshape(1, y_test.shape[0],y_test.shape[1])

#X_train, x_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

X_train_2 = X_train.reshape(-1, 2)
y_train_2 = y_train.reshape(-1).tolist()

mic1_coh_test = X_train_2[:,0]
mic2_coh_test = X_train_2[:,1]

mic1_coh_test_smooth = []
mic2_coh_test_smooth = []
mic1_coh_test_tmp = []
mic2_coh_test_tmp = []


for i in range(len(mic1_coh_test)):   
    mic1_coh_test_tmp.append(mic1_coh_test[i])
    m1_s = double_exponential_smoothing(mic1_coh_test_tmp, 0.05, 0.01) 
    mic1_coh_test_smooth.append(m1_s)

for i in range(len(mic2_coh_test)):  
    mic2_coh_test_tmp.append(mic2_coh_test[i])
    m2_s = double_exponential_smoothing(mic2_coh_test_tmp, 0.05,0.01) 
    mic2_coh_test_smooth.append(m2_s)

pred_listMic = []

for i in range(len(mic1_coh_test_smooth)):
    if mic1_coh_test_smooth[i] >= mic2_coh_test_smooth[i]:
        pred_listMic.append('1')
    else:
        pred_listMic.append('2')

listMic = []

for i in range(len(mic1_coh_test)):
    if mic1_coh_test[i] >= mic2_coh_test[i]:
        listMic.append('1')
    else:
        listMic.append('2')

# =============================================================================
# fig, axs = plt.subplots(
#         nrows=3, ncols=1, sharex=True, sharey=False,
#         gridspec_kw={'height_ratios':[5,5, 5]}
#         
#         )
# 
# axs[0].plot(y_train_2[0:200],c ="blue", label='GTruth')
# axs[0].legend(loc= 'upper left')
# axs[0].grid()
# axs[0].set_ylabel('$y$ (transition)')
# 
# axs[1].plot(listMic[0:200],c ="blue", label='GTruth')
# axs[1].legend(loc= 'upper left')
# axs[1].grid()
# axs[1].set_ylabel('$y$ (transition)')
# 
# axs[2].plot(pred_listMic[0:200], c ="green", label='!DCNN-CL')
# axs[2].legend(loc= 'upper left')
# axs[2].grid()  
# axs[2].set_xlabel('$x$ (locus steps)') 
# axs[2].set_ylabel('$y$ (transition)')
# 
# 
# plt.show()   
# =============================================================================


ideal = [int(x) for x in y_train_2]
xS = [int(x) for x in pred_listMic]

sq_error = np.subtract(ideal, xS) ** 2
mse = sq_error.mean()

wtd_mse = []    
                                        
for x, y in zip(sq_error, norm_dist_diff_train):   
    wtd_mse.append(x * y/np.sum(norm_dist_diff_train))

wtd_mse_smooth = sum(wtd_mse)

print("wtd_mse_smooth: ", wtd_mse_smooth)   


with open ('y_pred_exp.pickle', 'wb' ) as f:
    pickle.dump(xS, f)
                                                                          #print('wtd_mse_with_smooth: ', round(wtd_mse_mean,2))
with open ('wtd_mse_exp.pickle', 'wb' ) as f:
    pickle.dump(wtd_mse, f)
    





