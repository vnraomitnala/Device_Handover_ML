# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 11:53:05 2023

@author: Vijaya
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from kerastuner.tuners import RandomSearch
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split
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

     
n_mfcc = 13
n_mels = 13

standardRoom_mic_locs = [
    [1.0,3.5, 0.9], [5.5,3.5,0.9],  # mic 1  # mic 2
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

     
# =============================================================================
# with open ('C:/Users/Vijaya/PhD/chapter5/SignalDataset/datasetFinal_randum_locus_dev-clean1-100-locus-20000.pickle', 'rb') as f:
#      audio_features_dataset_10_locus_1 = pickle.load(f)                
# 
# with open ('C:/Users/Vijaya/PhD/chapter5/SignalDataset/distanceDataset_randum_locus_dev-clean1-100-20000.pickle', 'rb') as f:
#      distance_dataset_10_locus_1 = pickle.load(f)
#      
# =============================================================================

with open ('datasetFinal_randum_locus_dev-clean1-10-1000.pickle', 'rb') as f:
     audio_features_dataset_10_locus_1 = pickle.load(f)                

with open ('distanceDataset_randum_locus_dev-clean1-10-1000.pickle', 'rb') as f:
     distance_dataset_10_locus_1 = pickle.load(f)
      
     
audio_features_dataset = audio_features_dataset_10_locus_1

audio_features_dataset = flat_list(audio_features_dataset)
audio_features_dataset = np.expand_dims(np.array(audio_features_dataset), axis=1).tolist()
    
distance_dataset = distance_dataset_10_locus_1
distance_dataset = flat_list(distance_dataset)
distance_dataset = np.expand_dims(np.array(distance_dataset), axis=1).tolist() 


############# extract audio data and distance data #############


audio_features_dataset_list = []

for i in range(len(audio_features_dataset)):
    audio_features_dataset_list.append(audio_features_dataset[i])
    
    
dataset_list = []

for i in range(len(audio_features_dataset_list)):
    dataset_list.append(audio_features_dataset_list[i])


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
    
    abs_diff_flat_list[i] = (flat_list(get_item(dataset,"abs_diff" )))
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

lables_flat_list_train = []
for i in range(len( distance_dataset)):
    labels = flat_list(get_item(distance_dataset[i],"mic" ))
    lables_flat_list_train.append(labels)

####### test data ###################
test_signal_index = 0

ground_truth_lables_test = get_item(distance_dataset[test_signal_index],"mic" )
lables_flat_list_test = flat_list(ground_truth_lables_test)

mic1_distance_test = flat_list(get_item(distance_dataset[test_signal_index], 'mic1-distance'))
mic2_distance_test = flat_list(get_item(distance_dataset[test_signal_index], 'mic2-distance'))

norm_dist_diff_test = []
for i in range(len(mic1_distance_test)):
    n_d = np.subtract(mic1_distance_test[i], mic2_distance_test[i]) / math.dist(standardRoom_mic_locs[0], standardRoom_mic_locs[1])
    norm_dist_diff_test.append(n_d)  
    
norm_dist_diff_test = pd.DataFrame(norm_dist_diff_test, columns=['dist-diff'])    

test_X =   df_audio_features[test_signal_index][['mic1_coh' ,"mic2_coh", "abs_diff", 'mic1_abs_l', "mic1_abs_r", 'mic2_abs_l', "mic2_abs_r" ]] # df_audio_features[1] #
 # df_audio_features[1] #
test_X = scaler.fit_transform(test_X)
df_for_testing_X =   df_audio_features[test_signal_index]#[['mic1_coh' ,"mic2_coh", "abs_diff" ]] # just for plotting

test_Y = pd.DataFrame(lables_flat_list_test, columns=['mic'])  
test_Y = scaler.fit_transform(test_Y)  

    
##### train data ################### 

mic1_distance_train = flat_list(get_item(distance_dataset[0], 'mic1-distance'))
mic2_distance_train = flat_list(get_item(distance_dataset[0], 'mic2-distance'))

norm_dist_diff_train = []
for i in range(len(mic1_distance_train)):
    n_d = np.subtract(mic1_distance_train[i], mic2_distance_train[i]) / math.dist(standardRoom_mic_locs[0], standardRoom_mic_locs[1])
    norm_dist_diff_train.append(n_d)  

   
train_X_list = []

for i in range(len(df_audio_features)):
    features = df_audio_features[i][['mic1_coh' ,"mic2_coh", "abs_diff", 'mic1_abs_l', "mic1_abs_r", 'mic2_abs_l', "mic2_abs_r"]]  # df_audio_features[0] #df_audio_features[0][['mic1_coh',"mic2_coh"]] 
    features = scaler.fit_transform(features)
    train_X_list.append(features)
    #train_X_list.append(df_audio_features[i][['mic1_coh' ,"mic2_coh"]])

df_for_training_X_list = []

for i in range(len(train_X_list)):
    df_for_training_X_list.append(pd.DataFrame(train_X_list[i]))

train_Y_list = []

for i in range(len(lables_flat_list_train)):
    labels = pd.DataFrame(lables_flat_list_train[i], columns=['mic'])
    labels = scaler.fit_transform(labels)
    train_Y_list.append(labels)

df_for_training_Y_list = []

for i in range(len(train_Y_list)):
    df_for_training_Y_list.append(pd.DataFrame(train_Y_list[i]))


X_train = np.concatenate([df.values[np.newaxis, :] for df in df_for_training_X_list], axis=0)
X_test = test_X
X_test = X_test.reshape(1, X_test.shape[0],X_test.shape[1])

y_train = np.concatenate([df.values[np.newaxis, :] for df in df_for_training_Y_list], axis=0) #df_for_training_Y
y_test =  test_Y
y_test =  y_test.reshape(1, y_test.shape[0],y_test.shape[1])

scaler2 = MinMaxScaler()
norm_dist_diff_train_scaled = scaler2.fit_transform(pd.DataFrame(norm_dist_diff_train))
norm_dist_diff_train_scaled_reshaped = norm_dist_diff_train_scaled.reshape(1,norm_dist_diff_train_scaled.shape[0], norm_dist_diff_train_scaled.shape[1] )
# Custom loss function
def custom_loss(y_true, y_pred):
    y_suppl = norm_dist_diff_train_scaled_reshaped
    y_suppl = tf.cast(y_suppl, dtype='float32')
    #tf.print(tf.shape(y_true))
    
    wmsc =  K.mean(tf.multiply(y_suppl,
            tf.square(tf.subtract(y_pred, y_true))
            ), axis= -1)
    return wmsc #K.mean(K.square(y_true - y_pred), axis=-1)


no_timesteps = 99
no_featues = 7


def build_model(hp):
    model = keras.Sequential()

    model.add(layers.Conv1D(filters=hp.Int('filters_1', min_value=100, max_value=128, step=32),
                            kernel_size=hp.Int('kernel_1', min_value=10, max_value=250, step=10),
                            activation='relu',
                            strides=1,
                            padding='same',
                            input_shape=(no_timesteps, no_featues)))

    model.add(layers.Conv1D(filters=hp.Int('filters_2', min_value=64, max_value=128, step=32),
                            kernel_size=hp.Int('kernel_2', min_value=10, max_value=200, step=10),
                            activation='relu',
                            strides=1,
                            padding='same',
                            input_shape=(no_timesteps, no_featues)))
    
    model.add(layers.Conv1D(filters=hp.Int('filters_3', min_value=32, max_value=128, step=32),
                            kernel_size=hp.Int('kernel_3', min_value=50, max_value=100, step=10),
                            activation='relu',
                            strides=1,
                            padding='same',
                            input_shape=(no_timesteps, no_featues)))    

    # Output layer
    model.add(layers.Dense(1))

    model.compile(optimizer='adam', loss=custom_loss, metrics = custom_loss)
    return model

X = X_train
y = y_train

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tuner = RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=50,
    executions_per_trial=1,
    directory='my_dir',
    project_name='my_project'
)




#stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

tuner.search(X_train, y_train, validation_data=(X_test, y_test), epochs=100)


best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print("Best Hyperparameters:")
print(best_hps.values)

best_model = tuner.hypermodel.build(best_hps)
best_model.summary()


best_model.fit(X_train, y_train, epochs=100,  batch_size=1, shuffle=False,  verbose=0, validation_data=(X_test, y_test))

with open ('C:/Users/Vijaya/PhD/chapter5/SignalDataset/model_10-200-100.pickle', 'wb' ) as f:
    pickle.dump(best_model, f)
