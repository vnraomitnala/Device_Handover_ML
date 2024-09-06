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

     
with open ('datasetFinal_random_locus_dev-clean1-3-200-50.pickle', 'rb') as f:
     audio_features_dataset_training_10_locus_1 = pickle.load(f)                

with open ('distanceDataset_random_locus_dev-clean1-3-200-50.pickle', 'rb') as f:
     distance_dataset_10_locus_1 = pickle.load(f)
     
with open ('datasetFinal_random_locus_dev-clean1-10-10-50.pickle', 'rb') as f:
     audio_features_dataset_testing_10_locus_1 = pickle.load(f)                
     

with open ('distanceDataset_random_locus_dev-clean1-10-10-50.pickle', 'rb') as f:
     distance_dataset_testing_10_locus_1 = pickle.load(f)     
     
     
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


audio_features_dataset_testing = audio_features_dataset_testing_10_locus_1

audio_features_dataset_testing = flat_list(audio_features_dataset_testing)
audio_features_dataset_testing = np.expand_dims(np.array(audio_features_dataset_testing), axis=1).tolist()
    
distance_dataset_testing = distance_dataset_testing_10_locus_1
distance_dataset_testing = flat_list(distance_dataset_testing)
distance_dataset_testing = np.expand_dims(np.array(distance_dataset_testing), axis=1).tolist() 

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
    features = df_audio_features[i][['mic1_coh' ,"mic2_coh",  'mic1_abs_l', "mic1_abs_r", 'mic2_abs_l', "mic2_abs_r"]]  # df_audio_features[0] #df_audio_features[0][['mic1_coh',"mic2_coh"]] 
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


############# extract audio data and distance data for testing #############

audio_features_dataset_testing_list = []

for i in range(len(audio_features_dataset_testing)):
    audio_features_dataset_testing_list.append(audio_features_dataset_testing[i])
    
    
dataset_list = []

for i in range(len(audio_features_dataset_testing_list)):
    dataset_list.append(audio_features_dataset_testing_list[i])


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


    
####### test data ###################
lables_flat_list_test = []
for i in range(len( distance_dataset_testing)):
    labels = flat_list(get_item(distance_dataset_testing[i],"mic" ))
    lables_flat_list_test.append(labels)
    

mic1_distance_test = flat_list(get_item(distance_dataset_testing[0], 'mic1-distance'))
mic2_distance_test = flat_list(get_item(distance_dataset_testing[0], 'mic2-distance'))

norm_dist_diff_test = []
for i in range(len(mic1_distance_test)):
    n_d = np.subtract(mic1_distance_test[i], mic2_distance_test[i]) / math.dist(standardRoom_mic_locs[0], standardRoom_mic_locs[1])
    norm_dist_diff_test.append(n_d)  

   
test_X_list = []

for i in range(len(df_audio_features)):
    features = df_audio_features[i][['mic1_coh' ,"mic2_coh",  'mic1_abs_l', "mic1_abs_r", 'mic2_abs_l', "mic2_abs_r"]]  # df_audio_features[0] #df_audio_features[0][['mic1_coh',"mic2_coh"]] 
    features = scaler.fit_transform(features)
    test_X_list.append(features)
    #train_X_list.append(df_audio_features[i][['mic1_coh' ,"mic2_coh"]])

df_for_testing_X_list = []

for i in range(len(test_X_list)):
    df_for_testing_X_list.append(pd.DataFrame(test_X_list[i]))

test_Y_list = []

for i in range(len(lables_flat_list_test)):
    labels = pd.DataFrame(lables_flat_list_test[i], columns=['mic'])
    labels = scaler.fit_transform(labels)
    test_Y_list.append(labels)

df_for_testing_Y_list = []

for i in range(len(test_Y_list)):
    df_for_testing_Y_list.append(pd.DataFrame(test_Y_list[i]))

X_train = np.concatenate([df.values[np.newaxis, :] for df in df_for_training_X_list], axis=0)
x_test = np.concatenate([df.values[np.newaxis, :] for df in df_for_testing_X_list], axis=0)
#x_test = x_test.reshape(1, x_test.shape[0],x_test.shape[1])

y_train = np.concatenate([df.values[np.newaxis, :] for df in df_for_training_Y_list], axis=0) #df_for_training_Y
y_test =  np.concatenate([df.values[np.newaxis, :] for df in df_for_testing_Y_list], axis=0)
#y_test =  y_test.reshape(1, y_test.shape[0],y_test.shape[1])

# =============================================================================
# # Assuming your train data is stored in X_train with shape (100, 49, 7)
# X_train = np.random.random((100, 49, 7))
# # Assuming your target data is stored in y_train with shape (100, 49, 1)
# y_train = np.random.random((100, 49, 1))
# =============================================================================

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
    
    #tf.print(tf.shape(y_true))
    #tf.print(tf.shape(y_pred))
# =============================================================================
#     tf.print(tf.shape(y_true))
#     tf.print(y_true)
#     tf.print(y_pred)
#     tf.print(tf.shape(y_pred))
# =============================================================================
    return wmsc #K.mean(K.square(y_true - y_pred), axis=-1)

no_timesteps = 49
no_featues = 6
# Create a Sequential model
model = Sequential()
# Add Input layer
model.add(keras.layers.Input(shape=(no_timesteps, no_featues)))

# Add Conv1D layer

model.add(Conv1D(filters=120, kernel_size=45, strides=1, activation='relu', padding='same'))

model.add(Conv1D(filters=124, kernel_size=30, strides=1, activation='relu', padding='same'))

model.add(Conv1D(filters=112, kernel_size=15,  strides=1, activation='relu', padding='same'))
#model.add(MaxPooling1D(2))


#model.add(MaxPooling1D(2))
#model.add(Flatten())

# Add Dense layers
#model.add(Dense(64, activation='relu'))
model.add(Dense(1))


# Compile the model with custom loss function
model.compile(optimizer='adam', loss=custom_loss)
model.summary()

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=1, shuffle=False,  verbose=0)

loss = model.evaluate(x_test, y_test, verbose=0)


predictions = model.predict(x_test)

df_pred = pd.concat([pd.DataFrame(predictions.reshape(-1, 1))], axis=1)

df_pred_new = pd.concat([df_pred, df_pred],  axis=1)

rev_trans_pred = scaler.inverse_transform(df_pred_new)

df_true = pd.concat([pd.DataFrame(y_test.reshape(-1, 1))], axis=1)
df_true_new = pd.concat([df_true, df_true],  axis=1)

rev_trans_true = scaler.inverse_transform(df_true_new)

# Convert probabilities to class predictions
y_pred = rev_trans_pred[:,0].tolist() #np.argmax(rev_trans_pred[:,0])
y_true = rev_trans_true[:,0].tolist() #np.argmax(rev_trans_true[:,0])


mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)


print("Loss: {:.4f}".format(loss))
print("mae: {:.4f}".format(mae))
print("mse: {:.4f}".format(mse))
print("r2: {:.4f}".format(r2))




# Calculate precision, recall, and F1 score
# =============================================================================
# precision = precision_score(y_true, y_pred, average='weighted')
# recall = recall_score(y_true, y_pred, average='weighted')
# f1 = f1_score(y_true, y_pred, average='weighted')
# conusion_metrix = confusion_matrix(y_true, y_pred)
# 
# print("Loss: {:.4f}".format(loss))
# #print("Accuracy: {:.4f}".format(accuracy))
# print("Precision: {:.4f}".format(precision))
# print("Recall: {:.4f}".format(recall))
# print("F1 Score: {:.4f}".format(f1))
# print("conusion_metrix: {:.4f}".format(conusion_metrix))
# =============================================================================

with open ('model_10_200_50.pickle', 'wb' ) as f:
        pickle.dump(model, f) 






