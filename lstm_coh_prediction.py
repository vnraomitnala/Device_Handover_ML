# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 17:27:02 2023

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


with open ('datasetFinal_random_locus_dev-clean1-10-10-50.pickle', 'rb') as f:
     audio_features_dataset_10_locus_1 = pickle.load(f)                

with open ('distanceDataset_random_locus_dev-clean1-10-10-50.pickle', 'rb') as f:
     distance_dataset_10_locus_1 = pickle.load(f) 

     
     
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

audio_features_dataset = audio_features_dataset_10_locus_1

audio_features_dataset = flat_list(audio_features_dataset)
audio_features_dataset = np.expand_dims(np.array(audio_features_dataset), axis=1).tolist()
    
distance_dataset = distance_dataset_10_locus_1
distance_dataset = flat_list(distance_dataset)
distance_dataset = np.expand_dims(np.array(distance_dataset), axis=1).tolist() 

audio_features_dataset_list = []

for i in range(len(audio_features_dataset)):
    audio_features_dataset_list.append(audio_features_dataset[i])
    
    
dataset_list = []

for i in range(len(audio_features_dataset_list)):
    dataset_list.append(audio_features_dataset_list[i])

mic1_coh_flat_list = [[]] * len(dataset_list)
mic1_abs_l_flat_list = [[]] * len(dataset_list)
mic1_abs_r_flat_list = [[]] * len(dataset_list)

mic1_mfcc_l = [[]] * len(dataset_list)
mic1_mfcc_r = [[]] * len(dataset_list)
mic1_mels_l =  [[]] * len(dataset_list)
mic1_mels_r =  [[]] * len(dataset_list) 

mic2_coh_flat_list = [[]] * len(dataset_list)
mic2_abs_l_flat_list = [[]] * len(dataset_list)
mic2_abs_r_flat_list =[[]] * len(dataset_list)
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
    
      
    mic2_coh_flat_list[i] = flat_list(get_item(dataset,"mic2_coh" ))
    mic2_abs_l_flat_list[i] = flat_list(get_item(dataset,"mic2_abs_l" ))
    
    mic2_abs_r_flat_list[i] = flat_list(get_item(dataset,"mic2_abs_r" ))
    
    mic2_mfcc_l[i] = flat_list_n_items(get_items("mic2_mfcc_l", n_mfcc, dataset))
    mic2_mfcc_r[i] = flat_list_n_items(get_items("mic2_mfcc_r", n_mfcc, dataset))
    mic2_mels_l[i] = flat_list_n_items(get_items("mic2_mels_l", n_mels, dataset))
    mic2_mels_r[i] = flat_list_n_items(get_items("mic2_mels_r", n_mels, dataset))    
 
   
    df_audio_features[i] = pd.DataFrame(list(zip(mic1_coh_flat_list[i], mic1_abs_l_flat_list[i], mic1_abs_r_flat_list[i], mic2_coh_flat_list[i], mic2_abs_l_flat_list[i], mic2_abs_r_flat_list[i])),
                   columns =['mic1_coh', 'mic1_abs_l', "mic1_abs_r", "mic2_coh", "mic2_abs_l", "mic2_abs_r"])
    
    df_audio_features[i] = df_audio_features[i].add(update_dict("mic1_mfcc_l", len(mic1_mfcc_l[i]), mic1_mfcc_l[i]) , fill_value=0)
    df_audio_features[i] = df_audio_features[i].add(update_dict("mic1_mfcc_r", len(mic1_mfcc_r[i]), mic1_mfcc_r[i]) , fill_value=0)
    df_audio_features[i] = df_audio_features[i].add(update_dict("mic1_mels_l", len(mic1_mels_l[i]), mic1_mels_l[i]) , fill_value=0)
    df_audio_features[i] = df_audio_features[i].add(update_dict("mic1_mels_r", len(mic1_mels_r[i]), mic1_mels_r[i]) , fill_value=0)
    
    
    df_audio_features[i] = df_audio_features[i].add(update_dict("mic2_mfcc_l", len(mic2_mfcc_l[i]), mic2_mfcc_l[i]) , fill_value=0)
    df_audio_features[i] = df_audio_features[i].add(update_dict("mic2_mfcc_r", len(mic2_mfcc_r[i]), mic2_mfcc_r[i]) , fill_value=0)
    df_audio_features[i] = df_audio_features[i].add(update_dict("mic2_mels_l", len(mic2_mels_l[i]), mic2_mels_l[i]) , fill_value=0)
    df_audio_features[i] = df_audio_features[i].add(update_dict("mic2_mels_r", len(mic2_mels_r[i]), mic2_mels_r[i]) , fill_value=0)


        
mic1_distance_train = flat_list(get_item(distance_dataset[0], 'mic1-distance'))
mic2_distance_train = flat_list(get_item(distance_dataset[0], 'mic2-distance'))

norm_dist_diff_train = []
for i in range(len(mic1_distance_train)):
    n_d = np.subtract(mic1_distance_train[i], mic2_distance_train[i]) / math.dist(standardRoom_mic_locs[0], standardRoom_mic_locs[1])
    norm_dist_diff_train.append(n_d)
    
mic1_distance_test = flat_list(get_item(distance_dataset[0], 'mic1-distance'))
mic2_distance_test = flat_list(get_item(distance_dataset[0], 'mic2-distance'))

norm_dist_diff_test = []
for i in range(len(mic1_distance_train)):
    n_d = np.subtract(mic1_distance_test[i], mic2_distance_test[i]) / math.dist(standardRoom_mic_locs[0], standardRoom_mic_locs[1])
    norm_dist_diff_test.append(n_d)    


train_X_list = []

for i in range(len(df_audio_features)):
    features = df_audio_features[i][['mic1_coh' ,"mic2_coh"]]  # df_audio_features[0] #df_audio_features[0][['mic1_coh',"mic2_coh"]] 
    train_X_list.append(features)
    #train_X_list.append(df_audio_features[i][['mic1_coh' ,"mic2_coh"]])


test_X =   df_audio_features[0][['mic1_coh', "mic2_coh"]] # df_audio_features[1] #
#test_Y = pd.DataFrame(norm_dist_diff_test, columns=['dist-diff'])


#df_for_testing = pd.concat([test_Y.iloc[:101], test_X.iloc[:101]], axis=1)
df_for_testing = pd.concat([test_X], axis=1)



df_for_training = pd.concat(train_X_list) #pd.concat([df_for_training, df_for_testing, df_for_training_2, df_for_training_3 ], axis=0)

scaler = MinMaxScaler()
train_data_scaled = scaler.fit_transform(df_for_training)
test_data_scaled = scaler.fit_transform(df_for_testing)

features = train_data_scaled
#target = train_data_scaled[:, 0]

x_train = train_data_scaled
x_test = test_data_scaled

y_train = train_data_scaled[:,:2]
y_test = test_data_scaled[:,:2] # one target   # mic1_coh

#y_train_2 = train_data_scaled[:, 1]
#y_test_2 = test_data_scaled[:, 1]  # one target   # mic2_coh

# =============================================================================
# y_train = train_data_scaled
# y_test = test_data_scaled  # two targets
# =============================================================================

#x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=123, shuffle=False)

win_length = 3
batch_size = 1
num_features = features.shape[1]


train_generator = TimeseriesGenerator(x_train, y_train, length=win_length, sampling_rate=1, batch_size=batch_size)
test_generator =  TimeseriesGenerator(x_test, y_test, length=win_length, sampling_rate=1, batch_size=batch_size)
#test_generator_2 =  TimeseriesGenerator(x_test, y_test_2, length=win_length, sampling_rate=1, batch_size=batch_size)

model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(win_length, num_features), return_sequences=True))
model.add(LSTM(32, activation='relu', return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(2))


early_stopping = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience=2, mode= 'min')


def custom_loss(y_true, y_pred): 
    # calculating squared difference between target and predicted values 
    loss = tf.square(y_pred - y_true)  # (batch_size, 2)    
    # multiplying the values with weights along batch dimension
    loss = loss * [0.3, 0.7]          # (batch_size, 2)
                
    # summing both loss values along batch dimension 
    loss = K.sum(loss, axis=1)        # (batch_size,)
    
    return loss


model.compile(loss=tf.keras.losses.MeanSquaredError(),
    optimizer=tf.optimizers.Adam(), 
    metrics=[tf.metrics.MeanAbsoluteError()])


history = model.fit(train_generator, epochs=200, validation_data=(test_generator), shuffle=False, callbacks=[early_stopping])

model.evaluate(test_generator, verbose=0)

predictions = model.predict(test_generator)

#df_pred = pd.concat([pd.DataFrame(predictions), pd.DataFrame(x_test[win_length:])], axis=1)
#df_pred = pd.concat([pd.DataFrame(predictions), pd.DataFrame(x_test[1:,1:][win_length:])], axis=1)
df_pred = pd.concat([pd.DataFrame(predictions)], axis=1)

rev_trans = scaler.inverse_transform(df_pred)

df_final = df_for_testing[predictions.shape[0]*-1:]

df_final['pred-mic1_coh'] = rev_trans[:,0]
df_final['pred-mic2_coh'] = rev_trans[:,1]

df_actual_pred = pd.concat([df_final['mic1_coh'], df_final['pred-mic1_coh'], df_final['mic2_coh'], df_final['pred-mic2_coh']], axis = 1)


# =============================================================================
# history = model.fit(train_generator, epochs=200, validation_data=(test_generator_1), shuffle=False, callbacks=[early_stopping])
# 
# model.evaluate(test_generator_1, verbose=0)
# 
# predictions_1 = model.predict(test_generator_1)
# 
# history = model.fit(train_generator, epochs=200, validation_data=(test_generator_2), shuffle=False, callbacks=[early_stopping])
# 
# model.evaluate(test_generator_2, verbose=0)
# 
# predictions = model.predict(test_generator_2)
# 
# df_pred = pd.concat([pd.DataFrame(predictions_1), pd.DataFrame(x_test[:,1:][win_length:])], axis=1)
# #df_pred = pd.concat([pd.DataFrame(predictions), pd.DataFrame(x_test[win_length:])], axis=1)
# df_pred_2 = pd.concat([pd.DataFrame(predictions), pd.DataFrame(x_test[1:,1:][win_length:])], axis=1)
# 
# rev_trans_1 = scaler.inverse_transform(df_pred)
# rev_trans_2 = scaler.inverse_transform(df_pred_2)
# 
# df_final = df_for_training[predictions_1.shape[0]*-1:]
# df_final_2 = df_for_training[predictions.shape[0]*-1:]
# 
# df_final['pred-mic1_coh'] = rev_trans_1[:,0]
# 
# df_final_2['pred-mic2_coh'] = rev_trans_2[:,0]
# 
# df_actual_pred = pd.concat([df_final['mic1_coh'], df_final['pred-mic1_coh'], df_final_2['mic2_coh'], df_final_2['pred-mic2_coh'] ], axis = 1)
# 
# =============================================================================
print("    ")
print("df_actual_pred  ", df_actual_pred)

# =============================================================================
# #df_final['mic'] = df_final['mic'].values.astype(str)
# 
# df_final[['mic', 'pred-mic']].plot()
# =============================================================================


list1 = df_actual_pred['mic1_coh'].tolist()
list2 = df_actual_pred['pred-mic1_coh'].tolist()

list3 = df_actual_pred['mic2_coh'].tolist()
list4 = df_actual_pred['pred-mic2_coh'].tolist()

labels_without_pred = {"mic": []}
labels_with_pred = {"mic": []}

for i in range(len(list1)):

    if list1[i] > list3[i]:
        labels_without_pred['mic'].append('01')
    else:
        labels_without_pred['mic'].append('02')
        
    if list2[i] > list4[i]:
        labels_with_pred['mic'].append('01')
    else:
        labels_with_pred['mic'].append('02')


df = pd.DataFrame(df_final)


fig, axs = plt.subplots(
        nrows=5, ncols=1, sharex=True, sharey=False,
        gridspec_kw={'height_ratios':[2,2,2, 5,5]}
        )
        

ground_truth_lables_train = get_item(distance_dataset[0],"mic" )
#y = np.append(y, np.array(travel_time) + travel_time[-1])

index = win_length

axs[0].plot(ground_truth_lables_train[0][index:], c ="blue", label='GTruth')
axs[0].legend()
axs[0].grid()

axs[1].plot(labels_with_pred['mic'], c ="red", label= r'$\hat\rho_{1} < \hat\rho_{2}$')
axs[1].legend()
axs[1].grid()

axs[2].plot(labels_without_pred['mic'], c ="orange", label= r'$\rho_{1} < \rho_{2}$')
axs[2].legend()
axs[2].grid()

axs[3].plot(df_final['pred-mic1_coh'], c ="blue", label='D1_coh_S')
axs[3].plot(df_final['pred-mic2_coh'], c ="red", label='D2_coh_S')
axs[3].legend(loc="upper right")
axs[3].grid()

axs[4].plot(df_final['mic1_coh'], c ="blue", label='D1_coh_NS')
axs[4].plot(df_final['mic2_coh'], c ="red", label='D2_coh_NS')
axs[4].legend(loc="upper right")
axs[4].grid()



#ground_truth_lables_train = ground_truth_lables_train_1[0] + ground_truth_lables_train_1[1]



plt.savefig('lstm-randum-locus-smoothing-sig4.pdf')

plt.show()


# plot ground truth vs predicted
# =============================================================================
# duration =  10.2
# interval = 0.1
# 
# steps = duration/interval
# steps =  int(steps)
# start = 0.5
# end = 6
# velocity = 0.5
# 
# yy = 1
# 
# zz = 1.4
# delta = (end-start)/steps
# 
# travel_time1 = stats.uniform(0.0, duration).rvs(steps-win_length).tolist()
# travel_time1.sort()
# 
# travel_time = stats.uniform(0.0, duration).rvs(steps).tolist()
# travel_time.sort()
# 
# df = pd.DataFrame(df_final)
# 
# fig, axs = plt.subplots(
#         nrows=3, ncols=1, sharex=True, sharey=False,
#         gridspec_kw={'height_ratios':[2,2,2]}
#         
#         )
# 
# y = np.array(travel_time1)
# 
# list1 = df_final['dist-diff'].tolist()
# list2 = df_final['pred-dist-diff'].tolist()
# 
# labels_without_pred = {"mic": []}
# labels_with_pred = {"mic": []}
# 
# for i in range(len(list1)):
# 
#     if list1[i] < 0:
#         labels_without_pred['mic'].append('01')
#     else:
#         labels_without_pred['mic'].append('02')
#         
#     if list2[i] < 0:
#         labels_with_pred['mic'].append('01')
#     else:
#         labels_with_pred['mic'].append('02')
#         
#         
# axs[0].plot(y, (labels_without_pred['mic']), c ="orange", label='NP')
# axs[0].grid()
# axs[0].legend()
# #plt.ylabel('$y$ (transition)')
# 
# axs[1].plot(y, (labels_with_pred['mic']), c ="red", label='S')
# axs[1].grid()
# axs[1].legend()
# axs[1].set_ylabel('$y$ (transition)')
# =============================================================================






