# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 12:07:55 2023

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
from keras.utils.vis_utils import plot_model
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


     
n_mfcc = 13
n_mels = 13

standardRoom_mic_locs = [
    [1.0,3.5, 0.9], [5.5,3.5,0.9],  # mic 1  # mic 2
]

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



with open ('datasetFinal_random_locus_dev-clean1.pickle', 'rb') as f:
    audio_features_dataset_10_locus_1 = pickle.load(f)               

with open ('distanceDataset_random_locus_dev-clean1-10-10-50.pickle', 'rb') as f:
     distance_dataset_10_locus_1 = pickle.load(f)  

# =============================================================================
# with open ('distanceDataset_random_locus_dev-clean1-10-10-50.pickle', 'rb') as f:
#      distance_dataset_10_locus_1 = pickle.load(f)
# =============================================================================
     
audio_features_dataset = audio_features_dataset_10_locus_1

audio_features_dataset = flat_list(audio_features_dataset)
audio_features_dataset = np.expand_dims(np.array(audio_features_dataset), axis=1).tolist()
    
distance_dataset = distance_dataset_10_locus_1
distance_dataset = flat_list(distance_dataset)
distance_dataset = np.expand_dims(np.array(distance_dataset), axis=1).tolist() 

    

####### test data ###################
test_signal_index = 0

ground_truth_lables_test = get_item(distance_dataset[test_signal_index],"mic" )
lables_flat_list_test = flat_list(ground_truth_lables_test)

#lables_flat_list_test.append('2')
#lables_flat_list_test.append('1')
   
mic1_coh_test = (audio_features_dataset_10_locus_1[0][0]['mic1_coh'])
mic2_coh_test = (audio_features_dataset_10_locus_1[0][0]['mic2_coh'])

mic1_coh_test = mic1_coh_test[0:]
mic2_coh_test = mic2_coh_test[0:]

mic1_coh_test_smooth = []
mic2_coh_test_smooth = []
mic1_coh_test_tmp = []
mic2_coh_test_tmp = []


for i in range(len(mic1_coh_test)):   
    mic1_coh_test_tmp.append(mic1_coh_test[i])
    m1_s = double_exponential_smoothing(mic1_coh_test_tmp, 0.05, 0.036) 
    mic1_coh_test_smooth.append(m1_s)

for i in range(len(mic2_coh_test)):  
    mic2_coh_test_tmp.append(mic2_coh_test[i])
    m2_s = double_exponential_smoothing(mic2_coh_test_tmp, 0.05,0.036) 
    mic2_coh_test_smooth.append(m2_s)
        

fig, axs = plt.subplots(
        nrows=5, ncols=1, sharex=True, sharey=False,
        gridspec_kw={'height_ratios':[2,2, 2, 5, 5]}
        
        )


warmup_steps = 5

axs[0].plot((lables_flat_list_test[warmup_steps:]), c ="blue", label='GTruth')
axs[0].grid()
axs[0].legend(loc= 'upper right')

        
pred_listMic = []

for i in range(len(mic1_coh_test_smooth)):
    if mic1_coh_test_smooth[i] >= mic2_coh_test_smooth[i]:
        pred_listMic.append('1')
    else:
        pred_listMic.append('2')

axs[1].plot(pred_listMic[warmup_steps:], c ="red", label= r'$\max({\hat\rho_1},{\hat\rho_2})$')
axs[1].grid()
axs[1].legend(loc= 'upper right')
#axs[1].set_ylabel('$y$ (transition)')


listMic = []

for i in range(len(mic1_coh_test)):
    if mic1_coh_test[i] >= mic2_coh_test[i]:
        listMic.append('1')
    else:
        listMic.append('2')
        
axs[2].plot(listMic[warmup_steps:], c ="orange", label= r'$\max({\rho_1},{\rho_2})$')
axs[2].grid()
axs[2].legend(loc= 'upper right')
axs[1].set_ylabel('$y$ (transition)')        

axs[3].plot(mic1_coh_test_smooth[warmup_steps:], c ="blue", label=r'$\hat\rho_{1}$')
axs[3].plot(mic2_coh_test_smooth[warmup_steps:], c ="red", label=r'$\hat\rho_{2}$')
axs[3].grid()
axs[3].legend(loc= 'upper right')
axs[3].set_ylabel(r'$\hat\rho$')  

axs[4].plot(mic1_coh_test[warmup_steps:], c ="blue", label=r'$\rho_{1}$')
axs[4].plot(mic2_coh_test[warmup_steps:], c ="red", label=r'$\rho_{2}$')
axs[4].grid()
axs[4].legend(loc= 'upper right')
axs[4].set_ylabel(r'$\rho$')  
axs[4].set_xlabel('$x$ (locus steps)')

plt.savefig('transition_double_exp_random_move_0.8.pdf')

ideal = [int(x) for x in lables_flat_list_test]
xS = [int(x) for x in pred_listMic]
xNS = [int(x) for x in listMic]


sq_error_without_smooth = np.subtract(ideal, xNS) ** 2
mse_without_smooth = sq_error_without_smooth.mean()
print("mse_without_smooth: ", round(mse_without_smooth,2))

wtd_mse_without_smooth = []

xx = np.abs(np.subtract(flat_list(get_item(distance_dataset[test_signal_index],"mic1-distance" )), flat_list(get_item(distance_dataset[test_signal_index],"mic2-distance" ))))

for x, y in zip(sq_error_without_smooth, xx):
    wtd_mse_without_smooth.append(x * y/np.sum(xx))

wtd_mse_without_smooth_mean = sum(wtd_mse_without_smooth)

print('wtd_mse_without_smooth: ', round (wtd_mse_without_smooth_mean,2))


sq_error_with_smooth = np.subtract(ideal, xS) ** 2
mse_with_smooth = sq_error_with_smooth.mean()

print("mse_with_smooth: ", (mse_with_smooth))

wtd_mse_with_smooth = []



for x, y in zip(sq_error_with_smooth, xx):
    wtd_mse_with_smooth.append(x * y/np.sum(xx))

wtd_mse_with_smooth_mean = sum(wtd_mse_with_smooth)
print('wtd_mse_with_smooth: ', round(wtd_mse_with_smooth_mean,2))

def differences(a, b):
    if len(a) != len(b):
        raise ValueError("Lists of different length.")
    return sum(i != j for i, j in zip(a, b))

diff_I_xNS = differences(ideal, xNS)
diff_I_xS =  differences(ideal, xS) 

if diff_I_xNS > 0:
    diff_I_xNS = diff_I_xNS-1
    
if diff_I_xS > 0:
    diff_I_xS = diff_I_xS -1
    
without_smooth_count =0
with_smooth_count =0

for i in range(len(xS)):    
    if i == len(xS) -1:
        break 
     
    if xS[i] != xS[i+1]:
        with_smooth_count = with_smooth_count +1
        
for i in range(len(xNS)):    
    if i == len(xNS) -1:
        break     
    
    if xNS[i] != xNS[i+1]:
        without_smooth_count = without_smooth_count +1        
     
y_true = lables_flat_list_test
y_pred = pred_listMic

y_true_float = [float(x) for x in y_true]
y_pred_float = [float(x) for x in y_pred]

r2 = r2_score(y_true_float, y_pred_float)    
       
print( "No of transitions different from Ideal-without_smooth: ", without_smooth_count)
print( "No of transitions different from Ideal-with_smooth: ", with_smooth_count)
print("r2 ", r2)

precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')
#conusion_metrix = confusion_matrix(y_true_list, y_pred_list)

#print("Loss: {:.4f}".format(loss))
#print("Accuracy: {:.4f}".format(accuracy))
print("Precision: {:.4f}".format(precision))
print("Recall: {:.4f}".format(recall))
print("F1 Score: {:.4f}".format(f1))


#plot_model(model, show_shapes=True)


plt.show()





