# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 18:06:13 2023

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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, precision_score, recall_score, f1_score



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


with open ('distanceDataset_random_locus_dev-clean1-50-200-50-1.5.pickle', 'rb') as f:
     distance_dataset_10_locus_1 = pickle.load(f)

with open ('y_pred_cl.pickle', 'rb') as f:
     y_pred_cl_t = pickle.load(f)  

with open ('wtd_mse_t_cl.pickle', 'rb') as f:
     mse_t_cl = pickle.load(f)  

with open ('wtd_mse_t_2_cl.pickle', 'rb') as f:
     wtd_mse_t_cl = pickle.load(f) 
     
with open ('wtd_mse_t_mse.pickle', 'rb') as f:
     mse_t_mse = pickle.load(f)  
  
with open ('wtd_mse_t_2_mse.pickle', 'rb') as f:
     wtd_mse_t_mse = pickle.load(f)  

with open ('wtd_mse_exp.pickle', 'rb') as f:
     wtd_mse_t_exp = pickle.load(f)
     
     
with open ('y_true.pickle', 'rb') as f:
     y_true_t = pickle.load(f) 
     
with open ('y_pred_mse.pickle', 'rb') as f:
     y_pred_mse_t = pickle.load(f)      

with open ('y_pred_exp.pickle', 'rb') as f:
     y_pred_exp_t = pickle.load(f)  
     
y_pred_cl = []
y_pred_mse = []
y_pred_exp = []
y_true = []

for i in range(len(y_pred_cl_t[0:200])):
    t = round(y_pred_cl_t[i], 2)
    y_pred_cl.append(t)
    
for i in range(len(y_pred_mse_t[0:200])):
    t = round(y_pred_mse_t[i], 2)
    y_pred_mse.append(t)

for i in range(len(y_pred_exp_t[0:200])):
    t = round(y_pred_exp_t[i], 2)
    y_pred_exp.append(t)
    
for i in range(len(y_true_t[0:200])):
    t = round(y_true_t[i], 2)
    y_true.append(t)    
    
y_pred_cl_list = []
y_pred_mse_list = []
y_pred_exp_list = []

y_true_list = []


for i in range(len(y_pred_cl)):
    if y_pred_cl[i] <= 1.5:
        y_pred_cl_list.append("1")
    else:
        y_pred_cl_list.append("2")

for i in range(len(y_pred_mse)):
    if y_pred_mse[i] <= 1.5:
        y_pred_mse_list.append("1")
    else:
        y_pred_mse_list.append("2")

for i in range(len(y_pred_exp)):
    if y_pred_exp[i] == 1:
        y_pred_exp_list.append("1")
    else:
        y_pred_exp_list.append("2")    
        
for i in range(len(y_true)):
    if y_true[i] == 1.0:
        y_true_list.append("1")
    else:
        y_true_list.append("2")

        
# =============================================================================
# mae = mean_absolute_error(y_true, y_pred_cl)
# mse = mean_squared_error(y_true, y_pred_cl)
# r2 = r2_score(y_true, y_pred_cl)
# 
# 
# #print("Loss: {:.4f}".format(loss))
# print("mae: {:.4f}".format(mae))
# print("mse: {:.4f}".format(mse))
# print("r2: {:.4f}".format(r2))
# =============================================================================



ideal = [int(x) for x in y_true_list]
pred_cl = [int(x) for x in y_pred_cl_list]
pred_exp = [int(x) for x in y_pred_exp_list]
pred_mse = [int(x) for x in y_pred_mse_list]

ideal_count =0
pred_count_exp =0
pred_count_cl =0
pred_count_mse =0

for i in range(len(ideal)):    
    if i == len(ideal) -1:
        break 
     
    if ideal[i] != ideal[i+1]:
        ideal_count = ideal_count +1

for i in range(len(pred_exp)):    
    if i == len(pred_exp) -1:
        break 
     
    if pred_exp[i] != pred_exp[i+1]:
        pred_count_exp = pred_count_exp +1

for i in range(len(pred_cl)):    
    if i == len(pred_cl) -1:
        break 
     
    if pred_cl[i] != pred_cl[i+1]:
        pred_count_cl = pred_count_cl +1

for i in range(len(pred_mse)):    
    if i == len(pred_mse) -1:
        break
     
    if pred_mse[i] != pred_mse[i+1]:
        pred_count_mse = pred_count_mse +1
        
et_exp =  pred_count_exp - ideal_count 
et_cl =  pred_count_cl - ideal_count 
et_mse =  pred_count_mse - ideal_count
    
print("ideal count: ", ideal_count)
print("pred count exp: ", pred_count_exp)  
print("pred count cl: ", pred_count_cl)      
print("pred count mse: ", pred_count_mse)   
print( "No of transitions different from Ideal-with_exp: ", et_exp)
print( "No of transitions different from Ideal-with_cl: ", et_cl)
print( "No of transitions different from Ideal-with_mse: ", et_mse)
#plot_model(model, to_file='1d_cnn_model.png', show_shapes=True, show_layer_names=True)

# =============================================================================
# 
# # Calculate precision, recall, and F1 score
# precision = precision_score(y_true_list, y_pred_cl_list, average='weighted')
# recall = recall_score(y_true_list, y_pred_cl_list, average='weighted')
# f1 = f1_score(y_true_list, y_pred_cl_list, average='weighted')
# conusion_metrix = confusion_matrix(y_true_list, y_pred_cl_list)
# 
# #print("Loss: {:.4f}".format(loss))
# #print("Accuracy: {:.4f}".format(accuracy))
# print("Precision: {:.4f}".format(precision))
# print("Recall: {:.4f}".format(recall))
# print("F1 Score: {:.4f}".format(f1))
# #print("conusion_metrix: {:.4f}".format(conusion_metrix))
# =============================================================================


y_pred_exp_t_2 = np.array(y_pred_exp_t).reshape(10000, 49).tolist()

distance_dataset_training = distance_dataset_10_locus_1
distance_dataset_training = flat_list(distance_dataset_training)
distance_dataset_training = np.expand_dims(np.array(distance_dataset_training), axis=1).tolist() 


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
    
wtd_mse_exp = []

# =============================================================================
# pred_exp_2 = np.array(pred_exp).reshape(10000, 49)
# ideal_2 = np.array(ideal).reshape(10000, 49)
# 
# for i in range(len(y_pred_exp_t_2)):    
#     ideal = ideal_2[i]
#     xS = pred_exp_2[i]
# 
#     sq_error = np.subtract(ideal, xS) ** 2
#     mse = sq_error.mean()
# 
#     wtd_mse = []                                            
#     for x, y in zip(sq_error, norm_dist_diff_train):                                                                       
#         wtd_mse.append(x * y/np.sum(norm_dist_diff_train))
#     
#     wtd_mse_exp.append(sum(wtd_mse))
#     
# wtd_mse_exp_mean = np.mean(wtd_mse_exp)    
# 
# print("wtd_mse_cl: ", np.mean(wtd_mse_t_cl))
# print("wtd_mse_mse: ", np.mean(wtd_mse_t_mse))
# print("wtd_mse_exp: ", np.mean(wtd_mse_exp_mean))
# 
# =============================================================================


fig, axs = plt.subplots(
        nrows=4, ncols=1, sharex=True, sharey=False,
        gridspec_kw={'height_ratios':[5,5,5,5]}
        
        )

axs[0].plot(y_true_list,c ="blue", label='GTruth')
axs[0].legend(loc= 'upper left')
axs[0].grid()
#axs[0].set_ylabel('$y$ (transition)')

axs[1].plot(y_pred_cl_list, c ="green", label='1DCNN-CL')
axs[1].legend(loc= 'upper left')
axs[1].grid()  
axs[1].set_ylabel('   transition                       ')

axs[2].plot(y_pred_mse_list, c ="black", label='1DCNN-MSE')
axs[2].legend(loc= 'upper left')
axs[2].grid()  
axs[2].set_xlabel('$x$ (locus steps)') 
#axs[2].set_ylabel('$y$ (transition)')

axs[3].plot(y_pred_exp_list[1:], c ="red", label='Smoothed-MSC')
axs[3].legend(loc= 'upper left')
axs[3].grid()  
axs[3].set_xlabel(' locus steps ') 
#axs[3].set_ylabel('$y$ (             transition          )')

# =============================================================================
# 
# fig, axs = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=False, figsize=(5,5))
# 
# plot = axs.boxplot( [wtd_mse_t_cl, wtd_mse_t_mse])
# 
# xticklabels=['1DCNN-CL', '1DCNN-MSE']
# axs.set_xticks([1,2])
# axs.set_xticklabels(xticklabels)
# 
# axs.set_xlabel('$x$ (Locus move)')
# axs.set_ylabel('$y$ (WNE)')
# =============================================================================


#plt.savefig('1d_cnn_mse_exp2.pdf')

plt.show()


