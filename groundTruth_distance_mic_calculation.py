# -*- coding: utf-8 -*-
"""
Created on Sat May 28 12:35:50 2022

@author: Vijaya
"""

import math
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from scipy import stats
import numpy as np
from scipy import signal
from scipy.io import wavfile
import os


with open ('locusMovement-randum-locus-with-transitions.pickle', 'rb') as f:
    locusDataset = pickle.load(f)

with open ('travelTimeDataset_random_locus.pickle', 'rb') as f:
    travelTimeDatasetDatasetFinal = pickle.load(f) 
    
def locusThereAndBack(t,start,end,velocity):
    return (start + np.abs(np.abs(signal.sawtooth(np.pi * velocity/(end-start) * t )) - 1) 
            * (end - start))


standard_room_dim = [7,5.5,2.4]  # meters

standardRoom_mic_locs = [
    [1.5,3.5, 0.9], [6.0,3.5,0.9],  # mic 1  # mic 2
]

#standardRoom_source_locs = [ [ 0.5 +i*0.12,1, 1.4] for i in range(50) ]

datasetDistanceFinal = []

directory1 = "audio"

str_list = []

# Recursively iterate over the directory and its subdirectories
for root, dirs, files in os.walk(directory1):
    for file in files:
        file_path = os.path.join(root, file)
        str_list.append(file_path)
        

for s in range(len(str_list)):
    travelTimeDatasetDataset = travelTimeDatasetDatasetFinal[s]
    datasetDistance_final = []
    for l in range(len(locusDataset)):
        #print(velocity_list[l])
        datasetDistance = {"mic": [], "mic1-distance": [], "mic2-distance": []}
        
        travel_time = locusDataset[l]["travelTime"]
        locus = locusDataset[l]["locus"] #standardRoom_source_locs
        
        #velocity = velocity_list[l]
           
        for i in range(len(locus)):
            for t in range(len(travel_time)-1):
                if travel_time[i] == travelTimeDatasetDataset[t]:
                    print("passing")
                    mic1_distance = 0.0
                    mic2_distance = 0.0
                    for j in range(len(standardRoom_mic_locs)):
                        if j == 0:
                           print(standardRoom_mic_locs[j][0])
                           mic1_distance = math.dist(locus[i] , standardRoom_mic_locs[j]) # compare 3D values than 1D
                           datasetDistance["mic1-distance"].append(round(mic1_distance,2))
                        else:
                            print(standardRoom_mic_locs[j][0])                   
                            mic2_distance = math.dist(locus[i], standardRoom_mic_locs[j])
                            datasetDistance["mic2-distance"].append(round(mic2_distance,2))
                            
                    if round(mic1_distance,2) < round(mic2_distance,2):
                      datasetDistance["mic"].append('1')
                    else:
                      datasetDistance["mic"].append('2')  
# =============================================================================
#         my_list = datasetDistance["mic"]              
#         all_different = all(x != my_list[0] for x in my_list)  
#         if all_different is True:
#             datasetDistance_final.append(datasetDistance)  
# =============================================================================
        datasetDistance_final.append(datasetDistance)
    datasetDistanceFinal.append(datasetDistance_final)     
    
    
print(len(datasetDistanceFinal))


with open ('distanceDataset_random_move.pickle', 'wb' ) as f:
    pickle.dump(datasetDistanceFinal, f) 


