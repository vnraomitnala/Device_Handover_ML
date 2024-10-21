import numpy as np
from scipy.io import wavfile
from scipy import signal
import scipy.io.wavfile
from scipy import stats
import matplotlib.pyplot as plt
import pickle


with open ('locus_pos_list_20000.pickle', 'rb') as f:
     locus_ds = pickle.load(f)
     
locusDataset = {"locus": [], "travelTime": []}


def locusThereAndBack(t,start,end,velocity):
    return (start + np.abs(np.abs(signal.sawtooth(np.pi * velocity/(end-start) * t )) - 1) 
            * (end - start))

rt60_tgt = 0.8 # seconds

duration =  13
interval = 0.2

num_locus_pos = 50

steps = duration/interval
steps = 50# int(steps)
start = 0.5
end = 6
warm_up_steps = 10

yy = 1
zz = 1.4
velocity = 1.0

travel_time = stats.uniform(0.0, duration).rvs(steps).tolist()
travel_time.sort()
travel_time = [ 0.2 +i*interval for i in range(steps)]

warm_up_locus = [ [0.5,yy, zz] for i in range(warm_up_steps) ]

locus = locusThereAndBack(np.array(travel_time), start, end, velocity)
#locus.sort()
locus = list(locus)
#locus = [ [5.0,yy, zz] for i in range(len(locus)) ] # get the locus in length X width X height format
locus = [ [locus[i],yy, zz] for i in range(len(locus)) ]

locus = warm_up_locus + locus
travel_time =  [ 0.02*i for i in range(warm_up_steps)] + travel_time

locus = locus_ds

travelTime = []
for i in range(400):
    travel_time = [ 0 +i*interval for i in range(steps)] #stats.uniform(0.0, duration).rvs(num_locus_pos).tolist()
    travel_time.sort()
    travelTime.append(travel_time)
#travel_time = [ 0 +i*interval for i in range(steps)]

locusDataset["locus"] = locus
locusDataset["travelTime"] = travelTime

with open ('locusMovement-randum-locus_20000.pickle', 'wb' ) as f:
    pickle.dump(locusDataset, f)
     
    
# =============================================================================
# plt.plot(locus)
# 
# plt.xlabel('$x$ (travel time)')
# plt.ylabel('$y$ (transition)')
# 
# plt.grid()
# plt.legend()
# 
# plt.show()
# =============================================================================





