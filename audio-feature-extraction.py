import numpy as np
import matplotlib.pyplot as plt
import pyroomacoustics as pra
from scipy.io import wavfile
import librosa, librosa.display
import matplotlib.pyplot as plt
import os
import pickle
from scipy import signal
from shutil import rmtree
from scipy import stats
import math
from numpy import pi, polymul
from scipy.signal import lfilter
import soundfile as sf
from scipy.signal import bilinear
import colorednoise as cn
#from pyroomacoustics.directivities import (
#    DirectivityPattern,
#    DirectionVector,
#    CardioidFamily,
#)
from pyroomacoustics.doa import spher2cart
from scipy.io import wavfile
import IPython

rt60_tgt = 0.8 # seconds

#str1 = "examples_samples_guitar_16k_latest3.wav"   

directory1 = "audio"

str_list = []

noise_file = "99.wav"
noise_attendB = -12 # in dB
noise_atten = 10 ** (noise_attendB/20) # in absolute scale

# Recursively iterate over the directory and its subdirectories
for root, dirs, files in os.walk(directory1):
    for file in files:
        
        file_path = os.path.join(root, file)
        str_list.append(file_path)
with open ('locusMovement-randum-locus-with-transitions.pickle', 'rb') as f:
    locusDataset = pickle.load(f)

    
########## MJREED WARNING CHANGED FOR TESTING START ######
#str_list = [str1] # just for testing purpose
#str_list = [str_list[0],str_list[1]] # just for testing purpose
str_list = [str_list[0]] # just for testing purpose

locusDataset=[locusDataset[0],locusDataset[1]] ## only for testing
########## MJREED WARNING CHANGED FOR TESTING END ######

rt60_tgt = 0.8 # seconds

duration =  13.0
#duration =  audio_length
interval =0.2 # 5 # 100 ms 

steps = duration/interval
steps =  int(steps)
start = 0.5
end = 6

yy = 1
zz = 1.4
#velocity = 1

noise_position = [3.5,2.75,1.4] # in the middle of the room for starters

standard_room_dim = [7,5.5,2.4]  # meters

standardRoom_mic_locs = [
    [1.5,3.5, 0.9], [6.0,3.5,0.9],  # mic 1  # mic 2
]

num_mic_arrays = 2

# We invert Sabine's formula to obtain the parameters for the ISM simulator
e_absorption, max_order = pra.inverse_sabine(rt60_tgt, standard_room_dim)

extraTime = 40786
# use 20 ms overlap
overlap = 320

datasetFinal = []
travelTimeDataset = []


    
blockLength =  2**14

n_mfcc = 13
n_mels = 13

def locusThereAndBack(t,start,end,velocity):
    return (start + np.abs(np.abs(signal.sawtooth(np.pi * velocity/(end-start) * t )) - 1) 
            * (end - start))

def A_weighting(fs):
    """Design of an A-weighting filter.
    b, a = A_weighting(fs) designs a digital A-weighting filter for
    sampling frequency `fs`. Usage: y = scipy.signal.lfilter(b, a, x).
    Warning: `fs` should normally be higher than 20 kHz. For example,
    fs = 48000 yields a class 1-compliant filter.
    References:
       [1] IEC/CD 1672: Electroacoustics-Sound Level Meters, Nov. 1996.
    """
    # Definition of analog A-weighting filter according to IEC/CD 1672.
    f1 = 20.598997
    f2 = 107.65265
    f3 = 737.86223
    f4 = 12194.217
    A1000 = 1.9997

    NUMs = [(2*pi * f4)**2 * (10**(A1000/20)), 0, 0, 0, 0]
    DENs = polymul([1, 4*pi * f4, (2*pi * f4)**2],
                   [1, 4*pi * f1, (2*pi * f1)**2])
    DENs = polymul(polymul(DENs, [1, 2*pi * f3]),
                                 [1, 2*pi * f2])

    # Use the bilinear transformation to get the digital filter.
    # (Octave, MATLAB, and PyLab disagree about Fs vs 1/Fs)
    return bilinear(NUMs, DENs, fs)

def rms_flat(a):  # from matplotlib.mlab
    """
    Return the root mean square of all the elements of *a*, flattened out.
    """
    return np.sqrt(np.mean(np.absolute(a)**2))

  
def MSCMeanOnData(indata_0, indata_1):       
   
    b, a = A_weighting(fs)
    
    yL = indata_0
    yR = indata_1
    
    yL = lfilter(b, a, yL)       
    yR = lfilter(b, a, yR)    
   
    f, Cxy = signal.coherence(yR, yL, fs, nperseg=1024)   
    cxy = np.mean(Cxy)    
    
    return cxy

def ABSMeanOnData(indata):       
  
    b, a = A_weighting(fs)    
    y = lfilter(b, a, indata)

    fft = np.fft.fft(y)
    magnitude = np.abs(fft)
    
    return np.mean(magnitude)

def MfccOncData(indata, num_mfcc):           
  
    b, a = A_weighting(fs)    
    y = lfilter(b, a, indata)
    
    return (librosa.feature.mfcc(y=y, sr=samplerate, n_fft=len(y), hop_length=len(y)+1, n_mfcc=num_mfcc))[:,0]

def MelSOncData(indata, num_mels):       
  
    b, a = A_weighting(fs)    
    y = lfilter(b, a, indata)
   
    return (librosa.feature.melspectrogram(y=y, sr=samplerate, n_fft=len(y), hop_length=len(y)+1, n_mels=num_mels))[:,0]

dict_mfcc_keys = ["mic1_mfcc_l", "mic1_mfcc_r", "mic2_mfcc_l", "mic2_mfcc_r", ]
dict_mels_keys = ["mic1_mels_l", "mic1_mels_r", "mic2_mels_l", "mic2_mels_r"]

def update_dict(dict_key, count):
    dict_key_list = []
    for i in range(count):
        ss =  f'{i}'
        dict_key_updated = dict_key + "_" + ss
        dict_key_list.append(dict_key_updated)
 
    return {x: [] for x in dict_key_list}



def update_dataset(dict_key, count, indata, index2, outdata): 
    i =0    
    while i < count:
        for x in range(index2):  
            ss =  f'{i}'
            dict_key_updated = dict_key + "_" + ss
            outdata[dict_key_updated].append(indata[x][i]) 
        i = i + 1
 
def get_mfccs_mels(count, indata):                           
    mfccs = [[]] * count
    for n in range(count):
        mfccs[n] = indata[n]    
    return mfccs    


fs=16000
samples = 10 * fs
# use this as the audio

y = cn.powerlaw_psd_gaussian(1, samples,fmin=100/fs)

mic_rotation = 0
colatitude = 0
#pattern = DirectivityPattern.OMNI
#orientation = DirectionVector(azimuth=mic_rotation, colatitude=colatitude, degrees=True)
#directivity = CardioidFamily(orientation=orientation, pattern_enum=pattern)

rooms = []
for i in range(len(standardRoom_mic_locs)):
    room = pra.ShoeBox(standard_room_dim, fs=fs, materials=pra.Material(e_absorption), max_order=max_order)
    
    phi0_radient = 90 * math.pi/180
    
    R = pra.circular_2D_array(standardRoom_mic_locs[i][:2], 2, phi0_radient, 0.05)
    R = np.concatenate((R, np.ones((1, 2)) * standardRoom_mic_locs[i][2]), axis=0)
    
    print(R)
    
    mics = pra.MicrophoneArray(R, 16000, directivity=None)
    
    tmp  = room.add_microphone_array(mics)
    rooms.append(tmp) 
    
#standardRoom_source_locs = [ [ 0.5 +i*0.06,1, 1.4] for i in range(100) ]

fs, noise  = wavfile.read(noise_file)
   
for s in range(len(str_list)): 
    print(s)
    dataset_Final = []
    travelTime_Dataset = []
    audio_sig = str_list[s]
    
    fs, audio = wavfile.read(audio_sig)
    #audio = y # this is for pink noise signal
    
    
    audio_length = len(audio)/fs
    
    start_window =  int(0.1 * fs)
    # these are not used
    audio1= np.hanning(start_window * 2)[0:start_window] * audio[0:start_window]   
    audio2 = np.hanning(start_window * 2)[-start_window:] * audio[-start_window:]   
    
# =============================================================================
#     while len(audio) < (duration * fs):
#        audio = np.append(audio,audio1, audio2 )
# =============================================================================
       
    index = int(interval * fs) 
           
    for l in range(len(locusDataset)):

        rmtree('tmp/short4/' , ignore_errors=True)
        
        locus =  locusDataset[l]["locus"] #standardRoom_source_locs #locusDataset["locus"][0]
        travel_time = locusDataset[l]["travelTime"]
        
        k = 0
        t=0
        os.makedirs("tmp/short4", exist_ok = True) 
        
        final_mic1 = []
        final_mic1_l = []
        final_mic1_r = []
        final_mic2 = []
        final_mic2_l = []
        final_mic2_r = []
        counter = 0
        
        previous_sig = []
        previous_start1 =0
        previous_start2 =0
        maxSample=0
    
        
        for j in range (len(locus)):
    
            for i in range(len(standardRoom_mic_locs)):
                room = rooms[i]
                room.sources = []
                
                if j+1 >= len(locus):
                   t = j
                else:
                    t = j+1
                    
                minSample = np.floor(travel_time[j] *fs).astype(np.int32)
                
                if travel_time[t] * fs >= len(audio):
                    maxSample = len(audio)        
                else:
                    maxSample = np.floor(travel_time[t] * fs).astype(np.int32)
                noise_sig = noise[minSample  : maxSample]
                sig = audio[minSample  : maxSample]    
        
                if j > 0:
                    overlapped_sig = previous_sig[-overlap:]
                    overlapped_noise_sig=previous_noise_sig[-overlap:]
                    sig = np.append(overlapped_sig, sig)
                    noise_sig = np.append(overlapped_noise_sig,noise_sig)
        
                previous_sig = sig
                previous_noise_sig = noise_sig

                sig = np.hanning(len(sig))*sig
                sig = librosa.util.fix_length(sig,size=(len(sig) + extraTime))
                
                noise_sig = np.hanning(len(noise_sig))*noise_sig
                noise_sig = librosa.util.fix_length(noise_sig,size=(len(noise_sig) + extraTime))
        
                room.add_source(locus[j], signal=sig)
                room.add_source(noise_position,signal=noise_sig)
                room.simulate()
                
                #room.compute_rir() 
                
               
                if i == 0: # mic1
                    
                    tmp_final_l = final_mic1_l[previous_start1: len(final_mic1_l)]
                    tmp_final_r = final_mic1_r[previous_start1: len(final_mic1_r)]
                    
                    tmp_final_l_tmp3 = tmp_final_l
                    
                    output = room.mic_array.signals
                    output = librosa.util.fix_length(output,size=(len(sig)))
                    
                    output_l = output[0] 
                    output_r = output[1] 
                    
                      
                    if len(tmp_final_l) == 0:
                        tmp_final_l = output_l             
                    
                    if len(tmp_final_r) == 0:
                        tmp_final_r = output_r      
                        
                    tmp_final_l_tmp = tmp_final_l
                    
                    tmp_final_l = librosa.util.fix_length(tmp_final_l,size=(len(output_l)))
                    tmp_final_r = librosa.util.fix_length(tmp_final_r,size=(len(output_r)))  
                    
                    tmp_final_l_tmp2 = tmp_final_l
                   
                    tmp_final_l = list(np.array(tmp_final_l) + np.array(output_l))
                    tmp_final_r = list(np.array(tmp_final_r) + np.array(output_r))
                  
                    if minSample == 0 or len(final_mic1_l) == 0:
                        final_mic1_l = librosa.util.fix_length(output_l,size=(maxSample)) # output_l
                    else:
                        final_mic1_l = librosa.util.fix_length(final_mic1_l,size=(previous_start1))
                        
                    if minSample == 0 or len(final_mic1_r) == 0:
                        final_mic1_r = librosa.util.fix_length(output_r,size=(maxSample)) #output_r
                    else:
                        final_mic1_r = librosa.util.fix_length(final_mic1_r,size=(previous_start1))               
                    
                    final_mic1_l_tmp = final_mic1_l
                    
                    final_mic1_l = np.append(final_mic1_l,tmp_final_l)     
                    final_mic1_r = np.append(final_mic1_r,tmp_final_r) 
                    
                    tmp_final1 = np.array([tmp_final_l, tmp_final_r])
                    final_mic1 = np.array([final_mic1_l, final_mic1_r])
                    
                    
                    previous_start1 = maxSample - overlap 
                    
                    tmp = np.array([output_l, output_r])
                    
                    data_0 = list(np.array([tmp_final_l])/3)
                    data_1 = list(np.array([tmp_final_r])/3)
                    
                    data_0 = np.hanning(len(data_0[:index])) * data_0[:index]
                    data_1 = np.hanning(len(data_1[:index])) * data_1[:index]
                    
                    
                    abs_l = ABSMeanOnData(data_0)
                    abs_r = ABSMeanOnData(data_1)
                    msc= MSCMeanOnData(data_0, data_1)
                    
                    mic1_abs = (np.array(abs_l) + np.array(abs_r))    
                    
                    if j > 0 and int(mic1_abs) > 50: #and int(mic1_abs) <= 100000 :

                        ss =  f'{k}'
                
                
                        str3 = "tmp/short4/" + ss + "/"
                        os.makedirs(str3, exist_ok = True)                    
                              
                        
                        str4 = str3 + "tabletop01_speech01.wav"

                        wavfile.write( str4, room.fs, (tmp_final1/3).T.astype(np.int16))  
                        
  
                        data, samplerate  = sf.read(str4, dtype='int16')  
                                
                        data_0 = np.hanning(len(data[:index, 0])) * data[:index, 0]
                        data_1 = np.hanning(len(data[:index, 1])) * data[:index, 1] 
                        
                        abs_l = ABSMeanOnData(data_0)
                        abs_r = ABSMeanOnData(data_1)
                        msc= MSCMeanOnData(data_0, data_1)                   
                 
        
                else:  # mic2
                    tmp_final_l = final_mic2_l[previous_start2: len(final_mic2_l)]
                    tmp_final_r = final_mic2_r[previous_start2: len(final_mic2_r)]
                    
                    output = room.mic_array.signals
                    output = librosa.util.fix_length(output,size=(len(sig) + extraTime))
                    
                    output_l = output[0] 
                    output_r = output[1] 
                    
                      
                    if len(tmp_final_l) == 0:
                        tmp_final_l = output_l             
                    
                    if len(tmp_final_r) == 0:
                        tmp_final_r = output_r      
                        
                  
                    tmp_final_l = librosa.util.fix_length(tmp_final_l,size=(len(output_l)))
                    tmp_final_r = librosa.util.fix_length(tmp_final_r,size=(len(output_r)))           
                   
                    tmp_final_l = list(np.array(tmp_final_l) + np.array(output_l))
                    tmp_final_r = list(np.array(tmp_final_r) + np.array(output_r))
                  
                    if minSample == 0 or len(final_mic2_l) == 0:
                        final_mic2_l = librosa.util.fix_length(output_l,size=(maxSample))
                    else:
                        final_mic2_l = librosa.util.fix_length(final_mic2_l,size=(previous_start2))
                        
                    if minSample == 0 or len(final_mic2_r) == 0:
                        final_mic2_r = librosa.util.fix_length(output_r,size=(maxSample))
                    else:
                        final_mic2_r = librosa.util.fix_length(final_mic2_r,size=(previous_start2))               
                    
                    
                    final_mic2_l = np.append(final_mic2_l,tmp_final_l)     
                    final_mic2_r = np.append(final_mic2_r,tmp_final_r) 
                    
                    tmp_final2 = np.array([tmp_final_l, tmp_final_r])           
              
                    final_mic2 = np.array([final_mic2_l, final_mic2_r])
                    
                    previous_start2 = maxSample - overlap 
                    tmp = np.array([output_l, output_r])
                    
                    data_0 = list(np.array([tmp_final_l])/3)
                    data_1 = list(np.array([tmp_final_r])/3)
                    
                    data_0 = np.hanning(len(data_0[:index])) * data_0[:index]
                    data_1 = np.hanning(len(data_1[:index])) * data_1[:index]
                    
                    
                    abs_l = ABSMeanOnData(data_0)
                    abs_r = ABSMeanOnData(data_1)
                    msc= MSCMeanOnData(data_0, data_1)       
                    mic2_abs = (np.array(abs_l) + np.array(abs_r))           
                    
                    if j > 0 and int(mic2_abs) > 50: # and int(mic2_abs) <= 100000 :
                        
                        ss =  f'{k}'
                
                
                        str3 = "tmp/short4/" + ss + "/"
                        os.makedirs(str3, exist_ok = True)                    
                                            
                    
                        str5 = str3 + "tabletop02_speech01.wav"

                        wavfile.write(str5, room.fs, (tmp_final2/3).T.astype(np.int16))  
                       
            if j > 0 and mic1_abs > 50 and mic2_abs > 50: #and mic1_abs <=100000 and mic2_abs <=100000 :
                k= k+1   
                travelTime_Dataset.append(travel_time[t])
            
            wavfile.write("tabletop01-latest-whole-audio.wav", room.fs, (final_mic1).T.astype(np.int16))
            wavfile.write("tabletop02-latest-whole-audio.wav", room.fs, (final_mic2).T.astype(np.int16))
            
        SAMPLE_RATE = 16000
        
        blockLength =  2**14
        freqs = np.arange(0, 1 + 2**14 / 2) * 16000 / 2**14
    
        
        dataset = {"mic1_coh": [] ,"abs_diff": [], "mic1_mfcc_full": [], "mic1_mels_full": [], "mic1_abs_l": [],"mic1_abs_r": [], 
                   "mic2_coh": [], "mic2_mfcc_full": [], "mic2_mels_full": [], "mic2_abs_l": [],"mic2_abs_r": []}
        
        
        dataset.update(update_dict("mic1_mfcc_l", n_mfcc))
        dataset.update(update_dict("mic1_mfcc_r", n_mfcc))
        dataset.update(update_dict("mic1_mels_l", n_mels))
        dataset.update(update_dict("mic1_mels_r", n_mels))
        
        dataset.update(update_dict("mic2_mfcc_l", n_mfcc))
        dataset.update(update_dict("mic2_mfcc_r", n_mfcc))
        dataset.update(update_dict("mic2_mels_l", n_mels))
        dataset.update(update_dict("mic2_mels_r", n_mels))
       
        abs_diff = [] 
        mic1_abs = []
        mic1_coh = []
        mic1_abs_l = []
        mic1_abs_r = []   
        mic1_mfcc = []
        mic1_mels = []
        
        mic2_abs = []
        mic1_mfcc_l = []
        mic1_mfcc_r = []
        mic1_mels_l = []
        mic1_mels_r = []    
        mic2_coh = []
        mic2_abs_l = []
        mic2_abs_r = []   
        mic2_mfcc_l = []
        mic2_mfcc_r = []   
        mic2_mels_l = []
        mic2_mels_r = []  
        mic2_mfcc = []
        mic2_mels = []
        
        directory = 'tmp//short4//'
            
        for root, dirs, files in os.walk(directory):
            
                  dirs.sort(key=int)
                  file_index=0
        
                  coherence1_without_smooth = 0.0
                  coherence2_without_smooth = 0.0
                  mic1_abs_t = 0.0
                  mic2_abs_t = 0.0
                  
                  for file in (files):
                    str = os.path.join(root, file)
                    if (str.find('speech') != -1 ):
                            #print(str)
                            data, samplerate  = sf.read(str, dtype='int16')  
                            
                            data_0 = np.hanning(len(data[:index, 0])) * data[:index, 0]
                            data_1 = np.hanning(len(data[:index, 1])) * data[:index, 1]    
                                                
        
                            if file_index == 0:
                               coherence1_without_smooth = MSCMeanOnData(data_0, data_1) 
                               mic1_coh.append(coherence1_without_smooth)                           
                               
                               abs_l = ABSMeanOnData(data_0)
                               mic1_abs_l.append(abs_l)
                               
                               abs_r = ABSMeanOnData(data_1)
                               mic1_abs_r.append(abs_r)    
                               
                               mic1_abs_t = abs_l + abs_r 

                               mic1_abs.append(mic1_abs_t)
                               
                               # need to test with/without A_Weighing
                                                     
                               mfccs_l = MfccOncData(data_0, n_mfcc)                           
                               mfccs_ll = get_mfccs_mels(n_mfcc, mfccs_l)                               
                               mic1_mfcc_l.append(mfccs_ll)

                               
                               mfccs_r = MfccOncData(data_1, n_mfcc)
                               mfccs_rr = get_mfccs_mels(n_mfcc, mfccs_r)                               
                               mic1_mfcc_r.append(mfccs_rr)
                               
                               mic1_mfcc.append(sum(mfccs_ll) + sum(mfccs_rr))
                               
                               mels_l = MelSOncData(data_0, n_mels)
                               mels_ll = get_mfccs_mels(n_mels, mels_l)                               
                               mic1_mels_l.append(mels_ll)                      
                               
                               mels_r = MelSOncData(data_1, n_mels)
                               mels_rr = get_mfccs_mels(n_mels, mels_r)                               
                               mic1_mels_r.append(mels_rr) 
                               
                               mic1_mels.append(sum(mels_ll) + sum(mels_rr))
                                                     
                               
                            else:                           
                               coherence2_without_smooth = MSCMeanOnData(data_0, data_1)  
                               mic2_coh.append(coherence2_without_smooth)                           
                               
                               abs_l = ABSMeanOnData(data_0)
                               mic2_abs_l.append(abs_l)
                               
                               abs_r = ABSMeanOnData(data_1)
                               mic2_abs_r.append(abs_r)
                               
                               mic2_abs_t = abs_l + abs_r 

                               mic2_abs.append(mic2_abs_t)
                               
                               mfccs_l = MfccOncData(data_0, n_mfcc)                           
                               mfccs_ll = get_mfccs_mels(n_mfcc, mfccs_l)                               
                               mic2_mfcc_l.append(mfccs_ll)
                               
                               mfccs_r = MfccOncData(data_1, n_mfcc)
                               mfccs_rr = get_mfccs_mels(n_mfcc, mfccs_r)                               
                               mic2_mfcc_r.append(mfccs_rr)
                               
                               mic2_mfcc.append(sum(mfccs_ll) + sum(mfccs_rr))
                               
                               mels_l = MelSOncData(data_0, n_mels)
                               mels_ll = get_mfccs_mels(n_mels, mels_l)                               
                               mic2_mels_l.append(mels_ll)                      
                               
                               mels_r = MelSOncData(data_1, n_mels)
                               mels_rr = get_mfccs_mels(n_mels, mels_r)                               
                               mic2_mels_r.append(mels_rr) 
                               
                               mic2_mels.append(sum(mels_ll) + sum(mels_rr))
                             

                    file_index = file_index +1  
                  
      
        dataset["mic1_coh"] = mic1_coh
        dataset["mic1_abs_l"] = mic1_abs_l
        dataset["mic1_abs_r"] = mic1_abs_r  
        dataset["abs_diff"] = [(m - n)/ (m+n) for m,n in zip(mic1_abs,mic2_abs)]
        dataset["mic1_mfcc_full"] = mic1_mfcc
        dataset["mic1_mels_full"] = mic1_mels
        
        update_dataset("mic1_mfcc_l", n_mfcc, mic1_mfcc_l, len(mic1_mfcc_l),  dataset )
        update_dataset("mic1_mfcc_r", n_mfcc, mic1_mfcc_r, len(mic1_mfcc_r),  dataset )
        update_dataset("mic1_mels_l", n_mels, mic1_mels_l, len(mic1_mels_l),  dataset )
        update_dataset("mic1_mels_r", n_mels, mic1_mels_r, len(mic1_mels_r),  dataset )
     
      
        dataset["mic2_coh"] = mic2_coh
        dataset["mic2_abs_l"] = mic2_abs_l
        dataset["mic2_abs_r"] = mic2_abs_r   
        dataset["mic2_mfcc_full"] = mic2_mfcc
        dataset["mic2_mels_full"] = mic2_mels 
        
        update_dataset("mic2_mfcc_l", n_mfcc, mic2_mfcc_l, len(mic2_mfcc_l),  dataset )
        update_dataset("mic2_mfcc_r", n_mfcc, mic2_mfcc_r, len(mic2_mfcc_r),  dataset )
        update_dataset("mic2_mels_l", n_mels, mic2_mels_l, len(mic2_mels_l),  dataset )
        update_dataset("mic2_mels_r", n_mels, mic2_mels_r, len(mic2_mels_r),  dataset )    
            
        dataset_Final.append(dataset)    
        
    datasetFinal.append(dataset_Final)   
    travelTimeDataset.append(travelTime_Dataset)

with open ('datasetFinal_random_locus.pickle', 'wb' ) as f:
    pickle.dump(datasetFinal, f) 
    
with open ('travelTimeDataset_random_locus.pickle', 'wb' ) as f:
    pickle.dump(travelTimeDataset, f)     
 




