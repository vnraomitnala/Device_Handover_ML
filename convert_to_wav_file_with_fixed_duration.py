# -*- coding: utf-8 -*-
"""
Created on Mon May 22 22:17:49 2023

@author: Vijaya
"""

import os
import wave
from pydub import AudioSegment
from scipy.io import wavfile
import scipy.io.wavfile as wav
import soundfile as sf
from scipy.signal import resample
import numpy as np


output_sampling_rate = 16000  # 16kHz
output_duration = 13  # seconds
silence_threshold = 0.01

def convert_to_wav_file_with_fixed_duration(input_file, output_file):
    speech_signal, original_sampling_rate = sf.read(input_file)

    resampled_signal = resample(speech_signal, int(len(speech_signal) * output_sampling_rate / original_sampling_rate))

    non_silent_indices = np.where(np.abs(resampled_signal) > silence_threshold)[0]

    diffs = np.diff(non_silent_indices)

    gap_indices = np.where(diffs > 1)[0]

    non_silent_segments = []

# Iterate over the gap indices and extract non-silent segments
    start_idx = non_silent_indices[0]
    for gap_idx in gap_indices:
        end_idx = non_silent_indices[gap_idx]
        segment = speech_signal[start_idx:end_idx+1]
        non_silent_segments.append(segment)
        start_idx = non_silent_indices[gap_idx+1]

# Handle the last segment
    end_idx = non_silent_indices[-1]
    last_segment = speech_signal[start_idx:end_idx+1]
    non_silent_segments.append(last_segment)

# Concatenate the non-silent segments
    speech_without_silence = np.concatenate(non_silent_segments)
    resampled_signal = speech_without_silence
    
    #abs_signal = np.abs(resampled_signal)
    #resampled_signal = abs_signal * 1000
    
# Repeat the signal to fill the desired duration if it's shorter
    num_samples = output_sampling_rate * output_duration
    extended_signal = resampled_signal

    while len(extended_signal) < num_samples:
        extended_signal = np.concatenate((extended_signal, resampled_signal))

# Trim the signal to the desired duration if it's longer
    extended_signal = extended_signal[:num_samples]
    

# Save the extended signal as a .wav file
    
    new_file_name = os.path.splitext(output_file)[0] + ".wav"
    print(new_file_name)
    sf.write(new_file_name, extended_signal, output_sampling_rate )


directory = "audio"

k = 0

#convert_to_fixed_duration(str2, str4, duration_ms)

#os.rmdir()('tmp/dev-clean1' , ignore_errors=True)


for root, dirs, files in os.walk(directory):
    dirs.sort(key=int)
    for file in (files):
        str2 = os.path.join(root, file)  
        #print(str2)
        ss =  f'{k}'
        str3 = "audio-converted"
        str4 = str3 +"/" + ss
        os.makedirs(str3, exist_ok = True) 
        if (str2.find('.flac') != -1 ):        
            convert_to_wav_file_with_fixed_duration(str2, str4)
            k= k+1       
        os.remove(str2)