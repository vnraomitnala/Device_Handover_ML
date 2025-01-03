# Aim of next block of work

## Smart Audio for Smart Devices
 
And with a change to the direction of the problem: which smart speaker
is it best to collect the audio (exactly the same answer as the
previous paper for one talker, but how do we distinguish with multiple
talkers and can we separate the speech from each talker?).
 
For example we have two stationary talkers A and B talking in turn to
one CN. A and B are in a room with minimum of two smart devices D1 and
D2. We also have background noise in the room N. We want to use the
correct microphone in either D1 or D2 to collect the speech from A or
B (ie we use magnitude square coherence + amplitude, maybe we need
blind source separation as well, maybe we don’t). Maybe start without
any blind source separation, see how badly it fails if the background
noise is quieter than the speech, but still “noisy”. If it fails badly
then we will need blind source separation (BSS) to identify the noise
vs the speech from A or B.
 
If we need BSS then we can approach Joyraj for help and maybe also use
his noise suppression algorithm.
 
# Notes on running code

## Create curve points from Bezier Curve


```create-locus-positions.py```

this creates the file locus_pos_list_20000.pickle which just contains the positions

## Create locus from Bezier points

uses locus_pos_list_20000.pickle for the points and creates 50 steps, 10 warm up steps, uniform velocity 1 m/s
Adds the element of time

```create-random-locus.py```

outputs ```locusMovement-randum-locus_20000.pickle```

output is dictionary dict_keys(['locus', 'travelTime'])

locus is lists of 400 each with list of 50 3D points

``` python
locusDataset['locus'][0]
[[1.5, 2.75, 1.4], [1.55, 2.45, 1.4], [1.59, 2.19, 1.4], [1.63, 1.97, 1.4], [1.66, 1.79, 1.4], [1.69, 1.63, 1.4], [1.71, 1.51, 1.4], [1.73, 1.43, 1.4], [1.75, 1.37, 1.4], [1.76, 1.33, 1.4], [1.77, 1.33, 1.4], [1.77, 1.34, 1.4], [1.77, 1.38, 1.4], [1.77, 1.44, 1.4], [1.77, 1.51, 1.4], [1.76, 1.61, 1.4], [1.75, 1.71, 1.4], [1.74, 1.83, 1.4], [1.73, 1.96, 1.4], [1.72, 2.1, 1.4], [1.7, 2.25, 1.4], [1.69, 2.4, 1.4], [1.67, 2.56, 1.4], [1.65, 2.72, 1.4], [1.63, 2.87, 1.4], [1.62, 3.03, 1.4], [1.6, 3.19, 1.4], [1.58, 3.33, 1.4], [1.56, 3.48, 1.4], [1.55, 3.61, 1.4], [1.53, 3.73, 1.4], [1.52, 3.85, 1.4], [1.51, 3.94, 1.4], [1.5, 4.03, 1.4], [1.49, 4.09, 1.4], [1.48, 4.13, 1.4], [1.48, 4.16, 1.4], [1.48, 4.16, 1.4], [1.48, 4.14, 1.4], [1.48, 4.09, 1.4], [1.49, 4.01, 1.4], [1.5, 3.9, 1.4], [1.52, 3.76, 1.4], [1.54, 3.59, 1.4], [1.56, 3.38, 1.4], [1.59, 3.14, 1.4], [1.62, 2.85, 1.4], [1.66, 2.53, 1.4], [1.7, 2.16, 1.4], [1.75, 1.75, 1.4]]
```

``` python
locusDataset['travelTime'][10]
[0.0, 0.2, 0.4, 0.6000000000000001, 0.8, 1.0, 1.2000000000000002, 1.4000000000000001, 1.6, 1.8, 2.0, 2.2, 2.4000000000000004, 2.6, 2.8000000000000003, 3.0, 3.2, 3.4000000000000004, 3.6, 3.8000000000000003, 4.0, 4.2, 4.4, 4.6000000000000005, 4.800000000000001, 5.0, 5.2, 5.4, 5.6000000000000005, 5.800000000000001, 6.0, 6.2, 6.4, 6.6000000000000005, 6.800000000000001, 7.0, 7.2, 7.4, 7.6000000000000005, 7.800000000000001, 8.0, 8.200000000000001, 8.4, 8.6, 8.8, 9.0, 9.200000000000001, 9.4, 9.600000000000001, 9.8]
```

## Get locus position list that has transitions - TODO (couldn't find the python code doing this)

outputs ```locusMovement-randum-locus-with-transitions.pickle```

## Simulate room with audio and generate the features (noise soon)

```audio-feature-extraction.py```

uses ```locusMovement-randum-locus-with-transitions.pickle``` and ```audio``` (audio files directory) -- as inputs and generates features

outputs ```datasetFinal_random_locus.pickle```   -- audio features
outputs ```travelTimeDataset_random_locus.pickle```  -- dataset to get the ground truth mic distance from source

## Get ground truth distance from source to mic positions based on travel time

uses ```locusMovement-randum-locus-with-transitions.pickle```, ```travelTimeDataset_random_locus.pickle``` and ```audio``` (audio files directory) -- as inputs and generates distance

```groundTruth_distance_mic_calculation.py```

outputs ```distanceDataset_random_move.pickle```   -- distance dataset

## 1D CNN

uses ```datasetFinal_random_locus.pickle```` and ```distanceDataset_random_locus.pickle``` -- as inputs and processes 1DCNN

```ml_1D_CNN.py```



