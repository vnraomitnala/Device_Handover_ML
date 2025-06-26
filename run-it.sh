#!/bin/bash
set -e
set -x
echo "using python:"
which python3

#python3 create-locus-positions.py

#python3 create-random-locus.py

# modify this to change the noise
python3 audio-feature-extraction.py

python3 groundTruth_distance_mic_calculation.py

python3 ml_1D_CNN.py


