#!/bin/bash

source /usr/etc/profile.d/conda.sh
export PYTHONNOUSERSITE=1
conda activate foundation_stereo
python VGGT_3D.py
conda deactivate
