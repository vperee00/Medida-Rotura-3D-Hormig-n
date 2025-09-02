#!/bin/bash

source /usr/etc/profile.d/conda.sh
export PYTHONNOUSERSITE=1
conda activate foundation_stereo
python FoundationStereo_3D.py --ply ./tests/FoundationStereo/cloud_denoise.ply
conda deactivate
