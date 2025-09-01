#!/bin/bash

source /usr/etc/profile.d/conda.sh
export PYTHONNOUSERSITE=1
conda activate vggt
python main.py
conda deactivate
