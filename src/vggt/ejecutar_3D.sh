#!/bin/bash

source /usr/etc/profile.d/conda.sh

cd /home/ice/Montar/WindowsNewM2/xIce/Master/TFM/vggt/
conda activate vggt
python ver_3D.py $1
conda deactivate
