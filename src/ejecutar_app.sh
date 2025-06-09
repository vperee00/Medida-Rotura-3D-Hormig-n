#!/bin/bash

source /usr/etc/profile.d/conda.sh

conda activate vggt
python main.py
conda deactivate
