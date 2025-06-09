#!/bin/bash

source /usr/etc/profile.d/conda.sh

conda activate vggt
python run_vggt.py --left $1 --right $2 --out_dir $3
conda deactivate
