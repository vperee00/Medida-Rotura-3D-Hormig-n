#!/bin/bash

source /usr/etc/profile.d/conda.sh

mkdir -p "${2}"
ROI_DIR="./Modelos/vggt/ROI/0001/"

conda activate vggt
python vggt_Distancias.py \
	--input_dir "${1}" \
	--output_dir "${2}" \
	--roi_dir "${3}" \
	--camera 0 \
	--grad_thresh 0.00175 \
	--rows_radius 2
	
conda deactivate
