#!/bin/bash

echo "Uso ./ejecutar.sh imagen_izquierda imagen_derecha archivo_intrinsecos directorio_resultados"

source /usr/etc/profile.d/conda.sh

cd FoundationStereo-master
conda activate foundation_stereo
python scripts/run_demo.py \
	--left_file $1 \
	--right_file $2 \
	--intrinsic_file $3 \
	--ckpt_dir ./pretrained_models/model_best_bp2.pth \
	--out_dir $4 \
	--scale 1.0 \
	--get_pc 1 \
	--remove_invisible 1 \
	--denoise_cloud    1
conda deactivate
