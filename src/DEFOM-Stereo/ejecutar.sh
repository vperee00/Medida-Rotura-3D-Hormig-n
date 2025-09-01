#!/bin/bash

echo "Uso ./ejecutar.sh imagen_izquierda imagen_derecha directorio_resultados"

source /usr/etc/profile.d/conda.sh

cd DEFOM-Stereo-main/
conda activate defomstereo
python demo.py --restore_ckpt defomstereo_vitl_sceneflow.pth \
	-l $1 \
	-r $2 \
	--output_directory $3 \
	--save_numpy
conda deactivate
