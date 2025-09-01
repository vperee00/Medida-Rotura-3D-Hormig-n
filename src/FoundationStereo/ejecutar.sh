#!/bin/bash

echo "Uso ./ejecutar.sh imagen_izquierda imagen_derecha directorio_resultados"

source /usr/etc/profile.d/conda.sh

cd FoundationStereo-master
conda activate foundation_stereo
python scripts/run_demo.py --left_file $1 --right_file $2 --ckpt_dir ./pretrained_models/model_best_bp2.pth --out_dir $3
conda deactivate
