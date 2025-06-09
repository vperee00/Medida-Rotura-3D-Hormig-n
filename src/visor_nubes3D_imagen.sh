#!/bin/bash

source /usr/etc/profile.d/conda.sh

echo "Primer parametro: ruta a predictions.npz" 
echo "Segundo parametro: ruta a input_vggt.png"

conda activate vggt
python visor_nubes3D_imagen.py $1 $2
conda deactivate
