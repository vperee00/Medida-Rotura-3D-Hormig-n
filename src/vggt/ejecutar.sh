#!/bin/bash

echo "Abrir en 127.0.0.1:7860"

source /usr/etc/profile.d/conda.sh

cd vggt/
conda activate vggt
python demo_gradio.py
conda deactivate
