#!/bin/bash

source /usr/etc/profile.d/conda.sh

conda activate detectores

export PYTHONNOUSERSITE=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# Harris
# -t, --threshold 	Umbral relativo frente al máximo de la respuesta Harris. Valores menores (ej. 0.005) marcan más puntos; mayores (ej. 0.02) solo los más fuertes.
# -b, --block_size	Tamaño del vecindario (blockSize) para calcular la matriz de autovalores. Un valor mayor hace la detección más “suave” y menos sensible a ruidos pequeños.
# -s, --ksize		Apertura del filtro Sobel para derivadas. Kernel más grande (ej. 5) suaviza más gradientes, reduciendo esquinas falsas en texturas finas.
# -k float			Constante k del detector de Harris. Suele oscilar entre 0.04 y 0.06; valores altos dan más peso a esquinas muy definidas.	
python detectores_esquinas.py -d harris 400_left_0400.png -t 0.02 -b 4 -s 5 -k 0.06 -o 400_left_0400_harris.png
python detectores_esquinas.py -d harris 400_input_vggt.png -t 0.02 -b 4 -s 5 -k 0.06 -o 400_input_vggt_harris.png

# Shi–Tomasi
# --max_corners	Número máximo de esquinas a devolver.
# --quality		Umbral de calidad; solo se aceptan esquinas con respuesta ≥ (quality × respuesta máxima).
# --min_dist	Distancia mínima (en píxeles) entre esquinas detectadas.
python detectores_esquinas.py -d shi-tomasi 400_left_0400.png --max_corners 10 --quality 0.08 --min_dist 80 -o 400_left_0400_shitomasi.png
python detectores_esquinas.py -d shi-tomasi 400_input_vggt.png --max_corners 10 --quality 0.08 --min_dist 80 -o 400_input_vggt_shitomasi.png

# FAST
# -t, --threshold	Umbral de diferencia de intensidad en el círculo de 16 píxeles. A mayor valor, menos puntos.
# --nonmax			Activa supresión de no-máximos para quedarte solo con los keypoints más potentes.
python detectores_esquinas.py -d fast 400_left_0400.png -t 18 --nonmax -o 400_left_0400_fast.png
python detectores_esquinas.py -d fast 400_input_vggt.png -t 18 --nonmax -o 400_input_vggt_fast.png

# ORB
# --nfeatures	Número máximo de keypoints que ORB intentará detectar (si hay más, ordena por fuerza y coge los top).
python detectores_esquinas.py -d orb 400_left_0400.png --nfeatures 30 -o 400_left_0400_orb.png
python detectores_esquinas.py -d orb 400_input_vggt.png --nfeatures 30 -o 400_input_vggt_orb.png

# SuperPoint
# --conf_thresh	Ej:0.02 = umbral de confianza 2%
python detectores_esquinas.py -d superpoint 400_left_0400.png --conf_thresh 0.02 -o 400_left_0400_SuperPoint.png
python detectores_esquinas.py -d superpoint 400_input_vggt.png --conf_thresh 0.02 -o 400_input_vggt_SuperPoint.png

# D2-Net
# --model_file	d2_net.pth ruta a pesos D2-Net
# --use_cuda	usar GPU si está disponible
#python detectores_esquinas.py -d d2net 400_left_0400.png --model_file /home/ice/Montar/WindowsNewM2/xIce/Master/TFM/git/Modelos/Esquinas/d2_tf.pth --use_cuda -o 400_left_0400_D2-Net.png
#python detectores_esquinas.py -d d2net 400_input_vggt.png --model_file /home/ice/Montar/WindowsNewM2/xIce/Master/TFM/git/Modelos/Esquinas/d2_tf.pth --use_cuda -o 400_input_vggt_D2-Net.png

# R2D2
# --model_conf		r2d2_WASF_N16.yml \  # config YAML
# --model_weights	r2d2_WASF_N16.pth \ # pesos del modelo
#python detectores_esquinas.py -d r2d2 400_left_0400.png --model_weights /home/ice/Montar/WindowsNewM2/xIce/Master/TFM/git/Modelos/Esquinas/faster2d2_WASF_N8_big.pt -o 400_left_0400_R2D2.png
#python detectores_esquinas.py -d r2d2 400_input_vggt.png --model_weights /home/ice/Montar/WindowsNewM2/xIce/Master/TFM/git/Modelos/Esquinas/faster2d2_WASF_N8_big.pt -o 400_input_vggt_R2D2.png

conda deactivate
