#!/bin/bash

source /usr/etc/profile.d/conda.sh

conda activate vggt

# 
# --sigma 		suavizado para Canny (un valor menor detecta más bordes finos, pero puede generar ruido).
# --threshold	cuantos más puntos alineados requieras, más robustas pero menos líneas detectadas.
# --line_length	filtra líneas menores a ese largo.
# --line_gap	une tramos de línea separados por hasta ese número de píxeles.
python probabilistic_hough_line.py --input 400_left_0400.png --output 400_left_0400_lineas.png --sigma 1.0 --threshold 10 --line_length 10 --line_gap 1

python probabilistic_hough_line.py --input 400_input_vggt.png --output 400_input_vggt_lineas.png --sigma 0.0001 --threshold 10 --line_length 10 --line_gap 1

conda deactivate
