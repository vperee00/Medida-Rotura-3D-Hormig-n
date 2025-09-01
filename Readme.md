# Medida de la rotura del Hormigon 3D

Este proyecto consiste la implementación de una aplicacion que a partir de un par de imágenes estéreo, detecta las grietas que se hayan podido producir y devuelve una estimación relativa de su tamaño.

## Requisitos de usuarios
Tener instalado Python 3. Y se recomienda tener instalado Conda para crear un entorno donde instalar todas las dependencias de Python.

## Instalación

Para instalarlo se necesita clonar el repositorio o descargar el zip desde https://github.com/vperee00/Medida-Rotura-3D-Hormig-n/tree/master

Para crear el entorno con todas las dependencias necesarias se puede usar el archivo environment.yml con el siguiente comando.
conda env create -f environment.yml

Y descargar el archivo del modelo se puede encontrar en https://huggingface.co/facebook/VGGT-1B/blob/main/model.pt

## Ejecución

Para ejecutar la aplicación se puede usar el script creado para ese efecto: ejecutar_app.sh

O de forma manual con 
	conda activate vggt
	python main.py
	conda deactivate

# ![Licencia GPLv3](gplv3-127x51.png  "Licencia GPLv3")
