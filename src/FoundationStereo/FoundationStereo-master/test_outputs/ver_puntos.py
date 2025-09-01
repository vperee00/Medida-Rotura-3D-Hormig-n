import numpy as np

# Cargar el array desde el archivo .npy
puntos = np.load("depth_meter.npy")

# Mostrar la forma y una parte del contenido
print("Forma del array:", puntos.shape)
print("Algunos puntos:", puntos[:5])
