import numpy as np
import matplotlib.pyplot as plt

puntos = np.load("depth_meter.npy")
# Si es necesario, se puede transponer el array para recuperar la orientación original:
puntos = puntos.T  # Esto cambiaría la forma a (461, 1000) si es necesario.

plt.imshow(puntos, cmap='viridis')
plt.colorbar(label='Valor de profundidad')
plt.title("Mapa de Profundidad")
plt.show()
