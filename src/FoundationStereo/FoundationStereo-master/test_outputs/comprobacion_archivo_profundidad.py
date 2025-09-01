import numpy as np

# Cargar el array
depth_map = np.load("depth_meter.npy")
print("Forma del array:", depth_map.shape)  # Debería mostrar (1000, 461)

# Extraer la sección correspondiente al eje x del 0 al 600
seccion_x = depth_map[0:600, :]
print("Forma de la sección extraída:", seccion_x.shape)

# Comprobar si todos los valores de la sección son cero
if np.all(seccion_x == 0):
    print("La sección del eje x de 0 a 600 está completamente en 0.")
else:
    print("La sección del eje x de 0 a 600 contiene valores distintos de 0.")

# Para información adicional, podemos contar el porcentaje de ceros
total_elementos = seccion_x.size
num_ceros = np.sum(seccion_x == 0)
porcentaje_ceros = (num_ceros / total_elementos) * 100
print(f"Hay {num_ceros} ceros de un total de {total_elementos} elementos ({porcentaje_ceros:.2f}%).")
