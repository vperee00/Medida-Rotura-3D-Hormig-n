#!/usr/bin/env python3
import argparse
import os
from skimage import io, color, feature
from skimage.color import rgba2rgb
from skimage.transform import probabilistic_hough_line
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Detecta líneas en una imagen usando la Transformada de Hough probabilística."
    )
    p.add_argument("-i", "--input",    required=True, help="Ruta a la imagen de entrada.")
    p.add_argument("-o", "--output",   required=True, help="Ruta donde guardar la imagen de salida.")
    p.add_argument("--sigma",          type=float, default=2.0, help="Sigma para Canny.")
    p.add_argument("--threshold",      type=int,   default=10,  help="Puntos mínimos para línea.")
    p.add_argument("--line_length",    type=int,   default=50,  help="Longitud mínima de línea.")
    p.add_argument("--line_gap",       type=int,   default=10,  help="Gap máximo entre segmentos.")
    return p.parse_args()

# --------------------------------------------------------------------------------------------------

def main():
    args = parse_args()

    # --- 1. Carga y posible conversión RGBA→RGB ---
    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"No existe el archivo: {args.input}")
    image = io.imread(args.input)
    if image.ndim == 3 and image.shape[2] == 4:
        image = rgba2rgb(image)

    # --- 2. Escala de grises y bordes ---
    gray = color.rgb2gray(image)
    edges = feature.canny(gray, sigma=args.sigma)

    # --- 3. Hough probabilístico ---
    lines = probabilistic_hough_line(
        edges,
        threshold=args.threshold,
        line_length=args.line_length,
        line_gap=args.line_gap
    )

    # --- 4. Dibujar y guardar ---
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(gray, cmap='gray')
    for (x0, y0), (x1, y1) in lines:
        ax.plot((x0, x1), (y0, y1), '-r', linewidth=2)
    ax.set_axis_off()
    plt.tight_layout(pad=0)
    plt.savefig(args.output, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    print(f"Líneas detectadas: {len(lines)}")
    print(f"Resultado guardado en: {args.output}")

# --------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# --------------------------------------------------------------------------------------------------
