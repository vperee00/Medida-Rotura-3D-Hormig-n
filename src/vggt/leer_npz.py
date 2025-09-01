#!/usr/bin/env python3
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(
        description="Carga y visualiza mapas de profundidad desde un archivo .npz generado por VGGT"
    )
    parser.add_argument(
        "input",
        help="Ruta al directorio que contiene el archivo NPZ, o ruta completa al NPZ"
    )
    parser.add_argument(
        "--file",
        default="predictions.npz",
        help="Nombre del archivo NPZ dentro del directorio (por defecto: predictions.npz)"
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.5,
        help="Umbral de confianza para filtrar la nube 3D (0–1, por defecto: 0.5)"
    )
    parser.add_argument(
        "--show-values",
        action="store_true",
        help="Superpone valores numéricos en cada celda del mapa de profundidad (recomendado solo para matrices pequeñas)"
    )
    args = parser.parse_args()

    # Determina si 'input' es fichero o carpeta
    if os.path.isdir(args.input):
        npz_path = os.path.join(args.input, args.file)
    else:
        npz_path = args.input

    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f"No existe el fichero: {npz_path!r}")

    # 1) Carga
    data       = np.load(npz_path, allow_pickle=True)
    depth      = data["depth"]           # e.g. (S,1,H,W)
    confidence = data["depth_conf"]      # e.g. (S,1,H,W)
    points3D   = data["world_points_from_depth"]  # (S,H,W,3)

    # 2) Selecciona la primera vista/escala y quita dims de longitud 1
    depth_map  = np.squeeze(depth[0])     # (H, W)
    conf_map   = np.squeeze(confidence[0])# (H, W)
    cloud3D    = points3D[0]              # (H, W, 3)

    H, W = depth_map.shape

    # 3) Guardar mapa de profundidad completo en CSV separado por ';'
    csv_out = os.path.splitext(npz_path)[0] + "_depth_map.csv"
    np.savetxt(csv_out, depth_map, delimiter=";", fmt="%.6f")
    print(f"Mapa de profundidad guardado en CSV: {csv_out}")

    # 4) Muestra el mapa completo
    fig, ax = plt.subplots(figsize=(8, 6))
    heatmap = ax.imshow(depth_map, cmap="plasma")
    ax.set_title("Mapa de profundidad completo")
    plt.colorbar(heatmap, ax=ax, label="Profundidad (unidades)")

    # Añadir valores si se solicitan
    if args.show_values:
        fontsize = max(4, min(12, int(200 / max(H, W))))
        for i in range(H):
            for j in range(W):
                ax.text(
                    j, i,
                    f"{depth_map[i, j]:.2f}",
                    ha="center", va="center",
                    fontsize=fontsize,
                    color=("white" if depth_map[i, j] < depth_map.max() / 2 else "black")
                )
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()

    # 5) Aplica el umbral de confianza
    mask = conf_map > args.conf_threshold

    # 6) Muestra solo los puntos que superan el umbral
    masked_depth = np.where(mask, depth_map, np.nan)
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    heatmap2 = ax2.imshow(masked_depth, cmap="plasma")
    ax2.set_title(f"Mapa de profundidad (conf > {args.conf_threshold})")
    plt.colorbar(heatmap2, ax=ax2, label="Profundidad (unidades)")
    plt.tight_layout()
    plt.show()

    # 7) Filtrado por confianza y muestra cuántos puntos quedan
    xyz_high_conf = cloud3D[mask]        # (N, 3)
    print(f"Puntos con conf > {args.conf_threshold}: {xyz_high_conf.shape[0]}")

    # 8) Guardar nube filtrada a texto
    out_txt = os.path.splitext(npz_path)[0] + f"_pts_conf{args.conf_threshold:.2f}.txt"
    np.savetxt(out_txt, xyz_high_conf, header="X Y Z", comments="")
    print(f"Nube 3D filtrada guardada en: {out_txt}")

if __name__ == "__main__":
    main()
