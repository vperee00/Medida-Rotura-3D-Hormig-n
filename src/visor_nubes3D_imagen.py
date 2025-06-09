#!/usr/bin/env python3
import sys
import os

import numpy as np
import open3d as o3d
from PIL import Image

def main(npz_path, img_path):
    # 1) Comprueba que existan los ficheros
    if not os.path.isfile(npz_path):
        print(f"Error: no existe {npz_path}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(img_path):
        print(f"Error: no existe {img_path}", file=sys.stderr)
        sys.exit(1)

    # 2) Carga el NPZ y extrae world_points
    data = np.load(npz_path)
    if "world_points" not in data:
        print("Error: 'world_points' no está en el NPZ. Claves:", data.files, file=sys.stderr)
        sys.exit(1)
    pts = data["world_points"]  # shape esperada: (1, 2, H, W, 3)

    # 3) Selecciona el primer par (índice 0), cámara izquierda
    #    quedamos con pts[batch=0, camera=0, :, :, :] → (H, W, 3)
    if pts.ndim == 5 and pts.shape[0] >= 1 and pts.shape[1] >= 1 and pts.shape[4] == 3:
        pts = pts[0, 0, :, :, :]  # ahora shape = (H, W, 3)
    else:
        print(f"Error: formato inesperado para world_points: {pts.shape}", file=sys.stderr)
        sys.exit(1)

    # 4) Aplana a (N, 3)
    h, w, _ = pts.shape
    points = pts.reshape(-1, 3)

    # 5) Carga la imagen para colores
    img = Image.open(img_path).convert("RGB")
    img_arr = np.asarray(img, dtype=np.float32) / 255.0  # (H_img, W_img, 3)

    # 6) Ajusta si la imagen no coincide exactamente en tamaño
    if img_arr.shape[0] != h or img_arr.shape[1] != w:
        img = img.resize((w, h), Image.BILINEAR)
        img_arr = np.asarray(img, dtype=np.float32) / 255.0

    # 7) Aplana colores a (N, 3)
    colors = img_arr.reshape(-1, 3)

    # 8) Construye la nube de puntos Open3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # 9) Visualiza
    o3d.visualization.draw_geometries(
        [pcd],
        window_name="VGGT 3D Point Cloud",
        width=1024,
        height=768,
        left=50,
        top=50,
        point_show_normal=False
    )

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Uso: {os.path.basename(sys.argv[0])} <predictions.npz> <input_vggt.png>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
