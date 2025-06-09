#!/usr/bin/env python3
import sys
import os
import numpy as np
import open3d as o3d
from PIL import Image

def main(npz_path):
    if not os.path.isfile(npz_path):
        print(f"Error: no existe {npz_path}", file=sys.stderr)
        sys.exit(1)

    # 1) Carga world_points y confidence
    data = np.load(npz_path)
    pts = data["world_points"][0, 0]        # (H, W, 3)
    h, w, _ = pts.shape
    pts = pts.reshape(-1, 3)               # (N, 3)

    conf = None
    if "world_points_conf" in data:
        conf_arr = data["world_points_conf"][0, 0]  # (H, W)
        conf = conf_arr.reshape(-1)                 # (N,)
    # filtra solo puntos con confianza > 0
    if conf is not None:
        mask = conf > 0.0
        pts = pts[mask]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    # 2) Colorea con input_vggt.png
    img_path = os.path.join(os.path.dirname(npz_path), "input_vggt.png")
    if os.path.isfile(img_path):
        img = Image.open(img_path).convert("RGB")
        img_arr = np.asarray(img, dtype=np.float32) / 255.0  # (H_img, W_img, 3)
        # redimensiona a (H, W) si hace falta
        if img_arr.shape[0] != h or img_arr.shape[1] != w:
            img = img.resize((w, h), Image.BILINEAR)
            img_arr = np.asarray(img, dtype=np.float32) / 255.0
        colors = img_arr.reshape(-1, 3)
        if conf is not None:
            colors = colors[mask]
        pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        # gris si falta la imagen
        gray = np.full((len(pts), 3), 0.5, dtype=np.float32)
        pcd.colors = o3d.utility.Vector3dVector(gray)

    # 3) Ventana de picking con bucle manual
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(
        window_name="Pick 3D points (Shift+Click)",
        width=800, height=600
    )
    vis.add_geometry(pcd)

    picked_set = set()
    try:
        while True:
            vis.poll_events()
            vis.update_renderer()

            current = set(vis.get_picked_points())
            new = current - picked_set
            for idx in sorted(new):
                x, y, z = np.asarray(pcd.points)[idx]
                # imprime al vuelo cada pick
                print(f"PICK {x:.6f} {y:.6f} {z:.6f}", flush=True)
            picked_set = current
    except Exception:
        # sale cuando cierres la ventana
        pass

    vis.destroy_window()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: pick_points.py <predictions.npz>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
