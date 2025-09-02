#!/usr/bin/env python3
import numpy as np
import open3d as o3d

NPZ_PATH = "predictions.npz"
VIEW_IDX = 0             # 0 = izquierda (suele serlo), 1 = derecha
CONF_THRESH = 3.0        # súbelo si ves ruido (p.ej. 5–8)
DEPTH_MIN, DEPTH_MAX = 0.1, 5.0
VOXEL_M = 0.003          # 0: sin downsample; 0.003–0.01 suele ir bien

data = np.load(NPZ_PATH, allow_pickle=True)
wp   = data["world_points"]        # (1, 2, H, W, 3)
img  = data["images"]              # (1, 2, 3, H, W), float32 en [0,1]
wpc  = data.get("world_points_conf")  # (1, 2, H, W) o None

# Selección de vista
XYZ = wp[0, VIEW_IDX]              # (H, W, 3)
RGB = np.transpose(img[0, VIEW_IDX], (1, 2, 0))  # (H, W, 3) en [0,1]

# Máscara de confianza + recorte por rango Z
mask = np.ones(XYZ.shape[:2], dtype=bool)
if wpc is not None:
    mask &= (wpc[0, VIEW_IDX] >= CONF_THRESH)

Z = XYZ[..., 2]
mask &= np.isfinite(Z) & (Z > DEPTH_MIN) & (Z < DEPTH_MAX)

pts = XYZ[mask].reshape(-1, 3).astype(np.float32)
cols = RGB[mask].reshape(-1, 3).astype(np.float32)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pts)
pcd.colors = o3d.utility.Vector3dVector(cols)

if VOXEL_M > 0:
    pcd = pcd.voxel_down_sample(VOXEL_M)

# Normales opcionales (mejor shading)
pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=30))
pcd.orient_normals_consistent_tangent_plane(30)

o3d.visualization.draw_geometries([pcd], window_name="VGGT world_points -> Open3D", width=1280, height=800)
