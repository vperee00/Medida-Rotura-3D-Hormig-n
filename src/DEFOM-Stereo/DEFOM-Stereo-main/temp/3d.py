#!/usr/bin/env python3
import numpy as np, cv2, open3d as o3d

# === RUTAS ===
DISP_PATH = "0685_defomstereo_vitl_sceneflow.npy"  # .npy (1000x461, HxW)
LEFT_IMG  = "0685_left.png"    

# === INTRÍNSECOS ESCALADOS A 461x1000 (W x H) ===
fx, fy = 1041.0, 1050.27778
cx, cy = 231.88060, 499.5

# === BASELINE (m) ===
B = 0.270632   # asumiendo que T está en mm

# === FILTROS ===
MIN_DISP    = 0.1
MAX_DEPTH_M = 4.0
VOXEL_M     = 0.005

# --- Carga ---
disp = np.load(DISP_PATH).astype(np.float32)   # (H, W) = (1000, 461)
img  = cv2.cvtColor(cv2.imread(LEFT_IMG, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

H, W = disp.shape
if (img.shape[0], img.shape[1]) != (H, W):
    # reescala disparidad a la resolución de la imagen
    disp = cv2.resize(disp, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    H, W = img.shape[:2]

# --- Profundidad ---
valid = disp > MIN_DISP
Z = np.zeros_like(disp, dtype=np.float32)
Z[valid] = fx * B / disp[valid]
valid &= (Z > 0) & (Z < MAX_DEPTH_M)

# --- Reproyección ---
u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy

pts = np.stack([X[valid], Y[valid], Z[valid]], axis=1)
colors = (img.reshape(-1, 3)[valid.ravel()] / 255.0).astype(np.float32)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pts)
pcd.colors = o3d.utility.Vector3dVector(colors)

if VOXEL_M > 0:
    pcd = pcd.voxel_down_sample(VOXEL_M)

pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
pcd.orient_normals_consistent_tangent_plane(30)

o3d.visualization.draw_geometries([pcd], window_name="DEFOM -> Open3D", width=1280, height=800)
