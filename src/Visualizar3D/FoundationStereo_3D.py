import argparse
import numpy as np
import open3d as o3d

def main():
    ap = argparse.ArgumentParser(description="Mostrar PLY como hace FoundationStereo")
    ap.add_argument("--ply", required=True, help="Ruta a cloud_denoise.ply (o cloud.ply)")
    args = ap.parse_args()

    pcd = o3d.io.read_point_cloud(args.ply)
    if pcd.is_empty():
        raise SystemExit(f"No pude cargar puntos de {args.ply}")

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    main()
