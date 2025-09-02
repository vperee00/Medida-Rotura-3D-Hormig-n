#!/usr/bin/env python3
import os
import sys
import argparse
import tempfile

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

# --------------------------------------------------------------------------------------------------
# 1) Directorio donde está este script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 2) Carpeta “Modelos/” que contiene el paquete vggt/
MODELOS_DIR = os.path.join(SCRIPT_DIR, "Modelos")
# Importar modulos
sys.path.insert(0, MODELOS_DIR)
# --------------------------------------------------------------------------------------------------

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

# --------------------------------------------------------------------------------------------------

def resize_and_save(img_path, target_h=4000):
    """
    Redimensiona la imagen a altura=target_h manteniendo aspecto,
    guarda en un fichero temporal .png y devuelve su ruta.
    """
    img = Image.open(img_path)
    w, h = img.size
    new_w = int(w * target_h / h)
    resized = img.resize((new_w, target_h), Image.BILINEAR)
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    resized.save(tmp.name)
    return tmp.name

# --------------------------------------------------------------------------------------------------

def main():
    # — argumentos
    p = argparse.ArgumentParser("Run VGGT on a stereo pair")
    p.add_argument("--left",    required=True, help="Path to left image")
    p.add_argument("--right",   required=True, help="Path to right image")
    p.add_argument("--out_dir", required=True, help="Directory to save outputs")
    args = p.parse_args()

    # prepara carpeta de salida
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # dispositivo e inferencia en FP16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16

    # — instancia y carga modelo EN CPU
    model = VGGT()
    ckpt_dir = os.path.join(MODELOS_DIR, "vggt")
    pt1 = os.path.join(ckpt_dir, "model.pt")
    pt2 = os.path.join(ckpt_dir, "checkpoint_liberty_with_aug.pth")
    if   os.path.isfile(pt2):
        ckpt_path = pt2
    elif os.path.isfile(pt1):
        ckpt_path = pt1
    else:
        sys.stderr.write(f"No encontré ni {pt1} ni {pt2}\n")
        sys.exit(1)

    # cargamos el state_dict en CPU
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict)
    # movemos el modelo a GPU (float32) y a modo eval
    model = model.to(device)
    model.eval()

    # — prep imágenes en alta resolución y inferencia
    left_tmp  = resize_and_save(args.left,  target_h=4000)
    right_tmp = resize_and_save(args.right, target_h=4000)
    imgs = load_and_preprocess_images([left_tmp, right_tmp]).to(device)
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
        preds = model(imgs)

    # — guardar predictions.npz
    npz_path = os.path.join(out_dir, "predictions.npz")
    np.savez(npz_path, **{
        k: (v.cpu().numpy() if isinstance(v, torch.Tensor) else v)
        for k, v in preds.items()
    })
    print("Predictions guardadas en:", npz_path)

    # — cargar NPZ para post-procesado
    data = np.load(npz_path)

    # 1) Depth map (solo cámara izquierda)
    depth_raw = data["depth"][0, 0]        # shape = (H, W)
    depth_map = depth_raw
    np.save(os.path.join(out_dir, "depth.npy"), depth_map)
    plt.figure(figsize=(8,6))
    plt.imshow(depth_map, cmap="plasma", origin="lower")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "depth_map.png"), bbox_inches="tight")
    plt.close()

    # 2) World points (par izquierdo)
    pts = data["world_points"][0, 0]       # shape = (H, W, 3)
    h, w, _ = pts.shape
    points_flat = pts.reshape(-1, 3)       # (H*W, 3)
    np.save(os.path.join(out_dir, "points3d_flat.npy"), points_flat)
    np.savetxt(
        os.path.join(out_dir, "points3d_flat.csv"),
        points_flat,
        delimiter=";",
        fmt="%.6f"
    )

    # 3) Imagen preprocesada
    if "images" in data:
        # Extraemos la imagen izquierda: data["images"] tiene shape (1,2,3,H,W)
        img_proc = data["images"][0, 0]     # → (3, H, W)
        img_proc = np.transpose(img_proc, (1, 2, 0))  # → (H, W, 3)
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        img_proc = np.clip(img_proc * std + mean, 0, 1)
        plt.imsave(os.path.join(out_dir, "input_vggt.png"), img_proc)
    else:
        print("No se encontró 'images' en NPZ")

    # — libera VRAM
    del imgs, preds
    torch.cuda.empty_cache()

# --------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# --------------------------------------------------------------------------------------------------
