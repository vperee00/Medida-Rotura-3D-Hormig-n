#!/usr/bin/env python3
"""
Script para detectar grietas y medir distancias en múltiples imágenes organizadas en subdirectorios,
escribiendo todos los resultados en un directorio de salida con nombre prefijado.

Uso:
    python soloDistanciasMultiples.py ----input_dir <ruta_raiz> --out_dir <ruta_salida> \
        --roi_dir <ruta_roi> [--camera 0] [--grad_thresh 5.0] [--rows_radius 2]

Directorio raíz debe contener subdirs con:
  - predictions.npz
  - input_vggt.png

Resultados se guardan en <out_dir> como:
  <subdir>_crack_mask.png
  <subdir>_roi_annotation.png
  <subdir>_crack_overlay.png
  <subdir>_crack_overlay_roi.png

Parámetros (procesamiento interno sin modificar):
  --root_dir     Ruta a directorio con subdirectorios.
  --out_dir      Ruta a directorio donde guardar resultados.
  --camera       Cámara (0 o 1). Default: 0.
  --grad_thresh  Umbral Sobel. Default: 5.0.
  --rows_radius  Nº filas de mayor distancia a anotar. Default: 2.
"""
import os
import argparse
import numpy as np
import cv2

# Funciones de procesamiento interno (sin cambios)
def load_depth_and_image(npz_path, camera_idx=0):
    data = np.load(npz_path)
    depth = data['depth'][0, camera_idx, :, :, 0]
    imgs  = data['images'][0, camera_idx]
    img   = np.transpose(imgs, (1,2,0))
    if img.dtype != np.uint8:
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return depth, img

def detect_crack_mask(grad_mag, grad_thresh):
    mask = (grad_mag > grad_thresh).astype(np.uint8) * 255
    kernel = np.ones((5,5), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

def overlay_mask(img, mask, color=(0,0,255), alpha=0.5):
    ov = img.copy()
    ov[mask>0] = ((1-alpha)*ov[mask>0] + alpha*np.array(color)).astype(np.uint8)
    return ov

def find_key_rows(mask, radius):
    H,W = mask.shape
    rows = []
    
    # mask_roi: tu máscara limitada al ROI
    # pc: world_points[0, camera] de tu .npz
    # zones = measure_zones_max(mask_roi, pc)
    # zones será algo como [(y1,x0_1,x1_1,d1), (y2,x0_2,x1_2,d2), ...] una tupla por zona
    
    H, W = mask.shape
    # encontrar componentes conectados
    labels = cv2.connectedComponents((mask>0).astype(np.uint8))[1]
    selected = []  # lista global de (y,x0,x1)
    # procesar cada zona
    for lbl in range(1, labels.max()+1):
        # coordenadas de máscara de esta zona
        ys_lbl, _ = np.where(labels == lbl)
        if ys_lbl.size == 0:
            continue
        # reunir todos los tramos de esta zona
        zone_runs = []  # lista de (dist, y, x0, x1)
        for y in np.unique(ys_lbl):
            xs = np.where(labels[y] == lbl)[0]
            if xs.size > 1:
                x0, x1 = longest_run(xs)
                p0, p1 = pc[y, x0], pc[y, x1]
                dist = np.linalg.norm(p1 - p0)
                zone_runs.append((dist, y, x0, x1))
        # ordenar desc por distancia
        zone_runs.sort(key=lambda t: t[0], reverse=True)
        # seleccionar top runs separados verticalmente 10px
        zone_selected = []  # lista de (y,x0,x1)
        for dist, y, x0, x1 in zone_runs:
            if all(abs(y - y_sel) >= 50 for y_sel, *_ in zone_selected):
                zone_selected.append((y, x0, x1))
            if len(zone_selected) >= radius+1:
                break
        # añadir al global con prefijo de zona
        selected.extend(zone_selected)
    return selected

def annotate_distances(pc, overlay, rows_info):
    for y,x0,x1 in rows_info:
        P0 = pc[y, x0]; P1 = pc[y, x1]
        dist = np.linalg.norm(P1-P0)
        print(f"Fila {y}: distancia={dist:.3f}")
        cv2.line(overlay, (x0,y), (x1,y), (0,255,0), 2)
        cv2.circle(overlay, (x0,y), 4, (255,0,0), -1)
        cv2.circle(overlay, (x1,y), 4, (255,0,0), -1)
        mid = (x0+x1)//2
        cv2.putText(overlay, f"{dist:.3f}", (mid, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    return overlay
    
def longest_run(xs):
    """
    Dado un array 1D ordenado de posiciones xs, devuelve el tramo
    contiguo más largo como (x_start, x_end).
    """
    # dividir en trozos donde la diferencia >1
    splits = np.where(np.diff(xs) > 1)[0] + 1
    runs = np.split(xs, splits)
    # quedarnos con el run de mayor longitud
    best = max(runs, key=lambda r: r.size)
    return int(best[0]), int(best[-1])
    
def measure_zones_max(mask: np.ndarray, pc: np.ndarray) -> list:
    """
    Para cada zona conectada de la máscara (mask>0), calcula:
      - la fila y en que esa zona tiene su tramo horizontal más ancho
      - los extremos x0,x1 de ese tramo continuo
      - la distancia 3D entre world_points[y,x0] y world_points[y,x1]
    Devuelve una lista de tuplas [(y, x0, x1, dist), ...] una por cada componente.
    """
    # Etiqueta componentes conectados
    num_labels, labels = cv2.connectedComponents((mask>0).astype(np.uint8))
    out = []
    for label in range(1, num_labels):
        ys, xs = np.where(labels == label)
        if ys.size == 0: 
            continue
        best = (None, None, None, -1)  # (y,x0,x1,width)
        for y in np.unique(ys):
            row_xs = np.sort(xs[ys == y])
            if row_xs.size < 2:
                continue
            x0, x1 = longest_run(row_xs)
            width = x1 - x0
            if width > best[3]:
                best = (y, x0, x1, width)
        y, x0, x1, _ = best
        if y is not None:
            P0 = pc[y, x0]
            P1 = pc[y, x1]
            dist = np.linalg.norm(P1 - P0)
            out.append((y, x0, x1, dist))
    return out

def find_roi_rows_consecutive(mask: np.ndarray, num: int = 10) -> (int, int):
    """
    Busca en la primera columna las primeras `num` filas consecutivas marcadas (mask>0),
    y en la última columna las últimas `num` filas consecutivas, devolviendo:
      - y_start: fila inicial de la secuencia en la columna 0
      - y_end:   fila final de la secuencia en la última columna

    Args:
        mask (np.ndarray): Máscara binaria 2D (dtype uint8) con valores >0 en negro.
        num  (int):      Longitud de la secuencia de filas consecutivas a buscar.

    Returns:
        (y_start, y_end): dos enteros con la posición de fila.
    """
    H, W = mask.shape

    # Columna 0: buscar hacia abajo la primera secuencia de num
    consec = 0
    y_start = None
    #for y in range(H):
    for y in range(H-1, -1, -1):
        if mask[y, 0] == 0:
            consec += 1
            if consec == num:
                y_start = y - num + 1
                break
        else:
            consec = 0
    if y_start is None:
        raise ValueError(f"No se encontraron {num} filas consecutivas en la columna 0")

    # Última columna: buscar hacia arriba la última secuencia de num
    consec = 0
    y_end = None
    #for y in range(H-1, -1, -1):
    for y in range(H):
        if mask[y, W-1] == 0:
            consec += 1
            if consec == num:
                # y_end = y + num - 1
                y_end = y - num + 1
                break
        else:
            consec = 0
    if y_end is None:
        raise ValueError(f"No se encontraron {num} filas consecutivas en la última columna")

    #return y_start, y_end
    return y_end, y_start

def procesar_predicciones ( npz_path, args ) -> (np.ndarray, np.ndarray):
    
    depth, img = load_depth_and_image(npz_path, args.camera)
    gx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx*gx + gy*gy)
    mask = detect_crack_mask(grad, args.grad_thresh)
    
    return mask, img

# MAIN
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Procesar un único directorio', allow_abbrev=False)
    parser.add_argument('--input_dir',  required=True, help='Directorio con predictions.npz e input_vggt.png')
    parser.add_argument('--output_dir', required=True, help='Directorio para guardar resultados')
    parser.add_argument('--roi_dir',    required=False, default=None, help='Directorio para ROI')
    parser.add_argument('--camera',     type=int, choices=[0,1], default=0)
    parser.add_argument('--grad_thresh',type=float, default=5.0)
    parser.add_argument('--rows_radius',type=int, default=2)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    sub = os.path.basename(os.path.normpath(args.input_dir))

    npz_path = os.path.join(args.input_dir, 'predictions.npz')
    img_path = os.path.join(args.input_dir, 'input_vggt.png')
    
    if not os.path.isfile(npz_path) or not os.path.isfile(img_path):
        raise FileNotFoundError(f"No se encontraron predictions.npz o input_vggt.png en {args.input_dir}")
    
    print(f"Procesando: {sub}")
    mask, img = procesar_predicciones ( npz_path, args )
    cv2.imwrite(os.path.join(args.output_dir, f"{sub}_crack_mask.png"), mask)
    
    if args.roi_dir:
        # si me dan carpeta de ROI, la proceso como antes
        npz_path_roi = os.path.join(args.roi_dir, 'predictions.npz')
        img_path_roi = os.path.join(args.roi_dir, 'input_vggt.png')
        
        if not os.path.isfile(npz_path_roi) or not os.path.isfile(img_path_roi):
            raise FileNotFoundError(f"No se encontraron predictions.npz o input_vggt.png en {args.roi_dir}")
        mask_roi, img_roi = procesar_predicciones(npz_path_roi, args)
        y0, y1 = find_roi_rows_consecutive(mask_roi, 16)
        print(f" ROI filas (desde carpeta): {y0}-{y1}")
    else:
        # sin carpeta de ROI, uso valores por defecto
        y0, y1 = 96, 359
        print(f" ROI filas (por defecto): {y0}-{y1}")
    
    H, W = mask.shape
    roi_img = img.copy()
    cv2.rectangle(roi_img, (0, y0), (W-1, y1), (255,0,0), 2)
    cv2.imwrite(os.path.join(args.output_dir, f"{sub}_roi_annotation.png"), roi_img)

    mask_roi = mask.copy()
    mask_roi[:y0, :] = 0
    mask_roi[y1+1:, :] = 0
    overlay = overlay_mask(img, mask_roi)
    cv2.imwrite(os.path.join(args.output_dir, f"{sub}_crack_overlay.png"), overlay)

    data = np.load(npz_path)
    pc = data['world_points'][0, args.camera]
    rows = find_key_rows(mask_roi, args.rows_radius)
    overlay_roi = annotate_distances(pc, overlay, rows)
    cv2.imwrite(os.path.join(args.output_dir, f"{sub}_crack_overlay_roi.png"), overlay_roi)

    print(f"Guardados archivos con prefijo '{sub}_' en {args.output_dir}")
