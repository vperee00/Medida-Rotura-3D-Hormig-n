import os, time
import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, SuperPointForKeypointDetection

# --------------------------------------------------------------------------------------------------

# Opcional: uso de GPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --------------------------------------------------------------------------------------------------

# Carga perezosa de modelos profundos
_superpoint_processor = None
_superpoint_model = None
_d2net_model = None
_r2d2_model = None

# --------------------------------------------------------------------------------------------------

def detect_harris_corners(image, threshold=0.01, block_size=2, ksize=3, k=0.04, dilate_resp=True):
    if isinstance(image, str):
        img = cv2.imread(image)
        if img is None:
            raise FileNotFoundError(f"Imagen no encontrada: {image}")
    else:
        img = image.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dst = cv2.cornerHarris(np.float32(gray), block_size, ksize, k)
    
    if dilate_resp:
        dst = cv2.dilate(dst, None)
        
    mask = dst > threshold * dst.max()
    out = img.copy()
    out[mask] = [0, 0, 255]
    ys, xs = np.where(mask)
    corners = np.vstack((xs, ys)).T
    
    return out, corners

# --------------------------------------------------------------------------------------------------

def detect_shi_tomasi(image, max_corners=100, quality=0.01, min_dist=10):
    if isinstance(image, str): img = cv2.imread(image)
    else: img = image.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pts = cv2.goodFeaturesToTrack(gray, maxCorners=max_corners, qualityLevel=quality, minDistance=min_dist)
    out = img.copy()
    if pts is not None:
        pts = np.int0(pts)
        for p in pts.reshape(-1,2):
            cv2.circle(out, tuple(p), 4, (0,0,255), 1)
        corners = pts.reshape(-1,2)
    else:
        corners = np.empty((0,2), dtype=int)
    return out, corners

# --------------------------------------------------------------------------------------------------

def detect_fast(image, threshold=50, nonmax=True):
    if isinstance(image, str): img = cv2.imread(image)
    else: img = image.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fast = cv2.FastFeatureDetector_create(int(threshold), nonmaxSuppression=nonmax)
    kps = fast.detect(gray, None)
    out = cv2.drawKeypoints(img, kps, None, color=(0,0,255))
    corners = np.array([kp.pt for kp in kps], dtype=np.float32)
    return out, corners

# --------------------------------------------------------------------------------------------------

def detect_agast(image, threshold=30, nonmax=True):
    if isinstance(image, str): img = cv2.imread(image)
    else: img = image.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    agast = cv2.AgastFeatureDetector_create(int(threshold), nonmaxSuppression=nonmax)
    kps = agast.detect(gray, None)
    out = cv2.drawKeypoints(img, kps, None, color=(0,0,255))
    corners = np.array([kp.pt for kp in kps], dtype=np.float32)
    return out, corners

# --------------------------------------------------------------------------------------------------

def detect_orb(image, nfeatures=200):
    if isinstance(image, str): img = cv2.imread(image)
    else: img = image.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=nfeatures)
    kps = orb.detect(gray, None)
    out = cv2.drawKeypoints(img, kps, None, color=(0,0,255))
    corners = np.array([kp.pt for kp in kps], dtype=np.float32)
    return out, corners

# --------------------------------------------------------------------------------------------------

def detect_superpoint(image, conf_thresh=0.015):
    '''Detector SuperPoint usando HuggingFace Transformers sin reproyectar manualmente.'''
    global _superpoint_processor, _superpoint_model
    if _superpoint_model is None:
        repo_id = "stevenbucaille/superpoint"
        _superpoint_processor = AutoImageProcessor.from_pretrained(repo_id)
        _superpoint_model     = SuperPointForKeypointDetection.from_pretrained(repo_id) \
                                  .to(DEVICE).eval()
    # Leer imagen y preparar PIL
    if isinstance(image, str):
        bgr = cv2.imread(image)
    else:
        bgr = image.copy()
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    from PIL import Image
    pil_img = Image.fromarray(rgb)

    # Procesar SIN REDIMENSIONAR (pasa como lista de imágenes)
    inputs = _superpoint_processor(
        images=[pil_img],
        return_tensors="pt",
        do_resize=False,
        do_center_crop=False
    ).to(DEVICE)
    # DEBUG: imprimir shapes
    print(f"[DEBUG] original={bgr.shape[1]}x{bgr.shape[0]}, "
          f"proc={inputs['pixel_values'].shape[3]}x{inputs['pixel_values'].shape[2]}")

    with torch.no_grad():
        outputs = _superpoint_model(**inputs)

    # Extraer kpts y scores
    kpts = outputs.keypoints[0].cpu().numpy()
    scores = outputs.scores[0].cpu().numpy()
    # Filtrar
    mask = scores >= conf_thresh
    kpts = kpts[mask]
    print(f"[DEBUG] pts_before_filter={outputs.keypoints[0].shape[0]}, pts_after={len(kpts)}")
	
    print("Primeros 10 keypoints:", kpts[:10])
	
    # Dibujar sobre BGR original
    out = bgr.copy()
    #for x, y in kpts:
        #cv2.circle(out, (int(x), int(y)), 6, (0, 0, 255), 2)
        
    h, w = out.shape[:2]
    # Desnormalizamos
    pts = np.stack([kpts[:,0]*w, kpts[:,1]*h], axis=1)
    for x_px, y_px in pts:
        # Dibujamos puntos verdes, rellenos, radio 3 para que resalten
        cv2.circle(out, (int(x_px), int(y_px)), 3, (0,255,0), -1)
    
    return out, kpts

# --------------------------------------------------------------------------------------------------

def detect_d2net(image, model_file='d2_net.pth', use_cuda=False):
    '''Detector D2-Net. Asegúrate de clonar e instalar d2-net o añadirlo a PYTHONPATH.'''
    global _d2net_model
    if _d2net_model is None:
        from d2net.lib.models.d2net import D2Net
        _d2net_model = D2Net(model_file=model_file, use_relu=True, use_cuda=use_cuda)
    if isinstance(image, str): img = cv2.imread(image)
    else: img = image.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inp = torch.from_numpy(np.expand_dims(gray.astype(np.float32)/255.0,0)).unsqueeze(0)
    if use_cuda: inp = inp.cuda()
    with torch.no_grad(): keypoints, scores, _ = _d2net_model.run(inp, None)
    out = img.copy()
    corners = []
    for (y,x),s in zip(keypoints, scores):
        cv2.circle(out,(int(x),int(y)),3,(0,0,255),1)
        corners.append((x,y))
    return out, np.array(corners)

# --------------------------------------------------------------------------------------------------

def detect_r2d2(image, model_config='r2d2_WASF_N16.yml', model_weights='r2d2_WASF_N16.pth'):
    '''Detector R2D2. Clona e instala r2d2 para usarlo.'''
    global _r2d2_model
    if _r2d2_model is None:
        from r2d2.model import get_r2d2
        _r2d2_model = get_r2d2(model_config, model_weights).to(DEVICE).eval()
    if isinstance(image, str):
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inp = torch.from_numpy(img.astype(np.float32)/255.0).unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad(): keypoints, scores = _r2d2_model(inp)
    kpts = keypoints[0].cpu().numpy()
    out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for x,y in kpts:
        cv2.circle(out,(int(x),int(y)),3,(0,0,255),1)
    return out, kpts

# --------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Detect corners with various algorithms.")
    parser.add_argument("-d","--detector", required=True,
                        choices=["harris","shi-tomasi","fast","agast","orb","superpoint","d2net","r2d2"],
                        help="Detector a usar.")
    parser.add_argument("image", help="Ruta a la imagen de entrada.")
    parser.add_argument("-t","--threshold", type=float,
                        help="Umbral Harris/FAST/AGAST.")
    parser.add_argument("-b","--block_size", type=int, help="block_size Harris.")
    parser.add_argument("-s","--ksize", type=int, help="ksize Harris.")
    parser.add_argument("-k", type=float, help="k Harris.")
    parser.add_argument("--max_corners", type=int, help="maxCorners Shi-Tomasi.")
    parser.add_argument("--quality", type=float, help="qualityLevel Shi-Tomasi.")
    parser.add_argument("--min_dist", type=float, help="minDistance Shi-Tomasi.")
    parser.add_argument("--nonmax", action='store_true', help="Non-max suppression FAST/AGAST.")
    parser.add_argument("--nfeatures", type=int, help="nfeatures ORB.")
    parser.add_argument("--conf_thresh", type=float, default=0.015, help="confidence SuperPoint.")
    parser.add_argument("--model_file", type=str, default="d2_net.pth", help=".pth D2-Net.")
    parser.add_argument("--use_cuda", action='store_true', help="Use CUDA en D2-Net.")
    parser.add_argument("--model_config", type=str, default="r2d2_WASF_N16.yml", help="config R2D2.")
    parser.add_argument("--model_weights", type=str, default="r2d2_WASF_N16.pth", help="pesos R2D2.")
    parser.add_argument("-o","--output", default="output.png", help="Archivo de salida.")
    args = parser.parse_args()
    func_map = {
        "harris": lambda: detect_harris_corners(
            args.image,
            threshold=args.threshold or 0.01,
            block_size=args.block_size or 2,
            ksize=args.ksize or 3,
            k=args.k or 0.04
        ),
        "shi-tomasi": lambda: detect_shi_tomasi(
            args.image,
            max_corners=args.max_corners or 100,
            quality=args.quality or 0.01,
            min_dist=args.min_dist or 10
        ),
        "fast": lambda: detect_fast(
            args.image,
            threshold=int(args.threshold) if args.threshold else 50,
            nonmax=args.nonmax
        ),
        "agast": lambda: detect_agast(
            args.image,
            threshold=int(args.threshold) if args.threshold else 30,
            nonmax=args.nonmax
        ),
        "orb": lambda: detect_orb(
            args.image,
            nfeatures=args.nfeatures or 200
        ),
        "superpoint": lambda: detect_superpoint(
            args.image,
            conf_thresh=args.conf_thresh
        ),
        "d2net": lambda: detect_d2net(
            args.image,
            model_file=args.model_file,
            use_cuda=args.use_cuda
        ),
        "r2d2": lambda: detect_r2d2(
            args.image,
            model_config=args.model_config,
            model_weights=args.model_weights
        )
    }
    out_img, corners = func_map[args.detector]()
    # Intenta escribir y comprueba el resultado
    success = cv2.imwrite(args.output, out_img)
    abs_path = os.path.abspath(args.output)
    if not success:
        print(f"[ERROR] No se pudo guardar la imagen en {abs_path}")
    else:
        size = os.path.getsize(abs_path)
        mtime = os.path.getmtime(abs_path)
        print(f"Guardado en: {abs_path} ({size} bytes, modificado: {time.ctime(mtime)})")
    print(f"Detector: {args.detector}. Se han encontrado {len(corners)} puntos.")

# --------------------------------------------------------------------------------------------------
