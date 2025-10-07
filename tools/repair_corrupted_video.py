# tools/repair_corrupted_video.py
import os, cv2, time, argparse, runpy
import numpy as np
import torch

from simvp.models.simvp_model import SimVP_Model

# ------------------ util cfg/model ------------------
def load_cfg_from_py(cfg_path):
    """ Devuelve un dict con las claves de SimVP si existen; si no, intenta fallback. """
    g = runpy.run_path(cfg_path)
    for key in ('config','cfg','CONFIG','Config'):
        if key in g and isinstance(g[key], dict):
            return g[key]
    # fallback: recolectar claves sueltas
    keys = ['in_shape','hid_S','hid_T','N_S','N_T','spatio_kernel_enc','spatio_kernel_dec','drop','drop_path','model_type']
    if any(k in g for k in keys):
        cfg = {k: g[k] for k in keys if k in g}
        return cfg
    raise ValueError(f"No encontré dict de config ni claves sueltas en {cfg_path}")

def build_config_fallback(H, W, C, pre_seq=10, model_type='gSTA'):
    return {
        'in_shape': [pre_seq, C, H, W],
        'hid_S': 64, 'hid_T': 512, 'N_S': 4, 'N_T': 8,
        'spatio_kernel_enc': 3, 'spatio_kernel_dec': 3,
        'drop': 0.0, 'drop_path': 0.1, 'model_type': model_type
    }

def choose_device():
    if torch.backends.mps.is_available(): return torch.device('mps')
    if torch.cuda.is_available():         return torch.device('cuda:0')
    return torch.device('cpu')

# ------------------ IO & conversión ------------------
def to_model_tensor(frame_bgr, out_hw, use_rgb=True):
    """BGR HxWx3 -> CHW float[0,1] redimensionado a out_hw=(H,W)"""
    H, W = out_hw
    f = cv2.resize(frame_bgr, (W, H), interpolation=cv2.INTER_AREA)
    if use_rgb:
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    x = f.astype(np.float32)/255.0
    x = np.transpose(x, (2,0,1))  # CHW
    return x

def to_display_image(chw_float):
    x = np.clip(chw_float, 0, 1)
    x = (x*255.0).astype(np.uint8)
    x = np.transpose(x, (1,2,0))  # HWC RGB
    x = x[:, :, ::-1]             # RGB->BGR
    return x

# ------------------ detección & parches ------------------
def detect_corruption_mask(frame_bgr, prev_bgr=None, block=16, var_th=2.0, diff_th=18.0):
    """
    Máscara uint8 [H,W]=255 donde el frame parece corrupto.
    Heurísticas: baja varianza por bloque, discontinuidades en grilla, bloque congelado.
    """
    h, w = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    # varianza aprox por bloque
    small = cv2.resize(gray, (max(1,w//block), max(1,h//block)), interpolation=cv2.INTER_AREA)
    mean_blk = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    resid = gray.astype(np.float32) - mean_blk.astype(np.float32)
    var_mask = (np.abs(resid) < var_th).astype(np.uint8) * 255

    # discontinuidades en bordes de macro-bloques
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    grid = np.zeros_like(var_mask)
    grid[::block, :] = 255; grid[:, ::block] = 255
    edge_mask = (mag > 60).astype(np.uint8) * grid

    # bloques congelados respecto al anterior
    freeze_mask = np.zeros_like(var_mask)
    if prev_bgr is not None:
        prev_g = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray, prev_g)
        glob = cv2.blur(diff, (31,31))
        loc  = cv2.blur(diff, (block,block))
        freeze_mask = ((glob > diff_th) & (loc < (diff_th*0.5))).astype(np.uint8) * 255

    mask = cv2.max(var_mask, cv2.max(edge_mask, freeze_mask))
    mask = (mask > 0).astype(np.uint8)*255
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask

def composite_patch(frame_bgr, pred_bgr, mask_u8):
    soft = cv2.GaussianBlur(mask_u8, (0,0), 1.2)
    alpha = (soft.astype(np.float32)/255.0)[..., None]
    out = (alpha*pred_bgr.astype(np.float32) + (1-alpha)*frame_bgr.astype(np.float32)).astype(np.uint8)
    return out

# ------------------ buffer y forward ------------------
class RingBuffer:
    def __init__(self, capacity): self.capacity=capacity; self.data=[]
    def clear(self): self.data=[]
    def append(self, x):
        if len(self.data) >= self.capacity: self.data.pop(0)
        self.data.append(x)
    def ready(self): return len(self.data) >= self.capacity
    def as_tensor(self, device):
        arr = np.stack(self.data, axis=0)  # (T,C,H,W)
        return torch.from_numpy(arr).unsqueeze(0).to(device)  # (1,T,C,H,W)

@torch.no_grad()
def forward_pred(model, buf_tensor):
    return model(buf_tensor)  # (1, pred_len, C, H, W)

# ------------------ main ------------------
def main(args):
    device = choose_device()
    print("Device:", device)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened(): raise FileNotFoundError(args.input)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or args.fps or 30.0
    width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    # writer salida
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, src_fps if args.fps<=0 else args.fps, (width, height))
    if not out.isOpened(): raise RuntimeError(f"No pude abrir VideoWriter a {args.output}")

    # cargar checkpoint y cfg
    ckpt = torch.load(args.ckpt, map_location=device)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']; ckpt_cfg = ckpt.get('config', None)
    elif isinstance(ckpt, dict):
        state_dict = ckpt; ckpt_cfg = None
    else:
        raise ValueError("Formato de checkpoint no soportado")

    cfg = None
    if args.cfg:
        try:
            cfg = load_cfg_from_py(args.cfg)
        except Exception as e:
            print(f"[WARN] Config .py no legible: {e}")
    if cfg is None:
        cfg = ckpt_cfg
    if cfg is None:
        print("[WARN] usando fallback de config.")
        cfg = build_config_fallback(args.in_size, args.in_size, 3, pre_seq=args.pre_seq)

    # forzar in_shape coherente con CLI
    C=3
    cfg['in_shape'] = cfg.get('in_shape', [args.pre_seq, C, args.in_size, args.in_size])
    cfg['in_shape'][0] = args.pre_seq
    cfg['in_shape'][1] = C
    cfg['in_shape'][2] = args.in_size
    cfg['in_shape'][3] = args.in_size
    cfg.setdefault('model_type','gSTA')

    model = SimVP_Model(**cfg).to(device)
    new_sd = { (k[7:] if k.startswith('module.') else k): v for k,v in state_dict.items() }
    model.load_state_dict(new_sd, strict=False)
    model.eval()

    # warm-up para compilar kernels y reservar memoria
    with torch.no_grad():
        dummy = torch.zeros(1, args.pre_seq, C, args.in_size, args.in_size, device=device)
        _ = model(dummy)

    # procesamiento
    buf = RingBuffer(args.pre_seq)
    prev_rx = None
    pred_cache = None
    pred_idx = 0

    # prellenar buffer con los primeros frames (reparando mínimamente si hace falta)
    prefill = 0
    while prefill < args.pre_seq:
        ok, f = cap.read()
        if not ok: break
        # detección básica para evitar meter basura al contexto
        m = detect_corruption_mask(f, prev_bgr=prev_rx, block=args.block, var_th=args.var_th, diff_th=args.diff_th)
        ratio = np.mean(m>0)
        if ratio > args.small_thresh:
            # parche mínimo para no contaminar el contexto
            f = cv2.inpaint(f, (m>0).astype(np.uint8)*255, 3, cv2.INPAINT_TELEA)
        buf.append(to_model_tensor(f, (args.in_size,args.in_size), True))
        prev_rx = f
        prefill += 1
        out.write(f)

    # loop principal
    frame_id = prefill
    t0 = time.time()
    small_n = med_n = big_n = 0

    while True:
        ok, frame = cap.read()
        if not ok: break

        # detectar corrupción actual
        mask = detect_corruption_mask(frame, prev_bgr=prev_rx, block=args.block, var_th=args.var_th, diff_th=args.diff_th)
        ratio = float(np.mean(mask>0))
        prev_rx = frame.copy()

        repaired = frame  # por defecto, el mismo
        used_pred_this_frame = False

        if ratio < args.small_thresh:
            # (opcional) inpaint microdefectos
            if args.use_inpaint and np.any(mask):
                repaired = cv2.inpaint(repaired, (mask>0).astype(np.uint8)*255, 3, cv2.INPAINT_TELEA)
            small_n += 1

        elif ratio < args.large_thresh:
            # defecto mediano: blend con predicción del tiempo t
            if buf.ready():
                if pred_cache is None or pred_idx >= pred_cache.shape[1]:
                    inp = buf.as_tensor(device)
                    pred_cache = forward_pred(model, inp)  # (1, pred_len, C,H,W)
                    pred_idx = 0
                pred_t = pred_cache[0, 0].detach().cpu().numpy()  # CHW
                pred_bgr = to_display_image(pred_t)
                pred_bgr = cv2.resize(pred_bgr, (width, height), interpolation=cv2.INTER_LINEAR)
                repaired = composite_patch(repaired, pred_bgr, mask)
                used_pred_this_frame = True
            med_n += 1

        else:
            # muy roto: actuar como dropout (usar secuencia predicha)
            if buf.ready():
                if pred_cache is None or pred_idx >= pred_cache.shape[1]:
                    inp = buf.as_tensor(device)
                    pred_cache = forward_pred(model, inp)
                    pred_idx = 0
                pred_t = pred_cache[0, pred_idx].detach().cpu().numpy()
                pred_idx += 1
                repaired = cv2.resize(to_display_image(pred_t), (width, height), interpolation=cv2.INTER_LINEAR)
                used_pred_this_frame = True
            big_n += 1

        # escribir salida
        out.write(repaired)

        # actualizar buffer del modelo:
        # - si usamos predicción (total o parcial), alimentar el contexto con la versión "reparada"
        # - sino, usar el frame recibido (o inpaint si se aplicó)
        feed_chw = to_model_tensor(repaired if (used_pred_this_frame or (args.use_inpaint and np.any(mask))) else frame,
                                   (args.in_size,args.in_size), True)
        buf.append(feed_chw)

        frame_id += 1

    cap.release()
    out.release()
    dt = time.time() - t0
    print(f"Listo: {args.output}")
    print(f"Frames procesados: {frame_id}/{total} en {dt:.1f}s  "
          f"(chicos={small_n}, medianos={med_n}, grandes={big_n})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  type=str, required=True, help="MP4 corrupto (ej. openh264_lossy.mp4)")
    ap.add_argument("--output", type=str, required=True, help="MP4 reparado")
    ap.add_argument("--ckpt",   type=str, required=True)
    ap.add_argument("--cfg",    type=str, default=None)
    ap.add_argument("--pre_seq",   type=int, default=10)
    ap.add_argument("--pred_len",  type=int, default=10)  # (no siempre usado; el modelo ya lo define)
    ap.add_argument("--in_size",   type=int, default=64)
    ap.add_argument("--fps",       type=int, default=0, help="0=usar fps fuente")
    # detector
    ap.add_argument("--block",       type=int, default=16)
    ap.add_argument("--var_th",      type=float, default=2.0)
    ap.add_argument("--diff_th",     type=float, default=18.0)
    ap.add_argument("--small_thresh",type=float, default=0.05, help="área corrupta < este ratio => inpaint opcional")
    ap.add_argument("--large_thresh",type=float, default=0.40, help="área corrupta >= este ratio => dropout completo")
    ap.add_argument("--use_inpaint", action="store_true", help="aplica inpaint en defectos chicos")
    args = ap.parse_args()
    main(args)
