# tools/infer_fpv_mosaic.py
import os
import argparse
import cv2
import numpy as np
import torch
import runpy

from simvp.models.simvp_model import SimVP_Model

def load_dataset(path):
    arr = np.load(path, allow_pickle=False)
    # Soportar .npz guardado como data=... o arr_0
    if isinstance(arr, np.lib.npyio.NpzFile):
        if 'data' in arr.files:
            data = arr['data']
        elif 'arr_0' in arr.files:
            data = arr['arr_0']
        else:
            raise KeyError(f"{path} no contiene 'data' ni 'arr_0'. Contiene: {arr.files}")
    else:
        # Si fuera .npy
        data = arr
    return data.astype(np.float32)  # (N, T, C, H, W)

def load_cfg_from_py(cfg_path):
    """Carga un dict de config desde un .py (busca claves típicas)."""
    g = runpy.run_path(cfg_path)
    for key in ('config', 'cfg', 'CONFIG', 'Config'):
        if key in g and isinstance(g[key], dict):
            return g[key]
    # fallback: buscar un dict que tenga 'in_shape'
    for v in g.values():
        if isinstance(v, dict) and 'in_shape' in v:
            return v
    raise ValueError(f"No pude encontrar un dict de config en {cfg_path}")

def build_config_fallback(data, model_type='gSTA'):
    """Derivar un config mínimo desde el dataset si el ckpt/cfg no aportan."""
    # data: (N, T, C, H, W)
    _, T, C, H, W = data.shape
    cfg = {
        'in_shape': [min(10, T//2), C, H, W],  # heurística: 10 si se puede
        'hid_S': 64, 'hid_T': 512, 'N_S': 4, 'N_T': 8,
        'spatio_kernel_enc': 3, 'spatio_kernel_dec': 3,
        'drop': 0.0, 'drop_path': 0.1,
        'model_type': model_type
    }
    return cfg

def make_video(frames, path, fps=10):
    h, w, c = frames[0].shape
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for f in frames:
        f = np.clip(f * 255, 0, 255).astype(np.uint8)
        out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    out.release()

def add_labels(frame, labels, font_scale=0.5, thickness=1):
    h, w, _ = frame.shape
    part_w = w // len(labels)
    for i, text in enumerate(labels):
        x = i * part_w + 5
        y = 15
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thickness, cv2.LINE_AA)
    return frame

def main(args):
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # === Load checkpoint (tolerante a formatos) ===
    ckpt = torch.load(args.ckpt, map_location=device)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
        ckpt_cfg = ckpt.get('config', None)
    elif isinstance(ckpt, dict):
        # a veces guardan directamente el state_dict como dict de pesos
        state_dict = ckpt
        ckpt_cfg = None
    else:
        raise ValueError("Formato de checkpoint no soportado.")

    # === Config ===
    data = load_dataset(args.data)  # (N, T, C, H, W)
    cfg = None
    if ckpt_cfg is not None:
        cfg = ckpt_cfg
    elif args.cfg:
        cfg = load_cfg_from_py(args.cfg)
    else:
        cfg = build_config_fallback(data)

    # Validar/ajustar in_shape con el dataset si hace falta
    T, C, H, W = data.shape[1:]
    if 'in_shape' not in cfg or len(cfg['in_shape']) != 4:
        cfg['in_shape'] = [min(args.pred_len, T//2), C, H, W]
    else:
        # forzar canales y tamaño de imagen a lo que hay en data
        cfg['in_shape'][1] = C
        cfg['in_shape'][2] = H
        cfg['in_shape'][3] = W

    pre_seq = cfg['in_shape'][0]
    aft_seq = args.pred_len

    # === Build model ===
    if 'model_type' not in cfg:
        cfg['model_type'] = 'gSTA'  # default razonable
    model = SimVP_Model(**cfg).to(device)
    # limpiar posibles prefijos 'module.' si existieran
    new_sd = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_sd[k[len('module.'):]] = v
        else:
            new_sd[k] = v
    model.load_state_dict(new_sd, strict=False)
    model.eval()

    # === Predicciones para N ejemplos y mosaico ===
    N = min(len(data), args.max_samples)
    samples = data[:N]  # (N, T, C, H, W)

    pred_clips = []
    for idx in range(N):
        sample = torch.tensor(samples[idx]).unsqueeze(0).to(device)  # (1,T,C,H,W)
        inp = sample[:, :pre_seq]
        gt  = sample[:, pre_seq:pre_seq+aft_seq]
        with torch.no_grad():
            pred = model(inp)

        inp = inp.squeeze(0).cpu().numpy().transpose(0,2,3,1)   # (Tin,H,W,C)
        gt  = gt.squeeze(0).cpu().numpy().transpose(0,2,3,1)    # (Tout,H,W,C)
        pred = pred.squeeze(0).cpu().numpy().transpose(0,2,3,1) # (Tout,H,W,C)

        frames = []
        for i in range(len(pred)):
            row = np.concatenate([inp[-1], pred[i], gt[i]], axis=1)
            row = add_labels(row.copy(), ["Input", "Pred", "GT"])
            frames.append(row.astype(np.float32) / 255.0)
        pred_clips.append(frames)

    min_len = min(len(c) for c in pred_clips)
    pred_clips = [c[:min_len] for c in pred_clips]

    rows = int(np.ceil(np.sqrt(N)))
    cols = int(np.ceil(N / rows))
    h, w, c = pred_clips[0][0].shape

    out_frames = []
    for t in range(min_len):
        canvas = np.zeros((rows*h, cols*w, c), dtype=np.float32)
        for i, clip in enumerate(pred_clips):
            r, co = divmod(i, cols)
            canvas[r*h:(r+1)*h, co*w:(co+1)*w, :] = clip[t]
        out_frames.append(canvas)

    os.makedirs(args.outdir, exist_ok=True)
    out_path = os.path.join(args.outdir, "fpv_mosaic.mp4")
    make_video(out_frames, out_path, fps=args.fps)
    print(f"✅ Mosaico guardado en {out_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True, help="Ruta al checkpoint .pth")
    p.add_argument("--data", type=str, required=True, help="Ruta al .npz/.npy de test (fpv_test.npz)")
    p.add_argument("--cfg", type=str, default=None, help="Ruta al config .py usado en training (ej: ./configs/fpv/SimVP_gSTA.py)")
    p.add_argument("--pred_len", type=int, default=10, help="Cuántos frames futuros predecir")
    p.add_argument("--fps", type=int, default=10, help="FPS del video de salida")
    p.add_argument("--outdir", type=str, default="./results/vis", help="Carpeta de salida")
    p.add_argument("--max_samples", type=int, default=4, help="Número máximo de ejemplos en mosaico")
    args = p.parse_args()
    main(args)
