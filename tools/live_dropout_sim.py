import os
import cv2
import time
import argparse
import numpy as np
import torch
import runpy

from simvp.models.simvp_model import SimVP_Model

# ---------- Config helpers ----------
def load_cfg_from_py(cfg_path):
    g = runpy.run_path(cfg_path)
    for key in ('config', 'cfg', 'CONFIG', 'Config'):
        if key in g and isinstance(g[key], dict):
            return g[key]
    keys = ['in_shape','hid_S','hid_T','N_S','N_T','spatio_kernel_enc','spatio_kernel_dec','drop','drop_path','model_type']
    cfg = {k: g[k] for k in keys if k in g}
    if 'in_shape' in cfg:
        cfg.setdefault('hid_S', 64); cfg.setdefault('hid_T', 512)
        cfg.setdefault('N_S', 4);    cfg.setdefault('N_T', 8)
        cfg.setdefault('spatio_kernel_enc', 3); cfg.setdefault('spatio_kernel_dec', 3)
        cfg.setdefault('drop', 0.0); cfg.setdefault('drop_path', 0.1)
        cfg.setdefault('model_type', 'gSTA')
        return cfg
    raise ValueError(f"No pude encontrar ni dict ni variables útiles en {cfg_path}")

def build_config_fallback(frame_shape_hw_c, pred_len=10, model_type='gSTA'):
    H, W, C = frame_shape_hw_c
    pre = max(1, 10)
    return {
        'in_shape': [pre, C, H, W],
        'hid_S': 64, 'hid_T': 512, 'N_S': 4, 'N_T': 8,
        'spatio_kernel_enc': 3, 'spatio_kernel_dec': 3,
        'drop': 0.0, 'drop_path': 0.1, 'model_type': model_type
    }

def choose_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    return torch.device('cpu')

# ---------- I/O utils ----------
def to_model_tensor(frame_bgr, out_size_hw, use_rgb=True):
    H, W = out_size_hw
    f = cv2.resize(frame_bgr, (W, H), interpolation=cv2.INTER_AREA)
    if use_rgb:
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    x = f.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))  # CHW
    return x

def to_display_image(chw_float):
    x = np.clip(chw_float, 0, 1)
    x = (x * 255.0).astype(np.uint8)
    x = np.transpose(x, (1, 2, 0))  # HWC RGB
    x = x[:, :, ::-1]               # RGB->BGR
    return x

def draw_hud(img_bgr, text, bg=(0,0,0), fg=(255,255,255), alpha=0.6):
    overlay = img_bgr.copy()
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(overlay, (8, 8), (18 + tw, 18 + th), bg, -1)
    out = cv2.addWeighted(overlay, alpha, img_bgr, 1 - alpha, 0)
    cv2.putText(out, text, (12, 12 + th - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, fg, 2, cv2.LINE_AA)
    return out

def draw_button(img_bgr, rect, label, active=False):
    x0,y0,x1,y1 = rect
    color_bg = (40,40,40) if not active else (60,60,60)
    color_fg = (255,255,255)
    cv2.rectangle(img_bgr, (x0,y0), (x1,y1), color_bg, -1)
    cv2.rectangle(img_bgr, (x0,y0), (x1,y1), (180,180,180), 1)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    tx = x0 + (x1-x0 - tw)//2
    ty = y0 + (y1-y0 + th)//2
    cv2.putText(img_bgr, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_fg, 2, cv2.LINE_AA)
    return img_bgr

# ---------- Buffer ----------
class RingBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.data = []
    def clear(self): self.data = []
    def append(self, x):
        if len(self.data) >= self.capacity:
            self.data.pop(0)
        self.data.append(x)
    def ready(self):
        return len(self.data) >= self.capacity
    def as_tensor(self, device):
        arr = np.stack(self.data, axis=0)  # (T,C,H,W)
        return torch.from_numpy(arr).unsqueeze(0).to(device)  # (1,T,C,H,W)

# ---------- Model forward ----------
def forward_pred(model, buf_tensor):
    with torch.no_grad():
        return model(buf_tensor)  # (1, pred_len, C, H, W)

# ---------- Main ----------
def main(args):
    device = choose_device()
    print(f"Using device: {device}")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"No pude abrir {args.video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    ok, first = cap.read()
    if not ok:
        raise RuntimeError("Video vacío")

    # Tamaño FUENTE para mostrar SIEMPRE igual en ambos modos
    src_h, src_w = first.shape[:2]
    disp_w, disp_h = src_w * args.display_scale, src_h * args.display_scale

    # UI: botón pausa/play en coordenadas de la imagen fuente
    BTN_W, BTN_H = 110, 40
    BTN_MARGIN = 10
    btn_rect_base = (BTN_MARGIN, BTN_MARGIN, BTN_MARGIN+BTN_W, BTN_MARGIN+BTN_H)

    # Tamaño para el modelo (p.ej. 64x64)
    target_h, target_w = args.in_size, args.in_size
    fallback_cfg = build_config_fallback((target_h, target_w, 3), pred_len=args.pred_len)

    # Cargar ckpt/config
    ckpt = torch.load(args.ckpt, map_location=device)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']; ckpt_cfg = ckpt.get('config', None)
    elif isinstance(ckpt, dict):
        state_dict = ckpt; ckpt_cfg = None
    else:
        raise ValueError("Formato de checkpoint no soportado")

    cfg = ckpt_cfg
    if cfg is None and args.cfg:
        try:
            cfg = load_cfg_from_py(args.cfg)
        except Exception as e:
            print(f"[WARN] No pude leer config del .py: {e}")
    if cfg is None:
        print("[WARN] Usando config fallback.")
        cfg = fallback_cfg

    # Forzar in_shape coherente con CLI
    C = 3
    cfg['in_shape'] = cfg.get('in_shape', [args.pre_seq, C, target_h, target_w])
    cfg['in_shape'][0] = args.pre_seq
    cfg['in_shape'][1] = C
    cfg['in_shape'][2] = target_h
    cfg['in_shape'][3] = target_w
    cfg.setdefault('model_type', 'gSTA')

    model = SimVP_Model(**cfg).to(device)
    new_sd = { (k[7:] if k.startswith('module.') else k): v for k,v in state_dict.items() }
    model.load_state_dict(new_sd, strict=False)
    model.eval()

    # Warmup
    _ = forward_pred(model, torch.zeros(1, args.pre_seq, C, target_h, target_w, device=device))

    # Estado
    buf = RingBuffer(args.pre_seq)
    buf.append(to_model_tensor(first, (target_h, target_w), use_rgb=True))

    window_name = "FPV Live (SPACE=drop, p/pause, q/quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, disp_w, disp_h)

    state = {
        'seek_request': False,
        'seek_target': 0,
        'paused': False,
        'dropping': False,
        'drop_request': False,
        'remaining_drop': 0,
        'pred_cache': None,
        'pred_idx': 0,
        'hold_frame': first.copy(),
        'suppress_seek_cb': False,   # <-- evita pausa involuntaria
    }

    # --- Trackbar de seek ---
    def on_seek(val):
        if state['suppress_seek_cb']:
            return
        state['seek_target'] = int(val)
        state['seek_request'] = True

    if total_frames > 0:
        cv2.createTrackbar('seek', window_name, 0, total_frames-1, on_seek)

    # --- Mouse para botón PAUSE/PLAY ---
    def on_mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Escalar rect al tamaño de ventana mostrado
            scale = args.display_scale
            x0,y0,x1,y1 = btn_rect_base
            xr = (x0*scale, y0*scale, x1*scale, y1*scale)
            if xr[0] <= x <= xr[2] and xr[1] <= y <= xr[3]:
                state['paused'] = not state['paused']
                # Si pasamos a PLAY, limpiamos drop_request pendiente
                if not state['paused']:
                    state['drop_request'] = False
                # Si pausamos, cancelamos drop activo
                if state['paused'] and state['dropping']:
                    state['dropping'] = False

    cv2.setMouseCallback(window_name, on_mouse)

    # control de fps + debounce
    target_fps = args.fps if args.fps > 0 else src_fps
    frame_interval_ms = 1000.0 / max(1.0, target_fps)
    last_ms = time.time() * 1000.0
    last_space_ms = 0

    # continuar desde el 2do frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 1)

    while True:
        # throttle
        now_ms = time.time() * 1000.0
        if now_ms - last_ms < frame_interval_ms:
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            elif k == ord('p'):
                state['paused'] = not state['paused']
                if state['paused'] and state['dropping']:
                    state['dropping'] = False
            elif k == ord(' '):
                if now_ms - last_space_ms >= args.debounce_ms:
                    state['drop_request'] = True
                    last_space_ms = now_ms
            continue
        last_ms = now_ms

        # --- handle seek ---
        if state['seek_request']:
            # cancelar dropout
            state['dropping'] = False
            state['pred_cache'] = None
            state['pred_idx'] = 0
            # reposicionar: reconstruir buffer hasta seek_target
            target = max(0, state['seek_target'])
            start = max(0, target - args.pre_seq + 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            buf.clear()
            last_frame = None
            while True:
                pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                ok, frm = cap.read()
                if not ok:
                    break
                buf.append(to_model_tensor(frm, (target_h, target_w), use_rgb=True))
                last_frame = frm
                if pos >= target:
                    break
            if last_frame is not None:
                state['hold_frame'] = last_frame.copy()
            # actualizar trackbar (suprimimos callback para que NO pause)
            if total_frames > 0:
                state['suppress_seek_cb'] = True
                try:
                    cv2.setTrackbarPos('seek', window_name, min(target, total_frames-1))
                finally:
                    state['suppress_seek_cb'] = False
            # queda pausado tras seek
            state['paused'] = True
            state['seek_request'] = False

        # --- si pausado: mostrar hold_frame y seguir capturando eventos ---
        if state['paused']:
            disp = state['hold_frame'].copy()
            if args.hud:
                disp = draw_hud(disp, "PAUSE")
            label = "PLAY" if state['paused'] else "PAUSE"
            disp = draw_button(disp, btn_rect_base, label, active=state['paused'])
            if args.display_scale != 1:
                disp = cv2.resize(disp, (disp_w, disp_h), interpolation=cv2.INTER_NEAREST)
            cv2.imshow(window_name, disp)

            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            elif k == ord('p'):
                state['paused'] = not state['paused']
                if not state['paused']:
                    state['drop_request'] = False
            elif k == ord(' '):
                state['drop_request'] = True
            continue

        # --- modo reproducción normal / dropout ---
        if not state['dropping']:
            ok, frame = cap.read()
            if not ok:
                break

            disp = frame.copy()
            if args.hud:
                disp = draw_hud(disp, "LIVE")
            label = "PAUSE"
            disp = draw_button(disp, btn_rect_base, label, active=False)

            if args.display_scale != 1:
                disp = cv2.resize(disp, (disp_w, disp_h), interpolation=cv2.INTER_NEAREST)
            cv2.imshow(window_name, disp)

            # actualizar buffer para el modelo
            buf.append(to_model_tensor(frame, (target_h, target_w), use_rgb=True))
            state['hold_frame'] = frame.copy()

            # actualizar trackbar (suprimido el callback)
            if total_frames > 0:
                cur = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                cur = max(0, min(cur, total_frames-1))
                state['suppress_seek_cb'] = True
                try:
                    cv2.setTrackbarPos('seek', window_name, cur)
                finally:
                    state['suppress_seek_cb'] = False

            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            elif k == ord('p'):
                state['paused'] = True
                continue
            elif k == ord(' '):
                if now_ms - last_space_ms >= args.debounce_ms:
                    state['drop_request'] = True
                    last_space_ms = now_ms

            # ¿debo entrar en dropout?
            if state['drop_request'] and buf.ready():
                state['dropping'] = True
                state['remaining_drop'] = args.drop_len
                state['pred_cache'] = None
                state['pred_idx'] = 0
                state['drop_request'] = False

        else:
            if (state['pred_cache'] is None) or (state['pred_idx'] >= state['pred_cache'].shape[1]):
                inp = buf.as_tensor(device)
                state['pred_cache'] = forward_pred(model, inp)
                state['pred_idx'] = 0

            pred_chw = state['pred_cache'][0, state['pred_idx']].detach().cpu().numpy()
            state['pred_idx'] += 1
            state['remaining_drop'] -= 1

            pred_bgr = to_display_image(pred_chw)
            pred_bgr = cv2.resize(pred_bgr, (first.shape[1], first.shape[0]), interpolation=cv2.INTER_LINEAR)
            if args.hud:
                pred_bgr = draw_hud(pred_bgr, f"DROP {args.drop_len - state['remaining_drop']}/{args.drop_len}")
            label = "PAUSE"
            pred_bgr = draw_button(pred_bgr, btn_rect_base, label, active=False)

            if args.display_scale != 1:
                pred_bgr = cv2.resize(pred_bgr, (disp_w, disp_h), interpolation=cv2.INTER_NEAREST)
            cv2.imshow(window_name, pred_bgr)

            buf.append(pred_chw)
            state['hold_frame'] = pred_bgr.copy()

            _ = cap.read()

            if total_frames > 0:
                cur = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                cur = max(0, min(cur, total_frames-1))
                state['suppress_seek_cb'] = True
                try:
                    cv2.setTrackbarPos('seek', window_name, cur)
                finally:
                    state['suppress_seek_cb'] = False

            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            elif k == ord('p'):
                state['paused'] = True
                state['dropping'] = False
                continue

            if state['remaining_drop'] <= 0:
                state['dropping'] = False

    cap.release()
    cv2.destroyAllWindows()
    print("Bye.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--cfg",  type=str, default=None)
    p.add_argument("--video", type=str, required=True)
    p.add_argument("--pre_seq", type=int, default=10)
    p.add_argument("--pred_len", type=int, default=10)
    p.add_argument("--drop_len", type=int, default=10)
    p.add_argument("--in_size", type=int, default=64)
    p.add_argument("--fps", type=int, default=0)
    p.add_argument("--display_scale", type=int, default=1)   # 1 = tamaño fuente
    p.add_argument("--hud", type=int, default=1)             # 1 = ON, 0 = OFF
    p.add_argument("--debounce_ms", type=int, default=200)   # evita autorepeat del SO
    args = p.parse_args()
    main(args)
