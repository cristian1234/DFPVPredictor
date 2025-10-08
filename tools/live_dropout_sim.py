import os, cv2, time, argparse, numpy as np, torch, runpy
from simvp.models.simvp_model import SimVP_Model

# ---------- Config helpers ----------
def load_cfg_from_py(cfg_path):
    g = runpy.run_path(cfg_path)
    for key in ('config','cfg','CONFIG','Config'):
        if key in g and isinstance(g[key], dict):
            return g[key]
    keys = ['in_shape','hid_S','hid_T','N_S','N_T','spatio_kernel_enc','spatio_kernel_dec','drop','drop_path','model_type']
    cfg = {k: g[k] for k in keys if k in g}
    if 'in_shape' in cfg:
        cfg.setdefault('hid_S', 64); cfg.setdefault('hid_T', 512)
        cfg.setdefault('N_S', 4); cfg.setdefault('N_T', 8)
        cfg.setdefault('spatio_kernel_enc', 3); cfg.setdefault('spatio_kernel_dec', 3)
        cfg.setdefault('drop', 0.0); cfg.setdefault('drop_path', 0.1)
        cfg.setdefault('model_type', 'gSTA')
        return cfg
    raise ValueError(f"No pude leer config útil de {cfg_path}")

def build_config_fallback(frame_shape_hw_c, pred_len=10, model_type='gSTA'):
    H,W,C = frame_shape_hw_c
    pre = max(1, 10)
    return {'in_shape':[pre,C,H,W],'hid_S':64,'hid_T':512,'N_S':4,'N_T':8,
            'spatio_kernel_enc':3,'spatio_kernel_dec':3,'drop':0.0,'drop_path':0.1,'model_type':model_type}

def choose_device():
    if torch.backends.mps.is_available(): return torch.device('mps')
    if torch.cuda.is_available(): return torch.device('cuda:0')
    return torch.device('cpu')

# ---------- I/O utils ----------
def to_model_tensor(frame_bgr, out_size_hw, use_rgb=True, half=False):
    H, W = out_size_hw
    f = cv2.resize(frame_bgr, (W, H), interpolation=cv2.INTER_AREA)
    if use_rgb: f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    x = f.astype(np.float16 if half else np.float32) / 255.0
    x = np.transpose(x, (2,0,1))          # CHW
    return x

def to_display_image(chw_float):
    x = np.clip(chw_float, 0, 1)
    x = (x * 255.0).astype(np.uint8)
    x = np.transpose(x, (1,2,0))          # HWC RGB
    x = x[:, :, ::-1]                     # RGB->BGR
    return x

def draw_hud(img_bgr, text, bg=(0,0,0), fg=(255,255,255), alpha=0.6):
    overlay = img_bgr.copy()
    H, W = img_bgr.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    # centrado arriba
    x0 = (W - tw) // 2 - 10
    y0 = 10
    x1 = x0 + tw + 20
    y1 = y0 + th + 10
    cv2.rectangle(overlay, (x0, y0), (x1, y1), bg, -1)
    out = cv2.addWeighted(overlay, alpha, img_bgr, 1 - alpha, 0)
    cv2.putText(out, text, (x0 + 10, y0 + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, fg, 2, cv2.LINE_AA)
    return out

def draw_button(img_bgr, rect, label, active=False):
    x0,y0,x1,y1 = rect
    color_bg = (40,40,40) if not active else (60,60,60)
    color_fg = (255,255,255)
    cv2.rectangle(img_bgr, (x0,y0), (x1,y1), color_bg, -1)
    cv2.rectangle(img_bgr, (x0,y0), (x1,y1), (180,180,180), 1)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    tx = x0 + (x1-x0 - tw)//2; ty = y0 + (y1-y0 + th)//2
    cv2.putText(img_bgr, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_fg, 2, cv2.LINE_AA)
    return img_bgr

# ---------- Buffer (solo reales) ----------
class RingBuffer:
    def __init__(self, capacity): self.capacity=capacity; self.data=[]
    def clear(self): self.data=[]
    def append(self, x):
        if len(self.data) >= self.capacity: self.data.pop(0)
        self.data.append(x)
    def ready(self): return len(self.data) >= self.capacity
    def as_tensor(self, device, half=False):
        arr = np.stack(self.data, axis=0)            # (T,C,H,W)
        t = torch.from_numpy(arr).unsqueeze(0)       # (1,T,C,H,W)
        if half: t = t.half()
        return t.to(device)

# ---------- Model forward ----------
def forward_pred(model, buf_tensor, use_amp=False, device_type='cpu'):
    with torch.no_grad():
        if use_amp and device_type == 'cuda':
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                return model(buf_tensor)
        else:
            return model(buf_tensor)

# ---------- Main ----------
def main(args):
    try: cv2.setNumThreads(max(1, os.cpu_count()//2))
    except: pass

    device = choose_device()
    print(f"Using device: {device}")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened(): raise FileNotFoundError(f"No pude abrir {args.video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    ok, first = cap.read()
    if not ok: raise RuntimeError("Video vacío")
    src_h, src_w = first.shape[:2]

    # Resolución interna (ajusta landscape si hace falta)
    target_h, target_w = args.in_size_y, args.in_size_x
    if (src_w >= src_h) and (target_w < target_h):
        print(f"[WARN] in_size_x/in_size_y invertidos; ajusto {target_h}x{target_w} -> {target_w}x{target_h}")
        target_h, target_w = target_w, target_h

    # Tamaño de display fijo (sin flicker)
    if args.no_upscale:
        base_w, base_h = target_w, target_h
    else:
        base_w, base_h = src_w, src_h
    disp_w, disp_h = int(base_w * args.display_scale), int(base_h * args.display_scale)
    display_size = (disp_w, disp_h)

    BTN_W, BTN_H, BTN_MARGIN = 110, 40, 10
    btn_rect_base = (BTN_MARGIN, BTN_MARGIN, BTN_MARGIN+BTN_W, BTN_MARGIN+BTN_H)

    fallback_cfg = build_config_fallback((target_h, target_w, 3), pred_len=args.pred_len)

    ckpt = torch.load(args.ckpt, map_location=device)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']; ckpt_cfg = ckpt.get('config', None)
    elif isinstance(ckpt, dict):
        state_dict = ckpt; ckpt_cfg = None
    else:
        raise ValueError("Formato de checkpoint no soportado")

    cfg = ckpt_cfg
    if cfg is None and args.cfg:
        try: cfg = load_cfg_from_py(args.cfg)
        except Exception as e:
            print(f"[WARN] No pude leer config del .py: {e}")
    if cfg is None:
        print("[WARN] Usando config fallback."); cfg = fallback_cfg

    C = 3
    cfg['in_shape'] = [args.pre_seq, C, target_h, target_w]
    cfg.setdefault('model_type','gSTA')

    model = SimVP_Model(**cfg).to(device)
    new_sd = {(k[7:] if k.startswith('module.') else k): v for k,v in state_dict.items()}
    model.load_state_dict(new_sd, strict=False)
    if args.half: model.half()
    model.eval()

    # Warmup
    _ = forward_pred(model, torch.zeros(1, args.pre_seq, C, target_h, target_w,
                                        device=device, dtype=torch.float16 if args.half else torch.float32),
                     use_amp=args.amp, device_type=device.type)

    buf_real = RingBuffer(args.pre_seq)
    buf_real.append(to_model_tensor(first, (target_h, target_w), use_rgb=True, half=args.half))

    window_name = "FPV Live (SPACE=drop, I=auto, P=pause, Q=quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, display_size[0], display_size[1])

    state = dict(
        paused=False, dropping=False, drop_request=False,
        remaining_drop=0, pred_cache=None, pred_idx=0,
        hold_frame=cv2.resize(first.copy(), display_size),
        infer_mode=False
    )

    target_fps = args.fps if args.fps > 0 else src_fps
    frame_interval_ms = 1000.0 / max(1.0, target_fps)
    last_ms = time.time() * 1000.0
    last_space_ms = 0

    ema_fps = None
    cap.set(cv2.CAP_PROP_POS_FRAMES, 1)

    while True:
        start_loop = time.time()
        now_ms = start_loop * 1000.0
        if now_ms - last_ms < frame_interval_ms:
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'): break
            elif k == ord('p'): state['paused'] = not state['paused']
            elif k == ord(' '):
                if now_ms - last_space_ms >= args.debounce_ms:
                    state['drop_request'] = True; last_space_ms = now_ms
            elif k == ord('i'):
                state['infer_mode'] = not state['infer_mode']
                print(f"[INFO] Inferencia automática {'ON' if state['infer_mode'] else 'OFF'}.")
            continue
        last_ms = now_ms

        # --- PAUSE ---
        if state['paused']:
            disp = state['hold_frame'].copy()
            hud = f"PAUSE | {device.type.upper()} | in:{target_w}x{target_h} | AUTO:{'ON' if state['infer_mode'] else 'OFF'}"
            if ema_fps is not None: hud += f" | proc:{ema_fps:.1f}fps"
            if args.hud: disp = draw_hud(disp, hud)
            disp = draw_button(disp, btn_rect_base, "PLAY", active=True)
            cv2.imshow(window_name, disp)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'): break
            elif k == ord('p'): state['paused'] = not state['paused']
            elif k == ord('i'):
                state['infer_mode'] = not state['infer_mode']
                print(f"[INFO] Inferencia automática {'ON' if state['infer_mode'] else 'OFF'}.")
            continue

        # --- LIVE (cuando infer_mode está OFF) ---
        if not state['dropping'] and not state['infer_mode']:
            ok, frame = cap.read()
            if not ok: break
            disp = cv2.resize(frame.copy(), display_size, interpolation=cv2.INTER_LINEAR)
            display_fps = target_fps if target_fps > 0 else src_fps
            hud = f"LIVE @{display_fps:.1f}fps | {device.type.upper()} | in:{target_w}x{target_h}"
            if ema_fps is not None: hud += f" (proc:{ema_fps:.1f}fps)"
            if args.hud: disp = draw_hud(disp, hud)
            disp = draw_button(disp, btn_rect_base, "PAUSE", active=False)
            cv2.imshow(window_name, disp)

            buf_real.append(to_model_tensor(frame, (target_h, target_w), use_rgb=True, half=args.half))
            state['hold_frame'] = disp.copy()

            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'): break
            elif k == ord('p'): state['paused'] = True
            elif k == ord(' '):
                if now_ms - last_space_ms >= args.debounce_ms:
                    state['drop_request'] = True; last_space_ms = now_ms
            elif k == ord('i'):
                state['infer_mode'] = True
                print("[INFO] Inferencia automática ON.")

            if (state['drop_request']) and buf_real.ready():
                state['dropping'] = True
                state['remaining_drop'] = args.drop_len
                state['pred_cache'] = None
                state['pred_idx'] = 0
                state['drop_request'] = False

            continue

        # --- AUTO WARMUP: llenar buffer real sin mostrar frames reales ---
        if state['infer_mode'] and not buf_real.ready():
            ok, frame = cap.read()
            if not ok: break
            buf_real.append(to_model_tensor(frame, (target_h, target_w), use_rgb=True, half=args.half))
            disp = state['hold_frame'].copy()
            hud = f"AUTO WARMUP… {len(buf_real.data)}/{args.pre_seq} | {device.type.upper()} | in:{target_w}x{target_h}"
            if args.hud: disp = draw_hud(disp, hud)
            disp = draw_button(disp, btn_rect_base, "PAUSE", active=False)
            cv2.imshow(window_name, disp)
            continue

        # --- si infer_mode ON y buffer listo, predicción continua con ventana de reales (STRIDE 1) ---
        if state['infer_mode']:
            # STRIDE 1: recomputar SIEMPRE desde los últimos reales y tomar SOLO el primer paso (t+1)
            if args.auto_stride1:
                inp = buf_real.as_tensor(device, half=args.half)
                out = forward_pred(model, inp, use_amp=args.amp, device_type=device.type)
                # usamos sólo el primer frame predicho (t+1)
                pred_chw = out[0, 0].detach().float().cpu().numpy()
                pred_chw = np.clip(pred_chw, 0.0, 1.0)

            else:
                # MODO BLOQUE (antiguo): mantiene el comportamiento previo por si querés compararlo
                if (state['pred_cache'] is None) or (state['pred_idx'] >= args.pred_len):
                    inp = buf_real.as_tensor(device, half=args.half)
                    state['pred_cache'] = forward_pred(model, inp, use_amp=args.amp, device_type=device.type)
                    state['pred_idx'] = 0
                pred_chw = state['pred_cache'][0, state['pred_idx']].detach().float().cpu().numpy()
                pred_chw = np.clip(pred_chw, 0.0, 1.0)
                state['pred_idx'] += 1

            # mostrar pred (siempre al mismo tamaño de ventana)
            pred_bgr = to_display_image(pred_chw)
            pred_bgr = cv2.resize(pred_bgr, display_size, interpolation=cv2.INTER_LINEAR)
            hud = f"AUTO PRED | {device.type.upper()} | in:{target_w}x{target_h}"
            if ema_fps is not None: hud += f" | proc:{ema_fps:.1f}fps"
            if args.hud: pred_bgr = draw_hud(pred_bgr, hud)
            pred_bgr = draw_button(pred_bgr, btn_rect_base, "PAUSE", active=False)
            cv2.imshow(window_name, pred_bgr)
            state['hold_frame'] = pred_bgr.copy()

            # avanzar EXACTAMENTE 1 frame real y actualizar ventana real (teacher forcing real)
            ok, frame = cap.read()
            if not ok:
                break
            buf_real.append(to_model_tensor(frame, (target_h, target_w), use_rgb=True, half=args.half))

            # teclado
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'): break
            elif k == ord('p'):
                state['paused'] = True
            elif k == ord('i'):
                state['infer_mode'] = False
                state['pred_cache'] = None
                print("[INFO] Inferencia automática OFF.")
            continue

        # --- DROP manual (cuando infer_mode OFF) ---
        if state['dropping']:
            if (state['pred_cache'] is None) or (state['pred_idx'] >= args.pred_len):
                inp = buf_real.as_tensor(device, half=args.half)
                state['pred_cache'] = forward_pred(model, inp, use_amp=args.amp, device_type=device.type)
                state['pred_idx'] = 0

            pred_chw = state['pred_cache'][0, state['pred_idx']].detach().float().cpu().numpy()
            pred_chw = np.clip(pred_chw, 0.0, 1.0)
            state['pred_idx'] += 1
            state['remaining_drop'] -= 1

            pred_bgr = to_display_image(pred_chw)
            pred_bgr = cv2.resize(pred_bgr, display_size, interpolation=cv2.INTER_LINEAR)
            hud = f"DROP {args.drop_len - max(0,state['remaining_drop'])}/{args.drop_len} | {device.type.upper()} | in:{target_w}x{target_h}"
            if ema_fps is not None: hud += f" | proc:{ema_fps:.1f}fps"
            if args.hud: pred_bgr = draw_hud(pred_bgr, hud)
            pred_bgr = draw_button(pred_bgr, btn_rect_base, "PAUSE", active=False)
            cv2.imshow(window_name, pred_bgr)
            state['hold_frame'] = pred_bgr.copy()

            # avanzar 1 real para mantener ventana
            ok, frame = cap.read()
            if not ok: break
            buf_real.append(to_model_tensor(frame, (target_h, target_w), use_rgb=True, half=args.half))

            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'): break
            elif k == ord('p'):
                state['paused'] = True; state['dropping'] = False

            if state['remaining_drop'] <= 0:
                state['dropping'] = False
                state['pred_cache'] = None
            continue

        # fallback
        time.sleep(0.001)

        # FPS medido
        dt = time.time() - start_loop
        cur_fps = 1.0 / max(1e-6, dt)
        ema_fps = cur_fps if ema_fps is None else (0.9*ema_fps + 0.1*cur_fps)

    cap.release(); cv2.destroyAllWindows(); print("Bye.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--cfg",  type=str, default=None)
    p.add_argument("--video", type=str, required=True)
    p.add_argument("--pre_seq", type=int, default=8)
    p.add_argument("--pred_len", type=int, default=4)
    p.add_argument("--drop_len", type=int, default=4)
    # Landscape por defecto (256x144). Si lo pasás al revés, el script lo corrige.
    p.add_argument("--in_size_y", type=int, default=144)
    p.add_argument("--in_size_x", type=int, default=256)
    p.add_argument("--fps", type=int, default=0)
    p.add_argument("--display_scale", type=int, default=1)
    p.add_argument("--hud", type=int, default=1)
    p.add_argument("--auto_stride1", type=int, default=1,
                   help="En modo I (auto), recalcular cada frame y mostrar solo el paso t+1 con los últimos pre_seq reales. Elimina el efecto ‘por bloques’.")
    p.add_argument("--debounce_ms", type=int, default=200)
    p.add_argument("--amp", type=int, default=0, help="autocast fp16 (CUDA). En MPS no aplica")
    p.add_argument("--half", type=int, default=1, help="modelo/inputs en fp16")
    p.add_argument("--no_upscale", type=int, default=0, help="mostrar a resolución interna del modelo")
    args = p.parse_args()
    main(args)
