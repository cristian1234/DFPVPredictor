# tools/live_dropout_sim_optimized.py
import cv2, time, argparse, numpy as np, torch, runpy

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
        cfg.setdefault('N_S', 4);    cfg.setdefault('N_T', 8)
        cfg.setdefault('spatio_kernel_enc', 3); cfg.setdefault('spatio_kernel_dec', 3)
        cfg.setdefault('drop', 0.0); cfg.setdefault('drop_path', 0.1)
        cfg.setdefault('model_type', 'gSTA')
        return cfg
    raise ValueError(f"No pude leer config útil de {cfg_path}")

def build_config_fallback(frame_shape_hw_c, pred_len=10, model_type='gSTA'):
    H,W,C = frame_shape_hw_c
    pre = max(1, pred_len)  # por si acaso
    return {
        'in_shape':[pre,C,H,W],
        'hid_S':64,'hid_T':512,'N_S':4,'N_T':8,
        'spatio_kernel_enc':3,'spatio_kernel_dec':3,
        'drop':0.0,'drop_path':0.1,'model_type':model_type
    }

def choose_device():
    if torch.backends.mps.is_available():
        try:
            torch.backends.mps.set_per_process_memory_fraction(0.9)
        except Exception:
            pass
        return torch.device('mps')
    if torch.cuda.is_available(): return torch.device('cuda:0')
    return torch.device('cpu')

# ---------- I/O utils ----------
def to_model_tensor_torch(frame_bgr, out_size_hw, use_rgb=True, half=False):
    """Devuelve CHW torch tensor en [0,1], channels_last friendly (se transforma luego)."""
    H, W = out_size_hw
    if frame_bgr.shape[0] != H or frame_bgr.shape[1] != W:
        f = cv2.resize(frame_bgr, (W, H), interpolation=cv2.INTER_AREA)
    else:
        f = frame_bgr
    if use_rgb: f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)  # HWC
    x = torch.from_numpy(f)  # uint8 HWC
    x = x.to(torch.float16 if half else torch.float32).div_(255.0)
    x = x.permute(2,0,1).contiguous()  # CHW
    return x

def to_display_image(chw_float):
    # acepta Tensor o Numpy
    if isinstance(chw_float, torch.Tensor):
        x = chw_float.detach().cpu().float().numpy()
    else:
        x = np.array(chw_float, dtype=np.float32)

    x = np.clip(x, 0, 1)
    x = (x * 255.0).astype(np.uint8)

    if x.ndim == 3:
        x = np.transpose(x, (1, 2, 0))  # CHW → HWC
    else:
        raise ValueError(f"Esperaba array 3D CHW, obtuve {x.shape}")

    if x.shape[2] == 3:
        x = x[:, :, ::-1]  # RGB → BGR

    return x

def draw_hud(img_bgr, text, bg=(0,0,0), fg=(255,255,255), alpha=0.6):
    overlay = img_bgr.copy()
    H, W = img_bgr.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    x0 = max(8, (W - tw)//2 - 10); y0 = 8
    x1 = min(W-8, x0 + tw + 20);  y1 = y0 + th + 12
    cv2.rectangle(overlay, (x0, y0), (x1, y1), bg, -1)
    out = cv2.addWeighted(overlay, alpha, img_bgr, 1 - alpha, 0)
    cv2.putText(out, text, (x0 + 10, y0 + th + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, fg, 2, cv2.LINE_AA)
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

# ---------- Buffer de reales ----------
class RingBuffer:
    def __init__(self, capacity):
        self.capacity=capacity
        self.data=[]
    def clear(self): self.data=[]
    def append(self, x):
        if len(self.data) >= self.capacity: self.data.pop(0)
        self.data.append(x)
    def ready(self): return len(self.data) >= self.capacity
    def as_tensor(self, device, half=False, channels_last=False):
        # stack en torch para evitar ida/vuelta a numpy
        if len(self.data) == 0:
            return None
        t = torch.stack(self.data, dim=0)  # (T,C,H,W)
        t = t.unsqueeze(0)                 # (1,T,C,H,W)
        if half: t = t.half()
        if channels_last:
            # N T C H W -> seguimos N T C H W, channels_last se aplica tras permute dentro del modelo
            pass
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
    # OpenCV threads (reduce peleas con MPS)
    try: cv2.setNumThreads(max(1, args.resize_threads))
    except: pass

    device = choose_device()
    print(f"Using device: {device}")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened(): raise FileNotFoundError(f"No pude abrir {args.video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    ok, first = cap.read()
    if not ok: raise RuntimeError("Video vacío")
    src_h, src_w = first.shape[:2]

    # Resolución interna coherente con landscape
    target_h, target_w = args.in_size_y, args.in_size_x
    if (src_w >= src_h) and (target_w < target_h):
        print(f"[WARN] in_size_x/in_size_y invertidos; ajusto {target_h}x{target_w} -> {target_w}x{target_h}")
        target_h, target_w = target_w, target_h

    # Tamaño de display
    if args.no_upscale: base_w, base_h = target_w, target_h
    else:               base_w, base_h = src_w, src_h
    disp_w, disp_h = int(base_w * args.display_scale), int(base_h * args.display_scale)
    display_size = (disp_w, disp_h)

    # UI
    BTN_W, BTN_H, BTN_MARGIN = 110, 40, 10
    btn_rect_base = (BTN_MARGIN, BTN_MARGIN, BTN_MARGIN+BTN_W, BTN_MARGIN+BTN_H)

    # Cargar cfg y modelo
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
    model = model.to(memory_format=torch.channels_last)
    new_sd = {(k[7:] if k.startswith('module.') else k): v for k,v in state_dict.items()}
    model.load_state_dict(new_sd, strict=False)
    if args.half: model.half()
    model.eval()

    # Warmup (compila kernels)
    _ = forward_pred(
        model,
        torch.zeros(1, args.pre_seq, C, target_h, target_w, device=device,
                    dtype=torch.float16 if args.half else torch.float32),
        use_amp=args.amp, device_type=device.type
    )

    # Estado
    buf_real = RingBuffer(args.pre_seq)
    # primer frame al buffer
    buf_real.append(to_model_tensor_torch(first, (target_h, target_w), use_rgb=True, half=args.half))

    window_name = "FPV Live (SPACE=drop, I=auto, P=pause, Q=quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, display_size[0], display_size[1])

    state = dict(
        paused=False,
        infer_mode=False,      # I para ON/OFF
        hold_frame=cv2.resize(first.copy(), display_size),
        pred_cache=None,       # tensor (1, pred_len, C, H, W)
        pred_idx=0,
        reals_since_last_pred=0
    )

    target_fps = args.fps if args.fps > 0 else src_fps
    frame_interval_ms = 1000.0 / max(1.0, target_fps)
    last_ms = time.time() * 1000.0

    # métricas FPS
    ema_live_fps = None
    ema_proc_fps = None
    last_loop_t = time.time()

    # stride de bloque: por defecto = pred_len
    block_stride = args.block_stride if args.block_stride > 0 else args.pred_len

    # empezar desde el 2do frame (ya consumimos el primero)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 1)

    def show_frame(img_bgr, hud_text, play_active=False):
        if args.hud: img_bgr = draw_hud(img_bgr, hud_text)
        img_bgr = draw_button(img_bgr, btn_rect_base, "PAUSE" if not play_active else "PLAY", active=play_active)
        cv2.imshow(window_name, img_bgr)

    # 🧠 async setup
    import concurrent.futures
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    next_future = None
    forward_start = None
    pending_out = None  # 🧠 NUEVO: resultado listo para el próximo bloque

    while True:
        loop_start = time.time()
        now_ms = loop_start * 1000.0
        if now_ms - last_ms < frame_interval_ms:
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'): break
            elif k == ord('p'): state['paused'] = not state['paused']
            elif k == ord('i'):
                state['infer_mode'] = not state['infer_mode']
                state['pred_cache'] = None; state['pred_idx'] = 0; state['reals_since_last_pred'] = 0
                next_future = None  # 🧠 reset async
                print(f"[INFO] Inferencia automática {'ON' if state['infer_mode'] else 'OFF'}.")
            continue
        last_ms = now_ms

        # =================== PAUSA ===================
        if state['paused']:
            disp = state['hold_frame'].copy()
            hud = f"PAUSE | {device.type.upper()} | in:{target_w}x{target_h}"
            if ema_live_fps is not None: hud += f" | LIVE FPS:{ema_live_fps:.1f}"
            if ema_proc_fps is not None: hud += f" | PROC FPS:{ema_proc_fps:.1f}"
            show_frame(disp, hud, play_active=True)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'): break
            elif k == ord('p'): state['paused'] = not state['paused']
            elif k == ord('i'):
                state['infer_mode'] = not state['infer_mode']
                state['pred_cache'] = None; state['pred_idx'] = 0; state['reals_since_last_pred'] = 0
                next_future = None
            continue

        # ============== MODO LIVE (sin inferencia continua) ==============
        if not state['infer_mode']:
            ok, frame = cap.read()
            if not ok:
                # si se acabó el video, mantené último frame visible
                disp = state['hold_frame'].copy()
                hud = f"LIVE END | {device.type.upper()} | in:{target_w}x{target_h}"
                show_frame(disp, hud, play_active=False)
                break

            # asegurar que haya algo para mostrar
            if frame is None or frame.size == 0:
                disp = state['hold_frame'].copy()
            else:
                disp = frame if args.no_upscale else cv2.resize(frame, display_size, interpolation=cv2.INTER_LINEAR)
                state['hold_frame'] = disp.copy()

            # HUD live
            hud = f"LIVE @{(target_fps if target_fps>0 else src_fps):.1f}fps | {device.type.upper()} | in:{target_w}x{target_h}"
            if ema_live_fps is not None: hud += f" | LIVE FPS:{ema_live_fps:.1f}"
            if ema_proc_fps is not None: hud += f" | PROC FPS:{ema_proc_fps:.1f}"
            show_frame(disp, hud, play_active=False)

            # mantener ventana de reales
            t_in = to_model_tensor_torch(frame, (target_h, target_w), use_rgb=True, half=args.half)
            buf_real.append(t_in)

            # teclado
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'): break
            elif k == ord('p'): state['paused'] = True
            elif k == ord('i'):
                state['infer_mode'] = True
                state['pred_cache'] = None; state['pred_idx'] = 0; state['reals_since_last_pred'] = 0
                next_future = None
                print("[INFO] Inferencia automática ON.")

            # medir live FPS
            dt_live = loop_start - last_loop_t
            live_fps = 1.0 / max(dt_live, 1e-6)
            ema_live_fps = live_fps if ema_live_fps is None else (0.9*ema_live_fps + 0.1*live_fps)
            last_loop_t = loop_start
            continue

        # ============== MODO INFERENCIA CONTINUA (solo predicciones) ==============
        if state['infer_mode']:
            # 🧠 1) Lanzar un forward si no hay trabajo en curso ni resultado pendiente
            if next_future is None and pending_out is None and buf_real.ready():
                inp = buf_real.as_tensor(device, half=args.half, channels_last=True)
                inp = inp.contiguous()
                next_future = executor.submit(
                    forward_pred, model, inp, use_amp=args.amp, device_type=device.type
                )
                forward_start = time.time()

            # 🧠 2) Si terminó el future, guardar resultado en "pending_out" (NO tocar el bloque actual aún)
            if next_future is not None and next_future.done():
                out = next_future.result()
                forward_time = time.time() - forward_start
                print(f"[DBG] async forward {args.pred_len} frames = {forward_time*1000:.2f} ms")
                pending_out = out
                next_future = None
                proc_fps = float(args.pred_len) / max(forward_time, 1e-6)
                ema_proc_fps = proc_fps if ema_proc_fps is None else (0.8*ema_proc_fps + 0.2*proc_fps)

            # 🧠 3) Si no hay bloque activo, pero ya hay uno pendiente, empezalo ahora
            if state['pred_cache'] is None and pending_out is not None:
                state['pred_cache'] = pending_out
                state['pred_idx'] = 0
                pending_out = None
                # opcional: lanzar ya el siguiente prefetch
                if next_future is None and buf_real.ready():
                    inp = buf_real.as_tensor(device, half=args.half, channels_last=True).contiguous()
                    next_future = executor.submit(
                        forward_pred, model, inp, use_amp=args.amp, device_type=device.type
                    )
                    forward_start = time.time()

            # --- mostrar frame predicho actual o mantener último ---
            if state['pred_cache'] is not None and state['pred_idx'] < args.pred_len:
                # tenemos frame nuevo del bloque actual
                pred_chw = state['pred_cache'][0, state['pred_idx']].detach().float().cpu().numpy()
                pred_chw = np.clip(pred_chw, 0.0, 1.0)
                pred_bgr = to_display_image(pred_chw)
                disp = pred_bgr if args.no_upscale else cv2.resize(
                    pred_bgr, display_size, interpolation=cv2.INTER_LINEAR
                )

                # 🟥 marcar el primer frame del bloque (una sola vez por bloque)
                if state['pred_idx'] == 0:
                    disp = np.ascontiguousarray(disp, dtype=np.uint8)  # ✅ asegura compatibilidad OpenCV
                    h, w = disp.shape[:2]
                    cv2.rectangle(disp, (3, 3), (w - 4, h - 4), (0, 0, 255), 4)
                    cv2.putText(disp, "BLOCK START", (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, cv2.LINE_AA)

                state['hold_frame'] = disp.copy()
                state['pred_idx'] += 1
            else:
                # no hay bloque listo aún → mostrar último frame guardado (sin negro)
                disp = state['hold_frame'].copy()

            # --- dibujar HUD siempre ---
            hud = f"AUTO PRED | {device.type.upper()} | in:{target_w}x{target_h}"
            if ema_live_fps is not None: hud += f" | LIVE FPS:{ema_live_fps:.1f}"
            if ema_proc_fps is not None: hud += f" | PROC FPS:{ema_proc_fps:.1f}"
            show_frame(disp, hud, play_active=False)

            # 4) avanzar 1 frame real para refrescar ventana de entrada
            ok, frame = cap.read()
            if not ok: break
            t_in = to_model_tensor_torch(frame, (target_h, target_w), use_rgb=True, half=args.half)
            buf_real.append(t_in)
            state['reals_since_last_pred'] += 1

            # 5) al cumplir el stride de reales, cambiamos de bloque:
            #    - si ya tenemos pending_out, lo activamos
            #    - si no, invalidamos el actual y esperamos a que llegue
            if state['reals_since_last_pred'] >= block_stride:
                state['reals_since_last_pred'] = 0
                if pending_out is not None:
                    state['pred_cache'] = pending_out
                    state['pred_idx'] = 0
                    pending_out = None
                else:
                    state['pred_cache'] = None  # esperamos a que termine el próximo

                # opcional: si no hay un future en curso ni pending, lanzarlo ahora
                if next_future is None and pending_out is None and buf_real.ready():
                    inp = buf_real.as_tensor(device, half=args.half, channels_last=True).contiguous()
                    next_future = executor.submit(
                        forward_pred, model, inp, use_amp=args.amp, device_type=device.type
                    )
                    forward_start = time.time()

            # teclado
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'): break
            elif k == ord('p'): state['paused'] = True
            elif k == ord('i'):
                state['infer_mode'] = False
                state['pred_cache'] = None; state['pred_idx'] = 0; state['reals_since_last_pred'] = 0
                next_future = None
                pending_out = None

            # FPS loop
            dt_live = loop_start - last_loop_t
            live_fps = 1.0 / max(dt_live, 1e-6)
            ema_live_fps = live_fps if ema_live_fps is None else (0.9*ema_live_fps + 0.1*live_fps)
            last_loop_t = loop_start
            continue

    cap.release()
    executor.shutdown(wait=False)  # 🧠 cerrar thread pool
    cv2.destroyAllWindows()
    print("Bye.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--cfg",  type=str, default=None)
    p.add_argument("--video", type=str, required=True)

    # modelo/datos
    p.add_argument("--pre_seq", type=int, default=10, help="frames de entrada reales")
    p.add_argument("--pred_len", type=int, default=10, help="frames a predecir por bloque")
    p.add_argument("--block_stride", type=int, default=0,
                   help="cada cuántos frames REALES recalcular un bloque (0=usa pred_len)")

    # resolución interna (landscape)
    p.add_argument("--in_size_y", type=int, default=144)
    p.add_argument("--in_size_x", type=int, default=256)

    # visualización y control
    p.add_argument("--fps", type=int, default=0, help="cap de FPS del loop (0 = usar fps de fuente)")
    p.add_argument("--display_scale", type=int, default=1)
    p.add_argument("--hud", type=int, default=1)
    p.add_argument("--no_upscale", type=int, default=0, help="mostrar a resolución interna")

    # performance
    p.add_argument("--half", type=int, default=1, help="modelo/inputs en fp16")
    p.add_argument("--amp", type=int, default=0, help="autocast fp16 solo en CUDA")
    p.add_argument("--resize_threads", type=int, default=2, help="hilos OpenCV para resize/color")

    args = p.parse_args()
    if args.block_stride <= 0:
        args.block_stride = args.pred_len
    main(args)
