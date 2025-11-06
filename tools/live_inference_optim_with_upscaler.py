import cv2, time, argparse, numpy as np, torch, runpy
import concurrent.futures
from basicsr.archs.rrdbnet_arch import RRDBNet
from queue import Queue, Empty

try:
    from realesrgan import RealESRGANer
    _has_realesrgan = True
except ImportError:
    _has_realesrgan = False

from simvp.models.simvp_model import SimVP_Model

# ================= Cola SR y helpers ==================
sr_queue = Queue(maxsize=40)  # tamaño final se ajusta en main()

def clear_queue(q: Queue):
    """Limpia la cola completamente sin bloquear."""
    while not q.empty():
        try:
            q.get_nowait()
        except Empty:
            break

def enqueue_block_non_blocking(frames_list, q: Queue):
    """Encola frames sin bloquear; si la cola está llena, descarta los más viejos."""
    for fr in frames_list:
        try:
            q.put_nowait(fr)
        except:
            try:
                q.get_nowait()
            except Empty:
                pass
            q.put_nowait(fr)

def submit_sr_block(frames_bgr_list, q: Queue, sr):
    """Ejecuta SR en un hilo aparte (CPU) y encola el bloque al terminar."""
    def _job(lst):
        print("[SR] start")
        up_lst = upscale_block_async(lst, sr)
        enqueue_block_non_blocking(up_lst, q)
        print("[SR] end")

    print("[DEBUG] executor_sr is", executor_sr)
    try:
        fut = executor_sr.submit(_job, frames_bgr_list)
        print("[DEBUG] future:", fut)
    except Exception as e:
        print("[ERROR] submit_sr_block:", e)

# ================= Real-ESRGAN ==================
def upscale_block_async(frames_bgr_list, sr):
    out_frames = []
    for f in frames_bgr_list:
        upscaled, _ = sr.enhance(f.copy(), outscale=2.0)
        out_frames.append(np.ascontiguousarray(upscaled, dtype=np.uint8))
    return out_frames

def load_realesrgan_x4(weights_path, half=False, device='mps'):
    model = RRDBNet(num_in_ch=3,num_out_ch=3,num_feat=64,num_block=23,num_grow_ch=32,scale=4)
    #torch.set_default_device("cpu")
    upsampler = RealESRGANer(
        scale=4,
        model_path=weights_path,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=half
    )
    torch_device = torch.device(device)
    upsampler.device = torch_device
    upsampler.model.to(torch_device)
    return upsampler

# ================= Config helpers ==================
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
    pre = max(1, pred_len)
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

# ================= I/O utils ==================
def to_model_tensor_torch(frame_bgr, out_size_hw, use_rgb=True, half=False):
    H, W = out_size_hw
    if frame_bgr.shape[0] != H or frame_bgr.shape[1] != W:
        f = cv2.resize(frame_bgr, (W, H), interpolation=cv2.INTER_AREA)
    else:
        f = frame_bgr
    if use_rgb: f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(f)
    x = x.to(torch.float16 if half else torch.float32).div_(255.0)
    x = x.permute(2,0,1).contiguous()
    return x

def to_display_image(chw_float):
    if isinstance(chw_float, torch.Tensor):
        x = chw_float.detach().cpu().float().numpy()
    else:
        x = np.array(chw_float, dtype=np.float32)
    x = np.clip(x, 0, 1)
    x = (x * 255.0).astype(np.uint8)
    x = np.transpose(x, (1, 2, 0))
    x = x[:, :, ::-1]
    return np.ascontiguousarray(x, dtype=np.uint8)

def draw_hud(img_bgr, text, bg=(0,0,0), fg=(255,255,255), alpha=0.6):
    overlay = img_bgr.copy()
    H, W = img_bgr.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.24, 1)
    x0 = max(2, (W - tw)//2 - 10); y0 = 8
    x1 = min(W-2, x0 + tw + 20);  y1 = y0 + th + 12
    cv2.rectangle(overlay, (x0, y0), (x1, y1), bg, -1)
    out = cv2.addWeighted(overlay, alpha, img_bgr, 1 - alpha, 0)
    cv2.putText(out, text, (x0 + 2, y0 + th + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.24, fg, 1, cv2.LINE_AA)
    return out

def draw_button(img_bgr, rect, label, active=False):
    x0,y0,x1,y1 = rect
    color_bg = (40,40,40) if not active else (60,60,60)
    color_fg = (255,255,255)
    cv2.rectangle(img_bgr, (x0,y0), (x1,y1), color_bg, -1)
    cv2.rectangle(img_bgr, (x0,y0), (x1,y1), (180,180,180), 1)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)
    tx = x0 + (x1-x0 - tw)//2; ty = y0 + (y1-y0 + th)//2
    cv2.putText(img_bgr, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color_fg, 1, cv2.LINE_AA)
    return img_bgr

# ================= RingBuffer ==================
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
        if len(self.data) == 0:
            return None
        t = torch.stack(self.data, dim=0).unsqueeze(0)
        if half: t = t.half()
        return t.to(device)

# ================= Forward modelo ==================
def forward_pred(model, buf_tensor, use_amp=False, device_type='cpu'):
    with torch.no_grad():
        if use_amp and device_type == 'cuda':
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                return model(buf_tensor)
        else:
            return model(buf_tensor)

# ================= MAIN ==================
def main(args):
    global sr_queue, executor_sr

    try: cv2.setNumThreads(max(1, args.resize_threads))
    except: pass

    device = choose_device()
    print(f"Using device: {device}")

    sr_queue = Queue(maxsize=4 * args.pred_len)
    executor_sr = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened(): raise FileNotFoundError(f"No pude abrir {args.video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    ok, first = cap.read()
    if not ok: raise RuntimeError("Video vacío")
    src_h, src_w = first.shape[:2]

    target_h, target_w = args.in_size_y, args.in_size_x
    if (src_w >= src_h) and (target_w < target_h):
        print(f"[WARN] in_size_x/in_size_y invertidos; ajusto {target_h}x{target_w} -> {target_w}x{target_h}")
        target_h, target_w = target_w, target_h

    if args.no_upscale: base_w, base_h = target_w, target_h
    else:               base_w, base_h = src_w, src_h
    disp_w, disp_h = int(base_w * args.display_scale), int(base_h * args.display_scale)
    display_size = (disp_w, disp_h)

    BTN_W, BTN_H, BTN_MARGIN = 50, 20, 10
    btn_rect_base = (BTN_MARGIN, BTN_MARGIN + 20, BTN_MARGIN+BTN_W, BTN_MARGIN+20+BTN_H)

    fallback_cfg = build_config_fallback((target_h, target_w, 3), pred_len=args.pred_len)
    ckpt = torch.load(args.ckpt, map_location=device)
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    cfg = ckpt.get('config', None) if isinstance(ckpt, dict) else None
    if cfg is None and args.cfg:
        try: cfg = load_cfg_from_py(args.cfg)
        except Exception as e: print(f"[WARN] No pude leer config del .py: {e}")
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

    # SR
    sr = None
    if _has_realesrgan:
        #model_name =  "realesr-general-x4v3.pth" #"RealESRGAN_x4plus_netD.pth" # 'realesr-general-wdn-x4v3.pth'
        model_name = "RealESRGAN_x4plus.pth"
        sr = load_realesrgan_x4("models/" +model_name, half=False, device="cpu")
        print("[INFO] Real-ESRGAN inicializado.")
    else:
        print("[WARN] realesrgan no está instalado.")

    # warmup
    _ = forward_pred(model,
        torch.zeros(1, args.pre_seq, C, target_h, target_w, device=device,
                    dtype=torch.float16 if args.half else torch.float32),
        use_amp=args.amp, device_type=device.type
    )

    buf_real = RingBuffer(args.pre_seq)
    buf_real.append(to_model_tensor_torch(first, (target_h, target_w), use_rgb=True, half=args.half))

    window_name = "FPV Live (SPACE=drop, I=auto, P=pause, Q=quit, U=SR)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, display_size[0], display_size[1])

    state = dict(
        paused=False,
        infer_mode=False,
        upscaler_mode=False,
        hold_frame=cv2.resize(first.copy(), display_size),
        pred_idx=0
    )

    target_fps = args.fps if args.fps > 0 else src_fps
    frame_interval_ms = 1000.0 / max(1.0, target_fps)
    last_ms = time.time() * 1000.0
    ema_live_fps = None
    ema_proc_fps = None
    last_loop_t = time.time()
    block_stride = args.block_stride if args.block_stride > 0 else args.pred_len
    cap.set(cv2.CAP_PROP_POS_FRAMES, 1)

    need_block = True

    def show_frame(img_bgr, hud_text, play_active=False):
        if args.hud: img_bgr = draw_hud(img_bgr, hud_text)
        img_bgr = draw_button(img_bgr, btn_rect_base, "PAUSE" if not play_active else "PLAY", active=play_active)
        cv2.imshow(window_name, img_bgr)

    # ================= Main Loop =================
    while True:
        loop_start = time.time()
        now_ms = loop_start * 1000.0
        if now_ms - last_ms < frame_interval_ms:
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'): break
            elif k == ord('p'): state['paused'] = not state['paused']
            elif k == ord('i'):
                state['infer_mode'] = not state['infer_mode']
                if not state['infer_mode']:
                    state['pred_idx'] = 0
                    need_block = True
                    clear_queue(sr_queue)
                print(f"[INFO] Inferencia automática {'ON' if state['infer_mode'] else 'OFF'}.")
            elif k == ord('u'):
                state['upscaler_mode'] = not state['upscaler_mode']
                print(f"[INFO] Upscaler {'ON' if state['upscaler_mode'] else 'OFF'}.")
            continue
        last_ms = now_ms

        # PAUSA
        if state['paused']:
            disp = state['hold_frame'].copy()
            hud = f"PAUSE | {device.type.upper()} | in:{target_w}x{target_h}"
            show_frame(disp, hud, play_active=True)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'): break
            elif k == ord('p'): state['paused'] = not state['paused']
            continue

        # LIVE (sin inferencia)
        if not state['infer_mode']:
            ok, frame = cap.read()
            if not ok:
                clear_queue(sr_queue)
                break
            disp = frame if args.no_upscale else cv2.resize(frame, display_size)
            state['hold_frame'] = disp.copy()
            buf_real.append(to_model_tensor_torch(frame, (target_h, target_w), use_rgb=True, half=args.half))
            show_frame(disp, f"LIVE | {device.type.upper()} | in:{target_w}x{target_h}")
            ema_live_fps = 1.0/(loop_start - last_loop_t) if ema_live_fps is None else 0.9*ema_live_fps+0.1/(loop_start-last_loop_t)
            last_loop_t = loop_start
            continue

        # Inferencia continua síncrona
        if need_block and buf_real.ready():
            t0 = time.time()
            inp = buf_real.as_tensor(device, half=args.half, channels_last=True).contiguous()
            out = forward_pred(model, inp, use_amp=False, device_type=device.type)
            fwd_time = time.time() - t0
            print(f"[DBG] forward {args.pred_len} frames = {fwd_time*1000:.2f} ms")
            proc_fps = float(args.pred_len) / max(fwd_time, 1e-6)
            ema_proc_fps = proc_fps if ema_proc_fps is None else (0.8*ema_proc_fps + 0.2*proc_fps)

            frames_bgr_list = [to_display_image(out[0,i]) for i in range(args.pred_len)]
            if state['upscaler_mode'] and _has_realesrgan:
                submit_sr_block(frames_bgr_list, sr_queue, sr)
            else:
                enqueue_block_non_blocking(frames_bgr_list, sr_queue)

            state['pred_idx'] = 0
            need_block = False

        # Mostrar siguiente frame predicho
        try:
            disp = sr_queue.get(timeout=0.001)
            idx_in_block = state['pred_idx'] % args.pred_len
            if idx_in_block == 0:
                h, w = disp.shape[:2]
                cv2.rectangle(disp, (3, 3), (w - 4, h - 4), (0, 0, 255), 4)
                cv2.putText(disp, "BLOCK START", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, cv2.LINE_AA)
            if not args.no_upscale:
                disp = cv2.resize(disp, display_size, interpolation=cv2.INTER_LINEAR)
            state['hold_frame'] = disp.copy()
            state['pred_idx'] += 1
            if state['pred_idx'] >= args.pred_len:
                need_block = True
        except Empty:
            disp = state['hold_frame'].copy()

        # HUD
        hud = f"AUTO PRED | {device.type.upper()} | in:{target_w}x{target_h}"
        if ema_live_fps is not None: hud += f" | LIVE FPS:{ema_live_fps:.1f}"
        if ema_proc_fps is not None: hud += f" | PROC FPS:{ema_proc_fps:.1f}"
        if state['upscaler_mode']: hud += " | SR: ON"
        if need_block and sr_queue.empty():
            hud += " | waiting block…"
        show_frame(disp, hud, play_active=False)

        # avanzar un frame real para mantener ventana
        ok, frame = cap.read()
        if not ok:
            clear_queue(sr_queue)
            break
        buf_real.append(to_model_tensor_torch(frame, (target_h, target_w), use_rgb=True, half=args.half))

        # teclado
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'): break
        elif k == ord('p'):
            state['paused'] = True
        elif k == ord('i'):
            state['infer_mode'] = False
            state['pred_idx'] = 0
            need_block = True
            clear_queue(sr_queue)

        # FPS loop
        dt_live = loop_start - last_loop_t
        live_fps = 1.0 / max(dt_live, 1e-6)
        ema_live_fps = live_fps if ema_live_fps is None else (0.9*ema_live_fps + 0.1*live_fps)
        last_loop_t = loop_start
        time.sleep(0.001)

    # ================= Limpieza final =================
    clear_queue(sr_queue)
    executor_sr.shutdown(wait=False)
    cv2.destroyAllWindows()
    print("[INFO] Bye. Recursos liberados correctamente.")

# ================= CLI ==================
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--cfg",  type=str, default=None)
    p.add_argument("--video", type=str, required=True)
    p.add_argument("--pre_seq", type=int, default=10)
    p.add_argument("--pred_len", type=int, default=10)
    p.add_argument("--block_stride", type=int, default=0)
    p.add_argument("--in_size_y", type=int, default=144)
    p.add_argument("--in_size_x", type=int, default=256)
    p.add_argument("--fps", type=int, default=0)
    p.add_argument("--display_scale", type=float, default=1)
    p.add_argument("--hud", type=int, default=1)
    p.add_argument("--no_upscale", type=int, default=0)
    p.add_argument("--half", type=int, default=1)
    p.add_argument("--amp", type=int, default=0)
    p.add_argument("--resize_threads", type=int, default=2)
    args = p.parse_args()
    if args.block_stride <= 0:
        args.block_stride = args.pred_len
    main(args)
