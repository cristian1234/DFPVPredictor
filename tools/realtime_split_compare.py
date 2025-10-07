# tools/realtime_split_compare.py
# Realtime split-screen: izquierda RAW (decoder), derecha REPAIRED (detector+inpaint+modelo).
# Soporta filesrc y RTP. Usa GStreamer (openh264dec) + OpenCV para mostrar.

import os, sys, time, argparse, runpy
import numpy as np
import cv2
import torch

from simvp.models.simvp_model import SimVP_Model

# --- GStreamer (gst-python) ---
import gi
gi.require_version('Gst', '1.0')
#gi.require_version('GstApp', '1.0')
from gi.repository import Gst
#, GstApp)

# ------------------ util cfg/model ------------------
def load_cfg_from_py(cfg_path):
    g = runpy.run_path(cfg_path)
    for key in ('config','cfg','CONFIG','Config'):
        if key in g and isinstance(g[key], dict):
            return g[key]
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

# ------------------ conversión frames ------------------
def to_model_tensor(frame_bgr, out_hw, use_rgb=True):
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
    h, w = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    # baja varianza por bloque
    small = cv2.resize(gray, (max(1,w//block), max(1,h//block)), interpolation=cv2.INTER_AREA)
    mean_blk = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    resid = gray.astype(np.float32) - mean_blk.astype(np.float32)
    var_mask = (np.abs(resid) < var_th).astype(np.uint8) * 255

    # discontinuidades en grilla
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    grid = np.zeros_like(var_mask)
    grid[::block, :] = 255; grid[:, ::block] = 255
    edge_mask = (mag > 60).astype(np.uint8) * grid

    # freeze vs frame previo
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

# ------------------ GStreamer helpers ------------------
def bgr_from_sample(sample):
    buf = sample.get_buffer()
    caps = sample.get_caps()
    s = caps.get_structure(0)
    w = s.get_value('width'); h = s.get_value('height')
    ok, mapinfo = buf.map(Gst.MapFlags.READ)
    if not ok: return None
    try:
        arr = np.frombuffer(mapinfo.data, dtype=np.uint8)
        frame = arr.reshape((h, w, 3))  # format=BGR
    finally:
        buf.unmap(mapinfo)
    return frame

def build_pipeline_desc(args):
    def has_elem(name):
        return Gst.ElementFactory.find(name) is not None

    def fps_fraction(fps):
        if abs(fps - 29.97) < 0.02: return "30000/1001"
        if abs(fps - 59.94) < 0.02: return "60000/1001"
        return f"{int(round(fps))}/1"

    fps_caps = fps_fraction(float(args.target_fps))

    # decoder preferido, con fallback
    dec = args.decoder
    if dec != "decodebin" and not has_elem(dec):
        print(f"[WARN] No existe el decoder '{dec}', probando fallback…")
        dec = "avdec_h264" if has_elem("avdec_h264") else "decodebin"
        print(f"[INFO] Usando decoder: {dec}")

    # bloques comunes
    # OJO: ponemos videoconvert + videoscale ANTES de videorate para que la negociación sea crujiente.
    base_convert = "videoconvert ! videoscale"
    rate_caps    = f"videorate ! video/x-raw,format=BGR,framerate={fps_caps}"
    to_bgr_only  = "videoconvert ! video/x-raw,format=BGR"
    out_queue    = "queue leaky=downstream max-size-buffers=1 max-size-time=0 max-size-bytes=0"
    sink_caps    = "appsink name=sink emit-signals=false sync=true async=false max-buffers=1 drop=true"

    drop = ""
    if args.drop_prob > 0 and dec != "decodebin":
        drop = f"identity drop-probability={args.drop_prob} ! "
    elif args.drop_prob > 0:
        print("[WARN] --drop_prob ignorado con decodebin (no puede ir antes del decoder).")

    if args.rtp:
        caps = (args.rtp_caps if args.rtp_caps
                else 'application/x-rtp, media=video, encoding-name=H264, payload=96')
        # RTP
        pipe_main = (
            f"udpsrc port={args.port} caps=\"{caps}\" ! "
            f"rtpjitterbuffer do-lost=true ! rtph264depay ! h264parse ! queue ! "
            f"{drop}{dec} ! {base_convert} ! {rate_caps} ! {out_queue} ! {sink_caps}"
        )
        pipe_fallback = (
            f"udpsrc port={args.port} caps=\"{caps}\" ! "
            f"rtpjitterbuffer do-lost=true ! rtph264depay ! h264parse ! queue ! "
            f"{drop}{dec} ! {to_bgr_only} ! {out_queue} ! {sink_caps}"
        )
    else:
        # FILE
        if dec == "decodebin":
            pipe_main = (
                f"filesrc location=\"{args.input}\" ! decodebin ! "
                f"{base_convert} ! {rate_caps} ! {out_queue} ! {sink_caps}"
            )
            pipe_fallback = (
                f"filesrc location=\"{args.input}\" ! decodebin ! "
                f"{to_bgr_only} ! {out_queue} ! {sink_caps}"
            )
        else:
            pipe_main = (
                f"filesrc location=\"{args.input}\" ! qtdemux ! h264parse ! queue ! "
                f"{drop}{dec} ! {base_convert} ! {rate_caps} ! {out_queue} ! {sink_caps}"
            )
            pipe_fallback = (
                f"filesrc location=\"{args.input}\" ! qtdemux ! h264parse ! queue ! "
                f"{drop}{dec} ! {to_bgr_only} ! {out_queue} ! {sink_caps}"
            )

    # Intentamos la versión con videorate; si no, caemos a la simple.
    try:
        return pipe_main, pipe_fallback
    except Exception:
        return pipe_fallback, None


# ------------------ draw helpers ------------------
def draw_badge(img, text, pos=(10,24)):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

def hstack_sameh(a, b):
    h = min(a.shape[0], b.shape[0])
    a2 = cv2.resize(a, (int(a.shape[1]*h/a.shape[0]), h)) if a.shape[0]!=h else a
    b2 = cv2.resize(b, (int(b.shape[1]*h/b.shape[0]), h)) if b.shape[0]!=h else b
    return cv2.hconcat([a2, b2])

def predict_patch(model, buf, W, H, mask, base_bgr):
    """Predice solo parches dañados sobre base_bgr (BGR actual)."""
    device = next(model.parameters()).device
    pred_cache = None
    pred_idx = 0
    used_pred = False

    with torch.no_grad():
        inp = buf.as_tensor(device)
        pred_cache = model(inp)
        pred_t = pred_cache[0, 0].detach().cpu().numpy()
        pred_bgr = to_display_image(pred_t)
        pred_bgr = cv2.resize(pred_bgr, (W, H), interpolation=cv2.INTER_LINEAR)
        repaired = composite_patch(base_bgr, pred_bgr, mask)
        used_pred = True

    return pred_cache, pred_idx, repaired, used_pred



def predict_full(model, buf, W, H, pred_cache=None, pred_idx=0):
    """Predice frames completos (usado cuando hay stall o pérdida grande)."""
    device = next(model.parameters()).device
    with torch.no_grad():
        if pred_cache is None or pred_idx >= pred_cache.shape[1]:
            inp = buf.as_tensor(device)
            pred_cache = model(inp)
            pred_idx = 0

        pred_t = pred_cache[0, pred_idx].detach().cpu().numpy()
        pred_idx += 1
        repaired = cv2.resize(to_display_image(pred_t), (W, H), interpolation=cv2.INTER_LINEAR)

    return pred_cache, pred_idx, repaired, True



# ------------------ main ------------------
def main():
    parser = argparse.ArgumentParser()
    # entrada
    parser.add_argument("--input", type=str, default="drone.mp4", help="ruta mp4 si no es RTP")
    parser.add_argument("--rtp", action="store_true", help="usar RTP (udpsrc)")
    parser.add_argument("--port", type=int, default=5004)
    parser.add_argument("--rtp_caps", type=str, default="", help="caps RTP completos (opcional)")
    parser.add_argument("--drop_prob", type=float, default=0.0, help="dropear NALs antes del decoder (agresivo)")
    # modelo
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--cfg",  type=str, default=None)
    parser.add_argument("--pre_seq",   type=int, default=10)
    parser.add_argument("--in_size",   type=int, default=64)
    # detector/heurísticas
    parser.add_argument("--block",       type=int,   default=16)
    parser.add_argument("--var_th",      type=float, default=2.0)
    parser.add_argument("--diff_th",     type=float, default=18.0)
    parser.add_argument("--small_thresh",type=float, default=0.06)
    parser.add_argument("--large_thresh",type=float, default=0.40)
    parser.add_argument("--use_inpaint", action="store_true")
    # timing
    parser.add_argument("--target_fps",  type=float, default=30.0)
    parser.add_argument("--stall_timeout_ms", type=int, default=200)
    # salida opcional
    parser.add_argument("--decoder", type=str, default="avdec_h264",
                        help="h264 decoder: avdec_h264 | vtdec_h264 | openh264dec | decodebin")
    parser.add_argument("--save_out", type=str, default="", help="guardar split en mp4 opcional")
    args = parser.parse_args()

    # === Modelo ===
    device = choose_device()
    print("Device:", device)
    ckpt = torch.load(args.ckpt, map_location=device)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']; ckpt_cfg = ckpt.get('config', None)
    elif isinstance(ckpt, dict):
        state_dict = ckpt; ckpt_cfg = None
    else:
        raise ValueError("Formato de checkpoint no soportado")

    if args.cfg:
        cfg = load_cfg_from_py(args.cfg)
    else:
        cfg = ckpt_cfg or build_config_fallback(args.in_size, args.in_size, 3, pre_seq=args.pre_seq)

    C = 3
    cfg['in_shape'] = [args.pre_seq, C, args.in_size, args.in_size]
    cfg.setdefault('model_type','gSTA')

    model = SimVP_Model(**cfg).to(device)
    new_sd = { (k[7:] if k.startswith('module.') else k): v for k,v in state_dict.items() }
    model.load_state_dict(new_sd, strict=False)
    model.eval()

    # warm-up
    with torch.no_grad():
        dummy = torch.zeros(1, args.pre_seq, C, args.in_size, args.in_size, device=device)
        _ = model(dummy)

    # === GStreamer ===
    Gst.init(None)
    pipe_main, pipe_fallback = build_pipeline_desc(args)

    for desc in [pipe_main, pipe_fallback]:
        if not desc:
            continue
        try:
            print("[GST] Intentando pipeline:\n", desc)
            pipeline = Gst.parse_launch(desc)
            break
        except Exception as e:
            print("[GST] Falló ese pipeline:", e)
            pipeline = None

    if pipeline is None:
        print("[GST] No pude construir ningún pipeline.")
        sys.exit(1)

    appsink = pipeline.get_by_name("sink")
    pipeline.set_state(Gst.State.PLAYING)
    print("[GST] Pipeline PLAYING")



    # buffer y estado
    buf = RingBuffer(args.pre_seq)
    pred_cache, pred_idx = None, 0
    last_raw = None
    last_frame_time = time.monotonic()
    stall_dt = args.stall_timeout_ms / 1000.0

    first_sample = appsink.emit("try-pull-sample", 5_000_000_000)
    if first_sample is None:
        print("No llegó ningún frame inicial.")
        pipeline.set_state(Gst.State.NULL)
        sys.exit(1)

    raw = bgr_from_sample(first_sample)
    H, W = raw.shape[:2]
    if args.save_out:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.save_out, fourcc, args.target_fps, (W*2, H))
    else:
        writer = None

    cv2.namedWindow("RAW | REPAIRED", cv2.WINDOW_NORMAL)
    buf.append(to_model_tensor(raw, (args.in_size,args.in_size), True))
    prev_rx = raw.copy()
    last_raw = raw.copy()

    try:
        while True:
            sample = appsink.emit("try-pull-sample", 100_000_000)  # 100 ms
            now = time.monotonic()

            if sample is not None:
                raw = bgr_from_sample(sample)
                last_frame_time = now
                mask = detect_corruption_mask(raw, prev_bgr=prev_rx,
                                              block=args.block, var_th=args.var_th, diff_th=args.diff_th)
                ratio = float(np.mean(mask>0))
                repaired = raw.copy()
                used_pred = False

                if ratio < args.small_thresh:
                    if args.use_inpaint and np.any(mask):
                        repaired = cv2.inpaint(repaired, (mask>0).astype(np.uint8)*255, 3, cv2.INPAINT_TELEA)
                elif ratio < args.large_thresh and buf.ready():
                    pred_cache, pred_idx, repaired, used_pred = predict_patch(model, buf, W, H, mask, base_bgr=raw)
                elif buf.ready():
                    pred_cache, pred_idx, repaired, used_pred = predict_full(model, buf, W, H, pred_cache, pred_idx)

                feed = repaired if (used_pred or (args.use_inpaint and np.any(mask))) else raw
                buf.append(to_model_tensor(feed, (args.in_size,args.in_size), True))
                prev_rx, last_raw = raw.copy(), raw.copy()

            elif now - last_frame_time > stall_dt:
                # stall: sin frame nuevo
                raw = last_raw
                if buf.ready():
                    pred_cache, pred_idx, repaired, _ = predict_full(model, buf, W, H, pred_cache, pred_idx)
                    buf.append(to_model_tensor(repaired, (args.in_size,args.in_size), True))
                else:
                    repaired = raw
                time.sleep(1/args.target_fps)
            else:
                continue

            raw_disp, rep_disp = raw.copy(), repaired.copy()
            draw_badge(raw_disp, "RAW")
            draw_badge(rep_disp, "REPAIRED")
            combo = hstack_sameh(raw_disp, rep_disp)
            cv2.imshow("RAW | REPAIRED", combo)
            if writer: writer.write(combo)

            if cv2.waitKey(1) == 27:
                break

    finally:
        pipeline.set_state(Gst.State.NULL)
        if writer: writer.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
