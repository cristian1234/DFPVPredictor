import argparse, sys
import numpy as np
import cv2
import gi, time
gi.require_version('Gst', '1.0')
from gi.repository import Gst

# ---------- helpers GStreamer ----------
def bgr_from_sample(sample):
    buf = sample.get_buffer()
    caps = sample.get_caps()
    s = caps.get_structure(0)
    w = s.get_value('width'); h = s.get_value('height')
    ok, mapinfo = buf.map(Gst.MapFlags.READ)
    if not ok: return None
    try:
        arr = np.frombuffer(mapinfo.data, dtype=np.uint8)
        frame = arr.reshape((h, w, 3)).copy()
    finally:
        buf.unmap(mapinfo)
    return frame

def build_pipeline_desc(args):
    # helper: chequear si existe un elemento
    def has_elem(name):
        return Gst.ElementFactory.find(name) is not None

    # elegir decoder con fallback
    dec = getattr(args, "decoder", "decodebin")
    if dec != "decodebin" and not has_elem(dec):
        print(f"[WARN] No existe el decoder '{dec}', usando fallback…")
        if has_elem("avdec_h264"):
            dec = "avdec_h264"
        elif has_elem("vtdec_h264"):
            dec = "vtdec_h264"
        elif has_elem("openh264dec"):
            dec = "openh264dec"
        else:
            dec = "decodebin"
        print(f"[INFO] Decoder seleccionado: {dec}")

    # conversión + (opcional) resize antes del appsink
    if getattr(args, "gst_w", 0) and getattr(args, "gst_h", 0):
        tail = (
            "videoconvert ! videoscale ! "
            f"video/x-raw,format=BGR,width={int(args.gst_w)},height={int(args.gst_h)}"
        )
    else:
        tail = "videoconvert ! video/x-raw,format=BGR"

    sink = "appsink name=sink sync=true emit-signals=false max-buffers=1 drop=true"

    if getattr(args, "rtp", False):
        # Caps RTP por defecto (coinciden con pt=96 del sender)
        caps = args.rtp_caps or "application/x-rtp, media=video, encoding-name=H264, payload=96, clock-rate=90000"
        # RTP → jitterbuffer → depay → parse → decoder → BGR → appsink
        # (para RTP es más robusto elegir un decoder explícito)
        if dec == "decodebin":
            dec = "avdec_h264"
        pipe = (
            f'udpsrc port={int(args.port)} caps="{caps}" ! '
            "rtpjitterbuffer do-lost=true mode=1 latency=0 ! "
            "rtph264depay ! h264parse ! "
            f"{dec} ! {tail} ! {sink}"
        )
    else:
        # Archivo local
        if dec == "decodebin":
            pipe = f'filesrc location="{args.input}" ! decodebin ! {tail} ! {sink}'
        else:
            pipe = (
                f'filesrc location="{args.input}" ! qtdemux ! h264parse ! '
                f"{dec} ! {tail} ! {sink}"
            )

    return pipe

# ---------- cachés ligeras ----------
_grid_cache = {}
_kernel3 = np.ones((3,3), np.uint8)

def get_grid(h, w, block):
    key = (h, w, block)
    g = _grid_cache.get(key)
    if g is None:
        g = np.zeros((h, w), dtype=np.uint8)
        g[::block, :] = 255
        g[:, ::block] = 255
        _grid_cache[key] = g
    return g

# ---------- detector ----------
def detect_corruption_mask(frame_bgr, prev_bgr=None, block=16, var_th=2.0, diff_th=18.0, proc_scale=1.0):
    # Escalamos si hace falta (para performance)
    if proc_scale < 1.0:
        frame_bgr = cv2.resize(frame_bgr, None, fx=proc_scale, fy=proc_scale,
                               interpolation=cv2.INTER_AREA)
        if prev_bgr is not None:
            prev_bgr = cv2.resize(prev_bgr, (frame_bgr.shape[1], frame_bgr.shape[0]),
                                   interpolation=cv2.INTER_AREA)

    h, w = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    # --- VARIANCE MAP ---
    small = cv2.resize(gray, (max(1, w // block), max(1, h // block)), interpolation=cv2.INTER_AREA)
    mean_blk = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    resid = np.abs(gray.astype(np.float32) - mean_blk.astype(np.float32))

    # zonas con muy baja variación (planas) -> posiblemente *buenas*, NO malas
    var_mask = (resid > var_th).astype(np.uint8) * 255  # invertido
    var_mask = cv2.bitwise_not(var_mask)  # invertimos lógica: planas = buenas → invertimos para que no sumen

    # --- FREEZE DETECTION ---
    freeze_mask = np.zeros_like(var_mask)
    if prev_bgr is not None:
        prev_g = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray, prev_g)
        mean_diff = cv2.blur(diff, (block, block))
        freeze_mask = (mean_diff < diff_th).astype(np.uint8) * 255  # menor diferencia = congelado

    # --- COMBINACIÓN ---
    # Sólo consideramos corrupto si ambas condiciones coinciden (zona plana + congelada)
    mask = cv2.bitwise_and(var_mask, freeze_mask)

    # Limpieza morfológica
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)

    # --- Si usamos proc_scale, reescalamos la máscara al tamaño original ---
    if proc_scale < 1.0:
        mask = cv2.resize(mask, (int(w / proc_scale), int(h / proc_scale)), interpolation=cv2.INTER_NEAREST)

    return mask

def _detect_mask_core(frame_bgr, prev_bgr, block, var_th, diff_th):
    h, w = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    # baja varianza por bloque (aprox macroblock plano)
    small = cv2.resize(gray, (max(1,w//block), max(1,h//block)), interpolation=cv2.INTER_AREA)
    mean_blk = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    resid = gray.astype(np.float32) - mean_blk.astype(np.float32)
    var_mask = (np.abs(resid) < var_th).astype(np.uint8) * 255

    # discontinuidades alineadas a bloque
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    grid = get_grid(h, w, block)
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
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _kernel3, iterations=1)
    mask = cv2.dilate(mask, _kernel3, iterations=1)
    return mask

def overlay_mask(bgr, mask, color=(0,0,255), alpha=0.4):
    out = bgr.copy()
    m = mask.astype(bool)
    out[m] = (1 - alpha) * out[m] + alpha * np.array(color, dtype=np.uint8)
    return out

# ---------- inpaint ----------
def inpaint_bgr_full(bgr, mask, method='telea', radius=3):
    mask8 = (mask>0).astype(np.uint8)*255
    algo = cv2.INPAINT_TELEA if method.lower()=='telea' else cv2.INPAINT_NS
    return cv2.inpaint(bgr, mask8, radius, algo)

def inpaint_bgr_rois(bgr, mask, method='telea', radius=3, min_area=200, margin=4):
    algo = cv2.INPAINT_TELEA if method.lower()=='telea' else cv2.INPAINT_NS
    mask8 = (mask>0).astype(np.uint8)
    cnts,_ = cv2.findContours(mask8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = bgr.copy()
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        if w*h < min_area:  # filtra ruído
            continue
        x0 = max(0, x-margin); y0 = max(0, y-margin)
        x1 = min(bgr.shape[1], x+w+margin); y1 = min(bgr.shape[0], y+h+margin)
        roi = out[y0:y1, x0:x1]
        mroi = (mask8[y0:y1, x0:x1]*255).astype(np.uint8)
        repaired = cv2.inpaint(roi, mroi, radius, algo)
        out[y0:y1, x0:x1] = repaired
    return out

# ---------- draw ----------
def draw_badge(img, text, pos=(10,24)):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

def hstack_sameh(a, b):
    h = min(a.shape[0], b.shape[0])
    if a.shape[0] != h: a = cv2.resize(a, (int(a.shape[1]*h/a.shape[0]), h))
    if b.shape[0] != h: b = cv2.resize(b, (int(b.shape[1]*h/b.shape[0]), h))
    return cv2.hconcat([a, b])

# ---------- main ----------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, default="", help="ruta mp4 si no es RTP")
    p.add_argument("--decoder", default="avdec_h264",
                   help="decodebin | avdec_h264 | vtdec_h264 | openh264dec")
    p.add_argument("--mode", choices=["mask","inpaint"], default="inpaint")
    p.add_argument("--inpaint_method", type=int, default=cv2.INPAINT_TELEA)
    p.add_argument("--inpaint_radius", type=int, default=3)
    p.add_argument("--block", type=int, default=16)
    p.add_argument("--var_th", type=float, default=2.0)
    p.add_argument("--diff_th", type=float, default=18.0)
    p.add_argument("--small_thresh", type=float, default=0.03)
    p.add_argument("--large_thresh", type=float, default=0.40, help="Proporción máxima antes de descartar frame completo o usar predicción")
    p.add_argument("--proc_scale", type=float, default=0.5, help="0.5=procesar a mitad de resolución")
    p.add_argument("--freeze_margin", type=float, default=0.5, help="Margen de suavizado en detección de freeze")
    p.add_argument("--roi_inpaint", action="store_true", help="reparar sólo ROIs dañadas")
    p.add_argument("--min_roi", type=int, default=300)
    p.add_argument("--roi_margin", type=int, default=4)
    p.add_argument("--gst_w", type=int, default=0, help="reescala vía GStreamer (ancho)")
    p.add_argument("--gst_h", type=int, default=0, help="reescala vía GStreamer (alto)")

    p.add_argument("--rtp", action="store_true",
                        help="Si se pasa, recibe de udpsrc (RTP H.264).")
    p.add_argument("--port", type=int, default=5004,
                        help="Puerto UDP donde escucha udpsrc.")
    p.add_argument("--rtp_caps", type=str, default="",
                        help="Caps RTP completas (si vacías uso un default).")
    args = p.parse_args()

    if not args.rtp and not args.input:
        p.error("Debe especificar --input (archivo local) o --rtp para recibir desde red.")

    Gst.init(None)
    desc = build_pipeline_desc(args)
    print("[GST] pipeline:\n", desc)
    pipeline = Gst.parse_launch(desc)
    appsink  = pipeline.get_by_name("sink")
    pipeline.set_state(Gst.State.PLAYING)
    print("[GST] PLAYING")

    cv2.namedWindow("RAW | VIEW", cv2.WINDOW_NORMAL)

    prev = None
    t0 = time.time()
    n  = 0
    try:
        while True:
            sample = appsink.emit("pull-sample")
            if sample is None:
                print("[GST] EOS o error.")
                break

            raw = bgr_from_sample(sample)
            if raw is None:
                continue

            # --- DETECCIÓN ---
            mask = detect_corruption_mask(
                raw,
                prev_bgr=prev,
                block=args.block,
                var_th=args.var_th,
                diff_th=args.diff_th,
                proc_scale=args.proc_scale
            )
            prev = raw

            # --- MODO VISUALIZACIÓN ---
            if args.mode == "mask":
                right = overlay_mask(raw, mask, color=(0,0,255), alpha=0.45)
                draw_badge(right, "RAW + MASK")

            else:
                ratio = float(np.mean(mask > 0))

                # 🔹 1. Frame limpio o casi limpio
                if ratio < args.small_thresh:
                    right = raw.copy()

                # 🔹 2. Daño moderado: aplicar inpainting (ROI o full-frame)
                elif ratio < args.large_thresh:
                    if args.roi_inpaint:
                        # ROI inpaint — igual que antes
                        ys, xs = np.where(mask > 0)
                        if len(xs) > 0 and len(ys) > 0:
                            x1, y1, x2, y2 = (
                                max(0, xs.min() - args.roi_margin),
                                max(0, ys.min() - args.roi_margin),
                                min(mask.shape[1], xs.max() + args.roi_margin),
                                min(mask.shape[0], ys.max() + args.roi_margin),
                            )
                            roi_frame = raw[y1:y2, x1:x2]
                            roi_mask = mask[y1:y2, x1:x2]

                            if args.proc_scale < 1.0:
                                small = cv2.resize(roi_frame, None, fx=args.proc_scale, fy=args.proc_scale,
                                                   interpolation=cv2.INTER_AREA)
                                msmall = cv2.resize(roi_mask, (small.shape[1], small.shape[0]),
                                                    interpolation=cv2.INTER_NEAREST)
                                repaired_small = cv2.inpaint(small, msmall, args.inpaint_radius, args.inpaint_method)
                                roi_repaired = cv2.resize(repaired_small, (roi_frame.shape[1], roi_frame.shape[0]),
                                                          interpolation=cv2.INTER_LINEAR)
                            else:
                                roi_repaired = cv2.inpaint(roi_frame, roi_mask, args.inpaint_radius, args.inpaint_method)
                            right = raw.copy()
                            right[y1:y2, x1:x2] = roi_repaired
                        else:
                            right = raw.copy()
                    else:
                        # full-frame inpaint
                        if args.proc_scale < 1.0:
                            small = cv2.resize(raw, None, fx=args.proc_scale, fy=args.proc_scale,
                                               interpolation=cv2.INTER_AREA)
                            msmall = cv2.resize(mask, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
                            repaired_small = cv2.inpaint(small, msmall, args.inpaint_radius, args.inpaint_method)
                            right = cv2.resize(repaired_small, (raw.shape[1], raw.shape[0]),
                                               interpolation=cv2.INTER_LINEAR)
                        else:
                            right = cv2.inpaint(raw, mask, args.inpaint_radius, args.inpaint_method)

                # 🔹 3. Daño grande: descartar o freeze (para evitar falsos positivos fuertes)
                else:
                    right = prev.copy() if prev is not None else raw.copy()

                draw_badge(right, f"INPAINT ({ratio*100:.1f}%)")

            # --- Mostrar lado a lado ---
            left = raw.copy()
            draw_badge(left, "RAW")
            combo = hstack_sameh(left, right)
            cv2.imshow("RAW | VIEW", combo)
            if cv2.waitKey(1) == 27:
                break

            # FPS
            n += 1
            if n % 60 == 0:
                dt = time.time() - t0
                print(f"[Perf] {n/dt:.1f} fps (proc), frame {n}")
    finally:
        pipeline.set_state(Gst.State.NULL)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
