import cv2,argparse, numpy as np, os, math, random
from sklearn.model_selection import train_test_split

# ============ CONFIG ============
#VIDEO_PATH = "./drone_corto.mp4"
#OUTPUT_DIR = "./data/fpv_corto"

# Escenas en ÍNDICES DE FRAME (INCLUSIVO): [(start_f, end_f), ...]
# Ejemplo: primera escena de 0 a 899, segunda de 900 a 1799, etc.
scene_ranges_frames = [
    (2,98),
    (213,250),
(252,423),
(425,699),
(701,907),
(909,1023),
(1025,1150), #38 seconds
#(1152,1628),
#(1630,1743),
#(1745,1983),
#(1985,2221),
#(2223,2584),
#(2587,2821)
    #Hasta aca 1:31 minutos de video, frames a 30fps
]

# (Opcional) segmentos a excluir por TIEMPO (mm:ss o mm:ss.mmm)
bad_segments_seconds = [
    # ("00:08","00:09"),
    # ("00:14","00:15"),
]

# (Opcional) segmentos a excluir por FRAMES (INCLUSIVO)
bad_segments_frames = [
    # (1200, 1215),
]

#frame_size = (1280, 720)   # (W, H) salida; usa algo chico si querés entrenar rápido (p.ej. (256,144))
use_color  = True          # True=RGB (3 canales), False=Grayscale (1 canal)
#seq_len    = 12            # Tin + Tout (ej: 8 + 4)
split_ratio = (0.8, 0.1, 0.1)
seed = 42

# (Opcional) limitar cantidad total de secuencias
max_sequences = None   # ej. 200  (None = usar todas)
balance_by_scene = True
# =================================


def time_to_seconds(t: str) -> float:
    t = t.strip()
    if ":" not in t:
        return float(t)
    mm, ss = t.split(":")
    return int(mm) * 60 + float(ss)

def load_video_frames(video_path, frame_size, use_color):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"No pude abrir {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        f = cv2.resize(f, frame_size, interpolation=cv2.INTER_AREA)
        if use_color:
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)   # [H,W,3]
            f = np.transpose(f, (2,0,1))            # -> [3,H,W]
        else:
            f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) # [H,W]
            f = f[np.newaxis, :, :]                 # -> [1,H,W]
        frames.append(f)
    cap.release()
    frames = np.array(frames, dtype=np.float32) / 255.0  # [N,C,H,W] en [0,1]
    return frames, fps

def clamp_ranges_to_video(ranges, nf):
    out = []
    for a,b in ranges:
        if a is None: a = 0
        if b is None: b = nf-1
        a = max(0, min(nf-1, int(a)))
        b = max(0, min(nf-1, int(b)))
        if b >= a:
            out.append((a,b))
    return out

def mask_from_bad_seconds(nf, fps, bad_seconds):
    mask = np.ones(nf, dtype=bool)
    for start, end in bad_seconds:
        i0 = int(math.floor(time_to_seconds(start) * fps))
        i1 = int(math.ceil (time_to_seconds(end)   * fps))
        i0 = max(0, min(nf, i0))
        i1 = max(0, min(nf, i1))
        if i1 > i0:
            mask[i0:i1] = False
    return mask

def mask_from_bad_frames(nf, bad_frames):
    mask = np.ones(nf, dtype=bool)
    for a,b in bad_frames:
        a = max(0, min(nf-1, int(a)))
        b = max(0, min(nf-1, int(b)))
        if b >= a:
            mask[a:b+1] = False
    return mask

def sequences_inside_scenes(frames, seq_len, scene_ranges, allow_mask):
    """
    Genera índices [i, j) (j excluyente) de secuencias de longitud seq_len
    que están 100% contenidas dentro de cada (a,b) de scene_ranges
    y solo con frames permitidos por allow_mask.
    """
    nf = len(frames)
    seq_indices = []
    for (a, b) in scene_ranges:
        a = max(0, min(nf-1, a))
        b = max(0, min(nf-1, b))
        if b - a + 1 < seq_len:
            continue
        # ventana deslizante dentro de [a,b]
        start = a
        end_exclusive = b + 1
        for i in range(start, end_exclusive - seq_len + 1):
            j = i + seq_len
            if allow_mask[i:j].all():
                seq_indices.append((i, j))
    return seq_indices

def stratified_pick_by_scene(seq_indices, scene_ranges, max_total, rng):
    if max_total is None or max_total >= len(seq_indices):
        rng.shuffle(seq_indices)
        return seq_indices

    # asignar cada secuencia a una escena por su inicio
    per_scene = {k: [] for k in range(len(scene_ranges))}
    for si, sj in seq_indices:
        for idx, (a,b) in enumerate(scene_ranges):
            if a <= si and (sj-1) <= b:
                per_scene[idx].append((si,sj))
                break

    picked, leftovers = [], []
    k = max(1, len(per_scene))
    per_cap = max(1, max_total // k)

    for idx, lst in per_scene.items():
        rng.shuffle(lst)
        take = lst[:per_cap]
        picked.extend(take)
        leftovers.extend(lst[per_cap:])

    if len(picked) < max_total:
        rng.shuffle(leftovers)
        picked.extend(leftovers[: max_total - len(picked)])

    rng.shuffle(picked)
    return picked[:max_total]

def save_npz_splits(frames, seq_indices, out_dir, split_ratio, seq_len, prefix="fpv"):
    os.makedirs(out_dir, exist_ok=True)
    n = len(seq_indices)
    idx = np.arange(n)
    train_idx, test_idx = train_test_split(idx, test_size=split_ratio[2], random_state=42, shuffle=True)
    train_idx, val_idx = train_test_split(train_idx, test_size=split_ratio[1]/(split_ratio[0]+split_ratio[1]), random_state=42, shuffle=True)

    def build_array(sel_idx):
        sel = [seq_indices[k] for k in sel_idx]  # 🔥 CORRECCIÓN
        if len(sel) == 0:
            C,H,W = frames.shape[1:]
            return np.empty((0,seq_len,C,H,W), dtype=np.float32)
        seqs = [frames[i:j] for (i,j) in sel]   # [T,C,H,W]
        return np.stack(seqs, axis=0).astype(np.float32)

    Xtr = build_array(train_idx)
    Xva = build_array(val_idx)
    Xte = build_array(test_idx)

    np.savez(os.path.join(out_dir, f"{prefix}_train.npz"), data=Xtr)
    np.savez(os.path.join(out_dir, f"{prefix}_val.npz"  ), data=Xva)
    np.savez(os.path.join(out_dir, f"{prefix}_test.npz" ), data=Xte)

    print(f"✅ Guardado en {out_dir}")
    print(f"   Train: {len(train_idx)} secuencias")
    print(f"   Val:   {len(val_idx)} secuencias")
    print(f"   Test:  {len(test_idx)} secuencias")

def main(args):
    rng = random.Random(seed)

    frames, fps = load_video_frames(args.video, (args.in_size_x,args.in_size_y), use_color)   # [N,C,H,W] en [0,1]
    nf = len(frames)
    print(f"Frames cargados: {nf} | fps={fps:.3f}")

    # 1) Validación y normalización de escenas en FRAMES
    if not scene_ranges_frames:
        raise RuntimeError("Debes definir al menos una escena en 'scene_ranges_frames'.")
    scenes = clamp_ranges_to_video(scene_ranges_frames, nf)
    if not scenes:
        raise RuntimeError("Las escenas están fuera de rango. Revisá 'scene_ranges_frames'.")

    # 2) Máscaras de exclusión
    mask_allow = np.zeros(nf, dtype=bool)
    # habilitar SOLO frames dentro de escenas
    for a,b in scenes:
        mask_allow[a:b+1] = True

    # restar segmentos malos por segundos (si hay)
    if bad_segments_seconds:
        mask_bad_sec = mask_from_bad_seconds(nf, fps, bad_segments_seconds)
        mask_allow &= mask_bad_sec  # (mask_bad_sec tiene True donde se permite)

    # restar segmentos malos por frames (si hay)
    if bad_segments_frames:
        mask_bad_fr = mask_from_bad_frames(nf, bad_segments_frames)
        mask_allow &= mask_bad_fr

    # 3) Generar secuencias que NO crucen escenas
    seq_indices = sequences_inside_scenes(frames, args.seq_len, scenes, mask_allow)
    print(f"Secuencias candidatas (sin cruzar escenas): {len(seq_indices)}")
    if not seq_indices:
        raise RuntimeError("No hay secuencias válidas. Revisá escenas/bad segments/seq_len.")

    # 4) Limitar y balancear (opcional)
    if max_sequences is not None:
        if balance_by_scene:
            seq_indices = stratified_pick_by_scene(seq_indices, scenes, max_sequences, rng)
        else:
            rng.shuffle(seq_indices)
            seq_indices = seq_indices[:max_sequences]
        print(f"Secuencias tras limitar: {len(seq_indices)}")

    # 5) Guardar splits
    save_npz_splits(frames, seq_indices, args.output_dir, split_ratio, args.seq_len, prefix="fpv")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--seq_len", type=int, default=12)
    p.add_argument("--in_size_y", type=int,required=True)
    p.add_argument("--in_size_x", type=int,required=True)
    args = p.parse_args()
    main(args)
