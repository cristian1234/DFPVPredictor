import cv2, argparse, numpy as np, os, math, random
from sklearn.model_selection import train_test_split

# Escenas en ÍNDICES DE FRAME (INCLUSIVO): [(start_f, end_f), ...]
scene_ranges_frames = [
    (2,98),
    (213,250),
    (252,423),
    (425,699),
    (701,907),
    (909,1023),
    (1025,1150),  # ~38s

    (1152,1628),
    (1630,1743),
    (1745,1983),
    (1985,2221),
    (2223,2584),
    (2587,2821)
    # Hasta acá ~1:31 min @30fps
]

bad_segments_seconds = []   # p.ej. [("00:08","00:09")]
bad_segments_frames  = []   # p.ej. [(1200, 1215)]

use_color   = True
split_ratio = (0.8, 0.1, 0.1)
seed        = 42

max_sequences     = None   # None = usar todas
balance_by_scene  = True

def time_to_seconds(t: str) -> float:
    t = t.strip()
    if ":" not in t:
        return float(t)
    mm, ss = t.split(":")
    return int(mm) * 60 + float(ss)

def load_video_frames(video_path, frame_size, use_color, normalize=False):
    """
    Devuelve frames en CHW y dtype:
      - normalize=False  -> uint8 [0..255]  (RECOMENDADO para SimVP)
      - normalize=True   -> float32 [0..1]
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"No pude abrir {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        f = cv2.resize(f, frame_size, interpolation=cv2.INTER_AREA)  # (W,H)
        if use_color:
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)   # [H,W,3]
            f = np.transpose(f, (2,0,1))            # -> [3,H,W]
        else:
            f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) # [H,W]
            f = f[np.newaxis, :, :]                 # -> [1,H,W]
        frames.append(f)
    cap.release()

    arr = np.ascontiguousarray(np.array(frames))  # [N,C,H,W]
    if normalize:
        arr = arr.astype(np.float32) / 255.0
    else:
        # guardamos crudo como uint8 (como el script viejo)
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
    return arr, fps

def clamp_ranges_to_video(ranges, nf):
    out = []
    for a,b in ranges:
        a = 0 if a is None else int(a)
        b = nf-1 if b is None else int(b)
        a = max(0, min(nf-1, a))
        b = max(0, min(nf-1, b))
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
    nf = len(frames)
    seq_indices = []
    for (a, b) in scene_ranges:
        a = max(0, min(nf-1, a))
        b = max(0, min(nf-1, b))
        if b - a + 1 < seq_len:
            continue
        end_exclusive = b + 1
        for i in range(a, end_exclusive - seq_len + 1):
            j = i + seq_len
            if allow_mask[i:j].all():
                seq_indices.append((i, j))
    return seq_indices

def stratified_pick_by_scene(seq_indices, scene_ranges, max_total, rng):
    if max_total is None or max_total >= len(seq_indices):
        rng.shuffle(seq_indices)
        return seq_indices

    per_scene = {k: [] for k in range(len(scene_ranges))}
    for si, sj in seq_indices:
        for idx, (a,b) in enumerate(scene_ranges):
            if a <= si and (sj-1) <= b:
                per_scene[idx].append((si,sj)); break

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
    train_idx, val_idx = train_test_split(train_idx, test_size=split_ratio[1]/(split_ratio[0]+split_ratio[1]),
                                          random_state=42, shuffle=True)

    def build_array(sel_idx):
        sel = [seq_indices[k] for k in sel_idx]
        if len(sel) == 0:
            C,H,W = frames.shape[1:]
            return np.empty((0,seq_len,C,H,W), dtype=frames.dtype)
        # apilar manteniendo dtype de frames (uint8 si normalize=False)
        seqs = [frames[i:j] for (i,j) in sel]   # [T,C,H,W]
        arr = np.ascontiguousarray(np.stack(seqs, axis=0))
        return arr

    Xtr = build_array(train_idx)
    Xva = build_array(val_idx)
    Xte = build_array(test_idx)

    np.savez(os.path.join(out_dir, f"{prefix}_train.npz"), data=Xtr)
    np.savez(os.path.join(out_dir, f"{prefix}_val.npz"  ), data=Xva)
    np.savez(os.path.join(out_dir, f"{prefix}_test.npz" ), data=Xte)

    print(f"✅ Guardado en {out_dir}")
    print(f"   Train: {len(train_idx)} secuencias  | dtype={Xtr.dtype} | min/max={Xtr.min() if Xtr.size else '-'} / {Xtr.max() if Xtr.size else '-'}")
    print(f"   Val:   {len(val_idx)} secuencias    | dtype={Xva.dtype} | min/max={Xva.min() if Xva.size else '-'} / {Xva.max() if Xva.size else '-'}")
    print(f"   Test:  {len(test_idx)} secuencias   | dtype={Xte.dtype} | min/max={Xte.min() if Xte.size else '-'} / {Xte.max() if Xte.size else '-'}")

def main(args):
    rng = random.Random(seed)

    # ⚠️ cv2.resize usa (W,H)
    frames, fps = load_video_frames(args.video, (args.in_size_x,args.in_size_y), use_color, normalize=args.normalize)
    nf = len(frames)
    print(f"Frames cargados: {nf} | fps={fps:.3f} | dtype={frames.dtype} | range≈ {frames.min()}..{frames.max()}")

    if not scene_ranges_frames:
        raise RuntimeError("Debes definir al menos una escena en 'scene_ranges_frames'.")
    scenes = clamp_ranges_to_video(scene_ranges_frames, nf)
    if not scenes:
        raise RuntimeError("Las escenas están fuera de rango. Revisá 'scene_ranges_frames'.")

    # máscara de escenas válidas
    mask_allow = np.zeros(nf, dtype=bool)
    for a,b in scenes:
        mask_allow[a:b+1] = True

    if bad_segments_seconds:
        mask_bad_sec = mask_from_bad_seconds(nf, fps, bad_segments_seconds)
        mask_allow &= mask_bad_sec
    if bad_segments_frames:
        mask_bad_fr = mask_from_bad_frames(nf, bad_segments_frames)
        mask_allow &= mask_bad_fr

    seq_indices = sequences_inside_scenes(frames, args.seq_len, scenes, mask_allow)
    print(f"Secuencias candidatas (sin cruzar escenas): {len(seq_indices)}")
    if not seq_indices:
        raise RuntimeError("No hay secuencias válidas. Revisá escenas/bad segments/seq_len.")

    if max_sequences is not None:
        if balance_by_scene:
            seq_indices = stratified_pick_by_scene(seq_indices, scenes, max_sequences, rng)
        else:
            rng.shuffle(seq_indices)
            seq_indices = seq_indices[:max_sequences]
        print(f"Secuencias tras limitar: {len(seq_indices)}")

    save_npz_splits(frames, seq_indices, args.output_dir, split_ratio, args.seq_len, prefix="fpv")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--seq_len", type=int, default=20)     # 10 in + 10 out
    p.add_argument("--in_size_y", type=int, required=True)
    p.add_argument("--in_size_x", type=int, required=True)
    p.add_argument("--normalize", action="store_true",
                   help="Guarda en float32[0,1]. Por defecto guarda uint8[0,255] (recomendado).")
    args = p.parse_args()
    main(args)
