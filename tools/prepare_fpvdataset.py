import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split

# === CONFIGURACIÓN ===
VIDEO_PATH = "./drone.mp4"
OUTPUT_DIR = "./data/fpv"
bad_segments = [("00:00","00:03"),("00:08","00:09"),("00:14","00:15"),("00:23","00:24"),("00:30","00:31"),
                ("00:34","00:35"),("00:38","00:39"),("00:54","00:55"),("00:58","00:59"),("01:06","01:07"),("01:14","01:15"),
                ("01:26","01:27"),("01:34","01:35"),("01:38","01:39"),("01:50","01:51"),("01:54","01:55"),("02:08","02:09"),
                ("02:12","02:13"),("02:16","02:17"),("02:20","02:21"),("02:24","02:25"),("02:31","02:32"),("02:34","02:37"),
                ("02:40","02:41"),("02:44","02:45"),("02:48","02:49"),("02:51","02:52"),("02:56","02:57"),("03:00","03:01"),
                ("03:04","03:05"),("03:08","03:09"),("03:12","03:13"),("03:16","03:17"),("03:18","03:25"),("03:35","03:36")]  # lista de cortes MM:SS
frame_size = (64, 64)  # resolución (ancho, alto)
use_color = True       # True = RGB (3 canales), False = Grayscale (1 canal)
seq_len = 20           # 10 input + 10 output
split_ratio = (0.8, 0.1, 0.1)  # train, val, test
# ======================

def time_to_seconds(t):
    mm, ss = t.split(":")
    return int(mm) * 60 + int(ss)

def load_and_clean_video(video_path, bad_segments, frame_size, use_color):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []

    while True:
        ret, f = cap.read()
        if not ret:
            break
        f = cv2.resize(f, frame_size)
        if use_color:
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)   # [H,W,3]
            f = np.transpose(f, (2,0,1))            # -> [3,H,W]
        else:
            f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) # [H,W]
            f = f[np.newaxis, :, :]                 # -> [1,H,W]
        frames.append(f)
    cap.release()

    frames = np.array(frames)  # [N,C,H,W]

    # Máscara para descartar segmentos
    mask = np.ones(len(frames), dtype=bool)
    for start, end in bad_segments:
        i0, i1 = int(time_to_seconds(start) * fps), int(time_to_seconds(end) * fps)
        mask[i0:i1] = False

    return frames[mask]  # [N,C,H,W]

def make_sequences(frames, seq_len):
    seqs = []
    for i in range(len(frames) - seq_len + 1):
        seq = frames[i:i+seq_len]   # [T,C,H,W]
        seqs.append(seq)
    return np.array(seqs)  # [N,T,C,H,W]

def split_and_save(seqs, output_dir, split_ratio):
    os.makedirs(output_dir, exist_ok=True)

    # Dividir en train/val/test
    n = len(seqs)
    idx = np.arange(n)
    train_idx, test_idx = train_test_split(idx, test_size=split_ratio[2], random_state=42)
    train_idx, val_idx = train_test_split(train_idx, test_size=split_ratio[1]/(split_ratio[0]+split_ratio[1]), random_state=42)

    np.savez(os.path.join(output_dir, "fpv_train.npz"), data=seqs[train_idx])
    np.savez(os.path.join(output_dir, "fpv_val.npz"), data=seqs[val_idx])
    np.savez(os.path.join(output_dir, "fpv_test.npz"), data=seqs[test_idx])

    print(f"✅ Guardado en {output_dir}")
    print(f"   Train: {len(train_idx)} secuencias")
    print(f"   Val:   {len(val_idx)} secuencias")
    print(f"   Test:  {len(test_idx)} secuencias")

if __name__ == "__main__":
    frames = load_and_clean_video(VIDEO_PATH, bad_segments, frame_size, use_color)
    print(f"Frames limpios: {frames.shape}")  # [N,C,H,W]

    seqs = make_sequences(frames, seq_len)
    print(f"Total secuencias: {seqs.shape}") # [N,T,C,H,W]

    split_and_save(seqs, OUTPUT_DIR, split_ratio)



