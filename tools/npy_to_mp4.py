import numpy as np
import cv2
import os

# === CONFIG ===
RESULTS_DIR = "./results/fpv_simvp_gsta/saved"  # 🔹 Cambiá a tu experimento
OUT_DIR = os.path.join(RESULTS_DIR, "videos")
fps = 10
scale = 4   # upscale para visualización
max_samples = 5  # cuántos ejemplos exportar
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.6
font_color = (255, 255, 255)  # blanco
thickness = 2
bg_color = (0, 0, 0)  # negro para el fondo
alpha = 0.6  # transparencia del fondo
# ==============

os.makedirs(OUT_DIR, exist_ok=True)

# Cargamos npy
inputs = np.load(os.path.join(RESULTS_DIR, "inputs.npy"))   # [N, Tin, C, H, W]
trues  = np.load(os.path.join(RESULTS_DIR, "trues.npy"))    # [N, Tout, C, H, W]
preds  = np.load(os.path.join(RESULTS_DIR, "preds.npy"))    # [N, Tout, C, H, W]

def to_img(x):
    """Convierte [C,H,W] -> BGR listo para OpenCV."""
    x = (x * 255).clip(0, 255).astype(np.uint8)
    if x.shape[0] == 1:  # grayscale
        x = x[0]
        x = cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)
    else:  # RGB
        x = x.transpose(1, 2, 0)[:, :, ::-1]  # CHW->HWC y RGB->BGR
    return cv2.resize(x, (x.shape[1]*scale, x.shape[0]*scale), interpolation=cv2.INTER_NEAREST)

def add_label(img, text):
    """Dibuja un texto con fondo semitransparente arriba del frame."""
    overlay = img.copy()
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    # rectángulo detrás del texto
    cv2.rectangle(overlay, (5, 5), (10+tw, 10+th), bg_color, -1)
    # mezcla overlay con la imagen
    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    # escribir texto arriba del rectángulo
    cv2.putText(img, text, (10, 10+th-2), font, font_scale, font_color, thickness, cv2.LINE_AA)
    return img

N, Tin, C, H, W = inputs.shape
Tout = preds.shape[1]

for i in range(min(max_samples, N)):
    out_path = os.path.join(OUT_DIR, f"sample_{i}.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W*scale*3, H*scale))

    # === Mostrar los inputs primero ===
    for f in range(Tin):
        img_in = add_label(to_img(inputs[i, f]), "Input")
        blank1 = add_label(np.zeros_like(img_in), "Ground Truth")
        blank2 = add_label(np.zeros_like(img_in), "Prediction")
        frame = np.concatenate([img_in, blank1, blank2], axis=1)
        writer.write(frame)

    # === Luego mostrar GT vs Pred ===
    for f in range(Tout):
        gt = add_label(to_img(trues[i, f]), "Ground Truth")
        pr = add_label(to_img(preds[i, f]), "Prediction")
        blank_in = add_label(np.zeros_like(gt), "Input")
        frame = np.concatenate([blank_in, gt, pr], axis=1)
        writer.write(frame)

    writer.release()
    print(f"✅ Guardado {out_path}")
