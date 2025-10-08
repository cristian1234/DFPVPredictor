
# config/fpv/SimVP_gSTA_720p.py

method = 'SimVP'
model_type = 'gSTA'

# --- MODEL SETTINGS ---
spatio_kernel_enc = 3
spatio_kernel_dec = 3

# Reducimos los hidden dims porque 720p tiene muchísimos píxeles:
# si dejas los valores originales, se dispara el tamaño de los feature maps.
hid_S = 64
hid_T = 512
N_T = 8
N_S = 4

# --- TRAINING SETTINGS ---
lr = 1e-3
batch_size = 16
drop_path = 0.05    # un poquito de regularización ayuda
sched = 'onecycle'  # puede quedar igual

# --- TEMPORAL SETTINGS (lo importante para vos) ---
T_in = 8
T_out = 4
# total sequence length (si el código usa seq_len = T_in + T_out)
seq_len = T_in + T_out

# --- AMP y demás (si tu script lo soporta) ---
amp = True