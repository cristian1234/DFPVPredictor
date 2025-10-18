
# config/fpv/SimVP_gSTA_720p.py

method = 'SimVP'
model_type = 'gSTA'

# --- MODEL SETTINGS ---
spatio_kernel_enc = 3
spatio_kernel_dec = 3

# Reducimos los hidden dims porque 720p tiene muchísimos píxeles:
# si dejas los valores originales, se dispara el tamaño de los feature maps.
hid_S = 64       # antes 64
hid_T = 640      # antes 512
N_T = 8
N_S = 4

# --- TRAINING SETTINGS ---
lr = 1e-3
batch_size = 1
drop_path = 0
sched = 'onecycle'  # puede quedar igual