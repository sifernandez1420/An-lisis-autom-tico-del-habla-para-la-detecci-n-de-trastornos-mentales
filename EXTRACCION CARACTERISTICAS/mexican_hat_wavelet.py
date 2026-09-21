import os
import numpy as np
import librosa
import pywt
import pandas as pd
from tqdm import tqdm

# ================================================================
# CONFIGURACIÓN
# ================================================================
AUDIO_DIR = "AUDIOS_PREPROCESADOS_5s"       # Carpeta de audios
LABEL_FILE = "labels_balanceado_5s.csv"    # Archivo de labels

wRaw = 1024                                # Tamaño de ventana
scales_mexh = np.arange(1, 65)             # Escalas Mexican Hat

# ================================================================
# 1. CARGAR CSV DE LABELS
# ================================================================
print("\n=== Cargando labels ===")
labels_df = pd.read_csv(LABEL_FILE)
labels_df["PATOLOGY"] = labels_df["PATOLOGY"].astype(int)

label_dict = dict(zip(labels_df["ID PARTICIPANT"], labels_df["PATOLOGY"]))

# ================================================================
# 2. LISTAR AUDIOS
# ================================================================
print("\n=== Buscando audios ===")
audio_files = [f for f in os.listdir(AUDIO_DIR) if f.lower().endswith(".wav")]
print(f"Audios encontrados: {len(audio_files)}")

def find_id(filename):
    for ID in label_dict:
        if ID in filename:
            return ID
    return None

matches = []
for f in audio_files:
    ID = find_id(f)
    if ID is not None:
        matches.append((ID, os.path.join(AUDIO_DIR, f)))

print(f"Audios con ID válido: {len(matches)}")

# ================================================================
# 3. CONTAR VENTANAS
# ================================================================
print("\n=== Contando ventanas ===")
total_windows = 0

for ID, fp in tqdm(matches):
    y, _ = librosa.load(fp, sr=None, mono=True)
    total_windows += len(y) // wRaw

print(f"Total de ventanas: {total_windows}")

# ================================================================
# FUNCIÓN MEXICAN HAT
# ================================================================
def mexican_hat(signal, sr):
    coef, _ = pywt.cwt(signal, scales_mexh, "mexh", sampling_period=1/sr)
    return np.sum(np.abs(coef)**2, axis=1)

# ================================================================
# MATRICES Y LISTAS
# ================================================================
data_mexh = np.zeros((total_windows, len(scales_mexh) + 1))
audio_ids = []
participant_ids = []

# ================================================================
# 4. PROCESAR AUDIOS
# ================================================================
print("\n=== Procesando audios ===")
win_count = 0

for ID, fp in tqdm(matches):

    label = label_dict[ID]
    y, sr = librosa.load(fp, sr=None, mono=True)
    n_windows = len(y) // wRaw

    for k in range(n_windows):
        segment = y[k*wRaw:(k+1)*wRaw]

        mex = mexican_hat(segment, sr)
        data_mexh[win_count, :-1] = mex
        data_mexh[win_count, -1] = label

        audio_ids.append(os.path.basename(fp))
        participant_ids.append(ID)

        win_count += 1

# ================================================================
# 5. GUARDAR CSV FINAL
# ================================================================
print("\n=== Guardando MexicanHat.csv ===")

cols_mexh = (
    ["audio_id", "participant_id"] +
    [f"mexh_scale{i}" for i in range(len(scales_mexh))] +
    ["label"]
)

df_mex = pd.DataFrame(
    np.column_stack([audio_ids, participant_ids, data_mexh]),
    columns=cols_mexh
)

df_mex.to_csv("MexicanHat.csv", index=False)
print("Guardado: MexicanHat.csv")

print("\n=== PROCESO FINALIZADO ===")
