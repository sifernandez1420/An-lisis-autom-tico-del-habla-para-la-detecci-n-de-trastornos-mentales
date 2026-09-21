import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from tqdm import tqdm

# ================================================================
# CONFIGURACIÓN
# ================================================================
INPUT_FILE = "MexicanHat.csv"
OUTPUT_FILE = "MexicanHat_stats.csv"

# ================================================================
# CARGAR DATOS
# ================================================================
print("\n=== Cargando MexicanHat.csv ===")
df = pd.read_csv(INPUT_FILE)

# Columnas Mexican Hat
mexh_cols = [c for c in df.columns if c.startswith("mexh_scale")]

print(f"Número de escalas Mexican Hat: {len(mexh_cols)}")
print(f"Número total de ventanas: {len(df)}")

# ================================================================
# AGRUPAR POR AUDIO Y CALCULAR ESTADÍSTICOS
# ================================================================
print("\n=== Calculando estadísticos por audio ===")

rows = []

for audio_id, group in tqdm(df.groupby("audio_id")):

    row = {
        "audio_id": audio_id,
        "participant_id": group["participant_id"].iloc[0],
        "label": group["label"].iloc[0]
    }

    values = group[mexh_cols].values

    # Estadísticos por escala
    row_stats = np.concatenate([
        values.mean(axis=0),
        values.std(axis=0),
        skew(values, axis=0),
        kurtosis(values, axis=0),
        values.max(axis=0),
        values.min(axis=0)
    ])

    # Nombres de columnas
    col_names = (
        [f"{c}_mean" for c in mexh_cols] +
        [f"{c}_std" for c in mexh_cols] +
        [f"{c}_skew" for c in mexh_cols] +
        [f"{c}_kurt" for c in mexh_cols] +
        [f"{c}_max" for c in mexh_cols] +
        [f"{c}_min" for c in mexh_cols]
    )

    for name, val in zip(col_names, row_stats):
        row[name] = val

    rows.append(row)

# ================================================================
# CREAR DATAFRAME FINAL
# ================================================================
df_audio = pd.DataFrame(rows)

df_audio.to_csv(OUTPUT_FILE, index=False)

print(f"\n Guardado: {OUTPUT_FILE}")
print(f"Filas finales (audios): {len(df_audio)}")
print("\n=== PROCESO COMPLETADO ===")
