import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from tqdm import tqdm

# ================================================================
# CONFIGURACIÓN
# ================================================================
INPUT_FILE = "MexicanHat.csv"
OUTPUT_FILE = "MexicanHat_stats_part.csv"

# ================================================================
# CARGAR DATOS
# ================================================================
print("\n=== Cargando MexicanHat.csv ===")
df = pd.read_csv(INPUT_FILE)

# Columnas Mexican Hat
mexh_cols = [c for c in df.columns if c.startswith("mexh_scale")]

print(f"Escalas Mexican Hat: {len(mexh_cols)}")
print(f"Total filas (ventanas): {len(df)}")

# ================================================================
# CREAR ID DE PARTICIPANTE REAL (PRIMEROS 3 DÍGITOS)
# ================================================================
df["pid"] = df["participant_id"].astype(str).str[:3]

print(f"Participantes únicos detectados: {df['pid'].nunique()}")

# ================================================================
# ESTADÍSTICOS POR PARTICIPANTE
# ================================================================
print("\n=== Calculando estadísticos por participante ===")

rows = []

for pid, group in tqdm(df.groupby("pid")):

    row = {
        "participant_id": pid,
        "label": group["label"].iloc[0]
    }

    values = group[mexh_cols].values

    stats_values = np.concatenate([
        values.mean(axis=0),
        values.std(axis=0),
        skew(values, axis=0),
        kurtosis(values, axis=0),
        values.max(axis=0),
        values.min(axis=0)
    ])

    col_names = (
        [f"{c}_mean" for c in mexh_cols] +
        [f"{c}_std"  for c in mexh_cols] +
        [f"{c}_skew" for c in mexh_cols] +
        [f"{c}_kurt" for c in mexh_cols] +
        [f"{c}_max"  for c in mexh_cols] +
        [f"{c}_min"  for c in mexh_cols]
    )

    for name, val in zip(col_names, stats_values):
        row[name] = val

    rows.append(row)

# ================================================================
# GUARDAR RESULTADO FINAL
# ================================================================
df_participant = pd.DataFrame(rows)
df_participant.to_csv(OUTPUT_FILE, index=False)

print(f"\n Guardado: {OUTPUT_FILE}")
print(f" Filas finales (participantes): {len(df_participant)}")
print("\n=== PROCESO COMPLETADO ===")
