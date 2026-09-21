import pandas as pd
import numpy as np

# =====================================
# ARCHIVOS
# =====================================
INPUT_RTA = "phonet_stats_por_rta.csv"
OUTPUT_PART = "phonet_stats_por_participante.csv"

print(" Cargando CSV por respuesta...")
df = pd.read_csv(INPUT_RTA)

# =====================================
# VALIDACIONES BÁSICAS
# =====================================
required_cols = ["audio_id", "participant_id"]
for c in required_cols:
    if c not in df.columns:
        raise ValueError(f" Falta la columna obligatoria: {c}")

# Todas las columnas de features (excluye IDs)
feature_cols = [c for c in df.columns if c not in ["audio_id", "participant_id"]]

if len(feature_cols) == 0:
    raise ValueError(" No se encontraron columnas de características")

# =====================================
# DEFINIR AGREGACIONES POR TIPO
# =====================================
agg_dict = {}

for col in feature_cols:
    if col.endswith("_min"):
        agg_dict[col] = "min"
    elif col.endswith("_max"):
        agg_dict[col] = "max"
    else:
        # mean, std, skew, kurt
        agg_dict[col] = "mean"

# =====================================
# AGRUPAR POR PARTICIPANTE
# =====================================
print("▶ Calculando estadísticas por participante...")

df_part = (
    df
    .groupby("participant_id", as_index=False)
    .agg(agg_dict)
)

# =====================================
# GUARDAR
# =====================================
df_part.to_csv(OUTPUT_PART, index=False)

print(" Guardado correctamente:", OUTPUT_PART)
print(" Proceso finalizado (estructura preservada)")



