import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

def procesar(df):
    # Separar características e ID usando el nombre de la columna
    features = df.drop(columns=['id'])
    ids = df['id']

    estadisticos = []

    # Procesar cada respuesta por separado
    for i in range(len(df)):
        fila = features.iloc[i, :]

        fila_stats = pd.DataFrame([
            fila.mean(),
            fila.std(),
            skew(fila, bias=False),
            kurtosis(fila, bias=False),
            fila.min(),
            fila.max()
        ], columns=features.columns)

        # Agregar ID a cada una de las 6 filas
        fila_stats['id'] = ids.iloc[i]

        estadisticos.append(fila_stats)

    return pd.concat(estadisticos, ignore_index=True)


# ---- CARGAR TUS ARCHIVOS ----
pros = pd.read_csv("prosody_dynamic_5s.csv")
art = pd.read_csv("articulation_dynamic_5s.csv")
fon = pd.read_csv("phonation_dynamic_5s.csv")

# ---- PROCESAR ----
pros_out = procesar(pros)
art_out = procesar(art)
fon_out = procesar(fon)

# ---- GUARDAR ----
pros_out.to_csv("prosodia_estadisticos.csv", index=False)
art_out.to_csv("articulacion_estadisticos.csv", index=False)
fon_out.to_csv("fonacion_estadisticos.csv", index=False)

print("Eestadísticas generadas y guardadas.")
