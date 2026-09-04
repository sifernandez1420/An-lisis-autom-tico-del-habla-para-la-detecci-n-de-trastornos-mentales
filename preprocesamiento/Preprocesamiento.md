[normalizacion.py](https://github.com/user-attachments/files/31854157/normalizacion.py)


import os
import numpy as np
import librosa
import soundfile as sf

# Carpeta de entrada y salida
carpeta_entrada = os.path.expanduser("~/Documentos/AUDIOS PARTICIPANTE")
carpeta_salida = os.path.expanduser("~/Documentos/AUDIOS_NORMALIZADOS")

# Crear carpeta de salida 
os.makedirs(carpeta_salida, exist_ok=True)

# Función para normalizar audios[PREPROCESAMIENTO.py](https://github.com/user-attachments/files/31854158/PREPROCESAMIENTO.py)

def normalizar_audio(ruta_entrada, ruta_salida):
    audio, sr = librosa.load(ruta_entrada, sr=None)
    max_abs = np.max(np.abs(audio))
    if max_abs > 0:
        audio_normalizado = audio / max_abs
    else:
        audio_normalizado = audio
    sf.write(ruta_salida, audio_normalizado, sr)
    print(f"Normalizado: {os.path.basename(ruta_entrada)} -> {ruta_salida}")

# Procesar todos los archivos WAV de la carpeta
for archivo in os.listdir(carpeta_entrada):
    if archivo.lower().endswith(".wav"):
        ruta_entrada = os.path.join(carpeta_entrada, archivo)
        ruta_salida = os.path.join(carpeta_salida, archivo)
        normalizar_audio(ruta_entrada, ruta_salida)

print("¡Audios normalizados!")


