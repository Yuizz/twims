import sounddevice as sd
import numpy as np
import webrtcvad
from pywhispercpp.model import Model

# Configuraciones
MODEL_PATH = "models/ggml-large-v3-turbo-q8_0.bin"   # Ruta al modelo GGML descargado (ajusta esto)
NUM_THREADS = 4                            # Número de hilos de CPU a usar (ajusta según tu CPU)
VAD_AGGRESSIVENESS = 3                     # Nivel de agresividad del VAD (0=menos estricto, 3=más estricto)
SILENCE_THRESHOLD = 8                      # Umbral de bloques de silencio para cortar (8 bloques = ~0.24s con block_size=30ms)
MIN_VOICE_FRAMES = 16                      # Mínimo de bloques con voz antes de transcribir (16 bloques = ~0.5s de voz)
BLOCK_DURATION_MS = 30                     # Duración de cada bloque de audio en milisegundos

# Inicializar modelo Whisper.cpp (modo traducción al inglés habilitado)
model = Model(MODEL_PATH, n_threads=NUM_THREADS, translate=True,  # traducir siempre a inglés
              print_realtime=False, print_progress=False, print_timestamps=False,
              single_segment=True, no_context=True)

vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)  # Inicializar detector de voz
sample_rate = 16000  # Whisper modelo usa 16 kHz
block_size = int(sample_rate * BLOCK_DURATION_MS / 1000)
audio_queue = []     # Cola para acumular audio de voz detectada
silence_count = 0    # Contador de bloques de silencio consecutivos

def audio_callback(indata, frames, time, status):
    """Callback de grabación - se llama para cada bloque de audio capturado."""
    global silence_count, audio_queue
    if status:
        # Mensajes de advertencia de sounddevice (subdesbordes, etc)
        print(f"[Audio Warning] {status}", flush=True)
    # Convertir audio a formato int16 PCM para VAD (de floats -1 a 1)
    audio_pcm = (indata * 32768).astype(np.int16)  # convertir a enteros 16-bit
    # Asegurar límites [-32768, 32767]
    audio_pcm = np.clip(audio_pcm, -32768, 32767)
    pcm_bytes = audio_pcm.tobytes()

    # Detección de voz en el bloque actual
    has_voice = vad.is_speech(pcm_bytes, sample_rate)
    if has_voice:
        # Si hay voz, acumulamos el bloque y reiniciamos contador de silencios
        audio_queue.append(indata.copy())
        silence_count = 0
    else:
        if audio_queue:
            # Si no hay voz y tenemos audio en cola, incrementamos contador de silencio
            silence_count += 1
            if silence_count >= SILENCE_THRESHOLD:
                # Ha habido suficiente silencio: procesar transcripción
                if len(audio_queue) >= MIN_VOICE_FRAMES:
                    # Concatenar audio de la cola
                    voice_segments = np.concatenate(audio_queue, axis=None)
                    # Ejecutar transcripción con Whisper.cpp
                    segments = model.transcribe(voice_segments)
                    # Imprimir cada segmento transcrito
                    for seg in segments:
                        print(seg.text.strip(), flush=True)
                # Limpiar cola y reiniciar contador
                audio_queue = []
                silence_count = 0
        else:
            # Sin voz ni cola acumulada: no hacer nada (micrófono inactivo)
            silence_count = 0

# Iniciar captura de audio del micrófono con el callback
try:
    print("🔎 Escuchando... (habla en español o inglés, la transcripción se mostrará en inglés)")
    MIC_DEVICE_INDEX = 1
    with sd.InputStream(device=MIC_DEVICE_INDEX, channels=1, samplerate=sample_rate,
                        blocksize=block_size, callback=audio_callback):
        sd.sleep(-1)  # Mantener activo hasta interrupción manual
except KeyboardInterrupt:
    print("\nTranscripción finalizada.")
