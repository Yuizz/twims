import numpy as np
import pyaudio
import webrtcvad
from pywhispercpp.model import Model
import queue
import threading
import os
import contextlib

MODEL_PATH = "models/ggml-large-v3-turbo.bin"
NUM_THREADS = 4
SAMPLE_RATE = 16000
CHUNK_SIZE = 480  # 30 ms requerido por VAD
MIC_DEVICE_INDEX = 1  # tu índice verificado del micrófono

vad = webrtcvad.Vad(3)
audio_buffer = []
silence_counter = 0
SILENCE_LIMIT = 5
MIN_VOICE_FRAMES = 16

# Inicializa Whisper
model = Model(
    MODEL_PATH,
    n_threads=NUM_THREADS, 
    translate=True,
    language="auto",
    print_realtime=False,
    single_segment=True,
    no_context=True
    )

audio_queue = queue.Queue()

def transcribe_worker():
    SILENCE_PADDING_DURATION = 0.5  # segundos de padding al final
    silence_padding = np.zeros(int(SAMPLE_RATE * SILENCE_PADDING_DURATION), dtype=np.float32)

    while True:
        audio_data = audio_queue.get()
        if audio_data is None:
            break
        audio_np = np.frombuffer(b''.join(audio_data), dtype=np.int16).astype(np.float32) / 32768.0

        if len(audio_np) < SAMPLE_RATE:
            audio_np = np.concatenate((audio_np, silence_padding))

        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stderr(devnull):
                segments = model.transcribe(audio_np)

        for seg in segments:
            print(f"🎤 {seg.text.strip()}", flush=True)
        audio_queue.task_done()

# Hilo separado para Whisper
threading.Thread(target=transcribe_worker, daemon=True).start()

# PyAudio setup
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=SAMPLE_RATE,
                input=True,
                frames_per_buffer=CHUNK_SIZE,
                input_device_index=MIC_DEVICE_INDEX)

print("🔎 Escuchando... (presiona Ctrl+C para salir)")

try:
    while True:
        try:
            data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
        except IOError as e:
            print(f"⚠️ Overflow evitado: {e}")
            continue

        if vad.is_speech(data, SAMPLE_RATE):
            audio_buffer.append(data)
            silence_counter = 0
        else:
            if audio_buffer:
                silence_counter += 1
                if silence_counter > SILENCE_LIMIT:
                    if len(audio_buffer) >= MIN_VOICE_FRAMES:
                        audio_queue.put(audio_buffer.copy())  # Envía copia al hilo
                    audio_buffer.clear()
                    silence_counter = 0
except KeyboardInterrupt:
    print("\n🛑 Deteniendo...")
finally:
    audio_queue.put(None)  # Cierra el hilo
    stream.stop_stream()
    stream.close()
    p.terminate()