import numpy as np
import pyaudio
import webrtcvad
from pywhispercpp.model import Model
import queue
import threading
import contextlib

import os
import sys
from dotenv import load_dotenv

load_dotenv()
def get_model_path():
    # 1. Checa si existe una variable de entorno
    model_env = os.getenv("TWIMS_MODEL_PATH")
    if model_env and os.path.isfile(model_env):
        return model_env

    # 2. Checa si estás en modo PyInstaller (ejecutable)
    if getattr(sys, 'frozen', False):
        exe_dir = os.path.dirname(sys.executable)
    else:
        # 3. En desarrollo, usa la ruta del script
        exe_dir = os.path.dirname(os.path.abspath(__file__))

    # 4. Usa modelo en la misma carpeta que el ejecutable/script
    return os.path.join(exe_dir, "ggml.bin")


MODEL_PATH = get_model_path()

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(f"❌ Modelo no encontrado en: {MODEL_PATH}")

NUM_THREADS = 4
SAMPLE_RATE = 16000
CHUNK_SIZE = 480  # 30 ms requerido por VAD

def list_microphones():
    p = pyaudio.PyAudio()
    mic_list = []
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:  # Solo dispositivos de entrada
            mic_list.append((i, info['name']))
    p.terminate()
    return mic_list

def select_microphone():
    microphones = list_microphones()
    print("Selecciona un micrófono:")
    for index, name in microphones:
        print(f"{index}: {name}")
    
    selected_index = int(input("Ingresa el índice del micrófono: "))
    if selected_index not in dict(microphones).keys():
        raise ValueError("Índice de micrófono no válido.")
    
    return selected_index

# Agrega esta línea para seleccionar el micrófono antes de la configuración de PyAudio
MIC_DEVICE_INDEX = select_microphone()

vad = webrtcvad.Vad(2)
audio_buffer = []
silence_counter = 0
SILENCE_LIMIT = 8
MIN_VOICE_FRAMES = 20

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
    SILENCE_PADDING_DURATION = 0.5  # seconds of padding at the end
    silence_padding = np.zeros(int(SAMPLE_RATE * SILENCE_PADDING_DURATION), dtype=np.float32)

    while True:
        audio_data = audio_queue.get()
        if audio_data is None:
            break
        try:
            audio_np = np.frombuffer(b''.join(audio_data), dtype=np.int16).astype(np.float32) / 32768.0

            if len(audio_np) < SAMPLE_RATE:
                audio_np = np.concatenate((audio_np, silence_padding))

            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stderr(devnull):
                    segments = model.transcribe(audio_np)

            for seg in segments:
                print(f"🎤 {seg.text.strip()}", flush=True)
        except Exception as e:
            print(f"Error during transcription: {e}", flush=True)
        finally:
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