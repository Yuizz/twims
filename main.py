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
from engine_selector import get_engine_from_args_or_auto
from argparser import parse_args

import subprocess
# Función para detectar la versión de CUDA
def get_cuda_version():
    try:
        output = subprocess.check_output(['nvcc', '--version']).decode('utf-8')
        for line in output.splitlines():
            if 'release' in line:
                return line.split('release')[-1].strip().split(',')[0]
    except Exception as e:
        print(f"Error al detectar la versión de CUDA: {e}")
        return None

# Imprimir la versión de CUDA al inicio
cuda_version = get_cuda_version()
if cuda_version:
    print(f"Versión de CUDA detectada: {cuda_version}")
else:
    print("No se pudo detectar la versión de CUDA.")

args = parse_args()
init_model, run_transcription, engine, engine_info = get_engine_from_args_or_auto(args)

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

if engine_info["requires_model_path"] and not os.path.isfile(MODEL_PATH):
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

def select_model_size():
    print("Selecciona un tamaño de modelo:")
    model_sizes = [
        {
            "name": "tiny",
            "description": "El más pequeño y rápido, ideal para dispositivos más lentos",
            "ram_usage": "~1GB",
        },
        {
            "name": "base",
            "description": "El tamaño estándar, equilibrado entre velocidad y precisión",
            "ram_usage": "~1GB",
        },
        {
            "name": "small",
            "description": "Un poco más grande, mejor para dispositivos más potentes",
            "ram_usage": "~2GB",
        },
        {
            "name": "medium",
            "description": "Ideal para dispositivos de gama media",
            "ram_usage": "~5GB",
        },
        {
            "name": "large",
            "description": "El más grande, mejor para dispositivos de gama alta, \n\t recomendado no usar al mismo tiempo que otros programas pesados.",
            "ram_usage": "~10GB",
        },
        {
            "name": "turbo",
            "description": "El más grande, pero optimizado, mejor para dispositivos de gama alta, \n\t más ligero que el large, obligatorio el uso de GPU.",
            "ram_usage": "~6GB",
        },
    ]
    for index, size in enumerate(model_sizes):
        print(f"{index}: {size['name']} - {size['ram_usage']} RAM: {size['description']}")
    
    selected_index = int(input("Ingresa el índice del tamaño de modelo: "))
    if selected_index not in range(len(model_sizes)):
        raise ValueError("Índice de tamaño de modelo no válido.")
    return model_sizes[selected_index]["name"]

# Agrega esta línea para seleccionar el micrófono antes de la configuración de PyAudio
MIC_DEVICE_INDEX = select_microphone()

vad = webrtcvad.Vad(2)
audio_buffer = []
silence_counter = 0
SILENCE_LIMIT = 8
MIN_VOICE_FRAMES = 20

# Inicializa Whisper
model_size = None
if engine_info["requires_model_size"]:
    model_size = select_model_size()

model = init_model(MODEL_PATH, model_size=model_size)

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

            text = run_transcription(model, audio_np)
            print(f"{text.strip()}", flush=True)
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

print("Escuchando... (presiona Ctrl+C para salir)")

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
    print("\nDeteniendo...")
finally:
    audio_queue.put(None)  # Cierra el hilo
    stream.stop_stream()
    stream.close()
    p.terminate()