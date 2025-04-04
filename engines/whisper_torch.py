import whisper
import torch
import numpy as np
import tempfile
import os
import wave
from dotenv import load_dotenv

load_dotenv()
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_FORCE_FALLBACK"] = "1"

def get_device():
    if torch.cuda.is_available():
        print("Using CUDA")
        return "cuda"
    print("Using CPU")
    return "cpu"

def init_model(_model_path=None, model_size="base"):
    device = get_device()
    print(f"Loading model '{model_size}' on device: {device}")
    model = whisper.load_model(model_size, device=device)
    return model

def float32_to_int16(audio: np.ndarray) -> np.ndarray:
    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767).astype(np.int16)

def write_wav(audio_np: np.ndarray, filename: str, sample_rate: int = 16000):
    audio_int16 = float32_to_int16(audio_np)
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())

import contextlib
import io
def run_transcription(model, audio_np: np.ndarray) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_path = f.name

    try:
        write_wav(audio_np, temp_path)
        with contextlib.redirect_stderr(io.StringIO()):
            result = model.transcribe(
                temp_path,
                task="transcribe",
                fp16=torch.cuda.is_available(),
                no_speech_threshold=0.6,
                condition_on_previous_text=False,
                initial_prompt=None,
                # language="auto",
                # verbose=False
            )
        return result["text"]
    finally:
        try:
            os.remove(temp_path)
        except Exception as e:
            print(f"⚠️ No se pudo eliminar el archivo temporal: {e}")