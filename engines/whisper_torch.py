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

def init_model(_model_path=None):
    device = get_device()
    model_name = os.getenv("TWIMS_MODEL_SIZE", "base")
    print(f"☁️ Loading model '{model_name}' on device: {device}")
    model = whisper.load_model(model_name, device=device)
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

def run_transcription(model, audio_np: np.ndarray) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
        write_wav(audio_np, f.name)
        result = model.transcribe(
            f.name,
            task="translate",
            # language="en",
            fp16=torch.cuda.is_available(),
            no_speech_threshold=0.6,
            condition_on_previous_text=False,
            initial_prompt=None,
            verbose=False
        )
        return result["text"]