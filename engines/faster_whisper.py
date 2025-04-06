import os
import torch
import numpy as np
from faster_whisper import WhisperModel

def get_device():
    if torch.cuda.is_available():
        print("Using CUDA")
        return "cuda"
    else:
        print("Using CPU")
        return "cpu"

def init_model(_model_path=None, model_size="base"):
    device = get_device()
    print(f"Loading FasterWhisper model: {model_size} on {device}")

    # Disable loading of FFmpeg-based audio
    model = WhisperModel(
        model_size_or_path=model_size,
        device=device,
        compute_type="float16" if device == "cuda" else "int8",
    )

    return model

def run_transcription(model, audio_np: np.ndarray) -> str:
    # Use numpy-based decoding (avoid av completely)
    segments, _ = model.transcribe(audio_np, beam_size=1)
    return " ".join(segment.text.strip() for segment in segments)