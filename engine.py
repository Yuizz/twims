from pywhispercpp.model import Model
import os
import contextlib

NUM_THREADS = 4

def init_model(model_path: str):
    print("🍎 Usando pywhispercpp (Metal/MPS)")
    return Model(
        model_path,
        n_threads=NUM_THREADS,
        translate=True,
        language="auto",
        print_realtime=False,
        single_segment=True,
        no_context=True
    )

def run_transcription(model, audio_np):
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stderr(devnull):
            segments = model.transcribe(audio_np)
    return "".join([seg.text for seg in segments])