import platform
import argparse
import os

def get_engine_from_args_or_auto():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["torch", "cpp"], help="Engine to use (only in development)")
    args, unknown = parser.parse_known_args()

    # Manual override
    if args.engine:
        selected = args.engine
    else:
        selected = "cpp" if platform.system() == "Darwin" else "torch"

    if selected == "cpp":
        from engines.whisper_cpp import init_model, run_transcription
    elif selected == "torch":
        from engines.whisper_torch import init_model, run_transcription
    else:
        raise ValueError("Invalid engine selected")

    print(f"Engine selected: {selected}")
    return init_model, run_transcription