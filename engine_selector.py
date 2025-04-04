import platform
import argparse
import os

def get_engine_from_args_or_auto(args):
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
    return init_model, run_transcription, selected