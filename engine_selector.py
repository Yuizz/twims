import platform
import argparse
import os

engines = {
    "cpp": {
        "name": "cpp",
        "requires_model_size": False,
        "description": "Binary engine, useful to run on MacOS to use MPS",
    },
    "torch": {
        "name": "torch",
        "requires_model_size": True,
        "description": "Torch engine, useful to run on CPU or GPU",
    },
    "faster": {
        "name": "faster",
        "requires_model_size": True,
        "description": "Whisper Faster engine, useful to run on CPU or GPU",
    }
}
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
    elif selected == "faster":
        from engines.faster_whisper import init_model, run_transcription
    else:
        raise ValueError("Invalid engine selected")

    print(f"Engine selected: {selected}")
    return init_model, run_transcription, selected, engines[selected]