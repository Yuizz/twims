import platform
import argparse
import os
import sys
import json

engines = {
    "cpp": {
        "name": "cpp",
        "requires_model_size": False,
        "requires_model_path": True,
        "description": "Binary engine, useful to run on MacOS to use MPS",
    },
    "torch": {
        "name": "torch",
        "requires_model_size": True,
        "requires_model_path": False,
        "description": "Torch engine, useful to run on CPU or GPU",
    },
    "faster": {
        "name": "faster",
        "requires_model_size": True,
        "requires_model_path": False,
        "description": "Whisper Faster engine, useful to run on CPU or GPU",
    }
}

def get_config_path():
    # Determina la ruta del archivo de configuración
    if getattr(sys, 'frozen', False):
        # Si está en modo congelado (ejecutable)
        return os.path.join(os.path.dirname(sys.executable), '_internal', 'config.json')
    else:
        # En modo de desarrollo
        return os.path.join(os.path.dirname(__file__), 'config.json')

def load_engine_from_config():
    config_path = get_config_path()
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            return config.get("engine", "cpp")  # Valor por defecto
    except FileNotFoundError:
        return "cpp"  # Valor por defecto si no existe el archivo

def get_engine_from_args_or_auto(args):
    # Manual override
    if args.engine:
        selected = args.engine
    else:
        selected = load_engine_from_config()

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