import os
import subprocess
import shutil
import argparse
from datetime import datetime

def clean_build_dirs():
    for folder in ["build", "dist", "__pycache__"]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
            print(f"Limpiado: {folder}")

def download_model(url, target):
    if not os.path.exists(target):
        print(f"Descargando modelo desde:\n{url}")
        os.system(f"curl -L -o {target} {url}")
    else:
        print(f"Modelo ya existe: {target}")

def build_executable(entry_point, output_name, console):
    cmd = [
        "pyinstaller",
        entry_point,
        "--onefile",
        f"--name={output_name}",
        "--hidden-import=numpy",
        "--hidden-import=numpy.core._multiarray_umath",
        "--collect-submodules=numpy"
    ]

    if not console:
        cmd.append("--noconsole")

    print(f"Ejecutando: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def postprocess(model_path, model_dest_name):
    shutil.copy(model_path, os.path.join("dist", model_dest_name))
    print(f"Modelo copiado a dist/{model_dest_name}")

def parse_args():
    parser = argparse.ArgumentParser(description="Build script for TWIMS")
    parser.add_argument("--output", type=str, default="twims", help="Nombre del ejecutable (sin extensión)")
    parser.add_argument("--entry", type=str, default="main.py", help="Archivo principal a compilar")
    parser.add_argument("--model-url", type=str, help="URL del modelo a descargar")
    parser.add_argument("--model-name", type=str, default="ggml.bin", help="Nombre del archivo del modelo final")
    parser.add_argument("--download-model", action="store_true", help="Descargar el modelo antes del build")
    parser.add_argument("--model-path", type=str, help="Ruta local del modelo si ya lo tienes")
    parser.add_argument("--console", action="store_true", help="Mostrar consola al ejecutar el .exe")
    parser.add_argument("--clean", action="store_true", help="Limpiar carpetas build/dist antes de compilar")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    if args.clean:
        clean_build_dirs()

    model_path = args.model_path or args.model_name

    if args.download_model:
        if not args.model_url:
            raise ValueError("Debes proporcionar --model-url si usas --download-model")
        download_model(args.model_url, model_path)

    build_executable(args.entry, args.output, args.console)
    postprocess(model_path, args.model_name)

    print(f"\nBuild finalizado: dist/{args.output}.exe + dist/{args.model_name}")