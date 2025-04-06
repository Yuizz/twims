import os
import subprocess
import shutil
import argparse

def clean_build_dirs():
    for folder in ["build", "dist", "__pycache__"]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
            print(f"Cleaned: {folder}")

def download_model(url, target):
    if not os.path.exists(target):
        print(f"Downloading model from:\n{url}")
        os.system(f"curl -L -o {target} {url}")
    else:
        print(f"Model already exists: {target}")

def build_executable(entry_point, output_name, console, engine):
    sep = ";" if os.name == "nt" else ":"

    import whisper
    whisper_dir = os.path.dirname(whisper.__file__)

    cmd = [
        "pyinstaller",
        entry_point,
        "--onedir",
        f"--name={output_name}",
        "--hidden-import=numpy",
        "--hidden-import=numpy.core._multiarray_umath",
        "--collect-submodules=numpy",
        "--exclude-module=tkinter",
        "--exclude-module=matplotlib",
        "--exclude-module=scipy",
        "--exclude-module=pytest",
        "--exclude-module=setuptools",
        "--exclude-module=pkg_resources",
        "--exclude-module=torchvision",
        "--exclude-module=torchtext",
        "--exclude-module=torchaudio",
        "--exclude-module=scipy",
        "--exclude-module=matplotlib",
    ]

    if(engine == "torch"):
        cmd.append("--hidden-import=whisper")
        cmd.append("--collect-submodules=whisper")
        cmd.append("--hidden-import=torch")
        cmd.append("--collect-submodules=torch")
        cmd.append("--upx-dir=upx")  # Asumiendo que lo descargas antes

        for asset in os.listdir(os.path.join(whisper_dir, "assets")):
            full_path = os.path.join(whisper_dir, "assets", asset)
            cmd.append(f"--add-data={full_path}{sep}whisper/assets")

    if not console:
        cmd.append("--noconsole")

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def postprocess(model_path, model_dest_name):
    if model_path and os.path.isfile(model_path):
        shutil.copy(model_path, os.path.join("dist", model_dest_name))
        print(f"Model copied to dist/{model_dest_name}")

def parse_args():
    parser = argparse.ArgumentParser(description="TWIMS Build Script")

    parser.add_argument("--engine", choices=["torch", "cpp", "faster"], required=True,
                        help="Engine to inject: 'torch' or 'cpp' or 'faster'")
    parser.add_argument("--model-size", type=str, default="base",
                        help="Whisper model size (torch engine only): tiny, base, small, medium, large")
    parser.add_argument("--output", type=str, default="twims",
                        help="Output binary name")
    parser.add_argument("--entry", type=str, default="main.py",
                        help="Entry point script")
    parser.add_argument("--model-url", type=str,
                        help="URL to download model (cpp engine only)")
    parser.add_argument("--model-name", type=str, default="ggml.bin",
                        help="Model filename for cpp engine")
    parser.add_argument("--download-model", action="store_true",
                        help="Download model before build (cpp engine only)")
    parser.add_argument("--model-path", type=str,
                        help="Path to local model (cpp engine only)")
    parser.add_argument("--console", action="store_true",
                        help="Show console in executable")
    parser.add_argument("--clean", action="store_true",
                        help="Clean build/dist directories before building")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    if args.clean:
        clean_build_dirs()

    if args.engine == "cpp":
        model_path = args.model_path or args.model_name
        if args.download_model:
            if not args.model_url:
                raise ValueError("You must provide --model-url when using --download-model")
            download_model(args.model_url, model_path)
        postprocess(model_path, args.model_name)

    build_executable(args.entry, args.output, args.console, args.engine)

    print(f"\nBuild complete: dist/{args.output}/ (engine: {args.engine})")
    if args.engine == "cpp":
        print(f"Model copied as dist/{args.model_name}")
