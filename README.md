# TWIMS - Translate What I'm Saying

TWIMS is a real-time speech-to-text transcription tool that supports multiple engines:
- `whisper.cpp` (for macOS - Apple Silicon with Metal/MPS)
- `openai/whisper` with PyTorch (for Windows/Linux - CPU/CUDA)

It uses VAD (Voice Activity Detection) for efficient segmentation and allows you to choose your microphone and engine at runtime (for development) or compile builds per platform.

---

## 📦 Features
- Real-time microphone transcription
- Multilingual support with optional translation to English
- Automatically detects the appropriate backend in development
- Threaded architecture for smooth capture/transcribe flow
- Lightweight production builds with PyInstaller

---

## 🚀 Quick Start

### 1. Install dependencies

#### Install dependencies
```bash
pip install -r requirements.txt
```


#### For `torch` engine (CUDA/CPU):
```bash
pip install -r requirements-torch-cuda.txt
```
> Edit the file to match your desired CUDA version (cu121, cu118, etc).

### 2. Run in development

Auto-detects engine or allow override:
```bash
python main.py                   # auto-selects engine based on OS
python main.py --engine=torch    # manual override
python main.py --engine=cpp
```

---

## 🏗️ Building Executables

Use `build.py` to generate platform-specific builds:

### Example: Build for CUDA (Windows/Linux)
```bash
python build.py --engine=torch --output=twims_cuda
```

### Example: Build for macOS (with Whisper.cpp)
```bash
python build.py --engine=cpp --download-model \
  --model-url="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-q5_1.bin" \
  --model-name=ggml.bin --output=twims_mac
```

#### Optional flags:
- `--console`: shows terminal window on run
- `--clean`: cleans `build/` and `dist/` before compiling

---

## 🗂️ Folder Structure
```
twims/
├── main.py
├── engine_selector.py      # used to select and inject engine
├── engines/
│   ├── whisper_cpp.py
│   ├── whisper_torch.py
├── build.py
├── .env                    # optional (not required)
```

---

## ⚙️ Configuration

### Environment Variables (Optional)
Create a `.env` file or export environment variables:

```
TWIMS_MODEL_SIZE=base        # used in torch engine (tiny, base, small, ...)
TWIMS_MODEL_PATH=ggml.bin    # used in cpp engine
```

In development, you can override engine with:
```bash
python main.py --engine=torch
```

In production, `engine.py` is injected via `build.py` and should not be dynamic.

---

## 🧠 Known Limitations
- Only one engine is bundled per build for performance
- Whisper.cpp only works efficiently on Apple Silicon
- Audio format is mono, 16-bit PCM at 16kHz
- Short words or clipped speech might be missed by the model or VAD

---

## 📜 License
This project is open source for non-commercial use.  
You may:
- View, use and modify the code for personal or academic purposes

You may not:
- Sell, redistribute or use this software for commercial purposes without written permission

Parts of this project use:
- [pywhispercpp (MIT License)](https://github.com/aarnphm/pywhispercpp)
- [openai/whisper (MIT License)](https://github.com/openai/whisper)

---

## 🧪 Coming Soon
- GUI wrapper

---

## 👤 Author
Julian Gonzalez

Feel free to fork and contribute! PRs are welcome.


## License

TWIMS (Translate What I’m Saying) is released under the **TWIMS License (v1.0)**.  
You are free to use, modify, and distribute this software **for non-commercial purposes**.

> 🚫 Commercial use is **not allowed** without explicit written permission from the author.

To request a commercial license, please contact: **julianglz@outlook.es**

See the full [LICENSE](LICENSE) file for details.

## Third-party software

TWIMS uses the following open source components:

- [pywhispercpp](https://github.com/abdeladim-s/pywhispercpp) by Abdeladim Sadiki – licensed under the MIT License.

See `LICENSE-THIRD-PARTY.txt` for full details.

## 📦 Release Assets & Update Strategy

TWIMS provides two types of downloadable assets per release to support **easy updates** and **modular builds**.

### ✅ Release Asset Structure

| File                                      | Description                                                                 |
|-------------------------------------------|-----------------------------------------------------------------------------|
| `twims_windows_vX.Y.Z.exe`                | Standalone executable for Windows (can be updated independently)           |
| `twims_windows_vX.Y.Z.zip`                | Full build (`dist/`) with dependencies and assets for fresh install        |
| `ggml.bin`                                | Whisper.cpp model (if using `--engine=cpp`)                                |
| `README.txt`                              | Usage instructions included in the `.zip`                                  |

### 🧩 Update Options

- **🆕 Quick Update**  
  Download the latest `twims.exe` and replace it in your existing folder.  
  This is great for faster updates without re-downloading all dependencies.

- **📦 Fresh Install**  
  Download the `.zip`, extract all contents to a folder, and run `twims.exe`.

- **🔁 Replace Whisper Model (CPP engine only)**  
  1. Get a model from [ggerganov/whisper.cpp](https://huggingface.co/ggerganov/whisper.cpp)  
  2. Rename it to `ggml.bin`  
  3. Place it next to the executable

### 🛠 Why This Structure?

This modular release layout is designed to be:

- 🪶 Lightweight for updates  
- 🔁 Flexible to swap engines/models  
- 🔒 Safe for future OBS integration or user themes