# Drum Isolator

Real-time **drum stem isolation** from full mixes: a U-Net predicts a soft magnitude mask on stereo-to-mono spectrograms (MUSDB18 drums target). Training is in **PyTorch**; deployment uses **ONNX** for Python demos and a **C++** engine (ONNX Runtime + FFTW3 + PortAudio).

**Repository:** [github.com/AnubhavChoudhery/Drum_Isolator](https://github.com/AnubhavChoudhery/Drum_Isolator)

---

## Shared STFT contract

These values are fixed across `python/dataset.py`, `python/export.py`, `python/kaggle_train.py`, and `cpp/include/constants.h`. Do not change them unless you retrain and re-export end-to-end.

| Parameter | Value |
|-----------|------:|
| Sample rate | 44100 Hz |
| `n_fft` | 2048 |
| `hop_length` | 512 |
| `win_length` | 2048 |
| Frequency bins | 1025 (`n_fft // 2 + 1`) |
| ONNX / C++ chunk frames (`kFrameCount` / `FRAME_COUNT`) | 256 |

---

## Repository layout

```
python/
  dataset.py          # MUSDB loader, STFT, collate
  model.py            # DrumUNet
  train.py            # Local multi-GPU training (MUSDB on disk)
  kaggle_train.py     # Self-contained Kaggle notebook script (download + train + export)
  export.py           # Checkpoint → ONNX (opset 17, dynamic time axis)
  verify_models.py    # ONNX + optional PyTorch parity check
  demo_mir_yt.py      # Download or file input → isolated drums WAV
  ...
cpp/
  CMakeLists.txt, vcpkg.json
  include/            # constants, ring buffer, inference, audio reader
  src/                # mir_engine: PortAudio + ONNX + FFTW3
models/               # Place exports here locally (ignored by git)
```

---

## Python environment

Install [PyTorch](https://pytorch.org/) for your platform, then:

```bash
cd python
pip install -r requirements.txt
```

Training extras (`musdb`, `stempeg`) are required for `train.py` / `dataset.py`. For export verification, `onnx` and `onnxruntime` are listed in `requirements.txt`.

---

## Dataset (local training)

Use [MUSDB18](https://sigsep.github.io/datasets/musdb.html) (HQ stems recommended). Point `--musdb_root` at the folder that contains the dataset (the layout expected by `musdb` / `musdb.DB`).

---

## Local training (`train.py`)

Trains on magnitude spectrograms with **L1 / SmoothL1 / combined** loss (same family as Kaggle). Checkpoints are saved under `--ckpt_dir` (default `checkpoints/`).

```bash
python train.py --musdb_root /path/to/musdb18 --epochs 100 --batch_size 8
```

Useful flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs` | 100 | Epoch count |
| `--batch_size` | 8 | Per-step batch (DataParallel if multi-GPU) |
| `--lr` | 1e-4 | Adam learning rate |
| `--max_time_frames` | 1024 | Cap STFT time length per clip (matches `dataset.DEFAULT_MAX_TIME_FRAMES`; use `-1` only if you have enough VRAM) |
| `--loss` | `combined` | `l1`, `smooth_l1`, or `combined` (0.7×L1 + 0.3×SmoothL1) |
| `--ckpt_dir` | `checkpoints` | Checkpoint directory |
| `--resume` | — | Path to `.pt` to continue |
| `--require_gpu` | off | Exit if CUDA unavailable |

---

## Kaggle training (`kaggle_train.py`)

Single-file pipeline: installs **non-torch** deps only (do **not** `pip install --upgrade torch` on Kaggle — it replaces the CUDA build). Downloads MUSDB18-HQ from Zenodo, optionally builds a spectrogram cache, trains with AMP and validation metrics (magnitude SNR dB + cosine similarity), exports ONNX, and zips outputs.

Edit the **`CFG`** class at the top of the file (epochs, batch size, paths, `REQUIRE_GPU`, `LOSS`, etc.). Defaults are tuned for a long GPU session:

| Setting | `kaggle_train` `CFG` | `train.py` (local) |
|--------|----------------------|---------------------|
| Epochs | `EPOCHS = 150` | `--epochs` default **100** |
| Batch | `BATCH_SIZE = 4` | `--batch_size` default **8** |
| Max STFT frames | `MAX_TIME_FRAMES = 1024` | `--max_time_frames` default **1024** (from `dataset.py`) |
| LR | `1e-4` | `1e-4` |
| Loss | `combined` (configurable) | `combined` default |

Run as a script or paste into a notebook with GPU + Internet enabled. Outputs go under `/kaggle/working/` (e.g. `best_drums_unet.pt`, `drums_unet.onnx`, zip).

---

## Export to ONNX

From a checkpoint produced by `train.py` or Kaggle:

```bash
python export.py --checkpoint path/to/checkpoint.pt --output ../models/drums_unet.onnx --verify
```

Exports embed weights in a **single** `.onnx` file when possible. If you have an older two-file export (`.onnx` + `.onnx.data`), use `python/embed_onnx_sidecar.py` or re-export.

---

## Verify models

With `models/drums_unet.onnx` (and optionally `best_drums_unet.pt`) in `models/`:

```bash
python verify_models.py
```

---

## Python demo (isolated drums WAV)

Requires `soundfile` and `yt-dlp` (see `requirements.txt`). Example:

```bash
python demo_mir_yt.py --synthetic
python demo_mir_yt.py --audio-file path/to/mix.wav --max-seconds 90
python demo_mir_yt.py "https://example.com/track.mp3"   # yt-dlp supported URL
```

Outputs a `drums_isolated.wav` (or `--out-wav`). The pipeline pads STFT time to a multiple of 16 for U-Net alignment, then ISTFT-reconstructs the drum stem.

---

## C++ real-time engine (`mir_engine`)

Dependencies via **vcpkg** (manifest in `cpp/vcpkg.json`): PortAudio, libsndfile (WAV-focused default features), FFTW3. **ONNX Runtime** is fetched by CMake (`FetchContent`). On Windows, use the **x64 Native Tools** / `vcvars64.bat` environment with **Ninja** or Visual Studio, and pass the vcpkg toolchain file:

```bat
cmake -S cpp -B cpp/build -G Ninja -DCMAKE_TOOLCHAIN_FILE=vcpkg\scripts\buildsystems\vcpkg.cmake -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build
```

Copy `models/drums_unet.onnx` next to `mir_engine.exe` (or pass paths). Run:

```bat
mir_engine.exe input.wav drums_unet.onnx
```

---

## What not to commit

Checkpoints, ONNX, audio, zip exports, `downloads/`, local `vcpkg/`, and `cpp/build/` are **gitignored**. Keep large artifacts out of the repo; use releases or external storage if you need to share weights.

---

## License

Add a `LICENSE` file if you want a specific open-source terms; the MUSDB dataset and third-party libraries have their own licenses.
