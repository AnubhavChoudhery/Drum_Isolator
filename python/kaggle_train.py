"""
kaggle_train.py — DrumUNet Training Script for Kaggle
======================================================

Full self-contained training pipeline for DrumUNet drum source separation.
Downloads MUSDB18-HQ via the official Zenodo URL, trains the U-Net, saves
checkpoints, and exports to ONNX — all inside the Kaggle kernel environment.

Expected GPU:  Tesla T4 (16 GB) or P100 (16 GB)
Estimated run time (depends on ``CFG.EPOCHS``; default is **150** — see ``CFG`` below):
    - ~30 epochs  ~  a few hours  (smoke test — lower ``CFG.EPOCHS`` before running)
    - ~150 epochs ~  most of a 12 h Kaggle budget (full run with default ``CFG``)

How to use on Kaggle:
    1. Create a new Notebook (code type, not markdown).
    2. Upload this file as a Data Source OR paste the contents into a code cell.
    3. Set Accelerator to GPU (T4 x1 or P100).
    4. Turn on "Internet access" (required for dataset download).
    5. Set --epochs via the CONFIG block below.
    6. Hit "Save & Run All".
    7. After training, outputs land in /kaggle/working/:
       mir_drums_export.zip (best .pt + ONNX + readme), best_drums_unet.pt, drums_unet.onnx.
       Download from the Output tab, or use the clickable FileLink cells printed at the end
       if the tab fails to load large files.  Alternatively: `kaggle kernels output ... -p .`

NOTE: MUSDB18-HQ is hosted on Zenodo (~23 GB).  The download takes ~10–15 min
on Kaggle's network.  Total kernel wall-time budget is 12 hours (T4) / 9 hours
(P100) — plan your epoch count accordingly.
"""

# ===========================================================================
# 0.  Install dependencies (do NOT upgrade torch — breaks Kaggle CUDA builds)
# ===========================================================================
import subprocess, sys

def _pip_install(*packages):
    """
    Install extra wheels without ``--upgrade``.

    Using ``pip install --upgrade`` on ``onnx`` / ``onnxscript`` can pull a new
    ``torch`` CPU wheel and **replace** the image's CUDA PyTorch — then
    ``torch.device`` disappears and you get AttributeError inside cuda APIs.
    """
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", *packages],
    )


_pip_install(
    # --- Dataset / I/O ---
    "musdb>=0.4.0",
    "stempeg",
    "soundfile",
    # --- ONNX stack (torch.onnx.export may require onnxscript on newer PyTorch) ---
    "onnx",
    "onnxscript",
    "onnxruntime",
)
# PyTorch / torchaudio / numpy: use the kernel's preinstalled CUDA build — never pip upgrade torch.
#
# Manual one-liner (notebook cell), same packages, still without upgrading torch:
#   %pip install -q musdb stempeg soundfile onnx onnxscript onnxruntime

# ===========================================================================
# 1.  Standard imports
# ===========================================================================
import argparse
import os
import random
import shutil
import time
import zipfile
import urllib.request

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

# Catch a broken / shadowed PyTorch before training starts.
if not hasattr(torch, "device"):
    raise RuntimeError(
        "PyTorch is not loading correctly (missing torch.device).\n"
        "  • Do not run: pip install --upgrade torch  (breaks Kaggle GPU images)\n"
        "  • Rename any local file named torch.py that shadows the real package\n"
        "  • Session → Restart session, then run again without upgrading torch."
    )
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader

import musdb

# ===========================================================================
# 2.  CONFIG — tweak these before running
# ===========================================================================
class CFG:
    # --- Dataset ---
    # MUSDB18-HQ Zenodo URL (official, no login required).
    # If Kaggle blocks Zenodo, use the mirror below.
    MUSDB_URL      = "https://zenodo.org/record/3338373/files/musdb18hq.zip"
    MUSDB_ZIP      = "/tmp/musdb18hq.zip"
    MUSDB_ROOT     = "/tmp/musdb18hq"

    # --- Training ---
    EPOCHS          = 150         # safe budget for 12-hr Kaggle sessions (see timing notes)
    BATCH_SIZE      = 4           # safe with AMP on T4/P100 at 1024 frames
    LR              = 1e-4
    # STFT frames per training crop — keep at 1024 for quality; lower to 512 if OOM
    MAX_TIME_FRAMES = 1024
    NUM_WORKERS     = 4           # Kaggle GPU sessions have 4 vCPUs
    SAVE_EVERY      = 25
    CKPT_DIR        = "/kaggle/working/checkpoints"
    RESUME          = None        # set to a checkpoint path to resume

    # --- Spectrogram cache ---
    # Precomputing STFTs once to disk turns 16s/__getitem__ → ~0.05s/__getitem__.
    # Expected epoch time after caching: ~20-40s (vs 793s before).
    # Budget: ~30 min download + ~3 min build cache + 150 × ~30s = ~1.5 hr total.
    # CACHE_DIR must have ≥ ~1.7 GB free (float16, 4096 frames × 100 tracks).
    CACHE_DIR       = "/tmp/spec_cache"
    # Frames cached per track — must be > MAX_TIME_FRAMES so random crops get variety.
    CACHE_FRAMES    = 4096

    # --- Export (paths under /kaggle/working appear in the Output tab) ---
    WORKING_DIR    = "/kaggle/working"
    ONNX_OUT       = "/kaggle/working/drums_unet.onnx"
    # Stable filenames — easier to find than timestamped checkpoints/
    BEST_PT_ALIAS  = "/kaggle/working/best_drums_unet.pt"
    # Single zip: if the Output UI struggles with large files, download this one file.
    EXPORT_ZIP     = "/kaggle/working/mir_drums_export.zip"

    # --- Device ---
    # If True, training aborts with instructions when CUDA is unavailable (otherwise
    # you silently train on CPU — very slow).  On Kaggle you must set the notebook
    # Accelerator to GPU: Settings (gear) → Accelerator → GPU T4 / P100 → Save →
    # **Restart session** so `torch.cuda.is_available()` becomes True.
    REQUIRE_GPU      = True

    # --- Loss (magnitude-domain targets) ---
    # "l1"        — L1/MAE on masked mixture vs drum magnitude (standard, robust).
    # "smooth_l1" — Huber; slightly less sensitive to outliers than pure L1.
    # "combined"  — 0.7 * L1 + 0.3 * SmoothL1 (common tweak for audio).
    # For SOTA you’d add waveform metrics (SI-SDR) via ISTFT in the loop — heavier.
    LOSS             = "combined"

    # Mixed precision — faster convs on GPU, often higher reported GPU utilization.
    USE_AMP          = True

    # DataLoader: prefetch batches while GPU trains (reduces idle time).  STFT in
    # __getitem__ is still CPU-heavy — if GPU util stays low, that is expected unless
    # you precompute spectrograms.  Windows “GPU 0%” often measures the wrong engine;
    # use `nvidia-smi` or Kaggle’s GPU chart (CUDA compute), not Task Manager 3D.
    DATALOADER_PREFETCH = 4

# ===========================================================================
# 3.  STFT constants  (must match cpp/include/constants.h exactly)
# ===========================================================================
SAMPLE_RATE  = 44100
N_FFT        = 2048
HOP_LENGTH   = 512
WIN_LENGTH   = 2048
FREQ_BINS    = N_FFT // 2 + 1   # 1025
FRAME_COUNT  = 256               # kFrameCount in C++

# ===========================================================================
# 4.  Dataset download
# ===========================================================================

def download_musdb(url: str, zip_path: str, extract_to: str) -> None:
    """Download and extract MUSDB18-HQ with disk cleanup."""
    if os.path.isdir(extract_to) and os.listdir(extract_to):
        print(f"[dataset] Found existing MUSDB18 at {extract_to} — skipping download.")
        return

    if not os.path.isfile(zip_path):
        print(f"[dataset] Downloading MUSDB18-HQ to /tmp (~23 GB)…")
        print("          This will take 10–15 minutes on Kaggle's network.")
        start = time.time()
        urllib.request.urlretrieve(url, zip_path, reporthook=_download_progress)
        elapsed = time.time() - start
        print(f"\n[dataset] Download complete in {elapsed/60:.1f} min.")
    else:
        print(f"[dataset] ZIP already present at {zip_path} — skipping download.")

    print(f"[dataset] Extracting to {extract_to}…")
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_to)

    # Delete the zip immediately to free up space
    print("[dataset] Extraction complete. Deleting ZIP to free space...")
    os.remove(zip_path)
    print(f"[dataset] Disk clean. Dataset ready at {extract_to}")


_last_pct = -1
def _download_progress(block_num, block_size, total_size):
    global _last_pct
    if total_size <= 0:
        return
    pct = int(block_num * block_size * 100 / total_size)
    pct = min(pct, 100)
    if pct != _last_pct and pct % 5 == 0:
        print(f"  … {pct}%", flush=True)
        _last_pct = pct

# ===========================================================================
# 5.  STFT helpers
# ===========================================================================

def _build_hann_window(device: torch.device) -> torch.Tensor:
    return torch.hann_window(WIN_LENGTH, device=device)


def _to_mono(waveform: torch.Tensor) -> torch.Tensor:
    if waveform.ndim == 1:
        return waveform
    return waveform.mean(dim=0)


def compute_stft(waveform: torch.Tensor, window: torch.Tensor):
    stft_complex = torch.stft(
        waveform,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=window,
        center=True,
        pad_mode="reflect",
        normalized=False,
        return_complex=True,
    )
    return stft_complex.abs(), stft_complex.angle()


def crop_time_axis_pair(mix, drums, max_frames: int, *, random_crop: bool):
    """``mix`` / ``drums``: [1, FREQ_BINS, T].  Identical crop on both."""
    t = int(mix.shape[-1])
    if t <= max_frames:
        return mix, drums
    start = random.randint(0, t - max_frames) if random_crop else (t - max_frames) // 2
    return mix[..., start : start + max_frames], drums[..., start : start + max_frames]


def _stem_path(track, name: str) -> str:
    """
    Return the path to a stem WAV in a MUSDB18-HQ is_wav=True directory tree.
    Structure: {root}/{subset}/{track_name}/{stem_name}.wav

    musdb sets track.path to the *mixture.wav file path* (not its parent dir),
    so we call dirname() when the path points at a file rather than a directory.
    """
    base = track.path
    if os.path.isfile(base):           # track.path → …/mixture.wav → go up one level
        base = os.path.dirname(base)
    for fname in (f"{name}.wav", f"{name}.WAV"):
        p = os.path.join(base, fname)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Cannot find '{name}.wav' under {base}")


def _read_partial(path: str, n_samples: int, start: int = 0) -> torch.Tensor:
    """
    Read exactly n_samples from a WAV at byte offset start via soundfile.
    Returns [channels, n_samples] float32 tensor.
    soundfile supports partial reads without decoding the full file.
    """
    data, _ = sf.read(path, frames=n_samples, start=start, dtype="float32", always_2d=True)
    return torch.from_numpy(np.ascontiguousarray(data.T))  # [C, frames]


# ===========================================================================
# 6a.  Spectrogram cache  (precomputed once → fast __getitem__ on every epoch)
# ===========================================================================

def precompute_spectrogram_cache(
    root: str,
    subset: str,
    cache_dir: str,
    cache_frames: int,
) -> None:
    """
    Build STFT cache to disk before training.  Called once; skips files that
    already exist so resuming a kernel is safe.

    Each .npz stores float16 magnitude spectrograms of ``cache_frames`` STFT
    frames per track (~16 MB / track at 4096 frames).

    Why this matters:
        Without cache:  track.audio loads full ~13 M samples → STFT → crop = ~16s/item
        With cache:     np.load() + array slice = ~0.05s/item → 300x faster
    """
    os.makedirs(cache_dir, exist_ok=True)
    db = musdb.DB(root=root, subsets=subset, is_wav=True)
    window = _build_hann_window(torch.device("cpu"))
    tracks = list(db.tracks)
    n_built = 0

    print(f"\n[cache] Building spectrogram cache for {len(tracks)} '{subset}' tracks "
          f"({cache_frames} frames each) → {cache_dir}")
    t_start = time.time()

    for i, track in enumerate(tracks):
        cache_file = os.path.join(cache_dir, f"{track.name}.npz")
        if os.path.exists(cache_file):
            continue

        samples_needed = cache_frames * HOP_LENGTH
        mix_path    = _stem_path(track, "mixture")
        drums_path  = _stem_path(track, "drums")

        # soundfile partial read — avoids decoding the full 3-5 min file
        info = sf.info(mix_path)
        n = min(samples_needed, info.frames)

        mix_t   = _read_partial(mix_path,   n)
        drums_t = _read_partial(drums_path, n)

        mix_mag,   _ = compute_stft(_to_mono(mix_t),   window)
        drums_mag, _ = compute_stft(_to_mono(drums_t), window)

        np.savez_compressed(
            cache_file,
            mix   = mix_mag.numpy().astype(np.float16),
            drums = drums_mag.numpy().astype(np.float16),
        )
        n_built += 1

    elapsed = time.time() - t_start
    print(f"[cache] Done — {n_built} new files in {elapsed:.1f}s "
          f"(skipped {len(tracks) - n_built} already cached).\n")


# ===========================================================================
# 6b.  Dataset
# ===========================================================================

class MusdbDrumDataset(Dataset):
    """
    Fast MUSDB18 dataset.

    Fast path (cache_dir set):
        np.load(pre-computed .npz) → random-crop → done.  ~0.05s/item.
    Slow fallback (no cache):
        soundfile partial read → STFT on crop-only samples.  ~0.5s/item.
        Still 20-30x faster than the naive full-track STFT approach.
    """

    def __init__(
        self,
        root: str,
        subset: str = "train",
        max_time_frames: int = 1024,
        cache_dir: str | None = None,
    ):
        self.db     = musdb.DB(root=root, subsets=subset, is_wav=True)
        self.tracks = list(self.db.tracks)
        self.window = _build_hann_window(torch.device("cpu"))
        self.subset = subset
        self.max_time_frames = max_time_frames
        self.cache_dir = cache_dir
        mode = f"cache={cache_dir}" if cache_dir else "on-the-fly STFT"
        print(f"[dataset] {len(self.tracks)} '{subset}' tracks  |  {mode}")

    def __len__(self) -> int:
        return len(self.tracks)

    def __getitem__(self, idx: int):
        track = self.tracks[idx]
        is_train = self.subset == "train"

        # ------------------------------------------------------------------
        # Fast path: load from precomputed cache
        # ------------------------------------------------------------------
        if self.cache_dir:
            cache_file = os.path.join(self.cache_dir, f"{track.name}.npz")
            data = np.load(cache_file)
            mix_mag   = torch.from_numpy(data["mix"].astype("float32"))    # [F, T_cache]
            drums_mag = torch.from_numpy(data["drums"].astype("float32"))

        # ------------------------------------------------------------------
        # Slow fallback: soundfile partial read + on-the-fly STFT
        # Root cause of 793s/epoch was full-track STFT here.  Partial read
        # cuts samples from ~13M → ~500K so STFT is 25-30x cheaper.
        # ------------------------------------------------------------------
        else:
            samples_needed = self.max_time_frames * HOP_LENGTH

            mix_path   = _stem_path(track, "mixture")
            drums_path = _stem_path(track, "drums")
            info = sf.info(mix_path)

            if info.frames > samples_needed:
                start = (
                    random.randint(0, info.frames - samples_needed)
                    if is_train
                    else (info.frames - samples_needed) // 2
                )
            else:
                start = 0
                samples_needed = info.frames

            mix_t   = _read_partial(mix_path,   samples_needed, start)
            drums_t = _read_partial(drums_path, samples_needed, start)
            mix_mag,   _ = compute_stft(_to_mono(mix_t),   self.window)
            drums_mag, _ = compute_stft(_to_mono(drums_t), self.window)

        mix_mag   = mix_mag.unsqueeze(0)    # [1, F, T]
        drums_mag = drums_mag.unsqueeze(0)

        return crop_time_axis_pair(
            mix_mag, drums_mag, self.max_time_frames, random_crop=is_train
        )


def build_dataloader(
    root: str,
    subset: str,
    batch_size: int,
    num_workers: int,
    max_time_frames: int,
    *,
    pin_memory: bool,
    prefetch_factor: int,
    cache_dir: str | None,
) -> DataLoader:
    dataset = MusdbDrumDataset(
        root=root,
        subset=subset,
        max_time_frames=max_time_frames,
        cache_dir=cache_dir,
    )
    kw = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=(subset == "train"),
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    # With cached items all output the same size (max_time_frames), the default
    # collate just stacks — no custom padding needed.
    if num_workers > 0:
        kw["persistent_workers"] = True
        kw["prefetch_factor"] = prefetch_factor
    return DataLoader(**kw)


def warmup_cuda(model: nn.Module, device: torch.device) -> None:
    """One forward on random data so CUDA/cuDNN kernels are compiled; confirms GPU path."""
    if device.type != "cuda":
        return
    model.eval()
    tw = min(256, CFG.MAX_TIME_FRAMES)
    with torch.no_grad():
        dummy = torch.randn(1, 1, FREQ_BINS, tw, device=device, dtype=torch.float32)
        _ = model(dummy)
    torch.cuda.synchronize()
    model.train()
    print(f"  GPU warmup forward OK (dummy {tw} frames).")

# ===========================================================================
# 7.  Model  (self-contained copy of python/model.py)
# ===========================================================================

class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ks=3, pad=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, ks, padding=pad, bias=False), nn.BatchNorm2d(out_ch), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, ks, padding=pad, bias=False), nn.BatchNorm2d(out_ch), nn.LeakyReLU(0.2, inplace=True),
        )
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        skip = self.conv(x)
        return self.pool(skip), skip


class Bottleneck(nn.Module):
    def __init__(self, in_ch, out_ch, ks=3, pad=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, ks, padding=pad, bias=False), nn.BatchNorm2d(out_ch), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, ks, padding=pad, bias=False), nn.BatchNorm2d(out_ch), nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, ks=3, pad=1):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch // 2 + skip_ch, out_ch, ks, padding=pad, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch,               out_ch, ks, padding=pad, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        _, _, hs, ws = skip.shape
        _, _, hx, wx = x.shape
        if hx > hs:
            dh = hx - hs
            x = x[:, :, dh // 2 : dh // 2 + hs, :]
        if wx > ws:
            dw = wx - ws
            x = x[:, :, :, dw // 2 : dw // 2 + ws]
        _, _, hx, wx = x.shape
        dh = hs - hx
        dw = ws - wx
        if dh > 0 or dw > 0:
            x = F.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
        return self.conv(torch.cat([x, skip], dim=1))


class DrumUNet(nn.Module):
    """U-Net spectrogram mask predictor.  Input/Output: [B, 1, FREQ_BINS, T]"""

    def __init__(self):
        super().__init__()
        self.enc1      = EncoderBlock(1,   16)
        self.enc2      = EncoderBlock(16,  32)
        self.enc3      = EncoderBlock(32,  64)
        self.enc4      = EncoderBlock(64,  128)
        self.bottleneck = Bottleneck(128, 256)
        self.dec4      = DecoderBlock(256, 128, 128)
        self.dec3      = DecoderBlock(128,  64,  64)
        self.dec2      = DecoderBlock( 64,  32,  32)
        self.dec1      = DecoderBlock( 32,  16,  16)
        self.head      = nn.Sequential(nn.Conv2d(16, 1, 1), nn.Sigmoid())

    def forward(self, x):
        x, s1 = self.enc1(x)
        x, s2 = self.enc2(x)
        x, s3 = self.enc3(x)
        x, s4 = self.enc4(x)
        x = self.bottleneck(x)
        x = self.dec4(x, s4)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)
        return self.head(x)

# ===========================================================================
# 8.  Checkpoint helpers
# ===========================================================================

def save_checkpoint(model, optimizer, epoch, loss, ckpt_dir, tag=""):
    os.makedirs(ckpt_dir, exist_ok=True)
    filename = f"drum_unet_epoch{epoch:04d}{tag}.pt"
    path = os.path.join(ckpt_dir, filename)
    module = model.module if isinstance(model, nn.DataParallel) else model
    torch.save({
        "epoch": epoch,
        "model_state_dict": module.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }, path)
    print(f"  [ckpt] saved → {path}")
    return path


def load_checkpoint(path, model, optimizer=None):
    ckpt = torch.load(path, map_location="cpu")
    module = model.module if isinstance(model, nn.DataParallel) else model
    module.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    print(f"  [ckpt] resumed from {path} (epoch {ckpt['epoch']}, loss {ckpt['loss']:.6f})")
    return ckpt["epoch"]

# ===========================================================================
# 9.  ONNX export
# ===========================================================================

def _embed_onnx_weights_inline(path: str) -> None:
    """Merge `path.data` into `path` so one file can be copied (PyTorch often emits a sidecar)."""
    try:
        import onnx
    except ImportError:
        return
    try:
        proto = onnx.load(path)
        onnx.save(proto, path, save_as_external_data=False)
        sidecar = path + ".data"
        if os.path.isfile(sidecar):
            os.remove(sidecar)
    except Exception as ex:
        print(f"[export] ONNX single-file merge skipped: {ex}")


def export_onnx(model, output_path):
    """Export the trained model to ONNX for the C++ inference engine."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    module = model.module if isinstance(model, nn.DataParallel) else model
    module.eval().cpu()

    dummy = torch.zeros(1, 1, FREQ_BINS, FRAME_COUNT, dtype=torch.float32)
    torch.onnx.export(
        module,
        dummy,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["mix_magnitude"],
        output_names=["drum_mask"],
        dynamic_axes={"mix_magnitude": {3: "time_frames"}, "drum_mask": {3: "time_frames"}},
        verbose=False,
    )
    _embed_onnx_weights_inline(output_path)
    print(f"[export] ONNX model written → {output_path}")
    print(f"         Input : mix_magnitude [1, 1, {FREQ_BINS}, T]")
    print(f"         Output: drum_mask     [1, 1, {FREQ_BINS}, T]  in [0,1]")

    # Quick verification
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
        out  = sess.run(["drum_mask"], {"mix_magnitude": dummy.numpy()})[0]
        print(f"[export] Verified — output shape {out.shape}, range [{out.min():.4f}, {out.max():.4f}]")
    except Exception as e:
        print(f"[export] Verification skipped ({e})")


# ===========================================================================
# 9a.  Kaggle download bundle (stable paths + zip + notebook links)
# ===========================================================================

def _human_bytes(num: int) -> str:
    n = float(num)
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} TB"


def package_kaggle_outputs(
    best_ckpt_path: str | None,
    final_ckpt_path: str,
    onnx_path: str,
    best_val_loss: float,
) -> None:
    """
    1) Copy best PyTorch checkpoint to a fixed filename in /kaggle/working.
    2) ONNX is already at ONNX_OUT (same folder).
    3) Zip both + a small readme — one file is easier to download than many.
    4) In Jupyter/Kaggle notebooks, show IPython FileLink widgets (click to download
       when the Output tab spinner never finishes on large files).
    """
    os.makedirs(CFG.WORKING_DIR, exist_ok=True)

    readme = os.path.join(CFG.WORKING_DIR, "export_readme.txt")
    src_best = best_ckpt_path if (best_ckpt_path and os.path.isfile(best_ckpt_path)) else final_ckpt_path
    with open(readme, "w", encoding="utf-8") as f:
        f.write("MIR Drum Isolation — Kaggle export\n")
        f.write(f"best_validation_loss = {best_val_loss:.6f}\n")
        f.write(f"pytorch_source = {src_best}\n")
        f.write(f"onnx = {onnx_path}\n\n")
        f.write("Files:\n")
        f.write("  best_drums_unet.pt — load in Python with DrumUNet.load_state_dict\n")
        f.write("  drums_unet.onnx    — C++ / ONNX Runtime inference\n")

    if os.path.isfile(src_best):
        shutil.copy2(src_best, CFG.BEST_PT_ALIAS)
        print(
            f"[export] PyTorch (best) → {CFG.BEST_PT_ALIAS} "
            f"({_human_bytes(os.path.getsize(CFG.BEST_PT_ALIAS))})"
        )
    else:
        print(f"[export] Warning: no checkpoint found at {src_best}")

    if os.path.isfile(onnx_path):
        print(f"[export] ONNX → {onnx_path} ({_human_bytes(os.path.getsize(onnx_path))})")

    with zipfile.ZipFile(CFG.EXPORT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        if os.path.isfile(CFG.BEST_PT_ALIAS):
            zf.write(CFG.BEST_PT_ALIAS, arcname="best_drums_unet.pt")
        if os.path.isfile(onnx_path):
            zf.write(onnx_path, arcname="drums_unet.onnx")
        zf.write(readme, arcname="export_readme.txt")

    print(f"[export] Zip bundle → {CFG.EXPORT_ZIP} ({_human_bytes(os.path.getsize(CFG.EXPORT_ZIP))})")

    _try_notebook_download_links(CFG.EXPORT_ZIP, CFG.BEST_PT_ALIAS, onnx_path)


def _try_notebook_download_links(*paths: str) -> None:
    """Render clickable download links in Kaggle/Jupyter (helps when Output UI hangs)."""
    try:
        ip = __import__("IPython").get_ipython()
        if ip is None:
            return
        from IPython.display import HTML, display, FileLink

        display(HTML("<b>Direct download links (use if Output tab does not load):</b>"))
        for p in paths:
            if os.path.isfile(p):
                display(FileLink(p))
    except Exception as ex:
        print(f"[export] Notebook download links unavailable ({ex})")


# ===========================================================================
# 9b.  Loss factory
# ===========================================================================

class CombinedSpectralLoss(nn.Module):
    """Weighted L1 + SmoothL1 on magnitude spectrograms (drum estimates)."""

    def __init__(self, l1_weight: float = 0.7, smooth_weight: float = 0.3, beta: float = 0.05):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.smooth = nn.SmoothL1Loss(beta=beta)
        self.l1_weight = l1_weight
        self.smooth_weight = smooth_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.l1_weight * self.l1(pred, target) + self.smooth_weight * self.smooth(pred, target)


def build_criterion(name: str) -> nn.Module:
    n = name.lower().strip()
    if n == "l1":
        return nn.L1Loss()
    if n in ("smooth_l1", "smoothl1", "huber"):
        return nn.SmoothL1Loss(beta=0.1)
    if n == "combined":
        return CombinedSpectralLoss()
    raise ValueError(f"Unknown CFG.LOSS={name!r}; use 'l1', 'smooth_l1', or 'combined'.")


def resolve_device():
    """
    Prefer CUDA.  If REQUIRE_GPU and CUDA missing, fail with Kaggle-specific help.
    """
    if torch.cuda.is_available():
        d = torch.device("cuda:0")
        torch.backends.cudnn.benchmark = True
        return d, torch.cuda.device_count()

    if CFG.REQUIRE_GPU:
        raise RuntimeError(
            "\n  *** GPU not visible to PyTorch (training would use CPU). ***\n"
            "  On Kaggle: open Notebook → right sidebar **Settings (gear)** → "
            "**Accelerator** → choose **GPU** (T4 / P100) → **Save** → "
            "**Session → Restart Session**, then **Run All**.\n"
            "  If you truly want CPU, set CFG.REQUIRE_GPU = False (not recommended).\n"
        )
    return torch.device("cpu"), 0


# ===========================================================================
# 9c.  Metrics (magnitude domain — not SI-SDR; for waveform SI-SDR add ISTFT)
# ===========================================================================

def magnitude_snr_db(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    """
    SNR in dB between predicted and target **magnitude** spectrograms.

    Higher is better.  This is **not** SI-SDR (that needs waveforms + scale invariance).
    Interpretation: ~0 dB → error power ≈ signal power; +10 dB → error ~10× smaller.
    """
    pred = pred.float()
    target = target.float()
    mse = torch.mean((pred - target) ** 2)
    sig = torch.mean(target ** 2)
    return (10.0 * torch.log10(sig / (mse + eps))).item()


def magnitude_cosine(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Mean cosine similarity of flattened spectrograms per batch item, in [-1, 1] (1 = identical).
    """
    pred = pred.float().reshape(pred.size(0), -1)
    target = target.float().reshape(target.size(0), -1)
    return F.cosine_similarity(pred, target, dim=1).mean().item()


# ===========================================================================
# 10.  Training loop
# ===========================================================================

def train():
    # ---- Device ----
    device, n_gpus = resolve_device()
    print(f"\n{'='*60}")
    print(f"  DrumUNet Training — Kaggle Kernel")
    if device.type == "cuda":
        # Integer index avoids edge cases with torch.device on some broken installs
        _gpu_name = torch.cuda.get_device_name(0)
        print(f"  Device: {device}  ({_gpu_name})")
        print(f"  CUDA:   {torch.version.cuda}  |  cuDNN benchmark: on")
    else:
        print(f"  Device: {device}  (no GPU — training will be slow)")
    print(f"  GPUs visible: {n_gpus}")
    print(f"  Epochs: {CFG.EPOCHS}  |  Batch: {CFG.BATCH_SIZE}  |  LR: {CFG.LR}")
    print(f"  Max STFT frames / example: {CFG.MAX_TIME_FRAMES}  (reduces GPU memory)")
    print(f"  Loss: {CFG.LOSS}")
    print(f"{'='*60}\n")

    # ---- Dataset ----
    download_musdb(CFG.MUSDB_URL, CFG.MUSDB_ZIP, CFG.MUSDB_ROOT)

    # Build spectrogram cache once (skips any tracks already cached).
    # Cost: ~2-3 min for 100+50 tracks.  Speedup: epoch time ~800s → ~25s.
    if CFG.CACHE_DIR:
        precompute_spectrogram_cache(CFG.MUSDB_ROOT, "train", CFG.CACHE_DIR, CFG.CACHE_FRAMES)
        precompute_spectrogram_cache(CFG.MUSDB_ROOT, "test",  CFG.CACHE_DIR, CFG.CACHE_FRAMES)

    pin_mem = device.type == "cuda"
    train_loader = build_dataloader(
        CFG.MUSDB_ROOT,
        "train",
        CFG.BATCH_SIZE,
        CFG.NUM_WORKERS,
        CFG.MAX_TIME_FRAMES,
        pin_memory=pin_mem,
        prefetch_factor=CFG.DATALOADER_PREFETCH,
        cache_dir=CFG.CACHE_DIR,
    )
    val_loader = build_dataloader(
        CFG.MUSDB_ROOT,
        "test",
        CFG.BATCH_SIZE,
        CFG.NUM_WORKERS,
        CFG.MAX_TIME_FRAMES,
        pin_memory=pin_mem,
        prefetch_factor=CFG.DATALOADER_PREFETCH,
        cache_dir=CFG.CACHE_DIR,
    )

    # ---- Model ----
    model = DrumUNet()
    if n_gpus > 1:
        print(f"  Wrapping model in DataParallel across {n_gpus} GPUs.")
        model = nn.DataParallel(model)
    model = model.to(device)

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    warmup_cuda(model, device)

    use_amp = device.type == "cuda" and CFG.USE_AMP
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    if use_amp:
        print("  Mixed precision (AMP) enabled — faster GPU training.")

    # ---- Optimiser & loss ----
    optimizer = Adam(model.parameters(), lr=CFG.LR)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    criterion = build_criterion(CFG.LOSS)

    # ---- Resume ----
    start_epoch = 1
    if CFG.RESUME and os.path.isfile(CFG.RESUME):
        start_epoch = load_checkpoint(CFG.RESUME, model, optimizer) + 1

    # ---- Training log ----
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_snr_db": [],
        "train_cosine": [],
        "val_snr_db": [],
        "val_cosine": [],
    }
    best_val_loss = float("inf")
    best_ckpt     = None

    # ---- Epoch loop ----
    for epoch in range(start_epoch, CFG.EPOCHS + 1):
        # -- Train --
        model.train()
        epoch_loss = 0.0
        train_snr_sum = 0.0
        train_cos_sum = 0.0
        t0 = time.time()
        t_batch_wait = time.perf_counter()

        for step, (mix_mag, drums_mag) in enumerate(train_loader, 1):
            if step == 1 and epoch == start_epoch:
                print(
                    f"  [diag] Time to first batch: {time.perf_counter() - t_batch_wait:.1f}s "
                    f"(CPU loads audio + STFT in workers; GPU waits — low util until batches stream.)"
                )

            mix_mag   = mix_mag.to(device,   non_blocking=pin_mem)
            drums_mag = drums_mag.to(device, non_blocking=pin_mem)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                soft_mask       = model(mix_mag)
                predicted_drums = soft_mask * mix_mag
                loss = criterion(predicted_drums, drums_mag)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            # Magnitude-domain metrics (float32, outside autocast)
            with torch.cuda.amp.autocast(enabled=False):
                pf = predicted_drums.detach().float()
                tf = drums_mag.detach().float()
                train_snr_sum += magnitude_snr_db(pf, tf)
                train_cos_sum += magnitude_cosine(pf, tf)

            if step % 20 == 0:
                avg = epoch_loss / step
                elapsed = time.time() - t0
                print(f"  Epoch [{epoch}/{CFG.EPOCHS}] Step [{step}/{len(train_loader)}] "
                      f"Loss: {avg:.6f}  Elapsed: {elapsed:.1f}s")

        n_train = len(train_loader)
        avg_train_loss = epoch_loss / n_train
        avg_train_snr = train_snr_sum / n_train
        avg_train_cos = train_cos_sum / n_train

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # -- Validate --
        model.eval()
        val_loss = 0.0
        val_snr_sum = 0.0
        val_cos_sum = 0.0
        with torch.no_grad():
            for mix_mag, drums_mag in val_loader:
                mix_mag   = mix_mag.to(device,   non_blocking=pin_mem)
                drums_mag = drums_mag.to(device, non_blocking=pin_mem)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    soft_mask       = model(mix_mag)
                    predicted_drums = soft_mask * mix_mag
                    val_loss += criterion(predicted_drums, drums_mag).item()
                with torch.cuda.amp.autocast(enabled=False):
                    pf = predicted_drums.float()
                    tf = drums_mag.float()
                    val_snr_sum += magnitude_snr_db(pf, tf)
                    val_cos_sum += magnitude_cosine(pf, tf)
        n_val = len(val_loader)
        if n_val == 0:
            raise RuntimeError("Validation loader is empty — check MUSDB subset='test' and batch_size.")
        avg_val_loss = val_loss / n_val
        avg_val_snr = val_snr_sum / n_val
        avg_val_cos = val_cos_sum / n_val

        scheduler.step(avg_val_loss)

        print(f"\nEpoch {epoch}/{CFG.EPOCHS} — "
              f"Train Loss: {avg_train_loss:.6f}  Val Loss: {avg_val_loss:.6f}  "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}  Time: {time.time()-t0:.1f}s")
        print(f"           Mag SNR (dB): train {avg_train_snr:+.2f}  val {avg_val_snr:+.2f}  "
              f"|  Cosine: train {avg_train_cos:.4f}  val {avg_val_cos:.4f}")
        print(
            "           (Mag SNR / cosine are on magnitude spectrograms — not SI-SDR; "
            "higher SNR & cosine → better.)\n"
        )

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_snr_db"].append(avg_train_snr)
        history["train_cosine"].append(avg_train_cos)
        history["val_snr_db"].append(avg_val_snr)
        history["val_cosine"].append(avg_val_cos)

        # -- Checkpoint (periodic) --
        if epoch % CFG.SAVE_EVERY == 0:
            save_checkpoint(model, optimizer, epoch, avg_train_loss, CFG.CKPT_DIR)

        # -- Best model tracking --
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_ckpt = save_checkpoint(model, optimizer, epoch, avg_val_loss,
                                        CFG.CKPT_DIR, tag="_best")
            print(f"  *** New best val loss: {best_val_loss:.6f} — checkpoint saved ***")

    # ---- Final checkpoint & ONNX export ----
    final_ckpt = save_checkpoint(model, optimizer, CFG.EPOCHS, avg_train_loss,
                                  CFG.CKPT_DIR, tag="_final")
    print("\nTraining complete.")
    print(f"Best validation loss: {best_val_loss:.6f}")
    if history["val_snr_db"]:
        bi = int(np.argmin(history["val_loss"]))
        print(
            f"Best val mag-SNR (dB): {history['val_snr_db'][bi]:+.2f}  "
            f"(epoch {bi + 1})  |  val cosine at that epoch: {history['val_cosine'][bi]:.4f}"
        )
    print(f"Best checkpoint:      {best_ckpt}")

    # Export ONNX from the best checkpoint (fallback to final if no improvement was saved)
    ckpt_for_export = best_ckpt if best_ckpt is not None else final_ckpt
    print("\n[export] Loading checkpoint for ONNX export…")
    best_model = DrumUNet()
    load_checkpoint(ckpt_for_export, best_model)
    export_onnx(best_model, CFG.ONNX_OUT)

    package_kaggle_outputs(
        best_ckpt_path=best_ckpt,
        final_ckpt_path=final_ckpt,
        onnx_path=CFG.ONNX_OUT,
        best_val_loss=best_val_loss,
    )

    # Print loss / metric history summary
    print("\n[summary] Loss & magnitude metrics (last column = val cosine, 1=best):")
    for i, row in enumerate(
        zip(
            history["train_loss"],
            history["val_loss"],
            history["val_snr_db"],
            history["val_cosine"],
        ),
        1,
    ):
        tl, vl, vsnr, vcos = row
        print(f"  Epoch {i:3d}: train_loss={tl:.6f}  val_loss={vl:.6f}  "
              f"val_mag_SNR={vsnr:+.2f}dB  val_cos={vcos:.4f}")

    print("\n[download] Kaggle Output tab → Data / Output:")
    print(f"  • Prefer:  {CFG.EXPORT_ZIP}  (single zip: PyTorch + ONNX + readme)")
    print(f"  • Or:      {CFG.BEST_PT_ALIAS}  and  {CFG.ONNX_OUT}")
    print(f"  • All checkpoints: {CFG.CKPT_DIR}/")
    print(
        "\n[download] If the browser never finishes loading a file, use the clickable "
        "links above (notebook), or install Kaggle CLI and run from your PC:\n"
        "  kaggle kernels output <user>/<kernel-slug> -p .\n"
        "  (Requires API token from kaggle.com → Account → API → Create New Token)"
    )

# ===========================================================================
# 11.  Entry point
# ===========================================================================
if __name__ == "__main__":
    train()
