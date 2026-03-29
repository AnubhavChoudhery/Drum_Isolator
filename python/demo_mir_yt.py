"""
demo_mir_yt.py — Download audio with yt-dlp and run the drum-isolation MIR pipeline (ONNX).

Uses the same STFT contract as dataset.py / C++ (44.1 kHz, n_fft=2048, hop=512).

Run from repo root with venv:
    .venv\\Scripts\\pip install yt-dlp soundfile
    .venv\\Scripts\\python python/demo_mir_yt.py "https://www.youtube.com/watch?v=..."

Offline check (no download):
    .venv\\Scripts\\python python/demo_mir_yt.py --synthetic

Requires ffmpeg on PATH for yt-dlp -x (audio extract). If YouTube returns a bot check,
run ``pip install -U yt-dlp`` or ``yt-dlp --cookies-from-browser chrome <url>``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_ROOT / "python"))

from dataset import (  # noqa: E402
    HOP_LENGTH,
    N_FFT,
    SAMPLE_RATE,
    WIN_LENGTH,
    compute_stft,
    _build_hann_window,
    _resample_if_needed,
)


def _find_default_onnx(models_dir: Path) -> Path:
    for name in ("drums_unet.onnx", "best_drums_unet.onnx"):
        p = models_dir / name
        if p.is_file():
            return p
    raise FileNotFoundError(f"No drums_unet.onnx in {models_dir}")


def _write_synthetic_wav(path: Path, seconds: float = 4.0) -> None:
    """Short synthetic mix (tone + gated noise) for offline pipeline checks."""
    import soundfile as sf

    sr = SAMPLE_RATE
    n = int(sr * seconds)
    t = np.linspace(0.0, seconds, n, endpoint=False, dtype=np.float32)
    x = 0.15 * np.sin(2.0 * np.pi * 220.0 * t).astype(np.float32)
    gate = (np.sin(2.0 * np.pi * 8.0 * t) > 0.65).astype(np.float32)
    x = x + 0.35 * np.sin(2.0 * np.pi * 2.0 * t).astype(np.float32) * gate
    x *= 0.9 / (np.max(np.abs(x)) + 1e-8)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), x, sr)


def _download_ytdlp(
    url: str,
    out_dir: Path,
    *,
    prefer_wav: bool,
) -> Path:
    """Return path to downloaded audio file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / "%(title).80B_%(id)s.%(ext)s")
    cmd = [
        "yt-dlp",
        "--no-playlist",
        "--no-update",
        "-o",
        pattern,
    ]
    if prefer_wav:
        cmd += ["-x", "--audio-format", "wav"]
    else:
        cmd += ["-f", "bestaudio", "-x", "--audio-format", "wav"]
    cmd.append(url)

    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(
            "yt-dlp failed.\n"
            f"  command: {' '.join(cmd)}\n"
            f"  stderr: {r.stderr or r.stdout}"
        )

    wavs = sorted(out_dir.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not wavs:
        raise RuntimeError(f"yt-dlp reported success but no .wav in {out_dir}:\n{r.stdout}")
    return wavs[0]


def _load_waveform(path: Path, device: torch.device) -> torch.Tensor:
    """Load float32 mono @ SAMPLE_RATE using soundfile (avoids torchcodec in recent torchaudio)."""
    import soundfile as sf

    data, sr = sf.read(str(path), always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=-1)
    wav = torch.from_numpy(np.asarray(data, dtype=np.float32))
    wav = _resample_if_needed(wav, int(sr))
    return wav.to(device=device, dtype=torch.float32)


def _pad_time_to_multiple(x: torch.Tensor, mult: int) -> tuple[torch.Tensor, int]:
    """Pad [..., T] along last dim so T % mult == 0. Returns (padded, original_T)."""
    t_orig = int(x.shape[-1])
    rem = t_orig % mult
    if rem == 0:
        return x, t_orig
    pad = mult - rem
    # 4D [B,C,F,T]: pad pairs last dim (time) then freq — only pad time
    return Fn.pad(x, (0, pad, 0, 0), mode="replicate"), t_orig


def _infer_mask_onnx(
    mag: torch.Tensor,
    sess: "onnxruntime.InferenceSession",
    in_name: str,
    out_name: str,
    chunk_t: int,
    min_t: int,
    *,
    align_t: int = 16,
) -> torch.Tensor:
    """
    mag: [1, 1, FREQ_BINS, T] on CPU or CUDA tensor; ORT runs on numpy CPU.
    Returns mask same shape as mag.

    U-Net skip connections require T (and chunk sizes after padding) to stay aligned — pad
    time to a multiple of ``align_t`` (16 matches 4× MaxPool on the time axis).
    """
    import onnxruntime as ort

    _ = ort  # noqa: F841 — type hint only

    b, c, f, t = mag.shape
    out = torch.empty_like(mag)
    x_np = mag.detach().cpu().numpy().astype(np.float32)

    for start in range(0, t, chunk_t):
        end = min(start + chunk_t, t)
        sl = x_np[:, :, :, start:end]
        th = end - start
        if th < min_t:
            pad = min_t - th
            sl = np.pad(sl, ((0, 0), (0, 0), (0, 0), (0, pad)), mode="edge")
            th2 = min_t
        else:
            th2 = th
        rem = th2 % align_t
        if rem != 0:
            p2 = align_t - rem
            sl = np.pad(sl, ((0, 0), (0, 0), (0, 0), (0, p2)), mode="edge")
        pred = sess.run([out_name], {in_name: sl})[0]
        pred = pred[:, :, :, :th]
        out[:, :, :, start:end] = torch.from_numpy(pred.astype(np.float32)).to(mag.device)

    return out


def separate_drums(
    waveform: torch.Tensor,
    sess: "onnxruntime.InferenceSession",
    in_name: str,
    out_name: str,
    window: torch.Tensor,
    *,
    chunk_t: int,
    min_t: int,
) -> torch.Tensor:
    """Return time-domain drum estimate [samples] same length as input (center=True istft)."""
    device = waveform.device
    magnitude, phase = compute_stft(waveform, window)
    mag = magnitude.unsqueeze(0).unsqueeze(0)
    mag_u, t_orig = _pad_time_to_multiple(mag, 16)
    mask_u = _infer_mask_onnx(
        mag_u, sess, in_name, out_name, chunk_t, min_t, align_t=16
    )
    mask = mask_u[:, :, :, :t_orig]
    drum_mag = (mask * mag).squeeze(0).squeeze(0)
    drum_c = drum_mag * torch.exp(1j * phase)
    out = torch.istft(
        drum_c,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=window,
        center=True,
        normalized=False,
        length=waveform.shape[-1],
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="yt-dlp + ONNX drum separation demo")
    ap.add_argument("url", nargs="?", help="YouTube (or yt-dlp supported) URL")
    ap.add_argument(
        "--audio-file",
        type=Path,
        help="Skip download; use this wav/flac/etc. (soundfile)",
    )
    ap.add_argument(
        "--synthetic",
        action="store_true",
        help="Do not download — generate a short synthetic WAV in --out-dir and process it",
    )
    ap.add_argument("--onnx", type=Path, help=f"default: {_ROOT / 'models' / 'drums_unet.onnx'}")
    ap.add_argument("--out-dir", type=Path, default=_ROOT / "downloads" / "mir_demo")
    ap.add_argument("--out-wav", type=Path, help="Output path (default: out_dir/drums_isolated.wav)")
    ap.add_argument("--max-seconds", type=float, default=90.0, help="Trim after resampling (0 = full)")
    ap.add_argument("--chunk-t", type=int, default=1024, help="ONNX time-chunk size (frames)")
    ap.add_argument("--device", default="cpu", help="torch device (wav + STFT; ORT uses CPU here)")
    ap.add_argument(
        "--dry-download",
        action="store_true",
        help="Only run yt-dlp and print path; no separation",
    )
    args = ap.parse_args()

    if not args.synthetic and not args.audio_file and not args.url:
        ap.error("Provide a URL, --audio-file, or --synthetic")

    models_dir = _ROOT / "models"
    onnx_path = args.onnx or _find_default_onnx(models_dir)

    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
    ins = sess.get_inputs()[0].name
    outs = sess.get_outputs()[0].name

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        src = (out_dir / "synthetic_mix.wav").resolve()
        print(f"Writing synthetic mix: {src}")
        _write_synthetic_wav(src, seconds=5.0)
    elif args.audio_file:
        src = args.audio_file.resolve()
        if not src.is_file():
            print(f"ERROR: --audio-file not found: {src}", file=sys.stderr)
            sys.exit(1)
    else:
        try:
            subprocess.run(["yt-dlp", "--version"], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            print("ERROR: yt-dlp not found. Install in venv:  pip install yt-dlp", file=sys.stderr)
            sys.exit(1)
        print("Downloading with yt-dlp (needs ffmpeg for -x)...")
        print(
            "  Tip if YouTube fails: pip install -U yt-dlp  or  "
            "yt-dlp --cookies-from-browser chrome <url>"
        )
        src = _download_ytdlp(args.url, out_dir, prefer_wav=True)
        print(f"  saved: {src}")

    if args.dry_download:
        print("Dry download only; done.")
        return

    device = torch.device(args.device)
    window = _build_hann_window(device)
    wav = _load_waveform(src, device)

    n = wav.shape[-1]
    if args.max_seconds and args.max_seconds > 0:
        max_samples = int(SAMPLE_RATE * args.max_seconds)
        if n > max_samples:
            wav = wav[..., :max_samples]
            print(f"Trimmed to {args.max_seconds:g}s ({max_samples} samples)")

    print(f"Waveform samples: {wav.shape[-1]}  (~{wav.shape[-1] / SAMPLE_RATE:.2f}s)")
    print("Running ONNX drum mask + ISTFT...")

    drums = separate_drums(wav, sess, ins, outs, window, chunk_t=args.chunk_t, min_t=16)

    peak = drums.abs().max().item()
    if peak > 1.0:
        drums = drums / peak
        print(f"Peak-normalized output (peak was {peak:.4f})")

    out_wav = args.out_wav or (out_dir / "drums_isolated.wav")
    out_wav = out_wav.resolve()
    out_wav.parent.mkdir(parents=True, exist_ok=True)

    import soundfile as sf

    sf.write(str(out_wav), drums.cpu().numpy().astype(np.float32), SAMPLE_RATE)
    print(f"Wrote drum stem: {out_wav}")
    print("Done. Play alongside the original to hear the pipeline.")


if __name__ == "__main__":
    main()
