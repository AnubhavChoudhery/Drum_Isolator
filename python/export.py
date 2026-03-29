"""
export.py — Export Trained DrumUNet to ONNX

Loads a PyTorch checkpoint and exports the model to ONNX format for ingestion
by the C++ ONNX Runtime inference engine.

Usage:
    python export.py --checkpoint checkpoints/drum_unet_final.pt \
                     --output ../models/drums_unet.onnx

STFT Contract — must match cpp/include/constants.h:
    n_fft       = 2048   → FREQ_BINS = 1025
    hop_length  = 512
    sample_rate = 44100
    frame_count = 256    (time frames per inference chunk — matches kFrameCount in C++)
"""

import argparse
import os

import torch
import torch.nn as nn

from model import build_model, DrumUNet

# ---------------------------------------------------------------------------
# Shared STFT / chunk constants — must mirror cpp/include/constants.h
# ---------------------------------------------------------------------------
SAMPLE_RATE:  int = 44100
N_FFT:        int = 2048
HOP_LENGTH:   int = 512
FREQ_BINS:    int = N_FFT // 2 + 1   # 1025
FRAME_COUNT:  int = 256              # kFrameCount in C++
OPSET:        int = 17


def _embed_onnx_weights_inline(path: str) -> None:
    """
    Merge external `path.data` into `path` so a single .onnx can be copied to C++ / other PCs.
    No-op if the `onnx` package is missing.
    """
    try:
        import onnx
    except ImportError:
        return
    try:
        model_proto = onnx.load(path)
        onnx.save(model_proto, path, save_as_external_data=False)
        sidecar = path + ".data"
        if os.path.isfile(sidecar):
            os.remove(sidecar)
    except Exception as ex:
        print(f"  (onnx inline merge skipped: {ex})")


def load_model_from_checkpoint(ckpt_path: str) -> DrumUNet:
    """Load model weights from a training checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Support checkpoints saved with or without DataParallel wrapping
    state_dict = ckpt.get("model_state_dict", ckpt)

    # Strip 'module.' prefix if model was saved via DataParallel
    cleaned = {k.removeprefix("module."): v for k, v in state_dict.items()}

    model = build_model()
    model.load_state_dict(cleaned)
    model.eval()
    return model


def export_to_onnx(model: nn.Module, output_path: str) -> None:
    """
    Export the model to ONNX.

    Input tensor shape: [1, 1, FREQ_BINS, FRAME_COUNT]
        batch=1, channels=1, freq_bins=1025, time_frames=256

    Dynamic axes are set on the time dimension (axis 3) so the C++ engine
    can pass any time-frame count without re-exporting.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    dummy_input = torch.zeros(1, 1, FREQ_BINS, FRAME_COUNT, dtype=torch.float32)

    dynamic_axes = {
        "mix_magnitude": {3: "time_frames"},
        "drum_mask":     {3: "time_frames"},
    }

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=OPSET,
        do_constant_folding=True,
        input_names=["mix_magnitude"],
        output_names=["drum_mask"],
        dynamic_axes=dynamic_axes,
        verbose=False,
    )
    # PyTorch may emit weights in a sidecar `*.onnx.data` — C++/portable copy needs one file.
    _embed_onnx_weights_inline(output_path)

    print(f"ONNX model exported → {output_path}")
    print(f"  Input  : mix_magnitude  [1, 1, {FREQ_BINS}, T]  (T is dynamic)")
    print(f"  Output : drum_mask      [1, 1, {FREQ_BINS}, T]  in [0,1]")
    print(f"  Opset  : {OPSET}")


def verify_onnx(output_path: str) -> None:
    """Optional runtime verification using onnxruntime (if installed)."""
    try:
        import onnxruntime as ort
        import numpy as np

        sess = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
        dummy = np.zeros((1, 1, FREQ_BINS, FRAME_COUNT), dtype=np.float32)
        outputs = sess.run(["drum_mask"], {"mix_magnitude": dummy})
        mask = outputs[0]
        print(f"  Verification OK — output shape: {mask.shape}, "
              f"range [{mask.min():.4f}, {mask.max():.4f}]")
    except ImportError:
        print("  (onnxruntime not installed — skipping verification)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export DrumUNet checkpoint to ONNX")
    p.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to .pt checkpoint file produced by train.py",
    )
    p.add_argument(
        "--output", type=str, default=os.path.join("..", "models", "drums_unet.onnx"),
        help="Destination path for the .onnx file (repo: place under models/; see README)",
    )
    p.add_argument(
        "--verify", action="store_true",
        help="Run a quick onnxruntime inference pass to verify the export",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model = load_model_from_checkpoint(args.checkpoint)
    export_to_onnx(model, args.output)
    if args.verify:
        verify_onnx(args.output)
