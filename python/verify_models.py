"""
verify_models.py — Smoke-test exported models under ../models/

Run from repo root with the project venv:
    .venv\\Scripts\\python python/verify_models.py

Checks:
  • ONNX (onnxruntime): session + forward on dummy [1,1,1025,T]
  • Optional PyTorch .pt: load state_dict + match ONNX output on same input (if file present)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# Repo layout: python/verify_models.py → parents[1] = SP_Proj
_ROOT = Path(__file__).resolve().parents[1]
_MODELS = _ROOT / "models"

FREQ_BINS = 1025
FRAME_COUNT = 256


def _find_onnx() -> Path | None:
    for name in ("drums_unet.onnx", "best_drums_unet.onnx"):
        p = _MODELS / name
        if p.is_file():
            return p
    return None


def _find_pt() -> Path | None:
    for name in ("best_drums_unet.pt", "drum_unet_best.pt"):
        p = _MODELS / name
        if p.is_file():
            return p
    for p in _MODELS.glob("*.pt"):
        return p
    return None


def verify_onnx(path: Path) -> None:
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    try:
        sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    except Exception as ex:
        sidecar = Path(str(path) + ".data")
        hint = (
            "\n  PyTorch often saves weights in a sidecar file `drums_unet.onnx.data`. "
            "Copy that file next to your .onnx, then run:\n"
            f"    python python/embed_onnx_sidecar.py {path}\n"
            "  Or re-export a single-file ONNX from your .pt:\n"
            "    python python/export.py --checkpoint models/best_drums_unet.pt --output models/drums_unet.onnx --verify"
        )
        if "onnx.data" in str(ex) or "cannot find the file" in str(ex).lower():
            raise RuntimeError(f"{ex}{hint}") from ex
        raise

    inputs = sess.get_inputs()
    outputs = sess.get_outputs()
    print(f"  ONNX inputs : {[i.name + ' ' + str(i.shape) for i in inputs]}")
    print(f"  ONNX outputs: {[o.name + ' ' + str(o.shape) for o in outputs]}")

    for T in (FRAME_COUNT, 128, 512):
        x = np.random.randn(1, 1, FREQ_BINS, T).astype(np.float32) * 0.1
        out = sess.run([outputs[0].name], {inputs[0].name: x})[0]
        assert out.shape == (1, 1, FREQ_BINS, T), (out.shape, T)
        assert out.min() >= -1e-3 and out.max() <= 1.0 + 1e-3, (out.min(), out.max())
    print(f"  ONNX forward OK (T={FRAME_COUNT},128,512), sigmoid range ~[{out.min():.4f},{out.max():.4f}]")


def verify_pt_matches_onnx(pt_path: Path, onnx_path: Path) -> None:
    import torch

    sys.path.insert(0, str(_ROOT / "python"))
    from export import load_model_from_checkpoint  # noqa: E402

    ort = __import__("onnxruntime")
    sess = ort.InferenceSession(
        str(onnx_path), ort.SessionOptions(), providers=["CPUExecutionProvider"]
    )
    on_in = sess.get_inputs()[0].name
    on_out = sess.get_outputs()[0].name

    model = load_model_from_checkpoint(str(pt_path))
    model.eval()
    torch.manual_seed(0)
    x = torch.randn(1, 1, FREQ_BINS, FRAME_COUNT) * 0.1
    with torch.no_grad():
        y_pt = model(x).numpy()
    y_onnx = sess.run([on_out], {on_in: x.numpy()})[0]
    diff = np.abs(y_pt - y_onnx).max()
    print(f"  PyTorch vs ONNX max abs diff on same tensor: {diff:.6f}")
    if diff > 1e-3:
        print("  Warning: mismatch > 1e-3 — re-export ONNX from this checkpoint if unexpected.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", type=Path, default=_MODELS)
    args = ap.parse_args()
    models_dir = args.models_dir.resolve()
    print(f"Models directory: {models_dir}")

    onnx_p = _find_onnx()
    if onnx_p is None:
        print("ERROR: No drums_unet.onnx (or best_drums_unet.onnx) found in models/")
        sys.exit(1)

    print(f"\n[1] ONNX: {onnx_p}")
    verify_onnx(onnx_p)

    pt_p = _find_pt()
    if pt_p is not None:
        print(f"\n[2] PyTorch checkpoint: {pt_p}")
        verify_pt_matches_onnx(pt_p, onnx_p)
    else:
        print("\n[2] No .pt in models/ — skipped (optional). Copy best_drums_unet.pt here to compare.")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
