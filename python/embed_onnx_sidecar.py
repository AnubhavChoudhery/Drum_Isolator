"""
Merge a split ONNX export (model.onnx + model.onnx.data) into a single model.onnx.

Use when an older export left two files on Kaggle and you copied both locally:
    .venv\\Scripts\\python python/embed_onnx_sidecar.py models/drums_unet.onnx

Requires: pip install onnx
"""

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from export import _embed_onnx_weights_inline  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx_path", help="Path to .onnx (sidecar must be onnx_path + '.data')")
    args = ap.parse_args()
    path = os.path.abspath(args.onnx_path)
    sidecar = path + ".data"
    if not os.path.isfile(path):
        raise SystemExit(f"Missing: {path}")
    if not os.path.isfile(sidecar):
        raise SystemExit(
            f"Missing sidecar: {sidecar}\n"
            "If you only have a lone .onnx from an incomplete copy, re-export from your .pt:\n"
            "  python export.py --checkpoint path/to/best.pt --output models/drums_unet.onnx --verify"
        )
    _embed_onnx_weights_inline(path)
    print(f"OK — single-file ONNX: {path}")


if __name__ == "__main__":
    main()
