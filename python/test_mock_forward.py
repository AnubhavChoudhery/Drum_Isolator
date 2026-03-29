"""
test_mock_forward.py — No dataset download; validates U-Net + collate + loss.

Run from repo root or python/:
    python test_mock_forward.py
"""

from __future__ import annotations

import sys

import torch
import torch.nn as nn

# Allow running as script from python/
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dataset import FREQ_BINS, spectrogram_collate_fn
from model import DrumUNet


def make_fake_batch(batch_size: int = 4, t_list: list[int] | None = None):
    """Simulate variable-length spectrograms from different track lengths."""
    if t_list is None:
        t_list = [256 + 37 * i for i in range(batch_size)]  # distinct T values
    assert len(t_list) == batch_size
    items = []
    for t in t_list:
        mix = torch.rand(1, FREQ_BINS, t)
        drm = torch.rand(1, FREQ_BINS, t)
        items.append((mix, drm))
    return spectrogram_collate_fn(items)


def main() -> None:
    device = torch.device("cpu")
    model = DrumUNet().to(device)
    crit = nn.L1Loss()

    # Variable T per sample → collate pads to common time
    mix, drums = make_fake_batch(4)
    assert mix.shape == drums.shape
    b, _, f, tt = mix.shape
    assert f == FREQ_BINS
    print(f"collated batch shape: {tuple(mix.shape)}")

    mix = mix.to(device)
    drums = drums.to(device)

    mask = model(mix)
    assert mask.shape == mix.shape, f"mask {mask.shape} vs mix {mix.shape}"

    pred = mask * mix
    loss = crit(pred, drums)
    loss.backward()
    print(f"forward + backward OK, loss={loss.item():.6f}")

    # Fixed time (ONNX export shape)
    x = torch.randn(2, 1, FREQ_BINS, 256, device=device)
    y = model(x)
    assert y.shape == x.shape
    print("fixed T=256 forward OK")

    # Minimum practical T for 4× MaxPool2d on width (avoid 0 spatial size)
    for tmin in (16, 32, 64):
        x2 = torch.randn(1, 1, FREQ_BINS, tmin, device=device)
        y2 = model(x2)
        assert y2.shape == x2.shape, (tmin, y2.shape, x2.shape)
    print("small T sanity OK")

    print("All mock tests passed.")


if __name__ == "__main__":
    main()
