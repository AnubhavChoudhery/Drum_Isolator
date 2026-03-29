"""
train.py — Multi-GPU Training Loop for DrumUNet

Usage (single node, all visible GPUs):
    python train.py --musdb_root /path/to/musdb18 --epochs 100

The model wraps in nn.DataParallel automatically when multiple GPUs are present.
Checkpoints are saved every SAVE_EVERY epochs and at training end.
"""

import argparse
import os
import time

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset import DEFAULT_MAX_TIME_FRAMES, build_dataloader, SAMPLE_RATE
from model import build_model


class CombinedSpectralLoss(nn.Module):
    """0.7 L1 + 0.3 SmoothL1 on predicted vs target magnitude (optional)."""

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
    raise ValueError(f"Unknown loss {name!r}; use l1, smooth_l1, or combined.")

# ---------------------------------------------------------------------------
# Hyper-parameters (override via CLI args)
# ---------------------------------------------------------------------------
DEFAULT_EPOCHS:     int   = 100
DEFAULT_BATCH:      int   = 8
DEFAULT_LR:         float = 1e-4
DEFAULT_WORKERS:    int   = 4
DEFAULT_SAVE_EVERY: int   = 10
DEFAULT_CKPT_DIR:   str   = "checkpoints"


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: nn.Module,
    optimizer: Adam,
    epoch: int,
    loss: float,
    ckpt_dir: str,
    tag: str = "",
) -> None:
    os.makedirs(ckpt_dir, exist_ok=True)
    filename = f"drum_unet_epoch{epoch:04d}{tag}.pt"
    path = os.path.join(ckpt_dir, filename)

    # Unwrap DataParallel to save raw module state
    module = model.module if isinstance(model, nn.DataParallel) else model
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": module.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        path,
    )
    print(f"  [ckpt] saved → {path}")


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Adam | None = None,
) -> int:
    """Loads checkpoint; returns the epoch number to resume from."""
    ckpt = torch.load(path, map_location="cpu")
    module = model.module if isinstance(model, nn.DataParallel) else model
    module.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    print(f"  [ckpt] loaded from {path} (epoch {ckpt['epoch']}, loss {ckpt['loss']:.6f})")
    return ckpt["epoch"]


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.backends.cudnn.benchmark = True
        print(f"Device: {device}  ({torch.cuda.get_device_name(0)})  |  CUDA {torch.version.cuda}")
    else:
        device = torch.device("cpu")
        if args.require_gpu:
            raise RuntimeError(
                "CUDA not available but --require_gpu was set. "
                "Use a machine with a GPU or omit --require_gpu for CPU training."
            )
        print("Device: cpu (no GPU visible to PyTorch — training will be slow)")

    n_gpus = torch.cuda.device_count()
    print(f"GPUs visible: {n_gpus}")

    # ----- Model -----
    model = build_model()
    if n_gpus > 1:
        print(f"  Wrapping model in DataParallel across {n_gpus} GPUs.")
        model = nn.DataParallel(model)
    model = model.to(device)

    # ----- Optimiser & loss -----
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    criterion = build_criterion(args.loss)

    # ----- Resume -----
    start_epoch = 1
    if args.resume:
        start_epoch = load_checkpoint(args.resume, model, optimizer) + 1

    # ----- Data -----
    max_tf = args.max_time_frames if args.max_time_frames >= 0 else None

    train_loader = build_dataloader(
        root=args.musdb_root,
        subset="train",
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=True,
        max_time_frames=max_tf,
    )

    # ----- Epoch loop -----
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for step, (mix_mag, drums_mag) in enumerate(train_loader, start=1):
            mix_mag   = mix_mag.to(device, non_blocking=True)    # [B, 1, FREQ_BINS, T]
            drums_mag = drums_mag.to(device, non_blocking=True)  # [B, 1, FREQ_BINS, T]

            optimizer.zero_grad(set_to_none=True)

            # Forward: predict soft mask, apply to mixture
            soft_mask      = model(mix_mag)                   # [B, 1, FREQ_BINS, T]
            predicted_drums = soft_mask * mix_mag             # element-wise masking

            loss = criterion(predicted_drums, drums_mag)
            loss.backward()

            # Gradient clipping to stabilise training
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

            if step % 50 == 0:
                avg = epoch_loss / step
                elapsed = time.time() - t0
                print(f"  Epoch [{epoch}/{args.epochs}] Step [{step}/{len(train_loader)}] "
                      f"Loss: {avg:.6f}  Elapsed: {elapsed:.1f}s")

        avg_loss = epoch_loss / len(train_loader)
        scheduler.step(avg_loss)
        print(f"Epoch {epoch}/{args.epochs} complete — avg loss: {avg_loss:.6f}  "
              f"lr: {optimizer.param_groups[0]['lr']:.2e}")

        if epoch % args.save_every == 0:
            save_checkpoint(model, optimizer, epoch, avg_loss, args.ckpt_dir)

    # Final checkpoint
    save_checkpoint(model, optimizer, args.epochs, avg_loss, args.ckpt_dir, tag="_final")
    print("Training complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DrumUNet on MUSDB18")
    p.add_argument("--musdb_root",  type=str,   required=True,            help="Path to MUSDB18 dataset root")
    p.add_argument("--epochs",      type=int,   default=DEFAULT_EPOCHS,   help="Number of training epochs")
    p.add_argument("--batch_size",  type=int,   default=DEFAULT_BATCH,    help="Batch size per DataParallel step")
    p.add_argument("--lr",          type=float, default=DEFAULT_LR,       help="Initial Adam learning rate")
    p.add_argument("--workers",     type=int,   default=DEFAULT_WORKERS,  help="DataLoader worker count")
    p.add_argument("--save_every",  type=int,   default=DEFAULT_SAVE_EVERY, help="Checkpoint save interval (epochs)")
    p.add_argument("--ckpt_dir",    type=str,   default=DEFAULT_CKPT_DIR, help="Directory for checkpoint files")
    p.add_argument("--resume",      type=str,   default=None,             help="Path to checkpoint to resume from")
    p.add_argument(
        "--max_time_frames",
        type=int,
        default=DEFAULT_MAX_TIME_FRAMES,
        help="Max STFT time frames per example (caps GPU memory). Use -1 to disable cropping (needs huge VRAM).",
    )
    p.add_argument(
        "--loss",
        type=str,
        default="combined",
        choices=("l1", "smooth_l1", "combined"),
        help="Magnitude-domain loss: l1 (MAE), smooth_l1 (Huber), or combined (0.7*L1+0.3*SmoothL1).",
    )
    p.add_argument(
        "--require_gpu",
        action="store_true",
        help="Exit with error if CUDA is not available (recommended for cloud GPU jobs).",
    )
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
