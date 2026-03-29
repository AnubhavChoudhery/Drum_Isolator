"""
dataset.py — MUSDB18 Dataset Loader with STFT Preprocessing

Loads mixture and drum stems from MUSDB18, applies overlapping Hann-windowed
STFT, and returns (mix_magnitude, drums_magnitude) pairs for supervised training.
"""

from __future__ import annotations

import random

import torch
from torch.utils.data import Dataset, DataLoader
from torch import Tensor

# ---------------------------------------------------------------------------
# Shared STFT Contract — must match cpp/include/constants.h exactly
# ---------------------------------------------------------------------------
SAMPLE_RATE: int = 44100
N_FFT: int = 2048
HOP_LENGTH: int = 512
WIN_LENGTH: int = 2048
FREQ_BINS: int = N_FFT // 2 + 1   # 1025

# Upper bound on STFT time frames per sample — prevents GPU OOM when full songs
# yield T in the tens of thousands.  Must be ≥ minimum U-Net width (~16).
DEFAULT_MAX_TIME_FRAMES: int = 1024


def _build_hann_window(device: torch.device) -> torch.Tensor:
    return torch.hann_window(WIN_LENGTH, device=device)


def _to_mono(waveform: torch.Tensor) -> torch.Tensor:
    """Mix-down to mono. waveform shape: [channels, samples]."""
    if waveform.ndim == 1:
        return waveform
    return waveform.mean(dim=0)


def _resample_if_needed(waveform: torch.Tensor, orig_sr: int) -> torch.Tensor:
    if orig_sr != SAMPLE_RATE:
        import torchaudio

        waveform = torchaudio.functional.resample(waveform, orig_sr, SAMPLE_RATE)
    return waveform


def compute_stft(
    waveform: torch.Tensor,
    window: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute magnitude and phase spectrograms via overlapping Hann-windowed STFT.

    Args:
        waveform: 1-D float tensor of mono audio samples.
        window:   Pre-built Hann window tensor on the correct device.

    Returns:
        magnitude: [FREQ_BINS, T] float tensor.
        phase:     [FREQ_BINS, T] float tensor (radians).
    """
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
    )  # shape: [FREQ_BINS, T]

    magnitude = stft_complex.abs()          # [FREQ_BINS, T]
    phase = stft_complex.angle()            # [FREQ_BINS, T]
    return magnitude, phase


def crop_time_axis_pair(
    mix: Tensor,
    drums: Tensor,
    max_frames: int,
    *,
    random_crop: bool,
) -> tuple[Tensor, Tensor]:
    """
    Crop mix & drums identically along time so GPU memory stays bounded.

    ``mix`` / ``drums``: [1, FREQ_BINS, T].  If T > max_frames, take a
    contiguous slice of length max_frames (random start for training,
    centered for validation).
    """
    t = int(mix.shape[-1])
    if t <= max_frames:
        return mix, drums
    if random_crop:
        start = random.randint(0, t - max_frames)
    else:
        start = (t - max_frames) // 2
    sl = slice(start, start + max_frames)
    return mix[..., sl], drums[..., sl]


class MusdbDrumDataset(Dataset):
    """
    PyTorch Dataset wrapping MUSDB18.

    Each item is a pair of magnitude spectrograms:
        mix_magnitude:   [1, FREQ_BINS, T]  (mixture)
        drums_magnitude: [1, FREQ_BINS, T]  (isolated drums target)

    Args:
        root:     Path to the MUSDB18 root directory.
        subset:   "train" or "test".
        device:   Torch device for window tensor (keep on CPU for DataLoader workers).
        max_time_frames: Cap STFT length per example (default DEFAULT_MAX_TIME_FRAMES).
                         Set ``None`` only if you accept very high GPU memory use.
    """

    def __init__(
        self,
        root: str,
        subset: str = "train",
        device: torch.device = torch.device("cpu"),
        max_time_frames: int | None = DEFAULT_MAX_TIME_FRAMES,
    ):
        import musdb  # optional training dependency — not required for collate-only imports

        self.db = musdb.DB(root=root, subsets=subset)
        self.tracks = list(self.db.tracks)
        self.window = _build_hann_window(device)
        self.subset = subset
        self.max_time_frames = max_time_frames

    def __len__(self) -> int:
        return len(self.tracks)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        track = self.tracks[idx]

        # musdb returns numpy arrays; shape [samples, channels]
        mix_np = track.audio.T                 # [channels, samples]
        drums_np = track.stems[1].T            # drums stem index = 1

        mix_wav = torch.from_numpy(mix_np).float()
        drums_wav = torch.from_numpy(drums_np).float()

        mix_mono = _to_mono(mix_wav)
        drums_mono = _to_mono(drums_wav)

        mix_mag, _ = compute_stft(mix_mono, self.window)
        drums_mag, _ = compute_stft(drums_mono, self.window)

        # Add channel dim → [1, FREQ_BINS, T]
        mix_mag = mix_mag.unsqueeze(0)
        drums_mag = drums_mag.unsqueeze(0)

        if self.max_time_frames is not None:
            random_crop = self.subset == "train"
            mix_mag, drums_mag = crop_time_axis_pair(
                mix_mag,
                drums_mag,
                self.max_time_frames,
                random_crop=random_crop,
            )

        return mix_mag, drums_mag


def spectrogram_collate_fn(batch: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
    """
    Pad variable-length spectrograms along time so a batch can be stacked.

    Each track yields a different STFT length T; the default DataLoader collate
    cannot stack [1, F, T_i] tensors.  We pad (mix, drums) to max T in the batch
    with zeros so shapes match for the U-Net and L1 loss.
    """
    mix_list, drum_list = zip(*batch)
    t_max = max(int(x.shape[-1]) for x in mix_list)
    f_bins = int(mix_list[0].shape[1])
    bsz = len(batch)

    mix_b = torch.zeros(bsz, 1, f_bins, t_max, dtype=mix_list[0].dtype)
    drum_b = torch.zeros(bsz, 1, f_bins, t_max, dtype=drum_list[0].dtype)

    for i, (m, d) in enumerate(zip(mix_list, drum_list)):
        t = int(m.shape[-1])
        mix_b[i, :, :, :t] = m
        drum_b[i, :, :, :t] = d

    return mix_b, drum_b


def build_dataloader(
    root: str,
    subset: str = "train",
    batch_size: int = 8,
    num_workers: int = 4,
    shuffle: bool = True,
    max_time_frames: int | None = DEFAULT_MAX_TIME_FRAMES,
    pin_memory: bool | None = None,
    prefetch_factor: int = 4,
) -> DataLoader:
    """
    Convenience factory that returns a DataLoader ready for multi-GPU training.

    ``pin_memory`` defaults to ``torch.cuda.is_available()`` (faster H2D copy when training on GPU).
    """
    import torch as _torch

    if pin_memory is None:
        pin_memory = _torch.cuda.is_available()

    dataset = MusdbDrumDataset(root=root, subset=subset, max_time_frames=max_time_frames)
    kw = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        collate_fn=spectrogram_collate_fn,
    )
    if num_workers > 0:
        kw["persistent_workers"] = True
        kw["prefetch_factor"] = prefetch_factor
    return DataLoader(**kw)
