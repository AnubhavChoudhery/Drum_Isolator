#pragma once

/**
 * constants.h — Shared STFT and chunk-size constants for the MIR engine.
 *
 * CRITICAL: These values must match the Python training pipeline exactly:
 *   python/dataset.py  — N_FFT, HOP_LENGTH, WIN_LENGTH, SAMPLE_RATE
 *   python/export.py   — FREQ_BINS, FRAME_COUNT
 *
 * Any mismatch between Python and C++ will cause incorrect masking output.
 */

namespace mir {

// ---------------------------------------------------------------------------
// Audio properties
// ---------------------------------------------------------------------------

/// Playback and processing sample rate in Hz.
inline constexpr int kSampleRate = 44100;

/// Number of audio channels processed (mono).
inline constexpr int kChannels = 1;

// ---------------------------------------------------------------------------
// STFT parameters  (must mirror torch.stft call in dataset.py)
// ---------------------------------------------------------------------------

/// FFT size (window length). Power of two for FFTW3 efficiency.
inline constexpr int kNfft = 2048;

/// Analysis hop size in samples. 512 → 75 % overlap with a 2048-pt window.
inline constexpr int kHopLength = 512;

/// Analysis window length (equals kNfft — no zero-padding in the window).
inline constexpr int kWinLength = 2048;

// ---------------------------------------------------------------------------
// Derived spectrogram dimensions
// ---------------------------------------------------------------------------

/// Number of real-valued frequency bins from a real-input FFT: kNfft/2 + 1.
inline constexpr int kFreqBins = kNfft / 2 + 1;   // 1025

/// Number of time frames fed to the ONNX model per inference call.
/// Must match the dummy input T-dimension used during torch.onnx.export.
inline constexpr int kFrameCount = 256;

/// Total PCM samples needed to build kFrameCount STFT frames with full
/// context (including the initial overlap):
///   kHopLength * kFrameCount + (kNfft - kHopLength)
inline constexpr int kChunkSamples = kHopLength * kFrameCount + (kNfft - kHopLength);

// ---------------------------------------------------------------------------
// Ring buffer sizing
// ---------------------------------------------------------------------------

/// Input ring buffer capacity in float samples. Must be a power of two and
/// large enough to decouple the file-reader thread from the inference thread.
/// 4 × kChunkSamples rounded up to next power of two ≈ 524288.
inline constexpr std::size_t kInputRingCapacity  = 1u << 19;  // 524288 samples

/// Output ring buffer capacity in float samples. Sized to hold several
/// inference outputs ahead of the PortAudio callback.
inline constexpr std::size_t kOutputRingCapacity = 1u << 17;  // 131072 samples

// ---------------------------------------------------------------------------
// Overlap-add parameters
// ---------------------------------------------------------------------------

/// Synthesis hop equals analysis hop (no time-stretching).
inline constexpr int kSynthHop = kHopLength;

} // namespace mir
