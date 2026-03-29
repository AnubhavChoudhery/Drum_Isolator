/**
 * inference_engine.cpp — Core DSP + ONNX Inference Pipeline
 *
 * Full pipeline per chunk:
 *   PCM chunk → Hann window × kFrameCount → FFTW3 forward FFTs
 *   → magnitude + phase batch → ONNX session.Run() → soft mask
 *   → masked_mag × exp(i×phase) → FFTW3 inverse FFTs → overlap-add
 *   → push to output ring buffer for PortAudio playback
 */

#include "inference_engine.h"

#include <cmath>
#include <complex>
#include <stdexcept>
#include <thread>
#include <chrono>
#include <cassert>
#include <filesystem>
#include <numbers>

namespace mir {

// ---------------------------------------------------------------------------
// Construction / Destruction
// ---------------------------------------------------------------------------

InferenceEngine::InferenceEngine(
    const std::string&                      model_path,
    RingBuffer<float, kInputRingCapacity>&  input_rb,
    RingBuffer<float, kOutputRingCapacity>& output_rb,
    const std::atomic<bool>&               eof_flag
)
    : ort_env_(ORT_LOGGING_LEVEL_WARNING, "mir_engine")
    , session_(ort_env_, std::filesystem::path(model_path).c_str(), session_opts_)
    , input_rb_(input_rb)
    , output_rb_(output_rb)
    , eof_flag_(eof_flag)
{
    // ------------------------------------------------------------------
    // ONNX session configuration
    // ------------------------------------------------------------------
    session_opts_.SetIntraOpNumThreads(1);   // single-threaded: audio thread budget
    session_opts_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // ------------------------------------------------------------------
    // Allocate FFTW3 buffers using fftw_malloc for SIMD alignment
    // ------------------------------------------------------------------
    fftw_in_  = reinterpret_cast<float*>(
                    fftwf_malloc(sizeof(float) * static_cast<std::size_t>(kNfft)));
    fftw_out_ = reinterpret_cast<fftwf_complex*>(
                    fftwf_malloc(sizeof(fftwf_complex) * static_cast<std::size_t>(kFreqBins)));
    ifft_in_  = reinterpret_cast<fftwf_complex*>(
                    fftwf_malloc(sizeof(fftwf_complex) * static_cast<std::size_t>(kFreqBins)));
    ifft_out_ = reinterpret_cast<float*>(
                    fftwf_malloc(sizeof(float) * static_cast<std::size_t>(kNfft)));

    if (!fftw_in_ || !fftw_out_ || !ifft_in_ || !ifft_out_)
        throw std::runtime_error("InferenceEngine: fftwf_malloc failed.");

    // FFTW_MEASURE: times several FFT algorithms at startup, picks the fastest.
    // Safe here because plans are created once before audio streaming begins.
    fwd_plan_ = fftwf_plan_dft_r2c_1d(kNfft, fftw_in_, fftw_out_, FFTW_MEASURE);
    inv_plan_ = fftwf_plan_dft_c2r_1d(kNfft, ifft_in_, ifft_out_, FFTW_MEASURE);

    if (!fwd_plan_ || !inv_plan_)
        throw std::runtime_error("InferenceEngine: fftwf_plan creation failed.");

    // ------------------------------------------------------------------
    // Pre-allocate working buffers
    // ------------------------------------------------------------------
    magnitude_batch_.resize(static_cast<std::size_t>(kFrameCount * kFreqBins), 0.0f);
    phase_batch_.resize(    static_cast<std::size_t>(kFrameCount * kFreqBins), 0.0f);

    // OLA buffer: 2×kNfft provides one full window of run-out overlap.
    ola_buffer_.assign(static_cast<std::size_t>(2 * kNfft), 0.0f);

    build_hann_window();
}

InferenceEngine::~InferenceEngine() {
    if (fwd_plan_) fftwf_destroy_plan(fwd_plan_);
    if (inv_plan_) fftwf_destroy_plan(inv_plan_);
    if (fftw_in_)  fftwf_free(fftw_in_);
    if (fftw_out_) fftwf_free(fftw_out_);
    if (ifft_in_)  fftwf_free(ifft_in_);
    if (ifft_out_) fftwf_free(ifft_out_);
}

// ---------------------------------------------------------------------------
// Window construction
// ---------------------------------------------------------------------------

void InferenceEngine::build_hann_window() {
    hann_window_.resize(static_cast<std::size_t>(kNfft));
    const double two_pi = 2.0 * std::numbers::pi;
    for (int n = 0; n < kNfft; ++n) {
        // Periodic (non-symmetric) Hann window — matches torch.hann_window(periodic=True)
        hann_window_[static_cast<std::size_t>(n)] =
            0.5f * (1.0f - static_cast<float>(std::cos(two_pi * n / kNfft)));
    }
}

// ---------------------------------------------------------------------------
// Main thread loop
// ---------------------------------------------------------------------------

void InferenceEngine::run() {
    std::vector<float> pcm_chunk(static_cast<std::size_t>(kChunkSamples));

    while (!stop_requested_.load(std::memory_order_acquire)) {
        // Block until we have a full chunk to process
        if (!input_rb_.pop(pcm_chunk.data(), static_cast<std::size_t>(kChunkSamples))) {
            // If EOF and no more data, drain whatever is left and exit
            if (eof_flag_.load(std::memory_order_acquire) &&
                input_rb_.size_approx() < static_cast<std::size_t>(kChunkSamples)) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(200));
            continue;
        }

        process_chunk(pcm_chunk);
    }
}

// ---------------------------------------------------------------------------
// Chunk processing
// ---------------------------------------------------------------------------

bool InferenceEngine::process_chunk(const std::vector<float>& pcm_chunk) {
    // ------------------------------------------------------------------
    // Step 1–4: Forward FFT all frames → fill magnitude_batch_, phase_batch_
    // ------------------------------------------------------------------
    for (int frame = 0; frame < kFrameCount; ++frame) {
        const int sample_offset = frame * kHopLength;

        // Apply Hann window
        for (int n = 0; n < kNfft; ++n) {
            fftw_in_[n] = pcm_chunk[static_cast<std::size_t>(sample_offset + n)]
                          * hann_window_[static_cast<std::size_t>(n)];
        }

        run_forward_fft(fftw_in_, frame);
    }

    // ------------------------------------------------------------------
    // Step 5–6: ONNX inference → soft mask
    // ------------------------------------------------------------------
    std::vector<float> mask_data(static_cast<std::size_t>(kFrameCount * kFreqBins));
    run_onnx(mask_data);

    // ------------------------------------------------------------------
    // Step 7–10: Inverse FFT per frame + overlap-add + push to output
    // ------------------------------------------------------------------
    for (int frame = 0; frame < kFrameCount; ++frame) {
        run_inverse_fft_and_ola(mask_data.data(), frame);
        flush_hop_to_output();
    }

    return true;
}

// ---------------------------------------------------------------------------
// Forward FFT helpers
// ---------------------------------------------------------------------------

void InferenceEngine::run_forward_fft(const float* windowed_frame, int frame_idx) {
    // fftw_in_ already holds the windowed frame; execute the pre-built plan.
    // (The caller wrote into fftw_in_ before calling this method.)
    std::memcpy(fftw_in_, windowed_frame, sizeof(float) * static_cast<std::size_t>(kNfft));
    fftwf_execute(fwd_plan_);

    const int base = frame_idx * kFreqBins;
    for (int k = 0; k < kFreqBins; ++k) {
        const float re = fftw_out_[k][0];
        const float im = fftw_out_[k][1];
        magnitude_batch_[static_cast<std::size_t>(base + k)] = std::hypot(re, im);
        phase_batch_[    static_cast<std::size_t>(base + k)] = std::atan2(im, re);
    }
}

// ---------------------------------------------------------------------------
// ONNX Runtime inference
// ---------------------------------------------------------------------------

void InferenceEngine::run_onnx(std::vector<float>& mask_out) {
    // Input tensor: [1, 1, kFreqBins, kFrameCount]
    const std::array<int64_t, 4> input_shape{1, 1,
        static_cast<int64_t>(kFreqBins),
        static_cast<int64_t>(kFrameCount)};

    // Rearrange magnitude_batch_ from [frame, bin] row-major to [bin, frame]
    // as expected by the ONNX model input [1, 1, FREQ_BINS, T].
    std::vector<float> onnx_input(static_cast<std::size_t>(kFreqBins * kFrameCount));
    for (int f = 0; f < kFrameCount; ++f) {
        for (int k = 0; k < kFreqBins; ++k) {
            onnx_input[static_cast<std::size_t>(k * kFrameCount + f)] =
                magnitude_batch_[static_cast<std::size_t>(f * kFreqBins + k)];
        }
    }

    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info,
        onnx_input.data(),
        onnx_input.size(),
        input_shape.data(),
        input_shape.size()
    );

    // Run inference
    const char* input_names[]  = {"mix_magnitude"};
    const char* output_names[] = {"drum_mask"};

    auto output_tensors = session_.Run(
        Ort::RunOptions{nullptr},
        input_names,  &input_tensor, 1,
        output_names, 1
    );

    // Copy mask back; model output is [1, 1, kFreqBins, kFrameCount]
    const float* raw_mask = output_tensors[0].GetTensorData<float>();
    const std::size_t n   = static_cast<std::size_t>(kFreqBins * kFrameCount);

    // Transpose back to [frame, bin] row-major for our phase array indexing
    for (int f = 0; f < kFrameCount; ++f) {
        for (int k = 0; k < kFreqBins; ++k) {
            mask_out[static_cast<std::size_t>(f * kFreqBins + k)] =
                raw_mask[static_cast<std::size_t>(k * kFrameCount + f)];
        }
    }
    (void)n;
}

// ---------------------------------------------------------------------------
// Inverse FFT + Overlap-Add
// ---------------------------------------------------------------------------

void InferenceEngine::run_inverse_fft_and_ola(const float* mask_data, int frame_idx) {
    const int base = frame_idx * kFreqBins;

    // Reconstruct complex spectrum: masked_mag × exp(i × phase)
    for (int k = 0; k < kFreqBins; ++k) {
        const float mag   = mask_data[static_cast<std::size_t>(base + k)]
                          * magnitude_batch_[static_cast<std::size_t>(base + k)];
        const float phase = phase_batch_[static_cast<std::size_t>(base + k)];
        ifft_in_[k][0] = mag * std::cos(phase);
        ifft_in_[k][1] = mag * std::sin(phase);
    }

    fftwf_execute(inv_plan_);

    // Normalise IFFT output (FFTW3 c2r is unnormalised by convention)
    const float norm = 1.0f / static_cast<float>(kNfft);

    // Apply synthesis Hann window and add into the OLA accumulation buffer.
    // The OLA write position advances by kHopLength each frame.
    const std::size_t buf_size = ola_buffer_.size();
    for (int n = 0; n < kNfft; ++n) {
        const std::size_t buf_idx =
            static_cast<std::size_t>(ola_write_pos_ + n) % buf_size;
        ola_buffer_[buf_idx] +=
            ifft_out_[n] * norm * hann_window_[static_cast<std::size_t>(n)];
    }
}

void InferenceEngine::flush_hop_to_output() {
    // Extract one completed kHopLength-sample hop from the head of the OLA buffer.
    std::vector<float> hop(static_cast<std::size_t>(kHopLength));
    const std::size_t buf_size = ola_buffer_.size();

    for (int n = 0; n < kHopLength; ++n) {
        const std::size_t buf_idx =
            static_cast<std::size_t>(ola_write_pos_ + n) % buf_size;
        hop[static_cast<std::size_t>(n)] = ola_buffer_[buf_idx];
        ola_buffer_[buf_idx] = 0.0f;   // clear after reading
    }

    // Advance write pointer by one hop
    ola_write_pos_ =
        static_cast<int>((static_cast<std::size_t>(ola_write_pos_) + static_cast<std::size_t>(kHopLength)) % buf_size);

    // Push to output ring buffer; spin-wait if full
    while (!output_rb_.push(hop.data(), static_cast<std::size_t>(kHopLength))) {
        if (stop_requested_.load(std::memory_order_acquire))
            return;
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
}

} // namespace mir
