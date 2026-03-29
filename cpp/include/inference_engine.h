#pragma once

#include <atomic>
#include <string>
#include <vector>
#include <memory>

// ONNX Runtime C++ API
#include <onnxruntime_cxx_api.h>

// FFTW3 (double-precision — use fftwf_ for single-precision float)
#include <fftw3.h>

#include "constants.h"
#include "ring_buffer.h"

namespace mir {

/**
 * InferenceEngine — Core DSP + ONNX Inference Thread
 *
 * Drives the complete drum-isolation processing pipeline on a dedicated thread:
 *
 *   1. Accumulate kChunkSamples raw PCM floats from the SPSC input ring buffer.
 *   2. Slide a Hann-windowed analysis frame of kNfft samples with kHopLength
 *      stride across the chunk (kFrameCount iterations).
 *   3. Per frame: FFTW3 forward r2c FFT → extract magnitude + cache phase.
 *   4. Stack magnitudes into a [1, 1, kFreqBins, kFrameCount] ONNX input tensor.
 *   5. Ort::Session::Run() → soft_mask [1, 1, kFreqBins, kFrameCount] in [0,1].
 *   6. masked_mag = soft_mask × magnitude (element-wise).
 *   7. Per frame: reconstruct complex = masked_mag × exp(i × phase).
 *   8. FFTW3 inverse c2r IFFT + normalise → time-domain frame.
 *   9. Overlap-add synthesis with Hann window → accumulate into output buffer.
 *  10. Push kHopLength-sample hops to the SPSC output ring buffer.
 *
 * Lifecycle:
 *   Construct → call run() on a std::thread → stop() when done.
 */
class InferenceEngine {
public:
    /**
     * @param model_path  Path to the drums_unet.onnx file.
     * @param input_rb    SPSC input ring buffer populated by AudioReader.
     * @param output_rb   SPSC output ring buffer consumed by PortAudio callback.
     * @param eof_flag    Atomic flag set by AudioReader on end-of-file.
     */
    InferenceEngine(
        const std::string&                      model_path,
        RingBuffer<float, kInputRingCapacity>&  input_rb,
        RingBuffer<float, kOutputRingCapacity>& output_rb,
        const std::atomic<bool>&                eof_flag
    );

    ~InferenceEngine();

    InferenceEngine(const InferenceEngine&)            = delete;
    InferenceEngine& operator=(const InferenceEngine&) = delete;

    /**
     * Main entry point — call from a dedicated thread.
     * Runs until stop() is called or eof_flag becomes true and the input
     * ring buffer is drained.
     */
    void run();

    /** Signal the processing thread to exit cleanly. */
    void stop() noexcept { stop_requested_.store(true, std::memory_order_release); }

private:
    // ------------------------------------------------------------------
    // ONNX Runtime state
    // ------------------------------------------------------------------
    Ort::Env              ort_env_;
    Ort::SessionOptions   session_opts_;
    Ort::Session          session_;
    Ort::AllocatorWithDefaultOptions allocator_;

    // ------------------------------------------------------------------
    // FFTW3 plans and aligned buffers
    // ------------------------------------------------------------------
    float*            fftw_in_{nullptr};          // kNfft real input samples
    fftwf_complex*    fftw_out_{nullptr};          // kFreqBins complex output
    fftwf_plan        fwd_plan_{nullptr};          // r2c forward plan

    fftwf_complex*    ifft_in_{nullptr};           // kFreqBins complex input
    float*            ifft_out_{nullptr};           // kNfft real output
    fftwf_plan        inv_plan_{nullptr};           // c2r inverse plan

    // ------------------------------------------------------------------
    // Pre-computed windows
    // ------------------------------------------------------------------
    std::vector<float> hann_window_;               // analysis + synthesis Hann window

    // ------------------------------------------------------------------
    // Spectrogram buffers for one batch
    // ------------------------------------------------------------------
    // magnitude[frame * kFreqBins + bin]
    std::vector<float> magnitude_batch_;           // [kFrameCount × kFreqBins]
    // phase[frame * kFreqBins + bin]
    std::vector<float> phase_batch_;               // [kFrameCount × kFreqBins]

    // ------------------------------------------------------------------
    // Overlap-add synthesis state
    // ------------------------------------------------------------------
    std::vector<float> ola_buffer_;                // output accumulator, 2×kNfft
    int                ola_write_pos_{0};           // current write position in ola_buffer_

    // ------------------------------------------------------------------
    // Shared state
    // ------------------------------------------------------------------
    RingBuffer<float, kInputRingCapacity>&  input_rb_;
    RingBuffer<float, kOutputRingCapacity>& output_rb_;
    const std::atomic<bool>&               eof_flag_;
    std::atomic<bool>                      stop_requested_{false};

    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    /// Build the kNfft-point Hann analysis/synthesis window.
    void build_hann_window();

    /**
     * Run one full processing cycle:
     *   - Pops kChunkSamples from input_rb_
     *   - Fills magnitude_batch_ and phase_batch_
     *   - Runs ONNX inference
     *   - Runs IFFT + overlap-add
     *   - Pushes processed samples to output_rb_
     * Returns false if not enough input data is available yet.
     */
    bool process_chunk(const std::vector<float>& pcm_chunk);

    /// Forward FFT of one windowed frame into magnitude_batch_ and phase_batch_.
    void run_forward_fft(const float* windowed_frame, int frame_idx);

    /// ONNX Runtime inference: fills mask_data from magnitude_batch_.
    void run_onnx(std::vector<float>& mask_out);

    /// Inverse FFT of one reconstructed complex frame + overlap-add.
    void run_inverse_fft_and_ola(const float* mask_data, int frame_idx);

    /// Push one completed kHopLength-sample hop from ola_buffer_ to output_rb_.
    void flush_hop_to_output();
};

} // namespace mir
