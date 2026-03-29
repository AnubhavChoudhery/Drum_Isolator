#pragma once

#include <atomic>
#include <string>

#include "constants.h"
#include "ring_buffer.h"

namespace mir {

/**
 * AudioReader — File Reader Thread
 *
 * Opens a .wav file with libsndfile, reads it in kHopLength-sample chunks,
 * mono mix-down if needed, and pushes float32 PCM samples into the provided
 * input ring buffer.
 *
 * Lifecycle:
 *   1. Construct with the wav path and references to the shared ring buffer
 *      and the global EOF flag.
 *   2. Call run() on a dedicated std::thread.  The thread runs until all
 *      samples are consumed or stop() is called externally.
 *
 * Thread safety:
 *   - run() is the sole producer of g_input_rb; no other thread should push().
 *   - g_eof is written by this thread and read by the inference + main threads.
 */
class AudioReader {
public:
    /**
     * @param wav_path   Absolute or relative path to the input .wav file.
     * @param input_rb   Reference to the SPSC input ring buffer.
     * @param eof_flag   Atomic flag; set to true when the file is exhausted.
     */
    AudioReader(
        std::string              wav_path,
        RingBuffer<float, kInputRingCapacity>& input_rb,
        std::atomic<bool>&       eof_flag
    );

    ~AudioReader() = default;

    AudioReader(const AudioReader&)            = delete;
    AudioReader& operator=(const AudioReader&) = delete;

    /**
     * Main entry point — call from a dedicated thread.
     * Blocks until EOF or stop() is called.
     * Throws std::runtime_error if the file cannot be opened or its sample
     * rate does not match kSampleRate.
     */
    void run();

    /**
     * Signal the reader thread to stop early (e.g. on user interrupt).
     * The thread will exit cleanly after finishing its current chunk.
     */
    void stop() noexcept { stop_requested_.store(true, std::memory_order_release); }

private:
    std::string                           wav_path_;
    RingBuffer<float, kInputRingCapacity>& input_rb_;
    std::atomic<bool>&                    eof_flag_;
    std::atomic<bool>                     stop_requested_{false};
};

} // namespace mir
