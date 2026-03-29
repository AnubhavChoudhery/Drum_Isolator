/**
 * audio_reader.cpp — libsndfile-based WAV file reader thread.
 *
 * Reads a .wav file in kHopLength-sample chunks, mix-downs stereo to mono,
 * and pushes float32 PCM samples into the SPSC input ring buffer.
 *
 * The thread spins with a short sleep when the ring buffer is full, yielding
 * CPU to the inference thread without a mutex.
 */

#include "audio_reader.h"

#include <sndfile.h>
#include <stdexcept>
#include <string>
#include <thread>
#include <chrono>
#include <vector>
#include <cmath>

namespace mir {

AudioReader::AudioReader(
    std::string                            wav_path,
    RingBuffer<float, kInputRingCapacity>& input_rb,
    std::atomic<bool>&                     eof_flag
)
    : wav_path_(std::move(wav_path))
    , input_rb_(input_rb)
    , eof_flag_(eof_flag)
{}

void AudioReader::run() {
    // ------------------------------------------------------------------
    // Open file
    // ------------------------------------------------------------------
    SF_INFO sf_info{};
    SNDFILE* sndfile = sf_open(wav_path_.c_str(), SFM_READ, &sf_info);
    if (!sndfile) {
        throw std::runtime_error(
            std::string("AudioReader: Cannot open '") + wav_path_ +
            "': " + sf_strerror(nullptr)
        );
    }

    // Validate sample rate
    if (sf_info.samplerate != kSampleRate) {
        sf_close(sndfile);
        throw std::runtime_error(
            "AudioReader: File sample rate (" + std::to_string(sf_info.samplerate) +
            " Hz) does not match kSampleRate (" + std::to_string(kSampleRate) + " Hz). "
            "Resample the file before using this engine."
        );
    }

    const int file_channels = sf_info.channels;

    // ------------------------------------------------------------------
    // Interleaved read buffer: kHopLength frames × file_channels
    // ------------------------------------------------------------------
    const int interleaved_size = kHopLength * file_channels;
    std::vector<float> interleaved(interleaved_size);
    std::vector<float> mono_chunk(kHopLength);

    // ------------------------------------------------------------------
    // Read loop
    // ------------------------------------------------------------------
    while (!stop_requested_.load(std::memory_order_acquire)) {
        const sf_count_t frames_read = sf_readf_float(
            sndfile, interleaved.data(), static_cast<sf_count_t>(kHopLength)
        );

        if (frames_read == 0) {
            // End of file
            break;
        }

        // Mix down interleaved multi-channel samples to mono
        const float channel_scale = 1.0f / static_cast<float>(file_channels);
        for (sf_count_t f = 0; f < frames_read; ++f) {
            float sum = 0.0f;
            for (int c = 0; c < file_channels; ++c) {
                sum += interleaved[static_cast<std::size_t>(f * file_channels + c)];
            }
            mono_chunk[static_cast<std::size_t>(f)] = sum * channel_scale;
        }

        // If partial read (near EOF), zero-pad the rest
        if (frames_read < static_cast<sf_count_t>(kHopLength)) {
            for (sf_count_t f = frames_read; f < static_cast<sf_count_t>(kHopLength); ++f) {
                mono_chunk[static_cast<std::size_t>(f)] = 0.0f;
            }
        }

        // Spin-wait until there is room in the ring buffer, then push.
        // Sleep briefly to yield CPU rather than busy-spin.
        while (!input_rb_.push(mono_chunk.data(), static_cast<std::size_t>(kHopLength))) {
            if (stop_requested_.load(std::memory_order_acquire))
                goto cleanup;
            std::this_thread::sleep_for(std::chrono::microseconds(500));
        }
    }

cleanup:
    sf_close(sndfile);
    eof_flag_.store(true, std::memory_order_release);
}

} // namespace mir
