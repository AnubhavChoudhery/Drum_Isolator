/**
 * main.cpp — Entry Point: PortAudio Setup, Thread Launch, Main Loop
 *
 * Thread architecture:
 *
 *   [file_reader_thread]  AudioReader::run()
 *       ↓ push kHopLength samples
 *   [g_input_rb]          SPSC ring buffer (raw PCM)
 *       ↓ pop kChunkSamples
 *   [inference_thread]    InferenceEngine::run()
 *       ↓ push kHopLength samples
 *   [g_output_rb]         SPSC ring buffer (processed drums)
 *       ↓ pop kHopLength samples
 *   [PortAudio callback]  paCallback()   — real-time audio thread
 *       ↓
 *   System speakers
 *
 * Usage:
 *   mir_engine <input.wav> <drums_unet.onnx>
 */

#include <atomic>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <csignal>

#include <portaudio.h>

#include "constants.h"
#include "ring_buffer.h"
#include "audio_reader.h"
#include "inference_engine.h"

// ---------------------------------------------------------------------------
// Global shared state
// ---------------------------------------------------------------------------

namespace {

/// Input ring buffer: raw PCM floats from the file reader thread.
mir::RingBuffer<float, mir::kInputRingCapacity>  g_input_rb;

/// Output ring buffer: processed drum samples from the inference thread.
mir::RingBuffer<float, mir::kOutputRingCapacity> g_output_rb;

/// Set to true by AudioReader when the input file is fully decoded.
std::atomic<bool> g_eof{false};

/// Set to true on SIGINT / SIGTERM to trigger a clean shutdown.
std::atomic<bool> g_shutdown{false};

// Pointers held for stop() calls from signal handler context
mir::AudioReader*    g_audio_reader_ptr    = nullptr;
mir::InferenceEngine* g_inference_engine_ptr = nullptr;

} // anonymous namespace

// ---------------------------------------------------------------------------
// Signal handler
// ---------------------------------------------------------------------------

static void on_signal(int /*sig*/) {
    g_shutdown.store(true, std::memory_order_release);
    if (g_audio_reader_ptr)     g_audio_reader_ptr->stop();
    if (g_inference_engine_ptr) g_inference_engine_ptr->stop();
}

// ---------------------------------------------------------------------------
// PortAudio callback (real-time audio thread)
//
// CONTRACT: Must not allocate memory, block on mutexes, or call syscalls.
// It pops exactly kHopLength float samples from g_output_rb and writes them
// to PortAudio's output buffer.  If the buffer is momentarily empty (e.g.,
// on startup before the inference thread has caught up), silence is output.
// ---------------------------------------------------------------------------

static int pa_callback(
    const void* /*input_buffer*/,
    void*        output_buffer,
    unsigned long frames_per_buffer,
    const PaStreamCallbackTimeInfo* /*time_info*/,
    PaStreamCallbackFlags /*status_flags*/,
    void* /*user_data*/
) {
    float* out = static_cast<float*>(output_buffer);

    if (!g_output_rb.pop(out, static_cast<std::size_t>(frames_per_buffer))) {
        // Underrun: output silence rather than undefined data
        for (unsigned long i = 0; i < frames_per_buffer; ++i)
            out[i] = 0.0f;
    }

    // Return paComplete when EOF and the output buffer has been drained
    if (g_eof.load(std::memory_order_acquire) && g_output_rb.empty())
        return paComplete;

    return paContinue;
}

// ---------------------------------------------------------------------------
// PortAudio helpers
// ---------------------------------------------------------------------------

static void pa_check(PaError err, const char* ctx) {
    if (err != paNoError) {
        throw std::runtime_error(
            std::string(ctx) + ": " + Pa_GetErrorText(err)
        );
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input.wav> <drums_unet.onnx>\n";
        return 1;
    }

    const std::string wav_path   = argv[1];
    const std::string model_path = argv[2];

    std::signal(SIGINT,  on_signal);
    std::signal(SIGTERM, on_signal);

    // ------------------------------------------------------------------
    // Construct processing objects
    // ------------------------------------------------------------------
    mir::AudioReader     audio_reader(wav_path,   g_input_rb,  g_eof);
    mir::InferenceEngine inference_engine(model_path, g_input_rb, g_output_rb, g_eof);

    g_audio_reader_ptr     = &audio_reader;
    g_inference_engine_ptr = &inference_engine;

    // ------------------------------------------------------------------
    // Launch background threads
    // ------------------------------------------------------------------
    std::thread file_reader_thread([&] {
        try {
            audio_reader.run();
        } catch (const std::exception& e) {
            std::cerr << "[AudioReader] Fatal: " << e.what() << "\n";
            g_shutdown.store(true, std::memory_order_release);
            inference_engine.stop();
        }
    });

    std::thread inference_thread([&] {
        try {
            inference_engine.run();
        } catch (const std::exception& e) {
            std::cerr << "[InferenceEngine] Fatal: " << e.what() << "\n";
            g_shutdown.store(true, std::memory_order_release);
        }
    });

    // ------------------------------------------------------------------
    // PortAudio initialisation
    // ------------------------------------------------------------------
    pa_check(Pa_Initialize(), "Pa_Initialize");

    PaStream* stream = nullptr;
    PaError   pa_err = Pa_OpenDefaultStream(
        &stream,
        0,                              // no input channels
        mir::kChannels,                 // mono output
        paFloat32,                      // 32-bit float samples
        mir::kSampleRate,
        static_cast<unsigned long>(mir::kHopLength),   // frames per callback buffer
        pa_callback,
        nullptr                         // user data (we use globals)
    );
    pa_check(pa_err, "Pa_OpenDefaultStream");
    pa_check(Pa_StartStream(stream), "Pa_StartStream");

    std::cout << "MIR Drum Isolator running.\n"
              << "  Input : " << wav_path   << "\n"
              << "  Model : " << model_path << "\n"
              << "Press Ctrl+C to stop.\n";

    // ------------------------------------------------------------------
    // Main wait loop — poll until playback finishes or user interrupts
    // ------------------------------------------------------------------
    while (!g_shutdown.load(std::memory_order_acquire)) {
        if (!Pa_IsStreamActive(stream)) {
            // PortAudio callback returned paComplete
            break;
        }
        Pa_Sleep(100);  // 100 ms poll interval
    }

    // ------------------------------------------------------------------
    // Clean shutdown
    // ------------------------------------------------------------------
    std::cout << "\nShutting down...\n";

    audio_reader.stop();
    inference_engine.stop();

    pa_check(Pa_StopStream(stream),  "Pa_StopStream");
    pa_check(Pa_CloseStream(stream), "Pa_CloseStream");
    pa_check(Pa_Terminate(),         "Pa_Terminate");

    if (file_reader_thread.joinable()) file_reader_thread.join();
    if (inference_thread.joinable())   inference_thread.join();

    std::cout << "Done.\n";
    return 0;
}
