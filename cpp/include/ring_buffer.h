#pragma once

#include <atomic>
#include <array>
#include <cstddef>
#include <cstring>
#include <cassert>

namespace mir {

/**
 * RingBuffer<T, Capacity> — Lock-Free Single-Producer / Single-Consumer (SPSC)
 * circular buffer.
 *
 * Design guarantees:
 *   - Exactly ONE producer thread calls push(); exactly ONE consumer calls pop().
 *   - No mutex, no spin-lock: head_ and tail_ are std::atomic<std::size_t> with
 *     carefully chosen acquire/release memory orderings.
 *   - Capacity must be a power of two so the modulo operation reduces to a
 *     cheap bitwise AND (idx & kMask).
 *   - push() and pop() operate on contiguous blocks of N elements at once,
 *     which maps naturally to audio chunk processing.
 *
 * Template parameters:
 *   T        — Element type (e.g. float for PCM samples).
 *   Capacity — Buffer size in elements; MUST be a power of two.
 *
 * Usage:
 *   RingBuffer<float, 524288> buf;
 *
 *   // Producer thread
 *   buf.push(src_ptr, n_samples);   // returns false if not enough free space
 *
 *   // Consumer thread
 *   buf.pop(dst_ptr, n_samples);    // returns false if not enough data available
 */
template<typename T, std::size_t Capacity>
class RingBuffer {
    static_assert((Capacity & (Capacity - 1)) == 0,
                  "RingBuffer: Capacity must be a power of two.");

public:
    RingBuffer() : head_(0), tail_(0) {}

    // Non-copyable, non-movable — the buffer is meant to be a global/static resource.
    RingBuffer(const RingBuffer&)            = delete;
    RingBuffer& operator=(const RingBuffer&) = delete;

    // -----------------------------------------------------------------------
    // Producer interface (call from exactly one thread)
    // -----------------------------------------------------------------------

    /**
     * Attempt to write `count` elements from `src` into the buffer.
     *
     * Returns true on success.
     * Returns false (and writes nothing) if fewer than `count` free slots exist.
     *
     * Memory ordering:
     *   - tail_ is loaded with acquire to get the latest consumer position.
     *   - head_ is stored with release so the consumer sees the written data
     *     before it observes the updated head.
     */
    [[nodiscard]] bool push(const T* src, std::size_t count) noexcept {
        const std::size_t head = head_.load(std::memory_order_relaxed);
        const std::size_t tail = tail_.load(std::memory_order_acquire);

        if (available_write(head, tail) < count)
            return false;

        // Write in up to two segments to handle wrap-around.
        const std::size_t first_chunk = std::min(count, Capacity - (head & kMask));
        std::memcpy(&data_[head & kMask], src,               first_chunk * sizeof(T));
        std::memcpy(&data_[0],            src + first_chunk, (count - first_chunk) * sizeof(T));

        head_.store(head + count, std::memory_order_release);
        return true;
    }

    // -----------------------------------------------------------------------
    // Consumer interface (call from exactly one thread)
    // -----------------------------------------------------------------------

    /**
     * Attempt to read `count` elements from the buffer into `dst`.
     *
     * Returns true on success.
     * Returns false (and reads nothing) if fewer than `count` elements are available.
     *
     * Memory ordering:
     *   - head_ is loaded with acquire to observe the latest producer position.
     *   - tail_ is stored with release so the producer can reclaim the freed space.
     */
    [[nodiscard]] bool pop(T* dst, std::size_t count) noexcept {
        const std::size_t tail = tail_.load(std::memory_order_relaxed);
        const std::size_t head = head_.load(std::memory_order_acquire);

        if (available_read(head, tail) < count)
            return false;

        // Read in up to two segments to handle wrap-around.
        const std::size_t first_chunk = std::min(count, Capacity - (tail & kMask));
        std::memcpy(dst,               &data_[tail & kMask], first_chunk * sizeof(T));
        std::memcpy(dst + first_chunk, &data_[0],            (count - first_chunk) * sizeof(T));

        tail_.store(tail + count, std::memory_order_release);
        return true;
    }

    // -----------------------------------------------------------------------
    // Query helpers (approximate — snapshot only; not atomic across both indices)
    // -----------------------------------------------------------------------

    /// Number of elements currently available to read.
    [[nodiscard]] std::size_t size_approx() const noexcept {
        const std::size_t head = head_.load(std::memory_order_acquire);
        const std::size_t tail = tail_.load(std::memory_order_acquire);
        return available_read(head, tail);
    }

    /// Number of free slots available for writing.
    [[nodiscard]] std::size_t free_approx() const noexcept {
        return Capacity - size_approx();
    }

    /// True if the buffer contains no readable elements.
    [[nodiscard]] bool empty() const noexcept {
        return size_approx() == 0;
    }

    static constexpr std::size_t capacity() noexcept { return Capacity; }

private:
    static constexpr std::size_t kMask = Capacity - 1;

    [[nodiscard]] static std::size_t available_read(std::size_t head, std::size_t tail) noexcept {
        return head - tail;
    }

    [[nodiscard]] static std::size_t available_write(std::size_t head, std::size_t tail) noexcept {
        return Capacity - (head - tail);
    }

    // Cache-line padding prevents false sharing between producer and consumer.
    alignas(64) std::atomic<std::size_t> head_;
    alignas(64) std::atomic<std::size_t> tail_;
    alignas(64) std::array<T, Capacity>  data_;
};

} // namespace mir
