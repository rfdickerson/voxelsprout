#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

// Core RingBuffer subsystem
// Responsible for: fixed-capacity FIFO sample windows for telemetry and
// bounded drop-oldest history vectors.
// Should NOT do: growth, allocation, or thread safety.
namespace odai::core {

// Fixed-capacity ring of samples. Once full, each push overwrites the oldest.
//
// data()/writeIndex()/orderedStartIndex() exist because the consumers hand the
// raw storage to APIs that take (values, count, offset) -- ImGui::PlotLines in
// particular -- so the ring cannot hide its layout entirely.
template <typename T, std::size_t Capacity>
class RingBuffer {
public:
    static_assert(Capacity > 0u, "RingBuffer needs a non-zero capacity");

    static constexpr std::size_t kCapacity = Capacity;

    void push(const T& value) {
        m_samples[m_writeIndex] = value;
        m_writeIndex = (m_writeIndex + 1u) % Capacity;
        if (m_sampleCount < Capacity) {
            ++m_sampleCount;
        }
    }

    // Zero-fills the storage as well as resetting the cursors, matching what
    // the timing-history reset did by hand (fill(0.0f) plus two assignments).
    void clear() {
        m_samples.fill(T{});
        m_writeIndex = 0u;
        m_sampleCount = 0u;
    }

    [[nodiscard]] std::size_t size() const noexcept { return m_sampleCount; }
    [[nodiscard]] bool empty() const noexcept { return m_sampleCount == 0u; }
    [[nodiscard]] bool full() const noexcept { return m_sampleCount == Capacity; }

    // index 0 is the oldest retained sample, size() - 1 the newest.
    [[nodiscard]] const T& operator[](std::size_t index) const {
        const std::size_t start = orderedStartIndex();
        return m_samples[(start + index) % Capacity];
    }

    [[nodiscard]] const T* data() const noexcept { return m_samples.data(); }
    [[nodiscard]] std::size_t writeIndex() const noexcept { return m_writeIndex; }

    // Where the oldest sample sits in the raw storage. Until the ring wraps the
    // samples are already in order from 0, which is why this is not simply
    // writeIndex().
    [[nodiscard]] std::size_t orderedStartIndex() const noexcept {
        return full() ? m_writeIndex : 0u;
    }

private:
    std::array<T, Capacity> m_samples{};
    std::size_t m_writeIndex = 0u;
    std::size_t m_sampleCount = 0u;
};

// Nearest-rank percentile over the live window, percentile01 in [0, 1].
// Deliberately reproduces the index arithmetic of the implementation this
// replaced, including the ceil-based rank, so the debug overlay's reported
// numbers do not shift.
template <std::size_t Capacity>
[[nodiscard]] float percentile(const RingBuffer<float, Capacity>& history, float percentile01) {
    const std::size_t sampleCount = history.size();
    if (sampleCount == 0u) {
        return 0.0f;
    }

    std::array<float, Capacity> scratch{};
    for (std::size_t i = 0; i < sampleCount; ++i) {
        scratch[i] = history[i];
    }

    const float clamped = std::clamp(percentile01, 0.0f, 1.0f);
    const std::size_t targetIndex = static_cast<std::size_t>(
        std::ceil(clamped * static_cast<float>(sampleCount - 1u))
    );
    auto first = scratch.begin();
    auto nth = first + static_cast<std::ptrdiff_t>(targetIndex);
    auto last = first + static_cast<std::ptrdiff_t>(sampleCount);
    std::nth_element(first, nth, last);
    return *nth;
}

// Drop-oldest push into a plain vector. Not a ring: the call sites hand out
// `const std::vector<T>&` to chart widgets and one of them compacts in place,
// so the storage has to stay contiguous and in order.
template <typename T>
void pushBounded(std::vector<T>& values, T value, std::size_t maxSize) {
    values.push_back(std::move(value));
    while (values.size() > maxSize) {
        values.erase(values.begin());
    }
}

}  // namespace odai::core
