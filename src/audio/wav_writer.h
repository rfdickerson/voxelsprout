#pragma once

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <span>

namespace odai::audio {

// Minimal IEEE-float WAV writer for deterministic offline capture. It owns no
// mixer state; callers hand it already-interleaved PCM from Audio.
class WavWriter {
public:
    WavWriter() = default;
    ~WavWriter();
    WavWriter(const WavWriter&) = delete;
    WavWriter& operator=(const WavWriter&) = delete;

    bool open(const std::filesystem::path& path, std::uint32_t sampleRate,
              std::uint16_t channels);
    bool write(std::span<const float> samples);
    bool close();

    [[nodiscard]] bool isOpen() const { return m_file != nullptr; }
    [[nodiscard]] std::uint64_t framesWritten() const {
        return m_channels > 0u ? m_samplesWritten / m_channels : 0u;
    }

private:
    std::FILE* m_file = nullptr;
    std::filesystem::path m_path;
    std::uint16_t m_channels = 0u;
    std::uint64_t m_samplesWritten = 0u;
};

}  // namespace odai::audio
