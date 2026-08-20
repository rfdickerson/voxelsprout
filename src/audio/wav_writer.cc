#include "audio/wav_writer.h"

#include "core/log.h"

#include <array>
#include <cstring>
#include <limits>

namespace odai::audio {
namespace {

void writeU16(std::uint8_t* dst, std::uint16_t value) {
    dst[0] = static_cast<std::uint8_t>(value & 0xffu);
    dst[1] = static_cast<std::uint8_t>((value >> 8u) & 0xffu);
}

void writeU32(std::uint8_t* dst, std::uint32_t value) {
    for (int i = 0; i < 4; ++i) {
        dst[i] = static_cast<std::uint8_t>((value >> (i * 8)) & 0xffu);
    }
}

}  // namespace

WavWriter::~WavWriter() {
    close();
}

bool WavWriter::open(const std::filesystem::path& path, std::uint32_t sampleRate,
                     std::uint16_t channels) {
    close();
    if (sampleRate == 0u || channels == 0u || channels > 2u) {
        VOX_LOGE("audio") << "WAV capture: invalid format " << sampleRate << " Hz / "
                          << channels << " channels";
        return false;
    }
    m_file = std::fopen(path.string().c_str(), "wb");
    if (m_file == nullptr) {
        VOX_LOGE("audio") << "WAV capture: cannot open " << path.string();
        return false;
    }

    std::array<std::uint8_t, 44> header{};
    std::memcpy(header.data(), "RIFF", 4u);
    std::memcpy(header.data() + 8u, "WAVEfmt ", 8u);
    writeU32(header.data() + 16u, 16u);
    writeU16(header.data() + 20u, 3u);  // WAVE_FORMAT_IEEE_FLOAT
    writeU16(header.data() + 22u, channels);
    writeU32(header.data() + 24u, sampleRate);
    writeU32(header.data() + 28u, sampleRate * channels * sizeof(float));
    writeU16(header.data() + 32u, static_cast<std::uint16_t>(channels * sizeof(float)));
    writeU16(header.data() + 34u, 32u);
    std::memcpy(header.data() + 36u, "data", 4u);
    if (std::fwrite(header.data(), 1u, header.size(), m_file) != header.size()) {
        VOX_LOGE("audio") << "WAV capture: header write failed for " << path.string();
        std::fclose(m_file);
        m_file = nullptr;
        return false;
    }
    m_path = path;
    m_channels = channels;
    m_samplesWritten = 0u;
    return true;
}

bool WavWriter::write(std::span<const float> samples) {
    if (m_file == nullptr || samples.size() % m_channels != 0u) {
        return false;
    }
    if (samples.empty()) {
        return true;
    }
    const std::size_t written =
        std::fwrite(samples.data(), sizeof(float), samples.size(), m_file);
    if (written != samples.size()) {
        VOX_LOGE("audio") << "WAV capture: short write for " << m_path.string();
        return false;
    }
    m_samplesWritten += samples.size();
    return true;
}

bool WavWriter::close() {
    if (m_file == nullptr) {
        return true;
    }
    const std::uint64_t dataBytes64 = m_samplesWritten * sizeof(float);
    if (dataBytes64 > std::numeric_limits<std::uint32_t>::max() - 36u) {
        VOX_LOGE("audio") << "WAV capture exceeds RIFF 32-bit size limit: " << m_path.string();
        std::fclose(m_file);
        m_file = nullptr;
        return false;
    }
    const auto dataBytes = static_cast<std::uint32_t>(dataBytes64);
    std::array<std::uint8_t, 4> word{};
    bool ok = true;
    writeU32(word.data(), 36u + dataBytes);
    ok = ok && std::fseek(m_file, 4L, SEEK_SET) == 0;
    ok = ok && std::fwrite(word.data(), 1u, word.size(), m_file) == word.size();
    writeU32(word.data(), dataBytes);
    ok = ok && std::fseek(m_file, 40L, SEEK_SET) == 0;
    ok = ok && std::fwrite(word.data(), 1u, word.size(), m_file) == word.size();
    ok = ok && std::fclose(m_file) == 0;
    m_file = nullptr;
    if (!ok) {
        VOX_LOGE("audio") << "WAV capture: failed to finalize " << m_path.string();
    } else {
        VOX_LOGI("audio") << "WAV capture written: " << m_path.string() << " ("
                          << framesWritten() << " frames)";
    }
    return ok;
}

}  // namespace odai::audio
