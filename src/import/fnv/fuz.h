#pragma once
#include <cstdint>
#include <span>

namespace odai::importer::fnv {
// FUZE v1: magic, version, LIP byte count, LIP bytes, then an xWMA RIFF.
inline std::span<const std::uint8_t> fuzAudio(std::span<const std::uint8_t> bytes) {
    if (bytes.size() < 12 || bytes[0] != 'F' || bytes[1] != 'U' ||
        bytes[2] != 'Z' || bytes[3] != 'E') return {};
    const auto u32 = [&](std::size_t at) {
        return std::uint32_t(bytes[at]) | (std::uint32_t(bytes[at+1]) << 8) |
            (std::uint32_t(bytes[at+2]) << 16) | (std::uint32_t(bytes[at+3]) << 24);
    };
    if (u32(4) != 1 || u32(8) > bytes.size() - 12) return {};
    auto audio = bytes.subspan(12 + u32(8));
    if (audio.size() < 12 || audio[0] != 'R' || audio[1] != 'I' ||
        audio[2] != 'F' || audio[3] != 'F') return {};
    return audio;
}
} // namespace odai::importer::fnv
