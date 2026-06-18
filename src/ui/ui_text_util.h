#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace odai::ui {

// Decode one UTF-8 codepoint from `text` starting at `i`, advancing `i` past it.
// Returns U+FFFD on malformed input.
inline std::uint32_t decodeUtf8(std::string_view text, std::size_t& i) {
    const auto byteAt = [&](std::size_t k) -> std::uint32_t {
        return static_cast<std::uint32_t>(static_cast<unsigned char>(text[k]));
    };
    const std::uint32_t b0 = byteAt(i);
    if (b0 < 0x80u) {
        i += 1;
        return b0;
    }
    if ((b0 & 0xE0u) == 0xC0u && i + 1 < text.size()) {
        const std::uint32_t cp = ((b0 & 0x1Fu) << 6) | (byteAt(i + 1) & 0x3Fu);
        i += 2;
        return cp;
    }
    if ((b0 & 0xF0u) == 0xE0u && i + 2 < text.size()) {
        const std::uint32_t cp = ((b0 & 0x0Fu) << 12) | ((byteAt(i + 1) & 0x3Fu) << 6) | (byteAt(i + 2) & 0x3Fu);
        i += 3;
        return cp;
    }
    if ((b0 & 0xF8u) == 0xF0u && i + 3 < text.size()) {
        const std::uint32_t cp = ((b0 & 0x07u) << 18) | ((byteAt(i + 1) & 0x3Fu) << 12) |
                                 ((byteAt(i + 2) & 0x3Fu) << 6) | (byteAt(i + 3) & 0x3Fu);
        i += 4;
        return cp;
    }
    i += 1;
    return 0xFFFDu;
}

}  // namespace odai::ui
