#pragma once

#include "ui/ui_types.h"

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

// Bitmap-alpha font: rasterizes a TTF into an R8 coverage atlas (via stb_truetype
// in font.cc) and exposes per-glyph metrics + measurement. The atlas pixels are
// uploaded by the renderer as the kUiFontAtlas texture. The metric/measure logic
// is independent of stb so it can be unit-tested with a synthetic font.
namespace odai::ui {

struct Glyph {
    UiVec2 size{};     // Glyph bitmap size in pixels.
    UiVec2 bearing{};  // x: left bearing; y: top bearing above the baseline.
    float advance = 0.0f;
    UiRect uv{};       // Atlas UV (0..1); zero-area for whitespace.
};

class Font {
public:
    Font() = default;

    // Rasterize all glyphs in [firstCodepoint, lastCodepoint] at the given pixel
    // height into a square atlas. Returns false on failure (atlas too small, bad
    // font data). Implemented in font.cc with stb_truetype.
    bool loadFromMemory(const std::uint8_t* ttfData, std::size_t ttfSize, float pixelHeight,
                        std::uint32_t atlasSize = 512, std::uint32_t firstCodepoint = 32,
                        std::uint32_t lastCodepoint = 126);
    bool loadFromFile(const std::string& path, float pixelHeight, std::uint32_t atlasSize = 512,
                      std::uint32_t firstCodepoint = 32, std::uint32_t lastCodepoint = 126);

    [[nodiscard]] const Glyph& glyph(std::uint32_t codepoint) const;
    [[nodiscard]] float measureText(std::string_view utf8) const;

    [[nodiscard]] float ascentPx() const { return m_ascent; }
    [[nodiscard]] float descentPx() const { return m_descent; }     // Positive, below baseline.
    [[nodiscard]] float lineHeightPx() const { return m_lineHeight; }

    [[nodiscard]] const std::vector<std::uint8_t>& atlasPixels() const { return m_atlas; }
    [[nodiscard]] std::uint32_t atlasWidth() const { return m_atlasWidth; }
    [[nodiscard]] std::uint32_t atlasHeight() const { return m_atlasHeight; }
    [[nodiscard]] bool valid() const { return !m_glyphs.empty(); }

    // Build a synthetic fixed-advance font (no atlas) for tests and headless use.
    void initSyntheticMonospace(float advance, float ascent, float descent,
                                std::uint32_t firstCodepoint = 32, std::uint32_t lastCodepoint = 126);
    void setGlyph(std::uint32_t codepoint, const Glyph& glyph);

private:
    std::unordered_map<std::uint32_t, Glyph> m_glyphs;
    Glyph m_missing{};
    float m_ascent = 0.0f;
    float m_descent = 0.0f;
    float m_lineHeight = 0.0f;
    std::vector<std::uint8_t> m_atlas;  // R8 coverage.
    std::uint32_t m_atlasWidth = 0;
    std::uint32_t m_atlasHeight = 0;
};

}  // namespace odai::ui
