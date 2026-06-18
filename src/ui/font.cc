#include "ui/font.h"

#include "ui/ui_text_util.h"

#include <fstream>
#include <ios>

#define STB_RECT_PACK_IMPLEMENTATION
#include <stb_rect_pack.h>
#define STB_TRUETYPE_IMPLEMENTATION
#include <stb_truetype.h>

namespace odai::ui {

const Glyph& Font::glyph(std::uint32_t codepoint) const {
    const auto it = m_glyphs.find(codepoint);
    return it != m_glyphs.end() ? it->second : m_missing;
}

float Font::measureText(std::string_view utf8) const {
    float width = 0.0f;
    std::size_t i = 0;
    while (i < utf8.size()) {
        const std::uint32_t cp = decodeUtf8(utf8, i);
        width += glyph(cp).advance;
    }
    return width;
}

void Font::setGlyph(std::uint32_t codepoint, const Glyph& glyph) {
    m_glyphs[codepoint] = glyph;
}

void Font::initSyntheticMonospace(float advance, float ascent, float descent,
                                  std::uint32_t firstCodepoint, std::uint32_t lastCodepoint) {
    m_glyphs.clear();
    m_atlas.clear();
    m_atlasWidth = 0;
    m_atlasHeight = 0;
    m_ascent = ascent;
    m_descent = descent;
    m_lineHeight = ascent + descent;
    for (std::uint32_t cp = firstCodepoint; cp <= lastCodepoint; ++cp) {
        Glyph g{};
        g.advance = advance;
        if (cp != ' ') {
            g.size = {advance * 0.6f, ascent * 0.7f};
            g.bearing = {advance * 0.1f, ascent * 0.7f};
        }
        m_glyphs[cp] = g;
    }
    m_missing = Glyph{};
    m_missing.advance = advance;
}

bool Font::loadFromMemory(const std::uint8_t* ttfData, std::size_t ttfSize, float pixelHeight,
                          std::uint32_t atlasSize, std::uint32_t firstCodepoint,
                          std::uint32_t lastCodepoint) {
    if (ttfData == nullptr || ttfSize == 0 || lastCodepoint < firstCodepoint || atlasSize == 0) {
        return false;
    }

    stbtt_fontinfo info{};
    const int fontOffset = stbtt_GetFontOffsetForIndex(ttfData, 0);
    if (fontOffset < 0 || stbtt_InitFont(&info, ttfData, fontOffset) == 0) {
        return false;
    }

    const std::uint32_t glyphCount = (lastCodepoint - firstCodepoint) + 1u;
    m_atlas.assign(static_cast<std::size_t>(atlasSize) * static_cast<std::size_t>(atlasSize), 0u);
    m_atlasWidth = atlasSize;
    m_atlasHeight = atlasSize;

    stbtt_pack_context packContext{};
    if (stbtt_PackBegin(&packContext, m_atlas.data(), static_cast<int>(atlasSize),
                        static_cast<int>(atlasSize), 0, 1, nullptr) == 0) {
        return false;
    }
    stbtt_PackSetOversampling(&packContext, 1, 1);

    std::vector<stbtt_packedchar> packed(glyphCount);
    const int packOk = stbtt_PackFontRange(
        &packContext, ttfData, 0, pixelHeight, static_cast<int>(firstCodepoint),
        static_cast<int>(glyphCount), packed.data());
    stbtt_PackEnd(&packContext);
    if (packOk == 0) {
        m_atlas.clear();
        m_atlasWidth = 0;
        m_atlasHeight = 0;
        return false;
    }

    int ascent = 0;
    int descent = 0;
    int lineGap = 0;
    stbtt_GetFontVMetrics(&info, &ascent, &descent, &lineGap);
    const float scale = stbtt_ScaleForPixelHeight(&info, pixelHeight);
    m_ascent = static_cast<float>(ascent) * scale;
    m_descent = static_cast<float>(-descent) * scale;
    m_lineHeight = static_cast<float>(ascent - descent + lineGap) * scale;

    const float invAtlas = 1.0f / static_cast<float>(atlasSize);
    m_glyphs.clear();
    for (std::uint32_t i = 0; i < glyphCount; ++i) {
        const stbtt_packedchar& pc = packed[i];
        Glyph g{};
        g.size = {static_cast<float>(pc.x1 - pc.x0), static_cast<float>(pc.y1 - pc.y0)};
        g.bearing = {pc.xoff, -pc.yoff};
        g.advance = pc.xadvance;
        g.uv = UiRect{static_cast<float>(pc.x0) * invAtlas, static_cast<float>(pc.y0) * invAtlas,
                      static_cast<float>(pc.x1) * invAtlas, static_cast<float>(pc.y1) * invAtlas};
        m_glyphs[firstCodepoint + i] = g;
    }

    const auto spaceIt = m_glyphs.find(static_cast<std::uint32_t>(' '));
    m_missing = Glyph{};
    m_missing.advance = (spaceIt != m_glyphs.end()) ? spaceIt->second.advance : (pixelHeight * 0.4f);
    return true;
}

bool Font::loadFromFile(const std::string& path, float pixelHeight, std::uint32_t atlasSize,
                        std::uint32_t firstCodepoint, std::uint32_t lastCodepoint) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        return false;
    }
    const std::streamoff size = file.tellg();
    if (size <= 0) {
        return false;
    }
    file.seekg(0, std::ios::beg);
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
    if (!file.read(reinterpret_cast<char*>(bytes.data()), size)) {
        return false;
    }
    return loadFromMemory(bytes.data(), bytes.size(), pixelHeight, atlasSize, firstCodepoint, lastCodepoint);
}

}  // namespace odai::ui
