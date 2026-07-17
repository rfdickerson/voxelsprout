#pragma once

#include "ui/ui_types.h"

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

// Bitmap-alpha font: rasterizes a TTF into an R8 atlas (via stb_truetype in
// font.cc) and exposes per-glyph metrics + measurement. The atlas pixels are
// uploaded by the renderer as the kUiFontAtlas texture. The metric/measure logic
// is independent of stb so it can be unit-tested with a synthetic font.
//
// Two atlas encodings are supported, selected per-font at load time:
//   - Coverage (default): stbtt_PackFontRanges bakes plain glyph coverage
//     (0=empty, 255=fully covered), crisp only near the baked pixel height.
//     Rendered via UiDrawMode::GlyphAlpha with a gamma-based stem-darkening
//     hack in the fragment shader.
//   - SDF (opt-in via the `sdf` load parameter): each glyph is baked
//     individually with stbtt_GetGlyphSDF into a signed-distance bitmap, then
//     packed into the shared atlas by hand via stb_rect_pack (stbtt_PackFontRanges
//     only produces coverage output, so it can't be reused here). One atlas then
//     stays crisp at any render size; the fragment shader thresholds the
//     recovered signed distance at the zero crossing with an fwidth-based AA
//     band, the same idiom used for UiDrawMode::RoundRect. Rendered via
//     UiDrawMode::GlyphSdf. isSdf()/sdfDistScale()/sdfDistBias() give the
//     UiDrawList the per-font constants needed to recover a signed pixel
//     distance from a normalized R8 sample: distancePx = sdfDistBias() -
//     sample * sdfDistScale() (negative = inside the glyph, matching the sign
//     convention sdRoundBox uses in ui.frag.slang).
namespace odai::ui {

struct Glyph {
    UiVec2 size{};     // Glyph bitmap size in pixels.
    UiVec2 bearing{};  // x: left bearing; y: top bearing above the baseline.
    float advance = 0.0f;
    UiRect uv{};       // Atlas UV (0..1); zero-area for whitespace.
};

// A single glyph after OpenType shaping (GSUB ligature substitution + GPOS kerning).
struct ShapedGlyph {
    std::uint32_t codepoint = 0;
    float kern = 0.0f; // Extra x-advance to apply after this glyph (GPOS kern delta).
};

class Font {
public:
    Font();
    ~Font();
    Font(Font&&) noexcept;
    Font& operator=(Font&&) noexcept;
    Font(const Font&) = delete;
    Font& operator=(const Font&) = delete;

    // Rasterize all glyphs in [firstCodepoint, lastCodepoint] at the given pixel
    // height into a square atlas. Returns false on failure (atlas too small, bad
    // font data). Implemented in font.cc with stb_truetype.
    //
    // sdf=false (default) bakes the coverage atlas described above via
    // stbtt_PackFontRanges, including ligature (liga) and en/em-dash glyphs.
    // sdf=true bakes a per-glyph signed-distance atlas instead; this path
    // currently covers only the main [firstCodepoint, lastCodepoint] range
    // (no GSUB ligature substitution glyphs yet) but keeps full GPOS/legacy
    // kerning support.
    bool loadFromMemory(const std::uint8_t* ttfData, std::size_t ttfSize, float pixelHeight,
                        std::uint32_t atlasSize = 1024, std::uint32_t firstCodepoint = 32,
                        std::uint32_t lastCodepoint = 255, bool sdf = false);
    bool loadFromFile(const std::string& path, float pixelHeight, std::uint32_t atlasSize = 1024,
                      std::uint32_t firstCodepoint = 32, std::uint32_t lastCodepoint = 255,
                      bool sdf = false);

    [[nodiscard]] const Glyph& glyph(std::uint32_t codepoint) const;
    [[nodiscard]] float measureText(std::string_view utf8) const;

    // Shape a UTF-8 string: applies GSUB ligature substitution then GPOS kerning.
    // Returns one ShapedGlyph per output glyph (ligatures reduce the count).
    [[nodiscard]] std::vector<ShapedGlyph> shape(std::string_view utf8) const;

    [[nodiscard]] float ascentPx() const { return m_ascent; }
    [[nodiscard]] float descentPx() const { return m_descent; }     // Positive, below baseline.
    [[nodiscard]] float lineHeightPx() const { return m_lineHeight; }

    [[nodiscard]] const std::vector<std::uint8_t>& atlasPixels() const { return m_atlas; }
    [[nodiscard]] std::uint32_t atlasWidth() const { return m_atlasWidth; }
    [[nodiscard]] std::uint32_t atlasHeight() const { return m_atlasHeight; }
    [[nodiscard]] bool valid() const { return !m_glyphs.empty(); }

    // True when this font was baked as a signed-distance atlas (loadFrom* called
    // with sdf=true). UiDrawList::addText uses this to pick UiDrawMode::GlyphSdf
    // over UiDrawMode::GlyphAlpha and to fill in the sdfDistScale/Bias params.
    [[nodiscard]] bool isSdf() const { return m_isSdf; }
    // Recovers a signed pixel distance from a normalized (0..1) R8 atlas sample:
    // distancePx = sdfDistBias() - sample * sdfDistScale(); negative = inside the
    // glyph, matching sdRoundBox's sign convention in ui.frag.slang. Meaningless
    // (zero) unless isSdf() is true.
    [[nodiscard]] float sdfDistScale() const { return m_sdfDistScale; }
    [[nodiscard]] float sdfDistBias() const { return m_sdfDistBias; }

    // Renderer texture id this font's atlas was uploaded to. addText tags glyph
    // quads with it so multiple fonts (regular/bold/italic) can coexist, each
    // bound to its own atlas. Defaults to the built-in single-font atlas.
    void setTextureId(UiTextureId id) { m_textureId = id; }
    [[nodiscard]] UiTextureId textureId() const { return m_textureId; }

    // Build a synthetic fixed-advance font (no atlas) for tests and headless use.
    void initSyntheticMonospace(float advance, float ascent, float descent,
                                std::uint32_t firstCodepoint = 32, std::uint32_t lastCodepoint = 126);
    void setGlyph(std::uint32_t codepoint, const Glyph& glyph);

    // Forward declaration for GPOS/GSUB shaping state (defined in font.cc).
    struct ShapingData;

private:
    // SDF bake path: per-glyph stbtt_GetGlyphSDF + hand-rolled stb_rect_pack
    // packing (stbtt_PackFontRanges only emits coverage bitmaps, so it can't be
    // reused here). Populates m_atlas/m_glyphs/m_shaping like loadFromMemory's
    // coverage path, plus m_isSdf/m_sdfDistScale/m_sdfDistBias. No GSUB ligature
    // glyphs yet (see class comment).
    bool loadSdfFromMemory(const std::uint8_t* ttfData, std::size_t ttfSize, float pixelHeight,
                           std::uint32_t atlasSize, std::uint32_t firstCodepoint,
                           std::uint32_t lastCodepoint);

    // Font cache: serialises the baked atlas + glyph metrics + shaping tables to a
    // binary file alongside the source TTF. Subsequent loadFromFile calls restore
    // from cache instead of re-running stb_truetype, cutting first-frame stall by
    // ~50-70 ms per font. Cache is invalidated when the TTF's size or mtime changes.
    // `sdf` is folded into the cache filename (and re-checked against the stored
    // flag) so a coverage bake and an SDF bake of the same font/size never share
    // a cache file.
    static std::string makeCachePath(const std::string& ttfPath, float pixelHeight,
                                     std::uint32_t atlasSize, std::uint32_t firstCodepoint,
                                     std::uint32_t lastCodepoint, bool sdf);
    bool saveCache(const std::string& cachePath, const std::string& ttfPath) const;
    bool loadCache(const std::string& cachePath, const std::string& ttfPath,
                   float pixelHeight, std::uint32_t atlasSize,
                   std::uint32_t firstCodepoint, std::uint32_t lastCodepoint, bool sdf);
    // Refresh the printable-ASCII fast-path cache from m_glyphs. Called after any
    // bulk change to the glyph set.
    void rebuildAsciiCache();

    static constexpr std::uint32_t kAsciiFirst = 32;   // First printable ASCII.
    static constexpr std::size_t kAsciiCount = 96;     // 32..127 inclusive of slots.

    std::unordered_map<std::uint32_t, Glyph> m_glyphs;
    // Direct-indexed fast path for printable ASCII: avoids a hash lookup per
    // character in measureText/addText (the hot text-layout path).
    std::array<Glyph, kAsciiCount> m_ascii{};
    std::array<bool, kAsciiCount> m_asciiPresent{};
    Glyph m_missing{};
    float m_ascent = 0.0f;
    float m_descent = 0.0f;
    float m_lineHeight = 0.0f;
    std::vector<std::uint8_t> m_atlas;  // R8: coverage, or signed distance when m_isSdf.
    std::uint32_t m_atlasWidth = 0;
    std::uint32_t m_atlasHeight = 0;
    UiTextureId m_textureId = kUiFontAtlas;

    bool m_isSdf = false;
    // distancePx = m_sdfDistBias - sample * m_sdfDistScale. Zero when !m_isSdf.
    float m_sdfDistScale = 0.0f;
    float m_sdfDistBias = 0.0f;

    std::unique_ptr<ShapingData> m_shaping;
};

// A family of style variants. Rich text selects a variant per run from the
// <b>/<i> markup flags, falling back to regular when a variant is absent.
struct FontSet {
    const Font* regular = nullptr;
    const Font* bold = nullptr;
    const Font* italic = nullptr;
    const Font* boldItalic = nullptr;
    // Used explicitly by <num> markup for resource totals, turns, and other
    // values that benefit from fixed-width digits.
    const Font* numeric = nullptr;

    [[nodiscard]] const Font* select(bool wantBold, bool wantItalic) const {
        if (wantBold && wantItalic && boldItalic != nullptr) return boldItalic;
        if (wantBold && wantItalic && bold != nullptr) return bold;
        if (wantBold && bold != nullptr) return bold;
        if (wantItalic && italic != nullptr) return italic;
        return regular;
    }
};

}  // namespace odai::ui
