#pragma once

#include "ui/font.h"
#include "ui/ui_draw_list.h"
#include "ui/ui_types.h"
#include "ui/vector/vector_icon_registry.h"

#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

namespace odai::ui {

// Callback used by UiTheme::loadFromFile to upload frame/icon textures without
// coupling the headless UI library to any specific renderer.
// pixels — RGBA8 data, w/h — dimensions, mipmapped — generate mip chain.
// Returns kUiNoTexture on failure.
using UiTextureUploadFn =
    std::function<UiTextureId(const std::uint8_t* pixels, std::uint32_t w, std::uint32_t h, bool mipmapped)>;

// A theme is a named set of design tokens loaded from a JSON file.
// It provides fonts, colors, frame styles, and numeric sizes to the
// document loader and widgets, so artists can swap entire visual themes
// without touching C++.
//
// JSON format:
// {
//   "name": "AncientParchment",
//   "fonts": {
//     "body":    { "path": "assets/fonts/EBGaramond-Regular.ttf", "size": 18 },
//     "bold":    { "path": "assets/fonts/EBGaramond-Bold.ttf",    "size": 18 },
//     "italic":  { "path": "assets/fonts/EBGaramond-Italic.ttf",  "size": 18 },
//     "heading": { "path": "assets/fonts/EBGaramond-Bold.ttf",    "size": 24 }
//   },
//   "colors": {
//     "text":         "#E8D9B0",
//     "text.dim":     "#A08060",
//     "text.positive":"#7EC850",
//     "text.negative":"#E05050",
//     "panel.bg":     "#2C1F0E",
//     "accent":       "#C8963A"
//   },
//   "frames": {
//     "Panel.Default": { "texture": "assets/ui/frames/parchment.png",
//                        "border": [12, 12, 12, 12] },
//     "Tooltip":       { "texture": "assets/ui/frames/tooltip.png",
//                        "border": [8,8,8,8] }
//   },
//   "sizes": {
//     "padding":    [8, 8],
//     "titleBarH":  30,
//     "gap":        6
//   }
// }
class UiTheme {
public:
    // Load from a JSON file. Fonts are baked immediately; frame/icon textures
    // are uploaded via the provided callback. Returns false on parse/IO failure.
    bool loadFromFile(const std::filesystem::path& path, const UiTextureUploadFn& upload);

    // Look up a color token (e.g. "text", "panel.bg"). Returns transparent black
    // if the key is not found.
    UiColor color(std::string_view key) const;

    // Look up a font by role key ("body", "bold", "italic", "heading").
    // Returns nullptr if the key is not found.
    const Font* font(std::string_view key) const;

    // Convenience: returns a FontSet with body/bold/italic/boldItalic from the
    // theme. Slots with no matching font fall back to the body font.
    FontSet bodyFontSet() const;

    // Look up a named 9-slice frame (e.g. "Panel.Default", "Tooltip").
    // Returns nullopt if not found.
    std::optional<UiNineSlice> frame(std::string_view key) const;

    // Look up a vector (SVG) icon declared in this theme's "svgIcons" section.
    // Returns nullptr if the key was not registered by this theme. The geometry
    // lives in VectorIconRegistry::global(); draw it via UiDrawList::addVectorIcon.
    const VectorIcon* vectorIcon(std::string_view key) const;

    // Look up a numeric size token (e.g. "titleBarH", "gap"). Returns 0 if not found.
    float size(std::string_view key) const;

    // Look up a vec2 size token (e.g. "padding"). Returns {0,0} if not found.
    UiVec2 sizeVec2(std::string_view key) const;

    bool loaded() const { return m_loaded; }
    const std::string& name() const { return m_name; }

private:
    bool m_loaded = false;
    std::string m_name;
    std::unordered_map<std::string, UiColor>              m_colors;
    std::unordered_map<std::string, Font>                 m_fonts;
    std::unordered_map<std::string, UiNineSlice>          m_frames;
    std::unordered_map<std::string, float>                m_sizes;
    std::unordered_map<std::string, UiVec2>               m_sizeVec2s;
    std::unordered_set<std::string>                       m_vectorIcons;
};

}  // namespace odai::ui
