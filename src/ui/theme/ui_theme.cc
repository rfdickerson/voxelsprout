#include "ui/theme/ui_theme.h"

#include "core/log.h"
#include "ui/icon_atlas.h"

#include <nlohmann/json.hpp>

#include <fstream>

// stb_image for loading frame PNG textures.
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_ONLY_TGA
#define STBI_NO_FAILURE_STRINGS
#include <stb_image.h>

namespace odai::ui {

namespace {

UiColor parseHexColor(std::string_view hex) {
    if (!hex.empty() && hex.front() == '#') {
        hex.remove_prefix(1);
    }
    if (hex.size() == 6) {
        std::uint32_t v = 0;
        for (char c : hex) {
            std::uint32_t d = 0;
            if (c >= '0' && c <= '9') d = static_cast<std::uint32_t>(c - '0');
            else if (c >= 'a' && c <= 'f') d = static_cast<std::uint32_t>(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') d = static_cast<std::uint32_t>(c - 'A' + 10);
            else return UiColor{};
            v = (v << 4) | d;
        }
        return UiColor::fromRgbHex(v);
    }
    return UiColor{};
}

}  // namespace

bool UiTheme::loadFromFile(const std::filesystem::path& path, const UiTextureUploadFn& upload) {
    std::ifstream file(path);
    if (!file) {
        VOX_LOGE("ui") << "UiTheme: cannot open " << path.string() << "\n";
        return false;
    }

    nlohmann::json j;
    try {
        file >> j;
    } catch (const nlohmann::json::exception& e) {
        VOX_LOGE("ui") << "UiTheme: JSON parse error in " << path.string()
                       << ": " << e.what() << "\n";
        return false;
    }

    m_name = j.value("name", "Unnamed");
    const std::filesystem::path baseDir = path.parent_path();

    // --- Fonts ---
    if (j.contains("fonts") && j["fonts"].is_object()) {
        for (const auto& [key, val] : j["fonts"].items()) {
            if (!val.is_object()) continue;
            const std::string fontPath = val.value("path", "");
            const float fontSize = val.value("size", 18.0f);
            if (fontPath.empty()) continue;

            // Resolve relative to theme file's directory or project root.
            std::filesystem::path resolved = fontPath;
            if (resolved.is_relative()) {
                // Try relative to theme dir first, then as-is.
                const std::filesystem::path candidate = baseDir / resolved;
                if (std::filesystem::exists(candidate)) {
                    resolved = candidate;
                }
            }
            Font f;
            if (f.loadFromFile(resolved.string(), fontSize)) {
                m_fonts.emplace(key, std::move(f));
                VOX_LOGI("ui") << "UiTheme: loaded font '" << key << "' from "
                               << resolved.string() << " @ " << fontSize << "px\n";
            } else {
                VOX_LOGW("ui") << "UiTheme: failed to load font '" << key
                               << "' from " << resolved.string() << "\n";
            }
        }
    }

    // --- Colors ---
    if (j.contains("colors") && j["colors"].is_object()) {
        for (const auto& [key, val] : j["colors"].items()) {
            if (val.is_string()) {
                m_colors.emplace(key, parseHexColor(val.get<std::string>()));
            }
        }
    }

    // --- Frames (9-slice) ---
    if (j.contains("frames") && j["frames"].is_object()) {
        for (const auto& [key, val] : j["frames"].items()) {
            if (!val.is_object()) continue;
            const std::string texPath = val.value("texture", "");
            if (texPath.empty()) continue;
            const auto& border = val.value("border", std::vector<float>{8, 8, 8, 8});

            std::filesystem::path resolved = texPath;
            if (resolved.is_relative()) {
                const std::filesystem::path candidate = baseDir / resolved;
                if (std::filesystem::exists(candidate)) {
                    resolved = candidate;
                }
            }

            int w = 0, h = 0, ch = 0;
            stbi_uc* pixels = stbi_load(resolved.string().c_str(), &w, &h, &ch, 4);
            if (!pixels) {
                VOX_LOGW("ui") << "UiTheme: failed to load frame texture '" << key
                               << "' from " << resolved.string() << "\n";
                continue;
            }
            const UiTextureId texId = upload(
                pixels, static_cast<std::uint32_t>(w), static_cast<std::uint32_t>(h), false);
            stbi_image_free(pixels);
            if (texId == kUiNoTexture) {
                VOX_LOGW("ui") << "UiTheme: failed to register frame texture '" << key << "'\n";
                continue;
            }

            const float bL = border.size() > 0 ? border[0] : 8.0f;
            const float bT = border.size() > 1 ? border[1] : 8.0f;
            const float bR = border.size() > 2 ? border[2] : 8.0f;
            const float bB = border.size() > 3 ? border[3] : 8.0f;

            UiNineSlice ns;
            ns.textureId = texId;
            ns.uv = UiRect{0.0f, 0.0f, 1.0f, 1.0f};
            ns.borderLeftPx   = bL;
            ns.borderTopPx    = bT;
            ns.borderRightPx  = bR;
            ns.borderBottomPx = bB;
            ns.uvBorderLeft   = bL / static_cast<float>(w);
            ns.uvBorderTop    = bT / static_cast<float>(h);
            ns.uvBorderRight  = bR / static_cast<float>(w);
            ns.uvBorderBottom = bB / static_cast<float>(h);
            m_frames.emplace(key, ns);
            VOX_LOGI("ui") << "UiTheme: registered frame '" << key << "'\n";
        }
    }

    // --- Sizes ---
    if (j.contains("sizes") && j["sizes"].is_object()) {
        for (const auto& [key, val] : j["sizes"].items()) {
            if (val.is_number()) {
                m_sizes.emplace(key, val.get<float>());
            } else if (val.is_array() && val.size() >= 2) {
                m_sizeVec2s.emplace(key, UiVec2{val[0].get<float>(), val[1].get<float>()});
            }
        }
    }

    // --- Icons (full-image PNGs registered with mipmaps) ---
    if (j.contains("icons") && j["icons"].is_object()) {
        for (const auto& [key, val] : j["icons"].items()) {
            if (!val.is_object()) continue;
            const std::string iconPath = val.value("path", "");
            if (iconPath.empty()) continue;

            std::filesystem::path resolved = iconPath;
            if (resolved.is_relative()) {
                const std::filesystem::path candidate = baseDir / resolved;
                if (std::filesystem::exists(candidate)) {
                    resolved = candidate;
                }
            }

            int w = 0, h = 0, ch = 0;
            stbi_uc* pixels = stbi_load(resolved.string().c_str(), &w, &h, &ch, 4);
            if (!pixels) {
                VOX_LOGW("ui") << "UiTheme: failed to load icon '" << key
                               << "' from " << resolved.string() << "\n";
                continue;
            }
            const UiTextureId texId = upload(
                pixels, static_cast<std::uint32_t>(w), static_cast<std::uint32_t>(h), true);
            stbi_image_free(pixels);
            if (texId == kUiNoTexture) {
                VOX_LOGW("ui") << "UiTheme: failed to register icon '" << key << "'\n";
                continue;
            }
            // Register as a full-UV single icon in the global registry.
            const std::uint32_t iconSize = static_cast<std::uint32_t>(std::max(w, h));
            const std::string metaJson =
                "{\"iconSize\":" + std::to_string(iconSize) +
                ",\"icons\":{\"" + key + "\":[0,0]}}";
            UiIconRegistry::global().registerAtlas(
                texId, static_cast<std::uint32_t>(w), static_cast<std::uint32_t>(h), metaJson.c_str());
            VOX_LOGI("ui") << "UiTheme: registered icon '" << key << "' ("
                           << w << "x" << h << ", mipmapped)\n";
        }
    }

    m_loaded = true;
    VOX_LOGI("ui") << "UiTheme: loaded '" << m_name << "'\n";
    return true;
}

UiColor UiTheme::color(std::string_view key) const {
    const auto it = m_colors.find(std::string(key));
    return (it != m_colors.end()) ? it->second : UiColor{};
}

const Font* UiTheme::font(std::string_view key) const {
    const auto it = m_fonts.find(std::string(key));
    return (it != m_fonts.end()) ? &it->second : nullptr;
}

FontSet UiTheme::bodyFontSet() const {
    const Font* body = font("body");
    FontSet fs{};
    fs.regular    = body;
    fs.bold       = font("bold");
    fs.italic     = font("italic");
    fs.boldItalic = font("boldItalic");
    if (fs.bold       == nullptr) fs.bold       = body;
    if (fs.italic     == nullptr) fs.italic     = body;
    if (fs.boldItalic == nullptr) fs.boldItalic = body;
    return fs;
}

std::optional<UiNineSlice> UiTheme::frame(std::string_view key) const {
    const auto it = m_frames.find(std::string(key));
    if (it != m_frames.end()) {
        return it->second;
    }
    return std::nullopt;
}

float UiTheme::size(std::string_view key) const {
    const auto it = m_sizes.find(std::string(key));
    return (it != m_sizes.end()) ? it->second : 0.0f;
}

UiVec2 UiTheme::sizeVec2(std::string_view key) const {
    const auto it = m_sizeVec2s.find(std::string(key));
    return (it != m_sizeVec2s.end()) ? it->second : UiVec2{};
}

}  // namespace odai::ui
