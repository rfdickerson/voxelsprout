#include "ui/icon_atlas.h"

#include "core/log.h"

#include <nlohmann/json.hpp>

namespace odai::ui {

bool UiIconRegistry::registerAtlas(UiTextureId textureId,
                                   std::uint32_t atlasWidthPx,
                                   std::uint32_t atlasHeightPx,
                                   const char* metaJson) {
    if (textureId == kUiNoTexture || metaJson == nullptr) {
        VOX_LOGE("ui") << "UiIconRegistry::registerAtlas: invalid arguments\n";
        return false;
    }
    if (atlasWidthPx == 0 || atlasHeightPx == 0) {
        VOX_LOGE("ui") << "UiIconRegistry::registerAtlas: zero atlas dimensions\n";
        return false;
    }

    nlohmann::json j;
    try {
        j = nlohmann::json::parse(metaJson);
    } catch (const nlohmann::json::exception& e) {
        VOX_LOGE("ui") << "UiIconRegistry::registerAtlas: JSON parse error: " << e.what() << "\n";
        return false;
    }

    const std::uint32_t iconSize = j.value("iconSize", 32u);
    if (iconSize == 0) {
        VOX_LOGE("ui") << "UiIconRegistry::registerAtlas: iconSize must be > 0\n";
        return false;
    }

    const float invW = 1.0f / static_cast<float>(atlasWidthPx);
    const float invH = 1.0f / static_cast<float>(atlasHeightPx);
    const float uvW = static_cast<float>(iconSize) * invW;
    const float uvH = static_cast<float>(iconSize) * invH;

    if (!j.contains("icons") || !j["icons"].is_object()) {
        VOX_LOGW("ui") << "UiIconRegistry::registerAtlas: no 'icons' object in metadata\n";
        return true;  // Not an error; atlas registered but empty.
    }

    for (const auto& [name, coords] : j["icons"].items()) {
        if (!coords.is_array() || coords.size() < 2) {
            VOX_LOGW("ui") << "UiIconRegistry: skipping malformed entry '" << name << "'\n";
            continue;
        }
        const std::uint32_t gx = coords[0].get<std::uint32_t>();
        const std::uint32_t gy = coords[1].get<std::uint32_t>();
        const float u0 = static_cast<float>(gx * iconSize) * invW;
        const float v0 = static_cast<float>(gy * iconSize) * invH;
        UiIconEntry entry;
        entry.textureId = textureId;
        entry.uv = UiRect{u0, v0, u0 + uvW, v0 + uvH};
        m_icons.emplace(name, entry);
    }

    // Register aliases (alternate names that resolve to an existing icon).
    if (j.contains("aliases") && j["aliases"].is_object()) {
        for (const auto& [alias, target] : j["aliases"].items()) {
            const auto it = m_icons.find(target.get<std::string>());
            if (it != m_icons.end()) {
                m_icons.emplace(alias, it->second);
            }
        }
    }

    return true;
}

bool UiIconRegistry::resolve(std::string_view name, UiIconEntry& out) const {
    const auto it = m_icons.find(std::string(name));
    if (it == m_icons.end()) {
        return false;
    }
    out = it->second;
    return true;
}

UiIconRegistry& UiIconRegistry::global() {
    static UiIconRegistry s_instance;
    return s_instance;
}

}  // namespace odai::ui
