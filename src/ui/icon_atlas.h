#pragma once

#include "ui/ui_types.h"

#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace odai::ui {

struct UiIconEntry {
    UiTextureId textureId = kUiNoTexture;
    UiRect uv{};  // Sub-rect in [0,1] atlas space.
};

// Sprite-sheet registry. Icons are stored in power-of-two atlases; each atlas
// occupies one texture slot. Multiple atlases can be registered (e.g. yields,
// terrain, portraits). Lookup is by name across all registered atlases.
//
// Atlas metadata JSON format:
//   { "iconSize": 32, "icons": { "food": [0,0], "gold": [1,0] } }
// [x,y] are grid column/row; pixel rect = [x*iconSize, y*iconSize, iconSize, iconSize].
class UiIconRegistry {
public:
    // Register a spritesheet. Returns the assigned texture ID (kUiNoTexture on
    // failure). The textureId must already have been obtained from
    // Renderer::registerUiTextureRgba8() before calling this.
    bool registerAtlas(UiTextureId textureId,
                       std::uint32_t atlasWidthPx,
                       std::uint32_t atlasHeightPx,
                       const char* metaJson);

    // Resolve an icon name to its texture and UV rect.
    // Returns false if the name is not found.
    bool resolve(std::string_view name, UiIconEntry& out) const;

    // Global singleton — populated at app startup.
    static UiIconRegistry& global();

private:
    std::unordered_map<std::string, UiIconEntry> m_icons;
};

}  // namespace odai::ui
