#pragma once

#include "ui/font.h"
#include "ui/ui_draw_list.h"
#include "ui/ui_types.h"
#include "ui/widget.h"

#include <functional>
#include <string>
#include <vector>

namespace odai::ui {

class Panel;
class Label;
class Button;

// Bottom-center strategy minimap: a flat card framing a baked map texture, with
// a row of lens chips (Terrain / Owner / Relief ...) that swap which pre-baked
// texture is shown, and an optional camera-viewport rectangle overlay. The app
// bakes + registers the textures (one per lens) and hands the panel their ids;
// switching a lens only swaps the drawn id (no GPU work, no tree rebuild).
//
// Composite widget: owns a background Panel, a title Label, and chip Buttons.
// Vulkan-free; the map image is drawn directly in draw().
class MinimapPanel : public Widget {
public:
    explicit MinimapPanel(const FontSet& fonts) : fonts_(fonts) {}

    // (Re)build into `rect` at DPI scale `s`. `image` is the texture for the
    // active lens; `imageAspect` = mapWidth/mapHeight (the image is aspect-fit
    // into the available area). `lenses` are chip labels; `onLens(i)` fires when
    // chip i is clicked.
    void setMinimap(const UiRect& rect, float s, const std::string& title,
                    UiTextureId image, float imageAspect,
                    const std::vector<std::string>& lenses, int activeIndex,
                    std::function<void(int)> onLens);

    // Swap the displayed lens without rebuilding (safe to call from a chip
    // callback): shows `image` and moves the active accent to chip `index`.
    void setActive(int index, UiTextureId image);

    // Camera viewport overlay, in normalized [0,1] image space. Pass an invalid
    // (zero) rect to hide it.
    void setViewport(const UiRect& normalized) {
        viewport_ = normalized;
        hasViewport_ = normalized.valid();
    }

    void draw(UiDrawList& drawList) const override;

private:
    FontSet fonts_;
    Panel* bg_ = nullptr;
    Label* titleLabel_ = nullptr;
    std::vector<Button*> chips_;
    UiTextureId image_ = kUiNoTexture;
    UiRect imageRect_{};   // pixel rect the map texture is drawn into (aspect-fit)
    float scale_ = 1.0f;
    float radius_ = 0.0f;
    UiRect viewport_{};
    bool hasViewport_ = false;
};

}  // namespace odai::ui
