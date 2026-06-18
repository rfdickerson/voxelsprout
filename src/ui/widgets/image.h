#pragma once

#include "ui/ui_types.h"
#include "ui/widget.h"

namespace odai::ui {

// Displays a single texture, optionally tinted and UV-cropped.
class Image : public Widget {
public:
    enum class FitMode { Stretch, Contain, Cover, Tile };

    explicit Image(UiTextureId textureId = kUiNoTexture);

    UiTextureId textureId = kUiNoTexture;
    UiColor     tint{1.0f, 1.0f, 1.0f, 1.0f};
    UiRect      uvRect{0.0f, 0.0f, 1.0f, 1.0f};
    FitMode     fitMode = FitMode::Stretch;

    void draw(UiDrawList& dl) const override;
};

}  // namespace odai::ui
