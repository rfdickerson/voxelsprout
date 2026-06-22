#include "ui/widgets/minimap_panel.h"

#include "ui/widgets/button.h"
#include "ui/widgets/label.h"
#include "ui/widgets/panel.h"

namespace odai::ui {

void MinimapPanel::setMinimap(const UiRect& rect, float s, const std::string& title,
                              UiTextureId image, float imageAspect,
                              const std::vector<std::string>& lenses, int activeIndex,
                              std::function<void(int)> onLens) {
    setRect(rect);
    children_.clear();
    chips_.clear();
    scale_  = s;
    radius_ = 8.0f * s;

    const float pad = 10.0f * s;

    // Flat clean-modern card background.
    auto bg = std::make_unique<Panel>();
    bg->setRect(rect);
    bg->styleCard(s, 0.92f);
    bg_ = static_cast<Panel*>(addChild(std::move(bg)));

    // Title (top).
    const Font* hf = fonts_.bold != nullptr ? fonts_.bold : fonts_.regular;
    const float titleH = hf != nullptr ? hf->lineHeightPx() : 18.0f * s;
    auto tl = std::make_unique<Label>(fonts_, "<b><color=#C9A24B>" + title + "</color></b>");
    tl->align = UiTextAlign::Center;
    tl->setRect(UiRect::fromXYWH(rect.minX + pad, rect.minY + 4.0f * s,
                                 rect.width() - 2.0f * pad, titleH));
    titleLabel_ = static_cast<Label*>(addChild(std::move(tl)));

    // Lens chips (bottom row), evenly spaced.
    const float chipH = 20.0f * s;
    const float chipY = rect.maxY - pad - chipH;
    const int n = static_cast<int>(lenses.size());
    if (n > 0) {
        const float gap   = 6.0f * s;
        const float chipW = (rect.width() - 2.0f * pad - gap * static_cast<float>(n - 1)) /
                            static_cast<float>(n);
        for (int i = 0; i < n; ++i) {
            const float cx = rect.minX + pad + static_cast<float>(i) * (chipW + gap);
            auto chip = std::make_unique<Button>(fonts_.regular, lenses[static_cast<std::size_t>(i)],
                                                 [onLens, i]() { if (onLens) onLens(i); });
            chip->setRect(UiRect::fromXYWH(cx, chipY, chipW, chipH));
            chip->cornerRadiusPx    = 2.0f * s;
            chip->colorNormal       = UiColor{0.10f, 0.12f, 0.16f, 0.92f};
            chip->colorHover        = UiColor{0.18f, 0.22f, 0.28f, 1.0f};
            chip->colorPressed      = UiColor{0.08f, 0.09f, 0.12f, 1.0f};
            chip->borderColor       = UiColor{1.0f, 1.0f, 1.0f, 0.10f};
            chip->borderThicknessPx = 1.0f * s;
            chip->labelColor        = UiColor{0.85f, 0.89f, 0.94f, 1.0f};
            // Subtle brass halo on hover so the active lens choice reads clearly.
            chip->glowColor         = UiColor{0.788f, 0.635f, 0.294f, 0.45f};
            chip->glowSizePx        = 8.0f * s;
            chip->accentWidthPx     = 3.0f * s;
            chips_.push_back(static_cast<Button*>(addChild(std::move(chip))));
        }
    }

    // Map image area between the title and the chips, aspect-fit and centered.
    const float availTop = rect.minY + 4.0f * s + titleH + 4.0f * s;
    const float availBot = chipY - 6.0f * s;
    UiRect avail{rect.minX + pad, availTop, rect.maxX - pad, availBot};
    imageRect_ = avail;
    if (avail.valid() && imageAspect > 0.0f) {
        const float aw = avail.width();
        const float ah = avail.height();
        float w = aw;
        float h = aw / imageAspect;
        if (h > ah) { h = ah; w = ah * imageAspect; }
        const float cx = (avail.minX + avail.maxX) * 0.5f;
        const float cy = (avail.minY + avail.maxY) * 0.5f;
        imageRect_ = UiRect{cx - w * 0.5f, cy - h * 0.5f, cx + w * 0.5f, cy + h * 0.5f};
    }

    setActive(activeIndex, image);
}

void MinimapPanel::setActive(int index, UiTextureId image) {
    image_ = image;
    for (int i = 0; i < static_cast<int>(chips_.size()); ++i) {
        chips_[static_cast<std::size_t>(i)]->accentColor =
            (i == index) ? UiColor{0.788f, 0.635f, 0.294f, 1.0f}   // brass = active lens
                         : UiColor{0.0f, 0.0f, 0.0f, 0.0f};
    }
}

void MinimapPanel::draw(UiDrawList& drawList) const {
    if (!visible) {
        return;
    }
    drawChildren(drawList);  // card, title, chips

    if (image_ != kUiNoTexture && imageRect_.valid()) {
        drawList.pushClip(imageRect_);
        drawList.addImage(imageRect_, image_, UiColor{1.0f, 1.0f, 1.0f, 1.0f});
        drawList.popClip();
        // Inner frame around the map image.
        drawList.addRoundRect(imageRect_, UiColor{1.0f, 1.0f, 1.0f, 0.14f}, radius_, 1.0f * scale_);
        // Camera viewport overlay (normalized -> image pixel space).
        if (hasViewport_) {
            const UiRect v{
                imageRect_.minX + viewport_.minX * imageRect_.width(),
                imageRect_.minY + viewport_.minY * imageRect_.height(),
                imageRect_.minX + viewport_.maxX * imageRect_.width(),
                imageRect_.minY + viewport_.maxY * imageRect_.height()};
            drawList.addRoundRect(v, UiColor{1.0f, 0.92f, 0.65f, 0.95f}, 2.0f * scale_, 1.5f * scale_);
        }
    }
}

}  // namespace odai::ui
