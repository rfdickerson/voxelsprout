#include "ui/widgets/resource_bar_panel.h"

#include "ui/icon_atlas.h"
#include "ui/ui_draw_list.h"
#include "ui/widgets/label.h"
#include "ui/widgets/panel.h"

#include <memory>
#include <utility>

namespace odai::ui {
namespace {

// One clickable slot in the resource bar.
class ResourceSlot : public Widget {
public:
    ResourceSlot(const FontSet& fonts, float scale) : fonts_(fonts), s_(scale) {}

    std::string value;
    UiColor color{0.92f, 0.88f, 0.72f, 1.0f};
    UiTextureId iconTex = kUiNoTexture;
    UiRect iconUv{0.0f, 0.0f, 1.0f, 1.0f};
    std::function<void()> onClick;

    void draw(UiDrawList& dl) const override {
        if (hovered_ && onClick) {
            dl.addRoundRectFilled(rect_, UiColor{1.0f, 1.0f, 1.0f, 0.08f}, 3.0f * s_);
        }

        const float iconSide = std::min(rect_.height() - 4.0f * s_, 18.0f * s_);
        const float iconY    = rect_.minY + (rect_.height() - iconSide) * 0.5f;
        const float pad      = 6.0f * s_;

        float x = rect_.minX + pad;
        if (iconTex != kUiNoTexture) {
            dl.addImage(UiRect::fromXYWH(x, iconY, iconSide, iconSide),
                        iconTex, UiColor{1, 1, 1, 1}, iconUv);
            x += iconSide + 3.0f * s_;
        }

        if (fonts_.regular && !value.empty()) {
            const float ty = rect_.minY + (rect_.height() - fonts_.regular->lineHeightPx()) * 0.5f;
            dl.addText(*fonts_.regular, value, UiVec2{x, ty}, color);
        }
    }

    bool onEvent(UiEvent& ev) override {
        switch (ev.type) {
            case UiEvent::Type::MouseMove:
                hovered_ = rect_.contains(ev.mousePx);
                return false;
            case UiEvent::Type::MouseDown:
                if (ev.button == UiMouseButton::Left && rect_.contains(ev.mousePx)) {
                    pressedInside_ = true;
                    ev.handled = true;
                    return true;
                }
                return false;
            case UiEvent::Type::MouseUp:
                if (ev.button != UiMouseButton::Left || !pressedInside_) return false;
                pressedInside_ = false;
                if (rect_.contains(ev.mousePx) && onClick) {
                    onClick();
                    ev.handled = true;
                    return true;
                }
                return false;
            default:
                return false;
        }
    }

private:
    FontSet fonts_;
    float s_ = 1.0f;
    bool hovered_ = false;
    bool pressedInside_ = false;
};

}  // namespace

void ResourceBarPanel::setResources(const UiRect& rect, float s,
                                    const std::vector<ResourceEntry>& entries) {
    children_.clear();
    setRect(rect);

    auto bg = std::make_unique<Panel>();
    bg->setRect(rect);
    bg->background        = UiColor{0.06f, 0.07f, 0.10f, 0.88f};
    bg->borderColor       = UiColor{0.30f, 0.30f, 0.40f, 0.30f};
    bg->borderThicknessPx = 1.0f * s;
    bg->cornerRadiusPx    = 0.0f;
    bg_ = static_cast<Panel*>(addChild(std::move(bg)));

    if (entries.empty()) return;

    const float slotW = rect.width() / static_cast<float>(entries.size());

    for (std::size_t i = 0; i < entries.size(); ++i) {
        const ResourceEntry& e = entries[i];
        const float x = rect.minX + static_cast<float>(i) * slotW;

        auto slot = std::make_unique<ResourceSlot>(fonts_, s);
        slot->setRect(UiRect::fromXYWH(x, rect.minY, slotW, rect.height()));
        slot->value   = e.value;
        slot->color   = e.color;
        slot->onClick = e.onClick;

        UiIconEntry icon{};
        if (UiIconRegistry::global().resolve(e.iconName, icon)) {
            slot->iconTex = icon.textureId;
            slot->iconUv  = icon.uv;
        }
        addChild(std::move(slot));

        // Divider between slots
        if (i + 1 < entries.size()) {
            auto div = std::make_unique<Panel>();
            div->setRect(UiRect::fromXYWH(x + slotW - 0.5f * s, rect.minY + 4.0f * s,
                                           1.0f * s, rect.height() - 8.0f * s));
            div->background        = UiColor{1.0f, 1.0f, 1.0f, 0.10f};
            div->borderThicknessPx = 0.0f;
            addChild(std::move(div));
        }
    }
}

}  // namespace odai::ui
