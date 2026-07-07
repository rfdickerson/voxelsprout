#include "tools/retro_theme_demo/retro_widgets.h"

#include <algorithm>
#include <cmath>
#include <utility>

namespace odai::tools::retro_theme_demo {

using ui::UiColor;

void RetroUi::init() {
    auto root = std::make_unique<ui::Widget>();
    root->mousePassthrough = true;  // the container never claims the mouse itself
    root_ = ctx_.setRoot(std::move(root));
}

void RetroUi::beginFrame(const WidgetSkin& skin) {
    skin_ = skin;
    // Hide everything; accessor calls this frame re-show what's actually used.
    for (auto& [id, w] : byId_) {
        w->visible = false;
    }
}

template <typename T, typename Factory>
T& RetroUi::acquire(std::string_view id, Factory&& make) {
    const std::string key(id);
    const auto it = byId_.find(key);
    if (it != byId_.end()) {
        it->second->visible = true;
        return *static_cast<T*>(it->second);
    }
    std::unique_ptr<T> created = make();
    T* raw = created.get();
    root_->addChild(std::move(created));
    byId_.emplace(key, raw);
    raw->visible = true;
    return *raw;
}

ui::Slider& RetroUi::slider(std::string_view id, const ui::UiRect& rect) {
    auto& w = acquire<ui::Slider>(id, [] { return std::make_unique<ui::Slider>(); });
    w.setRect(rect);
    applySkin(w);
    return w;
}

ui::TextBox& RetroUi::textBox(std::string_view id, const ui::UiRect& rect, const ui::Font* font,
                             const char* placeholder) {
    auto& w = acquire<ui::TextBox>(id, [&] {
        return std::make_unique<ui::TextBox>(font, placeholder ? placeholder : "");
    });
    w.setRect(rect);
    applySkin(w);
    return w;
}

ui::Toggle& RetroUi::toggle(std::string_view id, const ui::UiRect& rect) {
    auto& w = acquire<ui::Toggle>(id, [] { return std::make_unique<ui::Toggle>(); });
    w.setRect(rect);
    applySkin(w);
    return w;
}

ui::Button& RetroUi::button(std::string_view id, const ui::UiRect& rect, const ui::Font* font,
                           const char* label) {
    auto& w = acquire<ui::Button>(id, [&] {
        return std::make_unique<ui::Button>(font, label ? label : "", [] {});
    });
    w.setRect(rect);
    w.setLabel(label ? label : "");
    applySkin(w);
    return w;
}

ui::TabBar& RetroUi::tabs(std::string_view id, const ui::UiRect& rect, const ui::Font* font,
                         const std::vector<std::string>& labels) {
    auto& w = acquire<ui::TabBar>(id, [&] {
        auto t = std::make_unique<ui::TabBar>(font);
        for (const std::string& label : labels) {
            t->addTab(label);
        }
        return t;
    });
    w.setRect(rect);
    applySkin(w);
    return w;
}

void RetroUi::update(const ui::UiVec2& mousePx, bool leftDown, const ui::UiVec2& viewportPx,
                     const std::vector<std::uint32_t>& textInput, float dt) {
    ctx_.tick(dt);
    input_.beginFrame();
    input_.mousePx = mousePx;
    input_.setButton(ui::UiMouseButton::Left, leftDown);
    input_.textInput = textInput;
    ctx_.setViewport(viewportPx);
    ctx_.update(input_);
}

void RetroUi::append(ui::UiDrawList& drawList) { ctx_.buildAppend(drawList); }

// --- Skinning: map WidgetSkin onto each widget's public style members ---------

void RetroUi::applySkin(ui::Slider& w) const {
    w.trackColor = skin_.trough;
    w.fillColor = skin_.accent;
    w.knobColor = skin_.bevel ? skin_.face : skin_.bevelLight;
    w.knobHoverColor = skin_.faceHover;
    w.glowColor = ui::UiColor{skin_.accent.r, skin_.accent.g, skin_.accent.b, 0.30f};
    w.cornerRadiusPx = skin_.cornerRadius;
    w.knobRadiusPx = std::round(7.0f * skin_.scale);
}

void RetroUi::applySkin(ui::TextBox& w) const {
    w.background = skin_.field;
    w.borderColor = skin_.border;
    w.borderFocusedColor = skin_.accent;
    w.textColor = skin_.text;
    w.placeholderColor = skin_.textDim;
    w.caretColor = skin_.text;
    w.cornerRadiusPx = skin_.cornerRadius;
    w.borderThicknessPx = std::max(1.0f, std::round(1.5f * skin_.scale));
    w.padding = ui::UiVec2{std::round(8.0f * skin_.scale), 0.0f};
}

void RetroUi::applySkin(ui::Toggle& w) const {
    w.trackOn = skin_.accent;
    w.trackOff = skin_.trough;
    w.thumbColor = skin_.bevelLight;
}

void RetroUi::applySkin(ui::Button& w) const {
    w.colorNormal = skin_.face;
    w.colorHover = skin_.faceHover;
    w.colorPressed = skin_.facePressed;
    w.colorDisabled = ui::UiColor{skin_.face.r, skin_.face.g, skin_.face.b, 0.5f};
    w.labelColor = skin_.text;
    w.borderColor = skin_.border;
    w.cornerRadiusPx = skin_.cornerRadius;
    w.glowSizePx = 0.0f;  // retro buttons don't glow
    w.showBevel = skin_.bevel;
    w.bevelHighlightColor = skin_.bevelLight;
    w.bevelShadowColor = skin_.bevelDark;
    w.bevelThicknessPx = std::max(1.0f, std::round(2.0f * skin_.scale));
    w.borderThicknessPx = skin_.bevel ? std::max(1.0f, std::round(skin_.scale)) : 1.0f;
}

void RetroUi::applySkin(ui::TabBar& w) const {
    w.activeTabColor = skin_.face;
    w.inactiveTabColor = skin_.trough;
    w.indicatorColor = skin_.accent;
    w.textActiveColor = skin_.text;
    w.textInactiveColor = skin_.textDim;
    w.dividerColor = skin_.border;
    w.indicatorThicknessPx = std::max(1.0f, std::round(2.0f * skin_.scale));
}

}  // namespace odai::tools::retro_theme_demo
