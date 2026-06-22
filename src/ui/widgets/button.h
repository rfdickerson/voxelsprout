#pragma once

#include "ui/font.h"
#include "ui/ui_draw_list.h"
#include "ui/ui_types.h"
#include "ui/widget.h"

#include <functional>
#include <string>

// A clickable button: 4 visual states and an onClick callback fired on a
// press-then-release inside the button.
namespace odai::ui {

class Button : public Widget {
public:
    enum class State { Normal, Hover, Pressed, Disabled };

    Button(const Font* font, std::string label, std::function<void()> onClick)
        : font_(font), label_(std::move(label)), onClick_(std::move(onClick)) {}

    UiColor colorNormal{0.12f, 0.18f, 0.24f, 0.90f};
    UiColor colorHover{0.18f, 0.26f, 0.34f, 0.95f};
    UiColor colorPressed{0.10f, 0.14f, 0.18f, 0.95f};
    UiColor colorDisabled{0.10f, 0.12f, 0.14f, 0.60f};
    UiColor borderColor{0.85f, 0.72f, 0.44f, 0.55f};
    UiColor labelColor{0.91f, 0.82f, 0.51f, 1.0f};
    float borderThicknessPx = 1.0f;
    // Corner radius in pixels (DPI-scaled by the caller). Drawn as an anti-aliased
    // SDF rounded rect; a very large value yields a pill. 0 = sharp corners.
    // Kept small for the squared-off Victorian look — just enough to soften aliasing.
    float cornerRadiusPx = 2.0f;
    // Mouse-over glow: a soft SDF halo drawn behind the button while hovered or
    // pressed. Set glowSizePx to 0 to disable. Caller scales glowSizePx by DPI.
    UiColor glowColor{0.95f, 0.72f, 0.35f, 0.55f};
    float glowSizePx = 12.0f;
    // When true, the glow is drawn even at rest (not only on hover/press) — used
    // by the smart turn button to pulse a "ready to advance" halo. Caller animates
    // glowSizePx/glowColor.a over time for the pulse.
    bool drawGlowAtRest = false;
    // Bevel: two-tone border drawn over the fill, simulating a top-left light source.
    // highlightColor lights the top edge; shadowColor darkens the bottom edge.
    // Both follow cornerRadiusPx. Set bevelInward=true for a pressed/recessed look.
    bool    showBevel           = false;
    UiColor bevelHighlightColor {1.0f, 1.0f, 1.0f, 0.28f};
    UiColor bevelShadowColor    {0.0f, 0.0f, 0.0f, 0.45f};
    float   bevelThicknessPx    = 2.0f;
    bool    bevelInward         = false;
    // Optional state accent: a thin vertical stripe just inside the left edge.
    // Used by the smart turn button to signal the current required action by
    // color. Drawn only when accentColor.a > 0, so default buttons are unchanged.
    // Caller scales accentWidthPx by DPI.
    UiColor accentColor{0.0f, 0.0f, 0.0f, 0.0f};
    float accentWidthPx = 4.0f;

    void setEnabled(bool enabled);
    [[nodiscard]] bool enabled() const { return enabled_; }
    [[nodiscard]] State state() const { return state_; }
    void setLabel(std::string label) { label_ = std::move(label); }

    void draw(UiDrawList& drawList) const override;
    bool onEvent(UiEvent& event) override;

private:
    [[nodiscard]] UiColor backgroundForState() const;

    const Font* font_ = nullptr;
    std::string label_;
    std::function<void()> onClick_;
    State state_ = State::Normal;
    bool enabled_ = true;
    bool hovered_ = false;
    bool pressedInside_ = false;
};

}  // namespace odai::ui
