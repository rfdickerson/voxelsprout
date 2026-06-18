#pragma once

#include "ui/animation.h"
#include "ui/font.h"
#include "ui/rich_text.h"
#include "ui/ui_types.h"

#include <optional>
#include <string>

namespace odai::ui {

class UiDrawList;

// Proper tooltip manager. Replaces the ad-hoc Tween+text in App.
// Supports: hover delay, fade tween, screen-edge avoidance, and a simple
// nested tooltip (a second requestTooltip while one is already visible shows
// both adjacently).
//
// Usage each frame:
//   1. Widgets call requestTooltip(anchor, markup, delay) on hover.
//   2. Call clearTooltip() when nothing is hovered.
//   3. Call update(dt) to advance the delay/fade.
//   4. Call draw(dl, viewport) to emit the tooltip geometry.
class UiTooltipManager {
public:
    // Set the font set and optional 9-slice frame used to render tooltips.
    void setFont(const FontSet& fonts) { m_fonts = fonts; }
    void setFrame(std::optional<UiNineSlice> frame) { m_frame = frame; }
    void setMaxWidth(float w) { m_maxWidth = w; }
    void setBackground(const UiColor& c) { m_background = c; }
    void setTextColor(const UiColor& c) { m_textColor = c; }

    // Called each frame by widgets that want a tooltip. Multiple calls per frame
    // are ok; the first one wins for the primary slot.
    void requestTooltip(const UiRect& anchor, std::string markup, float delaySeconds = 0.4f);

    // Called when nothing is hovered.
    void clearTooltip();

    void update(float dt);
    void draw(UiDrawList& dl, const UiVec2& viewport) const;

    bool isVisible() const;

private:
    struct Slot {
        std::string markup;
        UiRect anchor;
        float delaySeconds = 0.4f;
        float elapsed = 0.0f;      // Time since markup was set.
        bool showing = false;
        Tween fade{};
        mutable RichTextLayout layout;
        mutable bool layoutDirty = true;
    };

    Slot m_primary;
    bool m_hoveredThisFrame = false;

    FontSet m_fonts{};
    std::optional<UiNineSlice> m_frame;
    float m_maxWidth = 280.0f;
    UiColor m_background{0.12f, 0.09f, 0.06f, 0.95f};
    UiColor m_textColor{0.9f, 0.85f, 0.75f, 1.0f};

    static constexpr float kPadding = 10.0f;

    UiRect positionTooltip(const UiRect& contentRect, const UiRect& anchor,
                           const UiVec2& viewport) const;
};

}  // namespace odai::ui
