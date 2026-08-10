#pragma once

#include "ui/nav_input.h"
#include "ui/ui_types.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

// Spatial focus navigation: "which control does Down go to", answered from
// on-screen geometry rather than from a hand-maintained ordering.
//
// Declaration order is deliberately NOT used. A HUD is laid out in whatever
// order is convenient to draw, and any explicit next/prev wiring rots the
// moment a control moves -- silently, because a stale link still navigates
// somewhere plausible. Geometry cannot go stale: it is recomputed from the
// rects the UI just drew.
//
// Usage per frame:
//   ring.beginFrame();
//   const int id = ring.addItem(buttonRect);   // once per focusable control
//   ring.applyNavigation(navInput);            // after every item is added
//   if (ring.isFocused(id) && navInput.pressed(UiNavAction::Accept)) { ... }
namespace odai::ui {

class NavFocusRing {
public:
    // Clears this frame's item list. Focus itself survives -- it is identified
    // by index into a list the caller rebuilds in the same order every frame,
    // which is the same contract an immediate-mode UI already relies on.
    void beginFrame() { m_items.clear(); }

    // Registers a focusable control and returns its index for this frame.
    int addItem(const UiRect& bounds, bool enabled = true) {
        const auto index = static_cast<int>(m_items.size());
        m_items.push_back(Item{bounds, enabled});
        return index;
    }

    [[nodiscard]] int focused() const { return m_focused; }
    [[nodiscard]] bool isFocused(int index) const { return index >= 0 && index == m_focused; }
    [[nodiscard]] std::size_t itemCount() const { return m_items.size(); }

    void setFocus(int index) { m_focused = index; }

    // Moves focus in response to this frame's directional input.
    //
    // Call AFTER every addItem for the frame: navigation needs the complete
    // geometry, and running it against a partial list would let the first few
    // controls capture every move.
    void applyNavigation(const UiNavInput& input) {
        if (m_items.empty()) {
            m_focused = -1;
            return;
        }
        if (!isValidFocus(m_focused)) {
            // Nothing focused yet (first frame, or the focused control went
            // away). Adopt the first enabled item so the very first d-pad press
            // does something visible instead of being swallowed.
            m_focused = firstEnabled();
        }
        if (input.pressed(UiNavAction::Up)) {
            move(0.0f, -1.0f);
        }
        if (input.pressed(UiNavAction::Down)) {
            move(0.0f, 1.0f);
        }
        if (input.pressed(UiNavAction::Left)) {
            move(-1.0f, 0.0f);
        }
        if (input.pressed(UiNavAction::Right)) {
            move(1.0f, 0.0f);
        }
    }

    // Lets the mouse and the controller share one focus model: hovering a
    // control focuses it, so clicking and pressing A act on the same thing.
    void focusHovered(const UiVec2& pointerPx) {
        for (std::size_t i = 0; i < m_items.size(); ++i) {
            const Item& item = m_items[i];
            if (!item.enabled) {
                continue;
            }
            if (pointerPx.x >= item.bounds.minX && pointerPx.x <= item.bounds.maxX &&
                pointerPx.y >= item.bounds.minY && pointerPx.y <= item.bounds.maxY) {
                m_focused = static_cast<int>(i);
                return;
            }
        }
    }

    [[nodiscard]] UiRect focusedBounds() const {
        if (!isValidFocus(m_focused)) {
            return UiRect{};
        }
        return m_items[static_cast<std::size_t>(m_focused)].bounds;
    }

private:
    struct Item {
        UiRect bounds{};
        bool enabled = true;
    };

    [[nodiscard]] bool isValidFocus(int index) const {
        return index >= 0 && static_cast<std::size_t>(index) < m_items.size() &&
            m_items[static_cast<std::size_t>(index)].enabled;
    }

    [[nodiscard]] int firstEnabled() const {
        for (std::size_t i = 0; i < m_items.size(); ++i) {
            if (m_items[i].enabled) {
                return static_cast<int>(i);
            }
        }
        return -1;
    }

    static UiVec2 centerOf(const UiRect& rect) {
        return UiVec2{(rect.minX + rect.maxX) * 0.5f, (rect.minY + rect.maxY) * 0.5f};
    }

    // Picks the nearest item lying in the requested direction.
    //
    // The score weights cross-axis distance far more heavily than along-axis
    // distance (kCrossAxisPenalty). That asymmetry is the whole algorithm: with
    // equal weighting, pressing Down in a two-column menu happily jumps to the
    // other column because it happens to be a few pixels closer, which feels
    // like the UI is fighting you. Weighted, Down prefers the item directly
    // below even when a diagonal neighbour is nearer in raw distance.
    void move(float dirX, float dirY) {
        if (!isValidFocus(m_focused)) {
            m_focused = firstEnabled();
            return;
        }
        constexpr float kCrossAxisPenalty = 3.0f;
        const UiVec2 from = centerOf(m_items[static_cast<std::size_t>(m_focused)].bounds);

        int best = -1;
        float bestScore = std::numeric_limits<float>::max();
        for (std::size_t i = 0; i < m_items.size(); ++i) {
            if (static_cast<int>(i) == m_focused || !m_items[i].enabled) {
                continue;
            }
            const UiVec2 to = centerOf(m_items[i].bounds);
            const float deltaX = to.x - from.x;
            const float deltaY = to.y - from.y;
            // Projection onto the travel direction; must be genuinely forward.
            const float along = (deltaX * dirX) + (deltaY * dirY);
            if (along <= 0.0f) {
                continue;
            }
            const float cross = std::fabs((deltaX * dirY) - (deltaY * dirX));
            const float score = along + (cross * kCrossAxisPenalty);
            if (score < bestScore) {
                bestScore = score;
                best = static_cast<int>(i);
            }
        }
        if (best >= 0) {
            m_focused = best;
        }
        // No candidate: focus stays put. Deliberately NOT wrapping around --
        // pressing Down at the bottom of a list and landing back at the top is
        // disorienting on a controller, where you cannot see where the cursor
        // went the way you can follow a mouse.
    }

    std::vector<Item> m_items;
    int m_focused = -1;
};

}  // namespace odai::ui
