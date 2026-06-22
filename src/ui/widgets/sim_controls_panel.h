#pragma once

#include "ui/font.h"
#include "ui/ui_types.h"
#include "ui/widget.h"

#include <functional>
#include <string>

namespace odai::ui {

class Panel;
class Label;
class Button;

// Simulation speed controls + optional end-turn / next-period button. Covers
// the two primary control patterns:
//
//   Real-time sim  (city builders, colony sims, management games):
//     showSpeedButtons = true   →  Pause / Normal / Fast / Ultrafast chips
//     showEndTurn      = false
//
//   Turn-based     (4X strategy):
//     showSpeedButtons = false
//     showEndTurn      = true   →  "End Turn" / "Next Turn" button
//
//   Combined (pausable turn-based, real-time with manual advance):
//     Both flags = true
//
// The date/turn label is always shown (blank it if not applicable).
//
// Pure UI (Vulkan-free, no game types): the caller owns the simulation clock and
// passes in the desired display state each frame or on change.
class SimControlsPanel : public Widget {
public:
    enum class SimSpeed { Paused, Normal, Fast, Ultrafast };

    struct State {
        SimSpeed    speed = SimSpeed::Normal;
        std::string dateLabel;          // "1 AD", "Year 2050", "Week 32 · Turn 14", …
        bool        showSpeedButtons = true;   // Show pause/play/fast chips.
        bool        showEndTurn      = true;   // Show an end-turn / advance button.
        std::string endTurnLabel;              // "" → "End Turn".
        bool        endTurnEnabled = true;
        std::function<void(SimSpeed)> onSpeedChange;
        std::function<void()>         onEndTurn;
    };

    explicit SimControlsPanel(const FontSet& fonts) : fonts_(fonts) {}

    // (Re)build the panel into `rect` at DPI scale `s`.
    void setState(const UiRect& rect, float s, const State& state);

    // Swap the speed highlight without a full rebuild. Call from onSpeedChange.
    void setSpeed(SimSpeed speed);

    // Expose the background Panel for per-game styling overrides.
    Panel* bgPanel() const { return bg_; }

private:
    FontSet fonts_;
    Panel*  bg_          = nullptr;
    Button* pauseBtn_    = nullptr;
    Button* normalBtn_   = nullptr;
    Button* fastBtn_     = nullptr;
    Button* ultrafastBtn_ = nullptr;
    Button* endTurnBtn_  = nullptr;
    Label*  dateLabel_   = nullptr;
    SimSpeed speed_      = SimSpeed::Normal;
};

}  // namespace odai::ui
