#pragma once

#include "ui/ui_input.h"
#include "ui/ui_types.h"

#include <cstdint>

// Directional ("console") UI input: the model a game controller and the arrow
// keys share, kept separate from UiInput's pointer model rather than bolted
// onto it.
//
// Why separate: a pointer says WHERE, a d-pad says WHICH WAY. They are not the
// same event dressed differently, and merging them is what produces UIs where
// the controller has to drive a fake cursor around the screen. Everything here
// is device-agnostic -- GLFW's gamepad state, arrow keys and a touch swipe all
// funnel into the same six actions -- so src/ui/ stays free of platform
// headers, exactly as UiInput already is.
namespace odai::ui {

enum class UiNavAction : std::uint8_t {
    Up = 0,
    Down,
    Left,
    Right,
    Accept,   // A / Cross / Enter
    Cancel,   // B / Circle / Escape
    PrevTab,  // LB / L1 / Q
    NextTab,  // RB / R1 / E
    Menu,     // Start / Escape
    Count,
};

// Auto-repeat timing for held directions.
//
// These are not arbitrary: a stick held to one side must not scroll a list at
// frame rate, and a single tap must move exactly one item. The delay-then-rate
// shape is what every console UI uses, and the values are the usual ones --
// long enough that a deliberate single press never double-fires at 144 Hz,
// short enough that holding to scroll a long list does not feel stuck.
struct UiNavRepeatConfig {
    float initialDelaySeconds = 0.40f;
    float repeatIntervalSeconds = 0.11f;
};

// Per-frame directional input, with edges derived the same way UiButtonState
// does it for the mouse.
struct UiNavInput {
    UiButtonState actions[static_cast<std::size_t>(UiNavAction::Count)]{};

    // True once any directional/accept input has been seen. Drives the "is a
    // controller driving this?" decision: a focus highlight that is always on
    // looks broken on a mouse-driven PC screen, and one that never appears
    // makes the UI unusable on a couch. Set by setAction, cleared by the app
    // when the mouse moves.
    bool active = false;

    [[nodiscard]] const UiButtonState& action(UiNavAction a) const {
        return actions[static_cast<std::size_t>(a)];
    }
    [[nodiscard]] bool pressed(UiNavAction a) const { return action(a).pressed; }
    [[nodiscard]] bool down(UiNavAction a) const { return action(a).down; }

    void beginFrame() {
        for (UiButtonState& state : actions) {
            state.pressed = false;
            state.released = false;
        }
    }

    // Apply a raw held level for one action, deriving the press edge.
    void setAction(UiNavAction a, bool isDown) {
        UiButtonState& state = actions[static_cast<std::size_t>(a)];
        state.pressed = isDown && !state.down;
        state.released = !isDown && state.down;
        state.down = isDown;
        active = active || state.pressed;
    }
};

// Turns held directions into repeated presses. Owned by the caller (one per UI
// that navigates) because the timers are per-UI state, not global: a menu that
// opens while a direction is already held must not inherit a half-elapsed
// repeat timer from whatever was reading the stick before it.
class UiNavRepeater {
public:
    explicit UiNavRepeater(UiNavRepeatConfig config = {}) : m_config(config) {}

    // Call once per frame, after the raw levels are set. Synthesizes additional
    // `pressed` edges on the four directions while they are held.
    void update(UiNavInput& input, float deltaSeconds) {
        for (std::size_t i = 0; i < kDirectionCount; ++i) {
            const auto actionIndex = static_cast<std::size_t>(kDirections[i]);
            UiButtonState& state = input.actions[actionIndex];
            if (!state.down) {
                m_heldSeconds[i] = 0.0f;
                m_repeatsFired[i] = 0;
                continue;
            }
            if (state.pressed) {
                // The real press this frame; the repeat clock starts now and
                // must not also fire.
                m_heldSeconds[i] = 0.0f;
                m_repeatsFired[i] = 0;
                continue;
            }
            m_heldSeconds[i] += deltaSeconds;
            // How many repeats *should* have fired by now, from the elapsed
            // time rather than by accumulating -- so a long frame emits the
            // repeats it owes instead of silently dropping them.
            if (m_heldSeconds[i] < m_config.initialDelaySeconds) {
                continue;
            }
            const float sinceFirst = m_heldSeconds[i] - m_config.initialDelaySeconds;
            const auto due = 1 + static_cast<int>(sinceFirst / m_config.repeatIntervalSeconds);
            if (due > m_repeatsFired[i]) {
                m_repeatsFired[i] = due;
                state.pressed = true;
            }
        }
    }

    void reset() {
        for (std::size_t i = 0; i < kDirectionCount; ++i) {
            m_heldSeconds[i] = 0.0f;
            m_repeatsFired[i] = 0;
        }
    }

private:
    static constexpr std::size_t kDirectionCount = 4;
    static constexpr UiNavAction kDirections[kDirectionCount] = {
        UiNavAction::Up, UiNavAction::Down, UiNavAction::Left, UiNavAction::Right};

    UiNavRepeatConfig m_config{};
    float m_heldSeconds[kDirectionCount] = {};
    int m_repeatsFired[kDirectionCount] = {};
};

// Maps a stick axis pair onto the four directions, with a deadzone and
// hysteresis.
//
// Hysteresis (a lower release threshold than press threshold) is the part that
// is easy to leave out and impossible to live without: a stick resting near the
// press threshold otherwise chatters between down and up every frame, which
// reads as a list that scrolls by itself.
struct UiNavStickMapper {
    float pressThreshold = 0.55f;
    float releaseThreshold = 0.35f;

    void apply(UiNavInput& input, float axisX, float axisY) {
        m_x = resolve(m_x, axisX);
        m_y = resolve(m_y, axisY);
        input.setAction(UiNavAction::Left, m_x < 0);
        input.setAction(UiNavAction::Right, m_x > 0);
        // Screen-space: stick up (negative axis on every gamepad GLFW reports)
        // moves focus up.
        input.setAction(UiNavAction::Up, m_y < 0);
        input.setAction(UiNavAction::Down, m_y > 0);
    }

private:
    [[nodiscard]] int resolve(int current, float axis) const {
        const float magnitude = axis < 0.0f ? -axis : axis;
        if (current != 0) {
            // Already latched: hold until the stick falls below the lower bar,
            // or crosses to the other side.
            if (magnitude < releaseThreshold) {
                return 0;
            }
            return (axis > 0.0f) ? 1 : -1;
        }
        if (magnitude < pressThreshold) {
            return 0;
        }
        return (axis > 0.0f) ? 1 : -1;
    }

    int m_x = 0;
    int m_y = 0;
};

}  // namespace odai::ui
