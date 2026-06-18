#pragma once

#include "ui/ui_types.h"

#include <cstdint>
#include <vector>

// Per-frame UI input snapshot, populated by the platform layer (GLFW in app/)
// and consumed by UiContext to dispatch events. Pure data, no platform headers.
namespace odai::ui {

enum class UiMouseButton : std::uint8_t {
    Left = 0,
    Right = 1,
    Middle = 2,
    Count = 3,
};

struct UiButtonState {
    bool down = false;      // Held this frame.
    bool pressed = false;   // Transitioned up -> down this frame (rising edge).
    bool released = false;  // Transitioned down -> up this frame (falling edge).
};

struct UiInput {
    UiVec2 mousePx{};                       // Cursor position in framebuffer pixels.
    UiVec2 mouseDeltaPx{};                  // Movement since last frame.
    UiButtonState buttons[static_cast<std::size_t>(UiMouseButton::Count)]{};
    float scrollDelta = 0.0f;               // Wheel ticks this frame (+up).
    std::vector<std::uint32_t> textInput;   // Unicode codepoints entered this frame.

    [[nodiscard]] const UiButtonState& button(UiMouseButton b) const {
        return buttons[static_cast<std::size_t>(b)];
    }

    // Clear per-frame edges and text; preserve held state across frames.
    void beginFrame() {
        for (UiButtonState& state : buttons) {
            state.pressed = false;
            state.released = false;
        }
        scrollDelta = 0.0f;
        mouseDeltaPx = {};
        textInput.clear();
    }

    // Apply a new mouse-button level, deriving pressed/released edges.
    void setButton(UiMouseButton b, bool isDown) {
        UiButtonState& state = buttons[static_cast<std::size_t>(b)];
        state.pressed = isDown && !state.down;
        state.released = !isDown && state.down;
        state.down = isDown;
    }
};

}  // namespace odai::ui
