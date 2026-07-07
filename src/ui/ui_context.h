#pragma once

#include "ui/ui_input.h"
#include "ui/ui_types.h"
#include "ui/widget.h"

#include <functional>
#include <memory>

// Owns the widget tree and drives it each frame: dispatch input -> events ->
// callbacks (update), then emit geometry (build). Pure CPU.
namespace odai::ui {

class UiDrawList;

class UiContext {
public:
    void setViewport(const UiVec2& sizePx);
    [[nodiscard]] const UiVec2& viewport() const { return viewport_; }

    Widget* setRoot(std::unique_ptr<Widget> root);
    [[nodiscard]] Widget* root() const { return root_.get(); }

    // Dispatch a frame of input to the tree, firing widget callbacks.
    void update(const UiInput& input);

    // Advance every widget's animation state (tweens, Sequences) by dt seconds.
    // Call once per frame, independent of update(input) — this replaces having
    // to remember to call update(dt) on each animated widget (Panel, Toggle,
    // ToastManager, ...) individually.
    void tick(float dt);

    // Reset the draw list and emit the tree's geometry.
    void build(UiDrawList& drawList) const;

    // Emit the tree's geometry without resetting the draw list first, so callers
    // can pre-populate the list with background content (e.g. world-space labels).
    void buildAppend(UiDrawList& drawList) const;

    // True when the cursor is over a widget this frame; the app uses this to keep
    // the OS cursor visible and to stop clicks reaching the 3D scene.
    [[nodiscard]] bool wantsMouse() const { return wantsMouse_; }

    // Optional callback fired once whenever a left-click is consumed by a widget
    // this frame (e.g. a button activation). Lets the app play a UI click sound
    // without the Vulkan-free UI library depending on the audio module.
    void setClickFeedback(std::function<void()> cb) { clickFeedback_ = std::move(cb); }

private:
    std::unique_ptr<Widget> root_;
    UiVec2 viewport_{};
    bool wantsMouse_ = false;
    std::function<void()> clickFeedback_;
};

}  // namespace odai::ui
