#pragma once

#include "ui/ui_input.h"
#include "ui/ui_types.h"
#include "ui/widget.h"

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

    // Reset the draw list and emit the tree's geometry.
    void build(UiDrawList& drawList) const;

    // True when the cursor is over a widget this frame; the app uses this to keep
    // the OS cursor visible and to stop clicks reaching the 3D scene.
    [[nodiscard]] bool wantsMouse() const { return wantsMouse_; }

private:
    std::unique_ptr<Widget> root_;
    UiVec2 viewport_{};
    bool wantsMouse_ = false;
};

}  // namespace odai::ui
