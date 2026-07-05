#pragma once

#include "ui/font.h"
#include "ui/ui_types.h"
#include "ui/widget.h"
#include "ui/widgets/window.h"

#include <functional>
#include <string>

namespace odai::ui {

// Blocking dialog: a dimmed full-screen backdrop behind a centered Window.
// Size this widget's rect to the full viewport (it draws the backdrop over its
// own rect) and call layout() after open() or whenever the viewport resizes.
// Reuses Window for the title bar / close button / body chrome rather than
// duplicating it — access dialog() to add content or restyle the frame.
class Modal : public Widget {
public:
    Modal(const Font* font, std::string title, std::function<void()> onClose = {});

    UiColor backdropColor{0.0f, 0.0f, 0.0f, 0.55f};
    UiVec2 dialogSizePx{420.0f, 260.0f};
    // When true (default), input events are swallowed while open so widgets
    // behind the modal never see clicks intended for the dialog.
    bool blocking = true;

    // Center the dialog within this widget's current rect.
    void layout();

    void open();
    void close();
    [[nodiscard]] bool isOpen() const { return visible; }

    [[nodiscard]] Window& dialog() { return *dialog_; }
    [[nodiscard]] const Window& dialog() const { return *dialog_; }

    void draw(UiDrawList& dl) const override;
    bool onEvent(UiEvent& ev) override;

private:
    Window* dialog_ = nullptr;
    std::function<void()> onClose_;
};

}  // namespace odai::ui
