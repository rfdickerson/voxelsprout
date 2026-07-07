#include "ui/ui_context.h"

#include "ui/ui_draw_list.h"

namespace odai::ui {

void UiContext::setViewport(const UiVec2& sizePx) {
    viewport_ = sizePx;
    if (root_ != nullptr) {
        root_->setRect(UiRect{0.0f, 0.0f, sizePx.x, sizePx.y});
    }
}

Widget* UiContext::setRoot(std::unique_ptr<Widget> root) {
    root_ = std::move(root);
    if (root_ != nullptr) {
        root_->setRect(UiRect{0.0f, 0.0f, viewport_.x, viewport_.y});
    }
    return root_.get();
}

void UiContext::update(const UiInput& input) {
    if (root_ == nullptr) {
        wantsMouse_ = false;
        return;
    }

    UiEvent move{};
    move.type = UiEvent::Type::MouseMove;
    move.mousePx = input.mousePx;
    root_->onEvent(move);

    for (std::size_t b = 0; b < static_cast<std::size_t>(UiMouseButton::Count); ++b) {
        const auto button = static_cast<UiMouseButton>(b);
        const UiButtonState& state = input.button(button);
        if (state.pressed) {
            UiEvent down{};
            down.type = UiEvent::Type::MouseDown;
            down.mousePx = input.mousePx;
            down.button = button;
            root_->onEvent(down);
        }
        if (state.released) {
            UiEvent up{};
            up.type = UiEvent::Type::MouseUp;
            up.mousePx = input.mousePx;
            up.button = button;
            root_->onEvent(up);
            // A consumed left release is a widget activation (see Button::onEvent);
            // give the app a chance to play a click sound.
            if (up.handled && button == UiMouseButton::Left && clickFeedback_) {
                clickFeedback_();
            }
        }
    }

    if (input.scrollDelta != 0.0f) {
        UiEvent scroll{};
        scroll.type = UiEvent::Type::Scroll;
        scroll.mousePx = input.mousePx;
        scroll.scroll = input.scrollDelta;
        root_->onEvent(scroll);
    }

    for (std::uint32_t codepoint : input.textInput) {
        UiEvent text{};
        text.type = UiEvent::Type::Text;
        text.mousePx = input.mousePx;
        text.codepoint = codepoint;
        root_->onEvent(text);
    }

    wantsMouse_ = root_->hitTest(input.mousePx) != nullptr;
}

void UiContext::tick(float dt) {
    if (root_ != nullptr) {
        root_->onTick(dt);
    }
}

void UiContext::build(UiDrawList& drawList) const {
    drawList.reset(viewport_);
    buildAppend(drawList);
}

void UiContext::buildAppend(UiDrawList& drawList) const {
    if (root_ == nullptr || root_->opacity <= 0.0f || !root_->visible) {
        return;
    }
    if (root_->opacity < 1.0f) {
        drawList.pushOpacity(root_->opacity);
        root_->draw(drawList);
        drawList.popOpacity();
    } else {
        root_->draw(drawList);
    }
}

}  // namespace odai::ui
