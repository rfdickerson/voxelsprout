#pragma once

#include "ui/ui_input.h"
#include "ui/ui_types.h"

#include <memory>
#include <vector>

// Retained widget tree. Widgets own their children (explicit RAII ownership).
// Layout is absolute: each widget is given a rect; containers position children
// by assigning rects when the tree is built. Vertical/auto layout is a follow-on.
namespace odai::ui {

class UiDrawList;

struct UiEvent {
    enum class Type { MouseMove, MouseDown, MouseUp, Scroll, Text };
    Type type = Type::MouseMove;
    UiVec2 mousePx{};
    UiMouseButton button = UiMouseButton::Left;
    float scroll = 0.0f;
    std::uint32_t codepoint = 0;
    bool handled = false;
};

class Widget {
public:
    virtual ~Widget() = default;

    void setRect(const UiRect& rect) { rect_ = rect; }
    [[nodiscard]] const UiRect& rect() const { return rect_; }

    Widget* addChild(std::unique_ptr<Widget> child) {
        Widget* raw = child.get();
        children_.push_back(std::move(child));
        return raw;
    }

    // Draw self then children. Override to render a widget's own visuals (call
    // drawChildren() afterwards to keep descendants visible).
    virtual void draw(UiDrawList& drawList) const { drawChildren(drawList); }

    // Handle an event. Return true to consume it (stops propagation to siblings
    // below). Default: forward to children.
    virtual bool onEvent(UiEvent& event) {
        if (event.type == UiEvent::Type::MouseMove) {
            for (const std::unique_ptr<Widget>& child : children_) {
                if (child->visible) {
                    child->onEvent(event);
                }
            }
            return false;
        }
        return dispatchToChildren(event);
    }

    // Topmost visible widget containing the point, or nullptr.
    Widget* hitTest(const UiVec2& point) {
        if (!visible || !rect_.contains(point)) {
            return nullptr;
        }
        for (auto it = children_.rbegin(); it != children_.rend(); ++it) {
            if (Widget* hit = (*it)->hitTest(point)) {
                return hit;
            }
        }
        return this;
    }

    bool visible = true;

protected:
    void drawChildren(UiDrawList& drawList) const {
        for (const std::unique_ptr<Widget>& child : children_) {
            if (child->visible) {
                child->draw(drawList);
            }
        }
    }

    bool dispatchToChildren(UiEvent& event) {
        for (auto it = children_.rbegin(); it != children_.rend(); ++it) {
            if (!(*it)->visible) {
                continue;
            }
            if ((*it)->onEvent(event) || event.handled) {
                return true;
            }
        }
        return false;
    }

    UiRect rect_{};
    std::vector<std::unique_ptr<Widget>> children_;
};

}  // namespace odai::ui
