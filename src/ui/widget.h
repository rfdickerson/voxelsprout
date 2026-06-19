#pragma once

#include "ui/ui_draw_list.h"
#include "ui/ui_input.h"
#include "ui/ui_types.h"

#include <algorithm>
#include <memory>
#include <vector>

// Retained widget tree. Widgets own their children (explicit RAII ownership).
// Layout is absolute: each widget is given a rect; containers position children
// by assigning rects when the tree is built. Vertical/auto layout is a follow-on.
namespace odai::ui {

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

    // Shift this widget and all descendants by (dx, dy). Used to move a window and
    // its content together when the title bar is dragged.
    void translate(float dx, float dy) {
        rect_ = UiRect{rect_.minX + dx, rect_.minY + dy, rect_.maxX + dx, rect_.maxY + dy};
        for (const std::unique_ptr<Widget>& child : children_) {
            child->translate(dx, dy);
        }
    }

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
    float opacity = 1.0f;  // Drives fades; multiplies this widget's drawn alpha.
    // Drawing and event-dispatch order within a parent's child list. Children with
    // higher zOrder are drawn on top and receive input events first. Within the same
    // zOrder, insertion order is preserved (later-added children drawn last / on top).
    int zOrder = 0;

protected:
    void drawChildren(UiDrawList& drawList) const {
        // Collect visible children into a stable-sorted pointer list (ascending zOrder).
        // Lowest z draws first, highest z draws last (appears on top).
        std::vector<const Widget*> sorted;
        sorted.reserve(children_.size());
        for (const auto& child : children_) {
            if (child->visible && child->opacity > 0.0f) {
                sorted.push_back(child.get());
            }
        }
        std::stable_sort(sorted.begin(), sorted.end(), [](const Widget* a, const Widget* b) {
            return a->zOrder < b->zOrder;
        });
        for (const Widget* child : sorted) {
            const bool fade = child->opacity < 1.0f;
            if (fade) drawList.pushOpacity(child->opacity);
            child->draw(drawList);
            if (fade) drawList.popOpacity();
        }
    }

    bool dispatchToChildren(UiEvent& event) {
        // Dispatch to highest-z children first; within same z, reverse insertion order
        // so later-added (visually on top) children consume events before earlier ones.
        std::vector<Widget*> sorted;
        sorted.reserve(children_.size());
        for (const auto& child : children_) {
            if (child->visible) sorted.push_back(child.get());
        }
        std::stable_sort(sorted.begin(), sorted.end(), [](const Widget* a, const Widget* b) {
            return a->zOrder > b->zOrder;  // descending — highest z first
        });
        for (Widget* child : sorted) {
            if (child->onEvent(event) || event.handled) return true;
        }
        return false;
    }

    UiRect rect_{};
    std::vector<std::unique_ptr<Widget>> children_;
};

}  // namespace odai::ui
