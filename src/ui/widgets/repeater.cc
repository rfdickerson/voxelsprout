#include "ui/widgets/repeater.h"

#include "ui/ui_draw_list.h"

namespace odai::ui {

void Repeater::setItemFactory(ItemFactory factory) {
    m_factory = std::move(factory);
}

void Repeater::setItems(std::vector<std::shared_ptr<DataNode>> items) {
    children_.clear();
    if (!m_factory) return;
    for (std::size_t i = 0; i < items.size(); ++i) {
        auto child = m_factory(items[i], i);
        if (child) {
            // Give the child a default rect with the configured item height.
            child->setRect(UiRect::fromXYWH(rect_.minX, 0.0f, rect_.width(), itemHeight));
            addChild(std::move(child));
        }
    }
    // Update total height.
    const float total = children_.empty()
        ? 0.0f
        : static_cast<float>(children_.size()) * itemHeight
          + static_cast<float>(children_.size() - 1) * itemGap;
    setRect(UiRect::fromXYWH(rect_.minX, rect_.minY, rect_.width(), total));
}

void Repeater::draw(UiDrawList& dl) const {
    if (!visible) return;
    // Stack children vertically.
    float pen = rect_.minY;
    for (const std::unique_ptr<Widget>& child : children_) {
        if (!child->visible) continue;
        child->setRect(UiRect::fromXYWH(rect_.minX, pen, rect_.width(), child->rect().height()));
        pen += child->rect().height() + itemGap;
        child->draw(dl);
    }
}

}  // namespace odai::ui
