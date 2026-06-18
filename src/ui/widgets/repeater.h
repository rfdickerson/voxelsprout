#pragma once

#include "ui/document/ui_binding.h"
#include "ui/widget.h"

#include <functional>
#include <memory>
#include <vector>

namespace odai::ui {

// Data-driven list widget. Given a list of DataNodes and an item factory,
// it builds one child Widget per item and stacks them vertically.
// Use inside a ScrollView for long lists.
//
// Example (C++):
//   auto rep = std::make_unique<Repeater>();
//   rep->setItemFactory([&](const DataNode& item, std::size_t idx) {
//       auto lbl = std::make_unique<Label>(&font, item.selfString());
//       return lbl;
//   });
//   rep->setItems(cityDataNode->getList("buildings"));
//
// Example (JSON document, when a UiDocumentLoader is wired up):
//   { "type": "Repeater", "items": "{city.buildings}", "template": "BuildingRow" }
class Repeater : public Widget {
public:
    using ItemFactory = std::function<std::unique_ptr<Widget>(
        const std::shared_ptr<DataNode>& item, std::size_t index)>;

    float itemGap = 4.0f;
    float itemHeight = 32.0f;  // Default item height; factory may override via setRect later.

    void setItemFactory(ItemFactory factory);

    // Rebuild children from the given data list.
    void setItems(std::vector<std::shared_ptr<DataNode>> items);

    void draw(UiDrawList& dl) const override;

private:
    ItemFactory m_factory;
    void rebuild();
};

}  // namespace odai::ui
