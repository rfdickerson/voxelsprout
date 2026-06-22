#include "ui/signal.h"
#include "ui/widget.h"

namespace odai::ui {

// Walk the widget tree depth-first. For each widget with a non-empty slotName,
// look up the slot in the registry and connect it to widget.activated.
void SlotRegistry::wire(Widget& root) const {
    if (!root.slotName.empty() && root.activated.empty()) {
        auto it = m_slots.find(root.slotName);
        if (it != m_slots.end()) {
            root.activated.connect(it->second);
        }
    }
    for (const auto& child : root.children()) {
        wire(*child.get());
    }
}

}  // namespace odai::ui
