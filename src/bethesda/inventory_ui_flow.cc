#include "bethesda/inventory_ui_flow.h"

#include <algorithm>

namespace odai::bethesda {

InventoryUiResult InventoryUiFlow::update(const InventoryUiInput& input,
    BethesdaSession& session, bool available) {
    InventoryUiResult result;
    const bool openEdge = input.inventory && !m_previous.inventory;
    const bool upEdge = input.upPressed || (input.up && !m_previous.up);
    const bool downEdge = input.downPressed || (input.down && !m_previous.down);
    const bool dropEdge = input.drop && !m_previous.drop;
    const bool cancelEdge = input.cancel && !m_previous.cancel;
    m_previous = input;

    if (!available) return result;
    if (openEdge) {
        m_open = !m_open;
        m_selection = 0;
        result.opened = m_open;
        result.closed = !m_open;
    }
    if (!m_open) return result;

    const RuntimeObject* player = session.world().find(session.playerObject());
    const std::size_t count = player ? player->inventory.size() : 0;
    m_selection = count == 0 ? 0 : std::min(m_selection, count - 1);
    if (count > 0 && upEdge) m_selection = (m_selection + count - 1) % count;
    if (count > 0 && downEdge) m_selection = (m_selection + 1) % count;

    if (count > 0 && dropEdge) {
        const RecordKey item = player->inventory[m_selection].item;
        if (session.dropInventoryItem(session.playerObject(), item, result.error, player->inventory[m_selection].condition)) {
            result.dropped = true;
            result.droppedItem = item;
            result.closed = true;
            m_open = false;
        }
    }
    if (cancelEdge && m_open) {
        m_open = false;
        result.closed = true;
    }
    return result;
}

InventoryUiSnapshot InventoryUiFlow::snapshot(const BethesdaSession& session,
    const std::function<std::string(const RecordKey&)>& label) const {
    InventoryUiSnapshot result;
    result.open = m_open;
    result.selection = m_selection;
    if (!m_open) return result;
    const RuntimeObject* player = session.world().find(session.playerObject());
    if (!player) return result;
    for (std::size_t i = 0; i < player->inventory.size(); ++i) {
        const InventoryEntry& item = player->inventory[i];
        if (item.count <= 0) continue;
        result.entries.push_back({item.item, label(item.item), item.count,
            i == m_selection, true});
    }
    return result;
}

} // namespace odai::bethesda
