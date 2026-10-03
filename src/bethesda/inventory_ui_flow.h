#pragma once

#include "bethesda/bethesda_session.h"

#include <functional>
#include <string>
#include <vector>

namespace odai::bethesda {

// Device levels after the platform's normal key/gamepad mapping. The app and
// headless runner both feed this state once per UI frame.
struct InventoryUiInput {
    bool inventory = false;
    bool up = false;
    bool down = false;
    bool drop = false;
    bool cancel = false;
    bool upPressed = false;   // Includes normal nav auto-repeat pulses.
    bool downPressed = false;
};

struct InventoryUiEntry {
    RecordKey item;
    std::string label;
    std::int32_t quantity = 0;
    bool selected = false;
    bool enabled = true;
};

struct InventoryUiSnapshot {
    bool open = false;
    std::size_t selection = 0;
    std::vector<InventoryUiEntry> entries;
};

struct InventoryUiResult {
    bool opened = false;
    bool closed = false;
    bool dropped = false;
    RecordKey droppedItem;
    std::string error;
};

class InventoryUiFlow {
public:
    InventoryUiResult update(const InventoryUiInput& input, BethesdaSession& session,
        bool available = true);
    [[nodiscard]] InventoryUiSnapshot snapshot(const BethesdaSession& session,
        const std::function<std::string(const RecordKey&)>& label) const;
    [[nodiscard]] bool isOpen() const { return m_open; }
    [[nodiscard]] std::size_t selection() const { return m_selection; }
    void close() { m_open = false; }

private:
    InventoryUiInput m_previous{};
    bool m_open = false;
    std::size_t m_selection = 0;
};

} // namespace odai::bethesda
