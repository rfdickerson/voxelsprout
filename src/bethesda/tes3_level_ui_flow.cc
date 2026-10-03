#include "bethesda/tes3_level_ui_flow.h"
#include "bethesda/bethesda_session.h"
#include <algorithm>

namespace odai::bethesda {
bool Tes3LevelUiFlow::update(const Tes3LevelUiInput& input, BethesdaSession& session, std::string& error) {
    error.clear();
    if (!session.tes3().playerState().progression.selectionOpen) { reset(); return false; }
    if (!m_open) { reset(); m_open = true; }
    if (input.up) m_cursor = (m_cursor + 8) % 9;
    if (input.down) m_cursor = (m_cursor + 1) % 9;
    if (!input.accept) return false;
    if (m_cursor == 8) {
        if (!session.confirmTes3PlayerLevel(m_selected, error)) return false;
        reset(); return true;
    }
    const auto found = std::find(m_selected.begin(), m_selected.end(), m_cursor);
    if (found != m_selected.end()) m_selected.erase(found);
    else if (session.tes3().playerAttributeGain(m_cursor) > 0) {
        if (static_cast<int>(m_selected.size()) == session.tes3().levelChoiceCount()) m_selected.pop_back();
        m_selected.push_back(m_cursor);
    }
    return false;
}
}
