#include "tools/ui_editor/editor_history.h"

namespace odai::tools::ui_editor {

EditorHistory::EditorHistory(const EditorDocument& doc) { reset(doc); }

void EditorHistory::reset(const EditorDocument& doc) {
    m_entries.clear();
    m_entries.push_back(Entry{doc.toString(-1), "initial"});
    m_cursor = 0;
}

void EditorHistory::commit(const EditorDocument& doc, std::string label) {
    std::string state = doc.toString(-1);
    if (state == m_entries[m_cursor].state) return;  // nothing actually changed

    m_entries.resize(m_cursor + 1);
    m_entries.push_back(Entry{std::move(state), std::move(label)});

    if (m_entries.size() > kMaxEntries) {
        const std::size_t drop = m_entries.size() - kMaxEntries;
        m_entries.erase(m_entries.begin(), m_entries.begin() + static_cast<std::ptrdiff_t>(drop));
    }
    m_cursor = m_entries.size() - 1;
}

bool EditorHistory::apply(const Entry& e, EditorDocument& doc) {
    return doc.loadFromString(e.state);
}

bool EditorHistory::undo(EditorDocument& doc) {
    if (!canUndo()) return false;
    --m_cursor;
    return apply(m_entries[m_cursor], doc);
}

bool EditorHistory::redo(EditorDocument& doc) {
    if (!canRedo()) return false;
    ++m_cursor;
    return apply(m_entries[m_cursor], doc);
}

std::string EditorHistory::undoLabel() const {
    // The label describes the edit that *produced* the current state, so undoing
    // reverts that edit.
    return canUndo() ? m_entries[m_cursor].label : std::string();
}

std::string EditorHistory::redoLabel() const {
    return canRedo() ? m_entries[m_cursor + 1].label : std::string();
}

}  // namespace odai::tools::ui_editor
