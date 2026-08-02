#pragma once

#include "tools/ui_editor/editor_document.h"

#include <cstddef>
#include <string>
#include <vector>

namespace odai::tools::ui_editor {

// Snapshot undo/redo for the editor document.
//
// A UI document is a few kilobytes of JSON, so whole-document snapshots are both
// simpler and more robust than per-field inverse commands — there is no way for
// an edit to be undoable-but-wrong, and new edit operations get undo for free
// without writing a matching undo path. If documents ever grow large enough for
// this to matter, the seam to swap in is commit()/undo()/redo() alone.
//
// Usage: mutate the document, then commit("label"). Continuous gestures (a drag,
// a resize) should commit once, on release, so the whole gesture undoes as one.
class EditorHistory {
public:
    // Seeds the baseline from the document's current state.
    explicit EditorHistory(const EditorDocument& doc);

    // Records the document's current state as a new undoable step. Discards any
    // redo entries beyond the cursor. A commit whose state matches the current
    // entry is dropped, so a no-op gesture doesn't leave a dead undo step.
    void commit(const EditorDocument& doc, std::string label);

    [[nodiscard]] bool canUndo() const { return m_cursor > 0; }
    [[nodiscard]] bool canRedo() const { return m_cursor + 1 < m_entries.size(); }

    // Restore the previous/next snapshot into `doc`. Returns false at the ends.
    bool undo(EditorDocument& doc);
    bool redo(EditorDocument& doc);

    // Label of the step undo() would revert / redo() would reapply ("" if none).
    [[nodiscard]] std::string undoLabel() const;
    [[nodiscard]] std::string redoLabel() const;

    [[nodiscard]] std::size_t size() const { return m_entries.size(); }

    // Forget everything and re-seed from `doc` (used after Load/New).
    void reset(const EditorDocument& doc);

    // Cap on retained snapshots; the oldest are dropped past this.
    static constexpr std::size_t kMaxEntries = 128;

private:
    struct Entry {
        std::string state;  // compact JSON dump
        std::string label;  // what produced this state
    };

    std::vector<Entry> m_entries;
    std::size_t m_cursor = 0;

    static bool apply(const Entry& e, EditorDocument& doc);
};

}  // namespace odai::tools::ui_editor
