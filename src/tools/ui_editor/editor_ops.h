#pragma once

#include "tools/ui_editor/editor_document.h"
#include "ui/ui_types.h"

#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace odai::tools::ui_editor {

// Multi-selection over an EditorDocument, plus the standard editor operations
// that act on one (align, distribute, nudge, duplicate, clipboard).
//
// Paths are positional, so any structural edit invalidates them; every mutating
// helper here returns the selection it leaves behind rather than trying to keep
// stale paths alive.

class Selection {
public:
    void clear() { m_paths.clear(); }
    [[nodiscard]] bool empty() const { return m_paths.empty(); }
    [[nodiscard]] std::size_t size() const { return m_paths.size(); }
    [[nodiscard]] const std::vector<NodePath>& paths() const { return m_paths; }

    // The most recently added path — the anchor for align-to-last operations.
    [[nodiscard]] const NodePath* primary() const {
        return m_paths.empty() ? nullptr : &m_paths.back();
    }

    [[nodiscard]] bool contains(const NodePath& p) const;

    void set(const NodePath& p);
    void add(const NodePath& p);
    void remove(const NodePath& p);
    void toggle(const NodePath& p);
    void setAll(std::vector<NodePath> paths) { m_paths = std::move(paths); }

    // Drops paths that no longer resolve (after an undo, a delete, a reload).
    void prune(const EditorDocument& doc);

private:
    std::vector<NodePath> m_paths;
};

// Union of the selected nodes' absolute rects. Empty rect when nothing valid.
ui::UiRect selectionBounds(const EditorDocument& doc, const Selection& sel);

enum class AlignEdge { Left, HCenter, Right, Top, VCenter, Bottom };

// Aligns every selected node to the corresponding edge of the selection's
// bounding box. No-op below two nodes. Returns true if anything moved.
bool alignSelection(EditorDocument& doc, const Selection& sel, AlignEdge edge);

// Spaces the selected nodes so the gaps between them are equal along one axis,
// keeping the two extreme nodes fixed. Needs at least three nodes.
bool distributeSelection(EditorDocument& doc, const Selection& sel, bool horizontal);

// Moves every selected node by a pixel delta (keyboard nudge).
bool nudgeSelection(EditorDocument& doc, const Selection& sel, float dx, float dy);

// Deletes the selected nodes. Removes deepest/last-index first so earlier paths
// stay valid through the sweep. The root is skipped.
bool deleteSelection(EditorDocument& doc, Selection& sel);

// Copies the selected subtrees out of the document, deepest-last order preserved.
std::vector<nlohmann::json> copySelection(const EditorDocument& doc, const Selection& sel);

// Pastes clipboard subtrees under `parent`, offset by (dx, dy) pixels, with all
// ids freshened. Leaves the pasted nodes selected. Returns false if nothing was
// pasted.
bool pasteInto(EditorDocument& doc, const NodePath& parent,
               const std::vector<nlohmann::json>& clipboard, float dx, float dy, Selection& sel);

// Copy + paste-in-place-offset of the current selection, into each node's own
// existing parent.
bool duplicateSelection(EditorDocument& doc, Selection& sel, float dx, float dy);

// Reorders a node among its siblings. delta > 0 moves it later (further front).
bool reorderSibling(EditorDocument& doc, const NodePath& path, int delta, NodePath& outPath);

}  // namespace odai::tools::ui_editor
