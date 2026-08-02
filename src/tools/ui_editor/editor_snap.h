#pragma once

#include "tools/ui_editor/editor_document.h"
#include "ui/ui_types.h"

#include <vector>

namespace odai::tools::ui_editor {

// Alignment guides produced by a snap. `horizontal` means a horizontal guide
// line (a shared Y), at `pos` in *document* space — the same coordinate space
// as EditorDocument::absoluteRect(), not screen pixels. The caller (the app's
// drawCanvas()) converts to screen space via docToScreen() before drawing.
struct SnapLine {
    bool horizontal = false;
    float pos = 0.0f;
};

struct SnapSettings {
    bool grid = true;
    float gridSize = 10.0f;
    bool edges = true;
    float threshold = 8.0f;
};

// Snapping against a live document: sibling/ancestor edges and centers plus an
// optional pixel grid. Pure geometry, no input or rendering.
class SnapEngine {
public:
    // Rebuilds the candidate edge lists from every node except `exclude` and its
    // descendants (a node never snaps to itself). Coordinates are canvas space.
    void collectTargets(const EditorDocument& doc, const std::vector<NodePath>& exclude);

    // Snaps a rect's position (size fixed). Returns the snapped top-left.
    ui::UiVec2 snapPosition(const ui::UiRect& rect, const SnapSettings& s);

    // Snaps a single dragged edge value (used while resizing). `yAxis` selects
    // which target list to snap against.
    float snapEdge(float value, bool yAxis, const SnapSettings& s);

    [[nodiscard]] const std::vector<SnapLine>& lines() const { return m_lines; }
    void clearLines() { m_lines.clear(); }

private:
    // Tries every anchor offset (near edge / center / far edge) of a span
    // against the target list and returns the best snapped anchor position.
    // `horizontalGuide` is the orientation of the guide line a hit records:
    // snapping an X value draws a vertical guide, snapping a Y value a
    // horizontal one.
    float snap1D(float value, float span, const std::vector<float>& targets, bool horizontalGuide,
                 const SnapSettings& s, bool useSpanOffsets);

    std::vector<float> m_xTargets;
    std::vector<float> m_yTargets;
    std::vector<SnapLine> m_lines;
};

}  // namespace odai::tools::ui_editor
