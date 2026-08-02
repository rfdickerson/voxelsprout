#include "tools/ui_editor/editor_snap.h"

#include <algorithm>
#include <cmath>

namespace odai::tools::ui_editor {

static bool isPrefixOf(const NodePath& maybeAncestor, const NodePath& path) {
    if (maybeAncestor.size() > path.size()) return false;
    return std::equal(maybeAncestor.begin(), maybeAncestor.end(), path.begin());
}

void SnapEngine::collectTargets(const EditorDocument& doc, const std::vector<NodePath>& exclude) {
    m_xTargets.clear();
    m_yTargets.clear();

    for (const NodePath& path : doc.allPaths()) {
        const bool skip = std::any_of(exclude.begin(), exclude.end(),
                                      [&](const NodePath& e) { return isPrefixOf(e, path); });
        if (skip) continue;
        const ui::UiRect r = doc.absoluteRect(path);
        if (r.width() <= 0.0f && r.height() <= 0.0f) continue;
        m_xTargets.push_back(r.minX);
        m_xTargets.push_back((r.minX + r.maxX) * 0.5f);
        m_xTargets.push_back(r.maxX);
        m_yTargets.push_back(r.minY);
        m_yTargets.push_back((r.minY + r.maxY) * 0.5f);
        m_yTargets.push_back(r.maxY);
    }
}

float SnapEngine::snap1D(float value, float span, const std::vector<float>& targets,
                         bool horizontalGuide, const SnapSettings& s, bool useSpanOffsets) {
    float bestDist = s.threshold + 1.0f;
    float bestValue = value;
    float bestTarget = 0.0f;
    bool hit = false;

    const float offsets[3] = {0.0f, span * 0.5f, span};
    const int offsetCount = useSpanOffsets ? 3 : 1;

    auto consider = [&](float target, float offset) {
        const float snapped = target - offset;
        const float dist = std::abs(value - snapped);
        if (dist < bestDist) {
            bestDist = dist;
            bestValue = snapped;
            bestTarget = target;
            hit = true;
        }
    };

    // 1. Edge/center snap against other widgets.
    if (s.edges) {
        for (int i = 0; i < offsetCount; ++i) {
            for (const float t : targets) consider(t, offsets[i]);
        }
    }

    // 2. Grid snap, only when no edge snap landed inside the threshold. Grid hits
    //    draw no guide line — the grid is already visible on the canvas.
    if (s.grid && s.gridSize > 0.0f && bestDist > s.threshold) {
        float gridBest = value;
        float gridDist = s.threshold + 1.0f;
        for (int i = 0; i < offsetCount; ++i) {
            const float target = std::round((value + offsets[i]) / s.gridSize) * s.gridSize;
            const float snapped = target - offsets[i];
            const float dist = std::abs(value - snapped);
            if (dist < gridDist) {
                gridDist = dist;
                gridBest = snapped;
            }
        }
        if (gridDist <= s.threshold) return gridBest;
    }

    if (hit && bestDist <= s.threshold) {
        m_lines.push_back(SnapLine{horizontalGuide, bestTarget});
        return bestValue;
    }
    return value;
}

ui::UiVec2 SnapEngine::snapPosition(const ui::UiRect& rect, const SnapSettings& s) {
    m_lines.clear();
    const float x = snap1D(rect.minX, rect.width(), m_xTargets, /*horizontalGuide=*/false, s, true);
    const float y = snap1D(rect.minY, rect.height(), m_yTargets, /*horizontalGuide=*/true, s, true);
    return {x, y};
}

float SnapEngine::snapEdge(float value, bool yAxis, const SnapSettings& s) {
    return snap1D(value, 0.0f, yAxis ? m_yTargets : m_xTargets, /*horizontalGuide=*/yAxis, s,
                  /*useSpanOffsets=*/false);
}

}  // namespace odai::tools::ui_editor
