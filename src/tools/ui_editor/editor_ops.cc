#include "tools/ui_editor/editor_ops.h"

#include <algorithm>
#include <limits>

namespace odai::tools::ui_editor {

using json = nlohmann::json;

// ─── Selection ───────────────────────────────────────────────────────────────

bool Selection::contains(const NodePath& p) const {
    return std::find(m_paths.begin(), m_paths.end(), p) != m_paths.end();
}

void Selection::set(const NodePath& p) {
    m_paths.clear();
    m_paths.push_back(p);
}

void Selection::add(const NodePath& p) {
    if (!contains(p)) m_paths.push_back(p);
}

void Selection::remove(const NodePath& p) {
    m_paths.erase(std::remove(m_paths.begin(), m_paths.end(), p), m_paths.end());
}

void Selection::toggle(const NodePath& p) {
    if (contains(p)) {
        remove(p);
    } else {
        m_paths.push_back(p);
    }
}

void Selection::prune(const EditorDocument& doc) {
    m_paths.erase(std::remove_if(m_paths.begin(), m_paths.end(),
                                 [&](const NodePath& p) { return !doc.isValid(p); }),
                  m_paths.end());
}

// ─── Bounds ──────────────────────────────────────────────────────────────────

ui::UiRect selectionBounds(const EditorDocument& doc, const Selection& sel) {
    bool any = false;
    ui::UiRect out{std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
                   -std::numeric_limits<float>::max(), -std::numeric_limits<float>::max()};
    for (const NodePath& p : sel.paths()) {
        if (!doc.isValid(p)) continue;
        const ui::UiRect r = doc.absoluteRect(p);
        out.minX = std::min(out.minX, r.minX);
        out.minY = std::min(out.minY, r.minY);
        out.maxX = std::max(out.maxX, r.maxX);
        out.maxY = std::max(out.maxY, r.maxY);
        any = true;
    }
    return any ? out : ui::UiRect{};
}

// ─── Align / distribute / nudge ──────────────────────────────────────────────

bool alignSelection(EditorDocument& doc, const Selection& sel, AlignEdge edge) {
    if (sel.size() < 2) return false;
    const ui::UiRect bounds = selectionBounds(doc, sel);
    if (!bounds.valid()) return false;

    bool moved = false;
    for (const NodePath& p : sel.paths()) {
        if (!doc.isValid(p)) continue;
        const ui::UiRect r = doc.absoluteRect(p);
        float dx = 0.0f;
        float dy = 0.0f;
        switch (edge) {
            case AlignEdge::Left:    dx = bounds.minX - r.minX; break;
            case AlignEdge::Right:   dx = bounds.maxX - r.maxX; break;
            case AlignEdge::HCenter:
                dx = (bounds.minX + bounds.maxX) * 0.5f - (r.minX + r.maxX) * 0.5f;
                break;
            case AlignEdge::Top:     dy = bounds.minY - r.minY; break;
            case AlignEdge::Bottom:  dy = bounds.maxY - r.maxY; break;
            case AlignEdge::VCenter:
                dy = (bounds.minY + bounds.maxY) * 0.5f - (r.minY + r.maxY) * 0.5f;
                break;
        }
        if (dx == 0.0f && dy == 0.0f) continue;
        doc.translate(p, dx, dy);
        moved = true;
    }
    return moved;
}

bool distributeSelection(EditorDocument& doc, const Selection& sel, bool horizontal) {
    struct Item {
        NodePath path;
        ui::UiRect rect;
    };
    std::vector<Item> items;
    for (const NodePath& p : sel.paths()) {
        if (doc.isValid(p)) items.push_back(Item{p, doc.absoluteRect(p)});
    }
    if (items.size() < 3) return false;

    std::sort(items.begin(), items.end(), [&](const Item& a, const Item& b) {
        return horizontal ? a.rect.minX < b.rect.minX : a.rect.minY < b.rect.minY;
    });

    // Distribute by equal gaps between adjacent edges, holding the first and
    // last node in place. Equal-gap reads better than equal-centers when the
    // widgets have different sizes.
    float spanStart = horizontal ? items.front().rect.minX : items.front().rect.minY;
    float spanEnd = horizontal ? items.back().rect.maxX : items.back().rect.maxY;
    float usedExtent = 0.0f;
    for (const Item& it : items) {
        usedExtent += horizontal ? it.rect.width() : it.rect.height();
    }
    const float gap = (spanEnd - spanStart - usedExtent) /
                      static_cast<float>(items.size() - 1);

    bool moved = false;
    float cursor = spanStart;
    for (std::size_t i = 0; i < items.size(); ++i) {
        const Item& it = items[i];
        const float extent = horizontal ? it.rect.width() : it.rect.height();
        const float current = horizontal ? it.rect.minX : it.rect.minY;
        const float delta = cursor - current;
        if (i != 0 && i + 1 != items.size() && delta != 0.0f) {
            doc.translate(it.path, horizontal ? delta : 0.0f, horizontal ? 0.0f : delta);
            moved = true;
        }
        cursor += extent + gap;
    }
    return moved;
}

bool nudgeSelection(EditorDocument& doc, const Selection& sel, float dx, float dy) {
    if (dx == 0.0f && dy == 0.0f) return false;
    bool moved = false;
    for (const NodePath& p : sel.paths()) {
        if (!doc.isValid(p)) continue;
        if (doc.translate(p, dx, dy)) moved = true;
    }
    return moved;
}

// ─── Delete / clipboard ──────────────────────────────────────────────────────

// Deletion and duplication both need a removal order that keeps not-yet-visited
// paths valid: erasing a later sibling never shifts an earlier one, and erasing
// a node never shifts anything outside its own parent.
static std::vector<NodePath> deletionOrder(const Selection& sel) {
    std::vector<NodePath> paths = sel.paths();
    std::sort(paths.begin(), paths.end(), [](const NodePath& a, const NodePath& b) {
        return std::lexicographical_compare(b.begin(), b.end(), a.begin(), a.end());
    });
    return paths;
}

bool deleteSelection(EditorDocument& doc, Selection& sel) {
    bool removed = false;
    for (const NodePath& p : deletionOrder(sel)) {
        if (p.empty()) continue;  // never delete the root
        if (doc.removeNode(p)) removed = true;
    }
    sel.clear();
    return removed;
}

std::vector<json> copySelection(const EditorDocument& doc, const Selection& sel) {
    std::vector<json> out;
    for (const NodePath& p : sel.paths()) {
        if (p.empty()) continue;  // the root is not copyable as a child
        if (const json* n = doc.nodeAt(p)) out.push_back(*n);
    }
    return out;
}

bool pasteInto(EditorDocument& doc, const NodePath& parent, const std::vector<json>& clipboard,
               float dx, float dy, Selection& sel) {
    if (clipboard.empty()) return false;
    std::vector<NodePath> pasted;
    for (const json& node : clipboard) {
        NodePath out;
        if (!doc.insertNode(parent, node, -1, out)) continue;
        if (json* live = doc.nodeAt(out)) doc.freshenIds(*live);
        if (dx != 0.0f || dy != 0.0f) doc.translate(out, dx, dy);
        pasted.push_back(out);
    }
    if (pasted.empty()) return false;
    sel.setAll(std::move(pasted));
    return true;
}

bool duplicateSelection(EditorDocument& doc, Selection& sel, float dx, float dy) {
    // Duplicate into each node's own parent so a copy lands next to its original
    // rather than all copies collapsing into one container.
    std::vector<NodePath> created;
    for (const NodePath& p : sel.paths()) {
        if (p.empty()) continue;
        const json* src = doc.nodeAt(p);
        if (!src) continue;
        const json copy = *src;
        NodePath parent = p;
        parent.pop_back();
        NodePath out;
        if (!doc.insertNode(parent, copy, -1, out)) continue;
        if (json* live = doc.nodeAt(out)) doc.freshenIds(*live);
        doc.translate(out, dx, dy);
        created.push_back(out);
    }
    if (created.empty()) return false;
    sel.setAll(std::move(created));
    return true;
}

bool reorderSibling(EditorDocument& doc, const NodePath& path, int delta, NodePath& outPath) {
    if (path.empty() || delta == 0) return false;
    NodePath parent = path;
    const int index = parent.back();
    parent.pop_back();
    const int count = doc.childCount(parent);
    const int target = index + delta;
    if (target < 0 || target >= count) return false;
    return doc.reparent(path, parent, target, outPath);
}

}  // namespace odai::tools::ui_editor
