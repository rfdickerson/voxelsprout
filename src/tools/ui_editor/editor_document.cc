#include "tools/ui_editor/editor_document.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <unordered_set>

namespace odai::tools::ui_editor {

using json = nlohmann::json;

// ─── Widget type schema ──────────────────────────────────────────────────────
// Mirrors the factories registered in ui/document/ui_document.cc. Anything the
// loader reads from a node should have a PropDesc here so the inspector can
// author it; anything it ignores must not be listed.

static std::vector<TypeDesc> buildTypeTable() {
    std::vector<TypeDesc> t;

    {
        TypeDesc d;
        d.name = "Panel";
        d.defaultWidth = 260;
        d.defaultHeight = 160;
        d.acceptsChildren = true;
        d.accent = {0.30f, 0.50f, 0.90f, 1.0f};
        d.props = {
            {"background", "Background", PropKind::Color},
            {"borderColor", "Border", PropKind::Color},
            {"frame", "9-Slice", PropKind::Frame},
            {"on_click", "On click", PropKind::Slot},
        };
        d.defaults = json{{"background", "#0D1A26E0"}, {"borderColor", "#D9B870A0"}};
        t.push_back(std::move(d));
    }
    {
        TypeDesc d;
        d.name = "Label";
        d.defaultWidth = 180;
        d.defaultHeight = 26;
        d.accent = {0.90f, 0.85f, 0.30f, 1.0f};
        d.props = {
            {"text", "Text", PropKind::Text},
            {"color", "Color", PropKind::Color},
            {"style", "Font role", PropKind::String},
        };
        d.defaults = json{{"text", "Label"}, {"color", "#E8E8F0FF"}};
        t.push_back(std::move(d));
    }
    {
        TypeDesc d;
        d.name = "Button";
        d.defaultWidth = 140;
        d.defaultHeight = 36;
        d.acceptsChildren = true;
        d.accent = {0.30f, 0.80f, 0.45f, 1.0f};
        d.props = {
            {"label", "Label", PropKind::Text},
            {"background", "Background", PropKind::Color},
            {"borderColor", "Border", PropKind::Color},
            {"labelColor", "Text", PropKind::Color},
            {"cornerRadius", "Corner", PropKind::Float, 0.0f, 32.0f},
            {"on_click", "On click", PropKind::Slot},
        };
        d.defaults = json{{"label", "Button"},
                          {"background", "#1A2E47E6"},
                          {"cornerRadius", 4.0f}};
        t.push_back(std::move(d));
    }
    {
        TypeDesc d;
        d.name = "ProgressBar";
        d.defaultWidth = 220;
        d.defaultHeight = 18;
        d.accent = {0.25f, 0.80f, 0.38f, 1.0f};
        d.props = {
            {"value", "Value", PropKind::Float, 0.0f, 1.0f},
            {"foreground", "Fill", PropKind::Color},
            {"background", "Track", PropKind::Color},
            {"frame", "9-Slice", PropKind::Frame},
        };
        d.defaults = json{{"value", 0.65f}};
        t.push_back(std::move(d));
    }
    {
        TypeDesc d;
        d.name = "HorizontalStack";
        d.defaultWidth = 300;
        d.defaultHeight = 40;
        d.acceptsChildren = true;
        d.laysOutChildren = true;
        d.accent = {0.55f, 0.35f, 0.90f, 1.0f};
        d.props = {{"gap", "Gap", PropKind::Float, 0.0f, 64.0f},
                   {"on_click", "On click", PropKind::Slot}};
        d.defaults = json{{"gap", 8.0f}};
        t.push_back(std::move(d));
    }
    {
        TypeDesc d;
        d.name = "VerticalStack";
        d.defaultWidth = 200;
        d.defaultHeight = 180;
        d.acceptsChildren = true;
        d.laysOutChildren = true;
        d.accent = {0.65f, 0.40f, 0.85f, 1.0f};
        d.props = {{"gap", "Gap", PropKind::Float, 0.0f, 64.0f},
                   {"on_click", "On click", PropKind::Slot}};
        d.defaults = json{{"gap", 8.0f}};
        t.push_back(std::move(d));
    }
    {
        TypeDesc d;
        d.name = "ScrollView";
        d.defaultWidth = 240;
        d.defaultHeight = 200;
        d.acceptsChildren = true;
        d.laysOutChildren = true;
        d.accent = {0.30f, 0.75f, 0.95f, 1.0f};
        d.props = {{"childGap", "Child gap", PropKind::Float, 0.0f, 64.0f}};
        d.defaults = json{{"childGap", 4.0f}};
        t.push_back(std::move(d));
    }
    {
        TypeDesc d;
        d.name = "Image";
        d.defaultWidth = 64;
        d.defaultHeight = 64;
        d.accent = {0.90f, 0.55f, 0.20f, 1.0f};
        d.props = {{"tint", "Tint", PropKind::Color}};
        d.defaults = json{{"tint", "#FFFFFFFF"}};
        t.push_back(std::move(d));
    }
    {
        TypeDesc d;
        d.name = "Repeater";
        d.defaultWidth = 240;
        d.defaultHeight = 180;
        d.accent = {0.20f, 0.85f, 0.72f, 1.0f};
        d.props = {{"itemHeight", "Item h", PropKind::Float, 1.0f, 256.0f},
                   {"itemGap", "Item gap", PropKind::Float, 0.0f, 64.0f}};
        d.defaults = json{{"itemHeight", 28.0f}, {"itemGap", 4.0f}};
        t.push_back(std::move(d));
    }
    {
        TypeDesc d;
        d.name = "Spacer";
        d.defaultWidth = 40;
        d.defaultHeight = 40;
        d.accent = {0.45f, 0.47f, 0.52f, 1.0f};
        t.push_back(std::move(d));
    }
    return t;
}

const std::vector<TypeDesc>& widgetTypes() {
    static const std::vector<TypeDesc> table = buildTypeTable();
    return table;
}

const TypeDesc* findWidgetType(std::string_view name) {
    for (const TypeDesc& d : widgetTypes()) {
        if (d.name == name) return &d;
    }
    return nullptr;
}

// ─── Lengths ─────────────────────────────────────────────────────────────────

json Length::toJson() const {
    if (!percent) return json(value);
    char buf[32];
    // Trim to at most 3 decimals but keep integral percentages clean ("50%").
    std::snprintf(buf, sizeof(buf), "%g%%", static_cast<double>(value));
    return json(std::string(buf));
}

Length parseLength(const json& val, Length fallback) {
    if (val.is_number()) return Length{val.get<float>(), false};
    if (val.is_string()) {
        const std::string s = val.get<std::string>();
        if (s.empty()) return fallback;
        try {
            if (s.back() == '%') return Length{std::stof(s.substr(0, s.size() - 1)), true};
            return Length{std::stof(s), false};
        } catch (...) {
            return fallback;
        }
    }
    return fallback;
}

// ─── Colors ──────────────────────────────────────────────────────────────────

ui::UiColor parseColorString(const std::string& hex, const ui::UiColor& fallback) {
    const char* s = hex.c_str();
    if (*s == '#') ++s;
    const std::size_t len = hex.size() - (hex[0] == '#' ? 1u : 0u);
    if (len != 6 && len != 8) return fallback;

    auto nibble = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
    };
    float channels[4] = {0.0f, 0.0f, 0.0f, 1.0f};
    for (std::size_t i = 0; i * 2 < len; ++i) {
        const int hi = nibble(s[i * 2]);
        const int lo = nibble(s[i * 2 + 1]);
        if (hi < 0 || lo < 0) return fallback;
        channels[i] = static_cast<float>((hi << 4) | lo) / 255.0f;
    }
    return ui::UiColor{channels[0], channels[1], channels[2], channels[3]};
}

std::string colorToHex(const ui::UiColor& c) {
    auto byte = [](float v) {
        const float clamped = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
        return static_cast<int>(clamped * 255.0f + 0.5f);
    };
    char buf[10];
    std::snprintf(buf, sizeof(buf), "#%02X%02X%02X%02X", byte(c.r), byte(c.g), byte(c.b),
                  byte(c.a));
    return buf;
}

// Reads a node's four layout fields. Absent width/height mean "fill parent",
// which resolveRect() expresses as 100%.
static void readLengths(const json& node, Length& x, Length& y, Length& w, Length& h) {
    x = node.contains("x") ? parseLength(node.at("x")) : Length{0.0f, false};
    y = node.contains("y") ? parseLength(node.at("y")) : Length{0.0f, false};
    w = node.contains("width") ? parseLength(node.at("width")) : Length{100.0f, true};
    h = node.contains("height") ? parseLength(node.at("height")) : Length{100.0f, true};
}

// ─── Node construction ───────────────────────────────────────────────────────

json makeNode(std::string_view type, float xPx, float yPx) {
    const TypeDesc* d = findWidgetType(type);
    json n;
    n["type"] = std::string(type);
    n["x"] = xPx;
    n["y"] = yPx;
    n["width"] = d ? d->defaultWidth : 200.0f;
    n["height"] = d ? d->defaultHeight : 80.0f;
    if (d && d->defaults.is_object()) {
        for (const auto& [key, value] : d->defaults.items()) n[key] = value;
    }
    return n;
}

// ─── EditorDocument ──────────────────────────────────────────────────────────

EditorDocument::EditorDocument() {
    m_root = json{
        {"version", 1},
        {"type", "Panel"},
        {"id", "root"},
        {"x", 0},
        {"y", 0},
        {"width", 1280},
        {"height", 720},
        {"background", "#141420FF"},
        {"children", json::array()},
    };
}

json* EditorDocument::childArray(json& node) {
    if (!node.is_object()) return nullptr;
    auto it = node.find("children");
    if (it == node.end() || !it->is_array()) return nullptr;
    return &(*it);
}

const json* EditorDocument::childArray(const json& node) {
    if (!node.is_object()) return nullptr;
    auto it = node.find("children");
    if (it == node.end() || !it->is_array()) return nullptr;
    return &(*it);
}

json* EditorDocument::nodeAt(const NodePath& path) {
    json* cur = &m_root;
    for (const int idx : path) {
        json* kids = childArray(*cur);
        if (!kids || idx < 0 || idx >= static_cast<int>(kids->size())) return nullptr;
        cur = &(*kids)[static_cast<std::size_t>(idx)];
    }
    return cur;
}

const json* EditorDocument::nodeAt(const NodePath& path) const {
    const json* cur = &m_root;
    for (const int idx : path) {
        const json* kids = childArray(*cur);
        if (!kids || idx < 0 || idx >= static_cast<int>(kids->size())) return nullptr;
        cur = &(*kids)[static_cast<std::size_t>(idx)];
    }
    return cur;
}

std::string EditorDocument::typeOf(const NodePath& path) const {
    const json* n = nodeAt(path);
    return n ? n->value("type", "Panel") : std::string();
}

std::string EditorDocument::idOf(const NodePath& path) const {
    const json* n = nodeAt(path);
    return n ? n->value("id", "") : std::string();
}

int EditorDocument::childCount(const NodePath& path) const {
    const json* n = nodeAt(path);
    if (!n) return 0;
    const json* kids = childArray(*n);
    return kids ? static_cast<int>(kids->size()) : 0;
}

static void collectPaths(const EditorDocument& doc, const NodePath& path,
                         std::vector<NodePath>& out) {
    out.push_back(path);
    const int n = doc.childCount(path);
    for (int i = 0; i < n; ++i) {
        NodePath child = path;
        child.push_back(i);
        collectPaths(doc, child, out);
    }
}

std::vector<NodePath> EditorDocument::allPaths() const {
    std::vector<NodePath> out;
    collectPaths(*this, NodePath{}, out);
    return out;
}

// ─── Serialization ───────────────────────────────────────────────────────────

bool EditorDocument::loadFromString(const std::string& text, std::string* error) {
    json doc;
    try {
        doc = json::parse(text);
    } catch (const std::exception& e) {
        if (error) *error = e.what();
        return false;
    }
    if (!doc.is_object()) {
        if (error) *error = "document root must be a JSON object";
        return false;
    }
    if (!doc.contains("children")) doc["children"] = json::array();
    m_root = std::move(doc);
    return true;
}

bool EditorDocument::loadFile(const std::filesystem::path& path, std::string* error) {
    std::ifstream f(path);
    if (!f) {
        if (error) *error = "cannot open " + path.string();
        return false;
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    return loadFromString(ss.str(), error);
}

std::string EditorDocument::toString(int indent) const { return m_root.dump(indent); }

bool EditorDocument::saveFile(const std::filesystem::path& path, std::string* error) const {
    std::ofstream f(path);
    if (!f) {
        if (error) *error = "cannot write " + path.string();
        return false;
    }
    f << toString(2) << '\n';
    if (!f) {
        if (error) *error = "write failed for " + path.string();
        return false;
    }
    return true;
}

// ─── Layout ──────────────────────────────────────────────────────────────────

ui::UiRect EditorDocument::absoluteRect(const NodePath& path) const {
    if (!nodeAt(path)) return ui::UiRect{};

    // Walk down from the root, resolving each level against its parent, exactly
    // as UiDocumentLoader::resolveRect does.
    ui::UiRect parent = m_viewport;
    const json* cur = &m_root;
    std::size_t depth = 0;
    for (;;) {
        Length x, y, w, h;
        readLengths(*cur, x, y, w, h);
        const float pw = parent.width();
        const float ph = parent.height();
        const ui::UiRect r = ui::UiRect::fromXYWH(parent.minX + x.resolve(pw),
                                                  parent.minY + y.resolve(ph),
                                                  w.resolve(pw), h.resolve(ph));
        if (depth == path.size()) return r;
        const json* kids = childArray(*cur);
        cur = &(*kids)[static_cast<std::size_t>(path[depth])];
        parent = r;
        ++depth;
    }
}

bool EditorDocument::parentControlsLayout(const NodePath& path) const {
    if (path.empty()) return false;
    NodePath parent = path;
    parent.pop_back();
    const TypeDesc* d = findWidgetType(typeOf(parent));
    return d && d->laysOutChildren;
}

bool EditorDocument::hitTest(float x, float y, NodePath& outPath) const {
    // Depth-first, last sibling first: later children draw on top.
    struct Rec {
        static bool visit(const EditorDocument& doc, const NodePath& path, float px, float py,
                          NodePath& out) {
            if (!doc.absoluteRect(path).contains(px, py)) return false;
            const int n = doc.childCount(path);
            for (int i = n - 1; i >= 0; --i) {
                NodePath child = path;
                child.push_back(i);
                if (visit(doc, child, px, py, out)) return true;
            }
            out = path;
            return true;
        }
    };
    return Rec::visit(*this, NodePath{}, x, y, outPath);
}

// ─── Mutation ────────────────────────────────────────────────────────────────

bool EditorDocument::insertNode(const NodePath& parent, const json& node, int index,
                                NodePath& outPath) {
    json* p = nodeAt(parent);
    if (!p || !p->is_object()) return false;
    const TypeDesc* pd = findWidgetType(p->value("type", "Panel"));
    // Unknown types are assumed to accept children — the editor should not block
    // authoring for an app-registered factory it has no schema for.
    if (pd && !pd->acceptsChildren) return false;

    if (!p->contains("children") || !(*p)["children"].is_array()) {
        (*p)["children"] = json::array();
    }
    json& kids = (*p)["children"];
    const int count = static_cast<int>(kids.size());
    const int at = (index < 0 || index > count) ? count : index;
    kids.insert(kids.begin() + at, node);

    outPath = parent;
    outPath.push_back(at);
    return true;
}

bool EditorDocument::addNode(const NodePath& parent, std::string_view type, float xPx, float yPx,
                             NodePath& outPath) {
    json node = makeNode(type, xPx, yPx);
    node["id"] = uniqueId(type);
    return insertNode(parent, node, -1, outPath);
}

bool EditorDocument::removeNode(const NodePath& path) {
    if (path.empty()) return false;  // the root is not removable
    NodePath parent = path;
    const int idx = parent.back();
    parent.pop_back();
    json* p = nodeAt(parent);
    if (!p) return false;
    json* kids = childArray(*p);
    if (!kids || idx < 0 || idx >= static_cast<int>(kids->size())) return false;
    kids->erase(kids->begin() + idx);
    return true;
}

// True when `maybeAncestor` is `path` itself or one of its ancestors.
static bool isPrefix(const NodePath& maybeAncestor, const NodePath& path) {
    if (maybeAncestor.size() > path.size()) return false;
    return std::equal(maybeAncestor.begin(), maybeAncestor.end(), path.begin());
}

bool EditorDocument::reparent(const NodePath& path, const NodePath& newParent, int index,
                              NodePath& outPath) {
    if (path.empty()) return false;                 // can't move the root
    if (isPrefix(path, newParent)) return false;    // would orphan the subtree
    const json* src = nodeAt(path);
    if (!src) return false;
    const json* dst = nodeAt(newParent);
    if (!dst) return false;
    const TypeDesc* pd = findWidgetType(dst->value("type", "Panel"));
    if (pd && !pd->acceptsChildren) return false;

    // Preserve the on-screen position across the reparent.
    const ui::UiRect abs = absoluteRect(path);
    json moved = *src;

    // `index` is the node's final index among the destination's children, which
    // for a remove-then-insert is also the insertion point: dropping the node out
    // of the list shifts exactly the siblings that sit after its final position.
    if (!removeNode(path)) return false;
    if (!insertNode(newParent, moved, index, outPath)) return false;
    setAbsoluteRect(outPath, abs);
    return true;
}

bool EditorDocument::setAbsoluteRect(const NodePath& path, const ui::UiRect& absRect) {
    json* n = nodeAt(path);
    if (!n) return false;

    ui::UiRect parentRect = m_viewport;
    if (!path.empty()) {
        NodePath parent = path;
        parent.pop_back();
        parentRect = absoluteRect(parent);
    }
    const float pw = parentRect.width();
    const float ph = parentRect.height();

    Length x, y, w, h;
    readLengths(*n, x, y, w, h);

    // Write each field back in the unit it was authored in, so a document that
    // uses "50%" stays percentage-based after a nudge instead of silently
    // freezing into pixels.
    auto store = [](Length unit, float px, float parentSpan) -> json {
        if (unit.percent && parentSpan > 0.0f) {
            return Length{px / parentSpan * 100.0f, true}.toJson();
        }
        return Length{px, false}.toJson();
    };

    (*n)["x"] = store(x, absRect.minX - parentRect.minX, pw);
    (*n)["y"] = store(y, absRect.minY - parentRect.minY, ph);
    (*n)["width"] = store(w, absRect.width(), pw);
    (*n)["height"] = store(h, absRect.height(), ph);
    return true;
}

bool EditorDocument::translate(const NodePath& path, float dxPx, float dyPx) {
    const ui::UiRect r = absoluteRect(path);
    if (!nodeAt(path)) return false;
    return setAbsoluteRect(path, ui::UiRect{r.minX + dxPx, r.minY + dyPx, r.maxX + dxPx,
                                            r.maxY + dyPx});
}

bool EditorDocument::setProperty(const NodePath& path, std::string_view key, const json& value) {
    json* n = nodeAt(path);
    if (!n || !n->is_object()) return false;
    (*n)[std::string(key)] = value;
    return true;
}

bool EditorDocument::eraseProperty(const NodePath& path, std::string_view key) {
    json* n = nodeAt(path);
    if (!n || !n->is_object()) return false;
    return n->erase(std::string(key)) > 0;
}

// ─── Ids ─────────────────────────────────────────────────────────────────────

void EditorDocument::collectIds(const json& node, std::vector<std::string>& out) const {
    if (!node.is_object()) return;
    if (node.contains("id") && node.at("id").is_string()) {
        out.push_back(node.at("id").get<std::string>());
    }
    const json* kids = childArray(node);
    if (!kids) return;
    for (const json& c : *kids) collectIds(c, out);
}

std::string EditorDocument::uniqueId(std::string_view desired) const {
    std::vector<std::string> ids;
    collectIds(m_root, ids);
    const std::unordered_set<std::string> taken(ids.begin(), ids.end());

    // Lower-case the type name so "Panel" seeds "panel", "panel2", ...
    std::string base(desired);
    if (!base.empty()) {
        base[0] = static_cast<char>(std::tolower(static_cast<unsigned char>(base[0])));
    }
    // Strip an existing numeric suffix so "panel3" re-bases to "panel".
    std::size_t cut = base.size();
    while (cut > 0 && std::isdigit(static_cast<unsigned char>(base[cut - 1]))) --cut;
    if (cut > 0) base.resize(cut);
    if (base.empty()) base = "widget";

    if (taken.find(base) == taken.end()) return base;
    for (int i = 2;; ++i) {
        std::string candidate = base + std::to_string(i);
        if (taken.find(candidate) == taken.end()) return candidate;
    }
}

void EditorDocument::freshenIds(json& subtree) {
    if (!subtree.is_object()) return;
    const std::string desired =
        subtree.contains("id") && subtree.at("id").is_string()
            ? subtree.at("id").get<std::string>()
            : subtree.value("type", std::string("widget"));
    // uniqueId() reads the live document, so assigning as we descend keeps later
    // siblings from colliding with ids handed out earlier in this same walk.
    subtree["id"] = uniqueId(desired);
    json* kids = childArray(subtree);
    if (!kids) return;
    for (json& c : *kids) freshenIds(c);
}

}  // namespace odai::tools::ui_editor
