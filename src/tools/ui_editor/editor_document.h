#pragma once

#include "ui/ui_types.h"

#include <nlohmann/json.hpp>

#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

// Headless (Vulkan-free, GLFW-free) document model for odai_ui_editor.
//
// The editor authors the *real* `.ui.json` schema that ui::UiDocumentLoader
// consumes — not a parallel toy format. The document is stored as the JSON tree
// itself, so fields the editor has no inspector for still survive a load→save
// round-trip untouched and the editor can never drift from the loader's schema.
//
// Coordinates mirror UiDocumentLoader::resolveRect exactly: a node's x/y/width/
// height are relative to its parent's rect, and each may be an absolute pixel
// number or a "50%" string resolved against the parent's span. Omitted width/
// height mean "fill the parent".
namespace odai::tools::ui_editor {

// Chain of child indices from the document root. Empty path == the root node.
using NodePath = std::vector<int>;

// ─── Widget type schema ──────────────────────────────────────────────────────
// Drives the palette and the property inspector so both stay in lockstep with
// ui_document.cc's registerBuiltins() without a switch statement per feature.

enum class PropKind {
    Float,    // plain number
    Bool,
    String,
    Color,    // "#RRGGBB" / "#RRGGBBAA" hex, or a theme color token for Panel bg
    Text,     // rich-text markup, may contain {binding} expressions
    Frame,    // named 9-slice from the theme, e.g. "Panel.Default"
    Slot,     // named slot in the app's SlotRegistry, e.g. "on_click"
};

struct PropDesc {
    std::string_view key;    // JSON key, e.g. "cornerRadius"
    std::string_view label;  // inspector label, e.g. "Corner"
    PropKind kind = PropKind::Float;
    float min = 0.0f;
    float max = 0.0f;  // max <= min means unbounded
};

struct TypeDesc {
    std::string_view name;
    float defaultWidth = 200.0f;
    float defaultHeight = 80.0f;
    bool acceptsChildren = false;  // loader calls buildChildren() for this type
    // True when the runtime container repositions its children itself (stacks,
    // scroll views). The editor cannot show a truthful canvas rect for such a
    // node's children, so it marks them instead of pretending.
    bool laysOutChildren = false;
    ui::UiColor accent{0.5f, 0.5f, 0.5f, 1.0f};  // palette swatch
    std::vector<PropDesc> props;
    nlohmann::json defaults;  // seed properties for a newly created node
};

// All widget types the editor can author, in palette order.
const std::vector<TypeDesc>& widgetTypes();

// Nullptr when the type is unknown (e.g. an app-registered custom factory that
// the editor has no schema for — such nodes still load, move and save fine).
const TypeDesc* findWidgetType(std::string_view name);

// ─── Lengths ─────────────────────────────────────────────────────────────────

// A coordinate field: either absolute pixels or a percentage of the parent span.
struct Length {
    float value = 0.0f;
    bool percent = false;

    [[nodiscard]] float resolve(float parentSpan) const {
        return percent ? parentSpan * value / 100.0f : value;
    }
    [[nodiscard]] nlohmann::json toJson() const;
};

// Parses a number or a "50%" string. Returns `fallback` for anything else.
Length parseLength(const nlohmann::json& val, Length fallback = {});

// ─── Colors ──────────────────────────────────────────────────────────────────

// "#RRGGBB" / "#RRGGBBAA", matching ui_document.cc's parser. Anything else —
// including a theme color token like "panel.bg", which only the theme can
// resolve — yields `fallback`.
ui::UiColor parseColorString(const std::string& hex, const ui::UiColor& fallback);

// Always emits the 8-digit form so alpha survives the round-trip.
std::string colorToHex(const ui::UiColor& c);

// ─── Document ────────────────────────────────────────────────────────────────

class EditorDocument {
public:
    // Starts with an empty root Panel sized to the design canvas.
    EditorDocument();

    // ── Serialization ────────────────────────────────────────────────────────
    // All four return false and leave the document untouched on failure.
    bool loadFromString(const std::string& text, std::string* error = nullptr);
    bool loadFile(const std::filesystem::path& path, std::string* error = nullptr);
    [[nodiscard]] std::string toString(int indent = 2) const;
    bool saveFile(const std::filesystem::path& path, std::string* error = nullptr) const;

    // ── Node access ──────────────────────────────────────────────────────────
    [[nodiscard]] const nlohmann::json& root() const { return m_root; }
    [[nodiscard]] nlohmann::json* nodeAt(const NodePath& path);
    [[nodiscard]] const nlohmann::json* nodeAt(const NodePath& path) const;
    [[nodiscard]] bool isValid(const NodePath& path) const { return nodeAt(path) != nullptr; }

    [[nodiscard]] std::string typeOf(const NodePath& path) const;
    [[nodiscard]] std::string idOf(const NodePath& path) const;
    [[nodiscard]] int childCount(const NodePath& path) const;

    // Depth-first walk of every node including the root, in draw order.
    [[nodiscard]] std::vector<NodePath> allPaths() const;

    // ── Layout ───────────────────────────────────────────────────────────────
    // The rect the root node is laid out against. Defaults to 1280x720; the app
    // sets it to the design canvas size.
    void setViewport(const ui::UiRect& r) { m_viewport = r; }
    [[nodiscard]] const ui::UiRect& viewport() const { return m_viewport; }

    // Absolute rect, resolving percentage lengths against each ancestor exactly
    // as UiDocumentLoader::resolveRect does. Invalid paths give an empty rect.
    // NOTE: for a child of a laysOutChildren container this is the *authored*
    // rect, not where the runtime stack will actually put it.
    [[nodiscard]] ui::UiRect absoluteRect(const NodePath& path) const;

    // True when this node's position is overridden by its parent at runtime.
    [[nodiscard]] bool parentControlsLayout(const NodePath& path) const;

    // Deepest node whose absolute rect contains the point, preferring later
    // siblings (drawn on top). Returns the root path when nothing else hits and
    // the root contains the point; empty optional-equivalent is signalled by
    // `hit == false`.
    [[nodiscard]] bool hitTest(float x, float y, NodePath& outPath) const;

    // ── Mutation ─────────────────────────────────────────────────────────────
    // Appends a default-initialized node of `type` under `parent`, positioned at
    // (xPx, yPx) in the parent's local pixel space. Returns the new node's path;
    // an empty path plus false when the parent is invalid or can't hold children.
    bool addNode(const NodePath& parent, std::string_view type, float xPx, float yPx,
                 NodePath& outPath);

    // Inserts an existing subtree (e.g. from the clipboard). index < 0 appends.
    bool insertNode(const NodePath& parent, const nlohmann::json& node, int index,
                    NodePath& outPath);

    bool removeNode(const NodePath& path);

    // Moves `path` to be child `index` of `newParent`, where `index` is the
    // node's *final* position in the destination's child list (index < 0
    // appends). The on-screen rect is preserved across the move. Refuses to move
    // the root or to reparent a node into its own descendant.
    bool reparent(const NodePath& path, const NodePath& newParent, int index,
                  NodePath& outPath);

    // Sets the node's rect from absolute pixel coordinates, converting back to
    // the parent-relative (and percentage-preserving) form the schema uses.
    bool setAbsoluteRect(const NodePath& path, const ui::UiRect& absRect);

    // Translates by a pixel delta, preserving percentage authoring where used.
    bool translate(const NodePath& path, float dxPx, float dyPx);

    // Generic property set/erase on a node (used by the inspector).
    bool setProperty(const NodePath& path, std::string_view key, const nlohmann::json& value);
    bool eraseProperty(const NodePath& path, std::string_view key);

    // Ensures `id` is unique across the document by appending/incrementing a
    // numeric suffix. Also used when pasting.
    [[nodiscard]] std::string uniqueId(std::string_view desired) const;

    // Rewrites every id in the subtree so a pasted copy doesn't collide. Call it
    // on a node that is *already* in the document (e.g. via nodeAt(pastedPath)):
    // uniqueId() scans the live tree, so ids handed out earlier in this walk are
    // visible to later ones only once the subtree is attached.
    void freshenIds(nlohmann::json& subtree);

private:
    nlohmann::json m_root;
    ui::UiRect m_viewport{0.0f, 0.0f, 1280.0f, 720.0f};

    void collectIds(const nlohmann::json& node, std::vector<std::string>& out) const;
    static nlohmann::json* childArray(nlohmann::json& node);
    static const nlohmann::json* childArray(const nlohmann::json& node);
};

// Builds a fresh node of the given type with its schema defaults applied.
nlohmann::json makeNode(std::string_view type, float xPx, float yPx);

}  // namespace odai::tools::ui_editor
