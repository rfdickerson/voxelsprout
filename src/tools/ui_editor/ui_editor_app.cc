#include "tools/ui_editor/ui_editor_app.h"

#include "ui/widgets/button.h"
#include "ui/widgets/panel.h"
#include "ui/widget.h"

#include <nlohmann/json.hpp>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>

namespace odai::tools::ui_editor {

using namespace ui;
using json = nlohmann::json;

// ─── Layout constants ────────────────────────────────────────────────────────
static constexpr float kPaletteW = 200.0f;
static constexpr float kPropsW = 296.0f;
static constexpr float kToolbarH = 46.0f;
static constexpr float kStatusH = 24.0f;
static constexpr float kPaletteRowH = 30.0f;
static constexpr float kOutlinerRowH = 20.0f;

// ─── Colours ─────────────────────────────────────────────────────────────────
static constexpr UiColor kEditorBg{0.100f, 0.102f, 0.110f, 1.000f};
static constexpr UiColor kCanvasBg{0.140f, 0.142f, 0.158f, 1.000f};
static constexpr UiColor kSurfaceBg{0.085f, 0.088f, 0.100f, 1.000f};
static constexpr UiColor kGridLine{0.175f, 0.178f, 0.196f, 1.000f};
static constexpr UiColor kGridMajor{0.230f, 0.234f, 0.258f, 1.000f};
static constexpr UiColor kPanelBg{0.068f, 0.072f, 0.090f, 0.980f};
static constexpr UiColor kPanelBord{0.140f, 0.145f, 0.190f, 1.000f};
static constexpr UiColor kBlue{0.200f, 0.600f, 1.000f, 1.000f};
static constexpr UiColor kSnapLine{0.000f, 0.780f, 0.900f, 0.600f};
static constexpr UiColor kText{0.820f, 0.840f, 0.880f, 1.000f};
static constexpr UiColor kTextDim{0.500f, 0.520f, 0.570f, 1.000f};
static constexpr UiColor kGold{0.900f, 0.720f, 0.380f, 1.000f};
static constexpr UiColor kDeleteRed{0.800f, 0.200f, 0.180f, 1.000f};
static constexpr UiColor kManagedTag{0.850f, 0.550f, 0.200f, 0.900f};

static constexpr UiColor kColorPresets[] = {
    {0.05f, 0.10f, 0.15f, 0.88f}, {0.12f, 0.08f, 0.05f, 0.88f},
    {0.05f, 0.12f, 0.06f, 0.88f}, {0.08f, 0.05f, 0.14f, 0.88f},
    {0.14f, 0.06f, 0.06f, 0.88f}, {0.08f, 0.12f, 0.16f, 0.88f},
    {0.85f, 0.87f, 0.92f, 1.00f}, {0.90f, 0.72f, 0.38f, 1.00f},
};
static constexpr int kNumColorPresets = 8;

// Where Save/Load look by default: the same folder the engine's shipped sample
// document lives in, so the editor opens real content out of the box.
static constexpr const char* kDefaultDocument = "assets/ui/docs/city_panel.ui.json";

// ─── Small helpers ───────────────────────────────────────────────────────────

static UiColor withAlpha(const UiColor& c, float a) { return UiColor{c.r, c.g, c.b, a}; }

// Bottom edge of the palette section of the left column. The palette is a real
// widget-tree Panel (so its buttons get hover states) and the outliner below it
// is drawn immediate-mode; the Panel must stop here or it would be flushed over
// the outliner, since the widget tree is appended to the draw list last.
static float paletteBottom() {
    return kToolbarH + 26.0f + kPaletteRowH * static_cast<float>(widgetTypes().size()) + 6.0f;
}

// Reads a node property as a color. Theme tokens ("panel.bg") are not resolvable
// without a loaded theme, so they render as a neutral placeholder — the editor
// still round-trips the token untouched.
static UiColor nodeColor(const json& node, const char* key, const UiColor& fallback) {
    if (!node.contains(key) || !node.at(key).is_string()) return fallback;
    const std::string s = node.at(key).get<std::string>();
    if (s.empty() || s[0] != '#') return UiColor{0.35f, 0.33f, 0.40f, 0.85f};
    return parseColorString(s, fallback);
}

static float nodeFloat(const json& node, const char* key, float fallback) {
    if (!node.contains(key) || !node.at(key).is_number()) return fallback;
    return node.at(key).get<float>();
}

static std::string nodeString(const json& node, const char* key) {
    if (!node.contains(key)) return {};
    const json& v = node.at(key);
    if (v.is_string()) return v.get<std::string>();
    return v.dump();
}

// ─── Coordinate spaces ───────────────────────────────────────────────────────

UiVec2 UiEditorApp::docToScreen(float x, float y) const {
    return {m_canvasX + m_docOriginX + x, m_canvasY + m_docOriginY + y};
}

UiVec2 UiEditorApp::screenToDoc(float x, float y) const {
    return {x - m_canvasX - m_docOriginX, y - m_canvasY - m_docOriginY};
}

UiRect UiEditorApp::docRectToScreen(const UiRect& r) const {
    const UiVec2 tl = docToScreen(r.minX, r.minY);
    return UiRect::fromXYWH(tl.x, tl.y, r.width(), r.height());
}

void UiEditorApp::layoutPanes(int fbW, int fbH) {
    m_canvasX = kPaletteW;
    m_canvasY = kToolbarH;
    m_canvasW = std::max(1.0f, static_cast<float>(fbW) - kPaletteW - kPropsW);
    m_canvasH = std::max(1.0f, static_cast<float>(fbH) - kToolbarH - kStatusH);

    // The design surface is whatever the root node declares. Centre it in the
    // canvas pane when it fits; pin it to the top-left when it doesn't, so the
    // origin stays reachable instead of scrolling off-screen.
    const json& root = m_doc.root();
    m_docW = std::max(1.0f, nodeFloat(root, "width", 1280.0f));
    m_docH = std::max(1.0f, nodeFloat(root, "height", 720.0f));
    m_doc.setViewport(UiRect::fromXYWH(0.0f, 0.0f, m_docW, m_docH));

    m_docOriginX = std::max(0.0f, (m_canvasW - m_docW) * 0.5f);
    m_docOriginY = std::max(0.0f, (m_canvasH - m_docH) * 0.5f);
}

// ─── Canvas ──────────────────────────────────────────────────────────────────

void UiEditorApp::drawNode(const NodePath& path) {
    const json* node = m_doc.nodeAt(path);
    if (!node) return;

    const std::string type = m_doc.typeOf(path);
    const TypeDesc* desc = findWidgetType(type);
    const UiRect r = docRectToScreen(m_doc.absoluteRect(path));
    const float radius = nodeFloat(*node, "cornerRadius", 0.0f);

    // Body. An unknown type (an app-registered custom factory) gets a neutral
    // placeholder rather than being hidden.
    const UiColor accent = desc ? desc->accent : UiColor{0.55f, 0.55f, 0.60f, 1.0f};
    const UiColor bg = nodeColor(*node, "background", withAlpha(accent, 0.16f));
    const UiColor border = nodeColor(*node, "borderColor", withAlpha(accent, 0.45f));

    if (type != "Spacer") {
        if (radius > 0.5f) {
            m_uiDrawList.addRoundRectFilled(r, bg, radius);
            m_uiDrawList.addRoundRect(r, border, radius, 1.0f);
        } else {
            m_uiDrawList.addRectFilled(r, bg);
            m_uiDrawList.addRect(r, border, 1.0f);
        }
    } else {
        // A Spacer draws nothing at runtime; show it as an outline only.
        m_uiDrawList.addRect(r, withAlpha(accent, 0.5f), 1.0f);
    }

    // Type-specific affordances so the canvas reads as a layout, not a box list.
    if (type == "ProgressBar") {
        const float v = std::clamp(nodeFloat(*node, "value", 0.5f), 0.0f, 1.0f);
        const UiColor fg = nodeColor(*node, "foreground", UiColor{0.18f, 0.78f, 0.32f, 1.0f});
        if (r.width() > 4.0f && r.height() > 4.0f && v > 0.0f) {
            m_uiDrawList.addRoundRectFilled(
                UiRect::fromXYWH(r.minX + 2, r.minY + 2, (r.width() - 4) * v, r.height() - 4), fg,
                2.0f);
        }
    } else if (type == "Image") {
        const UiColor tint = nodeColor(*node, "tint", UiColor{1, 1, 1, 1});
        const float pad = std::min(r.width(), r.height()) * 0.22f;
        m_uiDrawList.addCircleFilled({(r.minX + r.maxX) * 0.5f, (r.minY + r.maxY) * 0.5f},
                                     std::max(2.0f, pad), withAlpha(tint, 0.7f));
    } else if (type == "Repeater") {
        const float itemH = std::max(4.0f, nodeFloat(*node, "itemHeight", 28.0f));
        const float itemGap = nodeFloat(*node, "itemGap", 4.0f);
        float y = r.minY + 4;
        while (y + itemH < r.maxY - 2) {
            m_uiDrawList.addRectFilled(UiRect::fromXYWH(r.minX + 4, y, r.width() - 8, itemH),
                                       withAlpha(accent, 0.18f));
            y += itemH + itemGap;
        }
    }

    // Label text: whichever of text/label this type actually uses.
    if (m_uiFonts.regular) {
        std::string caption = nodeString(*node, "text");
        if (caption.empty()) caption = nodeString(*node, "label");
        const bool isCaption = !caption.empty();
        if (caption.empty()) caption = m_doc.idOf(path);
        if (caption.empty()) caption = type;

        const float tw = m_uiFonts.regular->measureText(caption);
        const float lh = m_uiFonts.regular->lineHeightPx();
        if (tw < r.width() - 6.0f && lh < r.height() - 2.0f) {
            m_uiDrawList.addText(*m_uiFonts.regular, caption,
                                 {r.minX + (r.width() - tw) * 0.5f,
                                  r.minY + (r.height() - lh) * 0.5f},
                                 isCaption ? UiColor{0.92f, 0.93f, 0.96f, 0.92f}
                                           : UiColor{0.90f, 0.90f, 0.90f, 0.42f});
        }
    }

    // A stack/scroll container repositions its children at runtime, so the rect
    // the editor shows for them is the authored one, not where they will land.
    // Tag the container so that is visible rather than a silent lie.
    if (desc && desc->laysOutChildren && m_uiFonts.regular && r.width() > 46.0f) {
        m_uiDrawList.addText(*m_uiFonts.regular, "auto", {r.minX + 3.0f, r.minY + 1.0f},
                             kManagedTag);
    }

    // Children, clipped to the container like the runtime would.
    const int count = m_doc.childCount(path);
    if (count > 0) {
        m_uiDrawList.pushClip(r);
        for (int i = 0; i < count; ++i) {
            NodePath child = path;
            child.push_back(i);
            drawNode(child);
        }
        m_uiDrawList.popClip();
    }
}

void UiEditorApp::drawSelectionOverlay() {
    for (const NodePath& path : m_selection.paths()) {
        if (!m_doc.isValid(path)) continue;
        const UiRect r = docRectToScreen(m_doc.absoluteRect(path));
        const bool primary = (m_selection.primary() && *m_selection.primary() == path);
        m_uiDrawList.addRect(UiRect{r.minX - 2, r.minY - 2, r.maxX + 2, r.maxY + 2},
                             primary ? kBlue : withAlpha(kBlue, 0.45f), primary ? 2.0f : 1.0f);
        if (m_doc.parentControlsLayout(path)) {
            m_uiDrawList.addRect(UiRect{r.minX - 4, r.minY - 4, r.maxX + 4, r.maxY + 4},
                                 withAlpha(kManagedTag, 0.5f), 1.0f);
        }
    }

    // Resize handles on the primary selection only — dragging a handle with a
    // multi-selection would need a group transform, which this slice doesn't do.
    const NodePath* primary = m_selection.primary();
    if (!primary || !m_doc.isValid(*primary) || m_selection.size() != 1) return;
    const UiRect r = docRectToScreen(m_doc.absoluteRect(*primary));
    const float mx = (r.minX + r.maxX) * 0.5f;
    const float my = (r.minY + r.maxY) * 0.5f;
    const UiVec2 handles[8] = {{r.minX, r.minY}, {mx, r.minY},     {r.maxX, r.minY},
                               {r.maxX, my},     {r.maxX, r.maxY}, {mx, r.maxY},
                               {r.minX, r.maxY}, {r.minX, my}};
    constexpr float hs = 6.0f;
    for (const UiVec2& h : handles) {
        const UiRect hr = UiRect::fromXYWH(h.x - hs * 0.5f, h.y - hs * 0.5f, hs, hs);
        m_uiDrawList.addRectFilled(hr, {0.0f, 0.0f, 0.0f, 1.0f});
        m_uiDrawList.addRect(hr, kBlue, 1.5f);
    }
}

void UiEditorApp::drawPlacementGhost() {
    if (!m_placing) return;
    const UiVec2 d = screenToDoc(m_uiInput.mousePx.x, m_uiInput.mousePx.y);
    const TypeDesc* desc = findWidgetType(m_placeType);
    if (!desc) return;
    const float gw = desc->defaultWidth;
    const float gh = desc->defaultHeight;
    const UiRect ghost = docRectToScreen(UiRect::fromXYWH(d.x - gw * 0.5f, d.y - gh * 0.5f, gw, gh));
    m_uiDrawList.addRectFilled(ghost, withAlpha(desc->accent, 0.20f));
    m_uiDrawList.addRect(ghost, withAlpha(desc->accent, 0.75f), 1.5f);
}

void UiEditorApp::drawCanvas() {
    const UiRect pane = UiRect::fromXYWH(m_canvasX, m_canvasY, m_canvasW, m_canvasH);
    m_uiDrawList.pushClip(pane);
    m_uiDrawList.addRectFilled(pane, kCanvasBg);

    // The design surface itself, so the document bounds are visible.
    const UiRect surface = docRectToScreen(UiRect::fromXYWH(0, 0, m_docW, m_docH));
    m_uiDrawList.addRectFilled(surface, kSurfaceBg);

    const float step = std::max(2.0f, m_snapSettings.gridSize);
    for (float gx = 0; gx <= m_docW; gx += step) {
        const bool major = std::fmod(gx, step * 10.0f) < 0.5f;
        const float sx = surface.minX + gx;
        m_uiDrawList.addRectFilled({sx, surface.minY, sx + 0.5f, surface.maxY},
                                   major ? kGridMajor : kGridLine);
    }
    for (float gy = 0; gy <= m_docH; gy += step) {
        const bool major = std::fmod(gy, step * 10.0f) < 0.5f;
        const float sy = surface.minY + gy;
        m_uiDrawList.addRectFilled({surface.minX, sy, surface.maxX, sy + 0.5f},
                                   major ? kGridMajor : kGridLine);
    }

    // The root node's own children; the root itself is the surface.
    const int count = m_doc.childCount({});
    for (int i = 0; i < count; ++i) drawNode(NodePath{i});

    for (const SnapLine& line : m_snap.lines()) {
        if (line.horizontal) {
            const float sy = docToScreen(0.0f, line.pos).y;
            m_uiDrawList.addRectFilled({pane.minX, sy - 0.75f, pane.maxX, sy + 0.75f}, kSnapLine);
        } else {
            const float sx = docToScreen(line.pos, 0.0f).x;
            m_uiDrawList.addRectFilled({sx - 0.75f, pane.minY, sx + 0.75f, pane.maxY}, kSnapLine);
        }
    }

    drawSelectionOverlay();
    drawPlacementGhost();
    m_uiDrawList.popClip();
}

int UiEditorApp::hitTestHandle(float docX, float docY) const {
    const NodePath* primary = m_selection.primary();
    if (!primary || m_selection.size() != 1 || !m_doc.isValid(*primary)) return -1;
    const UiRect r = m_doc.absoluteRect(*primary);
    const float mx = (r.minX + r.maxX) * 0.5f;
    const float my = (r.minY + r.maxY) * 0.5f;
    const UiVec2 handles[8] = {{r.minX, r.minY}, {mx, r.minY},     {r.maxX, r.minY},
                               {r.maxX, my},     {r.maxX, r.maxY}, {mx, r.maxY},
                               {r.minX, r.maxY}, {r.minX, my}};
    constexpr float kR = 7.0f;
    for (int i = 0; i < 8; ++i) {
        const float dx = docX - handles[i].x;
        const float dy = docY - handles[i].y;
        if (dx * dx + dy * dy <= kR * kR) return i;
    }
    return -1;
}

// ─── Outliner ────────────────────────────────────────────────────────────────

void UiEditorApp::drawOutlinerRow(const NodePath& path, int depth, float& y) {
    const float paneBottom = static_cast<float>(m_lastFbH) - kStatusH;
    if (y > paneBottom - kOutlinerRowH) return;

    const bool selected = m_selection.contains(path);
    const UiRect row = UiRect::fromXYWH(0.0f, y, kPaletteW, kOutlinerRowH);
    if (selected) m_uiDrawList.addRectFilled(row, {0.14f, 0.22f, 0.36f, 0.90f});

    const std::string type = m_doc.typeOf(path);
    const TypeDesc* desc = findWidgetType(type);
    const std::string id = m_doc.idOf(path);
    std::string label = type;
    if (!id.empty()) label += "  " + id;

    const float indent = 6.0f + static_cast<float>(depth) * 10.0f;
    m_uiDrawList.addRectFilled(UiRect::fromXYWH(indent - 4.0f, y + 7.0f, 3.0f, 3.0f),
                               desc ? desc->accent : kTextDim);
    m_uiDrawList.addText(m_uiFont, label, {indent + 4.0f, y + 2.0f}, selected ? kBlue : kText);

    EditorHit hit;
    hit.rect = row;
    hit.action = EditorHit::Action::SelectNode;
    hit.path = path;
    pushHit(hit);

    y += kOutlinerRowH;
    const int count = m_doc.childCount(path);
    for (int i = 0; i < count; ++i) {
        NodePath child = path;
        child.push_back(i);
        drawOutlinerRow(child, depth + 1, y);
    }
}

void UiEditorApp::drawOutliner() {
    const float top = paletteBottom();
    const float bottom = static_cast<float>(m_lastFbH) - kStatusH;
    if (bottom - top < kOutlinerRowH) return;

    m_uiDrawList.addRectFilled(UiRect::fromXYWH(0, top, kPaletteW, 22.0f),
                               {0.09f, 0.10f, 0.14f, 1.0f});
    m_uiDrawList.addText(m_uiFont, "DOCUMENT", {8.0f, top + 3.0f}, {0.50f, 0.55f, 0.72f, 1.0f});

    m_uiDrawList.pushClip(UiRect::fromXYWH(0, top + 22.0f, kPaletteW, bottom - top - 22.0f));
    float y = top + 24.0f;
    drawOutlinerRow(NodePath{}, 0, y);
    m_uiDrawList.popClip();
}

// ─── Properties panel ────────────────────────────────────────────────────────

void UiEditorApp::drawSectionLabel(const char* label, float x, float width, float& y) {
    m_uiDrawList.addRectFilled(UiRect::fromXYWH(x + 6, y, width - 12, 1), {0.2f, 0.22f, 0.30f, 0.7f});
    y += 5.0f;
    m_uiDrawList.addText(m_uiFont, label, {x + 10, y}, kTextDim);
    y += m_uiFont.lineHeightPx() + 4.0f;
}

void UiEditorApp::drawStepRow(const char* label, float value, EditorHit::Action action,
                              const std::string& key, const float (&deltas)[4], float x,
                              float width, float& y) {
    (void)width;
    m_uiDrawList.addText(m_uiFont, label, {x + 10, y + 4}, kText);

    float bx = x + 78.0f;
    constexpr float bw = 30.0f, bh = 22.0f, gap = 2.0f;
    const char* labels[4] = {"<<", "<", ">", ">>"};
    char buf[32];
    for (int i = 0; i < 4; ++i) {
        const UiRect r = UiRect::fromXYWH(bx, y, bw, bh);
        m_uiDrawList.addRoundRectFilled(r, {0.10f, 0.12f, 0.17f, 0.90f}, 2.0f);
        m_uiDrawList.addRoundRect(r, {0.25f, 0.28f, 0.38f, 0.80f}, 2.0f, 1.0f);
        const float lw = m_uiFont.measureText(labels[i]);
        m_uiDrawList.addText(m_uiFont, labels[i],
                             {r.minX + (bw - lw) * 0.5f,
                              r.minY + (bh - m_uiFont.lineHeightPx()) * 0.5f},
                             kText);
        EditorHit hit;
        hit.rect = r;
        hit.action = action;
        hit.key = key;
        hit.delta = deltas[i];
        pushHit(hit);
        bx += bw + gap;

        if (i == 1) {
            const UiRect vr = UiRect::fromXYWH(bx, y, 52.0f, bh);
            if (std::fabs(value) < 100.0f && std::fabs(value - std::round(value)) > 0.005f) {
                std::snprintf(buf, sizeof(buf), "%.2f", static_cast<double>(value));
            } else {
                std::snprintf(buf, sizeof(buf), "%.0f", static_cast<double>(value));
            }
            m_uiDrawList.addRoundRectFilled(vr, {0.12f, 0.13f, 0.16f, 0.9f}, 2.0f);
            const float vw = m_uiFont.measureText(buf);
            m_uiDrawList.addText(m_uiFont, buf,
                                 {vr.minX + (52.0f - vw) * 0.5f,
                                  vr.minY + (bh - m_uiFont.lineHeightPx()) * 0.5f},
                                 kGold);
            bx += 52.0f + gap;
        }
    }
    y += bh + 5.0f;
}

void UiEditorApp::drawColorRow(const char* label, const std::string& key, const UiColor& current,
                               bool present, float x, float width, float& y) {
    const bool open = (m_colorKey == key);

    // Header: label, the live swatch (click to open the picker), and a clear.
    m_uiDrawList.addText(m_uiFont, label, {x + 10, y + 4}, kText);

    const UiRect swatch = UiRect::fromXYWH(x + 96.0f, y, 60.0f, 22.0f);
    // Checker behind the swatch so a translucent color reads as translucent.
    m_uiDrawList.addRectFilled(UiRect::fromXYWH(swatch.minX, swatch.minY, 30.0f, 22.0f),
                               {0.30f, 0.30f, 0.34f, 1.0f});
    m_uiDrawList.addRectFilled(UiRect::fromXYWH(swatch.minX + 30.0f, swatch.minY, 30.0f, 22.0f),
                               {0.18f, 0.18f, 0.21f, 1.0f});
    if (present) m_uiDrawList.addRoundRectFilled(swatch, current, 3.0f);
    m_uiDrawList.addRoundRect(swatch, open ? kBlue : UiColor{0.32f, 0.34f, 0.42f, 0.85f}, 3.0f,
                              open ? 2.0f : 1.0f);
    if (!present) {
        m_uiDrawList.addText(m_uiFont, "unset", {swatch.minX + 12.0f, swatch.minY + 3.0f}, kTextDim);
    }
    EditorHit openHit;
    openHit.rect = swatch;
    openHit.action = EditorHit::Action::TogglePicker;
    openHit.key = key;
    pushHit(openHit);

    // "--" clears the property so the widget falls back to its own default.
    const UiRect clear = UiRect::fromXYWH(x + width - 40.0f, y, 30.0f, 22.0f);
    m_uiDrawList.addRoundRectFilled(clear, {0.10f, 0.12f, 0.17f, 0.90f}, 3.0f);
    m_uiDrawList.addRoundRect(clear, present ? UiColor{0.3f, 0.3f, 0.4f, 0.6f} : kBlue, 3.0f, 1.0f);
    m_uiDrawList.addText(m_uiFont, "--", {clear.minX + 9.0f, clear.minY + 3.0f}, kTextDim);
    EditorHit clearHit;
    clearHit.rect = clear;
    clearHit.action = EditorHit::Action::ClearProp;
    clearHit.key = key;
    pushHit(clearHit);
    y += 26.0f;

    // Quick presets, always available so a one-click choice stays one click.
    float cx = x + 10.0f;
    constexpr float sz = 18.0f;
    for (int i = 0; i < kNumColorPresets; ++i) {
        const UiRect r = UiRect::fromXYWH(cx, y, sz, sz);
        m_uiDrawList.addRoundRectFilled(r, kColorPresets[i], 3.0f);
        const bool active = present && std::fabs(current.r - kColorPresets[i].r) < 0.01f &&
                            std::fabs(current.g - kColorPresets[i].g) < 0.01f &&
                            std::fabs(current.b - kColorPresets[i].b) < 0.01f;
        m_uiDrawList.addRoundRect(r, active ? kBlue : UiColor{0.3f, 0.3f, 0.4f, 0.5f}, 3.0f,
                                  active ? 2.0f : 1.0f);
        EditorHit hit;
        hit.rect = r;
        hit.action = EditorHit::Action::SetColorProp;
        hit.key = key;
        hit.colorValue = kColorPresets[i];
        pushHit(hit);
        cx += sz + 3.0f;
    }
    y += sz + 6.0f;

    if (open) drawColorPicker(key, current, x, width, y);
}

// ─── HSV picker + harmony ────────────────────────────────────────────────────

void UiEditorApp::drawColorPicker(const std::string& key, const UiColor& current, float x,
                                  float width, float& y) {
    // Keep the sticky HSV in step with the stored value. Dragging value to 0
    // collapses RGB to black and would otherwise throw the hue away, so the
    // picker only re-seeds when the property no longer matches what its own
    // HSV would produce (an undo, a preset click, a harmony pick).
    const UiColor fromPicker = hsvToRgb(m_pickerHsv, current.a);
    if (std::fabs(fromPicker.r - current.r) > 0.004f ||
        std::fabs(fromPicker.g - current.g) > 0.004f ||
        std::fabs(fromPicker.b - current.b) > 0.004f) {
        m_pickerHsv = rgbToHsv(current);
    }
    const Hsv hsv = m_pickerHsv;

    constexpr float kSquareH = 104.0f;
    constexpr float kBarW = 15.0f;
    const float squareW = width - 20.0f - (kBarW + 6.0f) * 2.0f;

    // ── Saturation/value square ──────────────────────────────────────────────
    // Value scales RGB linearly, so each saturation column is exactly a vertical
    // gradient from full value down to black: 32 strips, no per-pixel work.
    m_svRect = UiRect::fromXYWH(x + 10.0f, y, squareW, kSquareH);
    constexpr int kStrips = 32;
    for (int i = 0; i < kStrips; ++i) {
        const float s0 = static_cast<float>(i) / static_cast<float>(kStrips);
        const float stripW = squareW / static_cast<float>(kStrips);
        const UiColor top = hsvToRgb(Hsv{hsv.h, s0, 1.0f});
        m_uiDrawList.addRectFilledVGradient(
            UiRect::fromXYWH(m_svRect.minX + s0 * squareW, m_svRect.minY, stripW + 0.5f, kSquareH),
            top, UiColor{0, 0, 0, 1});
    }
    m_uiDrawList.addRect(m_svRect, {0.32f, 0.34f, 0.42f, 0.85f}, 1.0f);
    {
        const float cx = m_svRect.minX + hsv.s * squareW;
        const float cy = m_svRect.minY + (1.0f - hsv.v) * kSquareH;
        m_uiDrawList.addCircle({cx, cy}, 5.0f, {0, 0, 0, 0.85f}, 3.0f);
        m_uiDrawList.addCircle({cx, cy}, 5.0f, {1, 1, 1, 0.95f}, 1.5f);
    }

    // ── Hue bar ──────────────────────────────────────────────────────────────
    // Six exact gradient segments: between adjacent primaries, HSV hue at s=v=1
    // is linear in RGB.
    m_hueRect = UiRect::fromXYWH(m_svRect.maxX + 6.0f, y, kBarW, kSquareH);
    for (int i = 0; i < 6; ++i) {
        const float segH = kSquareH / 6.0f;
        m_uiDrawList.addRectFilledVGradient(
            UiRect::fromXYWH(m_hueRect.minX, m_hueRect.minY + static_cast<float>(i) * segH, kBarW,
                             segH + 0.5f),
            hsvToRgb(Hsv{static_cast<float>(i) * 60.0f, 1.0f, 1.0f}),
            hsvToRgb(Hsv{static_cast<float>(i + 1) * 60.0f, 1.0f, 1.0f}));
    }
    m_uiDrawList.addRect(m_hueRect, {0.32f, 0.34f, 0.42f, 0.85f}, 1.0f);
    {
        const float hy = m_hueRect.minY + (hsv.h / 360.0f) * kSquareH;
        m_uiDrawList.addRectFilled({m_hueRect.minX - 2.0f, hy - 1.5f, m_hueRect.maxX + 2.0f, hy + 1.5f},
                                   {1, 1, 1, 0.95f});
    }

    // ── Alpha bar ────────────────────────────────────────────────────────────
    m_alphaRect = UiRect::fromXYWH(m_hueRect.maxX + 6.0f, y, kBarW, kSquareH);
    // Checker so full transparency is distinguishable from a dark color.
    for (int i = 0; i < 8; ++i) {
        m_uiDrawList.addRectFilled(
            UiRect::fromXYWH(m_alphaRect.minX, m_alphaRect.minY + static_cast<float>(i) * (kSquareH / 8.0f),
                             kBarW, kSquareH / 8.0f),
            (i % 2 == 0) ? UiColor{0.30f, 0.30f, 0.34f, 1.0f} : UiColor{0.18f, 0.18f, 0.21f, 1.0f});
    }
    const UiColor opaque = hsvToRgb(hsv, 1.0f);
    m_uiDrawList.addRectFilledVGradient(m_alphaRect, opaque, withAlpha(opaque, 0.0f));
    m_uiDrawList.addRect(m_alphaRect, {0.32f, 0.34f, 0.42f, 0.85f}, 1.0f);
    {
        const float ay = m_alphaRect.minY + (1.0f - current.a) * kSquareH;
        m_uiDrawList.addRectFilled(
            {m_alphaRect.minX - 2.0f, ay - 1.5f, m_alphaRect.maxX + 2.0f, ay + 1.5f}, {1, 1, 1, 0.95f});
    }
    y += kSquareH + 6.0f;

    // ── Numeric readout ──────────────────────────────────────────────────────
    char buf[128];
    const std::string hex = colorToHex(current);
    std::snprintf(buf, sizeof(buf), "%s   H%.0f S%.0f V%.0f", hex.c_str(),
                  static_cast<double>(hsv.h), static_cast<double>(hsv.s * 100.0f),
                  static_cast<double>(hsv.v * 100.0f));
    m_uiDrawList.addText(m_uiFont, buf, {x + 10.0f, y}, kGold);
    y += m_uiFont.lineHeightPx() + 3.0f;

    // ── WCAG contrast against what this color actually sits on ───────────────
    const NodePath* primary = m_selection.primary();
    if (primary && m_doc.isValid(*primary)) {
        // "background" is read against its ancestors; every other color role is
        // a foreground, read against this node's own background as well.
        const bool isBackground = (key == "background");
        const UiColor behind = backdropFor(*primary, !isBackground);
        const UiColor effective = compositeOver(current, behind);
        const float ratio = contrastRatio(effective, behind);
        const std::string_view rating = contrastRating(ratio);
        std::snprintf(buf, sizeof(buf), "contrast %.1f:1  %.*s", static_cast<double>(ratio),
                      static_cast<int>(rating.size()), rating.data());
        const UiColor ratingColor = ratio >= 4.5f  ? UiColor{0.45f, 0.85f, 0.50f, 1.0f}
                                    : ratio >= 3.0f ? kGold
                                                    : UiColor{0.90f, 0.45f, 0.42f, 1.0f};
        m_uiDrawList.addText(m_uiFont, buf, {x + 10.0f, y}, ratingColor);

        // Preview chip: this color as it will actually appear on that backdrop.
        const UiRect chip = UiRect::fromXYWH(x + width - 52.0f, y, 42.0f, 16.0f);
        m_uiDrawList.addRectFilled(chip, behind);
        m_uiDrawList.addRectFilled(UiRect::fromXYWH(chip.minX + 3, chip.minY + 3, 36.0f, 10.0f),
                                   effective);
        m_uiDrawList.addRect(chip, {0.32f, 0.34f, 0.42f, 0.85f}, 1.0f);
    }
    y += m_uiFont.lineHeightPx() + 6.0f;

    // ── Harmony scheme selector ──────────────────────────────────────────────
    const std::vector<Harmony>& schemes = allHarmonies();
    constexpr float kSchemeW = 64.0f, kSchemeH = 19.0f;
    for (std::size_t i = 0; i < schemes.size(); ++i) {
        const float col = static_cast<float>(i % 4);
        const float row = static_cast<float>(i / 4);
        const UiRect r = UiRect::fromXYWH(x + 10.0f + col * (kSchemeW + 3.0f),
                                          y + row * (kSchemeH + 3.0f), kSchemeW, kSchemeH);
        const bool active = (schemes[i] == m_harmony);
        m_uiDrawList.addRoundRectFilled(
            r, active ? UiColor{0.15f, 0.25f, 0.45f, 0.95f} : UiColor{0.10f, 0.12f, 0.17f, 0.90f},
            3.0f);
        m_uiDrawList.addRoundRect(r, active ? kBlue : UiColor{0.25f, 0.28f, 0.38f, 0.80f}, 3.0f,
                                  active ? 1.5f : 1.0f);
        const std::string name(harmonyName(schemes[i]));
        const float lw = m_uiFont.measureText(name);
        m_uiDrawList.addText(m_uiFont, name,
                             {r.minX + (kSchemeW - lw) * 0.5f,
                              r.minY + (kSchemeH - m_uiFont.lineHeightPx()) * 0.5f},
                             active ? kBlue : kText);
        EditorHit hit;
        hit.rect = r;
        hit.action = EditorHit::Action::SetHarmony;
        hit.key = key;
        hit.delta = static_cast<float>(i);
        pushHit(hit);
    }
    y += (kSchemeH + 3.0f) * 2.0f + 4.0f;

    // ── The scheme's colors, click to apply ──────────────────────────────────
    const std::vector<UiColor> partners = harmonyColors(current, m_harmony);
    const float swatchW = std::min(34.0f, (width - 20.0f) / static_cast<float>(partners.size()) - 3.0f);
    float sx = x + 10.0f;
    for (std::size_t i = 0; i < partners.size(); ++i) {
        const UiRect r = UiRect::fromXYWH(sx, y, swatchW, 24.0f);
        m_uiDrawList.addRoundRectFilled(r, partners[i], 3.0f);
        // Index 0 is the base itself — mark it so the row reads as "you are here
        // plus what goes with it".
        m_uiDrawList.addRoundRect(r, i == 0 ? kBlue : UiColor{0.30f, 0.32f, 0.40f, 0.70f}, 3.0f,
                                  i == 0 ? 2.0f : 1.0f);
        EditorHit hit;
        hit.rect = r;
        hit.action = EditorHit::Action::SetColorProp;
        hit.key = key;
        hit.colorValue = partners[i];
        pushHit(hit);
        sx += swatchW + 3.0f;
    }
    y += 24.0f + 8.0f;
}

void UiEditorApp::drawBoolRow(const char* label, const std::string& key, bool value, float x,
                              float width, float& y) {
    (void)width;
    m_uiDrawList.addText(m_uiFont, label, {x + 10, y + 5}, kText);
    const char* labels[2] = {"Off", "On"};
    for (int i = 0; i < 2; ++i) {
        const bool active = (value == (i == 1));
        const UiRect r = UiRect::fromXYWH(x + 100.0f + static_cast<float>(i) * 62.0f, y, 58.0f, 24.0f);
        m_uiDrawList.addRoundRectFilled(
            r, active ? UiColor{0.15f, 0.25f, 0.45f, 0.95f} : UiColor{0.10f, 0.12f, 0.17f, 0.90f},
            4.0f);
        m_uiDrawList.addRoundRect(r, active ? kBlue : UiColor{0.25f, 0.28f, 0.38f, 0.80f}, 4.0f,
                                  active ? 1.5f : 1.0f);
        const float lw = m_uiFont.measureText(labels[i]);
        m_uiDrawList.addText(m_uiFont, labels[i],
                             {r.minX + (58.0f - lw) * 0.5f,
                              r.minY + (24.0f - m_uiFont.lineHeightPx()) * 0.5f},
                             active ? kBlue : kText);
        EditorHit hit;
        hit.rect = r;
        hit.action = EditorHit::Action::SetBoolProp;
        hit.key = key;
        hit.delta = (i == 1) ? 1.0f : 0.0f;
        pushHit(hit);
    }
    y += 24.0f + 6.0f;
}

void UiEditorApp::drawTextRow(const char* label, const std::string& key, const std::string& value,
                              float x, float width, float& y) {
    m_uiDrawList.addText(m_uiFont, label, {x + 10, y + 4}, kText);
    const UiRect field = UiRect::fromXYWH(x + 78.0f, y, width - 88.0f, 22.0f);
    const bool editing = m_editing && m_editingKey == key;

    m_uiDrawList.addRoundRectFilled(field, {0.12f, 0.13f, 0.16f, 0.95f}, 2.0f);
    m_uiDrawList.addRoundRect(field, editing ? kBlue : UiColor{0.25f, 0.28f, 0.38f, 0.70f}, 2.0f,
                              editing ? 1.5f : 1.0f);

    std::string shown = editing ? m_editingBuffer : value;
    // Trim from the left so the caret end of a long string stays visible.
    while (!shown.empty() && m_uiFont.measureText(shown) > field.width() - 12.0f) {
        shown.erase(shown.begin());
    }
    if (editing) shown += "_";
    m_uiDrawList.pushClip(field);
    m_uiDrawList.addText(m_uiFont, shown, {field.minX + 5.0f, field.minY + 3.0f},
                         editing ? kGold : (value.empty() ? kTextDim : kText));
    m_uiDrawList.popClip();

    EditorHit hit;
    hit.rect = field;
    hit.action = EditorHit::Action::BeginTextEdit;
    hit.key = key;
    pushHit(hit);

    y += 22.0f + 5.0f;
}

void UiEditorApp::drawProperties() {
    const float px = static_cast<float>(m_lastFbW) - kPropsW;
    const float py = kToolbarH;
    const float ph = static_cast<float>(m_lastFbH) - kToolbarH - kStatusH;

    m_uiDrawList.addRectFilled(UiRect::fromXYWH(px, py, kPropsW, ph), kPanelBg);
    m_uiDrawList.addRectFilled(UiRect::fromXYWH(px, py, kPropsW, 26.0f), {0.09f, 0.10f, 0.14f, 1.0f});
    m_uiDrawList.addText(m_uiFontBold, "PROPERTIES", {px + 10, py + 5}, kGold);

    float y = py + 34.0f;
    const NodePath* primary = m_selection.primary();

    // The open picker belongs to one node; selecting another closes it.
    if (!primary || m_colorPath != *primary) m_colorKey.clear();

    if (!primary || !m_doc.isValid(*primary)) {
        m_uiDrawList.addText(m_uiFont, "No node selected", {px + 10, y}, kTextDim);
        y += m_uiFont.lineHeightPx() + 14.0f;
        m_uiDrawList.addText(m_uiFont, "Click a palette type, then", {px + 10, y}, kTextDim);
        y += m_uiFont.lineHeightPx() + 2.0f;
        m_uiDrawList.addText(m_uiFont, "click the canvas to place it.", {px + 10, y}, kTextDim);
        return;
    }

    const NodePath path = *primary;
    const json* node = m_doc.nodeAt(path);
    const std::string type = m_doc.typeOf(path);
    const TypeDesc* desc = findWidgetType(type);

    char buf[160];
    std::snprintf(buf, sizeof(buf), "%s%s", type.c_str(), desc ? "" : "  (unknown type)");
    m_uiDrawList.addText(m_uiFontBold, buf, {px + 10, y}, kText);
    y += m_uiFontBold.lineHeightPx() + 6.0f;

    if (m_selection.size() > 1) {
        std::snprintf(buf, sizeof(buf), "%d nodes selected", static_cast<int>(m_selection.size()));
        m_uiDrawList.addText(m_uiFont, buf, {px + 10, y}, kBlue);
        y += m_uiFont.lineHeightPx() + 6.0f;
    }
    if (m_doc.parentControlsLayout(path)) {
        m_uiDrawList.addText(m_uiFont, "position set by parent at runtime", {px + 10, y},
                             kManagedTag);
        y += m_uiFont.lineHeightPx() + 6.0f;
    }

    // Everything from here down scrolls under the pinned reorder/delete buttons
    // when the pane runs out of room, so it is clipped — and pushHit() drops any
    // hit region that ends up outside the clip, so nothing stays clickable where
    // it isn't drawn.
    m_hitClip = UiRect{px, y - 4.0f, px + kPropsW,
                       static_cast<float>(m_lastFbH) - kStatusH - 78.0f};
    m_hitClipActive = true;
    m_uiDrawList.pushClip(m_hitClip);

    drawTextRow("id", "id", m_doc.idOf(path), px, kPropsW, y);

    const UiRect rect = m_doc.absoluteRect(path);
    static constexpr float kPixelSteps[4] = {-10.0f, -1.0f, 1.0f, 10.0f};
    drawSectionLabel("Position", px, kPropsW, y);
    drawStepRow("X", rect.minX, EditorHit::Action::StepRectX, "x", kPixelSteps, px, kPropsW, y);
    drawStepRow("Y", rect.minY, EditorHit::Action::StepRectY, "y", kPixelSteps, px, kPropsW, y);
    drawSectionLabel("Size", px, kPropsW, y);
    drawStepRow("W", rect.width(), EditorHit::Action::StepRectW, "width", kPixelSteps, px, kPropsW,
                y);
    drawStepRow("H", rect.height(), EditorHit::Action::StepRectH, "height", kPixelSteps, px,
                kPropsW, y);

    static constexpr float kFineSteps[4] = {-0.25f, -0.05f, 0.05f, 0.25f};
    static const std::vector<PropDesc> kNoProps;
    const std::vector<PropDesc>& props = desc ? desc->props : kNoProps;
    if (!props.empty()) drawSectionLabel(type.c_str(), px, kPropsW, y);
    for (const PropDesc& prop : props) {
        const std::string key(prop.key);
        const std::string label(prop.label);
        switch (prop.kind) {
            case PropKind::Float: {
                // A 0..1-style property needs finer steps than a pixel one.
                const bool fine = (prop.max > prop.min) && (prop.max - prop.min) <= 2.0f;
                drawStepRow(label.c_str(), nodeFloat(*node, key.c_str(), 0.0f),
                            EditorHit::Action::StepFloatProp, key, fine ? kFineSteps : kPixelSteps,
                            px, kPropsW, y);
                break;
            }
            case PropKind::Bool:
                drawBoolRow(label.c_str(), key, node->value(key, false), px, kPropsW, y);
                break;
            case PropKind::Color:
                drawColorRow(label.c_str(), key,
                             nodeColor(*node, key.c_str(), UiColor{0, 0, 0, 0}),
                             node->contains(key), px, kPropsW, y);
                break;
            case PropKind::String:
            case PropKind::Text:
            case PropKind::Frame:
            case PropKind::Slot:
                drawTextRow(label.c_str(), key, nodeString(*node, key.c_str()), px, kPropsW, y);
                break;
        }
    }

    m_uiDrawList.popClip();
    m_hitClipActive = false;

    // Delete, pinned to the bottom of the pane.
    const float dy = static_cast<float>(m_lastFbH) - kStatusH - 44.0f;
    const UiRect del = UiRect::fromXYWH(px + 10, dy, kPropsW - 20.0f, 30.0f);
    m_uiDrawList.addRoundRectFilled(del, {0.25f, 0.06f, 0.06f, 0.90f}, 5.0f);
    m_uiDrawList.addRoundRect(del, kDeleteRed, 5.0f, 1.5f);
    const char* delLabel = "Delete Selection";
    const float dlw = m_uiFontBold.measureText(delLabel);
    m_uiDrawList.addText(m_uiFontBold, delLabel,
                         {del.minX + (del.width() - dlw) * 0.5f,
                          del.minY + (30.0f - m_uiFontBold.lineHeightPx()) * 0.5f},
                         {1.0f, 0.55f, 0.50f, 1.0f});
    EditorHit delHit;
    delHit.rect = del;
    delHit.action = EditorHit::Action::DeleteNode;
    pushHit(delHit);

    // Sibling reorder, just above Delete.
    const float ry = dy - 30.0f;
    const char* orderLabels[2] = {"Send back", "Bring front"};
    for (int i = 0; i < 2; ++i) {
        const UiRect r =
            UiRect::fromXYWH(px + 10.0f + static_cast<float>(i) * ((kPropsW - 20.0f) * 0.5f + 2.0f),
                             ry, (kPropsW - 24.0f) * 0.5f, 24.0f);
        m_uiDrawList.addRoundRectFilled(r, {0.10f, 0.12f, 0.17f, 0.90f}, 3.0f);
        m_uiDrawList.addRoundRect(r, {0.25f, 0.28f, 0.38f, 0.80f}, 3.0f, 1.0f);
        const float lw = m_uiFont.measureText(orderLabels[i]);
        m_uiDrawList.addText(m_uiFont, orderLabels[i],
                             {r.minX + (r.width() - lw) * 0.5f,
                              r.minY + (24.0f - m_uiFont.lineHeightPx()) * 0.5f},
                             kText);
        EditorHit hit;
        hit.rect = r;
        hit.action = EditorHit::Action::ReorderNode;
        hit.path = path;
        hit.delta = (i == 0) ? -1.0f : 1.0f;
        pushHit(hit);
    }
}

// ─── Toolbar + palette (real widget tree, for hover states) ──────────────────

void UiEditorApp::buildWidgetTree(int fbW, int fbH) {
    auto root = std::make_unique<Widget>();
    root->mousePassthrough = true;
    root->setRect(UiRect::fromXYWH(0, 0, static_cast<float>(fbW), static_cast<float>(fbH)));

    auto toolbar = std::make_unique<Panel>();
    toolbar->background = {0.068f, 0.072f, 0.088f, 0.98f};
    toolbar->borderColor = {0.13f, 0.14f, 0.18f, 1.0f};
    toolbar->borderThicknessPx = 1.0f;
    toolbar->setRect(UiRect::fromXYWH(0, 0, static_cast<float>(fbW), kToolbarH));

    auto flatButton = [&](Panel& parent, float bx, float by, float bw, float bh,
                          const std::string& label, bool active, const UiColor& accent,
                          std::function<void()> fn) {
        auto btn = std::make_unique<Button>(&m_uiFont, label, std::move(fn));
        btn->colorNormal = active ? UiColor{0.10f, 0.18f, 0.30f, 0.90f}
                                  : UiColor{0.08f, 0.09f, 0.12f, 0.90f};
        btn->colorHover = active ? UiColor{0.14f, 0.22f, 0.38f, 0.95f}
                                 : UiColor{0.11f, 0.13f, 0.18f, 0.95f};
        btn->colorPressed = {0.06f, 0.08f, 0.12f, 0.95f};
        btn->borderColor = withAlpha(accent, active ? 0.85f : 0.40f);
        btn->labelColor = active ? accent : kText;
        btn->glowColor = accent;
        btn->glowSizePx = active ? 6.0f : 0.0f;
        btn->cornerRadiusPx = 3.0f;
        btn->setRect(UiRect::fromXYWH(bx, by, bw, bh));
        parent.addChild(std::move(btn));
    };

    // Title.
    {
        auto title = std::make_unique<Button>(&m_uiFontBold, "odai  UI Editor", []() {});
        title->colorNormal = {0, 0, 0, 0};
        title->colorHover = {0, 0, 0, 0};
        title->colorPressed = {0, 0, 0, 0};
        title->borderColor = {0, 0, 0, 0};
        title->glowSizePx = 0.0f;
        title->labelColor = kGold;
        title->setRect(UiRect::fromXYWH(10, 6, 150, 34));
        toolbar->addChild(std::move(title));
    }

    float bx = 168.0f;
    flatButton(*toolbar, bx, 7, 40, 32, "Undo", m_history.canUndo(), kBlue,
               [this]() { doUndo(); });
    bx += 44.0f;
    flatButton(*toolbar, bx, 7, 40, 32, "Redo", m_history.canRedo(), kBlue,
               [this]() { doRedo(); });
    bx += 52.0f;

    // Align / distribute act on the whole selection.
    struct AlignBtn {
        const char* label;
        AlignEdge edge;
    };
    static const AlignBtn kAlignButtons[6] = {
        {"L", AlignEdge::Left},  {"C", AlignEdge::HCenter}, {"R", AlignEdge::Right},
        {"T", AlignEdge::Top},   {"M", AlignEdge::VCenter}, {"B", AlignEdge::Bottom},
    };
    for (const AlignBtn& ab : kAlignButtons) {
        const AlignEdge edge = ab.edge;
        flatButton(*toolbar, bx, 7, 26, 32, ab.label, false, kGold, [this, edge]() {
            if (alignSelection(m_doc, m_selection, edge)) {
                m_history.commit(m_doc, "align");
                m_statusMessage = "Aligned selection";
            }
        });
        bx += 28.0f;
    }
    bx += 8.0f;
    flatButton(*toolbar, bx, 7, 34, 32, "H|||", false, kGold, [this]() {
        if (distributeSelection(m_doc, m_selection, true)) {
            m_history.commit(m_doc, "distribute");
            m_statusMessage = "Distributed horizontally";
        }
    });
    bx += 38.0f;
    flatButton(*toolbar, bx, 7, 34, 32, "V===", false, kGold, [this]() {
        if (distributeSelection(m_doc, m_selection, false)) {
            m_history.commit(m_doc, "distribute");
            m_statusMessage = "Distributed vertically";
        }
    });
    bx += 46.0f;

    flatButton(*toolbar, bx, 7, 76, 32, m_snapSettings.grid ? "Grid: ON" : "Grid: off",
               m_snapSettings.grid, kBlue, [this]() {
                   m_snapSettings.grid = !m_snapSettings.grid;
                   m_uiDirty = true;
               });
    bx += 80.0f;
    flatButton(*toolbar, bx, 7, 82, 32, m_snapSettings.edges ? "Edges: ON" : "Edges: off",
               m_snapSettings.edges, kBlue, [this]() {
                   m_snapSettings.edges = !m_snapSettings.edges;
                   m_uiDirty = true;
               });

    // Save/Load pinned to the right of the toolbar.
    const float ioX = static_cast<float>(fbW) - 176.0f;
    flatButton(*toolbar, ioX, 7, 80, 32, "Save", false, {0.50f, 0.75f, 1.00f, 1.0f},
               [this]() { doSave(); });
    flatButton(*toolbar, ioX + 84.0f, 7, 80, 32, "Load", false, {0.50f, 0.75f, 1.00f, 1.0f},
               [this]() { doLoad(); });

    root->addChild(std::move(toolbar));

    // ── Palette ──
    auto palette = std::make_unique<Panel>();
    palette->background = kPanelBg;
    palette->borderColor = kPanelBord;
    palette->borderThicknessPx = 1.0f;
    palette->setRect(UiRect::fromXYWH(0, kToolbarH, kPaletteW, paletteBottom() - kToolbarH));

    {
        auto header = std::make_unique<Button>(&m_uiFont, "WIDGET PALETTE", []() {});
        header->colorNormal = {0.09f, 0.10f, 0.14f, 1.0f};
        header->colorHover = {0.09f, 0.10f, 0.14f, 1.0f};
        header->colorPressed = {0.09f, 0.10f, 0.14f, 1.0f};
        header->borderColor = {0, 0, 0, 0};
        header->glowSizePx = 0.0f;
        header->labelColor = {0.50f, 0.55f, 0.72f, 1.0f};
        header->setRect(UiRect::fromXYWH(0, kToolbarH, kPaletteW, 22.0f));
        palette->addChild(std::move(header));
    }

    float py = kToolbarH + 26.0f;
    for (const TypeDesc& type : widgetTypes()) {
        const std::string name(type.name);
        const bool active = m_placing && m_placeType == name;
        flatButton(*palette, 5, py, kPaletteW - 10.0f, kPaletteRowH - 4.0f, "+ " + name, active,
                   type.accent, [this, name]() {
                       m_placing = true;
                       m_placeType = name;
                       m_statusMessage = "Click the canvas to place a " + name;
                       m_uiDirty = true;
                   });
        py += kPaletteRowH;
    }

    root->addChild(std::move(palette));
    m_uiContext.setRoot(std::move(root));
}

void UiEditorApp::drawStatusBar(int fbW, int fbH) {
    const float y = static_cast<float>(fbH) - kStatusH;
    m_uiDrawList.addRectFilled(UiRect::fromXYWH(0, y, static_cast<float>(fbW), kStatusH),
                               {0.075f, 0.078f, 0.095f, 1.0f});
    m_uiDrawList.addRectFilled(UiRect::fromXYWH(0, y, static_cast<float>(fbW), 1.0f),
                               {0.16f, 0.17f, 0.22f, 1.0f});

    const std::string path = m_documentPath.empty() ? std::string("(unsaved)") : m_documentPath;
    m_uiDrawList.addText(m_uiFont, path, {8.0f, y + 3.0f}, kTextDim);

    if (!m_statusMessage.empty()) {
        const float w = m_uiFont.measureText(m_statusMessage);
        m_uiDrawList.addText(m_uiFont, m_statusMessage,
                             {(static_cast<float>(fbW) - w) * 0.5f, y + 3.0f}, kGold);
    }

    char buf[160];
    std::snprintf(buf, sizeof(buf), "%d selected  |  Ctrl+Z undo  C/V/D clipboard  arrows nudge",
                  static_cast<int>(m_selection.size()));
    const float w = m_uiFont.measureText(buf);
    m_uiDrawList.addText(m_uiFont, buf, {static_cast<float>(fbW) - w - 8.0f, y + 3.0f}, kTextDim);
}

// ─── Editing operations ──────────────────────────────────────────────────────

void UiEditorApp::placeNode(float docX, float docY) {
    const TypeDesc* desc = findWidgetType(m_placeType);
    if (!desc) {
        m_placing = false;
        return;
    }

    // Drop into whatever container is under the cursor, so nesting is authored
    // by pointing at it rather than by a separate reparent step.
    NodePath parent;
    if (!m_doc.hitTest(docX, docY, parent)) parent = NodePath{};
    while (!parent.empty()) {
        const TypeDesc* pd = findWidgetType(m_doc.typeOf(parent));
        if (!pd || pd->acceptsChildren) break;
        parent.pop_back();
    }

    const UiRect parentRect = m_doc.absoluteRect(parent);
    const float localX = docX - parentRect.minX - desc->defaultWidth * 0.5f;
    const float localY = docY - parentRect.minY - desc->defaultHeight * 0.5f;

    NodePath created;
    if (!m_doc.addNode(parent, m_placeType, localX, localY, created)) {
        m_statusMessage = m_placeType + " cannot be placed there";
        m_placing = false;
        m_uiDirty = true;
        return;
    }

    // Snap the freshly placed node the same way a drag would.
    m_snap.collectTargets(m_doc, {created});
    const UiRect r = m_doc.absoluteRect(created);
    const UiVec2 snapped = m_snap.snapPosition(r, m_snapSettings);
    m_doc.setAbsoluteRect(created,
                          UiRect::fromXYWH(snapped.x, snapped.y, r.width(), r.height()));
    m_snap.clearLines();

    m_selection.set(created);
    m_placing = false;
    m_history.commit(m_doc, "add " + m_placeType);
    m_statusMessage = "Added " + m_placeType;
    m_uiDirty = true;
}

void UiEditorApp::commitTextEdit() {
    if (!m_editing) return;
    const bool valid = m_doc.isValid(m_editingPath);
    if (valid) {
        if (m_editingBuffer.empty() && m_editingKey != "id") {
            m_doc.eraseProperty(m_editingPath, m_editingKey);
        } else if (m_editingKey == "id") {
            // An id must stay unique or the document's slot/lookup wiring breaks.
            // Take the typed name verbatim when it's free — uniqueId() re-bases
            // names (it strips trailing digits), which would silently rewrite a
            // deliberate "row2" into "row".
            const std::string typed = m_editingBuffer.empty() ? "widget" : m_editingBuffer;
            m_doc.setProperty(m_editingPath, "id", json(std::string()));
            bool taken = false;
            for (const NodePath& p : m_doc.allPaths()) {
                if (m_doc.idOf(p) == typed) {
                    taken = true;
                    break;
                }
            }
            m_doc.setProperty(m_editingPath, "id",
                              json(taken ? m_doc.uniqueId(typed) : typed));
        } else {
            m_doc.setProperty(m_editingPath, m_editingKey, json(m_editingBuffer));
        }
        m_history.commit(m_doc, "set " + m_editingKey);
    }
    m_editing = false;
    m_editingKey.clear();
    m_editingBuffer.clear();
    m_uiDirty = true;
}

void UiEditorApp::cancelTextEdit() {
    m_editing = false;
    m_editingKey.clear();
    m_editingBuffer.clear();
}

void UiEditorApp::doSave() {
    const std::string path =
        m_documentPath.empty() ? resolveAssetPath(kDefaultDocument) : m_documentPath;
    std::string error;
    if (m_doc.saveFile(path, &error)) {
        m_documentPath = path;
        m_statusMessage = "Saved " + path;
    } else {
        m_statusMessage = "Save failed: " + error;
    }
    m_uiDirty = true;
}

void UiEditorApp::doLoad() {
    const std::string path =
        m_documentPath.empty() ? resolveAssetPath(kDefaultDocument) : m_documentPath;
    std::string error;
    if (m_doc.loadFile(path, &error)) {
        m_documentPath = path;
        m_selection.clear();
        m_history.reset(m_doc);
        cancelTextEdit();
        m_statusMessage = "Loaded " + path;
        layoutPanes(m_lastFbW, m_lastFbH);
    } else {
        m_statusMessage = "Load failed: " + error;
    }
    m_uiDirty = true;
}

void UiEditorApp::doUndo() {
    if (!m_history.undo(m_doc)) return;
    m_selection.prune(m_doc);
    cancelTextEdit();
    layoutPanes(m_lastFbW, m_lastFbH);
    m_statusMessage = "Undo";
    m_uiDirty = true;
}

void UiEditorApp::doRedo() {
    if (!m_history.redo(m_doc)) return;
    m_selection.prune(m_doc);
    cancelTextEdit();
    layoutPanes(m_lastFbW, m_lastFbH);
    m_statusMessage = "Redo";
    m_uiDirty = true;
}

// ─── Hit resolution ──────────────────────────────────────────────────────────

void UiEditorApp::pushHit(const EditorHit& hit) {
    if (m_hitClipActive && !UiRect::intersect(hit.rect, m_hitClip).valid()) return;
    m_hits.push_back(hit);
}

void UiEditorApp::setColorProperty(const std::string& key, const UiColor& color,
                                   bool commitToHistory) {
    const NodePath* primary = m_selection.primary();
    if (!primary || !m_doc.isValid(*primary)) return;
    m_doc.setProperty(*primary, key, json(colorToHex(color)));
    if (commitToHistory) m_history.commit(m_doc, "set " + key);
}

UiColor UiEditorApp::backdropFor(const NodePath& path, bool includeOwnBackground) const {
    // Start from the editor's design surface and composite each ancestor's
    // background down onto it, outermost first — that stack is what a color at
    // this depth is actually seen against.
    UiColor acc = kSurfaceBg;
    const std::size_t depth = includeOwnBackground ? path.size() + 1 : path.size();
    for (std::size_t i = 0; i < depth; ++i) {
        const NodePath ancestor(path.begin(), path.begin() + static_cast<std::ptrdiff_t>(i));
        const json* node = m_doc.nodeAt(ancestor);
        if (!node || !node->contains("background")) continue;
        const json& bg = node->at("background");
        // A theme color token can't be resolved without a loaded theme; skip it
        // rather than scoring the contrast against a made-up value.
        if (!bg.is_string()) continue;
        const std::string s = bg.get<std::string>();
        if (s.empty() || s[0] != '#') continue;
        acc = compositeOver(parseColorString(s, acc), acc);
    }
    return acc;
}

void UiEditorApp::applyHit(const EditorHit& hit) {
    const NodePath* primaryPtr = m_selection.primary();
    const NodePath primary = primaryPtr ? *primaryPtr : NodePath{};
    const bool haveNode = primaryPtr && m_doc.isValid(primary);

    switch (hit.action) {
        case EditorHit::Action::SelectNode: {
            const bool additive = glfwGetKey(m_window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
                                  glfwGetKey(m_window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS;
            if (additive) {
                m_selection.toggle(hit.path);
            } else {
                m_selection.set(hit.path);
            }
            cancelTextEdit();
            break;
        }
        case EditorHit::Action::StepRectX:
        case EditorHit::Action::StepRectY: {
            if (!haveNode) break;
            const bool xAxis = hit.action == EditorHit::Action::StepRectX;
            if (nudgeSelection(m_doc, m_selection, xAxis ? hit.delta : 0.0f,
                               xAxis ? 0.0f : hit.delta)) {
                m_history.commit(m_doc, "move");
            }
            break;
        }
        case EditorHit::Action::StepRectW:
        case EditorHit::Action::StepRectH: {
            if (!haveNode) break;
            const UiRect r = m_doc.absoluteRect(primary);
            const bool width = hit.action == EditorHit::Action::StepRectW;
            const float w = width ? std::max(4.0f, r.width() + hit.delta) : r.width();
            const float h = width ? r.height() : std::max(4.0f, r.height() + hit.delta);
            m_doc.setAbsoluteRect(primary, UiRect::fromXYWH(r.minX, r.minY, w, h));
            m_history.commit(m_doc, "resize");
            break;
        }
        case EditorHit::Action::StepFloatProp: {
            if (!haveNode) break;
            const json* node = m_doc.nodeAt(primary);
            const float next = nodeFloat(*node, hit.key.c_str(), 0.0f) + hit.delta;
            m_doc.setProperty(primary, hit.key, json(next));
            m_history.commit(m_doc, "set " + hit.key);
            break;
        }
        case EditorHit::Action::SetColorProp:
            if (!haveNode) break;
            m_doc.setProperty(primary, hit.key, json(colorToHex(hit.colorValue)));
            // Re-seed the picker so the swatch that was clicked becomes the new
            // base its harmony is generated from.
            m_pickerHsv = rgbToHsv(hit.colorValue);
            m_history.commit(m_doc, "set " + hit.key);
            break;
        case EditorHit::Action::TogglePicker: {
            if (!haveNode) break;
            if (m_colorKey == hit.key && m_colorPath == primary) {
                m_colorKey.clear();
                break;
            }
            m_colorKey = hit.key;
            m_colorPath = primary;
            const json* node = m_doc.nodeAt(primary);
            m_pickerHsv = rgbToHsv(nodeColor(*node, hit.key.c_str(), UiColor{0.5f, 0.5f, 0.5f, 1}));
            // Opening the picker on an unset property writes the current value
            // so there is something to edit; "--" puts it back.
            if (!node->contains(hit.key)) {
                m_doc.setProperty(primary, hit.key, json(colorToHex(hsvToRgb(m_pickerHsv, 1.0f))));
                m_history.commit(m_doc, "set " + hit.key);
            }
            break;
        }
        case EditorHit::Action::SetHarmony: {
            const std::vector<Harmony>& schemes = allHarmonies();
            const std::size_t index = static_cast<std::size_t>(hit.delta);
            if (index < schemes.size()) m_harmony = schemes[index];
            break;
        }
        case EditorHit::Action::SetBoolProp:
            if (!haveNode) break;
            m_doc.setProperty(primary, hit.key, json(hit.delta > 0.5f));
            m_history.commit(m_doc, "set " + hit.key);
            break;
        case EditorHit::Action::ClearProp:
            if (!haveNode) break;
            if (m_doc.eraseProperty(primary, hit.key)) {
                m_history.commit(m_doc, "clear " + hit.key);
            }
            break;
        case EditorHit::Action::BeginTextEdit: {
            if (!haveNode) break;
            commitTextEdit();
            m_editing = true;
            m_editingPath = primary;
            m_editingKey = hit.key;
            m_editingBuffer = hit.key == "id" ? m_doc.idOf(primary)
                                              : nodeString(*m_doc.nodeAt(primary), hit.key.c_str());
            break;
        }
        case EditorHit::Action::ReorderNode: {
            NodePath moved;
            if (reorderSibling(m_doc, hit.path, static_cast<int>(hit.delta), moved)) {
                m_selection.set(moved);
                m_history.commit(m_doc, "reorder");
            }
            break;
        }
        case EditorHit::Action::DeleteNode:
            if (deleteSelection(m_doc, m_selection)) {
                cancelTextEdit();
                m_history.commit(m_doc, "delete");
                m_statusMessage = "Deleted selection";
            }
            break;
    }
    m_uiDirty = true;
}

// ─── Input ───────────────────────────────────────────────────────────────────

bool UiEditorApp::keyPressed(int glfwKey) {
    if (glfwKey < 0 || glfwKey >= static_cast<int>(m_prevKeyDown.size())) return false;
    const bool down = glfwGetKey(m_window, glfwKey) == GLFW_PRESS;
    const bool wasDown = m_prevKeyDown[static_cast<std::size_t>(glfwKey)] != 0;
    m_prevKeyDown[static_cast<std::size_t>(glfwKey)] = down ? 1 : 0;
    return down && !wasDown;
}

void UiEditorApp::handleCanvasInput() {
    const float mx = m_uiInput.mousePx.x;
    const float my = m_uiInput.mousePx.y;
    const bool leftDown = m_uiInput.button(UiMouseButton::Left).down;
    const bool leftPressed = m_uiInput.button(UiMouseButton::Left).pressed;
    const bool overCanvas = mx >= m_canvasX && mx < m_canvasX + m_canvasW && my >= m_canvasY &&
                            my < m_canvasY + m_canvasH;
    const UiVec2 doc = screenToDoc(mx, my);

    // Continue an in-flight gesture.
    if ((m_dragging || m_resizing) && leftDown) {
        const float dx = doc.x - m_gestureStartDoc.x;
        const float dy = doc.y - m_gestureStartDoc.y;

        if (m_resizing) {
            if (m_gestureStartRects.empty()) return;
            const NodePath* primary = m_selection.primary();
            if (!primary || !m_doc.isValid(*primary)) return;
            const UiRect o = m_gestureStartRects.front();
            float minX = o.minX, minY = o.minY, maxX = o.maxX, maxY = o.maxY;
            const bool left = m_resizeHandle == 0 || m_resizeHandle == 6 || m_resizeHandle == 7;
            const bool right = m_resizeHandle == 2 || m_resizeHandle == 3 || m_resizeHandle == 4;
            const bool top = m_resizeHandle == 0 || m_resizeHandle == 1 || m_resizeHandle == 2;
            const bool bottom = m_resizeHandle == 4 || m_resizeHandle == 5 || m_resizeHandle == 6;
            if (left) minX = m_snap.snapEdge(o.minX + dx, false, m_snapSettings);
            if (right) maxX = m_snap.snapEdge(o.maxX + dx, false, m_snapSettings);
            if (top) minY = m_snap.snapEdge(o.minY + dy, true, m_snapSettings);
            if (bottom) maxY = m_snap.snapEdge(o.maxY + dy, true, m_snapSettings);
            constexpr float kMinW = 8.0f, kMinH = 6.0f;
            if (maxX - minX < kMinW) (left ? minX : maxX) = left ? maxX - kMinW : minX + kMinW;
            if (maxY - minY < kMinH) (top ? minY : maxY) = top ? maxY - kMinH : minY + kMinH;
            m_doc.setAbsoluteRect(*primary, UiRect{minX, minY, maxX, maxY});
        } else {
            // Snap the primary node, then apply the identical delta to the rest
            // of the selection so a multi-node drag keeps its relative layout.
            const NodePath* primary = m_selection.primary();
            if (!primary || m_gestureStartRects.empty()) return;
            const auto& paths = m_selection.paths();
            const auto it = std::find(paths.begin(), paths.end(), *primary);
            if (it == paths.end()) return;
            const std::size_t primaryIndex = static_cast<std::size_t>(it - paths.begin());
            if (primaryIndex >= m_gestureStartRects.size()) return;

            const UiRect po = m_gestureStartRects[primaryIndex];
            const UiRect moved =
                UiRect::fromXYWH(po.minX + dx, po.minY + dy, po.width(), po.height());
            const UiVec2 snapped = m_snap.snapPosition(moved, m_snapSettings);
            const float appliedDx = snapped.x - po.minX;
            const float appliedDy = snapped.y - po.minY;

            for (std::size_t i = 0; i < paths.size() && i < m_gestureStartRects.size(); ++i) {
                if (!m_doc.isValid(paths[i])) continue;
                const UiRect o = m_gestureStartRects[i];
                m_doc.setAbsoluteRect(paths[i],
                                      UiRect::fromXYWH(o.minX + appliedDx, o.minY + appliedDy,
                                                       o.width(), o.height()));
            }
        }
        return;
    }

    // Gesture release.
    if ((m_dragging || m_resizing) && !leftDown) {
        m_history.commit(m_doc, m_resizing ? "resize" : "move");
        m_dragging = false;
        m_resizing = false;
        m_resizeHandle = -1;
        m_gestureStartRects.clear();
        m_snap.clearLines();
        m_uiDirty = true;
        return;
    }

    if (!leftPressed || !overCanvas) return;

    if (m_placing) {
        placeNode(doc.x, doc.y);
        return;
    }

    commitTextEdit();

    const int handle = hitTestHandle(doc.x, doc.y);
    if (handle >= 0) {
        m_resizing = true;
        m_resizeHandle = handle;
        m_gestureStartDoc = doc;
        m_gestureStartRects.clear();
        m_gestureStartRects.push_back(m_doc.absoluteRect(*m_selection.primary()));
        m_snap.collectTargets(m_doc, m_selection.paths());
        return;
    }

    NodePath hitPath;
    const bool hit = m_doc.hitTest(doc.x, doc.y, hitPath);
    const bool additive = glfwGetKey(m_window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
                          glfwGetKey(m_window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS;

    if (!hit || hitPath.empty()) {
        // The root is the design surface itself: clicking it clears the selection
        // rather than selecting a node the user can't move.
        if (!additive) m_selection.clear();
        m_uiDirty = true;
        return;
    }

    if (additive) {
        m_selection.toggle(hitPath);
    } else if (!m_selection.contains(hitPath)) {
        m_selection.set(hitPath);
    }
    m_uiDirty = true;

    if (m_selection.empty()) return;
    m_dragging = true;
    m_gestureStartDoc = doc;
    m_gestureStartRects.clear();
    for (const NodePath& p : m_selection.paths()) {
        m_gestureStartRects.push_back(m_doc.absoluteRect(p));
    }
    m_snap.collectTargets(m_doc, m_selection.paths());
}

void UiEditorApp::handlePanelClicks() {
    if (!m_uiInput.button(UiMouseButton::Left).pressed) return;
    const float mx = m_uiInput.mousePx.x;
    const float my = m_uiInput.mousePx.y;
    for (const EditorHit& hit : m_hits) {
        if (!hit.rect.contains(mx, my)) continue;
        applyHit(hit);
        return;
    }
}

void UiEditorApp::handleColorDrag() {
    const bool down = m_uiInput.button(UiMouseButton::Left).down;
    const bool pressed = m_uiInput.button(UiMouseButton::Left).pressed;
    const float mx = m_uiInput.mousePx.x;
    const float my = m_uiInput.mousePx.y;

    if (m_colorDrag != ColorDrag::None && !down) {
        // One undo step per gesture, not per frame of the drag.
        m_history.commit(m_doc, "set " + m_colorKey);
        m_colorDrag = ColorDrag::None;
        m_uiDirty = true;
        return;
    }

    if (m_colorKey.empty()) return;

    if (pressed && m_colorDrag == ColorDrag::None) {
        if (m_svRect.contains(mx, my)) {
            m_colorDrag = ColorDrag::SaturationValue;
        } else if (m_hueRect.contains(mx, my)) {
            m_colorDrag = ColorDrag::Hue;
        } else if (m_alphaRect.contains(mx, my)) {
            m_colorDrag = ColorDrag::Alpha;
        }
    }
    if (m_colorDrag == ColorDrag::None || !down) return;

    const NodePath* primary = m_selection.primary();
    if (!primary || !m_doc.isValid(*primary)) {
        m_colorDrag = ColorDrag::None;
        return;
    }
    const json* node = m_doc.nodeAt(*primary);
    float alpha = nodeColor(*node, m_colorKey.c_str(), UiColor{0, 0, 0, 1}).a;

    auto frac = [](float value, float lo, float hi) {
        if (hi <= lo) return 0.0f;
        return std::clamp((value - lo) / (hi - lo), 0.0f, 1.0f);
    };

    switch (m_colorDrag) {
        case ColorDrag::SaturationValue:
            m_pickerHsv.s = frac(mx, m_svRect.minX, m_svRect.maxX);
            m_pickerHsv.v = 1.0f - frac(my, m_svRect.minY, m_svRect.maxY);
            break;
        case ColorDrag::Hue:
            m_pickerHsv.h = wrapHue(frac(my, m_hueRect.minY, m_hueRect.maxY) * 360.0f);
            break;
        case ColorDrag::Alpha:
            alpha = 1.0f - frac(my, m_alphaRect.minY, m_alphaRect.maxY);
            break;
        case ColorDrag::None:
            return;
    }
    setColorProperty(m_colorKey, hsvToRgb(m_pickerHsv, alpha), /*commitToHistory=*/false);
}

void UiEditorApp::handleKeyboard() {
    const bool ctrl = glfwGetKey(m_window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS ||
                      glfwGetKey(m_window, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS;
    const bool shift = glfwGetKey(m_window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
                       glfwGetKey(m_window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS;

    // ── Text entry swallows everything except Enter/Escape/Backspace ──────────
    if (m_editing) {
        for (const std::uint32_t cp : m_uiInput.textInput) {
            if (cp >= 0x20 && cp < 0x7F) m_editingBuffer.push_back(static_cast<char>(cp));
        }
        if (keyPressed(GLFW_KEY_BACKSPACE) && !m_editingBuffer.empty()) m_editingBuffer.pop_back();
        if (keyPressed(GLFW_KEY_ENTER) || keyPressed(GLFW_KEY_KP_ENTER)) commitTextEdit();
        if (keyPressed(GLFW_KEY_ESCAPE)) cancelTextEdit();
        // Keep the edge state of the shortcut keys current so releasing the field
        // doesn't immediately fire whatever was held during typing.
        keyPressed(GLFW_KEY_Z);
        keyPressed(GLFW_KEY_Y);
        keyPressed(GLFW_KEY_C);
        keyPressed(GLFW_KEY_V);
        keyPressed(GLFW_KEY_D);
        keyPressed(GLFW_KEY_S);
        keyPressed(GLFW_KEY_G);
        keyPressed(GLFW_KEY_DELETE);
        return;
    }

    if (ctrl) {
        if (keyPressed(GLFW_KEY_Z)) {
            if (shift) {
                doRedo();
            } else {
                doUndo();
            }
        }
        if (keyPressed(GLFW_KEY_Y)) doRedo();
        if (keyPressed(GLFW_KEY_S)) doSave();
        if (keyPressed(GLFW_KEY_C)) {
            m_clipboard = copySelection(m_doc, m_selection);
            m_statusMessage = m_clipboard.empty() ? "Nothing to copy"
                                                  : "Copied " + std::to_string(m_clipboard.size());
        }
        if (keyPressed(GLFW_KEY_V) && !m_clipboard.empty()) {
            // Paste into the primary selection's parent so a copy lands beside it.
            NodePath parent;
            if (const NodePath* primary = m_selection.primary();
                primary && !primary->empty() && m_doc.isValid(*primary)) {
                parent = *primary;
                parent.pop_back();
            }
            if (pasteInto(m_doc, parent, m_clipboard, 12.0f, 12.0f, m_selection)) {
                m_history.commit(m_doc, "paste");
                m_statusMessage = "Pasted";
            }
        }
        if (keyPressed(GLFW_KEY_D) && duplicateSelection(m_doc, m_selection, 12.0f, 12.0f)) {
            m_history.commit(m_doc, "duplicate");
            m_statusMessage = "Duplicated";
        }
        // Consume the bare-key toggles so Ctrl+S doesn't also flip edge snapping.
        keyPressed(GLFW_KEY_G);
        m_uiDirty = true;
        return;
    }

    if (keyPressed(GLFW_KEY_G)) {
        m_snapSettings.grid = !m_snapSettings.grid;
        m_uiDirty = true;
    }
    if (keyPressed(GLFW_KEY_S)) {
        m_snapSettings.edges = !m_snapSettings.edges;
        m_uiDirty = true;
    }

    const float step = shift ? 10.0f : 1.0f;
    float dx = 0.0f, dy = 0.0f;
    if (keyPressed(GLFW_KEY_LEFT)) dx -= step;
    if (keyPressed(GLFW_KEY_RIGHT)) dx += step;
    if (keyPressed(GLFW_KEY_UP)) dy -= step;
    if (keyPressed(GLFW_KEY_DOWN)) dy += step;
    if ((dx != 0.0f || dy != 0.0f) && nudgeSelection(m_doc, m_selection, dx, dy)) {
        m_history.commit(m_doc, "nudge");
        m_uiDirty = true;
    }

    if (keyPressed(GLFW_KEY_DELETE) && deleteSelection(m_doc, m_selection)) {
        m_history.commit(m_doc, "delete");
        m_statusMessage = "Deleted selection";
        m_uiDirty = true;
    }

    if (keyPressed(GLFW_KEY_ESCAPE)) {
        if (m_placing) {
            m_placing = false;
            m_statusMessage.clear();
            m_uiDirty = true;
        } else {
            glfwSetWindowShouldClose(m_window, GLFW_TRUE);
        }
    }
}

// ─── Lifecycle ───────────────────────────────────────────────────────────────

bool UiEditorApp::onInit() {
    if (!loadFonts(resolveAssetPath("assets/fonts/Inter-Regular.ttf"),
                   resolveAssetPath("assets/fonts/Inter-Bold.ttf"),
                   resolveAssetPath("assets/fonts/Inter-Italic.ttf"),
                   resolveAssetPath("assets/fonts/Inter-Regular.ttf"), 16.0f, 15.0f)) {
        return false;
    }

    m_prevKeyDown.assign(static_cast<std::size_t>(GLFW_KEY_LAST) + 1, 0);

    // Open the shipped sample if it's there; otherwise start from a small seed so
    // the canvas is never blank on first run.
    std::string error;
    const std::string samplePath = resolveAssetPath(kDefaultDocument);
    if (std::filesystem::exists(samplePath) && m_doc.loadFile(samplePath, &error)) {
        m_documentPath = samplePath;
        m_statusMessage = "Loaded " + samplePath;
    } else {
        NodePath panel;
        m_doc.addNode({}, "Panel", 60, 40, panel);
        m_doc.setProperty(panel, "id", "main_panel");
        m_doc.setProperty(panel, "width", 360);
        m_doc.setProperty(panel, "height", 220);
        NodePath title;
        m_doc.addNode(panel, "Label", 14, 12, title);
        m_doc.setProperty(title, "id", "title");
        m_doc.setProperty(title, "text", "<b>New Document</b>");
        NodePath confirm;
        m_doc.addNode(panel, "Button", 14, 168, confirm);
        m_doc.setProperty(confirm, "id", "confirm");
        m_statusMessage = "New document";
    }
    m_history.reset(m_doc);

    int fbW = 0, fbH = 0;
    framebufferSize(fbW, fbH);
    m_lastFbW = fbW;
    m_lastFbH = fbH;
    layoutPanes(fbW, fbH);
    buildWidgetTree(fbW, fbH);
    m_uiDirty = false;
    return true;
}

void UiEditorApp::onTick(float dt) {
    (void)dt;
    int fbW = 0, fbH = 0;
    framebufferSize(fbW, fbH);
    if (fbW <= 0 || fbH <= 0) return;

    if (fbW != m_lastFbW || fbH != m_lastFbH) {
        m_lastFbW = fbW;
        m_lastFbH = fbH;
        layoutPanes(fbW, fbH);
        m_uiDirty = true;
    }

    handleKeyboard();
    // The picker's square/bars are continuous-drag targets rather than click
    // regions, so they get first refusal on the press.
    handleColorDrag();
    if (m_colorDrag == ColorDrag::None) handlePanelClicks();
    handleCanvasInput();

    if (m_uiDirty) {
        // Rebuild the toolbar/palette so their active states track the model.
        // The document panes are immediate-mode and need no rebuild.
        layoutPanes(fbW, fbH);
        buildWidgetTree(fbW, fbH);
        m_uiDirty = false;
    }
}

void UiEditorApp::onRender(float dt) {
    (void)dt;
    int fbW = 0, fbH = 0;
    framebufferSize(fbW, fbH);
    if (fbW <= 0 || fbH <= 0) return;

    beginFrameDraw();

    m_uiDrawList.addRectFilled(
        UiRect::fromXYWH(0, 0, static_cast<float>(fbW), static_cast<float>(fbH)), kEditorBg);
    m_uiDrawList.addRectFilled(
        UiRect::fromXYWH(0, kToolbarH, kPaletteW, static_cast<float>(fbH) - kToolbarH), kPanelBg);

    drawCanvas();

    // The hit list is rebuilt every frame alongside the geometry it describes.
    m_hits.clear();
    drawOutliner();
    drawProperties();

    const float fW = static_cast<float>(fbW);
    const float fH = static_cast<float>(fbH);
    m_uiDrawList.addRectFilled({kPaletteW - 1, kToolbarH, kPaletteW, fH}, {0.16f, 0.17f, 0.22f, 1.0f});
    m_uiDrawList.addRectFilled({fW - kPropsW, kToolbarH, fW - kPropsW + 1.0f, fH},
                               {0.16f, 0.17f, 0.22f, 1.0f});
    m_uiDrawList.addRectFilled({0.0f, kToolbarH - 1.0f, fW, kToolbarH}, {0.16f, 0.17f, 0.22f, 1.0f});

    drawStatusBar(fbW, fbH);

    submitFrame(m_camera);
}

}  // namespace odai::tools::ui_editor
