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
#include <fstream>
#include <sstream>

namespace odai::tools::ui_editor {

using namespace ui;

// ─── Layout constants ────────────────────────────────────────────────────────
static constexpr float kPaletteW = 184.0f;
static constexpr float kPropsW   = 264.0f;
static constexpr float kToolbarH = 46.0f;

// ─── Colours ─────────────────────────────────────────────────────────────────
static constexpr UiColor kEditorBg  {0.100f,0.102f,0.110f,1.000f};
static constexpr UiColor kCanvasBg  {0.140f,0.142f,0.158f,1.000f};
static constexpr UiColor kGridLine  {0.175f,0.178f,0.196f,1.000f};
static constexpr UiColor kGridMajor {0.160f,0.163f,0.182f,1.000f};
static constexpr UiColor kPanelBg   {0.068f,0.072f,0.090f,0.980f};
static constexpr UiColor kPanelBord {0.140f,0.145f,0.190f,1.000f};
static constexpr UiColor kBlue      {0.200f,0.600f,1.000f,1.000f};
static constexpr UiColor kSnapLine  {0.000f,0.780f,0.900f,0.600f};  // cyan guide lines
static constexpr UiColor kText      {0.820f,0.840f,0.880f,1.000f};
static constexpr UiColor kTextDim   {0.500f,0.520f,0.570f,1.000f};
static constexpr UiColor kGold      {0.900f,0.720f,0.380f,1.000f};
static constexpr UiColor kDeleteRed {0.800f,0.200f,0.180f,1.000f};

static const struct { const char* name; UiColor accent; } kPaletteItems[] = {
    {"Panel",      {0.30f,0.50f,0.90f,1.0f}},
    {"Label",      {0.90f,0.85f,0.30f,1.0f}},
    {"Button",     {0.30f,0.80f,0.45f,1.0f}},
    {"ProgressBar",  {0.25f,0.80f,0.38f,1.0f}},
    {"IconSlot", {0.90f,0.55f,0.20f,1.0f}},
    {"FeedPanel", {0.55f,0.35f,0.90f,1.0f}},
    {"EntityCard",  {0.30f,0.75f,0.95f,1.0f}},
    {"Minimap",    {0.20f,0.85f,0.72f,1.0f}},
};
static constexpr int kNumPalette = 8;

static constexpr UiColor kColorPresets[] = {
    {0.05f,0.10f,0.15f,0.88f},{0.12f,0.08f,0.05f,0.88f},
    {0.05f,0.12f,0.06f,0.88f},{0.08f,0.05f,0.14f,0.88f},
    {0.14f,0.06f,0.06f,0.88f},{0.08f,0.12f,0.16f,0.88f},
    {0.10f,0.10f,0.10f,0.88f},{0.22f,0.20f,0.18f,0.88f},
};

// ─── Defaults per widget type ─────────────────────────────────────────────────
float UiEditorApp::defaultW(const std::string& t) {
    if (t=="IconSlot") return 54; if (t=="ProgressBar")  return 220;
    if (t=="FeedPanel") return 300; if (t=="EntityCard")  return 220;
    if (t=="Minimap")    return 160; if (t=="Label")      return 160;
    if (t=="Button")     return 140; return 200;
}
float UiEditorApp::defaultH(const std::string& t) {
    if (t=="IconSlot") return 54; if (t=="ProgressBar")  return 18;
    if (t=="FeedPanel") return 180; if (t=="EntityCard")  return 90;
    if (t=="Minimap")    return 160; if (t=="Label")      return 28;
    if (t=="Button")     return 36; return 80;
}
UiColor UiEditorApp::defaultBg(const std::string& t) {
    if (t=="Button")    return {0.10f,0.18f,0.28f,0.90f};
    if (t=="ProgressBar") return {0.05f,0.12f,0.06f,1.00f};
    if (t=="IconSlot")return {0.06f,0.08f,0.15f,0.92f};
    if (t=="Minimap")   return {0.02f,0.04f,0.08f,0.95f};
    if (t=="EntityCard") return {0.06f,0.10f,0.18f,0.88f};
    return {0.05f,0.10f,0.15f,0.88f};
}

// ─── Snap ────────────────────────────────────────────────────────────────────
// Finds the best 1-D snap for `val` (position of widget anchor at featureOffset).
// Tries each feature offset in {0, span/2, span} against the given targets.
// Records a snap-line (using screen axis = isHoriz) when the best snap is found.
float UiEditorApp::snap1D(float val, float span,
                           const std::vector<float>& targets, float threshold,
                           bool isHoriz, float gridSize, bool doGrid, bool doEdges) {
    float bestDist = threshold + 1.0f;
    float bestVal  = val;
    float snapPos  = -1.0f;

    float offsets[3] = {0.0f, span * 0.5f, span};

    auto trySnap = [&](float target, float offset) {
        float snapped = target - offset;
        float dist    = std::abs(val - snapped);
        if (dist < bestDist) {
            bestDist = dist;
            bestVal  = snapped;
            snapPos  = target;
        }
    };

    // 1. Edge snap (widget edges + canvas boundaries)
    if (doEdges) {
        for (float fo : offsets)
            for (float t : targets)
                trySnap(t, fo);
    }

    // 2. Grid snap (lower priority: only if edge snap didn't fire)
    if (doGrid && bestDist > threshold) {
        for (float fo : offsets) {
            float gridTarget = std::round((val + fo) / gridSize) * gridSize;
            trySnap(gridTarget, fo);
        }
    }

    if (bestDist <= threshold && snapPos >= -0.5f)
        m_snapLines.push_back({isHoriz, snapPos});

    return bestVal;
}

UiVec2 UiEditorApp::applySnap(float wx, float wy, float ww, float wh, int excludeIdx) {
    m_snapLines.clear();

    // Build snap-target lists
    std::vector<float> xTargets, yTargets;
    // Canvas boundaries + center
    xTargets.push_back(0.0f);
    xTargets.push_back(m_canvasW);
    xTargets.push_back(m_canvasW * 0.5f);
    yTargets.push_back(0.0f);
    yTargets.push_back(m_canvasH);
    yTargets.push_back(m_canvasH * 0.5f);

    for (int i = 0; i < (int)m_widgets.size(); ++i) {
        if (i == excludeIdx) continue;
        const auto& w = m_widgets[i];
        xTargets.push_back(w.x);
        xTargets.push_back(w.x + w.w * 0.5f);
        xTargets.push_back(w.x + w.w);
        yTargets.push_back(w.y);
        yTargets.push_back(w.y + w.h * 0.5f);
        yTargets.push_back(w.y + w.h);
    }

    float snappedX = snap1D(wx, ww, xTargets, m_snapThresh,
                             false, m_gridSnap, m_snapGrid, m_snapEdges);
    float snappedY = snap1D(wy, wh, yTargets, m_snapThresh,
                             true,  m_gridSnap, m_snapGrid, m_snapEdges);
    return {snappedX, snappedY};
}

// ─── Resize handle hit test ──────────────────────────────────────────────────
// Handle order: TL=0 TC=1 TR=2  MR=3  BR=4 BC=5 BL=6  ML=7
int UiEditorApp::hitTestHandle(float cx, float cy) const {
    if (m_selected < 0 || m_selected >= (int)m_widgets.size()) return -1;
    const auto& w  = m_widgets[m_selected];
    float mx = w.x + w.w * 0.5f;
    float my = w.y + w.h * 0.5f;
    float ex = w.x + w.w;
    float ey = w.y + w.h;

    UiVec2 handles[8] = {
        {w.x, w.y}, {mx, w.y}, {ex, w.y},
        {ex, my},
        {ex, ey}, {mx, ey}, {w.x, ey},
        {w.x, my},
    };
    constexpr float kR = 7.0f;
    for (int i = 0; i < 8; ++i) {
        float dx = cx - handles[i].x;
        float dy = cy - handles[i].y;
        if (dx*dx + dy*dy <= kR*kR) return i;
    }
    return -1;
}

// ─── Canvas drawing ───────────────────────────────────────────────────────────
void UiEditorApp::drawDesignWidget(const DesignWidget& w, bool selected) {
    float sx = m_canvasX + w.x;
    float sy = m_canvasY + w.y;
    UiRect r = UiRect::fromXYWH(sx, sy, w.w, w.h);

    if (w.shadow)
        m_uiDrawList.addDropShadow(r, {0,0,0,0.55f}, w.shadowBlur, 0, 4);

    if (w.cornerR > 0.5f) {
        m_uiDrawList.addRoundRectFilled(r, w.bg, w.cornerR);
        m_uiDrawList.addRoundRect(r, w.border, w.cornerR, w.borderW);
    } else {
        m_uiDrawList.addRectFilled(r, w.bg);
        m_uiDrawList.addRect(r, w.border, w.borderW);
    }

    // Type-specific visuals
    if (w.type == "ProgressBar") {
        m_uiDrawList.addRoundRectFilledHGradient(
            UiRect::fromXYWH(sx+2, sy+2, (w.w-4)*0.65f, w.h-4),
            {0.18f,0.78f,0.32f,1.0f}, {0.10f,0.45f,0.18f,1.0f}, 2.0f);
    } else if (w.type == "IconSlot") {
        float icR = std::min(w.w, w.h) * 0.32f;
        m_uiDrawList.addCircleFilled({sx + w.w*0.5f, sy + w.h*0.5f}, icR,
            {0.90f,0.55f,0.20f,0.80f});
    } else if (w.type == "Minimap") {
        float rad = std::min(w.w, w.h) * 0.46f;
        UiVec2 ctr{sx + w.w*0.5f, sy + w.h*0.5f};
        m_uiDrawList.addCircleFilled(ctr, rad, {0.02f,0.04f,0.08f,0.95f});
        m_uiDrawList.addCircle(ctr, rad, {0.85f,0.70f,0.30f,0.50f}, 1.5f);
        m_uiDrawList.addCircleFilled(ctr, 4.0f, {1.0f,1.0f,1.0f,0.9f});
    } else if (w.type == "EntityCard") {
        float ph = w.h - 10;
        m_uiDrawList.addRoundRectFilled(UiRect::fromXYWH(sx+5, sy+5, ph, ph),
            {0.10f,0.16f,0.26f,0.9f}, 3.0f);
        float bx = sx + ph + 10, bw = w.w - ph - 15;
        m_uiDrawList.addRoundRectFilledHGradient(
            UiRect::fromXYWH(bx, sy+20, bw, 10),
            {0.18f,0.78f,0.32f,1.0f}, {0.10f,0.45f,0.18f,1.0f}, 2.0f);
        m_uiDrawList.addRoundRectFilled(
            UiRect::fromXYWH(bx, sy+35, bw, 8), {0.10f,0.30f,0.90f,0.90f}, 2.0f);
    } else if (w.type == "FeedPanel") {
        float lh = m_uiFont.lineHeightPx();
        int nLines = std::max(1, (int)((w.h - 20) / lh));
        for (int i = 0; i < nLines; ++i) {
            float lw2 = w.w * (0.55f + 0.35f * std::fmod(i * 0.618f, 1.0f));
            m_uiDrawList.addRectFilled(
                UiRect::fromXYWH(sx+6, sy+10+i*lh, lw2-6, lh-4),
                {0.3f,0.35f,0.45f,0.20f});
        }
    }

    // Type label
    if (m_uiFonts.regular && w.w > 40 && w.h > 20) {
        const std::string& lbl = w.label.empty() ? w.type : w.label;
        float tw = m_uiFonts.regular->measureText(lbl);
        float lh = m_uiFonts.regular->lineHeightPx();
        if (tw < w.w - 6)
            m_uiDrawList.addText(*m_uiFonts.regular, lbl,
                {sx + (w.w-tw)*0.5f, sy + (w.h-lh)*0.5f}, {0.9f,0.9f,0.9f,0.55f});
    }

    // Selection outline
    if (selected) {
        UiRect sel = {r.minX-2, r.minY-2, r.maxX+2, r.maxY+2};
        m_uiDrawList.addRect(sel, kBlue, 2.0f);

        // 8 resize handles: TL TC TR  MR  BR BC BL  ML
        float mx2 = (r.minX+r.maxX)*0.5f, my2 = (r.minY+r.maxY)*0.5f;
        UiVec2 hpts[8] = {
            {r.minX,r.minY},{mx2,r.minY},{r.maxX,r.minY},
            {r.maxX,my2},
            {r.maxX,r.maxY},{mx2,r.maxY},{r.minX,r.maxY},
            {r.minX,my2},
        };
        constexpr float hs = 6.0f;
        for (const auto& h : hpts) {
            UiRect hr = UiRect::fromXYWH(h.x-hs*0.5f, h.y-hs*0.5f, hs, hs);
            m_uiDrawList.addRectFilled(hr, {0.0f,0.0f,0.0f,1.0f});
            m_uiDrawList.addRect(hr, kBlue, 1.5f);
        }
    }
}

void UiEditorApp::drawCanvas(int /*fbW*/, int /*fbH*/) {
    UiRect cvRect = UiRect::fromXYWH(m_canvasX, m_canvasY, m_canvasW, m_canvasH);
    m_uiDrawList.pushClip(cvRect);
    m_uiDrawList.addRectFilled(cvRect, kCanvasBg);

    // Grid — minor every 10 px, major every 100 px
    const float step = m_gridSnap;
    for (float gx = 0; gx <= m_canvasW; gx += step) {
        bool major = (std::fmod(gx, step * 10.0f) < 0.5f);
        m_uiDrawList.addRectFilled(
            {m_canvasX + gx, m_canvasY, m_canvasX + gx + 0.5f, m_canvasY + m_canvasH},
            major ? kGridMajor : kGridLine);
    }
    for (float gy = 0; gy <= m_canvasH; gy += step) {
        bool major = (std::fmod(gy, step * 10.0f) < 0.5f);
        m_uiDrawList.addRectFilled(
            {m_canvasX, m_canvasY + gy, m_canvasX + m_canvasW, m_canvasY + gy + 0.5f},
            major ? kGridMajor : kGridLine);
    }

    // Design widgets
    for (int i = 0; i < (int)m_widgets.size(); ++i)
        drawDesignWidget(m_widgets[i], i == m_selected);

    // Snap guide lines (drawn over widgets, below ghost)
    for (const auto& sl : m_snapLines) {
        if (!sl.horiz) {
            // Vertical line at canvas-space x = sl.pos
            float sx = m_canvasX + sl.pos;
            m_uiDrawList.addRectFilled(
                {sx - 0.75f, m_canvasY, sx + 0.75f, m_canvasY + m_canvasH}, kSnapLine);
            // Small tick marks every 20 px for readability
            for (float ty2 = m_canvasY + 6; ty2 < m_canvasY + m_canvasH; ty2 += 20) {
                m_uiDrawList.addRectFilled({sx-3.5f, ty2, sx+3.5f, ty2+1.5f}, kSnapLine);
            }
        } else {
            // Horizontal line at canvas-space y = sl.pos
            float sy = m_canvasY + sl.pos;
            m_uiDrawList.addRectFilled(
                {m_canvasX, sy - 0.75f, m_canvasX + m_canvasW, sy + 0.75f}, kSnapLine);
            for (float tx2 = m_canvasX + 6; tx2 < m_canvasX + m_canvasW; tx2 += 20) {
                m_uiDrawList.addRectFilled({tx2, sy-3.5f, tx2+1.5f, sy+3.5f}, kSnapLine);
            }
        }
    }

    // Placement ghost
    if (m_placing) {
        float mx = m_uiInput.mousePx.x - m_canvasX;
        float my = m_uiInput.mousePx.y - m_canvasY;
        if (mx >= 0 && my >= 0) {
            float gw = defaultW(m_placeType), gh = defaultH(m_placeType);
            float gx = mx - gw * 0.5f, gy = my - gh * 0.5f;
            // Snap the ghost too
            auto snapped = applySnap(gx, gy, gw, gh, -1);
            gx = snapped.x; gy = snapped.y;

            DesignWidget ghost;
            ghost.type = m_placeType; ghost.name = m_placeType;
            ghost.x = gx; ghost.y = gy; ghost.w = gw; ghost.h = gh;
            ghost.bg = defaultBg(m_placeType); ghost.bg.a *= 0.45f;
            ghost.border = {0.9f,0.9f,0.9f,0.25f};
            drawDesignWidget(ghost, false);
        }
    }

    m_uiDrawList.popClip();
}

// ─── Properties panel (draw-list) ────────────────────────────────────────────
void UiEditorApp::drawPropertiesPanel(int fbW, int fbH) {
    float px = fbW - kPropsW, py = kToolbarH, pw = kPropsW;
    float ph = fbH - kToolbarH;

    m_uiDrawList.addRectFilled(UiRect::fromXYWH(px, py, pw, ph), kPanelBg);
    m_uiDrawList.addRectFilled(UiRect::fromXYWH(px, py, pw, 32), {0.09f,0.10f,0.14f,1.0f});
    m_uiDrawList.addText(m_uiFontBold, "PROPERTIES", {px + 10, py + 8}, kGold);

    float ty = py + 40;

    if (m_selected < 0 || m_selected >= (int)m_widgets.size()) {
        m_uiDrawList.addText(m_uiFont, "No widget selected", {px+10, ty}, kTextDim);
        ty += m_uiFont.lineHeightPx() + 20;
        m_uiDrawList.addText(m_uiFont, "Click palette to add,", {px+10, ty}, kTextDim);
        ty += m_uiFont.lineHeightPx() + 4;
        m_uiDrawList.addText(m_uiFont, "then click canvas to place.", {px+10, ty}, kTextDim);
        return;
    }

    const DesignWidget& w = m_widgets[m_selected];
    char buf[128];

    std::snprintf(buf, sizeof(buf), "Type: %s", w.type.c_str());
    m_uiDrawList.addText(m_uiFontBold, buf, {px+10, ty}, kText);
    ty += m_uiFont.lineHeightPx() + 2;
    std::snprintf(buf, sizeof(buf), "Name: %s", w.name.c_str());
    m_uiDrawList.addText(m_uiFont, buf, {px+10, ty}, kTextDim);
    ty += m_uiFont.lineHeightPx() + 14;

    auto sectionLabel = [&](const char* lbl) {
        m_uiDrawList.addRectFilled(UiRect::fromXYWH(px+6, ty, pw-12, 1), {0.2f,0.22f,0.30f,0.7f});
        ty += 5;
        m_uiDrawList.addText(m_uiFont, lbl, {px+10, ty}, kTextDim);
        ty += m_uiFont.lineHeightPx() + 4;
    };

    // Step-button row: << < [VAL] > >>
    auto stepRow = [&](const char* rowLabel, float value, PropButton::Action action) {
        m_uiDrawList.addText(m_uiFont, rowLabel, {px+10, ty+4}, kText);
        float bx = px + 55, bw = 32, bh = 22, gap = 2;
        float deltas[4] = {-10,-1,1,10};
        const char* lbls[4] = {"<<","<",">",">>"};
        for (int i = 0; i < 4; ++i) {
            UiRect r = UiRect::fromXYWH(bx, ty, bw, bh);
            m_uiDrawList.addRoundRectFilled(r, {0.10f,0.12f,0.17f,0.90f}, 2);
            m_uiDrawList.addRoundRect(r, {0.25f,0.28f,0.38f,0.80f}, 2, 1);
            float lw2 = m_uiFont.measureText(lbls[i]);
            m_uiDrawList.addText(m_uiFont, lbls[i],
                {r.minX+(bw-lw2)*0.5f, r.minY+(bh-m_uiFont.lineHeightPx())*0.5f}, kText);
            PropButton pb; pb.rect = r; pb.action = action; pb.delta = deltas[i];
            m_propBtns.push_back(pb);
            bx += bw + gap;

            if (i == 1) {
                // Value display between < and >
                UiRect vr = UiRect::fromXYWH(bx, ty, 44, bh);
                std::snprintf(buf, sizeof(buf), "%.0f", value);
                m_uiDrawList.addRoundRectFilled(vr, {0.12f,0.13f,0.16f,0.9f}, 2);
                float vw2 = m_uiFont.measureText(buf);
                m_uiDrawList.addText(m_uiFont, buf,
                    {vr.minX+(44-vw2)*0.5f, vr.minY+(bh-m_uiFont.lineHeightPx())*0.5f}, kGold);
                bx += 44 + gap;
            }
        }
        ty += bh + 6;
    };

    sectionLabel("Position");
    stepRow("X", w.x, PropButton::Action::StepX);
    stepRow("Y", w.y, PropButton::Action::StepY);

    ty += 2;
    sectionLabel("Size");
    stepRow("W", w.w, PropButton::Action::StepW);
    stepRow("H", w.h, PropButton::Action::StepH);

    ty += 2;
    sectionLabel("Colour");
    {
        float cx2 = px + 10;
        float sz = 22.0f;
        for (int i = 0; i < 8; ++i) {
            UiRect cr = UiRect::fromXYWH(cx2, ty, sz, sz);
            m_uiDrawList.addRoundRectFilled(cr, kColorPresets[i], 3.0f);
            bool cur = (std::abs(w.bg.r-kColorPresets[i].r)<0.01f &&
                        std::abs(w.bg.g-kColorPresets[i].g)<0.01f &&
                        std::abs(w.bg.b-kColorPresets[i].b)<0.01f);
            m_uiDrawList.addRoundRect(cr, cur ? kBlue : UiColor{0.3f,0.3f,0.4f,0.5f},
                                       3.0f, cur ? 2.0f : 1.0f);
            PropButton pb; pb.rect = cr; pb.action = PropButton::Action::SetColor;
            pb.colorValue = kColorPresets[i];
            m_propBtns.push_back(pb);
            cx2 += sz + 4.0f;
        }
        ty += sz + 10;
    }

    sectionLabel("Corner Radius");
    {
        float cx2 = px + 10;
        float vals[4] = {0,4,8,16};
        const char* vlbls[4] = {"0","4","8","16"};
        for (int i = 0; i < 4; ++i) {
            bool active = (std::abs(w.cornerR - vals[i]) < 0.5f);
            UiRect cr = UiRect::fromXYWH(cx2, ty, 50, 24);
            m_uiDrawList.addRoundRectFilled(cr,
                active ? UiColor{0.15f,0.25f,0.45f,0.95f} : UiColor{0.10f,0.12f,0.17f,0.90f}, 4);
            m_uiDrawList.addRoundRect(cr, active ? kBlue : UiColor{0.25f,0.28f,0.38f,0.80f},
                                       4, active ? 1.5f : 1.0f);
            float lw2 = m_uiFont.measureText(vlbls[i]);
            m_uiDrawList.addText(m_uiFont, vlbls[i],
                {cr.minX+(50-lw2)*0.5f, cr.minY+(24-m_uiFont.lineHeightPx())*0.5f},
                active ? kBlue : kText);
            PropButton pb; pb.rect = cr; pb.action = PropButton::Action::SetCorner;
            pb.delta = vals[i];
            m_propBtns.push_back(pb);
            cx2 += 54;
        }
        ty += 24 + 10;
    }

    sectionLabel("Shadow");
    {
        auto shadowBtn = [&](float bx, const char* lbl, bool match) {
            bool active = (w.shadow == match);
            UiRect r = UiRect::fromXYWH(bx, ty, 70, 26);
            m_uiDrawList.addRoundRectFilled(r,
                active ? UiColor{0.15f,0.25f,0.45f,0.95f} : UiColor{0.10f,0.12f,0.17f,0.90f}, 4);
            m_uiDrawList.addRoundRect(r, active ? kBlue : UiColor{0.25f,0.28f,0.38f,0.80f},
                                       4, active ? 1.5f : 1.0f);
            float lw2 = m_uiFont.measureText(lbl);
            m_uiDrawList.addText(m_uiFont, lbl,
                {r.minX+(70-lw2)*0.5f, r.minY+(26-m_uiFont.lineHeightPx())*0.5f},
                active ? kBlue : kText);
            PropButton pb; pb.rect = r; pb.action = PropButton::Action::ToggleShadow;
            pb.delta = match ? 1.0f : 0.0f;
            m_propBtns.push_back(pb);
        };
        shadowBtn(px+10, "Off", false);
        shadowBtn(px+84, "On",  true);
        ty += 26 + 12;
    }

    // ── Snap info ──
    ty += 4;
    m_uiDrawList.addRectFilled(UiRect::fromXYWH(px+6, ty, pw-12, 1), {0.2f,0.22f,0.30f,0.5f});
    ty += 8;
    std::snprintf(buf, sizeof(buf), "Snap: grid=%s edges=%s  [G/S]",
        m_snapGrid  ? "ON" : "off",
        m_snapEdges ? "ON" : "off");
    m_uiDrawList.addText(m_uiFont, buf, {px+10, ty}, kTextDim);
    ty += m_uiFont.lineHeightPx() + 4;
    std::snprintf(buf, sizeof(buf), "Grid size: %.0fpx", m_gridSnap);
    m_uiDrawList.addText(m_uiFont, buf, {px+10, ty}, kTextDim);

    // Delete button
    float dy = (float)fbH - 60;
    UiRect delR = UiRect::fromXYWH(px+10, dy, pw-20, 34);
    m_uiDrawList.addDropShadow(delR, {0,0,0,0.4f}, 6, 0, 3);
    m_uiDrawList.addRoundRectFilled(delR, {0.25f,0.06f,0.06f,0.90f}, 5);
    m_uiDrawList.addRoundRect(delR, kDeleteRed, 5, 1.5f);
    const char* delLbl = "Delete Widget";
    float dlW = m_uiFontBold.measureText(delLbl);
    m_uiDrawList.addText(m_uiFontBold, delLbl,
        {delR.minX+(delR.width()-dlW)*0.5f, delR.minY+(34-m_uiFontBold.lineHeightPx())*0.5f},
        {1.0f,0.55f,0.50f,1.0f});
    PropButton delBtn; delBtn.rect = delR; delBtn.action = PropButton::Action::Delete;
    m_propBtns.push_back(delBtn);
}

// ─── Widget tree (toolbar + palette) ─────────────────────────────────────────
void UiEditorApp::setupWidgetTree(int fbW, int fbH) {
    m_canvasX = kPaletteW;
    m_canvasY = kToolbarH;
    m_canvasW = fbW - kPaletteW - kPropsW;
    m_canvasH = fbH - kToolbarH;

    auto root = std::make_unique<Widget>();
    root->mousePassthrough = true;
    root->setRect(UiRect::fromXYWH(0, 0, (float)fbW, (float)fbH));

    // ── Toolbar ──
    auto toolbar = std::make_unique<Panel>();
    toolbar->background        = {0.068f,0.072f,0.088f,0.98f};
    toolbar->borderColor       = {0.13f,0.14f,0.18f,1.0f};
    toolbar->borderThicknessPx = 1.0f;
    toolbar->setRect(UiRect::fromXYWH(0, 0, (float)fbW, kToolbarH));

    // App title
    auto title = std::make_unique<Button>(&m_uiFontBold, "odai  UI Editor", [](){});
    title->colorNormal = {0,0,0,0}; title->colorHover = {0,0,0,0};
    title->borderColor = {0,0,0,0}; title->glowSizePx = 0;
    title->labelColor  = kGold;
    title->setRect(UiRect::fromXYWH(12, 6, 180, 34));
    toolbar->addChild(std::move(title));

    // Snap toggle buttons
    auto snapBtn = [&](float bx, const char* lbl, bool active, std::function<void()> fn) {
        auto btn = std::make_unique<Button>(&m_uiFont, lbl, std::move(fn));
        btn->colorNormal    = active ? UiColor{0.10f,0.18f,0.30f,0.90f} : UiColor{0.08f,0.09f,0.12f,0.90f};
        btn->colorHover     = active ? UiColor{0.14f,0.22f,0.38f,0.95f} : UiColor{0.10f,0.12f,0.16f,0.95f};
        btn->borderColor    = active ? kBlue : UiColor{0.25f,0.26f,0.32f,0.55f};
        btn->borderColor.a  = active ? 0.80f : 0.40f;
        btn->labelColor     = active ? kBlue : kTextDim;
        btn->glowSizePx     = active ? 6.0f : 0.0f;
        btn->glowColor      = kBlue;
        btn->cornerRadiusPx = 3.0f;
        btn->setRect(UiRect::fromXYWH(bx, 7, 110, 32));
        toolbar->addChild(std::move(btn));
    };
    snapBtn(200, m_snapGrid  ? "Grid Snap: ON" : "Grid Snap: OFF",
            m_snapGrid,  [this]() { m_snapGrid  = !m_snapGrid;  m_lastFbW = 0; });
    snapBtn(316, m_snapEdges ? "Edge Snap: ON" : "Edge Snap: OFF",
            m_snapEdges, [this]() { m_snapEdges = !m_snapEdges; m_lastFbW = 0; });

    // Save / Load buttons
    auto makeIoBtn = [&](float bx, const char* lbl, std::function<void()> fn) {
        auto btn = std::make_unique<Button>(&m_uiFont, lbl, std::move(fn));
        btn->colorNormal    = {0.08f,0.10f,0.12f,0.90f};
        btn->colorHover     = {0.12f,0.16f,0.20f,0.95f};
        btn->borderColor    = {0.35f,0.55f,0.80f,0.50f};
        btn->labelColor     = {0.50f,0.75f,1.00f,1.00f};
        btn->glowSizePx     = 0.0f;
        btn->cornerRadiusPx = 3.0f;
        btn->setRect(UiRect::fromXYWH(bx, 7, 80, 32));
        toolbar->addChild(std::move(btn));
    };
    makeIoBtn(868, "Save JSON", [this]() {
        saveJson(m_loadedPath.empty() ? "ui_layout.json" : m_loadedPath);
    });
    makeIoBtn(954, "Load JSON", [this]() {
        loadJson(m_loadedPath.empty() ? "ui_layout.json" : m_loadedPath);
    });

    // Placement mode indicator
    if (m_placing) {
        std::string placeLbl = "Placing: " + m_placeType + "  [Esc to cancel]";
        auto placeBtn = std::make_unique<Button>(&m_uiFont, placeLbl,
            [this]() { m_placing = false; m_lastFbW = 0; });
        placeBtn->colorNormal   = {0.08f,0.14f,0.22f,0.90f};
        placeBtn->colorHover    = {0.10f,0.18f,0.28f,0.95f};
        placeBtn->borderColor   = {0.20f,0.60f,1.00f,0.60f};
        placeBtn->glowSizePx    = 8.0f;
        placeBtn->glowColor     = kBlue;
        placeBtn->labelColor    = kBlue;
        placeBtn->cornerRadiusPx = 3.0f;
        placeBtn->setRect(UiRect::fromXYWH(436, 7, 420, 32));
        toolbar->addChild(std::move(placeBtn));
    }

    root->addChild(std::move(toolbar));

    // ── Palette ──
    auto palette = std::make_unique<Panel>();
    palette->background        = kPanelBg;
    palette->borderColor       = kPanelBord;
    palette->borderThicknessPx = 1.0f;
    palette->setRect(UiRect::fromXYWH(0, kToolbarH, kPaletteW, fbH-kToolbarH));

    auto palHeader = std::make_unique<Button>(&m_uiFont, "WIDGET PALETTE", [](){});
    palHeader->colorNormal = {0.09f,0.10f,0.14f,1.0f}; palHeader->colorHover = {0.09f,0.10f,0.14f,1.0f};
    palHeader->borderColor = {0,0,0,0}; palHeader->glowSizePx = 0;
    palHeader->labelColor  = {0.50f,0.55f,0.72f,1.0f};
    palHeader->setRect(UiRect::fromXYWH(0, kToolbarH, kPaletteW, 30));
    palette->addChild(std::move(palHeader));

    float btnY = kToolbarH + 34;
    for (int i = 0; i < kNumPalette; ++i) {
        const auto& item = kPaletteItems[i];
        std::string typeName = item.name;
        auto btn = std::make_unique<Button>(&m_uiFont, std::string("+ ") + item.name,
            [this, typeName]() { m_placeType = typeName; m_placing = true; m_lastFbW = 0; });
        btn->colorNormal    = {0.08f,0.10f,0.14f,0.88f};
        btn->colorHover     = {0.11f,0.14f,0.20f,0.95f};
        btn->colorPressed   = {0.06f,0.08f,0.12f,0.95f};
        btn->borderColor    = {item.accent.r, item.accent.g, item.accent.b, 0.40f};
        btn->labelColor     = item.accent;
        btn->glowColor      = item.accent;
        btn->glowSizePx     = 5.0f;
        btn->cornerRadiusPx = 4.0f;
        btn->setRect(UiRect::fromXYWH(6, btnY, kPaletteW-12, 34));
        palette->addChild(std::move(btn));
        btnY += 38;
    }

    root->addChild(std::move(palette));
    m_uiContext.setRoot(std::move(root));
}

// ─── Lifecycle ────────────────────────────────────────────────────────────────
bool UiEditorApp::onInit() {
    if (!loadFonts(
            resolveAssetPath("assets/fonts/Inter-Regular.ttf"),
            resolveAssetPath("assets/fonts/Inter-Bold.ttf"),
            resolveAssetPath("assets/fonts/Inter-Italic.ttf"),
            resolveAssetPath("assets/fonts/Inter-Regular.ttf"),
            16.0f, 15.0f))
        return false;

    // Seed example widgets
    {
        DesignWidget w; w.type = "Panel"; w.name = "main_panel";
        w.x = 40; w.y = 30; w.w = 320; w.h = 180;
        w.cornerR = 4; w.shadow = true;
        m_widgets.push_back(w);
    }
    {
        DesignWidget w; w.type = "Button"; w.name = "confirm_btn";
        w.x = 120; w.y = 238; w.w = 140; w.h = 36;
        w.bg = {0.10f,0.18f,0.28f,0.90f}; w.cornerR = 4; w.label = "Confirm";
        m_widgets.push_back(w);
    }
    {
        DesignWidget w; w.type = "ProgressBar"; w.name = "progress_bar";
        w.x = 40; w.y = 230; w.w = 70; w.h = 18;
        w.bg = {0.04f,0.10f,0.05f,1.0f};
        m_widgets.push_back(w);
    }

    int fbW, fbH;
    framebufferSize(fbW, fbH);
    setupWidgetTree(fbW, fbH);
    m_lastFbW = fbW; m_lastFbH = fbH;
    return true;
}

void UiEditorApp::onTick(float dt) {
    (void)dt;
    int fbW, fbH;
    framebufferSize(fbW, fbH);

    if (fbW != m_lastFbW || fbH != m_lastFbH) {
        m_canvasX = kPaletteW; m_canvasY = kToolbarH;
        m_canvasW = fbW - kPaletteW - kPropsW;
        m_canvasH = fbH - kToolbarH;
        setupWidgetTree(fbW, fbH);
        m_lastFbW = fbW; m_lastFbH = fbH;
    }

    float mx = m_uiInput.mousePx.x;
    float my = m_uiInput.mousePx.y;
    bool leftDown    = m_uiInput.button(UiMouseButton::Left).down;
    bool leftPressed = m_uiInput.button(UiMouseButton::Left).pressed;

    bool overCanvas  = (mx >= m_canvasX && mx < m_canvasX + m_canvasW &&
                        my >= m_canvasY && my < m_canvasY + m_canvasH);
    float cx = mx - m_canvasX;
    float cy = my - m_canvasY;

    // ── Keyboard toggles ──────────────────────────────────────────────────────
    bool gKey = (glfwGetKey(m_window, GLFW_KEY_G) == GLFW_PRESS);
    if (gKey && !m_prevGKey) { m_snapGrid  = !m_snapGrid;  m_lastFbW = 0; }
    m_prevGKey = gKey;

    bool sKey = (glfwGetKey(m_window, GLFW_KEY_S) == GLFW_PRESS);
    if (sKey && !m_prevSKey) { m_snapEdges = !m_snapEdges; m_lastFbW = 0; }
    m_prevSKey = sKey;

    // ── Resize (continue from previous frame) ─────────────────────────────────
    if (m_resizing && leftDown && m_selected >= 0 &&
        m_selected < (int)m_widgets.size()) {
        auto& w   = m_widgets[m_selected];
        float dx  = cx - m_resizeMX;
        float dy  = cy - m_resizeMY;
        float ox  = m_resizeOrigX, oy = m_resizeOrigY;
        float ow  = m_resizeOrigW, oh = m_resizeOrigH;

        // Snap the "hot edge" being dragged
        auto snapEdgeX = [&](float rawEdge) {
            std::vector<float> xTargets = {0.0f, m_canvasW, m_canvasW*0.5f};
            for (int i = 0; i < (int)m_widgets.size(); ++i) {
                if (i == m_selected) continue;
                xTargets.push_back(m_widgets[i].x);
                xTargets.push_back(m_widgets[i].x + m_widgets[i].w);
            }
            return snap1D(rawEdge, 0.0f, xTargets, m_snapThresh,
                          false, m_gridSnap, m_snapGrid, m_snapEdges);
        };
        auto snapEdgeY = [&](float rawEdge) {
            std::vector<float> yTargets = {0.0f, m_canvasH, m_canvasH*0.5f};
            for (int i = 0; i < (int)m_widgets.size(); ++i) {
                if (i == m_selected) continue;
                yTargets.push_back(m_widgets[i].y);
                yTargets.push_back(m_widgets[i].y + m_widgets[i].h);
            }
            return snap1D(rawEdge, 0.0f, yTargets, m_snapThresh,
                          true, m_gridSnap, m_snapGrid, m_snapEdges);
        };

        // Clamp minimum widget size
        constexpr float kMinW = 20.0f, kMinH = 10.0f;

        switch (m_resizeHandle) {
        case 0: { // TL
            float nx = snapEdgeX(ox + dx), ny = snapEdgeY(oy + dy);
            w.w = std::max(kMinW, ox+ow - nx); w.x = ox+ow - w.w;
            w.h = std::max(kMinH, oy+oh - ny); w.y = oy+oh - w.h;
        } break;
        case 1: { // TC
            float ny = snapEdgeY(oy + dy);
            w.h = std::max(kMinH, oy+oh - ny); w.y = oy+oh - w.h;
        } break;
        case 2: { // TR
            float nx = snapEdgeX(ox + ow + dx), ny = snapEdgeY(oy + dy);
            w.w = std::max(kMinW, nx - ox);
            w.h = std::max(kMinH, oy+oh - ny); w.y = oy+oh - w.h;
        } break;
        case 3: { // MR
            float nx = snapEdgeX(ox + ow + dx);
            w.w = std::max(kMinW, nx - ox);
        } break;
        case 4: { // BR
            float nx = snapEdgeX(ox + ow + dx), ny = snapEdgeY(oy + oh + dy);
            w.w = std::max(kMinW, nx - ox);
            w.h = std::max(kMinH, ny - oy);
        } break;
        case 5: { // BC
            float ny = snapEdgeY(oy + oh + dy);
            w.h = std::max(kMinH, ny - oy);
        } break;
        case 6: { // BL
            float nx = snapEdgeX(ox + dx), ny = snapEdgeY(oy + oh + dy);
            w.w = std::max(kMinW, ox+ow - nx); w.x = ox+ow - w.w;
            w.h = std::max(kMinH, ny - oy);
        } break;
        case 7: { // ML
            float nx = snapEdgeX(ox + dx);
            w.w = std::max(kMinW, ox+ow - nx); w.x = ox+ow - w.w;
        } break;
        }
    } else if (m_dragging && leftDown && m_selected >= 0 &&
               m_selected < (int)m_widgets.size()) {
        // ── Move (with snap) ──────────────────────────────────────────────────
        auto& w = m_widgets[m_selected];
        float rawX = m_dragWX + (cx - m_dragMX);
        float rawY = m_dragWY + (cy - m_dragMY);
        auto snapped = applySnap(rawX, rawY, w.w, w.h, m_selected);
        w.x = snapped.x; w.y = snapped.y;
    }

    if (!leftDown) {
        m_resizing = false;
        m_dragging = false;
        m_snapLines.clear();
    }

    // ── New mouse-press actions ───────────────────────────────────────────────
    if (leftPressed && overCanvas) {
        if (m_placing) {
            // Place new widget (already ghost-snapped; place at snapped position)
            float gw = defaultW(m_placeType), gh = defaultH(m_placeType);
            float gx = cx - gw*0.5f,         gy = cy - gh*0.5f;
            auto snapped = applySnap(gx, gy, gw, gh, -1);

            DesignWidget w;
            w.type   = m_placeType;
            w.name   = m_placeType + std::to_string(m_widgets.size()+1);
            w.x = snapped.x; w.y = snapped.y; w.w = gw; w.h = gh;
            w.bg     = defaultBg(m_placeType);
            w.border = {0.85f,0.70f,0.38f,0.35f};
            m_widgets.push_back(std::move(w));
            m_selected = (int)m_widgets.size()-1;
            m_placing  = false;
            m_lastFbW  = 0;
        } else {
            // Check resize handles first (priority)
            int handle = hitTestHandle(cx, cy);
            if (handle >= 0 && m_selected >= 0) {
                m_resizing      = true;
                m_resizeHandle  = handle;
                m_resizeMX      = cx; m_resizeMY      = cy;
                const auto& w   = m_widgets[m_selected];
                m_resizeOrigX   = w.x; m_resizeOrigY   = w.y;
                m_resizeOrigW   = w.w; m_resizeOrigH   = w.h;
            } else {
                // Select + start drag
                int hit = hitTestCanvas(cx, cy);
                if (hit != m_selected) m_selected = hit;
                if (hit >= 0) {
                    m_dragging = true;
                    m_dragMX   = cx; m_dragMY   = cy;
                    m_dragWX   = m_widgets[hit].x;
                    m_dragWY   = m_widgets[hit].y;
                } else {
                    m_selected = -1;  // deselect on empty click
                }
            }
        }
    }

    // ── Property panel clicks ─────────────────────────────────────────────────
    bool overProps = (mx >= (float)fbW - kPropsW && my >= kToolbarH);
    if (leftPressed && overProps && m_selected >= 0 &&
        m_selected < (int)m_widgets.size()) {
        auto& w = m_widgets[m_selected];
        for (const auto& pb : m_propBtns) {
            if (!pb.rect.contains(mx, my)) continue;
            switch (pb.action) {
            case PropButton::Action::StepX:       w.x += pb.delta; break;
            case PropButton::Action::StepY:       w.y += pb.delta; break;
            case PropButton::Action::StepW:       w.w = std::max(20.0f, w.w + pb.delta); break;
            case PropButton::Action::StepH:       w.h = std::max(10.0f, w.h + pb.delta); break;
            case PropButton::Action::SetCorner:   w.cornerR = pb.delta; break;
            case PropButton::Action::ToggleShadow:w.shadow = (pb.delta > 0.5f); break;
            case PropButton::Action::SetColor:    w.bg = pb.colorValue; break;
            case PropButton::Action::Delete:      deleteSelected(); break;
            }
            break;
        }
    }

    // ── Delete key ──
    bool delKey = (glfwGetKey(m_window, GLFW_KEY_DELETE) == GLFW_PRESS);
    if (delKey && !m_prevDeleteKey && m_selected >= 0) deleteSelected();
    m_prevDeleteKey = delKey;

    // ── Escape ──
    if (glfwGetKey(m_window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        if (m_placing) { m_placing = false; m_lastFbW = 0; }
        else glfwSetWindowShouldClose(m_window, GLFW_TRUE);
    }
}

void UiEditorApp::onRender(float /*dt*/) {
    int fbW, fbH;
    framebufferSize(fbW, fbH);
    if (fbW <= 0 || fbH <= 0) return;

    beginFrameDraw();

    m_uiDrawList.addRectFilled(UiRect::fromXYWH(0, 0, (float)fbW, (float)fbH), kEditorBg);
    m_uiDrawList.addRectFilled(UiRect::fromXYWH(0, kToolbarH, kPaletteW, fbH-kToolbarH), kPanelBg);

    drawCanvas(fbW, fbH);

    m_propBtns.clear();
    drawPropertiesPanel(fbW, fbH);

    // Panel dividers
    float fW = (float)fbW, fH = (float)fbH;
    m_uiDrawList.addRectFilled({kPaletteW-1, kToolbarH, kPaletteW, fH}, {0.16f,0.17f,0.22f,1.0f});
    m_uiDrawList.addRectFilled({fW-kPropsW, kToolbarH, fW-kPropsW+1.0f, fH}, {0.16f,0.17f,0.22f,1.0f});
    m_uiDrawList.addRectFilled({0.0f, kToolbarH-1.0f, fW, kToolbarH}, {0.16f,0.17f,0.22f,1.0f});

    // Widget count + snap status in status strip
    char info[64];
    std::snprintf(info, sizeof(info), "%d widget%s  |  G: grid  S: edges",
                  (int)m_widgets.size(), m_widgets.size()==1?"":"s");
    float iw = m_uiFont.measureText(info);
    m_uiDrawList.addText(m_uiFont, info,
        {fW - kPropsW - iw - 10.0f, 14.0f}, kTextDim);

    submitFrame(m_camera);
}

// ─── Helpers ─────────────────────────────────────────────────────────────────
int UiEditorApp::hitTestCanvas(float cx, float cy) const {
    for (int i = (int)m_widgets.size()-1; i >= 0; --i) {
        const auto& w = m_widgets[i];
        if (cx >= w.x && cx <= w.x+w.w && cy >= w.y && cy <= w.y+w.h)
            return i;
    }
    return -1;
}

void UiEditorApp::deleteSelected() {
    if (m_selected < 0 || m_selected >= (int)m_widgets.size()) return;
    m_widgets.erase(m_widgets.begin() + m_selected);
    m_selected = -1;
}

// ─── JSON round-trip ─────────────────────────────────────────────────────────

static std::string colorHex(const ui::UiColor& c) {
    char buf[10];
    std::snprintf(buf, sizeof(buf), "#%02X%02X%02X%02X",
        (int)(c.r * 255.0f + 0.5f), (int)(c.g * 255.0f + 0.5f),
        (int)(c.b * 255.0f + 0.5f), (int)(c.a * 255.0f + 0.5f));
    return buf;
}

void UiEditorApp::saveJson(const std::string& path) {
    using json = nlohmann::json;
    json root;
    root["version"] = 1;
    root["type"]    = "Panel";
    root["id"]      = "root";
    root["x"]       = 0; root["y"] = 0;
    root["width"]   = 1280; root["height"] = 720;

    json children = json::array();
    for (const DesignWidget& w : m_widgets) {
        json node;
        node["type"]         = w.type;
        node["id"]           = w.name;
        node["x"]            = (int)w.x;
        node["y"]            = (int)w.y;
        node["width"]        = (int)w.w;
        node["height"]       = (int)w.h;
        node["background"]   = colorHex(w.bg);
        node["borderColor"]  = colorHex(w.border);
        node["borderWidth"]  = w.borderW;
        node["cornerRadius"] = w.cornerR;
        node["shadow"]       = w.shadow;
        if (!w.label.empty()) {
            node[w.type == "Label" ? "text" : "label"] = w.label;
        }
        children.push_back(std::move(node));
    }
    root["children"] = std::move(children);

    std::ofstream f(path);
    if (f) f << root.dump(2);
    m_loadedPath = path;
}

void UiEditorApp::loadJson(const std::string& path) {
    using json = nlohmann::json;
    std::ifstream f(path);
    if (!f) return;
    json doc;
    try { f >> doc; } catch (...) { return; }

    if (!doc.contains("children")) return;

    auto parseColor = [](const std::string& hex, const ui::UiColor& def) -> ui::UiColor {
        const char* s = hex.c_str();
        if (*s == '#') ++s;
        const std::size_t len = hex.size() - (*hex.c_str() == '#' ? 1 : 0);
        if (len != 6 && len != 8) return def;
        auto hb = [](const char* p) -> float {
            auto h = [](char c) -> int {
                return c>='a'?(c-'a'+10):c>='A'?(c-'A'+10):(c-'0');
            };
            return static_cast<float>((h(p[0])<<4)|h(p[1])) / 255.0f;
        };
        return {hb(s), hb(s+2), hb(s+4), len==8 ? hb(s+6) : 1.0f};
    };

    m_widgets.clear();
    m_selected = -1;
    for (const json& n : doc["children"]) {
        DesignWidget w;
        w.type    = n.value("type", "Panel");
        w.name    = n.value("id", "widget");
        w.x       = (float)n.value("x", 0);
        w.y       = (float)n.value("y", 0);
        w.w       = (float)n.value("width",  200);
        w.h       = (float)n.value("height",  80);
        w.borderW = n.value("borderWidth",  1.0f);
        w.cornerR = n.value("cornerRadius", 0.0f);
        w.shadow  = n.value("shadow", false);
        if (n.contains("background"))  w.bg     = parseColor(n["background"].get<std::string>(),  w.bg);
        if (n.contains("borderColor")) w.border = parseColor(n["borderColor"].get<std::string>(), w.border);
        if (n.contains("label"))       w.label  = n["label"].get<std::string>();
        else if (n.contains("text"))   w.label  = n["text"].get<std::string>();
        m_widgets.push_back(std::move(w));
    }
    m_loadedPath = path;
    m_lastFbW = 0;
}

} // namespace odai::tools::ui_editor
