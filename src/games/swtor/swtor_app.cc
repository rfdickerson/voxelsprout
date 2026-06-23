#include "games/swtor/swtor_app.h"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <stb_image.h>

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace odai::games::swtor {

using namespace ui;

static constexpr float kPi = 3.14159265f;

// ─── Colour palette ──────────────────────────────────────────────────────────
static constexpr UiColor kPanel  {0.018f, 0.105f, 0.145f, 0.91f};
static constexpr UiColor kBorder {0.100f, 0.820f, 0.970f, 0.58f};
static constexpr UiColor kText   {0.785f, 0.920f, 0.955f, 1.00f};
static constexpr UiColor kDim    {0.360f, 0.650f, 0.720f, 1.00f};
static constexpr UiColor kGold   {0.980f, 0.760f, 0.270f, 1.00f};
static constexpr UiColor kGreen  {0.180f, 0.900f, 0.520f, 1.00f};
static constexpr UiColor kBlue   {0.070f, 0.660f, 0.940f, 1.00f};
static constexpr UiColor kRed    {0.847f, 0.173f, 0.110f, 1.00f};
static constexpr UiColor kXpGold {0.847f, 0.627f, 0.125f, 1.00f};
static constexpr UiColor kSlotBg {0.018f, 0.090f, 0.130f, 0.94f};
static constexpr UiColor kShadow {0.000f, 0.000f, 0.000f, 0.60f};

// Quality border colors: grey / white / green / blue / purple / gold
static constexpr UiColor kQualityColor[] = {
    {0.50f,0.50f,0.52f,1.0f},
    {0.80f,0.82f,0.85f,1.0f},
    {0.22f,0.72f,0.28f,1.0f},
    {0.22f,0.48f,0.90f,1.0f},
    {0.62f,0.26f,0.90f,1.0f},
    {0.90f,0.65f,0.12f,1.0f},
};

static constexpr UiColor kAbilityColor[] = {
    {0.75f,0.20f,0.18f,1.0f},{0.80f,0.45f,0.10f,1.0f},{0.75f,0.70f,0.10f,1.0f},
    {0.15f,0.65f,0.85f,1.0f},{0.55f,0.20f,0.85f,1.0f},{0.20f,0.70f,0.35f,1.0f},
    {0.20f,0.40f,0.85f,1.0f},{0.85f,0.85f,0.85f,1.0f},{0.85f,0.30f,0.55f,1.0f},
    {0.20f,0.65f,0.65f,1.0f},{0.70f,0.30f,0.18f,1.0f},{0.50f,0.50f,0.50f,1.0f},
};
static const char* kAbilityKey[] = {"1","2","3","4","5","6","7","8","9","0","-","="};

static const struct { const char* ch; const char* from; const char* msg; } kChatMsgs[] = {
    {"General",  "Darth_Marauder",  "Anyone doing 16m HM Ops tonight?"},
    {"Guild",    "SithSorcerer99",  "Healing if you need, got stims"},
    {"General",  "Smuggler_Han",    "WTB stims x100 - 5k each PST"},
    {"Ops",      "OperationLead",   "Boss 4 in 10 min - healers left"},
    {"General",  "JediConsular",    "LFG HM FP - Tank/Heal needed"},
    {"Guild",    "TrooperVoss",     "Just hit 75! Story done"},
    {"General",  "AgentKaliyo",     "Dark vs Light event is live"},
    {"Ops",      "OperationLead",   "Watch red circles - spread!"},
};
static constexpr int kNumChatMsgs = 8;

// ─── Shared helpers ───────────────────────────────────────────────────────────
void SwtorApp::drawPanelBg(float x, float y, float w, float h, float s, float radius) {
    UiRect r = UiRect::fromXYWH(x, y, w, h);
    m_uiDrawList.addDropShadow(r, kShadow, 10.0f*s, 0, 5);
    m_uiDrawList.addRoundRectGlow(r, {kBorder.r, kBorder.g, kBorder.b, 0.13f}, radius*s, 10.0f*s);
    m_uiDrawList.addRoundRectFilled(r, kPanel, radius*s);
    m_uiDrawList.addRectFilledVGradient(r, {0.10f,0.75f,0.92f,0.14f}, {0.0f,0.01f,0.03f,0.20f});
    m_uiDrawList.addRoundRect(r, kBorder, radius*s, s);
    m_uiDrawList.addRectFilled(UiRect::fromXYWH(x + 8*s, y + 2*s, w - 16*s, s), {0.42f,0.94f,1.0f,0.32f});
}

void SwtorApp::drawBar(float x, float y, float w, float h, float frac,
                       const UiColor& fill, float s, float radius) {
    UiRect bg = UiRect::fromXYWH(x, y, w, h);
    m_uiDrawList.addRoundRectFilled(bg, {0.04f,0.06f,0.10f,1.0f}, radius*s);
    if (frac > 0.001f) {
        float fw = std::max(radius * s * 2.0f, w * std::min(frac, 1.0f));
        UiColor dark{fill.r*0.55f, fill.g*0.55f, fill.b*0.55f, fill.a};
        m_uiDrawList.addRoundRectFilledHGradient(
            UiRect::fromXYWH(x, y, fw, h), fill, dark, radius*s);
        float shimPhase = std::fmod(m_combatTimer * 0.45f, 1.0f);
        float shimW = fw * 0.30f;
        float shimX = x - shimW + (fw + shimW) * shimPhase;
        m_uiDrawList.pushClip(UiRect::fromXYWH(x, y, fw, h));
        m_uiDrawList.addRectFilledHGradient(
            UiRect::fromXYWH(shimX,              y + s, shimW * 0.5f, h - 2.0f*s),
            {1.0f,1.0f,1.0f,0.0f}, {1.0f,1.0f,1.0f,0.18f});
        m_uiDrawList.addRectFilledHGradient(
            UiRect::fromXYWH(shimX + shimW*0.5f, y + s, shimW * 0.5f, h - 2.0f*s),
            {1.0f,1.0f,1.0f,0.18f}, {1.0f,1.0f,1.0f,0.0f});
        m_uiDrawList.popClip();
    }
    m_uiDrawList.addRoundRect(bg, {0,0,0,0.35f}, radius*s, s);
}

void SwtorApp::drawCooldown(float cx, float cy, float radius, float frac) {
    float end = -kPi * 0.5f + frac * 2.0f * kPi;
    m_uiDrawList.addSectorFilled({cx, cy}, 0.0f, radius, -kPi * 0.5f, end,
                                  {0.0f,0.0f,0.0f,0.72f});
}

// ─── Gear icon ───────────────────────────────────────────────────────────────
// Draws a vector icon inside a square slot of side `sz`, centered at (cx, cy).
// type: 0=blaster 1=lightsaber 2=helmet 3=chest 4=legs 5=belt 6=boots 7=accessory
void SwtorApp::drawGearIcon(float cx, float cy, float sz, int type, float s) {
    const float r = sz * 0.5f; // half-slot radius for relative placement
    switch (type % 8) {

    case 0: { // Blaster pistol
        // Barrel (horizontal)
        m_uiDrawList.addRoundRectFilled(
            UiRect::fromXYWH(cx - r*0.60f, cy - r*0.13f, r*1.10f, r*0.26f),
            {0.68f,0.70f,0.78f,0.95f}, 2.0f*s);
        // Energy cell on barrel top
        m_uiDrawList.addRectFilled(
            UiRect::fromXYWH(cx - r*0.25f, cy - r*0.34f, r*0.28f, r*0.20f),
            {0.28f,0.72f,1.00f,0.90f});
        // Grip (vertical, below trigger guard)
        m_uiDrawList.addRoundRectFilled(
            UiRect::fromXYWH(cx + r*0.08f, cy + r*0.10f, r*0.22f, r*0.50f),
            {0.50f,0.52f,0.60f,0.95f}, 2.0f*s);
        // Muzzle accent
        m_uiDrawList.addCircleFilled({cx - r*0.55f, cy}, r*0.07f, {0.28f,0.72f,1.0f,0.80f});
        break;
    }

    case 1: { // Lightsaber
        // Hilt
        m_uiDrawList.addRoundRectFilled(
            UiRect::fromXYWH(cx - r*0.13f, cy + r*0.08f, r*0.26f, r*0.58f),
            {0.55f,0.55f,0.62f,0.95f}, 3.0f*s);
        // Crossguard
        m_uiDrawList.addRoundRectFilled(
            UiRect::fromXYWH(cx - r*0.32f, cy + r*0.02f, r*0.64f, r*0.10f),
            {0.60f,0.60f,0.68f,0.92f}, 2.0f*s);
        // Blade
        m_uiDrawList.addRoundRectFilled(
            UiRect::fromXYWH(cx - r*0.055f, cy - r*0.86f, r*0.11f, r*0.92f),
            {0.42f,0.82f,1.00f,0.92f}, 2.0f*s);
        // Blade inner glow
        m_uiDrawList.addRoundRectFilled(
            UiRect::fromXYWH(cx - r*0.025f, cy - r*0.84f, r*0.05f, r*0.88f),
            {0.80f,0.96f,1.00f,0.95f}, 1.5f*s);
        // Tip flare
        m_uiDrawList.addCircleFilled({cx, cy - r*0.83f}, r*0.075f, {0.85f,0.98f,1.00f,0.98f});
        break;
    }

    case 2: { // Helmet
        // Dome: π→2π sweeps upper half in screen-space clockwise convention
        m_uiDrawList.addSectorFilled(
            {cx, cy + r*0.14f}, 0.0f, r*0.72f, kPi, kPi*2.0f,
            {0.32f,0.35f,0.46f,0.95f});
        // Chin guard
        m_uiDrawList.addRoundRectFilled(
            UiRect::fromXYWH(cx - r*0.44f, cy + r*0.10f, r*0.88f, r*0.30f),
            {0.26f,0.28f,0.38f,0.95f}, 2.0f*s);
        // Visor slit
        m_uiDrawList.addRoundRectFilled(
            UiRect::fromXYWH(cx - r*0.38f, cy - r*0.04f, r*0.76f, r*0.16f),
            {0.18f,0.44f,0.80f,0.90f}, 2.0f*s);
        // Visor highlight
        m_uiDrawList.addRectFilledHGradient(
            UiRect::fromXYWH(cx - r*0.36f, cy - r*0.03f, r*0.36f, r*0.07f),
            {0.70f,0.90f,1.00f,0.40f}, {0.18f,0.44f,0.80f,0.00f});
        break;
    }

    case 3: { // Chest armor
        // Main plate
        m_uiDrawList.addRoundRectFilled(
            UiRect::fromXYWH(cx - r*0.52f, cy - r*0.48f, r*1.04f, r*0.92f),
            {0.24f,0.26f,0.36f,0.95f}, 3.0f*s);
        // Shoulder guards
        m_uiDrawList.addRoundRectFilled(
            UiRect::fromXYWH(cx - r*0.70f, cy - r*0.36f, r*0.18f, r*0.34f),
            {0.30f,0.33f,0.44f,0.92f}, 2.0f*s);
        m_uiDrawList.addRoundRectFilled(
            UiRect::fromXYWH(cx + r*0.52f, cy - r*0.36f, r*0.18f, r*0.34f),
            {0.30f,0.33f,0.44f,0.92f}, 2.0f*s);
        // Power core
        m_uiDrawList.addCircleFilled({cx, cy - r*0.06f}, r*0.22f, {0.28f,0.76f,1.00f,0.88f});
        m_uiDrawList.addCircleFilled({cx, cy - r*0.06f}, r*0.12f, {0.65f,0.92f,1.00f,0.95f});
        m_uiDrawList.addCircle({cx, cy - r*0.06f}, r*0.28f, {0.28f,0.76f,1.00f,0.35f}, 1.5f*s);
        break;
    }

    case 4: { // Legs
        // Left leg
        m_uiDrawList.addRoundRectFilled(
            UiRect::fromXYWH(cx - r*0.45f, cy - r*0.50f, r*0.38f, r*0.96f),
            {0.26f,0.28f,0.38f,0.95f}, 2.0f*s);
        // Right leg
        m_uiDrawList.addRoundRectFilled(
            UiRect::fromXYWH(cx + r*0.07f, cy - r*0.50f, r*0.38f, r*0.96f),
            {0.26f,0.28f,0.38f,0.95f}, 2.0f*s);
        // Knee pads
        m_uiDrawList.addRoundRectFilled(
            UiRect::fromXYWH(cx - r*0.46f, cy + r*0.04f, r*0.40f, r*0.22f),
            {0.34f,0.36f,0.50f,0.92f}, 2.0f*s);
        m_uiDrawList.addRoundRectFilled(
            UiRect::fromXYWH(cx + r*0.06f, cy + r*0.04f, r*0.40f, r*0.22f),
            {0.34f,0.36f,0.50f,0.92f}, 2.0f*s);
        break;
    }

    case 5: { // Belt
        // Belt strap
        m_uiDrawList.addRoundRectFilled(
            UiRect::fromXYWH(cx - r*0.70f, cy - r*0.12f, r*1.40f, r*0.24f),
            {0.32f,0.26f,0.18f,0.95f}, 2.0f*s);
        // Buckle plate
        m_uiDrawList.addRoundRectFilled(
            UiRect::fromXYWH(cx - r*0.24f, cy - r*0.26f, r*0.48f, r*0.52f),
            {0.68f,0.58f,0.20f,0.95f}, 3.0f*s);
        // Buckle inner
        m_uiDrawList.addRoundRect(
            UiRect::fromXYWH(cx - r*0.14f, cy - r*0.16f, r*0.28f, r*0.32f),
            {0.88f,0.76f,0.28f,0.90f}, 2.0f*s, 1.5f*s);
        break;
    }

    case 6: { // Boots
        // Boot shaft
        m_uiDrawList.addRoundRectFilled(
            UiRect::fromXYWH(cx - r*0.26f, cy - r*0.60f, r*0.52f, r*0.80f),
            {0.24f,0.26f,0.36f,0.95f}, 3.0f*s);
        // Sole (wider)
        m_uiDrawList.addRoundRectFilled(
            UiRect::fromXYWH(cx - r*0.34f, cy + r*0.16f, r*0.68f, r*0.26f),
            {0.18f,0.20f,0.28f,0.95f}, 2.0f*s);
        // Toe cap
        m_uiDrawList.addRoundRectFilled(
            UiRect::fromXYWH(cx + r*0.10f, cy + r*0.04f, r*0.24f, r*0.14f),
            {0.30f,0.32f,0.44f,0.90f}, 2.0f*s);
        break;
    }

    case 7: { // Accessory / implant
        // Outer ring
        m_uiDrawList.addCircleFilled({cx, cy}, r*0.60f, {0.24f,0.26f,0.38f,0.95f});
        // Circuit lines (cross)
        m_uiDrawList.addRectFilled(
            UiRect::fromXYWH(cx - r*0.56f, cy - r*0.06f, r*1.12f, r*0.12f),
            {0.22f,0.58f,0.92f,0.65f});
        m_uiDrawList.addRectFilled(
            UiRect::fromXYWH(cx - r*0.06f, cy - r*0.56f, r*0.12f, r*1.12f),
            {0.22f,0.58f,0.92f,0.65f});
        // Inner gem
        m_uiDrawList.addCircleFilled({cx, cy}, r*0.22f, {0.28f,0.72f,1.00f,0.95f});
        m_uiDrawList.addCircleFilled({cx, cy}, r*0.10f, {0.70f,0.92f,1.00f,1.00f});
        // Ring accent
        m_uiDrawList.addCircle({cx, cy}, r*0.60f, {0.22f,0.58f,0.92f,0.38f}, 1.5f*s);
        break;
    }

    } // switch
}

// ─── Gear / inventory slot ────────────────────────────────────────────────────
void SwtorApp::drawGearSlot(float x, float y, float sz, int iconType, int quality, int variant, float s) {
    UiRect slot = UiRect::fromXYWH(x, y, sz, sz);

    // Outer frame
    const float variantTint = 0.035f * static_cast<float>(variant % 4);
    m_uiDrawList.addRoundRectFilled(slot, {0.025f + variantTint,0.105f + variantTint,0.145f + variantTint,0.96f}, 3.0f*s);
    // Inner well (darker recess)
    m_uiDrawList.addRoundRectFilled(
        UiRect::fromXYWH(x+2*s, y+2*s, sz-4*s, sz-4*s),
        {0.015f,0.045f + variantTint,0.080f + variantTint,0.97f}, 2.0f*s);
    // Top-light sheen
    m_uiDrawList.addRectFilledVGradient(
        UiRect::fromXYWH(x+2*s, y+2*s, sz-4*s, (sz-4*s)*0.4f),
        {1.0f,1.0f,1.0f,0.05f}, {1.0f,1.0f,1.0f,0.0f});

    // Clip the icon to the inner well so it doesn't bleed over the border.
    m_uiDrawList.pushClip(UiRect::fromXYWH(x+2*s, y+2*s, sz-4*s, sz-4*s));
    if (m_iconSheet != kUiNoTexture) {
        static constexpr int kIconBase[] = {0, 5, 8, 16, 24, 28, 24, 32};
        const int iconIndex = kIconBase[iconType % 8] + (variant % 4);
        const float cell = 1.0f / 8.0f;
        const float u = static_cast<float>(iconIndex % 8) * cell;
        const float v = static_cast<float>(iconIndex / 8) * cell;
        m_uiDrawList.addImage(UiRect::fromXYWH(x + 3*s, y + 3*s, sz - 6*s, sz - 6*s),
            m_iconSheet, {1.0f, 1.0f, 1.0f, 1.0f}, {u, v, u + cell, v + cell});
    } else {
        drawGearIcon(x + sz*0.5f, y + sz*0.5f, sz - 6*s, iconType + variant, s);
    }
    m_uiDrawList.popClip();

    // Quality border
    const UiColor& qc = kQualityColor[std::min(quality, 5)];
    m_uiDrawList.addRoundRect(slot, {qc.r, qc.g, qc.b, 0.82f}, 3.0f*s, s);

    // Glow on purple+ quality
    if (quality >= 4) {
        m_uiDrawList.addRoundRectGlow(slot, {qc.r, qc.g, qc.b, 0.30f}, 3.0f*s, 6.0f*s);
    }
}

// ─── Character window ─────────────────────────────────────────────────────────
void SwtorApp::drawCharWindow(float x, float y, float w, float h, float s) {
    // Backing shadow + frame
    UiRect win = UiRect::fromXYWH(x, y, w, h);
    m_uiDrawList.addDropShadow(win, {0,0,0,0.75f}, 22.0f*s, 0, 6);
    m_uiDrawList.addRoundRectFilled(win, {0.04f,0.07f,0.12f,0.97f}, 4.0f*s);
    // Left teal accent stripe
    m_uiDrawList.addRoundRectFilled(
        UiRect::fromXYWH(x, y + 10*s, 3.0f*s, h - 20*s),
        {0.15f,0.72f,0.95f,0.60f}, 1.5f*s);
    // Bottom edge accent
    m_uiDrawList.addRectFilledHGradient(
        UiRect::fromXYWH(x, y + h - 2*s, w, 2*s),
        {0.15f,0.72f,0.95f,0.50f}, {kBorder.r,kBorder.g,kBorder.b,0.30f});
    // Outer border
    m_uiDrawList.addRoundRect(win, kBorder, 4.0f*s, s);

    // ── Title row
    const float titleH = 42.0f * s;
    m_uiDrawList.addRoundRectFilledHGradient(
        UiRect::fromXYWH(x, y, w, titleH),
        {0.03f,0.28f,0.38f,0.98f}, {0.01f,0.08f,0.13f,0.98f}, 4.0f*s);
    const char* charName = "SCYA";
    float nameW = m_uiFontBold.measureText(charName);
    m_uiDrawList.addText(m_uiFontBold, charName, {x + (w - nameW)*0.5f, y + 13.0f*s}, kText);
    m_uiDrawList.addCircleFilled({x + 17*s, y + titleH*0.5f}, 5*s, kBlue);
    m_uiDrawList.addCircle({x + 17*s, y + titleH*0.5f}, 8*s, {kBorder.r,kBorder.g,kBorder.b,0.48f}, s);
    // Close X
    m_uiDrawList.addText(m_uiFont, "[C]", {x + w - 36*s, y + 11.0f*s}, kDim);

    // ── Tabs row
    static const char* kTabs[] = {"GEAR","COMBAT STYLE","OUTFITTER","COMPANION","LOADOUTS"};
    const float tabY = y + titleH + 2*s;
    const float tabH = 28.0f * s;
    float tabX = x + 6.0f*s;
    for (int t = 0; t < 5; ++t) {
        bool active = (t == 0);
        float tw = m_uiFont.measureText(kTabs[t]) + 20.0f*s;
        UiRect tabR = UiRect::fromXYWH(tabX, tabY, tw, tabH);
        if (active) {
            m_uiDrawList.addRoundRectFilled(tabR, {0.04f,0.34f,0.46f,0.95f}, 2*s);
            // Underline bar
            m_uiDrawList.addRectFilled(
                UiRect::fromXYWH(tabX, tabY + tabH - 2*s, tw, 2*s), {0.45f,0.96f,1.0f,0.96f});
        } else {
            m_uiDrawList.addRoundRectFilled(tabR, {0.02f,0.11f,0.16f,0.76f}, 2*s);
        }
        m_uiDrawList.addText(m_uiFont, kTabs[t],
            {tabX + 10*s, tabY + (tabH - m_uiFont.lineHeightPx())*0.5f},
            active ? kText : kDim);
        tabX += tw + 2.0f*s;
    }

    // ── Content area
    const float contY = tabY + tabH + 4.0f*s;
    const float contH = (y + h - 4.0f*s) - contY;
    const float leftW  = w * 0.350f;   // character + stats
    const float midW   = w * 0.225f;   // gear slots
    const float rightW = w - leftW - midW; // inventory
    const float midX   = x + leftW;
    const float rightX = midX + midW;

    // Vertical dividers
    m_uiDrawList.addRectFilled(UiRect::fromXYWH(midX,   contY, s, contH), {0.2f,0.3f,0.5f,0.22f});
    m_uiDrawList.addRectFilled(UiRect::fromXYWH(rightX, contY, s, contH), {0.2f,0.3f,0.5f,0.22f});

    // ─────────────────────────────────────────────────────────────────────────
    // LEFT PANEL: Character silhouette + stat readout
    // ─────────────────────────────────────────────────────────────────────────
    {
        const float padX = x + 10*s;
        const float charAreaH = contH - 20*s;

        // Character viewport box
        UiRect cvp = UiRect::fromXYWH(padX, contY + 8*s, (leftW - 28*s) * 0.57f, charAreaH - 8*s);
        m_uiDrawList.addRoundRectFilled(cvp, {0.06f,0.09f,0.15f,0.95f}, 3*s);
        m_uiDrawList.addRoundRect(cvp, {0.18f,0.28f,0.48f,0.28f}, 3*s, s);

        // Floor gradient
        float cvW = cvp.maxX - cvp.minX, cvH = cvp.maxY - cvp.minY;
        float cCX = cvp.minX + cvW*0.5f;
        m_uiDrawList.addRectFilledVGradient(
            UiRect::fromXYWH(cvp.minX, cvp.minY + cvH*0.55f, cvW, cvH*0.45f),
            {0.08f,0.12f,0.20f,0.0f}, {0.05f,0.08f,0.14f,0.85f});

        // ── Human silhouette (proportional to cvH) ──
        const float hh = cvH;  // full height of character viewport
        const float top = cvp.minY;
        const UiColor cSuit{0.12f,0.13f,0.17f,0.94f};
        const UiColor cSkin{0.52f,0.40f,0.30f,0.96f};

        // Head
        m_uiDrawList.addCircleFilled({cCX, top + hh*0.115f}, hh*0.088f, cSkin);
        // Neck
        m_uiDrawList.addRoundRectFilled(
            UiRect::fromXYWH(cCX - hh*0.028f, top + hh*0.195f, hh*0.056f, hh*0.04f),
            cSkin, 2*s);
        // Torso
        m_uiDrawList.addRoundRectFilled(
            UiRect::fromXYWH(cCX - hh*0.115f, top + hh*0.235f, hh*0.23f, hh*0.295f),
            cSuit, 4*s);
        // Belt accent
        m_uiDrawList.addRoundRectFilled(
            UiRect::fromXYWH(cCX - hh*0.115f, top + hh*0.505f, hh*0.23f, hh*0.030f),
            {0.65f,0.55f,0.18f,0.80f}, 2*s);
        // Left arm
        m_uiDrawList.addRoundRectFilled(
            UiRect::fromXYWH(cCX - hh*0.185f, top + hh*0.240f, hh*0.068f, hh*0.260f),
            cSuit, 3*s);
        // Right arm
        m_uiDrawList.addRoundRectFilled(
            UiRect::fromXYWH(cCX + hh*0.117f, top + hh*0.240f, hh*0.068f, hh*0.260f),
            cSuit, 3*s);
        // Left hand
        m_uiDrawList.addCircleFilled({cCX - hh*0.155f, top + hh*0.52f}, hh*0.040f, cSkin);
        // Right hand
        m_uiDrawList.addCircleFilled({cCX + hh*0.152f, top + hh*0.52f}, hh*0.040f, cSkin);
        // Left leg
        m_uiDrawList.addRoundRectFilled(
            UiRect::fromXYWH(cCX - hh*0.113f, top + hh*0.535f, hh*0.096f, hh*0.310f),
            cSuit, 3*s);
        // Right leg
        m_uiDrawList.addRoundRectFilled(
            UiRect::fromXYWH(cCX + hh*0.017f, top + hh*0.535f, hh*0.096f, hh*0.310f),
            cSuit, 3*s);
        // Boots
        m_uiDrawList.addRoundRectFilled(
            UiRect::fromXYWH(cCX - hh*0.122f, top + hh*0.835f, hh*0.115f, hh*0.060f),
            {0.10f,0.10f,0.13f,0.95f}, 2*s);
        m_uiDrawList.addRoundRectFilled(
            UiRect::fromXYWH(cCX + hh*0.007f, top + hh*0.835f, hh*0.115f, hh*0.060f),
            {0.10f,0.10f,0.13f,0.95f}, 2*s);
        // Ground shadow
        m_uiDrawList.addRectFilledVGradient(
            UiRect::fromXYWH(cCX - hh*0.18f, top + hh*0.890f, hh*0.36f, hh*0.08f),
            {0.0f,0.0f,0.0f,0.22f}, {0.0f,0.0f,0.0f,0.0f});

        // "WIP" badge (top-left of viewport)
        UiRect wipR = UiRect::fromXYWH(cvp.minX + 5*s, cvp.minY + 5*s, 40*s, 20*s);
        m_uiDrawList.addRoundRectFilled(wipR, {0.84f,0.84f,0.84f,0.96f}, 2*s);
        m_uiDrawList.addText(m_uiFont, "WIP",
            {wipR.minX + 7*s, wipR.minY + (20*s - m_uiFont.lineHeightPx())*0.5f},
            {0.08f,0.08f,0.10f,1.0f});

        // Level ring (bottom-left of viewport)
        const float lvlCX = cvp.minX + 36*s;
        const float lvlCY = cvp.maxY - 38*s;
        const float lvlR  = 26*s;
        m_uiDrawList.addCircleFilled({lvlCX, lvlCY}, lvlR + 2*s, {0.20f,0.35f,0.65f,0.50f});
        m_uiDrawList.addCircleFilled({lvlCX, lvlCY}, lvlR, {0.06f,0.09f,0.14f,0.95f});
        // Colored arc segments
        m_uiDrawList.addSectorFilled({lvlCX, lvlCY}, lvlR-4*s, lvlR,
            kPi*1.0f, kPi*1.88f, {0.15f,0.42f,0.90f,0.90f});
        m_uiDrawList.addSectorFilled({lvlCX, lvlCY}, lvlR-4*s, lvlR,
            kPi*1.92f, kPi*2.00f, {0.88f,0.18f,0.12f,0.85f});
        // Level number
        const char* lvlStr = "5";
        float lvlTW = m_uiFontBold.measureText(lvlStr);
        m_uiDrawList.addText(m_uiFontBold, lvlStr,
            {lvlCX - lvlTW*0.5f, lvlCY - m_uiFont.lineHeightPx()*0.5f}, kText);
        // Small star above level ring
        m_uiDrawList.addCircleFilled({lvlCX, lvlCY - lvlR - 6*s}, 3.5f*s, {0.26f,0.56f,1.00f,0.90f});

        // ── Stats
        const float statX  = cvp.maxX + 10*s;
        const float statX2 = midX - 12*s;  // right-align values here
        float statY = cvp.minY + 28*s;
        const float lh = m_uiFont.lineHeightPx();
        struct StatRow { const char* label; const char* value; };
        static const StatRow kStats[] = {
            {"DAMAGE",        "13,741"},
            {"SURVIVABILITY", "42,516"},
            {"SUPPORT",       "10,498"},
        };
        for (const auto& st : kStats) {
            m_uiDrawList.addText(m_uiFont, st.label, {statX, statY}, kDim);
            float vw = m_uiFontBold.measureText(st.value);
            m_uiDrawList.addText(m_uiFontBold, st.value, {statX2 - vw, statY}, kGold);
            statY += lh * 1.9f;
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // MIDDLE PANEL: Gear slots
    // ─────────────────────────────────────────────────────────────────────────
    {
        const float gx  = midX + 10*s;
        float gy        = contY + 14*s;
        const float slotSz  = 54.0f * s;
        const float slotGap = 5.0f * s;
        const float lh  = m_uiFont.lineHeightPx();

        auto section = [&](const char* label) {
            m_uiDrawList.addText(m_uiFont, label, {gx, gy}, kDim);
            gy += lh + 4*s;
        };
        auto row2 = [&](int ta, int qa, int tb, int qb) {
            drawGearSlot(gx,              gy, slotSz, ta, qa, (ta + qa) % 4, s);
            drawGearSlot(gx + slotSz + slotGap, gy, slotSz, tb, qb, (tb + qb + 1) % 4, s);
            gy += slotSz + slotGap + 4*s;
        };

        section("WEAPONS");
        row2(0, 3, 0, 3);  // two blasters, blue quality

        section("ARMOR");
        row2(2, 4, 3, 4);  // helm(purple) + chest(purple)
        row2(4, 3, 5, 3);  // legs(blue)   + belt(blue)
        row2(6, 3, 6, 2);  // boots(blue)  + gloves(green)

        section("ACCESSORIES");
        row2(7, 4, 7, 3);  // implant(purple) + earpiece(blue)

        gy += 6*s;
        m_uiDrawList.addRectFilled(UiRect::fromXYWH(gx, gy, midW - 20*s, s), {0.2f,0.3f,0.5f,0.20f});
        gy += 8*s;

        // Item rank
        m_uiDrawList.addText(m_uiFont, "ITEM RANK", {gx, gy}, kDim);
        gy += lh + 2*s;
        m_uiDrawList.addText(m_uiFontBold, "318", {gx, gy}, kGold);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // RIGHT PANEL: Inventory grid
    // ─────────────────────────────────────────────────────────────────────────
    {
        const float rx = rightX + 6*s;
        const float rw = rightW - 10*s;
        float ry = contY + 8*s;
        const float lh = m_uiFont.lineHeightPx();

        // ── Inventory tabs
        static const char* kInvTabs[] = {"INVENTORY","MISSION ITEMS","CURRENCY"};
        static const UiColor kTabAccent[] = {
            {0.18f,0.88f,1.00f,1.0f},
            {0.34f,0.72f,0.90f,1.0f},
            {0.34f,0.88f,0.60f,1.0f},
        };
        float itx = rx;
        for (int t = 0; t < 3; ++t) {
            float tw = m_uiFont.measureText(kInvTabs[t]) + 18*s;
            bool active = (t == 0);
            UiRect tabR = UiRect::fromXYWH(itx, ry, tw, 26*s);
            if (active)
                m_uiDrawList.addRoundRectFilled(tabR, {0.03f,0.30f,0.40f,0.95f}, 2*s);
            else
                m_uiDrawList.addRoundRectFilled(tabR, {0.02f,0.11f,0.16f,0.76f}, 2*s);
            m_uiDrawList.addText(m_uiFont, kInvTabs[t],
                {itx + 9*s, ry + (26*s - lh)*0.5f},
                active ? kTabAccent[t] : kDim);
            if (active)
                m_uiDrawList.addRoundRect(tabR, {kTabAccent[t].r,kTabAccent[t].g,kTabAccent[t].b,0.45f}, 2*s, s);
            itx += tw + 2*s;
        }
        ry += 30*s;

        // ── Filter bar
        const float filterH = 22*s;
        UiRect filterR = UiRect::fromXYWH(rx, ry, rw - 88*s, filterH);
        m_uiDrawList.addRoundRectFilled(filterR, {0.06f,0.09f,0.14f,0.85f}, 2*s);
        m_uiDrawList.addRoundRect(filterR, {0.18f,0.28f,0.48f,0.30f}, 2*s, s);
        // Filter category dots
        static const UiColor kFilterC[] = {
            {0.38f,0.76f,1.00f,0.90f},{0.50f,0.50f,0.56f,0.70f},
            {0.62f,0.26f,0.90f,0.90f},{0.88f,0.63f,0.12f,0.90f},
            {0.22f,0.72f,0.28f,0.90f},
        };
        for (int f = 0; f < 5; ++f) {
            float fcx = filterR.minX + 10*s + f * 20*s;
            float fcy = ry + filterH*0.5f;
            m_uiDrawList.addCircleFilled({fcx, fcy}, 7*s, kFilterC[f]);
            m_uiDrawList.addCircle({fcx, fcy}, 7.5f*s, {0,0,0,0.30f}, 0.8f*s);
        }

        // Sort dropdown
        UiRect sortR = UiRect::fromXYWH(rx + rw - 86*s, ry, 86*s, filterH);
        m_uiDrawList.addRoundRectFilled(sortR, {0.06f,0.09f,0.14f,0.85f}, 2*s);
        m_uiDrawList.addRoundRect(sortR, {0.18f,0.28f,0.48f,0.30f}, 2*s, s);
        m_uiDrawList.addText(m_uiFont, "Compact",
            {sortR.minX + 6*s, sortR.minY + (filterH - lh)*0.5f}, kDim);
        // Chevron (3 dots approximation)
        float ax = sortR.maxX - 14*s, ay = sortR.minY + filterH*0.5f;
        m_uiDrawList.addCircleFilled({ax - 4*s, ay - 2*s}, 2*s, kDim);
        m_uiDrawList.addCircleFilled({ax,        ay + 2*s}, 2*s, kDim);
        m_uiDrawList.addCircleFilled({ax + 4*s,  ay - 2*s}, 2*s, kDim);

        ry += filterH + 7*s;

        // ── Inventory grid
        const float itemGap =  3.0f * s;
        const int   cols    = std::max(6, std::min(8, (int)((rw + itemGap) / (38.0f*s + itemGap))));
        const float itemSz  = std::min(44.0f * s, (rw - itemGap * (cols - 1)) / cols);
        const float gridW   = cols * (itemSz + itemGap) - itemGap;
        const float gridX   = rx + (rw - gridW) * 0.5f;
        const float gridBot = y + h - 26*s;

        m_uiDrawList.pushClip(UiRect::fromXYWH(rx, ry, rw, gridBot - ry));
        for (int i = 0; i < std::min(m_invCount, kInvCapacity); ++i) {
            int col = i % cols;
            int row = i / cols;
            float ix = gridX + col * (itemSz + itemGap);
            float iy = ry + row * (itemSz + itemGap);
            if (iy + itemSz > gridBot) break;

            const InvItem& it = m_inventory[i];
            drawGearSlot(ix, iy, itemSz, it.type, it.quality, it.variant, s);

            if (it.count > 1) {
                char cnt[8];
                std::snprintf(cnt, sizeof(cnt), "%d", it.count);
                float cw = m_uiFont.measureText(cnt);
                float tx = ix + itemSz - cw - 2*s;
                float ty = iy + itemSz - lh;
                m_uiDrawList.addText(m_uiFont, cnt, {tx + s, ty + s}, {0,0,0,0.70f});
                m_uiDrawList.addText(m_uiFont, cnt, {tx, ty}, kText);
            }
        }
        m_uiDrawList.popClip();

        // Slot count + credits bar
        char slotStr[32];
        std::snprintf(slotStr, sizeof(slotStr), "%d/%d", m_invCount, kInvCapacity);
        float sw = m_uiFont.measureText(slotStr);
        m_uiDrawList.addText(m_uiFont, slotStr, {rx + rw - sw, y + h - 16*s}, kDim);

        // WIP badge (pink button like reference)
        UiRect wipBtn = UiRect::fromXYWH(rx, y + h - 26*s, 56*s, 22*s);
        m_uiDrawList.addRoundRectFilled(wipBtn, {0.95f,0.08f,0.52f,0.92f}, 3*s);
        m_uiDrawList.addText(m_uiFontBold, "WIP",
            {wipBtn.minX + (56*s - m_uiFontBold.measureText("WIP"))*0.5f,
             wipBtn.minY + (22*s - lh)*0.5f}, {1,1,1,1});

        // Credits
        m_uiDrawList.addText(m_uiFont, "8,044,558",
            {wipBtn.maxX + 8*s, wipBtn.minY + (22*s - lh)*0.5f}, kGold);
        // Credit symbol (small circle approximation)
        float credCX = wipBtn.maxX + m_uiFont.measureText("8,044,558") + 14*s;
        m_uiDrawList.addCircleFilled({credCX, wipBtn.minY + 11*s}, 4*s, kXpGold);
    }
}

// ─── Unit frame ──────────────────────────────────────────────────────────────
void SwtorApp::drawUnitFrame(float x, float y, float w, float h, const char* name,
                              float hpFrac, float resFrac, bool isPlayer, int level, float s) {
    drawPanelBg(x, y, w, h, s);

    // Portrait
    float ph = h - 14.0f * s;
    UiRect port = UiRect::fromXYWH(x + 7*s, y + 7*s, ph, ph);
    UiColor portFill     = isPlayer ? UiColor{0.10f,0.16f,0.25f,1.0f} : UiColor{0.16f,0.06f,0.06f,1.0f};
    UiColor portFillDark = isPlayer ? UiColor{0.05f,0.08f,0.14f,1.0f} : UiColor{0.09f,0.03f,0.03f,1.0f};
    UiColor silhouette   = isPlayer ? UiColor{0.40f,0.70f,1.00f,0.75f} : UiColor{0.80f,0.40f,0.30f,0.75f};
    m_uiDrawList.addRoundRectFilled(port, portFillDark, 3.0f*s);
    m_uiDrawList.addRectFilledVGradient(port,
        {portFill.r, portFill.g, portFill.b, 0.75f}, {0.0f,0.0f,0.0f,0.0f});
    m_uiDrawList.addCircleFilled({port.minX + ph*0.5f, port.minY + ph*0.35f}, ph*0.22f, silhouette);
    m_uiDrawList.addRectFilledVGradient(
        UiRect::fromXYWH(port.minX + ph*0.22f, port.minY + ph*0.54f, ph*0.56f, ph*0.35f),
        silhouette, {silhouette.r, silhouette.g, silhouette.b, 0.0f});
    m_uiDrawList.addRoundRect(port, kBorder, 3.0f*s, s);

    float tx = x + ph + 13.0f * s;
    float tw = w - ph - 18.0f * s;
    float lh = m_uiFont.lineHeightPx();

    m_uiDrawList.addText(m_uiFontBold, name, {tx, y + 8.0f*s}, kText);
    char lvl[12]; std::snprintf(lvl, sizeof(lvl), "Lv.%d", level);
    float lvlW = m_uiFont.measureText(lvl);
    m_uiDrawList.addText(m_uiFont, lvl, {tx + tw - lvlW, y + 8.0f*s}, kGold);

    const float hpBarH = lh + 4.0f * s;
    float barY = y + 10.0f * s + lh;
    drawBar(tx, barY, tw, hpBarH, hpFrac, isPlayer ? kGreen : kRed, s);

    if (isPlayer && hpFrac < 0.30f) {
        float pulse = 0.5f + 0.5f * std::sin(m_combatTimer * 5.0f);
        m_uiDrawList.addRoundRectGlow(
            UiRect::fromXYWH(tx, barY, tw, hpBarH),
            {1.0f, 0.05f, 0.02f, 0.60f * pulse}, 2.0f*s, 10.0f*s);
    }

    int maxHp = isPlayer ? 48500 : 12800;
    char hpStr[24];
    std::snprintf(hpStr, sizeof(hpStr), "%d / %d", (int)(hpFrac * maxHp), maxHp);
    float numW = m_uiFont.measureText(hpStr);
    float numX = tx + (tw - numW) * 0.5f;
    float numY = barY + (hpBarH - lh) * 0.5f;
    m_uiDrawList.pushClip(UiRect::fromXYWH(tx, barY, tw, hpBarH));
    m_uiDrawList.addText(m_uiFont, hpStr, {numX + s, numY + s}, {0.0f, 0.0f, 0.0f, 0.75f});
    m_uiDrawList.addText(m_uiFont, hpStr, {numX,     numY    }, {1.0f, 1.0f, 1.0f, 0.92f});
    m_uiDrawList.popClip();

    barY += hpBarH + 4.0f * s;
    const char* resLabel = isPlayer ? "FP" : "EN";
    float resLW = m_uiFont.measureText(resLabel) + 4.0f * s;
    m_uiDrawList.addText(m_uiFont, resLabel, {tx, barY + s}, kDim);
    UiColor resFill = isPlayer ? kBlue : UiColor{0.70f, 0.25f, 0.05f, 1.0f};
    drawBar(tx + resLW, barY, tw - resLW, 8.0f * s, resFrac, resFill, s);
}

// ─── Buff row ────────────────────────────────────────────────────────────────
void SwtorApp::drawBuffRow(float x, float y, int count, float size, float s) {
    static constexpr UiColor kBuffPalette[] = {
        {0.30f,0.70f,1.00f,0.9f},{0.40f,0.80f,0.30f,0.9f},{0.80f,0.60f,0.20f,0.9f},
        {0.70f,0.30f,0.80f,0.9f},{0.90f,0.40f,0.20f,0.9f},{0.20f,0.70f,0.70f,0.9f},
        {0.80f,0.80f,0.30f,0.9f},{0.60f,0.60f,0.60f,0.9f},
    };
    const float gap = 2.0f * s;
    for (int i = 0; i < count; ++i) {
        float bx = x + i * (size + gap);
        UiRect frame = UiRect::fromXYWH(bx, y, size, size);
        m_uiDrawList.addRoundRectFilled(frame, {0.05f,0.08f,0.12f,0.85f}, 2.0f*s);
        m_uiDrawList.addRoundRectFilled(
            UiRect::fromXYWH(bx+2*s, y+2*s, size-4*s, size-4*s),
            kBuffPalette[i % 8], 2.0f*s);
        m_uiDrawList.addRoundRect(frame, kBorder, 2.0f*s, s);
        char num[4]; std::snprintf(num, sizeof(num), "%d", 30 - i * 4);
        m_uiDrawList.addText(m_uiFont, num, {bx + 2.0f*s, y + size - m_uiFont.lineHeightPx()}, kText);
    }
}

// ─── Minimap ─────────────────────────────────────────────────────────────────
void SwtorApp::drawMinimap(float cx, float cy, float radius, float s) {
    m_uiDrawList.addCircleFilled({cx, cy}, radius + 3.0f*s, kBorder);
    m_uiDrawList.addCircleFilled({cx, cy}, radius + 1.5f*s, {0.0f,0.0f,0.0f,0.8f});
    m_uiDrawList.addCircleFilled({cx, cy}, radius, {0.02f,0.04f,0.08f,0.96f});

    for (int i = 0; i < 18; ++i) {
        float angle = i * 2.3999632f + m_mapPhase * 0.08f;
        float r = radius * (0.15f + 0.72f * std::fmod(i * 0.618033f, 1.0f));
        m_uiDrawList.addCircleFilled(
            {cx + std::cos(angle)*r, cy + std::sin(angle)*r},
            1.3f*s, {0.3f,0.4f,0.6f,0.38f});
    }

    static const struct { float ax, ay; UiColor c; } kDots[] = {
        { 0.45f, 0.30f, {0.9f,0.3f,0.2f,0.9f}},
        { 0.62f, 0.50f, {0.9f,0.3f,0.2f,0.9f}},
        {-0.35f, 0.48f, {0.2f,0.8f,0.3f,0.9f}},
        {-0.50f,-0.28f, {0.2f,0.8f,0.3f,0.9f}},
        { 0.10f,-0.58f, {0.9f,0.8f,0.5f,0.9f}},
    };
    for (const auto& d : kDots) {
        float px = cx + d.ax * radius * 0.82f + std::sin(m_mapPhase + d.ax * 3.0f) * 2.5f*s;
        float py = cy + d.ay * radius * 0.82f + std::cos(m_mapPhase + d.ay * 3.0f) * 2.5f*s;
        m_uiDrawList.addCircleFilled({px, py}, 3.0f*s, d.c);
        m_uiDrawList.addCircle({px, py}, 4.5f*s, {d.c.r, d.c.g, d.c.b, 0.3f}, s);
    }

    m_uiDrawList.addCircleFilled({cx, cy}, 5.0f*s, {1.0f,1.0f,1.0f,1.0f});
    m_uiDrawList.addCircle({cx, cy}, 7.0f*s, {0.4f,0.8f,1.0f,0.6f}, 1.5f*s);

    const char* nLabel = "N";
    float nW = m_uiFont.measureText(nLabel);
    m_uiDrawList.addText(m_uiFont, nLabel,
        {cx - nW * 0.5f, cy - radius - m_uiFont.lineHeightPx() - 2.0f*s}, kDim);

    const char* zone = "Hutta \xe2\x80\x94 Jiguuna";
    float zW = m_uiFont.measureText(zone);
    m_uiDrawList.addText(m_uiFont, zone, {cx - zW * 0.5f, cy + radius + 4.0f*s}, kDim);
}

// ─── Action bar ──────────────────────────────────────────────────────────────
void SwtorApp::drawActionBar(float x, float y, int slots, float s) {
    const float kSlot = 50.0f * s;
    const float kGap  =  3.0f * s;

    float totalW = slots * (kSlot + kGap) - kGap;
    UiRect bg = UiRect::fromXYWH(x - 5*s, y - 5*s, totalW + 10*s, kSlot + 10*s);
    m_uiDrawList.addDropShadow(bg, kShadow, 6.0f*s, 0, 3);
    m_uiDrawList.addRoundRectFilledHGradient(bg,
        {0.05f,0.08f,0.14f,0.84f}, {0.03f,0.04f,0.08f,0.82f}, 5.0f*s);
    m_uiDrawList.addRoundRect(bg, kBorder, 5.0f*s, s);

    for (int i = 0; i < slots; ++i) {
        float sx = x + i * (kSlot + kGap);
        UiRect slot = UiRect::fromXYWH(sx, y, kSlot, kSlot);

        m_uiDrawList.addRoundRectFilled(slot, kSlotBg, 3.0f*s);

        const UiColor& ac = kAbilityColor[i % 12];
        UiRect icon = UiRect::fromXYWH(sx + 3*s, y + 3*s, kSlot - 6*s, kSlot - 6*s);
        m_uiDrawList.addRoundRectFilled(icon,
            {ac.r*0.35f, ac.g*0.35f, ac.b*0.35f, 0.9f}, 2.0f*s);

        float icx = sx + kSlot * 0.5f, icy = y + kSlot * 0.5f;
        m_uiDrawList.addCircleFilled({icx, icy}, kSlot * 0.22f, ac);
        m_uiDrawList.addCircle({icx, icy}, kSlot * 0.30f,
            {ac.r, ac.g, ac.b, 0.45f}, 1.5f*s);

        if (m_cooldowns[i] > 0.0f) {
            drawCooldown(icx, icy, kSlot * 0.40f, m_cooldowns[i]);
        } else {
            float pulse = 0.22f + 0.14f * std::sin(m_combatTimer * 2.2f + i * 0.9f);
            m_uiDrawList.addRoundRectGlow(slot, {ac.r, ac.g, ac.b, pulse}, 3.0f*s, 5.0f*s);
        }

        m_uiDrawList.addRoundRect(slot, {0.5f,0.55f,0.65f,0.38f}, 3.0f*s, s);
        m_uiDrawList.addText(m_uiFont, kAbilityKey[i], {sx + 3.0f*s, y + 2.0f*s}, {0.7f,0.7f,0.7f,0.8f});
    }
}

// ─── Chat window ─────────────────────────────────────────────────────────────
void SwtorApp::drawChatWindow(float x, float y, float w, float h, float s,
                              const std::string& inputText, float caretPhase) {
    drawPanelBg(x, y, w, h, s);

    static const char* kTabs[] = {"General", "Guild", "Operations"};
    static const UiColor kTabAccent[] = {
        {0.3f,0.6f,1.0f,1.0f},{0.3f,0.9f,0.4f,1.0f},{1.0f,0.6f,0.2f,1.0f}
    };
    float tabX = x + 4.0f*s;
    const float tabH = 22.0f * s;
    for (int t = 0; t < 3; ++t) {
        float tw = m_uiFont.measureText(kTabs[t]) + 16.0f*s;
        UiRect tabR = UiRect::fromXYWH(tabX, y + 4.0f*s, tw, tabH);
        bool active = (t == 0);
        if (active)
            m_uiDrawList.addRoundRectFilledHGradient(tabR,
                {0.16f,0.26f,0.44f,0.95f}, {0.09f,0.15f,0.24f,0.90f}, 3.0f*s);
        else
            m_uiDrawList.addRoundRectFilled(tabR, {0.06f,0.10f,0.16f,0.70f}, 3.0f*s);
        if (active)
            m_uiDrawList.addRoundRect(tabR, {kTabAccent[t].r,kTabAccent[t].g,kTabAccent[t].b,0.4f}, 3.0f*s, s);
        m_uiDrawList.addText(m_uiFont, kTabs[t], {tabX + 8.0f*s, y + 7.0f*s},
            active ? kTabAccent[t] : kDim);
        tabX += tw + 2.0f*s;
    }

    const float chatTop = y + 30.0f*s;
    m_uiDrawList.pushClip(UiRect::fromXYWH(x + 3*s, chatTop, w - 6*s, h - 56*s));

    float lh = m_uiFont.lineHeightPx();
    float textY = y + 32.0f*s;
    int first = std::max(0, (int)m_chatLog.size() - 7);
    for (int i = first; i < (int)m_chatLog.size(); ++i) {
        const auto& msg = m_chatLog[i];
        UiColor cc = (msg.channel == "Guild") ? kTabAccent[1]
                   : (msg.channel == "Ops")   ? kTabAccent[2]
                   : kTabAccent[0];
        std::string prefix = "[" + msg.channel + "] " + msg.sender + ": ";
        float pw = m_uiFont.measureText(prefix);
        m_uiDrawList.addText(m_uiFont, prefix,   {x + 6.0f*s,       textY}, cc);
        m_uiDrawList.addText(m_uiFont, msg.text,  {x + 6.0f*s + pw,  textY}, kText);
        textY += lh;
    }
    m_uiDrawList.popClip();

    const float boxH = lh + 6.0f * s;
    float iy = y + h - boxH - 4.0f * s;
    UiRect inputBox = UiRect::fromXYWH(x + 4*s, iy, w - 8*s, boxH);
    m_uiDrawList.addRoundRectFilled(inputBox, {0.06f,0.10f,0.16f,0.92f}, 2.0f*s);
    m_uiDrawList.addRoundRect(inputBox, kBorder, 2.0f*s, s);

    const float inputTextY = iy + (boxH - lh) * 0.5f;
    const float inputTextX = x + 8.0f * s;
    m_uiDrawList.pushClip(inputBox);
    if (inputText.empty()) {
        m_uiDrawList.addText(m_uiFont, "Type message here...", {inputTextX, inputTextY}, kDim);
    } else {
        m_uiDrawList.addText(m_uiFont, inputText, {inputTextX, inputTextY}, kText);
        if (std::fmod(caretPhase, 1.0f) < 0.55f) {
            float caretX = inputTextX + m_uiFont.measureText(inputText);
            m_uiDrawList.addRectFilled(
                UiRect::fromXYWH(caretX + 1.0f*s, inputTextY + 1.0f*s, 1.5f*s, lh - 2.0f*s),
                {0.91f, 0.82f, 0.51f, 0.9f});
        }
    }
    m_uiDrawList.popClip();
}

// ─── Mission tracker ─────────────────────────────────────────────────────────
void SwtorApp::drawMissionTracker(float x, float y, float w, float h, float s) {
    drawPanelBg(x, y, w, h, s);

    m_uiDrawList.addRoundRectFilledHGradient(
        UiRect::fromXYWH(x, y, w, 30*s),
        {0.14f,0.22f,0.38f,0.78f}, {0.06f,0.10f,0.18f,0.55f}, 4.0f*s);
    m_uiDrawList.addText(m_uiFontBold, "MISSION TRACKER", {x + 10.0f*s, y + 8.0f*s}, kGold);

    float ty = y + 38.0f*s;
    m_uiDrawList.addText(m_uiFontBold, "The Esseles: Imperial Attack", {x + 8.0f*s, ty}, kText);
    ty += m_uiFont.lineHeightPx() + 8.0f*s;

    struct Obj { const char* text; bool done; };
    static const Obj objs[] = {
        {"Defeat Commander Dumas",     true },
        {"Secure the Engineering Bay", true },
        {"Reach the Bridge",           false},
    };
    for (const auto& obj : objs) {
        UiRect cb = UiRect::fromXYWH(x + 8.0f*s, ty + 2.0f*s, 12.0f*s, 12.0f*s);
        m_uiDrawList.addRoundRectFilled(cb,
            obj.done ? kGreen : UiColor{0.2f,0.3f,0.4f,0.8f}, 2.0f*s);
        m_uiDrawList.addRoundRect(cb, kBorder, 2.0f*s, s);
        m_uiDrawList.addText(m_uiFont, obj.text, {x + 24.0f*s, ty},
            obj.done ? kDim : kText);
        ty += m_uiFont.lineHeightPx() + 6.0f*s;
    }

    ty += 4.0f*s;
    m_uiDrawList.addRectFilled(UiRect::fromXYWH(x + 6*s, ty, w - 12*s, s), {0.2f,0.3f,0.5f,0.3f});
    ty += 10.0f*s;
    m_uiDrawList.addText(m_uiFont, "Rewards", {x + 8.0f*s, ty}, kGold);
    ty += m_uiFont.lineHeightPx() + 4.0f*s;
    m_uiDrawList.addText(m_uiFont, "2,300 XP  |  5,400 Credits", {x + 8.0f*s, ty}, kText);
}

// ─── XP bar ──────────────────────────────────────────────────────────────────
void SwtorApp::drawXpBar(float x, float y, float w, float h, float frac, int level, float s) {
    UiRect bg = UiRect::fromXYWH(x, y, w, h);
    m_uiDrawList.addRectFilled(bg, {0.04f,0.06f,0.08f,1.0f});
    if (frac > 0.001f) {
        float fw = w * std::min(frac, 1.0f);
        m_uiDrawList.addRectFilledHGradient(
            UiRect::fromXYWH(x, y, fw, h), kXpGold,
            {kXpGold.r*0.65f, kXpGold.g*0.65f, kXpGold.b*0.65f, 1.0f});
        float shimPhase = std::fmod(m_combatTimer * 0.25f, 1.0f);
        float shimW = fw * 0.22f;
        float shimX = x - shimW + (fw + shimW) * shimPhase;
        m_uiDrawList.pushClip(UiRect::fromXYWH(x, y, fw, h));
        m_uiDrawList.addRectFilledHGradient(
            UiRect::fromXYWH(shimX,              y, shimW * 0.5f, h),
            {1.0f,1.0f,0.7f,0.0f}, {1.0f,1.0f,0.7f,0.22f});
        m_uiDrawList.addRectFilledHGradient(
            UiRect::fromXYWH(shimX + shimW*0.5f, y, shimW * 0.5f, h),
            {1.0f,1.0f,0.7f,0.22f}, {1.0f,1.0f,0.7f,0.0f});
        m_uiDrawList.popClip();
    }
    m_uiDrawList.addRect(bg, {0,0,0,0.45f}, 1.0f);

    char txt[64];
    std::snprintf(txt, sizeof(txt), "Level %d  |  %.0f / 125,000 XP", level, frac * 125000.0f);
    float tw = m_uiFont.measureText(txt);
    m_uiDrawList.addText(m_uiFont, txt,
        {x + (w - tw) * 0.5f, y + (h - m_uiFont.lineHeightPx()) * 0.5f}, kText);
}

// ─── Lifecycle ───────────────────────────────────────────────────────────────
void SwtorApp::drawSpaceportInterior(float fbW, float fbH, float s) {
    m_uiDrawList.addRectFilledVGradient(UiRect::fromXYWH(0, 0, fbW, fbH),
        {0.025f, 0.050f, 0.070f, 1.0f}, {0.004f, 0.012f, 0.022f, 1.0f});
    const float horizon = fbH * 0.46f;
    m_uiDrawList.addRectFilled(UiRect::fromXYWH(0, horizon, fbW, fbH - horizon),
        {0.025f,0.070f,0.085f,1.0f});
    for (int i = 0; i < 7; ++i) {
        const float x = (fbW / 7.0f) * i;
        m_uiDrawList.addRectFilledVGradient(UiRect::fromXYWH(x + 7*s, 0, 5*s, horizon),
            {0.10f,0.45f,0.52f,0.45f}, {0.01f,0.05f,0.08f,0.15f});
        m_uiDrawList.addRectFilled(UiRect::fromXYWH(x + 12*s, 0, fbW/7.0f - 24*s, 4*s),
            {0.10f,0.55f,0.64f,0.36f});
    }
    for (int i = 0; i < 4; ++i) {
        const float lightX = fbW * (0.16f + i * 0.23f);
        m_uiDrawList.addCircleFilled({lightX, horizon * 0.34f}, 34*s, {1.0f,0.57f,0.13f,0.08f});
        m_uiDrawList.addCircleFilled({lightX, horizon * 0.34f}, 5*s, {1.0f,0.72f,0.26f,0.82f});
    }
    for (int row = 0; row < 5; ++row) {
        const float y = horizon + (fbH - horizon) * (row / 5.0f);
        m_uiDrawList.addRectFilled(UiRect::fromXYWH(0, y, fbW, s), {0.15f,0.62f,0.68f,0.18f});
    }
    for (int i = -4; i <= 4; ++i) {
        const float baseX = fbW * 0.5f + i * fbW * 0.18f;
        const float topX = fbW * 0.5f + i * fbW * 0.045f;
        const float width = std::max(1.0f*s, std::abs(baseX - topX) * 0.015f);
        m_uiDrawList.addRectFilled(UiRect::fromXYWH(topX, horizon, width, fbH - horizon),
            {0.10f,0.62f,0.70f,0.20f});
    }
    m_uiDrawList.addRectFilledVGradient(UiRect::fromXYWH(0, 0, fbW, fbH),
        {0.0f,0.0f,0.0f,0.08f}, {0.0f,0.0f,0.0f,0.38f});
}

void SwtorApp::drawTopNavigation(float fbW, float s) {
    const float w = std::min(390.0f*s, fbW * 0.42f);
    const float h = 31.0f*s;
    const float x = (fbW - w) * 0.5f;
    const float y = 9.0f*s;
    UiRect ribbon = UiRect::fromXYWH(x, y, w, h);
    m_uiDrawList.addDropShadow(ribbon, kShadow, 8*s, 0, 3*s);
    m_uiDrawList.addRoundRectFilledHGradient(ribbon,
        {0.02f,0.26f,0.36f,0.95f}, {0.01f,0.10f,0.16f,0.96f}, 10*s);
    m_uiDrawList.addRoundRect(ribbon, kBorder, 10*s, s);
    static const char* kNav[] = {"CHAR", "STYLE", "MAP", "SOCIAL", "SET"};
    float nx = x + 13*s;
    for (int i = 0; i < 5; ++i) {
        m_uiDrawList.addCircleFilled({nx + 5*s, y + h*0.5f}, 4*s,
            i == 0 ? kBlue : UiColor{0.33f,0.68f,0.76f,0.76f});
        nx += 12*s;
        const float tw = m_uiFont.measureText(kNav[i]);
        m_uiDrawList.addText(m_uiFont, kNav[i], {nx, y + (h - m_uiFont.lineHeightPx())*0.5f},
            i == 0 ? kText : kDim);
        nx += tw + 13*s;
    }
    UiRect status = UiRect::fromXYWH(10*s, 10*s, 174*s, 26*s);
    m_uiDrawList.addRoundRectFilled(status, {0.01f,0.12f,0.18f,0.92f}, 5*s);
    m_uiDrawList.addRoundRect(status, kBorder, 5*s, s);
    m_uiDrawList.addText(m_uiFontBold, "55", {status.minX + 9*s, status.minY + 5*s}, kText);
    m_uiDrawList.addCircleFilled({status.minX + 42*s, status.minY + 13*s}, 4*s, kGreen);
    m_uiDrawList.addText(m_uiFont, "SCYA  |  8,044,558", {status.minX + 52*s, status.minY + 5*s}, kDim);
}

bool SwtorApp::onInit() {
    const float s = contentScale();
    if (!loadFonts(
            resolveAssetPath("assets/fonts/Inter-Regular.ttf"),
            resolveAssetPath("assets/fonts/Inter-Bold.ttf"),
            resolveAssetPath("assets/fonts/Inter-Italic.ttf"),
            resolveAssetPath("assets/fonts/Inter-Regular.ttf"),
            std::round(17.0f * s), std::round(15.0f * s)))
        return false;

    const std::string iconSheetPath =
        resolveAssetPath("assets/ui/swtor/swtor_icon_sheet.png");
    int iconSheetW = 0;
    int iconSheetH = 0;
    int iconSheetChannels = 0;
    stbi_uc* iconSheetPixels = stbi_load(
        iconSheetPath.c_str(), &iconSheetW, &iconSheetH, &iconSheetChannels, 4);
    if (iconSheetPixels != nullptr) {
        m_iconSheet = m_renderer.registerUiTextureRgba8Mipmapped(
            iconSheetPixels, static_cast<std::uint32_t>(iconSheetW),
            static_cast<std::uint32_t>(iconSheetH));
        stbi_image_free(iconSheetPixels);
    }

    auto root = std::make_unique<Widget>();
    root->mousePassthrough = true;
    m_uiContext.setRoot(std::move(root));

    // Seed initial chat
    m_chatLog.push_back({"General", "WelcomeBot",  "Welcome to The Esseles!"});
    m_chatLog.push_back({"Guild",   "GuildMaster", "Good luck on the Op tonight"});

    // Stagger initial cooldowns
    m_cooldowns[4] = 0.75f;
    m_cooldowns[6] = 0.40f;
    m_cooldowns[8] = 0.20f;

    // Generate inventory deterministically
    uint32_t seed = 0xDAEDBEEFu;
    auto lcg = [&]() -> uint32_t { return (seed = seed * 1664525u + 1013904223u); };
    for (int i = 0; i < std::min(m_invCount, kInvCapacity); ++i) {
        m_inventory[i].type = static_cast<int>(lcg() % 8);
        float qr = (lcg() >> 8) / float(1 << 24);
        m_inventory[i].quality = qr < 0.12f ? 0 : qr < 0.35f ? 1 : qr < 0.60f ? 2 : qr < 0.85f ? 3 : 4;
        m_inventory[i].count   = (lcg() % 7 == 0) ? static_cast<int>(lcg() % 98 + 2) : 0;
        m_inventory[i].variant = static_cast<int>(lcg() % 4);
    }

    return true;
}

void SwtorApp::onTick(float dt) {
    m_combatTimer += dt;
    m_mapPhase    += dt * 0.15f;

    if (m_combatTimer - m_lastDmg > 2.5f) {
        m_targetHp -= 0.05f + 0.04f * std::sin(m_combatTimer);
        m_playerHp -= 0.015f;
        m_playerHp  = std::max(0.30f, m_playerHp);
        if (m_targetHp <= 0.08f) {
            m_targetHp = 1.0f;
            m_cooldowns[4] = 1.0f;
            m_cooldowns[5] = 0.75f;
        }
        m_lastDmg = m_combatTimer;
        int used = ((int)(m_combatTimer * 0.8f)) % 12;
        m_cooldowns[used] = 1.0f;
    }

    m_playerHp    = std::min(1.0f, m_playerHp    + dt * 0.018f);
    m_playerForce = std::min(1.0f, m_playerForce + dt * 0.012f);
    m_xpFrac      = std::min(1.0f, m_xpFrac      + dt * 0.0004f);

    for (float& cd : m_cooldowns)
        if (cd > 0.0f) cd = std::max(0.0f, cd - dt * 0.22f);

    m_chatTimer += dt;
    if (m_chatTimer > 3.8f) {
        m_chatTimer = 0.0f;
        const auto& src = kChatMsgs[m_nextChat % kNumChatMsgs];
        m_chatLog.push_back({src.ch, src.from, src.msg});
        if (m_chatLog.size() > 14) m_chatLog.erase(m_chatLog.begin());
        ++m_nextChat;
    }

    for (std::uint32_t cp : m_uiInput.textInput) {
        if (cp >= 32 && m_chatInput.size() < 128) {
            if (cp < 0x80) {
                m_chatInput += static_cast<char>(cp);
            } else if (cp < 0x800) {
                m_chatInput += static_cast<char>(0xC0 | (cp >> 6));
                m_chatInput += static_cast<char>(0x80 | (cp & 0x3F));
            } else {
                m_chatInput += static_cast<char>(0xE0 | (cp >> 12));
                m_chatInput += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                m_chatInput += static_cast<char>(0x80 | (cp & 0x3F));
            }
        }
    }

    bool bsDown = (glfwGetKey(m_window, GLFW_KEY_BACKSPACE) == GLFW_PRESS);
    if (bsDown && !m_prevBackspace && !m_chatInput.empty()) {
        while (!m_chatInput.empty() && (m_chatInput.back() & 0xC0) == 0x80)
            m_chatInput.pop_back();
        if (!m_chatInput.empty()) m_chatInput.pop_back();
    }
    m_prevBackspace = bsDown;

    bool enterDown = (glfwGetKey(m_window, GLFW_KEY_ENTER) == GLFW_PRESS);
    if (enterDown && !m_prevEnter && !m_chatInput.empty()) {
        m_chatLog.push_back({"General", "You", m_chatInput});
        if (m_chatLog.size() > 14) m_chatLog.erase(m_chatLog.begin());
        m_chatInput.clear();
    }
    m_prevEnter = enterDown;

    // Toggle character window with C
    bool cDown = (glfwGetKey(m_window, GLFW_KEY_C) == GLFW_PRESS);
    if (cDown && !m_prevCKey) m_showCharWindow = !m_showCharWindow;
    m_prevCKey = cDown;

    if (glfwGetKey(m_window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(m_window, GLFW_TRUE);
}

void SwtorApp::onRender(float /*dt*/) {
    int fbW, fbH;
    framebufferSize(fbW, fbH);
    if (fbW <= 0 || fbH <= 0) return;

    beginFrameDraw();

    const float fw = (float)fbW, fh = (float)fbH;
    const float s  = contentScale();

    drawSpaceportInterior(fw, fh, s);
    drawTopNavigation(fw, s);

    // ── HUD ───────────────────────────────────────────────────────────────────
    drawUnitFrame(10*s, 48*s, 222*s, 90*s, "Jyn Erso",       m_playerHp, m_playerForce, true,  55, s);
    drawBuffRow  (10*s, 144*s, 8, 24*s, s);
    drawUnitFrame(10*s, 178*s, 222*s, 90*s, "Kaliyo Djannis", m_targetHp, 0.0f,          false, 52, s);
    drawBuffRow  (10*s, 274*s, 3, 22*s, s);

    float mmR  = 88.0f * s;
    float mmCX = fw - mmR - 15.0f * s;
    float mmCY = mmR + 15.0f * s;
    drawMinimap(mmCX, mmCY, mmR, s);

    drawMissionTracker(fw - 258.0f*s, mmCY + mmR + 22.0f*s, 248.0f*s, 200.0f*s, s);

    drawChatWindow(10.0f*s, fh - 244.0f*s, 365.0f*s, 214.0f*s, s,
                   m_chatInput, m_combatTimer);

    const float kSlotW = 50.0f * s;
    const float kGapW  =  3.0f * s;
    static constexpr int kNSlots = 12;
    float barTotalW = kNSlots * (kSlotW + kGapW) - kGapW;
    drawActionBar(fw * 0.5f - barTotalW * 0.5f, fh - 75.0f*s, kNSlots, s);

    drawXpBar(0.0f, fh - 10.0f*s, fw, 10.0f*s, m_xpFrac, 55, s);

    // ── Character window overlay (press C to toggle) ───────────────────────
    if (m_showCharWindow) {
        float winW = std::min(880.0f * s, fw * 0.92f);
        float winH = std::min(580.0f * s, fh * 0.88f);
        float winX = (fw - winW) * 0.5f;
        float winY = (fh - winH) * 0.5f;
        // Dim the HUD behind the window
        m_uiDrawList.addRectFilled(UiRect::fromXYWH(0, 0, fw, fh), {0.0f,0.0f,0.0f,0.45f});
        drawCharWindow(winX, winY, winW, winH, s);
    }

    submitFrame(m_camera);
}

} // namespace odai::games::swtor
