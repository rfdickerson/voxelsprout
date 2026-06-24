#include "tools/retro_theme_demo/retro_theme_demo_app.h"

#include "ui/vector/vector_path.h"
#include "ui/vector/vector_tessellator.h"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace odai::tools::retro_theme_demo {
using namespace ui;

static constexpr float kMacR  = 4.0f;
static constexpr float kFlatR = 8.0f;   // Flat Retro corner radius (logical px)

// Flat Retro accent colors (per-window title bars) + outline ink.
static const UiColor kFlatBlue  {0.34f,0.60f,0.83f,1.f};
static const UiColor kFlatGreen {0.46f,0.76f,0.36f,1.f};
static const UiColor kFlatPink  {0.82f,0.36f,0.56f,1.f};
static const UiColor kFlatOrange{0.89f,0.62f,0.22f,1.f};
static const UiColor kFlatInk   {0.16f,0.16f,0.22f,1.f};

// Retro-OS (Windows-10 flat) tokens.
static const UiColor kOsBlue    {0.000f,0.471f,0.843f,1.f};  // #0078D7
static const UiColor kOsBlueHov {0.063f,0.404f,0.710f,1.f};  // #106EBE
static const UiColor kOsRed     {0.910f,0.067f,0.137f,1.f};  // #E81123
static const UiColor kOsGray    {0.627f,0.627f,0.627f,1.f};  // #A0A0A0
static const UiColor kOsWhite   {0.941f,0.941f,0.941f,1.f};  // #F0F0F0
static const UiColor kOsBorder  {0.700f,0.700f,0.700f,1.f};
static const UiColor kOsDisable {0.855f,0.855f,0.855f,1.f};
static const UiColor kOsInk     {0.114f,0.114f,0.114f,1.f};

// ─── Menu data ────────────────────────────────────────────────────────────────
// Null pointer terminates the item list; "-" is a separator.
static const char* kItemsFile[]    = {"New","Open...","Save","-","Exit",nullptr};
static const char* kItemsEdit[]    = {"Undo","-","Cut","Copy","Paste","-","Select All",nullptr};
static const char* kItemsView[]    = {"Large Icons","Small Icons","List","Details",nullptr};
static const char* kItemsOptions[] = {"Arrange Icons","Refresh","-","Properties",nullptr};
static const char* kItemsSpecial[] = {"Empty Trash...","Eject","-","Restart...","Shut Down...",nullptr};
static const char* kItemsHelp[]    = {"Help Topics","About...",nullptr};

static const char*  kTitlesWin95[5] = {"File","Edit","View","Options","Help"};
static const char*  kTitlesMac[5]   = {"File","Edit","View","Special","Help"};
static const char** kMenusWin95[5]  = {kItemsFile,kItemsEdit,kItemsView,kItemsOptions,kItemsHelp};
static const char** kMenusMac[5]    = {kItemsFile,kItemsEdit,kItemsView,kItemsSpecial, kItemsHelp};

// ─── Palettes ─────────────────────────────────────────────────────────────────
static const ThemePalette kWin95{
    {0.000f,0.502f,0.502f,1.f},  // desktop: teal
    {0.753f,0.753f,0.753f,1.f},  // face:    #C0C0C0
    {0.000f,0.000f,0.502f,1.f},  // title:   navy
    {1.000f,1.000f,1.000f,1.f},  // titleText: white
    {0.000f,0.000f,0.000f,1.f},  // text
    {0.502f,0.502f,0.502f,1.f},  // textDim
    {0.000f,0.000f,0.502f,1.f},  // highlight: navy
    "Windows 95",
};
static const ThemePalette kMotif{
    {0.220f,0.220f,0.380f,1.f},
    {0.682f,0.698f,0.765f,1.f},
    {0.373f,0.373f,0.620f,1.f},
    {1.000f,1.000f,1.000f,1.f},
    {0.000f,0.000f,0.000f,1.f},
    {0.353f,0.353f,0.416f,1.f},
    {0.373f,0.373f,0.620f,1.f},
    "Motif / CDE",
};
static const ThemePalette kMacClassic{
    {0.667f,0.667f,0.667f,1.f},  // desktop: System-7 #AAAAAA gray
    {1.000f,1.000f,1.000f,1.f},
    {1.000f,1.000f,1.000f,1.f},
    {0.000f,0.000f,0.000f,1.f},
    {0.000f,0.000f,0.000f,1.f},
    {0.400f,0.400f,0.400f,1.f},
    {0.000f,0.000f,0.000f,1.f},
    "Mac Classic",
};
static const ThemePalette kFlatRetro{
    {0.929f,0.914f,0.878f,1.f},  // desktop: light warm gray #EDE9E0
    {0.984f,0.965f,0.914f,1.f},  // face:    cream #FBF6E9
    {0.34f, 0.60f, 0.83f, 1.f},  // titleActive: blue accent (per-window overrides)
    {1.000f,1.000f,1.000f,1.f},  // titleText: white
    {0.149f,0.149f,0.180f,1.f},  // text:    near-black #26262E
    {0.431f,0.431f,0.478f,1.f},  // textDim: #6E6E7A
    {0.34f, 0.60f, 0.83f, 1.f},  // highlight: blue accent
    "Flat Retro",
};
static const ThemePalette kRetroOs{
    {0.918f,0.937f,0.953f,1.f},  // desktop: very light blue-gray
    {0.941f,0.941f,0.941f,1.f},  // face:    #F0F0F0
    {0.000f,0.471f,0.843f,1.f},  // titleActive: blue #0078D7
    {1.000f,1.000f,1.000f,1.f},  // titleText: white
    {0.114f,0.114f,0.114f,1.f},  // text
    {0.502f,0.502f,0.502f,1.f},  // textDim
    {0.000f,0.471f,0.843f,1.f},  // highlight: blue
    "Retro-OS",
};

// ─── Accessors ────────────────────────────────────────────────────────────────

const ThemePalette& RetroDemoApp::palette() const {
    if (m_theme == Theme::Win95)     return kWin95;
    if (m_theme == Theme::Motif)     return kMotif;
    if (m_theme == Theme::FlatRetro) return kFlatRetro;
    if (m_theme == Theme::RetroOS)   return kRetroOs;
    return kMacClassic;
}
const Font& RetroDemoApp::themeFont() const {
    return (m_theme == Theme::ClassicMac && m_macFont.valid()) ? m_macFont : m_uiFont;
}
const Font& RetroDemoApp::themeFontBold() const {
    return (m_theme == Theme::ClassicMac && m_macFont.valid()) ? m_macFont : m_uiFontBold;
}
const char** RetroDemoApp::currentMenuItems(int idx) const {
    if (idx < 0 || idx >= 5) return nullptr;
    return (m_theme == Theme::ClassicMac) ? kMenusMac[idx] : kMenusWin95[idx];
}

// ─── Hit testing ─────────────────────────────────────────────────────────────

bool RetroDemoApp::isHovered(UiRect r) const {
    return m_mouseX >= r.minX && m_mouseX < r.maxX &&
           m_mouseY >= r.minY && m_mouseY < r.maxY;
}
bool RetroDemoApp::isPressed(UiRect r) const {
    return isHovered(r) && m_lmbDown;
}

// ─── Bevel panel (no SDF) ─────────────────────────────────────────────────────

void RetroDemoApp::drawBevelPanel(UiRect r, bool raised, float s) {
    const float t = std::max(1.f, std::round(s));
    if (m_theme == Theme::ClassicMac) {
        m_uiDrawList.addRoundRectFilled(r, {1,1,1,1}, kMacR * s);
        m_uiDrawList.addRoundRect(r, {0,0,0,1}, kMacR * s, t);
        return;
    }
    UiColor hiOut, hiIn, shIn, shOut, fill;
    if (m_theme == Theme::Win95) {
        fill  = {0.753f,0.753f,0.753f,1.f};
        hiOut = {1.f,   1.f,   1.f,   1.f};
        hiIn  = {0.871f,0.871f,0.871f,1.f};
        shIn  = {0.502f,0.502f,0.502f,1.f};
        shOut = {0.251f,0.251f,0.251f,1.f};
    } else {
        fill  = {0.682f,0.698f,0.765f,1.f};
        hiOut = hiIn = {0.871f,0.878f,0.914f,1.f};
        shOut = shIn = {0.298f,0.314f,0.376f,1.f};
    }
    if (!raised) { std::swap(hiOut,shOut); std::swap(hiIn,shIn); }
    m_uiDrawList.addRectFilled(r, fill);
    m_uiDrawList.addRectFilled({r.minX,     r.minY,     r.maxX,     r.minY+t  }, hiOut);
    m_uiDrawList.addRectFilled({r.minX,     r.minY,     r.minX+t,   r.maxY    }, hiOut);
    m_uiDrawList.addRectFilled({r.minX,     r.maxY-t,   r.maxX,     r.maxY    }, shOut);
    m_uiDrawList.addRectFilled({r.maxX-t,   r.minY,     r.maxX,     r.maxY    }, shOut);
    m_uiDrawList.addRectFilled({r.minX+t,   r.minY+t,   r.maxX-t,   r.minY+2*t}, hiIn);
    m_uiDrawList.addRectFilled({r.minX+t,   r.minY+t,   r.minX+2*t, r.maxY-t  }, hiIn);
    m_uiDrawList.addRectFilled({r.minX+t,   r.maxY-2*t, r.maxX-t,   r.maxY-t  }, shIn);
    m_uiDrawList.addRectFilled({r.maxX-2*t, r.minY+t,   r.maxX-t,   r.maxY-t  }, shIn);
}

// ─── Group border ─────────────────────────────────────────────────────────────

void RetroDemoApp::drawGroupBorder(float gbX, float gbY, float gbW, float gbH,
                                    float gapX, float gapW, float s) {
    const float t  = std::max(1.f, std::round(s));
    const float x1 = gbX + gbW, y1 = gbY + gbH;
    if (m_theme == Theme::ClassicMac) {
        m_uiDrawList.addRoundRect(UiRect::fromXYWH(gbX,gbY,gbW,gbH), {0,0,0,1}, kMacR*s, t);
        return;
    }
    UiColor outer, inner;
    if (m_theme == Theme::Win95) {
        outer = {0.502f,0.502f,0.502f,1.f}; inner = {1.f,1.f,1.f,1.f};
    } else {
        outer = {0.298f,0.314f,0.376f,1.f}; inner = {0.871f,0.878f,0.914f,1.f};
    }
    const float gx = gapX, gx2 = gapX + gapW;
    m_uiDrawList.addRectFilled({gbX,   gbY,    gx-3,    gbY+t  }, outer);
    m_uiDrawList.addRectFilled({gx2+3, gbY,    x1,      gbY+t  }, outer);
    m_uiDrawList.addRectFilled({gbX+t, gbY+t,  gx-3,    gbY+2*t}, inner);
    m_uiDrawList.addRectFilled({gx2+3, gbY+t,  x1-t,    gbY+2*t}, inner);
    m_uiDrawList.addRectFilled({gbX,   gbY,    gbX+t,   y1     }, outer);
    m_uiDrawList.addRectFilled({gbX+t, gbY+t,  gbX+2*t, y1-t   }, inner);
    m_uiDrawList.addRectFilled({gbX,   y1-t,   x1,      y1     }, inner);
    m_uiDrawList.addRectFilled({gbX+t, y1-2*t, x1-t,    y1-t   }, outer);
    m_uiDrawList.addRectFilled({x1-t,  gbY,    x1,      y1     }, inner);
    m_uiDrawList.addRectFilled({x1-2*t,gbY+t,  x1-t,    y1-t   }, outer);
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

void RetroDemoApp::drawCenteredText(UiRect r, const Font& font,
                                     const char* text, UiColor color) {
    m_uiDrawList.addText(font, text,
        {r.minX + (r.width()  - font.measureText(text)) * 0.5f,
         r.minY + (r.height() - font.lineHeightPx())    * 0.5f}, color);
}

// ─── Interactive button ───────────────────────────────────────────────────────

bool RetroDemoApp::drawButton(UiRect r, const char* label, float s) {
    const ThemePalette& pal = palette();
    const Font& fn  = themeFontBold();
    const float t   = std::max(1.f, std::round(s));
    const bool hov  = isHovered(r);
    const bool dn   = isPressed(r);
    const bool hit  = m_justClicked && hov;
    if (hit) m_clickHandled = true;

    if (m_theme == Theme::ClassicMac) {
        if (dn) {
            m_uiDrawList.addRoundRectFilled(r, {0,0,0,1}, kMacR * s);
            m_uiDrawList.addRoundRect(r, {0,0,0,1}, kMacR * s, t);
            drawCenteredText(r, fn, label, {1,1,1,1});
        } else {
            drawBevelPanel(r, true, s);
            drawCenteredText(r, fn, label, pal.text);
        }
    } else {
        drawBevelPanel(r, !dn, s);
        // Offset text by 1px when pressed (classic Win95 button depression)
        const float off = dn ? t : 0.f;
        const float tw  = fn.measureText(label);
        const float lh  = fn.lineHeightPx();
        m_uiDrawList.addText(fn, label,
            {r.minX + (r.width()  - tw) * 0.5f + off,
             r.minY + (r.height() - lh) * 0.5f + off}, pal.text);
    }
    return hit;
}

// ─── Menu strip (horizontal title bar) ───────────────────────────────────────

void RetroDemoApp::drawMenuStrip(float barX, float barY, float barW, float barH,
                                  const char** titles, int numTitles, float s) {
    const ThemePalette& pal = palette();
    const Font& fn  = themeFont();
    const float lh  = fn.lineHeightPx();
    const float iPad = 7.f * s;

    float mx = barX;
    for (int i = 0; i < numTitles; ++i) {
        const float tw = fn.measureText(titles[i]);
        const UiRect tr = UiRect::fromXYWH(mx, barY, tw + 2.f * iPad, barH);
        const bool hov = isHovered(tr);

        // Click to open; click same title again to close
        if (hov && m_justClicked) {
            m_clickHandled = true;
            m_openMenu = (m_openMenu == i) ? -1 : i;
        }
        // Hover-switch when any menu is already open
        if (hov && m_openMenu >= 0 && m_openMenu != i) {
            m_openMenu = i;
        }

        const bool active = (m_openMenu == i);
        if (active || (hov && m_lmbDown)) {
            m_uiDrawList.addRectFilled(tr, pal.titleActive);
            m_uiDrawList.addText(fn, titles[i],
                {mx + iPad, barY + (barH - lh) * 0.5f}, pal.titleText);
        } else {
            m_uiDrawList.addText(fn, titles[i],
                {mx + iPad, barY + (barH - lh) * 0.5f}, pal.text);
        }

        if (active) {
            m_dropdownX = tr.minX;
            m_dropdownY = barY + barH;
        }
        mx += tr.width();
    }
}

// ─── Dropdown (drawn last so it layers above everything) ──────────────────────

void RetroDemoApp::drawMenuDropdown(float s) {
    if (m_openMenu < 0) return;
    const char** items = currentMenuItems(m_openMenu);
    if (!items) return;

    const ThemePalette& pal = palette();
    const Font& fn  = themeFont();
    const float lh  = fn.lineHeightPx();
    const float t   = std::max(1.f, std::round(s));
    const float iPad = 10.f * s;
    const float itemH = lh + 6.f * s;
    const float sepH  = 8.f * s;

    // Width from widest item
    float dropW = 120.f * s;
    for (const char** p = items; *p; ++p) {
        if (**p != '-')
            dropW = std::max(dropW, fn.measureText(*p) + 2.f * iPad);
    }
    dropW = std::round(dropW);

    // Height
    float totalH = 4.f * s;
    for (const char** p = items; *p; ++p)
        totalH += (**p == '-') ? sepH : itemH;
    totalH += 4.f * s;

    const UiRect panel = UiRect::fromXYWH(m_dropdownX, m_dropdownY, dropW, totalH);

    if (m_theme == Theme::ClassicMac) {
        m_uiDrawList.addRoundRectFilled(panel, {1,1,1,1}, kMacR * s);
        m_uiDrawList.addRoundRect(panel, {0,0,0,1}, kMacR * s, t);
    } else {
        drawBevelPanel(panel, true, s);
    }

    float iy = m_dropdownY + 4.f * s;
    for (const char** p = items; *p; ++p) {
        if (**p == '-') {
            const float sy = std::round(iy + sepH * 0.5f);
            if (m_theme == Theme::ClassicMac) {
                m_uiDrawList.addRectFilled(
                    {m_dropdownX + 2.f*s, sy, m_dropdownX + dropW - 2.f*s, sy + t},
                    {0,0,0,1});
            } else {
                // Win95 / Motif: two-line sunken separator
                m_uiDrawList.addRectFilled(
                    {m_dropdownX+6.f*s, sy-t, m_dropdownX+dropW-6.f*s, sy  },
                    {0.502f,0.502f,0.502f,1.f});
                m_uiDrawList.addRectFilled(
                    {m_dropdownX+6.f*s, sy,   m_dropdownX+dropW-6.f*s, sy+t},
                    {1.f,1.f,1.f,1.f});
            }
            iy += sepH;
        } else {
            const UiRect ir = UiRect::fromXYWH(
                m_dropdownX + 2.f*s, iy, dropW - 4.f*s, itemH);
            const bool hov = isHovered(ir);

            if (hov && m_justClicked) {
                m_openMenu = -1;
                m_clickHandled = true;
            }
            if (hov) {
                if (m_theme == Theme::ClassicMac) {
                    m_uiDrawList.addRoundRectFilled(ir, {0,0,0,1}, 2.f*s);
                } else {
                    m_uiDrawList.addRectFilled(ir, pal.titleActive);
                }
                m_uiDrawList.addText(fn, *p,
                    {ir.minX + iPad * 0.5f, iy + (itemH - lh) * 0.5f},
                    (m_theme == Theme::ClassicMac) ? UiColor{1,1,1,1} : pal.titleText);
            } else {
                m_uiDrawList.addText(fn, *p,
                    {ir.minX + iPad * 0.5f, iy + (itemH - lh) * 0.5f}, pal.text);
            }
            iy += itemH;
        }
    }
}

// ─── Init ─────────────────────────────────────────────────────────────────────

bool RetroDemoApp::onInit() {
    const float s = contentScale();
    if (!loadFonts(
            resolveAssetPath("assets/fonts/Inter-Regular.ttf"),
            resolveAssetPath("assets/fonts/Inter-Bold.ttf"),
            resolveAssetPath("assets/fonts/Inter-Italic.ttf"),
            resolveAssetPath("assets/fonts/Inter-Regular.ttf"),
            std::round(14.f * s), std::round(13.f * s)))
        return false;

    if (m_macFont.loadFromFile(
            resolveAssetPath("assets/fonts/ChicagoKare-Regular.ttf"),
            std::round(13.f * s))) {
        const auto id = m_renderer.registerUiFontAtlas(
            m_macFont.atlasPixels().data(),
            m_macFont.atlasWidth(), m_macFont.atlasHeight());
        m_macFont.setTextureId(id);
    }

    // Retro-OS styleguide heading sizes (bold Inter at three scales).
    auto loadHeading = [&](Font& f, float px) {
        if (f.loadFromFile(resolveAssetPath("assets/fonts/Inter-Bold.ttf"),
                           std::round(px * s))) {
            const auto id = m_renderer.registerUiFontAtlas(
                f.atlasPixels().data(), f.atlasWidth(), f.atlasHeight());
            f.setTextureId(id);
        }
    };
    loadHeading(m_osH1, 34.f);
    loadHeading(m_osH2, 22.f);
    loadHeading(m_osH3, 16.f);
    return true;
}

// ─── Input (mouse + keyboard) ─────────────────────────────────────────────────

void RetroDemoApp::onTick(float dt) {
    m_time += dt;

    // Keyboard — switch themes, close any open menu on switch
    auto key = [&](int k){ return glfwGetKey(m_window, k) == GLFW_PRESS; };
    const bool k1 = key(GLFW_KEY_1), k2 = key(GLFW_KEY_2), k3 = key(GLFW_KEY_3),
               k4 = key(GLFW_KEY_4), k5 = key(GLFW_KEY_5);
    if (k1 && !m_prevKey1) { m_theme = Theme::Win95;      m_openMenu = -1; m_macWinPosInit = false; }
    if (k2 && !m_prevKey2) { m_theme = Theme::Motif;      m_openMenu = -1; m_macWinPosInit = false; }
    if (k3 && !m_prevKey3) { m_theme = Theme::ClassicMac; m_openMenu = -1; m_macWinPosInit = false; }
    if (k4 && !m_prevKey4) { m_theme = Theme::FlatRetro;  m_openMenu = -1; m_macWinPosInit = false; }
    if (k5 && !m_prevKey5) { m_theme = Theme::RetroOS;    m_openMenu = -1; m_macWinPosInit = false; }
    m_prevKey1 = k1; m_prevKey2 = k2; m_prevKey3 = k3; m_prevKey4 = k4; m_prevKey5 = k5;

    // Mouse — convert GLFW logical coords to framebuffer coords
    double cx, cy;
    glfwGetCursorPos(m_window, &cx, &cy);
    const float sc = contentScale();
    m_mouseX = static_cast<float>(cx) * sc;
    m_mouseY = static_cast<float>(cy) * sc;

    m_lmbWasDown   = m_lmbDown;
    m_lmbDown      = (glfwGetMouseButton(m_window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS);
    m_justClicked  = m_lmbWasDown && !m_lmbDown;
    m_clickHandled = false;
}

// ─── Mac Classic top menu bar ─────────────────────────────────────────────────

void RetroDemoApp::drawMenuBar(float fw, float s) {
    const float mbH = std::round(20.f * s);
    const Font& fn  = themeFont();
    const float lh  = fn.lineHeightPx();

    m_uiDrawList.addRectFilled(UiRect::fromXYWH(0, 0, fw, mbH), {1,1,1,1});
    m_uiDrawList.addRectFilled(UiRect::fromXYWH(0, mbH - s, fw, s), {0,0,0,1});

    // Apple placeholder (non-interactive circle)
    m_uiDrawList.addCircleFilled({14.f*s, mbH*0.5f}, 5.f*s, {0,0,0,1});

    // Interactive menu titles (leave room for clock on the right)
    const char* clock = "12:00 PM";
    const float clockW = fn.measureText(clock) + 20.f * s;
    drawMenuStrip(28.f*s, 0.f, fw - 28.f*s - clockW, mbH, kTitlesMac, 5, s);

    // Clock (non-interactive)
    const float cw = fn.measureText(clock);
    m_uiDrawList.addText(fn, clock, {fw - cw - 10.f*s, (mbH - lh)*0.5f}, {0,0,0,1});
}

// ─── Win95 / Motif taskbar ────────────────────────────────────────────────────

void RetroDemoApp::drawTaskbar(float fw, float fh, float s) {
    const ThemePalette& pal = palette();
    const float tbH = std::round(30.f * s);
    const float tbY = fh - tbH;
    drawBevelPanel(UiRect::fromXYWH(0, tbY, fw, tbH), true, s);

    const float sbH = std::round(22.f*s), sbW = std::round(68.f*s);
    const UiRect sb = UiRect::fromXYWH(
        std::round(4.f*s), std::round(tbY + (tbH-sbH)*0.5f), sbW, sbH);
    drawButton(sb, "Start", s);

    const float chW = std::round(180.f*s), chH = std::round(22.f*s);
    const UiRect ch = UiRect::fromXYWH(
        std::round(78.f*s), std::round(tbY + (tbH-chH)*0.5f), chW, chH);
    drawBevelPanel(ch, false, s);
    const float lh = m_uiFont.lineHeightPx();
    m_uiDrawList.addText(m_uiFont, "Display Properties",
        {ch.minX + 6.f*s, ch.minY + (chH - lh)*0.5f}, pal.text);

    const char* clock = "12:00 PM";
    const float cw = m_uiFont.measureText(clock);
    m_uiDrawList.addText(m_uiFont, clock,
        {fw - cw - 10.f*s, tbY + (tbH - lh)*0.5f}, pal.text);
}

// ─── Dialog window ────────────────────────────────────────────────────────────

void RetroDemoApp::drawWindow(float wx, float wy, float ww, float wh, float s) {
    const ThemePalette& pal  = palette();
    const bool isMac  = (m_theme == Theme::ClassicMac);
    const Font& fn    = themeFont();
    const Font& fnB   = themeFontBold();
    const float lh    = fn.lineHeightPx();
    const float lhB   = fnB.lineHeightPx();
    const float t     = std::max(1.f, std::round(s));
    const float macR  = kMacR * s;

    const float titleH = isMac ? std::round(20.f*s) : std::round(22.f*s);
    const float menuH  = isMac ? 0.f : std::round(20.f*s);
    const float border = isMac ? t   : std::round(3.f*s);
    const float padX   = std::round(10.f*s);
    const float padY   = std::round(8.f*s);

    // Mac drop shadow
    if (isMac) {
        const float sh = std::round(3.f*s);
        m_uiDrawList.addRoundRectFilled(
            UiRect::fromXYWH(wx+sh, wy+sh, ww, wh), {0,0,0,1}, macR);
    }

    // Window frame
    if (isMac) {
        m_uiDrawList.addRoundRectFilled(UiRect::fromXYWH(wx,wy,ww,wh), {1,1,1,1}, macR);
        m_uiDrawList.addRoundRect(UiRect::fromXYWH(wx,wy,ww,wh), {0,0,0,1}, macR, t);
    } else {
        drawBevelPanel(UiRect::fromXYWH(wx,wy,ww,wh), true, s);
    }

    // Title bar
    const UiRect titleRect = UiRect::fromXYWH(
        wx+border, wy+border, ww-2.f*border, titleH);

    if (isMac) {
        m_uiDrawList.addRectFilled(titleRect, {1,1,1,1});
        for (float sy = titleRect.minY+t; sy+t <= titleRect.maxY-t; sy += 2.f*t)
            m_uiDrawList.addRectFilled({titleRect.minX+t, sy, titleRect.maxX-t, sy+t}, {0,0,0,0.75f});
        m_uiDrawList.addRectFilled({wx+t, wy+t+titleH, wx+ww-t, wy+t+titleH+t}, {0,0,0,1});

        // Close box
        const float cbSz = std::round(12.f*s);
        const float cbY  = titleRect.minY + (titleH-cbSz)*0.5f;
        const UiRect closeBox = UiRect::fromXYWH(titleRect.minX+8.f*s, cbY, cbSz, cbSz);
        drawButton(closeBox, "", s);  // interactive: inverts on press

        // Zoom box
        const UiRect zoomBox = UiRect::fromXYWH(
            titleRect.maxX-8.f*s-cbSz, cbY, cbSz, cbSz);
        const bool zDn = isPressed(zoomBox);
        if (zDn) {
            m_uiDrawList.addRoundRectFilled(zoomBox, {0,0,0,1}, 2.f*s);
        } else {
            m_uiDrawList.addRoundRectFilled(zoomBox, {1,1,1,1}, 2.f*s);
            m_uiDrawList.addRoundRect(zoomBox, {0,0,0,1}, 2.f*s, t);
        }
        // Inner box indicator
        const float in2 = std::round(3.f*s);
        m_uiDrawList.addRoundRect(
            UiRect::fromXYWH(zoomBox.minX+in2, zoomBox.minY+in2, cbSz-2.f*in2, cbSz-2.f*in2),
            zDn ? UiColor{1,1,1,1} : UiColor{0,0,0,1}, 1.f*s, t);

        const float tw = fnB.measureText("Display Properties");
        m_uiDrawList.addText(fnB, "Display Properties",
            {titleRect.minX + (titleRect.width()-tw)*0.5f,
             titleRect.minY + (titleH-lhB)*0.5f}, {0,0,0,1});

    } else {
        m_uiDrawList.addRectFilled(titleRect, pal.titleActive);
        m_uiDrawList.addText(fnB, "Display Properties",
            {titleRect.minX+6.f*s, titleRect.minY+(titleH-lhB)*0.5f}, pal.titleText);

        // Control buttons (X, □, _) — interactive
        const float cbSz = std::round(18.f*s);
        float cbX = titleRect.maxX - cbSz - std::round(2.f*s);
        for (const char* lbl : {"X", "\xe2\x96\xa1", "_"}) {
            const UiRect cb = UiRect::fromXYWH(
                cbX, titleRect.minY+(titleH-cbSz)*0.5f, cbSz, cbSz);
            drawButton(cb, lbl, s);
            cbX -= cbSz + std::round(2.f*s);
        }

        // Window menu bar (interactive) — Win95/Motif only
        const float menuY = wy + border + titleH;
        const UiRect menuRect = UiRect::fromXYWH(
            wx+border, menuY, ww-2.f*border, menuH);
        m_uiDrawList.addRectFilled(menuRect, pal.face);
        drawMenuStrip(menuRect.minX + 4.f*s, menuRect.minY,
                      menuRect.width() - 4.f*s, menuRect.height(),
                      kTitlesWin95, 5, s);
    }

    // Client area
    const float clientY = wy + border + titleH + t + menuH;
    const float btnsH   = std::round(26.f*s) + 2.f*padY;
    const float clientH = wh - (border + titleH + t + menuH) - btnsH - border;
    const UiRect clientRect = UiRect::fromXYWH(wx+border, clientY, ww-2.f*border, clientH);
    m_uiDrawList.addRectFilled(clientRect, pal.face);

    // Group box
    const float gbX = clientRect.minX + padX;
    const float gbY = clientRect.minY + padY + lh*0.5f;
    const float gbW = clientRect.width() - 2.f*padX;
    const float gbH = std::round(140.f*s);
    const char* gbLabel = " Appearance ";
    const float glw = fn.measureText(gbLabel);
    const float gapX = gbX + 10.f*s;
    drawGroupBorder(gbX, gbY, gbW, gbH, gapX, glw, s);
    m_uiDrawList.addRectFilled(
        UiRect::fromXYWH(gapX-2.f, gbY-lh*0.5f-1.f, glw+4.f, lh+2.f), pal.face);
    m_uiDrawList.addText(fn, gbLabel, {gapX, gbY-lh*0.5f}, pal.text);

    // Group content rows
    const float rowX = gbX + 12.f*s;
    const float valX = gbX + gbW * 0.46f;
    float ry = gbY + lh*0.4f + 6.f*s;
    auto row = [&](const char* key, const char* val) {
        m_uiDrawList.addText(fn,  key, {rowX, ry}, pal.textDim);
        m_uiDrawList.addText(fnB, val, {valX, ry}, pal.text);
        ry += lh + std::round(5.f*s);
    };
    row("Color Scheme:", isMac ? "Mac Standard" : "Windows Standard");
    row("Desktop:",      pal.name);
    row("Icon Size:",    "32 \xc3\x97 32 pixels");

    ry += std::round(4.f*s);
    m_uiDrawList.addText(fn, "Palette:", {rowX, ry}, pal.textDim);
    static const UiColor kSwatches[6] = {
        {.80f,.20f,.20f,1},{.20f,.70f,.30f,1},{.20f,.30f,.80f,1},
        {.80f,.78f,.20f,1},{.70f,.28f,.70f,1},{.22f,.75f,.80f,1},
    };
    const float swSz = std::round(16.f*s);
    float swX = valX;
    for (const UiColor& c : kSwatches) {
        drawBevelPanel(UiRect::fromXYWH(swX, ry, swSz, swSz), true, s);
        m_uiDrawList.addRectFilled(
            UiRect::fromXYWH(swX+2.f*s, ry+2.f*s, swSz-4.f*s, swSz-4.f*s), c);
        swX += swSz + std::round(4.f*s);
    }

    // Progress bar
    const float prY  = gbY + gbH + padY;
    const float prH  = std::round(14.f*s);
    m_uiDrawList.addText(fn, "Loading:",
        {rowX, prY + (prH-lh)*0.5f}, pal.textDim);
    const float prX  = valX;
    const float prW  = gbW - (valX-gbX) - padX - std::round(38.f*s);
    const UiRect prRect = UiRect::fromXYWH(prX, prY, prW, prH);
    drawBevelPanel(prRect, false, s);
    const float fillFrac = 0.55f + 0.45f*(std::sin(m_time*0.65f)*0.5f+0.5f);
    const float segW = std::round(7.f*s), segG = std::round(2.f*s);
    for (float px2 = prX+2.f*s; px2+segW <= prX+2.f*s+(prW-4.f*s)*fillFrac; px2 += segW+segG)
        m_uiDrawList.addRectFilled(UiRect::fromXYWH(px2, prY+2.f*s, segW, prH-4.f*s), pal.highlight);
    char pctBuf[8];
    std::snprintf(pctBuf, sizeof(pctBuf), "%d%%", static_cast<int>(fillFrac*100.f));
    m_uiDrawList.addText(fn, pctBuf, {prX+prW+5.f*s, prY+(prH-lh)*0.5f}, pal.text);

    // OK / Cancel buttons
    const float btnW = std::round(80.f*s), btnH = std::round(26.f*s);
    const float btnsY = clientY + clientH - btnH - padY;
    const float cx    = wx + ww*0.5f;
    for (int i = 0; i < 2; ++i) {
        const char* lbl = (i == 0) ? "OK" : "Cancel";
        const float bx  = (i == 0) ? cx-btnW-6.f*s : cx+6.f*s;
        const UiRect br = UiRect::fromXYWH(std::round(bx), std::round(btnsY), btnW, btnH);
        drawButton(br, lbl, s);
        if (isMac && i == 0) {
            // Default-button concentric ring
            const float gap = std::round(4.f*s);
            const UiRect ring = UiRect::fromXYWH(
                br.minX-gap, br.minY-gap, btnW+2.f*gap, btnH+2.f*gap);
            m_uiDrawList.addRoundRect(ring, {0,0,0,1}, macR+gap*0.5f, t);
        }
    }
}

// ─── Render ───────────────────────────────────────────────────────────────────

void RetroDemoApp::onRender(float /*dt*/) {
    beginFrameDraw();

    const ThemePalette& pal  = palette();
    const bool          isMac  = (m_theme == Theme::ClassicMac);
    const bool          isFlat = (m_theme == Theme::FlatRetro);
    const bool          isOs   = (m_theme == Theme::RetroOS);
    const Font&         fn   = themeFont();
    const float s = contentScale();
    int fwi, fhi;
    framebufferSize(fwi, fhi);
    const float fw = static_cast<float>(fwi);
    const float fh = static_cast<float>(fhi);

    m_uiDrawList.addRectFilled(UiRect::fromXYWH(0, 0, fw, fh), pal.desktop);

    const float tbH = std::round(30.f*s);

    if (isMac) {
        drawMenuBar(fw, s);
        drawMacInterface(fw, fh, s);
    } else if (isFlat) {
        drawFlatInterface(fw, fh, s);
    } else if (isOs) {
        drawRetroOsInterface(fw, fh, s);
    } else {
        drawTaskbar(fw, fh, s);
        const float winW = std::round(480.f*s);
        const float winH = std::round(400.f*s);
        const float winX = std::round((fw-winW)*0.5f);
        const float winY = std::round((fh-tbH-winH)*0.5f);
        drawWindow(winX, winY, winW, winH, s);
    }

    // Hint text
    {
        char buf[160];
        std::snprintf(buf, sizeof(buf),
            "[1] Windows 95   [2] Motif/CDE   [3] Mac Classic   [4] Flat Retro   "
            "[5] Retro-OS   \xe2\x80\x94   %s", pal.name);
        const float tw = fn.measureText(buf);
        const float lh = fn.lineHeightPx();
        const float ty = (isMac || isFlat || isOs) ? fh-lh-8.f*s : fh-tbH-lh-6.f*s;
        const UiColor hintCol = (isFlat || isOs) ? UiColor{0.149f,0.149f,0.180f,0.65f}
                                                 : UiColor{1,1,1, isMac ? 0.5f : 0.75f};
        m_uiDrawList.addText(fn, buf, {std::round((fw-tw)*0.5f), ty}, hintCol);
    }

    // ── Dropdown drawn LAST so it always layers above all content ─────────────
    drawMenuDropdown(s);

    // Unhandled click when a menu is open → close it
    if (m_justClicked && !m_clickHandled && m_openMenu >= 0)
        m_openMenu = -1;

    submitFrame(m_camera);
}

// ─── Mac Classic: folder icon ─────────────────────────────────────────────────

void RetroDemoApp::drawMacFolderIcon(float x, float y, const char* label, float s) {
    const float t   = std::max(1.f, std::round(s));
    const Font& fn  = themeFont();
    const float iw  = std::round(32.f*s), ibH = std::round(22.f*s);
    const float tabW = std::round(14.f*s), tabH = std::round(4.f*s);

    // Tab
    m_uiDrawList.addRectFilled(UiRect::fromXYWH(x, y, tabW, tabH+t), {1,1,1,1});
    m_uiDrawList.addRectFilled({x,     y,      x+tabW,   y+t    }, {0,0,0,1});
    m_uiDrawList.addRectFilled({x,     y,      x+t,      y+tabH }, {0,0,0,1});
    m_uiDrawList.addRectFilled({x+tabW, y,     x+tabW+t, y+tabH+t}, {0,0,0,1});
    // Body
    m_uiDrawList.addRectFilled(UiRect::fromXYWH(x, y+tabH, iw, ibH), {1,1,1,1});
    m_uiDrawList.addRectFilled({x,    y+tabH,    x+t,  y+tabH+ibH}, {0,0,0,1});
    m_uiDrawList.addRectFilled({x,    y+tabH,    x+iw, y+tabH+t  }, {0,0,0,1});
    m_uiDrawList.addRectFilled({x,    y+tabH+ibH-t, x+iw, y+tabH+ibH}, {0,0,0,1});
    m_uiDrawList.addRectFilled({x+iw-t, y+tabH, x+iw, y+tabH+ibH}, {0,0,0,1});

    const float tw = fn.measureText(label);
    m_uiDrawList.addText(fn, label,
        {x + (iw-tw)*0.5f, y+tabH+ibH+2.f*s}, {0,0,0,1});
}

// ─── Mac Classic: document icon ───────────────────────────────────────────────

void RetroDemoApp::drawMacDocIcon(float x, float y, const char* label, float s) {
    const float t    = std::max(1.f, std::round(s));
    const Font& fn   = themeFont();
    const float iw   = std::round(26.f*s), ih = std::round(32.f*s);
    const float fold = std::round(8.f*s);

    // White body (split around dog-ear)
    m_uiDrawList.addRectFilled(UiRect::fromXYWH(x,        y,       iw-fold, ih  ), {1,1,1,1});
    m_uiDrawList.addRectFilled(UiRect::fromXYWH(x,        y+fold,  iw,      ih-fold), {1,1,1,1});
    // Dog-ear triangle (gray fill, framed)
    m_uiDrawList.addRectFilled(UiRect::fromXYWH(x+iw-fold, y, fold, fold), {0.82f,0.82f,0.82f,1.f});
    // Borders
    m_uiDrawList.addRectFilled({x,          y,       x+t,        y+ih       }, {0,0,0,1});
    m_uiDrawList.addRectFilled({x,          y,       x+iw-fold,  y+t        }, {0,0,0,1});
    m_uiDrawList.addRectFilled({x,          y+ih-t,  x+iw,       y+ih       }, {0,0,0,1});
    m_uiDrawList.addRectFilled({x+iw-t,     y+fold,  x+iw,       y+ih       }, {0,0,0,1});
    m_uiDrawList.addRectFilled({x+iw-fold,  y,       x+iw-fold+t,y+fold     }, {0,0,0,1});
    m_uiDrawList.addRectFilled({x+iw-fold,  y+fold-t,x+iw,       y+fold     }, {0,0,0,1});
    // Interior lines (document content suggestion)
    for (int i = 0; i < 3; ++i) {
        const float ly = y + fold + std::round((4+i*5)*s);
        const float lw = (i < 2) ? iw-5.f*s : (iw-5.f*s)*0.6f;
        m_uiDrawList.addRectFilled({x+3.f*s, ly, x+3.f*s+lw, ly+t}, {0.72f,0.72f,0.72f,1.f});
    }

    const float tw = fn.measureText(label);
    m_uiDrawList.addText(fn, label, {x+(iw-tw)*0.5f, y+ih+2.f*s}, {0,0,0,1});
}

// ─── Mac Classic: Trash icon ──────────────────────────────────────────────────

void RetroDemoApp::drawMacTrash(float x, float y, float s) {
    const float t  = std::max(1.f, std::round(s));
    const Font& fn = themeFont();
    const float iw = std::round(26.f*s), ih = std::round(30.f*s);
    const float lidH = std::round(5.f*s);
    const float bodyY = y + lidH + 2.f*s;
    const float bodyH = ih - lidH - 2.f*s;

    // Body (white fill, black border, slightly tapered)
    m_uiDrawList.addRectFilled(UiRect::fromXYWH(x, bodyY, iw, bodyH), {1,1,1,1});
    m_uiDrawList.addRectFilled({x,    bodyY,          x+t,  bodyY+bodyH}, {0,0,0,1});
    m_uiDrawList.addRectFilled({x+iw-t, bodyY,        x+iw, bodyY+bodyH}, {0,0,0,1});
    m_uiDrawList.addRectFilled({x,    bodyY+bodyH-t,  x+iw, bodyY+bodyH}, {0,0,0,1});
    // Vertical ribs
    for (float rx = x + iw*0.35f; rx < x+iw*0.7f; rx += iw*0.32f)
        m_uiDrawList.addRectFilled({std::round(rx), bodyY+3.f*s, std::round(rx)+t, bodyY+bodyH-3.f*s}, {0,0,0,1});

    // Lid (wider than body)
    const float lidX = x - 2.f*s;
    m_uiDrawList.addRectFilled(UiRect::fromXYWH(lidX, y, iw+4.f*s, lidH), {1,1,1,1});
    m_uiDrawList.addRectFilled({lidX,       y,      lidX+iw+4.f*s, y+t    }, {0,0,0,1});
    m_uiDrawList.addRectFilled({lidX,       y,      lidX+t,        y+lidH }, {0,0,0,1});
    m_uiDrawList.addRectFilled({lidX+iw+4.f*s-t, y, lidX+iw+4.f*s, y+lidH}, {0,0,0,1});
    m_uiDrawList.addRectFilled({lidX,       y+lidH-t, lidX+iw+4.f*s, y+lidH}, {0,0,0,1});
    // Handle on lid
    const float hx = x + iw*0.35f;
    m_uiDrawList.addRectFilled(UiRect::fromXYWH(hx, y-3.f*s, iw*0.3f, 4.f*s), {1,1,1,1});
    m_uiDrawList.addRectFilled({hx,           y-3.f*s, hx+t,           y+t    }, {0,0,0,1});
    m_uiDrawList.addRectFilled({hx+iw*0.3f-t, y-3.f*s, hx+iw*0.3f,    y+t    }, {0,0,0,1});
    m_uiDrawList.addRectFilled({hx,           y-3.f*s, hx+iw*0.3f, y-3.f*s+t  }, {0,0,0,1});

    const float tw = fn.measureText("Trash");
    m_uiDrawList.addText(fn, "Trash", {x+(iw-tw)*0.5f, bodyY+bodyH+2.f*s}, {0,0,0,1});
}

// ─── Mac Classic: Finder window ───────────────────────────────────────────────
// iconType: 0=none  1=folder icons (Mac System Software)  2=doc icons (System Folder)

void RetroDemoApp::drawMacFinderWindow(float wx, float wy, float ww, float wh,
                                        const char* title, const char* infoLine,
                                        int iconType, bool isActive, float s) {
    const Font& fn  = themeFont();
    const float lh  = fn.lineHeightPx();
    const float t   = std::max(1.f, std::round(s));
    const float r   = kMacR * s;

    const float titleH = std::round(19.f*s);
    const float infoH  = std::round(17.f*s);
    const float sbW    = std::round(15.f*s);
    const float cbSz   = std::round(11.f*s);

    // ── Window frame ──────────────────────────────────────────────────────────
    m_uiDrawList.addRoundRectFilled(UiRect::fromXYWH(wx,wy,ww,wh), {1,1,1,1}, r);
    m_uiDrawList.addRoundRect(UiRect::fromXYWH(wx,wy,ww,wh), {0,0,0,1}, r, t);

    // ── Title bar ─────────────────────────────────────────────────────────────
    const UiRect tb = UiRect::fromXYWH(wx+t, wy+t, ww-2.f*t, titleH);
    m_uiDrawList.addRectFilled(tb, {1,1,1,1});
    if (isActive) {
        for (float sy = tb.minY+t; sy+t <= tb.maxY-t; sy += 2.f*t)
            m_uiDrawList.addRectFilled({tb.minX, sy, tb.maxX, sy+t}, {0,0,0,0.85f});
    }

    // Title text — erase stripes behind it so it's legible
    const float ttw = fn.measureText(title);
    const float ttx = wx + (ww-ttw)*0.5f;
    const float tty = wy + t + (titleH-lh)*0.5f;
    if (isActive) {
        m_uiDrawList.addRectFilled(
            UiRect::fromXYWH(ttx-5.f*s, wy+t, ttw+10.f*s, titleH), {1,1,1,1});
    }
    m_uiDrawList.addText(fn, title, {ttx, tty}, {0,0,0,1});

    // Close box (left) — interactive when active
    const float cbY = wy + t + (titleH-cbSz)*0.5f;
    const UiRect closeBox = UiRect::fromXYWH(wx+t+6.f*s, cbY, cbSz, cbSz);
    if (isActive) {
        drawButton(closeBox, "", s);
    } else {
        // Inactive: plain close box outline only
        m_uiDrawList.addRoundRectFilled(closeBox, {1,1,1,1}, 2.f*s);
        m_uiDrawList.addRoundRect(closeBox, {0,0,0,1}, 2.f*s, t);
    }

    // Zoom box (right of title)
    const UiRect zoomBox = UiRect::fromXYWH(
        wx+ww-t-6.f*s-cbSz, cbY, cbSz, cbSz);
    const bool zDn = isActive && isPressed(zoomBox);
    if (isActive) {
        if (zDn) {
            m_uiDrawList.addRoundRectFilled(zoomBox, {0,0,0,1}, 2.f*s);
        } else {
            m_uiDrawList.addRoundRectFilled(zoomBox, {1,1,1,1}, 2.f*s);
            m_uiDrawList.addRoundRect(zoomBox, {0,0,0,1}, 2.f*s, t);
        }
        const float iz = std::round(3.f*s);
        m_uiDrawList.addRoundRect(
            UiRect::fromXYWH(zoomBox.minX+iz, zoomBox.minY+iz, cbSz-2.f*iz, cbSz-2.f*iz),
            zDn ? UiColor{1,1,1,1} : UiColor{0,0,0,1}, 1.f*s, t);
    }

    // Separator: title / info
    m_uiDrawList.addRectFilled({wx+t, wy+t+titleH, wx+ww-t, wy+t+titleH+t}, {0,0,0,1});

    // ── Info bar ──────────────────────────────────────────────────────────────
    const float infoY = wy + t + titleH + t;
    m_uiDrawList.addRectFilled(UiRect::fromXYWH(wx+t, infoY, ww-2.f*t, infoH), {1,1,1,1});
    const float iw2 = fn.measureText(infoLine);
    m_uiDrawList.addText(fn, infoLine,
        {wx + (ww-iw2)*0.5f, infoY + (infoH-lh)*0.5f}, {0,0,0,1});
    m_uiDrawList.addRectFilled({wx+t, infoY+infoH, wx+ww-t, infoY+infoH+t}, {0,0,0,1});

    // ── Content + scrollbar layout ────────────────────────────────────────────
    const float contentY = infoY + infoH + t;
    const float contentH = wh - (contentY-wy) - t;
    const float contentW = ww - sbW - t - t;
    const UiRect content = UiRect::fromXYWH(wx+t, contentY, contentW, contentH);
    m_uiDrawList.addRectFilled(content, {1,1,1,1});

    // Vertical divider between content and scroll bar
    m_uiDrawList.addRectFilled({content.maxX, contentY, content.maxX+t, contentY+contentH}, {0,0,0,1});

    // ── Icon grid ─────────────────────────────────────────────────────────────
    if (iconType == 1) {
        // Folder icons (Mac System Software window)
        const float iconY = contentY + std::round(14.f*s);
        const float iconX = content.minX + std::round(20.f*s);
        const float stride = std::round(68.f*s);
        const char* labels[] = {"System Folder", "Empty Folder", nullptr};
        for (int i = 0; labels[i]; ++i)
            drawMacFolderIcon(iconX + i*stride, iconY, labels[i], s);
        // One file icon
        drawMacDocIcon(iconX + 2.f*stride, iconY + 2.f*s, "SysVersion", s);
    } else if (iconType == 2) {
        // Doc icons in a row (System Folder window)
        const float iconY = contentY + contentH - std::round(52.f*s) - lh;
        float iconX = content.minX + std::round(16.f*s);
        const char* labels[] = {"Finder","System","Imagewriter",
                                 "Note Pad File","Scrapbook File","Clipboard File",nullptr};
        const float stride = std::round(60.f*s);
        for (int i = 0; labels[i] && iconX + 26.f*s < content.maxX - 4.f*s; ++i) {
            drawMacDocIcon(iconX, iconY, labels[i], s);
            iconX += stride;
        }
    }

    // ── Scroll bar ────────────────────────────────────────────────────────────
    const float sbX = content.maxX + t;
    const float sbH = contentH - sbW;  // leave space for grow box
    const float sbArrowH = sbW;

    // Track background (gray)
    m_uiDrawList.addRectFilled(UiRect::fromXYWH(sbX, contentY+sbArrowH, sbW, sbH-2.f*sbArrowH),
        {0.80f,0.80f,0.80f,1.f});

    // Up arrow button
    const UiRect upBtn = UiRect::fromXYWH(sbX, contentY, sbW, sbArrowH);
    drawBevelPanel(upBtn, !isPressed(upBtn), s);
    {   // Up triangle (rows widening from top)
        const float cx = upBtn.minX + sbW*0.5f, cy = upBtn.minY + sbArrowH*0.5f;
        for (int i = 0; i < 4; ++i) {
            const float hw = (i+1)*s;
            m_uiDrawList.addRectFilled({cx-hw, cy-(3-i)*t, cx+hw, cy-(2-i)*t}, {0,0,0,1});
        }
    }

    // Down arrow button
    const UiRect dnBtn = UiRect::fromXYWH(sbX, contentY+sbH-sbArrowH, sbW, sbArrowH);
    drawBevelPanel(dnBtn, !isPressed(dnBtn), s);
    {   // Down triangle (rows widening downward)
        const float cx = dnBtn.minX + sbW*0.5f, cy = dnBtn.minY + sbArrowH*0.5f;
        for (int i = 0; i < 4; ++i) {
            const float hw = (4-i)*s;
            m_uiDrawList.addRectFilled({cx-hw, cy+(i-1)*t, cx+hw, cy+i*t}, {0,0,0,1});
        }
    }

    // Scroll thumb (static, at top of track)
    const float trackY  = contentY + sbArrowH;
    const float trackH2 = sbH - 2.f*sbArrowH;
    const float thumbH  = std::round(std::max(20.f*s, trackH2*0.25f));
    m_uiDrawList.addRectFilled(UiRect::fromXYWH(sbX+t, trackY, sbW-2.f*t, thumbH), {1,1,1,1});
    m_uiDrawList.addRectFilled({sbX+t, trackY,         sbX+sbW-t, trackY+t       }, {0,0,0,1});
    m_uiDrawList.addRectFilled({sbX+t, trackY+thumbH-t,sbX+sbW-t, trackY+thumbH  }, {0,0,0,1});

    // ── Grow box (bottom-right, size-resize handle) ───────────────────────────
    const float gzX = sbX, gzY = contentY + sbH;
    const UiRect gz = UiRect::fromXYWH(gzX, gzY, sbW, sbW);
    m_uiDrawList.addRectFilled(gz, {0.80f,0.80f,0.80f,1.f});
    m_uiDrawList.addRectFilled({gzX, gzY, gzX+t, gzY+sbW}, {0,0,0,1});
    m_uiDrawList.addRectFilled({gzX, gzY, gzX+sbW, gzY+t}, {0,0,0,1});
    // Two overlapping rect outlines — Mac grow-box pattern
    const float p = std::round(2.f*s), q = std::round(6.f*s);
    auto outline = [&](float ox, float oy, float ow, float oh) {
        m_uiDrawList.addRectFilled({ox,    oy,    ox+ow, oy+t  }, {0,0,0,1});
        m_uiDrawList.addRectFilled({ox,    oy,    ox+t,  oy+oh }, {0,0,0,1});
        m_uiDrawList.addRectFilled({ox,    oy+oh-t, ox+ow, oy+oh}, {0,0,0,1});
        m_uiDrawList.addRectFilled({ox+ow-t, oy,  ox+ow, oy+oh }, {0,0,0,1});
    };
    outline(gzX+p,   gzY+q,   sbW-p-p,   sbW-q-p);   // foreground rect
    outline(gzX+q,   gzY+p,   sbW-q-p,   sbW-p-p);   // background rect (shifted)
}

// ─── Mac Classic: full desktop layout ────────────────────────────────────────

void RetroDemoApp::drawMacInterface(float fw, float fh, float s) {
    const float mbH    = std::round(20.f*s);
    const float titleH = std::round(19.f*s);

    // Window sizes (fixed, derived from framebuffer)
    const float ww[2] = { std::round(fw * 0.60f), std::round(fw * 0.74f) };
    const float wh[2] = { std::round(fh * 0.42f), std::round(fh * 0.46f) };

    // Initialise positions once (first Mac frame or after theme switch)
    if (!m_macWinPosInit) {
        m_macWinX[0] = std::round(fw * 0.06f);
        m_macWinY[0] = std::round(mbH + 14.f*s);
        m_macWinX[1] = std::round(fw * 0.04f);
        m_macWinY[1] = std::round(mbH + fh * 0.30f);
        m_macWinPosInit = true;
    }

    // ── Drag: start on title-bar press ───────────────────────────────────────
    const bool justPressed = m_lmbDown && !m_lmbWasDown;
    if (justPressed && m_dragWin < 0) {
        // Test front-to-back (win[1] is the active foreground window)
        for (int i = 1; i >= 0; --i) {
            const UiRect tb = UiRect::fromXYWH(m_macWinX[i], m_macWinY[i], ww[i], titleH);
            if (isHovered(tb)) {
                m_dragWin  = i;
                m_dragOffX = m_mouseX - m_macWinX[i];
                m_dragOffY = m_mouseY - m_macWinY[i];
                m_clickHandled = true;
                break;
            }
        }
    }

    // ── Drag: update or release ───────────────────────────────────────────────
    if (m_dragWin >= 0) {
        if (m_lmbDown) {
            m_macWinX[m_dragWin] = std::round(m_mouseX - m_dragOffX);
            m_macWinY[m_dragWin] = std::round(
                std::max(mbH, m_mouseY - m_dragOffY));   // don't go above menu bar
            m_clickHandled = true;
        } else {
            m_dragWin = -1;
        }
    }

    // ── Draw back-to-front ───────────────────────────────────────────────────
    drawMacFinderWindow(m_macWinX[0], m_macWinY[0], ww[0], wh[0],
                        "Mac System Software",
                        "3 items   227K in disk   173K available",
                        1, false, s);

    drawMacFinderWindow(m_macWinX[1], m_macWinY[1], ww[1], wh[1],
                        "System Folder",
                        "5 items   211K in folder   173K available",
                        2, true, s);

    // Trash icon (desktop, bottom-right)
    const float trW = std::round(26.f*s);
    drawMacTrash(std::round(fw - trW - 18.f*s), std::round(fh - 70.f*s), s);
}

// ════════════════════════════════════════════════════════════════════════════
//  Flat Retro UI kit
// ════════════════════════════════════════════════════════════════════════════

UiRect RetroDemoApp::drawFlatWindow(UiRect r, const char* title,
                                     UiColor accent, float s) {
    const Font& fnB   = m_uiFontBold;
    const float t     = std::max(1.f, std::round(s));
    const float ink   = std::round(2.5f*s);
    const float rad   = kFlatR * s;
    const float titleH = std::round(26.f*s);

    // Cream body
    m_uiDrawList.addRoundRectFilled(r, kFlatRetro.face, rad);

    // Title strip — accent roundrect extended down, then square off the bottom
    m_uiDrawList.addRoundRectFilled(
        {r.minX, r.minY, r.maxX, r.minY+titleH+rad}, accent, rad);
    m_uiDrawList.addRectFilled(
        {r.minX, r.minY+titleH, r.maxX, r.minY+titleH+rad}, kFlatRetro.face);
    // Separator under title
    m_uiDrawList.addRectFilled(
        {r.minX, r.minY+titleH, r.maxX, r.minY+titleH+ink}, kFlatInk);

    const UiRect titleRect = {r.minX, r.minY, r.maxX, r.minY+titleH};

    // Left control squares (cosmetic)
    const float sq = std::round(12.f*s);
    const float sqY = titleRect.minY + (titleH-sq)*0.5f;
    float sqX = r.minX + std::round(8.f*s);
    for (int i = 0; i < 2; ++i) {
        const UiRect b = UiRect::fromXYWH(sqX, sqY, sq, sq);
        m_uiDrawList.addRoundRectFilled(b, {1,1,1,0.92f}, 2.f*s);
        m_uiDrawList.addRoundRect(b, kFlatInk, 2.f*s, t);
        // tiny mark
        m_uiDrawList.addRectFilled(
            {b.minX+3.f*s, (b.minY+b.maxY)*0.5f-t*0.5f, b.maxX-3.f*s,
             (b.minY+b.maxY)*0.5f+t*0.5f}, kFlatInk);
        sqX += sq + std::round(4.f*s);
    }

    // Right control circles (cosmetic)
    const float cr = std::round(4.f*s);
    float ccX = r.maxX - std::round(12.f*s);
    for (int i = 0; i < 3; ++i) {
        m_uiDrawList.addCircle({ccX, titleRect.minY+titleH*0.5f}, cr, {1,1,1,0.95f}, t);
        ccX -= cr*2.f + std::round(5.f*s);
    }

    // Title text (white, centered)
    const float tw = fnB.measureText(title);
    m_uiDrawList.addText(fnB, title,
        {titleRect.minX + (titleRect.width()-tw)*0.5f,
         titleRect.minY + (titleH-fnB.lineHeightPx())*0.5f}, {1,1,1,1});

    // Outline last so it sits on top
    m_uiDrawList.addRoundRect(r, kFlatInk, rad, ink);

    // Inner content rect
    const float pad = std::round(12.f*s);
    return {r.minX+pad, r.minY+titleH+ink+pad, r.maxX-pad, r.maxY-pad};
}

bool RetroDemoApp::drawFlatButton(UiRect r, const char* label, float s) {
    const Font& fnB = m_uiFontBold;
    const float t   = std::max(1.f, std::round(s));
    const float ink = std::round(2.f*s);
    const float rad = std::round(kFlatR*0.75f*s);
    const bool hov  = isHovered(r);
    const bool dn   = isPressed(r);
    const bool hit  = m_justClicked && hov;
    if (hit) m_clickHandled = true;

    const UiColor fill = dn  ? UiColor{0.88f,0.86f,0.80f,1.f}
                       : hov ? UiColor{0.93f,0.91f,0.86f,1.f}
                             : UiColor{0.984f,0.965f,0.914f,1.f};
    m_uiDrawList.addRoundRectFilled(r, fill, rad);
    m_uiDrawList.addRoundRect(r, kFlatInk, rad, ink);

    const float off = dn ? t : 0.f;
    const float tw  = fnB.measureText(label);
    m_uiDrawList.addText(fnB, label,
        {r.minX + (r.width()-tw)*0.5f + off,
         r.minY + (r.height()-fnB.lineHeightPx())*0.5f + off}, kFlatInk);
    return hit;
}

void RetroDemoApp::drawFlatField(UiRect r, bool showLines, float s) {
    const float ink = std::round(2.f*s);
    const float rad = std::round(kFlatR*0.6f*s);
    m_uiDrawList.addRoundRectFilled(r, {1,1,1,1}, rad);
    m_uiDrawList.addRoundRect(r, kFlatInk, rad, ink);
    if (showLines) {
        const float lw = std::round(12.f*s);
        const float lx = r.maxX - lw - std::round(8.f*s);
        const float cy = (r.minY+r.maxY)*0.5f;
        const float g  = std::round(3.f*s);
        for (int i = -1; i <= 1; ++i)
            m_uiDrawList.addRectFilled(
                {lx, cy+i*g-ink*0.5f, lx+lw, cy+i*g+ink*0.5f}, kFlatInk);
    }
}

// ─── Flat icons ───────────────────────────────────────────────────────────────

void RetroDemoApp::drawIconMail(float x, float y, float sz, float s) {
    const float it = std::round(2.f*s);
    const float bh = sz*0.64f;
    const UiRect body = {x, y+sz*0.18f, x+sz, y+sz*0.18f+bh};
    m_uiDrawList.addRoundRectFilled(body, {1,1,1,1}, 3.f*s);

    // Envelope flap as a real curved path: two cubic beziers sweep from the top
    // corners down to a rounded dip at the centre — a shape the rect/SDF API
    // can't express. Filled blue, then outlined in ink.
    const float cx   = (body.minX + body.maxX) * 0.5f;
    const float topY = body.minY + it;
    const float dipY = body.minY + bh * 0.52f;
    VectorPath flap;
    flap.setTessellationTolerancePx(0.3f);
    flap.moveTo(body.minX + it, topY);
    flap.cubicBezierTo(body.minX + it, topY + bh * 0.28f,
                       cx - sz * 0.18f, dipY, cx, dipY);
    flap.cubicBezierTo(cx + sz * 0.18f, dipY,
                       body.maxX - it, topY + bh * 0.28f, body.maxX - it, topY);
    flap.close();
    m_uiDrawList.addPathFilled(flap, kFlatBlue);

    // Stroke the flap's V seam in ink for definition.
    VectorPath seam;
    seam.setTessellationTolerancePx(0.3f);
    seam.moveTo(body.minX + it, topY);
    seam.cubicBezierTo(body.minX + it, topY + bh * 0.28f,
                       cx - sz * 0.18f, dipY, cx, dipY);
    seam.cubicBezierTo(cx + sz * 0.18f, dipY,
                       body.maxX - it, topY + bh * 0.28f, body.maxX - it, topY);
    StrokeOptions so;
    so.widthPx = it;
    so.join = LineJoin::Round;
    so.cap = LineCap::Round;
    m_uiDrawList.addPathStroked(seam, kFlatInk, so);

    m_uiDrawList.addRoundRect(body, kFlatInk, 3.f*s, it);
}

void RetroDemoApp::drawIconPhoto(float x, float y, float sz, float s) {
    const float it = std::round(2.f*s);
    const UiRect body = {x, y+sz*0.16f, x+sz, y+sz*0.84f};
    m_uiDrawList.addRoundRectFilled(body, {1,1,1,1}, 3.f*s);
    // Sky band
    m_uiDrawList.addRectFilled(
        {body.minX+it, body.minY+it, body.maxX-it, body.maxY-sz*0.22f}, kFlatBlue);
    // Sun
    m_uiDrawList.addCircleFilled(
        {body.minX+sz*0.28f, body.minY+sz*0.24f}, sz*0.10f, kFlatOrange);
    // Ground
    m_uiDrawList.addRectFilled(
        {body.minX+it, body.maxY-sz*0.22f, body.maxX-it, body.maxY-it}, kFlatGreen);
    m_uiDrawList.addRoundRect(body, kFlatInk, 3.f*s, it);
}

void RetroDemoApp::drawIconVideo(float x, float y, float sz, float s) {
    const float it = std::round(2.f*s);
    const UiRect body = {x, y+sz*0.18f, x+sz, y+sz*0.82f};
    m_uiDrawList.addRoundRectFilled(body, {0.20f,0.22f,0.30f,1.f}, 3.f*s);
    // Film perforations
    const float pw = sz*0.12f, ph = sz*0.10f, g = sz*0.06f;
    for (float py = body.minY+g; py+ph <= body.maxY-g*0.5f; py += ph+g) {
        m_uiDrawList.addRectFilled({body.minX+it*1.5f, py, body.minX+it*1.5f+pw, py+ph}, {1,1,1,1});
        m_uiDrawList.addRectFilled({body.maxX-it*1.5f-pw, py, body.maxX-it*1.5f, py+ph}, {1,1,1,1});
    }
    // Play wedge (center)
    m_uiDrawList.addSectorFilled(
        {x+sz*0.5f, body.minY+body.height()*0.5f}, 0.f, sz*0.17f,
        -0.6f, 0.6f, {1,1,1,1}, 10);
    m_uiDrawList.addRoundRect(body, kFlatInk, 3.f*s, it);
}

void RetroDemoApp::drawIconInternet(float x, float y, float sz, float s) {
    const float it  = std::round(2.f*s);
    const UiVec2 c  = {x+sz*0.5f, y+sz*0.5f};
    const float rad = sz*0.40f;
    m_uiDrawList.addCircleFilled(c, rad, kFlatBlue);

    // Globe graticule drawn with real elliptical arcs (curved meridians and
    // latitude rings), not the straight rects the SDF path was limited to.
    const UiColor line{1, 1, 1, 0.9f};
    StrokeOptions so;
    so.widthPx = it;
    so.join = LineJoin::Round;
    so.cap = LineCap::Round;

    // Two latitude rings (horizontal ovals) + the equator.
    VectorPath equator;  equator.ellipse(c.x, c.y, rad, rad * 0.34f);
    VectorPath latTop;   latTop.ellipse(c.x, c.y - rad * 0.5f, rad * 0.78f, rad * 0.22f);
    VectorPath latBot;   latBot.ellipse(c.x, c.y + rad * 0.5f, rad * 0.78f, rad * 0.22f);
    m_uiDrawList.addPathStroked(equator, line, so);
    m_uiDrawList.addPathStroked(latTop, line, so);
    m_uiDrawList.addPathStroked(latBot, line, so);

    // Meridian (vertical oval) + the central pole-to-pole line.
    VectorPath meridian; meridian.ellipse(c.x, c.y, rad * 0.46f, rad);
    m_uiDrawList.addPathStroked(meridian, line, so);
    VectorPath pole; pole.moveTo(c.x, c.y - rad); pole.lineTo(c.x, c.y + rad);
    m_uiDrawList.addPathStroked(pole, line, so);

    m_uiDrawList.addCircle(c, rad, kFlatInk, it);
}

void RetroDemoApp::drawIconDocuments(float x, float y, float sz, float s) {
    const float it   = std::round(2.f*s);
    const float tabH = sz*0.16f, tabW = sz*0.42f;
    // Tab
    const UiRect tab = {x, y+sz*0.18f, x+tabW, y+sz*0.18f+tabH+sz*0.10f};
    m_uiDrawList.addRoundRectFilled(tab, kFlatOrange, 3.f*s);
    // Body
    const UiRect body = {x, y+sz*0.30f, x+sz, y+sz*0.82f};
    m_uiDrawList.addRoundRectFilled(body, kFlatOrange, 3.f*s);
    m_uiDrawList.addRoundRect(body, kFlatInk, 3.f*s, it);
}

void RetroDemoApp::drawIconDoc(float x, float y, float sz, float s) {
    const float it  = std::round(2.f*s);
    const float dog = sz*0.26f;
    const UiRect body = {x+sz*0.14f, y+sz*0.10f, x+sz*0.86f, y+sz*0.90f};
    m_uiDrawList.addRoundRectFilled(body, {1,1,1,1}, 3.f*s);
    // Dog-ear
    m_uiDrawList.addRectFilled({body.maxX-dog, body.minY, body.maxX, body.minY+dog}, kFlatBlue);
    m_uiDrawList.addRoundRect(body, kFlatInk, 3.f*s, it);
    // Text lines
    for (int i = 0; i < 3; ++i) {
        const float ly = body.minY + dog + sz*0.12f + i*sz*0.16f;
        m_uiDrawList.addRectFilled(
            {body.minX+sz*0.10f, ly, body.maxX-sz*0.10f, ly+it}, kFlatRetro.textDim);
    }
}

void RetroDemoApp::drawIconAvatar(float x, float y, float sz, float s) {
    const float it  = std::round(2.f*s);
    const UiVec2 c  = {x+sz*0.5f, y+sz*0.5f};
    const float rad = sz*0.42f;
    m_uiDrawList.addCircleFilled(c, rad, {0.86f,0.92f,0.98f,1.f});
    // Head
    m_uiDrawList.addCircleFilled({c.x, c.y-sz*0.10f}, sz*0.15f, kFlatInk);
    // Shoulders
    m_uiDrawList.addRoundRectFilled(
        {c.x-sz*0.24f, c.y+sz*0.06f, c.x+sz*0.24f, c.y+rad}, kFlatInk, sz*0.18f);
    m_uiDrawList.addCircle(c, rad, kFlatInk, it);
}

void RetroDemoApp::drawIconDisc(float x, float y, float sz, float s) {
    const float it  = std::round(2.f*s);
    const UiVec2 c  = {x+sz*0.5f, y+sz*0.5f};
    const float rad = sz*0.44f;
    m_uiDrawList.addCircleFilled(c, rad, kFlatGreen);
    m_uiDrawList.addCircle(c, rad, kFlatInk, it);
    m_uiDrawList.addCircleFilled(c, rad*0.30f, {1,1,1,1});
    m_uiDrawList.addCircle(c, rad*0.30f, kFlatInk, it);
    m_uiDrawList.addCircleFilled(c, rad*0.07f, kFlatInk);
}

// ─── Gallery layout ───────────────────────────────────────────────────────────

void RetroDemoApp::drawFlatInterface(float fw, float fh, float s) {
    const Font& fn  = m_uiFont;
    const Font& fnB = m_uiFontBold;
    const float lh  = fn.lineHeightPx();

    // Grid: 3 columns × 2 rows
    const int   cols = 3, rows = 2;
    const float marginX = std::round(fw*0.030f);
    const float top     = std::round(fh*0.045f);
    const float gap     = std::round(20.f*s);
    const float botMrg  = std::round(46.f*s);   // room for hint line
    const float cellW = std::round((fw - 2.f*marginX - (cols-1)*gap) / cols);
    const float cellH = std::round((fh - top - botMrg - (rows-1)*gap) / rows);

    auto cell = [&](int col, int rowi) {
        return UiRect::fromXYWH(marginX + col*(cellW+gap),
                                top + rowi*(cellH+gap), cellW, cellH);
    };
    auto textLine = [&](const Font& f, const char* str, float tx, float ty, UiColor col) {
        m_uiDrawList.addText(f, str, {tx, ty}, col);
    };

    const UiColor dim = kFlatRetro.textDim;
    const UiColor ink = kFlatInk;

    // ── 1. Installation (blue) ───────────────────────────────────────────────
    {
        UiRect ct = drawFlatWindow(cell(0,0), "Installation", kFlatBlue, s);
        drawIconDocuments(ct.minX, ct.minY, std::round(40.f*s), s);
        const float tx = ct.minX + std::round(52.f*s);
        textLine(fn, "Choose a destination folder", tx, ct.minY, ink);
        textLine(fn, "for the installation.", tx, ct.minY+lh*1.2f, dim);
        // Buttons row
        const float bh = std::round(26.f*s), bw = std::round(72.f*s);
        const float by = ct.maxY - bh;
        drawFlatButton(UiRect::fromXYWH(ct.minX, by, bw, bh), "Back", s);
        drawFlatButton(UiRect::fromXYWH(ct.minX+bw+8.f*s, by, bw, bh), "Browse", s);
        drawFlatButton(UiRect::fromXYWH(ct.maxX-bw, by, bw, bh), "Next", s);
    }

    // ── 2. Audio Player (blue) ───────────────────────────────────────────────
    {
        UiRect ct = drawFlatWindow(cell(1,0), "Audio Player", kFlatBlue, s);
        const float disc = std::round(48.f*s);
        drawIconDisc(ct.minX, ct.minY, disc, s);
        // Transport buttons
        const float tx = ct.minX + disc + std::round(16.f*s);
        const float ty = ct.minY + std::round(6.f*s);
        const UiVec2 pc = {tx+std::round(34.f*s), ty+std::round(12.f*s)};
        m_uiDrawList.addSectorFilled({pc.x-14.f*s, pc.y}, 0, 8.f*s, 2.54f, 3.74f, ink, 8);
        m_uiDrawList.addSectorFilled({pc.x-6.f*s,  pc.y}, 0, 8.f*s, 2.54f, 3.74f, ink, 8);
        m_uiDrawList.addSectorFilled(pc,            0, 12.f*s, -0.6f, 0.6f, ink, 8);
        m_uiDrawList.addSectorFilled({pc.x+22.f*s, pc.y}, 0, 8.f*s, -0.6f, 0.6f, ink, 8);
        m_uiDrawList.addSectorFilled({pc.x+30.f*s, pc.y}, 0, 8.f*s, -0.6f, 0.6f, ink, 8);
        // Slider
        const float sy = ct.minY + disc + std::round(2.f*s);
        const float sx0 = ct.minX, sx1 = ct.maxX;
        m_uiDrawList.addRoundRectFilled({sx0, sy, sx1, sy+std::round(6.f*s)},
            {0.80f,0.78f,0.72f,1.f}, 3.f*s);
        const float knobX = sx0 + (sx1-sx0)*0.4f;
        m_uiDrawList.addRoundRectFilled({sx0, sy, knobX, sy+std::round(6.f*s)}, kFlatBlue, 3.f*s);
        m_uiDrawList.addCircleFilled({knobX, sy+std::round(3.f*s)}, std::round(7.f*s), {1,1,1,1});
        m_uiDrawList.addCircle({knobX, sy+std::round(3.f*s)}, std::round(7.f*s), ink, std::round(2.f*s));
        textLine(fn, "Band - Track#_01.mp3", ct.minX, ct.maxY-lh, dim);
    }

    // ── 3. Sign In (green) ───────────────────────────────────────────────────
    {
        UiRect ct = drawFlatWindow(cell(2,0), "Sign In", kFlatGreen, s);
        const float av = std::round(40.f*s);
        drawIconAvatar(ct.minX+(ct.width()-av)*0.5f, ct.minY, av, s);
        const float fh2 = std::round(24.f*s);
        float fy = ct.minY + av + std::round(8.f*s);
        drawFlatField(UiRect::fromXYWH(ct.minX, fy, ct.width(), fh2), false, s);
        textLine(fn, "LOGIN", ct.minX+8.f*s, fy+(fh2-lh)*0.5f, dim);
        fy += fh2 + std::round(8.f*s);
        drawFlatField(UiRect::fromXYWH(ct.minX, fy, ct.width(), fh2), false, s);
        textLine(fn, "Password", ct.minX+8.f*s, fy+(fh2-lh)*0.5f, dim);
        fy += fh2 + std::round(6.f*s);
        textLine(fn, "Forgot password?", ct.minX, fy, dim);
        const float bh = std::round(26.f*s), bw = std::round(70.f*s);
        const float by = ct.maxY - bh;
        drawFlatButton(UiRect::fromXYWH(ct.minX, by, bw, bh), "OK", s);
        drawFlatButton(UiRect::fromXYWH(ct.minX+bw+8.f*s, by, bw, bh), "Cancel", s);
    }

    // ── 4. Mail (pink) ───────────────────────────────────────────────────────
    {
        UiRect ct = drawFlatWindow(cell(0,1), "Mail", kFlatPink, s);
        textLine(fnB, "Mail To:", ct.minX, ct.minY, ink);
        const float fh2 = std::round(22.f*s);
        float fy = ct.minY + lh*1.3f;
        drawFlatField(UiRect::fromXYWH(ct.minX, fy, ct.width(), fh2), true, s);
        fy += fh2 + std::round(7.f*s);
        drawFlatField(UiRect::fromXYWH(ct.minX, fy, ct.width(), fh2), true, s);
        fy += fh2 + std::round(8.f*s);
        // Message body lines
        for (int i = 0; i < 3; ++i)
            m_uiDrawList.addRectFilled(
                {ct.minX, fy+i*lh*1.1f, ct.maxX - (i==2? ct.width()*0.4f : 0.f),
                 fy+i*lh*1.1f+std::round(2.f*s)}, dim);
        const float bh = std::round(26.f*s), bw = std::round(72.f*s);
        drawFlatButton(UiRect::fromXYWH(ct.minX, ct.maxY-bh, bw, bh), "Send", s);
    }

    // ── 5. Notes (orange) ────────────────────────────────────────────────────
    {
        UiRect ct = drawFlatWindow(cell(1,1), "Notes", kFlatOrange, s);
        textLine(fnB, "Note #1", ct.minX, ct.minY, ink);
        float ly = ct.minY + lh*1.5f;
        for (int i = 0; i < 5 && ly < ct.maxY; ++i) {
            const float w = (i%3==2) ? ct.width()*0.55f : ct.width();
            m_uiDrawList.addRectFilled({ct.minX, ly, ct.minX+w, ly+std::round(2.f*s)}, dim);
            ly += lh*1.15f;
        }
    }

    // ── 6. Shortcuts / icon grid (green) ─────────────────────────────────────
    {
        UiRect ct = drawFlatWindow(cell(2,1), "Shortcuts", kFlatGreen, s);
        struct IconEntry { const char* label; int kind; };
        const IconEntry entries[] = {
            {"Mail",0},{"Photo",1},{"Video",2},{"Internet",3},{"Docs",4},{"Note",5},
        };
        const float isz = std::round(34.f*s);
        const int   gc  = 3;
        const float gx  = (ct.width() - gc*isz) / (gc+1);
        const float gy  = std::round(8.f*s);
        for (int i = 0; i < 6; ++i) {
            const int c = i % gc, rr = i / gc;
            const float ix = ct.minX + gx + c*(isz+gx);
            const float iy = ct.minY + rr*(isz + lh + gy) + std::round(2.f*s);
            switch (entries[i].kind) {
                case 0: drawIconMail(ix, iy, isz, s); break;
                case 1: drawIconPhoto(ix, iy, isz, s); break;
                case 2: drawIconVideo(ix, iy, isz, s); break;
                case 3: drawIconInternet(ix, iy, isz, s); break;
                case 4: drawIconDocuments(ix, iy, isz, s); break;
                case 5: drawIconDoc(ix, iy, isz, s); break;
            }
            const float tw = fn.measureText(entries[i].label);
            textLine(fn, entries[i].label, ix+(isz-tw)*0.5f, iy+isz+std::round(2.f*s), ink);
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
//  Retro-OS styleguide (Windows-10 flat, single blue accent, 1px borders)
// ════════════════════════════════════════════════════════════════════════════

void RetroDemoApp::drawOsIcon(int kind, float x, float y, float sz,
                               UiColor col, float s) {
    const float t = std::max(1.f, std::round(1.5f*s));
    const UiRect box = UiRect::fromXYWH(x, y, sz, sz);
    switch (kind) {
        case 0: { // folder
            const float tabH = sz*0.18f, tabW = sz*0.45f;
            m_uiDrawList.addRectFilled({x, y+sz*0.22f-tabH, x+tabW, y+sz*0.22f}, col);
            m_uiDrawList.addRect({x, y+sz*0.22f, x+sz, y+sz*0.80f}, col, t);
            m_uiDrawList.addRectFilled({x, y+sz*0.22f, x+sz, y+sz*0.22f+t}, col);
            break;
        }
        case 1: { // gear
            const UiVec2 c{x+sz*0.5f, y+sz*0.5f};
            const float r = sz*0.34f, tw = sz*0.14f, th = sz*0.12f;
            m_uiDrawList.addRectFilled({c.x-tw*0.5f, c.y-r-th*0.5f, c.x+tw*0.5f, c.y-r+th}, col);
            m_uiDrawList.addRectFilled({c.x-tw*0.5f, c.y+r-th, c.x+tw*0.5f, c.y+r+th*0.5f}, col);
            m_uiDrawList.addRectFilled({c.x-r-th*0.5f, c.y-tw*0.5f, c.x-r+th, c.y+tw*0.5f}, col);
            m_uiDrawList.addRectFilled({c.x+r-th, c.y-tw*0.5f, c.x+r+th*0.5f, c.y+tw*0.5f}, col);
            m_uiDrawList.addCircle(c, r*0.78f, col, t);
            m_uiDrawList.addCircle(c, r*0.30f, col, t);
            break;
        }
        case 2: { // document
            m_uiDrawList.addRect({x+sz*0.18f, y+sz*0.10f, x+sz*0.82f, y+sz*0.90f}, col, t);
            m_uiDrawList.addRectFilled({x+sz*0.62f, y+sz*0.10f, x+sz*0.82f, y+sz*0.30f}, col);
            for (int i = 0; i < 3; ++i) {
                const float ly = y+sz*0.42f + i*sz*0.16f;
                m_uiDrawList.addRectFilled({x+sz*0.28f, ly, x+sz*0.72f, ly+t}, col);
            }
            break;
        }
        case 3: { // chart
            m_uiDrawList.addRectFilled({x+sz*0.16f, y+sz*0.16f, x+sz*0.16f+t, y+sz*0.84f}, col);
            m_uiDrawList.addRectFilled({x+sz*0.16f, y+sz*0.84f-t, x+sz*0.84f, y+sz*0.84f}, col);
            const float bw = sz*0.13f;
            m_uiDrawList.addRectFilled({x+sz*0.26f, y+sz*0.55f, x+sz*0.26f+bw, y+sz*0.82f}, col);
            m_uiDrawList.addRectFilled({x+sz*0.46f, y+sz*0.38f, x+sz*0.46f+bw, y+sz*0.82f}, col);
            m_uiDrawList.addRectFilled({x+sz*0.66f, y+sz*0.24f, x+sz*0.66f+bw, y+sz*0.82f}, col);
            // Anti-aliased trend polyline over the bar tops (rounded joins).
            const UiVec2 trend[4] = {
                {x+sz*0.20f, y+sz*0.66f}, {x+sz*0.325f, y+sz*0.52f},
                {x+sz*0.525f, y+sz*0.34f}, {x+sz*0.78f, y+sz*0.20f}};
            m_uiDrawList.addPolylineAA(trend, 4, col, t, false);
            break;
        }
        case 4: { // trash
            m_uiDrawList.addRectFilled({x+sz*0.20f, y+sz*0.18f, x+sz*0.80f, y+sz*0.18f+t}, col);
            m_uiDrawList.addRectFilled({x+sz*0.40f, y+sz*0.10f, x+sz*0.60f, y+sz*0.18f}, col);
            m_uiDrawList.addRect({x+sz*0.26f, y+sz*0.22f, x+sz*0.74f, y+sz*0.86f}, col, t);
            for (int i = 0; i < 3; ++i) {
                const float lx = x+sz*0.36f + i*sz*0.14f;
                m_uiDrawList.addRectFilled({lx, y+sz*0.30f, lx+t, y+sz*0.78f}, col);
            }
            break;
        }
        case 5: { // image
            m_uiDrawList.addRect(box, col, t);
            m_uiDrawList.addCircleFilled({x+sz*0.32f, y+sz*0.32f}, sz*0.10f, col);
            m_uiDrawList.addRectFilled({x+t, y+sz*0.62f, x+sz*0.45f, y+sz-t}, col);
            m_uiDrawList.addRectFilled({x+sz*0.38f, y+sz*0.50f, x+sz-t, y+sz-t}, col);
            break;
        }
        default: break;
    }
}

void RetroDemoApp::drawRetroOsInterface(float fw, float fh, float s) {
    const Font& fn   = m_uiFont;
    const Font& fnB  = m_uiFontBold;
    const Font& h1   = m_osH1.valid() ? m_osH1 : m_uiFontBold;
    const Font& h2   = m_osH2.valid() ? m_osH2 : m_uiFontBold;
    const Font& h3   = m_osH3.valid() ? m_osH3 : m_uiFontBold;
    const float lh   = fn.lineHeightPx();
    const float lhB  = fnB.lineHeightPx();
    const float t    = std::max(1.f, std::round(s));

    // Faint horizontal pinstripes on the desktop
    for (float py = 0.f; py < fh; py += std::round(4.f*s))
        m_uiDrawList.addRectFilled({0, py, fw, py+t}, {1,1,1,0.35f});

    const float margin = std::round(40.f*s);
    const float pageW  = std::min(fw - 2.f*margin, std::round(920.f*s));
    const float x0     = std::round((fw - pageW)*0.5f);
    const float gap    = std::round(24.f*s);
    const float colGap = std::round(28.f*s);
    const float colW   = std::round((pageW - colGap)*0.5f);
    const float rightX = x0 + colW + colGap;

    auto fillBorder = [&](UiRect r, UiColor fill, UiColor border, float th) {
        m_uiDrawList.addRectFilled(r, fill);
        m_uiDrawList.addRect(r, border, th);
    };
    auto label = [&](const char* str, float lx, float ly) {
        m_uiDrawList.addText(fnB, str, {lx, ly}, kOsInk);
        return ly + lhB + std::round(6.f*s);
    };
    auto osButton = [&](UiRect r, const char* str, int state) {
        // state: 0 normal, 1 hover, 2 disabled
        const UiColor fill = state==2 ? kOsDisable : state==1 ? kOsBlueHov : kOsBlue;
        const UiColor txt  = state==2 ? UiColor{0.55f,0.55f,0.55f,1.f} : UiColor{1,1,1,1};
        m_uiDrawList.addRectFilled(r, fill);
        if (state==2) m_uiDrawList.addRect(r, kOsBorder, t);
        const float tw = fnB.measureText(str);
        m_uiDrawList.addText(fnB, str,
            {r.minX+(r.width()-tw)*0.5f, r.minY+(r.height()-lhB)*0.5f}, txt);
    };
    auto checkMark = [&](float cx, float cy, float sz, UiColor col) {
        // stepped-square approximation of a checkmark
        const float q = sz*0.16f;
        auto dot = [&](float nx, float ny) {
            m_uiDrawList.addRectFilled(
                {cx+nx*sz-q*0.5f, cy+ny*sz-q*0.5f, cx+nx*sz+q*0.5f, cy+ny*sz+q*0.5f}, col);
        };
        dot(0.20f,0.52f); dot(0.32f,0.64f); dot(0.44f,0.74f);
        dot(0.58f,0.54f); dot(0.72f,0.34f); dot(0.84f,0.18f);
    };

    float cy = std::round(26.f*s);

    // ── Title ────────────────────────────────────────────────────────────────
    m_uiDrawList.addText(h1, "RETRO-OS UI KIT", {x0, cy}, kOsBlue);
    cy += h1.lineHeightPx() + std::round(2.f*s);
    m_uiDrawList.addText(h3, "STYLEGUIDE & COMPONENT LIBRARY", {x0, cy}, kOsInk);
    cy += h3.lineHeightPx() + gap;

    // ── Color tokens ─────────────────────────────────────────────────────────
    cy = label("COLOR TOKENS", x0, cy);
    {
        struct Tok { const char* name; const char* hex; UiColor col; bool darkText; };
        const Tok toks[] = {
            {"GRAY","#A0A0A0",kOsGray,false},
            {"WHITE","#F0F0F0",kOsWhite,true},
            {"BLUE","#0078D7",kOsBlue,false},
            {"RED","#E81123",kOsRed,false},
        };
        const float sgap = std::round(14.f*s);
        const float swW  = std::round((pageW - 3.f*sgap)/4.f);
        const float swH  = std::round(74.f*s);
        for (int i = 0; i < 4; ++i) {
            const float sx = x0 + i*(swW+sgap);
            const UiRect sw = UiRect::fromXYWH(sx, cy, swW, swH);
            m_uiDrawList.addRectFilled(sw, toks[i].col);
            // gradient-to-white band (bottom third)
            m_uiDrawList.addRectFilledHGradient(
                {sw.minX, sw.maxY-std::round(22.f*s), sw.maxX, sw.maxY},
                toks[i].col, kOsWhite);
            m_uiDrawList.addRect(sw, kOsBorder, t);
            const UiColor tc = toks[i].darkText ? kOsInk : UiColor{1,1,1,1};
            m_uiDrawList.addText(fnB, toks[i].name, {sx+std::round(8.f*s), cy+std::round(8.f*s)}, tc);
            m_uiDrawList.addText(fn,  toks[i].hex,  {sx+std::round(8.f*s), cy+std::round(8.f*s)+lhB}, tc);
        }
        cy += swH + gap;
    }

    // ── Type scale (left) + Icon grid & borders (right) ──────────────────────
    {
        float ly = label("TYPE SCALE", x0, cy);
        auto scaleRow = [&](const Font& big, const char* tag, const Font& lab2,
                            const char* txt) {
            m_uiDrawList.addText(big, tag, {x0, ly}, kOsInk);
            const float bx = x0 + std::round(86.f*s);
            m_uiDrawList.addText(lab2, txt,
                {bx, ly + (big.lineHeightPx()-lab2.lineHeightPx())*0.5f}, kOsInk);
            ly += big.lineHeightPx() + std::round(6.f*s);
        };
        scaleRow(h1, "H1", fnB, "Bitmap label");
        scaleRow(h2, "H2", fnB, "Modern label");
        scaleRow(h3, "H3", fnB, "Modern label");
        m_uiDrawList.addText(fnB, "Body", {x0, ly}, kOsInk);
        m_uiDrawList.addText(fn, "Modern sans title", {x0+std::round(86.f*s), ly}, kOsInk);
        ly += lhB + std::round(6.f*s);
        m_uiDrawList.addText(fn, "Caption", {x0, ly}, kOsInk);
        m_uiDrawList.addText(fn, "Modern sans title", {x0+std::round(86.f*s), ly}, kOsInk);
        ly += lh + std::round(6.f*s);

        // Right: icon grid
        float ry = label("ICON GRID", rightX, cy);
        auto iconBox = [&](float bx, float by, float isz, const char* cap) {
            const float pad = std::round(10.f*s);
            const float gw = isz*3.f + pad*4.f;
            const float gh = isz*2.f + pad*3.f;
            const UiRect b = UiRect::fromXYWH(bx, by, gw, gh);
            fillBorder(b, kOsWhite, kOsBorder, t);
            const int kinds[6] = {0,1,2,3,1,4};
            for (int i = 0; i < 6; ++i) {
                const int c = i%3, r2 = i/3;
                drawOsIcon(kinds[i], bx+pad+c*(isz+pad), by+pad+r2*(isz+pad),
                           isz, kOsInk, s);
            }
            const float tw = fn.measureText(cap);
            m_uiDrawList.addText(fn, cap, {bx+(gw-tw)*0.5f, by+gh+std::round(4.f*s)}, kOsInk);
            return gw;
        };
        const float w16 = iconBox(rightX, ry, std::round(16.f*s), "16px");
        iconBox(rightX + w16 + std::round(18.f*s), ry, std::round(24.f*s), "24px");
        ry += std::round(16.f*s)*2.f + std::round(10.f*s)*3.f + lh + gap;

        // Right: borders & radius
        ry = label("BORDERS & RADIUS", rightX, ry);
        const float bw = std::round(120.f*s), bh = std::round(58.f*s);
        const UiRect b1 = UiRect::fromXYWH(rightX, ry, bw, bh);
        m_uiDrawList.addRectFilled(b1, kOsWhite);
        m_uiDrawList.addRect(b1, kOsBlue, t);
        m_uiDrawList.addText(fn, "1px Solid Blue", {rightX, ry+bh+std::round(4.f*s)}, kOsInk);
        const UiRect b2 = UiRect::fromXYWH(rightX+bw+std::round(24.f*s), ry, bw, bh);
        m_uiDrawList.addRoundRectFilled(b2, kOsDisable, std::round(4.f*s));
        m_uiDrawList.addRoundRect(b2, kOsBorder, std::round(4.f*s), t);
        m_uiDrawList.addText(fn, "4px Rounded",
            {b2.minX, ry+bh+std::round(4.f*s)}, kOsInk);
        ry += bh + lh + gap;

        cy = std::max(ly, ry);
    }

    // ── Component gallery ────────────────────────────────────────────────────
    cy = label("COMPONENT GALLERY", x0, cy);

    // Left column ---------------------------------------------------------------
    float lcy = cy;
    {
        lcy = label("WINDOW HEADER", x0, lcy);
        const float winH = std::round(86.f*s), titleH = std::round(24.f*s);
        const UiRect win = UiRect::fromXYWH(x0, lcy, colW, winH);
        fillBorder(win, kOsWhite, kOsBorder, t);
        m_uiDrawList.addRectFilled({win.minX, win.minY, win.maxX, win.minY+titleH}, kOsBlue);
        // window buttons _ □ ×
        float bx = win.maxX - std::round(22.f*s);
        for (const char* g : {"\xc3\x97", "\xe2\x96\xa1", "_"}) {
            const UiRect gb = UiRect::fromXYWH(bx, win.minY+std::round(3.f*s),
                                               std::round(18.f*s), titleH-std::round(6.f*s));
            m_uiDrawList.addRect(gb, {1,1,1,0.6f}, t);
            const float gw = fn.measureText(g);
            m_uiDrawList.addText(fn, g, {gb.minX+(gb.width()-gw)*0.5f,
                gb.minY+(gb.height()-lh)*0.5f}, {1,1,1,1});
            bx -= std::round(21.f*s);
        }
        lcy += winH + gap;

        lcy = label("INPUT", x0, lcy);
        const UiRect inp = UiRect::fromXYWH(x0, lcy, colW, std::round(30.f*s));
        m_uiDrawList.addRectFilled(inp, {1,1,1,1});
        m_uiDrawList.addRect(inp, kOsBlue, t);
        m_uiDrawList.addText(fn, "Type command",
            {inp.minX+std::round(8.f*s), inp.minY+(inp.height()-lh)*0.5f}, kOsGray);
        m_uiDrawList.addRectFilled(
            {inp.minX+std::round(8.f*s)+fn.measureText("Type command")+2.f*s,
             inp.minY+std::round(6.f*s), inp.minX+std::round(8.f*s)+fn.measureText("Type command")+2.f*s+t,
             inp.maxY-std::round(6.f*s)}, kOsInk);
        lcy += std::round(30.f*s) + gap;

        lcy = label("CHECKBOX", x0, lcy);
        const float cb = std::round(18.f*s), cbg = std::round(10.f*s);
        for (int i = 0; i < 4; ++i) {
            const int c = i%2, r2 = i/2;
            const UiRect box = UiRect::fromXYWH(x0+c*(cb+cbg), lcy+r2*(cb+cbg), cb, cb);
            if (r2 == 0) { // checked
                m_uiDrawList.addRectFilled(box, kOsBlue);
                checkMark(box.minX, box.minY, cb, {1,1,1,1});
            } else {
                m_uiDrawList.addRectFilled(box, c==0 ? UiColor{1,1,1,1} : kOsDisable);
                m_uiDrawList.addRect(box, kOsBorder, t);
            }
        }
        lcy += 2.f*(cb+cbg) + gap*0.5f;

        lcy = label("RULES CARD", x0, lcy);
        const UiRect card = UiRect::fromXYWH(x0, lcy, colW, std::round(96.f*s));
        m_uiDrawList.addRoundRectFilled(card, kOsDisable, std::round(4.f*s));
        m_uiDrawList.addRoundRect(card, kOsBorder, std::round(4.f*s), t);
        const char* rules[] = {"Use 1 accent color","Avoid heavy shadows","Keep borders 1px"};
        float ruleY = card.minY + std::round(12.f*s);
        for (const char* rstr : rules) {
            m_uiDrawList.addCircleFilled(
                {card.minX+std::round(16.f*s), ruleY+lh*0.5f}, std::round(2.5f*s), kOsInk);
            m_uiDrawList.addText(fn, rstr, {card.minX+std::round(28.f*s), ruleY}, kOsInk);
            ruleY += lh + std::round(7.f*s);
        }
        lcy += std::round(96.f*s) + gap;
    }

    // Right column --------------------------------------------------------------
    float rcy = cy;
    {
        rcy = label("BUTTONS", rightX, rcy);
        const float bw = std::round((colW - 2.f*std::round(10.f*s))/3.f);
        const float bh = std::round(30.f*s);
        osButton(UiRect::fromXYWH(rightX, rcy, bw, bh), "Normal", 0);
        osButton(UiRect::fromXYWH(rightX+bw+std::round(10.f*s), rcy, bw, bh), "Hover", 1);
        osButton(UiRect::fromXYWH(rightX+2.f*(bw+std::round(10.f*s)), rcy, bw, bh), "Disabled", 2);
        rcy += bh + gap;

        rcy = label("DISABLE", rightX, rcy);
        const UiRect track = UiRect::fromXYWH(rightX, rcy, colW, std::round(16.f*s));
        m_uiDrawList.addRoundRectFilled(track, kOsDisable, std::round(8.f*s));
        m_uiDrawList.addRoundRectFilled(
            {track.minX, track.minY, track.minX+colW*0.62f, track.maxY}, kOsBlue, std::round(8.f*s));
        m_uiDrawList.addRoundRect(track, kOsBorder, std::round(8.f*s), t);
        rcy += std::round(16.f*s) + gap;

        rcy = label("TABS", rightX, rcy);
        const float tabW = std::round(colW*0.5f), tabH = std::round(30.f*s);
        const UiRect tA = UiRect::fromXYWH(rightX, rcy, tabW, tabH);
        const UiRect tB = UiRect::fromXYWH(rightX+tabW, rcy, tabW, tabH);
        m_uiDrawList.addRectFilled(tB, kOsDisable);
        m_uiDrawList.addRect(tB, kOsBorder, t);
        m_uiDrawList.addRectFilled(tA, {1,1,1,1});
        m_uiDrawList.addRect(tA, kOsBorder, t);
        m_uiDrawList.addRectFilled({tA.minX, tA.minY, tA.maxX, tA.minY+std::round(3.f*s)}, kOsBlue);
        { const float w = fnB.measureText("Active");
          m_uiDrawList.addText(fnB,"Active",{tA.minX+(tabW-w)*0.5f,tA.minY+(tabH-lhB)*0.5f},kOsInk); }
        { const float w = fn.measureText("Inactive");
          m_uiDrawList.addText(fn,"Inactive",{tB.minX+(tabW-w)*0.5f,tB.minY+(tabH-lh)*0.5f},kOsGray); }
        rcy += tabH + gap;

        rcy = label("TOAST", rightX, rcy);
        const UiRect toast = UiRect::fromXYWH(rightX, rcy, colW, std::round(34.f*s));
        m_uiDrawList.addRoundRectFilled(toast, kOsBlue, std::round(4.f*s));
        // tail
        m_uiDrawList.addSectorFilled({toast.minX+std::round(16.f*s), toast.maxY},
            0.f, std::round(10.f*s), 0.2f, 1.37f, kOsBlue, 6);
        { const float w = fnB.measureText("System Update Complete");
          m_uiDrawList.addText(fnB, "System Update Complete",
            {toast.minX+(colW-w)*0.5f, toast.minY+(toast.height()-lhB)*0.5f}, {1,1,1,1}); }
        rcy += std::round(34.f*s) + std::round(10.f*s) + gap;

        rcy = label("MODAL", rightX, rcy);
        const float mW = colW, mH = std::round(104.f*s), mTitle = std::round(22.f*s);
        const UiRect modal = UiRect::fromXYWH(rightX, rcy, mW, mH);
        fillBorder(modal, kOsWhite, kOsBorder, t);
        m_uiDrawList.addRectFilled({modal.minX, modal.minY, modal.maxX, modal.minY+mTitle}, kOsBlue);
        { const UiRect xb = UiRect::fromXYWH(modal.maxX-std::round(20.f*s),
            modal.minY+std::round(3.f*s), std::round(16.f*s), mTitle-std::round(6.f*s));
          m_uiDrawList.addRect(xb, {1,1,1,0.6f}, t);
          const float w = fn.measureText("\xc3\x97");
          m_uiDrawList.addText(fn,"\xc3\x97",{xb.minX+(xb.width()-w)*0.5f,
            xb.minY+(xb.height()-lh)*0.5f},{1,1,1,1}); }
        m_uiDrawList.addText(fnB, "Confirm Action?",
            {modal.minX+std::round(14.f*s), modal.minY+mTitle+std::round(12.f*s)}, kOsInk);
        const float mbW = std::round(82.f*s), mbH = std::round(28.f*s);
        const float mby = modal.maxY - mbH - std::round(12.f*s);
        osButton(UiRect::fromXYWH(modal.minX+std::round(14.f*s), mby, mbW, mbH), "OK", 0);
        { const UiRect cancel = UiRect::fromXYWH(modal.minX+std::round(14.f*s)+mbW+std::round(10.f*s),
            mby, mbW, mbH);
          m_uiDrawList.addRectFilled(cancel, {1,1,1,1});
          m_uiDrawList.addRect(cancel, kOsBorder, t);
          const float w = fnB.measureText("Cancel");
          m_uiDrawList.addText(fnB,"Cancel",{cancel.minX+(mbW-w)*0.5f,
            cancel.minY+(mbH-lhB)*0.5f}, kOsInk); }
        rcy += mH + gap;
    }
}

} // namespace odai::tools::retro_theme_demo
