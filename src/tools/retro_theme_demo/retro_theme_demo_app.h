#pragma once
#include "engine/game_app.h"
#include "render/renderer_types.h"
#include "ui/font.h"
#include "ui/ui_types.h"

namespace odai::tools::retro_theme_demo {

enum class Theme { Win95, Motif, ClassicMac, FlatRetro, RetroOS };

struct ThemePalette {
    ui::UiColor desktop, face, titleActive, titleText, text, textDim, highlight;
    const char* name;
};

class RetroDemoApp : public engine::GameApp {
protected:
    bool onInit() override;
    void onTick(float dt) override;
    void onRender(float dt) override;

private:
    const ThemePalette& palette() const;
    const ui::Font& themeFont() const;
    const ui::Font& themeFontBold() const;
    // Returns the item list (null-terminated) for menu index idx in the current theme.
    const char** currentMenuItems(int idx) const;

    // ── Hit testing ──────────────────────────────────────────────────────────
    bool isHovered(ui::UiRect r) const;
    bool isPressed(ui::UiRect r) const;

    // ── Draw primitives (no interaction) ─────────────────────────────────────
    void drawBevelPanel(ui::UiRect r, bool raised, float s);
    void drawGroupBorder(float gbX, float gbY, float gbW, float gbH,
                         float gapX, float gapW, float s);
    void drawCenteredText(ui::UiRect r, const ui::Font& font,
                          const char* text, ui::UiColor color);

    // ── Interactive controls ──────────────────────────────────────────────────
    // Draw a button in hover/pressed state; returns true the frame it is clicked.
    bool drawButton(ui::UiRect r, const char* label, float s);

    // Draws a horizontal menu-title strip.  Handles hover, click-to-open, and
    // hover-switching between open menus.  Writes m_dropdownX/Y when a menu
    // is open so drawMenuDropdown() knows where to place the panel.
    void drawMenuStrip(float barX, float barY, float barW, float barH,
                       const char** titles, int numTitles, float s);

    // Draws the open dropdown panel + items at m_dropdownX/Y.
    // Must be called LAST in onRender so it layers on top of everything.
    void drawMenuDropdown(float s);

    // ── Scene ─────────────────────────────────────────────────────────────────
    void drawWindow(float wx, float wy, float ww, float wh, float s);
    void drawTaskbar(float fw, float fh, float s);
    void drawMenuBar(float fw, float s);   // Mac Classic top bar

    // ── Mac Classic Finder interface ──────────────────────────────────────────
    void drawMacInterface(float fw, float fh, float s);
    // iconType: 0=none, 1=folder icons, 2=document icons in a row
    void drawMacFinderWindow(float wx, float wy, float ww, float wh,
                              const char* title, const char* infoLine,
                              int iconType, bool isActive, float s);
    void drawMacDocIcon(float x, float y, const char* label, float s);
    void drawMacFolderIcon(float x, float y, const char* label, float s);
    void drawMacTrash(float x, float y, float s);

    // ── Flat Retro UI kit ─────────────────────────────────────────────────────
    void drawFlatInterface(float fw, float fh, float s);
    // Draws an outlined cream window with a colored title bar; returns the inner
    // content rect (below the title strip, inset by padding).
    ui::UiRect drawFlatWindow(ui::UiRect r, const char* title,
                              ui::UiColor accent, float s);
    bool drawFlatButton(ui::UiRect r, const char* label, float s);
    void drawFlatField(ui::UiRect r, bool showLines, float s);

    // Procedural flat icons (ink outline + flat accent fills), drawn in a box
    // of side `sz` with top-left at (x, y).
    void drawIconMail(float x, float y, float sz, float s);
    void drawIconPhoto(float x, float y, float sz, float s);
    void drawIconVideo(float x, float y, float sz, float s);
    void drawIconInternet(float x, float y, float sz, float s);
    void drawIconDocuments(float x, float y, float sz, float s);
    void drawIconDoc(float x, float y, float sz, float s);
    void drawIconAvatar(float x, float y, float sz, float s);
    void drawIconDisc(float x, float y, float sz, float s);

    // ── Retro-OS styleguide (Windows-10 flat, single blue accent) ─────────────
    void drawRetroOsInterface(float fw, float fh, float s);
    // Monochrome line icon. kind: 0=folder 1=gear 2=doc 3=chart 4=trash 5=image
    void drawOsIcon(int kind, float x, float y, float sz, ui::UiColor col, float s);

    // ── Theme ─────────────────────────────────────────────────────────────────
    Theme    m_theme   = Theme::Win95;
    ui::Font m_macFont;
    ui::Font m_osH1;       // large bold heading for Retro-OS styleguide
    ui::Font m_osH2;
    ui::Font m_osH3;
    float    m_time    = 0.0f;

    // ── Mac Finder window positions ───────────────────────────────────────────
    float m_macWinX[2]    = {};
    float m_macWinY[2]    = {};
    bool  m_macWinPosInit = false;
    int   m_dragWin       = -1;
    float m_dragOffX      = 0.f;
    float m_dragOffY      = 0.f;

    // ── Keyboard ──────────────────────────────────────────────────────────────
    bool m_prevKey1 = false, m_prevKey2 = false, m_prevKey3 = false,
         m_prevKey4 = false, m_prevKey5 = false;

    // ── Mouse / click state (refreshed at top of onTick) ─────────────────────
    float m_mouseX      = 0.f, m_mouseY = 0.f;
    bool  m_lmbDown     = false;
    bool  m_lmbWasDown  = false;
    bool  m_justClicked = false;   // true exactly the frame LMB is released

    // ── Menu state ────────────────────────────────────────────────────────────
    int   m_openMenu     = -1;     // which menu bar index is open (-1 = none)
    float m_dropdownX    = 0.f;   // top-left of dropdown (set by drawMenuStrip)
    float m_dropdownY    = 0.f;
    bool  m_clickHandled = false;  // set true when a click is consumed by UI

    render::CameraPose m_camera{};
};

} // namespace odai::tools::retro_theme_demo
