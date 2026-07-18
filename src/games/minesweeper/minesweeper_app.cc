#include "games/minesweeper/minesweeper_app.h"

#include "ui/signal.h"
#include "ui/widget.h"
#include "ui/widgets/button.h"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <utility>
#include <vector>

namespace odai::games::minesweeper {

using namespace ui;

// ─── Palette (same warm-navy family as Snake / SWTOR) ────────────────────────
static constexpr UiColor kBg      {0.020f, 0.030f, 0.055f, 1.00f};
static constexpr UiColor kToolbar {0.047f, 0.078f, 0.125f, 0.96f};
static constexpr UiColor kBorder  {0.847f, 0.482f, 0.110f, 0.55f};
static constexpr UiColor kText    {0.784f, 0.831f, 0.909f, 1.00f};
static constexpr UiColor kDim     {0.376f, 0.471f, 0.596f, 1.00f};
static constexpr UiColor kGold    {0.910f, 0.565f, 0.188f, 1.00f};
static constexpr UiColor kRed     {0.847f, 0.173f, 0.110f, 1.00f};
static constexpr UiColor kGreen   {0.133f, 0.816f, 0.275f, 1.00f};
static constexpr UiColor kCovered {0.145f, 0.204f, 0.290f, 1.00f};  // raised tile face
static constexpr UiColor kRevealed{0.055f, 0.086f, 0.135f, 1.00f};  // sunken tile face
static constexpr UiColor kHiLite  {1.000f, 1.000f, 1.000f, 0.22f};
static constexpr UiColor kShadow  {0.000f, 0.000f, 0.000f, 0.45f};

// Classic Minesweeper number colours, indexed 1..8.
static UiColor numberColor(int n) {
    switch (n) {
        case 1:  return {0.34f, 0.60f, 0.96f, 1.0f};
        case 2:  return {0.22f, 0.78f, 0.38f, 1.0f};
        case 3:  return {0.92f, 0.34f, 0.28f, 1.0f};
        case 4:  return {0.52f, 0.44f, 0.92f, 1.0f};
        case 5:  return {0.86f, 0.50f, 0.22f, 1.0f};
        case 6:  return {0.22f, 0.80f, 0.80f, 1.0f};
        case 7:  return {0.88f, 0.84f, 0.90f, 1.0f};
        default: return {0.62f, 0.64f, 0.72f, 1.0f};  // 8
    }
}

// ─── Layout ──────────────────────────────────────────────────────────────────

MinesweeperApp::BoardLayout MinesweeperApp::computeLayout() const {
    const float s = contentScale();
    int fwi = 0, fhi = 0;
    framebufferSize(fwi, fhi);
    const float fw = static_cast<float>(fwi);
    const float fh = static_cast<float>(fhi);

    const float toolbarH = 60.0f * s;
    const float hudH     = 34.0f * s;
    const float pad      = 18.0f * s;

    const float availTop = toolbarH + hudH + pad;
    const float availW   = fw - 2.0f * pad;
    const float availH   = fh - availTop - pad;

    float cell = std::floor(std::min(availW / m_cols, availH / m_rows));
    cell = std::clamp(cell, 14.0f * s, 46.0f * s);

    const float gridW = cell * m_cols;
    const float gridH = cell * m_rows;

    BoardLayout lo;
    lo.s    = s;
    lo.cell = cell;
    lo.ox   = std::round((fw - gridW) * 0.5f);
    lo.oy   = std::round(availTop + (availH - gridH) * 0.5f);
    return lo;
}

// ─── Rules ───────────────────────────────────────────────────────────────────

void MinesweeperApp::setDifficulty(Difficulty d) {
    m_difficulty = d;
    switch (d) {
        case Difficulty::Beginner:     m_cols = 9;  m_rows = 9;  m_mines = 10; break;
        case Difficulty::Intermediate: m_cols = 16; m_rows = 16; m_mines = 40; break;
        case Difficulty::Expert:       m_cols = 30; m_rows = 16; m_mines = 99; break;
    }
    resetGame();
}

void MinesweeperApp::resetGame() {
    m_cells.assign(static_cast<std::size_t>(m_cols) * m_rows, Cell{});
    m_phase         = Phase::Playing;
    m_minesPlaced   = false;
    m_flagsPlaced   = 0;
    m_revealedCount = 0;
    m_elapsed       = 0.0f;
}

void MinesweeperApp::placeMines(int safeC, int safeR) {
    // Never place a mine on the first-clicked cell or its 8 neighbours, so the
    // opening click always uncovers a region (the classic "safe first click").
    auto forbidden = [&](int c, int r) {
        return std::abs(c - safeC) <= 1 && std::abs(r - safeR) <= 1;
    };

    int placed = 0;
    const int total = m_cols * m_rows;
    // With our difficulties, free cells always exceed mines, so this terminates.
    while (placed < m_mines && placed < total) {
        const int idx = static_cast<int>(rng() % static_cast<std::uint32_t>(total));
        const int c   = idx % m_cols;
        const int r   = idx / m_cols;
        if (cell(c, r).mine || forbidden(c, r)) continue;
        cell(c, r).mine = true;
        ++placed;
    }

    for (int r = 0; r < m_rows; ++r) {
        for (int c = 0; c < m_cols; ++c) {
            if (cell(c, r).mine) continue;
            int n = 0;
            for (int dr = -1; dr <= 1; ++dr)
                for (int dc = -1; dc <= 1; ++dc)
                    if ((dc || dr) && inBounds(c + dc, r + dr) && cell(c + dc, r + dr).mine) ++n;
            cell(c, r).adjacent = static_cast<std::uint8_t>(n);
        }
    }
}

void MinesweeperApp::reveal(int c, int r) {
    if (!inBounds(c, r) || cell(c, r).revealed || cell(c, r).flagged) return;

    if (!m_minesPlaced) {
        placeMines(c, r);
        m_minesPlaced = true;
    }

    if (cell(c, r).mine) {
        cell(c, r).revealed = true;
        m_phase = Phase::Lost;
        revealAllMines();
        return;
    }

    // Iterative flood fill: expand only through empty (0-adjacent) cells.
    std::vector<std::pair<int, int>> stack;
    stack.push_back({c, r});
    while (!stack.empty()) {
        const auto [cc, cr] = stack.back();
        stack.pop_back();
        Cell& cur = cell(cc, cr);
        if (cur.revealed || cur.flagged) continue;
        cur.revealed = true;
        ++m_revealedCount;
        if (cur.adjacent == 0) {
            for (int dr = -1; dr <= 1; ++dr)
                for (int dc = -1; dc <= 1; ++dc) {
                    if (!dc && !dr) continue;
                    const int nc = cc + dc, nr = cr + dr;
                    if (inBounds(nc, nr) && !cell(nc, nr).revealed &&
                        !cell(nc, nr).flagged && !cell(nc, nr).mine)
                        stack.push_back({nc, nr});
                }
        }
    }

    checkWin();
}

void MinesweeperApp::toggleFlag(int c, int r) {
    if (!inBounds(c, r) || cell(c, r).revealed) return;
    Cell& cur = cell(c, r);
    cur.flagged = !cur.flagged;
    m_flagsPlaced += cur.flagged ? 1 : -1;
}

void MinesweeperApp::revealAllMines() {
    for (Cell& cur : m_cells)
        if (cur.mine) cur.revealed = true;
}

void MinesweeperApp::checkWin() {
    if (m_revealedCount == m_cols * m_rows - m_mines) m_phase = Phase::Won;
}

// ─── Drawing ─────────────────────────────────────────────────────────────────

void MinesweeperApp::textCentered(const Font& f, std::string_view str, float cx, float cy,
                                  const UiColor& col) {
    const float w = f.measureText(str);
    m_uiDrawList.addText(f, str, {cx - w * 0.5f, cy - f.lineHeightPx() * 0.5f}, col);
}

void MinesweeperApp::drawToolbarBg(float fw, float s) {
    const UiRect bar = UiRect::fromXYWH(0.0f, 0.0f, fw, 60.0f * s);
    m_uiDrawList.addRectFilled(bar, kToolbar);
    m_uiDrawList.addRectFilled(
        UiRect::fromXYWH(0.0f, 60.0f * s - s, fw, s), kBorder);
    textCentered(m_uiFontBold, "MINESWEEPER", 116.0f * s, 30.0f * s, kGold);
}

void MinesweeperApp::drawHud(const BoardLayout& lo) {
    const float s     = lo.s;
    const float gridW = lo.cell * m_cols;
    const float hudH  = 30.0f * s;
    const float hy    = lo.oy - hudH - 8.0f * s;
    const UiRect hud  = UiRect::fromXYWH(lo.ox, hy, gridW, hudH);
    m_uiDrawList.addRoundRectFilled(hud, kToolbar, 4.0f * s);
    m_uiDrawList.addRoundRect(hud, kBorder, 4.0f * s, s);

    const float cy = hy + hudH * 0.5f;
    char buf[64];

    // Mines remaining (mines minus flags placed) on the left.
    std::snprintf(buf, sizeof(buf), "%d", m_mines - m_flagsPlaced);
    m_uiDrawList.addText(m_uiFont, "MINES", {lo.ox + 12.0f * s, cy - m_uiFont.lineHeightPx() * 0.5f}, kDim);
    m_uiDrawList.addText(m_uiFontBold, buf,
                         {lo.ox + 12.0f * s + m_uiFont.measureText("MINES  "),
                          cy - m_uiFontBold.lineHeightPx() * 0.5f}, kGold);

    // Status in the centre.
    const char* status = m_phase == Phase::Won  ? "CLEARED"
                       : m_phase == Phase::Lost ? "BOOM"
                                                : "";
    if (status[0])
        textCentered(m_uiFontBold, status, lo.ox + gridW * 0.5f, cy,
                     m_phase == Phase::Won ? kGreen : kRed);

    // Timer on the right.
    std::snprintf(buf, sizeof(buf), "%03d", std::min(999, static_cast<int>(m_elapsed)));
    const float tw = m_uiFontBold.measureText(buf);
    m_uiDrawList.addText(m_uiFontBold, buf,
                         {lo.ox + gridW - tw - 12.0f * s, cy - m_uiFontBold.lineHeightPx() * 0.5f}, kText);
}

void MinesweeperApp::drawBoard(const BoardLayout& lo) {
    const float s    = lo.s;
    const float cs   = lo.cell;
    const float pad  = std::max(1.0f, 1.5f * s);
    const float bevel = std::max(1.0f, 2.0f * s);

    // Board backdrop + frame.
    const UiRect board = UiRect::fromXYWH(lo.ox, lo.oy, cs * m_cols, cs * m_rows);
    m_uiDrawList.addDropShadow(board, {0.0f, 0.0f, 0.0f, 0.55f}, 10.0f * s, 2.0f * s, 4.0f * s);
    m_uiDrawList.addRectFilled(board, kBg);

    for (int r = 0; r < m_rows; ++r) {
        for (int c = 0; c < m_cols; ++c) {
            const Cell& cur = cell(c, r);
            const float x = lo.ox + c * cs + pad;
            const float y = lo.oy + r * cs + pad;
            const float w = cs - 2.0f * pad;
            const UiRect tile = UiRect::fromXYWH(x, y, w, w);

            if (!cur.revealed) {
                // Raised, covered tile.
                m_uiDrawList.addRoundRectFilled(tile, kCovered, 2.0f * s);
                m_uiDrawList.addBevel(tile, kHiLite, kShadow, 2.0f * s, bevel, /*inward=*/false);
                if (cur.flagged) {
                    // Flag: dark pole + red pennant.
                    const float px = x + w * 0.44f;
                    m_uiDrawList.addRectFilled(
                        UiRect::fromXYWH(px, y + w * 0.24f, std::max(1.0f, 1.5f * s), w * 0.52f),
                        {0.05f, 0.05f, 0.08f, 0.95f});
                    m_uiDrawList.addRoundRectFilled(
                        UiRect::fromXYWH(x + w * 0.22f, y + w * 0.22f, w * 0.24f, w * 0.20f),
                        kRed, 2.0f * s);
                }
                continue;
            }

            // Revealed, sunken tile.
            m_uiDrawList.addRoundRectFilled(tile, kRevealed, 2.0f * s);
            m_uiDrawList.addBevel(tile, kHiLite, kShadow, 2.0f * s, bevel, /*inward=*/true);

            const float cx = x + w * 0.5f;
            const float cy = y + w * 0.5f;
            if (cur.mine) {
                const bool boom = m_phase == Phase::Lost;
                if (boom)
                    m_uiDrawList.addRoundRectFilled(tile, {0.45f, 0.09f, 0.06f, 0.90f}, 2.0f * s);
                m_uiDrawList.addCircleFilled({cx, cy}, w * 0.26f, {0.06f, 0.06f, 0.09f, 1.0f});
                m_uiDrawList.addCircleFilled({cx - w * 0.08f, cy - w * 0.08f}, w * 0.07f,
                                             {0.85f, 0.87f, 0.92f, 0.85f});
            } else if (cur.adjacent > 0) {
                char n[2] = {static_cast<char>('0' + cur.adjacent), '\0'};
                textCentered(m_uiFontBold, n, cx, cy, numberColor(cur.adjacent));
            }
        }
    }

    m_uiDrawList.addRoundRect(board, kBorder, 2.0f * s, s);
}

void MinesweeperApp::drawOverlay(float fw, float fh, float s) {
    const float pulse = std::sin(m_pulse * 2.2f) * 0.5f + 0.5f;
    m_uiDrawList.addRectFilled(UiRect::fromXYWH(0, 0, fw, fh), {0.0f, 0.0f, 0.0f, 0.42f});

    const float cardW = std::min(360.0f * s, fw * 0.86f);
    const float cardH = std::min(150.0f * s, fh * 0.45f);
    const float cardX = (fw - cardW) * 0.5f;
    const float cardY = fh * 0.36f;
    const UiRect card = UiRect::fromXYWH(cardX, cardY, cardW, cardH);
    m_uiDrawList.addDropShadow(card, {0, 0, 0, 0.60f}, 12.0f * s, 4.0f * s, 6.0f * s);
    m_uiDrawList.addRoundRectFilled(card, kToolbar, 6.0f * s);
    m_uiDrawList.addRoundRect(card, kBorder, 6.0f * s, s);

    const bool won = m_phase == Phase::Won;
    m_uiDrawList.addRoundRectFilled(
        UiRect::fromXYWH(cardX, cardY, cardW, 3.0f * s), won ? kGreen : kRed, 3.0f * s);

    const float cx = cardX + cardW * 0.5f;
    textCentered(m_uiFontBold, won ? "FIELD CLEARED" : "MINE DETONATED",
                 cx, cardY + 40.0f * s, won ? kGreen : kRed);

    char buf[64];
    std::snprintf(buf, sizeof(buf), "Time  %03d", std::min(999, static_cast<int>(m_elapsed)));
    textCentered(m_uiFont, buf, cx, cardY + 74.0f * s, kText);

    textCentered(m_uiFont, "Press R or 'New Game' to play again", cx, cardY + 104.0f * s,
                 {kDim.r, kDim.g, kDim.b, 0.5f + pulse * 0.5f});
}

// ─── GameApp overrides ───────────────────────────────────────────────────────

bool MinesweeperApp::onInit() {
    const float s = contentScale();

    if (!loadFonts(
            resolveAssetPath("assets/fonts/Inter-Regular.ttf"),
            resolveAssetPath("assets/fonts/Inter-Bold.ttf"),
            resolveAssetPath("assets/fonts/Inter-Italic.ttf"),
            resolveAssetPath("assets/fonts/JetBrainsMono-Regular.ttf"),
            std::round(15.0f * s), std::round(14.0f * s))) {
        return false;
    }

    // Retained toolbar: a passthrough root holding ui::Button widgets. Rather than
    // hand each button a lambda, we tag them with slot names and let SlotRegistry
    // wire the activated signals to game actions in one pass — the JSON-authored
    // wiring path, exercised from C++.
    auto root = std::make_unique<Widget>();
    root->mousePassthrough = true;
    Widget* rootRaw = m_uiContext.setRoot(std::move(root));

    const float btnY = 14.0f * s;
    const float btnH = 32.0f * s;
    auto addButton = [&](const char* label, const char* slot, float x, float w) {
        auto b = std::make_unique<Button>(&m_uiFontBold, label, nullptr);
        b->slotName        = slot;
        b->cornerRadiusPx  = 3.0f * s;
        b->glowSizePx      = 10.0f * s;
        b->setRect(UiRect::fromXYWH(x, btnY, w, btnH));
        rootRaw->addChild(std::move(b));
    };

    const float x0 = 232.0f * s;   // clear of the "MINESWEEPER" title
    const float gap = 8.0f * s;
    addButton("New Game",     "new_game",   x0,                        104.0f * s);
    addButton("Beginner",     "diff_easy",  x0 + (104.0f + 10.0f) * s + gap, 104.0f * s);
    addButton("Intermediate", "diff_med",   x0 + (218.0f + 20.0f) * s + gap, 126.0f * s);
    addButton("Expert",       "diff_hard",  x0 + (354.0f + 30.0f) * s + gap, 104.0f * s);

    SlotRegistry reg;
    reg.on("new_game",  [this] { resetGame(); });
    reg.on("diff_easy", [this] { setDifficulty(Difficulty::Beginner); });
    reg.on("diff_med",  [this] { setDifficulty(Difficulty::Intermediate); });
    reg.on("diff_hard", [this] { setDifficulty(Difficulty::Expert); });
    reg.wire(*rootRaw);

    setDifficulty(Difficulty::Beginner);
    return true;
}

void MinesweeperApp::onTick(float dt) {
    m_pulse += dt;
    if (m_phase == Phase::Playing && m_minesPlaced) m_elapsed += dt;

    // Keyboard shortcuts (raw GLFW — the UI input snapshot carries only the mouse
    // and text codepoints, so games reach for glfwGetKey for discrete keys).
    auto key = [&](int k) { return glfwGetKey(m_window, k) == GLFW_PRESS; };
    const bool rDown = key(GLFW_KEY_R);
    const bool d1    = key(GLFW_KEY_1);
    const bool d2    = key(GLFW_KEY_2);
    const bool d3    = key(GLFW_KEY_3);
    if (rDown && !m_prevR) resetGame();
    if (d1 && !m_prev1) setDifficulty(Difficulty::Beginner);
    if (d2 && !m_prev2) setDifficulty(Difficulty::Intermediate);
    if (d3 && !m_prev3) setDifficulty(Difficulty::Expert);
    m_prevR = rDown; m_prev1 = d1; m_prev2 = d2; m_prev3 = d3;

    if (m_phase != Phase::Playing) return;
    // Let the toolbar claim clicks first; wantsMouse() is true whenever the cursor
    // is over any widget, so the board never fires under a button.
    if (m_uiContext.wantsMouse()) return;

    const bool left  = m_uiInput.button(UiMouseButton::Left).pressed;
    const bool right = m_uiInput.button(UiMouseButton::Right).pressed;
    if (!left && !right) return;

    const BoardLayout lo = computeLayout();
    const UiVec2 m = m_uiInput.mousePx;
    if (m.x < lo.ox || m.y < lo.oy) return;
    const int c = static_cast<int>((m.x - lo.ox) / lo.cell);
    const int r = static_cast<int>((m.y - lo.oy) / lo.cell);
    if (!inBounds(c, r)) return;

    if (left)       reveal(c, r);
    else if (right) toggleFlag(c, r);
}

void MinesweeperApp::onRender(float /*dt*/) {
    beginFrameDraw();

    const float s = contentScale();
    int fwi = 0, fhi = 0;
    framebufferSize(fwi, fhi);
    const float fw = static_cast<float>(fwi);
    const float fh = static_cast<float>(fhi);

    m_uiDrawList.addRectFilled(UiRect::fromXYWH(0, 0, fw, fh), kBg);

    const BoardLayout lo = computeLayout();
    drawToolbarBg(fw, s);   // toolbar buttons are appended by the widget tree in submitFrame
    drawHud(lo);
    drawBoard(lo);
    if (m_phase != Phase::Playing) drawOverlay(fw, fh, s);

    submitFrame(m_camera);
}

}  // namespace odai::games::minesweeper
