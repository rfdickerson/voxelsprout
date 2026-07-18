#pragma once

#include "engine/game_app.h"
#include "render/renderer_types.h"
#include "ui/ui_types.h"

#include <cstdint>
#include <string_view>
#include <vector>

namespace odai::games::minesweeper {

enum class Phase      { Playing, Won, Lost };
enum class Difficulty { Beginner, Intermediate, Expert };

struct Cell {
    bool         mine     = false;
    bool         revealed = false;
    bool         flagged  = false;
    std::uint8_t adjacent = 0;  // mine count in the 8-neighbourhood
};

// A compact Minesweeper built on the headless UI stack. The board (covered/
// revealed tiles, numbers, flags, mines) is emitted every frame to the immediate
// draw list, exactly like Snake's playfield. Unlike Snake, the chrome is a real
// retained widget tree: the toolbar buttons are ui::Button widgets whose
// activated signals are wired to game actions through ui::SlotRegistry, and the
// board itself is driven by the ui::UiInput mouse snapshot rather than raw GLFW.
// UI-only, so it opts into wantsMinimalRendering().
class MinesweeperApp : public engine::GameApp {
protected:
    bool onInit()           override;
    void onTick(float dt)   override;
    void onRender(float dt) override;

    bool wantsMinimalRendering() const override { return true; }

private:
    // Board geometry, recomputed each frame purely from the framebuffer size so
    // onTick (hit-testing) and onRender (drawing) always agree on the grid rect.
    struct BoardLayout {
        float ox   = 0.0f;  // grid top-left x, pixels
        float oy   = 0.0f;  // grid top-left y, pixels
        float cell = 0.0f;  // cell side, pixels
        float s    = 1.0f;  // DPI scale
    };
    [[nodiscard]] BoardLayout computeLayout() const;

    // ── Rules ────────────────────────────────────────────────────────────────
    void setDifficulty(Difficulty d);
    void resetGame();
    void placeMines(int safeC, int safeR);  // first-click-safe placement
    void reveal(int c, int r);              // flood-fills empty regions
    void toggleFlag(int c, int r);
    void revealAllMines();
    void checkWin();

    [[nodiscard]] bool inBounds(int c, int r) const {
        return c >= 0 && c < m_cols && r >= 0 && r < m_rows;
    }
    Cell& cell(int c, int r) {
        return m_cells[static_cast<std::size_t>(r) * m_cols + c];
    }
    [[nodiscard]] const Cell& cell(int c, int r) const {
        return m_cells[static_cast<std::size_t>(r) * m_cols + c];
    }
    std::uint32_t rng() { return (m_rng = m_rng * 1664525u + 1013904223u); }

    // ── Drawing ──────────────────────────────────────────────────────────────
    void drawToolbarBg(float fw, float s);
    void drawHud(const BoardLayout& lo);
    void drawBoard(const BoardLayout& lo);
    void drawOverlay(float fw, float fh, float s);
    void textCentered(const ui::Font& f, std::string_view str, float cx, float cy,
                      const ui::UiColor& col);

    // ── State ────────────────────────────────────────────────────────────────
    Difficulty m_difficulty = Difficulty::Beginner;
    int        m_cols  = 9;
    int        m_rows  = 9;
    int        m_mines = 10;

    std::vector<Cell> m_cells;
    Phase m_phase         = Phase::Playing;
    bool  m_minesPlaced   = false;  // deferred until the first reveal
    int   m_flagsPlaced   = 0;
    int   m_revealedCount = 0;
    float m_elapsed       = 0.0f;   // seconds since the first reveal
    float m_pulse         = 0.0f;

    // Rising-edge latches for the keyboard shortcuts (R / 1 / 2 / 3).
    bool m_prevR = false, m_prev1 = false, m_prev2 = false, m_prev3 = false;

    std::uint32_t      m_rng = 0xD15EA5Eu;
    render::CameraPose m_camera{};
};

}  // namespace odai::games::minesweeper
