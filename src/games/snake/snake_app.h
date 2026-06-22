#pragma once
#include "engine/game_app.h"
#include "render/renderer_types.h"
#include <deque>
#include <utility>

namespace odai::games::snake {

using Cell = std::pair<int, int>;

enum class Dir   { Up, Down, Left, Right };
enum class State { Title, Playing, Paused, GameOver };

class SnakeApp : public engine::GameApp {
protected:
    bool onInit()           override;
    void onTick(float dt)   override;
    void onRender(float dt) override;

private:
    void resetGame();
    void step();
    void spawnFood();
    uint32_t rng() { return (m_rng = m_rng * 1664525u + 1013904223u); }
    float moveInterval() const { return std::max(0.07f, 0.20f - (m_level - 1) * 0.013f); }

    void drawBoard  (float bx, float by, float cellSz, float s);
    void drawHud    (float hx, float hy, float hw, float hh, float s);
    void drawOverlay(float fw, float fh, float s);

    static constexpr int kGridW = 28;
    static constexpr int kGridH = 20;

    std::deque<Cell> m_body;
    Cell    m_food       = {kGridW / 2, kGridH / 4};
    Dir     m_dir        = Dir::Right;
    Dir     m_nextDir    = Dir::Right;
    State   m_state      = State::Title;

    int     m_score      = 0;
    int     m_hiScore    = 0;
    int     m_level      = 1;
    int     m_foodsEaten = 0;

    float   m_moveTimer  = 0.0f;
    float   m_pulse      = 0.0f;

    bool    m_prevEnter  = false;
    bool    m_prevP      = false;

    uint32_t         m_rng = 0xC0FFEE17u;
    render::CameraPose m_camera{};
};

} // namespace odai::games::snake
