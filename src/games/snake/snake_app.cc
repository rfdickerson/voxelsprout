#include "games/snake/snake_app.h"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace odai::games::snake {

using namespace ui;

// ─── Palette (same family as SWTOR) ─────────────────────────────────────────
static constexpr UiColor kBg     {0.020f, 0.030f, 0.055f, 1.00f};
static constexpr UiColor kPanel  {0.047f, 0.078f, 0.125f, 0.90f};
static constexpr UiColor kBorder {0.847f, 0.482f, 0.110f, 0.50f};
static constexpr UiColor kText   {0.784f, 0.831f, 0.909f, 1.00f};
static constexpr UiColor kDim    {0.376f, 0.471f, 0.596f, 1.00f};
static constexpr UiColor kGold   {0.910f, 0.565f, 0.188f, 1.00f};
static constexpr UiColor kRed    {0.847f, 0.173f, 0.110f, 1.00f};
static constexpr UiColor kGreen  {0.133f, 0.816f, 0.275f, 1.00f};
static constexpr UiColor kSnakeH {0.22f,  0.90f,  0.65f,  1.00f};  // head (bright teal)
static constexpr UiColor kSnakeT {0.07f,  0.32f,  0.22f,  1.00f};  // tail (dark teal)
static constexpr UiColor kFood   {0.90f,  0.22f,  0.10f,  0.95f};  // food (red-orange)
static constexpr UiColor kBoardBg{0.020f, 0.038f, 0.065f, 1.00f};
static constexpr UiColor kGridLn {0.062f, 0.100f, 0.165f, 1.00f};

static UiColor lerpColor(const UiColor& a, const UiColor& b, float t) {
    return {a.r+(b.r-a.r)*t, a.g+(b.g-a.g)*t, a.b+(b.b-a.b)*t, a.a+(b.a-a.a)*t};
}

// ─── Game logic ─────────────────────────────────────────────────────────────

void SnakeApp::resetGame() {
    m_body.clear();
    const int sy = kGridH / 2;
    m_body.push_back({kGridW / 2 - 2, sy});  // tail  (oldest, index 0)
    m_body.push_back({kGridW / 2 - 1, sy});
    m_body.push_back({kGridW / 2,     sy});  // head  (newest, back())
    m_dir        = Dir::Right;
    m_nextDir    = Dir::Right;
    m_score      = 0;
    m_level      = 1;
    m_foodsEaten = 0;
    m_moveTimer  = 0.0f;
    spawnFood();
}

void SnakeApp::spawnFood() {
    bool occupied[kGridW][kGridH] = {};
    for (auto [bx, by] : m_body)
        occupied[bx][by] = true;

    int free = 0;
    for (int x = 0; x < kGridW; ++x)
        for (int y = 0; y < kGridH; ++y)
            if (!occupied[x][y]) ++free;

    if (free == 0) { m_state = State::GameOver; return; }

    int pick = static_cast<int>(rng() % static_cast<uint32_t>(free));
    for (int x = 0; x < kGridW; ++x)
        for (int y = 0; y < kGridH; ++y)
            if (!occupied[x][y] && pick-- == 0) { m_food = {x, y}; return; }
}

void SnakeApp::step() {
    m_dir = m_nextDir;
    auto [hx, hy] = m_body.back();
    int nx = hx, ny = hy;
    switch (m_dir) {
        case Dir::Up:    ny--; break;
        case Dir::Down:  ny++; break;
        case Dir::Left:  nx--; break;
        case Dir::Right: nx++; break;
    }

    // Wall collision
    if (nx < 0 || nx >= kGridW || ny < 0 || ny >= kGridH) {
        if (m_score > m_hiScore) m_hiScore = m_score;
        m_state = State::GameOver;
        return;
    }
    // Self collision (skip the tail — it moves away this same step)
    const int bodyEnd = static_cast<int>(m_body.size()) - 1;
    for (int i = 1; i < bodyEnd; ++i) {
        if (m_body[i].first == nx && m_body[i].second == ny) {
            if (m_score > m_hiScore) m_hiScore = m_score;
            m_state = State::GameOver;
            return;
        }
    }

    m_body.push_back({nx, ny});

    if (nx == m_food.first && ny == m_food.second) {
        ++m_foodsEaten;
        m_score  += 10 * m_level;
        m_level   = 1 + m_foodsEaten / 5;
        if (m_score > m_hiScore) m_hiScore = m_score;
        spawnFood();
        // Don't pop_front — snake grows
    } else {
        m_body.pop_front();
    }
}

// ─── Draw: game board ────────────────────────────────────────────────────────

void SnakeApp::drawBoard(float bx, float by, float cellSz, float s) {
    const float bw = cellSz * kGridW;
    const float bh = cellSz * kGridH;
    const UiRect board = UiRect::fromXYWH(bx, by, bw, bh);

    m_uiDrawList.addRectFilled(board, kBoardBg);

    // Subtle grid lines
    for (int gx = 1; gx < kGridW; ++gx)
        m_uiDrawList.addRectFilled(
            UiRect::fromXYWH(bx + gx * cellSz, by, s, bh), kGridLn);
    for (int gy = 1; gy < kGridH; ++gy)
        m_uiDrawList.addRectFilled(
            UiRect::fromXYWH(bx, by + gy * cellSz, bw, s), kGridLn);

    // Food — pulsing circle with glow
    {
        const float pulse = std::sin(m_pulse * 4.0f) * 0.5f + 0.5f;
        const float fx = bx + m_food.first  * cellSz + cellSz * 0.5f;
        const float fy = by + m_food.second * cellSz + cellSz * 0.5f;
        const float fr = cellSz * (0.30f + pulse * 0.04f);
        m_uiDrawList.addCircleFilled({fx, fy}, fr * 1.9f, {0.90f,0.22f,0.10f,0.13f});
        m_uiDrawList.addCircleFilled({fx, fy}, fr,         kFood);
        // Shine
        m_uiDrawList.addCircleFilled(
            {fx - fr*0.28f, fy - fr*0.28f}, fr * 0.32f, {1.0f,0.90f,0.80f,0.55f});
    }

    // Snake segments: tail (index 0) → head (back()), drawn tail-first
    const float pad = 3.0f * s;
    const int n = static_cast<int>(m_body.size());
    for (int i = 0; i < n; ++i) {
        auto [gx, gy] = m_body[i];
        const float t   = (n > 1) ? static_cast<float>(i) / (n - 1) : 1.0f;
        const UiColor col = lerpColor(kSnakeT, kSnakeH, t);
        const float sx = bx + gx * cellSz + pad;
        const float sy = by + gy * cellSz + pad;
        const float sw = cellSz - 2.0f * pad;
        m_uiDrawList.addRoundRectFilled(UiRect::fromXYWH(sx, sy, sw, sw), col, 3.0f*s);
        // Sheen stripe across the top third
        m_uiDrawList.addRoundRectFilled(
            UiRect::fromXYWH(sx + pad, sy + pad*0.5f, sw - 2.0f*pad, sw * 0.30f),
            {1.0f, 1.0f, 1.0f, 0.07f + t * 0.08f}, 2.0f*s);
    }

    // Head eyes
    if (!m_body.empty()) {
        auto [hx, hy] = m_body.back();
        const float hcx = bx + hx * cellSz + cellSz * 0.5f;
        const float hcy = by + hy * cellSz + cellSz * 0.5f;
        const float eyeR   = 2.5f * s;
        const float eyeOff = cellSz * 0.22f;
        UiVec2 eye1, eye2;
        switch (m_dir) {
            case Dir::Right: eye1={hcx+eyeOff,hcy-eyeOff}; eye2={hcx+eyeOff,hcy+eyeOff}; break;
            case Dir::Left:  eye1={hcx-eyeOff,hcy-eyeOff}; eye2={hcx-eyeOff,hcy+eyeOff}; break;
            case Dir::Up:    eye1={hcx-eyeOff,hcy-eyeOff}; eye2={hcx+eyeOff,hcy-eyeOff}; break;
            case Dir::Down:  eye1={hcx-eyeOff,hcy+eyeOff}; eye2={hcx+eyeOff,hcy+eyeOff}; break;
        }
        m_uiDrawList.addCircleFilled(eye1, eyeR,        {0.0f,0.0f,0.0f,0.88f});
        m_uiDrawList.addCircleFilled(eye2, eyeR,        {0.0f,0.0f,0.0f,0.88f});
        m_uiDrawList.addCircleFilled({eye1.x+s,eye1.y-s}, eyeR*0.42f, {1.0f,1.0f,1.0f,0.88f});
        m_uiDrawList.addCircleFilled({eye2.x+s,eye2.y-s}, eyeR*0.42f, {1.0f,1.0f,1.0f,0.88f});
    }

    // Board border
    m_uiDrawList.addDropShadow(board, {0.0f,0.0f,0.0f,0.55f}, 8.0f*s, 2.0f*s, 4);
    m_uiDrawList.addRoundRect(board, kBorder, 2.0f*s, s);
}

// ─── Draw: HUD ───────────────────────────────────────────────────────────────

void SnakeApp::drawHud(float x, float y, float w, float h, float s) {
    const UiRect r = UiRect::fromXYWH(x, y, w, h);
    m_uiDrawList.addRoundRectFilled(r, kPanel, 4.0f*s);
    m_uiDrawList.addRectFilledVGradient(r, {1.0f,1.0f,1.0f,0.04f}, {0.0f,0.0f,0.0f,0.06f});
    m_uiDrawList.addRoundRect(r, kBorder, 4.0f*s, s);

    const float lh = m_uiFont.lineHeightPx();
    const float ty = y + (h - lh) * 0.5f;

    char buf[64];

    // Score (left)
    m_uiDrawList.addText(m_uiFont, "SCORE", {x + 12.0f*s, ty}, kDim);
    std::snprintf(buf, sizeof(buf), "%d", m_score);
    const float scoreX = x + 12.0f*s + m_uiFont.measureText("SCORE ") + 2.0f*s;
    m_uiDrawList.addText(m_uiFontBold, buf, {scoreX, ty}, kGold);

    // Level (centred)
    std::snprintf(buf, sizeof(buf), "LEVEL  %d", m_level);
    const float lw = m_uiFont.measureText(buf);
    m_uiDrawList.addText(m_uiFont, buf, {x + (w - lw) * 0.5f, ty}, kText);

    // Hi-score (right)
    std::snprintf(buf, sizeof(buf), "BEST  %d", m_hiScore);
    const float hw = m_uiFont.measureText(buf);
    m_uiDrawList.addText(m_uiFont, buf, {x + w - hw - 12.0f*s, ty}, kDim);
}

// ─── Draw: overlay (title / pause / game-over) ───────────────────────────────

void SnakeApp::drawOverlay(float fw, float fh, float s) {
    if (m_state == State::Playing) return;

    const float lh  = m_uiFont.lineHeightPx();
    const float lhB = m_uiFontBold.lineHeightPx();
    const float pulse = std::sin(m_pulse * 2.2f) * 0.5f + 0.5f;

    // Dim the background
    const float dimAlpha = (m_state == State::Paused) ? 0.38f : 0.55f;
    m_uiDrawList.addRectFilled(UiRect::fromXYWH(0,0,fw,fh), {0.0f,0.0f,0.0f,dimAlpha});

    if (m_state == State::Paused) {
        const char* txt = "PAUSED";
        const float tw  = m_uiFontBold.measureText(txt);
        const float ty  = fh * 0.45f;
        m_uiDrawList.addText(m_uiFontBold, txt, {(fw-tw)*0.5f + s, ty + s}, {0,0,0,0.75f});
        m_uiDrawList.addText(m_uiFontBold, txt, {(fw-tw)*0.5f,     ty},     kText);
        const char* hint = "P — resume";
        const float hw  = m_uiFont.measureText(hint);
        m_uiDrawList.addText(m_uiFont, hint, {(fw-hw)*0.5f, ty + lhB + 8.0f*s},
            {kDim.r, kDim.g, kDim.b, 0.6f + pulse*0.4f});
        return;
    }

    // Central card
    const float cardW = std::min(340.0f*s, fw * 0.86f);
    const float cardH = std::min(180.0f*s, fh * 0.50f);
    const float cardX = (fw - cardW) * 0.5f;
    const float cardY = fh * 0.33f;
    const UiRect card = UiRect::fromXYWH(cardX, cardY, cardW, cardH);
    m_uiDrawList.addDropShadow(card, {0,0,0,0.60f}, 12.0f*s, 4.0f*s, 4);
    m_uiDrawList.addRoundRectFilled(card, kPanel, 6.0f*s);
    m_uiDrawList.addRoundRect(card, kBorder, 6.0f*s, s);
    // Top accent stripe
    const UiColor stripe = (m_state == State::Title) ? kSnakeH : kRed;
    m_uiDrawList.addRoundRectFilled(
        UiRect::fromXYWH(cardX, cardY, cardW, 3.0f*s), stripe, 3.0f*s);

    const float cx = cardX + cardW * 0.5f;
    float lineY = cardY + 22.0f*s;

    if (m_state == State::Title) {
        const char* title = "SNAKE";
        const float tw = m_uiFontBold.measureText(title);
        m_uiDrawList.addText(m_uiFontBold, title, {cx - tw*0.5f + s, lineY + s}, {0,0,0,0.75f});
        m_uiDrawList.addText(m_uiFontBold, title, {cx - tw*0.5f,     lineY},     kSnakeH);
        lineY += lhB + 10.0f*s;

        const char* sub = "PRESS ENTER TO START";
        const float sw  = m_uiFont.measureText(sub);
        m_uiDrawList.addText(m_uiFont, sub, {cx - sw*0.5f, lineY},
            {kText.r, kText.g, kText.b, 0.5f + pulse*0.5f});
        lineY += lh + 18.0f*s;

        const char* ctrl = "WASD / Arrows — move     P — pause";
        const float cw   = m_uiFont.measureText(ctrl);
        m_uiDrawList.addText(m_uiFont, ctrl, {cx - cw*0.5f, lineY}, kDim);
    } else {  // GameOver
        const char* title = "GAME OVER";
        const float tw = m_uiFontBold.measureText(title);
        m_uiDrawList.addText(m_uiFontBold, title, {cx - tw*0.5f + s, lineY + s}, {0,0,0,0.75f});
        m_uiDrawList.addText(m_uiFontBold, title, {cx - tw*0.5f,     lineY},     kRed);
        lineY += lhB + 8.0f*s;

        char score[32];
        std::snprintf(score, sizeof(score), "Score:  %d", m_score);
        const float sw = m_uiFontBold.measureText(score);
        m_uiDrawList.addText(m_uiFontBold, score, {cx - sw*0.5f, lineY}, kGold);
        lineY += lhB + 14.0f*s;

        const char* sub = "PRESS ENTER TO PLAY AGAIN";
        const float sbW = m_uiFont.measureText(sub);
        m_uiDrawList.addText(m_uiFont, sub, {cx - sbW*0.5f, lineY},
            {kText.r, kText.g, kText.b, 0.5f + pulse*0.5f});
    }
}

// ─── GameApp overrides ───────────────────────────────────────────────────────

bool SnakeApp::onInit() {
    const float s = contentScale();

    if (!loadFonts(
            resolveAssetPath("assets/fonts/Inter-Regular.ttf"),
            resolveAssetPath("assets/fonts/Inter-Bold.ttf"),
            resolveAssetPath("assets/fonts/Inter-Italic.ttf"),
            resolveAssetPath("assets/fonts/JetBrainsMono-Regular.ttf"),
            std::round(15.0f * s), std::round(14.0f * s))) {
        return false;
    }
    return true;
}

void SnakeApp::onTick(float dt) {
    m_pulse += dt;

    auto key = [&](int k) { return glfwGetKey(m_window, k) == GLFW_PRESS; };

    const bool enterDown = key(GLFW_KEY_ENTER);
    const bool pDown     = key(GLFW_KEY_P);

    // Title / game-over: Enter starts a new game
    if (m_state == State::Title || m_state == State::GameOver) {
        if (enterDown && !m_prevEnter) { resetGame(); m_state = State::Playing; }
        m_prevEnter = enterDown;
        m_prevP     = pDown;
        return;
    }

    // P toggles pause
    if (pDown && !m_prevP)
        m_state = (m_state == State::Playing) ? State::Paused : State::Playing;
    m_prevEnter = enterDown;
    m_prevP     = pDown;

    if (m_state != State::Playing) return;

    // Collect direction change (prevent 180-degree reversal)
    if ((key(GLFW_KEY_UP)    || key(GLFW_KEY_W)) && m_dir != Dir::Down)  m_nextDir = Dir::Up;
    if ((key(GLFW_KEY_DOWN)  || key(GLFW_KEY_S)) && m_dir != Dir::Up)    m_nextDir = Dir::Down;
    if ((key(GLFW_KEY_LEFT)  || key(GLFW_KEY_A)) && m_dir != Dir::Right) m_nextDir = Dir::Left;
    if ((key(GLFW_KEY_RIGHT) || key(GLFW_KEY_D)) && m_dir != Dir::Left)  m_nextDir = Dir::Right;

    m_moveTimer += dt;
    if (m_moveTimer >= moveInterval()) {
        m_moveTimer -= moveInterval();
        step();
    }
}

void SnakeApp::onRender(float /*dt*/) {
    beginFrameDraw();

    const float s = contentScale();
    int fwi, fhi;
    framebufferSize(fwi, fhi);
    const float fw = static_cast<float>(fwi);
    const float fh = static_cast<float>(fhi);

    m_uiDrawList.addRectFilled(UiRect::fromXYWH(0, 0, fw, fh), kBg);

    const float cellSz = std::round(22.0f * s);
    const float boardW = cellSz * kGridW;
    const float boardH = cellSz * kGridH;
    const float hudH   = std::round(40.0f * s);
    const float gap    = std::round(6.0f * s);
    const float totalH = hudH + gap + boardH;
    const float bx     = std::round((fw - boardW) * 0.5f);
    const float by     = std::round((fh - totalH) * 0.5f + hudH + gap);

    drawHud(bx, by - hudH - gap, boardW, hudH, s);
    drawBoard(bx, by, cellSz, s);
    drawOverlay(fw, fh, s);

    submitFrame(m_camera);
}

} // namespace odai::games::snake
