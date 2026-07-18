#include "tools/tween_demo/tween_demo_app.h"

#include "ui/widgets/button.h"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace odai::tools::tween_demo {

using namespace odai::ui;

namespace {

constexpr float kLaneDurationSec = 1.4f;

struct LaneSpec {
    Easing easing;
    const char* name;
    UiColor color;
};

// One lane per Easing value in animation.h, in declaration order.
constexpr std::array<LaneSpec, 8> kLaneSpecs{{
    {Easing::Linear,    "Linear",    UiColor{0.55f, 0.58f, 0.62f, 1.0f}},
    {Easing::EaseIn,    "EaseIn",    UiColor{0.36f, 0.55f, 0.82f, 1.0f}},
    {Easing::EaseOut,   "EaseOut",   UiColor{0.36f, 0.72f, 0.82f, 1.0f}},
    {Easing::EaseInOut, "EaseInOut", UiColor{0.40f, 0.78f, 0.52f, 1.0f}},
    {Easing::CubicIn,   "CubicIn",   UiColor{0.78f, 0.62f, 0.30f, 1.0f}},
    {Easing::CubicOut,  "CubicOut",  UiColor{0.86f, 0.50f, 0.28f, 1.0f}},
    {Easing::BackOut,   "BackOut",   UiColor{0.82f, 0.36f, 0.46f, 1.0f}},
    {Easing::Spring,    "Spring",    UiColor{0.68f, 0.44f, 0.82f, 1.0f}},
}};

UiColor randomColor(std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(0.30f, 0.95f);
    return UiColor{dist(rng), dist(rng), dist(rng), 1.0f};
}

}  // namespace

bool TweenDemoApp::onInit() {
    const float s = contentScale();
    if (!loadFonts(
            resolveAssetPath("assets/fonts/Inter-Regular.ttf"),
            resolveAssetPath("assets/fonts/Inter-Bold.ttf"),
            resolveAssetPath("assets/fonts/Inter-Italic.ttf"),
            resolveAssetPath("assets/fonts/Inter-Regular.ttf"),
            std::round(16.f * s), std::round(15.f * s))) {
        return false;
    }

    auto root = std::make_unique<Widget>();
    root->mousePassthrough = true;
    m_root = m_uiContext.setRoot(std::move(root));

    int fbW = 0, fbH = 0;
    framebufferSize(fbW, fbH);

    auto bg = std::make_unique<Panel>();
    bg->setRect(UiRect::fromXYWH(0.f, 0.f, static_cast<float>(fbW), static_cast<float>(fbH)));
    bg->background = UiColor{0.08f, 0.09f, 0.11f, 1.0f};
    m_root->addChild(std::move(bg));

    auto title = std::make_unique<Label>(m_uiFonts, "<b>odai_ui Tween Demo</b>");
    title->setRect(UiRect::fromXYWH(40.f * s, 26.f * s, 700.f * s, 30.f * s));
    title->color = UiColor{0.93f, 0.90f, 0.80f, 1.0f};
    m_root->addChild(std::move(title));

    auto fpsLabel = std::make_unique<Label>(m_uiFonts, "-- FPS");
    fpsLabel->setRect(UiRect::fromXYWH(static_cast<float>(fbW) - 340.f * s, 30.f * s, 300.f * s, 24.f * s));
    fpsLabel->align = UiTextAlign::Right;
    fpsLabel->color = UiColor{0.55f, 0.85f, 0.60f, 1.0f};
    m_frameStatsLabel = static_cast<Label*>(m_root->addChild(std::move(fpsLabel)));

    auto subtitle = std::make_unique<Label>(m_uiFonts,
        "Every Easing curve in animation.h, looping side by side. Randomize exercises "
        "ColorTween; Pop Card exercises RectTween + Sequence.");
    subtitle->setRect(UiRect::fromXYWH(40.f * s, 58.f * s, 900.f * s, 24.f * s));
    subtitle->color = UiColor{0.62f, 0.65f, 0.70f, 1.0f};
    m_root->addChild(std::move(subtitle));

    buildGallery(s);
    buildInteractiveSection(s);
    buildPopCard(s);

    return true;
}

void TweenDemoApp::buildGallery(float s) {
    constexpr float kGalleryTop = 108.f;
    constexpr float kRowH = 44.f;
    constexpr float kLabelW = 110.f;
    constexpr float kTrackX = 160.f;
    constexpr float kTrackW = 420.f;
    constexpr float kBoxSize = 32.f;

    for (std::size_t i = 0; i < kLaneSpecs.size(); ++i) {
        const LaneSpec& spec = kLaneSpecs[i];
        const float y = (kGalleryTop + static_cast<float>(i) * kRowH) * s;

        auto nameLabel = std::make_unique<Label>(m_uiFonts, spec.name);
        nameLabel->setRect(UiRect::fromXYWH(40.f * s, y + (kRowH * s - 20.f * s) * 0.5f,
                                             kLabelW * s, 20.f * s));
        nameLabel->color = UiColor{0.75f, 0.78f, 0.82f, 1.0f};
        m_root->addChild(std::move(nameLabel));

        auto track = std::make_unique<Panel>();
        track->setRect(UiRect::fromXYWH(kTrackX * s, y + (kRowH * s - 6.f * s) * 0.5f,
                                         kTrackW * s, 6.f * s));
        track->background = UiColor{1.0f, 1.0f, 1.0f, 0.08f};
        track->cornerRadiusPx = 3.f * s;
        m_root->addChild(std::move(track));

        auto box = std::make_unique<Panel>();
        box->cornerRadiusPx = 6.f * s;
        box->background = spec.color;
        Panel* boxPtr = static_cast<Panel*>(m_root->addChild(std::move(box)));

        GalleryLane& lane = m_gallery[i];
        lane.box = boxPtr;
        lane.easing = spec.easing;
        lane.y = y + (kRowH * s - kBoxSize * s) * 0.5f;
        lane.trackMinX = kTrackX * s;
        lane.trackMaxX = (kTrackX + kTrackW - kBoxSize) * s;
        lane.tween.snap(UiVec2{lane.trackMinX, lane.y});
        lane.tween.set(UiVec2{lane.trackMaxX, lane.y}, kLaneDurationSec, lane.easing);
        lane.box->setRect(UiRect::fromXYWH(lane.trackMinX, lane.y, kBoxSize * s, kBoxSize * s));
    }
}

void TweenDemoApp::buildInteractiveSection(float s) {
    const float sectionTop = (108.f + static_cast<float>(kLaneSpecs.size()) * 44.f + 30.f) * s;

    auto heading = std::make_unique<Label>(m_uiFonts, "<b>Interactive</b>");
    heading->setRect(UiRect::fromXYWH(40.f * s, sectionTop, 400.f * s, 24.f * s));
    heading->color = UiColor{0.93f, 0.90f, 0.80f, 1.0f};
    m_root->addChild(std::move(heading));

    const float row1 = sectionTop + 36.f * s;

    auto toggle = std::make_unique<Toggle>();
    toggle->setRect(UiRect::fromXYWH(40.f * s, row1, 52.f * s, 26.f * s));
    m_toggle = static_cast<Toggle*>(m_root->addChild(std::move(toggle)));

    auto toggleLabel = std::make_unique<Label>(m_uiFonts, "OFF — thumb Tween ticks via Widget::onTick");
    toggleLabel->setRect(UiRect::fromXYWH(102.f * s, row1 + 2.f * s, 400.f * s, 22.f * s));
    toggleLabel->color = UiColor{0.75f, 0.78f, 0.82f, 1.0f};
    m_toggleLabel = static_cast<Label*>(m_root->addChild(std::move(toggleLabel)));

    m_toggle->onChange = [this](bool checked) {
        m_toggleLabel->setText(checked ? "ON — thumb Tween ticks via Widget::onTick"
                                        : "OFF — thumb Tween ticks via Widget::onTick");
    };

    const float row2 = row1 + 44.f * s;

    auto bar = std::make_unique<ProgressBar>();
    bar->setRect(UiRect::fromXYWH(40.f * s, row2, 300.f * s, 20.f * s));
    bar->cornerRadiusPx = 10.f * s;
    bar->background = UiColor{1.0f, 1.0f, 1.0f, 0.08f};
    bar->foreground = UiColor{0.40f, 0.78f, 0.52f, 1.0f};
    bar->foregroundEnd = UiColor{0.36f, 0.55f, 0.82f, 1.0f};
    bar->value = 0.6f;
    m_progressBar = static_cast<ProgressBar*>(m_root->addChild(std::move(bar)));

    auto randomize = std::make_unique<Button>(&m_uiFontBold, "Randomize", [this] {
        std::uniform_real_distribution<float> valueDist(0.15f, 1.0f);
        m_progressBar->value = valueDist(m_rng);
        m_progressBar->foregroundAnim.set(randomColor(m_rng), 0.5f, Easing::EaseInOut);
        m_progressBar->foregroundEndAnim.set(randomColor(m_rng), 0.5f, Easing::EaseInOut);
    });
    randomize->setRect(UiRect::fromXYWH(360.f * s, row2 - 4.f * s, 140.f * s, 28.f * s));
    m_root->addChild(std::move(randomize));

    const float row3 = row2 + 44.f * s;
    auto popButton = std::make_unique<Button>(&m_uiFontBold, "Pop Card (Sequence)",
                                               [this] { showCard(); });
    popButton->setRect(UiRect::fromXYWH(40.f * s, row3, 220.f * s, 32.f * s));
    m_root->addChild(std::move(popButton));
}

void TweenDemoApp::buildPopCard(float s) {
    int fbW = 0, fbH = 0;
    framebufferSize(fbW, fbH);

    auto backdrop = std::make_unique<Panel>();
    backdrop->setRect(UiRect::fromXYWH(0.f, 0.f, static_cast<float>(fbW), static_cast<float>(fbH)));
    backdrop->background = UiColor{0.0f, 0.0f, 0.0f, 0.65f};
    backdrop->visible = false;
    backdrop->opacity = 0.0f;
    m_backdrop = static_cast<Panel*>(m_root->addChild(std::move(backdrop)));

    constexpr float kCardW = 420.f, kCardH = 220.f;
    const float cx = (static_cast<float>(fbW) - kCardW * s) * 0.5f;
    const float cy = (static_cast<float>(fbH) - kCardH * s) * 0.5f;
    m_cardOpenRect = UiRect::fromXYWH(cx, cy, kCardW * s, kCardH * s);
    // Same top-left as the open rect: repositionAndResize() only translates children
    // by the delta between successive top-left corners, so keeping it fixed means the
    // card's already-final-position children never move — only the clip window (see
    // clipContents below) grows, giving a clean "reveal" pop rather than a jump.
    m_cardClosedRect = UiRect::fromXYWH(cx, cy, 1.f, 1.f);

    auto card = std::make_unique<Panel>();
    card->styleCard(s, 0.97f);
    card->clipContents = true;
    Panel* cardPtr = static_cast<Panel*>(m_backdrop->addChild(std::move(card)));
    cardPtr->setRect(m_cardOpenRect);
    m_card = cardPtr;

    auto cardTitle = std::make_unique<Label>(m_uiFonts, "<b>RectTween + Sequence</b>");
    cardTitle->setRect(UiRect::fromXYWH(cx + 24.f * s, cy + 20.f * s, kCardW * s - 48.f * s, 28.f * s));
    cardTitle->color = UiColor{0.93f, 0.90f, 0.80f, 1.0f};
    m_card->addChild(std::move(cardTitle));

    auto cardBody = std::make_unique<Label>(m_uiFonts,
        "This card's rect is driven by a RectTween (BackOut in / CubicIn out). "
        "The backdrop's opacity is driven by a Sequence step, independently.");
    cardBody->setRect(UiRect::fromXYWH(cx + 24.f * s, cy + 56.f * s, kCardW * s - 48.f * s, 80.f * s));
    cardBody->color = UiColor{0.75f, 0.78f, 0.82f, 1.0f};
    m_card->addChild(std::move(cardBody));

    auto closeButton = std::make_unique<Button>(&m_uiFontBold, "Close", [this] { hideCard(); });
    closeButton->setRect(UiRect::fromXYWH(cx + 24.f * s, cy + kCardH * s - 56.f * s, 120.f * s, 32.f * s));
    m_card->addChild(std::move(closeButton));

    m_cardRectTween.snap(m_cardClosedRect);
    m_card->setRect(m_cardClosedRect);
}

void TweenDemoApp::showCard() {
    m_cardOpen = true;
    m_backdrop->visible = true;
    m_cardRectTween.set(m_cardOpenRect, 0.4f, Easing::BackOut);

    m_popSequence = Sequence{};
    m_popSequence.append(0.25f, [this](float t) { m_backdrop->opacity = t; });
}

void TweenDemoApp::hideCard() {
    m_cardOpen = false;
    m_cardRectTween.set(m_cardClosedRect, 0.22f, Easing::CubicIn);

    m_popSequence = Sequence{};
    m_popSequence.append(0.22f, [this](float t) { m_backdrop->opacity = 1.0f - t; },
                          [this] { m_backdrop->visible = false; });
}

void TweenDemoApp::tickGallery(float dt) {
    for (GalleryLane& lane : m_gallery) {
        lane.tween.update(dt);
        if (lane.tween.idle()) {
            const bool atRight = std::abs(lane.tween.current().x - lane.trackMaxX) <
                                 std::abs(lane.tween.current().x - lane.trackMinX);
            const float nextX = atRight ? lane.trackMinX : lane.trackMaxX;
            lane.tween.set(UiVec2{nextX, lane.y}, kLaneDurationSec, lane.easing);
        }
        const UiVec2 pos = lane.tween.current();
        lane.box->setRect(UiRect::fromXYWH(pos.x, pos.y, lane.box->rect().width(),
                                            lane.box->rect().height()));
    }
}

void TweenDemoApp::updateFrameStats(float dt) {
    const float ms = dt * 1000.0f;
    if (m_fpsWindowFrames == 0) {
        m_fpsWindowMinMs = ms;
        m_fpsWindowMaxMs = ms;
    } else {
        m_fpsWindowMinMs = std::min(m_fpsWindowMinMs, ms);
        m_fpsWindowMaxMs = std::max(m_fpsWindowMaxMs, ms);
    }
    ++m_fpsWindowFrames;
    m_fpsWindowElapsed += dt;

    // Half-second window: frequent enough to catch a stutter, coarse enough that
    // the label itself isn't unreadable noise.
    constexpr float kWindowSec = 0.5f;
    if (m_fpsWindowElapsed < kWindowSec || m_frameStatsLabel == nullptr) {
        return;
    }
    const float avgMs = (m_fpsWindowElapsed / static_cast<float>(m_fpsWindowFrames)) * 1000.0f;
    const float fps = static_cast<float>(m_fpsWindowFrames) / m_fpsWindowElapsed;
    char buf[128];
    std::snprintf(buf, sizeof(buf), "%.0f FPS  (avg %.1f / min %.1f / max %.1f ms)",
                  fps, avgMs, m_fpsWindowMinMs, m_fpsWindowMaxMs);
    m_frameStatsLabel->setText(buf);
    m_fpsWindowElapsed = 0.0f;
    m_fpsWindowFrames = 0;
}

void TweenDemoApp::onTick(float dt) {
    if (glfwGetKey(m_window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(m_window, GLFW_TRUE);
    }

    updateFrameStats(dt);
    tickGallery(dt);

    m_cardRectTween.update(dt);
    if (m_card != nullptr) {
        m_card->repositionAndResize(m_cardRectTween.current());
    }
    m_popSequence.update(dt);
}

void TweenDemoApp::onRender(float /*dt*/) {
    beginFrameDraw();
    render::CameraPose camera{};
    submitFrame(camera);
}

}  // namespace odai::tools::tween_demo
