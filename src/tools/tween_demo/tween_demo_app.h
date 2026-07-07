#pragma once

#include "engine/game_app.h"
#include "ui/animation.h"
#include "ui/ui_types.h"
#include "ui/widget.h"
#include "ui/widgets/label.h"
#include "ui/widgets/panel.h"
#include "ui/widgets/progress_bar.h"
#include "ui/widgets/toggle.h"

#include <array>
#include <random>

// Standalone showcase for src/ui/animation.h: a gallery of every Easing curve
// animating side by side, plus a Toggle/ProgressBar pair and a Sequence +
// RectTween "pop card" — nothing here depends on the rest of the game.
namespace odai::tools::tween_demo {

class TweenDemoApp : public engine::GameApp {
protected:
    bool onInit() override;
    void onTick(float dt) override;
    void onRender(float dt) override;

private:
    // One easing-curve lane in the gallery: a small square that ping-pongs
    // between the left and right ends of its track, re-targeting (and flipping
    // direction) every time it goes idle.
    struct GalleryLane {
        ui::Panel* box = nullptr;
        ui::Vec2Tween tween;
        ui::Easing easing = ui::Easing::Linear;
        float trackMinX = 0.0f;
        float trackMaxX = 0.0f;
        float y = 0.0f;
    };

    void buildGallery(float s);
    void buildInteractiveSection(float s);
    void buildPopCard(float s);

    void tickGallery(float dt);

    // Opens/closes the pop card via a Sequence: the backdrop fades in/out over
    // its full length while the card's RectTween pops/collapses in parallel.
    void showCard();
    void hideCard();

    // Rolling frame-time window (reset every kFrameStatsWindowSec): updates the
    // on-screen FPS/ms readout so the demo makes any stutter visible rather than
    // asking you to take smoothness on faith.
    void updateFrameStats(float dt);

    ui::Widget* m_root = nullptr;
    ui::Label*  m_frameStatsLabel = nullptr;
    float m_fpsWindowElapsed = 0.0f;
    int   m_fpsWindowFrames  = 0;
    float m_fpsWindowMinMs   = 0.0f;
    float m_fpsWindowMaxMs   = 0.0f;

    std::array<GalleryLane, 8> m_gallery{};

    ui::ProgressBar* m_progressBar = nullptr;
    ui::Toggle*      m_toggle      = nullptr;
    ui::Label*       m_toggleLabel = nullptr;

    ui::Panel* m_backdrop = nullptr;
    ui::Panel* m_card     = nullptr;
    ui::RectTween m_cardRectTween;
    ui::Sequence  m_popSequence;
    ui::UiRect    m_cardOpenRect{};
    ui::UiRect    m_cardClosedRect{};
    bool          m_cardOpen = false;

    std::mt19937 m_rng{std::random_device{}()};
};

}  // namespace odai::tools::tween_demo
