#pragma once

#include "engine/game_app.h"
#include "ui/animation.h"
#include "ui/ui_types.h"
#include "ui/widget.h"
#include "ui/widgets/context_menu.h"
#include "ui/widgets/donut_chart.h"
#include "ui/widgets/dropdown.h"
#include "ui/widgets/label.h"
#include "ui/widgets/line_chart.h"
#include "ui/widgets/modal.h"
#include "ui/widgets/panel.h"
#include "ui/widgets/progress_bar.h"
#include "ui/widgets/radio_button.h"
#include "ui/widgets/repeater.h"
#include "ui/widgets/rich_text_view.h"
#include "ui/widgets/scroll_view.h"
#include "ui/widgets/slider.h"
#include "ui/widgets/spinner.h"
#include "ui/widgets/stat_badge.h"
#include "ui/widgets/tab_bar.h"
#include "ui/widgets/text_box.h"
#include "ui/widgets/toast.h"
#include "ui/widgets/toggle.h"
#include "ui/widgets/window.h"

#include <array>
#include <random>

// Design-system / component-library showcase: every reusable primitive widget
// under src/ui/widgets/ (buttons, inputs, panels, charts, overlays, layout),
// every Easing curve + tween type in src/ui/animation.h, and the raw visual-
// effect draw calls on UiDrawList (shadow, glow, bevel, gradient), organized
// into tabs. Game-specific compound panels (advisors_panel, resource_bar_panel,
// etc., which need simulated game data) are intentionally out of scope — this
// is the component-library layer, not a game-screen mockup.
namespace odai::tools::design_system_demo {

class DesignSystemDemoApp : public engine::GameApp {
protected:
    bool onInit() override;
    void onTick(float dt) override;
    void onRender(float dt) override;
    // Pure UI showcase: no 3D scene, so skip the pipe/imported/sky-cloud/water/
    // grass, SSAO, and hex-terrain pipelines the renderer otherwise builds.
    bool wantsMinimalRendering() const override { return true; }

private:
    // One easing-curve lane in the Animation tab's gallery (same idea as
    // tools/tween_demo): a box that ping-pongs between the ends of its track,
    // re-targeting with the lane's own Easing every time it goes idle.
    struct GalleryLane {
        ui::Panel* box = nullptr;
        ui::Vec2Tween tween;
        ui::Easing easing = ui::Easing::Linear;
        float trackMinX = 0.0f;
        float trackMaxX = 0.0f;
        float y = 0.0f;
    };

    // --- Chrome: title, tab bar, per-tab page containers ----------------------
    void buildChrome(float s);
    ui::Widget* addPage();

    // --- Tab content builders --------------------------------------------------
    void buildButtonsTab(ui::Widget* page, float s);
    void buildInputsTab(ui::Widget* page, float s);
    void buildPanelsTab(ui::Widget* page, float s);
    void buildEffectsTab(ui::Widget* page, float s);
    void buildAnimationTab(ui::Widget* page, float s);
    void buildDataFeedbackTab(ui::Widget* page, float s);
    void buildLayoutTab(ui::Widget* page, float s);

    void tickGallery(float dt);
    void showPopCard();
    void hidePopCard();
    void updateFrameStats(float dt);

    ui::Widget*  m_root = nullptr;
    ui::TabBar*  m_tabBar = nullptr;
    std::vector<ui::Widget*> m_pages;
    ui::Label*   m_frameStatsLabel = nullptr;
    float m_fpsWindowElapsed = 0.0f;
    int   m_fpsWindowFrames  = 0;
    float m_fpsWindowMinMs   = 0.0f;
    float m_fpsWindowMaxMs   = 0.0f;

    // Root-level overlays: shared across tabs, always ticked/drawn above them.
    ui::ToastManager* m_toastManager = nullptr;
    ui::Modal*        m_modal = nullptr;
    ui::ContextMenu*  m_contextMenu = nullptr;
    ui::Window*       m_demoWindow = nullptr;

    // Buttons tab.
    ui::ButtonGroup m_radioGroup;

    // Animation tab: easing gallery + RectTween/Sequence pop card.
    std::array<GalleryLane, 8> m_gallery{};
    ui::Panel*    m_popBackdrop = nullptr;
    ui::Panel*    m_popCard     = nullptr;
    ui::RectTween m_popRectTween;
    ui::Sequence  m_popSequence;
    ui::UiRect    m_popOpenRect{};
    ui::UiRect    m_popClosedRect{};

    std::mt19937 m_rng{std::random_device{}()};
};

}  // namespace odai::tools::design_system_demo
