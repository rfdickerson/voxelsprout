#include "tools/design_system_demo/design_system_demo_app.h"

#include "ui/widgets/button.h"
#include "ui/widgets/icon_button.h"
#include "ui/widgets/image.h"
#include "ui/widgets/spacer.h"
#include "ui/widgets/stack_layout.h"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>

namespace odai::tools::design_system_demo {

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

// Retro-OS / Windows-10 "flat UI" light theme — the same tokens Panel::styleRetroOS
// uses, extended to every widget in this app so the whole demo reads as one theme
// rather than just the Panel gallery swatch.
constexpr UiColor kRetroDesktopBg = UiColor{0.902f, 0.925f, 0.945f, 1.0f};  // #E6ECF1
constexpr UiColor kRetroFace      = UiColor{0.941f, 0.941f, 0.941f, 1.0f}; // #F0F0F0
constexpr UiColor kRetroWhite     = UiColor{1.000f, 1.000f, 1.000f, 1.0f};
constexpr UiColor kRetroBlue      = UiColor{0.000f, 0.471f, 0.843f, 1.0f}; // #0078D7
constexpr UiColor kRetroBlueHover = UiColor{0.063f, 0.404f, 0.710f, 1.0f}; // #106EBE
constexpr UiColor kRetroBluePress = UiColor{0.000f, 0.353f, 0.620f, 1.0f}; // #005A9E
constexpr UiColor kRetroBlueTint  = UiColor{0.831f, 0.918f, 0.973f, 1.0f}; // #D4EAF8
constexpr UiColor kRetroBlueTint2 = UiColor{0.671f, 0.847f, 0.949f, 1.0f}; // #ABD8F2
constexpr UiColor kRetroRed       = UiColor{0.910f, 0.067f, 0.137f, 1.0f}; // #E81123
constexpr UiColor kRetroGray      = UiColor{0.627f, 0.627f, 0.627f, 1.0f}; // #A0A0A0
constexpr UiColor kRetroDisable   = UiColor{0.878f, 0.878f, 0.878f, 1.0f}; // #E0E0E0
constexpr UiColor kRetroInk       = UiColor{0.114f, 0.114f, 0.114f, 1.0f}; // heading/body text
constexpr UiColor kRetroInkDim    = UiColor{0.361f, 0.361f, 0.361f, 1.0f}; // secondary text
constexpr UiColor kRetroTrack     = UiColor{0.0f, 0.0f, 0.0f, 0.08f};      // low-alpha ink on light bg
constexpr UiColor kRetroBacking   = UiColor{0.0f, 0.0f, 0.0f, 0.04f};

void applyRetroButtonStyle(Button& b, float s) {
    b.colorNormal       = kRetroFace;
    b.colorHover         = kRetroBlueTint;
    b.colorPressed       = kRetroBlueTint2;
    b.colorDisabled      = kRetroDisable;
    b.borderColor        = kRetroBlue;
    b.borderThicknessPx  = 1.0f * s;
    b.labelColor         = kRetroInk;
    b.cornerRadiusPx     = 4.0f * s;
}

void applyRetroIconButtonStyle(IconButton& b, float s) {
    b.colorNormal        = kRetroFace;
    b.colorHover          = kRetroBlueTint;
    b.colorPressed        = kRetroBlueTint2;
    b.colorDisabled       = kRetroDisable;
    b.borderColor         = kRetroBlue;
    b.borderHoverColor    = kRetroBlueHover;
    b.borderThicknessPx   = 1.0f * s;
    b.cornerRadiusPx      = 4.0f * s;
}

// Retheme a Window's chrome (used directly and via Modal::dialog()) to the
// Retro-OS palette: blue title bar, white body, red close glyph, small radius.
void applyRetroWindowStyle(Window& w, float s) {
    w.titleBarColor = kRetroBlue;
    w.bodyColor      = kRetroFace;
    w.borderColor    = kRetroBlue;
    w.titleColor     = kRetroWhite;
    w.closeColor      = kRetroRed;
    w.closeHoverColor = UiColor{0.780f, 0.055f, 0.114f, 1.0f};  // darker red
    w.cornerRadiusPx = 4.0f * s;
    w.frameBevelHighlightColor = UiColor{1.0f, 1.0f, 1.0f, 0.0f};
    w.frameBevelShadowColor    = UiColor{0.0f, 0.0f, 0.0f, 0.0f};
    w.toolbarBevelHighlightColor = UiColor{1.0f, 1.0f, 1.0f, 0.0f};
    w.toolbarBevelShadowColor    = UiColor{0.0f, 0.0f, 0.0f, 0.0f};
    w.showShadow    = true;
    w.shadowColor   = UiColor{0.0f, 0.0f, 0.0f, 0.18f};  // light — "avoid heavy shadows"
    w.shadowBlurPx  = 4.0f * s;
    w.shadowOffsetX = 0.0f;
    w.shadowOffsetY = 2.0f * s;
}

// A plain-string DataNode so the Layout tab's Repeater has something to bind
// to without pulling in JSON parsing or a game data model.
class StringDataNode : public DataNode {
public:
    explicit StringDataNode(std::string value) : value_(std::move(value)) {}
    std::string getString(std::string_view) const override { return value_; }
    float getFloat(std::string_view) const override { return 0.0f; }
    std::shared_ptr<DataNode> getChild(std::string_view) const override { return nullptr; }
    std::vector<std::shared_ptr<DataNode>> getList(std::string_view) const override { return {}; }
    std::string selfString() const override { return value_; }

private:
    std::string value_;
};

// Immediate-mode canvas for the Effects tab: hosts a raw UiDrawList paint
// callback so the demo can call addDropShadow/addRoundRectGlow/addBevel/
// gradients directly, the same way Panel/Button do internally, without a
// dedicated Widget subclass per effect.
class EffectSwatch : public Widget {
public:
    std::function<void(UiDrawList&, const UiRect&)> paint;
    void draw(UiDrawList& dl) const override {
        if (paint) paint(dl, rect_);
        drawChildren(dl);
    }
};

}  // namespace

bool DesignSystemDemoApp::onInit() {
    const float s = contentScale();
    if (!loadFonts(
            resolveAssetPath("assets/fonts/Inter-Regular.ttf"),
            resolveAssetPath("assets/fonts/Inter-Bold.ttf"),
            resolveAssetPath("assets/fonts/Inter-Italic.ttf"),
            resolveAssetPath("assets/fonts/JetBrainsMono-Regular.ttf"),
            std::round(15.f * s), std::round(14.f * s))) {
        return false;
    }

    auto root = std::make_unique<Widget>();
    root->mousePassthrough = true;
    m_root = m_uiContext.setRoot(std::move(root));

    buildChrome(s);

    int fbW = 0, fbH = 0;
    framebufferSize(fbW, fbH);

    // --- Root-level overlays: shared across every tab ------------------------
    auto toast = std::make_unique<ToastManager>(&m_uiFontBold);
    toast->setRect(UiRect::fromXYWH(static_cast<float>(fbW) - 300.f * s, 96.f * s,
                                    280.f * s, static_cast<float>(fbH) - 140.f * s));
    toast->background   = kRetroFace;
    toast->borderColor   = kRetroBlue;
    toast->textColor      = kRetroInk;
    toast->cornerRadiusPx = 4.0f * s;
    m_toastManager = static_cast<ToastManager*>(m_root->addChild(std::move(toast)));

    auto modal = std::make_unique<Modal>(&m_uiFontBold, "Modal", nullptr);
    modal->setRect(UiRect::fromXYWH(0.f, 0.f, static_cast<float>(fbW), static_cast<float>(fbH)));
    modal->dialogSizePx = UiVec2{420.f * s, 220.f * s};
    modal->dialog().padding = {18.f * s, 14.f * s};
    applyRetroWindowStyle(modal->dialog(), s);
    auto modalBody = std::make_unique<Label>(m_uiFonts,
        "Modal reuses Window for its chrome (title bar, close button, drag) "
        "behind a dimmed, input-blocking backdrop. Reopen it any time from "
        "the Data & Feedback tab.");
    modalBody->color = kRetroInk;
    modalBody->setRect(UiRect::fromXYWH(0.f, 4.f * s, 384.f * s, 120.f * s));
    modal->dialog().addChild(std::move(modalBody));
    m_modal = static_cast<Modal*>(m_root->addChild(std::move(modal)));

    auto ctxMenu = std::make_unique<ContextMenu>(&m_uiFontBold);
    ctxMenu->bgColor         = kRetroFace;
    ctxMenu->borderColor     = kRetroBlue;
    ctxMenu->itemHoverColor   = kRetroBlueTint;
    ctxMenu->textColor        = kRetroInk;
    ctxMenu->textDisabledColor = kRetroGray;
    ctxMenu->separatorColor   = kRetroGray;
    ctxMenu->items = {
        {"Rename", [this] { m_toastManager->push("", "Rename clicked"); }, false, true},
        {"Duplicate", [this] { m_toastManager->push("", "Duplicate clicked"); }, false, true},
        {"", nullptr, true, true},
        {"Delete (disabled)", nullptr, false, false},
    };
    m_contextMenu = static_cast<ContextMenu*>(m_root->addChild(std::move(ctxMenu)));

    // --- Tabs ------------------------------------------------------------------
    struct TabSpec {
        const char* label;
        void (DesignSystemDemoApp::*build)(Widget*, float);
    };
    const std::array<TabSpec, 7> tabs{{
        {"Buttons", &DesignSystemDemoApp::buildButtonsTab},
        {"Inputs", &DesignSystemDemoApp::buildInputsTab},
        {"Panels & Shadows", &DesignSystemDemoApp::buildPanelsTab},
        {"Effects", &DesignSystemDemoApp::buildEffectsTab},
        {"Animation", &DesignSystemDemoApp::buildAnimationTab},
        {"Data & Feedback", &DesignSystemDemoApp::buildDataFeedbackTab},
        {"Layout", &DesignSystemDemoApp::buildLayoutTab},
    }};

    for (const TabSpec& spec : tabs) {
        m_tabBar->addTab(spec.label);
        Widget* page = addPage();
        (this->*spec.build)(page, s);
    }
    m_pages[0]->visible = true;

    m_tabBar->onTabChanged = [this](int index) {
        for (std::size_t i = 0; i < m_pages.size(); ++i) {
            m_pages[i]->visible = (static_cast<int>(i) == index);
        }
    };

    return true;
}

Widget* DesignSystemDemoApp::addPage() {
    auto page = std::make_unique<Widget>();
    page->mousePassthrough = true;
    page->visible = false;
    Widget* raw = static_cast<Widget*>(m_root->addChild(std::move(page)));
    m_pages.push_back(raw);
    return raw;
}

void DesignSystemDemoApp::buildChrome(float s) {
    int fbW = 0, fbH = 0;
    framebufferSize(fbW, fbH);

    auto bg = std::make_unique<Panel>();
    bg->setRect(UiRect::fromXYWH(0.f, 0.f, static_cast<float>(fbW), static_cast<float>(fbH)));
    bg->background = kRetroDesktopBg;
    m_root->addChild(std::move(bg));

    auto title = std::make_unique<Label>(m_uiFonts, "<b>odai_ui Design System</b>");
    title->setRect(UiRect::fromXYWH(24.f * s, 14.f * s, 700.f * s, 26.f * s));
    title->color = kRetroInk;
    m_root->addChild(std::move(title));

    auto fpsLabel = std::make_unique<Label>(m_uiFonts, "-- FPS");
    fpsLabel->setRect(UiRect::fromXYWH(static_cast<float>(fbW) - 340.f * s, 16.f * s, 300.f * s, 22.f * s));
    fpsLabel->align = UiTextAlign::Right;
    fpsLabel->color = UiColor{0.031f, 0.475f, 0.212f, 1.0f};  // #087936 — readable green on light bg
    m_frameStatsLabel = static_cast<Label*>(m_root->addChild(std::move(fpsLabel)));

    auto subtitle = std::make_unique<Label>(m_uiFonts,
        "Every reusable widget, tween, and UiDrawList effect in one place. "
        "Game-specific compound panels (advisors, resource bar, ...) are out of scope.");
    subtitle->setRect(UiRect::fromXYWH(24.f * s, 42.f * s, 1000.f * s, 22.f * s));
    subtitle->color = kRetroInkDim;
    m_root->addChild(std::move(subtitle));

    auto tabBar = std::make_unique<TabBar>(&m_uiFontBold);
    tabBar->setRect(UiRect::fromXYWH(24.f * s, 72.f * s, static_cast<float>(fbW) - 48.f * s, 28.f * s));
    tabBar->activeTabColor   = kRetroFace;
    tabBar->inactiveTabColor = kRetroDesktopBg;
    tabBar->indicatorColor   = kRetroBlue;
    tabBar->textActiveColor   = kRetroInk;
    tabBar->textInactiveColor = kRetroInkDim;
    tabBar->dividerColor      = kRetroGray;
    m_tabBar = static_cast<TabBar*>(m_root->addChild(std::move(tabBar)));
}

// ---------------------------------------------------------------------------
// Buttons tab: Button (states + bevel + accent + hover/press tween), IconButton,
// Toggle, RadioButton + ButtonGroup.
// ---------------------------------------------------------------------------
void DesignSystemDemoApp::buildButtonsTab(Widget* page, float s) {
    const float x0 = 24.f * s;
    float y = 108.f * s;

    auto heading = std::make_unique<Label>(m_uiFonts, "<b>Button</b>");
    // 20px tall, matching every other section heading in the app (IconButton/
    // Toggle/RadioButton below, and every heading in Data & Feedback / Layout) —
    // this one was the sole 22px outlier.
    heading->setRect(UiRect::fromXYWH(x0, y, 400.f * s, 20.f * s));
    heading->color = kRetroInk;
    page->addChild(std::move(heading));
    y += 26.f * s;  // heading-to-content gap (standardized; see Data & Feedback / Layout).

    auto note = std::make_unique<Label>(m_uiFonts,
        "Fill color eases between normal/hover/pressed via a ColorTween (Button::onTick) "
        "instead of snapping. Hover and click the buttons below to feel it.");
    note->setRect(UiRect::fromXYWH(x0, y, 900.f * s, 20.f * s));
    note->color = kRetroInkDim;
    page->addChild(std::move(note));
    y += 34.f * s;

    auto clickCount = std::make_unique<Label>(m_uiFonts, "Clicks: 0");
    clickCount->color = kRetroInkDim;
    clickCount->setRect(UiRect::fromXYWH(x0, y + 40.f * s, 200.f * s, 20.f * s));
    Label* clickLabel = static_cast<Label*>(page->addChild(std::move(clickCount)));

    auto clicksPtr = std::make_shared<int>(0);
    float bx = x0;
    const float btnY = y;
    const float btnW = 168.f * s, btnH = 34.f * s, gap = 14.f * s;

    auto makeButton = [&](const char* label, std::function<void()> onClick) -> Button* {
        auto btn = std::make_unique<Button>(&m_uiFontBold, label, std::move(onClick));
        applyRetroButtonStyle(*btn, s);
        btn->setRect(UiRect::fromXYWH(bx, btnY, btnW, btnH));
        bx += btnW + gap;
        return static_cast<Button*>(page->addChild(std::move(btn)));
    };

    makeButton("Default", [clicksPtr, clickLabel] {
        ++(*clicksPtr);
        clickLabel->setText("Clicks: " + std::to_string(*clicksPtr));
    });

    Button* rounded = makeButton("Rounded + Glow", [clicksPtr, clickLabel] {
        ++(*clicksPtr);
        clickLabel->setText("Clicks: " + std::to_string(*clicksPtr));
    });
    rounded->cornerRadiusPx = 14.f * s;
    rounded->glowSizePx = 12.f * s;

    Button* bevel = makeButton("Bevel", [clicksPtr, clickLabel] {
        ++(*clicksPtr);
        clickLabel->setText("Clicks: " + std::to_string(*clicksPtr));
    });
    bevel->showBevel = true;

    Button* accent = makeButton("Accent Stripe", [clicksPtr, clickLabel] {
        ++(*clicksPtr);
        clickLabel->setText("Clicks: " + std::to_string(*clicksPtr));
    });
    accent->accentColor = kRetroRed;

    Button* disabled = makeButton("Disabled", [] {});
    disabled->setEnabled(false);

    y += 70.f * s;

    auto iconHeading = std::make_unique<Label>(m_uiFonts, "<b>IconButton</b>");
    iconHeading->setRect(UiRect::fromXYWH(x0, y, 400.f * s, 20.f * s));
    iconHeading->color = kRetroInk;
    page->addChild(std::move(iconHeading));
    y += 26.f * s;  // heading-to-content gap (standardized; see Data & Feedback / Layout).

    float ix = x0;
    for (int i = 0; i < 4; ++i) {
        auto icon = std::make_unique<IconButton>([] {});
        applyRetroIconButtonStyle(*icon, s);
        icon->setRect(UiRect::fromXYWH(ix, y, 40.f * s, 40.f * s));
        icon->textureId = kUiFontAtlas;  // reuse the font atlas texture as a visible fill
        icon->uvRect = UiRect{0.0f, 0.0f, 0.12f, 0.12f};
        icon->cornerRadiusPx = 6.f * s;
        if (i == 1) icon->glowSizePx = 10.f * s;
        if (i == 2) icon->showBevel = true;
        if (i == 3) icon->setEnabled(false);
        page->addChild(std::move(icon));
        ix += 52.f * s;
    }
    y += 60.f * s;

    auto toggleHeading = std::make_unique<Label>(m_uiFonts, "<b>Toggle</b>");
    toggleHeading->setRect(UiRect::fromXYWH(x0, y, 300.f * s, 20.f * s));
    toggleHeading->color = kRetroInk;
    page->addChild(std::move(toggleHeading));
    y += 26.f * s;  // heading-to-content gap (standardized; see Data & Feedback / Layout).

    auto toggle = std::make_unique<Toggle>();
    toggle->trackOn = kRetroBlue;
    toggle->setRect(UiRect::fromXYWH(x0, y, 52.f * s, 26.f * s));
    Toggle* togglePtr = static_cast<Toggle*>(page->addChild(std::move(toggle)));

    auto toggleLabel = std::make_unique<Label>(m_uiFonts, "OFF — thumb slides via a Tween on Widget::onTick");
    toggleLabel->setRect(UiRect::fromXYWH(x0 + 64.f * s, y + 2.f * s, 500.f * s, 22.f * s));
    toggleLabel->color = kRetroInkDim;
    Label* toggleLabelPtr = static_cast<Label*>(page->addChild(std::move(toggleLabel)));
    togglePtr->onChange = [toggleLabelPtr](bool checked) {
        toggleLabelPtr->setText(checked ? "ON — thumb slides via a Tween on Widget::onTick"
                                        : "OFF — thumb slides via a Tween on Widget::onTick");
    };
    y += 46.f * s;

    auto radioHeading = std::make_unique<Label>(m_uiFonts, "<b>RadioButton + ButtonGroup</b>");
    radioHeading->setRect(UiRect::fromXYWH(x0, y, 400.f * s, 20.f * s));
    radioHeading->color = kRetroInk;
    page->addChild(std::move(radioHeading));
    y += 26.f * s;  // heading-to-content gap (standardized; see Data & Feedback / Layout).

    auto radioStatus = std::make_unique<Label>(m_uiFonts, "Selected: Alpha");
    radioStatus->setRect(UiRect::fromXYWH(x0 + 260.f * s, y, 260.f * s, 22.f * s));
    radioStatus->color = kRetroInkDim;
    Label* radioStatusPtr = static_cast<Label*>(page->addChild(std::move(radioStatus)));

    const std::array<const char*, 3> radioNames{"Alpha", "Beta", "Gamma"};
    float rx = x0;
    for (std::size_t i = 0; i < radioNames.size(); ++i) {
        auto radio = std::make_unique<RadioButton>();
        radio->dotColor = kRetroBlue;
        radio->setRect(UiRect::fromXYWH(rx, y, 18.f * s, 18.f * s));
        radio->selected = (i == 0);
        RadioButton* radioPtr = static_cast<RadioButton*>(page->addChild(std::move(radio)));
        m_radioGroup.add(radioPtr);

        auto lbl = std::make_unique<Label>(m_uiFonts, radioNames[i]);
        lbl->setRect(UiRect::fromXYWH(rx + 24.f * s, y - 1.f * s, 80.f * s, 20.f * s));
        lbl->color = kRetroInkDim;
        page->addChild(std::move(lbl));

        rx += 110.f * s;
    }
    m_radioGroup.selectIndex(0);
    m_radioGroup.onChange = [radioStatusPtr, radioNames](int index) {
        if (index >= 0 && index < static_cast<int>(radioNames.size())) {
            radioStatusPtr->setText(std::string("Selected: ") + radioNames[index]);
        }
    };
}

// ---------------------------------------------------------------------------
// Inputs tab: Slider, Spinner, TextBox, ProgressBar (solid + gradient), Dropdown.
// ---------------------------------------------------------------------------
void DesignSystemDemoApp::buildInputsTab(Widget* page, float s) {
    const float x0 = 24.f * s;
    const float labelW = 140.f * s;
    float y = 108.f * s;

    // Intro note, matching the Buttons/Panels/Effects/Animation tabs so every
    // tab's first row of real content lands at the same rhythm (note, then
    // +34px gap) instead of this tab alone starting flush at the top.
    auto note = std::make_unique<Label>(m_uiFonts,
        "Every input primitive under src/ui/widgets/: Slider, Spinner, TextBox, "
        "ProgressBar (solid + gradient fill), and Dropdown.");
    note->setRect(UiRect::fromXYWH(x0, y, 1000.f * s, 20.f * s));
    note->color = kRetroInkDim;
    page->addChild(std::move(note));
    y += 34.f * s;

    auto addRowLabel = [&](const char* text) {
        auto lbl = std::make_unique<Label>(m_uiFonts, text);
        lbl->setRect(UiRect::fromXYWH(x0, y + 4.f * s, labelW, 22.f * s));
        lbl->color = kRetroInkDim;
        page->addChild(std::move(lbl));
    };

    // Slider.
    addRowLabel("Slider");
    auto slider = std::make_unique<Slider>();
    slider->trackColor = kRetroDisable;
    slider->fillColor   = kRetroBlue;
    slider->knobColor    = kRetroBlue;
    slider->knobHoverColor = kRetroBlueHover;
    slider->glowColor       = UiColor{0.0f, 0.471f, 0.843f, 0.30f};
    slider->setRect(UiRect::fromXYWH(x0 + labelW, y, 260.f * s, 24.f * s));
    slider->value = 0.4f;
    Slider* sliderPtr = static_cast<Slider*>(page->addChild(std::move(slider)));

    auto sliderValue = std::make_unique<Label>(m_uiFonts, "0.40");
    sliderValue->setRect(UiRect::fromXYWH(x0 + labelW + 276.f * s, y + 2.f * s, 80.f * s, 20.f * s));
    sliderValue->color = kRetroInkDim;
    Label* sliderValuePtr = static_cast<Label*>(page->addChild(std::move(sliderValue)));
    sliderPtr->onChange = [sliderValuePtr](float v) {
        char buf[16];
        std::snprintf(buf, sizeof(buf), "%.2f", v);
        sliderValuePtr->setText(buf);
    };
    y += 40.f * s;

    // Spinner.
    addRowLabel("Spinner");
    auto spinner = std::make_unique<Spinner>(&m_uiFontNumeric);
    spinner->fieldBg           = kRetroWhite;
    spinner->fieldBorderColor   = kRetroBlue;
    spinner->nubBg               = kRetroFace;
    spinner->nubHoverBg          = kRetroBlueTint;
    spinner->textColor            = kRetroInk;
    spinner->chevronColor         = kRetroBlue;
    spinner->setRect(UiRect::fromXYWH(x0 + labelW, y, 120.f * s, 26.f * s));
    spinner->minValue = 0.0;
    spinner->maxValue = 20.0;
    spinner->value = 5.0;
    page->addChild(std::move(spinner));
    y += 42.f * s;

    // TextBox.
    addRowLabel("TextBox");
    auto textBox = std::make_unique<TextBox>(&m_uiFont, "Type and press Enter...");
    textBox->background        = kRetroWhite;
    textBox->borderColor        = kRetroGray;
    textBox->borderFocusedColor  = kRetroBlue;
    textBox->textColor            = kRetroInk;
    textBox->placeholderColor     = kRetroGray;
    textBox->caretColor           = kRetroBlue;
    textBox->cornerRadiusPx = 4.f * s;
    textBox->setRect(UiRect::fromXYWH(x0 + labelW, y, 300.f * s, 30.f * s));
    TextBox* textBoxPtr = static_cast<TextBox*>(page->addChild(std::move(textBox)));
    textBoxPtr->onSubmit = [this, textBoxPtr] {
        if (!textBoxPtr->value().empty() && m_toastManager != nullptr) {
            m_toastManager->push("", "TextBox submitted: " + textBoxPtr->value());
        }
    };
    y += 46.f * s;

    // ProgressBar — solid fill.
    addRowLabel("ProgressBar (solid)");
    auto bar = std::make_unique<ProgressBar>();
    bar->setRect(UiRect::fromXYWH(x0 + labelW, y, 260.f * s, 18.f * s));
    bar->cornerRadiusPx = 9.f * s;
    bar->background = kRetroTrack;
    bar->foreground = kRetroBlue;
    bar->value = 0.55f;
    ProgressBar* barPtr = static_cast<ProgressBar*>(page->addChild(std::move(bar)));
    y += 34.f * s;

    // ProgressBar — gradient fill + Randomize button (exercises foregroundAnim).
    addRowLabel("ProgressBar (gradient)");
    auto gradBar = std::make_unique<ProgressBar>();
    gradBar->setRect(UiRect::fromXYWH(x0 + labelW, y, 260.f * s, 18.f * s));
    gradBar->cornerRadiusPx = 9.f * s;
    gradBar->background = kRetroTrack;
    gradBar->foreground = kRetroBlue;
    gradBar->foregroundEnd = kRetroRed;
    gradBar->value = 0.7f;
    ProgressBar* gradBarPtr = static_cast<ProgressBar*>(page->addChild(std::move(gradBar)));

    auto randomize = std::make_unique<Button>(&m_uiFontBold, "Randomize",
        [this, barPtr, gradBarPtr] {
            std::uniform_real_distribution<float> valueDist(0.15f, 1.0f);
            barPtr->value = valueDist(m_rng);
            gradBarPtr->value = valueDist(m_rng);
            gradBarPtr->foregroundAnim.set(randomColor(m_rng), 0.5f, Easing::EaseInOut);
            gradBarPtr->foregroundEndAnim.set(randomColor(m_rng), 0.5f, Easing::EaseInOut);
        });
    applyRetroButtonStyle(*randomize, s);
    randomize->setRect(UiRect::fromXYWH(x0 + labelW + 276.f * s, y - 8.f * s, 120.f * s, 32.f * s));
    page->addChild(std::move(randomize));
    y += 50.f * s;

    // Dropdown — placed last in this page's children so its popup layers above
    // every widget added before it (per Dropdown's class-comment contract).
    addRowLabel("Dropdown");
    auto dropdown = std::make_unique<Dropdown>(&m_uiFont);
    dropdown->headerBg        = kRetroWhite;
    dropdown->headerBorderColor = kRetroBlue;
    dropdown->headerHoverBg     = kRetroBlueTint;
    dropdown->popupBg            = kRetroWhite;
    dropdown->popupBorderColor    = kRetroBlue;
    dropdown->itemHoverColor       = kRetroBlueTint;
    dropdown->textColor             = kRetroInk;
    dropdown->chevronColor          = kRetroBlue;
    dropdown->cornerRadiusPx = 4.f * s;
    dropdown->setRect(UiRect::fromXYWH(x0 + labelW, y, 220.f * s, 28.f * s));
    dropdown->items = {"Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta"};
    dropdown->selectedIndex = 0;
    Dropdown* dropdownPtr = static_cast<Dropdown*>(page->addChild(std::move(dropdown)));

    auto dropdownValue = std::make_unique<Label>(m_uiFonts, "Selected: Alpha");
    dropdownValue->setRect(UiRect::fromXYWH(x0 + labelW + 236.f * s, y + 4.f * s, 240.f * s, 20.f * s));
    dropdownValue->color = kRetroInkDim;
    Label* dropdownValuePtr = static_cast<Label*>(page->addChild(std::move(dropdownValue)));
    dropdownPtr->onSelect = [dropdownPtr, dropdownValuePtr](int index) {
        if (index >= 0 && index < static_cast<int>(dropdownPtr->items.size())) {
            dropdownValuePtr->setText("Selected: " + dropdownPtr->items[static_cast<std::size_t>(index)]);
        }
    };
}

// ---------------------------------------------------------------------------
// Panels & Shadows tab: one card per Panel style preset, showShadow left as
// each preset's default so the corner-radius-aware shadow (styleSoft's
// generously-rounded corners in particular) hugs the fill correctly.
// ---------------------------------------------------------------------------
void DesignSystemDemoApp::buildPanelsTab(Widget* page, float s) {
    const float x0 = 24.f * s;
    float y = 108.f * s;

    auto note = std::make_unique<Label>(m_uiFonts,
        "Every Panel::style* preset. Drop shadows now match each panel's cornerRadiusPx "
        "(most visible on the generously-rounded 'Soft' and 'Gradient Card' presets).");
    note->setRect(UiRect::fromXYWH(x0, y, 1000.f * s, 20.f * s));
    note->color = kRetroInkDim;
    page->addChild(std::move(note));
    y += 34.f * s;

    struct StyleSpec {
        const char* name;
        std::function<void(Panel&, float)> apply;
    };
    const std::array<StyleSpec, 9> styles{{
        {"Ornate", [](Panel& p, float scale) { p.styleOrnate(scale); }},
        {"Card", [](Panel& p, float scale) { p.styleCard(scale); }},
        {"Soft", [](Panel& p, float scale) { p.styleSoft(scale); }},
        {"Gradient Card", [](Panel& p, float scale) {
             p.styleGradientCard(scale, UiColor::fromRgbHex(0x2E6F8E), UiColor::fromRgbHex(0x123247));
         }},
        {"Civ6", [](Panel& p, float scale) { p.styleCiv6(scale); }},
        {"Win95", [](Panel& p, float scale) { p.styleWin95(scale); }},
        {"Motif", [](Panel& p, float scale) { p.styleMotif(scale); }},
        {"Classic Mac", [](Panel& p, float scale) { p.styleClassicMac(scale); }},
        {"Retro-OS", [](Panel& p, float scale) { p.styleRetroOS(scale); }},
    }};

    constexpr int kCols = 4;
    const float cardW = 220.f * s, cardH = 130.f * s;
    const float gapX = 24.f * s, gapY = 46.f * s;

    for (std::size_t i = 0; i < styles.size(); ++i) {
        const int col = static_cast<int>(i) % kCols;
        const int row = static_cast<int>(i) / kCols;
        const float cx = x0 + static_cast<float>(col) * (cardW + gapX);
        const float cy = y + static_cast<float>(row) * (cardH + gapY);

        auto card = std::make_unique<Panel>();
        styles[i].apply(*card, s);
        card->setRect(UiRect::fromXYWH(cx, cy, cardW, cardH));
        page->addChild(std::move(card));

        auto label = std::make_unique<Label>(m_uiFonts, styles[i].name);
        label->setRect(UiRect::fromXYWH(cx, cy + cardH + 4.f * s, cardW, 20.f * s));
        label->align = UiTextAlign::Center;
        label->color = kRetroInkDim;
        page->addChild(std::move(label));
    }
}

// ---------------------------------------------------------------------------
// Effects tab: raw UiDrawList calls (the primitives Panel/Button/ProgressBar
// build on top of), demoed directly via EffectSwatch's paint callback.
// ---------------------------------------------------------------------------
void DesignSystemDemoApp::buildEffectsTab(Widget* page, float s) {
    const float x0 = 24.f * s;
    float y = 108.f * s;

    auto note = std::make_unique<Label>(m_uiFonts,
        "UiDrawList's raw vector/effect primitives, called directly (no Widget wraps these).");
    note->setRect(UiRect::fromXYWH(x0, y, 1000.f * s, 20.f * s));
    note->color = kRetroInkDim;
    page->addChild(std::move(note));
    y += 34.f * s;

    struct SwatchSpec {
        const char* name;
        std::function<void(UiDrawList&, const UiRect&, float)> paint;
    };
    const std::array<SwatchSpec, 6> swatches{{
        {"addDropShadow — sharp (r=0)", [](UiDrawList& dl, const UiRect& r, float scale) {
             const UiRect box{r.minX + 30.f * scale, r.minY + 20.f * scale,
                              r.maxX - 30.f * scale, r.maxY - 30.f * scale};
             dl.addDropShadow(box, UiColor{0, 0, 0, 0.6f}, 10.f * scale, 0.f, 6.f * scale, 0.f);
             dl.addRoundRectFilled(box, UiColor{0.20f, 0.45f, 0.55f, 1.0f}, 0.f);
         }},
        {"addDropShadow — rounded (r=16)", [](UiDrawList& dl, const UiRect& r, float scale) {
             const UiRect box{r.minX + 30.f * scale, r.minY + 20.f * scale,
                              r.maxX - 30.f * scale, r.maxY - 30.f * scale};
             const float radius = 16.f * scale;
             dl.addDropShadow(box, UiColor{0, 0, 0, 0.6f}, 10.f * scale, 0.f, 6.f * scale, radius);
             dl.addRoundRectFilled(box, UiColor{0.20f, 0.45f, 0.55f, 1.0f}, radius);
         }},
        {"addRoundRectGlow", [](UiDrawList& dl, const UiRect& r, float scale) {
             const UiRect box{r.minX + 40.f * scale, r.minY + 30.f * scale,
                              r.maxX - 40.f * scale, r.maxY - 40.f * scale};
             const float radius = 10.f * scale;
             dl.addRoundRectGlow(box, UiColor{0.95f, 0.72f, 0.35f, 0.55f}, radius, 16.f * scale);
             dl.addRoundRectFilled(box, UiColor{0.22f, 0.20f, 0.14f, 1.0f}, radius);
         }},
        {"addBevel — raised / recessed", [](UiDrawList& dl, const UiRect& r, float scale) {
             const UiRect left{r.minX + 20.f * scale, r.minY + 30.f * scale,
                               r.minX + 110.f * scale, r.maxY - 30.f * scale};
             const UiRect right{r.maxX - 110.f * scale, r.minY + 30.f * scale,
                                r.maxX - 20.f * scale, r.maxY - 30.f * scale};
             const UiColor fill{0.20f, 0.20f, 0.22f, 1.0f};
             const UiColor hi{1.0f, 1.0f, 1.0f, 0.28f};
             const UiColor sh{0.0f, 0.0f, 0.0f, 0.45f};
             dl.addRoundRectFilled(left, fill, 4.f * scale);
             dl.addBevel(left, hi, sh, 4.f * scale, 3.f * scale, false);
             dl.addRoundRectFilled(right, fill, 4.f * scale);
             dl.addBevel(right, hi, sh, 4.f * scale, 3.f * scale, true);
         }},
        {"Gradients (rect + rounded-rect)", [](UiDrawList& dl, const UiRect& r, float scale) {
             const UiRect top{r.minX + 20.f * scale, r.minY + 20.f * scale,
                              r.maxX - 20.f * scale, r.minY + 55.f * scale};
             const UiRect bottom{r.minX + 20.f * scale, r.minY + 65.f * scale,
                                 r.maxX - 20.f * scale, r.maxY - 20.f * scale};
             dl.addRectFilledHGradient(top, UiColor{0.82f, 0.36f, 0.46f, 1.0f},
                                       UiColor{0.36f, 0.55f, 0.82f, 1.0f});
             dl.addRoundRectFilledVGradient(bottom, UiColor{0.40f, 0.78f, 0.52f, 1.0f},
                                            UiColor{0.20f, 0.30f, 0.45f, 1.0f}, 10.f * scale);
         }},
        {"addCircle / addCircleFilled", [](UiDrawList& dl, const UiRect& r, float scale) {
             const UiVec2 c1{r.minX + 70.f * scale, r.minY + r.height() * 0.5f};
             const UiVec2 c2{r.maxX - 70.f * scale, r.minY + r.height() * 0.5f};
             dl.addCircleFilled(c1, 34.f * scale, UiColor{0.68f, 0.44f, 0.82f, 1.0f});
             dl.addCircle(c2, 34.f * scale, UiColor{0.86f, 0.50f, 0.28f, 1.0f}, 4.f * scale);
         }},
    }};

    constexpr int kCols = 3;
    const float cardW = 300.f * s, cardH = 170.f * s;
    const float gapX = 24.f * s, gapY = 46.f * s;

    for (std::size_t i = 0; i < swatches.size(); ++i) {
        const int col = static_cast<int>(i) % kCols;
        const int row = static_cast<int>(i) / kCols;
        const float cx = x0 + static_cast<float>(col) * (cardW + gapX);
        const float cy = y + static_cast<float>(row) * (cardH + gapY);

        auto backing = std::make_unique<Panel>();
        backing->setRect(UiRect::fromXYWH(cx, cy, cardW, cardH));
        backing->background = kRetroBacking;
        backing->cornerRadiusPx = 4.f * s;
        page->addChild(std::move(backing));

        auto swatch = std::make_unique<EffectSwatch>();
        swatch->setRect(UiRect::fromXYWH(cx, cy, cardW, cardH));
        swatch->paint = [fn = swatches[i].paint, s](UiDrawList& dl, const UiRect& rect) { fn(dl, rect, s); };
        page->addChild(std::move(swatch));

        auto label = std::make_unique<Label>(m_uiFonts, swatches[i].name);
        label->setRect(UiRect::fromXYWH(cx, cy + cardH + 4.f * s, cardW, 20.f * s));
        label->align = UiTextAlign::Center;
        label->color = kRetroInkDim;
        page->addChild(std::move(label));
    }
}

// ---------------------------------------------------------------------------
// Animation tab: one lane per Easing curve (Vec2Tween), plus a RectTween +
// Sequence "pop card" — mirrors tools/tween_demo, folded into this app so
// every animation type lives alongside the rest of the component library.
// ---------------------------------------------------------------------------
void DesignSystemDemoApp::buildAnimationTab(Widget* page, float s) {
    const float x0 = 24.f * s;
    float y = 108.f * s;

    auto note = std::make_unique<Label>(m_uiFonts,
        "Every Easing curve in animation.h (Vec2Tween), looping side by side. "
        "Toggle/ProgressBar/Button elsewhere in this app exercise Tween and ColorTween "
        "the same way; Pop Card below exercises RectTween + Sequence.");
    note->setRect(UiRect::fromXYWH(x0, y, 1000.f * s, 34.f * s));
    note->color = kRetroInkDim;
    page->addChild(std::move(note));
    y += 44.f * s;

    constexpr float kRowH = 40.f;
    constexpr float kLabelW = 110.f;
    // Track starts after the label gutter (same zero-gap convention as Inputs
    // tab's addRowLabel: control begins at x0 + labelW). The old fixed 20px
    // offset put the track underneath the up-to-110px-wide label text.
    const float trackX = x0 + kLabelW * s;
    constexpr float kTrackW = 380.f;
    constexpr float kBoxSize = 28.f;

    for (std::size_t i = 0; i < kLaneSpecs.size(); ++i) {
        const LaneSpec& spec = kLaneSpecs[i];
        const float laneY = y + static_cast<float>(i) * kRowH * s;

        auto nameLabel = std::make_unique<Label>(m_uiFonts, spec.name);
        nameLabel->setRect(UiRect::fromXYWH(x0, laneY + (kRowH * s - 18.f * s) * 0.5f, kLabelW * s, 18.f * s));
        nameLabel->color = kRetroInkDim;
        page->addChild(std::move(nameLabel));

        auto track = std::make_unique<Panel>();
        track->setRect(UiRect::fromXYWH(trackX, laneY + (kRowH * s - 6.f * s) * 0.5f, kTrackW * s, 6.f * s));
        track->background = kRetroTrack;
        track->cornerRadiusPx = 3.f * s;
        page->addChild(std::move(track));

        auto box = std::make_unique<Panel>();
        box->cornerRadiusPx = 6.f * s;
        box->background = spec.color;
        Panel* boxPtr = static_cast<Panel*>(page->addChild(std::move(box)));

        GalleryLane& lane = m_gallery[i];
        lane.box = boxPtr;
        lane.easing = spec.easing;
        lane.y = laneY + (kRowH * s - kBoxSize * s) * 0.5f;
        lane.trackMinX = trackX;
        lane.trackMaxX = trackX + (kTrackW - kBoxSize) * s;
        lane.tween.snap(UiVec2{lane.trackMinX, lane.y});
        lane.tween.set(UiVec2{lane.trackMaxX, lane.y}, kLaneDurationSec, lane.easing);
        lane.box->setRect(UiRect::fromXYWH(lane.trackMinX, lane.y, kBoxSize * s, kBoxSize * s));
    }
    y += static_cast<float>(kLaneSpecs.size()) * kRowH * s + 20.f * s;

    auto popButton = std::make_unique<Button>(&m_uiFontBold, "Pop Card (RectTween + Sequence)",
                                              [this] { showPopCard(); });
    applyRetroButtonStyle(*popButton, s);
    popButton->setRect(UiRect::fromXYWH(x0, y, 280.f * s, 32.f * s));
    page->addChild(std::move(popButton));

    // Backdrop + card live at the root (not this page) so they can cover the
    // whole viewport regardless of which tab is active when they're open.
    int fbW = 0, fbH = 0;
    framebufferSize(fbW, fbH);

    auto backdrop = std::make_unique<Panel>();
    backdrop->setRect(UiRect::fromXYWH(0.f, 0.f, static_cast<float>(fbW), static_cast<float>(fbH)));
    backdrop->background = UiColor{0.0f, 0.0f, 0.0f, 0.65f};
    backdrop->visible = false;
    backdrop->opacity = 0.0f;
    m_popBackdrop = static_cast<Panel*>(m_root->addChild(std::move(backdrop)));

    constexpr float kCardW = 420.f, kCardH = 220.f;
    const float cx = (static_cast<float>(fbW) - kCardW * s) * 0.5f;
    const float cy = (static_cast<float>(fbH) - kCardH * s) * 0.5f;
    m_popOpenRect = UiRect::fromXYWH(cx, cy, kCardW * s, kCardH * s);
    m_popClosedRect = UiRect::fromXYWH(cx, cy, 1.f, 1.f);

    auto card = std::make_unique<Panel>();
    card->styleRetroOS(s);
    card->clipContents = true;
    Panel* cardPtr = static_cast<Panel*>(m_popBackdrop->addChild(std::move(card)));
    cardPtr->setRect(m_popOpenRect);
    m_popCard = cardPtr;

    auto cardTitle = std::make_unique<Label>(m_uiFonts, "<b>RectTween + Sequence</b>");
    cardTitle->setRect(UiRect::fromXYWH(cx + 24.f * s, cy + 20.f * s, kCardW * s - 48.f * s, 28.f * s));
    cardTitle->color = kRetroInk;
    m_popCard->addChild(std::move(cardTitle));

    auto cardBody = std::make_unique<Label>(m_uiFonts,
        "This card's rect is driven by a RectTween (BackOut in / CubicIn out). "
        "The backdrop's opacity is driven by a Sequence step, independently.");
    cardBody->setRect(UiRect::fromXYWH(cx + 24.f * s, cy + 56.f * s, kCardW * s - 48.f * s, 80.f * s));
    cardBody->color = kRetroInkDim;
    m_popCard->addChild(std::move(cardBody));

    auto closeButton = std::make_unique<Button>(&m_uiFontBold, "Close", [this] { hidePopCard(); });
    applyRetroButtonStyle(*closeButton, s);
    closeButton->setRect(UiRect::fromXYWH(cx + 24.f * s, cy + kCardH * s - 56.f * s, 120.f * s, 32.f * s));
    m_popCard->addChild(std::move(closeButton));

    m_popRectTween.snap(m_popClosedRect);
    m_popCard->setRect(m_popClosedRect);
}

void DesignSystemDemoApp::showPopCard() {
    m_popBackdrop->visible = true;
    m_popRectTween.set(m_popOpenRect, 0.4f, Easing::BackOut);

    m_popSequence = Sequence{};
    m_popSequence.append(0.25f, [this](float t) { m_popBackdrop->opacity = t; });
}

void DesignSystemDemoApp::hidePopCard() {
    m_popRectTween.set(m_popClosedRect, 0.22f, Easing::CubicIn);

    m_popSequence = Sequence{};
    m_popSequence.append(0.22f, [this](float t) { m_popBackdrop->opacity = 1.0f - t; },
                          [this] { m_popBackdrop->visible = false; });
}

// ---------------------------------------------------------------------------
// Data & Feedback tab: DonutChart, LineChart, StatBadgeRow, Image, RichTextView,
// plus the overlay triggers (Toast, Modal, ContextMenu) and a draggable Window.
// ---------------------------------------------------------------------------
void DesignSystemDemoApp::buildDataFeedbackTab(Widget* page, float s) {
    const float x0 = 24.f * s;
    float y = 108.f * s;

    // Intro note, matching the Buttons/Panels/Effects/Animation tabs so every
    // tab's first row of real content lands at the same rhythm (note, then
    // +34px gap) instead of this tab alone starting flush at the top.
    auto note = std::make_unique<Label>(m_uiFonts,
        "DonutChart, LineChart, StatBadgeRow, Image, and RichTextView, plus the "
        "overlay triggers (Toast, Modal, Context Menu) and a draggable Window.");
    note->setRect(UiRect::fromXYWH(x0, y, 1000.f * s, 20.f * s));
    note->color = kRetroInkDim;
    page->addChild(std::move(note));
    y += 34.f * s;

    // DonutChart.
    auto donutHeading = std::make_unique<Label>(m_uiFonts, "<b>DonutChart</b>");
    donutHeading->setRect(UiRect::fromXYWH(x0, y, 220.f * s, 20.f * s));
    donutHeading->color = kRetroInk;
    page->addChild(std::move(donutHeading));

    auto donut = std::make_unique<DonutChart>(&m_uiFont);
    donut->setRect(UiRect::fromXYWH(x0, y + 26.f * s, 200.f * s, 200.f * s));
    donut->segments = {
        {0.40f, UiColor{0.40f, 0.78f, 0.52f, 1.0f}, "Gold"},
        {0.25f, UiColor{0.36f, 0.55f, 0.82f, 1.0f}, "Science"},
        {0.20f, UiColor{0.86f, 0.50f, 0.28f, 1.0f}, "Culture"},
        {0.15f, UiColor{0.68f, 0.44f, 0.82f, 1.0f}, "Faith"},
    };
    donut->centerLabel = "100%";
    donut->centerLabelColor = kRetroInk;
    page->addChild(std::move(donut));

    // LineChart.
    const float chartX = x0 + 240.f * s;
    auto lineHeading = std::make_unique<Label>(m_uiFonts, "<b>LineChart</b>");
    lineHeading->setRect(UiRect::fromXYWH(chartX, y, 300.f * s, 20.f * s));
    lineHeading->color = kRetroInk;
    page->addChild(std::move(lineHeading));

    auto line = std::make_unique<LineChart>(&m_uiFont);
    line->setRect(UiRect::fromXYWH(chartX, y + 26.f * s, 340.f * s, 200.f * s));
    ChartSeries series1, series2;
    series1.color = UiColor{0.40f, 0.78f, 0.52f, 1.0f};
    series1.label = "Gold";
    series2.color = UiColor{0.36f, 0.55f, 0.82f, 1.0f};
    series2.label = "Science";
    for (int i = 0; i < 12; ++i) {
        const float t = static_cast<float>(i);
        series1.values.push_back(50.f + 30.f * std::sin(t * 0.5f));
        series2.values.push_back(40.f + 25.f * std::sin(t * 0.4f + 1.2f));
    }
    line->series = {series1, series2};
    line->minValue = 0.f;
    line->maxValue = 100.f;
    line->autoRange = false;
    page->addChild(std::move(line));

    // StatBadgeRow.
    const float statX = chartX + 360.f * s;
    auto statHeading = std::make_unique<Label>(m_uiFonts, "<b>StatBadgeRow</b>");
    statHeading->setRect(UiRect::fromXYWH(statX, y, 260.f * s, 20.f * s));
    statHeading->color = kRetroInk;
    page->addChild(std::move(statHeading));

    auto stats = std::make_unique<StatBadgeRow>(&m_uiFont);
    stats->setRect(UiRect::fromXYWH(statX, y + 26.f * s, 260.f * s, 24.f * s));
    stats->stats = {
        {"", "128g", UiColor{0.90f, 0.80f, 0.40f, 1.0f}},
        {"", "+12/turn", UiColor{0.60f, 0.85f, 0.95f, 1.0f}},
        {"", "45c", UiColor{0.85f, 0.55f, 0.90f, 1.0f}},
    };
    page->addChild(std::move(stats));

    // Image (reuses the font atlas texture — no extra asset needed).
    auto imageHeading = std::make_unique<Label>(m_uiFonts, "<b>Image</b> (font atlas texture)");
    imageHeading->setRect(UiRect::fromXYWH(statX, y + 60.f * s, 300.f * s, 20.f * s));
    imageHeading->color = kRetroInk;
    page->addChild(std::move(imageHeading));

    auto image = std::make_unique<Image>(kUiFontAtlas);
    image->setRect(UiRect::fromXYWH(statX, y + 86.f * s, 140.f * s, 140.f * s));
    page->addChild(std::move(image));

    y += 240.f * s;

    // RichTextView with a hoverable tooltip span.
    auto richHeading = std::make_unique<Label>(m_uiFonts, "<b>RichTextView</b>");
    richHeading->setRect(UiRect::fromXYWH(x0, y, 300.f * s, 20.f * s));
    richHeading->color = kRetroInk;
    page->addChild(std::move(richHeading));

    auto rich = std::make_unique<RichTextView>(m_uiFonts,
        "Rich text supports <b>bold</b>, <i>italic</i>, <color=#0078D7>custom colors</color>, "
        "and hoverable <tip=Design tokens live in Panel::style* and UiTheme.>tooltip spans</tip> "
        "like this one.");
    rich->color = kRetroInk;
    rich->setRect(UiRect::fromXYWH(x0, y + 26.f * s, 560.f * s, 70.f * s));
    page->addChild(std::move(rich));

    y += 116.f * s;

    // Overlay triggers.
    auto overlayHeading = std::make_unique<Label>(m_uiFonts, "<b>Overlays</b>");
    overlayHeading->setRect(UiRect::fromXYWH(x0, y, 300.f * s, 20.f * s));
    overlayHeading->color = kRetroInk;
    page->addChild(std::move(overlayHeading));
    y += 26.f * s;  // heading-to-content gap (standardized; see Data & Feedback / Layout).

    auto toastBtn = std::make_unique<Button>(&m_uiFontBold, "Push Toast", [this] {
        if (m_toastManager != nullptr) m_toastManager->push("", "This is a toast notification.");
    });
    applyRetroButtonStyle(*toastBtn, s);
    toastBtn->setRect(UiRect::fromXYWH(x0, y, 180.f * s, 32.f * s));
    page->addChild(std::move(toastBtn));

    auto modalBtn = std::make_unique<Button>(&m_uiFontBold, "Open Modal", [this] {
        if (m_modal != nullptr) m_modal->open();
    });
    applyRetroButtonStyle(*modalBtn, s);
    modalBtn->setRect(UiRect::fromXYWH(x0 + 196.f * s, y, 180.f * s, 32.f * s));
    page->addChild(std::move(modalBtn));

    auto ctxBtn = std::make_unique<Button>(&m_uiFontBold, "Open Context Menu", [this, x0, y, s] {
        if (m_contextMenu != nullptr) {
            m_contextMenu->openAt(UiVec2{x0 + 392.f * s, y + 36.f * s});
        }
    });
    applyRetroButtonStyle(*ctxBtn, s);
    ctxBtn->setRect(UiRect::fromXYWH(x0 + 392.f * s, y, 200.f * s, 32.f * s));
    page->addChild(std::move(ctxBtn));

    // Window: a real draggable, closeable Window instance.
    y += 50.f * s;
    auto showWindowBtn = std::make_unique<Button>(&m_uiFontBold, "Show Window", [this] {
        if (m_demoWindow != nullptr) {
            m_demoWindow->visible = true;
            m_demoWindow->bringToFront();
        }
    });
    applyRetroButtonStyle(*showWindowBtn, s);
    showWindowBtn->setRect(UiRect::fromXYWH(x0, y, 180.f * s, 32.f * s));
    page->addChild(std::move(showWindowBtn));

    auto window = std::make_unique<Window>(&m_uiFontBold, "Window",
        [this] { if (m_demoWindow != nullptr) m_demoWindow->visible = false; });
    applyRetroWindowStyle(*window, s);
    window->setRect(UiRect::fromXYWH(x0 + 260.f * s, y - 20.f * s, 320.f * s, 200.f * s));
    window->padding = {14.f * s, 10.f * s};
    auto windowBody = std::make_unique<Label>(m_uiFonts,
        "Drag the title bar to move me. Close and reopen with the button on the left.");
    windowBody->setRect(UiRect::fromXYWH(0.f, 0.f, 280.f * s, 100.f * s));
    windowBody->color = kRetroInkDim;
    window->addChild(std::move(windowBody));
    m_demoWindow = static_cast<Window*>(page->addChild(std::move(window)));
}

// ---------------------------------------------------------------------------
// Layout tab: ScrollView + Repeater (data-driven list), HorizontalStack /
// VerticalStack (with a Spacer).
// ---------------------------------------------------------------------------
void DesignSystemDemoApp::buildLayoutTab(Widget* page, float s) {
    const float x0 = 24.f * s;
    float y = 108.f * s;

    // Intro note, matching the Buttons/Panels/Effects/Animation tabs so every
    // tab's first row of real content lands at the same rhythm (note, then
    // +34px gap) instead of this tab alone starting flush at the top.
    auto note = std::make_unique<Label>(m_uiFonts,
        "Layout primitives: a scrolling, data-driven Repeater list next to "
        "HorizontalStack and VerticalStack composition.");
    note->setRect(UiRect::fromXYWH(x0, y, 1000.f * s, 20.f * s));
    note->color = kRetroInkDim;
    page->addChild(std::move(note));
    y += 34.f * s;

    auto scrollHeading = std::make_unique<Label>(m_uiFonts, "<b>ScrollView + Repeater</b>");
    scrollHeading->setRect(UiRect::fromXYWH(x0, y, 320.f * s, 20.f * s));
    scrollHeading->color = kRetroInk;
    page->addChild(std::move(scrollHeading));

    auto scroll = std::make_unique<ScrollView>();
    scroll->setRect(UiRect::fromXYWH(x0, y + 26.f * s, 320.f * s, 320.f * s));
    scroll->childGap = 4.f * s;
    ScrollView* scrollPtr = static_cast<ScrollView*>(page->addChild(std::move(scroll)));

    auto repeater = std::make_unique<Repeater>();
    repeater->itemHeight = 26.f * s;
    repeater->itemGap = 4.f * s;
    const FontSet& fontsRef = m_uiFonts;
    repeater->setItemFactory([&fontsRef](const std::shared_ptr<DataNode>& item, std::size_t index) {
        auto lbl = std::make_unique<Label>(fontsRef,
            "Row " + std::to_string(index) + " — " + item->selfString());
        lbl->color = kRetroInkDim;
        return lbl;
    });
    std::vector<std::shared_ptr<DataNode>> items;
    for (int i = 0; i < 24; ++i) {
        items.push_back(std::make_shared<StringDataNode>("data-driven via Repeater::setItems"));
    }
    repeater->setItems(std::move(items));
    scrollPtr->addChild(std::move(repeater));

    const float stackX = x0 + 360.f * s;

    auto hHeading = std::make_unique<Label>(m_uiFonts, "<b>HorizontalStack</b> (with a Spacer)");
    hHeading->setRect(UiRect::fromXYWH(stackX, y, 400.f * s, 20.f * s));
    hHeading->color = kRetroInk;
    page->addChild(std::move(hHeading));

    auto hStack = std::make_unique<HorizontalStack>();
    hStack->setRect(UiRect::fromXYWH(stackX, y + 26.f * s, 500.f * s, 60.f * s));
    hStack->gap = 10.f * s;
    hStack->crossAlign = StackLayout::Align::Center;
    HorizontalStack* hStackPtr = static_cast<HorizontalStack*>(page->addChild(std::move(hStack)));

    const std::array<UiColor, 3> chipColors{{
        UiColor{0.40f, 0.78f, 0.52f, 1.0f},
        UiColor{0.36f, 0.55f, 0.82f, 1.0f},
        UiColor{0.86f, 0.50f, 0.28f, 1.0f},
    }};
    for (std::size_t i = 0; i < chipColors.size(); ++i) {
        auto chip = std::make_unique<Panel>();
        chip->setRect(UiRect::fromXYWH(0.f, 0.f, 90.f * s, 50.f * s));
        chip->background = chipColors[i];
        chip->cornerRadiusPx = 6.f * s;
        hStackPtr->addChild(std::move(chip));
        if (i == 0) {
            auto spacer = std::make_unique<Spacer>();
            spacer->setRect(UiRect::fromXYWH(0.f, 0.f, 40.f * s, 1.f * s));
            hStackPtr->addChild(std::move(spacer));
        }
    }

    const float vY = y + 110.f * s;
    auto vHeading = std::make_unique<Label>(m_uiFonts, "<b>VerticalStack</b>");
    vHeading->setRect(UiRect::fromXYWH(stackX, vY, 400.f * s, 20.f * s));
    vHeading->color = kRetroInk;
    page->addChild(std::move(vHeading));

    auto vStack = std::make_unique<VerticalStack>();
    vStack->setRect(UiRect::fromXYWH(stackX, vY + 26.f * s, 320.f * s, 200.f * s));
    vStack->gap = 6.f * s;
    VerticalStack* vStackPtr = static_cast<VerticalStack*>(page->addChild(std::move(vStack)));
    for (int i = 0; i < 4; ++i) {
        auto row = std::make_unique<Label>(m_uiFonts, "Stack row " + std::to_string(i));
        row->setRect(UiRect::fromXYWH(0.f, 0.f, 320.f * s, 22.f * s));
        row->color = kRetroInkDim;
        vStackPtr->addChild(std::move(row));
    }
}

void DesignSystemDemoApp::tickGallery(float dt) {
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

void DesignSystemDemoApp::updateFrameStats(float dt) {
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

void DesignSystemDemoApp::onTick(float dt) {
    if (glfwGetKey(m_window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(m_window, GLFW_TRUE);
    }

    updateFrameStats(dt);
    tickGallery(dt);

    m_popRectTween.update(dt);
    if (m_popCard != nullptr) {
        m_popCard->repositionAndResize(m_popRectTween.current());
    }
    m_popSequence.update(dt);
}

void DesignSystemDemoApp::onRender(float /*dt*/) {
    beginFrameDraw();
    render::CameraPose camera{};
    submitFrame(camera);
}

}  // namespace odai::tools::design_system_demo
