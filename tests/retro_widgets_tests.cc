// Headless tests for the retro demo's RetroUi immediate-mode widget wrapper.
// Validates the interaction mechanism the demo relies on (slider drag, textbox
// typing, toggle, per-frame visibility/skin) without any Vulkan/GLFW.

#include "tools/retro_theme_demo/retro_widgets.h"
#include "ui/font.h"
#include "ui/ui_draw_list.h"

#include <cmath>
#include <iostream>

namespace {

using namespace odai::ui;
using odai::tools::retro_theme_demo::RetroUi;
using odai::tools::retro_theme_demo::WidgetSkin;

int g_failures = 0;

void expectTrue(bool cond, const char* msg) {
    if (!cond) {
        std::cerr << "[retro widgets test] FAIL: " << msg << '\n';
        ++g_failures;
    }
}

void expectNear(float a, float b, float eps, const char* msg) {
    if (std::fabs(a - b) > eps) {
        std::cerr << "[retro widgets test] FAIL: " << msg << " (expected " << b << ", got " << a
                  << ")\n";
        ++g_failures;
    }
}

Font makeFont() {
    Font f;
    f.initSyntheticMonospace(8.0f, 14.0f, 11.0f);
    return f;
}

const UiVec2 kViewport{800.0f, 600.0f};
const std::vector<std::uint32_t> kNoText{};

void testSliderDrag() {
    RetroUi ui;
    ui.init();
    WidgetSkin skin;

    // Frame 1: place the slider, then press the left button at its midpoint.
    ui.beginFrame(skin);
    Slider& sl = ui.slider("vol", UiRect{0.0f, 0.0f, 100.0f, 20.0f});
    const float before = sl.value;
    ui.update(UiVec2{50.0f, 10.0f}, /*leftDown=*/true, kViewport, kNoText, 0.016f);
    expectNear(sl.value, 0.5f, 0.06f, "slider value tracks mouse to ~midpoint on press");
    expectTrue(sl.value != before || before == 0.5f, "slider value responds to drag");

    // Frame 2: drag toward the right while still held.
    ui.beginFrame(skin);
    ui.slider("vol", UiRect{0.0f, 0.0f, 100.0f, 20.0f});
    ui.update(UiVec2{90.0f, 10.0f}, true, kViewport, kNoText, 0.016f);
    expectTrue(sl.value > 0.7f, "slider value increases when dragged right");

    // Geometry is produced.
    UiDrawList dl;
    dl.reset(kViewport);
    ui.append(dl);
    expectTrue(!dl.data().vertices.empty(), "slider emits geometry into the draw list");
}

void testTextBoxTyping() {
    Font font = makeFont();
    RetroUi ui;
    ui.init();
    WidgetSkin skin;
    const UiRect box{10.0f, 10.0f, 210.0f, 40.0f};

    // Frame 1: click inside to focus.
    ui.beginFrame(skin);
    TextBox& tb = ui.textBox("login", box, &font, "LOGIN");
    ui.update(UiVec2{50.0f, 25.0f}, true, kViewport, kNoText, 0.016f);
    expectTrue(tb.focused(), "textbox gains focus on click inside");

    // Frame 2: release + type "Hi".
    ui.beginFrame(skin);
    ui.textBox("login", box, &font, "LOGIN");
    ui.update(UiVec2{50.0f, 25.0f}, false, kViewport, {'H', 'i'}, 0.016f);
    expectTrue(tb.value() == "Hi", "textbox accepts typed characters");

    // Frame 3: backspace (codepoint 8) removes the last char.
    ui.beginFrame(skin);
    ui.textBox("login", box, &font, "LOGIN");
    ui.update(UiVec2{50.0f, 25.0f}, false, kViewport, {8u}, 0.016f);
    expectTrue(tb.value() == "H", "textbox backspace removes a character");

    // Frame 4: click outside removes focus.
    ui.beginFrame(skin);
    ui.textBox("login", box, &font, "LOGIN");
    ui.update(UiVec2{400.0f, 400.0f}, true, kViewport, kNoText, 0.016f);
    expectTrue(!tb.focused(), "textbox loses focus on click outside");
}

void testToggleClick() {
    RetroUi ui;
    ui.init();
    WidgetSkin skin;
    const UiRect t{0.0f, 0.0f, 48.0f, 24.0f};

    ui.beginFrame(skin);
    Toggle& tg = ui.toggle("snd", t);
    const bool initial = tg.checked;
    ui.update(UiVec2{24.0f, 12.0f}, true, kViewport, kNoText, 0.016f);
    expectTrue(tg.checked != initial, "toggle flips on click");
}

void testVisibilityAndReuse() {
    RetroUi ui;
    ui.init();
    WidgetSkin skin;

    // Frame 1: reference the slider — it becomes visible and is the same instance.
    ui.beginFrame(skin);
    Slider& a = ui.slider("x", UiRect{0, 0, 50, 10});
    expectTrue(a.visible, "referenced widget is visible");
    ui.update(UiVec2{0, 0}, false, kViewport, kNoText, 0.016f);

    // Frame 2: do NOT reference it — it should be hidden, but persist.
    ui.beginFrame(skin);
    ui.update(UiVec2{0, 0}, false, kViewport, kNoText, 0.016f);
    expectTrue(!a.visible, "un-referenced widget is hidden");

    // Frame 3: reference again — same object returns (state persisted).
    ui.beginFrame(skin);
    Slider& b = ui.slider("x", UiRect{0, 0, 50, 10});
    expectTrue(&a == &b, "same id returns the same persistent widget");
    expectTrue(b.visible, "re-referenced widget is visible again");
}

void testSkinApplied() {
    RetroUi ui;
    ui.init();
    WidgetSkin skin;
    skin.accent = UiColor{1.0f, 0.0f, 0.0f, 1.0f};
    skin.trough = UiColor{0.2f, 0.2f, 0.2f, 1.0f};

    ui.beginFrame(skin);
    Slider& sl = ui.slider("v", UiRect{0, 0, 100, 20});
    expectTrue(sl.fillColor.r == 1.0f && sl.fillColor.g == 0.0f, "slider fill takes skin accent");
    expectTrue(sl.trackColor.r == 0.2f, "slider track takes skin trough");
}

}  // namespace

int main() {
    testSliderDrag();
    testTextBoxTyping();
    testToggleClick();
    testVisibilityAndReuse();
    testSkinApplied();

    if (g_failures != 0) {
        std::cerr << "[retro widgets test] " << g_failures << " failures\n";
        return 1;
    }
    std::cout << "[retro widgets test] all checks passed\n";
    return 0;
}
