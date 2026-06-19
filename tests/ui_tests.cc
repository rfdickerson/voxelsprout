#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>

#include "game/buildable.h"
#include "ui/animation.h"
#include "ui/cached_rich_text.h"
#include "ui/document/ui_binding.h"
#include "ui/font.h"
#include "ui/icon_atlas.h"
#include "ui/rich_text.h"
#include "ui/ui_context.h"
#include "ui/ui_draw_list.h"
#include "ui/ui_input.h"
#include "ui/ui_types.h"
#include "ui/widgets/button.h"
#include "ui/widgets/donut_chart.h"
#include "ui/widgets/dropdown.h"
#include "ui/widgets/line_chart.h"
#include "ui/widgets/panel.h"
#include "ui/widgets/production_panel.h"
#include "ui/widgets/progress_bar.h"
#include "ui/widgets/scroll_view.h"
#include "ui/widgets/slider.h"
#include "ui/widgets/spacer.h"
#include "ui/widgets/stack_layout.h"
#include "ui/widgets/stat_badge.h"
#include "ui/widgets/tab_bar.h"
#include "ui/widgets/text_box.h"
#include "ui/widgets/toast.h"
#include "ui/widgets/toggle.h"
#include "ui/widgets/toolbar.h"

#include <cstddef>

namespace {

int g_failures = 0;

void expectTrue(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "[ui test] FAIL: " << message << '\n';
        ++g_failures;
    }
}

void expectNear(float actual, float expected, float epsilon, const char* message) {
    if (std::fabs(actual - expected) > epsilon) {
        std::cerr << "[ui test] FAIL: " << message
                  << " (expected " << expected << ", got " << actual << ")\n";
        ++g_failures;
    }
}

odai::ui::Font makeMonospaceFont(float advance = 10.0f) {
    odai::ui::Font font;
    font.initSyntheticMonospace(advance, 12.0f, 4.0f);
    return font;
}

void testNineSliceQuadGen() {
    using namespace odai::ui;
    UiDrawList dl;
    dl.reset(UiVec2{200.0f, 200.0f});
    const UiNineSlice slice = UiNineSlice::uniform(7u, 16.0f, 64.0f);
    dl.add9Slice(UiRect::fromXYWH(0.0f, 0.0f, 100.0f, 100.0f), slice);

    const UiDrawData& data = dl.data();
    expectTrue(data.vertices.size() == 36u, "9-slice emits 9 quads (36 vertices)");
    expectTrue(data.indices.size() == 54u, "9-slice emits 54 indices");
    expectTrue(data.commands.size() == 1u, "9-slice shares one draw command");

    // Corner top-left quad keeps the 16px border size.
    expectNear(data.vertices[0].posPx[0], 0.0f, 1e-4f, "9-slice TL corner min x");
    expectNear(data.vertices[0].posPx[1], 0.0f, 1e-4f, "9-slice TL corner min y");
    expectNear(data.vertices[2].posPx[0], 16.0f, 1e-4f, "9-slice TL corner max x");
    expectNear(data.vertices[2].posPx[1], 16.0f, 1e-4f, "9-slice TL corner max y");

    // Center quad (row1,col1 -> quad index 4 -> vertex 16) stretches across the middle.
    expectNear(data.vertices[16].posPx[0], 16.0f, 1e-4f, "9-slice center min x");
    expectNear(data.vertices[18].posPx[0], 84.0f, 1e-4f, "9-slice center max x");
    expectNear(data.vertices[18].posPx[1], 84.0f, 1e-4f, "9-slice center max y");
}

void testFontMeasure() {
    using namespace odai::ui;
    const Font font = makeMonospaceFont(10.0f);
    expectNear(font.measureText("AB"), 20.0f, 1e-4f, "measureText sums advances");
    expectNear(font.measureText(""), 0.0f, 1e-4f, "measureText of empty string is 0");
    expectNear(font.measureText("hello"), 50.0f, 1e-4f, "measureText of 5 glyphs");

    UiDrawList dl;
    dl.reset(UiVec2{100.0f, 100.0f});
    const float advance = dl.addText(font, "AB", UiVec2{0.0f, 0.0f}, UiColor{1, 1, 1, 1});
    expectNear(advance, 20.0f, 1e-4f, "addText advance matches measureText");
}

void testRichTextWrap() {
    using namespace odai::ui;
    const Font font = makeMonospaceFont(10.0f);
    const RichTextLayout layout =
        layoutRichText("aaaa bbbb cccc", UiColor{1, 1, 1, 1}, font, 90.0f, UiTextAlign::Left);

    expectTrue(layout.lines.size() == 2u, "wrap breaks 3 words into 2 lines at width 90");
    bool withinWidth = true;
    for (const RichLine& line : layout.lines) {
        if (line.width > 90.0f + 1e-3f) {
            withinWidth = false;
        }
    }
    expectTrue(withinWidth, "no wrapped line exceeds the wrap width");
}

void testRichTextSpans() {
    using namespace odai::ui;
    const std::vector<RichSpan> spans =
        parseRichText("<color=#ff0000>red</color> plain", UiColor{1, 1, 1, 1});
    expectTrue(spans.size() == 2u, "markup parses into two spans");
    if (spans.size() == 2u) {
        expectNear(spans[0].color.r, 1.0f, 1e-3f, "first span is red (r)");
        expectNear(spans[0].color.g, 0.0f, 1e-3f, "first span is red (g)");
        expectTrue(spans[0].text == "red", "first span text is 'red'");
        expectNear(spans[1].color.r, 1.0f, 1e-3f, "second span restores default color");
        expectNear(spans[1].color.g, 1.0f, 1e-3f, "second span default green");
        expectTrue(spans[1].text == " plain", "second span text is ' plain'");
    }
}

void testColorPacking() {
    using namespace odai::ui;
    const UiColor c = UiColor::fromRgbHex(0x3366CCu);
    expectNear(c.r, 0x33 / 255.0f, 1e-3f, "fromRgbHex red channel");
    expectNear(c.g, 0x66 / 255.0f, 1e-3f, "fromRgbHex green channel");
    expectNear(c.b, 0xCC / 255.0f, 1e-3f, "fromRgbHex blue channel");
    const std::uint32_t packed = UiColor{1.0f, 0.0f, 0.0f, 1.0f}.packAbgr8();
    expectTrue((packed & 0xFFu) == 0xFFu, "packAbgr8 red in low byte");
    expectTrue(((packed >> 24) & 0xFFu) == 0xFFu, "packAbgr8 alpha in high byte");
}

odai::ui::UiInput inputAt(float x, float y) {
    odai::ui::UiInput in;
    in.mousePx = {x, y};
    return in;
}

void testButtonCallbackAndHitTest() {
    using namespace odai::ui;
    const Font font = makeMonospaceFont(8.0f);

    int clicks = 0;
    auto button = std::make_unique<Button>(&font, "Ok", [&clicks]() { ++clicks; });
    Button* buttonPtr = button.get();

    UiContext ctx;
    ctx.setViewport(UiVec2{400.0f, 300.0f});
    ctx.setRoot(std::move(button));
    buttonPtr->setRect(UiRect::fromXYWH(10.0f, 10.0f, 100.0f, 40.0f));  // 10..110, 10..50

    // Hover.
    UiInput hover = inputAt(50.0f, 30.0f);
    ctx.update(hover);
    expectTrue(buttonPtr->state() == Button::State::Hover, "button hovers when mouse is inside");
    expectTrue(ctx.wantsMouse(), "context wants mouse when cursor is over a widget");

    // Press inside.
    UiInput press = inputAt(50.0f, 30.0f);
    press.setButton(UiMouseButton::Left, true);
    ctx.update(press);
    expectTrue(buttonPtr->state() == Button::State::Pressed, "button shows pressed state");
    expectTrue(clicks == 0, "no click fires on press alone");

    // Release inside -> click. Seed the prior-frame held state so setButton
    // derives a released edge (the real app carries this in a persistent UiInput).
    UiInput release = inputAt(50.0f, 30.0f);
    release.buttons[static_cast<std::size_t>(UiMouseButton::Left)].down = true;
    release.setButton(UiMouseButton::Left, false);
    ctx.update(release);
    expectTrue(clicks == 1, "click fires once on press-then-release inside");

    // Press inside, release outside -> no click.
    UiInput press2 = inputAt(50.0f, 30.0f);
    press2.setButton(UiMouseButton::Left, true);
    ctx.update(press2);
    UiInput releaseOutside = inputAt(300.0f, 250.0f);
    releaseOutside.buttons[static_cast<std::size_t>(UiMouseButton::Left)].down = true;
    releaseOutside.setButton(UiMouseButton::Left, false);
    ctx.update(releaseOutside);
    expectTrue(clicks == 1, "no click when released outside the button");

    // Cursor away from any widget.
    UiInput away = inputAt(380.0f, 280.0f);
    ctx.update(away);
    expectTrue(!ctx.wantsMouse(), "context does not want mouse when cursor is off all widgets");
}

void testSliderDrag() {
    using namespace odai::ui;
    Slider sl;
    sl.setRect(UiRect::fromXYWH(0.0f, 0.0f, 200.0f, 24.0f));
    sl.value = 0.0f;

    float reported = -1.0f;
    sl.onChange = [&](float v) { reported = v; };

    // Click on the track at x=100 → value ≈ 0.5.
    UiEvent down{};
    down.type = UiEvent::Type::MouseDown;
    down.button = UiMouseButton::Left;
    down.mousePx = UiVec2{100.0f, 12.0f};
    sl.onEvent(down);
    expectTrue(reported > 0.4f && reported < 0.6f, "Slider: click mid-track yields ~0.5");

    // Drag to the right edge.
    UiEvent move{};
    move.type = UiEvent::Type::MouseMove;
    move.mousePx = UiVec2{200.0f, 12.0f};
    sl.onEvent(move);
    expectNear(sl.value, 1.0f, 0.01f, "Slider: drag to right edge clamps to 1.0");

    // Release stops the drag.
    UiEvent up{};
    up.type = UiEvent::Type::MouseUp;
    up.mousePx = UiVec2{200.0f, 12.0f};
    sl.onEvent(up);
    // Move without dragging should not change value.
    UiEvent move2{};
    move2.type = UiEvent::Type::MouseMove;
    move2.mousePx = UiVec2{0.0f, 12.0f};
    sl.onEvent(move2);
    expectNear(sl.value, 1.0f, 0.01f, "Slider: value unchanged after MouseUp (drag released)");

    // Emits geometry without crashing.
    UiDrawList dl;
    dl.reset(UiVec2{400.0f, 200.0f});
    sl.draw(dl);
    expectTrue(!dl.data().vertices.empty(), "Slider: emits geometry");
}

void testToggleClick() {
    using namespace odai::ui;
    Toggle tog;
    tog.setRect(UiRect::fromXYWH(0.0f, 0.0f, 50.0f, 24.0f));

    bool lastChecked = false;
    int callCount = 0;
    tog.onChange = [&](bool v) { lastChecked = v; ++callCount; };

    expectTrue(!tog.checked, "Toggle: starts unchecked");

    UiEvent click{};
    click.type = UiEvent::Type::MouseDown;
    click.button = UiMouseButton::Left;
    click.mousePx = UiVec2{25.0f, 12.0f};
    tog.onEvent(click);
    expectTrue(tog.checked, "Toggle: click flips to checked");
    expectTrue(callCount == 1, "Toggle: onChange fired once");
    expectTrue(lastChecked, "Toggle: onChange receives true");

    tog.onEvent(click);
    expectTrue(!tog.checked, "Toggle: second click flips back to unchecked");
    expectTrue(!lastChecked, "Toggle: onChange receives false");

    // Emits geometry without crashing.
    UiDrawList dl;
    dl.reset(UiVec2{400.0f, 200.0f});
    tog.update(0.0f);
    tog.draw(dl);
    expectTrue(!dl.data().vertices.empty(), "Toggle: emits geometry");
}

void testTabBarSwitch() {
    using namespace odai::ui;
    Font font = makeMonospaceFont(8.0f);
    TabBar bar(&font);
    bar.setRect(UiRect::fromXYWH(0.0f, 0.0f, 300.0f, 32.0f));
    const int t0 = bar.addTab("Overview");
    const int t1 = bar.addTab("Units");
    const int t2 = bar.addTab("History");
    expectTrue(t0 == 0 && t1 == 1 && t2 == 2, "TabBar: addTab returns sequential indices");
    expectTrue(bar.activeTab == 0, "TabBar: starts at tab 0");

    int changed = -1;
    bar.onTabChanged = [&](int i) { changed = i; };

    // Click in the third tab's region (x ≈ 200..300 for 300px / 3 tabs).
    UiEvent click{};
    click.type = UiEvent::Type::MouseDown;
    click.button = UiMouseButton::Left;
    click.mousePx = UiVec2{250.0f, 16.0f};
    bar.onEvent(click);
    expectTrue(bar.activeTab == 2, "TabBar: click third tab activates index 2");
    expectTrue(changed == 2, "TabBar: onTabChanged fires with correct index");

    // Clicking the already-active tab does not fire the callback again.
    changed = -1;
    bar.onEvent(click);
    expectTrue(changed == -1, "TabBar: clicking active tab does not re-fire callback");

    // Geometry.
    UiDrawList dl;
    dl.reset(UiVec2{800.0f, 600.0f});
    bar.draw(dl);
    expectTrue(!dl.data().vertices.empty(), "TabBar: emits geometry");
}

void testDropdownSelect() {
    using namespace odai::ui;
    Font font = makeMonospaceFont(8.0f);
    Dropdown dd(&font);
    dd.setRect(UiRect::fromXYWH(10.0f, 10.0f, 150.0f, 28.0f));
    dd.items = {"Warrior", "Spearman", "Archer"};
    dd.itemHeightPx = 24.0f;

    int selected = -1;
    dd.onSelect = [&](int i) { selected = i; };

    expectTrue(!dd.isOpen(), "Dropdown: starts closed");

    // Click header → open.
    UiEvent click{};
    click.type = UiEvent::Type::MouseDown;
    click.button = UiMouseButton::Left;
    click.mousePx = UiVec2{80.0f, 20.0f};
    dd.onEvent(click);
    expectTrue(dd.isOpen(), "Dropdown: click header opens popup");

    // Click on item 1 (Spearman) at popup rect row 1.
    // Popup starts at rect_.maxY = 38; item 1 occupies y=[62, 86], mid = 74.
    UiEvent pick{};
    pick.type = UiEvent::Type::MouseDown;
    pick.button = UiMouseButton::Left;
    pick.mousePx = UiVec2{80.0f, 74.0f};
    dd.onEvent(pick);
    expectTrue(!dd.isOpen(), "Dropdown: selecting an item closes popup");
    expectTrue(selected == 1, "Dropdown: onSelect fires with correct index");
    expectTrue(dd.selectedIndex == 1, "Dropdown: selectedIndex updates");

    // Geometry.
    UiDrawList dl;
    dl.reset(UiVec2{800.0f, 600.0f});
    dd.draw(dl);
    expectTrue(!dl.data().vertices.empty(), "Dropdown: emits geometry");
}

void testDonutChartDraw() {
    using namespace odai::ui;
    DonutChart chart;
    chart.setRect(UiRect::fromXYWH(0.0f, 0.0f, 100.0f, 100.0f));
    chart.segments = {
        DonutSegment{0.4f, UiColor{0.8f, 0.2f, 0.2f, 1.0f}, "A"},
        DonutSegment{0.6f, UiColor{0.2f, 0.6f, 0.8f, 1.0f}, "B"},
    };
    chart.innerRadiusFraction = 0.5f;

    UiDrawList dl;
    dl.reset(UiVec2{200.0f, 200.0f});
    chart.draw(dl);
    expectTrue(!dl.data().vertices.empty(), "DonutChart: emits geometry for 2 segments");
    // Two segments × 32 steps × 4 vertices each = 256 vertices minimum.
    expectTrue(dl.data().vertices.size() >= 128u,
               "DonutChart: enough vertices for 32-step sectors");
}

void testSectorFilledGeometry() {
    using namespace odai::ui;
    // A solid wedge (innerR = 0) over a quarter circle produces 32 triangles = 96 indices.
    UiDrawList dl;
    dl.reset(UiVec2{200.0f, 200.0f});
    constexpr float kPi = 3.14159265f;
    dl.addSectorFilled(UiVec2{100.0f, 100.0f}, 0.0f, 40.0f,
                       0.0f, kPi * 0.5f, UiColor{1, 0, 0, 1}, 32);
    expectTrue(dl.data().indices.size() == 3u * 32u,
               "addSectorFilled (solid): 32 steps × 3 indices each");
    expectTrue(dl.data().vertices.size() == 3u * 32u,
               "addSectorFilled (solid): 32 steps × 3 vertices each");

    // A ring sector produces 32 quads = 192 indices.
    dl.reset(UiVec2{200.0f, 200.0f});
    dl.addSectorFilled(UiVec2{100.0f, 100.0f}, 20.0f, 40.0f,
                       0.0f, kPi * 0.5f, UiColor{1, 0, 0, 1}, 32);
    expectTrue(dl.data().indices.size() == 6u * 32u,
               "addSectorFilled (ring): 32 steps × 6 indices each");
}

void testToastManager() {
    using namespace odai::ui;
    Font font = makeMonospaceFont(8.0f);
    ToastManager mgr(&font);
    mgr.setRect(UiRect::fromXYWH(0.0f, 0.0f, 400.0f, 300.0f));
    mgr.displaySeconds = 1.0f;

    expectTrue(mgr.activeCount() == 0, "ToastManager: starts empty");
    mgr.push("", "City captured!");
    expectTrue(mgr.activeCount() == 1, "ToastManager: push adds a toast");
    mgr.push("", "New era reached!");
    expectTrue(mgr.activeCount() == 2, "ToastManager: two toasts active");

    // Advance past display time (1s display + 0.3s fade = 1.3s lifetime).
    mgr.update(1.4f);
    expectTrue(mgr.activeCount() == 0, "ToastManager: toasts expire after displaySeconds");

    // Emits geometry.
    mgr.push("", "Testing...");
    UiDrawList dl;
    dl.reset(UiVec2{800.0f, 600.0f});
    mgr.draw(dl);
    expectTrue(!dl.data().vertices.empty(), "ToastManager: emits geometry for active toast");
}

void testLineChartDraw() {
    using namespace odai::ui;
    LineChart chart;
    chart.setRect(UiRect::fromXYWH(0.0f, 0.0f, 200.0f, 100.0f));
    chart.series.push_back(ChartSeries{{10.0f, 50.0f, 30.0f, 80.0f},
                                       UiColor{0.4f, 0.8f, 0.4f, 1.0f}, "Score"});
    chart.series.push_back(ChartSeries{{5.0f, 25.0f, 60.0f, 40.0f},
                                       UiColor{0.8f, 0.4f, 0.4f, 1.0f}, "Rival"});

    UiDrawList dl;
    dl.reset(UiVec2{400.0f, 300.0f});
    chart.draw(dl);
    expectTrue(!dl.data().vertices.empty(), "LineChart: emits geometry for two series");
}

void testStatBadgeDraw() {
    using namespace odai::ui;
    Font font = makeMonospaceFont(8.0f);
    StatBadgeRow row(&font);
    row.setRect(UiRect::fromXYWH(0.0f, 0.0f, 200.0f, 24.0f));
    row.stats = {
        Stat{"", "ATK: 12", UiColor{1.0f, 0.6f, 0.2f, 1.0f}},
        Stat{"", "DEF: 8",  UiColor{0.4f, 0.8f, 1.0f, 1.0f}},
    };

    UiDrawList dl;
    dl.reset(UiVec2{400.0f, 200.0f});
    row.draw(dl);
    expectTrue(!dl.data().vertices.empty(), "StatBadgeRow: emits geometry for two stats");
}

}  // namespace

void testCachedRichTextRebuilds() {
    using namespace odai::ui;
    Font font = makeMonospaceFont();
    FontSet fonts{&font, &font, &font, &font};
    CachedRichText cache(fonts, "hello world");
    UiDrawList dl;
    dl.reset(UiVec2{800.0f, 600.0f});

    const UiRect r0 = UiRect::fromXYWH(10.0f, 10.0f, 200.0f, 50.0f);
    cache.emit(dl, r0);
    expectTrue(cache.rebuildCount() == 1u, "cache builds once on first emit");

    cache.emit(dl, r0);
    expectTrue(cache.rebuildCount() == 1u, "second emit at same rect reuses cache");

    // Move only (same size) must NOT rebuild — just re-translate.
    cache.emit(dl, UiRect::fromXYWH(100.0f, 80.0f, 200.0f, 50.0f));
    expectTrue(cache.rebuildCount() == 1u, "move-only (same size) reuses cache");

    // Width change rebuilds (wrap width differs).
    cache.emit(dl, UiRect::fromXYWH(100.0f, 80.0f, 120.0f, 50.0f));
    expectTrue(cache.rebuildCount() == 2u, "width change rebuilds cache");

    // Content change rebuilds.
    cache.setMarkup("different text here");
    cache.emit(dl, UiRect::fromXYWH(100.0f, 80.0f, 120.0f, 50.0f));
    expectTrue(cache.rebuildCount() == 3u, "setText rebuilds cache");

    // Color change rebuilds; an identical color does not.
    cache.setColor(UiColor{1.0f, 0.0f, 0.0f, 1.0f});
    cache.emit(dl, UiRect::fromXYWH(100.0f, 80.0f, 120.0f, 50.0f));
    expectTrue(cache.rebuildCount() == 4u, "color change rebuilds cache");
    cache.setColor(UiColor{1.0f, 0.0f, 0.0f, 1.0f});
    cache.emit(dl, UiRect::fromXYWH(100.0f, 80.0f, 120.0f, 50.0f));
    expectTrue(cache.rebuildCount() == 4u, "identical color does not rebuild cache");
}

void testAppendCachedParity() {
    using namespace odai::ui;
    Font font = makeMonospaceFont();
    FontSet fonts{&font, &font, &font, &font};
    const std::string markup = "the quick brown fox jumps over";
    const UiRect rect = UiRect::fromXYWH(25.0f, 40.0f, 150.0f, 90.0f);
    const UiColor color{0.80f, 0.90f, 0.95f, 1.0f};

    // Cached path: build a block once, emit it.
    CachedRichText cache(fonts, markup);
    cache.setColor(color);
    UiDrawList cached;
    cached.reset(UiVec2{800.0f, 600.0f});
    cache.emit(cached, rect);

    // Direct path: lay out + draw immediately under the same clip + origin.
    UiDrawList direct;
    direct.reset(UiVec2{800.0f, 600.0f});
    const RichTextLayout layout = layoutRichText(markup, color, fonts, rect.width(), UiTextAlign::Left);
    direct.pushClip(rect);
    drawRichText(direct, layout, fonts, UiVec2{rect.minX, rect.minY});
    direct.popClip();

    const UiDrawData& a = cached.data();
    const UiDrawData& b = direct.data();
    expectTrue(!a.vertices.empty(), "cached emit produced geometry");
    expectTrue(a.vertices.size() == b.vertices.size(), "cached vertex count matches direct");
    expectTrue(a.indices.size() == b.indices.size(), "cached index count matches direct");

    bool vertsMatch = a.vertices.size() == b.vertices.size();
    for (std::size_t i = 0; vertsMatch && i < a.vertices.size(); ++i) {
        const UiVertex& va = a.vertices[i];
        const UiVertex& vb = b.vertices[i];
        if (va.posPx[0] != vb.posPx[0] || va.posPx[1] != vb.posPx[1] || va.uv[0] != vb.uv[0] ||
            va.uv[1] != vb.uv[1] || va.rgba8 != vb.rgba8 || va.mode != vb.mode) {
            vertsMatch = false;
        }
    }
    expectTrue(vertsMatch, "cached vertices identical to direct");

    bool idxMatch = a.indices.size() == b.indices.size();
    for (std::size_t i = 0; idxMatch && i < a.indices.size(); ++i) {
        if (a.indices[i] != b.indices[i]) {
            idxMatch = false;
        }
    }
    expectTrue(idxMatch, "cached indices identical to direct");
}

void testEasing() {
    using namespace odai::ui;
    expectNear(applyEasing(Easing::Linear, 0.0f), 0.0f, 1e-5f, "linear 0");
    expectNear(applyEasing(Easing::Linear, 1.0f), 1.0f, 1e-5f, "linear 1");
    expectNear(applyEasing(Easing::EaseInOut, 0.0f), 0.0f, 1e-5f, "easeInOut start");
    expectNear(applyEasing(Easing::EaseInOut, 1.0f), 1.0f, 1e-5f, "easeInOut end");
    expectNear(applyEasing(Easing::EaseInOut, 0.5f), 0.5f, 1e-5f, "easeInOut symmetric midpoint");
    expectTrue(applyEasing(Easing::EaseOut, 0.5f) > 0.5f, "easeOut leads linear at mid");
    expectTrue(applyEasing(Easing::EaseIn, 0.5f) < 0.5f, "easeIn lags linear at mid");
    expectNear(applyEasing(Easing::Linear, 1.5f), 1.0f, 1e-5f, "easing clamps above 1");
    expectNear(applyEasing(Easing::Linear, -0.5f), 0.0f, 1e-5f, "easing clamps below 0");
}

void testTween() {
    using namespace odai::ui;
    Tween t;
    t.durationSec = 1.0f;
    t.easing = Easing::Linear;
    t.setTarget(1.0f);
    t.update(0.5f);
    expectNear(t.value, 0.5f, 1e-5f, "tween advances half over half duration");
    t.update(0.5f);
    expectNear(t.value, 1.0f, 1e-5f, "tween reaches target");
    expectTrue(t.idle(), "tween idle at target");
    t.update(0.5f);
    expectNear(t.value, 1.0f, 1e-5f, "tween clamps at target (no overshoot)");
    t.setTarget(0.0f);
    t.update(0.25f);
    expectNear(t.value, 0.75f, 1e-5f, "tween fades back out");
}

void testDrawListOpacity() {
    using namespace odai::ui;

    // Direct emission under opacity scales the vertex alpha byte.
    UiDrawList dl;
    dl.reset(UiVec2{100.0f, 100.0f});
    dl.pushOpacity(0.5f);
    dl.addRectFilled(UiRect::fromXYWH(0.0f, 0.0f, 10.0f, 10.0f), UiColor{1.0f, 1.0f, 1.0f, 1.0f});
    dl.popOpacity();
    const std::uint32_t a = (dl.data().vertices[0].rgba8 >> 24) & 0xFFu;
    expectTrue(a >= 126u && a <= 128u, "pushOpacity halves emitted alpha (~127)");

    // After popOpacity the alpha is full again.
    dl.addRectFilled(UiRect::fromXYWH(0.0f, 0.0f, 10.0f, 10.0f), UiColor{1.0f, 1.0f, 1.0f, 1.0f});
    const std::uint32_t a2 = (dl.data().vertices[4].rgba8 >> 24) & 0xFFu;
    expectTrue(a2 == 255u, "alpha restored to full after popOpacity");

    // appendCached also honours the opacity stack.
    UiDrawList scratch;
    scratch.reset(UiVec2{100.0f, 100.0f});
    scratch.addRectFilled(UiRect::fromXYWH(0.0f, 0.0f, 10.0f, 10.0f), UiColor{1.0f, 1.0f, 1.0f, 1.0f});
    UiGeometryBlock block;
    block.vertices = scratch.data().vertices;
    block.indices = scratch.data().indices;
    block.commands = scratch.data().commands;

    UiDrawList dl2;
    dl2.reset(UiVec2{100.0f, 100.0f});
    dl2.pushOpacity(0.5f);
    dl2.appendCached(block, UiVec2{0.0f, 0.0f});
    dl2.popOpacity();
    const std::uint32_t ac = (dl2.data().vertices[0].rgba8 >> 24) & 0xFFu;
    expectTrue(ac >= 126u && ac <= 128u, "appendCached applies opacity to cached alpha");
}

void testInlineIconParsing() {
    using namespace odai::ui;
    const std::vector<RichSpan> spans =
        parseRichText("food: [icon=food] +3", UiColor{1, 1, 1, 1});
    bool foundIcon = false;
    for (const RichSpan& s : spans) {
        if (s.iconName == "food") {
            foundIcon = true;
            expectTrue(s.text.empty(), "icon span has empty text");
        }
    }
    expectTrue(foundIcon, "[icon=food] tag parsed into a span with iconName='food'");
}

void testInlineIconLayout() {
    using namespace odai::ui;
    const Font font = makeMonospaceFont(10.0f);
    const FontSet fonts{&font, &font, &font, &font};
    const RichTextLayout layout =
        layoutRichText("cost: [icon=gold] 5", UiColor{1, 1, 1, 1}, fonts, 400.0f);
    bool foundIconRun = false;
    for (const RichLine& line : layout.lines) {
        for (const RichRun& run : line.runs) {
            if (!run.iconName.empty() && run.iconName == "gold") {
                foundIconRun = true;
                expectTrue(run.width > 0.0f, "icon run has positive width");
            }
        }
    }
    expectTrue(foundIconRun, "[icon=gold] produces a run with iconName='gold' after layout");
}

void testBindingContextResolve() {
    using namespace odai::ui;
    const char* jsonText = R"({ "city": { "name": "Balmora", "population": 4200 } })";
    auto root = JsonDataNode::fromString(jsonText);
    expectTrue(root != nullptr, "JsonDataNode::fromString parses valid JSON");
    if (!root) return;

    BindingContext ctx;
    auto cityNode = root->getChild("city");
    expectTrue(cityNode != nullptr, "getChild('city') returns a node");
    if (!cityNode) return;
    ctx.bind("city", cityNode);

    const std::string resolved = ctx.resolve("{city.name}");
    expectTrue(resolved == "Balmora", "BindingContext resolves {city.name}");
    expectNear(ctx.resolveFloat("{city.population}"), 4200.0f, 0.5f,
               "BindingContext resolves {city.population} as float");
}

void testHorizontalStackLayout() {
    using namespace odai::ui;
    auto stack = std::make_unique<HorizontalStack>();
    stack->gap = 4.0f;
    stack->setRect(UiRect::fromXYWH(0.0f, 0.0f, 300.0f, 40.0f));

    auto a = std::make_unique<Widget>(); a->setRect(UiRect::fromXYWH(0, 0, 80, 40));
    auto b = std::make_unique<Widget>(); b->setRect(UiRect::fromXYWH(0, 0, 60, 40));
    Widget* aPtr = stack->addChild(std::move(a));
    Widget* bPtr = stack->addChild(std::move(b));

    UiDrawList dl;
    dl.reset(UiVec2{800.0f, 600.0f});
    stack->draw(dl);

    expectNear(aPtr->rect().minX, 0.0f, 0.5f, "HStack: first child starts at 0");
    expectNear(bPtr->rect().minX, 84.0f, 0.5f, "HStack: second child starts after first + gap");
}

void testScrollViewLayout() {
    using namespace odai::ui;
    auto sv = std::make_unique<ScrollView>();
    sv->setRect(UiRect::fromXYWH(0.0f, 0.0f, 200.0f, 100.0f));
    sv->childGap = 4.0f;

    for (int i = 0; i < 5; ++i) {
        auto child = std::make_unique<Widget>();
        child->setRect(UiRect::fromXYWH(0, 0, 200, 30));
        sv->addChild(std::move(child));
    }

    UiDrawList dl;
    dl.reset(UiVec2{800.0f, 600.0f});
    sv->draw(dl);

    // 5 children of 30px + 4 gaps of 4px = 166px total content height.
    expectNear(sv->contentHeight(), 166.0f, 0.5f,
               "ScrollView: content height = 5*30 + 4*4 = 166");
}

void testProgressBarDraw() {
    using namespace odai::ui;
    ProgressBar bar;
    bar.setRect(UiRect::fromXYWH(10.0f, 10.0f, 200.0f, 12.0f));
    bar.value = 0.6f;

    UiDrawList dl;
    dl.reset(UiVec2{800.0f, 600.0f});
    bar.draw(dl);
    expectTrue(!dl.data().vertices.empty(), "ProgressBar emits geometry");
}

void testVectorPrimitives() {
    using namespace odai::ui;
    constexpr auto kRoundRect = static_cast<std::uint32_t>(UiDrawMode::RoundRect);

    // Filled rounded rect: one quad, RoundRect mode, sdf = {halfW, halfH, r, 0}.
    {
        UiDrawList dl;
        dl.reset(UiVec2{200.0f, 200.0f});
        dl.addRoundRectFilled(UiRect::fromXYWH(20.0f, 30.0f, 100.0f, 40.0f), UiColor{1, 1, 1, 1}, 8.0f);
        const UiDrawData& d = dl.data();
        expectTrue(d.vertices.size() == 4u, "RoundRectFilled emits one quad");
        expectTrue(d.indices.size() == 6u, "RoundRectFilled emits 6 indices");
        const UiVertex& v = d.vertices[0];
        expectTrue(v.mode == kRoundRect, "RoundRectFilled uses RoundRect mode");
        expectNear(v.sdf[0], 50.0f, 0.01f, "RoundRect halfWidth");
        expectNear(v.sdf[1], 20.0f, 0.01f, "RoundRect halfHeight");
        expectNear(v.sdf[2], 8.0f, 0.01f, "RoundRect radius");
        expectNear(v.sdf[3], 0.0f, 0.01f, "RoundRectFilled has zero border");
    }

    // Radius is clamped to half the shorter side (pill from an oversized radius).
    {
        UiDrawList dl;
        dl.reset(UiVec2{200.0f, 200.0f});
        dl.addRoundRectFilled(UiRect::fromXYWH(0.0f, 0.0f, 120.0f, 40.0f), UiColor{1, 1, 1, 1}, 999.0f);
        expectNear(dl.data().vertices[0].sdf[2], 20.0f, 0.01f, "Oversized radius clamps to half-height (pill)");
    }

    // Stroke carries thickness in sdf.w; the quad grows to fit the outer half.
    {
        UiDrawList dl;
        dl.reset(UiVec2{200.0f, 200.0f});
        dl.addRoundRect(UiRect::fromXYWH(50.0f, 50.0f, 60.0f, 60.0f), UiColor{1, 1, 1, 1}, 10.0f, 4.0f);
        const UiVertex& v = dl.data().vertices[0];
        expectNear(v.sdf[3], 4.0f, 0.01f, "RoundRect stroke records thickness");
        expectTrue(v.posPx[0] < 50.0f, "Stroke quad grows outward for AA + outer half");
    }

    // Circle: square bounds centred on the point, sdf radius == circle radius.
    {
        UiDrawList dl;
        dl.reset(UiVec2{200.0f, 200.0f});
        dl.addCircleFilled(UiVec2{100.0f, 100.0f}, 25.0f, UiColor{1, 1, 1, 1});
        const UiVertex& v = dl.data().vertices[0];
        expectTrue(v.mode == kRoundRect, "Circle uses RoundRect SDF mode");
        expectNear(v.sdf[0], 25.0f, 0.01f, "Circle halfWidth == radius");
        expectNear(v.sdf[1], 25.0f, 0.01f, "Circle halfHeight == radius");
        expectNear(v.sdf[2], 25.0f, 0.01f, "Circle radius");
    }

    // Glow: one quad in RoundRectGlow mode, grown beyond the rect, sdf.w = glow px.
    {
        UiDrawList dl;
        dl.reset(UiVec2{200.0f, 200.0f});
        dl.addRoundRectGlow(UiRect::fromXYWH(40.0f, 40.0f, 80.0f, 30.0f), UiColor{1, 0.8f, 0.3f, 0.6f},
                            8.0f, 14.0f);
        const UiDrawData& d = dl.data();
        expectTrue(d.vertices.size() == 4u, "Glow emits one quad");
        const UiVertex& v = d.vertices[0];
        expectTrue(v.mode == static_cast<std::uint32_t>(UiDrawMode::RoundRectGlow), "Glow uses glow mode");
        expectNear(v.sdf[3], 14.0f, 0.01f, "Glow records falloff size in sdf.w");
        expectTrue(v.posPx[0] < 40.0f, "Glow quad extends outside the rect for the halo");
    }

    // Degenerate inputs emit nothing.
    {
        UiDrawList dl;
        dl.reset(UiVec2{200.0f, 200.0f});
        dl.addCircleFilled(UiVec2{10.0f, 10.0f}, 0.0f, UiColor{1, 1, 1, 1});
        dl.addRoundRectFilled(UiRect{0, 0, 0, 0}, UiColor{1, 1, 1, 1}, 4.0f);
        dl.addRoundRect(UiRect::fromXYWH(0, 0, 10, 10), UiColor{1, 1, 1, 1}, 2.0f, 0.0f);
        dl.addRoundRectGlow(UiRect::fromXYWH(0, 0, 10, 10), UiColor{1, 1, 1, 1}, 2.0f, 0.0f);
        expectTrue(dl.data().vertices.empty(), "Degenerate vector primitives emit no geometry");
    }
}

void testButtonHoverGlow() {
    using namespace odai::ui;
    const Font font = makeMonospaceFont(8.0f);
    Button btn(&font, "Go", []() {});
    btn.setRect(UiRect::fromXYWH(20.0f, 20.0f, 100.0f, 40.0f));
    btn.cornerRadiusPx = 8.0f;
    btn.glowSizePx = 14.0f;

    auto glowVerts = [](const UiDrawList& dl) {
        std::size_t n = 0;
        for (const auto& v : dl.data().vertices) {
            if (v.mode == static_cast<std::uint32_t>(UiDrawMode::RoundRectGlow)) ++n;
        }
        return n;
    };

    // Normal state: no glow.
    {
        UiDrawList dl; dl.reset(UiVec2{400.0f, 300.0f});
        btn.draw(dl);
        expectTrue(glowVerts(dl) == 0u, "Button has no glow when not hovered");
    }
    // Hover the button -> glow appears (one quad = 4 vertices).
    {
        UiEvent move{}; move.type = UiEvent::Type::MouseMove; move.mousePx = UiVec2{50.0f, 35.0f};
        btn.onEvent(move);
        UiDrawList dl; dl.reset(UiVec2{400.0f, 300.0f});
        btn.draw(dl);
        expectTrue(glowVerts(dl) == 4u, "Hovered button draws a glow halo");
    }
    // glowSizePx = 0 disables it even when hovered.
    {
        btn.glowSizePx = 0.0f;
        UiDrawList dl; dl.reset(UiVec2{400.0f, 300.0f});
        btn.draw(dl);
        expectTrue(glowVerts(dl) == 0u, "glowSizePx=0 disables the glow");
    }
}

void testTextBox() {
    using namespace odai::ui;
    Font font = makeMonospaceFont(8.0f);
    TextBox box(&font, "placeholder");
    box.setRect(UiRect::fromXYWH(0.0f, 0.0f, 200.0f, 40.0f));

    auto sendText = [&](std::uint32_t cp) {
        UiEvent e{}; e.type = UiEvent::Type::Text; e.codepoint = cp; box.onEvent(e); return e.handled;
    };
    auto click = [&](float x, float y) {
        UiEvent e{}; e.type = UiEvent::Type::MouseDown; e.button = UiMouseButton::Left;
        e.mousePx = UiVec2{x, y}; box.onEvent(e); return e.handled;
    };

    // Typing is ignored until the box is focused.
    expectTrue(!sendText('a'), "Unfocused TextBox ignores text");
    expectTrue(box.value().empty(), "Unfocused TextBox stays empty");

    // Click inside focuses; subsequent characters are appended.
    expectTrue(click(20.0f, 20.0f), "Click inside focuses TextBox");
    expectTrue(box.focused(), "TextBox is focused after click inside");
    sendText('H'); sendText('i'); sendText('!');
    expectTrue(box.value() == "Hi!", "Focused TextBox appends typed characters");

    // Backspace (codepoint 8) deletes the last character.
    sendText(8u);
    expectTrue(box.value() == "Hi", "Backspace deletes the last character");

    // Enter (codepoint 13) fires onSubmit and does not modify the value.
    bool submitted = false;
    box.onSubmit = [&]() { submitted = true; };
    sendText(13u);
    expectTrue(submitted, "Enter fires onSubmit");
    expectTrue(box.value() == "Hi", "Enter leaves the value unchanged");

    // Clicking outside removes focus and stops accepting text.
    click(500.0f, 500.0f);
    expectTrue(!box.focused(), "Click outside unfocuses TextBox");
    expectTrue(!sendText('x'), "Unfocused-again TextBox ignores text");
    expectTrue(box.value() == "Hi", "Value unchanged while unfocused");

    // It emits geometry (vector frame + text) without crashing.
    UiDrawList dl;
    dl.reset(UiVec2{400.0f, 300.0f});
    box.draw(dl);
    expectTrue(!dl.data().vertices.empty(), "TextBox emits geometry");
}

void testToolbar() {
    using namespace odai::ui;
    constexpr auto kTextured = static_cast<std::uint32_t>(UiDrawMode::Textured);
    Font font = makeMonospaceFont(8.0f);
    UiIconRegistry::global().registerAtlas(
        42u, 384u, 256u,
        "{\"iconSize\":128,\"icons\":{\"science\":[0,0],\"gold\":[2,0]}}");

    Toolbar tb(&font);
    tb.setRect(UiRect::fromXYWH(0.0f, 0.0f, 600.0f, 40.0f));
    const std::size_t gold = tb.addItem(Toolbar::IconKind::Coin, UiColor{1, 1, 0, 1}, "240", UiColor{1, 1, 1, 1});
    tb.addItem(Toolbar::IconKind::Science, UiColor{0, 0.5f, 1, 1}, "+14", UiColor{1, 1, 1, 1});
    expectTrue(tb.itemCount() == 2u, "Toolbar tracks added items");

    UiDrawList dl;
    dl.reset(UiVec2{800.0f, 600.0f});
    tb.draw(dl);
    const UiDrawData& d = dl.data();
    expectTrue(!d.vertices.empty(), "Toolbar emits geometry");

    bool hasTexturedIcon = false;
    for (const auto& v : d.vertices) {
        if (v.mode == kTextured) { hasTexturedIcon = true; break; }
    }
    expectTrue(hasTexturedIcon, "Toolbar icons use registered atlas textures");

    // setValue is bounds-checked and updates the badge without crashing.
    tb.setValue(gold, "999");
    tb.setValue(99u, "ignored");  // out of range: no-op
    UiDrawList dl2;
    dl2.reset(UiVec2{800.0f, 600.0f});
    tb.draw(dl2);
    expectTrue(!dl2.data().vertices.empty(), "Toolbar redraws after setValue");
}

void testPanelOpacity() {
    using namespace odai::ui;

    // A Panel with opacity=0.5 emits background geometry at half alpha.
    {
        auto panel = std::make_unique<Panel>();
        panel->setRect(UiRect::fromXYWH(0.0f, 0.0f, 100.0f, 100.0f));
        panel->background = UiColor{1.0f, 1.0f, 1.0f, 1.0f};
        panel->opacity = 0.5f;

        UiContext ctx;
        ctx.setViewport(UiVec2{800.0f, 600.0f});
        ctx.setRoot(std::move(panel));

        UiDrawList dl;
        ctx.build(dl);

        const std::uint32_t a = (dl.data().vertices[0].rgba8 >> 24) & 0xFFu;
        expectTrue(a >= 126u && a <= 128u, "Panel opacity=0.5 halves background alpha");
    }

    // A nested Panel with opacity=0.5 inside an opaque Panel also halves alpha.
    {
        auto root = std::make_unique<Panel>();
        root->setRect(UiRect::fromXYWH(0.0f, 0.0f, 200.0f, 200.0f));
        root->background = UiColor{0.0f, 0.0f, 0.0f, 0.0f}; // transparent root

        auto child = std::make_unique<Panel>();
        child->setRect(UiRect::fromXYWH(10.0f, 10.0f, 80.0f, 80.0f));
        child->background = UiColor{1.0f, 1.0f, 1.0f, 1.0f};
        child->opacity = 0.5f;
        root->addChild(std::move(child));

        UiContext ctx;
        ctx.setViewport(UiVec2{800.0f, 600.0f});
        ctx.setRoot(std::move(root));

        UiDrawList dl;
        ctx.build(dl);

        // Scan all vertices: the child background quad should have alpha ~127.
        bool foundHalfAlpha = false;
        for (const auto& v : dl.data().vertices) {
            const std::uint32_t a = (v.rgba8 >> 24) & 0xFFu;
            if (a >= 126u && a <= 128u) { foundHalfAlpha = true; break; }
        }
        expectTrue(foundHalfAlpha, "Child Panel opacity=0.5 halves its background alpha");
    }

    // opacity=0 hides the panel entirely (alpha=0).
    {
        auto panel = std::make_unique<Panel>();
        panel->setRect(UiRect::fromXYWH(0.0f, 0.0f, 100.0f, 100.0f));
        panel->background = UiColor{1.0f, 1.0f, 1.0f, 1.0f};
        panel->opacity = 0.0f;

        UiContext ctx;
        ctx.setViewport(UiVec2{800.0f, 600.0f});
        ctx.setRoot(std::move(panel));

        UiDrawList dl;
        ctx.build(dl);

        expectTrue(dl.data().vertices.empty(), "Panel opacity=0 emits no geometry (invisible)");
    }
}

void testTurnsToBuild() {
    using namespace odai::game;
    expectTrue(turnsToBuild(60, 8, 3) == 18, "ceil((60-8)/3) == 18");
    expectTrue(turnsToBuild(60, 60, 3) == 0, "Already-met cost yields 0 turns");
    expectTrue(turnsToBuild(60, 70, 3) == 0, "Over-met cost yields 0 turns");
    expectTrue(turnsToBuild(60, 0, 0) == 60, "Zero per-turn is clamped to 1 (worst case)");
    expectTrue(turnsToBuild(10, 0, 4) == 3, "ceil(10/4) == 3");
    expectTrue(!defaultBuildables().empty(), "Default buildable catalog is non-empty");
}

void testProductionPanelBuild() {
    using namespace odai::ui;
    Font font = makeMonospaceFont(8.0f);
    FontSet fonts{&font, &font, &font, &font};

    std::vector<ProductionPanel::Row> rows;
    int selectCount = 0;
    int pediaCount = 0;
    for (const auto& item : odai::game::defaultBuildables()) {
        ProductionPanel::Row row;
        row.id = item.id;
        row.name = item.name;
        row.iconName = item.iconName;
        row.productionCost = item.productionCost;
        row.turns = odai::game::turnsToBuild(item.productionCost, 8, 3);
        row.onSelect = [&]() { ++selectCount; };
        row.onOpenPedia = [&]() { ++pediaCount; };
        rows.push_back(std::move(row));
    }

    ProductionPanel panel(fonts);
    panel.setItems(UiRect::fromXYWH(0.0f, 0.0f, 240.0f, 600.0f), 1.0f, "Production", rows);

    UiDrawList dl;
    dl.reset(UiVec2{800.0f, 600.0f});
    panel.draw(dl);
    expectTrue(!dl.data().vertices.empty(), "ProductionPanel emits geometry");

    // Selecting an item updates highlight without crashing.
    panel.setSelected(rows.front().id);

    // Click the first row (top-left area) to fire its onSelect.
    UiEvent down{};
    down.type = UiEvent::Type::MouseDown;
    down.button = UiMouseButton::Left;
    down.mousePx = UiVec2{40.0f, 80.0f};
    panel.onEvent(down);
    UiEvent up{};
    up.type = UiEvent::Type::MouseUp;
    up.button = UiMouseButton::Left;
    up.mousePx = UiVec2{40.0f, 80.0f};
    panel.onEvent(up);
    expectTrue(selectCount >= 1, "Clicking a production row fires onSelect");
}

int main() {
    testNineSliceQuadGen();
    testFontMeasure();
    testRichTextWrap();
    testRichTextSpans();
    testColorPacking();
    testButtonCallbackAndHitTest();
    testCachedRichTextRebuilds();
    testAppendCachedParity();
    testEasing();
    testTween();
    testDrawListOpacity();
    testInlineIconParsing();
    testInlineIconLayout();
    testBindingContextResolve();
    testHorizontalStackLayout();
    testScrollViewLayout();
    testProgressBarDraw();
    testVectorPrimitives();
    testButtonHoverGlow();
    testTextBox();
    testToolbar();
    testPanelOpacity();
    testSliderDrag();
    testToggleClick();
    testTabBarSwitch();
    testDropdownSelect();
    testDonutChartDraw();
    testSectorFilledGeometry();
    testToastManager();
    testLineChartDraw();
    testStatBadgeDraw();
    testTurnsToBuild();
    testProductionPanelBuild();

    if (g_failures != 0) {
        std::cerr << "[ui test] " << g_failures << " failures\n";
        return 1;
    }
    std::cout << "[ui test] all checks passed\n";
    return 0;
}
