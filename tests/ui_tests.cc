#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>

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
#include "ui/widgets/progress_bar.h"
#include "ui/widgets/scroll_view.h"
#include "ui/widgets/spacer.h"
#include "ui/widgets/stack_layout.h"

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

    if (g_failures != 0) {
        std::cerr << "[ui test] " << g_failures << " failures\n";
        return 1;
    }
    std::cout << "[ui test] all checks passed\n";
    return 0;
}
