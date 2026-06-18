#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>

#include "ui/font.h"
#include "ui/rich_text.h"
#include "ui/ui_context.h"
#include "ui/ui_draw_list.h"
#include "ui/ui_input.h"
#include "ui/ui_types.h"
#include "ui/widgets/button.h"

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

int main() {
    testNineSliceQuadGen();
    testFontMeasure();
    testRichTextWrap();
    testRichTextSpans();
    testColorPacking();
    testButtonCallbackAndHitTest();

    if (g_failures != 0) {
        std::cerr << "[ui test] " << g_failures << " failures\n";
        return 1;
    }
    std::cout << "[ui test] all checks passed\n";
    return 0;
}
