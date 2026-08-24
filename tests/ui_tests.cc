#include <cmath>
#include <iostream>

#include "ui/font.h"
#include "ui/nav_focus.h"
#include "ui/nav_input.h"
#include "ui/toast_host.h"
#include "ui/ui_draw_list.h"
#include "ui/widget.h"
#include "ui/widgets/button.h"
#include "ui/widgets/slider.h"
#include "ui/widgets/tes3_journal_panel.h"
#include "ui/widgets/toggle.h"

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

odai::ui::Font makeFont() {
    odai::ui::Font font;
    font.initSyntheticMonospace(10.0f, 12.0f, 4.0f);
    return font;
}

void testDrawListAndFont() {
    using namespace odai::ui;
    const Font font = makeFont();
    expect(std::fabs(font.measureText("RPG") - 30.0f) < 0.001f,
           "font measurement remains deterministic");

    UiDrawList drawList;
    drawList.reset({320.0f, 200.0f});
    drawList.addRectFilled(UiRect::fromXYWH(4.0f, 4.0f, 40.0f, 20.0f), {1, 0, 0, 1});
    drawList.addRoundRectFilled(UiRect::fromXYWH(50.0f, 4.0f, 40.0f, 20.0f),
                                {0, 1, 0, 1}, 3.0f);
    drawList.addText(font, "HP", {8.0f, 8.0f}, {1, 1, 1, 1});
    expect(!drawList.data().vertices.empty() && !drawList.data().commands.empty(),
           "retained primitives and text emit draw data");
}

void testNativeFramebufferCoordinateContract() {
    using namespace odai::ui;
    // The world may render at a lower temporal-upscale input extent, but custom
    // UI is built in physical framebuffer pixels and composited onto the native
    // presentation image. Keep that distinction explicit in the retained data.
    constexpr UiVec2 nativeFramebuffer{1920.0f, 1080.0f};
    constexpr UiVec2 scaledWorldBuffer{864.0f, 486.0f};
    UiDrawList drawList;
    drawList.reset(nativeFramebuffer);
    drawList.addRectFilled(
        UiRect::fromXYWH(0.0f, 0.0f, nativeFramebuffer.x, nativeFramebuffer.y),
        {1.0f, 1.0f, 1.0f, 1.0f});

    const UiDrawData& data = drawList.data();
    expect(data.framebufferSizePx.x == nativeFramebuffer.x &&
               data.framebufferSizePx.y == nativeFramebuffer.y,
           "UI draw data retains the native physical framebuffer extent");
    expect(data.framebufferSizePx.x != scaledWorldBuffer.x &&
               data.framebufferSizePx.y != scaledWorldBuffer.y,
           "UI coordinate space is independent of the scaled 3D buffer");
    expect(!data.commands.empty() &&
               data.commands.front().clipRect.maxX == nativeFramebuffer.x &&
               data.commands.front().clipRect.maxY == nativeFramebuffer.y,
           "UI clipping covers the native presentation image");
}

void testButtonAndControls() {
    using namespace odai::ui;
    const Font font = makeFont();
    int clickCount = 0;
    Button button(&font, "Take", [&] { ++clickCount; });
    button.setRect(UiRect::fromXYWH(10.0f, 10.0f, 100.0f, 30.0f));

    UiEvent move{UiEvent::Type::MouseMove, {20.0f, 20.0f}};
    UiEvent down{UiEvent::Type::MouseDown, {20.0f, 20.0f}, UiMouseButton::Left};
    UiEvent up{UiEvent::Type::MouseUp, {20.0f, 20.0f}, UiMouseButton::Left};
    button.onEvent(move);
    button.onEvent(down);
    button.onEvent(up);
    expect(clickCount == 1, "button fires once for an inside press and release");

    Toggle toggle;
    toggle.setRect(UiRect::fromXYWH(0.0f, 0.0f, 48.0f, 24.0f));
    toggle.onEvent(down);
    expect(toggle.checked, "toggle changes state on click");

    Slider slider;
    slider.setRect(UiRect::fromXYWH(0.0f, 0.0f, 100.0f, 20.0f));
    UiEvent sliderDown{UiEvent::Type::MouseDown, {75.0f, 10.0f}, UiMouseButton::Left};
    slider.onEvent(sliderDown);
    expect(slider.value > 0.7f && slider.value < 0.8f, "slider maps pointer to value");
}

void testNavigation() {
    using namespace odai::ui;
    NavFocusRing focus;
    focus.beginFrame();
    focus.addItem(UiRect::fromXYWH(0, 0, 80, 24));
    focus.addItem(UiRect::fromXYWH(0, 40, 80, 24));

    UiNavInput input;
    input.setAction(UiNavAction::Down, true);
    focus.applyNavigation(input);
    expect(focus.focused() == 1, "directional navigation follows screen geometry");

    UiNavRepeater repeat({0.1f, 0.05f});
    input.beginFrame();
    repeat.update(input, 0.11f);
    expect(input.pressed(UiNavAction::Down), "held navigation repeats after its delay");
}

void testToastLifetime() {
    using namespace odai::ui;
    ToastTiming timing{};
    timing.fadeInSeconds = 0.01f;
    timing.holdSeconds = 0.01f;
    timing.fadeOutSeconds = 0.01f;
    ToastHost host({}, timing);
    host.push("Quest Updated", "Find the exit", "quest");
    host.update(0.005f);
    expect(host.visibleCount() == 1, "toast becomes visible");
    host.push("Quest Updated", "Find the exit", "quest");
    expect(host.visibleCount() == 1 && host.queuedCount() == 0,
           "keyed toast coalesces instead of stacking");
    host.update(1.0f);
    host.update(1.0f);
    host.update(1.0f);
    expect(host.visibleCount() == 0, "toast expires");
}

void testTes3JournalPanelModel() {
    using namespace odai::ui;
    const Font font = makeFont();
    Tes3JournalPanel panel(FontSet{&font, &font, &font, &font});
    panel.setJournal({
        {1u, "TR_Quest", "Temple Work", "The first authored entry", 10,
            Tes3JournalPanel::QuestState::Active, true},
        {2u, "Legacy", "An Old Matter", "A legacy entry", 5,
            Tes3JournalPanel::QuestState::Legacy, false},
        {3u, "TR_Quest", "Temple Work", "The latest authored entry", 20,
            Tes3JournalPanel::QuestState::Active, true},
        {4u, "Done", "Finished Work", "The final entry", 100,
            Tes3JournalPanel::QuestState::Completed, true}},
        {"Almas Thirr", "Bloodstone"});
    expect(panel.visibleCount() == 4u, "TES3 journal chronology retains every visit");
    panel.setView(Tes3JournalPanel::View::Active);
    expect(panel.visibleCount() == 1u && panel.visibleEntries()[0].index == 20,
           "active quest view shows the latest authored entry without objectives");
    panel.pinSelected();
    expect(panel.latestPinnedEntry().has_value() && panel.latestPinnedEntry()->index == 20,
           "pinned tracker resolves the quest's latest journal entry");
    panel.setView(Tes3JournalPanel::View::KnownTopics);
    panel.setSearch("blood");
    expect(panel.visibleCount() == 1u && panel.visibleEntries()[0].title == "Bloodstone",
           "known-topic view supports case-insensitive search");
    panel.rebuild(UiRect::fromXYWH(0.0f, 0.0f, 420.0f, 320.0f), 1.0f);
    UiDrawList drawList;
    drawList.reset({420.0f, 320.0f});
    panel.draw(drawList);
    expect(!drawList.data().vertices.empty(), "TES3 journal emits native retained UI geometry");
}

}  // namespace

int main() {
    testDrawListAndFont();
    testNativeFramebufferCoordinateContract();
    testButtonAndControls();
    testNavigation();
    testToastLifetime();
    testTes3JournalPanelModel();
    if (failures != 0) {
        return 1;
    }
    std::cout << "UI tests passed\n";
    return 0;
}
