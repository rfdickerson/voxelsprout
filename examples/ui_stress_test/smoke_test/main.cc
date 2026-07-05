// Smoke test for the offscreen capture harness itself (not a stress-test
// contestant). Builds a trivial scene and captures it, to prove the harness
// pipeline (headless Vulkan init -> font -> widget tree -> PNG) works before
// any contestant relies on it.
#include "offscreen_capture.h"

#include "ui/odai_ui.h"

#include <cstdio>
#include <memory>

using namespace odai::ui;
using namespace odai::uistress;

int main() {
    OffscreenCapture capture;
    OffscreenCapture::Config config{};
    config.width = 640;
    config.height = 400;
    if (!capture.init(config)) {
        std::fprintf(stderr, "harness init failed\n");
        return 1;
    }

    Font font;
    if (!capture.loadPrimaryFont(font, "assets/fonts/EBGaramond-Regular.ttf", 20.0f)) {
        std::fprintf(stderr, "font load failed\n");
        return 1;
    }

    UiContext ctx;
    ctx.setViewport(capture.sizePx());
    auto root = std::make_unique<Panel>();
    root->background = UiColor{0.10f, 0.11f, 0.13f, 1.0f};

    auto card = std::make_unique<Panel>();
    card->setRect(UiRect::fromXYWH(40.0f, 40.0f, 300.0f, 140.0f));
    card->styleCard(1.0f);

    auto label = std::make_unique<Label>(&font, "Harness smoke test OK");
    label->setRect(UiRect::fromXYWH(20.0f, 50.0f, 260.0f, 30.0f));
    card->addChild(std::move(label));

    auto button = std::make_unique<Button>(&font, "A Button", nullptr);
    button->setRect(UiRect::fromXYWH(20.0f, 90.0f, 140.0f, 36.0f));
    card->addChild(std::move(button));

    root->addChild(std::move(card));
    ctx.setRoot(std::move(root));

    UiInput input;
    ctx.update(input);
    UiDrawList drawList;
    ctx.build(drawList);

    const CaptureResult result = capture.captureToPng(drawList.data(), "harness_smoke_test.png");
    if (!result.success) {
        std::fprintf(stderr, "capture failed\n");
        return 1;
    }
    std::printf("capture OK: %u draw calls, %u commands, %u vertices, %u indices, %.3f ms submit->idle\n",
                result.drawCallCount, result.commandCount, result.vertexCount, result.indexCount,
                result.submitToIdleMs);
    capture.shutdown();
    return 0;
}
