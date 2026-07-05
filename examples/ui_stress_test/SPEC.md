# Stress test spec: City Management Panel

You are building this screen using **only `odai_ui`** (`src/ui/`, umbrella
header `ui/odai_ui.h`). Do not modify `src/ui/` itself — the point of this
exercise is to find out how the framework behaves *as it exists today*, not to
patch it. If something is awkward or missing, work around it (or don't,
depending on your assigned approach) and write it up in `REPORT.md`.

## What to build

A single screen: **the city management panel** for a 4X/city-builder game.
Root size: **1600×1000px** (this is the harness's off-screen render target
size — see `examples/ui_stress_test/harness/offscreen_capture.h`). No window,
no live input loop is required — this is a static screen render, though you
may drive the widget tree through a couple of different *states* if you want
extra captures (see "Deliverables").

The screen has three regions:

### 1. Resource bar (top strip, full width, ~64px tall)

Show **six yields**: Gold, Food, Production, Science, Culture, Faith. For
each: an icon-sized colored swatch (a plain colored square/circle is fine —
no icon assets are provided), the current amount, and a net-per-turn delta
(e.g. `1,240  (+18/turn)`). Lay them out evenly spaced across the full width.
Background should read as a distinct toolbar strip from the content below it
(a flat panel is fine, use `styleCard` or similar from `Panel`).

Use this mock data (hardcode it — no live game state exists):

| Yield | Amount | Per-turn |
|---|---|---|
| Gold | 1,240 | +18 |
| Food | 86 | +4 |
| Production | 52 | +11 |
| Science | 340 | +22 |
| Culture | 128 | +7 |
| Faith | 64 | −2 |

### 2. Building queue (left column, ~420px wide, below the resource bar)

A **vertical list of 5 queued items**, each row showing:
- Building name (e.g. "Granary", "Library", "Aqueduct", "Walls", "Market")
- A progress bar showing completion (0.0–1.0)
- Turns remaining (e.g. "3 turns")
- A way to visually distinguish the currently-in-progress item (row 0) from
  queued-but-not-started items (rows 1–4) — e.g. dimmed/muted styling for the
  latter.

Mock data:

| # | Name | Progress | Turns left |
|---|---|---|---|
| 0 | Granary | 0.65 | 3 |
| 1 | Library | 0.0 | 8 |
| 2 | Aqueduct | 0.0 | 12 |
| 3 | Walls | 0.0 | 6 |
| 4 | Market | 0.0 | 9 |

### 3. Production menu (right column, remaining width, below the resource bar)

A **scrollable list of at least 14 buildable items** (more than fit on
screen at once — this must actually scroll, not just be a static list that
happens to fit). Each row: building name, production cost (e.g. "120 🔨"),
one-line description. Group into at least 2 categories with a header row
(e.g. "Infrastructure" / "Military" / "Wonders") — headers are visually
distinct from item rows (different background/weight) and don't scroll out
of sync with their items.

Mock data — invent 14+ plausible city-builder buildings across 2–3
categories (e.g. Granary, Library, Aqueduct, Walls, Barracks, Stable,
Workshop, Market, Temple, Amphitheater, Harbor, Lighthouse, University,
Bank, Colosseum, Great Library — pick freely, keep costs/descriptions
plausible).

## Visual style

Match the existing "clean-modern" HUD look already established in this
codebase: flat cards, rounded corners, hairline borders, translucent dark
panels over a notional map background. Reference `Panel::styleCard()` in
`src/ui/widgets/panel.h` and `docs/UI_LIBRARY.md`'s theming section. You are
free to also load `assets/ui/themes/theme_modern.json` via `UiTheme` if you
find it useful, or just set colors directly on widgets — either is a valid
approach to evaluate.

## Constraints

- No hard frame-time budget, but don't do anything obviously wasteful (e.g.
  rebuilding the entire draw list geometry every single frame when nothing
  changed would be a flag — though for a one-shot static capture this mostly
  doesn't apply; note it in `REPORT.md` if your approach wouldn't scale to a
  live per-frame UI).
- Must build and produce a real PNG via the shared harness (see below).
- Use the mock data given above so all three results are visually comparable.

## The harness (already built, do not modify)

`examples/ui_stress_test/harness/` is a small static library
(`ui_stress_capture`) that gives you a headless (no window, no display
needed) Vulkan capture pipeline: build your `odai_ui` widget tree, hand its
`UiDrawData` to `OffscreenCapture::captureToPng(...)`, get a PNG on disk plus
draw-call/vertex/timing stats back. Full API in
`examples/ui_stress_test/harness/offscreen_capture.h`. See
`examples/ui_stress_test/smoke_test/main.cc` for a complete minimal usage
example (build it and look at `harness_smoke_test.png` in your build dir to
see a working reference render before you start).

Skeleton for your implementation:

```cpp
#include "offscreen_capture.h"
#include "ui/odai_ui.h"

int main() {
    odai::uistress::OffscreenCapture capture;
    odai::uistress::OffscreenCapture::Config config{};
    config.width = 1600;
    config.height = 1000;
    capture.init(config);

    odai::ui::Font font;
    capture.loadPrimaryFont(font, "assets/fonts/EBGaramond-Regular.ttf", 16.0f);

    // ... build your widget tree here, using `font` for text widgets ...

    odai::ui::UiContext ctx;
    ctx.setViewport(capture.sizePx());
    ctx.setRoot(std::move(root));
    odai::ui::UiInput input;  // no live input; empty snapshot is fine
    ctx.update(input);
    odai::ui::UiDrawList drawList;
    ctx.build(drawList);

    const auto result = capture.captureToPng(drawList.data(), "city_panel.png");
    // result.drawCallCount, result.vertexCount, result.indexCount, result.submitToIdleMs
}
```

Wire your executable target into your worktree's `CMakeLists.txt` the same
way `examples/ui_stress_test/smoke_test/CMakeLists.txt` does (link
`ui_stress_capture`, add a `slang_shaders` dependency if that target exists).
Build with `-DODAI_BUILD_EXAMPLES=ON`.

## Deliverables (in your worktree root)

1. Your implementation, wired into the build (must compile and run,
   producing a real PNG).
2. `city_panel.png` (or several PNGs if you captured more than one state —
   e.g. scrolled vs. not) at the repo root or your example directory —
   wherever your program writes it, just tell me the path in `REPORT.md`.
3. `REPORT.md` with:
   - Lines of code (your implementation only, not the harness)
   - Number of distinct `odai_ui` API calls/widget types used
   - Any workarounds, custom draw-list code, or code written *outside*
     `odai_ui`'s widget system to get the result you wanted
   - Draw call count / vertex count / index count / `submitToIdleMs` from
     `CaptureResult`
   - What the framework made easy
   - What the framework made hard or you had to work around
   - Any actual bugs you hit in `odai_ui` itself (not just missing features)
   - Your own confidence rating (1–5) that the result is visually/functionally
     correct against this spec, and why
