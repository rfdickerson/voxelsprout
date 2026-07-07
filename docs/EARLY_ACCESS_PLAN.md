# odai_ui: Early Access Release Plan

**Assumption:** treating this as a public release of the `odai_ui`/`odai_ui_vulkan` libraries (MIT-licensed, already pitched in [UI_LIBRARY.md](UI_LIBRARY.md) as vendorable into other projects, with a working external-integration sample at `examples/vulkan_ui_integration/`). Adjust if the intended venue/audience differs.

## Current state

**Core library is genuinely release-shaped already:** retained widget tree, JSON theming (`UiTheme`) with hot reload, rich text markup, vector icon pipeline, `Signal`/`SlotRegistry`, genre kits (`kits/strategy_4x_kit.h` etc.), CMake install/export (`find_package(odai_ui)` works), MIT license, versioned `0.1.0`.

**Two dev tools exist but are prototypes, not release-ready:**
- `odai_theme_viewer` — terminal ANSI-swatch theme previewer with hot reload. Works, keep as-is.
- `odai_ui_editor` ([src/tools/ui_editor/ui_editor_app.h](../src/tools/ui_editor/ui_editor_app.h)) — GLFW/Vulkan canvas with drag/resize/snap and a hand-drawn properties panel (`PropButton` step-buttons, not real inputs). It only knows a flat `DesignWidget` struct (bg/border/corner/shadow/label) — it doesn't speak the real `UiDocumentLoader` JSON schema (children, bindings, frames, nine-slice), doesn't preview through the actual `UiTheme`, has no undo, and links Dear ImGui without using it for the property panel.

**Tweens** ([animation.h](../src/ui/animation.h)) are a single `Tween`/`ColorTween` pair: 4 basic easings, no sequencing/composition, no `onComplete`, scalar+color only (no rect/position), retarget-seam fix only applied to `ColorTween` not the base `Tween`, and no central per-frame tick — `UiContext::update` takes only input, not `dt`, so animated widgets (`Panel::update(dt)`, `ToastManager::update(dt)`) are ticked ad hoc by app code.

## What "early access" should mean

Ship a smaller, honest core plus the two requested capabilities done well, rather than everything at once.

**Must have:** the tween rework and editor below at a genuinely usable bar, a documented 0.x versioning policy, packaging verified from a clean external project, a known-issues doc so adopters know what's intentionally unfinished.

**Explicitly out of scope for EA** (state this up front to avoid scope creep and set expectations): full flex/grid auto-layout beyond the existing single-axis stacks, promoting every `Panel::styleX()` skin to theme-JSON-driven tokens, spring-physics tweens beyond one basic damped-spring easing, non-Windows/non-Vulkan backends.

## Phase 1 — Tween rework (~1–2 weeks, do first)

1. Fix the retarget-seam in the base `Tween` itself (snapshot the eased value on `setTarget()` mid-flight, same trick `ColorTween::set()` already does) so every consumer gets seamless retargeting for free.
2. Add `Vec2Tween`/`RectTween` alongside the existing scalar and color tweens — needed for window slide-ins / card pop-ins.
3. Add missing easing curves: cubic in/out, back (overshoot), and one critically-damped spring option — covers modern "snap" game-feel without over-building.
4. Add a small `Sequence` type (`Append`/`Join`/`Delay`/`OnComplete`) so a toast-in→hold→toast-out or a modal pop-in+backdrop-fade can be authored as one composed sequence instead of a bespoke field per widget.
5. Centralize ticking: add `Widget::onTick(dt)` (default no-op) and drive it once per frame from `UiContext`, replacing the current pattern of app code manually calling `Panel::update(dt)`/`ToastManager::update(dt)` per instance.
6. Migrate the existing ad hoc tweens (`Panel::backgroundAnim`/`bgTopAnim`/`bgBotAnim`, `Toast::fadeTween`, the smart-turn-button glow pulse) onto the new primitives as the reference migration and regression check.
7. Add `odai_ui_tests` coverage: retarget-seam correctness, sequence ordering/`onComplete` firing, centralized tick dispatch.

## Phase 2 — Editor (~3–5 weeks, the headline EA feature)

1. Wire the already-linked Dear ImGui into `odai_ui_editor` for real inspector controls (color pickers, numeric drags, dropdowns) instead of the current hand-drawn step-buttons — probably the single biggest lever on "does this feel intuitive."
2. Replace the flat `DesignWidget` model with real `UiDocumentLoader`-schema authoring (children/nesting, frame/nine-slice, binding expressions) so the editor round-trips the actual `.ui.json` format apps consume, not a parallel toy format.
3. Live preview through the real `UiTheme` + `UiContext` + `odai_ui_vulkan` renderer, so the canvas shows the widget as it will actually render/theme — reuse the hot-reload plumbing that already exists.
4. Undo/redo stack (currently absent — table stakes for "intuitive").
5. Multi-select, align/distribute, copy/paste, keyboard nudge — standard editor expectations.
6. Save/export straight to the same `.ui.json` consumed by `UiDocumentLoader::load`, with a round-trip test (author in editor → load in-game → matches).
7. Acceptance test, not just a feature checklist: a 5-minute onboarding flow — open editor → open a sample theme → drop 3 widgets → wire an `on_click` slot → run in-game.

## Phase 3 — Packaging & docs polish (~1 week, can overlap Phase 2's tail)

1. Verify `find_package(odai_ui)` from a clean external project on a clean machine/CI (not just in-tree) — `examples/vulkan_ui_integration/` is the existing proof point.
2. Add `CHANGELOG.md` and a documented versioning policy (0.x = no API stability guarantee yet, but changes get logged).
3. Update `UI_LIBRARY.md` once the editor and new tween API land.
4. Write a known-issues/EA-scope doc (auto-layout gap, skin-theming gap, etc. — see the widget-system review from earlier in this session).
5. Clean up the untracked scratch artifacts currently sitting at repo root (`build_out.txt`, `debug_err.txt`, `debug_out.txt`, `smoke.txt`) and add them to `.gitignore` before going public.

## Phase 4 — Rollout (~few days)

1. Tag `v0.1.0-early-access`; release notes link back to the known-issues doc.
2. Pick a feedback channel — GitHub Discussions/Issues is the natural default given MIT + GitHub — and say so in the README.
3. Record a short demo GIF/video of the editor in action — the most persuasive asset for an early-access UI-library audience.

## Sequencing note

Do Phase 1 before Phase 2: the editor's live preview and any inspector-driven transitions (panel animating on selection, etc.) will want the improved tween primitives, and Phase 1 is the smaller, lower-risk piece to land and validate first.
