# odai_ui: Early Access Release Plan

**Assumption:** treating this as a public release of the `odai_ui`/`odai_ui_vulkan` libraries (MIT-licensed, already pitched in [UI_LIBRARY.md](UI_LIBRARY.md) as vendorable into other projects, with a working external-integration sample at `examples/vulkan_ui_integration/`). Adjust if the intended venue/audience differs.

## Current state

**Core library is genuinely release-shaped already:** retained widget tree, JSON theming (`UiTheme`) with hot reload, rich text markup, vector icon pipeline, `Signal`/`SlotRegistry`, genre kits (`kits/strategy_4x_kit.h` etc.), CMake install/export (`find_package(odai_ui)` works), MIT license, versioned `0.1.0`.

**Two dev tools exist:**
- `odai_theme_viewer` — terminal ANSI-swatch theme previewer with hot reload. Works, keep as-is.
- `odai_ui_editor` ([src/tools/ui_editor/](../src/tools/ui_editor/)) — GLFW/Vulkan canvas editor. Its document core was rewritten (see Phase 2 below) and now authors the real `UiDocumentLoader` schema; what's still outstanding there is the Dear ImGui inspector and live theme preview.

**Tweens** ([animation.h](../src/ui/animation.h)) — **done**, see Phase 1 below.

## What "early access" should mean

Ship a smaller, honest core plus the two requested capabilities done well, rather than everything at once.

**Must have:** the tween rework and editor below at a genuinely usable bar, a documented 0.x versioning policy, packaging verified from a clean external project, a known-issues doc so adopters know what's intentionally unfinished.

**Explicitly out of scope for EA** (state this up front to avoid scope creep and set expectations): full flex/grid auto-layout beyond the existing single-axis stacks, promoting every `Panel::styleX()` skin to theme-JSON-driven tokens, spring-physics tweens beyond one basic damped-spring easing, non-Windows/non-Vulkan backends.

## Phase 1 — Tween rework — ✅ done

All seven items landed: `Vec2Tween`/`RectTween`, the cubic/back/spring easings,
`Sequence`, `Widget::onTick(dt)` driven from `UiContext`, and the retarget-seam fix
on the base `Tween`. Covered by `odai_ui_tests` / `odai_animation_tests`.

<details>
<summary>Original item list</summary>

1. Fix the retarget-seam in the base `Tween` itself (snapshot the eased value on `setTarget()` mid-flight, same trick `ColorTween::set()` already does) so every consumer gets seamless retargeting for free.
2. Add `Vec2Tween`/`RectTween` alongside the existing scalar and color tweens — needed for window slide-ins / card pop-ins.
3. Add missing easing curves: cubic in/out, back (overshoot), and one critically-damped spring option — covers modern "snap" game-feel without over-building.
4. Add a small `Sequence` type (`Append`/`Join`/`Delay`/`OnComplete`) so a toast-in→hold→toast-out or a modal pop-in+backdrop-fade can be authored as one composed sequence instead of a bespoke field per widget.
5. Centralize ticking: add `Widget::onTick(dt)` (default no-op) and drive it once per frame from `UiContext`, replacing the current pattern of app code manually calling `Panel::update(dt)`/`ToastManager::update(dt)` per instance.
6. Migrate the existing ad hoc tweens (`Panel::backgroundAnim`/`bgTopAnim`/`bgBotAnim`, `Toast::fadeTween`, the smart-turn-button glow pulse) onto the new primitives as the reference migration and regression check.
7. Add `odai_ui_tests` coverage: retarget-seam correctness, sequence ordering/`onComplete` firing, centralized tick dispatch.

</details>

## Phase 2 — Editor (the headline EA feature)

1. 🟡 Wire the already-linked Dear ImGui into `odai_ui_editor` for real inspector controls instead of the current hand-drawn step-buttons. The inspector is now generated from the `TypeDesc`/`PropDesc` schema table in `editor_document.h`, so this is a swap of the control-drawing layer, not a rewrite of what it edits. **Color is done and does not need ImGui**: `PropKind::Color` opens a hand-drawn HSV square + hue/alpha bars with a classical harmony picker (complementary, split-complementary, analogous, triad, tetrad, square, and monochromatic/saturation ramps) and a live WCAG contrast score against the surface the color actually sits on — see [editor_color.h](../src/tools/ui_editor/editor_color.h). What still wants better controls is numeric drags and enum dropdowns.
2. ✅ Replace the flat `DesignWidget` model with real `UiDocumentLoader`-schema authoring. `EditorDocument` ([editor_document.h](../src/tools/ui_editor/editor_document.h)) stores the JSON tree itself, so nesting, `"50%"` lengths, `frame` nine-slices, `on_click` slots, `{binding}` expressions and any app-specific field the editor has no inspector for all survive a load→save round-trip untouched.
3. ⬜ Live preview through the real `UiTheme` + `UiContext` + `odai_ui_vulkan` renderer. The canvas today draws its own approximation from the node's properties; it does not resolve theme color tokens or nine-slice frames (a token like `"panel.bg"` renders as a neutral placeholder and round-trips unchanged).
4. ✅ Undo/redo stack — `EditorHistory` ([editor_history.h](../src/tools/ui_editor/editor_history.h)), whole-document snapshots, one step per completed gesture.
5. ✅ Multi-select, align/distribute, copy/paste, duplicate, keyboard nudge, sibling reorder — `Selection` and friends in [editor_ops.h](../src/tools/ui_editor/editor_ops.h).
6. ✅ Save/export straight to the same `.ui.json` consumed by `UiDocumentLoader::load`, with a round-trip test. `odai_ui_editor_tests` authors a document, serializes it, instantiates it through the real `ui::UiDocumentLoader`, and asserts the resulting widget tree's ids, slot names and rects match the editor canvas — plus a test that opens the shipped `assets/ui/docs/city_panel.ui.json` and hands it back unchanged.
7. 🟡 Acceptance test — the headless half is covered by the round-trip tests above. The 5-minute hands-on flow (open editor → drop widgets → wire an `on_click` slot → run in-game) still needs a real Vulkan run.

> **Build caveat on the editor app itself.** The document core (`editor_document`,
> `editor_history`, `editor_ops`, `editor_snap`) is headless and covered by
> `odai_ui_editor_tests`. `ui_editor_app.cc` — the GLFW/Vulkan shell that drives
> it — was ported to the new core on a machine with no Vulkan SDK and has only
> been type-checked (`g++ -fsyntax-only`), never built or run. Verify it on a real
> Vulkan build before relying on its behaviour; this is the same caveat `CLAUDE.md`
> carries for `skinning_resources.cc`.

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

Phase 1 landed first, as planned — the editor's live preview and any
inspector-driven transitions want the improved tween primitives.

Within Phase 2, the remaining two items (ImGui inspector, live theme preview) both
need a working Vulkan build to develop against, so they are the natural next step
for whoever has one.
