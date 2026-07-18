---
name: new-game
description: Scaffold a brand-new mini-game in this engine's src/games/<name>/ directory using the existing engine::GameApp pattern — creates the app header/source/main files, registers the CMake target, and builds it. Use this whenever the user asks to build, add, scaffold, or prototype a new game or mini-game in this repo, e.g. "build a new game called X", "let's prototype a puzzle game", "add a snake-like game", "create a new game similar to minesweeper" — even if they don't say "scaffold" explicitly. Reads docs/GAME_API.md first instead of re-exploring src/engine, src/render, and src/ui from scratch each time, so new games get built fast and stay consistent with the existing ones (minesweeper, snake, citybuilder, stellaris, swtor).
---

# Scaffolding a new game

This engine (voxelsprout) already has a working pattern for mini-games: a `GameApp` base
class, a UI framework, and five reference implementations. Re-discovering that pattern by
reading the engine from scratch every time a new game gets built is slow. Instead:

## 1. Read the index first

Read [`docs/GAME_API.md`](../../../docs/GAME_API.md) before touching any source file. It
already indexes:
- the `engine::GameApp` lifecycle contract (`onInit`/`onTick`/`onRender`/`onShutdown`,
  `wantsMinimalRendering`, `loadFonts`, `submitFrame`)
- the relevant subset of the `Renderer` facade
- the UI framework (immediate-mode `UiDrawList` vs. retained `Widget` tree + `SlotRegistry`,
  the widget catalog, and the genre "kits" in `src/ui/kits/`)
- input conventions (`UiInput` for mouse, raw `glfwGetKey` + hand-rolled edge latches for
  keyboard — there's no input-mapping abstraction, and there shouldn't be one)
- the optional `sim/simulation.h` factory-sim module (belts/pipes/tracks) if the new game
  wants that
- a table of which existing game to copy from for which pattern
- the exact CMake boilerplate block to copy/rename

Do not re-derive any of this by grepping `src/render/renderer.h` or `src/ui/` cold — that
work has already been done and written down. Only fall back to reading the actual source
if `docs/GAME_API.md` seems stale or is missing something you need (see step 6).

## 2. Pick a reference game to copy from

Default to **`src/games/minesweeper/`** unless the request clearly matches another shape:

| New game is like... | Copy from | Why |
|---|---|---|
| A simple 2D board/puzzle game | `minesweeper/` | Smallest, cleanest mix of immediate-mode board drawing + a retained-widget toolbar. Good default. |
| A simple real-time arcade loop | `snake/` | Minimal game loop, ~370 lines, no widget tree. |
| A city/economy/management sim with a lot of custom HUD | `citybuilder/` | Larger, fully hand-rolled immediate-mode UI — reference for heavy `UiDrawList` usage. |
| A larger 4X or RPG-style genre game | `stellaris/` or `swtor/` | Bigger genre-specific examples if minesweeper/citybuilder don't fit. |

When unsure, ask the user which shape fits, or just default to minesweeper — it's the
cheapest starting point to strip down or grow from.

## 3. Scaffold the files

Read the chosen reference game's `_app.h`/`_app.cc`/`_main.cc` in full, then create the
new game's equivalents:

```
src/games/<name>/<name>_app.h     — class <Name>App : public engine::GameApp
src/games/<name>/<name>_app.cc    — onInit/onTick/onRender (+ onShutdown if needed)
src/games/<name>/<name>_main.cc   — the 4-line main() (see docs/GAME_API.md §2)
```

Keep the reference game's *shape* (lifecycle hooks, drawing style, namespace pattern
`odai::games::<name>`), not its game-specific state — strip out the board/rules code and
replace it with whatever the new game actually needs. Match the reference's choice of
`wantsMinimalRendering()` (almost always `true` for a self-contained 2D game — it skips
building 3D pipelines the game will never use).

## 4. Register the CMake target

Open `CMakeLists.txt`, find the `odai_game_minesweeper` block (search for
`# odai_game_minesweeper`, roughly line 720), copy the whole block, and rename every
`odai_game_minesweeper` → `odai_game_<name>` and every `minesweeper` path segment →
`<name>`. Paste it next to the other `odai_game_*` targets, inside the
`if(ODAI_RENDER_BACKEND_UPPER STREQUAL "VULKAN")` guard. The exact boilerplate is also
reproduced in `docs/GAME_API.md` §8 if you want to paste from there instead of re-reading
the CMake file.

## 5. Build just the new target

```powershell
cmake -S . -B cmake-build-release
cmake --build cmake-build-release --target odai_game_<name> -j 4
```

Fix any compile errors by cross-checking `docs/GAME_API.md` and the reference game's
source — don't guess at renderer/UI signatures. If the build is clean, run it to confirm
it opens a window (`cmake-build-release\odai_game_<name>.exe`, or use the `run` skill).

## 6. Keep the reference doc honest

If this scaffolding surfaces an API detail, gotcha, or pattern that isn't already in
`docs/GAME_API.md` — a renamed function, a new widget worth knowing about, a subtlety in
how `wantsMinimalRendering()` or the CMake sources list behaves — add it to that doc in
the same change. It's meant to stay accurate across many future "build a new game"
requests, not just this one.

## Guardrails (from AGENTS.md — these bind every game)

- Never include anything under `src/render/backend/vulkan/`, and never let a `Vk*` type
  appear in `src/games/<name>/` — only `src/render/` may touch Vulkan directly.
- Don't invent a new input-mapping layer, ECS, or other abstraction "for later" — follow
  the existing raw-`glfwGetKey` + `UiInput` conventions even though they're unglamorous.
- Don't base a new game on `src/app/app.cc` / `odai::app::App` — that's a separate, older
  Civ-style system that predates `GameApp` and shares no code with it.
- Keep `onTick`/`onRender` allocation-free where reasonable — performance is a stated
  project priority, not just for the 3D renderer.
