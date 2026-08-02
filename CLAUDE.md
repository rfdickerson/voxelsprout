# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Reference Touchstones

Four games define what this project is actually building toward. Each maps to an existing area of the codebase — none are being cloned literally, but each names the specific feel/systems to chase:

- **Morrowind** — open-world exploration, hand-placed regional identity, readable terrain and settlements (`src/world/`, `src/import/`).
- **Civilization VI** — turn-based hex-grid 4X strategy: yields, tech tree, borders, map modes/lenses, diplomacy (`src/game/`).
- **SimCity (2013)** — agent-driven city simulation: RCI zoning, traffic congestion, land value, service coverage, data-layer overlays (`src/games/citybuilder/`, already explicitly modeled on it).
- **Dragon Age: Origins** — party-based real-time-with-pause tactical combat, branching dialogue with consequences, companion relationships/approval. This is the least-built touchstone today — see `docs/ROADMAP.md`'s Party RPG / Narrative section.

These are creative direction, not a mandate to merge four genres into one game — each pillar can keep developing largely on its own track. `docs/ROADMAP.md` tracks concrete status and priority per touchstone.

## Build Commands

**Windows (app + tools):**
```powershell
cmake -S . -B cmake-build-release
cmake --build cmake-build-release --target odai -j 4
```

**Linux / WSL2 (tools + tests only, no Vulkan required):**
```bash
cmake -S . -B cmake-build-linux \
  -DODAI_BUILD_APP=OFF -DODAI_BUILD_TOOLS=ON -DBUILD_TESTING=ON \
  -DCMAKE_BUILD_TYPE=Debug
cmake --build cmake-build-linux -j 4
```

**Build a single test target:**
```powershell
cmake --build cmake-build-release --target odai_ui_tests
```

**Run all tests:**
```powershell
ctest --test-dir cmake-build-release -V
```

**Run a specific test suite:**
```powershell
ctest --test-dir cmake-build-release -R odai_ui_tests -V
```

**Test targets:**
- `odai_foundation_tests` — chunk grid, world gen, mesher
- `odai_ui_tests` — draw list, font metrics, rich text, widgets (headless, no Vulkan)
- `odai_strategy_map_tests` — hex grid model, serialization, mesher
- `odai_imported_scene_tests` — scene import/export round-trip
- `odai_fnv_import_tests` — Fallout: New Vegas import pipeline: BSA archive reader, ESM record walker, typed record extraction, NIF geometry parser (synthetic fixtures only — see README's "Fallout: New Vegas Import Pipeline" section)
- `odai_stability_gtests` — GTest suite covering frame graph, render math, sim network (requires GTest via vcpkg)

**Content generation tools:**
```powershell
cmake-build-release\odai_strategy_map_gen.exe          # generates strategy_map.smap + strategy_map_scene.bin
cmake-build-release\odai_balmora_cooker.exe "C:\GOG Games\Morrowind\Data Files" balmora.bin
cmake-build-release\odai_newvegas_cooker.exe "<Fallout New Vegas Data Files>" FalloutNV.esm fnv_scene.bin --cell <EditorID>
```

**Run the app (strategy map mode):**
```powershell
$env:ODAI_STRATEGY_MAP = "strategy_map.smap"
cmake-build-release\odai.exe
```

**Shaders** are compiled automatically by CMake when `slangc` is on PATH. Outputs are `.slang.spv` files next to the source. To compile a shader manually:
```bash
slangc -target spirv -entry main -stage fragment -matrix-layout-column-major \
  -I src/render/shaders src/render/shaders/ui.frag.slang -o src/render/shaders/ui.frag.slang.spv
```
Use `add_slang_shader_variant(..., -DODAI_RT_SHADOWS=1)` for define-based shader variants.

## Architecture

### Module boundaries

```
app/      — lifecycle, input routing, per-frame coordination (older Civ-style app; predates engine/)
core/     — math, time, logging (VOX_LOGE/W/I/D/T macros), input state
world/    — terrain, chunk grids, voxels, static placement
import/   — Bethesda asset parsing (Morrowind ESM/terrain, Fallout: New Vegas ESM/BSA/NIF in import/fnv/) + scene serialization
game/     — strategy map model, hex grid, serialization, mesh building
sim/      — factory simulation (pipes, belts, items)
ui/       — Vulkan-free UI framework: draw list, font, rich text, widget tree
render/   — public Renderer facade + everything Vulkan (only place that includes Vulkan headers)
engine/   — GameApp lifecycle base + PluginRegistry; what src/games/* build on (see docs/GAME_API.md)
tools/    — offline content generators (balmora cooker, map gen)
tests/    — correctness tests; no Vulkan in test executables
```

**Hard rule:** only `src/render/` may include Vulkan headers. No Vulkan types cross this boundary.

### Data flow

World state never flows back from the renderer. The direction is:

```
World/Game data  →  Meshing / ImportedScene  →  Renderer::upload*()
                                                       ↓
                                              FrameArena (per-frame GPU upload)
                                                       ↓
                                              Render passes (Shadow → Main → Post → UI)
                                                       ↓
                                              Swapchain present
```

The seam between `src/ui/` and the renderer is `Renderer::setUiDrawData(const ui::UiDrawData&)` + `setUiFontAtlas(...)`. All UI logic and geometry assembly happens Vulkan-free in `src/ui/`; `UiRenderer` only streams the resulting vertex/index data to the GPU.

### Renderer internals

- **`render/renderer.h`** — narrow public facade (~96 lines). Call `upload*` to push world data, `setUiDrawData` for UI, `renderFrame` to record and submit.
- **`render/backend/vulkan/renderer_backend.h`** — the actual Vulkan state machine. Owns instance, device, swapchain, command pools, pipelines, descriptor sets, and `FrameArena`. All per-pass recording happens in files named `frame_*.cc`.
- **`render/frame_graph.{h,cc}`** + **`frame_graph_runtime.cc`** — declarative pass dependency graph; resolves barriers and execution order.
- **`docs/FrameArena.md`** — how per-frame transient GPU memory works (two layers: host-visible upload arena and device-local scratch arena; reclaimed after timeline fence).

### UI framework

`src/ui/` is a fully headless, Vulkan-free retained widget tree on top of an immediate draw list:

- `UiDrawList` emits `UiDrawData` (quads, 9-slice, glyph-alpha quads, per-command texture + clip rect)
- `Font` bakes an R8 atlas via `stb_truetype` + `stb_rect_pack`
- `rich_text` parses `<b>/<i>/<color=#rrggbb>/<br>` markup, wraps, and aligns
- `Widget` → `Panel / Label / Button`; callbacks are `std::function<void()>`; `UiContext` owns the root and dispatches input
- `render/backend/vulkan/ui_renderer.cc` is the only UI file touching Vulkan: owns the alpha-blend pipeline, per-texture descriptor sets, and per-frame geometry streaming

Swapchain format is `B8G8R8A8_UNORM` (driver presents raw bytes, display interprets as sRGB). The UI fragment shader works in linear space and applies a manual `linearToSrgb` encode before output — matching the `pow(1/2.2)` the tonemapper applies for the 3-D pass. Vertex colors authored as sRGB hex are decoded to linear on entry so hex values are WYSIWYG. Color textures use `VK_FORMAT_R8G8B8A8_SRGB` image views so the sampler returns linear values.

### Engine plugins (proposed pattern, unproven — read before citing this as settled)

`GameApp` (`src/engine/game_app.h`) — the shared lifecycle base for `src/games/*` and
several tools — owns a `PluginRegistry`. The intent is an extension point: a game
registers `IEnginePlugin` implementations from its own `onInit()` to compose in optional
behavior (a debug overlay, a stats collector, a `mods/`-driven hook) without `GameApp` or
the base game needing to know about it. See `docs/GAME_API.md` §10 for the lifecycle
contract and the one ordering caveat (`onRender` must be invoked manually from the game's
own `onRender()`, after `beginFrameDraw()` and before `submitFrame()`).

**As of the commit that added it, `IEnginePlugin` has zero implementations anywhere in
this codebase** and has never been built against a real Vulkan+GLFW target (checked via
`g++ -fsyntax-only` only). Do not treat this as a settled architectural decision or cite
it as having superseded anything — that requires a real plugin, in a real game, on a real
build, first. If you're reading this and no plugin exists yet, treat the interface as a
draft: verify the shape still fits before building on it, and expect `GameApp`'s
`protected` members (`m_renderer`, `m_uiContext`, etc.) may need new accessors before a
plugin holding only `GameApp&` can actually do anything useful.

No dynamic loading (`dlopen`/DLL) is involved anywhere in this engine — a "plugin" is a
statically linked C++ class registered at startup, the same shape as the existing
`IModHost` seam (`src/game/mod_host.h`) and `mods/base` content loading
(`src/content/mod_loader.h`). Prefer composing through one of these three existing
interfaces over inventing a fourth.

### Hex strategy map

`src/game/` is pure CPU (no Vulkan, no imgui). Pointy-top hex grid in odd-r offset coordinates. `strategy_map_mesh.cc` converts a `StrategyMap` into an `ImportedScene` using the packed vertex-color render path (textureIndex=0xFFFFFFFF → per-vertex color), so no new renderer code is required to view a map.

### Shader system

Shaders use **Slang** (`.slang` → `.slang.spv` SPIR-V). Shared includes live in `src/render/shaders/`:
- `camera_uniform.slang` — MVP, inverse matrices, FOV, near/far planes
- `chunk_push_constants.slang` — per-draw chunk offset and LOD
- `fullscreen_triangle.slang` — clip-space triangle for post passes
- `sh_lighting.slang` — spherical-harmonics GI evaluation
- `voxel_decode.slang` — voxel color unpacking

Ray-traced shadow/reflection variants compile the same `.slang` source with `-DODAI_RT_SHADOWS=1` or `-DODAI_RT_REFLECTIONS=1`.

### Naming conventions

- Namespaces: `odai::app`, `odai::render`, `odai::world`, `odai::ui`, etc.
- Types: PascalCase — `StrategyMap`, `UiDrawList`, `RendererBackend`
- Functions: camelCase — `buildStrategyMapScene`, `setUiDrawData`
- Private members: `m_camelCase` prefix
- Module-scoped constants: `k` prefix — `kMaxFramesInFlight`, `kUiNoTexture`
- Source files: `.cc` (not `.cpp`)

## Local Paths

| Resource | Path |
|---|---|
| Morrowind Data Files (Windows) | `C:\GOG Games\Morrowind\Data Files` |
| Morrowind Data Files (WSL) | `/mnt/c/GOG Games/Morrowind/Data Files` |
| Fallout: New Vegas Data Files | not yet recorded — add the real path here once known; the cooker takes it as its first argument regardless |
| OpenMW source (Windows) | `C:\Users\rfdic\OneDrive\Documents\GitHub\openmw` |
| Build dir (Windows) | `cmake-build-release` |
| Build dir (Linux) | `cmake-build-linux` |
