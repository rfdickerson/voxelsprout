# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Reference Touchstones

Four games define what this project is actually building toward. Each maps to an existing area of the codebase — none are being cloned literally, but each names the specific feel/systems to chase:

- **Morrowind** — open-world exploration, hand-placed regional identity, readable terrain and settlements (`src/world/`, `src/import/`).
- **Civilization VI** — turn-based hex-grid 4X strategy: yields, tech tree, borders, map modes/lenses, diplomacy (`src/game/`).
- **SimCity (2013)** — agent-driven city simulation: RCI zoning, traffic congestion, land value, service coverage, data-layer overlays (`src/games/citybuilder/`, already explicitly modeled on it).
- **Dragon Age: Origins** — party-based real-time-with-pause tactical combat, branching dialogue with consequences, companion relationships/approval. The foundations exist (`src/dialogue/`, `src/anim/`, the GPU skinning pass) but no party/combat layer is built — see `docs/ROADMAP.md`'s Party RPG / Narrative section.

These are creative direction, not a mandate to merge four genres into one game — each pillar can keep developing largely on its own track. `docs/ROADMAP.md` tracks concrete status and priority per touchstone.

## Non-goals

This project is not:
- an ECS experiment
- an enterprise architecture exercise
- a dynamic-loading plugin/mod distribution platform (no `dlopen`/DLL loading, no Steam Workshop-style discovery)

It prioritizes explicit, hand-written control flow over implicit machinery (e.g. render passes never assume implicit barriers — see Renderer internals below). These are style constraints on *how* things get built, not a cap on *which* genres get pursued — see Reference Touchstones above. Note the "Engine plugins" pattern documented below is a statically linked, compile-time composition seam — a different thing than the distribution platform this list rules out.

## Build Commands

**Dependencies come from vcpkg** (`vcpkg.json` manifest). `nlohmann_json`, `ZLIB`, `Lua`, and `sol2` are `REQUIRED` at the top of `CMakeLists.txt`, so **even a tools/tests-only build needs the vcpkg toolchain** — there is no dependency-free configuration. `stb` and `nanosvg` are header-only and degrade gracefully (`ODAI_STB_INCLUDE_DIR` / `ODAI_NANOSVG_INCLUDE_DIR_OVERRIDE` override them for headless checks; without nanosvg the SVG importer compiles to a no-op).

**Windows (app + games + tools):**
```powershell
cmake -S . -B cmake-build-release `
  -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" `
  -DVCPKG_TARGET_TRIPLET=x64-windows
cmake --build cmake-build-release --target odai -j 4
```

**Linux / WSL2 — full build, matching CI** (Vulkan targets build and run against mesa/lavapipe):
```bash
cmake -S . -B cmake-build-linux -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTING=ON -DODAI_BUILD_EXAMPLES=ON \
  -DCMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" \
  -DVCPKG_TARGET_TRIPLET=x64-linux
cmake --build cmake-build-linux -j
```

**Linux / WSL2 — tools + tests only** (skips everything Vulkan/GLFW):
```bash
cmake -S . -B cmake-build-linux \
  -DODAI_BUILD_APP=OFF -DODAI_BUILD_TOOLS=ON -DBUILD_TESTING=ON \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"
cmake --build cmake-build-linux -j 4
```

CMake options: `ODAI_BUILD_APP` (ON), `ODAI_BUILD_TOOLS` (ON), `ODAI_BUILD_EXAMPLES` (OFF), `ODAI_RENDER_BACKEND` (`VULKAN`; `DX12`/`METAL` are declared but unimplemented). `CMakePresets.json` provides `default` and `vcpkg` Ninja presets. Every app/game/demo target sits behind `if(ODAI_BUILD_APP)` *and* the `VULKAN` backend guard.

**Build a single target:**
```powershell
cmake --build cmake-build-release --target odai_ui_tests -j 4
```

**Run tests:**
```powershell
ctest --test-dir cmake-build-release --output-on-failure   # all
ctest --test-dir cmake-build-release -R odai_ui_tests -V   # one suite
ctest --test-dir cmake-build-release -N                    # list registered tests
```

### Test targets

All are plain executables with a hand-rolled `int main()` and inline assertions — **no test framework** — except `odai_stability_gtests`, the one GTest suite (skipped with a `message(STATUS)` if GTest isn't found for your triplet). No test executable links Vulkan; the only headless-Vulkan check is the offscreen-capture smoke test under `examples/ui_stress_test/`, built when `ODAI_BUILD_EXAMPLES=ON`.

| Target | Covers |
|---|---|
| `odai_foundation_tests` | chunk grid, world gen, mesher |
| `odai_job_system_tests` | `core/job_system` |
| `odai_chunk_mesh_scheduler_tests` | async chunk-meshing scheduler |
| `odai_ui_tests` | draw list, font metrics, rich text, widgets, bindings (headless) |
| `odai_retro_widgets_tests` | retro theme widget rendering |
| `odai_svg_vector_tests` | SVG import, tessellation, `.odaivec` cache |
| `odai_strategy_map_tests` | hex grid model, `.smap` serialization, mesher |
| `odai_economy_tests`, `odai_advisor_tests`, `odai_great_people_tests` | `src/game/` 4X systems |
| `odai_procgen_tests` | building/civic generators, CSG, primitives |
| `odai_content_tests` | `mods/base` JSON content database + load order |
| `odai_dialogue_tests` | dialogue graph runtime + state serialization |
| `odai_animation_tests` | skeleton hierarchy, clip sampling |
| `odai_audio_tests` | audio facade against the null backend |
| `odai_lua_hook_tests`, `odai_city_script_tests` | Lua `IModHost` dispatch, citybuilder scripts |
| `odai_imported_scene_tests` | scene import/export round-trip |
| `odai_fnv_import_tests` | Fallout: New Vegas BSA/ESM/NIF readers (synthetic fixtures only — see README) |
| `odai_core_types_tests` | `core/hash.h`, `core/lcg.h`, `core/ring_buffer.h`, `math/geometry.h` + math scalar helpers (golden vectors pin content-affecting hashes and RNG) |
| `odai_stability_gtests` | GTest: frame graph, render math, shadow culling, sim network |

CI (`.github/workflows/ci.yml`) runs Linux (full build including Vulkan on lavapipe, `slangc` installed, examples ON) and Windows (build + ctest; `slangc` is absent there, so shader targets are skipped at configure time).

### Content generation tools (`ODAI_BUILD_TOOLS=ON`, all pure CPU)

```powershell
odai_strategy_map_gen [smap] [bin] [w] [h] [seed]   # writes strategy_map.smap + strategy_map_scene.bin
odai_civ_sim        [turns] [seed] [empires] [--quiet|--sweep N]   # headless 4X playtest metrics
odai_stellaris_sim  [turns] [seed] [empires] [--quiet|--sweep N]   # headless space-4X playtest metrics
odai_dds_bundler    <file.png>... | --dir <dir>     # offline PNG -> BC3 .dds sidecars
odai_svg_bundler    <file.svg>... | --dir <dir> [--sizes 16,32,64]  # SVG -> .odaivec cache
odai_theme_viewer                                   # terminal theme-token dump with hot reload
odai_newvegas_cooker <DataFiles> <Plugin.esm> <out.bin> --cell <EditorID>
odai_newvegas_cooker <DataFiles> <Plugin.esm> <out.bin> --worldspace <EditorID> <x0> <z0> <x1> <z1>
```

**Stale README warning:** `README.md` still documents a Morrowind `odai_balmora_cooker` plus ESM/LAND/LTEX extraction. **Neither the target nor any Morrowind import code exists in the tree** — `src/import/` holds only the shared `ImportedScene`/DDS/GPU-scene code plus `import/fnv/`. Trust the source and this file over that part of the README.

**Run the app / games:**
```powershell
$env:ODAI_STRATEGY_MAP = "strategy_map.smap"
cmake-build-release\odai.exe
cmake-build-release\odai_game_citybuilder.exe   # also _snake, _minesweeper, _stellaris, _swtor, _voxelcraft
```

Runtime env vars: `ODAI_STRATEGY_MAP`, `ODAI_IMPORTED_SCENE` (view any cooked `.bin` with no strategy-map support compiled in), `ODAI_LOG_LEVEL`, `ODAI_PRESENT_MODE`, and `ODAI_CITY_DEMO` / `ODAI_CITY_SEED` / `ODAI_CITY_STORM` / `ODAI_CITY_STORY` for citybuilder.

**Shaders** compile automatically when `slangc` is on PATH; if it isn't, shader targets are skipped rather than failing the configure. Outputs are `.slang.spv` next to the source. Manual compile:
```bash
slangc -target spirv -entry main -stage fragment -matrix-layout-column-major \
  -I src/render/shaders src/render/shaders/ui.frag.slang -o src/render/shaders/ui.frag.slang.spv
```
Use `add_slang_shader_variant(..., -DODAI_RT_SHADOWS=1)` for define-based shader variants.

## Architecture

### Module boundaries

```
core/     — time, logging (VOX_LOGE/W/I/D/T macros), input state, job system, grid utils,
            shared containers/primitives: hash.h (spatial + coordinate hashes), lcg.h
            (the project's one deterministic RNG), ring_buffer.h
math/     — header-only vector/matrix/quaternion + noise + geometry.h (Aabb3f, Ray,
            ray-triangle/ray-AABB intersection)
world/    — terrain, chunk grids, voxels, meshing/scheduling, clipmap, grass scatter
import/   — ImportedScene (de)serialization, DDS, GPU scene upload + import/fnv/ (Fallout: NV BSA/ESM/NIF)
game/     — Civ-style 4X model: hex strategy map, economy, advisors, religion, great people, units, AI
sim/      — header-only factory simulation (pipes, belts, tracks, network graph)
procgen/  — building/civic generators, CSG, primitives, props, city terrain
content/  — JSON content database + mod discovery/load order (mods/base)
script/   — Lua (sol2) scripting: sandboxed ScriptHost implementing game::IModHost
dialogue/ — branching dialogue graph: types, runtime, save-state serialization
anim/     — skeleton/bone hierarchy, animation clips, samplers (feeds the GPU skinning pass)
audio/    — PIMPL audio facade over miniaudio, with a silent null backend fallback
ui/       — Vulkan-free UI framework: draw list, font, rich text, widget tree, theming (see below)
render/   — public Renderer facade + everything Vulkan (only place that includes Vulkan headers)
engine/   — GameApp lifecycle base + PluginRegistry; what src/games/* build on (docs/GAME_API.md)
games/    — self-contained games on GameApp: citybuilder, minesweeper, snake, stellaris, swtor, voxelcraft
app/      — lifecycle, input routing, per-frame coordination for the older Civ-style `odai` app
tools/    — offline generators, headless sim harnesses, asset bundlers, UI editor + demos
tests/    — correctness tests; no Vulkan in test executables
```

**Hard rule:** only `src/render/` may include Vulkan headers. No Vulkan types cross this boundary — never let a `Vk*` type reach `src/games/`.

**Two app lineages, no shared code:** `odai::app::App` (`src/app/`) is the older, larger Civ-style application. `odai::engine::GameApp` (`src/engine/game_app.h`) is the thin GLFW + Renderer + UiContext loop that every `src/games/*` game subclasses. New games use `GameApp`; `src/app/app.cc` is not a template for anything.

### Data flow

World state never flows back from the renderer. The direction is:

```
World/Game data  →  Meshing / ImportedScene  →  Renderer::upload*()
                                                       ↓
                                              FrameArena (per-frame GPU upload)
                                                       ↓
                                 Passes (Shadow → Prepass → Skinning → Main → SSAO → Post → UI)
                                                       ↓
                                              Swapchain present
```

The seam between `src/ui/` and the renderer is `Renderer::setUiDrawData(const ui::UiDrawData&)` + `setUiFontAtlas(...)`. All UI logic and geometry assembly happens Vulkan-free in `src/ui/`; `UiRenderer` only streams the resulting vertex/index data to the GPU.

### Renderer internals

- **`render/renderer.h`** — narrow public facade. Call `upload*` to push world data, `setUiDrawData` for UI, `renderFrame` to record and submit.
- **`render/backend/vulkan/renderer_backend.h`** — the actual Vulkan state machine. Owns instance, device, swapchain, command pools, pipelines, descriptor sets, and `FrameArena`. All per-pass recording happens in files named `frame_*.cc`.
- **`render/frame_graph.{h,cc}`** + **`frame_graph_runtime.cc`** — declarative pass dependency graph. It resolves execution order and validates it; **barriers are hand-written per pass by design**, never inferred.
- **`descriptors.cc`** — bindless texture table, plus a classic-descriptor-set fallback path so headless CI renders on lavapipe.
- **`docs/FrameArena.md`** — per-frame transient GPU memory (host-visible upload arena + device-local scratch arena; reclaimed after the timeline fence).

Swapchain format is `B8G8R8A8_UNORM` (driver presents raw bytes, display interprets as sRGB). The UI fragment shader works in linear space and applies a manual `linearToSrgb` encode before output — matching the `pow(1/2.2)` the tonemapper applies for the 3-D pass. Vertex colors authored as sRGB hex are decoded to linear on entry so hex values are WYSIWYG. Color textures use `VK_FORMAT_R8G8B8A8_SRGB` image views so the sampler returns linear values.

### UI framework (`src/ui/`)

Fully headless and Vulkan-free: a retained widget tree over an immediate draw list. Packaged for vendoring into other engines (`odai_ui` + `odai_ui_vulkan` install/export targets; see `examples/vulkan_ui_integration/` for a from-scratch embedding and `docs/UI_LIBRARY.md` for the full guide). ImGui exists only as a separate dev/debug overlay with no code in common.

- `ui_draw_list` emits `UiDrawData` (quads, 9-slice, glyph-alpha quads, per-command texture + clip rect)
- `font` bakes an R8 atlas via `stb_truetype` + `stb_rect_pack`; `icon_atlas`, `resource_style`, `tooltip`, `ui_cursor`, and `animation.h` (easings, `Sequence`, `RectTween`) round out the primitives
- `rich_text` parses `<b>/<i>/<color=#rrggbb>/<br>` markup, wraps, and aligns; `cached_rich_text` memoizes layout
- `widgets/` — ~35 widgets from `Button`/`Label`/`Panel` up through domain panels (`research_panel`, `minimap_panel`, `advisors_panel`, `dialogue_panel`, charts). `signal.h` is the callback/slot registry; `UiContext` owns the root and dispatches input
- `theme/ui_theme` — JSON-driven color/font/size tokens with hot reload (`odai_theme_viewer`)
- `document/` — `.ui.json` documents instantiate live widget trees with `{binding}` expressions and hot reload (`assets/ui/docs/city_panel.ui.json`)
- `kits/` — prebuilt genre panel sets (`strategy_4x_kit`, `city_builder_kit`, `colony_sim_kit`)
- `vector/` — nanosvg-backed SVG import, tessellation, and an `.odaivec` geometry cache
- `render/backend/vulkan/ui_renderer.cc` is the only UI file touching Vulkan: alpha-blend pipeline, per-texture descriptor sets, per-frame geometry streaming

### Content, mods, and scripting

Three existing seams — prefer composing through one of them over inventing a fourth:

- **`content/`** — `ContentDatabase` built from `mods/base/data/*.json` (buildables, techs, units, leaders, religions, great people, balance, civpedia). `mod_loader` resolves the base game today; user-mod discovery and override order are planned, not built.
- **`game/mod_host.h`** — `IModHost`, the simulation's gameplay-event hook interface. `script::ScriptHost` (`src/script/`) is the Lua implementation: a sandboxed sol2 state dispatching to `Events.on` / `Effects.register` callbacks from `mods/*/scripts/*.lua`. `script_engine.h` is Lua-free (PIMPL) so callers don't pull in Lua; script errors are recorded, never thrown into the sim.
- **`engine/plugin.h`** — see below.

### Engine plugins (proposed pattern, still unproven — read before citing this as settled)

`GameApp` owns a `PluginRegistry`. The intent is an extension point: a game registers `IEnginePlugin` implementations from its own `onInit()` to compose in optional behavior (a debug overlay, a stats collector, a `mods/`-driven hook) without `GameApp` or the base game needing to know about it. `docs/GAME_API.md` §10 has the lifecycle contract and the one ordering caveat (`onRender` must be invoked manually from the game's own `onRender()`, after `beginFrameDraw()` and before `submitFrame()`; `onTick` is fanned out automatically).

**`IEnginePlugin` still has zero implementations anywhere in this codebase** and has never been built against a real Vulkan+GLFW target. Do not treat it as a settled architectural decision or cite it as having superseded anything — that requires a real plugin, in a real game, on a real build, first. Treat the interface as a draft: verify the shape still fits before building on it, and expect `GameApp`'s `protected` members (`m_renderer`, `m_uiContext`, …) may need new accessors before a plugin holding only `GameApp&` can do anything useful.

No dynamic loading (`dlopen`/DLL) is involved anywhere — a "plugin" here is a statically linked C++ class registered at startup, the same shape as the `IModHost` and `mod_loader` seams above.

### Hex strategy map

`src/game/` is pure CPU (no Vulkan, no imgui). Pointy-top hex grid in odd-r offset coordinates. `strategy_map_mesh.cc` converts a `StrategyMap` into an `ImportedScene` using the packed vertex-color render path (`textureIndex=0xFFFFFFFF` → per-vertex color), so no new renderer code is required to view a map. `strategy_map_io.cc` is versioned binary `.smap` serialization.

### Shader system

Shaders use **Slang** (`.slang` → `.slang.spv` SPIR-V). Shared includes live in `src/render/shaders/`:
- `camera_uniform.slang` — MVP, inverse matrices, FOV, near/far planes
- `chunk_push_constants.slang` — per-draw chunk offset and LOD
- `fullscreen_triangle.slang` — clip-space triangle for post passes
- `sh_lighting.slang` — spherical-harmonics GI evaluation
- `voxel_decode.slang` — voxel color unpacking
- `noise.slang` — shared noise functions
- `pbr.slang` — metallic-roughness BRDF (GGX/Smith/Schlick + analytic env BRDF). Specular only: it layers onto the existing baked-GI diffuse chain rather than replacing it. Materials are opt-in per vertex through the packed flag bits defined in `src/import/imported_scene.h` — geometry without the PBR bit shades exactly as it did before

Ray-traced shadow/reflection variants compile the same `.slang` source with `-DODAI_RT_SHADOWS=1` or `-DODAI_RT_REFLECTIONS=1`.

### Naming conventions

- Namespaces: `odai::app`, `odai::render`, `odai::world`, `odai::ui`, `odai::anim`, `odai::content`, etc.
- Types: PascalCase — `StrategyMap`, `UiDrawList`, `RendererBackend`
- Functions: camelCase — `buildStrategyMapScene`, `setUiDrawData`
- Private members: `m_camelCase` prefix
- Module-scoped constants: `k` prefix — `kMaxFramesInFlight`, `kUiNoTexture`
- Source files: `.cc` (not `.cpp`)
- Warnings: every target compiles with `-Wall -Wextra -Wpedantic` (GCC/Clang) or `${ODAI_WARN_FLAGS_MSVC}`. `.clang-tidy` enables `bugprone/cppcoreguidelines/modernize/performance/readability/portability`, with magic-number and trailing-return-type checks disabled.

## Docs index

| Doc | What's in it |
|---|---|
| `docs/GAME_API.md` | **Read first when adding a game under `src/games/`** — `GameApp` contract, renderer/UI API surface, the CMake block to copy |
| `docs/UI_LIBRARY.md` | UI architecture, widget catalog, theming, container reflow contract, integration walkthrough |
| `docs/FrameArena.md` | Per-frame GPU memory model |
| `docs/ROADMAP.md` | Feature-by-feature status against the four touchstones, with explicit out-of-scope calls |
| `docs/EARLY_ACCESS_PLAN.md`, `docs/devlog.md` | Planning and history |
| `docs/bloom.md`, `voxel_gi.md`, `shadow_occluder.md`, `spatial_partitioning_plan.md`, `stylized_low_poly.md`, `minecraft_clone_modernization.md` | Per-feature rendering/design notes |

The `.claude/skills/new-game` skill scaffolds a new `src/games/<name>/` target; `.claude/skills/vulkan-docs` checks current Vulkan practice before touching `render/backend/vulkan/`.

## Local Paths

| Resource | Path |
|---|---|
| Morrowind Data Files (Windows) | `C:\GOG Games\Morrowind\Data Files` |
| Morrowind Data Files (WSL) | `/mnt/c/GOG Games/Morrowind/Data Files` |
| Fallout: New Vegas Data Files | not yet recorded — add the real path here once known; the cooker takes it as its first argument regardless |
| OpenMW source (Windows) | `C:\Users\rfdic\OneDrive\Documents\GitHub\openmw` |
| Build dir (Windows) | `cmake-build-release` |
| Build dir (Linux) | `cmake-build-linux` |
