# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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
- `odai_stability_gtests` — GTest suite covering frame graph, render math, sim network (requires GTest via vcpkg)

**Content generation tools:**
```powershell
cmake-build-release\odai_strategy_map_gen.exe          # generates strategy_map.smap + strategy_map_scene.bin
cmake-build-release\odai_balmora_cooker.exe "C:\GOG Games\Morrowind\Data Files" balmora.bin
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

Read **`AGENTS.md`** first — it defines the project's non-negotiable rules (Vulkan encapsulation, one-way data flow, Morrowind world feel, performance expectations).

### Module boundaries

```
app/      — lifecycle, input routing, per-frame coordination
core/     — math, time, logging (VOX_LOGE/W/I/D/T macros), input state
world/    — terrain, chunk grids, voxels, static placement
import/   — Morrowind asset parsing (ESM, terrain, scene serialization)
game/     — strategy map model, hex grid, serialization, mesh building
sim/      — factory simulation (pipes, belts, items)
ui/       — Vulkan-free UI framework: draw list, font, rich text, widget tree
render/   — public Renderer facade + everything Vulkan (only place that includes Vulkan headers)
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
| OpenMW source (Windows) | `C:\Users\rfdic\OneDrive\Documents\GitHub\openmw` |
| Build dir (Windows) | `cmake-build-release` |
| Build dir (Linux) | `cmake-build-linux` |
