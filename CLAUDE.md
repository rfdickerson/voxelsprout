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

CMake options: `ODAI_BUILD_APP` (ON), `ODAI_BUILD_TOOLS` (ON), `ODAI_BUILD_EXAMPLES` (OFF), `ODAI_RENDER_BACKEND` (`VULKAN`; `DX12`/`METAL` are declared but unimplemented). Every app/game/demo target sits behind `if(ODAI_BUILD_APP)` *and* the `VULKAN` backend guard.

### Optimized builds — required before believing any perf number

**Never profile a Debug build.** Measured on this tree (64 procedural chunks, gcc 13, best of 3):

| | worldgen | chunk meshing |
|---|---|---|
| `Debug` (`-O0 -g`) | 16255 ms | 617 ms |
| `RelWithDebInfo` (`-O2 -g`) | 6551 ms | 85 ms |
| `Release` (`-O3`) | 2041 ms | 75 ms |

That's **8x** on worldgen and **8x** on meshing between Debug and Release — a Debug measurement is not a measurement. Note worldgen is a further 3.2x from `-O2` to `-O3` (the noise loops vectorize), so use `Release` when the number is the point and `RelWithDebInfo` when you also need readable stacks.

Presets (`CMakePresets.json`) — the three original presets are Debug and unchanged:

| Preset | Build type |
|---|---|
| `default`, `vcpkg`, `linux-vcpkg` | `Debug` |
| `vcpkg-relwithdebinfo`, `linux-vcpkg-relwithdebinfo` | `RelWithDebInfo` — **use this to profile** |
| `vcpkg-release`, `linux-vcpkg-release` | `Release` |

```bash
cmake --preset linux-vcpkg-relwithdebinfo && cmake --build --preset linux-vcpkg-relwithdebinfo
```

Optimization level comes from `CMAKE_BUILD_TYPE` — no `-O` flags are hardcoded. Two opt-in switches a plain build type does not turn on:

- **`ODAI_ENABLE_LTO`** (OFF) — interprocedural optimization for optimized configs only; costs link time. Falls back with a warning if the toolchain can't do it.
- **`ODAI_ENABLE_NATIVE_ARCH`** (OFF) — `-march=native` / `/arch:AVX2`. **Local profiling only.** Wider SIMD and FMA contraction change floating-point results, and this project pins content-affecting hashes and RNG with golden vectors because worldgen output must reproduce; the binary may also fault outright on an older CPU. `-ffast-math` is never enabled anywhere, deliberately.

If `CMAKE_BUILD_TYPE` is unset on a single-config generator it now defaults to `RelWithDebInfo` (an empty build type means no `-O` *and* no `-g`). Passing a build type explicitly always wins — the Debug presets above are unaffected.

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

All are plain executables with a hand-rolled `int main()` and inline assertions — **no test framework** — except `odai_stability_gtests`, the one GTest suite (skipped with a `message(STATUS)` if GTest isn't found for your triplet). No test executable links Vulkan; the only headless-Vulkan check is the offscreen-capture smoke test under `examples/ui_stress_test/`, built when `ODAI_BUILD_EXAMPLES=ON`. `ctest -N` lists the full set; the sources in `tests/` map onto the target names.

Three suites carry contracts the source alone won't tell you: `odai_core_types_tests` pins content-affecting hashes and RNG with **golden vectors** (worldgen output must reproduce), `odai_fnv_import_tests` runs on **synthetic fixtures only** — never real game data — and `odai_content_tests` covers `mods/base` load order, not just parsing.

CI (`.github/workflows/ci.yml`) runs Linux only (full build including Vulkan on lavapipe, `slangc` installed, examples ON). Windows is a supported local dev target (see Build Commands above) but is not built in CI.

### Content generation tools

The offline generators and headless sim harnesses (`ODAI_BUILD_TOOLS=ON`, all pure CPU) are
documented in the `.claude/skills/content-tools` skill — including `--sweep N`, the CPU
regression harness for the turn loop.

**Stale README warning:** `README.md` still documents a Morrowind `odai_balmora_cooker` plus ESM/LAND/LTEX extraction. **Neither the target nor any Morrowind import code exists in the tree** — `src/import/` holds only the shared `ImportedScene`/DDS/GPU-scene code plus `import/fnv/`. Trust the source and this file over that part of the README.

**Run the app / games:**
```powershell
$env:ODAI_STRATEGY_MAP = "strategy_map.smap"
cmake-build-release\odai.exe
cmake-build-release\odai_game_citybuilder.exe   # also _snake, _minesweeper, _stellaris, _swtor, _voxelcraft
```

**Fallout: New Vegas viewer** — streams the worldspace straight out of the game's own
`FalloutNV.esm` and `.bsa` archives. No cooking step and no cooked assets on disk:

```bash
cmake-build-linux/odai_game_newvegas          # finds an installed copy on its own
cmake-build-linux/odai_game_newvegas --stream "<.../Fallout New Vegas/Data>"   # or point at it
```

`--worldspace <EditorID>` (default `WastelandNV`) and `--plugin <Plugin.esm>` (default
`FalloutNV.esm`) select what to stream; `ODAI_FNV_LOAD_RADIUS` sets the cell load radius.
`--scene <path.bin>` still loads a single cooked scene instead, which is what
`odai_newvegas_cooker` produces — that path is for baking a fixed region or a distant-LOD
tier (`--lod`), not for normal play.

**Asset mods (texture packs).** `docs/FNV_MODS.md` has the install + launch commands; this
section is the mechanism behind them. `--mod <dir>` (repeatable, later wins; or
`ODAI_FNV_MODS`, `:`-separated) adds a root laid out like `Data` itself — `textures\...`, `meshes\...` —
searched ahead of the game's loose files and archives. Mod roots are indexed recursively
and looked up **case-insensitively**, which is load-bearing on Linux: packs ship
`Textures/` while NIFs ask for `textures/`, and the game's own loose-file path is a
case-sensitive `exists()` that would miss every one of them.

Two traps, both silent:

- **`ODAI_FNV_TEX_SIZE` gates whether a pack is visible at all.** The mip-drop ceiling
  defaults to **512 px** (`CellSceneBuilder::m_maxTextureSize`), so a 1K/2K pack is
  dropped straight back to vanilla resolution — full I/O cost, zero visual change. Set it
  to `1024` to actually see the pack. Memory goes as the square (BC1 1024² with mips is
  683 KB against 171 KB at 512), so this is the first knob to turn down on an iGPU.
- **Cached cells bake their textures in.** The mod set and the ceiling are folded into the
  cell-cache directory name, so installing a pack lands on a fresh cache rather than
  serving vanilla art forever. Nothing is deleted — the old cache directory just stops
  being consulted, and costs disk until removed.

Run a downloaded pack through `odai_fnv_texture_pack` first (see Content generation tools):
it mip-drops to the ceiling you intend to run and lowercases every path. NMC's SMALL pack
goes 2.0 GB → 1.5 GB at a 1024 ceiling in about a second, and the levels above the ceiling
are then read once here instead of on every cache miss.

Only the streaming path honours mods — a cooked `--scene` already has its textures baked
in, and `odai_newvegas_cooker` has no `--mod` flag yet.

**Plugin mods and weather.** `--plugin-add <Plugin.esp>` (repeatable; or `ODAI_FNV_PLUGINS`,
comma-separated) loads an additional plugin after `--plugin`. Plugins resolve from the
`--mod` directories first and the `--stream` directory second (`FalloutLoadOrder::addSearchRoot`),
so a mod's `.esp` lives beside the `.bsa` it ships with and nothing is copied into the game
install. Masters resolve automatically, so naming only the mod is enough — `--plugin-add NevadaSkies.esp` pulls in FalloutNV.esm and
the four DLC masters, in order.

`src/import/fnv/plugin_load_order.{h,cc}` is what makes more than one plugin loadable. A
formID's high byte is a mod index that is **local to the file it is stored in** — it indexes
that plugin's own TES4 master list, and one past the last master means "a record I define".
Inside `NevadaSkies.esp`, `0x05` is its own 375 new weather records; load the same mod next
to a different plugin set and that byte has to mean something else. Getting this wrong does
not fail loudly, it resolves references to the wrong records. `remapFormId()` rewrites local
→ global; every formID that crosses into engine data structures must go through it.

`src/import/fnv/weather_records.{h,cc}` reads WTHR/CLMT across the load order (473 weathers
with Nevada Skies installed). `--weather <EditorID>` forces one by name; otherwise the
worldspace's climate picks its highest-chance entry. WTHR's NAM0 is **10 color channels × 6
time-of-day slots × RGBA** — New Vegas added noon and midnight to Fallout 3's four, which is
why the record carries six `\x00IAD`–`\x05IAD` adapters and NAM0 is 240 bytes, not 160.

Colors reach the sky through `Renderer::setWeatherSky(WeatherSkyParams)`. They **blend over**
the procedural Rayleigh/Mie sky rather than replacing it: `weight` 0 is the default and
renders byte-identically to before, which is what keeps every other game unaffected. WTHR
authors three vertical stops (Horizon at the skyline, SkyLower just above it, SkyUpper at the
zenith); collapsing that into a two-end gradient loses the narrow near-ground band that gives
these skies their look.

**Cloud layers** render from the same records. A weather's four layers (DNAM/CNAM/ANAM/BNAM,
tinted per time of day by PNAM) are uploaded through the imported-texture slot table and
sampled in the fullscreen sky pass. Two things there are non-obvious and were both bugs
first: the cloud SHAPE lives in the texture's **alpha**, not its luminance (rgb is nearly
white, so a luminance key makes the layer opaque), and the layer colour is PNAM's tint —
multiplying it by the texture's rgb squares the darkness and renders a heavy overcast pure
black.

**These textures are SKY DOME MAPS, not tiling planes.** Each is a fisheye of the whole sky:
zenith at the centre of the image, horizon on the rim of an inscribed circle. Nevada Skies'
`WesternSky1.dds` is unmistakable about it — a cloud disc with sun rays radiating into the
dark corners. So the mapping is angle-from-zenith → radius, compass bearing → angle, landing
every sample inside one copy of the texture. Two earlier attempts got this wrong: a true
plane projection `dir.xz / dir.y` diverges at the horizon into radial streaks, and the
softened `dir.xz / (dir.y + k)` that replaced it maps the sky into roughly [-0.76, 0.76] of
UV space, crosses zero, and lets the sampler WRAP — tiling the entire fisheye and putting a
visible seam down the sky with the image repeating from it. Tiling a fisheye is meaningless;
no amount of scale tuning hides it. Scrolling is likewise a ROTATION about the zenith, not a
UV translation, which would slide the fisheye off its own centre.

**`ODAI_FNV_SKY_GAIN` / `ODAI_FNV_SKY_CONTRAST` exist because WTHR colours are
display-referred.** They were authored as final sRGB for a renderer that showed them
directly; this one is HDR with an ACES curve and auto-exposure keyed to a sunlit desert.
Decoding to linear and stopping there renders an overcast sky (sRGB 23,27,30) as pure
black while the terrain looks correct. A flat gain cannot fix it — the value that makes
overcast readable washes a clear zenith to pale haze — so the decode applies
`pow(linear, contrast) * gain`, which lifts darks more than brights. This is a fudge; the
principled fix is inverting the tonemap on the GPU, where the exposure scale is known.
`ODAI_FNV_NOCLOUDS=1` separates "authored-dark gradient" from "total cloud cover", which
look identical on screen.

The decode shapes **magnitude and hue separately** — `pow(length(rgb), contrast)` on the
magnitude, `pow(rgb/length(rgb), saturation)` on the direction. Doing it per channel (which
it did first) pulls the channels toward each other, so lifting the darks also desaturated a
clear zenith from deep blue to pale haze. The split is lifted from ENB's tonemap
(Enhanced Shaders' `enbeffect.fx`), which applies contrast to `color/normalize(color)` and
saturation to `normalize(color)` for exactly this reason. `ODAI_FNV_SKY_SATURATION` tunes
it; ENB itself runs 1.25 by day against these same records and 0.9 at night.

**`ODAI_FNV_TONEMAP=enb`** swaps the post pass's ACES fit for ENB's extended Reinhard
`x(1+x/L)/(x+C)`, with Enhanced Shaders' tuned Fallout values. Two details are load-bearing:
ENB *divides* by an adaptation term (~0.1) to put scene values in the 1-10 range its `C=8`
knee expects — multiplying by this engine's exposure scale instead renders pure black — and
`grayadaptation` is adapted scene *luminance*, recovered as `0.18 / exposureScale`, not the
reciprocal of the exposure multiplier. Both mistakes were made and measured on the way in.
Default is ACES, so every other game is unaffected.

**Render scale defaults to native.** It used to default to 0.6, drawing the 3D scene at 36%
of the pixels and upscaling into a native-resolution UI composite — visibly soft. Measured
on the LNL iGPU at a 1920x1080 swapchain, 0.6 -> 1.0 costs ~3.5 ms (8.8 -> 12.3 ms/frame,
~81 fps), and only two passes move: main (3.6 -> 5.7) and SSAO (0.46 -> 1.5). Shadow (~1.3)
and prepass (~2.3) are **vertex-bound and do not change with resolution at all**, which is
why dropping resolution bought far less than it appeared to. `ODAI_RENDER_SCALE` still dials
it back, and is worth reaching for on a 4K swapchain.

**Weather audio** comes from the installed game. Fallout ships music as ~199 loose `.mp3`
under `Data/Music` (playable directly), but every ambient loop is `.ogg` inside a BSA, and
**miniaudio has no Vorbis decoder** — only WAV/FLAC/MP3. `newvegas_ogg.cc` compiles
`stb_vorbis` in its own TU and converts to `.wav` in the cell cache on first use, which is
a smaller change than threading a custom decoding backend through the audio PIMPL. Rain
plays only when the record's classification bits say it is raining. `ODAI_FNV_MUSIC`
overrides the track.

Not wired up yet: WTHR's **Ambient/Sunlight** channels, so terrain lighting does not yet
respond to the weather, and there are no rain **particles** — the mood is sky, fog and
sound only.

Runtime env vars: `ODAI_STRATEGY_MAP`, `ODAI_IMPORTED_SCENE` (view any cooked `.bin` with no strategy-map support compiled in), `ODAI_LOG_LEVEL`, `ODAI_PRESENT_MODE`, `ODAI_PERF_OVERLAY` (start every `GameApp` game with the CPU timing overlay up; **F3** toggles it at runtime), `ODAI_FNV_MODS` / `ODAI_FNV_TEX_SIZE` (see Asset mods above), and `ODAI_CITY_DEMO` / `ODAI_CITY_SEED` / `ODAI_CITY_STORM` / `ODAI_CITY_STORY` for citybuilder.

`ODAI_CITY_SCREENSHOT=<path>` (plus `ODAI_CITY_SCREENSHOT_FRAMES`) renders N frames, writes a
PPM and quits — the same headless-verification hook `odai_game_newvegas --screenshot` has, and
the only way to check a citybuilder visual change from a script on Wayland. Note the capture
trips `VUID-vkCmdCopyImageToBuffer-srcImage-00186` (the swapchain image lacks
`TRANSFER_SRC_BIT`); the images come out correct on this driver, but that is luck, not
contract.

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
            (the project's one deterministic RNG), ring_buffer.h, frame_profiler.h
            (Stopwatch/ScopedTimerMs/TimingChannel — CPU timing primitives)
math/     — header-only vector/matrix/quaternion + noise + geometry.h (Aabb3f, Ray,
            ray-triangle/ray-AABB intersection)
world/    — terrain, chunk grids, voxels, meshing/scheduling, clipmap, grass scatter
            (ChunkMeshScheduler::stats() reports wasted-work counters: meshes built
             then discarded because the chunk was edited or evicted mid-flight)
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

`GameApp` owns a `PluginRegistry`. The intent is an extension point: a game registers `IEnginePlugin` implementations from its own `onInit()` to compose in optional behavior (a debug overlay, a stats collector, a `mods/`-driven hook) without `GameApp` or the base game needing to know about it. `docs/GAME_API.md` §11 has the lifecycle contract and the one ordering caveat (`onRender` must be invoked manually from the game's own `onRender()`, after `beginFrameDraw()` and before `submitFrame()`; `onTick` is fanned out automatically).

**`IEnginePlugin` still has zero implementations anywhere in this codebase** and has never been built against a real Vulkan+GLFW target. Do not treat it as a settled architectural decision or cite it as having superseded anything — that requires a real plugin, in a real game, on a real build, first. Treat the interface as a draft: verify the shape still fits before building on it, and expect `GameApp`'s `protected` members (`m_renderer`, `m_uiContext`, …) may need new accessors before a plugin holding only `GameApp&` can do anything useful.

No dynamic loading (`dlopen`/DLL) is involved anywhere — a "plugin" here is a statically linked C++ class registered at startup, the same shape as the `IModHost` and `mod_loader` seams above.

### Hex strategy map

`src/game/` is pure CPU (no Vulkan, no imgui). Pointy-top hex grid in odd-r offset coordinates. `strategy_map_mesh.cc` converts a `StrategyMap` into an `ImportedScene` using the packed vertex-color render path (`textureIndex=0xFFFFFFFF` → per-vertex color), so no new renderer code is required to view a map. `strategy_map_io.cc` is versioned binary `.smap` serialization.

### Shader system

Shaders use **Slang** (`.slang` → `.slang.spv` SPIR-V). Shared includes live in `src/render/shaders/`. One of them has a contract worth stating up front:

- `pbr.slang` — metallic-roughness BRDF (GGX/Smith/Schlick + analytic env BRDF). Specular only: it layers onto the existing baked-GI diffuse chain rather than replacing it. Materials are opt-in per vertex through the packed flag bits defined in `src/import/imported_scene.h` — geometry without the PBR bit shades exactly as it did before

Ray-traced shadow/reflection variants compile the same `.slang` source with `-DODAI_RT_SHADOWS=1` or `-DODAI_RT_REFLECTIONS=1`.

### Naming conventions

- Namespaces: `odai::app`, `odai::render`, `odai::world`, `odai::ui`, `odai::anim`, `odai::content`, etc.
- Types: PascalCase — `StrategyMap`, `UiDrawList`, `RendererBackend`
- Functions: camelCase — `buildStrategyMapScene`, `setUiDrawData`
- Private members: `m_camelCase` prefix
- Module-scoped constants: `k` prefix — `kMaxFramesInFlight`, `kUiNoTexture`
- Source files: `.cc` (not `.cpp`)

## Docs index

| Doc | What's in it |
|---|---|
| `docs/GAME_API.md` | **Read first when adding a game under `src/games/`** — `GameApp` contract, renderer/UI API surface, the CMake block to copy |
| `docs/UI_LIBRARY.md` | UI architecture, widget catalog, theming, container reflow contract, integration walkthrough |
| `docs/FNV_MODS.md` | **Read first to launch the FNV viewer with mods** — verified install + launch commands, env-var reference, and the silent traps |
| `docs/FrameArena.md` | Per-frame GPU memory model |
| `docs/ROADMAP.md` | Feature-by-feature status against the four touchstones, with explicit out-of-scope calls |
| `docs/EARLY_ACCESS_PLAN.md`, `docs/devlog.md` | Planning and history |
| `docs/bloom.md`, `voxel_gi.md`, `shadow_occluder.md`, `spatial_partitioning_plan.md`, `stylized_low_poly.md`, `minecraft_clone_modernization.md` | Per-feature rendering/design notes |

The `.claude/skills/new-game` skill scaffolds a new `src/games/<name>/` target; `.claude/skills/vulkan-docs` checks current Vulkan practice before touching `render/backend/vulkan/`.

## Review agents (`.claude/agents/`)

Nine subagents, each carrying a distinct lens grounded in this codebase rather than generic advice. They are review/advice specialists — pick the one whose lens matches the question rather than asking the same question of several. Most can edit; `jonas` is read-only by design. Each agent's own `description:` states its lens and when to reach for it.

## Local Paths

| Resource | Path |
|---|---|
| Morrowind Data Files (Windows) | `C:\GOG Games\Morrowind\Data Files` |
| Morrowind Data Files (WSL) | `/mnt/c/GOG Games/Morrowind/Data Files` |
| Fallout: New Vegas Data Files (Linux/Steam) | `~/.steam/steam/steamapps/common/Fallout New Vegas/Data` — `odai_game_newvegas` finds this (and the usual GOG/WSL/Windows locations) on its own; the cooker and probe take it as their first argument |
| OpenMW source (Windows) | `C:\Users\rfdic\OneDrive\Documents\GitHub\openmw` |
| Build dir (Windows) | `cmake-build-release` |
| Build dir (Linux) | `cmake-build-linux` |
