# Engine Roadmap: Wishlist Reality Check

This doc exists because a large feature wishlist for a "Modern Vulkan Strategy Game Engine" landed against this repo — deferred/clustered rendering, GPU-driven culling, mass-battle unit rendering, lockstep multiplayer, a full in-engine editor suite, and more, organized across roughly twenty categories. Rather than restate that list as-is, this doc checks it against what actually exists in the codebase today, cites the real files, and turns the gap into a prioritized plan that fits where this project is actually going.

## What this repo actually is

Per `AGENTS.md`, voxelsprout is a C++20/Vulkan engine for exploring and rendering Morrowind-style worlds (with a Fallout: New Vegas import path alongside it), built around four real pillars:

- **World exploration** — voxel terrain, chunk streaming, and Bethesda-asset import (`src/world/`, `src/import/`)
- **Hex strategy layer** — a turn-based 4X game on a hex grid (`src/game/`)
- **Factory simulation** — a conveyor-belt/pipe sim (`src/sim/`)
- **UI framework + mini-games** — a Vulkan-free retained UI toolkit and several small prototypes (`src/ui/`, `src/games/`)

`AGENTS.md` is also explicit about what this project is **not**: "a generic engine framework," "an ECS experiment," "an enterprise architecture exercise," or "a plugin-based platform." It prioritizes "practical implementation over generic engine architecture." That matters here because a large chunk of the wishlist assumes a different kind of engine — a real-time mass-battle RTS with netcode and a full scenario-editor suite — which isn't this project's direction.

**Ground rule for what follows:** a feature earns a spot on this roadmap because it serves one of the four pillars above, not because a generic strategy-engine spec listed it.

## Legend

| Symbol | Meaning |
|---|---|
| ✅ | Done — exists and is in use today |
| 🟡 | Partial — some of the mechanism exists, notable gaps remain |
| ⬜ | Not started |
| 🚫 | Out of scope — conflicts with `AGENTS.md`'s stated non-goals |

## Status inventory

### Core Rendering

| Feature | Status | Notes |
|---|---|---|
| Vulkan renderer | ✅ | `src/render/backend/vulkan/renderer_backend.h` — instance/device/swapchain/pipelines/descriptors/FrameArena |
| Reverse-depth buffer | ✅ | `VK_COMPARE_OP_GREATER_OR_EQUAL` throughout `pass_pipelines.cc` |
| Cascaded shadow maps | ✅ | `frame_pass_shadow.cc`, `shadow_culling.cc` |
| Screen-space ambient occlusion | ✅ | `frame_pass_ssao.cc`, `ssao.comp.slang` |
| HDR + tonemapping | ✅ | ACES filmic + auto-exposure histogram + bloom, `tone_map.frag.slang` |
| Post-processing pipeline | ✅ | Tonemap + bloom + auto-exposure + sun shafts chain (`frame_run.cc`) |
| PBR | 🟡 | Baked-GI + albedo + slope-tinted shading, not a metallic-roughness/BRDF model |
| Volumetric fog | 🟡 | Analytic height-fog + 2D fog-of-war mask, not a froxel/raymarched volume |
| Dynamic weather / day-night | 🟡 | Full sky/atmosphere + sun-position system exists; no rain/snow/cloud-density state |
| Deferred / clustered forward rendering | ⬜ | Single forward pass, flat light list, no clustering |
| Temporal anti-aliasing | ⬜ | No jitter/reprojection |
| MSAA | ⬜ | All pipelines pinned to `VK_SAMPLE_COUNT_1_BIT` |
| Screen-space reflections | ⬜ | RT reflections exist instead as a gated shader variant (`-DODAI_RT_REFLECTIONS=1`) |
| Shader hot reload | ⬜ | Shaders compile offline via `slangc` at build time only |
| Render graph with automatic barrier sync | ⬜ | `frame_graph.{h,cc}` orders passes and validates order; barriers are hand-written per pass by design (`AGENTS.md`: "do not assume implicit barriers") |

### GPU-Driven Rendering

| Feature | Status | Notes |
|---|---|---|
| Indirect / multi-draw indirect | ✅ | `frame_draws.cc`, `frame_chunks.cc` |
| Bindless textures / materials | ✅ | `descriptors.cc` bindless texture table |
| GPU mesh instancing | ✅ | Grass billboards, hex/pipe instances |
| GPU frustum / occlusion culling | 🟡 | Computed CPU-side, results consumed via indirect draw buffers — no compute culling pass |
| Automatic LOD selection | 🟡 | Voxel chunk LOD only (`world/chunk_mesher.h`); no general per-object LOD |
| Impostor rendering | 🟡 | Grass/foliage billboards only; no generic object impostor system |
| GPU-generated draw commands | ⬜ | Command buffers built on CPU, not by a compute pass |
| GPU animation sampling | ⬜ | No skinning/bone system anywhere |
| Async compute | ⬜ | Compute passes run in-order on the graphics queue |

### Strategic World Rendering

| Feature | Status | Notes |
|---|---|---|
| Hex grid rendering | ✅ | Pointy-top, odd-r offset, `game/strategy_map.h` + `strategy_map_mesh.cc` |
| Political borders | ✅ | Owner-tinted border edges, `strategy_map_mesh.cc` |
| Fog of war | ✅ | Per-tile `TileVisibility` + blurred fog-map texture |
| Roads / rivers | 🟡 | Tile flags exist and render; no trade-route rendering |
| Terrain blending / biomes | 🟡 | 11 biome types with per-terrain color; adjacent-tile blending not confirmed |
| Supply / logistics overlays | 🟡 | Supply simulation exists (`game/units.h`); no visual overlay |
| Strategic map labels | ⬜ | Settlement names stored but not rendered as labels |
| Line-of-sight visualization | ⬜ | Only binary fog-of-war state, no LOS cone/ray viz |
| Movement range overlays | ⬜ | Pathfinding exists; no reachable-tile highlight rendering |
| Influence / control zones | ⬜ | Only binary tile ownership, no gradient influence map |
| Heatmaps | ⬜ | Not implemented |

### Terrain System

| Feature | Status | Notes |
|---|---|---|
| Chunked terrain streaming | ✅ | `world/world.h`, `world/clipmap_index.h` |
| Heightmap / procedural terrain | ✅ | Domain warping + hydraulic erosion (`chunk_grid_worldgen.cc`) + tessellated heightmap pipeline |
| Terrain LOD | ✅ | Voxel chunk LOD + tessellation-based heightmap |
| Water rendering | ✅ | Fresnel, depth absorption, foam, wind ripples (`imported_water.frag.slang`) |
| Shoreline effects | ✅ | Shallows band, foam crest, damping (`imported_water.frag.slang`) |
| Terrain deformation | 🟡 | CSG carve/paint exists for authoring (`world/csg.h`); not confirmed as a runtime gameplay mechanic |
| Splat maps | ⬜ | Blending is procedural/slope-based, not texture-painted |
| Decals | ⬜ | Not implemented |
| Virtual texturing | ⬜ | Not implemented |
| Nav-aware terrain data | ⬜ | Pathfinding lives entirely in the separate hex-grid layer, not derived from voxel terrain |

### Units and Armies

| Feature | Status | Notes |
|---|---|---|
| Selection UI | 🟡 | `ui/widgets/selection_inspector_panel.h` is a UI-side inspector, not a 3D outline pass |
| Rendering thousands of units, skinning, flocking, formations, batching/instancing for units | ⬜ / 🚫 | Current units are flat per-hex 4X data (`game/units.h`); no skinned-mesh rendering pipeline exists at all. Building a mass real-time battle renderer is a genre pivot away from the turn-based 4X + exploration identity this project has — see Out of Scope below. |

### Simulation Architecture

| Feature | Status | Notes |
|---|---|---|
| Deterministic turn-based simulation | ✅ | `game/game_sim.h` (`stepTurn`, seeded RNG) |
| Economic / population / tech simulation | ✅ | `game/economy.h/.cc`, `game/great_people.h`, `game/religion.h` |
| Moddable data tables | ✅ | `content/content_database.h` + JSON under `mods/base`, `content/mod_loader.h` |
| Event log | 🟡 | `GameEvent`/`World::events` recorded and displayed; no general pub/sub bus |
| Multithreaded job system | 🟡 | `core/job_system.h` fixed worker pool, used only by terrain meshing |
| Fixed-timestep sim | 🟡 | Turn-based sim is deterministic by construction; the factory sim (`sim/simulation.h`) runs on variable `dt` |
| Save-game serialization | 🟡 | Static `StrategyMap` serializes (`game/strategy_map_io.h`); live `GameState` (units, cities, empires, tech) does not |
| Background/async AI processing | ⬜ | `stepTurn`/`stepAiUnits` run synchronously |
| Snapshots / rollback | ⬜ | Not implemented |
| Data-oriented ECS | 🚫 | Explicitly rejected by `AGENTS.md` ("not... an ECS experiment") |

### AI and Navigation

| Feature | Status | Notes |
|---|---|---|
| Hex-grid A* pathfinding | ✅ | `game/units.h::findHexPath`, terrain/road-cost aware |
| Strategic + tactical AI layers | ✅ | Personality-weighted empire decisions (`stepTurn`) + unit orders (`game/ai_units.h`) |
| Movement costs | ✅ | `supplyCostForStep`, per-unit `movement` |
| Utility-style AI scoring | 🟡 | `Personality` weights are simple utility scoring, not a general utility-AI framework |
| Path previews | 🟡 | Paths stored and followed; no dedicated preview overlay found |
| Hierarchical A* / navmesh / flow-field nav | ⬜ | Not implemented |
| Behavior trees / GOAP | ⬜ | Not implemented |
| Influence maps | ⬜ | Not implemented |
| Diplomacy AI | ⬜ | No treaty/alliance system found |
| Async AI jobs / AI debug visualization | ⬜ | Not implemented |

### UI Framework

This is the strongest area of the codebase relative to the wishlist — see `docs/UI_LIBRARY.md`.

| Feature | Status | Notes |
|---|---|---|
| Retained-mode widget tree | ✅ | `ui/widget.h`, `ui/ui_context.h` |
| 9-slice panels | ✅ | `UiNineSlice` |
| Rich text markup | ✅ | `ui/rich_text.h/.cc`, `ui/cached_rich_text.h` |
| Data-bound UI | ✅ | `ui/document/ui_document.h` + `ui_binding.h` |
| Animated transitions (tweening) | ✅ | `ui/animation.h` |
| Tooltips | ✅ | `ui/tooltip.h` |
| Minimap / strategic map modes | ✅ | `ui/widgets/minimap_panel.h` |
| Scalable vector icons | ✅ | `ui/vector/` SVG-to-mesh pipeline |
| Theming | ✅ | `ui/theme/ui_theme.h`, JSON theme files |
| Hot reload (UI documents) | ✅ | `ui/document/ui_hot_reload.h` |
| Dockable/resizable windows | 🟡 | Resizable windows exist (`ui/widgets/window.h`); no docking |
| Unicode / localization | 🟡 | UTF-8 + glyph ranges; no locale/string-table system |
| High-DPI | 🟡 | Widgets take a scale factor; no platform DPI query confirmed |
| Drag-and-drop | ⬜ | Not implemented |
| Gamepad / keyboard navigation | ⬜ | Mouse/keyboard input only |
| Accessibility settings | ⬜ | Not implemented |

### Editor and Development Tools

| Feature | Status | Notes |
|---|---|---|
| UI layout editor | ✅ | `src/tools/ui_editor/` — drag/resize/snap widget design tool |
| Live sim inspection | 🟡 | `ui/widgets/selection_inspector_panel.h` inspects selected entities; no broader debug console |
| Hot reload (assets) | 🟡 | UI documents only; no shader/gameplay hot reload |
| Render debug / GPU profiling | 🟡 | GPU timestamp queries exist (`init.cc`); no profiler UI |
| World/terrain editor, scenario editor, faction/tech-tree/quest editors | ⬜ | World content is authored via offline cookers instead (`tools/strategy_map_gen_main.cc`, `tools/newvegas_cooker_main.cc`) — see note below |
| Entity inspector, nav debugging, AI decision inspector, replay/timeline debugger | ⬜ | Not implemented |

> A full in-engine editor suite (world sculpting, scenario/faction/tech-tree/quest editors) is a large, generic-tooling investment that cuts against `AGENTS.md`'s "prefer practical implementation over generic engine architecture." The offline-cooker approach already in use is the intentional tradeoff here, not a gap to urgently close.

### Asset Pipeline

| Feature | Status | Notes |
|---|---|---|
| Texture compression | ✅ | PNG → BC3/DXT5 with mip chain, `tools/dds_bundler_main.cc` |
| Shader compilation pipeline | ✅ | CMake invokes `slangc`, including define-based variants |
| DDS support | ✅ | `import/dds.h/.cc` |
| Font/vector atlas generation | 🟡 | Font atlas baking (`ui::Font` via stb_truetype) and SVG vector caching (`tools/svg_bundler_main.cc`) exist; no raster sprite-atlas packer |
| glTF import | ⬜ | Import targets Bethesda formats (ESM/BSA/NIF) instead, by design |
| KTX2 support | ⬜ | Not implemented |
| Mesh optimization / LOD generation (offline) | ⬜ | Only runtime voxel-chunk LOD exists |
| Impostor generation (offline) | ⬜ | Only runtime grass billboards exist |
| Content-addressed cache | ⬜ | Bundlers use sidecar files checked by freshness, not content hash |

### Audio

| Feature | Status | Notes |
|---|---|---|
| Spatial/backend audio system | ✅ | `src/audio/` — backend abstraction with miniaudio + null implementations |
| Music layering, dynamic soundtrack, occlusion, prioritization | ⬜ | Not surveyed in depth; no evidence found beyond the backend itself |

### Networking and Multiplayer

| Feature | Status | Notes |
|---|---|---|
| Any networking/multiplayer | 🚫 | No sockets/multiplayer code anywhere. `sim/network_graph.h` is a factory-belt topology graph, unrelated. Lockstep/deterministic netcode is a large investment with no current project driver — see Out of Scope. |

### Save, Replay, and Debugging

| Feature | Status | Notes |
|---|---|---|
| Static map serialization | ✅ | `game/strategy_map_io.h` |
| Live game-state save/load | ⬜ | `GameState` (units, cities, empires, tech progress) has no save path |
| Replay recording/rewind | ⬜ | "Replay" hits in code are all about UI draw-list re-issue, not gameplay replay |
| Debug command console, crash recovery, structured logging beyond `VOX_LOG*` | ⬜ | Not implemented beyond existing log macros |

### Modding and Scripting

| Feature | Status | Notes |
|---|---|---|
| Embedded scripting | ✅ | `src/script/` Lua integration hooking `game/mod_host.h` (`onTurnStart`, `onCityYields`, `onTechResearched`, etc.), tested (`tests/lua_hook_tests.cc`) |
| Data-driven content | ✅ | `content/content_database.h` loads techs/buildings/units/leaders from JSON mods |
| Mod dependency management | 🟡 | `content/mod_loader.h` resolves mod dirs; full load-order/dependency resolution flagged as future work in-repo |
| Steam Workshop integration, sandboxed scripting hardening | ⬜ | Not implemented |

### Performance and Scalability

| Feature | Status | Notes |
|---|---|---|
| Reverse-depth, GPU timestamp measurement, explicit barriers | ✅ | Matches `AGENTS.md`'s Performance Rules directly |
| Chunk streaming, FrameArena transient memory | ✅ | `world/world.h`, `docs/FrameArena.md` |
| Configurable quality presets, headless sim mode | ⬜ | Not confirmed |

### Platform Support

| Feature | Status | Notes |
|---|---|---|
| Windows | ✅ | Primary target; documented build path in `CLAUDE.md` |
| Linux | 🟡 | Tools/tests build only — `-DODAI_BUILD_APP=OFF`, no Vulkan app build documented |
| macOS / MoltenVK | ⬜ | No `__APPLE__`/MoltenVK references anywhere |

## Out of scope

These wishlist items directly conflict with `AGENTS.md`'s stated non-goals and are **not planned** unless the project's stated focus changes:

- **Mass real-time battle rendering** (thousands of skinned/animated units, flocking, formation systems, battlefield VFX at RTS scale) — this is a genre pivot away from the turn-based hex 4X + exploration identity the project actually has, and would require building an entire skeletal-animation and crowd-simulation stack from nothing.
- **Data-oriented ECS** — explicitly rejected: "This is not... an ECS experiment."
- **Lockstep/deterministic multiplayer netcode** — conflicts with "prefer practical implementation over generic engine architecture" and "not an enterprise architecture exercise"; no current pillar needs it.
- **Full in-engine editor suite** (world sculpting, scenario editor, faction/diplomacy editor, tech-tree editor, quest editor) — conflicts with the same practical-over-generic principle; the existing offline-cooker content pipeline is the intentional alternative.
- **Plugin/Workshop-style mod distribution platform** — conflicts with "not a plugin-based platform"; the existing JSON-mod + Lua-hook system already covers data/behavior modding without building a distribution platform.

## Prioritized next steps

This is a starting recommendation, not a commitment — sequencing is the user's call.

**Tier 1 — strategy-layer polish (near-term, high value, small/local).** The hex 4X layer already renders borders and fog of war and simulates supply/pathfinding, but gives the player no visual feedback loop for any of it:
- Strategic map labels (settlement names already stored, just not drawn)
- Movement-range overlay (reachability already computed by `findHexPath`)
- Supply-line overlay (supply cost already simulated in `game/units.h`)
- Line-of-sight visualization

**Tier 2 — rendering polish within the existing pass structure.**
- TAA (jitter + reprojection added into the existing frame graph)
- Terrain splat-map blending (replacing/augmenting the current slope-based blend)
- Shader hot reload for the Slang pipeline
- Live save/load for `GameState` (not just the static map)

**Tier 3 — larger, still aligned with existing pillars.**
- Metallic-roughness PBR terms layered onto the existing baked-GI model
- GPU-side (compute) frustum/occlusion culling, replacing the current CPU-computed path
- Offline mesh-LOD generation for imported static meshes
