# Engine Roadmap: Wishlist Reality Check

This doc exists because a large feature wishlist for a "Modern Vulkan Strategy Game Engine" landed against this repo — deferred/clustered rendering, GPU-driven culling, mass-battle unit rendering, lockstep multiplayer, a full in-engine editor suite, and more, organized across roughly twenty categories. Rather than restate that list as-is, this doc checks it against what actually exists in the codebase today, cites the real files, and turns the gap into a prioritized plan that fits where this project is actually going.

## What this repo actually is

Per `CLAUDE.md`'s Reference Touchstones, this project's direction is defined by four reference touchstones — not generic engine categories, but specific games whose feel/systems are the actual target:

| Touchstone | What it means here | Where it lives |
|---|---|---|
| **Morrowind** | Open-world exploration, hand-placed regional identity, readable terrain/settlements | `src/world/`, `src/import/` |
| **Civilization VI** | Turn-based hex-grid 4X: yields, tech, borders, diplomacy, map modes | `src/game/` |
| **SimCity (2013)** | Agent-driven city sim: RCI zoning, traffic, land value, data overlays | `src/games/citybuilder/` |
| **Dragon Age: Origins** | Party-based real-time-with-pause combat, branching dialogue, companion approval | greenfield — see below |

`CLAUDE.md`'s Non-goals section is also explicit about what this project is **not**: an ECS experiment, an enterprise architecture exercise, or a dynamic-loading plugin/mod distribution platform. It prioritizes explicit, hand-written control flow over implicit machinery. Those are style constraints on *how* things get built, not a cap on *which genres* get pursued — chasing four different touchstones is fine as long as each is built the same explicit, flat-data way the rest of the codebase already is. (Note: `CLAUDE.md` also documents a statically linked `IEnginePlugin`/`PluginRegistry` composition seam under "Engine plugins" — that's a compile-time extension point, not the dynamic-loading distribution platform this non-goals list rules out; see the Modding and Scripting section below.)

That said, a large chunk of the original wishlist assumes a different kind of engine entirely — a real-time mass-battle RTS with netcode and a full scenario-editor suite — which lines up with none of the four touchstones above and stays out of scope (see below).

**Ground rule for what follows:** a feature earns a spot on this roadmap because it serves one of the four touchstones above, not because a generic strategy-engine spec listed it.

## Legend

| Symbol | Meaning |
|---|---|
| ✅ | Done — exists and is in use today |
| 🟡 | Partial — some of the mechanism exists, notable gaps remain |
| ⬜ | Not started |
| 🚫 | Out of scope — conflicts with this project's stated non-goals (`CLAUDE.md`; see also Out of scope below) |

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
| PBR | 🟡 | Cook-Torrance metallic-roughness specular (`pbr.slang`) layered onto the baked-GI diffuse chain, opt-in per vertex via packed material flags. Values are authored per face by the procgen generators; no metallic/roughness *textures* and no prefiltered radiance probe yet — see Asset Pipeline |
| Volumetric fog | 🟡 | Analytic height-fog + 2D fog-of-war mask, not a froxel/raymarched volume |
| Dynamic weather / day-night | 🟡 | Full sky/atmosphere + sun-position system exists; no rain/snow/cloud-density state |
| Deferred / clustered forward rendering | ⬜ | Single forward pass, flat light list, no clustering |
| Temporal anti-aliasing | ⬜ | No jitter/reprojection |
| MSAA | ⬜ | All pipelines pinned to `VK_SAMPLE_COUNT_1_BIT` |
| Screen-space reflections | ⬜ | RT reflections exist instead as a gated shader variant (`-DODAI_RT_REFLECTIONS=1`) |
| Shader hot reload | ⬜ | Shaders compile offline via `slangc` at build time only |
| Render graph with automatic barrier sync | ⬜ | `frame_graph.{h,cc}` orders passes and validates order; barriers are hand-written per pass by design (explicit-barriers convention — see Renderer internals in `CLAUDE.md`) |

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
| Supply / logistics overlays | ✅ | A line to the nearest friendly settlement when the selected unit has marched out of supply range, `App::drawStrategyMapLabels`; backed by `game::cheapestSupplyRoute` (multi-source Dijkstra over `supplyCostForStep`, `src/game/units.cc`) |
| Strategic map labels | ✅ | Civ6-style city banners (owner color, tier chip, name) + floating unit HP/supply labels, `App::drawStrategyMapLabels` in `src/app/app.cc`, fog-of-war gated and cached |
| Line-of-sight visualization | 🟡 | A ring around the selected unit's current vision-radius boundary, `App::drawStrategyMapLabels` + `game::sightRadiusForUnit`; still no *obstruction* — sight is an unobstructed `hexDistance <= radius` circle with no terrain/elevation blocking, so this shows the existing (crude) vision shape rather than a true line-of-sight raycast |
| Movement range overlays | ✅ | Translucent per-tile hex wash for the selected unit's reachable set this turn, `App::drawStrategyMapLabels`; backed by `game::reachableTiles` (BFS, `src/game/units.cc`) |
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

### City Simulation (SimCity 2013 touchstone)

This is further along than the original wishlist survey suggested — that survey only looked at `src/game/`/`src/sim/`; `src/games/citybuilder/` is a separate, substantial prototype (~6,600 lines across `citybuilder_app.{h,cc}` and `citybuilder_citizens.{h,cc}`) whose own header comment describes it as "a compact SimCity-2013-style city builder."

| Feature | Status | Notes |
|---|---|---|
| RCI zoning | ✅ | `Zone` enum (Residential/Commercial/Industrial), `citybuilder_app.h` |
| Traffic simulation | ✅ | Per-tile `trafficLoad` congestion EMA + destination-routed citizen trips (`citybuilder_app.h`, `citybuilder_citizens.h::rollTrip`) |
| Land value / desirability overlay | ✅ | Toggleable data overlay, `m_showLandValue` in `citybuilder_app.h` |
| Named-citizen roster with schedules | ✅ | Homes/workplaces/spouses/traits, commute-aware trip rolling (`citybuilder_citizens.h`) |
| Moddable "tabloid" story system | ✅ | Lua-driven story templates/weights (`games/citybuilder/script/city_script.h`) so narrative text is data, not hardcoded |
| Emergency services (traffic-aware dispatch) | 🟡 | Siren-speed unit exists (`citybuilder_app.h`); full coverage-radius service modeling not confirmed |
| Population/economy/pollution heatmaps | 🟡 | Land-value overlay exists; other data layers from the original wishlist (pollution, education, health coverage) not confirmed present |
| Regional play (multiple connected cities) | ⬜ | Not implemented — single city/map only |
| Modular building growth stages (SimCity 2013's signature) | 🟡 | Zoning + citizen growth exist; whether buildings visually grow through density tiers wasn't confirmed without a deeper read |

### Party RPG / Narrative (Dragon Age: Origins touchstone)

This is the newest touchstone and, honestly, close to a blank page. The one thing in the repo that looks adjacent — `src/games/swtor/` — turns out not to be: it's a static UI-chrome mockup of an MMO HUD (`swtor_app.cc`, 1,178 lines) with **hardcoded** placeholder state (`m_playerHp = 0.82f`, fake chat log, fake cooldowns) and no underlying game logic. It demonstrates HUD layout patterns (unit frames, action bars, buff rows, character/inventory window) that are reusable, but none of Dragon Age: Origins' defining systems exist anywhere in the codebase.

| Feature | Status | Notes |
|---|---|---|
| MMO-style HUD chrome (unit frames, action bars, chat, inventory grid) | 🟡 | Exists as static mockup only (`games/swtor/swtor_app.cc`) — layout is reusable, none of it is wired to real state |
| Skeletal animation / GPU skinning | 🟡 | CPU side done and tested: `src/anim/` (`Skeleton`/`AnimationClip`, `AnimationSampler` hierarchy + lerp/slerp keyframe evaluation, JSON loader), `odai_animation_tests` all passing. GPU side is now wired into the actual frame: `recordSkinningPass` runs every frame before the shadow/prepass/main passes (`frame_run.cc`), each of those three passes has its own skinned-actor draw block reusing the existing `m_importedStaticPipeline` family (`frame_pass_shadow.cc`/`frame_pass_prepass.cc`/`frame_pass_main.cc`), `createSkinningComputeResources`/`destroySkinningComputeResources` are called from device init/shutdown (`init.cc`) alongside SSAO's, and it has real GPU timestamp query slots. Two real bugs caught and fixed while wiring this in: the bone-matrix upload was missing the `transpose()` step the camera-MVP path always does before a `-matrix-layout-column-major` shader reads it, and `setSkinnedActorPose` was calling into the FrameArena before `beginFrame()` had run for that frame (fixed by splitting it into a cheap setter + a `uploadSkinnedActorPoseForFrame()` step called right after `beginFrame()`). The "Windows CI job is the next real compile signal" note above turned out to matter: the first real CI run caught a genuine compile bug — `frame_pass_skinning.cc` and `skinning_resources.cc` both pulled in `render/renderer_shared.h` without first including `<GLFW/glfw3.h>` and `sim/network_procedural.h` (every other `frame_pass_*.cc` does this before the shared-header include), leaving `neighborMask6`/`glfwGetFramebufferSize` undeclared. Fixed by matching the established include order; a fresh CI run is in flight to confirm. Still not fully verified end-to-end (frame output correctness, GPU timestamps actually populating) until that run is green and someone eyeballs a real frame. Correcting an earlier inaccuracy in this doc: NIF skin data is not "parsed but discarded" — nothing skin/bone/keyframe-related is parsed anywhere in `import/fnv/nif_scene.cc` today; that import work is still fully ahead (see Explicitly Deferred below). |
| Branching dialogue system | 🟡 | First slice landed: `src/dialogue/` (data format, `DialogueRuntime` state machine, JSON loader with non-fatal error reporting) + `ui::DialoguePanel` (`src/ui/widgets/dialogue_panel.h/.cc`), tested by `tests/dialogue_tests.cc` (`odai_dialogue_tests`). Not yet wired into any actual game/content beyond the test fixture tree, and not yet exercised through a real build (see verification note in the commit) |
| Companion approval / relationship system | 🟡 | The runtime supports per-companion approval reads/deltas as a dialogue effect (`DialogueContext::approval`/`adjustApproval`, `DialogueCondition::minApproval`); `MapDialogueContext` state (flags + approvals) now round-trips to/from JSON (`src/dialogue/dialogue_state_io.h/.cc`), closing the persistence gap — tested by `odai_dialogue_tests` (file and in-memory round-trip, malformed-input and missing-file error paths), compiled and run standalone in this sandbox (no vcpkg here to build the full test suite). No UI display or companion roster concept exists yet outside the dialogue module itself |
| Real-time-with-pause tactical party combat | ⬜ | Not implemented. Distinct from — and much smaller in scope than — the mass-battle rendering that's out of scope below (4–8 characters, not thousands) |
| Origin-story branching narrative structure | ⬜ | Not implemented; would likely build on the same dialogue-tree infrastructure above |

### Units and Armies

Two different scales matter here and shouldn't be conflated: Civ6-scale strategic units (dozens on a hex map, already covered under Simulation/AI below) versus Dragon-Age-scale party combat (4–8 fully rendered, skeletally animated characters on screen at once). The wishlist's "rendering thousands of units" assumes RTS mass-battle scale, which neither touchstone calls for.

| Feature | Status | Notes |
|---|---|---|
| Selection UI | 🟡 | `ui/widgets/selection_inspector_panel.h` is a UI-side inspector, not a 3D outline pass |
| Hex-scale unit data (Civ6 touchstone) | ✅ | Flat per-hex unit data, HP/movement/supply (`game/units.h`) — adequate at this scale, no rendering gap to close |
| Small-party skeletal animation + rendering (Dragon Age touchstone) | 🟡 | See Party RPG / Narrative below for the current split (CPU sampler done + tested, GPU pipeline written but unwired). `import/fnv/nif_scene.cc` still parses no skin/bone/keyframe data at all — real character content is a separate, deferred import task. |
| Mass real-time battle rendering (thousands of skinned units, flocking, formations) | 🚫 | Out of scope — see below. Distinct from the small-party rendering above; do not conflate the two when scoping work. |

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
| Data-oriented ECS | 🚫 | Explicitly rejected — see `CLAUDE.md`'s Non-goals ("not... an ECS experiment") |

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

> A full in-engine editor suite (world sculpting, scenario/faction/tech-tree/quest editors) is a large tooling investment with no current pillar driving it. The offline-cooker approach already in use is the intentional tradeoff here, not a gap to urgently close.

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
| Engine plugin composition seam | 🟡 | `src/engine/plugin.h` (`IEnginePlugin`/`PluginRegistry`) lets a `GameApp` compose in optional lifecycle behavior (attach/tick/render/detach) without a dynamic loader — statically linked, same shape as `IModHost`/`mod_loader.h`. Wired into `GameApp::init/run/shutdown`, documented in `docs/GAME_API.md` §10 and `CLAUDE.md`. **Has zero implementations anywhere in the codebase yet and has never been built on a real Vulkan+GLFW target** — treat as an unproven interface, not a settled pattern, until a real plugin exists in a real game |
| Steam Workshop integration, sandboxed scripting hardening | ⬜ | Not implemented — and Workshop-style dynamic-loading distribution is explicitly out of scope regardless (see `CLAUDE.md`'s Non-goals) |

### Performance and Scalability

| Feature | Status | Notes |
|---|---|---|
| Reverse-depth, GPU timestamp measurement, explicit barriers | ✅ | Matches this project's performance-first conventions directly (see the `performance-engineer` agent's performance contract) |
| Chunk streaming, FrameArena transient memory | ✅ | `world/world.h`, `docs/FrameArena.md` |
| Configurable quality presets, headless sim mode | ⬜ | Not confirmed |

### Platform Support

| Feature | Status | Notes |
|---|---|---|
| Windows | ✅ | Primary target; documented build path in `CLAUDE.md` |
| Linux | 🟡 | Tools/tests build only — `-DODAI_BUILD_APP=OFF`, no Vulkan app build documented |
| macOS / MoltenVK | ⬜ | No `__APPLE__`/MoltenVK references anywhere |

## Out of scope

These wishlist items directly conflict with `CLAUDE.md`'s stated Non-goals and are **not planned** unless the project's stated focus changes:

- **Mass real-time battle rendering** (thousands of skinned/animated units, flocking, formation systems, battlefield VFX at RTS scale) — none of the four reference touchstones call for this. Do not confuse this with small-party (4–8 character) skeletal animation for the Dragon Age touchstone, which *is* in scope (see Party RPG / Narrative above) — the "no crowd-simulation stack" line is drawn at scale, not at "any animated characters at all."
- **Data-oriented ECS** — explicitly rejected: "not... an ECS experiment."
- **Lockstep/deterministic multiplayer netcode** — conflicts with "not an enterprise architecture exercise"; no current pillar needs it.
- **Full in-engine editor suite** (world sculpting, scenario editor, faction/diplomacy editor, tech-tree editor, quest editor) — a large tooling investment no current pillar is asking for; the existing offline-cooker content pipeline is the intentional alternative.
- **Dynamic-loading plugin/Workshop-style mod distribution platform** (`dlopen`/DLL loading, Steam Workshop-style discovery) — conflicts with `CLAUDE.md`'s "no dynamic-loading plugin/mod distribution platform" non-goal; the existing JSON-mod + Lua-hook system, plus the statically linked `IEnginePlugin`/`PluginRegistry` composition seam (see `CLAUDE.md`'s "Engine plugins"), already cover data/behavior/lifecycle composition without building a distribution platform.

## Prioritized next steps

This is a starting recommendation, not a commitment — sequencing is the user's call. Ordered roughly by (existing maturity + cost) across the four touchstones, cheapest/most-built first:

**Tier 1 — Civ6 strategy-layer polish (near-term, high value, small/local).** The hex 4X layer already renders borders and fog of war and simulates supply/pathfinding:
- Strategic map labels — **done.** `App::drawStrategyMapLabels` (`src/app/app.cc`) draws Civ6-style city banners and floating unit HP/supply labels, fog-gated and cached.
- Movement-range overlay — **done.** `game::reachableTiles` (`src/game/units.cc`) is a plain BFS bounded by `Unit::movementLeft` — not the same cost model as `findHexPath`'s `pathStepCost` (route-choice bias) or `supplyCostForStep` (provisions), since movement depletes by exactly 1 per hop regardless of terrain; `findHexPath`'s A* frontier is goal-directed and incomplete, so it couldn't be reused directly as the roadmap previously assumed. Rendered as a translucent per-tile hex wash in `drawStrategyMapLabels`. Fixed a related bug found along the way: `App::selectUnitAtHex` had no ownership check, so a player could select and issue move/attack orders through an enemy unit.
- Supply-line overlay — **done.** `game::cheapestSupplyRoute` (`src/game/units.cc`) is multi-source Dijkstra seeded from every one of the owner's settlements at once (no single admissible heuristic exists across multiple goals, so this couldn't reuse `findHexPath`'s A* shape). `supplyCostForStep` is direction-dependent — only the step's *destination* terrain incurs the hills/mountains surcharge — so edges are relaxed with reversed arguments to price the unit's actual walking direction rather than the search's outward-from-settlement expansion direction; this asymmetry is easy to get backwards, worth double-checking if this function is ever touched. Empty (nothing drawn) when the unit is already in range. Rendered as a single line via `UiDrawList::addPolylineAA`, distinct in color from the movement-range wash and vision ring below.
- Line-of-sight visualization — **done, scoped as a ring, not obstruction.** `game::sightRadiusForUnit` (`src/game/units.cc`) extracts what was previously hardcoded inline in `App::recomputeFogOfWarVisibility` (radius 2, scouts 3 — the scout bonus is a hardcoded typeId string match, not a `UnitStats` field, since no unit varies sight by content data today) so the fog computation and this overlay share one source of truth. The overlay draws an outline (not a fill, to avoid competing with the movement-range wash) around the selected unit's vision boundary. **Not implemented, and explicitly out of scope for this pass:** any actual obstruction — hills/forest blocking sight, elevation granting bonus range. Sight today, and in this overlay, is a flat unobstructed `hexDistance <= radius` circle; a real LOS raycast would be a simulation change (some currently-visible tiles would become hidden), not just a rendering addition, and needs its own design pass.

**Tier 2 — SimCity city-sim depth.** `citybuilder` is already the most mature of the four touchstone prototypes; the remaining gaps are additive data layers, not new architecture:
- Pollution/education/health-coverage heatmaps alongside the existing land-value overlay
- Service coverage-radius modeling (fire/police/hospital) building on the existing siren-dispatch unit
- Confirm/finish modular building-growth visual tiers if not already complete
- Regional play (multiple connected maps) — larger, do later within this tier

**Tier 3 — Morrowind rendering fidelity (polish within the existing pass structure).**
- TAA (jitter + reprojection added into the existing frame graph)
- Terrain splat-map blending (replacing/augmenting the current slope-based blend)
- Shader hot reload for the Slang pipeline
- Metallic-roughness PBR terms layered onto the existing baked-GI model — **first slice done.**
  `src/render/shaders/pbr.slang` (GGX + Smith + Schlick, analytic env BRDF) feeds
  `imported_static.frag.slang`; materials ride in `ImportedScenePackedVertex::flags`
  (layout in `src/import/imported_scene.h`) and are set per face by the CSG generators.
  Remaining, in rough order: metallic/roughness *texture* maps (needs a second bindless
  texture index per surface, so a packed-vertex format change), a prefiltered radiance
  probe to replace the irradiance-as-reflection stand-in in `pbrAmbientSpecular`, and
  extending materials to the mesh/part import path (`ImportedSceneMeshPart` carries no
  material, so `gpu_scene.cc` drops to the default for cooked Bethesda geometry)
- GPU-side (compute) frustum/occlusion culling, replacing the current CPU-computed path

**Tier 4 — Dragon Age foundation (biggest lift, infrastructure before features).** Sequenced bottom-up; two pieces are now in progress:
1. GPU skeletal animation (bone buffers + skinning shader) — **in progress.** `src/anim/` (skeleton/clip data model, hierarchy + keyframe sampler, JSON loader) is done and tested. The GPU side (`ImportedSkinnedMeshVertex`, `skinning.comp.slang`, `skinning_resources.cc`, `frame_pass_skinning.cc`) is written following the existing SSAO/auto-exposure compute-pass conventions but is **not yet called from the frame** — see the integration checklist at the top of `skinning_resources.cc` for the exact remaining steps. This was written without a Vulkan-capable build available (verify on a real build before relying on it); real NIF skin/keyframe import to drive it with actual Fallout NV character data is still a separate, deferred task (see below).
2. Dialogue-tree data format + UI — **done** (first slice). `src/dialogue/` + `ui::DialoguePanel`, tested by `odai_dialogue_tests`. Not yet wired into any actual game/content beyond the test fixture tree.
3. Companion approval/relationship state and small-party (4–8 character) real-time-with-pause combat loop — **persistence done.** `dialogue_state_io.h/.cc` saves/loads `MapDialogueContext`'s flags and per-companion approval as JSON (file or in-memory string), so a game can carry approval across sessions once it owns a save slot to put it in. Still missing: a companion roster concept (names/portraits/who's actually in the party), any UI display of approval, and the combat loop itself.
4. Live save/load for `GameState` generally (units, cities, empires, tech, and eventually companion/relationship state) — currently only the static map serializes
