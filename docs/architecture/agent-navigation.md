# Agent navigation architecture

This inventory describes the configured Linux Debug build at the time of generation.
Start with root/local AGENTS.md and `tools/ai/nav.py`; this report records evidence
and uncertainty rather than requiring ordinary agents to read it in full.

## Inventory and hierarchy

| Boundary | Responsibility / interface starting points | Target / consumers |
| --- | --- | --- |
| Foundation: `src/core`, `src/math` | Jobs/log/resource paths; header math | odai_core; shared header utilities |
| Focused: `src/anim` | Clips/skeletons/pose graphs/programs; animation_sampler.h, pose_graph.h | odai_anim → importer, renderer/application via importer |
| Focused: `src/dialogue` | Generic dialogue/runtime/state IO | odai_dialogue → importer |
| Focused: `src/audio` | audio.h service and audio_backend.h | odai_audio → application |
| Focused: `src/ui` | ui_context.h, draw lists, fonts, widgets | odai_ui → renderer |
| Import: `src/import` | DDS, ImportedScene serialization/query, residency; Bethesda formats/profiles/builders | odai_bethesda_import → runtime, renderer, tools |
| Simulation: `src/bethesda` | bethesda_session.h, runtime_world.h, script VMs, inventories, saves, physics | odai_bethesda_runtime → odai, headless, probe |
| Presentation: `src/render` | renderer.h facade; backend/vulkan implementation, shaders | odai_renderer → odai, renderer tests |
| Resolution: `src/render/upscale` | upscale_policy.h, upscale_contract.h | odai_upscale; execution backends compiled into renderer |
| Integration: `src/games/bethesda`, `src/engine` | bethesda_main.cc → BethesdaApp; game_app.cc event/frame loop | odai; traversal_state.cc is odai_traversal |
| CLI: `src/tools` | Main/dispatch files and coverage helpers | executables listed below |
| Tests | tests/*.cc, tests/*.py; synthetic fixtures/harness; render/tests GPU pose | 65 current CTest registrations; map includes commands and association evidence |
| Build/package | CMakeLists.txt, CMakePresets.json, vcpkg.json; cmake/Packaging.cmake, packaging/ | Ninja builds, resource tree and packages |
| Validation/capabilities | docs/PARITY.md, docs/capabilities, docs/validation | Read the named capability before implementation; no status changed by this work |
| Scripts/benchmarks | scripts/*.py, odai-launcher, iridius-bench; benchmarks/ | Capture, visual checks, performance/release audits; inspect only the selected workflow |
| Assets/content | assets/fonts, assets/textures; tests/fixtures | Bundled resources and synthetic test data; not full game content |
| Generated/vendor/reference | build*/share/odai/shaders/*.spv, File API, compile metadata, vcpkg-installed packages; references/openmw | Generated outputs excluded from source ownership. OpenMW is read-only, not engine linkage |
| Local runtime evidence | captures/ and optimized builds | Optional game/mod data and output; never commit game assets |

Non-test executables: `odai`, `odai_headless`, `odai_headless_ui`,
`odai_bethesda_probe`, `odai_bethesda_cooker`, `odai_texture_pack`.
Linux also builds `odai_bench_heap` as a shared interposer.
Each executable's compiled sources and reachability are queryable via maps.
Primary mains are in games/bethesda/bethesda_main.cc or tools/*_main.cc.
The runtime's mod-check dispatch occurs before normal application startup.

Libraries: odai_core, odai_anim, odai_dialogue, odai_audio, odai_ui,
odai_upscale, odai_bethesda_import, odai_bethesda_runtime, odai_renderer,
odai_traversal. Public headers are exposed via the entire src root, not installed
per-library interfaces. The local contracts identify useful entry headers without
claiming every header is a supported API.

Do not normally inspect builds, installed packages, references, captures or binary
assets. Legacy directories `src/world`, `src/games/newvegas`, `src/import/fnv`, and
`src/tools/oblivion_nif_lab` are not active directory-level subsystems in this build.
Do not assume similarly named classes or standalone files are built: use ownership.

## Validated graph

Arrows mean **direct declared internal link dependency**, from consumer to provider.
Evidence is target_link_libraries in CMakeLists.txt, cross-checked with configured
codemodel dependencies and link fragments. File API dependency lists can also
include transitive dependencies; generated JSON deliberately retains that evidence.

```text
odai → renderer, audio, traversal, bethesda_runtime
headless / headless_ui → bethesda_runtime
probe → bethesda_import, bethesda_runtime
cooker / texture_pack → bethesda_import
renderer → core, ui, upscale, bethesda_import (+ slang_shaders build order)
bethesda_runtime → core, bethesda_import
bethesda_import → core, anim, dialogue
ui / audio → core
core / anim / dialogue / upscale / traversal → no internal library link dependencies
```

External direct links: core → Threads; anim/dialogue/traversal → JSON;
UI → JSON, ZLIB (Stb include); importer → ZLIB, Jolt, JSON;
runtime → JSON; renderer → Vulkan, VMA, ImGui, GLFW (Stb include).
Audio optionally compiles miniaudio with Threads/dl on Linux; optional XeSS SDK
links in upscale/renderer. vcpkg manifest declares base physics/JSON/ZLIB and a
presentation feature. Slang is discovered separately and compiles SPIR-V into the
resource tree. Headless-only configuration returns before renderer/runtime-window
targets and most plain tests: full-build test counts must not be applied to it.

Validation performed: configured source ownership checks, all mapped source paths
exist, graph DFS detects no configured target cycles, CTest names agree with current
`ctest --show-only=json-v1`, selected test includes match queries, focused builds
and tests pass, synthetic headless execution emits status pass. Existing architecture
directory had no documents to supersede. No broad source body crawl was used:
metadata came first, then selected interfaces/tests; automation parsed only quoted
include lines in configured sources and their local header closure (440 files).

## Coupling and ambiguity

- No cycle in the configured target graph. This does not prove absence of header
  or runtime callback cycles, or cycles in disabled configurations.
- Importer and core have substantial fan-in; renderer has the widest library
  fan-out (core/UI/upscale/importer plus transitive animation/dialogue). This follows
  aggregation roles, but broad target-level test selection consumes unnecessary
  context. Prefer exact header/direct test evidence.
- Header encapsulation is conceptual: every target gets src as an include root.
  `src/import/bethesda/character_builder.h` includes renderer_types.h even though
  the importer does not link renderer. This is a real reverse header coupling.
  Do not claim importer has a clean presentation-free interface.
- `cell_streamer.cc` includes renderer.h but is compiled into odai, not importer.
  `condition.cc` is compiled into importer despite living under bethesda.
  Navigation/simulation production sources are compiled separately into selected
  tests. These ownership exceptions are recorded in local guidance and map checks.
- Resolution policy is a separate library, but its backend implementations are
  renderer sources. Directory boundaries alone would give incorrect ownership.
- Header/test associations are static textual evidence, not coverage. Application
  command tests may include gameplay headers while only exercising --help.
  Python import graphs, virtual dispatch and runtime callback reachability are
  not indexed. Use clangd and targeted runtime evidence when needed.
- No generalized dynamic engine plugin registration exists in the active target
  inventory. Game plugin/load-order data is content import, not engine extension.
- Shader include closure and GPU resource bindings require inspecting the selected
  shader/CMake custom command/pass together. C++ include maps do not establish
  shader correctness. GPU/real-data verification was not performed here.
- Capability descriptions and locally modified sources are active work in progress.
  These maps describe this checkout, not a clean release baseline. No capability
  acceptance or parity status was altered.

## Context-efficiency experiments

These are navigation/validation rehearsals, not implementation changes. Root/local
contracts add roughly 100–150 lines per task depending on routing; maps are queried,
not loaded into the agent context. Source windows below are actual inspected volume;
full-file totals are conservative upper bounds for a future agent making the fix.
Full-file context bounds are about 1,867 tokens for jobs (7,469 bytes),
847 tokens for TAA (3,386 bytes), and 8,710 tokens for residency (34,841 bytes),
using bytes / 4. These exclude routing guidance and query output.

| Representative task | Files inspected | Source inspected / full file bound | Searches | Executed verification |
| --- | --- | --- | --- | --- |
| Adjust JobSystem idle/drain behavior | core/job_system.h and .cc; tests/job_system_tests.cc | 200 lines / 254 lines | owner lookup; scoped find waitIdle; tests header | odai_job_system_tests passed; header-associated set 4/4 passed |
| Adjust TAA depth rejection | render/taa_depth_policy.h; tests/taa_depth_tests.cc | 65 / 65 lines | scoped find taaDepthMatches; tests header | odai_taa_depth_tests 1/1 passed |
| Adjust residency prediction/grid axis | import/cell_residency_planner.h and .cc; tests/cell_residency_planner_tests.cc | 220 / 871 lines | scoped find CellResidencyPlanner; tests header | odai_cell_residency_planner_tests passed; header-associated set 7/7 passed |

To reproduce the focused path:

```bash
python3 tools/ai/nav.py owner src/core/job_system.cc
python3 tools/ai/nav.py find waitIdle --scope src/core/job_system.cc
python3 tools/ai/nav.py tests src/core/job_system.h
python3 tools/ai/nav.py run-tests odai_job_system_tests
python3 tools/ai/nav.py find taaDepthMatches --scope src/render/taa_depth_policy.h
python3 tools/ai/nav.py run-tests src/render/taa_depth_policy.h
python3 tools/ai/nav.py find CellResidencyPlanner --scope src/import/cell_residency_planner.cc
python3 tools/ai/nav.py tests src/import/cell_residency_planner.h
python3 tools/ai/nav.py run-tests odai_cell_residency_planner_tests
python3 tools/ai/validate.py
build-linux/odai_headless tests/fixtures/harness/one_actor.json
```

Experiments improved the tool: direct-test-include associations are now distinguished
from indirect associations; exact test target lookup permits narrowing after source
selection. CTest uses supported capturing groups and `--no-tests=error` so an empty
selection cannot silently pass. Five navigation checks pass, including ownership
exceptions, graph/source integrity, actual test registration, focused queries and
unknown-test rejection. A subsequent audit refreshed all maps (67 targets, 65
tests, 440 include files) and reran the three exact test targets above. Seven
navigation checks now also verify direct includers, executable reachability and
matching map provenance. Synthetic runtime evidence was saved locally at
`/tmp/odai-navigation-headless.json`; JSON status is pass.

Remaining validation scope: this documentation/tooling change does not rerun full
engine builds, full CTest, optimized performance captures, other platform builds or
real game data. Reconfigure/regenerate after changing options, CMake or includes.

## Follow-up audit: bounded discovery and navigation gaps

The audit read build declarations, presets, package dependencies, the navigation
scripts and local contracts first. Selected source checks confirmed the reverse
renderer header dependency, application-owned streaming and main/mod-check entry
dispatch. It did not recursively read source bodies. Existing measurements above
are the initial experiments; the repeat inspection was smaller: job_system.cc
lines 35–68 (34 lines), taa_depth_policy.h (27 lines), residency planner header
lines 1–55 and its test lines 1–45 (100 lines), plus filtered symbol matches.
The repeat ran owner/find/tests queries and all three exact CTest targets, each
passing 1/1; the synthetic actor runner exited successfully.

Two new queries reduce follow-up discovery:

```bash
python3 tools/ai/nav.py includers src/core/job_system.h
python3 tools/ai/nav.py reaches odai_bethesda_runtime
```

The first reports direct quoted include references; the second follows configured
target dependencies and local include closure to executable consumers, including
tests. Neither claims semantic caller discovery or runtime coverage. Virtual and
overloaded calls still require clangd or a selected integration scenario. Map
queries now reject inconsistent provenance across repository/dependency/test maps.
Include changes still require explicit regeneration; provenance does not hash all
source files. Maps should be regenerated from a freshly configured build after
build changes; the tool cannot establish that an old File API reply reflects an
unconfigured edit. These limitations are deliberate and visible rather than
silently treating the index as authoritative.

## Proposed architecture improvements

The ranked [architecture improvement tickets](tickets/README.md) record the
subsequent audit, with evidence, bounded directions, risks and acceptance criteria.
They remain proposals and do not change the current-state architecture or parity.
