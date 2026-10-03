# Focused navigation

Run from the repository root with Python 3.9+ and CMake/CTest. Maps are generated
from CMake File API codemodel and CTest JSON, not a second hand-maintained build.

```bash
mkdir -p build-linux/.cmake/api/v1/query
touch build-linux/.cmake/api/v1/query/codemodel-v2
cmake --preset linux-vcpkg
python3 tools/ai/nav.py generate
python3 tools/ai/nav.py owner src/import/cell_residency_planner.cc
python3 tools/ai/nav.py neighbors odai_bethesda_import
python3 tools/ai/nav.py includers src/import/cell_residency_planner.h
python3 tools/ai/nav.py reaches odai_bethesda_runtime
python3 tools/ai/nav.py find CellResidencyPlanner --scope odai_bethesda_import
python3 tools/ai/nav.py tests src/import/cell_residency_planner.h
python3 tools/ai/nav.py run-tests src/import/cell_residency_planner.h
python3 tools/ai/nav.py check odai_bethesda_import
```

Use `--build build-linux-headless` **before** the command for another build, request
its File API, configure its preset and regenerate. Committed maps describe the full
Debug Linux configuration. Regenerate after changing build metadata or includes.
Maps reject a different build or changed CMakeLists.txt. Regeneration must follow
configuration: source metadata alone cannot reveal disabled options. The CMake
codemodel dependency edges can include transitive/build-order dependencies;
`link_fragments` supplies linker evidence, and CMakeLists.txt remains authoritative
for PRIVATE/PUBLIC direct links. External packages may appear only as link fragments.

`find` reports literal declarations/definitions/references in target sources and
local header closure. It is not a semantic symbol index. Narrow by exact file,
then use `rg -n -F 'symbol' src/<neighbor> tests/<named-test>.cc` for callers or
`rg -n 'override|public Interface' <candidate-files>` for implementations. Use a
configured compile_commands.json with clangd for overloaded/virtual callers;
request `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON` at configuration. No standalone lint
preset is configured: `.clang-tidy` is policy, not evidence of a passing lint job.

`includers <file>` finds direct quoted includes in the configured source/header
closure. `reaches <target-or-file>` lists executable consumers through configured
dependencies and local includes; this is build reachability, not proof that a
runtime branch executes. Pair it with scoped `find` and a named harness scenario.
All three maps must have matching provenance. Run `python3 tools/ai/validate.py`
after regeneration to check graph, ownership, live CTest registration and queries.

Run `python3 tools/ai/validate.py` to check map integrity, configured test
registration and query behavior.

`tests <file>` uses test translation-unit include closure; `tests <target>` adds
configured dependent test targets. Neither measures runtime coverage or catches
Python dynamic imports. `run-tests` compiles discovered test executables first,
then uses an anchored exact-name regex. For a header with no owner, inspect direct
includers; never interpret missing ownership as dead code. Select `run-tests <exact-test-target>` after inspecting the associations when
a smaller suite is sufficient. Use full CTest for
cross-cutting changes. Do not run a broad target test set when exact file evidence
exists. Search `repo-map.json` through these queries instead of reading it whole.

For portable runtime evidence: `build-linux/odai_headless
 tests/fixtures/harness/one_actor.json` emits JSON snapshots/results; use the
headless replay tests for stable comparison. `odai_headless_ui` consumes UI intent
fixtures, not widget pixels. For real content, locate dispatch in the probe main
with a scoped symbol/flag search before running the relevant command. GI capture
analysis lives in `scripts/gi_probe.py`; scene/capture scripts remain optional and
local. Filter saved logs by category with `rg`, or select fields from JSON reports
with Python; retain exit status and the full local artifact path as evidence.
