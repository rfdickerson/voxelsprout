# ARCH-001: Decouple portable validation from presentation builds

Status: Implemented
Priority: P1
Rank: 1

## Current structure

Most plain tests are registered after the headless-only return; several CPU policy/bookkeeping tests link the complete renderer.

## Evidence

- [CMakeLists.txt](../../../CMakeLists.txt) returns for ODAI_HEADLESS_ONLY before odai_plain_test and simulation test registration.
- [tests/imported_render_policy_tests.cc](../../../tests/imported_render_policy_tests.cc) exercises CPU policy/math helpers but links odai_renderer. Bindless and GPU-arena bookkeeping tests also inherit renderer dependencies.
- The audit observed nine tests in the existing headless build. This is a configured-build observation, not a permanent expected count.

## Why it is problematic

Portable simulation verification is unavailable in the configuration intended to avoid presentation dependencies. CPU checks acquire shader and graphics build prerequisites.

## Impact on human development

Developers need a larger toolchain and build surface to verify localized state or policy changes.

## Impact on AI development and context usage

Agents must distinguish behavioral requirements from accidental build dependencies and expand into unrelated renderer context.

## Proposed architectural direction

Register applicable existing simulation/import/core tests before the headless return. Remove renderer linkage from demonstrably standalone policy/bookkeeping tests; keep GPU tests explicit.

## Expected blast radius

CMake registration/linking and, if needed, small helper targets. Begin with simulation test registration; production behavior should remain unchanged.

## Risk of changing it

Low to moderate: removing transitive dependencies may expose missing direct dependencies. Preserve assertion-enabling flags and test semantics.

## Validation and acceptance criteria

- [x] Configure linux-vcpkg-headless without presentation packages and build/run applicable existing simulation suites.
- [x] Build selected standalone CPU policy/bookkeeping tests without Vulkan, GLFW or Slang; document any genuine remaining dependencies.
- [x] Reconfigure the full build and confirm existing test registrations and assertion settings remain intact.
- [x] Record before/after configured test inventories and prerequisites; do not lower coverage to obtain a passing headless build.

## Implementation and verification

Portable test registration now precedes the headless return. Header-only policy
and bookkeeping tests no longer link the renderer; frame-order checks compile
their existing CPU implementations and link core. Upscale policy/contract and
JSON-only traversal state are available headless. Presentation/GPU tests retain
their explicit dependencies and assertion-enabling options are preserved.

Headless Debug and clean headless Release each passed all 56 registered tests;
the full Debug configuration retained and passed all 65 tests. The clean Release
dependency tree contains no presentation packages, and Vulkan/GLFW/UI package
discovery was explicitly disabled. See [validation evidence](ARCH-001-validation.md)
for inventories, remaining dependencies, commands and assertion checks.

## Priority rationale

P1; ranked 1 in the [architecture improvement backlog](README.md).
This is a proposed bounded change, not authorization for a broad rewrite.
