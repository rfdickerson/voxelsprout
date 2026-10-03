# ARCH-004: Resolve renderer configuration explicitly and report effective values

Status: Proposed
Priority: P1
Rank: 4

## Current structure

CLI handling, application initialization and renderer execution read environment variables independently; some renderer values are cached in function-local statics.

## Evidence

- A literal ODAI_* getenv scan of configured production translation units found 205 distinct keys across 28 files; bethesda_app.cc contained 169 getenv call sites at audit time.
- [frame_run.cc](../../../src/render/backend/vulkan/frame_run.cc) caches ODAI_RENDER_NEAR and ODAI_RENDER_FAR in function-local statics.
- ODAI_RENDER_SCALE/ODAI_RENDER_SIZE handling spans [bethesda_main.cc](../../../src/games/bethesda/bethesda_main.cc) and [init.cc](../../../src/render/backend/vulkan/init.cc).

## Why it is problematic

Effective behavior depends on precedence, read timing and prior process initialization; later renderer instances may retain earlier settings.

## Impact on human development

Reproducing a visual/performance result requires reconstructing settings across several layers.

## Impact on AI development and context usage

Agents must search multiple consumers before confidently changing an option or interpreting a capture.

## Proposed architectural direction

Resolve immutable renderer settings at initialization, preserving CLI/environment compatibility. Emit structured effective values and their origins; keep genuinely dynamic controls explicit.

## Expected blast radius

Renderer configuration first, then CLI/application adapters and evidence scripts. Do not convert all engine options simultaneously.

## Risk of changing it

Moderate: defaults, aliases and precedence are compatibility behavior, including ODAI_FNV_* names.

## Validation and acceptance criteria

- [ ] Add table-driven parsing/precedence and invalid-value checks for migrated options.
- [ ] Initialize two distinct renderer configurations in one process and verify no settings leak through static caches.
- [ ] Attach structured effective configuration to captures/benchmark evidence.
- [ ] Verify default native-DPI/render-scale behavior and existing CLI/environment precedence.

## Priority rationale

P1; ranked 4 in the [architecture improvement backlog](README.md).
This is a proposed bounded change, not authorization for a broad rewrite.
