# ARCH-007: Localize Vulkan effect resource state and lifecycle

Status: Proposed
Priority: P2
Rank: 7

## Current structure

Separate pass files operate on one RendererBackend class containing resource handles, histories, initialization flags and lifecycle methods.

## Evidence

- [renderer_backend.h](../../../src/render/backend/vulkan/renderer_backend.h) measured 2,758 lines / 152,079 bytes and had 28 indexed direct includers at audit time.
- AO, temporal resources and other effects share that owner; [frame_run.cc](../../../src/render/backend/vulkan/frame_run.cc) and [init_resources.cc](../../../src/render/backend/vulkan/init_resources.cc) each exceeded 200 KB.

## Why it is problematic

File separation does not constrain state access; creation, resize invalidation and destruction require cross-file reasoning.

## Impact on human development

Effect changes require inspecting centralized state and several lifecycle sites.

## Impact on AI development and context usage

A pass-local task may require substantial renderer initialization and frame-execution context.

## Proposed architectural direction

Extract one effect's resource state and lifecycle together, with explicit inputs and invalidation rules. Preserve explicit pass order/barriers and avoid generalized render-graph machinery.

## Expected blast radius

One effect initially: backend members, resource creation/destruction, resize and command recording routines. Choose the effect based on an actual pending change.

## Risk of changing it

High around GPU lifetime/synchronization; bound the extraction and avoid simultaneous functional rendering changes.

## Validation and acceptance criteria

- [ ] Run Vulkan validation layers through initialization, resize/recreation, and repeated enable/disable cycles.
- [ ] Verify failure cleanup and resource lifetime behavior.
- [ ] Compare rendered output and appropriate capture evidence before/after extraction.
- [ ] Run affected CPU policy tests, but do not treat them as a substitute for required GPU verification.

## Priority rationale

P2; ranked 7 in the [architecture improvement backlog](README.md).
This is a proposed bounded change, not authorization for a broad rewrite.
