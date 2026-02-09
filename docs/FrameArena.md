 # FrameArena Design (Renderer Foundation)

## Purpose
FrameArena provides deterministic, per-frame transient allocation for the Vulkan renderer pass chain:

`SSAO -> Shadows -> Voxels -> Objects -> Post -> UI`

The goal is stable performance and simple lifetime management:
- O(1)-style reset behavior when a frame slot is reused.
- No per-draw map/unmap churn for CPU uploads.
- Predictable memory ownership for transient buffers (and later transient images).

## Two-Layer Model

### Layer A: Per-Frame Upload Arena (Host Visible)
One persistently mapped upload buffer is divided into fixed frame regions.

Used for:
- Per-pass/per-frame UBO updates (camera, cascade, SSAO params).
- Per-draw instance payloads.
- Small streaming CPU->GPU payloads.

Characteristics:
- Linear bump allocation.
- Alignment-aware suballocation.
- Reset at `beginFrame(frameIndex)`.

### Layer B: Transient GPU Arena (Device Local)
Frame-scoped transient resources for data that does not need to survive to another frame.

Used for:
- Scratch buffers.
- Intermediate buffers for pass-local work.
- (Next phase) transient images for AO/post/shadow intermediates with aliasing support.

Characteristics:
- Allocations are owned by frame slot.
- Automatically reclaimed when that frame slot is reused (after timeline wait).

## Current API (Implemented Foundation)
Implemented in `src/render/BufferHelpers.hpp` and `src/render/BufferHelpers.cpp`.

- `FrameArenaConfig`
  - `uploadBytesPerFrame`
  - `frameCount`
  - `uploadUsage`
- `FrameArena::init(...)`
- `FrameArena::beginFrame(frameIndex)`
- `FrameArena::allocateUpload(size, alignment, kind)`
- `FrameArena::createTransientBuffer(desc)`
- `FrameArena::createTransientImage(desc, lifetime)`
- `FrameArena::destroyTransientImage(handle)`
- `FrameArena::getTransientImage(handle)`
- `FrameArena::shutdown(...)`
- `FrameArena::activeStats()`

`TransientImageDesc` now includes pass tags (`firstPass`, `lastPass`) and alias eligibility.

The renderer now uses `FrameArena` for upload allocations.

## Lifetime Contract
1. Wait for frame slot completion (timeline semaphore value).
2. Call `FrameArena::beginFrame(currentFrame)`.
3. Allocate upload/transient resources during command recording.
4. Submit frame.
5. Reuse the frame slot only after wait in step (1).

This keeps reclamation deterministic and avoids freeing resources still in use by the GPU.

## Integration Status

### Done
- Replaced direct `FrameRingBuffer` usage in renderer with `FrameArena`.
- Camera UBO dynamic slice allocation uses `FrameArena`.
- Pipe/transport/preview instance uploads use `FrameArena`.
- Per-frame transient buffer ownership and reclamation hooks are implemented.
- Added transient image API in `FrameArena` (VMA-backed when available).
- Migrated AO normal-depth + SSAO raw/blur targets to `FrameArena` image allocation.
- AO intermediate targets are now per-frame-in-flight resources instead of swapchain-image-count resources.
- Added pass lifetime tags and simple image alias reuse for non-overlapping pass ranges.
- Migrated AO depth targets and HDR resolve post intermediate to `FrameArena` image allocation.
- Exposed FrameArena stats in ImGui frame panel (per-frame allocations + resident arena usage + alias reuse count).
- Upgraded image aliasing to memory-block aliasing (shared `VkDeviceMemory` across non-overlapping pass ranges).
- Added a live ImGui alias table (resource name, pass range, alias block id, ref count, handle).
- Added foundation tests for pass-range overlap and alias block refcount utilities.

### Next
- Expand alias planning beyond descriptor-match reuse to explicit virtual-resource scheduling.
- Add resize-aware rebuild path for transient image pools.

## Why This Helps
- Eliminates repetitive allocation/free patterns from frame code.
- Reduces fragmentation pressure by enforcing frame-local ownership.
- Keeps renderer code readable and deterministic for a small systems-driven codebase.
