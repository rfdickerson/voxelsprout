# DevLog: Building a Voxel Factory Toy

This is a running engineering log of what worked, what broke, and what changed.
The goal is to share practical lessons while building a small Vulkan-based voxel game.

## 1) Ambient Occlusion: “Good Enough” Beats Fancy

### Problem
AO quality was unstable:
- SSAO showed blobs, speckles, and camera-dependent popping.
- Some dark edges looked like AO but were actually shadow/self-shadow behavior.

### What I changed
- Leaned on vertex AO as the baseline because it was stable and cheap.
- Added toggles in ImGui to isolate Vertex AO vs SSAO contribution.
- Revisited normals/face duplication assumptions for voxel surfaces.

### Lesson
If a screen-space effect is noisy and unstable, start from deterministic data (vertex AO) and add complexity only when the artifact budget is acceptable.

## 2) Pipe Geometry Iteration: Simple Primitives Win Early

### Problem
Pipe visuals had multiple issues while iterating:
- Endcaps everywhere.
- Segment seams and z-fighting flicker.
- Side connections not touching correctly.

### What I changed
- Moved through multiple representations (cuboids, caps, then cylinders).
- Fixed pipe join logic and seam/coplanar face cases.
- Improved split/connect behavior and visual flow direction.

### Lesson
For emergent sandbox systems, robust connectivity and readable behavior matter more than geometric detail at first.

## 3) FrameArena + VMA: Huge Stability Improvement

### Problem
Manual lifetime management caused shutdown validation errors:
- Undestroyed `VkImage` and `VkDeviceMemory`.
- Confusion around transient resources vs resident resources.

### What I changed
- Introduced `FrameArena` for per-frame uploads and transient resources.
- Migrated image/buffer allocations to VMA.
- Added leak logging and labels for images/resources.
- Added alias tracking and debug stats in ImGui.

### Lesson
Transient allocation is not just optimization. It is a correctness tool for lifetime control in multi-pass Vulkan renderers.

## 4) Indirect Rendering: Great Throughput, New Constraints

### Problem
Duplicated draw command streams across shadow/prepass/main were expensive and repetitive.

### What I changed
- Added indirect indexed draws for chunk rendering.
- Hit and fixed feature gate issue:
  - `drawCount > 1` requires `multiDrawIndirect`.
- Enabled and logged modern device features explicitly.

### Lesson
Indirect rendering helps, but feature negotiation must be explicit and validated per GPU.

## 5) Spatial Queries: Correctness Before “Speed Numbers”

### Problem
Spatial query stats looked constant (`49/49/49`) regardless of camera, while draw call counters were misleading.

### What I changed
- Instrumented query nodes/candidates/visible counts.
- Added toggles and frame metrics to compare query on/off.
- Improved frustum query robustness to reduce edge pop-in.

### Lesson
If profiling counters are wrong, optimization decisions are wrong. Instrumentation quality is part of renderer quality.

## 6) Greedy Meshing: Immediate, Measurable Gain

### Problem
Naive meshing generated too much geometry and draw overhead.

### What I changed
- Added greedy meshing mode.
- Added runtime toggle and timing comparisons in ImGui.
- Switched greedy meshing to default after validation.

### Lesson
Meshing strategy is one of the highest leverage optimizations in voxel renderers.

## 7) Clipmap Transition: Better Large-World Structure

### Problem
Octree-style workflows were less aligned with predictable, camera-centered streaming and updates.

### What I changed
- Planned and integrated clipmap-based spatial foundations.
- Removed unused octree source.
- Kept clipmap for culling/querying; removed unstable clipmap GI path when it caused popping/leaks.

### Lesson
Clipmaps are excellent derived caches for visibility/LOD. Keep gameplay truth in chunks and avoid coupling unstable lighting experiments into core rendering.

## 8) Slang Migration: Cleaner Shader Workflow

### Problem
Mixed shader workflows and path fallback complexity made iteration brittle.

### What I changed
- Migrated GLSL shaders to Slang and compiled to SPIR-V.
- Simplified shader lookup to a single expected path with hard fail.
- Added shared shader structures and camera uniform reuse.
- Added Slang build target in CMake.

### Lesson
A unified shader language and strict loading path reduces ambiguity and makes failures obvious.

## 9) Startup Profiling: Found a Real Bottleneck

### Problem
Startup had large stalls.

### What I measured
- Debug run showed world decode around 7 seconds.
- Release reduced decode to under 1 second.
- After targeted improvements, world load reached single-digit milliseconds in release.

### Lesson
Measure first, then optimize. Logs with step timings quickly separate I/O, decode, renderer init, and upload costs.

## 10) Vegetation Pipeline: Cheap, Readable World Detail

### Problem
Needed denser world “life” without heavy mesh cost.

### What I changed
- Added instanced crossed billboards for grass/flowers.
- Procedurally generated atlas sprites with alpha.
- Added random tint variation and per-instance rotation/placement.
- Fixed alpha edge issues and shadow pass integration.
- Added smaller red flower scale relative to green plants.

### Lesson
Instanced billboards are a high-value visual multiplier for voxel scenes when integrated with fog, shadowing, and stable alpha handling.

## 11) Presentation Timing Reality Check

### Problem
Expected `present_id` support, but active device path reported unsupported.

### What I changed
- Replaced `present_id/present_wait` path with `VK_GOOGLE_display_timing` support path.
- Added capability logs per GPU candidate and runtime status.

### Lesson
Presentation extensions are platform/driver-path dependent. Build robust fallbacks and log capability details clearly.

## 12) Current Direction

- Keep renderer deterministic and debuggable.
- Favor stable visual features over fragile effects.
- Keep systems readable and small.
- Continue profiling with concrete metrics before each major optimization.

## 13) Frame Pacing + FIFO Sync Cleanup

### Problem
- Uncapped rendering (MAILBOX path) drove very high FPS and coil whine.
- Switching to FIFO exposed timing jitter and occasional apparent "stalls" when editing voxels.
- Main loop could spin too fast when render submission was temporarily blocked.

### What I changed
- Added fixed-step simulation accumulator in `App::run()` and decoupled simulation from variable render dt.
- Tuned simulation tick rate to 60 Hz.
- Switched present mode preference to FIFO (MAILBOX fallback).
- Added finite `vkAcquireNextImageKHR` timeout handling to avoid indefinite waits.
- Widened timeline wait stage for transfer dependency to reduce cross-queue sync sensitivity.
- Added and then tuned frame timeline stall logging:
  - warn only for meaningful lag (`>= 6`) with cooldown (`2s`).
- Added a small backoff sleep on non-progress render paths to prevent CPU busy-spinning.
- Increased frames-in-flight from 2 to 3 to smooth bursty workloads.

### Lesson
FIFO often reveals synchronization and pacing assumptions that MAILBOX masks. Deterministic simulation + bounded waits + conservative sync stages + small non-progress backoff gives smoother behavior and cleaner diagnostics.
