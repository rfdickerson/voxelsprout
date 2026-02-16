# Spatial Partitioning Plan

## Goal
Reduce per-frame CPU work and draw submission overhead by culling chunks and dynamic factory entities before render pass command generation.

## Recommendation
Use a hybrid structure:

1. `ChunkSpatialIndex` (linear tree over chunk AABBs) for voxel chunks.
2. `DynamicSpatialGrid` (uniform hash grid) for dynamic entities like pipes, belts, tracks, and train cars.
3. Keep the current indirect draw path and feed it with culled lists.

This matches project constraints:
- deterministic behavior
- simple data-first structures
- easy incremental integration

## Why not a single full BVH?

- Chunk bounds are static and grid-aligned. A chunk tree is cheaper to keep correct than a dynamic BVH.
- Factory entities are edit-heavy and often move/extend; hash-grid updates are O(1)-style and easy to reason about.
- Hybrid keeps implementation small and readable.

## Data Structures

### ChunkSpatialIndex (implemented foundation)
- Input: `ChunkGrid::chunks()`
- Stored:
  - chunk AABBs (world-cell aligned)
  - deterministic sorted chunk index array
  - binary linear nodes (`bounds`, `childA`, `childB`, leaf range)
- Query:
  - `queryChunksIntersecting(bounds)` returns chunk indices
  - optional `SpatialQueryStats` for debug counters

### DynamicSpatialGrid (next)
- Cell size: tune to gameplay scale (recommended: 8-16 voxels).
- Key: integer cell coordinate hash.
- Value: compact vector of entity ids.
- Supports:
  - insert/remove/update per entity
  - broad-phase query by bounds

## Renderer Integration Plan

### Phase 1: Chunk visibility (next)
1. Build camera world-cell AABB for current view range.
2. Query `ChunkSpatialIndex`.
3. Build visible chunk draw list from query results only.
4. Feed indirect command generation from this list.

Expected gain: reduced CPU iteration over non-visible chunks across shadow/prepass/main.

### Phase 2: Cascade-aware shadow culling
1. Build per-cascade bounds.
2. Query index per cascade.
3. Build cascade-specific indirect lists.

Expected gain: lower shadow pass draw count.

### Phase 3: Dynamic entity culling
1. Add `DynamicSpatialGrid`.
2. Register all dynamic render entities.
3. Query per frame and emit instance buffers only for visible entities.

Expected gain: fewer instance uploads and draw work for large transport networks.

## LOD Integration
- Keep existing deterministic LOD policy.
- Apply LOD selection only to chunks returned by spatial query.
- Preserve sorted output for stable behavior and repeatable frame-to-frame results.

## Debug/Validation Counters
Track in debug UI:
- total chunk count
- spatial nodes visited
- candidate chunks
- visible chunks (main + each cascade)
- visible dynamic entities
- indirect command count by pass

## Risk Notes
- Query bounds too conservative: less gain but safe.
- Query bounds too tight: visual pop-in. Start conservative and tighten iteratively.
- Keep deterministic ordering of results to avoid flicker and non-reproducible debugging.

## Current Status
- Added foundational `ChunkSpatialIndex` subsystem:
  - `src/world/spatial_index.h`
- Rebuilt at app world load/regenerate:
  - `src/app/app.cc`

Renderer still uses existing visibility loop for now. Next step is Phase 1 integration.

