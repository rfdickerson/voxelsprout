# Shadow Occluder Querying

This document describes how shadow caster selection works in the current renderer.

## Goal

Reduce shadow-pass draw calls without missing important occluders that cast onto camera-visible receivers.

Naive camera-frustum culling is not sufficient for shadows, because offscreen casters can still project onto visible pixels.

## High-Level Strategy

The shadow pass uses a conservative, receiver-driven query in three stages:

1. Build a **receiver seed set** from clipmap-visible chunks (`visibleChunkIndices`).
2. Expand that set upstream along light direction (`-sunDirection`) to include offscreen potential casters.
3. For each candidate chunk, test intersection against each cascade clip volume and build **per-cascade** draw lists.

Only chunks that survive these tests are emitted into shadow indirect draw commands.

## Detailed Flow

### 1) Receiver Seeds

The main pass already has a clipmap/spatial result: `visibleChunkIndices`.

Those chunks are treated as shadow receivers and become the seed set for occluder lookup.

If `visibleChunkIndices` is empty, shadow culling falls back to scanning all chunks (safe fallback).

### 2) Upstream Expansion (Conservative Caster Sweep)

For each seed chunk:

- Compute chunk center in world space.
- March sample points along `-sunDirection`.
- Convert each sample position to chunk coordinates and mark that chunk as a candidate.

This expansion is conservative: it includes potential offscreen casters that can shadow visible receivers.

Current tunables in code:

- `kShadowCasterExtrusionOverscan = 48.0f` world units.
- Extrusion base distance uses far cascade split (`m_shadowCascadeSplits[last]`).
- Sample spacing is half of the minimum chunk axis size.

### 3) Cascade Intersection and Masking

Each candidate chunk is tested against each cascade via:

- `chunkIntersectsShadowCascadeClip(chunk, lightViewProjMatrices[cascade], margin)`

If the chunk intersects cascade `i`, bit `i` is set in a cascade mask.

Current margin:

- `kShadowCasterClipMargin = 0.08f`

### 4) Per-Cascade Draw List Construction

For each chunk/Lod draw range that has a non-zero cascade mask:

- A shadow instance is appended once to shared shadow instance data.
- An indirect draw command is created.
- The command is copied into each `shadowCascadeIndirectCommands[i]` where mask bit `i` is set.

This avoids the previous behavior where all cascades drew one unioned chunk list.

Result: each cascade draws only its own occluder subset.

## Why This Works

- Receiver-driven seeding prevents wasting work on distant unrelated chunks.
- Upstream expansion preserves correctness for offscreen casters.
- Cascade-specific command lists reduce overdraw from cascade overlap.
- Final cascade clip test keeps the expansion conservative but bounded.

## Data Dependencies

The algorithm lives entirely in render-side data and uses:

- `visibleChunkIndices` from spatial/clipmap query
- chunk bounds and mesh draw ranges
- per-frame light matrices (`lightViewProjMatrices`)
- existing shadow cascade settings/splits

No additional subsystem or persistent scene graph is introduced.

## Current Tradeoffs

- The upstream march is conservative and may still include extra chunks in dense worlds.
- March sampling uses fixed spacing; too coarse can miss thin cases, too fine costs CPU.
- Magica/pipe/grass shadow draws are still broad; chunk occluder filtering currently targets chunk mesh commands.

## Practical Tuning Knobs

If you need further reduction:

1. Lower extrusion overscan.
2. Shorten extrusion distance multiplier from far cascade split.
3. Increase clip margin precision only if you see popping.
4. Add per-cascade upstream distances (near cascades can use much shorter extrusion).
5. Add separate culling for non-chunk shadow casters (pipes/grass/magica) if they dominate.

## File Location

Primary implementation is in `src/render/Renderer.cpp` inside `Renderer::renderFrame`, around shadow candidate generation and shadow indirect command building.
