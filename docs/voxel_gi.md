---
layout: default
title: Voxel GI
---

# Voxel-Aware Global Illumination

This renderer uses a compute-driven voxel GI volume to approximate diffuse indirect light.

## High-Level Model

- A fixed `64 x 64 x 64` grid is centered around the camera (`kVoxelGiGridResolution = 64`, `kVoxelGiCellSize = 1.0`).
- GI radiance is stored in two ping-pong `Texture3D` images (`float16` RGBA format where supported).
- A separate occupancy/albedo 3D texture stores whether each GI cell is solid plus an albedo payload.

## Frame Pipeline

1. Build/update occupancy payload on CPU when needed.
2. Upload occupancy payload to `voxelGiOccupancy` 3D texture (RGBA8).
3. Run **inject** compute (`voxel_gi_inject.comp.slang`):
   - For each empty GI cell, inspect 6 neighboring cells.
   - If a neighbor is solid, treat it as a bounce source with that face normal.
   - Evaluate direct sun on that face (including cascaded shadow test).
   - Add SH sky irradiance contribution.
   - Modulate by surface albedo and write injected radiance.
4. Run **propagate** compute (`voxel_gi_propagate.comp.slang`) for 4 iterations:
   - 6-neighbor diffusion for empty cells.
   - `lerp(center, neighborAverage, propagateBlend)` then decay by `propagateDecay`.
   - Copy ping-pong output back for the next iteration.
5. Sample the final radiance volume in world shading (`voxel_packed.frag.slang`) and scale by GI strength.

## When GI Recomputes

GI is **not** forced every frame. It recomputes only when needed:

- World/mesh changes mark `m_voxelGiWorldDirty = true`.
- GI grid origin moves (camera-relative volume shift).
- Sun direction changes beyond threshold.
- Sun color changes beyond threshold.
- SH coefficients change.
- GI tuning sliders change.
- First initialization.

Occupancy upload is done only for world/grid/init changes. Lighting-only changes can reuse occupancy and run inject+propagate only.

## Debug and Tuning Controls

Renderer UI exposes:

- `GI Strength`
- `GI Inject Sun`
- `GI Inject SH`
- `GI Inject Bounce`
- `GI Propagate Blend`
- `GI Propagate Decay`
- `GI Ambient Balance`
- `GI Ambient Floor`
- GI visualization modes

Frame stats also show GI compute times:

- `GI Inject (compute)`
- `GI Propagate (compute)`

## Current Limitations

- Diffuse/low-frequency only (no specular GI).
- Single local 3D grid around camera can miss far-field lighting context.
- Thin geometry can still leak due to voxel resolution.
- Propagation is a diffusion approximation, not physically correct transport.

## Good Next Steps

1. Add **temporal GI accumulation** with camera-motion rejection to reduce flicker and noise.
2. Add **multi-resolution GI volumes** (near high-res + far low-res) to stabilize large scenes.
3. Improve leakage handling with directional occlusion or occupancy dilation around thin blockers.
4. Switch propagation to directional basis (e.g. low-order SH per cell) for better directional bounce.
5. Add optional per-cascade or per-region GI update budgeting to cap worst-case frame cost.
