---
layout: default
title: Voxel GI
---

# Voxel-Aware Global Illumination

This renderer uses a voxel-aware GI path that builds a low-frequency radiance field in a `Texture3D`, then samples it during world shading.

## Pipeline

1. Build/maintain a fixed voxel GI volume (`64 x 64 x 64`, float16 RGBA radiance).
2. Inject direct lighting into the volume with a compute pass:
   - Reconstruct voxel cell world position.
   - Test sun visibility via cascaded shadow map.
   - Write direct radiance + small ambient seed.
3. Propagate lighting with a second compute pass:
   - Read neighboring cells (6-neighbor stencil).
   - Diffuse/mix and attenuate energy.
4. Sample GI in material shading:
   - Convert shaded world position to GI UVW.
   - Fetch radiance from the 3D texture.
   - Add as diffuse indirect term (modulated by vertex AO).

## Why This Fits VoxelSprout

- Grid-native: aligns with voxel world representation.
- Deterministic and data-parallel: clean compute-shader pass structure.
- Dynamic: updates each frame with sun/shadow changes.
- Cheap diffuse bounce: captures broad ambient transport without ray tracing.

## Current Limits

- Low-frequency only (not specular GI).
- Can leak through thin geometry due to coarse resolution.
- Single propagation step is lightweight but not multi-bounce accurate.

## Tunable Parameters

- GI grid resolution.
- Cell size / world coverage around camera.
- Injection strength.
- Propagation blend and attenuation.

