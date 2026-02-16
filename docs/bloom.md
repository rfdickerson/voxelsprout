---
layout: default
title: Bloom
---

# Bloom

Bloom is implemented as an HDR, mip-chain-based post effect integrated with auto exposure and ACES tonemapping.

## Where It Runs

Render order is:

1. HDR scene render into resolve image.
2. Build HDR mip chain via linear blits.
3. Auto exposure compute (histogram + exposure update).
4. Tonemap pass (`tone_map.frag.slang`) that applies:
   - SSAO composite
   - volumetric fog composite
   - mip-chain bloom
   - analytic sun bloom boost
   - sun shafts / froxel volumetric light composite
   - ACES filmic tonemap + saturation + gamma

## Mip-Chain Bloom

Bloom uses the same HDR scene texture sampled at multiple mip levels.

- Mips are generated in `renderer.cc` using `vkCmdBlitImage`.
- Tonemap shader samples mips 0..5 and applies bright-pass extraction per mip.
- Weighted sum (current defaults):
  - mip0 `0.20`
  - mip1 `0.40`
  - mip2 `0.24`
  - mip3 `0.11`
  - mip4 `0.05`
  - mip5 `0.02`
- Bright-pass uses threshold + soft-knee (`extractBloom`).

This makes bloom work for any bright HDR source, not just the sun.

## Exposure Interaction

Bloom threshold is exposure-aware:

- Effective threshold is scaled by current exposure.
- Auto exposure is computed in two compute passes:
  - `auto_exposure_histogram.comp.slang`
  - `auto_exposure_update.comp.slang`
- Exposure state buffer provides dynamic exposure to tonemap.

Result: bloom intensity remains more stable across dark and bright scenes.

## Sun-Specific Additions

In addition to global mip-chain bloom, tonemap adds:

- Analytic sun bloom term (screen-space radial glow around sun direction).
- Occlusion attenuation near sun UV to reduce bloom leaking through occluders.
- Sun shafts / volumetric light texture composite (applies to sky and geometry with depth attenuation).

## Sun Shafts + Froxel Volumetric Pass

Sun-light shafts are produced in a dedicated compute pass (`sun_shafts.comp.slang`) before tonemapping.

### Inputs

- Camera UBO (`camera_uniform`)
- `normalDepthTexture` (view-space depth in `.w`)
- Cascaded shadow atlas (`shadowMap`)
- Push constants:
  - output `width/height`
  - ray-march `sampleCount`

### Core Method

This pass now behaves as a froxel-style per-pixel volume march:

1. Reconstruct per-pixel view ray from UV + projection.
2. Convert to world ray using inverse view rotation.
3. March from camera to scene depth (or max range for sky).
4. At each step:
   - evaluate fog density from height-fog params (`skyConfig4`)
   - sample cascaded shadow visibility at world position
   - accumulate in-scattering and transmittance
5. Write shaft intensity to `sunShaftTexture`.

This is participating-media integration, not just a radial blur mask.

### Composition in Tonemap

`tone_map.frag.slang` samples `sunShaftTexture` and adds it in HDR with:

- sun-tinted color
- shaft strength derived from fog density/scattering
- depth-based geometry attenuation (so shafts can appear on interior geometry, e.g. through windows, without full-screen haze)

## User Controls

From renderer debug UI:

- `Bloom Threshold`
- `Bloom Soft Knee`
- `Bloom Global Intensity`
- `Bloom Sun Boost`

Related controls that affect perceived bloom:

- Auto exposure enable/manual exposure.
- Volumetric fog density/scattering.
- Fog height falloff/base height (strongly affects shaft visibility indoors).

## Current Limitations

- No dirt mask or lens texture response.
- Fixed per-mip weights (not artist curve-driven yet).
- No temporal anti-flicker specifically for bloom extraction.
- Bloom and volumetric shaft interplay is tuned heuristically, not full atmospheric multiple-scattering.
- No temporal reprojection for froxel shafts yet (can shimmer under motion).

## Good Next Steps

1. Add a **lens dirt mask** modulated by bloom energy for stronger highlight character.
2. Expose a **per-mip weight curve** (or 2-parameter curve) instead of fixed constants.
3. Add **temporal stabilization** for bloom extraction to reduce shimmer on subpixel highlights.
4. Add **clamp/debug views** for pre-bloom, extracted bloom, and final bloom contribution.
5. Optionally split bloom into **near/far bands** for more cinematic control.
