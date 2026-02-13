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
   - sun shafts composite
   - ACES filmic tonemap + saturation + gamma

## Mip-Chain Bloom

Bloom uses the same HDR scene texture sampled at multiple mip levels.

- Mips are generated in `Renderer.cpp` using `vkCmdBlitImage`.
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
- Optional sun shafts texture composite (mostly sky-only mask).

## User Controls

From renderer debug UI:

- `Bloom Threshold`
- `Bloom Soft Knee`
- `Bloom Global Intensity`
- `Bloom Sun Boost`

Related controls that affect perceived bloom:

- Auto exposure enable/manual exposure.
- Volumetric fog density/scattering.

## Current Limitations

- No dirt mask or lens texture response.
- Fixed per-mip weights (not artist curve-driven yet).
- No temporal anti-flicker specifically for bloom extraction.
- Bloom and fog interplay is tuned heuristically, not camera/lens physically based.

## Good Next Steps

1. Add a **lens dirt mask** modulated by bloom energy for stronger highlight character.
2. Expose a **per-mip weight curve** (or 2-parameter curve) instead of fixed constants.
3. Add **temporal stabilization** for bloom extraction to reduce shimmer on subpixel highlights.
4. Add **clamp/debug views** for pre-bloom, extracted bloom, and final bloom contribution.
5. Optionally split bloom into **near/far bands** for more cinematic control.
