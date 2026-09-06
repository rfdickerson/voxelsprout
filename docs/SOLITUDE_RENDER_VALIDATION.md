# Solitude render validation

## Sunset volume capture

### Stronger sunset AO

Reproduce the updated still with `python3 scripts/capture_solitude.py --view sunset
--name solitude-sunset-strong-ao-4k --frames 600` (one shell command).
This preset retains the sunset camera, exposure and light shafts, and uses
XeGTAO radius 360 and intensity 2.8, up from the earlier capture's 180 and 1.4.
Fine scale remains 0.25. `solitude-sunset-ao-debug.png` isolates AO on the geometry
and confirms contact shading at posts, rocks, steps and door recesses.
`solitude-sunset-ao-off-4k.png` is the same lit capture with intensity zero as a
control. AO continues to affect ambient illumination and indirect diffuse wrap;
direct sunlight and volumetric scattering retain their own visibility.
The stronger capture visibly deepens the porch contacts and door/step recesses.

`captures/solitude-dramatic-sunset-4k.png` uses clear weather at 17.6 hours,
camera `-63500,-8200,-105600`, yaw 145, pitch 4, and exposure 0.85.
Light shafts are enabled with `ODAI_FNV_SHAFTS=1`; fog base is -8300,
density 0.00012, height falloff 0.0005, and scattering strength 3.
The companion JSON records the complete command, environment and shader hashes.

The volume ray now covers 8192 Bethesda units instead of 128, with at least
64 shadow samples. Composition adds the integrated scattering directly: the
former metre-scale depth attenuation and screen-space sun-visibility multiplier
incorrectly erased beams in front of buildings. Density, extinction and shadow
visibility are already evaluated in the volume integration. This increases the
cost of the optional shaft pass; performance has not been benchmarked.
The optimized shader build and focused render-policy/PBR tests pass.

## Authored placement correction (cache 82)

Removed the sign-model translation and the lantern FormID-specific translation.
Skyrim file-root NiNode transforms are replaced by the placed reference transform;
child-node transforms and standalone root geometry transforms remain authored.
The previous importer baked the root transform into geometry, reversing the sign
and moving the separately authored support arm away from both hanging objects.
The close captures `solitude-authored-root.png` and
`solitude-authored-root-angle.png` show the sign rings meeting the timber beam
and the lantern hanging at its end using unchanged REFR positions and scales.
The rule matches [OpenMW's NiNode root handling](https://github.com/OpenMW/openmw/blob/master/components/nif/node.cpp).
Synthetic import coverage checks root translation, rotation and scale replacement,
child translation preservation, and standalone shape-root preservation. The scene
serialization layout is unchanged; existing cooked scenes need rebuilding.

Earlier sign-offset captures are superseded and do not establish correct placement.

Local captures use the installed official Skyrim SE load order. No game assets are included in source control.

## Reproduce

```bash
cmake --build --preset linux-vcpkg-relwithdebinfo -j 6
python3 scripts/capture_solitude.py --view market --frames 600
python3 scripts/capture_solitude.py --view waterfront --frames 600
```

The capture script renders native 3840×2160 with temporal AA, 2048-pixel texture ceilings, 16× anisotropy, clear weather at 14:00, and XeGTAO (radius 180, intensity 1.4, fine scale 0.25, denoise 2). Exposure is fixed at 0.95 for the market and 0.70 for the waterfront; the original market captures used 0.52. The waterfront uses fog density 0.00002. PNG conversion preserves the rendered RGB bytes; there is no image grading or generated imagery.

Each capture has adjacent `.ppm`, `.png`, `.log`, `.json`, and `.patch` evidence. JSON records command, environment, executable and shader hashes, source revision/diff, and lossless image verification. Files are under the ignored `captures/` directory.

## Corrections

- **Placement:** TES4/TES5 reference matrices now transpose the ordinary Euler matrix. The previous positive-angle placement opened large gaps in streets, rotated buildings away from doors, and separated cliffs. Matching overhead captures demonstrate the correction. Single-axis and compound-axis regressions guard the sign and order. TES3 retains its previous convention. The convention agrees with [OpenMW's conversion implementation](https://github.com/OpenMW/openmw/blob/master/components/misc/convert.hpp) and the independent [Skyrim placement measurements, section 4.2](https://github.com/block-town/openmw-engineering-notes#42-refr-euler-is-clockwise--the-ref-matrix-is-the-transpose). AABB overlap scores alone were misleading for this scene.
- **Persistent cells:** persistent exterior records with dummy XCLC coordinates are kept separate from real grid `(0,0)` cells. Child worldspaces keep their persistent cell resident, including ordinary launches and transitions. The sentinel is excluded from default-spawn centroid calculations.
- **Object LOD:** inherited parent proxies are clipped against detailed child residency using Bethesda northing, not altitude. Residency changes invalidate the handoff; rebuilding waits for the load batch to settle. This removes duplicated coarse roofs and walls from the close view.
- **Clutter and debris:** streamed XESP references follow their indexed initial parent state, including opposite-state chains. Disabled siege debris no longer appears over an intact market. Authored barrels, chairs, signs, moss overlays, foliage, and banners remain. This resolves initial static state, not dynamic quest-driven enable propagation.
- **Water:** conventional mip dimensions round down. The bundled 1254×1254 normal map previously generated oversized odd mips. This now follows [Vulkan image mip sizing](https://docs.vulkan.org/spec/latest/chapters/resources.html#resources-image-miplevel-sizing).
- **Clouds:** zero-alpha edges of authored cloud sheets remain transparent; the old 0.35 floor exposed polygon boundaries.
- **Terrain:** the domain shader now declares the same fractional-odd tessellation spacing as the hull shader. The mismatched execution-mode validation error is gone; [Vulkan requires these modes to agree](https://docs.vulkan.org/spec/latest/chapters/tessellation.html).

Cell build cache version 78 invalidates stale generated geometry. The serialized scene layout is unchanged. Existing independently cooked scenes need recooking to receive corrected placements.

## Evidence and limits

The matching overhead progression is `solitude-overhead.png`, `solitude-overhead-fixed2.png`, and `solitude-overhead-negative.png`. The last is the intermediate clockwise-angle diagnostic; production uses the complete transposed matrix and has no diagnostic placement override. Market captures before and after initial enable-state resolution are `solitude-market-preview.png` and `solitude-market-clean.png`. `solitude-market-ao.png` isolates AO, and `solitude-work/market-final-crop.png` shows full-resolution contact regions.

The Debug CTest suite passes 30/30. Import regression coverage includes persistent dummy-grid identity, compound placement rotations, and initial enable-parent chains. The optimized suite passed 28/30; runtime/save tests fail in that configuration and contain setup calls inside `assert`, which NDEBUG removes. They are not being reported as passing.

Real-data camera audits and runtime logs are retained in `captures/solitude-work/`. Both final camera audits built nine sampled cells with zero invalid triangles, skipped visible shapes, or visible import failures. Audits test the local sampled cells and imported triangle validity, not every surface of the city or every future camera route. They do not establish vanilla parity.

**Vulkan validation remains unclean.** The water mip and terrain tessellation errors are absent after the fixes, but the validation run still reports image-layout, shader-feature, vertex-buffer, debug-label, geometry-write synchronization, and screenshot/presentation errors. See `solitude-validation.log` and `solitude-work/validation-summary.json`. Ordinary capture runs complete successfully with image teardown checks.

The market's large placement gaps and duplicate proxies are resolved in the inspected view. The waterfront still exposes a dark gap near the lower tower and imperfect distant scenery/reflection boundaries. Actor import also logs unsupported skeleton/face geometry. These remain quality limitations: the captures are substantial improvements, not a claim that all Solitude geometry is flawless or that the engine matches or exceeds vanilla Skyrim everywhere.

## Lighting revision

The flat-lighting follow-up uses `solitude-market-lighting-4k.png`. Camera, 14:00 time, weather, neutral grade, AO, and 0.52 exposure match the original market capture. SolitudeWorld now uses diffuse wrap 0.08 (formerly 0.35), ambient scale 0.45 (formerly 1.0), sunlight scale 1.20, and screen-space bounce strength 0.35. This reduces orientation-independent fill and retains bounded indirect light in visible recesses. Exterior lighting environment overrides now work outside the Whiterun showcase as well. Other worldspace defaults are preserved.

`solitude-lighting-afternoon.png` is a separate 16:00 comparison, generated with the new `--hour` capture option; it is not the matching-time before/after. The original image is preserved as `solitude-work/market-flat-before.png`. The optimized build and focused import/render-policy tests pass. Previously documented geometry and validation limitations still apply.

## Cast-shadow revision

Shadow-only and direct-ratio captures (`solitude-shadow-debug.png`, `solitude-direct-debug.png`) show the authored pennants, porch, and building silhouettes already present in the shadow atlas. Disabling TAA and SSGI did not recover them in the lit image. Temporary diffuse-only diagnostics isolated the cause: adding local-light diffuse back (`solitude-shadow-local.png`) washed out the cast shadows visible in the sun-and-ambient-only image (`solitude-shadow-diffuse.png`). Diagnostic shader overrides were removed.

SolitudeWorld now scales local lights to 0.08 of their previous strength at sun elevations of ten degrees or higher, smoothly returning to full strength at the horizon. Interiors and other worldspace defaults are unchanged. `ODAI_FNV_DAYTIME_LOCAL_LIGHT_SCALE=1` restores the prior balance for comparison. Both diffuse and specular local-light contributions use this scale. The correction changes light balance, not shadow-map resolution or AO darkness.

The final full-pipeline capture is `solitude-market-shadows-4k.png`, with the original camera, clear weather, 14:00 sun, AA, and SSGI. Market capture exposure is now 0.95 to compensate for the removed lamp fill. Pennant shadows cross the cobbles, the porch shades the foreground, and the right façade casts onto the street while barrels and timber retain detail. The optimized build and import, render-policy, and PBR tests pass (3/3). Previously documented Vulkan validation and waterfront geometry limitations remain.
