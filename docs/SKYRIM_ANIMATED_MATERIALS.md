# Skyrim animated materials

The Skyrim P1 material-animation path now imports linked Bethesda shader float
and color controllers into typed tracks, cooks those tracks with their material,
and samples them in the existing imported-scene renderer. Supported animated
waterfall/creek geometry also passes the former blanket `Effects` directory
exclusion. Particle-only effects and fire presets keep their existing paths.

## Supported scope

- `BSEffectShaderPropertyFloatController`: UV offset/scale, alpha and emissive multiplier.
- `BSLightingShaderPropertyFloatController`: UV offset/scale, alpha, emissive multiplier,
  specular strength, glossiness and environment scale.
- Both Bethesda shader color-controller families: effect/emissive RGB and lighting
  specular RGB. Palette-based effects retain animated palette rows and intensity.
- `NiFloatInterpolator`/`NiFloatData` and `NiPoint3Interpolator`/`NiPosData`:
  linear, quadratic/Hermite, constant and TCB keys; constant interpolator poses.
- `NiFlipController`: ordered diffuse texture frames with resident texture remapping.
  Missing decoded frames retain the static diffuse binding. Missing source references
  reject the track and are reported.
- Active linked-controller chains, frequency/phase, start/stop, looping, reverse
  cycles, backwards playback and clamping. Detached tracks do not automatically play.

All 71 tracks on the 13 inspected Riverwood effect models resolve, including
waterfalls, rapids, creek sheets, splashes and ant trails. This counts tracks on
imported shapes, not unique NIF controller blocks or proven visible pixels. The
initial audit exposed that those models were excluded during cell building; the
final audit verifies all **26 eligible placements** produce scene instances after
removing that blocker. Total preserved scene outputs rise from 888 to 914.

## Data and runtime

Animated materials retain original UVs and vertex coverage. The renderer applies
material UV transforms and opacity once, in both the main and shadow vertex
passes. Source lighting coefficients are evaluated through the same material
table as static materials. No second renderer or render graph was introduced.

Cooked scene version **39** appends bounded animation records, preserving readers
for versions **34–38** with neutral animation defaults. Cell cache version **103**
invalidates the old baked-UV and effect-exclusion results. Texture frames belong to
the cell's existing texture residency list. Material slots own animation state;
eviction removes it, and reloading starts a fresh local clock. The clock does not
wrap at the water shader's 4096-second interval.

Each frame writes only its retired, independent material-buffer region. After an
initial table upload, steady-state updates copy only animated entries. Vertex and
fragment descriptor visibility include the shared material table. The compact
shadow stream now carries the same material index as the main stream.

The GPU material stride grows from 96 to 160 bytes: **8 MiB additional allocation**
across two 65,536-entry frame regions, plus 4 MiB for the CPU mirror. Compact
shadow vertices grow from 32 to 36 bytes (**4 bytes per allocated shadow vertex**).
The daylight capture allocated 6,055,184 shadow vertices, making the shadow-stride
addition about 23.10 MiB at that capacity (31.10 MiB including the material regions).
Additional restored effect geometry and textures have their own residency cost;
these structural figures are not a measured total memory delta.

## Validation

- 39/39 Debug CTests pass, including new controller/sampling/serialization tests.
- 5/5 focused optimized import/material/coverage tests pass.
- Full optimized suite: 35/39 pass. The pre-existing inventory, Bethesda runtime,
  save and world-map-cache failures remain; the runtime test segfaults.
- Synthetic full NIFs prove linked controllers reach shapes without double-baked
  UVs or opacity. Tests include malformed/truncated payloads, missing references,
  controller cycles, nonfinite data, interpolation/timing, texture frames, full
  and runtime scene reloads, and version-38 neutral defaults.
- The GPU smoke test exercises animation with camera motion and cell eviction/
  reload under Vulkan synchronization validation.
- Local Riverwood captures and timing manifests use the `riverwood-animated-effects-`
  prefix under `captures/`. Asset coverage is local at
  `captures/asset-coverage/riverwood-animated-effects.{json,md}`.

### Native 1080p diagnostics

Intel Graphics (LNL), native temporal rendering, 16× anisotropy, validation on,
240 warmup frames, fixed mill camera. Last logged values:

| Weather | GPU frame (ms) | Rolling p50 (ms) | Rolling p95 (ms) |
|---|---:|---:|---:|
| Day | 41.18 | 46.19 | 139.88 |
| Overcast | 41.87 | 40.26 | 75.17 |
| Dusk | 38.62 | 39.72 | 80.80 |
| Night | 38.55 | 38.45 | 54.99 |

The 3840×2160 waterfall detail still is
`captures/riverwood-animated-waterfall-4k.png`; its last logged GPU frame is
63.69 ms, rolling p50 64.24 ms and p95 84.22 ms, with validation enabled.

These runs include streaming; they do not isolate animation cost. The broader
99-cell video preload/encoding run is heavier: last logged GPU frame 134.53 ms,
rolling p50 136.27 ms and p95 147.21 ms. Do not compare that recording directly
with the fixed-view stills as a performance regression.

## Boundaries

This completes the selected Skyrim material-animation path, not every Gamebryo
controller or effect shader. Legacy `NiUVController`/`NiTextureTransformController`,
controller-manager/blend interpolators, effect falloff/refraction controllers,
non-diffuse texture flipping and particle atlas animation remain unsupported.
Particle emitters/modifiers are a separate roadmap item; animated water sheets
alone do not restore waterfall spray. Skeletal actor material animation is not
connected through this static imported-material path.

The 16-second, 480-frame native-1080p recording
`captures/riverwood-material-animation-route.mp4` follows an authored forest,
village and waterfall route. Its command/settings and waypoint file are retained
locally. The route and the synthetic motion/eviction test report no Vulkan
validation errors. Inspected route frames show the restored waterfall sheets;
foliage aliasing and incomplete waterfall spray remain visible limitations.
Retail-equivalent color, transparency, TCB timing and absence of ghosting are not
established by these checks.

Format references: [NifTools definitions](https://raw.githubusercontent.com/niftools/nifxml/develop/nif.xml).
GPU update policy follows the existing per-frame design and the
[Khronos synchronization guide](https://docs.vulkan.org/guide/latest/synchronization.html).
