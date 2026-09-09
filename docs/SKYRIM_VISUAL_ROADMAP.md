# Skyrim visual parity roadmap

Updated 2026-09-09. Target: unmodded Skyrim Special Edition, with Riverwood as the
acceptance scene. Keep the small **768×432 logical-point window**, native display
DPI, and the current detailed-cell range. On the tested 2× display, rendering and
presentation are **1536×864** with no upscaling. The user explicitly chose LOD
correctness over extending the detailed range.

Earlier native average performance met **60 fps in the measured small-window views**:
60.70 fps at the mill, 62.78 fps on the moving route, and 61.34 fps at the overcast
riverbank, all at **1536×864**. Some frames still exceed 16.67 ms; streaming stalls
and a strict uninterrupted 60 fps floor remain open. Earlier 64–69 fps results at
768×432 pixels are historical. See [measurements and tradeoffs](RIVERWOOD_PERFORMANCE.md).
The previous longer moving-stability rerun averaged **59.7–60.0 fps**, with
650–700 ms outliers. The first retained-tile/preloading/BTO-fade increment improves
the same four-condition route to **60.9–63.8 fps**, p95 **18.6–19.4 ms**,
p99 **21.8–22.7 ms**, with **203–212 ms** remaining tails. Stable 60 fps remains
open; see the
[implementation and measurements](SKYRIM_STREAMING_HANDOFF.md).
Use original installed assets, one imported-scene renderer, and preserve working-tree
changes. This document is the current summary; milestone documents retain detailed
scope and local evidence.

“Delivered” means the stated implementation scope is available and tested. It
does not mean the entire asset family or retail visual equivalence is complete.

## Progress

Latest distant-asset corrections restore Skyrim's two-plane BTT layout, tree-atlas
sRGB sampling, consistent billboard lighting, and LOD vertex RGB. The current
detailed-cell range is retained by user preference. See
[distant asset audit and validation](SKYRIM_DISTANT_ASSETS.md); authored terrain
atlas resolution remains a visible limit, and matched retail parity is unverified.

| Milestone | Status | Implemented | Remaining |
|---|---|---|---|
| Measurable asset coverage | First deliverable complete | Winning virtual assets and plugin records; resolution/decoding/scene-preservation evidence; unsupported, missing, malformed, excluded and unassessed classifications; deduplicated asset/placement counts; local JSON and Markdown with Riverwood examples | Instrument actual runtime/GPU consumption; trace transitive record dependencies; classify unassessed blocks/flags/families; expand detailed inspection beyond four Riverwood cells |
| P1 NIF lighting materials | Standard path delivered; vertex RGB corrected | Typed material identity independent of diffuse texture; authored UVs, independent vertex RGB/alpha enables, alpha/clamp modes, normal/specular/emissive parameters and texture roles; cooking and residency | Remaining specialized shader families/flags, model-space normals, unsupported texture slots, and matched retail material response |
| P1 LTEX/TXST terrain | Diffuse/normal path delivered; override removal corrected | Authored terrain associations/blending and normal channels reach scene bindings; shared albedo no longer merges distinct normal/material identities; winning LTEX/TXST records clear removed channels and missing links; SNAM exponent, DNAM flags, normal-alpha gloss and terrain model-space normal conversion | Remaining authored terrain channels; diagnose dropped landscape layers observed outside the acceptance sample |
| P1 cubemaps/environment masks | Supported DDS cube path delivered | Six faces and mips, residency, environment masks, authored environment scale and coordinate conversion; reflection shader path; native signed/unsigned BC6H HDR cubes and mip chains | Matched retail orientation/roughness response; cube arrays and remaining unsupported DDS layouts |
| P1 NIF particles | Mill-mist fixture delivered | Original mist texture; typed box/cylinder/sphere emitters and directional emission, lifetime/velocity/radius variation, color/scale curves, helper transforms, rotation, planar gravity and drag; alpha billboards with soft intersections; cell ownership and deterministic reset; capability-based admission with explicit fallback diagnostics | Turbulence and particle LOD; broader emitter/modifier/controller support; mesh emitters, multiple systems, atlases, smoke/fire/spray fixtures; scalable batching and transparent-surface ordering |
| P1 animated materials | Skyrim fixture path delivered | Linked Bethesda float/color controllers; UV/alpha/color and diffuse/normal/glow texture frames; linear/quadratic/constant/TCB timing; cooked v40 and cell-owned GPU updates; supported effect geometry admitted | Legacy UV controllers and blend/manager semantics, falloff/refraction, other texture slots/particle atlases, actor materials and matched retail visual acceptance |
| P2 IMGS/IMAD | Supported post-processing subset delivered | Weather/interior bindings, bloom, adaptation speed, cinematic grading/tint, no-sky DOF, modifier curves/fade, Apply/Remove and interruptible ApplyCrossFade/RemoveCrossFade, explicit overrides | Remaining HDR channels, radial/motion/double-vision blur, targeted/sky DOF, room/underwater overrides and exact authored transfer functions |
| P2 weather presentation | Partial | WTHR→SPGD import and dependency resolution; rain type/density fallback; corrected downward rain motion; continuous dawn/dusk fog-distance transitions; WTHR wind direction/range/speed supplied to existing wind controls | Authored world-space precipitation, roof occlusion, snow/ash, sky statics/aurora, RFCT, precipitation thresholds and maximum fog |
| Vegetation/grass | Placement foundation delivered | LTEX→GRAS associations, deterministic terrain-constrained roots, slope/water/density constraints, collision exclusion, cell ownership, authored species and distance fade | Position/color variation, wave period and vertex-lighting metadata; tree wind/LOD/alpha-coverage refinement; matched retail density and moving-camera shimmer checks |
| Lighting, AO and GI | Integrated; road self-occlusion corrections delivered | Indirect-only AO, exterior SSGI, Skyrim weather/light policy; XeGTAO reconstructs at fetched texel centers and offsets the surface for depth precision | Confirm the exact reported road-edge overlap; matched contact/halo, bounce, shadow-leak and history-rejection checks during motion/streaming/weather |
| Native resolution and brightness | Native DPI is the engine default | Small window retained; explicitly sized windows honor DPI; default upscaler quality Native; showcase logical-pixel/0.8-scale defaults removed; capture tools opt out explicitly; corrected Skyrim exposure startup | Display/resize regression checks; matched retail grading; eliminate streaming stalls and improve 60 fps frame pacing |
| Foliage temporal stability | Targeted corrections delivered | Neutral small-profile mip bias; depth-valid TAA history retained at silhouettes; depth rejection, clipping and motion limits retained | Quantitative moving-camera shimmer/ghosting tests; alpha coverage and specular aliasing |
| Water and shoreline | Targeted corrections delivered | Jitter-correct depth reconstruction; known riverbed fallback on refraction misses; depth-based shoreline fade; authored WATR appearance avoids legacy green wash/grade/luminance clamp | Matched retail optics, underwater presentation, reflection boundaries and specular shimmer; streaming/frame-time outliers |
| Distant terrain, objects and trees | LOD correctness increment delivered | Authored two-plane BTT layout, tree-atlas sRGB, consistent crown lighting, no invented crown wind, preserved LOD vertex RGB; complete authored texture mips | Matched retail lighting and handoff/blending; original low-resolution terrain atlas remains a limit; keep current detail range |
| P2 EFSH/ARTO/IPCT/IPDS | Not yet delivered | Census and known reader gaps recorded | Import and present effects, art objects and material-dependent impacts/decals after material/particle dependencies |
| P3 face/speech, HKX semantics, interface movies | Facial reader started; remaining P3 work deferred | FRTRI003 reader and indexed CPU expression evaluation; profile-aware probe decodes 569/569 local TRI assets; existing skeleton/animation, behavior topology plus state/wildcard transition-rule import ([scope](SKYRIM_HKX_BEHAVIOR.md)) and SWF font extraction retained | Actor/head-part morph binding and GPU deformation, FUZ/LIP synchronization, broader authored behavior execution, SWF display/actions; see [facial morph scope](SKYRIM_FACIAL_MORPHS.md) |

## Latest corrections and evidence

- **P1–P2 integration, September 9:** terrain specular/normal metadata, BC6H HDR
  cubemaps, normal/glow flipbooks, volume particle emitters, image-space
  crossfades and weather wind inputs are implemented. See
  [scope, validation and outstanding work](SKYRIM_P1_P2_PROGRESS.md). **The full
  P1–P2 roadmap is not complete.** Specialized materials, broader particles,
  world-space weather, vegetation metadata and EFSH/ARTO/impact presentation
  remain open.

- **Moving-camera stability:** added a road-guided forest-to-village tour with
  capture preloading disabled and per-frame native-DPI timing evidence. Corrected
  tree LOD invalidation on detailed-cell arrival/eviction. Synchronous object-LOD
  rebuilding still produces large frame-time outliers; full acceptance remains
  open. Camera orientation now avoids cancellation at large world coordinates.
  See [moving stability findings](SKYRIM_MOVING_STABILITY.md).

- **Night sky and emitters:** original Stars.nif/sky DDS assets and full-phase
  Masser/Secunda are visible in Whiterun. Skyrim local lighting now uses authored
  LIGH records only; phase/orbit parity remains pending.

- **Night lighting:** authored Skyrim night ambient replaces double-tinted fallback
  fill; IMGS contrast retains shadow detail. Whiterun market before/after, daylight
  and dusk inspected at native DPI. See [night correction](SKYRIM_NIGHT_LIGHTING.md).

- **Visual acceptance run:** 23 cases / 4,080 lossless native frames across four
  conditions, AO/TAA controls and two streamed routes. Full signoff is withheld:
  the original run failed night readability; residual temporal artifacts and a hard tree handoff
  remain open; the exact reported road patch and matched retail references are
  unconfirmed. The corrected street fixture now targets the actual village road.
  See [acceptance findings and local viewer](SKYRIM_VISUAL_ACCEPTANCE.md).

- **Native DPI:** verified an ordinary launch without DPI/render-scale/upscaler
  overrides: logical 768×432, framebuffer and render extent 1536×864, scale 1.
  Exact-pixel screenshot scripts explicitly set `ODAI_WINDOW_HIDPI=0`.
- **Foliage and road AO:** removed the small profile's negative mip bias, corrected
  silhouette history weighting, fixed XeGTAO sample reconstruction, and added a
  bounded depth-precision offset. The exact user-reported road patch still needs
  a matched camera check. Stills do not prove shimmer or ghosting elimination.
- **Water:** fixed refraction fallback patches and shoreline transitions; removed
  jitter mismatch and extra grading on authored water. Daylight/dusk validation
  passed. See [water and shoreline](SKYRIM_WATER_SHORELINE.md).
- **Distant assets:** retained 5,014 trees while restoring two planes per tree:
  30,084 → 20,056 triangles. Synthetic fixtures verify layout, sRGB, full mips,
  packing and cooked runtime reload. Original atlas sizes: trees 1024², objects
  2048², sampled four-cell terrain tile 256². No replacement detail or wider
  detailed-cell range. See [distant assets](SKYRIM_DISTANT_ASSETS.md).
- **Animated materials:** 71 tracks on 13 models restore all 26 eligible Riverwood
  effect placements. Supported linked UV/alpha/color/texture controllers reach
  runtime with cell-owned lifetimes. See [animated materials](SKYRIM_ANIMATED_MATERIALS.md).
- **Earlier material/terrain fixes:** independent authored vertex RGB/alpha policy;
  LTEX/TXST winning records clear removed channels and missing links rather than
  retaining stale textures. Remaining terrain channels are preserved source data,
  not yet implemented effects.

Earlier optimized native measurements (validation off; 899 frames per run):

| View | Average FPS | Frame p50 | Frame p95 |
|---|---:|---:|---:|
| Mill, clear daylight | 60.70 | 15.89 ms | 16.88 ms |
| Forest-to-village moving route | 62.78 | 14.44 ms | 16.20 ms |
| Water-heavy riverbank, overcast | 61.34 | 15.73 ms | 16.67 ms |
| Previous street-labelled mill view, night | 62.46 | 15.40 ms | 16.42 ms |
| Corrected village road, daylight | 67.97 | 13.90 ms | 15.04 ms |

The performance pass fixed homogeneous page culling, gave reflections their own
visibility list, removed invalid main-camera screen-space inputs from reflections,
reduced equivalent shadow/fog calculations, and staggered shadow updates. Small-fast
uses 1024 far shadow maps (nearest stays 2048) and quarter-size reflection buffers;
reflection filtering/history stays at its own resolution. Native scene resolution,
textures and detail range are unchanged. Distant shadows/reflections lose some
fine detail. See [performance implementation and evidence](RIVERWOOD_PERFORMANCE.md).

Initial streaming still causes 614–777 ms outliers. Local evidence is
`captures/riverwood-native-60-{mill,route,water,street}.{png,json,log}`.
The optimized runtime and shader build passed. The 2026-09-09 optimized CTest
run passes **39/39**. The four earlier failures were caused by test setup inside
assertions being compiled out under NDEBUG. Test targets now retain assertions;
production libraries and runtime remain optimized. New terrain, HDR cube,
particle-volume, material-frame and image-space crossfade tests pass.
Dusk validation reports no Vulkan validation errors or renderer-owned image leaks.

The latest completed local coverage report is
`captures/asset-coverage/riverwood-animated-effects.{json,md}`. It records 172,882
winning assets, 880,028 active plugin records, 620 authored model placements plus
532 grass candidates, and **914 scene outputs** after restoring effect geometry
(the earlier 888-output report predates that fix). The report includes all 26
restored animated-effect placements. Actual GPU consumption is still unmeasured
by the CPU coverage probe; a new census has not been run for the latest LOD fixes.

## Next implementation sequence

1. **Stabilize native 60 fps.** Retained object-LOD tiles, predictive CPU preparation
   and readiness-controlled BTO fades are implemented. Next split large cell/tile
   uploads, cap prepared bytes and avoid remaining arena-copy stalls; see the
   [source audit and implementation sequence](SKYRIM_STREAMING_HANDOFF.md).
   Keep residency-based handoff correct, including inherited city proxies.
   Re-run the recorded moving route and address remaining frames over 16.67 ms.
   Keep the current window, native DPI, textures and detailed-cell range.
2. **Resolve visual acceptance failures.** The four-condition run and control
   recordings are complete; full signoff is not. Revalidate the corrected night
   lighting across the full matrix, confirm
   the original road patch, investigate residual foliage/water temporal changes,
   and validate a tree handoff fade. Matched retail captures and a longer
   forest-to-village/eviction route remain required. Retain authored asset limits.
   See [acceptance report](SKYRIM_VISUAL_ACCEPTANCE.md).
3. **Expand authored environmental particles.** Build on box/cylinder/sphere sampling with
   mesh emitters, hidden emission geometry and atlas/subtexture support for
   creek/splash/waterfall fixtures. Finish required turbulence/LOD and activation
   controls. Import selected smoke/fire assets; retain reported presets only for
   unsupported assets. Avoid replacing every fire preset before equivalent
   authored support is validated.
4. **World-space weather precipitation.** Reuse particle primitives for SPGD
   textures, motion and geometry. Add rain occlusion and snow, then consume the
   remaining weather-linked sky/transition metadata. NIF particle import alone
   does not complete this separate record path.
5. **Finish vegetation metadata and visual stability.** Connect grass variation,
   wind and lighting controls; inspect authored foliage backlighting/shadows,
   alpha coverage and tree/grass LOD transitions along a moving route.
6. **Close measured material/image-space gaps.** Order specialized shader and
   remaining post-processing work by affected visible fixtures and dependencies,
   using new coverage evidence instead of archive-file counts alone. Validate the new
   terrain specular path against matched references and diagnose dropped layers.
7. **Add EFSH/ARTO and impacts/decals.** Reuse the material, animation and particle
   infrastructure above. Keep P3 work and general gameplay expansion outside this
   visual milestone.

Refresh the coverage report after each increment, distinguishing preserved data,
implemented runtime paths and observed runtime consumption. Add runtime evidence
instrumentation alongside these milestones; the CPU probe currently marks GPU
consumption unmeasured.

## Validation still required

The capture harness, fixed views and moving routes exist. Individual prior
milestones have daylight/overcast/dusk/night captures, but the complete matrix has
**not** been rerun after the latest native-DPI, water and distant-asset fixes.

- Re-capture identical views in all four lighting conditions, including 1080p
  timing runs and native/4K detail views. Retain effect-disabled and material/AO/GI
  debug comparisons and the exact camera/weather/warmup configuration.
- Exercise a moving forest-to-village route with actual cell eviction, weather
  changes and reload. Earlier pinned-cell routes do not prove eviction behavior.
- Inspect foliage shimmer, LOD popping, alpha coverage, ghosting, shadow leaks,
  AO halos and excessive bounce. Mill mist has a synthetic GPU eviction/reload
  test, but broader authored-effect motion and retail dynamics remain unverified.
- Measure sustained GPU timings and total texture/process/GPU memory deltas.
  Existing short rolling timing windows include streaming variation and are not
  isolated feature-cost measurements; do not carry them forward as current performance.
- Add synthetic parser, missing-reference, malformed-input, load-order,
  serialization and streaming tests for each newly supported feature.
- Continue synchronization validation and visual regression checks for other
  supported games. Passing shared tests is not matched visual evidence for each game.
- Keep assertions enabled in optimized test executables and rerun the complete
  suite after subsequent integrated changes. The previous four setup failures
  are resolved; this does not replace GPU or retail-reference validation.

Validation covers the exercised configurations, not every pass or supported game.

## Compatibility and local evidence

Keep one explicit imported-scene rendering path, existing residency interfaces and
explicit barriers. Use typed materials/controllers/emitters; do not overload packed
vertex fields with unrelated semantics. Current cooked format is **40**, with neutral
loading of versions **34–39**. Generated cell-cache version is **108**. Change serialized
layouts only when required; invalidate generated caches when import semantics change.

Game assets and captures remain local and Git-ignored. Use synthetic fixtures for repository tests. Claim retail parity only where matched retail references
support it.

- [Coverage implementation and results](SKYRIM_ASSET_COVERAGE.md)
- [Rendering, native resolution, grass and capture evidence](RIVERWOOD_RENDER_VALIDATION.md)
- [NIF particle implementation and fixture audit](SKYRIM_NIF_PARTICLES.md)
- [Image-space support and limitations](SKYRIM_IMAGE_SPACE.md)
- [Weather presentation support and limitations](SKYRIM_WEATHER_PRESENTATION.md)
- [Animated material implementation](SKYRIM_ANIMATED_MATERIALS.md)
- [Performance and DPI policy](RIVERWOOD_PERFORMANCE.md)
- [Water and shoreline corrections](SKYRIM_WATER_SHORELINE.md)
- [Distant asset audit](SKYRIM_DISTANT_ASSETS.md)
- Local coverage: `captures/asset-coverage/riverwood-animated-effects.{json,md}`.
- Recent captures: `captures/riverwood-foliage-*`, `riverwood-road-ao-*`,
  `riverwood-water-*`, and `riverwood-distant-*` with adjacent command/settings logs.
- Native-default launch evidence: `captures/riverwood-native-default-live.{json,log}`.
