# Riverwood small-window performance

Run the optimized runtime interactively:

```bash
python3 scripts/capture_riverwood.py --profile small-fast --play --view mill
```

Interactive play keeps a 768×432 logical-point window and enables native display
DPI (`ODAI_WINDOW_HIDPI=1`). On the tested 2× display, both the framebuffer and
3D render extent are **1536×864**; no fixed render-size override stretches a
768×432 image into that window. Captures still request exact output pixels.
The current native measurements are below. Older measurements later in this
document describe 768×432 pixel captures and are retained as historical evidence.

Native DPI is now the engine default, including explicitly sized windows;
`ODAI_WINDOW_HIDPI=1` is no longer required. The default upscaler quality is Native,
and showcase presets no longer force logical-pixel presentation or 0.8 render
scale. Exact-pixel capture scripts opt out with `ODAI_WINDOW_HIDPI=0`.

The profile uses detailed-cell radius 1 (nine cells),
LOD radius 5, and shadow distance 3000 engine units. Terrain tessellation stays
disabled. Original textures at the existing 2048 upload limit, 16× anisotropy,
animated materials, shadows, XeGTAO, exterior GI, and native temporal AA remain
enabled. Distant detail and shadow reach are reduced. This is an explicit launch
profile; the fidelity capture defaults and normal runtime defaults are unchanged.
`--size` and `--env` can override the profile. GPU desktop access and the runtime's
Vulkan library environment must be available as with other Riverwood runs.

## Native 60 fps pass — 2026-09-08

Optimized build, Intel LNL, validation off, **1536×864 render and output pixels**.
Each run captures 899 frames; the existing frame-stat recorder excludes the first
30 frames but still includes subsequent initial streaming stalls. These are
whole-run averages, not a guarantee that every frame stays below 16.67 ms.

| View | Average FPS | Frame p50 | Frame p95 | Frame p99 |
|---|---:|---:|---:|---:|
| Mill, clear daylight | 60.70 | 15.89 ms | 16.88 ms | 18.74 ms |
| Forest-to-village moving route | 62.78 | 14.44 ms | 16.20 ms | 20.05 ms |
| Riverbank looking down, overcast | 61.34 | 15.73 ms | 16.67 ms | 17.34 ms |
| Previous street-labelled mill view, night | 62.46 | 15.40 ms | 16.42 ms | 17.68 ms |
| Corrected village road, daylight (acceptance pass) | 67.97 | 13.90 ms | 15.04 ms | 16.22 ms |

The later visual acceptance audit found the old "street" camera was aimed at
the mill. Its night result above is retained with a corrected label; it is not
an actual village-road measurement. The new daylight road row uses the corrected
pose and has separate evidence in `captures/acceptance/road-native-performance.*`.
See [visual acceptance](SKYRIM_VISUAL_ACCEPTANCE.md).

Earlier native views were approximately 41–43 fps. Local evidence:
`captures/riverwood-native-60-{mill,route,water,street}.{png,json,log}`. Streaming maxima
remain 614–777 ms and require separate CPU/import/upload work. The goal is met
for average performance in these measured views; an uninterrupted 60 fps floor
and all-weather/all-location coverage remain unproven.

Changes:

- Homogeneous AABB clip tests reject pages wholly behind the eye without
  incorrectly discarding boxes crossing the eye. Reflections use their own
  mirrored frustum and reject wholly submerged pages with a movement margin.
- Main-camera AO, GI, light-cluster indices and contact masks are not reused in
  the mirrored view. Reflections retain world-space lighting and local lights.
- Four hardware bilinear comparisons implement the same nine-tap tent shadow
  filter. Authored-weather fog sums the same density samples and integrates
  constant incident light once instead of repeating segment shading.
- Cascade zero updates every frame; cascades 1–3 rotate one per frame. Cached
  matrices stay paired with cached depth, and residency invalidation still forces
  updates. Far shadows can be two frames old; `ODAI_SHADOW_INTERLEAVE=0` restores
  full-rate rendering.
- The small-fast profile uses `ODAI_SHADOW_FAR_RESOLUTION=1024`, preserving the
  nearest 2048 map and the 3000-unit reach. Other launches retain 2048 far maps.
  This trades some distant shadow sharpness for performance, not texture detail.
- Small-fast uses `ODAI_WATER_REFLECTION_DIVISOR=4` (384×216 here); the general
  default remains 2. Reflection history/filtering runs at reflection resolution
  instead of expanding into a full-size buffer before water sampling. Water
  distortion and native TAA remain active. This reduces fine reflected detail.
  The color/depth history pair saves approximately 28.5 MiB at this native size;
  raw per-frame reflection targets also shrink. No extra GPU resources or passes.

Validation: optimized runtime and both standard/RT shader variants built. The
fresh optimized CTest run passed 35/39; the same pre-existing inventory, Bethesda
runtime, save and world-map cache failures remain. New clip-culling tests cover
reverse-Z, behind-eye/eye-crossing boxes, side/near/far planes, orthographic bounds
and malformed bounds. A 10,000-sample numerical check of the separable PCF weights
matched to double-precision rounding. Dusk capture with synchronization validation
(`riverwood-native-60-validation`) reported no validation errors or live images.
Existing explicit barriers were retained, following the
[Khronos synchronization guidance](https://docs.vulkan.org/guide/latest/synchronization.html).

The native framebuffer, detail radius, authored texture resolution, anisotropy,
AO, GI, material animation and LOD correctness remain unchanged. Exact pixel
capture opts out of HiDPI only to avoid doubling an already explicit pixel size;
interactive play continues to use the display's native pixel density.

The small profile now uses neutral mip bias (0), replacing its inherited -0.35
sharpening bias to reduce minified foliage shimmer. Native TAA keeps normalized,
depth-valid history at silhouettes instead of attenuating it again by the valid
bilinear footprint. Tiny footprints still fade; depth rejection, variance clipping,
and motion limits remain enabled.

XeGTAO now reconstructs horizon samples at the fetched mip texel centers, including
clamped screen edges. Previously the sampled depth was paired with the unsnapped
march coordinate, fabricating height on sloping ground. A bounded surface offset
also compensates for FP16 depth rounding and near-coplanar road overlays. This
does not remove or classify road geometry. Exact reproduction of the user's
reported road-edge patch still needs an identified camera position.

After these fixes: fixed mill **67.98 fps**, p50 **13.74 ms**, p95 **14.83 ms**,
p99 **15.29 ms** (800 frames, first 30 excluded). Loading still produced a 712 ms
outlier. Optimized runtime/shaders built successfully; three focused CTest targets
passed, and the TAA policy test also ran with assertions explicitly enabled.
The road AO capture passed Vulkan synchronization validation with no reported
errors or renderer image leaks. Before/after stills are local under
`captures/riverwood-road-ao-*` and `captures/riverwood-foliage-*`; stills alone do
not establish a quantitative shimmer or ghosting improvement.

No GPU resources, descriptors, or pass dependencies changed. The existing explicit
synchronization remains in place, checked against the
[Khronos synchronization guide](https://docs.vulkan.org/guide/latest/synchronization.html).

Measured September 8, 2026 on Intel Graphics (LNL), RelWithDebInfo, validation off:

| Run | Wall-clock average | Frame p50 | Frame p95 | Frame p99 |
| --- | ---: | ---: | ---: | ---: |
| Fixed mill, 800 frames | 68.79 fps | 13.56 ms | 14.60 ms | 15.25 ms |
| Moving forest toward village, 800 frames | 63.62 fps | 13.80 ms | 15.25 ms | 17.40 ms |
| Interactive HUD, fixed mill, 25 seconds | 68.66 fps | 14.04 ms | 15.17 ms | 16.39 ms |

These distributions exclude the engine's first 30 frames but include subsequent
loading. Initial streaming still caused 763/846 ms worst frames respectively;
this profile meets 60 fps on average in these runs, not a guarantee that every
frame stays below 16.67 ms. Long routes, other views/weather, and cell eviction
still need broader testing. The full-detail 768×432 mill baseline had a late GPU
median of 41.02 ms; reducing resolution alone did not solve its geometry cost.

Local evidence is in `captures/riverwood-small-performance.*`,
`riverwood-small-tuned.*`, `riverwood-small-fast.*`, and
`riverwood-small-moving.*`. Each capture JSON preserves the exact command,
environment, executable hash, and GPU timings. Logs contain wall-clock frame
statistics. These assets and screenshots remain local.

To repeat the moving measurement, use `--profile small-fast --view forest
--frames 800 --env ODAI_FRAME_STATS=120 --env ODAI_FNV_BENCH=1
--env ODAI_FNV_BENCH_FIXED_DT=1 --env ODAI_FNV_BENCH_SPEED=250
--env ODAI_FNV_BENCH_TURN=0 --env ODAI_FNV_BENCH_HEADING=-45`.
