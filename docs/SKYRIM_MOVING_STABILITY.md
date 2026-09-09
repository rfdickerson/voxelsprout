# Moving-camera stability — 2026-09-08

This pass retains the small 768×432 logical window, native 1536×864 rendering,
original texture policy, detail radius 1 and LOD radius 5. No quality reduction or
larger preloaded detail corridor is used. Full visual and uninterrupted-60-fps
acceptance remains open.

Local evidence viewer: [moving stability](../captures/moving-stability/index.html).
The capture inventory is `captures/moving-stability/summary.json`.

## Completed visual coverage

**9 validated cases / 11,400 lossless native frames**, with zero Vulkan validation
errors or warnings and clean renderer-owned image shutdown checks:

| Recording | Frames | Revision scope |
|---|---:|---|
| Road route, four weather/time conditions | 7,200 | Tree residency correction; before camera precision correction |
| Identical daylight road route | 1,800 | Final camera and residency corrections |
| Riverbank strafe/return, all four conditions | 2,400 | Final camera and residency corrections |

The manifests identify exact binary/shader hashes and route contents. Final
daylight route and riverbank captures use binary hash prefix `671a0b2f`; the
four-condition road baseline uses `563882a2`. The final camera correction was
not followed by another full four-condition road matrix; all four conditions
were instead exercised on the final riverbank pan.

Selected route, handoff and consecutive native-pixel samples were inspected:

- No broad shoreline fallback rectangle or continuous dark bank seam was found
  in the sampled pans. High-contrast water highlights still change sharply.
- Foliage silhouettes retain fine aliasing. The precision correction removes a
  camera error, but incomplete vegetation velocity/alpha filtering remains open.
  Raw adjacent-frame image change includes wind, water and intended camera motion;
  it is not a calibrated shimmer score.
- Road edges remain visible without a newly identified broad AO seam on this
  route. The exact originally reported patch is still unconfirmed.
- Dusk and night retain visible ground/building detail near authored emitters;
  unlit banks and the approach remain dark. Four road night samples have mean
  display luma 9.3–18.4/255, with 4.7–22.2% at luma ≤5. These are descriptive
  statistics, not matched retail exposure targets.
- LOD replacement is still abrupt. Sampled handoff frames and native crops are
  retained; a blend/fade and stronger object-specific temporal acceptance remain
  pending. This pass does not test weather changes during a single traversal.

## Reproduction

The road route follows authored road placements from the forest approach into
Riverwood. `scripts/fixtures/riverwood-forest-road.tour` retains its camera and aim
points. The old “forest” fixture was beside the mill and is excluded from this
route's acceptance. The riverbank fixture uses a short strafe and return to
exercise water edges and foliage disocclusion.

```bash
python scripts/riverwood_visual_acceptance.py \
  --fixture forest-approach --look day overcast dusk night --frames 1800 \
  --tour scripts/fixtures/riverwood-forest-road.tour --validation \
  --output captures/moving-stability/visual
python scripts/riverwood_visual_acceptance.py \
  --fixture water --look day overcast dusk night --frames 600 \
  --tour scripts/fixtures/riverwood-water-pan.tour --validation \
  --output captures/moving-stability/water
python scripts/riverwood_stability_benchmark.py \
  --output captures/moving-stability/final-timings
```

Load the installed Vulkan SDK environment before validated captures. Output case
directories must not already exist. Evidence stays local. The visual viewer stores
lossless frames and manifests; capture readback timings are **not** FPS results.
Performance runs capture only an endpoint still, discard 240 stationary warmup
frames and retain all subsequent wall-clock outliers in `frames.csv`.

The four authored weather/hour presets are SkyrimClearFF at 10.5, SkyrimCloudyTU
at 13.0, and SkyrimClearFF at 19.25 and 0.5. Camera simulation advances at 60 Hz;
async cell completion remains dependent on actual worker/GPU timing. The route
crosses cell boundaries but does not travel far enough to force eviction under
the normal unload margin. Eviction/revisit stress remains a separate check.

## Changes

- Added opt-in frame interval/CPU tick/render CSV output (`ODAI_FRAME_STATS_CSV`).
  Interval row N measures the work from row N−1; the report retains this distinction.
- Added tour streaming mode (`ODAI_TOUR_STREAMING`) to bypass capture-only route
  pinning, plus stationary warmup and fixed-step performance tour controls.
  Normal interactive camera behavior is unchanged.
- Corrected the forest route fixture and preserved the tour in capture manifests.
- Tree LOD now invalidates on cell arrival and eviction as well as camera movement.
  Previously a late detailed-cell arrival after the camera stopped could leave its
  billboard until the next camera-cell change. This is a residency correctness
  fix, not a transition fade. A synthetic BTT fixture verifies removal/restoration
  against changing detail residency while the requested LOD tile stays fixed.
- Camera orientation is built at the origin before applying world translation.
  The previous `lookAt(eye, eye + unitForward, up)` lost direction precision at
  large coordinates. A CPU reproduction found basis-element error 0.000686 near
  48,800 units and 0.004335 at 180,000 units. The new regression requires identical
  orientation at all tested translations and verifies the eye maps to the origin.
  This removes a numerical jitter source; it does not eliminate material aliasing.

## Findings and remaining work

Final optimized, unvalidated native-DPI traversal intervals, 1,799 measured frames
per condition after 240 warmup frames:

| Condition | Average FPS | p50 | p95 | p99 | Maximum |
|---|---:|---:|---:|---:|---:|
| Daylight | 59.79 | 16.23 ms | 29.26 ms | 30.95 ms | 700.81 ms |
| Overcast | 59.68 | 16.26 ms | 28.80 ms | 31.06 ms | 687.82 ms |
| Dusk | 59.99 | 16.11 ms | 28.92 ms | 30.81 ms | 653.02 ms |
| Night | 59.97 | 16.26 ms | 29.97 ms | 31.44 ms | 680.26 ms |

These are CPU-observed wall-clock frame intervals, not GPU timestamps or guaranteed
displayed-frame delivery. GPU timestamp samples and queue waits remain in each
runtime log. The frame-time tail, including non-streaming queue waits, fails stable
60 fps. The four diagnostic camera traces are identical: 2,040 samples each,
maximum translation step 6.73 units, complete route endpoint reached. Exact trace
agreement does not make worker completion or animated inhabitants deterministic.

The initial daylight benchmark averaged 64.23 fps, with p95 17.25 ms and a
672.93 ms maximum. The largest intervals were dominated by CPU tick work. Runtime
logs show synchronous rebuilding/uploading of the 121-tile object-LOD window as
residency changes. Those stalls remain open; a higher average does not meet an
uninterrupted 60 fps target. Avoid rebuilding/uploading the full window to change
nearby handoff state, while retaining correct child-worldspace clipping.

Hard tree/object LOD handoff, residual foliage/water aliasing, and matched retail
lighting/material comparisons are not signed off by the code correction. No
matched retail reference set is attached. The original authored distant atlas
limits and current detailed range remain in effect.

Optimized runtime/shader build passed. CTest remains 35/39: the same previously
recorded inventory, Bethesda runtime, save, and world-map cache failures. Import
coverage including the new residency fixture passes. Local logs are under
`captures/moving-stability/`.
