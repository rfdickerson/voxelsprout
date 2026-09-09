# Riverwood visual acceptance — 2026-09-08

This document retains the earlier acceptance run, including its pre-correction
night failure. See [the subsequent moving-camera pass](SKYRIM_MOVING_STABILITY.md)
for the road-guided traversal, updated night captures and current findings.

**Result: partial acceptance; full visual signoff is withheld.** This is a local
engine acceptance run, not a matched retail comparison. Night readability fails;
foliage/water temporal stability and LOD transition appearance remain open. The
reported road-edge artifact was not reproduced at the sampled village junction;
its exact user-observed location is still unconfirmed.

## Scope and evidence

23 cases, **4,080 lossless 1536×864 RGB frames**, fixed 60 Hz simulation. Retained
the small-fast profile: native render/output size, detail radius 1, LOD radius 5,
original texture upload policy, 16× anisotropy, neutral mip bias, AO/GI and material
animation. The performance profile's 1024 far shadow maps and quarter-size water
reflection buffer remain enabled. No rendering-quality settings were changed in
this acceptance pass.

Local viewer: `captures/visual-acceptance-locked/index.html`. Each case has PNG
frames, a runtime log and a manifest containing command, settings, executable and
shader hashes, RGB hashes and image dimensions. The viewer supports frame stepping
and native-pixel display; browser playback is for inspection, not timing.

| Fixture | Coverage | Initial engine position; yaw/pitch |
|---|---|---|
| Village road junction | 120 frames in each condition; AO-only and AO-off daylight controls | (22300, -60, 44400); 130 / -12 degrees |
| Riverbank | 120 frames in each condition | (21300, 500, 42300); 165 / -18 degrees |
| Foliage and distant crowns | 120 frames in each condition; TAA-off daylight control | (21300, 500, 42300); 137 / -3 degrees |
| Village-road streaming route | 240 frames in each condition | (20300, terrain eye height, 46000); heading -26 degrees, 250 units/s |
| River streaming route | 360 frames in each condition | (20200, terrain eye height, 44800); heading -45 degrees, 250 units/s |

Conditions: `SkyrimClearFF` at 10.5, `SkyrimCloudyTU` at 13.0,
`SkyrimClearFF` at 19.25 and 0.5 hours. "Overcast" is the existing capture preset
name for that authored cloudy weather, not a measured cloud-coverage target.

The village route was initially recorded under the filename prefix `forest-route`;
inspection establishes that it traverses the village street, not a forest path.
The viewer and current script call it `village-route`. The river route is linked
into the viewer from `captures/visual-acceptance/handoff-*`. It passes close to
wooden supports and is a diagnostic route, not a collision-safe gameplay walk.
Neither moving route pins a larger detailed corridor: normal cell residency is
allowed to change during capture. These routes do **not** complete a long
forest-to-village gameplay route or an eviction/revisit stress test.

## Findings

| Check | Assessment | Evidence and limit |
|---|---|---|
| Road overlay AO | Not reproduced at sampled junction | AO debug is smooth across the visible road surface; no broad planar seam appears in the daylight control. Contact under steps/vegetation remains. This cannot close the exact original report without location confirmation. |
| Road/terrain blending | Partial | Authored road overlays and irregular edges remain visible with AO disabled; no missing terrain patch was identified in the sampled road frames. Material scale/color has not been matched to retail. |
| Foliage motion/TAA | Partial; residual instability remains open | In the locked daylight comparison, leaf-region adjacent-frame RGB change is 1.197 with TAA and 1.327 without (0–255 scale), about 9.8% lower. This includes real wind and lighting changes; it is not a calibrated shimmer metric or proof that ghosting is eliminated. |
| Bank/reflection edges | Limited daylight/cloudy pass | No earlier opaque fallback rectangle or continuous dark shoreline seam was identified in the sampled stationary views. The moving river route retains visible reflections around supports and banks. Fine reflection detail is reduced by the performance buffer size. |
| Water highlights | Open | Bright high-frequency highlights still change sharply in consecutive frames. The current bank fix does not establish specular temporal stability. |
| Dusk/night readability | Fail | At frame 60, about **98.5%** of road pixels and **97.4%** of riverbank pixels have display luma ≤5/255 at night. Road mean luma is approximately 0.55/255. Dusk also obscures substantial ground detail. Do not approve night bank/terrain detail from these nearly black images. |
| Distant tree lighting | Partial | Crowns respond consistently to the four conditions, but remain visibly flatter than nearby branch meshes. No matched retail reference supports lighting-parity approval. The authored two-plane/card representation and atlas limits are retained. |
| Streamed LOD handoff | Functional, visual fade incomplete | River-route logs change trees handed to detail from 53 to 68; village route from 70 to 90. No missing tile was identified in the sampled transition frames. The importer drops a billboard when its cell is resident; this is a hard residency switch, not an implemented transition fade. |

Temporal comparison ROIs and results are in
`captures/visual-acceptance-locked/foliage-temporal-comparison.json`; lighting
statistics are in `luminance.json`. The leaf ROI is (1040,130)–(1390,540).
Static roof and sky controls show similar low changes between the two variants.
The initial, unlocked TAA control moved to a different view and is **excluded**:
its much larger differences were camera motion, not evidence of TAA improvement.

Largest river-route image differences include the camera passing extremely close
to supports; global frame differences cannot classify those as LOD popping.
Longer pans, reverse traversal, disocclusion tracking and targeted object-ID
comparisons are still required for a stronger temporal acceptance claim.

## Capture corrections

- Corrected `capture_riverwood.py --view street` to the actual road. The previous
  fixture (21400,420,42600; 145/-4) looked toward the mill. Earlier timing labelled
  "street" therefore does not establish performance for the actual village road.
- Stationary screenshot/sequence capture ignores desktop camera input. Tours and
  explicit benchmark movement still operate; normal interactive play is unchanged.
- Benchmark sequence warmup holds horizontal position and settles eye height on
  terrain. Recording no longer consumes an unpredictable route prefix while
  streaming is loading. Visual animation already uses the fixed capture clock.
- Added `scripts/riverwood_visual_acceptance.py`, reusing the existing capture
  profile. It refuses to overwrite case directories, verifies frame counts and
  native extent, stores lossless PNGs and hashes, and generates the local viewer.

The road fixture was located from authored placements including Road3way01
(reference 0x1cd88) and RoadChunkM02 (0x2c69f/0x2c69e). The local audit is
`captures/acceptance/landmarks.json`; no game files or extracted geometry are
committed. Superseded scouting/unlocked captures remain separate local evidence.

## Retail comparison and authored limits

No matched retail reference set was found in the local captures, Pictures or
Steam screenshot folders inspected. The supplied illustrative images have no
verified camera, hour, weather, load order or mod state. They are not ground truth
for pixel-level comparison, so **matched retail comparison is not complete**.

A usable reference set must identify Skyrim SE build, active masters/mods, INI
settings, worldspace/cell, camera position/orientation/FOV, resolution, weather,
hour and settled streaming state. Match the same landmarks and projection before
judging material scale, crown brightness, terrain blending or LOD transitions.
Keep the original screenshots and provenance alongside the engine manifests;
record any unavoidable FOV or asset-version mismatch rather than compensating
with invented detail or a larger detail radius.

The prior authored-asset audit remains the limit: tree atlas 1024²/10 mips, object
atlas 2048²/11 mips, sampled four-cell terrain atlas 256²/8 mips. Original mips and
two-plane BTT geometry remain in use. See [distant assets](SKYRIM_DISTANT_ASSETS.md).
No new source-resolution claim was inferred from image sharpness in this run.

## Validation and next work

Optimized runtime build and Python syntax checks passed. Nine sequence cases ran
with Vulkan synchronization validation enabled; their logs contain no validation
errors or renderer-owned image leaks. Fresh optimized CTest is **35/39**, with the
same pre-existing inventory, Bethesda runtime, save and world-map cache failures.
Offline readback and PNG encoding timings are **not GPU performance measurements**.
A separate optimized 899-frame daylight run at the corrected road camera measured
**67.97 fps**, p50 **13.90 ms**, p95 **15.04 ms**, p99 **16.22 ms**, at native
1536×864. It still included a 638 ms streaming outlier. Evidence:
`captures/acceptance/road-native-performance.{json,png,log}`.

1. Correct night sky/ambient/exposure balance against matched retail conditions;
   retain day grading and explicit user overrides.
2. Confirm the exact reported road patch; capture its AO/depth/material controls
   if it differs from the current junction.
3. Add longer foliage pans and water-edge/specular disocclusion fixtures; separate
   animation from aliasing before further TAA changes.
4. Implement/validate a residency-safe tree transition fade, and compare terrain
   handoff/material scale against matched retail references within the same range.
5. Complete a longer forest-to-village route and cell eviction/revisit test.

Reproduce into a **new** local directory:

```bash
python3 scripts/riverwood_visual_acceptance.py --fixture road water foliage \
  --frames 120 --output captures/visual-acceptance-new
python3 scripts/riverwood_visual_acceptance.py --fixture village-route \
  --frames 240 --validation --output captures/visual-acceptance-new
python3 scripts/riverwood_visual_acceptance.py --fixture road --look day \
  --variant ao --frames 60 --validation --output captures/visual-acceptance-new
python3 scripts/riverwood_visual_acceptance.py --fixture road --look day \
  --variant ao-off --frames 60 --output captures/visual-acceptance-new
python3 scripts/riverwood_visual_acceptance.py --fixture foliage --look day \
  --variant taa-off --frames 120 --output captures/visual-acceptance-new
```

## Night correction follow-up

The failed night captures above predate the authored ambient and contrast fix.
See [Whiterun market verification](SKYRIM_NIGHT_LIGHTING.md). The historical
measurements remain intact; full four-condition acceptance is still pending.
