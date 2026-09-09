# Preloading and dithered LOD handoff

Implementation progress, 2026-09-08. The first increment is implemented: retained
object-LOD tiles, bounded worker requests, predictive CPU preparation, upload
readiness polling and dithered BTO replacement. Keep native DPI, the small window,
authored assets and the current visible detail range.

## Implemented increment

- Object LOD now uses one imported-scene chunk per four-cell BTO tile. Overlapping
  tiles remain resident; a changed clipping mask rebuilds only its tile.
- Workers resolve/decode the tile, apply the existing large-reference/partial-city
  clipping policy, and build packed scene data. The main thread still converts
  renderer vertex streams and stages/uploads a whole tile.
- A three-second velocity prediction, clamped to one tile, prepares the entering
  strip. Predicted-only assets remain CPU-only and cannot activate lights,
  particles, collision or rendering. Limits: two running jobs, four outstanding
  results, at most two hidden prepared requests. These are count limits, **not a
  hard byte budget**.
- Only one ready tile is uploaded per frame. Stale request results are discarded;
  worldspace changes discard tile state; worker lifetime is drained before the
  asset source is destroyed. Leaving the visible window releases GPU chunks.
- A replacement holds its outgoing BTO until the upload timeline completes, then
  uses a stable world-space complementary coverage mask for 0.4 seconds. Main,
  depth, shadows and reflection use the same state. Indirect batches split by
  transition state; authored alpha tests remain intact. Only intersecting directional
  shadow cascades are invalidated during a fade; unrelated distant tiles retain
  the nearby shadow cache.
- Fragmentation retries reserve limited headroom instead of an exact-size tail:
  at most 262144 extra elements per relocation (25 MiB across vertex streams,
  1 MiB of indices). This reduces repeated copies without doubling for a hole.
- All transition/readiness state is runtime-only; cooked layouts are unchanged.
  The shared 128-byte push-constant layout uses incremental per-draw updates,
  following [Khronos push-constant guidance](https://docs.vulkan.org/guide/latest/push_constants.html).

This fades between **BTO variants**, including clipped replacements. It does not
implement paired detailed-tree/BTT fades, per-reference ownership, or hidden GPU
cell uploads. Existing near-tile selection policy is retained. The upload timeline
poll protects replacement readiness; the existing global upload dependency remains
necessary and can still delay a frame. A single large tile/cell is not yet sliced.

### Evidence

Local evidence: `captures/preload-handoff/`. Native rendering/presentation remains
1536×864 in a 768×432 logical window. No texture or visible-range reduction.

Daylight, identical 1799 measured intervals after 240 warmup frames:

| Version | Mean FPS | p95 ms | p99 ms | Max ms |
|---|---:|---:|---:|---:|
| Previous moving-stability baseline | 59.79 | 29.26 | 30.95 | 700.81 |
| Retained tiles, before fade/headroom | 62.70 | 18.30 | 23.11 | 248.51 |
| Initial fade + headroom | 60.28 | 17.87 | 20.95 | 215.10 |
| Final: intersecting shadow refreshes | 60.93 | 19.27 | 22.67 | 208.76 |

Final four-condition timing run (`final-timings/`), each 1799 measured intervals:

| Look | Mean FPS | p95 ms | p99 ms | Max ms | Intervals >33.33 ms |
|---|---:|---:|---:|---:|---:|
| Day | 60.93 | 19.27 | 22.67 | 208.76 | 7 |
| Overcast | 61.04 | 19.38 | 22.70 | 211.56 | 7 |
| Dusk | 62.04 | 18.74 | 22.43 | 211.62 | 6 |
| Night | 63.83 | 18.61 | 21.81 | 203.18 | 8 |

These are wall-clock frame intervals, not isolated GPU times. The initial fade
implementation regressed dusk/night averages to 56.7/56.6 fps. Limiting directional
shadow invalidation to intersecting tile bounds and branching around disabled
shader fades recovered those averages; the earlier measurements are retained in
`fade-timings/`. The final build and shader hashes match all four final runs.

Large cell staging and arena relocation remain visible. No sustained 16.67 ms
floor is claimed; the 203–212 ms tails still require incremental upload work.

The final allocated geometry arenas fell from 4,821,986 vertices / 12,883,794
indices to 2,359,980 / 7,625,574. At 64-byte main + 36-byte shadow vertices and
4-byte indices this is about 509 MiB → 254 MiB. This is **arena capacity**, not
peak process or total VRAM usage; temporary retired buffers, textures and CPU
preparation are additional. Final night capacity was 283 MiB; asynchronous
allocation order changes fragmentation.

Optimized build passes. CTest remains 35/39, with the same four pre-existing
failures: Skyrim inventory, Bethesda runtime, save, and world-map cache. Synthetic
import coverage verifies large-reference retention, partial-cell clipping, source
coordinate axes, material preservation and restoration after detail eviction.
The Vulkan smoke fixture passes with synchronization validation and exercises
fade state, readiness and evict/reload.
Final daylight, overcast, dusk and night validation captured **7200 native frames**
in the small logical window, with no validation errors, synchronization hazards
or warnings, and clean renderer image shutdown. All four captures match the final
benchmark binary. Local viewer: `captures/preload-handoff/final-visual/index.html`.
Sampled route frames retain building/terrain coverage; full motion/retail acceptance
remains open. Existing actor clothing gaps are also present in the matched
pre-change day frame 900 and are unrelated to this handoff increment.

## Remaining work

1. Bound actual bytes and CPU work: worker renderer-stream conversion, incremental
   tile/cell uploads and preparation memory caps. Extend worker preparation to
   terrain/tree LOD; this increment only moves object-LOD preparation. Profile arena relocation and
   queue contention separately. The 203–212 ms tails are still unacceptable.
2. Retain immutable source tile geometry to avoid re-decoding when only clipping
   changes. Introduce reference/cell ownership groups before paired detail fades.
3. Add hidden detailed-cell GPU readiness without early gameplay/light activation,
   then implement complementary detailed-tree/BTT handoff and history rejection.
4. Validate cold-cache, reversal/teleport, long eviction/revisit and partial-city
   routes, plus effect-disabled/depth/AO comparisons and matched retail views.

The original design below remains the guide for these subsequent increments.

## Original hitch diagnosis

The final daylight traversal logged a 673.027 ms object-LOD update. Its upload
phase spent 388.955 ms in geometry upload, 36.043 ms building vertex streams, and
only 0.102 ms handling textures. These are CPU phase durations, not isolated GPU
copy timings. The surrounding import/build work accounts for additional time.
Evidence: `captures/moving-stability/final-timings/day/runtime.log`.

The previous `BethesdaApp::updateSkyrimObjectLod` rebuilt a single scene containing the entire
11×11 BTO window: 121 tiles and about 663,460 triangles. Even while the camera
remains in tile (4,-12), detailed-cell callbacks invalidate that scene. Rebuilding
and replacing the whole window turns a small handoff change into large import,
packing, allocation and upload work.

Existing mechanisms help but do not solve this:

- `CellResidencyPlanner` predicts velocity for ranking, but its candidate loop is
  limited to the current load radius. It does not preload a future outer strip.
- `CellStreamer` already builds cells on workers. Applying one result still runs
  synchronously; its per-frame count/time budget cannot split an oversized item.
- `uploadIntoBufferRange` already submits copies with timeline signals, rather
  than waiting for queue idle. Merely calling this path “asynchronous” again will
  not remove main-thread packing, allocation, staging copies or queue contention.
- Frame submission waits on the global pending upload timeline. Hidden preloads
  need explicit readiness and dependency ownership, not immediate publication.
- BTT trees are trimmed out as soon as detail residency replaces them. There is
  no complementary outgoing/incoming coverage transition.

## Implementation order

### 1. Retain individual LOD tiles and prepare replacements off-thread

Replace the monolithic BTO-window residency object with tile entries, each using
existing imported-scene chunks. Key entries by source game/worldspace, tile/tier
and import/load-order generation. Preserve shared atlas residency.

Moving one tile across an 11×11 window should normally add one 11-tile strip and
retain the 110 overlapping tiles. A detail arrival should change only affected
handoff state, not decode/re-upload all 121 tiles.

Build immutable CPU tile data and renderer-ready vertex/shadow/index streams on
workers. Keep Vulkan calls in the renderer. Split unusually large tiles into
bounded upload pages. Avoid full draw-table, water, GI or bounds rebuilds once per
page: publish one completed batch. Preserve current parent/child-worldspace and
large-reference clipping; a partially detailed city cannot lose its entire BTO.

Where clipping changes with detail residency, retain source geometry and update
small ownership/visibility ranges. Do not repeatedly re-import source assets just
to remove a few nearby shapes. BTOs can merge many objects, so retain stable
handoff groups rather than assuming a whole tile corresponds to one detailed cell.

### 2. Separate preload readiness from visible/gameplay residency

Use explicit runtime states:

`requested → preparing → staged → uploading → ready → fading → active → retiring`

Only activation triggers gameplay residency, lights, particles, collision and
navigation callbacks. Prepared/hidden chunks must not accidentally contribute to
main, shadow, reflection, GI, water or light tables. Keep these states separate
from the existing planner's resident meaning.

Predict the next boundary from position and velocity. Start with the upcoming
outer strip, rather than a larger symmetric visible ring. Choose lead time from
measured preparation/upload latency plus a margin; cap pending bytes and work.
Camera orientation can refine priority, but must not exclude off-camera assets
needed after a turn. Stopping, reversing, teleporting and changing worldspaces
must reprioritize or invalidate requests by generation. Late results cannot
activate in the wrong worldspace.

### 3. Budget actual upload work and publish only ready chunks

Reuse the current arenas, texture sharing and timeline submission path. Preallocate
staging space and geometry headroom during warmup; prevent hidden preloads from
forcing a huge arena relocation on a live frame. If contiguous arenas remain a
bottleneck, use bounded arena pages through existing draw buffer handles.

Queue incremental copies with both byte and CPU-time limits. Initial experiments
can target 1–2 ms of CPU publication work and a few MiB per frame, then adjust from
measured GPU headroom. These are tuning starting points, not guarantees. Every
large packing/allocation/copy operation must itself be bounded for a time budget
to work.

Give each pending chunk its completion dependency. Poll readiness without a host
wait; keep drawing the outgoing representation until the new one is ready. Keep
staging and descriptors alive until transfer completion, and retire old geometry
only after its last graphics use.

Do not simply remove the global upload wait: arena relocation affects already
visible data and still requires synchronization. Likewise, a copy submitted on
the graphics queue ahead of rendering still delays rendering even if its explicit
wait is removed. Interleave small uploads with frames; evaluate a separate transfer
queue only if profiling justifies its ownership/synchronization cost. No second
rendering path or generalized render graph is needed.

Synchronization follows the existing explicit barriers and timeline model; check
transfer-write to actual consumers and resource retirement with synchronization
validation. Reference: [Khronos synchronization guide](https://docs.vulkan.org/guide/latest/synchronization.html).

### 4. Dither the visual handoff after readiness

Start with trees, where owning cells and near/distant representations are clear.
Use a typed runtime transition parameter and stable ownership seed, not another
meaning overloaded onto a packed material field. Keep both representations alive
briefly; an initial 0.4-second smooth transition is a tuning candidate.

For compatible opaque/alpha-tested surfaces, apply complementary coverage tests
using the same threshold and transition value. Preserve the authored leaf alpha
test, material identity and lighting. Share the policy across main color,
normal/depth, shadows and reflections; otherwise AO halos and shadow/reflection
popping can replace the original artifact. Validate the mask with TAA, including
history rejection at newly revealed surfaces. Avoid an unrelated random mask on
each frame and avoid transparent blending of solid terrain/buildings.

BTOs need cell/reference ownership groups before selective fades. A whole-tile
fade when only some detailed cells are ready creates holes. Near and distant
silhouettes differ, so complementary tests alone do not guarantee coverage at
all silhouette edges; inspect overlap and keep outgoing coverage where needed.
Water and blended particle materials need their own existing transparency rules,
not the opaque dither policy.

If a request is late, hold the old representation. If direction reverses, continue
or reverse the transition without resetting it discontinuously. Release outgoing
resources only after the fade and the final GPU use both complete.

## Acceptance

1. Repeat the identical native-DPI road route in all four conditions, with warm
   and cold caches. Add stop-at-boundary, reverse/revisit, rapid turns and eviction
   routes. Keep performance runs separate from image readback/validation runs.
2. Log per-frame CPU preparation/publication, uploaded bytes, allocation/growth,
   upload readiness latency, resident/prefetched bytes, wasted work and GPU time.
   Retain maxima and p95/p99. Removal of the 600–700 ms LOD rebuild is the first
   milestone; stable 16.67 ms frame delivery is the final performance criterion.
3. Test partial child-city residency, missing tiles/dependencies, late/cancelled
   results and worldspace changes. Synthetic fixtures must verify that hidden
   preparation never activates gameplay or lighting early.
4. Compare transition-enabled/disabled sequences, depth/AO, shadows and reflections.
   Check foliage sparkle, trails, holes, duplicate geometry and lighting changes.
5. Keep cooked scenes compatible: transition/readiness state is runtime-only.
   Version caches only if preserved import semantics/layouts change. Run existing
   tests and Vulkan synchronization validation, separating the four known CTest
   failures from new failures.

First implementation deliverable: retained per-tile object LOD and bounded,
worker-prepared uploads, verified by the same route. Add dither once readiness is
correct; a fade cannot conceal a frozen frame.
