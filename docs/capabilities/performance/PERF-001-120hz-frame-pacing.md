# PERF-001: 120 Hz Frame Pacing

Status: Partial

## Goal

Iridius should provide smooth, consistently paced gameplay at 120 Hz on the project's reference hardware and representative Morrowind workloads.

Performance is defined in terms of frame time and frame-time stability, not average FPS alone.

At 120 Hz, the total frame budget is approximately:

    8.33 ms/frame

Sustained average frame rate must not conceal periodic stalls, microstutter, or frame-pacing instability.

## Required Behavior

### PERF-001-A: 120 Hz target

Under the project's defined reference scene and reference hardware configuration, Iridius should sustain a 120 Hz update/render cadence.

Steady-state frame time should normally remain within the 8.33 ms frame budget.

### PERF-001-B: Stable frame pacing

Frame times should remain consistent during steady-state execution.

Performance verification must record frame-time distributions rather than only average FPS.

At minimum report:

- median frame time
- p95 frame time
- p99 frame time
- maximum frame time
- frame-time standard deviation or equivalent jitter metric

### PERF-001-C: No microstutter

After scene warm-up, routine engine operation must not produce recurring frame-time spikes that are perceptible as microstutter.

For the reference benchmark, the initial target is:

- median frame time <= 8.33 ms
- p95 frame time <= 8.33 ms
- p99 frame time <= 10 ms
- no routine frame > 16.67 ms
- p99 - median frame time <= 2 ms

Exceptional events such as initial application startup may be measured separately.

These thresholds may be refined as representative workloads and reference hardware are established.

### PERF-001-D: No hidden synchronous stalls

Steady-state gameplay systems should not introduce avoidable synchronous work capable of causing frame-time spikes.

Examples that require particular scrutiny include:

- synchronous asset loading
- shader compilation
- terrain generation
- physics-mesh construction
- resource uploads
- world-cell activation
- filesystem access
- large memory allocations
- blocking synchronization with worker threads or GPU work

Expensive work required by these systems should be scheduled or amortized appropriately where practical.

### PERF-001-E: Benchmark reproducibility

Performance tests must use deterministic or sufficiently controlled scenarios so regressions can be distinguished from ordinary run-to-run variance.

Benchmark results should include:

- scenario identifier
- build configuration
- frame count
- warm-up period
- relevant hardware information
- frame-time distribution

Release or optimized builds should be used for performance acceptance testing.

## Stutter Regression Testing

Representative automated scenarios should record individual frame times.

A performance regression should fail verification when a change:

- causes the agreed frame-time thresholds to be exceeded
- introduces recurring long-frame spikes
- materially increases frame-time variance
- introduces a new synchronous stall during normal gameplay

Average FPS alone must not be used as the acceptance criterion.

## Definition of Done

PERF-001 is Implemented when:

- a repeatable frame-time benchmark exists
- Iridius can record per-frame timing data
- median, p95, p99, maximum, and jitter statistics are reported
- a representative scene satisfies the project's 120 Hz frame budget
- automated verification can detect an intentional frame-time stall
- performance regression thresholds are enforced on suitable reference hardware

## Current verification (Intel Core Ultra 7 258V / Intel LNL, 768x432)

`./iridius-bench balmora --target-120hz` records median, p95, p99, maximum,
standard deviation, and count of frames over 16.67 ms, then enforces the
thresholds above. Its CTest fixture injects recurring 25 ms frame stalls into
raw samples and confirms the strict gate fails. Optimized runs use the fixed
Balmora route and retain CSV/report evidence locally.

Morrowind generated navigation previously took 15–25 ms on the main thread at
each cell arrival. Cell workers now prepare the same navigation data before
publication; resident-cell navigation publication measures about 0.01 ms in
the local probe. Existing path behavior remains covered by navigation tests.

The optimized three-run Balmora measurement after that change reported frame
interval median 8.329 ms, p95 8.804 ms, p99 15.372 ms, maximum 35.958 ms,
and 22 frames over 16.67 ms across all repetitions. The 120 Hz gate therefore
fails. Remaining work includes amortizing streamed renderer vertex conversion
and collision publication (observed 4–14 ms per arriving cell), and verifying
presentation jitter on a fixed reference display. The local Vulkan surface
exposes FIFO presentation even when immediate mode is requested. Keep this
capability Partial until the strict gate and full suite pass together.

## Completion plan — streaming and presentation are P0

This is an investigation and implementation plan, not evidence of a completed
optimization. It builds on HARNESS-005 and must preserve WORLD-003 residency and
persistence behavior, PHYS-001 static collision, and PHYS-002 terrain collision.
Keep the existing imported-scene path and serialized scene/cache layouts.

### Evidence reviewed on 2026-10-02

Reanalysis of the three retained optimized runs in
`/tmp/iridius-perf-001-nav/timing-*/frames.csv` gives:

| Metric | Run 0 | Run 1 | Run 2 |
|---|---:|---:|---:|
| Frame interval p95 (ms) | 8.795 | 8.804 | 8.827 |
| Frame interval p99 (ms) | 13.879 | 15.519 | 15.372 |
| Maximum frame interval (ms) | 29.321 | 35.958 | 34.683 |
| Frames >16.67 ms | 7 | 8 | 7 |
| Tick p99 (ms) | 7.867 | 9.807 | 9.409 |
| GPU p95 (ms) | 5.988 | 5.928 | 5.903 |

Fourteen of the 22 long intervals have a preceding tick of 15.7–28.3 ms;
the other eight have a preceding render call of 16.7–35.9 ms. Tick spikes
recur around route frames 556, 954–956, and 1341–1355. This supports tackling
cell arrival work first while investigating render/wait outliers separately.
It does not establish that the eight render outliers are all WSI stalls.
GPU samples are delayed readbacks and cannot yet be paired with those frames.

The saved aggregate `report.json` predates the reporter's worst-maximum and
summed-long-frame handling: it reports 34.683 ms and eight long frames. Use
the raw runs above for this diagnosis and regenerate reports for acceptance.
No new live GPU measurement was made for this investigation.

`/tmp/iridius-chunk-probe.log` provides phase evidence, not acceptance timings
(its short diagnostic run has no warmup). For cell (-5,0), renderer addition
took 9.30 ms, followed by callbacks reporting 5.03 ms under `physics`, 1.62 ms
under `collision`, and 0.014 ms under `navigation`. Within renderer upload,
vertex stream allocation/conversion took 6.71 ms. Crucially, `physics` measures
all of `cacheBethesdaCollisionCell`, including runtime object registration,
gameplay sidecar loading/compilation/writing, and gameplay residency updates;
it is not a measurement of Jolt insertion alone.

### Frame attribution and wait timings — implemented 2026-10-02

CSV schema 2 now pairs each start-to-start interval with the CPU frame that
produced it. Terminal intervals remain missing. Graphics submission IDs link
deferred GPU readbacks to their original frame; `attributed-frames.csv` retains
the measured window after that join. The benchmark collects three trailing
frames outside the measured window for interval completion and GPU readback.

Frame-slot, acquire, present, and transfer waits are exported, including on
renderer early returns. Reports expose tick/render and residual CPU/render work
distributions, plus skipped-submission and rejected-presentation counts. Missing
GPU data is not repeated, and the reader rejects ambiguous or invalid records.
See HARNESS-005 for the complete timing contract.

Verification: the optimized runtime and smoke target built; engine statistics
tests and all ten Python reporter tests passed. The Vulkan smoke test passed
with validation requested, including checks that GPU samples refer to prior
submissions. A single optimized Balmora probe retained in
`/tmp/iridius-perf-attribution` yielded 1200 attributed frames, 100% GPU coverage,
and zero skipped submissions or unaccepted presents. Frame 342 took 32.104 ms:
31.656 ms was acquire wait, with 0.358 ms residual render work. Separately,
frame 956 took 23.937 ms with 21.990 ms in tick. The run's p95 was 8.823 ms and
p99 13.999 ms, so PERF-001 remains Partial. This instrumentation does not fix
either source of jitter, and one probe is not the three-run acceptance gate.

### Remaining attribution work

Historical unversioned CSV files require attributing row N's interval to N-1;
do not apply that correction to schema 2 files.

Schema 3 adds frame-aligned tick stages for session, actors, streaming, cell
arrival, gameplay publication, collision, navigation, and eviction. An optimized
1200-frame Balmora diagnostic run is recorded in PERF-002: a 21.20 ms tick
included 13.31 ms renderer chunk addition and 4.32 ms living-world upsert;
another frame spent 11.32 ms removing a navigation cell. The CSV and report
retain these timings for prioritizing worker preparation and bounded cleanup.

The next instrumentation work is:

- Record cell coordinates, publication stage, stage duration, uploaded bytes,
  pending work, and retirement cost without synchronous per-cell logging in
  acceptance runs. Further split renderer chunk addition and navigation
  retirement where the new stage timings show large costs.
- Preserve individual presentation feedback records keyed by present ID where
  supported. Missing feedback must remain unavailable, not zero or a repeated
  last value. `actualPresentDeltaMs` currently means actual minus earliest
  presentation time; it is not an inter-presentation interval.

Existing regression tests cover synthetic tick/wait stalls, delayed GPU results,
warmup/endpoint boundaries, and skipped submissions. Extend these as cell and
presentation event attribution is added.

### P0: remove whole-cell work from the frame deadline

Relevant code: `CellStreamer::applyCompletedLoads` in
`src/import/bethesda/cell_streamer.cc`, `uploadImportedSceneInternal` in
`src/render/backend/vulkan/chunk_upload.cc`, and the preparation/residency
callbacks in `src/games/bethesda/bethesda_app.cc`.

The current 6 ms apply limit is checked only between cells and always permits
one complete cell. Reducing the cell count or that limit cannot bound one
oversized arrival. Implement resumable preparation and publication:

1. Move gameplay sidecar reads, cache-miss compilation, and atomic cache writes
   out of `upsertMorrowindGameplayCell` on the frame thread. Prepare against an
   immutable content snapshot with its fingerprint and cell generation. Keep
   mutable runtime object/script publication on the owning thread; apply current
   saved overrides there. Prepare the lightweight `CollisionWorld` terrain and
   triangle buckets on workers too; Jolt shapes and generated navigation already
   have worker preparation paths.
2. Split renderer CPU preparation from device publication. Perform vertex normal
   and color packing, shadow stream construction, and large vector allocation
   off the frame thread. Bindless/material remaps require renderer-owned slots:
   either give a preparation job an immutable mapping with pinned lifetime or
   apply bounded remap batches after pure preparation. Keep Vulkan calls inside
   `src/render/`. The current large-cell thread fan-out still joins inside upload;
   it is synchronous, and ordinary Balmora cells are below its 200k threshold.
3. Add a small explicit pending-cell state machine: prepared, uploading, ready
   to publish, resident, or cancelled. Limit both CPU work and upload bytes per
   frame, resuming oversized cells over several frames. Start with a measured
   1 ms CPU publication allowance as an engineering target, then tune using
   actual headroom; it is not a replacement for the full frame acceptance gate.
   Defer on exhausted upload capacity rather than waiting for a free slot.
4. Commit rendering, collision, navigation, and gameplay residency coherently.
   `markLoadFinished` currently precedes renderer upload and callbacks; staged
   work needs a distinct pending state so an incomplete cell is not treated as
   playable. Preserve old resident resources until a replacement is ready.
   Recheck cancellation and visibility generations before publication. Requeue
   stale collision preparation instead of synchronously rebuilding a mesh when
   enable state changed during preparation.
5. Include cleanup in the budget: scene destruction, slot releases, retired GPU
   resources, water/light table rebuilds, and broad-phase optimization. The latter
   currently runs when streaming becomes idle, which is still inside gameplay.
   Bound pending CPU/GPU memory and prioritize nearby collision so amortization
   does not turn a hitch into missing floors or an ever-growing queue.

Verification must cover oversized cells making progress across frames, cache hit
and miss, exhausted upload capacity, teleport/cancel during upload, eviction and
revisit with persistent state, visibility changes during preparation, failed
publication cleanup, and unchanged collision/navigation results. Reuse residency,
physics, navigation, bindless, GPU-arena, and Vulkan smoke coverage. Confirm a
streamed arrival adds no synchronous cache I/O or worker join to the frame path.

### P0: establish and fix presentation pacing

The reference log shows two graphics frames in flight, three swapchain images,
FIFO presentation, and no `VK_GOOGLE_display_timing`. Requesting immediate mode
does not establish an uncapped control on this surface. The existing graphics
timeline throttle measures unfinished GPU submissions, not images waiting for
presentation. Display scheduling is inactive without its timing extension.

1. Record the actual display refresh, monitor, WSI backend, compositor/session,
   VRR state, supported and selected present modes, image count, and available
   presentation feedback. Pin a reference display configuration for acceptance.
   Keep the calibrated 768x432 benchmark extent and optimized build.
2. Compare a stationary resident scene, the same resident camera motion, and
   the streaming route on that display. Keep route speed, scene quality, and
   sampling consistent. Use immediate/mailbox only where actually supported as
   diagnostic controls; retain the intended presentation mode for acceptance.
3. Use wait attribution to decide the fix. Frame-slot/transfer stalls require
   GPU/upload scheduling work; acquire/present stalls with stable CPU/GPU work
   require WSI queue/pacing investigation. Account for timeouts and skipped
   submissions rather than treating every main-loop tick as a displayed frame.
4. If queue depth is implicated, evaluate capability-gated present IDs and
   `VK_KHR_present_wait2` (with the required device feature, surface support,
   swapchain flag, and `VK_KHR_present_id2`); use the older present-wait path
   only where supported and needed. Bound waits and handle timeout, resize,
   out-of-date swapchains, minimization, and recreation. Preserve a FIFO fallback.
   Present completion controls outstanding presents; it does not supply precise
   display timestamps. Use actual presentation feedback or platform evidence to
   verify display cadence separately from CPU loop intervals.

Vulkan references: [presentation modes](https://docs.vulkan.org/refpages/latest/refpages/source/VkPresentModeKHR.html),
[present wait 2](https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_present_wait2.html),
and [actual presentation timing](https://docs.vulkan.org/refpages/latest/refpages/source/VkPastPresentationTimingGOOGLE.html).

### Closing the capability

- Keep the thresholds above enforced. Document that 1000/120 is 8.333333... ms:
  the literal 8.33 ms gate is slightly stricter than an exact 120 Hz refresh.
  Resolve units/rounding explicitly before final display acceptance; do not add
  a broad jitter tolerance to conceal the current 8.8 ms p95 or long frames.
- Retain per-run distributions and require every acceptance repetition to pass.
  Current `aggregate` takes medians of run p95/p99 values, which can hide one
  failing repetition. Add a fixture with one failing run and two passing runs,
  as well as the existing recurring-stall fixture. Reject incomplete, non-finite,
  or mismatched timing/configuration data.
- Run three optimized timing repetitions and separate allocation repetitions;
  retain raw frames, phase data, reference display metadata, binary/build identity,
  cache state, and report. Run diagnostic cold-cache cases separately without
  deleting the user's caches. Streaming stays enabled in the acceptance route.
- Run focused regression coverage, Vulkan validation smoke, and the full CTest
  suite. Enforce `./iridius-bench balmora --target-120hz` on a suitable reference
  machine; shared CI continues to exercise synthetic regressions only.
- Change this capability and `docs/PARITY.md` to Implemented only after all
  acceptance criteria pass. Initial investigation verification: all seven existing
  reporter tests passed; rechecking the retained report correctly failed the
  strict 120 Hz gate. Subsequent instrumentation verification is recorded above.
