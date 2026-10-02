# PERF-002: Multithreaded Game Updates

Status: Partial

## Goal

Keep Bethesda gameplay responsive while independent simulation and cell work run
concurrently. Move measured expensive preparation off the frame thread, and
publish results in a bounded, deterministic order. This capability covers
Morrowind, Oblivion, Fallout 3/New Vegas, and Skyrim on the one imported-scene
runtime path.

## Dependencies

- HARNESS-005 supplies reproducible per-frame timing and regression reporting.
- WORLD-003 defines cell residency, cancellation, and persistent-state behavior.
- PHYS-001 and PHYS-002 define collision results that parallel preparation must
  preserve.
- `odai_core` already owns the job system; extend it only where measured work
  needs additional scheduling or diagnostics.

PERF-001 benefits from this work, but its display-pacing gate is separate from
the game-update CPU gate here.

## Initial tick-stage evidence

One optimized Balmora run with 1200 measured frames is retained locally at
`/tmp/iridius-perf-tick`. It is a diagnostic run, not the acceptance gate. Tick
p99 was 9.23 ms, maximum 21.20 ms; 100% of measured frames had attributed GPU
timings. Frame 1350 spent 19.91 ms in streaming: 13.31 ms in renderer chunk
addition and 6.41 ms in resident callbacks. Within the callback, gameplay
upsert took 4.32 ms, sidecar I/O 0.44 ms, and lightweight collision 1.27 ms.
Sidecar compilation did not run on this warm-cache frame. Frame 1340 took
14.58 ms to evict cells: renderer removal was 2.63 ms, while callbacks were
11.94 ms. Navigation removal accounted for 11.32 ms of those callbacks;
collision and door cleanup were 0.38 and 0.22 ms.

Prioritize moving render CPU preparation ahead of the frame and bounding its
owning-thread GPU publication. `LivingWorldSimulation::upsertCell` currently
recompiles schedules and rebuilds indexes for all resident cells after each
arrival; prepare an immutable replacement on a worker and publish in stable
order, or make the incremental update cheap enough for the budget. Navigation
removal erases resident cell data and can destroy a large graph on the frame
thread; detach it cheaply and retire its storage off-thread after readers are
finished. Renderer removal needs its own bounded retirement path. These are
measured priorities; worker execution and budget compliance are not yet done.

## Required behavior

1. Sample input and advance the fixed gameplay clock in a stable order.
   Independent work may run on workers from immutable snapshots. Mutable world,
   scripts, quests, inventories, actor identity, and renderer-facing state must
   have a clear owning thread and publication point. Jolt use must follow its
   documented thread-safety constraints; a worker may prepare shapes without
   mutating the live physics world concurrently.
2. Schedule at least two independent steady-state game-update jobs concurrently
   when work exists. Identify dependencies explicitly, without introducing a
   generalized render graph or dynamic engine plugin system. Workers must not
   write Vulkan objects; renderer calls remain in `src/render/` on their owning
   thread.
3. Precompute cell collision, gameplay sidecars, navigation, and render CPU data
   on workers as appropriate. Bound main-thread activation, GPU uploads,
   eviction, and cleanup by work and byte budgets. A single large cell must
   make progress across frames without monopolizing one frame. Nearby collision
   and terrain must remain available before a cell becomes playable.
4. Tag every job with its cell/object identity, content generation, and relevant
   state version. Reject obsolete work after a teleport, eviction, save/load,
   changed visibility, or worldspace transition. Cancel and retire resources
   without unbounded memory or deferred queues.
5. Preserve deterministic gameplay outcomes. Commit order is stable and does
   not depend on worker completion order. Save/load, persistent objects, actor
   behavior, dialogue, scripts, navigation, and physics contacts must match the
   single-threaded reference at equivalent fixed ticks. Keep a single-threaded
   mode for debugging and comparison.
6. Never introduce an unconditional per-frame worker join or synchronous cache
   read/write in steady-state gameplay. If a dependency is late, use the last
   valid published snapshot or defer dependent work according to an explicit
   gameplay rule; record the delay.

## Measurement and acceptance

Use the versioned Balmora route and additional representative workloads for
the other four supported games. Record frame-aligned tick, session, actor,
streaming, cell upload, callbacks, object registration, sidecar I/O/compile,
collision, navigation, and broad-phase timings. Retain raw frames, job start/end
and wait events, queue depth, memory use, and publication latency. Identify
which work is preparation, owning-thread commit, or unavoidable synchronization.

The reference CPU gate is: after warmup, tick p99 <= 2 ms, no routine tick >
8.33 ms, and no streaming cell publication or cleanup step > 2 ms in one frame.
These are CPU budgets, not a substitute for PERF-001's wall-frame and display
acceptance. A different reference workload may define a reviewed budget before
acceptance. Verify that worker scheduling improves the measured outliers; do not
infer improvement from average FPS or worker utilization alone.

## Verification

- Synthetic slow jobs and oversized cells demonstrate real overlap, progress
  across frames, bounded queues, no frame-thread wait, and stable publication.
- Vary worker count, job completion order, cache hit/miss, and cancellation at
  each stage. Compare deterministic state hashes and scene/collision results
  against single-threaded execution at fixed ticks.
- Exercise teleport, door transition, save/load, visibility changes, failed
  preparation, eviction/revisit, and player movement across a loading boundary.
- Run optimized reference routes, the full CTest suite, and Vulkan validation
  smoke. Keep retail assets and captures local. A shared CI runner uses synthetic
  correctness tests, while a fixed reference host enforces CPU budgets.

## Definition of Done

PERF-002 is Implemented only when the required behavior and verification above
pass, including measured steady-state concurrency, deterministic equivalence,
bounded publication, and the reference CPU gate. Otherwise retain Planned or
Partial status with the remaining failure documented in `docs/PARITY.md`.

## Current verification (2026-10-02)

The optimized Balmora route at `/tmp/iridius-perf-002-retire` passed the
existing HARNESS-005 budget and retained three raw timing runs. Its tick p99
was 8.56 ms, maximum 26.66 ms. A cell upload reached 19.37 ms, gameplay
upsert 5.80 ms, and renderer removal 2.54 ms. Generated navigation and
collision prepare on cell workers. Navigation eviction now detaches resident
storage for worker retirement; the owning-thread navigation phase reached
0.045 ms, down from 7.96 ms in the preceding local run. The worker pool now
releases completed job captures outside its queue mutex, with a regression
test covering a slow destructor and concurrent submission. The full 58-test
optimized CTest suite passed, as did the Vulkan validation smoke. The required
CPU gate, stable publication, bounded queues and remaining cleanup, synthetic
overlap tests, and cross-game deterministic comparison remain open.

## Verification matrix (2026-10-02 continuation)

| Check | Result |
|---|---|
| Optimized full CTest suite | Pass: 58/58 before and after the change. |
| Vulkan validation smoke | Pass before and after the change. |
| Versioned optimized Balmora route | Fail PERF-002 CPU gate. Before: tick p99 7.46 ms, maximum 22.92 ms, upload maximum 15.80 ms. After: tick p99 7.20 ms, maximum 19.28 ms, upload maximum 10.91 ms. Raw runs: `/tmp/iridius-perf-002-matrix-before` and `/tmp/iridius-perf-002-matrix-after`. |
| Synthetic overlap, oversized cell, and deterministic publication matrix | Incomplete: existing tests cover residency cancellation and worker cleanup, but these required cases have no complete harness yet. |
| Oblivion, Fallout 3/New Vegas, and Skyrim reference routes | Incomplete: no versioned PERF-002 route or CPU budget exists for these games. |

The measured upload failure was prioritized. A diagnostic 1000-frame Balmora
trace showed median vertex-stream conversion at 3.15 ms (maximum 6.34 ms),
with arena growth driving geometry-upload spikes to 16.54 ms. Normal and sRGB
color encoding now run on cell workers using a transient vector omitted from
cooked-scene serialization; rebuilding packed geometry invalidates it. The
same diagnostic trace after this change measured vertex-stream conversion at
2.19 ms median (maximum 4.45 ms). Geometry upload still reached 16.68 ms.
The single-cell publication limit and all other open gates remain unchanged.
