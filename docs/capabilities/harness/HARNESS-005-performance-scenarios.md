# HARNESS-005: Performance scenarios

Status: Implemented

## Goal

Make performance a repeatable local regression gate. `./iridius-bench balmora`
runs an optimized, fixed-route Morrowind traversal and emits machine-readable
measurements and an actionable budget verdict.

## Required behavior

- A versioned route, fixed framebuffer extent, simulation step, warmup and
  measured frame count define the scenario.
- Collect CPU work time and wall frame interval, GPU timestamp duration, draw
  calls, triangle count, peak process RSS, and CPU heap allocation calls per
  frame. Report median and p95 for frame time metrics.
- Run allocation counting separately from timing so its overhead cannot change
  the CPU/GPU result. Include allocations from all engine threads.
- Keep raw samples, a JSON report, route/build/hardware metadata, and failures
  for audit. Fail closed on missing samples or different hardware/resolution.
- Store a baseline and budgets in Git. The CLI exits nonzero and prints current,
  baseline, budget, and regression percent when a budget is exceeded.
- CI exercises parser and budget behavior with synthetic fixtures. Shared CI
  runners must not use their variable GPUs for the performance gate. Retail
  game data remains local and never enters the repository.

## Use

Build `linux-vcpkg-relwithdebinfo`, then run `./iridius-bench balmora` from the
repository root. The command uses the locally installed Morrowind data by
default; `--data` selects another installation. `--output` retains raw CSVs,
runtime logs, and the JSON report in a chosen directory. `--report <report.json>`
checks a saved report against a budget without replaying the route.

The scenario warms up for 240 frames, then measures 1200 route frames at
768x432. Three normal runs determine timing and RSS; three separate runs with a
glibc allocator interposer determine allocation p95 across engine threads.
The Git budget records CPU/GPU identity, driver, framebuffer extent, actual
present mode, route hash, baseline, and limits. The requested immediate mode
can fall back to FIFO when unsupported; the actual mode is part of the host
match. GPU timestamps arrive through frame-slot readback, so a sample can be
listed a few CPU frames later. Coverage below 95% invalidates the run.

Timing CSV schema 3 attributes `interval_ms` to the frame on that row: it runs
from that frame's start to the next frame's start. The writer holds one sample
until that next start; the terminal row has a blank interval. The benchmark runs
three extra frames after the measured window to close its last interval and
collect delayed GPU results before screenshot readback and exit.

`frames.csv` retains raw GPU readback events with `gpu_submission_id` identifying
the original graphics submission. `submission_id` describes the current CPU
frame (zero means no submission). `attributed-frames.csv` contains the measured
window with GPU results joined to their originating frame and a
`gpu_readback_frame` column. GPU coverage is computed after this join, so warmup
results cannot substitute for missing measured samples. The reader rejects old
CSV schemas, duplicate/nonconsecutive frame IDs, invalid timings, inconsistent
submission outcomes, and ambiguous GPU events. Existing saved JSON reports can
still be checked with `--report`; rerun the runtime to obtain attribution data.

Benchmark rows additionally expose `wait_frame_slot_ms`, `wait_acquire_ms`,
`wait_present_ms`, `wait_transfer_ms`, `queued_frames`, `render_attempted`, and
`present_accepted`. Acceptance means the presentation request was accepted by
Vulkan; it is not evidence that an image reached the display. `cpu_ms` and
`render_ms` remain elapsed durations including waits. `cpu_work_ms` and
`render_work_ms` subtract the four measured waits; other uninstrumented blocking
and OS preemption can still appear in these residual durations. Reports include
median/p95/p99/max/stddev for each phase and counts of skipped submissions and
unaccepted presents. The strict 120 Hz gate rejects either outcome.

Bethesda timing rows further split the tick into session, actors, and streaming.
Streaming records the streamer's apply, renderer upload, resident callbacks,
eviction, renderer removal, and eviction callbacks. Eviction callbacks separate
collision, navigation, and door cleanup. Resident callbacks expose
runtime object registration, gameplay cell I/O, compilation, anchor publication,
living-world upsert, prepared physics insertion, lightweight collision, and
navigation. `resident_cells` counts cell arrivals on that frame. Nested values
are subsets: for example `gameplay_anchor_ms` is within `gameplay_publish_ms`,
which is within `gameplay_cell_ms`; do not sum them into `tick_ms`. The report
keeps distributions for each stage and the ten largest tick samples with their
stage values. These timers measure main-thread wall time, including blocking
inside a stage.

The initial budget was calibrated on Intel Core Ultra 7 258V / Intel LNL
graphics with FIFO presentation. It allows 25% above the measured p95 timing
and allocation baselines and 15% above peak RSS. A different host must run
`./iridius-bench balmora --calibrate` and review its resulting budget before
using the gate. Shared GitHub CI runs only the synthetic reporter and heap
counter tests and explicitly reports that the performance gate was not run.

## Verification

1. Synthetic reporter tests cover pass, budget fail, missing metric, and
   hardware mismatch.
2. `./iridius-bench balmora --calibrate` succeeds on local retail data with
   three optimized timing repetitions and three allocation repetitions.
3. `./iridius-bench balmora` produces a JSON report and passes its calibrated
   budgets on the same host.
4. A deliberately lowered budget fails with an actionable diagnostic.
5. Full CTest passes.

## Definition of Done

All required behavior and all five verification checks pass; only then mark
HARNESS-005 Implemented in `docs/PARITY.md`.
