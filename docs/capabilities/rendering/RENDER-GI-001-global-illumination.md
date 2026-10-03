# RENDER-GI-001: Global Illumination

Status: Partial — deterministic colored bounce, occlusion, off-screen retention,
light changes, residency removal, and volume movement are verified. Authored TES3
ambient colors and compressed diffuse textures now feed GI. The required real-scene
visual acceptance and whole-frame PERF-001 gate remain open.

## Current verification (2026-10-02)

### Implementation and deterministic coverage

- The existing 64³ volume uses 32-unit cells, triangle/box coverage, linear
  RGBA/BC1/BC2/BC3 albedo, opaque local-light visibility, six bounded 256-unit
  gather directions, and one propagation pass. Default surface lighting uses
  the non-ray-traced path. Geometry follows resident chunks and is removed on
  eviction. The grid follows camera movement in bounded steps.
- TES3 `AMBI` ambient/sunlight/fog colors survive cell indexing and merged
  extraction; interior bounce does not inject exterior sun or sky.
- `tests/imported_gi_fixture.h`, invoked by the imported-scene Vulkan smoke,
  measures linear receiver RGB before exposure: red reflector
  `(0.258057, 0.115845, 0.115845)`, blue reflector
  `(0.115845, 0.115845, 0.258057)`. Changing the reflector transfers zero red
  difference to the blocked control. Turning the camera 180° retains the red
  contribution; extinguishing the light reduces the sampled bounce to zero.
  The fixture also checks chunk addition/removal, volume movement, interior-mode
  re-entry, stationary equality, and the rendered GI toggle.
- Response in this fixture is bounded to four render calls after upload
  completion or a state change, including window-system recovery. The volume
  is recomputed without temporal accumulation. This does **not** establish a
  wall-clock bound for real cell loading or screen-space history convergence.
- RelWithDebInfo full CTest: **60/60 passed**. Explicit Vulkan validation smoke
  passed with no validation errors or non-finite GI output; local log:
  `/tmp/gi-fixture.log`. CPU policy tests cover DDS decoding, conservative wall
  coverage, anchor movement, and update scheduling; import tests cover AMBI.

### Repeatable local comparison procedure

```sh
python3 scripts/gi_probe.py --output /tmp/gi-001-fly --scene all \
  --runs 3 --warmup 180 --frames 1200
```

- Each scene has GI on/off and an indirect-only view (`ODAI_GI_VOXEL_DEBUG=6`).
  Hour 14, clear weather, no HUD, fixed exposure: 8 indoors, 1 outdoors.
- Stationary cameras explicitly use `ODAI_FNV_FLY=1`, zero benchmark speed and
  turn. Benchmark ground snapping now respects fly mode; interior startup
  honors camera overrides. The probe rejects changed startup/screenshot poses.
  Earlier ground-snapped interior comparisons are superseded.
- Engine-space eye poses (position; yaw, pitch in degrees): Mages Guild
  `(-35.032,-575.067,828.033); 0,-8`; Fighters Guild
  `(639.329,-382.082,-127.586); 0,-8`; Balmora exterior
  `(-19920,300,12960); -90,-8`. Streaming uses
  `benchmarks/routes/balmora.tour` over 20 seconds at a fixed 60 Hz camera step.
- Reference system: Intel Core Ultra 7 258V / Intel Graphics LNL (Arc 140V),
  balanced platform profile, Vulkan 1.4.354, driver version 109060099.
  Logical window 768×432, native framebuffer **1152×648**, render scale 1.
  Allocated volume/occupancy/surface/reservoir resources: **170,917,888 bytes**;
  screen-space buffers are additional. Optional reservoir resources remain
  allocated even though the default GI surface mode is legacy.
- Each local run retains configuration, binary/shader hashes, runtime log,
  screenshot, frame CSV, GPU CSV, and distributions in `report.json`.
  GPU samples are joined by submission ID. The voxel timestamp span includes
  sky exposure and barriers; SSGI is added, and shared screen-depth cost is
  reported separately. This span excludes receiver shading; paired full-GPU
  distributions must also be considered. Disabled timestamp bookkeeping can
  produce a small nonzero span without a GI dispatch.
- Real game assets and captures remain local and must not be committed.

### Remaining acceptance

Real-scene source/receiver/control regions still need convincing visual proof
for all three scenes, including real traversal and enter/exit history inspection.
Stationary image differences alone do not establish source-consistent color
transfer. The full PERF-001 frame-pacing gate must pass before promotion.
CPU occupancy rebuilding on grid/residency changes is still synchronous; its
streaming cost and unused reservoir allocation need further optimization.

## Goal

Render responsive diffuse indirect light in Bethesda imported scenes. Light
bouncing from colored surfaces must visibly tint nearby surfaces in Balmora
exteriors and interiors while remaining practical on integrated GPUs such as
the Intel Arc 140V.

## Dependencies

- WORLD-001 and WORLD-002: exterior and interior cells
- WORLD-003: cell streaming and residency
- HARNESS-004: visual regression testing
- HARNESS-005: repeatable performance scenarios
- PERF-001: 120 Hz frame pacing
- Existing imported-scene materials, lights, depth, and Vulkan renderer

## Required Behavior

### RENDER-GI-001-A: Diffuse bounce and color bleed

Indirect diffuse light responds to lit surface color and illuminates nearby
receivers with the corresponding hue. The effect must be spatially plausible:
receivers behind solid occluders should not receive the same bounce as exposed
receivers. GI must preserve direct-light shadows, material color, and authored
interior lighting rather than replacing them with a uniform ambient tint.

At least one useful off-screen contribution must remain visible in the
validation scenes; a purely screen-space solution that loses all bounce when
the source leaves the camera view does not satisfy this capability.

### RENDER-GI-001-B: Exterior and interior coverage

GI works in Balmora's streamed exterior cells under daylight and in at least
two representative Balmora interiors with different lighting and materials.
Interior GI uses interior lights and ambient state without leaking exterior sky
or sun through opaque walls. Exterior lighting changes must update the bounce.
Entering or leaving an interior must not reuse incompatible GI history.

### RENDER-GI-001-C: Temporal response and stability

Lighting, camera, and residency changes update indirect light within a bounded
time without sustained ghosting, flashing, or stale color bleed. Stable views
must not shimmer conspicuously. History is invalidated or safely reprojected
when a cell, light, material, camera, or GI volume changes enough to make prior
samples invalid. The implementation must define and measure its response bound.

### RENDER-GI-001-D: Integrated-GPU path

The default GI configuration on Intel Arc 140V must use a bounded amount of GPU
work and memory. It must work without hardware ray tracing, CPU readback, or a
full rebuild of the visible world every frame. Expensive residency updates may
be amortized, but they must converge and must not leave newly visible cells
unlit indefinitely. The chosen representation may combine the existing voxel
and screen-space paths; this capability does not prescribe a new algorithm.

### RENDER-GI-001-E: Control and diagnostics

GI can be switched on and off independently of direct lighting for comparison.
Record the active GI mode, output extent, resource memory, history resets,
updated cells or probes, and per-stage GPU time. The disabled path must skip GI
work and must not affect unrelated scene state.

## Validation Scenes

Maintain reproducible camera, time-of-day, weather, and lighting settings for:

1. A Balmora exterior view with a sunlit, saturated wall or ground surface and
   a nearby neutral receiver in shade. Include adjacent streamed cells and an
   occluded control receiver.
2. A Balmora interior with a colored surface lit by an authored local light and
   a nearby neutral receiver, including an opaque occluder.
3. A second Balmora interior with different geometry or light color, to expose
   assumptions specific to one room.
4. A deterministic small fixture with known emitter, colored reflector,
   receiver, and blocker, usable without distributing game data.

Keep real game assets and captures local. Record scene identifiers and camera
transforms so the same views can be rerun. Capture GI enabled, GI disabled,
and an indirect-light-only debug view at fixed exposure and identical color
grading. Do not use auto-exposure changes as evidence of color bleed.

## Automated and Visual Verification

### GI-001: Color transfer

In the deterministic fixture, assert that enabling GI raises the appropriate
color channel on the neutral receiver relative to GI disabled. Changing the
reflector from red to blue must change the receiver's dominant indirect hue.
The occluded control must receive materially less transferred light. Compare
linear-light values before tone mapping with tolerances chosen for the fixture.

### GI-002: Balmora exterior and interiors

For each Balmora scene, retain paired images and sampled receiver regions with
GI enabled and disabled. A reviewer must be able to identify the source and
receiver, see a hue change consistent with the source surface, and confirm that
the result is not a direct light, exposure, or material-color change. Inspect
the occluded control for obvious light leaks. Require passing evidence for the
exterior and both interiors before declaring this capability Implemented.

### GI-003: Movement and transitions

Move the camera until the bounce source leaves the screen, stream neighboring
exterior cells, change a local light, and enter and exit an interior. Verify
that bounce remains plausible, converges within the documented response bound,
and leaves no persistent history trails or stale exterior lighting indoors.

### GI-004: Vulkan and regression coverage

Run a deterministic renderer smoke with Vulkan validation enabled. It must
report no validation errors or non-finite GI output. Add focused tests for GI
selection, history invalidation, and update scheduling where those policies
are implemented. Run the relevant rendering tests and full CTest suite.

## Performance Acceptance

Measure a RelWithDebInfo or Release build on an Intel Arc 140V system at the
project's 768×432 logical window with native-DPI framebuffer rendering and
render scale 1. Record the actual framebuffer extent, driver, power mode,
scene, warm-up, and at least three steady-state runs. Include exterior camera
motion and cell streaming, both interiors, and a stationary view. Report GI
stage times and total incremental GI GPU critical-path time at median, p95,
p99, and maximum, plus whole-frame time with GI on and off.

The initial acceptance target is **GI p95 <= 2.0 ms and p99 <= 3.0 ms** at the
actual reference framebuffer extent in each scene. No recurring GI update may
cause a frame over 16.67 ms after warm-up, and the GI-enabled run must remain
compatible with PERF-001's frame-pacing gate. Report transient cell-entry
costs separately; do not hide them by measuring only stationary frames. If the
reference display or system cannot meet the whole-frame PERF-001 gate for
unrelated reasons, retain the isolated GI measurements and keep this
capability Partial until the full gate can pass.

## Architectural Constraints

- Use the existing explicit imported-scene Vulkan path and explicit barriers.
- Keep Bethesda content interpretation outside renderer shaders and backend.
- Preserve streamed chunk and cooked-scene serialization compatibility unless
  a genuine data-layout change requires a versioned migration.
- Avoid a generalized render graph, per-frame CPU readback, and mandatory ray
  tracing for the integrated-GPU path.
- Do not commit or redistribute Bethesda game data or mod assets.

## Definition of Done

RENDER-GI-001 is Implemented when:

- the deterministic fixture proves colored diffuse transfer and occlusion;
- Balmora exterior and two interiors have repeatable visual evidence of
  source-consistent color bleed;
- off-screen bounce, cell streaming, lighting changes, and interior transitions
  satisfy the documented response and stability checks;
- the Arc 140V configuration passes the GI latency targets and applicable
  PERF-001 frame-pacing gate with GI enabled;
- Vulkan validation, focused rendering tests, and the full CTest suite pass;
- measurements and local capture locations are recorded; and
- `docs/PARITY.md` is updated only after all preceding criteria pass.
