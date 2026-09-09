# P1 NIF particles: Riverwood audit

## Mill mist implementation

`Effects/Ambient/FXMistMillWheel01.nif` now imports into a typed mist definition
and renders through alpha-blended billboards in the existing main scene pass.
The original SmokeParticles01 DDS uses the existing texture resolver and chunk
residency. This path does not add a fire preset or a synthetic point light.

The supported fixture uses a 300 × 50 × 300 box, approximately six births per
second, a ten-second lifetime with two-second variation, speed/radius variation,
three-color/fade controls, 600 scale samples, particle rotation, planar gravity,
and three equal axis-drag modifiers. Helper-node XYZ quadratic rotation curves
are sampled through their parent hierarchy. Placement applies the same Bethesda
Z-up to engine Y-up transform as other imported geometry.

Particles are evaluated deterministically on the CPU from birth IDs and elapsed
time, using integration steps no larger than 1/30 second. The renderer sorts
billboards back to front and issues one six-vertex draw per visible particle.
It reuses the main descriptor layouts and 48-byte push-constant range. Original
soft-depth falloff blends particles against the existing normal/depth texture;
depth testing remains enabled and depth writes disabled. No additional image,
storage buffer, compute pass, or transfer dependency is introduced.

Each emitter belongs to its streamed chunk. Eviction releases its texture
reference and emission epoch; reloading starts fresh. Cooked scenes are now
version 38, with versions 34–37 retaining neutral mist defaults. Cell-cache
version 98 invalidates generated data from the earlier import semantics.

This is a scoped mill-mist path. Turbulence, the NIF particle LOD modifier and
full retail integration equivalence remain unsupported/unverified. Particle
ordering is internal to the mist draw list, not interleaved with every other
transparent surface. CPU integration and one draw per particle favor a small
number of fixtures; batching is needed before enabling large effect populations.
`ODAI_SKYRIM_NO_MIST=1` provides an effect-disabled comparison.

Synthetic coverage checks block references, truncation, deterministic motion,
signed gravity, cooked definition round trips, streaming-loader preservation,
and version-37 compatibility. All 37 Debug CTests pass, and the optimized particle
tests pass. The synthetic Vulkan test now renders mist, evicts/reloads its chunk,
and uses the current pre-render capture API; it exits successfully with no Vulkan
validation errors. Final capture evidence is kept under `captures/mill-mist-*`.

The final overcast/night 1080p captures, daylight 4K still, and 30-frame stationary
sequence contain no Vulkan validation messages. The bank-view on/off pair shows
the plume more clearly than the close wheel view, where geometry obscures much
of the effect. These captures do not establish matched retail appearance or
moving-camera parity; the night view is too dark for useful fine-detail review.

Two optimized native-1080p runs without validation, captured at frame 180, report
rolling GPU p50/p95 of 44.61/70.25 ms with mist and 43.54/69.95 ms with mist disabled.
Streaming and short-run variability prevent attributing this difference reliably
to mist. Total GPU/process memory deltas were not measured. The implementation
adds the resident source texture and CPU definitions/particles, with no additional
render targets or GPU particle buffers. Existing optimized-suite failures in
inventory/runtime/save/world-map coverage remain separate from the passing new
optimized particle test.

The audit below describes the original gaps and the remaining P1 rollout.

The local Riverwood coverage scan contains 13 particle-bearing NIF paths across
29 references. These are candidate references, not a count of visible failures:
the report retains initially-disabled references and mixed mesh/effect models.
The coverage classifier now includes Bethesda-specific BSPSys modifiers as well
as NiPSys modifiers and particle-system blocks.

| Fixture | References | Main dependencies |
|---|---:|---|
| Effects/FXSplashSmallParticlesLong.nif | 8 | Mesh emitter, gravity, scale/color curves, subtextures |
| Effects/FXPineDroppings.nif | 6 | Cylinder emitter, rotation, gravity, color, subtextures |
| Effects/FXCreekFlatLong01.nif | 3 | Mesh emitter, scale/color curves, subtextures |
| Effects/FXWaterfallThin512x128.nif | 2 | Box emitter, drag, gravity, speed/active controllers |
| Effects/Ambient/FXMistMillWheel01.nif | 1 | Box emitter, drag, gravity, scale/color, active controller |

## Confirmed runtime gaps

The NIF reader structurally consumes many particle blocks without preserving
their semantics. CellSceneBuilder creates fire presets from model paths before
static mesh decoding. ImportedSceneParticleEmitter carries origin, radius,
lifetime, upward speed, size, tint and count, but no authored texture, emitter
shape, controller graph, blend mode, or modifier curves.

The existing main-pass particle draw accepts only Fire, clamps vertical speed
nonnegative, and uses additive procedural billboards. It is unsuitable for smoke,
falling spray, or lit foliage particles. Streaming already aggregates emitter
instances with their owning chunks; retain that ownership mechanism.

## Implementation order

1. Preserve typed particle systems, emitter references, local transforms,
   texture/material bindings and ordered modifiers. Start with box/cylinder
   emitters, lifetime/velocity and emitter timing; report unsupported blocks.
2. Support alpha-blended textured billboards, color/scale curves, rotation,
   gravity and drag in the existing imported-scene path. Use mill mist as the
   first local fixture. Preserve additive fire separately from alpha smoke.
3. Add mesh emitters and atlas controllers for creek and splash fixtures.
   Keep hidden source meshes available to emitters without drawing them.
4. Version cooked emitter data with neutral loading of old scenes and invalidate
   generated cell caches. Test controller timing, transform/axis conversion,
   malformed references and deterministic unload/reload. Compare fixed-camera
   sequences; a still cannot validate velocity direction or animation.

Weather precipitation is a separate WTHR → SPGD path. It can share particle
rendering primitives later, but NIF import alone does not fix weather rain.
The current rain overlay sampled its moving pattern with the wrong time sign.
Fullscreen UV.y increases downward with the positive-height presentation viewport;
subtracting time now makes streaks fall down, with a matching rightward slant.
Roof occlusion and world-space weather precipitation remain unsupported.

Validation: optimized runtime/shader build succeeds. The stationary storm run
produced 30 frames with Vulkan validation enabled, zero validation errors or
warnings, and clean shutdown (`captures/rain-direction-sequence`). Updated
particle coverage is retained locally at `captures/asset-coverage/particles.json`.

References: [NifTools format definitions](https://raw.githubusercontent.com/niftools/nifxml/develop/nif.xml)
and [Khronos viewport transformation guidance](https://docs.vulkan.org/guide/latest/depth.html#_viewport_transformation).

## Environmental-effect importer continuation (2026-09-09)

Removed the mill-specific requirement that every emitter contain a color modifier,
scale modifier, gravity modifier and three drag modifiers. Absent modifiers now
preserve authored initial RGBA and radius, zero force/drag and ballistic motion.
Constant-velocity particles avoid unnecessary integration steps. Zero-duration
color fades no longer hide particles at birth. Unsupported alternative color/grow
modifiers remain rejected rather than being treated as absent.

The existing cooked representation can express these defaults without a layout
change; generated cell cache version 109 rebuilds previously rejected definitions.
Synthetic tests cover initial RGBA, absent modifiers, unsupported alternatives,
birth alpha and cooked roundtrips.

The local JK+SMIM Riverwood audit is
captures/asset-coverage/environment-restored.json. It still admits only mill mist
among the 13 particle-bearing NIFs: this increment does not establish restoration
of additional Riverwood effects. Waterfall sheets remain blocked by directional
drag; creek/splash systems need mesh emission, other fixtures need atlases or
collision. Existing unsupported-asset fallbacks remain active.

The native-DPI 1536x864 capture under captures/environment-effects completed with
Vulkan synchronization validation and clean shutdown, without validation errors.
The roof obscures the plume from this camera, so it is a runtime regression check,
not visual acceptance of mist. No new GPU interfaces, passes or allocations.
