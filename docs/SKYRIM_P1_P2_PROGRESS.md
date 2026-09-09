# P1–P2 implementation progress — 2026-09-09

The full P1–P2 roadmap remains incomplete. This increment closes additional
information-loss paths and adds runtime support; it does not establish retail
visual parity. Preserve the small 768×432 logical window, native 1536×864 framebuffer
on the tested display, and the existing detailed-cell range.

## Implemented in this increment

- **Terrain:** LTEX SNAM specular exponent and TXST DNAM flags survive winning-record
  resolution, scene cooking, streamed upload and terrain shading. Normal alpha
  controls specular coverage; No Specular Map selects uniform coverage. Authored
  layer opacity also blends roughness/coverage. Terrain model-space normals use
  the source-to-engine coordinate conversion. This does not implement model-space
  normals for arbitrary NIF shapes.
- **HDR cubemaps:** DXGI BC6H unsigned and signed formats retain compressed HDR
  face/mip data through DDS decoding, cooking and Vulkan residency. No conversion
  through RGBA8. Cube arrays remain unsupported.
- **Animated materials:** NiFlipController base, normal and glow slots reach their
  corresponding material-table bindings. Normal frames use linear texture data;
  missing frames retain the authored static normal/glow binding. Unsupported slots
  retain explicit partial-support status.
- **Particles:** typed cylinder/sphere volumes and directional emission join the
  existing box sampler. Sampling is deterministic and uniformly distributed in
  volume. Skyrim effect models are inspected by capability, with cached parsing
  and explicit fallback diagnostics. Unsupported systems keep existing fallback
  behavior. Particle LOD remains unconsumed: its presence must not remove the
  previously supported mill-mist subset. The real-data probe caught that regression
  during integration; admission was corrected and a synthetic regression added.
- **Image space:** ApplyCrossFade and RemoveCrossFade use a separate, interruptible
  weighted chain. Interrupting a transition retains its current mixture; removal
  fades toward the current base image space. Cell/session reset clears the chain.
- **Weather:** WTHR wind heading/range/speed feed the existing renderer wind input.
  The bounded temporal gust is an engine approximation. Skyrim grass/tree wind
  weights and remaining GRAS metadata are still unfinished; this input alone does
  not establish authored vegetation movement.
- **Coverage:** particle classification now inspects all referenced particle NIFs,
  rather than only the mill filename. Absent LOD blocks no longer create invented
  affected-asset counts. Runtime/GPU consumption remains explicitly unmeasured by
  the CPU probe.

## Compatibility and validation

Cooked format **40** appends terrain surface properties and reads versions 34–39
with neutral defaults. Existing packed scene vertex/draw layouts are unchanged.
The separate GPU vertex stride increases from **64 to 76 bytes**: +12 bytes per
uploaded vertex (18.75% vertex-buffer storage, not total VRAM). Total memory and
isolated performance deltas have not yet been measured. Particle payloads have an
optional versioned tail; older payloads default to box emitters. Generated cell
cache **108** invalidates intermediate import results.

Optimized runtime and Slang shaders build successfully. The final optimized CTest
run passes **39/39**. Tests cover terrain record overrides and serialization,
malformed properties, BC6H cube payloads, normal/glow flipbooks, particle volumes,
legacy payloads, deterministic sampling, image-space transitions and native
command dispatch. The four earlier optimized test failures were caused by setup
inside assertions being removed by NDEBUG. Test executables now retain assertions;
production libraries and runtime remain optimized.

The Vulkan smoke run exercised HDR texture upload, animated materials and scene
reload with synchronization validation and exited successfully. Local log:
`/tmp/smoke-p1p2-final.log`. Explicit GPU dependencies follow the existing path;
see the [Khronos synchronization guide](https://docs.vulkan.org/guide/latest/synchronization.html).

Local native-DPI captures and configuration/timing records are under
`captures/p1p2/mill-{day,night,overcast,dusk}.{png,json,log}`. These are integration
checks, not matched retail acceptance. The first three were captured before the
particle-admission correction and must be repeated for final particle signoff.
Day used 240 warmup frames; other views use 600. Short rolling measurements include
streaming, validation and some concurrent CPU probing; they are not isolated
benchmarks. Day's final GPU p50/p95 was 16.62/21.00 ms and night's 16.87/21.53 ms.
These do **not** demonstrate a sustained 60 fps floor. Foliage edges remain visibly
aliased and the unlit night mill is very dark; no visual signoff is claimed.

The refreshed CPU coverage report is `captures/p1p2/coverage-final.{json,md}`.
It uses Skyrim.esm only; counts must not be compared directly with earlier reports
using additional active plugins. Proprietary assets and evidence stay local.

## Outstanding P1–P2 work

1. Specialized NIF shader families, shape model-space normals, remaining texture
   slots and matched material/reflection response.
2. Remaining terrain channels and dropped-layer diagnosis; cube arrays if actually
   required by winning referenced assets.
3. Multi-system/mesh particles, atlases, animated birth/activation, turbulence,
   particle LOD, smoke/fire/spray fixtures, batching and transparent ordering.
4. Legacy UV and blend/manager controllers, remaining texture roles,
   falloff/refraction and actor material animation.
5. Remaining IMGS/IMAD HDR channels, radial/motion/double-vision blur, targeted/sky
   DOF and room/underwater overrides.
6. Authored world-space precipitation, roof occlusion, snow/ash, sky statics/aurora,
   RFCT, precipitation thresholds and maximum fog.
7. GRAS position/color variation, wave period and vertex-lighting flags; tree wind,
   alpha coverage and motion/LOD acceptance.
8. EFSH/ARTO/IPCT/IPDS parsing **and runtime presentation**. These remain undelivered.
9. Runtime consumption/transitive dependency evidence, matched four-condition
   retail references, moving-camera eviction/weather tests, native 1080p GPU and
   total memory deltas, 4K detail stills and other-game visual regressions.
