# Skyrim asset coverage

Profile-wide character and animation discovery is documented in
[character asset coverage](SKYRIM_CHARACTER_ASSETS.md). It extends inspection
beyond the Riverwood visual closure without claiming full runtime support.

## Ralof skin geometry validation (2026-09-10)

Retail SSE Ralof FaceGen (`0002bf9d.nif`) has 898 `MaleHeadNord`
vertices with no vertex normals. The Stormcloak cuirass also has 200
`MaleBodyStormcloakSLEEVES` vertices with no normals. The character builder
previously assigned an upward normal to these surfaces. It now derives
area-weighted smooth normals from triangles when the normal array is absent
or incomplete, retaining authored normals otherwise. These normals follow the
existing bind-space conversion and skinning path.

The importer and GPU position skinning also normalize retained positive bone
weights, matching the velocity pass. Retail probes already report unit weight
sums, so weight normalization is not identified as the cause of this defect.

Local evidence in `captures/ralof-skinning-validation/` includes before/after
normal views, corrected color and shadow views, and a retail body probe.
The optimized JK's Skyrim + SMIM run used a 768×432 logical window,
1536×864 native-DPI framebuffer and render scale 1. Face, arm and hand normals
now follow their surfaces; shadow visibility is largely unoccluded on Ralof
in this earlier view. The later skin-shading investigation found that the
shadow diagnostic returned white for skipped back-facing surfaces, so this
was not proof of unoccluded skin. That diagnostic is now corrected.
The color render remained dark. This is a geometric-normal
fallback correction, not verification of Skyrim SE model-space skin normal
textures or skin material parity. The subsequent [skin shading fix](SKYRIM_SKIN_SHADING.md)
adds animated model-space normals, authored soft lighting, and external skin specular maps.

All 45 Debug CTest tests pass, including missing-normal generation,
authored-normal preservation and non-unit skin-weight regression coverage.
Debug and RelWithDebInfo builds succeed. No serialized layouts change.

The probe now traces the Riverwood static-asset closure through archive
resolution, NIF decoding, source-material interpretation, and scene emission.
GPU consumption is explicitly unmeasured. This is the first implementation
slice of the visual-parity roadmap, not completion of its rendering milestones.

## Coverage report schema 2

The census reads each winning virtual asset through `FalloutAssetSource` and
records its provider, archive/loose path, fingerprint and byte count. It enumerates
active plugin record winners and deleted tombstones through `ContentRecordIndex`,
including plugin provenance and override counts. It does not decode every archive
file or every record body: detailed format inspection remains scoped to Riverwood.
A profile is required when the report should use configured mod/load-order rules;
the explicit Data-directory invocation is a baseline archive census.

Per-model and per-texture stages distinguish resolved bytes, decoded data,
scene-builder preservation and runtime consumption. Source lighting parameters
are compared with actual scene material definitions. Texture inventory presence
is weaker evidence than a material binding, and is labeled accordingly. GPU use
and cooked round trips are not measured by this command. Fire presets are labeled
as fallbacks; the authored mill-mist path has separate evidence.

`gaps` groups known missing NIF/particle blocks, material controllers, shader
families, model-space normals and texture slots. Each group contains occurrence
counts, unique referencing models, deduplicated reference counts, eligible counts,
and representative paths/cells/references. Unknown shader flags, controllers and
record families are enumerated as unassessed, not automatically unsupported.
Every observed block remains in the per-asset block inventory. Missing dependencies,
malformed NIF data and unsupported features have separate statuses; a boolean DDS
decode failure remains unassessed because that API cannot distinguish codecs from
malformed input.

Initially disabled placements and wholly excluded marker/hidden models do not
contribute to eligible counts. Generated grass candidates use position-based keys
because their form IDs are zero; road/building filtering may remove candidates
before scene output. Neither eligibility nor a missing feature proves an on-screen
failure. Unreferenced virtual files and deleted record winners are never counted as
visible failures. Direct plugin base references are reported; transitive effect
usage remains unassessed.

The local JSON is intentionally large because it includes the complete winning
record/virtual-path census. Its adjacent Markdown concentrates on confirmed gaps.
Support classifications are a conservative source-audit policy and should be
updated alongside importer/runtime feature changes.

## Schema 2 validation (2026-09-08)

The local Skyrim.esm + Update.esm baseline report is
`captures/asset-coverage/riverwood-v2.json` with adjacent Markdown. It resolves
172,882 virtual assets and enumerates 880,028 active record winners plus 97 deleted
tombstones. Detailed Riverwood coverage includes 195 models, 620 authored model
placements and 532 generated grass candidates. Of 1,152 candidates, 1,118 remain
initially eligible and 888 have scene-instance/effect output. All 348 grass
instances emitted by the builder match individual candidate evidence. Thirteen
models are wholly excluded markers/hidden geometry. All 19 landscape texture sets
have diffuse/normal dependency evidence (38 channels).

These corrected counts supersede the old probe's counts that collapsed generated
grass sharing form ID zero. Shader-property float controllers affect 23 eligible
placements across 11 models; shader-property color controllers affect one. Known
visual-reader gaps include 201 ARTO, 169 EFSH, 532 IPCT and 237 IPDS active records;
their transitive Riverwood use is unmeasured. No referenced model dependency is
missing in this baseline. Unknown support remains explicitly unassessed.

Both complete Debug and optimized runtime/tool builds succeed. Debug CTest passes
38/38; optimized CTest passes 34/38 with the same previously recorded inventory,
Bethesda runtime, save and world-map-cache failures. New synthetic coverage tests
pass in both configurations, covering duplicate blocks/references, marker and
disabled exclusions, distinct grass roots, texture failure states, unsupported
versus malformed NIF diagnostics, asset overrides, mixed-case loose files, plugin
overrides/deleted winners and malformed plugin headers.

The census uncovered a POSIX base loose-file case mismatch. The shared resolver
now indexes canonical base keys while preserving mod/archive precedence. Generated
cell-cache version 99 invalidates results that may have missed those assets;
cooked-scene and streamed-chunk layouts are unchanged. Capture/report files remain
local and ignored by Git. This reporting change makes no new GPU-use or visual
parity claim.

## Run

```bash
cmake --build --preset linux-vcpkg-relwithdebinfo -j6 --target odai_bethesda_probe
build-linux-relwithdebinfo/odai_bethesda_probe "$SKYRIM_DATA" \
  --asset-coverage captures/asset-coverage/riverwood.json Skyrim.esm Update.esm
```

The command writes JSON and adjacent Markdown. It inventories unique virtual
paths and winning plugin record families, then inspects Tamriel cells (4,-11),
(5,-11), (4,-10), and (5,-10). Archived duplicates resolve through the ordinary
asset source. The explicit plugin list is expanded through the existing master
resolver. Without a list, the baseline is Skyrim.esm and Update.esm.

For configured mod precedence use the profile form:

```bash
build-linux-relwithdebinfo/odai_bethesda_probe --asset-coverage \
  profile.json captures/asset-coverage/profile.json --data "$SKYRIM_DATA"
```

The report includes provider fingerprints, placement references, initial disabled
state, emitted instance counts, shader types/flags, texture slots, and particle
and controller occurrences. Known ignored families have separate unique-asset
and affected-placement counts. Hidden geometry, editor markers, and inactive
switch branches are intentional exclusions. Unreferenced archive files are not
counted as failures. Source code-page bytes are replaced only when formatting
JSON; resolution uses original bytes. Non-texture tokens are distinguished from
missing texture dependencies.

## Implemented material changes

- `NifLightingMaterial` preserves source shader type/flags, UV transforms,
  emissive/specular parameters, alpha, and texture slots for static shapes.
  Optional truncated or nonfinite parameter tails remain invalid without
  discarding otherwise valid geometry.
- Static lighting UV transforms and material opacity now reach emitted vertices.
  Opacity remains effective when the vertex-alpha flag is disabled.
- Static normal maps belong to their source shape, rather than being selected
  solely by diffuse texture identity. Existing packed texture indices carry the
  result through scene loading and texture residency.
- TXST import retains TX00 through TX07. Merged world tables resolve a TXST-only
  override against existing LTEX references, including its normal slot.
- Terrain base and four overlay normal maps now retain their LTEX/TXST identity
  through typed per-part bindings, packed vertex sidecars, streamed texture
  residency, vertex/tessellation interfaces, and tangent-space shading. Albedo
  and normals share the same blend weights. Missing overlay normals flatten the
  underlying relief instead of retaining the wrong material's normal map.
- Cooked format 37 includes normal, lighting, and cubemap metadata and still loads
  formats 34–36 with neutral defaults. Cell cache version 96 invalidates prior imports.
  Existing raw serialized vertex/part layouts and GPU pass ordering are unchanged.

## Local findings and verification

The Skyrim.esm + Update.esm Riverwood report identified 195 unique referenced
assets: 182 emit decoded static geometry and 13 contain intentional exclusions.
No referenced mesh files were missing. Thirteen assets across 29 placements
contain 16 particle-system blocks; their authored particle behavior remains a
clear implementation target. The report contains 19 landscape materials and
392 standard, eight environment-map, and two glow lighting-shader occurrences.
Occurrences count shapes, not unique material definitions.

The optimized runtime and probe build, focused import/serialization tests, and
all 35 Debug CTest tests pass. Synthetic tests exercise material tail layouts,
UV offsets, opacity, distinct normal maps sharing one diffuse texture, scene
serialization, TXST-only load-order overrides, and archive inventory.

Local captures are `captures/material-import-street.*` and
`captures/material-import-validation.*`. The Debug capture logged validation
enabled, no validation errors, and no renderer-owned image leaks. The native
1080p optimized capture's last timing window was p50 41.90 ms / p95 123.74 ms;
this includes streaming/startup history and is not an isolated material cost.
No new 4K, moving-camera, or memory-delta claim is made.

## Remaining roadmap work

Standard and glow BSLighting materials now have typed cooked records and GPU
bindings. Environment-mapped materials now sample authored cubes and masks.
The actor path supports animated model-space normals and the authored skin
soft-light/specular inputs; see [skin shading coverage](SKYRIM_SKIN_SHADING.md).
Parallax, multilayer, and other specialized shader families retain source metadata
and explicit fallback status. IMGS/IMAD now have readers and a core post-processing runtime
path; specialized channels remain documented in [image-space coverage](SKYRIM_IMAGE_SPACE.md).
Authored particle modifiers, material controllers, precipitation, remaining
vegetation controls, and EFSH/ARTO/impacts remain.

Material layout reference: [NifTools NIF definitions](https://raw.githubusercontent.com/niftools/nifxml/develop/nif.xml).

## Terrain normal validation (2026-09-07)

All 19 Riverwood landscape normal paths in the coverage fixture reach source
scene bindings. The probe reports this preservation separately from actual GPU
consumption, which it cannot measure. Synthetic tests verify distinct normals
sharing one albedo, all overlay slots, full/runtime scene reload, repacking,
invalid texture references, and format-34 neutral defaults.

Debug and optimized runtime/shader builds succeeded; all 35 Debug CTests pass.
The optimized suite retains the four previously recorded failures in inventory,
Bethesda runtime, save, and world-map cache tests (31/35 pass).

Local evidence: `captures/terrain-normal-street.*` (native 1080p),
`captures/terrain-normal-validation.*` (Debug Vulkan validation), and
`captures/terrain-normal-4k.*` (native 3840×2160 detail still). Validation recorded
no VUID/errors and no live renderer-owned images at shutdown. The 1080p run
finished with 81 resident cells, 0 missing, and 1,102 resident textures versus
1,093 in the prior capture. Its final timing window was p50 44.19 ms / p95
54.82 ms, including streaming history; this is not an isolated GPU-cost delta.
The main vertex stride is now 60 rather than 48 bytes. Both runs allocated
capacity for 6,055,184 vertices, so that buffer's additional allocation is
69.3 MiB. Total VRAM delta was not measured.

The stills retain visible haze, bright foliage, and material-response differences
from the supplied retail reference. No retail-parity claim is made. Other
weather states, a moving-camera eviction route, effect-disabled comparisons,
and total texture-memory deltas remain unverified for this change.

## P1 NIF lighting implementation (2026-09-07)

Static `BSLightingShaderProperty` now retains shader family, both flag words,
all nine texture paths and resolved 2D slots, UV transforms, material opacity,
clamp mode, refraction strength, glossiness, specular color/strength, and
emissive color/multiplier in `ImportedNifLightingMaterial`. Each mesh part owns
an index independent of diffuse identity. A packed vertex sidecar preserves
that index through cooking, repacking and streamed reload. Format 36 appends
these records and texture sampling policies; formats 34 and 35 remain readable
with neutral defaults. Cell-cache version 95 invalidates earlier imports.

The existing material SSBO now has chunk-owned slots separate from the legacy
256-entry library. Slots are released on eviction and failed upload, reused
without overwriting live chunks, and copied into the existing fence-protected
frame regions. Vertex, tessellation, and skinning output layouts agree on the
64-byte runtime vertex (the skinning path uses the absent-material sentinel).
No additional rendering path or GPU pass was introduced.

Standard/glow shading consumes authored specular color/strength and the normal
map's alpha mask. Glossiness is converted from a Blinn exponent to approximate
GGX perceptual roughness; this is an approximation, not Skyrim's exact BRDF.
Own-emission and glow-map flags control additive HDR emission; tree lighting
flags do not create emission. Missing glow textures contribute black. Albedo,
normal and glow share the already-baked UV transform. Explicit linear-data
roles and all four authored S/T clamp combinations reach texture residency and
samplers, including shadow alpha sampling. Roughness debug view uses the same
authored coefficient as shaded rendering.

The coverage report marks 402 Riverwood shape occurrences as supported standard
lighting (including environment materials' base response), out of 437 inspected
shapes; the latter includes non-lighting shader shapes. It distinguishes source
preservation and supported runtime features from GPU consumption measurements.
Cube texture paths were preserved at this milestone; the following milestone
adds their six-face upload and shading path.

Synthetic coverage exercises parser tails and wrap modes, distinct materials
sharing albedo, source and runtime serialization, texture-role/color-space
survival, invalid references and nonfinite coefficients, emissive/specular flag
policy, glossiness conversion, and slot reuse across cell eviction.

GPU layout and synchronization were checked against the Khronos
[shader memory layout](https://docs.vulkan.org/guide/latest/shader_memory_layout.html)
and [synchronization](https://docs.vulkan.org/guide/latest/synchronization.html)
guides. Local captures and final verification results are recorded below.

Final P1 verification: both Debug and optimized runtime/shader builds pass;
35/35 Debug CTests pass. Optimized CTest remains 31/35 with the same pre-existing
inventory, Bethesda runtime, save, and world-map-cache failures. The final Debug
capture enabled synchronization validation and completed with no VUID errors or
renderer-owned live-image leaks.

Local evidence is `captures/nif-lighting-{street,night,dusk,overcast,4k}.*`,
`captures/nif-lighting-validation.*`, and `captures/nif-lighting-roughness.*`.
The four lighting conditions use the same camera; the 4K still is native
3840×2160. Final daylight native-1080p timing was p50 43.65 ms / p95 51.53 ms.
This includes loading history and is not a controlled before/after GPU-cost
measurement. The scene reached 81 resident cells with zero missing cells.

Relative to the preceding terrain-normal implementation, the main vertex stride
increases from 60 to 64 bytes: +23.1 MiB at the logged capacity of 6,055,184
vertices. The two GPU material-table regions total 10 MiB (previously 16 KiB).
Additional texture memory and total VRAM delta remain unmeasured. The source
material library and per-vertex sidecars also add CPU/cooked-scene storage.

Night foliage remains dark rather than acting as an emitter. Daylight still has
strong haze and bright foliage, and night clouds remain conspicuous; those
presentation differences are not solved by material import alone. Moving-camera
cell eviction, isolated effect-disabled captures, and matched retail material
fixtures remain validation follow-ups. No retail-parity claim is made.

## P1 cubemaps and environment masks (2026-09-07)

Validation correction: earlier captures that reported no VUIDs were not
sufficient evidence of clean validation. The final rerun found the layer manifest
but initially failed to load its library. Sourcing the installed Vulkan SDK with
`--set-dep-ld` loaded the layer and exposed the failures listed below. This
supersedes earlier clean-validation claims in this document.

DDS decoding now preserves six square faces and every mip in face-major order
(+X, -X, +Y, -Y, +Z, -Z), for legacy DDS and single-cube DX10 headers using
supported codecs. Mip reduction preserves every face. Partial cubes, truncated
payloads, cube arrays, and invalid dimensions are rejected. BC6H is not added.
The ordering follows Microsoft's [DDS cube layout](https://learn.microsoft.com/en-us/windows/win32/direct3ddds/dds-file-layout-for-cubic-environment-maps).

Lighting shader family 1 retains its environment scale and resolves texture
slots 4 (cube) and 5 (linear environment mask) independently of albedo. Missing
masks fall back to normal-map alpha, or unity without a normal map. Reflection
directions convert engine coordinates back to Skyrim coordinates. Roughness
selects an authored mip; this remains an approximation of retail shading.
A valid authored cube replaces the generic ambient reflection contribution;
missing cubes retain that fallback. AO affects this indirect reflection without
attenuating direct sunlight.

The existing residency system owns cube resources, reference counts and eviction.
Uploads use cube-compatible images, six-layer copy regions and barriers, and cube
views. A separate cube binding shares the existing descriptor set and slot
allocator; capacity accounts for both descriptor arrays. No GPU pass or second
rendering path was added. Format 37 appends layer counts and environment scales,
with neutral defaults for formats 34–36. Cell-cache version 96 forces reimport.

The refreshed local report identifies eight environment-mapped shape occurrences
and four unique cubemaps in the Riverwood reference closure; all four decode.
Examples include the sawmill waterwheel, iron ore, modeled water, an iron mace,
and steel dagger/scabbard. This is not a claim about every archived cubemap.

Synthetic tests cover face/mip ordering, mip reduction, truncated and partial
cubes, DX10 cube-array rejection, full/runtime serialization, malformed cooked
payloads, and independent cube/mask material bindings. Debug CTest passes 35/35;
optimized runtime and shaders build successfully. Optimized CTest remains 31/35,
with the pre-existing inventory, Bethesda runtime, save, and world-map-cache
failures. The final capture with the validation library loaded reports geometry
arena transfer read/write hazards, missing shader Int16/Float16 feature enablement,
vertex-buffer usage errors, and screenshot/swapchain usage, acquisition, debug-label
and presentation errors. No cube image upload or cube descriptor error was found.
These broader errors remain unresolved; a baseline run has not established their
pre-existing status. Shutdown reports no renderer-owned live images.

Local evidence: `captures/cubemap-street.*`, `captures/cubemap-night.*`,
`captures/cubemap-4k.*`, and `captures/cubemap-validation-final.*`. The daylight
native-1080p timing window was p50 42.94 ms / p95 43.80 ms. This is whole-scene
GPU timing, not an isolated before/after cubemap cost. The 4K still is native
3840×2160. The material table grows from 80 to 96 bytes per entry: 12 MiB across
two frame regions, up 2 MiB. Vertex stride remains 64 bytes. Additional cube,
descriptor, and total residency memory deltas have not been measured.

Matched retail orientation/response fixtures, effect-disabled comparisons and a
moving-camera eviction sequence remain follow-ups. Existing haze, foliage and
night-cloud differences remain; no retail-parity claim is made.

## Vulkan validation corrections (2026-09-07)

The validation failures reported above are addressed in the shared Vulkan path:

- Streamed arena uploads and growth copies now establish an explicit memory
  dependency on earlier GPU reads/writes before transfer reads/writes. Queue
  submission order and timeline signals alone did not establish that dependency.
- Device selection checks and device creation enables the Int16/Float16 shader
  features declared by the compiled shaders.
- The actor rest-pose buffer includes vertex usage because the velocity pass
  consumes it through vertex input as well as storage descriptors.
- TAA no longer ends a debug label owned and already closed by the upscaler.
- Capture is requested before rendering and copies the acquired swapchain image
  before presentation. Swapchain transfer-source usage is enabled only when the
  surface supports it. Explicit image/buffer barriers order the copy and host
  readback; repeated captures reuse the buffer. Post-present access to an image
  owned by presentation has been removed.

The fixes follow Khronos [synchronization guidance](https://docs.vulkan.org/guide/latest/synchronization.html)
and [WSI ownership](https://docs.vulkan.org/guide/latest/wsi.html).
The capture helper now requests `VK_LAYER_VALIDATE_SYNC=1` and fails when the
validation library did not load or the log contains validation errors. Load the
installed SDK environment before running it (on this workstation:
`source ${HOME}/vulkan/1.4.357.0/setup-env.sh --set-dep-ld`).

Debug and optimized runtime builds pass; all 35 Debug CTests pass. Corrected
native-daylight and upscaled-night captures completed with zero Vulkan validation
errors or warnings and no renderer-owned live-image leaks:
`captures/validation-fixed-day.*` and `captures/validation-fixed-upscaled.*`.
The daylight run exercised multiple geometry-arena growth operations. Validation
runs are correctness evidence, not representative GPU performance measurements.

The optimized runtime also completed a 30-frame moving-camera sequence along the
Riverwood/Bleak Falls route with zero validation errors or warnings and clean
image teardown (`captures/validation-fixed-sequence.log` and the matching local
frame directory). Route cells were pinned for capture, so this checks repeated
readback and camera motion, not an eviction stress test. The previously reported
validation failures are no longer reproduced by these runs; other games and
unexercised rendering configurations are not covered by this result.

An explicit `VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation` request is now
honored in optimized builds too. Explicit requests fail initialization when the
layer is unavailable or cannot load; only optional Debug auto-detection may
continue without it. The final optimized sequence logs `validation=on`.
