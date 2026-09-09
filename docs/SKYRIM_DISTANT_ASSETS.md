# Skyrim distant-asset correctness

September 8, 2026. Keep the current detailed-cell range at the user's request.
The small window continues to render at native display DPI.

## Corrections

- Tree LOD now uses the standard two perpendicular, double-sided cards, preserving
  BTT placement/rotation/scale and LST dimensions/atlas rectangles. The previous
  three-plane approximation changed crown density and silhouette.
- Legacy RGBA tree atlases are explicitly color (sRGB), rather than linear data.
  The installed Tamriel tree atlas uses this otherwise ambiguous DDS layout.
  The distant object color resolver also handles untagged RGBA albedo explicitly;
  the installed Tamriel object atlas already decodes as sRGB.
- A dedicated non-PBR tree-LOD material flag supplies a consistent crown-lighting
  approximation: hemispherical ambient and a fixed diffuse directional response,
  preserving weather, shadows, and fog without relighting each billboard plane
  as a solid wall. AO normals agree with the lighting. Atlas trees no longer
  receive the invented whole-crown wind deformation.
- Distant NIF geometry retains authored vertex RGB as well as alpha. Previously
  the LOD conversion discarded RGB before runtime packing.

No packed vertex or cooked-scene stride changed. The material marker uses an
existing reserved mesh-part bit and a non-PBR runtime flag. Old scenes remain
loadable. Streamed LOD is regenerated at startup, outside generated detailed-cell
caches; old cooked LOD needs recooking to acquire the new geometry and metadata.

## Authored resolution audit

| Installed asset | Source resolution | Mips |
| --- | ---: | ---: |
| Tamriel tree atlas | 1024×1024 | 10 |
| Tamriel object atlas | 2048×2048 | 11 |
| Tamriel terrain tile 4.4.-12 | 256×256 | 8 |

These textures already reach the LOD importer at their original resolution with
their mip chains. The distant terrain softness is partly in the authored four-cell
atlas itself. No replacement textures, sharpening bias, extra foliage, or larger
detailed-cell range were introduced. Closer matching of terrain LOD blending and
retail color policy remains useful follow-up work.

## Validation and evidence

Optimized runtime and shaders built successfully. Focused import, scene, and
material tests passed. Synthetic LST/BTT/RGBA-DDS fixtures verify the two-plane
layout, perpendicular orientation, full atlas mip retention, sRGB role, absence
of invented wind, and material identity through packing and cooked runtime reload.
Daylight Riverwood passed Vulkan validation with no reported errors or image leaks.

Local before/after captures: `captures/riverwood-distant-before.*` and
`captures/riverwood-distant-after.*`; overcast capture and timings:
`captures/riverwood-distant-overcast.*`. Native 1536×864, detailed radius 1, LOD
radius 5. The same 5014 trees use 20056 triangles instead of 30084. Overcast run:
43.25 fps average, frame p50 23.27 ms, p95 24.63 ms. This is not an isolated
feature-cost comparison or a claim that the 60-fps target has been reached.

The standard billboard arrangement and lack of normal maps are documented by
[DynDOLOD's tree LOD documentation](https://dyndolod.info/Help/Tree-LOD).
[xEdit's LOD definitions](https://raw.githubusercontent.com/TES5Edit/TES5Edit/dev-4.1.6/Core/wbLOD.pas)
describe the authored records. The open-source
[DistantTree shader](https://raw.githubusercontent.com/doodlum/skyrim-community-shaders/main/package/Shaders/DistantTree.hlsl)
is a useful comparison for diffuse/ambient lighting without card-normal Lambert
lighting; it includes community enhancements and is not a retail reference.
Exact Skyrim SE visual parity still requires matched retail captures.
