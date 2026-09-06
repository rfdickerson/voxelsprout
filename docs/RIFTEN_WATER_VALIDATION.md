# Riften water

## Authored water revision

RiftenWorld NAM2 selects `RiftenWater` (`0x000f2407`). Its PNAM is 0x77:
the use-parent-water bit is clear. The winning Update.esm override supplies
`Data\Textures\Water\DefaultWater.dds` for all three normal layers. FNAM is
zero, so the material does not use the exterior cell flowmap. The renderer
previously ignored NAM2 and used a flowmap's existence to select that texture.

Imported properties now drive the material per patch: shallow/deep/reflection
RGB bytes `(36,53,39)`, `(16,15,5)`, `(106,120,105)` are converted to linear;
opacity is 30%; above-water fog near/far is -10/80 with amount 1. Layer UV sizes
are 1522/6884/616 world units, directions 270/209/235 degrees, speeds
0.029/0.011/0.051, and amplitudes 0.8696/0.7464/0.5181. Specular powers,
brightness, Fresnel amount and reflection magnitude are also carried through.
These are read from records, not hardcoded Riften overrides.

The parser follows [xEdit's TES5 WATR and WRLD definitions](https://github.com/TES5Edit/TES5Edit/blob/dev-4.1.5/Core/wbDefinitionsTES5.pas).
GPU attributes follow the existing interleaved vertex path described by
[Vulkan vertex input](https://docs.vulkan.org/spec/latest/chapters/fxvertex.html).
CELL XCWT takes priority over WRLD NAM2; PNAM controls parent-water inheritance,
and references are remapped across the load order. Synthetic import tests cover
these selections and scene tests cover both serialization readers. Scene format
34 and cell cache revision 84 invalidate caches lacking the new material layout.

This maps authored settings into this renderer; it does not reproduce every
Bethesda water shader feature. Underwater image-space effects and displacement
simulation remain outside this revision. Flow-enabled materials retain the
existing flow animation; Riften uses its three independently moving layers.

The final image is `captures/riften-authored-water-4k.png`. Its log confirms
RiftenWater, flow disabled, the authored noise path, and fog -10/80. The render
was checked from the same camera as the preceding captures. Normal and RT shader
variants and the full optimized build succeed. All four focused import,
serialization, lighting-policy and material tests pass.
The full optimized CTest run passes 28/30. The two failures (`odai_save_tests`
and `odai_bethesda_runtime_tests`) call required setup inside `assert`, which
NDEBUG removes. Recompiling just those test translation units with assertions
enabled against the same built libraries makes both pass. The test build
configuration was not changed in this water revision.

## Earlier generic-water revision

The matched captures are `captures/riften-canals-4k.png` (before) and
`captures/riften-water-improved-4k.png` (after). Their companion JSON files
record the camera, environment, executable and shader hashes.

The water shader now projects a Snell-law refracted ray into the opaque depth
buffer, using 32 march steps and five intersection refinements. Accepted hits
must lie below the water plane and near the ray, rejecting foreground objects
and depth discontinuities. Transmission uses the resulting path length for
absorption. The surface thickness lookup uses the undisplaced pixel rather than
an unrelated world-normal XZ offset. Absorption no longer preferentially leaves
blue behind so strongly; fallback scattering is a subdued green-brown.

The 4K render was visually checked with the original camera and lighting.
Normal and RT shader variants compile; render-policy and PBR tests pass (2/2).
Screen-space transmission cannot recover hidden or off-screen geometry and uses
the fallback water colour when no valid intersection is available. The added
depth lookups have not been performance-benchmarked. Optical constants remain
shared by imported water, so these changes also affect other locations.
