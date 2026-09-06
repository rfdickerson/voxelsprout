# Dragonsreach waterfall and mountain capture

The capture preset extends authored Skyrim BTR terrain and BTO objects from a
3-tile to a 9-tile radius (`ODAI_SKYRIM_LOD_RADIUS`, clamped to 3–12).
`ODAI_RENDER_FAR=180000` keeps those mountains in the camera frustum.
The larger ring loads 361 terrain tiles and 346 object tiles at this location.
Inherited object LOD now yields to resident child-city cells across the whole
ring, including tiles beyond the camera-nearest tile. This removes overlapping
coarse city surfaces without moving the detailed authored objects.
The texture-slot target is 4096, still bounded by the device budget: the old
1024-slot target exhausted its slots and dropped city textures in this view.

BSEffectShaderProperty now respects Effect_Lighting instead of treating all
effect textures as bright emitters. It also applies authored UV offset/scale,
vertex-alpha enable, base material alpha, and the authored grayscale palette.
Explicitly blended NIF shapes retain RGB vertex tint, which also selects the
color palette row. Water tints must not be discarded as baked ambient occlusion.
Palette color/opacity travel through the existing non-terrain vertex sidecar,
so the serialized layout does not change.
The later texture-histogram blend heuristic now preserves materials whose
transparency comes from vertex alpha or a palette; an opaque diffuse does not
imply an opaque surface. This preserves the original
waterfall meshes and placements. The cell-build cache version is 88.

References: [NifTools material layout](https://github.com/niftools/nifxml/blob/master/nif.xml)
[Community Shaders effect lookup](https://github.com/community-shaders/skyrim-community-shaders/blob/dev/package/Shaders/Effect.hlsl),
and [Vulkan descriptor limits](https://docs.vulkan.org/spec/latest/chapters/limits.html).

Validation: RelWithDebInfo runtime build passed. The synthetic NIF test exercises
lit and unlit effects with vertex alpha enabled and disabled, material opacity,
and UV offset. The NIF import, imported-scene, imported-render-policy and bindless-slot tests
passed (4/4). This is not a claim of a full validation-layer clean run.

Remaining material scope: animated UV controllers, angle-based effect falloff and
BSWaterShaderProperty mesh surfaces are not fully implemented by this change.

## Distant terrain detail

`dragonsreach-terrain-detail-4k.png` adds Skyrim's authored
`textures/terrain/noise.dds` (512x512, complete mip chain). The BTR diffuse maps
were already loaded at source resolution. The shared detail is sampled in
world coordinates at three repeats per four cells, preserving continuity across
tile boundaries. Its measured mean is removed before bounded scalar modulation,
so the existing atlas continues to supply the land-cover colors and patterns.
This improves surface detail; it does not reconstruct higher-resolution LAND
textures or change terrain geometry. No generated replacement game textures.

The native LOD-noise path is documented by the
[Community Shaders lighting implementation](https://github.com/community-shaders/skyrim-community-shaders/blob/dev/package/Shaders/Lighting.hlsl).
Our mean-centered blend is an engine adaptation, not an exact reproduction of
Skyrim's color-dependent multiplier. One additional shared texture is resident;
no bindless truncation or runtime error was reported in the capture. Runtime
build and four focused tests passed.
