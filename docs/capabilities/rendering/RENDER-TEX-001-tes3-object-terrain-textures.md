# RENDER-TEX-001: TES3 Object and Terrain Textures

Status: Implemented — format, replacement, mapping, resolution, residency,
strict harness, and Vulkan pixel acceptance pass. Local vanilla/generated
replacement comparisons and optimized measurements are recorded in
[verification evidence](../../validation/RENDER-TEX-001.md).

## Goal

Load and correctly map Morrowind (TES3) textures onto object meshes and exterior
LAND terrain through the existing imported-scene Vulkan renderer. Vanilla assets
and modded high-resolution replacements use the same material and resource path.
Replacing an image increases detail without changing its placement, orientation,
tiling scale, transparency, or terrain assignment.

This capability covers Morrowind and its expansions. Other Bethesda games are
outside its acceptance scope, but changes to shared code must preserve their
existing behavior. Support applies to valid TES3 texture assets and replacements
within documented image-format and device limits; it must not depend on specific
mod names, directories, texture dimensions, or a hand-selected texture pack.

## Dependencies

- TES3 NIF geometry, material properties, and texture references
- TES3 LAND/VTEX and LTEX records, including plugin-local texture ownership
- Profile-aware VFS, loose-file/archive precedence, and resource identity
- RENDER-TERRAIN-001: Morrowind exterior LAND geometry
- RENDER-TERRAIN-002: terrain assignments, blending, default texture, and mapping
- Existing imported-scene texture decoding, upload, sampling, and residency
- HARNESS-004: deterministic rendered-output verification
- HARNESS-007: Bethesda asset and record validation

RENDER-TERRAIN-002 remains the terrain-specific contract. This capability adds
shared asset-format, replacement, resolution, object-mapping, and end-to-end
verification requirements; it does not weaken its existing acceptance criteria.

## Architecture

- `src/import/bethesda/nif_scene.*` imports mesh UVs and texture/material state.
- `src/import/bethesda/asset_source.*` resolves DDS/TGA/BMP candidates through
  ordered profile providers, including legacy-extension aliases.
- `src/import/bethesda/decoded_texture_cache.*` shares the DDS/TGA/BMP decoder
  and keys entries by fresh resource identity, color role, and size policy.
- `src/import/bethesda/cell_builder.*` assembles object materials and TES3
  terrain layers; TES3 defaults to full source resolution with an explicit
  optional size ceiling.
- `src/import/dds.*` and the Vulkan upload path carry compressed image formats
  and mip chains. Imported-scene Vulkan smoke already exercises TES3 terrain.

The acceptance matrix, strict retail audits, Vulkan pixel checks, and optimized
comparison measurements are recorded in the linked verification evidence.

## Required Behavior

### TEX-001-A: Asset resolution and replacements

- Resolve object NIF references, LTEX references, and the default terrain image
  through the normal profile-aware VFS. Honor configured mod priority and
  loose-file/archive precedence for both objects and terrain.
- Handle mixed case, slash direction, and paths with or without the `textures`
  prefix consistently on case-sensitive hosts.
- Support TES3 legacy `.tga`/`.bmp` references whose shipped image is DDS, and
  actual TGA/BMP images supplied by mods. Specify and test candidate-extension
  ordering within each resource layer so a lower-priority vanilla DDS cannot
  mask a valid higher-priority replacement. Ambiguous same-layer candidates
  must follow a documented, deterministic TES3-compatible rule.
- Resolve LTEX identifiers in the correct source-plugin namespace after load
  order changes. Identical numeric indices from different plugins must not
  accidentally share one terrain texture.
- A fresh load after changing profiles or replacement assets must use the new
  bytes. Decoded, cooked, and GPU cache identity/invalidation must include the
  relevant resource provenance and quality policy. Live file hot reload is
  not required.

### TEX-001-B: Image decoding and upload

- Required color-image formats: DDS (uncompressed RGB/RGBA and BC1/DXT1,
  BC2/DXT3, BC3/DXT5, BC7), true-color TGA (uncompressed and RLE), and
  uncompressed RGB/RGBA BMP. Document supported variants and reject unsupported
  encodings with useful diagnostics rather than interpreting them as DDS.
- Preserve channel order, row orientation, alpha, dimensions, and valid mip
  chains. Diffuse color must follow one consistent sRGB/linear conversion
  policy across compressed and uncompressed assets.
- Correctly handle rectangular images, small mips below compression-block
  dimensions, and valid images without authored mips. Provide a defined mip
  policy for single-level inputs so distant surfaces do not visibly shimmer.
- Validate dimensions, byte counts, mip layout, and allocation arithmetic before
  decoding/uploading. Truncated or malformed input must not read out of bounds
  or leave a partially published texture.
- Devices lacking a required compression format must use a verified compatible
  decode/upload fallback or report the limitation explicitly. Valid assets
  must never silently render with the wrong Vulkan format.

### TEX-001-C: Authored object mapping

- Preserve TES3 NIF base-texture UV selection and coordinates, including seams,
  wrap/clamp behavior, and applicable authored texture transforms. Do not
  substitute planar/world coordinates for a mesh's authored UVs.
- Texture orientation and placement remain correct under translated, rotated,
  and scaled object instances and across multiple textured mesh parts.
- Two materials sharing an image but requiring different sampler state must
  retain their own sampling behavior after deduplication and GPU remapping.
- Preserve TES3 material/vertex tint and authored alpha-test/blend behavior.
  Verify opaque objects, cutout foliage/fences, and blended surfaces separately.
- Geometry intentionally lacking a texture remains distinguishable from a
  texture-resolution failure. Missing UVs must produce deterministic behavior
  and a useful diagnostic when a material requires them.

### TEX-001-D: Terrain mapping

- Preserve LAND/VTEX spatial assignment, LTEX resolution, default texture,
  blending, vertex tint, and stable TES3 terrain tiling scale as required by
  RENDER-TERRAIN-002.
- Increasing image resolution must not change terrain coverage, blend weights,
  dominant texture at a sample, UV scale, or cell-edge alignment.
- Verify asymmetric regions and elevation changes to detect swapped axes,
  mirroring, offsets, stretching, and per-triangle discontinuities.
- Use the same replacement-resolution, decode, quality, and residency rules as
  object textures. Do not introduce a terrain-only file lookup or decoder.

### TEX-001-E: High-resolution quality and residency

- Provide a documented full-resolution policy that retains the source top mip
  up to device limits, plus an explicit size/budget policy when requested.
  A hidden 512-pixel ceiling must not prevent high-resolution rendering.
- Support valid 2K, 4K, and 8K replacements when device dimension and memory
  limits permit. Handle rectangular replacements and single-level inputs too.
  Report any requested/actual resolution difference and its reason.
- Size reduction must preserve aspect ratio, mapping, alpha, and valid mip
  layout. Images without mip chains must obey the chosen quality policy too.
- Resource upload, scene-local to GPU texture-index remapping, streamed chunk
  publication, eviction, and re-entry must preserve object and terrain bindings.
  Shared textures must not become stale or disappear while still referenced.
- Enforce device/budget limits predictably. Failure must produce an identified
  fallback or a controlled load failure without crashing or aliasing another
  material's texture. Record decoded and GPU-resident texture bytes.

### TEX-001-F: Diagnostics and normal renderer integration

- Report requested path, selected source/profile layer, actual image format,
  source/resident dimensions, mip count, and failure/fallback reason. Identify
  the affected object/material or LAND cell/LTEX reference.
- Missing/corrupt required fixture assets fail verification. A visible
  deterministic runtime fallback must not count as successful texture support.
- Objects and terrain retain the existing depth, reverse-Z, lighting, alpha,
  synchronization, and imported-scene rendering paths. Avoid game-specific
  image decoding in Vulkan code or a second material/rendering architecture.
- Preserve cooked-scene/chunk compatibility unless a genuine layout change
  requires explicit versioning and compatibility tests.

## Automated Verification

Use generated, redistributable assets and synthetic TES3 NIF/LAND/LTEX fixtures.
Expected coordinates, regions, and pixel colors must be independently specified,
not derived by calling the production mapping routine under test.

| Test | Required assertions |
|---|---|
| TEX-TEST-001: Resolution | Archive vanilla assets, higher-priority loose/mod archives, mixed-case paths, legacy-extension aliases, actual TGA/BMP overrides, conflicting candidates, and plugin-local LTEX indices select the expected bytes and provenance. |
| TEX-TEST-002: Decode | Each required image encoding preserves dimensions, channel/row order, alpha, and all mips; cover rectangular and sub-block mips, RLE/origin variants, and malformed/truncated input. |
| TEX-TEST-003: Object mapping | An asymmetric labeled/checker image maps to known NIF UV samples across seams, material boundaries, wrap/clamp modes, authored transforms, and transformed instances; shared-image sampler differences survive packing. |
| TEX-TEST-004: Terrain mapping | At least three asymmetric VTEX regions plus default terrain resolve expected LTEX assets, sample locations, blend weights, UV scale, and boundary coordinates under vanilla and replacement profiles. |
| TEX-TEST-005: Resolution policy | Low-resolution originals and generated 2K/4K/8K replacements retain mapping; verify full-resolution and reduced-size policies, top mip/byte counts, rectangular images, and inputs without mips. Exercise limit handling without allocating unbounded memory. |
| TEX-TEST-006: Cache and residency | Profile changes, replacement-byte changes followed by fresh loading, different quality/sampler policies, shared bindings, chunk eviction, and re-entry do not reuse stale bytes or incorrect GPU slots. |
| TEX-TEST-007: Rendered acceptance | Render the combined TES3 object/terrain fixture with vanilla-sized and high-resolution assets through normal import/upload/draw. Verify predetermined image regions for orientation, repeat scale, terrain blending, tint, and alpha; require zero unexpected fallbacks and zero Vulkan validation errors. |
| TEX-TEST-008: Failure paths | Missing, unsupported, corrupt, oversized, and over-budget textures produce the expected diagnostics and deterministic fallback/load failure without affecting unrelated material bindings. |

Rendered acceptance must check identifiable pixel regions or tolerant golden
images; successful upload and a non-empty framebuffer alone are insufficient.
Use controlled lighting/exposure and cameras to distinguish UV/color/alpha
failures from lighting variation. Test full-resolution detail with a close enough
camera and framebuffer to resolve it; assert resident dimensions as well.

Run focused importer, scene serialization, render policy, residency/bindless,
and Vulkan smoke checks, then the full relevant CTest suite before promotion.
Extend the smallest existing owning tests where practical. Missing GPU support
is recorded as unverified acceptance, not a passing renderer check.

## Local Real-Content Verification

Record a reproducible vanilla Morrowind scene with textured objects and LAND,
then load the same scene/camera with a high-resolution replacement profile.
Include interior objects, exterior architecture, alpha-tested vegetation, and
terrain with multiple painted textures. Record profile order, texture formats,
selected sources, source/resident sizes, renderer settings, and capture poses.
Inspect orientation, object UV seams, terrain scale/blends, alpha, and missing
textures. Use read-only OpenMW behavior as a reference when mapping is uncertain;
record the resolved expected behavior without copying its architecture.

Keep retail/mod assets and captures local. Commit only synthetic fixtures and
reproducible procedures. A particular local mod pack is evidence, not a special
case in the implementation or a dependency of automated tests.

## Performance Evidence

Use RelWithDebInfo or Release. Record load/decode/upload time, texture count,
decoded and GPU-resident bytes, draw count, and available GPU frame timings for
the same scene under vanilla-sized and high-resolution profiles. Separate cold
and cached loading. Resolution changes must not multiply geometry or draws;
no one-material-per-triangle workaround or unbounded texture duplication is
acceptable. This capability does not introduce a new frame-rate target.

## Out of Scope

- Oblivion, Fallout 3/New Vegas, and Skyrim texture parity
- Additional normal/parallax/PBR material features or new shader models
- Arbitrary image codecs or mod-specific rendering extensions
- UI icons, sky/water textures, animated texture controllers, and terrain LOD
- Live mod installation/hot reload, virtual texturing, or a new streaming system

## Definition of Done

Mark RENDER-TEX-001 Implemented only when:

- All required vanilla/replacement formats resolve, decode, and render correctly.
- Authored object mapping, sampler behavior, tint, and alpha pass acceptance.
- Terrain assignment, default texture, blending, and scale pass acceptance with
  both vanilla-sized and high-resolution assets.
- Full-resolution and explicit reduced-resolution policies are verified, with
  2K/4K/8K evidence on a capable device and deterministic limit tests.
- Cache/profile identity and streamed residency verification pass.
- TEX-TEST-001 through TEX-TEST-008 pass, including image-region assertions and
  Vulkan validation; relevant existing tests pass without weakened requirements.
- The local vanilla/replacement comparison and optimized performance baseline
  are recorded with reproducible settings and no unexpected fallback textures.
- Final diff is reviewed and `docs/PARITY.md` is updated only after verification.

If only some requirements pass, record Partial with the remaining gaps and
evidence. Existing texture plumbing or previous terrain smoke success alone
does not satisfy this capability.
