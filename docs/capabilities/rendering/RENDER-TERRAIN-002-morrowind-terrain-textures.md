# RENDER-TERRAIN-002: Morrowind Terrain Textures

Status: Implemented

## Goal

Iridius renders Morrowind exterior terrain using the terrain textures assigned by Morrowind LAND data rather than a single placeholder material.

This capability builds on RENDER-TERRAIN-001 and establishes the minimum terrain-material pipeline required for recognizable Morrowind exterior environments.

The initial implementation should reproduce the terrain texture assignment and blending behavior required by Morrowind content.

## Background

Morrowind exterior terrain geometry is represented by LAND records.

Terrain surfaces are painted using Land Texture records, which reference texture assets used specifically for exterior terrain.

A given position on the terrain may be influenced by multiple neighboring terrain textures, so the renderer must support spatially varying terrain material assignment rather than treating an entire exterior cell as one material.

OpenMW may be used as a behavioral and file-format reference.

Iridius should implement terrain rendering using its existing renderer, resource system, and material architecture.

## Dependencies

- RENDER-TERRAIN-001: Render Morrowind Exterior LAND Cell
- Morrowind content/plugin loading
- VFS/resource lookup
- texture loading sufficient to load Morrowind terrain texture assets
- existing Iridius material/rendering pipeline

## Required Behavior

### RENDER-TERRAIN-002-A: Decode terrain texture assignments

Iridius can decode the terrain texture assignment information associated with a Morrowind LAND record.

The decoded representation must preserve the spatial relationship between texture assignments and the terrain surface.

The implementation must correctly resolve texture references through the Morrowind Land Texture records rather than treating texture indices as direct filesystem paths.

Morrowind-specific decoding should remain outside Vulkan rendering code.

### RENDER-TERRAIN-002-B: Resolve terrain texture assets

For every terrain texture referenced by the test LAND record, Iridius can resolve and load the corresponding texture asset through the normal VFS/resource system.

Terrain textures must not bypass normal asset resolution.

Missing or invalid terrain textures must produce useful diagnostics and a deterministic fallback rather than undefined rendering behavior.

### RENDER-TERRAIN-002-C: Spatial terrain texturing

Terrain appearance varies according to the texture assignment data in the LAND record.

Different portions of the same exterior cell may display different terrain textures.

For example, a cell containing rock, dirt, and grass assignments should render those textures in the appropriate regions rather than applying one texture to the entire cell.

### RENDER-TERRAIN-002-D: Texture blending

Transitions between neighboring terrain texture regions should reproduce the intended blended appearance of Morrowind terrain.

The renderer should not produce clearly incorrect hard polygon boundaries where the source terrain data expects interpolation between nearby terrain textures.

The exact GPU representation is an implementation decision.

Possible implementations may involve:

- blend-weight maps
- texture arrays
- material-layer indices
- generated terrain material data

The capability specification does not prescribe one of these approaches.

### RENDER-TERRAIN-002-E: Default terrain texture

Terrain regions without an explicit custom land-texture assignment must render using the expected default terrain texture behavior.

The default path should use the same VFS/resource machinery as ordinary terrain textures.

### RENDER-TERRAIN-002-F: Cell-edge consistency

Texture assignment near the boundary of an exterior cell must preserve the source terrain layout.

The implementation should not introduce coordinate shifts, transposition, mirroring, or a one-sample offset at cell boundaries.

RENDER-TERRAIN-002 does not require multi-cell streaming, but the representation must not assume that every terrain cell can be textured independently in a way that makes later neighboring-cell continuity impossible.

### RENDER-TERRAIN-002-G: Normal rendering integration

Textured terrain must continue to use the normal Iridius rendering architecture.

Terrain must remain compatible with:

- depth testing
- existing camera transforms
- reverse-Z
- existing opaque rendering
- frame-graph synchronization
- existing lighting/material conventions appropriate to the current terrain implementation

Terrain texturing must not introduce a completely separate rendering architecture.

## Initial Vertical Slice

Use a deterministic exterior LAND fixture containing at least two visibly distinct terrain texture assignments.

Prefer a fixture containing three or more terrain texture regions if such a fixture is readily available.

The scenario should:

1. load the LAND record
2. generate terrain geometry through RENDER-TERRAIN-001
3. decode the terrain texture assignments
4. resolve the corresponding Land Texture records
5. load the referenced image assets
6. render the terrain with spatially varying textures
7. capture enough output for automated verification

The selected fixture should make obvious errors such as swapped axes or incorrect texture indices easy to detect.

## Automated Verification

Verification must test the data mapping as well as final rendering.

### TERRAIN-TEX-001: Texture assignment decoding

Given a known LAND fixture:

Assert that selected known terrain locations resolve to the expected terrain texture identifiers.

Test positions should include multiple distinct texture regions.

This test should detect:

- transposed coordinates
- incorrect indexing
- off-by-one texture references
- incorrect default-texture behavior

### TERRAIN-TEX-002: Asset resolution

For every terrain texture required by the fixture:

Assert that:

- the Land Texture record resolves correctly
- its texture asset can be found through the VFS
- the asset can be loaded successfully

Missing fixture assets must cause the test to fail with a useful diagnostic.

### TERRAIN-TEX-003: Spatial mapping

Select several terrain sample positions whose expected dominant texture is known.

Assert that Iridius's generated terrain-material representation associates those locations with the correct source texture.

This test should operate below the final framebuffer where practical so that incorrect data mapping can be distinguished from shader/rendering failures.

### TERRAIN-TEX-004: Blending

Use a known boundary between two different terrain textures.

Verify that the generated terrain-material representation contains an appropriate transition between the two texture regions rather than assigning one texture uniformly across the boundary.

The exact assertion should reflect the chosen Iridius representation.

### TERRAIN-TEX-005: Default texture

Use a fixture or terrain region without an explicit terrain texture override.

Assert that it resolves to the expected default terrain material behavior.

### TERRAIN-TEX-006: Rendering smoke test

Render the deterministic terrain-texture fixture.

Assert that:

- terrain rendering completes
- all expected terrain textures are resident or successfully resolved
- no Vulkan validation errors occur
- output is non-empty
- no fallback texture is used unexpectedly

If deterministic image regression support exists, capture and compare a golden image.

## Visual Verification

If the renderer harness supports screenshot testing, maintain a golden image containing:

- multiple terrain texture regions
- at least one blended transition
- recognizable elevation variation

The image should make common failures visually obvious, including:

- all terrain using one texture
- swapped texture regions
- UV scaling errors
- mirrored terrain painting
- hard incorrect transitions
- fallback/missing textures

Golden-image verification should supplement the data-level tests rather than replace them.

## Texture Coordinates

Terrain texture coordinate generation must produce stable world-relative or terrain-relative mapping appropriate to Morrowind terrain.

The implementation should avoid:

- obvious stretching caused by terrain elevation
- per-triangle discontinuities
- texture swimming
- cell-local mapping that will necessarily create visible seams between neighboring cells

Exact UV scale should match expected Morrowind terrain behavior closely enough that source textures appear at an appropriate world scale.

## Debugging and Observability

Expose enough information to diagnose terrain-material failures.

Where practical, provide debug information for:

- LAND cell coordinate
- terrain texture identifiers referenced by the cell
- resolved texture paths
- texture-layer/index mapping
- dominant texture at a queried terrain position

A terrain texture-index or blend-weight debug visualization is desirable if it fits naturally into the existing renderer debugging infrastructure.

## Architectural Constraints

- LAND/LTEX parsing belongs in the content/world layer rather than Vulkan renderer code.
- Texture paths must resolve through the existing VFS/resource system.
- The renderer should consume an Iridius-native terrain-material representation.
- Avoid allocating one unique GPU material object for every terrain vertex or triangle.
- Avoid hard-coding a fixed set of Morrowind terrain texture names.
- Terrain texturing must leave a viable path toward rendering several adjacent terrain cells simultaneously.
- Do not couple terrain material lookup directly to a specific test fixture.
- Preserve compatibility with the existing frame graph and material architecture.

## Performance

Record a baseline for the terrain-textured fixture.

At minimum record:

- terrain draw count
- number of terrain textures/layers used
- GPU frame time if renderer benchmarking infrastructure already exists
- GPU memory attributable to terrain textures where practical

RENDER-TERRAIN-002 does not require advanced terrain batching or streaming optimization.

However, the initial architecture must not require one draw call per terrain texture assignment.

Fixture baseline (320×180 Vulkan smoke, Intel LNL): three terrain draws for
256 VTEX assignments, three resident texture layers, and 12 bytes of decoded
RGBA texels in the deliberately minimal 1×1 DDS fixtures. The smoke harness
does not report a per-scene GPU timestamp; its frame timing log covers the
initial synthetic scene before the terrain upload.

## Out of Scope

The following are intentionally deferred:

- multiple active terrain cells
- terrain-cell streaming
- distant terrain
- terrain LOD
- terrain collision
- player movement across terrain
- normal maps
- parallax terrain mapping
- PBR terrain material extensions
- dynamic terrain deformation
- procedural terrain
- grass/ground cover
- snow accumulation
- wetness
- texture streaming
- virtual texturing
- arbitrary modern terrain-layer authoring

These should be implemented as later capabilities.

## Likely Follow-Up Capabilities

- RENDER-TERRAIN-003: Adjacent Exterior Terrain Cells
- RENDER-TERRAIN-004: Terrain Cell Streaming
- RENDER-TERRAIN-005: Terrain LOD
- PHYS-TERRAIN-001: Terrain Collision
- RENDER-TERRAIN-006: Distant Terrain
- RENDER-TERRAIN-007: Enhanced Terrain Materials

## Definition of Done

RENDER-TERRAIN-002 is Implemented when:

- Morrowind terrain texture assignment data is decoded correctly
- Land Texture records resolve to the correct source texture assets
- terrain uses more than one source texture within a cell when required
- terrain transitions are blended appropriately
- default terrain texture behavior works
- known sample locations resolve to expected terrain textures
- texture mapping is not obviously transposed, mirrored, or offset
- terrain textures render through the normal Iridius pipeline
- deterministic terrain-material verification exists
- renderer smoke test passes
- no unexpected fallback textures are present
- no Vulkan validation errors are introduced
- relevant existing tests continue to pass
- docs/PARITY.md is updated to mark RENDER-TERRAIN-002 Implemented
