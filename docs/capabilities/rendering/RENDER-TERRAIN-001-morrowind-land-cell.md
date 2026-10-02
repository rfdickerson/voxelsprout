# RENDER-TERRAIN-001: Render Morrowind Exterior LAND Cell

Status: Planned

## Goal

Iridius can render the terrain geometry for a Morrowind exterior cell using terrain data from the corresponding Morrowind LAND record.

This capability establishes the minimum vertical slice required to prove that Morrowind terrain data can flow through Iridius from content loading to the renderer.

The initial capability covers terrain geometry for a single exterior cell.

It does not yet require a complete terrain streaming, LOD, collision, or texture-blending system.

## Background

Morrowind's exterior world is divided into grid-aligned exterior cells.

Terrain associated with an exterior cell is stored in a corresponding LAND record.

For this capability, Iridius should use the real terrain elevation information from that LAND record to construct and render the cell's terrain surface.

OpenMW may be used as a behavioral and format reference, but Iridius should use its own rendering architecture.

## Dependencies

This capability assumes Iridius can already:

- load the relevant Morrowind content/plugin data
- identify an exterior cell by grid coordinates
- access or decode the LAND record associated with that cell
- submit ordinary indexed geometry through the renderer

If LAND decoding does not yet exist, implement only the minimum decoding necessary for this capability rather than building the complete terrain subsystem.

## Required Behavior

### RENDER-TERRAIN-001-A: Locate LAND data

Given an exterior cell coordinate such as:

    (0, 0)

Iridius can locate the LAND record associated with that exterior cell.

Failure to locate required terrain data must produce a useful diagnostic rather than undefined behavior or a crash.

### RENDER-TERRAIN-001-B: Decode terrain elevation

Iridius can decode the LAND terrain elevation information into a representation suitable for constructing terrain geometry.

The resulting elevations must preserve the shape of the source terrain.

The decoding implementation should live in the content/world layer appropriate to the existing Iridius architecture rather than being embedded directly inside Vulkan rendering code.

### RENDER-TERRAIN-001-C: Generate terrain geometry

Iridius generates terrain geometry representing the LAND elevation data.

The generated geometry must:

- cover the correct area for one exterior cell
- use the source terrain elevations
- produce a connected terrain surface
- use consistent triangle winding
- generate or provide appropriate surface normals
- avoid visible cracks within the cell

The exact CPU/GPU representation is an implementation decision.

### RENDER-TERRAIN-001-D: Correct world placement

The rendered terrain must appear at the correct world position implied by the exterior cell's grid coordinates.

For example, terrain loaded for neighboring cell coordinates should occupy neighboring world-space regions rather than overlapping at the origin.

The mapping between Morrowind coordinates and Iridius coordinates should be centralized and documented rather than duplicated inside terrain rendering code.

### RENDER-TERRAIN-001-E: Render through normal Iridius pipeline

Terrain must render through the normal Iridius rendering architecture.

It should participate appropriately in:

- camera transforms
- depth testing
- reverse-Z, if used by the renderer
- ordinary opaque rendering
- existing frame-graph synchronization

Do not create a standalone rendering path solely for the terrain test.

### RENDER-TERRAIN-001-F: Basic terrain material

The terrain must use a visible material sufficient to inspect its geometry.

For RENDER-TERRAIN-001, a simple neutral or default terrain material is acceptable.

Accurate Morrowind terrain texture assignment and blending are explicitly deferred to a later capability.

## Initial Vertical Slice

Create a deterministic scenario that:

1. loads one known Morrowind exterior LAND record
2. constructs the corresponding terrain surface
3. places it at the correct exterior-cell location
4. positions a camera so the terrain is visible
5. renders the terrain through the normal Iridius renderer

The selected LAND record should contain non-flat terrain so incorrect elevation decoding is readily detectable.

## Automated Verification

Verification should not rely exclusively on a human looking at the scene.

### TERRAIN-001: Geometry construction

Given a known LAND fixture:

Assert that:

- terrain geometry is generated
- vertex/index data is non-empty
- generated heights fall within expected bounds
- all generated values are finite

### TERRAIN-002: Known elevation samples

For several known positions within the test LAND record:

Assert that generated terrain elevation matches the decoded source elevation within an appropriate tolerance.

This should catch errors such as:

- incorrect delta decoding
- incorrect scaling
- transposed axes
- incorrect row ordering

### TERRAIN-003: Cell positioning

Load terrain fixtures representing two different exterior cell coordinates.

Assert that their calculated world-space bounds occupy the expected neighboring regions and do not overlap incorrectly.

This test does not require both cells to be rendered simultaneously.

### TERRAIN-004: Rendering smoke test

Render the terrain fixture using the existing renderer-test infrastructure.

Assert that:

- the terrain draw occurs
- no Vulkan validation errors occur
- rendering completes successfully
- output is non-empty

If the existing harness supports deterministic screenshots, store a visual regression image.

Visual regression should supplement the geometry-level tests rather than replace them.

## Debugging / Observability

Provide enough information to diagnose terrain failures.

Where appropriate, terrain debug information should expose:

- exterior cell coordinate
- terrain world-space bounds
- minimum elevation
- maximum elevation
- vertex count
- triangle count

A wireframe/debug visualization is useful if it already fits naturally into Iridius renderer debugging infrastructure, but is not required.

## Architectural Constraints

- LAND parsing must not be implemented inside Vulkan renderer code.
- Renderer code should consume an engine-native terrain representation.
- Morrowind-specific coordinate conversion should have a clear ownership boundary.
- Terrain geometry must use the existing renderer/frame-graph infrastructure.
- Avoid architecture that assumes only one terrain cell will ever exist.
- Do not build a full terrain streaming system as part of this capability.
- Preserve a clean path toward later terrain LOD and cell streaming.

## Out of Scope

The following are intentionally NOT required for RENDER-TERRAIN-001:

- LAND texture blending
- Morrowind LTEX terrain textures
- multi-cell terrain streaming
- terrain LOD
- distant terrain
- terrain collision
- player movement on terrain
- terrain editing
- grass or ground cover
- water
- shadows
- terrain normal maps
- PBR terrain materials
- atmospheric effects
- procedural terrain
- terrain occlusion
- asynchronous terrain loading

These should be separate capabilities.

## Likely Follow-Up Capabilities

- RENDER-TERRAIN-002: Morrowind Terrain Textures
- RENDER-TERRAIN-003: Adjacent Terrain Cells
- RENDER-TERRAIN-004: Terrain Cell Streaming
- RENDER-TERRAIN-005: Terrain LOD
- PHYS-TERRAIN-001: Terrain Collision
- RENDER-TERRAIN-006: Distant Terrain

## Definition of Done

RENDER-TERRAIN-001 is Implemented when:

- a real Morrowind LAND record can be located for an exterior cell
- its terrain elevation data is decoded correctly
- Iridius generates terrain geometry from that data
- terrain occupies the correct world-space cell location
- terrain renders through the normal Iridius rendering pipeline
- known elevation samples are verified automatically
- geometry/world-position verification passes
- renderer smoke test passes
- no Vulkan validation errors are introduced
- existing relevant tests continue to pass
- docs/PARITY.md is updated to mark RENDER-TERRAIN-001 Implemented