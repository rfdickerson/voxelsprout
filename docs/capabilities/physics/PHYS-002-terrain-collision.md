# PHYS-002: Terrain Collision

Status: Implemented

## Goal

Iridius supports collision queries against Morrowind exterior terrain.

Terrain collision must be derived from the same decoded terrain data used to represent Morrowind LAND geometry, without depending on renderer-owned geometry or Vulkan resources.

This capability establishes the physics representation required for later player movement, NPC movement, projectiles, and grounded world interaction on exterior terrain.

## Dependencies

- PHYS-001: Static World Collision
- RENDER-TERRAIN-001: Render Morrowind Exterior LAND Cell
- Morrowind LAND decoding
- Runtime exterior-cell representation
- HARNESS-001: Headless Engine Test Runner

## Required Behavior

### PHYS-002-A: Terrain collision representation

A loaded Morrowind exterior terrain cell can be represented in the physics world as static collision geometry.

The collision representation must preserve:

- terrain elevation
- cell world position
- terrain orientation
- correspondence with the originating exterior cell

Terrain collision must be derived from engine-native terrain data rather than extracted from renderer buffers.

### PHYS-002-B: Correct elevation

Collision height must match the terrain elevation represented by the source LAND data within an appropriate numeric tolerance.

A physics query against terrain at a known horizontal position must report a vertical position consistent with the decoded terrain surface.

### PHYS-002-C: Correct cell placement

Terrain collision must appear at the correct world-space location for its exterior-cell coordinates.

Collision geometry from different terrain cells must not overlap at the origin or otherwise ignore exterior-cell placement.

### PHYS-002-D: Ray queries

PHYS-001 ray queries must work against terrain.

For a downward ray above known terrain:

- a hit is reported
- hit position is correct
- hit distance is correct
- surface normal is reasonable
- hit result identifies the originating terrain cell or corresponding world object

### PHYS-002-E: Shape queries

The physics shape-query mechanism established in PHYS-001 must work against terrain.

This should be sufficient to support later character-controller work.

For example, a downward capsule or sphere sweep should be able to detect terrain beneath the query shape.

The specific query shape is an implementation decision.

### PHYS-002-F: Surface normals

Terrain collision queries must provide surface normals corresponding to the local terrain slope.

Normals should not always report straight upward for non-flat terrain.

Returned normals must use the expected Iridius world-space coordinate system.

### PHYS-002-G: Adjacent-cell compatibility

The terrain-collision representation must leave a viable path for multiple neighboring exterior cells to coexist.

PHYS-002 does not require full terrain streaming, but its design must not assume only one terrain cell can exist in the physics world.

When two known neighboring cells are loaded for verification, their collision surfaces should occupy adjacent world-space regions without incorrect overlap.

### PHYS-002-H: Headless operation

Terrain collision must work without renderer initialization or a visible window.

The headless harness must be able to:

- load terrain data
- create terrain collision
- perform collision queries
- inspect results
- terminate automatically

## Initial Vertical Slice

Create a deterministic scenario using a known Morrowind LAND fixture containing:

- at least one flat or nearly flat region
- at least one sloped region
- meaningful elevation variation

The scenario should:

1. load the LAND record
2. create the engine-native terrain representation
3. register corresponding terrain collision with the physics world
4. cast rays and/or shape queries against known terrain locations
5. verify resulting elevations and normals

Rendering is not required for this scenario.

## Automated Verification

### TERRAIN-PHYS-001: Known terrain elevation

Given a known LAND fixture and horizontal sample position:

Cast a downward ray from above the terrain.

Assert:

- terrain is hit
- hit height matches expected terrain elevation within tolerance
- hit distance is correct within tolerance

Test several sample positions with different heights.

### TERRAIN-PHYS-002: Terrain slope

Select a known sloped terrain region.

Perform a downward ray query.

Assert:

- hit occurs
- returned normal is finite
- returned normal points generally upward
- returned normal is not equivalent to a flat-surface normal when the sampled source terrain is sloped

Where practical, compare against an expected normal derived from the source terrain data.

### TERRAIN-PHYS-003: Empty space

Perform a query outside the bounds of any loaded terrain cell.

Assert:

    hit == false

The physics system must not behave as though unloaded terrain exists.

### TERRAIN-PHYS-004: Cell placement

Load terrain for an exterior cell with non-zero grid coordinates.

Query a known terrain position.

Assert that collision occurs at the expected world-space location rather than at coordinates appropriate to cell (0,0).

### TERRAIN-PHYS-005: Adjacent cells

Load two neighboring exterior terrain cells.

Assert that:

- both cells register collision
- their world-space bounds are adjacent
- queries against each cell return the correct originating cell
- collision geometry does not incorrectly overlap

This test does not require streaming.

### TERRAIN-PHYS-006: Shape query

Perform the selected PHYS-001 shape query against terrain.

Assert expected collision and non-collision cases.

The test should demonstrate that terrain collision is usable by a future character controller rather than only by raycasts.

### TERRAIN-PHYS-007: Physics/render terrain agreement

Using the shared engine-native terrain representation, verify that sampled physics terrain heights correspond to the heights used for terrain geometry generation.

This test should catch accidental divergence between the physics and rendering terrain representations.

The test should not require the renderer itself to initialize.

## Terrain Representation

Rendering and physics should consume a shared engine-native terrain representation where appropriate.

The intended conceptual relationship is:

    Morrowind LAND
          |
          v
    decoded terrain data
          |
          +-------------------+
          |                   |
          v                   v
    renderer terrain      physics terrain
    representation        representation

Physics must not retrieve terrain by reading GPU vertex/index buffers.

Renderer-specific and physics-backend-specific representations may differ, but they should derive from the same authoritative terrain data.

## Lifecycle

Terrain collision must have a defined lifecycle tied to loaded world state.

At minimum:

- collision may be registered when a terrain cell becomes active/loaded
- collision may later be removed when that cell is unloaded
- removing one terrain cell must not invalidate unrelated static collision

PHYS-002 does not require automatic cell streaming or unloading, but the API/design must allow those operations later.

## Diagnostics and Observability

Where practical, expose terrain-physics diagnostic information including:

- exterior-cell coordinate
- physics bounds
- minimum and maximum terrain elevation
- number/type of collision primitives
- sampled hit position
- sampled hit normal
- originating terrain cell

Headless test failures should provide enough information to distinguish:

- incorrect LAND decoding
- incorrect coordinate conversion
- incorrect physics registration
- incorrect query behavior

## Architectural Constraints

- Terrain physics must not depend on renderer or Vulkan resources.
- Terrain collision must derive from authoritative engine-native terrain data.
- Morrowind-specific LAND decoding must remain outside the physics backend.
- Avoid duplicating terrain elevation data unnecessarily.
- Physics-backend types should not leak broadly into world/content code.
- Terrain-cell coordinate conversion should use the same centralized mapping as terrain rendering.
- Do not build a character controller as part of PHYS-002.
- Do not implement terrain streaming as part of PHYS-002.
- The design must support multiple terrain cells existing concurrently.

## Performance

Record basic baseline information where practical:

- terrain collision creation time
- memory used by one terrain cell's collision representation
- representative ray-query time
- representative shape-query time

No advanced terrain-physics optimization is required for PHYS-002.

However, the implementation should avoid regenerating terrain collision every frame.

Baseline from the deterministic two-cell Debug headless fixture (Intel LNL
machine, 2026-10-02): 132.9 ms to create both Jolt terrain bodies, 0.148 ms
for four mixed ray and sphere queries, and 33,800 bytes of decoded source
heights (16,900 bytes per cell). Jolt's internal mesh allocation is not
exposed by this harness. Terrain bodies are prepared once per loaded cell.

## Out of Scope

The following are intentionally deferred:

- player character controller
- grounded-state logic
- gravity
- step climbing
- slope movement policy
- NPC terrain movement
- terrain streaming
- asynchronous collision generation
- dynamic terrain deformation
- water collision
- cliff-edge gameplay behavior
- navmesh generation
- terrain-material-dependent physics
- footstep/material queries
- physics debug drawing

These should be implemented as later capabilities.

## Likely Follow-Up Capabilities

- PHYS-003: Character Controller
- PHYS-004: Dynamic Rigid Bodies
- PHYS-005: Trigger Volumes
- WORLD-004: Adjacent Exterior Cell Loading
- RENDER-TERRAIN-004: Terrain Cell Streaming
- AI-NAV-001: Exterior Navigation
- PHYS-TERRAIN-003: Terrain Surface Material Queries

## Definition of Done

PHYS-002 is Implemented when:

- Morrowind terrain cells can register static collision
- terrain collision derives from authoritative terrain data
- terrain collision is correctly positioned in world space
- downward ray queries return correct terrain elevations
- slope queries return meaningful surface normals
- shape queries work against terrain
- unloaded/out-of-bounds terrain does not report false collisions
- neighboring terrain cells can coexist without incorrect overlap
- physics and rendered terrain heights remain consistent
- terrain physics works headlessly
- deterministic terrain-physics verification exists
- existing relevant tests continue to pass
- docs/PARITY.md is updated to mark PHYS-002 Implemented
