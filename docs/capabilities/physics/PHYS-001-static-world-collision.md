# PHYS-001: Static World Collision

Status: Implemented

Verification: `odai_headless_static_physics` exercises PHYS-STATIC-001 through
PHYS-STATIC-006 without a renderer. The synthetic fixture registers five static
triangle meshes and checks eight ray/sphere queries, including translated,
rotated, scaled Bethesda-local geometry and nearest-hit object identity.
`ctest --test-dir build-linux --output-on-failure -j 4` passed 56/56 tests.
One local Debug run measured 2.83 ms physics initialization, 0.49 ms mesh
registration, and 0.33 ms for the eight-query batch; these are diagnostic
baselines, not performance targets.

## Goal

Iridius supports collision against static world geometry.

Runtime world objects designated as static collision geometry can be represented in the physics system and queried for intersections without depending on rendering state.

This capability establishes the minimum physics foundation required for later player movement, terrain collision, NPC movement, projectiles, and world interaction.

## Dependencies

- Basic runtime world representation
- Runtime object transforms
- Mesh/resource loading sufficient to obtain collision geometry
- HARNESS-001: Headless Engine Test Runner

This capability must not depend on a visible renderer.

## Required Behavior

### PHYS-001-A: Physics world

Iridius provides a runtime physics world capable of containing static collision objects.

The physics world must have a clearly defined lifetime and ownership relationship with the runtime world.

Physics state must not be owned by renderer objects.

### PHYS-001-B: Static collision body

A runtime world object may have a corresponding static collision body.

At minimum, the collision body must preserve:

- world position
- orientation
- collision geometry
- association with the originating runtime object

Moving or transforming the source object before physics registration must result in collision geometry appearing at the expected world-space location.

Dynamic-body behavior is outside this capability.

### PHYS-001-C: Collision geometry

Static collision geometry may be generated or loaded from engine-native geometry appropriate to the existing Iridius architecture.

The implementation must support enough geometry complexity to represent ordinary world surfaces such as:

- floors
- walls
- ramps
- static architecture

The exact physics representation is an implementation decision.

Possible representations may include:

- triangle meshes
- compound shapes
- convex shapes where appropriate

The capability does not prescribe a particular physics library or shape strategy.

### PHYS-001-D: Ray queries

The physics world supports a ray query against static geometry.

A ray query must provide enough result information to determine:

- whether a hit occurred
- hit position
- hit distance
- surface normal
- originating world object when applicable

Ray queries must operate independently of rendering.

### PHYS-001-E: Shape collision queries

The physics world supports at least one non-ray collision query suitable for later gameplay movement.

For example, the implementation may support:

- overlap query
- shape cast
- sweep test

The specific query should be selected based on the existing physics architecture and likely future character-controller needs.

### PHYS-001-F: Transform correctness

Collision geometry must respect the world transform of its corresponding object.

Automated verification must detect incorrect:

- translation
- rotation
- scaling where scaling is supported
- coordinate-system conversion

### PHYS-001-G: Headless operation

All PHYS-001 verification must run without a visible window or active rendering pipeline.

Physics must therefore depend on world/content abstractions rather than GPU/render objects.

## Initial Vertical Slice

Create a deterministic synthetic world containing:

- a floor
- a vertical wall
- an elevated or sloped static surface

Register those objects with the physics world.

The scenario should then perform collision queries that demonstrate:

1. downward ray hits the floor
2. forward ray hits the wall
3. ray through empty space reports no hit
4. returned hit locations are correct
5. returned surface normals are correct
6. the hit result identifies the expected world object

## Automated Verification

### PHYS-STATIC-001: Floor raycast

Given a horizontal floor at a known elevation:

Cast a ray downward.

Assert:

- hit occurs
- expected floor object is returned
- hit position is within tolerance
- hit normal points in the expected direction
- hit distance is within tolerance

### PHYS-STATIC-002: Wall raycast

Given a vertical wall:

Cast a ray toward it.

Assert:

- hit occurs
- expected wall object is returned
- world-space hit position is correct
- surface normal is correct

### PHYS-STATIC-003: No collision

Cast a ray through known empty space.

Assert:

    hit == false

### PHYS-STATIC-004: Transformed object

Place collision geometry at a non-zero translation and rotation.

Assert that collision results occur at the transformed world-space position rather than the original mesh-local position.

### PHYS-STATIC-005: Multiple objects

Place several static collision objects in the physics world.

Perform a query intersecting more than one possible target.

Assert that the nearest valid collision is reported according to the query semantics.

### PHYS-STATIC-006: Non-ray query

Exercise the selected overlap or sweep-query mechanism against known static geometry.

Assert expected collision and non-collision cases.

## Diagnostics and Observability

Physics queries should provide useful diagnostics when run through the test harness.

Where practical, expose:

- number of static physics objects
- collision-shape type
- world-space bounds
- originating world object ID
- query start/end
- hit position
- hit normal
- hit distance

Physics debugging should not require a renderer, although a later debug-drawing capability may visualize physics geometry.

## Architectural Constraints

- Physics must not depend on renderer objects or Vulkan resources.
- Runtime world state remains authoritative for object identity and transforms.
- Physics-specific representation should be owned by the physics layer.
- Avoid embedding physics-library-specific types throughout generic world/gameplay code.
- Maintain a clear boundary between engine-native world data and physics-backend representation.
- Collision geometry should reuse source asset information where sensible rather than requiring manually duplicated test-only geometry.
- Design must leave a clean path toward dynamic bodies and character movement.
- Do not build a character controller as part of PHYS-001.

## Performance

Record basic baseline information for the deterministic fixture where practical:

- number of registered static collision objects
- physics-world initialization time
- query time for representative collision queries

No strict optimization target is required for PHYS-001.

The design must not require rebuilding all static collision geometry for every query or frame.

## Out of Scope

The following are intentionally deferred:

- player character controller
- gravity-driven player movement
- dynamic rigid bodies
- object-to-object physical simulation
- terrain collision
- NPC collision
- ragdolls
- projectiles
- triggers
- water collision
- climbing
- step handling
- slopes as character-movement policy
- collision response
- physics-based animation
- destructible objects
- renderer debug visualization

These should be separate capabilities.

## Likely Follow-Up Capabilities

- PHYS-002: Terrain Collision
- PHYS-003: Character Controller
- PHYS-004: Dynamic Rigid Bodies
- PHYS-005: Trigger Volumes
- PHYS-006: World Object Collision Import
- PHYS-007: Physics Debug Visualization

## Definition of Done

PHYS-001 is Implemented when:

- a physics world exists
- static collision geometry can be registered
- collision geometry respects runtime world transforms
- ray queries work against static geometry
- at least one shape/overlap/sweep query works
- collision results identify corresponding world objects where applicable
- deterministic headless verification exists
- floor, wall, empty-space, transformed-object, and multi-object tests pass
- physics operates independently of rendering
- existing relevant tests continue to pass
- docs/PARITY.md is updated to mark PHYS-001 Implemented
