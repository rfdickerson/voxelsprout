# RENDER-WATER-001: Basic World Water Rendering

Status: Partial — world-owned exterior/interior patches, water material, animation,
independent renderer toggle, and Vulkan smoke exist. The smoke checks repeated
game time, the packaged normal map, above-water occlusion, submerged terrain,
and validation. A portable visual golden with clearly readable normal and
specular detail remains before Implemented.

## Goal

Iridius renders water surfaces defined by Bethesda world content using the normal rendering pipeline.

The initial capability provides correctly positioned, visually identifiable, animated water suitable for Morrowind exterior and interior environments.

This capability establishes the water-rendering foundation without requiring advanced reflection, refraction, underwater, or shoreline effects.

## Dependencies

- Runtime world/cell representation
- Bethesda content loading sufficient to expose cell water state
- Existing Iridius material and texture systems
- Existing Vulkan renderer and frame graph
- HARNESS-001: Headless Engine Test Runner

## Required Behavior

### RENDER-WATER-001-A: World-owned water state

Water existence and elevation originate from world/content state rather than renderer-owned state.

The renderer consumes an engine-native representation describing water relevant to the currently rendered world or cell.

The world layer must remain usable headlessly without initializing water rendering.

### RENDER-WATER-001-B: Correct water elevation

When a cell or worldspace specifies a water level, Iridius renders the water surface at the corresponding world-space elevation.

Automated verification must detect incorrect:

- elevation
- axis mapping
- cell/worldspace placement
- coordinate conversion

### RENDER-WATER-001-C: Water surface coverage

Water must cover the appropriate visible region without requiring a unique manually authored mesh for each cell.

The implementation must leave a viable path toward:

- multiple active cells
- large exterior water bodies
- interior water
- streamed worlds

The exact water-surface representation is an implementation decision.

### RENDER-WATER-001-D: Basic water material

Water must be visually distinguishable from ordinary opaque geometry.

At minimum, the material should support:

- configurable water color
- partial transparency or equivalent physically plausible transmission treatment
- surface normal variation
- animated surface motion
- appropriate specular response

The exact shading model is an implementation decision and should integrate with existing Iridius lighting/material conventions.

### RENDER-WATER-001-E: Animated surface

Water appearance changes over time to avoid looking like a static transparent plane.

Animation must be deterministic with respect to engine/game time where practical.

Animation should not require continuously modifying water mesh geometry on the CPU unless justified by the existing architecture.

### RENDER-WATER-001-F: Depth and ordering

Water must interact correctly with scene depth.

Geometry in front of the water surface must occlude it correctly.

Water rendering must not corrupt opaque-scene depth behavior.

The implementation must explicitly define whether and how water writes depth.

### RENDER-WATER-001-G: Transparent rendering integration

Water must participate correctly in Iridius's transparency/render-order model.

Do not introduce a separate ad hoc rendering architecture solely for water.

If the existing renderer lacks infrastructure necessary for water transparency, add only the smallest reusable infrastructure required by this capability.

### RENDER-WATER-001-H: Camera interaction

Water remains correctly positioned and visually stable as the camera:

- translates
- rotates
- crosses cell boundaries
- approaches the water surface

The surface must not move with the camera in world space or exhibit obvious coordinate instability.

### RENDER-WATER-001-I: Enable/disable

Water rendering must be independently enableable for debugging and verification.

Disabling water rendering must not alter unrelated world or renderer state.

## Initial Vertical Slice

Create a deterministic test scene containing:

- terrain or static geometry
- a known water level
- geometry extending both above and below the water surface
- a fixed camera observing the water at an oblique angle

The scene should demonstrate:

1. water appears at the expected elevation
2. terrain above the water remains visible normally
3. submerged geometry is visible according to the initial material behavior
4. the water surface animates over time
5. scene geometry correctly occludes the water where appropriate

## Automated Verification

### WATER-001: Water elevation

Given a known world water level:

Assert that the water surface transform or generated geometry corresponds to the expected world-space elevation within tolerance.

### WATER-002: Water presence

Given:

    water enabled

Assert that the renderer schedules the expected water rendering work.

Given:

    water disabled

Assert that the water rendering work is absent.

### WATER-003: Surface bounds

Given a known test scene:

Assert that the generated or selected water surface covers the expected visible world region without invalid or non-finite bounds.

### WATER-004: Animation

Render or evaluate the water state at two known game times.

Assert that:

- relevant animated water state differs
- repeating the same time value produces equivalent state

### WATER-005: Rendering smoke

Render the deterministic water fixture.

Assert that:

- rendering completes successfully
- output is non-empty
- required water textures/resources resolve
- no unexpected fallback resources are used
- no Vulkan validation errors occur

### WATER-006: Depth interaction

Render known geometry located in front of and behind the water surface.

Verify that foreground geometry correctly occludes the water and that the water pass does not incorrectly alter unrelated opaque geometry.

If deterministic screenshot testing exists, retain a golden image for this scenario.

## Visual Verification

Where screenshot regression infrastructure exists, maintain at least one deterministic water image containing:

- visible shoreline or geometry intersection
- above-water geometry
- below-water geometry
- surface-normal variation
- specular response

The image should make obvious failures easy to detect, including:

- incorrect water elevation
- opaque water
- missing water
- upside-down or incorrect normals
- broken depth ordering
- stationary water animation
- water moving with the camera

Visual verification supplements structural tests rather than replacing them.

## Renderer Integration

The intended conceptual ownership is:

    Bethesda content
          |
          v
      world/cell
      water state
          |
          v
    engine-native
      water data
          |
          v
       renderer
          |
          v
    water surface/material

Content parsing and world-state interpretation must not occur inside Vulkan shaders or renderer backend code.

## Debugging and Observability

Where practical, expose:

- active water level
- active water region/worldspace
- water-rendering enabled state
- water draw count
- water material/resource identifiers
- water pass CPU/GPU timing

A debug visualization of water bounds is desirable if it naturally fits existing renderer debugging infrastructure.

## Performance

Water rendering must remain compatible with PERF-001 frame-pacing requirements.

The implementation must not:

- generate large water meshes every frame
- synchronously load water resources during steady-state rendering
- introduce routine frame-time stalls
- require CPU readback from the GPU

Record a baseline for:

- water draw count
- CPU water-pass preparation time
- GPU water-pass time
- relevant GPU memory usage

Advanced optimization is outside this capability.

### Baseline (RelWithDebInfo, Intel LNL, synthetic Morrowind LAND, 320×180)

The validation smoke records one water draw for 60 shoreline patches. Measured
CPU draw-command recording was 0.013–0.040 ms and the isolated GPU water draw
was 1.13–1.48 ms across five passing runs on 2026-10-02; cold GPU runs varied
higher. The geometry payload is 40,800 bytes (240 vertices and 360 indices).
The packaged RGBA8 normal map is 1254×1254 with 11 mips, or 8,384,072 bytes
of texel payload. These are payload sizes; Vulkan allocation overhead and the
optional reflection/refraction targets are outside this baseline.

## Architectural Constraints

- World state owns whether water exists and its logical elevation.
- Vulkan renderer code must not parse Bethesda cell records.
- Water should use the existing frame graph.
- Water resources should use the existing resource/VFS system.
- Avoid creating a unique giant mesh for every exterior cell if a simpler reusable representation satisfies the requirement.
- Preserve a clean path toward reflections, refraction, underwater rendering, and multi-cell streaming.
- Do not couple water rendering to the player character.
- Do not implement gameplay swimming behavior as part of this capability.

## Out of Scope

The following are intentionally deferred:

- screen-space reflections
- planar reflections
- ray-traced reflections
- scene refraction
- depth-based water coloration
- shoreline foam
- caustics
- underwater fog
- underwater post-processing
- wave simulation
- FFT oceans
- water physics
- swimming
- buoyancy
- water collision
- water audio
- waterfalls
- flowing rivers
- weather-driven waves
- water interaction with actors
- displacement tessellation

These should be separate capabilities.

## Likely Follow-Up Capabilities

- RENDER-WATER-002: Water Reflection
- RENDER-WATER-003: Water Refraction and Depth Color
- RENDER-WATER-004: Underwater Rendering
- RENDER-WATER-005: Shoreline and Water Intersection
- RENDER-WATER-006: Enhanced Surface Waves
- PHYS-WATER-001: Water Volume Queries
- MECH-WATER-001: Swimming and Water Interaction

## Definition of Done

RENDER-WATER-001 is Implemented when:

- Bethesda/world water state is represented outside the renderer
- a water surface renders at the correct world-space elevation
- the water material is visibly distinct from opaque geometry
- surface appearance animates deterministically
- depth interaction with ordinary scene geometry is correct
- water can be independently enabled and disabled
- deterministic water verification exists
- renderer smoke test passes
- required water resources resolve normally
- no Vulkan validation errors are introduced
- relevant performance baseline is recorded
- existing relevant tests continue to pass
- docs/PARITY.md is updated to mark RENDER-WATER-001 Implemented
