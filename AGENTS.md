# Morrowind Engine Agent Config

## Project Focus
This project is a custom C++20 / Vulkan engine for exploring and rendering Morrowind-style worlds.

Priorities:
- Preserve Morrowind-like world structure, scale, and readability
- Keep code explicit, small, and debuggable
- Support renderer experimentation without destabilizing the engine
- Prefer practical implementation over generic engine architecture

This is not:
- a generic engine framework
- an ECS experiment
- an enterprise architecture exercise
- a plugin-based platform

## Code Style
Prefer:
- Clear C++20
- Google C++ style
- Small focused functions
- Explicit ownership and lifetime
- Flat data and simple structs
- Value semantics where reasonable
- Stack allocation where reasonable

Avoid:
- Deep inheritance
- Refactoring unrelated systems
- New abstraction layers unless clearly justified
- Template-heavy indirection
- Over-generalizing for hypothetical reuse

## Architecture Rules
Directory responsibilities:
- app/    : main loop, lifecycle
- core/   : math, time, input, utilities
- world/  : terrain, cells, static placement, world data
- render/ : Vulkan renderer, shaders, GPU resources
- assets/ : textures, shaders, materials
- tests/  : focused correctness tests

Rules:
- Only render/ may include Vulkan headers
- Do not leak Vulkan types outside render/
- Rendering consumes world data; it does not control world logic
- Keep data flow one-way: world -> meshing -> renderer

## Morrowind-Specific Guidance
When making changes, preserve the feel of Morrowind-style spaces:
- Terrain and settlements should feel hand-placed, not overly procedural
- Favor irregularity over symmetry
- Avoid repetitive spacing in bridges, canals, levees, roads, and statics
- Respect recognizable regional identity and settlement layout
- Keep architecture, terrain transitions, and waterways visually readable at gameplay scale

For Balmora-like scenes in particular:
- Prefer segmented waterways over uniform canals
- Use asymmetry in bridges and crossings
- Treat embankments as terrain-guided forms, not continuous flood walls
- Preserve district readability from both ground and elevated views

## Rendering Rules
The renderer is a modern Vulkan renderer with explicit synchronization.

When changing rendering code:
- Keep passes explicit
- Do not assume implicit barriers
- Preserve existing pass structure unless asked otherwise
- Keep debug UI hooks and tunables intact
- Make GPU work measurable
- Prefer stable frame pacing over flashy additions

When adding rendering features:
- integrate with GPU timing/profiling
- expose useful debug controls
- document memory/perf impact if nontrivial

## Water Rendering Guidance
Water should support Morrowind-like rivers, canals, and coastal spaces.

Prefer:
- Depth-based absorption
- Fresnel-based reflection/refraction balance
- Refraction in shallow water
- Stronger reflection in deeper water
- Shoreline blending against terrain
- Parameterized tuning for normals, reflection, refraction, and animation

Avoid:
- Uniform opaque blue water
- Tiled-looking ripple patterns
- Perfect mirror water unless explicitly requested
- Hard water mesh cut lines at shore intersections

If ray tracing is used:
- Prefer rasterized water surfaces with ray-traced reflection/refraction queries
- Use thickness/absorption to reduce bottom visibility in deeper water
- Keep the result controllable and stable

## World Building Rules
Prefer:
- Readable terrain silhouettes
- Strong landmark composition
- Plausible paths, waterways, and district transitions
- Manual control over layout-critical areas

Avoid:
- Over-randomized placement
- Uniform spacing patterns
- Large systemic rewrites for small content/layout issues

## Performance Rules
Performance is a feature.

Maintain:
- Smooth frame pacing
- Stable GPU timings
- Predictable allocations
- Clear synchronization boundaries

Avoid:
- Hidden work in hot paths
- Implicit allocations during rendering
- Unbounded per-frame CPU/GPU growth

## Testing Expectations
Add tests for:
- math utilities
- world/grid logic
- deterministic layout utilities
- data transformations with correctness risk

Rendering changes do not need unit tests, but must be:
- measurable
- toggleable
- stable in debug views

## Agent Behavior
When generating code:
- solve the requested problem directly
- keep implementations minimal and specific
- do not redesign the engine unless asked
- do not introduce new subsystems without request
- preserve readability over cleverness

When uncertain:
- choose the simplest correct implementation
- keep changes local
- explain tradeoffs briefly and concretely

## Local Reference Paths
Morrowind Data Files
- Windows: C:\GOG Games\Morrowind\Data Files
- WSL: /mnt/c/GOG Games/Morrowind/Data Files

OpenMW source tree
- Windows: C:\Users\rfdic\OneDrive\Documents\GitHub\openmw
- WSL: /mnt/c/Users/rfdic/OneDrive/Documents/GitHub/openmw

Build directories
- Linux: cmake-build-linux
- Windows: cmake-build-release