# AGENTS.md

## Scope

This repository is a Bethesda-only Vulkan engine/runtime for Morrowind, Oblivion,
Fallout 3/New Vegas, and Skyrim. Keep one explicit imported-scene rendering path.
Do not reintroduce voxel worlds, strategy/city simulations, mini-games, Lua content
systems, dynamic engine plugins, or generalized render-graph machinery.

The runtime target is `odai`. Internal `newvegas` class names and `ODAI_FNV_*`
environment variables remain for compatibility while the importer supports all five
games.

## Build

Dependencies come from `vcpkg.json`.

```bash
cmake --preset linux-vcpkg
cmake --build --preset linux-vcpkg -j
ctest --test-dir build-linux --output-on-failure
```

For performance work use `linux-vcpkg-relwithdebinfo` or
`linux-vcpkg-release`; Debug numbers are not representative.

Options retained by policy: `ODAI_BUILD_RUNTIME`, `ODAI_BUILD_TOOLS`,
`BUILD_TESTING`, ccache, LTO, native-arch, temporal upscaling, and XeSS.

## Architecture

- `odai_core`: logging and jobs.
- `odai_bethesda_import`: archives/plugins/NIFs/records, DDS, cell building,
  actors, dialogue, weather, animation, and `ImportedScene` serialization.
- `odai_renderer`: the Vulkan imported-scene renderer and RPG UI backend.
- Focused `odai_ui`, `odai_audio`, `odai_anim`, `odai_dialogue`, and
  `odai_upscale` libraries.

Renderer passes use explicit barriers and explicit control flow. Preserve streamed
chunk and cooked-scene serialization compatibility unless the serialized layout truly
changes.

## Testing

Tests are hand-rolled executables registered with CTest. Preserve Bethesda import,
scene serialization, animation, dialogue, audio, upscaling, residency, bindless,
GPU-arena, jobs, frame stats, core math, imported lighting/SSGI policy, frame graph,
PBR packing, and retained RPG UI coverage.

Real-data probes are optional and must never commit or redistribute game data.

## Local Skyrim preview preferences

Use JK’s Skyrim + SMIM for future Skyrim scene runs unless the user explicitly
requests another profile or a vanilla comparison. The local combined profile is
`captures/jk-skyrim-showcase/profile.json`. Keep the small 768x432 logical window
and native-DPI framebuffer rendering (render scale 1; no forced smaller render
extent). Mod assets and capture evidence remain local.

## Capability Status

Capability status is tracked in:

    docs/PARITY.md

When completing work associated with a capability:

1. Read the capability's Definition of Done.
2. Verify that all required acceptance criteria are satisfied.
3. Run the required verification.
4. Only after successful verification, update the capability's status
   in docs/PARITY.md.
5. Do not mark a capability Implemented if required tests are failing
   or acceptance criteria remain incomplete.
6. If the capability is only partially implemented, record it as Partial
   and briefly document what remains.

## Capability work

Product capabilities are defined under:

    docs/capabilities/

Capability IDs such as WORLD-003 refer to files in that directory.

Before implementing a capability:

1. Read its capability specification.
2. Identify its dependencies.
3. Inspect the relevant existing architecture.
4. Determine current implementation status.
5. Add or update verification tests.
6. Implement the smallest coherent change.
7. Run verification.
8. Review the final diff.

Do not weaken capability requirements merely to make tests pass.

## OpenMW

OpenMW is available as a read-only behavioral reference at:

    ../references/openmw/

Use it to understand expected Morrowind-compatible behavior.

Do not modify it.

Do not copy OpenMW architecture into Iridius unless the existing
Iridius architecture independently warrants that design.

## Architecture

Architecture documentation lives under:

    docs/architecture/

Only read architecture documents relevant to the current task.
