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
