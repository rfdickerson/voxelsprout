# Morrowind Engine

This repo is being repurposed into a small Vulkan-based Morrowind scene renderer, starting with a Balmora demo pipeline.

Current implemented scope:

- Offline Balmora scene cooking from vanilla `Morrowind.esm`
- Terrain extraction from `LAND` records
- Landscape texture reference extraction from `LTEX`
- Exterior cell reference extraction for Balmora and a 1-cell border ring
- Binary imported-scene serialization
- Terrain OBJ export for inspection

Not implemented yet:

- NIF static object geometry import
- Runtime in-engine rendering of the imported scene
- Texture decode/upload path for Morrowind assets

## Data Files Path

The cooker expects the `Data Files` directory inside a Morrowind install.

Typical locations for this machine:

- Windows install root: `C:\GOG Games\Morrowind`
- Windows data path: `C:\GOG Games\Morrowind\Data Files`
- WSL2 install root: `/mnt/c/GOG Games/Morrowind`
- WSL2 data path: `/mnt/c/GOG Games/Morrowind/Data Files`

Use the `Data Files` path as the first cooker argument.

## Build

Linux/WSL2 tool-only build:

```bash
cmake -S . -B /tmp/odai-cmake-check \
  -DODAI_BUILD_APP=OFF \
  -DODAI_BUILD_TOOLS=ON \
  -DBUILD_TESTING=ON \
  -DCMAKE_BUILD_TYPE=Debug

cmake --build /tmp/odai-cmake-check --target odai_balmora_cooker -j 4
cmake --build /tmp/odai-cmake-check --target odai_imported_scene_tests -j 4
```

Windows app builds should continue using `cmake-build-release`.
Linux builds should continue using `cmake-build-linux`.

## Cook Balmora

WSL2 example:

```bash
/tmp/odai-cmake-check/odai_balmora_cooker \
  "/mnt/c/GOG Games/Morrowind/Data Files" \
  /tmp/balmora_scene.bin \
  /tmp/balmora_terrain.obj
```

Windows example from PowerShell after building the tool:

```powershell
odai_balmora_cooker.exe `
  "C:\GOG Games\Morrowind\Data Files" `
  "C:\temp\balmora_scene.bin" `
  "C:\temp\balmora_terrain.obj"
```

Arguments:

- `arg1`: Morrowind `Data Files` path
- `arg2`: output imported-scene binary
- `arg3`: optional terrain OBJ output path

## How Path Selection Works

There is no config file or environment variable yet. You specify the Morrowind asset path directly on the command line when running the cooker.

Use:

- `C:\GOG Games\Morrowind\Data Files` on Windows
- `/mnt/c/GOG Games/Morrowind/Data Files` on WSL2

The cooker reads:

- `Morrowind.esm`
- referenced terrain texture names from the same `Data Files` tree
- object model record metadata from `Morrowind.esm`

## Tests

Run the importer-focused test target:

```bash
/tmp/odai-cmake-check/odai_imported_scene_tests
```

This currently validates imported-scene save/load round-tripping and terrain OBJ export.

## Current Workflow

1. Build `odai_balmora_cooker`
2. Point it at the Morrowind `Data Files` directory
3. Generate `balmora_scene.bin`
4. Optionally inspect `balmora_terrain.obj`
5. Continue implementing NIF mesh import and runtime Vulkan rendering

## Strategic Map Prototype

The repo also hosts the first vertical slice of a Civ / Total War-inspired strategy
layer: an explorable 3D hex map rendered with the existing Vulkan engine. It is a
small, self-contained game layer that reuses the imported-scene render path, so no
renderer or shader changes were required.

### Architecture

Game code lives under `src/game/` and stays free of any Vulkan/renderer types
(the engine's reusable code remains in `core/`, `math/`, `render/`, `world/`,
`import/`; offline generators in `tools/`; tests in `tests/`):

- `src/game/strategy_map.{h,cc}` — the data model: a pointy-top hex grid in odd-r
  offset coordinates, per-tile terrain type, elevation, river/road/border flags,
  fog-of-war visibility, territory owner, and settlement markers. Pure CPU data.
- `src/game/strategy_map_io.{h,cc}` — versioned binary serialization (`.smap`).
- `src/game/strategy_map_mesh.{h,cc}` — meshes a map into an `ImportedScene`
  (terrain prisms colored per terrain, a debug grid overlay, and settlement
  markers) using the engine's packed vertex-color render stream.

The map is drawn by feeding the meshed scene to the existing imported-scene
pipeline, which already provides an angled camera and free-fly pan/zoom/orbit.

### Build

```powershell
cmake -S . -B cmake-build-release
cmake --build cmake-build-release --target odai_strategy_map_gen
cmake --build cmake-build-release --target odai_strategy_map_tests
cmake --build cmake-build-release --target odai            # the runtime viewer
```

Tools and tests also build on Linux/WSL2 with `-DODAI_BUILD_TOOLS=ON
-DBUILD_TESTING=ON` (see the tool-only build above).

### Generate and view a map

```powershell
# 1. Generate a sample hex map: writes strategy_map.smap and strategy_map_scene.bin
#    Optional args: <smap> <bin> <width> <height> <seed>
cmake-build-release\odai_strategy_map_gen.exe

# 2. Run the viewer. The app loads strategy_map.smap from the working directory,
#    or set ODAI_STRATEGY_MAP to an explicit path.
$env:ODAI_STRATEGY_MAP = "strategy_map.smap"
cmake-build-release\odai.exe
```

Camera controls reuse the imported-scene fly camera: `WASD` to pan, mouse to
orbit/look, `Space` to rise and `Shift` to descend (altitude "zoom"), and `Ctrl`
to move faster.

The generated `strategy_map_scene.bin` is also a plain imported scene, so it can
alternatively be viewed via `ODAI_IMPORTED_SCENE=strategy_map_scene.bin` with no
strategy-map support compiled in.

### Tests

```powershell
cmake-build-release\odai_strategy_map_tests.exe
# or via CTest:
ctest --test-dir cmake-build-release -R odai_strategy_map_tests
```

Covers hex indexing/bounds, neighbor symmetry, hex geometry (corner distance and
elevation), `.smap` serialization round-trip, malformed-file rejection, and that
the mesher produces a valid, terrain-colored, renderable scene.

### Known limitations

- Square topology was not implemented; the model is hex-only for now.
- Rivers/roads are shown as tile tint and borders as colored hex edges, rather
  than true edge-following river/road geometry.
- The map is meshed offline by the generator and at app startup; there is no
  in-app map editing or live regeneration yet.
- Fog-of-war state is stored per tile but not yet consumed by the renderer.
- The `src/game` + `src/app` runtime wiring was verified by building and running
  the model, serialization, mesher, and generator headlessly; the full Vulkan app
  build requires the vcpkg Vulkan/GLFW/imgui dependencies on your machine.

## Custom UI Framework

A first-party, retained-mode UI library (`odai_ui` + `odai_ui_vulkan`) renders the
game's own interface — ImGui is kept only as a dev/debug overlay, a separate system
with no code in common. `odai_ui` is Vulkan-free, unit-testable, and packaged to be
vendored into other Vulkan engines (CMake install/export targets, a standalone
integration sample). See [`docs/UI_LIBRARY.md`](docs/UI_LIBRARY.md) for the full
architecture, widget catalog, theming guide, and integration walkthrough, and
[`examples/vulkan_ui_integration/`](examples/vulkan_ui_integration/) for a from-scratch
embedding with no dependency on this repo's app/game/world code.

## Next Planned Engine Work

- Minimal NIF static mesh import for Balmora buildings and props
- Texture loading and mip preparation for Morrowind assets
- Imported-scene GPU upload path
- Separate Vulkan terrain/object passes
- Camera spawn and debug mode for a Balmora scene viewer
