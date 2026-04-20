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
cmake -S . -B /tmp/voxelsprout-cmake-check \
  -DVOXEL_BUILD_APP=OFF \
  -DVOXEL_BUILD_TOOLS=ON \
  -DBUILD_TESTING=ON \
  -DCMAKE_BUILD_TYPE=Debug

cmake --build /tmp/voxelsprout-cmake-check --target voxel_morrowind_balmora_cooker -j 4
cmake --build /tmp/voxelsprout-cmake-check --target voxel_imported_scene_tests -j 4
```

Windows app builds should continue using `cmake-build-release`.
Linux builds should continue using `cmake-build-linux`.

## Cook Balmora

WSL2 example:

```bash
/tmp/voxelsprout-cmake-check/voxel_morrowind_balmora_cooker \
  "/mnt/c/GOG Games/Morrowind/Data Files" \
  /tmp/balmora_scene.bin \
  /tmp/balmora_terrain.obj
```

Windows example from PowerShell after building the tool:

```powershell
voxel_morrowind_balmora_cooker.exe `
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
/tmp/voxelsprout-cmake-check/voxel_imported_scene_tests
```

This currently validates imported-scene save/load round-tripping and terrain OBJ export.

## Current Workflow

1. Build `voxel_morrowind_balmora_cooker`
2. Point it at the Morrowind `Data Files` directory
3. Generate `balmora_scene.bin`
4. Optionally inspect `balmora_terrain.obj`
5. Continue implementing NIF mesh import and runtime Vulkan rendering

## Next Planned Engine Work

- Minimal NIF static mesh import for Balmora buildings and props
- Texture loading and mip preparation for Morrowind assets
- Imported-scene GPU upload path
- Separate Vulkan terrain/object passes
- Camera spawn and debug mode for a Balmora scene viewer
