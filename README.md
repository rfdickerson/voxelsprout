# Voxelsprout Compute Lab

Voxelsprout is now a compute-first Vulkan rendering laboratory for progressive Monte Carlo path tracing and volumetric experiments.

## Scope

- Vulkan 1.3 compute-centric renderer
- Progressive HDR accumulation
- Compute CloudPathTrace pass + compute tone map pass
- Explicit synchronization2 barriers and image layouts
- Timeline semaphore driven frame loop
- GPU timestamp profiling
- ImGui runtime controls

Removed from this codebase:

- Voxel/chunk storage and meshing
- Raster pipelines and shadow maps
- SSAO/G-buffer/reverse-Z systems
- World simulation and deterministic tick systems

## Project Layout

- `app/` main loop, input, UI wiring
- `core/` math, camera, RNG, logging
- `scene/` simple scene/volume parameters
- `render/` Vulkan setup and compute pass execution
- `shaders/` Slang compute shaders
- `tests/` math and sampling validation
- `assets/` data assets (currently empty/minimal)

## Build

### Linux

```bash
cmake -S . -B cmake-build-linux -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build cmake-build-linux -j
```

### Windows (vcpkg)

```powershell
cmake --preset vcpkg
cmake --build --preset vcpkg
```

## Run

Linux:

```bash
./cmake-build-linux/voxelsprout_compute_lab
```

## Controls

- `W/A/S/D/Q/E` move camera
- Mouse look
- `M` toggle mouse capture
- `Tab` toggle UI
- `ESC` quit

## Tests

```bash
ctest --test-dir cmake-build-linux --output-on-failure
```
