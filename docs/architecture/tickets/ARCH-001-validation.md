# ARCH-001 validation evidence

Configured checkout snapshot, 2026-10-03. Counts describe these configurations,
not fixed coverage requirements. Existing uncommitted work was preserved.

## Inventories

| Configuration | Before | After | Removed registrations |
| --- | ---: | ---: | --- |
| `linux-vcpkg-headless` (Debug) | 9 | 56 | None |
| `linux-vcpkg` (Debug) | 65 | 65 | None |
| Clean headless Release | Not previously configured | 56 | Same portable inventory |

The original nine headless registrations were:

- `odai_harness_scenario_visual`
- `odai_headless_actor`
- `odai_headless_actor_replay`
- `odai_headless_inventory`
- `odai_headless_snapshots`
- `odai_headless_static_physics`
- `odai_headless_terrain_physics`
- `odai_headless_ui_inventory`
- `odai_headless_ui_replay`

The following 47 registrations became available in headless builds (one release
audit registration was already in source but absent from the old configured tree):

- `odai_animation_tests`
- `odai_asset_coverage_tests`
- `odai_bethesda_import_tests`
- `odai_bethesda_runtime_tests`
- `odai_bindless_slot_table_tests`
- `odai_cell_residency_planner_tests`
- `odai_character_coverage_tests`
- `odai_character_dynamics_tests`
- `odai_content_profile_tests`
- `odai_core_types_tests`
- `odai_dialogue_tests`
- `odai_engine_stats_tests`
- `odai_frame_graph_tests`
- `odai_gpu_arena_allocator_tests`
- `odai_humanoid_rig_tests`
- `odai_image_space_tests`
- `odai_imported_render_policy_tests`
- `odai_imported_scene_tests`
- `odai_iridius_bench_tests`
- `odai_job_system_tests`
- `odai_launcher_tests`
- `odai_living_world_tests`
- `odai_material_animation_tests`
- `odai_mod_check_tests`
- `odai_native_animation_tests`
- `odai_navigation_tests`
- `odai_nif_particle_tests`
- `odai_papyrus_vm_tests`
- `odai_pbr_material_tests`
- `odai_pose_graph_tests`
- `odai_route_checkpoint_tests`
- `odai_save_tests`
- `odai_skyrim_animation_tests`
- `odai_skyrim_inventory_tests`
- `odai_skyrim_persistent_reference_tests`
- `odai_skyrim_showcase_manifest_tests`
- `odai_taa_depth_tests`
- `odai_tes3_content_tests`
- `odai_tes3_navigation_simulation_tests`
- `odai_tes3_progression_tests`
- `odai_tes3_release_audit`
- `odai_tes3_runtime_tests`
- `odai_tes3_script_tests`
- `odai_traversal_state_tests`
- `odai_tri_morph_tests`
- `odai_upscaler_tests`
- `odai_world_map_cache_tests`

The unchanged full inventory is the 56 headless registrations above plus:

- `odai_actor_movement_tests`
- `odai_audio_tests`
- `odai_bench_heap_tests`
- `odai_bethesda_probe_mod_check_headless`
- `odai_mod_check_headless`
- `odai_pose_compute_tests`
- `odai_swf_font_tests`
- `odai_tes3_probe_records`
- `odai_ui_tests`

`odai_imported_scene_vulkan_smoke` remains built and conditionally registered when
`xvfb-run` is available. It was unavailable in both full-build snapshots here.
The pose-compute Vulkan test remains registered with skip return code 77.

## Prerequisites and boundaries

| Checks | Before | After |
| --- | --- | --- |
| Bindless slots, GPU arena bookkeeping, imported-render policy | Complete renderer, Vulkan/VMA, GLFW, ImGui, UI, shader compilation | Test source and project headers; no linked project library |
| Frame ordering | Complete renderer and its presentation/shader dependencies | Existing `frame_graph.cc`, `frame_graph_runtime.cc`, `odai_core`/Threads |
| Upscale policy/contract | Unavailable headless | `odai_upscale`; CPU policy/contract only |
| Import/simulation/animation/dialogue/navigation/core | Excluded by early headless return | Existing focused libraries; Jolt, nlohmann-json, zlib, Threads as applicable |

UI/font tests still need the Stb-backed UI library. Actor movement still compiles
audio decoding and links audio (Stb and optional miniaudio). Audio backend tests
retain the existing presentation configuration. Probe command tests and the heap
interposer test follow their existing tool/runtime target availability; this ticket
does not expand those targets into headless builds. Vulkan pose/smoke tests still
require graphics support and shader compilation. Optional XeSS SDK discovery is
skipped in headless builds; the full-build XeSS behavior is unchanged.

## Verification

- `cmake --preset linux-vcpkg-headless -DCMAKE_EXPORT_COMPILE_COMMANDS=ON`
- `cmake --build --preset linux-vcpkg-headless -j 4`
- `ctest --test-dir build-linux-headless --output-on-failure -j 4`: 56/56 passed.
- `cmake --preset linux-vcpkg -DCMAKE_EXPORT_COMPILE_COMMANDS=ON`
- `cmake --build build-linux -j 4`
- `ctest --test-dir build-linux --output-on-failure -j 4`: 65/65 passed.
- `python3 tools/ai/nav.py generate` and `python3 tools/ai/validate.py`: maps refreshed;
  8 checks passed, including a regression check that standalone CPU targets cannot
  reach renderer/UI/shader targets or Vulkan/GLFW/ImGui link fragments.

For the clean Release check, configure `linux-vcpkg-headless` with a fresh binary
and vcpkg-installed directory, `CMAKE_BUILD_TYPE=Release`, exported compile commands,
and `CMAKE_DISABLE_FIND_PACKAGE_<name>=TRUE` for Vulkan, VulkanMemoryAllocator,
glfw3, imgui and Stb. This run used `/tmp/arch-001/release` and
`/tmp/arch-001/portable-deps`, with the explicit local vcpkg toolchain path.
Only Jolt, nlohmann-json, zlib, vcpkg-cmake and vcpkg-cmake-config were installed.
No Slang shader target is configured. Build all targets and run all registered tests.

- `cmake --build /tmp/arch-001/release -j 4`: all portable targets built.
- `ctest --test-dir /tmp/arch-001/release --output-on-failure -j 4`: 56/56 passed.

All 43 portable C++ test translation units have `-DNDEBUG` followed by `-UNDEBUG`
in Release. All plain test translation units in the full build retain `-UNDEBUG`;
the shared helper still specifies `/UNDEBUG` on MSVC. Test source and assertions
were not weakened.

Raw before/after CTest JSON inventories are local at `/tmp/arch-001/`.
