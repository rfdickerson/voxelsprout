cmake_minimum_required(VERSION 3.21)
project(odai VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CTest)
include(CheckCXXCompilerFlag)
find_package(Threads REQUIRED)

option(ODAI_BUILD_RUNTIME "Build the Bethesda runtime." ON)
option(ODAI_BUILD_TOOLS "Build Bethesda content and inspection tools." ON)
option(ODAI_ENABLE_CCACHE "Use ccache when available." ON)
option(ODAI_ENABLE_LTO "Enable IPO/LTO for optimized configurations." OFF)
option(ODAI_ENABLE_NATIVE_ARCH "Tune optimized builds for this machine." OFF)
option(ODAI_ENABLE_XESS "Build the Intel XeSS backend when its Windows SDK is available." OFF)
set(ODAI_XESS_SDK_DIR "" CACHE PATH "Root of the Intel XeSS SDK.")

if(ODAI_ENABLE_CCACHE AND NOT CMAKE_CXX_COMPILER_LAUNCHER)
    find_program(ODAI_CCACHE_PROGRAM ccache)
    if(ODAI_CCACHE_PROGRAM)
        set(CMAKE_CXX_COMPILER_LAUNCHER "${ODAI_CCACHE_PROGRAM}")
        message(STATUS "odai: compiling through ccache (${ODAI_CCACHE_PROGRAM})")
    endif()
endif()

get_property(ODAI_IS_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT ODAI_IS_MULTI_CONFIG AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING
        "Build type: Debug, Release, RelWithDebInfo, or MinSizeRel." FORCE)
endif()

if(ODAI_ENABLE_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT ODAI_IPO_SUPPORTED OUTPUT ODAI_IPO_ERROR LANGUAGES CXX)
    if(ODAI_IPO_SUPPORTED)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO ON)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_MINSIZEREL ON)
    else()
        message(WARNING "odai: LTO unavailable: ${ODAI_IPO_ERROR}")
    endif()
endif()

if(ODAI_ENABLE_NATIVE_ARCH)
    if(MSVC)
        set(ODAI_NATIVE_ARCH_FLAG "/arch:AVX2")
    else()
        set(ODAI_NATIVE_ARCH_FLAG "-march=native")
    endif()
    check_cxx_compiler_flag("${ODAI_NATIVE_ARCH_FLAG}" ODAI_NATIVE_ARCH_FLAG_OK)
    if(ODAI_NATIVE_ARCH_FLAG_OK)
        add_compile_options("$<$<CONFIG:Release,RelWithDebInfo,MinSizeRel>:${ODAI_NATIVE_ARCH_FLAG}>")
    endif()
endif()

find_package(nlohmann_json CONFIG REQUIRED)
find_package(Jolt CONFIG REQUIRED)
find_package(ZLIB REQUIRED)
find_package(Stb REQUIRED)
find_package(Vulkan REQUIRED)
find_package(VulkanMemoryAllocator CONFIG REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_path(ODAI_MINIAUDIO_INCLUDE_DIR NAMES miniaudio.h)

set(ODAI_WARN_FLAGS_MSVC /W4 /permissive-)
function(odai_warnings target)
    if(MSVC)
        target_compile_options(${target} PRIVATE ${ODAI_WARN_FLAGS_MSVC})
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endfunction()

function(odai_source_includes target)
    target_include_directories(${target} PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>)
    target_compile_features(${target} PUBLIC cxx_std_20)
endfunction()

# Optional Intel XeSS support. Intel currently ships a Windows runtime only.
set(ODAI_XESS_FOUND OFF)
if(ODAI_ENABLE_XESS)
    find_path(ODAI_XESS_INCLUDE_DIR NAMES xess/xess.h
        HINTS "${ODAI_XESS_SDK_DIR}/inc" "${ODAI_XESS_SDK_DIR}/include")
    if(WIN32)
        find_library(ODAI_XESS_LIBRARY NAMES libxess xess
            HINTS "${ODAI_XESS_SDK_DIR}/lib" "${ODAI_XESS_SDK_DIR}/lib/x64")
    endif()
    if(ODAI_XESS_INCLUDE_DIR AND ODAI_XESS_LIBRARY)
        set(ODAI_XESS_FOUND ON)
    else()
        message(WARNING "XeSS requested but no supported SDK runtime was found; using temporal fallback.")
    endif()
endif()

# Build Slang directly from the retained source shaders. Outputs remain ignored
# generated files next to the sources because the runtime's development lookup
# resolves them there.
find_program(SLANGC_EXECUTABLE NAMES slangc
    HINTS "$ENV{VULKAN_SDK}/bin" "$ENV{VULKAN_SDK}/x86_64/bin")
if(SLANGC_EXECUTABLE)
    set(SLANG_SHARED_INCLUDES
        "${CMAKE_SOURCE_DIR}/src/render/shaders/camera_uniform.slang"
        "${CMAKE_SOURCE_DIR}/src/render/shaders/chunk_push_constants.slang"
        "${CMAKE_SOURCE_DIR}/src/render/shaders/fullscreen_triangle.slang"
        "${CMAKE_SOURCE_DIR}/src/render/shaders/imported_vertex_pack.slang"
        "${CMAKE_SOURCE_DIR}/src/render/shaders/noise.slang"
        "${CMAKE_SOURCE_DIR}/src/render/shaders/sh_lighting.slang"
        "${CMAKE_SOURCE_DIR}/src/render/shaders/voxel_decode.slang")
    set(SLANG_SHADER_OUTPUTS "")
    function(add_slang_shader shader stage)
        set(output "${shader}.spv")
        add_custom_command(OUTPUT "${output}"
            COMMAND "${SLANGC_EXECUTABLE}" -target spirv -entry main -stage "${stage}"
                -matrix-layout-column-major -I "${CMAKE_SOURCE_DIR}/src/render/shaders"
                ${ARGN} "${shader}" -o "${output}"
            DEPENDS "${shader}" ${SLANG_SHARED_INCLUDES}
            VERBATIM)
        list(APPEND SLANG_SHADER_OUTPUTS "${output}")
        set(SLANG_SHADER_OUTPUTS "${SLANG_SHADER_OUTPUTS}" PARENT_SCOPE)
    endfunction()
    function(add_slang_variant shader stage output)
        add_custom_command(OUTPUT "${output}"
            COMMAND "${SLANGC_EXECUTABLE}" -target spirv -entry main -stage "${stage}"
                -matrix-layout-column-major -I "${CMAKE_SOURCE_DIR}/src/render/shaders"
                ${ARGN} "${shader}" -o "${output}"
            DEPENDS "${shader}" ${SLANG_SHARED_INCLUDES}
            VERBATIM)
        list(APPEND SLANG_SHADER_OUTPUTS "${output}")
        set(SLANG_SHADER_OUTPUTS "${SLANG_SHADER_OUTPUTS}" PARENT_SCOPE)
    endfunction()

    # This list is reduced further when the imported-only renderer is pruned.
    add_slang_shader("${CMAKE_SOURCE_DIR}/src/render/shaders/voxel_packed.vert.slang" vertex)
    add_slang_shader("${CMAKE_SOURCE_DIR}/src/render/shaders/voxel_packed.frag.slang" fragment)
    add_slang_variant("${CMAKE_SOURCE_DIR}/src/render/shaders/voxel_packed.frag.slang" fragment
        "${CMAKE_SOURCE_DIR}/src/render/shaders/voxel_packed_rt.frag.slang.spv" -DODAI_RT_SHADOWS=1)
    add_slang_shader("${CMAKE_SOURCE_DIR}/src/render/shaders/voxel_normaldepth.frag.slang" fragment)
    add_slang_shader("${CMAKE_SOURCE_DIR}/src/render/shaders/terrain_heightmap.vert.slang" vertex)
    add_slang_shader("${CMAKE_SOURCE_DIR}/src/render/shaders/terrain_heightmap.tesc.slang" hull)
    add_slang_shader("${CMAKE_SOURCE_DIR}/src/render/shaders/terrain_heightmap.tese.slang" domain)
    add_slang_shader("${CMAKE_SOURCE_DIR}/src/render/shaders/terrain_heightmap.frag.slang" fragment)
    add_slang_shader("${CMAKE_SOURCE_DIR}/src/render/shaders/hex_terrain.vert.slang" vertex)
    add_slang_shader("${CMAKE_SOURCE_DIR}/src/render/shaders/hex_terrain.tesc.slang" hull)
    add_slang_shader("${CMAKE_SOURCE_DIR}/src/render/shaders/hex_terrain.tese.slang" domain)
    add_slang_shader("${CMAKE_SOURCE_DIR}/src/render/shaders/hex_terrain.frag.slang" fragment)
    foreach(name skybox.vert skybox.frag sky_cloud.vert sky_cloud.frag tone_map.vert tone_map.frag
                 ui.vert ui.frag shadow_depth.vert shadow_depth.frag pipe_shadow.vert pipe_shadow.frag
                 imported_static_shadow.vert imported_static_shadow.frag
                 pipe_instanced.vert pipe_instanced.frag imported_static.vert imported_static.frag
                 imported_fire_particle.vert imported_fire_particle.frag imported_water.vert
                 imported_water.frag imported_static_normaldepth.frag imported_static_depthonly.frag
                 imported_water_normaldepth.frag pipe_normaldepth.frag skinned_velocity.vert
                 skinned_velocity.frag)
        if(name MATCHES "\\.vert$")
            set(stage vertex)
        else()
            set(stage fragment)
        endif()
        add_slang_shader("${CMAKE_SOURCE_DIR}/src/render/shaders/${name}.slang" ${stage})
    endforeach()
    add_slang_shader("${CMAKE_SOURCE_DIR}/src/render/shaders/imported_terrain.tesc.slang" hull)
    add_slang_shader("${CMAKE_SOURCE_DIR}/src/render/shaders/imported_terrain.tese.slang" domain)
    add_slang_variant("${CMAKE_SOURCE_DIR}/src/render/shaders/imported_static.frag.slang" fragment
        "${CMAKE_SOURCE_DIR}/src/render/shaders/imported_static_rt.frag.slang.spv" -DODAI_RT_SHADOWS=1)
    add_slang_variant("${CMAKE_SOURCE_DIR}/src/render/shaders/imported_water.frag.slang" fragment
        "${CMAKE_SOURCE_DIR}/src/render/shaders/imported_water_rt.frag.slang.spv" -DODAI_RT_REFLECTIONS=1)
    foreach(name taa water_reflection_resolve light_cluster_cull contact_shadow_depth
                 contact_shadow_trace contact_shadow_resolve screen_space_gi ssao ssao_blur
                 xegtao_prefilter xegtao_main xegtao_denoise temporal_upscale voxel_gi_occupancy
                 voxel_gi_surface voxel_gi_sky_exposure voxel_gi_restir_candidate
                 voxel_gi_restir_temporal voxel_gi_restir_spatial voxel_gi_restir_resolve
                 voxel_gi_inject voxel_gi_propagate auto_exposure_histogram auto_exposure_update
                 sun_shafts skinning)
        add_slang_shader("${CMAKE_SOURCE_DIR}/src/render/shaders/${name}.comp.slang" compute)
    endforeach()
    add_slang_variant("${CMAKE_SOURCE_DIR}/src/render/shaders/ssao.comp.slang" compute
        "${CMAKE_SOURCE_DIR}/src/render/shaders/ssao_hbao.comp.slang.spv" -DODAI_AO_MODE=1)
    add_slang_variant("${CMAKE_SOURCE_DIR}/src/render/shaders/ssao.comp.slang" compute
        "${CMAKE_SOURCE_DIR}/src/render/shaders/ssao_gtao.comp.slang.spv" -DODAI_AO_MODE=2)
    add_slang_variant("${CMAKE_SOURCE_DIR}/src/render/shaders/voxel_gi_surface.comp.slang" compute
        "${CMAKE_SOURCE_DIR}/src/render/shaders/voxel_gi_surface_rt.comp.slang.spv" -DODAI_GI_RT_SURFACE=1)
    add_custom_target(slang_shaders DEPENDS ${SLANG_SHADER_OUTPUTS})
endif()

add_library(odai_core STATIC src/core/log.cc src/core/job_system.cc)
odai_source_includes(odai_core)
target_link_libraries(odai_core PUBLIC Threads::Threads)
odai_warnings(odai_core)

add_library(odai_ui STATIC
    src/ui/ui_draw_list.cc src/ui/ui_cursor.cc src/ui/font.cc src/ui/rich_text.cc
    src/ui/cached_rich_text.cc src/ui/ui_context.cc src/ui/icon_atlas.cc
    src/ui/toast_host.cc src/ui/tooltip.cc src/ui/signal.cc src/ui/resource_style.cc
    src/ui/widgets/panel.cc src/ui/widgets/label.cc src/ui/widgets/button.cc
    src/ui/widgets/window.cc src/ui/widgets/rich_text_view.cc src/ui/widgets/image.cc
    src/ui/widgets/progress_bar.cc src/ui/widgets/text_box.cc src/ui/widgets/toolbar.cc
    src/ui/widgets/stack_layout.cc src/ui/widgets/scroll_view.cc src/ui/widgets/slider.cc
    src/ui/widgets/toggle.cc src/ui/widgets/tab_bar.cc src/ui/widgets/dropdown.cc
    src/ui/widgets/radio_button.cc src/ui/widgets/spinner.cc src/ui/widgets/context_menu.cc
    src/ui/widgets/modal.cc src/ui/widgets/toast.cc src/ui/widgets/icon_button.cc
    src/ui/widgets/minimap_panel.cc src/ui/widgets/dialogue_panel.cc
    src/ui/widgets/event_tracker_panel.cc src/ui/widgets/faction_panel.cc
    src/ui/widgets/tes3_journal_panel.cc
    src/ui/widgets/grid_picker_panel.cc src/ui/widgets/resource_bar_panel.cc
    src/ui/widgets/selection_inspector_panel.cc)
odai_source_includes(odai_ui)
target_include_directories(odai_ui PRIVATE ${Stb_INCLUDE_DIR})
target_link_libraries(odai_ui PUBLIC odai_core nlohmann_json::nlohmann_json)
odai_warnings(odai_ui)

add_library(odai_audio STATIC src/audio/audio.cc src/audio/null_backend.cc
    src/audio/wav_writer.cc)
odai_source_includes(odai_audio)
target_link_libraries(odai_audio PUBLIC odai_core)
if(ODAI_MINIAUDIO_INCLUDE_DIR)
    target_sources(odai_audio PRIVATE src/audio/miniaudio_backend.cc src/audio/miniaudio_impl.cc)
    target_include_directories(odai_audio PRIVATE ${ODAI_MINIAUDIO_INCLUDE_DIR})
    target_compile_definitions(odai_audio PRIVATE ODAI_AUDIO_HAVE_MINIAUDIO)
    if(NOT WIN32 AND NOT APPLE)
        target_link_libraries(odai_audio PRIVATE Threads::Threads ${CMAKE_DL_LIBS})
    endif()
endif()
odai_warnings(odai_audio)

add_library(odai_anim STATIC src/anim/animation_sampler.cc src/anim/skeleton_io.cc
    src/anim/hkx_packfile.cc src/anim/skyrim_animation.cc)
odai_source_includes(odai_anim)
target_link_libraries(odai_anim PRIVATE nlohmann_json::nlohmann_json)
odai_warnings(odai_anim)

add_library(odai_dialogue STATIC src/dialogue/dialogue_io.cc src/dialogue/dialogue_runtime.cc
    src/dialogue/dialogue_state_io.cc)
odai_source_includes(odai_dialogue)
target_link_libraries(odai_dialogue PRIVATE nlohmann_json::nlohmann_json)
odai_warnings(odai_dialogue)

add_library(odai_upscale STATIC src/render/upscale/upscale_policy.cc
    src/render/upscale/upscale_contract.cc)
odai_source_includes(odai_upscale)
if(ODAI_XESS_FOUND)
    target_compile_definitions(odai_upscale PRIVATE ODAI_HAS_XESS=1)
    target_include_directories(odai_upscale PRIVATE ${ODAI_XESS_INCLUDE_DIR})
    target_link_libraries(odai_upscale PRIVATE ${ODAI_XESS_LIBRARY})
endif()
odai_warnings(odai_upscale)

add_library(odai_bethesda_import STATIC
    src/bethesda/condition.cc
    src/import/dds.cc src/import/imported_scene.cc src/import/imported_scene_query.cc
    src/import/cell_residency_planner.cc
    src/import/fnv/actor_records.cc src/import/fnv/asset_source.cc
    src/import/fnv/async_asset_loader.cc src/import/fnv/bsa_archive.cc
    src/import/fnv/cell_builder.cc src/import/fnv/character_builder.cc
    src/import/fnv/content_profile.cc src/import/fnv/content_record_index.cc
    src/import/fnv/decoded_texture_cache.cc src/import/fnv/dialogue_records.cc
    src/import/fnv/esm_reader.cc src/import/fnv/fallout_records.cc
    src/import/fnv/kf_animation.cc src/import/fnv/land_lod.cc src/import/fnv/lz4_frame.cc
    src/import/fnv/nif_scene.cc src/import/fnv/plugin_load_order.cc
    src/import/fnv/skyrim_animation_assets.cc
    src/import/fnv/strings_table.cc src/import/fnv/weather_records.cc)
odai_source_includes(odai_bethesda_import)
target_link_libraries(odai_bethesda_import PUBLIC odai_core odai_anim odai_dialogue
    ZLIB::ZLIB Jolt::Jolt nlohmann_json::nlohmann_json)
odai_warnings(odai_bethesda_import)

add_library(odai_bethesda_runtime STATIC
    src/bethesda/runtime_ids.cc src/bethesda/record_resolver.cc src/bethesda/tes3_content.cc
    src/bethesda/tes3_script.cc src/bethesda/tes3_runtime.cc
    src/bethesda/fixed_step_clock.cc
    src/bethesda/runtime_world.cc src/bethesda/vmad_reader.cc src/bethesda/skyrim_quest.cc
    src/bethesda/skyrim_dialogue.cc
    src/bethesda/skyrim_runtime_records.cc
    src/bethesda/skyrim_scenario_content.cc
    src/bethesda/pex_reader.cc src/bethesda/papyrus_vm.cc
    src/bethesda/scenario.cc src/bethesda/bethesda_session.cc
    src/bethesda/save_game.cc src/bethesda/bethesda_physics_world.cc)
odai_source_includes(odai_bethesda_runtime)
target_link_libraries(odai_bethesda_runtime PUBLIC
    odai_core odai_bethesda_import nlohmann_json::nlohmann_json)
odai_warnings(odai_bethesda_runtime)

# The Vulkan renderer is compiled once. Until the imported-only API pruning is
# complete, its old world helper sources remain here rather than being repeated
# by every executable.
add_library(odai_renderer STATIC
    src/render/frame_graph.cc src/render/renderer.cc src/render/video_writer.cc
    src/render/stb_image_impl.cc
    src/render/upscale/temporal_upscaler.cc src/render/upscale/vendor_backends.cc
    src/render/backend/vulkan/buffer_helpers.cc
    src/render/backend/vulkan/chunk_upload.cc src/render/backend/vulkan/contact_shadows.cc
    src/render/backend/vulkan/descriptor_buffer.cc src/render/backend/vulkan/descriptors.cc
    src/render/backend/vulkan/device.cc src/render/backend/vulkan/frame.cc
    src/render/backend/vulkan/frame_capture.cc src/render/backend/vulkan/frame_chunks.cc
    src/render/backend/vulkan/frame_draws.cc src/render/backend/vulkan/frame_graph_core.cc
    src/render/backend/vulkan/frame_graph_runtime.cc
    src/render/backend/vulkan/frame_pass_main.cc src/render/backend/vulkan/frame_pass_prepass.cc
    src/render/backend/vulkan/frame_pass_shadow.cc src/render/backend/vulkan/frame_pass_skinning.cc
    src/render/backend/vulkan/frame_pass_ssao.cc src/render/backend/vulkan/frame_pass_velocity.cc
    src/render/backend/vulkan/frame_run.cc src/render/backend/vulkan/frame_ui.cc
    src/render/backend/vulkan/init.cc src/render/backend/vulkan/init_resources.cc
    src/render/backend/vulkan/light_clusters.cc src/render/backend/vulkan/pass_gi.cc
    src/render/backend/vulkan/pass_pipelines.cc src/render/backend/vulkan/post_compute.cc
    src/render/backend/vulkan/screen_space_gi.cc src/render/backend/vulkan/shadow_culling.cc
    src/render/backend/vulkan/shadow_culling_utils.cc src/render/backend/vulkan/skinning_resources.cc
    src/render/backend/vulkan/swapchain_targets.cc src/render/backend/vulkan/ui_panels.cc
    src/render/backend/vulkan/ui_renderer.cc src/render/backend/vulkan/vma_usage.cc
    # PackedVoxelVertex and the legacy chunk entry points are still referenced
    # inside the Vulkan backend. The procedural world, clipmap, and MagicaVoxel
    # implementations are unreachable from Renderer/ImportedScene and are not
    # part of the Bethesda runtime target.
    src/world/chunk_mesher.cc)
odai_source_includes(odai_renderer)
target_include_directories(odai_renderer PRIVATE ${Stb_INCLUDE_DIR})
target_compile_definitions(odai_renderer PUBLIC ODAI_HAS_VULKAN=1)
target_link_libraries(odai_renderer PUBLIC odai_core odai_ui odai_upscale odai_bethesda_import
    Vulkan::Vulkan GPUOpen::VulkanMemoryAllocator imgui::imgui glfw)
if(ODAI_XESS_FOUND)
    target_compile_definitions(odai_renderer PRIVATE ODAI_HAS_XESS=1)
    target_include_directories(odai_renderer PRIVATE ${ODAI_XESS_INCLUDE_DIR})
    target_link_libraries(odai_renderer PRIVATE ${ODAI_XESS_LIBRARY})
endif()
if(TARGET slang_shaders)
    add_dependencies(odai_renderer slang_shaders)
endif()
odai_warnings(odai_renderer)

if(ODAI_BUILD_RUNTIME)
    add_library(odai_traversal STATIC
        src/games/newvegas/newvegas_traversal_state.cc)
    odai_source_includes(odai_traversal)
    target_link_libraries(odai_traversal PUBLIC nlohmann_json::nlohmann_json)
    odai_warnings(odai_traversal)

    add_executable(odai
        src/games/newvegas/newvegas_main.cc src/games/newvegas/bethesda_app.cc
        src/games/newvegas/bethesda_collision.cc src/bethesda/navigation_world.cc
        src/games/newvegas/newvegas_ogg.cc src/games/newvegas/newvegas_victor.cc
        src/games/newvegas/bethesda_actors.cc src/import/fnv/cell_streamer.cc
        src/engine/game_app.cc)
    target_include_directories(odai PRIVATE src ${Stb_INCLUDE_DIR})
    target_compile_definitions(odai PRIVATE ODAI_HAS_GLFW=1
        ODAI_PROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
    target_link_libraries(odai PRIVATE odai_renderer odai_audio odai_traversal odai_bethesda_runtime)
    odai_warnings(odai)
endif()

if(ODAI_BUILD_TOOLS)
    add_executable(odai_bethesda_probe src/tools/newvegas_probe_main.cc)
    target_link_libraries(odai_bethesda_probe PRIVATE odai_bethesda_import odai_bethesda_runtime)
    odai_source_includes(odai_bethesda_probe)
    odai_warnings(odai_bethesda_probe)

    add_executable(odai_newvegas_cooker src/tools/newvegas_cooker_main.cc)
    target_link_libraries(odai_newvegas_cooker PRIVATE odai_bethesda_import)
    odai_source_includes(odai_newvegas_cooker)
    odai_warnings(odai_newvegas_cooker)

    add_executable(odai_fnv_texture_pack src/tools/fnv_texture_pack_main.cc)
    target_link_libraries(odai_fnv_texture_pack PRIVATE odai_bethesda_import)
    odai_source_includes(odai_fnv_texture_pack)
    odai_warnings(odai_fnv_texture_pack)
endif()

if(BUILD_TESTING)
    function(odai_plain_test target source)
        add_executable(${target} ${source})
        odai_source_includes(${target})
        odai_warnings(${target})
        add_test(NAME ${target} COMMAND ${target})
    endfunction()

    odai_plain_test(odai_imported_scene_tests tests/imported_scene_tests.cc)
    target_link_libraries(odai_imported_scene_tests PRIVATE odai_bethesda_import)

    odai_plain_test(odai_fnv_import_tests tests/fnv_import_tests.cc)
    target_link_libraries(odai_fnv_import_tests PRIVATE odai_bethesda_import)

    odai_plain_test(odai_skyrim_animation_tests tests/skyrim_animation_tests.cc)
    target_link_libraries(odai_skyrim_animation_tests PRIVATE
        odai_anim odai_bethesda_import odai_bethesda_runtime)

    odai_plain_test(odai_content_profile_tests tests/content_profile_tests.cc)
    target_link_libraries(odai_content_profile_tests PRIVATE odai_bethesda_import)

    if(TARGET odai_traversal)
        odai_plain_test(odai_traversal_state_tests tests/traversal_state_tests.cc)
        target_link_libraries(odai_traversal_state_tests PRIVATE odai_traversal)
    endif()

    odai_plain_test(odai_animation_tests tests/animation_tests.cc)
    target_link_libraries(odai_animation_tests PRIVATE odai_anim)

    odai_plain_test(odai_dialogue_tests tests/dialogue_tests.cc)
    target_link_libraries(odai_dialogue_tests PRIVATE odai_dialogue)

    odai_plain_test(odai_audio_tests tests/audio_tests.cc)
    target_link_libraries(odai_audio_tests PRIVATE odai_audio)

    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        add_test(
            NAME odai_skyrim_showcase_manifest_tests
            COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_SOURCE_DIR}/scripts/render_skyrim_showcase.py
                --odai ${CMAKE_BINARY_DIR}/odai
                --data ${CMAKE_BINARY_DIR}/missing-skyrim-data
                --dry-run)
    endif()

    odai_plain_test(odai_upscaler_tests tests/upscaler_tests.cc)
    target_link_libraries(odai_upscaler_tests PRIVATE odai_upscale)

    odai_plain_test(odai_cell_residency_planner_tests tests/cell_residency_planner_tests.cc)
    target_link_libraries(odai_cell_residency_planner_tests PRIVATE odai_bethesda_import)

    odai_plain_test(odai_bindless_slot_table_tests tests/bindless_slot_table_tests.cc)
    target_link_libraries(odai_bindless_slot_table_tests PRIVATE odai_renderer)

    odai_plain_test(odai_gpu_arena_allocator_tests tests/gpu_arena_allocator_tests.cc)
    target_link_libraries(odai_gpu_arena_allocator_tests PRIVATE odai_renderer)

    odai_plain_test(odai_job_system_tests tests/job_system_tests.cc)
    target_link_libraries(odai_job_system_tests PRIVATE odai_core)

    odai_plain_test(odai_engine_stats_tests tests/engine_frame_stats_tests.cc)

    odai_plain_test(odai_core_types_tests tests/core_types_tests.cc)

    odai_plain_test(odai_bethesda_runtime_tests tests/bethesda_runtime_tests.cc)
    target_link_libraries(odai_bethesda_runtime_tests PRIVATE odai_bethesda_runtime)

    odai_plain_test(odai_tes3_content_tests tests/tes3_content_tests.cc)
    target_link_libraries(odai_tes3_content_tests PRIVATE odai_bethesda_runtime)

    odai_plain_test(odai_tes3_script_tests tests/tes3_script_tests.cc)
    target_link_libraries(odai_tes3_script_tests PRIVATE odai_bethesda_runtime)

    odai_plain_test(odai_tes3_runtime_tests tests/tes3_runtime_tests.cc)
    target_link_libraries(odai_tes3_runtime_tests PRIVATE odai_bethesda_runtime)

    odai_plain_test(odai_navigation_tests tests/navigation_world_tests.cc)
    target_sources(odai_navigation_tests PRIVATE
        src/bethesda/navigation_world.cc)
    target_link_libraries(odai_navigation_tests PRIVATE odai_bethesda_import)

    odai_plain_test(odai_tes3_navigation_simulation_tests
        tests/tes3_navigation_simulation_tests.cc)
    target_sources(odai_tes3_navigation_simulation_tests PRIVATE
        src/bethesda/navigation_world.cc
        src/bethesda/navigation_simulation.cc
        src/games/newvegas/bethesda_collision.cc)
    target_link_libraries(odai_tes3_navigation_simulation_tests PRIVATE
        odai_bethesda_runtime)

    odai_plain_test(odai_actor_movement_tests tests/bethesda_actor_movement_tests.cc)
    target_sources(odai_actor_movement_tests PRIVATE
        src/games/newvegas/bethesda_actors.cc
        src/bethesda/navigation_world.cc
        src/games/newvegas/newvegas_ogg.cc)
    target_include_directories(odai_actor_movement_tests PRIVATE ${Stb_INCLUDE_DIR})
    target_link_libraries(odai_actor_movement_tests PRIVATE
        odai_bethesda_import odai_bethesda_runtime odai_audio)

    odai_plain_test(odai_papyrus_vm_tests tests/papyrus_vm_tests.cc)
    target_link_libraries(odai_papyrus_vm_tests PRIVATE odai_bethesda_runtime)

    odai_plain_test(odai_save_tests tests/odai_save_tests.cc)
    target_link_libraries(odai_save_tests PRIVATE odai_bethesda_runtime)

    odai_plain_test(odai_ui_tests tests/ui_tests.cc)
    target_compile_definitions(odai_ui_tests PRIVATE ODAI_TEST_ASSETS_DIR="${CMAKE_SOURCE_DIR}/assets")
    target_link_libraries(odai_ui_tests PRIVATE odai_ui)

    odai_plain_test(odai_frame_graph_tests tests/frame_graph_plain_tests.cc)
    target_link_libraries(odai_frame_graph_tests PRIVATE odai_renderer)

    odai_plain_test(odai_imported_render_policy_tests tests/imported_render_policy_tests.cc)
    target_link_libraries(odai_imported_render_policy_tests PRIVATE odai_renderer)

    odai_plain_test(odai_pbr_material_tests tests/pbr_material_tests.cc)
    target_link_libraries(odai_pbr_material_tests PRIVATE odai_bethesda_import)

    add_executable(odai_imported_scene_vulkan_smoke tests/imported_scene_vulkan_smoke.cc)
    odai_source_includes(odai_imported_scene_vulkan_smoke)
    target_link_libraries(odai_imported_scene_vulkan_smoke PRIVATE odai_renderer)
    odai_warnings(odai_imported_scene_vulkan_smoke)
    find_program(XVFB_RUN_EXECUTABLE NAMES xvfb-run)
    set(ODAI_VULKAN_SMOKE_REGISTERED OFF)
    if(XVFB_RUN_EXECUTABLE)
        add_test(NAME odai_imported_scene_vulkan_smoke
            COMMAND "${XVFB_RUN_EXECUTABLE}" -a $<TARGET_FILE:odai_imported_scene_vulkan_smoke>)
        set(ODAI_VULKAN_SMOKE_REGISTERED ON)
    else()
        message(STATUS
            "Skipping imported-scene Vulkan smoke test: xvfb-run is unavailable")
    endif()
    if(ODAI_VULKAN_SMOKE_REGISTERED)
        set_tests_properties(odai_imported_scene_vulkan_smoke PROPERTIES
            # Shader development paths are rooted one level above the process CWD.
            # `tests/` is stable for both in-tree and out-of-tree build directories.
            WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/tests" TIMEOUT 60)
    endif()
endif()
