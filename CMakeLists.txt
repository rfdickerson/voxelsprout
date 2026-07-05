cmake_minimum_required(VERSION 3.21)
project(odai VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
include(CTest)
option(ODAI_BUILD_APP "Build the main Vulkan toy application." ON)
option(ODAI_BUILD_TOOLS "Build offline content tools." ON)
option(ODAI_BUILD_EXAMPLES "Build standalone odai_ui SDK integration samples." OFF)

# stb (stb_truetype.h, stb_rect_pack.h) for the UI font rasterizer. Provided by
# the vcpkg `stb` port in normal builds; ODAI_STB_INCLUDE_DIR overrides it for
# headless checks without vcpkg.
find_package(nlohmann_json CONFIG REQUIRED)
find_package(Stb QUIET)
if(Stb_FOUND)
    set(ODAI_STB_INCLUDE_DIRS "${Stb_INCLUDE_DIR}")
elseif(DEFINED ODAI_STB_INCLUDE_DIR)
    set(ODAI_STB_INCLUDE_DIRS "${ODAI_STB_INCLUDE_DIR}")
else()
    set(ODAI_STB_INCLUDE_DIRS "")
endif()

# nanosvg (nanosvg.h) for the SVG icon importer. Header-only like stb, so we just
# need its include dir; src/ui/vector/svg_document.cc guards its body with
# __has_include so odai_ui still builds (the importer degrades to a no-op) when
# nanosvg is absent. ODAI_NANOSVG_INCLUDE_DIR overrides for headless builds.
find_path(ODAI_NANOSVG_INCLUDE_DIR NAMES nanosvg.h nanosvg/nanosvg.h)
if(ODAI_NANOSVG_INCLUDE_DIR)
    set(ODAI_NANOSVG_INCLUDE_DIRS "${ODAI_NANOSVG_INCLUDE_DIR}")
    message(STATUS "odai_ui: nanosvg found (${ODAI_NANOSVG_INCLUDE_DIR})")
elseif(DEFINED ODAI_NANOSVG_INCLUDE_DIR_OVERRIDE)
    set(ODAI_NANOSVG_INCLUDE_DIRS "${ODAI_NANOSVG_INCLUDE_DIR_OVERRIDE}")
else()
    set(ODAI_NANOSVG_INCLUDE_DIRS "")
    message(STATUS "odai_ui: nanosvg.h not found - SVG importer compiles as a no-op")
endif()

# Optional Slang shader compilation (to SPIR-V for Vulkan).
# Produces .slang.spv files next to the source shaders.
find_program(SLANGC_EXECUTABLE NAMES slangc)
if(SLANGC_EXECUTABLE)
    set(SLANG_SHARED_INCLUDES
        "${CMAKE_SOURCE_DIR}/src/render/shaders/camera_uniform.slang"
        "${CMAKE_SOURCE_DIR}/src/render/shaders/chunk_push_constants.slang"
        "${CMAKE_SOURCE_DIR}/src/render/shaders/fullscreen_triangle.slang"
        "${CMAKE_SOURCE_DIR}/src/render/shaders/noise.slang"
        "${CMAKE_SOURCE_DIR}/src/render/shaders/sh_lighting.slang"
        "${CMAKE_SOURCE_DIR}/src/render/shaders/voxel_decode.slang"
    )
    set(SLANG_SHADER_OUTPUTS "")
    function(add_slang_shader SLANG_SHADER SLANG_STAGE)
        set(SLANG_OUTPUT "${SLANG_SHADER}.spv")
        add_custom_command(
            OUTPUT "${SLANG_OUTPUT}"
            COMMAND "${SLANGC_EXECUTABLE}"
                -target spirv
                -entry main
                -stage "${SLANG_STAGE}"
                -matrix-layout-column-major
                -I "${CMAKE_SOURCE_DIR}/src/render/shaders"
                ${ARGN}
                "${SLANG_SHADER}"
                -o "${SLANG_OUTPUT}"
            DEPENDS "${SLANG_SHADER}" ${SLANG_SHARED_INCLUDES}
            COMMENT "Compiling Slang shader ${SLANG_SHADER}"
            VERBATIM
        )
        list(APPEND SLANG_SHADER_OUTPUTS "${SLANG_OUTPUT}")
        set(SLANG_SHADER_OUTPUTS "${SLANG_SHADER_OUTPUTS}" PARENT_SCOPE)
    endfunction()
    function(add_slang_shader_variant SLANG_SHADER SLANG_STAGE SLANG_OUTPUT)
        add_custom_command(
            OUTPUT "${SLANG_OUTPUT}"
            COMMAND "${SLANGC_EXECUTABLE}"
                -target spirv
                -entry main
                -stage "${SLANG_STAGE}"
                -matrix-layout-column-major
                -I "${CMAKE_SOURCE_DIR}/src/render/shaders"
                ${ARGN}
                "${SLANG_SHADER}"
                -o "${SLANG_OUTPUT}"
            DEPENDS "${SLANG_SHADER}" ${SLANG_SHARED_INCLUDES}
            COMMENT "Compiling Slang shader variant ${SLANG_OUTPUT}"
            VERBATIM
        )
        list(APPEND SLANG_SHADER_OUTPUTS "${SLANG_OUTPUT}")
        set(SLANG_SHADER_OUTPUTS "${SLANG_SHADER_OUTPUTS}" PARENT_SCOPE)
    endfunction()

    add_slang_shader("${CMAKE_SOURCE_DIR}/src/render/shaders/voxel_packed.vert.slang" vertex)
    add_slang_shader("${CMAKE_SOURCE_DIR}/src/render/shaders/voxel_packed.frag.slang" fragment)
    add_slang_shader("${CMAKE_SOURCE_DIR}/src/render/shaders/terrain_heightmap.vert.slang" vertex)
    add_slang_shader("${CMAKE_SOURCE_DIR}/src/render/shaders/terrain_heightmap.tesc.slang" hull)
    add_slang_shader("${CMAKE_SOURCE_DIR}/src/render/shaders/terrain_heightmap.tese.slang" domain)
    add_slang_shader("${CMAKE_SOURCE_DIR}/src/render/shaders/terrain_heightmap.frag.slang" fragment)
    add_slang_shader("${CMAKE_SOURCE_DIR}/src/render/shaders/hex_terrain.vert.slang" vertex)
    add_slang_shader("${CMAKE_SOURCE_DIR}/src/render/shaders/hex_terrain.tesc.slang" hull)
    add_slang_shader("${CMAKE_SOURCE_DIR}/src/render/shaders/hex_terrain.tese.slang" domain)
    add_slang_shader("${CMAKE_SOURCE_DIR}/src/render/shaders/hex_terrain.frag.slang" fragment)
    add_slang_shader_variant(
        "${CMAKE_SOURCE_DIR}/src/render/shaders/voxel_packed.frag.slang"
        fragment
        "${CMAKE_SOURCE_DIR}/src/render/shaders/voxel_packed_rt.frag.slang.spv"
        -DODAI_RT_SHADOWS=1
    )
    add_slang_shader("${CMAKE_SOURCE_DIR}/src/render/shaders/voxel_normaldepth.frag.slang" fragment)
    add_slang_shader("${CMAKE_SOURCE_DIR}/src/render/shaders/skybox.vert.slang" vertex)
    add_slang_shader("${CMAKE_SOURCE_DIR}/src/render/shaders/skybox.frag.slang" fragment)
    add_slang_shader("${CMAKE_SOURCE_DIR}/src/render/shaders/sky_cloud.vert.slang" vertex)
    add_slang_shader("${CMAKE_SOURCE_DIR}/src/render/shaders/sky_cloud.frag.slang" fragment)
    add_slang_shader("${CMAKE_SOURCE_DIR}/src/render/shaders/tone_map.vert.slang" vertex)
    add_slang_shader("${CMAKE_SOURCE_DIR}/src/render/shaders/tone_map.frag.slang" fragment)
    add_slang_shader("${CMAKE_SOURCE_DIR}/src/render/shaders/ui.vert.slang" vertex)
    add_slang_shader("${CMAKE_SOURCE_DIR}/src/render/shaders/ui.frag.slang" fragment)
    add_slang_shader("${CMAKE_SOURCE_DIR}/src/render/shaders/shadow_depth.vert.slang" vertex)
    add_slang_shader("${CMAKE_SOURCE_DIR}/src/render/shaders/shadow_depth.frag.slang" fragment)
    add_slang_shader("${CMAKE_SOURCE_DIR}/src/render/shaders/pipe_shadow.vert.slang" vertex)
    add_slang_shader("${CMAKE_SOURCE_DIR}/src/render/shaders/pipe_shadow.frag.slang" fragment)
    add_slang_shader("${CMAKE_SOURCE_DIR}/src/render/shaders/imported_static_shadow.vert.slang" vertex)
    add_slang_shader("${CMAKE_SOURCE_DIR}/src/render/shaders/imported_static_shadow.frag.slang" fragment)
    add_slang_shader("${CMAKE_SOURCE_DIR}/src/render/shaders/grass_billboard_shadow.vert.slang" vertex)
    add_slang_shader("${CMAKE_SOURCE_DIR}/src/render/shaders/grass_billboard_shadow.frag.slang" fragment)
    add_slang_shader("${CMAKE_SOURCE_DIR}/src/render/shaders/pipe_instanced.vert.slang" vertex)
    add_slang_shader("${CMAKE_SOURCE_DIR}/src/render/shaders/pipe_instanced.frag.slang" fragment)
    add_slang_shader("${CMAKE_SOURCE_DIR}/src/render/shaders/imported_static.vert.slang" vertex)
    add_slang_shader("${CMAKE_SOURCE_DIR}/src/render/shaders/imported_static.frag.slang" fragment)
    add_slang_shader_variant(
        "${CMAKE_SOURCE_DIR}/src/render/shaders/imported_static.frag.slang"
        fragment
        "${CMAKE_SOURCE_DIR}/src/render/shaders/imported_static_rt.frag.slang.spv"
        -DODAI_RT_SHADOWS=1
    )
    add_slang_shader("${CMAKE_SOURCE_DIR}/src/render/shaders/imported_water.vert.slang" vertex)
    add_slang_shader("${CMAKE_SOURCE_DIR}/src/render/shaders/imported_water.frag.slang" fragment)
    add_slang_shader_variant(
        "${CMAKE_SOURCE_DIR}/src/render/shaders/imported_water.frag.slang"
        fragment
        "${CMAKE_SOURCE_DIR}/src/render/shaders/imported_water_rt.frag.slang.spv"
        -DODAI_RT_REFLECTIONS=1
    )
    add_slang_shader("${CMAKE_SOURCE_DIR}/src/render/shaders/imported_static_normaldepth.frag.slang" fragment)
    add_slang_shader("${CMAKE_SOURCE_DIR}/src/render/shaders/imported_water_normaldepth.frag.slang" fragment)
    add_slang_shader("${CMAKE_SOURCE_DIR}/src/render/shaders/grass_billboard.vert.slang" vertex)
    add_slang_shader("${CMAKE_SOURCE_DIR}/src/render/shaders/grass_billboard.frag.slang" fragment)
    add_slang_shader("${CMAKE_SOURCE_DIR}/src/render/shaders/grass_billboard_normaldepth.frag.slang" fragment)
    add_slang_shader("${CMAKE_SOURCE_DIR}/src/render/shaders/pipe_normaldepth.frag.slang" fragment)
    add_slang_shader("${CMAKE_SOURCE_DIR}/src/render/shaders/ssao.frag.slang" fragment)
    add_slang_shader("${CMAKE_SOURCE_DIR}/src/render/shaders/ssao_blur.frag.slang" fragment)
    add_slang_shader("${CMAKE_SOURCE_DIR}/src/render/shaders/voxel_gi_occupancy.comp.slang" compute)
    add_slang_shader("${CMAKE_SOURCE_DIR}/src/render/shaders/voxel_gi_surface.comp.slang" compute)
    add_slang_shader_variant(
        "${CMAKE_SOURCE_DIR}/src/render/shaders/voxel_gi_surface.comp.slang"
        compute
        "${CMAKE_SOURCE_DIR}/src/render/shaders/voxel_gi_surface_rt.comp.slang.spv"
        -DODAI_GI_RT_SURFACE=1
    )
    add_slang_shader("${CMAKE_SOURCE_DIR}/src/render/shaders/voxel_gi_sky_exposure.comp.slang" compute)
    add_slang_shader("${CMAKE_SOURCE_DIR}/src/render/shaders/voxel_gi_restir_candidate.comp.slang" compute)
    add_slang_shader("${CMAKE_SOURCE_DIR}/src/render/shaders/voxel_gi_restir_temporal.comp.slang" compute)
    add_slang_shader("${CMAKE_SOURCE_DIR}/src/render/shaders/voxel_gi_restir_spatial.comp.slang" compute)
    add_slang_shader("${CMAKE_SOURCE_DIR}/src/render/shaders/voxel_gi_restir_resolve.comp.slang" compute)
    add_slang_shader("${CMAKE_SOURCE_DIR}/src/render/shaders/voxel_gi_inject.comp.slang" compute)
    add_slang_shader("${CMAKE_SOURCE_DIR}/src/render/shaders/voxel_gi_propagate.comp.slang" compute)
    add_slang_shader("${CMAKE_SOURCE_DIR}/src/render/shaders/auto_exposure_histogram.comp.slang" compute)
    add_slang_shader("${CMAKE_SOURCE_DIR}/src/render/shaders/auto_exposure_update.comp.slang" compute)
    add_slang_shader("${CMAKE_SOURCE_DIR}/src/render/shaders/sun_shafts.comp.slang" compute)
    add_custom_target(slang_shaders DEPENDS ${SLANG_SHADER_OUTPUTS})
else()
    message(STATUS "slangc not found; skipping Slang shader target")
endif()

# ---------------------------------------------------------------------------
# odai_ui — headless, Vulkan-free UI library
# ---------------------------------------------------------------------------
# Compiling warning flags into a reusable list avoids repetition across targets.
set(ODAI_WARN_FLAGS_MSVC
    /W4 /permissive-
    /w14242 /w14254 /w14263 /w14265 /w14287 /we4289
    /w14296 /w14311 /w14545 /w14546 /w14547 /w14549
    /w14555 /w14619 /w14640 /w14826 /w14905 /w14906 /w14928
)

add_library(odai_ui STATIC
    src/core/log.cc
    src/ui/ui_draw_list.cc
    src/ui/font.cc
    src/ui/rich_text.cc
    src/ui/cached_rich_text.cc
    src/ui/ui_context.cc
    src/ui/icon_atlas.cc
    src/ui/tooltip.cc
    src/ui/theme/ui_theme.cc
    src/ui/signal.cc
    src/ui/document/ui_binding.cc
    src/ui/document/ui_document.cc
    src/ui/document/ui_hot_reload.cc
    src/ui/widgets/panel.cc
    src/ui/widgets/label.cc
    src/ui/widgets/button.cc
    src/ui/widgets/window.cc
    src/ui/widgets/rich_text_view.cc
    src/ui/widgets/image.cc
    src/ui/widgets/progress_bar.cc
    src/ui/widgets/text_box.cc
    src/ui/widgets/toolbar.cc
    src/ui/widgets/stack_layout.cc
    src/ui/widgets/scroll_view.cc
    src/ui/widgets/repeater.cc
    src/ui/widgets/slider.cc
    src/ui/widgets/toggle.cc
    src/ui/widgets/tab_bar.cc
    src/ui/widgets/dropdown.cc
    src/ui/widgets/radio_button.cc
    src/ui/widgets/spinner.cc
    src/ui/widgets/context_menu.cc
    src/ui/widgets/modal.cc
    src/ui/widgets/donut_chart.cc
    src/ui/widgets/toast.cc
    src/ui/widgets/line_chart.cc
    src/ui/widgets/stat_badge.cc
    src/ui/widgets/icon_button.cc
    src/ui/resource_style.cc
    src/ui/widgets/minimap_panel.cc
    # Renamed panels (genre-neutral names, data-agnostic, no game-type deps):
    src/ui/widgets/advisors_panel.cc
    src/ui/widgets/build_queue_panel.cc
    src/ui/widgets/event_tracker_panel.cc
    src/ui/widgets/faction_panel.cc
    src/ui/widgets/grid_picker_panel.cc
    src/ui/widgets/notable_entity_panel.cc
    src/ui/widgets/research_panel.cc
    src/ui/widgets/resource_bar_panel.cc
    src/ui/widgets/selection_inspector_panel.cc
    src/ui/widgets/sim_controls_panel.cc
    src/ui/vector/vector_path.cc
    src/ui/vector/vector_tessellator.cc
    src/ui/vector/vector_icon_registry.cc
    src/ui/vector/vector_cache_io.cc
    src/ui/vector/svg_document.cc
)
target_include_directories(odai_ui
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${ODAI_STB_INCLUDE_DIRS}
        ${ODAI_NANOSVG_INCLUDE_DIRS}
)
target_link_libraries(odai_ui PUBLIC nlohmann_json::nlohmann_json)
target_compile_features(odai_ui PUBLIC cxx_std_20)
if(MSVC)
    target_compile_options(odai_ui PRIVATE ${ODAI_WARN_FLAGS_MSVC})
else()
    target_compile_options(odai_ui PRIVATE -Wall -Wextra -Wpedantic)
endif()

# ---------------------------------------------------------------------------
# odai_audio — audio subsystem (miniaudio backend behind a PIMPL facade)
# ---------------------------------------------------------------------------
# Always built, even in headless tools/tests configs: the NullBackend has no
# third-party dependency and gives every consumer a valid, silent Audio. The
# real miniaudio backend is compiled only when the vcpkg miniaudio package is
# present, so a missing package degrades to silence rather than failing config.
add_library(odai_audio STATIC
    src/audio/audio.cc
    src/audio/null_backend.cc
)
target_include_directories(odai_audio
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:include>
)
target_compile_features(odai_audio PUBLIC cxx_std_20)

# The vcpkg miniaudio port is header-only and ships no CMake config package, so
# locate the header directly rather than via find_package. Header found -> the
# real backend is compiled; not found -> the silent NullBackend-only build.
find_path(ODAI_MINIAUDIO_INCLUDE_DIR NAMES miniaudio.h)
if(ODAI_MINIAUDIO_INCLUDE_DIR)
    target_sources(odai_audio PRIVATE
        src/audio/miniaudio_backend.cc
        src/audio/miniaudio_impl.cc
    )
    target_include_directories(odai_audio PRIVATE ${ODAI_MINIAUDIO_INCLUDE_DIR})
    target_compile_definitions(odai_audio PRIVATE ODAI_AUDIO_HAVE_MINIAUDIO)
    if(NOT WIN32 AND NOT APPLE)
        # miniaudio dlopen()s ALSA/PulseAudio at runtime; only threads + libdl to build.
        find_package(Threads REQUIRED)
        target_link_libraries(odai_audio PRIVATE Threads::Threads ${CMAKE_DL_LIBS})
    endif()
    message(STATUS "odai_audio: miniaudio backend enabled (${ODAI_MINIAUDIO_INCLUDE_DIR})")
else()
    message(STATUS "odai_audio: miniaudio.h not found - building silent NullBackend only")
endif()

# Project warning flags apply to our own sources only; the miniaudio
# implementation TU keeps compiler defaults (mirrors vma_usage.cc).
if(MSVC)
    set_source_files_properties(
        src/audio/audio.cc src/audio/null_backend.cc src/audio/miniaudio_backend.cc
        PROPERTIES COMPILE_OPTIONS "${ODAI_WARN_FLAGS_MSVC}")
else()
    set_source_files_properties(
        src/audio/audio.cc src/audio/null_backend.cc src/audio/miniaudio_backend.cc
        PROPERTIES COMPILE_OPTIONS "-Wall;-Wextra;-Wpedantic")
endif()

# ---------------------------------------------------------------------------
# odai_content — data-driven game content (JSON). Vulkan-free, app-free, so the
# game core, tools, and tests can all load mods/base/data/*.json. The game
# accessors (techTree, buildingDefs, getPediaArticle, balance, ...) delegate to
# the active ContentDatabase served by this library.
# ---------------------------------------------------------------------------
add_library(odai_content STATIC
    src/content/content_database.cc
    src/content/content_loader.cc
    src/content/content_paths.cc
    src/content/mod_loader.cc
)
target_include_directories(odai_content PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(odai_content PUBLIC nlohmann_json::nlohmann_json)
target_compile_features(odai_content PUBLIC cxx_std_20)
# So tools/tests resolve mods/base regardless of working directory (mirrors odai).
target_compile_definitions(odai_content PRIVATE ODAI_PROJECT_SOURCE_DIR=\"${CMAKE_SOURCE_DIR}\")
if(MSVC)
    target_compile_options(odai_content PRIVATE ${ODAI_WARN_FLAGS_MSVC})
else()
    target_compile_options(odai_content PRIVATE -Wall -Wextra -Wpedantic)
endif()

# ---------------------------------------------------------------------------
# odai_script — Lua scripting engine (sol2). The ONLY place Lua is included.
# Implements odai::game::IModHost (src/game/mod_host.h) so mods attach to gameplay
# events and author wonder/building effects in Lua. Targets that must stay Lua-free
# and deterministic (headless tools, content/economy tests) do NOT link this; they
# run on the no-op NullModHost. Relaxed warning flags: sol2's headers are heavily
# templated and not /W4-clean, but no warnings-as-errors are in play.
# ---------------------------------------------------------------------------
find_package(Lua REQUIRED)
find_package(sol2 CONFIG REQUIRED)
add_library(odai_script STATIC
    src/script/script_engine.cc
    src/script/lua_bindings.cc
    src/script/lua_api.cc
    src/script/lua_mod_host.cc
)
target_include_directories(odai_script
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${LUA_INCLUDE_DIR}
)
target_link_libraries(odai_script PUBLIC odai_content)
target_link_libraries(odai_script PRIVATE sol2 ${LUA_LIBRARIES})
target_compile_features(odai_script PUBLIC cxx_std_20)
target_compile_definitions(odai_script PRIVATE ODAI_PROJECT_SOURCE_DIR=\"${CMAKE_SOURCE_DIR}\")

if(ODAI_BUILD_APP)
    add_executable(odai
        src/app/main.cc
        src/app/app.cc
        src/game/advisor.cc
        src/game/buildable.cc
        src/game/economy.cc
        src/game/game_sim.cc
        src/game/great_people.cc
        src/game/mod_host.cc
        src/game/religion.cc
        src/game/strategy_map.cc
        src/game/strategy_map_io.cc
        src/game/strategy_map_mesh.cc
        src/game/strategy_hex_terrain.cc
        src/game/units.cc
        src/game/ai_units.cc
        src/import/dds.cc
        src/import/gpu_scene.cc
        src/import/imported_scene.cc
        src/render/frame_graph.cc
        src/render/renderer.cc
        src/world/magica_voxel.cc
        src/world/world.cc
        src/world/chunk_grid_worldgen.cc
        src/world/chunk_mesher.cc
        src/world/clipmap_index.cc
    )
    set(ODAI_RENDER_BACKEND "VULKAN" CACHE STRING "Render backend to build. Supported values: VULKAN, DX12, METAL.")
    set_property(CACHE ODAI_RENDER_BACKEND PROPERTY STRINGS VULKAN DX12 METAL)
    string(TOUPPER "${ODAI_RENDER_BACKEND}" ODAI_RENDER_BACKEND_UPPER)

    set(RENDER_BACKEND_SOURCES)

    if(ODAI_RENDER_BACKEND_UPPER STREQUAL "VULKAN")
        # ---------------------------------------------------------------------------
        # odai_ui_vulkan — Vulkan rendering backend for the UI library
        # ---------------------------------------------------------------------------
        find_package(Vulkan REQUIRED)
        find_package(VulkanMemoryAllocator CONFIG REQUIRED)
        add_library(odai_ui_vulkan STATIC
            src/render/backend/vulkan/ui_renderer.cc
            src/render/backend/vulkan/frame_ui.cc
            # buffer_helpers.cc defines BufferAllocator/FrameArena, which
            # ui_renderer.h's public API (UiRenderer::InitInfo, uploadGeometry)
            # depends on directly — it belongs here, not in the app's own
            # RENDER_BACKEND_SOURCES, so a consumer linking only odai_ui_vulkan
            # gets a complete, self-sufficient SDK with no missing symbols.
            src/render/backend/vulkan/buffer_helpers.cc
        )
        target_include_directories(odai_ui_vulkan PUBLIC
            $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
            $<INSTALL_INTERFACE:include>
        )
        target_link_libraries(odai_ui_vulkan
            # PUBLIC: ui_renderer.h is an installed public header that directly
            # includes <vulkan/vulkan.h> and <vk_mem_alloc.h> and exposes VkDevice /
            # VmaAllocator in InitInfo, so consumers need these propagated to compile.
            PUBLIC odai_ui Vulkan::Vulkan GPUOpen::VulkanMemoryAllocator
        )
        if(MSVC)
            target_compile_options(odai_ui_vulkan PRIVATE ${ODAI_WARN_FLAGS_MSVC})
        else()
            target_compile_options(odai_ui_vulkan PRIVATE -Wall -Wextra -Wpedantic)
        endif()

        if(ODAI_BUILD_EXAMPLES)
            add_subdirectory(examples/vulkan_ui_integration)
            add_subdirectory(examples/ui_stress_test/harness)
            add_subdirectory(examples/ui_stress_test/smoke_test)
        endif()

        list(APPEND RENDER_BACKEND_SOURCES
            src/render/backend/vulkan/device.cc
            src/render/backend/vulkan/frame.cc
            src/render/backend/vulkan/frame_run.cc
            src/render/backend/vulkan/frame_graph_core.cc
            src/render/backend/vulkan/frame_graph_runtime.cc
            src/render/backend/vulkan/frame_pass_shadow.cc
            src/render/backend/vulkan/frame_pass_prepass.cc
            src/render/backend/vulkan/frame_pass_ssao.cc
            src/render/backend/vulkan/frame_pass_main.cc
            src/render/backend/vulkan/frame_chunks.cc
            src/render/backend/vulkan/frame_draws.cc
            src/render/backend/vulkan/frame_instances.cc
            src/render/backend/vulkan/init.cc
            src/render/backend/vulkan/init_resources.cc
            src/render/backend/vulkan/chunk_upload.cc
            src/render/backend/vulkan/shadow_culling_utils.cc
            src/render/backend/vulkan/descriptors.cc
            src/render/backend/vulkan/ui_panels.cc
            src/render/backend/vulkan/pass_pipelines.cc
            src/render/backend/vulkan/swapchain_targets.cc
            src/render/backend/vulkan/post_compute.cc
            src/render/backend/vulkan/pass_gi.cc
            src/render/backend/vulkan/shadow_culling.cc
            src/render/backend/vulkan/vma_usage.cc
        )
        target_compile_definitions(odai PRIVATE ODAI_RENDER_BACKEND_VULKAN=1)
        target_compile_definitions(odai PRIVATE ODAI_HAS_VULKAN=1)
        set(VULKAN_RENDER_BACKEND "ON")
    elseif(ODAI_RENDER_BACKEND_UPPER STREQUAL "DX12")
        message(FATAL_ERROR "DX12 backend is not implemented yet.")
    elseif(ODAI_RENDER_BACKEND_UPPER STREQUAL "METAL")
        message(FATAL_ERROR "Metal backend is not implemented yet.")
    else()
        message(
            FATAL_ERROR
            "Unsupported ODAI_RENDER_BACKEND value '${ODAI_RENDER_BACKEND}'."
            " Expected VULKAN, DX12, or METAL."
        )
    endif()

    target_sources(odai PRIVATE ${RENDER_BACKEND_SOURCES})

    if(TARGET slang_shaders)
        add_dependencies(odai slang_shaders)
    endif()

    target_include_directories(odai PRIVATE src ${ODAI_STB_INCLUDE_DIRS})
    target_compile_definitions(
        odai PRIVATE
        ODAI_PROJECT_SOURCE_DIR=\"${CMAKE_SOURCE_DIR}\"
        ODAI_APP_VERSION=\"${PROJECT_VERSION}\"
        ODAI_RELEASE_PROFILE=\"windows_runnable_rt_beta\"
    )

    if(MSVC)
        target_compile_options(odai PRIVATE ${ODAI_WARN_FLAGS_MSVC})
    else()
        target_compile_options(odai PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    # UI libraries (Vulkan + VMA already found in the VULKAN block above)
    target_link_libraries(odai PRIVATE odai_ui)
    target_link_libraries(odai PRIVATE odai_content)
    target_link_libraries(odai PRIVATE odai_script)
    target_link_libraries(odai PRIVATE odai_audio)
    if(VULKAN_RENDER_BACKEND)
        target_link_libraries(odai PRIVATE odai_ui_vulkan)
    endif()

    find_package(glfw3 CONFIG QUIET)

    if(VULKAN_RENDER_BACKEND)
        find_package(imgui CONFIG REQUIRED)
        target_link_libraries(odai PRIVATE Vulkan::Vulkan)
        target_link_libraries(odai PRIVATE GPUOpen::VulkanMemoryAllocator)
        target_link_libraries(odai PRIVATE imgui::imgui)
    endif()

    if(TARGET glfw)
        target_link_libraries(odai PRIVATE glfw)
        target_compile_definitions(odai PRIVATE ODAI_HAS_GLFW=1)
    elseif(TARGET glfw3::glfw)
        target_link_libraries(odai PRIVATE glfw3::glfw)
        target_compile_definitions(odai PRIVATE ODAI_HAS_GLFW=1)
    else()
        find_package(PkgConfig QUIET)
        if(PkgConfig_FOUND)
            pkg_check_modules(GLFW3 QUIET IMPORTED_TARGET glfw3)
            if(TARGET PkgConfig::GLFW3)
                target_link_libraries(odai PRIVATE PkgConfig::GLFW3)
                target_compile_definitions(odai PRIVATE ODAI_HAS_GLFW=1)
            else()
                message(FATAL_ERROR "GLFW is required. Install via vcpkg or provide glfw3 to CMake.")
            endif()
        else()
            message(FATAL_ERROR "GLFW is required. Install via vcpkg or provide glfw3 to CMake.")
        endif()
    endif()

    # ---------------------------------------------------------------------------
    # odai_game_stellaris — Stellaris-style space 4X (Vulkan, GPU-rendered)
    # ---------------------------------------------------------------------------
    if(VULKAN_RENDER_BACKEND)
        add_executable(odai_game_stellaris
            src/games/stellaris/stellaris_main.cc
            src/games/stellaris/stellaris_app.cc
            src/engine/game_app.cc
            src/import/dds.cc
            src/import/gpu_scene.cc
            src/import/imported_scene.cc
            src/render/frame_graph.cc
            src/render/renderer.cc
            src/world/world.cc
            src/world/chunk_grid_worldgen.cc
            src/world/chunk_mesher.cc
            src/world/clipmap_index.cc
            src/world/magica_voxel.cc
            ${RENDER_BACKEND_SOURCES}
        )
        target_include_directories(odai_game_stellaris PRIVATE src ${ODAI_STB_INCLUDE_DIRS})
        target_compile_definitions(odai_game_stellaris PRIVATE
            ODAI_RENDER_BACKEND_VULKAN=1
            ODAI_HAS_VULKAN=1
            ODAI_HAS_GLFW=1
            ODAI_PROJECT_SOURCE_DIR=\"${CMAKE_SOURCE_DIR}\"
        )
        target_link_libraries(odai_game_stellaris PRIVATE
            odai_ui
            odai_ui_vulkan
            odai_audio
            Vulkan::Vulkan
            GPUOpen::VulkanMemoryAllocator
            imgui::imgui
        )
        if(TARGET glfw)
            target_link_libraries(odai_game_stellaris PRIVATE glfw)
        elseif(TARGET glfw3::glfw)
            target_link_libraries(odai_game_stellaris PRIVATE glfw3::glfw)
        endif()
        if(TARGET slang_shaders)
            add_dependencies(odai_game_stellaris slang_shaders)
        endif()
        if(MSVC)
            target_compile_options(odai_game_stellaris PRIVATE ${ODAI_WARN_FLAGS_MSVC})
        else()
            target_compile_options(odai_game_stellaris PRIVATE -Wall -Wextra -Wpedantic)
        endif()

    # ---------------------------------------------------------------------------
    # odai_game_swtor — Star Wars: The Old Republic mock MMO UI (Vulkan)
    # ---------------------------------------------------------------------------
        add_executable(odai_game_swtor
            src/games/swtor/swtor_main.cc
            src/games/swtor/swtor_app.cc
            src/engine/game_app.cc
            src/import/dds.cc
            src/import/gpu_scene.cc
            src/import/imported_scene.cc
            src/render/frame_graph.cc
            src/render/renderer.cc
            src/world/world.cc
            src/world/chunk_grid_worldgen.cc
            src/world/chunk_mesher.cc
            src/world/clipmap_index.cc
            src/world/magica_voxel.cc
            ${RENDER_BACKEND_SOURCES}
        )
        target_include_directories(odai_game_swtor PRIVATE src ${ODAI_STB_INCLUDE_DIRS})
        target_compile_definitions(odai_game_swtor PRIVATE
            ODAI_RENDER_BACKEND_VULKAN=1
            ODAI_HAS_VULKAN=1
            ODAI_HAS_GLFW=1
            ODAI_PROJECT_SOURCE_DIR=\"${CMAKE_SOURCE_DIR}\"
        )
        target_link_libraries(odai_game_swtor PRIVATE
            odai_ui odai_ui_vulkan odai_audio
            Vulkan::Vulkan GPUOpen::VulkanMemoryAllocator imgui::imgui
        )
        if(TARGET glfw)
            target_link_libraries(odai_game_swtor PRIVATE glfw)
        elseif(TARGET glfw3::glfw)
            target_link_libraries(odai_game_swtor PRIVATE glfw3::glfw)
        endif()
        if(TARGET slang_shaders)
            add_dependencies(odai_game_swtor slang_shaders)
        endif()
        if(MSVC)
            target_compile_options(odai_game_swtor PRIVATE ${ODAI_WARN_FLAGS_MSVC})
        else()
            target_compile_options(odai_game_swtor PRIVATE -Wall -Wextra -Wpedantic)
        endif()

    # ---------------------------------------------------------------------------
    # odai_game_citybuilder — SimCity-style city builder (Vulkan)
    # ---------------------------------------------------------------------------
        add_executable(odai_game_citybuilder
            src/games/citybuilder/citybuilder_main.cc
            src/games/citybuilder/citybuilder_app.cc
            src/engine/game_app.cc
            src/import/dds.cc
            src/import/gpu_scene.cc
            src/import/imported_scene.cc
            src/render/frame_graph.cc
            src/render/renderer.cc
            src/world/world.cc
            src/world/chunk_grid_worldgen.cc
            src/world/chunk_mesher.cc
            src/world/clipmap_index.cc
            src/world/magica_voxel.cc
            ${RENDER_BACKEND_SOURCES}
        )
        target_include_directories(odai_game_citybuilder PRIVATE src ${ODAI_STB_INCLUDE_DIRS})
        target_compile_definitions(odai_game_citybuilder PRIVATE
            ODAI_RENDER_BACKEND_VULKAN=1
            ODAI_HAS_VULKAN=1
            ODAI_HAS_GLFW=1
            ODAI_PROJECT_SOURCE_DIR=\"${CMAKE_SOURCE_DIR}\"
        )
        target_link_libraries(odai_game_citybuilder PRIVATE
            odai_ui odai_ui_vulkan odai_audio
            Vulkan::Vulkan GPUOpen::VulkanMemoryAllocator imgui::imgui
        )
        if(TARGET glfw)
            target_link_libraries(odai_game_citybuilder PRIVATE glfw)
        elseif(TARGET glfw3::glfw)
            target_link_libraries(odai_game_citybuilder PRIVATE glfw3::glfw)
        endif()
        if(TARGET slang_shaders)
            add_dependencies(odai_game_citybuilder slang_shaders)
        endif()
        if(MSVC)
            target_compile_options(odai_game_citybuilder PRIVATE ${ODAI_WARN_FLAGS_MSVC})
        else()
            target_compile_options(odai_game_citybuilder PRIVATE -Wall -Wextra -Wpedantic)
        endif()

    # ---------------------------------------------------------------------------
    # odai_game_snake — Snake / Nibbles (Vulkan)
    # ---------------------------------------------------------------------------
        add_executable(odai_game_snake
            src/games/snake/snake_main.cc
            src/games/snake/snake_app.cc
            src/engine/game_app.cc
            src/import/dds.cc
            src/import/gpu_scene.cc
            src/import/imported_scene.cc
            src/render/frame_graph.cc
            src/render/renderer.cc
            src/world/world.cc
            src/world/chunk_grid_worldgen.cc
            src/world/chunk_mesher.cc
            src/world/clipmap_index.cc
            src/world/magica_voxel.cc
            ${RENDER_BACKEND_SOURCES}
        )
        target_include_directories(odai_game_snake PRIVATE src ${ODAI_STB_INCLUDE_DIRS})
        target_compile_definitions(odai_game_snake PRIVATE
            ODAI_RENDER_BACKEND_VULKAN=1
            ODAI_HAS_VULKAN=1
            ODAI_HAS_GLFW=1
            ODAI_PROJECT_SOURCE_DIR=\"${CMAKE_SOURCE_DIR}\"
        )
        target_link_libraries(odai_game_snake PRIVATE
            odai_ui odai_ui_vulkan odai_audio
            Vulkan::Vulkan GPUOpen::VulkanMemoryAllocator imgui::imgui
        )
        if(TARGET glfw)
            target_link_libraries(odai_game_snake PRIVATE glfw)
        elseif(TARGET glfw3::glfw)
            target_link_libraries(odai_game_snake PRIVATE glfw3::glfw)
        endif()
        if(TARGET slang_shaders)
            add_dependencies(odai_game_snake slang_shaders)
        endif()
        if(MSVC)
            target_compile_options(odai_game_snake PRIVATE ${ODAI_WARN_FLAGS_MSVC})
        else()
            target_compile_options(odai_game_snake PRIVATE -Wall -Wextra -Wpedantic)
        endif()

    # ---------------------------------------------------------------------------
    # odai_ui_editor — Drag-and-drop UI editor for artists (Vulkan)
    # ---------------------------------------------------------------------------
        add_executable(odai_ui_editor
            src/tools/ui_editor/ui_editor_main.cc
            src/tools/ui_editor/ui_editor_app.cc
            src/engine/game_app.cc
            src/import/dds.cc
            src/import/gpu_scene.cc
            src/import/imported_scene.cc
            src/render/frame_graph.cc
            src/render/renderer.cc
            src/world/world.cc
            src/world/chunk_grid_worldgen.cc
            src/world/chunk_mesher.cc
            src/world/clipmap_index.cc
            src/world/magica_voxel.cc
            ${RENDER_BACKEND_SOURCES}
        )
        target_include_directories(odai_ui_editor PRIVATE src ${ODAI_STB_INCLUDE_DIRS})
        target_compile_definitions(odai_ui_editor PRIVATE
            ODAI_RENDER_BACKEND_VULKAN=1
            ODAI_HAS_VULKAN=1
            ODAI_HAS_GLFW=1
            ODAI_PROJECT_SOURCE_DIR=\"${CMAKE_SOURCE_DIR}\"
        )
        target_link_libraries(odai_ui_editor PRIVATE
            odai_ui odai_ui_vulkan odai_audio
            Vulkan::Vulkan GPUOpen::VulkanMemoryAllocator imgui::imgui
        )
        if(TARGET glfw)
            target_link_libraries(odai_ui_editor PRIVATE glfw)
        elseif(TARGET glfw3::glfw)
            target_link_libraries(odai_ui_editor PRIVATE glfw3::glfw)
        endif()
        if(TARGET slang_shaders)
            add_dependencies(odai_ui_editor slang_shaders)
        endif()
        if(MSVC)
            target_compile_options(odai_ui_editor PRIVATE ${ODAI_WARN_FLAGS_MSVC} /FS)
        else()
            target_compile_options(odai_ui_editor PRIVATE -Wall -Wextra -Wpedantic)
        endif()

    # ---------------------------------------------------------------------------
    # odai_retro_theme_demo — Windows 95 / Motif-CDE theme showcase (Vulkan)
    # ---------------------------------------------------------------------------
        add_executable(odai_retro_theme_demo
            src/tools/retro_theme_demo/retro_theme_demo_main.cc
            src/tools/retro_theme_demo/retro_theme_demo_app.cc
            src/tools/retro_theme_demo/retro_widgets.cc
            src/engine/game_app.cc
            src/import/dds.cc
            src/import/gpu_scene.cc
            src/import/imported_scene.cc
            src/render/frame_graph.cc
            src/render/renderer.cc
            src/world/world.cc
            src/world/chunk_grid_worldgen.cc
            src/world/chunk_mesher.cc
            src/world/clipmap_index.cc
            src/world/magica_voxel.cc
            ${RENDER_BACKEND_SOURCES}
        )
        target_include_directories(odai_retro_theme_demo PRIVATE src ${ODAI_STB_INCLUDE_DIRS})
        target_compile_definitions(odai_retro_theme_demo PRIVATE
            ODAI_RENDER_BACKEND_VULKAN=1
            ODAI_HAS_VULKAN=1
            ODAI_HAS_GLFW=1
            ODAI_PROJECT_SOURCE_DIR=\"${CMAKE_SOURCE_DIR}\"
        )
        target_link_libraries(odai_retro_theme_demo PRIVATE
            odai_ui odai_ui_vulkan odai_audio
            Vulkan::Vulkan GPUOpen::VulkanMemoryAllocator imgui::imgui
        )
        if(TARGET glfw)
            target_link_libraries(odai_retro_theme_demo PRIVATE glfw)
        elseif(TARGET glfw3::glfw)
            target_link_libraries(odai_retro_theme_demo PRIVATE glfw3::glfw)
        endif()
        if(TARGET slang_shaders)
            add_dependencies(odai_retro_theme_demo slang_shaders)
        endif()
        if(MSVC)
            target_compile_options(odai_retro_theme_demo PRIVATE ${ODAI_WARN_FLAGS_MSVC})
        else()
            target_compile_options(odai_retro_theme_demo PRIVATE -Wall -Wextra -Wpedantic)
        endif()
    endif()
endif()

# ---------------------------------------------------------------------------
# Install / CMake package export for odai_ui and odai_ui_vulkan
# ---------------------------------------------------------------------------
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(ODAI_UI_INSTALL_TARGETS odai_ui)
set(ODAI_UI_HAS_VULKAN FALSE)
if(TARGET odai_ui_vulkan)
    list(APPEND ODAI_UI_INSTALL_TARGETS odai_ui_vulkan)
    set(ODAI_UI_HAS_VULKAN TRUE)
endif()

install(TARGETS ${ODAI_UI_INSTALL_TARGETS}
    EXPORT odai_ui_targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(DIRECTORY src/ui/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/odai_ui
    FILES_MATCHING PATTERN "*.h"
)
install(DIRECTORY src/core/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/odai_ui/core
    FILES_MATCHING PATTERN "log.h"
)
install(EXPORT odai_ui_targets
    FILE odai_ui-targets.cmake
    NAMESPACE odai::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/odai_ui
)
configure_package_config_file(
    cmake/odai_ui-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/odai_ui-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/odai_ui
)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/odai_ui-config.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/odai_ui
)

if(ODAI_BUILD_TOOLS)
    # Offline generator for the strategic-map prototype. Writes a native .smap and
    # a renderable .bin ImportedScene.
    add_executable(odai_strategy_map_gen
        src/game/strategy_map.cc
        src/game/strategy_map_io.cc
        src/game/strategy_map_mesh.cc
        src/import/gpu_scene.cc
        src/import/imported_scene.cc
        src/tools/strategy_map_gen_main.cc
    )
    target_include_directories(odai_strategy_map_gen PRIVATE src)
    if(MSVC)
        target_compile_options(odai_strategy_map_gen PRIVATE /W4 /permissive-)
    else()
        target_compile_options(odai_strategy_map_gen PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    # Headless strategic-economy playtest harness. Pure CPU (no Vulkan): runs a
    # whole multi-empire match for N turns and prints fun-factor metrics.
    #   odai_civ_sim [turns] [seed] [empires] [--quiet|--sweep N]
    add_executable(odai_civ_sim
        src/game/strategy_map.cc
        src/game/economy.cc
        src/game/game_sim.cc
        src/game/great_people.cc
        src/game/mod_host.cc
        src/game/religion.cc
        src/tools/civ_sim_main.cc
    )
    target_include_directories(odai_civ_sim PRIVATE src)
    target_link_libraries(odai_civ_sim PRIVATE odai_content)
    if(MSVC)
        target_compile_options(odai_civ_sim PRIVATE /W4 /permissive-)
    else()
        target_compile_options(odai_civ_sim PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    # Headless Stellaris-style space 4X playtest harness. Pure CPU (no Vulkan):
    # runs a multi-empire galaxy for N turns, prints fun-factor metrics, then
    # constructs all strategy-4x UI panels with sci-fi resource types to verify
    # the panel kit is genre-agnostic. Pairs with theme_stellaris.json.
    #   odai_stellaris_sim [turns] [seed] [empires] [--quiet|--sweep N]
    add_executable(odai_stellaris_sim
        src/tools/stellaris_sim_main.cc
    )
    target_include_directories(odai_stellaris_sim PRIVATE src)
    target_link_libraries(odai_stellaris_sim PRIVATE odai_ui)
    if(MSVC)
        target_compile_options(odai_stellaris_sim PRIVATE /W4 /permissive-)
    else()
        target_compile_options(odai_stellaris_sim PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    # DDS bundler — offline PNG → BC3 compressor. Run once per asset directory to
    # produce .dds sidecars; the runtime loader prefers .dds over the source .png.
    #   odai_dds_bundler <file.png> [...]
    #   odai_dds_bundler --dir <directory>
    add_executable(odai_dds_bundler
        src/import/dds.cc
        src/tools/dds_bundler_main.cc
    )
    target_include_directories(odai_dds_bundler PRIVATE src ${ODAI_STB_INCLUDE_DIRS})
    if(MSVC)
        target_compile_options(odai_dds_bundler PRIVATE ${ODAI_WARN_FLAGS_MSVC})
    else()
        target_compile_options(odai_dds_bundler PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    # SVG bundler — offline .svg → .odaivec tessellated geometry cache. Run once
    # per icon directory; VectorIconRegistry prefers a fresh .odaivec sidecar.
    #   odai_svg_bundler <file.svg> [...]
    #   odai_svg_bundler --dir <directory> [--sizes 16,32,64] [--out <dir>]
    add_executable(odai_svg_bundler src/tools/svg_bundler_main.cc)
    target_link_libraries(odai_svg_bundler PRIVATE odai_ui)
    target_include_directories(odai_svg_bundler PRIVATE src)
    if(MSVC)
        target_compile_options(odai_svg_bundler PRIVATE ${ODAI_WARN_FLAGS_MSVC})
    else()
        target_compile_options(odai_svg_bundler PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    # Theme viewer — artist tool for iterating on theme JSON files with live
    # hot-reload. Terminal-based: prints all color/font/size tokens with ANSI
    # color swatches and re-dumps whenever the theme file changes on disk.
    add_executable(odai_theme_viewer src/tools/theme_viewer_main.cc)
    target_link_libraries(odai_theme_viewer PRIVATE odai_ui)
    target_include_directories(odai_theme_viewer PRIVATE src)
    if(MSVC)
        target_compile_options(odai_theme_viewer PRIVATE ${ODAI_WARN_FLAGS_MSVC})
    else()
        target_compile_options(odai_theme_viewer PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endif()

if(BUILD_TESTING)
    add_executable(odai_imported_scene_tests
        tests/imported_scene_tests.cc
        src/import/dds.cc
        src/import/gpu_scene.cc
        src/import/imported_scene.cc
    )
    target_include_directories(odai_imported_scene_tests PRIVATE src)

    if(MSVC)
        target_compile_options(odai_imported_scene_tests PRIVATE
            /W4
            /permissive-
            /w14242
            /w14254
            /w14263
            /w14265
            /w14287
            /we4289
            /w14296
            /w14311
            /w14545
            /w14546
            /w14547
            /w14549
            /w14555
            /w14619
            /w14640
            /w14826
            /w14905
            /w14906
            /w14928
        )
    else()
        target_compile_options(odai_imported_scene_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    add_test(NAME odai_imported_scene_tests COMMAND odai_imported_scene_tests)

    add_executable(odai_foundation_tests
        tests/foundation_tests.cc
        src/core/log.cc
        src/world/chunk_grid_worldgen.cc
        src/world/magica_voxel.cc
        src/world/chunk_mesher.cc
        src/world/clipmap_index.cc
    )
    target_include_directories(odai_foundation_tests PRIVATE src)

    if(MSVC)
        target_compile_options(odai_foundation_tests PRIVATE
            /W4
            /permissive-
            /w14242
            /w14254
            /w14263
            /w14265
            /w14287
            /we4289
            /w14296
            /w14311
            /w14545
            /w14546
            /w14547
            /w14549
            /w14555
            /w14619
            /w14640
            /w14826
            /w14905
            /w14906
            /w14928
        )
    else()
        target_compile_options(odai_foundation_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    add_test(NAME odai_foundation_tests COMMAND odai_foundation_tests)

    add_executable(odai_strategy_map_tests
        tests/strategy_map_tests.cc
        src/game/buildable.cc
        src/game/great_people.cc
        src/game/strategy_map.cc
        src/game/strategy_map_io.cc
        src/game/strategy_map_mesh.cc
        src/game/strategy_hex_terrain.cc
        src/game/units.cc
    )
    target_include_directories(odai_strategy_map_tests PRIVATE src)
    target_link_libraries(odai_strategy_map_tests PRIVATE odai_content)

    if(MSVC)
        target_compile_options(odai_strategy_map_tests PRIVATE
            /W4
            /permissive-
            /w14242
            /w14254
            /w14263
            /w14265
            /w14287
            /we4289
            /w14296
            /w14311
            /w14545
            /w14546
            /w14547
            /w14549
            /w14555
            /w14619
            /w14640
            /w14826
            /w14905
            /w14906
            /w14928
        )
    else()
        target_compile_options(odai_strategy_map_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    add_test(NAME odai_strategy_map_tests COMMAND odai_strategy_map_tests)

    # Strategic economy + headless 4X simulation (yields, growth, happiness,
    # research, wonders, AI). Pure CPU; no Vulkan.
    add_executable(odai_economy_tests
        tests/economy_tests.cc
        src/game/ai_units.cc
        src/game/buildable.cc
        src/game/economy.cc
        src/game/game_sim.cc
        src/game/great_people.cc
        src/game/mod_host.cc
        src/game/religion.cc
        src/game/strategy_map.cc
        src/game/units.cc
    )
    target_include_directories(odai_economy_tests PRIVATE src)
    target_link_libraries(odai_economy_tests PRIVATE odai_content)
    if(MSVC)
        target_compile_options(odai_economy_tests PRIVATE /W4 /permissive-)
    else()
        target_compile_options(odai_economy_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME odai_economy_tests COMMAND odai_economy_tests)

    # Council of Houses advisor rule engine (catalog + evaluateAdvisors). Pure CPU;
    # no Vulkan, no UI.
    add_executable(odai_advisor_tests
        tests/advisor_tests.cc
        src/game/advisor.cc
    )
    target_include_directories(odai_advisor_tests PRIVATE src)
    if(MSVC)
        target_compile_options(odai_advisor_tests PRIVATE /W4 /permissive-)
    else()
        target_compile_options(odai_advisor_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME odai_advisor_tests COMMAND odai_advisor_tests)

    # Great People: data-driven catalog, the city-bonus application, global
    # uniqueness, and great-person-point birth/integration in the headless sim.
    # Pure CPU; no Vulkan, no UI.
    add_executable(odai_great_people_tests
        tests/great_people_tests.cc
        src/game/economy.cc
        src/game/game_sim.cc
        src/game/great_people.cc
        src/game/mod_host.cc
        src/game/religion.cc
        src/game/strategy_map.cc
    )
    target_include_directories(odai_great_people_tests PRIVATE src)
    target_link_libraries(odai_great_people_tests PRIVATE odai_content)
    if(MSVC)
        target_compile_options(odai_great_people_tests PRIVATE /W4 /permissive-)
    else()
        target_compile_options(odai_great_people_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME odai_great_people_tests COMMAND odai_great_people_tests)

    # Content/mod system: base game loads cleanly from JSON, referential integrity
    # holds, and the loaded values match the pre-migration hardcoded tables
    # (parity oracle). Pure CPU; no Vulkan.
    add_executable(odai_content_tests
        tests/content_tests.cc
        src/game/economy.cc
        src/game/game_sim.cc
        src/game/great_people.cc
        src/game/mod_host.cc
        src/game/religion.cc
        src/game/strategy_map.cc
        src/game/units.cc
        src/game/buildable.cc
    )
    target_include_directories(odai_content_tests PRIVATE src)
    target_link_libraries(odai_content_tests PRIVATE odai_content)
    if(MSVC)
        target_compile_options(odai_content_tests PRIVATE /W4 /permissive-)
    else()
        target_compile_options(odai_content_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME odai_content_tests COMMAND odai_content_tests)

    # Lua scripting engine: event hooks fire with the right counts/order, and a
    # scripted run is deterministic (same seed+mod -> identical TurnSamples; a
    # no-op mod matches a NullModHost run). Links odai_script (Lua).
    add_executable(odai_lua_hook_tests
        tests/lua_hook_tests.cc
        src/game/economy.cc
        src/game/game_sim.cc
        src/game/great_people.cc
        src/game/mod_host.cc
        src/game/religion.cc
        src/game/strategy_map.cc
        src/game/units.cc
        src/game/buildable.cc
    )
    target_include_directories(odai_lua_hook_tests PRIVATE src)
    target_link_libraries(odai_lua_hook_tests PRIVATE odai_script)
    add_test(NAME odai_lua_hook_tests COMMAND odai_lua_hook_tests)

    add_executable(odai_ui_tests tests/ui_tests.cc)
    target_link_libraries(odai_ui_tests PRIVATE odai_ui)
    if(MSVC)
        target_compile_options(odai_ui_tests PRIVATE ${ODAI_WARN_FLAGS_MSVC})
    else()
        target_compile_options(odai_ui_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME odai_ui_tests COMMAND odai_ui_tests)

    # Vector graphics + SVG importer: path flattening, fill/stroke tessellation,
    # AA fringe, cache round-trip, inline-SVG parse. Headless, links only odai_ui.
    add_executable(odai_svg_vector_tests tests/svg_vector_tests.cc)
    target_link_libraries(odai_svg_vector_tests PRIVATE odai_ui)
    if(MSVC)
        target_compile_options(odai_svg_vector_tests PRIVATE ${ODAI_WARN_FLAGS_MSVC})
    else()
        target_compile_options(odai_svg_vector_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME odai_svg_vector_tests COMMAND odai_svg_vector_tests)

    # Retro demo's RetroUi widget wrapper: slider drag, textbox typing, toggle,
    # visibility/skin. Headless (links odai_ui + the Vulkan-free retro_widgets.cc).
    add_executable(odai_retro_widgets_tests
        tests/retro_widgets_tests.cc
        src/tools/retro_theme_demo/retro_widgets.cc
    )
    target_include_directories(odai_retro_widgets_tests PRIVATE src)
    target_link_libraries(odai_retro_widgets_tests PRIVATE odai_ui)
    if(MSVC)
        target_compile_options(odai_retro_widgets_tests PRIVATE ${ODAI_WARN_FLAGS_MSVC})
    else()
        target_compile_options(odai_retro_widgets_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME odai_retro_widgets_tests COMMAND odai_retro_widgets_tests)

    # Audio subsystem: headless NullBackend behavior (no device, no miniaudio).
    # Compiles only the facade + null backend so it builds/runs anywhere.
    add_executable(odai_audio_tests
        tests/audio_tests.cc
        src/audio/audio.cc
        src/audio/null_backend.cc
        src/core/log.cc
    )
    target_include_directories(odai_audio_tests PRIVATE src)
    if(MSVC)
        target_compile_options(odai_audio_tests PRIVATE ${ODAI_WARN_FLAGS_MSVC})
    else()
        target_compile_options(odai_audio_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME odai_audio_tests COMMAND odai_audio_tests)

    find_package(GTest CONFIG QUIET)
    if(NOT GTest_FOUND)
        find_package(GTest QUIET)
    endif()
    if(GTest_FOUND)
        add_executable(odai_stability_gtests
            tests/stability_g_tests.cc
            tests/frame_math_tests.cc
            tests/frame_graph_tests.cc
            tests/frame_render_tests.cc
            tests/sim_network_tests.cc
            src/render/frame_graph.cc
            src/render/backend/vulkan/frame_graph_runtime.cc
            src/render/backend/vulkan/shadow_culling_utils.cc
            src/core/log.cc
            src/world/magica_voxel.cc
            src/world/chunk_mesher.cc
            src/world/clipmap_index.cc
        )
        target_include_directories(odai_stability_gtests PRIVATE src)
        target_link_libraries(odai_stability_gtests PRIVATE GTest::gtest_main)

        if(MSVC)
            target_compile_options(odai_stability_gtests PRIVATE
                /W4
                /permissive-
                /w14242
                /w14254
                /w14263
                /w14265
                /w14287
                /we4289
                /w14296
                /w14311
                /w14545
                /w14546
                /w14547
                /w14549
                /w14555
                /w14619
                /w14640
                /w14826
                /w14905
                /w14906
                /w14928
            )
        else()
            target_compile_options(odai_stability_gtests PRIVATE -Wall -Wextra -Wpedantic)
        endif()

        include(GoogleTest)
        gtest_discover_tests(odai_stability_gtests DISCOVERY_TIMEOUT 60)
    else()
        message(STATUS "GTest not found; skipping odai_stability_gtests target (install gtest for your active triplet)")
    endif()
endif()
