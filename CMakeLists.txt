cmake_minimum_required(VERSION 3.21)
project(odai VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
include(CTest)
option(ODAI_BUILD_APP "Build the main Vulkan toy application." ON)
option(ODAI_BUILD_TOOLS "Build offline content tools." ON)

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

# Optional Slang shader compilation (to SPIR-V for Vulkan).
# Produces .slang.spv files next to the source shaders.
find_program(SLANGC_EXECUTABLE NAMES slangc)
if(SLANGC_EXECUTABLE)
    set(SLANG_SHARED_INCLUDES
        "${CMAKE_SOURCE_DIR}/src/render/shaders/camera_uniform.slang"
        "${CMAKE_SOURCE_DIR}/src/render/shaders/chunk_push_constants.slang"
        "${CMAKE_SOURCE_DIR}/src/render/shaders/fullscreen_triangle.slang"
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
    src/ui/widgets/donut_chart.cc
    src/ui/widgets/toast.cc
    src/ui/widgets/line_chart.cc
    src/ui/widgets/stat_badge.cc
    src/ui/widgets/icon_button.cc
    # Game-domain widgets (production_panel, etc.) are NOT included here;
    # they live in the app source list and link against odai_ui.
)
target_include_directories(odai_ui
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${ODAI_STB_INCLUDE_DIRS}
)
target_link_libraries(odai_ui PUBLIC nlohmann_json::nlohmann_json)
target_compile_features(odai_ui PUBLIC cxx_std_20)
if(MSVC)
    target_compile_options(odai_ui PRIVATE ${ODAI_WARN_FLAGS_MSVC})
else()
    target_compile_options(odai_ui PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(ODAI_BUILD_APP)
    add_executable(odai
        src/app/main.cc
        src/app/app.cc
        src/game/buildable.cc
        src/game/strategy_map.cc
        src/game/strategy_map_io.cc
        src/game/strategy_map_mesh.cc
        src/game/strategy_hex_terrain.cc
        src/game/units.cc
        src/ui/widgets/production_panel.cc
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
        )
        target_include_directories(odai_ui_vulkan PUBLIC
            $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
            $<INSTALL_INTERFACE:include>
        )
        target_link_libraries(odai_ui_vulkan
            PUBLIC  odai_ui
            PRIVATE Vulkan::Vulkan GPUOpen::VulkanMemoryAllocator
        )
        if(MSVC)
            target_compile_options(odai_ui_vulkan PRIVATE ${ODAI_WARN_FLAGS_MSVC})
        else()
            target_compile_options(odai_ui_vulkan PRIVATE -Wall -Wextra -Wpedantic)
        endif()

        list(APPEND RENDER_BACKEND_SOURCES
            src/render/backend/vulkan/buffer_helpers.cc
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
endif()

# ---------------------------------------------------------------------------
# Install / CMake package export for odai_ui and odai_ui_vulkan
# ---------------------------------------------------------------------------
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(ODAI_UI_INSTALL_TARGETS odai_ui)
if(TARGET odai_ui_vulkan)
    list(APPEND ODAI_UI_INSTALL_TARGETS odai_ui_vulkan)
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
        src/tools/civ_sim_main.cc
    )
    target_include_directories(odai_civ_sim PRIVATE src)
    if(MSVC)
        target_compile_options(odai_civ_sim PRIVATE /W4 /permissive-)
    else()
        target_compile_options(odai_civ_sim PRIVATE -Wall -Wextra -Wpedantic)
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
        src/game/strategy_map.cc
        src/game/strategy_map_io.cc
        src/game/strategy_map_mesh.cc
        src/game/strategy_hex_terrain.cc
        src/game/units.cc
    )
    target_include_directories(odai_strategy_map_tests PRIVATE src)

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
        src/game/economy.cc
        src/game/game_sim.cc
        src/game/strategy_map.cc
    )
    target_include_directories(odai_economy_tests PRIVATE src)
    if(MSVC)
        target_compile_options(odai_economy_tests PRIVATE /W4 /permissive-)
    else()
        target_compile_options(odai_economy_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME odai_economy_tests COMMAND odai_economy_tests)

    add_executable(odai_ui_tests tests/ui_tests.cc)
    target_link_libraries(odai_ui_tests PRIVATE odai_ui)
    if(MSVC)
        target_compile_options(odai_ui_tests PRIVATE ${ODAI_WARN_FLAGS_MSVC})
    else()
        target_compile_options(odai_ui_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME odai_ui_tests COMMAND odai_ui_tests)

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
