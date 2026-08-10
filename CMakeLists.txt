cmake_minimum_required(VERSION 3.21)
project(odai VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
include(CTest)
find_package(Threads REQUIRED)
option(ODAI_BUILD_APP "Build the main Vulkan toy application." ON)
option(ODAI_BUILD_TOOLS "Build offline content tools." ON)
option(ODAI_BUILD_EXAMPLES "Build standalone odai_ui SDK integration samples." OFF)

# ---------------------------------------------------------------------------
# Optimized builds
#
# Optimization level itself comes from CMAKE_BUILD_TYPE -- this project does not
# hardcode -O flags, so `Release` and `RelWithDebInfo` behave the way every
# other CMake project does. What lives here is (a) refusing to silently produce
# an unoptimized *and* undebuggable binary when no build type is given, and
# (b) two opt-in switches that a plain build type does not turn on.
#
# Use RelWithDebInfo for profiling: optimized code with symbols, so the F3
# overlay (engine/game_frame_stats.h) and GPU timestamps report numbers that
# mean something while stacks are still readable. A Debug measurement is not a
# measurement -- docs/devlog.md records Debug-vs-Release worth ~7s -> <1s on
# world decode alone.
# ---------------------------------------------------------------------------
get_property(ODAI_IS_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT ODAI_IS_MULTI_CONFIG AND NOT CMAKE_BUILD_TYPE)
    # An empty CMAKE_BUILD_TYPE means no -O and no -g: the worst of both. Pick
    # the profiling-friendly default rather than leaving it unset.
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING
        "Build type: Debug, Release, RelWithDebInfo, or MinSizeRel." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
        Debug Release RelWithDebInfo MinSizeRel)
    message(STATUS "odai: CMAKE_BUILD_TYPE was unset, defaulting to RelWithDebInfo")
endif()

option(ODAI_ENABLE_LTO
    "Link-time / interprocedural optimization in optimized build types. Slower links." OFF)
option(ODAI_ENABLE_NATIVE_ARCH
    "Tune codegen for THIS machine's CPU (-march=native / /arch:AVX2). Non-portable." OFF)

if(ODAI_ENABLE_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT ODAI_IPO_SUPPORTED OUTPUT ODAI_IPO_ERROR LANGUAGES CXX)
    if(ODAI_IPO_SUPPORTED)
        # Per-config rather than the blanket CMAKE_INTERPROCEDURAL_OPTIMIZATION:
        # LTO in a Debug build costs link time and buys nothing.
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO ON)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_MINSIZEREL ON)
        message(STATUS "odai: LTO enabled for optimized build types")
    else()
        message(WARNING "odai: ODAI_ENABLE_LTO requested but unsupported: ${ODAI_IPO_ERROR}")
    endif()
endif()

if(ODAI_ENABLE_NATIVE_ARCH)
    # Deliberately NOT default-on, and deliberately no -ffast-math anywhere:
    # wider SIMD and FMA contraction change floating-point results, and this
    # project pins content-affecting hashes and RNG with golden vectors
    # (tests/core_types_tests.cc) because worldgen output has to reproduce.
    # Binaries built this way may also crash outright on an older CPU. Turn it
    # on for a local profiling run, not for anything you ship or share.
    include(CheckCXXCompilerFlag)
    if(MSVC)
        set(ODAI_NATIVE_ARCH_FLAG "/arch:AVX2")
    else()
        set(ODAI_NATIVE_ARCH_FLAG "-march=native")
    endif()
    check_cxx_compiler_flag("${ODAI_NATIVE_ARCH_FLAG}" ODAI_NATIVE_ARCH_FLAG_OK)
    if(ODAI_NATIVE_ARCH_FLAG_OK)
        add_compile_options(
            "$<$<CONFIG:Release,RelWithDebInfo,MinSizeRel>:${ODAI_NATIVE_ARCH_FLAG}>")
        message(STATUS "odai: native-arch codegen enabled (${ODAI_NATIVE_ARCH_FLAG}) "
                       "-- non-portable binary, float results may shift")
    else()
        message(WARNING "odai: ${ODAI_NATIVE_ARCH_FLAG} rejected by the compiler; ignoring")
    endif()
endif()

# stb (stb_truetype.h, stb_rect_pack.h) for the UI font rasterizer. Provided by
# the vcpkg `stb` port in normal builds; ODAI_STB_INCLUDE_DIR overrides it for
# headless checks without vcpkg.
find_package(nlohmann_json CONFIG REQUIRED)
# zlib: BSA archive/ESM record decompression for the Fallout: New Vegas
# import pipeline (src/import/fnv/). The only place zlib is used directly.
find_package(ZLIB REQUIRED)
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
# The Vulkan SDK ships slangc, so prefer it when VULKAN_SDK is set (HINTS are
# searched ahead of PATH) and fall back to any slangc on PATH otherwise. Both
# layouts are covered: VULKAN_SDK pointing at the arch dir or at the SDK root.
find_program(SLANGC_EXECUTABLE
    NAMES slangc
    HINTS
        "$ENV{VULKAN_SDK}/bin"
        "$ENV{VULKAN_SDK}/x86_64/bin"
)
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
    # Ambient occlusion: one source, three estimators. ODAI_AO_MODE picks which
    # estimateVisibility() compiles in, so each pipeline's sample loop is branch-free.
    add_slang_shader("${CMAKE_SOURCE_DIR}/src/render/shaders/taa.comp.slang" compute)
    add_slang_shader("${CMAKE_SOURCE_DIR}/src/render/shaders/ssao.comp.slang" compute)
    add_slang_shader_variant(
        "${CMAKE_SOURCE_DIR}/src/render/shaders/ssao.comp.slang"
        compute
        "${CMAKE_SOURCE_DIR}/src/render/shaders/ssao_hbao.comp.slang.spv"
        -DODAI_AO_MODE=1
    )
    add_slang_shader_variant(
        "${CMAKE_SOURCE_DIR}/src/render/shaders/ssao.comp.slang"
        compute
        "${CMAKE_SOURCE_DIR}/src/render/shaders/ssao_gtao.comp.slang.spv"
        -DODAI_AO_MODE=2
    )
    add_slang_shader("${CMAKE_SOURCE_DIR}/src/render/shaders/ssao_blur.comp.slang" compute)
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
    add_slang_shader("${CMAKE_SOURCE_DIR}/src/render/shaders/skinning.comp.slang" compute)
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
    src/ui/ui_cursor.cc
    src/ui/font.cc
    src/ui/rich_text.cc
    src/ui/cached_rich_text.cc
    src/ui/ui_context.cc
    src/ui/icon_atlas.cc
    src/ui/toast_host.cc
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
    src/ui/widgets/dialogue_panel.cc
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
# odai_ui_editor core — the Vulkan-free half of src/tools/ui_editor/
# ---------------------------------------------------------------------------
# Document model, undo history, snapping and editor operations. Shared by the
# GLFW/Vulkan odai_ui_editor app and by odai_ui_editor_tests, which is why it is
# a plain source list declared up here rather than inside the app-only block.
set(ODAI_UI_EDITOR_CORE_SOURCES
    ${CMAKE_SOURCE_DIR}/src/tools/ui_editor/editor_color.cc
    ${CMAKE_SOURCE_DIR}/src/tools/ui_editor/editor_document.cc
    ${CMAKE_SOURCE_DIR}/src/tools/ui_editor/editor_history.cc
    ${CMAKE_SOURCE_DIR}/src/tools/ui_editor/editor_ops.cc
    ${CMAKE_SOURCE_DIR}/src/tools/ui_editor/editor_snap.cc
)

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
    src/content/material_library.cc
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
# odai_anim — skeleton/animation-clip data model and CPU sampler (Dragon Age:
# Origins touchstone, see docs/ROADMAP.md). Vulkan-free: produces flat
# odai::math::Matrix4 skinning matrices; only src/render/ turns those into GPU
# buffers.
# ---------------------------------------------------------------------------
add_library(odai_anim STATIC
    src/anim/animation_sampler.cc
    src/anim/skeleton_io.cc
    src/anim/biped_rig.cc
)
target_include_directories(odai_anim PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(odai_anim PRIVATE nlohmann_json::nlohmann_json)
target_compile_features(odai_anim PUBLIC cxx_std_20)
if(MSVC)
    target_compile_options(odai_anim PRIVATE ${ODAI_WARN_FLAGS_MSVC})
else()
    target_compile_options(odai_anim PRIVATE -Wall -Wextra -Wpedantic)
endif()

# ---------------------------------------------------------------------------
# odai_dialogue — branching dialogue trees (Dragon Age: Origins touchstone, see
# docs/ROADMAP.md). Vulkan-free, game-free: conditions/effects run against the
# abstract DialogueContext, not any specific game's state, so this links into
# any game prototype without pulling in its data model.
# ---------------------------------------------------------------------------
add_library(odai_dialogue STATIC
    src/dialogue/dialogue_io.cc
    src/dialogue/dialogue_runtime.cc
    src/dialogue/dialogue_state_io.cc
)
target_include_directories(odai_dialogue PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(odai_dialogue PRIVATE nlohmann_json::nlohmann_json)
target_compile_features(odai_dialogue PUBLIC cxx_std_20)
if(MSVC)
    target_compile_options(odai_dialogue PRIVATE ${ODAI_WARN_FLAGS_MSVC})
else()
    target_compile_options(odai_dialogue PRIVATE -Wall -Wextra -Wpedantic)
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

# Newer sol2 packages export only the namespaced target; older ones also export
# a bare `sol2`. Without this, a bare `sol2` silently degrades to a `-lsol2`
# link flag and fails at link time.
if(NOT TARGET sol2 AND TARGET sol2::sol2)
    add_library(sol2 ALIAS sol2::sol2)
endif()

# Game-agnostic Lua plumbing (sandbox, sorted script loading, Rng/Log tables)
# shared by every script host: the 4X ScriptHost below and the citybuilder's
# CityScriptHost. No game types.
add_library(odai_script_core STATIC
    src/script/script_core.cc
)
target_include_directories(odai_script_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${LUA_INCLUDE_DIR}
)
target_link_libraries(odai_script_core PRIVATE sol2 ${LUA_LIBRARIES})
target_compile_features(odai_script_core PUBLIC cxx_std_20)

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
target_link_libraries(odai_script PRIVATE odai_script_core sol2 ${LUA_LIBRARIES})
target_compile_features(odai_script PUBLIC cxx_std_20)
target_compile_definitions(odai_script PRIVATE ODAI_PROJECT_SOURCE_DIR=\"${CMAKE_SOURCE_DIR}\")

# The citybuilder's Lua content layer (names, story templates, need schedules,
# tuning config). Public header is Lua-free; registration-based API with
# compiled-in fallbacks so the game runs scriptless. See mods/citybuilder/.
add_library(odai_city_script STATIC
    src/games/citybuilder/script/city_script.cc
)
target_include_directories(odai_city_script
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${LUA_INCLUDE_DIR}
)
target_link_libraries(odai_city_script PUBLIC odai_content)
target_link_libraries(odai_city_script PRIVATE odai_script_core sol2 ${LUA_LIBRARIES})
target_compile_features(odai_city_script PUBLIC cxx_std_20)
target_compile_definitions(odai_city_script PRIVATE ODAI_PROJECT_SOURCE_DIR=\"${CMAKE_SOURCE_DIR}\")

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
            src/import/imported_scene_query.cc
        src/render/frame_graph.cc
        src/render/renderer.cc
        src/world/magica_voxel.cc
        src/world/world.cc
        src/world/chunk_grid_worldgen.cc
        src/world/chunk_mesher.cc
        src/world/chunk_mesh_scheduler.cc
        src/world/clipmap_index.cc
        src/core/job_system.cc
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
            src/render/backend/vulkan/frame_capture.cc
            src/render/backend/vulkan/frame_run.cc
            src/render/backend/vulkan/frame_graph_core.cc
            src/render/backend/vulkan/frame_graph_runtime.cc
            src/render/backend/vulkan/frame_pass_shadow.cc
            src/render/backend/vulkan/frame_pass_prepass.cc
            src/render/backend/vulkan/frame_pass_ssao.cc
            src/render/backend/vulkan/frame_pass_skinning.cc
            src/render/backend/vulkan/frame_pass_main.cc
            src/render/backend/vulkan/frame_chunks.cc
            src/render/backend/vulkan/frame_draws.cc
            src/render/backend/vulkan/frame_instances.cc
            src/render/backend/vulkan/init.cc
            src/render/backend/vulkan/init_resources.cc
            src/render/backend/vulkan/chunk_upload.cc
            src/render/backend/vulkan/shadow_culling_utils.cc
            src/render/backend/vulkan/descriptors.cc
            src/render/backend/vulkan/descriptor_buffer.cc
            src/render/backend/vulkan/ui_panels.cc
            src/render/backend/vulkan/pass_pipelines.cc
            src/render/backend/vulkan/swapchain_targets.cc
            src/render/backend/vulkan/post_compute.cc
            src/render/backend/vulkan/skinning_resources.cc
            src/render/backend/vulkan/pass_gi.cc
            src/render/backend/vulkan/shadow_culling.cc
            src/render/backend/vulkan/vma_usage.cc
            # chunk_upload.cc scatters grass through the Vulkan-free world helper.
            src/world/grass_scatter.cc
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

    if(WIN32)
        set(ODAI_RELEASE_PROFILE_VALUE "windows_runnable_rt_beta")
    elseif(APPLE)
        set(ODAI_RELEASE_PROFILE_VALUE "macos_runnable_rt_beta")
    else()
        set(ODAI_RELEASE_PROFILE_VALUE "linux_runnable_rt_beta")
    endif()

    target_include_directories(odai PRIVATE src ${ODAI_STB_INCLUDE_DIRS})
    target_compile_definitions(
        odai PRIVATE
        ODAI_PROJECT_SOURCE_DIR=\"${CMAKE_SOURCE_DIR}\"
        ODAI_APP_VERSION=\"${PROJECT_VERSION}\"
        ODAI_RELEASE_PROFILE=\"${ODAI_RELEASE_PROFILE_VALUE}\"
    )

    if(MSVC)
        target_compile_options(odai PRIVATE ${ODAI_WARN_FLAGS_MSVC})
    else()
        target_compile_options(odai PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    # UI libraries (Vulkan + VMA already found in the VULKAN block above)
    target_link_libraries(odai PRIVATE Threads::Threads)
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
            src/engine/plugin.cc
            src/import/dds.cc
            src/import/gpu_scene.cc
            src/import/imported_scene.cc
            src/import/imported_scene_query.cc
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
            src/engine/plugin.cc
            src/import/dds.cc
            src/import/gpu_scene.cc
            src/import/imported_scene.cc
            src/import/imported_scene_query.cc
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
            src/core/file_watch.cc
            src/games/citybuilder/citybuilder_citizens.cc
            src/games/citybuilder/citybuilder_fields.cc
            src/games/citybuilder/citybuilder_save.cc
            src/engine/game_app.cc
            src/engine/plugin.cc
            src/import/dds.cc
            src/import/gpu_scene.cc
            src/import/imported_scene.cc
            src/import/imported_scene_query.cc
            src/procgen/csg.cc
            src/procgen/primitives.cc
            src/procgen/mesh_emit.cc
            src/procgen/building_generator.cc
            src/procgen/props.cc
            src/procgen/city_terrain.cc
            src/procgen/civic_generator.cc
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
            odai_ui odai_ui_vulkan odai_audio odai_city_script odai_content
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
            target_compile_options(odai_game_citybuilder PRIVATE ${ODAI_WARN_FLAGS_MSVC} /FS)
        else()
            target_compile_options(odai_game_citybuilder PRIVATE -Wall -Wextra -Wpedantic)
        endif()

    # ---------------------------------------------------------------------------
    # odai_game_legion — Rome-themed small-party (4 vs 4) real-time-with-pause
    # skirmish (Dragon Age: Origins touchstone, Vulkan). First game exercising
    # the multi-instance GPU skinning pipeline end to end.
    # ---------------------------------------------------------------------------
        # Free-roam viewer for cooked Fallout: New Vegas scenes. Needs a full
        # GameApp (not odai_material_editor, which is minimal-rendering) to show
        # sun, sky, shadows and ambient occlusion.
        #   odai_game_newvegas --scene <cooked.bin>
        add_executable(odai_game_legion
            src/games/legion/legion_main.cc
            src/games/legion/legion_app.cc
            src/engine/game_app.cc
            src/engine/plugin.cc
            src/import/dds.cc
            src/import/gpu_scene.cc
            src/import/imported_scene.cc
            src/import/imported_scene_query.cc
            src/procgen/humanoid_generator.cc
            src/render/frame_graph.cc
            src/render/renderer.cc
            src/world/world.cc
            src/world/chunk_grid_worldgen.cc
            src/world/chunk_mesher.cc
            src/world/clipmap_index.cc
            src/world/magica_voxel.cc
            ${RENDER_BACKEND_SOURCES}
        )
        target_include_directories(odai_game_legion PRIVATE src ${ODAI_STB_INCLUDE_DIRS})
        target_compile_definitions(odai_game_legion PRIVATE
            ODAI_RENDER_BACKEND_VULKAN=1
            ODAI_HAS_VULKAN=1
            ODAI_HAS_GLFW=1
            ODAI_PROJECT_SOURCE_DIR=\"${CMAKE_SOURCE_DIR}\"
        )
        target_link_libraries(odai_game_legion PRIVATE
            odai_ui odai_ui_vulkan odai_audio odai_anim
            Vulkan::Vulkan GPUOpen::VulkanMemoryAllocator imgui::imgui
        )
        if(TARGET glfw)
            target_link_libraries(odai_game_legion PRIVATE glfw)
        elseif(TARGET glfw3::glfw)
            target_link_libraries(odai_game_legion PRIVATE glfw3::glfw)
        endif()
        if(TARGET slang_shaders)
            add_dependencies(odai_game_legion slang_shaders)
        endif()
        if(MSVC)
            target_compile_options(odai_game_legion PRIVATE ${ODAI_WARN_FLAGS_MSVC})
        else()
            target_compile_options(odai_game_legion PRIVATE -Wall -Wextra -Wpedantic)
        endif()

        add_executable(odai_game_newvegas
            src/games/newvegas/newvegas_main.cc
            src/games/newvegas/newvegas_app.cc
            src/games/newvegas/newvegas_collision.cc
            src/games/newvegas/newvegas_ogg.cc
            src/games/newvegas/newvegas_victor.cc
            src/import/fnv/dialogue_records.cc
            src/dialogue/dialogue_runtime.cc
            src/core/job_system.cc
            src/import/cell_residency_planner.cc
            # The runtime reads the game's own files now: BSA archives, the ESM
            # record walker, typed record extraction, the NIF geometry parser
            # and the shared per-cell scene builder.
            src/import/fnv/asset_source.cc
            src/import/fnv/bsa_archive.cc
            src/import/fnv/cell_builder.cc
            src/import/fnv/decoded_texture_cache.cc
        src/import/fnv/decoded_texture_cache.cc
            src/import/fnv/cell_streamer.cc
            src/import/fnv/esm_reader.cc
            src/import/fnv/plugin_load_order.cc
            src/import/fnv/weather_records.cc
            src/import/fnv/fallout_records.cc
            src/import/fnv/nif_scene.cc
            src/import/fnv/character_builder.cc
            src/engine/game_app.cc
            src/engine/plugin.cc
            src/import/dds.cc
            src/import/gpu_scene.cc
            src/import/imported_scene.cc
            src/import/imported_scene_query.cc
            src/render/frame_graph.cc
            src/render/renderer.cc
            src/world/world.cc
            src/world/chunk_grid_worldgen.cc
            src/world/chunk_mesher.cc
            src/world/clipmap_index.cc
            src/world/magica_voxel.cc
            ${RENDER_BACKEND_SOURCES}
        )
        target_include_directories(odai_game_newvegas PRIVATE src ${ODAI_STB_INCLUDE_DIRS})
        target_compile_definitions(odai_game_newvegas PRIVATE
            ODAI_RENDER_BACKEND_VULKAN=1
            ODAI_HAS_VULKAN=1
            ODAI_HAS_GLFW=1
            ODAI_PROJECT_SOURCE_DIR=\"${CMAKE_SOURCE_DIR}\"
        )
        target_link_libraries(odai_game_newvegas PRIVATE
            odai_ui odai_ui_vulkan odai_audio
            Vulkan::Vulkan GPUOpen::VulkanMemoryAllocator imgui::imgui
            # zlib: the runtime now inflates compressed ESM records and BSA
            # entries itself, streaming from the game's own files.
            ZLIB::ZLIB
        )
        if(TARGET glfw)
            target_link_libraries(odai_game_newvegas PRIVATE glfw)
        elseif(TARGET glfw3::glfw)
            target_link_libraries(odai_game_newvegas PRIVATE glfw3::glfw)
        endif()
        if(TARGET slang_shaders)
            add_dependencies(odai_game_newvegas slang_shaders)
        endif()
        if(MSVC)
            target_compile_options(odai_game_newvegas PRIVATE ${ODAI_WARN_FLAGS_MSVC})
        else()
            target_compile_options(odai_game_newvegas PRIVATE -Wall -Wextra -Wpedantic)
        endif()


    # ---------------------------------------------------------------------------
    # odai_game_snake — Snake / Nibbles (Vulkan)
    # ---------------------------------------------------------------------------
        add_executable(odai_game_snake
            src/games/snake/snake_main.cc
            src/games/snake/snake_app.cc
            src/engine/game_app.cc
            src/engine/plugin.cc
            src/import/dds.cc
            src/import/gpu_scene.cc
            src/import/imported_scene.cc
            src/import/imported_scene_query.cc
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
    # odai_game_minesweeper — Minesweeper (UI-only, Vulkan)
    # ---------------------------------------------------------------------------
        add_executable(odai_game_minesweeper
            src/games/minesweeper/minesweeper_main.cc
            src/games/minesweeper/minesweeper_app.cc
            src/engine/game_app.cc
            src/engine/plugin.cc
            src/import/dds.cc
            src/import/gpu_scene.cc
            src/import/imported_scene.cc
            src/import/imported_scene_query.cc
            src/render/frame_graph.cc
            src/render/renderer.cc
            src/world/world.cc
            src/world/chunk_grid_worldgen.cc
            src/world/chunk_mesher.cc
            src/world/clipmap_index.cc
            src/world/magica_voxel.cc
            ${RENDER_BACKEND_SOURCES}
        )
        target_include_directories(odai_game_minesweeper PRIVATE src ${ODAI_STB_INCLUDE_DIRS})
        target_compile_definitions(odai_game_minesweeper PRIVATE
            ODAI_RENDER_BACKEND_VULKAN=1
            ODAI_HAS_VULKAN=1
            ODAI_HAS_GLFW=1
            ODAI_PROJECT_SOURCE_DIR=\"${CMAKE_SOURCE_DIR}\"
        )
        target_link_libraries(odai_game_minesweeper PRIVATE
            odai_ui odai_ui_vulkan odai_audio
            Vulkan::Vulkan GPUOpen::VulkanMemoryAllocator imgui::imgui
        )
        if(TARGET glfw)
            target_link_libraries(odai_game_minesweeper PRIVATE glfw)
        elseif(TARGET glfw3::glfw)
            target_link_libraries(odai_game_minesweeper PRIVATE glfw3::glfw)
        endif()
        if(TARGET slang_shaders)
            add_dependencies(odai_game_minesweeper slang_shaders)
        endif()
        if(MSVC)
            target_compile_options(odai_game_minesweeper PRIVATE ${ODAI_WARN_FLAGS_MSVC})
        else()
            target_compile_options(odai_game_minesweeper PRIVATE -Wall -Wextra -Wpedantic)
        endif()

    # ---------------------------------------------------------------------------
    # odai_game_voxelcraft — first-person voxel sandbox (Vulkan)
    # ---------------------------------------------------------------------------
        add_executable(odai_game_voxelcraft
            src/games/voxelcraft/voxelcraft_main.cc
            src/games/voxelcraft/voxelcraft_app.cc
            src/games/voxelcraft/voxelcraft_player.cc
            src/games/voxelcraft/voxelcraft_streaming.cc
            src/engine/game_app.cc
            src/engine/plugin.cc
            src/import/dds.cc
            src/import/gpu_scene.cc
            src/import/imported_scene.cc
            src/import/imported_scene_query.cc
            src/render/frame_graph.cc
            src/render/renderer.cc
            src/world/world.cc
            src/world/chunk_grid_worldgen.cc
            src/world/chunk_mesher.cc
            src/world/clipmap_index.cc
            src/world/magica_voxel.cc
            ${RENDER_BACKEND_SOURCES}
        )
        target_include_directories(odai_game_voxelcraft PRIVATE src ${ODAI_STB_INCLUDE_DIRS})
        target_compile_definitions(odai_game_voxelcraft PRIVATE
            ODAI_RENDER_BACKEND_VULKAN=1
            ODAI_HAS_VULKAN=1
            ODAI_HAS_GLFW=1
            ODAI_PROJECT_SOURCE_DIR=\"${CMAKE_SOURCE_DIR}\"
        )
        target_link_libraries(odai_game_voxelcraft PRIVATE
            odai_ui odai_ui_vulkan odai_audio
            Vulkan::Vulkan GPUOpen::VulkanMemoryAllocator imgui::imgui
        )
        if(TARGET glfw)
            target_link_libraries(odai_game_voxelcraft PRIVATE glfw)
        elseif(TARGET glfw3::glfw)
            target_link_libraries(odai_game_voxelcraft PRIVATE glfw3::glfw)
        endif()
        if(TARGET slang_shaders)
            add_dependencies(odai_game_voxelcraft slang_shaders)
        endif()
        if(MSVC)
            target_compile_options(odai_game_voxelcraft PRIVATE ${ODAI_WARN_FLAGS_MSVC})
        else()
            target_compile_options(odai_game_voxelcraft PRIVATE -Wall -Wextra -Wpedantic)
        endif()

    # ---------------------------------------------------------------------------
    # odai_ui_editor — Drag-and-drop UI editor for artists (Vulkan)
    # ---------------------------------------------------------------------------
        add_executable(odai_ui_editor
            src/tools/ui_editor/ui_editor_main.cc
            src/tools/ui_editor/ui_editor_app.cc
            ${ODAI_UI_EDITOR_CORE_SOURCES}
            src/engine/game_app.cc
            src/engine/plugin.cc
            src/import/dds.cc
            src/import/gpu_scene.cc
            src/import/imported_scene.cc
            src/import/imported_scene_query.cc
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
            src/engine/plugin.cc
            src/import/dds.cc
            src/import/gpu_scene.cc
            src/import/imported_scene.cc
            src/import/imported_scene_query.cc
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

    # ---------------------------------------------------------------------------
    # odai_tween_demo — animation.h showcase: easing gallery, Sequence, RectTween (Vulkan)
    # ---------------------------------------------------------------------------
        add_executable(odai_tween_demo
            src/tools/tween_demo/tween_demo_main.cc
            src/tools/tween_demo/tween_demo_app.cc
            src/engine/game_app.cc
            src/engine/plugin.cc
            src/import/dds.cc
            src/import/gpu_scene.cc
            src/import/imported_scene.cc
            src/import/imported_scene_query.cc
            src/render/frame_graph.cc
            src/render/renderer.cc
            src/world/world.cc
            src/world/chunk_grid_worldgen.cc
            src/world/chunk_mesher.cc
            src/world/clipmap_index.cc
            src/world/magica_voxel.cc
            ${RENDER_BACKEND_SOURCES}
        )
        target_include_directories(odai_tween_demo PRIVATE src ${ODAI_STB_INCLUDE_DIRS})
        target_compile_definitions(odai_tween_demo PRIVATE
            ODAI_RENDER_BACKEND_VULKAN=1
            ODAI_HAS_VULKAN=1
            ODAI_HAS_GLFW=1
            ODAI_PROJECT_SOURCE_DIR=\"${CMAKE_SOURCE_DIR}\"
        )
        target_link_libraries(odai_tween_demo PRIVATE
            odai_ui odai_ui_vulkan odai_audio
            Vulkan::Vulkan GPUOpen::VulkanMemoryAllocator imgui::imgui
        )
        if(TARGET glfw)
            target_link_libraries(odai_tween_demo PRIVATE glfw)
        elseif(TARGET glfw3::glfw)
            target_link_libraries(odai_tween_demo PRIVATE glfw3::glfw)
        endif()
        if(TARGET slang_shaders)
            add_dependencies(odai_tween_demo slang_shaders)
        endif()
        if(MSVC)
            target_compile_options(odai_tween_demo PRIVATE ${ODAI_WARN_FLAGS_MSVC})
        else()
            target_compile_options(odai_tween_demo PRIVATE -Wall -Wextra -Wpedantic)
        endif()

    # ---------------------------------------------------------------------------
    # odai_design_system_demo — full component-library showcase: every widget,
    # animation, and UiDrawList effect in one tabbed app (Vulkan)
    # ---------------------------------------------------------------------------
        add_executable(odai_design_system_demo
            src/tools/design_system_demo/design_system_demo_main.cc
            src/tools/design_system_demo/design_system_demo_app.cc
            src/engine/game_app.cc
            src/engine/plugin.cc
            src/import/dds.cc
            src/import/gpu_scene.cc
            src/import/imported_scene.cc
            src/import/imported_scene_query.cc
            src/render/frame_graph.cc
            src/render/renderer.cc
            src/world/world.cc
            src/world/chunk_grid_worldgen.cc
            src/world/chunk_mesher.cc
            src/world/clipmap_index.cc
            src/world/magica_voxel.cc
            ${RENDER_BACKEND_SOURCES}
        )
        target_include_directories(odai_design_system_demo PRIVATE src ${ODAI_STB_INCLUDE_DIRS})
        target_compile_definitions(odai_design_system_demo PRIVATE
            ODAI_RENDER_BACKEND_VULKAN=1
            ODAI_HAS_VULKAN=1
            ODAI_HAS_GLFW=1
            ODAI_PROJECT_SOURCE_DIR=\"${CMAKE_SOURCE_DIR}\"
        )
        target_link_libraries(odai_design_system_demo PRIVATE
            odai_ui odai_ui_vulkan odai_audio
            Vulkan::Vulkan GPUOpen::VulkanMemoryAllocator imgui::imgui
        )
        if(TARGET glfw)
            target_link_libraries(odai_design_system_demo PRIVATE glfw)
        elseif(TARGET glfw3::glfw)
            target_link_libraries(odai_design_system_demo PRIVATE glfw3::glfw)
        endif()
        if(TARGET slang_shaders)
            add_dependencies(odai_design_system_demo slang_shaders)
        endif()
        if(MSVC)
            target_compile_options(odai_design_system_demo PRIVATE ${ODAI_WARN_FLAGS_MSVC})
        else()
            target_compile_options(odai_design_system_demo PRIVATE -Wall -Wextra -Wpedantic)
        endif()
        add_executable(odai_material_editor
            src/tools/material_editor/material_editor_main.cc
            src/tools/material_editor/material_editor_app.cc
            src/core/file_watch.cc
            src/procgen/csg.cc
            src/procgen/primitives.cc
            src/procgen/mesh_emit.cc
            src/procgen/building_generator.cc
            src/engine/game_app.cc
            src/engine/plugin.cc
            src/import/dds.cc
            src/import/gpu_scene.cc
            src/import/imported_scene.cc
            src/import/imported_scene_query.cc
            src/render/frame_graph.cc
            src/render/renderer.cc
            src/world/world.cc
            src/world/chunk_grid_worldgen.cc
            src/world/chunk_mesher.cc
            src/world/clipmap_index.cc
            src/world/magica_voxel.cc
            ${RENDER_BACKEND_SOURCES}
        )
        target_include_directories(odai_material_editor PRIVATE src ${ODAI_STB_INCLUDE_DIRS})
        target_compile_definitions(odai_material_editor PRIVATE
            ODAI_RENDER_BACKEND_VULKAN=1
            ODAI_HAS_VULKAN=1
            ODAI_HAS_GLFW=1
            ODAI_PROJECT_SOURCE_DIR=\"${CMAKE_SOURCE_DIR}\"
        )
        target_link_libraries(odai_material_editor PRIVATE
            odai_ui odai_ui_vulkan odai_audio odai_content
            Vulkan::Vulkan GPUOpen::VulkanMemoryAllocator imgui::imgui
        )
        if(TARGET glfw)
            target_link_libraries(odai_material_editor PRIVATE glfw)
        elseif(TARGET glfw3::glfw)
            target_link_libraries(odai_material_editor PRIVATE glfw3::glfw)
        endif()
        if(TARGET slang_shaders)
            add_dependencies(odai_material_editor slang_shaders)
        endif()
        if(MSVC)
            target_compile_options(odai_material_editor PRIVATE ${ODAI_WARN_FLAGS_MSVC})
        else()
            target_compile_options(odai_material_editor PRIVATE -Wall -Wextra -Wpedantic)
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
            src/import/imported_scene_query.cc
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
        src/game/buildable.cc
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

    # FNV texture-pack normalizer — mip-drops a downloaded texture mod to the
    # ceiling the runtime will ask for and lowercases its paths, producing a
    # directory to hand to `odai_game_newvegas --mod`.
    #   odai_fnv_texture_pack --in <dir> --out <dir> [--max-size 1024]
    add_executable(odai_fnv_texture_pack
        src/import/dds.cc
        src/tools/fnv_texture_pack_main.cc
    )
    target_include_directories(odai_fnv_texture_pack PRIVATE src)
    if(MSVC)
        target_compile_options(odai_fnv_texture_pack PRIVATE ${ODAI_WARN_FLAGS_MSVC})
    else()
        target_compile_options(odai_fnv_texture_pack PRIVATE -Wall -Wextra -Wpedantic)
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

    # New Vegas cooker — offline Fallout: New Vegas Data Files (.esm + .bsa +
    # .nif) -> ImportedScene .bin, mirroring odai_strategy_map_gen's role for
    # the hex map prototype. See src/tools/newvegas_cooker_main.cc for usage
    # and src/import/fnv/ for the format readers (BSA/ESM/NIF).
    #   odai_newvegas_cooker <DataFilesPath> <PluginName.esm> <output.bin> --cell <EditorID>
    #   odai_newvegas_cooker <DataFilesPath> <PluginName.esm> <output.bin> --worldspace <EditorID> <x0> <z0> <x1> <z1>
    add_executable(odai_newvegas_cooker
        src/tools/newvegas_cooker_main.cc
        src/import/dds.cc
        src/import/fnv/asset_source.cc
        src/import/fnv/bsa_archive.cc
        src/import/fnv/cell_builder.cc
        src/import/fnv/decoded_texture_cache.cc
        src/import/fnv/esm_reader.cc
        src/import/fnv/plugin_load_order.cc
        src/import/fnv/weather_records.cc
        src/import/fnv/fallout_records.cc
        src/import/fnv/nif_scene.cc
        src/import/imported_scene.cc
            src/import/imported_scene_query.cc
    )
    target_include_directories(odai_newvegas_cooker PRIVATE src)
    target_link_libraries(odai_newvegas_cooker PRIVATE ZLIB::ZLIB)
    if(MSVC)
        target_compile_options(odai_newvegas_cooker PRIVATE ${ODAI_WARN_FLAGS_MSVC})
    else()
        target_compile_options(odai_newvegas_cooker PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    # New Vegas probe — real-data harness for the BSA/ESM/NIF readers.
    # odai_fnv_import_tests is synthetic-fixture-only by design, so it cannot
    # catch a wrong assumption about the retail format; this is the tool that
    # can. Deliberately NOT registered with add_test: it needs an installed
    # copy of the game.
    #   odai_newvegas_probe <DataFilesPath> --archives | --nifs [limit] |
    #                                       --nif <path> | --plugin <Plugin.esm>
    add_executable(odai_newvegas_probe
        src/tools/newvegas_probe_main.cc
        src/import/dds.cc
        src/import/imported_scene.cc
        src/import/imported_scene_query.cc
        src/import/fnv/asset_source.cc
        src/import/fnv/cell_builder.cc
        src/import/fnv/decoded_texture_cache.cc
        src/import/fnv/bsa_archive.cc
        src/import/fnv/esm_reader.cc
        src/import/fnv/plugin_load_order.cc
        src/import/fnv/weather_records.cc
        src/import/fnv/fallout_records.cc
        src/import/fnv/dialogue_records.cc
        src/import/fnv/nif_scene.cc
        src/import/fnv/character_builder.cc
    )
    target_include_directories(odai_newvegas_probe PRIVATE src)
    target_link_libraries(odai_newvegas_probe PRIVATE ZLIB::ZLIB)
    if(MSVC)
        target_compile_options(odai_newvegas_probe PRIVATE ${ODAI_WARN_FLAGS_MSVC})
    else()
        target_compile_options(odai_newvegas_probe PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endif()

if(BUILD_TESTING)
    add_executable(odai_imported_scene_tests
        tests/imported_scene_tests.cc
        src/import/dds.cc
        src/import/gpu_scene.cc
        src/import/imported_scene.cc
            src/import/imported_scene_query.cc
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

    # Fallout: New Vegas import pipeline — BSA archive reader, generic ESM
    # record walker, typed record extraction, and the Gamebryo NIF geometry
    # parser (src/import/fnv/). All synthetic-fixture tests: no real Fallout:
    # New Vegas Data Files are available to test against in CI.
    add_executable(odai_fnv_import_tests
        tests/fnv_import_tests.cc
        src/core/job_system.cc
        src/import/fnv/asset_source.cc
        src/import/fnv/async_asset_loader.cc
        src/import/fnv/bsa_archive.cc
        src/import/fnv/esm_reader.cc
        src/import/fnv/plugin_load_order.cc
        src/import/fnv/weather_records.cc
        src/import/fnv/fallout_records.cc
        src/import/fnv/nif_scene.cc
        src/import/fnv/character_builder.cc
    )
    target_include_directories(odai_fnv_import_tests PRIVATE src)
    target_link_libraries(odai_fnv_import_tests PRIVATE ZLIB::ZLIB)
    if(MSVC)
        target_compile_options(odai_fnv_import_tests PRIVATE /W4 /permissive-)
    else()
        target_compile_options(odai_fnv_import_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME odai_fnv_import_tests COMMAND odai_fnv_import_tests)

    add_executable(odai_foundation_tests
        tests/foundation_tests.cc
        src/core/log.cc
        src/world/world.cc
        src/world/chunk_grid_worldgen.cc
        src/world/magica_voxel.cc
        src/world/chunk_mesher.cc
        src/world/clipmap_index.cc
        src/games/voxelcraft/voxelcraft_player.cc
        src/games/voxelcraft/voxelcraft_streaming.cc
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

    add_executable(odai_job_system_tests
        tests/job_system_tests.cc
        src/core/job_system.cc
    )
    target_include_directories(odai_job_system_tests PRIVATE src)
    target_link_libraries(odai_job_system_tests PRIVATE Threads::Threads)
    if(MSVC)
        target_compile_options(odai_job_system_tests PRIVATE /W4 /permissive-)
    else()
        target_compile_options(odai_job_system_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME odai_job_system_tests COMMAND odai_job_system_tests)

    add_executable(odai_chunk_mesh_scheduler_tests
        tests/chunk_mesh_scheduler_tests.cc
        src/core/job_system.cc
        src/world/chunk_mesher.cc
        src/world/chunk_mesh_scheduler.cc
        src/world/grass_scatter.cc
    )
    target_include_directories(odai_chunk_mesh_scheduler_tests PRIVATE src)
    target_link_libraries(odai_chunk_mesh_scheduler_tests PRIVATE Threads::Threads)
    if(MSVC)
        target_compile_options(odai_chunk_mesh_scheduler_tests PRIVATE /W4 /permissive-)
    else()
        target_compile_options(odai_chunk_mesh_scheduler_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME odai_chunk_mesh_scheduler_tests COMMAND odai_chunk_mesh_scheduler_tests)

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

    # BSP CSG toolkit + low-poly building primitives (procgen). Pure CPU;
    # no Vulkan, no UI.
    add_executable(odai_core_types_tests
        tests/core_types_tests.cc
    )
    target_include_directories(odai_core_types_tests PRIVATE src)
    if(MSVC)
        target_compile_options(odai_core_types_tests PRIVATE /W4 /permissive-)
    else()
        target_compile_options(odai_core_types_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME odai_core_types_tests COMMAND odai_core_types_tests)

    # Refcounted bindless slot bookkeeping (render/bindless_slot_table.h) that
    # imported-texture streaming rests on. Deliberately Vulkan-free so it can be
    # tested here at all -- no test executable in this project links Vulkan.
    add_executable(odai_bindless_slot_table_tests
        tests/bindless_slot_table_tests.cc
    )
    target_include_directories(odai_bindless_slot_table_tests PRIVATE src)
    if(MSVC)
        target_compile_options(odai_bindless_slot_table_tests PRIVATE /W4 /permissive-)
    else()
        target_compile_options(odai_bindless_slot_table_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME odai_bindless_slot_table_tests COMMAND odai_bindless_slot_table_tests)

    # Geometry arena suballocator (render/gpu_arena_allocator.h) behind imported
    # scene streaming. Also Vulkan-free; the coalescing tests are the guard
    # against an arena that fragments slowly over a long session.
    add_executable(odai_gpu_arena_allocator_tests
        tests/gpu_arena_allocator_tests.cc
    )
    target_include_directories(odai_gpu_arena_allocator_tests PRIVATE src)
    if(MSVC)
        target_compile_options(odai_gpu_arena_allocator_tests PRIVATE /W4 /permissive-)
    else()
        target_compile_options(odai_gpu_arena_allocator_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME odai_gpu_arena_allocator_tests COMMAND odai_gpu_arena_allocator_tests)

    # Exterior-cell streaming residency policy (import/cell_residency_planner.h):
    # prediction, load/unload hysteresis, in-flight budget, wasted-work counting.
    # Pure decision logic -- no Vulkan, no IO, no threads -- so the behaviours
    # that would otherwise only show up as runtime stutter are testable.
    add_executable(odai_cell_residency_planner_tests
        tests/cell_residency_planner_tests.cc
        src/import/cell_residency_planner.cc
    )
    target_include_directories(odai_cell_residency_planner_tests PRIVATE src)
    if(MSVC)
        target_compile_options(odai_cell_residency_planner_tests PRIVATE /W4 /permissive-)
    else()
        target_compile_options(odai_cell_residency_planner_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME odai_cell_residency_planner_tests COMMAND odai_cell_residency_planner_tests)

    # GameApp CPU timing zones (engine/game_frame_stats.h). Header-only and
    # Vulkan-free, so the zone bookkeeping is testable without a window.
    add_executable(odai_engine_stats_tests
        tests/engine_frame_stats_tests.cc
    )
    target_include_directories(odai_engine_stats_tests PRIVATE src)
    if(MSVC)
        target_compile_options(odai_engine_stats_tests PRIVATE /W4 /permissive-)
    else()
        target_compile_options(odai_engine_stats_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME odai_engine_stats_tests COMMAND odai_engine_stats_tests)

    add_executable(odai_procgen_tests
        tests/procgen_tests.cc
        src/procgen/csg.cc
        src/procgen/primitives.cc
        src/procgen/mesh_emit.cc
        src/procgen/building_generator.cc
        src/procgen/props.cc
        src/procgen/city_terrain.cc
        src/procgen/civic_generator.cc
    )
    target_include_directories(odai_procgen_tests PRIVATE src)
    if(MSVC)
        target_compile_options(odai_procgen_tests PRIVATE /W4 /permissive-)
    else()
        target_compile_options(odai_procgen_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME odai_procgen_tests COMMAND odai_procgen_tests)

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
        src/game/buildable.cc
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

    add_executable(odai_material_library_tests
        tests/material_library_tests.cc
        src/import/imported_scene.cc
            src/import/imported_scene_query.cc
    )
    target_include_directories(odai_material_library_tests PRIVATE src)
    target_link_libraries(odai_material_library_tests PRIVATE odai_content ZLIB::ZLIB)
    if(MSVC)
        target_compile_options(odai_material_library_tests PRIVATE /W4 /permissive-)
    else()
        target_compile_options(odai_material_library_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME odai_material_library_tests COMMAND odai_material_library_tests)

    # Dialogue trees (Dragon Age touchstone): JSON loading, condition filtering,
    # effect application, auto-continue chains, and dangling-reference/cycle
    # guards. Pure CPU; no Vulkan, no game-state dependency.
    add_executable(odai_dialogue_tests tests/dialogue_tests.cc)
    target_include_directories(odai_dialogue_tests PRIVATE src)
    target_link_libraries(odai_dialogue_tests PRIVATE odai_dialogue)
    if(MSVC)
        target_compile_options(odai_dialogue_tests PRIVATE /W4 /permissive-)
    else()
        target_compile_options(odai_dialogue_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME odai_dialogue_tests COMMAND odai_dialogue_tests)

    # Skeleton/animation-clip sampler (Dragon Age touchstone): bind-pose
    # sanity, hierarchy composition, keyframe interpolation, loop/clamp time
    # handling, and JSON loading. Pure CPU; no Vulkan.
    add_executable(odai_animation_tests tests/animation_tests.cc)
    target_include_directories(odai_animation_tests PRIVATE src)
    target_link_libraries(odai_animation_tests PRIVATE odai_anim)
    if(MSVC)
        target_compile_options(odai_animation_tests PRIVATE /W4 /permissive-)
    else()
        target_compile_options(odai_animation_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME odai_animation_tests COMMAND odai_animation_tests)

    # PBR materials: the metallic-roughness flag packing, the humanoid generator's
    # per-color-group material assignment (and that leaving it unset is a no-op),
    # and a C++ transcription of src/render/shaders/pbr.slang checked against
    # reference values -- the GGX normalization identity, Schlick's endpoints, and
    # an energy-conservation bound that pins the diffuse-convention pi scale.
    # Pure CPU; no Vulkan.
    add_executable(odai_pbr_material_tests
        tests/pbr_material_tests.cc
        src/procgen/humanoid_generator.cc
        src/import/imported_scene.cc
            src/import/imported_scene_query.cc
    )
    target_include_directories(odai_pbr_material_tests PRIVATE src)
    target_link_libraries(odai_pbr_material_tests PRIVATE odai_anim ZLIB::ZLIB)
    if(MSVC)
        target_compile_options(odai_pbr_material_tests PRIVATE /W4 /permissive-)
    else()
        target_compile_options(odai_pbr_material_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME odai_pbr_material_tests COMMAND odai_pbr_material_tests)

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

    # Citybuilder Lua content layer: sandbox holds, registrations round-trip,
    # name generation is deterministic per seed, broken scripts fall back to the
    # compiled-in defaults, and the shipped mods/citybuilder scripts load clean.
    add_executable(odai_city_script_tests tests/city_script_tests.cc)
    target_include_directories(odai_city_script_tests PRIVATE src)
    target_link_libraries(odai_city_script_tests PRIVATE odai_city_script)
    add_test(NAME odai_city_script_tests COMMAND odai_city_script_tests)

    # Citybuilder spatial fields: influence/coverage tables, the disc splat and
    # the population-weighted mean. No Vulkan, no GameApp — the whole point of
    # keeping this math out of citybuilder_app.cc is that it tests headlessly.
    add_executable(odai_city_fields_tests
        tests/city_fields_tests.cc
        src/games/citybuilder/citybuilder_fields.cc
    )
    target_include_directories(odai_city_fields_tests PRIVATE src)
    if(MSVC)
        target_compile_options(odai_city_fields_tests PRIVATE ${ODAI_WARN_FLAGS_MSVC})
    else()
        target_compile_options(odai_city_fields_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME odai_city_fields_tests COMMAND odai_city_fields_tests)

    add_executable(odai_ui_tests tests/ui_tests.cc)
    target_link_libraries(odai_ui_tests PRIVATE odai_ui)
    # Absolute path to a real TTF for tests that bake an actual atlas (e.g. the
    # SDF glyph bake structure test) instead of the synthetic monospace font.
    target_compile_definitions(odai_ui_tests PRIVATE
        ODAI_TEST_ASSETS_DIR="${CMAKE_SOURCE_DIR}/assets")
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

    # UI editor document core: real .ui.json schema authoring, layout/hit-test,
    # undo/redo, snapping, align/distribute/clipboard, HSV + color harmony +
    # WCAG contrast. Headless — links odai_ui
    # plus the Vulkan-free half of src/tools/ui_editor/, and round-trips its
    # output through the production ui::UiDocumentLoader.
    add_executable(odai_ui_editor_tests
        tests/ui_editor_tests.cc
        ${ODAI_UI_EDITOR_CORE_SOURCES}
    )
    target_include_directories(odai_ui_editor_tests PRIVATE src)
    target_link_libraries(odai_ui_editor_tests PRIVATE odai_ui)
    # Lets the suite open the engine's own shipped .ui.json sample.
    target_compile_definitions(odai_ui_editor_tests PRIVATE
        ODAI_TEST_ASSETS_DIR="${CMAKE_SOURCE_DIR}/assets")
    if(MSVC)
        target_compile_options(odai_ui_editor_tests PRIVATE ${ODAI_WARN_FLAGS_MSVC})
    else()
        target_compile_options(odai_ui_editor_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME odai_ui_editor_tests COMMAND odai_ui_editor_tests)

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
