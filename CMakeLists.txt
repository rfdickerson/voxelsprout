cmake_minimum_required(VERSION 3.21)
project(voxel_factory_toy VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
include(CTest)

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
                "${SLANG_SHADER}"
                -o "${SLANG_OUTPUT}"
            DEPENDS "${SLANG_SHADER}" ${SLANG_SHARED_INCLUDES}
            COMMENT "Compiling Slang shader ${SLANG_SHADER}"
            VERBATIM
        )
        list(APPEND SLANG_SHADER_OUTPUTS "${SLANG_OUTPUT}")
        set(SLANG_SHADER_OUTPUTS "${SLANG_SHADER_OUTPUTS}" PARENT_SCOPE)
    endfunction()

    add_slang_shader("${CMAKE_SOURCE_DIR}/src/render/shaders/voxel_packed.vert.slang" vertex)
    add_slang_shader("${CMAKE_SOURCE_DIR}/src/render/shaders/voxel_packed.frag.slang" fragment)
    add_slang_shader("${CMAKE_SOURCE_DIR}/src/render/shaders/voxel_normaldepth.frag.slang" fragment)
    add_slang_shader("${CMAKE_SOURCE_DIR}/src/render/shaders/skybox.vert.slang" vertex)
    add_slang_shader("${CMAKE_SOURCE_DIR}/src/render/shaders/skybox.frag.slang" fragment)
    add_slang_shader("${CMAKE_SOURCE_DIR}/src/render/shaders/tone_map.vert.slang" vertex)
    add_slang_shader("${CMAKE_SOURCE_DIR}/src/render/shaders/tone_map.frag.slang" fragment)
    add_slang_shader("${CMAKE_SOURCE_DIR}/src/render/shaders/shadow_depth.vert.slang" vertex)
    add_slang_shader("${CMAKE_SOURCE_DIR}/src/render/shaders/shadow_depth.frag.slang" fragment)
    add_slang_shader("${CMAKE_SOURCE_DIR}/src/render/shaders/pipe_shadow.vert.slang" vertex)
    add_slang_shader("${CMAKE_SOURCE_DIR}/src/render/shaders/pipe_shadow.frag.slang" fragment)
    add_slang_shader("${CMAKE_SOURCE_DIR}/src/render/shaders/grass_billboard_shadow.vert.slang" vertex)
    add_slang_shader("${CMAKE_SOURCE_DIR}/src/render/shaders/grass_billboard_shadow.frag.slang" fragment)
    add_slang_shader("${CMAKE_SOURCE_DIR}/src/render/shaders/pipe_instanced.vert.slang" vertex)
    add_slang_shader("${CMAKE_SOURCE_DIR}/src/render/shaders/pipe_instanced.frag.slang" fragment)
    add_slang_shader("${CMAKE_SOURCE_DIR}/src/render/shaders/grass_billboard.vert.slang" vertex)
    add_slang_shader("${CMAKE_SOURCE_DIR}/src/render/shaders/grass_billboard.frag.slang" fragment)
    add_slang_shader("${CMAKE_SOURCE_DIR}/src/render/shaders/pipe_normaldepth.frag.slang" fragment)
    add_slang_shader("${CMAKE_SOURCE_DIR}/src/render/shaders/ssao.frag.slang" fragment)
    add_slang_shader("${CMAKE_SOURCE_DIR}/src/render/shaders/ssao_blur.frag.slang" fragment)
    add_custom_target(slang_shaders DEPENDS ${SLANG_SHADER_OUTPUTS})
else()
    message(STATUS "slangc not found; skipping Slang shader target")
endif()

add_executable(voxel_factory_toy
    src/app/main.cpp
    src/app/App.cpp
    src/core/Log.cpp
    src/render/BufferHelpers.cpp
    src/render/Renderer.cpp
    src/render/VmaUsage.cpp
    src/world/ChunkMesher.cpp
    src/world/ClipmapIndex.cpp
)

if(TARGET slang_shaders)
    add_dependencies(voxel_factory_toy slang_shaders)
endif()

target_include_directories(voxel_factory_toy PRIVATE src)

if(MSVC)
    target_compile_options(voxel_factory_toy PRIVATE
        /W4
        /permissive-
        /w14242 # possible data loss in conversion
        /w14254 # possible loss of data from conversion/operator
        /w14263 # member function does not override any base virtual member function
        /w14265 # class has virtual functions, but destructor is not virtual
        /w14287 # unsigned/negative constant mismatch
        /we4289 # loop control variable used outside loop scope
        /w14296 # expression is always true/false
        /w14311 # pointer truncation
        /w14545 # expression before comma has no effect
        /w14546 # function call before comma missing argument list
        /w14547 # operator before comma has no effect
        /w14549 # operator before comma has no effect
        /w14555 # expression has no effect
        /w14619 # pragma warning: no warning number
        /w14640 # thread-unsafe static member initialization
        /w14826 # conversion is sign-extended
        /w14905 # wide string literal cast to LPCSTR
        /w14906 # string literal cast to LPWSTR
        /w14928 # illegal copy-initialization
    )
else()
    target_compile_options(voxel_factory_toy PRIVATE -Wall -Wextra -Wpedantic)
endif()

# GLFW is required for the minimal window + input runtime.
find_package(glfw3 CONFIG QUIET)
find_package(OpenGL REQUIRED)

# Optional third-party dependencies for future rendering integration.
find_package(Vulkan QUIET)
find_package(imgui CONFIG QUIET)
find_package(unofficial-vulkan-memory-allocator CONFIG QUIET)

if(Vulkan_FOUND)
    target_link_libraries(voxel_factory_toy PRIVATE Vulkan::Vulkan)
    target_compile_definitions(voxel_factory_toy PRIVATE VOXEL_HAS_VULKAN=1)
endif()

if(TARGET glfw)
    target_link_libraries(voxel_factory_toy PRIVATE glfw)
    target_compile_definitions(voxel_factory_toy PRIVATE VOXEL_HAS_GLFW=1)
elseif(TARGET glfw3::glfw)
    target_link_libraries(voxel_factory_toy PRIVATE glfw3::glfw)
    target_compile_definitions(voxel_factory_toy PRIVATE VOXEL_HAS_GLFW=1)
else()
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(GLFW3 QUIET IMPORTED_TARGET glfw3)
        if(TARGET PkgConfig::GLFW3)
            target_link_libraries(voxel_factory_toy PRIVATE PkgConfig::GLFW3)
            target_compile_definitions(voxel_factory_toy PRIVATE VOXEL_HAS_GLFW=1)
        else()
            message(FATAL_ERROR "GLFW is required. Install via vcpkg or provide glfw3 to CMake.")
        endif()
    else()
        message(FATAL_ERROR "GLFW is required. Install via vcpkg or provide glfw3 to CMake.")
    endif()
endif()

target_link_libraries(voxel_factory_toy PRIVATE OpenGL::GL)

if(TARGET imgui::imgui)
    target_link_libraries(voxel_factory_toy PRIVATE imgui::imgui)
    target_compile_definitions(voxel_factory_toy PRIVATE VOXEL_HAS_IMGUI=1)
endif()

if(TARGET unofficial::VulkanMemoryAllocator::VulkanMemoryAllocator)
    target_link_libraries(voxel_factory_toy PRIVATE unofficial::VulkanMemoryAllocator::VulkanMemoryAllocator)
    target_compile_definitions(voxel_factory_toy PRIVATE VOXEL_HAS_VMA=1)
else()
    set(VMA_HEADER_HINTS
        "$ENV{VCPKG_ROOT}/installed/$ENV{VCPKG_DEFAULT_TRIPLET}/include"
        "$ENV{VCPKG_ROOT}/packages/vulkan-memory-allocator_$ENV{VCPKG_DEFAULT_TRIPLET}/include"
        "/mnt/c/Users/rfdic/vcpkg/packages/vulkan-memory-allocator_x64-windows/include"
        "/mnt/c/Users/rfdic/vcpkg/installed/x64-windows/include"
    )
    find_path(VMA_INCLUDE_DIR
        NAMES vk_mem_alloc.h vma/vk_mem_alloc.h
        HINTS ${VMA_HEADER_HINTS}
    )
    if(VMA_INCLUDE_DIR)
        message(STATUS "Using VMA header-only include path: ${VMA_INCLUDE_DIR}")
        target_include_directories(voxel_factory_toy PRIVATE "${VMA_INCLUDE_DIR}")
        target_compile_definitions(voxel_factory_toy PRIVATE VOXEL_HAS_VMA=1 VOXEL_VMA_HEADER_ONLY=1)
    else()
        message(WARNING "VMA not found. Renderer will fall back to raw Vulkan allocations.")
    endif()
endif()

if(BUILD_TESTING)
    add_executable(voxel_foundation_tests
        tests/FoundationTests.cpp
        src/world/ChunkMesher.cpp
        src/world/ClipmapIndex.cpp
    )
    target_include_directories(voxel_foundation_tests PRIVATE src)

    if(MSVC)
        target_compile_options(voxel_foundation_tests PRIVATE
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
        target_compile_options(voxel_foundation_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    add_test(NAME voxel_foundation_tests COMMAND voxel_foundation_tests)
endif()
