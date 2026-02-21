cmake_minimum_required(VERSION 3.21)
project(voxelsprout_compute_lab VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CTest)

find_program(SLANGC_EXECUTABLE NAMES slangc)
if(SLANGC_EXECUTABLE)
    set(SLANG_SHADER_OUTPUTS "")
    function(add_slang_shader SLANG_SHADER)
        set(SLANG_OUTPUT "${SLANG_SHADER}.spv")
        add_custom_command(
            OUTPUT "${SLANG_OUTPUT}"
            COMMAND "${SLANGC_EXECUTABLE}"
                -target spirv
                -entry main
                -stage compute
                -matrix-layout-column-major
                -I "${CMAKE_SOURCE_DIR}/shaders"
                "${SLANG_SHADER}"
                -o "${SLANG_OUTPUT}"
            DEPENDS "${SLANG_SHADER}"
            COMMENT "Compiling Slang shader ${SLANG_SHADER}"
            VERBATIM
        )
        list(APPEND SLANG_SHADER_OUTPUTS "${SLANG_OUTPUT}")
        set(SLANG_SHADER_OUTPUTS "${SLANG_SHADER_OUTPUTS}" PARENT_SCOPE)
    endfunction()

    add_slang_shader("${CMAKE_SOURCE_DIR}/shaders/cloud_path_trace.comp.slang")
    add_slang_shader("${CMAKE_SOURCE_DIR}/shaders/tone_map.comp.slang")
    add_custom_target(slang_shaders DEPENDS ${SLANG_SHADER_OUTPUTS})
else()
    message(STATUS "slangc not found; skipping Slang shader target")
endif()

add_executable(voxelsprout_compute_lab
    app/main.cc
    app/app.cc
    core/log.cc
    core/camera.cc
    core/noise_field.cc
    render/vma_usage.cc
    render/renderer.cc
)

target_include_directories(voxelsprout_compute_lab PRIVATE .)
target_compile_definitions(voxelsprout_compute_lab PRIVATE VOXEL_PROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

if(MSVC)
    target_compile_options(voxelsprout_compute_lab PRIVATE /W4 /permissive-)
else()
    target_compile_options(voxelsprout_compute_lab PRIVATE -Wall -Wextra -Wpedantic)
endif()

find_package(glfw3 CONFIG QUIET)
find_package(Vulkan REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(VulkanMemoryAllocator CONFIG REQUIRED)

if(TARGET glfw)
    target_link_libraries(voxelsprout_compute_lab PRIVATE glfw)
elseif(TARGET glfw3::glfw)
    target_link_libraries(voxelsprout_compute_lab PRIVATE glfw3::glfw)
else()
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(GLFW3 QUIET IMPORTED_TARGET glfw3)
        if(TARGET PkgConfig::GLFW3)
            target_link_libraries(voxelsprout_compute_lab PRIVATE PkgConfig::GLFW3)
        else()
            message(FATAL_ERROR "GLFW is required. Install via vcpkg or provide glfw3 to CMake.")
        endif()
    else()
        message(FATAL_ERROR "GLFW is required. Install via vcpkg or provide glfw3 to CMake.")
    endif()
endif()

target_link_libraries(voxelsprout_compute_lab PRIVATE
    Vulkan::Vulkan
    GPUOpen::VulkanMemoryAllocator
    imgui::imgui
)

if(TARGET slang_shaders)
    add_dependencies(voxelsprout_compute_lab slang_shaders)
endif()

if(BUILD_TESTING)
    add_executable(voxelsprout_math_tests tests/math_sampling_tests.cc)
    target_include_directories(voxelsprout_math_tests PRIVATE .)
    if(MSVC)
        target_compile_options(voxelsprout_math_tests PRIVATE /W4 /permissive-)
    else()
        target_compile_options(voxelsprout_math_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME voxelsprout_math_tests COMMAND voxelsprout_math_tests)
endif()
