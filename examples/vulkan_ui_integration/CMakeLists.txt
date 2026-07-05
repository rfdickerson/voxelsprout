# Minimal standalone integration sample: proves odai_ui + odai_ui_vulkan can be
# embedded in a from-scratch Vulkan app with no dependency on this repo's game,
# world, or app modules — only the two SDK libraries plus GLFW and Vulkan.
add_executable(odai_ui_integration_sample main.cc vma_impl.cc)

target_compile_definitions(odai_ui_integration_sample PRIVATE
    ODAI_UI_SAMPLE_SOURCE_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}\"
    ODAI_PROJECT_SOURCE_DIR=\"${CMAKE_SOURCE_DIR}\"
)
target_link_libraries(odai_ui_integration_sample PRIVATE odai_ui odai_ui_vulkan)

if(TARGET glfw)
    target_link_libraries(odai_ui_integration_sample PRIVATE glfw)
elseif(TARGET glfw3::glfw)
    target_link_libraries(odai_ui_integration_sample PRIVATE glfw3::glfw)
else()
    find_package(glfw3 CONFIG REQUIRED)
    target_link_libraries(odai_ui_integration_sample PRIVATE glfw)
endif()

if(TARGET slang_shaders)
    add_dependencies(odai_ui_integration_sample slang_shaders)
endif()

if(MSVC)
    target_compile_options(odai_ui_integration_sample PRIVATE ${ODAI_WARN_FLAGS_MSVC})
else()
    target_compile_options(odai_ui_integration_sample PRIVATE -Wall -Wextra -Wpedantic)
endif()
