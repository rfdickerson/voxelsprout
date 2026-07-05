# Shared headless capture harness for the odai_ui stress test contestants.
# No window, no swapchain, no display required — renders straight to an
# off-screen image and dumps a PNG, so every contestant links this instead of
# writing their own Vulkan bootstrap / screenshot code.
add_library(ui_stress_capture STATIC
    offscreen_capture.cc
    vma_impl.cc
)
target_include_directories(ui_stress_capture
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE ${ODAI_STB_INCLUDE_DIRS}
)
target_compile_definitions(ui_stress_capture PRIVATE
    ODAI_PROJECT_SOURCE_DIR=\"${CMAKE_SOURCE_DIR}\"
)
target_link_libraries(ui_stress_capture PUBLIC odai_ui odai_ui_vulkan)

if(MSVC)
    target_compile_options(ui_stress_capture PRIVATE ${ODAI_WARN_FLAGS_MSVC})
else()
    target_compile_options(ui_stress_capture PRIVATE -Wall -Wextra -Wpedantic)
endif()
