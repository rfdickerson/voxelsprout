add_executable(ui_stress_harness_smoke_test main.cc)
target_link_libraries(ui_stress_harness_smoke_test PRIVATE ui_stress_capture)
target_include_directories(ui_stress_harness_smoke_test PRIVATE ${ODAI_STB_INCLUDE_DIRS})
target_compile_definitions(ui_stress_harness_smoke_test PRIVATE
    ODAI_PROJECT_SOURCE_DIR=\"${CMAKE_SOURCE_DIR}\"
)
if(TARGET slang_shaders)
    add_dependencies(ui_stress_harness_smoke_test slang_shaders)
endif()
if(MSVC)
    target_compile_options(ui_stress_harness_smoke_test PRIVATE ${ODAI_WARN_FLAGS_MSVC})
else()
    target_compile_options(ui_stress_harness_smoke_test PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(BUILD_TESTING)
    add_test(NAME ui_stress_harness_smoke_test COMMAND ui_stress_harness_smoke_test)
endif()
