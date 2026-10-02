# Engine-owned resources only. Never install whole assets/ or captures/ trees.
set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib;$ORIGIN")
if(TARGET odai)
    set_target_properties(odai PROPERTIES INSTALL_RPATH "$ORIGIN/../lib;$ORIGIN")
    install(TARGETS odai RUNTIME_DEPENDENCY_SET odai_dependencies RUNTIME DESTINATION bin)
endif()
foreach(tool odai_bethesda_probe odai_bethesda_cooker odai_texture_pack)
    if(TARGET ${tool})
        set_target_properties(${tool} PROPERTIES INSTALL_RPATH "$ORIGIN/../lib;$ORIGIN")
        install(TARGETS ${tool} RUNTIME_DEPENDENCY_SET odai_dependencies RUNTIME DESTINATION bin)
    endif()
endforeach()
if(UNIX AND NOT APPLE)
    # glibc, the Vulkan loader and GPU drivers remain supplied by the host.
    install(RUNTIME_DEPENDENCY_SET odai_dependencies
        PRE_EXCLUDE_REGEXES "^ld-linux" "^lib(c|m|dl|pthread|rt|resolv)\\.so" "^libvulkan\\.so"
        POST_EXCLUDE_REGEXES "ld-linux.*" LIBRARY DESTINATION lib)
endif()
install(DIRECTORY "${ODAI_RESOURCE_DIR}/shaders/" DESTINATION share/odai/shaders
    FILES_MATCHING PATTERN "*.spv")
install(DIRECTORY "${CMAKE_SOURCE_DIR}/assets/fonts/" DESTINATION share/odai/assets/fonts
    FILES_MATCHING PATTERN "Inter-*.ttf" PATTERN "LICENSE-Inter.txt")
install(PROGRAMS "${CMAKE_SOURCE_DIR}/scripts/odai-launcher" DESTINATION bin)
install(FILES "${CMAKE_SOURCE_DIR}/packaging/skyrim-slice.json" DESTINATION share/odai)
install(FILES "${CMAKE_SOURCE_DIR}/packaging/odai.desktop" DESTINATION share/applications)
install(FILES "${CMAKE_SOURCE_DIR}/LICENSE" "${CMAKE_SOURCE_DIR}/THIRD_PARTY_NOTICES.md"
    "${CMAKE_SOURCE_DIR}/README.md" "${CMAKE_SOURCE_DIR}/CONTRIBUTING.md"
    "${CMAKE_SOURCE_DIR}/SECURITY.md" "${CMAKE_SOURCE_DIR}/CHANGELOG.md"
    DESTINATION share/doc/odai)
install(DIRECTORY "${CMAKE_SOURCE_DIR}/docs/" DESTINATION share/doc/odai/docs
    FILES_MATCHING PATTERN "*.md")
install(FILES "${CMAKE_SOURCE_DIR}/packaging/skyrim-slice.json"
    DESTINATION share/doc/odai/packaging)
# vcpkg ports carry their dependency license texts alongside installed metadata.
if(DEFINED VCPKG_INSTALLED_DIR AND DEFINED VCPKG_TARGET_TRIPLET)
    install(DIRECTORY "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/share/"
        DESTINATION share/doc/odai/dependencies FILES_MATCHING PATTERN copyright)
endif()
# GCC runtime libraries may be bundled by dependency resolution. Preserve the
# host distribution's license texts, including the GCC Runtime Library Exception.
if(EXISTS "/usr/share/licenses/libgcc")
    install(DIRECTORY /usr/share/licenses/libgcc/ DESTINATION share/doc/odai/gcc-runtime)
elseif(EXISTS "/usr/share/doc/libstdc++6/copyright")
    install(FILES /usr/share/doc/libstdc++6/copyright DESTINATION share/doc/odai/gcc-runtime)
endif()
set(CPACK_PACKAGE_NAME odai)
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CPACK_PACKAGE_FILE_NAME "odai-${PROJECT_VERSION}-experimental-${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Experimental Skyrim runtime — playable release gates pending")
set(CPACK_GENERATOR TGZ)
set(CPACK_PACKAGE_CHECKSUM SHA256)
set(CPACK_PACKAGING_INSTALL_PREFIX "/")
include(CPack)
