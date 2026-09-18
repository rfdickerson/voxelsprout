#pragma once
#include <filesystem>
namespace odai::core {
// Resources are relative to the executable, never the checkout or working directory.
// ODAI_RESOURCE_DIR overrides the root for development and package verification.
std::filesystem::path resourcePath(const std::filesystem::path& name);
}
