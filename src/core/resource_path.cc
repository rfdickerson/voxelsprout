#include "core/resource_path.h"
#include <cstdlib>
#include <string>
namespace odai::core {
std::filesystem::path resourcePath(const std::filesystem::path& name) {
    std::filesystem::path relative = name;
    if (name.extension() == ".spv") relative = std::filesystem::path("shaders") / name.filename();
    if (const char* root = std::getenv("ODAI_RESOURCE_DIR"); root && *root)
        return std::filesystem::path(root) / relative;
    std::error_code error;
    const auto executable = std::filesystem::read_symlink("/proc/self/exe", error);
    if (!error) {
        const auto directory = executable.parent_path();
        const auto installed = directory / "../share/odai";
        if (std::filesystem::is_directory(installed, error)) return installed / relative;
        return directory / "share/odai" / relative;
    }
    return std::filesystem::path("share/odai") / relative;
}
}
