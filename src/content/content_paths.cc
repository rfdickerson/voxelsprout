#include "content/content_paths.h"

#include <system_error>
#include <vector>

namespace odai::content {

std::filesystem::path resolveContentPath(const std::filesystem::path& relative) {
    if (relative.is_absolute()) {
        return relative;
    }

    std::vector<std::filesystem::path> bases;
#if defined(ODAI_PROJECT_SOURCE_DIR)
    bases.emplace_back(std::filesystem::path{ODAI_PROJECT_SOURCE_DIR});
#endif
    std::error_code ec;
    const std::filesystem::path cwd = std::filesystem::current_path(ec);
    if (!ec) {
        bases.push_back(cwd);
        bases.push_back(cwd / "..");
        bases.push_back(cwd / ".." / "..");
        bases.push_back(cwd / ".." / ".." / "..");
    }

    for (const std::filesystem::path& base : bases) {
        std::error_code existsErr;
        const std::filesystem::path candidate = base / relative;
        if (std::filesystem::exists(candidate, existsErr)) {
            return candidate;
        }
    }
    return relative;
}

}  // namespace odai::content
