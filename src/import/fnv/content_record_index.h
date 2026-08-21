#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

#include "import/fnv/plugin_load_order.h"

namespace odai::importer::fnv {

struct ContentRecordVersion {
    std::uint32_t globalFormId = 0;
    std::string type;
    std::size_t pluginIndex = 0;
    std::string pluginName;
    std::filesystem::path pluginPath;
    std::uint32_t flags = 0;
    bool deleted = false;
};

// Header-only provenance index. It does not decompress record bodies and is
// therefore cheap enough for profile diagnostics over thousands of plugins.
// Versions are ascending priority; the last version is the active winner.
class ContentRecordIndex {
public:
    bool build(const FalloutLoadOrder& order, std::string& outError);

    [[nodiscard]] const std::vector<ContentRecordVersion>* versions(
        std::uint32_t globalFormId) const;
    [[nodiscard]] std::size_t recordCount() const { return m_versions.size(); }
    [[nodiscard]] std::size_t overrideCount() const { return m_overrideCount; }
    [[nodiscard]] std::size_t deletionCount() const { return m_deletionCount; }

private:
    std::unordered_map<std::uint32_t, std::vector<ContentRecordVersion>> m_versions;
    std::size_t m_overrideCount = 0u;
    std::size_t m_deletionCount = 0u;
};

}  // namespace odai::importer::fnv
