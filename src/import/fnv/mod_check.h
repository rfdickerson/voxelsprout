#pragma once

#include "import/fnv/content_profile.h"
#include <iosfwd>
#include <optional>

namespace odai::importer::fnv {
struct ModDiagnostic {
    ContentDiagnosticSeverity severity = ContentDiagnosticSeverity::Info;
    std::string code;
    std::string message;
    std::filesystem::path source;
    std::optional<std::uint32_t> formId;
    std::optional<std::uint64_t> fileOffset;
};
struct ModCheckReport {
    std::vector<ModDiagnostic> diagnostics;
    std::vector<std::string> requestedOrder;
    std::vector<std::string> resolvedOrder;
    std::vector<std::string> changes;
    std::size_t overrideCount = 0;
    [[nodiscard]] bool hasErrors() const;
};
// Header/dependency preflight is shared with FalloutLoadOrder::open. Full scans
// are opt-in here and are never required by runtime startup.
ModCheckReport checkSkyrimMods(const ResolvedContentProfile& profile, bool scanRecords = true);
// argv contains only arguments after the subcommand. No renderer initialization.
int runModCheckCommand(int argc, const char* const* argv, std::ostream& out, std::ostream& err);
} // namespace odai::importer::fnv
