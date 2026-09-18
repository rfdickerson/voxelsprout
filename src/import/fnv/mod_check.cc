#include "import/fnv/mod_check.h"
#include "import/fnv/content_record_index.h"
#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <nlohmann/json.hpp>

namespace odai::importer::fnv {
namespace {
std::string lower(std::string value) {
    for (char& c : value) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return value;
}
const char* severityName(ContentDiagnosticSeverity severity) {
    switch (severity) {
        case ContentDiagnosticSeverity::Error: return "error";
        case ContentDiagnosticSeverity::Warning: return "warning";
        default: return "info";
    }
}
std::string orderErrorCode(const std::string& error) {
    for (const char* code : {"dependency-cycle", "missing-plugin", "incompatible-layout", "malformed-header"}) {
        if (error.find(code) != std::string::npos) return code;
    }
    if (error.find("more than") != std::string::npos) return "slot-limit";
    return "malformed-header";
}
}
bool ModCheckReport::hasErrors() const {
    return std::any_of(diagnostics.begin(), diagnostics.end(), [](const auto& d) {
        return d.severity == ContentDiagnosticSeverity::Error;
    });
}
ModCheckReport checkSkyrimMods(const ResolvedContentProfile& profile, bool scanRecords) {
    ModCheckReport report;
    report.requestedOrder = profile.plugins;
    auto add = [&](ContentDiagnosticSeverity severity, std::string code, std::string message,
                   std::filesystem::path source = {}, std::optional<std::uint32_t> id = {},
                   std::optional<std::uint64_t> offset = {}) {
        report.diagnostics.push_back({severity, std::move(code), std::move(message), std::move(source), id, offset});
    };
    for (const auto& d : profile.diagnostics) add(d.severity, d.code, d.message, d.source);
    if (profile.game != BethesdaGame::SkyrimSpecialEdition) {
        add(ContentDiagnosticSeverity::Error, "unsupported-game", "mod-check currently requires a Skyrim SE profile", profile.sourcePath);
        return report;
    }
    FalloutLoadOrder order;
    std::string error;
    if (!order.open(profile, error)) {
        add(ContentDiagnosticSeverity::Error, orderErrorCode(error), error, order.lastErrorSource());
        return report;
    }
    if (order.empty()) {
        add(ContentDiagnosticSeverity::Error, "empty-order", "profile has no active plugins", profile.sourcePath);
        return report;
    }
    // Stable Kahn order: requested plugins have their original priority;
    // implicit masters follow discovery order when no requested node is ready.
    std::vector<std::size_t> candidates(order.size());
    std::iota(candidates.begin(), candidates.end(), 0u);
    std::unordered_map<std::string, std::size_t> requestedRanks;
    for (std::size_t i = 0; i < profile.plugins.size(); ++i)
        requestedRanks.emplace(lower(profile.plugins[i]), i);
    auto rank = [&](std::size_t index) {
        const auto found = requestedRanks.find(lower(order.entries()[index].header.fileName));
        return found == requestedRanks.end() ? profile.plugins.size() + index : found->second;
    };
    std::stable_sort(candidates.begin(), candidates.end(), [&](auto a, auto b) { return rank(a) < rank(b); });
    std::unordered_set<std::string> placed;
    while (!candidates.empty()) {
        const auto ready = std::find_if(candidates.begin(), candidates.end(), [&](auto i) {
            const auto& masters = order.entries()[i].header.masters;
            return std::all_of(masters.begin(), masters.end(), [&](const auto& m) { return placed.count(lower(m)) != 0; });
        });
        if (ready == candidates.end()) { // open() has already rejected cycles.
            add(ContentDiagnosticSeverity::Error, "dependency-cycle", "cannot resolve dependency order");
            return report;
        }
        const auto& name = order.entries()[*ready].header.fileName;
        report.resolvedOrder.push_back(name);
        placed.insert(lower(name));
        candidates.erase(ready);
    }
    for (std::size_t i = 0; i < report.resolvedOrder.size(); ++i) {
        const auto& name = report.resolvedOrder[i];
        auto found = std::find_if(profile.plugins.begin(), profile.plugins.end(), [&](const auto& p) { return lower(p) == lower(name); });
        if (found == profile.plugins.end()) report.changes.push_back("insert " + name + " at " + std::to_string(i + 1));
        else if (static_cast<std::size_t>(found - profile.plugins.begin()) != i)
            report.changes.push_back("move " + name + " to " + std::to_string(i + 1));
    }
    ResolvedContentProfile sorted = profile;
    sorted.plugins = report.resolvedOrder;
    if (!order.open(sorted, error)) {
        add(ContentDiagnosticSeverity::Error, orderErrorCode(error), error, order.lastErrorSource());
        return report;
    }
    if (!scanRecords || report.hasErrors()) return report;
    for (const auto& entry : order.entries()) {
        EsmReader reader;
        if (!reader.open(entry.path)) {
            add(ContentDiagnosticSeverity::Error, "plugin-read", reader.lastError(), entry.path);
            continue;
        }
        std::unordered_set<std::uint32_t> ids;
        std::optional<std::uint32_t> lastId;
        std::uint64_t lastOffset = 0;
        EsmReader::Visitor visitor;
        visitor.onRecordHeader = [&](const EsmRecordHeaderView& h) {
            lastId = h.formId;
            lastOffset = h.fileOffset;
            if (h.type == "TES4") return true;
            if (!ids.insert(h.formId).second)
                add(ContentDiagnosticSeverity::Error, "duplicate-record", "duplicate record identity within plugin", entry.path, h.formId, h.fileOffset);
            const auto local = h.formId >> 24u;
            if (local >= entry.localToGlobal.size())
                add(ContentDiagnosticSeverity::Error, "invalid-master-index", "record addresses an undeclared master index", entry.path, h.formId, h.fileOffset);
            else if (entry.localToGlobal[local].kind == FalloutPluginSlotKind::Light && (h.formId & 0x00FFFFFFu) > 0xFFFu)
                add(ContentDiagnosticSeverity::Error, "invalid-light-id", "record object ID exceeds the light namespace", entry.path, h.formId, h.fileOffset);
            if ((h.flags & 0x20u) != 0 && (h.type == "REFR" || h.type == "ACHR" || h.type == "NAVM"))
                add(ContentDiagnosticSeverity::Warning, h.type == "NAVM" ? "deleted-navmesh" : "deleted-reference",
                    "deleted record; review in xEdit before changing it", entry.path, h.formId, h.fileOffset);
            return true;
        };
        if (!reader.walk(visitor)) {
            const auto offset = reader.lastErrorOffset();
            add(ContentDiagnosticSeverity::Error, "record-integrity", reader.lastError(), entry.path,
                offset == lastOffset ? lastId : std::nullopt, offset);
        }
        if (reader.toleratedChecksumFailures() != 0)
            add(ContentDiagnosticSeverity::Warning, "compression-checksum", "reader tolerated " +
                std::to_string(reader.toleratedChecksumFailures()) + " compressed checksum failures", entry.path);
    }
    if (!report.hasErrors()) {
        ContentRecordIndex index;
        if (index.build(order, error)) {
            report.overrideCount = index.overrideCount();
            add(ContentDiagnosticSeverity::Info, "record-overrides", std::to_string(report.overrideCount) + " overrides (normal load-order behavior)");
        } else add(ContentDiagnosticSeverity::Error, "record-index", error);
    }
    return report;
}

int runModCheckCommand(int argc, const char* const* argv, std::ostream& out, std::ostream& err) {
    namespace fs = std::filesystem;
    fs::path source, exportPath;
    ContentProfileResolveOptions options;
    bool json = false;
    for (int i = 0; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--json") { json = true; continue; }
        if (arg == "--help") {
            out << "odai mod-check --profile <path> [--data <Data>] [--mods-root <dir>] [--json] [--export-profile <new.json>]\n";
            return 0;
        }
        if ((arg != "--profile" && arg != "--data" && arg != "--mods-root" && arg != "--export-profile") || i + 1 >= argc) {
            err << "invalid or incomplete mod-check option: " << arg << '\n'; return 2;
        }
        const fs::path value = argv[++i];
        if (arg == "--profile") source = value;
        else if (arg == "--data") options.dataRootOverride = value;
        else if (arg == "--mods-root") options.modsRoot = value;
        else exportPath = value;
    }
    if (source.empty()) { err << "mod-check requires --profile <path>\n"; return 2; }
    try {
        ResolvedContentProfile profile;
        ModCheckReport report;
        std::string error;
        if (!resolveContentProfile(source, options, profile, error)) {
            for (const auto& d : profile.diagnostics) report.diagnostics.push_back({d.severity, d.code, d.message, d.source, {}, {}});
            report.diagnostics.push_back({ContentDiagnosticSeverity::Error, "profile-read", error, source, {}, {}});
        } else report = checkSkyrimMods(profile);
        int status = report.hasErrors() ? 1 : 0;
        if (!exportPath.empty() && status == 0) {
            // Stage using the existing writer inside a private directory. A
            // hard-link publishes without replacing even a racing destination.
            const auto stage = fs::path(exportPath.string() + ".mod-check-stage");
            std::error_code ec;
            if (fs::symlink_status(exportPath, ec).type() != fs::file_type::not_found && !ec) {
                error = "export destination already exists";
            } else if (!fs::create_directory(stage, ec)) {
                error = "cannot create export staging directory: " + stage.string();
            } else {
                struct Cleanup { fs::path path; ~Cleanup() { std::error_code ignored; fs::remove_all(path, ignored); } } cleanup{stage};
                auto exported = profile;
                exported.plugins = report.resolvedOrder;
                const auto staged = stage / "profile.json";
                if (writeOdaiContentProfile(staged, exported, error)) {
                    ResolvedContentProfile verified;
                    if (resolveContentProfile(staged, {}, verified, error) && !checkSkyrimMods(verified, false).hasErrors()) {
                        fs::create_hard_link(staged, exportPath, ec);
                        if (ec) error = "cannot publish profile: " + ec.message();
                    } else if (error.empty()) error = "exported profile failed validation";
                }
            }
            if (!error.empty()) {
                report.diagnostics.push_back({ContentDiagnosticSeverity::Error, "export-failed", error, exportPath, {}, {}});
                status = 2;
            }
        }
        if (json) {
            nlohmann::json diagnostics = nlohmann::json::array();
            for (const auto& d : report.diagnostics) {
                nlohmann::json item = {{"severity", severityName(d.severity)}, {"code", d.code}, {"message", d.message}, {"source", d.source.string()}};
                if (d.formId) item["form_id"] = *d.formId;
                if (d.fileOffset) item["file_offset"] = *d.fileOffset;
                diagnostics.push_back(std::move(item));
            }
            out << nlohmann::json({{"version", 1}, {"ok", status == 0}, {"requested_order", report.requestedOrder},
                {"resolved_order", report.resolvedOrder}, {"changes", report.changes}, {"override_count", report.overrideCount},
                {"diagnostics", diagnostics}}).dump(2) << '\n';
        } else {
            out << "Mod check: " << (status == 0 ? "passed" : "failed") << '\n';
            for (const auto& d : report.diagnostics) {
                out << severityName(d.severity) << " [" << d.code << "] " << d.source.string();
                if (d.formId) out << " FormID=0x" << std::hex << *d.formId << std::dec;
                if (d.fileOffset) out << " offset=" << *d.fileOffset;
                out << ": " << d.message << '\n';
            }
            for (const auto& change : report.changes) out << change << '\n';
            out << "Resolved order:\n";
            for (const auto& plugin : report.resolvedOrder) out << "  " << plugin << '\n';
        }
        if (!out) return 2;
        return status;
    } catch (const std::exception& e) {
        err << "mod-check failed: " << e.what() << '\n';
        return 2;
    }
}
} // namespace odai::importer::fnv
