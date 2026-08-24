#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace odai::importer::fnv {

// A resolved content profile is data, not executable engine extension code. It
// is the single immutable description of the virtual Data directory consumed
// by the runtime, importer and tools.
enum class BethesdaGame : std::uint8_t {
    Unknown,
    Morrowind,
    Oblivion,
    Fallout3,
    FalloutNewVegas,
    SkyrimSpecialEdition,
};

enum class ContentDiagnosticSeverity : std::uint8_t {
    Info,
    Warning,
    Error,
};

struct ContentDiagnostic {
    ContentDiagnosticSeverity severity = ContentDiagnosticSeverity::Info;
    std::string code;
    std::string message;
    std::filesystem::path source;
};

struct ContentLayer {
    std::string id;
    std::string name;
    std::filesystem::path root;
    bool enabled = true;
    std::uint32_t priority = 0;  // ascending; later layers win
    std::string version;
    std::string source;
};

struct ContentArchive {
    std::filesystem::path path;
    std::string layerId;
    bool required = false;
    std::uint32_t priority = 0;  // ascending; later archives win
};

struct ResolvedContentProfile {
    static constexpr std::uint32_t kVersion = 1u;

    std::uint32_t version = kVersion;
    std::string name;
    BethesdaGame game = BethesdaGame::Unknown;
    std::filesystem::path sourcePath;
    std::filesystem::path dataRoot;
    std::vector<ContentLayer> layers;
    std::vector<std::string> plugins;
    // TES3 strings are decoded according to the profile. OpenMW defaults
    // Western content to Windows-1252 when no encoding is specified.
    std::string encoding = "windows-1252";
    // Ordered OpenMW Lua manifests from `content=` entries. They are data in
    // the immutable content graph even when the Lua runtime is not enabled.
    std::vector<std::string> openMwScripts;
    std::vector<ContentArchive> archives;
    std::vector<ContentDiagnostic> diagnostics;
    std::string fingerprint;
    // Runtime policy, deliberately excluded from the content fingerprint.
    bool forceContentReindex = false;

    [[nodiscard]] bool hasErrors() const;
};

struct ContentProfileResolveOptions {
    std::optional<std::filesystem::path> dataRootOverride;
    std::optional<std::filesystem::path> modsRoot;
    // Explicit --mod/ODAI_FNV_MODS roots appended above the selected profile.
    std::vector<std::filesystem::path> extraLayers;
    bool forceContentReindex = false;
};

// Auto-detects an ODAI JSON file, an OpenMW openmw.cfg, or an MO2 profile
// directory. Inputs are never modified.
bool resolveContentProfile(
    const std::filesystem::path& source,
    const ContentProfileResolveOptions& options,
    ResolvedContentProfile& outProfile,
    std::string& outError);

bool writeOdaiContentProfile(
    const std::filesystem::path& path,
    const ResolvedContentProfile& profile,
    std::string& outError);

bool writeContentCompatibilityReport(
    const std::filesystem::path& path,
    const ResolvedContentProfile& profile,
    std::string& outError);

// Conventional profiles plus previously explicit roots supplied through
// ODAI_PROFILE_ROOTS (':' separated). Returned paths are unique and sorted.
std::vector<std::filesystem::path> discoverContentProfiles();

[[nodiscard]] const char* bethesdaGameName(BethesdaGame game);
[[nodiscard]] bool parseBethesdaGame(const std::string& text, BethesdaGame& outGame);

}  // namespace odai::importer::fnv
