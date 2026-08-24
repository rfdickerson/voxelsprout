#include "import/fnv/content_profile.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <set>
#include <sstream>
#include <system_error>
#include <unordered_set>

#include <nlohmann/json.hpp>

namespace odai::importer::fnv {
namespace {

namespace fs = std::filesystem;

std::string lower(std::string value) {
    for (char& c : value) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return value;
}

std::string trim(std::string value) {
    const auto space = [](unsigned char c) { return std::isspace(c) != 0; };
    while (!value.empty() && space(static_cast<unsigned char>(value.front()))) value.erase(value.begin());
    while (!value.empty() && space(static_cast<unsigned char>(value.back()))) value.pop_back();
    if (value.size() >= 2u && ((value.front() == '"' && value.back() == '"') ||
                              (value.front() == '\'' && value.back() == '\''))) {
        value = value.substr(1u, value.size() - 2u);
    }
    return value;
}

std::string safeId(std::string value) {
    value = lower(std::move(value));
    for (char& c : value) {
        if (!std::isalnum(static_cast<unsigned char>(c)) && c != '-' && c != '_') c = '_';
    }
    return value.empty() ? "content" : value;
}

fs::path absoluteFrom(const fs::path& base, fs::path value) {
    if (value.is_relative()) value = base / value;
    std::error_code ec;
    fs::path canonical = fs::weakly_canonical(value, ec);
    return ec ? value.lexically_normal() : canonical;
}

void addDiagnostic(
    ResolvedContentProfile& profile, ContentDiagnosticSeverity severity,
    std::string code, std::string message, fs::path source = {}) {
    profile.diagnostics.push_back(ContentDiagnostic{
        severity, std::move(code), std::move(message), std::move(source)});
}

std::uint64_t hashBytes(std::uint64_t hash, const std::string& text) {
    constexpr std::uint64_t prime = 1099511628211ull;
    for (unsigned char c : text) {
        hash ^= static_cast<std::uint64_t>(c);
        hash *= prime;
    }
    hash ^= 0xffu;
    hash *= prime;
    return hash;
}

std::string metadata(const fs::path& path) {
    std::error_code ec;
    const auto size = fs::is_regular_file(path, ec) ? fs::file_size(path, ec) : 0u;
    const auto stamp = fs::last_write_time(path, ec);
    const auto ticks = ec ? 0ll : stamp.time_since_epoch().count();
    return path.generic_string() + ":" + std::to_string(size) + ":" + std::to_string(ticks);
}

void finishProfile(ResolvedContentProfile& profile) {
    const char* basePlugin = nullptr;
    switch (profile.game) {
        case BethesdaGame::Morrowind: basePlugin = "Morrowind.esm"; break;
        case BethesdaGame::Oblivion: basePlugin = "Oblivion.esm"; break;
        case BethesdaGame::Fallout3: basePlugin = "Fallout3.esm"; break;
        case BethesdaGame::FalloutNewVegas: basePlugin = "FalloutNV.esm"; break;
        case BethesdaGame::SkyrimSpecialEdition: basePlugin = "Skyrim.esm"; break;
        case BethesdaGame::Unknown: break;
    }
    if (basePlugin != nullptr) {
        const auto found = std::find_if(profile.plugins.begin(), profile.plugins.end(),
            [&](const std::string& value) { return lower(value) == lower(basePlugin); });
        if (found == profile.plugins.end() && fs::is_regular_file(profile.dataRoot / basePlugin)) {
            profile.plugins.insert(profile.plugins.begin(), basePlugin);
        }
    }
    for (std::size_t i = 0; i < profile.layers.size(); ++i) {
        profile.layers[i].priority = static_cast<std::uint32_t>(i);
    }
    for (std::size_t i = 0; i < profile.archives.size(); ++i) {
        profile.archives[i].priority = static_cast<std::uint32_t>(i);
    }
    std::uint64_t a = 1469598103934665603ull;
    std::uint64_t b = 1099511628211ull;
    auto mix = [&](const std::string& value) {
        a = hashBytes(a, value);
        b = hashBytes(b, std::string(value.rbegin(), value.rend()));
    };
    mix(std::to_string(profile.version));
    mix(bethesdaGameName(profile.game));
    mix(lower(profile.encoding));
    mix(metadata(profile.sourcePath));
    mix(metadata(profile.dataRoot));
    for (const ContentLayer& layer : profile.layers) {
        mix(layer.id); mix(layer.name); mix(metadata(layer.root)); mix(layer.version); mix(layer.source);
    }
    for (const std::string& plugin : profile.plugins) {
        mix(lower(plugin));
        fs::path candidate = profile.dataRoot / plugin;
        for (auto it = profile.layers.rbegin(); it != profile.layers.rend(); ++it) {
            std::error_code ec;
            const fs::path inLayer = it->root / plugin;
            if (fs::is_regular_file(inLayer, ec) && !ec) { candidate = inLayer; break; }
        }
        mix(metadata(candidate));
    }
    for (const std::string& scripts : profile.openMwScripts) {
        mix(lower(scripts));
        fs::path candidate = profile.dataRoot / scripts;
        for (auto it = profile.layers.rbegin(); it != profile.layers.rend(); ++it) {
            std::error_code ec;
            const fs::path inLayer = it->root / scripts;
            if (fs::is_regular_file(inLayer, ec) && !ec) { candidate = inLayer; break; }
        }
        mix(metadata(candidate));
    }
    for (const ContentArchive& archive : profile.archives) mix(metadata(archive.path));
    std::ostringstream text;
    text << std::hex << std::setfill('0') << std::setw(16) << a << std::setw(16) << b;
    profile.fingerprint = text.str();
}

bool parseJsonProfile(
    const fs::path& path, const ContentProfileResolveOptions& options,
    ResolvedContentProfile& profile, std::string& error) {
    std::ifstream input(path);
    if (!input) { error = "cannot open profile " + path.string(); return false; }
    try {
        nlohmann::json json; input >> json;
        if (json.value("version", 0u) != ResolvedContentProfile::kVersion) {
            error = "unsupported ODAI profile version"; return false;
        }
        profile.version = ResolvedContentProfile::kVersion;
        profile.name = json.value("name", path.stem().string());
        if (!parseBethesdaGame(json.value("game", std::string{}), profile.game)) {
            error = "unknown or missing profile game"; return false;
        }
        profile.encoding = json.value("encoding", std::string("windows-1252"));
        for (const auto& item : json.value("openmw_scripts", nlohmann::json::array())) {
            profile.openMwScripts.push_back(item.get<std::string>());
        }
        const fs::path base = path.parent_path();
        if (options.dataRootOverride) profile.dataRoot = absoluteFrom(base, *options.dataRootOverride);
        else profile.dataRoot = absoluteFrom(base, json.value("data_root", std::string{}));
        if (profile.dataRoot.empty()) { error = "profile has no data_root"; return false; }
        std::size_t ordinal = 0u;
        for (const auto& item : json.value("layers", nlohmann::json::array())) {
            if (!item.value("enabled", true)) continue;
            ContentLayer layer;
            layer.name = item.value("name", item.value("id", std::string("layer-" + std::to_string(ordinal))));
            layer.id = safeId(item.value("id", layer.name));
            layer.root = absoluteFrom(base, item.at("path").get<std::string>());
            layer.version = item.value("version", std::string{});
            layer.source = item.value("source", std::string{});
            profile.layers.push_back(std::move(layer)); ++ordinal;
        }
        for (const auto& item : json.value("plugins", nlohmann::json::array())) {
            if (item.is_string()) profile.plugins.push_back(item.get<std::string>());
            else if (item.value("enabled", true)) profile.plugins.push_back(item.at("name").get<std::string>());
        }
        for (const auto& item : json.value("archives", nlohmann::json::array())) {
            ContentArchive archive;
            if (item.is_string()) archive.path = absoluteFrom(base, item.get<std::string>());
            else {
                archive.path = absoluteFrom(base, item.at("path").get<std::string>());
                archive.layerId = item.value("layer", std::string{});
                archive.required = item.value("required", false);
            }
            profile.archives.push_back(std::move(archive));
        }
    } catch (const std::exception& exception) {
        error = "invalid ODAI profile: " + std::string(exception.what()); return false;
    }
    return true;
}

std::vector<std::string> readList(const fs::path& path, bool activeOnly) {
    std::ifstream input(path);
    std::vector<std::string> result;
    std::string line;
    bool sawStar = false;
    std::vector<std::pair<std::string, bool>> parsed;
    while (std::getline(input, line)) {
        line = trim(std::move(line));
        if (line.empty() || line[0] == '#' || line[0] == ';') continue;
        const bool starred = line[0] == '*';
        if (starred) { line = trim(line.substr(1)); sawStar = true; }
        parsed.emplace_back(std::move(line), starred);
    }
    for (const auto& [name, starred] : parsed) {
        if (!activeOnly || !sawStar || starred) result.push_back(name);
    }
    return result;
}

fs::path findCaseInsensitive(const std::vector<fs::path>& roots, const std::string& name) {
    const std::string wanted = lower(name);
    for (auto root = roots.rbegin(); root != roots.rend(); ++root) {
        std::error_code ec;
        for (fs::directory_iterator it(*root, ec), end; !ec && it != end; it.increment(ec)) {
            if (lower(it->path().filename().string()) == wanted) return it->path();
        }
    }
    return {};
}

bool archiveMatchesActivePlugin(
    const fs::path& archivePath, const std::vector<std::string>& plugins) {
    const std::string archiveStem = lower(archivePath.stem().string());
    if (archiveStem.rfind("skyrim -", 0u) == 0u ||
        archiveStem.rfind("oblivion -", 0u) == 0u ||
        archiveStem.rfind("fallout -", 0u) == 0u ||
        archiveStem == "morrowind") {
        return true;
    }
    for (const std::string& plugin : plugins) {
        const std::string pluginStem = lower(fs::path(plugin).stem().string());
        if (archiveStem == pluginStem || archiveStem.rfind(pluginStem + " -", 0u) == 0u) {
            return true;
        }
        // Fallout 3/New Vegas call their base archives "Fallout - ..." while
        // the masters are Fallout3.esm/FalloutNV.esm.
        if ((pluginStem == "fallout3" || pluginStem == "falloutnv") &&
            archiveStem.rfind("fallout -", 0u) == 0u) {
            return true;
        }
    }
    return false;
}

void inferMo2Archives(ResolvedContentProfile& profile) {
    struct Candidate {
        fs::path path;
        std::string layerId;
    };
    std::vector<Candidate> candidates;
    const auto scanRoot = [&](const fs::path& root, const std::string& layerId) {
        std::error_code ec;
        std::vector<fs::path> rootArchives;
        for (fs::directory_iterator it(root, ec), end; !ec && it != end; it.increment(ec)) {
            std::error_code typeError;
            if (it->is_regular_file(typeError) && !typeError &&
                lower(it->path().extension().string()) == ".bsa") {
                rootArchives.push_back(it->path());
            }
        }
        std::sort(rootArchives.begin(), rootArchives.end(), [](const fs::path& a, const fs::path& b) {
            return lower(a.filename().string()) < lower(b.filename().string());
        });
        for (const fs::path& path : rootArchives) {
            const std::string name = lower(path.filename().string());
            candidates.erase(std::remove_if(candidates.begin(), candidates.end(),
                [&](const Candidate& existing) {
                    return lower(existing.path.filename().string()) == name;
                }), candidates.end());
            candidates.push_back(Candidate{path, layerId});
        }
    };
    scanRoot(profile.dataRoot, "base-data");
    for (const ContentLayer& layer : profile.layers) scanRoot(layer.root, layer.id);

    for (const Candidate& candidate : candidates) {
        if (archiveMatchesActivePlugin(candidate.path, profile.plugins)) {
            profile.archives.push_back(ContentArchive{
                candidate.path, candidate.layerId, false, 0u});
        } else {
            addDiagnostic(profile, ContentDiagnosticSeverity::Warning, "inactive-unassociated-archive",
                "archive is not associated with an active plugin and was not activated",
                candidate.path);
        }
    }
}

bool parseMo2Profile(
    const fs::path& directory, const ContentProfileResolveOptions& options,
    ResolvedContentProfile& profile, std::string& error) {
    const fs::path modlist = directory / "modlist.txt";
    if (!fs::is_regular_file(modlist)) { error = "MO2 profile has no modlist.txt"; return false; }
    const fs::path instance = lower(directory.parent_path().filename().string()) == "profiles"
        ? directory.parent_path().parent_path() : directory.parent_path();
    const fs::path modsRoot = options.modsRoot.value_or(instance / "mods");
    if (!fs::is_directory(modsRoot)) {
        error = "cannot resolve MO2 mods directory; pass --mods-root"; return false;
    }
    if (!options.dataRootOverride) {
        error = "MO2 profiles require the game Data directory via --stream"; return false;
    }
    profile.dataRoot = absoluteFrom(directory, *options.dataRootOverride);
    profile.name = directory.filename().string();

    struct ModLine { char state = '+'; std::string name; };
    std::ifstream input(modlist);
    std::vector<ModLine> lines;
    std::string line;
    while (std::getline(input, line)) {
        line = trim(std::move(line));
        if (line.empty() || line[0] == '#') continue;
        ModLine item;
        if (line[0] == '+' || line[0] == '-' || line[0] == '*') {
            item.state = line[0]; line = trim(line.substr(1));
        }
        item.name = line;
        if (!item.name.empty()) lines.push_back(std::move(item));
    }
    // MO2 writes highest priority first; the engine consumes ascending order.
    std::reverse(lines.begin(), lines.end());
    std::size_t ordinal = 0u;
    for (const ModLine& item : lines) {
        if (item.state == '-') continue;
        const fs::path root = modsRoot / item.name;
        if (!fs::is_directory(root)) {
            addDiagnostic(profile, ContentDiagnosticSeverity::Warning, "missing-mod-root",
                          "enabled MO2 mod is missing: " + item.name, root);
            continue;
        }
        profile.layers.push_back(ContentLayer{
            safeId(item.name), item.name, absoluteFrom(modsRoot, root), true,
            static_cast<std::uint32_t>(ordinal++), {}, "mo2"});
    }
    const fs::path overwrite = instance / "overwrite";
    if (fs::is_directory(overwrite)) {
        profile.layers.push_back(ContentLayer{
            "overwrite", "MO2 overwrite", absoluteFrom(instance, overwrite), true,
            static_cast<std::uint32_t>(ordinal++), {}, "mo2"});
    }

    const std::vector<std::string> active = readList(directory / "plugins.txt", true);
    const std::vector<std::string> order = readList(directory / "loadorder.txt", false);
    std::set<std::string> activeSet;
    for (const std::string& name : active) activeSet.insert(lower(name));
    for (const std::string& name : order) {
        if (activeSet.empty() || activeSet.contains(lower(name))) profile.plugins.push_back(name);
    }
    for (const std::string& name : active) {
        const auto found = std::find_if(profile.plugins.begin(), profile.plugins.end(),
            [&](const std::string& existing) { return lower(existing) == lower(name); });
        if (found == profile.plugins.end()) profile.plugins.push_back(name);
    }
    if (profile.plugins.empty()) {
        error = "MO2 profile has no active plugins"; return false;
    }

    // Total conversions deliberately do not have a game name in their first
    // active plugin (Skywind.esp is the motivating case). Identify the target
    // runtime from the installed base master rather than guessing from a mod.
    if (fs::is_regular_file(profile.dataRoot / "Skyrim.esm")) profile.game = BethesdaGame::SkyrimSpecialEdition;
    else if (fs::is_regular_file(profile.dataRoot / "Oblivion.esm")) profile.game = BethesdaGame::Oblivion;
    else if (fs::is_regular_file(profile.dataRoot / "FalloutNV.esm")) profile.game = BethesdaGame::FalloutNewVegas;
    else if (fs::is_regular_file(profile.dataRoot / "Fallout3.esm")) profile.game = BethesdaGame::Fallout3;
    else if (fs::is_regular_file(profile.dataRoot / "Morrowind.esm")) profile.game = BethesdaGame::Morrowind;
    else addDiagnostic(profile, ContentDiagnosticSeverity::Error, "unknown-target-game",
        "MO2 Data directory contains no supported Bethesda base master", profile.dataRoot);

    std::vector<fs::path> roots{profile.dataRoot};
    for (const ContentLayer& layer : profile.layers) roots.push_back(layer.root);
    const fs::path archiveList = directory / "archives.txt";
    if (fs::is_regular_file(archiveList)) {
        for (const std::string& archiveName : readList(archiveList, true)) {
            const fs::path resolved = findCaseInsensitive(roots, archiveName);
            if (resolved.empty()) {
                addDiagnostic(profile, ContentDiagnosticSeverity::Warning, "missing-archive",
                              "active MO2 archive is missing: " + archiveName, archiveList);
            } else {
                profile.archives.push_back(ContentArchive{resolved, {}, false, 0u});
            }
        }
    } else {
        inferMo2Archives(profile);
    }

    return true;
}

struct OpenMwValues {
    std::vector<fs::path> data;
    std::vector<std::string> archives;
    std::vector<std::string> content;
    std::optional<fs::path> dataLocal;
    std::string encoding = "windows-1252";
};

std::string replaceAll(std::string value, const std::string& token, const std::string& replacement) {
    std::size_t pos = 0u;
    while ((pos = value.find(token, pos)) != std::string::npos) {
        value.replace(pos, token.size(), replacement); pos += replacement.size();
    }
    return value;
}

fs::path expandOpenMwPath(std::string value, const fs::path& configDir) {
    value = trim(std::move(value));
    value = replaceAll(value, "?userconfig?", configDir.string());
    fs::path userData;
    if (const char* xdg = std::getenv("XDG_DATA_HOME")) userData = fs::path(xdg) / "openmw";
    else if (const char* home = std::getenv("HOME")) userData = fs::path(home) / ".local/share/openmw";
    value = replaceAll(value, "?userdata?", userData.string());
    std::size_t begin = 0u;
    while ((begin = value.find("${", begin)) != std::string::npos) {
        const std::size_t end = value.find('}', begin + 2u);
        if (end == std::string::npos) break;
        const std::string name = value.substr(begin + 2u, end - begin - 2u);
        const char* env = std::getenv(name.c_str());
        value.replace(begin, end - begin + 1u, env == nullptr ? "" : env);
    }
    return absoluteFrom(configDir, fs::path(value));
}

bool parseOpenMwConfigRecursive(
    const fs::path& path, OpenMwValues& values, std::unordered_set<std::string>& active,
    ResolvedContentProfile& profile, std::string& error) {
    const fs::path configPath = fs::is_directory(path) ? path / "openmw.cfg" : path;
    const std::string key = absoluteFrom(configPath.parent_path(), configPath).generic_string();
    if (!active.insert(key).second) { error = "OpenMW config include cycle at " + key; return false; }
    std::ifstream input(configPath);
    if (!input) { error = "cannot open OpenMW config " + configPath.string(); active.erase(key); return false; }
    std::string line;
    while (std::getline(input, line)) {
        line = trim(std::move(line));
        if (line.empty() || line[0] == '#') continue;
        const std::size_t equals = line.find('=');
        if (equals == std::string::npos) continue;
        const std::string option = lower(trim(line.substr(0u, equals)));
        const std::string raw = trim(line.substr(equals + 1u));
        if (option == "replace") {
            const std::string replaced = lower(raw);
            if (replaced == "data") values.data.clear();
            else if (replaced == "content") values.content.clear();
            else if (replaced == "fallback-archive") values.archives.clear();
            // `replace=config` changes OpenMW's inherited config list, but it
            // must not erase the recursion stack used to detect include loops.
        } else if (option == "config") {
            if (!parseOpenMwConfigRecursive(expandOpenMwPath(raw, configPath.parent_path()),
                                            values, active, profile, error)) return false;
        } else if (option == "data") {
            values.data.push_back(expandOpenMwPath(raw, configPath.parent_path()));
        } else if (option == "data-local") {
            values.dataLocal = expandOpenMwPath(raw, configPath.parent_path());
        } else if (option == "fallback-archive") {
            values.archives.push_back(raw);
        } else if (option == "content") {
            values.content.push_back(raw);
        } else if (option == "encoding") {
            values.encoding = raw.empty() ? "windows-1252" : lower(raw);
        }
    }
    active.erase(key);
    return true;
}

bool parseOpenMwProfile(
    const fs::path& path, const ContentProfileResolveOptions& options,
    ResolvedContentProfile& profile, std::string& error) {
    OpenMwValues values;
    std::unordered_set<std::string> active;
    if (!parseOpenMwConfigRecursive(path, values, active, profile, error)) return false;
    if (values.dataLocal) values.data.push_back(*values.dataLocal);
    profile.name = path.parent_path().filename().string().empty() ? "OpenMW" : path.parent_path().filename().string();
    profile.game = BethesdaGame::Morrowind;
    profile.encoding = values.encoding;

    std::size_t baseIndex = values.data.size();
    if (options.dataRootOverride) profile.dataRoot = absoluteFrom(path.parent_path(), *options.dataRootOverride);
    else {
        for (std::size_t i = 0; i < values.data.size(); ++i) {
            if (fs::is_regular_file(values.data[i] / "Morrowind.esm")) { baseIndex = i; break; }
        }
        if (baseIndex != values.data.size()) profile.dataRoot = values.data[baseIndex];
    }
    if (profile.dataRoot.empty()) { error = "OpenMW config has no data directory containing Morrowind.esm"; return false; }
    std::size_t ordinal = 0u;
    for (std::size_t i = 0; i < values.data.size(); ++i) {
        if (i == baseIndex || values.data[i] == profile.dataRoot) continue;
        if (!fs::is_directory(values.data[i])) {
            addDiagnostic(profile, ContentDiagnosticSeverity::Warning, "missing-data-root",
                          "OpenMW data directory is missing", values.data[i]);
            continue;
        }
        const std::string name = values.data[i].filename().string();
        profile.layers.push_back(ContentLayer{
            safeId(name + "-" + std::to_string(ordinal)), name, values.data[i], true,
            static_cast<std::uint32_t>(ordinal++), {}, "openmw"});
    }
    for (const std::string& content : values.content) {
        const std::string ext = lower(fs::path(content).extension().string());
        if (ext == ".esm" || ext == ".esp" || ext == ".omwaddon") profile.plugins.push_back(content);
        else if (ext == ".omwscripts") {
            profile.openMwScripts.push_back(content);
        }
    }
    std::vector<fs::path> roots{profile.dataRoot};
    for (const ContentLayer& layer : profile.layers) roots.push_back(layer.root);
    for (const std::string& archive : values.archives) {
        const fs::path resolved = findCaseInsensitive(roots, archive);
        if (resolved.empty()) addDiagnostic(profile, ContentDiagnosticSeverity::Error, "missing-archive",
            "required OpenMW fallback archive is missing: " + archive, path);
        else profile.archives.push_back(ContentArchive{resolved, {}, true, 0u});
    }
    if (profile.plugins.empty()) { error = "OpenMW config has no supported content plugins"; return false; }
    return true;
}

}  // namespace

bool ResolvedContentProfile::hasErrors() const {
    return std::any_of(diagnostics.begin(), diagnostics.end(), [](const ContentDiagnostic& item) {
        return item.severity == ContentDiagnosticSeverity::Error;
    });
}

const char* bethesdaGameName(BethesdaGame game) {
    switch (game) {
        case BethesdaGame::Morrowind: return "morrowind";
        case BethesdaGame::Oblivion: return "oblivion";
        case BethesdaGame::Fallout3: return "fallout3";
        case BethesdaGame::FalloutNewVegas: return "newvegas";
        case BethesdaGame::SkyrimSpecialEdition: return "skyrim-se";
        case BethesdaGame::Unknown: break;
    }
    return "unknown";
}

bool parseBethesdaGame(const std::string& text, BethesdaGame& outGame) {
    const std::string value = lower(text);
    if (value == "morrowind") outGame = BethesdaGame::Morrowind;
    else if (value == "oblivion") outGame = BethesdaGame::Oblivion;
    else if (value == "fallout3" || value == "fallout-3") outGame = BethesdaGame::Fallout3;
    else if (value == "newvegas" || value == "fallout-new-vegas") outGame = BethesdaGame::FalloutNewVegas;
    else if (value == "skyrim" || value == "skyrim-se" || value == "skyrim-special-edition") outGame = BethesdaGame::SkyrimSpecialEdition;
    else { outGame = BethesdaGame::Unknown; return false; }
    return true;
}

bool resolveContentProfile(
    const fs::path& source, const ContentProfileResolveOptions& options,
    ResolvedContentProfile& outProfile, std::string& outError) {
    outProfile = ResolvedContentProfile{};
    outError.clear();
    outProfile.sourcePath = absoluteFrom(fs::current_path(), source);
    outProfile.forceContentReindex = options.forceContentReindex;
    bool ok = false;
    if (fs::is_directory(source)) {
        ok = parseMo2Profile(source, options, outProfile, outError);
    } else if (lower(source.filename().string()) == "openmw.cfg") {
        ok = parseOpenMwProfile(source, options, outProfile, outError);
    } else {
        ok = parseJsonProfile(source, options, outProfile, outError);
    }
    if (!ok) return false;
    for (const fs::path& root : options.extraLayers) {
        const std::string name = root.filename().string();
        outProfile.layers.push_back(ContentLayer{
            safeId("overlay-" + name + "-" + std::to_string(outProfile.layers.size())),
            name.empty() ? "command-line overlay" : name,
            absoluteFrom(fs::current_path(), root), true,
            static_cast<std::uint32_t>(outProfile.layers.size()), {}, "odai-cli"});
    }
    if (!fs::is_directory(outProfile.dataRoot)) {
        outError = "profile data root does not exist: " + outProfile.dataRoot.string(); return false;
    }
    for (const ContentLayer& layer : outProfile.layers) {
        if (!fs::is_directory(layer.root)) addDiagnostic(outProfile, ContentDiagnosticSeverity::Error,
            "missing-layer", "enabled content layer does not exist", layer.root);
    }
    finishProfile(outProfile);
    if (outProfile.hasErrors()) {
        outError = "profile contains launch-blocking content errors"; return false;
    }
    return true;
}

bool writeOdaiContentProfile(
    const fs::path& path, const ResolvedContentProfile& profile, std::string& outError) {
    outError.clear();
    nlohmann::json layers = nlohmann::json::array();
    for (const ContentLayer& layer : profile.layers) layers.push_back({
        {"id", layer.id}, {"name", layer.name}, {"path", layer.root.string()},
        {"enabled", layer.enabled}, {"version", layer.version}, {"source", layer.source}});
    nlohmann::json archives = nlohmann::json::array();
    for (const ContentArchive& archive : profile.archives) archives.push_back({
        {"path", archive.path.string()}, {"layer", archive.layerId}, {"required", archive.required}});
    const nlohmann::json json = {{"version", ResolvedContentProfile::kVersion},
        {"name", profile.name}, {"game", bethesdaGameName(profile.game)},
        {"encoding", profile.encoding}, {"openmw_scripts", profile.openMwScripts},
        {"data_root", profile.dataRoot.string()}, {"layers", std::move(layers)},
        {"plugins", profile.plugins}, {"archives", std::move(archives)}};
    std::error_code ec;
    if (!path.parent_path().empty()) fs::create_directories(path.parent_path(), ec);
    const fs::path temporary = path.string() + ".tmp";
    std::ofstream output(temporary, std::ios::trunc);
    if (!output) { outError = "cannot write " + temporary.string(); return false; }
    output << json.dump(2) << '\n'; output.close();
    if (!output) { outError = "failed while writing " + temporary.string(); return false; }
    fs::rename(temporary, path, ec);
    if (ec) { fs::remove(temporary); outError = "cannot replace " + path.string() + ": " + ec.message(); return false; }
    return true;
}

bool writeContentCompatibilityReport(
    const fs::path& path, const ResolvedContentProfile& profile, std::string& outError) {
    outError.clear();
    nlohmann::json diagnostics = nlohmann::json::array();
    for (const ContentDiagnostic& item : profile.diagnostics) {
        const char* severity = item.severity == ContentDiagnosticSeverity::Error ? "error" :
            item.severity == ContentDiagnosticSeverity::Warning ? "warning" : "info";
        diagnostics.push_back({{"severity", severity}, {"code", item.code},
            {"message", item.message}, {"source", item.source.string()}});
    }
    const nlohmann::json json = {{"version", 1u}, {"profile", profile.name},
        {"game", bethesdaGameName(profile.game)}, {"source", profile.sourcePath.string()},
        {"data_root", profile.dataRoot.string()}, {"fingerprint", profile.fingerprint},
        {"layer_count", profile.layers.size()}, {"plugin_count", profile.plugins.size()},
        {"encoding", profile.encoding}, {"openmw_script_count", profile.openMwScripts.size()},
        {"archive_count", profile.archives.size()}, {"diagnostics", std::move(diagnostics)}};
    std::error_code ec;
    if (!path.parent_path().empty()) fs::create_directories(path.parent_path(), ec);
    const fs::path temporary = path.string() + ".tmp";
    std::ofstream output(temporary, std::ios::trunc);
    if (!output) { outError = "cannot write compatibility report " + temporary.string(); return false; }
    output << json.dump(2) << '\n'; output.close();
    if (!output) { outError = "failed while writing compatibility report"; return false; }
    fs::rename(temporary, path, ec);
    if (ec) { fs::remove(temporary); outError = "cannot replace compatibility report: " + ec.message(); return false; }
    return true;
}

std::vector<fs::path> discoverContentProfiles() {
    std::vector<fs::path> candidates;
    const auto scanRoot = [&](const fs::path& root, int maxDepth) {
        if (fs::is_regular_file(root)) {
            candidates.push_back(root);
            return;
        }
        if (!fs::is_directory(root)) return;
        if (fs::is_regular_file(root / "modlist.txt")) candidates.push_back(root);
        std::error_code ec;
        fs::recursive_directory_iterator it(
            root, fs::directory_options::skip_permission_denied, ec), end;
        while (!ec && it != end) {
            if (it.depth() >= maxDepth) it.disable_recursion_pending();
            std::error_code typeError;
            if (it->is_directory(typeError) && !typeError &&
                fs::is_regular_file(it->path() / "modlist.txt")) {
                candidates.push_back(it->path());
                it.disable_recursion_pending();
            } else if (it->is_regular_file(typeError) && !typeError &&
                       lower(it->path().extension().string()) == ".json") {
                candidates.push_back(it->path());
            }
            it.increment(ec);
        }
    };
    if (const char* roots = std::getenv("ODAI_PROFILE_ROOTS")) {
        std::string value = roots; std::size_t start = 0u;
        while (start <= value.size()) {
            const std::size_t end = value.find(':', start);
            fs::path root = value.substr(start, end == std::string::npos ? std::string::npos : end - start);
            scanRoot(root, 3);
            if (end == std::string::npos) break;
            start = end + 1u;
        }
    }
    if (const char* xdg = std::getenv("XDG_CONFIG_HOME")) {
        scanRoot(fs::path(xdg) / "odai/profiles", 1);
    } else if (const char* home = std::getenv("HOME")) {
        scanRoot(fs::path(home) / ".config/odai/profiles", 1);
    }
    if (const char* home = std::getenv("HOME")) {
        scanRoot(fs::path(home) / ".local/share/ModOrganizer2", 3);
        scanRoot(fs::path(home) / ".config/ModOrganizer2", 3);
    }
    if (const char* local = std::getenv("LOCALAPPDATA")) {
        scanRoot(fs::path(local) / "ModOrganizer", 3);
        scanRoot(fs::path(local) / "ModOrganizer2", 3);
    }
    if (const char* appData = std::getenv("APPDATA")) {
        scanRoot(fs::path(appData) / "ModOrganizer", 3);
    }
    if (const char* xdg = std::getenv("XDG_CONFIG_HOME")) {
        const fs::path openmw = fs::path(xdg) / "openmw/openmw.cfg";
        if (fs::is_regular_file(openmw)) candidates.push_back(openmw);
    } else if (const char* home = std::getenv("HOME")) {
        const fs::path openmw = fs::path(home) / ".config/openmw/openmw.cfg";
        if (fs::is_regular_file(openmw)) candidates.push_back(openmw);
    }
    std::sort(candidates.begin(), candidates.end());
    candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
    return candidates;
}

}  // namespace odai::importer::fnv
