#include "import/fnv/asset_source.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <system_error>

#include <nlohmann/json.hpp>

namespace odai::importer::fnv {

namespace {

std::string toLowerAsciiCopy(std::string text) {
    for (char& c : text) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return text;
}

bool readWholeFile(
    const std::filesystem::path& path, std::vector<std::uint8_t>& outBytes, std::string& outError) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        outError = "failed to open " + path.string();
        return false;
    }
    const auto size = static_cast<std::size_t>(input.tellg());
    input.seekg(0);
    outBytes.resize(size);
    if (size == 0) {
        return true;
    }
    if (!input.read(reinterpret_cast<char*>(outBytes.data()), static_cast<std::streamsize>(size))) {
        outError = "failed to read " + path.string();
        return false;
    }
    return true;
}

struct CachedLayerIndex {
    struct File {
        std::string key;
        std::filesystem::path path;
        std::uintmax_t size = 0u;
        std::int64_t stamp = 0;
    };
    struct Directory {
        std::filesystem::path path;
        std::int64_t stamp = 0;
    };
    std::vector<File> looseFiles;
    std::vector<File> archives;
    std::vector<Directory> directories;
    std::uint64_t totalBytes = 0u;
};

std::int64_t pathStamp(const std::filesystem::path& path) {
    std::error_code error;
    const auto stamp = std::filesystem::last_write_time(path, error);
    return error ? 0 : static_cast<std::int64_t>(stamp.time_since_epoch().count());
}

std::uintmax_t pathSize(const std::filesystem::path& path) {
    std::error_code error;
    const auto size = std::filesystem::file_size(path, error);
    return error ? 0u : size;
}

std::string stablePathKey(const std::filesystem::path& path) {
    std::uint64_t hash = 1469598103934665603ull;
    for (unsigned char c : path.generic_string()) {
        hash ^= static_cast<std::uint64_t>(c);
        hash *= 1099511628211ull;
    }
    std::ostringstream text;
    text << std::hex << std::setfill('0') << std::setw(16) << hash;
    return text.str();
}

std::string indexedLayerFingerprint(
    const std::filesystem::path& root,
    const std::unordered_map<std::string, std::filesystem::path>& looseFiles,
    const std::vector<std::filesystem::path>& archives) {
    std::vector<std::string> entries;
    entries.reserve(looseFiles.size() + archives.size());
    const auto describe = [&](const std::string& key, const std::filesystem::path& path) {
        std::error_code relativeError;
        const auto relative = std::filesystem::relative(path, root, relativeError);
        entries.push_back(key + ":" +
            (relativeError ? path.generic_string() : relative.generic_string()) + ":" +
            std::to_string(pathSize(path)) + ":" + std::to_string(pathStamp(path)));
    };
    for (const auto& [key, path] : looseFiles) describe(key, path);
    for (const std::filesystem::path& path : archives) {
        describe("archive", path);
    }
    std::sort(entries.begin(), entries.end());
    std::uint64_t hash = 1469598103934665603ull;
    for (const std::string& entry : entries) {
        for (unsigned char c : entry) {
            hash ^= static_cast<std::uint64_t>(c);
            hash *= 1099511628211ull;
        }
        hash ^= 0xffu;
        hash *= 1099511628211ull;
    }
    std::ostringstream text;
    text << std::hex << std::setfill('0') << std::setw(16) << hash;
    return text.str();
}

std::string pathIdentity(const std::filesystem::path& path) {
    std::error_code error;
    const auto canonical = std::filesystem::weakly_canonical(path, error);
    return toLowerAsciiCopy((error ? path.lexically_normal() : canonical).generic_string());
}

std::filesystem::path defaultContentIndexDirectory() {
    if (const char* xdg = std::getenv("XDG_CACHE_HOME")) {
        if (*xdg != '\0') return std::filesystem::path(xdg) / "odai/content-index/v1";
    }
    if (const char* home = std::getenv("HOME")) {
        if (*home != '\0') return std::filesystem::path(home) / ".cache/odai/content-index/v1";
    }
    return {};
}

bool loadCachedLayerIndex(
    const std::filesystem::path& cacheDirectory, const std::filesystem::path& root,
    CachedLayerIndex& out) {
    if (cacheDirectory.empty()) return false;
    std::ifstream input(cacheDirectory / (stablePathKey(root) + ".json"));
    if (!input) return false;
    try {
        nlohmann::json json; input >> json;
        if (json.value("version", 0u) != 1u ||
            json.value("root", std::string{}) != root.generic_string() ||
            json.value("root_stamp", std::int64_t{}) != pathStamp(root)) return false;
        out.totalBytes = json.value("total_bytes", std::uint64_t{});
        for (const auto& item : json.at("directories")) {
            const std::filesystem::path path = root / item.at("relative").get<std::string>();
            const std::int64_t stamp = item.at("stamp").get<std::int64_t>();
            if (!std::filesystem::is_directory(path) || pathStamp(path) != stamp) return false;
            out.directories.push_back(CachedLayerIndex::Directory{path, stamp});
        }
        for (const auto& item : json.at("files")) {
            const std::string key = item.at("key").get<std::string>();
            const std::filesystem::path relative = item.at("relative").get<std::string>();
            const std::filesystem::path path = root / relative;
            const std::uintmax_t size = item.at("size").get<std::uintmax_t>();
            const std::int64_t stamp = item.at("stamp").get<std::int64_t>();
            if (!std::filesystem::is_regular_file(path) || pathSize(path) != size ||
                pathStamp(path) != stamp) return false;
            out.looseFiles.push_back(CachedLayerIndex::File{key, path, size, stamp});
        }
        for (const auto& item : json.at("archives")) {
            const std::filesystem::path path = root / item.at("relative").get<std::string>();
            const std::uintmax_t size = item.at("size").get<std::uintmax_t>();
            const std::int64_t stamp = item.at("stamp").get<std::int64_t>();
            if (!std::filesystem::is_regular_file(path) || pathSize(path) != size ||
                pathStamp(path) != stamp) return false;
            out.archives.push_back(CachedLayerIndex::File{{}, path, size, stamp});
        }
        return true;
    } catch (const std::exception&) {
        out = CachedLayerIndex{};
        return false;
    }
}

void saveCachedLayerIndex(
    const std::filesystem::path& cacheDirectory, const std::filesystem::path& root,
    const CachedLayerIndex& index) {
    if (cacheDirectory.empty()) return;
    std::error_code error;
    std::filesystem::create_directories(cacheDirectory, error);
    if (error) return;
    nlohmann::json files = nlohmann::json::array();
    for (const CachedLayerIndex::File& file : index.looseFiles) {
        std::error_code relativeError;
        const auto relative = std::filesystem::relative(file.path, root, relativeError);
        if (!relativeError) files.push_back({{"key", file.key},
            {"relative", relative.generic_string()}, {"size", file.size}, {"stamp", file.stamp}});
    }
    nlohmann::json archives = nlohmann::json::array();
    for (const CachedLayerIndex::File& file : index.archives) {
        std::error_code relativeError;
        const auto relative = std::filesystem::relative(file.path, root, relativeError);
        if (!relativeError) archives.push_back({{"relative", relative.generic_string()},
            {"size", file.size}, {"stamp", file.stamp}});
    }
    nlohmann::json directories = nlohmann::json::array();
    for (const CachedLayerIndex::Directory& directory : index.directories) {
        std::error_code relativeError;
        const auto relative = std::filesystem::relative(directory.path, root, relativeError);
        if (!relativeError) directories.push_back({{"relative", relative.generic_string()},
            {"stamp", directory.stamp}});
    }
    const nlohmann::json json = {{"version", 1u}, {"root", root.generic_string()},
        {"root_stamp", pathStamp(root)}, {"total_bytes", index.totalBytes},
        {"files", std::move(files)}, {"archives", std::move(archives)},
        {"directories", std::move(directories)}};
    const std::filesystem::path destination = cacheDirectory / (stablePathKey(root) + ".json");
    const std::filesystem::path temporary = destination.string() + ".tmp";
    std::ofstream output(temporary, std::ios::trunc);
    if (!output) return;
    output << json.dump() << '\n'; output.close();
    if (!output) { std::filesystem::remove(temporary, error); return; }
    std::filesystem::rename(temporary, destination, error);
    if (error) std::filesystem::remove(temporary, error);
}

}  // namespace

std::filesystem::path joinBackslashPath(std::filesystem::path base, const std::string& backslashPath) {
    std::string component;
    for (char c : backslashPath) {
        if (c == '\\' || c == '/') {
            if (!component.empty()) {
                base /= component;
                component.clear();
            }
        } else {
            component.push_back(c);
        }
    }
    if (!component.empty()) {
        base /= component;
    }
    return base;
}

std::string normalizeModelPath(std::string path) {
    for (char& c : path) {
        if (c == '/') {
            c = '\\';
        }
    }
    return path;
}

std::string normalizeTexturePath(const std::string& path) {
    std::string normalized = normalizeModelPath(path);

    // Some texture paths are stored rooted at the Data directory rather than
    // relative to it -- the distant-LOD blocks name theirs
    // "Data\Textures\Landscape\LOD\...". Strip that prefix before the
    // "textures\" check, or the result is "textures\Data\Textures\..." which
    // resolves to nothing. This is silent: an unresolvable texture is not an
    // error, the surface just shades untextured, which is how the whole LOD
    // tier cooked with zero textures.
    if (toLowerAsciiCopy(normalized).rfind("data\\", 0) == 0) {
        normalized = normalized.substr(5);
    }

    // SKYRIM SHIPS SOMEONE'S BUILD MACHINE PATH in a few texture sets:
    // "skyrimhd\build\pc\data\textures\plants\potato01.dds". Same failure as
    // the LOD case above and same shape of fix, generalized -- cut to the FIRST
    // "textures\" component wherever it appears, rather than matching each
    // leading prefix that turns up. First rather than last so a legitimate
    // "textures\...\textures\..." keeps its real root.
    //
    // Silent when wrong, again: these are Whiterun's potato, tundrashrub and
    // yellowshrub, and unresolved they render as untextured pale blobs sitting
    // where the shrubs should be.
    {
        const std::string lowered = toLowerAsciiCopy(normalized);
        const std::size_t root = lowered.find("textures\\");
        if (root != std::string::npos && root > 0u) {
            normalized = normalized.substr(root);
        }
    }

    if (toLowerAsciiCopy(normalized).rfind("textures\\", 0) != 0) {
        normalized = "textures\\" + normalized;
    }

    // MORROWIND NAMES .tga AND SHIPS .dds. Bethesda converted the archives to
    // DDS for load speed at some point and never updated the references, so
    // every texture path inside a NetImmerse mesh and every LTEX filename still
    // carries the original extension: measured over the architecture meshes in
    // Morrowind.bsa, 3429 .tga and 125 .bmp, and not one .dds. The archive holds
    // all of them as textures\<name>.dds.
    //
    // Rewritten here rather than at each call site because the same rule covers
    // meshes, land textures and anything else that names a texture -- and doing
    // it anywhere else means one of those quietly keeps resolving to nothing,
    // which shades untextured rather than failing.
    const std::string lowered = toLowerAsciiCopy(normalized);
    for (const char* legacy : {".tga", ".bmp"}) {
        if (lowered.size() > 4u && lowered.compare(lowered.size() - 4u, 4u, legacy) == 0) {
            normalized.replace(normalized.size() - 4u, 4u, ".dds");
            break;
        }
    }
    return normalized;
}

bool FalloutAssetSource::open(
    const std::filesystem::path& dataFilesPath, std::uint32_t contentMask) {
    m_profileFingerprint.clear();
    m_forceContentReindex = false;
    m_archiveAllowListEnabled = false;
    m_allowedArchivePaths.clear();
    return openDataFiles(dataFilesPath, contentMask);
}

bool FalloutAssetSource::openDataFiles(
    const std::filesystem::path& dataFilesPath, std::uint32_t contentMask) {
    m_dataFilesPath = dataFilesPath;
    m_archives.clear();
    m_warnings.clear();
    m_contentMask = contentMask;

    std::error_code directoryError;
    std::filesystem::directory_iterator iterator(dataFilesPath, directoryError);
    if (directoryError) {
        return false;
    }

    std::vector<std::filesystem::path> archivePaths;
    for (const auto& entry : iterator) {
        if (!entry.is_regular_file()) {
            continue;
        }
        if (toLowerAsciiCopy(entry.path().extension().string()) != ".bsa") {
            continue;
        }
        archivePaths.push_back(entry.path());
    }

    // Sort into Fallout's own load order, because "last match wins" is
    // meaningless without one.
    //
    // directory_iterator yields readdir order, which is filesystem state rather
    // than load order, so which archive won a name varied by machine and
    // Update.bsa (the shipped patch archive) generally lost. It overrides 36
    // meshes and 2 textures that also exist in the base archives: the Novac
    // motel and bungalows, the McCarran wall set, the NCR guard towers, the
    // Hoover Dam observation deck. All resolved to stale pre-patch versions.
    //
    // Rank base (0) < Update.bsa (1) < DLC (2), alphabetical within a rank so
    // the result is reproducible. Plain alphabetical is NOT good enough -- it
    // sorts DeadMoney.bsa ahead of "Fallout - Textures.bsa" and silently makes
    // the DLC lose.
    const auto loadOrderRank = [](const std::filesystem::path& path) {
        const std::string name = toLowerAsciiCopy(path.filename().string());
        if (name.rfind("fallout - ", 0) == 0) {
            return 0;
        }
        if (name == "update.bsa") {
            return 1;
        }
        return 2;  // DeadMoney, HonestHearts, OldWorldBlues, LonesomeRoad, packs
    };
    std::sort(
        archivePaths.begin(), archivePaths.end(),
        [&](const std::filesystem::path& a, const std::filesystem::path& b) {
            const int rankA = loadOrderRank(a);
            const int rankB = loadOrderRank(b);
            if (rankA != rankB) {
                return rankA < rankB;
            }
            return toLowerAsciiCopy(a.filename().string()) < toLowerAsciiCopy(b.filename().string());
        });

    for (const std::filesystem::path& path : archivePaths) {
        if (m_archiveAllowListEnabled &&
            !m_allowedArchivePaths.contains(pathIdentity(path))) {
            continue;
        }
        // Opening an archive costs a std::string and a hash-map entry per file.
        // "Fallout - Sound.bsa" and "Fallout - Voices1.bsa" hold 111982 files
        // between them and cannot contain a mesh or texture, so indexing them
        // was ~85% of the cooker's runtime.
        //
        // The test is deliberately conservative: skip only archives that
        // declare content AND declare nothing outside what the caller excluded.
        // An archive declaring nothing at all (fileFlags == 0) is indexed,
        // because "says nothing" is not evidence of "contains nothing" and the
        // failure mode would be a silently missing mesh rather than an error.
        std::uint32_t fileFlags = 0;
        if (!peekBsaContentFlags(path, fileFlags)) {
            m_warnings.push_back("not a readable v103/v104 BSA: " + path.string());
            continue;
        }
        if (fileFlags != 0u && (fileFlags & contentMask) == 0u) {
            continue;
        }
        BsaArchive archive;
        if (!archive.open(path)) {
            m_warnings.push_back("failed to open BSA archive " + path.string());
            continue;
        }
        m_archives.push_back(std::move(archive));
    }
    return true;
}

bool FalloutAssetSource::open(
    const ResolvedContentProfile& profile, std::uint32_t contentMask) {
    // A FalloutAssetSource is setup-only until open returns. Clear profile
    // layers explicitly so reusing one for a different profile cannot retain
    // providers from the previous virtual Data directory.
    m_modDirectories.clear();
    m_profileFingerprint = profile.fingerprint;
    m_forceContentReindex = profile.forceContentReindex;
    m_contentIndexCacheDirectory = defaultContentIndexDirectory();
    m_archiveAllowListEnabled = true;
    m_allowedArchivePaths.clear();
    for (const ContentArchive& archive : profile.archives) {
        m_allowedArchivePaths.insert(pathIdentity(archive.path));
    }
    if (!openDataFiles(profile.dataRoot, contentMask)) {
        return false;
    }
    for (const ContentLayer& layer : profile.layers) {
        if (layer.enabled && !addModDirectory(layer.root)) {
            return false;
        }
    }
    return true;
}

bool FalloutAssetSource::addModDirectory(const std::filesystem::path& directory) {
    ModDirectory mod;
    mod.root = directory;
    std::size_t caseCollisions = 0;
    std::vector<std::filesystem::path> archivePaths;
    CachedLayerIndex cached;
    const bool cacheHit = !m_forceContentReindex &&
        loadCachedLayerIndex(m_contentIndexCacheDirectory, directory, cached);
    if (cacheHit) {
        mod.totalBytes = cached.totalBytes;
        for (CachedLayerIndex::File& archive : cached.archives) {
            archivePaths.push_back(std::move(archive.path));
        }
        for (CachedLayerIndex::File& file : cached.looseFiles) {
            if (!mod.filesByLowerPath.emplace(std::move(file.key), std::move(file.path)).second) {
                ++caseCollisions;
            }
        }
    } else {
        std::error_code iterError;
        std::filesystem::recursive_directory_iterator iterator(
            directory, std::filesystem::directory_options::skip_permission_denied, iterError);
        if (iterError) {
            m_warnings.push_back("cannot read mod directory " + directory.string());
            return false;
        }
        CachedLayerIndex generated;
        generated.directories.push_back(CachedLayerIndex::Directory{directory, pathStamp(directory)});
        for (const auto& entry : iterator) {
            std::error_code typeError;
            if (entry.is_directory(typeError) && !typeError) {
                generated.directories.push_back(
                    CachedLayerIndex::Directory{entry.path(), pathStamp(entry.path())});
                continue;
            }
            typeError.clear();
            if (!entry.is_regular_file(typeError) || typeError) continue;
            if (toLowerAsciiCopy(entry.path().extension().string()) == ".bsa" &&
                entry.path().parent_path() == directory) {
                archivePaths.push_back(entry.path());
                std::error_code archiveSizeError;
                const auto archiveSize = std::filesystem::file_size(entry.path(), archiveSizeError);
                if (!archiveSizeError) mod.totalBytes += static_cast<std::uint64_t>(archiveSize);
                continue;
            }
            std::error_code relativeError;
            const std::filesystem::path relative =
                std::filesystem::relative(entry.path(), directory, relativeError);
            if (relativeError) continue;
            std::string key = toLowerAsciiCopy(relative.generic_string());
            for (char& c : key) if (c == '/') c = '\\';
            std::error_code sizeError;
            const auto size = entry.file_size(sizeError);
            if (!sizeError) mod.totalBytes += static_cast<std::uint64_t>(size);
            if (!mod.filesByLowerPath.emplace(std::move(key), entry.path()).second) {
                ++caseCollisions;
            }
        }
        generated.totalBytes = mod.totalBytes;
        for (const std::filesystem::path& path : archivePaths) {
            generated.archives.push_back(CachedLayerIndex::File{
                {}, path, pathSize(path), pathStamp(path)});
        }
        generated.looseFiles.reserve(mod.filesByLowerPath.size());
        for (const auto& [key, path] : mod.filesByLowerPath) {
            generated.looseFiles.push_back(CachedLayerIndex::File{
                key, path, pathSize(path), pathStamp(path)});
        }
        saveCachedLayerIndex(m_contentIndexCacheDirectory, directory, generated);
    }

    if (caseCollisions != 0u) {
        m_warnings.push_back(
            "mod directory " + directory.string() + " has " + std::to_string(caseCollisions) +
            " files differing only by case; the first of each was kept");
    }

    // Name order, so which archive wins a duplicate name is reproducible rather
    // than being whatever readdir returned.
    std::sort(
        archivePaths.begin(), archivePaths.end(),
        [](const std::filesystem::path& a, const std::filesystem::path& b) {
            return toLowerAsciiCopy(a.filename().string()) < toLowerAsciiCopy(b.filename().string());
        });
    if (!m_profileFingerprint.empty()) {
        // Fold the normalized indexed manifest into the cache identity. The
        // profile resolver captures ordered roots and plugin metadata; this
        // captures nested loose files too, including in-place replacements
        // that do not change the root directory timestamp.
        m_profileFingerprint += "-" +
            indexedLayerFingerprint(directory, mod.filesByLowerPath, archivePaths);
    }
    for (const std::filesystem::path& path : archivePaths) {
        if (m_archiveAllowListEnabled &&
            !m_allowedArchivePaths.contains(pathIdentity(path))) {
            continue;
        }
        std::uint32_t fileFlags = 0;
        if (!peekBsaContentFlags(path, fileFlags)) {
            m_warnings.push_back("not a readable v103/v104 BSA: " + path.string());
            continue;
        }
        // Same conservative test as the game's own archives: skip only an
        // archive that declares content AND declares nothing the caller wants.
        if (fileFlags != 0u && (fileFlags & m_contentMask) == 0u) {
            continue;
        }
        BsaArchive archive;
        if (!archive.open(path)) {
            m_warnings.push_back("failed to open BSA archive " + path.string());
            continue;
        }
        mod.archives.push_back(std::move(archive));
    }

    m_modDirectories.push_back(std::move(mod));
    return true;
}

std::size_t FalloutAssetSource::modArchiveCount() const {
    std::size_t count = 0;
    for (const ModDirectory& mod : m_modDirectories) {
        count += mod.archives.size();
    }
    return count;
}

std::size_t FalloutAssetSource::modFileCount() const {
    std::size_t count = 0;
    for (const ModDirectory& mod : m_modDirectories) {
        count += mod.filesByLowerPath.size();
    }
    return count;
}

std::string FalloutAssetSource::modFingerprint() const {
    if (!m_profileFingerprint.empty()) {
        return m_profileFingerprint;
    }
    if (m_modDirectories.empty()) {
        return {};
    }
    std::string fingerprint;
    for (const ModDirectory& mod : m_modDirectories) {
        if (!fingerprint.empty()) {
            fingerprint += "-";
        }
        // The result becomes a directory name, so keep it to characters no
        // filesystem argues about.
        for (char c : toLowerAsciiCopy(mod.root.filename().string())) {
            const bool safe = (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9');
            fingerprint.push_back(safe ? c : '_');
        }
        fingerprint += "_" + std::to_string(mod.filesByLowerPath.size()) + "_" +
            std::to_string(mod.totalBytes);
    }
    return fingerprint;
}

bool FalloutAssetSource::resolve(
    const std::string& loosePathSuffix,
    const std::filesystem::path& looseRoot,
    const std::string& archiveVirtualPath,
    std::vector<std::uint8_t>& outBytes,
    std::string& outError) const {
    // Mods win over everything the game shipped. Reverse, for the same reason
    // the archive loop below is reversed: m_modDirectories is in load order.
    if (!m_modDirectories.empty()) {
        const std::string modKey = toLowerAsciiCopy(archiveVirtualPath);
        for (auto it = m_modDirectories.rbegin(); it != m_modDirectories.rend(); ++it) {
            // This mod's loose files first, then its own archives -- the same
            // "loose beats archives" rule the game applies, scoped to one mod.
            const auto found = it->filesByLowerPath.find(modKey);
            if (found != it->filesByLowerPath.end()) {
                std::string modError;
                if (readWholeFile(found->second, outBytes, modError)) {
                    return true;
                }
                // Indexed but unreadable is worth reporting even if something
                // later satisfies the request.
                outError = modError;
            }
            for (auto archiveIt = it->archives.rbegin(); archiveIt != it->archives.rend();
                 ++archiveIt) {
                const BsaFileEntry* entry = archiveIt->find(archiveVirtualPath);
                if (entry == nullptr) {
                    continue;
                }
                std::string archiveError;
                if (archiveIt->extract(*entry, outBytes, archiveError)) {
                    return true;
                }
                outError = archiveError;
            }
        }
    }

    const std::filesystem::path loosePath = joinBackslashPath(looseRoot, loosePathSuffix);
    std::error_code existsError;
    if (std::filesystem::exists(loosePath, existsError) && !existsError) {
        std::string looseError;
        if (readWholeFile(loosePath, outBytes, looseError)) {
            return true;
        }
        // A loose file that exists but will not read is worth reporting even if
        // an archive copy then satisfies the request.
        outError = looseError;
    }

    // Reverse: m_archives is in load order, so the last archive holding a name
    // is the one the game would use.
    for (auto it = m_archives.rbegin(); it != m_archives.rend(); ++it) {
        const BsaFileEntry* entry = it->find(archiveVirtualPath);
        if (entry == nullptr) {
            continue;
        }
        std::string archiveError;
        if (it->extract(*entry, outBytes, archiveError)) {
            return true;
        }
        outError = archiveError;
    }
    if (outError.empty()) {
        outError = "not found in loose files or archives: " + archiveVirtualPath;
    }
    return false;
}

bool FalloutAssetSource::resolveAsset(
    const std::string& virtualPath, std::vector<std::uint8_t>& outBytes,
    std::string& outError) const {
    outError.clear();
    const std::string normalized = normalizeModelPath(virtualPath);
    // Loose root is the Data directory itself, because the path is already
    // rooted there -- exactly the texture rule, minus the prefix insertion.
    return resolve(normalized, m_dataFilesPath, normalized, outBytes, outError);
}

bool FalloutAssetSource::resolveMesh(
    const std::string& modelPath, std::vector<std::uint8_t>& outBytes, std::string& outError) const {
    outError.clear();
    // MODL paths are relative to Data\Meshes on disk but carry the "meshes\"
    // prefix inside archives.
    const std::string normalized = normalizeModelPath(modelPath);
    return resolve(
        normalized, m_dataFilesPath / "meshes", "meshes\\" + normalized, outBytes, outError);
}

bool FalloutAssetSource::resolveTexture(
    const std::string& texturePath, std::vector<std::uint8_t>& outBytes, std::string& outError) const {
    outError.clear();
    // Texture paths already carry "textures\", so the loose root is Data Files
    // itself rather than Data Files\textures.
    const std::string normalized = normalizeTexturePath(texturePath);
    return resolve(normalized, m_dataFilesPath, normalized, outBytes, outError);
}

}  // namespace odai::importer::fnv
