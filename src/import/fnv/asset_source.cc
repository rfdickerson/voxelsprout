#include "import/fnv/asset_source.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <system_error>

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

    if (toLowerAsciiCopy(normalized).rfind("textures\\", 0) != 0) {
        normalized = "textures\\" + normalized;
    }
    return normalized;
}

bool FalloutAssetSource::open(
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

bool FalloutAssetSource::addModDirectory(const std::filesystem::path& directory) {
    std::error_code iterError;
    std::filesystem::recursive_directory_iterator iterator(
        directory, std::filesystem::directory_options::skip_permission_denied, iterError);
    if (iterError) {
        m_warnings.push_back("cannot read mod directory " + directory.string());
        return false;
    }

    ModDirectory mod;
    mod.root = directory;
    std::size_t caseCollisions = 0;
    std::vector<std::filesystem::path> archivePaths;
    for (const auto& entry : iterator) {
        std::error_code typeError;
        if (!entry.is_regular_file(typeError) || typeError) {
            continue;
        }
        // A .bsa at the mod root is an archive to open, not a file to serve by
        // path -- nothing ever asks to resolve "nevadaskies.bsa".
        if (toLowerAsciiCopy(entry.path().extension().string()) == ".bsa" &&
            entry.path().parent_path() == directory) {
            archivePaths.push_back(entry.path());
            // Counted toward the fingerprint even though it is not served by
            // path. A mod that ships ONE archive and no loose files -- Nevada
            // Skies is 330 MB of exactly that -- otherwise fingerprints as
            // "<name>_0_0" whatever the archive holds, so updating the pack
            // leaves the cell-cache key identical and every cached cell keeps
            // serving the old art. That is the failure the fingerprint exists
            // to prevent.
            std::error_code archiveSizeError;
            const auto archiveSize = std::filesystem::file_size(entry.path(), archiveSizeError);
            if (!archiveSizeError) {
                mod.totalBytes += static_cast<std::uint64_t>(archiveSize);
            }
            continue;
        }
        std::error_code relativeError;
        const std::filesystem::path relative =
            std::filesystem::relative(entry.path(), directory, relativeError);
        if (relativeError) {
            continue;
        }
        // generic_string() so the separator is '/' on every host before the
        // rewrite to Bethesda's '\'.
        std::string key = toLowerAsciiCopy(relative.generic_string());
        for (char& c : key) {
            if (c == '/') {
                c = '\\';
            }
        }
        std::error_code sizeError;
        const auto size = entry.file_size(sizeError);
        if (!sizeError) {
            mod.totalBytes += static_cast<std::uint64_t>(size);
        }
        // Two files differing only by case collapse to one key. The first wins,
        // arbitrarily -- there is no right answer, and the game itself could not
        // tell them apart either.
        if (!mod.filesByLowerPath.emplace(std::move(key), entry.path()).second) {
            ++caseCollisions;
        }
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
    for (const std::filesystem::path& path : archivePaths) {
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
