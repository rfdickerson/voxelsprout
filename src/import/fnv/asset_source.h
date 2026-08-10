#pragma once

// Resolves Fallout asset paths to bytes, from loose files under Data Files or
// from the BSA archives, honouring the game's own precedence.
//
// Extracted from the cooker so the runtime streamer can share one
// implementation. The precedence rules here are not obvious and were arrived at
// by measurement, so having two copies drift apart would be a genuine hazard:
//   * loose files beat archives;
//   * among archives, LAST in load order wins (Update.bsa overrides 36
//     base-game meshes and has to win).
//
// Mod directories (see addModDirectory) sit ahead of both, which is where an
// installed texture pack lives.
//
// THREADING: open() and addModDirectory() are single-threaded setup. Once they
// have returned, resolveMesh() and resolveTexture() are const and safe to call
// concurrently from any number of worker threads -- BsaArchive::extract() opens
// its own ifstream per call, and neither the archive list nor the mod index is
// mutated afterwards. This is the property the background asset pipeline
// depends on.

#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

#include "import/fnv/bsa_archive.h"

namespace odai::importer::fnv {

// Bethesda paths are backslash-separated regardless of host OS. On POSIX,
// std::filesystem::path::operator/ treats '\' as an ordinary filename character
// rather than a separator, so appending a raw backslash-joined string looks for
// one literal (and wrong) filename instead of walking subdirectories. Split
// explicitly and join each component so the result is correct everywhere.
std::filesystem::path joinBackslashPath(std::filesystem::path base, const std::string& backslashPath);

// Slashes unified to backslashes; case is left alone (BsaArchive::find is
// already case-insensitive).
std::string normalizeModelPath(std::string path);

// As above, plus the "textures\" prefix when it is missing -- paths stored in a
// BSShaderTextureSet usually carry it, but not always.
std::string normalizeTexturePath(const std::string& path);

class FalloutAssetSource {
public:
    // Indexes the BSA archives in `dataFilesPath`. `contentMask` is a mask of
    // BsaContentFlags; archives declaring none of those bits are skipped
    // without being indexed, which matters because indexing builds a string and
    // a hash entry per file and "Fallout - Voices1.bsa" alone has 105517.
    //
    // Returns false only if the directory cannot be read; individual archives
    // that fail to open are reported through warnings() and skipped.
    bool open(
        const std::filesystem::path& dataFilesPath,
        std::uint32_t contentMask = kBsaContentMeshes | kBsaContentTextures);

    // Adds a mod root laid out like the Data directory itself
    // ("textures\...", "meshes\..."), searched ahead of the game's own loose
    // files and archives.
    //
    // The directory is indexed recursively here rather than probed per lookup,
    // keyed by LOWERCASED backslash-joined relative path. That is not an
    // optimization, it is the only thing that works: resolve()'s loose-file path
    // is a plain exists() on whatever case the NIF happened to store, which on a
    // case-sensitive filesystem finds nothing when the pack ships "Textures\"
    // and the NIF asks for "textures\". Archives never had this problem because
    // BsaArchive::find is already case-insensitive. Indexing makes a lookup one
    // hash probe with no stat at all, and 5403 files cost a few ms.
    //
    // Any .bsa sitting at the mod root is opened too, and loses to that same
    // mod's loose files but beats everything earlier in the order -- the same
    // shape as the game's own "loose beats archives" rule, applied per mod.
    //
    // Precedence matches the archive rule: LAST directory added wins, so adding
    // mods in load order does what a mod manager would. May be called before or
    // after open(); open() does not clear the mod list. When called before
    // open(), mod archives are filtered by the default content mask, since the
    // caller's mask is not known yet.
    //
    // Returns false if the directory cannot be read, which is also reported
    // through warnings().
    bool addModDirectory(const std::filesystem::path& directory);

    // Archives opened from mod directories, across all of them.
    [[nodiscard]] std::size_t modArchiveCount() const;

    // Identifies the mod set, for folding into a cache key -- cached cells bake
    // their textures in, so a cache built before a texture pack was installed
    // has to miss rather than serve vanilla art forever. Empty when no mod
    // directory was added, which keeps an unmodded cache key byte-identical to
    // what it was before mods existed.
    [[nodiscard]] std::string modFingerprint() const;

    // Resolves an arbitrary virtual path -- "sound\\fx\\weather\\...", a menu XML,
    // anything that is neither a mesh nor a texture -- with the same precedence
    // as the two typed resolvers, and with no prefix inserted.
    //
    // This is what lets a mod's own audio win: Nevada Skies ships its weather
    // loops inside its BSA, which is already indexed as a mod archive, so the
    // sound the mod authored for a weather beats anything the base game has.
    bool resolveAsset(
        const std::string& virtualPath, std::vector<std::uint8_t>& outBytes,
        std::string& outError) const;

    // Both are const and thread safe once open() has returned. outError is
    // written only on failure.
    bool resolveMesh(
        const std::string& modelPath, std::vector<std::uint8_t>& outBytes, std::string& outError) const;
    bool resolveTexture(
        const std::string& texturePath, std::vector<std::uint8_t>& outBytes, std::string& outError) const;

    [[nodiscard]] const std::vector<std::string>& warnings() const { return m_warnings; }
    [[nodiscard]] std::size_t archiveCount() const { return m_archives.size(); }
    [[nodiscard]] std::size_t modDirectoryCount() const { return m_modDirectories.size(); }
    [[nodiscard]] std::size_t modFileCount() const;
    [[nodiscard]] const std::filesystem::path& dataFilesPath() const { return m_dataFilesPath; }

private:
    struct ModDirectory {
        std::filesystem::path root;
        // Lowercased "textures\foo\bar.dds" -> where it actually is on disk.
        std::unordered_map<std::string, std::filesystem::path> filesByLowerPath;
        // .bsa archives found at the mod root, in name order. A mod that ships
        // an archive rather than loose files (Nevada Skies is 330 MB of one) is
        // otherwise invisible: its cloud and sky textures exist only inside it.
        std::vector<BsaArchive> archives;
        std::uint64_t totalBytes = 0;
    };

    // Shared tail of both resolvers: mod directories first, then the game's own
    // loose file, then archives in reverse load order.
    bool resolve(
        const std::string& loosePathSuffix,
        const std::filesystem::path& looseRoot,
        const std::string& archiveVirtualPath,
        std::vector<std::uint8_t>& outBytes,
        std::string& outError) const;

    std::filesystem::path m_dataFilesPath;
    std::vector<BsaArchive> m_archives;  // load order; later entries win
    std::vector<ModDirectory> m_modDirectories;  // load order; later entries win
    // Remembered from open() so a mod directory added afterwards filters its
    // archives the same way the game's were.
    std::uint32_t m_contentMask = kBsaContentMeshes | kBsaContentTextures;
    std::vector<std::string> m_warnings;
};

}  // namespace odai::importer::fnv
