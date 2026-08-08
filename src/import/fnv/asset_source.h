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
// THREADING: open() is single-threaded setup. Once it returns, resolveMesh()
// and resolveTexture() are const and safe to call concurrently from any number
// of worker threads -- BsaArchive::extract() opens its own ifstream per call
// and the archive list is not mutated afterwards. This is the property the
// background asset pipeline depends on.

#include <cstdint>
#include <filesystem>
#include <string>
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

    // Both are const and thread safe once open() has returned. outError is
    // written only on failure.
    bool resolveMesh(
        const std::string& modelPath, std::vector<std::uint8_t>& outBytes, std::string& outError) const;
    bool resolveTexture(
        const std::string& texturePath, std::vector<std::uint8_t>& outBytes, std::string& outError) const;

    [[nodiscard]] const std::vector<std::string>& warnings() const { return m_warnings; }
    [[nodiscard]] std::size_t archiveCount() const { return m_archives.size(); }
    [[nodiscard]] const std::filesystem::path& dataFilesPath() const { return m_dataFilesPath; }

private:
    // Shared tail of both resolvers: loose file first, then archives in reverse
    // load order.
    bool resolve(
        const std::string& loosePathSuffix,
        const std::filesystem::path& looseRoot,
        const std::string& archiveVirtualPath,
        std::vector<std::uint8_t>& outBytes,
        std::string& outError) const;

    std::filesystem::path m_dataFilesPath;
    std::vector<BsaArchive> m_archives;  // load order; later entries win
    std::vector<std::string> m_warnings;
};

}  // namespace odai::importer::fnv
