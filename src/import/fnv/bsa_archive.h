#pragma once

// Reader for Bethesda BSA archives version 104 (Fallout 3 / Fallout: New
// Vegas / original Skyrim). Big-endian ("Xbox360") archives and the Skyrim
// Special Edition v105 embedded-file-name extension are not supported —
// New Vegas never produced either.
//
// Format reference (BSA v103/v104, all fields little-endian):
//   Header (36 bytes): "BSA\0", version, folderRecordOffset, archiveFlags,
//     folderCount, fileCount, totalFolderNameLength, totalFileNameLength,
//     fileFlags.
//   Folder records (16 bytes each, folderCount of them): nameHash, fileCount,
//     offset. The offset is NOT a plain file offset: it is biased by
//     totalFileNameLength, so the real seek target is
//     `offset - header.totalFileNameLength`. Retail archives all carry this
//     bias; reading it as absolute lands in the middle of the file-name block
//     and yields garbage folder names.
//   Per folder, at its (unbiased) offset: an optional BString folder name
//     (present when archiveFlags & kHasFolderNames), followed by that folder's
//     file records (16 bytes each: nameHash, size, dataOffset).
//   File name block (present when archiveFlags & kHasFileNames): every file's
//     name as a NUL-terminated string, concatenated in file-record order.
//   File data, in this order:
//     - when archiveFlags & kEmbedFileNames (0x100): a BString holding the
//       file's full virtual path. Its bytes count toward the file record's
//       `size`, so they must be deducted before reading the payload. Both
//       retail texture archives set this flag.
//     - when the file is compressed: a 4-byte uncompressed-size prefix.
//     - the payload: raw bytes, or zlib-deflate data when compressed.

#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace odai::importer::fnv {

// Header `fileFlags` bits: what kinds of content an archive holds. Bethesda
// sets these correctly on retail archives, so they can be used to skip opening
// an archive whose contents are irrelevant — which matters because opening one
// builds a std::string path and a hash-map entry per file, and
// "Fallout - Voices1.bsa" alone has 105517 of them.
enum BsaContentFlags : std::uint32_t {
    kBsaContentMeshes   = 0x001u,  // .nif
    kBsaContentTextures = 0x002u,  // .dds
    kBsaContentMenus    = 0x004u,  // .xml
    kBsaContentSounds   = 0x008u,  // .wav
    kBsaContentVoices   = 0x010u,  // .mp3 / .ogg
    kBsaContentShaders  = 0x020u,
    kBsaContentTrees    = 0x040u,  // .spt
    kBsaContentFonts    = 0x080u,
    kBsaContentMisc     = 0x100u,
};

struct BsaFileEntry {
    std::string virtualPath;  // lowercase, backslash-normalized, e.g. "meshes\\x\\ex_wall_01.nif"
    std::uint64_t nameHash = 0;
    std::uint32_t dataOffset = 0;
    std::uint32_t sizeOnDisk = 0;  // compressed size on disk when compressed, else the file size
    bool compressed = false;
};

// Reads only the 36-byte header and reports the archive's content flags (a
// mask of BsaContentFlags). Lets a caller decide whether an archive is worth
// opening before paying to index every file in it. Returns false if the file
// is not a readable v104 BSA.
bool peekBsaContentFlags(const std::filesystem::path& path, std::uint32_t& outFileFlags);

class BsaArchive {
public:
    bool open(const std::filesystem::path& path);

    // Mask of BsaContentFlags declared by the archive header.
    std::uint32_t contentFlags() const { return m_fileFlags; }

    const std::vector<BsaFileEntry>& files() const { return m_files; }

    // Case-insensitive lookup by virtual path (backslash or forward-slash
    // separators both accepted).
    const BsaFileEntry* find(std::string_view virtualPath) const;

    // Reads and (if needed) inflates a file's bytes.
    //
    // THREAD SAFE against other extract() calls on the same archive: it opens
    // its own ifstream per call and touches no member state, which is what lets
    // several worker threads pull assets out of one archive at once. The error
    // goes through outError rather than lastError() precisely so this stays
    // true -- a shared mutable error string would be a data race, and one that
    // corrupts a std::string rather than failing visibly.
    //
    // open() is NOT thread safe and must complete before any extract() begins.
    bool extract(
        const BsaFileEntry& entry,
        std::vector<std::uint8_t>& outBytes,
        std::string& outError) const;

    // Convenience overload for single-threaded callers; records the error in
    // lastError(). Do not use from worker threads.
    bool extract(const BsaFileEntry& entry, std::vector<std::uint8_t>& outBytes) const {
        return extract(entry, outBytes, m_lastError);
    }

    // Only meaningful for open() and the single-threaded extract() overload.
    const std::string& lastError() const { return m_lastError; }

private:
    std::filesystem::path m_path;
    std::uint32_t m_archiveFlags = 0;
    std::uint32_t m_fileFlags = 0;
    std::vector<BsaFileEntry> m_files;
    std::unordered_map<std::string, std::size_t> m_pathIndex;  // lowercase virtualPath -> m_files index
    mutable std::string m_lastError;
};

}  // namespace odai::importer::fnv
