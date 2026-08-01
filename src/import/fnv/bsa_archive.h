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
//     offset-to-that-folder's-name+file-records.
//   Per folder, at its offset: an optional BString folder name (present when
//     archiveFlags & kHasFolderNames), followed by that folder's file records
//     (16 bytes each: nameHash, size, dataOffset).
//   File name block (present when archiveFlags & kHasFileNames): every file's
//     name as a NUL-terminated string, concatenated in file-record order.
//   File data: raw bytes, or (when compressed) a 4-byte original-size prefix
//     followed by zlib-deflate data.

#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace odai::importer::fnv {

struct BsaFileEntry {
    std::string virtualPath;  // lowercase, backslash-normalized, e.g. "meshes\\x\\ex_wall_01.nif"
    std::uint64_t nameHash = 0;
    std::uint32_t dataOffset = 0;
    std::uint32_t sizeOnDisk = 0;  // compressed size on disk when compressed, else the file size
    bool compressed = false;
};

class BsaArchive {
public:
    bool open(const std::filesystem::path& path);

    const std::vector<BsaFileEntry>& files() const { return m_files; }

    // Case-insensitive lookup by virtual path (backslash or forward-slash
    // separators both accepted).
    const BsaFileEntry* find(std::string_view virtualPath) const;

    // Reads and (if needed) inflates a file's bytes.
    bool extract(const BsaFileEntry& entry, std::vector<std::uint8_t>& outBytes) const;

    const std::string& lastError() const { return m_lastError; }

private:
    std::filesystem::path m_path;
    std::vector<BsaFileEntry> m_files;
    std::unordered_map<std::string, std::size_t> m_pathIndex;  // lowercase virtualPath -> m_files index
    mutable std::string m_lastError;
};

}  // namespace odai::importer::fnv
