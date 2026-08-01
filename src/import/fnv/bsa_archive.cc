#include "import/fnv/bsa_archive.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstring>
#include <fstream>

#include <zlib.h>

namespace odai::importer::fnv {

namespace {

constexpr std::uint32_t kBsaMagic = 0x00415342u;  // "BSA\0"
constexpr std::uint32_t kBsaVersionFo3Fnv = 104u;

constexpr std::uint32_t kFlagHasFolderNames = 0x1u;
constexpr std::uint32_t kFlagHasFileNames = 0x2u;
constexpr std::uint32_t kFlagCompressedArchive = 0x4u;
// Per-file size field: bit 30 toggles this file's compression relative to the
// archive-wide default set by kFlagCompressedArchive.
constexpr std::uint32_t kFileCompressionToggleBit = 0x40000000u;
constexpr std::uint32_t kFileSizeMask = 0x3fffffffu;

#pragma pack(push, 1)
struct BsaHeader {
    std::uint32_t magic;
    std::uint32_t version;
    std::uint32_t folderRecordOffset;
    std::uint32_t archiveFlags;
    std::uint32_t folderCount;
    std::uint32_t fileCount;
    std::uint32_t totalFolderNameLength;
    std::uint32_t totalFileNameLength;
    std::uint32_t fileFlags;
};

struct BsaFolderRecord {
    std::uint64_t nameHash;
    std::uint32_t fileCount;
    std::uint32_t offset;  // absolute file offset of this folder's name+file records
};

struct BsaRawFileRecord {
    std::uint64_t nameHash;
    std::uint32_t size;
    std::uint32_t offset;  // absolute file offset of this file's data
};
#pragma pack(pop)

static_assert(sizeof(BsaHeader) == 36);
static_assert(sizeof(BsaFolderRecord) == 16);
static_assert(sizeof(BsaRawFileRecord) == 16);

std::string toLowerAscii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

std::string normalizeSeparators(std::string value) {
    for (char& c : value) {
        if (c == '/') {
            c = '\\';
        }
    }
    return value;
}

template <typename T>
bool readValue(std::istream& input, T& out) {
    static_assert(std::is_trivially_copyable_v<T>);
    input.read(reinterpret_cast<char*>(&out), static_cast<std::streamsize>(sizeof(T)));
    return input.good();
}

// Reads a BSA "BString": a single length-prefix byte followed by that many
// bytes (folder-name strings include a trailing NUL in the count; file names
// in the name block are plain NUL-terminated strings instead).
bool readBString(std::istream& input, std::string& out) {
    std::uint8_t length = 0;
    if (!readValue(input, length)) {
        return false;
    }
    out.resize(length);
    if (length != 0 && !input.read(out.data(), length)) {
        return false;
    }
    while (!out.empty() && out.back() == '\0') {
        out.pop_back();
    }
    return true;
}

bool readNulTerminated(std::istream& input, std::string& out) {
    out.clear();
    char ch = 0;
    while (input.get(ch)) {
        if (ch == '\0') {
            return true;
        }
        out.push_back(ch);
    }
    return false;
}

}  // namespace

bool BsaArchive::open(const std::filesystem::path& path) {
    m_lastError.clear();
    m_files.clear();
    m_pathIndex.clear();
    m_path = path;

    std::ifstream input(path, std::ios::binary);
    if (!input) {
        m_lastError = "Failed to open BSA archive: " + path.string();
        return false;
    }

    BsaHeader header{};
    if (!readValue(input, header) || header.magic != kBsaMagic) {
        m_lastError = "Not a BSA archive (bad magic): " + path.string();
        return false;
    }
    if (header.version != kBsaVersionFo3Fnv) {
        m_lastError = "Unsupported BSA version " + std::to_string(header.version) +
            " (only Fallout 3 / New Vegas v104 archives are supported): " + path.string();
        return false;
    }

    const bool hasFolderNames = (header.archiveFlags & kFlagHasFolderNames) != 0u;
    const bool hasFileNames = (header.archiveFlags & kFlagHasFileNames) != 0u;
    const bool defaultCompressed = (header.archiveFlags & kFlagCompressedArchive) != 0u;

    input.seekg(static_cast<std::streamoff>(header.folderRecordOffset), std::ios::beg);
    std::vector<BsaFolderRecord> folders(header.folderCount);
    for (BsaFolderRecord& folder : folders) {
        if (!readValue(input, folder)) {
            m_lastError = "Truncated BSA folder record table: " + path.string();
            return false;
        }
    }

    struct PendingFile {
        std::string folderPath;
        BsaRawFileRecord raw;
    };
    std::vector<PendingFile> pending;
    pending.reserve(header.fileCount);

    for (const BsaFolderRecord& folder : folders) {
        input.seekg(static_cast<std::streamoff>(folder.offset), std::ios::beg);
        std::string folderName;
        if (hasFolderNames && !readBString(input, folderName)) {
            m_lastError = "Truncated BSA folder name: " + path.string();
            return false;
        }
        for (std::uint32_t fileIndex = 0; fileIndex < folder.fileCount; ++fileIndex) {
            PendingFile file{};
            file.folderPath = folderName;
            if (!readValue(input, file.raw)) {
                m_lastError = "Truncated BSA file record table: " + path.string();
                return false;
            }
            pending.push_back(std::move(file));
        }
    }

    if (hasFileNames) {
        for (PendingFile& file : pending) {
            std::string fileName;
            if (!readNulTerminated(input, fileName)) {
                m_lastError = "Truncated BSA file name block: " + path.string();
                return false;
            }
            std::string virtualPath = file.folderPath.empty()
                ? fileName
                : file.folderPath + "\\" + fileName;
            virtualPath = normalizeSeparators(std::move(virtualPath));

            BsaFileEntry entry{};
            entry.virtualPath = virtualPath;
            entry.nameHash = file.raw.nameHash;
            entry.dataOffset = file.raw.offset;
            entry.sizeOnDisk = file.raw.size & kFileSizeMask;
            const bool toggled = (file.raw.size & kFileCompressionToggleBit) != 0u;
            entry.compressed = defaultCompressed != toggled;
            m_files.push_back(entry);
        }
    } else {
        // No file-name block: expose entries by hash only. Path-based lookup
        // (find()) will not resolve these; callers must iterate files().
        for (const PendingFile& file : pending) {
            BsaFileEntry entry{};
            entry.nameHash = file.raw.nameHash;
            entry.dataOffset = file.raw.offset;
            entry.sizeOnDisk = file.raw.size & kFileSizeMask;
            const bool toggled = (file.raw.size & kFileCompressionToggleBit) != 0u;
            entry.compressed = defaultCompressed != toggled;
            m_files.push_back(entry);
        }
    }

    for (std::size_t i = 0; i < m_files.size(); ++i) {
        if (!m_files[i].virtualPath.empty()) {
            m_pathIndex[toLowerAscii(m_files[i].virtualPath)] = i;
        }
    }

    return true;
}

const BsaFileEntry* BsaArchive::find(std::string_view virtualPath) const {
    const std::string key = toLowerAscii(normalizeSeparators(std::string(virtualPath)));
    const auto it = m_pathIndex.find(key);
    return it == m_pathIndex.end() ? nullptr : &m_files[it->second];
}

bool BsaArchive::extract(const BsaFileEntry& entry, std::vector<std::uint8_t>& outBytes) const {
    m_lastError.clear();
    std::ifstream input(m_path, std::ios::binary);
    if (!input) {
        m_lastError = "Failed to reopen BSA archive: " + m_path.string();
        return false;
    }
    input.seekg(static_cast<std::streamoff>(entry.dataOffset), std::ios::beg);

    if (!entry.compressed) {
        outBytes.resize(entry.sizeOnDisk);
        if (entry.sizeOnDisk != 0 &&
            !input.read(reinterpret_cast<char*>(outBytes.data()), static_cast<std::streamsize>(entry.sizeOnDisk))) {
            m_lastError = "Truncated BSA file data: " + entry.virtualPath;
            return false;
        }
        return true;
    }

    std::uint32_t originalSize = 0;
    if (!readValue(input, originalSize) || entry.sizeOnDisk < sizeof(originalSize)) {
        m_lastError = "Truncated compressed BSA entry header: " + entry.virtualPath;
        return false;
    }
    const std::uint32_t compressedSize = entry.sizeOnDisk - static_cast<std::uint32_t>(sizeof(originalSize));
    std::vector<std::uint8_t> compressed(compressedSize);
    if (compressedSize != 0 &&
        !input.read(reinterpret_cast<char*>(compressed.data()), static_cast<std::streamsize>(compressedSize))) {
        m_lastError = "Truncated compressed BSA data: " + entry.virtualPath;
        return false;
    }

    outBytes.resize(originalSize);
    uLongf destLen = static_cast<uLongf>(originalSize);
    const int result = uncompress(
        outBytes.empty() ? nullptr : reinterpret_cast<Bytef*>(outBytes.data()),
        &destLen,
        compressed.empty() ? nullptr : reinterpret_cast<const Bytef*>(compressed.data()),
        static_cast<uLong>(compressed.size()));
    if (result != Z_OK || destLen != originalSize) {
        m_lastError = "zlib inflate failed for BSA entry: " + entry.virtualPath;
        return false;
    }
    return true;
}

}  // namespace odai::importer::fnv
