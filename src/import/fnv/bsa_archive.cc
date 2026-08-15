#include "import/fnv/bsa_archive.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstring>
#include <fstream>

#include <zlib.h>

#include "import/fnv/lz4_frame.h"

namespace odai::importer::fnv {

namespace {

constexpr std::uint32_t kBsaMagic = 0x00415342u;  // "BSA\0"
// Morrowind, which predates the magic. The first word of the file is this
// value and nothing else identifies it, so it has to be recognized before the
// magic check rather than after it.
constexpr std::uint32_t kMorrowindArchiveVersion = 0x00000100u;
constexpr std::uint32_t kBsaVersionFo3Fnv = 104u;
// Oblivion-era BSA (TES4), and two separate reasons to read one.
//
// Plenty of New Vegas mods still ship v103 -- Willow's 258 MB archive is one --
// and the game reads them. Rejecting it cost every mesh, texture and voice line
// those mods ship. Oblivion itself is the other.
//
// Every structure this reader touches is byte-identical to v104: the 36-byte
// header, 16-byte folder records, 16-byte file records, the totalFileNameLength
// bias on folder offsets, the NUL-terminated name block, and zlib payloads
// behind a uint32 original-size prefix. Verified by parsing all 17 retail
// Oblivion archives with the v104 layout -- 20182 entries index out of
// "Oblivion - Meshes.bsa" with correct paths, and 40 of 40 sampled files
// inflate to their declared size. The version differences are in which archive
// FLAGS are defined, and the ONE that matters here is kFlagEmbedFileNames below.
constexpr std::uint32_t kBsaVersionOblivion = 103u;
// Skyrim Special Edition (TES5 SE). Measured on the retail archives.
//
// TWO STRUCTURAL DIFFERENCES, not one. The folder record grew from 16 bytes to
// 24 -- a 4-byte pad and a 64-bit data offset replacing the 32-bit one -- so a
// reader that assumes the v104 stride reads the folder table as noise and every
// name comes out as binary garbage rather than failing. And compressed entries
// are LZ4 FRAMES, not zlib: the payload opens with 04 22 4d 18. Both were
// confirmed against "Skyrim - Meshes0.bsa" (978 folders, 19443 files), where
// the 24-byte stride yields "meshes\\dungeons\\nordic\\levers\\pullchainanim"
// as the first folder and the 16-byte stride yields nothing readable.
//
// Everything else is v104: the 36-byte header, 16-byte FILE records, the
// totalFileNameLength bias, the name block, and the uint32 original-size prefix
// ahead of the compressed payload.
constexpr std::uint32_t kBsaVersionSkyrimSe = 105u;

constexpr std::uint32_t kFlagHasFolderNames = 0x1u;
constexpr std::uint32_t kFlagHasFileNames = 0x2u;
constexpr std::uint32_t kFlagCompressedArchive = 0x4u;
// Each file's data block is prefixed by a BString holding its full virtual
// path. Set on both retail texture archives (archiveFlags 0x107).
//
// FALLOUT 3 ONWARD. THIS BIT IS MEANINGLESS IN v103 AND MUST BE IGNORED THERE.
// Oblivion sets it anyway -- "Oblivion - Meshes.bsa" is 0x787 and "Oblivion -
// Textures - Compressed.bsa" is 0x707, both with the bit on -- but no v103 data
// block carries an embedded name. Honouring it eats the first 1+N bytes of the
// payload, so each extraction starts a few bytes late: a compressed archive
// fails outright with "incorrect header check", and an uncompressed one yields
// garbage that does NOT fail loudly -- a DDS decodes to nothing, a NIF parses to
// no shapes -- so it reads as "the mod ships broken assets" rather than as a
// reader bug. Measured both ways: on all five sampled files of Willow's v103
// archive, and on all 40 sampled Oblivion files, before the version guard
// existed. Embedded names arrived with Fallout 3's v104.
constexpr std::uint32_t kFlagEmbedFileNames = 0x100u;
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

// v103/v104 on-disk layout. v105 widens `offset` to 64 bits behind 4 bytes of
// padding, so this is no longer a straight image of the file for every version
// -- see readFolderRecord.
struct BsaFolderRecordV104 {
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
static_assert(sizeof(BsaFolderRecordV104) == 16);
static_assert(sizeof(BsaRawFileRecord) == 16);

// A folder record as the rest of this reader wants it, independent of which
// on-disk layout it came from. `offset` is 64-bit because v105's is.
struct BsaFolderRecord {
    std::uint64_t nameHash = 0;
    std::uint32_t fileCount = 0;
    std::uint64_t offset = 0;
};

// Bytes one folder record occupies on disk. The ONLY structural difference
// between v105 and its predecessors, and the one that decides whether the
// folder table reads as paths or as noise.
constexpr std::size_t bsaFolderRecordSize(std::uint32_t version) {
    return version >= kBsaVersionSkyrimSe ? 24u : 16u;
}

constexpr bool isSupportedBsaVersion(std::uint32_t version) {
    return version == kBsaVersionOblivion || version == kBsaVersionFo3Fnv ||
           version == kBsaVersionSkyrimSe;
}

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

bool peekBsaContentFlags(const std::filesystem::path& path, std::uint32_t& outFileFlags) {
    outFileFlags = 0;
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        return false;
    }
    BsaHeader header{};
    if (!readValue(input, header)) {
        return false;
    }
    // Morrowind declares no content flags -- the format has none. Reporting
    // "everything" rather than zero, because callers use these to SKIP an
    // archive, and zero would skip the only archives Morrowind has. Getting
    // this wrong is silent: every mesh and texture simply fails to resolve,
    // which shades untextured rather than reporting a missing archive.
    if (header.magic == kMorrowindArchiveVersion) {
        outFileFlags = kBsaContentMeshes | kBsaContentTextures | kBsaContentMenus |
                       kBsaContentSounds | kBsaContentVoices | kBsaContentShaders |
                       kBsaContentTrees | kBsaContentFonts | kBsaContentMisc;
        return true;
    }
    if (header.magic != kBsaMagic || !isSupportedBsaVersion(header.version)) {
        return false;
    }
    outFileFlags = header.fileFlags;
    return true;
}

// Morrowind's archive, which is a different format that happens to share the
// .bsa extension. Layout, all little-endian, verified against Morrowind.bsa
// (11090 files) and Tribunal/Bloodmoon:
//
//   uint32 version = 0x100
//   uint32 hashTableOffset   -- from the END of this 12-byte header
//   uint32 fileCount
//   fileCount x { uint32 size, uint32 offset }   -- offset is from dataStart
//   fileCount x   uint32 nameOffset              -- into the name block
//   name block: NUL-terminated names, in no particular order
//   fileCount x { uint64 hash }
//   file data
//
// dataStart = 12 + hashTableOffset + fileCount * 8. The self-check that pins
// this is that the furthest-reaching file must end EXACTLY at the end of the
// archive, which it does for all three retail files -- an off-by-one in the
// base offset shifts every extraction and shows up as garbage rather than as
// a failure, so it is worth asserting rather than assuming.
//
// Nothing here is compressed, there are no folders, and there are no content
// flags. The name is the whole path, so the folder filter matches against it
// directly.
bool BsaArchive::openMorrowind(
    const std::filesystem::path& path, std::string_view folderPrefixFilter) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        m_lastError = "Failed to open Morrowind archive: " + path.string();
        return false;
    }
    std::uint32_t version = 0;
    std::uint32_t hashTableOffset = 0;
    std::uint32_t fileCount = 0;
    if (!readValue(input, version) || !readValue(input, hashTableOffset) ||
        !readValue(input, fileCount)) {
        m_lastError = "Truncated Morrowind archive header: " + path.string();
        return false;
    }
    m_version = kMorrowindArchiveVersion;
    m_archiveFlags = 0;
    // No content flags exist in this format. Reporting "everything" rather than
    // zero, because a caller uses these to SKIP archives and zero would skip
    // the only archives Morrowind has.
    m_fileFlags = kBsaContentMeshes | kBsaContentTextures | kBsaContentMenus |
                  kBsaContentSounds | kBsaContentVoices | kBsaContentShaders |
                  kBsaContentTrees | kBsaContentFonts | kBsaContentMisc;

    constexpr std::uint32_t kHeaderSize = 12u;
    constexpr std::uint32_t kSizeOffsetPairBytes = 8u;
    constexpr std::uint32_t kNameOffsetBytes = 4u;
    constexpr std::uint32_t kHashBytes = 8u;
    if (fileCount == 0u) {
        return true;  // an empty archive is legal and indexes to nothing
    }
    if (hashTableOffset < fileCount * (kSizeOffsetPairBytes + kNameOffsetBytes)) {
        m_lastError = "Morrowind archive hash offset overlaps its own tables: " + path.string();
        return false;
    }

    std::vector<std::uint32_t> sizes(fileCount);
    std::vector<std::uint32_t> offsets(fileCount);
    for (std::uint32_t i = 0; i < fileCount; ++i) {
        if (!readValue(input, sizes[i]) || !readValue(input, offsets[i])) {
            m_lastError = "Truncated Morrowind archive file table: " + path.string();
            return false;
        }
    }
    std::vector<std::uint32_t> nameOffsets(fileCount);
    for (std::uint32_t i = 0; i < fileCount; ++i) {
        if (!readValue(input, nameOffsets[i])) {
            m_lastError = "Truncated Morrowind archive name-offset table: " + path.string();
            return false;
        }
    }

    const std::uint64_t nameBlockStart =
        static_cast<std::uint64_t>(kHeaderSize) + fileCount * (kSizeOffsetPairBytes + kNameOffsetBytes);
    const std::uint64_t hashTableStart = static_cast<std::uint64_t>(kHeaderSize) + hashTableOffset;
    const std::uint64_t nameBlockBytes = hashTableStart - nameBlockStart;
    std::vector<char> nameBlock(static_cast<std::size_t>(nameBlockBytes));
    input.seekg(static_cast<std::streamoff>(nameBlockStart), std::ios::beg);
    if (nameBlockBytes != 0u &&
        !input.read(nameBlock.data(), static_cast<std::streamsize>(nameBlockBytes))) {
        m_lastError = "Truncated Morrowind archive name block: " + path.string();
        return false;
    }
    // Names are indexed, not sequential, so the whole block is read once and
    // sliced rather than streamed -- unlike every later BSA, where the name
    // order matches the file order and a filtered-out name still has to be
    // stepped past.
    const std::uint64_t dataStart = hashTableStart + static_cast<std::uint64_t>(fileCount) * kHashBytes;

    const std::string loweredFolderPrefix = toLowerAscii(std::string(folderPrefixFilter));
    m_files.reserve(fileCount);
    for (std::uint32_t i = 0; i < fileCount; ++i) {
        if (nameOffsets[i] >= nameBlockBytes) {
            m_lastError = "Morrowind archive name offset runs past the name block: " + path.string();
            return false;
        }
        const char* nameStart = nameBlock.data() + nameOffsets[i];
        const std::size_t maxLength = static_cast<std::size_t>(nameBlockBytes - nameOffsets[i]);
        const std::size_t nameLength = ::strnlen(nameStart, maxLength);
        std::string virtualPath =
            toLowerAscii(normalizeSeparators(std::string(nameStart, nameLength)));
        if (!loweredFolderPrefix.empty() &&
            virtualPath.compare(0, loweredFolderPrefix.size(), loweredFolderPrefix) != 0) {
            continue;
        }
        const std::uint64_t absoluteOffset = dataStart + offsets[i];
        BsaFileEntry entry{};
        entry.virtualPath = std::move(virtualPath);
        entry.dataOffset = static_cast<std::uint32_t>(absoluteOffset);
        entry.sizeOnDisk = sizes[i];
        entry.compressed = false;
        m_pathIndex.emplace(entry.virtualPath, m_files.size());
        m_files.push_back(std::move(entry));
    }
    return true;
}

bool BsaArchive::open(const std::filesystem::path& path, std::string_view folderPrefixFilter) {
    m_lastError.clear();
    m_files.clear();
    m_pathIndex.clear();
    m_path = path;
    m_archiveFlags = 0;

    std::ifstream input(path, std::ios::binary);
    if (!input) {
        m_lastError = "Failed to open BSA archive: " + path.string();
        return false;
    }

    BsaHeader header{};
    if (!readValue(input, header)) {
        m_lastError = "Truncated BSA header: " + path.string();
        return false;
    }
    // Morrowind's archive has no magic to check -- its first word IS the
    // version, and it is 0x100. Dispatched before the magic test because
    // "bad magic" is the wrong thing to say about a file that never had any.
    if (header.magic == kMorrowindArchiveVersion) {
        input.close();
        return openMorrowind(path, folderPrefixFilter);
    }
    if (header.magic != kBsaMagic) {
        m_lastError = "Not a BSA archive (bad magic): " + path.string();
        return false;
    }
    if (!isSupportedBsaVersion(header.version)) {
        m_lastError = "Unsupported BSA version " + std::to_string(header.version) +
            " (Oblivion v103, Fallout 3 / New Vegas v104 and Skyrim SE v105 are supported): " +
            path.string();
        return false;
    }

    // Mask the embed-file-names bit out of the flags this object keeps, rather
    // than testing the version again down in extract(). Nothing else in the
    // reader branches on the version, so this keeps the v103 difference in one
    // place and leaves extract() reading exactly the code path it always did.
    m_archiveFlags = header.archiveFlags;
    if (header.version == kBsaVersionOblivion) {
        m_archiveFlags &= ~kFlagEmbedFileNames;
    }
    m_fileFlags = header.fileFlags;
    m_version = header.version;
    const bool hasFolderNames = (header.archiveFlags & kFlagHasFolderNames) != 0u;
    const bool hasFileNames = (header.archiveFlags & kFlagHasFileNames) != 0u;
    const bool defaultCompressed = (header.archiveFlags & kFlagCompressedArchive) != 0u;

    input.seekg(static_cast<std::streamoff>(header.folderRecordOffset), std::ios::beg);
    std::vector<BsaFolderRecord> folders(header.folderCount);
    for (BsaFolderRecord& folder : folders) {
        if (header.version >= kBsaVersionSkyrimSe) {
            // hash, fileCount, 4 bytes of padding, then a 64-bit offset.
            std::uint32_t padding = 0;
            if (!readValue(input, folder.nameHash) || !readValue(input, folder.fileCount) ||
                !readValue(input, padding) || !readValue(input, folder.offset)) {
                m_lastError = "Truncated BSA folder record table: " + path.string();
                return false;
            }
        } else {
            BsaFolderRecordV104 raw{};
            if (!readValue(input, raw)) {
                m_lastError = "Truncated BSA folder record table: " + path.string();
                return false;
            }
            folder.nameHash = raw.nameHash;
            folder.fileCount = raw.fileCount;
            folder.offset = raw.offset;
        }
    }

    struct PendingFile {
        std::string folderPath;
        BsaRawFileRecord raw;
        // Decided once per FOLDER, carried per file so the name-block pass can
        // read past a filtered-out name without rebuilding the decision.
        bool keep = true;
    };
    const std::string loweredFolderPrefix = toLowerAscii(std::string(folderPrefixFilter));
    std::vector<PendingFile> pending;
    pending.reserve(header.fileCount);

    for (const BsaFolderRecord& folder : folders) {
        // Folder-record offsets are biased by totalFileNameLength (see the
        // layout notes in bsa_archive.h). Compute in 64-bit signed so a
        // malformed archive underflows into a detectable negative rather than
        // wrapping to a huge unsigned seek.
        const std::int64_t folderBlockOffset =
            static_cast<std::int64_t>(folder.offset) - static_cast<std::int64_t>(header.totalFileNameLength);
        if (folderBlockOffset < 0) {
            m_lastError = "BSA folder record offset underflows the file-name-length bias: " + path.string();
            return false;
        }
        input.seekg(static_cast<std::streamoff>(folderBlockOffset), std::ios::beg);
        std::string folderName;
        if (hasFolderNames && !readBString(input, folderName)) {
            m_lastError = "Truncated BSA folder name: " + path.string();
            return false;
        }
        // An archive without folder names cannot be filtered by folder, so it
        // keeps everything rather than silently indexing nothing.
        const bool folderKept = loweredFolderPrefix.empty() || !hasFolderNames ||
            toLowerAscii(folderName).compare(
                0, loweredFolderPrefix.size(), loweredFolderPrefix) == 0;
        for (std::uint32_t fileIndex = 0; fileIndex < folder.fileCount; ++fileIndex) {
            PendingFile file{};
            file.folderPath = folderName;
            file.keep = folderKept;
            if (!readValue(input, file.raw)) {
                m_lastError = "Truncated BSA file record table: " + path.string();
                return false;
            }
            pending.push_back(std::move(file));
        }
    }

    if (hasFileNames) {
        // Seek the file-name block explicitly rather than relying on wherever
        // the folder walk above happened to leave the stream. The two agree
        // for retail archives (verified against Meshes/Textures/Update/Voices1),
        // but only because folder blocks happen to be contiguous and in table
        // order — a layout the format does not actually guarantee.
        // totalFolderNameLength counts the name bytes and their NULs but not
        // the per-folder BString length byte, hence the `+ folderCount`.
        const std::uint64_t nameBlockOffset =
            static_cast<std::uint64_t>(header.folderRecordOffset) +
            (static_cast<std::uint64_t>(header.folderCount) * bsaFolderRecordSize(header.version)) +
            header.totalFolderNameLength + header.folderCount +
            (static_cast<std::uint64_t>(header.fileCount) * sizeof(BsaRawFileRecord));
        input.seekg(static_cast<std::streamoff>(nameBlockOffset), std::ios::beg);

        for (PendingFile& file : pending) {
            std::string fileName;
            if (!readNulTerminated(input, fileName)) {
                m_lastError = "Truncated BSA file name block: " + path.string();
                return false;
            }
            if (!file.keep) {
                continue;  // name consumed to keep the stream aligned, nothing kept
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
            if (!file.keep) {
                continue;
            }
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

bool BsaArchive::extract(
    const BsaFileEntry& entry,
    std::vector<std::uint8_t>& outBytes,
    std::string& outError) const {
    outError.clear();
    std::ifstream input(m_path, std::ios::binary);
    if (!input) {
        outError = "Failed to reopen BSA archive: " + m_path.string();
        return false;
    }
    input.seekg(static_cast<std::streamoff>(entry.dataOffset), std::ios::beg);

    // When kFlagEmbedFileNames is set the block opens with a BString holding
    // the file's full virtual path, ahead of both the compressed-size prefix
    // and the payload. Its bytes are counted in the file record's size, so
    // they come off the remaining budget. Both retail texture archives set
    // this; reading past it is what made every texture extraction garbage.
    std::uint32_t remainingSize = entry.sizeOnDisk;
    if (m_version >= kBsaVersionFo3Fnv && (m_archiveFlags & kFlagEmbedFileNames) != 0u) {
        std::uint8_t embeddedNameLength = 0;
        if (!readValue(input, embeddedNameLength)) {
            outError = "Truncated embedded BSA file name: " + entry.virtualPath;
            return false;
        }
        const std::uint32_t embeddedBytes = 1u + embeddedNameLength;
        if (remainingSize < embeddedBytes) {
            outError = "Embedded BSA file name overruns the entry size: " + entry.virtualPath;
            return false;
        }
        input.seekg(static_cast<std::streamoff>(embeddedNameLength), std::ios::cur);
        remainingSize -= embeddedBytes;
    }

    if (!entry.compressed) {
        outBytes.resize(remainingSize);
        if (remainingSize != 0 &&
            !input.read(reinterpret_cast<char*>(outBytes.data()), static_cast<std::streamsize>(remainingSize))) {
            outError = "Truncated BSA file data: " + entry.virtualPath;
            return false;
        }
        return true;
    }

    if (remainingSize < sizeof(std::uint32_t)) {
        outError = "Truncated compressed BSA entry header: " + entry.virtualPath;
        return false;
    }
    std::uint32_t originalSize = 0;
    if (!readValue(input, originalSize)) {
        outError = "Truncated compressed BSA entry header: " + entry.virtualPath;
        return false;
    }
    const std::uint32_t compressedSize = remainingSize - static_cast<std::uint32_t>(sizeof(originalSize));
    std::vector<std::uint8_t> compressed(compressedSize);
    if (compressedSize != 0 &&
        !input.read(reinterpret_cast<char*>(compressed.data()), static_cast<std::streamsize>(compressedSize))) {
        outError = "Truncated compressed BSA data: " + entry.virtualPath;
        return false;
    }

    // Skyrim SE swapped the codec, and it is the payload that says so rather
    // than the version alone: sniffing the frame magic means a v105 archive
    // that still holds a zlib entry -- or a mod-built archive that mislabels
    // itself -- decodes on what it actually contains.
    if (isLz4Frame(compressed.data(), compressed.size())) {
        std::string lz4Error;
        if (!lz4FrameDecompress(
                compressed.data(), compressed.size(), originalSize, outBytes, lz4Error)) {
            outError = lz4Error + " for BSA entry: " + entry.virtualPath;
            return false;
        }
        return true;
    }

    outBytes.resize(originalSize);
    uLongf destLen = static_cast<uLongf>(originalSize);
    const int result = uncompress(
        outBytes.empty() ? nullptr : reinterpret_cast<Bytef*>(outBytes.data()),
        &destLen,
        compressed.empty() ? nullptr : reinterpret_cast<const Bytef*>(compressed.data()),
        static_cast<uLong>(compressed.size()));
    if (result != Z_OK || destLen != originalSize) {
        outError = "zlib inflate failed for BSA entry: " + entry.virtualPath;
        return false;
    }
    return true;
}

}  // namespace odai::importer::fnv
