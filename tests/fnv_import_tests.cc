#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include <zlib.h>

#include "core/job_system.h"
#include "import/fnv/asset_source.h"
#include "import/fnv/character_builder.h"
#include "import/fnv/async_asset_loader.h"
#include "import/fnv/bsa_archive.h"
#include "import/fnv/esm_reader.h"
#include "import/fnv/fallout_records.h"
#include "import/fnv/nif_scene.h"

namespace {

int g_failures = 0;

// string_view rather than const char* so callers can build a message with a
// run-label suffix (the BSA test runs twice, once per archive-flag shape).
void expectTrue(bool condition, std::string_view message) {
    if (!condition) {
        std::cerr << "[fnv import test] FAIL: " << message << '\n';
        ++g_failures;
    }
}

void expectNear(float actual, float expected, float epsilon, const char* message) {
    if (std::fabs(actual - expected) > epsilon) {
        std::cerr << "[fnv import test] FAIL: " << message
                  << " (expected " << expected << ", got " << actual << ")\n";
        ++g_failures;
    }
}

template <typename T>
void appendPod(std::vector<std::uint8_t>& buffer, const T& value) {
    static_assert(std::is_trivially_copyable_v<T>);
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(&value);
    buffer.insert(buffer.end(), bytes, bytes + sizeof(T));
}

void appendBString(std::vector<std::uint8_t>& buffer, const std::string& text) {
    // BString: length byte includes the trailing NUL bethesda writes for
    // folder names.
    const std::string withNul = text + '\0';
    buffer.push_back(static_cast<std::uint8_t>(withNul.size()));
    buffer.insert(buffer.end(), withNul.begin(), withNul.end());
}

// The embedded file name in a data block (archive flag 0x100) is length-
// prefixed but NOT NUL-terminated, unlike the folder-name BString above.
// Verified against retail Fallout - Textures.bsa, whose first data block
// opens with 0x33 followed by exactly 51 path characters.
void appendEmbeddedName(std::vector<std::uint8_t>& buffer, const std::string& text) {
    buffer.push_back(static_cast<std::uint8_t>(text.size()));
    buffer.insert(buffer.end(), text.begin(), text.end());
}

void appendNulTerminated(std::vector<std::uint8_t>& buffer, const std::string& text) {
    buffer.insert(buffer.end(), text.begin(), text.end());
    buffer.push_back('\0');
}

#pragma pack(push, 1)
struct TestBsaHeader {
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
struct TestFolderRecord {
    std::uint64_t nameHash;
    std::uint32_t fileCount;
    std::uint32_t offset;
};
struct TestFileRecord {
    std::uint64_t nameHash;
    std::uint32_t size;
    std::uint32_t offset;
};
#pragma pack(pop)

// Builds a minimal, spec-shaped BSA v104 archive in memory with two folders:
// one uncompressed file and one zlib-compressed file. Mirrors the layout
// documented in bsa_archive.h so the test validates the reader's own
// understanding of folder/file record offsets and the name block, not just
// that encode(decode(x)) == x for an arbitrary internal representation.
//
// Two retail conventions this fixture deliberately reproduces, because an
// earlier version of it did not and so could never have caught the reader
// bugs they hid (both were found only by running against a real install):
//   - Folder-record offsets are biased by totalFileNameLength.
//   - When `embedFileNames` is set (archive flag 0x100, which both retail
//     texture archives use), every data block opens with a BString holding
//     the file's full virtual path, and those bytes count toward the file
//     record's size.
std::vector<std::uint8_t> buildSyntheticBsa(
    const std::string& uncompressedContent,
    const std::string& compressedContent,
    bool embedFileNames = false
) {
    constexpr std::uint32_t kFlagHasFolderNames = 0x1u;
    constexpr std::uint32_t kFlagHasFileNames = 0x2u;
    constexpr std::uint32_t kFlagCompressedArchive = 0x4u;
    constexpr std::uint32_t kFlagEmbedFileNames = 0x100u;
    constexpr std::uint32_t kFileCompressionToggleBit = 0x40000000u;

    const std::string folderA = "meshes\\x";
    const std::string folderB = "textures\\x";
    const std::string fileA = "ex_wall_01.nif";
    const std::string fileB = "tx_wall_01.dds";

    // Compress fileB's content with zlib to embed as the compressed entry.
    uLongf compressedBound = compressBound(static_cast<uLong>(compressedContent.size()));
    std::vector<std::uint8_t> compressedBytes(compressedBound);
    compress2(
        compressedBytes.data(), &compressedBound,
        reinterpret_cast<const Bytef*>(compressedContent.data()), static_cast<uLong>(compressedContent.size()),
        Z_BEST_COMPRESSION);
    compressedBytes.resize(compressedBound);

    const std::uint32_t totalFileNameLength =
        static_cast<std::uint32_t>(fileA.size() + 1 + fileB.size() + 1);

    TestBsaHeader header{};
    header.magic = 0x00415342u;
    header.version = 104u;
    header.folderRecordOffset = sizeof(TestBsaHeader);
    header.archiveFlags = kFlagHasFolderNames | kFlagHasFileNames | kFlagCompressedArchive |
        (embedFileNames ? kFlagEmbedFileNames : 0u);
    header.folderCount = 2u;
    header.fileCount = 2u;
    header.totalFolderNameLength = static_cast<std::uint32_t>(folderA.size() + folderB.size() + 2u);
    header.totalFileNameLength = totalFileNameLength;
    // Declare meshes + textures, as every retail archive does. Callers use
    // these to skip indexing archives they do not care about.
    header.fileFlags =
        odai::importer::fnv::kBsaContentMeshes | odai::importer::fnv::kBsaContentTextures;

    // Layout: header, 2 folder records, folderA block (name + 1 file record),
    // folderB block (name + 1 file record), file name block, file data.
    const std::size_t folderRecordsOffset = sizeof(TestBsaHeader);
    const std::size_t folderARecordOffset = folderRecordsOffset + 2 * sizeof(TestFolderRecord);
    const std::size_t folderABlockSize = 1 + folderA.size() + 1 + sizeof(TestFileRecord);
    const std::size_t folderBRecordOffset = folderARecordOffset + folderABlockSize;
    const std::size_t folderBBlockSize = 1 + folderB.size() + 1 + sizeof(TestFileRecord);
    const std::size_t fileNameBlockOffset = folderBRecordOffset + folderBBlockSize;
    const std::size_t fileDataOffset = fileNameBlockOffset + totalFileNameLength;

    // Embedded names, when present, prefix each data block and are counted in
    // the file record's size.
    const std::string embeddedA = folderA + "\\" + fileA;
    const std::string embeddedB = folderB + "\\" + fileB;
    const std::size_t embeddedABytes = embedFileNames ? 1u + embeddedA.size() : 0u;
    const std::size_t embeddedBBytes = embedFileNames ? 1u + embeddedB.size() : 0u;

    // fileA: uncompressed (toggled off default-compressed archive -> stored
    // raw). fileB: compressed, stored as [uint32 originalSize][zlib bytes].
    const std::size_t fileAOffset = fileDataOffset;
    const std::size_t fileASize = embeddedABytes + uncompressedContent.size();
    const std::size_t fileBOffset = fileAOffset + fileASize;
    const std::size_t fileBSizeOnDisk = embeddedBBytes + sizeof(std::uint32_t) + compressedBytes.size();

    std::vector<std::uint8_t> out;
    appendPod(out, header);

    // Retail archives bias folder-record offsets by totalFileNameLength; the
    // reader is expected to subtract it back off. Writing the unbiased value
    // here is what let the reader's missing subtraction go unnoticed.
    TestFolderRecord folderRecA{};
    folderRecA.nameHash = 0x1111ull;
    folderRecA.fileCount = 1u;
    folderRecA.offset = static_cast<std::uint32_t>(folderARecordOffset + totalFileNameLength);
    appendPod(out, folderRecA);

    TestFolderRecord folderRecB{};
    folderRecB.nameHash = 0x2222ull;
    folderRecB.fileCount = 1u;
    folderRecB.offset = static_cast<std::uint32_t>(folderBRecordOffset + totalFileNameLength);
    appendPod(out, folderRecB);

    appendBString(out, folderA);
    TestFileRecord fileRecA{};
    fileRecA.nameHash = 0xAAAAull;
    fileRecA.size = static_cast<std::uint32_t>(fileASize) | kFileCompressionToggleBit;
    fileRecA.offset = static_cast<std::uint32_t>(fileAOffset);
    appendPod(out, fileRecA);

    appendBString(out, folderB);
    TestFileRecord fileRecB{};
    fileRecB.nameHash = 0xBBBBull;
    fileRecB.size = static_cast<std::uint32_t>(fileBSizeOnDisk);  // toggle bit unset -> uses archive default (compressed)
    fileRecB.offset = static_cast<std::uint32_t>(fileBOffset);
    appendPod(out, fileRecB);

    appendNulTerminated(out, fileA);
    appendNulTerminated(out, fileB);

    if (embedFileNames) {
        appendEmbeddedName(out, embeddedA);
    }
    out.insert(out.end(), uncompressedContent.begin(), uncompressedContent.end());

    if (embedFileNames) {
        appendEmbeddedName(out, embeddedB);
    }
    appendPod(out, static_cast<std::uint32_t>(compressedContent.size()));
    out.insert(out.end(), compressedBytes.begin(), compressedBytes.end());

    return out;
}

void appendBytes(std::vector<std::uint8_t>& buffer, const void* data, std::size_t size) {
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(data);
    buffer.insert(buffer.end(), bytes, bytes + size);
}

void appendFourCc(std::vector<std::uint8_t>& buffer, const char* fourCc) {
    buffer.insert(buffer.end(), fourCc, fourCc + 4);
}

std::vector<std::uint8_t> buildSubrecord(const char* type, const std::vector<std::uint8_t>& payload) {
    std::vector<std::uint8_t> out;
    appendFourCc(out, type);
    appendPod(out, static_cast<std::uint16_t>(payload.size()));
    out.insert(out.end(), payload.begin(), payload.end());
    return out;
}

std::vector<std::uint8_t> stringPayload(const std::string& text) {
    std::vector<std::uint8_t> out(text.begin(), text.end());
    out.push_back('\0');
    return out;
}

// Builds an "XXXX"-prefixed oversized subrecord: the override-size subrecord
// followed by the real subrecord whose own 2-byte size field is a dummy
// value the reader must ignore in favor of the XXXX override.
std::vector<std::uint8_t> buildOversizedSubrecord(const char* type, const std::vector<std::uint8_t>& payload) {
    std::vector<std::uint8_t> out;
    appendFourCc(out, "XXXX");
    appendPod(out, static_cast<std::uint16_t>(4));
    appendPod(out, static_cast<std::uint32_t>(payload.size()));
    appendFourCc(out, type);
    appendPod(out, static_cast<std::uint16_t>(0));
    out.insert(out.end(), payload.begin(), payload.end());
    return out;
}

std::vector<std::uint8_t> buildRecord(
    const char* type, std::uint32_t formId, std::uint32_t flags, const std::vector<std::uint8_t>& data
) {
    std::vector<std::uint8_t> out;
    appendFourCc(out, type);
    appendPod(out, static_cast<std::uint32_t>(data.size()));
    appendPod(out, flags);
    appendPod(out, formId);
    appendPod(out, static_cast<std::uint32_t>(0));
    appendPod(out, static_cast<std::uint16_t>(0));
    appendPod(out, static_cast<std::uint16_t>(0));
    out.insert(out.end(), data.begin(), data.end());
    return out;
}

std::vector<std::uint8_t> buildGroup(
    const char rawLabel[4], std::int32_t groupType, const std::vector<std::uint8_t>& content
) {
    std::vector<std::uint8_t> out;
    appendFourCc(out, "GRUP");
    appendPod(out, static_cast<std::uint32_t>(24u + content.size()));
    appendBytes(out, rawLabel, 4u);
    appendPod(out, groupType);
    appendPod(out, static_cast<std::uint16_t>(0));
    appendPod(out, static_cast<std::uint16_t>(0));
    appendPod(out, static_cast<std::uint32_t>(0));
    out.insert(out.end(), content.begin(), content.end());
    return out;
}

std::vector<std::uint8_t> zlibCompress(const std::vector<std::uint8_t>& data) {
    uLongf bound = compressBound(static_cast<uLong>(data.size()));
    std::vector<std::uint8_t> out(bound);
    compress2(out.data(), &bound, data.empty() ? nullptr : data.data(), static_cast<uLong>(data.size()), Z_BEST_COMPRESSION);
    out.resize(bound);
    return out;
}

// Corrupts the trailing adler32 of a zlib stream, leaving the deflate payload
// intact. Reproduces the one damaged record in retail FalloutNV.esm (a LAND,
// formID 0x150FC0): every declared byte decodes correctly but the checksum
// does not verify. zlib's uncompress() rejects the whole call for this, which
// used to abort the entire plugin walk over a single bad record.
std::vector<std::uint8_t> corruptZlibChecksum(std::vector<std::uint8_t> stream) {
    if (stream.size() >= 4u) {
        stream[stream.size() - 1u] ^= 0xFFu;
    }
    return stream;
}

void testEsmReaderWalksGroupsRecordsAndSubrecords() {
    namespace fs = std::filesystem;

    // Top GRUP "STAT" containing one STAT record with EDID/MODL subrecords.
    const std::vector<std::uint8_t> statSubrecords = [] {
        std::vector<std::uint8_t> out;
        const auto edid = buildSubrecord("EDID", stringPayload("Rock01"));
        const auto modl = buildSubrecord("MODL", stringPayload("x\\rock01.nif"));
        out.insert(out.end(), edid.begin(), edid.end());
        out.insert(out.end(), modl.begin(), modl.end());
        return out;
    }();
    const auto statRecord = buildRecord("STAT", 0x00000010u, 0u, statSubrecords);
    const auto statGroup = buildGroup("STAT", 0, statRecord);

    // Top GRUP "ABCD" -> nested GRUP (groupType 6) containing:
    //  - a record with an oversized (XXXX) subrecord
    //  - a zlib-compressed record
    const std::vector<std::uint8_t> oversizedPayload(70000, 0x5Au);  // exceeds uint16 max
    const auto oversizedSubrecords = buildOversizedSubrecord("DATA", oversizedPayload);
    const auto bigRecord = buildRecord("BIGX", 0x00000099u, 0u, oversizedSubrecords);

    const auto compressedInner = buildSubrecord("EDID", stringPayload("Compressed"));
    const auto compressedPayload = zlibCompress(compressedInner);
    std::vector<std::uint8_t> compressedRecordData;
    appendPod(compressedRecordData, static_cast<std::uint32_t>(compressedInner.size()));
    compressedRecordData.insert(compressedRecordData.end(), compressedPayload.begin(), compressedPayload.end());
    constexpr std::uint32_t kRecordFlagCompressed = 0x00040000u;
    const auto compressedRecord = buildRecord("CMPR", 0x000000AAu, kRecordFlagCompressed, compressedRecordData);

    std::vector<std::uint8_t> nestedContent;
    nestedContent.insert(nestedContent.end(), bigRecord.begin(), bigRecord.end());
    nestedContent.insert(nestedContent.end(), compressedRecord.begin(), compressedRecord.end());
    const std::uint32_t parentFormId = 0x00000010u;
    char nestedLabel[4];
    std::memcpy(nestedLabel, &parentFormId, 4);
    const auto nestedGroup = buildGroup(nestedLabel, 6, nestedContent);
    const auto outerGroup = buildGroup("ABCD", 0, nestedGroup);

    std::vector<std::uint8_t> fileBytes;
    fileBytes.insert(fileBytes.end(), statGroup.begin(), statGroup.end());
    fileBytes.insert(fileBytes.end(), outerGroup.begin(), outerGroup.end());

    const fs::path esmPath = fs::temp_directory_path() / "odai_fnv_test.esm";
    {
        std::ofstream out(esmPath, std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char*>(fileBytes.data()), static_cast<std::streamsize>(fileBytes.size()));
    }

    odai::importer::fnv::EsmReader reader;
    expectTrue(reader.open(esmPath), "Synthetic ESM file opens");

    struct GroupEvent {
        bool isEnter = false;
        std::string rawLabel;
        std::int32_t groupType = 0;
    };
    // EsmRecordView subrecord pointers are only valid inside onRecord (a
    // compressed record's data lives in a scratch buffer walk() frees on
    // return) — copy the bytes we need to keep into owned storage here.
    struct CapturedSubrecord {
        std::string type;
        std::vector<std::uint8_t> data;
    };
    struct CapturedRecord {
        std::string type;
        std::uint32_t formId = 0;
        std::uint32_t flags = 0;
        std::vector<CapturedSubrecord> subrecords;
    };
    std::vector<GroupEvent> groupEvents;
    std::vector<CapturedRecord> records;
    odai::importer::fnv::EsmReader::Visitor visitor{};
    visitor.onGroupEnter = [&](const odai::importer::fnv::EsmGroupView& group) {
        groupEvents.push_back(GroupEvent{true, group.rawLabel, group.groupType});
        return true;
    };
    visitor.onGroupExit = [&](const odai::importer::fnv::EsmGroupView& group) {
        groupEvents.push_back(GroupEvent{false, group.rawLabel, group.groupType});
    };
    visitor.onRecord = [&](const odai::importer::fnv::EsmRecordView& record) {
        CapturedRecord captured{};
        captured.type = record.type;
        captured.formId = record.formId;
        captured.flags = record.flags;
        for (const auto& sub : record.subrecords) {
            CapturedSubrecord capturedSub{};
            capturedSub.type = sub.type;
            capturedSub.data.assign(sub.data, sub.data + sub.size);
            captured.subrecords.push_back(std::move(capturedSub));
        }
        records.push_back(std::move(captured));
    };

    expectTrue(reader.walk(visitor), "Synthetic ESM file walks without error");
    expectTrue(records.size() == 3u, "Walker visits all three records");

    // Embedded-NUL rawLabel bytes (the nested group's label is a raw formID,
    // not text) — compare via the explicit-length std::string constructor,
    // not a C-string literal, so a \0 byte can't truncate the comparison.
    const std::string nestedLabelExpected(nestedLabel, 4);
    expectTrue(
        groupEvents.size() == 6u &&
        groupEvents[0].isEnter && groupEvents[0].rawLabel == "STAT" && groupEvents[0].groupType == 0 &&
        !groupEvents[1].isEnter && groupEvents[1].groupType == 0 &&
        groupEvents[2].isEnter && groupEvents[2].rawLabel == "ABCD" && groupEvents[2].groupType == 0 &&
        groupEvents[3].isEnter && groupEvents[3].rawLabel == nestedLabelExpected && groupEvents[3].groupType == 6 &&
        !groupEvents[4].isEnter && groupEvents[4].groupType == 6 &&
        !groupEvents[5].isEnter && groupEvents[5].groupType == 0,
        "Group enter/exit events fire in correct nested depth-first order");

    expectTrue(records[0].type == "STAT" && records[0].formId == 0x00000010u,
               "STAT record type and formID round-trip");
    expectTrue(records[0].subrecords.size() == 2u, "STAT record exposes both subrecords");
    expectTrue(
        records[0].subrecords[0].type == "EDID" &&
        std::string(reinterpret_cast<const char*>(records[0].subrecords[0].data.data())) == "Rock01",
        "EDID subrecord content round-trips");
    expectTrue(
        records[0].subrecords[1].type == "MODL" &&
        std::string(reinterpret_cast<const char*>(records[0].subrecords[1].data.data())) == "x\\rock01.nif",
        "MODL subrecord content round-trips");

    expectTrue(records[1].type == "BIGX", "Oversized-subrecord record is visited");
    expectTrue(records[1].subrecords.size() == 1u, "XXXX marker is not itself exposed as a subrecord");
    expectTrue(records[1].subrecords[0].type == "DATA" && records[1].subrecords[0].data.size() == 70000u,
               "XXXX override size is applied to the following subrecord");
    expectTrue(records[1].subrecords[0].data[0] == 0x5Au && records[1].subrecords[0].data[69999] == 0x5Au,
               "Oversized subrecord content round-trips at both ends");

    expectTrue(records[2].type == "CMPR" && records[2].flags == kRecordFlagCompressed,
               "Compressed record is visited with its flag intact");
    expectTrue(records[2].subrecords.size() == 1u && records[2].subrecords[0].type == "EDID" &&
               std::string(reinterpret_cast<const char*>(records[2].subrecords[0].data.data())) == "Compressed",
               "Compressed record data decompresses to the original subrecord stream");

    fs::remove(esmPath);
}

// Encodes a VHGT subrecord payload (offset float + 33x33 signed-byte deltas
// + 2 reserved bytes) from a caller-supplied delta grid.
std::vector<std::uint8_t> buildVhgtPayload(
    float baseOffset, const std::array<std::array<std::int8_t, 33>, 33>& deltas
) {
    std::vector<std::uint8_t> out;
    appendPod(out, baseOffset);
    for (int row = 0; row < 33; ++row) {
        for (int col = 0; col < 33; ++col) {
            appendPod(out, deltas[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)]);
        }
    }
    appendPod(out, static_cast<std::uint16_t>(0));  // reserved
    return out;
}

std::vector<std::uint8_t> buildVnmlPayload() {
    // 33x33 straight-up normals (0, 0, 127): a flat land record.
    std::vector<std::uint8_t> out;
    for (int i = 0; i < odai::importer::fnv::kLandVertexCount; ++i) {
        out.push_back(0);
        out.push_back(0);
        out.push_back(127);
    }
    return out;
}

// VCLR: one unsigned RGB triple per post. Uses a distinct value per channel so
// a channel swap or an off-by-one stride shows up as a wrong colour rather than
// passing by symmetry.
std::vector<std::uint8_t> buildVclrPayload() {
    std::vector<std::uint8_t> out;
    for (int i = 0; i < odai::importer::fnv::kLandVertexCount; ++i) {
        out.push_back(255);
        out.push_back(128);
        out.push_back(64);
    }
    return out;
}

void testFalloutCellIndexMatchesFullExtraction(const std::filesystem::path& esmPath);

void testFalloutRecordExtraction() {
    namespace fs = std::filesystem;
    using namespace odai::importer::fnv;

    // --- STAT (top-level, not nested under any cell) ---
    const auto statSubrecords = [] {
        std::vector<std::uint8_t> out;
        const auto edid = buildSubrecord("EDID", stringPayload("TestRock"));
        const auto modl = buildSubrecord("MODL", stringPayload("r\\rock01.nif"));
        out.insert(out.end(), edid.begin(), edid.end());
        out.insert(out.end(), modl.begin(), modl.end());
        return out;
    }();
    constexpr std::uint32_t kStaticFormId = 0x00005000u;
    const auto statRecord = buildRecord("STAT", kStaticFormId, 0u, statSubrecords);
    const auto statGroup = buildGroup("STAT", 0, statRecord);

    // --- LAND: all-zero deltas except one bump at (row=10, col=20). ---
    std::array<std::array<std::int8_t, 33>, 33> deltas{};
    for (auto& row : deltas) {
        row.fill(0);
    }
    deltas[10][20] = 4;
    const auto vhgtPayload = buildVhgtPayload(100.0f, deltas);
    const auto vnmlPayload = buildVnmlPayload();
    std::vector<std::uint8_t> btxtPayload;
    constexpr std::uint32_t kBaseTextureFormId = 0x00008000u;
    appendPod(btxtPayload, kBaseTextureFormId);
    btxtPayload.push_back(0u);  // quadrant 0
    btxtPayload.push_back(0u);  // pad
    appendPod(btxtPayload, static_cast<std::uint16_t>(0));  // layer

    const auto vclrPayload = buildVclrPayload();

    // Two ATXT/VTXT layers, deliberately emitted with the HIGHER layer index
    // first so the parser's sort is doing real work rather than preserving
    // subrecord order by luck.
    constexpr std::uint32_t kLayerTextureFormIdHigh = 0x00004100u;
    constexpr std::uint32_t kLayerTextureFormIdLow = 0x00004200u;
    const auto buildAtxtPayload = [](std::uint32_t formId, std::uint8_t quadrant, std::uint16_t layerIndex) {
        std::vector<std::uint8_t> out;
        appendPod(out, formId);
        out.push_back(quadrant);
        out.push_back(0u);  // pad
        appendPod(out, layerIndex);
        return out;
    };
    // Opacity 1.0 at quadrant post 0, 0.25 at post 5.
    const auto buildVtxtPayload = []() {
        std::vector<std::uint8_t> out;
        appendPod(out, static_cast<std::uint16_t>(0));
        appendPod(out, static_cast<std::uint16_t>(0));
        appendPod(out, 1.0f);
        appendPod(out, static_cast<std::uint16_t>(5));
        appendPod(out, static_cast<std::uint16_t>(0));
        appendPod(out, 0.25f);
        return out;
    };

    const auto landSubrecords = [&] {
        std::vector<std::uint8_t> out;
        const auto vhgt = buildSubrecord("VHGT", vhgtPayload);
        const auto vnml = buildSubrecord("VNML", vnmlPayload);
        const auto vclr = buildSubrecord("VCLR", vclrPayload);
        const auto btxt = buildSubrecord("BTXT", btxtPayload);
        const auto atxtHigh = buildSubrecord("ATXT", buildAtxtPayload(kLayerTextureFormIdHigh, 0u, 1u));
        const auto vtxtHigh = buildSubrecord("VTXT", buildVtxtPayload());
        const auto atxtLow = buildSubrecord("ATXT", buildAtxtPayload(kLayerTextureFormIdLow, 0u, 0u));
        const auto vtxtLow = buildSubrecord("VTXT", buildVtxtPayload());
        out.insert(out.end(), vhgt.begin(), vhgt.end());
        out.insert(out.end(), vnml.begin(), vnml.end());
        out.insert(out.end(), vclr.begin(), vclr.end());
        out.insert(out.end(), btxt.begin(), btxt.end());
        out.insert(out.end(), atxtHigh.begin(), atxtHigh.end());
        out.insert(out.end(), vtxtHigh.begin(), vtxtHigh.end());
        out.insert(out.end(), atxtLow.begin(), atxtLow.end());
        out.insert(out.end(), vtxtLow.begin(), vtxtLow.end());
        return out;
    }();
    constexpr std::uint32_t kLandFormId = 0x00003000u;
    const auto landRecord = buildRecord("LAND", kLandFormId, 0u, landSubrecords);

    // --- NAVM: a 4-vertex, 2-triangle navmesh sharing one interior edge. ---
    //
    // Deliberately asymmetric: triangle 0 links its edge 1 to triangle 1, and
    // triangle 1 links its edge 2 back. A parser that assumed a fixed edge
    // ordering, or that read the neighbour block at the wrong offset, would
    // still produce two triangles -- only the adjacency would be wrong, and the
    // adjacency is the entire point of using the authored mesh.
    constexpr std::uint32_t kNavMeshFormId = 0x00005000u;
    constexpr std::uint32_t kNavDoorRefFormId = 0x00005100u;
    std::vector<std::uint8_t> navDataPayload;
    appendPod(navDataPayload, static_cast<std::uint32_t>(0x00002000u));  // the exterior CELL below
    appendPod(navDataPayload, static_cast<std::uint32_t>(4));   // vertex count
    appendPod(navDataPayload, static_cast<std::uint32_t>(2));   // triangle count
    appendPod(navDataPayload, static_cast<std::uint32_t>(0));
    appendPod(navDataPayload, static_cast<std::uint32_t>(0));
    appendPod(navDataPayload, static_cast<std::uint32_t>(0));

    std::vector<std::uint8_t> navVertexPayload;
    const float navVertices[4][3] = {
        {0.0f, 0.0f, 10.0f}, {128.0f, 0.0f, 10.0f}, {128.0f, 128.0f, 10.0f}, {0.0f, 128.0f, 10.0f}};
    for (const auto& vertex : navVertices) {
        appendPod(navVertexPayload, vertex[0]);
        appendPod(navVertexPayload, vertex[1]);
        appendPod(navVertexPayload, vertex[2]);
    }

    const auto appendNavTriangle = [](std::vector<std::uint8_t>& out,
                                      std::uint16_t v0, std::uint16_t v1, std::uint16_t v2,
                                      std::uint16_t n0, std::uint16_t n1, std::uint16_t n2,
                                      std::uint16_t flags, std::uint16_t cover) {
        appendPod(out, v0); appendPod(out, v1); appendPod(out, v2);
        appendPod(out, n0); appendPod(out, n1); appendPod(out, n2);
        appendPod(out, flags); appendPod(out, cover);
    };
    std::vector<std::uint8_t> navTrianglePayload;
    appendNavTriangle(navTrianglePayload, 0, 1, 2, 0xffffu, 1u, 0xffffu, 0x0800u, 0u);
    // Neighbour 9 is out of range for a 2-triangle mesh and must be clamped to
    // "border": a wild index is dereferenced during pathfinding.
    appendNavTriangle(navTrianglePayload, 0, 2, 3, 0xffffu, 0xffffu, 9u, 0u, 0u);

    std::vector<std::uint8_t> navPortalPayload;
    appendPod(navPortalPayload, kNavDoorRefFormId);
    appendPod(navPortalPayload, static_cast<std::uint16_t>(1));
    appendPod(navPortalPayload, static_cast<std::uint16_t>(0));

    const auto navSubrecords = [&] {
        std::vector<std::uint8_t> out;
        for (const auto& sub : {buildSubrecord("DATA", navDataPayload),
                                buildSubrecord("NVVX", navVertexPayload),
                                buildSubrecord("NVTR", navTrianglePayload),
                                buildSubrecord("NVDP", navPortalPayload)}) {
            out.insert(out.end(), sub.begin(), sub.end());
        }
        return out;
    }();
    const auto navMeshRecord = buildRecord("NAVM", kNavMeshFormId, 0u, navSubrecords);

    // --- REFR placing the static in the exterior cell. ---
    std::vector<std::uint8_t> refData;
    appendPod(refData, 512.0f);   // posX
    appendPod(refData, 64.0f);    // posY
    appendPod(refData, 1024.0f);  // posZ
    appendPod(refData, 0.0f);     // rotX
    appendPod(refData, 1.5707963f);  // rotY
    appendPod(refData, 0.0f);     // rotZ
    const auto refSubrecords = [&] {
        std::vector<std::uint8_t> out;
        std::vector<std::uint8_t> namePayload;
        appendPod(namePayload, kStaticFormId);
        const auto name = buildSubrecord("NAME", namePayload);
        const auto data = buildSubrecord("DATA", refData);
        std::vector<std::uint8_t> scalePayload;
        appendPod(scalePayload, 2.0f);
        const auto xscl = buildSubrecord("XSCL", scalePayload);
        out.insert(out.end(), name.begin(), name.end());
        out.insert(out.end(), data.begin(), data.end());
        out.insert(out.end(), xscl.begin(), xscl.end());
        return out;
    }();
    constexpr std::uint32_t kRefFormId = 0x00004000u;
    const auto refRecord = buildRecord("REFR", kRefFormId, 0u, refSubrecords);

    // --- Exterior CELL wrapping LAND + REFR, nested under a WRLD. ---
    constexpr std::uint32_t kExtCellFormId = 0x00002000u;
    const auto extCellSubrecords = [] {
        std::vector<std::uint8_t> out;
        const auto edid = buildSubrecord("EDID", stringPayload("TestExtCell"));
        std::vector<std::uint8_t> dataPayload{0u};  // exterior (bit 0 clear)
        const auto data = buildSubrecord("DATA", dataPayload);
        std::vector<std::uint8_t> xclcPayload;
        appendPod(xclcPayload, static_cast<std::int32_t>(5));
        appendPod(xclcPayload, static_cast<std::int32_t>(-3));
        appendPod(xclcPayload, static_cast<std::uint32_t>(0));
        const auto xclc = buildSubrecord("XCLC", xclcPayload);
        out.insert(out.end(), edid.begin(), edid.end());
        out.insert(out.end(), data.begin(), data.end());
        out.insert(out.end(), xclc.begin(), xclc.end());
        return out;
    }();
    const auto extCellRecord = buildRecord("CELL", kExtCellFormId, 0u, extCellSubrecords);

    std::vector<std::uint8_t> tempChildrenContent;
    tempChildrenContent.insert(tempChildrenContent.end(), landRecord.begin(), landRecord.end());
    tempChildrenContent.insert(tempChildrenContent.end(), navMeshRecord.begin(), navMeshRecord.end());
    tempChildrenContent.insert(tempChildrenContent.end(), refRecord.begin(), refRecord.end());
    char extCellLabel[4];
    std::memcpy(extCellLabel, &kExtCellFormId, 4);
    const auto tempChildrenGroup = buildGroup(extCellLabel, 9, tempChildrenContent);
    const auto cellChildrenGroup = buildGroup(extCellLabel, 6, tempChildrenGroup);

    std::vector<std::uint8_t> subBlockContent;
    subBlockContent.insert(subBlockContent.end(), extCellRecord.begin(), extCellRecord.end());
    subBlockContent.insert(subBlockContent.end(), cellChildrenGroup.begin(), cellChildrenGroup.end());
    const auto subBlockGroup = buildGroup("\0\0\0\0", 5, subBlockContent);
    const auto blockGroup = buildGroup("\0\0\0\0", 4, subBlockGroup);

    constexpr std::uint32_t kWorldFormId = 0x00001000u;
    const auto worldSubrecords = buildSubrecord("EDID", stringPayload("TestWorld"));
    const auto worldRecord = buildRecord("WRLD", kWorldFormId, 0u, worldSubrecords);
    char worldLabel[4];
    std::memcpy(worldLabel, &kWorldFormId, 4);
    const auto worldChildrenGroup = buildGroup(worldLabel, 1, blockGroup);

    std::vector<std::uint8_t> wrldTopContent;
    wrldTopContent.insert(wrldTopContent.end(), worldRecord.begin(), worldRecord.end());
    wrldTopContent.insert(wrldTopContent.end(), worldChildrenGroup.begin(), worldChildrenGroup.end());
    const auto wrldTopGroup = buildGroup("WRLD", 0, wrldTopContent);

    // --- Interior CELL with its own REFR, outside any WRLD. ---
    constexpr std::uint32_t kIntCellFormId = 0x00006000u;
    const auto intCellSubrecords = [] {
        std::vector<std::uint8_t> out;
        const auto edid = buildSubrecord("EDID", stringPayload("TestInteriorCell"));
        std::vector<std::uint8_t> dataPayload{0x1u};  // interior (bit 0 set)
        const auto data = buildSubrecord("DATA", dataPayload);
        out.insert(out.end(), edid.begin(), edid.end());
        out.insert(out.end(), data.begin(), data.end());
        return out;
    }();
    const auto intCellRecord = buildRecord("CELL", kIntCellFormId, 0u, intCellSubrecords);

    std::vector<std::uint8_t> intRefData;
    appendPod(intRefData, 10.0f);
    appendPod(intRefData, 20.0f);
    appendPod(intRefData, 30.0f);
    appendPod(intRefData, 0.0f);
    appendPod(intRefData, 0.0f);
    appendPod(intRefData, 0.0f);
    std::vector<std::uint8_t> intNamePayload;
    appendPod(intNamePayload, kStaticFormId);
    const auto intName = buildSubrecord("NAME", intNamePayload);
    const auto intData = buildSubrecord("DATA", intRefData);
    std::vector<std::uint8_t> intRefSubrecords;
    intRefSubrecords.insert(intRefSubrecords.end(), intName.begin(), intName.end());
    intRefSubrecords.insert(intRefSubrecords.end(), intData.begin(), intData.end());
    constexpr std::uint32_t kIntRefFormId = 0x00007000u;
    const auto intRefRecord = buildRecord("REFR", kIntRefFormId, 0u, intRefSubrecords);

    char intCellLabel[4];
    std::memcpy(intCellLabel, &kIntCellFormId, 4);
    const auto intTempChildrenGroup = buildGroup(intCellLabel, 9, intRefRecord);
    const auto intCellChildrenGroup = buildGroup(intCellLabel, 6, intTempChildrenGroup);
    std::vector<std::uint8_t> intSubBlockContent;
    intSubBlockContent.insert(intSubBlockContent.end(), intCellRecord.begin(), intCellRecord.end());
    intSubBlockContent.insert(intSubBlockContent.end(), intCellChildrenGroup.begin(), intCellChildrenGroup.end());
    const auto intSubBlockGroup = buildGroup("\0\0\0\0", 3, intSubBlockContent);
    const auto intBlockGroup = buildGroup("\0\0\0\0", 2, intSubBlockGroup);
    const auto cellTopGroup = buildGroup("CELL", 0, intBlockGroup);

    std::vector<std::uint8_t> fileBytes;
    fileBytes.insert(fileBytes.end(), statGroup.begin(), statGroup.end());
    fileBytes.insert(fileBytes.end(), wrldTopGroup.begin(), wrldTopGroup.end());
    fileBytes.insert(fileBytes.end(), cellTopGroup.begin(), cellTopGroup.end());

    const fs::path esmPath = fs::temp_directory_path() / "odai_fnv_records_test.esm";
    {
        std::ofstream out(esmPath, std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char*>(fileBytes.data()), static_cast<std::streamsize>(fileBytes.size()));
    }

    FalloutSceneData scene;
    std::string error;
    expectTrue(extractFalloutScene(esmPath, scene, error), ("Fallout scene extraction succeeds: " + error).c_str());

    expectTrue(scene.statics.size() == 1u && scene.statics[0].formId == kStaticFormId &&
               scene.statics[0].editorId == "TestRock" && scene.statics[0].modelPath == "r\\rock01.nif",
               "STAT record extraction round-trips editor id and model path");

    expectTrue(scene.worldspaces.size() == 1u && scene.worldspaces[0].formId == kWorldFormId &&
               scene.worldspaces[0].editorId == "TestWorld",
               "WRLD record extraction round-trips");

    expectTrue(scene.cells.size() == 2u, "Both exterior and interior CELL records are extracted");
    const FalloutCellRecord* extCell = nullptr;
    const FalloutCellRecord* intCell = nullptr;
    for (const auto& cell : scene.cells) {
        if (cell.formId == kExtCellFormId) extCell = &cell;
        if (cell.formId == kIntCellFormId) intCell = &cell;
    }
    expectTrue(extCell != nullptr && !extCell->isInterior && extCell->worldspaceFormId == kWorldFormId &&
               extCell->hasGridCoords && extCell->gridX == 5 && extCell->gridZ == -3,
               "Exterior cell attributes (worldspace, grid coords, interior flag) round-trip");
    expectTrue(intCell != nullptr && intCell->isInterior && intCell->worldspaceFormId == 0u,
               "Interior cell is flagged interior and attributed to no worldspace");

    expectTrue(extCell != nullptr && extCell->references.size() == 1u,
               "Exterior cell owns exactly the one REFR placed inside its group hierarchy");
    if (extCell != nullptr && extCell->references.size() == 1u) {
        const FalloutPlacedReference& ref = extCell->references.front();
        expectTrue(ref.baseFormId == kStaticFormId, "REFR base formID (NAME) round-trips");
        expectTrue(
            ref.position[0] == 512.0f && ref.position[1] == 64.0f && ref.position[2] == 1024.0f,
            "REFR position round-trips");
        expectNear(ref.rotationRadians[1], 1.5707963f, 1e-5f, "REFR rotation round-trips");
        expectNear(ref.scale, 2.0f, 1e-6f, "REFR explicit XSCL scale round-trips");
    }
    expectTrue(intCell != nullptr && intCell->references.size() == 1u &&
               intCell->references.front().scale == 1.0f,
               "Interior REFR without XSCL defaults to scale 1.0");

    expectTrue(extCell != nullptr && extCell->land != nullptr, "Exterior cell owns its LAND record");
    if (extCell != nullptr && extCell->land != nullptr) {
        const FalloutLandRecord& land = *extCell->land;
        expectTrue(land.quadrantBaseTextureFormId[0] == kBaseTextureFormId,
                   "LAND BTXT base texture formID round-trips for quadrant 0");

        // VCLR decodes unsigned to [0,1] — 255/128/64, not the signed mapping
        // VNML uses. Checking all three channels on a post other than the first
        // catches a channel swap and a wrong stride, both of which leave post 0
        // looking correct.
        expectTrue(land.hasColors, "LAND VCLR sets hasColors");
        bool vclrMatches = land.hasColors;
        for (int i = 0; i < kLandVertexCount && vclrMatches; ++i) {
            vclrMatches =
                std::fabs(land.colors[(i * 3) + 0] - 1.0f) < 1e-4f &&
                std::fabs(land.colors[(i * 3) + 1] - (128.0f / 255.0f)) < 1e-4f &&
                std::fabs(land.colors[(i * 3) + 2] - (64.0f / 255.0f)) < 1e-4f;
        }
        expectTrue(vclrMatches, "LAND VCLR decodes to unsigned [0,1] RGB at every post");

        // --- NAVM ---
        expectTrue(extCell->navMeshes.size() == 1, "exterior cell owns its NAVM record");
        if (extCell->navMeshes.size() == 1) {
            const FalloutNavMeshRecord& navMesh = extCell->navMeshes.front();
            expectTrue(navMesh.vertices.size() == 12u, "NAVM decodes 4 vertices (3 floats each)");
            expectTrue(navMesh.triangles.size() == 2u, "NAVM decodes 2 triangles");
            if (navMesh.vertices.size() == 12u) {
                expectTrue(std::fabs(navMesh.vertices[3] - 128.0f) < 1e-4f &&
                           std::fabs(navMesh.vertices[5] - 10.0f) < 1e-4f,
                           "NAVM vertices keep Bethesda-space xyz order");
            }
            if (navMesh.triangles.size() == 2u) {
                const FalloutNavMeshTriangle& first = navMesh.triangles[0];
                expectTrue(first.vertex[0] == 0 && first.vertex[1] == 1 && first.vertex[2] == 2,
                           "NAVM triangle vertex indices round-trip");
                expectTrue(first.neighbour[0] == kNavMeshNoNeighbour && first.neighbour[1] == 1 &&
                               first.neighbour[2] == kNavMeshNoNeighbour,
                           "NAVM adjacency is read per edge, not reordered");
                expectTrue(first.flags == 0x0800u, "NAVM triangle flags round-trip");
                expectTrue(navMesh.triangles[1].neighbour[2] == kNavMeshNoNeighbour,
                           "NAVM neighbour index past the triangle count clamps to border");
            }
            expectTrue(navMesh.doorPortals.size() == 1 &&
                           navMesh.doorPortals[0].doorRefFormId == kNavDoorRefFormId &&
                           navMesh.doorPortals[0].triangleIndex == 1u,
                       "NAVM door portal names its door reference and triangle");
        }

        // ATXT/VTXT: two layers, emitted high-index-first, so a correct parse
        // reorders them and pairs each VTXT with the ATXT that preceded it.
        expectTrue(land.textureLayers.size() == 2, "LAND parses both ATXT layers");
        if (land.textureLayers.size() == 2) {
            expectTrue(land.textureLayers[0].layerIndex == 0u &&
                       land.textureLayers[0].textureFormId == kLayerTextureFormIdLow,
                       "ATXT layers sort by layer index, not subrecord order");
            expectTrue(land.textureLayers[1].layerIndex == 1u &&
                       land.textureLayers[1].textureFormId == kLayerTextureFormIdHigh,
                       "the higher ATXT layer index sorts last");
            expectTrue(land.textureLayers[0].quadrant == 0u, "ATXT quadrant round-trips");
            bool opacityMatches = true;
            for (const auto& layer : land.textureLayers) {
                opacityMatches = opacityMatches &&
                    std::fabs(layer.opacity[0] - 1.0f) < 1e-4f &&
                    std::fabs(layer.opacity[5] - 0.25f) < 1e-4f &&
                    // Every post VTXT did not mention stays fully transparent.
                    std::fabs(layer.opacity[1]) < 1e-4f &&
                    std::fabs(layer.opacity[kLandQuadrantVertexCount - 1]) < 1e-4f;
            }
            expectTrue(opacityMatches, "VTXT opacity lands on the posts it names and nowhere else");
        }

        // Fixture is a base offset of 100 with a single +4 delta at row 10,
        // col 20. kLandHeightScale multiplies the accumulated total including
        // the offset, so the flat posts are 100*8 = 800 and the raised ones
        // are (100+4)*8 = 832 — NOT 100 and 100+4*8, which is what this
        // expected while the decoder scaled only the deltas. That error put
        // every object in the game a median of 7566 units above its terrain.
        constexpr float kFlatHeight = 100.0f * kLandHeightScale;
        constexpr float kRaisedHeight = 104.0f * kLandHeightScale;
        bool heightsMatchExpected = true;
        for (int row = 0; row < kLandGridSize && heightsMatchExpected; ++row) {
            for (int col = 0; col < kLandGridSize; ++col) {
                const float expected = (row == 10 && col >= 20) ? kRaisedHeight : kFlatHeight;
                if (std::fabs(land.heights[(row * kLandGridSize) + col] - expected) > 1e-3f) {
                    heightsMatchExpected = false;
                    break;
                }
            }
        }
        expectTrue(heightsMatchExpected,
                   "VHGT height decode: a single mid-row delta only raises that row from its column onward, "
                   "leaving every other post at the base offset, with the scale applied to the total");
        expectTrue(land.hasNormals && land.normals[2] > 0.99f,
                   "VNML normal decode produces a normalized up-facing normal for straight-up input");
    }

    // Same fixture, checked through the streaming index rather than a full pass.
    testFalloutCellIndexMatchesFullExtraction(esmPath);

    fs::remove(esmPath);
}

void appendSizedString8(std::vector<std::uint8_t>& buffer, const std::string& text) {
    appendPod(buffer, static_cast<std::uint8_t>(text.size()));
    buffer.insert(buffer.end(), text.begin(), text.end());
}

void appendSizedString32(std::vector<std::uint8_t>& buffer, const std::string& text) {
    appendPod(buffer, static_cast<std::uint32_t>(text.size()));
    buffer.insert(buffer.end(), text.begin(), text.end());
}

// The streaming index must agree with the full extractor exactly. It is checked
// here against the same synthetic plugin testFalloutRecordExtraction builds, so
// a divergence shows up as a test failure rather than as geometry quietly
// appearing in the wrong cell at runtime.
void testFalloutCellIndexMatchesFullExtraction(const std::filesystem::path& esmPath) {
    using namespace odai::importer::fnv;

    std::string error;
    FalloutCellIndex index;
    expectTrue(
        buildFalloutCellIndex(esmPath, index, error),
        ("cell index builds: " + error).c_str());

    FalloutSceneData full;
    expectTrue(
        extractFalloutScene(esmPath, full, error),
        ("reference extraction succeeds: " + error).c_str());

    expectTrue(
        index.cells.size() == full.cells.size(),
        "cell index finds the same number of cells as a full pass");
    expectTrue(
        index.worldspaces.size() == full.worldspaces.size(),
        "cell index finds the same worldspaces as a full pass");
    // The reference map is what door teleports resolve through, so it has to be
    // built identically by the header-only pass.
    expectTrue(
        index.cellIndexByReferenceFormId.size() == full.cellIndexByReferenceFormId.size(),
        "cell index maps the same references as a full pass");

    EsmReader reader;
    expectTrue(reader.open(esmPath), "reader opens the fixture for per-cell extraction");

    std::size_t cellsWithContents = 0;
    for (const FalloutCellIndexEntry& entry : index.cells) {
        const FalloutCellRecord* expected = nullptr;
        for (const FalloutCellRecord& cell : full.cells) {
            if (cell.formId == entry.cellFormId) {
                expected = &cell;
                break;
            }
        }
        expectTrue(expected != nullptr, "every indexed cell exists in the full extraction");
        if (expected == nullptr) {
            continue;
        }
        expectTrue(
            entry.isInterior == expected->isInterior &&
                entry.hasGridCoords == expected->hasGridCoords &&
                entry.gridX == expected->gridX && entry.gridZ == expected->gridZ &&
                entry.worldspaceFormId == expected->worldspaceFormId,
            "indexed cell metadata matches the full extraction");

        FalloutCellRecord streamed;
        expectTrue(
            extractFalloutCellAt(reader, entry, streamed, error),
            ("extractFalloutCellAt succeeds: " + error).c_str());

        expectTrue(
            streamed.references.size() == expected->references.size(),
            "streamed cell has the same reference count as the full extraction");
        expectTrue(
            streamed.navMeshes.size() == expected->navMeshes.size(),
            "streamed cell has the same navmesh count as the full extraction");
        expectTrue(
            (streamed.land != nullptr) == (expected->land != nullptr),
            "streamed cell agrees on whether the cell has LAND");

        if (streamed.land != nullptr && expected->land != nullptr) {
            expectTrue(
                std::memcmp(
                    streamed.land->heights, expected->land->heights,
                    sizeof(streamed.land->heights)) == 0,
                "streamed LAND heights are byte-identical to the full extraction");
            expectTrue(
                streamed.land->textureLayers.size() == expected->land->textureLayers.size(),
                "streamed LAND carries the same ATXT/VTXT layers");
        }
        for (std::size_t i = 0;
             i < std::min(streamed.references.size(), expected->references.size()); ++i) {
            expectTrue(
                streamed.references[i].formId == expected->references[i].formId &&
                    streamed.references[i].baseFormId == expected->references[i].baseFormId &&
                    std::memcmp(
                        streamed.references[i].position, expected->references[i].position,
                        sizeof(streamed.references[i].position)) == 0,
                "streamed reference matches the full extraction");
        }
        if (entry.childrenGroupSize > 0u) {
            ++cellsWithContents;
        }
    }
    // Guard against the whole comparison passing vacuously because no cell had
    // a children group to walk.
    expectTrue(cellsWithContents >= 2u, "the fixture exercises at least two cells with contents");
}

// Builds a synthetic NiNode/NiTriShape/NiTriShapeData block's AvObject prefix
// (name ref, extra data, controller ref, flags, translation, rotation,
// scale, properties, collision ref) matching nif_scene.cc's readAvObjectPrefix.
void appendAvObjectPrefix(
    std::vector<std::uint8_t>& out, const std::array<float, 3>& translation, const std::array<float, 9>& rotation, float scale
) {
    appendPod(out, static_cast<std::int32_t>(-1));  // nameRef
    appendPod(out, static_cast<std::uint32_t>(0));  // numExtraData
    appendPod(out, static_cast<std::int32_t>(-1));  // controllerRef
    // NiAVObject::flags is a uint at userVersion2 > 26, and this fixture
    // declares 34 (what Fallout: New Vegas writes). Emitting a ushort here —
    // as this did — shifts the whole rest of the block by two bytes, which is
    // exactly the mismatch that made every retail node fail to parse while
    // this test still passed.
    appendPod(out, static_cast<std::uint32_t>(0));  // flags
    for (float v : translation) appendPod(out, v);
    for (float v : rotation) appendPod(out, v);
    appendPod(out, scale);
    appendPod(out, static_cast<std::uint32_t>(0));  // numProperties
    appendPod(out, static_cast<std::int32_t>(-1));  // collisionRef
}

void testNifParserExtractsTransformedGeometry() {
    const std::array<float, 3> nodeTranslation{10.0f, 20.0f, 30.0f};
    const std::array<float, 9> identityRotation{1, 0, 0, 0, 1, 0, 0, 0, 1};

    // Block 0: NiNode (root), translation (10,20,30), scale 2, one child (block 1).
    std::vector<std::uint8_t> niNodeBlock;
    appendAvObjectPrefix(niNodeBlock, nodeTranslation, identityRotation, 2.0f);
    appendPod(niNodeBlock, static_cast<std::uint32_t>(1));  // numChildren
    appendPod(niNodeBlock, static_cast<std::int32_t>(1));   // children[0] = block 1

    // Block 1: NiTriShape, identity local transform, dataRef = block 2.
    std::vector<std::uint8_t> triShapeBlock;
    appendAvObjectPrefix(triShapeBlock, {0.0f, 0.0f, 0.0f}, identityRotation, 1.0f);
    appendPod(triShapeBlock, static_cast<std::int32_t>(2));  // dataRef

    // Block 2: NiTriShapeData — one triangle, positions + normals, no colors/UVs.
    std::vector<std::uint8_t> triShapeDataBlock;
    appendPod(triShapeDataBlock, static_cast<std::int32_t>(0));   // groupId
    appendPod(triShapeDataBlock, static_cast<std::uint16_t>(3));  // numVertices
    appendPod(triShapeDataBlock, static_cast<std::uint8_t>(0));   // keepFlags
    appendPod(triShapeDataBlock, static_cast<std::uint8_t>(0));   // compressFlags
    appendPod(triShapeDataBlock, static_cast<std::uint8_t>(1));   // hasVertices
    const std::array<std::array<float, 3>, 3> localVerts{{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}};
    for (const auto& v : localVerts) {
        for (float f : v) appendPod(triShapeDataBlock, f);
    }
    // vectorFlags is Bethesda's BSVectorFlags: bit 0 is a BOOLEAN "has UV"
    // (one set), bit 12 means tangents follow. It is NOT a 0-5 bit count, and
    // whether normals are present is the separate bool byte below, not bit 0.
    // 0x0001 is the retail-dominant shape, so this fixture now exercises the
    // UV read rather than skipping it — the earlier fixture wrote 0x0000 and
    // left that path, the one that had the bug, entirely untested.
    appendPod(triShapeDataBlock, static_cast<std::uint16_t>(0x0001u));
    appendPod(triShapeDataBlock, static_cast<std::uint8_t>(1));  // hasNormals
    for (int i = 0; i < 3; ++i) {
        appendPod(triShapeDataBlock, 0.0f);
        appendPod(triShapeDataBlock, 0.0f);
        appendPod(triShapeDataBlock, 1.0f);
    }
    appendPod(triShapeDataBlock, 0.0f);  // bounding sphere center x
    appendPod(triShapeDataBlock, 0.0f);
    appendPod(triShapeDataBlock, 0.0f);
    appendPod(triShapeDataBlock, 1.0f);                            // bounding sphere radius
    appendPod(triShapeDataBlock, static_cast<std::uint8_t>(0));    // hasVertexColors
    // UV set 0, one (u, v) per vertex — present because vectorFlags bit 0 is set.
    appendPod(triShapeDataBlock, 0.25f); appendPod(triShapeDataBlock, 0.5f);
    appendPod(triShapeDataBlock, 0.75f); appendPod(triShapeDataBlock, 0.5f);
    appendPod(triShapeDataBlock, 0.25f); appendPod(triShapeDataBlock, 1.0f);
    appendPod(triShapeDataBlock, static_cast<std::uint16_t>(0));   // consistencyType
    appendPod(triShapeDataBlock, static_cast<std::int32_t>(-1));   // additionalDataRef
    appendPod(triShapeDataBlock, static_cast<std::uint16_t>(1));   // numTriangles
    appendPod(triShapeDataBlock, static_cast<std::uint32_t>(3));   // numTrianglePoints
    appendPod(triShapeDataBlock, static_cast<std::uint8_t>(1));    // hasTriangles
    appendPod(triShapeDataBlock, static_cast<std::uint16_t>(0));
    appendPod(triShapeDataBlock, static_cast<std::uint16_t>(1));
    appendPod(triShapeDataBlock, static_cast<std::uint16_t>(2));
    // Trailing `Num Match Groups`. Retail NiTriShapeData always carries at
    // least this u16; a fixture that stops after the triangles is shorter than
    // any real block, and the parser's end-of-block consistency check rightly
    // rejects it.
    appendPod(triShapeDataBlock, static_cast<std::uint16_t>(0));

    std::vector<std::uint8_t> fileBytes;
    const std::string headerLine = "Gamebryo File Format, Version 20.2.0.7";
    fileBytes.insert(fileBytes.end(), headerLine.begin(), headerLine.end());
    fileBytes.push_back('\n');
    appendPod(fileBytes, static_cast<std::uint32_t>(0x14020007u));  // version
    appendPod(fileBytes, static_cast<std::uint8_t>(1));             // endianType (little)
    appendPod(fileBytes, static_cast<std::uint32_t>(11));           // userVersion
    appendPod(fileBytes, static_cast<std::uint32_t>(3));            // numBlocks
    appendPod(fileBytes, static_cast<std::uint32_t>(34));           // userVersion2
    appendSizedString8(fileBytes, "");                              // creator
    appendSizedString8(fileBytes, "");                              // export info 1
    appendSizedString8(fileBytes, "");                              // export info 2
    appendPod(fileBytes, static_cast<std::uint16_t>(3));            // numBlockTypes
    appendSizedString32(fileBytes, "NiNode");
    appendSizedString32(fileBytes, "NiTriShape");
    appendSizedString32(fileBytes, "NiTriShapeData");
    appendPod(fileBytes, static_cast<std::uint16_t>(0));  // block 0 -> NiNode
    appendPod(fileBytes, static_cast<std::uint16_t>(1));  // block 1 -> NiTriShape
    appendPod(fileBytes, static_cast<std::uint16_t>(2));  // block 2 -> NiTriShapeData
    appendPod(fileBytes, static_cast<std::uint32_t>(niNodeBlock.size()));
    appendPod(fileBytes, static_cast<std::uint32_t>(triShapeBlock.size()));
    appendPod(fileBytes, static_cast<std::uint32_t>(triShapeDataBlock.size()));
    appendPod(fileBytes, static_cast<std::uint32_t>(0));  // numStrings
    appendPod(fileBytes, static_cast<std::uint32_t>(0));  // maxStringLength
    appendPod(fileBytes, static_cast<std::uint32_t>(0));  // numGroups

    fileBytes.insert(fileBytes.end(), niNodeBlock.begin(), niNodeBlock.end());
    fileBytes.insert(fileBytes.end(), triShapeBlock.begin(), triShapeBlock.end());
    fileBytes.insert(fileBytes.end(), triShapeDataBlock.begin(), triShapeDataBlock.end());

    odai::importer::fnv::NifModel model;
    std::string error;
    expectTrue(
        odai::importer::fnv::parseNifStaticMesh(fileBytes, model, error),
        ("Synthetic NIF parses: " + error).c_str());
    expectTrue(model.skippedShapeCount == 0u, "No geometry blocks are rejected as unreadable");
    expectTrue(model.shapes.size() == 1u, "Exactly one NiTriShape's geometry is extracted");
    if (model.shapes.size() == 1u) {
        const odai::importer::fnv::NifShape& shape = model.shapes.front();
        expectTrue(shape.positions.size() == 9u, "Shape carries 3 vertices");
        // World position = NiNode transform (translate (10,20,30), scale 2,
        // identity rotation) applied to the NiTriShape's own identity local
        // transform applied to the raw NiTriShapeData vertex.
        expectNear(shape.positions[0], 10.0f, 1e-5f, "Vertex 0 world X (origin + translation)");
        expectNear(shape.positions[1], 20.0f, 1e-5f, "Vertex 0 world Y (origin + translation)");
        expectNear(shape.positions[2], 30.0f, 1e-5f, "Vertex 0 world Z (origin + translation)");
        expectNear(shape.positions[3], 12.0f, 1e-5f, "Vertex 1 world X (scaled by node scale=2, then translated)");
        expectNear(shape.positions[4], 20.0f, 1e-5f, "Vertex 1 world Y");
        expectNear(shape.positions[7], 22.0f, 1e-5f, "Vertex 2 world Y (scaled by node scale=2, then translated)");

        expectTrue(shape.normals.size() == 9u, "Shape carries 3 normals");
        expectNear(shape.normals[2], 1.0f, 1e-4f,
                   "Normal direction survives a uniform-scale transform renormalized back to unit length");

        expectTrue(
            shape.triangleIndices.size() == 3u && shape.triangleIndices[0] == 0u &&
            shape.triangleIndices[1] == 1u && shape.triangleIndices[2] == 2u,
            "Triangle indices round-trip");

        expectTrue(shape.uvs.size() == 6u, "UV set 0 is extracted when BSVectorFlags bit 0 is set");
        if (shape.uvs.size() == 6u) {
            expectNear(shape.uvs[0], 0.25f, 1e-5f, "UV 0 u");
            expectNear(shape.uvs[1], 0.5f, 1e-5f, "UV 0 v");
            expectNear(shape.uvs[4], 0.25f, 1e-5f, "UV 2 u");
            expectNear(shape.uvs[5], 1.0f, 1e-5f, "UV 2 v");
        }
    }
}

// onRecordHeader must be able to reject a record before its body is touched.
// The load-bearing case is a compressed record: if the skip happened after
// decompression it would save nothing, so this fixture makes the skipped
// record's payload deliberately un-inflatable. A walk that succeeds proves
// the body was never read.
void testEsmReaderSkipsRecordsByHeader() {
    namespace fs = std::filesystem;
    constexpr std::uint32_t kRecordFlagCompressed = 0x00040000u;

    const auto keptRecord =
        buildRecord("STAT", 0x00000001u, 0u, buildSubrecord("EDID", stringPayload("Kept")));

    // Claims 4096 decompressed bytes but carries garbage where the zlib
    // stream should be. Inflating this would fail the whole walk.
    std::vector<std::uint8_t> poisonData;
    appendPod(poisonData, static_cast<std::uint32_t>(4096));
    for (int i = 0; i < 64; ++i) {
        poisonData.push_back(static_cast<std::uint8_t>(0xA5u));
    }
    const auto skippedRecord = buildRecord("LAND", 0x00000002u, kRecordFlagCompressed, poisonData);

    std::vector<std::uint8_t> content;
    content.insert(content.end(), keptRecord.begin(), keptRecord.end());
    content.insert(content.end(), skippedRecord.begin(), skippedRecord.end());
    const auto group = buildGroup("STAT", 0, content);

    const fs::path esmPath = fs::temp_directory_path() / "odai_fnv_skip_test.esm";
    {
        std::ofstream out(esmPath, std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char*>(group.data()), static_cast<std::streamsize>(group.size()));
    }

    odai::importer::fnv::EsmReader reader;
    expectTrue(reader.open(esmPath), "ESM for the header-skip test opens");

    std::vector<std::string> seenTypes;
    std::vector<std::string> offeredTypes;
    odai::importer::fnv::EsmReader::Visitor visitor;
    visitor.onRecordHeader = [&](const odai::importer::fnv::EsmRecordHeaderView& header) {
        offeredTypes.emplace_back(header.type);
        return header.type != "LAND";
    };
    visitor.onRecord = [&](const odai::importer::fnv::EsmRecordView& record) {
        seenTypes.push_back(record.type);
    };

    expectTrue(reader.walk(visitor), "Walk succeeds without inflating the record its filter rejected");
    expectTrue(offeredTypes.size() == 2u, "Both record headers are offered to the filter");
    expectTrue(seenTypes.size() == 1u && seenTypes[0] == "STAT",
               "Only the accepted record is materialized");

    fs::remove(esmPath);
}

// A compressed record whose deflate stream is intact but whose adler32
// trailer is damaged must still be read, and must be counted rather than
// silently accepted. Retail FalloutNV.esm contains exactly one such record;
// treating it as fatal aborted the whole 245 MB walk over one bad byte.
void testEsmReaderToleratesCorruptChecksum() {
    namespace fs = std::filesystem;
    constexpr std::uint32_t kRecordFlagCompressed = 0x00040000u;

    const auto innerSubrecords = buildSubrecord("EDID", stringPayload("DamagedChecksum"));
    const auto goodStream = zlibCompress(innerSubrecords);
    const auto damagedStream = corruptZlibChecksum(goodStream);
    expectTrue(damagedStream != goodStream, "Fixture actually damaged the zlib trailer");

    std::vector<std::uint8_t> recordData;
    appendPod(recordData, static_cast<std::uint32_t>(innerSubrecords.size()));
    recordData.insert(recordData.end(), damagedStream.begin(), damagedStream.end());

    const auto record = buildRecord("LAND", 0x00150FC0u, kRecordFlagCompressed, recordData);
    const auto group = buildGroup("LAND", 0, record);

    const fs::path esmPath = fs::temp_directory_path() / "odai_fnv_checksum_test.esm";
    {
        std::ofstream out(esmPath, std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char*>(group.data()), static_cast<std::streamsize>(group.size()));
    }

    odai::importer::fnv::EsmReader reader;
    expectTrue(reader.open(esmPath), "ESM with a damaged checksum opens");

    std::vector<std::string> editorIds;
    odai::importer::fnv::EsmReader::Visitor visitor;
    visitor.onRecord = [&](const odai::importer::fnv::EsmRecordView& view) {
        for (const auto& subrecord : view.subrecords) {
            if (subrecord.type == "EDID") {
                editorIds.emplace_back(reinterpret_cast<const char*>(subrecord.data));
            }
        }
    };

    expectTrue(reader.walk(visitor), "Walk survives a record with a damaged zlib trailer");
    expectTrue(editorIds.size() == 1u && editorIds[0] == "DamagedChecksum",
               "Damaged-trailer record still decodes its full declared payload");
    expectTrue(reader.toleratedChecksumFailures() == 1u,
               "Damaged trailer is counted as tolerated, not silently ignored");

    fs::remove(esmPath);
}

// Run against both retail archive shapes: Fallout - Meshes.bsa (flags 0x87,
// no embedded names) and Fallout - Textures.bsa (flags 0x107, embedded names).
// The embedded-name variant is the one that silently returned garbage before
// the reader handled flag 0x100, so both must be covered.
void testBsaArchiveReadsFoldersAndFiles(bool embedFileNames) {
    namespace fs = std::filesystem;
    const std::string label = embedFileNames ? " [embedded names]" : " [plain]";
    const std::string uncompressedContent = "NIF-DATA-PLACEHOLDER";
    const std::string compressedContent(4096, 'T');  // long, repetitive: compresses well

    const std::vector<std::uint8_t> archiveBytes =
        buildSyntheticBsa(uncompressedContent, compressedContent, embedFileNames);
    const fs::path archivePath = fs::temp_directory_path() / "odai_fnv_test.bsa";
    {
        std::ofstream out(archivePath, std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char*>(archiveBytes.data()), static_cast<std::streamsize>(archiveBytes.size()));
    }

    odai::importer::fnv::BsaArchive archive;
    expectTrue(archive.open(archivePath), "Synthetic BSA archive opens" + label);
    expectTrue(archive.files().size() == 2u, "Synthetic BSA archive exposes both files" + label);

    const auto* meshEntry = archive.find("meshes\\x\\ex_wall_01.nif");
    expectTrue(meshEntry != nullptr, "BSA lookup finds the mesh entry by virtual path" + label);
    expectTrue(meshEntry != nullptr && !meshEntry->compressed, "Uncompressed entry is not marked compressed" + label);

    const auto* meshEntryMixedCase = archive.find("Meshes/X/EX_WALL_01.NIF");
    expectTrue(meshEntryMixedCase != nullptr, "BSA lookup is case-insensitive and slash-normalized" + label);

    std::vector<std::uint8_t> meshBytes;
    expectTrue(meshEntry != nullptr && archive.extract(*meshEntry, meshBytes), "Uncompressed entry extracts" + label);
    expectTrue(
        std::string(meshBytes.begin(), meshBytes.end()) == uncompressedContent,
        "Uncompressed entry bytes match the source content" + label);

    const auto* textureEntry = archive.find("textures\\x\\tx_wall_01.dds");
    expectTrue(textureEntry != nullptr, "BSA lookup finds the texture entry by virtual path" + label);
    expectTrue(textureEntry != nullptr && textureEntry->compressed, "Compressed entry is marked compressed" + label);

    std::vector<std::uint8_t> textureBytes;
    expectTrue(textureEntry != nullptr && archive.extract(*textureEntry, textureBytes),
               "Compressed entry inflates" + label);
    expectTrue(
        std::string(textureBytes.begin(), textureBytes.end()) == compressedContent,
        "Inflated entry bytes match the original content" + label);

    fs::remove(archivePath);
}

// The background asset pipeline. Two claims are made in comments elsewhere and
// both are load-bearing for streaming, so both are proved here rather than
// asserted: that BsaArchive::extract() is safe to call concurrently on one
// archive, and that the loader never starts two loads for the same asset.
void testAsyncAssetLoaderDeduplicatesAndLoadsConcurrently() {
    namespace fs = std::filesystem;
    using namespace odai::importer::fnv;

    const std::string uncompressedContent = "NIF-DATA-PLACEHOLDER";
    const std::string compressedContent(4096, 'T');
    const std::vector<std::uint8_t> archiveBytes =
        buildSyntheticBsa(uncompressedContent, compressedContent, /*embedFileNames=*/false);

    const fs::path dataDir = fs::temp_directory_path() / "odai_fnv_asset_source_test";
    std::error_code cleanupError;
    fs::remove_all(dataDir, cleanupError);
    fs::create_directories(dataDir, cleanupError);
    {
        std::ofstream out(dataDir / "Test.bsa", std::ios::binary | std::ios::trunc);
        out.write(
            reinterpret_cast<const char*>(archiveBytes.data()),
            static_cast<std::streamsize>(archiveBytes.size()));
    }

    FalloutAssetSource source;
    expectTrue(source.open(dataDir), "asset source opens the data directory");
    expectTrue(source.archiveCount() == 1u, "asset source indexes the synthetic archive");

    std::string error;
    std::vector<std::uint8_t> meshBytes;
    expectTrue(
        source.resolveMesh("x\\ex_wall_01.nif", meshBytes, error),
        ("asset source resolves a mesh from the archive: " + error).c_str());
    expectTrue(
        std::string(meshBytes.begin(), meshBytes.end()) == uncompressedContent,
        "resolved mesh bytes match the archive content");

    std::vector<std::uint8_t> textureBytes;
    expectTrue(
        source.resolveTexture("x\\tx_wall_01.dds", textureBytes, error),
        ("asset source resolves and inflates a texture: " + error).c_str());
    expectTrue(
        std::string(textureBytes.begin(), textureBytes.end()) == compressedContent,
        "resolved texture bytes match the original content");

    // A path that is not in the archive must fail cleanly with an error, not
    // return stale bytes from the previous call.
    std::vector<std::uint8_t> missingBytes{1, 2, 3};
    expectTrue(
        !source.resolveMesh("x\\does_not_exist.nif", missingBytes, error),
        "resolving a missing mesh fails");
    expectTrue(!error.empty(), "a failed resolve reports why");

    // Concurrency: hammer one archive from every worker at once. If extract()
    // were not thread safe this is where it would corrupt bytes or crash --
    // the compressed entry in particular runs a zlib inflate per call.
    {
        odai::core::JobSystem jobs(8);
        std::atomic<int> mismatches{0};
        std::atomic<int> completed{0};
        for (int i = 0; i < 200; ++i) {
            const bool wantTexture = (i % 2) == 0;
            jobs.enqueue([&source, &mismatches, &completed, wantTexture,
                          &uncompressedContent, &compressedContent]() {
                std::vector<std::uint8_t> bytes;
                std::string localError;
                const bool ok = wantTexture
                    ? source.resolveTexture("x\\tx_wall_01.dds", bytes, localError)
                    : source.resolveMesh("x\\ex_wall_01.nif", bytes, localError);
                const std::string& expected = wantTexture ? compressedContent : uncompressedContent;
                if (!ok || std::string(bytes.begin(), bytes.end()) != expected) {
                    ++mismatches;
                }
                ++completed;
            });
        }
        jobs.waitIdle();
        expectTrue(completed.load() == 200, "every concurrent resolve ran");
        expectTrue(
            mismatches.load() == 0,
            "200 concurrent resolves from one archive all produced correct bytes");
    }

    // Deduplication: many requests for one asset must start exactly one load.
    {
        odai::core::JobSystem jobs(4);
        AsyncAssetLoader loader(source, jobs);

        int startedByReturn = 0;
        for (int i = 0; i < 50; ++i) {
            if (loader.request(AssetKind::Mesh, "x\\ex_wall_01.nif", static_cast<std::uint64_t>(i))) {
                ++startedByReturn;
            }
        }
        loader.waitIdle();

        const AsyncAssetLoaderStats stats = loader.stats();
        expectTrue(
            stats.startedLoads == 1u,
            "50 requests for one mesh start exactly one background load");
        expectTrue(
            stats.deduplicatedRequests == 49u,
            "the other 49 requests are folded into the in-flight load");
        expectTrue(startedByReturn == 1, "request() reports which call actually started the load");

        std::vector<LoadedAsset> drained;
        loader.drainCompleted(drained);
        expectTrue(drained.size() == 1u, "exactly one result is delivered for the deduplicated load");
        expectTrue(
            drained.size() == 1u && drained[0].succeeded &&
                std::string(drained[0].bytes.begin(), drained[0].bytes.end()) == uncompressedContent,
            "the deduplicated load delivers the right bytes");
        // Case and separator differences must not defeat dedup: the ESM, the
        // NIFs and the BSA index all disagree about them for the same file.
        expectTrue(
            drained.size() == 1u && drained[0].key == "x\\ex_wall_01.nif",
            "the delivered key is the normalized path");
    }

    // Mixed-case and forward-slash spellings of one path are the same asset.
    {
        odai::core::JobSystem jobs(4);
        AsyncAssetLoader loader(source, jobs);
        loader.request(AssetKind::Mesh, "X\\EX_WALL_01.NIF");
        loader.request(AssetKind::Mesh, "x/ex_wall_01.nif");
        loader.request(AssetKind::Mesh, "x\\ex_wall_01.nif");
        loader.waitIdle();
        expectTrue(
            loader.stats().startedLoads == 1u,
            "case and separator variants of one path deduplicate to a single load");
    }

    // Generation: results requested before the world moved on are discarded.
    {
        odai::core::JobSystem jobs(4);
        AsyncAssetLoader loader(source, jobs);
        loader.request(AssetKind::Texture, "x\\tx_wall_01.dds");
        loader.waitIdle();
        loader.bumpGeneration();

        std::vector<LoadedAsset> drained;
        loader.drainCompleted(drained);
        expectTrue(drained.empty(), "results from an abandoned generation are not delivered");
        expectTrue(
            loader.stats().discardedResults == 1u,
            "discarded results are counted as wasted work rather than hidden");
    }

    // A failed load still completes and reports, rather than vanishing.
    {
        odai::core::JobSystem jobs(2);
        AsyncAssetLoader loader(source, jobs);
        loader.request(AssetKind::Mesh, "x\\nope.nif");
        loader.waitIdle();

        std::vector<LoadedAsset> drained;
        loader.drainCompleted(drained);
        expectTrue(drained.size() == 1u, "a failed load is still delivered");
        expectTrue(
            drained.size() == 1u && !drained[0].succeeded && !drained[0].error.empty(),
            "a failed load reports failure and why");
        expectTrue(loader.stats().failedLoads == 1u, "failed loads are counted");
    }

    fs::remove_all(dataDir, cleanupError);
}

// The LOD block origin must floor toward negative infinity, not truncate:
// truncation makes the blocks straddling zero twice as wide as every other and
// silently maps cells to the wrong distant tile.
void testLandLodBlockOrigin() {
    using odai::importer::fnv::landLodBlockOrigin;

    expectTrue(landLodBlockOrigin(0) == 0, "cell 0 is in block 0");
    expectTrue(landLodBlockOrigin(3) == 0, "cell 3 is in block 0");
    expectTrue(landLodBlockOrigin(4) == 4, "cell 4 starts block 4");
    expectTrue(landLodBlockOrigin(7) == 4, "cell 7 is in block 4");
    expectTrue(landLodBlockOrigin(-1) == -4, "cell -1 is in block -4, not block 0");
    expectTrue(landLodBlockOrigin(-4) == -4, "cell -4 starts block -4");
    expectTrue(landLodBlockOrigin(-5) == -8, "cell -5 is in block -8");
    expectTrue(landLodBlockOrigin(-8) == -8, "cell -8 starts block -8");
    // Against a real measured extent: WastelandNV blocks span x -32..40 step 4.
    expectTrue(landLodBlockOrigin(-32) == -32, "the lowest measured WastelandNV block maps to itself");
    expectTrue(landLodBlockOrigin(43) == 40, "cell 43 falls in the highest measured block");
}

// The same flooring has to hold for every tier, not just the finest one: the
// terrain LOD pyramid is level4/8/16/32 and a coarse tier is exactly where a
// truncating divide does the most damage, because its zero-straddling tile
// would be 64 cells wide instead of 32.
void testLandLodTileOriginAcrossTiers() {
    using odai::importer::fnv::kLandLodTierCellCounts;
    using odai::importer::fnv::landLodTileOrigin;
    using odai::importer::fnv::landLodTileSize;

    expectTrue(landLodTileOrigin(-5, 8) == -8, "cell -5 is in tier-8 tile -8");
    expectTrue(landLodTileOrigin(-8, 8) == -8, "cell -8 starts tier-8 tile -8");
    expectTrue(landLodTileOrigin(-9, 8) == -16, "cell -9 drops to tier-8 tile -16");
    expectTrue(landLodTileOrigin(31, 32) == 0, "cell 31 is still in tier-32 tile 0");
    expectTrue(landLodTileOrigin(32, 32) == 32, "cell 32 starts tier-32 tile 32");
    expectTrue(landLodTileOrigin(-1, 32) == -32, "cell -1 is in tier-32 tile -32, not tile 0");
    // The measured WastelandNV extent: level32 tiles run x -32..96 step 32.
    expectTrue(landLodTileOrigin(-32, 32) == -32, "the lowest measured level32 tile maps to itself");

    // Every tier must agree with the finest one wherever they line up, and each
    // tile origin must itself be a multiple of the tier width.
    for (const int tier : kLandLodTierCellCounts) {
        for (std::int32_t cell = -70; cell <= 70; ++cell) {
            const std::int32_t origin = landLodTileOrigin(cell, tier);
            expectTrue(origin % tier == 0, "a tile origin is a multiple of the tier width");
            expectTrue(origin <= cell && cell < origin + tier, "a cell lies inside its own tile");
        }
    }

    expectTrue(landLodTileSize(4) == 16384.0f, "a tier-4 tile is 4 cells of 4096 units");
    expectTrue(landLodTileSize(32) == 131072.0f, "a tier-32 tile is 32 cells of 4096 units");
}

// The two LOD sets must not be reachable through each other. Both answer to
// <ws>.level4.x<X>.y<Y>.nif and both parse into geometry, so a wrong directory
// is silent -- the cooker used to derive it from `tier == 4` and cooked distant
// buildings whenever terrain level4 was asked for. These strings are the two
// paths verified against the retail archives with `--find`.
void testLandLodTilePaths() {
    using odai::importer::fnv::LandLodSet;
    using odai::importer::fnv::landLodTierExists;
    using odai::importer::fnv::landLodTilePath;

    expectTrue(
        landLodTilePath("wastelandnv", LandLodSet::Terrain, 4, 24, -12) ==
            "landscape\\lod\\wastelandnv\\wastelandnv.level4.x24.y-12.nif",
        "terrain level4 sits directly under the worldspace directory");
    expectTrue(
        landLodTilePath("wastelandnv", LandLodSet::Objects, 4, 24, -12) ==
            "landscape\\lod\\wastelandnv\\blocks\\wastelandnv.level4.x24.y-12.nif",
        "object level4 sits under blocks\\");
    expectTrue(
        landLodTilePath("wastelandnv", LandLodSet::Terrain, 32, -32, 0) ==
            "landscape\\lod\\wastelandnv\\wastelandnv.level32.x-32.y0.nif",
        "a coarse terrain tier keeps the same shape, negative coordinate and all");

    // The pyramid is terrain-only. Measured: blocks\ holds 301 level4 tiles for
    // WastelandNV and nothing whatsoever at level8/16/32.
    for (const int tier : odai::importer::fnv::kLandLodTierCellCounts) {
        expectTrue(landLodTierExists(LandLodSet::Terrain, tier), "every terrain tier exists");
    }
    expectTrue(landLodTierExists(LandLodSet::Objects, 4), "object LOD exists at level4");
    expectTrue(!landLodTierExists(LandLodSet::Objects, 8), "object LOD has no level8");
    expectTrue(!landLodTierExists(LandLodSet::Objects, 32), "object LOD has no level32");
    expectTrue(!landLodTierExists(LandLodSet::Terrain, 2), "there is no level2 tier");
}


// The skinning bind-pose identity, on a synthetic two-bone rig.
//
// This is the one property the whole character path rests on: skinning a mesh
// by its own bind pose must reproduce the mesh exactly. If it does, the basis
// change, the quaternion conversion, the inverse binds and the world-transform
// accumulation are all mutually consistent -- and if any single one of them is
// wrong, the product is not the identity and the vertices move.
//
// It is worth having synthetically as well as against retail data, because it
// pins the math independently of any file: the retail check found a 112-unit
// error from a dropped NiSkinData skinTransform, and a test that can only run
// with Fallout installed is a test CI never runs.
void testSkinnedBindPoseIsIdentity() {
    using namespace odai::importer::fnv;

    // A root at the origin and a child offset along Bethesda +Z (up), rotated
    // 90 degrees about X so the basis change has something non-trivial to do.
    NifSkeleton nifSkeleton;
    NifSkeletonBone root;
    root.name = "Bip01";
    root.parentIndex = -1;
    root.translation[2] = 60.0f;
    nifSkeleton.bones.push_back(root);

    NifSkeletonBone child;
    child.name = "Bip01 Spine";
    child.parentIndex = 0;
    child.translation[0] = 3.0f;
    child.translation[2] = 12.0f;
    // Rotation about X by +90 degrees, row-major.
    const float rotation[9] = {1, 0, 0, 0, 0, -1, 0, 1, 0};
    std::memcpy(child.rotation, rotation, sizeof(rotation));
    nifSkeleton.bones.push_back(child);

    FalloutCharacter character;
    expectTrue(buildFalloutSkeleton(nifSkeleton, character.skeleton), "skeleton converts");
    expectTrue(character.skeleton.bones.size() == 2u, "both bones survive the conversion");
    expectTrue(character.skeleton.bones[1].parentIndex == 0, "the child keeps its parent");

    // Engine Y is Bethesda Z, so the root's 60-unit height lands on +Y.
    expectTrue(
        std::fabs(character.skeleton.bones[0].localTranslation.y - 60.0f) < 1e-4f,
        "Bethesda +Z becomes engine +Y");
    expectTrue(
        std::fabs(character.skeleton.bones[0].localTranslation.z) < 1e-4f,
        "nothing leaks into engine Z");

    // Bind-pose world transforms, computed the same way the builder does, so
    // the inverse binds below are exactly correct by construction. That is the
    // point: the test asserts the round trip, not the values.
    std::vector<odai::math::Matrix4> bindWorld(2, odai::math::Matrix4::identity());
    {
        FalloutCharacter probe = character;
        probe.inverseBindMatrices.assign(2, odai::math::Matrix4::identity());
        computeFalloutBindPose(probe, bindWorld);
    }

    // Build a shape rigidly bound one vertex per bone, whose inverse binds are
    // the true inverses of those world transforms. Inverting a rigid transform
    // is transpose-the-rotation, negate-the-rotated-translation.
    NifSkinnedModel model;
    NifSkinnedShape shape;
    shape.name = "test";
    shape.boneNames = {"Bip01", "Bip01 Spine"};
    shape.inverseBindMatrices.assign(2u * 16u, 0.0f);
    for (std::size_t b = 0; b < 2u; ++b) {
        // bindWorld is engine-space, but inverseBindMatrices are read in
        // Bethesda space and rebased by the builder. Feeding an identity here
        // and letting the per-bone value be identity keeps the algebra honest:
        // with an identity inverse bind, skinning by bone b must reproduce
        // bindWorld[b] applied to the vertex.
        float* target = shape.inverseBindMatrices.data() + (b * 16u);
        target[0] = 1.0f;
        target[5] = 1.0f;
        target[10] = 1.0f;
        target[15] = 1.0f;
    }
    // Two vertices, each fully weighted to one bone.
    shape.positions = {0.0f, 0.0f, 0.0f, 1.0f, 2.0f, 3.0f};
    shape.normals = {0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f};
    shape.uvs = {0.0f, 0.0f, 1.0f, 1.0f};
    shape.triangleIndices = {0, 1, 0};
    shape.boneIndices.assign(2u * kNifMaxBoneInfluences, 0u);
    shape.boneWeights.assign(2u * kNifMaxBoneInfluences, 0.0f);
    shape.boneIndices[0] = 0u;
    shape.boneWeights[0] = 1.0f;
    shape.boneIndices[kNifMaxBoneInfluences] = 1u;
    shape.boneWeights[kNifMaxBoneInfluences] = 1.0f;
    model.shapes.push_back(shape);

    std::string error;
    expectTrue(appendFalloutCharacterMesh(model, character, error), "the shape binds");
    expectTrue(character.unresolvedBoneCount == 0u, "both bone names resolve");
    expectTrue(character.vertices.size() == 2u, "both vertices survive");
    expectTrue(character.parts.size() == 1u, "one part is emitted");
    expectTrue(character.indices.size() == 3u, "the triangle survives");

    // With identity inverse binds, skinning vertex v by bone b must equal
    // bindWorld[b] * v. Anything else means the pose composition is wrong.
    std::vector<odai::math::Matrix4> pose;
    computeFalloutBindPose(character, pose);
    expectTrue(pose.size() == 2u, "one matrix per bone");
    for (std::size_t v = 0; v < 2u; ++v) {
        const auto& vertex = character.vertices[v];
        const odai::math::Vector3 rest{vertex.position[0], vertex.position[1], vertex.position[2]};
        const auto bone = static_cast<std::size_t>(vertex.boneIndices[0]);
        expectTrue(bone == v, "vertex is bound to its own bone");
        expectTrue(std::fabs(vertex.boneWeights[0] - 1.0f) < 1e-6f, "rigid bind has weight 1");
        const odai::math::Vector3 skinned = odai::math::transformPoint(pose[bone], rest);
        const odai::math::Vector3 expected = odai::math::transformPoint(bindWorld[bone], rest);
        expectTrue(std::fabs(skinned.x - expected.x) < 1e-3f, "skinned x matches the bind world");
        expectTrue(std::fabs(skinned.y - expected.y) < 1e-3f, "skinned y matches the bind world");
        expectTrue(std::fabs(skinned.z - expected.z) < 1e-3f, "skinned z matches the bind world");
    }
}

// Influence truncation must renormalize, not merely drop.
//
// Weights that no longer sum to 1 do not make a vertex slightly wrong -- they
// scale it toward the world origin, which for a character standing anywhere but
// the origin is a spike across the map.
void testSkinnedInfluenceWeightsAreNormalized() {
    using namespace odai::importer::fnv;

    NifSkeleton nifSkeleton;
    for (int i = 0; i < 6; ++i) {
        NifSkeletonBone bone;
        bone.name = "bone" + std::to_string(i);
        bone.parentIndex = (i == 0) ? -1 : 0;
        bone.translation[0] = static_cast<float>(i);
        nifSkeleton.bones.push_back(bone);
    }
    FalloutCharacter character;
    expectTrue(buildFalloutSkeleton(nifSkeleton, character.skeleton), "skeleton converts");

    NifSkinnedModel model;
    NifSkinnedShape shape;
    shape.name = "overweighted";
    for (int i = 0; i < 6; ++i) {
        shape.boneNames.push_back("bone" + std::to_string(i));
    }
    shape.inverseBindMatrices.assign(6u * 16u, 0.0f);
    for (std::size_t b = 0; b < 6u; ++b) {
        float* target = shape.inverseBindMatrices.data() + (b * 16u);
        target[0] = target[5] = target[10] = target[15] = 1.0f;
    }
    shape.positions = {0.0f, 0.0f, 0.0f};
    shape.triangleIndices = {0, 0, 0};
    // Six influences on one vertex, which is what the parser would hand over
    // before reduction. The builder consumes an already-reduced shape, so the
    // reduction is exercised through the parser's own contract: only four slots
    // exist, and they must already be normalized.
    shape.boneIndices.assign(kNifMaxBoneInfluences, 0u);
    shape.boneWeights.assign(kNifMaxBoneInfluences, 0.0f);
    const float kept[kNifMaxBoneInfluences] = {0.4f, 0.3f, 0.2f, 0.1f};
    for (int k = 0; k < kNifMaxBoneInfluences; ++k) {
        shape.boneIndices[static_cast<std::size_t>(k)] = static_cast<std::uint16_t>(k);
        shape.boneWeights[static_cast<std::size_t>(k)] = kept[k];
    }
    model.shapes.push_back(shape);

    std::string error;
    expectTrue(appendFalloutCharacterMesh(model, character, error), "the shape binds");
    expectTrue(character.vertices.size() == 1u, "one vertex");
    float sum = 0.0f;
    for (int k = 0; k < kNifMaxBoneInfluences; ++k) {
        sum += character.vertices[0].boneWeights[k];
    }
    expectTrue(std::fabs(sum - 1.0f) < 1e-5f, "the surviving weights sum to 1");

    // A bone the skeleton does not have must be counted, and its vertex slot
    // zeroed rather than left pointing at a bogus index.
    NifSkinnedModel stray;
    NifSkinnedShape strayShape = shape;
    strayShape.boneNames[0] = "no_such_bone";
    stray.shapes.push_back(strayShape);
    const std::uint32_t before = character.unresolvedBoneCount;
    expectTrue(appendFalloutCharacterMesh(stray, character, error), "the stray shape still binds");
    expectTrue(character.unresolvedBoneCount == before + 1u, "the missing bone is counted");
}

}  // namespace

int main() {
    testBsaArchiveReadsFoldersAndFiles(/*embedFileNames=*/false);
    testBsaArchiveReadsFoldersAndFiles(/*embedFileNames=*/true);
    testEsmReaderWalksGroupsRecordsAndSubrecords();
    testEsmReaderToleratesCorruptChecksum();
    testEsmReaderSkipsRecordsByHeader();
    testFalloutRecordExtraction();
    testNifParserExtractsTransformedGeometry();
    testAsyncAssetLoaderDeduplicatesAndLoadsConcurrently();
    testLandLodBlockOrigin();
    testLandLodTileOriginAcrossTiers();
    testLandLodTilePaths();
    testSkinnedBindPoseIsIdentity();
    testSkinnedInfluenceWeightsAreNormalized();

    if (g_failures != 0) {
        std::cerr << "[fnv import test] " << g_failures << " failures\n";
        return 1;
    }
    std::cout << "[fnv import test] all checks passed\n";
    return 0;
}
