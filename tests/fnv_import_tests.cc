#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <zlib.h>

#include "import/fnv/bsa_archive.h"
#include "import/fnv/esm_reader.h"
#include "import/fnv/fallout_records.h"
#include "import/fnv/nif_scene.h"

namespace {

int g_failures = 0;

void expectTrue(bool condition, const char* message) {
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
std::vector<std::uint8_t> buildSyntheticBsa(
    const std::string& uncompressedContent,
    const std::string& compressedContent
) {
    constexpr std::uint32_t kFlagHasFolderNames = 0x1u;
    constexpr std::uint32_t kFlagHasFileNames = 0x2u;
    constexpr std::uint32_t kFlagCompressedArchive = 0x4u;
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
    header.archiveFlags = kFlagHasFolderNames | kFlagHasFileNames | kFlagCompressedArchive;
    header.folderCount = 2u;
    header.fileCount = 2u;
    header.totalFolderNameLength = static_cast<std::uint32_t>(folderA.size() + folderB.size() + 2u);
    header.totalFileNameLength = totalFileNameLength;
    header.fileFlags = 0u;

    // Layout: header, 2 folder records, folderA block (name + 1 file record),
    // folderB block (name + 1 file record), file name block, file data.
    const std::size_t folderRecordsOffset = sizeof(TestBsaHeader);
    const std::size_t folderARecordOffset = folderRecordsOffset + 2 * sizeof(TestFolderRecord);
    const std::size_t folderABlockSize = 1 + folderA.size() + 1 + sizeof(TestFileRecord);
    const std::size_t folderBRecordOffset = folderARecordOffset + folderABlockSize;
    const std::size_t folderBBlockSize = 1 + folderB.size() + 1 + sizeof(TestFileRecord);
    const std::size_t fileNameBlockOffset = folderBRecordOffset + folderBBlockSize;
    const std::size_t fileDataOffset = fileNameBlockOffset + totalFileNameLength;

    // fileA: uncompressed (toggled off default-compressed archive -> stored
    // raw). fileB: compressed, stored as [uint32 originalSize][zlib bytes].
    const std::size_t fileAOffset = fileDataOffset;
    const std::size_t fileASize = uncompressedContent.size();
    const std::size_t fileBOffset = fileAOffset + fileASize;
    const std::size_t fileBSizeOnDisk = sizeof(std::uint32_t) + compressedBytes.size();

    std::vector<std::uint8_t> out;
    appendPod(out, header);

    TestFolderRecord folderRecA{};
    folderRecA.nameHash = 0x1111ull;
    folderRecA.fileCount = 1u;
    folderRecA.offset = static_cast<std::uint32_t>(folderARecordOffset);
    appendPod(out, folderRecA);

    TestFolderRecord folderRecB{};
    folderRecB.nameHash = 0x2222ull;
    folderRecB.fileCount = 1u;
    folderRecB.offset = static_cast<std::uint32_t>(folderBRecordOffset);
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

    out.insert(out.end(), uncompressedContent.begin(), uncompressedContent.end());
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

    const auto landSubrecords = [&] {
        std::vector<std::uint8_t> out;
        const auto vhgt = buildSubrecord("VHGT", vhgtPayload);
        const auto vnml = buildSubrecord("VNML", vnmlPayload);
        const auto btxt = buildSubrecord("BTXT", btxtPayload);
        out.insert(out.end(), vhgt.begin(), vhgt.end());
        out.insert(out.end(), vnml.begin(), vnml.end());
        out.insert(out.end(), btxt.begin(), btxt.end());
        return out;
    }();
    constexpr std::uint32_t kLandFormId = 0x00003000u;
    const auto landRecord = buildRecord("LAND", kLandFormId, 0u, landSubrecords);

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

    expectTrue(extCell != nullptr && extCell->land.has_value(), "Exterior cell owns its LAND record");
    if (extCell != nullptr && extCell->land.has_value()) {
        const FalloutLandRecord& land = *extCell->land;
        expectTrue(land.quadrantBaseTextureFormId[0] == kBaseTextureFormId,
                   "LAND BTXT base texture formID round-trips for quadrant 0");

        bool heightsMatchExpected = true;
        for (int row = 0; row < kLandGridSize && heightsMatchExpected; ++row) {
            for (int col = 0; col < kLandGridSize; ++col) {
                const float expected = (row == 10 && col >= 20) ? 132.0f : 100.0f;
                if (std::fabs(land.heights[(row * kLandGridSize) + col] - expected) > 1e-3f) {
                    heightsMatchExpected = false;
                    break;
                }
            }
        }
        expectTrue(heightsMatchExpected,
                   "VHGT height decode: a single mid-row delta only raises that row from its column onward, "
                   "leaving every other post at the base offset");
        expectTrue(land.hasNormals && land.normals[2] > 0.99f,
                   "VNML normal decode produces a normalized up-facing normal for straight-up input");
    }

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

// Builds a synthetic NiNode/NiTriShape/NiTriShapeData block's AvObject prefix
// (name ref, extra data, controller ref, flags, translation, rotation,
// scale, properties, collision ref) matching nif_scene.cc's readAvObjectPrefix.
void appendAvObjectPrefix(
    std::vector<std::uint8_t>& out, const std::array<float, 3>& translation, const std::array<float, 9>& rotation, float scale
) {
    appendPod(out, static_cast<std::int32_t>(-1));  // nameRef
    appendPod(out, static_cast<std::uint32_t>(0));  // numExtraData
    appendPod(out, static_cast<std::int32_t>(-1));  // controllerRef
    appendPod(out, static_cast<std::uint16_t>(0));  // flags
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
    appendPod(triShapeDataBlock, static_cast<std::uint16_t>(0x0001u));  // vectorFlags: hasNormals, no tangents, 0 UV sets
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
    appendPod(triShapeDataBlock, static_cast<std::uint16_t>(0));   // consistencyType
    appendPod(triShapeDataBlock, static_cast<std::int32_t>(-1));   // additionalDataRef
    appendPod(triShapeDataBlock, static_cast<std::uint16_t>(1));   // numTriangles
    appendPod(triShapeDataBlock, static_cast<std::uint32_t>(3));   // numTrianglePoints
    appendPod(triShapeDataBlock, static_cast<std::uint8_t>(1));    // hasTriangles
    appendPod(triShapeDataBlock, static_cast<std::uint16_t>(0));
    appendPod(triShapeDataBlock, static_cast<std::uint16_t>(1));
    appendPod(triShapeDataBlock, static_cast<std::uint16_t>(2));

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
    }
}

void testBsaArchiveReadsFoldersAndFiles() {
    namespace fs = std::filesystem;
    const std::string uncompressedContent = "NIF-DATA-PLACEHOLDER";
    const std::string compressedContent(4096, 'T');  // long, repetitive: compresses well

    const std::vector<std::uint8_t> archiveBytes = buildSyntheticBsa(uncompressedContent, compressedContent);
    const fs::path archivePath = fs::temp_directory_path() / "odai_fnv_test.bsa";
    {
        std::ofstream out(archivePath, std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char*>(archiveBytes.data()), static_cast<std::streamsize>(archiveBytes.size()));
    }

    odai::importer::fnv::BsaArchive archive;
    expectTrue(archive.open(archivePath), "Synthetic BSA archive opens");
    expectTrue(archive.files().size() == 2u, "Synthetic BSA archive exposes both files");

    const auto* meshEntry = archive.find("meshes\\x\\ex_wall_01.nif");
    expectTrue(meshEntry != nullptr, "BSA lookup finds the mesh entry by virtual path");
    expectTrue(meshEntry != nullptr && !meshEntry->compressed, "Uncompressed entry is not marked compressed");

    const auto* meshEntryMixedCase = archive.find("Meshes/X/EX_WALL_01.NIF");
    expectTrue(meshEntryMixedCase != nullptr, "BSA lookup is case-insensitive and slash-normalized");

    std::vector<std::uint8_t> meshBytes;
    expectTrue(meshEntry != nullptr && archive.extract(*meshEntry, meshBytes), "Uncompressed entry extracts");
    expectTrue(
        std::string(meshBytes.begin(), meshBytes.end()) == uncompressedContent,
        "Uncompressed entry bytes match the source content");

    const auto* textureEntry = archive.find("textures\\x\\tx_wall_01.dds");
    expectTrue(textureEntry != nullptr, "BSA lookup finds the texture entry by virtual path");
    expectTrue(textureEntry != nullptr && textureEntry->compressed, "Compressed entry is marked compressed");

    std::vector<std::uint8_t> textureBytes;
    expectTrue(textureEntry != nullptr && archive.extract(*textureEntry, textureBytes), "Compressed entry inflates");
    expectTrue(
        std::string(textureBytes.begin(), textureBytes.end()) == compressedContent,
        "Inflated entry bytes match the original content");

    fs::remove(archivePath);
}

}  // namespace

int main() {
    testBsaArchiveReadsFoldersAndFiles();
    testEsmReaderWalksGroupsRecordsAndSubrecords();
    testFalloutRecordExtraction();
    testNifParserExtractsTransformedGeometry();

    if (g_failures != 0) {
        std::cerr << "[fnv import test] " << g_failures << " failures\n";
        return 1;
    }
    std::cout << "[fnv import test] all checks passed\n";
    return 0;
}
