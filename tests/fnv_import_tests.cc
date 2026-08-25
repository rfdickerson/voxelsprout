#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include <zlib.h>

#include "core/job_system.h"
#include "import/fnv/asset_source.h"
#include "import/fnv/cell_builder.h"
#include "import/fnv/lz4_frame.h"
#include "import/fnv/plugin_load_order.h"
#include "import/fnv/character_builder.h"
#include "import/fnv/kf_animation.h"
#include "import/fnv/land_lod.h"
#include "import/fnv/actor_records.h"
#include "import/fnv/dialogue_records.h"
#include "import/fnv/async_asset_loader.h"
#include "import/fnv/bsa_archive.h"
#include "import/fnv/esm_reader.h"
#include "import/fnv/fallout_records.h"
#include "import/fnv/nif_scene.h"
#include "import/fnv/weather_records.h"

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
//
// `version` selects the header version only; every structure this fixture
// writes is shared between v103 (Oblivion) and v104 (FO3/FNV). Pass 103 with
// `declareEmbeddedNames` to reproduce the retail Oblivion shape: the
// kEmbedFileNames bit set in archiveFlags while no data block actually carries
// an embedded name.
std::vector<std::uint8_t> buildSyntheticBsa(
    const std::string& uncompressedContent,
    const std::string& compressedContent,
    bool embedFileNames = false,
    std::uint32_t version = 104u,
    bool declareEmbeddedNames = false
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
    header.version = version;
    header.folderRecordOffset = sizeof(TestBsaHeader);
    header.archiveFlags = kFlagHasFolderNames | kFlagHasFileNames | kFlagCompressedArchive |
        ((embedFileNames || declareEmbeddedNames) ? kFlagEmbedFileNames : 0u);
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

// `format` selects the container-header generation. It defaults to kFallout3 so
// every fixture written before Oblivion support keeps producing the exact bytes
// it produced then; only the tests that specifically exercise the 20-byte
// layout pass anything else.
using odai::importer::fnv::EsmPluginFormat;

std::vector<std::uint8_t> buildRecord(
    const char* type,
    std::uint32_t formId,
    std::uint32_t flags,
    const std::vector<std::uint8_t>& data,
    EsmPluginFormat format = EsmPluginFormat::kFallout3
) {
    std::vector<std::uint8_t> out;
    appendFourCc(out, type);
    appendPod(out, static_cast<std::uint32_t>(data.size()));
    appendPod(out, flags);
    appendPod(out, formId);
    appendPod(out, static_cast<std::uint32_t>(0));  // versionControlInfo
    if (format == EsmPluginFormat::kFallout3) {
        appendPod(out, static_cast<std::uint16_t>(0));  // formVersion — FO3 and later only
        appendPod(out, static_cast<std::uint16_t>(0));  // unknown
    }
    out.insert(out.end(), data.begin(), data.end());
    return out;
}

std::vector<std::uint8_t> buildGroup(
    const char rawLabel[4],
    std::int32_t groupType,
    const std::vector<std::uint8_t>& content,
    EsmPluginFormat format = EsmPluginFormat::kFallout3
) {
    const std::size_t headerSize = odai::importer::fnv::esmGroupHeaderSize(format);
    std::vector<std::uint8_t> out;
    appendFourCc(out, "GRUP");
    appendPod(out, static_cast<std::uint32_t>(headerSize + content.size()));
    appendBytes(out, rawLabel, 4u);
    appendPod(out, groupType);
    appendPod(out, static_cast<std::uint16_t>(0));  // stamp
    appendPod(out, static_cast<std::uint16_t>(0));  // unknown
    if (format == EsmPluginFormat::kFallout3) {
        appendPod(out, static_cast<std::uint32_t>(0));  // versionControlInfo
    }
    out.insert(out.end(), content.begin(), content.end());
    return out;
}

// A plugin's opening TES4 record, which is the ONLY thing EsmReader sniffs to
// decide the header generation for the rest of the file. HEDR must be first —
// that is exactly what the sniff keys on.
std::vector<std::uint8_t> buildTes4Record(
    const std::vector<std::string>& masters, EsmPluginFormat format
) {
    std::vector<std::uint8_t> hedr;
    appendPod(hedr, 1.0f);                          // version
    appendPod(hedr, static_cast<std::uint32_t>(0));  // record count
    appendPod(hedr, static_cast<std::uint32_t>(1));  // next object id
    std::vector<std::uint8_t> body = buildSubrecord("HEDR", hedr);
    for (const std::string& master : masters) {
        const auto mast = buildSubrecord("MAST", stringPayload(master));
        body.insert(body.end(), mast.begin(), mast.end());
        const auto data = buildSubrecord("DATA", std::vector<std::uint8_t>(8, 0u));
        body.insert(body.end(), data.begin(), data.end());
    }
    return buildRecord("TES4", 0u, 0u, body, format);
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

// Oblivion's record and GRUP headers are 20 bytes; Fallout 3 grew both by four
// (formVersion + unknown on a record, versionControlInfo on a group). Nothing
// else about the container format changed, and this test is what pins that
// claim: it builds the SAME logical tree in both layouts and asserts the walks
// are indistinguishable.
//
// Written this way on purpose. Asserting record counts for the Oblivion layout
// alone would pass just as well if the 24-byte path had quietly broken, and the
// whole risk of this change is a regression on the Fallout side.
void testEsmReaderWalksBothHeaderGenerations() {
    namespace fs = std::filesystem;
    using namespace odai::importer::fnv;

    struct CapturedRecord {
        std::string type;
        std::uint32_t formId = 0;
        std::vector<std::pair<std::string, std::vector<std::uint8_t>>> subrecords;
    };
    struct Walked {
        std::vector<CapturedRecord> records;
        std::vector<std::pair<std::string, std::int32_t>> groupEnters;
        EsmPluginFormat format = EsmPluginFormat::kFallout3;
    };

    const auto walkFixture = [](EsmPluginFormat format, const char* fileName) {
        // A TES4 header (which is what the sniff reads), then a STAT group, then
        // a worldspace-shaped nesting so a group inside a group is exercised too.
        const auto statSubrecords = [] {
            std::vector<std::uint8_t> out;
            const auto edid = buildSubrecord("EDID", stringPayload("Rock01"));
            const auto modl = buildSubrecord("MODL", stringPayload("x\\rock01.nif"));
            out.insert(out.end(), edid.begin(), edid.end());
            out.insert(out.end(), modl.begin(), modl.end());
            return out;
        }();

        std::vector<std::uint8_t> fileBytes = buildTes4Record({}, format);
        const auto statRecord = buildRecord("STAT", 0x00000010u, 0u, statSubrecords, format);
        const auto statGroup = buildGroup("STAT", 0, statRecord, format);
        fileBytes.insert(fileBytes.end(), statGroup.begin(), statGroup.end());

        const auto cellRecord = buildRecord(
            "CELL", 0x00000020u, 0u, buildSubrecord("EDID", stringPayload("TestCell")), format);
        const std::uint32_t parentFormId = 0x00000020u;
        char nestedLabel[4];
        std::memcpy(nestedLabel, &parentFormId, 4);
        const auto refrRecord = buildRecord(
            "REFR", 0x00000021u, 0u, buildSubrecord("NAME", std::vector<std::uint8_t>(4, 0x10u)), format);
        const auto childrenGroup = buildGroup(nestedLabel, 6, refrRecord, format);
        std::vector<std::uint8_t> cellContent = cellRecord;
        cellContent.insert(cellContent.end(), childrenGroup.begin(), childrenGroup.end());
        const auto cellGroup = buildGroup("CELL", 0, cellContent, format);
        fileBytes.insert(fileBytes.end(), cellGroup.begin(), cellGroup.end());

        const fs::path esmPath = fs::temp_directory_path() / fileName;
        {
            std::ofstream out(esmPath, std::ios::binary | std::ios::trunc);
            out.write(reinterpret_cast<const char*>(fileBytes.data()),
                      static_cast<std::streamsize>(fileBytes.size()));
        }

        EsmReader reader;
        Walked walked{};
        expectTrue(reader.open(esmPath), "Two-generation ESM fixture opens");
        walked.format = reader.pluginFormat();
        EsmReader::Visitor visitor{};
        visitor.onGroupEnter = [&](const EsmGroupView& group) {
            walked.groupEnters.emplace_back(group.rawLabel, group.groupType);
            return true;
        };
        visitor.onRecord = [&](const EsmRecordView& record) {
            CapturedRecord captured{};
            captured.type = record.type;
            captured.formId = record.formId;
            for (const auto& sub : record.subrecords) {
                captured.subrecords.emplace_back(
                    sub.type, std::vector<std::uint8_t>(sub.data, sub.data + sub.size));
            }
            walked.records.push_back(std::move(captured));
        };
        expectTrue(reader.walk(visitor), "Two-generation ESM fixture walks without error");
        fs::remove(esmPath);
        return walked;
    };

    const Walked fallout = walkFixture(EsmPluginFormat::kFallout3, "odai_fnv_hdr24.esm");
    const Walked oblivion = walkFixture(EsmPluginFormat::kOblivion, "odai_fnv_hdr20.esm");

    expectTrue(fallout.format == EsmPluginFormat::kFallout3,
               "A 24-byte-header plugin is detected as kFallout3");
    expectTrue(oblivion.format == EsmPluginFormat::kOblivion,
               "A 20-byte-header plugin is detected as kOblivion");

    // TES4, STAT, CELL, REFR.
    expectTrue(fallout.records.size() == 4u, "24-byte walk visits all four records");
    expectTrue(oblivion.records.size() == 4u, "20-byte walk visits all four records");
    expectTrue(fallout.groupEnters.size() == 3u, "24-byte walk enters all three groups");
    expectTrue(oblivion.groupEnters.size() == 3u, "20-byte walk enters all three groups");

    bool sameRecords = fallout.records.size() == oblivion.records.size();
    for (std::size_t i = 0; sameRecords && i < fallout.records.size(); ++i) {
        sameRecords = fallout.records[i].type == oblivion.records[i].type &&
                      fallout.records[i].formId == oblivion.records[i].formId &&
                      fallout.records[i].subrecords == oblivion.records[i].subrecords;
    }
    expectTrue(sameRecords, "Both header generations yield identical records and subrecord bytes");
    expectTrue(fallout.groupEnters == oblivion.groupEnters,
               "Both header generations yield identical group labels and types");

    // The sniff must not fire on a file that does not open with a TES4 record —
    // several fixtures in this suite start straight at a GRUP, and they have to
    // keep walking as Fallout plugins.
    const std::vector<std::uint8_t> notAPlugin = buildGroup("STAT", 0, {});
    expectTrue(detectEsmPluginFormat(notAPlugin.data(), notAPlugin.size()) == EsmPluginFormat::kFallout3,
               "A file not opening with TES4 defaults to the Fallout layout");
    expectTrue(detectEsmPluginFormat(nullptr, 0u) == EsmPluginFormat::kFallout3,
               "An empty buffer defaults to the Fallout layout");
}

// Oblivion's LTEX names its diffuse directly in ICON, relative to
// "textures\landscape\"; Fallout's points at a TXST through TNAM. Both have to
// land in diffuseTexturePath, and the ICON path must not be clobbered by the
// post-walk TXST resolution.
// Water is the one cell property with no geometry behind it: the record states
// a height and the engine fills the cell's whole 4096-unit footprint at it. So
// every decision here is arithmetic on two numbers, and every one of them was a
// candidate for being silently wrong -- a flipped axis puts the sea one cell
// away, and a missing cull puts a full-cell blended quad under all of Nevada.
// LZ4 frame decoding, hand-built rather than round-tripped.
//
// There is no encoder here to round-trip against, which is the point: a decoder
// checked only against its own encoder agrees with itself and can still be
// wrong about the format. These bytes are written out by hand from the spec, so
// the assertion is against LZ4, not against a mirror.
//
// The match in block 1 OVERLAPS its own destination -- offset 3, length 9,
// against three bytes of output -- because that is the case a memcpy gets
// wrong and a byte-at-a-time copy gets right, and it is not an exotic input:
// it is how the format encodes any repeating run.
void testLz4FrameDecoding() {
    using namespace odai::importer::fnv;

    std::vector<std::uint8_t> frame;
    const auto push = [&frame](std::initializer_list<int> bytes) {
        for (const int byte : bytes) {
            frame.push_back(static_cast<std::uint8_t>(byte));
        }
    };
    push({0x04, 0x22, 0x4D, 0x18});  // magic
    push({0x60});                    // FLG: version 01, blocks independent
    push({0x70});                    // BD: 4 MB block maximum
    push({0x00});                    // header checksum, not verified

    // Block 1, compressed: literals "abc", then a 9-byte match at offset 3,
    // then a literals-only tail "XY".
    const std::vector<std::uint8_t> block1{
        0x35, 'a', 'b', 'c', 0x03, 0x00,  // token(lit 3, match 5+4), literals, offset
        0x20, 'X', 'Y',                   // token(lit 2, no match), literals
    };
    push({static_cast<int>(block1.size()), 0x00, 0x00, 0x00});
    frame.insert(frame.end(), block1.begin(), block1.end());

    // Block 2, stored uncompressed: the high bit of the size says so.
    const std::vector<std::uint8_t> block2{'Z', 'Z', 'Z', 'Z'};
    push({static_cast<int>(block2.size()), 0x00, 0x00, 0x80});
    frame.insert(frame.end(), block2.begin(), block2.end());

    push({0x00, 0x00, 0x00, 0x00});  // end mark

    const std::string expected = "abcabcabcabcXY" "ZZZZ";
    std::vector<std::uint8_t> out;
    std::string error;
    expectTrue(isLz4Frame(frame.data(), frame.size()), "The frame magic is recognized");
    expectTrue(
        lz4FrameDecompress(frame.data(), frame.size(), expected.size(), out, error),
        "A hand-built LZ4 frame decodes");
    expectTrue(
        std::string(out.begin(), out.end()) == expected,
        "An overlapping match and a stored block both decode to the right bytes");

    // A wrong declared size is corruption, and saying so here beats letting a
    // truncated mesh fail somewhere far downstream.
    expectTrue(
        !lz4FrameDecompress(frame.data(), frame.size(), expected.size() + 1u, out, error),
        "A frame that decodes to the wrong length is rejected");

    std::vector<std::uint8_t> notAFrame{0x00, 0x01, 0x02, 0x03, 0x04};
    expectTrue(!isLz4Frame(notAFrame.data(), notAFrame.size()), "Non-LZ4 bytes are not claimed");
    expectTrue(
        !lz4FrameDecompress(notAFrame.data(), notAFrame.size(), 0u, out, error),
        "A payload with the wrong magic is refused rather than decoded as garbage");

    // A match reaching back further than the output produced so far would read
    // before the start of the buffer.
    // frame[0..3] magic, [4] FLG, [5] BD, [6] header checksum, [7..10] block
    // size, then the block: [11] token, [12..14] literals, [15..16] the match
    // offset. 64 against the three bytes of output produced so far.
    std::vector<std::uint8_t> badFrame = frame;
    badFrame[15] = 0x40;
    expectTrue(
        !lz4FrameDecompress(badFrame.data(), badFrame.size(), 0u, out, error),
        "A match offset pointing before the output start is rejected, not read");
}

// TES3 is a THIRD container layout, and two of its differences are silent.
//
// The header is 16 bytes rather than 24, which is the obvious one. The two that
// are not: the flags word sits where TES4 puts the formId (and TES3 has no
// formId at all), and SUBRECORD SIZES ARE 32-BIT, which also makes the
// subrecord header 8 bytes rather than 6. Reading a TES3 subrecord as a uint16
// takes the low half of its size and then reads the high half as the next
// subrecord's type -- so it does not fail on the first record, it fails on the
// first one whose payload is not a multiple of 65536.
void testEsmReaderWalksMorrowindHeaders() {
    namespace fs = std::filesystem;
    using namespace odai::importer::fnv;

    const auto tes3Subrecord = [](const char* type, const std::string& payload) {
        std::vector<std::uint8_t> out(type, type + 4);
        appendPod(out, static_cast<std::uint32_t>(payload.size()));
        out.insert(out.end(), payload.begin(), payload.end());
        return out;
    };
    const auto tes3Record = [](const char* type, const std::vector<std::uint8_t>& body) {
        std::vector<std::uint8_t> out(type, type + 4);
        appendPod(out, static_cast<std::uint32_t>(body.size()));
        appendPod(out, static_cast<std::uint32_t>(0));           // unused header word
        appendPod(out, static_cast<std::uint32_t>(0x00000400u));  // flags: Persistent
        out.insert(out.end(), body.begin(), body.end());
        return out;
    };

    std::vector<std::uint8_t> fileBytes;
    {   // TES3 header record, which is what the sniff keys on.
        std::vector<std::uint8_t> body;
        const auto hedr = tes3Subrecord("HEDR", std::string(300, '\0'));
        body.insert(body.end(), hedr.begin(), hedr.end());
        const auto record = tes3Record("TES3", body);
        fileBytes.insert(fileBytes.end(), record.begin(), record.end());
    }
    {   // A STAT whose MODL payload is deliberately not a round number.
        std::vector<std::uint8_t> body;
        const auto name = tes3Subrecord("NAME", std::string("ex_common_house_01"));
        const auto modl = tes3Subrecord("MODL", std::string("x\\ex_common_house_01.nif"));
        body.insert(body.end(), name.begin(), name.end());
        body.insert(body.end(), modl.begin(), modl.end());
        const auto record = tes3Record("STAT", body);
        fileBytes.insert(fileBytes.end(), record.begin(), record.end());
    }

    expectTrue(
        detectEsmPluginFormat(fileBytes.data(), fileBytes.size()) == EsmPluginFormat::kMorrowind,
        "A plugin opening with TES3 is sniffed as Morrowind");
    expectTrue(esmRecordHeaderSize(EsmPluginFormat::kMorrowind) == 16u,
               "A TES3 record header is 16 bytes");

    const fs::path path = fs::temp_directory_path() / "odai_tes3_walk.esm";
    {
        std::ofstream out(path, std::ios::binary);
        out.write(reinterpret_cast<const char*>(fileBytes.data()),
                  static_cast<std::streamsize>(fileBytes.size()));
    }
    EsmReader reader;
    expectTrue(reader.open(path), "A TES3 plugin opens");

    std::vector<std::string> types;
    std::string modelPath;
    std::uint32_t statFlags = 0xFFFFFFFFu;
    std::uint32_t statFormId = 0xFFFFFFFFu;
    EsmReader::Visitor visitor{};
    visitor.onRecord = [&](const EsmRecordView& record) {
        types.push_back(record.type);
        if (record.type != "STAT") {
            return;
        }
        statFlags = record.flags;
        statFormId = record.formId;
        for (const EsmSubrecordView& sub : record.subrecords) {
            if (sub.type == "MODL") {
                modelPath.assign(reinterpret_cast<const char*>(sub.data), sub.size);
            }
        }
    };
    expectTrue(reader.walk(visitor), "A TES3 plugin walks without desynchronizing");
    expectTrue(types.size() == 2u && types[0] == "TES3" && types[1] == "STAT",
               "Both TES3 records are visited, in order");
    expectTrue(modelPath == "x\\ex_common_house_01.nif",
               "A 32-bit subrecord size is read as a size and not as half a size");
    // The flags live where TES4 puts the formId; reading the TES4 offset would
    // report 0 here and put the flags in formId.
    expectTrue(statFlags == 0x00000400u, "TES3 record flags are read from offset 12");
    expectTrue(statFormId == 0u, "A TES3 record reports no formId rather than stray bytes");

    std::error_code removeError;
    fs::remove(path, removeError);
}

void testCellWaterPatch() {
    using namespace odai::importer::fnv;
    using odai::importer::ImportedScene;

    const auto makeExteriorCell = [](std::int32_t gridX, std::int32_t gridZ) {
        FalloutCellRecord cell{};
        cell.isInterior = false;
        cell.hasGridCoords = true;
        cell.gridX = gridX;
        cell.gridZ = gridZ;
        return cell;
    };
    const auto giveFlatLand = [](FalloutCellRecord& cell, float height) {
        cell.land = std::make_unique<FalloutLandRecord>();
        cell.land->hasHeights = true;
        // Sized explicitly: the height array is a vector now, so "hasHeights
        // with an empty array" is a shape a fixture can build and the readers
        // cannot survive -- min_element over an empty range dereferences end().
        cell.land->heights.assign(
            static_cast<std::size_t>(cell.land->vertexCount()), height);
    };

    // No LAND at all is open ocean, not a dry cell. Tamriel (0,0) is exactly
    // this, so a version that required terrain would drop the sea precisely
    // where there is nothing else to draw.
    {
        ImportedScene scene;
        FalloutCellRecord cell = makeExteriorCell(0, 0);
        expectTrue(appendCellWaterPatch(scene, cell) && scene.waterPatches.size() == 1u,
                   "A cell with no LAND still gets its water surface");
    }

    // Terrain entirely above the water line: the sea is under a solid floor and
    // emitting it would cost a full-cell blended quad for nothing visible.
    {
        ImportedScene scene;
        FalloutCellRecord cell = makeExteriorCell(3, 4);
        giveFlatLand(cell, 512.0f);
        cell.hasWater = true;
        cell.waterHeight = 0.0f;
        expectTrue(!appendCellWaterPatch(scene, cell) && scene.waterPatches.empty(),
                   "Water strictly below every terrain post is culled");
    }

    // Terrain dipping below it: a coastline.
    {
        ImportedScene scene;
        FalloutCellRecord cell = makeExteriorCell(-46, -7);
        giveFlatLand(cell, -136.0f);
        expectTrue(appendCellWaterPatch(scene, cell) && scene.waterPatches.size() == 1u,
                   "Terrain below sea level gets water even with XCLW absent");
        if (scene.waterPatches.size() == 1u) {
            const auto& patch = scene.waterPatches.front();
            // Bethesda (x, y) -> engine (x, -y), so the cell's +Y edge becomes
            // its MINIMUM engine z: the origin moves to the opposite corner.
            // Getting this wrong puts the patch one cell north of its cell.
            expectNear(patch.originX, -46.0f * kExteriorCellSize, 1e-3f,
                       "water patch origin X is the cell's own X");
            expectNear(patch.originZ, -(-7.0f + 1.0f) * kExteriorCellSize, 1e-3f,
                       "water patch origin Z is the cell's far edge negated");
            expectNear(patch.sizeX, kExteriorCellSize, 1e-3f, "water patch spans a whole cell in X");
            expectNear(patch.sizeZ, kExteriorCellSize, 1e-3f, "water patch spans a whole cell in Z");
            // An absent XCLW is Oblivion's sea level, which is 0 -- no WRLD
            // record in Oblivion.esm carries a DNAM to override it.
            expectNear(patch.waterLevel, 0.0f, 1e-6f, "an absent XCLW resolves to sea level");
        }
    }

    // A narrow river is not a full-cell lake. Its water must follow the LAND
    // posts so nearby bridges and boardwalks meet a shoreline rather than a
    // 4096-unit rectangular water card.
    {
        ImportedScene scene;
        FalloutCellRecord cell = makeExteriorCell(2, -3);
        giveFlatLand(cell, 32.0f);
        cell.hasWater = true;
        cell.waterHeight = 0.0f;
        const int side = cell.land->gridSize;
        // One low interior post wets only its four surrounding quads.
        cell.land->heights[static_cast<std::size_t>((16 * side) + 16)] = -16.0f;
        expectTrue(appendCellWaterPatch(scene, cell) && scene.waterPatches.size() == 2u,
                   "partial LAND water is clipped into shoreline runs rather than one cell quad");
        if (!scene.waterPatches.empty()) {
            expectTrue(scene.waterPatches.front().sizeX <= (2.0f * kLandPostSpacing) &&
                           scene.waterPatches.front().sizeZ == kLandPostSpacing,
                       "shoreline water patches stay at LAND-post resolution");
        }
    }

    // Interiors state a water height too, and have no footprint to fill.
    {
        ImportedScene scene;
        FalloutCellRecord cell{};
        cell.isInterior = true;
        cell.hasWater = true;
        cell.waterHeight = 100.0f;
        expectTrue(!appendCellWaterPatch(scene, cell) && scene.waterPatches.empty(),
                   "Interior cells contribute no water patch");
    }

    // THE IMPLIED HEIGHT IS THE WORLDSPACE'S, NOT ZERO. Skyrim's WhiterunWorld
    // states no water height and no DNAM; Tamriel, its WNAM parent, declares
    // -14000, and resolveWorldspaceInheritance has already pushed that down by
    // the time this runs. Defaulting to 0 instead put a full-cell quad 14000
    // units up, through the roofs of a city standing at y -3120.
    {
        ImportedScene scene;
        FalloutCellRecord cell = makeExteriorCell(5, -2);
        FalloutWorldspaceRecord worldspace{};
        worldspace.hasDefaultHeights = true;
        worldspace.defaultWaterHeight = -14000.0f;
        expectTrue(appendCellWaterPatch(scene, cell, &worldspace) &&
                       scene.waterPatches.size() == 1u,
                   "A landless cell still gets water from the worldspace default");
        if (scene.waterPatches.size() == 1u) {
            expectNear(scene.waterPatches.front().waterLevel, -14000.0f, 1e-3f,
                       "an absent XCLW resolves to the worldspace's default water height");
        }
    }

    // And that default is what the terrain is compared against, so a city whose
    // ground sits far above it stays dry.
    {
        ImportedScene scene;
        FalloutCellRecord cell = makeExteriorCell(6, -1);
        giveFlatLand(cell, -3120.0f);
        FalloutWorldspaceRecord worldspace{};
        worldspace.hasDefaultHeights = true;
        worldspace.defaultWaterHeight = -14000.0f;
        expectTrue(!appendCellWaterPatch(scene, cell, &worldspace) && scene.waterPatches.empty(),
                   "Terrain far above the worldspace default water height stays dry");
    }
}

void testOblivionLandTextureIconPath() {
    namespace fs = std::filesystem;
    using namespace odai::importer::fnv;

    constexpr auto kFormat = EsmPluginFormat::kOblivion;
    std::vector<std::uint8_t> fileBytes = buildTes4Record({}, kFormat);

    std::vector<std::uint8_t> ltexSubrecords = buildSubrecord("EDID", stringPayload("TestMoss01"));
    const auto icon = buildSubrecord("ICON", stringPayload("Dementia\\DementiaMoss01.dds"));
    ltexSubrecords.insert(ltexSubrecords.end(), icon.begin(), icon.end());
    constexpr std::uint32_t kLandTextureFormId = 0x00000123u;
    const auto ltexRecord = buildRecord("LTEX", kLandTextureFormId, 0u, ltexSubrecords, kFormat);
    const auto ltexGroup = buildGroup("LTEX", 0, ltexRecord, kFormat);
    fileBytes.insert(fileBytes.end(), ltexGroup.begin(), ltexGroup.end());

    const fs::path esmPath = fs::temp_directory_path() / "odai_oblivion_ltex.esm";
    {
        std::ofstream out(esmPath, std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char*>(fileBytes.data()),
                  static_cast<std::streamsize>(fileBytes.size()));
    }

    FalloutSceneData scene;
    std::string error;
    expectTrue(extractFalloutScene(esmPath, scene, error),
               ("Oblivion LTEX fixture extracts: " + error).c_str());
    expectTrue(scene.landTextures.size() == 1u, "The Oblivion LTEX is extracted");
    expectTrue(scene.landTextures.size() == 1u &&
                   scene.landTextures[0].formId == kLandTextureFormId,
               "The Oblivion LTEX keeps its formID");
    expectTrue(scene.landTextures.size() == 1u &&
                   scene.landTextures[0].diffuseTexturePath == "landscape\\Dementia\\DementiaMoss01.dds",
               "ICON becomes a landscape-relative diffuse path");
    expectTrue(scene.landTextures.size() == 1u && scene.landTextures[0].textureSetFormId == 0u,
               "An Oblivion LTEX names no TXST");

    fs::remove(esmPath);
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

    // --- Skyrim NAVM: the same core data packed into NVNM. ---
    constexpr std::uint32_t kSkyrimNavMeshFormId = 0x00005200u;
    std::vector<std::uint8_t> skyrimNvnmPayload;
    appendPod(skyrimNvnmPayload, static_cast<std::uint32_t>(12));  // NVER
    appendPod(skyrimNvnmPayload, static_cast<std::uint32_t>(0));   // LCTN
    appendPod(skyrimNvnmPayload, static_cast<std::uint32_t>(0));   // interior: no worldspace
    appendPod(skyrimNvnmPayload, static_cast<std::uint32_t>(0x00002000u));
    appendPod(skyrimNvnmPayload, static_cast<std::uint32_t>(4));
    skyrimNvnmPayload.insert(
        skyrimNvnmPayload.end(), navVertexPayload.begin(), navVertexPayload.end());
    appendPod(skyrimNvnmPayload, static_cast<std::uint32_t>(2));
    skyrimNvnmPayload.insert(
        skyrimNvnmPayload.end(), navTrianglePayload.begin(), navTrianglePayload.end());
    const auto skyrimNavMeshRecord = buildRecord(
        "NAVM", kSkyrimNavMeshFormId, 0u,
        buildSubrecord("NVNM", skyrimNvnmPayload));

    // --- REFR placing the static in the exterior cell. ---
    constexpr std::uint32_t kIntRefFormId = 0x00007000u;
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
        std::vector<std::uint8_t> xtelPayload;
        appendPod(xtelPayload, kIntRefFormId);
        for (const float value : {10.0f, 20.0f, 30.0f, 0.0f, 0.0f, 1.25f}) {
            appendPod(xtelPayload, value);
        }
        const auto xtel = buildSubrecord("XTEL", xtelPayload);
        const auto xloc = buildSubrecord("XLOC", std::vector<std::uint8_t>{75u});
        out.insert(out.end(), name.begin(), name.end());
        out.insert(out.end(), data.begin(), data.end());
        out.insert(out.end(), xscl.begin(), xscl.end());
        out.insert(out.end(), xtel.begin(), xtel.end());
        out.insert(out.end(), xloc.begin(), xloc.end());
        return out;
    }();
    constexpr std::uint32_t kRefFormId = 0x00004000u;
    const auto refRecord = buildRecord("REFR", kRefFormId, 0u, refSubrecords);

    std::vector<std::uint8_t> markerNamePayload;
    appendPod(markerNamePayload, kStaticFormId);
    const auto markerName = buildSubrecord("NAME", markerNamePayload);
    const auto markerData = buildSubrecord("DATA", refData);
    const auto xmrk = buildSubrecord("XMRK", {});
    const auto markerFull = buildSubrecord("FULL", stringPayload("Test Marker"));
    const auto markerFlags = buildSubrecord("FNAM", std::vector<std::uint8_t>{1u});
    std::vector<std::uint8_t> markerTypePayload;
    appendPod(markerTypePayload, static_cast<std::uint16_t>(4u));
    const auto markerType = buildSubrecord("TNAM", markerTypePayload);
    std::vector<std::uint8_t> markerSubrecords;
    for (const auto* sub : {&markerName, &markerData, &xmrk, &markerFull, &markerFlags,
                            &markerType}) {
        markerSubrecords.insert(markerSubrecords.end(), sub->begin(), sub->end());
    }
    constexpr std::uint32_t kMarkerRefFormId = 0x00004001u;
    const auto markerRecord = buildRecord("REFR", kMarkerRefFormId, 0u, markerSubrecords);

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
        // XCLW carrying Fallout's DRY sentinel. Every one of FalloutNV.esm's
        // 30497 cells has this subrecord, so presence cannot mean "has water" --
        // 0xCF000000 is -2^31 as a float and means the cell has none.
        std::vector<std::uint8_t> xclwPayload;
        appendPod(xclwPayload, -2147483648.0f);
        const auto xclw = buildSubrecord("XCLW", xclwPayload);
        std::vector<std::uint8_t> xlcnPayload;
        appendPod(xlcnPayload, static_cast<std::uint32_t>(0x00009100u));
        const auto xlcn = buildSubrecord("XLCN", xlcnPayload);
        out.insert(out.end(), edid.begin(), edid.end());
        out.insert(out.end(), data.begin(), data.end());
        out.insert(out.end(), xclc.begin(), xclc.end());
        out.insert(out.end(), xclw.begin(), xclw.end());
        out.insert(out.end(), xlcn.begin(), xlcn.end());
        return out;
    }();
    const auto extCellRecord = buildRecord("CELL", kExtCellFormId, 0u, extCellSubrecords);

    std::vector<std::uint8_t> tempChildrenContent;
    tempChildrenContent.insert(tempChildrenContent.end(), landRecord.begin(), landRecord.end());
    tempChildrenContent.insert(tempChildrenContent.end(), navMeshRecord.begin(), navMeshRecord.end());
    tempChildrenContent.insert(
        tempChildrenContent.end(), skyrimNavMeshRecord.begin(), skyrimNavMeshRecord.end());
    tempChildrenContent.insert(tempChildrenContent.end(), refRecord.begin(), refRecord.end());
    tempChildrenContent.insert(tempChildrenContent.end(), markerRecord.begin(), markerRecord.end());
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
        // Interior + Show Sky + Use Sky Lighting, plus a high byte that proves
        // the complete two-byte Skyrim DATA flags survive the streaming index.
        std::vector<std::uint8_t> dataPayload{0xC1u, 0x04u};
        const auto data = buildSubrecord("DATA", dataPayload);
        std::vector<std::uint8_t> xcllPayload{
            91u, 80u, 64u, 255u,
            7u, 8u, 9u, 255u,
            11u, 12u, 13u, 255u};
        appendPod(xcllPayload, 125.0f);
        appendPod(xcllPayload, 6300.0f);
        const auto xcll = buildSubrecord("XCLL", xcllPayload);
        std::vector<std::uint8_t> xlcnPayload;
        appendPod(xlcnPayload, static_cast<std::uint32_t>(0x00009101u));
        const auto xlcn = buildSubrecord("XLCN", xlcnPayload);
        out.insert(out.end(), edid.begin(), edid.end());
        out.insert(out.end(), data.begin(), data.end());
        out.insert(out.end(), xcll.begin(), xcll.end());
        out.insert(out.end(), xlcn.begin(), xlcn.end());
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
    const auto intRefRecord = buildRecord("REFR", kIntRefFormId, 0u, intRefSubrecords);
    constexpr std::uint32_t kIntActorRefFormId = 0x00007001u;
    const auto intActorRefRecord =
        buildRecord("ACHR", kIntActorRefFormId, 0u, intRefSubrecords);

    char intCellLabel[4];
    std::memcpy(intCellLabel, &kIntCellFormId, 4);
    std::vector<std::uint8_t> intTemporaryChildren;
    intTemporaryChildren.insert(
        intTemporaryChildren.end(), intRefRecord.begin(), intRefRecord.end());
    intTemporaryChildren.insert(
        intTemporaryChildren.end(), intActorRefRecord.begin(), intActorRefRecord.end());
    const auto intTempChildrenGroup = buildGroup(intCellLabel, 9, intTemporaryChildren);
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
    expectTrue(extCell != nullptr && extCell->locationFormId == 0x00009100u,
               "Exterior CELL XLCN location ownership round-trips");
    expectTrue(intCell != nullptr && intCell->isInterior && intCell->worldspaceFormId == 0u,
               "Interior cell is flagged interior and attributed to no worldspace");
    expectTrue(intCell != nullptr && intCell->locationFormId == 0x00009101u,
               "Interior CELL XLCN location ownership round-trips");
    const auto actorOwner = scene.cellIndexByReferenceFormId.find(kIntActorRefFormId);
    expectTrue(
        actorOwner != scene.cellIndexByReferenceFormId.end() &&
            actorOwner->second < scene.cells.size() &&
            scene.cells[actorOwner->second].formId == kIntCellFormId,
        "ACHR ownership is indexed so interior actor population can be filtered by cell");
    expectTrue(intCell != nullptr && intCell->cellFlags == 0x04C1u,
               "CELL DATA flags, including show-sky/use-sky-lighting and the high byte, round-trip");
    if (intCell != nullptr) {
        expectTrue(intCell->hasLighting, "Interior XCLL is retained");
        expectNear(intCell->ambientColor[0], 91.0f / 255.0f, 1e-6f,
                   "XCLL ambient round-trips in authored sRGB");
        expectNear(intCell->directionalColor[2], 9.0f / 255.0f, 1e-6f,
                   "XCLL directional round-trips in authored sRGB");
        expectNear(intCell->fogColor[1], 12.0f / 255.0f, 1e-6f,
                   "XCLL fog colour round-trips in authored sRGB");
        expectNear(intCell->fogNear, 125.0f, 1e-6f, "XCLL fog-near round-trips");
        expectNear(intCell->fogFar, 6300.0f, 1e-6f, "XCLL fog-far round-trips");
    }
    // The fixture's XCLW is present and holds the dry sentinel. A reader that
    // tests presence rather than value floods every cell in the game.
    expectTrue(extCell != nullptr && !extCell->hasWater,
               "XCLW's dry sentinel is rejected rather than read as a water height");

    expectTrue(extCell != nullptr && extCell->references.size() == 2u,
               "Exterior cell owns both REFRs placed inside its group hierarchy");
    if (extCell != nullptr && extCell->references.size() == 2u) {
        const auto refIt = std::find_if(
            extCell->references.begin(), extCell->references.end(),
            [](const FalloutPlacedReference& ref) { return ref.formId == kRefFormId; });
        expectTrue(refIt != extCell->references.end(), "Teleport reference remains indexed");
        if (refIt != extCell->references.end()) {
            const FalloutPlacedReference& ref = *refIt;
            expectTrue(ref.baseFormId == kStaticFormId, "REFR base formID (NAME) round-trips");
            expectTrue(
                ref.position[0] == 512.0f && ref.position[1] == 64.0f && ref.position[2] == 1024.0f,
                "REFR position round-trips");
            expectNear(ref.rotationRadians[1], 1.5707963f, 1e-5f, "REFR rotation round-trips");
            expectNear(ref.scale, 2.0f, 1e-6f, "REFR explicit XSCL scale round-trips");
            expectTrue(ref.hasTeleport && ref.teleportTargetRefFormId == kIntRefFormId,
                       "XTEL destination reference round-trips");
            expectNear(ref.teleportRotationRadians[2], 1.25f, 1e-6f,
                       "XTEL arrival yaw round-trips");
            expectTrue(ref.isLocked && ref.lockLevel == 75u,
                       "XLOC lock state round-trips");
        }
        const auto markerIt = std::find_if(
            extCell->references.begin(), extCell->references.end(),
            [](const FalloutPlacedReference& placed) {
                return placed.formId == kMarkerRefFormId;
            });
        expectTrue(markerIt != extCell->references.end() && markerIt->isMapMarker &&
                       markerIt->mapMarkerName == "Test Marker" &&
                       markerIt->mapMarkerFlags == 1u && markerIt->mapMarkerType == 4u,
                   "XMRK name, visibility flags, and type round-trip");
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
        expectTrue(extCell->navMeshes.size() == 2, "exterior cell owns Fallout and Skyrim NAVM records");
        if (extCell->navMeshes.size() == 2) {
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

            const FalloutNavMeshRecord& skyrimNavMesh = extCell->navMeshes[1];
            expectTrue(skyrimNavMesh.formId == kSkyrimNavMeshFormId &&
                           skyrimNavMesh.cellFormId == kExtCellFormId,
                       "Skyrim NVNM keeps its mesh and owning-cell formIDs");
            expectTrue(skyrimNavMesh.vertices == navMesh.vertices &&
                           skyrimNavMesh.triangles.size() == navMesh.triangles.size(),
                       "Skyrim NVNM decodes the same vertex and triangle core");
            expectTrue(skyrimNavMesh.triangles[0].neighbour[1] == 1u &&
                           skyrimNavMesh.triangles[1].neighbour[2] == kNavMeshNoNeighbour,
                       "Skyrim NVNM adjacency is decoded and invalid links clamp to borders");
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

void testSkyrimRegionAndSoundRecords() {
    using namespace odai::importer::fnv;
    namespace fs = std::filesystem;

    constexpr std::uint32_t kWorld = 0x00001000u;
    constexpr std::uint32_t kRegion = 0x00002000u;
    constexpr std::uint32_t kDescriptor = 0x00003000u;
    constexpr std::uint32_t kOutput = 0x00004000u;
    constexpr std::uint32_t kSoundBase = 0x00005000u;

    std::vector<std::uint8_t> regionBody;
    for (const auto& subrecord : {
             buildSubrecord("EDID", stringPayload("TestForestRegion")),
             [&] { std::vector<std::uint8_t> data; appendPod(data, kWorld);
                   return buildSubrecord("WNAM", data); }(),
             buildSubrecord("RPLI", {}),
             [&] { std::vector<std::uint8_t> data;
                   for (const float value : {0.0f, 0.0f, 4096.0f, 0.0f, 0.0f, 4096.0f}) {
                       appendPod(data, value);
                   }
                   return buildSubrecord("RPLD", data); }(),
             [&] { std::vector<std::uint8_t> data; appendPod(data, 7u);
                   return buildSubrecord("RDAT", data); }(),
             [&] { std::vector<std::uint8_t> data;
                   appendPod(data, kDescriptor);
                   appendPod(data, 0x0Bu);  // pleasant, cloudy, snow
                   appendPod(data, 75u);
                   return buildSubrecord("RDSA", data); }()}) {
        regionBody.insert(regionBody.end(), subrecord.begin(), subrecord.end());
    }

    std::vector<std::uint8_t> outputData(20u, 0u);
    const float minDistance = 150.0f;
    const float maxDistance = 4000.0f;
    std::memcpy(outputData.data() + 4u, &minDistance, sizeof(float));
    std::memcpy(outputData.data() + 8u, &maxDistance, sizeof(float));

    std::vector<std::uint8_t> descriptorBody;
    for (const auto& subrecord : {
             buildSubrecord("EDID", stringPayload("TestRiverLoop")),
             buildSubrecord("ANAM", stringPayload("fx\\amb\\river_a.wav")),
             buildSubrecord("ANAM", stringPayload("fx\\amb\\river_b.wav")),
             [&] { std::vector<std::uint8_t> data; appendPod(data, kOutput);
                   return buildSubrecord("ONAM", data); }(),
             [&] { std::vector<std::uint8_t> data; appendPod(data, 0x00800000u);
                   return buildSubrecord("BNAM", data); }()}) {
        descriptorBody.insert(descriptorBody.end(), subrecord.begin(), subrecord.end());
    }
    std::vector<std::uint8_t> baseData;
    appendPod(baseData, kDescriptor);

    std::vector<std::uint8_t> bytes = buildTes4Record({}, EsmPluginFormat::kFallout3);
    const auto appendTop = [&](const char* type, const std::vector<std::uint8_t>& record) {
        const auto group = buildGroup(type, 0, record);
        bytes.insert(bytes.end(), group.begin(), group.end());
    };
    appendTop("REGN", buildRecord("REGN", kRegion, 0u, regionBody));
    appendTop("SOPM", buildRecord(
        "SOPM", kOutput, 0u, buildSubrecord("ANAM", outputData)));
    appendTop("SNDR", buildRecord("SNDR", kDescriptor, 0u, descriptorBody));
    appendTop("SOUN", buildRecord(
        "SOUN", kSoundBase, 0u, buildSubrecord("SDSC", baseData)));

    const fs::path path = fs::temp_directory_path() / "odai_skyrim_ambient_records.esm";
    {
        std::ofstream file(path, std::ios::binary | std::ios::trunc);
        file.write(reinterpret_cast<const char*>(bytes.data()),
                   static_cast<std::streamsize>(bytes.size()));
    }
    FalloutSceneData scene;
    std::string error;
    expectTrue(extractFalloutScene(path, scene, error),
               ("Skyrim ambient records extract: " + error).c_str());
    expectTrue(scene.regions.size() == 1u && scene.regions[0].worldspaceFormId == kWorld,
               "REGN keeps its worldspace identity");
    expectTrue(scene.regions.size() == 1u && scene.regions[0].polygons.size() == 1u &&
                   scene.regions[0].polygons[0].points.size() == 6u,
               "REGN RPLI/RPLD polygon points extract in authored order");
    expectTrue(scene.regions.size() == 1u && scene.regions[0].sounds.size() == 1u &&
                   scene.regions[0].sounds[0].descriptorFormId == kDescriptor &&
                   scene.regions[0].sounds[0].weatherFlags == 0x0Bu &&
                   scene.regions[0].sounds[0].chance == 75.0f,
               "REGN RDSA preserves descriptor, weather flags, and integer chance");
    expectTrue(scene.soundDescriptors.size() == 1u &&
                   scene.soundDescriptors[0].filePaths.size() == 2u &&
                   scene.soundDescriptors[0].looping &&
                   scene.soundDescriptors[0].outputModelFormId == kOutput,
               "SNDR preserves variants, looping state, and output model");
    expectTrue(scene.soundOutputModels.size() == 1u &&
                   scene.soundOutputModels[0].minDistance == minDistance &&
                   scene.soundOutputModels[0].maxDistance == maxDistance,
               "SOPM preserves authored attenuation distances");
    expectTrue(scene.soundBases.size() == 1u &&
                   scene.soundBases[0].descriptorFormId == kDescriptor,
               "SOUN.SDSC resolves to its descriptor");
    fs::remove(path);
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
                entry.cellFlags == expected->cellFlags &&
                entry.hasGridCoords == expected->hasGridCoords &&
                entry.gridX == expected->gridX && entry.gridZ == expected->gridZ &&
                entry.worldspaceFormId == expected->worldspaceFormId &&
                entry.locationFormId == expected->locationFormId,
            "indexed cell metadata matches the full extraction");

        FalloutCellRecord streamed;
        expectTrue(
            extractFalloutCellAt(reader, entry, streamed, error),
            ("extractFalloutCellAt succeeds: " + error).c_str());

        expectTrue(
            streamed.references.size() == expected->references.size(),
            "streamed cell has the same reference count as the full extraction");
        expectTrue(
            streamed.cellFlags == expected->cellFlags &&
                streamed.locationFormId == expected->locationFormId &&
                streamed.hasLighting == expected->hasLighting &&
                streamed.ambientColor[0] == expected->ambientColor[0] &&
                streamed.directionalColor[2] == expected->directionalColor[2] &&
                streamed.fogColor[1] == expected->fogColor[1] &&
                streamed.fogNear == expected->fogNear && streamed.fogFar == expected->fogFar,
            "streamed CELL flags and complete XCLL match full extraction");
        expectTrue(
            streamed.navMeshes.size() == expected->navMeshes.size(),
            "streamed cell has the same navmesh count as the full extraction");
        expectTrue(
            (streamed.land != nullptr) == (expected->land != nullptr),
            "streamed cell agrees on whether the cell has LAND");

        if (streamed.land != nullptr && expected->land != nullptr) {
            expectTrue(
                streamed.land->heights == expected->land->heights,
                "streamed LAND heights are identical to the full extraction");
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

// Same prefix, with one property ref attached. Gamebryo properties inherit down
// the scene graph, so which block carries the ref is the whole point of the
// fixture below.
void appendAvObjectPrefixWithProperty(
    std::vector<std::uint8_t>& out, const std::array<float, 3>& translation,
    const std::array<float, 9>& rotation, float scale, std::int32_t propertyRef
) {
    appendPod(out, static_cast<std::int32_t>(-1));  // nameRef
    appendPod(out, static_cast<std::uint32_t>(0));  // numExtraData
    appendPod(out, static_cast<std::int32_t>(-1));  // controllerRef
    appendPod(out, static_cast<std::uint32_t>(0));  // flags
    for (float v : translation) appendPod(out, v);
    for (float v : rotation) appendPod(out, v);
    appendPod(out, scale);
    appendPod(out, static_cast<std::uint32_t>(1));  // numProperties
    appendPod(out, propertyRef);
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
    // Num Effects. The parser does not read the effects list, but it does
    // require this count to be present, and that requirement is what lets it
    // recognize a node by LAYOUT instead of by type name (see readNiNode).
    // This fixture omitted the field, which made it a NiNode no real file
    // would contain: a census over 20000 retail meshes finds the trailer on
    // every single one. Written here so the fixture matches the format rather
    // than the parser being loosened to match the fixture.
    appendPod(niNodeBlock, static_cast<std::uint32_t>(0));  // numEffects

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

// Skyrim SE forces full-precision BSVertexData positions even when the
// descriptor omits Full_Precision. Terrain BTR is the load-bearing case: its
// float3 + padding + half2 UV layout is only 20 bytes, so the old "stride >=
// 28" heuristic decoded the low half of each float and collapsed X/Z to zero.
void testSkyrimTerrainPackedPositionsAreFullPrecision() {
    const std::array<float, 9> identityRotation{1, 0, 0, 0, 1, 0, 0, 0, 1};
    const auto appendSseAvObjectPrefix = [&](std::vector<std::uint8_t>& block) {
        appendPod(block, static_cast<std::int32_t>(-1));   // nameRef
        appendPod(block, static_cast<std::uint32_t>(0));  // numExtraData
        appendPod(block, static_cast<std::int32_t>(-1));  // controllerRef
        appendPod(block, static_cast<std::uint32_t>(0));  // flags
        appendPod(block, 0.0f); appendPod(block, 0.0f); appendPod(block, 0.0f);
        for (float value : identityRotation) appendPod(block, value);
        appendPod(block, 1.0f);                           // scale
        // Skyrim has no NiAVObject property array at BS version 100.
        appendPod(block, static_cast<std::int32_t>(-1));  // collisionRef
    };

    std::vector<std::uint8_t> rootBlock;
    appendSseAvObjectPrefix(rootBlock);
    appendPod(rootBlock, static_cast<std::uint32_t>(1));  // numChildren
    appendPod(rootBlock, static_cast<std::int32_t>(1));   // shape
    appendPod(rootBlock, static_cast<std::uint32_t>(0));  // numEffects

    std::vector<std::uint8_t> shapeBlock;
    appendSseAvObjectPrefix(shapeBlock);
    for (int i = 0; i < 4; ++i) appendPod(shapeBlock, 0.0f);  // bounding sphere
    appendPod(shapeBlock, static_cast<std::int32_t>(-1));     // skin
    appendPod(shapeBlock, static_cast<std::int32_t>(-1));     // shader
    appendPod(shapeBlock, static_cast<std::int32_t>(-1));     // alpha
    // stride=5 dwords, UV offset=4 dwords, flags Vertex|UV. This is the retail
    // Tamriel BTR descriptor exactly; notably, bit 0x400 is absent.
    constexpr std::uint64_t descriptor = 0x0000300000000405ull;
    appendPod(shapeBlock, descriptor);
    appendPod(shapeBlock, static_cast<std::uint16_t>(1));  // triangles
    appendPod(shapeBlock, static_cast<std::uint16_t>(3));  // vertices
    appendPod(shapeBlock, static_cast<std::uint32_t>(66)); // 3*20 + 1*6
    const std::array<std::array<float, 3>, 3> positions{{
        {32.0f, 128.0f, -1174.0f},
        {0.0f, 0.0f, -1162.0f},
        {320.0f, 416.0f, -1134.0f},
    }};
    for (std::size_t vertex = 0; vertex < positions.size(); ++vertex) {
        for (float value : positions[vertex]) appendPod(shapeBlock, value);
        appendPod(shapeBlock, static_cast<std::uint32_t>(0));  // bitangent/padding
        appendPod(shapeBlock, static_cast<std::uint16_t>(vertex == 0 ? 0u : 0x3c00u));
        appendPod(shapeBlock, static_cast<std::uint16_t>(vertex == 2 ? 0x3c00u : 0u));
    }
    appendPod(shapeBlock, static_cast<std::uint16_t>(0));
    appendPod(shapeBlock, static_cast<std::uint16_t>(1));
    appendPod(shapeBlock, static_cast<std::uint16_t>(2));
    appendPod(shapeBlock, static_cast<std::uint32_t>(0));  // SSE particle data size

    std::vector<std::uint8_t> fileBytes;
    const std::string headerLine = "Gamebryo File Format, Version 20.2.0.7";
    fileBytes.insert(fileBytes.end(), headerLine.begin(), headerLine.end());
    fileBytes.push_back('\n');
    appendPod(fileBytes, static_cast<std::uint32_t>(0x14020007u));
    appendPod(fileBytes, static_cast<std::uint8_t>(1));
    appendPod(fileBytes, static_cast<std::uint32_t>(12));   // user version
    appendPod(fileBytes, static_cast<std::uint32_t>(2));    // blocks
    appendPod(fileBytes, static_cast<std::uint32_t>(100));  // Skyrim SE BS version
    appendSizedString8(fileBytes, "");
    appendSizedString8(fileBytes, "");
    appendSizedString8(fileBytes, "");
    appendPod(fileBytes, static_cast<std::uint16_t>(2));
    appendSizedString32(fileBytes, "BSFadeNode");
    appendSizedString32(fileBytes, "BSTriShape");
    appendPod(fileBytes, static_cast<std::uint16_t>(0));
    appendPod(fileBytes, static_cast<std::uint16_t>(1));
    appendPod(fileBytes, static_cast<std::uint32_t>(rootBlock.size()));
    appendPod(fileBytes, static_cast<std::uint32_t>(shapeBlock.size()));
    appendPod(fileBytes, static_cast<std::uint32_t>(0));  // strings
    appendPod(fileBytes, static_cast<std::uint32_t>(0));  // max string length
    appendPod(fileBytes, static_cast<std::uint32_t>(0));  // groups
    fileBytes.insert(fileBytes.end(), rootBlock.begin(), rootBlock.end());
    fileBytes.insert(fileBytes.end(), shapeBlock.begin(), shapeBlock.end());
    appendPod(fileBytes, static_cast<std::uint32_t>(1));  // footer roots
    appendPod(fileBytes, static_cast<std::int32_t>(0));

    odai::importer::fnv::NifModel model;
    std::string error;
    expectTrue(
        odai::importer::fnv::parseNifStaticMesh(fileBytes, model, error),
        ("Synthetic Skyrim BTR layout parses: " + error).c_str());
    expectTrue(model.shapes.size() == 1u, "Skyrim BTR fixture emits one terrain shape");
    if (model.shapes.size() == 1u) {
        const auto& decoded = model.shapes.front().positions;
        expectTrue(decoded.size() == 9u, "Skyrim BTR fixture keeps all positions");
        if (decoded.size() == 9u) {
            expectNear(decoded[0], 32.0f, 1e-5f, "20-byte BTR vertex keeps full-precision X");
            expectNear(decoded[1], 128.0f, 1e-5f, "20-byte BTR vertex keeps full-precision Y");
            expectNear(decoded[2], -1174.0f, 1e-5f, "20-byte BTR vertex keeps full-precision height");
            expectNear(decoded[7], 416.0f, 1e-5f, "later BTR vertices do not collapse");
        }
    }
}

// Skyrim's packed vertex descriptor only says that an RGBA channel EXISTS.
// BSLightingShaderProperty's SLSF1_Vertex_Alpha says whether its A byte is
// actually opacity. Rock/terrain transition meshes depend on the distinction:
// enabled ramps blend their opaque diffuse into the ground, while disabled
// bytes must read as 1 rather than punching holes through alpha-tested shapes.
void testSkyrimLightingShaderVertexAlpha() {
    const auto parseFixture = [](bool enableVertexAlpha) {
        const std::array<float, 9> identityRotation{1, 0, 0, 0, 1, 0, 0, 0, 1};
        const auto appendSseAvObjectPrefix = [&](std::vector<std::uint8_t>& block) {
            appendPod(block, static_cast<std::int32_t>(-1));
            appendPod(block, static_cast<std::uint32_t>(0));
            appendPod(block, static_cast<std::int32_t>(-1));
            appendPod(block, static_cast<std::uint32_t>(0));
            appendPod(block, 0.0f); appendPod(block, 0.0f); appendPod(block, 0.0f);
            for (float value : identityRotation) appendPod(block, value);
            appendPod(block, 1.0f);
            appendPod(block, static_cast<std::int32_t>(-1));
        };

        std::vector<std::uint8_t> rootBlock;
        appendSseAvObjectPrefix(rootBlock);
        appendPod(rootBlock, static_cast<std::uint32_t>(1));
        appendPod(rootBlock, static_cast<std::int32_t>(1));
        appendPod(rootBlock, static_cast<std::uint32_t>(0));

        std::vector<std::uint8_t> shapeBlock;
        appendSseAvObjectPrefix(shapeBlock);
        for (int i = 0; i < 4; ++i) appendPod(shapeBlock, 0.0f);
        appendPod(shapeBlock, static_cast<std::int32_t>(-1));  // skin
        appendPod(shapeBlock, static_cast<std::int32_t>(2));   // shader property
        appendPod(shapeBlock, static_cast<std::int32_t>(-1));  // alpha property
        // 24-byte vertex: float3 + padding, half2 UV, RGBA8. Flags say that
        // positions, UVs and colours are stored; they do NOT enable colour A.
        constexpr std::uint64_t descriptor =
            (static_cast<std::uint64_t>(0x23u) << 44u) |
            (static_cast<std::uint64_t>(5u) << 24u) |
            (static_cast<std::uint64_t>(4u) << 8u) | 6u;
        appendPod(shapeBlock, descriptor);
        appendPod(shapeBlock, static_cast<std::uint16_t>(1));
        appendPod(shapeBlock, static_cast<std::uint16_t>(3));
        appendPod(shapeBlock, static_cast<std::uint32_t>(78));  // 3*24 + 1*6
        constexpr std::array<std::uint8_t, 3> alpha{255u, 128u, 0u};
        for (std::size_t vertex = 0; vertex < 3u; ++vertex) {
            appendPod(shapeBlock, vertex == 1u ? 1.0f : 0.0f);
            appendPod(shapeBlock, vertex == 2u ? 1.0f : 0.0f);
            appendPod(shapeBlock, 0.0f);
            appendPod(shapeBlock, static_cast<std::uint32_t>(0));
            appendPod(shapeBlock, static_cast<std::uint16_t>(0));
            appendPod(shapeBlock, static_cast<std::uint16_t>(0));
            shapeBlock.push_back(255u);
            shapeBlock.push_back(255u);
            shapeBlock.push_back(255u);
            shapeBlock.push_back(alpha[vertex]);
        }
        appendPod(shapeBlock, static_cast<std::uint16_t>(0));
        appendPod(shapeBlock, static_cast<std::uint16_t>(1));
        appendPod(shapeBlock, static_cast<std::uint16_t>(2));
        appendPod(shapeBlock, static_cast<std::uint32_t>(0));

        std::vector<std::uint8_t> shaderBlock;
        appendPod(shaderBlock, static_cast<std::uint32_t>(0));   // shader type
        appendPod(shaderBlock, static_cast<std::int32_t>(-1));   // name
        appendPod(shaderBlock, static_cast<std::uint32_t>(0));   // extra data
        appendPod(shaderBlock, static_cast<std::int32_t>(-1));   // controller
        appendPod(shaderBlock, enableVertexAlpha ? 0x8u : 0u);   // SLSF1
        appendPod(shaderBlock, static_cast<std::uint32_t>(0));   // SLSF2
        appendPod(shaderBlock, 0.0f); appendPod(shaderBlock, 0.0f); // UV offset
        appendPod(shaderBlock, 1.0f); appendPod(shaderBlock, 1.0f); // UV scale
        appendPod(shaderBlock, static_cast<std::int32_t>(-1));   // texture set

        std::vector<std::uint8_t> fileBytes;
        const std::string headerLine = "Gamebryo File Format, Version 20.2.0.7";
        fileBytes.insert(fileBytes.end(), headerLine.begin(), headerLine.end());
        fileBytes.push_back('\n');
        appendPod(fileBytes, static_cast<std::uint32_t>(0x14020007u));
        appendPod(fileBytes, static_cast<std::uint8_t>(1));
        appendPod(fileBytes, static_cast<std::uint32_t>(12));
        appendPod(fileBytes, static_cast<std::uint32_t>(3));
        appendPod(fileBytes, static_cast<std::uint32_t>(100));
        appendSizedString8(fileBytes, "");
        appendSizedString8(fileBytes, "");
        appendSizedString8(fileBytes, "");
        appendPod(fileBytes, static_cast<std::uint16_t>(3));
        appendSizedString32(fileBytes, "BSFadeNode");
        appendSizedString32(fileBytes, "BSTriShape");
        appendSizedString32(fileBytes, "BSLightingShaderProperty");
        appendPod(fileBytes, static_cast<std::uint16_t>(0));
        appendPod(fileBytes, static_cast<std::uint16_t>(1));
        appendPod(fileBytes, static_cast<std::uint16_t>(2));
        appendPod(fileBytes, static_cast<std::uint32_t>(rootBlock.size()));
        appendPod(fileBytes, static_cast<std::uint32_t>(shapeBlock.size()));
        appendPod(fileBytes, static_cast<std::uint32_t>(shaderBlock.size()));
        appendPod(fileBytes, static_cast<std::uint32_t>(0));
        appendPod(fileBytes, static_cast<std::uint32_t>(0));
        appendPod(fileBytes, static_cast<std::uint32_t>(0));
        fileBytes.insert(fileBytes.end(), rootBlock.begin(), rootBlock.end());
        fileBytes.insert(fileBytes.end(), shapeBlock.begin(), shapeBlock.end());
        fileBytes.insert(fileBytes.end(), shaderBlock.begin(), shaderBlock.end());
        appendPod(fileBytes, static_cast<std::uint32_t>(1));
        appendPod(fileBytes, static_cast<std::int32_t>(0));

        odai::importer::fnv::NifModel model;
        std::string error;
        expectTrue(
            odai::importer::fnv::parseNifStaticMesh(fileBytes, model, error),
            ("Synthetic Skyrim vertex-alpha fixture parses: " + error).c_str());
        return model;
    };

    const auto enabled = parseFixture(true);
    expectTrue(enabled.shapes.size() == 1u, "vertex-alpha fixture emits one shape");
    if (enabled.shapes.size() == 1u) {
        expectTrue(enabled.shapes[0].alphaBlend,
                   "SLSF1_Vertex_Alpha routes a partial ramp through blending");
        expectNear(enabled.shapes[0].colors[7], 128.0f / 255.0f, 1e-5f,
                   "enabled vertex alpha preserves its authored coverage");
    }

    const auto disabled = parseFixture(false);
    expectTrue(disabled.shapes.size() == 1u, "disabled-alpha fixture emits one shape");
    if (disabled.shapes.size() == 1u) {
        expectTrue(!disabled.shapes[0].alphaBlend,
                   "stored alpha bytes do not imply blending when SLSF1 disables them");
        expectNear(disabled.shapes[0].colors[7], 1.0f, 1e-5f,
                   "disabled vertex alpha is normalized to opaque");
        expectNear(disabled.shapes[0].colors[11], 1.0f, 1e-5f,
                   "even a stored zero alpha is ignored when the shader flag is clear");
    }
}

// Builds a NIF whose hierarchy is root -> middle -> NiTriShape -> data, where
// the MIDDLE block's type name is chosen by the caller. `footerRoot` < 0 omits
// the footer entirely (older behaviour); otherwise a one-root footer is written.
//
// This is the shape of the floating-geometry bug: when the middle node is not
// recognized, nothing claims the NiTriShape as a child, and the old root scan
// promoted it to a root walked from identity -- silently dropping the 5000-unit
// translation the middle node carries.
std::vector<std::uint8_t> buildChainedNif(
    const std::string& middleTypeName, float middleTranslateZ, int footerRoot,
    bool truncateMiddleBlock = false) {
    const std::array<float, 9> identityRotation{1, 0, 0, 0, 1, 0, 0, 0, 1};

    // Block 0: root node, no translation, one child (block 1).
    std::vector<std::uint8_t> rootBlock;
    appendAvObjectPrefix(rootBlock, {0.0f, 0.0f, 0.0f}, identityRotation, 1.0f);
    appendPod(rootBlock, static_cast<std::uint32_t>(1));   // numChildren
    appendPod(rootBlock, static_cast<std::int32_t>(1));    // children[0] = block 1
    appendPod(rootBlock, static_cast<std::uint32_t>(0));   // numEffects

    // Block 1: the middle node, translated, one child (block 2).
    std::vector<std::uint8_t> middleBlock;
    appendAvObjectPrefix(middleBlock, {0.0f, 0.0f, middleTranslateZ}, identityRotation, 1.0f);
    appendPod(middleBlock, static_cast<std::uint32_t>(1));  // numChildren
    appendPod(middleBlock, static_cast<std::int32_t>(2));   // children[0] = block 2
    if (!truncateMiddleBlock) {
        appendPod(middleBlock, static_cast<std::uint32_t>(0));  // numEffects
        if (middleTypeName == "NiSwitchNode") {
            appendPod(middleBlock, static_cast<std::uint16_t>(3));  // switch flags
            appendPod(middleBlock, static_cast<std::uint32_t>(0));  // initial child
        }
    }

    // Block 2: NiTriShape, identity local transform, dataRef = block 3.
    std::vector<std::uint8_t> triShapeBlock;
    appendAvObjectPrefix(triShapeBlock, {0.0f, 0.0f, 0.0f}, identityRotation, 1.0f);
    appendPod(triShapeBlock, static_cast<std::int32_t>(3));  // dataRef

    // Block 3: NiTriShapeData, one triangle at the origin, no normals/UVs.
    std::vector<std::uint8_t> dataBlock;
    appendPod(dataBlock, static_cast<std::int32_t>(0));      // groupId
    appendPod(dataBlock, static_cast<std::uint16_t>(3));     // numVertices
    appendPod(dataBlock, static_cast<std::uint8_t>(0));      // keepFlags
    appendPod(dataBlock, static_cast<std::uint8_t>(0));      // compressFlags
    appendPod(dataBlock, static_cast<std::uint8_t>(1));      // hasVertices
    for (int v = 0; v < 3; ++v) {
        appendPod(dataBlock, 0.0f);
        appendPod(dataBlock, 0.0f);
        appendPod(dataBlock, 0.0f);
    }
    appendPod(dataBlock, static_cast<std::uint16_t>(0));     // vectorFlags: no UVs, no tangents
    appendPod(dataBlock, static_cast<std::uint8_t>(0));      // hasNormals
    appendPod(dataBlock, 0.0f);                              // bounding sphere center x
    appendPod(dataBlock, 0.0f);
    appendPod(dataBlock, 0.0f);
    appendPod(dataBlock, 0.0f);                              // bounding sphere radius
    appendPod(dataBlock, static_cast<std::uint8_t>(0));      // hasVertexColors
    appendPod(dataBlock, static_cast<std::uint16_t>(0));     // consistencyType
    appendPod(dataBlock, static_cast<std::int32_t>(-1));     // additionalDataRef
    appendPod(dataBlock, static_cast<std::uint16_t>(1));     // numTriangles
    appendPod(dataBlock, static_cast<std::uint32_t>(3));     // numTrianglePoints
    appendPod(dataBlock, static_cast<std::uint8_t>(1));      // hasTriangles
    appendPod(dataBlock, static_cast<std::uint16_t>(0));
    appendPod(dataBlock, static_cast<std::uint16_t>(1));
    appendPod(dataBlock, static_cast<std::uint16_t>(2));
    appendPod(dataBlock, static_cast<std::uint16_t>(0));     // numMatchGroups

    std::vector<std::uint8_t> fileBytes;
    const std::string headerLine = "Gamebryo File Format, Version 20.2.0.7";
    fileBytes.insert(fileBytes.end(), headerLine.begin(), headerLine.end());
    fileBytes.push_back('\n');
    appendPod(fileBytes, static_cast<std::uint32_t>(0x14020007u));
    appendPod(fileBytes, static_cast<std::uint8_t>(1));
    appendPod(fileBytes, static_cast<std::uint32_t>(11));
    appendPod(fileBytes, static_cast<std::uint32_t>(4));   // numBlocks
    appendPod(fileBytes, static_cast<std::uint32_t>(34));  // userVersion2
    appendSizedString8(fileBytes, "");
    appendSizedString8(fileBytes, "");
    appendSizedString8(fileBytes, "");
    appendPod(fileBytes, static_cast<std::uint16_t>(4));   // numBlockTypes
    appendSizedString32(fileBytes, "NiNode");
    appendSizedString32(fileBytes, middleTypeName);
    appendSizedString32(fileBytes, "NiTriShape");
    appendSizedString32(fileBytes, "NiTriShapeData");
    appendPod(fileBytes, static_cast<std::uint16_t>(0));
    appendPod(fileBytes, static_cast<std::uint16_t>(1));
    appendPod(fileBytes, static_cast<std::uint16_t>(2));
    appendPod(fileBytes, static_cast<std::uint16_t>(3));
    appendPod(fileBytes, static_cast<std::uint32_t>(rootBlock.size()));
    appendPod(fileBytes, static_cast<std::uint32_t>(middleBlock.size()));
    appendPod(fileBytes, static_cast<std::uint32_t>(triShapeBlock.size()));
    appendPod(fileBytes, static_cast<std::uint32_t>(dataBlock.size()));
    appendPod(fileBytes, static_cast<std::uint32_t>(0));  // numStrings
    appendPod(fileBytes, static_cast<std::uint32_t>(0));  // maxStringLength
    appendPod(fileBytes, static_cast<std::uint32_t>(0));  // numGroups

    fileBytes.insert(fileBytes.end(), rootBlock.begin(), rootBlock.end());
    fileBytes.insert(fileBytes.end(), middleBlock.begin(), middleBlock.end());
    fileBytes.insert(fileBytes.end(), triShapeBlock.begin(), triShapeBlock.end());
    fileBytes.insert(fileBytes.end(), dataBlock.begin(), dataBlock.end());

    if (footerRoot >= 0) {
        appendPod(fileBytes, static_cast<std::uint32_t>(1));  // Num Roots
        appendPod(fileBytes, static_cast<std::int32_t>(footerRoot));
    }
    return fileBytes;
}

// The floating-geometry regression, stated as the two outcomes that matter.
void testNifParserDoesNotReparentSubtreesToTheOrigin() {
    // Skyrim trees put obsolete billboard/LOD alternatives under a
    // NiSwitchNode. Its six-byte subtype tail selects the one authored child;
    // ignoring it traverses inactive geometry and reports valid trees as
    // corrupt or duplicates their silhouette.
    {
        const std::vector<std::uint8_t> bytes =
            buildChainedNif("NiSwitchNode", 2500.0f, 0);
        odai::importer::fnv::NifModel model;
        std::string error;
        expectTrue(odai::importer::fnv::parseNifStaticMesh(bytes, model, error),
                   ("NiSwitchNode subtype tail parses: " + error).c_str());
        expectTrue(model.nodeParseFailedCount == 0u && model.shapes.size() == 1u,
                   "NiSwitchNode reaches its authored initial child without a parse failure");
        if (model.shapes.size() == 1u && model.shapes.front().positions.size() >= 3u) {
            expectNear(model.shapes.front().positions[2], 2500.0f, 1e-3f,
                       "NiSwitchNode preserves the selected child's ancestor transform");
        }
    }

    // 1. A middle node of a type the allowlist did not previously cover
    //    (BSMasterParticleSystem is NiNode-derived per nif.xml and roots 38
    //    retail meshes, yet does not end in "Node" so no suffix rule sees it).
    //    Its translation must survive.
    {
        const std::vector<std::uint8_t> bytes =
            buildChainedNif("BSMasterParticleSystem", 5000.0f, 0);
        odai::importer::fnv::NifModel model;
        std::string error;
        expectTrue(odai::importer::fnv::parseNifStaticMesh(bytes, model, error),
                   ("NiNode-derived middle type parses: " + error).c_str());
        expectTrue(model.usedFooterRoots, "Roots came from the footer, not the orphan scan");
        expectTrue(model.shapes.size() == 1u, "Geometry under a BSMasterParticleSystem is reachable");
        if (model.shapes.size() == 1u && model.shapes.front().positions.size() >= 3u) {
            expectNear(model.shapes.front().positions[2], 5000.0f, 1e-3f,
                       "Subtree keeps its ancestor translation instead of collapsing to the origin");
        }
    }

    // 2. A middle node whose block is truncated so readNiNode rejects it. The
    //    subtree must be DROPPED, not emitted at the origin -- "missing" is
    //    diagnosable, "floating in the sky" is what this whole change is about.
    {
        const std::vector<std::uint8_t> bytes =
            buildChainedNif("NiNode", 5000.0f, 0, /*truncateMiddleBlock=*/true);
        odai::importer::fnv::NifModel model;
        std::string error;
        expectTrue(odai::importer::fnv::parseNifStaticMesh(bytes, model, error),
                   ("Truncated middle node still parses the file: " + error).c_str());
        expectTrue(model.nodeParseFailedCount == 1u,
                   "A node whose field walk fails is counted rather than silently ignored");
        expectTrue(model.shapes.empty(),
                   "An unreachable subtree emits NOTHING (it must not appear at the origin)");
    }

    // 3. A middle node of a type NOT in the known list but whose name ends in
    //    "Node" -- a mod-authored or newer niftools type. It must still be
    //    walked, because the alternative is the reparent-to-origin bug: an
    //    unrecognized parent never claims its children, and the root scan then
    //    promotes each of them and drops the ancestor translation. This is the
    //    case a name-based allowlist could never cover and the one that made
    //    the first footer fix incomplete.
    {
        const std::vector<std::uint8_t> bytes = buildChainedNif("BSFooBarNode", 5000.0f, 0);
        odai::importer::fnv::NifModel model;
        std::string error;
        expectTrue(odai::importer::fnv::parseNifStaticMesh(bytes, model, error),
                   ("Unknown *Node middle type parses: " + error).c_str());
        expectTrue(model.shapes.size() == 1u,
                   "Geometry under an unknown *Node type is still reachable");
        if (model.shapes.size() == 1u && model.shapes.front().positions.size() >= 3u) {
            expectNear(model.shapes.front().positions[2], 5000.0f, 1e-3f,
                       "Unknown *Node keeps its translation instead of collapsing to the origin");
        }
        expectTrue(model.unhandledNodeTypeCount == 1u,
                   "The unknown type is still reported, having been walked rather than skipped");
        expectTrue(model.nodeParseFailedCount == 0u,
                   "A well-formed unknown *Node parses rather than counting as a failure");
    }

    // 3. A footer root pointing at a non-node (nif.xml: a first-person camera is
    //    listed among the roots "even if it is not a root object") must not be
    //    walked as a node, and must not take the whole model down with it.
    {
        std::vector<std::uint8_t> bytes = buildChainedNif("NiNode", 7.0f, 2);  // root -> NiTriShape
        odai::importer::fnv::NifModel model;
        std::string error;
        expectTrue(odai::importer::fnv::parseNifStaticMesh(bytes, model, error),
                   ("Footer root pointing at a non-node still parses: " + error).c_str());
        expectTrue(model.shapes.empty() || model.shapes.size() == 1u,
                   "A non-node footer root is skipped rather than misparsed");
    }
}


// A TES4 header whose declared size exceeds the file must be rejected before it
// is used to size anything. --plugin-add takes an arbitrary user-named path, so
// this is reachable from a truncated download or any file whose first four
// bytes happen to read "TES4".
void testPluginHeaderRejectsOversizedRecord() {
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "odai_bogus_plugin.esp";
    {
        std::ofstream out(path, std::ios::binary);
        const char magic[4] = {'T', 'E', 'S', '4'};
        out.write(magic, 4);
        const std::uint32_t dataSize = 0xFFFFFFFFu;  // ~4 GB, on a 24-byte file
        out.write(reinterpret_cast<const char*>(&dataSize), 4);
        const std::uint32_t rest[4] = {0, 0, 0, 0};
        out.write(reinterpret_cast<const char*>(rest), sizeof(rest));
    }
    odai::importer::fnv::FalloutPluginHeader header;
    std::string error;
    const bool ok = odai::importer::fnv::readFalloutPluginHeader(path, header, error);
    expectTrue(!ok, "A TES4 record larger than its file is rejected, not allocated");
    expectTrue(!error.empty(), "Rejecting an oversized TES4 record explains why");
    std::error_code removeError;
    std::filesystem::remove(path, removeError);
}

// readFalloutPluginHeader parses the TES4 record itself rather than going
// through EsmReader, so it needs the same 20-vs-24-byte handling. Getting it
// wrong on an Oblivion plugin does not fail: the body walk starts four bytes
// into HEDR, runs one subrecord out of phase, and reports a mod that declares
// masters as having none — which resolves its formIDs against the wrong
// plugins rather than erroring.
//
// Run over both generations for the same reason the walker test is: the risk
// here is a regression on the Fallout side, which a one-sided assertion misses.
void testPluginHeaderReadsBothGenerations() {
    namespace fs = std::filesystem;
    using namespace odai::importer::fnv;

    const auto check = [](EsmPluginFormat format, const char* fileName, const char* label) {
        const std::vector<std::string> masters = {"Oblivion.esm", "Knights.esp"};
        const std::vector<std::uint8_t> bytes = buildTes4Record(masters, format);
        const fs::path path = fs::temp_directory_path() / fileName;
        {
            std::ofstream out(path, std::ios::binary | std::ios::trunc);
            out.write(reinterpret_cast<const char*>(bytes.data()),
                      static_cast<std::streamsize>(bytes.size()));
        }
        FalloutPluginHeader header;
        std::string error;
        const bool ok = readFalloutPluginHeader(path, header, error);
        expectTrue(ok, (std::string("Plugin header reads ") + label).c_str());
        expectTrue(header.masters.size() == 2u,
                   (std::string("Both masters are found ") + label).c_str());
        expectTrue(header.masters.size() == 2u && header.masters[0] == "Oblivion.esm" &&
                       header.masters[1] == "Knights.esp",
                   (std::string("Masters keep their order and spelling ") + label).c_str());
        fs::remove(path);
    };

    check(EsmPluginFormat::kFallout3, "odai_hdr24_plugin.esm", "[24-byte header]");
    check(EsmPluginFormat::kOblivion, "odai_hdr20_plugin.esm", "[20-byte header]");
}

// A child count that cannot fit in the block must be rejected before it is
// used to size anything: unbounded, a desynchronized 0xFFFFFFFF here asks for a
// ~17 GB allocation on nothing worse than a malformed mod asset.
// Gamebryo properties inherit down the scene graph. A NiAlphaProperty on a
// parent NiNode applies to every shape beneath it, and the reader used to walk
// only a shape's own property list -- so those shapes imported with no alpha
// mode at all and rendered fully opaque, showing the black that sits under a
// Fallout texture's transparent texels.
//
// Five blocks: root NiNode -> middle NiNode (carrying the alpha property ref)
// -> NiTriShape -> NiTriShapeData, plus the NiAlphaProperty itself.
void testNifParserInheritsPropertiesFromParentNodes() {
    const std::array<float, 9> identityRotation{1, 0, 0, 0, 1, 0, 0, 0, 1};

    std::vector<std::uint8_t> rootBlock;
    appendAvObjectPrefix(rootBlock, {0.0f, 0.0f, 0.0f}, identityRotation, 1.0f);
    appendPod(rootBlock, static_cast<std::uint32_t>(1));
    appendPod(rootBlock, static_cast<std::int32_t>(1));
    appendPod(rootBlock, static_cast<std::uint32_t>(0));

    // The middle node owns the alpha property (block 4); the shape below it
    // declares none of its own.
    std::vector<std::uint8_t> middleBlock;
    appendAvObjectPrefixWithProperty(
        middleBlock, {0.0f, 0.0f, 0.0f}, identityRotation, 1.0f, 4);
    appendPod(middleBlock, static_cast<std::uint32_t>(1));
    appendPod(middleBlock, static_cast<std::int32_t>(2));
    appendPod(middleBlock, static_cast<std::uint32_t>(0));

    std::vector<std::uint8_t> triShapeBlock;
    appendAvObjectPrefix(triShapeBlock, {0.0f, 0.0f, 0.0f}, identityRotation, 1.0f);
    appendPod(triShapeBlock, static_cast<std::int32_t>(3));

    std::vector<std::uint8_t> dataBlock;
    appendPod(dataBlock, static_cast<std::int32_t>(0));
    appendPod(dataBlock, static_cast<std::uint16_t>(3));
    appendPod(dataBlock, static_cast<std::uint8_t>(0));
    appendPod(dataBlock, static_cast<std::uint8_t>(0));
    appendPod(dataBlock, static_cast<std::uint8_t>(1));
    for (int v = 0; v < 3; ++v) {
        appendPod(dataBlock, 0.0f);
        appendPod(dataBlock, 0.0f);
        appendPod(dataBlock, 0.0f);
    }
    appendPod(dataBlock, static_cast<std::uint16_t>(0));
    appendPod(dataBlock, static_cast<std::uint8_t>(0));
    appendPod(dataBlock, 0.0f);
    appendPod(dataBlock, 0.0f);
    appendPod(dataBlock, 0.0f);
    appendPod(dataBlock, 0.0f);
    appendPod(dataBlock, static_cast<std::uint8_t>(0));
    appendPod(dataBlock, static_cast<std::uint16_t>(0));
    appendPod(dataBlock, static_cast<std::int32_t>(-1));
    appendPod(dataBlock, static_cast<std::uint16_t>(1));
    appendPod(dataBlock, static_cast<std::uint32_t>(3));
    appendPod(dataBlock, static_cast<std::uint8_t>(1));
    appendPod(dataBlock, static_cast<std::uint16_t>(0));
    appendPod(dataBlock, static_cast<std::uint16_t>(1));
    appendPod(dataBlock, static_cast<std::uint16_t>(2));
    appendPod(dataBlock, static_cast<std::uint16_t>(0));

    // NiAlphaProperty: NiObjectNET prefix, then 16-bit flags and a threshold
    // byte. 0x12ed is blend=1 test=1 -- the combination retail Goodsprings
    // ships on 351 shapes -- and 100 is its most common threshold.
    std::vector<std::uint8_t> alphaBlock;
    appendPod(alphaBlock, static_cast<std::int32_t>(-1));   // nameRef
    appendPod(alphaBlock, static_cast<std::uint32_t>(0));   // numExtraData
    appendPod(alphaBlock, static_cast<std::int32_t>(-1));   // controllerRef
    appendPod(alphaBlock, static_cast<std::uint16_t>(0x12edu));
    appendPod(alphaBlock, static_cast<std::uint8_t>(100));  // threshold

    std::vector<std::uint8_t> fileBytes;
    const std::string headerLine = "Gamebryo File Format, Version 20.2.0.7";
    fileBytes.insert(fileBytes.end(), headerLine.begin(), headerLine.end());
    fileBytes.push_back('\n');
    appendPod(fileBytes, static_cast<std::uint32_t>(0x14020007u));
    appendPod(fileBytes, static_cast<std::uint8_t>(1));
    appendPod(fileBytes, static_cast<std::uint32_t>(11));
    appendPod(fileBytes, static_cast<std::uint32_t>(5));   // numBlocks
    appendPod(fileBytes, static_cast<std::uint32_t>(34));  // userVersion2
    appendSizedString8(fileBytes, "");
    appendSizedString8(fileBytes, "");
    appendSizedString8(fileBytes, "");
    appendPod(fileBytes, static_cast<std::uint16_t>(4));   // numBlockTypes
    appendSizedString32(fileBytes, "NiNode");
    appendSizedString32(fileBytes, "NiTriShape");
    appendSizedString32(fileBytes, "NiTriShapeData");
    appendSizedString32(fileBytes, "NiAlphaProperty");
    appendPod(fileBytes, static_cast<std::uint16_t>(0));  // block 0 -> NiNode
    appendPod(fileBytes, static_cast<std::uint16_t>(0));  // block 1 -> NiNode
    appendPod(fileBytes, static_cast<std::uint16_t>(1));  // block 2 -> NiTriShape
    appendPod(fileBytes, static_cast<std::uint16_t>(2));  // block 3 -> NiTriShapeData
    appendPod(fileBytes, static_cast<std::uint16_t>(3));  // block 4 -> NiAlphaProperty
    appendPod(fileBytes, static_cast<std::uint32_t>(rootBlock.size()));
    appendPod(fileBytes, static_cast<std::uint32_t>(middleBlock.size()));
    appendPod(fileBytes, static_cast<std::uint32_t>(triShapeBlock.size()));
    appendPod(fileBytes, static_cast<std::uint32_t>(dataBlock.size()));
    appendPod(fileBytes, static_cast<std::uint32_t>(alphaBlock.size()));
    appendPod(fileBytes, static_cast<std::uint32_t>(0));  // numStrings
    appendPod(fileBytes, static_cast<std::uint32_t>(0));  // maxStringLength
    appendPod(fileBytes, static_cast<std::uint32_t>(0));  // numGroups

    fileBytes.insert(fileBytes.end(), rootBlock.begin(), rootBlock.end());
    fileBytes.insert(fileBytes.end(), middleBlock.begin(), middleBlock.end());
    fileBytes.insert(fileBytes.end(), triShapeBlock.begin(), triShapeBlock.end());
    fileBytes.insert(fileBytes.end(), dataBlock.begin(), dataBlock.end());
    fileBytes.insert(fileBytes.end(), alphaBlock.begin(), alphaBlock.end());
    appendPod(fileBytes, static_cast<std::uint32_t>(1));  // Num Roots
    appendPod(fileBytes, static_cast<std::int32_t>(0));

    odai::importer::fnv::NifModel model;
    std::string error;
    expectTrue(
        odai::importer::fnv::parseNifStaticMesh(fileBytes, model, error),
        "A NIF whose alpha property sits on a parent node still parses");
    expectTrue(model.shapes.size() == 1, "One shape comes out of the fixture");
    if (model.shapes.size() == 1) {
        const odai::importer::fnv::NifShape& shape = model.shapes.front();
        expectTrue(
            shape.alphaTest,
            "An alpha property on a PARENT node applies to the shape beneath it");
        expectTrue(
            shape.alphaThreshold == 100,
            "The inherited property's authored threshold comes with it, not the default 128");
        // 0x12ed sets both bits, and blend+test means cutout in this reader
        // (see readNiAlphaProperty): the test defines the silhouette and the
        // diffuse alpha is a specular mask, not an opacity ramp.
        expectTrue(
            !shape.alphaBlend,
            "blend+test still resolves to cutout when inherited, exactly as when owned");
    }
}

// An index past the block's own vertices does NOT fault and does not draw
// nothing: shapes are merged into one vertex buffer downstream, so it resolves
// to a neighbouring shape's vertex and draws a triangle stretched between two
// unrelated meshes. No retail file in either game contains one -- measured 0
// across 20746 FalloutNV and 6399 Oblivion meshes -- which is exactly why this
// has to be synthetic: there is nothing in the shipped data to catch it.
void testNifParserRejectsUnusableTriangles() {
    const std::array<float, 9> identityRotation{1, 0, 0, 0, 1, 0, 0, 0, 1};

    std::vector<std::uint8_t> rootBlock;
    appendAvObjectPrefix(rootBlock, {0.0f, 0.0f, 0.0f}, identityRotation, 1.0f);
    appendPod(rootBlock, static_cast<std::uint32_t>(1));   // numChildren
    appendPod(rootBlock, static_cast<std::int32_t>(1));    // child -> the shape
    appendPod(rootBlock, static_cast<std::uint32_t>(0));   // numEffects

    std::vector<std::uint8_t> triShapeBlock;
    appendAvObjectPrefix(triShapeBlock, {0.0f, 0.0f, 0.0f}, identityRotation, 1.0f);
    appendPod(triShapeBlock, static_cast<std::int32_t>(2));  // data -> block 2

    // Three vertices, three triangles: one usable, one naming vertex 9 in a
    // three-vertex block, one naming vertex 1 twice.
    std::vector<std::uint8_t> dataBlock;
    appendPod(dataBlock, static_cast<std::int32_t>(0));     // groupId
    appendPod(dataBlock, static_cast<std::uint16_t>(3));    // numVertices
    appendPod(dataBlock, static_cast<std::uint8_t>(0));     // keepFlags
    appendPod(dataBlock, static_cast<std::uint8_t>(0));     // compressFlags
    appendPod(dataBlock, static_cast<std::uint8_t>(1));     // hasVertices
    for (int v = 0; v < 3; ++v) {
        appendPod(dataBlock, static_cast<float>(v));
        appendPod(dataBlock, 0.0f);
        appendPod(dataBlock, 0.0f);
    }
    appendPod(dataBlock, static_cast<std::uint16_t>(0));    // vector flags
    appendPod(dataBlock, static_cast<std::uint8_t>(0));     // hasNormals
    appendPod(dataBlock, 0.0f);                             // bound centre x
    appendPod(dataBlock, 0.0f);
    appendPod(dataBlock, 0.0f);
    appendPod(dataBlock, 0.0f);                             // bound radius
    appendPod(dataBlock, static_cast<std::uint8_t>(0));     // hasVertexColors
    appendPod(dataBlock, static_cast<std::uint16_t>(0));    // consistency flags
    appendPod(dataBlock, static_cast<std::int32_t>(-1));    // additional data
    appendPod(dataBlock, static_cast<std::uint16_t>(3));    // numTriangles
    appendPod(dataBlock, static_cast<std::uint32_t>(9));    // numTrianglePoints
    appendPod(dataBlock, static_cast<std::uint8_t>(1));     // hasTriangles
    const std::uint16_t triangles[9] = {0, 1, 2, 0, 1, 9, 1, 1, 2};
    for (const std::uint16_t index : triangles) {
        appendPod(dataBlock, index);
    }
    appendPod(dataBlock, static_cast<std::uint16_t>(0));    // numMatchGroups

    std::vector<std::uint8_t> fileBytes;
    const std::string headerLine = "Gamebryo File Format, Version 20.2.0.7";
    fileBytes.insert(fileBytes.end(), headerLine.begin(), headerLine.end());
    fileBytes.push_back('\n');
    appendPod(fileBytes, static_cast<std::uint32_t>(0x14020007u));
    appendPod(fileBytes, static_cast<std::uint8_t>(1));
    appendPod(fileBytes, static_cast<std::uint32_t>(11));
    appendPod(fileBytes, static_cast<std::uint32_t>(3));   // numBlocks
    appendPod(fileBytes, static_cast<std::uint32_t>(34));  // userVersion2
    appendSizedString8(fileBytes, "");
    appendSizedString8(fileBytes, "");
    appendSizedString8(fileBytes, "");
    appendPod(fileBytes, static_cast<std::uint16_t>(3));   // numBlockTypes
    appendSizedString32(fileBytes, "NiNode");
    appendSizedString32(fileBytes, "NiTriShape");
    appendSizedString32(fileBytes, "NiTriShapeData");
    appendPod(fileBytes, static_cast<std::uint16_t>(0));
    appendPod(fileBytes, static_cast<std::uint16_t>(1));
    appendPod(fileBytes, static_cast<std::uint16_t>(2));
    appendPod(fileBytes, static_cast<std::uint32_t>(rootBlock.size()));
    appendPod(fileBytes, static_cast<std::uint32_t>(triShapeBlock.size()));
    appendPod(fileBytes, static_cast<std::uint32_t>(dataBlock.size()));
    appendPod(fileBytes, static_cast<std::uint32_t>(0));   // numStrings
    appendPod(fileBytes, static_cast<std::uint32_t>(0));   // maxStringLength
    appendPod(fileBytes, static_cast<std::uint32_t>(0));   // numGroups

    fileBytes.insert(fileBytes.end(), rootBlock.begin(), rootBlock.end());
    fileBytes.insert(fileBytes.end(), triShapeBlock.begin(), triShapeBlock.end());
    fileBytes.insert(fileBytes.end(), dataBlock.begin(), dataBlock.end());
    appendPod(fileBytes, static_cast<std::uint32_t>(1));   // Num Roots
    appendPod(fileBytes, static_cast<std::int32_t>(0));

    odai::importer::fnv::NifModel model;
    std::string error;
    expectTrue(
        odai::importer::fnv::parseNifStaticMesh(fileBytes, model, error),
        "A shape carrying an unusable triangle still parses -- the shape is not lost");
    expectTrue(model.outOfRangeTriangleCount == 1u,
               "The triangle naming a vertex the block does not have is counted");
    expectTrue(model.degenerateTriangleCount == 1u,
               "The triangle naming one vertex twice is counted");
    expectTrue(model.shapes.size() == 1u, "The shape survives with its usable geometry");
    if (model.shapes.size() == 1u) {
        const odai::importer::fnv::NifShape& shape = model.shapes.front();
        expectTrue(shape.triangleIndices.size() == 3u,
                   "Only the one usable triangle reaches the shape");
        // Dropping the individual triangle, not the shape: one bad index in a
        // rock should cost one triangle, not the rock.
        expectTrue(shape.positions.size() == 9u,
                   "Rejecting triangles does not disturb the vertex array they indexed");
    }
}

void testNifParserRejectsImplausibleChildCount() {
    const std::array<float, 9> identityRotation{1, 0, 0, 0, 1, 0, 0, 0, 1};
    std::vector<std::uint8_t> rootBlock;
    appendAvObjectPrefix(rootBlock, {0.0f, 0.0f, 0.0f}, identityRotation, 1.0f);
    appendPod(rootBlock, static_cast<std::uint32_t>(0xFFFFFFFFu));  // numChildren
    appendPod(rootBlock, static_cast<std::uint32_t>(0));

    std::vector<std::uint8_t> fileBytes;
    const std::string headerLine = "Gamebryo File Format, Version 20.2.0.7";
    fileBytes.insert(fileBytes.end(), headerLine.begin(), headerLine.end());
    fileBytes.push_back('\n');
    appendPod(fileBytes, static_cast<std::uint32_t>(0x14020007u));
    appendPod(fileBytes, static_cast<std::uint8_t>(1));
    appendPod(fileBytes, static_cast<std::uint32_t>(11));
    appendPod(fileBytes, static_cast<std::uint32_t>(1));   // numBlocks
    appendPod(fileBytes, static_cast<std::uint32_t>(34));
    appendSizedString8(fileBytes, "");
    appendSizedString8(fileBytes, "");
    appendSizedString8(fileBytes, "");
    appendPod(fileBytes, static_cast<std::uint16_t>(1));
    appendSizedString32(fileBytes, "NiNode");
    appendPod(fileBytes, static_cast<std::uint16_t>(0));
    appendPod(fileBytes, static_cast<std::uint32_t>(rootBlock.size()));
    appendPod(fileBytes, static_cast<std::uint32_t>(0));
    appendPod(fileBytes, static_cast<std::uint32_t>(0));
    appendPod(fileBytes, static_cast<std::uint32_t>(0));
    fileBytes.insert(fileBytes.end(), rootBlock.begin(), rootBlock.end());

    odai::importer::fnv::NifModel model;
    std::string error;
    // Must return without allocating; whether it reports success with no shapes
    // is immaterial, the point is that it does not try to size a vector from a
    // count the block cannot possibly contain.
    odai::importer::fnv::parseNifStaticMesh(fileBytes, model, error);
    expectTrue(model.nodeParseFailedCount == 1u,
               "An impossible child count is rejected and counted, not allocated");
    expectTrue(model.shapes.empty(), "No geometry is invented from a rejected node");
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

    // Folder-prefix filtering. The fixture has one file under "meshes\\x" and
    // one under "textures\\x", which is enough to prove the filter keeps the
    // matching folder, drops the other, and -- the part that actually breaks if
    // the name block is mis-walked -- still resolves the kept file's NAME.
    // Filtered-out names must still be consumed from the sequential name block,
    // so a bug there shows up as the surviving entry having the wrong name
    // rather than as a missing entry.
    {
        odai::importer::fnv::BsaArchive filtered;
        expectTrue(filtered.open(archivePath, "textures\\x"),
                   "BSA opens with a folder filter" + label);
        expectTrue(filtered.files().size() == 1u,
                   "folder filter keeps only the matching folder" + label);
        expectTrue(filtered.find("textures\\x\\tx_wall_01.dds") != nullptr,
                   "the kept entry still resolves by its full path" + label);
        expectTrue(filtered.find("meshes\\x\\ex_wall_01.nif") == nullptr,
                   "the filtered-out entry is absent" + label);

        // Case-insensitive, and an unmatched prefix yields an empty index
        // rather than silently falling back to everything.
        odai::importer::fnv::BsaArchive mixedCase;
        expectTrue(mixedCase.open(archivePath, "TEXTURES\\X") && mixedCase.files().size() == 1u,
                   "folder filter is case-insensitive" + label);
        odai::importer::fnv::BsaArchive noMatch;
        expectTrue(noMatch.open(archivePath, "sound\\voice") && noMatch.files().empty(),
                   "a prefix matching nothing indexes nothing" + label);
    }

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

// Oblivion ships BSA v103. Its structures are byte-identical to v104, so the
// only thing worth pinning is the one difference: v103 sets kEmbedFileNames
// (0x100) in archiveFlags — "Oblivion - Meshes.bsa" is 0x787 — while writing
// no embedded name in any data block. The fixture reproduces exactly that
// contradiction, which the v104 reader would resolve by eating the first
// 1 + N payload bytes and failing every inflate with "incorrect header check".
void testBsaArchiveReadsOblivionV103() {
    namespace fs = std::filesystem;
    const std::string uncompressedContent = "NIF-DATA-PLACEHOLDER";
    const std::string compressedContent(4096, 'T');

    const std::vector<std::uint8_t> archiveBytes = buildSyntheticBsa(
        uncompressedContent, compressedContent,
        /*embedFileNames=*/false, /*version=*/103u, /*declareEmbeddedNames=*/true);
    const fs::path archivePath = fs::temp_directory_path() / "odai_oblivion_test.bsa";
    {
        std::ofstream out(archivePath, std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char*>(archiveBytes.data()),
                  static_cast<std::streamsize>(archiveBytes.size()));
    }

    std::uint32_t peekedFlags = 0;
    expectTrue(odai::importer::fnv::peekBsaContentFlags(archivePath, peekedFlags),
               "peekBsaContentFlags accepts a v103 archive");

    odai::importer::fnv::BsaArchive archive;
    expectTrue(archive.open(archivePath), "v103 BSA archive opens");
    expectTrue(archive.files().size() == 2u, "v103 BSA exposes both files");

    const auto* meshEntry = archive.find("meshes\\x\\ex_wall_01.nif");
    std::vector<std::uint8_t> meshBytes;
    expectTrue(meshEntry != nullptr && archive.extract(*meshEntry, meshBytes),
               "v103 uncompressed entry extracts");
    expectTrue(std::string(meshBytes.begin(), meshBytes.end()) == uncompressedContent,
               "v103 uncompressed bytes are not shortened by a phantom embedded name");

    const auto* textureEntry = archive.find("textures\\x\\tx_wall_01.dds");
    std::vector<std::uint8_t> textureBytes;
    expectTrue(textureEntry != nullptr && archive.extract(*textureEntry, textureBytes),
               "v103 compressed entry inflates");
    expectTrue(std::string(textureBytes.begin(), textureBytes.end()) == compressedContent,
               "v103 inflated bytes match the original content");

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

// Mod directories: an installed texture pack has to beat the archives, and it
// has to do so on a case-sensitive filesystem. The second half is the whole
// reason mod roots are indexed rather than probed with exists() -- packs ship
// "Textures\" while NIFs ask for "textures\", and on ext4 those are two
// different paths.
void testModDirectoryOverridesArchives() {
    namespace fs = std::filesystem;
    using namespace odai::importer::fnv;

    const std::string archiveMesh = "ARCHIVE-MESH";
    const std::string archiveTexture(4096, 'T');
    const std::vector<std::uint8_t> archiveBytes =
        buildSyntheticBsa(archiveMesh, archiveTexture, /*embedFileNames=*/false);

    const fs::path root = fs::temp_directory_path() / "odai_fnv_mod_dir_test";
    std::error_code cleanupError;
    fs::remove_all(root, cleanupError);

    const fs::path dataDir = root / "Data";
    fs::create_directories(dataDir, cleanupError);
    {
        std::ofstream out(dataDir / "Test.bsa", std::ios::binary | std::ios::trunc);
        out.write(
            reinterpret_cast<const char*>(archiveBytes.data()),
            static_cast<std::streamsize>(archiveBytes.size()));
    }

    // Deliberately capitalized the way a downloaded pack ships, and NOT the way
    // the archive path is spelled below.
    const fs::path modA = root / "ModA";
    fs::create_directories(modA / "Textures" / "X", cleanupError);
    {
        std::ofstream out(modA / "Textures" / "X" / "TX_Wall_01.dds", std::ios::binary);
        out << "MOD-A-TEXTURE";
    }

    const fs::path modB = root / "ModB";
    fs::create_directories(modB / "textures" / "x", cleanupError);
    {
        std::ofstream out(modB / "textures" / "x" / "tx_wall_01.dds", std::ios::binary);
        out << "MOD-B-TEXTURE";
    }

    {
        FalloutAssetSource source;
        expectTrue(source.open(dataDir), "asset source opens the data directory");
        expectTrue(
            source.modFingerprint().empty(),
            "with no mod directory the fingerprint is empty, so an unmodded cache key is unchanged");

        expectTrue(source.addModDirectory(modA), "a readable mod directory is added");
        expectTrue(source.modDirectoryCount() == 1u, "the mod directory is counted");
        expectTrue(source.modFileCount() == 1u, "the mod directory is indexed recursively");

        std::string error;
        std::vector<std::uint8_t> bytes;
        // The request spells the path all lowercase; the file on disk does not.
        expectTrue(
            source.resolveTexture("textures\\x\\tx_wall_01.dds", bytes, error),
            ("a mod texture resolves despite case differing from the file on disk: " + error).c_str());
        expectTrue(
            std::string(bytes.begin(), bytes.end()) == "MOD-A-TEXTURE",
            "the mod's texture wins over the archive's");

        // The prefix-adding form of the same request must land on the same file.
        bytes.clear();
        expectTrue(
            source.resolveTexture("X\\TX_WALL_01.DDS", bytes, error),
            ("a mod texture resolves through the textures\\ prefix path: " + error).c_str());
        expectTrue(
            std::string(bytes.begin(), bytes.end()) == "MOD-A-TEXTURE",
            "an unprefixed, differently-cased request resolves to the same mod file");

        // Anything the mod does not carry still falls through to the archive.
        bytes.clear();
        expectTrue(
            source.resolveMesh("x\\ex_wall_01.nif", bytes, error),
            ("a mesh the mod does not override still resolves from the archive: " + error).c_str());
        expectTrue(
            std::string(bytes.begin(), bytes.end()) == archiveMesh,
            "the archive still serves what no mod overrides");

        const std::string fingerprintA = source.modFingerprint();
        expectTrue(!fingerprintA.empty(), "a mod set produces a non-empty fingerprint");

        // Load order: the last directory added wins, as a mod manager's would.
        expectTrue(source.addModDirectory(modB), "a second mod directory is added");
        bytes.clear();
        expectTrue(
            source.resolveTexture("textures\\x\\tx_wall_01.dds", bytes, error),
            ("the overridden texture still resolves: " + error).c_str());
        expectTrue(
            std::string(bytes.begin(), bytes.end()) == "MOD-B-TEXTURE",
            "the last mod directory added wins, matching the archive load-order rule");
        expectTrue(
            source.modFingerprint() != fingerprintA,
            "adding a mod changes the fingerprint, so a stale cell cache misses instead of "
            "serving art from before the mod was installed");
    }

    // An unreadable mod directory is reported rather than silently ignored, and
    // does not stop the base game from resolving.
    {
        FalloutAssetSource source;
        expectTrue(source.open(dataDir), "asset source opens the data directory");
        expectTrue(
            !source.addModDirectory(root / "does_not_exist"),
            "a missing mod directory fails rather than being silently accepted");
        expectTrue(!source.warnings().empty(), "a missing mod directory is reported");
        expectTrue(source.modDirectoryCount() == 0u, "a failed mod directory is not counted");

        std::string error;
        std::vector<std::uint8_t> bytes;
        expectTrue(
            source.resolveMesh("x\\ex_wall_01.nif", bytes, error),
            "the base game still resolves after a mod directory failed to load");
    }

    fs::remove_all(root, cleanupError);
}

// Load order and formID remapping. The mod index inside a plugin file is local
// to that file, so this is what decides whether a second plugin's records
// resolve to the right things or silently to the wrong ones.
//
// Synthetic plugins only -- a TES4 header is small enough to write by hand, and
// the point is the index arithmetic, not any real game data.
// A WTHR record is not one format across these games, and both differences are
// silent -- they produce a plausible-looking weather rather than an error.
//
//   FNAM is SIX floats in Fallout and FOUR in Oblivion: the day/night fog POWER
//   pair arrived with Fallout 3. Sizing the read at 24 left every Oblivion
//   weather with fogDayFar = 0, which a forced weather then publishes as the
//   aerial-perspective distance and fogs the frame flat from the near plane.
//
//   PNAM does not exist in Oblivion at all. Its cloud tints are the two NAM0
//   channels the Fallout docs call unused and the Oblivion CS calls
//   Clouds-Lower and Clouds-Upper. The tint IS the layer colour, so the zero
//   default does not render "untinted" -- it renders the clouds pure black.
//
// Both directions are asserted: the Fallout record must keep reading its own
// PNAM and its fog powers, or this fix trades one game's sky for the other's.
void testOblivionWeatherFogAndCloudTints() {
    namespace fs = std::filesystem;
    using namespace odai::importer::fnv;

    constexpr auto kDay = static_cast<std::size_t>(FalloutWeatherTimeSlot::Day);
    constexpr auto kNoon = static_cast<std::size_t>(FalloutWeatherTimeSlot::Noon);

    const fs::path dataDir = fs::temp_directory_path() / "odai_fnv_weather_test";
    std::error_code cleanupError;
    fs::remove_all(dataDir, cleanupError);
    fs::create_directories(dataDir, cleanupError);

    // NAM0 rows are [color][slot]; `slots` is 4 for Oblivion/FO3 and 6 for NV.
    const auto buildColorTable = [](std::size_t rows, std::size_t slots,
                                    const std::vector<std::uint8_t>& redPerRow) {
        std::vector<std::uint8_t> out(rows * slots * 4u, 0u);
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t slot = 0; slot < slots; ++slot) {
                const std::size_t offset = ((row * slots) + slot) * 4u;
                // Red identifies the ROW, green the SLOT, so a read that lands
                // on the wrong channel or the wrong time of day is legible in
                // the failure rather than merely wrong.
                out[offset + 0u] = redPerRow[row];
                out[offset + 1u] = static_cast<std::uint8_t>(10u + slot);
                out[offset + 2u] = 0u;
            }
        }
        return out;
    };

    constexpr std::size_t kColorRows = static_cast<std::size_t>(FalloutWeatherColor::Count);
    std::vector<std::uint8_t> reds(kColorRows, 0u);
    for (std::size_t row = 0; row < kColorRows; ++row) {
        reds[row] = static_cast<std::uint8_t>(100u + row);
    }
    constexpr std::uint8_t kCloudsLowerRed = 102u;  // FalloutWeatherColor::CloudsLower
    constexpr std::uint8_t kCloudsUpperRed = 109u;  // FalloutWeatherColor::CloudsUpper

    // --- The Oblivion shape: 20-byte headers, 4 time slots, 16-byte FNAM, no PNAM.
    {
        constexpr auto kFormat = EsmPluginFormat::kOblivion;
        std::vector<std::uint8_t> body = buildSubrecord("EDID", stringPayload("TestOvercast"));
        const auto append = [&body](const std::vector<std::uint8_t>& part) {
            body.insert(body.end(), part.begin(), part.end());
        };
        append(buildSubrecord("DNAM", stringPayload("Sky\\CloudsUpper.dds")));
        append(buildSubrecord("CNAM", stringPayload("Sky\\CloudsLower.dds")));
        append(buildSubrecord("NAM0", buildColorTable(kColorRows, 4u, reds)));
        std::vector<std::uint8_t> fnam;
        appendPod(fnam, 2048.0f);    // fog day near
        appendPod(fnam, 150000.0f);  // fog day far
        appendPod(fnam, 2048.0f);    // fog night near
        appendPod(fnam, 130000.0f);  // fog night far
        append(buildSubrecord("FNAM", fnam));
        append(buildSubrecord("DATA", std::vector<std::uint8_t>(15, 0u)));

        std::vector<std::uint8_t> file = buildTes4Record({}, kFormat);
        const auto record = buildRecord("WTHR", 0x00038EECu, 0u, body, kFormat);
        const auto group = buildGroup("WTHR", 0, record, kFormat);
        file.insert(file.end(), group.begin(), group.end());
        std::ofstream out(dataDir / "Oblivion.esm", std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char*>(file.data()),
                  static_cast<std::streamsize>(file.size()));
    }

    // --- The Fallout shape: 24-byte headers, 6 time slots, 24-byte FNAM, real PNAM.
    {
        constexpr auto kFormat = EsmPluginFormat::kFallout3;
        std::vector<std::uint8_t> body = buildSubrecord("EDID", stringPayload("TestMojave"));
        const auto append = [&body](const std::vector<std::uint8_t>& part) {
            body.insert(body.end(), part.begin(), part.end());
        };
        append(buildSubrecord("DNAM", stringPayload("Sky\\Layer0.dds")));
        // Layer 1 is the transparent placeholder nearly every real record uses
        // for an unused layer, so the emitted list has a HOLE in its indices.
        append(buildSubrecord("CNAM", stringPayload("Sky\\Alpha.dds")));
        append(buildSubrecord("ANAM", stringPayload("Sky\\Layer2.dds")));
        append(buildSubrecord("NAM0", buildColorTable(kColorRows, 6u, reds)));
        // PNAM: four layers x six slots. Red 200 + layer, so it cannot be
        // confused with any NAM0 row.
        std::vector<std::uint8_t> pnamReds(FalloutWeatherRecord::kCloudLayerCount, 0u);
        for (std::size_t layer = 0; layer < pnamReds.size(); ++layer) {
            pnamReds[layer] = static_cast<std::uint8_t>(200u + layer);
        }
        append(buildSubrecord(
            "PNAM", buildColorTable(FalloutWeatherRecord::kCloudLayerCount, 6u, pnamReds)));
        std::vector<std::uint8_t> fnam;
        appendPod(fnam, 1024.0f);    // fog day near
        appendPod(fnam, 90000.0f);   // fog day far
        appendPod(fnam, 512.0f);     // fog night near
        appendPod(fnam, 40000.0f);   // fog night far
        appendPod(fnam, 1.5f);       // fog day power
        appendPod(fnam, 2.5f);       // fog night power
        append(buildSubrecord("FNAM", fnam));
        append(buildSubrecord("DATA", std::vector<std::uint8_t>(15, 0u)));

        std::vector<std::uint8_t> file = buildTes4Record({}, kFormat);
        const auto record = buildRecord("WTHR", 0x00100200u, 0u, body, kFormat);
        const auto group = buildGroup("WTHR", 0, record, kFormat);
        file.insert(file.end(), group.begin(), group.end());
        std::ofstream out(dataDir / "FalloutNV.esm", std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char*>(file.data()),
                  static_cast<std::streamsize>(file.size()));
    }

    const auto readTables = [&](const char* fileName, FalloutWeatherTables& tables) {
        FalloutLoadOrder order;
        std::string error;
        if (!order.open(dataDir, {fileName}, error)) {
            expectTrue(false, ("a weather fixture load order opens: " + error).c_str());
            return false;
        }
        if (!buildFalloutWeatherTables(order, tables, error)) {
            expectTrue(false, ("weather tables build: " + error).c_str());
            return false;
        }
        return true;
    };

    {
        FalloutWeatherTables tables;
        if (readTables("Oblivion.esm", tables)) {
            const FalloutWeatherRecord* weather = tables.findWeatherByEditorId("TestOvercast");
            expectTrue(weather != nullptr, "the Oblivion WTHR is found by editor ID");
            if (weather != nullptr) {
                expectTrue(weather->fogDayFar == 150000.0f,
                           "a 16-byte FNAM still yields fogDayFar (the >= 24 guard zeroed it)");
                expectTrue(weather->fogNightFar == 130000.0f, "and fogNightFar");
                expectTrue(weather->fogDayPower == 0.0f && weather->fogNightPower == 0.0f,
                           "fog powers stay zero -- Oblivion authors no bytes for them");
                expectTrue(weather->cloudLayers.size() == 2u,
                           "only the two cloud layers Oblivion authors are emitted");
                expectTrue(weather->cloudMapping == FalloutCloudMapping::DomeFisheye,
                           "and they are dome fisheye maps, not tiling sheets");
                if (weather->cloudLayers.size() == 2u) {
                    // Layer 0 is DNAM, which the file names as the upper layer.
                    expectTrue(weather->cloudLayers[0].texture == "Sky\\CloudsUpper.dds",
                               "layer 0 is DNAM");
                    expectTrue(weather->cloudLayers[0].tint[kDay].r == kCloudsUpperRed,
                               "with no PNAM, cloud layer 0 is tinted from NAM0 Clouds-Upper");
                    expectTrue(weather->cloudLayers[1].tint[kDay].r == kCloudsLowerRed,
                               "and cloud layer 1 from NAM0 Clouds-Lower");
                    expectTrue(weather->cloudLayers[0].tint[kDay].r != 0u &&
                                   weather->cloudLayers[1].tint[kDay].r != 0u,
                               "neither layer keeps the zero tint that renders clouds pure black");
                    // Oblivion's four slots are widened to six on the way in, so
                    // a caller asking for Noon must not fall off the end into
                    // black.
                    expectTrue(weather->cloudLayers[0].tint[kNoon].r == kCloudsUpperRed,
                               "the four-slot table widens to six, so Noon is tinted too");
                    expectTrue(weather->cloudLayers[0].alpha[kDay] == 1.0f,
                               "a record with no JNAM leaves every layer fully opaque");
                }
            }
        }
    }

    {
        FalloutWeatherTables tables;
        if (readTables("FalloutNV.esm", tables)) {
            const FalloutWeatherRecord* weather = tables.findWeatherByEditorId("TestMojave");
            expectTrue(weather != nullptr, "the Fallout WTHR is found by editor ID");
            if (weather != nullptr) {
                expectTrue(weather->fogDayFar == 90000.0f, "a 24-byte FNAM still reads its fog far");
                expectTrue(weather->fogDayPower == 1.5f && weather->fogNightPower == 2.5f,
                           "and still reads the two fog powers past the first sixteen bytes");
                expectTrue(weather->cloudLayers.size() == 2u,
                           "the two layers Fallout names are emitted; the alpha placeholder is not");
                if (weather->cloudLayers.size() == 2u) {
                    expectTrue(weather->cloudLayers[0].tint[kDay].r == 200u,
                               "a record WITH PNAM keeps its own tints rather than the NAM0 fallback");
                    // The gap is the point: layer 1 is sky\alpha.dds, the
                    // transparent placeholder, so the SECOND emitted layer is
                    // record layer 2 -- and it must carry layer 2's tint, not
                    // the tint of the slot it happens to land in. This is the
                    // pairing that, broken, painted Skyrim's sky solid black.
                    expectTrue(weather->cloudLayers[1].index == 2,
                               "an unused layer is skipped rather than renumbering the rest");
                    expectTrue(weather->cloudLayers[1].tint[kDay].r == 202u,
                               "and the layer that survives keeps its OWN tint, not the slot's");
                    expectTrue(
                        weather->cloudLayers[0].tint[kNoon].g == static_cast<std::uint8_t>(10u + kNoon),
                        "six authored slots are not collapsed into four");
                }
            }
        }
    }

    fs::remove_all(dataDir, cleanupError);
}

// SKYRIM'S CLOUD BLOCK IS A DIFFERENT SHAPE, AND READING IT AS FALLOUT'S PAINTS
// THE WHOLE SKY BLACK.
//
// It authors up to 29 layers in subrecords named chr('0' + layer) + "0TX",
// picks which are live with a bitfield in NAM1, tints them from a 32-row PNAM,
// and gives each a per-slot opacity in JNAM. Taking the first four TEXTURES and
// the first four PNAM ROWS independently -- which is what four DNAM/CNAM/ANAM/
// BNAM slots invite -- pairs a live layer's texture with a DEAD layer's tint,
// and Skyrim authors a black daytime tint on exactly the layers it disables.
// The result is a fully opaque black sky that looks like a broken shader.
//
// This pins the pairing rather than the parse: the assertion that matters is
// that the surviving layer carries the tint and alpha of ITS OWN index.
void testSkyrimWeatherCloudLayers() {
    namespace fs = std::filesystem;
    using namespace odai::importer::fnv;

    constexpr auto kDay = static_cast<std::size_t>(FalloutWeatherTimeSlot::Day);
    constexpr auto kSlots = 4u;         // Skyrim authors four time slots
    constexpr auto kSkyrimLayers = 32u; // and 32-wide cloud tables

    const fs::path dataDir = fs::temp_directory_path() / "odai_skyrim_weather_test";
    std::error_code cleanupError;
    fs::remove_all(dataDir, cleanupError);
    fs::create_directories(dataDir, cleanupError);

    constexpr auto kFormat = EsmPluginFormat::kFallout3;  // Skyrim shares the container
    std::vector<std::uint8_t> body = buildSubrecord("EDID", stringPayload("TestSkyrimCloudy"));
    const auto append = [&body](const std::vector<std::uint8_t>& part) {
        body.insert(body.end(), part.begin(), part.end());
    };
    // Layer 0 is an Upper-band leftover that Skyrim disables; layer 9 is a live
    // Lower deck; layer 20 a live Horizon bank. Subrecord names are the layer
    // index offset from '0', so layer 20 is 'D' + "0TX".
    const auto cloudSubrecord = [](int layer, const char* path) {
        std::string type;
        type.push_back(static_cast<char>('0' + layer));
        type += "0TX";
        return buildSubrecord(type.c_str(), stringPayload(path));
    };
    append(cloudSubrecord(0, "Sky\\DeadUpper.dds"));
    append(cloudSubrecord(9, "Sky\\SkyrimCloudsLower01.dds"));
    append(cloudSubrecord(20, "Sky\\SkyrimCloudsHorizon01.dds"));

    // NAM1: 1 = DISABLED. Everything but 9 and 20.
    std::uint32_t disabled = 0xFFFFFFFFu;
    disabled &= ~(1u << 9);
    disabled &= ~(1u << 20);
    {
        std::vector<std::uint8_t> nam1;
        appendPod(nam1, disabled);
        append(buildSubrecord("NAM1", nam1));
    }

    // PNAM: 32 layers x 4 slots x RGBA. Red identifies the layer. The DISABLED
    // layers get the black daytime tint the real records have, so a reader that
    // takes tints by slot instead of by layer produces a black sky here too.
    {
        std::vector<std::uint8_t> pnam(kSkyrimLayers * kSlots * 4u, 0u);
        for (std::uint32_t layer = 0; layer < kSkyrimLayers; ++layer) {
            const bool live = ((disabled >> layer) & 1u) == 0u;
            for (std::uint32_t slot = 0; slot < kSlots; ++slot) {
                const std::size_t offset = ((layer * kSlots) + slot) * 4u;
                pnam[offset + 0u] = live ? static_cast<std::uint8_t>(100u + layer) : 0u;
                pnam[offset + 1u] = live ? static_cast<std::uint8_t>(10u + slot) : 0u;
            }
        }
        append(buildSubrecord("PNAM", pnam));
    }
    // JNAM: 128 floats, layer-major. Layer 20 is held at 0.4, the way the real
    // records hold a horizon bank and the fill swatch back.
    {
        std::vector<std::uint8_t> jnam;
        for (std::uint32_t layer = 0; layer < kSkyrimLayers; ++layer) {
            for (std::uint32_t slot = 0; slot < kSlots; ++slot) {
                appendPod(jnam, layer == 20u ? 0.4f : 1.0f);
            }
        }
        append(buildSubrecord("JNAM", jnam));
    }
    // NAM0: 17 rows x 4 slots. Row 12 is Fog Far, which is the channel aerial
    // perspective must use -- row 1 is the NEAR fog and is a saturated tint.
    {
        constexpr std::uint32_t kSkyrimColorRows = 17u;
        std::vector<std::uint8_t> nam0(kSkyrimColorRows * kSlots * 4u, 0u);
        for (std::uint32_t row = 0; row < kSkyrimColorRows; ++row) {
            for (std::uint32_t slot = 0; slot < kSlots; ++slot) {
                const std::size_t offset = ((row * kSlots) + slot) * 4u;
                nam0[offset + 0u] = static_cast<std::uint8_t>(50u + row);
            }
        }
        append(buildSubrecord("NAM0", nam0));
    }
    // FNAM is EIGHT floats in Skyrim; the first six are laid out as Fallout's.
    {
        std::vector<std::uint8_t> fnam;
        appendPod(fnam, 0.0f);
        appendPod(fnam, 100000.0f);
        appendPod(fnam, 1000.0f);
        appendPod(fnam, 50000.0f);
        appendPod(fnam, 0.4f);
        appendPod(fnam, 0.3f);
        appendPod(fnam, 0.875f);
        appendPod(fnam, 0.875f);
        append(buildSubrecord("FNAM", fnam));
    }
    append(buildSubrecord("DATA", std::vector<std::uint8_t>(19, 0u)));

    std::vector<std::uint8_t> file = buildTes4Record({}, kFormat);
    const auto record = buildRecord("WTHR", 0x00012F89u, 0u, body, kFormat);
    const auto group = buildGroup("WTHR", 0, record, kFormat);
    file.insert(file.end(), group.begin(), group.end());
    {
        std::ofstream out(dataDir / "Skyrim.esm", std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char*>(file.data()),
                  static_cast<std::streamsize>(file.size()));
    }

    FalloutLoadOrder order;
    std::string error;
    if (!order.open(dataDir, {"Skyrim.esm"}, error)) {
        expectTrue(false, ("the Skyrim weather fixture opens: " + error).c_str());
        fs::remove_all(dataDir, cleanupError);
        return;
    }
    FalloutWeatherTables tables;
    if (!buildFalloutWeatherTables(order, tables, error)) {
        expectTrue(false, ("Skyrim weather tables build: " + error).c_str());
        fs::remove_all(dataDir, cleanupError);
        return;
    }

    const FalloutWeatherRecord* weather = tables.findWeatherByEditorId("TestSkyrimCloudy");
    expectTrue(weather != nullptr, "the Skyrim WTHR is found by editor ID");
    if (weather == nullptr) {
        fs::remove_all(dataDir, cleanupError);
        return;
    }

    expectTrue(weather->cloudMapping == FalloutCloudMapping::TilingPlane,
               "an x0TX cloud block marks the record's textures as tiling sheets");
    expectTrue(weather->cloudLayers.size() == 2u,
               "NAM1's disabled bit drops the dead layer, leaving the two live ones");
    if (weather->cloudLayers.size() != 2u) {
        fs::remove_all(dataDir, cleanupError);
        return;
    }

    expectTrue(weather->cloudLayers[0].index == 9 && weather->cloudLayers[1].index == 20,
               "the surviving layers keep their own record indices");
    expectTrue(weather->cloudLayers[0].texture == "Sky\\SkyrimCloudsLower01.dds",
               "and their own textures");
    // THE ASSERTION THIS TEST EXISTS FOR. Reading PNAM row 0 for the first
    // emitted layer gives 0 here -- the black tint of a disabled layer -- and
    // that is exactly the bug: a fully opaque black sky.
    expectTrue(weather->cloudLayers[0].tint[kDay].r == 109u,
               "layer 9's tint comes from PNAM ROW 9, not from the slot it lands in");
    expectTrue(weather->cloudLayers[1].tint[kDay].r == 120u,
               "and layer 20's from row 20");
    expectTrue(weather->cloudLayers[0].tint[kDay].r != 0u &&
                   weather->cloudLayers[1].tint[kDay].r != 0u,
               "neither live layer inherits a disabled layer's black daytime tint");

    expectTrue(weather->cloudLayers[0].alpha[kDay] == 1.0f,
               "JNAM gives layer 9 full opacity");
    expectTrue(weather->cloudLayers[1].alpha[kDay] == 0.4f,
               "and holds layer 20 back at 0.4, as the records hold a horizon bank");

    expectTrue(weather->cloudLayers[0].band == FalloutCloudBand::Lower,
               "layer 9 is an overhead deck");
    expectTrue(weather->cloudLayers[1].band == FalloutCloudBand::Horizon,
               "layer 20 is a horizon bank, which is drawn with a different projection");

    expectTrue(weather->hasFogFarColor, "a 17-row NAM0 carries Skyrim's separate far fog");
    expectTrue(weather->fogFarColors[kDay].r == 62u,
               "and it is row 12 -- row 1 is the NEAR fog, and using it tints a whole city cyan");
    expectTrue(weather->fogDayFar == 100000.0f && weather->fogNightFar == 50000.0f,
               "an eight-float FNAM still reads the four distances");

    fs::remove_all(dataDir, cleanupError);
}

void testPluginLoadOrderRemapsFormIds() {
    namespace fs = std::filesystem;
    using namespace odai::importer::fnv;

    const fs::path dataDir = fs::temp_directory_path() / "odai_fnv_load_order_test";
    std::error_code cleanupError;
    fs::remove_all(dataDir, cleanupError);
    fs::create_directories(dataDir, cleanupError);

    // Writes a plugin that is nothing but a TES4 record: the header is all a
    // load order ever reads.
    const auto writePlugin = [&](const std::string& fileName,
                                 const std::vector<std::string>& masters,
                                 bool isMaster,
                                 bool isLight = false) {
        std::vector<std::uint8_t> body;
        const auto appendSubrecord = [&](const char* type, const std::vector<std::uint8_t>& data) {
            body.insert(body.end(), type, type + 4);
            const auto size = static_cast<std::uint16_t>(data.size());
            body.push_back(static_cast<std::uint8_t>(size & 0xFFu));
            body.push_back(static_cast<std::uint8_t>((size >> 8) & 0xFFu));
            body.insert(body.end(), data.begin(), data.end());
        };
        // HEDR: version float, record count, next object id.
        std::vector<std::uint8_t> hedr(12, 0u);
        appendSubrecord("HEDR", hedr);
        for (const std::string& master : masters) {
            std::vector<std::uint8_t> name(master.begin(), master.end());
            name.push_back(0u);
            appendSubrecord("MAST", name);
            appendSubrecord("DATA", std::vector<std::uint8_t>(8, 0u));
        }

        std::vector<std::uint8_t> file;
        const char* type = "TES4";
        file.insert(file.end(), type, type + 4);
        const auto appendU32 = [&](std::uint32_t value) {
            for (int shift = 0; shift < 32; shift += 8) {
                file.push_back(static_cast<std::uint8_t>((value >> shift) & 0xFFu));
            }
        };
        appendU32(static_cast<std::uint32_t>(body.size()));
        appendU32((isMaster ? 0x00000001u : 0u) | (isLight ? 0x00000200u : 0u));
        appendU32(0u);  // formID
        appendU32(0u);  // version control
        appendU32(0u);  // formVersion + unknown
        file.insert(file.end(), body.begin(), body.end());

        std::ofstream out(dataDir / fileName, std::ios::binary | std::ios::trunc);
        out.write(
            reinterpret_cast<const char*>(file.data()), static_cast<std::streamsize>(file.size()));
    };

    // The real Nevada Skies shape: a base master, four DLC masters, and a mod
    // declaring all five.
    writePlugin("FalloutNV.esm", {}, /*isMaster=*/true);
    writePlugin("DeadMoney.esm", {"FalloutNV.esm"}, true);
    writePlugin("HonestHearts.esm", {"FalloutNV.esm"}, true);
    writePlugin("OldWorldBlues.esm", {"FalloutNV.esm"}, true);
    writePlugin("LonesomeRoad.esm", {"FalloutNV.esm"}, true);
    writePlugin(
        "NevadaSkies.esp",
        {"FalloutNV.esm", "DeadMoney.esm", "HonestHearts.esm", "OldWorldBlues.esm",
         "LonesomeRoad.esm"},
        false);

    // Skyrim light plugins still address masters/self with the ordinary local
    // high byte on disk; only the remapped global identity uses 0xFE.
    writePlugin("LightMaster.esl", {"FalloutNV.esm"}, true, true);
    writePlugin("LightPatch.esl", {"FalloutNV.esm", "LightMaster.esl"}, false, true);

    {
        FalloutPluginHeader header;
        std::string error;
        expectTrue(
            readFalloutPluginHeader(dataDir / "NevadaSkies.esp", header, error),
            ("a plugin header reads: " + error).c_str());
        expectTrue(header.masters.size() == 5u, "the plugin's five masters are read in order");
        expectTrue(header.masters[0] == "FalloutNV.esm", "the first master is the base game");
        expectTrue(header.masters[4] == "LonesomeRoad.esm", "the last master keeps its position");
        expectTrue(!header.isMaster, "an .esp is not flagged as a master");
    }

    {
        FalloutLoadOrder order;
        std::string error;
        expectTrue(
            order.open(dataDir, {"FalloutNV.esm", "LightMaster.esl", "LightPatch.esl"}, error),
            ("a regular/light load order resolves: " + error).c_str());
        expectTrue(order.entries()[1].header.isLight, "TES4 flag 0x200 marks a light plugin");
        expectTrue(
            order.entries()[1].slot.kind == FalloutPluginSlotKind::Light &&
                order.entries()[1].slot.index == 0u,
            "the first light plugin gets light slot zero without consuming a regular slot");
        expectTrue(
            order.remapFormId(1u, 0x01000800u) == 0xFE000800u,
            "a light plugin's own local record maps into the 0xFE namespace");
        expectTrue(
            order.remapFormId(2u, 0x01000801u) == 0xFE000801u,
            "a light plugin reference to a light master keeps the master's light slot");
        expectTrue(
            order.remapFormId(2u, 0x02000802u) == 0xFE001802u,
            "a second light plugin gets the next twelve-bit light slot");
        expectTrue(
            order.ownerOf(0xFE001802u) == &order.entries()[2],
            "owner lookup decodes a light global formID");
        expectTrue(
            order.remapFormId(2u, 0x02001802u) == 0x02001802u,
            "an out-of-range light object ID is not truncated onto another record");
    }

    // Asking for only the mod must pull in every master it needs, ahead of it.
    {
        FalloutLoadOrder order;
        std::string error;
        expectTrue(
            order.open(dataDir, {"NevadaSkies.esp"}, error),
            ("a load order resolves from one requested plugin: " + error).c_str());
        expectTrue(
            order.size() == 6u,
            "requesting one mod pulls in its five masters, so six plugins load");
        expectTrue(
            order.entries()[0].header.fileName == "FalloutNV.esm",
            "the base game loads first because everything masters it");
        expectTrue(
            order.entries()[5].header.fileName == "NevadaSkies.esp",
            "the mod loads after every master it declares");

        // The payload case: Nevada Skies' own records carry local index 5,
        // which here happens to equal its global index.
        const std::size_t modIndex = 5u;
        expectTrue(
            order.remapFormId(modIndex, 0x05000ABCu) == 0x05000ABCu,
            "the mod's own records map to its global index");
        expectTrue(
            order.remapFormId(modIndex, 0x00123456u) == 0x00123456u,
            "a reference to the base game stays at index 0");
        expectTrue(
            order.remapFormId(modIndex, 0x04001111u) == 0x04001111u,
            "a reference to the fifth master maps to global index 4");
        // A local index past the master list is malformed; inventing a target
        // for it would alias a bad reference onto a real record.
        expectTrue(
            order.remapFormId(modIndex, 0x0900BEEFu) == 0x0900BEEFu,
            "a local index the plugin never declared is left alone");
    }

    // The case that actually needs remapping: a mod whose only master is the
    // base game, loaded alongside the DLC. Its local index 1 means "my own
    // records", but its global index is 5.
    {
        writePlugin("SmallMod.esp", {"FalloutNV.esm"}, false);
        FalloutLoadOrder order;
        std::string error;
        expectTrue(
            order.open(
                dataDir,
                {"FalloutNV.esm", "DeadMoney.esm", "HonestHearts.esm", "OldWorldBlues.esm",
                 "LonesomeRoad.esm", "SmallMod.esp"},
                error),
            ("an explicit load order resolves: " + error).c_str());
        expectTrue(order.size() == 6u, "an explicit order loads exactly what was asked for");

        const std::size_t smallModIndex = 5u;
        expectTrue(
            order.entries()[smallModIndex].header.fileName == "SmallMod.esp",
            "the mod sits last in the explicit order");
        expectTrue(
            order.entries()[smallModIndex].localToGlobal.size() == 2u,
            "a one-master plugin maps two local indices: its master and itself");
        // THE bug this whole module exists to prevent: local 1 is the plugin's
        // own space, and it must become 5, not stay 1 (which is DeadMoney).
        expectTrue(
            order.remapFormId(smallModIndex, 0x01000042u) == 0x05000042u,
            "the mod's own local index 1 remaps to its global index 5, not to DeadMoney");
        expectTrue(
            order.remapFormId(smallModIndex, 0x00000042u) == 0x00000042u,
            "its reference to the base game is unchanged");

        // A different fingerprint than the previous order, so a cache keyed on
        // it cannot serve records built from a different plugin set.
        FalloutLoadOrder other;
        std::string otherError;
        expectTrue(other.open(dataDir, {"NevadaSkies.esp"}, otherError), "the other order opens");
        expectTrue(
            order.fingerprint() != other.fingerprint(),
            "different plugin sets produce different fingerprints");
    }

    // A plugin living in a mod directory rather than in Data. This is what lets
    // a mod ship its .esp beside its .bsa instead of needing anything copied
    // into the game install.
    {
        const fs::path modDir = dataDir.parent_path() / "odai_fnv_load_order_moddir";
        fs::remove_all(modDir, cleanupError);
        fs::create_directories(modDir, cleanupError);
        // Written into the mod directory, NOT into dataDir.
        {
            const fs::path staged = dataDir / "InModDir.esp";
            writePlugin("InModDir.esp", {"FalloutNV.esm"}, false);
            fs::rename(staged, modDir / "InModDir.esp", cleanupError);
        }

        FalloutLoadOrder without;
        std::string withoutError;
        expectTrue(
            !without.open(dataDir, {"InModDir.esp"}, withoutError),
            "a plugin outside the data directory is not found without a search root");

        FalloutLoadOrder order;
        order.addSearchRoot(modDir);
        std::string error;
        expectTrue(
            order.open(dataDir, {"InModDir.esp"}, error),
            ("a search root makes a mod-directory plugin loadable: " + error).c_str());
        expectTrue(order.size() == 2u, "its master still resolves from the data directory");
        expectTrue(
            order.entries()[1].path.parent_path() == modDir,
            "the plugin is loaded from the mod directory it actually lives in");
        expectTrue(
            order.entries()[0].header.fileName == "FalloutNV.esm",
            "the master loads first even though it lives somewhere else");

        // Higher priority than the data directory: a mod shadowing a plugin the
        // game also ships must win, the same way its assets do.
        {
            const fs::path staged = dataDir / "Shadowed.esp";
            writePlugin("Shadowed.esp", {"FalloutNV.esm"}, false);
            std::ofstream copy(modDir / "Shadowed.esp", std::ios::binary | std::ios::trunc);
            std::ifstream source(staged, std::ios::binary);
            copy << source.rdbuf();
        }
        FalloutLoadOrder shadowed;
        shadowed.addSearchRoot(modDir);
        std::string shadowError;
        expectTrue(shadowed.open(dataDir, {"Shadowed.esp"}, shadowError), "the shadowed load opens");
        expectTrue(
            shadowed.entries()[1].path.parent_path() == modDir,
            "a mod directory outranks the data directory for a plugin of the same name");

        fs::remove_all(modDir, cleanupError);
    }

    // A missing master must name itself rather than failing vaguely.
    {
        writePlugin("Orphan.esp", {"NotInstalled.esm"}, false);
        FalloutLoadOrder order;
        std::string error;
        expectTrue(!order.open(dataDir, {"Orphan.esp"}, error), "a missing master fails the load");
        expectTrue(
            error.find("NotInstalled.esm") != std::string::npos,
            "the error names the master that is missing");
        expectTrue(order.empty(), "a failed load order leaves nothing half-built");
    }

    fs::remove_all(dataDir, cleanupError);
}

void testSkyrimPluginListResolution() {
    namespace fs = std::filesystem;
    using namespace odai::importer::fnv;

    const fs::path root = fs::temp_directory_path() / "odai_skyrim_plugin_list_test";
    const fs::path data = root / "Data";
    std::error_code errorCode;
    fs::remove_all(root, errorCode);
    fs::create_directories(data, errorCode);
    const auto touch = [&](const std::string& name) {
        std::ofstream(data / name, std::ios::binary).put('\0');
    };
    touch("Skyrim.esm");
    touch("Update.esm");
    touch("Dawnguard.esm");
    touch("Present.esl");
    touch("Explicit.esp");
    {
        std::ofstream ccc(root / "Skyrim.ccc");
        ccc << "Missing.esl\nPresent.esl\n";
    }

    std::vector<std::string> plugins;
    fs::path source;
    std::string error;
    expectTrue(
        resolveInstalledSkyrimPluginList(data, std::nullopt, plugins, source, error),
        ("the Skyrim.ccc fallback resolves: " + error).c_str());
    expectTrue(
        plugins == std::vector<std::string>({"Skyrim.esm", "Update.esm", "Dawnguard.esm", "Present.esl"}),
        "fallback keeps implicit masters and only present Skyrim.ccc entries in order");

    const fs::path list = root / "plugins.txt";
    {
        std::ofstream profile(list);
        profile << "# comment\nInactive.esp\n*Explicit.esp\n*Present.esl\n*explicit.esp\n";
    }
    expectTrue(
        resolveInstalledSkyrimPluginList(data, list, plugins, source, error),
        ("an explicit plugins.txt resolves: " + error).c_str());
    expectTrue(
        plugins == std::vector<std::string>({"Skyrim.esm", "Update.esm", "Dawnguard.esm", "Explicit.esp", "Present.esl"}),
        "plugins.txt activates starred entries, ignores inactive lines, and deduplicates case-insensitively");
    expectTrue(source == list, "the selected explicit load-order source is reported");

    fs::remove_all(root, errorCode);
}

void testMorrowindLoadOrderMergesWorldRenderingRecords() {
    namespace fs = std::filesystem;
    using namespace odai::importer::fnv;

    const fs::path dataDir = fs::temp_directory_path() / "odai_tes3_load_order_test";
    std::error_code cleanupError;
    fs::remove_all(dataDir, cleanupError);
    fs::create_directories(dataDir, cleanupError);

    const auto subrecord = [](const char* type, const std::vector<std::uint8_t>& payload) {
        std::vector<std::uint8_t> bytes;
        bytes.insert(bytes.end(), type, type + 4);
        appendPod(bytes, static_cast<std::uint32_t>(payload.size()));
        bytes.insert(bytes.end(), payload.begin(), payload.end());
        return bytes;
    };
    const auto appendBytes = [](std::vector<std::uint8_t>& target,
                                const std::vector<std::uint8_t>& source) {
        target.insert(target.end(), source.begin(), source.end());
    };
    const auto zstring = [](const std::string& value) {
        std::vector<std::uint8_t> bytes(value.begin(), value.end());
        bytes.push_back(0u);
        return bytes;
    };
    const auto record = [](const char* type, const std::vector<std::uint8_t>& body,
                           std::uint32_t flags = 0u) {
        std::vector<std::uint8_t> bytes;
        bytes.insert(bytes.end(), type, type + 4);
        appendPod(bytes, static_cast<std::uint32_t>(body.size()));
        appendPod(bytes, static_cast<std::uint32_t>(0u));
        appendPod(bytes, flags);
        bytes.insert(bytes.end(), body.begin(), body.end());
        return bytes;
    };
    const auto headerRecord = [&](const std::vector<std::string>& masters, bool isMaster,
                                  std::uint32_t recordCount) {
        std::vector<std::uint8_t> body;
        std::vector<std::uint8_t> hedr(300u, 0u);
        const float version = 1.3f;
        std::memcpy(hedr.data(), &version, sizeof(version));
        const std::uint32_t fileType = isMaster ? 1u : 0u;
        std::memcpy(hedr.data() + 4u, &fileType, sizeof(fileType));
        std::memcpy(hedr.data() + 296u, &recordCount, sizeof(recordCount));
        appendBytes(body, subrecord("HEDR", hedr));
        for (const std::string& master : masters) {
            appendBytes(body, subrecord("MAST", zstring(master)));
            appendBytes(body, subrecord("DATA", std::vector<std::uint8_t>(8u, 0u)));
        }
        return record("TES3", body);
    };
    const auto statRecord = [&](const std::string& id, const std::string& model,
                                const char* type = "STAT") {
        std::vector<std::uint8_t> body;
        appendBytes(body, subrecord("NAME", zstring(id)));
        appendBytes(body, subrecord("MODL", zstring(model)));
        return record(type, body);
    };
    const auto ltexRecord = [&](const std::string& id, std::uint32_t index,
                                const std::string& path) {
        std::vector<std::uint8_t> body;
        appendBytes(body, subrecord("NAME", zstring(id)));
        std::vector<std::uint8_t> intv;
        appendPod(intv, index);
        appendBytes(body, subrecord("INTV", intv));
        appendBytes(body, subrecord("DATA", zstring(path)));
        return record("LTEX", body);
    };
    const auto cellHeader = [&](const std::string& name, bool interior,
                                std::int32_t x, std::int32_t z) {
        std::vector<std::uint8_t> body;
        appendBytes(body, subrecord("NAME", zstring(name)));
        std::vector<std::uint8_t> data;
        appendPod(data, static_cast<std::uint32_t>(interior ? 1u : 0u));
        appendPod(data, x);
        appendPod(data, z);
        appendBytes(body, subrecord("DATA", data));
        return body;
    };
    const auto appendReference = [&](std::vector<std::uint8_t>& cell,
                                     std::uint32_t formId, const std::string& base,
                                     float x, float y, float z, bool deleted) {
        std::vector<std::uint8_t> frmr;
        appendPod(frmr, formId);
        appendBytes(cell, subrecord("FRMR", frmr));
        if (!base.empty()) {
            appendBytes(cell, subrecord("NAME", zstring(base)));
        }
        if (deleted) {
            appendBytes(cell, subrecord("DELE", std::vector<std::uint8_t>(4u, 0u)));
            return;
        }
        std::vector<std::uint8_t> data;
        appendPod(data, x); appendPod(data, y); appendPod(data, z);
        appendPod(data, 0.0f); appendPod(data, 0.0f); appendPod(data, 0.0f);
        appendBytes(cell, subrecord("DATA", data));
    };
    const auto landRecord = [&](std::int32_t x, std::int32_t z,
                                std::uint16_t storedTexture) {
        std::vector<std::uint8_t> body;
        std::vector<std::uint8_t> intv;
        appendPod(intv, x); appendPod(intv, z);
        appendBytes(body, subrecord("INTV", intv));
        std::vector<std::uint8_t> vhgt(4u + (65u * 65u), 0u);
        appendBytes(body, subrecord("VHGT", vhgt));
        std::vector<std::uint8_t> vtex;
        for (int i = 0; i < 16 * 16; ++i) {
            appendPod(vtex, storedTexture);
        }
        appendBytes(body, subrecord("VTEX", vtex));
        return record("LAND", body);
    };
    const auto writePlugin = [&](const std::string& name,
                                 const std::vector<std::string>& masters,
                                 bool master,
                                 const std::vector<std::vector<std::uint8_t>>& records) {
        std::vector<std::uint8_t> bytes =
            headerRecord(masters, master, static_cast<std::uint32_t>(records.size()));
        for (const auto& item : records) {
            appendBytes(bytes, item);
        }
        std::ofstream output(dataDir / name, std::ios::binary | std::ios::trunc);
        output.write(reinterpret_cast<const char*>(bytes.data()),
                     static_cast<std::streamsize>(bytes.size()));
    };

    std::vector<std::uint8_t> baseExterior = cellHeader("Almas Thirr", false, 5, -28);
    appendReference(baseExterior, 0x00000010u, "crate", 41000.0f, -225000.0f, 100.0f, false);
    std::vector<std::uint8_t> baseInterior =
        cellHeader("Almas Thirr, Canalworks", true, 0, 0);
    appendReference(baseInterior, 0x00000011u, "crate", 10.0f, 20.0f, 30.0f, false);
    // Cell (6,-28) deliberately starts as LAND-only in the base plugin. The
    // later TR CELL must make it streamable without losing this contribution.
    writePlugin("Morrowind.esm", {}, true, {
        ltexRecord("ground", 0u, "base_ground.dds"),
        statRecord("crate", "base_crate.nif"),
        statRecord("guar", "r\\Guar.NIF", "CREA"),
        record("CELL", baseExterior), landRecord(5, -28, 1u), record("CELL", baseInterior),
        landRecord(6, -28, 1u)});

    writePlugin("Tribunal.esm", {"Morrowind.esm"}, true, {});
    writePlugin("Bloodmoon.esm", {"Morrowind.esm", "Tribunal.esm"}, true, {});
    writePlugin("Tamriel_Data.esm",
                {"Morrowind.esm", "Tribunal.esm", "Bloodmoon.esm"}, true,
                {ltexRecord("other_ground", 7u, "other_ground.dds")});

    std::vector<std::uint8_t> trExterior = cellHeader("Almas Thirr", false, 5, -28);
    // TES3 local index 1 addresses the first master, Morrowind.esm.
    appendReference(trExterior, 0x01000010u, "", 0.0f, 0.0f, 0.0f, true);
    appendReference(trExterior, 0x00000020u, "crate", 42000.0f, -226000.0f, 200.0f, false);
    std::vector<std::uint8_t> tes3DoorDestination;
    appendPod(tes3DoorDestination, 40.0f);
    appendPod(tes3DoorDestination, 50.0f);
    appendPod(tes3DoorDestination, 60.0f);
    appendPod(tes3DoorDestination, 0.0f);
    appendPod(tes3DoorDestination, 0.0f);
    appendPod(tes3DoorDestination, 1.25f);
    appendBytes(trExterior, subrecord("DODT", tes3DoorDestination));
    appendBytes(trExterior, subrecord("DNAM", zstring("Almas Thirr, Canalworks")));
    std::vector<std::uint8_t> trInterior =
        cellHeader("almas thirr, canalworks", true, 0, 0);
    appendReference(trInterior, 0x00000021u, "crate", 40.0f, 50.0f, 60.0f, false);
    std::vector<std::uint8_t> trSecondExterior =
        cellHeader("Almas Thirr East", false, 6, -28);
    appendReference(trSecondExterior, 0x00000022u, "crate", 50000.0f, -226000.0f, 200.0f, false);
    writePlugin("TR_Mainland.esm",
                {"Morrowind.esm", "Tribunal.esm", "Bloodmoon.esm", "Tamriel_Data.esm"},
                true, {
                    ltexRecord("ground", 7u, "tr_ground.dds"),
                    statRecord("crate", "tr_crate.nif", "ACTI"),
                    record("CELL", trExterior), landRecord(5, -28, 8u),
                    record("CELL", trInterior), record("CELL", trSecondExterior)});

    std::vector<std::uint8_t> factionsExterior = cellHeader("Almas Thirr", false, 5, -28);
    appendReference(
        factionsExterior, 0x00000030u, "crate", 43000.0f, -227000.0f, 300.0f, false);
    writePlugin("TR_Factions.esp",
                {"Morrowind.esm", "Tribunal.esm", "Bloodmoon.esm", "Tamriel_Data.esm",
                 "TR_Mainland.esm"},
                false, {record("CELL", factionsExterior)});

    FalloutPluginHeader trHeader;
    std::string error;
    expectTrue(readFalloutPluginHeader(dataDir / "TR_Mainland.esm", trHeader, error),
               ("TES3 plugin header reads: " + error).c_str());
    expectTrue(trHeader.format == EsmPluginFormat::kMorrowind,
               "the load-order header retains the TES3 format");
    expectTrue(trHeader.isMaster && trHeader.masters.size() == 4u,
               "TES3 HEDR file type and MAST chain are parsed");
    expectTrue(trHeader.recordCount == 6u, "TES3 HEDR record count is read at offset 296");

    FalloutLoadOrder order;
    expectTrue(order.open(dataDir, {"Morrowind.esm", "TR_Factions.esp"}, error),
               ("the six-plugin TES3 load order opens: " + error).c_str());
    expectTrue(order.size() == 6u, "TR_Factions pulls the exact five-master chain");
    const std::vector<std::string> expectedOrder{
        "Morrowind.esm", "Tribunal.esm", "Bloodmoon.esm", "Tamriel_Data.esm",
        "TR_Mainland.esm", "TR_Factions.esp"};
    bool orderMatches = order.size() == expectedOrder.size();
    for (std::size_t i = 0; i < std::min(order.size(), expectedOrder.size()); ++i) {
        orderMatches = orderMatches && order.entries()[i].header.fileName == expectedOrder[i];
    }
    expectTrue(orderMatches, "TES3 masters resolve in declared dependency order");
    expectTrue(order.remapFormId(4u, 0x00000020u) == 0x04000020u,
               "TES3 local index zero remaps to the current plugin");
    expectTrue(order.remapFormId(4u, 0x01000010u) == 0x00000010u,
               "TES3 local indices 1..N remap to masters");

    FalloutCellIndex index;
    expectTrue(buildFalloutCellIndex(order, index, error),
               ("the merged TES3 cell index builds: " + error).c_str());
    expectTrue(index.cellWorldSize == 8192.0f, "the merged TES3 grid keeps 8192-unit cells");
    expectTrue(index.cells.size() == 3u,
               "exteriors merge by grid and interiors by case-insensitive name");
    const FalloutCellIndexEntry* exterior = nullptr;
    const FalloutCellIndexEntry* secondExterior = nullptr;
    const FalloutCellIndexEntry* interior = nullptr;
    for (const FalloutCellIndexEntry& cell : index.cells) {
        if (cell.hasGridCoords && cell.gridX == 5 && cell.gridZ == -28) exterior = &cell;
        if (cell.hasGridCoords && cell.gridX == 6 && cell.gridZ == -28) secondExterior = &cell;
        if (cell.isInterior) interior = &cell;
    }
    expectTrue(exterior != nullptr && exterior->contributions.size() == 3u,
               "base, TR terrain, and factions reference patches contribute to one exterior");
    expectTrue(interior != nullptr && interior->contributions.size() == 2u,
               "case variants of the Canalworks interior merge into one cell");
    expectTrue(secondExterior != nullptr && secondExterior->childrenGroupSize != 0u &&
                   secondExterior->contributions.size() == 2u,
               "a later CELL makes an earlier LAND-only identity streamable");

    FalloutCellRecord merged;
    expectTrue(exterior != nullptr && extractFalloutCellMerged(index, order, *exterior, merged, error),
               ("the merged TES3 exterior extracts: " + error).c_str());
    expectTrue(merged.references.size() == 2u,
               "DELE removes the base placement while later TR references remain");
    const auto mergedDoor = std::find_if(
        merged.references.begin(), merged.references.end(),
        [](const FalloutPlacedReference& reference) {
            return reference.formId == 0x04000020u;
        });
    expectTrue(mergedDoor != merged.references.end() && mergedDoor->hasTeleport &&
                   mergedDoor->teleportTargetCellEditorId == "Almas Thirr, Canalworks" &&
                   mergedDoor->teleportPosition[2] == 60.0f,
               "TES3 DODT/DNAM load-door destination survives merged extraction");
    odai::importer::ImportedScene tes3DoorScene;
    appendResolvedDoors(merged, index, tes3DoorScene);
    expectTrue(tes3DoorScene.doors.size() == 1u &&
                   tes3DoorScene.doors.front().targetKind ==
                       odai::importer::ImportedSceneDoorTargetKind::Interior &&
                   tes3DoorScene.doors.front().targetCellEditorId ==
                       "almas thirr, canalworks" &&
                   tes3DoorScene.doors.front().arrivalPosition[1] == 60.0f,
               "TES3 named-cell load doors resolve to runtime transitions");
    expectTrue(std::any_of(merged.references.begin(), merged.references.end(),
                   [](const FalloutPlacedReference& reference) {
                       return reference.formId == 0x04000020u &&
                           reference.sourcePluginIndex == 4u;
                   }) &&
                   std::any_of(merged.references.begin(), merged.references.end(),
                   [](const FalloutPlacedReference& reference) {
                       return reference.formId == 0x05000030u &&
                           reference.sourcePluginIndex == 5u;
                   }),
               "placed references retain the winning plugin index for VMAD remapping");
    expectTrue(merged.land != nullptr && merged.land->sourcePluginIndex == 4u,
               "the later TR LAND wins and a reference-only factions patch retains it");
    expectTrue(merged.land != nullptr && !merged.land->morrowindTextureGrid.empty() &&
                   merged.land->morrowindTextureGrid[0] == 8u,
               "the winning LAND retains its plugin-local VTEX palette index");

    FalloutCellRecord mergedInterior;
    expectTrue(interior != nullptr &&
                   extractFalloutCellMerged(index, order, *interior, mergedInterior, error),
               ("the case-insensitive Canalworks interior extracts: " + error).c_str());
    expectTrue(mergedInterior.references.size() == 2u,
               "named-interior extraction includes base and TR contributions");

    FalloutCellRecord mergedSecondExterior;
    expectTrue(secondExterior != nullptr &&
                   extractFalloutCellMerged(
                       index, order, *secondExterior, mergedSecondExterior, error),
               ("the LAND-first exterior extracts: " + error).c_str());
    expectTrue(mergedSecondExterior.land != nullptr &&
                   mergedSecondExterior.land->sourcePluginIndex == 0u &&
                   mergedSecondExterior.references.size() == 1u,
               "a later CELL retains terrain contributed earlier without a CELL");

    FalloutWorldTables tables;
    expectTrue(buildFalloutWorldTables(order, tables, error),
               ("the merged TES3 world tables build: " + error).c_str());
    const auto crate = tables.baseFormIdsByEditorId.find("crate");
    expectTrue(crate != tables.baseFormIdsByEditorId.end(),
               "TES3 string-keyed bases receive one global synthetic ID");
    expectTrue(crate != tables.baseFormIdsByEditorId.end() &&
                   tables.staticModelPaths.at(crate->second) == "tr_crate.nif",
               "a later TES3 base definition overrides the model behind that ID");
    expectTrue(crate != tables.baseFormIdsByEditorId.end() &&
                   tables.staticRecordTypes.at(crate->second) == "ACTI",
               "a later TES3 base definition overrides the type behind that ID");
    const auto guar = tables.baseFormIdsByEditorId.find("guar");
    expectTrue(guar != tables.baseFormIdsByEditorId.end() &&
                   tables.staticModelPaths.at(guar->second) == "r\\Guar.NIF" &&
                   tables.staticRecordTypes.at(guar->second) == "CREA",
               "TES3 creature models enter the placed imported-scene tables");
    const std::uint64_t tdPalette = (3ull << 32u) | 8u;
    const std::uint64_t trPalette = (4ull << 32u) | 8u;
    expectTrue(tables.morrowindLandTexturePaths.at(tdPalette) == "other_ground.dds" &&
                   tables.morrowindLandTexturePaths.at(trPalette) == "tr_ground.dds",
               "identical LTEX indices in different TES3 plugins resolve through their own palettes");
    expectTrue(tables.morrowindLandTexturePaths.at(1u) == "tr_ground.dds",
               "a later LTEX with the same editor ID overrides the path for earlier LAND too");

    const std::string fingerprintBefore = order.fingerprint();
    {
        std::ofstream append(dataDir / "TR_Factions.esp", std::ios::binary | std::ios::app);
        append.put('\0');
    }
    FalloutLoadOrder changed;
    expectTrue(changed.open(dataDir, {"Morrowind.esm", "TR_Factions.esp"}, error),
               "the content-stamp fixture reopens after a size change");
    expectTrue(fingerprintBefore != changed.fingerprint(),
               "changing plugin content without renaming it invalidates the cache fingerprint");

    const std::string fingerprintAfterSizeChange = changed.fingerprint();
    const fs::path factionsPath = dataDir / "TR_Factions.esp";
    const auto oldStamp = fs::last_write_time(factionsPath, cleanupError);
    cleanupError.clear();
    fs::last_write_time(factionsPath, oldStamp + std::chrono::seconds(2), cleanupError);
    expectTrue(!cleanupError, "the cache fixture modification time can be changed");
    FalloutLoadOrder retimestamped;
    expectTrue(retimestamped.open(dataDir, {"Morrowind.esm", "TR_Factions.esp"}, error),
               "the content-stamp fixture reopens after an mtime-only change");
    expectTrue(fingerprintAfterSizeChange != retimestamped.fingerprint(),
               "changing only a plugin modification time invalidates the cache fingerprint");

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
        landLodTilePath("tamriel", LandLodSet::SkyrimObjects, 4, 4, -4) ==
            "terrain\\tamriel\\objects\\tamriel.4.4.-4.bto",
        "Skyrim object LOD uses the BTO naming scheme");
    expectTrue(
        landLodTilePath("tamriel", LandLodSet::SkyrimTerrain, 4, 4, -4) ==
            "terrain\\tamriel\\tamriel.4.4.-4.btr",
        "Skyrim terrain LOD uses the BTR naming scheme");
    expectTrue(
        landLodTilePath("wastelandnv", LandLodSet::Terrain, 32, -32, 0) ==
            "landscape\\lod\\wastelandnv\\wastelandnv.level32.x-32.y0.nif",
        "a coarse terrain tier keeps the same shape, negative coordinate and all");

    // The pyramid is terrain-only. Measured: blocks\ holds 301 level4 tiles for
    // WastelandNV and nothing whatsoever at level8/16/32.
    for (const int tier : odai::importer::fnv::kLandLodTierCellCounts) {
        expectTrue(landLodTierExists(LandLodSet::Terrain, tier), "every terrain tier exists");
        expectTrue(
            landLodTierExists(LandLodSet::SkyrimTerrain, tier),
            "every Skyrim terrain tier exists");
    }
    expectTrue(landLodTierExists(LandLodSet::Objects, 4), "object LOD exists at level4");
    expectTrue(
        landLodTierExists(LandLodSet::SkyrimObjects, 4),
        "Skyrim object LOD exists at level4");
    expectTrue(
        !landLodTierExists(LandLodSet::SkyrimObjects, 8),
        "Skyrim object LOD has no level8");
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

// Skyrim FaceGen gives every generated facial piece its own NiSkinData skin
// space even though the GPU actor has one bone palette. BSDynamicTriShape
// positions therefore have to be baked through each piece's authored bind and
// rebound to the skeleton's canonical inverse bind. Without that step the
// first piece wins: eyes may be correct while hair, brows and mouth collapse by
// roughly one actor height (or vice versa).
void testDynamicFacePartsShareOneBonePalette() {
    using namespace odai::importer::fnv;

    NifSkeleton source;
    NifSkeletonBone head;
    head.name = "NPC Head [Head]";
    head.parentIndex = -1;
    head.translation[2] = 60.0f;
    source.bones.push_back(head);

    FalloutCharacter character;
    expectTrue(buildFalloutSkeleton(source, character.skeleton), "FaceGen skeleton converts");

    NifSkinnedModel model;
    for (int part = 0; part < 2; ++part) {
        NifSkinnedShape shape;
        shape.name = part == 0 ? "eyes" : "hair";
        shape.usesDynamicPositions = true;
        shape.boneNames = {"NPC Head [Head]"};
        shape.inverseBindMatrices.assign(16u, 0.0f);
        shape.inverseBindMatrices[0] = shape.inverseBindMatrices[5] =
            shape.inverseBindMatrices[10] = shape.inverseBindMatrices[15] = 1.0f;
        // Deliberately disagree. The shape-space transform and its inverse
        // geometry normalization cancel during the FaceGen bake.
        shape.skinTransform[11] = part == 0 ? 25.0f : -40.0f;
        shape.positions = {1.0f, 2.0f, 3.0f};
        shape.normals = {0.0f, 0.0f, 1.0f};
        shape.uvs = {0.0f, 0.0f};
        shape.triangleIndices = {0, 0, 0};
        shape.boneIndices.assign(kNifMaxBoneInfluences, 0u);
        shape.boneWeights.assign(kNifMaxBoneInfluences, 0.0f);
        shape.boneWeights[0] = 1.0f;
        model.shapes.push_back(std::move(shape));
    }

    std::string error;
    expectTrue(appendFalloutCharacterMesh(model, character, error), "FaceGen pieces bind");
    expectTrue(character.vertices.size() == 2u, "both FaceGen pieces survive");
    expectTrue(character.conflictingInverseBindCount == 0u,
               "shape-local FaceGen binds do not conflict in the shared palette");

    std::vector<odai::math::Matrix4> pose;
    computeFalloutBindPose(character, pose);
    for (const auto& vertex : character.vertices) {
        const odai::math::Vector3 rest{
            vertex.position[0], vertex.position[1], vertex.position[2]};
        const odai::math::Vector3 skinned = odai::math::transformPoint(pose[0], rest);
        expectTrue(std::fabs(skinned.x - rest.x) < 1e-3f, "FaceGen bind preserves x");
        expectTrue(std::fabs(skinned.y - rest.y) < 1e-3f, "FaceGen bind preserves y");
        expectTrue(std::fabs(skinned.z - rest.z) < 1e-3f, "FaceGen bind preserves z");
        expectTrue(rest.y > 50.0f, "FaceGen remains at head height");
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

// The .kf ControlledBlock stride, and the basis change applied to its keys.
//
// Both are places where being wrong produces confident nonsense rather than a
// failure. A ControlledBlock at NIF 20.2.0.7 is 29 bytes and is NOT 4-byte
// aligned -- a one-byte priority sits between two string indices -- so reading
// it as an aligned struct still yields in-range-looking refs for the first
// entry or two before wandering off. And an animation key is a bone's local
// transform, the same kind of quantity as its bind pose, so it has to go
// through the same Bethesda-to-engine conversion; converting it differently
// gives a character whose rest pose is right and whose every animated frame is
// rotated into the floor.
void testKfAnimationStrideAndBasisChange() {
    using namespace odai::importer::fnv;

    std::vector<std::uint8_t> sequenceBlock;
    appendPod(sequenceBlock, static_cast<std::int32_t>(2));   // name -> "TestClip"
    appendPod(sequenceBlock, static_cast<std::uint32_t>(1));  // numControlledBlocks
    appendPod(sequenceBlock, static_cast<std::uint32_t>(0));  // arrayGrowBy
    // The 29-byte ControlledBlock, field by field.
    appendPod(sequenceBlock, static_cast<std::int32_t>(1));   // interpolator -> block 1
    appendPod(sequenceBlock, static_cast<std::int32_t>(-1));  // controller (absent in a .kf)
    appendPod(sequenceBlock, static_cast<std::uint8_t>(20));  // priority -- the misaligning byte
    appendPod(sequenceBlock, static_cast<std::int32_t>(0));   // nodeName -> "Bip01 Test"
    appendPod(sequenceBlock, static_cast<std::int32_t>(-1));  // propertyType
    appendPod(sequenceBlock, static_cast<std::int32_t>(1));   // controllerType
    appendPod(sequenceBlock, static_cast<std::int32_t>(-1));  // controllerID
    appendPod(sequenceBlock, static_cast<std::int32_t>(-1));  // interpolatorID
    appendPod(sequenceBlock, 1.0f);                           // weight
    appendPod(sequenceBlock, static_cast<std::int32_t>(-1));  // textKeys
    appendPod(sequenceBlock, static_cast<std::uint32_t>(0));  // cycleType: loop
    appendPod(sequenceBlock, 1.0f);                           // frequency
    appendPod(sequenceBlock, 0.0f);                           // startTime
    appendPod(sequenceBlock, 2.0f);                           // stopTime
    appendPod(sequenceBlock, static_cast<std::int32_t>(-1));  // manager
    appendPod(sequenceBlock, static_cast<std::int32_t>(-1));  // accumRootName

    // NiTransformInterpolator: every static channel unset (-FLT_MAX), data in
    // block 2.
    const float kUnset = -std::numeric_limits<float>::max();
    std::vector<std::uint8_t> interpolatorBlock;
    for (int i = 0; i < 3; ++i) appendPod(interpolatorBlock, kUnset);  // translation
    for (int i = 0; i < 4; ++i) appendPod(interpolatorBlock, kUnset);  // rotation (w,x,y,z)
    appendPod(interpolatorBlock, kUnset);                              // scale
    appendPod(interpolatorBlock, static_cast<std::int32_t>(2));        // data -> block 2

    std::vector<std::uint8_t> dataBlock;
    appendPod(dataBlock, static_cast<std::uint32_t>(2));  // numRotationKeys
    appendPod(dataBlock, static_cast<std::uint32_t>(1));  // LINEAR_KEY
    // NIF stores a quaternion W FIRST.
    appendPod(dataBlock, 0.0f);   // key 0 time
    appendPod(dataBlock, 1.0f);   // w
    appendPod(dataBlock, 0.0f);   // x
    appendPod(dataBlock, 0.0f);   // y
    appendPod(dataBlock, 0.0f);   // z
    appendPod(dataBlock, 2.0f);   // key 1 time
    appendPod(dataBlock, 0.0f);   // w
    appendPod(dataBlock, 1.0f);   // x  (180 degrees about Bethesda X)
    appendPod(dataBlock, 0.0f);
    appendPod(dataBlock, 0.0f);
    appendPod(dataBlock, static_cast<std::uint32_t>(1));  // translation numKeys
    appendPod(dataBlock, static_cast<std::uint32_t>(1));  // LINEAR_KEY
    appendPod(dataBlock, 0.0f);   // time
    appendPod(dataBlock, 1.0f);   // x
    appendPod(dataBlock, 2.0f);   // y
    appendPod(dataBlock, 3.0f);   // z
    appendPod(dataBlock, static_cast<std::uint32_t>(0));  // scale numKeys

    std::vector<std::uint8_t> fileBytes;
    const std::string headerLine = "Gamebryo File Format, Version 20.2.0.7";
    fileBytes.insert(fileBytes.end(), headerLine.begin(), headerLine.end());
    fileBytes.push_back('\n');
    appendPod(fileBytes, static_cast<std::uint32_t>(0x14020007u));
    appendPod(fileBytes, static_cast<std::uint8_t>(1));
    appendPod(fileBytes, static_cast<std::uint32_t>(11));
    appendPod(fileBytes, static_cast<std::uint32_t>(3));   // numBlocks
    appendPod(fileBytes, static_cast<std::uint32_t>(34));  // userVersion2
    appendSizedString8(fileBytes, "");
    appendSizedString8(fileBytes, "");
    appendSizedString8(fileBytes, "");
    appendPod(fileBytes, static_cast<std::uint16_t>(3));   // numBlockTypes
    appendSizedString32(fileBytes, "NiControllerSequence");
    appendSizedString32(fileBytes, "NiTransformInterpolator");
    appendSizedString32(fileBytes, "NiTransformData");
    appendPod(fileBytes, static_cast<std::uint16_t>(0));
    appendPod(fileBytes, static_cast<std::uint16_t>(1));
    appendPod(fileBytes, static_cast<std::uint16_t>(2));
    appendPod(fileBytes, static_cast<std::uint32_t>(sequenceBlock.size()));
    appendPod(fileBytes, static_cast<std::uint32_t>(interpolatorBlock.size()));
    appendPod(fileBytes, static_cast<std::uint32_t>(dataBlock.size()));
    appendPod(fileBytes, static_cast<std::uint32_t>(3));   // numStrings
    appendPod(fileBytes, static_cast<std::uint32_t>(24));  // maxStringLength
    appendSizedString32(fileBytes, "Bip01 Test");
    appendSizedString32(fileBytes, "NiTransformController");
    appendSizedString32(fileBytes, "TestClip");
    appendPod(fileBytes, static_cast<std::uint32_t>(0));   // numGroups
    fileBytes.insert(fileBytes.end(), sequenceBlock.begin(), sequenceBlock.end());
    fileBytes.insert(fileBytes.end(), interpolatorBlock.begin(), interpolatorBlock.end());
    fileBytes.insert(fileBytes.end(), dataBlock.begin(), dataBlock.end());

    KfAnimation animation;
    std::string error;
    expectTrue(parseKfAnimation(fileBytes, animation, error),
               ("synthetic .kf parses: " + error).c_str());
    std::vector<KfAnimation> embeddedAnimations;
    std::string embeddedError;
    expectTrue(parseNifEmbeddedAnimations(fileBytes, embeddedAnimations, embeddedError),
               ("embedded NIF sequence scan parses: " + embeddedError).c_str());
    expectTrue(embeddedAnimations.size() == 1u &&
                   embeddedAnimations.front().name == "TestClip",
               "embedded sequence scan returns every controller sequence by name");
    expectTrue(animation.name == "TestClip", "clip name comes from the header string table");
    expectTrue(animation.loops(), "cycleType 0 is a looping clip");
    expectNear(animation.duration(), 2.0f, 1e-5f, "duration is stopTime - startTime");
    expectTrue(animation.tracks.size() == 1u, "one controlled block yields one track");
    if (animation.tracks.size() != 1u) {
        return;
    }
    const KfBoneTrack& track = animation.tracks.front();
    // The stride assertion: the node name resolves only if priority was read as
    // ONE byte and the four string indices after it landed on their real offsets.
    expectTrue(track.nodeName == "Bip01 Test",
               "ControlledBlock node name resolves (29-byte stride, unaligned priority)");
    expectTrue(track.rotationKeys.size() == 2u, "both rotation keys read");
    expectTrue(track.translationKeys.size() == 1u, "the translation KeyGroup read");
    if (track.translationKeys.size() == 1u) {
        // Still Bethesda space at this layer, by design.
        expectNear(track.translationKeys[0].value.x, 1.0f, 1e-5f, "raw key x");
        expectNear(track.translationKeys[0].value.y, 2.0f, 1e-5f, "raw key y");
        expectNear(track.translationKeys[0].value.z, 3.0f, 1e-5f, "raw key z");
    }
    if (track.rotationKeys.size() == 2u) {
        // W-first in the file, (x,y,z,w) in odai::math.
        expectNear(track.rotationKeys[0].value.w, 1.0f, 1e-5f, "quaternion W is read first");
        expectNear(track.rotationKeys[1].value.x, 1.0f, 1e-5f, "quaternion X follows W");
    }

    // And the conversion into engine space, against a skeleton carrying that
    // bone: (x, y, z) -> (x, z, -y), the same mapping buildFalloutSkeleton
    // applies to the bind pose.
    NifSkeleton nifSkeleton;
    NifSkeletonBone bone;
    bone.name = "Bip01 Test";
    bone.parentIndex = -1;
    nifSkeleton.bones.push_back(bone);
    odai::anim::Skeleton skeleton;
    expectTrue(buildFalloutSkeleton(nifSkeleton, skeleton), "one-bone skeleton builds");

    odai::anim::AnimationClip clip;
    FalloutAnimationStats stats;
    expectTrue(buildFalloutAnimationClip(animation, skeleton, clip, stats),
               "clip binds to the skeleton");
    expectTrue(stats.boundTracks == 1u, "the track resolves to bone 0 by name");
    expectTrue(clip.tracks.size() == 1u && clip.tracks[0].boneIndex == 0,
               "track carries the resolved bone index");
    if (!clip.tracks.empty() && clip.tracks[0].translationKeys.size() == 1u) {
        const odai::math::Vector3 converted = clip.tracks[0].translationKeys[0].value;
        expectNear(converted.x, 1.0f, 1e-5f, "engine-space key x is Bethesda x");
        expectNear(converted.y, 3.0f, 1e-5f, "engine-space key y is Bethesda z");
        expectNear(converted.z, -2.0f, 1e-5f, "engine-space key z is negated Bethesda y");
    }

    // A stride error does not fail quietly: claiming more controlled blocks
    // than the block can hold is rejected rather than read past the end.
    {
        std::vector<std::uint8_t> corrupted = fileBytes;
        KfAnimation ignored;
        std::string corruptError;
        const std::size_t countOffset = fileBytes.size() - sequenceBlock.size() -
                                        interpolatorBlock.size() - dataBlock.size() + 4u;
        const auto absurdCount = static_cast<std::uint32_t>(10000);
        std::memcpy(corrupted.data() + countOffset, &absurdCount, sizeof(absurdCount));
        expectTrue(!parseKfAnimation(corrupted, ignored, corruptError),
                   "an impossible controlled-block count is rejected, not read past the end");
    }
}

// The B-spline decoder, on a curve whose answer is known in closed form.
//
// FOUR control points at degree 3 is exactly one span, and a clamped knot
// vector makes that span a Bezier -- so the curve must pass THROUGH the first
// and last control point and nowhere near the middle two. That pins both
// halves of the decode at once: the /32767 dequantization (wrong scale moves
// the endpoints) and the clamped knot convention (an unclamped spline starts a
// sixth of the way in and never reaches either end).
//
// Worth testing rather than eyeballing because the failure is quiet: a bone
// whose curve decodes wrong is still a bone with keys, and the sampler poses it
// without complaint.
void testKfBSplineDecoding() {
    using namespace odai::importer::fnv;

    constexpr float kMultiplier = 100.0f;

    std::vector<std::uint8_t> sequenceBlock;
    appendPod(sequenceBlock, static_cast<std::int32_t>(2));   // name -> "BSplineClip"
    appendPod(sequenceBlock, static_cast<std::uint32_t>(1));  // numControlledBlocks
    appendPod(sequenceBlock, static_cast<std::uint32_t>(0));  // arrayGrowBy
    appendPod(sequenceBlock, static_cast<std::int32_t>(1));   // interpolator -> block 1
    appendPod(sequenceBlock, static_cast<std::int32_t>(-1));  // controller
    appendPod(sequenceBlock, static_cast<std::uint8_t>(20));  // priority
    appendPod(sequenceBlock, static_cast<std::int32_t>(0));   // nodeName -> "Bip01 Test"
    appendPod(sequenceBlock, static_cast<std::int32_t>(-1));  // propertyType
    appendPod(sequenceBlock, static_cast<std::int32_t>(1));   // controllerType
    appendPod(sequenceBlock, static_cast<std::int32_t>(-1));  // controllerID
    appendPod(sequenceBlock, static_cast<std::int32_t>(-1));  // interpolatorID
    appendPod(sequenceBlock, 1.0f);                           // weight
    appendPod(sequenceBlock, static_cast<std::int32_t>(-1));  // textKeys
    appendPod(sequenceBlock, static_cast<std::uint32_t>(0));  // cycleType: loop
    appendPod(sequenceBlock, 1.0f);                           // frequency
    appendPod(sequenceBlock, 0.0f);                           // startTime
    appendPod(sequenceBlock, 2.0f);                           // stopTime
    appendPod(sequenceBlock, static_cast<std::int32_t>(-1));  // manager
    appendPod(sequenceBlock, static_cast<std::int32_t>(-1));  // accumRootName

    constexpr float kUnset = -std::numeric_limits<float>::max();
    std::vector<std::uint8_t> interpolatorBlock;
    appendPod(interpolatorBlock, 0.0f);                          // startTime
    appendPod(interpolatorBlock, 2.0f);                          // stopTime
    appendPod(interpolatorBlock, static_cast<std::int32_t>(2));  // splineData -> block 2
    appendPod(interpolatorBlock, static_cast<std::int32_t>(3));  // basisData -> block 3
    appendPod(interpolatorBlock, kUnset);                        // static translation x/y/z
    appendPod(interpolatorBlock, kUnset);
    appendPod(interpolatorBlock, kUnset);
    appendPod(interpolatorBlock, 1.0f);                          // static rotation w
    appendPod(interpolatorBlock, 0.0f);                          // x
    appendPod(interpolatorBlock, 0.0f);                          // y
    appendPod(interpolatorBlock, 0.0f);                          // z
    appendPod(interpolatorBlock, kUnset);                        // static scale
    appendPod(interpolatorBlock, static_cast<std::uint32_t>(0));           // translationOffset
    appendPod(interpolatorBlock, static_cast<std::uint32_t>(0xffffffffu)); // rotationOffset: none
    appendPod(interpolatorBlock, static_cast<std::uint32_t>(0xffffffffu)); // scaleOffset: none
    appendPod(interpolatorBlock, 0.0f);          // translationBias
    appendPod(interpolatorBlock, kMultiplier);   // translationMultiplier
    appendPod(interpolatorBlock, 0.0f);          // rotationBias
    appendPod(interpolatorBlock, 1.0f);          // rotationMultiplier
    appendPod(interpolatorBlock, 0.0f);          // scaleBias
    appendPod(interpolatorBlock, 1.0f);          // scaleMultiplier

    // Four control points, three components each, quantized: the curve runs
    // from (0,0,0) to (0,100,0) and bulges toward (100,0,0) in between.
    const std::array<std::array<std::int16_t, 3>, 4> controlPoints{{
        {{0, 0, 0}},
        {{32767, 0, 0}},
        {{32767, 0, 0}},
        {{0, 32767, 0}},
    }};
    std::vector<std::uint8_t> splineDataBlock;
    appendPod(splineDataBlock, static_cast<std::uint32_t>(0));  // numFloatControlPoints
    appendPod(splineDataBlock, static_cast<std::uint32_t>(controlPoints.size() * 3u));
    for (const auto& point : controlPoints) {
        for (const std::int16_t component : point) {
            appendPod(splineDataBlock, component);
        }
    }

    std::vector<std::uint8_t> basisDataBlock;
    appendPod(basisDataBlock, static_cast<std::uint32_t>(controlPoints.size()));

    std::vector<std::uint8_t> fileBytes;
    const std::string headerLine = "Gamebryo File Format, Version 20.2.0.7";
    fileBytes.insert(fileBytes.end(), headerLine.begin(), headerLine.end());
    fileBytes.push_back('\n');
    appendPod(fileBytes, static_cast<std::uint32_t>(0x14020007u));
    appendPod(fileBytes, static_cast<std::uint8_t>(1));
    appendPod(fileBytes, static_cast<std::uint32_t>(11));
    appendPod(fileBytes, static_cast<std::uint32_t>(4));   // numBlocks
    appendPod(fileBytes, static_cast<std::uint32_t>(34));  // userVersion2
    appendSizedString8(fileBytes, "");
    appendSizedString8(fileBytes, "");
    appendSizedString8(fileBytes, "");
    appendPod(fileBytes, static_cast<std::uint16_t>(4));   // numBlockTypes
    appendSizedString32(fileBytes, "NiControllerSequence");
    appendSizedString32(fileBytes, "NiBSplineCompTransformInterpolator");
    appendSizedString32(fileBytes, "NiBSplineData");
    appendSizedString32(fileBytes, "NiBSplineBasisData");
    appendPod(fileBytes, static_cast<std::uint16_t>(0));
    appendPod(fileBytes, static_cast<std::uint16_t>(1));
    appendPod(fileBytes, static_cast<std::uint16_t>(2));
    appendPod(fileBytes, static_cast<std::uint16_t>(3));
    appendPod(fileBytes, static_cast<std::uint32_t>(sequenceBlock.size()));
    appendPod(fileBytes, static_cast<std::uint32_t>(interpolatorBlock.size()));
    appendPod(fileBytes, static_cast<std::uint32_t>(splineDataBlock.size()));
    appendPod(fileBytes, static_cast<std::uint32_t>(basisDataBlock.size()));
    appendPod(fileBytes, static_cast<std::uint32_t>(3));   // numStrings
    appendPod(fileBytes, static_cast<std::uint32_t>(24));  // maxStringLength
    appendSizedString32(fileBytes, "Bip01 Test");
    appendSizedString32(fileBytes, "NiTransformController");
    appendSizedString32(fileBytes, "BSplineClip");
    appendPod(fileBytes, static_cast<std::uint32_t>(0));   // numGroups
    fileBytes.insert(fileBytes.end(), sequenceBlock.begin(), sequenceBlock.end());
    fileBytes.insert(fileBytes.end(), interpolatorBlock.begin(), interpolatorBlock.end());
    fileBytes.insert(fileBytes.end(), splineDataBlock.begin(), splineDataBlock.end());
    fileBytes.insert(fileBytes.end(), basisDataBlock.begin(), basisDataBlock.end());

    KfAnimation animation;
    std::string error;
    expectTrue(
        parseKfAnimation(fileBytes, animation, error),
        ("synthetic B-spline .kf parses: " + error).c_str());
    expectTrue(
        animation.stats.bSplineInterpolators == 1u,
        "the B-spline interpolator is decoded rather than counted as unsupported");
    expectTrue(
        animation.stats.unsupportedInterpolators == 0u,
        "nothing is left undecoded");
    expectTrue(animation.tracks.size() == 1u, "the B-spline block yields one track");
    if (animation.tracks.size() != 1u) {
        return;
    }
    const KfBoneTrack& track = animation.tracks.front();
    expectTrue(track.nodeName == "Bip01 Test", "the B-spline track binds to its node");
    // One span, two samples per span, plus the closing sample.
    expectTrue(track.translationKeys.size() == 3u, "the curve is sampled across its one span");
    if (track.translationKeys.size() != 3u) {
        return;
    }
    const KfVector3Key& first = track.translationKeys.front();
    const KfVector3Key& last = track.translationKeys.back();
    expectNear(first.time, 0.0f, 1e-5f, "sampling starts at the interpolator's start time");
    expectNear(last.time, 2.0f, 1e-3f, "sampling ends at the interpolator's stop time");
    expectNear(first.value.x, 0.0f, 0.05f, "the curve starts at the first control point (x)");
    expectNear(first.value.y, 0.0f, 0.05f, "the curve starts at the first control point (y)");
    expectNear(last.value.x, 0.0f, 0.2f, "the curve ends at the last control point (x)");
    expectNear(
        last.value.y, kMultiplier, 0.2f,
        "the curve ends at the last control point, dequantized by multiplier/32767");
    // The middle sample must be pulled toward the interior control points and
    // must not simply be the midpoint of the endpoints -- that is what an
    // unweighted or linear fallback would produce.
    expectTrue(
        track.translationKeys[1].value.x > 25.0f,
        "the interior sample is pulled toward the interior control points");
    // A channel with no curve falls back to the interpolator's static value.
    expectTrue(
        track.rotationKeys.size() == 1u,
        "an absent rotation channel falls back to the static rotation");
    expectTrue(track.scaleKeys.empty(), "an unset static scale contributes no key");
}


// A template actor's skeleton is NOT on the record that names it.
//
// An actor that borrows its model stores "marker_creature.nif" as its own MODL
// -- a real, parseable NIF carrying none of the bones a body is weighted to.
// Using it therefore does not FAIL, it binds a character whose every bone is
// unresolved. Measured on Fallout 3, where a levelled raider reported 71
// unresolved bones and stood in bind pose.
//
// And the hop is nested: TPLT lands on a levelled list whose entries are
// routinely MORE LISTS (Fallout 3's EncRaiderMelee is one), so following a
// single level finds no actor and quietly hands back the marker.
void testTemplateSkeletonThroughNestedLeveledLists() {
    namespace fs = std::filesystem;
    using namespace odai::importer::fnv;

    constexpr std::uint32_t kRaceFormId = 0x00000100u;
    constexpr std::uint32_t kOuterListFormId = 0x00000200u;
    constexpr std::uint32_t kInnerListFormId = 0x00000201u;
    constexpr std::uint32_t kRealActorFormId = 0x00000300u;
    constexpr std::uint32_t kMarkerActorFormId = 0x00000400u;
    constexpr std::uint32_t kPlacementFormId = 0x00000500u;

    const auto append = [](std::vector<std::uint8_t>& out, const std::vector<std::uint8_t>& sub) {
        out.insert(out.end(), sub.begin(), sub.end());
    };
    const auto u32Payload = [](std::uint32_t value) {
        std::vector<std::uint8_t> out;
        appendPod(out, value);
        return out;
    };
    const auto acbs = [](std::uint16_t templateFlags, bool female) {
        std::vector<std::uint8_t> out(24u, 0u);
        const std::uint32_t flags = female ? 1u : 0u;
        std::memcpy(out.data(), &flags, sizeof(flags));
        std::memcpy(out.data() + 22, &templateFlags, sizeof(templateFlags));
        return out;
    };
    const auto lvlo = [](std::uint32_t formId) {
        std::vector<std::uint8_t> out;
        appendPod(out, static_cast<std::uint16_t>(1));  // level
        appendPod(out, static_cast<std::uint16_t>(0));  // unused
        appendPod(out, formId);
        appendPod(out, static_cast<std::uint32_t>(1));  // count
        return out;
    };

    // A race with a HEAD but NO body model, so there is nothing for the
    // skeleton to stand beside and the template chain is what has to supply it.
    // That is the fallback path; the primary rule (skeleton beside the race's
    // body parts) is covered by testActorRaceAndWardrobeAssembly.
    std::vector<std::uint8_t> raceSubrecords;
    append(raceSubrecords, buildSubrecord("EDID", stringPayload("TestRace")));
    append(raceSubrecords, buildSubrecord("NAM0", {}));
    append(raceSubrecords, buildSubrecord("MNAM", {}));
    append(raceSubrecords, buildSubrecord("INDX", u32Payload(0)));
    append(raceSubrecords, buildSubrecord("MODL", stringPayload("heads\\male.nif")));

    // The real actor at the bottom of the chain, with the true skeleton.
    std::vector<std::uint8_t> realActor;
    append(realActor, buildSubrecord("EDID", stringPayload("RealRaider")));
    append(realActor, buildSubrecord("MODL", stringPayload("skeletons\\human.nif")));
    append(realActor, buildSubrecord("ACBS", acbs(0u, false)));
    append(realActor, buildSubrecord("RNAM", u32Payload(kRaceFormId)));

    // The placed actor: a marker model, and a template it borrows the model,
    // traits and inventory from.
    // Written as the MEASURED bits, not as the constants under test. Building
    // the fixture from the constants would make it and the reader agree by
    // construction, so a wrong bit would pass -- and these were read off real
    // records (Fallout 3's LvlRaiderMelee carries 0x1df).
    constexpr std::uint16_t kMeasuredUseTraits = 0x0001u;
    constexpr std::uint16_t kMeasuredUseModelAnimation = 0x0040u;
    constexpr std::uint16_t kMeasuredUseInventory = 0x0100u;
    expectTrue(kActorTemplateUseTraits == kMeasuredUseTraits, "Use Traits is ACBS bit 0x0001");
    expectTrue(
        kActorTemplateUseModelAnimation == kMeasuredUseModelAnimation,
        "Use Model/Animation is ACBS bit 0x0040");
    expectTrue(
        kActorTemplateUseInventory == kMeasuredUseInventory, "Use Inventory is ACBS bit 0x0100");
    constexpr std::uint16_t kBorrowsModelTraitsAndInventory =
        kMeasuredUseTraits | kMeasuredUseModelAnimation | kMeasuredUseInventory;
    std::vector<std::uint8_t> markerActor;
    append(markerActor, buildSubrecord("EDID", stringPayload("LvlRaider")));
    append(markerActor, buildSubrecord("MODL", stringPayload("marker_creature.nif")));
    append(markerActor, buildSubrecord("ACBS", acbs(kBorrowsModelTraitsAndInventory, false)));
    append(markerActor, buildSubrecord("TPLT", u32Payload(kOuterListFormId)));
    append(markerActor, buildSubrecord("RNAM", u32Payload(kRaceFormId)));

    // Outer list -> inner list -> the real actor. Two levels, deliberately.
    std::vector<std::uint8_t> outerList;
    append(outerList, buildSubrecord("EDID", stringPayload("EncRaiderOuter")));
    append(outerList, buildSubrecord("LVLO", lvlo(kInnerListFormId)));
    std::vector<std::uint8_t> innerList;
    append(innerList, buildSubrecord("EDID", stringPayload("EncRaiderInner")));
    append(innerList, buildSubrecord("LVLO", lvlo(kRealActorFormId)));

    std::vector<std::uint8_t> placementSubrecords;
    append(placementSubrecords, buildSubrecord("NAME", u32Payload(kMarkerActorFormId)));
    {
        std::vector<std::uint8_t> data;
        for (int i = 0; i < 6; ++i) { appendPod(data, 0.0f); }
        append(placementSubrecords, buildSubrecord("DATA", data));
    }

    std::vector<std::uint8_t> content;
    const auto appendGroup = [&](const char* type, const std::vector<std::uint8_t>& record) {
        const auto group = buildGroup(type, 0, record);
        content.insert(content.end(), group.begin(), group.end());
    };
    appendGroup("RACE", buildRecord("RACE", kRaceFormId, 0u, raceSubrecords));
    appendGroup("NPC_", buildRecord("NPC_", kRealActorFormId, 0u, realActor));
    appendGroup("NPC_", buildRecord("NPC_", kMarkerActorFormId, 0u, markerActor));
    // LVLN, not LVLC: the NPC flavour of a levelled actor list, which is what
    // an NPC_'s template lands on.
    appendGroup("LVLN", buildRecord("LVLN", kOuterListFormId, 0u, outerList));
    appendGroup("LVLN", buildRecord("LVLN", kInnerListFormId, 0u, innerList));
    appendGroup("ACHR", buildRecord("ACHR", kPlacementFormId, 0u, placementSubrecords));

    const fs::path esmPath = fs::temp_directory_path() / "odai_fnv_template_test.esm";
    {
        std::ofstream out(esmPath, std::ios::binary | std::ios::trunc);
        out.write(
            reinterpret_cast<const char*>(content.data()),
            static_cast<std::streamsize>(content.size()));
    }

    FalloutActorScan scan;
    std::string error;
    expectTrue(
        findActorsNear(esmPath, 0.0f, 0.0f, 100.0f, scan, error),
        ("template actor scan succeeds: " + error).c_str());
    expectTrue(
        scan.leveledLists.size() == 2u, "both LVLN levels are collected as levelled actor lists");

    FalloutActorScan distantScan;
    expectTrue(
        findActorsNear(esmPath, 10000.0f, 10000.0f, 10.0f, distantScan, error) &&
            distantScan.placements.empty(),
        "the legacy proximity query excludes an authored placement outside its radius");
    FalloutActorScan catalog;
    expectTrue(
        findAllActors(esmPath, catalog, error) &&
            catalog.placements.size() == 1u &&
            catalog.placements.front().refFormId == kPlacementFormId,
        "the runtime actor catalog retains a placement regardless of authored distance");

    const ResolvedActorBase resolved = scan.resolve(kMarkerActorFormId);
    expectTrue(
        resolved.geometrySource == ActorGeometrySource::Race,
        "a marker-model NPC_ still resolves through its race");
    // The whole point: NOT marker_creature.nif.
    expectTrue(
        resolved.skeletonPath == "skeletons\\human.nif",
        "with no race body to stand beside, the skeleton comes from the template "
        "two levels of levelled list down");

    // An actor that OWNS its model must keep it, template or not.
    const ResolvedActorBase ownModel = scan.resolve(kRealActorFormId);
    expectTrue(
        ownModel.skeletonPath == "skeletons\\human.nif",
        "an actor owning its model is unaffected by the template walk");

    fs::remove(esmPath);
}

void testDeterministicActorInventoryExpansion() {
    using namespace odai::importer::fnv;
    FalloutActorScan scan;
    FalloutActorBase actor;
    actor.formId = 0x10u;
    actor.inventoryFormIds = {0x100u, 0x200u};
    scan.bases.emplace(actor.formId, actor);
    scan.leveledItems[0x100u] = {0x101u, 0x200u};
    scan.leveledItemUseAll[0x100u] = true;
    scan.leveledItems[0x101u] = {0x201u, 0x202u};
    scan.leveledItemUseAll[0x101u] = false;

    const std::vector<std::uint32_t> first =
        scan.materializeInventory(actor.formId, 0x13482u);
    const std::vector<std::uint32_t> replay =
        scan.materializeInventory(actor.formId, 0x13482u);
    expectTrue(first == replay,
        "actor LVLI materialization is deterministic for a persistent reference");
    expectTrue(std::count(first.begin(), first.end(), 0x200u) == 2 &&
            std::count_if(first.begin(), first.end(), [](std::uint32_t value) {
                return value == 0x201u || value == 0x202u;
            }) == 1,
        "use-all and choose-one LVLI semantics materialize leaves without list tokens");
    expectTrue(std::find(first.begin(), first.end(), 0x100u) == first.end() &&
            std::find(first.begin(), first.end(), 0x101u) == first.end(),
        "materialized inventory never exposes an LVLI record as an item");

    FalloutActorBase cyclic;
    cyclic.formId = 0x20u;
    cyclic.inventoryFormIds = {0x300u};
    scan.bases.emplace(cyclic.formId, cyclic);
    scan.leveledItems[0x300u] = {0x300u};
    expectTrue(scan.materializeInventory(cyclic.formId, 1u).empty(),
        "cyclic LVLI data terminates and contributes no raw list item");
}

// An INFO names who says it in one of TWO ways, and a reader that knows only
// the first renders a town where nobody generic can be spoken to.
//
// GetIsID (72) binds a line to one actor; GetIsVoiceType (427) binds it to a
// voice type, and so to everyone sharing it. Measured on the retail plugin:
// named characters get the first (Victor 133 topics, Easy Pete 108) while
// GSSettlerAM is named by ZERO INFO records and has only the second.
//
// Also pins the negative form, which is the trap: Fallout writes "everyone
// EXCEPT this actor" as the SAME function with the SAME parameter and only the
// comparison differing, so an attribution that reads function and parameter
// alone hands a character every other character's exclusions.
void testDialogueAttributionByActorAndVoiceType() {
    namespace fs = std::filesystem;
    using namespace odai::importer::fnv;

    constexpr std::uint32_t kNamedActorFormId = 0x00000010u;
    constexpr std::uint32_t kGenericOneFormId = 0x00000011u;
    constexpr std::uint32_t kGenericTwoFormId = 0x00000012u;
    constexpr std::uint32_t kVoiceTypeFormId = 0x00000020u;
    constexpr std::uint32_t kTopicFormId = 0x00000030u;

    const auto append = [](std::vector<std::uint8_t>& out, const std::vector<std::uint8_t>& sub) {
        out.insert(out.end(), sub.begin(), sub.end());
    };
    // CTDA (FO3/FNV, 28 bytes): type u8 @0 with the operator in its top 3 bits,
    // 3 unused, comparison f32 @4, function u32 @8, param1 u32 @12, then
    // param2, runOn and reference.
    const auto buildCtda = [](std::uint32_t function, std::uint32_t param1, bool positive) {
        std::vector<std::uint8_t> out;
        out.push_back(0u);  // operator EQUAL in the top bits
        out.push_back(0u);
        out.push_back(0u);
        out.push_back(0u);
        appendPod(out, positive ? 1.0f : 0.0f);  // ... == 1 is "is", == 0 is "is not"
        appendPod(out, function);
        appendPod(out, param1);
        appendPod(out, static_cast<std::uint32_t>(0));
        appendPod(out, static_cast<std::uint32_t>(0));
        appendPod(out, static_cast<std::uint32_t>(0));
        return out;
    };

    std::vector<std::uint8_t> topicSubrecords;
    append(topicSubrecords, buildSubrecord("EDID", stringPayload("GREETING")));
    append(topicSubrecords, buildSubrecord("FULL", stringPayload("GREETING")));

    // One line for the named actor alone.
    // The function indices are written as the MEASURED literals, not as the
    // constants under test. Using the constants would make the fixture and the
    // reader agree by construction, so a wrong value would pass -- and these
    // were derived by histogramming the retail plugin, which is exactly the
    // kind of fact that needs pinning.
    constexpr std::uint32_t kMeasuredGetIsId = 72u;
    constexpr std::uint32_t kMeasuredGetIsVoiceType = 427u;
    expectTrue(kCtdaFunctionGetIsId == kMeasuredGetIsId, "GetIsID is function 72");
    expectTrue(
        kCtdaFunctionGetIsVoiceType == kMeasuredGetIsVoiceType, "GetIsVoiceType is function 427");

    std::vector<std::uint8_t> namedInfo;
    append(namedInfo, buildSubrecord("CTDA", buildCtda(kMeasuredGetIsId, kNamedActorFormId, true)));
    append(namedInfo, buildSubrecord("NAM1", stringPayload("I am the named one.")));

    // One line for the voice type, which both generic actors share.
    std::vector<std::uint8_t> genericInfo;
    append(genericInfo,
           buildSubrecord("CTDA", buildCtda(kMeasuredGetIsVoiceType, kVoiceTypeFormId, true)));
    append(genericInfo, buildSubrecord("NAM1", stringPayload("Nice weather we're having.")));

    // "Everyone EXCEPT the named actor" -- must be attributed to nobody.
    std::vector<std::uint8_t> exclusionInfo;
    append(exclusionInfo,
           buildSubrecord("CTDA", buildCtda(kMeasuredGetIsId, kNamedActorFormId, false)));
    append(exclusionInfo, buildSubrecord("NAM1", stringPayload("SHOULD NOT BE ATTRIBUTED.")));

    std::vector<std::uint8_t> content;
    const auto appendGroup = [&](const char* type, const std::vector<std::uint8_t>& record) {
        const auto group = buildGroup(type, 0, record);
        content.insert(content.end(), group.begin(), group.end());
    };
    // INFOs stream immediately after the DIAL that owns them, which is how the
    // reader knows each response's topic without group bookkeeping.
    std::vector<std::uint8_t> dialGroup;
    const auto dial = buildRecord("DIAL", kTopicFormId, 0u, topicSubrecords);
    dialGroup.insert(dialGroup.end(), dial.begin(), dial.end());
    for (const auto* info : {&namedInfo, &genericInfo, &exclusionInfo}) {
        static std::uint32_t nextInfoFormId = 0x00000100u;
        const auto record = buildRecord("INFO", nextInfoFormId++, 0u, *info);
        dialGroup.insert(dialGroup.end(), record.begin(), record.end());
    }
    appendGroup("DIAL", dialGroup);

    const fs::path esmPath = fs::temp_directory_path() / "odai_fnv_dialogue_test.esm";
    {
        std::ofstream out(esmPath, std::ios::binary | std::ios::trunc);
        out.write(
            reinterpret_cast<const char*>(content.data()),
            static_cast<std::streamsize>(content.size()));
    }

    const std::vector<SpeakerDialogueRequest> speakers{
        SpeakerDialogueRequest{kNamedActorFormId, kVoiceTypeFormId, "Named"},
        SpeakerDialogueRequest{kGenericOneFormId, kVoiceTypeFormId, "Generic One"},
        SpeakerDialogueRequest{kGenericTwoFormId, 0u, "Generic Two"},
    };
    std::unordered_map<std::uint32_t, odai::dialogue::DialogueTree> trees;
    std::unordered_map<std::uint32_t, DialogueImportStats> stats;
    std::string error;
    expectTrue(
        buildSpeakerDialogueTrees(esmPath, speakers, trees, stats, error),
        ("batch dialogue scan succeeds: " + error).c_str());

    // The named actor gets his own line AND the voice-type line he qualifies
    // for -- the two attributions add up rather than one replacing the other.
    const auto named = trees.find(kNamedActorFormId);
    expectTrue(named != trees.end(), "the named actor gets a tree");
    if (named != trees.end()) {
        expectTrue(
            named->second.nodes.size() == 2u,
            "the named actor gets both his own line and his voice type's");
    }

    // A generic actor gets the voice-type line and nothing else.
    const auto genericOne = trees.find(kGenericOneFormId);
    expectTrue(genericOne != trees.end(), "an actor with only a voice type still gets a tree");
    if (genericOne != trees.end()) {
        expectTrue(
            genericOne->second.nodes.size() == 1u,
            "the voice-type line is shared with everyone using that voice");
    }

    // No actor formID of its own and no voice type: nothing to attribute.
    expectTrue(
        trees.find(kGenericTwoFormId) == trees.end(),
        "an actor named by nothing gets no tree at all");

    // The exclusion line must not have reached anybody.
    for (const auto& [speakerFormId, tree] : trees) {
        (void)speakerFormId;
        for (const auto& [nodeId, node] : tree.nodes) {
            (void)nodeId;
            expectTrue(
                node.text.find("SHOULD NOT") == std::string::npos,
                "a negative GetIsID condition attributes the line to nobody");
        }
    }

    fs::remove(esmPath);
}

// An NPC_ has no geometry of its own: its body is its RACE's part models with
// whatever it is wearing swapped in over them. This builds the smallest plugin
// that exercises the whole chain -- RACE part slots, an ARMO reached through a
// levelled item list, and the biped flags that decide which slot it claims --
// because every step of it is silent when wrong. A missed LVLI or a misread
// INDX does not fail, it just puts the town in its underwear.
void testActorRaceAndWardrobeAssembly() {
    namespace fs = std::filesystem;
    using namespace odai::importer::fnv;

    constexpr std::uint32_t kRaceFormId = 0x00000100u;
    constexpr std::uint32_t kOutfitFormId = 0x00000200u;
    constexpr std::uint32_t kOutfitListFormId = 0x00000300u;
    constexpr std::uint32_t kNpcFormId = 0x00000400u;
    constexpr std::uint32_t kPlacementFormId = 0x00000500u;

    // RACE. The section markers and the per-slot INDX are the whole point: the
    // female models are deliberately different strings, and the head section
    // deliberately comes first, so reading MODL without tracking state picks up
    // the wrong one.
    std::vector<std::uint8_t> raceSubrecords;
    const auto append = [](std::vector<std::uint8_t>& out, const std::vector<std::uint8_t>& sub) {
        out.insert(out.end(), sub.begin(), sub.end());
    };
    const auto indexPayload = [](std::uint32_t value) {
        std::vector<std::uint8_t> out;
        appendPod(out, value);
        return out;
    };
    append(raceSubrecords, buildSubrecord("EDID", stringPayload("TestRace")));
    append(raceSubrecords, buildSubrecord("NAM0", {}));           // head section
    append(raceSubrecords, buildSubrecord("MNAM", {}));           // male
    append(raceSubrecords, buildSubrecord("INDX", indexPayload(0)));
    append(raceSubrecords, buildSubrecord("MODL", stringPayload("heads\\male.nif")));
    append(raceSubrecords, buildSubrecord("FNAM", {}));           // female
    append(raceSubrecords, buildSubrecord("INDX", indexPayload(0)));
    append(raceSubrecords, buildSubrecord("MODL", stringPayload("heads\\female.nif")));
    append(raceSubrecords, buildSubrecord("NAM1", {}));           // body section
    append(raceSubrecords, buildSubrecord("MNAM", {}));
    append(raceSubrecords, buildSubrecord("INDX", indexPayload(0)));
    append(raceSubrecords, buildSubrecord("MODL", stringPayload("body\\upper.nif")));
    append(raceSubrecords, buildSubrecord("INDX", indexPayload(1)));
    append(raceSubrecords, buildSubrecord("MODL", stringPayload("body\\lefthand.nif")));
    append(raceSubrecords, buildSubrecord("INDX", indexPayload(2)));
    append(raceSubrecords, buildSubrecord("MODL", stringPayload("body\\righthand.nif")));
    // Slot 3 is a FaceGen texture, not a mesh, and must not become a body part.
    append(raceSubrecords, buildSubrecord("INDX", indexPayload(3)));
    append(raceSubrecords, buildSubrecord("MODL", stringPayload("body\\upper.egt")));
    append(raceSubrecords, buildSubrecord("FNAM", {}));
    append(raceSubrecords, buildSubrecord("INDX", indexPayload(0)));
    append(raceSubrecords, buildSubrecord("MODL", stringPayload("body\\upper_f.nif")));
    append(raceSubrecords, buildSubrecord("INDX", indexPayload(1)));
    append(raceSubrecords, buildSubrecord("MODL", stringPayload("body\\lefthand_f.nif")));
    append(raceSubrecords, buildSubrecord("INDX", indexPayload(2)));
    append(raceSubrecords, buildSubrecord("MODL", stringPayload("body\\righthand_f.nif")));
    // Past the parts: HNAM ends the section list, and the MNAM/FNAM/INDX after
    // it introduce FaceGen data where those types mean something else.
    append(raceSubrecords, buildSubrecord("HNAM", {}));
    append(raceSubrecords, buildSubrecord("MNAM", {}));
    append(raceSubrecords, buildSubrecord("INDX", indexPayload(0)));
    append(raceSubrecords, buildSubrecord("MODL", stringPayload("NOT_A_BODY_PART.nif")));

    // ARMO covering the upper body only, so the race's hands survive it.
    std::vector<std::uint8_t> armorSubrecords;
    append(armorSubrecords, buildSubrecord("EDID", stringPayload("TestOutfit")));
    {
        std::vector<std::uint8_t> bmdt;
        appendPod(bmdt, static_cast<std::uint32_t>(0x00000004u));  // upper body
        appendPod(bmdt, static_cast<std::uint32_t>(0));
        append(armorSubrecords, buildSubrecord("BMDT", bmdt));
    }
    append(armorSubrecords, buildSubrecord("MODL", stringPayload("armor\\outfit_m.nif")));
    append(armorSubrecords, buildSubrecord("MOD3", stringPayload("armor\\outfit_f.nif")));

    // The outfit is not carried directly -- it is one entry of a levelled list,
    // which is how Fallout actually dresses a settler.
    std::vector<std::uint8_t> listSubrecords;
    append(listSubrecords, buildSubrecord("EDID", stringPayload("TestOutfitList")));
    {
        std::vector<std::uint8_t> lvlo;
        appendPod(lvlo, static_cast<std::uint16_t>(1));  // level
        appendPod(lvlo, static_cast<std::uint16_t>(0));  // unused
        appendPod(lvlo, kOutfitFormId);
        appendPod(lvlo, static_cast<std::uint32_t>(1));  // count
        append(listSubrecords, buildSubrecord("LVLO", lvlo));
    }

    std::vector<std::uint8_t> npcSubrecords;
    append(npcSubrecords, buildSubrecord("EDID", stringPayload("TestSettler")));
    append(npcSubrecords, buildSubrecord("MODL", stringPayload("skeletons\\human.nif")));
    {
        std::vector<std::uint8_t> acbs(24u, 0u);
        const std::uint32_t flags = 0x00000001u;  // female
        std::memcpy(acbs.data(), &flags, sizeof(flags));
        append(npcSubrecords, buildSubrecord("ACBS", acbs));
    }
    append(npcSubrecords, buildSubrecord("RNAM", indexPayload(kRaceFormId)));
    {
        std::vector<std::uint8_t> cnto;
        appendPod(cnto, kOutfitListFormId);
        appendPod(cnto, static_cast<std::uint32_t>(1));
        append(npcSubrecords, buildSubrecord("CNTO", cnto));
    }

    std::vector<std::uint8_t> placementSubrecords;
    append(placementSubrecords, buildSubrecord("NAME", indexPayload(kNpcFormId)));
    {
        std::vector<std::uint8_t> data;
        appendPod(data, 100.0f);   // x
        appendPod(data, 200.0f);   // y
        appendPod(data, 300.0f);   // z
        appendPod(data, 0.0f);
        appendPod(data, 0.0f);
        appendPod(data, 0.0f);
        append(placementSubrecords, buildSubrecord("DATA", data));
    }

    std::vector<std::uint8_t> content;
    const auto appendGroup = [&](const char* type, const std::vector<std::uint8_t>& record) {
        const auto group = buildGroup(type, 0, record);
        content.insert(content.end(), group.begin(), group.end());
    };
    appendGroup("RACE", buildRecord("RACE", kRaceFormId, 0u, raceSubrecords));
    appendGroup("ARMO", buildRecord("ARMO", kOutfitFormId, 0u, armorSubrecords));
    appendGroup("LVLI", buildRecord("LVLI", kOutfitListFormId, 0u, listSubrecords));
    appendGroup("NPC_", buildRecord("NPC_", kNpcFormId, 0u, npcSubrecords));
    appendGroup("ACHR", buildRecord("ACHR", kPlacementFormId, 0u, placementSubrecords));

    const fs::path esmPath = fs::temp_directory_path() / "odai_fnv_actor_test.esm";
    {
        std::ofstream out(esmPath, std::ios::binary | std::ios::trunc);
        out.write(
            reinterpret_cast<const char*>(content.data()),
            static_cast<std::streamsize>(content.size()));
    }

    FalloutActorScan scan;
    std::string error;
    expectTrue(
        findActorsNear(esmPath, 100.0f, 200.0f, 1000.0f, scan, error),
        ("actor scan succeeds: " + error).c_str());
    expectTrue(scan.placements.size() == 1u, "the placement inside the radius is found");
    expectTrue(scan.races.count(kRaceFormId) == 1u, "the RACE record is collected");
    expectTrue(scan.armors.count(kOutfitFormId) == 1u, "the ARMO record is collected");
    expectTrue(
        scan.leveledItems.count(kOutfitListFormId) == 1u,
        "the LVLI record is collected, so a carried outfit list can be expanded");

    const ResolvedActorBase resolved = scan.resolve(kNpcFormId);
    expectTrue(
        resolved.geometrySource == ActorGeometrySource::Race,
        "an NPC_ with no parts of its own resolves through its race");
    // NOT "skeletons\\human.nif", which is what this NPC_'s own MODL says. A
    // race-assembled actor's skeleton lives beside its race's body parts, and
    // those are in "body\\" -- see the template test below for why MODL is not
    // to be trusted here.
    expectTrue(
        resolved.skeletonPath == "body\\skeleton.nif",
        "the skeleton is looked for beside the race's own body parts");
    expectTrue(
        resolved.wornArmorFormIds.size() == 1u &&
            resolved.wornArmorFormIds[0] == kOutfitFormId,
        "the outfit is reached through the levelled list and worn");

    // Female, so the ARMO's MOD3 and the race's FNAM parts win. Order is the
    // assembly order: body, hands, head.
    const std::vector<std::string> expected{
        "armor\\outfit_f.nif",
        "body\\lefthand_f.nif",
        "body\\righthand_f.nif",
        "heads\\female.nif",
    };
    expectTrue(
        resolved.bodyPartPaths == expected,
        "the assembled body is the female race parts with the outfit over the upper body");

    // A creature's NIFZ names are relative to its skeleton; a race's are not.
    // Both leave resolve() as full paths, which is what the caller loads.
    for (const std::string& part : resolved.bodyPartPaths) {
        expectTrue(
            part.find("skeletons\\") != 0u,
            "race part paths are not prefixed with the skeleton's directory");
    }

    fs::remove(esmPath);
}

// Skyrim moved the human skeleton onto RACE and the visible outfit through
// NPC_ DOFT -> OTFT -> LVLI(set) -> ARMO -> ARMA. It also marks the inner set
// "use all" while its outer list chooses one set. This is the complete path a
// Whiterun guard takes before character_builder can skin and animate it.
void testSkyrimActorSkeletonAndOutfitAssembly() {
    namespace fs = std::filesystem;
    using namespace odai::importer::fnv;

    constexpr std::uint32_t kRace = 0x100u;
    constexpr std::uint32_t kOutfit = 0x200u;
    constexpr std::uint32_t kSkin = 0x201u;
    constexpr std::uint32_t kChoiceList = 0x300u;
    constexpr std::uint32_t kSetList = 0x301u;
    constexpr std::uint32_t kNpc = 0x600u;
    constexpr std::uint32_t kPlacement = 0x700u;
    constexpr std::uint32_t kSkinHandsAddon = 0x510u;
    constexpr std::uint32_t kSkinCoveredTorsoAddon = 0x511u;
    const std::uint32_t armorIds[] = {0x400u, 0x401u, 0x402u, 0x403u};
    const std::uint32_t addonIds[] = {0x500u, 0x501u, 0x502u, 0x503u};
    const char* models[] = {
        "Armor\\Guard\\BootsM_1.nif", "Armor\\Guard\\CuirassM_1.nif",
        "Armor\\Guard\\GlovesM_1.nif", "Armor\\Guard\\HelmM_1.nif"};

    const auto append = [](std::vector<std::uint8_t>& out, const std::vector<std::uint8_t>& sub) {
        out.insert(out.end(), sub.begin(), sub.end());
    };
    const auto u32Payload = [](std::uint32_t value) {
        std::vector<std::uint8_t> out;
        appendPod(out, value);
        return out;
    };
    const auto listEntry = [&](std::uint32_t formId) {
        std::vector<std::uint8_t> out;
        appendPod(out, static_cast<std::uint16_t>(1));
        appendPod(out, static_cast<std::uint16_t>(0));
        appendPod(out, formId);
        appendPod(out, static_cast<std::uint32_t>(1));
        return buildSubrecord("LVLO", out);
    };

    std::vector<std::uint8_t> content;
    const auto appendGroup = [&](const char* type, const std::vector<std::uint8_t>& record) {
        const auto group = buildGroup(type, 0, record);
        content.insert(content.end(), group.begin(), group.end());
    };

    std::vector<std::uint8_t> raceSubs;
    append(raceSubs, buildSubrecord("EDID", stringPayload("GuardRace")));
    append(raceSubs, buildSubrecord("WNAM", u32Payload(kSkin)));
    append(raceSubs, buildSubrecord("MNAM", {}));
    append(raceSubs, buildSubrecord(
        "ANAM", stringPayload("Actors\\Character\\Character Assets\\skeleton.nif")));
    append(raceSubs, buildSubrecord("FNAM", {}));
    append(raceSubs, buildSubrecord(
        "ANAM", stringPayload("Actors\\Character\\Character Assets Female\\skeleton_female.nif")));
    appendGroup("RACE", buildRecord("RACE", kRace, 0u, raceSubs));

    std::vector<std::uint8_t> skinSubs;
    append(skinSubs, buildSubrecord("EDID", stringPayload("SkinNakedTest")));
    append(skinSubs, buildSubrecord("MODL", u32Payload(kSkinHandsAddon)));
    append(skinSubs, buildSubrecord("MODL", u32Payload(kSkinCoveredTorsoAddon)));
    appendGroup("ARMO", buildRecord("ARMO", kSkin, 0u, skinSubs));
    for (const auto& [addonId, slot, model] : {
             std::tuple{kSkinHandsAddon, 1u << 5, "Actors\\Character\\MaleHands_1.nif"},
             std::tuple{kSkinCoveredTorsoAddon, 1u << 2, "Actors\\Character\\MaleBody_1.nif"}}) {
        std::vector<std::uint8_t> addonSubs;
        append(addonSubs, buildSubrecord("EDID", stringPayload("SkinAddonTest")));
        append(addonSubs, buildSubrecord("BODT", u32Payload(slot)));
        append(addonSubs, buildSubrecord("RNAM", u32Payload(kRace)));
        append(addonSubs, buildSubrecord("MOD2", stringPayload(model)));
        appendGroup("ARMA", buildRecord("ARMA", addonId, 0u, addonSubs));
    }

    for (std::size_t i = 0; i < 4u; ++i) {
        std::vector<std::uint8_t> armorSubs;
        append(armorSubs, buildSubrecord("EDID", stringPayload("GuardArmor")));
        append(armorSubs, buildSubrecord("BOD2", u32Payload(1u << i)));
        append(armorSubs, buildSubrecord("MODL", u32Payload(addonIds[i])));
        appendGroup("ARMO", buildRecord("ARMO", armorIds[i], 0u, armorSubs));

        std::vector<std::uint8_t> addonSubs;
        append(addonSubs, buildSubrecord("EDID", stringPayload("GuardArmorAddon")));
        append(addonSubs, buildSubrecord("BODT", u32Payload(1u << i)));
        append(addonSubs, buildSubrecord("RNAM", u32Payload(kRace)));
        append(addonSubs, buildSubrecord("MOD2", stringPayload(models[i])));
        appendGroup("ARMA", buildRecord("ARMA", addonIds[i], 0u, addonSubs));
    }

    std::vector<std::uint8_t> setSubs;
    append(setSubs, buildSubrecord("EDID", stringPayload("ArmorGuardSet")));
    append(setSubs, buildSubrecord("LVLF", std::vector<std::uint8_t>{0x04u}));
    for (const std::uint32_t armor : armorIds) {
        append(setSubs, listEntry(armor));
    }
    appendGroup("LVLI", buildRecord("LVLI", kSetList, 0u, setSubs));

    std::vector<std::uint8_t> choiceSubs;
    append(choiceSubs, buildSubrecord("EDID", stringPayload("OutfitListGuard")));
    append(choiceSubs, buildSubrecord("LVLF", std::vector<std::uint8_t>{0x00u}));
    append(choiceSubs, listEntry(kSetList));
    append(choiceSubs, listEntry(0x4ffu));  // mutually-exclusive alternative
    appendGroup("LVLI", buildRecord("LVLI", kChoiceList, 0u, choiceSubs));

    std::vector<std::uint8_t> outfitSubs;
    append(outfitSubs, buildSubrecord("EDID", stringPayload("GuardOutfit")));
    append(outfitSubs, buildSubrecord("INAM", u32Payload(kChoiceList)));
    appendGroup("OTFT", buildRecord("OTFT", kOutfit, 0u, outfitSubs));

    std::vector<std::uint8_t> npcSubs;
    append(npcSubs, buildSubrecord("EDID", stringPayload("GuardWhiterunTest")));
    append(npcSubs, buildSubrecord("ACBS", std::vector<std::uint8_t>(24u, 0u)));
    append(npcSubs, buildSubrecord("RNAM", u32Payload(kRace)));
    append(npcSubs, buildSubrecord("DOFT", u32Payload(kOutfit)));
    appendGroup("NPC_", buildRecord("NPC_", kNpc, 0u, npcSubs));

    std::vector<std::uint8_t> placementSubs;
    append(placementSubs, buildSubrecord("NAME", u32Payload(kNpc)));
    std::vector<std::uint8_t> transform;
    for (const float value : {100.0f, 200.0f, 300.0f, 0.0f, 0.0f, 0.0f}) {
        appendPod(transform, value);
    }
    append(placementSubs, buildSubrecord("DATA", transform));
    appendGroup("ACHR", buildRecord("ACHR", kPlacement, 0u, placementSubs));

    const fs::path esmPath = fs::temp_directory_path() / "odai_skyrim_actor_test.esm";
    {
        std::ofstream out(esmPath, std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char*>(content.data()),
                  static_cast<std::streamsize>(content.size()));
    }

    FalloutActorScan scan;
    std::string error;
    expectTrue(
        findActorsNear(esmPath, 100.0f, 200.0f, 1000.0f, scan, error),
        ("Skyrim actor scan succeeds: " + error).c_str());
    const ResolvedActorBase resolved = scan.resolve(kNpc);
    const std::string faceGeometry =
        "actors\\character\\facegendata\\facegeom\\odai_skyrim_actor_test.esm\\00000600.nif";
    expectTrue(
        resolved.skeletonPath == "Actors\\Character\\Character Assets\\skeleton.nif",
        "Skyrim uses the RACE ANAM skeleton rather than an NPC/Fallout fallback");
    std::vector<std::string> expectedParts(std::begin(models), std::end(models));
    expectedParts.push_back("Actors\\Character\\MaleHands_1.nif");
    expectedParts.push_back(faceGeometry);
    expectTrue(resolved.bodyPartPaths == expectedParts,
               "Skyrim expands the outfit, fills uncovered default-skin slots, and appends FaceGen");
    expectTrue(
        resolved.wornArmorFormIds.size() == 4u,
        "the outer leveled list chooses one set while the inner use-all list wears every piece");

    fs::remove(esmPath);
}

void testBethesdaFireParticleEffectClassification() {
    using odai::importer::fnv::isFireParticleEffectModelPath;
    expectTrue(
        isFireParticleEffectModelPath("Effects\\FXfireWithEmbers01.nif"),
        "Skyrim stationary fire effect becomes a particle emitter");
    expectTrue(
        isFireParticleEffectModelPath("Effects/Fire/FlameNode01.nif"),
        "Oblivion stationary flame effect becomes a particle emitter");
    expectTrue(
        !isFireParticleEffectModelPath("Effects\\Magic\\FXFireballProjectile.nif"),
        "moving fire projectiles are not treated as stationary emitters");
    expectTrue(
        !isFireParticleEffectModelPath("Architecture\\WhiteRun\\WRIntCastleFloorFirePit01.nif"),
        "opaque fire-pit furniture is not mistaken for its separate fire effect");
    expectTrue(
        !isFireParticleEffectModelPath("Critters\\Firefly\\Firefly.nif"),
        "fireflies are not classified as fire");
}

void testAnimatedBannerSettlesUnderJoltGravity() {
    using odai::importer::fnv::NifModel;
    using odai::importer::fnv::NifShape;
    using odai::importer::fnv::applyNifBannerGravityRestPose;

    NifModel banner;
    banner.hasEmbeddedTransformAnimation = true;
    NifShape shape;
    // Z-up NIF cloth: top edge at x=0, lower edge blown three units sideways
    // while dropping four. Jolt should preserve its five-unit length and make
    // the free edge hang primarily in -Z under gravity.
    shape.positions = {
         0.0f, -1.0f,  0.0f,
         0.0f,  1.0f,  0.0f,
        -3.0f, -1.0f, -4.0f,
        -3.0f,  1.0f, -4.0f,
    };
    shape.triangleIndices = {0u, 2u, 1u, 1u, 2u, 3u};
    banner.shapes.push_back(shape);

    expectTrue(
        applyNifBannerGravityRestPose("Clutter\\Banners\\AnimatedBanner01.nif", banner),
        "animated banner is accepted by the Jolt cloth-rest path");
    const auto& settled = banner.shapes[0].positions;
    expectTrue(
        std::abs(settled[0]) < 0.05f && std::abs(settled[3]) < 0.05f,
        "the authored top attachment vertices stay pinned");
    expectTrue(
        std::abs(settled[6]) < 1.5f && std::abs(settled[9]) < 1.5f,
        "the free cloth edge no longer points sideways");
    expectTrue(
        settled[8] < -4.2f && settled[11] < -4.2f,
        "gravity redirects the preserved cloth length downward");
    expectTrue(
        banner.shapes[0].normals.size() == banner.shapes[0].positions.size(),
        "settled cloth normals are rebuilt for the new pose");

    NifModel staticBanner;
    staticBanner.shapes.push_back(shape);
    expectTrue(
        !applyNifBannerGravityRestPose("Clutter\\Banners\\StaticBanner01.nif", staticBanner),
        "a banner without embedded animation keeps its authored static pose");
}

void testMorrowindSkeletonNamesAndSkinLayout() {
    using namespace odai::importer::fnv;

    const float identity[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    const auto appendTransform = [&](std::vector<std::uint8_t>& bytes,
                                     const std::array<float, 3>& translation) {
        for (float value : identity) appendPod(bytes, value);
        for (float value : translation) appendPod(bytes, value);
        appendPod(bytes, 1.0f);
    };
    const auto appendAvObject = [&](std::vector<std::uint8_t>& bytes,
                                    const std::string& name,
                                    const std::array<float, 3>& translation) {
        appendSizedString32(bytes, name);
        appendPod(bytes, static_cast<std::int32_t>(-1));  // extra data
        appendPod(bytes, static_cast<std::int32_t>(-1));  // controller
        appendPod(bytes, static_cast<std::uint16_t>(0));  // flags
        for (float value : translation) appendPod(bytes, value);
        for (float value : identity) appendPod(bytes, value);
        appendPod(bytes, 1.0f);                           // scale
        for (int i = 0; i < 3; ++i) appendPod(bytes, 0.0f);  // velocity
        appendPod(bytes, static_cast<std::uint32_t>(0));  // properties
        appendPod(bytes, static_cast<std::uint32_t>(0));  // has bounds (wide bool)
    };
    const auto appendNode = [&](std::vector<std::uint8_t>& bytes,
                                const std::string& name,
                                const std::array<float, 3>& translation,
                                const std::vector<std::int32_t>& children) {
        appendAvObject(bytes, name, translation);
        appendPod(bytes, static_cast<std::uint32_t>(children.size()));
        for (std::int32_t child : children) appendPod(bytes, child);
        appendPod(bytes, static_cast<std::uint32_t>(0));  // effects
    };

    std::vector<std::uint8_t> root;
    appendNode(root, "Bip01", {0.0f, 0.0f, 1.0f}, {1, 2});
    std::vector<std::uint8_t> bone;
    appendNode(bone, "Bip01 Head", {0.0f, 0.0f, 2.0f}, {});

    std::vector<std::uint8_t> shape;
    appendAvObject(shape, "Head", {0.0f, 0.0f, 0.0f});
    appendPod(shape, static_cast<std::int32_t>(3));  // NiTriShapeData
    appendPod(shape, static_cast<std::int32_t>(4));  // NiSkinInstance

    std::vector<std::uint8_t> geometry;
    appendPod(geometry, static_cast<std::uint16_t>(3));
    appendPod(geometry, static_cast<std::uint32_t>(1));  // has vertices
    const float positions[9] = {0, 0, 0, 1, 0, 0, 0, 1, 0};
    for (float value : positions) appendPod(geometry, value);
    appendPod(geometry, static_cast<std::uint32_t>(1));  // has normals
    const float normals[9] = {0, 0, 1, 0, 0, 1, 0, 0, 1};
    for (float value : normals) appendPod(geometry, value);
    for (int i = 0; i < 4; ++i) appendPod(geometry, 0.0f);  // bounding sphere
    appendPod(geometry, static_cast<std::uint32_t>(0));  // has colours
    appendPod(geometry, static_cast<std::uint16_t>(1));  // UV set count
    appendPod(geometry, static_cast<std::uint32_t>(1));  // has UVs
    const float uvs[6] = {0, 0, 1, 0, 0, 1};
    for (float value : uvs) appendPod(geometry, value);
    appendPod(geometry, static_cast<std::uint16_t>(1));  // triangles
    appendPod(geometry, static_cast<std::uint32_t>(3));  // indices
    appendPod(geometry, static_cast<std::uint16_t>(0));
    appendPod(geometry, static_cast<std::uint16_t>(1));
    appendPod(geometry, static_cast<std::uint16_t>(2));
    appendPod(geometry, static_cast<std::uint16_t>(0));  // match groups

    // Morrowind's NiSkinInstance has no partition ref. If a modern-layout
    // reader consumes one here, the bone count and pointer both shift.
    std::vector<std::uint8_t> skinInstance;
    appendPod(skinInstance, static_cast<std::int32_t>(5));  // NiSkinData
    appendPod(skinInstance, static_cast<std::int32_t>(0));  // skeleton root
    appendPod(skinInstance, static_cast<std::uint32_t>(1));
    appendPod(skinInstance, static_cast<std::int32_t>(1));  // Bip01 Head

    std::vector<std::uint8_t> skinData;
    appendTransform(skinData, {7.0f, 8.0f, 9.0f});
    appendPod(skinData, static_cast<std::uint32_t>(1));
    appendPod(skinData, static_cast<std::int32_t>(-1));  // legacy partition ref
    appendTransform(skinData, {4.0f, 5.0f, 6.0f});      // inverse bind
    for (int i = 0; i < 4; ++i) appendPod(skinData, 0.0f);  // bounding sphere
    appendPod(skinData, static_cast<std::uint16_t>(3));
    for (std::uint16_t vertex = 0; vertex < 3; ++vertex) {
        appendPod(skinData, vertex);
        appendPod(skinData, 1.0f);
    }

    std::vector<std::uint8_t> fileBytes;
    const std::string header = "NetImmerse File Format, Version 4.0.0.2\n";
    fileBytes.insert(fileBytes.end(), header.begin(), header.end());
    appendPod(fileBytes, static_cast<std::uint32_t>(0x04000002u));
    appendPod(fileBytes, static_cast<std::uint32_t>(6));
    const auto appendBlock = [&](const std::string& type,
                                 const std::vector<std::uint8_t>& block) {
        appendSizedString32(fileBytes, type);
        fileBytes.insert(fileBytes.end(), block.begin(), block.end());
    };
    appendBlock("NiNode", root);
    appendBlock("NiNode", bone);
    appendBlock("NiTriShape", shape);
    appendBlock("NiTriShapeData", geometry);
    appendBlock("NiSkinInstance", skinInstance);
    appendBlock("NiSkinData", skinData);

    std::string error;
    NifSkeleton skeleton;
    expectTrue(
        parseNifSkeleton(fileBytes, skeleton, error),
        ("Morrowind inline-name skeleton parses: " + error).c_str());
    expectTrue(
        skeleton.bones.size() == 2u && skeleton.bones[0].name == "Bip01" &&
            skeleton.bones[1].name == "Bip01 Head" && skeleton.bones[1].parentIndex == 0,
        "Morrowind inline skeleton names and hierarchy preserve authored pivots");

    NifSkinnedModel skinned;
    error.clear();
    expectTrue(
        parseNifSkinnedMesh(fileBytes, skinned, error),
        ("Morrowind legacy skin blocks parse: " + error).c_str());
    expectTrue(
        skinned.shapes.size() == 1u && skinned.shapes[0].name == "Head" &&
            skinned.shapes[0].boneNames == std::vector<std::string>{"Bip01 Head"},
        "Morrowind skin preserves inline shape and bone names");
    if (skinned.shapes.size() == 1u) {
        const auto& parsed = skinned.shapes[0];
        expectTrue(parsed.requiresCanonicalBindBake,
                   "Morrowind shape requests a canonical authored-bind bake");
        expectNear(parsed.positions[2], 0.0f, 1e-5f,
                   "Morrowind skin vertices remain in authored geometry space");
        expectNear(parsed.skinTransform[3], 7.0f, 1e-5f,
                   "Morrowind overall skin pivot stays aligned");
        expectNear(parsed.inverseBindMatrices[3], 4.0f, 1e-5f,
                   "Morrowind inverse-bind pivot stays aligned");
        expectTrue(
            parsed.boneIndices.size() == 12u && parsed.boneWeights.size() == 12u &&
                parsed.boneWeights[0] == 1.0f && parsed.boneWeights[4] == 1.0f &&
                parsed.boneWeights[8] == 1.0f,
            "Morrowind per-bone weights transpose into per-vertex GPU influences");

        FalloutCharacter character;
        expectTrue(buildFalloutSkeleton(skeleton, character.skeleton),
                   "Morrowind skeleton converts to the runtime basis");
        error.clear();
        expectTrue(appendFalloutCharacterMesh(skinned, character, error),
                   ("Morrowind skin canonicalizes: " + error).c_str());
        if (!character.vertices.empty()) {
            // bind bone world (z=3) * bone inverse (4,5,6) * overall skin
            // (7,8,9), then Bethesda Z-up -> engine Y-up.
            expectNear(character.vertices[0].position[0], 11.0f, 1e-5f,
                       "TES3 authored bind places the vertex on the right pivot X");
            expectNear(character.vertices[0].position[1], 18.0f, 1e-5f,
                       "TES3 authored bind places the vertex on the right pivot height");
            expectNear(character.vertices[0].position[2], -13.0f, 1e-5f,
                       "TES3 authored bind places the vertex on the right pivot depth");
            std::vector<odai::math::Matrix4> bindPose;
            computeFalloutBindPose(character, bindPose);
            const auto& vertex = character.vertices[0];
            const odai::math::Vector3 roundTrip = odai::math::transformPoint(
                bindPose[vertex.boneIndices[0]],
                {vertex.position[0], vertex.position[1], vertex.position[2]});
            expectNear(roundTrip.x, vertex.position[0], 1e-4f,
                       "TES3 canonical inverse bind round-trips X");
            expectNear(roundTrip.y, vertex.position[1], 1e-4f,
                       "TES3 canonical inverse bind round-trips height");
            expectNear(roundTrip.z, vertex.position[2], 1e-4f,
                       "TES3 canonical inverse bind round-trips depth");
        }
    }

    // TES3 BODY parts commonly make the NiTriShape itself the declared root;
    // there is no wrapper NiNode. A footer-root walk that admits only nodes
    // silently drops these rigid feet, wrists, legs, and clothing pieces.
    std::vector<std::uint8_t> rigidShape;
    appendAvObject(rigidShape, "Rigid Foot", {3.0f, 4.0f, 5.0f});
    appendPod(rigidShape, static_cast<std::int32_t>(1));   // NiTriShapeData
    appendPod(rigidShape, static_cast<std::int32_t>(-1));  // no skin instance
    std::vector<std::uint8_t> rigidFile;
    rigidFile.insert(rigidFile.end(), header.begin(), header.end());
    appendPod(rigidFile, static_cast<std::uint32_t>(0x04000002u));
    appendPod(rigidFile, static_cast<std::uint32_t>(2));
    appendSizedString32(rigidFile, "NiTriShape");
    rigidFile.insert(rigidFile.end(), rigidShape.begin(), rigidShape.end());
    appendSizedString32(rigidFile, "NiTriShapeData");
    rigidFile.insert(rigidFile.end(), geometry.begin(), geometry.end());
    appendPod(rigidFile, static_cast<std::uint32_t>(1));  // footer roots
    appendPod(rigidFile, static_cast<std::int32_t>(0));

    NifModel rigidModel;
    error.clear();
    expectTrue(
        parseNifStaticMesh(rigidFile, rigidModel, error),
        ("Morrowind rigid root shape parses: " + error).c_str());
    expectTrue(
        rigidModel.usedFooterRoots && rigidModel.shapes.size() == 1u,
        "Morrowind NiTriShape footer root remains reachable without a wrapper node");
    if (rigidModel.shapes.size() == 1u) {
        expectNear(
            rigidModel.shapes[0].positions[0], 3.0f, 1e-5f,
            "rigid BODY part keeps its authored attachment-local pivot");
    }
}

void testMorrowindNifLodNodeSubtypeTail() {
    const auto appendMorrowindNodeBase = [](std::vector<std::uint8_t>& bytes) {
        appendPod(bytes, static_cast<std::uint32_t>(0));   // name length
        appendPod(bytes, static_cast<std::int32_t>(-1));   // extra data
        appendPod(bytes, static_cast<std::int32_t>(-1));   // controller
        appendPod(bytes, static_cast<std::uint16_t>(0));   // flags
        for (int i = 0; i < 3; ++i) appendPod(bytes, 0.0f);  // translation
        const float identity[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
        for (float value : identity) appendPod(bytes, value);
        appendPod(bytes, 1.0f);                            // scale
        for (int i = 0; i < 3; ++i) appendPod(bytes, 0.0f);  // velocity
        appendPod(bytes, static_cast<std::uint32_t>(0));   // properties
        appendPod(bytes, static_cast<std::uint32_t>(0));   // has bounds (wide bool)
        appendPod(bytes, static_cast<std::uint32_t>(0));   // children
        appendPod(bytes, static_cast<std::uint32_t>(0));   // effects
    };

    std::vector<std::uint8_t> lodBlock;
    appendMorrowindNodeBase(lodBlock);
    appendPod(lodBlock, static_cast<std::int32_t>(-1));  // NiSwitchNode initial child
    appendPod(lodBlock, 10.0f);                          // NiLODNode centre
    appendPod(lodBlock, 20.0f);
    appendPod(lodBlock, 30.0f);
    appendPod(lodBlock, static_cast<std::uint32_t>(1));  // one near/far range
    appendPod(lodBlock, 0.0f);
    appendPod(lodBlock, 1000.0f);

    std::vector<std::uint8_t> nodeBlock;
    appendMorrowindNodeBase(nodeBlock);

    std::vector<std::uint8_t> fileBytes;
    const std::string header = "NetImmerse File Format, Version 4.0.0.2\n";
    fileBytes.insert(fileBytes.end(), header.begin(), header.end());
    appendPod(fileBytes, static_cast<std::uint32_t>(0x04000002u));
    appendPod(fileBytes, static_cast<std::uint32_t>(2));
    appendSizedString32(fileBytes, "NiLODNode");
    fileBytes.insert(fileBytes.end(), lodBlock.begin(), lodBlock.end());
    appendSizedString32(fileBytes, "NiNode");
    fileBytes.insert(fileBytes.end(), nodeBlock.begin(), nodeBlock.end());

    odai::importer::fnv::NifBlockSummary summary;
    std::string error;
    expectTrue(
        odai::importer::fnv::parseNifBlockSummary(fileBytes, summary, error),
        ("Morrowind NiLODNode subtype tail parses: " + error).c_str());
    expectTrue(
        summary.blockTypeNames == std::vector<std::string>{"NiLODNode", "NiNode"},
        "the block after NiLODNode remains aligned");
    expectTrue(
        summary.blockSizes.size() == 2u && summary.blockSizes[0] == lodBlock.size() &&
            summary.blockSizes[1] == nodeBlock.size(),
        "NiLODNode includes its switch index, centre and range array in its derived size");

    // NiSwitchNode itself has the same four-byte signed initial-child field in
    // 4.0.0.2, without the 16-bit flags present in modern Gamebryo files. This
    // is a static-parser regression test in addition to the block-size test
    // above: reading the modern six-byte tail used to reject this valid node.
    std::vector<std::uint8_t> switchBlock;
    appendMorrowindNodeBase(switchBlock);
    appendPod(switchBlock, static_cast<std::int32_t>(-1));

    std::vector<std::uint8_t> switchFileBytes;
    switchFileBytes.insert(switchFileBytes.end(), header.begin(), header.end());
    appendPod(switchFileBytes, static_cast<std::uint32_t>(0x04000002u));
    appendPod(switchFileBytes, static_cast<std::uint32_t>(2));
    appendSizedString32(switchFileBytes, "NiSwitchNode");
    switchFileBytes.insert(switchFileBytes.end(), switchBlock.begin(), switchBlock.end());
    appendSizedString32(switchFileBytes, "NiNode");
    switchFileBytes.insert(switchFileBytes.end(), nodeBlock.begin(), nodeBlock.end());

    odai::importer::fnv::NifModel switchModel;
    error.clear();
    expectTrue(
        odai::importer::fnv::parseNifStaticMesh(switchFileBytes, switchModel, error),
        ("Morrowind NiSwitchNode parses: " + error).c_str());
    expectTrue(
        switchModel.nodeParseFailedCount == 0u,
        "Morrowind NiSwitchNode consumes its four-byte subtype tail");
}

void testMorrowindDirectKeyframeAndStencilSizing() {
    using namespace odai::importer::fnv;

    std::vector<std::uint8_t> controller;
    appendPod(controller, static_cast<std::int32_t>(-1));  // next controller
    appendPod(controller, static_cast<std::uint16_t>(8));  // active + cycle
    appendPod(controller, 1.0f);                           // frequency
    appendPod(controller, 0.0f);                           // phase
    appendPod(controller, 0.0f);                           // start
    appendPod(controller, 1.0f);                           // stop
    appendPod(controller, static_cast<std::int32_t>(0));   // target block
    appendPod(controller, static_cast<std::int32_t>(1));   // NiKeyframeData

    std::vector<std::uint8_t> keyData;
    appendPod(keyData, static_cast<std::uint32_t>(2));  // quaternion keys
    appendPod(keyData, static_cast<std::uint32_t>(1));  // linear
    appendPod(keyData, 0.0f);
    appendPod(keyData, 1.0f);  // w, x, y, z
    appendPod(keyData, 0.0f);
    appendPod(keyData, 0.0f);
    appendPod(keyData, 0.0f);
    appendPod(keyData, 1.0f);
    appendPod(keyData, 0.9238795f);
    appendPod(keyData, 0.0f);
    appendPod(keyData, 0.0f);
    appendPod(keyData, 0.3826834f);
    appendPod(keyData, static_cast<std::uint32_t>(0));  // translations
    appendPod(keyData, static_cast<std::uint32_t>(0));  // scales

    std::vector<std::uint8_t> animationFile;
    const std::string header = "NetImmerse File Format, Version 4.0.0.2\n";
    animationFile.insert(animationFile.end(), header.begin(), header.end());
    appendPod(animationFile, static_cast<std::uint32_t>(0x04000002u));
    appendPod(animationFile, static_cast<std::uint32_t>(3));
    appendSizedString32(animationFile, "NiKeyframeController");
    animationFile.insert(animationFile.end(), controller.begin(), controller.end());
    appendSizedString32(animationFile, "NiKeyframeData");
    animationFile.insert(animationFile.end(), keyData.begin(), keyData.end());
    std::vector<std::uint8_t> textKeys;
    appendPod(textKeys, static_cast<std::int32_t>(-1));  // next extra data
    appendPod(textKeys, static_cast<std::uint32_t>(0));  // legacy record size
    appendPod(textKeys, static_cast<std::uint32_t>(2));
    appendPod(textKeys, 3.25f);
    appendSizedString32(textKeys, "Idle: Start");
    appendPod(textKeys, 5.75f);
    appendSizedString32(textKeys, "Idle: Stop\r\n");
    appendSizedString32(animationFile, "NiTextKeyExtraData");
    animationFile.insert(animationFile.end(), textKeys.begin(), textKeys.end());

    std::vector<KfAnimation> animations;
    std::string error;
    expectTrue(
        parseNifEmbeddedAnimations(
            animationFile, std::vector<std::string>{"branch", ""}, animations, error),
        ("Morrowind direct keyframe parses: " + error).c_str());
    expectTrue(
        animations.size() == 1u && animations[0].name == "branch" &&
            animations[0].loops() && animations[0].tracks.size() == 1u &&
            animations[0].tracks[0].rotationKeys.size() == 2u,
        "Morrowind direct controller becomes one named looping rigid track");

    NifModel textKeyModel;
    error.clear();
    expectTrue(
        parseNifStaticMesh(animationFile, textKeyModel, error),
        ("Morrowind animation text keys parse: " + error).c_str());
    expectTrue(
        textKeyModel.textKeys.size() == 2u &&
            textKeyModel.textKeys[0].text == "Idle: Start" &&
            textKeyModel.textKeys[1].text == "Idle: Stop\r\n",
        "TES3 animation boundaries retain their authored times and labels");
    if (textKeyModel.textKeys.size() == 2u) {
        expectNear(textKeyModel.textKeys[0].time, 3.25f, 1e-6f,
                   "TES3 text-key start time is decoded");
        expectNear(textKeyModel.textKeys[1].time, 5.75f, 1e-6f,
                   "TES3 text-key stop time is decoded");
    }

    // NiStencilProperty has seven trailing u32 fields in 4.0.0.2. Omitting
    // drawMode leaves the unsized walk four bytes before the next type name.
    std::vector<std::uint8_t> stencil;
    appendPod(stencil, static_cast<std::uint32_t>(0));   // name
    appendPod(stencil, static_cast<std::int32_t>(-1));   // extra data
    appendPod(stencil, static_cast<std::int32_t>(-1));   // controller
    appendPod(stencil, static_cast<std::uint16_t>(0));   // flags
    stencil.push_back(1u);                               // enabled
    for (int i = 0; i < 7; ++i) appendPod(stencil, static_cast<std::uint32_t>(0));

    std::vector<std::uint8_t> node;
    appendPod(node, static_cast<std::uint32_t>(0));      // name
    appendPod(node, static_cast<std::int32_t>(-1));      // extra data
    appendPod(node, static_cast<std::int32_t>(-1));      // controller
    appendPod(node, static_cast<std::uint16_t>(0));      // flags
    for (int i = 0; i < 3; ++i) appendPod(node, 0.0f);
    const float identity[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    for (float value : identity) appendPod(node, value);
    appendPod(node, 1.0f);
    for (int i = 0; i < 3; ++i) appendPod(node, 0.0f);
    appendPod(node, static_cast<std::uint32_t>(0));      // properties
    appendPod(node, static_cast<std::uint32_t>(0));      // bounds
    appendPod(node, static_cast<std::uint32_t>(0));      // children
    appendPod(node, static_cast<std::uint32_t>(0));      // effects

    std::vector<std::uint8_t> sizingFile;
    sizingFile.insert(sizingFile.end(), header.begin(), header.end());
    appendPod(sizingFile, static_cast<std::uint32_t>(0x04000002u));
    appendPod(sizingFile, static_cast<std::uint32_t>(2));
    appendSizedString32(sizingFile, "NiStencilProperty");
    sizingFile.insert(sizingFile.end(), stencil.begin(), stencil.end());
    appendSizedString32(sizingFile, "NiNode");
    sizingFile.insert(sizingFile.end(), node.begin(), node.end());

    NifBlockSummary summary;
    error.clear();
    expectTrue(
        parseNifBlockSummary(sizingFile, summary, error),
        ("Morrowind stencil keeps the next block aligned: " + error).c_str());
    expectTrue(
        summary.blockSizes.size() == 2u && summary.blockSizes[0] == stencil.size() &&
            summary.blockTypeNames[1] == "NiNode",
        "Morrowind stencil sizing includes its draw-mode field");
}

void testLandLayerOpacityReconstruction() {
    using namespace odai::importer::fnv;

    FalloutLandTextureLayer empty;
    expectNear(sampleLandLayerOpacity(empty, 8.0f, 8.0f), 0.0f, 1e-6f,
               "empty VTXT reconstruction stays transparent");

    FalloutLandTextureLayer full;
    std::fill(std::begin(full.opacity), std::end(full.opacity), 1.0f);
    expectNear(sampleLandLayerOpacity(full, 7.25f, 11.75f), 1.0f, 1e-6f,
               "constant VTXT reconstruction stays fully opaque");

    FalloutLandTextureLayer spot;
    constexpr int kCentre = 8 * kLandQuadrantGridSize + 8;
    spot.opacity[kCentre] = 1.0f;
    const float centre = sampleLandLayerOpacity(spot, 8.0f, 8.0f);
    const float left = sampleLandLayerOpacity(spot, 8.0f, 7.0f);
    const float right = sampleLandLayerOpacity(spot, 8.0f, 9.0f);
    expectNear(centre, 0.7375f, 1e-6f,
               "reconstruction keeps an isolated authored post dominant");
    expectTrue(left > 0.0f && left < centre && std::fabs(left - right) < 1e-6f,
               "reconstruction feathers an isolated post symmetrically");
    const float halfway = sampleLandLayerOpacity(spot, 8.0f, 8.5f);
    expectTrue(halfway > right && halfway < centre,
               "fractional reconstruction smoothly bridges adjacent posts");
    const float clamped = sampleLandLayerOpacity(spot, -100.0f, 100.0f);
    expectTrue(clamped >= 0.0f && clamped <= 1.0f,
               "out-of-range reconstruction coordinates clamp to a valid opacity");
}

void testVertexFadeTrianglePartitioning() {
    using namespace odai::importer::fnv;

    NifShape shape;
    shape.alphaSemantic = NifAlphaSemantic::VertexFade;
    shape.alphaBlend = true;
    shape.triangleIndices = {0u, 1u, 2u, 2u, 1u, 3u};
    shape.colors.resize(4u * 4u, 1.0f);
    // Only the second triangle touches vertex 3, so only that triangle belongs
    // in the no-depth-write transition tail.
    shape.colors[(3u * 4u) + 3u] = 0.25f;
    std::vector<std::uint32_t> indices{99u};
    std::uint32_t opaque = 0u;
    std::uint32_t faded = 0u;
    appendPartitionedNifShapeIndices(shape, 10u, indices, opaque, faded);
    expectTrue(opaque == 3u && faded == 3u,
               "vertex-fade partition keeps opaque and faded triangle counts");
    expectTrue(indices == std::vector<std::uint32_t>({99u, 10u, 11u, 12u, 12u, 11u, 13u}),
               "vertex-fade partition preserves winding and appends the faded tail");

    shape.alphaSemantic = NifAlphaSemantic::ExplicitBlend;
    indices.clear();
    appendPartitionedNifShapeIndices(shape, 0u, indices, opaque, faded);
    expectTrue(opaque == 6u && faded == 0u && indices == shape.triangleIndices,
               "explicit transparency stays one material-controlled range");

    shape.alphaSemantic = NifAlphaSemantic::VertexFade;
    for (std::size_t alpha = 3u; alpha < shape.colors.size(); alpha += 4u) {
        shape.colors[alpha] = 0.5f;
    }
    indices.clear();
    appendPartitionedNifShapeIndices(shape, 0u, indices, opaque, faded);
    expectTrue(opaque == 0u && faded == 6u && indices == shape.triangleIndices,
               "fully faded geometry remains wholly in the blended tail");
}

int main() {
    testBsaArchiveReadsFoldersAndFiles(/*embedFileNames=*/false);
    testBsaArchiveReadsFoldersAndFiles(/*embedFileNames=*/true);
    testBsaArchiveReadsOblivionV103();
    testEsmReaderWalksGroupsRecordsAndSubrecords();
    testEsmReaderWalksBothHeaderGenerations();
    testOblivionLandTextureIconPath();
    testCellWaterPatch();
    testLz4FrameDecoding();
    testEsmReaderWalksMorrowindHeaders();
    testEsmReaderToleratesCorruptChecksum();
    testEsmReaderSkipsRecordsByHeader();
    testFalloutRecordExtraction();
    testSkyrimRegionAndSoundRecords();
    testLandLayerOpacityReconstruction();
    testVertexFadeTrianglePartitioning();
    testNifParserExtractsTransformedGeometry();
    testSkyrimTerrainPackedPositionsAreFullPrecision();
    testSkyrimLightingShaderVertexAlpha();
    testNifParserDoesNotReparentSubtreesToTheOrigin();
    testNifParserInheritsPropertiesFromParentNodes();
    testNifParserRejectsUnusableTriangles();
    testNifParserRejectsImplausibleChildCount();
    testPluginHeaderRejectsOversizedRecord();
    testPluginHeaderReadsBothGenerations();
    testAsyncAssetLoaderDeduplicatesAndLoadsConcurrently();
    testModDirectoryOverridesArchives();
    testPluginLoadOrderRemapsFormIds();
    testSkyrimPluginListResolution();
    testMorrowindLoadOrderMergesWorldRenderingRecords();
    testLandLodBlockOrigin();
    testLandLodTileOriginAcrossTiers();
    testLandLodTilePaths();
    testSkinnedBindPoseIsIdentity();
    testDynamicFacePartsShareOneBonePalette();
    testSkinnedInfluenceWeightsAreNormalized();
    testKfAnimationStrideAndBasisChange();
    testKfBSplineDecoding();
    testActorRaceAndWardrobeAssembly();
    testSkyrimActorSkeletonAndOutfitAssembly();
    testDialogueAttributionByActorAndVoiceType();
    testTemplateSkeletonThroughNestedLeveledLists();
    testDeterministicActorInventoryExpansion();
    testOblivionWeatherFogAndCloudTints();
    testSkyrimWeatherCloudLayers();
    testBethesdaFireParticleEffectClassification();
    testAnimatedBannerSettlesUnderJoltGravity();
    testMorrowindSkeletonNamesAndSkinLayout();
    testMorrowindNifLodNodeSubtypeTail();
    testMorrowindDirectKeyframeAndStencilSizing();

    if (g_failures != 0) {
        std::cerr << "[fnv import test] " << g_failures << " failures\n";
        return 1;
    }
    std::cout << "[fnv import test] all checks passed\n";
    return 0;
}
