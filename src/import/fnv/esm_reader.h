#pragma once

// Generic reader for Bethesda "Fallout"-style ESM/ESP plugin files (Fallout 3
// / Fallout: New Vegas record layout — a 24-byte record header, unlike
// Morrowind's 16-byte TES3 header or Oblivion/Skyrim's 20-byte TES4 header).
// This module only understands the generic GRUP/record/subrecord container
// format; it has no idea what a CELL or STAT record means. See
// fallout_records.h for typed extraction built on top of this walker.
//
// Format reference (all fields little-endian):
//   GRUP header (24 bytes): "GRUP", groupSize (includes this header), label
//     (4 bytes, meaning depends on groupType), groupType (int32), stamp
//     (uint16), unknown (uint16), versionControlInfo (uint32).
//   Record header (24 bytes): 4-char type, dataSize (excludes this header),
//     flags (bit 0x00040000 = compressed), formID, versionControlInfo,
//     formVersion (uint16), unknown (uint16).
//   Record data: a sequence of subrecords, each a 4-char type + uint16 size
//     + that many bytes of data. When compressed (flags bit set), the record
//     data is [uint32 decompressedSize][zlib deflate stream] instead.
//   Oversized subrecord: an "XXXX" subrecord holds a uint32 giving the real
//     size of the subrecord immediately following it (whose own 2-byte size
//     field is then ignored) — used when a subrecord exceeds 65535 bytes.

#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

namespace odai::importer::fnv {

struct EsmSubrecordView {
    std::string type;  // 4-char subrecord type, e.g. "EDID"
    const std::uint8_t* data = nullptr;
    std::uint32_t size = 0;
};

// Subrecord `data` pointers are valid only for the duration of the
// Visitor::onRecord callback that receives this record: uncompressed records
// point into the reader's whole-file buffer (stable for the reader's
// lifetime), but compressed records point into a decompression scratch
// buffer that walk() reuses for the next compressed record and frees when
// walk() returns. Copy out any bytes a caller needs to keep past the
// callback — never store an EsmRecordView/EsmSubrecordView itself.
struct EsmRecordView {
    std::string type;  // 4-char record type, e.g. "STAT", "CELL", "REFR"
    std::uint32_t formId = 0;
    std::uint32_t flags = 0;
    std::vector<EsmSubrecordView> subrecords;
};

// Raw GRUP metadata. For top-level groups (groupType == 0), rawLabel holds
// the 4-char record type the group contains (e.g. "STAT"). For nested
// per-cell/per-worldspace groups, rawLabel holds the parent formID's raw
// bytes (little-endian) instead — callers that need the formID should
// reinterpret rawLabel as a uint32.
struct EsmGroupView {
    std::string rawLabel;
    std::int32_t groupType = 0;
};

class EsmReader {
public:
    bool open(const std::filesystem::path& path);
    const std::string& lastError() const { return m_lastError; }

    struct Visitor {
        // Called on entering a GRUP. Return false to skip the group's
        // contents entirely (onGroupExit is still called).
        std::function<bool(const EsmGroupView&)> onGroupEnter;
        std::function<void(const EsmGroupView&)> onGroupExit;
        // See the EsmRecordView comment: extract what you need from `record`
        // synchronously — its subrecord data pointers do not outlive this call.
        std::function<void(const EsmRecordView&)> onRecord;
    };

    // Depth-first walk of the whole file.
    bool walk(const Visitor& visitor);

private:
    std::vector<std::uint8_t> m_bytes;
    std::string m_lastError;
};

}  // namespace odai::importer::fnv
