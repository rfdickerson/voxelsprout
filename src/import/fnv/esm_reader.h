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
#include <string_view>
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

// The cheap half of a record: everything readable from its 24-byte header,
// with no decompression and no subrecord parsing. `type` points into the
// reader's own file buffer and is valid only for the duration of the
// onRecordHeader callback.
struct EsmRecordHeaderView {
    std::string_view type;
    std::uint32_t formId = 0;
    std::uint32_t flags = 0;
};

// Move-only: an open reader owns a memory mapping of the plugin file.
class EsmReader {
public:
    EsmReader() = default;
    ~EsmReader();
    EsmReader(const EsmReader&) = delete;
    EsmReader& operator=(const EsmReader&) = delete;
    EsmReader(EsmReader&& other) noexcept;
    EsmReader& operator=(EsmReader&& other) noexcept;

    // Memory-maps the plugin rather than reading it. FalloutNV.esm is 234 MB
    // and a filtered walk touches only a fraction of it, so mapping saves both
    // the read itself (~80 ms warm) and the resident pages for everything the
    // walk skips. Falls back to a plain read if the mapping fails.
    bool open(const std::filesystem::path& path);
    const std::string& lastError() const { return m_lastError; }

    struct Visitor {
        // Called on entering a GRUP. Return false to skip the group's
        // contents entirely (onGroupExit is still called).
        std::function<bool(const EsmGroupView&)> onGroupEnter;
        std::function<void(const EsmGroupView&)> onGroupExit;
        // Called with a record's header BEFORE its body is decompressed or
        // split into subrecords. Return false to skip the record entirely, at
        // the cost of only a header read.
        //
        // This is the difference between reading a plugin and parsing one.
        // FalloutNV.esm holds 29363 compressed LAND records; materializing
        // them costs ~129 MB of inflate and ~489 MB of heap, and a caller
        // cooking an interior cell wants none of it. Leave this null to
        // materialize every record.
        std::function<bool(const EsmRecordHeaderView&)> onRecordHeader;
        // See the EsmRecordView comment: extract what you need from `record`
        // synchronously — its subrecord data pointers do not outlive this call.
        std::function<void(const EsmRecordView&)> onRecord;
    };

    // Depth-first walk of the whole file.
    bool walk(const Visitor& visitor);

    // Compressed records whose deflate stream produced every declared byte but
    // whose trailing checksum did not verify. These are accepted rather than
    // treated as fatal — retail FalloutNV.esm ships exactly one (a LAND,
    // formID 0x150FC0). A nonzero count here is worth reporting, not ignoring;
    // a large one means something is actually wrong with the file.
    std::uint32_t toleratedChecksumFailures() const { return m_toleratedChecksumFailures; }

private:
    // Releases the mapping (if any) and drops back to an empty reader.
    void close();

    // The plugin bytes, however they got here. When the file is mapped,
    // m_data points into the mapping and m_ownedBytes is empty; on the
    // fallback path m_data points at m_ownedBytes. Either way the buffer is
    // stable for the reader's lifetime, which the subrecord views rely on.
    const std::uint8_t* m_data = nullptr;
    std::size_t m_size = 0;
    std::vector<std::uint8_t> m_ownedBytes;
    // Platform mapping handles, owned and interpreted only by esm_reader.cc.
    void* m_mappingAddress = nullptr;
    std::uint64_t m_mappingHandles[2] = {0, 0};

    std::string m_lastError;
    std::uint32_t m_toleratedChecksumFailures = 0;
};

}  // namespace odai::importer::fnv
