#pragma once

// Generic reader for Bethesda TES4-family ESM/ESP plugin files: Oblivion
// (20-byte record header) and Fallout 3 / Fallout: New Vegas / Skyrim
// (24-byte). Morrowind's 16-byte TES3 header is a different format and is not
// supported. This module only understands the generic GRUP/record/subrecord
// container format; it has no idea what a CELL or STAT record means. See
// fallout_records.h for typed extraction built on top of this walker.
//
// Format reference (all fields little-endian):
//   GRUP header: "GRUP", groupSize (includes this header), label (4 bytes,
//     meaning depends on groupType), groupType (int32), stamp (uint16),
//     unknown (uint16) — 20 bytes so far, which is the whole header on
//     Oblivion — then versionControlInfo (uint32) on Fallout 3 and later.
//   Record header: 4-char type, dataSize (excludes this header), flags
//     (bit 0x00040000 = compressed), formID, versionControlInfo — 20 bytes,
//     the whole header on Oblivion — then formVersion (uint16) + unknown
//     (uint16) on Fallout 3 and later.
//   Record data: a sequence of subrecords, each a 4-char type + uint16 size
//     + that many bytes of data. When compressed (flags bit set), the record
//     data is [uint32 decompressedSize][zlib deflate stream] instead.
//     Identical in both generations — only the two container headers grew.
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

// Which generation's container headers a plugin uses. Nothing else about the
// walk depends on the game, so this is deliberately named after the layout
// break rather than after a title.
enum class EsmPluginFormat {
    kFallout3,  // 24-byte record and GRUP headers: Fallout 3, New Vegas, Skyrim
    kOblivion,  // 20-byte record and GRUP headers: Oblivion and the TES4-era DLC
    // 16-byte record headers: Morrowind. NOT just a narrower header --
    //   TES3: type, dataSize, header(unused), flags
    //   TES4: type, dataSize, flags,          formId
    // so the flags move and THE FORMID DOES NOT EXIST. Morrowind keys records
    // by a string id instead, which is why every formId this walker reports for
    // a TES3 plugin is 0 and callers have to identify records by name.
    //
    // There is also no GRUP tree at all: records are a flat list. That needs no
    // code, because no TES3 record is named "GRUP" and the group branch simply
    // never fires.
    kMorrowind,
};

std::size_t esmRecordHeaderSize(EsmPluginFormat format);
std::size_t esmGroupHeaderSize(EsmPluginFormat format);

// Sniffs the header generation from a plugin's opening TES4 record, given at
// least its first 28 bytes.
//
// The discriminator is where "HEDR" lands: Fallout 3 inserted a 4-byte
// formVersion/unknown field after the record's versionControlInfo, so a
// TES4-era plugin's first subrecord starts at offset 20 and a Fallout-era
// one's at 24. Measured across all 21 retail plugins in both installs —
// Oblivion.esm plus its ten DLC .esp, FalloutNV.esm plus its nine — with no
// ambiguity: offset 24 in an Oblivion plugin holds HEDR's *size* (12), which
// can never spell "HEDR".
//
// The HEDR version float does NOT discriminate and must not be used for this:
// Oblivion writes 1.0, Fallout 3 writes 0.94 and New Vegas 1.34, so 1.0 is not
// a boundary between them.
//
// Morrowind is identified by its own magic instead: a TES3 plugin opens with
// the literal record type "TES3", which no later generation writes.
//
// Returns kFallout3 for anything it cannot positively identify, so every
// pre-existing caller keeps the behaviour it had before this existed.
EsmPluginFormat detectEsmPluginFormat(const std::uint8_t* bytes, std::size_t size);

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
    // Byte offset of this GRUP's own header, and its declared size (which
    // includes that header). Together they bound the group exactly, which
    // is what lets a caller record a cell's children group during one full pass
    // and later re-walk just that range with walkRange() -- the basis of
    // streaming a single cell without re-reading the plugin.
    std::uint64_t fileOffset = 0;
    std::uint32_t groupSize = 0;
};

// The cheap half of a record: everything readable from its header, with no
// decompression and no subrecord parsing. `type` points into the reader's own
// file buffer and is valid only for the duration of the onRecordHeader
// callback.
struct EsmRecordHeaderView {
    std::string_view type;
    std::uint32_t formId = 0;
    std::uint32_t flags = 0;
    // Byte offset of this record's own header.
    std::uint64_t fileOffset = 0;
    // Bytes of record DATA, excluding the header. Together with fileOffset and
    // esmRecordHeaderSize() this gives the record's full extent, which is what
    // lets a caller record a single record's byte range and re-walk it later --
    // the only way to address a Morrowind cell, which has no children group.
    std::uint32_t dataSize = 0;
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

    // Sniffed by open() from the plugin's own TES4 record. An empty or
    // unopened reader reports kFallout3.
    EsmPluginFormat pluginFormat() const { return m_pluginFormat; }

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

    // Depth-first walk of one byte range, which must begin exactly on a GRUP or
    // record header and end on a boundary. Used to revisit a single cell's
    // children group recorded by an earlier full pass; because the plugin is
    // memory-mapped this costs a pointer walk plus whatever decompression that
    // cell's own records need, not a re-read of the file.
    bool walkRange(std::uint64_t beginOffset, std::uint64_t endOffset, const Visitor& visitor);

    // Compressed records whose deflate stream produced every declared byte but
    // whose trailing checksum did not verify. These are accepted rather than
    // treated as fatal — retail FalloutNV.esm ships exactly one (a LAND,
    // formID 0x150FC0). A nonzero count here is worth reporting, not ignoring;
    // a large one means something is actually wrong with the file.
    std::uint32_t toleratedChecksumFailures() const { return m_toleratedChecksumFailures; }

private:
    // Releases the mapping (if any) and drops back to an empty reader.
    void close();

    // Common tail of open()'s three acquisition paths (Win32 mapping, POSIX
    // mapping, whole-file read): sniffs the header generation now that m_data
    // is live. One place, so the mapped and read paths cannot disagree about
    // the same file. Always returns true.
    bool finishOpen();

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
    // Sniffed once in open(), never re-derived per record: the two container
    // header sizes are a property of the file, and a per-record guess would be
    // both slower and able to disagree with itself mid-walk.
    EsmPluginFormat m_pluginFormat = EsmPluginFormat::kFallout3;

};

}  // namespace odai::importer::fnv
