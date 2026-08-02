#include "import/fnv/esm_reader.h"

#include <cstring>
#include <fstream>

#include <zlib.h>

namespace odai::importer::fnv {

namespace {

constexpr std::uint32_t kRecordFlagCompressed = 0x00040000u;
constexpr std::size_t kGroupHeaderSize = 24u;
constexpr std::size_t kRecordHeaderSize = 24u;
constexpr std::size_t kSubrecordHeaderSize = 6u;  // 4-char type + uint16 size

bool typeIs(const std::uint8_t* bytes, const char* fourCc) {
    return std::memcmp(bytes, fourCc, 4) == 0;
}

std::string readFourCc(const std::uint8_t* bytes) {
    return std::string(reinterpret_cast<const char*>(bytes), 4);
}

std::uint16_t readU16(const std::uint8_t* bytes) {
    std::uint16_t value = 0;
    std::memcpy(&value, bytes, sizeof(value));
    return value;
}

std::uint32_t readU32(const std::uint8_t* bytes) {
    std::uint32_t value = 0;
    std::memcpy(&value, bytes, sizeof(value));
    return value;
}

// Splits a contiguous record-data buffer into subrecords, honoring the
// "XXXX"-prefixed oversized-subrecord convention. Subrecord views point
// directly into `data`, so `data` must outlive the returned vector.
bool parseSubrecords(const std::uint8_t* data, std::size_t size, std::vector<EsmSubrecordView>& out) {
    std::size_t pos = 0;
    std::int64_t pendingOverrideSize = -1;
    while (pos + kSubrecordHeaderSize <= size) {
        const std::string type = readFourCc(data + pos);
        const std::uint16_t declaredSize = readU16(data + pos + 4);
        pos += kSubrecordHeaderSize;

        if (type == "XXXX") {
            if (declaredSize != 4u || pos + 4u > size) {
                return false;
            }
            pendingOverrideSize = static_cast<std::int64_t>(readU32(data + pos));
            pos += 4u;
            continue;
        }

        const std::size_t actualSize = pendingOverrideSize >= 0
            ? static_cast<std::size_t>(pendingOverrideSize)
            : static_cast<std::size_t>(declaredSize);
        pendingOverrideSize = -1;
        if (pos + actualSize > size) {
            return false;
        }

        EsmSubrecordView subrecord{};
        subrecord.type = type;
        subrecord.data = data + pos;
        subrecord.size = static_cast<std::uint32_t>(actualSize);
        out.push_back(subrecord);
        pos += actualSize;
    }
    return true;
}

}  // namespace

bool EsmReader::open(const std::filesystem::path& path) {
    m_lastError.clear();
    m_bytes.clear();

    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        m_lastError = "Failed to open ESM/ESP file: " + path.string();
        return false;
    }
    const auto fileSize = static_cast<std::size_t>(input.tellg());
    input.seekg(0);
    m_bytes.resize(fileSize);
    if (fileSize != 0 &&
        !input.read(reinterpret_cast<char*>(m_bytes.data()), static_cast<std::streamsize>(fileSize))) {
        m_lastError = "Failed to read ESM/ESP file: " + path.string();
        m_bytes.clear();
        return false;
    }
    return true;
}

bool EsmReader::walk(const Visitor& visitor) {
    m_lastError.clear();
    if (m_bytes.empty()) {
        m_lastError = "ESM/ESP reader has no file open";
        return false;
    }

    // Iterative stack-based walk (avoids unbounded C++ call-stack recursion
    // on deeply nested worldspace/cell group hierarchies). Each frame tracks
    // the group that owns it (if any) so onGroupExit fires once that group's
    // full byte range has been consumed, giving proper depth-first
    // enter/exit ordering.
    struct Frame {
        std::size_t pos;
        std::size_t end;
        bool hasGroup = false;
        EsmGroupView group;
    };
    std::vector<Frame> stack;
    stack.push_back(Frame{0, m_bytes.size(), false, {}});

    // Decompression scratch buffer reused per compressed record so we don't
    // reallocate for every one in a large plugin.
    std::vector<std::uint8_t> decompressScratch;

    while (!stack.empty()) {
        if (stack.back().pos >= stack.back().end) {
            Frame finished = std::move(stack.back());
            stack.pop_back();
            if (finished.hasGroup && visitor.onGroupExit) {
                visitor.onGroupExit(finished.group);
            }
            continue;
        }
        Frame& frame = stack.back();

        if (frame.pos + 4 > frame.end) {
            m_lastError = "Truncated GRUP/record tag";
            return false;
        }

        if (typeIs(m_bytes.data() + frame.pos, "GRUP")) {
            if (frame.pos + kGroupHeaderSize > frame.end) {
                m_lastError = "Truncated GRUP header";
                return false;
            }
            const std::uint8_t* header = m_bytes.data() + frame.pos;
            const std::uint32_t groupSize = readU32(header + 4);
            if (groupSize < kGroupHeaderSize || frame.pos + groupSize > frame.end) {
                m_lastError = "Malformed GRUP size";
                return false;
            }

            EsmGroupView group{};
            group.rawLabel = std::string(reinterpret_cast<const char*>(header + 8), 4);
            group.groupType = static_cast<std::int32_t>(readU32(header + 12));

            const std::size_t contentStart = frame.pos + kGroupHeaderSize;
            const std::size_t contentEnd = frame.pos + groupSize;
            frame.pos = contentEnd;  // resume parent after this group once popped

            const bool descend = !visitor.onGroupEnter || visitor.onGroupEnter(group);
            if (descend && contentStart < contentEnd) {
                stack.push_back(Frame{contentStart, contentEnd, true, group});
            } else if (visitor.onGroupExit) {
                visitor.onGroupExit(group);
            }
            continue;
        }

        // Regular record.
        if (frame.pos + kRecordHeaderSize > frame.end) {
            m_lastError = "Truncated record header";
            return false;
        }
        const std::uint8_t* header = m_bytes.data() + frame.pos;
        const std::uint32_t dataSize = readU32(header + 4);
        const std::uint32_t flags = readU32(header + 8);
        const std::uint32_t formId = readU32(header + 12);
        const std::size_t dataStart = frame.pos + kRecordHeaderSize;
        const std::size_t dataEnd = dataStart + dataSize;
        if (dataEnd > frame.end) {
            m_lastError = "Truncated record data";
            return false;
        }

        EsmRecordView record{};
        record.type = readFourCc(header);
        record.formId = formId;
        record.flags = flags;

        if ((flags & kRecordFlagCompressed) != 0u) {
            if (dataSize < 4u) {
                m_lastError = "Compressed record missing size prefix: " + record.type;
                return false;
            }
            const std::uint32_t decompressedSize = readU32(m_bytes.data() + dataStart);
            decompressScratch.assign(decompressedSize, 0u);
            uLongf destLen = static_cast<uLongf>(decompressedSize);
            const int result = decompressedSize == 0
                ? Z_OK
                : uncompress(
                      reinterpret_cast<Bytef*>(decompressScratch.data()),
                      &destLen,
                      m_bytes.data() + dataStart + 4,
                      static_cast<uLong>(dataSize - 4u));
            if (result != Z_OK || destLen != decompressedSize) {
                m_lastError = "zlib inflate failed for record: " + record.type;
                return false;
            }
            if (!parseSubrecords(decompressScratch.data(), decompressScratch.size(), record.subrecords)) {
                m_lastError = "Malformed subrecords in compressed record: " + record.type;
                return false;
            }
            if (visitor.onRecord) {
                visitor.onRecord(record);
            }
        } else {
            if (!parseSubrecords(m_bytes.data() + dataStart, dataSize, record.subrecords)) {
                m_lastError = "Malformed subrecords in record: " + record.type;
                return false;
            }
            if (visitor.onRecord) {
                visitor.onRecord(record);
            }
        }

        frame.pos = dataEnd;
    }

    return true;
}

}  // namespace odai::importer::fnv
