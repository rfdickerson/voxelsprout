#include "import/fnv/esm_reader.h"

#include <cstring>
#include <fstream>
#include <utility>

#include <zlib.h>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

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

// Inflates one compressed record body into `out`, sized to `declaredSize`.
//
// Deliberately does NOT use zlib's uncompress(), which fails the whole call on
// a bad adler32 trailer even when every declared byte decoded correctly.
// Retail FalloutNV.esm contains exactly one such record (a LAND, formID
// 0x150FC0): its deflate stream is intact and yields all 4385 declared bytes,
// but the trailer checksum is wrong, and uncompress() rejecting it aborted the
// entire plugin walk. The declared size is the authority here — if that many
// bytes came out, the record is usable.
//
// Sets `outTrailerFailed` when the stream decoded fully but ended on a
// checksum/trailer error, so the caller can report it rather than hide it.
bool inflateRecordBody(
    const std::uint8_t* source,
    std::size_t sourceSize,
    std::uint32_t declaredSize,
    std::vector<std::uint8_t>& out,
    bool& outTrailerFailed
) {
    outTrailerFailed = false;
    out.assign(declaredSize, 0u);
    if (declaredSize == 0u) {
        return true;
    }

    z_stream stream{};
    if (inflateInit(&stream) != Z_OK) {
        return false;
    }
    stream.next_in = const_cast<Bytef*>(reinterpret_cast<const Bytef*>(source));
    stream.avail_in = static_cast<uInt>(sourceSize);
    stream.next_out = reinterpret_cast<Bytef*>(out.data());
    stream.avail_out = static_cast<uInt>(declaredSize);

    const int result = inflate(&stream, Z_FINISH);
    const uLong produced = stream.total_out;
    inflateEnd(&stream);

    if (produced != declaredSize) {
        return false;
    }
    // Z_STREAM_END is the clean case. Z_BUF_ERROR means the output buffer
    // filled exactly as the input ran out, which is fine because `produced`
    // already matched. Anything else means the trailer did not verify.
    outTrailerFailed = (result != Z_STREAM_END && result != Z_BUF_ERROR);
    return true;
}

}  // namespace

EsmReader::~EsmReader() {
    close();
}

EsmReader::EsmReader(EsmReader&& other) noexcept {
    *this = std::move(other);
}

EsmReader& EsmReader::operator=(EsmReader&& other) noexcept {
    if (this == &other) {
        return *this;
    }
    close();
    m_ownedBytes = std::move(other.m_ownedBytes);
    m_mappingAddress = other.m_mappingAddress;
    m_mappingHandles[0] = other.m_mappingHandles[0];
    m_mappingHandles[1] = other.m_mappingHandles[1];
    m_size = other.m_size;
    // Rebind rather than copy: on the fallback path m_data must point at this
    // object's own vector, not the moved-from one's.
    m_data = other.m_mappingAddress != nullptr
        ? static_cast<const std::uint8_t*>(m_mappingAddress)
        : m_ownedBytes.data();
    m_lastError = std::move(other.m_lastError);
    m_toleratedChecksumFailures = other.m_toleratedChecksumFailures;

    other.m_mappingAddress = nullptr;
    other.m_mappingHandles[0] = 0;
    other.m_mappingHandles[1] = 0;
    other.m_data = nullptr;
    other.m_size = 0;
    return *this;
}

void EsmReader::close() {
    if (m_mappingAddress != nullptr) {
#ifdef _WIN32
        ::UnmapViewOfFile(m_mappingAddress);
        if (m_mappingHandles[0] != 0) {
            ::CloseHandle(reinterpret_cast<HANDLE>(m_mappingHandles[0]));
        }
        if (m_mappingHandles[1] != 0) {
            ::CloseHandle(reinterpret_cast<HANDLE>(m_mappingHandles[1]));
        }
#else
        ::munmap(const_cast<void*>(m_mappingAddress), m_size);
        if (m_mappingHandles[0] != 0) {
            ::close(static_cast<int>(m_mappingHandles[0]) - 1);
        }
#endif
    }
    m_mappingAddress = nullptr;
    m_mappingHandles[0] = 0;
    m_mappingHandles[1] = 0;
    m_ownedBytes.clear();
    m_ownedBytes.shrink_to_fit();
    m_data = nullptr;
    m_size = 0;
}

bool EsmReader::open(const std::filesystem::path& path) {
    close();
    m_lastError.clear();

    std::error_code sizeError;
    const auto fileSize = static_cast<std::size_t>(std::filesystem::file_size(path, sizeError));
    if (sizeError) {
        m_lastError = "Failed to stat ESM/ESP file: " + path.string();
        return false;
    }
    if (fileSize == 0) {
        return true;  // empty but valid; the walk simply visits nothing
    }

#ifdef _WIN32
    HANDLE file = ::CreateFileW(
        path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file != INVALID_HANDLE_VALUE) {
        HANDLE mapping = ::CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (mapping != nullptr) {
            void* view = ::MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
            if (view != nullptr) {
                m_mappingAddress = view;
                m_mappingHandles[0] = reinterpret_cast<std::uint64_t>(file);
                m_mappingHandles[1] = reinterpret_cast<std::uint64_t>(mapping);
                m_data = static_cast<const std::uint8_t*>(view);
                m_size = fileSize;
                return true;
            }
            ::CloseHandle(mapping);
        }
        ::CloseHandle(file);
    }
#else
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd >= 0) {
        void* view = ::mmap(nullptr, fileSize, PROT_READ, MAP_PRIVATE, fd, 0);
        if (view != MAP_FAILED) {
            // The walk is a forward scan, so tell the kernel to read ahead.
            ::madvise(view, fileSize, MADV_SEQUENTIAL);
            m_mappingAddress = view;
            // Offset by one so that fd 0 is distinguishable from "no handle".
            m_mappingHandles[0] = static_cast<std::uint64_t>(fd) + 1u;
            m_data = static_cast<const std::uint8_t*>(view);
            m_size = fileSize;
            return true;
        }
        ::close(fd);
    }
#endif

    // Mapping unavailable (unusual filesystem, sandbox, out of address space
    // on 32-bit): fall back to reading the whole file.
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        m_lastError = "Failed to open ESM/ESP file: " + path.string();
        return false;
    }
    m_ownedBytes.resize(fileSize);
    if (!input.read(reinterpret_cast<char*>(m_ownedBytes.data()), static_cast<std::streamsize>(fileSize))) {
        m_lastError = "Failed to read ESM/ESP file: " + path.string();
        m_ownedBytes.clear();
        return false;
    }
    m_data = m_ownedBytes.data();
    m_size = fileSize;
    return true;
}

bool EsmReader::walk(const Visitor& visitor) {
    m_lastError.clear();
    m_toleratedChecksumFailures = 0;
    if (m_data == nullptr || m_size == 0) {
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
    stack.push_back(Frame{0, m_size, false, {}});

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

        if (typeIs(m_data + frame.pos, "GRUP")) {
            if (frame.pos + kGroupHeaderSize > frame.end) {
                m_lastError = "Truncated GRUP header";
                return false;
            }
            const std::uint8_t* header = m_data + frame.pos;
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
        const std::uint8_t* header = m_data + frame.pos;
        const std::uint32_t dataSize = readU32(header + 4);
        const std::uint32_t flags = readU32(header + 8);
        const std::uint32_t formId = readU32(header + 12);
        const std::size_t dataStart = frame.pos + kRecordHeaderSize;
        const std::size_t dataEnd = dataStart + dataSize;
        if (dataEnd > frame.end) {
            m_lastError = "Truncated record data";
            return false;
        }

        // Give the caller a chance to reject this record before we pay for it.
        // Everything below — inflate, subrecord splitting, the EsmRecordView
        // itself — is skipped when the header alone is enough to say no.
        if (visitor.onRecordHeader) {
            EsmRecordHeaderView headerView{};
            headerView.type = std::string_view(reinterpret_cast<const char*>(header), 4u);
            headerView.formId = formId;
            headerView.flags = flags;
            if (!visitor.onRecordHeader(headerView)) {
                frame.pos = dataEnd;
                continue;
            }
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
            const std::uint32_t decompressedSize = readU32(m_data + dataStart);
            bool trailerFailed = false;
            if (!inflateRecordBody(
                    m_data + dataStart + 4,
                    dataSize - 4u,
                    decompressedSize,
                    decompressScratch,
                    trailerFailed)) {
                m_lastError = "zlib inflate failed for record: " + record.type;
                return false;
            }
            if (trailerFailed) {
                ++m_toleratedChecksumFailures;
            }
            if (!parseSubrecords(decompressScratch.data(), decompressScratch.size(), record.subrecords)) {
                m_lastError = "Malformed subrecords in compressed record: " + record.type;
                return false;
            }
            if (visitor.onRecord) {
                visitor.onRecord(record);
            }
        } else {
            if (!parseSubrecords(m_data + dataStart, dataSize, record.subrecords)) {
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
