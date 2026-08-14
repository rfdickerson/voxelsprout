#include "import/fnv/lz4_frame.h"

#include <cstring>

namespace odai::importer::fnv {

namespace {

constexpr std::uint32_t kLz4FrameMagic = 0x184D2204u;

// FLG byte, LZ4 frame spec section "Frame Descriptor".
constexpr std::uint8_t kFlgVersionMask = 0xC0u;
constexpr std::uint8_t kFlgVersion1 = 0x40u;
constexpr std::uint8_t kFlgBlockChecksum = 0x10u;
constexpr std::uint8_t kFlgContentSize = 0x08u;
constexpr std::uint8_t kFlgContentChecksum = 0x04u;
constexpr std::uint8_t kFlgDictId = 0x01u;

constexpr std::uint32_t kBlockUncompressedFlag = 0x80000000u;

bool readU32Le(const std::uint8_t* bytes, std::size_t size, std::size_t& cursor, std::uint32_t& out) {
    if (cursor + 4u > size) {
        return false;
    }
    out = static_cast<std::uint32_t>(bytes[cursor]) |
          (static_cast<std::uint32_t>(bytes[cursor + 1u]) << 8) |
          (static_cast<std::uint32_t>(bytes[cursor + 2u]) << 16) |
          (static_cast<std::uint32_t>(bytes[cursor + 3u]) << 24);
    cursor += 4u;
    return true;
}

// A length encoded as "15 means keep adding 255-terminated bytes". Used for both
// the literal run and the match length, which is why it is factored out: the two
// call sites are identical and getting one of them subtly different is the
// classic way to write a decoder that works until it does not.
bool readVarLength(
    const std::uint8_t* bytes, std::size_t size, std::size_t& cursor, std::size_t& length) {
    while (true) {
        if (cursor >= size) {
            return false;
        }
        const std::uint8_t part = bytes[cursor++];
        length += part;
        if (part != 0xFFu) {
            return true;
        }
    }
}

// One LZ4 block, appended to `out`.
//
// Matches are copied byte by byte on purpose. An LZ4 match may overlap its own
// destination -- offset 1 with length 40 is how the format encodes a run of one
// repeated byte -- so a memcpy or a memmove here produces the wrong bytes for
// exactly the inputs the format uses most.
bool decodeBlock(
    const std::uint8_t* bytes,
    std::size_t size,
    std::vector<std::uint8_t>& out,
    std::string& outError) {
    std::size_t cursor = 0;
    while (cursor < size) {
        const std::uint8_t token = bytes[cursor++];

        std::size_t literalLength = static_cast<std::size_t>(token >> 4);
        if (literalLength == 15u && !readVarLength(bytes, size, cursor, literalLength)) {
            outError = "LZ4: truncated literal length";
            return false;
        }
        if (cursor + literalLength > size) {
            outError = "LZ4: literal run runs past the end of the block";
            return false;
        }
        out.insert(out.end(), bytes + cursor, bytes + cursor + literalLength);
        cursor += literalLength;

        // The last sequence in a block is literals only and stops here. The spec
        // guarantees it carries no match, so running out of input now is the
        // normal end, not a truncation.
        if (cursor == size) {
            return true;
        }
        if (cursor + 2u > size) {
            outError = "LZ4: truncated match offset";
            return false;
        }
        const std::size_t offset =
            static_cast<std::size_t>(bytes[cursor]) |
            (static_cast<std::size_t>(bytes[cursor + 1u]) << 8);
        cursor += 2u;
        if (offset == 0u || offset > out.size()) {
            outError = "LZ4: match offset points outside the output produced so far";
            return false;
        }

        std::size_t matchLength = static_cast<std::size_t>(token & 0x0Fu);
        if (matchLength == 15u && !readVarLength(bytes, size, cursor, matchLength)) {
            outError = "LZ4: truncated match length";
            return false;
        }
        matchLength += 4u;  // the minimum match, never encoded

        const std::size_t start = out.size() - offset;
        out.reserve(out.size() + matchLength);
        for (std::size_t i = 0; i < matchLength; ++i) {
            out.push_back(out[start + i]);
        }
    }
    return true;
}

}  // namespace

bool isLz4Frame(const std::uint8_t* bytes, std::size_t size) {
    if (bytes == nullptr || size < 4u) {
        return false;
    }
    std::size_t cursor = 0;
    std::uint32_t magic = 0;
    return readU32Le(bytes, size, cursor, magic) && magic == kLz4FrameMagic;
}

bool lz4FrameDecompress(
    const std::uint8_t* bytes,
    std::size_t size,
    std::size_t expectedSize,
    std::vector<std::uint8_t>& outBytes,
    std::string& outError) {
    outBytes.clear();
    if (bytes == nullptr) {
        outError = "LZ4: null input";
        return false;
    }
    std::size_t cursor = 0;
    std::uint32_t magic = 0;
    if (!readU32Le(bytes, size, cursor, magic) || magic != kLz4FrameMagic) {
        outError = "LZ4: not a frame (bad magic)";
        return false;
    }
    if (cursor + 3u > size) {  // FLG, BD, header checksum
        outError = "LZ4: truncated frame descriptor";
        return false;
    }
    const std::uint8_t flg = bytes[cursor++];
    ++cursor;  // BD: block maximum size, which a decoder that grows its output does not need
    if ((flg & kFlgVersionMask) != kFlgVersion1) {
        outError = "LZ4: unsupported frame version";
        return false;
    }
    if ((flg & kFlgContentSize) != 0u) {
        cursor += 8u;
    }
    if ((flg & kFlgDictId) != 0u) {
        // A dictionary the caller would have to supply, and nothing here can.
        // Rejected rather than skipped: decoding without it silently produces
        // wrong bytes instead of failing.
        outError = "LZ4: dictionary frames are not supported";
        return false;
    }
    ++cursor;  // header checksum, not verified
    if (cursor > size) {
        outError = "LZ4: truncated frame descriptor";
        return false;
    }

    if (expectedSize != 0u) {
        outBytes.reserve(expectedSize);
    }
    while (true) {
        std::uint32_t blockSize = 0;
        if (!readU32Le(bytes, size, cursor, blockSize)) {
            outError = "LZ4: truncated block size";
            return false;
        }
        if (blockSize == 0u) {
            break;  // end mark
        }
        const bool uncompressed = (blockSize & kBlockUncompressedFlag) != 0u;
        const std::size_t payloadSize = static_cast<std::size_t>(blockSize & ~kBlockUncompressedFlag);
        if (cursor + payloadSize > size) {
            outError = "LZ4: block runs past the end of the frame";
            return false;
        }
        if (uncompressed) {
            outBytes.insert(outBytes.end(), bytes + cursor, bytes + cursor + payloadSize);
        } else if (!decodeBlock(bytes + cursor, payloadSize, outBytes, outError)) {
            return false;
        }
        cursor += payloadSize;
        if ((flg & kFlgBlockChecksum) != 0u) {
            cursor += 4u;  // xxHash32 of the block, not verified
        }
    }
    if ((flg & kFlgContentChecksum) != 0u) {
        cursor += 4u;
    }

    // Deliberately after the decode rather than as a running cap: a frame whose
    // blocks are individually well-formed but sum to the wrong length is corrupt,
    // and finding that out here names it as such instead of leaving a truncated
    // mesh to fail somewhere far downstream.
    if (expectedSize != 0u && outBytes.size() != expectedSize) {
        outError = "LZ4: decoded " + std::to_string(outBytes.size()) + " bytes, expected " +
                   std::to_string(expectedSize);
        return false;
    }
    return true;
}

}  // namespace odai::importer::fnv
