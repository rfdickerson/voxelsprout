#pragma once

// LZ4 frame DECOMPRESSION, and nothing else.
//
// Skyrim Special Edition's BSA v105 compresses its entries with LZ4 rather than
// the zlib every earlier Bethesda archive used, so without this the whole SSE
// asset set is unreadable -- 19443 files in `Skyrim - Meshes0.bsa` alone.
//
// WHY THIS IS HERE RATHER THAN A DEPENDENCY. Decompression is about 120 lines
// and the format is frozen; the encoder, the dictionary support, the legacy
// frame variants and the streaming API are the parts of liblz4 that are large,
// and none of them are wanted. Against that, a vcpkg dependency would be one
// manifest line and six link lines. The deciding argument is not size but
// verification: this exact algorithm was run over 3000 retail SSE meshes before
// it was written here, and every one produced a well-formed NIF header, so what
// is vendored is a decoder that has already been checked against the only data
// it will ever be asked to read. Take the dependency the moment anything needs
// to WRITE lz4, or needs dictionaries or the streaming API.
//
// Untrusted input is assumed: every length, offset and match is bounds-checked
// against the buffers, and a malformed frame reports an error rather than
// reading out of range.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace odai::importer::fnv {

// True when `bytes` opens with the LZ4 frame magic (0x184D2204). Cheap enough to
// call before committing to a decode, which is what lets a caller tell an LZ4
// payload from a zlib one without knowing the archive version.
[[nodiscard]] bool isLz4Frame(const std::uint8_t* bytes, std::size_t size);

// Decodes one LZ4 frame. `expectedSize` is what the container claims the result
// should be; pass 0 when it is unknown. A nonzero value is enforced, because a
// frame that decodes to the wrong length is corrupt in a way that is otherwise
// only noticed much later, as an unparseable mesh.
[[nodiscard]] bool lz4FrameDecompress(
    const std::uint8_t* bytes,
    std::size_t size,
    std::size_t expectedSize,
    std::vector<std::uint8_t>& outBytes,
    std::string& outError);

}  // namespace odai::importer::fnv
