#include "import/fnv/nif_scene.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <unordered_set>

namespace odai::importer::fnv {

namespace {

constexpr std::uint32_t kSupportedNifVersion = 0x14020007u;  // 20.2.0.7 (FO3/FNV/Skyrim base)
constexpr std::string_view kHeaderMagicPrefix = "Gamebryo File Format, Version 20.2.0.7";

// Bounds-checked sequential byte reader. Every multi-byte read fails cleanly
// (returns false) on truncation instead of reading out of bounds.
class ByteCursor {
public:
    ByteCursor(const std::uint8_t* data, std::size_t size) : m_data(data), m_size(size) {}

    bool readBytes(void* dst, std::size_t count) {
        if (m_pos + count > m_size) {
            return false;
        }
        std::memcpy(dst, m_data + m_pos, count);
        m_pos += count;
        return true;
    }

    template <typename T>
    bool read(T& out) {
        static_assert(std::is_trivially_copyable_v<T>);
        return readBytes(&out, sizeof(T));
    }

    // Length-prefixed string with a `LengthT` byte count prefix (no NUL terminator in the data).
    //
    // The length is validated against what is actually left BEFORE resizing.
    // Resizing first meant a 4-byte length read from a bad offset -- a corrupt
    // file, a truncated one, or a caller probing for a string that isn't there
    // -- allocated up to 4 GB, discovered the read failed, and freed it again.
    // That turned a bounded-cost probe into a multi-minute one.
    template <typename LengthT>
    bool readSizedString(std::string& out) {
        LengthT length = 0;
        if (!read(length)) {
            return false;
        }
        const std::size_t byteCount = static_cast<std::size_t>(length);
        if (m_pos + byteCount > m_size) {
            return false;
        }
        out.resize(byteCount);
        return byteCount == 0 || readBytes(out.data(), byteCount);
    }

    bool readLine(std::string& out, char terminator) {
        out.clear();
        char ch = 0;
        while (m_pos < m_size) {
            if (!read(ch)) {
                return false;
            }
            if (ch == terminator) {
                return true;
            }
            out.push_back(ch);
            if (out.size() > 256u) {
                return false;  // sanity bound; a real NIF header line is short
            }
        }
        return false;
    }

    std::size_t pos() const { return m_pos; }
    std::size_t size() const { return m_size; }
    // Only ever called with offsets derived from the header's own declared
    // block-size table, never from inferred field-length accounting — see
    // the file header comment.
    void seekAbsolute(std::size_t position) { m_pos = std::min(position, m_size); }

private:
    const std::uint8_t* m_data;
    std::size_t m_size;
    std::size_t m_pos = 0;
};

// Every NiNode-derived type retail Fallout: New Vegas actually uses. All of
// them share the "AVObject prefix, numChildren, children[]" layout — their
// extra fields come after, and readNiNode stops at the children list and
// resumes at the block's declared end — so one reader covers the set.
//
// This being a short allowlist was a real bug, and it failed in two directions
// at once. Geometry under an unlisted node was unreachable (3753 NiTriShape/
// NiTriStrips blocks across the retail archives, including every
// meshes\landscape\lod\* mesh), and — worse, because it is silent and wrong
// rather than silent and absent — an NiNode whose only parent was an unlisted
// node did not appear in `referencedAsChild`, so it was promoted to a root and
// walked with an identity transform. That misplaced 459 subtrees by as much as
// 25000 units. Neither case incremented skippedShapeCount.
bool isNodeTypeName(std::string_view typeName) {
    return typeName == "NiNode" || typeName == "BSFadeNode" || typeName == "NiBSAnimationNode" ||
        typeName == "BSMultiBoundNode" || typeName == "NiBillboardNode" || typeName == "BSOrderedNode" ||
        typeName == "BSValueNode" || typeName == "BSDamageStage" || typeName == "BSBlastNode" ||
        typeName == "BSDebrisNode" || typeName == "BSLeafAnimNode" || typeName == "BSTreeNode";
}

// A type this parser does not know, whose name looks like a node, is the exact
// shape of the bug above: it will silently drop or misplace whatever hangs off
// it. Worth surfacing rather than absorbing — mods and DLC use node types the
// base game happens not to.
bool looksLikeUnhandledNodeType(std::string_view typeName) {
    return !isNodeTypeName(typeName) && typeName.size() > 4u &&
        typeName.substr(typeName.size() - 4u) == "Node";
}

struct NifHeader {
    std::uint32_t version = 0;
    std::uint32_t userVersion = 0;
    std::uint32_t userVersion2 = 0;
    std::vector<std::string> blockTypeNames;
    std::vector<std::uint16_t> blockTypeIndex;  // per block
    std::vector<std::uint32_t> blockSize;       // per block, bytes
    std::vector<std::string> strings;           // global string table (names, texture paths, ...)
};

bool parseHeader(ByteCursor& cursor, NifHeader& header, std::string& outError) {
    std::string magicLine;
    if (!cursor.readLine(magicLine, '\n') || magicLine.rfind(kHeaderMagicPrefix, 0) != 0) {
        outError = "Not a Gamebryo NIF 20.2.0.7 file (unrecognized header line)";
        return false;
    }
    if (!cursor.read(header.version) || header.version != kSupportedNifVersion) {
        outError = "Unsupported NIF version (only 20.2.0.7 / 0x14020007 is supported)";
        return false;
    }
    std::uint8_t endianType = 0;
    if (!cursor.read(endianType) || endianType != 1u) {
        outError = "Unsupported NIF endianness (only little-endian archives are supported)";
        return false;
    }
    std::uint32_t numBlocks = 0;
    if (!cursor.read(header.userVersion) || !cursor.read(numBlocks) || !cursor.read(header.userVersion2)) {
        outError = "Truncated NIF header";
        return false;
    }

    // Creator + two export-info sized strings (u8-length-prefixed); content unused.
    for (int i = 0; i < 3; ++i) {
        std::string unused;
        if (!cursor.readSizedString<std::uint8_t>(unused)) {
            outError = "Truncated NIF header export-info strings";
            return false;
        }
    }

    std::uint16_t numBlockTypes = 0;
    if (!cursor.read(numBlockTypes)) {
        outError = "Truncated NIF header block-type count";
        return false;
    }
    header.blockTypeNames.resize(numBlockTypes);
    for (std::string& typeName : header.blockTypeNames) {
        if (!cursor.readSizedString<std::uint32_t>(typeName)) {
            outError = "Truncated NIF header block-type name";
            return false;
        }
    }

    header.blockTypeIndex.resize(numBlocks);
    for (std::uint16_t& index : header.blockTypeIndex) {
        if (!cursor.read(index) || index >= numBlockTypes) {
            outError = "Truncated or invalid NIF header block-type index table";
            return false;
        }
    }

    header.blockSize.resize(numBlocks);
    for (std::uint32_t& size : header.blockSize) {
        if (!cursor.read(size)) {
            outError = "Truncated NIF header block-size table";
            return false;
        }
    }

    std::uint32_t numStrings = 0;
    std::uint32_t maxStringLength = 0;
    if (!cursor.read(numStrings) || !cursor.read(maxStringLength)) {
        outError = "Truncated NIF header string-table sizes";
        return false;
    }
    header.strings.resize(numStrings);
    for (std::string& text : header.strings) {
        if (!cursor.readSizedString<std::uint32_t>(text)) {
            outError = "Truncated NIF header string table";
            return false;
        }
    }

    std::uint32_t numGroups = 0;
    if (!cursor.read(numGroups)) {
        outError = "Truncated NIF header group count";
        return false;
    }
    for (std::uint32_t i = 0; i < numGroups; ++i) {
        std::uint32_t unused = 0;
        if (!cursor.read(unused)) {
            outError = "Truncated NIF header group table";
            return false;
        }
    }

    return true;
}

// Row-major 4x4, translation in indices 3/7/11 — same convention imported_scene.cc uses.
struct Mat4 {
    float m[16] = {
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1};
};

Mat4 makeTrs(const float translation[3], const float rotation3x3[9], float scale) {
    Mat4 out{};
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            out.m[(row * 4) + col] = rotation3x3[(row * 3) + col] * scale;
        }
        out.m[(row * 4) + 3] = translation[row];
    }
    return out;
}

Mat4 multiply(const Mat4& a, const Mat4& b) {
    Mat4 out{};
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < 4; ++k) {
                sum += a.m[(row * 4) + k] * b.m[(k * 4) + col];
            }
            out.m[(row * 4) + col] = sum;
        }
    }
    return out;
}

std::array<float, 3> transformPoint(const Mat4& mat, const float p[3]) {
    return {
        (mat.m[0] * p[0]) + (mat.m[1] * p[1]) + (mat.m[2] * p[2]) + mat.m[3],
        (mat.m[4] * p[0]) + (mat.m[5] * p[1]) + (mat.m[6] * p[2]) + mat.m[7],
        (mat.m[8] * p[0]) + (mat.m[9] * p[1]) + (mat.m[10] * p[2]) + mat.m[11]};
}

// Directions ignore translation. Correct in general only via the
// inverse-transpose; NIF node scale is overwhelmingly uniform in practice
// (matches how imported_scene.cc's own normalizeVector(transformDirection())
// pattern already assumes for instance transforms), so the plain 3x3 is used
// and the result is renormalized.
std::array<float, 3> transformDirection(const Mat4& mat, const float d[3]) {
    std::array<float, 3> out{
        (mat.m[0] * d[0]) + (mat.m[1] * d[1]) + (mat.m[2] * d[2]),
        (mat.m[4] * d[0]) + (mat.m[5] * d[1]) + (mat.m[6] * d[2]),
        (mat.m[8] * d[0]) + (mat.m[9] * d[1]) + (mat.m[10] * d[2])};
    const float length = std::sqrt((out[0] * out[0]) + (out[1] * out[1]) + (out[2] * out[2]));
    if (length > 1e-6f) {
        out[0] /= length;
        out[1] /= length;
        out[2] /= length;
    }
    return out;
}

// Parsed NiAVObject-derived fields shared by NiNode and NiTriShape.
struct AvObjectFields {
    float translation[3] = {0, 0, 0};
    float rotation[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    float scale = 1.0f;
    std::vector<std::int32_t> children;  // only meaningful for NiNode
    std::int32_t dataRef = -1;           // only meaningful for NiTriShape/NiTriStrips
    std::int32_t nameRef = -1;           // index into the header's string table
    std::vector<std::int32_t> properties;  // NiProperty refs (shader, alpha, material)
};

// One BSShaderTextureSet: an array of texture paths, diffuse first, normal
// second, glow third. Paths are relative to Data\textures.
struct TextureSetBlock {
    std::vector<std::string> textures;
    bool valid = false;
};

bool readBsShaderTextureSet(ByteCursor& cursor, TextureSetBlock& out) {
    std::uint32_t count = 0;
    if (!cursor.read(count) || count > 64u) {
        return false;
    }
    out.textures.resize(count);
    for (std::string& text : out.textures) {
        if (!cursor.readSizedString<std::uint32_t>(text)) {
            return false;
        }
    }
    out.valid = true;
    return true;
}

// NiAlphaProperty: u16 flags then a u8 threshold. Bit 9 (0x200) is alpha test;
// bit 0 is alpha blend. Only the test flag maps onto this engine's cutout
// path, which is a discard in the fragment shader rather than sorted blending.
struct AlphaPropertyBlock {
    bool alphaTest = false;
    bool valid = false;
};

// BSShaderPPLightingProperty's texture-set ref, read by its real layout rather
// than searched for.
//
// This used to scan the block for any int32 that happened to index a
// BSShaderTextureSet, on the theory that "validated by type" made the search
// safe. It is not: the ref lives at block-relative offset 34, and offset 30
// holds Texture Clamp Mode, whose value is 3 in 54221 of 58608 retail blocks.
// Wherever block 3 also happened to be a texture set, the scan stopped on the
// clamp mode and returned the wrong texture — 275 shapes — and on
// BSShaderNoLightingProperty it grabbed the shaderType word at offset 14 for
// another 421. The sample that "confirmed" offset 30 had clamp mode 3 AND its
// texture set at block 3; the agreement was a collision, not a measurement.
//
//   NiObjectNET : nameRef u32, numExtraData u32, extraData refs, controller ref
//   NiProperty  : flags u16
//   BSShaderProperty      : shaderType u32, shaderFlags u32, shaderFlags2 u32,
//                           envMapScale f32
//   BSShaderLightingProperty : textureClampMode u32
//   BSShaderPPLightingProperty : textureSet ref  <-- what we want
bool readBsShaderTextureSetRef(ByteCursor& cursor, std::int32_t& outTextureSetRef) {
    std::int32_t nameRef = 0;
    if (!cursor.read(nameRef)) {
        return false;
    }
    std::uint32_t numExtraData = 0;
    if (!cursor.read(numExtraData) || numExtraData > 1024u) {
        return false;
    }
    for (std::uint32_t i = 0; i < numExtraData; ++i) {
        std::int32_t ref = 0;
        if (!cursor.read(ref)) {
            return false;
        }
    }
    std::int32_t controllerRef = 0;
    std::uint16_t flags = 0;
    std::uint32_t shaderType = 0;
    std::uint32_t shaderFlags = 0;
    std::uint32_t shaderFlags2 = 0;
    float envMapScale = 0.0f;
    std::uint32_t textureClampMode = 0;
    if (!cursor.read(controllerRef) || !cursor.read(flags) || !cursor.read(shaderType) ||
        !cursor.read(shaderFlags) || !cursor.read(shaderFlags2) || !cursor.read(envMapScale) ||
        !cursor.read(textureClampMode)) {
        return false;
    }
    return cursor.read(outTextureSetRef);
}

// BSShaderNoLightingProperty names its texture DIRECTLY, as a length-prefixed
// string, instead of pointing at a BSShaderTextureSet. That is the whole reason
// it needs its own reader: the texture-set path above finds nothing for these
// blocks, so every shape using one came out with no diffuse texture and shaded
// from the per-model hashed colour -- grey patches on an otherwise textured
// model, on 44 of 800 shapes in a 169-cell Mojave cook.
//
// Shares BSShaderProperty's prefix with the reader above, so the field walk is
// identical up to where the texture-set ref would be. Then the string.
// NiTexturingProperty reaches its texture through NiSourceTexture blocks, not a
// BSShaderTextureSet. It is the older Gamebryo path and retail FNV still uses it
// on plenty of geometry -- cliffs and rock formations especially -- so shapes
// carrying one came out with no diffuse texture at all and shaded from the
// per-model hashed colour.
//
// Both readers below SCAN the block for a value that validates, rather than
// walking to a computed offset. That is the same technique the texture-set
// resolution documents above, and for the same reason: the field layout of
// these types shifts with userVersion2, and two attempts at deriving an offset
// for BSShaderNoLightingProperty both landed wrong. A scan cannot be wrong by
// construction here -- a candidate is only accepted if it resolves to something
// that is actually a texture path.
//
// Scanning is cheap: these are 4-byte integer reads with no allocation.

bool looksLikeDdsPath(const std::string& value) {
    if (value.size() < 5u || value.size() > 256u) {
        return false;
    }
    for (const char ch : value) {
        if (static_cast<unsigned char>(ch) < 0x20u || static_cast<unsigned char>(ch) > 0x7eu) {
            return false;
        }
    }
    std::string lowered = value;
    std::transform(lowered.begin(), lowered.end(), lowered.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return lowered.compare(lowered.size() - 4u, 4u, ".dds") == 0;
}

// A NiSourceTexture names its file either through the header string table or
// inline. Accept the first word that resolves either way to a texture path.
bool findSourceTextureFileName(
    ByteCursor& cursor,
    const std::vector<std::string>& strings,
    std::string& outFileName
) {
    // Byte stride, NOT 4-byte. These types put a ushort (flags) and a byte
    // (hasBaseTexture / useExternal) ahead of their refs, so everything after is
    // misaligned: the cliff NIF stores its texture as `01 08 00 00 00` -- a
    // useExternal byte then string index 8 -- and a 4-byte-aligned scan steps
    // clean over it. That misalignment is why three separate attempts to reach
    // these fields by computed offset all missed.
    for (std::size_t offset = 0; offset + sizeof(std::int32_t) <= cursor.size(); offset += 1u) {
        cursor.seekAbsolute(offset);
        std::int32_t candidate = 0;
        if (!cursor.read(candidate)) {
            break;
        }
        // Two spellings, because NiSourceTexture uses both across the retail
        // archives: an index into the header string table, or the path stored
        // inline as a length-prefixed string. Try the index first (cheap), then
        // the inline form. Both are accepted only if the result is a .dds path,
        // so neither can match by coincidence.
        if (candidate >= 0 && static_cast<std::size_t>(candidate) < strings.size() &&
            looksLikeDdsPath(strings[static_cast<std::size_t>(candidate)])) {
            outFileName = strings[static_cast<std::size_t>(candidate)];
            return true;
        }
        cursor.seekAbsolute(offset);
        std::string inlineText;
        if (cursor.readSizedString<std::uint32_t>(inlineText) && looksLikeDdsPath(inlineText)) {
            outFileName = std::move(inlineText);
            return true;
        }
    }
    return false;
}

// A NiTexturingProperty's base texture is a ref to a NiSourceTexture. Accept the
// first word that indexes a block we already resolved a path for.
bool findTexturingPropertySource(
    ByteCursor& cursor,
    const std::vector<std::string>& sourceTexturePaths,
    std::string& outFileName
) {
    // Byte stride, NOT 4-byte. These types put a ushort (flags) and a byte
    // (hasBaseTexture / useExternal) ahead of their refs, so everything after is
    // misaligned: the cliff NIF stores its texture as `01 08 00 00 00` -- a
    // useExternal byte then string index 8 -- and a 4-byte-aligned scan steps
    // clean over it. That misalignment is why three separate attempts to reach
    // these fields by computed offset all missed.
    for (std::size_t offset = 0; offset + sizeof(std::int32_t) <= cursor.size(); offset += 1u) {
        cursor.seekAbsolute(offset);
        std::int32_t candidate = 0;
        if (!cursor.read(candidate)) {
            break;
        }
        if (candidate < 0 || static_cast<std::size_t>(candidate) >= sourceTexturePaths.size()) {
            continue;
        }
        if (!sourceTexturePaths[static_cast<std::size_t>(candidate)].empty()) {
            outFileName = sourceTexturePaths[static_cast<std::size_t>(candidate)];
            return true;
        }
    }
    return false;
}

bool readNiAlphaProperty(ByteCursor& cursor, std::uint32_t userVersion2, AlphaPropertyBlock& out) {
    AvObjectFields unusedNameFields{};
    // NiAlphaProperty derives from NiObjectNET: name, extra data, controller.
    if (!cursor.read(unusedNameFields.nameRef)) {
        return false;
    }
    std::uint32_t numExtraData = 0;
    if (!cursor.read(numExtraData) || numExtraData > 1024u) {
        return false;
    }
    for (std::uint32_t i = 0; i < numExtraData; ++i) {
        std::int32_t ref = 0;
        if (!cursor.read(ref)) {
            return false;
        }
    }
    std::int32_t controllerRef = 0;
    if (!cursor.read(controllerRef)) {
        return false;
    }
    (void)userVersion2;  // NiProperty flags stay 16-bit regardless
    std::uint16_t flags = 0;
    if (!cursor.read(flags)) {
        return false;
    }
    constexpr std::uint16_t kAlphaTestBit = 0x0200u;
    out.alphaTest = (flags & kAlphaTestBit) != 0u;
    out.valid = true;
    return true;
}

// Reads the NiObjectNET + NiAVObject fields common to NiNode and
// NiTriShape/NiTriStrips. Only reads through `scale` + property/collision
// refs — callers resume via the block's declared size, not by continuing
// this cursor, so anything after this point (children, data ref, ...) is
// read separately per block type. Returns false only on truncation this
// early in the block, which should not happen for a well-formed file.
bool readAvObjectPrefix(ByteCursor& cursor, std::uint32_t userVersion2, AvObjectFields& out) {
    if (!cursor.read(out.nameRef)) {
        return false;
    }
    std::uint32_t numExtraData = 0;
    if (!cursor.read(numExtraData)) {
        return false;
    }
    for (std::uint32_t i = 0; i < numExtraData; ++i) {
        std::int32_t ref = 0;
        if (!cursor.read(ref)) {
            return false;
        }
    }
    std::int32_t controllerRef = 0;
    if (!cursor.read(controllerRef)) {
        return false;
    }
    // NiAVObject::flags widened from ushort to uint at userVersion2 > 26.
    // Fallout: New Vegas writes 34, so this is a uint for every retail mesh.
    // Reading it as a ushort shifts everything after it by two bytes: verified
    // on clutter\RorchachTest\PipBoyCard01.nif, where the u16 reading yields
    // scale=2.3e-41 and numProperties=16256, and the u32 reading yields
    // scale=1.0, numProperties=0, numChildren=1. The bad parse made
    // readNiNode fail outright, which left the file with no root node and so
    // emitted no geometry at all — silently, because an empty model is not an
    // error.
    if (userVersion2 > 26u) {
        std::uint32_t flags32 = 0;
        if (!cursor.read(flags32)) {
            return false;
        }
    } else {
        std::uint16_t flags16 = 0;
        if (!cursor.read(flags16)) {
            return false;
        }
    }
    if (!cursor.read(out.translation)) {
        return false;
    }
    if (!cursor.read(out.rotation)) {
        return false;
    }
    if (!cursor.read(out.scale)) {
        return false;
    }
    std::uint32_t numProperties = 0;
    if (!cursor.read(numProperties)) {
        return false;
    }
    // Kept, not discarded: these refs are how a shape reaches its
    // BSShaderPPLightingProperty (-> BSShaderTextureSet) and NiAlphaProperty.
    for (std::uint32_t i = 0; i < numProperties; ++i) {
        std::int32_t ref = 0;
        if (!cursor.read(ref)) {
            return false;
        }
        out.properties.push_back(ref);
    }
    std::int32_t collisionRef = 0;
    if (!cursor.read(collisionRef)) {
        return false;
    }
    return true;
}

bool readNiNode(ByteCursor& cursor, std::uint32_t userVersion2, AvObjectFields& out) {
    if (!readAvObjectPrefix(cursor, userVersion2, out)) {
        return false;
    }
    std::uint32_t numChildren = 0;
    if (!cursor.read(numChildren)) {
        return false;
    }
    out.children.resize(numChildren);
    for (std::int32_t& child : out.children) {
        if (!cursor.read(child)) {
            return false;
        }
    }
    // Effects list follows; unused, and the caller resumes at the block's
    // declared end regardless, so it is intentionally not read here.
    return true;
}

bool readNiTriBasedGeom(ByteCursor& cursor, std::uint32_t userVersion2, AvObjectFields& out) {
    if (!readAvObjectPrefix(cursor, userVersion2, out)) {
        return false;
    }
    if (!cursor.read(out.dataRef)) {
        return false;
    }
    // Skin instance ref and any trailing material-data fields follow;
    // unused, and the caller resumes at the block's declared end.
    return true;
}

struct GeometryBlock {
    std::vector<float> positions;
    std::vector<float> normals;
    std::vector<float> uvs;  // UV set 0 only, 2 floats per vertex
    std::vector<std::uint32_t> triangleIndices;
    bool valid = false;
};

// Expands one triangle strip into an indexed triangle list.
//
// Strips alternate winding per triangle, and Bethesda stitches separate strips
// together with degenerate triangles (two equal indices) rather than starting a
// new strip, so both have to be handled or the mesh comes out inside-out and
// full of slivers.
void appendStripTriangles(const std::vector<std::uint16_t>& strip, std::vector<std::uint32_t>& out) {
    if (strip.size() < 3u) {
        return;
    }
    for (std::size_t i = 0; i + 2u < strip.size(); ++i) {
        const std::uint16_t a = strip[i];
        const std::uint16_t b = strip[i + 1u];
        const std::uint16_t c = strip[i + 2u];
        if (a == b || b == c || a == c) {
            continue;  // degenerate: a stitch between strips, not a triangle
        }
        out.push_back(a);
        if ((i % 2u) == 0u) {
            out.push_back(b);
            out.push_back(c);
        } else {
            out.push_back(c);
            out.push_back(b);
        }
    }
}

// The NiGeometryData + NiTriBasedGeomData prefix shared by NiTriShapeData and
// NiTriStripsData: everything from groupId through numTriangles. The caller
// then reads whichever tail its own block type declares.
//
// This is the least-certain part of this parser (see the file header comment)
// — vertex colors are read only far enough to skip their bytes correctly; if
// the final read position doesn't land inside the block's own declared size,
// the geometry is rejected rather than trusted.
//
// `outNumTriangles` is the count the tail needs; it is advisory for strips,
// where the real triangle count only falls out of expanding them.
bool readNiTriBasedGeomDataPrefix(ByteCursor& cursor, GeometryBlock& out, std::uint16_t& outNumTriangles) {
    std::int32_t groupId = 0;
    if (!cursor.read(groupId)) {
        return false;
    }
    std::uint16_t numVertices = 0;
    if (!cursor.read(numVertices)) {
        return false;
    }
    std::uint8_t keepFlags = 0;
    std::uint8_t compressFlags = 0;
    if (!cursor.read(keepFlags) || !cursor.read(compressFlags)) {
        return false;
    }
    std::uint8_t hasVertices = 0;
    if (!cursor.read(hasVertices)) {
        return false;
    }
    if (hasVertices != 0u) {
        out.positions.resize(static_cast<std::size_t>(numVertices) * 3u);
        for (std::uint16_t i = 0; i < numVertices; ++i) {
            float xyz[3];
            if (!cursor.read(xyz)) {
                return false;
            }
            out.positions[(i * 3u) + 0] = xyz[0];
            out.positions[(i * 3u) + 1] = xyz[1];
            out.positions[(i * 3u) + 2] = xyz[2];
        }
    }

    // VectorFlags: bits 0-5 are the UV-set COUNT, bit 12 means tangents follow
    // the normals. Whether normals are present is a separate bool byte after
    // this field, not a bit inside it.
    //
    // This was previously read as "bit 0 = has normals, bits 6-11 = UV count",
    // with no separate bool — a plausible guess that the synthetic test fixture
    // was then written to match, so nothing caught it. Against real data it
    // desynchronizes every geometry block: verified on
    // clutter\RorchachTest\PipBoyCard01.nif, whose NiTriStripsData has
    // vectorFlags 0x1001 (1 UV set + tangents) and a separate hasNormals byte
    // of 1. Parsing it this way lands exactly on the block's declared end.
    std::uint16_t vectorFlags = 0;
    if (!cursor.read(vectorFlags)) {
        return false;
    }
    // Bethesda writes BSVectorFlags, where bit 0 is a BOOLEAN "has UV" — one
    // set — and the upper bits are unrelated flags. Only stock Gamebryo's
    // NiVectorFlags treats bits 0-5 as a count. The two readings agree for
    // 0x0001 and 0x1001, which is 99.94% of retail, so a count mask looks
    // correct almost everywhere and then destroys the rest: `vectorFlags &
    // 0x3F` yields 33 for the 36 blocks storing 0x1FE1 and 3 for 0x1003, and
    // reading 33 UV sets over-consumes the block and drops the geometry.
    // Measured on meshes\clutter\hiddenvalley\nv_hv_graffiti01.nif and its
    // siblings, which emitted zero shapes.
    constexpr std::uint16_t kHasUvBit = 0x0001u;
    constexpr std::uint16_t kHasTangentsBit = 0x1000u;

    std::uint8_t hasNormals = 0;
    if (!cursor.read(hasNormals)) {
        return false;
    }
    if (hasNormals != 0u) {
        out.normals.resize(static_cast<std::size_t>(numVertices) * 3u);
        for (std::uint16_t i = 0; i < numVertices; ++i) {
            float xyz[3];
            if (!cursor.read(xyz)) {
                return false;
            }
            out.normals[(i * 3u) + 0] = xyz[0];
            out.normals[(i * 3u) + 1] = xyz[1];
            out.normals[(i * 3u) + 2] = xyz[2];
        }
        if ((vectorFlags & kHasTangentsBit) != 0u) {
            // Tangents + binormals: 2 * numVertices * Vector3, unused here.
            const std::size_t skipBytes = static_cast<std::size_t>(numVertices) * 3u * sizeof(float) * 2u;
            cursor.seekAbsolute(cursor.pos() + skipBytes);
        }
    }

    float boundingSphereCenter[3];
    float boundingSphereRadius = 0.0f;
    if (!cursor.read(boundingSphereCenter) || !cursor.read(boundingSphereRadius)) {
        return false;
    }

    std::uint8_t hasVertexColors = 0;
    if (!cursor.read(hasVertexColors)) {
        return false;
    }
    if (hasVertexColors != 0u) {
        const std::size_t skipBytes = static_cast<std::size_t>(numVertices) * 4u * sizeof(float);
        cursor.seekAbsolute(cursor.pos() + skipBytes);
    }

    // UV set 0 is kept; any further sets are skipped. Without UVs nothing
    // downstream can texture this geometry, which is why they are read rather
    // than stepped over as they used to be.
    const std::uint16_t uvSetCount = ((vectorFlags & kHasUvBit) != 0u) ? 1u : 0u;
    if (uvSetCount != 0u) {
        out.uvs.resize(static_cast<std::size_t>(numVertices) * 2u);
        for (std::uint16_t i = 0; i < numVertices; ++i) {
            float uv[2];
            if (!cursor.read(uv)) {
                return false;
            }
            out.uvs[(i * 2u) + 0] = uv[0];
            out.uvs[(i * 2u) + 1] = uv[1];
        }
        const std::size_t remainingSets = static_cast<std::size_t>(uvSetCount) - 1u;
        if (remainingSets != 0u) {
            cursor.seekAbsolute(
                cursor.pos() + (remainingSets * static_cast<std::size_t>(numVertices) * 2u * sizeof(float)));
        }
    }

    std::uint16_t consistencyType = 0;
    if (!cursor.read(consistencyType)) {
        return false;
    }
    std::int32_t additionalDataRef = 0;
    if (!cursor.read(additionalDataRef)) {
        return false;
    }

    return cursor.read(outNumTriangles);
}

// NiTriShapeData tail: an explicit triangle list.
bool readNiTriShapeDataTail(ByteCursor& cursor, std::uint16_t numTriangles, GeometryBlock& out) {
    std::uint32_t numTrianglePoints = 0;
    if (!cursor.read(numTrianglePoints)) {
        return false;
    }
    std::uint8_t hasTriangles = 0;
    if (!cursor.read(hasTriangles)) {
        return false;
    }
    if (hasTriangles != 0u) {
        out.triangleIndices.resize(static_cast<std::size_t>(numTriangles) * 3u);
        for (std::uint16_t i = 0; i < numTriangles; ++i) {
            std::uint16_t indices[3];
            if (!cursor.read(indices)) {
                return false;
            }
            out.triangleIndices[(i * 3u) + 0] = indices[0];
            out.triangleIndices[(i * 3u) + 1] = indices[1];
            out.triangleIndices[(i * 3u) + 2] = indices[2];
        }
    }
    return true;
}

// NiTriStripsData tail: per-strip index runs, expanded here into the same
// triangle list the rest of the pipeline already consumes. This is the
// dominant geometry form in Fallout: New Vegas — strips outnumber explicit
// triangle lists roughly 5.7 to 1 across the retail meshes.
bool readNiTriStripsDataTail(ByteCursor& cursor, GeometryBlock& out) {
    std::uint16_t numStrips = 0;
    if (!cursor.read(numStrips)) {
        return false;
    }
    std::vector<std::uint16_t> stripLengths(numStrips);
    for (std::uint16_t& length : stripLengths) {
        if (!cursor.read(length)) {
            return false;
        }
    }
    std::uint8_t hasPoints = 0;
    if (!cursor.read(hasPoints)) {
        return false;
    }
    if (hasPoints == 0u) {
        return true;
    }

    std::vector<std::uint16_t> strip;
    for (const std::uint16_t length : stripLengths) {
        strip.resize(length);
        for (std::uint16_t& point : strip) {
            if (!cursor.read(point)) {
                return false;
            }
        }
        appendStripTriangles(strip, out.triangleIndices);
    }
    return true;
}

// Reads one geometry-data block of either shape. `isStrips` selects the tail.
bool readGeometryData(ByteCursor& cursor, std::size_t blockEnd, bool isStrips, GeometryBlock& out) {
    std::uint16_t numTriangles = 0;
    if (!readNiTriBasedGeomDataPrefix(cursor, out, numTriangles)) {
        return false;
    }
    const bool tailOk = isStrips ? readNiTriStripsDataTail(cursor, out)
                                 : readNiTriShapeDataTail(cursor, numTriangles, out);
    if (!tailOk) {
        return false;
    }

    // Self-consistency check against the block's declared size.
    //
    // The old test was `pos() > blockEnd`, which can never fire: the cursor is
    // constructed over exactly blockSize bytes, readBytes refuses to run past
    // the end, and seekAbsolute clamps. So it asserted nothing, and a layout
    // error surfaced only as a truncation or as silently wrong geometry.
    //
    // Real data gives a check with teeth. Measured across the retail archives,
    // NiTriStripsData consumes its block exactly (48477 of 48477 with zero
    // trailing bytes), while NiTriShapeData always leaves at least the
    // trailing `Num Match Groups` u16. Anything else means the field walk
    // desynchronized, which is precisely how the UV-count and normals-flag
    // bugs presented.
    if (isStrips ? (cursor.pos() != blockEnd) : (cursor.pos() + sizeof(std::uint16_t) > blockEnd)) {
        return false;
    }

    out.valid = !out.positions.empty();
    return true;
}

}  // namespace

bool parseNifBlockSummary(
    const std::vector<std::uint8_t>& bytes, NifBlockSummary& outSummary, std::string& outError
) {
    outSummary = NifBlockSummary{};
    ByteCursor cursor(bytes.data(), bytes.size());
    NifHeader header;
    if (!parseHeader(cursor, header, outError)) {
        return false;
    }
    outSummary.blockTypeNames.reserve(header.blockTypeIndex.size());
    for (const std::uint16_t typeIndex : header.blockTypeIndex) {
        outSummary.blockTypeNames.push_back(header.blockTypeNames[typeIndex]);
    }
    outSummary.blockSizes = header.blockSize;
    outSummary.strings = header.strings;
    std::size_t offset = cursor.pos();
    for (const std::uint32_t size : header.blockSize) {
        outSummary.blockStarts.push_back(offset);
        offset += size;
    }
    return true;
}

bool parseNifStaticMesh(const std::vector<std::uint8_t>& bytes, NifModel& outModel, std::string& outError) {
    outModel = NifModel{};
    ByteCursor cursor(bytes.data(), bytes.size());
    NifHeader header;
    if (!parseHeader(cursor, header, outError)) {
        return false;
    }

    const std::size_t numBlocks = header.blockSize.size();
    std::vector<std::size_t> blockStart(numBlocks);
    std::vector<std::size_t> blockEnd(numBlocks);
    std::size_t cursorPos = cursor.pos();
    for (std::size_t i = 0; i < numBlocks; ++i) {
        blockStart[i] = cursorPos;
        blockEnd[i] = cursorPos + header.blockSize[i];
        if (blockEnd[i] > bytes.size()) {
            outError = "NIF block size table overruns the file";
            return false;
        }
        cursorPos = blockEnd[i];
    }

    std::vector<AvObjectFields> nodeFields(numBlocks);
    std::vector<bool> isNiNode(numBlocks, false);
    std::vector<bool> isTriShape(numBlocks, false);
    std::vector<GeometryBlock> geometry(numBlocks);
    std::vector<TextureSetBlock> textureSets(numBlocks);
    // Per shader-property block, the texture set it names (-1 when absent).
    std::vector<std::int32_t> shaderTextureSetRefs(numBlocks, -1);
    // Diffuse paths reached through the older NiTexturingProperty ->
    // NiSourceTexture chain instead of BSShaderTextureSet.
    std::vector<std::string> sourceTexturePaths(numBlocks);
    std::vector<std::string> texturingPropertyPaths(numBlocks);
    std::vector<AlphaPropertyBlock> alphaProperties(numBlocks);
    std::unordered_set<std::int32_t> referencedAsChild;

    for (std::size_t i = 0; i < numBlocks; ++i) {
        const std::string& typeName = header.blockTypeNames[header.blockTypeIndex[i]];
        ByteCursor blockCursor(bytes.data() + blockStart[i], header.blockSize[i]);

        if (isNodeTypeName(typeName)) {
            AvObjectFields fields;
            if (readNiNode(blockCursor, header.userVersion2, fields)) {
                nodeFields[i] = std::move(fields);
                isNiNode[i] = true;
                for (const std::int32_t child : nodeFields[i].children) {
                    referencedAsChild.insert(child);
                }
            }
        } else if (looksLikeUnhandledNodeType(typeName)) {
            ++outModel.unhandledNodeTypeCount;
        } else if (typeName == "NiTriShape" || typeName == "NiTriStrips") {
            AvObjectFields fields;
            if (readNiTriBasedGeom(blockCursor, header.userVersion2, fields)) {
                nodeFields[i] = std::move(fields);
                isTriShape[i] = true;
            }
        } else if (typeName == "BSShaderPPLightingProperty") {
            std::int32_t textureSetRef = -1;
            if (readBsShaderTextureSetRef(blockCursor, textureSetRef)) {
                shaderTextureSetRefs[i] = textureSetRef;
            }
        } else if (typeName == "NiSourceTexture") {
            std::string fileName;
            if (findSourceTextureFileName(blockCursor, header.strings, fileName)) {
                sourceTexturePaths[i] = std::move(fileName);
            }
        } else if (typeName == "BSShaderTextureSet") {
            TextureSetBlock set;
            if (readBsShaderTextureSet(blockCursor, set)) {
                textureSets[i] = std::move(set);
            }
        } else if (typeName == "NiAlphaProperty") {
            AlphaPropertyBlock alpha;
            if (readNiAlphaProperty(blockCursor, header.userVersion2, alpha)) {
                alphaProperties[i] = alpha;
            }
        } else if (typeName == "NiTriShapeData" || typeName == "NiTriStripsData") {
            GeometryBlock block;
            const bool isStrips = (typeName == "NiTriStripsData");
            // Not counted here: a data block that fails to parse is counted
            // once, at the shape that references it in the DFS below. Counting
            // both sites reported 2 for a single lost shape.
            if (readGeometryData(blockCursor, header.blockSize[i], isStrips, block) && block.valid) {
                geometry[i] = std::move(block);
            }
        }
        // Every other block type is intentionally left unparsed: the next
        // block always starts at blockStart[i+1] regardless of what (if
        // anything) was read here.
    }

    // Second pass over NiTexturingProperty: a NiSourceTexture can appear after
    // the property that references it, so this cannot fold into the loop above.
    for (std::size_t i = 0; i < numBlocks; ++i) {
        if (header.blockTypeNames[header.blockTypeIndex[i]] != "NiTexturingProperty") {
            continue;
        }
        ByteCursor propertyCursor(bytes.data() + blockStart[i], header.blockSize[i]);
        std::string fileName;
        if (findTexturingPropertySource(propertyCursor, sourceTexturePaths, fileName)) {
            texturingPropertyPaths[i] = std::move(fileName);
        }
    }

    // DFS from every NiNode never referenced as a child (the file's own
    // root(s)), accumulating world transforms and emitting one NifShape per
    // reachable NiTriShape with valid geometry.
    std::vector<std::size_t> stack;
    std::vector<Mat4> transformStack;
    for (std::size_t i = 0; i < numBlocks; ++i) {
        if (isNiNode[i] && referencedAsChild.find(static_cast<std::int32_t>(i)) == referencedAsChild.end()) {
            stack.push_back(i);
            transformStack.push_back(Mat4{});
        }
    }

    std::vector<bool> visited(numBlocks, false);
    while (!stack.empty()) {
        const std::size_t blockIndex = stack.back();
        const Mat4 parentTransform = transformStack.back();
        stack.pop_back();
        transformStack.pop_back();

        if (blockIndex >= numBlocks || visited[blockIndex]) {
            continue;  // guards against malformed/cyclic child references
        }
        visited[blockIndex] = true;

        const Mat4 localTransform = makeTrs(
            nodeFields[blockIndex].translation, nodeFields[blockIndex].rotation, nodeFields[blockIndex].scale);
        const Mat4 worldTransform = multiply(parentTransform, localTransform);

        if (isNiNode[blockIndex]) {
            for (const std::int32_t child : nodeFields[blockIndex].children) {
                if (child >= 0 && static_cast<std::size_t>(child) < numBlocks) {
                    stack.push_back(static_cast<std::size_t>(child));
                    transformStack.push_back(worldTransform);
                }
            }
        } else if (isTriShape[blockIndex]) {
            const std::int32_t dataRef = nodeFields[blockIndex].dataRef;
            if (dataRef < 0 || static_cast<std::size_t>(dataRef) >= numBlocks || !geometry[dataRef].valid) {
                // Count it. This path used to drop the shape silently, which is
                // how every NiTriStrips mesh in the game — the majority of them
                // — went missing with no diagnostic at all.
                ++outModel.skippedShapeCount;
                continue;
            }
            const GeometryBlock& src = geometry[dataRef];
            NifShape shape;
            const std::int32_t nameRef = nodeFields[blockIndex].nameRef;
            if (nameRef >= 0 && static_cast<std::size_t>(nameRef) < header.strings.size()) {
                shape.name = header.strings[static_cast<std::size_t>(nameRef)];
            }
            // Resolve this shape's material properties. A shader property
            // (BSShaderPPLightingProperty / BSShaderNoLightingProperty) points
            // at a BSShaderTextureSet, but its field layout shifts with
            // userVersion2 across FO3/FNV patch levels. Rather than hard-code
            // an offset — the exact class of guess that produced the format
            // bugs already found in this file — scan the block's 4-byte-aligned
            // words for the first one that indexes a block actually parsed as
            // a BSShaderTextureSet. Wrong-by-construction is impossible: a
            // candidate is only accepted if it resolves to the right type.
            for (const std::int32_t propertyRef : nodeFields[blockIndex].properties) {
                if (propertyRef < 0 || static_cast<std::size_t>(propertyRef) >= numBlocks) {
                    continue;
                }
                const auto propertyIndex = static_cast<std::size_t>(propertyRef);
                if (alphaProperties[propertyIndex].valid) {
                    shape.alphaTest = shape.alphaTest || alphaProperties[propertyIndex].alphaTest;
                    continue;
                }
                if (!shape.diffuseTexturePath.empty()) {
                    continue;
                }
                if (!texturingPropertyPaths[propertyIndex].empty()) {
                    shape.diffuseTexturePath = texturingPropertyPaths[propertyIndex];
                    continue;
                }
                const std::int32_t textureSetRef = shaderTextureSetRefs[propertyIndex];
                if (textureSetRef < 0 || static_cast<std::size_t>(textureSetRef) >= numBlocks) {
                    // Record what this actually was. A property that is neither
                    // an alpha property nor something we got a texture set out
                    // of is exactly the case that leaves a shape untextured.
                    if (propertyIndex < header.blockTypeIndex.size()) {
                        const std::string& typeName =
                            header.blockTypeNames[header.blockTypeIndex[propertyIndex]];
                        if (std::find(
                                outModel.unresolvedPropertyTypes.begin(),
                                outModel.unresolvedPropertyTypes.end(),
                                typeName) == outModel.unresolvedPropertyTypes.end()) {
                            outModel.unresolvedPropertyTypes.push_back(typeName);
                        }
                    }
                    continue;
                }
                // Type-validate the resolved ref. The layout above is read, not
                // guessed, so this is an assertion rather than a search.
                const TextureSetBlock& set = textureSets[static_cast<std::size_t>(textureSetRef)];
                if (set.valid && !set.textures.empty() && !set.textures.front().empty()) {
                    shape.diffuseTexturePath = set.textures.front();
                }
            }

            shape.uvs = src.uvs;
            shape.positions.resize(src.positions.size());
            for (std::size_t v = 0; v * 3u < src.positions.size(); ++v) {
                const float local[3] = {src.positions[v * 3u], src.positions[(v * 3u) + 1], src.positions[(v * 3u) + 2]};
                const auto world = transformPoint(worldTransform, local);
                shape.positions[v * 3u] = world[0];
                shape.positions[(v * 3u) + 1] = world[1];
                shape.positions[(v * 3u) + 2] = world[2];
            }
            if (!src.normals.empty()) {
                shape.normals.resize(src.normals.size());
                for (std::size_t v = 0; v * 3u < src.normals.size(); ++v) {
                    const float local[3] = {src.normals[v * 3u], src.normals[(v * 3u) + 1], src.normals[(v * 3u) + 2]};
                    const auto world = transformDirection(worldTransform, local);
                    shape.normals[v * 3u] = world[0];
                    shape.normals[(v * 3u) + 1] = world[1];
                    shape.normals[(v * 3u) + 2] = world[2];
                }
            }
            shape.triangleIndices = src.triangleIndices;
            outModel.shapes.push_back(std::move(shape));
        }
    }

    return true;
}

bool loadNifStaticMesh(const std::filesystem::path& path, NifModel& outModel, std::string& outError) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        outError = "Failed to open NIF file: " + path.string();
        return false;
    }
    const auto fileSize = static_cast<std::size_t>(input.tellg());
    input.seekg(0);
    std::vector<std::uint8_t> bytes(fileSize);
    if (fileSize != 0 && !input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(fileSize))) {
        outError = "Failed to read NIF file: " + path.string();
        return false;
    }
    return parseNifStaticMesh(bytes, outModel, outError);
}

}  // namespace odai::importer::fnv
