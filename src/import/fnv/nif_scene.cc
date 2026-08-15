#include "import/fnv/nif_scene.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <unordered_map>
#include <unordered_set>

namespace odai::importer::fnv {

namespace {

constexpr std::uint32_t kSupportedNifVersion = 0x14020007u;  // 20.2.0.7 (FO3/FNV/Skyrim base)
// Oblivion. 20.0.0.5 exists in the retail archives too but is 1.2% of them.
constexpr std::uint32_t kOblivionNifVersion = 0x14000004u;   // 20.0.0.4
constexpr std::uint32_t kOblivionNifVersion5 = 0x14000005u;  // 20.0.0.5
// Morrowind, and a THIRD structural generation rather than an older version of
// the second. See NifHeader::inlineBlockTypes and ::wideBools.
constexpr std::uint32_t kMorrowindNifVersion = 0x04000002u;  // 4.0.0.2
// Oblivion ships a MIX, and the older half of it is this one. 580 of the game's
// 9612 meshes are 10.1.x/10.2.0.0 rather than 20.0.0.x, including
// icpalacetower01.nif -- the White-Gold Tower, the tallest thing in Cyrodiil.
//
// Structurally this generation is 20.0.0.4 MINUS THE ENDIAN BYTE, and nothing
// else. Verified by hexdump rather than from the spec, because the ordering
// question is exactly the kind that is easy to misremember:
//
//   20.0.0.4  ... 0a | 04 00 00 14 | 01 | 0b000000 | 5d000000 | 0b000000 | 0a "mcarofano"
//   10.2.0.0  ... 0a | 00 00 02 0a |    | 0a000000 | 68000000 | 06000000 | 0a "mcarofano"
//                 ^nl   ^version    ^end  ^userVer   ^numBlocks ^userVer2  ^export strings
//
// Endianness arrived at 20.0.0.4, so every field below it shifts back one byte
// and everything after the export strings -- block-type table, per-block type
// indices, sequential block bodies with no size table -- is byte-identical.
//
// 10.0.1.0 and 10.0.1.2 (64 meshes) are NOT included: User Version and the
// export-info strings both arrive at 10.0.1.8, so those two have a genuinely
// different header and would mis-parse silently rather than fail.
constexpr std::uint32_t kGamebryo10MinVersion = 0x0a010065u;  // 10.1.0.101
constexpr std::uint32_t kGamebryo10MaxVersion = 0x0a020000u;  // 10.2.0.0
// TexDesc carried two PlayStation 2 fields up to and including this version.
constexpr std::uint32_t kPs2TexDescMaxVersion = 0x0a040001u;  // 10.4.0.1
// NiGeometryData grew a Group ID at 10.1.0.114 and an Additional Data ref at
// 20.0.0.4 -- one at each END of the same block.
constexpr std::uint32_t kGroupIdMinVersion = 0x0a010072u;         // 10.1.0.114
constexpr std::uint32_t kAdditionalDataMinVersion = 0x14000004u;  // 20.0.0.4

// The endian byte arrived at 20.0.0.4. Below it the field does not exist, and
// reading one anyway consumes the low byte of User Version.
constexpr bool nifHasEndianByte(std::uint32_t version) { return version >= 0x14000004u; }
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

    // Bounds-checked forward skip. Unlike seekAbsolute this REFUSES to run past
    // the end rather than clamping, because it is used by the sequential
    // walkers, where clamping would turn a desync into a silently short block.
    bool skip(std::size_t count) {
        if (m_pos + count > m_size) {
            return false;
        }
        m_pos += count;
        return true;
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
// The list below is not hand-curated any more: it is EVERY type that inherits
// from NiNode, transitively, in niftools' nif.xml -- the format's own schema --
// rather than the twelve someone happened to hit. Nine of them do NOT end in
// "Node" (BSMasterParticleSystem, NiBone, NiRoom, NiRoomGroup, NiWall,
// FxButton, FxRadioButton, FxWidget, NiCollisionSwitch), which is why matching
// on the name suffix alone was never going to be enough: BSMasterParticleSystem
// is the ROOT of 38 retail meshes and no suffix rule sees it.
//
// Deliberately NOT here: NiCamera. It inherits NiAVObject, not NiNode, so it has
// no children list -- and nif.xml's Footer note says a camera is referenced in
// the root list "even if it is not a root object", so it does turn up where
// roots are read. Walking it as a node would read its projection fields as a
// child count.
bool isNodeTypeName(std::string_view typeName) {
    // Sorted; keep it that way for the binary search.
    static constexpr std::string_view kNodeTypes[] = {
        "AvoidNode",         "BSBlastNode",   "BSDamageStage",
        "BSDebrisNode",      "BSDistantObjectInstancedNode",
        "BSFadeNode",        "BSLeafAnimNode", "BSMasterParticleSystem",
        "BSMultiBoundNode",  "BSOrderedNode", "BSRangeNode",
        "BSTreeNode",        "BSValueNode",   "CsNiNode",
        "FxButton",          "FxRadioButton", "FxWidget",
        "JPSJigsawNode",     "NiBSAnimationNode", "NiBSParticleNode",
        "NiBillboardNode",   "NiBone",        "NiCollisionSwitch",
        "NiLODNode",         "NiNode",        "NiRoom",
        "NiRoomGroup",       "NiSortAdjustNode", "NiSwitchNode",
        "NiWall",            "RootCollisionNode",
    };
    return std::binary_search(std::begin(kNodeTypes), std::end(kNodeTypes), typeName);
}

// A type this parser does not know, whose name looks like a node, is the exact
// shape of the bug above: it will silently drop or misplace whatever hangs off
// it. Worth surfacing rather than absorbing — mods and DLC use node types the
// base game happens not to.
bool looksLikeUnhandledNodeType(std::string_view typeName) {
    return !isNodeTypeName(typeName) && typeName.size() > 4u &&
        typeName.substr(typeName.size() - 4u) == "Node";
}

// The file's own root list, which this parser used to ignore.
//
// nif.xml's Footer struct is "Num Roots" (uint) then that many Ref<NiObject>,
// sitting immediately after the last block. Reading it replaces the heuristic
// that produced the floating-geometry bug: promoting every node no other node
// claimed as a child, each walked with an IDENTITY transform, so one
// unrecognized parent relocated its whole subtree to the model origin.
//
// Measured over 20000 retail meshes before anything was made to depend on it:
// the footer is present and exactly 4+4*numRoots bytes on 19999 of 19999, every
// ref in range, 19998 files with one root and one with two. Multi-root files
// existing is why "just take block 0" would have been wrong.
//
// Returns false when there is no plausible footer, leaving the caller to fall
// back. Roots are NOT filtered to nodes here -- nif.xml notes a camera is listed
// "even if it is not a root object", so the caller skips what it cannot walk.
bool readFooterRoots(
    const std::vector<std::uint8_t>& bytes,
    const std::vector<std::size_t>& blockEnd,
    std::vector<std::int32_t>& outRoots
) {
    outRoots.clear();
    if (blockEnd.empty()) {
        return false;
    }
    const std::size_t footerOffset = blockEnd.back();
    if (footerOffset + sizeof(std::uint32_t) > bytes.size()) {
        return false;
    }
    std::uint32_t rootCount = 0;
    std::memcpy(&rootCount, bytes.data() + footerOffset, sizeof(rootCount));
    // A count that does not fit the bytes that are actually there means this is
    // not the footer -- a desynchronized block-size table, most likely.
    const std::size_t needed =
        sizeof(std::uint32_t) + (static_cast<std::size_t>(rootCount) * sizeof(std::int32_t));
    if (rootCount == 0u || footerOffset + needed > bytes.size()) {
        return false;
    }
    outRoots.resize(rootCount);
    for (std::uint32_t i = 0; i < rootCount; ++i) {
        std::memcpy(
            &outRoots[i],
            bytes.data() + footerOffset + sizeof(std::uint32_t) + (i * sizeof(std::int32_t)),
            sizeof(std::int32_t));
    }
    return true;
}

struct NifHeader {
    // TWO STRUCTURALLY DIFFERENT HEADERS, and the differences are not cosmetic.
    //
    // 20.0.0.4 (Oblivion) has no block-SIZE table and no global STRING table.
    // Losing the size table means blocks can only be walked strictly
    // sequentially -- every type in the file must be consumed exactly, or the
    // reader desyncs from the first unknown block onward. Losing the string
    // table means every name is stored inline in its own block as a
    // uint32-length-prefixed string rather than as an index, which changes the
    // FIRST FIELD of nearly every block.
    bool sequentialBlocks = false;
    bool inlineNames = false;
    // 4.0.0.2 has no block-TYPE table either: each block is preceded by its own
    // uint32-length-prefixed type name. That is actually easier than Oblivion's
    // arrangement -- the type is always known before the block is read -- but it
    // means the type table has to be synthesized during the walk rather than
    // read up front.
    bool inlineBlockTypes = false;
    // AND EVERY BOOLEAN IS FOUR BYTES. Below 4.1.0.0 the format stores bools as
    // int32; from 4.1.0.0 they are int8. This is the single most common way a
    // 4.0.0.2 reader desyncs, because hasVertices/hasNormals/hasVertexColors/
    // hasUV sit between the arrays they gate -- read one as a byte and the next
    // three fields are garbage while the parse still "succeeds".
    bool wideBools = false;
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
    if (!cursor.readLine(magicLine, '\n') ||
        (magicLine.rfind("Gamebryo File Format, Version ", 0) != 0 &&
         magicLine.rfind("NetImmerse File Format, Version ", 0) != 0)) {
        outError = "Not a Gamebryo/NetImmerse NIF file (unrecognized header line)";
        return false;
    }
    if (!cursor.read(header.version)) {
        outError = "Truncated NIF version";
        return false;
    }
    if (header.version == kMorrowindNifVersion) {
        header.sequentialBlocks = true;
        header.inlineNames = true;
        header.inlineBlockTypes = true;
        header.wideBools = true;
    } else if (header.version == kOblivionNifVersion || header.version == kOblivionNifVersion5 ||
               (header.version >= kGamebryo10MinVersion &&
                header.version <= kGamebryo10MaxVersion)) {
        header.sequentialBlocks = true;
        header.inlineNames = true;
    } else if (header.version != kSupportedNifVersion) {
        // Name the version that was actually found. Listing only the SUPPORTED
        // ones says nothing about which unsupported generation a file belongs
        // to, so a whole class of failures reads as one undifferentiated
        // "unsupported" -- and the header line carries the authoring tool's own
        // spelling of it, which is what a NIF reference is indexed by.
        char found[64];
        std::snprintf(found, sizeof(found), "%u.%u.%u.%u", (header.version >> 24) & 0xFFu,
                      (header.version >> 16) & 0xFFu, (header.version >> 8) & 0xFFu,
                      header.version & 0xFFu);
        outError = "Unsupported NIF version " + std::string(found) + " (\"" + magicLine +
                   "\"); 4.0.0.2, 20.0.0.4, 20.0.0.5 and 20.2.0.7 are supported";
        return false;
    }
    // 4.0.0.2 predates the endianness byte, the user version and the export
    // strings alike: the version word is followed immediately by the block
    // count and then by the first block.
    if (header.inlineBlockTypes) {
        std::uint32_t morrowindBlockCount = 0;
        if (!cursor.read(morrowindBlockCount)) {
            outError = "Truncated NIF header";
            return false;
        }
        header.blockTypeIndex.assign(morrowindBlockCount, 0u);
        header.blockSize.assign(morrowindBlockCount, 0u);
        return true;
    }
    if (nifHasEndianByte(header.version)) {
        std::uint8_t endianType = 0;
        if (!cursor.read(endianType) || endianType != 1u) {
            outError = "Unsupported NIF endianness (only little-endian archives are supported)";
            return false;
        }
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

    // Both tables arrived AFTER 20.0.0.4 -- the size table at 20.2.0.5 and the
    // string table at 20.1.0.x -- so reading them there consumes the first
    // block's data as if it were sizes. The give-away when that happens is a
    // size table whose entries are the ASCII of the block's own name.
    if (!header.sequentialBlocks) {
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
    } else {
        header.blockSize.assign(numBlocks, 0u);
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

// Determinant of a row-major Mat4's upper-left 3x3. Negative means the
// transform includes a reflection.
float determinant3x3(const Mat4& mat) {
    const float a = mat.m[0], b = mat.m[1], c = mat.m[2];
    const float d = mat.m[4], e = mat.m[5], f = mat.m[6];
    const float g = mat.m[8], h = mat.m[9], i = mat.m[10];
    return (a * ((e * i) - (f * h))) - (b * ((d * i) - (f * g))) + (c * ((d * h) - (e * g)));
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
bool consumeMorrowindBoundingVolume(ByteCursor& cursor);

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
    bool alphaBlend = false;
    std::uint8_t alphaThreshold = 128;
    // AlphaFlags bits 10-12. Read for census only: the renderer's discard is
    // hardcoded to GREATER, so a shape declaring anything else is shaded wrong.
    std::uint8_t alphaTestFunction = 4;  // TEST_GREATER
    bool valid = false;
};

// NiStencilProperty. Only the draw mode is wanted: Fallout marks its thin alpha
// geometry -- window glass, foliage cards, awnings -- DRAW_BOTH, and rendering
// those single-sided loses whichever face happens to point away.
struct StencilPropertyBlock {
    bool valid = false;
    bool twoSided = false;
    // DRAW_CW: front face is the clockwise winding, the reverse of this
    // renderer's convention.
    bool reversedWinding = false;
    std::uint16_t drawMode = 0;
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
//
// Leaves the cursor positioned immediately after textureClampMode, i.e. at the
// first field the concrete subclass adds. BSShaderPPLightingProperty puts a
// texture-set ref there; BSShaderNoLightingProperty puts a sized string.
bool readBsShaderLightingPrefix(ByteCursor& cursor) {
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
    return cursor.read(controllerRef) && cursor.read(flags) && cursor.read(shaderType) &&
        cursor.read(shaderFlags) && cursor.read(shaderFlags2) && cursor.read(envMapScale) &&
        cursor.read(textureClampMode);
}

bool readBsShaderTextureSetRef(ByteCursor& cursor, std::int32_t& outTextureSetRef) {
    return readBsShaderLightingPrefix(cursor) && cursor.read(outTextureSetRef);
}

// BSShaderNoLightingProperty names its texture DIRECTLY, as a length-prefixed
// string, instead of pointing at a BSShaderTextureSet.
//
// READ THIS BEFORE ASSUMING IT MATTERS: it recovers ZERO textures on retail
// data. Measured over all 20746 NIFs in the retail archives, before and after
// this reader existed, `--nifs 30000` reports the same 63404 shapes with a
// diffuse and the same 4569 without. Every shape carrying one of these already
// gets its diffuse from another property on the same shape -- almost always a
// NiTexturingProperty resolved through NiSourceTexture, which is why adding
// that reader is what actually fixed the untextured effect meshes.
//
// The comment that used to sit here claimed 44 of 800 shapes in a Mojave cook
// lost their texture to this type. That was measured before the
// NiTexturingProperty path existed and has not been true since. Checked
// directly on meshes\effects\nv\sanddust\sanddust02.nif: both sStorm shapes
// resolve textures\effects\nv\dustStorm.dds with this reader deleted.
//
// What it does buy is a truthful diagnostic. unresolvedPropertyTypes is the
// list used to decide what to implement next, and this type was 3539 of its
// entries -- the largest by far, and every one of them a false lead. With the
// reader in, that drops to 2508, and those remaining are blocks whose fileName
// is legitimately the empty string (editor markers and similar), not holes.
// The types actually still costing textures are visible underneath it now:
// TallGrassShaderProperty (18), WaterShaderProperty (8), SkyShaderProperty (6),
// TileShaderProperty (5).
//
// Because it changes no geometry and no texture, it does NOT need a
// kCellBuildVersion bump -- invalidating every cached cell for a diagnostic
// would cost a full re-cook and buy nothing.
//
// It derives from BSShaderLightingProperty, so it shares the prefix walked by
// readBsShaderLightingPrefix above and puts a sized string exactly where
// BSShaderPPLightingProperty puts its texture-set ref -- fileName length at
// block-relative offset 34, characters at 38, then four falloff floats.
//
// This IS a computed offset, and two earlier attempts at one landed wrong, so
// it was verified against three retail blocks in
// meshes\effects\nv\sanddust\sanddust02.nif before being written. The block
// sizes close exactly on the layout, which a wrong offset cannot do:
//
//   block [50]  87 bytes = 38 + 33 (fileName) + 16 (4 falloff floats)
//   block [63]  54 bytes = 38 +  0 (empty)    + 16
//   block [70]  86 bytes = 38 + 32            + 16
//
// and envMapScale reads exactly 1.0f at offset 26 in all three. A scan was the
// alternative and is the weaker option here: these blocks begin with a nameRef
// into the header string table, so a scan accepting "the first word that
// resolves to a .dds path" returns the block's NAME on any file whose strings
// happen to include texture paths. The offset closes on arithmetic; the scan
// cannot be checked at all.
//
// NiTexturingProperty is a different problem and keeps the scan below. It
// reaches its texture through NiSourceTexture blocks rather than a texture set
// -- the older Gamebryo path, still used by retail FNV on cliffs and rock
// formations -- and those blocks put a ushort and a byte ahead of their refs,
// so the fields are genuinely misaligned rather than at a stable offset.

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
    // .tga and .bmp are Morrowind's spelling for what ships as .dds -- see
    // normalizeTexturePath in asset_source.cc, which rewrites the extension.
    // Accepting them here is what lets the same scan find a NetImmerse mesh's
    // texture; rejecting them made every Morrowind shape come out untextured.
    return lowered.compare(lowered.size() - 4u, 4u, ".dds") == 0 ||
           lowered.compare(lowered.size() - 4u, 4u, ".tga") == 0 ||
           lowered.compare(lowered.size() - 4u, 4u, ".bmp") == 0;
}

// BSShaderNoLightingProperty's own field: a sized string holding the texture
// path, immediately after the shared BSShaderLightingProperty prefix. Empty
// strings are real (block [63] above) and are reported as "no texture" rather
// than as a parse failure.
bool readBsShaderNoLightingTexture(ByteCursor& cursor, std::string& outFileName) {
    if (!readBsShaderLightingPrefix(cursor)) {
        return false;
    }
    std::string fileName;
    if (!cursor.readSizedString<std::uint32_t>(fileName)) {
        return false;
    }
    if (!looksLikeDdsPath(fileName)) {
        return false;
    }
    outFileName = std::move(fileName);
    return true;
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

// The NiObjectNET prefix every NiProperty derives from, in all three shapes it
// takes. This used to be open-coded as the Fallout spelling only, which read an
// inline name's LENGTH as a name index and its bytes as the extra-data count --
// so every Oblivion and Morrowind alpha property was misparsed, and misparsed
// silently: the reader returns false, the shape keeps its default of "opaque",
// and an alpha-tested leaf renders as a solid green slab.
bool readNiObjectNetPrefix(ByteCursor& cursor, bool inlineNames, bool morrowind) {
    if (inlineNames) {
        std::string name;
        if (!cursor.readSizedString<std::uint32_t>(name)) {
            return false;
        }
    } else {
        std::int32_t nameRef = 0;
        if (!cursor.read(nameRef)) {
            return false;
        }
    }
    if (morrowind) {
        // ONE extra-data ref, the head of a linked list, rather than a count
        // and a list. The list form arrives at 10.0.1.0.
        return cursor.skip(4u) && cursor.skip(4u);
    }
    std::uint32_t numExtraData = 0;
    if (!cursor.read(numExtraData) || numExtraData > 1024u) {
        return false;
    }
    if (!cursor.skip(static_cast<std::size_t>(numExtraData) * 4u)) {
        return false;
    }
    return cursor.skip(4u);  // controller ref
}

bool readNiAlphaProperty(
    ByteCursor& cursor, std::uint32_t userVersion2, bool inlineNames, bool morrowind,
    AlphaPropertyBlock& out) {
    if (!readNiObjectNetPrefix(cursor, inlineNames, morrowind)) {
        return false;
    }
    (void)userVersion2;  // NiProperty flags stay 16-bit regardless
    std::uint16_t flags = 0;
    if (!cursor.read(flags)) {
        return false;
    }
    // NiAlphaProperty flags: bit 0 enables alpha BLENDING, bit 9 enables alpha
    // TESTING. Only the test bit used to be read, so a blended shape was
    // indistinguishable from an opaque one -- and the imported static path
    // could not blend, so glass panes and effect billboards drew as solid slabs.
    constexpr std::uint16_t kAlphaBlendBit = 0x0001u;
    constexpr std::uint16_t kAlphaTestBit = 0x0200u;
    const bool blendBit = (flags & kAlphaBlendBit) != 0u;
    out.alphaTest = (flags & kAlphaTestBit) != 0u;
    // nif.xml AlphaFlags: Test Func is bits 10-12, default TEST_GREATER.
    out.alphaTestFunction = static_cast<std::uint8_t>((flags >> 10) & 0x7u);

    // A surface that alpha-TESTS is a cutout, even when the blend bit is also
    // set -- and blend+test together is the single most common combination
    // Fallout ships. Retail Goodsprings:
    //
    //   0x12ec  blend=0 test=1   394 shapes   cutout
    //   0x12ed  blend=1 test=1   351 shapes   cutout, with blending also on
    //   0x10ed  blend=1 test=0    36 shapes   genuine transparency
    //   0x1242  blend=0 test=1    17 shapes   cutout
    //   0x1043  blend=1 test=0     1 shape    genuine transparency
    //
    // The test is what defines a cutout's silhouette; on those surfaces the
    // diffuse alpha channel is not an opacity ramp at all -- Bethesda's shaders
    // read it as a specular mask, so a building wall's alpha sits around 0.5
    // across the whole texture. Honouring the blend bit there drew 351 shapes
    // at half opacity and you could see straight through Goodsprings' walls.
    //
    // So: blended means blend WITHOUT test. That leaves 37 truly transparent
    // shapes in the region, which is about how much glass is actually there.
    out.alphaBlend = blendBit && !out.alphaTest;

    // The threshold the test compares against, one byte after the flags.
    // Retail Goodsprings runs 0, 16, 32, 50, 60, 63, 64, 70, 80, 90, 92, 100,
    // 120, 124, 127, 128, 160 and 200, with 100 by far the most common -- so
    // the 0.5 every renderer hardcodes is wrong more often than right, and
    // being too high erodes thin cutouts (foliage, chain-link, grating).
    //
    // A truncated read keeps the neutral 128 rather than failing the property:
    // losing the shape's alpha mode entirely would be a much worse outcome
    // than shading it at the default threshold.
    std::uint8_t threshold = 128;
    if (cursor.read(threshold)) {
        out.alphaThreshold = threshold;
    }
    out.valid = true;
    return true;
}

bool readNiStencilProperty(
    ByteCursor& cursor, bool inlineNames, StencilPropertyBlock& out) {
    // Same NiObjectNET prefix as NiAlphaProperty above: name, extra data,
    // controller, then the property's own 16-bit flags.
    std::uint32_t nameRef = 0;
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
    if (!cursor.read(controllerRef)) {
        return false;
    }
    std::uint16_t flags = 0;
    if (!cursor.read(flags)) {
        return false;
    }
    // nif.xml's StencilFlags bitfield, for ver >= 20.1.0.3 (which Fallout's
    // 20.2.0.7 is): stencil enabled at bit 0 (width 1), fail action at 1 (3),
    // z-fail action at 4 (3), pass action at 7 (3), DRAW MODE AT 10 (width 2),
    // stencil function at 12 (3).
    //
    // Every NiStencilProperty in the retail Goodsprings set reads 0x4d80, which
    // decodes as enabled=0, fail=KEEP, zfail=KEEP, pass=INCREMENT, drawMode=3,
    // function=4. Draw mode 3 is DRAW_BOTH -- these blocks exist precisely to
    // say "two-sided", which is the whole reason to read them.
    //
    // Reading the draw mode at bit 13 instead yields 2 for that same word, i.e.
    // two-sided-never, silently and for every file. Do not adjust these
    // positions without re-dumping the flags against real data.
    constexpr std::uint16_t kDrawModeShift = 10u;
    constexpr std::uint16_t kDrawModeMask = 0x3u;
    constexpr std::uint16_t kDrawBoth = 3u;
    const std::uint16_t drawMode = (flags >> kDrawModeShift) & kDrawModeMask;
    out.twoSided = drawMode == kDrawBoth;
    // DRAW_CW (2) means the shape's front face is the CLOCKWISE winding, i.e.
    // the opposite of DRAW_CCW_OR_BOTH (0) / DRAW_CCW (1). Only twoSided used
    // to be read, so a DRAW_CW shape kept the default convention, the
    // back-face-culling pipeline culled its front faces, and the mesh looked
    // solid from one side and see-through from the other.
    constexpr std::uint16_t kDrawCw = 2u;
    out.reversedWinding = drawMode == kDrawCw;
    out.drawMode = drawMode;
    out.valid = true;
    return true;
}

// Reads the NiObjectNET + NiAVObject fields common to NiNode and
// NiTriShape/NiTriStrips. Only reads through `scale` + property/collision
// refs — callers resume via the block's declared size, not by continuing
// this cursor, so anything after this point (children, data ref, ...) is
// read separately per block type. Returns false only on truncation this
// early in the block, which should not happen for a well-formed file.
bool readAvObjectPrefix(
    ByteCursor& cursor, std::uint32_t userVersion2, bool inlineNames, bool morrowind,
    AvObjectFields& out) {
    // 4.0.0.2's NiAVObject differs in four places, and none of them is the
    // header: extra data is ONE ref rather than a counted list, velocity sits
    // between scale and the property list, the bounds flag is a four-byte bool,
    // and there is no collision-object ref at all.
    if (morrowind) {
        std::string name;
        if (!cursor.readSizedString<std::uint32_t>(name)) {
            return false;
        }
        out.nameRef = -1;
        if (!cursor.skip(4u) || !cursor.skip(4u)) {  // extra-data ref, controller ref
            return false;
        }
        std::uint16_t flags16 = 0;
        if (!cursor.read(flags16) || !cursor.read(out.translation) ||
            !cursor.read(out.rotation) || !cursor.read(out.scale) || !cursor.skip(12u)) {
            return false;
        }
        std::uint32_t numProperties = 0;
        if (!cursor.read(numProperties)) {
            return false;
        }
        const std::size_t remaining = cursor.size() - cursor.pos();
        if (static_cast<std::size_t>(numProperties) * sizeof(std::int32_t) > remaining) {
            return false;
        }
        for (std::uint32_t i = 0; i < numProperties; ++i) {
            std::int32_t ref = 0;
            if (!cursor.read(ref)) {
                return false;
            }
            out.properties.push_back(ref);
        }
        std::uint32_t hasBounds = 0;
        if (!cursor.read(hasBounds)) {
            return false;
        }
        return hasBounds == 0u || consumeMorrowindBoundingVolume(cursor);
    }
    // 20.0.0.4 has no string table, so the name is stored inline here rather
    // than as an index into one. This is the single most invasive difference
    // between the two header families: it changes the FIRST field of nearly
    // every block, so getting it wrong desynchronizes everything after it.
    if (inlineNames) {
        std::string name;
        if (!cursor.readSizedString<std::uint32_t>(name)) {
            return false;
        }
        out.nameRef = -1;
    } else if (!cursor.read(out.nameRef)) {
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

// Returns false when the block does not look like an NiNode, which is now a
// LOAD-BEARING answer rather than a rare truncation report: the caller uses it
// to decide whether a block is a node at all, in place of matching its type
// NAME. Measured justification for that move, from --footers over 20000 retail
// meshes: BSMasterParticleSystem is NiNode-derived and is the ROOT of 38
// models, yet it neither appears in any allowlist nor ends in "Node", so no
// name-based rule recognizes it -- while NiCamera is a root that is NOT a node
// and must be rejected. The name simply does not carry the information; the
// layout does.
//
// The checks are deliberately STRUCTURAL rather than numeric. A desync shows up
// as a child count that cannot fit in the block long before it shows up as an
// implausible float, and a numeric tolerance risks rejecting a legitimate node
// -- which costs a whole model, the exact outcome this file is trying to avoid.
bool readNiNode(
    ByteCursor& cursor, std::uint32_t userVersion2, bool inlineNames, bool morrowind,
    AvObjectFields& out) {
    if (!readAvObjectPrefix(cursor, userVersion2, inlineNames, morrowind, out)) {
        return false;
    }
    std::uint32_t numChildren = 0;
    if (!cursor.read(numChildren)) {
        return false;
    }
    // Bound the count against what is actually left in the block BEFORE
    // resizing. Unbounded, a desynchronized read of 0xFFFFFFFF here asks for a
    // ~17 GB allocation and takes the process down, on nothing worse than a
    // malformed mod asset -- the vector was previously sized from this value
    // and only then discovered to be unreadable, one element at a time.
    const std::size_t remaining = cursor.size() - cursor.pos();
    if (static_cast<std::size_t>(numChildren) * sizeof(std::int32_t) > remaining) {
        return false;
    }
    out.children.resize(numChildren);
    for (std::int32_t& child : out.children) {
        if (!cursor.read(child)) {
            return false;
        }
    }
    // The effects list follows and is unused -- the caller resumes at the
    // block's declared end regardless -- but its COUNT must still be there.
    // Requiring those 4 bytes is what gives this function teeth as a
    // recognizer: a non-node block that happened to survive the field walk
    // this far almost never lands with exactly enough room left.
    //
    // Note a `pos() > size()` check could never fire: the cursor spans exactly
    // this block, and reads refuse to run past its end (the same argument the
    // geometry reader's own comment makes). A minimum-trailer check is the
    // form that actually asserts something here.
    if ((cursor.size() - cursor.pos()) < sizeof(std::uint32_t)) {
        return false;
    }
    return true;
}

bool readNiTriBasedGeom(
    ByteCursor& cursor, std::uint32_t userVersion2, bool inlineNames, bool morrowind,
    AvObjectFields& out) {
    if (!readAvObjectPrefix(cursor, userVersion2, inlineNames, morrowind, out)) {
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
    std::uint32_t outOfRangeTriangles = 0;
    std::uint32_t degenerateTriangles = 0;
    bool valid = false;
};

// Drops triangles that name a vertex this block does not have, and triangles
// with two equal indices.
//
// The bounds check is the load-bearing one, and it has to happen HERE rather
// than downstream: by the time shapes are merged into one vertex buffer an
// out-of-range index is indistinguishable from a valid one pointing at a
// neighbour's vertex, and the result is a triangle stretched between two
// unrelated meshes. Neither reader validated this -- the strip path could not,
// because it does not know the vertex count until the prefix has been read, and
// the explicit-list path simply never did.
//
// Dropping the individual triangle rather than failing the shape is deliberate:
// one bad index in a rock should cost one triangle, not the rock.
void rejectUnusableTriangles(GeometryBlock& out) {
    const std::size_t vertexCount = out.positions.size() / 3u;
    std::vector<std::uint32_t> kept;
    kept.reserve(out.triangleIndices.size());
    for (std::size_t i = 0; i + 2u < out.triangleIndices.size(); i += 3u) {
        const std::uint32_t a = out.triangleIndices[i];
        const std::uint32_t b = out.triangleIndices[i + 1u];
        const std::uint32_t c = out.triangleIndices[i + 2u];
        if (a >= vertexCount || b >= vertexCount || c >= vertexCount) {
            ++out.outOfRangeTriangles;
            continue;
        }
        if (a == b || b == c || a == c) {
            ++out.degenerateTriangles;
            continue;
        }
        kept.push_back(a);
        kept.push_back(b);
        kept.push_back(c);
    }
    // A trailing partial triangle is not a triangle; the loop above already
    // leaves it out, and keeping it would hand the renderer two thirds of one.
    out.triangleIndices = std::move(kept);
}

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
bool readNiTriBasedGeomDataPrefix(ByteCursor& cursor, GeometryBlock& out,
                                  std::uint16_t& outNumTriangles, std::uint32_t version) {
    std::int32_t groupId = 0;
    if (version >= kGroupIdMinVersion && !cursor.read(groupId)) {
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
    if (version >= kAdditionalDataMinVersion && !cursor.read(additionalDataRef)) {
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
bool consumeMorrowindGeometryData(ByteCursor& cursor, bool strips, GeometryBlock* out);

bool readGeometryData(ByteCursor& cursor, std::size_t blockEnd, bool isStrips, bool morrowind,
                      std::uint32_t version, GeometryBlock& out) {
    // 4.0.0.2's data block shares no field order with the later ones -- four-byte
    // bools, float vertex colours, the UV count read after the colours. It has
    // its own reader, which is the same code the sequential walker measures with.
    if (morrowind) {
        return consumeMorrowindGeometryData(cursor, isStrips, &out);
    }
    std::uint16_t numTriangles = 0;
    if (!readNiTriBasedGeomDataPrefix(cursor, out, numTriangles, version)) {
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

    // After the size check, not before: a block that failed its own layout
    // check is discarded whole, and counting its garbage triangles as rejects
    // would report a parse failure as a data defect.
    rejectUnusableTriangles(out);

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

// ---------------------------------------------------------------------------
// Sequential block walk for 20.0.0.4 (Oblivion)
// ---------------------------------------------------------------------------
//
// With no block-size table, offsets can only be derived by consuming every
// block in order -- so a reader must know the exact length of every type in the
// file, including the ~12 Havok types it will never look at. Those are pure
// skippers: the bytes have to be counted, not understood.
//
// Every layout below is resolved from nif.xml for version 20.0.0.4 / BSVER 11
// (see src/tools/oblivion_nif_lab/from_nifxml.py) and then confirmed against all
// 7382 retail 20.0.0.x meshes by walking each file and requiring it to land
// exactly on its footer. 82.1% of the archive walks, and 94% of architecture.
//
// A type with no entry here fails the FILE rather than being guessed at. That is
// deliberate: a wrong length does not produce a wrong mesh, it desynchronizes
// every subsequent block, and silently importing the resulting garbage is far
// worse than skipping the file.

bool consumeSizedString(ByteCursor& cursor) {
    std::string unused;
    return cursor.readSizedString<std::uint32_t>(unused);
}

bool consumeSkip(ByteCursor& cursor, std::size_t bytes) {
    if (cursor.pos() + bytes > cursor.size()) {
        return false;
    }
    cursor.seekAbsolute(cursor.pos() + bytes);
    return true;
}

// Reads a u32 count and skips count * stride bytes. The count is bounds-checked
// against the file rather than trusted: a desync upstream turns arbitrary bytes
// into a count, and a 4-billion-element array would otherwise be "read".
bool consumeCountedArray(ByteCursor& cursor, std::size_t stride) {
    std::uint32_t count = 0;
    if (!cursor.read(count)) {
        return false;
    }
    const std::size_t bytes = static_cast<std::size_t>(count) * stride;
    if (cursor.pos() + bytes > cursor.size()) {
        return false;
    }
    return consumeSkip(cursor, bytes);
}

// NiObjectNET: name + extra-data refs + controller ref.
bool consumeNiObjectNet(ByteCursor& cursor) {
    return consumeSizedString(cursor) && consumeCountedArray(cursor, 4u) &&
           consumeSkip(cursor, 4u);
}

// NiAVObject adds transform, properties and the collision-object ref. `flags` is
// a ushort here -- it widens to uint only past userVersion2 26, and Oblivion is 11.
bool consumeNiAvObject(ByteCursor& cursor) {
    return consumeNiObjectNet(cursor) &&
           consumeSkip(cursor, 2u + 12u + 36u + 4u) &&  // flags, translation, rotation, scale
           consumeCountedArray(cursor, 4u) &&           // properties
           consumeSkip(cursor, 4u);                     // collision object
}

// TexDesc: source + clamp + filter + uvSet, then an optional transform.
bool consumeTexDesc(ByteCursor& cursor, std::uint32_t version) {
    if (!consumeSkip(cursor, 4u + 4u + 4u + 4u)) {
        return false;
    }
    // PS2 L and PS2 K are a leftover of the PlayStation 2 renderer and were
    // dropped after 10.4.0.1. They sit in the MIDDLE of the descriptor, so
    // skipping them on a 10.x file does not merely lose four bytes -- every
    // texture slot after this one reads from the wrong offset, and the first
    // thing that notices is the block AFTER the texturing property.
    if (version <= kPs2TexDescMaxVersion && !consumeSkip(cursor, 2u + 2u)) {
        return false;
    }
    std::uint8_t hasTransform = 0;
    if (!cursor.read(hasTransform)) {
        return false;
    }
    return hasTransform == 0u || consumeSkip(cursor, 8u + 8u + 4u + 4u + 8u);
}

bool consumeNiGeometryData(ByteCursor& cursor, std::uint16_t& outNumVertices,
                           std::uint32_t version) {
    std::uint16_t numVertices = 0;
    std::uint8_t flag = 0;
    // Group ID arrives at 10.1.0.114, so 10.1.0.101/106 do not carry one.
    if (version >= kGroupIdMinVersion && !consumeSkip(cursor, 4u)) {
        return false;
    }
    if (!cursor.read(numVertices) ||
        !consumeSkip(cursor, 2u)) {  // numVertices, keep+compress flags
        return false;
    }
    outNumVertices = numVertices;
    const std::size_t vertexCount = static_cast<std::size_t>(numVertices);
    if (!cursor.read(flag)) {
        return false;
    }
    if (flag != 0u && !consumeSkip(cursor, vertexCount * 12u)) {
        return false;
    }
    std::uint16_t dataFlags = 0;
    if (!cursor.read(dataFlags) || !cursor.read(flag)) {
        return false;
    }
    if (flag != 0u) {
        if (!consumeSkip(cursor, vertexCount * 12u)) {
            return false;
        }
        if ((dataFlags & 0x1000u) != 0u &&
            !consumeSkip(cursor, vertexCount * 24u)) {  // tangents + bitangents
            return false;
        }
    }
    if (!consumeSkip(cursor, 16u) || !cursor.read(flag)) {  // bounding sphere
        return false;
    }
    if (flag != 0u && !consumeSkip(cursor, vertexCount * 16u)) {
        return false;
    }
    // UV SET COUNT IS THE LOW SIX BITS HERE. That is stock Gamebryo's
    // NiVectorFlags; Bethesda's BSVectorFlags reading, where bit 0 is a boolean,
    // is correct for FO3/FNV and wrong for Oblivion.
    const std::size_t uvSets = static_cast<std::size_t>(dataFlags & 0x3Fu);
    if (!consumeSkip(cursor, uvSets * vertexCount * 8u)) {
        return false;
    }
    // Consistency flags, then the Additional Data ref -- and THAT ref arrives at
    // 20.0.0.4. Consuming four bytes for it on a 10.x file eats the head of the
    // next block, and because these blocks carry no sizes the walk only notices
    // one block later: the White-Gold Tower reported a NiTriStrips it has always
    // been able to read, with NiTriStripsData as the actual culprit.
    if (!consumeSkip(cursor, 2u)) {
        return false;
    }
    return version < kAdditionalDataMinVersion || consumeSkip(cursor, 4u);
}

// Consumes exactly one block of `typeName`. False means "this reader does not
// know how long that block is", which fails the file.
bool consumeOblivionBlock(ByteCursor& cursor, std::string_view typeName,
                          std::uint32_t version) {
    // --- Havok. Skipped entirely; only the byte counts matter. ---
    if (typeName == "bhkCollisionObject" || typeName == "bhkBlendCollisionObject") {
        return consumeSkip(cursor, 10u);
    }
    if (typeName == "bhkRigidBody" || typeName == "bhkRigidBodyT") {
        // bhkRigidBodyT stores the same bytes; the T only changes whether the
        // translation and rotation are honoured.
        return consumeSkip(cursor, 228u) && consumeCountedArray(cursor, 4u) &&
               consumeSkip(cursor, 4u);
    }
    if (typeName == "bhkMoppBvTreeShape") {
        std::uint32_t moppSize = 0;
        return consumeSkip(cursor, 4u + 12u + 4u) && cursor.read(moppSize) &&
               consumeSkip(cursor, 16u) && consumeSkip(cursor, moppSize);
    }
    if (typeName == "bhkNiTriStripsShape") {
        return consumeSkip(cursor, 4u + 4u + 20u + 4u + 16u) &&
               consumeCountedArray(cursor, 4u) && consumeCountedArray(cursor, 4u);
    }
    if (typeName == "bhkConvexVerticesShape") {
        return consumeSkip(cursor, 4u + 4u + 12u + 12u) &&
               consumeCountedArray(cursor, 16u) && consumeCountedArray(cursor, 16u);
    }
    if (typeName == "bhkBoxShape") {
        return consumeSkip(cursor, 32u);
    }
    if (typeName == "bhkCapsuleShape") {
        return consumeSkip(cursor, 48u);
    }
    if (typeName == "bhkSphereShape") {
        return consumeSkip(cursor, 8u);
    }
    if (typeName == "bhkTransformShape" || typeName == "bhkConvexTransformShape") {
        return consumeSkip(cursor, 84u);
    }
    if (typeName == "bhkListShape") {
        return consumeCountedArray(cursor, 4u) && consumeSkip(cursor, 4u + 12u + 12u) &&
               consumeCountedArray(cursor, 4u);
    }

    // --- Extra data. These derive from NiObject, so name ONLY: no extra-data
    // list and no controller ref. Giving them NiObjectNET desyncs immediately. ---
    if (typeName == "NiStringExtraData") {
        return consumeSizedString(cursor) && consumeSizedString(cursor);
    }
    if (typeName == "BSXFlags" || typeName == "NiIntegerExtraData") {
        return consumeSizedString(cursor) && consumeSkip(cursor, 4u);
    }
    if (typeName == "NiBinaryExtraData") {
        return consumeSizedString(cursor) && consumeCountedArray(cursor, 1u);
    }
    if (typeName == "BSBound") {
        return consumeSizedString(cursor) && consumeSkip(cursor, 24u);
    }
    if (typeName == "BSFurnitureMarker") {
        return consumeSizedString(cursor) && consumeCountedArray(cursor, 16u);
    }
    if (typeName == "NiTextKeyExtraData") {
        std::uint32_t keyCount = 0;
        if (!consumeSizedString(cursor) || !cursor.read(keyCount)) {
            return false;
        }
        for (std::uint32_t i = 0; i < keyCount; ++i) {
            if (!consumeSkip(cursor, 4u) || !consumeSizedString(cursor)) {
                return false;
            }
        }
        return true;
    }

    // --- Nodes and geometry. ---
    if (typeName == "NiNode" || typeName == "NiBillboardNode" ||
        typeName == "BSFadeNode" || typeName == "NiSwitchNode" ||
        typeName == "NiLODNode" || typeName == "RootCollisionNode" ||
        typeName == "AvoidNode" || typeName == "BSValueNode" ||
        typeName == "NiBSAnimationNode" || typeName == "NiBSParticleNode") {
        if (!consumeNiAvObject(cursor) || !consumeCountedArray(cursor, 4u) ||
            !consumeCountedArray(cursor, 4u)) {
            return false;
        }
        return typeName != "NiBillboardNode" || consumeSkip(cursor, 2u);
    }
    if (typeName == "NiTriShape" || typeName == "NiTriStrips") {
        if (!consumeNiAvObject(cursor) || !consumeSkip(cursor, 8u)) {  // data, skin
            return false;
        }
        std::uint8_t hasShader = 0;
        if (!cursor.read(hasShader)) {
            return false;
        }
        return hasShader == 0u || (consumeSizedString(cursor) && consumeSkip(cursor, 4u));
    }
    if (typeName == "NiTriShapeData") {
        std::uint16_t numVertices = 0;
        std::uint16_t numTriangles = 0;
        std::uint8_t hasTriangles = 0;
        std::uint16_t matchGroups = 0;
        if (!consumeNiGeometryData(cursor, numVertices, version) || !cursor.read(numTriangles) ||
            !consumeSkip(cursor, 4u) || !cursor.read(hasTriangles)) {
            return false;
        }
        if (hasTriangles != 0u &&
            !consumeSkip(cursor, static_cast<std::size_t>(numTriangles) * 6u)) {
            return false;
        }
        if (!cursor.read(matchGroups)) {
            return false;
        }
        for (std::uint16_t i = 0; i < matchGroups; ++i) {
            std::uint16_t count = 0;
            if (!cursor.read(count) ||
                !consumeSkip(cursor, static_cast<std::size_t>(count) * 2u)) {
                return false;
            }
        }
        return true;
    }
    if (typeName == "NiTriStripsData") {
        std::uint16_t numVertices = 0;
        std::uint16_t numStrips = 0;
        if (!consumeNiGeometryData(cursor, numVertices, version) || !consumeSkip(cursor, 2u) ||
            !cursor.read(numStrips)) {
            return false;
        }
        std::size_t totalPoints = 0;
        for (std::uint16_t i = 0; i < numStrips; ++i) {
            std::uint16_t length = 0;
            if (!cursor.read(length)) {
                return false;
            }
            totalPoints += length;
        }
        std::uint8_t hasPoints = 0;
        if (!cursor.read(hasPoints)) {
            return false;
        }
        return hasPoints == 0u || consumeSkip(cursor, totalPoints * 2u);
    }

    // --- Properties. ---
    if (typeName == "NiAlphaProperty") {
        return consumeNiObjectNet(cursor) && consumeSkip(cursor, 3u);
    }
    if (typeName == "NiSpecularProperty" || typeName == "NiShadeProperty" ||
        typeName == "NiDitherProperty" || typeName == "NiWireframeProperty") {
        return consumeNiObjectNet(cursor) && consumeSkip(cursor, 2u);
    }
    if (typeName == "NiZBufferProperty") {
        return consumeNiObjectNet(cursor) && consumeSkip(cursor, 6u);
    }
    if (typeName == "NiVertexColorProperty") {
        return consumeNiObjectNet(cursor) && consumeSkip(cursor, 10u);
    }
    if (typeName == "NiStencilProperty") {
        return consumeNiObjectNet(cursor) && consumeSkip(cursor, 29u);
    }
    if (typeName == "NiMaterialProperty") {
        return consumeNiObjectNet(cursor) && consumeSkip(cursor, 48u + 4u + 4u);
    }
    if (typeName == "NiTexturingProperty") {
        std::uint32_t textureCount = 0;
        if (!consumeNiObjectNet(cursor) || !consumeSkip(cursor, 4u) ||
            !cursor.read(textureCount)) {
            return false;
        }
        const auto slot = [&cursor, version]() {
            std::uint8_t has = 0;
            if (!cursor.read(has)) {
                return false;
            }
            return has == 0u || consumeTexDesc(cursor, version);
        };
        for (int i = 0; i < 5; ++i) {  // base, dark, detail, gloss, glow
            if (!slot()) {
                return false;
            }
        }
        if (textureCount > 5u) {
            std::uint8_t hasBump = 0;
            if (!cursor.read(hasBump)) {
                return false;
            }
            if (hasBump != 0u &&
                (!consumeTexDesc(cursor, version) || !consumeSkip(cursor, 4u + 4u + 16u))) {
                return false;
            }
        }
        for (std::uint32_t threshold = 6u; threshold <= 9u; ++threshold) {
            if (textureCount > threshold && !slot()) {
                return false;
            }
        }
        // THE SHADER-TEXTURE ARRAY IS THE ONE EVERYONE MISSES. It exists from
        // 10.0.1.0, and omitting it under-consumes every texturing property in
        // the archive -- which desynchronizes the NiSourceTexture right after it
        // and looks like a bug in that block instead.
        std::uint32_t shaderTextures = 0;
        if (!cursor.read(shaderTextures)) {
            return false;
        }
        for (std::uint32_t i = 0; i < shaderTextures; ++i) {
            std::uint8_t hasMap = 0;
            if (!cursor.read(hasMap)) {
                return false;
            }
            if (hasMap != 0u && (!consumeTexDesc(cursor, version) || !consumeSkip(cursor, 4u))) {
                return false;
            }
        }
        return true;
    }
    if (typeName == "NiSourceTexture") {
        std::uint8_t useExternal = 0;
        if (!consumeNiObjectNet(cursor) || !cursor.read(useExternal)) {
            return false;
        }
        // Both branches are a file name followed by a Ref; only what the Ref
        // points at differs.
        return consumeSizedString(cursor) && consumeSkip(cursor, 4u) &&
               consumeSkip(cursor, 12u + 1u + 1u);  // format prefs, isStatic, directRender
    }
    return false;
}

// Fills blockStart/blockEnd by consuming each block in order. Any type without a
// reader aborts the whole file -- see consumeOblivionBlock.
namespace {

// ---------------------------------------------------------------------------
// NetImmerse 4.0.0.2 (Morrowind)
//
// A third block layout, and the differences from 20.0.0.4 are not a matter of a
// few fields. Each block is preceded by its own uint32-length-prefixed TYPE
// NAME, so unlike Oblivion the type is always known before the block is read --
// but there is still no size, so every block must be consumed exactly.
//
// AND EVERY BOOLEAN IS FOUR BYTES. Below 4.1.0.0 the format writes bools as
// int32. The gating bools in NiTriShapeData (hasVertices, hasNormals,
// hasVertexColors, hasUV) sit between the arrays they gate, so reading one as a
// byte does not fail -- it shifts everything after it by three bytes and yields
// a mesh made of noise.
//
// Measured coverage over the 5798 meshes in Morrowind.bsa: ten block readers
// cover 94.6% of the architecture set (meshes\x\, 865 files), which is what a
// town is built from. Sixteen cover 90.8% of the whole archive.

// Reads a 4.0.0.2 boolean: int32, not int8.
bool readWideBool(ByteCursor& cursor, bool& out) {
    std::uint32_t value = 0;
    if (!cursor.read(value)) {
        return false;
    }
    out = value != 0u;
    return true;
}

bool consumeMorrowindSizedString(ByteCursor& cursor) {
    std::string unused;
    return cursor.readSizedString<std::uint32_t>(unused);
}

// uint32 count followed by that many int32 refs.
bool consumeMorrowindRefList(ByteCursor& cursor) {
    std::uint32_t count = 0;
    if (!cursor.read(count)) {
        return false;
    }
    return cursor.skip(static_cast<std::size_t>(count) * 4u);
}

// NiObjectNET at 4.0.0.2: name, ONE extra-data ref (not a list), controller ref.
bool consumeMorrowindObjectNet(ByteCursor& cursor) {
    return consumeMorrowindSizedString(cursor) && cursor.skip(4u) && cursor.skip(4u);
}

bool consumeMorrowindBoundingVolume(ByteCursor& cursor) {
    std::uint32_t type = 0;
    if (!cursor.read(type)) {
        return false;
    }
    switch (type) {
        case 0xFFFFFFFFu: return true;              // base, no payload
        case 0u: return cursor.skip(16u);            // sphere: centre + radius
        case 1u: return cursor.skip(12u + 36u + 12u);  // box: centre, axes, extents
        case 2u: return cursor.skip(12u + 12u + 8u);   // capsule
        // Lozenge gains two extents only at 4.2.1.0, and half-space an origin.
        case 3u: return cursor.skip(4u + 12u + 12u + 12u);
        case 5u: return cursor.skip(16u);
        default: return false;                       // union and anything else
    }
}

// NiAVObject at 4.0.0.2. Note velocity IS present here (it goes away after
// 4.2.2.0) and the bounds flag is a four-byte bool.
bool consumeMorrowindAvObject(ByteCursor& cursor) {
    if (!consumeMorrowindObjectNet(cursor)) {
        return false;
    }
    // flags u16, translation 3f, rotation 9f, scale f, velocity 3f
    if (!cursor.skip(2u + 12u + 36u + 4u + 12u)) {
        return false;
    }
    if (!consumeMorrowindRefList(cursor)) {  // properties
        return false;
    }
    bool hasBounds = false;
    if (!readWideBool(cursor, hasBounds)) {
        return false;
    }
    return !hasBounds || consumeMorrowindBoundingVolume(cursor);
}

bool consumeMorrowindGeometryData(ByteCursor& cursor, bool strips, GeometryBlock* out);

// Consumes exactly one 4.0.0.2 block of `typeName`. False means "this reader
// does not know how long that block is", which fails the file -- there is no
// size to skip by and no separator to resynchronize on.
bool consumeMorrowindBlock(ByteCursor& cursor, std::string_view typeName) {
    // Every one of these is a bare NiNode on disk. RootCollisionNode's type name
    // is its entire semantics: its children are collision geometry that must not
    // be drawn, which the caller handles -- here it is just a node.
    if (typeName == "NiNode" || typeName == "RootCollisionNode" || typeName == "AvoidNode" ||
        typeName == "NiBSAnimationNode" || typeName == "NiBSParticleNode" ||
        typeName == "NiBillboardNode" || typeName == "NiCollisionSwitch" ||
        typeName == "NiSortAdjustNode" || typeName == "NiLODNode" ||
        typeName == "NiSwitchNode" || typeName == "NiFltAnimationNode") {
        return consumeMorrowindAvObject(cursor) && consumeMorrowindRefList(cursor) &&
               consumeMorrowindRefList(cursor);  // children, effects
    }
    if (typeName == "NiTriShape" || typeName == "NiTriStrips" || typeName == "NiLines") {
        // NiGeometry tail: data ref + skin-instance ref. MaterialData does not
        // exist before 10.0.1.0, so there is nothing after them.
        return consumeMorrowindAvObject(cursor) && cursor.skip(8u);
    }
    if (typeName == "NiTriShapeData") {
        return consumeMorrowindGeometryData(cursor, /*strips=*/false, nullptr);
    }
    if (typeName == "NiTriStripsData") {
        return consumeMorrowindGeometryData(cursor, /*strips=*/true, nullptr);
    }
    if (typeName == "NiSourceTexture") {
        if (!consumeMorrowindObjectNet(cursor)) {
            return false;
        }
        std::uint8_t useExternal = 0;
        if (!cursor.read(useExternal)) {
            return false;
        }
        bool hasPixelData = false;
        if (useExternal != 0u) {
            if (!consumeMorrowindSizedString(cursor)) {  // file name
                return false;
            }
        } else {
            std::uint8_t hasData = 0;
            if (!cursor.read(hasData)) {
                return false;
            }
            hasPixelData = hasData != 0u;
        }
        if (hasPixelData && !cursor.skip(4u)) {  // NiPixelData ref
            return false;
        }
        // pixelLayout, useMipMaps, alphaFormat, then isStatic (one byte).
        return cursor.skip(12u) && cursor.skip(1u);
    }
    if (typeName == "NiTexturingProperty") {
        if (!consumeMorrowindObjectNet(cursor) || !cursor.skip(2u)) {  // flags
            return false;
        }
        if (!cursor.skip(4u)) {  // apply mode
            return false;
        }
        std::uint32_t textureCount = 0;
        if (!cursor.read(textureCount)) {
            return false;
        }
        for (std::uint32_t slot = 0; slot < textureCount; ++slot) {
            bool enabled = false;
            if (!readWideBool(cursor, enabled)) {
                return false;
            }
            if (!enabled) {
                continue;  // a disabled slot is the four bytes and nothing else
            }
            // source ref, clamp, filter, uvSet, the PS2 filter dword, then a
            // two-byte unknown that only exists at or below 4.1.0.12.
            if (!cursor.skip(4u * 5u) || !cursor.skip(2u)) {
                return false;
            }
            // The BUMP slot carries an inline luma bias and matrix right here,
            // which is a classic desync: it belongs to the property, not to the
            // slot struct, and only appears when slot 5 is enabled.
            if (slot == 5u && !cursor.skip(8u + 16u)) {
                return false;
            }
        }
        return true;
    }
    if (typeName == "NiMaterialProperty") {
        // flags, ambient/diffuse/specular/emissive (3f each), glossiness, alpha
        return consumeMorrowindObjectNet(cursor) && cursor.skip(2u + 48u + 8u);
    }
    if (typeName == "NiAlphaProperty") {
        return consumeMorrowindObjectNet(cursor) && cursor.skip(2u + 1u);
    }
    if (typeName == "NiZBufferProperty" || typeName == "NiShadeProperty" ||
        typeName == "NiDitherProperty" || typeName == "NiSpecularProperty" ||
        typeName == "NiWireframeProperty") {
        // All of these are NiObjectNET + a single u16 at this version. The
        // z-buffer test function is packed into those flag bits until 4.1.0.12.
        return consumeMorrowindObjectNet(cursor) && cursor.skip(2u);
    }
    if (typeName == "NiVertexColorProperty") {
        return consumeMorrowindObjectNet(cursor) && cursor.skip(2u + 4u + 4u);
    }
    if (typeName == "NiStencilProperty") {
        return consumeMorrowindObjectNet(cursor) && cursor.skip(2u + 1u + 4u * 6u);
    }
    if (typeName == "NiFogProperty") {
        return consumeMorrowindObjectNet(cursor) && cursor.skip(2u + 4u + 12u);
    }
    // Every extra-data block at 4.0.0.2 opens with a next-ref and a byte count
    // INSTEAD of a name -- the name arrives at 10.0.1.0. The byte count is the
    // payload length for NiExtraData and is otherwise redundant.
    if (typeName == "NiStringExtraData") {
        return cursor.skip(4u) && cursor.skip(4u) && consumeMorrowindSizedString(cursor);
    }
    if (typeName == "NiTextKeyExtraData") {
        if (!cursor.skip(4u) || !cursor.skip(4u)) {
            return false;
        }
        std::uint32_t keyCount = 0;
        if (!cursor.read(keyCount)) {
            return false;
        }
        for (std::uint32_t i = 0; i < keyCount; ++i) {
            if (!cursor.skip(4u) || !consumeMorrowindSizedString(cursor)) {
                return false;
            }
        }
        return true;
    }
    if (typeName == "NiVertWeightsExtraData") {
        if (!cursor.skip(4u)) {
            return false;
        }
        std::uint32_t recordSize = 0;
        if (!cursor.read(recordSize)) {
            return false;
        }
        std::uint16_t weightCount = 0;
        if (!cursor.read(weightCount)) {
            return false;
        }
        return cursor.skip(static_cast<std::size_t>(weightCount) * 4u);
    }
    if (typeName == "NiExtraData") {
        if (!cursor.skip(4u)) {
            return false;
        }
        std::uint32_t recordSize = 0;
        if (!cursor.read(recordSize)) {
            return false;
        }
        return cursor.skip(recordSize);
    }
    return false;
}

// NiTriShapeData / NiTriStripsData at 4.0.0.2.
//
// The UV handling is the trap. At this version the "data flags" word IS the UV
// set count with no masking, and it is read AFTER the colours -- then a separate
// four-byte hasUV bool follows, and when it is false the count is forced to zero
// even though the count field has already been consumed.
// One layout, two uses: `out` null measures the block for the sequential walk,
// non-null also extracts it. They must stay the same code -- a walker and a
// reader that disagree by one field produce a file that "parses" into noise.
bool consumeMorrowindGeometryData(ByteCursor& cursor, bool strips, GeometryBlock* out) {
    std::uint16_t vertexCount = 0;
    if (!cursor.read(vertexCount)) {
        return false;
    }
    const auto readVectorOrSkip = [&](bool present, std::size_t componentsPerVertex,
                                      std::vector<float>* dst) {
        const std::size_t floatCount = static_cast<std::size_t>(vertexCount) * componentsPerVertex;
        if (!present) {
            return true;
        }
        if (out == nullptr || dst == nullptr) {
            return cursor.skip(floatCount * sizeof(float));
        }
        dst->resize(floatCount);
        for (float& value : *dst) {
            if (!cursor.read(value)) {
                return false;
            }
        }
        return true;
    };

    bool hasVertices = false;
    if (!readWideBool(cursor, hasVertices) ||
        !readVectorOrSkip(hasVertices, 3u, out ? &out->positions : nullptr)) {
        return false;
    }
    bool hasNormals = false;
    if (!readWideBool(cursor, hasNormals) ||
        !readVectorOrSkip(hasNormals, 3u, out ? &out->normals : nullptr)) {
        return false;
    }
    if (!cursor.skip(16u)) {  // bounding sphere: centre + radius
        return false;
    }
    // Vertex colours are RGBA FLOATS here, not the packed bytes later versions
    // use -- 16 bytes per vertex, and skipping 4 desynchronizes the UV count.
    bool hasColors = false;
    if (!readWideBool(cursor, hasColors) ||
        !cursor.skip(hasColors ? static_cast<std::size_t>(vertexCount) * 16u : 0u)) {
        return false;
    }
    std::uint16_t uvSetCount = 0;
    if (!cursor.read(uvSetCount)) {
        return false;
    }
    bool hasUv = false;
    if (!readWideBool(cursor, hasUv)) {
        return false;
    }
    if (!hasUv) {
        // The COUNT was already consumed; only the arrays are absent.
        uvSetCount = 0;
    }
    for (std::uint16_t set = 0; set < uvSetCount; ++set) {
        // Only UV set 0 is kept, matching every other generation's reader.
        std::vector<float>* dst = (set == 0u && out != nullptr) ? &out->uvs : nullptr;
        if (!readVectorOrSkip(true, 2u, dst)) {
            return false;
        }
    }
    std::uint16_t triangleCount = 0;
    if (!cursor.read(triangleCount)) {
        return false;
    }
    if (strips) {
        std::vector<std::uint16_t> lengths(triangleCount);
        for (std::uint16_t& length : lengths) {
            if (!cursor.read(length)) {
                return false;
            }
        }
        std::vector<std::uint16_t> strip;
        for (const std::uint16_t length : lengths) {
            strip.resize(length);
            for (std::uint16_t& point : strip) {
                if (!cursor.read(point)) {
                    return false;
                }
            }
            if (out != nullptr) {
                appendStripTriangles(strip, out->triangleIndices);
            }
        }
    } else {
        std::uint32_t indexCount = 0;
        if (!cursor.read(indexCount)) {
            return false;
        }
        // No hasTriangles bool at this version; it arrives after 10.0.1.2.
        if (out == nullptr) {
            if (!cursor.skip(static_cast<std::size_t>(indexCount) * 2u)) {
                return false;
            }
        } else {
            out->triangleIndices.resize(indexCount);
            for (std::uint32_t& index : out->triangleIndices) {
                std::uint16_t value = 0;
                if (!cursor.read(value)) {
                    return false;
                }
                index = value;
            }
        }
    }
    std::uint16_t matchGroupCount = 0;
    if (!cursor.read(matchGroupCount)) {
        return false;
    }
    for (std::uint16_t i = 0; i < matchGroupCount; ++i) {
        std::uint16_t count = 0;
        if (!cursor.read(count) || !cursor.skip(static_cast<std::size_t>(count) * 2u)) {
            return false;
        }
    }
    if (out != nullptr) {
        rejectUnusableTriangles(*out);
        out->valid = !out->positions.empty();
    }
    return true;
}

// Walks a 4.0.0.2 file, recording each block's bounds and synthesizing the type
// table the rest of the reader expects.
bool computeMorrowindBlockBounds(
    const std::vector<std::uint8_t>& bytes,
    NifHeader& header,
    std::size_t firstBlockOffset,
    std::vector<std::size_t>& blockStart,
    std::vector<std::size_t>& blockEnd,
    std::string& outError) {
    ByteCursor cursor(bytes.data(), bytes.size());
    cursor.seekAbsolute(firstBlockOffset);
    const std::size_t blockCount = header.blockTypeIndex.size();
    blockStart.assign(blockCount, 0u);
    blockEnd.assign(blockCount, 0u);
    std::unordered_map<std::string, std::uint16_t> typeIds;
    for (std::size_t i = 0; i < blockCount; ++i) {
        std::string typeName;
        if (!cursor.readSizedString<std::uint32_t>(typeName) || typeName.empty()) {
            outError = "Truncated NIF 4.0.0.2 block type name";
            return false;
        }
        const auto inserted =
            typeIds.emplace(typeName, static_cast<std::uint16_t>(header.blockTypeNames.size()));
        if (inserted.second) {
            header.blockTypeNames.push_back(typeName);
        }
        header.blockTypeIndex[i] = inserted.first->second;
        blockStart[i] = cursor.pos();
        if (!consumeMorrowindBlock(cursor, typeName)) {
            outError = "Unsupported NIF 4.0.0.2 block type '" + typeName +
                       "' (no size is derivable without it)";
            return false;
        }
        blockEnd[i] = cursor.pos();
        header.blockSize[i] = static_cast<std::uint32_t>(blockEnd[i] - blockStart[i]);
    }
    return true;
}

}  // namespace

bool computeSequentialBlockBounds(
    const std::vector<std::uint8_t>& bytes,
    const NifHeader& header,
    std::size_t firstBlockOffset,
    std::vector<std::size_t>& blockStart,
    std::vector<std::size_t>& blockEnd,
    std::string& outError
) {
    const std::size_t numBlocks = header.blockTypeIndex.size();
    blockStart.assign(numBlocks, 0u);
    blockEnd.assign(numBlocks, 0u);
    ByteCursor cursor(bytes.data(), bytes.size());
    cursor.seekAbsolute(firstBlockOffset);
    for (std::size_t i = 0; i < numBlocks; ++i) {
        blockStart[i] = cursor.pos();
        const std::string& typeName = header.blockTypeNames[header.blockTypeIndex[i]];
        if (!consumeOblivionBlock(cursor, typeName, header.version)) {
            // Name the block that broke AND the one before it. These blocks
            // carry no size, so the walk can only fail where it notices, which
            // is one block PAST wherever it actually desynchronized -- and the
            // failing type is then a red herring that has usually been handled
            // correctly for years. The predecessor is the suspect.
            char where[192];
            std::snprintf(where, sizeof(where),
                          " at block %zu of %zu, offset %zu; previous block was '%s'",
                          i, numBlocks, blockStart[i],
                          i == 0u
                              ? "(none)"
                              : header.blockTypeNames[header.blockTypeIndex[i - 1u]].c_str());
            outError = "Could not size NIF block type '" + typeName + "'" + where;
            return false;
        }
        blockEnd[i] = cursor.pos();
        if (blockEnd[i] > bytes.size()) {
            outError = "NIF block walk overran the file at block '" + typeName + "'";
            return false;
        }
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
    if (header.inlineBlockTypes) {
        // 4.0.0.2 has no type table either, so the walk synthesizes one as it
        // goes -- which is why this runs before the dispatch below can name a
        // single block.
        if (!computeMorrowindBlockBounds(
                bytes, header, cursor.pos(), blockStart, blockEnd, outError)) {
            return false;
        }
    } else if (header.sequentialBlocks) {
        if (!computeSequentialBlockBounds(
                bytes, header, cursor.pos(), blockStart, blockEnd, outError)) {
            return false;
        }
    } else {
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
    std::vector<std::string> noLightingTexturePaths(numBlocks);
    std::vector<bool> noLightingProperty(numBlocks, false);
    std::vector<AlphaPropertyBlock> alphaProperties(numBlocks);
    std::vector<StencilPropertyBlock> stencilProperties(numBlocks);
    std::unordered_set<std::int32_t> referencedAsChild;

    for (std::size_t i = 0; i < numBlocks; ++i) {
        const std::string& typeName = header.blockTypeNames[header.blockTypeIndex[i]];
        ByteCursor blockCursor(
            bytes.data() + blockStart[i], blockEnd[i] - blockStart[i]);

        // Any type whose name ends in "Node" is ATTEMPTED as a node even when
        // it is not in the list above. That collapses the old "unhandled node
        // type" case into the parse-failure case, which matters because the two
        // were handled differently and only one of them was guarded: an
        // unhandled type never reached readNiNode, so nodeParseFailedCount
        // stayed 0, the root-scan fallback below ran anyway, and the type's
        // children were promoted to roots and walked from identity -- the
        // reparent-to-origin bug this file exists to prevent, still reachable
        // through any NiNode-derived type the list happens not to name.
        //
        // Safe because recognition is proved structurally, not by name:
        // readNiNode rejects anything whose child count does not fit the block
        // or that lacks the effects-count trailer. A "*Node" type that is not
        // NiNode-derived fails that and is counted, rather than misparsed.
        if (isNodeTypeName(typeName) || looksLikeUnhandledNodeType(typeName)) {
            AvObjectFields fields;
            if (readNiNode(blockCursor, header.userVersion2, header.inlineNames, header.inlineBlockTypes, fields)) {
                nodeFields[i] = std::move(fields);
                isNiNode[i] = true;
                for (const std::int32_t child : nodeFields[i].children) {
                    referencedAsChild.insert(child);
                }
                if (!isNodeTypeName(typeName)) {
                    // Parsed fine, but the type is not one we knew about.
                    ++outModel.unhandledNodeTypeCount;
                }
            } else {
                // Counted rather than ignored. Salvaging the child refs here
                // would be unsound -- a failed walk IS a desync, so those refs
                // are exactly the bytes that cannot be trusted -- so the
                // subtree is left unreachable and simply not drawn.
                ++outModel.nodeParseFailedCount;
            }
        } else if (typeName == "NiTriShape" || typeName == "NiTriStrips" ||
                   typeName == "BSSegmentedTriShape") {
            // BSSegmentedTriShape is NiTriShape with a segment array appended
            // after the fields this reader consumes, so the same decode works
            // and the trailing segments are simply not read.
            //
            // It is what every distant-LOD landscape block is built from:
            // meshes\landscape\lod\<worldspace>\blocks\*.level4.x<X>.y<Y>.nif
            // holds a BSMultiBoundNode wrapping one BSSegmentedTriShape and a
            // plain NiTriShapeData. Without this branch those files parsed
            // "successfully" and yielded zero shapes -- the silent-drop failure
            // mode, not an error.
            AvObjectFields fields;
            if (readNiTriBasedGeom(blockCursor, header.userVersion2, header.inlineNames, header.inlineBlockTypes, fields)) {
                nodeFields[i] = std::move(fields);
                isTriShape[i] = true;
            }
        } else if (typeName == "BSShaderPPLightingProperty") {
            std::int32_t textureSetRef = -1;
            if (readBsShaderTextureSetRef(blockCursor, textureSetRef)) {
                shaderTextureSetRefs[i] = textureSetRef;
            }
        } else if (typeName == "BSShaderNoLightingProperty") {
            noLightingProperty[i] = true;
            std::string fileName;
            if (readBsShaderNoLightingTexture(blockCursor, fileName)) {
                noLightingTexturePaths[i] = std::move(fileName);
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
            if (readNiAlphaProperty(blockCursor, header.userVersion2, header.inlineNames, header.inlineBlockTypes, alpha)) {
                alphaProperties[i] = alpha;
            }
        } else if (typeName == "NiStencilProperty") {
            StencilPropertyBlock stencil;
            if (readNiStencilProperty(blockCursor, header.inlineNames, stencil)) {
                stencilProperties[i] = stencil;
            }
        } else if (typeName == "NiTriShapeData" || typeName == "NiTriStripsData") {
            GeometryBlock block;
            const bool isStrips = (typeName == "NiTriStripsData");
            // Not counted here: a data block that fails to parse is counted
            // once, at the shape that references it in the DFS below. Counting
            // both sites reported 2 for a single lost shape.
            if (readGeometryData(blockCursor, blockEnd[i] - blockStart[i], isStrips,
                             header.inlineBlockTypes, header.version, block) &&
                block.valid) {
                outModel.outOfRangeTriangleCount += block.outOfRangeTriangles;
                outModel.degenerateTriangleCount += block.degenerateTriangles;
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
        ByteCursor propertyCursor(
            bytes.data() + blockStart[i], blockEnd[i] - blockStart[i]);
        std::string fileName;
        if (findTexturingPropertySource(propertyCursor, sourceTexturePaths, fileName)) {
            texturingPropertyPaths[i] = std::move(fileName);
        }
    }

    // DFS from the file's declared roots, accumulating world transforms and
    // emitting one NifShape per reachable NiTriShape with valid geometry.
    //
    // Roots come from the FOOTER, which states them explicitly. The old rule --
    // "every node nobody claims as a child is a root" -- is what produced the
    // floating geometry: an unrecognized or unparsed parent left its children
    // unclaimed, so each was promoted to a root and walked from IDENTITY,
    // dropping every ancestor translation and landing the subtree on the model
    // origin. Reading the roots instead means an unreachable subtree is simply
    // not drawn, which is a diagnosable outcome rather than a wrong one.
    std::vector<std::size_t> stack;
    std::vector<Mat4> transformStack;
    std::vector<std::int32_t> footerRoots;
    if (readFooterRoots(bytes, blockEnd, footerRoots)) {
        for (const std::int32_t root : footerRoots) {
            // Skip what cannot be walked: nif.xml lists a first-person camera
            // among the roots "even if it is not a root object", and NiCamera
            // derives from NiAVObject, not NiNode -- it has no children list.
            if (root >= 0 && static_cast<std::size_t>(root) < numBlocks &&
                isNiNode[static_cast<std::size_t>(root)]) {
                stack.push_back(static_cast<std::size_t>(root));
                transformStack.push_back(Mat4{});
            }
        }
        outModel.usedFooterRoots = !stack.empty();
    }
    // Fall back to the old scan only when the footer gave us nothing walkable,
    // and only when every node in the file parsed -- a parse failure is exactly
    // the condition under which the scan invents a wrong root.
    if (stack.empty() && outModel.nodeParseFailedCount == 0u) {
        for (std::size_t i = 0; i < numBlocks; ++i) {
            if (isNiNode[i] &&
                referencedAsChild.find(static_cast<std::int32_t>(i)) == referencedAsChild.end()) {
                stack.push_back(i);
                transformStack.push_back(Mat4{});
            }
        }
    }

    // Gamebryo properties INHERIT down the scene graph: a NiAlphaProperty or
    // NiStencilProperty on a NiNode applies to every shape beneath it, not just
    // to the node itself. Walking only a shape's own property list therefore
    // misses them entirely, and a missed alpha property is not a subtle error --
    // the shape renders fully opaque, showing the black that Fallout's DDS files
    // carry underneath their transparent texels. That is the "black polygons on
    // the saloon sign" symptom.
    //
    // It used to be masked: applyTextureAlphaCutoutFlags inferred a cutout from
    // texture CONTENT and happened to catch these. That inference is off for
    // this importer now (ImportedScene::alphaFlagsAuthored, set by cell_builder)
    // because it also forced alpha test onto authored-opaque shapes sharing a
    // texture with a real cutout, so nothing covers the gap any more.
    //
    // Carried as the accumulated ref list rather than as resolved flags, so the
    // shape's own resolution loop handles an inherited property with exactly
    // the same code it handles its own with.
    std::vector<std::vector<std::int32_t>> inheritedPropertyStack;
    inheritedPropertyStack.assign(stack.size(), std::vector<std::int32_t>{});

    std::vector<bool> visited(numBlocks, false);
    while (!stack.empty()) {
        const std::size_t blockIndex = stack.back();
        const Mat4 parentTransform = transformStack.back();
        std::vector<std::int32_t> inheritedProperties = std::move(inheritedPropertyStack.back());
        stack.pop_back();
        transformStack.pop_back();
        inheritedPropertyStack.pop_back();

        if (blockIndex >= numBlocks || visited[blockIndex]) {
            continue;  // guards against malformed/cyclic child references
        }
        visited[blockIndex] = true;

        const Mat4 localTransform = makeTrs(
            nodeFields[blockIndex].translation, nodeFields[blockIndex].rotation, nodeFields[blockIndex].scale);
        const Mat4 worldTransform = multiply(parentTransform, localTransform);

        // COLLISION GEOMETRY IS REAL GEOMETRY, and it must not be drawn.
        //
        // Morrowind has no Havok: a mesh's collision hull is an ordinary
        // NiTriShape subtree parented to a node whose TYPE NAME is
        // RootCollisionNode, and the type name is the entire semantics -- on
        // disk the block is a bare NiNode with no extra fields. Drawn, those
        // hulls are untextured, UV-less slabs sitting a few units outside the
        // wall they approximate: they read as flat green panels over the
        // building, as faces missing where the hull occludes the real wall, and
        // as z-fighting where the two are coplanar. All three of those were
        // visible on Seyda Neen's shacks.
        //
        // ex_de_shack_01.nif is the worked example: 39 shapes, of which the
        // first three have uvs 0 and no texture. Those three are the hull.
        //
        // Skipped by not descending, rather than by dropping the shapes: the
        // subtree contributes nothing else, and stopping here also skips its
        // transform and property accumulation.
        if (blockIndex < header.blockTypeIndex.size() &&
            header.blockTypeNames[header.blockTypeIndex[blockIndex]] == "RootCollisionNode") {
            continue;
        }
        if (isNiNode[blockIndex]) {
            // What this node contributes to everything below it: whatever it
            // inherited, plus its own properties.
            std::vector<std::int32_t> childInherited = inheritedProperties;
            childInherited.insert(
                childInherited.end(),
                nodeFields[blockIndex].properties.begin(),
                nodeFields[blockIndex].properties.end());
            for (const std::int32_t child : nodeFields[blockIndex].children) {
                if (child >= 0 && static_cast<std::size_t>(child) < numBlocks) {
                    stack.push_back(static_cast<std::size_t>(child));
                    transformStack.push_back(worldTransform);
                    inheritedPropertyStack.push_back(childInherited);
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
            // BSShaderNoLightingProperty's directly-named texture is held back
            // until every property has been walked, rather than assigned in
            // place. The loop below is first-property-wins -- it bails on any
            // property once shape.diffuseTexturePath is non-empty -- so
            // assigning inline would let a NoLighting property that happens to
            // come first suppress a BSShaderTextureSet later in the same list.
            // A texture set is the richer source -- it also names the normal map,
            // which the normal-mapping path reads out of slot 1 -- so it wins,
            // and this is the fallback for shapes that have nothing else.
            // Holding it back is what makes this reader provably unable to
            // change any texture that already resolved, which the before/after
            // in the header comment relies on.
            std::string noLightingFallback;
            bool shapeReversedWinding = false;
            const auto applyProperty = [&](std::int32_t propertyRef) {
                if (propertyRef < 0 || static_cast<std::size_t>(propertyRef) >= numBlocks) {
                    return;
                }
                const auto propertyIndex = static_cast<std::size_t>(propertyRef);
                shape.unlit = shape.unlit || noLightingProperty[propertyIndex];
                if (noLightingFallback.empty() && !noLightingTexturePaths[propertyIndex].empty()) {
                    noLightingFallback = noLightingTexturePaths[propertyIndex];
                }
                if (alphaProperties[propertyIndex].valid) {
                    // A shape carries at most one alpha property in practice;
                    // if it somehow carries two, the one that turns the test on
                    // is the one whose threshold means anything.
                    if (alphaProperties[propertyIndex].alphaTest && !shape.alphaTest) {
                        shape.alphaThreshold = alphaProperties[propertyIndex].alphaThreshold;
                        if (alphaProperties[propertyIndex].alphaTest) {
                            ++outModel.alphaTestFunctionCounts
                                 [alphaProperties[propertyIndex].alphaTestFunction & 0x7u];
                        }
                    }
                    shape.alphaTest = shape.alphaTest || alphaProperties[propertyIndex].alphaTest;
                    shape.alphaBlend = shape.alphaBlend || alphaProperties[propertyIndex].alphaBlend;
                    return;
                }
                if (stencilProperties[propertyIndex].valid) {
                    shape.twoSided = shape.twoSided || stencilProperties[propertyIndex].twoSided;
                    shapeReversedWinding =
                        shapeReversedWinding || stencilProperties[propertyIndex].reversedWinding;
                    ++outModel.stencilDrawModeCounts[stencilProperties[propertyIndex].drawMode & 0x3u];
                    return;
                }
                if (!shape.diffuseTexturePath.empty()) {
                    return;
                }
                if (!texturingPropertyPaths[propertyIndex].empty()) {
                    shape.diffuseTexturePath = texturingPropertyPaths[propertyIndex];
                    return;
                }
                // A texture set, when the property has one. Type-validate the
                // resolved ref: the layout above is read, not guessed, so this
                // is an assertion rather than a search.
                const std::int32_t textureSetRef = shaderTextureSetRefs[propertyIndex];
                if (textureSetRef >= 0 && static_cast<std::size_t>(textureSetRef) < numBlocks) {
                    const TextureSetBlock& set = textureSets[static_cast<std::size_t>(textureSetRef)];
                    if (set.valid && !set.textures.empty() && !set.textures.front().empty()) {
                        shape.diffuseTexturePath = set.textures.front();
                    }
                    return;
                }
                if (!noLightingTexturePaths[propertyIndex].empty()) {
                    return;  // held back; applied after the loop if nothing better won
                }
                // Record what this actually was. A property that is neither an
                // alpha property nor something we got a texture out of is
                // exactly the case that leaves a shape untextured.
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
            };
            // The shape's OWN properties first, then the ones inherited from its
            // ancestors. Order matters and this one is deliberate: the diffuse
            // texture resolves first-wins, so going own-first means a shape that
            // resolves a texture today resolves the same texture after this
            // change, and a parent's texture property can never override a
            // shape's own. The alpha, stencil and unlit flags accumulate with
            // ||, so an inherited property can only ever turn one ON for a shape
            // that had none -- which is the only direction this fix is allowed
            // to move, since every shape rendering correctly today is a shape
            // whose own properties already said everything.
            for (const std::int32_t propertyRef : nodeFields[blockIndex].properties) {
                applyProperty(propertyRef);
            }
            for (const std::int32_t propertyRef : inheritedProperties) {
                applyProperty(propertyRef);
            }
            if (shape.diffuseTexturePath.empty() && !noLightingFallback.empty()) {
                shape.diffuseTexturePath = std::move(noLightingFallback);
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

            // Mirrored shapes: reverse the winding.
            //
            // A node transform with a NEGATIVE determinant is a reflection, and
            // Bethesda uses them freely -- a negative scale on one axis is how
            // one rock or rubble mesh becomes several distinct-looking ones.
            // Baking such a transform into world-space vertices (which this
            // parser does; there is no per-shape instance matrix downstream)
            // flips which side of every triangle faces out, while the index
            // order still says the old one does. The imported-static pipeline
            // culls back faces, so the visible result is a rock that is solid
            // from most angles and see-through from the mirrored one -- exactly
            // the "90% fine, a few transparent on one side" symptom.
            //
            // Swapping two indices per triangle restores the outward face.
            // Normals need no flip: they are transformed by the same matrix, so
            // they are already mirrored consistently with the positions.
            // Either source of inversion flips the winding, and BOTH together
            // cancel out -- so this is an XOR, not two independent reversals.
            const bool mirroredTransform = determinant3x3(worldTransform) < 0.0f;
            if (mirroredTransform) {
                ++outModel.mirroredShapeCount;
            }
            if (shapeReversedWinding) {
                ++outModel.reversedWindingShapeCount;
            }
            if (mirroredTransform != shapeReversedWinding) {
                for (std::size_t i = 0; i + 2u < shape.triangleIndices.size(); i += 3u) {
                    std::swap(shape.triangleIndices[i + 1u], shape.triangleIndices[i + 2u]);
                }
            }

            outModel.shapes.push_back(std::move(shape));
        }
    }

    return true;
}

// Parses a skeleton NIF's NiNode hierarchy into a flat, topologically ordered
// bone array.
//
// This reuses readNiNode rather than adding a second node decoder, which
// matters: readAvObjectPrefix carries the userVersion2 flags-width fix that
// cost a real bug to find (see its comment), and a parallel implementation
// would not have it.
//
// What is deliberately NOT read: a skeleton file is roughly half physics.
// meshes\characters\_male\skeleton.nif is 388 blocks, of which 65 are NiNodes
// and the rest are bhkRigidBody / bhkCapsuleShape / bhkRagdollConstraint
// (ragdoll), NiTransformController / NiTransformInterpolator (the idle the
// skeleton ships with), and one NiTriShape. Only the NiNodes are bones. The
// controllers are skipped because animation comes from .kf files against these
// bone names, not from whatever this file happens to embed.
bool parseNifSkeleton(
    const std::vector<std::uint8_t>& bytes, NifSkeleton& outSkeleton, std::string& outError) {
    outSkeleton = NifSkeleton{};
    ByteCursor cursor(bytes.data(), bytes.size());
    NifHeader header;
    if (!parseHeader(cursor, header, outError)) {
        return false;
    }

    if (header.sequentialBlocks) {
        // 20.0.0.4 has no block-size table, and only the static-mesh path has a
        // sequential walker. Saying so beats returning an empty skeleton, which
        // reads downstream as "this actor has no bones" rather than as an
        // unsupported format.
        outError = "Skinned/skeleton NIF parsing is not implemented for 20.0.0.4 yet";
        return false;
    }
    const std::size_t numBlocks = header.blockSize.size();
    std::vector<std::size_t> blockStart(numBlocks);
    std::size_t cursorPos = cursor.pos();
    for (std::size_t i = 0; i < numBlocks; ++i) {
        blockStart[i] = cursorPos;
        if (cursorPos + header.blockSize[i] > bytes.size()) {
            outError = "NIF block size table overruns the file";
            return false;
        }
        cursorPos += header.blockSize[i];
    }

    std::vector<AvObjectFields> nodeFields(numBlocks);
    std::vector<bool> isNiNode(numBlocks, false);
    for (std::size_t i = 0; i < numBlocks; ++i) {
        const std::string& typeName = header.blockTypeNames[header.blockTypeIndex[i]];
        if (!isNodeTypeName(typeName)) {
            continue;
        }
        ByteCursor blockCursor(
            bytes.data() + blockStart[i], header.blockSize[i]);
        AvObjectFields fields;
        if (readNiNode(blockCursor, header.userVersion2, header.inlineNames, header.inlineBlockTypes, fields)) {
            nodeFields[i] = std::move(fields);
            isNiNode[i] = true;
        }
    }

    // Parent links, derived from the children lists. A node claimed by two
    // parents keeps the first -- the format does not permit it, and preferring
    // the first at least keeps the hierarchy a tree rather than a cycle.
    std::vector<std::int32_t> parentBlock(numBlocks, -1);
    for (std::size_t i = 0; i < numBlocks; ++i) {
        if (!isNiNode[i]) {
            continue;
        }
        for (const std::int32_t child : nodeFields[i].children) {
            if (child < 0 || static_cast<std::size_t>(child) >= numBlocks) {
                continue;
            }
            const auto childIndex = static_cast<std::size_t>(child);
            if (isNiNode[childIndex] && parentBlock[childIndex] < 0) {
                parentBlock[childIndex] = static_cast<std::int32_t>(i);
            }
        }
    }

    // Emit depth-first from every root so parents always precede children,
    // which is the ordering NifSkeleton promises and what lets a consumer
    // accumulate world transforms in one forward pass.
    std::vector<int> boneIndexByBlock(numBlocks, -1);
    std::vector<std::size_t> stack;
    for (std::size_t i = numBlocks; i-- > 0;) {
        if (isNiNode[i] && parentBlock[i] < 0) {
            stack.push_back(i);
        }
    }
    while (!stack.empty()) {
        const std::size_t blockIndex = stack.back();
        stack.pop_back();

        NifSkeletonBone bone;
        const std::int32_t nameRef = nodeFields[blockIndex].nameRef;
        if (nameRef >= 0 && static_cast<std::size_t>(nameRef) < header.strings.size()) {
            bone.name = header.strings[static_cast<std::size_t>(nameRef)];
        }
        const AvObjectFields& fields = nodeFields[blockIndex];
        std::memcpy(bone.translation, fields.translation, sizeof(bone.translation));
        std::memcpy(bone.rotation, fields.rotation, sizeof(bone.rotation));
        bone.scale = fields.scale;
        const std::int32_t parent = parentBlock[blockIndex];
        bone.parentIndex =
            (parent >= 0) ? boneIndexByBlock[static_cast<std::size_t>(parent)] : -1;
        boneIndexByBlock[blockIndex] = static_cast<int>(outSkeleton.bones.size());
        outSkeleton.bones.push_back(std::move(bone));

        // Reversed so the traversal visits children in file order, which keeps
        // the emitted bone order stable and readable against a block dump.
        const auto& children = nodeFields[blockIndex].children;
        for (std::size_t c = children.size(); c-- > 0;) {
            const std::int32_t child = children[c];
            if (child < 0 || static_cast<std::size_t>(child) >= numBlocks) {
                continue;
            }
            const auto childIndex = static_cast<std::size_t>(child);
            if (isNiNode[childIndex] && boneIndexByBlock[childIndex] < 0) {
                stack.push_back(childIndex);
            }
        }
    }

    // Anything still unemitted was a NiNode reachable from no root -- a cycle,
    // or a child of a node type this parser does not walk. Counted, not
    // dropped silently: a missing bone means every vertex weighted to it
    // collapses somewhere it should not be.
    for (std::size_t i = 0; i < numBlocks; ++i) {
        if (isNiNode[i] && boneIndexByBlock[i] < 0) {
            ++outSkeleton.orphanedBoneCount;
        }
    }

    if (outSkeleton.bones.empty()) {
        outError = "NIF contains no NiNode hierarchy to read as a skeleton";
        return false;
    }
    return true;
}

namespace {

// NiSkinInstance, and its FNV subclass BSDismemberSkinInstance.
//
// NiSkinInstance layout (20.2.0.7):
//   data          ref  -> NiSkinData
//   skinPartition ref  -> NiSkinPartition
//   skeletonRoot  ptr  -> NiNode (a POINTER, same 4 bytes on disk as a ref)
//   numBones      u32
//   bones[]       ptr  -> NiNode, one per bone
//
// BSDismemberSkinInstance appends numPartitions + a partition array AFTER all
// of that, so the prefix above decodes identically and the dismemberment data
// -- which body part each triangle belongs to, for gore and armour swapping --
// is simply not read. That is why one reader serves both.
struct SkinInstanceBlock {
    std::int32_t dataRef = -1;
    std::int32_t partitionRef = -1;
    std::vector<std::int32_t> boneNodeRefs;
    bool valid = false;
};

bool readSkinInstance(ByteCursor& cursor, SkinInstanceBlock& out) {
    std::int32_t skeletonRoot = 0;
    std::uint32_t numBones = 0;
    if (!cursor.read(out.dataRef) || !cursor.read(out.partitionRef) ||
        !cursor.read(skeletonRoot) || !cursor.read(numBones)) {
        return false;
    }
    // A body mesh binds a few dozen bones. Anything past this is a misparse,
    // and rejecting it here keeps a bad offset from allocating wildly.
    if (numBones > 512u) {
        return false;
    }
    out.boneNodeRefs.resize(numBones);
    for (std::int32_t& ref : out.boneNodeRefs) {
        if (!cursor.read(ref)) {
            return false;
        }
    }
    out.valid = true;
    return true;
}

// One bone's entry inside NiSkinData.
struct SkinDataBone {
    float inverseBind[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};  // row-major
    std::vector<std::pair<std::uint16_t, float>> weights;  // (vertex index, weight)
};

struct SkinDataBlock {
    // The overall geometry-space -> skeleton-root-space transform. See
    // NifSkinnedShape::skinTransform for why it cannot be skipped.
    float skinTransform[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    std::vector<SkinDataBone> bones;
    bool valid = false;
};

// Builds a row-major 4x4 from NIF's (rotation 3x3, translation, scale) triple,
// which is how every transform in this format is stored.
void composeInverseBind(
    const float rotation[9], const float translation[3], float scale, float outMatrix[16]) {
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            outMatrix[(row * 4) + col] = rotation[(row * 3) + col] * scale;
        }
        outMatrix[(row * 4) + 3] = translation[row];
    }
    outMatrix[12] = 0.0f;
    outMatrix[13] = 0.0f;
    outMatrix[14] = 0.0f;
    outMatrix[15] = 1.0f;
}

// NiSkinData layout (20.2.0.7, userVersion2 = 34):
//   skinTransform      rotation 3x3, translation 3, scale 1   (36 + 12 + 4)
//   numBones           u32
//   skinPartition      ref     -- present only when version <= 10.1.0.0, so
//                                 NOT present here. Reading it anyway shifts
//                                 every bone by four bytes.
//   hasVertexWeights   u8
//   boneList[numBones]:
//       skinTransform  rotation 3x3, translation 3, scale 1
//       boundingSphere centre 3, radius 1
//       numVertices    u16
//       vertexWeights[numVertices]: index u16, weight f32
//
// The per-bone skinTransform IS the inverse bind pose -- it maps a vertex from
// skin space into that bone's local space, which is exactly what a skinning
// matrix needs on its right-hand side. It is not a bind pose to be inverted.
//
// The layout is confirmed by arithmetic rather than assumed: the block closes
// exactly when the last bone's weight array ends, and readSkinData rejects the
// parse otherwise. On upperbody.nif the NiSkinData blocks are 10477 / 8517 /
// ... bytes and every one of them closes.
bool readSkinData(ByteCursor& cursor, SkinDataBlock& out) {
    float rootRotation[9] = {};
    float rootTranslation[3] = {};
    float rootScale = 1.0f;
    std::uint32_t numBones = 0;
    std::uint8_t hasVertexWeights = 0;
    if (!cursor.read(rootRotation) || !cursor.read(rootTranslation) || !cursor.read(rootScale) ||
        !cursor.read(numBones) || !cursor.read(hasVertexWeights)) {
        return false;
    }
    composeInverseBind(rootRotation, rootTranslation, rootScale, out.skinTransform);
    if (numBones > 512u) {
        return false;
    }
    out.bones.resize(numBones);
    for (SkinDataBone& bone : out.bones) {
        float rotation[9] = {};
        float translation[3] = {};
        float scale = 1.0f;
        float boundingSphere[4] = {};
        std::uint16_t numVertices = 0;
        if (!cursor.read(rotation) || !cursor.read(translation) || !cursor.read(scale) ||
            !cursor.read(boundingSphere) || !cursor.read(numVertices)) {
            return false;
        }
        composeInverseBind(rotation, translation, scale, bone.inverseBind);
        if (hasVertexWeights == 0u) {
            // Legal per the format: the weights then live in the
            // NiSkinPartition instead. Not seen in retail FNV bodies, and the
            // caller reports a shape with no weights rather than emitting one
            // that would collapse to the origin.
            continue;
        }
        bone.weights.resize(numVertices);
        for (auto& weight : bone.weights) {
            if (!cursor.read(weight.first) || !cursor.read(weight.second)) {
                return false;
            }
        }
    }
    out.valid = true;
    return true;
}

// Reduces one vertex's influence list to kNifMaxBoneInfluences, keeping the
// largest weights, and renormalizes so they sum to 1.
//
// Renormalizing rather than just dropping is what keeps a truncated vertex on
// the model: skinning is a weighted average of bone transforms, so weights
// summing to 0.85 shrink the vertex 15% of the way toward the world origin --
// which on a character at the far end of the Mojave is a spike across the map,
// not a subtle error.
void reduceInfluences(
    std::vector<std::pair<std::uint16_t, float>>& influences, bool& outTruncated) {
    outTruncated = false;
    if (influences.size() > static_cast<std::size_t>(kNifMaxBoneInfluences)) {
        std::partial_sort(
            influences.begin(), influences.begin() + kNifMaxBoneInfluences, influences.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });
        influences.resize(static_cast<std::size_t>(kNifMaxBoneInfluences));
        outTruncated = true;
    }
    float total = 0.0f;
    for (const auto& influence : influences) {
        total += influence.second;
    }
    if (total <= 0.0f) {
        return;
    }
    for (auto& influence : influences) {
        influence.second /= total;
    }
}

}  // namespace

bool parseNifSkinnedMesh(
    const std::vector<std::uint8_t>& bytes, NifSkinnedModel& outModel, std::string& outError) {
    outModel = NifSkinnedModel{};
    ByteCursor cursor(bytes.data(), bytes.size());
    NifHeader header;
    if (!parseHeader(cursor, header, outError)) {
        return false;
    }

    if (header.sequentialBlocks) {
        // 20.0.0.4 has no block-size table, and only the static-mesh path has a
        // sequential walker. Saying so beats returning an empty skeleton, which
        // reads downstream as "this actor has no bones" rather than as an
        // unsupported format.
        outError = "Skinned/skeleton NIF parsing is not implemented for 20.0.0.4 yet";
        return false;
    }
    const std::size_t numBlocks = header.blockSize.size();
    std::vector<std::size_t> blockStart(numBlocks);
    std::size_t cursorPos = cursor.pos();
    for (std::size_t i = 0; i < numBlocks; ++i) {
        blockStart[i] = cursorPos;
        if (cursorPos + header.blockSize[i] > bytes.size()) {
            outError = "NIF block size table overruns the file";
            return false;
        }
        cursorPos += header.blockSize[i];
    }

    // A skinned shape's vertices are in SKIN space already -- the whole point
    // of the inverse-bind transform is that they are not in any node's local
    // space. So unlike parseNifStaticMesh, no node world transform is
    // accumulated or applied here. Baking the NiTriShape's own transform in
    // would double-apply it, since NiSkinData's per-bone transform is relative
    // to the same skin space.
    std::vector<AvObjectFields> nodeFields(numBlocks);
    std::vector<bool> isTriShape(numBlocks, false);
    std::vector<GeometryBlock> geometry(numBlocks);
    std::vector<TextureSetBlock> textureSets(numBlocks);
    std::vector<bool> noLightingProperty(numBlocks, false);
    std::vector<std::int32_t> shaderTextureSetRefs(numBlocks, -1);
    std::vector<AlphaPropertyBlock> alphaProperties(numBlocks);
    std::vector<StencilPropertyBlock> stencilProperties(numBlocks);
    std::vector<SkinInstanceBlock> skinInstances(numBlocks);
    std::vector<SkinDataBlock> skinData(numBlocks);
    std::vector<std::string> nodeNames(numBlocks);
    // Which skin instance each shape names. Read here rather than in
    // readNiTriBasedGeom because the static path has no use for it and the
    // field sits immediately after the data ref that path already reads.
    std::vector<std::int32_t> shapeSkinRefs(numBlocks, -1);

    for (std::size_t i = 0; i < numBlocks; ++i) {
        const std::string& typeName = header.blockTypeNames[header.blockTypeIndex[i]];
        ByteCursor blockCursor(
            bytes.data() + blockStart[i], header.blockSize[i]);

        if (isNodeTypeName(typeName)) {
            AvObjectFields fields;
            if (readNiNode(blockCursor, header.userVersion2, header.inlineNames, header.inlineBlockTypes, fields)) {
                const std::int32_t nameRef = fields.nameRef;
                if (nameRef >= 0 && static_cast<std::size_t>(nameRef) < header.strings.size()) {
                    nodeNames[i] = header.strings[static_cast<std::size_t>(nameRef)];
                }
                nodeFields[i] = std::move(fields);
            }
        } else if (typeName == "NiTriShape" || typeName == "NiTriStrips" ||
                   typeName == "BSSegmentedTriShape") {
            AvObjectFields fields;
            if (readNiTriBasedGeom(blockCursor, header.userVersion2, header.inlineNames, header.inlineBlockTypes, fields)) {
                // The skin instance ref is the very next field after the data
                // ref readNiTriBasedGeom stops at.
                std::int32_t skinRef = -1;
                if (blockCursor.read(skinRef)) {
                    shapeSkinRefs[i] = skinRef;
                }
                nodeFields[i] = std::move(fields);
                isTriShape[i] = true;
            }
        } else if (typeName == "NiSkinInstance" || typeName == "BSDismemberSkinInstance") {
            SkinInstanceBlock instance;
            if (readSkinInstance(blockCursor, instance)) {
                skinInstances[i] = std::move(instance);
            }
        } else if (typeName == "NiSkinData") {
            SkinDataBlock data;
            if (readSkinData(blockCursor, data)) {
                skinData[i] = std::move(data);
            }
        } else if (typeName == "BSShaderPPLightingProperty") {
            std::int32_t textureSetRef = -1;
            if (readBsShaderTextureSetRef(blockCursor, textureSetRef)) {
                shaderTextureSetRefs[i] = textureSetRef;
            }
        } else if (typeName == "BSShaderNoLightingProperty") {
            noLightingProperty[i] = true;
        } else if (typeName == "BSShaderTextureSet") {
            TextureSetBlock set;
            if (readBsShaderTextureSet(blockCursor, set)) {
                textureSets[i] = std::move(set);
            }
        } else if (typeName == "NiAlphaProperty") {
            AlphaPropertyBlock alpha;
            if (readNiAlphaProperty(blockCursor, header.userVersion2, header.inlineNames, header.inlineBlockTypes, alpha)) {
                alphaProperties[i] = alpha;
            }
        } else if (typeName == "NiStencilProperty") {
            StencilPropertyBlock stencil;
            if (readNiStencilProperty(blockCursor, header.inlineNames, stencil)) {
                stencilProperties[i] = stencil;
            }
        } else if (typeName == "NiTriShapeData" || typeName == "NiTriStripsData") {
            GeometryBlock block;
            const bool isStrips = (typeName == "NiTriStripsData");
            if (readGeometryData(blockCursor, header.blockSize[i], isStrips, header.inlineBlockTypes,
                             header.version, block) &&
                block.valid) {
                outModel.outOfRangeTriangleCount += block.outOfRangeTriangles;
                outModel.degenerateTriangleCount += block.degenerateTriangles;
                geometry[i] = std::move(block);
            }
        }
    }

    // Child -> parent, so a shape can collect the properties its ancestors
    // declare. Unlike the static path this walk is a flat scan over every block
    // rather than a DFS, so there is no parent on hand to inherit from and one
    // has to be reconstructed. Same reason as the static path: Gamebryo
    // properties inherit down the graph, and an alpha property a shape does not
    // itself carry still applies to it.
    std::vector<std::int32_t> parentOf(numBlocks, -1);
    for (std::size_t i = 0; i < numBlocks; ++i) {
        for (const std::int32_t child : nodeFields[i].children) {
            if (child >= 0 && static_cast<std::size_t>(child) < numBlocks &&
                parentOf[static_cast<std::size_t>(child)] < 0) {
                parentOf[static_cast<std::size_t>(child)] = static_cast<std::int32_t>(i);
            }
        }
    }

    for (std::size_t blockIndex = 0; blockIndex < numBlocks; ++blockIndex) {
        if (!isTriShape[blockIndex]) {
            continue;
        }
        const std::int32_t dataRef = nodeFields[blockIndex].dataRef;
        if (dataRef < 0 || static_cast<std::size_t>(dataRef) >= numBlocks ||
            !geometry[static_cast<std::size_t>(dataRef)].valid) {
            continue;
        }
        const GeometryBlock& src = geometry[static_cast<std::size_t>(dataRef)];
        const std::size_t vertexCount = src.positions.size() / 3u;

        // The skin instance, and through it the skin data. A shape missing
        // either is real geometry that simply is not skinned.
        const std::int32_t skinRef = shapeSkinRefs[blockIndex];
        if (skinRef < 0 || static_cast<std::size_t>(skinRef) >= numBlocks ||
            !skinInstances[static_cast<std::size_t>(skinRef)].valid) {
            ++outModel.unskinnedShapeCount;
            continue;
        }
        const SkinInstanceBlock& instance = skinInstances[static_cast<std::size_t>(skinRef)];
        const std::int32_t dataBlockRef = instance.dataRef;
        if (dataBlockRef < 0 || static_cast<std::size_t>(dataBlockRef) >= numBlocks ||
            !skinData[static_cast<std::size_t>(dataBlockRef)].valid) {
            ++outModel.unskinnedShapeCount;
            continue;
        }
        const SkinDataBlock& data = skinData[static_cast<std::size_t>(dataBlockRef)];
        // The two bone lists must agree: NiSkinInstance names the nodes,
        // NiSkinData holds their transforms and weights, and they are parallel
        // arrays with no cross-reference. A mismatch means one of the two
        // parsed wrong, and pairing them anyway would bind every vertex to the
        // wrong bone -- a failure that looks like a mangled character rather
        // than like a parse error.
        if (data.bones.size() != instance.boneNodeRefs.size()) {
            ++outModel.unskinnedShapeCount;
            continue;
        }

        NifSkinnedShape shape;
        const std::int32_t nameRef = nodeFields[blockIndex].nameRef;
        if (nameRef >= 0 && static_cast<std::size_t>(nameRef) < header.strings.size()) {
            shape.name = header.strings[static_cast<std::size_t>(nameRef)];
        }
        shape.positions = src.positions;
        shape.normals = src.normals;
        shape.uvs = src.uvs;
        shape.triangleIndices = src.triangleIndices;

        const auto applySkinnedProperty = [&](std::int32_t propertyRef) {
            if (propertyRef < 0 || static_cast<std::size_t>(propertyRef) >= numBlocks) {
                return;
            }
            const auto propertyIndex = static_cast<std::size_t>(propertyRef);
            shape.unlit = shape.unlit || noLightingProperty[propertyIndex];
            if (alphaProperties[propertyIndex].valid) {
                // Accumulate with ||, matching the static path. This used to
                // assign, which made it last-property-wins: a second alpha
                // property on the same shape silently erased the first one's
                // answer, and the threshold went with it.
                if (alphaProperties[propertyIndex].alphaTest && !shape.alphaTest) {
                    shape.alphaThreshold = alphaProperties[propertyIndex].alphaThreshold;
                }
                shape.alphaTest = shape.alphaTest || alphaProperties[propertyIndex].alphaTest;
                shape.alphaBlend = shape.alphaBlend || alphaProperties[propertyIndex].alphaBlend;
                return;
            }
            if (stencilProperties[propertyIndex].valid) {
                shape.twoSided = shape.twoSided || stencilProperties[propertyIndex].twoSided;
                return;
            }
            if (!shape.diffuseTexturePath.empty()) {
                return;
            }
            const std::int32_t textureSetRef = shaderTextureSetRefs[propertyIndex];
            if (textureSetRef >= 0 && static_cast<std::size_t>(textureSetRef) < numBlocks) {
                const TextureSetBlock& set = textureSets[static_cast<std::size_t>(textureSetRef)];
                if (set.valid && !set.textures.empty() && !set.textures.front().empty()) {
                    shape.diffuseTexturePath = set.textures.front();
                }
            }
        };
        // Own properties first, then each ancestor's in turn walking up to the
        // root -- same own-before-inherited ordering as the static path, for
        // the same reason: it cannot change any texture that already resolves.
        // Bounded by numBlocks rather than by reaching a null parent, so a
        // malformed file with a cyclic child list terminates instead of hanging.
        for (const std::int32_t propertyRef : nodeFields[blockIndex].properties) {
            applySkinnedProperty(propertyRef);
        }
        std::int32_t ancestor = parentOf[blockIndex];
        for (std::size_t step = 0; ancestor >= 0 && step < numBlocks; ++step) {
            const auto ancestorIndex = static_cast<std::size_t>(ancestor);
            for (const std::int32_t propertyRef : nodeFields[ancestorIndex].properties) {
                applySkinnedProperty(propertyRef);
            }
            ancestor = parentOf[ancestorIndex];
        }

        // Bone names, resolved through the instance's node pointers. A name
        // that does not resolve stays empty rather than shifting every later
        // bone down by one -- the arrays are positional, so dropping an entry
        // would silently rebind the whole shape.
        std::memcpy(shape.skinTransform, data.skinTransform, sizeof(shape.skinTransform));
        shape.boneNames.resize(instance.boneNodeRefs.size());
        shape.inverseBindMatrices.resize(instance.boneNodeRefs.size() * 16u);
        for (std::size_t b = 0; b < instance.boneNodeRefs.size(); ++b) {
            const std::int32_t nodeRef = instance.boneNodeRefs[b];
            if (nodeRef >= 0 && static_cast<std::size_t>(nodeRef) < numBlocks) {
                shape.boneNames[b] = nodeNames[static_cast<std::size_t>(nodeRef)];
            }
            std::memcpy(
                shape.inverseBindMatrices.data() + (b * 16u), data.bones[b].inverseBind,
                sizeof(float) * 16u);
        }

        // Transpose the weight lists: NiSkinData stores them per bone, the GPU
        // wants them per vertex.
        std::vector<std::vector<std::pair<std::uint16_t, float>>> perVertex(vertexCount);
        for (std::size_t b = 0; b < data.bones.size(); ++b) {
            for (const auto& [vertexIndex, weight] : data.bones[b].weights) {
                if (vertexIndex >= vertexCount || weight <= 0.0f) {
                    continue;
                }
                perVertex[vertexIndex].emplace_back(static_cast<std::uint16_t>(b), weight);
            }
        }

        shape.boneIndices.assign(vertexCount * kNifMaxBoneInfluences, 0u);
        shape.boneWeights.assign(vertexCount * kNifMaxBoneInfluences, 0.0f);
        for (std::size_t v = 0; v < vertexCount; ++v) {
            bool truncated = false;
            reduceInfluences(perVertex[v], truncated);
            if (truncated) {
                ++outModel.truncatedInfluenceVertexCount;
            }
            for (std::size_t k = 0; k < perVertex[v].size(); ++k) {
                shape.boneIndices[(v * kNifMaxBoneInfluences) + k] = perVertex[v][k].first;
                shape.boneWeights[(v * kNifMaxBoneInfluences) + k] = perVertex[v][k].second;
            }
        }

        outModel.shapes.push_back(std::move(shape));
    }

    if (outModel.shapes.empty() && outModel.unskinnedShapeCount == 0u) {
        outError = "NIF contains no geometry";
        return false;
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
