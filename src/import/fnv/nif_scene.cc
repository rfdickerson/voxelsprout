#include "import/fnv/nif_scene.h"

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
    template <typename LengthT>
    bool readSizedString(std::string& out) {
        LengthT length = 0;
        if (!read(length)) {
            return false;
        }
        out.resize(static_cast<std::size_t>(length));
        return length == 0 || readBytes(out.data(), static_cast<std::size_t>(length));
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
};

// Reads the NiObjectNET + NiAVObject fields common to NiNode and
// NiTriShape/NiTriStrips. Only reads through `scale` + property/collision
// refs — callers resume via the block's declared size, not by continuing
// this cursor, so anything after this point (children, data ref, ...) is
// read separately per block type. Returns false only on truncation this
// early in the block, which should not happen for a well-formed file.
bool readAvObjectPrefix(ByteCursor& cursor, AvObjectFields& out) {
    std::int32_t nameRef = 0;
    if (!cursor.read(nameRef)) {
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
    std::uint16_t flags = 0;
    if (!cursor.read(flags)) {
        return false;
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
    for (std::uint32_t i = 0; i < numProperties; ++i) {
        std::int32_t ref = 0;
        if (!cursor.read(ref)) {
            return false;
        }
    }
    std::int32_t collisionRef = 0;
    if (!cursor.read(collisionRef)) {
        return false;
    }
    return true;
}

bool readNiNode(ByteCursor& cursor, AvObjectFields& out) {
    if (!readAvObjectPrefix(cursor, out)) {
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

bool readNiTriBasedGeom(ByteCursor& cursor, AvObjectFields& out) {
    if (!readAvObjectPrefix(cursor, out)) {
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
    std::vector<std::uint32_t> triangleIndices;
    bool valid = false;
};

// NiTriShapeData / NiTriBasedGeomData layout. This is the least-certain part
// of this parser (see the file header comment) — vertex colors and
// multi-UV-set presence are read only far enough to skip their bytes
// correctly; if the final read position doesn't land inside the block's own
// declared size, the geometry is rejected rather than trusted.
bool readNiTriShapeData(ByteCursor& cursor, std::size_t blockEnd, GeometryBlock& out) {
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

    std::uint16_t vectorFlags = 0;
    if (!cursor.read(vectorFlags)) {
        return false;
    }
    constexpr std::uint16_t kHasNormalsBit = 0x0001u;
    constexpr std::uint16_t kHasTangentsBit = 0x1000u;
    // Bits 6-11: number of UV sets. Deliberately non-overlapping with
    // kHasNormalsBit/kHasTangentsBit above (an earlier version of this mask
    // aliased bit 0 with kHasNormalsBit, corrupting the byte-skip math
    // whenever normals were present but no UV sets were — caught by the
    // self-consistency check below, not by any external reference).
    constexpr std::uint16_t kUvSetCountShift = 6u;
    constexpr std::uint16_t kUvSetCountMask = 0x003Fu;

    if ((vectorFlags & kHasNormalsBit) != 0u) {
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

    const std::uint16_t uvSetCount = (vectorFlags >> kUvSetCountShift) & kUvSetCountMask;
    if (uvSetCount != 0u) {
        const std::size_t skipBytes =
            static_cast<std::size_t>(uvSetCount) * static_cast<std::size_t>(numVertices) * 2u * sizeof(float);
        cursor.seekAbsolute(cursor.pos() + skipBytes);
    }

    std::uint16_t consistencyType = 0;
    if (!cursor.read(consistencyType)) {
        return false;
    }
    std::int32_t additionalDataRef = 0;
    if (!cursor.read(additionalDataRef)) {
        return false;
    }

    std::uint16_t numTriangles = 0;
    std::uint32_t numTrianglePoints = 0;
    if (!cursor.read(numTriangles) || !cursor.read(numTrianglePoints)) {
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

    // Self-consistency check: if the fields above were misread (wrong guess
    // about a flag/skip), the cursor will almost certainly have run past
    // this block's own declared end by now (each block is bounds-checked
    // independently by ByteCursor, but nothing stops us reading into the
    // *next* block's bytes as if they were still ours). Reject rather than
    // trust a shape whose parse overran its own block.
    if (cursor.pos() > blockEnd) {
        return false;
    }

    out.valid = !out.positions.empty();
    return true;
}

}  // namespace

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
    std::unordered_set<std::int32_t> referencedAsChild;

    for (std::size_t i = 0; i < numBlocks; ++i) {
        const std::string& typeName = header.blockTypeNames[header.blockTypeIndex[i]];
        ByteCursor blockCursor(bytes.data() + blockStart[i], header.blockSize[i]);

        if (typeName == "NiNode" || typeName == "NiBSAnimationNode" || typeName == "BSFadeNode") {
            AvObjectFields fields;
            if (readNiNode(blockCursor, fields)) {
                nodeFields[i] = std::move(fields);
                isNiNode[i] = true;
                for (const std::int32_t child : nodeFields[i].children) {
                    referencedAsChild.insert(child);
                }
            }
        } else if (typeName == "NiTriShape" || typeName == "NiTriStrips") {
            AvObjectFields fields;
            if (readNiTriBasedGeom(blockCursor, fields)) {
                nodeFields[i] = std::move(fields);
                isTriShape[i] = true;
            }
        } else if (typeName == "NiTriShapeData") {
            GeometryBlock block;
            if (readNiTriShapeData(blockCursor, header.blockSize[i], block) && block.valid) {
                geometry[i] = std::move(block);
            } else {
                ++outModel.skippedShapeCount;
            }
        }
        // Every other block type is intentionally left unparsed: the next
        // block always starts at blockStart[i+1] regardless of what (if
        // anything) was read here.
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
                continue;
            }
            const GeometryBlock& src = geometry[dataRef];
            NifShape shape;
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
