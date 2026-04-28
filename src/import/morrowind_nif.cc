#include "import/morrowind_nif.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace odai::importer {

namespace {

constexpr std::string_view kExpectedHeader = "NetImmerse File Format, Version 4.0.0.2\n";
constexpr std::uint32_t kExpectedVersion = 0x04000002u;

struct NifCursor {
    std::vector<std::uint8_t> bytes;
    std::size_t offset = 0;

    std::size_t remaining() const {
        return bytes.size() - std::min(offset, bytes.size());
    }

    template <typename T>
    T readValue() {
        if (remaining() < sizeof(T)) {
            throw std::runtime_error("Unexpected end of file");
        }
        T value{};
        std::memcpy(&value, bytes.data() + offset, sizeof(T));
        offset += sizeof(T);
        return value;
    }

    void skip(std::size_t size) {
        if (remaining() < size) {
            throw std::runtime_error("Unexpected end of file");
        }
        offset += size;
    }

    std::string readSizedString() {
        const std::uint32_t size = readValue<std::uint32_t>();
        if (remaining() < size) {
            throw std::runtime_error("Unexpected end of file");
        }
        std::string value(reinterpret_cast<const char*>(bytes.data() + offset), size);
        offset += size;
        const std::size_t nullPos = value.find('\0');
        if (nullPos != std::string::npos) {
            value.resize(nullPos);
        }
        return value;
    }

    bool readBool() {
        return readValue<std::int32_t>() != 0;
    }
};

struct NifTransform {
    std::array<float, 3> translation{0.0f, 0.0f, 0.0f};
    std::array<float, 9> rotation{
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f
    };
    float scale = 1.0f;
};

struct NifAvObject {
    std::string name;
    std::uint32_t flags = 0u;
    NifTransform transform;
    std::vector<std::int32_t> propertyRefs;
};

struct NifNodeRecord {
    NifAvObject avObject;
    std::vector<std::int32_t> childRefs;
    bool isRootCollisionNode = false;
};

struct NifGeometryRecord {
    NifAvObject avObject;
    std::int32_t dataRef = -1;
};

struct NifTriShapeDataRecord {
    std::vector<std::array<float, 3>> vertices;
    std::vector<std::array<float, 3>> normals;
    std::vector<std::array<float, 2>> uv0;
    std::vector<std::uint16_t> indices;
};

struct NifTexturingPropertyRecord {
    std::int32_t baseTextureSourceRef = -1;
};

struct NifSourceTextureRecord {
    std::string filePath;
};

struct NifAlphaPropertyRecord {
    bool alphaTest = false;
};

enum class RecordKind {
    Unknown,
    Node,
    Geometry,
    TriShapeData,
    ParticleData,
    TexturingProperty,
    SourceTexture,
    AlphaProperty
};

struct NifRecord {
    RecordKind kind = RecordKind::Unknown;
    NifNodeRecord node{};
    NifGeometryRecord geometry{};
    NifTriShapeDataRecord triShapeData{};
    NifTexturingPropertyRecord texturingProperty{};
    NifSourceTextureRecord sourceTexture{};
    NifAlphaPropertyRecord alphaProperty{};
};

std::array<float, 16> identityMatrix() {
    return {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    };
}

std::array<float, 16> multiplyMatrices(
    const std::array<float, 16>& lhs,
    const std::array<float, 16>& rhs
) {
    std::array<float, 16> out{};
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            float sum = 0.0f;
            for (int i = 0; i < 4; ++i) {
                sum += lhs[row * 4 + i] * rhs[i * 4 + col];
            }
            out[row * 4 + col] = sum;
        }
    }
    return out;
}

std::array<float, 3> transformPoint(
    const std::array<float, 16>& matrix,
    const std::array<float, 3>& point
) {
    return {
        matrix[0] * point[0] + matrix[1] * point[1] + matrix[2] * point[2] + matrix[3],
        matrix[4] * point[0] + matrix[5] * point[1] + matrix[6] * point[2] + matrix[7],
        matrix[8] * point[0] + matrix[9] * point[1] + matrix[10] * point[2] + matrix[11]
    };
}

std::array<float, 3> transformVector(
    const std::array<float, 16>& matrix,
    const std::array<float, 3>& vector
) {
    return {
        matrix[0] * vector[0] + matrix[1] * vector[1] + matrix[2] * vector[2],
        matrix[4] * vector[0] + matrix[5] * vector[1] + matrix[6] * vector[2],
        matrix[8] * vector[0] + matrix[9] * vector[1] + matrix[10] * vector[2]
    };
}

std::array<float, 3> toEngineSpace(const std::array<float, 3>& value) {
    // Morrowind content uses X/Y as the ground plane with Z up.
    // Convert to the engine's X/Z ground plane with Y up.
    return {
        value[0],
        value[2],
        value[1]
    };
}

std::array<float, 16> makeTransformMatrix(const NifTransform& transform) {
    std::array<float, 16> matrix = identityMatrix();
    // Match OpenMW's NiTransform::toMatrix conversion. NIF stores the 3x3 basis
    // with the opposite row/column convention from our local matrix helpers, so
    // the authored rotation-scale block must be transposed during import.
    matrix[0] = transform.rotation[0] * transform.scale;
    matrix[1] = transform.rotation[3] * transform.scale;
    matrix[2] = transform.rotation[6] * transform.scale;
    matrix[4] = transform.rotation[1] * transform.scale;
    matrix[5] = transform.rotation[4] * transform.scale;
    matrix[6] = transform.rotation[7] * transform.scale;
    matrix[8] = transform.rotation[2] * transform.scale;
    matrix[9] = transform.rotation[5] * transform.scale;
    matrix[10] = transform.rotation[8] * transform.scale;
    matrix[3] = transform.translation[0];
    matrix[7] = transform.translation[1];
    matrix[11] = transform.translation[2];
    return matrix;
}

bool avObjectIsHidden(const NifAvObject& avObject) {
    return (avObject.flags & 0x0001u) != 0u;
}

bool equalsIgnoreCaseAscii(std::string_view lhs, std::string_view rhs) {
    if (lhs.size() != rhs.size()) {
        return false;
    }
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        const unsigned char lhsChar = static_cast<unsigned char>(lhs[i]);
        const unsigned char rhsChar = static_cast<unsigned char>(rhs[i]);
        if (std::tolower(lhsChar) != std::tolower(rhsChar)) {
            return false;
        }
    }
    return true;
}

std::vector<std::int32_t> readRefList(NifCursor& cursor) {
    const std::uint32_t count = cursor.readValue<std::uint32_t>();
    std::vector<std::int32_t> refs;
    refs.reserve(count);
    for (std::uint32_t i = 0; i < count; ++i) {
        refs.push_back(cursor.readValue<std::int32_t>());
    }
    return refs;
}

void skipBoundingVolume(NifCursor& cursor) {
    const std::uint32_t type = cursor.readValue<std::uint32_t>();
    switch (type) {
        case 0u:
            return;
        case 1u:
            cursor.skip(sizeof(float) * 4u);
            return;
        case 2u:
            cursor.skip(sizeof(float) * 15u);
            return;
        case 3u:
            cursor.skip(sizeof(float) * 8u);
            return;
        case 4u:
            cursor.skip(sizeof(float) * 10u);
            return;
        case 5u: {
            const std::uint32_t count = cursor.readValue<std::uint32_t>();
            cursor.skip(sizeof(std::int32_t) * count);
            return;
        }
        case 6u:
            cursor.skip(sizeof(float) * 4u);
            return;
        default:
            throw std::runtime_error("Unsupported bounding volume type " + std::to_string(type));
    }
}

std::string readObjectNetName(NifCursor& cursor) {
    const std::string name = cursor.readSizedString();
    cursor.readValue<std::int32_t>();
    cursor.readValue<std::int32_t>();
    return name;
}

void skipExtraRecordHeader(NifCursor& cursor) {
    cursor.readValue<std::int32_t>();
    cursor.readValue<std::uint32_t>();
}

std::uint32_t skipKeyMap(
    NifCursor& cursor,
    std::size_t valueComponentCount,
    bool quaternionValues = false,
    bool readInterpolationWhenEmpty = false
) {
    const std::uint32_t keyCount = cursor.readValue<std::uint32_t>();
    if (keyCount == 0u && !readInterpolationWhenEmpty) {
        return 0u;
    }

    const std::uint32_t interpolationType = cursor.readValue<std::uint32_t>();
    switch (interpolationType) {
        case 1u:  // Linear
        case 5u:  // Constant
            cursor.skip(static_cast<std::size_t>(keyCount) * (1u + valueComponentCount) * sizeof(float));
            return interpolationType;
        case 2u: {  // Quadratic
            const std::size_t tangentComponentCount = quaternionValues ? 0u : valueComponentCount * 2u;
            cursor.skip(static_cast<std::size_t>(keyCount) *
                        (1u + valueComponentCount + tangentComponentCount) * sizeof(float));
            return interpolationType;
        }
        case 3u:  // TCB
            cursor.skip(static_cast<std::size_t>(keyCount) * (1u + valueComponentCount + 3u) * sizeof(float));
            return interpolationType;
        case 4u:  // XYZ, caller consumes follow-up key maps.
            return interpolationType;
        default:
            throw std::runtime_error("Unsupported NIF key interpolation type " + std::to_string(interpolationType));
    }
}

void skipTimeControllerRecord(NifCursor& cursor) {
    cursor.readValue<std::int32_t>();
    cursor.readValue<std::uint16_t>();
    cursor.skip(sizeof(float) * 4u);
    cursor.readValue<std::int32_t>();
}

NifAvObject readAvObject(NifCursor& cursor) {
    NifAvObject object{};
    object.name = readObjectNetName(cursor);
    object.flags = cursor.readValue<std::uint16_t>();
    for (float& value : object.transform.translation) {
        value = cursor.readValue<float>();
    }
    for (float& value : object.transform.rotation) {
        value = cursor.readValue<float>();
    }
    object.transform.scale = cursor.readValue<float>();
    cursor.skip(sizeof(float) * 3u);
    object.propertyRefs = readRefList(cursor);
    if (cursor.readBool()) {
        skipBoundingVolume(cursor);
    }
    return object;
}

NifNodeRecord readNodeRecord(NifCursor& cursor) {
    NifNodeRecord node{};
    node.avObject = readAvObject(cursor);
    node.childRefs = readRefList(cursor);
    readRefList(cursor);
    return node;
}

NifGeometryRecord readGeometryRecord(NifCursor& cursor) {
    NifGeometryRecord geometry{};
    geometry.avObject = readAvObject(cursor);
    geometry.dataRef = cursor.readValue<std::int32_t>();
    cursor.readValue<std::int32_t>();
    return geometry;
}

NifTriShapeDataRecord readTriShapeDataRecord(NifCursor& cursor) {
    NifTriShapeDataRecord data{};
    const std::uint16_t vertexCount = cursor.readValue<std::uint16_t>();
    if (cursor.readBool()) {
        data.vertices.resize(vertexCount);
        for (auto& vertex : data.vertices) {
            vertex[0] = cursor.readValue<float>();
            vertex[1] = cursor.readValue<float>();
            vertex[2] = cursor.readValue<float>();
        }
    }
    if (cursor.readBool()) {
        data.normals.resize(vertexCount);
        for (auto& normal : data.normals) {
            normal[0] = cursor.readValue<float>();
            normal[1] = cursor.readValue<float>();
            normal[2] = cursor.readValue<float>();
        }
    }
    cursor.skip(sizeof(float) * 4u);
    if (cursor.readBool()) {
        cursor.skip(sizeof(float) * 4u * vertexCount);
    }
    const std::uint16_t dataFlags = cursor.readValue<std::uint16_t>();
    std::uint16_t uvSetCount = dataFlags;
    // Morrowind-era NIFs store the UV-set count in the flags field and then follow it
    // with a separate "has UVs" int32 flag. OpenMW consumes that extra flag before the
    // UV stream, and skipping it here keeps the vertex UV data byte-aligned.
    if (!cursor.readBool()) {
        uvSetCount = 0u;
    }
    data.uv0.reserve(vertexCount);
    for (std::uint16_t uvSet = 0; uvSet < uvSetCount; ++uvSet) {
        for (std::uint16_t vertexIndex = 0; vertexIndex < vertexCount; ++vertexIndex) {
            std::array<float, 2> uv{
                cursor.readValue<float>(),
                cursor.readValue<float>()
            };
            if (uvSet == 0u) {
                data.uv0.push_back(uv);
            }
        }
    }

    const std::size_t triangleHeaderOffset = cursor.offset;
    std::uint32_t indexCount = 0u;
    bool foundTriangleHeader = false;
    for (std::size_t shift = 0; shift <= 8u && (triangleHeaderOffset + shift + 6u) <= cursor.bytes.size(); shift += 2u) {
        std::uint16_t candidateTriangleCount = 0u;
        std::uint32_t candidateIndexCount = 0u;
        std::memcpy(&candidateTriangleCount, cursor.bytes.data() + triangleHeaderOffset + shift, sizeof(candidateTriangleCount));
        std::memcpy(&candidateIndexCount, cursor.bytes.data() + triangleHeaderOffset + shift + 2u, sizeof(candidateIndexCount));
        const std::size_t bytesRemaining = cursor.bytes.size() - (triangleHeaderOffset + shift + 6u);
        if (candidateTriangleCount == 0u || candidateIndexCount == 0u) {
            continue;
        }
        if (candidateIndexCount != static_cast<std::uint32_t>(candidateTriangleCount) * 3u) {
            continue;
        }
        if (bytesRemaining < static_cast<std::size_t>(candidateIndexCount) * sizeof(std::uint16_t)) {
            continue;
        }
        cursor.offset = triangleHeaderOffset + shift;
        indexCount = candidateIndexCount;
        foundTriangleHeader = true;
        break;
    }
    if (!foundTriangleHeader) {
        throw std::runtime_error("Failed to locate NiTriShapeData triangle header");
    }

    cursor.readValue<std::uint16_t>();
    cursor.readValue<std::uint32_t>();
    data.indices.resize(indexCount);
    for (std::uint16_t& index : data.indices) {
        index = cursor.readValue<std::uint16_t>();
    }

    const std::uint16_t matchGroupCount = cursor.readValue<std::uint16_t>();
    for (std::uint16_t i = 0; i < matchGroupCount; ++i) {
        const std::uint16_t matchCount = cursor.readValue<std::uint16_t>();
        cursor.skip(sizeof(std::uint16_t) * matchCount);
    }
    return data;
}

NifTriShapeDataRecord readTriStripsDataRecord(NifCursor& cursor) {
    NifTriShapeDataRecord data{};
    const std::uint16_t vertexCount = cursor.readValue<std::uint16_t>();
    if (cursor.readBool()) {
        data.vertices.resize(vertexCount);
        for (auto& vertex : data.vertices) {
            vertex[0] = cursor.readValue<float>();
            vertex[1] = cursor.readValue<float>();
            vertex[2] = cursor.readValue<float>();
        }
    }
    if (cursor.readBool()) {
        data.normals.resize(vertexCount);
        for (auto& normal : data.normals) {
            normal[0] = cursor.readValue<float>();
            normal[1] = cursor.readValue<float>();
            normal[2] = cursor.readValue<float>();
        }
    }
    cursor.skip(sizeof(float) * 4u);
    if (cursor.readBool()) {
        cursor.skip(sizeof(float) * 4u * vertexCount);
    }
    const std::uint16_t dataFlags = cursor.readValue<std::uint16_t>();
    std::uint16_t uvSetCount = dataFlags;
    // Morrowind 4.0.0.2 geometry carries the same "has UVs" int32 after the UV flags.
    if (!cursor.readBool()) {
        uvSetCount = 0u;
    }
    data.uv0.reserve(vertexCount);
    for (std::uint16_t uvSet = 0; uvSet < uvSetCount; ++uvSet) {
        for (std::uint16_t vertexIndex = 0; vertexIndex < vertexCount; ++vertexIndex) {
            std::array<float, 2> uv{
                cursor.readValue<float>(),
                cursor.readValue<float>()
            };
            if (uvSet == 0u) {
                data.uv0.push_back(uv);
            }
        }
    }

    cursor.readValue<std::uint16_t>();
    const std::uint16_t stripCount = cursor.readValue<std::uint16_t>();
    std::vector<std::uint16_t> stripLengths(stripCount, 0u);
    for (std::uint16_t& length : stripLengths) {
        length = cursor.readValue<std::uint16_t>();
    }
    for (const std::uint16_t stripLength : stripLengths) {
        std::vector<std::uint16_t> strip(stripLength, 0u);
        for (std::uint16_t& index : strip) {
            index = cursor.readValue<std::uint16_t>();
        }
        if (strip.size() < 3u) {
            continue;
        }
        for (std::size_t i = 2; i < strip.size(); ++i) {
            const std::uint16_t a = strip[i - 2];
            const std::uint16_t b = strip[i - 1];
            const std::uint16_t c = strip[i];
            if (a == b || b == c || a == c) {
                continue;
            }
            if ((i % 2u) == 0u) {
                data.indices.push_back(a);
                data.indices.push_back(b);
                data.indices.push_back(c);
            } else {
                data.indices.push_back(a);
                data.indices.push_back(c);
                data.indices.push_back(b);
            }
        }
    }
    return data;
}

NifTexturingPropertyRecord readTexturingPropertyRecord(NifCursor& cursor) {
    NifTexturingPropertyRecord property{};
    readObjectNetName(cursor);
    cursor.readValue<std::uint16_t>();
    cursor.readValue<std::uint32_t>();
    const std::uint32_t textureCount = cursor.readValue<std::uint32_t>();
    for (std::uint32_t i = 0; i < textureCount; ++i) {
        const bool enabled = cursor.readBool();
        if (!enabled) {
            continue;
        }
        const std::int32_t sourceTextureRef = cursor.readValue<std::int32_t>();
        cursor.readValue<std::uint32_t>();
        cursor.readValue<std::uint32_t>();
        cursor.readValue<std::uint32_t>();
        cursor.skip(4u);
        cursor.skip(2u);
        if (i == 0u) {
            property.baseTextureSourceRef = sourceTextureRef;
        }
    }
    return property;
}

NifSourceTextureRecord readSourceTextureRecord(NifCursor& cursor) {
    NifSourceTextureRecord texture{};
    readObjectNetName(cursor);
    const std::uint8_t external = cursor.readValue<std::uint8_t>();
    bool hasData = false;
    if (external == 0u) {
        hasData = cursor.readValue<std::uint8_t>() != 0u;
    }
    if (external != 0u) {
        texture.filePath = cursor.readSizedString();
    }
    if (hasData) {
        cursor.readValue<std::int32_t>();
    }
    cursor.readValue<std::uint32_t>();
    cursor.readValue<std::uint32_t>();
    cursor.readValue<std::uint32_t>();
    cursor.readValue<std::uint8_t>();
    return texture;
}

NifAlphaPropertyRecord readAlphaPropertyRecord(NifCursor& cursor) {
    NifAlphaPropertyRecord property{};
    readObjectNetName(cursor);
    const std::uint16_t flags = cursor.readValue<std::uint16_t>();
    cursor.readValue<std::uint8_t>();
    property.alphaTest = (flags & 0x0201u) != 0u;
    return property;
}

void skipTextKeyExtraDataRecord(NifCursor& cursor) {
    skipExtraRecordHeader(cursor);
    const std::uint32_t keyCount = cursor.readValue<std::uint32_t>();
    for (std::uint32_t i = 0; i < keyCount; ++i) {
        cursor.readValue<float>();
        cursor.readSizedString();
    }
}

void skipKeyframeControllerRecord(NifCursor& cursor) {
    skipTimeControllerRecord(cursor);
    cursor.readValue<std::int32_t>();
}

void skipGeomMorpherControllerRecord(NifCursor& cursor) {
    skipTimeControllerRecord(cursor);
    cursor.readValue<std::int32_t>();
    cursor.readValue<std::uint8_t>();
}

void skipMorphDataRecord(NifCursor& cursor) {
    const std::uint32_t morphCount = cursor.readValue<std::uint32_t>();
    const std::uint32_t vertexCount = cursor.readValue<std::uint32_t>();
    cursor.readValue<std::uint8_t>();
    for (std::uint32_t morphIndex = 0; morphIndex < morphCount; ++morphIndex) {
        skipKeyMap(cursor, 1u, false, true);
        cursor.skip(static_cast<std::size_t>(vertexCount) * sizeof(float) * 3u);
    }
}

void skipParticleSystemControllerRecord(NifCursor& cursor) {
    skipTimeControllerRecord(cursor);
    cursor.skip(sizeof(float) * 6u);
    cursor.skip(sizeof(float) * 3u);
    cursor.skip(sizeof(float) * 4u);
    cursor.skip(sizeof(float) * 3u);
    cursor.skip(sizeof(std::uint8_t));
    cursor.skip(sizeof(float) * 3u);
    cursor.skip(sizeof(std::uint16_t));
    cursor.skip(sizeof(float) * 3u);
    cursor.readValue<std::int32_t>();
    cursor.skip(sizeof(std::uint16_t));
    cursor.skip(sizeof(float));
    cursor.skip(sizeof(std::uint16_t));
    cursor.skip(sizeof(float) * 2u);
    const std::uint16_t numParticles = cursor.readValue<std::uint16_t>();
    cursor.skip(sizeof(std::uint16_t));
    const std::size_t perParticleBytes = (sizeof(float) * 9u) + (sizeof(std::uint16_t) * 2u);
    cursor.skip(static_cast<std::size_t>(numParticles) * perParticleBytes);
    cursor.readValue<std::int32_t>();
    cursor.readValue<std::int32_t>();
    cursor.readValue<std::int32_t>();
    cursor.readValue<std::uint8_t>();
}

void skipParticleModifierRecord(NifCursor& cursor, std::size_t payloadBytes) {
    cursor.readValue<std::int32_t>();
    cursor.readValue<std::int32_t>();
    cursor.skip(payloadBytes);
}

void skipColorDataRecord(NifCursor& cursor) {
    skipKeyMap(cursor, 4u);
}

void skipKeyframeDataRecord(NifCursor& cursor) {
    const std::uint32_t rotationInterpolation = skipKeyMap(cursor, 4u, true);
    if (rotationInterpolation == 4u) {
        cursor.readValue<std::uint32_t>();
        skipKeyMap(cursor, 1u);
        skipKeyMap(cursor, 1u);
        skipKeyMap(cursor, 1u);
    }
    skipKeyMap(cursor, 3u);
    skipKeyMap(cursor, 1u);
}

void skipGeometryDataPrefix(
    NifCursor& cursor,
    std::uint16_t& outVertexCount
) {
    outVertexCount = cursor.readValue<std::uint16_t>();
    if (cursor.readBool()) {
        cursor.skip(static_cast<std::size_t>(outVertexCount) * sizeof(float) * 3u);
    }
    if (cursor.readBool()) {
        cursor.skip(static_cast<std::size_t>(outVertexCount) * sizeof(float) * 3u);
    }
    cursor.skip(sizeof(float) * 4u);
    if (cursor.readBool()) {
        cursor.skip(sizeof(float) * 4u * outVertexCount);
    }
    std::uint16_t uvSetCount = cursor.readValue<std::uint16_t>();
    if (!cursor.readBool()) {
        uvSetCount = 0u;
    }
    cursor.skip(static_cast<std::size_t>(uvSetCount) * outVertexCount * sizeof(float) * 2u);
}

void skipRotatingParticlesDataRecord(NifCursor& cursor) {
    std::uint16_t vertexCount = 0u;
    skipGeometryDataPrefix(cursor, vertexCount);
    cursor.readValue<std::uint16_t>();
    cursor.skip(sizeof(float));
    cursor.readValue<std::uint16_t>();
    if (cursor.readBool()) {
        cursor.skip(static_cast<std::size_t>(vertexCount) * sizeof(float));
    }
    if (cursor.readBool()) {
        cursor.skip(static_cast<std::size_t>(vertexCount) * sizeof(float) * 4u);
    }
}

void skipSkinPartitionRecord(NifCursor& cursor) {
    const std::uint32_t partitionCount = cursor.readValue<std::uint32_t>();
    for (std::uint32_t partitionIndex = 0; partitionIndex < partitionCount; ++partitionIndex) {
        const std::uint16_t numVertices = cursor.readValue<std::uint16_t>();
        const std::uint16_t numTriangles = cursor.readValue<std::uint16_t>();
        const std::uint16_t numBones = cursor.readValue<std::uint16_t>();
        const std::uint16_t numStrips = cursor.readValue<std::uint16_t>();
        const std::uint16_t bonesPerVertex = cursor.readValue<std::uint16_t>();
        cursor.skip(static_cast<std::size_t>(numBones) * sizeof(std::uint16_t));
        cursor.skip(static_cast<std::size_t>(numVertices) * sizeof(std::uint16_t));
        cursor.skip(static_cast<std::size_t>(numVertices) * bonesPerVertex * sizeof(float));
        std::vector<std::uint16_t> stripLengths(numStrips, 0u);
        for (std::uint16_t& stripLength : stripLengths) {
            stripLength = cursor.readValue<std::uint16_t>();
        }
        if (numStrips != 0u) {
            for (const std::uint16_t stripLength : stripLengths) {
                cursor.skip(static_cast<std::size_t>(stripLength) * sizeof(std::uint16_t));
            }
        } else {
            cursor.skip(static_cast<std::size_t>(numTriangles) * 3u * sizeof(std::uint16_t));
        }
        if (cursor.readValue<std::uint8_t>() != 0u) {
            cursor.skip(static_cast<std::size_t>(numVertices) * bonesPerVertex * sizeof(char));
        }
    }
}

void skipSkinDataRecord(NifCursor& cursor) {
    cursor.skip((sizeof(float) * 13u));
    const std::uint32_t boneCount = cursor.readValue<std::uint32_t>();
    cursor.readValue<std::int32_t>();
    for (std::uint32_t boneIndex = 0; boneIndex < boneCount; ++boneIndex) {
        cursor.skip(sizeof(float) * 13u);
        cursor.skip(sizeof(float) * 4u);
        const std::uint16_t weightCount = cursor.readValue<std::uint16_t>();
        cursor.skip(static_cast<std::size_t>(weightCount) * (sizeof(std::uint16_t) + sizeof(float)));
    }
}

void skipSkinInstanceRecord(NifCursor& cursor) {
    cursor.readValue<std::int32_t>();
    cursor.readValue<std::int32_t>();
    readRefList(cursor);
}

void skipSimplePropertyRecord(NifCursor& cursor, std::size_t payloadBytes) {
    readObjectNetName(cursor);
    cursor.skip(payloadBytes);
}

void skipStringExtraDataRecord(NifCursor& cursor) {
    cursor.readValue<std::int32_t>();
    cursor.readValue<std::uint32_t>();
    cursor.readSizedString();
}

bool isFiniteVector(const std::array<float, 3>& value) {
    return std::isfinite(value[0]) && std::isfinite(value[1]) && std::isfinite(value[2]);
}

struct GeometryTraversalState {
    std::array<float, 16> parentTransform = identityMatrix();
    bool skipMeshes = false;
    bool discardRootTransform = false;
};

void appendGeometry(
    const std::vector<NifRecord>& records,
    std::int32_t geometryRecordIndex,
    const GeometryTraversalState& state,
    ImportedNifResult& outResult,
    bool& sawGeometry
) {
    if (geometryRecordIndex < 0 ||
        static_cast<std::size_t>(geometryRecordIndex) >= records.size()) {
        return;
    }
    const NifRecord& geometryRecord = records[static_cast<std::size_t>(geometryRecordIndex)];
    if (geometryRecord.kind == RecordKind::Node) {
        std::array<float, 16> localTransform = makeTransformMatrix(geometryRecord.node.avObject.transform);
        if (state.discardRootTransform &&
            !equalsIgnoreCaseAscii(geometryRecord.node.avObject.name, "bip01")) {
            localTransform = identityMatrix();
        }
        GeometryTraversalState childState = state;
        childState.parentTransform = multiplyMatrices(state.parentTransform, localTransform);
        childState.discardRootTransform = false;
        childState.skipMeshes = childState.skipMeshes ||
            geometryRecord.node.isRootCollisionNode ||
            avObjectIsHidden(geometryRecord.node.avObject);
        for (const std::int32_t childRef : geometryRecord.node.childRefs) {
            appendGeometry(records, childRef, childState, outResult, sawGeometry);
        }
        return;
    }
    if (geometryRecord.kind != RecordKind::Geometry || state.skipMeshes) {
        return;
    }

    const std::array<float, 16> localTransform = makeTransformMatrix(geometryRecord.geometry.avObject.transform);
    const std::array<float, 16> worldTransform = multiplyMatrices(state.parentTransform, localTransform);
    const std::int32_t dataRef = geometryRecord.geometry.dataRef;
    if (dataRef < 0 || static_cast<std::size_t>(dataRef) >= records.size()) {
        return;
    }
    const NifRecord& dataRecord = records[static_cast<std::size_t>(dataRef)];
    if (dataRecord.kind != RecordKind::TriShapeData || dataRecord.triShapeData.indices.empty() ||
        dataRecord.triShapeData.vertices.empty()) {
        return;
    }

    std::string diffuseTexturePath;
    bool alphaTest = false;
    for (const std::int32_t propertyRef : geometryRecord.geometry.avObject.propertyRefs) {
        if (propertyRef < 0 || static_cast<std::size_t>(propertyRef) >= records.size()) {
            continue;
        }
        const NifRecord& propertyRecord = records[static_cast<std::size_t>(propertyRef)];
        if (propertyRecord.kind == RecordKind::TexturingProperty) {
            const std::int32_t sourceRef = propertyRecord.texturingProperty.baseTextureSourceRef;
            if (sourceRef >= 0 && static_cast<std::size_t>(sourceRef) < records.size()) {
                const NifRecord& sourceRecord = records[static_cast<std::size_t>(sourceRef)];
                if (sourceRecord.kind == RecordKind::SourceTexture && !sourceRecord.sourceTexture.filePath.empty()) {
                    diffuseTexturePath = sourceRecord.sourceTexture.filePath;
                }
            }
        } else if (propertyRecord.kind == RecordKind::AlphaProperty) {
            alphaTest = alphaTest || propertyRecord.alphaProperty.alphaTest;
        }
    }
    if (outResult.diffuseTexturePath.empty() && !diffuseTexturePath.empty()) {
        outResult.diffuseTexturePath = diffuseTexturePath;
    }

    ImportedSceneMeshPart part{};
    part.firstIndex = static_cast<std::uint32_t>(outResult.mesh.indices.size());
    part.alphaTest = alphaTest;

    const std::uint32_t baseVertex = static_cast<std::uint32_t>(outResult.mesh.vertices.size());
    outResult.mesh.vertices.reserve(outResult.mesh.vertices.size() + dataRecord.triShapeData.vertices.size());
    for (std::size_t vertexIndex = 0; vertexIndex < dataRecord.triShapeData.vertices.size(); ++vertexIndex) {
        const auto& sourceVertex = dataRecord.triShapeData.vertices[vertexIndex];
        ImportedSceneVertex vertex{};
        const std::array<float, 3> transformedPosition = transformPoint(worldTransform, sourceVertex);
        if (!isFiniteVector(transformedPosition)) {
            continue;
        }
        const std::array<float, 3> enginePosition = toEngineSpace(transformedPosition);
        vertex.position[0] = enginePosition[0];
        vertex.position[1] = enginePosition[1];
        vertex.position[2] = enginePosition[2];

        std::array<float, 3> normal{0.0f, 1.0f, 0.0f};
        if (vertexIndex < dataRecord.triShapeData.normals.size()) {
            normal = transformVector(worldTransform, dataRecord.triShapeData.normals[vertexIndex]);
            const float normalLength = std::sqrt(
                normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
            if (normalLength > 0.00001f) {
                normal[0] /= normalLength;
                normal[1] /= normalLength;
                normal[2] /= normalLength;
            } else {
                normal = {0.0f, 1.0f, 0.0f};
            }
        }
        const std::array<float, 3> engineNormal = toEngineSpace(normal);
        vertex.normal[0] = engineNormal[0];
        vertex.normal[1] = engineNormal[1];
        vertex.normal[2] = engineNormal[2];

        if (vertexIndex < dataRecord.triShapeData.uv0.size()) {
            vertex.uv[0] = dataRecord.triShapeData.uv0[vertexIndex][0];
            vertex.uv[1] = dataRecord.triShapeData.uv0[vertexIndex][1];
        }
        outResult.mesh.vertices.push_back(vertex);
    }

    outResult.mesh.indices.reserve(outResult.mesh.indices.size() + dataRecord.triShapeData.indices.size());
    for (std::size_t indexOffset = 0; indexOffset + 2u < dataRecord.triShapeData.indices.size(); indexOffset += 3u) {
        const std::uint32_t i0 = baseVertex + static_cast<std::uint32_t>(dataRecord.triShapeData.indices[indexOffset + 0u]);
        const std::uint32_t i1 = baseVertex + static_cast<std::uint32_t>(dataRecord.triShapeData.indices[indexOffset + 1u]);
        const std::uint32_t i2 = baseVertex + static_cast<std::uint32_t>(dataRecord.triShapeData.indices[indexOffset + 2u]);
        // The Morrowind-to-engine axis remap changes handedness, so imported static triangles
        // need their winding flipped to keep face orientation consistent across main/prepass/shadow.
        outResult.mesh.indices.push_back(i0);
        outResult.mesh.indices.push_back(i2);
        outResult.mesh.indices.push_back(i1);
    }
    part.indexCount = static_cast<std::uint32_t>(outResult.mesh.indices.size()) - part.firstIndex;
    if (part.indexCount != 0u) {
        outResult.mesh.parts.push_back(part);
        sawGeometry = true;
    }
}

}  // namespace

bool loadMorrowindStaticNif(
    const std::filesystem::path& nifPath,
    ImportedNifResult& outResult,
    std::string& outError
) {
    outResult = ImportedNifResult{};
    outError.clear();

    std::ifstream input(nifPath, std::ios::binary);
    if (!input) {
        outError = "Failed to open NIF file";
        return false;
    }

    NifCursor cursor{};
    input.seekg(0, std::ios::end);
    const std::streamsize fileSize = input.tellg();
    if (fileSize <= 0) {
        outError = "NIF file is empty";
        return false;
    }
    input.seekg(0, std::ios::beg);
    cursor.bytes.resize(static_cast<std::size_t>(fileSize));
    input.read(reinterpret_cast<char*>(cursor.bytes.data()), fileSize);
    if (!input.good()) {
        outError = "Failed to read NIF file";
        return false;
    }

    std::string currentRecordType;
    std::uint32_t currentRecordIndex = 0u;

    try {
        if (cursor.remaining() < kExpectedHeader.size()) {
            throw std::runtime_error("Missing NIF header");
        }
        const std::string_view header(
            reinterpret_cast<const char*>(cursor.bytes.data()),
            kExpectedHeader.size());
        if (header != kExpectedHeader) {
            throw std::runtime_error("Unsupported NIF header");
        }
        cursor.offset += kExpectedHeader.size();

        const std::uint32_t version = cursor.readValue<std::uint32_t>();
        if (version != kExpectedVersion) {
            throw std::runtime_error("Unsupported NIF version");
        }

        const std::uint32_t recordCount = cursor.readValue<std::uint32_t>();
        std::vector<NifRecord> records(recordCount);
        for (std::uint32_t recordIndex = 0; recordIndex < recordCount; ++recordIndex) {
            currentRecordIndex = recordIndex;
            const std::string recordType = cursor.readSizedString();
            currentRecordType = recordType;
            NifRecord record{};
            if (recordType == "NiNode" || recordType == "RootCollisionNode" ||
                recordType == "NiBSAnimationNode" || recordType == "NiBSParticleNode" ||
                recordType == "NiSwitchNode" || recordType == "NiLODNode" ||
                recordType == "NiBillboardNode") {
                record.kind = RecordKind::Node;
                record.node = readNodeRecord(cursor);
                record.node.isRootCollisionNode = (recordType == "RootCollisionNode");
            } else if (recordType == "NiTriShape" || recordType == "NiParticles" ||
                       recordType == "NiRotatingParticles") {
                record.kind = RecordKind::Geometry;
                record.geometry = readGeometryRecord(cursor);
            } else if (recordType == "NiTriShapeData") {
                record.kind = RecordKind::TriShapeData;
                record.triShapeData = readTriShapeDataRecord(cursor);
            } else if (recordType == "NiTriStrips") {
                record.kind = RecordKind::Geometry;
                record.geometry = readGeometryRecord(cursor);
            } else if (recordType == "NiTriStripsData") {
                record.kind = RecordKind::TriShapeData;
                record.triShapeData = readTriStripsDataRecord(cursor);
            } else if (recordType == "NiRotatingParticlesData") {
                record.kind = RecordKind::ParticleData;
                skipRotatingParticlesDataRecord(cursor);
            } else if (recordType == "NiTexturingProperty") {
                record.kind = RecordKind::TexturingProperty;
                record.texturingProperty = readTexturingPropertyRecord(cursor);
            } else if (recordType == "NiSourceTexture") {
                record.kind = RecordKind::SourceTexture;
                record.sourceTexture = readSourceTextureRecord(cursor);
            } else if (recordType == "NiAlphaProperty") {
                record.kind = RecordKind::AlphaProperty;
                record.alphaProperty = readAlphaPropertyRecord(cursor);
            } else if (recordType == "NiStringExtraData") {
                skipStringExtraDataRecord(cursor);
            } else if (recordType == "NiTextKeyExtraData") {
                skipTextKeyExtraDataRecord(cursor);
            } else if (recordType == "NiKeyframeController") {
                skipKeyframeControllerRecord(cursor);
            } else if (recordType == "NiGeomMorpherController") {
                skipGeomMorpherControllerRecord(cursor);
            } else if (recordType == "NiKeyframeData") {
                skipKeyframeDataRecord(cursor);
            } else if (recordType == "NiMorphData") {
                skipMorphDataRecord(cursor);
            } else if (recordType == "NiParticleSystemController") {
                skipParticleSystemControllerRecord(cursor);
            } else if (recordType == "NiParticleGrowFade") {
                skipParticleModifierRecord(cursor, sizeof(float) * 2u);
            } else if (recordType == "NiParticleColorModifier") {
                cursor.readValue<std::int32_t>();
                cursor.readValue<std::int32_t>();
                cursor.readValue<std::int32_t>();
            } else if (recordType == "NiColorData") {
                skipColorDataRecord(cursor);
            } else if (recordType == "NiSkinInstance") {
                skipSkinInstanceRecord(cursor);
            } else if (recordType == "NiSkinData") {
                skipSkinDataRecord(cursor);
            } else if (recordType == "NiSkinPartition") {
                skipSkinPartitionRecord(cursor);
            } else if (recordType == "NiMaterialProperty") {
                skipSimplePropertyRecord(cursor, sizeof(std::uint16_t) + (sizeof(float) * 14u));
            } else if (recordType == "NiVertexColorProperty") {
                skipSimplePropertyRecord(cursor, sizeof(std::uint16_t) + sizeof(std::uint32_t) * 2u);
            } else if (recordType == "NiZBufferProperty" || recordType == "NiSpecularProperty" ||
                       recordType == "NiWireframeProperty") {
                skipSimplePropertyRecord(cursor, sizeof(std::uint16_t));
            } else {
                throw std::runtime_error("Unsupported NIF record type " + recordType);
            }
            records[recordIndex] = std::move(record);
        }

        outResult.mesh.name = nifPath.filename().string();
        bool sawGeometry = false;
        std::vector<std::int32_t> rootRefs;
        if (cursor.remaining() >= sizeof(std::uint32_t)) {
            const std::uint32_t rootCount = cursor.readValue<std::uint32_t>();
            if (rootCount > (cursor.remaining() / sizeof(std::int32_t))) {
                throw std::runtime_error("Invalid NIF root list");
            }
            rootRefs.reserve(rootCount);
            for (std::uint32_t rootIndex = 0; rootIndex < rootCount; ++rootIndex) {
                rootRefs.push_back(cursor.readValue<std::int32_t>());
            }
        }
        if (rootRefs.empty()) {
            rootRefs.push_back(0);
        }
        const bool discardSingleRootZeroTransform =
            (rootRefs.size() == 1u && rootRefs.front() == 0);
        for (const std::int32_t rootRef : rootRefs) {
            GeometryTraversalState traversalState{};
            traversalState.parentTransform = identityMatrix();
            traversalState.skipMeshes = false;
            traversalState.discardRootTransform = discardSingleRootZeroTransform;
            appendGeometry(records, rootRef, traversalState, outResult, sawGeometry);
        }
        if (!sawGeometry) {
            throw std::runtime_error("NIF did not contain supported static geometry");
        }
        return true;
    } catch (const std::exception& e) {
        outError = e.what();
        if (!currentRecordType.empty()) {
            outError = "record " + std::to_string(currentRecordIndex) + " (" + currentRecordType + "): " + outError;
        } else if (currentRecordIndex != 0u) {
            outError = "record " + std::to_string(currentRecordIndex) + ": " + outError;
        }
        outResult = ImportedNifResult{};
        return false;
    }
}

}  // namespace odai::importer
