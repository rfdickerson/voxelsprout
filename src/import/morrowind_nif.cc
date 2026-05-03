#include "import/morrowind_nif.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
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
    std::uint32_t activeChildIndex = std::numeric_limits<std::uint32_t>::max();
    bool traverseActiveChildOnly = false;
    bool isRootCollisionNode = false;
};

struct NifGeometryRecord {
    NifAvObject avObject;
    std::int32_t dataRef = -1;
    std::int32_t skinInstanceRef = -1;
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
    bool alphaBlend = false;
};

struct NifSkinInstanceRecord {
    std::int32_t skinDataRef = -1;
    std::int32_t skeletonRootRef = -1;
    std::vector<std::int32_t> boneRefs;
};

struct NifSkinDataRecord {
    struct VertexWeight {
        std::uint16_t vertexIndex = 0u;
        float weight = 0.0f;
    };

    struct BoneData {
        NifTransform transform;
        std::vector<VertexWeight> weights;
    };

    NifTransform skinTransform;
    std::vector<BoneData> bones;
};

struct NifTimeControllerRecord {
    std::int32_t nextControllerRef = -1;
    std::uint16_t flags = 0u;
    float frequency = 1.0f;
    float phase = 0.0f;
    float startTime = 0.0f;
    float stopTime = 0.0f;
    std::int32_t targetRef = -1;
};

struct NifKeyframeControllerRecord {
    NifTimeControllerRecord time;
    std::int32_t dataRef = -1;
};

struct NifKeyframeDataRecord {
    std::vector<ImportedNifVec3Key> translationKeys;
    std::vector<ImportedNifQuatKey> rotationKeys;
    std::vector<ImportedNifFloatKey> scaleKeys;
    std::vector<ImportedNifFloatKey> xRotationKeys;
    std::vector<ImportedNifFloatKey> yRotationKeys;
    std::vector<ImportedNifFloatKey> zRotationKeys;
};

struct NifTextKeyExtraDataRecord {
    std::vector<ImportedNifTextKey> keys;
};

enum class RecordKind {
    Unknown,
    Node,
    Geometry,
    TriShapeData,
    ParticleData,
    TexturingProperty,
    SourceTexture,
    AlphaProperty,
    SkinInstance,
    SkinData,
    KeyframeController,
    KeyframeData,
    TextKeyExtraData
};

struct NifRecord {
    RecordKind kind = RecordKind::Unknown;
    NifNodeRecord node{};
    NifGeometryRecord geometry{};
    NifTriShapeDataRecord triShapeData{};
    NifTexturingPropertyRecord texturingProperty{};
    NifSourceTextureRecord sourceTexture{};
    NifAlphaPropertyRecord alphaProperty{};
    NifSkinInstanceRecord skinInstance{};
    NifSkinDataRecord skinData{};
    NifKeyframeControllerRecord keyframeController{};
    NifKeyframeDataRecord keyframeData{};
    NifTextKeyExtraDataRecord textKeyExtraData{};
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

NifTransform readSkinTransform(NifCursor& cursor) {
    NifTransform transform{};
    for (float& value : transform.rotation) {
        value = cursor.readValue<float>();
    }
    for (float& value : transform.translation) {
        value = cursor.readValue<float>();
    }
    transform.scale = cursor.readValue<float>();
    return transform;
}

bool avObjectIsHidden(const NifAvObject& avObject) {
    return (avObject.flags & 0x0001u) != 0u;
}

std::string lowerAsciiCopy(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    for (const char ch : text) {
        out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
    }
    return out;
}

bool containsAnyFoliageAlphaHint(std::string_view text) {
    return text.find("flora") != std::string_view::npos ||
        text.find("leaf") != std::string_view::npos ||
        text.find("leaves") != std::string_view::npos ||
        text.find("fern") != std::string_view::npos ||
        text.find("grass") != std::string_view::npos ||
        text.find("vine") != std::string_view::npos ||
        text.find("ivy") != std::string_view::npos ||
        text.find("moss") != std::string_view::npos ||
        text.find("lilypad") != std::string_view::npos ||
        text.find("shrub") != std::string_view::npos ||
        text.find("branch") != std::string_view::npos ||
        text.find("heather") != std::string_view::npos ||
        text.find("chokeweed") != std::string_view::npos ||
        text.find("comberry") != std::string_view::npos ||
        text.find("corkbulb") != std::string_view::npos ||
        text.find("roobrush") != std::string_view::npos ||
        text.find("podplant") != std::string_view::npos;
}

bool alphaBlendOnlyPartCanUseCutoutFallback(
    std::string_view meshName,
    std::string_view geometryName,
    std::string_view diffuseTexturePath
) {
    const std::string combined =
        lowerAsciiCopy(meshName) + "|" +
        lowerAsciiCopy(geometryName) + "|" +
        lowerAsciiCopy(diffuseTexturePath);
    return containsAnyFoliageAlphaHint(combined);
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

bool nodeIsRootCollisionNode(const NifNodeRecord& node) {
    return node.isRootCollisionNode ||
        equalsIgnoreCaseAscii(node.avObject.name, "RootCollisionNode");
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

template <typename ReadValue>
std::uint32_t readKeyMapHeader(NifCursor& cursor, ReadValue readValues) {
    const std::uint32_t keyCount = cursor.readValue<std::uint32_t>();
    if (keyCount == 0u) {
        return 0u;
    }
    const std::uint32_t interpolationType = cursor.readValue<std::uint32_t>();
    for (std::uint32_t keyIndex = 0; keyIndex < keyCount; ++keyIndex) {
        const float time = cursor.readValue<float>();
        readValues(time, interpolationType);
    }
    return interpolationType;
}

std::vector<ImportedNifFloatKey> readFloatKeyMap(NifCursor& cursor) {
    std::vector<ImportedNifFloatKey> keys;
    const std::uint32_t keyCount = cursor.readValue<std::uint32_t>();
    if (keyCount == 0u) {
        return keys;
    }
    const std::uint32_t interpolationType = cursor.readValue<std::uint32_t>();
    keys.reserve(keyCount);
    for (std::uint32_t keyIndex = 0; keyIndex < keyCount; ++keyIndex) {
        ImportedNifFloatKey key{};
        key.time = cursor.readValue<float>();
        key.value = cursor.readValue<float>();
        if (interpolationType == 2u) {
            cursor.skip(sizeof(float) * 2u);
        } else if (interpolationType == 3u) {
            cursor.skip(sizeof(float) * 3u);
        } else if (interpolationType != 1u && interpolationType != 5u) {
            throw std::runtime_error("Unsupported NIF scalar key interpolation type " + std::to_string(interpolationType));
        }
        keys.push_back(key);
    }
    return keys;
}

std::vector<ImportedNifVec3Key> readVec3KeyMap(NifCursor& cursor) {
    std::vector<ImportedNifVec3Key> keys;
    const std::uint32_t keyCount = cursor.readValue<std::uint32_t>();
    if (keyCount == 0u) {
        return keys;
    }
    const std::uint32_t interpolationType = cursor.readValue<std::uint32_t>();
    keys.reserve(keyCount);
    for (std::uint32_t keyIndex = 0; keyIndex < keyCount; ++keyIndex) {
        ImportedNifVec3Key key{};
        key.time = cursor.readValue<float>();
        key.value[0] = cursor.readValue<float>();
        key.value[1] = cursor.readValue<float>();
        key.value[2] = cursor.readValue<float>();
        if (interpolationType == 2u) {
            cursor.skip(sizeof(float) * 6u);
        } else if (interpolationType == 3u) {
            cursor.skip(sizeof(float) * 3u);
        } else if (interpolationType != 1u && interpolationType != 5u) {
            throw std::runtime_error("Unsupported NIF vector key interpolation type " + std::to_string(interpolationType));
        }
        keys.push_back(key);
    }
    return keys;
}

std::vector<ImportedNifQuatKey> readQuatKeyMapBody(
    NifCursor& cursor,
    std::uint32_t keyCount,
    std::uint32_t interpolationType
) {
    std::vector<ImportedNifQuatKey> keys;
    keys.reserve(keyCount);
    for (std::uint32_t keyIndex = 0; keyIndex < keyCount; ++keyIndex) {
        ImportedNifQuatKey key{};
        key.time = cursor.readValue<float>();
        key.value[0] = cursor.readValue<float>();
        key.value[1] = cursor.readValue<float>();
        key.value[2] = cursor.readValue<float>();
        key.value[3] = cursor.readValue<float>();
        if (interpolationType == 3u) {
            cursor.skip(sizeof(float) * 3u);
        } else if (interpolationType != 1u && interpolationType != 2u && interpolationType != 5u) {
            throw std::runtime_error("Unsupported NIF quaternion key interpolation type " + std::to_string(interpolationType));
        }
        keys.push_back(key);
    }
    return keys;
}

void skipTimeControllerRecord(NifCursor& cursor) {
    cursor.readValue<std::int32_t>();
    cursor.readValue<std::uint16_t>();
    cursor.skip(sizeof(float) * 4u);
    cursor.readValue<std::int32_t>();
}

NifTimeControllerRecord readTimeControllerRecord(NifCursor& cursor) {
    NifTimeControllerRecord record{};
    record.nextControllerRef = cursor.readValue<std::int32_t>();
    record.flags = cursor.readValue<std::uint16_t>();
    record.frequency = cursor.readValue<float>();
    record.phase = cursor.readValue<float>();
    record.startTime = cursor.readValue<float>();
    record.stopTime = cursor.readValue<float>();
    record.targetRef = cursor.readValue<std::int32_t>();
    return record;
}

bool isKnownRecordType(std::string_view recordType) {
    return recordType == "NiNode" || recordType == "RootCollisionNode" ||
        recordType == "NiBSAnimationNode" || recordType == "NiBSParticleNode" ||
        recordType == "NiBillboardNode" || recordType == "NiSwitchNode" ||
        recordType == "NiLODNode" || recordType == "NiTriShape" ||
        recordType == "NiParticles" || recordType == "NiRotatingParticles" ||
        recordType == "NiTriShapeData" || recordType == "NiTriStrips" ||
        recordType == "NiTriStripsData" || recordType == "NiRotatingParticlesData" ||
        recordType == "NiTexturingProperty" || recordType == "NiSourceTexture" ||
        recordType == "NiAlphaProperty" || recordType == "NiStringExtraData" ||
        recordType == "NiTextKeyExtraData" || recordType == "NiKeyframeController" ||
        recordType == "NiGeomMorpherController" || recordType == "NiUVController" ||
        recordType == "NiKeyframeData" || recordType == "NiMorphData" ||
        recordType == "NiParticleSystemController" || recordType == "NiParticleGrowFade" ||
        recordType == "NiGravity" || recordType == "NiParticleColorModifier" ||
        recordType == "NiColorData" || recordType == "NiSkinInstance" ||
        recordType == "NiSkinData" || recordType == "NiSkinPartition" ||
        recordType == "NiMaterialProperty" || recordType == "NiVertexColorProperty" ||
        recordType == "NiZBufferProperty" || recordType == "NiSpecularProperty" ||
        recordType == "NiWireframeProperty";
}

bool looksLikeKnownRecordAt(const NifCursor& cursor, std::size_t offset) {
    if (offset + sizeof(std::uint32_t) > cursor.bytes.size()) {
        return false;
    }
    std::uint32_t size = 0u;
    std::memcpy(&size, cursor.bytes.data() + offset, sizeof(size));
    if (size == 0u || size > 64u || offset + sizeof(std::uint32_t) + size > cursor.bytes.size()) {
        return false;
    }
    const char* data = reinterpret_cast<const char*>(cursor.bytes.data() + offset + sizeof(std::uint32_t));
    for (std::uint32_t index = 0; index < size; ++index) {
        const unsigned char ch = static_cast<unsigned char>(data[index]);
        if (ch < 32u || ch > 126u) {
            return false;
        }
    }
    return isKnownRecordType(std::string_view(data, size));
}

bool resyncToNextKnownRecord(NifCursor& cursor, std::size_t maxScanBytes) {
    const std::size_t begin = cursor.offset;
    const std::size_t end = std::min(cursor.bytes.size(), begin + maxScanBytes);
    for (std::size_t offset = begin; offset < end; ++offset) {
        if (looksLikeKnownRecordAt(cursor, offset)) {
            cursor.offset = offset;
            return true;
        }
    }
    return false;
}

NifAvObject readAvObjectHeader(NifCursor& cursor) {
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
    return object;
}

void readAvObjectBounds(NifCursor& cursor) {
    if (cursor.readBool()) {
        skipBoundingVolume(cursor);
    }
}

NifAvObject readAvObject(NifCursor& cursor) {
    NifAvObject object = readAvObjectHeader(cursor);
    readAvObjectBounds(cursor);
    return object;
}

NifNodeRecord readNodeRecord(NifCursor& cursor) {
    NifNodeRecord node{};
    node.avObject = readAvObjectHeader(cursor);
    if (equalsIgnoreCaseAscii(node.avObject.name, "bounding box")) {
        if (!resyncToNextKnownRecord(cursor, 512u)) {
            throw std::runtime_error("Failed to resynchronize after NIF Bounding Box node");
        }
        node.isRootCollisionNode = true;
        return node;
    }
    readAvObjectBounds(cursor);
    node.childRefs = readRefList(cursor);
    readRefList(cursor);
    return node;
}

NifNodeRecord readSwitchNodeRecord(NifCursor& cursor) {
    NifNodeRecord node = readNodeRecord(cursor);
    node.activeChildIndex = cursor.readValue<std::uint32_t>();
    node.traverseActiveChildOnly = true;
    return node;
}

NifNodeRecord readLodNodeRecord(NifCursor& cursor) {
    NifNodeRecord node = readSwitchNodeRecord(cursor);
    node.activeChildIndex = 0u;
    node.traverseActiveChildOnly = true;
    cursor.skip(sizeof(float) * 3u);
    const std::uint32_t levelCount = cursor.readValue<std::uint32_t>();
    cursor.skip(static_cast<std::size_t>(levelCount) * sizeof(float) * 2u);
    return node;
}

NifGeometryRecord readGeometryRecord(NifCursor& cursor) {
    NifGeometryRecord geometry{};
    geometry.avObject = readAvObject(cursor);
    geometry.dataRef = cursor.readValue<std::int32_t>();
    geometry.skinInstanceRef = cursor.readValue<std::int32_t>();
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
    property.alphaBlend = (flags & 0x0001u) != 0u;
    property.alphaTest = (flags & 0x0200u) != 0u;
    return property;
}

NifTextKeyExtraDataRecord readTextKeyExtraDataRecord(NifCursor& cursor) {
    NifTextKeyExtraDataRecord record{};
    skipExtraRecordHeader(cursor);
    const std::uint32_t keyCount = cursor.readValue<std::uint32_t>();
    record.keys.reserve(keyCount);
    for (std::uint32_t i = 0; i < keyCount; ++i) {
        ImportedNifTextKey key{};
        key.time = cursor.readValue<float>();
        key.text = cursor.readSizedString();
        record.keys.push_back(std::move(key));
    }
    return record;
}

void skipKeyframeControllerRecord(NifCursor& cursor) {
    skipTimeControllerRecord(cursor);
    cursor.readValue<std::int32_t>();
}

NifKeyframeControllerRecord readKeyframeControllerRecord(NifCursor& cursor) {
    NifKeyframeControllerRecord record{};
    record.time = readTimeControllerRecord(cursor);
    record.dataRef = cursor.readValue<std::int32_t>();
    return record;
}

void skipGeomMorpherControllerRecord(NifCursor& cursor) {
    skipTimeControllerRecord(cursor);
    cursor.readValue<std::int32_t>();
    cursor.readValue<std::uint8_t>();
}

void skipUvControllerRecord(NifCursor& cursor) {
    skipTimeControllerRecord(cursor);
    cursor.readValue<std::uint16_t>();
    cursor.readValue<std::int32_t>();
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

void skipGravityRecord(NifCursor& cursor) {
    skipParticleModifierRecord(
        cursor,
        sizeof(float) * 2u + sizeof(std::uint32_t) + sizeof(float) * 6u);
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

NifKeyframeDataRecord readKeyframeDataRecord(NifCursor& cursor) {
    NifKeyframeDataRecord record{};
    const std::uint32_t rotationKeyCount = cursor.readValue<std::uint32_t>();
    if (rotationKeyCount != 0u) {
        const std::uint32_t rotationInterpolation = cursor.readValue<std::uint32_t>();
        if (rotationInterpolation == 4u) {
            record.xRotationKeys = readFloatKeyMap(cursor);
            record.yRotationKeys = readFloatKeyMap(cursor);
            record.zRotationKeys = readFloatKeyMap(cursor);
        } else {
            record.rotationKeys = readQuatKeyMapBody(cursor, rotationKeyCount, rotationInterpolation);
        }
    }
    record.translationKeys = readVec3KeyMap(cursor);
    record.scaleKeys = readFloatKeyMap(cursor);
    return record;
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

NifSkinDataRecord readSkinDataRecord(NifCursor& cursor) {
    NifSkinDataRecord record{};
    record.skinTransform = readSkinTransform(cursor);
    const std::uint32_t boneCount = cursor.readValue<std::uint32_t>();
    cursor.readValue<std::int32_t>();
    record.bones.reserve(boneCount);
    for (std::uint32_t boneIndex = 0; boneIndex < boneCount; ++boneIndex) {
        NifSkinDataRecord::BoneData bone{};
        bone.transform = readSkinTransform(cursor);
        cursor.skip(sizeof(float) * 4u);
        const std::uint16_t weightCount = cursor.readValue<std::uint16_t>();
        bone.weights.reserve(weightCount);
        for (std::uint16_t weightIndex = 0; weightIndex < weightCount; ++weightIndex) {
            NifSkinDataRecord::VertexWeight weight{};
            weight.vertexIndex = cursor.readValue<std::uint16_t>();
            weight.weight = cursor.readValue<float>();
            bone.weights.push_back(weight);
        }
        record.bones.push_back(std::move(bone));
    }
    return record;
}

NifSkinInstanceRecord readSkinInstanceRecord(NifCursor& cursor) {
    NifSkinInstanceRecord record{};
    record.skinDataRef = cursor.readValue<std::int32_t>();
    record.skeletonRootRef = cursor.readValue<std::int32_t>();
    record.boneRefs = readRefList(cursor);
    return record;
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
    const std::vector<std::array<float, 16>>* recordWorldTransforms = nullptr;
    const std::unordered_map<std::string, std::uint32_t>* canonicalBoneIndexByName = nullptr;
    std::vector<std::array<std::uint16_t, 4>>* outBoneIndices = nullptr;
    std::vector<std::array<float, 4>>* outBoneWeights = nullptr;
    std::uint32_t* weightedVertexCount = nullptr;
    std::uint32_t* unweightedVertexCount = nullptr;
    std::uint32_t* unknownBoneInfluenceCount = nullptr;
    std::uint32_t* droppedInfluenceCount = nullptr;
    bool skipMeshes = false;
    bool discardRootTransform = false;
    bool includeHiddenGeometry = false;
    bool keepAlphaBlendOnlyParts = false;
    bool bakeSkinBindPose = false;
};

void buildRecordWorldTransformsRecursive(
    const std::vector<NifRecord>& records,
    std::int32_t recordIndex,
    const std::array<float, 16>& parentTransform,
    bool discardRootTransform,
    std::vector<std::array<float, 16>>& outTransforms
) {
    if (recordIndex < 0 || static_cast<std::size_t>(recordIndex) >= records.size()) {
        return;
    }
    const NifRecord& record = records[static_cast<std::size_t>(recordIndex)];
    if (record.kind != RecordKind::Node && record.kind != RecordKind::Geometry) {
        return;
    }

    std::array<float, 16> localTransform = record.kind == RecordKind::Node
        ? makeTransformMatrix(record.node.avObject.transform)
        : makeTransformMatrix(record.geometry.avObject.transform);
    if (discardRootTransform &&
        record.kind == RecordKind::Node &&
        !equalsIgnoreCaseAscii(record.node.avObject.name, "bip01")) {
        localTransform = identityMatrix();
    }
    const std::array<float, 16> worldTransform = multiplyMatrices(parentTransform, localTransform);
    outTransforms[static_cast<std::size_t>(recordIndex)] = worldTransform;

    if (record.kind == RecordKind::Node) {
        for (const std::int32_t childRef : record.node.childRefs) {
            buildRecordWorldTransformsRecursive(records, childRef, worldTransform, false, outTransforms);
        }
    }
}

std::vector<std::array<float, 16>> buildRecordWorldTransforms(
    const std::vector<NifRecord>& records,
    std::span<const std::int32_t> rootRefs,
    bool discardSingleRootZeroTransform
) {
    std::vector<std::array<float, 16>> transforms(records.size(), identityMatrix());
    for (const std::int32_t rootRef : rootRefs) {
        buildRecordWorldTransformsRecursive(
            records,
            rootRef,
            identityMatrix(),
            discardSingleRootZeroTransform,
            transforms);
    }
    return transforms;
}

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
            nodeIsRootCollisionNode(geometryRecord.node) ||
            (!childState.includeHiddenGeometry && avObjectIsHidden(geometryRecord.node.avObject));
        if (geometryRecord.node.traverseActiveChildOnly) {
            const std::size_t activeChildIndex =
                static_cast<std::size_t>(geometryRecord.node.activeChildIndex);
            if (activeChildIndex < geometryRecord.node.childRefs.size()) {
                appendGeometry(
                    records,
                    geometryRecord.node.childRefs[activeChildIndex],
                    childState,
                    outResult,
                    sawGeometry);
            }
            return;
        }
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
    bool alphaBlend = false;
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
            alphaBlend = alphaBlend || propertyRecord.alphaProperty.alphaBlend;
        }
    }
    if (alphaBlend && !alphaTest) {
        // The imported static path is opaque/alpha-test only. Keep vanilla foliage
        // usable as cutouts, but drop blend-only overlays such as stains/decals
        // until imported statics have a dedicated transparent pass.
        if (state.keepAlphaBlendOnlyParts ||
            alphaBlendOnlyPartCanUseCutoutFallback(
                outResult.mesh.name,
                geometryRecord.geometry.avObject.name,
                diffuseTexturePath)) {
            alphaTest = true;
        } else {
            ++outResult.skippedAlphaBlendOnlyParts;
            return;
        }
    }
    if (outResult.diffuseTexturePath.empty() && !diffuseTexturePath.empty()) {
        outResult.diffuseTexturePath = diffuseTexturePath;
    }

    ImportedSceneMeshPart part{};
    part.firstIndex = static_cast<std::uint32_t>(outResult.mesh.indices.size());
    part.textureIndex = std::numeric_limits<std::uint32_t>::max();
    part.alphaTest = alphaTest;

    std::vector<std::array<float, 3>> bakedSkinPositions;
    std::vector<std::array<float, 3>> bakedSkinNormals;
    std::vector<float> bakedSkinWeights;
    if (state.bakeSkinBindPose &&
        state.recordWorldTransforms != nullptr &&
        geometryRecord.geometry.skinInstanceRef >= 0 &&
        static_cast<std::size_t>(geometryRecord.geometry.skinInstanceRef) < records.size()) {
        const NifRecord& skinInstanceRecord =
            records[static_cast<std::size_t>(geometryRecord.geometry.skinInstanceRef)];
        if (skinInstanceRecord.kind == RecordKind::SkinInstance &&
            skinInstanceRecord.skinInstance.skinDataRef >= 0 &&
            static_cast<std::size_t>(skinInstanceRecord.skinInstance.skinDataRef) < records.size()) {
            const NifRecord& skinDataRecord =
                records[static_cast<std::size_t>(skinInstanceRecord.skinInstance.skinDataRef)];
            if (skinDataRecord.kind == RecordKind::SkinData) {
                const std::size_t vertexCount = dataRecord.triShapeData.vertices.size();
                bakedSkinPositions.assign(vertexCount, {0.0f, 0.0f, 0.0f});
                bakedSkinNormals.assign(vertexCount, {0.0f, 0.0f, 0.0f});
                bakedSkinWeights.assign(vertexCount, 0.0f);
                const std::array<float, 16> skinMatrix =
                    makeTransformMatrix(skinDataRecord.skinData.skinTransform);
                const std::size_t boneCount = std::min(
                    skinDataRecord.skinData.bones.size(),
                    skinInstanceRecord.skinInstance.boneRefs.size());
                for (std::size_t boneIndex = 0; boneIndex < boneCount; ++boneIndex) {
                    const std::int32_t boneRef = skinInstanceRecord.skinInstance.boneRefs[boneIndex];
                    if (boneRef < 0 ||
                        static_cast<std::size_t>(boneRef) >= state.recordWorldTransforms->size()) {
                        continue;
                    }
                    const NifSkinDataRecord::BoneData& bone = skinDataRecord.skinData.bones[boneIndex];
                    // NiSkinData stores each bone transform as the inverse bind
                    // matrix. Match OpenMW's solve order:
                    // inverseBind * currentBoneWorld * skinTransform.
                    const std::array<float, 16> boneMatrix = multiplyMatrices(
                        multiplyMatrices(
                            makeTransformMatrix(bone.transform),
                            (*state.recordWorldTransforms)[static_cast<std::size_t>(boneRef)]),
                        skinMatrix);
                    for (const NifSkinDataRecord::VertexWeight& weight : bone.weights) {
                        if (weight.vertexIndex >= vertexCount || weight.weight <= 0.0f) {
                            continue;
                        }
                        const std::size_t vertexIndex = weight.vertexIndex;
                        const std::array<float, 3> position =
                            transformPoint(boneMatrix, dataRecord.triShapeData.vertices[vertexIndex]);
                        std::array<float, 3> normal{0.0f, 1.0f, 0.0f};
                        if (vertexIndex < dataRecord.triShapeData.normals.size()) {
                            normal = transformVector(boneMatrix, dataRecord.triShapeData.normals[vertexIndex]);
                        }
                        bakedSkinPositions[vertexIndex][0] += position[0] * weight.weight;
                        bakedSkinPositions[vertexIndex][1] += position[1] * weight.weight;
                        bakedSkinPositions[vertexIndex][2] += position[2] * weight.weight;
                        bakedSkinNormals[vertexIndex][0] += normal[0] * weight.weight;
                        bakedSkinNormals[vertexIndex][1] += normal[1] * weight.weight;
                        bakedSkinNormals[vertexIndex][2] += normal[2] * weight.weight;
                        bakedSkinWeights[vertexIndex] += weight.weight;
                    }
                }
            }
        }
    }

    std::vector<std::array<std::uint16_t, 4>> sourceBoneIndices;
    std::vector<std::array<float, 4>> sourceBoneWeights;
    if (state.outBoneIndices != nullptr && state.outBoneWeights != nullptr) {
        const std::size_t vertexCount = dataRecord.triShapeData.vertices.size();
        std::vector<std::vector<ImportedBoneInfluence>> allInfluences(vertexCount);
        if (state.canonicalBoneIndexByName != nullptr &&
            geometryRecord.geometry.skinInstanceRef >= 0 &&
            static_cast<std::size_t>(geometryRecord.geometry.skinInstanceRef) < records.size()) {
            const NifRecord& skinInstanceRecord =
                records[static_cast<std::size_t>(geometryRecord.geometry.skinInstanceRef)];
            if (skinInstanceRecord.kind == RecordKind::SkinInstance &&
                skinInstanceRecord.skinInstance.skinDataRef >= 0 &&
                static_cast<std::size_t>(skinInstanceRecord.skinInstance.skinDataRef) < records.size()) {
                const NifRecord& skinDataRecord =
                    records[static_cast<std::size_t>(skinInstanceRecord.skinInstance.skinDataRef)];
                if (skinDataRecord.kind == RecordKind::SkinData) {
                    const std::size_t boneCount = std::min(
                        skinDataRecord.skinData.bones.size(),
                        skinInstanceRecord.skinInstance.boneRefs.size());
                    for (std::size_t skinBoneIndex = 0; skinBoneIndex < boneCount; ++skinBoneIndex) {
                        const std::int32_t boneRef = skinInstanceRecord.skinInstance.boneRefs[skinBoneIndex];
                        if (boneRef < 0 || static_cast<std::size_t>(boneRef) >= records.size()) {
                            continue;
                        }
                        const NifRecord& boneRecord = records[static_cast<std::size_t>(boneRef)];
                        const std::string boneName =
                            boneRecord.kind == RecordKind::Node ? boneRecord.node.avObject.name : std::string{};
                        const auto foundBone = state.canonicalBoneIndexByName->find(lowerAsciiCopy(boneName));
                        if (foundBone == state.canonicalBoneIndexByName->end()) {
                            if (state.unknownBoneInfluenceCount != nullptr) {
                                *state.unknownBoneInfluenceCount += static_cast<std::uint32_t>(
                                    skinDataRecord.skinData.bones[skinBoneIndex].weights.size());
                            }
                            continue;
                        }
                        const std::uint32_t canonicalBoneIndex = foundBone->second;
                        if (canonicalBoneIndex > std::numeric_limits<std::uint16_t>::max()) {
                            continue;
                        }
                        for (const NifSkinDataRecord::VertexWeight& weight :
                             skinDataRecord.skinData.bones[skinBoneIndex].weights) {
                            if (weight.vertexIndex >= vertexCount || weight.weight <= 0.0f) {
                                continue;
                            }
                            allInfluences[weight.vertexIndex].push_back(
                                ImportedBoneInfluence{
                                    canonicalBoneIndex,
                                    weight.weight
                                });
                        }
                    }
                }
            }
        }

        sourceBoneIndices.assign(vertexCount, {0u, 0u, 0u, 0u});
        sourceBoneWeights.assign(vertexCount, {0.0f, 0.0f, 0.0f, 0.0f});
        for (std::size_t vertexIndex = 0; vertexIndex < vertexCount; ++vertexIndex) {
            std::vector<ImportedBoneInfluence>& influences = allInfluences[vertexIndex];
            std::sort(
                influences.begin(),
                influences.end(),
                [](const ImportedBoneInfluence& lhs, const ImportedBoneInfluence& rhs) {
                    return lhs.weight > rhs.weight;
                });
            if (influences.size() > 4u && state.droppedInfluenceCount != nullptr) {
                *state.droppedInfluenceCount += static_cast<std::uint32_t>(influences.size() - 4u);
            }
            float totalWeight = 0.0f;
            const std::size_t keptInfluenceCount = std::min<std::size_t>(influences.size(), 4u);
            for (std::size_t influenceIndex = 0; influenceIndex < keptInfluenceCount; ++influenceIndex) {
                totalWeight += influences[influenceIndex].weight;
            }
            if (totalWeight > 0.00001f) {
                const float invWeight = 1.0f / totalWeight;
                for (std::size_t influenceIndex = 0; influenceIndex < keptInfluenceCount; ++influenceIndex) {
                    sourceBoneIndices[vertexIndex][influenceIndex] =
                        static_cast<std::uint16_t>(influences[influenceIndex].boneIndex);
                    sourceBoneWeights[vertexIndex][influenceIndex] =
                        influences[influenceIndex].weight * invWeight;
                }
                if (state.weightedVertexCount != nullptr) {
                    ++(*state.weightedVertexCount);
                }
            } else if (state.unweightedVertexCount != nullptr) {
                ++(*state.unweightedVertexCount);
            }
        }
    }

    const std::uint32_t baseVertex = static_cast<std::uint32_t>(outResult.mesh.vertices.size());
    outResult.mesh.vertices.reserve(outResult.mesh.vertices.size() + dataRecord.triShapeData.vertices.size());
    for (std::size_t vertexIndex = 0; vertexIndex < dataRecord.triShapeData.vertices.size(); ++vertexIndex) {
        const auto& sourceVertex = dataRecord.triShapeData.vertices[vertexIndex];
        ImportedSceneVertex vertex{};
        std::array<float, 3> transformedPosition = transformPoint(worldTransform, sourceVertex);
        if (vertexIndex < bakedSkinWeights.size() && bakedSkinWeights[vertexIndex] > 0.0001f) {
            const float invWeight = 1.0f / bakedSkinWeights[vertexIndex];
            transformedPosition = {
                bakedSkinPositions[vertexIndex][0] * invWeight,
                bakedSkinPositions[vertexIndex][1] * invWeight,
                bakedSkinPositions[vertexIndex][2] * invWeight
            };
        }
        if (!isFiniteVector(transformedPosition)) {
            continue;
        }
        const std::array<float, 3> enginePosition = toEngineSpace(transformedPosition);
        vertex.position[0] = enginePosition[0];
        vertex.position[1] = enginePosition[1];
        vertex.position[2] = enginePosition[2];

        std::array<float, 3> normal{0.0f, 1.0f, 0.0f};
        if (vertexIndex < bakedSkinWeights.size() && bakedSkinWeights[vertexIndex] > 0.0001f) {
            const float invWeight = 1.0f / bakedSkinWeights[vertexIndex];
            normal = {
                bakedSkinNormals[vertexIndex][0] * invWeight,
                bakedSkinNormals[vertexIndex][1] * invWeight,
                bakedSkinNormals[vertexIndex][2] * invWeight
            };
        } else if (vertexIndex < dataRecord.triShapeData.normals.size()) {
            normal = transformVector(worldTransform, dataRecord.triShapeData.normals[vertexIndex]);
        }
        const float normalLength = std::sqrt(
            normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
        if (normalLength > 0.00001f) {
            normal[0] /= normalLength;
            normal[1] /= normalLength;
            normal[2] /= normalLength;
        } else {
            normal = {0.0f, 1.0f, 0.0f};
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
        if (state.outBoneIndices != nullptr && state.outBoneWeights != nullptr) {
            if (vertexIndex < sourceBoneIndices.size()) {
                state.outBoneIndices->push_back(sourceBoneIndices[vertexIndex]);
                state.outBoneWeights->push_back(sourceBoneWeights[vertexIndex]);
            } else {
                state.outBoneIndices->push_back({0u, 0u, 0u, 0u});
                state.outBoneWeights->push_back({0.0f, 0.0f, 0.0f, 0.0f});
            }
        }
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
        outResult.partDiffuseTexturePaths.push_back(diffuseTexturePath);
        sawGeometry = true;
    }
}

void copyMatrixToFloatArray(const std::array<float, 16>& matrix, float out[16]) {
    std::copy(matrix.begin(), matrix.end(), out);
}

std::string trimAsciiWhitespace(std::string value) {
    while (!value.empty() && (value.back() == ' ' || value.back() == '\t' ||
                              value.back() == '\r' || value.back() == '\n')) {
        value.pop_back();
    }
    std::size_t first = 0u;
    while (first < value.size() && (value[first] == ' ' || value[first] == '\t' ||
                                    value[first] == '\r' || value[first] == '\n')) {
        ++first;
    }
    if (first != 0u) {
        value.erase(0u, first);
    }
    return value;
}

bool parseAnimationMarkerText(
    const std::string& text,
    std::string& outName,
    std::string& outMarker
) {
    const std::string lowerText = lowerAsciiCopy(text);
    const std::size_t colon = lowerText.find(':');
    if (colon == std::string::npos) {
        return false;
    }
    outName = trimAsciiWhitespace(lowerText.substr(0u, colon));
    outMarker = trimAsciiWhitespace(lowerText.substr(colon + 1u));
    return !outName.empty() && !outMarker.empty();
}

std::array<float, 16> inverseAffineMatrix(const std::array<float, 16>& matrix) {
    const float a00 = matrix[0];
    const float a01 = matrix[1];
    const float a02 = matrix[2];
    const float a10 = matrix[4];
    const float a11 = matrix[5];
    const float a12 = matrix[6];
    const float a20 = matrix[8];
    const float a21 = matrix[9];
    const float a22 = matrix[10];
    const float det =
        a00 * (a11 * a22 - a12 * a21) -
        a01 * (a10 * a22 - a12 * a20) +
        a02 * (a10 * a21 - a11 * a20);
    if (std::abs(det) <= 0.000001f) {
        return identityMatrix();
    }
    const float invDet = 1.0f / det;
    std::array<float, 16> out = identityMatrix();
    out[0] = (a11 * a22 - a12 * a21) * invDet;
    out[1] = (a02 * a21 - a01 * a22) * invDet;
    out[2] = (a01 * a12 - a02 * a11) * invDet;
    out[4] = (a12 * a20 - a10 * a22) * invDet;
    out[5] = (a00 * a22 - a02 * a20) * invDet;
    out[6] = (a02 * a10 - a00 * a12) * invDet;
    out[8] = (a10 * a21 - a11 * a20) * invDet;
    out[9] = (a01 * a20 - a00 * a21) * invDet;
    out[10] = (a00 * a11 - a01 * a10) * invDet;

    const float tx = matrix[3];
    const float ty = matrix[7];
    const float tz = matrix[11];
    out[3] = -(out[0] * tx + out[1] * ty + out[2] * tz);
    out[7] = -(out[4] * tx + out[5] * ty + out[6] * tz);
    out[11] = -(out[8] * tx + out[9] * ty + out[10] * tz);
    return out;
}

void appendAnimatedGeometry(
    const std::vector<NifRecord>& records,
    std::int32_t recordIndex,
    std::int32_t parentNodeIndex,
    const GeometryTraversalState& state,
    ImportedAnimatedNifResult& outResult,
    std::vector<std::int32_t>& recordToAnimatedNodeIndex,
    bool& sawGeometry
) {
    if (recordIndex < 0 || static_cast<std::size_t>(recordIndex) >= records.size()) {
        return;
    }
    const NifRecord& record = records[static_cast<std::size_t>(recordIndex)];
    if (record.kind == RecordKind::Node) {
        std::array<float, 16> localTransform = makeTransformMatrix(record.node.avObject.transform);
        if (state.discardRootTransform &&
            !equalsIgnoreCaseAscii(record.node.avObject.name, "bip01")) {
            localTransform = identityMatrix();
        }

        GeometryTraversalState childState = state;
        childState.discardRootTransform = false;
        childState.skipMeshes = childState.skipMeshes ||
            nodeIsRootCollisionNode(record.node) ||
            (!childState.includeHiddenGeometry && avObjectIsHidden(record.node.avObject));

        std::int32_t animatedNodeIndex = parentNodeIndex;
        if (!childState.skipMeshes) {
            ImportedAnimatedNifNode node{};
            node.name = record.node.avObject.name;
            node.parentIndex = parentNodeIndex;
            copyMatrixToFloatArray(localTransform, node.localTransform);
            animatedNodeIndex = static_cast<std::int32_t>(outResult.nodes.size());
            outResult.nodes.push_back(std::move(node));
            recordToAnimatedNodeIndex[static_cast<std::size_t>(recordIndex)] = animatedNodeIndex;
        }

        if (record.node.traverseActiveChildOnly) {
            const std::size_t activeChildIndex = static_cast<std::size_t>(record.node.activeChildIndex);
            if (activeChildIndex < record.node.childRefs.size()) {
                appendAnimatedGeometry(
                    records,
                    record.node.childRefs[activeChildIndex],
                    animatedNodeIndex,
                    childState,
                    outResult,
                    recordToAnimatedNodeIndex,
                    sawGeometry);
            }
            return;
        }
        for (const std::int32_t childRef : record.node.childRefs) {
            appendAnimatedGeometry(
                records,
                childRef,
                animatedNodeIndex,
                childState,
                outResult,
                recordToAnimatedNodeIndex,
                sawGeometry);
        }
        return;
    }
    if (record.kind != RecordKind::Geometry || state.skipMeshes || parentNodeIndex < 0) {
        return;
    }

    const std::array<float, 16> geometryTransform = makeTransformMatrix(record.geometry.avObject.transform);
    const std::int32_t dataRef = record.geometry.dataRef;
    if (dataRef < 0 || static_cast<std::size_t>(dataRef) >= records.size()) {
        return;
    }
    const NifRecord& dataRecord = records[static_cast<std::size_t>(dataRef)];
    if (dataRecord.kind != RecordKind::TriShapeData ||
        dataRecord.triShapeData.indices.empty() ||
        dataRecord.triShapeData.vertices.empty()) {
        return;
    }

    std::string diffuseTexturePath;
    bool alphaTest = false;
    bool alphaBlend = false;
    for (const std::int32_t propertyRef : record.geometry.avObject.propertyRefs) {
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
            alphaBlend = alphaBlend || propertyRecord.alphaProperty.alphaBlend;
        }
    }
    if (alphaBlend && !alphaTest) {
        if (state.keepAlphaBlendOnlyParts ||
            alphaBlendOnlyPartCanUseCutoutFallback(
                outResult.name,
                record.geometry.avObject.name,
                diffuseTexturePath)) {
            alphaTest = true;
        } else {
            ++outResult.skippedAlphaBlendOnlyParts;
            return;
        }
    }
    if (outResult.diffuseTexturePath.empty() && !diffuseTexturePath.empty()) {
        outResult.diffuseTexturePath = diffuseTexturePath;
    }

    ImportedAnimatedNifPart part{};
    part.firstIndex = static_cast<std::uint32_t>(outResult.indices.size());
    part.textureIndex = std::numeric_limits<std::uint32_t>::max();
    part.alphaTest = alphaTest;
    part.nodeIndex = static_cast<std::uint32_t>(parentNodeIndex);

    const std::uint32_t baseVertex = static_cast<std::uint32_t>(outResult.vertices.size());
    outResult.vertices.reserve(outResult.vertices.size() + dataRecord.triShapeData.vertices.size());
    for (std::size_t vertexIndex = 0; vertexIndex < dataRecord.triShapeData.vertices.size(); ++vertexIndex) {
        const auto& sourceVertex = dataRecord.triShapeData.vertices[vertexIndex];
        ImportedSceneVertex vertex{};
        const std::array<float, 3> localPosition = transformPoint(geometryTransform, sourceVertex);
        if (!isFiniteVector(localPosition)) {
            continue;
        }
        const std::array<float, 3> enginePosition = toEngineSpace(localPosition);
        vertex.position[0] = enginePosition[0];
        vertex.position[1] = enginePosition[1];
        vertex.position[2] = enginePosition[2];

        std::array<float, 3> normal{0.0f, 1.0f, 0.0f};
        if (vertexIndex < dataRecord.triShapeData.normals.size()) {
            normal = transformVector(geometryTransform, dataRecord.triShapeData.normals[vertexIndex]);
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
        outResult.vertices.push_back(vertex);
    }

    outResult.indices.reserve(outResult.indices.size() + dataRecord.triShapeData.indices.size());
    for (std::size_t indexOffset = 0; indexOffset + 2u < dataRecord.triShapeData.indices.size(); indexOffset += 3u) {
        const std::uint32_t i0 = baseVertex + static_cast<std::uint32_t>(dataRecord.triShapeData.indices[indexOffset + 0u]);
        const std::uint32_t i1 = baseVertex + static_cast<std::uint32_t>(dataRecord.triShapeData.indices[indexOffset + 1u]);
        const std::uint32_t i2 = baseVertex + static_cast<std::uint32_t>(dataRecord.triShapeData.indices[indexOffset + 2u]);
        outResult.indices.push_back(i0);
        outResult.indices.push_back(i2);
        outResult.indices.push_back(i1);
    }
    part.indexCount = static_cast<std::uint32_t>(outResult.indices.size()) - part.firstIndex;
    if (part.indexCount != 0u) {
        outResult.parts.push_back(part);
        outResult.partDiffuseTexturePaths.push_back(diffuseTexturePath);
        sawGeometry = true;
    }
}

struct StaticNifLoadOptions {
    bool includeHiddenGeometry = false;
    bool keepAlphaBlendOnlyParts = false;
    bool preserveRootTransform = false;
    bool bakeSkinBindPose = false;
    const std::unordered_map<std::string, std::uint32_t>* canonicalBoneIndexByName = nullptr;
    std::vector<std::array<std::uint16_t, 4>>* outBoneIndices = nullptr;
    std::vector<std::array<float, 4>>* outBoneWeights = nullptr;
    std::uint32_t* weightedVertexCount = nullptr;
    std::uint32_t* unweightedVertexCount = nullptr;
    std::uint32_t* unknownBoneInfluenceCount = nullptr;
    std::uint32_t* droppedInfluenceCount = nullptr;
};

bool loadMorrowindStaticNifWithOptions(
    const std::filesystem::path& nifPath,
    ImportedNifResult& outResult,
    std::string& outError,
    const StaticNifLoadOptions& options
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
                recordType == "NiBillboardNode") {
                record.kind = RecordKind::Node;
                record.node = readNodeRecord(cursor);
                record.node.isRootCollisionNode = (recordType == "RootCollisionNode");
            } else if (recordType == "NiSwitchNode") {
                record.kind = RecordKind::Node;
                record.node = readSwitchNodeRecord(cursor);
            } else if (recordType == "NiLODNode") {
                record.kind = RecordKind::Node;
                record.node = readLodNodeRecord(cursor);
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
                readTextKeyExtraDataRecord(cursor);
            } else if (recordType == "NiKeyframeController") {
                skipKeyframeControllerRecord(cursor);
            } else if (recordType == "NiGeomMorpherController") {
                skipGeomMorpherControllerRecord(cursor);
            } else if (recordType == "NiUVController") {
                skipUvControllerRecord(cursor);
            } else if (recordType == "NiKeyframeData") {
                skipKeyframeDataRecord(cursor);
            } else if (recordType == "NiMorphData") {
                skipMorphDataRecord(cursor);
            } else if (recordType == "NiParticleSystemController") {
                skipParticleSystemControllerRecord(cursor);
            } else if (recordType == "NiParticleGrowFade") {
                skipParticleModifierRecord(cursor, sizeof(float) * 2u);
            } else if (recordType == "NiGravity") {
                skipGravityRecord(cursor);
            } else if (recordType == "NiParticleColorModifier") {
                cursor.readValue<std::int32_t>();
                cursor.readValue<std::int32_t>();
                cursor.readValue<std::int32_t>();
            } else if (recordType == "NiColorData") {
                skipColorDataRecord(cursor);
            } else if (recordType == "NiSkinInstance") {
                record.kind = RecordKind::SkinInstance;
                record.skinInstance = readSkinInstanceRecord(cursor);
            } else if (recordType == "NiSkinData") {
                record.kind = RecordKind::SkinData;
                record.skinData = readSkinDataRecord(cursor);
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
        const std::vector<std::array<float, 16>> recordWorldTransforms =
            buildRecordWorldTransforms(
                records,
                rootRefs,
                discardSingleRootZeroTransform && !options.preserveRootTransform);
        for (const std::int32_t rootRef : rootRefs) {
            GeometryTraversalState traversalState{};
            traversalState.parentTransform = identityMatrix();
            traversalState.recordWorldTransforms = &recordWorldTransforms;
            traversalState.skipMeshes = false;
            traversalState.discardRootTransform = discardSingleRootZeroTransform && !options.preserveRootTransform;
            traversalState.includeHiddenGeometry = options.includeHiddenGeometry;
            traversalState.keepAlphaBlendOnlyParts = options.keepAlphaBlendOnlyParts;
            traversalState.bakeSkinBindPose = options.bakeSkinBindPose;
            traversalState.canonicalBoneIndexByName = options.canonicalBoneIndexByName;
            traversalState.outBoneIndices = options.outBoneIndices;
            traversalState.outBoneWeights = options.outBoneWeights;
            traversalState.weightedVertexCount = options.weightedVertexCount;
            traversalState.unweightedVertexCount = options.unweightedVertexCount;
            traversalState.unknownBoneInfluenceCount = options.unknownBoneInfluenceCount;
            traversalState.droppedInfluenceCount = options.droppedInfluenceCount;
            appendGeometry(records, rootRef, traversalState, outResult, sawGeometry);
        }
        if (!sawGeometry) {
            throw std::runtime_error("NIF did not contain supported static geometry");
        }
        return true;
    } catch (const std::exception& e) {
        outError = e.what();
        if (!currentRecordType.empty()) {
            outError = "record " + std::to_string(currentRecordIndex) + " (" + currentRecordType + ") @"
                + std::to_string(cursor.offset) + ": " + outError;
        } else if (currentRecordIndex != 0u) {
            outError = "record " + std::to_string(currentRecordIndex) + " @"
                + std::to_string(cursor.offset) + ": " + outError;
        }
        outResult = ImportedNifResult{};
        return false;
    }
}

}  // namespace

bool loadMorrowindStaticNif(
    const std::filesystem::path& nifPath,
    ImportedNifResult& outResult,
    std::string& outError
) {
    return loadMorrowindStaticNifWithOptions(nifPath, outResult, outError, StaticNifLoadOptions{});
}

bool loadMorrowindActorPartNif(
    const std::filesystem::path& nifPath,
    ImportedNifResult& outResult,
    std::string& outError
) {
    StaticNifLoadOptions options{};
    options.includeHiddenGeometry = false;
    options.keepAlphaBlendOnlyParts = true;
    options.preserveRootTransform = true;
    options.bakeSkinBindPose = true;
    return loadMorrowindStaticNifWithOptions(nifPath, outResult, outError, options);
}

bool loadMorrowindAnimatedNif(
    const std::filesystem::path& nifPath,
    ImportedAnimatedNifResult& outResult,
    std::string& outError
) {
    outResult = ImportedAnimatedNifResult{};
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
                recordType == "NiBillboardNode") {
                record.kind = RecordKind::Node;
                record.node = readNodeRecord(cursor);
                record.node.isRootCollisionNode = (recordType == "RootCollisionNode");
            } else if (recordType == "NiSwitchNode") {
                record.kind = RecordKind::Node;
                record.node = readSwitchNodeRecord(cursor);
            } else if (recordType == "NiLODNode") {
                record.kind = RecordKind::Node;
                record.node = readLodNodeRecord(cursor);
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
                record.kind = RecordKind::TextKeyExtraData;
                record.textKeyExtraData = readTextKeyExtraDataRecord(cursor);
            } else if (recordType == "NiKeyframeController") {
                record.kind = RecordKind::KeyframeController;
                record.keyframeController = readKeyframeControllerRecord(cursor);
            } else if (recordType == "NiGeomMorpherController") {
                skipGeomMorpherControllerRecord(cursor);
            } else if (recordType == "NiUVController") {
                skipUvControllerRecord(cursor);
            } else if (recordType == "NiKeyframeData") {
                record.kind = RecordKind::KeyframeData;
                record.keyframeData = readKeyframeDataRecord(cursor);
            } else if (recordType == "NiMorphData") {
                skipMorphDataRecord(cursor);
            } else if (recordType == "NiParticleSystemController") {
                skipParticleSystemControllerRecord(cursor);
            } else if (recordType == "NiParticleGrowFade") {
                skipParticleModifierRecord(cursor, sizeof(float) * 2u);
            } else if (recordType == "NiGravity") {
                skipGravityRecord(cursor);
            } else if (recordType == "NiParticleColorModifier") {
                cursor.readValue<std::int32_t>();
                cursor.readValue<std::int32_t>();
                cursor.readValue<std::int32_t>();
            } else if (recordType == "NiColorData") {
                skipColorDataRecord(cursor);
            } else if (recordType == "NiSkinInstance") {
                record.kind = RecordKind::SkinInstance;
                record.skinInstance = readSkinInstanceRecord(cursor);
            } else if (recordType == "NiSkinData") {
                record.kind = RecordKind::SkinData;
                record.skinData = readSkinDataRecord(cursor);
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

        outResult.name = nifPath.filename().string();
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

        std::vector<std::int32_t> recordToAnimatedNodeIndex(records.size(), -1);
        const bool discardSingleRootZeroTransform =
            (rootRefs.size() == 1u && rootRefs.front() == 0);
        for (const std::int32_t rootRef : rootRefs) {
            GeometryTraversalState traversalState{};
            traversalState.parentTransform = identityMatrix();
            traversalState.skipMeshes = false;
            traversalState.discardRootTransform = discardSingleRootZeroTransform;
            appendAnimatedGeometry(
                records,
                rootRef,
                -1,
                traversalState,
                outResult,
                recordToAnimatedNodeIndex,
                sawGeometry);
        }
        if (!sawGeometry && outResult.nodes.empty()) {
            throw std::runtime_error("NIF did not contain supported animated nodes or geometry");
        }

        for (const NifRecord& record : records) {
            if (record.kind == RecordKind::TextKeyExtraData) {
                outResult.textKeys.insert(
                    outResult.textKeys.end(),
                    record.textKeyExtraData.keys.begin(),
                    record.textKeyExtraData.keys.end());
            }
            if (record.kind != RecordKind::KeyframeController ||
                record.keyframeController.time.targetRef < 0 ||
                record.keyframeController.dataRef < 0 ||
                static_cast<std::size_t>(record.keyframeController.time.targetRef) >= recordToAnimatedNodeIndex.size() ||
                static_cast<std::size_t>(record.keyframeController.dataRef) >= records.size()) {
                continue;
            }
            const std::int32_t animatedNodeIndex =
                recordToAnimatedNodeIndex[static_cast<std::size_t>(record.keyframeController.time.targetRef)];
            const NifRecord& dataRecord = records[static_cast<std::size_t>(record.keyframeController.dataRef)];
            if (animatedNodeIndex < 0 || dataRecord.kind != RecordKind::KeyframeData) {
                continue;
            }
            ImportedNifNodeAnimation animation{};
            animation.nodeIndex = static_cast<std::uint32_t>(animatedNodeIndex);
            animation.startTime = record.keyframeController.time.startTime;
            animation.stopTime = record.keyframeController.time.stopTime;
            animation.frequency = record.keyframeController.time.frequency;
            animation.phase = record.keyframeController.time.phase;
            animation.translationKeys = dataRecord.keyframeData.translationKeys;
            animation.rotationKeys = dataRecord.keyframeData.rotationKeys;
            animation.scaleKeys = dataRecord.keyframeData.scaleKeys;
            animation.xRotationKeys = dataRecord.keyframeData.xRotationKeys;
            animation.yRotationKeys = dataRecord.keyframeData.yRotationKeys;
            animation.zRotationKeys = dataRecord.keyframeData.zRotationKeys;
            outResult.nodeAnimations.push_back(std::move(animation));
        }
        std::sort(
            outResult.textKeys.begin(),
            outResult.textKeys.end(),
            [](const ImportedNifTextKey& lhs, const ImportedNifTextKey& rhs) {
                return lhs.time < rhs.time;
            });
        return true;
    } catch (const std::exception& e) {
        outError = e.what();
        if (!currentRecordType.empty()) {
            outError = "record " + std::to_string(currentRecordIndex) + " (" + currentRecordType + ") @"
                + std::to_string(cursor.offset) + ": " + outError;
        } else if (currentRecordIndex != 0u) {
            outError = "record " + std::to_string(currentRecordIndex) + " @"
                + std::to_string(cursor.offset) + ": " + outError;
        }
        outResult = ImportedAnimatedNifResult{};
        return false;
    }
}

bool loadMorrowindSkinnedActorSkeleton(
    const std::filesystem::path& baseAnimPath,
    ImportedSkinnedActorAsset& outAsset,
    std::string& outError
) {
    ImportedAnimatedNifResult baseAnim{};
    if (!loadMorrowindAnimatedNif(baseAnimPath, baseAnim, outError)) {
        outAsset.skeleton.clear();
        outAsset.nodeAnimations.clear();
        outAsset.animationClips.clear();
        return false;
    }
    outAsset.skeleton.clear();
    outAsset.nodeAnimations = baseAnim.nodeAnimations;
    outAsset.animationClips.clear();
    outAsset.skeleton.reserve(baseAnim.nodes.size());
    for (const ImportedAnimatedNifNode& node : baseAnim.nodes) {
        ImportedSkeletonNode skeletonNode{};
        skeletonNode.name = node.name;
        skeletonNode.parentIndex = node.parentIndex;
        std::memcpy(skeletonNode.localTransform, node.localTransform, sizeof(skeletonNode.localTransform));
        outAsset.skeleton.push_back(std::move(skeletonNode));
    }
    std::vector<std::array<float, 16>> worldTransforms(outAsset.skeleton.size(), identityMatrix());
    for (std::size_t nodeIndex = 0; nodeIndex < outAsset.skeleton.size(); ++nodeIndex) {
        std::array<float, 16> local{};
        std::copy(
            outAsset.skeleton[nodeIndex].localTransform,
            outAsset.skeleton[nodeIndex].localTransform + 16,
            local.begin());
        const std::int32_t parentIndex = outAsset.skeleton[nodeIndex].parentIndex;
        if (parentIndex >= 0 && static_cast<std::size_t>(parentIndex) < worldTransforms.size()) {
            worldTransforms[nodeIndex] =
                multiplyMatrices(worldTransforms[static_cast<std::size_t>(parentIndex)], local);
        } else {
            worldTransforms[nodeIndex] = local;
        }
        const std::array<float, 16> inverseBind = inverseAffineMatrix(worldTransforms[nodeIndex]);
        copyMatrixToFloatArray(inverseBind, outAsset.skeleton[nodeIndex].inverseBindWorldTransform);
    }

    float startTime = std::numeric_limits<float>::max();
    float stopTime = 0.0f;
    for (const ImportedNifNodeAnimation& animation : outAsset.nodeAnimations) {
        startTime = std::min(startTime, animation.startTime);
        stopTime = std::max(stopTime, animation.stopTime);
    }
    std::unordered_map<std::string, float> pendingClipStarts;
    for (std::size_t keyIndex = 0; keyIndex < baseAnim.textKeys.size(); ++keyIndex) {
        const ImportedNifTextKey& key = baseAnim.textKeys[keyIndex];
        std::string name;
        std::string marker;
        if (!parseAnimationMarkerText(key.text, name, marker)) {
            continue;
        }
        if (marker == "start") {
            pendingClipStarts[name] = key.time;
        } else if (marker == "stop" || marker == "end") {
            const auto startIt = pendingClipStarts.find(name);
            if (startIt != pendingClipStarts.end() && key.time > startIt->second) {
                outAsset.animationClips.push_back(ImportedAnimationClip{name, startIt->second, key.time});
                pendingClipStarts.erase(startIt);
            }
        }
    }
    if (outAsset.animationClips.empty() && !outAsset.nodeAnimations.empty() && startTime < stopTime) {
        const float duration = stopTime - startTime;
        ImportedAnimationClip idleClip{};
        idleClip.name = "idle";
        idleClip.startTime = startTime;
        idleClip.stopTime = startTime + std::min(duration, 2.0f);
        outAsset.animationClips.push_back(idleClip);

        ImportedAnimationClip walkClip{};
        walkClip.name = "walk";
        walkClip.startTime = startTime + std::min(duration * 0.15f, 2.0f);
        walkClip.stopTime = stopTime;
        outAsset.animationClips.push_back(walkClip);
    }
    return !outAsset.skeleton.empty();
}

bool appendMorrowindSkinnedActorPartNif(
    const std::filesystem::path& nifPath,
    ImportedSkinnedActorAsset& ioAsset,
    std::string& outError
) {
    if (ioAsset.skeleton.empty()) {
        outError = "Skinned actor skeleton must be loaded before actor parts";
        return false;
    }

    std::unordered_map<std::string, std::uint32_t> boneIndexByName;
    boneIndexByName.reserve(ioAsset.skeleton.size());
    for (std::uint32_t boneIndex = 0; boneIndex < ioAsset.skeleton.size(); ++boneIndex) {
        boneIndexByName.emplace(lowerAsciiCopy(ioAsset.skeleton[boneIndex].name), boneIndex);
    }

    ImportedNifResult partResult{};
    std::vector<std::array<std::uint16_t, 4>> partBoneIndices;
    std::vector<std::array<float, 4>> partBoneWeights;
    StaticNifLoadOptions options{};
    options.includeHiddenGeometry = false;
    options.keepAlphaBlendOnlyParts = true;
    options.preserveRootTransform = true;
    // Body-part NIFs are authored in part-local spaces and rely on NiSkinData to
    // place them onto the shared humanoid skeleton. Keep the weights for runtime
    // animation, but bake the bind pose here so head, hair, clothing, hands, and
    // feet assemble into one coherent NPC before per-frame deformation.
    options.bakeSkinBindPose = true;
    options.canonicalBoneIndexByName = &boneIndexByName;
    options.outBoneIndices = &partBoneIndices;
    options.outBoneWeights = &partBoneWeights;
    options.weightedVertexCount = &ioAsset.weightedVertexCount;
    options.unweightedVertexCount = &ioAsset.unweightedVertexCount;
    options.unknownBoneInfluenceCount = &ioAsset.unknownBoneInfluenceCount;
    options.droppedInfluenceCount = &ioAsset.droppedInfluenceCount;
    if (!loadMorrowindStaticNifWithOptions(nifPath, partResult, outError, options)) {
        return false;
    }
    if (partResult.mesh.vertices.empty() || partResult.mesh.indices.empty()) {
        outError = "Skinned actor part had no geometry";
        return false;
    }
    if (partBoneIndices.size() != partResult.mesh.vertices.size() ||
        partBoneWeights.size() != partResult.mesh.vertices.size()) {
        outError = "Skinned actor part influence count did not match vertex count";
        return false;
    }

    const std::uint32_t vertexBase = static_cast<std::uint32_t>(ioAsset.vertices.size());
    const std::uint32_t indexBase = static_cast<std::uint32_t>(ioAsset.indices.size());
    ioAsset.vertices.reserve(ioAsset.vertices.size() + partResult.mesh.vertices.size());
    ioAsset.boneIndices.reserve(ioAsset.boneIndices.size() + partBoneIndices.size());
    ioAsset.boneWeights.reserve(ioAsset.boneWeights.size() + partBoneWeights.size());
    for (std::size_t vertexIndex = 0; vertexIndex < partResult.mesh.vertices.size(); ++vertexIndex) {
        const ImportedSceneVertex& srcVertex = partResult.mesh.vertices[vertexIndex];
        ImportedScenePackedVertex dstVertex{};
        std::memcpy(dstVertex.position, srcVertex.position, sizeof(dstVertex.position));
        std::memcpy(dstVertex.normal, srcVertex.normal, sizeof(dstVertex.normal));
        dstVertex.color[0] = 0.78f;
        dstVertex.color[1] = 0.72f;
        dstVertex.color[2] = 0.62f;
        std::memcpy(dstVertex.uv, srcVertex.uv, sizeof(dstVertex.uv));
        dstVertex.textureIndex = std::numeric_limits<std::uint32_t>::max();
        ioAsset.vertices.push_back(dstVertex);
        ioAsset.boneIndices.push_back(partBoneIndices[vertexIndex]);
        ioAsset.boneWeights.push_back(partBoneWeights[vertexIndex]);
    }
    ioAsset.indices.reserve(ioAsset.indices.size() + partResult.mesh.indices.size());
    for (const std::uint32_t index : partResult.mesh.indices) {
        ioAsset.indices.push_back(vertexBase + index);
    }

    if (partResult.mesh.parts.empty()) {
        ImportedScenePackedDraw draw{};
        draw.firstIndex = indexBase;
        draw.indexCount = static_cast<std::uint32_t>(partResult.mesh.indices.size());
        ioAsset.draws.push_back(draw);
        ioAsset.partDiffuseTexturePaths.push_back(partResult.diffuseTexturePath);
    } else {
        for (std::size_t partIndex = 0; partIndex < partResult.mesh.parts.size(); ++partIndex) {
            const ImportedSceneMeshPart& srcPart = partResult.mesh.parts[partIndex];
            ImportedScenePackedDraw draw{};
            draw.firstIndex = indexBase + srcPart.firstIndex;
            draw.indexCount = srcPart.indexCount;
            ioAsset.draws.push_back(draw);
            std::string texturePath = partResult.diffuseTexturePath;
            if (partIndex < partResult.partDiffuseTexturePaths.size() &&
                !partResult.partDiffuseTexturePaths[partIndex].empty()) {
                texturePath = partResult.partDiffuseTexturePaths[partIndex];
            }
            ioAsset.partDiffuseTexturePaths.push_back(texturePath);
        }
    }
    return true;
}

}  // namespace odai::importer
