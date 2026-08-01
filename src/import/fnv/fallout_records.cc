#include "import/fnv/fallout_records.h"

#include <cmath>
#include <cstring>
#include <vector>

namespace odai::importer::fnv {

namespace {

std::uint32_t readU32(const std::uint8_t* bytes) {
    std::uint32_t value = 0;
    std::memcpy(&value, bytes, sizeof(value));
    return value;
}

std::int32_t readI32(const std::uint8_t* bytes) {
    std::int32_t value = 0;
    std::memcpy(&value, bytes, sizeof(value));
    return value;
}

float readF32(const std::uint8_t* bytes) {
    float value = 0.0f;
    std::memcpy(&value, bytes, sizeof(value));
    return value;
}

// Subrecord text fields are NUL-terminated in the plugin; trim the
// terminator rather than including it in the extracted string.
std::string subrecordString(const EsmSubrecordView& sub) {
    std::size_t length = sub.size;
    if (length > 0 && sub.data[length - 1] == '\0') {
        length -= 1;
    }
    return std::string(reinterpret_cast<const char*>(sub.data), length);
}

void parseStatRecord(const EsmRecordView& record, FalloutSceneData& scene) {
    FalloutStaticRecord entry{};
    entry.formId = record.formId;
    for (const EsmSubrecordView& sub : record.subrecords) {
        if (sub.type == "EDID") {
            entry.editorId = subrecordString(sub);
        } else if (sub.type == "MODL") {
            entry.modelPath = subrecordString(sub);
        }
    }
    scene.statics.push_back(std::move(entry));
}

void parseWorldspaceRecord(const EsmRecordView& record, FalloutSceneData& scene) {
    FalloutWorldspaceRecord entry{};
    entry.formId = record.formId;
    for (const EsmSubrecordView& sub : record.subrecords) {
        if (sub.type == "EDID") {
            entry.editorId = subrecordString(sub);
        }
    }
    scene.worldspaces.push_back(std::move(entry));
}

void parseCellRecord(const EsmRecordView& record, std::uint32_t currentWorldspaceFormId, FalloutSceneData& scene) {
    FalloutCellRecord entry{};
    entry.formId = record.formId;
    entry.worldspaceFormId = currentWorldspaceFormId;
    for (const EsmSubrecordView& sub : record.subrecords) {
        if (sub.type == "EDID") {
            entry.editorId = subrecordString(sub);
        } else if (sub.type == "DATA" && sub.size >= 1u) {
            entry.isInterior = (sub.data[0] & 0x1u) != 0u;
        } else if (sub.type == "XCLC" && sub.size >= 8u) {
            entry.hasGridCoords = true;
            entry.gridX = readI32(sub.data);
            entry.gridZ = readI32(sub.data + 4);
        }
    }
    scene.cells.push_back(std::move(entry));
}

void parseReferenceRecord(const EsmRecordView& record, FalloutCellRecord* currentCell) {
    if (currentCell == nullptr) {
        return;  // REFR outside any tracked CELL context; nothing to attach it to.
    }
    FalloutPlacedReference ref{};
    ref.formId = record.formId;
    ref.scale = 1.0f;
    bool hasData = false;
    for (const EsmSubrecordView& sub : record.subrecords) {
        if (sub.type == "NAME" && sub.size >= 4u) {
            ref.baseFormId = readU32(sub.data);
        } else if (sub.type == "DATA" && sub.size >= 24u) {
            ref.position[0] = readF32(sub.data);
            ref.position[1] = readF32(sub.data + 4);
            ref.position[2] = readF32(sub.data + 8);
            ref.rotationRadians[0] = readF32(sub.data + 12);
            ref.rotationRadians[1] = readF32(sub.data + 16);
            ref.rotationRadians[2] = readF32(sub.data + 20);
            hasData = true;
        } else if (sub.type == "XSCL" && sub.size >= 4u) {
            ref.scale = readF32(sub.data);
        }
    }
    if (hasData && ref.baseFormId != 0u) {
        currentCell->references.push_back(ref);
    }
}

// Reconstructs the 33x33 absolute height grid from VHGT's delta encoding:
// each row's first post continues accumulating from the previous row's
// first post, and each subsequent post in a row accumulates from the
// previous post in that same row.
void decodeLandHeights(const std::uint8_t* vhgtData, FalloutLandRecord& land) {
    const float baseOffset = readF32(vhgtData);
    const auto* deltas = reinterpret_cast<const std::int8_t*>(vhgtData + 4);

    float rowStart = baseOffset;
    for (int row = 0; row < kLandGridSize; ++row) {
        float current = rowStart;
        for (int col = 0; col < kLandGridSize; ++col) {
            const std::int8_t delta = deltas[(row * kLandGridSize) + col];
            if (!(row == 0 && col == 0)) {
                current += static_cast<float>(delta) * kLandHeightScale;
            }
            land.heights[(row * kLandGridSize) + col] = current;
            if (col == 0) {
                rowStart = current;
            }
        }
    }
    land.hasHeights = true;
}

void decodeLandNormals(const std::uint8_t* vnmlData, FalloutLandRecord& land) {
    for (int i = 0; i < kLandVertexCount; ++i) {
        const auto* signedBytes = reinterpret_cast<const std::int8_t*>(vnmlData + (i * 3));
        float x = static_cast<float>(signedBytes[0]) / 127.0f;
        float y = static_cast<float>(signedBytes[1]) / 127.0f;
        float z = static_cast<float>(signedBytes[2]) / 127.0f;
        const float length = std::sqrt((x * x) + (y * y) + (z * z));
        if (length > 1e-6f) {
            x /= length;
            y /= length;
            z /= length;
        } else {
            x = 0.0f;
            y = 0.0f;
            z = 1.0f;
        }
        land.normals[(i * 3) + 0] = x;
        land.normals[(i * 3) + 1] = y;
        land.normals[(i * 3) + 2] = z;
    }
    land.hasNormals = true;
}

void parseLandRecord(const EsmRecordView& record, FalloutCellRecord* currentCell) {
    if (currentCell == nullptr) {
        return;
    }
    FalloutLandRecord land{};
    land.cellFormId = currentCell->formId;
    for (const EsmSubrecordView& sub : record.subrecords) {
        if (sub.type == "VHGT" && sub.size >= 4u + kLandVertexCount) {
            decodeLandHeights(sub.data, land);
        } else if (sub.type == "VNML" && sub.size >= static_cast<std::uint32_t>(kLandVertexCount * 3)) {
            decodeLandNormals(sub.data, land);
        } else if (sub.type == "BTXT" && sub.size >= 8u) {
            const std::uint32_t textureFormId = readU32(sub.data);
            const std::uint8_t quadrant = sub.data[4];
            if (quadrant < 4u) {
                land.quadrantBaseTextureFormId[quadrant] = textureFormId;
            }
        }
    }
    if (land.hasHeights) {
        currentCell->land = land;
    }
}

}  // namespace

bool extractFalloutScene(const std::filesystem::path& esmPath, FalloutSceneData& outScene, std::string& outError) {
    outScene = FalloutSceneData{};
    EsmReader reader;
    if (!reader.open(esmPath)) {
        outError = reader.lastError();
        return false;
    }

    std::vector<std::uint32_t> worldspaceStack;
    // Index (not formID) into outScene.cells, re-resolved to a pointer on
    // every lookup so it stays valid across the vector's own reallocations
    // as later cells are pushed — O(1) instead of a per-record linear scan.
    std::size_t currentCellIndex = 0;
    bool hasCurrentCell = false;

    auto findCurrentCell = [&]() -> FalloutCellRecord* {
        return hasCurrentCell ? &outScene.cells[currentCellIndex] : nullptr;
    };

    EsmReader::Visitor visitor{};
    visitor.onGroupEnter = [&](const EsmGroupView& group) {
        constexpr std::int32_t kWorldChildrenGroup = 1;
        if (group.groupType == kWorldChildrenGroup && group.rawLabel.size() == 4u) {
            std::uint32_t formId = 0;
            std::memcpy(&formId, group.rawLabel.data(), 4u);
            worldspaceStack.push_back(formId);
        }
        return true;
    };
    visitor.onGroupExit = [&](const EsmGroupView& group) {
        constexpr std::int32_t kWorldChildrenGroup = 1;
        if (group.groupType == kWorldChildrenGroup && !worldspaceStack.empty()) {
            worldspaceStack.pop_back();
        }
    };
    visitor.onRecord = [&](const EsmRecordView& record) {
        const std::uint32_t currentWorldspace = worldspaceStack.empty() ? 0u : worldspaceStack.back();
        if (record.type == "STAT") {
            parseStatRecord(record, outScene);
        } else if (record.type == "WRLD") {
            parseWorldspaceRecord(record, outScene);
        } else if (record.type == "CELL") {
            parseCellRecord(record, currentWorldspace, outScene);
            currentCellIndex = outScene.cells.size() - 1u;
            hasCurrentCell = true;
        } else if (record.type == "REFR") {
            parseReferenceRecord(record, findCurrentCell());
        } else if (record.type == "LAND") {
            parseLandRecord(record, findCurrentCell());
        }
    };

    if (!reader.walk(visitor)) {
        outError = reader.lastError();
        return false;
    }
    return true;
}

}  // namespace odai::importer::fnv
