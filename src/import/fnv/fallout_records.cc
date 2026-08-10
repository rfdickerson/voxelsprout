#include "import/fnv/fallout_records.h"

#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <unordered_map>
#include <vector>

namespace odai::importer::fnv {

namespace {

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

// Base record types that place a model in the world and carry EDID + MODL in
// the same shape as STAT.
//
// Handling only STAT dropped 20-37% of the references in a typical Goodsprings
// cell -- measured with `odai_newvegas_probe --floaters`. The visible symptom is
// not a missing object so much as a floating one: road segments rest on
// embankment and fill pieces that are MSTT/ACTI, and with those gone the road
// hangs in the air over the terrain.
//
// SCOL (static collection) IS here, and the reason is worth recording because
// the opposite was assumed for a long time. The comment that used to sit here
// said SCOL was "a container of transformed sub-statics, not an EDID+MODL
// record", and that placing it as a model would land its origin marker rather
// than its contents. Both halves are wrong, checked against retail data:
//
//   * All 98 SCOL records in FalloutNV.esm carry exactly one MODL, and 88 of
//     them resolve to a real mesh. The 10 that do not are SCOLtest01..10 --
//     unshipped developer records, which fail to resolve exactly like any other
//     missing mesh and land in m_failedStatics.
//   * That mesh is the MERGED geometry, not a marker. meshes\scol\
//     scolgoodpringsfenceb01.nif is 126 KB holding two textured shapes of 1486
//     and 58 vertices, and its bounds (x -793..7) match the part positions in
//     the record's own DATA subrecords.
//
// The ONAM/DATA container is authoring data: ONAM names a sub-static and DATA
// is an array of 28-byte {pos[3], rotationRadians[3], scale} placements, which
// is what the GECK consumes to BAKE the merged NIF. The game ships the bake, so
// reading the container at runtime would rebuild geometry that already exists.
// (423 DATA subrecords across the 98 records, every size a multiple of 28,
// which is what pins the stride.)
bool isModelBearingBaseType(std::string_view type) {
    // Diagnostic: narrow placement back to STAT alone, which is what this
    // importer did before the other base types were added. Bisects "is this
    // artifact coming from a type we only recently started placing?".
    // Read once -- this runs per record.
    static const bool statOnly = std::getenv("ODAI_FNV_STAT_ONLY") != nullptr;
    if (statOnly) {
        return type == "STAT";
    }
    return type == "STAT" || type == "MSTT" || type == "ACTI" || type == "DOOR" ||
           type == "CONT" || type == "FURN" || type == "TREE" || type == "MISC" ||
           type == "TERM" || type == "LIGH" || type == "BOOK" || type == "KEYM" ||
           type == "ALCH" || type == "AMMO" || type == "WEAP" || type == "ARMO" ||
           type == "NOTE" || type == "IMOD" || type == "CCRD" || type == "CHIP" ||
           type == "CMNY" || type == "SCOL";
    // PWAT (placeable water) is deliberately excluded: it belongs to the water
    // render path, and going through the opaque static path draws it as a solid
    // pale slab lying across the scene.
}

void parseStatRecord(const EsmRecordView& record, FalloutSceneData& scene) {
    FalloutStaticRecord entry{};
    entry.formId = record.formId;
    entry.recordType = record.type;
    for (const EsmSubrecordView& sub : record.subrecords) {
        if (sub.type == "EDID") {
            entry.editorId = subrecordString(sub);
        } else if (sub.type == "MODL") {
            entry.modelPath = subrecordString(sub);
        }
    }
    scene.statics.push_back(std::move(entry));
}

// LIGH's light parameters. Layout and the reasoning behind every field is in
// FalloutLightRecord's comment in the header -- it was read off the file, not
// taken from documentation.
//
// A LIGH is parsed TWICE on purpose: once by parseStatRecord, because 29 of the
// 501 carry a MODL and a lamp is a visible object, and once here for the light
// itself. The two are additive, not alternatives.
constexpr std::size_t kLightDataSize = 32u;

void parseLightRecord(const EsmRecordView& record, FalloutSceneData& scene) {
    FalloutLightRecord entry{};
    entry.formId = record.formId;
    bool haveData = false;
    for (const EsmSubrecordView& sub : record.subrecords) {
        if (sub.type == "EDID") {
            entry.editorId = subrecordString(sub);
        } else if (sub.type == "MODL") {
            entry.modelPath = subrecordString(sub);
        } else if (sub.type == "DATA" && sub.size >= kLightDataSize) {
            // Every retail DATA is exactly 32 bytes; >= rather than == so a
            // longer one from a mod is read rather than dropped.
            entry.radius = static_cast<float>(readU32(sub.data + 4));
            entry.color[0] = static_cast<float>(sub.data[8]) / 255.0f;
            entry.color[1] = static_cast<float>(sub.data[9]) / 255.0f;
            entry.color[2] = static_cast<float>(sub.data[10]) / 255.0f;
            entry.flags = readU32(sub.data + 12);
            entry.falloffExponent = readF32(sub.data + 16);
            haveData = true;
        } else if (sub.type == "FNAM" && sub.size >= sizeof(float)) {
            entry.fadeValue = readF32(sub.data);
        }
    }
    if (!haveData) {
        return;  // no DATA means nothing to light with; not an error
    }
    scene.lights.push_back(std::move(entry));
}

// REGN. Only EDID and RDMP are read -- see FalloutRegionRecord for why the
// displayed name is RDMP and why a region without one is not a fallback case.
void parseRegionRecord(const EsmRecordView& record, FalloutSceneData& scene) {
    FalloutRegionRecord entry{};
    entry.formId = record.formId;
    for (const EsmSubrecordView& sub : record.subrecords) {
        if (sub.type == "EDID") {
            entry.editorId = subrecordString(sub);
        } else if (sub.type == "RDMP") {
            entry.mapName = subrecordString(sub);
        }
    }
    scene.regions.push_back(std::move(entry));
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
        } else if (sub.type == "XCLR") {
            // A packed array of REGN formIDs. Every retail size is a multiple
            // of 4 (measured: 4, 8, 12, 16, 20 and one 24), which is what pins
            // the stride; the loop tolerates a trailing partial anyway rather
            // than reading past the subrecord.
            for (std::uint32_t offset = 0; offset + 4u <= sub.size; offset += 4u) {
                entry.regionFormIds.push_back(readU32(sub.data + offset));
            }
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
        } else if (sub.type == "XTEL" && sub.size >= 28u) {
            // formID of the destination door reference, then the arrival
            // position and rotation in that door's cell.
            ref.hasTeleport = true;
            ref.teleportTargetRefFormId = readU32(sub.data);
            ref.teleportPosition[0] = readF32(sub.data + 4);
            ref.teleportPosition[1] = readF32(sub.data + 8);
            ref.teleportPosition[2] = readF32(sub.data + 12);
            ref.teleportRotationRadians[0] = readF32(sub.data + 16);
            ref.teleportRotationRadians[1] = readF32(sub.data + 20);
            ref.teleportRotationRadians[2] = readF32(sub.data + 24);
        }
    }
    if (hasData && ref.baseFormId != 0u) {
        currentCell->references.push_back(ref);
    }
}

// NAVM. Layout measured from FalloutNV.esm, not taken from documentation:
//
//   DATA  u32 cellFormId, u32 vertexCount, u32 triangleCount, then fields this
//         reader does not use.
//   NVVX  vertexCount * 3 floats, Bethesda space.
//   NVTR  triangleCount * 16 bytes: 3 u16 vertex indices, 3 u16 neighbouring
//         triangle indices (0xFFFF = border), u16 flags, u16 cover.
//   NVDP  8 bytes each: u32 door reference formID, u16 triangle index, u16 pad.
//
// The counts are cross-checked against the subrecord sizes rather than trusted:
// a DATA that disagrees with its own NVVX/NVTR means the layout assumption is
// wrong, and silently reading a wrong number of triangles would produce a
// plausible mesh with garbage adjacency. Mismatches drop the record.
void parseNavMeshRecord(const EsmRecordView& record, FalloutCellRecord* currentCell) {
    if (currentCell == nullptr) {
        return;
    }
    constexpr std::size_t kNavMeshVertexBytes = 12u;
    constexpr std::size_t kNavMeshTriangleBytes = 16u;
    constexpr std::size_t kNavMeshDoorPortalBytes = 8u;

    FalloutNavMeshRecord navMesh{};
    navMesh.formId = record.formId;
    std::uint32_t declaredVertexCount = 0;
    std::uint32_t declaredTriangleCount = 0;
    const EsmSubrecordView* vertexData = nullptr;
    const EsmSubrecordView* triangleData = nullptr;
    const EsmSubrecordView* doorPortalData = nullptr;

    for (const EsmSubrecordView& sub : record.subrecords) {
        if (sub.type == "DATA" && sub.size >= 12u) {
            navMesh.cellFormId = readU32(sub.data);
            declaredVertexCount = readU32(sub.data + 4);
            declaredTriangleCount = readU32(sub.data + 8);
        } else if (sub.type == "NVVX") {
            vertexData = &sub;
        } else if (sub.type == "NVTR") {
            triangleData = &sub;
        } else if (sub.type == "NVDP") {
            doorPortalData = &sub;
        }
    }
    if (vertexData == nullptr || triangleData == nullptr) {
        return;
    }
    if (vertexData->size != declaredVertexCount * kNavMeshVertexBytes ||
        triangleData->size != declaredTriangleCount * kNavMeshTriangleBytes) {
        return;  // DATA disagrees with the arrays it describes
    }

    navMesh.vertices.resize(static_cast<std::size_t>(declaredVertexCount) * 3u);
    for (std::uint32_t i = 0; i < declaredVertexCount * 3u; ++i) {
        navMesh.vertices[i] = readF32(vertexData->data + (static_cast<std::size_t>(i) * 4u));
    }

    navMesh.triangles.resize(declaredTriangleCount);
    for (std::uint32_t i = 0; i < declaredTriangleCount; ++i) {
        const std::uint8_t* entry = triangleData->data + (static_cast<std::size_t>(i) * kNavMeshTriangleBytes);
        FalloutNavMeshTriangle& triangle = navMesh.triangles[i];
        bool indicesValid = true;
        for (int corner = 0; corner < 3; ++corner) {
            triangle.vertex[corner] = readU16(entry + (corner * 2));
            if (triangle.vertex[corner] >= declaredVertexCount) {
                indicesValid = false;
            }
        }
        for (int edge = 0; edge < 3; ++edge) {
            const std::uint16_t neighbour = readU16(entry + 6 + (edge * 2));
            // Anything out of range becomes a border rather than a wild index:
            // a neighbour link is dereferenced during pathfinding, so a bad one
            // is a crash rather than a cosmetic fault.
            triangle.neighbour[edge] =
                (neighbour < declaredTriangleCount) ? neighbour : kNavMeshNoNeighbour;
        }
        triangle.flags = readU16(entry + 12);
        triangle.coverFlags = readU16(entry + 14);
        if (!indicesValid) {
            return;  // a triangle indexing past its own vertex array is not salvageable
        }
    }

    if (doorPortalData != nullptr) {
        const std::size_t portalCount = doorPortalData->size / kNavMeshDoorPortalBytes;
        navMesh.doorPortals.reserve(portalCount);
        for (std::size_t i = 0; i < portalCount; ++i) {
            const std::uint8_t* entry = doorPortalData->data + (i * kNavMeshDoorPortalBytes);
            FalloutNavMeshDoorPortal portal{};
            portal.doorRefFormId = readU32(entry);
            portal.triangleIndex = readU16(entry + 4);
            if (portal.triangleIndex < declaredTriangleCount) {
                navMesh.doorPortals.push_back(portal);
            }
        }
    }

    currentCell->navMeshes.push_back(std::move(navMesh));
}

// Reconstructs the 33x33 absolute height grid from VHGT's delta encoding:
// each row's first post continues accumulating from the previous row's
// first post, and each subsequent post in a row accumulates from the
// previous post in that same row.
void decodeLandHeights(const std::uint8_t* vhgtData, FalloutLandRecord& land) {
    const float baseOffset = readF32(vhgtData);
    const auto* deltas = reinterpret_cast<const std::int8_t*>(vhgtData + 4);

    // The height scale multiplies the accumulated total INCLUDING the VHGT
    // offset, not just the deltas: height = (offset + sum(deltas)) * 8. So
    // accumulate in raw units and scale once on store.
    //
    // Scaling only the deltas (what this did before) leaves the whole cell
    // displaced by 7 * offset. Measured against real data: across 3531 placed
    // references in the Goodsprings area, the old formula put every object a
    // median of 7566 units — about 108 m — above the terrain, with 0% of them
    // resting on it; this one gives a median of -2.0 units with 96.1% sitting
    // within [-200, +600] of the ground beneath them. Objects sitting on the
    // terrain they were authored against is the check, and it is what the
    // cell-edge continuity test could not see, being scale-invariant.
    float rowStart = baseOffset;
    for (int row = 0; row < kLandGridSize; ++row) {
        float current = rowStart;
        for (int col = 0; col < kLandGridSize; ++col) {
            const std::int8_t delta = deltas[(row * kLandGridSize) + col];
            if (!(row == 0 && col == 0)) {
                current += static_cast<float>(delta);
            }
            land.heights[(row * kLandGridSize) + col] = current * kLandHeightScale;
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

// VCLR is one unsigned RGB triple per post, same row-major order as VHGT/VNML.
// Unlike VNML these are unsigned: 255 is neutral (leave the texture alone), not
// a signed component, so they scale straight to [0,1] rather than [-1,1].
void decodeLandColors(const std::uint8_t* vclrData, FalloutLandRecord& land) {
    for (int i = 0; i < kLandVertexCount * 3; ++i) {
        land.colors[i] = static_cast<float>(vclrData[i]) / 255.0f;
    }
    land.hasColors = true;
}

// TXST holds a texture set; TX00 is its diffuse slot. Collected separately
// because an LTEX only names the TXST by formID, and the TXST may appear
// either before or after it in the file.
void parseTextureSetRecord(const EsmRecordView& record, std::unordered_map<std::uint32_t, std::string>& outPaths) {
    for (const EsmSubrecordView& sub : record.subrecords) {
        if (sub.type == "TX00" && sub.size > 0u) {
            outPaths[record.formId] = subrecordString(sub);
            return;
        }
    }
}

void parseLandTextureRecord(const EsmRecordView& record, FalloutSceneData& outScene) {
    FalloutLandTextureRecord landTexture{};
    landTexture.formId = record.formId;
    for (const EsmSubrecordView& sub : record.subrecords) {
        if (sub.type == "EDID") {
            landTexture.editorId = subrecordString(sub);
        } else if (sub.type == "TNAM" && sub.size >= 4u) {
            landTexture.textureSetFormId = readU32(sub.data);
        }
    }
    outScene.landTextures.push_back(std::move(landTexture));
}

void parseLandRecord(const EsmRecordView& record, FalloutCellRecord* currentCell) {
    if (currentCell == nullptr) {
        return;
    }
    // Built directly in its final heap home so the ~17 KB record is never
    // copied: previously this filled a stack temporary and then copied it into
    // the cell.
    auto land = std::make_unique<FalloutLandRecord>();
    land->cellFormId = currentCell->formId;
    for (const EsmSubrecordView& sub : record.subrecords) {
        if (sub.type == "VHGT" && sub.size >= 4u + kLandVertexCount) {
            decodeLandHeights(sub.data, *land);
        } else if (sub.type == "VNML" && sub.size >= static_cast<std::uint32_t>(kLandVertexCount * 3)) {
            decodeLandNormals(sub.data, *land);
        } else if (sub.type == "VCLR" && sub.size >= static_cast<std::uint32_t>(kLandVertexCount * 3)) {
            decodeLandColors(sub.data, *land);
        } else if (sub.type == "BTXT" && sub.size >= 8u) {
            const std::uint32_t textureFormId = readU32(sub.data);
            const std::uint8_t quadrant = sub.data[4];
            if (quadrant < 4u) {
                land->quadrantBaseTextureFormId[quadrant] = textureFormId;
            }
        } else if (sub.type == "ATXT" && sub.size >= 8u) {
            // ATXT opens a layer; the VTXT that follows fills its opacity map.
            // Same 8-byte shape as BTXT: formID, quadrant, pad, then a u16 that
            // is the layer index here rather than BTXT's unused field.
            FalloutLandTextureLayer layer{};
            layer.textureFormId = readU32(sub.data);
            layer.quadrant = sub.data[4];
            layer.layerIndex = static_cast<std::uint16_t>(sub.data[6] | (sub.data[7] << 8));
            if (layer.quadrant < 4u) {
                land->textureLayers.push_back(layer);
            }
        } else if (sub.type == "VTXT" && sub.size >= 8u) {
            // Applies to the most recent ATXT. A VTXT with no ATXT before it is
            // malformed; skip rather than guessing which layer it belongs to.
            if (land->textureLayers.empty()) {
                continue;
            }
            FalloutLandTextureLayer& layer = land->textureLayers.back();
            // 8-byte entries: u16 post position, u16 unused, float opacity.
            const std::uint32_t entryCount = sub.size / 8u;
            for (std::uint32_t entry = 0; entry < entryCount; ++entry) {
                const std::uint8_t* entryData = sub.data + (static_cast<std::size_t>(entry) * 8u);
                const std::uint16_t position =
                    static_cast<std::uint16_t>(entryData[0] | (entryData[1] << 8));
                if (position >= static_cast<std::uint16_t>(kLandQuadrantVertexCount)) {
                    continue;
                }
                float opacity = 0.0f;
                std::memcpy(&opacity, entryData + 4, sizeof(float));
                if (!std::isfinite(opacity)) {
                    continue;
                }
                layer.opacity[position] = std::clamp(opacity, 0.0f, 1.0f);
            }
        }
    }
    // ATXT's layer index is authoritative for blend order, and subrecord order
    // is not guaranteed to match it. stable_sort so layers that declare the same
    // index keep the order the file listed them in.
    std::stable_sort(
        land->textureLayers.begin(),
        land->textureLayers.end(),
        [](const FalloutLandTextureLayer& a, const FalloutLandTextureLayer& b) {
            return a.layerIndex < b.layerIndex;
        });
    if (land->hasHeights) {
        currentCell->land = std::move(land);
    }
}

}  // namespace

bool buildFalloutCellIndex(
    const std::filesystem::path& esmPath, FalloutCellIndex& outIndex, std::string& outError) {
    outIndex = FalloutCellIndex{};
    EsmReader reader;
    if (!reader.open(esmPath)) {
        outError = reader.lastError();
        return false;
    }

    // Group type 6 is a cell-children group; its label is the owning CELL's
    // formID. That group is the unit this index addresses -- everything a cell
    // owns (persistent, temporary and visible-distant children) is inside it.
    constexpr std::int32_t kTopLevelGroup = 0;
    constexpr std::int32_t kWorldChildrenGroup = 1;
    constexpr std::int32_t kCellChildrenGroup = 6;

    std::vector<std::uint32_t> worldspaceStack;
    std::size_t currentCellIndex = 0;
    bool hasCurrentCell = false;
    // Cell index by formID, so a children group can be attributed to its cell
    // without assuming the group immediately follows the record.
    std::unordered_map<std::uint32_t, std::size_t> cellIndexByFormId;
    // Scratch scene used only to reuse parseCellRecord/parseWorldspaceRecord;
    // its cells are converted into index entries and then discarded.
    FalloutSceneData scratch;

    EsmReader::Visitor visitor{};
    // onRecordHeader fires before onRecord for the same record, so this carries
    // the CELL's file offset across to where the entry is built.
    std::uint64_t pendingCellRecordOffset = 0;
    visitor.onRecordHeader = [&](const EsmRecordHeaderView& header) {
        if (header.type == "REFR" && hasCurrentCell) {
            outIndex.cellIndexByReferenceFormId.emplace(header.formId, currentCellIndex);
        }
        if (header.type == "CELL") {
            pendingCellRecordOffset = header.fileOffset;
        }
        // The whole point of the index: never materialize cell contents. Only
        // CELL and WRLD records are parsed; LAND in particular stays compressed.
        return header.type == "CELL" || header.type == "WRLD";
    };
    visitor.onGroupEnter = [&](const EsmGroupView& group) {
        if (group.groupType == kTopLevelGroup && group.rawLabel.size() == 4u) {
            if (group.rawLabel != "WRLD" && group.rawLabel != "CELL") {
                return false;
            }
        }
        if (group.groupType == kWorldChildrenGroup && group.rawLabel.size() == 4u) {
            std::uint32_t formId = 0;
            std::memcpy(&formId, group.rawLabel.data(), 4u);
            worldspaceStack.push_back(formId);
        }
        if (group.groupType == kCellChildrenGroup && group.rawLabel.size() == 4u) {
            std::uint32_t cellFormId = 0;
            std::memcpy(&cellFormId, group.rawLabel.data(), 4u);
            const auto found = cellIndexByFormId.find(cellFormId);
            if (found != cellIndexByFormId.end()) {
                FalloutCellIndexEntry& entry = outIndex.cells[found->second];
                entry.childrenGroupOffset = group.fileOffset;
                entry.childrenGroupSize = group.groupSize;
            }
            // Descend anyway: the REFR headers inside are what build
            // cellIndexByReferenceFormId, and headers are cheap.
            currentCellIndex = (found != cellIndexByFormId.end()) ? found->second : currentCellIndex;
            hasCurrentCell = found != cellIndexByFormId.end();
        }
        return true;
    };
    visitor.onGroupExit = [&](const EsmGroupView& group) {
        if (group.groupType != kWorldChildrenGroup || worldspaceStack.empty() ||
            group.rawLabel.size() != 4u) {
            return;
        }
        std::uint32_t formId = 0;
        std::memcpy(&formId, group.rawLabel.data(), 4u);
        if (worldspaceStack.back() == formId) {
            worldspaceStack.pop_back();
        }
    };
    visitor.onRecord = [&](const EsmRecordView& record) {
        const std::uint32_t currentWorldspace = worldspaceStack.empty() ? 0u : worldspaceStack.back();
        if (record.type == "WRLD") {
            parseWorldspaceRecord(record, scratch);
            outIndex.worldspaces = scratch.worldspaces;
            return;
        }
        if (record.type != "CELL") {
            return;
        }
        scratch.cells.clear();
        parseCellRecord(record, currentWorldspace, scratch);
        if (scratch.cells.empty()) {
            return;
        }
        const FalloutCellRecord& parsed = scratch.cells.back();
        FalloutCellIndexEntry entry{};
        entry.cellFormId = parsed.formId;
        entry.editorId = parsed.editorId;
        entry.worldspaceFormId = parsed.worldspaceFormId;
        entry.gridX = parsed.gridX;
        entry.gridZ = parsed.gridZ;
        entry.hasGridCoords = parsed.hasGridCoords;
        entry.isInterior = parsed.isInterior;
        entry.regionFormIds = parsed.regionFormIds;
        entry.cellRecordOffset = pendingCellRecordOffset;
        outIndex.cells.push_back(entry);
        cellIndexByFormId[parsed.formId] = outIndex.cells.size() - 1u;
        currentCellIndex = outIndex.cells.size() - 1u;
        hasCurrentCell = true;
    };

    if (!reader.walk(visitor)) {
        outError = reader.lastError();
        return false;
    }
    return true;
}

bool extractFalloutCellAt(
    EsmReader& reader,
    const FalloutCellIndexEntry& entry,
    FalloutCellRecord& outCell,
    std::string& outError) {
    outCell = FalloutCellRecord{};
    outCell.formId = entry.cellFormId;
    outCell.isInterior = entry.isInterior;
    outCell.hasGridCoords = entry.hasGridCoords;
    outCell.gridX = entry.gridX;
    outCell.gridZ = entry.gridZ;
    outCell.worldspaceFormId = entry.worldspaceFormId;

    if (entry.childrenGroupSize == 0u) {
        return true;  // a cell with no children group simply has no contents
    }

    EsmReader::Visitor visitor{};
    visitor.onRecord = [&](const EsmRecordView& record) {
        if (record.type == "REFR") {
            parseReferenceRecord(record, &outCell);
        } else if (record.type == "LAND") {
            parseLandRecord(record, &outCell);
        } else if (record.type == "NAVM") {
            parseNavMeshRecord(record, &outCell);
        }
    };

    if (!reader.walkRange(
            entry.childrenGroupOffset,
            entry.childrenGroupOffset + entry.childrenGroupSize,
            visitor)) {
        outError = reader.lastError();
        return false;
    }
    return true;
}

bool extractFalloutScene(const std::filesystem::path& esmPath, FalloutSceneData& outScene, std::string& outError) {
    return extractFalloutScene(esmPath, FalloutExtractFilter{}, outScene, outError);
}

bool extractFalloutScene(
    const std::filesystem::path& esmPath,
    const FalloutExtractFilter& filter,
    FalloutSceneData& outScene,
    std::string& outError
) {
    outScene = FalloutSceneData{};
    EsmReader reader;
    if (!reader.open(esmPath)) {
        outError = reader.lastError();
        return false;
    }

    // TXST diffuse paths, resolved into landTextures after the walk since an
    // LTEX may be parsed before the TXST it names.
    std::unordered_map<std::uint32_t, std::string> textureSetPaths;

    std::vector<std::uint32_t> worldspaceStack;
    // Index (not formID) into outScene.cells, re-resolved to a pointer on
    // every lookup so it stays valid across the vector's own reallocations
    // as later cells are pushed — O(1) instead of a per-record linear scan.
    std::size_t currentCellIndex = 0;
    bool hasCurrentCell = false;
    // Whether the filter accepted the cell we are currently inside. When it
    // did not, that cell's LAND and REFR records are rejected from the header
    // callback and never decompressed or parsed at all.
    bool wantCurrentCellContents = true;

    auto findCurrentCell = [&]() -> FalloutCellRecord* {
        return hasCurrentCell ? &outScene.cells[currentCellIndex] : nullptr;
    };

    EsmReader::Visitor visitor{};
    // Runs for every record whether or not its contents are wanted, which is
    // what makes the door index affordable: a teleport's XTEL names the door
    // reference on the far side and nothing about which cell that is, so the
    // mapping has to cover references in cells this cook never parses. Reading
    // it from the header costs a hash insert per REFR instead of a full
    // subrecord walk.
    visitor.onRecordHeader = [&](const EsmRecordHeaderView& header) {
        if (header.type == "REFR" && hasCurrentCell) {
            outScene.cellIndexByReferenceFormId.emplace(header.formId, currentCellIndex);
        }
        if (filter.wantCellContents &&
            (header.type == "LAND" || header.type == "REFR" || header.type == "NAVM")) {
            return wantCurrentCellContents;
        }
        return true;
    };
    visitor.onGroupEnter = [&](const EsmGroupView& group) {
        constexpr std::int32_t kTopLevelGroup = 0;
        constexpr std::int32_t kWorldChildrenGroup = 1;

        // A top-level group's label is the record type it contains. This
        // function extracts a narrow set, so every other top group — DIAL,
        // INFO, NAVM, SCPT, PACK, SOUN and the rest, which together are most of
        // the record count — can be seeked past without being read.
        // Unconditional rather than caller-controlled: parsing them would
        // produce nothing either way.
        //
        // THIS LIST IS A SECOND GATE, and forgetting it is silent. A record
        // type reaching the dispatch below still yields nothing unless its top
        // group is admitted here — REGN was added to the dispatch first and
        // parsed exactly zero records, because its group was seeked past before
        // any record header was ever read. Add types in both places.
        if (group.groupType == kTopLevelGroup && group.rawLabel.size() == 4u) {
            if (!isModelBearingBaseType(group.rawLabel) && group.rawLabel != "WRLD" &&
                group.rawLabel != "CELL" && group.rawLabel != "LTEX" &&
                group.rawLabel != "TXST" && group.rawLabel != "REGN") {
                return false;
            }
        }

        if (group.groupType == kWorldChildrenGroup && group.rawLabel.size() == 4u) {
            std::uint32_t formId = 0;
            std::memcpy(&formId, group.rawLabel.data(), 4u);
            if (filter.wantWorldspace && !filter.wantWorldspace(formId)) {
                // Refuse before pushing: onGroupExit still fires for a skipped
                // group, and it must not pop a worldspace we never pushed.
                return false;
            }
            worldspaceStack.push_back(formId);
        }
        return true;
    };
    visitor.onGroupExit = [&](const EsmGroupView& group) {
        constexpr std::int32_t kWorldChildrenGroup = 1;
        if (group.groupType != kWorldChildrenGroup || worldspaceStack.empty() || group.rawLabel.size() != 4u) {
            return;
        }
        // onGroupExit also fires for groups onGroupEnter refused, which were
        // never pushed. Pop only when the top actually is this group.
        std::uint32_t formId = 0;
        std::memcpy(&formId, group.rawLabel.data(), 4u);
        if (worldspaceStack.back() == formId) {
            worldspaceStack.pop_back();
        }
    };
    visitor.onRecord = [&](const EsmRecordView& record) {
        const std::uint32_t currentWorldspace = worldspaceStack.empty() ? 0u : worldspaceStack.back();
        if (isModelBearingBaseType(record.type)) {
            parseStatRecord(record, outScene);
            // Not an "else": a LIGH is both a (usually absent) model and a
            // light source, and both halves are wanted.
            if (record.type == "LIGH") {
                parseLightRecord(record, outScene);
            }
        } else if (record.type == "REGN") {
            parseRegionRecord(record, outScene);
        } else if (record.type == "LTEX") {
            parseLandTextureRecord(record, outScene);
        } else if (record.type == "TXST") {
            parseTextureSetRecord(record, textureSetPaths);
        } else if (record.type == "WRLD") {
            parseWorldspaceRecord(record, outScene);
        } else if (record.type == "CELL") {
            parseCellRecord(record, currentWorldspace, outScene);
            currentCellIndex = outScene.cells.size() - 1u;
            hasCurrentCell = true;
            wantCurrentCellContents =
                !filter.wantCellContents || filter.wantCellContents(outScene.cells[currentCellIndex]);
        } else if (record.type == "REFR") {
            parseReferenceRecord(record, findCurrentCell());
        } else if (record.type == "LAND") {
            parseLandRecord(record, findCurrentCell());
        } else if (record.type == "NAVM") {
            parseNavMeshRecord(record, findCurrentCell());
        }
    };

    if (!reader.walk(visitor)) {
        outError = reader.lastError();
        return false;
    }

    for (FalloutLandTextureRecord& landTexture : outScene.landTextures) {
        const auto it = textureSetPaths.find(landTexture.textureSetFormId);
        if (it != textureSetPaths.end()) {
            landTexture.diffuseTexturePath = it->second;
        }
    }
    return true;
}

}  // namespace odai::importer::fnv
