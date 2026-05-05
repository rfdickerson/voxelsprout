#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace odai::importer {

struct ImportedSceneVertex {
    float position[3] = {};
    float normal[3] = {};
    float uv[2] = {};
};

struct ImportedSceneMeshPart {
    std::uint32_t firstIndex = 0;
    std::uint32_t indexCount = 0;
    std::uint32_t textureIndex = 0;
    bool alphaTest = false;
};

struct ImportedSceneMesh {
    std::string name;
    std::vector<ImportedSceneVertex> vertices;
    std::vector<std::uint32_t> indices;
    std::vector<ImportedSceneMeshPart> parts;
};

struct ImportedSceneInstance {
    std::uint32_t meshIndex = 0;
    float transform[16] = {};
    std::string sourceId;
    std::string modelPath;
};

struct ImportedSceneTexture {
    std::string sourcePath;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t mipLevelCount = 1;
    std::vector<std::uint8_t> rgba8;
};

struct ImportedScenePackedVertex {
    float position[3] = {};
    float normal[3] = {};
    float color[3] = {};
    float uv[2] = {};
    std::uint32_t textureIndex = 0xffffffffu;
    std::uint32_t flags = 0u;
};

struct ImportedScenePackedDraw {
    std::uint32_t firstIndex = 0;
    std::uint32_t indexCount = 0;
};

struct ImportedSceneCellRef {
    std::string refId;
    std::string modelPath;
    std::uint32_t refNum = 0u;
    bool hasRefNum = false;
    int cellX = 0;
    int cellY = 0;
    bool deleted = false;
    float position[3] = {};
    float rotationRadians[3] = {};
    float scale = 1.0f;
};

struct ImportedSceneLandscapeCell {
    int gridX = 0;
    int gridY = 0;
    std::vector<float> heights;
    std::vector<std::uint16_t> textureIndices;
};

struct ImportedSceneWaterPatch {
    float originX = 0.0f;
    float originZ = 0.0f;
    float sizeX = 0.0f;
    float sizeZ = 0.0f;
    float waterLevel = 0.0f;
};

struct ImportedSceneLight {
    std::string sourceId;
    float position[3] = {};
    float color[3] = {1.0f, 1.0f, 1.0f};
    float radius = 0.0f;
    float intensity = 1.0f;
    std::uint32_t flags = 0u;
};

struct MorrowindDoorDestination {
    std::string destinationCell;
    float position[3] = {};
    float rotationRadians[3] = {};
};

struct MorrowindDoorReference {
    std::string refId;
    std::string sourceCell;
    float position[3] = {};
    float rotationRadians[3] = {};
    MorrowindDoorDestination destination;
};

struct MorrowindDoorCache {
    std::vector<MorrowindDoorReference> exteriorDoors;
    std::unordered_map<std::string, std::vector<MorrowindDoorReference>> interiorDoorsByCell;
};

enum class MorrowindActorKind : std::uint8_t {
    Npc = 0,
    Creature = 1,
};

struct MorrowindActorRecord {
    std::string id;
    MorrowindActorKind kind = MorrowindActorKind::Npc;
    std::string modelPath;
    std::string raceId;
    std::string headBodyPartId;
    std::string hairBodyPartId;
    std::vector<std::string> inventoryItemIds;
};

struct MorrowindActorCatalog {
    std::unordered_map<std::string, MorrowindActorRecord> actorsById;
};

struct MorrowindEquipmentCatalog {
    struct BodyPartRecord {
        std::string id;
        std::string modelPath;
        std::string slot;
        std::string side;
        std::string attachBone;
        std::string meshFilter;
        std::uint8_t meshPart = 0xffu;
        std::string inferredRaceId;
        std::string inferredGender;
    };
    struct ResolvedActorPart {
        std::string modelPath;
        std::string slot;
        std::string side;
        std::string bodyPartId;
        std::string attachBone;
        std::string meshFilter;
        std::uint8_t meshPart = 0xffu;
        std::uint8_t partReferenceType = 0xffu;
    };
    std::unordered_map<std::string, BodyPartRecord> bodyPartById;
    std::unordered_map<std::string, std::string> modelPathByBodyPartId;
    std::unordered_map<std::string, std::vector<std::string>> bodyPartModelPathsByItemId;
    std::unordered_map<std::string, std::vector<ResolvedActorPart>> actorPartsByItemId;
};

struct MorrowindContentFile {
    std::filesystem::path path;
    std::string displayName;
};

struct MorrowindDataRoot {
    std::filesystem::path path;
    std::string displayName;
};

struct MorrowindContentLoadOrder {
    std::filesystem::path dataFilesPath;
    std::vector<MorrowindDataRoot> dataRoots;
    std::vector<MorrowindContentFile> files;
    std::string profileName;
};

struct MorrowindContentIndex;

struct MorrowindContentIndexStats {
    std::uint32_t modelRecordCount = 0;
    std::uint32_t lightRecordCount = 0;
    std::uint32_t landscapeTextureCount = 0;
    std::uint32_t exteriorCellRecordCount = 0;
    std::uint32_t landRecordCount = 0;
    std::uint32_t interiorCellRecordCount = 0;
    std::uint32_t actorRecordCount = 0;
    std::uint32_t bodyRecordCount = 0;
    std::uint32_t equipmentRecordCount = 0;
    std::uint32_t dialogueRecordCount = 0;
    std::uint32_t scriptRecordCount = 0;
};

struct MorrowindContentSession {
    MorrowindContentLoadOrder loadOrder;
    std::uint64_t contentFingerprint = 0u;
    bool indexCacheHit = false;
    MorrowindContentIndexStats indexStats;
    std::shared_ptr<const MorrowindContentIndex> index;
};

struct MorrowindSceneRequest {
    std::vector<std::pair<int, int>> exteriorCells;
    std::string interiorCellName;
    bool includeStatics = true;
    bool includeTerrain = true;
    bool includeWater = true;
    bool includeLights = true;
};

struct MorrowindActorPartMetadata {
    std::string modelPath;
    std::string bodyPartId;
    std::string slot;
    std::string side;
    std::string attachBone;
    std::string meshFilter;
};

struct ImportedScene {
    std::string sourceTag;
    std::vector<ImportedSceneTexture> textures;
    std::vector<ImportedSceneMesh> meshes;
    std::vector<ImportedSceneInstance> instances;
    std::vector<ImportedSceneLandscapeCell> landscapeCells;
    std::vector<ImportedSceneWaterPatch> waterPatches;
    std::vector<ImportedSceneLight> lights;
    std::vector<ImportedSceneCellRef> unresolvedRefs;
    std::vector<ImportedScenePackedVertex> packedVertices;
    std::vector<std::uint32_t> packedIndices;
    std::vector<ImportedScenePackedDraw> packedDraws;
    std::uint32_t sourceTextureCount = 0;
    std::uint32_t sourceFileVersion = 0;
    std::uint32_t sourceMeshCount = 0;
    std::uint32_t sourceInstanceCount = 0;
    std::uint32_t sourceLandscapeCellCount = 0;
    std::uint32_t sourceWaterPatchCount = 0;
    std::uint32_t sourceLightCount = 0;
    std::uint32_t sourceUnresolvedRefCount = 0;
    float boundsMin[3] = {};
    float boundsMax[3] = {};
};

struct MorrowindSceneBuildResult {
    ImportedScene scene;
    std::vector<std::pair<int, int>> loadedExteriorCells;
    std::string loadedInteriorCellName;
    std::string diagnostics;
};

struct MorrowindBalmoraCookResult {
    ImportedScene scene;
    std::vector<std::pair<int, int>> balmoraCells;
    std::vector<std::pair<int, int>> includedCells;
    std::unordered_map<std::string, std::string> modelPathById;
    std::unordered_map<std::uint32_t, std::string> texturePathByLandscapeIndex;
};

struct MorrowindExteriorCellSelection {
    std::vector<std::string> exteriorCellNames;
    std::vector<std::pair<int, int>> explicitCells;
    int anchorNeighborRadius = 1;
    int corridorRadius = 0;
    bool connectAnchorsInOrder = false;
};

struct MorrowindExteriorCookResult {
    ImportedScene scene;
    std::vector<std::pair<int, int>> anchorCells;
    std::vector<std::pair<int, int>> includedCells;
    std::unordered_map<std::string, std::string> modelPathById;
    std::unordered_map<std::uint32_t, std::string> texturePathByLandscapeIndex;
};

struct MorrowindExteriorRuntimeLoadOptions {
    std::vector<std::pair<int, int>> cells;
    std::filesystem::path cacheRoot;
    bool useCellCache = true;
};

struct MorrowindExteriorRuntimeLoadResult {
    ImportedScene scene;
    std::vector<std::pair<int, int>> loadedCells;
    std::uint32_t cacheHitCount = 0;
    std::uint32_t cacheMissCount = 0;
};

bool saveImportedScene(const ImportedScene& scene, const std::filesystem::path& outputPath);
bool loadImportedScene(const std::filesystem::path& inputPath, ImportedScene& outScene);
bool loadImportedSceneRuntime(const std::filesystem::path& inputPath, ImportedScene& outScene);
const std::string& getImportedSceneLastError();
void buildImportedScenePackedRenderData(ImportedScene& scene);
MorrowindContentLoadOrder resolveMorrowindAutoContentLoadOrder(
    const std::filesystem::path& morrowindDataFilesPath
);
MorrowindContentLoadOrder resolveMorrowindOpenMwContentLoadOrder(
    const std::filesystem::path& openMwConfigPath,
    const std::filesystem::path& fallbackDataFilesPath = {}
);
MorrowindContentLoadOrder resolveMorrowindContentLoadOrder(
    const std::filesystem::path& fallbackDataFilesPath
);
bool createMorrowindContentSession(
    const MorrowindContentLoadOrder& loadOrder,
    const std::filesystem::path& cacheRoot,
    MorrowindContentSession& outSession
);
bool buildMorrowindScene(
    const MorrowindContentSession& session,
    const MorrowindSceneRequest& request,
    MorrowindSceneBuildResult& outResult
);
bool loadMorrowindTexture(
    const std::filesystem::path& morrowindDataFilesPath,
    const std::string& sourcePath,
    ImportedSceneTexture& outTexture
);
bool loadMorrowindActorCatalog(
    const std::filesystem::path& morrowindDataFilesPath,
    MorrowindActorCatalog& outCatalog
);
bool loadMorrowindActorCatalog(
    const MorrowindContentLoadOrder& loadOrder,
    MorrowindActorCatalog& outCatalog
);
bool loadMorrowindActorCatalog(
    const MorrowindContentSession& session,
    MorrowindActorCatalog& outCatalog
);
bool loadMorrowindEquipmentCatalog(
    const std::filesystem::path& morrowindDataFilesPath,
    MorrowindEquipmentCatalog& outCatalog
);
bool loadMorrowindEquipmentCatalog(
    const MorrowindContentLoadOrder& loadOrder,
    MorrowindEquipmentCatalog& outCatalog
);
bool loadMorrowindEquipmentCatalog(
    const MorrowindContentSession& session,
    MorrowindEquipmentCatalog& outCatalog
);
std::vector<std::string> resolveMorrowindNpcPartModelPaths(
    const MorrowindActorRecord& actor,
    const MorrowindEquipmentCatalog& equipmentCatalog
);
std::vector<MorrowindEquipmentCatalog::ResolvedActorPart> resolveMorrowindNpcParts(
    const MorrowindActorRecord& actor,
    const MorrowindEquipmentCatalog& equipmentCatalog
);
MorrowindActorPartMetadata toMorrowindActorPartMetadata(
    const MorrowindEquipmentCatalog::ResolvedActorPart& part
);

bool exportImportedSceneTerrainObj(const ImportedScene& scene, const std::filesystem::path& outputObjPath);

bool cookMorrowindExteriorRegionScene(
    const std::filesystem::path& morrowindDataFilesPath,
    const MorrowindExteriorCellSelection& selection,
    MorrowindExteriorCookResult& outResult
);
bool cookMorrowindExteriorRegionScene(
    const MorrowindContentLoadOrder& loadOrder,
    const MorrowindExteriorCellSelection& selection,
    MorrowindExteriorCookResult& outResult
);
bool loadMorrowindExteriorCellsRuntime(
    const std::filesystem::path& morrowindDataFilesPath,
    const MorrowindExteriorRuntimeLoadOptions& options,
    MorrowindExteriorRuntimeLoadResult& outResult
);
bool loadMorrowindExteriorCellsRuntime(
    const MorrowindContentLoadOrder& loadOrder,
    const MorrowindExteriorRuntimeLoadOptions& options,
    MorrowindExteriorRuntimeLoadResult& outResult
);
bool loadMorrowindExteriorCellsRuntime(
    const MorrowindContentSession& session,
    const MorrowindExteriorRuntimeLoadOptions& options,
    MorrowindExteriorRuntimeLoadResult& outResult
);
bool cookMorrowindBalmoraScene(
    const std::filesystem::path& morrowindDataFilesPath,
    MorrowindBalmoraCookResult& outResult
);
bool loadMorrowindBalmoraDoorReferences(
    const std::filesystem::path& morrowindDataFilesPath,
    std::vector<MorrowindDoorReference>& outDoors
);
bool loadMorrowindExteriorDoorReferences(
    const MorrowindContentSession& session,
    const std::vector<std::pair<int, int>>& exteriorCells,
    std::vector<MorrowindDoorReference>& outDoors
);
bool loadMorrowindInteriorDoorReferences(
    const std::filesystem::path& morrowindDataFilesPath,
    const std::string& cellName,
    std::vector<MorrowindDoorReference>& outDoors
);
bool loadMorrowindInteriorDoorReferences(
    const MorrowindContentSession& session,
    const std::string& cellName,
    std::vector<MorrowindDoorReference>& outDoors
);
bool saveMorrowindDoorCache(const MorrowindDoorCache& cache, const std::filesystem::path& outputPath);
bool loadMorrowindDoorCache(const std::filesystem::path& inputPath, MorrowindDoorCache& outCache);
bool cookMorrowindInteriorCellScene(
    const std::filesystem::path& morrowindDataFilesPath,
    const std::string& cellName,
    ImportedScene& outScene
);
bool cookMorrowindInteriorCellScene(
    const MorrowindContentSession& session,
    const std::string& cellName,
    ImportedScene& outScene
);

}  // namespace odai::importer
