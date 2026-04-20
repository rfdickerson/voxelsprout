#pragma once

#include "sim/simulation.h"
#include "import/imported_scene.h"
#include "world/chunk_grid.h"
#include "world/chunk_mesher.h"
#include "world/clipmap_index.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

#include "render/renderer_types.h"

struct GLFWwindow;

namespace voxelsprout::render {

class RendererBackend;

class Renderer {
public:
    Renderer();
    ~Renderer();
    Renderer(Renderer&&) noexcept;
    Renderer& operator=(Renderer&&) noexcept;
    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    bool init(GLFWwindow* window, const voxelsprout::world::ChunkGrid& chunkGrid);
    void clearMagicaVoxelMeshes();
    bool uploadMagicaVoxelMesh(const voxelsprout::world::ChunkMeshData& mesh, float worldOffsetX, float worldOffsetY, float worldOffsetZ);
    void clearImportedSceneMeshes();
    bool uploadImportedScene(const voxelsprout::importer::ImportedScene& scene);
    void setVoxelBaseColorPalette(const std::array<std::uint32_t, 16>& paletteRgba);
    bool updateChunkMesh(const voxelsprout::world::ChunkGrid& chunkGrid);
    bool updateChunkMesh(const voxelsprout::world::ChunkGrid& chunkGrid, std::size_t chunkIndex);
    bool updateChunkMesh(const voxelsprout::world::ChunkGrid& chunkGrid, std::span<const std::size_t> chunkIndices);
    bool useSpatialPartitioningQueries() const;
    voxelsprout::world::ClipmapConfig clipmapQueryConfig() const;
    void setSpatialQueryStats(bool used, const voxelsprout::world::SpatialQueryStats& stats, std::uint32_t visibleChunkCount);
    void setGameplayUiState(const GameplayUiState& state);
    void renderFrame(
        const voxelsprout::world::ChunkGrid& chunkGrid,
        const voxelsprout::sim::Simulation& simulation,
        const CameraPose& camera,
        const VoxelPreview& preview,
        float simulationAlpha,
        std::span<const std::size_t> visibleChunkIndices
    );
    void setDebugUiVisible(bool visible);
    bool isDebugUiVisible() const;
    void setFrameStatsVisible(bool visible);
    bool isFrameStatsVisible() const;
    void setFramePacingSettings(const FramePacingSettings& settings);
    [[nodiscard]] FramePacingSettings framePacingSettings() const;
    [[nodiscard]] FramePacingStats framePacingStats() const;
    void setVertexAoEnabled(bool enabled);
    [[nodiscard]] bool isVertexAoEnabled() const;
    void setSsaoEnabled(bool enabled);
    [[nodiscard]] bool isSsaoEnabled() const;
    void setShadowSettings(const ShadowSettings& settings);
    [[nodiscard]] ShadowSettings shadowSettings() const;
    [[nodiscard]] ShadowStats shadowStats() const;
    void setSunAngles(float yawDegrees, float pitchDegrees);
    void setImportedSceneDebugState(bool showTerrain, bool showStatics, bool showTextures, bool flatShading, bool waterDebug);
    void importedSceneDebugState(
        bool& outShowTerrain,
        bool& outShowStatics,
        bool& outShowTextures,
        bool& outFlatShading,
        bool& outWaterDebug
    ) const;
    float cameraFovDegrees() const;
    void shutdown();

private:
    std::unique_ptr<RendererBackend> m_backend;
};

} // namespace voxelsprout::render
