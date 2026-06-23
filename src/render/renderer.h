#pragma once

#include "sim/simulation.h"
#include "import/gpu_scene.h"
#include "import/hex_terrain_data.h"
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

namespace odai::ui {
struct UiDrawData;
using UiTextureId = std::uint32_t;
}

namespace odai::render {

class RendererBackend;

class Renderer {
public:
    Renderer();
    ~Renderer();
    Renderer(Renderer&&) noexcept;
    Renderer& operator=(Renderer&&) noexcept;
    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    bool init(GLFWwindow* window, const odai::world::ChunkGrid& chunkGrid);
    void clearMagicaVoxelMeshes();
    bool uploadMagicaVoxelMesh(const odai::world::ChunkMeshData& mesh, float worldOffsetX, float worldOffsetY, float worldOffsetZ);
    void clearGpuScene();
    bool uploadGpuScene(const odai::importer::GpuSceneAsset& scene);
    void clearImportedSceneMeshes();
    bool uploadImportedScene(const odai::importer::ImportedScene& scene);
    // GPU-instanced, tessellated, height-displaced hex land surface (strategy map).
    // hexTerrainReady() reports whether the device created the pipeline (tessellation
    // support); the caller keeps the flat imported-static land otherwise.
    void clearHexTerrain();
    bool uploadHexTerrain(const odai::importer::HexTerrainData& data);
    [[nodiscard]] bool hexTerrainReady() const;
    void setHexTerrainEnabled(bool enabled);
    void setVoxelBaseColorPalette(const std::array<std::uint32_t, 16>& paletteRgba);
    bool updateChunkMesh(const odai::world::ChunkGrid& chunkGrid);
    bool updateChunkMesh(const odai::world::ChunkGrid& chunkGrid, std::size_t chunkIndex);
    bool updateChunkMesh(const odai::world::ChunkGrid& chunkGrid, std::span<const std::size_t> chunkIndices);
    bool useSpatialPartitioningQueries() const;
    odai::world::ClipmapConfig clipmapQueryConfig() const;
    void setSpatialQueryStats(bool used, const odai::world::SpatialQueryStats& stats, std::uint32_t visibleChunkCount);
    void setStrategyMapMode(bool enabled);
    void setGameplayUiState(const GameplayUiState& state);
    // Hand the renderer the UI geometry to draw over the scene this frame.
    void setUiDrawData(const odai::ui::UiDrawData& drawData);
    // Upload the UI font's R8 coverage atlas (call once after init / on font change).
    bool setUiFontAtlas(const std::uint8_t* pixels, std::uint32_t width, std::uint32_t height);
    // Register an extra UI font atlas (e.g. bold/italic) and return its texture id
    // (kUiNoTexture on failure). Assign it to the Font via Font::setTextureId.
    odai::ui::UiTextureId registerUiFontAtlas(const std::uint8_t* pixels, std::uint32_t width, std::uint32_t height);
    // Register an RGBA8 UI texture (e.g. a 9-slice window frame) and return its
    // texture id (kUiNoTexture on failure).
    odai::ui::UiTextureId registerUiTextureRgba8(const std::uint8_t* pixels, std::uint32_t width, std::uint32_t height);
    // Same as registerUiTextureRgba8 but generates a full mip chain via CPU box-filter.
    odai::ui::UiTextureId registerUiTextureRgba8Mipmapped(const std::uint8_t* pixels, std::uint32_t width, std::uint32_t height);
    void renderFrame(
        const odai::world::ChunkGrid& chunkGrid,
        const odai::sim::Simulation& simulation,
        const CameraPose& camera,
        const VoxelPreview& preview,
        float simulationAlpha,
        std::span<const std::size_t> visibleChunkIndices,
        const ImportedActorFrameData* importedActors = nullptr
    );
    void setDebugUiVisible(bool visible);
    bool isDebugUiVisible() const;
    void setFrameStatsVisible(bool visible);
    bool isFrameStatsVisible() const;
    void setFramePacingSettings(const FramePacingSettings& settings);
    [[nodiscard]] FramePacingSettings framePacingSettings() const;
    [[nodiscard]] FramePacingStats framePacingStats() const;
    [[nodiscard]] UiRenderStats uiRenderStats() const;
    void setVertexAoEnabled(bool enabled);
    [[nodiscard]] bool isVertexAoEnabled() const;
    void setSsaoEnabled(bool enabled);
    [[nodiscard]] bool isSsaoEnabled() const;
    void setShadowSettings(const ShadowSettings& settings);
    [[nodiscard]] ShadowSettings shadowSettings() const;
    [[nodiscard]] ShadowStats shadowStats() const;
    void setSunAngles(float yawDegrees, float pitchDegrees);
    void setImportedSceneDebugState(bool showTerrain, bool showStatics, bool showTextures, bool flatShading, bool waterDebug);
    void setImportedSceneInteriorMode(bool enabled);
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

} // namespace odai::render
