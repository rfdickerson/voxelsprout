#pragma once

#include "sim/simulation.h"
#include "world/chunk_grid.h"
#include "world/chunk_mesher.h"
#include "world/clipmap_index.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

struct GLFWwindow;

namespace render {

struct CameraPose {
    float x;
    float y;
    float z;
    float yawDegrees;
    float pitchDegrees;
    float fovDegrees;
};

struct VoxelPreview {
    enum class Mode {
        Add,
        Remove
    };

    bool visible = false;
    int x = 0;
    int y = 0;
    int z = 0;
    int brushSize = 1;
    Mode mode = Mode::Add;
    bool faceVisible = false;
    int faceX = 0;
    int faceY = 0;
    int faceZ = 0;
    uint32_t faceId = 0;
    bool pipeStyle = false;
    float pipeAxisX = 0.0f;
    float pipeAxisY = 1.0f;
    float pipeAxisZ = 0.0f;
    float pipeRadius = 0.45f;
    float pipeStyleId = 0.0f;
};

class RendererBackend;

class Renderer {
public:
    Renderer();
    ~Renderer();
    Renderer(Renderer&&) noexcept;
    Renderer& operator=(Renderer&&) noexcept;
    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    bool init(GLFWwindow* window, const world::ChunkGrid& chunkGrid);
    void clearMagicaVoxelMeshes();
    bool uploadMagicaVoxelMesh(const world::ChunkMeshData& mesh, float worldOffsetX, float worldOffsetY, float worldOffsetZ);
    void setVoxelBaseColorPalette(const std::array<std::uint32_t, 16>& paletteRgba);
    bool updateChunkMesh(const world::ChunkGrid& chunkGrid);
    bool updateChunkMesh(const world::ChunkGrid& chunkGrid, std::size_t chunkIndex);
    bool updateChunkMesh(const world::ChunkGrid& chunkGrid, std::span<const std::size_t> chunkIndices);
    bool useSpatialPartitioningQueries() const;
    world::ClipmapConfig clipmapQueryConfig() const;
    void setSpatialQueryStats(bool used, const world::SpatialQueryStats& stats, std::uint32_t visibleChunkCount);
    void renderFrame(
        const world::ChunkGrid& chunkGrid,
        const sim::Simulation& simulation,
        const CameraPose& camera,
        const VoxelPreview& preview,
        float simulationAlpha,
        std::span<const std::size_t> visibleChunkIndices
    );
    void setDebugUiVisible(bool visible);
    bool isDebugUiVisible() const;
    void setFrameStatsVisible(bool visible);
    bool isFrameStatsVisible() const;
    void setSunAngles(float yawDegrees, float pitchDegrees);
    float cameraFovDegrees() const;
    void shutdown();

private:
    std::unique_ptr<RendererBackend> m_backend;
};

} // namespace render
