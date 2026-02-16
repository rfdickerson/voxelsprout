#include "render/renderer.h"

#include "render/renderer_backend.h"

#include <memory>
#include <utility>

namespace render {

Renderer::Renderer()
    : m_backend(std::make_unique<RendererBackend>()) {}

Renderer::~Renderer() = default;

Renderer::Renderer(Renderer&&) noexcept = default;
Renderer& Renderer::operator=(Renderer&&) noexcept = default;

bool Renderer::init(GLFWwindow* window, const world::ChunkGrid& chunkGrid) {
    return m_backend->init(window, chunkGrid);
}

void Renderer::clearMagicaVoxelMeshes() {
    m_backend->clearMagicaVoxelMeshes();
}

bool Renderer::uploadMagicaVoxelMesh(
    const world::ChunkMeshData& mesh,
    float worldOffsetX,
    float worldOffsetY,
    float worldOffsetZ
) {
    return m_backend->uploadMagicaVoxelMesh(mesh, worldOffsetX, worldOffsetY, worldOffsetZ);
}

void Renderer::setVoxelBaseColorPalette(const std::array<std::uint32_t, 16>& paletteRgba) {
    m_backend->setVoxelBaseColorPalette(paletteRgba);
}

bool Renderer::updateChunkMesh(const world::ChunkGrid& chunkGrid) {
    return m_backend->updateChunkMesh(chunkGrid);
}

bool Renderer::updateChunkMesh(const world::ChunkGrid& chunkGrid, std::size_t chunkIndex) {
    return m_backend->updateChunkMesh(chunkGrid, chunkIndex);
}

bool Renderer::updateChunkMesh(const world::ChunkGrid& chunkGrid, std::span<const std::size_t> chunkIndices) {
    return m_backend->updateChunkMesh(chunkGrid, chunkIndices);
}

bool Renderer::useSpatialPartitioningQueries() const {
    return m_backend->useSpatialPartitioningQueries();
}

world::ClipmapConfig Renderer::clipmapQueryConfig() const {
    return m_backend->clipmapQueryConfig();
}

void Renderer::setSpatialQueryStats(bool used, const world::SpatialQueryStats& stats, std::uint32_t visibleChunkCount) {
    m_backend->setSpatialQueryStats(used, stats, visibleChunkCount);
}

void Renderer::renderFrame(
    const world::ChunkGrid& chunkGrid,
    const sim::Simulation& simulation,
    const CameraPose& camera,
    const VoxelPreview& preview,
    float simulationAlpha,
    std::span<const std::size_t> visibleChunkIndices
) {
    m_backend->renderFrame(chunkGrid, simulation, camera, preview, simulationAlpha, visibleChunkIndices);
}

void Renderer::setDebugUiVisible(bool visible) {
    m_backend->setDebugUiVisible(visible);
}

bool Renderer::isDebugUiVisible() const {
    return m_backend->isDebugUiVisible();
}

void Renderer::setFrameStatsVisible(bool visible) {
    m_backend->setFrameStatsVisible(visible);
}

bool Renderer::isFrameStatsVisible() const {
    return m_backend->isFrameStatsVisible();
}

void Renderer::setSunAngles(float yawDegrees, float pitchDegrees) {
    m_backend->setSunAngles(yawDegrees, pitchDegrees);
}

float Renderer::cameraFovDegrees() const {
    return m_backend->cameraFovDegrees();
}

void Renderer::shutdown() {
    m_backend->shutdown();
}

} // namespace render
