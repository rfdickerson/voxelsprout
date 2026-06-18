#include "render/renderer.h"

#include "render/backend/render_backend_selector.h"

#include <memory>
#include <utility>

namespace odai::render {

Renderer::Renderer()
    : m_backend(std::make_unique<RendererBackend>()) {}

Renderer::~Renderer() = default;

Renderer::Renderer(Renderer&&) noexcept = default;
Renderer& Renderer::operator=(Renderer&&) noexcept = default;

bool Renderer::init(GLFWwindow* window, const odai::world::ChunkGrid& chunkGrid) {
    return m_backend->init(window, chunkGrid);
}

void Renderer::clearMagicaVoxelMeshes() {
    m_backend->clearMagicaVoxelMeshes();
}

void Renderer::clearGpuScene() {
    m_backend->clearGpuScene();
}

void Renderer::clearImportedSceneMeshes() {
    m_backend->clearImportedSceneMeshes();
}

bool Renderer::uploadMagicaVoxelMesh(
    const odai::world::ChunkMeshData& mesh,
    float worldOffsetX,
    float worldOffsetY,
    float worldOffsetZ
) {
    return m_backend->uploadMagicaVoxelMesh(mesh, worldOffsetX, worldOffsetY, worldOffsetZ);
}

bool Renderer::uploadGpuScene(const odai::importer::GpuSceneAsset& scene) {
    return m_backend->uploadGpuScene(scene);
}

bool Renderer::uploadImportedScene(const odai::importer::ImportedScene& scene) {
    return m_backend->uploadImportedScene(scene);
}

void Renderer::setVoxelBaseColorPalette(const std::array<std::uint32_t, 16>& paletteRgba) {
    m_backend->setVoxelBaseColorPalette(paletteRgba);
}

bool Renderer::updateChunkMesh(const odai::world::ChunkGrid& chunkGrid) {
    return m_backend->updateChunkMesh(chunkGrid);
}

bool Renderer::updateChunkMesh(const odai::world::ChunkGrid& chunkGrid, std::size_t chunkIndex) {
    return m_backend->updateChunkMesh(chunkGrid, chunkIndex);
}

bool Renderer::updateChunkMesh(const odai::world::ChunkGrid& chunkGrid, std::span<const std::size_t> chunkIndices) {
    return m_backend->updateChunkMesh(chunkGrid, chunkIndices);
}

bool Renderer::useSpatialPartitioningQueries() const {
    return m_backend->useSpatialPartitioningQueries();
}

odai::world::ClipmapConfig Renderer::clipmapQueryConfig() const {
    return m_backend->clipmapQueryConfig();
}

void Renderer::setSpatialQueryStats(bool used, const odai::world::SpatialQueryStats& stats, std::uint32_t visibleChunkCount) {
    m_backend->setSpatialQueryStats(used, stats, visibleChunkCount);
}

void Renderer::setStrategyMapMode(bool enabled) {
    m_backend->setStrategyMapMode(enabled);
}

void Renderer::setGameplayUiState(const GameplayUiState& state) {
    m_backend->setGameplayUiState(state);
}

void Renderer::setUiDrawData(const odai::ui::UiDrawData& drawData) {
    m_backend->setUiDrawData(drawData);
}

bool Renderer::setUiFontAtlas(const std::uint8_t* pixels, std::uint32_t width, std::uint32_t height) {
    return m_backend->setUiFontAtlas(pixels, width, height);
}

odai::ui::UiTextureId Renderer::registerUiFontAtlas(const std::uint8_t* pixels, std::uint32_t width,
                                                    std::uint32_t height) {
    return m_backend->registerUiFontAtlas(pixels, width, height);
}

odai::ui::UiTextureId Renderer::registerUiTextureRgba8(const std::uint8_t* pixels, std::uint32_t width,
                                                       std::uint32_t height) {
    return m_backend->registerUiTextureRgba8(pixels, width, height);
}

odai::ui::UiTextureId Renderer::registerUiTextureRgba8Mipmapped(const std::uint8_t* pixels,
                                                                  std::uint32_t width,
                                                                  std::uint32_t height) {
    return m_backend->registerUiTextureRgba8Mipmapped(pixels, width, height);
}

void Renderer::renderFrame(
    const odai::world::ChunkGrid& chunkGrid,
    const odai::sim::Simulation& simulation,
    const CameraPose& camera,
    const VoxelPreview& preview,
    float simulationAlpha,
    std::span<const std::size_t> visibleChunkIndices,
    const ImportedActorFrameData* importedActors
) {
    m_backend->renderFrame(chunkGrid, simulation, camera, preview, simulationAlpha, visibleChunkIndices, importedActors);
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

void Renderer::setFramePacingSettings(const FramePacingSettings& settings) {
    m_backend->setFramePacingSettings(settings);
}

FramePacingSettings Renderer::framePacingSettings() const {
    return m_backend->framePacingSettings();
}

FramePacingStats Renderer::framePacingStats() const {
    return m_backend->framePacingStats();
}

void Renderer::setVertexAoEnabled(bool enabled) {
    m_backend->setVertexAoEnabled(enabled);
}

bool Renderer::isVertexAoEnabled() const {
    return m_backend->isVertexAoEnabled();
}

void Renderer::setSsaoEnabled(bool enabled) {
    m_backend->setSsaoEnabled(enabled);
}

bool Renderer::isSsaoEnabled() const {
    return m_backend->isSsaoEnabled();
}

void Renderer::setShadowSettings(const ShadowSettings& settings) {
    m_backend->setShadowSettings(settings);
}

ShadowSettings Renderer::shadowSettings() const {
    return m_backend->shadowSettings();
}

ShadowStats Renderer::shadowStats() const {
    return m_backend->shadowStats();
}

void Renderer::setSunAngles(float yawDegrees, float pitchDegrees) {
    m_backend->setSunAngles(yawDegrees, pitchDegrees);
}

void Renderer::setImportedSceneDebugState(bool showTerrain, bool showStatics, bool showTextures, bool flatShading, bool waterDebug) {
    m_backend->setImportedSceneDebugState(showTerrain, showStatics, showTextures, flatShading, waterDebug);
}

void Renderer::setImportedSceneInteriorMode(bool enabled) {
    m_backend->setImportedSceneInteriorMode(enabled);
}

void Renderer::importedSceneDebugState(
    bool& outShowTerrain,
    bool& outShowStatics,
    bool& outShowTextures,
    bool& outFlatShading,
    bool& outWaterDebug
) const {
    m_backend->importedSceneDebugState(
        outShowTerrain,
        outShowStatics,
        outShowTextures,
        outFlatShading,
        outWaterDebug);
}

float Renderer::cameraFovDegrees() const {
    return m_backend->cameraFovDegrees();
}

void Renderer::shutdown() {
    m_backend->shutdown();
}

} // namespace odai::render
