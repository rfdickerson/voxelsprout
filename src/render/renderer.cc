#include "render/renderer.h"

#include "render/backend/vulkan/renderer_backend.h"

#include <memory>
#include <utility>

namespace odai::render {

Renderer::Renderer()
    : m_backend(std::make_unique<RendererBackend>()) {}

Renderer::~Renderer() = default;

Renderer::Renderer(Renderer&&) noexcept = default;
Renderer& Renderer::operator=(Renderer&&) noexcept = default;

bool Renderer::init(GLFWwindow* window) {
    return m_backend->init(window);
}

void Renderer::clearImportedSceneMeshes() {
    m_backend->clearImportedSceneMeshes();
}

bool Renderer::reserveImportedSceneGeometry(
    std::uint64_t vertexCapacity, std::uint64_t indexCapacity) {
    return m_backend->reserveImportedSceneGeometry(vertexCapacity, indexCapacity);
}

std::size_t Renderer::addImportedSceneChunk(const odai::importer::ImportedScene& scene) {
    const std::size_t chunkIndex = m_backend->addImportedSceneChunk(scene);
    return chunkIndex == RendererBackend::kInvalidImportedChunkIndex
        ? kInvalidImportedChunkIndex
        : chunkIndex;
}

void Renderer::removeImportedSceneChunk(std::size_t chunkIndex) {
    if (chunkIndex == kInvalidImportedChunkIndex) {
        return;
    }
    m_backend->removeImportedSceneChunkAt(chunkIndex);
}

bool Renderer::waitForImportedSceneUploads() {
    return m_backend->waitForImportedSceneUploads();
}

std::size_t Renderer::liveImportedSceneChunkCount() const {
    return m_backend->liveImportedSceneChunkCount();
}

std::size_t Renderer::importedLocalLightCount() const {
    return m_backend->importedLocalLightCount();
}

bool Renderer::uploadImportedScene(const odai::importer::ImportedScene& scene) {
    return m_backend->uploadImportedScene(scene);
}

bool Renderer::uploadSkinnedMeshTemplate(std::uint32_t instanceIndex, const ImportedSkinnedMeshTemplate& meshTemplate) {
    return m_backend->uploadSkinnedMeshTemplate(instanceIndex, meshTemplate);
}

std::vector<std::uint32_t> Renderer::uploadSkinnedActorTextures(
    std::uint32_t instanceIndex, const std::vector<odai::importer::ImportedSceneTexture>& textures
) {
    return m_backend->uploadSkinnedActorTextures(instanceIndex, textures);
}

void Renderer::setSkinnedActorVisible(std::uint32_t instanceIndex, bool visible) {
    m_backend->setSkinnedActorVisible(instanceIndex, visible);
}

void Renderer::setSkinnedActorPose(std::uint32_t instanceIndex, const ImportedSkinnedActorFrameData& pose) {
    m_backend->setSkinnedActorPose(instanceIndex, pose);
}

RuntimeDeltaApplyStats Renderer::applyRuntimeRenderDeltas(
    const odai::bethesda::RuntimeRenderDeltaBatch& deltas) {
    RuntimeDeltaApplyStats stats;
    stats.received = deltas.size();
    for (const odai::bethesda::RuntimeRenderDelta& delta : deltas) {
        if (delta.skinningSlot != 0xffffffffu &&
            (delta.changes & odai::bethesda::RuntimeRenderVisibility) != 0u) {
            m_backend->setSkinnedActorVisible(delta.skinningSlot, delta.visible);
            ++stats.skinnedVisibilityApplied;
        } else {
            ++stats.unresolved;
        }
    }
    return stats;
}

void Renderer::setSkinningDebugBypass(bool bypass) {
    m_backend->setSkinningDebugBypass(bypass);
}

void Renderer::setTaaEnabled(bool enabled) {
    m_backend->setTaaEnabled(enabled);
}

void Renderer::setRayTracingEnabled(bool enabled) {
    m_backend->setRayTracingEnabled(enabled);
}

bool Renderer::captureFrameToFile(const std::string& outputPath) {
    return m_backend->captureLastFrameToFile(outputPath);
}

bool Renderer::captureFrameRgb(std::vector<std::uint8_t>& outRgb,
                               std::uint32_t& outWidth,
                               std::uint32_t& outHeight) {
    return m_backend->captureLastFrameRgb(outRgb, outWidth, outHeight);
}

void Renderer::setNeutralColorGrading() {
    m_backend->setNeutralColorGrading();
}

void Renderer::setColorGrading(const ColorGradingSettings& settings) {
    m_backend->setColorGrading(settings);
}

void Renderer::setAutoExposureEnabled(bool enabled) {
    m_backend->setAutoExposureEnabled(enabled);
}

void Renderer::setAutoExposureKeyValue(float keyValue) {
    m_backend->setAutoExposureKeyValue(keyValue);
}

bool Renderer::isAutoExposureEnabled() const {
    return m_backend->isAutoExposureEnabled();
}

void Renderer::setDebugView(DebugView view) {
    m_backend->setDebugView(view);
}

DebugView Renderer::debugView() const {
    return m_backend->debugView();
}

void Renderer::setSunShaftsEnabled(bool enabled) {
    m_backend->setSunShaftsEnabled(enabled);
}

bool Renderer::isSunShaftsEnabled() const {
    return m_backend->isSunShaftsEnabled();
}

void Renderer::setMsaaSamples(std::uint32_t samples) {
    m_backend->setRequestedMsaaSamples(samples);
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

void Renderer::renderFrame(const CameraPose& camera) {
    m_backend->renderFrame(camera);
}

void Renderer::setUpscalerSettings(const UpscalerSettings& settings) {
    m_backend->setUpscalerSettings(settings);
}

UpscalerStatus Renderer::upscalerStatus() const {
    return m_backend->upscalerStatus();
}

void Renderer::setDebugUiVisible(bool visible) {
    m_backend->setDebugUiVisible(visible);
}

bool Renderer::isDebugUiVisible() const {
    return m_backend->isDebugUiVisible();
}

void Renderer::setDebugUiMode(DebugUiMode mode) {
    m_backend->setDebugUiMode(mode);
}

DebugUiMode Renderer::debugUiMode() const {
    return m_backend->debugUiMode();
}

void Renderer::setDebugStatGroups(std::vector<DebugStatGroup> groups) {
    m_backend->setDebugStatGroups(std::move(groups));
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

UiRenderStats Renderer::uiRenderStats() const {
    return m_backend->uiRenderStats();
}

void Renderer::setVertexAoEnabled(bool enabled) {
    m_backend->setVertexAoEnabled(enabled);
}

bool Renderer::isVertexAoEnabled() const {
    return m_backend->isVertexAoEnabled();
}

void Renderer::setImportedMaterial(std::uint32_t index,
                                   const odai::importer::ImportedSceneMaterial& material) {
    m_backend->setImportedMaterial(index, material);
}

void Renderer::setImportedMaterialTable(
    const std::vector<odai::importer::ImportedSceneMaterial>& materials) {
    m_backend->setImportedMaterialTable(materials);
}

void Renderer::setSsaoEnabled(bool enabled) {
    m_backend->setSsaoEnabled(enabled);
}

bool Renderer::isSsaoEnabled() const {
    return m_backend->isSsaoEnabled();
}

void Renderer::setAmbientOcclusionTuning(float radius, float bias, float intensity) {
    m_backend->setAmbientOcclusionTuning(radius, bias, intensity);
}

void Renderer::setAmbientOcclusionFineScale(float fineRadiusScale) {
    m_backend->setAmbientOcclusionFineScale(fineRadiusScale);
}

void Renderer::setAmbientOcclusionMode(AoMode mode) {
    m_backend->setAmbientOcclusionMode(mode);
}

AoMode Renderer::ambientOcclusionMode() const {
    return m_backend->ambientOcclusionMode();
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

void Renderer::setVisualTimeSeconds(float seconds) {
    m_backend->setVisualTimeSeconds(seconds);
}

void Renderer::setWeatherSky(const WeatherSkyParams& params) {
    m_backend->setWeatherSky(params);
}

void Renderer::setMountainCloudVolume(
    float centerX, float centerZ, float baseHeight, float topHeight,
    float fullRadius, float outerRadius) {
    m_backend->setMountainCloudVolume(
        centerX, centerZ, baseHeight, topHeight, fullRadius, outerRadius);
}

void Renderer::setWeatherClouds(const WeatherCloudTextures& clouds) {
    m_backend->setWeatherClouds(clouds);
}

void Renderer::setWeatherCloudMesh(const WeatherCloudMesh& mesh) {
    m_backend->setWeatherCloudMesh(mesh);
}

void Renderer::setTonemapSettings(const TonemapSettings& settings) {
    m_backend->setTonemapSettings(settings);
}

TonemapSettings Renderer::tonemapSettings() const {
    return m_backend->tonemapSettings();
}

void Renderer::setImportedPbrDefaults(const ImportedPbrDefaults& defaults) {
    m_backend->setImportedPbrDefaults(defaults);
}

void Renderer::setDepthOfField(bool enabled, float focusDistance, float focusRange,
                               float maxRadiusPixels, float nearBlurScale) {
    m_backend->setDepthOfField(enabled, focusDistance, focusRange, maxRadiusPixels, nearBlurScale);
}

void Renderer::setImportedSceneDebugState(bool showTerrain, bool showStatics, bool showTextures, bool flatShading, bool waterDebug) {
    m_backend->setImportedSceneDebugState(showTerrain, showStatics, showTextures, flatShading, waterDebug);
}

void Renderer::setImportedInteriorLighting(const ImportedInteriorLighting& lighting) {
    m_backend->setImportedInteriorLighting(lighting);
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
