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

    // Streaming: add one more resident chunk without disturbing those already
    // loaded, and evict one by the index add returned. Unlike
    // uploadImportedScene(), neither replaces the scene or waits for the device
    // to go idle, so both are safe to call between frames while the player moves.
    //
    // Geometry is suballocated from shared arenas and textures are reference
    // counted by source path, so a texture used by several chunks uploads once
    // and survives until the last chunk referencing it is evicted.
    //
    // addImportedSceneChunk returns kInvalidImportedChunkIndex on failure.
    // Indices stay valid across evictions.
    static constexpr std::size_t kInvalidImportedChunkIndex = static_cast<std::size_t>(-1);
    std::size_t addImportedSceneChunk(const odai::importer::ImportedScene& scene);
    void removeImportedSceneChunk(std::size_t chunkIndex);
    [[nodiscard]] std::size_t liveImportedSceneChunkCount() const;
    [[nodiscard]] std::size_t importedLocalLightCount() const;

    // Named material library, indexed by vertex flag bits 24-31 (see
    // import/imported_material.h). Index 0 is a reserved sentinel and is
    // ignored; out-of-range indices are ignored.
    //
    // setImportedMaterial() writes one 32-byte record and touches no geometry:
    // this is the live-edit path, and the reason it exists is that changing a
    // coefficient must NOT go through uploadImportedScene(), which deep-copies
    // the whole scene and drains the device mid-frame.
    void setImportedMaterial(std::uint32_t index,
                             const odai::importer::ImportedSceneMaterial& material);
    void setImportedMaterialTable(
        const std::vector<odai::importer::ImportedSceneMaterial>& materials);
    // GPU skeletal animation (Dragon Age: Origins touchstone, see
    // docs/ROADMAP.md). Uploads a skinned mesh's rest-pose geometry once per
    // instance slot, device-local; pose it per-frame via setSkinnedActorPose
    // without re-uploading geometry. instanceIndex must be < kMaxSkinnedInstances
    // (see renderer_types.h) -- each slot is fully independent (own template,
    // own pose, own draws), sized for a small party, not a mass-battle crowd.
    bool uploadSkinnedMeshTemplate(std::uint32_t instanceIndex, const ImportedSkinnedMeshTemplate& meshTemplate);
    void setSkinnedActorPose(std::uint32_t instanceIndex, const ImportedSkinnedActorFrameData& pose);
    void setSkinningDebugBypass(bool bypass);
    // Temporal AA (camera reprojection; static world). Off by default.
    void setTaaEnabled(bool enabled);
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
    // Queue meshes built off the render thread (world::ChunkMeshScheduler) for
    // upload on the next frame; the renderer skips its inline mesher for them.
    bool uploadChunkMeshes(const odai::world::ChunkGrid& chunkGrid, std::vector<odai::world::ChunkMeshResult> results);
    // Meshing mode the renderer's own full-rebuild path uses; mirror it in any
    // off-thread mesher so both paths produce the same geometry.
    [[nodiscard]] odai::world::MeshingOptions chunkMeshingOptions() const;
    bool useSpatialPartitioningQueries() const;
    odai::world::ClipmapConfig clipmapQueryConfig() const;
    void setSpatialQueryStats(bool used, const odai::world::SpatialQueryStats& stats, std::uint32_t visibleChunkCount);
    void setStrategyMapMode(bool enabled);
    // App-level opt-out of the ray-traced BLAS/TLAS scene (shadows/reflections/
    // voxel GI over ray queries). false unconditionally skips building and
    // tearing down per-scene acceleration structures on every uploadImportedScene()
    // call, even on hardware that supports it -- for apps (like CityBuilder) that
    // never enable ShadowMode::RayTraced/Auto or RT-backed voxel GI, that
    // per-upload BLAS churn is pure wasted GPU work. true only restores whatever
    // the hardware/driver capability probe already determined at init() time; it
    // can never force RT on where it isn't supported. Call any time after init().
    void setRayTracingEnabled(bool enabled);
    // App-level opt-out of the voxel GI dispatch sequence (occupancy, sky exposure,
    // surface/ReSTIR, inject, propagate). Like setRayTracingEnabled this can only opt
    // further out, never force GI on where the device can't run it. On by default.
    //
    // The GI volume is 64 world units wide and follows the camera, so any world at a
    // much larger scale gets no contribution from it while still paying for every
    // dispatch — imported Bethesda scenes run ~70 units/metre, which puts the whole
    // volume under a metre across. Turn it off for those. Call any time after init().
    // Histogram-driven eye adaptation, off by default. With it off the scene
    // renders at a fixed exposure, so content whose light levels differ from
    // that baseline comes out uniformly too dark or too bright. Worth enabling
    // for any scene with a wide dynamic range or a day/night cycle.
    void setAutoExposureEnabled(bool enabled);
    [[nodiscard]] bool isAutoExposureEnabled() const;
    void setVoxelGiEnabled(bool enabled);
    [[nodiscard]] bool isVoxelGiEnabled() const;
    // App-level opt-out of the sun shaft pass (a 20-tap radial march per pixel at AO
    // resolution). On by default; when off the shaft texture reads as black and the
    // main pass is otherwise unchanged. Call any time after init().
    void setSunShaftsEnabled(bool enabled);
    [[nodiscard]] bool isSunShaftsEnabled() const;
    // Opt-in UI-only rendering for showcase/tooling executables. Skips building the
    // 3D scene pipelines (pipe/imported/sky-cloud/water/grass, SSAO, hex terrain)
    // those tools cannot use. Must be called BEFORE init(); off by default.
    void setMinimalRenderMode(bool enabled);
    // MSAA sample count (1, 2, 4, 8), clamped to device support. Must be called
    // BEFORE init(): it sizes the render targets and is baked into every
    // pipeline. 4 is the default. On a fill-rate-bound device this is the
    // cheapest large reduction in main-pass cost available.
    void setMsaaSamples(std::uint32_t samples);
    // Writes the last presented frame to a binary PPM (convert with e.g.
    // `ffmpeg -i shot.ppm shot.png`). Diagnostic, not a feature: it stalls the
    // device, so call it once rather than per frame. false if nothing has been
    // presented yet. See frame_capture.cc for why this lives in the engine
    // instead of relying on an external screenshot tool.
    bool captureFrameToFile(const std::string& outputPath);
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
    void setAmbientOcclusionTuning(float radius, float bias, float intensity);

    // Multi-scale AO. The fine march runs at radius * fineRadiusScale and is
    // combined with the coarse one by min(), so contact darkening and wide
    // occlusion both survive instead of one radius having to serve both.
    // Zero (or >= 1) runs a single march and costs nothing extra.
    void setAmbientOcclusionFineScale(float fineRadiusScale);
    // Picks the AO estimator (see AoMode in renderer_types.h). Orthogonal to
    // setSsaoEnabled(), which gates the pass entirely — both must be on for AO
    // to appear. Radius is in world units, so re-tune it with the mode if the
    // world scale is unusual (setAmbientOcclusionTuning).
    void setAmbientOcclusionMode(AoMode mode);
    [[nodiscard]] AoMode ambientOcclusionMode() const;
    void setShadowSettings(const ShadowSettings& settings);
    [[nodiscard]] ShadowSettings shadowSettings() const;
    [[nodiscard]] ShadowStats shadowStats() const;
    void setSunAngles(float yawDegrees, float pitchDegrees);
    // Authored sky from a Fallout WTHR record. Default-constructed params
    // (weight 0) restore the procedural sky.
    void setWeatherSky(const WeatherSkyParams& params);
    // Cloud layer textures for the active weather. Upload-heavy, so call it
    // when the weather changes rather than per frame.
    void setWeatherClouds(const WeatherCloudTextures& clouds);
    // Tone curve for the post pass. Default is ACES, so this is inert
    // unless a game selects otherwise.
    void setTonemapSettings(const TonemapSettings& settings);
    // Post-process depth of field (tilt-shift/diorama look). Focus is a view
    // distance in world units; geometry within +-focusRange stays sharp and
    // blur ramps to maxRadiusPixels beyond it.
    void setDepthOfField(bool enabled, float focusDistance, float focusRange, float maxRadiusPixels);
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
