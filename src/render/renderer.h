#pragma once

#include "import/imported_scene.h"
#include <cstddef>
#include <cstdint>
#include <memory>

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

    bool init(GLFWwindow* window);
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
    // Capture-only readiness fence. Waits on the upload timeline values already
    // signalled by chunk staging; it never drains unrelated device work.
    bool waitForImportedSceneUploads();
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
    // Uploads a skinned actor's textures and returns one bindless slot per
    // input, in order (0xffffffff where a texture was unusable). Those slots
    // are what ImportedSkinnedMeshVertex::textureIndex must hold: a skinned
    // template's vertices go to the GPU verbatim, with none of the scene-index
    // remapping addImportedSceneChunk does for world geometry, so there is no
    // other way for a skinned actor to be textured. Call before
    // uploadSkinnedMeshTemplate and write the slots into the vertices.
    std::vector<std::uint32_t> uploadSkinnedActorTextures(
        std::uint32_t instanceIndex,
        const std::vector<odai::importer::ImportedSceneTexture>& textures);
    // Removes an actor from skinning and every geometry pass without releasing
    // its persistent template. Making it visible again resumes from the next
    // submitted pose, so callers can cheaply cull a town-sized actor set.
    void setSkinnedActorVisible(std::uint32_t instanceIndex, bool visible);
    void setSkinnedActorPose(std::uint32_t instanceIndex, const ImportedSkinnedActorFrameData& pose);
    void setSkinningDebugBypass(bool bypass);
    // Temporal AA (camera reprojection; static world). Off by default.
    void setTaaEnabled(bool enabled);
    // Enable or disable the retained Bethesda ray-traced scene variants.
    // Capability probing still decides whether the device can actually use them.
    void setRayTracingEnabled(bool enabled);
    // Histogram-driven eye adaptation, off by default. With it off the scene
    // renders at a fixed exposure, so content whose light levels differ from
    // that baseline comes out uniformly too dark or too bright. Worth enabling
    // for any scene with a wide dynamic range or a day/night cycle.
    void setAutoExposureEnabled(bool enabled);
    // Sets every colour-grading term to its neutral value. See the backend's
    // definition for why this is a reset rather than a bypass.
    void setNeutralColorGrading();
    // Set the whole post grade at once. setNeutralColorGrading() is exactly
    // setColorGrading(ColorGradingSettings{}).
    void setColorGrading(const ColorGradingSettings& settings);
    [[nodiscard]] bool isAutoExposureEnabled() const;
    // Replaces the shaded frame with a single visualization of what the main
    // pass shaded with -- see DebugView. Off by default and free when off: the
    // mode rides in an already-spare camera-uniform channel and every consumer
    // is behind a "!= Off" branch, so an unset view compiles to the same work
    // the shader always did. Call any time after init().
    void setDebugView(DebugView view);
    [[nodiscard]] DebugView debugView() const;
    // App-level opt-out of the sun shaft pass (a 20-tap radial march per pixel at AO
    // resolution). On by default; when off the shaft texture reads as black and the
    // main pass is otherwise unchanged. Call any time after init().
    void setSunShaftsEnabled(bool enabled);
    [[nodiscard]] bool isSunShaftsEnabled() const;
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
    // The same readback as tightly packed RGB, for streaming a sequence into an
    // encoder rather than to disk (see render/video_writer.h). Unlike the file
    // form, this is meant to be called every frame: the readback resources are
    // built once and reused.
    bool captureFrameRgb(std::vector<std::uint8_t>& outRgb,
                         std::uint32_t& outWidth,
                         std::uint32_t& outHeight);
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
    void renderFrame(const CameraPose& camera);
    // Upscaling. Set before init() to take effect on the first swapchain build:
    // the quality preset chooses the internal render resolution, which sizes
    // every render target. upscalerStatus() reports what actually runs, which is
    // not always what was asked for -- see UpscalerStatus.
    void setUpscalerSettings(const UpscalerSettings& settings);
    [[nodiscard]] UpscalerStatus upscalerStatus() const;
    void setDebugUiVisible(bool visible);
    bool isDebugUiVisible() const;
    // Stats keeps the readouts and drops every tuning control; Full is the
    // whole console. Visibility and mode are independent -- F4 toggles the
    // former without disturbing the latter.
    void setDebugUiMode(DebugUiMode mode);
    [[nodiscard]] DebugUiMode debugUiMode() const;
    // Game-supplied readouts, appended to the stats window in the order given.
    // Rebuilt per frame by the caller; cheap to skip entirely by checking
    // isDebugUiVisible() first, which is what avoids formatting strings nobody
    // is going to look at.
    void setDebugStatGroups(std::vector<DebugStatGroup> groups);
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
    // Optional game/simulation clock for shader animation. Negative restores
    // the backend wall clock. Fixed-step captures publish their own time so
    // water, rigid machinery, and future foliage advance at the encoded rate.
    void setVisualTimeSeconds(float seconds);
    // Authored sky from a Fallout WTHR record. Default-constructed params
    // (weight 0) restore the procedural sky.
    void setWeatherSky(const WeatherSkyParams& params);
    // Cloud layer textures for the active weather. Upload-heavy, so call it
    // when the weather changes rather than per frame.
    void setWeatherClouds(const WeatherCloudTextures& clouds);
    // Tone curve for the post pass. Default is ACES, so this is inert
    // unless a game selects otherwise.
    void setTonemapSettings(const TonemapSettings& settings);
    // Read back what is in force, so a caller can change ONE field without
    // silently resetting the rest to struct defaults -- which is exactly what
    // `setTonemapSettings(TonemapSettings{})` does to an ENB configuration.
    [[nodiscard]] TonemapSettings tonemapSettings() const;
    // Supplies metallic/roughness only when an imported surface has no authored
    // PBR material. Intended for renderer presets over legacy TES3/TES4 art.
    void setImportedPbrDefaults(const ImportedPbrDefaults& defaults);
    // Post-process depth of field. Focus is a view distance in world units;
    // blur ramps to maxRadiusPixels over focusRange BEHIND the focal plane and,
    // scaled by nearBlurScale, over focusRange/nearBlurScale IN FRONT of it.
    //
    // nearBlurScale is the near-field ramp rate, and it is the knob that picks
    // the look: 0 is far-only (a blurred backdrop with a sharp foreground),
    // ~1.25 blurs both ends hard for a tilt-shift/diorama miniature, and a
    // value BELOW 1 stretches the near ramp out -- which is what a portrait
    // framing wants, so a subject standing a little in front of the focal
    // plane does not go soft along with the ground it is standing on.
    //
    // It DEFAULTS TO 0 so that adding the near field changed no existing
    // caller's look: before this parameter the shader blurred only the far
    // side, and a caller that does not ask for a near field still gets exactly
    // that.
    void setDepthOfField(bool enabled, float focusDistance, float focusRange,
                         float maxRadiusPixels, float nearBlurScale = 0.0f);
    void setImportedSceneDebugState(bool showTerrain, bool showStatics, bool showTextures, bool flatShading, bool waterDebug);
    void setImportedInteriorLighting(const ImportedInteriorLighting& lighting);
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
