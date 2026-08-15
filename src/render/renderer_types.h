#pragma once

#include "import/imported_scene.h"
#include "math/math.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace odai::render {

enum class FramePacingMode : std::uint8_t {
    Off = 0,
    Passive = 1,
    Scheduled = 2,
};

struct FramePacingSettings {
    FramePacingMode mode = FramePacingMode::Passive;
    std::uint32_t cadenceDivisor = 1;
    std::uint32_t maxQueuedFrames = 2;
};

struct FramePacingStats {
    bool displayTimingSupported = false;
    bool displayTimingEnabled = false;
    bool schedulingActive = false;
    std::uint32_t cadenceDivisor = 1;
    std::uint32_t maxQueuedFrames = 2;
    std::uint32_t queuedFrames = 0;
    std::uint32_t latePresentCount = 0;
    std::uint32_t gpuTimestampSkippedFrames = 0;
    float refreshMs = 0.0f;
    float targetPresentIntervalMs = 0.0f;
    float desiredLeadTimeMs = 0.0f;
    float presentMarginMs = 0.0f;
    float actualPresentDeltaMs = 0.0f;
    float presentScheduleErrorMs = 0.0f;
    float cpuWaitFrameSlotMs = 0.0f;
    float cpuWaitAcquireMs = 0.0f;
    float cpuWaitPresentMs = 0.0f;
    float cpuWaitTransferMs = 0.0f;
    bool gpuTimestampsPending = false;
    std::uint64_t desiredPresentTimeNs = 0;
};

struct UiRenderStats {
    std::uint32_t textureSlots = 0;
    std::uint32_t commandCount = 0;
    std::uint32_t drawCallCount = 0;
    std::uint64_t dynamicUploadBytes = 0;
    std::uint64_t skippedDrawCalls = 0;
};

enum class ShadowMode : std::uint8_t {
    ShadowMaps = 0,
    RayTraced = 1,
    Auto = 2,
};

// Temporal upscaling backend.
//
// The renderer renders the 3D scene at a fraction of the swapchain extent and an
// upscaler reconstructs it. Which one is a runtime choice, but availability is a
// COMPILE-time fact: XeSS is Intel's closed-source SDK and DLSS is NVIDIA's, so
// neither can be implemented here -- they are linked, and only when the SDK is
// present. Requesting an unavailable backend is not an error; it reports
// unavailable and falls back, so the same command line works on every machine.
enum class UpscalerBackend : std::uint8_t {
    // Render at native resolution. No reconstruction, no jitter requirement.
    Off = 0,
    // This engine's own temporal upscaler, built on the existing jittered TAA
    // and the skinned motion vectors. Always available -- it is the default and
    // the fallback for every other backend.
    Temporal = 1,
    // Intel XeSS-SR. Requires ODAI_ENABLE_XESS and the SDK at build time.
    Xess = 2,
    // AMD FidelityFX Super Resolution. DECLARED, NOT IMPLEMENTED.
    Fsr = 3,
    // NVIDIA DLSS. DECLARED, NOT IMPLEMENTED.
    Dlss = 4,
};

// Quality preset, which is really a render-scale choice. The ratios are XeSS's
// published ones so a preset means the same internal resolution whichever
// backend renders it -- otherwise "Quality" would be a different measurement on
// each and no comparison between them would mean anything.
enum class UpscalerQuality : std::uint8_t {
    UltraQuality = 0,   // 1.3x
    Quality = 1,        // 1.5x
    Balanced = 2,       // 1.7x
    Performance = 3,    // 2.0x
    UltraPerformance = 4,  // 3.0x
    // Reconstruct at the SAME resolution: no upscale, just the temporal
    // resolve. This exists because TAA and upscaling are separate things that
    // this API had welded together -- recordTaaPass returns early without an
    // upscaler object, so setTaaEnabled(true) with the default backend of Off
    // was a silent no-op, and the only way to get TAA was to accept a 1/1.5
    // render scale along with it.
    Native = 5,  // 1.0x
};

inline constexpr float upscalerQualityScale(UpscalerQuality quality) {
    switch (quality) {
    case UpscalerQuality::UltraQuality: return 1.0f / 1.3f;
    case UpscalerQuality::Quality: return 1.0f / 1.5f;
    case UpscalerQuality::Balanced: return 1.0f / 1.7f;
    case UpscalerQuality::Performance: return 1.0f / 2.0f;
    case UpscalerQuality::UltraPerformance: return 1.0f / 3.0f;
    case UpscalerQuality::Native: return 1.0f;
    }
    return 1.0f;
}

struct UpscalerSettings {
    UpscalerBackend backend = UpscalerBackend::Off;
    UpscalerQuality quality = UpscalerQuality::Quality;
    // 0 = none. Applied by backends that expose it; ignored by those that do not.
    float sharpness = 0.0f;
};

// What actually happened, as opposed to what was asked for. `reason` is only
// meaningful when requested != active, and exists because "I asked for XeSS and
// got something else" has several distinct causes that look identical on screen.
struct UpscalerStatus {
    UpscalerBackend requested = UpscalerBackend::Off;
    UpscalerBackend active = UpscalerBackend::Off;
    bool compiledIn = false;      // the backend was built into this binary
    bool runtimeAvailable = false;  // and its runtime/device support is present
    float renderScale = 1.0f;
    const char* reason = "";
};

// Screen-space ambient-occlusion estimator. Each maps to its own compute pipeline
// built from ssao.comp.slang with a different ODAI_AO_MODE, so the sample loop
// carries no uniform branch. Off dispatches neither the AO nor the blur pass and
// leaves the world shaders' ambient factor at 1.
//
// Ssao is the original normal-oriented hemisphere point sampler and the cheapest.
// Hbao (Bavoil 2008) marches horizons and catches contact darkening a point
// sampler misses. Gtao (Jimenez 2016) does the same horizon search through the
// correct cosine-weighted visibility integral, so it converges on ray-traced AO;
// it is the default and the right pick unless you are budget-bound.
enum class AoMode : std::uint8_t {
    Off = 0,
    Ssao = 1,
    Hbao = 2,
    Gtao = 3,
    // Intel's XeGTAO: the same ground-truth integral as Gtao, but marched
    // against a prefiltered depth pyramid with blue noise and adaptive sample
    // counts, and producing bent normals alongside the AO term.
    Xegtao = 4,
};

// Single-channel visualizations of what the main pass actually shaded with,
// for answering "is this surface wrong because of the geometry, the texture,
// or the material flags" without guessing. One global mode rather than a set
// of booleans: they are mutually exclusive by construction (each one replaces
// the frame's colour entirely), and one uniform channel is cheaper to plumb
// than one per view.
//
// MaterialFlags is the one that earns its keep. Alpha handling in this engine
// is decided entirely by flags the importer sets -- the shader cannot tell an
// opacity mask from a specular mask by looking at a texture -- so a surface
// that renders as a black slab where it should be transparent is either
// "the flag never got set" or "the flag is set and the alpha is wrong", and
// nothing on screen distinguishes those two until you false-colour the flags.
//
// Alpha deliberately bypasses the alpha-test discard: discarding first would
// throw away exactly the texels this view exists to look at.
//
// These are read by imported_static.frag.slang, which for a Fallout scene
// covers the terrain and every static and actor -- but not the sky or water,
// which have their own shaders and pass through unchanged.
enum class DebugView : std::uint8_t {
    Off = 0,
    Albedo = 1,        // sampled base colour, unlit
    Normal = 2,        // shading normal, remapped to 0..1
    Alpha = 3,         // sampled alpha as greyscale, discard bypassed
    MaterialFlags = 4, // red=alphaTest green=alphaBlend blue=twoSided yellow=unlit
    Roughness = 5,
    Metallic = 6,
    MipLevel = 7,      // false-coloured texture LOD
    CascadeIndex = 8,  // false-coloured shadow cascade selection
    TextureId = 9,     // bindless slot hashed to a colour
    LinearDepth = 10,  // post-pass; every other view is main-pass
    // The two lighting-balance views. Both are computed mid-shading rather than
    // in debugViewColor(), because their inputs do not exist until the sun and
    // ambient terms have been evaluated.
    Shadow = 11,       // cascaded shadow visibility: white lit, black occluded
    DirectRatio = 12,  // direct / (direct + ambient): how much of the lighting
                       // a shadow is even able to remove. Black means a surface
                       // is lit entirely by unshadowed ambient, which is what
                       // "the shadows do nothing" looks like as a measurement.
    // Terrain layer blend, false-coloured: red/green/blue are ATXT layers 0/1/2
    // and grey is the quadrant's BTXT base showing through. Geometry carrying no
    // terrain layers at all is left near-black.
    //
    // It shows the weights AS AUTHORED -- before the noise perturbation and
    // smoothstep the lit shader applies -- which is the whole point: a wedge-
    // shaped path boundary is either in the DATA or in that stylization, and
    // those have completely different fixes. Comparing this against the lit
    // frame is what separates them.
    TerrainLayers = 13,
    // Ambient occlusion alone, as a clay render: white unoccluded, dark
    // occluded, no albedo and no lighting. The view to reach for when AO is
    // suspected, because every other way of looking at it measures AO through
    // the albedo-weighted ambient term and so returns a picture of the diffuse
    // texture rather than of the occlusion.
    AmbientOcclusion = 14,
};

// True for views whose value is already a display-ready colour and must not be
// run through exposure, tonemapping or grading on the way to the screen.
// LinearDepth is produced by the post pass itself and so is not in this set.
inline constexpr bool debugViewBypassesTonemap(DebugView view) {
    return view != DebugView::Off && view != DebugView::LinearDepth;
}

enum class VoxelGiSurfaceMode : std::uint8_t {
    Legacy = 0,
    RtSurface = 1,
    RestirSurface = 2,
};

enum class ShadowFallbackReason : std::uint8_t {
    None = 0,
    RayTracingUnsupported = 1,
    RayTracingDisabled = 2,
    MainPassNotImplemented = 3,
    RayTracingSceneUnavailable = 4,
};

struct ShadowSettings {
    ShadowMode mode = ShadowMode::Auto;
};

struct ShadowStats {
    ShadowMode requestedMode = ShadowMode::ShadowMaps;
    ShadowMode activeMode = ShadowMode::ShadowMaps;
    bool rayTracingSupported = false;
    bool rayQuerySupported = false;
    bool accelerationStructureSupported = false;
    bool rayTracingRuntimeEnabled = false;
    bool mainPassRayTracingReady = false;
    bool mainPassRayTracingActive = false;
    bool fallbackActive = false;
    ShadowFallbackReason fallbackReason = ShadowFallbackReason::None;
};

// An authored sky, as a Fallout WTHR record describes one. Colors are LINEAR
// rgb in 0..1 -- WTHR stores them as sRGB bytes, so the caller decodes before
// getting here, the same rule the rest of the renderer follows.
//
// `weight` blends over the procedural Rayleigh/Mie sky rather than replacing
// it: 0 is the default and renders exactly as before, which is what keeps every
// other game unaffected by this existing.

// Fallout authors four cloud layers per weather. The ones a weather does not
// use point at "sky\alpha.dds", a fully transparent 1520-byte placeholder, so
// "four layers" in the record usually means one or two on screen.
inline constexpr int kWeatherCloudLayerCount = 4;

struct WeatherSkyParams {
    float weight = 0.0f;
    float skyUpper[3] = {0.0f, 0.0f, 0.0f};   // zenith
    float skyLower[3] = {0.0f, 0.0f, 0.0f};   // the band just above the horizon
    float horizon[3] = {0.0f, 0.0f, 0.0f};    // the skyline itself
    float fogColor[3] = {0.0f, 0.0f, 0.0f};
    float fogFarDistance = 0.0f;              // world units; 0 leaves fog alone

    // Per-layer tint (linear, from PNAM) and coverage, updated per frame
    // because both track time of day. A layer with opacity 0 costs nothing:
    // the shader skips it.
    float cloudTint[kWeatherCloudLayerCount][3] = {};
    float cloudOpacity[kWeatherCloudLayerCount] = {};
};

// The textures behind those layers. Separate from WeatherSkyParams because
// uploading is expensive and only happens when the weather changes, while the
// tints above move every frame.
struct WeatherCloudTextures {
    // A layer with an empty texture (no pixels) is disabled. Textures are
    // decoded DDS exactly as they come out of the mod's archive.
    odai::importer::ImportedSceneTexture layers[kWeatherCloudLayerCount];
    // Texture units per second, from the WTHR DATA block's cloud speeds.
    float scrollSpeed[kWeatherCloudLayerCount] = {};
    // Dome scale. These textures are fisheye maps of the whole sky, so this is
    // NOT a tiling count: 1.0 puts the horizon on the texture's inscribed
    // circle, and values above 1 push the rim past the horizon (and out of the
    // texture). Never a reason to tile one of these.
    float domeScale[kWeatherCloudLayerCount] = {};
};

// Which tone curve the post pass runs, and its parameters.
//
// The ENB values are Enhanced Shaders' own defaults for Fallout: New Vegas,
// read out of its enbeffect.fx.ini. They were tuned against the same weather
// records this engine now reads, which is why they are a better starting point
// for that game than numbers picked here would be.
enum class TonemapMode : std::uint32_t {
    Aces = 0,  // the fixed rational fit; what every game rendered with before
    Enb = 1,   // extended Reinhard, contrast on magnitude, saturation on hue
};

struct TonemapSettings {
    TonemapMode mode = TonemapMode::Aces;
    float contrast = 1.35f;             // ENB "Contrast Day"
    float saturation = 1.25f;           // ENB "Saturation Day"
    float curve = 8.0f;                 // ENB "ToneMapping Curve Day"
    float overbrightDampening = 75.0f;  // ENB "Overbright Dampening Day"
    // ACES-path highlight shaping. The scene-linear value (post-exposure) that
    // should reach display white, and how much of that normalization to apply.
    //
    // 0 is the DEFAULT and is exactly the plain fit, so no existing game's look
    // changes. Above 0 the curve is divided by its own value at this point --
    // the standard curve(x)/curve(W) construction -- which is the only knob in
    // this chain that moves the top of the histogram. Without it, auto-exposure
    // holding the scene near middle grey leaves the frame's 99th percentile
    // around 0.65 and the brightest third of the display range unused.
    float whitePoint = 0.0f;
    float highlightShoulder = 0.7f;
};

// The post colour grade, as one value instead of fourteen scattered fields on
// the debug-settings blob.
//
// The chain in tone_map.frag.slang runs UNCONDITIONALLY -- there is no enable
// bit -- so "no grading" is this struct's defaults, every term at identity, and
// that is what Renderer::setNeutralColorGrading() writes. Anything else is a
// look, and a look is worth naming and measuring rather than leaving as whatever
// the debug panel happened to be initialised with.
struct ColorGradingSettings {
    float whiteBalance[3] = {1.0f, 1.0f, 1.0f};
    float contrast = 1.0f;         // clamped to [0.70, 1.40] in the shader
    float midtoneContrast = 1.0f;  // clamped to [0.80, 1.40]
    float saturation = 1.0f;
    // Vibrance pushes the LEAST saturated pixels hardest, so it compounds with
    // saturation in a way that is easy to over-apply: a dusty or foggy scene is
    // mostly low-saturation pixels, which is exactly what this term targets.
    float vibrance = 0.0f;
    // Below 1 DARKENS shadows. Worth being careful with on any scene carrying
    // real aerial perspective -- the fog IS the depth cue, and crushing it to
    // black removes the thing that made the distance readable.
    float shadowDensity = 1.0f;
    float shadowTint[3] = {0.0f, 0.0f, 0.0f};
    float highlightTint[3] = {0.0f, 0.0f, 0.0f};
};

struct CameraPose {
    float x;
    float y;
    float z;
    float yawDegrees;
    float pitchDegrees;
    float fovDegrees;
    bool orthographic = false;
    float orthoHalfHeight = 1000.0f;
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
    std::uint32_t faceId = 0;
    bool pipeStyle = false;
    float pipeAxisX = 0.0f;
    float pipeAxisY = 1.0f;
    float pipeAxisZ = 0.0f;
    float pipeRadius = 0.45f;
    float pipeStyleId = 0.0f;
};

struct ImportedActorFrameData {
    std::span<const odai::importer::ImportedScenePackedVertex> vertices;
    std::span<const std::uint32_t> indices;
    std::span<const odai::importer::ImportedScenePackedDraw> draws;
};

// A rest-pose (bind-pose) vertex for a GPU-skinned mesh (see docs/ROADMAP.md's
// Party RPG / Narrative section). position/normal feed the skinning compute
// pass; color/uv/textureIndex/flags pass through unchanged into the skinned
// output, matching ImportedMeshVertex's layout exactly so the existing
// imported_static.vert/frag.slang consume it with no changes. Up to 4 bone
// influences per vertex.
struct ImportedSkinnedMeshVertex {
    float position[3] = {};
    float normal[3] = {};
    float color[3] = {};
    float uv[2] = {};
    std::uint32_t textureIndex = 0xffffffffu;
    std::uint32_t flags = 0u;
    std::uint16_t boneIndices[4] = {};
    float boneWeights[4] = {};
};

// A one-time, device-local GPU-skinned mesh template: rest-pose geometry for
// a skinned actor, uploaded once via Renderer::uploadSkinnedMeshTemplate. Posed
// per-frame through ImportedSkinnedActorFrameData without re-uploading
// geometry every frame.
struct ImportedSkinnedMeshTemplate {
    std::span<const ImportedSkinnedMeshVertex> vertices;
    std::span<const std::uint32_t> indices;
    std::span<const odai::importer::ImportedScenePackedDraw> draws;
    std::uint32_t boneCount = 0;
};

// One frame's pose for a previously uploaded ImportedSkinnedMeshTemplate.
// boneMatrices.size() must equal that instance slot's bound template's
// boneCount.
struct ImportedSkinnedActorFrameData {
    std::span<const odai::math::Matrix4> boneMatrices;
};

// Skinning supports a fixed number of independent instance slots. Each has its
// own rest-pose template, pose, and draws via
// Renderer::uploadSkinnedMeshTemplate(instanceIndex, ...) /
// Renderer::setSkinnedActorPose(instanceIndex, ...).
//
// This is still NOT a mass-battle crowd system (see docs/ROADMAP.md's
// out-of-scope note on thousands of units): every slot is fully independent,
// so an ACTIVE one costs its own device-local rest-pose/index/output buffers,
// its own descriptor-buffer set, and one compute dispatch plus one FrameArena
// pose upload per frame. The ceiling is what a populated Fallout settlement
// needs -- Goodsprings alone places 37 actors -- not what a battle does.
//
// An UNUSED slot costs only an empty struct, so the array being larger than any
// one game needs is close to free; raising it from the original 8 was measured
// at no startup or frame cost with nothing extra uploaded.
inline constexpr std::uint32_t kMaxSkinnedInstances = 48;

enum class InventoryItemId : std::uint8_t {
    Empty = 0,
    Stone = 1,
    Dirt = 2,
    Grass = 3,
    Wood = 4,
    Red = 5,
};

static constexpr std::size_t kGameplayHotbarSlotCount = 9;
static constexpr std::size_t kCreativeInventoryItemCount = 5;

struct GameplayUiRect {
    float minX = 0.0f;
    float minY = 0.0f;
    float maxX = 0.0f;
    float maxY = 0.0f;

    [[nodiscard]] bool contains(float x, float y) const {
        return x >= minX && x <= maxX && y >= minY && y <= maxY;
    }
};

struct GameplayUiLayout {
    GameplayUiRect hotbarPanel{};
    std::array<GameplayUiRect, kGameplayHotbarSlotCount> hotbarSlots{};
    GameplayUiRect inventoryPanel{};
    std::array<GameplayUiRect, kCreativeInventoryItemCount> inventorySlots{};
};

// How much of the renderer's ImGui surface is built.
//
// The full panel is a tuning console -- FOV, shadow bias, AO mode, the whole
// sky and exposure chain -- which is the right thing when you are dialling a
// look in and the wrong thing when you are watching numbers during play. Stats
// keeps the readouts (frame timings, GPU stages, memory, draw calls, whatever
// the game pushed through setDebugStatGroups) and drops every control.
enum class DebugUiMode : std::uint8_t {
    Stats = 0,
    Full,
};

// One labelled readout in the debug stats window.
//
// Deliberately just two strings. The alternative -- the renderer knowing what a
// "cell streamer" is so it can format one -- would drag importer types across
// the src/render boundary for the sake of a debug panel. The game formats its
// own numbers and the renderer only prints them.
struct DebugStatRow {
    std::string label;
    std::string value;
};

struct DebugStatGroup {
    std::string title;
    std::vector<DebugStatRow> rows;
};

struct GameplayUiState {
    bool inventoryVisible = false;
    std::uint32_t selectedHotbarSlot = 0;
    std::array<InventoryItemId, kGameplayHotbarSlotCount> hotbarItems{};
    std::array<InventoryItemId, kCreativeInventoryItemCount> creativeInventoryItems = {
        InventoryItemId::Stone,
        InventoryItemId::Dirt,
        InventoryItemId::Grass,
        InventoryItemId::Wood,
        InventoryItemId::Red,
    };
};

inline GameplayUiLayout buildGameplayUiLayout(float displayWidth, float displayHeight) {
    GameplayUiLayout layout{};
    const float hotbarSlotSize = 52.0f;
    const float hotbarGap = 8.0f;
    const float hotbarWidth =
        (hotbarSlotSize * static_cast<float>(kGameplayHotbarSlotCount)) +
        (hotbarGap * static_cast<float>(kGameplayHotbarSlotCount - 1));
    const float hotbarMinX = (displayWidth - hotbarWidth) * 0.5f;
    const float hotbarMinY = displayHeight - 84.0f;
    layout.hotbarPanel = {
        hotbarMinX - 14.0f,
        hotbarMinY - 14.0f,
        hotbarMinX + hotbarWidth + 14.0f,
        hotbarMinY + hotbarSlotSize + 14.0f
    };
    for (std::size_t slotIndex = 0; slotIndex < kGameplayHotbarSlotCount; ++slotIndex) {
        const float slotMinX = hotbarMinX + (static_cast<float>(slotIndex) * (hotbarSlotSize + hotbarGap));
        layout.hotbarSlots[slotIndex] = {
            slotMinX,
            hotbarMinY,
            slotMinX + hotbarSlotSize,
            hotbarMinY + hotbarSlotSize
        };
    }

    const float inventorySlotSize = 76.0f;
    const float inventoryGap = 18.0f;
    const float inventoryWidth =
        (inventorySlotSize * static_cast<float>(kCreativeInventoryItemCount)) +
        (inventoryGap * static_cast<float>(kCreativeInventoryItemCount - 1));
    const float inventoryPanelMinX = (displayWidth - (inventoryWidth + 56.0f)) * 0.5f;
    const float inventoryPanelMinY = (displayHeight - 214.0f) * 0.5f;
    layout.inventoryPanel = {
        inventoryPanelMinX,
        inventoryPanelMinY,
        inventoryPanelMinX + inventoryWidth + 56.0f,
        inventoryPanelMinY + 214.0f
    };
    const float inventorySlotMinX = layout.inventoryPanel.minX + 28.0f;
    const float inventorySlotMinY = layout.inventoryPanel.minY + 84.0f;
    for (std::size_t itemIndex = 0; itemIndex < kCreativeInventoryItemCount; ++itemIndex) {
        const float slotMinX = inventorySlotMinX + (static_cast<float>(itemIndex) * (inventorySlotSize + inventoryGap));
        layout.inventorySlots[itemIndex] = {
            slotMinX,
            inventorySlotMinY,
            slotMinX + inventorySlotSize,
            inventorySlotMinY + inventorySlotSize
        };
    }
    return layout;
}

} // namespace odai::render
