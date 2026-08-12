#pragma once

#include "import/imported_scene.h"
#include "math/math.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

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
};

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
