#pragma once

#include "import/imported_scene.h"

#include <algorithm>
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
    std::uint32_t maxQueuedFrames = 3;
};

struct FramePacingStats {
    bool displayTimingSupported = false;
    bool displayTimingEnabled = false;
    bool schedulingActive = false;
    std::uint32_t cadenceDivisor = 1;
    std::uint32_t maxQueuedFrames = 3;
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

enum class ShadowMode : std::uint8_t {
    ShadowMaps = 0,
    RayTraced = 1,
    Auto = 2,
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
    std::uint32_t faceId = 0;
    bool pipeStyle = false;
    float pipeAxisX = 0.0f;
    float pipeAxisY = 1.0f;
    float pipeAxisZ = 0.0f;
    float pipeRadius = 0.45f;
    float pipeStyleId = 0.0f;
};

struct ImportedActorGpuAnimationFrameData;

struct ImportedActorGpuNodeKeyframe {
    // Row-major 3x4 affine matrices. The vertex shader interpolates these rows
    // and applies the result per rigid animated NIF draw.
    float previousRows[12] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f
    };
    float nextRows[12] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f
    };
    float blend = 0.0f;
    float normalBlend = 0.0f;
    float _pad0 = 0.0f;
    float _pad1 = 0.0f;
};

struct ImportedActorGpuAnimatedDraw {
    std::uint32_t firstIndex = 0;
    std::uint32_t indexCount = 0;
    std::uint32_t nodeKeyframeIndex = 0;
    std::uint32_t _pad0 = 0;
};

struct ImportedActorBonePaletteMatrix {
    // Row-major 3x4 affine matrix in engine actor-local space.
    float rows[12] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f
    };
};

struct ImportedActorGpuAnimationFrameData {
    std::span<const odai::importer::ImportedScenePackedVertex> localVertices;
    std::span<const std::uint32_t> indices;
    std::span<const ImportedActorGpuAnimatedDraw> draws;
    std::span<const ImportedActorGpuNodeKeyframe> nodeKeyframes;
};

struct ImportedActorRenderAssetData {
    std::span<const odai::importer::ImportedScenePackedVertex> vertices;
    std::span<const std::uint32_t> indices;
    std::span<const odai::importer::ImportedScenePackedDraw> draws;
    std::span<const std::array<std::uint16_t, 4>> boneIndices;
    std::span<const std::array<float, 4>> boneWeights;
};

struct ImportedActorDebugLineVertex {
    float position[3] = {};
    float color[3] = {1.0f, 1.0f, 1.0f};
};

struct ImportedActorInstanceData {
    float position[3] = {};
    float yawRadians = 0.0f;
    float tint[3] = {0.78f, 0.72f, 0.62f};
    float animationTime = 0.0f;
    std::uint32_t flags = 0u;
    std::uint32_t assetIndex = 0u;
    std::uint32_t firstDraw = 0u;
    std::uint32_t drawCount = 0u;
    std::uint32_t bonePaletteOffset = 0u;
    std::uint32_t clipIndex = 0u;
};

struct ImportedActorFrameData {
    std::span<const ImportedActorInstanceData> instances;
    std::span<const ImportedActorBonePaletteMatrix> bonePalette;
    std::span<const ImportedActorDebugLineVertex> debugBoneLines;
    const ImportedActorGpuAnimationFrameData* gpuAnimation = nullptr;
};

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
    GameplayUiRect inventoryPanel{};
};

struct GameplayUiState {
    int health = 100;
    int maxHealth = 100;
    int magicka = 60;
    int maxMagicka = 60;
    int fatigue = 100;
    int maxFatigue = 100;
    int gold = 0;
    bool playerDead = false;
    bool inventoryVisible = false;
    bool dialogueVisible = false;
    std::string dialogueActorName;
    std::string dialogueText;
    std::string dialogueSelectedTopicId;
    std::string dialogueLastMessage;
    std::string dialogueJournalSummary;
    std::vector<std::pair<std::string, std::string>> dialogueTopics;
    std::vector<std::pair<std::string, std::string>> dialogueChoices;
    std::vector<std::pair<std::string, std::string>> inventoryEntries;
    std::vector<std::pair<std::string, std::string>> questEntries;
    std::string trackedQuestText;
    std::string dialogueFont;
    float dialogueFontSize = 18.0f;
    bool dialogueFontSizeConfigured = false;
    std::string resolvedDialogueFontPath;
};

enum class GameplayUiCommandType : std::uint8_t {
    None = 0,
    CloseDialogue = 1,
    SelectDialogueTopic = 2,
    SelectDialogueChoice = 3,
    SetDialogueFont = 4,
    ClearDialogueFont = 5,
};

struct GameplayUiCommand {
    GameplayUiCommandType type = GameplayUiCommandType::None;
    std::string id;
    float value = 0.0f;
};

inline GameplayUiLayout buildGameplayUiLayout(float displayWidth, float displayHeight) {
    GameplayUiLayout layout{};
    const float inventoryPanelWidth = std::clamp(displayWidth * 0.72f, 620.0f, 1040.0f);
    const float inventoryPanelHeight = std::clamp(displayHeight * 0.62f, 420.0f, 680.0f);
    const float inventoryPanelMinX = (displayWidth - inventoryPanelWidth) * 0.5f;
    const float inventoryPanelMinY = (displayHeight - inventoryPanelHeight) * 0.5f;
    layout.inventoryPanel = {
        inventoryPanelMinX,
        inventoryPanelMinY,
        inventoryPanelMinX + inventoryPanelWidth,
        inventoryPanelMinY + inventoryPanelHeight
    };
    return layout;
}

} // namespace odai::render
