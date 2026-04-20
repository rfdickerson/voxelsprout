#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

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
