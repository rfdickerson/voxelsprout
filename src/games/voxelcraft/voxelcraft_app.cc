#include "games/voxelcraft/voxelcraft_app.h"

#include "core/grid3.h"
#include "core/log.h"
#include "math/math.h"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <algorithm>
#include <array>
#include <cmath>

namespace odai::games::voxelcraft {

namespace {

constexpr double kSimulationFixedHz = 60.0;
constexpr double kSimulationFixedStepSeconds = 1.0 / kSimulationFixedHz;
constexpr int kMaxSimulationStepsPerFrame = 8;
constexpr float kWorldAutosaveDelaySeconds = 30.0f;
constexpr float kRenderAspectFallback = 16.0f / 9.0f;
constexpr float kFrustumCullPadVoxels = 8.0f;
constexpr float kFrustumNearDistance = 0.1f;
constexpr float kFrustumFarDistance = 400.0f;
constexpr float kFootstepStrideVoxels = 1.6f;

float wrapDegreesSigned(float degrees) {
    float wrapped = std::fmod(degrees, 360.0f);
    if (wrapped <= -180.0f) {
        wrapped += 360.0f;
    } else if (wrapped > 180.0f) {
        wrapped -= 360.0f;
    }
    return wrapped;
}

float lerpWrappedDegrees(float fromDegrees, float toDegrees, float alpha) {
    const float delta = wrapDegreesSigned(toDegrees - fromDegrees);
    return wrapDegreesSigned(fromDegrees + (delta * alpha));
}

// Shared by the frustum broad-phase bounds below and the per-frame audio listener update.
odai::math::Vector3 cameraForwardVector(float yawDegrees, float pitchDegrees) {
    const float yawRadians = odai::math::radians(yawDegrees);
    const float pitchRadians = odai::math::radians(pitchDegrees);
    const float cosPitch = std::cos(pitchRadians);
    return odai::math::normalize(odai::math::Vector3{
        std::cos(yawRadians) * cosPitch,
        std::sin(pitchRadians),
        std::sin(yawRadians) * cosPitch
    });
}

// Broad-phase view-frustum AABB (corner min/max, no exact per-plane test -- that's all
// ChunkClipmapIndex::queryChunksIntersecting needs). This is the only surviving
// frustum helper; src/app/app.cc's unwired buildCameraFrustum was deleted.
odai::core::CellAabb computeFrustumBroadPhaseBounds(
    const odai::math::Vector3& eye,
    float yawDegrees,
    float pitchDegrees,
    float fovDegrees,
    float aspectRatio
) {
    odai::core::CellAabb bounds{};
    const float clampedAspect = std::max(aspectRatio, 0.1f);
    const float clampedFovDegrees = std::clamp(fovDegrees, 20.0f, 120.0f);
    const odai::math::Vector3 forward = cameraForwardVector(yawDegrees, pitchDegrees);
    if (odai::math::lengthSquared(forward) <= 0.0001f) {
        return bounds;
    }

    const odai::math::Vector3 worldUp{0.0f, 1.0f, 0.0f};
    odai::math::Vector3 right = odai::math::normalize(odai::math::cross(forward, worldUp));
    if (odai::math::lengthSquared(right) <= 0.0001f) {
        right = odai::math::Vector3{1.0f, 0.0f, 0.0f};
    }
    odai::math::Vector3 up = odai::math::normalize(odai::math::cross(right, forward));
    if (odai::math::lengthSquared(up) <= 0.0001f) {
        up = worldUp;
    }

    const float halfFovY = odai::math::radians(clampedFovDegrees) * 0.5f;
    const float tanHalfY = std::tan(halfFovY);
    const float tanHalfX = tanHalfY * clampedAspect;

    const odai::math::Vector3 nearCenter = eye + (forward * kFrustumNearDistance);
    const odai::math::Vector3 farCenter = eye + (forward * kFrustumFarDistance);
    const float nearHalfHeight = kFrustumNearDistance * tanHalfY;
    const float nearHalfWidth = kFrustumNearDistance * tanHalfX;
    const float farHalfHeight = kFrustumFarDistance * tanHalfY;
    const float farHalfWidth = kFrustumFarDistance * tanHalfX;

    const odai::math::Vector3 nearUp = up * nearHalfHeight;
    const odai::math::Vector3 nearRight = right * nearHalfWidth;
    const odai::math::Vector3 farUp = up * farHalfHeight;
    const odai::math::Vector3 farRight = right * farHalfWidth;

    const std::array<odai::math::Vector3, 8> corners = {
        nearCenter + nearUp - nearRight,
        nearCenter + nearUp + nearRight,
        nearCenter - nearUp - nearRight,
        nearCenter - nearUp + nearRight,
        farCenter + farUp - farRight,
        farCenter + farUp + farRight,
        farCenter - farUp - farRight,
        farCenter - farUp + farRight
    };

    float minX = corners[0].x, minY = corners[0].y, minZ = corners[0].z;
    float maxX = corners[0].x, maxY = corners[0].y, maxZ = corners[0].z;
    for (const odai::math::Vector3& corner : corners) {
        minX = std::min(minX, corner.x);
        minY = std::min(minY, corner.y);
        minZ = std::min(minZ, corner.z);
        maxX = std::max(maxX, corner.x);
        maxY = std::max(maxY, corner.y);
        maxZ = std::max(maxZ, corner.z);
    }

    bounds.valid = true;
    bounds.minInclusive = odai::core::Cell3i{
        static_cast<int>(std::floor(minX - kFrustumCullPadVoxels)),
        static_cast<int>(std::floor(minY - kFrustumCullPadVoxels)),
        static_cast<int>(std::floor(minZ - kFrustumCullPadVoxels))
    };
    bounds.maxExclusive = odai::core::Cell3i{
        static_cast<int>(std::floor(maxX + kFrustumCullPadVoxels)) + 1,
        static_cast<int>(std::floor(maxY + kFrustumCullPadVoxels)) + 1,
        static_cast<int>(std::floor(maxZ + kFrustumCullPadVoxels)) + 1
    };
    return bounds;
}

} // namespace

bool VoxelCraftApp::onInit() {
    if (!loadFonts(
            resolveAssetPath("assets/fonts/Inter-Regular.ttf"),
            resolveAssetPath("assets/fonts/Inter-Bold.ttf"),
            resolveAssetPath("assets/fonts/Inter-Italic.ttf"),
            resolveAssetPath("assets/fonts/Inter-Regular.ttf"))) {
        return false;
    }

    m_gameplayUiState.hotbarItems.fill(odai::render::InventoryItemId::Empty);
    m_gameplayUiState.hotbarItems[0] = odai::render::InventoryItemId::Stone;
    m_gameplayUiState.hotbarItems[1] = odai::render::InventoryItemId::Dirt;
    m_gameplayUiState.hotbarItems[2] = odai::render::InventoryItemId::Grass;
    m_gameplayUiState.hotbarItems[3] = odai::render::InventoryItemId::Wood;
    m_gameplayUiState.hotbarItems[4] = odai::render::InventoryItemId::Red;

    world::World::LoadResult loadResult{};
    m_world.loadOrInitialize(m_worldPath, &loadResult);
    VOX_LOGI("voxelcraft") << (loadResult.loadedFromFile ? "loaded world " : "generating fresh world ")
                           << m_worldPath.string();

    // Demo environmental audio: two positional ambient beds (torch, river) plus one global
    // wind bed, all started once here and left running for the session -- distance
    // attenuation comes entirely from the per-frame listener update in onRender(), not from
    // any zone/proximity system. Missing asset files (assets/audio/ doesn't ship real clips
    // yet) leave every handle invalid, so these calls simply no-op until content is supplied.
    m_ambientTorchSound = m_audio.loadSound(
        resolveAssetPath("assets/audio/ambient/torch_crackle.wav"), odai::audio::SoundCategory::Ambient);
    m_ambientRiverSound = m_audio.loadSound(
        resolveAssetPath("assets/audio/ambient/river.wav"), odai::audio::SoundCategory::Ambient);
    m_ambientWindSound = m_audio.loadSound(
        resolveAssetPath("assets/audio/ambient/wind.wav"), odai::audio::SoundCategory::Ambient);
    m_sfxFootstep = m_audio.loadSound(
        resolveAssetPath("assets/audio/sfx/footstep.wav"), odai::audio::SoundCategory::Ambient);
    m_sfxBlockBreak = m_audio.loadSound(
        resolveAssetPath("assets/audio/sfx/block_break.wav"), odai::audio::SoundCategory::Ambient);

    m_ambientTorchHandle = m_audio.startAmbientAt(
        m_ambientTorchSound, odai::math::Vector3{8.0f, 11.0f, 8.0f},
        odai::audio::AttenuationParams{1.0f, 12.0f, 1.0f});
    m_ambientRiverHandle = m_audio.startAmbientAt(
        m_ambientRiverSound, odai::math::Vector3{30.0f, 9.0f, -10.0f},
        odai::audio::AttenuationParams{2.0f, 40.0f, 1.0f});
    m_ambientWindHandle = m_audio.startAmbient(m_ambientWindSound, /*fadeSeconds=*/2.0f);

    m_streamingPipeline.start();
    refreshStreamingWindow(true);

    m_playerPrevious = m_player;
    setMouseCaptured(true);
    m_mouseCaptured = true;
    return true;
}

void VoxelCraftApp::onShutdown() {
    m_streamingPipeline.stop();
    saveWorld();
}

void VoxelCraftApp::sampleInput() {
    m_pendingInput.moveForward = glfwGetKey(m_window, GLFW_KEY_W) == GLFW_PRESS;
    m_pendingInput.moveBackward = glfwGetKey(m_window, GLFW_KEY_S) == GLFW_PRESS;
    m_pendingInput.moveLeft = glfwGetKey(m_window, GLFW_KEY_A) == GLFW_PRESS;
    m_pendingInput.moveRight = glfwGetKey(m_window, GLFW_KEY_D) == GLFW_PRESS;
    m_pendingInput.jumpDown = glfwGetKey(m_window, GLFW_KEY_SPACE) == GLFW_PRESS;
    m_pendingInput.sneakDown =
        glfwGetKey(m_window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
        glfwGetKey(m_window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS;
    m_pendingInput.sprintDown =
        glfwGetKey(m_window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS ||
        glfwGetKey(m_window, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS;

    m_pendingInput.gamepadMoveForward = 0.0f;
    m_pendingInput.gamepadMoveRight = 0.0f;
    m_pendingInput.gamepadLookX = 0.0f;
    m_pendingInput.gamepadLookY = 0.0f;
    GLFWgamepadstate gamepadState{};
    if (glfwJoystickIsGamepad(GLFW_JOYSTICK_1) == GLFW_TRUE &&
        glfwGetGamepadState(GLFW_JOYSTICK_1, &gamepadState) == GLFW_TRUE) {
        const auto deadzone = [](float value, float dz) {
            const float magnitude = std::abs(value);
            if (magnitude <= dz) {
                return 0.0f;
            }
            return std::copysign((magnitude - dz) / (1.0f - dz), value);
        };
        m_pendingInput.gamepadMoveForward =
            -deadzone(gamepadState.axes[GLFW_GAMEPAD_AXIS_LEFT_Y], kGamepadMoveDeadzone);
        m_pendingInput.gamepadMoveRight =
            deadzone(gamepadState.axes[GLFW_GAMEPAD_AXIS_LEFT_X], kGamepadMoveDeadzone);
        m_pendingInput.gamepadLookX =
            deadzone(gamepadState.axes[GLFW_GAMEPAD_AXIS_RIGHT_X], kGamepadLookDeadzone);
        m_pendingInput.gamepadLookY =
            -deadzone(gamepadState.axes[GLFW_GAMEPAD_AXIS_RIGHT_Y], kGamepadLookDeadzone);
        if (!m_pendingInput.jumpDown) {
            m_pendingInput.jumpDown = gamepadState.buttons[GLFW_GAMEPAD_BUTTON_A] == GLFW_PRESS;
        }
    }

    const bool escapeDown = glfwGetKey(m_window, GLFW_KEY_ESCAPE) == GLFW_PRESS;
    if (escapeDown && !m_wasEscapeDown) {
        if (m_gameplayUiState.inventoryVisible) {
            m_gameplayUiState.inventoryVisible = false;
            setMouseCaptured(true);
            m_mouseCaptured = true;
        } else {
            glfwSetWindowShouldClose(m_window, GLFW_TRUE);
        }
    }
    m_wasEscapeDown = escapeDown;

    const bool saveKeyDown = glfwGetKey(m_window, GLFW_KEY_F5) == GLFW_PRESS;
    if (saveKeyDown && !m_wasSaveKeyDown) {
        saveWorld();
    }
    m_wasSaveKeyDown = saveKeyDown;

    const bool frameStatsKeyDown = glfwGetKey(m_window, GLFW_KEY_F) == GLFW_PRESS;
    static bool wasFrameStatsKeyDown = false;
    if (frameStatsKeyDown && !wasFrameStatsKeyDown) {
        m_renderer.setFrameStatsVisible(!m_renderer.isFrameStatsVisible());
    }
    wasFrameStatsKeyDown = frameStatsKeyDown;
}

void VoxelCraftApp::updateHotbarAndInventoryInput() {
    const bool inventoryKeyDown = glfwGetKey(m_window, GLFW_KEY_E) == GLFW_PRESS;
    if (inventoryKeyDown && !m_wasInventoryKeyDown) {
        m_gameplayUiState.inventoryVisible = !m_gameplayUiState.inventoryVisible;
        setMouseCaptured(!m_gameplayUiState.inventoryVisible);
        m_mouseCaptured = !m_gameplayUiState.inventoryVisible;
    }
    m_wasInventoryKeyDown = inventoryKeyDown;

    static constexpr int kNumberKeys[9] = {
        GLFW_KEY_1, GLFW_KEY_2, GLFW_KEY_3, GLFW_KEY_4, GLFW_KEY_5,
        GLFW_KEY_6, GLFW_KEY_7, GLFW_KEY_8, GLFW_KEY_9
    };
    std::uint32_t keyMask = 0;
    for (int slot = 0; slot < 9; ++slot) {
        const bool down = glfwGetKey(m_window, kNumberKeys[slot]) == GLFW_PRESS;
        if (down) {
            keyMask |= (1u << slot);
        }
        if (down && !((m_hotbarKeyPrevMask >> slot) & 1u)) {
            selectHotbarSlot(m_gameplayUiState, slot);
        }
    }
    m_hotbarKeyPrevMask = keyMask;

    if (!m_gameplayUiState.inventoryVisible && !m_uiContext.wantsMouse() && m_uiInput.scrollDelta != 0.0f) {
        cycleSelectedHotbar(m_gameplayUiState, m_uiInput.scrollDelta > 0.0f ? -1 : 1);
    }

    if (m_gameplayUiState.inventoryVisible &&
        !m_uiContext.wantsMouse() &&
        m_uiInput.button(odai::ui::UiMouseButton::Left).pressed) {
        int fbW = 0, fbH = 0;
        framebufferSize(fbW, fbH);
        const odai::render::GameplayUiLayout layout =
            odai::render::buildGameplayUiLayout(static_cast<float>(fbW), static_cast<float>(fbH));
        const float mouseX = m_uiInput.mousePx.x;
        const float mouseY = m_uiInput.mousePx.y;
        bool handled = false;
        for (std::size_t slotIndex = 0; slotIndex < layout.hotbarSlots.size() && !handled; ++slotIndex) {
            if (layout.hotbarSlots[slotIndex].contains(mouseX, mouseY)) {
                selectHotbarSlot(m_gameplayUiState, static_cast<int>(slotIndex));
                handled = true;
            }
        }
        for (std::size_t itemIndex = 0; itemIndex < layout.inventorySlots.size() && !handled; ++itemIndex) {
            if (layout.inventorySlots[itemIndex].contains(mouseX, mouseY)) {
                assignInventoryItemToSelectedHotbar(m_gameplayUiState, m_gameplayUiState.creativeInventoryItems[itemIndex]);
                handled = true;
            }
        }
    }

    m_renderer.setGameplayUiState(m_gameplayUiState);
}

void VoxelCraftApp::updateBlockInteraction() {
    if (m_gameplayUiState.inventoryVisible || m_uiContext.wantsMouse()) {
        return;
    }

    std::vector<std::size_t> dirtyChunkIndices;
    bool edited = false;

    if (m_uiInput.button(odai::ui::UiMouseButton::Left).pressed) {
        // tryRemoveVoxelFromCameraRay resets m_breakProgress on success and doesn't hand the
        // broken block's position back, so raycast once more here (cheap voxel DDA) purely to
        // get a position for the break SFX.
        const CameraRaycastResult raycast = raycastFromCamera(m_player, m_world);
        const bool removed = tryRemoveVoxelFromCameraRay(m_world, m_player, m_breakProgress, dirtyChunkIndices);
        if (removed && raycast.hitSolid) {
            m_audio.playSoundAt(m_sfxBlockBreak, odai::math::Vector3{
                static_cast<float>(raycast.solidX) + 0.5f,
                static_cast<float>(raycast.solidY) + 0.5f,
                static_cast<float>(raycast.solidZ) + 0.5f});
        }
        edited = removed || edited;
    }
    if (m_uiInput.button(odai::ui::UiMouseButton::Right).pressed) {
        edited = tryPlaceVoxelFromCameraRay(
            m_world, m_player, selectedPlaceVoxel(m_gameplayUiState), dirtyChunkIndices) || edited;
    }

    if (edited && !dirtyChunkIndices.empty()) {
        m_renderer.updateChunkMesh(m_world.chunkGrid(), std::span<const std::size_t>(dirtyChunkIndices));
        m_worldDirty = true;
    }
}

void VoxelCraftApp::tickAutosave(float dt) {
    if (!m_worldDirty) {
        return;
    }
    m_worldAutosaveElapsedSeconds += dt;
    if (m_worldAutosaveElapsedSeconds >= kWorldAutosaveDelaySeconds) {
        saveWorld();
    }
}

void VoxelCraftApp::tickFootstepAudio(float dxHorizontal, float dzHorizontal) {
    if (!m_player.onGround) {
        return;
    }
    m_footstepDistanceAccumulator += std::sqrt((dxHorizontal * dxHorizontal) + (dzHorizontal * dzHorizontal));
    if (m_footstepDistanceAccumulator < kFootstepStrideVoxels) {
        return;
    }
    m_footstepDistanceAccumulator = 0.0f;
    m_audio.playSoundAt(m_sfxFootstep, odai::math::Vector3{m_player.x, m_player.y, m_player.z});
}

void VoxelCraftApp::saveWorld() {
    if (m_world.save(m_worldPath)) {
        m_worldDirty = false;
        m_worldAutosaveElapsedSeconds = 0.0f;
    } else {
        VOX_LOGE("voxelcraft") << "failed to save world to " << m_worldPath.string();
    }
}

void VoxelCraftApp::refreshStreamingWindow(bool forceRendererUpload) {
    const world::World::ChunkStreamingUpdate update =
        m_world.updateStreamingWindowForWorldPosition(m_player.x, m_player.z);
    if (!update.stats.changed && !forceRendererUpload) {
        return;
    }

    const world::ClipmapConfig requestedClipmapConfig = m_renderer.clipmapQueryConfig();
    const bool clipmapConfigChanged =
        !m_hasAppliedClipmapConfig ||
        requestedClipmapConfig.levelCount != m_appliedClipmapConfig.levelCount ||
        requestedClipmapConfig.gridResolution != m_appliedClipmapConfig.gridResolution ||
        requestedClipmapConfig.baseVoxelSize != m_appliedClipmapConfig.baseVoxelSize ||
        requestedClipmapConfig.brickResolution != m_appliedClipmapConfig.brickResolution;
    m_clipmapIndex.setConfig(requestedClipmapConfig);
    m_appliedClipmapConfig = requestedClipmapConfig;
    m_hasAppliedClipmapConfig = true;
    if (clipmapConfigChanged || !m_clipmapIndex.valid()) {
        m_clipmapIndex.rebuild(m_world.chunkGrid());
    } else {
        m_clipmapIndex.syncResidentChunks(m_world.chunkGrid());
    }

    const bool uploadOk = update.requiresFullMeshUpload || forceRendererUpload
        ? m_renderer.updateChunkMesh(m_world.chunkGrid())
        : m_renderer.updateChunkMesh(
            m_world.chunkGrid(), std::span<const std::size_t>(update.residentChunkIndicesNeedingUpload));
    if (!uploadOk) {
        VOX_LOGE("voxelcraft") << "streaming update failed to refresh resident chunk meshes";
        return;
    }

    m_worldDirty = true;
}

void VoxelCraftApp::updateChunkStreaming() {
    for (ChunkStreamingPipeline::CompletedChunk& completed : m_streamingPipeline.drainCompleted()) {
        m_world.insertGeneratedChunk(completed.key, std::move(completed.chunk));
    }

    const world::World::ChunkStreamingConfig config = m_world.streamingConfig();
    std::vector<world::World::ChunkKey> desiredKeys = computeDesiredChunkKeys(config, m_player.x, m_player.z);

    for (const world::World::ChunkKey& key : desiredKeys) {
        m_streamingPipeline.requestChunk(key);
    }
    for (const world::World::ChunkKey& previousKey : m_lastDesiredChunkKeys) {
        const bool stillDesired =
            std::find(desiredKeys.begin(), desiredKeys.end(), previousKey) != desiredKeys.end();
        if (!stillDesired) {
            m_streamingPipeline.cancelChunk(previousKey);
        }
    }
    m_lastDesiredChunkKeys = std::move(desiredKeys);

    refreshStreamingWindow(false);
}

void VoxelCraftApp::onTick(float dt) {
    sampleInput();

    float pendingMouseDeltaX = m_mouseCaptured ? m_uiInput.mouseDeltaPx.x : 0.0f;
    float pendingMouseDeltaY = m_mouseCaptured ? m_uiInput.mouseDeltaPx.y : 0.0f;

    m_simAccumulatorSeconds += static_cast<double>(dt);
    int stepCount = 0;
    while (m_simAccumulatorSeconds >= kSimulationFixedStepSeconds && stepCount < kMaxSimulationStepsPerFrame) {
        m_playerPrevious = m_player;
        m_pendingInput.mouseDeltaX = pendingMouseDeltaX;
        m_pendingInput.mouseDeltaY = pendingMouseDeltaY;
        pendingMouseDeltaX = 0.0f;
        pendingMouseDeltaY = 0.0f;
        updatePlayer(m_player, m_pendingInput, m_world, static_cast<float>(kSimulationFixedStepSeconds));
        tickFootstepAudio(m_player.x - m_playerPrevious.x, m_player.z - m_playerPrevious.z);
        m_simAccumulatorSeconds -= kSimulationFixedStepSeconds;
        ++stepCount;
    }
    if (stepCount == kMaxSimulationStepsPerFrame && m_simAccumulatorSeconds >= kSimulationFixedStepSeconds) {
        m_simAccumulatorSeconds = std::fmod(m_simAccumulatorSeconds, kSimulationFixedStepSeconds);
    }

    updateChunkStreaming();
    updateHotbarAndInventoryInput();
    updateBlockInteraction();
    tickAutosave(dt);
}

render::CameraPose VoxelCraftApp::interpolatedCameraPose(float alpha) const {
    const float clampedAlpha = std::clamp(alpha, 0.0f, 1.0f);
    return render::CameraPose{
        m_playerPrevious.x + ((m_player.x - m_playerPrevious.x) * clampedAlpha),
        m_playerPrevious.y + ((m_player.y - m_playerPrevious.y) * clampedAlpha),
        m_playerPrevious.z + ((m_player.z - m_playerPrevious.z) * clampedAlpha),
        lerpWrappedDegrees(m_playerPrevious.yawDegrees, m_player.yawDegrees, clampedAlpha),
        m_playerPrevious.pitchDegrees + ((m_player.pitchDegrees - m_playerPrevious.pitchDegrees) * clampedAlpha),
        m_player.fovDegrees
    };
}

void VoxelCraftApp::drawHud() {
    int fbW = 0, fbH = 0;
    framebufferSize(fbW, fbH);
    const float cx = static_cast<float>(fbW) * 0.5f;
    const float cy = static_cast<float>(fbH) * 0.5f;
    constexpr float kCrosshairHalfLength = 8.0f;
    constexpr float kCrosshairThickness = 2.0f;
    const odai::ui::UiColor crosshairColor{1.0f, 1.0f, 1.0f, 0.82f};
    m_uiDrawList.addRectFilled(
        odai::ui::UiRect{cx - kCrosshairHalfLength, cy - (kCrosshairThickness * 0.5f),
                         cx + kCrosshairHalfLength, cy + (kCrosshairThickness * 0.5f)},
        crosshairColor);
    m_uiDrawList.addRectFilled(
        odai::ui::UiRect{cx - (kCrosshairThickness * 0.5f), cy - kCrosshairHalfLength,
                         cx + (kCrosshairThickness * 0.5f), cy + kCrosshairHalfLength},
        crosshairColor);
}

void VoxelCraftApp::onRender(float dt) {
    (void)dt;
    beginFrameDraw();
    if (!m_gameplayUiState.inventoryVisible) {
        drawHud();
    }

    const float alpha = static_cast<float>(
        std::clamp(m_simAccumulatorSeconds / kSimulationFixedStepSeconds, 0.0, 1.0));
    const render::CameraPose camera = interpolatedCameraPose(alpha);

    odai::audio::ListenerTransform listener{};
    listener.position = odai::math::Vector3{camera.x, camera.y, camera.z};
    listener.forward = cameraForwardVector(camera.yawDegrees, camera.pitchDegrees);
    m_audio.setListenerTransform(listener);

    int fbW = 0, fbH = 0;
    framebufferSize(fbW, fbH);
    const float aspect = fbH > 0 ? static_cast<float>(fbW) / static_cast<float>(fbH) : kRenderAspectFallback;
    const odai::core::CellAabb frustumBounds = computeFrustumBroadPhaseBounds(
        odai::math::Vector3{camera.x, camera.y, camera.z},
        camera.yawDegrees, camera.pitchDegrees, camera.fovDegrees, aspect);
    m_visibleChunkIndices = m_clipmapIndex.queryChunksIntersecting(frustumBounds);

    WorldFrameContent worldContent{};
    worldContent.chunkGrid = &m_world.chunkGrid();
    worldContent.visibleChunkIndices = std::span<const std::size_t>(m_visibleChunkIndices);
    submitFrame(camera, alpha, nullptr, &worldContent);
}

} // namespace odai::games::voxelcraft
