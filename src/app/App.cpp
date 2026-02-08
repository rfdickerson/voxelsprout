#include "app/App.hpp"

#include <GLFW/glfw3.h>

#include "math/Math.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <ctime>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <limits>
#include <string>

namespace {

constexpr float kMouseSensitivity = 0.1f;
constexpr float kMouseSmoothingSeconds = 0.035f;
constexpr float kMoveMaxSpeed = 5.0f;
constexpr float kMoveAcceleration = 14.0f;
constexpr float kMoveDeceleration = 18.0f;
constexpr float kJumpSpeed = 7.8f;
constexpr float kGravity = -24.0f;
constexpr float kMaxFallSpeed = -35.0f;
constexpr float kPitchMinDegrees = -89.0f;
constexpr float kPitchMaxDegrees = 89.0f;
[[maybe_unused]] constexpr float kVoxelSizeMeters = 0.25f;
constexpr float kBlockInteractMaxDistance = 6.0f;
// 1x voxel world scale: roughly Minecraft-like player proportions.
constexpr float kPlayerHeightVoxels = 1.8f;
constexpr float kPlayerDiameterVoxels = 0.8f;
constexpr float kPlayerEyeHeightVoxels = 1.62f;

constexpr float kPlayerRadius = kPlayerDiameterVoxels * 0.5f;
constexpr float kPlayerEyeHeight = kPlayerEyeHeightVoxels;
constexpr float kPlayerHeight = kPlayerHeightVoxels;
constexpr float kPlayerTopOffset = kPlayerHeight - kPlayerEyeHeight;
constexpr float kCollisionEpsilon = 0.001f;
constexpr float kHoverHeightAboveGround = 0.15f;
constexpr float kHoverResponse = 8.0f;
constexpr float kHoverMaxVerticalSpeed = 12.0f;
constexpr float kHoverManualVerticalSpeed = 8.0f;
constexpr int kHoverGroundSearchDepth = 96;
constexpr float kGamepadTriggerPressedThreshold = 0.30f;
constexpr const char* kWorldFilePath = "world.vxw";

constexpr std::array<world::VoxelType, 1> kPlaceableBlockTypes = {
    world::VoxelType::Solid
};

std::string appTimestamp();

void glfwErrorCallback(int errorCode, const char* description) {
    std::cerr << "[" << appTimestamp() << "][app][glfw] error " << errorCode << ": "
              << (description != nullptr ? description : "(no description)") << "\n";
}

std::string appTimestamp() {
    const auto now = std::chrono::system_clock::now();
    const auto epochMs = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch());
    const auto ms = static_cast<int>(epochMs.count() % 1000);
    const std::time_t timeValue = std::chrono::system_clock::to_time_t(now);

    std::tm localTime{};
#if defined(_WIN32)
    localtime_s(&localTime, &timeValue);
#else
    localtime_r(&timeValue, &localTime);
#endif

    char buffer[32]{};
    std::snprintf(
        buffer,
        sizeof(buffer),
        "%02d:%02d:%02d.%03d",
        localTime.tm_hour,
        localTime.tm_min,
        localTime.tm_sec,
        ms
    );
    return std::string(buffer);
}

float approach(float current, float target, float maxDelta) {
    const float delta = target - current;
    if (delta > maxDelta) {
        return current + maxDelta;
    }
    if (delta < -maxDelta) {
        return current - maxDelta;
    }
    return target;
}

} // namespace

namespace app {

bool App::init() {
    std::cerr << "[" << appTimestamp() << "][app] init begin\n";
    glfwSetErrorCallback(glfwErrorCallback);

    if (glfwInit() == GLFW_FALSE) {
        std::cerr << "[" << appTimestamp() << "][app] glfwInit failed\n";
        return false;
    }

    // Vulkan renderer path requires no OpenGL context.
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    m_window = glfwCreateWindow(1280, 720, "voxel_factory_toy", nullptr, nullptr);
    if (m_window == nullptr) {
        std::cerr << "[" << appTimestamp() << "][app] glfwCreateWindow failed\n";
        glfwTerminate();
        return false;
    }

    // Relative mouse mode for camera look.
    glfwSetInputMode(m_window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    const auto worldLoadStart = std::chrono::steady_clock::now();
    const std::filesystem::path worldPath{kWorldFilePath};
    if (m_chunkGrid.loadFromBinaryFile(worldPath)) {
        const auto worldLoadMs = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - worldLoadStart
        ).count();
        std::cerr << "[" << appTimestamp() << "][app] loaded world from "
                  << std::filesystem::absolute(worldPath).string()
                  << " in " << worldLoadMs << " ms\n";
    } else {
        m_chunkGrid.initializeEmptyWorld();
        const auto worldLoadMs = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - worldLoadStart
        ).count();
        std::cerr << "[" << appTimestamp() << "][app] world file missing/invalid at "
                  << std::filesystem::absolute(worldPath).string()
                  << "; using empty world (press R to regenerate) in "
                  << worldLoadMs << " ms\n";
    }
    m_simulation.initializeSingleBelt();
    const bool rendererOk = m_renderer.init(m_window, m_chunkGrid);
    if (!rendererOk) {
        std::cerr << "[" << appTimestamp() << "][app] renderer init failed\n";
        return false;
    }

    std::cerr << "[" << appTimestamp() << "][app] init complete\n";
    return true;
}

void App::run() {
    std::cerr << "[" << appTimestamp() << "][app] run begin\n";
    double previousTime = glfwGetTime();
    uint64_t frameCount = 0;

    while (m_window != nullptr && glfwWindowShouldClose(m_window) == GLFW_FALSE) {
        const double currentTime = glfwGetTime();
        const float dt = static_cast<float>(currentTime - previousTime);
        previousTime = currentTime;

        pollInput();
        if (m_input.quitRequested) {
            glfwSetWindowShouldClose(m_window, GLFW_TRUE);
            break;
        }

        update(dt);
        ++frameCount;
    }

    std::cerr << "[" << appTimestamp() << "][app] run exit after " << frameCount
              << " frame(s), windowShouldClose="
              << (m_window != nullptr ? glfwWindowShouldClose(m_window) : 1) << "\n";
}

void App::update(float dt) {
    updateCamera(dt);
    m_simulation.update(dt);

    const bool regeneratePressedThisFrame =
        !m_debugUiVisible && m_input.regenerateWorldDown && !m_wasRegenerateWorldDown;
    m_wasRegenerateWorldDown = m_input.regenerateWorldDown;
    if (regeneratePressedThisFrame) {
        regenerateWorld();
    }

    const CameraRaycastResult raycast = raycastFromCamera();

    const bool blockInteractionEnabled = !m_debugUiVisible;
    const bool placePressedThisFrame = blockInteractionEnabled && m_input.placeBlockDown && !m_wasPlaceBlockDown;
    const bool removePressedThisFrame = blockInteractionEnabled && m_input.removeBlockDown && !m_wasRemoveBlockDown;
    m_wasPlaceBlockDown = m_input.placeBlockDown;
    m_wasRemoveBlockDown = m_input.removeBlockDown;

    bool chunkEdited = false;
    std::size_t editedChunkIndex = 0;
    if (placePressedThisFrame && tryPlaceVoxelFromCameraRay(editedChunkIndex)) {
        chunkEdited = true;
    }
    if (removePressedThisFrame && tryRemoveVoxelFromCameraRay(editedChunkIndex)) {
        chunkEdited = true;
    }

    if (chunkEdited) {
        if (!m_renderer.updateChunkMesh(m_chunkGrid, editedChunkIndex)) {
            std::cerr << "[" << appTimestamp() << "][app] chunk mesh update failed after voxel edit\n";
        }
        const std::filesystem::path worldPath{kWorldFilePath};
        if (!m_chunkGrid.saveToBinaryFile(worldPath)) {
            std::cerr << "[" << appTimestamp() << "][app] failed to save world to " << worldPath.string() << " after voxel edit\n";
        }
    }

    render::VoxelPreview preview{};
    if (!m_debugUiVisible && raycast.hitSolid && raycast.hitDistance <= kBlockInteractMaxDistance) {
        if (raycast.hasHitFaceNormal) {
            auto normalToFaceId = [](int nx, int ny, int nz) -> uint32_t {
                if (nx > 0) { return 0u; }
                if (nx < 0) { return 1u; }
                if (ny > 0) { return 2u; }
                if (ny < 0) { return 3u; }
                if (nz > 0) { return 4u; }
                return 5u;
            };
            preview.faceVisible = true;
            preview.faceX = raycast.solidX;
            preview.faceY = raycast.solidY;
            preview.faceZ = raycast.solidZ;
            preview.faceId = normalToFaceId(raycast.hitFaceNormalX, raycast.hitFaceNormalY, raycast.hitFaceNormalZ);
        }

        const bool showRemovePreview = m_input.removeBlockDown;
        if (showRemovePreview) {
            preview.visible = true;
            preview.mode = render::VoxelPreview::Mode::Remove;
            preview.x = raycast.solidX;
            preview.y = raycast.solidY;
            preview.z = raycast.solidZ;
            preview.brushSize = 1;
        } else {
            int targetX = 0;
            int targetY = 0;
            int targetZ = 0;
            if (computePlacementVoxelFromRaycast(raycast, targetX, targetY, targetZ)) {
                if (isWorldVoxelInBounds(targetX, targetY, targetZ)) {
                    preview.visible = true;
                    preview.mode = render::VoxelPreview::Mode::Add;
                    preview.x = targetX;
                    preview.y = targetY;
                    preview.z = targetZ;
                    preview.brushSize = 1;
                }
            }
        }
    }

    const render::CameraPose cameraPose{
        m_camera.x,
        m_camera.y,
        m_camera.z,
        m_camera.yawDegrees,
        m_camera.pitchDegrees,
        m_camera.fovDegrees
    };
    m_renderer.renderFrame(m_chunkGrid, m_simulation, cameraPose, preview);
}

void App::shutdown() {
    std::cerr << "[" << appTimestamp() << "][app] shutdown begin\n";
    m_renderer.shutdown();

    if (m_window != nullptr) {
        glfwDestroyWindow(m_window);
        m_window = nullptr;
    }

    glfwTerminate();
    std::cerr << "[" << appTimestamp() << "][app] shutdown complete\n";
}

void App::pollInput() {
    glfwPollEvents();

    bool uiVisibilityChanged = false;
    const bool toggleUiDown = glfwGetKey(m_window, GLFW_KEY_F1) == GLFW_PRESS;
    if (toggleUiDown && !m_wasToggleDebugUiDown) {
        m_debugUiVisible = !m_debugUiVisible;
        uiVisibilityChanged = true;
    }
    m_wasToggleDebugUiDown = toggleUiDown;
    m_renderer.setDebugUiVisible(m_debugUiVisible);
    const bool rendererUiVisible = m_renderer.isDebugUiVisible();
    if (rendererUiVisible != m_debugUiVisible) {
        m_debugUiVisible = rendererUiVisible;
        uiVisibilityChanged = true;
    }
    if (uiVisibilityChanged) {
        glfwSetInputMode(m_window, GLFW_CURSOR, m_debugUiVisible ? GLFW_CURSOR_NORMAL : GLFW_CURSOR_DISABLED);
        m_hasMouseSample = false;
    }

    m_input.quitRequested = glfwGetKey(m_window, GLFW_KEY_ESCAPE) == GLFW_PRESS;
    m_input.moveForward = glfwGetKey(m_window, GLFW_KEY_W) == GLFW_PRESS;
    m_input.moveBackward = glfwGetKey(m_window, GLFW_KEY_S) == GLFW_PRESS;
    m_input.moveLeft = glfwGetKey(m_window, GLFW_KEY_A) == GLFW_PRESS;
    m_input.moveRight = glfwGetKey(m_window, GLFW_KEY_D) == GLFW_PRESS;
    m_input.moveUp = glfwGetKey(m_window, GLFW_KEY_SPACE) == GLFW_PRESS;
    m_input.moveDown =
        glfwGetKey(m_window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
        glfwGetKey(m_window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS;
    m_input.toggleHoverDown = glfwGetKey(m_window, GLFW_KEY_H) == GLFW_PRESS;
    m_input.regenerateWorldDown = glfwGetKey(m_window, GLFW_KEY_R) == GLFW_PRESS;
    bool controllerPlaceDown = false;
    bool controllerRemoveDown = false;
    bool controllerPrevBlockDown = false;
    bool controllerNextBlockDown = false;
    GLFWgamepadstate gamepadState{};
    const bool hasGamepad =
        glfwJoystickIsGamepad(GLFW_JOYSTICK_1) == GLFW_TRUE &&
        glfwGetGamepadState(GLFW_JOYSTICK_1, &gamepadState) == GLFW_TRUE;
    if (hasGamepad != m_gamepadConnected) {
        m_gamepadConnected = hasGamepad;
        if (m_gamepadConnected) {
            std::cerr << "[" << appTimestamp() << "][app] gamepad connected: RT place, LT remove, LB/RB block\n";
        } else {
            std::cerr << "[" << appTimestamp() << "][app] gamepad disconnected\n";
        }
    }
    if (hasGamepad) {
        controllerPlaceDown = gamepadState.axes[GLFW_GAMEPAD_AXIS_RIGHT_TRIGGER] > kGamepadTriggerPressedThreshold;
        controllerRemoveDown = gamepadState.axes[GLFW_GAMEPAD_AXIS_LEFT_TRIGGER] > kGamepadTriggerPressedThreshold;
        controllerPrevBlockDown = gamepadState.buttons[GLFW_GAMEPAD_BUTTON_LEFT_BUMPER] == GLFW_PRESS;
        controllerNextBlockDown = gamepadState.buttons[GLFW_GAMEPAD_BUTTON_RIGHT_BUMPER] == GLFW_PRESS;
    }

    const bool prevBlockDown = controllerPrevBlockDown;
    const bool nextBlockDown = controllerNextBlockDown;
    if (!m_debugUiVisible && prevBlockDown && !m_wasPrevBlockDown) {
        cycleSelectedBlock(-1);
    }
    if (!m_debugUiVisible && nextBlockDown && !m_wasNextBlockDown) {
        cycleSelectedBlock(+1);
    }
    m_wasPrevBlockDown = prevBlockDown;
    m_wasNextBlockDown = nextBlockDown;

    m_input.placeBlockDown =
        glfwGetMouseButton(m_window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS ||
        controllerPlaceDown;
    m_input.removeBlockDown =
        glfwGetMouseButton(m_window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS ||
        controllerRemoveDown;

    double mouseX = 0.0;
    double mouseY = 0.0;
    glfwGetCursorPos(m_window, &mouseX, &mouseY);

    if (!m_hasMouseSample) {
        m_lastMouseX = mouseX;
        m_lastMouseY = mouseY;
        m_hasMouseSample = true;
    }

    m_input.mouseDeltaX = static_cast<float>(mouseX - m_lastMouseX);
    m_input.mouseDeltaY = static_cast<float>(mouseY - m_lastMouseY);

    if (m_debugUiVisible) {
        m_input.mouseDeltaX = 0.0f;
        m_input.mouseDeltaY = 0.0f;
    }

    m_lastMouseX = mouseX;
    m_lastMouseY = mouseY;
}

void App::updateCamera(float dt) {
    if (m_input.toggleHoverDown && !m_wasToggleHoverDown) {
        m_hoverEnabled = !m_hoverEnabled;
        m_camera.velocityY = 0.0f;
        std::cerr << "[" << appTimestamp() << "][app] hover " << (m_hoverEnabled ? "enabled" : "disabled") << " (H)\n";
    }
    m_wasToggleHoverDown = m_input.toggleHoverDown;

    const float mouseSmoothingAlpha = 1.0f - std::exp(-dt / kMouseSmoothingSeconds);
    m_camera.smoothedMouseDeltaX += (m_input.mouseDeltaX - m_camera.smoothedMouseDeltaX) * mouseSmoothingAlpha;
    m_camera.smoothedMouseDeltaY += (m_input.mouseDeltaY - m_camera.smoothedMouseDeltaY) * mouseSmoothingAlpha;

    m_camera.yawDegrees += m_camera.smoothedMouseDeltaX * kMouseSensitivity;
    m_camera.pitchDegrees += m_camera.smoothedMouseDeltaY * kMouseSensitivity;
    m_camera.pitchDegrees = std::clamp(m_camera.pitchDegrees, kPitchMinDegrees, kPitchMaxDegrees);

    const float yawRadians = math::radians(m_camera.yawDegrees);
    const math::Vector3 forward{std::cos(yawRadians), 0.0f, std::sin(yawRadians)};
    const math::Vector3 right{-forward.z, 0.0f, forward.x};
    math::Vector3 moveDirection{};

    if (m_input.moveForward) {
        moveDirection += forward;
    }
    if (m_input.moveBackward) {
        moveDirection -= forward;
    }
    if (m_input.moveRight) {
        moveDirection += right;
    }
    if (m_input.moveLeft) {
        moveDirection -= right;
    }

    const float moveLengthSq = math::lengthSquared(moveDirection);
    const float moveLength = std::sqrt(moveLengthSq);
    float targetVelocityX = 0.0f;
    float targetVelocityZ = 0.0f;
    if (moveLength > 0.0f) {
        moveDirection /= moveLength;
        const math::Vector3 targetVelocity = moveDirection * kMoveMaxSpeed;
        targetVelocityX = targetVelocity.x;
        targetVelocityZ = targetVelocity.z;
    }

    const float accelPerFrame = kMoveAcceleration * dt;
    const float decelPerFrame = kMoveDeceleration * dt;

    const float maxDeltaX = (std::fabs(targetVelocityX) > std::fabs(m_camera.velocityX)) ? accelPerFrame : decelPerFrame;
    const float maxDeltaZ = (std::fabs(targetVelocityZ) > std::fabs(m_camera.velocityZ)) ? accelPerFrame : decelPerFrame;

    m_camera.velocityX = approach(m_camera.velocityX, targetVelocityX, maxDeltaX);
    m_camera.velocityZ = approach(m_camera.velocityZ, targetVelocityZ, maxDeltaZ);

    if (m_hoverEnabled) {
        float hoverVerticalSpeed = 0.0f;
        int supportY = 0;
        if (findGroundSupportY(m_camera.x, m_camera.y, m_camera.z, supportY)) {
            const float targetEyeY = static_cast<float>(supportY + 1) + kPlayerEyeHeight + kHoverHeightAboveGround;
            const float yError = targetEyeY - m_camera.y;
            hoverVerticalSpeed = std::clamp(
                yError * kHoverResponse,
                -kHoverMaxVerticalSpeed,
                kHoverMaxVerticalSpeed
            );
        } else {
            hoverVerticalSpeed = 0.0f;
        }

        if (m_input.moveUp) {
            hoverVerticalSpeed = std::max(hoverVerticalSpeed, kHoverManualVerticalSpeed);
        }
        if (m_input.moveDown) {
            hoverVerticalSpeed = std::min(hoverVerticalSpeed, -kHoverManualVerticalSpeed);
        }

        m_camera.velocityY = hoverVerticalSpeed;
        m_camera.onGround = false;
    } else {
        if (m_input.moveUp && m_camera.onGround) {
            m_camera.velocityY = kJumpSpeed;
            m_camera.onGround = false;
        }
        m_camera.velocityY = std::max(m_camera.velocityY + (kGravity * dt), kMaxFallSpeed);
    }
    resolvePlayerCollisions(dt);
}

bool App::isSolidWorldVoxel(int worldX, int worldY, int worldZ) const {
    if (worldY < 0) {
        return true;
    }

    const world::Chunk* chunk = nullptr;
    int localX = 0;
    int localY = 0;
    int localZ = 0;
    if (worldToChunkLocalConst(worldX, worldY, worldZ, chunk, localX, localY, localZ)) {
        return chunk->isSolid(localX, localY, localZ);
    }

    return false;
}

bool App::worldToChunkLocal(
    int worldX,
    int worldY,
    int worldZ,
    std::size_t& outChunkIndex,
    int& outLocalX,
    int& outLocalY,
    int& outLocalZ
) const {
    const std::vector<world::Chunk>& chunks = m_chunkGrid.chunks();
    for (std::size_t chunkIndex = 0; chunkIndex < chunks.size(); ++chunkIndex) {
        const world::Chunk& chunk = chunks[chunkIndex];
        const int chunkMinX = chunk.chunkX() * world::Chunk::kSizeX;
        const int chunkMinY = chunk.chunkY() * world::Chunk::kSizeY;
        const int chunkMinZ = chunk.chunkZ() * world::Chunk::kSizeZ;
        const int localX = worldX - chunkMinX;
        const int localY = worldY - chunkMinY;
        const int localZ = worldZ - chunkMinZ;
        const bool insideChunk =
            localX >= 0 && localX < world::Chunk::kSizeX &&
            localY >= 0 && localY < world::Chunk::kSizeY &&
            localZ >= 0 && localZ < world::Chunk::kSizeZ;
        if (!insideChunk) {
            continue;
        }

        outChunkIndex = chunkIndex;
        outLocalX = localX;
        outLocalY = localY;
        outLocalZ = localZ;
        return true;
    }

    return false;
}

bool App::worldToChunkLocalConst(
    int worldX,
    int worldY,
    int worldZ,
    const world::Chunk*& outChunk,
    int& outLocalX,
    int& outLocalY,
    int& outLocalZ
) const {
    std::size_t chunkIndex = 0;
    if (!worldToChunkLocal(worldX, worldY, worldZ, chunkIndex, outLocalX, outLocalY, outLocalZ)) {
        return false;
    }

    outChunk = &m_chunkGrid.chunks()[chunkIndex];
    return true;
}

bool App::findGroundSupportY(float eyeX, float eyeY, float eyeZ, int& outSupportY) const {
    const int startX = static_cast<int>(std::floor(eyeX - kPlayerRadius));
    const int endX = static_cast<int>(std::floor(eyeX + kPlayerRadius - kCollisionEpsilon));
    const int startZ = static_cast<int>(std::floor(eyeZ - kPlayerRadius));
    const int endZ = static_cast<int>(std::floor(eyeZ + kPlayerRadius - kCollisionEpsilon));

    const float feetY = eyeY - kPlayerEyeHeight;
    const int topSupportY = static_cast<int>(std::floor(feetY - kCollisionEpsilon)) - 1;
    const int minSupportY = std::max(0, topSupportY - kHoverGroundSearchDepth);

    for (int supportY = topSupportY; supportY >= minSupportY; --supportY) {
        for (int z = startZ; z <= endZ; ++z) {
            for (int x = startX; x <= endX; ++x) {
                if (isSolidWorldVoxel(x, supportY, z)) {
                    outSupportY = supportY;
                    return true;
                }
            }
        }
    }

    return false;
}

bool App::doesPlayerOverlapSolid(float eyeX, float eyeY, float eyeZ) const {
    const float minX = eyeX - kPlayerRadius;
    const float maxX = eyeX + kPlayerRadius;
    const float minY = eyeY - kPlayerEyeHeight;
    const float maxY = eyeY + kPlayerTopOffset;
    const float minZ = eyeZ - kPlayerRadius;
    const float maxZ = eyeZ + kPlayerRadius;

    const int startX = static_cast<int>(std::floor(minX));
    const int endX = static_cast<int>(std::floor(maxX - kCollisionEpsilon));
    const int startY = static_cast<int>(std::floor(minY));
    const int endY = static_cast<int>(std::floor(maxY - kCollisionEpsilon));
    const int startZ = static_cast<int>(std::floor(minZ));
    const int endZ = static_cast<int>(std::floor(maxZ - kCollisionEpsilon));

    for (int y = startY; y <= endY; ++y) {
        for (int z = startZ; z <= endZ; ++z) {
            for (int x = startX; x <= endX; ++x) {
                if (isSolidWorldVoxel(x, y, z)) {
                    return true;
                }
            }
        }
    }
    return false;
}

void App::resolvePlayerCollisions(float dt) {
    const float totalDx = m_camera.velocityX * dt;
    const float totalDy = m_camera.velocityY * dt;
    const float totalDz = m_camera.velocityZ * dt;
    const float maxDelta = std::max({std::fabs(totalDx), std::fabs(totalDy), std::fabs(totalDz)});
    const int steps = std::max(1, static_cast<int>(std::ceil(maxDelta / 0.45f)));
    const float stepDx = totalDx / static_cast<float>(steps);
    const float stepDy = totalDy / static_cast<float>(steps);
    const float stepDz = totalDz / static_cast<float>(steps);

    bool groundedThisFrame = false;

    auto resolveHorizontalX = [&](float deltaX) {
        if (deltaX == 0.0f) {
            return;
        }

        m_camera.x += deltaX;
        if (!doesPlayerOverlapSolid(m_camera.x, m_camera.y, m_camera.z)) {
            return;
        }

        const float minY = m_camera.y - kPlayerEyeHeight;
        const float maxY = m_camera.y + kPlayerTopOffset;
        const float minZ = m_camera.z - kPlayerRadius;
        const float maxZ = m_camera.z + kPlayerRadius;
        const int startY = static_cast<int>(std::floor(minY));
        const int endY = static_cast<int>(std::floor(maxY - kCollisionEpsilon));
        const int startZ = static_cast<int>(std::floor(minZ));
        const int endZ = static_cast<int>(std::floor(maxZ - kCollisionEpsilon));
        const int startX = static_cast<int>(std::floor(m_camera.x - kPlayerRadius));
        const int endX = static_cast<int>(std::floor(m_camera.x + kPlayerRadius - kCollisionEpsilon));

        if (deltaX > 0.0f) {
            int blockingX = std::numeric_limits<int>::max();
            for (int y = startY; y <= endY; ++y) {
                for (int z = startZ; z <= endZ; ++z) {
                    for (int x = startX; x <= endX; ++x) {
                        if (isSolidWorldVoxel(x, y, z)) {
                            blockingX = std::min(blockingX, x);
                        }
                    }
                }
            }
            if (blockingX != std::numeric_limits<int>::max()) {
                m_camera.x = static_cast<float>(blockingX) - kPlayerRadius - kCollisionEpsilon;
            }
        } else {
            int blockingX = std::numeric_limits<int>::lowest();
            for (int y = startY; y <= endY; ++y) {
                for (int z = startZ; z <= endZ; ++z) {
                    for (int x = startX; x <= endX; ++x) {
                        if (isSolidWorldVoxel(x, y, z)) {
                            blockingX = std::max(blockingX, x);
                        }
                    }
                }
            }
            if (blockingX != std::numeric_limits<int>::lowest()) {
                m_camera.x = static_cast<float>(blockingX + 1) + kPlayerRadius + kCollisionEpsilon;
            }
        }

        if (doesPlayerOverlapSolid(m_camera.x, m_camera.y, m_camera.z)) {
            m_camera.x -= deltaX;
        }
        m_camera.velocityX = 0.0f;
    };

    auto resolveHorizontalZ = [&](float deltaZ) {
        if (deltaZ == 0.0f) {
            return;
        }

        m_camera.z += deltaZ;
        if (!doesPlayerOverlapSolid(m_camera.x, m_camera.y, m_camera.z)) {
            return;
        }

        const float minX = m_camera.x - kPlayerRadius;
        const float maxX = m_camera.x + kPlayerRadius;
        const float minY = m_camera.y - kPlayerEyeHeight;
        const float maxY = m_camera.y + kPlayerTopOffset;
        const int startX = static_cast<int>(std::floor(minX));
        const int endX = static_cast<int>(std::floor(maxX - kCollisionEpsilon));
        const int startY = static_cast<int>(std::floor(minY));
        const int endY = static_cast<int>(std::floor(maxY - kCollisionEpsilon));
        const int startZ = static_cast<int>(std::floor(m_camera.z - kPlayerRadius));
        const int endZ = static_cast<int>(std::floor(m_camera.z + kPlayerRadius - kCollisionEpsilon));

        if (deltaZ > 0.0f) {
            int blockingZ = std::numeric_limits<int>::max();
            for (int y = startY; y <= endY; ++y) {
                for (int z = startZ; z <= endZ; ++z) {
                    for (int x = startX; x <= endX; ++x) {
                        if (isSolidWorldVoxel(x, y, z)) {
                            blockingZ = std::min(blockingZ, z);
                        }
                    }
                }
            }
            if (blockingZ != std::numeric_limits<int>::max()) {
                m_camera.z = static_cast<float>(blockingZ) - kPlayerRadius - kCollisionEpsilon;
            }
        } else {
            int blockingZ = std::numeric_limits<int>::lowest();
            for (int y = startY; y <= endY; ++y) {
                for (int z = startZ; z <= endZ; ++z) {
                    for (int x = startX; x <= endX; ++x) {
                        if (isSolidWorldVoxel(x, y, z)) {
                            blockingZ = std::max(blockingZ, z);
                        }
                    }
                }
            }
            if (blockingZ != std::numeric_limits<int>::lowest()) {
                m_camera.z = static_cast<float>(blockingZ + 1) + kPlayerRadius + kCollisionEpsilon;
            }
        }

        if (doesPlayerOverlapSolid(m_camera.x, m_camera.y, m_camera.z)) {
            m_camera.z -= deltaZ;
        }
        m_camera.velocityZ = 0.0f;
    };

    auto resolveVerticalY = [&](float deltaY) {
        if (deltaY == 0.0f) {
            return;
        }

        m_camera.y += deltaY;
        if (!doesPlayerOverlapSolid(m_camera.x, m_camera.y, m_camera.z)) {
            return;
        }

        const float minX = m_camera.x - kPlayerRadius;
        const float maxX = m_camera.x + kPlayerRadius;
        const float minZ = m_camera.z - kPlayerRadius;
        const float maxZ = m_camera.z + kPlayerRadius;
        const int startX = static_cast<int>(std::floor(minX));
        const int endX = static_cast<int>(std::floor(maxX - kCollisionEpsilon));
        const int startZ = static_cast<int>(std::floor(minZ));
        const int endZ = static_cast<int>(std::floor(maxZ - kCollisionEpsilon));
        const int startY = static_cast<int>(std::floor(m_camera.y - kPlayerEyeHeight));
        const int endY = static_cast<int>(std::floor(m_camera.y + kPlayerTopOffset - kCollisionEpsilon));

        if (deltaY > 0.0f) {
            int blockingY = std::numeric_limits<int>::max();
            for (int y = startY; y <= endY; ++y) {
                for (int z = startZ; z <= endZ; ++z) {
                    for (int x = startX; x <= endX; ++x) {
                        if (isSolidWorldVoxel(x, y, z)) {
                            blockingY = std::min(blockingY, y);
                        }
                    }
                }
            }
            if (blockingY != std::numeric_limits<int>::max()) {
                m_camera.y = static_cast<float>(blockingY) - kPlayerTopOffset - kCollisionEpsilon;
            }
        } else {
            int blockingY = std::numeric_limits<int>::lowest();
            for (int y = startY; y <= endY; ++y) {
                for (int z = startZ; z <= endZ; ++z) {
                    for (int x = startX; x <= endX; ++x) {
                        if (isSolidWorldVoxel(x, y, z)) {
                            blockingY = std::max(blockingY, y);
                        }
                    }
                }
            }
            if (blockingY != std::numeric_limits<int>::lowest()) {
                m_camera.y = static_cast<float>(blockingY + 1) + kPlayerEyeHeight + kCollisionEpsilon;
                groundedThisFrame = true;
            }
        }

        if (doesPlayerOverlapSolid(m_camera.x, m_camera.y, m_camera.z)) {
            m_camera.y -= deltaY;
        }
        m_camera.velocityY = 0.0f;
    };

    for (int step = 0; step < steps; ++step) {
        resolveHorizontalX(stepDx);
        resolveHorizontalZ(stepDz);
        resolveVerticalY(stepDy);
    }

    m_camera.onGround = groundedThisFrame;
}

App::CameraRaycastResult App::raycastFromCamera() const {
    CameraRaycastResult result{};
    if (m_chunkGrid.chunks().empty()) {
        return result;
    }

    const float yawRadians = math::radians(m_camera.yawDegrees);
    const float pitchRadians = math::radians(m_camera.pitchDegrees);
    const float cosPitch = std::cos(pitchRadians);
    const math::Vector3 rayDirection = math::normalize(math::Vector3{
        std::cos(yawRadians) * cosPitch,
        std::sin(pitchRadians),
        std::sin(yawRadians) * cosPitch
    });
    if (math::lengthSquared(rayDirection) <= 0.0f) {
        return result;
    }

    // Nudge origin slightly forward so close-surface targeting does not start inside solids.
    const math::Vector3 rayOrigin =
        math::Vector3{m_camera.x, m_camera.y, m_camera.z} + (rayDirection * 0.02f);
    constexpr float kRayMaxDistance = kBlockInteractMaxDistance + 1.0f;

    int vx = static_cast<int>(std::floor(rayOrigin.x));
    int vy = static_cast<int>(std::floor(rayOrigin.y));
    int vz = static_cast<int>(std::floor(rayOrigin.z));

    const float kInf = std::numeric_limits<float>::infinity();

    const int stepX = (rayDirection.x > 0.0f) ? 1 : (rayDirection.x < 0.0f ? -1 : 0);
    const int stepY = (rayDirection.y > 0.0f) ? 1 : (rayDirection.y < 0.0f ? -1 : 0);
    const int stepZ = (rayDirection.z > 0.0f) ? 1 : (rayDirection.z < 0.0f ? -1 : 0);

    const float invAbsDirX = (stepX != 0) ? (1.0f / std::abs(rayDirection.x)) : kInf;
    const float invAbsDirY = (stepY != 0) ? (1.0f / std::abs(rayDirection.y)) : kInf;
    const float invAbsDirZ = (stepZ != 0) ? (1.0f / std::abs(rayDirection.z)) : kInf;

    const float voxelBoundaryX = (stepX > 0) ? static_cast<float>(vx + 1) : static_cast<float>(vx);
    const float voxelBoundaryY = (stepY > 0) ? static_cast<float>(vy + 1) : static_cast<float>(vy);
    const float voxelBoundaryZ = (stepZ > 0) ? static_cast<float>(vz + 1) : static_cast<float>(vz);

    float tMaxX = (stepX != 0) ? ((voxelBoundaryX - rayOrigin.x) / rayDirection.x) : kInf;
    float tMaxY = (stepY != 0) ? ((voxelBoundaryY - rayOrigin.y) / rayDirection.y) : kInf;
    float tMaxZ = (stepZ != 0) ? ((voxelBoundaryZ - rayOrigin.z) / rayDirection.z) : kInf;
    float tDeltaX = invAbsDirX;
    float tDeltaY = invAbsDirY;
    float tDeltaZ = invAbsDirZ;

    int hitFaceNormalX = 0;
    int hitFaceNormalY = 0;
    int hitFaceNormalZ = 0;
    bool hasHitFaceNormal = false;

    float distance = 0.0f;
    while (distance <= kRayMaxDistance) {
        if (isSolidWorldVoxel(vx, vy, vz)) {
            if (!hasHitFaceNormal) {
                // If we start inside a solid voxel, take one DDA step first so we get a stable face.
                if (tMaxX <= tMaxY && tMaxX <= tMaxZ) {
                    vx += stepX;
                    distance = tMaxX;
                    tMaxX += tDeltaX;
                    hitFaceNormalX = -stepX;
                    hitFaceNormalY = 0;
                    hitFaceNormalZ = 0;
                    hasHitFaceNormal = (stepX != 0);
                } else if (tMaxY <= tMaxX && tMaxY <= tMaxZ) {
                    vy += stepY;
                    distance = tMaxY;
                    tMaxY += tDeltaY;
                    hitFaceNormalX = 0;
                    hitFaceNormalY = -stepY;
                    hitFaceNormalZ = 0;
                    hasHitFaceNormal = (stepY != 0);
                } else {
                    vz += stepZ;
                    distance = tMaxZ;
                    tMaxZ += tDeltaZ;
                    hitFaceNormalX = 0;
                    hitFaceNormalY = 0;
                    hitFaceNormalZ = -stepZ;
                    hasHitFaceNormal = (stepZ != 0);
                }
                continue;
            }

            result.hitSolid = true;
            result.solidX = vx;
            result.solidY = vy;
            result.solidZ = vz;
            result.hitDistance = distance;
            result.hasHitFaceNormal = hasHitFaceNormal;
            result.hitFaceNormalX = hitFaceNormalX;
            result.hitFaceNormalY = hitFaceNormalY;
            result.hitFaceNormalZ = hitFaceNormalZ;

            if (hasHitFaceNormal) {
                const int adjacentX = vx + hitFaceNormalX;
                const int adjacentY = vy + hitFaceNormalY;
                const int adjacentZ = vz + hitFaceNormalZ;
                if (isWorldVoxelInBounds(adjacentX, adjacentY, adjacentZ) &&
                    !isSolidWorldVoxel(adjacentX, adjacentY, adjacentZ)) {
                    result.hasAdjacentEmpty = true;
                    result.adjacentEmptyX = adjacentX;
                    result.adjacentEmptyY = adjacentY;
                    result.adjacentEmptyZ = adjacentZ;
                }
            }
            return result;
        }

        if (tMaxX <= tMaxY && tMaxX <= tMaxZ) {
            vx += stepX;
            distance = tMaxX;
            tMaxX += tDeltaX;
            hitFaceNormalX = -stepX;
            hitFaceNormalY = 0;
            hitFaceNormalZ = 0;
            hasHitFaceNormal = (stepX != 0);
        } else if (tMaxY <= tMaxX && tMaxY <= tMaxZ) {
            vy += stepY;
            distance = tMaxY;
            tMaxY += tDeltaY;
            hitFaceNormalX = 0;
            hitFaceNormalY = -stepY;
            hitFaceNormalZ = 0;
            hasHitFaceNormal = (stepY != 0);
        } else {
            vz += stepZ;
            distance = tMaxZ;
            tMaxZ += tDeltaZ;
            hitFaceNormalX = 0;
            hitFaceNormalY = 0;
            hitFaceNormalZ = -stepZ;
            hasHitFaceNormal = (stepZ != 0);
        }
    }

    return result;
}

bool App::isWorldVoxelInBounds(int x, int y, int z) const {
    std::size_t chunkIndex = 0;
    int localX = 0;
    int localY = 0;
    int localZ = 0;
    return worldToChunkLocal(x, y, z, chunkIndex, localX, localY, localZ);
}

void App::cycleSelectedBlock(int direction) {
    if (kPlaceableBlockTypes.empty()) {
        m_selectedBlockIndex = 0;
        return;
    }

    const int count = static_cast<int>(kPlaceableBlockTypes.size());
    m_selectedBlockIndex = (m_selectedBlockIndex + direction) % count;
    if (m_selectedBlockIndex < 0) {
        m_selectedBlockIndex += count;
    }
    std::cerr << "[" << appTimestamp() << "][app] selected block " << (m_selectedBlockIndex + 1) << "/" << count << "\n";
}

world::Voxel App::selectedPlaceVoxel() const {
    if (kPlaceableBlockTypes.empty()) {
        return world::Voxel{world::VoxelType::Solid};
    }
    const int clampedIndex = std::clamp(
        m_selectedBlockIndex,
        0,
        static_cast<int>(kPlaceableBlockTypes.size()) - 1
    );
    return world::Voxel{kPlaceableBlockTypes[static_cast<std::size_t>(clampedIndex)]};
}

bool App::computePlacementVoxelFromRaycast(const CameraRaycastResult& raycast, int& outX, int& outY, int& outZ) const {
    if (!raycast.hitSolid || !raycast.hasHitFaceNormal) {
        return false;
    }

    const int candidateX = raycast.solidX + raycast.hitFaceNormalX;
    const int candidateY = raycast.solidY + raycast.hitFaceNormalY;
    const int candidateZ = raycast.solidZ + raycast.hitFaceNormalZ;
    if (!isWorldVoxelInBounds(candidateX, candidateY, candidateZ)) {
        return false;
    }

    outX = candidateX;
    outY = candidateY;
    outZ = candidateZ;
    return true;
}

bool App::applyVoxelEdit(
    int targetX,
    int targetY,
    int targetZ,
    world::Voxel voxel,
    std::size_t& outEditedChunkIndex
) {
    int localX = 0;
    int localY = 0;
    int localZ = 0;
    if (!worldToChunkLocal(targetX, targetY, targetZ, outEditedChunkIndex, localX, localY, localZ)) {
        return false;
    }

    world::Chunk& chunk = m_chunkGrid.chunks()[outEditedChunkIndex];
    if (chunk.voxelAt(localX, localY, localZ).type == voxel.type) {
        return false;
    }
    chunk.setVoxel(localX, localY, localZ, voxel);
    return true;
}

void App::regenerateWorld() {
    m_chunkGrid.initializeFlatWorld();
    if (!m_renderer.updateChunkMesh(m_chunkGrid)) {
        std::cerr << "[" << appTimestamp() << "][app] world regenerate failed to update chunk meshes\n";
    }

    const std::filesystem::path worldPath{kWorldFilePath};
    if (m_chunkGrid.saveToBinaryFile(worldPath)) {
        std::cerr << "[" << appTimestamp() << "][app] world regenerated and saved to " << worldPath.string() << " (R)\n";
    } else {
        std::cerr << "[" << appTimestamp() << "][app] world regenerated, but failed to save " << worldPath.string() << "\n";
    }
}

bool App::tryPlaceVoxelFromCameraRay(std::size_t& outEditedChunkIndex) {
    if (m_chunkGrid.chunks().empty()) {
        return false;
    }

    const CameraRaycastResult raycast = raycastFromCamera();
    if (!raycast.hitSolid || raycast.hitDistance > kBlockInteractMaxDistance) {
        return false;
    }

    int targetX = 0;
    int targetY = 0;
    int targetZ = 0;
    if (!computePlacementVoxelFromRaycast(raycast, targetX, targetY, targetZ)) {
        return false;
    }

    return applyVoxelEdit(targetX, targetY, targetZ, selectedPlaceVoxel(), outEditedChunkIndex);
}

bool App::tryRemoveVoxelFromCameraRay(std::size_t& outEditedChunkIndex) {
    if (m_chunkGrid.chunks().empty()) {
        return false;
    }

    const CameraRaycastResult raycast = raycastFromCamera();
    if (!raycast.hitSolid || raycast.hitDistance > kBlockInteractMaxDistance) {
        return false;
    }

    return applyVoxelEdit(
        raycast.solidX,
        raycast.solidY,
        raycast.solidZ,
        world::Voxel{world::VoxelType::Empty},
        outEditedChunkIndex
    );
}

} // namespace app
