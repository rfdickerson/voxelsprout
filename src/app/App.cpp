#include "app/App.hpp"

#include <GLFW/glfw3.h>

#include "math/Math.hpp"

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <iostream>
#include <limits>

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
constexpr float kBlockInteractMaxDistance = 5.0f;
[[maybe_unused]] constexpr float kVoxelSizeMeters = 0.25f;
constexpr float kPlayerHeightVoxels = 7.0f;
constexpr float kPlayerDiameterVoxels = 2.0f;
constexpr float kPlayerEyeHeightVoxels = 6.0f;

constexpr float kPlayerRadius = kPlayerDiameterVoxels * 0.5f;
constexpr float kPlayerEyeHeight = kPlayerEyeHeightVoxels;
constexpr float kPlayerHeight = kPlayerHeightVoxels;
constexpr float kPlayerTopOffset = kPlayerHeight - kPlayerEyeHeight;
constexpr float kCollisionEpsilon = 0.001f;
constexpr float kHoverHeightAboveGround = 2.0f;
constexpr float kHoverResponse = 8.0f;
constexpr float kHoverMaxVerticalSpeed = 12.0f;
constexpr float kHoverManualVerticalSpeed = 8.0f;
constexpr int kHoverGroundSearchDepth = 96;

void glfwErrorCallback(int errorCode, const char* description) {
    std::cerr << "[app][glfw] error " << errorCode << ": "
              << (description != nullptr ? description : "(no description)") << "\n";
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
    std::cerr << "[app] init begin\n";
    glfwSetErrorCallback(glfwErrorCallback);

    if (glfwInit() == GLFW_FALSE) {
        std::cerr << "[app] glfwInit failed\n";
        return false;
    }

    // Vulkan renderer path requires no OpenGL context.
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    m_window = glfwCreateWindow(1280, 720, "voxel_factory_toy", nullptr, nullptr);
    if (m_window == nullptr) {
        std::cerr << "[app] glfwCreateWindow failed\n";
        glfwTerminate();
        return false;
    }

    // Relative mouse mode for camera look.
    glfwSetInputMode(m_window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    m_chunkGrid.initializeFlatWorld();
    m_simulation.initializeSingleBelt();
    const bool rendererOk = m_renderer.init(m_window, m_chunkGrid);
    if (!rendererOk) {
        std::cerr << "[app] renderer init failed\n";
        return false;
    }

    std::cerr << "[app] init complete\n";
    return true;
}

void App::run() {
    std::cerr << "[app] run begin\n";
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

    std::cerr << "[app] run exit after " << frameCount
              << " frame(s), windowShouldClose="
              << (m_window != nullptr ? glfwWindowShouldClose(m_window) : 1) << "\n";
}

void App::update(float dt) {
    updateCamera(dt);
    m_simulation.update(dt);

    const CameraRaycastResult raycast = raycastFromCamera();

    const bool blockInteractionEnabled = !m_debugUiVisible;
    const bool placePressedThisFrame = blockInteractionEnabled && m_input.placeBlockDown && !m_wasPlaceBlockDown;
    const bool removePressedThisFrame = blockInteractionEnabled && m_input.removeBlockDown && !m_wasRemoveBlockDown;
    m_wasPlaceBlockDown = m_input.placeBlockDown;
    m_wasRemoveBlockDown = m_input.removeBlockDown;

    bool chunkEdited = false;
    if (placePressedThisFrame && tryPlaceVoxelFromCameraRay()) {
        chunkEdited = true;
    }
    if (removePressedThisFrame && tryRemoveVoxelFromCameraRay()) {
        chunkEdited = true;
    }

    if (chunkEdited) {
        if (!m_renderer.updateChunkMesh(m_chunkGrid)) {
            std::cerr << "[app] chunk mesh update failed after voxel edit\n";
        }
    }

    render::VoxelPreview preview{};
    if (!m_debugUiVisible && raycast.hitSolid && raycast.hitDistance <= kBlockInteractMaxDistance) {
        const bool showRemovePreview = m_input.removeBlockDown;
        if (showRemovePreview) {
            preview.visible = true;
            preview.mode = render::VoxelPreview::Mode::Remove;
            preview.x = raycast.solidX;
            preview.y = raycast.solidY;
            preview.z = raycast.solidZ;
        } else if (raycast.hasAdjacentEmpty) {
            preview.visible = true;
            preview.mode = render::VoxelPreview::Mode::Add;
            preview.x = raycast.adjacentEmptyX;
            preview.y = raycast.adjacentEmptyY;
            preview.z = raycast.adjacentEmptyZ;
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
    std::cerr << "[app] shutdown begin\n";
    m_renderer.shutdown();

    if (m_window != nullptr) {
        glfwDestroyWindow(m_window);
        m_window = nullptr;
    }

    glfwTerminate();
    std::cerr << "[app] shutdown complete\n";
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
    m_input.placeBlockDown = glfwGetMouseButton(m_window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
    m_input.removeBlockDown = glfwGetMouseButton(m_window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;

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
        std::cerr << "[app] hover " << (m_hoverEnabled ? "enabled" : "disabled") << " (H)\n";
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

    for (const world::Chunk& chunk : m_chunkGrid.chunks()) {
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
        if (insideChunk) {
            return chunk.isSolid(localX, localY, localZ);
        }
    }

    return false;
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

    const world::Chunk& chunk = m_chunkGrid.chunks().front();

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

    const math::Vector3 rayOrigin{m_camera.x, m_camera.y, m_camera.z};
    constexpr float kRayStep = 0.05f;
    constexpr float kRayMaxDistance = 8.0f;

    bool hasPreviousEmpty = false;
    int previousX = 0;
    int previousY = 0;
    int previousZ = 0;

    for (float distance = 0.0f; distance <= kRayMaxDistance; distance += kRayStep) {
        const math::Vector3 sample = rayOrigin + (rayDirection * distance);
        const int vx = static_cast<int>(std::floor(sample.x));
        const int vy = static_cast<int>(std::floor(sample.y));
        const int vz = static_cast<int>(std::floor(sample.z));

        const bool inBounds =
            vx >= 0 && vx < world::Chunk::kSizeX &&
            vy >= 0 && vy < world::Chunk::kSizeY &&
            vz >= 0 && vz < world::Chunk::kSizeZ;
        if (!inBounds) {
            hasPreviousEmpty = false;
            continue;
        }

        if (chunk.isSolid(vx, vy, vz)) {
            result.hitSolid = true;
            result.solidX = vx;
            result.solidY = vy;
            result.solidZ = vz;
            result.hitDistance = distance;
            if (hasPreviousEmpty && !chunk.isSolid(previousX, previousY, previousZ)) {
                result.hasAdjacentEmpty = true;
                result.adjacentEmptyX = previousX;
                result.adjacentEmptyY = previousY;
                result.adjacentEmptyZ = previousZ;
            }
            return result;
        }

        hasPreviousEmpty = true;
        previousX = vx;
        previousY = vy;
        previousZ = vz;
    }

    return result;
}

bool App::tryPlaceVoxelFromCameraRay() {
    if (m_chunkGrid.chunks().empty()) {
        return false;
    }

    const CameraRaycastResult raycast = raycastFromCamera();
    if (!raycast.hitSolid || !raycast.hasAdjacentEmpty || raycast.hitDistance > kBlockInteractMaxDistance) {
        return false;
    }

    world::Chunk& chunk = m_chunkGrid.chunks().front();
    chunk.setVoxel(raycast.adjacentEmptyX, raycast.adjacentEmptyY, raycast.adjacentEmptyZ, world::Voxel{world::VoxelType::Solid});
    return true;
}

bool App::tryRemoveVoxelFromCameraRay() {
    if (m_chunkGrid.chunks().empty()) {
        return false;
    }

    const CameraRaycastResult raycast = raycastFromCamera();
    if (!raycast.hitSolid || raycast.hitDistance > kBlockInteractMaxDistance) {
        return false;
    }

    world::Chunk& chunk = m_chunkGrid.chunks().front();
    chunk.setVoxel(raycast.solidX, raycast.solidY, raycast.solidZ, world::Voxel{world::VoxelType::Empty});
    return true;
}

} // namespace app
