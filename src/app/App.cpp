#include "app/App.hpp"

#include <GLFW/glfw3.h>

#include "math/Math.hpp"

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <iostream>

namespace {

constexpr float kMouseSensitivity = 0.1f;
constexpr float kMouseSmoothingSeconds = 0.035f;
constexpr float kMoveMaxSpeed = 5.0f;
constexpr float kMoveAcceleration = 14.0f;
constexpr float kMoveDeceleration = 18.0f;
constexpr float kPitchMinDegrees = -89.0f;
constexpr float kPitchMaxDegrees = 89.0f;

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

    const bool placePressedThisFrame = m_input.placeBlockDown && !m_wasPlaceBlockDown;
    const bool removePressedThisFrame = m_input.removeBlockDown && !m_wasRemoveBlockDown;
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
    if (raycast.hitSolid) {
        const bool showRemovePreview = m_input.removeBlockDown || !raycast.hasAdjacentEmpty;
        if (showRemovePreview) {
            preview.visible = true;
            preview.mode = render::VoxelPreview::Mode::Remove;
            preview.x = raycast.solidX;
            preview.y = raycast.solidY;
            preview.z = raycast.solidZ;
        } else {
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

    m_input.quitRequested = glfwGetKey(m_window, GLFW_KEY_ESCAPE) == GLFW_PRESS;
    m_input.moveForward = glfwGetKey(m_window, GLFW_KEY_W) == GLFW_PRESS;
    m_input.moveBackward = glfwGetKey(m_window, GLFW_KEY_S) == GLFW_PRESS;
    m_input.moveLeft = glfwGetKey(m_window, GLFW_KEY_A) == GLFW_PRESS;
    m_input.moveRight = glfwGetKey(m_window, GLFW_KEY_D) == GLFW_PRESS;
    m_input.moveUp = glfwGetKey(m_window, GLFW_KEY_SPACE) == GLFW_PRESS;
    m_input.moveDown =
        glfwGetKey(m_window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
        glfwGetKey(m_window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS;
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

    m_lastMouseX = mouseX;
    m_lastMouseY = mouseY;
}

void App::updateCamera(float dt) {
    const float mouseSmoothingAlpha = 1.0f - std::exp(-dt / kMouseSmoothingSeconds);
    m_camera.smoothedMouseDeltaX += (m_input.mouseDeltaX - m_camera.smoothedMouseDeltaX) * mouseSmoothingAlpha;
    m_camera.smoothedMouseDeltaY += (m_input.mouseDeltaY - m_camera.smoothedMouseDeltaY) * mouseSmoothingAlpha;

    m_camera.yawDegrees += m_camera.smoothedMouseDeltaX * kMouseSensitivity;
    m_camera.pitchDegrees += m_camera.smoothedMouseDeltaY * kMouseSensitivity;
    m_camera.pitchDegrees = std::clamp(m_camera.pitchDegrees, kPitchMinDegrees, kPitchMaxDegrees);

    const float yawRadians = math::radians(m_camera.yawDegrees);
    const math::Vector3 forward{std::cos(yawRadians), 0.0f, std::sin(yawRadians)};
    const math::Vector3 right{-forward.z, 0.0f, forward.x};
    const math::Vector3 up{0.0f, 1.0f, 0.0f};
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
    if (m_input.moveUp) {
        moveDirection += up;
    }
    if (m_input.moveDown) {
        moveDirection -= up;
    }

    const float moveLengthSq = math::lengthSquared(moveDirection);
    const float moveLength = std::sqrt(moveLengthSq);
    float targetVelocityX = 0.0f;
    float targetVelocityY = 0.0f;
    float targetVelocityZ = 0.0f;
    if (moveLength > 0.0f) {
        moveDirection /= moveLength;
        const math::Vector3 targetVelocity = moveDirection * kMoveMaxSpeed;
        targetVelocityX = targetVelocity.x;
        targetVelocityY = targetVelocity.y;
        targetVelocityZ = targetVelocity.z;
    }

    const float accelPerFrame = kMoveAcceleration * dt;
    const float decelPerFrame = kMoveDeceleration * dt;

    const float maxDeltaX = (std::fabs(targetVelocityX) > std::fabs(m_camera.velocityX)) ? accelPerFrame : decelPerFrame;
    const float maxDeltaY = (std::fabs(targetVelocityY) > std::fabs(m_camera.velocityY)) ? accelPerFrame : decelPerFrame;
    const float maxDeltaZ = (std::fabs(targetVelocityZ) > std::fabs(m_camera.velocityZ)) ? accelPerFrame : decelPerFrame;

    m_camera.velocityX = approach(m_camera.velocityX, targetVelocityX, maxDeltaX);
    m_camera.velocityY = approach(m_camera.velocityY, targetVelocityY, maxDeltaY);
    m_camera.velocityZ = approach(m_camera.velocityZ, targetVelocityZ, maxDeltaZ);

    m_camera.x += m_camera.velocityX * dt;
    m_camera.y += m_camera.velocityY * dt;
    m_camera.z += m_camera.velocityZ * dt;
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
    if (!raycast.hitSolid || !raycast.hasAdjacentEmpty) {
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
    if (!raycast.hitSolid) {
        return false;
    }

    world::Chunk& chunk = m_chunkGrid.chunks().front();
    chunk.setVoxel(raycast.solidX, raycast.solidY, raycast.solidZ, world::Voxel{world::VoxelType::Empty});
    return true;
}

} // namespace app
