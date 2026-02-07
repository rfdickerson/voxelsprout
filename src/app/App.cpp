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
    const render::CameraPose cameraPose{
        m_camera.x,
        m_camera.y,
        m_camera.z,
        m_camera.yawDegrees,
        m_camera.pitchDegrees,
        m_camera.fovDegrees
    };
    m_renderer.renderFrame(m_chunkGrid, m_simulation, cameraPose);
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

} // namespace app
