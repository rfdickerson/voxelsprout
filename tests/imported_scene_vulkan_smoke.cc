#include "import/imported_scene.h"
#include "render/renderer.h"

#include <GLFW/glfw3.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

odai::importer::ImportedScene makeSyntheticScene() {
    odai::importer::ImportedScene scene;
    scene.sourceTag = "synthetic_exterior";

    odai::importer::ImportedSceneTexture texture;
    texture.sourcePath = "synthetic/checker";
    texture.width = 2;
    texture.height = 2;
    texture.rgba8 = {
        220, 72, 48, 255, 48, 160, 220, 255,
        48, 160, 220, 255, 220, 72, 48, 255,
    };
    scene.textures.push_back(std::move(texture));

    const auto vertex = [](float x, float y, float z, float u, float v) {
        odai::importer::ImportedScenePackedVertex result{};
        result.position[0] = x;
        result.position[1] = y;
        result.position[2] = z;
        result.normal[2] = 1.0f;
        result.color[0] = 1.0f;
        result.color[1] = 1.0f;
        result.color[2] = 1.0f;
        result.uv[0] = u;
        result.uv[1] = v;
        result.textureIndex = 0;
        return result;
    };
    scene.packedVertices = {
        vertex(-1.5f, -1.0f, 0.0f, 0.0f, 1.0f),
        vertex(1.5f, -1.0f, 0.0f, 1.0f, 1.0f),
        vertex(0.0f, 1.5f, 0.0f, 0.5f, 0.0f),
    };
    scene.packedIndices = {0, 1, 2};
    scene.packedDraws.push_back(odai::importer::ImportedScenePackedDraw{0, 3});
    scene.boundsMin[0] = -1.5f;
    scene.boundsMin[1] = -1.0f;
    scene.boundsMax[0] = 1.5f;
    scene.boundsMax[1] = 1.5f;
    return scene;
}

}  // namespace

int main() {
#if defined(__linux__)
    // CI supplies an Xvfb display; prefer it even when a stale Wayland
    // environment variable is inherited by the test process.
    glfwInitHint(GLFW_PLATFORM, GLFW_PLATFORM_X11);
#endif
    if (glfwInit() != GLFW_TRUE) {
        std::cerr << "GLFW initialization failed\n";
        return 1;
    }
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    GLFWwindow* window = glfwCreateWindow(320, 180, "odai imported-scene smoke", nullptr, nullptr);
    if (window == nullptr) {
        std::cerr << "hidden GLFW window creation failed\n";
        glfwTerminate();
        return 1;
    }

    bool passed = false;
    {
        odai::render::Renderer renderer;
        renderer.setMsaaSamples(1);
        if (!renderer.init(window)) {
            std::cerr << "Vulkan renderer initialization failed\n";
        } else if (!renderer.uploadImportedScene(makeSyntheticScene())) {
            std::cerr << "synthetic ImportedScene upload failed\n";
        } else {
            odai::render::CameraPose camera{};
            camera.z = 4.0f;
            camera.yawDegrees = 180.0f;
            camera.fovDegrees = 60.0f;
            for (int frame = 0; frame < 4; ++frame) {
                glfwPollEvents();
                renderer.renderFrame(camera);
            }
            std::vector<std::uint8_t> rgb;
            std::uint32_t width = 0;
            std::uint32_t height = 0;
            passed = renderer.captureFrameRgb(rgb, width, height) && width > 0 && height > 0 &&
                rgb.size() == static_cast<std::size_t>(width) * height * 3u &&
                std::any_of(rgb.begin(), rgb.end(), [](std::uint8_t value) { return value != 0; });
            if (!passed) {
                std::cerr << "rendered frame capture was empty or invalid\n";
            }
        }
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    return passed ? 0 : 1;
}
