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
        result.flags=odai::importer::kImportedSceneMaterialFlagAlphaBlend;
        return result;
    };
    scene.packedVertices = {
        vertex(-1.5f, -1.0f, 0.0f, 0.0f, 1.0f),
        vertex(1.5f, -1.0f, 0.0f, 1.0f, 1.0f),
        vertex(0.0f, 1.5f, 0.0f, 0.5f, 0.0f),
    };
    auto frameTexture=scene.textures[0];
    frameTexture.sourcePath="synthetic/animation_frame";
    for(std::size_t i=0;i<frameTexture.rgba8.size();i+=4) std::swap(frameTexture.rgba8[i],frameTexture.rgba8[i+1]);
    scene.textures.push_back(std::move(frameTexture));
    odai::importer::ImportedSceneTexture hdrCube;
    hdrCube.sourcePath = "synthetic/bc6h_cube";
    hdrCube.width = hdrCube.height = 4;
    hdrCube.arrayLayers = 6;
    hdrCube.format = odai::importer::TextureFormat::BC6HUfloat;
    hdrCube.rgba8.resize(6 * 16, 0); // zero-valued HDR blocks
    scene.textures.push_back(std::move(hdrCube));
    odai::importer::ImportedNifLightingMaterial material;
    material.valid=1;material.shaderType=0xffffffffu;material.emissiveMultiplier=1;
    material.emissive[0]=material.emissive[1]=material.emissive[2]=1;
    odai::importer::MaterialAnimationTrack track;
    track.stop=2;track.keys={{0,0},{2,1}};
    material.animations.push_back(track);
    track.target=odai::importer::MaterialAnimatedValue::Alpha;
    track.keys={{0,.25f},{2,1}};material.animations.push_back(track);
    track.target=odai::importer::MaterialAnimatedValue::DiffuseFrame;
    track.keys={{0,0},{1,1},{2,0}};track.interpolation=5;track.textures={0,1};
    material.animations.push_back(track);
    scene.lightingMaterials.push_back(material);
    for(int i=0;i<3;++i) {
        auto cutout=scene.packedVertices[i];cutout.position[0]+=1;
        cutout.flags=odai::importer::kImportedSceneMaterialFlagAlphaTest;
        scene.packedVertices.push_back(cutout);
    }
    scene.packedLightingMaterialIndices={0,0,0,0,0,0};
    scene.packedIndices = {0, 1, 2, 3, 4, 5};
    scene.packedDraws.push_back(odai::importer::ImportedScenePackedDraw{0, 3});
    scene.packedDraws.push_back(odai::importer::ImportedScenePackedDraw{3, 3});
    scene.boundsMin[0] = -1.5f;
    scene.boundsMin[1] = -1.0f;
    scene.boundsMax[0] = 2.5f;
    scene.boundsMax[1] = 1.5f;
    odai::importer::ImportedSceneParticleEmitter emitter;
    emitter.sourceId="synthetic_mist";emitter.effect=odai::importer::ImportedParticleEffect::Mist;
    emitter.textureIndex=0;emitter.seed=42;emitter.mist.emplace();
    auto& mist=*emitter.mist;mist.nodes.resize(1);mist.emitterNode=mist.gravityNode=0;
    mist.rate=6;mist.life=1;mist.radius=.3f;mist.speed=.2f;mist.softDepth=.1f;
    mist.box={.2f,.2f,.2f};mist.scales={1,2};
    mist.colorTimes={.1f,.9f,.1f,.3f,.7f,1};mist.colors={1,1,1,0,1,1,1,.5f,1,1,1,0};
    scene.particleEmitters.push_back(emitter);
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
            camera.yawDegrees = -90.0f;
            camera.fovDegrees = 60.0f;
            bool streamingValid = renderer.waitForImportedSceneUploads() &&
                renderer.isImportedSceneChunkReady(0);
            for (int frame = 0; frame < 8; ++frame) {
                renderer.setVisualTimeSeconds(float(frame)*.25f);
                camera.x=float(frame)*.025f;
                if (frame == 1) renderer.setImportedSceneChunkLodTransition(0, 0.25f, 1.0f, 42.0f);
                if (frame == 2) renderer.setImportedSceneChunkLodTransition(0, 0.75f, -1.0f, 42.0f);
                if (frame == 3) {
                    renderer.removeImportedSceneChunk(0);
                    streamingValid = streamingValid && !renderer.isImportedSceneChunkReady(0);
                    renderer.setImportedSceneChunkLodTransition(0, 0.5f, 1.0f, 42.0f);
                }
                if(frame==4 && renderer.addImportedSceneChunk(makeSyntheticScene())==odai::render::Renderer::kInvalidImportedChunkIndex) {
                    std::cerr<<"animated material/mist streaming reload failed\n";break;
                }
                if(frame==7 && !renderer.prepareFrameCapture())break;
                glfwPollEvents();
                renderer.renderFrame(camera);
            }
            std::vector<std::uint8_t> rgb;
            std::uint32_t width = 0;
            std::uint32_t height = 0;
            passed = streamingValid && renderer.captureFrameRgb(rgb, width, height) && width > 0 && height > 0 &&
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
