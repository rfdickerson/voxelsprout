#include "import/bethesda/cell_builder.h"
#include "import/imported_scene.h"
#include "render/renderer.h"

#include <GLFW/glfw3.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <cmath>
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

odai::importer::ImportedScene makeMorrowindTerrainScene() {
    using namespace odai::importer::bethesda;
    const auto root = std::filesystem::temp_directory_path() / "odai_terrain_texture_smoke";
    std::filesystem::create_directories(root / "textures");
    const auto writeDds = [&](const char* name, std::uint32_t rgba) {
        std::vector<std::uint8_t> dds(132u, 0u);
        const auto word = [&](std::size_t at, std::uint32_t value) {
            std::memcpy(dds.data() + at, &value, 4u);
        };
        word(0, 0x20534444u); word(4, 124u); word(8, 0x100fu);
        word(12, 1u); word(16, 1u); word(20, 4u); word(28, 1u);
        word(76, 32u); word(80, 0x41u); word(88, 32u);
        word(92, 0xff0000u); word(96, 0xff00u); word(100, 0xffu);
        word(104, 0xff000000u); word(108, 0x1000u); word(128, rgba);
        std::ofstream out(root / "textures" / name, std::ios::binary);
        out.write(reinterpret_cast<const char*>(dds.data()), static_cast<std::streamsize>(dds.size()));
    };
    writeDds("_land_default.dds", 0xff406080u);
    writeDds("ground.dds", 0xff208020u);
    writeDds("rock.dds", 0xff808080u);
    FalloutCellRecord cell;
    cell.hasGridCoords = true;
    cell.hasWater = true;
    cell.waterHeight = 200.0f;
    cell.land = std::make_unique<FalloutLandRecord>();
    cell.land->gridSize = kMorrowindLandGridSize;
    cell.land->hasHeights = true;
    cell.land->morrowindTextureGrid.resize(256u);
    for (int row = 0; row < 16; ++row)
        for (int col = 0; col < 16; ++col)
            cell.land->morrowindTextureGrid[static_cast<std::size_t>(row * 16 + col)] =
                col < 5 ? 1u : col < 11 ? 2u : 0u;
    cell.land->heights.resize(65u * 65u);
    for (int row = 0; row < 65; ++row)
        for (int col = 0; col < 65; ++col)
            cell.land->heights[static_cast<std::size_t>(row * 65 + col)] =
                80.0f + static_cast<float>(row * 2 + col);
    FalloutAssetSource assets;
    if (!assets.open(root)) return {};
    FalloutWorldTables tables;
    tables.morrowind = true;
    tables.morrowindLandTexturePaths.emplace(1u, "ground.dds");
    tables.morrowindLandTexturePaths.emplace(2u, "rock.dds");
    CellSceneBuilder builder(assets, tables);
    builder.addCellTerrain(cell);
    odai::importer::ImportedScene scene;
    builder.finish(scene);
    std::error_code cleanupError;
    std::filesystem::remove_all(root, cleanupError);
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
        } else if (!renderer.waterNormalAssetReady()) {
            std::cerr << "Packaged water normal asset was not loaded\n";
        } else if (!renderer.uploadImportedScene(makeSyntheticScene())) {
            std::cerr << "synthetic ImportedScene upload failed\n";
        } else {
            odai::render::CameraPose camera{};
            camera.z = 4.0f;
            camera.yawDegrees = -90.0f;
            camera.fovDegrees = 60.0f;
            bool streamingValid = renderer.waitForImportedSceneUploads() &&
                renderer.isImportedSceneChunkReady(0);
            bool timingValid = true;
            std::vector<std::uint64_t> submissions;
            std::uint64_t lastAttempt = 0, lastGpuSample = 0;
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
                const auto timing = renderer.framePacingStats();
                timingValid = timingValid && timing.renderAttempt > lastAttempt;
                lastAttempt = timing.renderAttempt;
                for (const float wait : {timing.cpuWaitFrameSlotMs, timing.cpuWaitAcquireMs,
                                        timing.cpuWaitPresentMs, timing.cpuWaitTransferMs}) {
                    timingValid = timingValid && std::isfinite(wait) && wait >= 0;
                }
                if (renderer.benchmarkGpuSampleSerial() != lastGpuSample) {
                    lastGpuSample = renderer.benchmarkGpuSampleSerial();
                    timingValid = timingValid && std::find(submissions.begin(), submissions.end(),
                        renderer.benchmarkGpuSubmissionId()) != submissions.end();
                }
                if (timing.submissionId) submissions.push_back(timing.submissionId);
                timingValid = timingValid && (!timing.presentAccepted || timing.submissionId != 0);
            }
            std::vector<std::uint8_t> rgb;
            std::uint32_t width = 0;
            std::uint32_t height = 0;
            passed = timingValid && streamingValid && renderer.captureFrameRgb(rgb, width, height) && width > 0 && height > 0 &&
                rgb.size() == static_cast<std::size_t>(width) * height * 3u &&
                std::any_of(rgb.begin(), rgb.end(), [](std::uint8_t value) { return value != 0; });
            if (!passed) {
                std::cerr << "rendered frame capture was empty or invalid\n";
            }
            if (passed) {
                auto terrain = makeMorrowindTerrainScene();
                const bool materialIndicesValid = std::all_of(
                    terrain.packedVertices.begin(), terrain.packedVertices.end(),
                    [&](const odai::importer::ImportedScenePackedVertex& vertex) {
                        if (vertex.textureIndex >= terrain.textures.size()) return false;
                        for (const std::uint32_t layer : vertex.layerTextureIndex) {
                            if (layer != odai::importer::kImportedSceneNoTerrainLayer &&
                                layer >= terrain.textures.size()) return false;
                        }
                        return true;
                    });
                if (terrain.textures.size() != 3u || terrain.packedDraws.size() < 3u ||
                    terrain.sourceLandscapeCellCount == 0u || !materialIndicesValid ||
                    !renderer.uploadImportedScene(terrain) ||
                    !renderer.waitForImportedSceneUploads()) {
                    std::cerr << "Morrowind LAND terrain draw was not uploaded\n";
                    passed = false;
                } else {
                    camera = {};
                    camera.x = 4096.0f;
                    camera.y = 1200.0f;
                    camera.z = 600.0f;
                    camera.yawDegrees = -90.0f;
                    camera.pitchDegrees = -30.0f;
                    camera.fovDegrees = 60.0f;
                    for (int frame = 0; frame < 8; ++frame) {
                        glfwPollEvents();
                        renderer.renderFrame(camera);
                    }
                    // Timestamp readback trails submission. Drain until the
                    // measured frame actually contains the water fixture.
                    for (int frame = 0; frame < 24 &&
                         renderer.benchmarkWaterGpuMs() <= 0.0f; ++frame) {
                        renderer.renderFrame(camera);
                    }
                    if (!renderer.prepareFrameCapture()) {
                        passed = false;
                    }
                    if (passed) renderer.renderFrame(camera);
                    rgb.clear();
                    passed = passed && renderer.captureFrameRgb(rgb, width, height) &&
                        rgb.size() == static_cast<std::size_t>(width) * height * 3u &&
                        std::any_of(rgb.begin(), rgb.end(), [](std::uint8_t value) { return value != 0; });
                    if (!passed) std::cerr << "Morrowind LAND terrain frame capture failed\n";
                    if (passed) {
                        const auto waterDraws = renderer.benchmarkWaterDrawCalls();
                        const auto waterCpuMs = renderer.benchmarkWaterCpuRecordMs();
                        const auto waterGpuMs = renderer.benchmarkWaterGpuMs();
                        const auto waterBytes = renderer.benchmarkWaterGeometryBytes();
                        std::cerr << "water baseline: draws=" << waterDraws
                                  << " cpuRecordMs=" << waterCpuMs
                                  << " gpuDrawMs=" << waterGpuMs
                                  << " gpuFrameMs=" << renderer.benchmarkGpuFrameMs()
                                  << " gpuSample=" << renderer.benchmarkGpuSampleSerial()
                                  << " geometryBytes=" << waterBytes << '\n';
                        passed = waterDraws == 1u && waterBytes > 0u &&
                            std::isfinite(waterCpuMs) && waterCpuMs >= 0.0f &&
                            std::isfinite(waterGpuMs) && waterGpuMs > 0.0f;
                    }
                    if (passed) {
                        const auto withWater = rgb;
                        renderer.setVisualTimeSeconds(20.0f);
                        passed = renderer.prepareFrameCapture();
                        if (passed) {
                            renderer.renderFrame(camera);
                            rgb.clear();
                            passed = renderer.captureFrameRgb(rgb, width, height) &&
                                rgb.size() == withWater.size() && rgb != withWater;
                        }
                        if (!passed) std::cerr << "Water animation did not change the rendered surface\n";
                        if (passed) {
                            const auto animatedWater = rgb;
                            if (const char* capturePath = std::getenv("ODAI_WATER_SMOKE_CAPTURE")) {
                                renderer.captureFrameToFile(capturePath);
                            }
                            const auto drawsWithWater = renderer.benchmarkDrawCalls();
                            renderer.setWaterRenderingEnabled(false);
                            passed = !renderer.waterRenderingEnabled() && renderer.prepareFrameCapture();
                            if (passed) {
                                renderer.renderFrame(camera);
                                rgb.clear();
                                passed = renderer.captureFrameRgb(rgb, width, height) &&
                                    renderer.benchmarkDrawCalls() < drawsWithWater &&
                                    rgb.size() == animatedWater.size() && rgb != animatedWater;
                                if (passed) {
                                    const auto pixel = [&](const std::vector<std::uint8_t>& image,
                                                           std::uint32_t x, std::uint32_t y) {
                                        return image.data() + (static_cast<std::size_t>(y) * width + x) * 3u;
                                    };
                                    // The crest at the upper left is above the
                                    // authored water plane; the centre is a
                                    // submerged part of the same LAND fixture.
                                    const auto* crestOn = pixel(animatedWater, width * 90u / 320u,
                                                                 height * 22u / 180u);
                                    const auto* crestOff = pixel(rgb, width * 90u / 320u,
                                                                  height * 22u / 180u);
                                    const auto* bottomOn = pixel(animatedWater, width / 2u, height / 2u);
                                    const auto* bottomOff = pixel(rgb, width / 2u, height / 2u);
                                    const auto difference = [](const std::uint8_t* a, const std::uint8_t* b) {
                                        return std::abs(int(a[0]) - int(b[0])) +
                                            std::abs(int(a[1]) - int(b[1])) +
                                            std::abs(int(a[2]) - int(b[2]));
                                    };
                                    passed = crestOff[1] > crestOff[0] &&
                                        crestOff[1] > crestOff[2] &&
                                        difference(crestOn, crestOff) <= 8 &&
                                        difference(bottomOn, bottomOff) > 20;
                                }
                                if (passed) {
                                    if (const char* capturePath = std::getenv("ODAI_WATER_SMOKE_CAPTURE")) {
                                        renderer.captureFrameToFile(std::string(capturePath) + ".off.ppm");
                                    }
                                }
                            }
                            if (!passed) std::cerr << "Water disable did not remove its draw and image contribution\n";
                            renderer.setWaterRenderingEnabled(true);
                            if (passed) {
                                passed = renderer.waterRenderingEnabled() &&
                                    renderer.prepareFrameCapture();
                                if (passed) {
                                    renderer.renderFrame(camera);
                                    rgb.clear();
                                    passed = renderer.captureFrameRgb(rgb, width, height);
                                    if (passed) {
                                        if (const char* capturePath = std::getenv("ODAI_WATER_SMOKE_CAPTURE")) {
                                            renderer.captureFrameToFile(std::string(capturePath) + ".repeat.ppm");
                                        }
                                        passed = rgb.size() == animatedWater.size() &&
                                            std::equal(rgb.begin(), rgb.end(), animatedWater.begin(),
                                                [](std::uint8_t a, std::uint8_t b) {
                                                    return std::abs(int(a) - int(b)) <= 1;
                                                });
                                    }
                                }
                                if (!passed) std::cerr << "Repeating water time did not reproduce the image\n";
                            }
                        }
                    }
                }
            }
        }
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    return passed ? 0 : 1;
}
