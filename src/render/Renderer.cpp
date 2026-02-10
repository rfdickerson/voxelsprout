#include "render/Renderer.hpp"

#include <GLFW/glfw3.h>
#include "core/Grid3.hpp"
#include "core/Log.hpp"
#include "math/Math.hpp"
#include "sim/NetworkProcedural.hpp"
#include "world/ChunkMesher.hpp"

#if defined(VOXEL_HAS_IMGUI)
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace render {

namespace {

constexpr std::array<const char*, 1> kValidationLayers = {"VK_LAYER_KHRONOS_validation"};
constexpr std::array<const char*, 7> kDeviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    VK_KHR_MAINTENANCE_4_EXTENSION_NAME,
    VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME,
    VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME,
    VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME,
    VK_EXT_MEMORY_BUDGET_EXTENSION_NAME,
    VK_EXT_MEMORY_PRIORITY_EXTENSION_NAME,
};
constexpr uint32_t kBindlessTargetTextureCapacity = 1024;
constexpr uint32_t kBindlessMinTextureCapacity = 64;
constexpr uint32_t kBindlessReservedSampledDescriptors = 16;
constexpr uint32_t kBindlessTextureIndexDiffuse = 0;
constexpr uint32_t kBindlessTextureIndexHdrResolved = 1;
constexpr uint32_t kBindlessTextureIndexShadowAtlas = 2;
constexpr uint32_t kBindlessTextureIndexNormalDepth = 3;
constexpr uint32_t kBindlessTextureIndexSsaoBlur = 4;
constexpr uint32_t kBindlessTextureIndexSsaoRaw = 5;
constexpr uint32_t kBindlessTextureStaticCount = 6;
constexpr uint32_t kShadowCascadeCount = 4;
constexpr std::array<uint32_t, kShadowCascadeCount> kShadowCascadeResolution = {4096u, 2048u, 2048u, 1024u};
struct ShadowAtlasRect {
    uint32_t x;
    uint32_t y;
    uint32_t size;
};
constexpr std::array<ShadowAtlasRect, kShadowCascadeCount> kShadowAtlasRects = {
    ShadowAtlasRect{0u, 0u, 4096u},
    ShadowAtlasRect{4096u, 0u, 2048u},
    ShadowAtlasRect{6144u, 0u, 2048u},
    ShadowAtlasRect{4096u, 2048u, 1024u}
};
constexpr uint32_t kShadowAtlasSize = 8192u;
constexpr float kPipeTransferHalfExtent = 0.58f;
constexpr float kPipeMinRadius = 0.02f;
constexpr float kPipeMaxRadius = 0.5f;
constexpr float kPipeBranchRadiusBoost = 0.05f;
constexpr float kPipeMaxEndExtension = 0.49f;
constexpr float kBeltRadius = 0.49f;
constexpr float kTrackRadius = 0.38f;
constexpr math::Vector3 kBeltTint{0.78f, 0.62f, 0.18f};
constexpr math::Vector3 kTrackTint{0.52f, 0.54f, 0.58f};

#if defined(VOXEL_HAS_IMGUI)
void imguiCheckVkResult(VkResult result) {
    if (result != VK_SUCCESS) {
        VOX_LOGE("imgui") << "Vulkan backend error: " << static_cast<int>(result);
    }
}
#endif

struct alignas(16) CameraUniform {
    float mvp[16];
    float view[16];
    float proj[16];
    float lightViewProj[kShadowCascadeCount][16];
    float shadowCascadeSplits[4];
    float shadowAtlasUvRects[kShadowCascadeCount][4];
    float sunDirectionIntensity[4];
    float sunColorShadow[4];
    float shIrradiance[9][4];
    float shadowConfig0[4];
    float shadowConfig1[4];
    float shadowConfig2[4];
    float shadowConfig3[4];
    float shadowVoxelGridOrigin[4];
    float shadowVoxelGridSize[4];
    float skyConfig0[4];
    float skyConfig1[4];
};

struct alignas(16) ChunkPushConstants {
    float chunkOffset[4];
    float cascadeData[4];
};

struct alignas(16) ChunkInstanceData {
    float chunkOffset[4];
};

struct PipeMeshData {
    struct Vertex {
        float position[3];
        float normal[3];
    };
    std::vector<Vertex> vertices;
    std::vector<std::uint32_t> indices;
};

world::ChunkMeshData buildSingleVoxelPreviewMesh(
    std::uint32_t x,
    std::uint32_t y,
    std::uint32_t z,
    std::uint32_t ao,
    std::uint32_t material
) {
    world::ChunkMeshData mesh{};
    mesh.vertices.reserve(24);
    mesh.indices.reserve(36);

    for (std::uint32_t faceId = 0; faceId < 6; ++faceId) {
        const std::uint32_t baseVertex = static_cast<std::uint32_t>(mesh.vertices.size());
        for (std::uint32_t corner = 0; corner < 4; ++corner) {
            world::PackedVoxelVertex vertex{};
            vertex.bits = world::PackedVoxelVertex::pack(x, y, z, faceId, corner, ao, material, 2u);
            mesh.vertices.push_back(vertex);
        }

        mesh.indices.push_back(baseVertex + 0);
        mesh.indices.push_back(baseVertex + 1);
        mesh.indices.push_back(baseVertex + 2);
        mesh.indices.push_back(baseVertex + 0);
        mesh.indices.push_back(baseVertex + 2);
        mesh.indices.push_back(baseVertex + 3);
    }

    return mesh;
}

void appendBoxMesh(
    PipeMeshData& mesh,
    float minX,
    float minY,
    float minZ,
    float maxX,
    float maxY,
    float maxZ
) {
    auto appendFace = [&mesh](
                          const std::array<std::array<float, 3>, 4>& corners,
                          const std::array<float, 3>& normal
                      ) {
        const std::uint32_t base = static_cast<std::uint32_t>(mesh.vertices.size());
        for (const std::array<float, 3>& corner : corners) {
            PipeMeshData::Vertex vertex{};
            vertex.position[0] = corner[0];
            vertex.position[1] = corner[1];
            vertex.position[2] = corner[2];
            vertex.normal[0] = normal[0];
            vertex.normal[1] = normal[1];
            vertex.normal[2] = normal[2];
            mesh.vertices.push_back(vertex);
        }
        mesh.indices.push_back(base + 0u);
        mesh.indices.push_back(base + 1u);
        mesh.indices.push_back(base + 2u);
        mesh.indices.push_back(base + 0u);
        mesh.indices.push_back(base + 2u);
        mesh.indices.push_back(base + 3u);
    };

    appendFace(
        {{
            {{maxX, minY, minZ}},
            {{maxX, maxY, minZ}},
            {{maxX, maxY, maxZ}},
            {{maxX, minY, maxZ}},
        }},
        {{1.0f, 0.0f, 0.0f}}
    );
    appendFace(
        {{
            {{minX, minY, maxZ}},
            {{minX, maxY, maxZ}},
            {{minX, maxY, minZ}},
            {{minX, minY, minZ}},
        }},
        {{-1.0f, 0.0f, 0.0f}}
    );
    appendFace(
        {{
            {{minX, maxY, minZ}},
            {{minX, maxY, maxZ}},
            {{maxX, maxY, maxZ}},
            {{maxX, maxY, minZ}},
        }},
        {{0.0f, 1.0f, 0.0f}}
    );
    appendFace(
        {{
            {{minX, minY, maxZ}},
            {{minX, minY, minZ}},
            {{maxX, minY, minZ}},
            {{maxX, minY, maxZ}},
        }},
        {{0.0f, -1.0f, 0.0f}}
    );
    appendFace(
        {{
            {{minX, minY, maxZ}},
            {{maxX, minY, maxZ}},
            {{maxX, maxY, maxZ}},
            {{minX, maxY, maxZ}},
        }},
        {{0.0f, 0.0f, 1.0f}}
    );
    appendFace(
        {{
            {{maxX, minY, minZ}},
            {{minX, minY, minZ}},
            {{minX, maxY, minZ}},
            {{maxX, maxY, minZ}},
        }},
        {{0.0f, 0.0f, -1.0f}}
    );
}

PipeMeshData buildTransportBoxMesh() {
    PipeMeshData mesh{};
    mesh.vertices.reserve(24u);
    mesh.indices.reserve(36u);
    appendBoxMesh(
        mesh,
        -kPipeTransferHalfExtent,
        0.0f,
        -kPipeTransferHalfExtent,
        kPipeTransferHalfExtent,
        1.0f,
        kPipeTransferHalfExtent
    );
    return mesh;
}

PipeMeshData buildPipeCylinderMesh() {
    PipeMeshData mesh{};
    constexpr std::uint32_t kSegments = 16u;
    mesh.vertices.reserve(static_cast<std::size_t>(kSegments * 4u + 2u));
    mesh.indices.reserve(static_cast<std::size_t>(kSegments * 12u));

    const float radius = kPipeTransferHalfExtent;
    const float twoPi = 6.28318530718f;

    for (std::uint32_t i = 0; i < kSegments; ++i) {
        const float t0 = (static_cast<float>(i) / static_cast<float>(kSegments)) * twoPi;
        const float t1 = (static_cast<float>(i + 1u) / static_cast<float>(kSegments)) * twoPi;
        const float x0 = std::cos(t0) * radius;
        const float z0 = std::sin(t0) * radius;
        const float x1 = std::cos(t1) * radius;
        const float z1 = std::sin(t1) * radius;

        // Side quad
        const std::uint32_t sideBase = static_cast<std::uint32_t>(mesh.vertices.size());
        PipeMeshData::Vertex v0{{x0, 0.0f, z0}, {std::cos(t0), 0.0f, std::sin(t0)}};
        PipeMeshData::Vertex v1{{x0, 1.0f, z0}, {std::cos(t0), 0.0f, std::sin(t0)}};
        PipeMeshData::Vertex v2{{x1, 1.0f, z1}, {std::cos(t1), 0.0f, std::sin(t1)}};
        PipeMeshData::Vertex v3{{x1, 0.0f, z1}, {std::cos(t1), 0.0f, std::sin(t1)}};
        mesh.vertices.push_back(v0);
        mesh.vertices.push_back(v1);
        mesh.vertices.push_back(v2);
        mesh.vertices.push_back(v3);
        mesh.indices.push_back(sideBase + 0u);
        mesh.indices.push_back(sideBase + 1u);
        mesh.indices.push_back(sideBase + 2u);
        mesh.indices.push_back(sideBase + 0u);
        mesh.indices.push_back(sideBase + 2u);
        mesh.indices.push_back(sideBase + 3u);
    }

    const std::uint32_t bottomCenter = static_cast<std::uint32_t>(mesh.vertices.size());
    mesh.vertices.push_back(PipeMeshData::Vertex{{0.0f, 0.0f, 0.0f}, {0.0f, -1.0f, 0.0f}});
    const std::uint32_t topCenter = static_cast<std::uint32_t>(mesh.vertices.size());
    mesh.vertices.push_back(PipeMeshData::Vertex{{0.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 0.0f}});

    for (std::uint32_t i = 0; i < kSegments; ++i) {
        const float t0 = (static_cast<float>(i) / static_cast<float>(kSegments)) * twoPi;
        const float t1 = (static_cast<float>(i + 1u) / static_cast<float>(kSegments)) * twoPi;
        const float x0 = std::cos(t0) * radius;
        const float z0 = std::sin(t0) * radius;
        const float x1 = std::cos(t1) * radius;
        const float z1 = std::sin(t1) * radius;

        const std::uint32_t bottomBase = static_cast<std::uint32_t>(mesh.vertices.size());
        mesh.vertices.push_back(PipeMeshData::Vertex{{x0, 0.0f, z0}, {0.0f, -1.0f, 0.0f}});
        mesh.vertices.push_back(PipeMeshData::Vertex{{x1, 0.0f, z1}, {0.0f, -1.0f, 0.0f}});
        mesh.indices.push_back(bottomCenter);
        mesh.indices.push_back(bottomBase + 1u);
        mesh.indices.push_back(bottomBase + 0u);

        const std::uint32_t topBase = static_cast<std::uint32_t>(mesh.vertices.size());
        mesh.vertices.push_back(PipeMeshData::Vertex{{x0, 1.0f, z0}, {0.0f, 1.0f, 0.0f}});
        mesh.vertices.push_back(PipeMeshData::Vertex{{x1, 1.0f, z1}, {0.0f, 1.0f, 0.0f}});
        mesh.indices.push_back(topCenter);
        mesh.indices.push_back(topBase + 0u);
        mesh.indices.push_back(topBase + 1u);
    }

    return mesh;
}

struct PipeEndpointState {
    math::Vector3 axis{0.0f, 1.0f, 0.0f};
    float renderedRadius = 0.45f;
    float startExtension = 0.0f;
    float endExtension = 0.0f;
};

core::Dir6 dominantAxisDir6(const math::Vector3& direction) {
    if (math::lengthSquared(direction) <= 0.000001f) {
        return core::Dir6::PosY;
    }
    const math::Vector3 normalized = math::normalize(direction);
    const float absX = std::abs(normalized.x);
    const float absY = std::abs(normalized.y);
    const float absZ = std::abs(normalized.z);
    if (absX >= absY && absX >= absZ) {
        return normalized.x >= 0.0f ? core::Dir6::PosX : core::Dir6::NegX;
    }
    if (absY >= absX && absY >= absZ) {
        return normalized.y >= 0.0f ? core::Dir6::PosY : core::Dir6::NegY;
    }
    return normalized.z >= 0.0f ? core::Dir6::PosZ : core::Dir6::NegZ;
}

math::Vector3 beltDirectionAxis(sim::BeltDirection direction) {
    switch (direction) {
    case sim::BeltDirection::East:
        return math::Vector3{1.0f, 0.0f, 0.0f};
    case sim::BeltDirection::West:
        return math::Vector3{-1.0f, 0.0f, 0.0f};
    case sim::BeltDirection::South:
        return math::Vector3{0.0f, 0.0f, 1.0f};
    case sim::BeltDirection::North:
    default:
        return math::Vector3{0.0f, 0.0f, -1.0f};
    }
}

math::Vector3 trackDirectionAxis(sim::TrackDirection direction) {
    switch (direction) {
    case sim::TrackDirection::East:
        return math::Vector3{1.0f, 0.0f, 0.0f};
    case sim::TrackDirection::West:
        return math::Vector3{-1.0f, 0.0f, 0.0f};
    case sim::TrackDirection::South:
        return math::Vector3{0.0f, 0.0f, 1.0f};
    case sim::TrackDirection::North:
    default:
        return math::Vector3{0.0f, 0.0f, -1.0f};
    }
}

bool dirSharesAxis(core::Dir6 lhs, core::Dir6 rhs) {
    return lhs == rhs || core::areOpposite(lhs, rhs);
}

float computeRenderedPipeRadius(float baseRadius, bool hasBranchConnection) {
    float renderedRadius = std::clamp(baseRadius, kPipeMinRadius, kPipeMaxRadius);
    if (hasBranchConnection) {
        renderedRadius = std::min(kPipeMaxRadius, renderedRadius + kPipeBranchRadiusBoost);
    }
    return renderedRadius;
}

std::uint64_t pipeCellKey(const core::Cell3i& cell) {
    constexpr std::uint64_t kMask = (1ull << 21u) - 1ull;
    const std::uint64_t x = static_cast<std::uint64_t>(static_cast<std::uint32_t>(cell.x) & kMask);
    const std::uint64_t y = static_cast<std::uint64_t>(static_cast<std::uint32_t>(cell.y) & kMask);
    const std::uint64_t z = static_cast<std::uint64_t>(static_cast<std::uint32_t>(cell.z) & kMask);
    return x | (y << 21u) | (z << 42u);
}

std::vector<PipeEndpointState> buildPipeEndpointStates(
    const std::vector<sim::Pipe>& pipes
) {
    std::unordered_map<std::uint64_t, std::size_t> pipeCellToIndex;
    pipeCellToIndex.reserve(pipes.size() * 2u);
    for (std::size_t i = 0; i < pipes.size(); ++i) {
        const core::Cell3i cell{
            pipes[i].x,
            pipes[i].y,
            pipes[i].z
        };
        pipeCellToIndex.emplace(pipeCellKey(cell), i);
    }

    auto hasPipeAtCell = [&pipeCellToIndex](const core::Cell3i& cell) -> bool {
        return pipeCellToIndex.find(pipeCellKey(cell)) != pipeCellToIndex.end();
    };

    std::vector<core::Dir6> axisDirections(pipes.size(), core::Dir6::PosY);
    std::vector<float> renderedRadii(pipes.size(), 0.45f);
    std::vector<bool> hasBranchConnections(pipes.size(), false);
    for (std::size_t i = 0; i < pipes.size(); ++i) {
        const sim::Pipe& pipe = pipes[i];
        const core::Cell3i cell{
            pipe.x,
            pipe.y,
            pipe.z
        };
        const core::Dir6 axisDir = dominantAxisDir6(pipe.axis);
        const core::Dir6 startDir = core::oppositeDir(axisDir);
        const core::Dir6 endDir = axisDir;
        const std::uint8_t neighborMask = sim::neighborMask6(cell, hasPipeAtCell);
        const std::uint8_t axialMask = static_cast<std::uint8_t>(core::dirBit(startDir) | core::dirBit(endDir));
        const bool hasBranchConnection = (neighborMask & static_cast<std::uint8_t>(~axialMask & 0x3Fu)) != 0u;

        axisDirections[i] = axisDir;
        hasBranchConnections[i] = hasBranchConnection;
        renderedRadii[i] = computeRenderedPipeRadius(pipe.radius, hasBranchConnection);
    }

    auto endExtensionForDirection = [&](
                                        std::size_t pipeIndex,
                                        const core::Cell3i& cell,
                                        core::Dir6 endDirection
                                    ) -> float {
        const core::Cell3i neighborCell = core::neighborCell(cell, endDirection);
        const auto neighborIt = pipeCellToIndex.find(pipeCellKey(neighborCell));
        if (neighborIt == pipeCellToIndex.end()) {
            return 0.0f;
        }

        const std::size_t neighborIndex = neighborIt->second;
        if (neighborIndex >= pipes.size()) {
            return 0.0f;
        }

        if (dirSharesAxis(axisDirections[pipeIndex], axisDirections[neighborIndex])) {
            return 0.0f;
        }

        const float neighborHalfExtent = kPipeTransferHalfExtent * renderedRadii[neighborIndex];
        return std::clamp(0.5f - neighborHalfExtent, 0.0f, kPipeMaxEndExtension);
    };

    std::vector<PipeEndpointState> states(pipes.size());
    for (std::size_t i = 0; i < pipes.size(); ++i) {
        const sim::Pipe& pipe = pipes[i];
        const core::Cell3i cell{
            pipe.x,
            pipe.y,
            pipe.z
        };
        const core::Dir6 axisDir = axisDirections[i];
        const core::Dir6 startDir = core::oppositeDir(axisDir);
        const core::Dir6 endDir = axisDir;
        states[i].axis = core::dirToUnitVector(axisDir);
        states[i].renderedRadius = renderedRadii[i];
        states[i].startExtension = endExtensionForDirection(i, cell, startDir);
        states[i].endExtension = endExtensionForDirection(i, cell, endDir);
    }

    return states;
}

math::Matrix4 transpose(const math::Matrix4& matrix) {
    math::Matrix4 result{};
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            result(row, col) = matrix(col, row);
        }
    }
    return result;
}

math::Matrix4 perspectiveVulkan(float fovYRadians, float aspectRatio, float nearPlane, float farPlane) {
    return math::perspectiveVulkanReverseZ(fovYRadians, aspectRatio, nearPlane, farPlane);
}

math::Matrix4 orthographicVulkan(
    float left,
    float right,
    float bottom,
    float top,
    float nearPlane,
    float farPlane
) {
    return math::orthographicVulkanReverseZ(left, right, bottom, top, nearPlane, farPlane);
}

math::Matrix4 lookAt(const math::Vector3& eye, const math::Vector3& target, const math::Vector3& up) {
    const math::Vector3 forward = math::normalize(target - eye);
    const math::Vector3 right = math::normalize(math::cross(forward, up));
    const math::Vector3 cameraUp = math::cross(right, forward);

    math::Matrix4 view = math::Matrix4::identity();
    view(0, 0) = right.x;
    view(0, 1) = right.y;
    view(0, 2) = right.z;
    view(0, 3) = -math::dot(right, eye);

    view(1, 0) = cameraUp.x;
    view(1, 1) = cameraUp.y;
    view(1, 2) = cameraUp.z;
    view(1, 3) = -math::dot(cameraUp, eye);

    view(2, 0) = -forward.x;
    view(2, 1) = -forward.y;
    view(2, 2) = -forward.z;
    view(2, 3) = math::dot(forward, eye);
    return view;
}

float saturate(float value) {
    return std::clamp(value, 0.0f, 1.0f);
}

float smoothStep(float edge0, float edge1, float x) {
    const float t = saturate((x - edge0) / std::max(edge1 - edge0, 1e-6f));
    return t * t * (3.0f - (2.0f * t));
}

math::Vector3 lerpVec3(const math::Vector3& a, const math::Vector3& b, float t) {
    return (a * (1.0f - t)) + (b * t);
}

math::Vector3 computeSunColor(
    const Renderer::SkyDebugSettings& settings,
    const math::Vector3& sunDirection
) {
    const math::Vector3 toSun = -math::normalize(sunDirection);
    const float sunAltitude = std::clamp(toSun.y, -1.0f, 1.0f);
    const float dayFactor = smoothStep(0.05f, 0.65f, sunAltitude);
    const float twilightFactor = 1.0f - dayFactor;
    const float horizonBand = saturate(1.0f - (std::abs(sunAltitude) / 0.35f));
    const float warmAmount = twilightFactor * std::pow(horizonBand, 1.2f);
    const float pinkAmount = warmAmount * saturate((0.10f - sunAltitude) / 0.30f);

    const float rayleigh = std::max(settings.rayleighStrength, 0.01f);
    const float mie = std::max(settings.mieStrength, 0.01f);
    const math::Vector3 dayTint{1.00f, 0.98f, 0.94f};
    const math::Vector3 goldenTint{1.18f, 0.72f, 0.34f};
    const math::Vector3 pinkTint{1.08f, 0.56f, 0.74f};

    math::Vector3 sunTint = lerpVec3(dayTint, goldenTint, warmAmount);
    sunTint = lerpVec3(sunTint, pinkTint, pinkAmount * 0.45f);

    const float scatteringScale = (rayleigh * 0.55f) + (mie * 0.80f);
    const float twilightBoost = 0.85f + (warmAmount * 0.45f);
    return sunTint * (scatteringScale * twilightBoost);
}

math::Vector3 proceduralSkyRadiance(
    const math::Vector3& direction,
    const math::Vector3& sunDirection,
    const math::Vector3& sunColor,
    const Renderer::SkyDebugSettings& settings
) {
    const math::Vector3 dir = math::normalize(direction);
    const math::Vector3 toSun = -math::normalize(sunDirection);
    const float horizonT = saturate((dir.y * 0.5f) + 0.5f);
    const float skyT = std::pow(horizonT, 0.35f);
    const float sunAltitude = std::clamp(toSun.y, -1.0f, 1.0f);
    const float dayFactor = smoothStep(0.05f, 0.65f, sunAltitude);
    const float twilightFactor = 1.0f - dayFactor;
    const float horizonBand = saturate(1.0f - (std::abs(sunAltitude) / 0.35f));
    const float warmAmount = twilightFactor * std::pow(horizonBand, 1.2f);
    const float pinkAmount = warmAmount * saturate((0.10f - sunAltitude) / 0.30f);

    const float rayleigh = std::max(settings.rayleighStrength, 0.01f);
    const float mie = std::max(settings.mieStrength, 0.01f);

    const math::Vector3 dayHorizonRayleigh{0.54f, 0.70f, 1.00f};
    const math::Vector3 dayHorizonMie{1.00f, 0.74f, 0.42f};
    const math::Vector3 sunsetHorizonRayleigh{0.74f, 0.44f, 0.52f};
    const math::Vector3 sunsetHorizonMie{1.18f, 0.54f, 0.30f};
    const math::Vector3 pinkHorizonRayleigh{0.70f, 0.36f, 0.68f};
    const math::Vector3 pinkHorizonMie{1.08f, 0.46f, 0.72f};

    const float zenithWarm = twilightFactor * 0.58f;
    const math::Vector3 dayZenithRayleigh{0.06f, 0.24f, 0.54f};
    const math::Vector3 dayZenithMie{0.22f, 0.20f, 0.15f};
    const math::Vector3 duskZenithRayleigh{0.16f, 0.12f, 0.30f};
    const math::Vector3 duskZenithMie{0.30f, 0.18f, 0.24f};

    math::Vector3 horizonRayleigh = lerpVec3(dayHorizonRayleigh, sunsetHorizonRayleigh, warmAmount);
    math::Vector3 horizonMie = lerpVec3(dayHorizonMie, sunsetHorizonMie, warmAmount);
    horizonRayleigh = lerpVec3(horizonRayleigh, pinkHorizonRayleigh, pinkAmount * 0.70f);
    horizonMie = lerpVec3(horizonMie, pinkHorizonMie, pinkAmount * 0.85f);

    const math::Vector3 zenithRayleigh = lerpVec3(dayZenithRayleigh, duskZenithRayleigh, zenithWarm);
    const math::Vector3 zenithMie = lerpVec3(dayZenithMie, duskZenithMie, zenithWarm);

    const math::Vector3 horizonColor =
        (horizonRayleigh * rayleigh) +
        (horizonMie * (mie * 0.58f));
    const math::Vector3 zenithColor =
        (zenithRayleigh * rayleigh) +
        (zenithMie * (mie * 0.25f));
    const math::Vector3 baseSky = (horizonColor * (1.0f - skyT)) + (zenithColor * skyT);

    const float sunDot = std::max(math::dot(dir, toSun), 0.0f);
    const float sunDisk = std::pow(sunDot, 1100.0f);
    const float sunGlow = std::pow(sunDot, 24.0f);
    const float g = std::clamp(settings.mieAnisotropy, 0.0f, 0.98f);
    constexpr float kInv4Pi = 0.0795774715f;
    const float phaseRayleigh = kInv4Pi * 0.75f * (1.0f + (sunDot * sunDot));
    const float phaseMie = kInv4Pi * (1.0f - (g * g)) /
        std::max(0.001f, std::pow(1.0f + (g * g) - (2.0f * g * sunDot), 1.5f));
    const float phaseBoost = (phaseRayleigh * rayleigh) + (phaseMie * mie * 1.4f);

    const float aboveHorizon = saturate(dir.y * 4.0f + 0.2f);
    const math::Vector3 sky = (baseSky * aboveHorizon)
        + (sunColor * (((sunDisk * 5.0f) + (sunGlow * 1.2f)) * (1.0f + phaseBoost)));

    const math::Vector3 groundColor{0.05f, 0.06f, 0.07f};
    const float belowHorizon = saturate(-dir.y);
    const math::Vector3 horizonGroundColor = horizonColor * 0.32f;
    const float groundWeight = std::pow(belowHorizon, 0.55f);
    const math::Vector3 ground = (horizonGroundColor * (1.0f - groundWeight)) + (groundColor * groundWeight);

    const float skyWeight = saturate((dir.y + 0.18f) / 0.20f);
    const float skyExposure = std::max(settings.skyExposure, 0.01f);
    return ((ground * (1.0f - skyWeight)) + (sky * skyWeight)) * skyExposure;
}

float shBasis(int index, const math::Vector3& direction) {
    const float x = direction.x;
    const float y = direction.y;
    const float z = direction.z;
    switch (index) {
    case 0: return 0.282095f;
    case 1: return 0.488603f * y;
    case 2: return 0.488603f * z;
    case 3: return 0.488603f * x;
    case 4: return 1.092548f * x * y;
    case 5: return 1.092548f * y * z;
    case 6: return 0.315392f * ((3.0f * z * z) - 1.0f);
    case 7: return 1.092548f * x * z;
    case 8: return 0.546274f * ((x * x) - (y * y));
    default: return 0.0f;
    }
}

std::array<math::Vector3, 9> computeIrradianceShCoefficients(
    const math::Vector3& sunDirection,
    const math::Vector3& sunColor,
    const Renderer::SkyDebugSettings& settings
) {
    constexpr uint32_t kThetaSamples = 16;
    constexpr uint32_t kPhiSamples = 32;
    constexpr float kPi = 3.14159265358979323846f;
    constexpr float kTwoPi = 2.0f * kPi;

    std::array<math::Vector3, 9> coefficients{};
    for (math::Vector3& coefficient : coefficients) {
        coefficient = math::Vector3{};
    }

    float weightSum = 0.0f;
    for (uint32_t thetaIdx = 0; thetaIdx < kThetaSamples; ++thetaIdx) {
        const float v = (static_cast<float>(thetaIdx) + 0.5f) / static_cast<float>(kThetaSamples);
        const float theta = v * kPi;
        const float sinTheta = std::sin(theta);
        const float cosTheta = std::cos(theta);

        for (uint32_t phiIdx = 0; phiIdx < kPhiSamples; ++phiIdx) {
            const float u = (static_cast<float>(phiIdx) + 0.5f) / static_cast<float>(kPhiSamples);
            const float phi = u * kTwoPi;
            const math::Vector3 dir{
                std::cos(phi) * sinTheta,
                cosTheta,
                std::sin(phi) * sinTheta
            };

            const math::Vector3 radiance = proceduralSkyRadiance(dir, sunDirection, sunColor, settings);
            const float sampleWeight = sinTheta;
            for (int basisIndex = 0; basisIndex < 9; ++basisIndex) {
                const float basisValue = shBasis(basisIndex, dir);
                coefficients[basisIndex] += radiance * (basisValue * sampleWeight);
            }
            weightSum += sampleWeight;
        }
    }

    if (weightSum <= 0.0f) {
        return coefficients;
    }

    const float normalization = (4.0f * kPi) / weightSum;
    for (math::Vector3& coefficient : coefficients) {
        coefficient *= normalization;
    }

    // Convolve SH radiance with Lambert kernel for diffuse irradiance.
    coefficients[0] *= kPi;
    coefficients[1] *= (2.0f * kPi / 3.0f);
    coefficients[2] *= (2.0f * kPi / 3.0f);
    coefficients[3] *= (2.0f * kPi / 3.0f);
    coefficients[4] *= (kPi * 0.25f);
    coefficients[5] *= (kPi * 0.25f);
    coefficients[6] *= (kPi * 0.25f);
    coefficients[7] *= (kPi * 0.25f);
    coefficients[8] *= (kPi * 0.25f);

    return coefficients;
}

uint32_t findMemoryTypeIndex(
    VkPhysicalDevice physicalDevice,
    uint32_t typeBits,
    VkMemoryPropertyFlags requiredProperties
) {
    VkPhysicalDeviceMemoryProperties memoryProperties{};
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);

    for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i) {
        const bool typeMatches = (typeBits & (1u << i)) != 0;
        const bool propertiesMatch =
            (memoryProperties.memoryTypes[i].propertyFlags & requiredProperties) == requiredProperties;
        if (typeMatches && propertiesMatch) {
            return i;
        }
    }
    return std::numeric_limits<uint32_t>::max();
}

void transitionImageLayout(
    VkCommandBuffer commandBuffer,
    VkImage image,
    VkImageLayout oldLayout,
    VkImageLayout newLayout,
    VkPipelineStageFlags2 srcStageMask,
    VkAccessFlags2 srcAccessMask,
    VkPipelineStageFlags2 dstStageMask,
    VkAccessFlags2 dstAccessMask,
    VkImageAspectFlags aspectMask,
    uint32_t baseArrayLayer = 0,
    uint32_t layerCount = 1,
    uint32_t baseMipLevel = 0,
    uint32_t levelCount = 1
) {
    VkImageMemoryBarrier2 imageBarrier{};
    imageBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    imageBarrier.srcStageMask = srcStageMask;
    imageBarrier.srcAccessMask = srcAccessMask;
    imageBarrier.dstStageMask = dstStageMask;
    imageBarrier.dstAccessMask = dstAccessMask;
    imageBarrier.oldLayout = oldLayout;
    imageBarrier.newLayout = newLayout;
    imageBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    imageBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    imageBarrier.image = image;
    imageBarrier.subresourceRange.aspectMask = aspectMask;
    imageBarrier.subresourceRange.baseMipLevel = baseMipLevel;
    imageBarrier.subresourceRange.levelCount = levelCount;
    imageBarrier.subresourceRange.baseArrayLayer = baseArrayLayer;
    imageBarrier.subresourceRange.layerCount = layerCount;

    VkDependencyInfo dependencyInfo{};
    dependencyInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dependencyInfo.imageMemoryBarrierCount = 1;
    dependencyInfo.pImageMemoryBarriers = &imageBarrier;
    vkCmdPipelineBarrier2(commandBuffer, &dependencyInfo);
}

VkFormat findSupportedDepthFormat(VkPhysicalDevice physicalDevice) {
    constexpr std::array<VkFormat, 3> kDepthCandidates = {
        VK_FORMAT_D32_SFLOAT,
        VK_FORMAT_D32_SFLOAT_S8_UINT,
        VK_FORMAT_D24_UNORM_S8_UINT
    };

    for (VkFormat format : kDepthCandidates) {
        VkFormatProperties properties{};
        vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &properties);
        if ((properties.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) != 0) {
            return format;
        }
    }
    return VK_FORMAT_UNDEFINED;
}

VkFormat findSupportedShadowDepthFormat(VkPhysicalDevice physicalDevice) {
    constexpr std::array<VkFormat, 2> kShadowDepthCandidates = {
        VK_FORMAT_D32_SFLOAT,
        VK_FORMAT_D16_UNORM
    };

    for (VkFormat format : kShadowDepthCandidates) {
        VkFormatProperties properties{};
        vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &properties);
        const VkFormatFeatureFlags requiredFeatures =
            VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
        if ((properties.optimalTilingFeatures & requiredFeatures) == requiredFeatures) {
            return format;
        }
    }
    return VK_FORMAT_UNDEFINED;
}

VkFormat findSupportedHdrColorFormat(VkPhysicalDevice physicalDevice) {
    constexpr std::array<VkFormat, 2> kHdrCandidates = {
        VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_FORMAT_B10G11R11_UFLOAT_PACK32
    };

    for (VkFormat format : kHdrCandidates) {
        VkFormatProperties properties{};
        vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &properties);
        const VkFormatFeatureFlags requiredFeatures =
            VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
        if ((properties.optimalTilingFeatures & requiredFeatures) == requiredFeatures) {
            return format;
        }
    }
    return VK_FORMAT_UNDEFINED;
}

VkFormat findSupportedNormalDepthFormat(VkPhysicalDevice physicalDevice) {
    constexpr std::array<VkFormat, 2> kNormalDepthCandidates = {
        VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_FORMAT_R32G32B32A32_SFLOAT
    };

    for (VkFormat format : kNormalDepthCandidates) {
        VkFormatProperties properties{};
        vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &properties);
        const VkFormatFeatureFlags requiredFeatures =
            VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
        if ((properties.optimalTilingFeatures & requiredFeatures) == requiredFeatures) {
            return format;
        }
    }
    return VK_FORMAT_UNDEFINED;
}

VkFormat findSupportedSsaoFormat(VkPhysicalDevice physicalDevice) {
    constexpr std::array<VkFormat, 2> kSsaoCandidates = {
        VK_FORMAT_R16_SFLOAT,
        VK_FORMAT_R8_UNORM
    };

    for (VkFormat format : kSsaoCandidates) {
        VkFormatProperties properties{};
        vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &properties);
        const VkFormatFeatureFlags requiredFeatures =
            VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
        if ((properties.optimalTilingFeatures & requiredFeatures) == requiredFeatures) {
            return format;
        }
    }
    return VK_FORMAT_UNDEFINED;
}

struct QueueFamilyChoice {
    std::optional<uint32_t> graphicsAndPresent;
    std::optional<uint32_t> transfer;
    uint32_t graphicsQueueIndex = 0;
    uint32_t transferQueueIndex = 0;

    [[nodiscard]] bool valid() const {
        return graphicsAndPresent.has_value() && transfer.has_value();
    }
};

struct SwapchainSupport {
    VkSurfaceCapabilitiesKHR capabilities{};
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

const char* vkResultName(VkResult result) {
    switch (result) {
    case VK_SUCCESS: return "VK_SUCCESS";
    case VK_NOT_READY: return "VK_NOT_READY";
    case VK_TIMEOUT: return "VK_TIMEOUT";
    case VK_EVENT_SET: return "VK_EVENT_SET";
    case VK_EVENT_RESET: return "VK_EVENT_RESET";
    case VK_INCOMPLETE: return "VK_INCOMPLETE";
    case VK_ERROR_OUT_OF_HOST_MEMORY: return "VK_ERROR_OUT_OF_HOST_MEMORY";
    case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
    case VK_ERROR_INITIALIZATION_FAILED: return "VK_ERROR_INITIALIZATION_FAILED";
    case VK_ERROR_DEVICE_LOST: return "VK_ERROR_DEVICE_LOST";
    case VK_ERROR_MEMORY_MAP_FAILED: return "VK_ERROR_MEMORY_MAP_FAILED";
    case VK_ERROR_LAYER_NOT_PRESENT: return "VK_ERROR_LAYER_NOT_PRESENT";
    case VK_ERROR_EXTENSION_NOT_PRESENT: return "VK_ERROR_EXTENSION_NOT_PRESENT";
    case VK_ERROR_FEATURE_NOT_PRESENT: return "VK_ERROR_FEATURE_NOT_PRESENT";
    case VK_ERROR_INCOMPATIBLE_DRIVER: return "VK_ERROR_INCOMPATIBLE_DRIVER";
    case VK_ERROR_SURFACE_LOST_KHR: return "VK_ERROR_SURFACE_LOST_KHR";
    case VK_ERROR_NATIVE_WINDOW_IN_USE_KHR: return "VK_ERROR_NATIVE_WINDOW_IN_USE_KHR";
    case VK_SUBOPTIMAL_KHR: return "VK_SUBOPTIMAL_KHR";
    case VK_ERROR_OUT_OF_DATE_KHR: return "VK_ERROR_OUT_OF_DATE_KHR";
    default: return "VK_RESULT_UNKNOWN";
    }
}

void logVkFailure(const char* context, VkResult result) {
    VOX_LOGE("render") << context << " failed: "
                       << vkResultName(result) << " (" << static_cast<int>(result) << ")";
}

template <typename VkHandleT>
uint64_t vkHandleToUint64(VkHandleT handle) {
    if constexpr (std::is_pointer_v<VkHandleT>) {
        return reinterpret_cast<uint64_t>(handle);
    } else {
        return static_cast<uint64_t>(handle);
    }
}

bool isLayerAvailable(const char* layerName) {
    uint32_t layerCount = 0;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
    std::vector<VkLayerProperties> layers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, layers.data());

    for (const VkLayerProperties& layer : layers) {
        if (std::strcmp(layer.layerName, layerName) == 0) {
            return true;
        }
    }
    return false;
}

bool isInstanceExtensionAvailable(const char* extensionName) {
    uint32_t extensionCount = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);
    std::vector<VkExtensionProperties> extensions(extensionCount);
    vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data());

    for (const VkExtensionProperties& extension : extensions) {
        if (std::strcmp(extension.extensionName, extensionName) == 0) {
            return true;
        }
    }
    return false;
}

void appendInstanceExtensionIfMissing(std::vector<const char*>& extensions, const char* extensionName) {
    const auto found = std::find_if(
        extensions.begin(),
        extensions.end(),
        [extensionName](const char* existing) {
            return std::strcmp(existing, extensionName) == 0;
        }
    );
    if (found == extensions.end()) {
        extensions.push_back(extensionName);
    }
}

QueueFamilyChoice findQueueFamily(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface) {
    QueueFamilyChoice choice;

    uint32_t familyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &familyCount, nullptr);
    std::vector<VkQueueFamilyProperties> families(familyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &familyCount, families.data());

    std::optional<uint32_t> dedicatedTransferFamily;
    std::optional<uint32_t> anyTransferFamily;

    for (uint32_t familyIndex = 0; familyIndex < familyCount; ++familyIndex) {
        const VkQueueFlags queueFlags = families[familyIndex].queueFlags;
        const bool hasGraphics = (queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0;
        const bool hasTransfer = (queueFlags & VK_QUEUE_TRANSFER_BIT) != 0;

        if (hasGraphics && !choice.graphicsAndPresent.has_value()) {
            VkBool32 hasPresent = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice, familyIndex, surface, &hasPresent);
            if (hasPresent == VK_TRUE) {
                choice.graphicsAndPresent = familyIndex;
            }
        }

        if (hasTransfer) {
            if (!anyTransferFamily.has_value()) {
                anyTransferFamily = familyIndex;
            }
            if (!dedicatedTransferFamily.has_value() && !hasGraphics) {
                dedicatedTransferFamily = familyIndex;
            }
        }
    }

    if (!choice.graphicsAndPresent.has_value()) {
        return choice;
    }

    if (dedicatedTransferFamily.has_value()) {
        choice.transfer = dedicatedTransferFamily.value();
    } else if (anyTransferFamily.has_value()) {
        choice.transfer = anyTransferFamily.value();
    } else {
        choice.transfer = choice.graphicsAndPresent.value();
    }

    if (choice.transfer.value() == choice.graphicsAndPresent.value()) {
        const uint32_t queueCount = families[choice.graphicsAndPresent.value()].queueCount;
        if (queueCount > 1) {
            choice.transferQueueIndex = 1;
        }
    }

    return choice;
}

bool hasRequiredDeviceExtensions(VkPhysicalDevice physicalDevice) {
    uint32_t extensionCount = 0;
    vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extensionCount, nullptr);
    std::vector<VkExtensionProperties> extensions(extensionCount);
    vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extensionCount, extensions.data());

    for (const char* required : kDeviceExtensions) {
        bool found = false;
        for (const VkExtensionProperties& available : extensions) {
            if (std::strcmp(required, available.extensionName) == 0) {
                found = true;
                break;
            }
        }
        if (!found) {
            return false;
        }
    }

    return true;
}

SwapchainSupport querySwapchainSupport(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface) {
    SwapchainSupport support;

    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface, &support.capabilities);

    uint32_t formatCount = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount, nullptr);
    support.formats.resize(formatCount);
    if (formatCount > 0) {
        vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount, support.formats.data());
    }

    uint32_t presentModeCount = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &presentModeCount, nullptr);
    support.presentModes.resize(presentModeCount);
    if (presentModeCount > 0) {
        vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &presentModeCount, support.presentModes.data());
    }

    return support;
}

VkSurfaceFormatKHR chooseSwapchainFormat(const std::vector<VkSurfaceFormatKHR>& formats) {
    for (const VkSurfaceFormatKHR format : formats) {
        if (format.format == VK_FORMAT_B8G8R8A8_UNORM && format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return format;
        }
    }
    return formats.front();
}

VkPresentModeKHR choosePresentMode(const std::vector<VkPresentModeKHR>& presentModes) {
    for (const VkPresentModeKHR presentMode : presentModes) {
        if (presentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
            return presentMode;
        }
    }
    return VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D chooseExtent(GLFWwindow* window, const VkSurfaceCapabilitiesKHR& capabilities) {
    if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
        return capabilities.currentExtent;
    }

    int width = 0;
    int height = 0;
    glfwGetFramebufferSize(window, &width, &height);

    VkExtent2D extent{};
    extent.width = std::clamp(
        static_cast<uint32_t>(std::max(width, 1)),
        capabilities.minImageExtent.width,
        capabilities.maxImageExtent.width
    );
    extent.height = std::clamp(
        static_cast<uint32_t>(std::max(height, 1)),
        capabilities.minImageExtent.height,
        capabilities.maxImageExtent.height
    );
    return extent;
}

std::optional<std::vector<std::uint8_t>> readBinaryFile(const char* filePath) {
    if (filePath == nullptr) {
        return std::nullopt;
    }

    const std::filesystem::path path(filePath);
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        return std::nullopt;
    }

    const std::streamsize size = file.tellg();
    if (size <= 0) {
        return std::nullopt;
    }
    file.seekg(0, std::ios::beg);

    std::vector<std::uint8_t> data(static_cast<size_t>(size));
    if (!file.read(reinterpret_cast<char*>(data.data()), size)) {
        return std::nullopt;
    }
    return data;
}

bool createShaderModuleFromFile(
    VkDevice device,
    const char* filePath,
    const char* debugName,
    VkShaderModule& outShaderModule
) {
    outShaderModule = VK_NULL_HANDLE;

    const std::optional<std::vector<std::uint8_t>> shaderFileData = readBinaryFile(filePath);
    if (!shaderFileData.has_value()) {
        VOX_LOGE("render") << "missing shader file for " << debugName << ": " << (filePath != nullptr ? filePath : "<null>") << "\n";
        return false;
    }
    if ((shaderFileData->size() % sizeof(std::uint32_t)) != 0) {
        VOX_LOGE("render") << "invalid SPIR-V byte size for " << debugName << ": " << filePath << "\n";
        return false;
    }
    const std::uint32_t* code = reinterpret_cast<const std::uint32_t*>(shaderFileData->data());
    const size_t codeSize = shaderFileData->size();

    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = codeSize;
    createInfo.pCode = code;
    const VkResult result = vkCreateShaderModule(device, &createInfo, nullptr, &outShaderModule);
    if (result != VK_SUCCESS) {
        logVkFailure("vkCreateShaderModule(fileOrFallback)", result);
        return false;
    }
    return true;
}

} // namespace

void Renderer::setDebugUiVisible(bool visible) {
    if (m_debugUiVisible == visible) {
        return;
    }
    m_debugUiVisible = visible;
    m_showMeshingPanel = visible;
    m_showShadowPanel = visible;
    m_showSunPanel = visible;
}

bool Renderer::isDebugUiVisible() const {
    return m_debugUiVisible;
}

void Renderer::setFrameStatsVisible(bool visible) {
    m_showFrameStatsPanel = visible;
}

bool Renderer::isFrameStatsVisible() const {
    return m_showFrameStatsPanel;
}

bool Renderer::init(GLFWwindow* window, const world::ChunkGrid& chunkGrid) {
    using Clock = std::chrono::steady_clock;
    const auto initStart = Clock::now();
    auto elapsedMs = [](const Clock::time_point& start) -> std::int64_t {
        return std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - start).count();
    };
    auto runStep = [&](const char* stepName, auto&& stepFn) -> bool {
        const auto stepStart = Clock::now();
        const bool ok = stepFn();
        VOX_LOGI("render") << "init step " << stepName << " took " << elapsedMs(stepStart) << " ms\n";
        return ok;
    };

    VOX_LOGI("render") << "init begin\n";
    m_window = window;
    if (m_window == nullptr) {
        VOX_LOGE("render") << "init failed: window is null\n";
        return false;
    }

    if (glfwVulkanSupported() == GLFW_FALSE) {
        VOX_LOGE("render") << "init failed: glfwVulkanSupported returned false\n";
        return false;
    }

    if (!runStep("createInstance", [&] { return createInstance(); })) {
        VOX_LOGE("render") << "init failed at createInstance\n";
        shutdown();
        return false;
    }
    if (!runStep("createSurface", [&] { return createSurface(); })) {
        VOX_LOGE("render") << "init failed at createSurface\n";
        shutdown();
        return false;
    }
    if (!runStep("pickPhysicalDevice", [&] { return pickPhysicalDevice(); })) {
        VOX_LOGE("render") << "init failed at pickPhysicalDevice\n";
        shutdown();
        return false;
    }
    if (!runStep("createLogicalDevice", [&] { return createLogicalDevice(); })) {
        VOX_LOGE("render") << "init failed at createLogicalDevice\n";
        shutdown();
        return false;
    }
    if (!runStep("createTimelineSemaphore", [&] { return createTimelineSemaphore(); })) {
        VOX_LOGE("render") << "init failed at createTimelineSemaphore\n";
        shutdown();
        return false;
    }
    if (!runStep("bufferAllocator.init", [&] {
            return m_bufferAllocator.init(
                m_physicalDevice,
                m_device
#if defined(VOXEL_HAS_VMA)
                ,
                m_vmaAllocator
#endif
            );
        })) {
        VOX_LOGE("render") << "init failed at buffer allocator init\n";
        shutdown();
        return false;
    }
    if (!runStep("createUploadRingBuffer", [&] { return createUploadRingBuffer(); })) {
        VOX_LOGE("render") << "init failed at createUploadRingBuffer\n";
        shutdown();
        return false;
    }
    if (!runStep("createTransferResources", [&] { return createTransferResources(); })) {
        VOX_LOGE("render") << "init failed at createTransferResources\n";
        shutdown();
        return false;
    }
    if (!runStep("createEnvironmentResources", [&] { return createEnvironmentResources(); })) {
        VOX_LOGE("render") << "init failed at createEnvironmentResources\n";
        shutdown();
        return false;
    }
    if (!runStep("createShadowResources", [&] { return createShadowResources(); })) {
        VOX_LOGE("render") << "init failed at createShadowResources\n";
        shutdown();
        return false;
    }
    if (!runStep("createSwapchain", [&] { return createSwapchain(); })) {
        VOX_LOGE("render") << "init failed at createSwapchain\n";
        shutdown();
        return false;
    }
    if (!runStep("createDescriptorResources", [&] { return createDescriptorResources(); })) {
        VOX_LOGE("render") << "init failed at createDescriptorResources\n";
        shutdown();
        return false;
    }
    if (!runStep("createGraphicsPipeline", [&] { return createGraphicsPipeline(); })) {
        VOX_LOGE("render") << "init failed at createGraphicsPipeline\n";
        shutdown();
        return false;
    }
    if (!runStep("createPipePipeline", [&] { return createPipePipeline(); })) {
        VOX_LOGE("render") << "init failed at createPipePipeline\n";
        shutdown();
        return false;
    }
    if (!runStep("createAoPipelines", [&] { return createAoPipelines(); })) {
        VOX_LOGE("render") << "init failed at createAoPipelines\n";
        shutdown();
        return false;
    }
    {
        const auto frameArenaStart = Clock::now();
        m_frameArena.beginFrame(0);
        VOX_LOGI("render") << "init step frameArena.beginFrame(0) took " << elapsedMs(frameArenaStart) << " ms\n";
    }
    if (!runStep("createChunkBuffers", [&] { return createChunkBuffers(chunkGrid, {}); })) {
        VOX_LOGE("render") << "init failed at createChunkBuffers\n";
        shutdown();
        return false;
    }
    if (!runStep("createPipeBuffers", [&] { return createPipeBuffers(); })) {
        VOX_LOGE("render") << "init failed at createPipeBuffers\n";
        shutdown();
        return false;
    }
    if (!runStep("createPreviewBuffers", [&] { return createPreviewBuffers(); })) {
        VOX_LOGE("render") << "init failed at createPreviewBuffers\n";
        shutdown();
        return false;
    }
    if (!runStep("createFrameResources", [&] { return createFrameResources(); })) {
        VOX_LOGE("render") << "init failed at createFrameResources\n";
        shutdown();
        return false;
    }
    if (!runStep("createGpuTimestampResources", [&] { return createGpuTimestampResources(); })) {
        VOX_LOGE("render") << "init failed at createGpuTimestampResources\n";
        shutdown();
        return false;
    }
#if defined(VOXEL_HAS_IMGUI)
    if (!runStep("createImGuiResources", [&] { return createImGuiResources(); })) {
        VOX_LOGE("render") << "init failed at createImGuiResources\n";
        shutdown();
        return false;
    }
#endif

    VOX_LOGI("render") << "init complete in " << elapsedMs(initStart) << " ms\n";
    return true;
}

bool Renderer::updateChunkMesh(const world::ChunkGrid& chunkGrid) {
    if (m_device == VK_NULL_HANDLE) {
        return false;
    }
    (void)chunkGrid;
    m_chunkMeshRebuildRequested = true;
    m_pendingChunkRemeshIndices.clear();
    return true;
}

bool Renderer::updateChunkMesh(const world::ChunkGrid& chunkGrid, std::size_t chunkIndex) {
    if (m_device == VK_NULL_HANDLE) {
        return false;
    }
    if (chunkIndex >= chunkGrid.chunks().size()) {
        return false;
    }
    if (m_chunkMeshRebuildRequested) {
        return true;
    }
    if (std::find(m_pendingChunkRemeshIndices.begin(), m_pendingChunkRemeshIndices.end(), chunkIndex) ==
        m_pendingChunkRemeshIndices.end()) {
        m_pendingChunkRemeshIndices.push_back(chunkIndex);
    }
    return true;
}

bool Renderer::updateChunkMesh(const world::ChunkGrid& chunkGrid, std::span<const std::size_t> chunkIndices) {
    if (m_device == VK_NULL_HANDLE) {
        return false;
    }
    if (chunkIndices.empty()) {
        return true;
    }
    if (m_chunkMeshRebuildRequested) {
        return true;
    }
    for (const std::size_t chunkIndex : chunkIndices) {
        if (chunkIndex >= chunkGrid.chunks().size()) {
            return false;
        }
        if (std::find(m_pendingChunkRemeshIndices.begin(), m_pendingChunkRemeshIndices.end(), chunkIndex) ==
            m_pendingChunkRemeshIndices.end()) {
            m_pendingChunkRemeshIndices.push_back(chunkIndex);
        }
    }
    return true;
}

bool Renderer::useSpatialPartitioningQueries() const {
    return m_debugEnableSpatialQueries;
}

world::ClipmapConfig Renderer::clipmapQueryConfig() const {
    return m_debugClipmapConfig;
}

void Renderer::setSpatialQueryStats(
    bool used,
    const world::SpatialQueryStats& stats,
    std::uint32_t visibleChunkCount
) {
    m_debugSpatialQueriesUsed = used;
    m_debugSpatialQueryStats = stats;
    m_debugSpatialVisibleChunkCount = visibleChunkCount;
}

bool Renderer::createInstance() {
#ifndef NDEBUG
    const bool enableValidationLayers = isLayerAvailable(kValidationLayers[0]);
#else
    const bool enableValidationLayers = false;
#endif
    uint32_t glfwExtensionCount = 0;
    const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
    if (glfwExtensions == nullptr || glfwExtensionCount == 0) {
        VOX_LOGI("render") << "no GLFW Vulkan instance extensions available\n";
        return false;
    }

    std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
    m_debugUtilsEnabled = isInstanceExtensionAvailable(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    if (m_debugUtilsEnabled) {
        appendInstanceExtensionIfMissing(extensions, VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    } else {
        VOX_LOGI("render") << "instance extension unavailable: " << VK_EXT_DEBUG_UTILS_EXTENSION_NAME << "\n";
    }
    VOX_LOGI("render") << "createInstance (validation="
              << (enableValidationLayers ? "on" : "off")
              << ", debugUtils=" << (m_debugUtilsEnabled ? "on" : "off")
              << ")\n";

    VkApplicationInfo applicationInfo{};
    applicationInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    applicationInfo.pApplicationName = "voxel_factory_toy";
    applicationInfo.applicationVersion = VK_MAKE_API_VERSION(0, 0, 1, 0);
    applicationInfo.pEngineName = "none";
    applicationInfo.engineVersion = VK_MAKE_API_VERSION(0, 0, 1, 0);
    applicationInfo.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &applicationInfo;
    createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();
    if (enableValidationLayers) {
        createInfo.enabledLayerCount = static_cast<uint32_t>(kValidationLayers.size());
        createInfo.ppEnabledLayerNames = kValidationLayers.data();
    }

    const VkResult result = vkCreateInstance(&createInfo, nullptr, &m_instance);
    if (result != VK_SUCCESS) {
        logVkFailure("vkCreateInstance", result);
        return false;
    }
    return true;
}

bool Renderer::createSurface() {
    const VkResult result = glfwCreateWindowSurface(m_instance, m_window, nullptr, &m_surface);
    if (result != VK_SUCCESS) {
        logVkFailure("glfwCreateWindowSurface", result);
        return false;
    }
    return true;
}

bool Renderer::pickPhysicalDevice() {
    m_supportsBindlessDescriptors = false;
    m_bindlessTextureCapacity = 0;

    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(m_instance, &deviceCount, nullptr);
    if (deviceCount == 0) {
        VOX_LOGI("render") << "no Vulkan physical devices found\n";
        return false;
    }
    VOX_LOGI("render") << "physical devices found: " << deviceCount << "\n";

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(m_instance, &deviceCount, devices.data());

    for (VkPhysicalDevice candidate : devices) {
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(candidate, &properties);
        VOX_LOGI("render") << "evaluating GPU: " << properties.deviceName
                  << ", apiVersion=" << VK_VERSION_MAJOR(properties.apiVersion) << "."
                  << VK_VERSION_MINOR(properties.apiVersion) << "."
                  << VK_VERSION_PATCH(properties.apiVersion) << "\n";
        if (properties.apiVersion < VK_API_VERSION_1_3) {
            VOX_LOGI("render") << "skip GPU: Vulkan 1.3 required\n";
            continue;
        }
        if ((properties.limits.framebufferColorSampleCounts & VK_SAMPLE_COUNT_4_BIT) == 0) {
            VOX_LOGI("render") << "skip GPU: 4x MSAA color attachments not supported\n";
            continue;
        }
        if ((properties.limits.framebufferDepthSampleCounts & VK_SAMPLE_COUNT_4_BIT) == 0) {
            VOX_LOGI("render") << "skip GPU: 4x MSAA depth attachments not supported\n";
            continue;
        }

        const QueueFamilyChoice queueFamily = findQueueFamily(candidate, m_surface);
        if (!queueFamily.valid()) {
            VOX_LOGI("render") << "skip GPU: missing graphics/present/transfer queue support\n";
            continue;
        }
        if (!hasRequiredDeviceExtensions(candidate)) {
            VOX_LOGI("render") << "skip GPU: missing required device extensions\n";
            continue;
        }

        const SwapchainSupport swapchainSupport = querySwapchainSupport(candidate, m_surface);
        if (swapchainSupport.formats.empty() || swapchainSupport.presentModes.empty()) {
            VOX_LOGI("render") << "skip GPU: swapchain support incomplete\n";
            continue;
        }
        const VkFormat depthFormat = findSupportedDepthFormat(candidate);
        if (depthFormat == VK_FORMAT_UNDEFINED) {
            VOX_LOGI("render") << "skip GPU: no supported depth format\n";
            continue;
        }
        const VkFormat shadowDepthFormat = findSupportedShadowDepthFormat(candidate);
        if (shadowDepthFormat == VK_FORMAT_UNDEFINED) {
            VOX_LOGI("render") << "skip GPU: no supported shadow depth format\n";
            continue;
        }
        const VkFormat hdrColorFormat = findSupportedHdrColorFormat(candidate);
        if (hdrColorFormat == VK_FORMAT_UNDEFINED) {
            VOX_LOGI("render") << "skip GPU: no supported HDR color format\n";
            continue;
        }
        const VkFormat normalDepthFormat = findSupportedNormalDepthFormat(candidate);
        if (normalDepthFormat == VK_FORMAT_UNDEFINED) {
            VOX_LOGI("render") << "skip GPU: no supported normal-depth color format\n";
            continue;
        }
        const VkFormat ssaoFormat = findSupportedSsaoFormat(candidate);
        if (ssaoFormat == VK_FORMAT_UNDEFINED) {
            VOX_LOGI("render") << "skip GPU: no supported SSAO format\n";
            continue;
        }

        VkPhysicalDeviceVulkan11Features vulkan11Features{};
        vulkan11Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
        VkPhysicalDeviceVulkan12Features vulkan12Features{};
        vulkan12Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        vulkan12Features.pNext = &vulkan11Features;
        VkPhysicalDeviceVulkan13Features vulkan13Features{};
        vulkan13Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
        vulkan13Features.pNext = &vulkan12Features;
        VkPhysicalDeviceMemoryPriorityFeaturesEXT memoryPriorityFeatures{};
        memoryPriorityFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PRIORITY_FEATURES_EXT;
        memoryPriorityFeatures.pNext = &vulkan13Features;
        VkPhysicalDeviceFeatures2 features2{};
        features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        features2.pNext = &memoryPriorityFeatures;
        vkGetPhysicalDeviceFeatures2(candidate, &features2);
        if (vulkan13Features.dynamicRendering != VK_TRUE) {
            VOX_LOGI("render") << "skip GPU: dynamicRendering not supported\n";
            continue;
        }
        if (vulkan12Features.timelineSemaphore != VK_TRUE) {
            VOX_LOGI("render") << "skip GPU: timelineSemaphore not supported\n";
            continue;
        }
        if (vulkan13Features.synchronization2 != VK_TRUE) {
            VOX_LOGI("render") << "skip GPU: synchronization2 not supported\n";
            continue;
        }
        if (vulkan13Features.maintenance4 != VK_TRUE) {
            VOX_LOGI("render") << "skip GPU: maintenance4 not supported\n";
            continue;
        }
        if (vulkan12Features.bufferDeviceAddress != VK_TRUE) {
            VOX_LOGI("render") << "skip GPU: bufferDeviceAddress not supported\n";
            continue;
        }
        if (memoryPriorityFeatures.memoryPriority != VK_TRUE) {
            VOX_LOGI("render") << "skip GPU: memoryPriority not supported\n";
            continue;
        }
        if (features2.features.drawIndirectFirstInstance != VK_TRUE) {
            VOX_LOGI("render") << "skip GPU: drawIndirectFirstInstance not supported\n";
            continue;
        }
        if (vulkan11Features.shaderDrawParameters != VK_TRUE) {
            VOX_LOGI("render") << "skip GPU: shaderDrawParameters not supported\n";
            continue;
        }
        const bool supportsBindlessDescriptors =
            vulkan12Features.descriptorIndexing == VK_TRUE &&
            vulkan12Features.runtimeDescriptorArray == VK_TRUE &&
            vulkan12Features.shaderSampledImageArrayNonUniformIndexing == VK_TRUE &&
            vulkan12Features.descriptorBindingPartiallyBound == VK_TRUE;
        if (!supportsBindlessDescriptors) {
            VOX_LOGI("render") << "skip GPU: bindless descriptor indexing not supported\n";
            continue;
        }

        uint32_t bindlessTextureCapacity = 0;
        const uint32_t perStageSamplerLimit = properties.limits.maxPerStageDescriptorSamplers;
        const uint32_t perStageSampledLimit = properties.limits.maxPerStageDescriptorSampledImages;
        const uint32_t descriptorSetSampledLimit = properties.limits.maxDescriptorSetSampledImages;
        uint32_t safeBudget = std::min({perStageSamplerLimit, perStageSampledLimit, descriptorSetSampledLimit});
        if (safeBudget > kBindlessReservedSampledDescriptors) {
            safeBudget -= kBindlessReservedSampledDescriptors;
        } else {
            safeBudget = 0;
        }
        bindlessTextureCapacity = std::min(kBindlessTargetTextureCapacity, safeBudget);
        if (bindlessTextureCapacity < kBindlessMinTextureCapacity) {
            VOX_LOGI("render") << "skip GPU: bindless descriptor budget too small\n";
            continue;
        }

        const bool supportsWireframe = features2.features.fillModeNonSolid == VK_TRUE;
        const bool supportsSamplerAnisotropy = features2.features.samplerAnisotropy == VK_TRUE;
        const bool supportsDrawIndirectFirstInstance = features2.features.drawIndirectFirstInstance == VK_TRUE;
        const bool supportsMultiDrawIndirect = features2.features.multiDrawIndirect == VK_TRUE;
        const float maxSamplerAnisotropy = properties.limits.maxSamplerAnisotropy;
        m_physicalDevice = candidate;
        m_graphicsQueueFamilyIndex = queueFamily.graphicsAndPresent.value();
        m_graphicsQueueIndex = queueFamily.graphicsQueueIndex;
        m_transferQueueFamilyIndex = queueFamily.transfer.value();
        m_transferQueueIndex = queueFamily.transferQueueIndex;
        m_supportsWireframePreview = supportsWireframe;
        m_supportsSamplerAnisotropy = supportsSamplerAnisotropy;
        m_supportsMultiDrawIndirect = supportsMultiDrawIndirect;
        m_supportsBindlessDescriptors = true;
        m_bindlessTextureCapacity = bindlessTextureCapacity;
        m_maxSamplerAnisotropy = maxSamplerAnisotropy;
        m_depthFormat = depthFormat;
        m_shadowDepthFormat = shadowDepthFormat;
        m_hdrColorFormat = hdrColorFormat;
        m_normalDepthFormat = normalDepthFormat;
        m_ssaoFormat = ssaoFormat;
        m_colorSampleCount = VK_SAMPLE_COUNT_4_BIT;
        VOX_LOGI("render") << "selected GPU: " << properties.deviceName
                  << ", graphicsQueueFamily=" << m_graphicsQueueFamilyIndex
                  << ", graphicsQueueIndex=" << m_graphicsQueueIndex
                  << ", transferQueueFamily=" << m_transferQueueFamilyIndex
                  << ", transferQueueIndex=" << m_transferQueueIndex
                  << ", wireframePreview=" << (m_supportsWireframePreview ? "yes" : "no")
                  << ", samplerAnisotropy=" << (m_supportsSamplerAnisotropy ? "yes" : "no")
                  << ", drawIndirectFirstInstance=" << (supportsDrawIndirectFirstInstance ? "yes" : "no")
                  << ", multiDrawIndirect=" << (m_supportsMultiDrawIndirect ? "yes" : "no")
                  << ", bindlessDescriptors=" << (m_supportsBindlessDescriptors ? "yes" : "no")
                  << ", bindlessTextureCapacity=" << m_bindlessTextureCapacity
                  << ", maxSamplerAnisotropy=" << m_maxSamplerAnisotropy
                  << ", msaaSamples=" << static_cast<uint32_t>(m_colorSampleCount)
                  << ", shadowDepthFormat=" << static_cast<int>(m_shadowDepthFormat)
                  << ", hdrColorFormat=" << static_cast<int>(m_hdrColorFormat)
                  << ", normalDepthFormat=" << static_cast<int>(m_normalDepthFormat)
                  << ", ssaoFormat=" << static_cast<int>(m_ssaoFormat)
                  << "\n";
        return true;
    }

    VOX_LOGI("render") << "no suitable GPU found\n";
    return false;
}

bool Renderer::createLogicalDevice() {
    const bool sameFamily = (m_graphicsQueueFamilyIndex == m_transferQueueFamilyIndex);
    std::array<VkDeviceQueueCreateInfo, 2> queueCreateInfos{};
    uint32_t queueCreateInfoCount = 0;
    std::array<float, 2> sharedFamilyPriorities = {1.0f, 1.0f};
    float graphicsQueuePriority = 1.0f;
    float transferQueuePriority = 1.0f;

    if (sameFamily) {
        const uint32_t queueCount = std::max(m_graphicsQueueIndex, m_transferQueueIndex) + 1;
        VkDeviceQueueCreateInfo queueCreateInfo{};
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = m_graphicsQueueFamilyIndex;
        queueCreateInfo.queueCount = queueCount;
        queueCreateInfo.pQueuePriorities = sharedFamilyPriorities.data();
        queueCreateInfos[queueCreateInfoCount++] = queueCreateInfo;
    } else {
        VkDeviceQueueCreateInfo graphicsQueueCreateInfo{};
        graphicsQueueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        graphicsQueueCreateInfo.queueFamilyIndex = m_graphicsQueueFamilyIndex;
        graphicsQueueCreateInfo.queueCount = m_graphicsQueueIndex + 1;
        graphicsQueueCreateInfo.pQueuePriorities = &graphicsQueuePriority;
        queueCreateInfos[queueCreateInfoCount++] = graphicsQueueCreateInfo;

        VkDeviceQueueCreateInfo transferQueueCreateInfo{};
        transferQueueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        transferQueueCreateInfo.queueFamilyIndex = m_transferQueueFamilyIndex;
        transferQueueCreateInfo.queueCount = m_transferQueueIndex + 1;
        transferQueueCreateInfo.pQueuePriorities = &transferQueuePriority;
        queueCreateInfos[queueCreateInfoCount++] = transferQueueCreateInfo;
    }

    VkPhysicalDeviceFeatures2 enabledFeatures2{};
    enabledFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    enabledFeatures2.features.fillModeNonSolid = m_supportsWireframePreview ? VK_TRUE : VK_FALSE;
    enabledFeatures2.features.samplerAnisotropy = m_supportsSamplerAnisotropy ? VK_TRUE : VK_FALSE;
    enabledFeatures2.features.multiDrawIndirect = m_supportsMultiDrawIndirect ? VK_TRUE : VK_FALSE;
    enabledFeatures2.features.drawIndirectFirstInstance = VK_TRUE;

    VkPhysicalDeviceVulkan11Features vulkan11Features{};
    vulkan11Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    vulkan11Features.shaderDrawParameters = VK_TRUE;

    VkPhysicalDeviceVulkan12Features vulkan12Features{};
    vulkan12Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    vulkan12Features.pNext = &vulkan11Features;
    vulkan12Features.timelineSemaphore = VK_TRUE;
    vulkan12Features.bufferDeviceAddress = VK_TRUE;
    if (m_supportsBindlessDescriptors) {
        vulkan12Features.descriptorIndexing = VK_TRUE;
        vulkan12Features.runtimeDescriptorArray = VK_TRUE;
        vulkan12Features.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
        vulkan12Features.descriptorBindingPartiallyBound = VK_TRUE;
    }

    VkPhysicalDeviceVulkan13Features vulkan13Features{};
    vulkan13Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    vulkan13Features.pNext = &vulkan12Features;
    vulkan13Features.dynamicRendering = VK_TRUE;
    vulkan13Features.synchronization2 = VK_TRUE;
    vulkan13Features.maintenance4 = VK_TRUE;

    VkPhysicalDeviceMemoryPriorityFeaturesEXT memoryPriorityFeatures{};
    memoryPriorityFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PRIORITY_FEATURES_EXT;
    memoryPriorityFeatures.pNext = &vulkan13Features;
    memoryPriorityFeatures.memoryPriority = VK_TRUE;
    enabledFeatures2.pNext = &memoryPriorityFeatures;

    VkDeviceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.pNext = &enabledFeatures2;
    createInfo.queueCreateInfoCount = queueCreateInfoCount;
    createInfo.pQueueCreateInfos = queueCreateInfos.data();
    createInfo.pEnabledFeatures = nullptr;
    createInfo.enabledExtensionCount = static_cast<uint32_t>(kDeviceExtensions.size());
    createInfo.ppEnabledExtensionNames = kDeviceExtensions.data();

    const VkResult result = vkCreateDevice(m_physicalDevice, &createInfo, nullptr, &m_device);
    if (result != VK_SUCCESS) {
        logVkFailure("vkCreateDevice", result);
        return false;
    }
    VOX_LOGI("render") << "device features enabled: dynamicRendering=1, synchronization2=1, maintenance4=1, "
        << "timelineSemaphore=1, bufferDeviceAddress=1, memoryPriority=1, shaderDrawParameters=1, drawIndirectFirstInstance=1, "
        << "multiDrawIndirect=" << (m_supportsMultiDrawIndirect ? 1 : 0)
        << ", descriptorIndexing=" << (m_supportsBindlessDescriptors ? 1 : 0)
        << ", runtimeDescriptorArray=" << (m_supportsBindlessDescriptors ? 1 : 0)
        << ", sampledImageArrayNonUniformIndexing=" << (m_supportsBindlessDescriptors ? 1 : 0)
        << ", descriptorBindingPartiallyBound=" << (m_supportsBindlessDescriptors ? 1 : 0)
        << "\n";
    VOX_LOGI("render") << "device extensions enabled: "
        << "VK_KHR_swapchain, VK_KHR_maintenance4, VK_KHR_timeline_semaphore, "
        << "VK_KHR_synchronization2, VK_KHR_dynamic_rendering, "
        << "VK_EXT_memory_budget, VK_EXT_memory_priority\n";
    if (m_supportsBindlessDescriptors) {
        VOX_LOGI("render") << "bindless descriptor support enabled (capacity="
            << m_bindlessTextureCapacity << ")\n";
    } else {
        VOX_LOGI("render") << "bindless descriptor support disabled (missing descriptor-indexing features)\n";
    }

    vkGetDeviceQueue(m_device, m_graphicsQueueFamilyIndex, m_graphicsQueueIndex, &m_graphicsQueue);
    vkGetDeviceQueue(m_device, m_transferQueueFamilyIndex, m_transferQueueIndex, &m_transferQueue);
    loadDebugUtilsFunctions();
    setObjectName(VK_OBJECT_TYPE_DEVICE, vkHandleToUint64(m_device), "renderer.device");
    setObjectName(VK_OBJECT_TYPE_QUEUE, vkHandleToUint64(m_graphicsQueue), "renderer.queue.graphics");
    setObjectName(VK_OBJECT_TYPE_QUEUE, vkHandleToUint64(m_transferQueue), "renderer.queue.transfer");

    VkPhysicalDeviceProperties deviceProperties{};
    vkGetPhysicalDeviceProperties(m_physicalDevice, &deviceProperties);
    m_uniformBufferAlignment = std::max<VkDeviceSize>(
        deviceProperties.limits.minUniformBufferOffsetAlignment,
        static_cast<VkDeviceSize>(16)
    );
    m_gpuTimestampPeriodNs = deviceProperties.limits.timestampPeriod;
    std::uint32_t queueFamilyPropertyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(m_physicalDevice, &queueFamilyPropertyCount, nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilyProperties(queueFamilyPropertyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(
        m_physicalDevice,
        &queueFamilyPropertyCount,
        queueFamilyProperties.data()
    );
    const bool graphicsQueueHasTimestamps =
        m_graphicsQueueFamilyIndex < queueFamilyProperties.size() &&
        queueFamilyProperties[m_graphicsQueueFamilyIndex].timestampValidBits > 0;
    m_gpuTimestampsSupported = graphicsQueueHasTimestamps && m_gpuTimestampPeriodNs > 0.0f;
    VOX_LOGI("render") << "GPU timestamps: supported=" << (m_gpuTimestampsSupported ? "yes" : "no")
        << ", periodNs=" << m_gpuTimestampPeriodNs
        << ", graphicsTimestampBits="
        << (graphicsQueueHasTimestamps
                ? queueFamilyProperties[m_graphicsQueueFamilyIndex].timestampValidBits
                : 0u)
        << "\n";

#if defined(VOXEL_HAS_VMA)
    if (m_vmaAllocator == VK_NULL_HANDLE) {
        VmaAllocatorCreateInfo allocatorCreateInfo{};
        allocatorCreateInfo.flags =
            VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT |
            VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT |
            VMA_ALLOCATOR_CREATE_EXT_MEMORY_PRIORITY_BIT;
        allocatorCreateInfo.physicalDevice = m_physicalDevice;
        allocatorCreateInfo.device = m_device;
        allocatorCreateInfo.instance = m_instance;
        allocatorCreateInfo.vulkanApiVersion = VK_API_VERSION_1_3;
        const VkResult allocatorResult = vmaCreateAllocator(&allocatorCreateInfo, &m_vmaAllocator);
        if (allocatorResult != VK_SUCCESS) {
            logVkFailure("vmaCreateAllocator", allocatorResult);
            return false;
        }
        VOX_LOGI("render") << "VMA allocator created: flags="
            << "BUFFER_DEVICE_ADDRESS|EXT_MEMORY_BUDGET|EXT_MEMORY_PRIORITY\n";
    }
#endif
    return true;
}

void Renderer::loadDebugUtilsFunctions() {
    m_setDebugUtilsObjectName = nullptr;
    m_cmdBeginDebugUtilsLabel = nullptr;
    m_cmdEndDebugUtilsLabel = nullptr;
    m_cmdInsertDebugUtilsLabel = nullptr;

    if (!m_debugUtilsEnabled || m_device == VK_NULL_HANDLE) {
        return;
    }

    m_setDebugUtilsObjectName = reinterpret_cast<PFN_vkSetDebugUtilsObjectNameEXT>(
        vkGetDeviceProcAddr(m_device, "vkSetDebugUtilsObjectNameEXT")
    );
    m_cmdBeginDebugUtilsLabel = reinterpret_cast<PFN_vkCmdBeginDebugUtilsLabelEXT>(
        vkGetDeviceProcAddr(m_device, "vkCmdBeginDebugUtilsLabelEXT")
    );
    m_cmdEndDebugUtilsLabel = reinterpret_cast<PFN_vkCmdEndDebugUtilsLabelEXT>(
        vkGetDeviceProcAddr(m_device, "vkCmdEndDebugUtilsLabelEXT")
    );
    m_cmdInsertDebugUtilsLabel = reinterpret_cast<PFN_vkCmdInsertDebugUtilsLabelEXT>(
        vkGetDeviceProcAddr(m_device, "vkCmdInsertDebugUtilsLabelEXT")
    );

    const bool namesReady = m_setDebugUtilsObjectName != nullptr;
    const bool labelsReady = m_cmdBeginDebugUtilsLabel != nullptr && m_cmdEndDebugUtilsLabel != nullptr;
    if (!namesReady && !labelsReady) {
        VOX_LOGI("render") << "debug utils extension enabled but debug functions were not loaded\n";
        m_debugUtilsEnabled = false;
        return;
    }

    VOX_LOGI("render") << "debug utils loaded: objectNames=" << (namesReady ? "yes" : "no")
        << ", cmdLabels=" << (labelsReady ? "yes" : "no")
        << ", cmdInsertLabel=" << (m_cmdInsertDebugUtilsLabel != nullptr ? "yes" : "no")
        << "\n";
}

void Renderer::setObjectName(VkObjectType objectType, uint64_t objectHandle, const char* name) const {
    if (m_setDebugUtilsObjectName == nullptr || m_device == VK_NULL_HANDLE || objectHandle == 0 || name == nullptr) {
        return;
    }
    VkDebugUtilsObjectNameInfoEXT nameInfo{};
    nameInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
    nameInfo.objectType = objectType;
    nameInfo.objectHandle = objectHandle;
    nameInfo.pObjectName = name;
    m_setDebugUtilsObjectName(m_device, &nameInfo);
}

void Renderer::beginDebugLabel(
    VkCommandBuffer commandBuffer,
    const char* name,
    float r,
    float g,
    float b,
    float a
) const {
    if (m_cmdBeginDebugUtilsLabel == nullptr || commandBuffer == VK_NULL_HANDLE || name == nullptr) {
        return;
    }
    VkDebugUtilsLabelEXT label{};
    label.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
    label.pLabelName = name;
    label.color[0] = r;
    label.color[1] = g;
    label.color[2] = b;
    label.color[3] = a;
    m_cmdBeginDebugUtilsLabel(commandBuffer, &label);
}

void Renderer::endDebugLabel(VkCommandBuffer commandBuffer) const {
    if (m_cmdEndDebugUtilsLabel == nullptr || commandBuffer == VK_NULL_HANDLE) {
        return;
    }
    m_cmdEndDebugUtilsLabel(commandBuffer);
}

void Renderer::insertDebugLabel(
    VkCommandBuffer commandBuffer,
    const char* name,
    float r,
    float g,
    float b,
    float a
) const {
    if (m_cmdInsertDebugUtilsLabel == nullptr || commandBuffer == VK_NULL_HANDLE || name == nullptr) {
        return;
    }
    VkDebugUtilsLabelEXT label{};
    label.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
    label.pLabelName = name;
    label.color[0] = r;
    label.color[1] = g;
    label.color[2] = b;
    label.color[3] = a;
    m_cmdInsertDebugUtilsLabel(commandBuffer, &label);
}

bool Renderer::createTimelineSemaphore() {
    if (m_renderTimelineSemaphore != VK_NULL_HANDLE) {
        return true;
    }

    VkSemaphoreTypeCreateInfo timelineCreateInfo{};
    timelineCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    timelineCreateInfo.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    timelineCreateInfo.initialValue = 0;

    VkSemaphoreCreateInfo semaphoreCreateInfo{};
    semaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    semaphoreCreateInfo.pNext = &timelineCreateInfo;

    const VkResult result = vkCreateSemaphore(m_device, &semaphoreCreateInfo, nullptr, &m_renderTimelineSemaphore);
    if (result != VK_SUCCESS) {
        logVkFailure("vkCreateSemaphore(timeline)", result);
        return false;
    }
    setObjectName(
        VK_OBJECT_TYPE_SEMAPHORE,
        vkHandleToUint64(m_renderTimelineSemaphore),
        "renderer.timeline.render"
    );

    m_frameTimelineValues.fill(0);
    m_pendingTransferTimelineValue = 0;
    m_currentChunkReadyTimelineValue = 0;
    m_transferCommandBufferInFlightValue = 0;
    m_lastGraphicsTimelineValue = 0;
    m_nextTimelineValue = 1;
    return true;
}

bool Renderer::createUploadRingBuffer() {
    // FrameArena layer A foundation: one persistently mapped upload arena per frame-in-flight.
    FrameArenaConfig config{};
    config.uploadBytesPerFrame = 1024ull * 1024ull * 64ull;
    config.frameCount = kMaxFramesInFlight;
    config.uploadUsage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                         VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                         VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                         VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    const bool ok = m_frameArena.init(
        &m_bufferAllocator,
        m_physicalDevice,
        m_device,
        config
#if defined(VOXEL_HAS_VMA)
        , m_vmaAllocator
#endif
    );
    if (!ok) {
        VOX_LOGE("render") << "frame arena init failed\n";
    } else {
        const BufferHandle uploadHandle = m_frameArena.uploadBufferHandle();
        if (uploadHandle != kInvalidBufferHandle) {
            const VkBuffer uploadBuffer = m_bufferAllocator.getBuffer(uploadHandle);
            if (uploadBuffer != VK_NULL_HANDLE) {
                setObjectName(
                    VK_OBJECT_TYPE_BUFFER,
                    vkHandleToUint64(uploadBuffer),
                    "framearena.uploadRing"
                );
            }
        }
    }
    return ok;
}

bool Renderer::createTransferResources() {
    if (m_transferCommandPool != VK_NULL_HANDLE && m_transferCommandBuffer != VK_NULL_HANDLE) {
        return true;
    }

    VkCommandPoolCreateInfo poolCreateInfo{};
    poolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolCreateInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolCreateInfo.queueFamilyIndex = m_transferQueueFamilyIndex;

    const VkResult poolResult = vkCreateCommandPool(m_device, &poolCreateInfo, nullptr, &m_transferCommandPool);
    if (poolResult != VK_SUCCESS) {
        logVkFailure("vkCreateCommandPool(transfer)", poolResult);
        return false;
    }
    setObjectName(
        VK_OBJECT_TYPE_COMMAND_POOL,
        vkHandleToUint64(m_transferCommandPool),
        "renderer.transfer.commandPool"
    );

    VkCommandBufferAllocateInfo allocateInfo{};
    allocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocateInfo.commandPool = m_transferCommandPool;
    allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocateInfo.commandBufferCount = 1;

    const VkResult commandBufferResult = vkAllocateCommandBuffers(m_device, &allocateInfo, &m_transferCommandBuffer);
    if (commandBufferResult != VK_SUCCESS) {
        logVkFailure("vkAllocateCommandBuffers(transfer)", commandBufferResult);
        vkDestroyCommandPool(m_device, m_transferCommandPool, nullptr);
        m_transferCommandPool = VK_NULL_HANDLE;
        return false;
    }
    setObjectName(
        VK_OBJECT_TYPE_COMMAND_BUFFER,
        vkHandleToUint64(m_transferCommandBuffer),
        "renderer.transfer.commandBuffer"
    );

    return true;
}

bool Renderer::createPipeBuffers() {
    if (m_pipeVertexBufferHandle != kInvalidBufferHandle &&
        m_pipeIndexBufferHandle != kInvalidBufferHandle &&
        m_transportVertexBufferHandle != kInvalidBufferHandle &&
        m_transportIndexBufferHandle != kInvalidBufferHandle &&
        m_grassBillboardVertexBufferHandle != kInvalidBufferHandle &&
        m_grassBillboardIndexBufferHandle != kInvalidBufferHandle) {
        return true;
    }

    const PipeMeshData pipeMesh = buildPipeCylinderMesh();
    const PipeMeshData transportMesh = buildTransportBoxMesh();
    if (pipeMesh.vertices.empty() || pipeMesh.indices.empty()) {
        VOX_LOGE("render") << "pipe cylinder mesh build failed\n";
        return false;
    }
    if (transportMesh.vertices.empty() || transportMesh.indices.empty()) {
        VOX_LOGE("render") << "transport box mesh build failed\n";
        return false;
    }

    auto createMeshBuffers = [&](const PipeMeshData& mesh, BufferHandle& outVertex, BufferHandle& outIndex, const char* label) -> bool {
        if (outVertex != kInvalidBufferHandle || outIndex != kInvalidBufferHandle) {
            return true;
        }
        BufferCreateDesc vertexCreateDesc{};
        vertexCreateDesc.size = static_cast<VkDeviceSize>(mesh.vertices.size() * sizeof(PipeMeshData::Vertex));
        vertexCreateDesc.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
        vertexCreateDesc.memoryProperties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        vertexCreateDesc.initialData = mesh.vertices.data();
        outVertex = m_bufferAllocator.createBuffer(vertexCreateDesc);
        if (outVertex == kInvalidBufferHandle) {
            VOX_LOGE("render") << label << " vertex buffer allocation failed\n";
            return false;
        }
        const VkBuffer vertexBuffer = m_bufferAllocator.getBuffer(outVertex);
        if (vertexBuffer != VK_NULL_HANDLE) {
            const std::string vertexName = std::string("mesh.") + label + ".vertex";
            setObjectName(VK_OBJECT_TYPE_BUFFER, vkHandleToUint64(vertexBuffer), vertexName.c_str());
        }

        BufferCreateDesc indexCreateDesc{};
        indexCreateDesc.size = static_cast<VkDeviceSize>(mesh.indices.size() * sizeof(std::uint32_t));
        indexCreateDesc.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
        indexCreateDesc.memoryProperties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        indexCreateDesc.initialData = mesh.indices.data();
        outIndex = m_bufferAllocator.createBuffer(indexCreateDesc);
        if (outIndex == kInvalidBufferHandle) {
            VOX_LOGE("render") << label << " index buffer allocation failed\n";
            m_bufferAllocator.destroyBuffer(outVertex);
            outVertex = kInvalidBufferHandle;
            return false;
        }
        const VkBuffer indexBuffer = m_bufferAllocator.getBuffer(outIndex);
        if (indexBuffer != VK_NULL_HANDLE) {
            const std::string indexName = std::string("mesh.") + label + ".index";
            setObjectName(VK_OBJECT_TYPE_BUFFER, vkHandleToUint64(indexBuffer), indexName.c_str());
        }
        return true;
    };

    if (!createMeshBuffers(pipeMesh, m_pipeVertexBufferHandle, m_pipeIndexBufferHandle, "pipe")) {
        return false;
    }
    if (!createMeshBuffers(
            transportMesh,
            m_transportVertexBufferHandle,
            m_transportIndexBufferHandle,
            "transport"
        )) {
        VOX_LOGE("render") << "transport mesh buffer setup failed\n";
        return false;
    }

    if (m_grassBillboardVertexBufferHandle == kInvalidBufferHandle ||
        m_grassBillboardIndexBufferHandle == kInvalidBufferHandle) {
        constexpr std::array<GrassBillboardVertex, 8> kGrassBillboardVertices = {{
            // Plane 0 (X axis).
            GrassBillboardVertex{{-0.38f, 0.0f}, {0.0f, 1.0f}, 0.0f},
            GrassBillboardVertex{{ 0.38f, 0.0f}, {1.0f, 1.0f}, 0.0f},
            GrassBillboardVertex{{-0.38f, 0.88f}, {0.0f, 0.0f}, 0.0f},
            GrassBillboardVertex{{ 0.38f, 0.88f}, {1.0f, 0.0f}, 0.0f},
            // Plane 1 (Z axis).
            GrassBillboardVertex{{-0.38f, 0.0f}, {0.0f, 1.0f}, 1.0f},
            GrassBillboardVertex{{ 0.38f, 0.0f}, {1.0f, 1.0f}, 1.0f},
            GrassBillboardVertex{{-0.38f, 0.88f}, {0.0f, 0.0f}, 1.0f},
            GrassBillboardVertex{{ 0.38f, 0.88f}, {1.0f, 0.0f}, 1.0f},
        }};
        constexpr std::array<std::uint32_t, 12> kGrassBillboardIndices = {{
            0u, 1u, 2u, 2u, 1u, 3u,
            4u, 5u, 6u, 6u, 5u, 7u
        }};

        BufferCreateDesc grassVertexCreateDesc{};
        grassVertexCreateDesc.size = static_cast<VkDeviceSize>(kGrassBillboardVertices.size() * sizeof(GrassBillboardVertex));
        grassVertexCreateDesc.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
        grassVertexCreateDesc.memoryProperties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        grassVertexCreateDesc.initialData = kGrassBillboardVertices.data();
        m_grassBillboardVertexBufferHandle = m_bufferAllocator.createBuffer(grassVertexCreateDesc);
        if (m_grassBillboardVertexBufferHandle == kInvalidBufferHandle) {
            VOX_LOGE("render") << "grass billboard vertex buffer allocation failed\n";
            return false;
        }
        {
            const VkBuffer grassVertexBuffer = m_bufferAllocator.getBuffer(m_grassBillboardVertexBufferHandle);
            if (grassVertexBuffer != VK_NULL_HANDLE) {
                setObjectName(VK_OBJECT_TYPE_BUFFER, vkHandleToUint64(grassVertexBuffer), "mesh.grassBillboard.vertex");
            }
        }

        BufferCreateDesc grassIndexCreateDesc{};
        grassIndexCreateDesc.size = static_cast<VkDeviceSize>(kGrassBillboardIndices.size() * sizeof(std::uint32_t));
        grassIndexCreateDesc.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
        grassIndexCreateDesc.memoryProperties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        grassIndexCreateDesc.initialData = kGrassBillboardIndices.data();
        m_grassBillboardIndexBufferHandle = m_bufferAllocator.createBuffer(grassIndexCreateDesc);
        if (m_grassBillboardIndexBufferHandle == kInvalidBufferHandle) {
            VOX_LOGE("render") << "grass billboard index buffer allocation failed\n";
            m_bufferAllocator.destroyBuffer(m_grassBillboardVertexBufferHandle);
            m_grassBillboardVertexBufferHandle = kInvalidBufferHandle;
            return false;
        }
        {
            const VkBuffer grassIndexBuffer = m_bufferAllocator.getBuffer(m_grassBillboardIndexBufferHandle);
            if (grassIndexBuffer != VK_NULL_HANDLE) {
                setObjectName(VK_OBJECT_TYPE_BUFFER, vkHandleToUint64(grassIndexBuffer), "mesh.grassBillboard.index");
            }
        }
        m_grassBillboardIndexCount = static_cast<uint32_t>(kGrassBillboardIndices.size());
    }

    m_pipeIndexCount = static_cast<uint32_t>(pipeMesh.indices.size());
    m_transportIndexCount = static_cast<uint32_t>(transportMesh.indices.size());
    return true;
}

bool Renderer::createPreviewBuffers() {
    if (m_previewVertexBufferHandle != kInvalidBufferHandle && m_previewIndexBufferHandle != kInvalidBufferHandle) {
        return true;
    }

    const world::ChunkMeshData addMesh = buildSingleVoxelPreviewMesh(0, 0, 0, 3, 250);
    const world::ChunkMeshData removeMesh = buildSingleVoxelPreviewMesh(0, 0, 0, 3, 251);
    if (addMesh.vertices.empty() || addMesh.indices.empty() || removeMesh.vertices.empty() || removeMesh.indices.empty()) {
        VOX_LOGE("render") << "preview mesh build failed\n";
        return false;
    }

    world::ChunkMeshData mesh{};
    mesh.vertices = addMesh.vertices;
    mesh.indices = addMesh.indices;
    mesh.vertices.insert(mesh.vertices.end(), removeMesh.vertices.begin(), removeMesh.vertices.end());
    mesh.indices.reserve(mesh.indices.size() + removeMesh.indices.size());
    const uint32_t removeBaseVertex = static_cast<uint32_t>(addMesh.vertices.size());
    for (const uint32_t index : removeMesh.indices) {
        mesh.indices.push_back(index + removeBaseVertex);
    }

    BufferCreateDesc vertexCreateDesc{};
    vertexCreateDesc.size = static_cast<VkDeviceSize>(mesh.vertices.size() * sizeof(world::PackedVoxelVertex));
    vertexCreateDesc.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    vertexCreateDesc.memoryProperties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    vertexCreateDesc.initialData = mesh.vertices.data();
    m_previewVertexBufferHandle = m_bufferAllocator.createBuffer(vertexCreateDesc);
    if (m_previewVertexBufferHandle == kInvalidBufferHandle) {
        VOX_LOGE("render") << "preview vertex buffer allocation failed\n";
        return false;
    }
    {
        const VkBuffer previewVertexBuffer = m_bufferAllocator.getBuffer(m_previewVertexBufferHandle);
        if (previewVertexBuffer != VK_NULL_HANDLE) {
            setObjectName(VK_OBJECT_TYPE_BUFFER, vkHandleToUint64(previewVertexBuffer), "preview.voxel.vertex");
        }
    }

    BufferCreateDesc indexCreateDesc{};
    indexCreateDesc.size = static_cast<VkDeviceSize>(mesh.indices.size() * sizeof(std::uint32_t));
    indexCreateDesc.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
    indexCreateDesc.memoryProperties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    indexCreateDesc.initialData = mesh.indices.data();
    m_previewIndexBufferHandle = m_bufferAllocator.createBuffer(indexCreateDesc);
    if (m_previewIndexBufferHandle == kInvalidBufferHandle) {
        VOX_LOGE("render") << "preview index buffer allocation failed\n";
        m_bufferAllocator.destroyBuffer(m_previewVertexBufferHandle);
        m_previewVertexBufferHandle = kInvalidBufferHandle;
        return false;
    }
    {
        const VkBuffer previewIndexBuffer = m_bufferAllocator.getBuffer(m_previewIndexBufferHandle);
        if (previewIndexBuffer != VK_NULL_HANDLE) {
            setObjectName(VK_OBJECT_TYPE_BUFFER, vkHandleToUint64(previewIndexBuffer), "preview.voxel.index");
        }
    }

    m_previewIndexCount = static_cast<uint32_t>(mesh.indices.size());
    return true;
}

bool Renderer::createEnvironmentResources() {
    if (!createDiffuseTextureResources()) {
        VOX_LOGE("render") << "diffuse texture creation failed\n";
        return false;
    }
    VOX_LOGI("render") << "environment uses procedural sky + SH irradiance + diffuse albedo texture\n";
    return true;
}

bool Renderer::createDiffuseTextureResources() {
    bool hasDiffuseAllocation = (m_diffuseTextureMemory != VK_NULL_HANDLE);
#if defined(VOXEL_HAS_VMA)
    if (m_vmaAllocator != VK_NULL_HANDLE) {
        hasDiffuseAllocation = (m_diffuseTextureAllocation != VK_NULL_HANDLE);
    }
#endif
    if (
        m_diffuseTextureImage != VK_NULL_HANDLE &&
        hasDiffuseAllocation &&
        m_diffuseTextureImageView != VK_NULL_HANDLE &&
        m_diffuseTextureSampler != VK_NULL_HANDLE
    ) {
        return true;
    }

    constexpr uint32_t kTileSize = 16;
    constexpr uint32_t kTextureTilesX = 5;
    constexpr uint32_t kTextureTilesY = 1;
    constexpr uint32_t kTextureWidth = kTileSize * kTextureTilesX;
    constexpr uint32_t kTextureHeight = kTileSize * kTextureTilesY;
    constexpr VkFormat kTextureFormat = VK_FORMAT_R8G8B8A8_UNORM;
    uint32_t diffuseMipLevels = 1u;
    for (uint32_t tileExtent = kTileSize; tileExtent > 1u; tileExtent >>= 1u) {
        ++diffuseMipLevels;
    }
    constexpr VkDeviceSize kTextureBytes = kTextureWidth * kTextureHeight * 4;

    std::vector<std::uint8_t> pixels(static_cast<size_t>(kTextureBytes), 0);
    auto hash8 = [](uint32_t x, uint32_t y, uint32_t seed) -> std::uint8_t {
        uint32_t h = x * 374761393u;
        h += y * 668265263u;
        h += seed * 2246822519u;
        h = (h ^ (h >> 13u)) * 1274126177u;
        return static_cast<std::uint8_t>((h >> 24u) & 0xFFu);
    };
    auto writePixel = [&](uint32_t px, uint32_t py, std::uint8_t r, std::uint8_t g, std::uint8_t b, std::uint8_t a = 255u) {
        const size_t i = static_cast<size_t>((py * kTextureWidth + px) * 4u);
        pixels[i + 0] = r;
        pixels[i + 1] = g;
        pixels[i + 2] = b;
        pixels[i + 3] = a;
    };

    for (uint32_t y = 0; y < kTextureHeight; ++y) {
        for (uint32_t x = 0; x < kTextureWidth; ++x) {
            const uint32_t tileIndex = x / kTileSize;
            const uint32_t localX = x % kTileSize;
            const uint32_t localY = y % kTileSize;
            const std::uint8_t noiseA = hash8(localX, localY, tileIndex + 11u);
            const std::uint8_t noiseB = hash8(localX, localY, tileIndex + 37u);

            std::uint8_t r = 128u;
            std::uint8_t g = 128u;
            std::uint8_t b = 128u;
            if (tileIndex == 0u) {
                // Stone.
                const int tone = 108 + static_cast<int>(noiseA % 34u) - 17;
                r = static_cast<std::uint8_t>(std::clamp(tone, 72, 146));
                g = static_cast<std::uint8_t>(std::clamp(tone - 5, 66, 140));
                b = static_cast<std::uint8_t>(std::clamp(tone - 10, 58, 132));
            } else if (tileIndex == 1u) {
                // Dirt.
                const int warm = 94 + static_cast<int>(noiseA % 28u) - 14;
                const int cool = 68 + static_cast<int>(noiseB % 20u) - 10;
                r = static_cast<std::uint8_t>(std::clamp(warm + 20, 70, 138));
                g = static_cast<std::uint8_t>(std::clamp(warm - 2, 48, 112));
                b = static_cast<std::uint8_t>(std::clamp(cool - 8, 26, 84));
            } else if (tileIndex == 2u) {
                // Grass.
                const int green = 118 + static_cast<int>(noiseA % 32u) - 16;
                r = static_cast<std::uint8_t>(std::clamp(52 + static_cast<int>(noiseB % 18u) - 9, 34, 74));
                g = static_cast<std::uint8_t>(std::clamp(green, 82, 154));
                b = static_cast<std::uint8_t>(std::clamp(44 + static_cast<int>(noiseA % 14u) - 7, 26, 64));
            } else {
                if (tileIndex == 3u) {
                    // Wood.
                    const int stripe = ((localX / 3u) + (localY / 5u)) % 3u;
                    const int base = (stripe == 0) ? 112 : (stripe == 1 ? 96 : 84);
                    const int grain = static_cast<int>(noiseA % 16u) - 8;
                    r = static_cast<std::uint8_t>(std::clamp(base + 34 + grain, 78, 168));
                    g = static_cast<std::uint8_t>(std::clamp(base + 12 + grain, 56, 136));
                    b = static_cast<std::uint8_t>(std::clamp(base - 6 + (grain / 2), 36, 110));
                } else {
                    // Billboard grass sprite (transparent background).
                    const int rowFromBottom = static_cast<int>(kTileSize - 1u - localY);
                    const float growthT = std::clamp(static_cast<float>(rowFromBottom) / static_cast<float>(kTileSize - 1u), 0.0f, 1.0f);
                    const int center = static_cast<int>(kTileSize / 2u);
                    const int leftBlade = center - 3 + static_cast<int>(growthT * 2.0f);
                    const int rightBlade = center + 2 - static_cast<int>(growthT * 2.0f);
                    const bool centerBlade = (std::abs(static_cast<int>(localX) - center) <= 1) && rowFromBottom <= 13;
                    const bool bladeL = (std::abs(static_cast<int>(localX) - leftBlade) <= 1) && rowFromBottom <= 10;
                    const bool bladeR = (std::abs(static_cast<int>(localX) - rightBlade) <= 1) && rowFromBottom <= 11;
                    const bool baseTuft = (rowFromBottom <= 3) && (std::abs(static_cast<int>(localX) - center) <= 4);
                    const bool isBlade = centerBlade || bladeL || bladeR || baseTuft;
                    if (!isBlade) {
                        writePixel(x, y, 0u, 0u, 0u, 0u);
                        continue;
                    }

                    const int green = 132 + static_cast<int>(noiseA % 52u) - 18;
                    const int red = 48 + static_cast<int>(noiseB % 28u) - 10;
                    const int blue = 34 + static_cast<int>(noiseA % 18u) - 6;
                    r = static_cast<std::uint8_t>(std::clamp(red, 22, 88));
                    g = static_cast<std::uint8_t>(std::clamp(green, 92, 196));
                    b = static_cast<std::uint8_t>(std::clamp(blue, 16, 84));
                    const std::uint8_t alpha = static_cast<std::uint8_t>(std::clamp(160 + static_cast<int>(noiseB % 72u), 140, 240));
                    writePixel(x, y, r, g, b, alpha);
                    continue;
                }
            }
            writePixel(x, y, r, g, b);
        }
    }

    VkBuffer stagingBuffer = VK_NULL_HANDLE;
    VkDeviceMemory stagingMemory = VK_NULL_HANDLE;
    VkBufferCreateInfo stagingCreateInfo{};
    stagingCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    stagingCreateInfo.size = kTextureBytes;
    stagingCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    stagingCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkResult result = vkCreateBuffer(m_device, &stagingCreateInfo, nullptr, &stagingBuffer);
    if (result != VK_SUCCESS) {
        logVkFailure("vkCreateBuffer(diffuseStaging)", result);
        return false;
    }
    setObjectName(VK_OBJECT_TYPE_BUFFER, vkHandleToUint64(stagingBuffer), "diffuse.staging.buffer");

    VkMemoryRequirements stagingMemReq{};
    vkGetBufferMemoryRequirements(m_device, stagingBuffer, &stagingMemReq);
    uint32_t memoryTypeIndex = findMemoryTypeIndex(
        m_physicalDevice,
        stagingMemReq.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );
    if (memoryTypeIndex == std::numeric_limits<uint32_t>::max()) {
        VOX_LOGI("render") << "no staging memory type for diffuse texture\n";
        vkDestroyBuffer(m_device, stagingBuffer, nullptr);
        return false;
    }

    VkMemoryAllocateInfo stagingAllocInfo{};
    stagingAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    stagingAllocInfo.allocationSize = stagingMemReq.size;
    stagingAllocInfo.memoryTypeIndex = memoryTypeIndex;
    result = vkAllocateMemory(m_device, &stagingAllocInfo, nullptr, &stagingMemory);
    if (result != VK_SUCCESS) {
        logVkFailure("vkAllocateMemory(diffuseStaging)", result);
        vkDestroyBuffer(m_device, stagingBuffer, nullptr);
        return false;
    }
    result = vkBindBufferMemory(m_device, stagingBuffer, stagingMemory, 0);
    if (result != VK_SUCCESS) {
        logVkFailure("vkBindBufferMemory(diffuseStaging)", result);
        vkFreeMemory(m_device, stagingMemory, nullptr);
        vkDestroyBuffer(m_device, stagingBuffer, nullptr);
        return false;
    }

    void* mapped = nullptr;
    result = vkMapMemory(m_device, stagingMemory, 0, kTextureBytes, 0, &mapped);
    if (result != VK_SUCCESS || mapped == nullptr) {
        logVkFailure("vkMapMemory(diffuseStaging)", result);
        vkFreeMemory(m_device, stagingMemory, nullptr);
        vkDestroyBuffer(m_device, stagingBuffer, nullptr);
        return false;
    }
    std::memcpy(mapped, pixels.data(), static_cast<size_t>(kTextureBytes));
    vkUnmapMemory(m_device, stagingMemory);

    VkImageCreateInfo imageCreateInfo{};
    imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
    imageCreateInfo.format = kTextureFormat;
    imageCreateInfo.extent = {kTextureWidth, kTextureHeight, 1};
    imageCreateInfo.mipLevels = diffuseMipLevels;
    imageCreateInfo.arrayLayers = 1;
    imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageCreateInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    m_diffuseTextureMemory = VK_NULL_HANDLE;
#if defined(VOXEL_HAS_VMA)
    m_diffuseTextureAllocation = VK_NULL_HANDLE;
    if (m_vmaAllocator != VK_NULL_HANDLE) {
        VmaAllocationCreateInfo allocationCreateInfo{};
        allocationCreateInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        allocationCreateInfo.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        result = vmaCreateImage(
            m_vmaAllocator,
            &imageCreateInfo,
            &allocationCreateInfo,
            &m_diffuseTextureImage,
            &m_diffuseTextureAllocation,
            nullptr
        );
        if (result != VK_SUCCESS) {
            logVkFailure("vmaCreateImage(diffuseTexture)", result);
            vkFreeMemory(m_device, stagingMemory, nullptr);
            vkDestroyBuffer(m_device, stagingBuffer, nullptr);
            return false;
        }
    } else
#endif
    {
        result = vkCreateImage(m_device, &imageCreateInfo, nullptr, &m_diffuseTextureImage);
        if (result != VK_SUCCESS) {
            logVkFailure("vkCreateImage(diffuseTexture)", result);
            vkFreeMemory(m_device, stagingMemory, nullptr);
            vkDestroyBuffer(m_device, stagingBuffer, nullptr);
            return false;
        }

        VkMemoryRequirements imageMemReq{};
        vkGetImageMemoryRequirements(m_device, m_diffuseTextureImage, &imageMemReq);
        memoryTypeIndex = findMemoryTypeIndex(
            m_physicalDevice,
            imageMemReq.memoryTypeBits,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        );
        if (memoryTypeIndex == std::numeric_limits<uint32_t>::max()) {
            VOX_LOGI("render") << "no device-local memory for diffuse texture\n";
            vkDestroyImage(m_device, m_diffuseTextureImage, nullptr);
            m_diffuseTextureImage = VK_NULL_HANDLE;
            vkFreeMemory(m_device, stagingMemory, nullptr);
            vkDestroyBuffer(m_device, stagingBuffer, nullptr);
            return false;
        }

        VkMemoryAllocateInfo imageAllocInfo{};
        imageAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        imageAllocInfo.allocationSize = imageMemReq.size;
        imageAllocInfo.memoryTypeIndex = memoryTypeIndex;
        result = vkAllocateMemory(m_device, &imageAllocInfo, nullptr, &m_diffuseTextureMemory);
        if (result != VK_SUCCESS) {
            logVkFailure("vkAllocateMemory(diffuseTexture)", result);
            vkDestroyImage(m_device, m_diffuseTextureImage, nullptr);
            m_diffuseTextureImage = VK_NULL_HANDLE;
            vkFreeMemory(m_device, stagingMemory, nullptr);
            vkDestroyBuffer(m_device, stagingBuffer, nullptr);
            return false;
        }
        result = vkBindImageMemory(m_device, m_diffuseTextureImage, m_diffuseTextureMemory, 0);
        if (result != VK_SUCCESS) {
            logVkFailure("vkBindImageMemory(diffuseTexture)", result);
            destroyDiffuseTextureResources();
            vkFreeMemory(m_device, stagingMemory, nullptr);
            vkDestroyBuffer(m_device, stagingBuffer, nullptr);
            return false;
        }
    }
    setObjectName(VK_OBJECT_TYPE_IMAGE, vkHandleToUint64(m_diffuseTextureImage), "diffuse.albedo.image");

    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkCommandPoolCreateInfo poolCreateInfo{};
    poolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolCreateInfo.queueFamilyIndex = m_graphicsQueueFamilyIndex;
    poolCreateInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    result = vkCreateCommandPool(m_device, &poolCreateInfo, nullptr, &commandPool);
    if (result != VK_SUCCESS) {
        logVkFailure("vkCreateCommandPool(diffuseUpload)", result);
        destroyDiffuseTextureResources();
        vkFreeMemory(m_device, stagingMemory, nullptr);
        vkDestroyBuffer(m_device, stagingBuffer, nullptr);
        return false;
    }
    setObjectName(VK_OBJECT_TYPE_COMMAND_POOL, vkHandleToUint64(commandPool), "diffuse.upload.commandPool");

    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    VkCommandBufferAllocateInfo cmdAllocInfo{};
    cmdAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdAllocInfo.commandPool = commandPool;
    cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAllocInfo.commandBufferCount = 1;
    result = vkAllocateCommandBuffers(m_device, &cmdAllocInfo, &commandBuffer);
    if (result != VK_SUCCESS) {
        logVkFailure("vkAllocateCommandBuffers(diffuseUpload)", result);
        vkDestroyCommandPool(m_device, commandPool, nullptr);
        destroyDiffuseTextureResources();
        vkFreeMemory(m_device, stagingMemory, nullptr);
        vkDestroyBuffer(m_device, stagingBuffer, nullptr);
        return false;
    }
    setObjectName(VK_OBJECT_TYPE_COMMAND_BUFFER, vkHandleToUint64(commandBuffer), "diffuse.upload.commandBuffer");

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    result = vkBeginCommandBuffer(commandBuffer, &beginInfo);
    if (result != VK_SUCCESS) {
        logVkFailure("vkBeginCommandBuffer(diffuseUpload)", result);
        vkDestroyCommandPool(m_device, commandPool, nullptr);
        destroyDiffuseTextureResources();
        vkFreeMemory(m_device, stagingMemory, nullptr);
        vkDestroyBuffer(m_device, stagingBuffer, nullptr);
        return false;
    }

    transitionImageLayout(
        commandBuffer,
        m_diffuseTextureImage,
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_PIPELINE_STAGE_2_NONE,
        VK_ACCESS_2_NONE,
        VK_PIPELINE_STAGE_2_TRANSFER_BIT,
        VK_ACCESS_2_TRANSFER_WRITE_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT,
        0u,
        1u,
        0u,
        diffuseMipLevels
    );

    VkBufferImageCopy copyRegion{};
    copyRegion.bufferOffset = 0;
    copyRegion.bufferRowLength = 0;
    copyRegion.bufferImageHeight = 0;
    copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copyRegion.imageSubresource.mipLevel = 0;
    copyRegion.imageSubresource.baseArrayLayer = 0;
    copyRegion.imageSubresource.layerCount = 1;
    copyRegion.imageOffset = {0, 0, 0};
    copyRegion.imageExtent = {kTextureWidth, kTextureHeight, 1};
    vkCmdCopyBufferToImage(
        commandBuffer,
        stagingBuffer,
        m_diffuseTextureImage,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1,
        &copyRegion
    );

    for (uint32_t mipLevel = 1u; mipLevel < diffuseMipLevels; ++mipLevel) {
        const uint32_t srcMip = mipLevel - 1u;
        transitionImageLayout(
            commandBuffer,
            m_diffuseTextureImage,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            VK_ACCESS_2_TRANSFER_READ_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT,
            0u,
            1u,
            srcMip,
            1u
        );

        const int32_t srcTileWidth = static_cast<int32_t>(std::max(1u, kTileSize >> srcMip));
        const int32_t srcTileHeight = static_cast<int32_t>(std::max(1u, kTileSize >> srcMip));
        const int32_t dstTileWidth = static_cast<int32_t>(std::max(1u, kTileSize >> mipLevel));
        const int32_t dstTileHeight = static_cast<int32_t>(std::max(1u, kTileSize >> mipLevel));

        for (uint32_t tileY = 0; tileY < kTextureTilesY; ++tileY) {
            for (uint32_t tileX = 0; tileX < kTextureTilesX; ++tileX) {
                VkImageBlit blitRegion{};
                blitRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                blitRegion.srcSubresource.mipLevel = srcMip;
                blitRegion.srcSubresource.baseArrayLayer = 0;
                blitRegion.srcSubresource.layerCount = 1;
                blitRegion.srcOffsets[0] = {
                    static_cast<int32_t>(tileX) * srcTileWidth,
                    static_cast<int32_t>(tileY) * srcTileHeight,
                    0
                };
                blitRegion.srcOffsets[1] = {
                    blitRegion.srcOffsets[0].x + srcTileWidth,
                    blitRegion.srcOffsets[0].y + srcTileHeight,
                    1
                };

                blitRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                blitRegion.dstSubresource.mipLevel = mipLevel;
                blitRegion.dstSubresource.baseArrayLayer = 0;
                blitRegion.dstSubresource.layerCount = 1;
                blitRegion.dstOffsets[0] = {
                    static_cast<int32_t>(tileX) * dstTileWidth,
                    static_cast<int32_t>(tileY) * dstTileHeight,
                    0
                };
                blitRegion.dstOffsets[1] = {
                    blitRegion.dstOffsets[0].x + dstTileWidth,
                    blitRegion.dstOffsets[0].y + dstTileHeight,
                    1
                };

                vkCmdBlitImage(
                    commandBuffer,
                    m_diffuseTextureImage,
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    m_diffuseTextureImage,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    1,
                    &blitRegion,
                    VK_FILTER_LINEAR
                );
            }
        }
    }

    if (diffuseMipLevels > 1u) {
        transitionImageLayout(
            commandBuffer,
            m_diffuseTextureImage,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            VK_ACCESS_2_TRANSFER_READ_BIT,
            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
            VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT,
            0u,
            1u,
            0u,
            diffuseMipLevels - 1u
        );
    }

    transitionImageLayout(
        commandBuffer,
        m_diffuseTextureImage,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_2_TRANSFER_BIT,
        VK_ACCESS_2_TRANSFER_WRITE_BIT,
        VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
        VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT,
        0u,
        1u,
        diffuseMipLevels - 1u,
        1u
    );

    result = vkEndCommandBuffer(commandBuffer);
    if (result != VK_SUCCESS) {
        logVkFailure("vkEndCommandBuffer(diffuseUpload)", result);
        vkDestroyCommandPool(m_device, commandPool, nullptr);
        destroyDiffuseTextureResources();
        vkFreeMemory(m_device, stagingMemory, nullptr);
        vkDestroyBuffer(m_device, stagingBuffer, nullptr);
        return false;
    }

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    result = vkQueueSubmit(m_graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
    if (result != VK_SUCCESS) {
        logVkFailure("vkQueueSubmit(diffuseUpload)", result);
        vkDestroyCommandPool(m_device, commandPool, nullptr);
        destroyDiffuseTextureResources();
        vkFreeMemory(m_device, stagingMemory, nullptr);
        vkDestroyBuffer(m_device, stagingBuffer, nullptr);
        return false;
    }
    result = vkQueueWaitIdle(m_graphicsQueue);
    if (result != VK_SUCCESS) {
        logVkFailure("vkQueueWaitIdle(diffuseUpload)", result);
        vkDestroyCommandPool(m_device, commandPool, nullptr);
        destroyDiffuseTextureResources();
        vkFreeMemory(m_device, stagingMemory, nullptr);
        vkDestroyBuffer(m_device, stagingBuffer, nullptr);
        return false;
    }

    vkDestroyCommandPool(m_device, commandPool, nullptr);
    vkFreeMemory(m_device, stagingMemory, nullptr);
    vkDestroyBuffer(m_device, stagingBuffer, nullptr);

    VkImageViewCreateInfo viewCreateInfo{};
    viewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewCreateInfo.image = m_diffuseTextureImage;
    viewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewCreateInfo.format = kTextureFormat;
    viewCreateInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewCreateInfo.subresourceRange.baseMipLevel = 0;
    viewCreateInfo.subresourceRange.levelCount = diffuseMipLevels;
    viewCreateInfo.subresourceRange.baseArrayLayer = 0;
    viewCreateInfo.subresourceRange.layerCount = 1;
    result = vkCreateImageView(m_device, &viewCreateInfo, nullptr, &m_diffuseTextureImageView);
    if (result != VK_SUCCESS) {
        logVkFailure("vkCreateImageView(diffuseTexture)", result);
        destroyDiffuseTextureResources();
        return false;
    }
    setObjectName(
        VK_OBJECT_TYPE_IMAGE_VIEW,
        vkHandleToUint64(m_diffuseTextureImageView),
        "diffuse.albedo.imageView"
    );

    VkSamplerCreateInfo samplerCreateInfo{};
    samplerCreateInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerCreateInfo.magFilter = VK_FILTER_NEAREST;
    samplerCreateInfo.minFilter = VK_FILTER_NEAREST;
    samplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    samplerCreateInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerCreateInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerCreateInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerCreateInfo.mipLodBias = 0.0f;
    samplerCreateInfo.anisotropyEnable = m_supportsSamplerAnisotropy ? VK_TRUE : VK_FALSE;
    samplerCreateInfo.maxAnisotropy = m_supportsSamplerAnisotropy
        ? std::min(8.0f, m_maxSamplerAnisotropy)
        : 1.0f;
    samplerCreateInfo.compareEnable = VK_FALSE;
    samplerCreateInfo.minLod = 0.0f;
    samplerCreateInfo.maxLod = static_cast<float>(diffuseMipLevels - 1u);
    samplerCreateInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samplerCreateInfo.unnormalizedCoordinates = VK_FALSE;
    result = vkCreateSampler(m_device, &samplerCreateInfo, nullptr, &m_diffuseTextureSampler);
    if (result != VK_SUCCESS) {
        logVkFailure("vkCreateSampler(diffuseTexture)", result);
        destroyDiffuseTextureResources();
        return false;
    }
    setObjectName(
        VK_OBJECT_TYPE_SAMPLER,
        vkHandleToUint64(m_diffuseTextureSampler),
        "diffuse.albedo.sampler"
    );

    VOX_LOGI("render") << "diffuse atlas mipmaps generated: levels=" << diffuseMipLevels
                       << ", tileSize=" << kTileSize << ", atlas=" << kTextureWidth << "x" << kTextureHeight << "\n";

    return true;
}

bool Renderer::createShadowResources() {
    if (
        m_shadowDepthImage != VK_NULL_HANDLE &&
        m_shadowDepthImageView != VK_NULL_HANDLE &&
        m_shadowDepthSampler != VK_NULL_HANDLE
    ) {
        return true;
    }

    if (m_shadowDepthFormat == VK_FORMAT_UNDEFINED) {
        VOX_LOGE("render") << "shadow depth format is undefined\n";
        return false;
    }

    VkImageCreateInfo imageCreateInfo{};
    imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
    imageCreateInfo.format = m_shadowDepthFormat;
    imageCreateInfo.extent.width = kShadowAtlasSize;
    imageCreateInfo.extent.height = kShadowAtlasSize;
    imageCreateInfo.extent.depth = 1;
    imageCreateInfo.mipLevels = 1;
    imageCreateInfo.arrayLayers = 1;
    imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageCreateInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
#if defined(VOXEL_HAS_VMA)
    if (m_vmaAllocator != VK_NULL_HANDLE) {
        VmaAllocationCreateInfo allocationCreateInfo{};
        allocationCreateInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        allocationCreateInfo.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        const VkResult imageResult = vmaCreateImage(
            m_vmaAllocator,
            &imageCreateInfo,
            &allocationCreateInfo,
            &m_shadowDepthImage,
            &m_shadowDepthAllocation,
            nullptr
        );
        if (imageResult != VK_SUCCESS) {
            logVkFailure("vmaCreateImage(shadowDepth)", imageResult);
            return false;
        }
        setObjectName(VK_OBJECT_TYPE_IMAGE, vkHandleToUint64(m_shadowDepthImage), "shadow.atlas.image");
        VOX_LOGI("render") << "alloc shadow depth atlas (VMA): "
                  << kShadowAtlasSize << "x" << kShadowAtlasSize
                  << ", format=" << static_cast<int>(m_shadowDepthFormat)
                  << ", cascades=" << kShadowCascadeCount << "\n";
    } else
#endif
    {
        const VkResult imageResult = vkCreateImage(m_device, &imageCreateInfo, nullptr, &m_shadowDepthImage);
        if (imageResult != VK_SUCCESS) {
            logVkFailure("vkCreateImage(shadowDepth)", imageResult);
            return false;
        }
        setObjectName(VK_OBJECT_TYPE_IMAGE, vkHandleToUint64(m_shadowDepthImage), "shadow.atlas.image");

        VkMemoryRequirements memoryRequirements{};
        vkGetImageMemoryRequirements(m_device, m_shadowDepthImage, &memoryRequirements);
        const uint32_t memoryTypeIndex = findMemoryTypeIndex(
            m_physicalDevice,
            memoryRequirements.memoryTypeBits,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        );
        if (memoryTypeIndex == std::numeric_limits<uint32_t>::max()) {
            VOX_LOGI("render") << "no memory type for shadow depth image\n";
            destroyShadowResources();
            return false;
        }

        VkMemoryAllocateInfo allocateInfo{};
        allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocateInfo.allocationSize = memoryRequirements.size;
        allocateInfo.memoryTypeIndex = memoryTypeIndex;
        const VkResult allocResult = vkAllocateMemory(m_device, &allocateInfo, nullptr, &m_shadowDepthMemory);
        if (allocResult != VK_SUCCESS) {
            logVkFailure("vkAllocateMemory(shadowDepth)", allocResult);
            destroyShadowResources();
            return false;
        }

        const VkResult bindResult = vkBindImageMemory(m_device, m_shadowDepthImage, m_shadowDepthMemory, 0);
        if (bindResult != VK_SUCCESS) {
            logVkFailure("vkBindImageMemory(shadowDepth)", bindResult);
            destroyShadowResources();
            return false;
        }
        VOX_LOGI("render") << "alloc shadow depth atlas (vk): "
                  << kShadowAtlasSize << "x" << kShadowAtlasSize
                  << ", format=" << static_cast<int>(m_shadowDepthFormat)
                  << ", cascades=" << kShadowCascadeCount << "\n";
    }

    VkImageViewCreateInfo viewCreateInfo{};
    viewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewCreateInfo.image = m_shadowDepthImage;
    viewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewCreateInfo.format = m_shadowDepthFormat;
    viewCreateInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    viewCreateInfo.subresourceRange.baseMipLevel = 0;
    viewCreateInfo.subresourceRange.levelCount = 1;
    viewCreateInfo.subresourceRange.baseArrayLayer = 0;
    viewCreateInfo.subresourceRange.layerCount = 1;
    const VkResult viewResult = vkCreateImageView(m_device, &viewCreateInfo, nullptr, &m_shadowDepthImageView);
    if (viewResult != VK_SUCCESS) {
        logVkFailure("vkCreateImageView(shadowDepth)", viewResult);
        destroyShadowResources();
        return false;
    }
    setObjectName(
        VK_OBJECT_TYPE_IMAGE_VIEW,
        vkHandleToUint64(m_shadowDepthImageView),
        "shadow.atlas.imageView"
    );

    VkSamplerCreateInfo samplerCreateInfo{};
    samplerCreateInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerCreateInfo.magFilter = VK_FILTER_LINEAR;
    samplerCreateInfo.minFilter = VK_FILTER_LINEAR;
    samplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    samplerCreateInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    samplerCreateInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    samplerCreateInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    samplerCreateInfo.mipLodBias = 0.0f;
    samplerCreateInfo.anisotropyEnable = VK_FALSE;
    samplerCreateInfo.compareEnable = VK_TRUE;
    samplerCreateInfo.compareOp = VK_COMPARE_OP_GREATER_OR_EQUAL;
    samplerCreateInfo.minLod = 0.0f;
    samplerCreateInfo.maxLod = 0.0f;
    samplerCreateInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
    samplerCreateInfo.unnormalizedCoordinates = VK_FALSE;
    const VkResult samplerResult = vkCreateSampler(m_device, &samplerCreateInfo, nullptr, &m_shadowDepthSampler);
    if (samplerResult != VK_SUCCESS) {
        logVkFailure("vkCreateSampler(shadowDepth)", samplerResult);
        destroyShadowResources();
        return false;
    }
    setObjectName(
        VK_OBJECT_TYPE_SAMPLER,
        vkHandleToUint64(m_shadowDepthSampler),
        "shadow.atlas.sampler"
    );

    m_shadowDepthInitialized = false;
    VOX_LOGI("render") << "shadow resources ready (atlas " << kShadowAtlasSize << "x" << kShadowAtlasSize
              << ", cascades=" << kShadowCascadeCount << ")\n";
    return true;
}

bool Renderer::createSwapchain() {
    const SwapchainSupport support = querySwapchainSupport(m_physicalDevice, m_surface);
    if (support.formats.empty() || support.presentModes.empty()) {
        VOX_LOGI("render") << "swapchain support query returned no formats or present modes\n";
        return false;
    }

    const VkSurfaceFormatKHR surfaceFormat = chooseSwapchainFormat(support.formats);
    const VkPresentModeKHR presentMode = choosePresentMode(support.presentModes);
    const VkExtent2D extent = chooseExtent(m_window, support.capabilities);

    uint32_t imageCount = support.capabilities.minImageCount + 1;
    if (support.capabilities.maxImageCount > 0 && imageCount > support.capabilities.maxImageCount) {
        imageCount = support.capabilities.maxImageCount;
    }

    VkSwapchainCreateInfoKHR createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createInfo.surface = m_surface;
    createInfo.minImageCount = imageCount;
    createInfo.imageFormat = surfaceFormat.format;
    createInfo.imageColorSpace = surfaceFormat.colorSpace;
    createInfo.imageExtent = extent;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    createInfo.preTransform = support.capabilities.currentTransform;
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    createInfo.presentMode = presentMode;
    createInfo.clipped = VK_TRUE;

    const VkResult swapchainResult = vkCreateSwapchainKHR(m_device, &createInfo, nullptr, &m_swapchain);
    if (swapchainResult != VK_SUCCESS) {
        logVkFailure("vkCreateSwapchainKHR", swapchainResult);
        return false;
    }
    setObjectName(VK_OBJECT_TYPE_SWAPCHAIN_KHR, vkHandleToUint64(m_swapchain), "swapchain.main");

    vkGetSwapchainImagesKHR(m_device, m_swapchain, &imageCount, nullptr);
    m_swapchainImages.resize(imageCount);
    vkGetSwapchainImagesKHR(m_device, m_swapchain, &imageCount, m_swapchainImages.data());
    for (uint32_t i = 0; i < imageCount; ++i) {
        const std::string imageName = "swapchain.image." + std::to_string(i);
        setObjectName(VK_OBJECT_TYPE_IMAGE, vkHandleToUint64(m_swapchainImages[i]), imageName.c_str());
    }

    m_swapchainFormat = surfaceFormat.format;
    m_swapchainExtent = extent;

    m_swapchainImageViews.resize(imageCount, VK_NULL_HANDLE);
    for (uint32_t i = 0; i < imageCount; ++i) {
        VkImageViewCreateInfo viewCreateInfo{};
        viewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewCreateInfo.image = m_swapchainImages[i];
        viewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewCreateInfo.format = m_swapchainFormat;
        viewCreateInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewCreateInfo.subresourceRange.baseMipLevel = 0;
        viewCreateInfo.subresourceRange.levelCount = 1;
        viewCreateInfo.subresourceRange.baseArrayLayer = 0;
        viewCreateInfo.subresourceRange.layerCount = 1;

        if (vkCreateImageView(m_device, &viewCreateInfo, nullptr, &m_swapchainImageViews[i]) != VK_SUCCESS) {
            VOX_LOGE("render") << "failed to create swapchain image view " << i << "\n";
            return false;
        }
        const std::string viewName = "swapchain.imageView." + std::to_string(i);
        setObjectName(VK_OBJECT_TYPE_IMAGE_VIEW, vkHandleToUint64(m_swapchainImageViews[i]), viewName.c_str());
    }

    VOX_LOGI("render") << "swapchain ready: images=" << imageCount
              << ", extent=" << m_swapchainExtent.width << "x" << m_swapchainExtent.height << "\n";
    m_swapchainImageInitialized.assign(imageCount, false);
    m_swapchainImageTimelineValues.assign(imageCount, 0);
    if (!createHdrResolveTargets()) {
        VOX_LOGE("render") << "HDR resolve target creation failed\n";
        return false;
    }
    if (!createMsaaColorTargets()) {
        VOX_LOGE("render") << "MSAA color target creation failed\n";
        return false;
    }
    if (!createDepthTargets()) {
        VOX_LOGE("render") << "depth target creation failed\n";
        return false;
    }
    if (!createAoTargets()) {
        VOX_LOGE("render") << "AO target creation failed\n";
        return false;
    }
    m_renderFinishedSemaphores.resize(imageCount, VK_NULL_HANDLE);
    for (uint32_t i = 0; i < imageCount; ++i) {
        VkSemaphoreCreateInfo semaphoreCreateInfo{};
        semaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        const VkResult semaphoreResult =
            vkCreateSemaphore(m_device, &semaphoreCreateInfo, nullptr, &m_renderFinishedSemaphores[i]);
        if (semaphoreResult != VK_SUCCESS) {
            logVkFailure("vkCreateSemaphore(renderFinishedPerImage)", semaphoreResult);
            return false;
        }
        const std::string semaphoreName = "swapchain.renderFinished." + std::to_string(i);
        setObjectName(VK_OBJECT_TYPE_SEMAPHORE, vkHandleToUint64(m_renderFinishedSemaphores[i]), semaphoreName.c_str());
    }

    return true;
}

bool Renderer::createDepthTargets() {
    if (m_depthFormat == VK_FORMAT_UNDEFINED) {
        VOX_LOGE("render") << "depth format is undefined\n";
        return false;
    }

    const uint32_t imageCount = static_cast<uint32_t>(m_swapchainImages.size());
    m_depthImages.assign(imageCount, VK_NULL_HANDLE);
    m_depthImageMemories.assign(imageCount, VK_NULL_HANDLE);
    m_depthImageViews.assign(imageCount, VK_NULL_HANDLE);
#if defined(VOXEL_HAS_VMA)
    m_depthImageAllocations.assign(imageCount, VK_NULL_HANDLE);
#endif

    for (uint32_t i = 0; i < imageCount; ++i) {
        VkImageCreateInfo imageCreateInfo{};
        imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
        imageCreateInfo.format = m_depthFormat;
        imageCreateInfo.extent.width = m_swapchainExtent.width;
        imageCreateInfo.extent.height = m_swapchainExtent.height;
        imageCreateInfo.extent.depth = 1;
        imageCreateInfo.mipLevels = 1;
        imageCreateInfo.arrayLayers = 1;
        imageCreateInfo.samples = m_colorSampleCount;
        imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageCreateInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
        imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        VkResult imageResult = VK_ERROR_INITIALIZATION_FAILED;
#if defined(VOXEL_HAS_VMA)
        if (m_vmaAllocator != VK_NULL_HANDLE) {
            VmaAllocationCreateInfo allocationCreateInfo{};
            allocationCreateInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
            allocationCreateInfo.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
            imageResult = vmaCreateImage(
                m_vmaAllocator,
                &imageCreateInfo,
                &allocationCreateInfo,
                &m_depthImages[i],
                &m_depthImageAllocations[i],
                nullptr
            );
            if (imageResult != VK_SUCCESS) {
                logVkFailure("vmaCreateImage(depth)", imageResult);
                return false;
            }
        } else
#endif
        {
            imageResult = vkCreateImage(m_device, &imageCreateInfo, nullptr, &m_depthImages[i]);
            if (imageResult != VK_SUCCESS) {
                logVkFailure("vkCreateImage(depth)", imageResult);
                return false;
            }

            VkMemoryRequirements memoryRequirements{};
            vkGetImageMemoryRequirements(m_device, m_depthImages[i], &memoryRequirements);

            const uint32_t memoryTypeIndex = findMemoryTypeIndex(
                m_physicalDevice,
                memoryRequirements.memoryTypeBits,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
            );
            if (memoryTypeIndex == std::numeric_limits<uint32_t>::max()) {
                VOX_LOGI("render") << "no memory type for depth image\n";
                return false;
            }

            VkMemoryAllocateInfo allocateInfo{};
            allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            allocateInfo.allocationSize = memoryRequirements.size;
            allocateInfo.memoryTypeIndex = memoryTypeIndex;

            const VkResult allocResult = vkAllocateMemory(m_device, &allocateInfo, nullptr, &m_depthImageMemories[i]);
            if (allocResult != VK_SUCCESS) {
                logVkFailure("vkAllocateMemory(depth)", allocResult);
                return false;
            }

            const VkResult bindResult = vkBindImageMemory(m_device, m_depthImages[i], m_depthImageMemories[i], 0);
            if (bindResult != VK_SUCCESS) {
                logVkFailure("vkBindImageMemory(depth)", bindResult);
                return false;
            }
        }
        {
            const std::string imageName = "depth.msaa.image." + std::to_string(i);
            setObjectName(VK_OBJECT_TYPE_IMAGE, vkHandleToUint64(m_depthImages[i]), imageName.c_str());
        }

        VkImageViewCreateInfo viewCreateInfo{};
        viewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewCreateInfo.image = m_depthImages[i];
        viewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewCreateInfo.format = m_depthFormat;
        viewCreateInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        viewCreateInfo.subresourceRange.baseMipLevel = 0;
        viewCreateInfo.subresourceRange.levelCount = 1;
        viewCreateInfo.subresourceRange.baseArrayLayer = 0;
        viewCreateInfo.subresourceRange.layerCount = 1;

        const VkResult viewResult = vkCreateImageView(m_device, &viewCreateInfo, nullptr, &m_depthImageViews[i]);
        if (viewResult != VK_SUCCESS) {
            logVkFailure("vkCreateImageView(depth)", viewResult);
            return false;
        }
        {
            const std::string viewName = "depth.msaa.imageView." + std::to_string(i);
            setObjectName(VK_OBJECT_TYPE_IMAGE_VIEW, vkHandleToUint64(m_depthImageViews[i]), viewName.c_str());
        }
    }

    return true;
}

bool Renderer::createAoTargets() {
    if (m_normalDepthFormat == VK_FORMAT_UNDEFINED || m_ssaoFormat == VK_FORMAT_UNDEFINED) {
        VOX_LOGE("render") << "AO formats are undefined\n";
        return false;
    }
    if (m_depthFormat == VK_FORMAT_UNDEFINED) {
        VOX_LOGE("render") << "depth format is undefined for AO targets\n";
        return false;
    }

    const uint32_t imageCount = static_cast<uint32_t>(m_swapchainImages.size());
    const uint32_t frameTargetCount = kMaxFramesInFlight;
    m_aoExtent.width = std::max(1u, m_swapchainExtent.width / 2u);
    m_aoExtent.height = std::max(1u, m_swapchainExtent.height / 2u);

    auto createColorTargets = [&](VkFormat format,
                                  std::vector<VkImage>& outImages,
                                  std::vector<VkDeviceMemory>& outMemories,
                                  std::vector<VkImageView>& outViews,
                                  std::vector<TransientImageHandle>& outHandles,
                                  const char* debugLabel,
                                  FrameArenaPass firstPass,
                                  FrameArenaPass lastPass) -> bool {
        outImages.assign(frameTargetCount, VK_NULL_HANDLE);
        outMemories.assign(frameTargetCount, VK_NULL_HANDLE);
        outViews.assign(frameTargetCount, VK_NULL_HANDLE);
        outHandles.assign(frameTargetCount, kInvalidTransientImageHandle);
        for (uint32_t i = 0; i < frameTargetCount; ++i) {
            TransientImageDesc imageDesc{};
            imageDesc.imageType = VK_IMAGE_TYPE_2D;
            imageDesc.viewType = VK_IMAGE_VIEW_TYPE_2D;
            imageDesc.format = format;
            imageDesc.extent = {m_aoExtent.width, m_aoExtent.height, 1u};
            imageDesc.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
            imageDesc.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            imageDesc.mipLevels = 1;
            imageDesc.arrayLayers = 1;
            imageDesc.samples = VK_SAMPLE_COUNT_1_BIT;
            imageDesc.tiling = VK_IMAGE_TILING_OPTIMAL;
            imageDesc.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            imageDesc.firstPass = firstPass;
            imageDesc.lastPass = lastPass;
            imageDesc.debugName = std::string(debugLabel) + "[" + std::to_string(i) + "]";
            const TransientImageHandle handle = m_frameArena.createTransientImage(
                imageDesc,
                FrameArenaImageLifetime::Persistent
            );
            if (handle == kInvalidTransientImageHandle) {
                VOX_LOGE("render") << "failed creating transient image " << debugLabel << "\n";
                return false;
            }
            const TransientImageInfo* imageInfo = m_frameArena.getTransientImage(handle);
            if (imageInfo == nullptr || imageInfo->image == VK_NULL_HANDLE || imageInfo->view == VK_NULL_HANDLE) {
                VOX_LOGE("render") << "invalid transient image " << debugLabel << "\n";
                return false;
            }
            outHandles[i] = handle;
            outImages[i] = imageInfo->image;
            outViews[i] = imageInfo->view;
            outMemories[i] = VK_NULL_HANDLE;
            setObjectName(VK_OBJECT_TYPE_IMAGE, vkHandleToUint64(outImages[i]), imageDesc.debugName.c_str());
            {
                const std::string viewName = std::string(debugLabel) + ".view[" + std::to_string(i) + "]";
                setObjectName(VK_OBJECT_TYPE_IMAGE_VIEW, vkHandleToUint64(outViews[i]), viewName.c_str());
            }
        }
        return true;
    };

    m_normalDepthImageInitialized.assign(frameTargetCount, false);
    m_aoDepthImageInitialized.assign(imageCount, false);
    m_ssaoRawImageInitialized.assign(frameTargetCount, false);
    m_ssaoBlurImageInitialized.assign(frameTargetCount, false);

    if (!createColorTargets(
            m_normalDepthFormat,
            m_normalDepthImages,
            m_normalDepthImageMemories,
            m_normalDepthImageViews,
            m_normalDepthTransientHandles,
            "ao.normalDepth",
            FrameArenaPass::Ssao,
            FrameArenaPass::Ssao
        )) {
        return false;
    }

    m_aoDepthImages.assign(imageCount, VK_NULL_HANDLE);
    m_aoDepthImageMemories.assign(imageCount, VK_NULL_HANDLE);
    m_aoDepthImageViews.assign(imageCount, VK_NULL_HANDLE);
    m_aoDepthTransientHandles.assign(imageCount, kInvalidTransientImageHandle);
    for (uint32_t i = 0; i < imageCount; ++i) {
        TransientImageDesc depthDesc{};
        depthDesc.imageType = VK_IMAGE_TYPE_2D;
        depthDesc.viewType = VK_IMAGE_VIEW_TYPE_2D;
        depthDesc.format = m_depthFormat;
        depthDesc.extent = {m_aoExtent.width, m_aoExtent.height, 1u};
        depthDesc.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
        depthDesc.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        depthDesc.mipLevels = 1;
        depthDesc.arrayLayers = 1;
        depthDesc.samples = VK_SAMPLE_COUNT_1_BIT;
        depthDesc.tiling = VK_IMAGE_TILING_OPTIMAL;
        depthDesc.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        depthDesc.firstPass = FrameArenaPass::Ssao;
        depthDesc.lastPass = FrameArenaPass::Ssao;
        depthDesc.debugName = "ao.depth[" + std::to_string(i) + "]";
        const TransientImageHandle depthHandle = m_frameArena.createTransientImage(
            depthDesc,
            FrameArenaImageLifetime::Persistent
        );
        if (depthHandle == kInvalidTransientImageHandle) {
            VOX_LOGE("render") << "failed creating AO depth transient image\n";
            return false;
        }
        const TransientImageInfo* depthInfo = m_frameArena.getTransientImage(depthHandle);
        if (depthInfo == nullptr || depthInfo->image == VK_NULL_HANDLE || depthInfo->view == VK_NULL_HANDLE) {
            VOX_LOGE("render") << "invalid AO depth transient image info\n";
            return false;
        }
        m_aoDepthTransientHandles[i] = depthHandle;
        m_aoDepthImages[i] = depthInfo->image;
        m_aoDepthImageViews[i] = depthInfo->view;
        m_aoDepthImageMemories[i] = VK_NULL_HANDLE;
        setObjectName(VK_OBJECT_TYPE_IMAGE, vkHandleToUint64(m_aoDepthImages[i]), depthDesc.debugName.c_str());
        {
            const std::string viewName = "ao.depth.view[" + std::to_string(i) + "]";
            setObjectName(VK_OBJECT_TYPE_IMAGE_VIEW, vkHandleToUint64(m_aoDepthImageViews[i]), viewName.c_str());
        }
    }

    if (!createColorTargets(
            m_ssaoFormat,
            m_ssaoRawImages,
            m_ssaoRawImageMemories,
            m_ssaoRawImageViews,
            m_ssaoRawTransientHandles,
            "ao.ssaoRaw",
            FrameArenaPass::Ssao,
            FrameArenaPass::Ssao
        )) {
        return false;
    }
    if (!createColorTargets(
            m_ssaoFormat,
            m_ssaoBlurImages,
            m_ssaoBlurImageMemories,
            m_ssaoBlurImageViews,
            m_ssaoBlurTransientHandles,
            "ao.ssaoBlur",
            FrameArenaPass::Ssao,
            FrameArenaPass::Main
        )) {
        return false;
    }

    if (m_normalDepthSampler == VK_NULL_HANDLE) {
        VkSamplerCreateInfo samplerCreateInfo{};
        samplerCreateInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerCreateInfo.magFilter = VK_FILTER_NEAREST;
        samplerCreateInfo.minFilter = VK_FILTER_NEAREST;
        samplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        samplerCreateInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerCreateInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerCreateInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerCreateInfo.minLod = 0.0f;
        samplerCreateInfo.maxLod = 0.0f;
        samplerCreateInfo.maxAnisotropy = 1.0f;
        samplerCreateInfo.anisotropyEnable = VK_FALSE;
        samplerCreateInfo.compareEnable = VK_FALSE;
        samplerCreateInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
        samplerCreateInfo.unnormalizedCoordinates = VK_FALSE;
        const VkResult samplerResult = vkCreateSampler(m_device, &samplerCreateInfo, nullptr, &m_normalDepthSampler);
        if (samplerResult != VK_SUCCESS) {
            logVkFailure("vkCreateSampler(normalDepth)", samplerResult);
            return false;
        }
        setObjectName(
            VK_OBJECT_TYPE_SAMPLER,
            vkHandleToUint64(m_normalDepthSampler),
            "normalDepth.sampler"
        );
    }

    if (m_ssaoSampler == VK_NULL_HANDLE) {
        VkSamplerCreateInfo samplerCreateInfo{};
        samplerCreateInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerCreateInfo.magFilter = VK_FILTER_LINEAR;
        samplerCreateInfo.minFilter = VK_FILTER_LINEAR;
        samplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        samplerCreateInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerCreateInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerCreateInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerCreateInfo.minLod = 0.0f;
        samplerCreateInfo.maxLod = 0.0f;
        samplerCreateInfo.maxAnisotropy = 1.0f;
        samplerCreateInfo.anisotropyEnable = VK_FALSE;
        samplerCreateInfo.compareEnable = VK_FALSE;
        samplerCreateInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
        samplerCreateInfo.unnormalizedCoordinates = VK_FALSE;
        const VkResult samplerResult = vkCreateSampler(m_device, &samplerCreateInfo, nullptr, &m_ssaoSampler);
        if (samplerResult != VK_SUCCESS) {
            logVkFailure("vkCreateSampler(ssao)", samplerResult);
            return false;
        }
        setObjectName(
            VK_OBJECT_TYPE_SAMPLER,
            vkHandleToUint64(m_ssaoSampler),
            "ssao.sampler"
        );
    }

    return true;
}

bool Renderer::createHdrResolveTargets() {
    if (m_hdrColorFormat == VK_FORMAT_UNDEFINED) {
        VOX_LOGE("render") << "HDR color format is undefined\n";
        return false;
    }

    const uint32_t frameTargetCount = kMaxFramesInFlight;
    m_hdrResolveImages.assign(frameTargetCount, VK_NULL_HANDLE);
    m_hdrResolveImageMemories.assign(frameTargetCount, VK_NULL_HANDLE);
    m_hdrResolveImageViews.assign(frameTargetCount, VK_NULL_HANDLE);
    m_hdrResolveTransientHandles.assign(frameTargetCount, kInvalidTransientImageHandle);
    m_hdrResolveImageInitialized.assign(frameTargetCount, false);

    for (uint32_t i = 0; i < frameTargetCount; ++i) {
        TransientImageDesc hdrResolveDesc{};
        hdrResolveDesc.imageType = VK_IMAGE_TYPE_2D;
        hdrResolveDesc.viewType = VK_IMAGE_VIEW_TYPE_2D;
        hdrResolveDesc.format = m_hdrColorFormat;
        hdrResolveDesc.extent = {m_swapchainExtent.width, m_swapchainExtent.height, 1u};
        hdrResolveDesc.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        hdrResolveDesc.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        hdrResolveDesc.mipLevels = 1;
        hdrResolveDesc.arrayLayers = 1;
        hdrResolveDesc.samples = VK_SAMPLE_COUNT_1_BIT;
        hdrResolveDesc.tiling = VK_IMAGE_TILING_OPTIMAL;
        hdrResolveDesc.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        hdrResolveDesc.firstPass = FrameArenaPass::Main;
        hdrResolveDesc.lastPass = FrameArenaPass::Post;
        hdrResolveDesc.debugName = "hdr.resolve[" + std::to_string(i) + "]";
        const TransientImageHandle hdrResolveHandle = m_frameArena.createTransientImage(
            hdrResolveDesc,
            FrameArenaImageLifetime::Persistent
        );
        if (hdrResolveHandle == kInvalidTransientImageHandle) {
            VOX_LOGE("render") << "failed creating HDR resolve transient image\n";
            return false;
        }
        const TransientImageInfo* hdrResolveInfo = m_frameArena.getTransientImage(hdrResolveHandle);
        if (hdrResolveInfo == nullptr || hdrResolveInfo->image == VK_NULL_HANDLE || hdrResolveInfo->view == VK_NULL_HANDLE) {
            VOX_LOGE("render") << "invalid HDR resolve transient image info\n";
            return false;
        }
        m_hdrResolveTransientHandles[i] = hdrResolveHandle;
        m_hdrResolveImages[i] = hdrResolveInfo->image;
        m_hdrResolveImageViews[i] = hdrResolveInfo->view;
        m_hdrResolveImageMemories[i] = VK_NULL_HANDLE;
        setObjectName(
            VK_OBJECT_TYPE_IMAGE,
            vkHandleToUint64(m_hdrResolveImages[i]),
            hdrResolveDesc.debugName.c_str()
        );
        {
            const std::string viewName = "hdr.resolve.view[" + std::to_string(i) + "]";
            setObjectName(VK_OBJECT_TYPE_IMAGE_VIEW, vkHandleToUint64(m_hdrResolveImageViews[i]), viewName.c_str());
        }
    }

    if (m_hdrResolveSampler == VK_NULL_HANDLE) {
        VkSamplerCreateInfo samplerCreateInfo{};
        samplerCreateInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerCreateInfo.magFilter = VK_FILTER_LINEAR;
        samplerCreateInfo.minFilter = VK_FILTER_LINEAR;
        samplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        samplerCreateInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerCreateInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerCreateInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerCreateInfo.mipLodBias = 0.0f;
        samplerCreateInfo.anisotropyEnable = VK_FALSE;
        samplerCreateInfo.compareEnable = VK_FALSE;
        samplerCreateInfo.minLod = 0.0f;
        samplerCreateInfo.maxLod = 0.0f;
        samplerCreateInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
        samplerCreateInfo.unnormalizedCoordinates = VK_FALSE;

        const VkResult samplerResult = vkCreateSampler(m_device, &samplerCreateInfo, nullptr, &m_hdrResolveSampler);
        if (samplerResult != VK_SUCCESS) {
            logVkFailure("vkCreateSampler(hdrResolve)", samplerResult);
            return false;
        }
        setObjectName(
            VK_OBJECT_TYPE_SAMPLER,
            vkHandleToUint64(m_hdrResolveSampler),
            "hdrResolve.sampler"
        );
    }

    return true;
}

bool Renderer::createMsaaColorTargets() {
    const uint32_t imageCount = static_cast<uint32_t>(m_swapchainImages.size());
    m_msaaColorImages.assign(imageCount, VK_NULL_HANDLE);
    m_msaaColorImageMemories.assign(imageCount, VK_NULL_HANDLE);
    m_msaaColorImageViews.assign(imageCount, VK_NULL_HANDLE);
    m_msaaColorImageInitialized.assign(imageCount, false);
#if defined(VOXEL_HAS_VMA)
    m_msaaColorImageAllocations.assign(imageCount, VK_NULL_HANDLE);
#endif

    for (uint32_t i = 0; i < imageCount; ++i) {
        VkImageCreateInfo imageCreateInfo{};
        imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
        imageCreateInfo.format = m_hdrColorFormat;
        imageCreateInfo.extent.width = m_swapchainExtent.width;
        imageCreateInfo.extent.height = m_swapchainExtent.height;
        imageCreateInfo.extent.depth = 1;
        imageCreateInfo.mipLevels = 1;
        imageCreateInfo.arrayLayers = 1;
        imageCreateInfo.samples = m_colorSampleCount;
        imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageCreateInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT;
        imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        VkResult imageResult = VK_ERROR_INITIALIZATION_FAILED;
#if defined(VOXEL_HAS_VMA)
        if (m_vmaAllocator != VK_NULL_HANDLE) {
            VmaAllocationCreateInfo allocationCreateInfo{};
            allocationCreateInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
            allocationCreateInfo.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
            imageResult = vmaCreateImage(
                m_vmaAllocator,
                &imageCreateInfo,
                &allocationCreateInfo,
                &m_msaaColorImages[i],
                &m_msaaColorImageAllocations[i],
                nullptr
            );
            if (imageResult != VK_SUCCESS) {
                logVkFailure("vmaCreateImage(msaaColor)", imageResult);
                return false;
            }
        } else
#endif
        {
            imageResult = vkCreateImage(m_device, &imageCreateInfo, nullptr, &m_msaaColorImages[i]);
            if (imageResult != VK_SUCCESS) {
                logVkFailure("vkCreateImage(msaaColor)", imageResult);
                return false;
            }

            VkMemoryRequirements memoryRequirements{};
            vkGetImageMemoryRequirements(m_device, m_msaaColorImages[i], &memoryRequirements);

            const uint32_t memoryTypeIndex = findMemoryTypeIndex(
                m_physicalDevice,
                memoryRequirements.memoryTypeBits,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
            );
            if (memoryTypeIndex == std::numeric_limits<uint32_t>::max()) {
                VOX_LOGI("render") << "no memory type for MSAA color image\n";
                return false;
            }

            VkMemoryAllocateInfo allocateInfo{};
            allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            allocateInfo.allocationSize = memoryRequirements.size;
            allocateInfo.memoryTypeIndex = memoryTypeIndex;

            const VkResult allocResult = vkAllocateMemory(m_device, &allocateInfo, nullptr, &m_msaaColorImageMemories[i]);
            if (allocResult != VK_SUCCESS) {
                logVkFailure("vkAllocateMemory(msaaColor)", allocResult);
                return false;
            }

            const VkResult bindResult = vkBindImageMemory(m_device, m_msaaColorImages[i], m_msaaColorImageMemories[i], 0);
            if (bindResult != VK_SUCCESS) {
                logVkFailure("vkBindImageMemory(msaaColor)", bindResult);
                return false;
            }
        }
        {
            const std::string imageName = "hdr.msaaColor.image." + std::to_string(i);
            setObjectName(VK_OBJECT_TYPE_IMAGE, vkHandleToUint64(m_msaaColorImages[i]), imageName.c_str());
        }

        VkImageViewCreateInfo viewCreateInfo{};
        viewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewCreateInfo.image = m_msaaColorImages[i];
        viewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewCreateInfo.format = m_hdrColorFormat;
        viewCreateInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewCreateInfo.subresourceRange.baseMipLevel = 0;
        viewCreateInfo.subresourceRange.levelCount = 1;
        viewCreateInfo.subresourceRange.baseArrayLayer = 0;
        viewCreateInfo.subresourceRange.layerCount = 1;

        const VkResult viewResult = vkCreateImageView(m_device, &viewCreateInfo, nullptr, &m_msaaColorImageViews[i]);
        if (viewResult != VK_SUCCESS) {
            logVkFailure("vkCreateImageView(msaaColor)", viewResult);
            return false;
        }
        {
            const std::string viewName = "hdr.msaaColor.imageView." + std::to_string(i);
            setObjectName(VK_OBJECT_TYPE_IMAGE_VIEW, vkHandleToUint64(m_msaaColorImageViews[i]), viewName.c_str());
        }
    }

    return true;
}

bool Renderer::createDescriptorResources() {
    if (m_descriptorSetLayout == VK_NULL_HANDLE) {
        VkDescriptorSetLayoutBinding mvpBinding{};
        mvpBinding.binding = 0;
        mvpBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
        mvpBinding.descriptorCount = 1;
        mvpBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutBinding diffuseTextureBinding{};
        diffuseTextureBinding.binding = 1;
        diffuseTextureBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        diffuseTextureBinding.descriptorCount = 1;
        diffuseTextureBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutBinding hdrSceneBinding{};
        hdrSceneBinding.binding = 3;
        hdrSceneBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        hdrSceneBinding.descriptorCount = 1;
        hdrSceneBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutBinding shadowMapBinding{};
        shadowMapBinding.binding = 4;
        shadowMapBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        shadowMapBinding.descriptorCount = 1;
        shadowMapBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutBinding normalDepthBinding{};
        normalDepthBinding.binding = 6;
        normalDepthBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        normalDepthBinding.descriptorCount = 1;
        normalDepthBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutBinding ssaoBlurBinding{};
        ssaoBlurBinding.binding = 7;
        ssaoBlurBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        ssaoBlurBinding.descriptorCount = 1;
        ssaoBlurBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutBinding ssaoRawBinding{};
        ssaoRawBinding.binding = 8;
        ssaoRawBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        ssaoRawBinding.descriptorCount = 1;
        ssaoRawBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        const std::array<VkDescriptorSetLayoutBinding, 7> bindings = {
            mvpBinding,
            diffuseTextureBinding,
            hdrSceneBinding,
            shadowMapBinding,
            normalDepthBinding,
            ssaoBlurBinding,
            ssaoRawBinding
        };

        VkDescriptorSetLayoutCreateInfo layoutCreateInfo{};
        layoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutCreateInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        layoutCreateInfo.pBindings = bindings.data();

        const VkResult layoutResult =
            vkCreateDescriptorSetLayout(m_device, &layoutCreateInfo, nullptr, &m_descriptorSetLayout);
        if (layoutResult != VK_SUCCESS) {
            logVkFailure("vkCreateDescriptorSetLayout", layoutResult);
            return false;
        }
        setObjectName(
            VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT,
            vkHandleToUint64(m_descriptorSetLayout),
            "renderer.descriptorSetLayout.main"
        );
    }

    if (m_descriptorPool == VK_NULL_HANDLE) {
        const std::array<VkDescriptorPoolSize, 2> poolSizes = {
            VkDescriptorPoolSize{
                VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                kMaxFramesInFlight
            },
            VkDescriptorPoolSize{
                VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                6 * kMaxFramesInFlight
            }
        };

        VkDescriptorPoolCreateInfo poolCreateInfo{};
        poolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolCreateInfo.maxSets = kMaxFramesInFlight;
        poolCreateInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolCreateInfo.pPoolSizes = poolSizes.data();

        const VkResult poolResult = vkCreateDescriptorPool(m_device, &poolCreateInfo, nullptr, &m_descriptorPool);
        if (poolResult != VK_SUCCESS) {
            logVkFailure("vkCreateDescriptorPool", poolResult);
            return false;
        }
        setObjectName(
            VK_OBJECT_TYPE_DESCRIPTOR_POOL,
            vkHandleToUint64(m_descriptorPool),
            "renderer.descriptorPool.main"
        );
    }

    std::array<VkDescriptorSetLayout, kMaxFramesInFlight> setLayouts{};
    setLayouts.fill(m_descriptorSetLayout);

    VkDescriptorSetAllocateInfo allocateInfo{};
    allocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocateInfo.descriptorPool = m_descriptorPool;
    allocateInfo.descriptorSetCount = static_cast<uint32_t>(setLayouts.size());
    allocateInfo.pSetLayouts = setLayouts.data();

    const VkResult allocateResult = vkAllocateDescriptorSets(m_device, &allocateInfo, m_descriptorSets.data());
    if (allocateResult != VK_SUCCESS) {
        logVkFailure("vkAllocateDescriptorSets", allocateResult);
        return false;
    }
    for (size_t i = 0; i < m_descriptorSets.size(); ++i) {
        const std::string setName = "renderer.descriptorSet.frame" + std::to_string(i);
        setObjectName(VK_OBJECT_TYPE_DESCRIPTOR_SET, vkHandleToUint64(m_descriptorSets[i]), setName.c_str());
    }

    if (m_supportsBindlessDescriptors && m_bindlessTextureCapacity > 0) {
        if (m_bindlessDescriptorSetLayout == VK_NULL_HANDLE) {
            VkDescriptorSetLayoutBinding bindlessTexturesBinding{};
            bindlessTexturesBinding.binding = 0;
            bindlessTexturesBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            bindlessTexturesBinding.descriptorCount = m_bindlessTextureCapacity;
            bindlessTexturesBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

            const VkDescriptorBindingFlags bindlessBindingFlags = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT;
            VkDescriptorSetLayoutBindingFlagsCreateInfo bindingFlagsCreateInfo{};
            bindingFlagsCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
            bindingFlagsCreateInfo.bindingCount = 1;
            bindingFlagsCreateInfo.pBindingFlags = &bindlessBindingFlags;

            VkDescriptorSetLayoutCreateInfo bindlessLayoutCreateInfo{};
            bindlessLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            bindlessLayoutCreateInfo.pNext = &bindingFlagsCreateInfo;
            bindlessLayoutCreateInfo.bindingCount = 1;
            bindlessLayoutCreateInfo.pBindings = &bindlessTexturesBinding;

            const VkResult bindlessLayoutResult = vkCreateDescriptorSetLayout(
                m_device,
                &bindlessLayoutCreateInfo,
                nullptr,
                &m_bindlessDescriptorSetLayout
            );
            if (bindlessLayoutResult != VK_SUCCESS) {
                logVkFailure("vkCreateDescriptorSetLayout(bindless)", bindlessLayoutResult);
                return false;
            }
            setObjectName(
                VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT,
                vkHandleToUint64(m_bindlessDescriptorSetLayout),
                "renderer.descriptorSetLayout.bindless"
            );
        }

        if (m_bindlessDescriptorPool == VK_NULL_HANDLE) {
            const VkDescriptorPoolSize bindlessPoolSize{
                VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                m_bindlessTextureCapacity
            };

            VkDescriptorPoolCreateInfo bindlessPoolCreateInfo{};
            bindlessPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            bindlessPoolCreateInfo.maxSets = 1;
            bindlessPoolCreateInfo.poolSizeCount = 1;
            bindlessPoolCreateInfo.pPoolSizes = &bindlessPoolSize;

            const VkResult bindlessPoolResult = vkCreateDescriptorPool(
                m_device,
                &bindlessPoolCreateInfo,
                nullptr,
                &m_bindlessDescriptorPool
            );
            if (bindlessPoolResult != VK_SUCCESS) {
                logVkFailure("vkCreateDescriptorPool(bindless)", bindlessPoolResult);
                return false;
            }
            setObjectName(
                VK_OBJECT_TYPE_DESCRIPTOR_POOL,
                vkHandleToUint64(m_bindlessDescriptorPool),
                "renderer.descriptorPool.bindless"
            );
        }

        if (m_bindlessDescriptorSet == VK_NULL_HANDLE) {
            VkDescriptorSetAllocateInfo bindlessAllocateInfo{};
            bindlessAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            bindlessAllocateInfo.descriptorPool = m_bindlessDescriptorPool;
            bindlessAllocateInfo.descriptorSetCount = 1;
            bindlessAllocateInfo.pSetLayouts = &m_bindlessDescriptorSetLayout;
            const VkResult bindlessAllocateResult = vkAllocateDescriptorSets(
                m_device,
                &bindlessAllocateInfo,
                &m_bindlessDescriptorSet
            );
            if (bindlessAllocateResult != VK_SUCCESS) {
                logVkFailure("vkAllocateDescriptorSets(bindless)", bindlessAllocateResult);
                return false;
            }
            setObjectName(
                VK_OBJECT_TYPE_DESCRIPTOR_SET,
                vkHandleToUint64(m_bindlessDescriptorSet),
                "renderer.descriptorSet.bindless"
            );
        }
    }

    return true;
}

bool Renderer::createGraphicsPipeline() {
    if (m_depthFormat == VK_FORMAT_UNDEFINED) {
        VOX_LOGE("render") << "cannot create pipeline: depth format undefined\n";
        return false;
    }
    if (m_hdrColorFormat == VK_FORMAT_UNDEFINED) {
        VOX_LOGE("render") << "cannot create pipeline: HDR color format undefined\n";
        return false;
    }
    if (m_shadowDepthFormat == VK_FORMAT_UNDEFINED) {
        VOX_LOGE("render") << "cannot create pipeline: shadow depth format undefined\n";
        return false;
    }

    if (m_pipelineLayout == VK_NULL_HANDLE) {
        VkPushConstantRange chunkPushConstantRange{};
        chunkPushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        chunkPushConstantRange.offset = 0;
        chunkPushConstantRange.size = sizeof(ChunkPushConstants);

        VkPipelineLayoutCreateInfo layoutCreateInfo{};
        layoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        std::array<VkDescriptorSetLayout, 2> pipelineSetLayouts = {
            m_descriptorSetLayout,
            m_bindlessDescriptorSetLayout
        };
        if (m_supportsBindlessDescriptors && m_bindlessDescriptorSetLayout != VK_NULL_HANDLE) {
            layoutCreateInfo.setLayoutCount = 2;
            layoutCreateInfo.pSetLayouts = pipelineSetLayouts.data();
        } else {
            layoutCreateInfo.setLayoutCount = 1;
            layoutCreateInfo.pSetLayouts = &m_descriptorSetLayout;
        }
        layoutCreateInfo.pushConstantRangeCount = 1;
        layoutCreateInfo.pPushConstantRanges = &chunkPushConstantRange;
        const VkResult layoutResult = vkCreatePipelineLayout(m_device, &layoutCreateInfo, nullptr, &m_pipelineLayout);
        if (layoutResult != VK_SUCCESS) {
            logVkFailure("vkCreatePipelineLayout", layoutResult);
            return false;
        }
        setObjectName(
            VK_OBJECT_TYPE_PIPELINE_LAYOUT,
            vkHandleToUint64(m_pipelineLayout),
            "renderer.pipelineLayout.main"
        );
    }

    constexpr const char* kWorldVertexShaderPath = "../src/render/shaders/voxel_packed.vert.slang.spv";
    constexpr const char* kWorldFragmentShaderPath = "../src/render/shaders/voxel_packed.frag.slang.spv";
    constexpr const char* kSkyboxVertexShaderPath = "../src/render/shaders/skybox.vert.slang.spv";
    constexpr const char* kSkyboxFragmentShaderPath = "../src/render/shaders/skybox.frag.slang.spv";
    constexpr const char* kToneMapVertexShaderPath = "../src/render/shaders/tone_map.vert.slang.spv";
    constexpr const char* kToneMapFragmentShaderPath = "../src/render/shaders/tone_map.frag.slang.spv";
    constexpr const char* kShadowVertexShaderPath = "../src/render/shaders/shadow_depth.vert.slang.spv";
    constexpr const char* kShadowFragmentShaderPath = "../src/render/shaders/shadow_depth.frag.slang.spv";
    constexpr const char* kPipeShadowVertexShaderPath = "../src/render/shaders/pipe_shadow.vert.slang.spv";
    constexpr const char* kPipeShadowFragmentShaderPath = "../src/render/shaders/pipe_shadow.frag.slang.spv";
    constexpr const char* kGrassShadowVertexShaderPath = "../src/render/shaders/grass_billboard_shadow.vert.slang.spv";
    constexpr const char* kGrassShadowFragmentShaderPath = "../src/render/shaders/grass_billboard_shadow.frag.slang.spv";

    VkShaderModule worldVertShaderModule = VK_NULL_HANDLE;
    VkShaderModule worldFragShaderModule = VK_NULL_HANDLE;
    VkShaderModule skyboxVertShaderModule = VK_NULL_HANDLE;
    VkShaderModule skyboxFragShaderModule = VK_NULL_HANDLE;
    VkShaderModule toneMapVertShaderModule = VK_NULL_HANDLE;
    VkShaderModule toneMapFragShaderModule = VK_NULL_HANDLE;
    VkShaderModule shadowVertShaderModule = VK_NULL_HANDLE;
    VkShaderModule shadowFragShaderModule = VK_NULL_HANDLE;

    if (!createShaderModuleFromFile(
            m_device,
            kWorldVertexShaderPath,
            "voxel_packed.vert",
            worldVertShaderModule
        )) {
        return false;
    }
    if (!createShaderModuleFromFile(
            m_device,
            kWorldFragmentShaderPath,
            "voxel_packed.frag",
            worldFragShaderModule
        )) {
        vkDestroyShaderModule(m_device, worldVertShaderModule, nullptr);
        return false;
    }
    if (!createShaderModuleFromFile(
            m_device,
            kSkyboxVertexShaderPath,
            "skybox.vert",
            skyboxVertShaderModule
        )) {
        vkDestroyShaderModule(m_device, worldFragShaderModule, nullptr);
        vkDestroyShaderModule(m_device, worldVertShaderModule, nullptr);
        return false;
    }
    if (!createShaderModuleFromFile(
            m_device,
            kSkyboxFragmentShaderPath,
            "skybox.frag",
            skyboxFragShaderModule
        )) {
        vkDestroyShaderModule(m_device, skyboxVertShaderModule, nullptr);
        vkDestroyShaderModule(m_device, worldFragShaderModule, nullptr);
        vkDestroyShaderModule(m_device, worldVertShaderModule, nullptr);
        return false;
    }
    if (!createShaderModuleFromFile(
            m_device,
            kToneMapVertexShaderPath,
            "tone_map.vert",
            toneMapVertShaderModule
        )) {
        vkDestroyShaderModule(m_device, skyboxFragShaderModule, nullptr);
        vkDestroyShaderModule(m_device, skyboxVertShaderModule, nullptr);
        vkDestroyShaderModule(m_device, worldFragShaderModule, nullptr);
        vkDestroyShaderModule(m_device, worldVertShaderModule, nullptr);
        return false;
    }
    if (!createShaderModuleFromFile(
            m_device,
            kToneMapFragmentShaderPath,
            "tone_map.frag",
            toneMapFragShaderModule
        )) {
        vkDestroyShaderModule(m_device, toneMapVertShaderModule, nullptr);
        vkDestroyShaderModule(m_device, skyboxFragShaderModule, nullptr);
        vkDestroyShaderModule(m_device, skyboxVertShaderModule, nullptr);
        vkDestroyShaderModule(m_device, worldFragShaderModule, nullptr);
        vkDestroyShaderModule(m_device, worldVertShaderModule, nullptr);
        return false;
    }

    VkPipelineShaderStageCreateInfo worldVertexShaderStage{};
    worldVertexShaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    worldVertexShaderStage.stage = VK_SHADER_STAGE_VERTEX_BIT;
    worldVertexShaderStage.module = worldVertShaderModule;
    worldVertexShaderStage.pName = "main";

    VkPipelineShaderStageCreateInfo worldFragmentShaderStage{};
    worldFragmentShaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    worldFragmentShaderStage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    worldFragmentShaderStage.module = worldFragShaderModule;
    worldFragmentShaderStage.pName = "main";
    struct WorldFragmentSpecializationData {
        std::int32_t shadowPolicyMode = 2;  // 0=no shadows, 1=single-cascade PCF, 2=cascade-blended PCF
        std::int32_t ambientPolicyMode = 2; // 0=SH only, 1=SH hemisphere, 2=SH hemisphere + vertex AO
    };
    const WorldFragmentSpecializationData worldFragmentSpecializationData{};
    const std::array<VkSpecializationMapEntry, 2> worldFragmentSpecializationMapEntries = {{
        VkSpecializationMapEntry{
            6u,
            static_cast<uint32_t>(offsetof(WorldFragmentSpecializationData, shadowPolicyMode)),
            sizeof(std::int32_t)
        },
        VkSpecializationMapEntry{
            7u,
            static_cast<uint32_t>(offsetof(WorldFragmentSpecializationData, ambientPolicyMode)),
            sizeof(std::int32_t)
        }
    }};
    const VkSpecializationInfo worldFragmentSpecializationInfo{
        static_cast<uint32_t>(worldFragmentSpecializationMapEntries.size()),
        worldFragmentSpecializationMapEntries.data(),
        sizeof(worldFragmentSpecializationData),
        &worldFragmentSpecializationData
    };
    worldFragmentShaderStage.pSpecializationInfo = &worldFragmentSpecializationInfo;

    std::array<VkPipelineShaderStageCreateInfo, 2> worldShaderStages = {worldVertexShaderStage, worldFragmentShaderStage};

    // Binding 0: packed voxel vertices. Binding 1: per-draw chunk origin.
    VkVertexInputBindingDescription bindingDescriptions[2]{};
    bindingDescriptions[0].binding = 0;
    bindingDescriptions[0].stride = sizeof(world::PackedVoxelVertex);
    bindingDescriptions[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    bindingDescriptions[1].binding = 1;
    bindingDescriptions[1].stride = sizeof(ChunkInstanceData);
    bindingDescriptions[1].inputRate = VK_VERTEX_INPUT_RATE_INSTANCE;

    VkVertexInputAttributeDescription attributeDescriptions[2]{};
    attributeDescriptions[0].location = 0;
    attributeDescriptions[0].binding = 0;
    attributeDescriptions[0].format = VK_FORMAT_R32_UINT;
    attributeDescriptions[0].offset = 0;
    attributeDescriptions[1].location = 1;
    attributeDescriptions[1].binding = 1;
    attributeDescriptions[1].format = VK_FORMAT_R32G32B32A32_SFLOAT;
    attributeDescriptions[1].offset = 0;

    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = 2;
    vertexInputInfo.pVertexBindingDescriptions = bindingDescriptions;
    vertexInputInfo.vertexAttributeDescriptionCount = 2;
    vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions;

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples = m_colorSampleCount;

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_GREATER_OR_EQUAL;
    depthStencil.depthBoundsTestEnable = VK_FALSE;
    depthStencil.stencilTestEnable = VK_FALSE;

    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;

    std::array<VkDynamicState, 2> dynamicStates = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR,
    };
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
    dynamicState.pDynamicStates = dynamicStates.data();

    VkPipelineRenderingCreateInfo renderingCreateInfo{};
    renderingCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    renderingCreateInfo.colorAttachmentCount = 1;
    renderingCreateInfo.pColorAttachmentFormats = &m_hdrColorFormat;
    renderingCreateInfo.depthAttachmentFormat = m_depthFormat;

    VkGraphicsPipelineCreateInfo pipelineCreateInfo{};
    pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineCreateInfo.pNext = &renderingCreateInfo;
    pipelineCreateInfo.stageCount = static_cast<uint32_t>(worldShaderStages.size());
    pipelineCreateInfo.pStages = worldShaderStages.data();
    pipelineCreateInfo.pVertexInputState = &vertexInputInfo;
    pipelineCreateInfo.pInputAssemblyState = &inputAssembly;
    pipelineCreateInfo.pViewportState = &viewportState;
    pipelineCreateInfo.pRasterizationState = &rasterizer;
    pipelineCreateInfo.pMultisampleState = &multisampling;
    pipelineCreateInfo.pDepthStencilState = &depthStencil;
    pipelineCreateInfo.pColorBlendState = &colorBlending;
    pipelineCreateInfo.pDynamicState = &dynamicState;
    pipelineCreateInfo.layout = m_pipelineLayout;
    pipelineCreateInfo.renderPass = VK_NULL_HANDLE;
    pipelineCreateInfo.subpass = 0;

    VkPipeline worldPipeline = VK_NULL_HANDLE;
    const VkResult worldPipelineResult = vkCreateGraphicsPipelines(
        m_device,
        VK_NULL_HANDLE,
        1,
        &pipelineCreateInfo,
        nullptr,
        &worldPipeline
    );
    if (worldPipelineResult != VK_SUCCESS) {
        vkDestroyShaderModule(m_device, toneMapFragShaderModule, nullptr);
        vkDestroyShaderModule(m_device, toneMapVertShaderModule, nullptr);
        vkDestroyShaderModule(m_device, skyboxFragShaderModule, nullptr);
        vkDestroyShaderModule(m_device, skyboxVertShaderModule, nullptr);
        vkDestroyShaderModule(m_device, worldFragShaderModule, nullptr);
        vkDestroyShaderModule(m_device, worldVertShaderModule, nullptr);
        logVkFailure("vkCreateGraphicsPipelines(world)", worldPipelineResult);
        return false;
    }
    VOX_LOGI("render") << "pipeline config (world): samples=" << static_cast<uint32_t>(m_colorSampleCount)
              << ", cullMode=" << static_cast<uint32_t>(rasterizer.cullMode)
              << ", depthCompare=" << static_cast<uint32_t>(depthStencil.depthCompareOp)
              << ", shadowPolicyMode=" << worldFragmentSpecializationData.shadowPolicyMode
              << ", ambientPolicyMode=" << worldFragmentSpecializationData.ambientPolicyMode
              << "\n";

    VkPipelineRasterizationStateCreateInfo previewAddRasterizer = rasterizer;
    previewAddRasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    // Preview draws closed helper geometry; disable culling to avoid face dropouts from winding mismatches.
    previewAddRasterizer.cullMode = VK_CULL_MODE_NONE;
    previewAddRasterizer.depthBiasEnable = VK_FALSE;

    VkPipelineRasterizationStateCreateInfo previewRemoveRasterizer = rasterizer;
    previewRemoveRasterizer.polygonMode = m_supportsWireframePreview ? VK_POLYGON_MODE_LINE : VK_POLYGON_MODE_FILL;
    previewRemoveRasterizer.cullMode = VK_CULL_MODE_NONE;
    previewRemoveRasterizer.depthBiasEnable = VK_FALSE;

    VkPipelineDepthStencilStateCreateInfo previewDepthStencil = depthStencil;
    previewDepthStencil.depthWriteEnable = VK_TRUE;
    previewDepthStencil.depthCompareOp = VK_COMPARE_OP_GREATER_OR_EQUAL;

    std::array<VkDynamicState, 3> previewDynamicStates = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR,
        VK_DYNAMIC_STATE_DEPTH_BIAS,
    };
    VkPipelineDynamicStateCreateInfo previewDynamicState{};
    previewDynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    previewDynamicState.dynamicStateCount = static_cast<uint32_t>(previewDynamicStates.size());
    previewDynamicState.pDynamicStates = previewDynamicStates.data();

    VkGraphicsPipelineCreateInfo previewAddPipelineCreateInfo = pipelineCreateInfo;
    previewAddPipelineCreateInfo.pRasterizationState = &previewAddRasterizer;
    previewAddPipelineCreateInfo.pDepthStencilState = &previewDepthStencil;
    previewAddPipelineCreateInfo.pDynamicState = &previewDynamicState;

    VkPipeline previewAddPipeline = VK_NULL_HANDLE;
    const VkResult previewAddPipelineResult = vkCreateGraphicsPipelines(
        m_device,
        VK_NULL_HANDLE,
        1,
        &previewAddPipelineCreateInfo,
        nullptr,
        &previewAddPipeline
    );
    if (previewAddPipelineResult != VK_SUCCESS) {
        vkDestroyPipeline(m_device, worldPipeline, nullptr);
        vkDestroyShaderModule(m_device, toneMapFragShaderModule, nullptr);
        vkDestroyShaderModule(m_device, toneMapVertShaderModule, nullptr);
        vkDestroyShaderModule(m_device, skyboxFragShaderModule, nullptr);
        vkDestroyShaderModule(m_device, skyboxVertShaderModule, nullptr);
        vkDestroyShaderModule(m_device, worldFragShaderModule, nullptr);
        vkDestroyShaderModule(m_device, worldVertShaderModule, nullptr);
        logVkFailure("vkCreateGraphicsPipelines(previewAdd)", previewAddPipelineResult);
        return false;
    }

    VkGraphicsPipelineCreateInfo previewRemovePipelineCreateInfo = pipelineCreateInfo;
    previewRemovePipelineCreateInfo.pRasterizationState = &previewRemoveRasterizer;
    previewRemovePipelineCreateInfo.pDepthStencilState = &previewDepthStencil;
    previewRemovePipelineCreateInfo.pDynamicState = &previewDynamicState;

    VkPipeline previewRemovePipeline = VK_NULL_HANDLE;
    const VkResult previewRemovePipelineResult = vkCreateGraphicsPipelines(
        m_device,
        VK_NULL_HANDLE,
        1,
        &previewRemovePipelineCreateInfo,
        nullptr,
        &previewRemovePipeline
    );

    if (previewRemovePipelineResult != VK_SUCCESS) {
        vkDestroyPipeline(m_device, worldPipeline, nullptr);
        vkDestroyPipeline(m_device, previewAddPipeline, nullptr);
        vkDestroyShaderModule(m_device, toneMapFragShaderModule, nullptr);
        vkDestroyShaderModule(m_device, toneMapVertShaderModule, nullptr);
        vkDestroyShaderModule(m_device, skyboxFragShaderModule, nullptr);
        vkDestroyShaderModule(m_device, skyboxVertShaderModule, nullptr);
        vkDestroyShaderModule(m_device, worldFragShaderModule, nullptr);
        vkDestroyShaderModule(m_device, worldVertShaderModule, nullptr);
        logVkFailure("vkCreateGraphicsPipelines(previewRemove)", previewRemovePipelineResult);
        return false;
    }

    VkPipelineShaderStageCreateInfo skyboxVertexShaderStage{};
    skyboxVertexShaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    skyboxVertexShaderStage.stage = VK_SHADER_STAGE_VERTEX_BIT;
    skyboxVertexShaderStage.module = skyboxVertShaderModule;
    skyboxVertexShaderStage.pName = "main";

    VkPipelineShaderStageCreateInfo skyboxFragmentShaderStage{};
    skyboxFragmentShaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    skyboxFragmentShaderStage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    skyboxFragmentShaderStage.module = skyboxFragShaderModule;
    skyboxFragmentShaderStage.pName = "main";

    const std::array<VkPipelineShaderStageCreateInfo, 2> skyboxShaderStages = {
        skyboxVertexShaderStage,
        skyboxFragmentShaderStage
    };

    VkPipelineVertexInputStateCreateInfo skyboxVertexInputInfo{};
    skyboxVertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    VkPipelineInputAssemblyStateCreateInfo skyboxInputAssembly = inputAssembly;

    VkPipelineRasterizationStateCreateInfo skyboxRasterizer = rasterizer;
    skyboxRasterizer.cullMode = VK_CULL_MODE_NONE;

    VkPipelineDepthStencilStateCreateInfo skyboxDepthStencil = depthStencil;
    skyboxDepthStencil.depthTestEnable = VK_FALSE;
    skyboxDepthStencil.depthWriteEnable = VK_FALSE;
    skyboxDepthStencil.depthCompareOp = VK_COMPARE_OP_ALWAYS;

    VkGraphicsPipelineCreateInfo skyboxPipelineCreateInfo = pipelineCreateInfo;
    skyboxPipelineCreateInfo.stageCount = static_cast<uint32_t>(skyboxShaderStages.size());
    skyboxPipelineCreateInfo.pStages = skyboxShaderStages.data();
    skyboxPipelineCreateInfo.pVertexInputState = &skyboxVertexInputInfo;
    skyboxPipelineCreateInfo.pInputAssemblyState = &skyboxInputAssembly;
    skyboxPipelineCreateInfo.pDepthStencilState = &skyboxDepthStencil;
    skyboxPipelineCreateInfo.pRasterizationState = &skyboxRasterizer;

    VkPipeline skyboxPipeline = VK_NULL_HANDLE;
    const VkResult skyboxPipelineResult = vkCreateGraphicsPipelines(
        m_device,
        VK_NULL_HANDLE,
        1,
        &skyboxPipelineCreateInfo,
        nullptr,
        &skyboxPipeline
    );

    if (skyboxPipelineResult != VK_SUCCESS) {
        vkDestroyPipeline(m_device, worldPipeline, nullptr);
        vkDestroyPipeline(m_device, previewAddPipeline, nullptr);
        vkDestroyPipeline(m_device, previewRemovePipeline, nullptr);
        vkDestroyShaderModule(m_device, toneMapFragShaderModule, nullptr);
        vkDestroyShaderModule(m_device, toneMapVertShaderModule, nullptr);
        vkDestroyShaderModule(m_device, skyboxFragShaderModule, nullptr);
        vkDestroyShaderModule(m_device, skyboxVertShaderModule, nullptr);
        vkDestroyShaderModule(m_device, worldFragShaderModule, nullptr);
        vkDestroyShaderModule(m_device, worldVertShaderModule, nullptr);
        logVkFailure("vkCreateGraphicsPipelines(skybox)", skyboxPipelineResult);
        return false;
    }
    VOX_LOGI("render") << "pipeline config (skybox): cullMode=" << static_cast<uint32_t>(skyboxRasterizer.cullMode)
              << ", depthTest=" << (skyboxDepthStencil.depthTestEnable == VK_TRUE ? 1 : 0)
              << ", depthWrite=" << (skyboxDepthStencil.depthWriteEnable == VK_TRUE ? 1 : 0)
              << "\n";

    VkPipelineShaderStageCreateInfo toneMapVertexShaderStage{};
    toneMapVertexShaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    toneMapVertexShaderStage.stage = VK_SHADER_STAGE_VERTEX_BIT;
    toneMapVertexShaderStage.module = toneMapVertShaderModule;
    toneMapVertexShaderStage.pName = "main";

    VkPipelineShaderStageCreateInfo toneMapFragmentShaderStage{};
    toneMapFragmentShaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    toneMapFragmentShaderStage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    toneMapFragmentShaderStage.module = toneMapFragShaderModule;
    toneMapFragmentShaderStage.pName = "main";

    const std::array<VkPipelineShaderStageCreateInfo, 2> toneMapShaderStages = {
        toneMapVertexShaderStage,
        toneMapFragmentShaderStage
    };

    VkPipelineVertexInputStateCreateInfo toneMapVertexInputInfo{};
    toneMapVertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    VkPipelineInputAssemblyStateCreateInfo toneMapInputAssembly = inputAssembly;

    VkPipelineRasterizationStateCreateInfo toneMapRasterizer = rasterizer;
    toneMapRasterizer.cullMode = VK_CULL_MODE_NONE;

    VkPipelineMultisampleStateCreateInfo toneMapMultisampling{};
    toneMapMultisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    toneMapMultisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo toneMapDepthStencil{};
    toneMapDepthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    toneMapDepthStencil.depthTestEnable = VK_FALSE;
    toneMapDepthStencil.depthWriteEnable = VK_FALSE;
    toneMapDepthStencil.depthBoundsTestEnable = VK_FALSE;
    toneMapDepthStencil.stencilTestEnable = VK_FALSE;

    VkPipelineRenderingCreateInfo toneMapRenderingCreateInfo{};
    toneMapRenderingCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    toneMapRenderingCreateInfo.colorAttachmentCount = 1;
    toneMapRenderingCreateInfo.pColorAttachmentFormats = &m_swapchainFormat;
    toneMapRenderingCreateInfo.depthAttachmentFormat = VK_FORMAT_UNDEFINED;

    VkGraphicsPipelineCreateInfo toneMapPipelineCreateInfo{};
    toneMapPipelineCreateInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    toneMapPipelineCreateInfo.pNext = &toneMapRenderingCreateInfo;
    toneMapPipelineCreateInfo.stageCount = static_cast<uint32_t>(toneMapShaderStages.size());
    toneMapPipelineCreateInfo.pStages = toneMapShaderStages.data();
    toneMapPipelineCreateInfo.pVertexInputState = &toneMapVertexInputInfo;
    toneMapPipelineCreateInfo.pInputAssemblyState = &toneMapInputAssembly;
    toneMapPipelineCreateInfo.pViewportState = &viewportState;
    toneMapPipelineCreateInfo.pRasterizationState = &toneMapRasterizer;
    toneMapPipelineCreateInfo.pMultisampleState = &toneMapMultisampling;
    toneMapPipelineCreateInfo.pDepthStencilState = &toneMapDepthStencil;
    toneMapPipelineCreateInfo.pColorBlendState = &colorBlending;
    toneMapPipelineCreateInfo.pDynamicState = &dynamicState;
    toneMapPipelineCreateInfo.layout = m_pipelineLayout;
    toneMapPipelineCreateInfo.renderPass = VK_NULL_HANDLE;
    toneMapPipelineCreateInfo.subpass = 0;

    VkPipeline toneMapPipeline = VK_NULL_HANDLE;
    const VkResult toneMapPipelineResult = vkCreateGraphicsPipelines(
        m_device,
        VK_NULL_HANDLE,
        1,
        &toneMapPipelineCreateInfo,
        nullptr,
        &toneMapPipeline
    );

    if (toneMapPipelineResult != VK_SUCCESS) {
        vkDestroyShaderModule(m_device, toneMapFragShaderModule, nullptr);
        vkDestroyShaderModule(m_device, toneMapVertShaderModule, nullptr);
        vkDestroyShaderModule(m_device, skyboxFragShaderModule, nullptr);
        vkDestroyShaderModule(m_device, skyboxVertShaderModule, nullptr);
        vkDestroyShaderModule(m_device, worldFragShaderModule, nullptr);
        vkDestroyShaderModule(m_device, worldVertShaderModule, nullptr);
        vkDestroyPipeline(m_device, worldPipeline, nullptr);
        vkDestroyPipeline(m_device, previewAddPipeline, nullptr);
        vkDestroyPipeline(m_device, previewRemovePipeline, nullptr);
        vkDestroyPipeline(m_device, skyboxPipeline, nullptr);
        logVkFailure("vkCreateGraphicsPipelines(toneMap)", toneMapPipelineResult);
        return false;
    }
    VOX_LOGI("render") << "pipeline config (tonemap): samples="
              << static_cast<uint32_t>(toneMapMultisampling.rasterizationSamples)
              << ", swapchainFormat=" << static_cast<int>(m_swapchainFormat)
              << "\n";

    if (!createShaderModuleFromFile(
            m_device,
            kShadowVertexShaderPath,
            "shadow_depth.vert",
            shadowVertShaderModule
        )) {
        vkDestroyShaderModule(m_device, toneMapFragShaderModule, nullptr);
        vkDestroyShaderModule(m_device, toneMapVertShaderModule, nullptr);
        vkDestroyShaderModule(m_device, skyboxFragShaderModule, nullptr);
        vkDestroyShaderModule(m_device, skyboxVertShaderModule, nullptr);
        vkDestroyShaderModule(m_device, worldFragShaderModule, nullptr);
        vkDestroyShaderModule(m_device, worldVertShaderModule, nullptr);
        vkDestroyPipeline(m_device, worldPipeline, nullptr);
        vkDestroyPipeline(m_device, previewAddPipeline, nullptr);
        vkDestroyPipeline(m_device, previewRemovePipeline, nullptr);
        vkDestroyPipeline(m_device, skyboxPipeline, nullptr);
        vkDestroyPipeline(m_device, toneMapPipeline, nullptr);
        return false;
    }
    if (!createShaderModuleFromFile(
            m_device,
            kShadowFragmentShaderPath,
            "shadow_depth.frag",
            shadowFragShaderModule
        )) {
        vkDestroyShaderModule(m_device, shadowVertShaderModule, nullptr);
        vkDestroyShaderModule(m_device, toneMapFragShaderModule, nullptr);
        vkDestroyShaderModule(m_device, toneMapVertShaderModule, nullptr);
        vkDestroyShaderModule(m_device, skyboxFragShaderModule, nullptr);
        vkDestroyShaderModule(m_device, skyboxVertShaderModule, nullptr);
        vkDestroyShaderModule(m_device, worldFragShaderModule, nullptr);
        vkDestroyShaderModule(m_device, worldVertShaderModule, nullptr);
        vkDestroyPipeline(m_device, worldPipeline, nullptr);
        vkDestroyPipeline(m_device, previewAddPipeline, nullptr);
        vkDestroyPipeline(m_device, previewRemovePipeline, nullptr);
        vkDestroyPipeline(m_device, skyboxPipeline, nullptr);
        vkDestroyPipeline(m_device, toneMapPipeline, nullptr);
        return false;
    }

    VkPipelineShaderStageCreateInfo shadowVertexShaderStage{};
    shadowVertexShaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shadowVertexShaderStage.stage = VK_SHADER_STAGE_VERTEX_BIT;
    shadowVertexShaderStage.module = shadowVertShaderModule;
    shadowVertexShaderStage.pName = "main";

    VkPipelineShaderStageCreateInfo shadowFragmentShaderStage{};
    shadowFragmentShaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shadowFragmentShaderStage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    shadowFragmentShaderStage.module = shadowFragShaderModule;
    shadowFragmentShaderStage.pName = "main";

    const std::array<VkPipelineShaderStageCreateInfo, 2> shadowShaderStages = {
        shadowVertexShaderStage,
        shadowFragmentShaderStage
    };

    VkPipelineMultisampleStateCreateInfo shadowMultisampling{};
    shadowMultisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    shadowMultisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineRasterizationStateCreateInfo shadowRasterizer = rasterizer;
    shadowRasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
    shadowRasterizer.depthBiasEnable = VK_TRUE;

    VkPipelineDepthStencilStateCreateInfo shadowDepthStencil = depthStencil;
    shadowDepthStencil.depthTestEnable = VK_TRUE;
    shadowDepthStencil.depthWriteEnable = VK_TRUE;
    shadowDepthStencil.depthCompareOp = VK_COMPARE_OP_GREATER_OR_EQUAL;

    std::array<VkDynamicState, 3> shadowDynamicStates = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR,
        VK_DYNAMIC_STATE_DEPTH_BIAS
    };
    VkPipelineDynamicStateCreateInfo shadowDynamicState{};
    shadowDynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    shadowDynamicState.dynamicStateCount = static_cast<uint32_t>(shadowDynamicStates.size());
    shadowDynamicState.pDynamicStates = shadowDynamicStates.data();

    VkPipelineColorBlendStateCreateInfo shadowColorBlending{};
    shadowColorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    shadowColorBlending.attachmentCount = 0;
    shadowColorBlending.pAttachments = nullptr;

    VkPipelineRenderingCreateInfo shadowRenderingCreateInfo{};
    shadowRenderingCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    shadowRenderingCreateInfo.colorAttachmentCount = 0;
    shadowRenderingCreateInfo.pColorAttachmentFormats = nullptr;
    shadowRenderingCreateInfo.depthAttachmentFormat = m_shadowDepthFormat;

    VkGraphicsPipelineCreateInfo shadowPipelineCreateInfo{};
    shadowPipelineCreateInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    shadowPipelineCreateInfo.pNext = &shadowRenderingCreateInfo;
    shadowPipelineCreateInfo.stageCount = static_cast<uint32_t>(shadowShaderStages.size());
    shadowPipelineCreateInfo.pStages = shadowShaderStages.data();
    shadowPipelineCreateInfo.pVertexInputState = &vertexInputInfo;
    shadowPipelineCreateInfo.pInputAssemblyState = &inputAssembly;
    shadowPipelineCreateInfo.pViewportState = &viewportState;
    shadowPipelineCreateInfo.pRasterizationState = &shadowRasterizer;
    shadowPipelineCreateInfo.pMultisampleState = &shadowMultisampling;
    shadowPipelineCreateInfo.pDepthStencilState = &shadowDepthStencil;
    shadowPipelineCreateInfo.pColorBlendState = &shadowColorBlending;
    shadowPipelineCreateInfo.pDynamicState = &shadowDynamicState;
    shadowPipelineCreateInfo.layout = m_pipelineLayout;
    shadowPipelineCreateInfo.renderPass = VK_NULL_HANDLE;
    shadowPipelineCreateInfo.subpass = 0;

    VkPipeline shadowPipeline = VK_NULL_HANDLE;
    const VkResult shadowPipelineResult = vkCreateGraphicsPipelines(
        m_device,
        VK_NULL_HANDLE,
        1,
        &shadowPipelineCreateInfo,
        nullptr,
        &shadowPipeline
    );

    vkDestroyShaderModule(m_device, shadowFragShaderModule, nullptr);
    vkDestroyShaderModule(m_device, shadowVertShaderModule, nullptr);
    vkDestroyShaderModule(m_device, toneMapFragShaderModule, nullptr);
    vkDestroyShaderModule(m_device, toneMapVertShaderModule, nullptr);
    vkDestroyShaderModule(m_device, skyboxFragShaderModule, nullptr);
    vkDestroyShaderModule(m_device, skyboxVertShaderModule, nullptr);
    vkDestroyShaderModule(m_device, worldFragShaderModule, nullptr);
    vkDestroyShaderModule(m_device, worldVertShaderModule, nullptr);

    if (shadowPipelineResult != VK_SUCCESS) {
        vkDestroyPipeline(m_device, worldPipeline, nullptr);
        vkDestroyPipeline(m_device, previewAddPipeline, nullptr);
        vkDestroyPipeline(m_device, previewRemovePipeline, nullptr);
        vkDestroyPipeline(m_device, skyboxPipeline, nullptr);
        vkDestroyPipeline(m_device, toneMapPipeline, nullptr);
        logVkFailure("vkCreateGraphicsPipelines(shadow)", shadowPipelineResult);
        return false;
    }
    VOX_LOGI("render") << "pipeline config (shadow): depthFormat=" << static_cast<int>(m_shadowDepthFormat)
              << ", depthBias=" << (shadowRasterizer.depthBiasEnable == VK_TRUE ? 1 : 0)
              << ", cullMode=" << static_cast<uint32_t>(shadowRasterizer.cullMode)
              << ", samples=" << static_cast<uint32_t>(shadowMultisampling.rasterizationSamples)
              << "\n";

    VkShaderModule pipeShadowVertShaderModule = VK_NULL_HANDLE;
    VkShaderModule pipeShadowFragShaderModule = VK_NULL_HANDLE;
    if (!createShaderModuleFromFile(
            m_device,
            kPipeShadowVertexShaderPath,
            "pipe_shadow.vert",
            pipeShadowVertShaderModule
        )) {
        vkDestroyPipeline(m_device, shadowPipeline, nullptr);
        vkDestroyPipeline(m_device, worldPipeline, nullptr);
        vkDestroyPipeline(m_device, previewAddPipeline, nullptr);
        vkDestroyPipeline(m_device, previewRemovePipeline, nullptr);
        vkDestroyPipeline(m_device, skyboxPipeline, nullptr);
        vkDestroyPipeline(m_device, toneMapPipeline, nullptr);
        return false;
    }
    if (!createShaderModuleFromFile(
            m_device,
            kPipeShadowFragmentShaderPath,
            "pipe_shadow.frag",
            pipeShadowFragShaderModule
        )) {
        vkDestroyShaderModule(m_device, pipeShadowVertShaderModule, nullptr);
        vkDestroyPipeline(m_device, shadowPipeline, nullptr);
        vkDestroyPipeline(m_device, worldPipeline, nullptr);
        vkDestroyPipeline(m_device, previewAddPipeline, nullptr);
        vkDestroyPipeline(m_device, previewRemovePipeline, nullptr);
        vkDestroyPipeline(m_device, skyboxPipeline, nullptr);
        vkDestroyPipeline(m_device, toneMapPipeline, nullptr);
        return false;
    }

    VkPipelineShaderStageCreateInfo pipeShadowVertexShaderStage{};
    pipeShadowVertexShaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeShadowVertexShaderStage.stage = VK_SHADER_STAGE_VERTEX_BIT;
    pipeShadowVertexShaderStage.module = pipeShadowVertShaderModule;
    pipeShadowVertexShaderStage.pName = "main";

    VkPipelineShaderStageCreateInfo pipeShadowFragmentShaderStage{};
    pipeShadowFragmentShaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeShadowFragmentShaderStage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    pipeShadowFragmentShaderStage.module = pipeShadowFragShaderModule;
    pipeShadowFragmentShaderStage.pName = "main";

    const std::array<VkPipelineShaderStageCreateInfo, 2> pipeShadowShaderStages = {
        pipeShadowVertexShaderStage,
        pipeShadowFragmentShaderStage
    };

    VkVertexInputBindingDescription pipeShadowBindings[2]{};
    pipeShadowBindings[0].binding = 0;
    pipeShadowBindings[0].stride = sizeof(PipeVertex);
    pipeShadowBindings[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    pipeShadowBindings[1].binding = 1;
    pipeShadowBindings[1].stride = sizeof(PipeInstance);
    pipeShadowBindings[1].inputRate = VK_VERTEX_INPUT_RATE_INSTANCE;

    VkVertexInputAttributeDescription pipeShadowAttributes[6]{};
    pipeShadowAttributes[0].location = 0;
    pipeShadowAttributes[0].binding = 0;
    pipeShadowAttributes[0].format = VK_FORMAT_R32G32B32_SFLOAT;
    pipeShadowAttributes[0].offset = static_cast<uint32_t>(offsetof(PipeVertex, position));
    pipeShadowAttributes[1].location = 1;
    pipeShadowAttributes[1].binding = 0;
    pipeShadowAttributes[1].format = VK_FORMAT_R32G32B32_SFLOAT;
    pipeShadowAttributes[1].offset = static_cast<uint32_t>(offsetof(PipeVertex, normal));
    pipeShadowAttributes[2].location = 2;
    pipeShadowAttributes[2].binding = 1;
    pipeShadowAttributes[2].format = VK_FORMAT_R32G32B32A32_SFLOAT;
    pipeShadowAttributes[2].offset = static_cast<uint32_t>(offsetof(PipeInstance, originLength));
    pipeShadowAttributes[3].location = 3;
    pipeShadowAttributes[3].binding = 1;
    pipeShadowAttributes[3].format = VK_FORMAT_R32G32B32A32_SFLOAT;
    pipeShadowAttributes[3].offset = static_cast<uint32_t>(offsetof(PipeInstance, axisRadius));
    pipeShadowAttributes[4].location = 4;
    pipeShadowAttributes[4].binding = 1;
    pipeShadowAttributes[4].format = VK_FORMAT_R32G32B32A32_SFLOAT;
    pipeShadowAttributes[4].offset = static_cast<uint32_t>(offsetof(PipeInstance, tint));
    pipeShadowAttributes[5].location = 5;
    pipeShadowAttributes[5].binding = 1;
    pipeShadowAttributes[5].format = VK_FORMAT_R32G32B32A32_SFLOAT;
    pipeShadowAttributes[5].offset = static_cast<uint32_t>(offsetof(PipeInstance, extensions));

    VkPipelineVertexInputStateCreateInfo pipeShadowVertexInputInfo{};
    pipeShadowVertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    pipeShadowVertexInputInfo.vertexBindingDescriptionCount = 2;
    pipeShadowVertexInputInfo.pVertexBindingDescriptions = pipeShadowBindings;
    pipeShadowVertexInputInfo.vertexAttributeDescriptionCount = 6;
    pipeShadowVertexInputInfo.pVertexAttributeDescriptions = pipeShadowAttributes;

    VkGraphicsPipelineCreateInfo pipeShadowPipelineCreateInfo = shadowPipelineCreateInfo;
    pipeShadowPipelineCreateInfo.stageCount = static_cast<uint32_t>(pipeShadowShaderStages.size());
    pipeShadowPipelineCreateInfo.pStages = pipeShadowShaderStages.data();
    pipeShadowPipelineCreateInfo.pVertexInputState = &pipeShadowVertexInputInfo;
    VkPipelineRasterizationStateCreateInfo pipeShadowRasterizer = shadowRasterizer;
    pipeShadowRasterizer.cullMode = VK_CULL_MODE_NONE;
    pipeShadowPipelineCreateInfo.pRasterizationState = &pipeShadowRasterizer;

    VkPipeline pipeShadowPipeline = VK_NULL_HANDLE;
    const VkResult pipeShadowPipelineResult = vkCreateGraphicsPipelines(
        m_device,
        VK_NULL_HANDLE,
        1,
        &pipeShadowPipelineCreateInfo,
        nullptr,
        &pipeShadowPipeline
    );

    vkDestroyShaderModule(m_device, pipeShadowFragShaderModule, nullptr);
    vkDestroyShaderModule(m_device, pipeShadowVertShaderModule, nullptr);

    if (pipeShadowPipelineResult != VK_SUCCESS) {
        vkDestroyPipeline(m_device, shadowPipeline, nullptr);
        vkDestroyPipeline(m_device, worldPipeline, nullptr);
        vkDestroyPipeline(m_device, previewAddPipeline, nullptr);
        vkDestroyPipeline(m_device, previewRemovePipeline, nullptr);
        vkDestroyPipeline(m_device, skyboxPipeline, nullptr);
        vkDestroyPipeline(m_device, toneMapPipeline, nullptr);
        logVkFailure("vkCreateGraphicsPipelines(pipeShadow)", pipeShadowPipelineResult);
        return false;
    }
    VOX_LOGI("render") << "pipeline config (pipeShadow): cullMode="
              << static_cast<uint32_t>(pipeShadowRasterizer.cullMode)
              << ", depthBias=" << (pipeShadowRasterizer.depthBiasEnable == VK_TRUE ? 1 : 0)
              << "\n";

    VkShaderModule grassShadowVertShaderModule = VK_NULL_HANDLE;
    VkShaderModule grassShadowFragShaderModule = VK_NULL_HANDLE;
    if (!createShaderModuleFromFile(
            m_device,
            kGrassShadowVertexShaderPath,
            "grass_billboard_shadow.vert",
            grassShadowVertShaderModule
        )) {
        vkDestroyPipeline(m_device, pipeShadowPipeline, nullptr);
        vkDestroyPipeline(m_device, shadowPipeline, nullptr);
        vkDestroyPipeline(m_device, worldPipeline, nullptr);
        vkDestroyPipeline(m_device, previewAddPipeline, nullptr);
        vkDestroyPipeline(m_device, previewRemovePipeline, nullptr);
        vkDestroyPipeline(m_device, skyboxPipeline, nullptr);
        vkDestroyPipeline(m_device, toneMapPipeline, nullptr);
        return false;
    }
    if (!createShaderModuleFromFile(
            m_device,
            kGrassShadowFragmentShaderPath,
            "grass_billboard_shadow.frag",
            grassShadowFragShaderModule
        )) {
        vkDestroyShaderModule(m_device, grassShadowVertShaderModule, nullptr);
        vkDestroyPipeline(m_device, pipeShadowPipeline, nullptr);
        vkDestroyPipeline(m_device, shadowPipeline, nullptr);
        vkDestroyPipeline(m_device, worldPipeline, nullptr);
        vkDestroyPipeline(m_device, previewAddPipeline, nullptr);
        vkDestroyPipeline(m_device, previewRemovePipeline, nullptr);
        vkDestroyPipeline(m_device, skyboxPipeline, nullptr);
        vkDestroyPipeline(m_device, toneMapPipeline, nullptr);
        return false;
    }

    VkPipelineShaderStageCreateInfo grassShadowVertexShaderStage{};
    grassShadowVertexShaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    grassShadowVertexShaderStage.stage = VK_SHADER_STAGE_VERTEX_BIT;
    grassShadowVertexShaderStage.module = grassShadowVertShaderModule;
    grassShadowVertexShaderStage.pName = "main";

    VkPipelineShaderStageCreateInfo grassShadowFragmentShaderStage{};
    grassShadowFragmentShaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    grassShadowFragmentShaderStage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    grassShadowFragmentShaderStage.module = grassShadowFragShaderModule;
    grassShadowFragmentShaderStage.pName = "main";

    const std::array<VkPipelineShaderStageCreateInfo, 2> grassShadowShaderStages = {
        grassShadowVertexShaderStage,
        grassShadowFragmentShaderStage
    };

    VkVertexInputBindingDescription grassShadowBindings[2]{};
    grassShadowBindings[0].binding = 0;
    grassShadowBindings[0].stride = sizeof(GrassBillboardVertex);
    grassShadowBindings[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    grassShadowBindings[1].binding = 1;
    grassShadowBindings[1].stride = sizeof(GrassBillboardInstance);
    grassShadowBindings[1].inputRate = VK_VERTEX_INPUT_RATE_INSTANCE;

    VkVertexInputAttributeDescription grassShadowAttributes[4]{};
    grassShadowAttributes[0].location = 0;
    grassShadowAttributes[0].binding = 0;
    grassShadowAttributes[0].format = VK_FORMAT_R32G32_SFLOAT;
    grassShadowAttributes[0].offset = static_cast<uint32_t>(offsetof(GrassBillboardVertex, corner));
    grassShadowAttributes[1].location = 1;
    grassShadowAttributes[1].binding = 0;
    grassShadowAttributes[1].format = VK_FORMAT_R32G32_SFLOAT;
    grassShadowAttributes[1].offset = static_cast<uint32_t>(offsetof(GrassBillboardVertex, uv));
    grassShadowAttributes[2].location = 2;
    grassShadowAttributes[2].binding = 0;
    grassShadowAttributes[2].format = VK_FORMAT_R32_SFLOAT;
    grassShadowAttributes[2].offset = static_cast<uint32_t>(offsetof(GrassBillboardVertex, plane));
    grassShadowAttributes[3].location = 3;
    grassShadowAttributes[3].binding = 1;
    grassShadowAttributes[3].format = VK_FORMAT_R32G32B32A32_SFLOAT;
    grassShadowAttributes[3].offset = static_cast<uint32_t>(offsetof(GrassBillboardInstance, worldPosYaw));

    VkPipelineVertexInputStateCreateInfo grassShadowVertexInputInfo{};
    grassShadowVertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    grassShadowVertexInputInfo.vertexBindingDescriptionCount = 2;
    grassShadowVertexInputInfo.pVertexBindingDescriptions = grassShadowBindings;
    grassShadowVertexInputInfo.vertexAttributeDescriptionCount = 4;
    grassShadowVertexInputInfo.pVertexAttributeDescriptions = grassShadowAttributes;

    VkGraphicsPipelineCreateInfo grassShadowPipelineCreateInfo = shadowPipelineCreateInfo;
    grassShadowPipelineCreateInfo.stageCount = static_cast<uint32_t>(grassShadowShaderStages.size());
    grassShadowPipelineCreateInfo.pStages = grassShadowShaderStages.data();
    grassShadowPipelineCreateInfo.pVertexInputState = &grassShadowVertexInputInfo;
    VkPipelineRasterizationStateCreateInfo grassShadowRasterizer = shadowRasterizer;
    grassShadowRasterizer.cullMode = VK_CULL_MODE_NONE;
    grassShadowPipelineCreateInfo.pRasterizationState = &grassShadowRasterizer;

    VkPipeline grassShadowPipeline = VK_NULL_HANDLE;
    const VkResult grassShadowPipelineResult = vkCreateGraphicsPipelines(
        m_device,
        VK_NULL_HANDLE,
        1,
        &grassShadowPipelineCreateInfo,
        nullptr,
        &grassShadowPipeline
    );

    vkDestroyShaderModule(m_device, grassShadowFragShaderModule, nullptr);
    vkDestroyShaderModule(m_device, grassShadowVertShaderModule, nullptr);

    if (grassShadowPipelineResult != VK_SUCCESS) {
        vkDestroyPipeline(m_device, pipeShadowPipeline, nullptr);
        vkDestroyPipeline(m_device, shadowPipeline, nullptr);
        vkDestroyPipeline(m_device, worldPipeline, nullptr);
        vkDestroyPipeline(m_device, previewAddPipeline, nullptr);
        vkDestroyPipeline(m_device, previewRemovePipeline, nullptr);
        vkDestroyPipeline(m_device, skyboxPipeline, nullptr);
        vkDestroyPipeline(m_device, toneMapPipeline, nullptr);
        logVkFailure("vkCreateGraphicsPipelines(grassShadow)", grassShadowPipelineResult);
        return false;
    }
    VOX_LOGI("render") << "pipeline config (grassShadow): cullMode="
              << static_cast<uint32_t>(grassShadowRasterizer.cullMode)
              << ", depthBias=" << (grassShadowRasterizer.depthBiasEnable == VK_TRUE ? 1 : 0)
              << "\n";

    if (m_pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_device, m_pipeline, nullptr);
    }
    if (m_skyboxPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_device, m_skyboxPipeline, nullptr);
    }
    if (m_shadowPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_device, m_shadowPipeline, nullptr);
    }
    if (m_pipeShadowPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_device, m_pipeShadowPipeline, nullptr);
    }
    if (m_grassBillboardShadowPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_device, m_grassBillboardShadowPipeline, nullptr);
    }
    if (m_tonemapPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_device, m_tonemapPipeline, nullptr);
    }
    if (m_previewAddPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_device, m_previewAddPipeline, nullptr);
    }
    if (m_previewRemovePipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_device, m_previewRemovePipeline, nullptr);
    }
    m_pipeline = worldPipeline;
    m_skyboxPipeline = skyboxPipeline;
    m_shadowPipeline = shadowPipeline;
    m_pipeShadowPipeline = pipeShadowPipeline;
    m_grassBillboardShadowPipeline = grassShadowPipeline;
    m_tonemapPipeline = toneMapPipeline;
    m_previewAddPipeline = previewAddPipeline;
    m_previewRemovePipeline = previewRemovePipeline;
    setObjectName(VK_OBJECT_TYPE_PIPELINE, vkHandleToUint64(m_pipeline), "pipeline.world");
    setObjectName(VK_OBJECT_TYPE_PIPELINE, vkHandleToUint64(m_skyboxPipeline), "pipeline.skybox");
    setObjectName(VK_OBJECT_TYPE_PIPELINE, vkHandleToUint64(m_shadowPipeline), "pipeline.shadow.voxels");
    setObjectName(VK_OBJECT_TYPE_PIPELINE, vkHandleToUint64(m_pipeShadowPipeline), "pipeline.shadow.pipes");
    setObjectName(
        VK_OBJECT_TYPE_PIPELINE,
        vkHandleToUint64(m_grassBillboardShadowPipeline),
        "pipeline.shadow.grass"
    );
    setObjectName(VK_OBJECT_TYPE_PIPELINE, vkHandleToUint64(m_tonemapPipeline), "pipeline.tonemap");
    setObjectName(VK_OBJECT_TYPE_PIPELINE, vkHandleToUint64(m_previewAddPipeline), "pipeline.preview.add");
    setObjectName(VK_OBJECT_TYPE_PIPELINE, vkHandleToUint64(m_previewRemovePipeline), "pipeline.preview.remove");
    VOX_LOGI("render") << "graphics pipelines ready (shadow + hdr scene + tonemap + preview="
              << (m_supportsWireframePreview ? "wireframe" : "ghost")
              << ")\n";
    return true;
}

bool Renderer::createPipePipeline() {
    if (m_pipelineLayout == VK_NULL_HANDLE) {
        return false;
    }
    if (m_depthFormat == VK_FORMAT_UNDEFINED || m_hdrColorFormat == VK_FORMAT_UNDEFINED) {
        return false;
    }

    constexpr const char* kPipeVertexShaderPath = "../src/render/shaders/pipe_instanced.vert.slang.spv";
    constexpr const char* kPipeFragmentShaderPath = "../src/render/shaders/pipe_instanced.frag.slang.spv";

    VkShaderModule pipeVertShaderModule = VK_NULL_HANDLE;
    VkShaderModule pipeFragShaderModule = VK_NULL_HANDLE;
    if (!createShaderModuleFromFile(
            m_device,
            kPipeVertexShaderPath,
            "pipe_instanced.vert",
            pipeVertShaderModule
        )) {
        return false;
    }
    if (!createShaderModuleFromFile(
            m_device,
            kPipeFragmentShaderPath,
            "pipe_instanced.frag",
            pipeFragShaderModule
        )) {
        vkDestroyShaderModule(m_device, pipeVertShaderModule, nullptr);
        return false;
    }

    VkPipelineShaderStageCreateInfo pipeVertexShaderStage{};
    pipeVertexShaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeVertexShaderStage.stage = VK_SHADER_STAGE_VERTEX_BIT;
    pipeVertexShaderStage.module = pipeVertShaderModule;
    pipeVertexShaderStage.pName = "main";

    VkPipelineShaderStageCreateInfo pipeFragmentShaderStage{};
    pipeFragmentShaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeFragmentShaderStage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    pipeFragmentShaderStage.module = pipeFragShaderModule;
    pipeFragmentShaderStage.pName = "main";

    const std::array<VkPipelineShaderStageCreateInfo, 2> pipeShaderStages = {
        pipeVertexShaderStage,
        pipeFragmentShaderStage
    };

    VkVertexInputBindingDescription bindings[2]{};
    bindings[0].binding = 0;
    bindings[0].stride = sizeof(PipeVertex);
    bindings[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    bindings[1].binding = 1;
    bindings[1].stride = sizeof(PipeInstance);
    bindings[1].inputRate = VK_VERTEX_INPUT_RATE_INSTANCE;

    VkVertexInputAttributeDescription attributes[6]{};
    attributes[0].location = 0;
    attributes[0].binding = 0;
    attributes[0].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributes[0].offset = static_cast<uint32_t>(offsetof(PipeVertex, position));
    attributes[1].location = 1;
    attributes[1].binding = 0;
    attributes[1].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributes[1].offset = static_cast<uint32_t>(offsetof(PipeVertex, normal));
    attributes[2].location = 2;
    attributes[2].binding = 1;
    attributes[2].format = VK_FORMAT_R32G32B32A32_SFLOAT;
    attributes[2].offset = static_cast<uint32_t>(offsetof(PipeInstance, originLength));
    attributes[3].location = 3;
    attributes[3].binding = 1;
    attributes[3].format = VK_FORMAT_R32G32B32A32_SFLOAT;
    attributes[3].offset = static_cast<uint32_t>(offsetof(PipeInstance, axisRadius));
    attributes[4].location = 4;
    attributes[4].binding = 1;
    attributes[4].format = VK_FORMAT_R32G32B32A32_SFLOAT;
    attributes[4].offset = static_cast<uint32_t>(offsetof(PipeInstance, tint));
    attributes[5].location = 5;
    attributes[5].binding = 1;
    attributes[5].format = VK_FORMAT_R32G32B32A32_SFLOAT;
    attributes[5].offset = static_cast<uint32_t>(offsetof(PipeInstance, extensions));

    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = 2;
    vertexInputInfo.pVertexBindingDescriptions = bindings;
    vertexInputInfo.vertexAttributeDescriptionCount = 6;
    vertexInputInfo.pVertexAttributeDescriptions = attributes;

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_NONE;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples = m_colorSampleCount;

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_GREATER_OR_EQUAL;
    depthStencil.depthBoundsTestEnable = VK_FALSE;
    depthStencil.stencilTestEnable = VK_FALSE;

    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;

    std::array<VkDynamicState, 2> dynamicStates = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR,
    };
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
    dynamicState.pDynamicStates = dynamicStates.data();

    VkPipelineRenderingCreateInfo renderingCreateInfo{};
    renderingCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    renderingCreateInfo.colorAttachmentCount = 1;
    renderingCreateInfo.pColorAttachmentFormats = &m_hdrColorFormat;
    renderingCreateInfo.depthAttachmentFormat = m_depthFormat;

    VkGraphicsPipelineCreateInfo pipelineCreateInfo{};
    pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineCreateInfo.pNext = &renderingCreateInfo;
    pipelineCreateInfo.stageCount = static_cast<uint32_t>(pipeShaderStages.size());
    pipelineCreateInfo.pStages = pipeShaderStages.data();
    pipelineCreateInfo.pVertexInputState = &vertexInputInfo;
    pipelineCreateInfo.pInputAssemblyState = &inputAssembly;
    pipelineCreateInfo.pViewportState = &viewportState;
    pipelineCreateInfo.pRasterizationState = &rasterizer;
    pipelineCreateInfo.pMultisampleState = &multisampling;
    pipelineCreateInfo.pDepthStencilState = &depthStencil;
    pipelineCreateInfo.pColorBlendState = &colorBlending;
    pipelineCreateInfo.pDynamicState = &dynamicState;
    pipelineCreateInfo.layout = m_pipelineLayout;
    pipelineCreateInfo.renderPass = VK_NULL_HANDLE;
    pipelineCreateInfo.subpass = 0;

    VkPipeline pipePipeline = VK_NULL_HANDLE;
    const VkResult pipePipelineResult = vkCreateGraphicsPipelines(
        m_device,
        VK_NULL_HANDLE,
        1,
        &pipelineCreateInfo,
        nullptr,
        &pipePipeline
    );

    vkDestroyShaderModule(m_device, pipeFragShaderModule, nullptr);
    vkDestroyShaderModule(m_device, pipeVertShaderModule, nullptr);

    if (pipePipelineResult != VK_SUCCESS) {
        logVkFailure("vkCreateGraphicsPipelines(pipe)", pipePipelineResult);
        return false;
    }
    VOX_LOGI("render") << "pipeline config (pipeLit): samples=" << static_cast<uint32_t>(m_colorSampleCount)
              << ", cullMode=" << static_cast<uint32_t>(rasterizer.cullMode)
              << ", depthCompare=" << static_cast<uint32_t>(depthStencil.depthCompareOp)
              << "\n";

    constexpr const char* kGrassBillboardVertexShaderPath = "../src/render/shaders/grass_billboard.vert.slang.spv";
    constexpr const char* kGrassBillboardFragmentShaderPath = "../src/render/shaders/grass_billboard.frag.slang.spv";
    VkShaderModule grassVertShaderModule = VK_NULL_HANDLE;
    VkShaderModule grassFragShaderModule = VK_NULL_HANDLE;
    if (!createShaderModuleFromFile(
            m_device,
            kGrassBillboardVertexShaderPath,
            "grass_billboard.vert",
            grassVertShaderModule
        )) {
        vkDestroyPipeline(m_device, pipePipeline, nullptr);
        return false;
    }
    if (!createShaderModuleFromFile(
            m_device,
            kGrassBillboardFragmentShaderPath,
            "grass_billboard.frag",
            grassFragShaderModule
        )) {
        vkDestroyShaderModule(m_device, grassVertShaderModule, nullptr);
        vkDestroyPipeline(m_device, pipePipeline, nullptr);
        return false;
    }

    VkPipelineShaderStageCreateInfo grassVertexShaderStage{};
    grassVertexShaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    grassVertexShaderStage.stage = VK_SHADER_STAGE_VERTEX_BIT;
    grassVertexShaderStage.module = grassVertShaderModule;
    grassVertexShaderStage.pName = "main";

    VkPipelineShaderStageCreateInfo grassFragmentShaderStage{};
    grassFragmentShaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    grassFragmentShaderStage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    grassFragmentShaderStage.module = grassFragShaderModule;
    grassFragmentShaderStage.pName = "main";

    const std::array<VkPipelineShaderStageCreateInfo, 2> grassShaderStages = {
        grassVertexShaderStage,
        grassFragmentShaderStage
    };

    VkVertexInputBindingDescription grassBindings[2]{};
    grassBindings[0].binding = 0;
    grassBindings[0].stride = sizeof(GrassBillboardVertex);
    grassBindings[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    grassBindings[1].binding = 1;
    grassBindings[1].stride = sizeof(GrassBillboardInstance);
    grassBindings[1].inputRate = VK_VERTEX_INPUT_RATE_INSTANCE;

    VkVertexInputAttributeDescription grassAttributes[4]{};
    grassAttributes[0].location = 0;
    grassAttributes[0].binding = 0;
    grassAttributes[0].format = VK_FORMAT_R32G32_SFLOAT;
    grassAttributes[0].offset = static_cast<uint32_t>(offsetof(GrassBillboardVertex, corner));
    grassAttributes[1].location = 1;
    grassAttributes[1].binding = 0;
    grassAttributes[1].format = VK_FORMAT_R32G32_SFLOAT;
    grassAttributes[1].offset = static_cast<uint32_t>(offsetof(GrassBillboardVertex, uv));
    grassAttributes[2].location = 2;
    grassAttributes[2].binding = 0;
    grassAttributes[2].format = VK_FORMAT_R32_SFLOAT;
    grassAttributes[2].offset = static_cast<uint32_t>(offsetof(GrassBillboardVertex, plane));
    grassAttributes[3].location = 3;
    grassAttributes[3].binding = 1;
    grassAttributes[3].format = VK_FORMAT_R32G32B32A32_SFLOAT;
    grassAttributes[3].offset = static_cast<uint32_t>(offsetof(GrassBillboardInstance, worldPosYaw));

    VkPipelineVertexInputStateCreateInfo grassVertexInputInfo{};
    grassVertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    grassVertexInputInfo.vertexBindingDescriptionCount = 2;
    grassVertexInputInfo.pVertexBindingDescriptions = grassBindings;
    grassVertexInputInfo.vertexAttributeDescriptionCount = 4;
    grassVertexInputInfo.pVertexAttributeDescriptions = grassAttributes;

    VkGraphicsPipelineCreateInfo grassPipelineCreateInfo = pipelineCreateInfo;
    grassPipelineCreateInfo.stageCount = static_cast<uint32_t>(grassShaderStages.size());
    grassPipelineCreateInfo.pStages = grassShaderStages.data();
    grassPipelineCreateInfo.pVertexInputState = &grassVertexInputInfo;
    VkPipelineRasterizationStateCreateInfo grassRasterizer = rasterizer;
    grassRasterizer.cullMode = VK_CULL_MODE_NONE;
    grassPipelineCreateInfo.pRasterizationState = &grassRasterizer;
    VkPipelineDepthStencilStateCreateInfo grassDepthStencil = depthStencil;
    grassDepthStencil.depthWriteEnable = VK_TRUE;
    grassPipelineCreateInfo.pDepthStencilState = &grassDepthStencil;
    VkPipelineMultisampleStateCreateInfo grassMultisampling = multisampling;
    grassMultisampling.alphaToCoverageEnable = VK_FALSE;
    grassPipelineCreateInfo.pMultisampleState = &grassMultisampling;

    VkPipeline grassBillboardPipeline = VK_NULL_HANDLE;
    const VkResult grassPipelineResult = vkCreateGraphicsPipelines(
        m_device,
        VK_NULL_HANDLE,
        1,
        &grassPipelineCreateInfo,
        nullptr,
        &grassBillboardPipeline
    );
    vkDestroyShaderModule(m_device, grassFragShaderModule, nullptr);
    vkDestroyShaderModule(m_device, grassVertShaderModule, nullptr);
    if (grassPipelineResult != VK_SUCCESS) {
        logVkFailure("vkCreateGraphicsPipelines(grassBillboard)", grassPipelineResult);
        vkDestroyPipeline(m_device, pipePipeline, nullptr);
        return false;
    }
    VOX_LOGI("render") << "pipeline config (grassBillboard): samples=" << static_cast<uint32_t>(m_colorSampleCount)
              << ", cullMode=" << static_cast<uint32_t>(grassRasterizer.cullMode)
              << ", depthCompare=" << static_cast<uint32_t>(depthStencil.depthCompareOp)
              << "\n";

    if (m_pipePipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_device, m_pipePipeline, nullptr);
    }
    if (m_grassBillboardPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_device, m_grassBillboardPipeline, nullptr);
    }
    m_pipePipeline = pipePipeline;
    m_grassBillboardPipeline = grassBillboardPipeline;
    setObjectName(VK_OBJECT_TYPE_PIPELINE, vkHandleToUint64(m_pipePipeline), "pipeline.pipe.lit");
    setObjectName(VK_OBJECT_TYPE_PIPELINE, vkHandleToUint64(m_grassBillboardPipeline), "pipeline.grass.billboard");
    return true;
}

bool Renderer::createAoPipelines() {
    if (m_pipelineLayout == VK_NULL_HANDLE) {
        return false;
    }
    if (
        m_normalDepthFormat == VK_FORMAT_UNDEFINED ||
        m_ssaoFormat == VK_FORMAT_UNDEFINED ||
        m_depthFormat == VK_FORMAT_UNDEFINED
    ) {
        return false;
    }

    constexpr const char* kVoxelVertShaderPath = "../src/render/shaders/voxel_packed.vert.slang.spv";
    constexpr const char* kVoxelNormalDepthFragShaderPath = "../src/render/shaders/voxel_normaldepth.frag.slang.spv";
    constexpr const char* kPipeVertShaderPath = "../src/render/shaders/pipe_instanced.vert.slang.spv";
    constexpr const char* kPipeNormalDepthFragShaderPath = "../src/render/shaders/pipe_normaldepth.frag.slang.spv";
    constexpr const char* kFullscreenVertShaderPath = "../src/render/shaders/tone_map.vert.slang.spv";
    constexpr const char* kSsaoFragShaderPath = "../src/render/shaders/ssao.frag.slang.spv";
    constexpr const char* kSsaoBlurFragShaderPath = "../src/render/shaders/ssao_blur.frag.slang.spv";

    VkShaderModule voxelVertShaderModule = VK_NULL_HANDLE;
    VkShaderModule voxelNormalDepthFragShaderModule = VK_NULL_HANDLE;
    VkShaderModule pipeVertShaderModule = VK_NULL_HANDLE;
    VkShaderModule pipeNormalDepthFragShaderModule = VK_NULL_HANDLE;
    VkShaderModule fullscreenVertShaderModule = VK_NULL_HANDLE;
    VkShaderModule ssaoFragShaderModule = VK_NULL_HANDLE;
    VkShaderModule ssaoBlurFragShaderModule = VK_NULL_HANDLE;

    auto destroyShaderModules = [&]() {
        if (ssaoBlurFragShaderModule != VK_NULL_HANDLE) {
            vkDestroyShaderModule(m_device, ssaoBlurFragShaderModule, nullptr);
            ssaoBlurFragShaderModule = VK_NULL_HANDLE;
        }
        if (ssaoFragShaderModule != VK_NULL_HANDLE) {
            vkDestroyShaderModule(m_device, ssaoFragShaderModule, nullptr);
            ssaoFragShaderModule = VK_NULL_HANDLE;
        }
        if (fullscreenVertShaderModule != VK_NULL_HANDLE) {
            vkDestroyShaderModule(m_device, fullscreenVertShaderModule, nullptr);
            fullscreenVertShaderModule = VK_NULL_HANDLE;
        }
        if (pipeNormalDepthFragShaderModule != VK_NULL_HANDLE) {
            vkDestroyShaderModule(m_device, pipeNormalDepthFragShaderModule, nullptr);
            pipeNormalDepthFragShaderModule = VK_NULL_HANDLE;
        }
        if (pipeVertShaderModule != VK_NULL_HANDLE) {
            vkDestroyShaderModule(m_device, pipeVertShaderModule, nullptr);
            pipeVertShaderModule = VK_NULL_HANDLE;
        }
        if (voxelNormalDepthFragShaderModule != VK_NULL_HANDLE) {
            vkDestroyShaderModule(m_device, voxelNormalDepthFragShaderModule, nullptr);
            voxelNormalDepthFragShaderModule = VK_NULL_HANDLE;
        }
        if (voxelVertShaderModule != VK_NULL_HANDLE) {
            vkDestroyShaderModule(m_device, voxelVertShaderModule, nullptr);
            voxelVertShaderModule = VK_NULL_HANDLE;
        }
    };

    if (!createShaderModuleFromFile(
            m_device,
            kVoxelVertShaderPath,
            "voxel_packed.vert",
            voxelVertShaderModule
        )) {
        return false;
    }
    if (!createShaderModuleFromFile(
            m_device,
            kVoxelNormalDepthFragShaderPath,
            "voxel_normaldepth.frag",
            voxelNormalDepthFragShaderModule
        )) {
        destroyShaderModules();
        return false;
    }
    if (!createShaderModuleFromFile(
            m_device,
            kPipeVertShaderPath,
            "pipe_instanced.vert",
            pipeVertShaderModule
        )) {
        destroyShaderModules();
        return false;
    }
    if (!createShaderModuleFromFile(
            m_device,
            kPipeNormalDepthFragShaderPath,
            "pipe_normaldepth.frag",
            pipeNormalDepthFragShaderModule
        )) {
        destroyShaderModules();
        return false;
    }
    if (!createShaderModuleFromFile(
            m_device,
            kFullscreenVertShaderPath,
            "tone_map.vert",
            fullscreenVertShaderModule
        )) {
        destroyShaderModules();
        return false;
    }
    if (!createShaderModuleFromFile(
            m_device,
            kSsaoFragShaderPath,
            "ssao.frag",
            ssaoFragShaderModule
        )) {
        destroyShaderModules();
        return false;
    }
    if (!createShaderModuleFromFile(
            m_device,
            kSsaoBlurFragShaderPath,
            "ssao_blur.frag",
            ssaoBlurFragShaderModule
        )) {
        destroyShaderModules();
        return false;
    }

    VkPipeline voxelNormalDepthPipeline = VK_NULL_HANDLE;
    VkPipeline pipeNormalDepthPipeline = VK_NULL_HANDLE;
    VkPipeline ssaoPipeline = VK_NULL_HANDLE;
    VkPipeline ssaoBlurPipeline = VK_NULL_HANDLE;
    auto destroyNewPipelines = [&]() {
        if (ssaoBlurPipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(m_device, ssaoBlurPipeline, nullptr);
            ssaoBlurPipeline = VK_NULL_HANDLE;
        }
        if (ssaoPipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(m_device, ssaoPipeline, nullptr);
            ssaoPipeline = VK_NULL_HANDLE;
        }
        if (pipeNormalDepthPipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(m_device, pipeNormalDepthPipeline, nullptr);
            pipeNormalDepthPipeline = VK_NULL_HANDLE;
        }
        if (voxelNormalDepthPipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(m_device, voxelNormalDepthPipeline, nullptr);
            voxelNormalDepthPipeline = VK_NULL_HANDLE;
        }
    };

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_GREATER_OR_EQUAL;
    depthStencil.depthBoundsTestEnable = VK_FALSE;
    depthStencil.stencilTestEnable = VK_FALSE;

    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;

    std::array<VkDynamicState, 2> dynamicStates = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR,
    };
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
    dynamicState.pDynamicStates = dynamicStates.data();

    VkPipelineRenderingCreateInfo normalDepthRenderingCreateInfo{};
    normalDepthRenderingCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    normalDepthRenderingCreateInfo.colorAttachmentCount = 1;
    normalDepthRenderingCreateInfo.pColorAttachmentFormats = &m_normalDepthFormat;
    normalDepthRenderingCreateInfo.depthAttachmentFormat = m_depthFormat;

    VkGraphicsPipelineCreateInfo pipelineCreateInfo{};
    pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineCreateInfo.pNext = &normalDepthRenderingCreateInfo;
    pipelineCreateInfo.pInputAssemblyState = &inputAssembly;
    pipelineCreateInfo.pViewportState = &viewportState;
    pipelineCreateInfo.pRasterizationState = &rasterizer;
    pipelineCreateInfo.pMultisampleState = &multisampling;
    pipelineCreateInfo.pDepthStencilState = &depthStencil;
    pipelineCreateInfo.pColorBlendState = &colorBlending;
    pipelineCreateInfo.pDynamicState = &dynamicState;
    pipelineCreateInfo.layout = m_pipelineLayout;
    pipelineCreateInfo.renderPass = VK_NULL_HANDLE;
    pipelineCreateInfo.subpass = 0;

    // Voxel normal-depth pipeline.
    VkPipelineShaderStageCreateInfo voxelStageInfos[2]{};
    voxelStageInfos[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    voxelStageInfos[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    voxelStageInfos[0].module = voxelVertShaderModule;
    voxelStageInfos[0].pName = "main";
    voxelStageInfos[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    voxelStageInfos[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    voxelStageInfos[1].module = voxelNormalDepthFragShaderModule;
    voxelStageInfos[1].pName = "main";

    VkVertexInputBindingDescription voxelBindings[2]{};
    voxelBindings[0].binding = 0;
    voxelBindings[0].stride = sizeof(world::PackedVoxelVertex);
    voxelBindings[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    voxelBindings[1].binding = 1;
    voxelBindings[1].stride = sizeof(ChunkInstanceData);
    voxelBindings[1].inputRate = VK_VERTEX_INPUT_RATE_INSTANCE;
    VkVertexInputAttributeDescription voxelAttributes[2]{};
    voxelAttributes[0].location = 0;
    voxelAttributes[0].binding = 0;
    voxelAttributes[0].format = VK_FORMAT_R32_UINT;
    voxelAttributes[0].offset = 0;
    voxelAttributes[1].location = 1;
    voxelAttributes[1].binding = 1;
    voxelAttributes[1].format = VK_FORMAT_R32G32B32A32_SFLOAT;
    voxelAttributes[1].offset = 0;
    VkPipelineVertexInputStateCreateInfo voxelVertexInputInfo{};
    voxelVertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    voxelVertexInputInfo.vertexBindingDescriptionCount = 2;
    voxelVertexInputInfo.pVertexBindingDescriptions = voxelBindings;
    voxelVertexInputInfo.vertexAttributeDescriptionCount = 2;
    voxelVertexInputInfo.pVertexAttributeDescriptions = voxelAttributes;

    pipelineCreateInfo.stageCount = 2;
    pipelineCreateInfo.pStages = voxelStageInfos;
    pipelineCreateInfo.pVertexInputState = &voxelVertexInputInfo;
    const VkResult voxelPipelineResult = vkCreateGraphicsPipelines(
        m_device,
        VK_NULL_HANDLE,
        1,
        &pipelineCreateInfo,
        nullptr,
        &voxelNormalDepthPipeline
    );
    if (voxelPipelineResult != VK_SUCCESS) {
        logVkFailure("vkCreateGraphicsPipelines(voxelNormalDepth)", voxelPipelineResult);
        destroyNewPipelines();
        destroyShaderModules();
        return false;
    }

    // Pipe normal-depth pipeline.
    VkPipelineShaderStageCreateInfo pipeStageInfos[2]{};
    pipeStageInfos[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeStageInfos[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    pipeStageInfos[0].module = pipeVertShaderModule;
    pipeStageInfos[0].pName = "main";
    pipeStageInfos[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeStageInfos[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    pipeStageInfos[1].module = pipeNormalDepthFragShaderModule;
    pipeStageInfos[1].pName = "main";

    VkVertexInputBindingDescription pipeBindings[2]{};
    pipeBindings[0].binding = 0;
    pipeBindings[0].stride = sizeof(PipeVertex);
    pipeBindings[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    pipeBindings[1].binding = 1;
    pipeBindings[1].stride = sizeof(PipeInstance);
    pipeBindings[1].inputRate = VK_VERTEX_INPUT_RATE_INSTANCE;

    VkVertexInputAttributeDescription pipeAttributes[6]{};
    pipeAttributes[0].location = 0;
    pipeAttributes[0].binding = 0;
    pipeAttributes[0].format = VK_FORMAT_R32G32B32_SFLOAT;
    pipeAttributes[0].offset = static_cast<uint32_t>(offsetof(PipeVertex, position));
    pipeAttributes[1].location = 1;
    pipeAttributes[1].binding = 0;
    pipeAttributes[1].format = VK_FORMAT_R32G32B32_SFLOAT;
    pipeAttributes[1].offset = static_cast<uint32_t>(offsetof(PipeVertex, normal));
    pipeAttributes[2].location = 2;
    pipeAttributes[2].binding = 1;
    pipeAttributes[2].format = VK_FORMAT_R32G32B32A32_SFLOAT;
    pipeAttributes[2].offset = static_cast<uint32_t>(offsetof(PipeInstance, originLength));
    pipeAttributes[3].location = 3;
    pipeAttributes[3].binding = 1;
    pipeAttributes[3].format = VK_FORMAT_R32G32B32A32_SFLOAT;
    pipeAttributes[3].offset = static_cast<uint32_t>(offsetof(PipeInstance, axisRadius));
    pipeAttributes[4].location = 4;
    pipeAttributes[4].binding = 1;
    pipeAttributes[4].format = VK_FORMAT_R32G32B32A32_SFLOAT;
    pipeAttributes[4].offset = static_cast<uint32_t>(offsetof(PipeInstance, tint));
    pipeAttributes[5].location = 5;
    pipeAttributes[5].binding = 1;
    pipeAttributes[5].format = VK_FORMAT_R32G32B32A32_SFLOAT;
    pipeAttributes[5].offset = static_cast<uint32_t>(offsetof(PipeInstance, extensions));

    VkPipelineVertexInputStateCreateInfo pipeVertexInputInfo{};
    pipeVertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    pipeVertexInputInfo.vertexBindingDescriptionCount = 2;
    pipeVertexInputInfo.pVertexBindingDescriptions = pipeBindings;
    pipeVertexInputInfo.vertexAttributeDescriptionCount = 6;
    pipeVertexInputInfo.pVertexAttributeDescriptions = pipeAttributes;

    VkPipelineRasterizationStateCreateInfo pipeRasterizer = rasterizer;
    pipeRasterizer.cullMode = VK_CULL_MODE_NONE;

    pipelineCreateInfo.pStages = pipeStageInfos;
    pipelineCreateInfo.pVertexInputState = &pipeVertexInputInfo;
    pipelineCreateInfo.pRasterizationState = &pipeRasterizer;
    const VkResult pipePipelineResult = vkCreateGraphicsPipelines(
        m_device,
        VK_NULL_HANDLE,
        1,
        &pipelineCreateInfo,
        nullptr,
        &pipeNormalDepthPipeline
    );
    if (pipePipelineResult != VK_SUCCESS) {
        logVkFailure("vkCreateGraphicsPipelines(pipeNormalDepth)", pipePipelineResult);
        destroyNewPipelines();
        destroyShaderModules();
        return false;
    }

    // SSAO fullscreen pipelines.
    VkPipelineShaderStageCreateInfo ssaoStageInfos[2]{};
    ssaoStageInfos[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    ssaoStageInfos[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    ssaoStageInfos[0].module = fullscreenVertShaderModule;
    ssaoStageInfos[0].pName = "main";
    ssaoStageInfos[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    ssaoStageInfos[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    ssaoStageInfos[1].module = ssaoFragShaderModule;
    ssaoStageInfos[1].pName = "main";

    struct SsaoSpecializationData {
        std::int32_t sampleCount = 32; // constant_id 0
        float power = 1.4f;            // constant_id 1
        std::int32_t blurRadius = 6;   // constant_id 2
        float blurSigma = 3.0f;        // constant_id 3
    };
    const SsaoSpecializationData ssaoSpecializationData{};
    const std::array<VkSpecializationMapEntry, 2> ssaoSpecializationMapEntries = {{
        VkSpecializationMapEntry{
            0u,
            static_cast<uint32_t>(offsetof(SsaoSpecializationData, sampleCount)),
            sizeof(std::int32_t)
        },
        VkSpecializationMapEntry{
            1u,
            static_cast<uint32_t>(offsetof(SsaoSpecializationData, power)),
            sizeof(float)
        }
    }};
    const VkSpecializationInfo ssaoSpecializationInfo{
        static_cast<uint32_t>(ssaoSpecializationMapEntries.size()),
        ssaoSpecializationMapEntries.data(),
        sizeof(ssaoSpecializationData),
        &ssaoSpecializationData
    };
    const std::array<VkSpecializationMapEntry, 2> ssaoBlurSpecializationMapEntries = {{
        VkSpecializationMapEntry{
            2u,
            static_cast<uint32_t>(offsetof(SsaoSpecializationData, blurRadius)),
            sizeof(std::int32_t)
        },
        VkSpecializationMapEntry{
            3u,
            static_cast<uint32_t>(offsetof(SsaoSpecializationData, blurSigma)),
            sizeof(float)
        }
    }};
    const VkSpecializationInfo ssaoBlurSpecializationInfo{
        static_cast<uint32_t>(ssaoBlurSpecializationMapEntries.size()),
        ssaoBlurSpecializationMapEntries.data(),
        sizeof(ssaoSpecializationData),
        &ssaoSpecializationData
    };
    ssaoStageInfos[1].pSpecializationInfo = &ssaoSpecializationInfo;

    VkPipelineVertexInputStateCreateInfo fullscreenVertexInputInfo{};
    fullscreenVertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    VkPipelineRasterizationStateCreateInfo fullscreenRasterizer = rasterizer;
    fullscreenRasterizer.cullMode = VK_CULL_MODE_NONE;

    VkPipelineDepthStencilStateCreateInfo fullscreenDepthStencil{};
    fullscreenDepthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    fullscreenDepthStencil.depthTestEnable = VK_FALSE;
    fullscreenDepthStencil.depthWriteEnable = VK_FALSE;
    fullscreenDepthStencil.depthBoundsTestEnable = VK_FALSE;
    fullscreenDepthStencil.stencilTestEnable = VK_FALSE;

    VkPipelineRenderingCreateInfo ssaoRenderingCreateInfo{};
    ssaoRenderingCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    ssaoRenderingCreateInfo.colorAttachmentCount = 1;
    ssaoRenderingCreateInfo.pColorAttachmentFormats = &m_ssaoFormat;
    ssaoRenderingCreateInfo.depthAttachmentFormat = VK_FORMAT_UNDEFINED;

    VkGraphicsPipelineCreateInfo ssaoPipelineCreateInfo{};
    ssaoPipelineCreateInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    ssaoPipelineCreateInfo.pNext = &ssaoRenderingCreateInfo;
    ssaoPipelineCreateInfo.stageCount = 2;
    ssaoPipelineCreateInfo.pStages = ssaoStageInfos;
    ssaoPipelineCreateInfo.pVertexInputState = &fullscreenVertexInputInfo;
    ssaoPipelineCreateInfo.pInputAssemblyState = &inputAssembly;
    ssaoPipelineCreateInfo.pViewportState = &viewportState;
    ssaoPipelineCreateInfo.pRasterizationState = &fullscreenRasterizer;
    ssaoPipelineCreateInfo.pMultisampleState = &multisampling;
    ssaoPipelineCreateInfo.pDepthStencilState = &fullscreenDepthStencil;
    ssaoPipelineCreateInfo.pColorBlendState = &colorBlending;
    ssaoPipelineCreateInfo.pDynamicState = &dynamicState;
    ssaoPipelineCreateInfo.layout = m_pipelineLayout;
    ssaoPipelineCreateInfo.renderPass = VK_NULL_HANDLE;
    ssaoPipelineCreateInfo.subpass = 0;

    const VkResult ssaoPipelineResult = vkCreateGraphicsPipelines(
        m_device,
        VK_NULL_HANDLE,
        1,
        &ssaoPipelineCreateInfo,
        nullptr,
        &ssaoPipeline
    );
    if (ssaoPipelineResult != VK_SUCCESS) {
        logVkFailure("vkCreateGraphicsPipelines(ssao)", ssaoPipelineResult);
        destroyNewPipelines();
        destroyShaderModules();
        return false;
    }
    VOX_LOGI("render") << "pipeline config (ssao): sampleCount=" << ssaoSpecializationData.sampleCount
              << ", power=" << ssaoSpecializationData.power
              << ", format=" << static_cast<int>(m_ssaoFormat)
              << "\n";

    ssaoStageInfos[1].module = ssaoBlurFragShaderModule;
    ssaoStageInfos[1].pSpecializationInfo = &ssaoBlurSpecializationInfo;
    const VkResult ssaoBlurPipelineResult = vkCreateGraphicsPipelines(
        m_device,
        VK_NULL_HANDLE,
        1,
        &ssaoPipelineCreateInfo,
        nullptr,
        &ssaoBlurPipeline
    );
    if (ssaoBlurPipelineResult != VK_SUCCESS) {
        logVkFailure("vkCreateGraphicsPipelines(ssaoBlur)", ssaoBlurPipelineResult);
        destroyNewPipelines();
        destroyShaderModules();
        return false;
    }
    VOX_LOGI("render") << "pipeline config (ssaoBlur): radius=" << ssaoSpecializationData.blurRadius
              << ", sigma=" << ssaoSpecializationData.blurSigma
              << ", format=" << static_cast<int>(m_ssaoFormat)
              << "\n";

    destroyShaderModules();

    if (m_voxelNormalDepthPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_device, m_voxelNormalDepthPipeline, nullptr);
    }
    if (m_pipeNormalDepthPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_device, m_pipeNormalDepthPipeline, nullptr);
    }
    if (m_ssaoPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_device, m_ssaoPipeline, nullptr);
    }
    if (m_ssaoBlurPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_device, m_ssaoBlurPipeline, nullptr);
    }

    m_voxelNormalDepthPipeline = voxelNormalDepthPipeline;
    m_pipeNormalDepthPipeline = pipeNormalDepthPipeline;
    m_ssaoPipeline = ssaoPipeline;
    m_ssaoBlurPipeline = ssaoBlurPipeline;
    setObjectName(
        VK_OBJECT_TYPE_PIPELINE,
        vkHandleToUint64(m_voxelNormalDepthPipeline),
        "pipeline.prepass.voxelNormalDepth"
    );
    setObjectName(
        VK_OBJECT_TYPE_PIPELINE,
        vkHandleToUint64(m_pipeNormalDepthPipeline),
        "pipeline.prepass.pipeNormalDepth"
    );
    setObjectName(VK_OBJECT_TYPE_PIPELINE, vkHandleToUint64(m_ssaoPipeline), "pipeline.ssao");
    setObjectName(VK_OBJECT_TYPE_PIPELINE, vkHandleToUint64(m_ssaoBlurPipeline), "pipeline.ssaoBlur");
    return true;
}

bool Renderer::createChunkBuffers(const world::ChunkGrid& chunkGrid, std::span<const std::size_t> remeshChunkIndices) {
    if (chunkGrid.chunks().empty()) {
        return false;
    }

    const std::vector<world::Chunk>& chunks = chunkGrid.chunks();
    const std::size_t expectedDrawRangeCount = chunks.size() * world::kChunkMeshLodCount;
    if (m_chunkDrawRanges.size() != expectedDrawRangeCount) {
        m_chunkDrawRanges.assign(expectedDrawRangeCount, ChunkDrawRange{});
    }
    if (m_chunkLodMeshCache.size() != chunks.size()) {
        m_chunkLodMeshCache.assign(chunks.size(), world::ChunkLodMeshes{});
        m_chunkLodMeshCacheValid = false;
    }
    if (m_chunkGrassInstanceCache.size() != chunks.size()) {
        m_chunkGrassInstanceCache.assign(chunks.size(), std::vector<GrassBillboardInstance>{});
    }

    auto rebuildGrassInstancesForChunk = [&](std::size_t chunkArrayIndex) {
        if (chunkArrayIndex >= chunks.size()) {
            return;
        }
        const world::Chunk& chunk = chunks[chunkArrayIndex];
        std::vector<GrassBillboardInstance>& grassInstances = m_chunkGrassInstanceCache[chunkArrayIndex];
        grassInstances.clear();
        grassInstances.reserve(192);

        const float chunkWorldX = static_cast<float>(chunk.chunkX() * world::Chunk::kSizeX);
        const float chunkWorldY = static_cast<float>(chunk.chunkY() * world::Chunk::kSizeY);
        const float chunkWorldZ = static_cast<float>(chunk.chunkZ() * world::Chunk::kSizeZ);

        for (int y = 0; y < world::Chunk::kSizeY - 1; ++y) {
            for (int z = 0; z < world::Chunk::kSizeZ; ++z) {
                for (int x = 0; x < world::Chunk::kSizeX; ++x) {
                    if (chunk.voxelAt(x, y, z).type != world::VoxelType::Grass) {
                        continue;
                    }
                    if (chunk.voxelAt(x, y + 1, z).type != world::VoxelType::Empty) {
                        continue;
                    }

                    const std::uint32_t hash =
                        static_cast<std::uint32_t>(x * 73856093) ^
                        static_cast<std::uint32_t>(y * 19349663) ^
                        static_cast<std::uint32_t>(z * 83492791) ^
                        static_cast<std::uint32_t>((chunk.chunkX() + 101) * 2654435761u) ^
                        static_cast<std::uint32_t>((chunk.chunkZ() + 193) * 2246822519u);
                    // Keep grass sparse and deterministic so placement feels natural and stable.
                    if ((hash % 100u) >= 36u) {
                        continue;
                    }
                    const float rand0 = static_cast<float>(hash & 0xFFu) / 255.0f;
                    const float rand1 = static_cast<float>((hash >> 8u) & 0xFFu) / 255.0f;
                    const float rand2 = static_cast<float>((hash >> 16u) & 0xFFu) / 255.0f;
                    const float jitterX = (rand0 - 0.5f) * 0.20f;
                    const float jitterZ = (rand1 - 0.5f) * 0.20f;
                    const float yawRadians = rand2 * (2.0f * 3.14159265f);

                    GrassBillboardInstance instance{};
                    instance.worldPosYaw[0] = chunkWorldX + static_cast<float>(x) + 0.5f + jitterX;
                    // Lift slightly above the supporting voxel top to avoid depth tie flicker.
                    instance.worldPosYaw[1] = chunkWorldY + static_cast<float>(y) + 1.02f;
                    instance.worldPosYaw[2] = chunkWorldZ + static_cast<float>(z) + 0.5f + jitterZ;
                    instance.worldPosYaw[3] = yawRadians;
                    grassInstances.push_back(instance);
                }
            }
        }
    };

    std::size_t remeshedChunkCount = 0;
    std::size_t remeshedActiveVertexCount = 0;
    std::size_t remeshedActiveIndexCount = 0;
    std::size_t remeshedNaiveVertexCount = 0;
    std::size_t remeshedNaiveIndexCount = 0;
    const auto countMeshGeometry = [](const world::ChunkLodMeshes& lodMeshes, std::size_t& outVertices, std::size_t& outIndices) {
        for (const world::ChunkMeshData& lodMesh : lodMeshes.lodMeshes) {
            outVertices += lodMesh.vertices.size();
            outIndices += lodMesh.indices.size();
        }
    };
    const bool fullRemesh = !m_chunkLodMeshCacheValid || remeshChunkIndices.empty();
    const auto remeshStart = std::chrono::steady_clock::now();
    if (fullRemesh) {
        for (std::size_t chunkArrayIndex = 0; chunkArrayIndex < chunks.size(); ++chunkArrayIndex) {
            m_chunkLodMeshCache[chunkArrayIndex] =
                world::buildChunkLodMeshes(chunks[chunkArrayIndex], m_chunkMeshingOptions);
            rebuildGrassInstancesForChunk(chunkArrayIndex);
            countMeshGeometry(
                m_chunkLodMeshCache[chunkArrayIndex],
                remeshedActiveVertexCount,
                remeshedActiveIndexCount
            );
            if (m_chunkMeshingOptions.mode == world::MeshingMode::Naive) {
                remeshedNaiveVertexCount = remeshedActiveVertexCount;
                remeshedNaiveIndexCount = remeshedActiveIndexCount;
            } else {
                const world::ChunkLodMeshes naiveLodMeshes =
                    world::buildChunkLodMeshes(chunks[chunkArrayIndex], world::MeshingOptions{world::MeshingMode::Naive});
                countMeshGeometry(naiveLodMeshes, remeshedNaiveVertexCount, remeshedNaiveIndexCount);
            }
        }
        remeshedChunkCount = chunks.size();
        m_chunkLodMeshCacheValid = true;
    } else {
        std::vector<std::uint8_t> remeshMask(chunks.size(), 0u);
        std::vector<std::size_t> uniqueRemeshChunkIndices;
        uniqueRemeshChunkIndices.reserve(remeshChunkIndices.size());
        for (const std::size_t chunkArrayIndex : remeshChunkIndices) {
            if (chunkArrayIndex >= chunks.size()) {
                return false;
            }
            if (remeshMask[chunkArrayIndex] != 0u) {
                continue;
            }
            remeshMask[chunkArrayIndex] = 1u;
            uniqueRemeshChunkIndices.push_back(chunkArrayIndex);
        }

        for (const std::size_t chunkArrayIndex : uniqueRemeshChunkIndices) {
            m_chunkLodMeshCache[chunkArrayIndex] =
                world::buildChunkLodMeshes(chunks[chunkArrayIndex], m_chunkMeshingOptions);
            rebuildGrassInstancesForChunk(chunkArrayIndex);
            countMeshGeometry(
                m_chunkLodMeshCache[chunkArrayIndex],
                remeshedActiveVertexCount,
                remeshedActiveIndexCount
            );
            if (m_chunkMeshingOptions.mode == world::MeshingMode::Naive) {
                remeshedNaiveVertexCount = remeshedActiveVertexCount;
                remeshedNaiveIndexCount = remeshedActiveIndexCount;
            } else {
                const world::ChunkLodMeshes naiveLodMeshes =
                    world::buildChunkLodMeshes(chunks[chunkArrayIndex], world::MeshingOptions{world::MeshingMode::Naive});
                countMeshGeometry(naiveLodMeshes, remeshedNaiveVertexCount, remeshedNaiveIndexCount);
            }
        }
        remeshedChunkCount = uniqueRemeshChunkIndices.size();
    }
    const auto remeshEnd = std::chrono::steady_clock::now();
    const std::chrono::duration<float, std::milli> remeshMs = remeshEnd - remeshStart;
    m_debugChunkLastRemeshedChunkCount = static_cast<std::uint32_t>(remeshedChunkCount);
    m_debugChunkLastRemeshActiveVertexCount = static_cast<std::uint32_t>(remeshedActiveVertexCount);
    m_debugChunkLastRemeshActiveIndexCount = static_cast<std::uint32_t>(remeshedActiveIndexCount);
    m_debugChunkLastRemeshNaiveVertexCount = static_cast<std::uint32_t>(remeshedNaiveVertexCount);
    m_debugChunkLastRemeshNaiveIndexCount = static_cast<std::uint32_t>(remeshedNaiveIndexCount);
    m_debugChunkLastRemeshMs = remeshMs.count();
    if (remeshedNaiveIndexCount > 0) {
        const float ratio = static_cast<float>(remeshedActiveIndexCount) / static_cast<float>(remeshedNaiveIndexCount);
        m_debugChunkLastRemeshReductionPercent = std::clamp(100.0f * (1.0f - ratio), 0.0f, 100.0f);
    } else {
        m_debugChunkLastRemeshReductionPercent = 0.0f;
    }
    if (fullRemesh) {
        m_debugChunkLastFullRemeshMs = remeshMs.count();
    }

    std::vector<GrassBillboardInstance> combinedGrassInstances;
    {
        std::size_t totalGrassInstanceCount = 0;
        for (const std::vector<GrassBillboardInstance>& chunkGrass : m_chunkGrassInstanceCache) {
            totalGrassInstanceCount += chunkGrass.size();
        }
        combinedGrassInstances.reserve(totalGrassInstanceCount);
        for (const std::vector<GrassBillboardInstance>& chunkGrass : m_chunkGrassInstanceCache) {
            combinedGrassInstances.insert(combinedGrassInstances.end(), chunkGrass.begin(), chunkGrass.end());
        }
    }

    if (combinedGrassInstances.empty()) {
        if (m_grassBillboardInstanceBufferHandle != kInvalidBufferHandle) {
            m_bufferAllocator.destroyBuffer(m_grassBillboardInstanceBufferHandle);
            m_grassBillboardInstanceBufferHandle = kInvalidBufferHandle;
        }
        m_grassBillboardInstanceCount = 0;
    } else {
        BufferCreateDesc grassInstanceCreateDesc{};
        grassInstanceCreateDesc.size = static_cast<VkDeviceSize>(combinedGrassInstances.size() * sizeof(GrassBillboardInstance));
        grassInstanceCreateDesc.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
        grassInstanceCreateDesc.memoryProperties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        grassInstanceCreateDesc.initialData = combinedGrassInstances.data();

        const BufferHandle newGrassInstanceBufferHandle = m_bufferAllocator.createBuffer(grassInstanceCreateDesc);
        if (newGrassInstanceBufferHandle != kInvalidBufferHandle) {
            const VkBuffer grassInstanceBuffer = m_bufferAllocator.getBuffer(newGrassInstanceBufferHandle);
            if (grassInstanceBuffer != VK_NULL_HANDLE) {
                setObjectName(
                    VK_OBJECT_TYPE_BUFFER,
                    vkHandleToUint64(grassInstanceBuffer),
                    "mesh.grassBillboard.instances"
                );
            }
            if (m_grassBillboardInstanceBufferHandle != kInvalidBufferHandle) {
                m_bufferAllocator.destroyBuffer(m_grassBillboardInstanceBufferHandle);
            }
            m_grassBillboardInstanceBufferHandle = newGrassInstanceBufferHandle;
            m_grassBillboardInstanceCount = static_cast<uint32_t>(combinedGrassInstances.size());
        } else {
            VOX_LOGE("render") << "grass billboard instance buffer allocation failed";
        }
    }

    std::vector<world::PackedVoxelVertex> combinedVertices;
    std::vector<std::uint32_t> combinedIndices;
    std::size_t uploadedVertexCount = 0;
    std::size_t uploadedIndexCount = 0;

    for (std::size_t chunkArrayIndex = 0; chunkArrayIndex < chunks.size(); ++chunkArrayIndex) {
        const world::Chunk& chunk = chunks[chunkArrayIndex];
        const world::ChunkLodMeshes& chunkLodMeshes = m_chunkLodMeshCache[chunkArrayIndex];

        for (std::size_t lodIndex = 0; lodIndex < world::kChunkMeshLodCount; ++lodIndex) {
            const world::ChunkMeshData& chunkMesh = chunkLodMeshes.lodMeshes[lodIndex];
            const std::size_t drawRangeArrayIndex = (chunkArrayIndex * world::kChunkMeshLodCount) + lodIndex;
            ChunkDrawRange& drawRange = m_chunkDrawRanges[drawRangeArrayIndex];

            drawRange.offsetX = static_cast<float>(chunk.chunkX() * world::Chunk::kSizeX);
            drawRange.offsetY = static_cast<float>(chunk.chunkY() * world::Chunk::kSizeY);
            drawRange.offsetZ = static_cast<float>(chunk.chunkZ() * world::Chunk::kSizeZ);
            drawRange.firstIndex = 0;
            drawRange.vertexOffset = 0;
            drawRange.indexCount = 0;

            if (chunkMesh.vertices.empty() || chunkMesh.indices.empty()) {
                continue;
            }

            const std::size_t baseVertexSize = combinedVertices.size();
            if (baseVertexSize > static_cast<std::size_t>(std::numeric_limits<int32_t>::max())) {
                VOX_LOGE("render") << "chunk mesh vertex offset exceeds int32 range";
                return false;
            }
            const uint32_t baseVertex = static_cast<uint32_t>(baseVertexSize);
            const uint32_t firstIndex = static_cast<uint32_t>(combinedIndices.size());

            combinedVertices.insert(combinedVertices.end(), chunkMesh.vertices.begin(), chunkMesh.vertices.end());
            combinedIndices.reserve(combinedIndices.size() + chunkMesh.indices.size());
            for (const std::uint32_t index : chunkMesh.indices) {
                combinedIndices.push_back(index + baseVertex);
            }

            drawRange.firstIndex = firstIndex;
            // Indices are already rebased into global vertex space.
            drawRange.vertexOffset = 0;
            drawRange.indexCount = static_cast<uint32_t>(chunkMesh.indices.size());
            uploadedVertexCount += chunkMesh.vertices.size();
            uploadedIndexCount += chunkMesh.indices.size();
        }
    }
    m_debugChunkMeshVertexCount = static_cast<std::uint32_t>(uploadedVertexCount);
    m_debugChunkMeshIndexCount = static_cast<std::uint32_t>(uploadedIndexCount);

    std::array<uint32_t, 2> meshQueueFamilies = {
        m_graphicsQueueFamilyIndex,
        m_transferQueueFamilyIndex
    };
    if (meshQueueFamilies[0] == meshQueueFamilies[1]) {
        meshQueueFamilies[1] = UINT32_MAX;
    }

    BufferHandle newChunkVertexBufferHandle = kInvalidBufferHandle;
    BufferHandle newChunkIndexBufferHandle = kInvalidBufferHandle;
    std::optional<FrameArenaSlice> chunkVertexUploadSliceOpt = std::nullopt;
    std::optional<FrameArenaSlice> chunkIndexUploadSliceOpt = std::nullopt;
    auto cleanupPendingAllocations = [&]() {
        if (newChunkVertexBufferHandle != kInvalidBufferHandle) {
            m_bufferAllocator.destroyBuffer(newChunkVertexBufferHandle);
            newChunkVertexBufferHandle = kInvalidBufferHandle;
        }
        if (newChunkIndexBufferHandle != kInvalidBufferHandle) {
            m_bufferAllocator.destroyBuffer(newChunkIndexBufferHandle);
            newChunkIndexBufferHandle = kInvalidBufferHandle;
        }
    };

    collectCompletedBufferReleases();

    if (m_transferCommandBufferInFlightValue > 0 && !waitForTimelineValue(m_transferCommandBufferInFlightValue)) {
        VOX_LOGE("render") << "failed waiting for prior transfer upload\n";
        cleanupPendingAllocations();
        return false;
    }
    m_transferCommandBufferInFlightValue = 0;
    collectCompletedBufferReleases();
    const uint64_t previousChunkReadyTimelineValue = m_currentChunkReadyTimelineValue;
    const bool hasChunkCopies = !combinedVertices.empty() && !combinedIndices.empty();

    if (hasChunkCopies) {
        const VkDeviceSize vertexBufferSize =
            static_cast<VkDeviceSize>(combinedVertices.size() * sizeof(world::PackedVoxelVertex));
        const VkDeviceSize indexBufferSize =
            static_cast<VkDeviceSize>(combinedIndices.size() * sizeof(std::uint32_t));

        BufferCreateDesc vertexCreateDesc{};
        vertexCreateDesc.size = vertexBufferSize;
        vertexCreateDesc.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
        vertexCreateDesc.memoryProperties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        if (meshQueueFamilies[1] != UINT32_MAX) {
            vertexCreateDesc.queueFamilyIndices = meshQueueFamilies.data();
            vertexCreateDesc.queueFamilyIndexCount = 2;
        }
        newChunkVertexBufferHandle = m_bufferAllocator.createBuffer(vertexCreateDesc);
        if (newChunkVertexBufferHandle == kInvalidBufferHandle) {
            VOX_LOGE("render") << "chunk global vertex buffer allocation failed";
            cleanupPendingAllocations();
            return false;
        }
        {
            const VkBuffer vertexBuffer = m_bufferAllocator.getBuffer(newChunkVertexBufferHandle);
            if (vertexBuffer != VK_NULL_HANDLE) {
                setObjectName(VK_OBJECT_TYPE_BUFFER, vkHandleToUint64(vertexBuffer), "chunk.global.vertex");
            }
        }

        BufferCreateDesc indexCreateDesc{};
        indexCreateDesc.size = indexBufferSize;
        indexCreateDesc.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
        indexCreateDesc.memoryProperties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        if (meshQueueFamilies[1] != UINT32_MAX) {
            indexCreateDesc.queueFamilyIndices = meshQueueFamilies.data();
            indexCreateDesc.queueFamilyIndexCount = 2;
        }
        newChunkIndexBufferHandle = m_bufferAllocator.createBuffer(indexCreateDesc);
        if (newChunkIndexBufferHandle == kInvalidBufferHandle) {
            VOX_LOGE("render") << "chunk global index buffer allocation failed";
            cleanupPendingAllocations();
            return false;
        }
        {
            const VkBuffer indexBuffer = m_bufferAllocator.getBuffer(newChunkIndexBufferHandle);
            if (indexBuffer != VK_NULL_HANDLE) {
                setObjectName(VK_OBJECT_TYPE_BUFFER, vkHandleToUint64(indexBuffer), "chunk.global.index");
            }
        }

        chunkVertexUploadSliceOpt = m_frameArena.allocateUpload(
            vertexBufferSize,
            static_cast<VkDeviceSize>(alignof(world::PackedVoxelVertex)),
            FrameArenaUploadKind::Unknown
        );
        if (!chunkVertexUploadSliceOpt.has_value() || chunkVertexUploadSliceOpt->mapped == nullptr) {
            VOX_LOGE("render") << "chunk global vertex upload slice allocation failed";
            cleanupPendingAllocations();
            return false;
        }
        std::memcpy(
            chunkVertexUploadSliceOpt->mapped,
            combinedVertices.data(),
            static_cast<size_t>(vertexBufferSize)
        );

        chunkIndexUploadSliceOpt = m_frameArena.allocateUpload(
            indexBufferSize,
            static_cast<VkDeviceSize>(alignof(std::uint32_t)),
            FrameArenaUploadKind::Unknown
        );
        if (!chunkIndexUploadSliceOpt.has_value() || chunkIndexUploadSliceOpt->mapped == nullptr) {
            VOX_LOGE("render") << "chunk global index upload slice allocation failed";
            cleanupPendingAllocations();
            return false;
        }
        std::memcpy(
            chunkIndexUploadSliceOpt->mapped,
            combinedIndices.data(),
            static_cast<size_t>(indexBufferSize)
        );
    }

    uint64_t transferSignalValue = 0;
    if (hasChunkCopies) {
        const VkResult resetResult = vkResetCommandPool(m_device, m_transferCommandPool, 0);
        if (resetResult != VK_SUCCESS) {
            logVkFailure("vkResetCommandPool(transfer)", resetResult);
            cleanupPendingAllocations();
            return false;
        }

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (vkBeginCommandBuffer(m_transferCommandBuffer, &beginInfo) != VK_SUCCESS) {
            VOX_LOGE("render") << "vkBeginCommandBuffer (transfer) failed\n";
            cleanupPendingAllocations();
            return false;
        }

        {
            const VkDeviceSize vertexBufferSize = m_bufferAllocator.getSize(newChunkVertexBufferHandle);
            const VkDeviceSize indexBufferSize = m_bufferAllocator.getSize(newChunkIndexBufferHandle);

            VkBufferCopy vertexCopy{};
            vertexCopy.srcOffset = chunkVertexUploadSliceOpt->offset;
            vertexCopy.size = vertexBufferSize;
            vkCmdCopyBuffer(
                m_transferCommandBuffer,
                m_bufferAllocator.getBuffer(chunkVertexUploadSliceOpt->buffer),
                m_bufferAllocator.getBuffer(newChunkVertexBufferHandle),
                1,
                &vertexCopy
            );

            VkBufferCopy indexCopy{};
            indexCopy.srcOffset = chunkIndexUploadSliceOpt->offset;
            indexCopy.size = indexBufferSize;
            vkCmdCopyBuffer(
                m_transferCommandBuffer,
                m_bufferAllocator.getBuffer(chunkIndexUploadSliceOpt->buffer),
                m_bufferAllocator.getBuffer(newChunkIndexBufferHandle),
                1,
                &indexCopy
            );
        }

        if (vkEndCommandBuffer(m_transferCommandBuffer) != VK_SUCCESS) {
            VOX_LOGE("render") << "vkEndCommandBuffer (transfer) failed\n";
            cleanupPendingAllocations();
            return false;
        }

        transferSignalValue = m_nextTimelineValue++;
        VkSemaphore timelineSemaphore = m_renderTimelineSemaphore;
        VkTimelineSemaphoreSubmitInfo timelineSubmitInfo{};
        timelineSubmitInfo.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
        timelineSubmitInfo.signalSemaphoreValueCount = 1;
        timelineSubmitInfo.pSignalSemaphoreValues = &transferSignalValue;

        VkSubmitInfo transferSubmitInfo{};
        transferSubmitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        transferSubmitInfo.pNext = &timelineSubmitInfo;
        transferSubmitInfo.commandBufferCount = 1;
        transferSubmitInfo.pCommandBuffers = &m_transferCommandBuffer;
        transferSubmitInfo.signalSemaphoreCount = 1;
        transferSubmitInfo.pSignalSemaphores = &timelineSemaphore;

        const VkResult submitResult = vkQueueSubmit(m_transferQueue, 1, &transferSubmitInfo, VK_NULL_HANDLE);
        if (submitResult != VK_SUCCESS) {
            logVkFailure("vkQueueSubmit(transfer)", submitResult);
            cleanupPendingAllocations();
            return false;
        }

        m_currentChunkReadyTimelineValue = transferSignalValue;
        m_pendingTransferTimelineValue = transferSignalValue;
        m_transferCommandBufferInFlightValue = transferSignalValue;
    }

    const uint64_t oldChunkReleaseValue = std::max(m_lastGraphicsTimelineValue, previousChunkReadyTimelineValue);
    scheduleBufferRelease(m_chunkVertexBufferHandle, oldChunkReleaseValue);
    scheduleBufferRelease(m_chunkIndexBufferHandle, oldChunkReleaseValue);
    m_chunkVertexBufferHandle = newChunkVertexBufferHandle;
    m_chunkIndexBufferHandle = newChunkIndexBufferHandle;
    newChunkVertexBufferHandle = kInvalidBufferHandle;
    newChunkIndexBufferHandle = kInvalidBufferHandle;

    VOX_LOGD("render") << "chunk upload queued (ranges=" << m_chunkDrawRanges.size()
                       << ", remeshedChunks=" << remeshedChunkCount
                       << ", meshingMode="
                       << (m_chunkMeshingOptions.mode == world::MeshingMode::Greedy ? "greedy" : "naive")
                       << ", vertices=" << uploadedVertexCount
                       << ", indices=" << uploadedIndexCount
                       << (hasChunkCopies
                               ? (", timelineValue=" + std::to_string(transferSignalValue))
                               : ", immediate=true")
                       << ")";
    return true;
}

bool Renderer::createFrameResources() {
    for (size_t frameIndex = 0; frameIndex < m_frames.size(); ++frameIndex) {
        FrameResources& frame = m_frames[frameIndex];
        VkCommandPoolCreateInfo poolCreateInfo{};
        poolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolCreateInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        poolCreateInfo.queueFamilyIndex = m_graphicsQueueFamilyIndex;

        if (vkCreateCommandPool(m_device, &poolCreateInfo, nullptr, &frame.commandPool) != VK_SUCCESS) {
            VOX_LOGE("render") << "failed creating command pool for frame resource\n";
            return false;
        }
        {
            const std::string poolName = "frame." + std::to_string(frameIndex) + ".graphics.commandPool";
            setObjectName(VK_OBJECT_TYPE_COMMAND_POOL, vkHandleToUint64(frame.commandPool), poolName.c_str());
        }

        VkSemaphoreCreateInfo semaphoreCreateInfo{};
        semaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        if (vkCreateSemaphore(m_device, &semaphoreCreateInfo, nullptr, &frame.imageAvailable) != VK_SUCCESS) {
            VOX_LOGE("render") << "failed creating imageAvailable semaphore\n";
            return false;
        }
        {
            const std::string semaphoreName = "frame." + std::to_string(frameIndex) + ".imageAvailable";
            setObjectName(VK_OBJECT_TYPE_SEMAPHORE, vkHandleToUint64(frame.imageAvailable), semaphoreName.c_str());
        }
    }

    VOX_LOGI("render") << "frame resources ready (" << kMaxFramesInFlight << " frames in flight)\n";
    return true;
}

bool Renderer::createGpuTimestampResources() {
    if (!m_gpuTimestampsSupported) {
        return true;
    }
    for (size_t frameIndex = 0; frameIndex < m_gpuTimestampQueryPools.size(); ++frameIndex) {
        if (m_gpuTimestampQueryPools[frameIndex] != VK_NULL_HANDLE) {
            continue;
        }
        VkQueryPoolCreateInfo queryPoolCreateInfo{};
        queryPoolCreateInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        queryPoolCreateInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
        queryPoolCreateInfo.queryCount = kGpuTimestampQueryCount;
        const VkResult result = vkCreateQueryPool(
            m_device,
            &queryPoolCreateInfo,
            nullptr,
            &m_gpuTimestampQueryPools[frameIndex]
        );
        if (result != VK_SUCCESS) {
            logVkFailure("vkCreateQueryPool(gpuTimestamps)", result);
            return false;
        }
        const std::string queryPoolName = "frame." + std::to_string(frameIndex) + ".gpuTimestampQueryPool";
        setObjectName(
            VK_OBJECT_TYPE_QUERY_POOL,
            vkHandleToUint64(m_gpuTimestampQueryPools[frameIndex]),
            queryPoolName.c_str()
        );
    }
    VOX_LOGI("render") << "GPU timestamp query pools ready (" << m_gpuTimestampQueryPools.size()
        << " pools, " << kGpuTimestampQueryCount << " queries each)\n";
    return true;
}

#if defined(VOXEL_HAS_IMGUI)
bool Renderer::createImGuiResources() {
    if (m_imguiInitialized) {
        return true;
    }

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    if (!ImGui_ImplGlfw_InitForVulkan(m_window, true)) {
        VOX_LOGE("imgui") << "ImGui_ImplGlfw_InitForVulkan failed\n";
        ImGui::DestroyContext();
        return false;
    }

    VkDescriptorPoolSize poolSizes[] = {
        {VK_DESCRIPTOR_TYPE_SAMPLER, 256},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 256},
        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 256},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 256},
        {VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 256},
        {VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 256},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 256},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 256},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 256},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 256},
        {VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 256},
    };

    VkDescriptorPoolCreateInfo poolCreateInfo{};
    poolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolCreateInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    poolCreateInfo.maxSets = 256;
    poolCreateInfo.poolSizeCount = static_cast<uint32_t>(sizeof(poolSizes) / sizeof(poolSizes[0]));
    poolCreateInfo.pPoolSizes = poolSizes;
    const VkResult poolResult = vkCreateDescriptorPool(m_device, &poolCreateInfo, nullptr, &m_imguiDescriptorPool);
    if (poolResult != VK_SUCCESS) {
        logVkFailure("vkCreateDescriptorPool(imgui)", poolResult);
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        return false;
    }
    setObjectName(
        VK_OBJECT_TYPE_DESCRIPTOR_POOL,
        vkHandleToUint64(m_imguiDescriptorPool),
        "imgui.descriptorPool"
    );

    ImGui_ImplVulkan_InitInfo initInfo{};
    initInfo.ApiVersion = VK_API_VERSION_1_3;
    initInfo.Instance = m_instance;
    initInfo.PhysicalDevice = m_physicalDevice;
    initInfo.Device = m_device;
    initInfo.QueueFamily = m_graphicsQueueFamilyIndex;
    initInfo.Queue = m_graphicsQueue;
    initInfo.DescriptorPool = m_imguiDescriptorPool;
    initInfo.MinImageCount = std::max<uint32_t>(2u, static_cast<uint32_t>(m_swapchainImages.size()));
    initInfo.ImageCount = static_cast<uint32_t>(m_swapchainImages.size());
    initInfo.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    initInfo.UseDynamicRendering = true;
    initInfo.PipelineRenderingCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    initInfo.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
    initInfo.PipelineRenderingCreateInfo.pColorAttachmentFormats = &m_swapchainFormat;
    initInfo.PipelineRenderingCreateInfo.depthAttachmentFormat = VK_FORMAT_UNDEFINED;
    initInfo.CheckVkResultFn = imguiCheckVkResult;
    if (!ImGui_ImplVulkan_Init(&initInfo)) {
        VOX_LOGE("imgui") << "ImGui_ImplVulkan_Init failed\n";
        vkDestroyDescriptorPool(m_device, m_imguiDescriptorPool, nullptr);
        m_imguiDescriptorPool = VK_NULL_HANDLE;
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        return false;
    }

    if (!ImGui_ImplVulkan_CreateFontsTexture()) {
        VOX_LOGE("imgui") << "ImGui_ImplVulkan_CreateFontsTexture failed\n";
        ImGui_ImplVulkan_Shutdown();
        vkDestroyDescriptorPool(m_device, m_imguiDescriptorPool, nullptr);
        m_imguiDescriptorPool = VK_NULL_HANDLE;
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        return false;
    }

    m_imguiInitialized = true;
    return true;
}

void Renderer::destroyImGuiResources() {
    if (!m_imguiInitialized) {
        return;
    }

    VOX_LOGI("imgui") << "destroy begin\n";
    ImGui_ImplVulkan_DestroyFontsTexture();
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    if (m_imguiDescriptorPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(m_device, m_imguiDescriptorPool, nullptr);
        m_imguiDescriptorPool = VK_NULL_HANDLE;
    }
    m_imguiInitialized = false;
    VOX_LOGI("imgui") << "destroy complete\n";
}

void Renderer::buildFrameStatsUi() {
    if (!m_showFrameStatsPanel) {
        return;
    }

    constexpr ImGuiWindowFlags kPanelFlags =
        ImGuiWindowFlags_AlwaysAutoResize |
        ImGuiWindowFlags_NoSavedSettings;
    if (!ImGui::Begin("Frame Stats", &m_showFrameStatsPanel, kPanelFlags)) {
        ImGui::End();
        return;
    }

    const float autoScale = std::numeric_limits<float>::max();
    if (m_debugCpuFrameTimingMsHistoryCount > 0) {
        const int cpuHistoryCount = static_cast<int>(m_debugCpuFrameTimingMsHistoryCount);
        const int cpuHistoryOffset =
            (m_debugCpuFrameTimingMsHistoryCount == kTimingHistorySampleCount)
                ? static_cast<int>(m_debugCpuFrameTimingMsHistoryWrite)
                : 0;
        ImGui::PlotLines(
            "CPU Frame (ms)",
            m_debugCpuFrameTimingMsHistory.data(),
            cpuHistoryCount,
            cpuHistoryOffset,
            nullptr,
            0.0f,
            autoScale,
            ImVec2(0.0f, 64.0f)
        );
    } else {
        ImGui::Text("CPU Frame (ms): collecting...");
    }

    if (m_gpuTimestampsSupported) {
        if (m_debugGpuFrameTimingMsHistoryCount > 0) {
            const int gpuHistoryCount = static_cast<int>(m_debugGpuFrameTimingMsHistoryCount);
            const int gpuHistoryOffset =
                (m_debugGpuFrameTimingMsHistoryCount == kTimingHistorySampleCount)
                    ? static_cast<int>(m_debugGpuFrameTimingMsHistoryWrite)
                    : 0;
            ImGui::PlotLines(
                "GPU Frame (ms)",
                m_debugGpuFrameTimingMsHistory.data(),
                gpuHistoryCount,
                gpuHistoryOffset,
                nullptr,
                0.0f,
                autoScale,
                ImVec2(0.0f, 64.0f)
            );
        } else {
            ImGui::Text("GPU Frame (ms): collecting...");
        }
    } else {
        ImGui::Text("GPU Frame (ms): unavailable");
    }

    ImGui::Text("FPS: %.1f", m_debugFps);
    ImGui::Text("Chunks (visible/total): %u / %u", m_debugSpatialVisibleChunkCount, m_debugChunkCount);
    if (m_gpuTimestampsSupported) {
        ImGui::Text("Frame (CPU/GPU): %.2f / %.2f ms", m_debugFrameTimeMs, m_debugGpuFrameTimeMs);
        ImGui::Text(
            "GPU Stages (ms): Shadow %.2f, Prepass %.2f, SSAO %.2f, Blur %.2f, Main %.2f, Post %.2f",
            m_debugGpuShadowTimeMs,
            m_debugGpuPrepassTimeMs,
            m_debugGpuSsaoTimeMs,
            m_debugGpuSsaoBlurTimeMs,
            m_debugGpuMainTimeMs,
            m_debugGpuPostTimeMs
        );
    } else {
        ImGui::Text("Frame (CPU/GPU): %.2f / n/a ms", m_debugFrameTimeMs);
    }
    ImGui::Text(
        "Draw Calls Total: %u (Shadow %u, Prepass %u, Main %u, Post %u)",
        m_debugDrawCallsTotal,
        m_debugDrawCallsShadow,
        m_debugDrawCallsPrepass,
        m_debugDrawCallsMain,
        m_debugDrawCallsPost
    );
    ImGui::Text("Chunk Indirect Commands: %u", m_debugChunkIndirectCommandCount);
    ImGui::Text(
        "Spatial Query N/C/V: %u / %u / %u",
        m_debugSpatialQueryStats.visitedNodeCount,
        m_debugSpatialQueryStats.candidateChunkCount,
        m_debugSpatialQueryStats.visibleChunkCount
    );
    if (m_debugSpatialQueryStats.clipmapActiveLevelCount > 0) {
        ImGui::Text(
            "Clipmap L/U/S/B: %u / %u / %u / %u",
            m_debugSpatialQueryStats.clipmapActiveLevelCount,
            m_debugSpatialQueryStats.clipmapUpdatedLevelCount,
            m_debugSpatialQueryStats.clipmapUpdatedSlabCount,
            m_debugSpatialQueryStats.clipmapUpdatedBrickCount
        );
    }
    ImGui::Text("Chunk Mesh Vert/Idx: %u / %u", m_debugChunkMeshVertexCount, m_debugChunkMeshIndexCount);
    ImGui::Text("Last Chunk Remesh: %.2f ms (%u)", m_debugChunkLastRemeshMs, m_debugChunkLastRemeshedChunkCount);
    ImGui::Text("Greedy Reduction vs Naive: %.1f%%", m_debugChunkLastRemeshReductionPercent);
    const bool hasFrameArenaMetrics =
        m_debugFrameArenaUploadBytes > 0 ||
        m_debugFrameArenaUploadAllocs > 0 ||
        m_debugFrameArenaTransientBufferBytes > 0 ||
        m_debugFrameArenaTransientBufferCount > 0 ||
        m_debugFrameArenaTransientImageBytes > 0 ||
        m_debugFrameArenaTransientImageCount > 0 ||
        m_debugFrameArenaAliasReuses > 0 ||
        m_debugFrameArenaResidentBufferBytes > 0 ||
        m_debugFrameArenaResidentBufferCount > 0 ||
        m_debugFrameArenaResidentImageBytes > 0 ||
        m_debugFrameArenaResidentImageCount > 0 ||
        m_debugFrameArenaResidentAliasReuses > 0 ||
        !m_debugAliasedImages.empty();
    if (hasFrameArenaMetrics) {
        ImGui::Separator();
        ImGui::Text("FrameArena");
        if (m_debugFrameArenaUploadBytes > 0 || m_debugFrameArenaUploadAllocs > 0) {
            ImGui::Text(
                "Upload this frame: %llu B (%u allocs)",
                static_cast<unsigned long long>(m_debugFrameArenaUploadBytes),
                m_debugFrameArenaUploadAllocs
            );
        }
        ImGui::Text("Image alias reuses (frame/live): %u / %u", m_debugFrameArenaAliasReuses, m_debugFrameArenaResidentAliasReuses);
        ImGui::Text("Resident images (live): %u", m_debugFrameArenaResidentImageCount);
    }
    ImGui::End();
}

void Renderer::buildMeshingDebugUi() {
    if (!m_debugUiVisible || !m_showMeshingPanel) {
        return;
    }

    if (!ImGui::Begin("Meshing", &m_showMeshingPanel)) {
        ImGui::End();
        return;
    }

    ImGui::Checkbox("Use Spatial Queries", &m_debugEnableSpatialQueries);
    int clipmapLevels = static_cast<int>(m_debugClipmapConfig.levelCount);
    int clipmapGridResolution = m_debugClipmapConfig.gridResolution;
    int clipmapBaseVoxelSize = m_debugClipmapConfig.baseVoxelSize;
    int clipmapBrickResolution = m_debugClipmapConfig.brickResolution;
    if (ImGui::SliderInt("Clipmap Levels", &clipmapLevels, 1, 8)) {
        m_debugClipmapConfig.levelCount = static_cast<std::uint32_t>(clipmapLevels);
    }
    if (ImGui::SliderInt("Clipmap Grid Res", &clipmapGridResolution, 32, 256)) {
        m_debugClipmapConfig.gridResolution = clipmapGridResolution;
    }
    if (ImGui::SliderInt("Clipmap Base Voxel", &clipmapBaseVoxelSize, 1, 8)) {
        m_debugClipmapConfig.baseVoxelSize = clipmapBaseVoxelSize;
    }
    if (ImGui::SliderInt("Clipmap Brick Res", &clipmapBrickResolution, 2, 32)) {
        m_debugClipmapConfig.brickResolution = clipmapBrickResolution;
    }

    int meshingModeSelection = (m_chunkMeshingOptions.mode == world::MeshingMode::Greedy) ? 1 : 0;
    if (ImGui::Combo("Chunk Meshing", &meshingModeSelection, "Naive\0Greedy\0")) {
        const world::MeshingMode nextMode =
            (meshingModeSelection == 1) ? world::MeshingMode::Greedy : world::MeshingMode::Naive;
        if (nextMode != m_chunkMeshingOptions.mode) {
            m_chunkMeshingOptions.mode = nextMode;
            m_chunkLodMeshCacheValid = false;
            m_chunkMeshRebuildRequested = true;
            m_pendingChunkRemeshIndices.clear();
            VOX_LOGI("render") << "chunk meshing mode changed to "
                               << (nextMode == world::MeshingMode::Greedy ? "Greedy" : "Naive")
                               << ", scheduling full remesh";
        }
    }

    ImGui::Text(
        "Query N/C/V: %u / %u / %u",
        m_debugSpatialQueryStats.visitedNodeCount,
        m_debugSpatialQueryStats.candidateChunkCount,
        m_debugSpatialQueryStats.visibleChunkCount
    );
    if (m_debugSpatialQueryStats.clipmapActiveLevelCount > 0) {
        ImGui::Text(
            "Clipmap L/U/S/B: %u / %u / %u / %u",
            m_debugSpatialQueryStats.clipmapActiveLevelCount,
            m_debugSpatialQueryStats.clipmapUpdatedLevelCount,
            m_debugSpatialQueryStats.clipmapUpdatedSlabCount,
            m_debugSpatialQueryStats.clipmapUpdatedBrickCount
        );
    }

    ImGui::Text("Chunk Mesh Vert/Idx: %u / %u", m_debugChunkMeshVertexCount, m_debugChunkMeshIndexCount);
    ImGui::Text("Last Chunk Remesh: %.2f ms (%u)", m_debugChunkLastRemeshMs, m_debugChunkLastRemeshedChunkCount);
    ImGui::Text("Greedy Reduction vs Naive: %.1f%%", m_debugChunkLastRemeshReductionPercent);
    ImGui::End();
}

void Renderer::buildShadowDebugUi() {
    if (!m_debugUiVisible || !m_showShadowPanel) {
        return;
    }

    if (!ImGui::Begin("Shadows", &m_showShadowPanel)) {
        ImGui::End();
        return;
    }

    ImGui::Text("Cascaded Shadow Maps");
    ImGui::Text(
        "Macro Cells U/R4/R1: %u / %u / %u",
        m_debugMacroCellUniformCount,
        m_debugMacroCellRefined4Count,
        m_debugMacroCellRefined1Count
    );
    ImGui::Text(
        "Drawn LOD ranges 0/1/2: %u / %u / %u",
        m_debugDrawnLod0Ranges,
        m_debugDrawnLod1Ranges,
        m_debugDrawnLod2Ranges
    );
    ImGui::Separator();
    ImGui::SliderFloat("PCF Radius", &m_shadowDebugSettings.pcfRadius, 1.0f, 3.0f, "%.2f");
    ImGui::SliderFloat("Cascade Blend Min", &m_shadowDebugSettings.cascadeBlendMin, 1.0f, 20.0f, "%.2f");
    ImGui::SliderFloat("Cascade Blend Factor", &m_shadowDebugSettings.cascadeBlendFactor, 0.05f, 0.60f, "%.2f");

    ImGui::Separator();
    ImGui::Text("Receiver Bias");
    ImGui::SliderFloat("Normal Offset Near", &m_shadowDebugSettings.receiverNormalOffsetNear, 0.0f, 0.20f, "%.3f");
    ImGui::SliderFloat("Normal Offset Far", &m_shadowDebugSettings.receiverNormalOffsetFar, 0.0f, 0.35f, "%.3f");
    ImGui::SliderFloat("Base Bias Near (texel)", &m_shadowDebugSettings.receiverBaseBiasNearTexel, 0.0f, 12.0f, "%.2f");
    ImGui::SliderFloat("Base Bias Far (texel)", &m_shadowDebugSettings.receiverBaseBiasFarTexel, 0.0f, 16.0f, "%.2f");
    ImGui::SliderFloat("Slope Bias Near (texel)", &m_shadowDebugSettings.receiverSlopeBiasNearTexel, 0.0f, 14.0f, "%.2f");
    ImGui::SliderFloat("Slope Bias Far (texel)", &m_shadowDebugSettings.receiverSlopeBiasFarTexel, 0.0f, 18.0f, "%.2f");
    ImGui::Separator();
    ImGui::Text("Caster Bias");
    ImGui::SliderFloat("Const Bias Base", &m_shadowDebugSettings.casterConstantBiasBase, 0.0f, 6.0f, "%.2f");
    ImGui::SliderFloat("Const Bias Cascade Scale", &m_shadowDebugSettings.casterConstantBiasCascadeScale, 0.0f, 3.0f, "%.2f");
    ImGui::SliderFloat("Slope Bias Base", &m_shadowDebugSettings.casterSlopeBiasBase, 0.0f, 8.0f, "%.2f");
    ImGui::SliderFloat("Slope Bias Cascade Scale", &m_shadowDebugSettings.casterSlopeBiasCascadeScale, 0.0f, 4.0f, "%.2f");

    ImGui::Separator();
    ImGui::Text("Ambient Occlusion");
    ImGui::Checkbox("Enable Vertex AO", &m_debugEnableVertexAo);
    ImGui::Checkbox("Enable SSAO", &m_debugEnableSsao);
    ImGui::Checkbox("Visualize SSAO", &m_debugVisualizeSsao);
    ImGui::Checkbox("Visualize AO Normals", &m_debugVisualizeAoNormals);
    ImGui::SliderFloat("SSAO Radius", &m_shadowDebugSettings.ssaoRadius, 0.10f, 2.00f, "%.2f");
    ImGui::SliderFloat("SSAO Bias", &m_shadowDebugSettings.ssaoBias, 0.0f, 0.20f, "%.3f");
    ImGui::SliderFloat("SSAO Intensity", &m_shadowDebugSettings.ssaoIntensity, 0.0f, 1.50f, "%.2f");

    ImGui::Separator();
    ImGui::Text("Cascade Splits: %.1f / %.1f / %.1f / %.1f",
        m_shadowCascadeSplits[0],
        m_shadowCascadeSplits[1],
        m_shadowCascadeSplits[2],
        m_shadowCascadeSplits[3]
    );
    if (ImGui::Button("Reset Shadow Defaults")) {
        m_shadowDebugSettings = ShadowDebugSettings{};
    }

    ImGui::End();
}

void Renderer::buildSunDebugUi() {
    if (!m_debugUiVisible || !m_showSunPanel) {
        return;
    }

    if (!ImGui::Begin("Sun/Sky", &m_showSunPanel)) {
        ImGui::End();
        return;
    }

    ImGui::SliderFloat("Sun Yaw", &m_skyDebugSettings.sunYawDegrees, -180.0f, 180.0f, "%.1f deg");
    ImGui::SliderFloat("Sun Pitch", &m_skyDebugSettings.sunPitchDegrees, -89.0f, 5.0f, "%.1f deg");
    ImGui::SliderFloat("Rayleigh Strength", &m_skyDebugSettings.rayleighStrength, 0.1f, 4.0f, "%.2f");
    ImGui::SliderFloat("Mie Strength", &m_skyDebugSettings.mieStrength, 0.05f, 4.0f, "%.2f");
    ImGui::SliderFloat("Mie Anisotropy", &m_skyDebugSettings.mieAnisotropy, 0.0f, 0.95f, "%.2f");
    ImGui::SliderFloat("Sky Exposure", &m_skyDebugSettings.skyExposure, 0.25f, 3.0f, "%.2f");
    if (ImGui::Button("Reset Sun/Sky Defaults")) {
        m_skyDebugSettings = SkyDebugSettings{};
    }
    ImGui::End();
}

void Renderer::buildAimReticleUi() {
    ImDrawList* drawList = ImGui::GetBackgroundDrawList();
    if (drawList == nullptr) {
        return;
    }

    const ImVec2 displaySize = ImGui::GetIO().DisplaySize;
    const ImVec2 center{displaySize.x * 0.5f, displaySize.y * 0.5f};
    constexpr float kOuter = 9.0f;
    constexpr float kInner = 3.0f;
    constexpr float kThickness = 1.6f;
    const ImU32 color = IM_COL32(235, 245, 255, 220);

    drawList->AddLine(ImVec2(center.x - kOuter, center.y), ImVec2(center.x - kInner, center.y), color, kThickness);
    drawList->AddLine(ImVec2(center.x + kInner, center.y), ImVec2(center.x + kOuter, center.y), color, kThickness);
    drawList->AddLine(ImVec2(center.x, center.y - kOuter), ImVec2(center.x, center.y - kInner), color, kThickness);
    drawList->AddLine(ImVec2(center.x, center.y + kInner), ImVec2(center.x, center.y + kOuter), color, kThickness);
}
#endif

bool Renderer::waitForTimelineValue(uint64_t value) const {
    if (value == 0 || m_renderTimelineSemaphore == VK_NULL_HANDLE) {
        return true;
    }

    VkSemaphore waitSemaphore = m_renderTimelineSemaphore;
    VkSemaphoreWaitInfo waitInfo{};
    waitInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
    waitInfo.semaphoreCount = 1;
    waitInfo.pSemaphores = &waitSemaphore;
    waitInfo.pValues = &value;
    const VkResult waitResult = vkWaitSemaphores(m_device, &waitInfo, std::numeric_limits<uint64_t>::max());
    if (waitResult != VK_SUCCESS) {
        logVkFailure("vkWaitSemaphores(timeline)", waitResult);
        return false;
    }
    return true;
}

void Renderer::readGpuTimestampResults(uint32_t frameIndex) {
    if (!m_gpuTimestampsSupported || m_device == VK_NULL_HANDLE || frameIndex >= m_gpuTimestampQueryPools.size()) {
        return;
    }
    const VkQueryPool queryPool = m_gpuTimestampQueryPools[frameIndex];
    if (queryPool == VK_NULL_HANDLE) {
        return;
    }

    std::array<std::uint64_t, kGpuTimestampQueryCount> timestamps{};
    const VkResult result = vkGetQueryPoolResults(
        m_device,
        queryPool,
        0,
        kGpuTimestampQueryCount,
        sizeof(timestamps),
        timestamps.data(),
        sizeof(std::uint64_t),
        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT
    );
    if (result != VK_SUCCESS) {
        logVkFailure("vkGetQueryPoolResults(gpuTimestamps)", result);
        return;
    }

    const auto durationMs = [&](uint32_t startIndex, uint32_t endIndex) -> float {
        if (startIndex >= kGpuTimestampQueryCount || endIndex >= kGpuTimestampQueryCount) {
            return 0.0f;
        }
        const std::uint64_t startTicks = timestamps[startIndex];
        const std::uint64_t endTicks = timestamps[endIndex];
        if (endTicks <= startTicks) {
            return 0.0f;
        }
        const double deltaNs = static_cast<double>(endTicks - startTicks) * static_cast<double>(m_gpuTimestampPeriodNs);
        return static_cast<float>(deltaNs * 1.0e-6);
    };

    m_debugGpuFrameTimeMs = durationMs(kGpuTimestampQueryFrameStart, kGpuTimestampQueryFrameEnd);
    m_debugGpuShadowTimeMs = durationMs(kGpuTimestampQueryShadowStart, kGpuTimestampQueryShadowEnd);
    m_debugGpuPrepassTimeMs = durationMs(kGpuTimestampQueryPrepassStart, kGpuTimestampQueryPrepassEnd);
    m_debugGpuSsaoTimeMs = durationMs(kGpuTimestampQuerySsaoStart, kGpuTimestampQuerySsaoEnd);
    m_debugGpuSsaoBlurTimeMs = durationMs(kGpuTimestampQuerySsaoBlurStart, kGpuTimestampQuerySsaoBlurEnd);
    m_debugGpuMainTimeMs = durationMs(kGpuTimestampQueryMainStart, kGpuTimestampQueryMainEnd);
    m_debugGpuPostTimeMs = durationMs(kGpuTimestampQueryPostStart, kGpuTimestampQueryPostEnd);
    m_debugGpuFrameTimingMsHistory[m_debugGpuFrameTimingMsHistoryWrite] = m_debugGpuFrameTimeMs;
    m_debugGpuFrameTimingMsHistoryWrite =
        (m_debugGpuFrameTimingMsHistoryWrite + 1u) % kTimingHistorySampleCount;
    m_debugGpuFrameTimingMsHistoryCount =
        std::min(m_debugGpuFrameTimingMsHistoryCount + 1u, kTimingHistorySampleCount);
}

void Renderer::scheduleBufferRelease(BufferHandle handle, uint64_t timelineValue) {
    if (handle == kInvalidBufferHandle) {
        return;
    }
    if (timelineValue == 0 || m_renderTimelineSemaphore == VK_NULL_HANDLE) {
        m_bufferAllocator.destroyBuffer(handle);
        return;
    }
    m_deferredBufferReleases.push_back({handle, timelineValue});
}

void Renderer::collectCompletedBufferReleases() {
    if (m_renderTimelineSemaphore == VK_NULL_HANDLE) {
        return;
    }

    uint64_t completedValue = 0;
    const VkResult counterResult = vkGetSemaphoreCounterValue(m_device, m_renderTimelineSemaphore, &completedValue);
    if (counterResult != VK_SUCCESS) {
        logVkFailure("vkGetSemaphoreCounterValue", counterResult);
        return;
    }

    for (const DeferredBufferRelease& release : m_deferredBufferReleases) {
        if (release.timelineValue <= completedValue) {
            m_bufferAllocator.destroyBuffer(release.handle);
        }
    }
    std::erase_if(
        m_deferredBufferReleases,
        [completedValue](const DeferredBufferRelease& release) {
            return release.timelineValue <= completedValue;
        }
    );

    if (m_pendingTransferTimelineValue > 0 && m_pendingTransferTimelineValue <= completedValue) {
        m_pendingTransferTimelineValue = 0;
    }
    if (m_transferCommandBufferInFlightValue > 0 && m_transferCommandBufferInFlightValue <= completedValue) {
        m_transferCommandBufferInFlightValue = 0;
    }
}

void Renderer::renderFrame(
    const world::ChunkGrid& chunkGrid,
    const sim::Simulation& simulation,
    const CameraPose& camera,
    const VoxelPreview& preview,
    std::span<const std::size_t> visibleChunkIndices
) {
    if (m_device == VK_NULL_HANDLE || m_swapchain == VK_NULL_HANDLE) {
        return;
    }
    if (m_window != nullptr && glfwWindowShouldClose(m_window) == GLFW_TRUE) {
        return;
    }

    const double frameNowSeconds = glfwGetTime();
    if (m_lastFrameTimestampSeconds > 0.0) {
        const double deltaSeconds = std::max(0.0, frameNowSeconds - m_lastFrameTimestampSeconds);
        m_debugFrameTimeMs = static_cast<float>(deltaSeconds * 1000.0);
        m_debugFps = (deltaSeconds > 0.0) ? static_cast<float>(1.0 / deltaSeconds) : 0.0f;
        m_debugCpuFrameTimingMsHistory[m_debugCpuFrameTimingMsHistoryWrite] = m_debugFrameTimeMs;
        m_debugCpuFrameTimingMsHistoryWrite =
            (m_debugCpuFrameTimingMsHistoryWrite + 1u) % kTimingHistorySampleCount;
        m_debugCpuFrameTimingMsHistoryCount =
            std::min(m_debugCpuFrameTimingMsHistoryCount + 1u, kTimingHistorySampleCount);
    }
    m_lastFrameTimestampSeconds = frameNowSeconds;

    m_debugChunkCount = static_cast<std::uint32_t>(chunkGrid.chunks().size());
    m_debugMacroCellUniformCount = 0;
    m_debugMacroCellRefined4Count = 0;
    m_debugMacroCellRefined1Count = 0;
    for (const world::Chunk& chunk : chunkGrid.chunks()) {
        for (int my = 0; my < world::Chunk::kMacroSizeY; ++my) {
            for (int mz = 0; mz < world::Chunk::kMacroSizeZ; ++mz) {
                for (int mx = 0; mx < world::Chunk::kMacroSizeX; ++mx) {
                    const world::Chunk::MacroCell cell = chunk.macroCellAt(mx, my, mz);
                    switch (cell.resolution) {
                    case world::Chunk::CellResolution::Uniform:
                        ++m_debugMacroCellUniformCount;
                        break;
                    case world::Chunk::CellResolution::Refined4:
                        ++m_debugMacroCellRefined4Count;
                        break;
                    case world::Chunk::CellResolution::Refined1:
                        ++m_debugMacroCellRefined1Count;
                        break;
                    }
                }
            }
        }
    }
    collectCompletedBufferReleases();

    FrameResources& frame = m_frames[m_currentFrame];
    if (!waitForTimelineValue(m_frameTimelineValues[m_currentFrame])) {
        return;
    }
    if (m_frameTimelineValues[m_currentFrame] > 0) {
        readGpuTimestampResults(m_currentFrame);
    }
    if (m_transferCommandBufferInFlightValue > 0) {
        if (!waitForTimelineValue(m_transferCommandBufferInFlightValue)) {
            return;
        }
        m_transferCommandBufferInFlightValue = 0;
        m_pendingTransferTimelineValue = 0;
        collectCompletedBufferReleases();
    }
    m_frameArena.beginFrame(m_currentFrame);

    if (m_chunkMeshRebuildRequested || !m_pendingChunkRemeshIndices.empty()) {
        const std::span<const std::size_t> pendingRemeshIndices =
            m_chunkMeshRebuildRequested
                ? std::span<const std::size_t>{}
                : std::span<const std::size_t>(m_pendingChunkRemeshIndices.data(), m_pendingChunkRemeshIndices.size());
        if (createChunkBuffers(chunkGrid, pendingRemeshIndices)) {
            m_chunkMeshRebuildRequested = false;
            m_pendingChunkRemeshIndices.clear();
        } else {
            VOX_LOGE("render") << "failed deferred chunk remesh";
        }
    }

    uint32_t imageIndex = 0;
    const VkResult acquireResult = vkAcquireNextImageKHR(
        m_device,
        m_swapchain,
        std::numeric_limits<uint64_t>::max(),
        frame.imageAvailable,
        VK_NULL_HANDLE,
        &imageIndex
    );

    if (acquireResult == VK_ERROR_OUT_OF_DATE_KHR) {
        VOX_LOGI("render") << "swapchain out of date during acquire, recreating\n";
        recreateSwapchain();
        return;
    }
    if (acquireResult != VK_SUCCESS && acquireResult != VK_SUBOPTIMAL_KHR) {
        logVkFailure("vkAcquireNextImageKHR", acquireResult);
        return;
    }

    if (!waitForTimelineValue(m_swapchainImageTimelineValues[imageIndex])) {
        return;
    }
    const VkSemaphore renderFinishedSemaphore = m_renderFinishedSemaphores[imageIndex];
    const uint32_t aoFrameIndex = m_currentFrame % kMaxFramesInFlight;

    vkResetCommandPool(m_device, frame.commandPool, 0);

    VkCommandBufferAllocateInfo allocateInfo{};
    allocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocateInfo.commandPool = frame.commandPool;
    allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocateInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(m_device, &allocateInfo, &commandBuffer) != VK_SUCCESS) {
        VOX_LOGE("render") << "vkAllocateCommandBuffers failed\n";
        return;
    }
    {
        const std::string commandBufferName = "frame." + std::to_string(m_currentFrame) + ".graphics.commandBuffer";
        setObjectName(VK_OBJECT_TYPE_COMMAND_BUFFER, vkHandleToUint64(commandBuffer), commandBufferName.c_str());
    }

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
        VOX_LOGE("render") << "vkBeginCommandBuffer failed\n";
        return;
    }
    const VkQueryPool gpuTimestampQueryPool =
        m_gpuTimestampsSupported ? m_gpuTimestampQueryPools[m_currentFrame] : VK_NULL_HANDLE;
    auto writeGpuTimestampTop = [&](uint32_t queryIndex) {
        if (gpuTimestampQueryPool == VK_NULL_HANDLE) {
            return;
        }
        vkCmdWriteTimestamp(
            commandBuffer,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            gpuTimestampQueryPool,
            queryIndex
        );
    };
    auto writeGpuTimestampBottom = [&](uint32_t queryIndex) {
        if (gpuTimestampQueryPool == VK_NULL_HANDLE) {
            return;
        }
        vkCmdWriteTimestamp(
            commandBuffer,
            VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            gpuTimestampQueryPool,
            queryIndex
        );
    };
    if (gpuTimestampQueryPool != VK_NULL_HANDLE) {
        vkCmdResetQueryPool(commandBuffer, gpuTimestampQueryPool, 0, kGpuTimestampQueryCount);
        writeGpuTimestampTop(kGpuTimestampQueryFrameStart);
    }
    beginDebugLabel(commandBuffer, "Frame", 0.22f, 0.22f, 0.26f, 1.0f);
#if defined(VOXEL_HAS_IMGUI)
    if (m_imguiInitialized) {
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        buildFrameStatsUi();
        buildMeshingDebugUi();
        buildShadowDebugUi();
        buildSunDebugUi();
        m_debugUiVisible = m_showMeshingPanel || m_showShadowPanel || m_showSunPanel;
        buildAimReticleUi();
        ImGui::Render();
    }
#endif
    // Keep previous frame counters visible in UI, then reset for this frame's capture.
    m_debugDrawnLod0Ranges = 0;
    m_debugDrawnLod1Ranges = 0;
    m_debugDrawnLod2Ranges = 0;
    m_debugChunkIndirectCommandCount = 0;
    m_debugDrawCallsTotal = 0;
    m_debugDrawCallsShadow = 0;
    m_debugDrawCallsPrepass = 0;
    m_debugDrawCallsMain = 0;
    m_debugDrawCallsPost = 0;

    const float aspectRatio = static_cast<float>(m_swapchainExtent.width) / static_cast<float>(m_swapchainExtent.height);
    const float nearPlane = 0.1f;
    const float farPlane = 500.0f;
    const float yawRadians = math::radians(camera.yawDegrees);
    const float pitchRadians = math::radians(camera.pitchDegrees);
    const float halfFovRadians = math::radians(camera.fovDegrees) * 0.5f;
    const float tanHalfFov = std::tan(halfFovRadians);
    const float cosPitch = std::cos(pitchRadians);
    const math::Vector3 eye{camera.x, camera.y, camera.z};
    const int cameraChunkX = static_cast<int>(std::floor(camera.x / static_cast<float>(world::Chunk::kSizeX)));
    const int cameraChunkY = static_cast<int>(std::floor(camera.y / static_cast<float>(world::Chunk::kSizeY)));
    const int cameraChunkZ = static_cast<int>(std::floor(camera.z / static_cast<float>(world::Chunk::kSizeZ)));
    const math::Vector3 forward{
        std::cos(yawRadians) * cosPitch,
        std::sin(pitchRadians),
        std::sin(yawRadians) * cosPitch
    };

    const math::Matrix4 view = lookAt(eye, eye + forward, math::Vector3{0.0f, 1.0f, 0.0f});
    const math::Matrix4 projection = perspectiveVulkan(math::radians(camera.fovDegrees), aspectRatio, nearPlane, farPlane);
    const math::Matrix4 mvp = projection * view;
    const math::Matrix4 mvpColumnMajor = transpose(mvp);
    const math::Matrix4 viewColumnMajor = transpose(view);
    const math::Matrix4 projectionColumnMajor = transpose(projection);

    const bool projectionParamsChanged =
        std::abs(m_shadowStableAspectRatio - aspectRatio) > 0.0001f ||
        std::abs(m_shadowStableFovDegrees - camera.fovDegrees) > 0.0001f;
    if (projectionParamsChanged) {
        m_shadowStableAspectRatio = aspectRatio;
        m_shadowStableFovDegrees = camera.fovDegrees;
        m_shadowStableCascadeRadii.fill(0.0f);
    }

    const float sunYawRadians = math::radians(m_skyDebugSettings.sunYawDegrees);
    const float sunPitchRadians = math::radians(m_skyDebugSettings.sunPitchDegrees);
    const float sunCosPitch = std::cos(sunPitchRadians);
    math::Vector3 sunDirection = math::normalize(math::Vector3{
        std::cos(sunYawRadians) * sunCosPitch,
        std::sin(sunPitchRadians),
        std::sin(sunYawRadians) * sunCosPitch
    });
    if (math::lengthSquared(sunDirection) <= 0.0001f) {
        sunDirection = math::Vector3{-0.58f, -0.42f, -0.24f};
    }
    const math::Vector3 sunColor = computeSunColor(m_skyDebugSettings, sunDirection);

    constexpr float kCascadeLambda = 0.70f;
    constexpr float kCascadeSplitQuantization = 0.5f;
    constexpr float kCascadeSplitUpdateThreshold = 0.5f;
    std::array<float, kShadowCascadeCount> cascadeDistances{};
    for (uint32_t cascadeIndex = 0; cascadeIndex < kShadowCascadeCount; ++cascadeIndex) {
        const float p = static_cast<float>(cascadeIndex + 1) / static_cast<float>(kShadowCascadeCount);
        const float logarithmicSplit = nearPlane * std::pow(farPlane / nearPlane, p);
        const float uniformSplit = nearPlane + ((farPlane - nearPlane) * p);
        const float desiredSplit =
            (kCascadeLambda * logarithmicSplit) + ((1.0f - kCascadeLambda) * uniformSplit);
        const float quantizedSplit =
            std::round(desiredSplit / kCascadeSplitQuantization) * kCascadeSplitQuantization;

        float split = m_shadowCascadeSplits[cascadeIndex];
        if (projectionParamsChanged || std::abs(quantizedSplit - split) > kCascadeSplitUpdateThreshold) {
            split = quantizedSplit;
        }

        const float previousSplit = (cascadeIndex == 0) ? nearPlane : m_shadowCascadeSplits[cascadeIndex - 1];
        split = std::max(split, previousSplit + kCascadeSplitQuantization);
        split = std::min(split, farPlane);
        m_shadowCascadeSplits[cascadeIndex] = split;
        cascadeDistances[cascadeIndex] = split;
    }

    std::array<math::Matrix4, kShadowCascadeCount> lightViewProjMatrices{};
    for (uint32_t cascadeIndex = 0; cascadeIndex < kShadowCascadeCount; ++cascadeIndex) {
        const float cascadeFar = cascadeDistances[cascadeIndex];
        const float farHalfHeight = cascadeFar * tanHalfFov;
        const float farHalfWidth = farHalfHeight * aspectRatio;

        // Camera-position-only cascades: only translation moves cascade centers; rotation does not.
        const math::Vector3 frustumCenter = eye;
        float boundingRadius =
            std::sqrt((cascadeFar * cascadeFar) + (farHalfWidth * farHalfWidth) + (farHalfHeight * farHalfHeight));
        boundingRadius = std::max(boundingRadius * 1.04f, 24.0f);
        boundingRadius = std::ceil(boundingRadius * 16.0f) / 16.0f;
        if (m_shadowStableCascadeRadii[cascadeIndex] <= 0.0f) {
            m_shadowStableCascadeRadii[cascadeIndex] = boundingRadius;
        }
        const float cascadeRadius = m_shadowStableCascadeRadii[cascadeIndex];
        const float orthoWidth = 2.0f * cascadeRadius;
        const float texelSize = orthoWidth / static_cast<float>(kShadowCascadeResolution[cascadeIndex]);

        // Keep the light farther than the cascade sphere but avoid overly large depth spans.
        const float lightDistance = (cascadeRadius * 1.9f) + 48.0f;
        const float sunUpDot = std::abs(math::dot(sunDirection, math::Vector3{0.0f, 1.0f, 0.0f}));
        const math::Vector3 lightUpHint =
            (sunUpDot > 0.95f) ? math::Vector3{0.0f, 0.0f, 1.0f} : math::Vector3{0.0f, 1.0f, 0.0f};
        const math::Vector3 lightForward = math::normalize(sunDirection);
        const math::Vector3 lightRight = math::normalize(math::cross(lightForward, lightUpHint));
        const math::Vector3 lightUp = math::cross(lightRight, lightForward);

        // Stabilize translation by snapping the cascade center along light-view right/up texel units
        // before constructing the view matrix.
        const float centerRight = math::dot(frustumCenter, lightRight);
        const float centerUp = math::dot(frustumCenter, lightUp);
        const float snappedCenterRight = std::floor((centerRight / texelSize) + 0.5f) * texelSize;
        const float snappedCenterUp = std::floor((centerUp / texelSize) + 0.5f) * texelSize;
        const math::Vector3 snappedFrustumCenter =
            frustumCenter +
            (lightRight * (snappedCenterRight - centerRight)) +
            (lightUp * (snappedCenterUp - centerUp));

        const math::Vector3 lightPosition = snappedFrustumCenter - (lightForward * lightDistance);
        const math::Matrix4 lightView = lookAt(lightPosition, snappedFrustumCenter, lightUp);

        const float left = -cascadeRadius;
        const float right = cascadeRadius;
        const float bottom = -cascadeRadius;
        const float top = cascadeRadius;
        // Keep a stable but tighter depth range per cascade to improve depth precision.
        const float casterPadding = std::max(24.0f, cascadeRadius * 0.35f);
        const float lightNear = std::max(0.1f, lightDistance - cascadeRadius - casterPadding);
        const float lightFar = lightDistance + cascadeRadius + casterPadding;
        const math::Matrix4 lightProjection = orthographicVulkan(
            left,
            right,
            bottom,
            top,
            lightNear,
            lightFar
        );
        lightViewProjMatrices[cascadeIndex] = lightProjection * lightView;
    }

    const std::array<math::Vector3, 9> shIrradiance =
        computeIrradianceShCoefficients(sunDirection, sunColor, m_skyDebugSettings);

    const std::optional<FrameArenaSlice> mvpSliceOpt =
        m_frameArena.allocateUpload(
            sizeof(CameraUniform),
            m_uniformBufferAlignment,
            FrameArenaUploadKind::CameraUniform
        );
    if (!mvpSliceOpt.has_value() || mvpSliceOpt->mapped == nullptr) {
        VOX_LOGE("render") << "failed to allocate MVP uniform slice\n";
        return;
    }

    CameraUniform mvpUniform{};
    std::memcpy(mvpUniform.mvp, mvpColumnMajor.m, sizeof(mvpUniform.mvp));
    std::memcpy(mvpUniform.view, viewColumnMajor.m, sizeof(mvpUniform.view));
    std::memcpy(mvpUniform.proj, projectionColumnMajor.m, sizeof(mvpUniform.proj));
    for (uint32_t cascadeIndex = 0; cascadeIndex < kShadowCascadeCount; ++cascadeIndex) {
        const math::Matrix4 lightViewProjColumnMajor = transpose(lightViewProjMatrices[cascadeIndex]);
        std::memcpy(
            mvpUniform.lightViewProj[cascadeIndex],
            lightViewProjColumnMajor.m,
            sizeof(mvpUniform.lightViewProj[cascadeIndex])
        );
        mvpUniform.shadowCascadeSplits[cascadeIndex] = cascadeDistances[cascadeIndex];
        const ShadowAtlasRect atlasRect = kShadowAtlasRects[cascadeIndex];
        mvpUniform.shadowAtlasUvRects[cascadeIndex][0] = static_cast<float>(atlasRect.x) / static_cast<float>(kShadowAtlasSize);
        mvpUniform.shadowAtlasUvRects[cascadeIndex][1] = static_cast<float>(atlasRect.y) / static_cast<float>(kShadowAtlasSize);
        mvpUniform.shadowAtlasUvRects[cascadeIndex][2] = static_cast<float>(atlasRect.size) / static_cast<float>(kShadowAtlasSize);
        mvpUniform.shadowAtlasUvRects[cascadeIndex][3] = static_cast<float>(atlasRect.size) / static_cast<float>(kShadowAtlasSize);
    }
    mvpUniform.sunDirectionIntensity[0] = sunDirection.x;
    mvpUniform.sunDirectionIntensity[1] = sunDirection.y;
    mvpUniform.sunDirectionIntensity[2] = sunDirection.z;
    mvpUniform.sunDirectionIntensity[3] = 2.2f;
    mvpUniform.sunColorShadow[0] = sunColor.x;
    mvpUniform.sunColorShadow[1] = sunColor.y;
    mvpUniform.sunColorShadow[2] = sunColor.z;
    mvpUniform.sunColorShadow[3] = 1.0f;
    for (uint32_t i = 0; i < shIrradiance.size(); ++i) {
        mvpUniform.shIrradiance[i][0] = shIrradiance[i].x;
        mvpUniform.shIrradiance[i][1] = shIrradiance[i].y;
        mvpUniform.shIrradiance[i][2] = shIrradiance[i].z;
        mvpUniform.shIrradiance[i][3] = 0.0f;
    }
    mvpUniform.shadowConfig0[0] = m_shadowDebugSettings.receiverNormalOffsetNear;
    mvpUniform.shadowConfig0[1] = m_shadowDebugSettings.receiverNormalOffsetFar;
    mvpUniform.shadowConfig0[2] = m_shadowDebugSettings.receiverBaseBiasNearTexel;
    mvpUniform.shadowConfig0[3] = m_shadowDebugSettings.receiverBaseBiasFarTexel;

    mvpUniform.shadowConfig1[0] = m_shadowDebugSettings.receiverSlopeBiasNearTexel;
    mvpUniform.shadowConfig1[1] = m_shadowDebugSettings.receiverSlopeBiasFarTexel;
    mvpUniform.shadowConfig1[2] = m_shadowDebugSettings.cascadeBlendMin;
    mvpUniform.shadowConfig1[3] = m_shadowDebugSettings.cascadeBlendFactor;

    mvpUniform.shadowConfig2[0] = m_shadowDebugSettings.ssaoRadius;
    mvpUniform.shadowConfig2[1] = m_shadowDebugSettings.ssaoBias;
    mvpUniform.shadowConfig2[2] = m_shadowDebugSettings.ssaoIntensity;
    mvpUniform.shadowConfig2[3] = 0.0f;

    mvpUniform.shadowConfig3[0] = 0.0f;
    mvpUniform.shadowConfig3[1] = 0.0f;
    mvpUniform.shadowConfig3[2] = 0.0f;
    mvpUniform.shadowConfig3[3] = m_shadowDebugSettings.pcfRadius;

    mvpUniform.shadowVoxelGridOrigin[0] = 0.0f;
    mvpUniform.shadowVoxelGridOrigin[1] = 0.0f;
    mvpUniform.shadowVoxelGridOrigin[2] = 0.0f;
    // Reuse unused W channel for AO debug: 1.0 enables vertex AO, 0.0 disables.
    mvpUniform.shadowVoxelGridOrigin[3] = m_debugEnableVertexAo ? 1.0f : 0.0f;

    // Reuse currently-unused XYZ channels to provide camera world position to shaders.
    mvpUniform.shadowVoxelGridSize[0] = camera.x;
    mvpUniform.shadowVoxelGridSize[1] = camera.y;
    mvpUniform.shadowVoxelGridSize[2] = camera.z;
    // Reuse unused W channel for AO debug mode:
    // 0.0 = SSAO off, 1.0 = SSAO on, 2.0 = visualize SSAO, 3.0 = visualize AO normals.
    if (m_debugVisualizeAoNormals) {
        mvpUniform.shadowVoxelGridSize[3] = 3.0f;
    } else if (m_debugVisualizeSsao) {
        mvpUniform.shadowVoxelGridSize[3] = 2.0f;
    } else {
        mvpUniform.shadowVoxelGridSize[3] = m_debugEnableSsao ? 1.0f : 0.0f;
    }

    mvpUniform.skyConfig0[0] = m_skyDebugSettings.rayleighStrength;
    mvpUniform.skyConfig0[1] = m_skyDebugSettings.mieStrength;
    mvpUniform.skyConfig0[2] = m_skyDebugSettings.mieAnisotropy;
    mvpUniform.skyConfig0[3] = m_skyDebugSettings.skyExposure;

    const float flowTimeSeconds = static_cast<float>(std::fmod(frameNowSeconds, 4096.0));
    mvpUniform.skyConfig1[0] = 1150.0f;
    mvpUniform.skyConfig1[1] = 22.0f;
    mvpUniform.skyConfig1[2] = flowTimeSeconds;
    mvpUniform.skyConfig1[3] = 1.85f;
    std::memcpy(mvpSliceOpt->mapped, &mvpUniform, sizeof(mvpUniform));

    VkDescriptorBufferInfo bufferInfo{};
    bufferInfo.buffer = m_bufferAllocator.getBuffer(mvpSliceOpt->buffer);
    bufferInfo.offset = 0;
    bufferInfo.range = sizeof(CameraUniform);
    if (mvpSliceOpt->offset > static_cast<VkDeviceSize>(std::numeric_limits<uint32_t>::max())) {
        VOX_LOGI("render") << "dynamic UBO offset exceeds uint32 range\n";
        return;
    }
    const uint32_t mvpDynamicOffset = static_cast<uint32_t>(mvpSliceOpt->offset);

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;

    VkDescriptorImageInfo hdrSceneImageInfo{};
    hdrSceneImageInfo.sampler = m_hdrResolveSampler;
    hdrSceneImageInfo.imageView = m_hdrResolveImageViews[aoFrameIndex];
    hdrSceneImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDescriptorImageInfo diffuseTextureImageInfo{};
    diffuseTextureImageInfo.sampler = m_diffuseTextureSampler;
    diffuseTextureImageInfo.imageView = m_diffuseTextureImageView;
    diffuseTextureImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDescriptorImageInfo shadowMapImageInfo{};
    shadowMapImageInfo.sampler = m_shadowDepthSampler;
    shadowMapImageInfo.imageView = m_shadowDepthImageView;
    shadowMapImageInfo.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

    VkDescriptorImageInfo normalDepthImageInfo{};
    normalDepthImageInfo.sampler = m_normalDepthSampler;
    normalDepthImageInfo.imageView = m_normalDepthImageViews[aoFrameIndex];
    normalDepthImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDescriptorImageInfo ssaoBlurImageInfo{};
    ssaoBlurImageInfo.sampler = m_ssaoSampler;
    ssaoBlurImageInfo.imageView = m_ssaoBlurImageViews[aoFrameIndex];
    ssaoBlurImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDescriptorImageInfo ssaoRawImageInfo{};
    ssaoRawImageInfo.sampler = m_ssaoSampler;
    ssaoRawImageInfo.imageView = m_ssaoRawImageViews[aoFrameIndex];
    ssaoRawImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    std::array<VkWriteDescriptorSet, 7> writes{};
    writes[0] = write;
    writes[0].dstSet = m_descriptorSets[m_currentFrame];
    writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
    writes[0].pBufferInfo = &bufferInfo;

    writes[1] = write;
    writes[1].dstSet = m_descriptorSets[m_currentFrame];
    writes[1].dstBinding = 1;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[1].pImageInfo = &diffuseTextureImageInfo;

    writes[2] = write;
    writes[2].dstSet = m_descriptorSets[m_currentFrame];
    writes[2].dstBinding = 3;
    writes[2].descriptorCount = 1;
    writes[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[2].pImageInfo = &hdrSceneImageInfo;

    writes[3] = write;
    writes[3].dstSet = m_descriptorSets[m_currentFrame];
    writes[3].dstBinding = 4;
    writes[3].descriptorCount = 1;
    writes[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[3].pImageInfo = &shadowMapImageInfo;

    writes[4] = write;
    writes[4].dstSet = m_descriptorSets[m_currentFrame];
    writes[4].dstBinding = 6;
    writes[4].descriptorCount = 1;
    writes[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[4].pImageInfo = &normalDepthImageInfo;

    writes[5] = write;
    writes[5].dstSet = m_descriptorSets[m_currentFrame];
    writes[5].dstBinding = 7;
    writes[5].descriptorCount = 1;
    writes[5].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[5].pImageInfo = &ssaoBlurImageInfo;

    writes[6] = write;
    writes[6].dstSet = m_descriptorSets[m_currentFrame];
    writes[6].dstBinding = 8;
    writes[6].descriptorCount = 1;
    writes[6].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[6].pImageInfo = &ssaoRawImageInfo;

    vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

    if (m_bindlessDescriptorSet != VK_NULL_HANDLE && m_bindlessTextureCapacity >= kBindlessTextureStaticCount) {
        std::array<VkDescriptorImageInfo, kBindlessTextureStaticCount> bindlessImageInfos{};
        bindlessImageInfos[kBindlessTextureIndexDiffuse] = diffuseTextureImageInfo;
        bindlessImageInfos[kBindlessTextureIndexHdrResolved] = hdrSceneImageInfo;
        bindlessImageInfos[kBindlessTextureIndexShadowAtlas] = shadowMapImageInfo;
        bindlessImageInfos[kBindlessTextureIndexNormalDepth] = normalDepthImageInfo;
        bindlessImageInfos[kBindlessTextureIndexSsaoBlur] = ssaoBlurImageInfo;
        bindlessImageInfos[kBindlessTextureIndexSsaoRaw] = ssaoRawImageInfo;

        VkWriteDescriptorSet bindlessWrite{};
        bindlessWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        bindlessWrite.dstSet = m_bindlessDescriptorSet;
        bindlessWrite.dstBinding = 0;
        bindlessWrite.dstArrayElement = 0;
        bindlessWrite.descriptorCount = kBindlessTextureStaticCount;
        bindlessWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindlessWrite.pImageInfo = bindlessImageInfos.data();
        vkUpdateDescriptorSets(m_device, 1, &bindlessWrite, 0, nullptr);
    }
    std::array<VkDescriptorSet, 2> boundDescriptorSets = {
        m_descriptorSets[m_currentFrame],
        m_bindlessDescriptorSet
    };
    const uint32_t boundDescriptorSetCount =
        (m_bindlessDescriptorSet != VK_NULL_HANDLE) ? 2u : 1u;

    uint32_t pipeInstanceCount = 0;
    std::optional<FrameArenaSlice> pipeInstanceSliceOpt = std::nullopt;
    uint32_t transportInstanceCount = 0;
    std::optional<FrameArenaSlice> transportInstanceSliceOpt = std::nullopt;
    if (m_pipeIndexCount > 0 || m_transportIndexCount > 0) {
        const std::vector<sim::Pipe>& pipes = simulation.pipes();
        const std::vector<sim::Belt>& belts = simulation.belts();
        const std::vector<sim::Track>& tracks = simulation.tracks();
        const std::vector<PipeEndpointState> endpointStates =
            pipes.empty() ? std::vector<PipeEndpointState>{} : buildPipeEndpointStates(pipes);
        std::vector<PipeInstance> pipeInstances;
        pipeInstances.reserve(pipes.size());
        for (std::size_t pipeIndex = 0; pipeIndex < pipes.size(); ++pipeIndex) {
            const sim::Pipe& pipe = pipes[pipeIndex];
            const PipeEndpointState& endpointState = endpointStates[pipeIndex];
            PipeInstance instance{};
            instance.originLength[0] = static_cast<float>(pipe.x);
            instance.originLength[1] = static_cast<float>(pipe.y);
            instance.originLength[2] = static_cast<float>(pipe.z);
            instance.originLength[3] = std::max(pipe.length, 0.05f);
            instance.axisRadius[0] = endpointState.axis.x;
            instance.axisRadius[1] = endpointState.axis.y;
            instance.axisRadius[2] = endpointState.axis.z;
            instance.axisRadius[3] = endpointState.renderedRadius;
            instance.tint[0] = std::clamp(pipe.tint.x, 0.0f, 1.0f);
            instance.tint[1] = std::clamp(pipe.tint.y, 0.0f, 1.0f);
            instance.tint[2] = std::clamp(pipe.tint.z, 0.0f, 1.0f);
            instance.tint[3] = 0.0f; // style 0 = pipe
            instance.extensions[0] = endpointState.startExtension;
            instance.extensions[1] = endpointState.endExtension;
            instance.extensions[2] = 1.0f;
            instance.extensions[3] = 1.0f;
            pipeInstances.push_back(instance);
        }

        std::vector<PipeInstance> transportInstances;
        transportInstances.reserve(belts.size() + tracks.size());
        for (const sim::Belt& belt : belts) {
            PipeInstance instance{};
            instance.originLength[0] = static_cast<float>(belt.x);
            instance.originLength[1] = static_cast<float>(belt.y);
            instance.originLength[2] = static_cast<float>(belt.z);
            instance.originLength[3] = 1.0f;
            const math::Vector3 axis = beltDirectionAxis(belt.direction);
            instance.axisRadius[0] = axis.x;
            instance.axisRadius[1] = axis.y;
            instance.axisRadius[2] = axis.z;
            instance.axisRadius[3] = kBeltRadius;
            instance.tint[0] = kBeltTint.x;
            instance.tint[1] = kBeltTint.y;
            instance.tint[2] = kBeltTint.z;
            instance.tint[3] = 1.0f; // style 1 = conveyor
            instance.extensions[0] = 0.0f;
            instance.extensions[1] = 0.0f;
            // Conveyors: 2x wider cross-span, 0.25x height.
            instance.extensions[2] = 2.0f;
            instance.extensions[3] = 0.25f;
            transportInstances.push_back(instance);
        }

        for (const sim::Track& track : tracks) {
            PipeInstance instance{};
            instance.originLength[0] = static_cast<float>(track.x);
            instance.originLength[1] = static_cast<float>(track.y);
            instance.originLength[2] = static_cast<float>(track.z);
            instance.originLength[3] = 1.0f;
            const math::Vector3 axis = trackDirectionAxis(track.direction);
            instance.axisRadius[0] = axis.x;
            instance.axisRadius[1] = axis.y;
            instance.axisRadius[2] = axis.z;
            instance.axisRadius[3] = kTrackRadius;
            instance.tint[0] = kTrackTint.x;
            instance.tint[1] = kTrackTint.y;
            instance.tint[2] = kTrackTint.z;
            instance.tint[3] = 2.0f; // style 2 = track
            instance.extensions[0] = 0.0f;
            instance.extensions[1] = 0.0f;
            // Tracks: 2x wider cross-span, 0.25x height.
            instance.extensions[2] = 2.0f;
            instance.extensions[3] = 0.25f;
            transportInstances.push_back(instance);
        }

        if (!pipeInstances.empty() && m_pipeIndexCount > 0) {
            pipeInstanceSliceOpt = m_frameArena.allocateUpload(
                static_cast<VkDeviceSize>(pipeInstances.size() * sizeof(PipeInstance)),
                static_cast<VkDeviceSize>(alignof(PipeInstance)),
                FrameArenaUploadKind::InstanceData
            );
            if (pipeInstanceSliceOpt.has_value() && pipeInstanceSliceOpt->mapped != nullptr) {
                std::memcpy(
                    pipeInstanceSliceOpt->mapped,
                    pipeInstances.data(),
                    static_cast<size_t>(pipeInstanceSliceOpt->size)
                );
                pipeInstanceCount = static_cast<uint32_t>(pipeInstances.size());
            }
        }

        if (!transportInstances.empty() && m_transportIndexCount > 0) {
            transportInstanceSliceOpt = m_frameArena.allocateUpload(
                static_cast<VkDeviceSize>(transportInstances.size() * sizeof(PipeInstance)),
                static_cast<VkDeviceSize>(alignof(PipeInstance)),
                FrameArenaUploadKind::InstanceData
            );
            if (transportInstanceSliceOpt.has_value() && transportInstanceSliceOpt->mapped != nullptr) {
                std::memcpy(
                    transportInstanceSliceOpt->mapped,
                    transportInstances.data(),
                    static_cast<size_t>(transportInstanceSliceOpt->size)
                );
                transportInstanceCount = static_cast<uint32_t>(transportInstances.size());
            }
        }
    }

    const VkBuffer chunkVertexBuffer = m_bufferAllocator.getBuffer(m_chunkVertexBufferHandle);
    const VkBuffer chunkIndexBuffer = m_bufferAllocator.getBuffer(m_chunkIndexBufferHandle);
    const bool chunkDrawBuffersReady = chunkVertexBuffer != VK_NULL_HANDLE && chunkIndexBuffer != VK_NULL_HANDLE;

    std::vector<ChunkInstanceData> chunkInstanceData;
    chunkInstanceData.reserve(m_chunkDrawRanges.size() + 1);
    chunkInstanceData.push_back(ChunkInstanceData{});
    std::vector<VkDrawIndexedIndirectCommand> chunkIndirectCommands;
    chunkIndirectCommands.reserve(m_chunkDrawRanges.size());
    const std::vector<world::Chunk>& chunks = chunkGrid.chunks();
    auto appendChunkLods = [&](std::size_t chunkArrayIndex) {
        if (chunkArrayIndex >= chunkGrid.chunks().size()) {
            return;
        }
        const world::Chunk& drawChunk = chunks[chunkArrayIndex];
        const bool allowDetailLods =
            drawChunk.chunkX() == cameraChunkX &&
            drawChunk.chunkY() == cameraChunkY &&
            drawChunk.chunkZ() == cameraChunkZ;
        for (std::size_t lodIndex = 0; lodIndex < world::kChunkMeshLodCount; ++lodIndex) {
            if (lodIndex > 0 && !allowDetailLods) {
                continue;
            }
            const std::size_t drawRangeIndex = (chunkArrayIndex * world::kChunkMeshLodCount) + lodIndex;
            if (drawRangeIndex >= m_chunkDrawRanges.size()) {
                continue;
            }
            const ChunkDrawRange& drawRange = m_chunkDrawRanges[drawRangeIndex];
            if (drawRange.indexCount == 0 || !chunkDrawBuffersReady) {
                continue;
            }

            const uint32_t instanceIndex = static_cast<uint32_t>(chunkInstanceData.size());
            ChunkInstanceData instance{};
            instance.chunkOffset[0] = drawRange.offsetX;
            instance.chunkOffset[1] = drawRange.offsetY;
            instance.chunkOffset[2] = drawRange.offsetZ;
            instance.chunkOffset[3] = 0.0f;
            chunkInstanceData.push_back(instance);

            VkDrawIndexedIndirectCommand indirectCommand{};
            indirectCommand.indexCount = drawRange.indexCount;
            indirectCommand.instanceCount = 1;
            indirectCommand.firstIndex = drawRange.firstIndex;
            indirectCommand.vertexOffset = drawRange.vertexOffset;
            indirectCommand.firstInstance = instanceIndex;
            chunkIndirectCommands.push_back(indirectCommand);

            if (lodIndex == 0) {
                ++m_debugDrawnLod0Ranges;
            } else if (lodIndex == 1) {
                ++m_debugDrawnLod1Ranges;
            } else {
                ++m_debugDrawnLod2Ranges;
            }
        }
    };
    if (!visibleChunkIndices.empty()) {
        for (const std::size_t chunkArrayIndex : visibleChunkIndices) {
            appendChunkLods(chunkArrayIndex);
        }
    } else {
        for (std::size_t chunkArrayIndex = 0; chunkArrayIndex < chunks.size(); ++chunkArrayIndex) {
            appendChunkLods(chunkArrayIndex);
        }
    }

    const VkDeviceSize chunkInstanceBytes =
        static_cast<VkDeviceSize>(chunkInstanceData.size() * sizeof(ChunkInstanceData));
    std::optional<FrameArenaSlice> chunkInstanceSliceOpt = std::nullopt;
    if (chunkInstanceBytes > 0) {
        chunkInstanceSliceOpt = m_frameArena.allocateUpload(
            chunkInstanceBytes,
            static_cast<VkDeviceSize>(alignof(ChunkInstanceData)),
            FrameArenaUploadKind::InstanceData
        );
        if (chunkInstanceSliceOpt.has_value() && chunkInstanceSliceOpt->mapped != nullptr) {
            std::memcpy(chunkInstanceSliceOpt->mapped, chunkInstanceData.data(), static_cast<size_t>(chunkInstanceBytes));
        } else {
            chunkInstanceSliceOpt.reset();
        }
    }

    const VkDeviceSize chunkIndirectBytes =
        static_cast<VkDeviceSize>(chunkIndirectCommands.size() * sizeof(VkDrawIndexedIndirectCommand));
    std::optional<FrameArenaSlice> chunkIndirectSliceOpt = std::nullopt;
    if (chunkIndirectBytes > 0) {
        chunkIndirectSliceOpt = m_frameArena.allocateUpload(
            chunkIndirectBytes,
            static_cast<VkDeviceSize>(alignof(VkDrawIndexedIndirectCommand)),
            FrameArenaUploadKind::Unknown
        );
        if (chunkIndirectSliceOpt.has_value() && chunkIndirectSliceOpt->mapped != nullptr) {
            std::memcpy(
                chunkIndirectSliceOpt->mapped,
                chunkIndirectCommands.data(),
                static_cast<size_t>(chunkIndirectBytes)
            );
        } else {
            chunkIndirectSliceOpt.reset();
        }
    }

    const VkBuffer chunkInstanceBuffer =
        chunkInstanceSliceOpt.has_value() ? m_bufferAllocator.getBuffer(chunkInstanceSliceOpt->buffer) : VK_NULL_HANDLE;
    const VkBuffer chunkIndirectBuffer =
        chunkIndirectSliceOpt.has_value() ? m_bufferAllocator.getBuffer(chunkIndirectSliceOpt->buffer) : VK_NULL_HANDLE;
    const uint32_t chunkIndirectDrawCount = static_cast<uint32_t>(chunkIndirectCommands.size());
    m_debugChunkIndirectCommandCount = chunkIndirectDrawCount;
    const bool canDrawChunksIndirect =
        chunkIndirectDrawCount > 0 &&
        chunkInstanceSliceOpt.has_value() &&
        chunkIndirectSliceOpt.has_value() &&
        chunkInstanceBuffer != VK_NULL_HANDLE &&
        chunkIndirectBuffer != VK_NULL_HANDLE &&
        chunkDrawBuffersReady;
    auto countDrawCalls = [&](std::uint32_t& passCounter, std::uint32_t drawCount) {
        passCounter += drawCount;
        m_debugDrawCallsTotal += drawCount;
    };
    const auto drawChunkIndirect = [&](std::uint32_t& passCounter) {
        if (!canDrawChunksIndirect) {
            return;
        }
        if (m_supportsMultiDrawIndirect) {
            countDrawCalls(passCounter, chunkIndirectDrawCount);
            vkCmdDrawIndexedIndirect(
                commandBuffer,
                chunkIndirectBuffer,
                chunkIndirectSliceOpt->offset,
                chunkIndirectDrawCount,
                sizeof(VkDrawIndexedIndirectCommand)
            );
            return;
        }
        const VkDeviceSize stride = static_cast<VkDeviceSize>(sizeof(VkDrawIndexedIndirectCommand));
        VkDeviceSize drawOffset = chunkIndirectSliceOpt->offset;
        for (uint32_t drawIndex = 0; drawIndex < chunkIndirectDrawCount; ++drawIndex) {
            countDrawCalls(passCounter, 1);
            vkCmdDrawIndexedIndirect(
                commandBuffer,
                chunkIndirectBuffer,
                drawOffset,
                1,
                static_cast<uint32_t>(stride)
            );
            drawOffset += stride;
        }
    };

    writeGpuTimestampTop(kGpuTimestampQueryShadowStart);
    beginDebugLabel(commandBuffer, "Pass: Shadow Atlas", 0.28f, 0.22f, 0.22f, 1.0f);
    const bool shadowInitialized = m_shadowDepthInitialized;
    transitionImageLayout(
        commandBuffer,
        m_shadowDepthImage,
        shadowInitialized ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
        shadowInitialized ? VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT : VK_PIPELINE_STAGE_2_NONE,
        shadowInitialized ? VK_ACCESS_2_SHADER_SAMPLED_READ_BIT : VK_ACCESS_2_NONE,
        VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
        VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
        VK_IMAGE_ASPECT_DEPTH_BIT,
        0,
        1
    );

    VkClearValue shadowDepthClearValue{};
    shadowDepthClearValue.depthStencil.depth = 0.0f;
    shadowDepthClearValue.depthStencil.stencil = 0;

    if (m_shadowPipeline != VK_NULL_HANDLE) {
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_shadowPipeline);
        vkCmdBindDescriptorSets(
            commandBuffer,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            m_pipelineLayout,
            0,
            boundDescriptorSetCount,
            boundDescriptorSets.data(),
            1,
            &mvpDynamicOffset
        );

        for (uint32_t cascadeIndex = 0; cascadeIndex < kShadowCascadeCount; ++cascadeIndex) {
            if (m_cmdInsertDebugUtilsLabel != nullptr) {
                const std::string cascadeLabel = "Shadow Cascade " + std::to_string(cascadeIndex);
                insertDebugLabel(commandBuffer, cascadeLabel.c_str(), 0.48f, 0.32f, 0.32f, 1.0f);
            }
            const ShadowAtlasRect atlasRect = kShadowAtlasRects[cascadeIndex];
            VkViewport shadowViewport{};
            shadowViewport.x = static_cast<float>(atlasRect.x);
            shadowViewport.y = static_cast<float>(atlasRect.y);
            shadowViewport.width = static_cast<float>(atlasRect.size);
            shadowViewport.height = static_cast<float>(atlasRect.size);
            shadowViewport.minDepth = 0.0f;
            shadowViewport.maxDepth = 1.0f;

            VkRect2D shadowScissor{};
            shadowScissor.offset = {
                static_cast<int32_t>(atlasRect.x),
                static_cast<int32_t>(atlasRect.y)
            };
            shadowScissor.extent = {atlasRect.size, atlasRect.size};

            VkRenderingAttachmentInfo shadowDepthAttachment{};
            shadowDepthAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            shadowDepthAttachment.imageView = m_shadowDepthImageView;
            shadowDepthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
            shadowDepthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            shadowDepthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            shadowDepthAttachment.clearValue = shadowDepthClearValue;

            VkRenderingInfo shadowRenderingInfo{};
            shadowRenderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
            shadowRenderingInfo.renderArea.offset = shadowScissor.offset;
            shadowRenderingInfo.renderArea.extent = shadowScissor.extent;
            shadowRenderingInfo.layerCount = 1;
            shadowRenderingInfo.colorAttachmentCount = 0;
            shadowRenderingInfo.pDepthAttachment = &shadowDepthAttachment;

            vkCmdBeginRendering(commandBuffer, &shadowRenderingInfo);
            vkCmdSetViewport(commandBuffer, 0, 1, &shadowViewport);
            vkCmdSetScissor(commandBuffer, 0, 1, &shadowScissor);
            const float cascadeF = static_cast<float>(cascadeIndex);
            const float constantBias =
                m_shadowDebugSettings.casterConstantBiasBase +
                (m_shadowDebugSettings.casterConstantBiasCascadeScale * cascadeF);
            const float slopeBias =
                m_shadowDebugSettings.casterSlopeBiasBase +
                (m_shadowDebugSettings.casterSlopeBiasCascadeScale * cascadeF);
            // Reverse-Z uses GREATER depth tests, so flip bias sign.
            vkCmdSetDepthBias(commandBuffer, -constantBias, 0.0f, -slopeBias);

            if (canDrawChunksIndirect) {
                vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_shadowPipeline);
                vkCmdBindDescriptorSets(
                    commandBuffer,
                    VK_PIPELINE_BIND_POINT_GRAPHICS,
                    m_pipelineLayout,
                    0,
                    boundDescriptorSetCount,
                    boundDescriptorSets.data(),
                    1,
                    &mvpDynamicOffset
                );
                const VkBuffer voxelVertexBuffers[2] = {chunkVertexBuffer, chunkInstanceBuffer};
                const VkDeviceSize voxelVertexOffsets[2] = {0, chunkInstanceSliceOpt->offset};
                vkCmdBindVertexBuffers(commandBuffer, 0, 2, voxelVertexBuffers, voxelVertexOffsets);
                vkCmdBindIndexBuffer(commandBuffer, chunkIndexBuffer, 0, VK_INDEX_TYPE_UINT32);

                ChunkPushConstants chunkPushConstants{};
                chunkPushConstants.chunkOffset[0] = 0.0f;
                chunkPushConstants.chunkOffset[1] = 0.0f;
                chunkPushConstants.chunkOffset[2] = 0.0f;
                chunkPushConstants.chunkOffset[3] = 0.0f;
                chunkPushConstants.cascadeData[0] = static_cast<float>(cascadeIndex);
                chunkPushConstants.cascadeData[1] = 0.0f;
                chunkPushConstants.cascadeData[2] = 0.0f;
                chunkPushConstants.cascadeData[3] = 0.0f;
                vkCmdPushConstants(
                    commandBuffer,
                    m_pipelineLayout,
                    VK_SHADER_STAGE_VERTEX_BIT,
                    0,
                    sizeof(ChunkPushConstants),
                    &chunkPushConstants
                );
                drawChunkIndirect(m_debugDrawCallsShadow);
            }

            if (m_pipeShadowPipeline != VK_NULL_HANDLE) {
                auto drawShadowInstances = [&](
                                               BufferHandle vertexHandle,
                                               BufferHandle indexHandle,
                                               uint32_t indexCount,
                                               uint32_t instanceCount,
                                           const std::optional<FrameArenaSlice>& instanceSlice
                                           ) {
                    if (instanceCount == 0 || !instanceSlice.has_value() || indexCount == 0) {
                        return;
                    }
                    const VkBuffer vertexBuffer = m_bufferAllocator.getBuffer(vertexHandle);
                    const VkBuffer indexBuffer = m_bufferAllocator.getBuffer(indexHandle);
                    const VkBuffer instanceBuffer = m_bufferAllocator.getBuffer(instanceSlice->buffer);
                    if (vertexBuffer == VK_NULL_HANDLE || indexBuffer == VK_NULL_HANDLE || instanceBuffer == VK_NULL_HANDLE) {
                        return;
                    }
                    const VkBuffer vertexBuffers[2] = {vertexBuffer, instanceBuffer};
                    const VkDeviceSize vertexOffsets[2] = {0, instanceSlice->offset};
                    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeShadowPipeline);
                    vkCmdBindDescriptorSets(
                        commandBuffer,
                        VK_PIPELINE_BIND_POINT_GRAPHICS,
                        m_pipelineLayout,
                        0,
                        boundDescriptorSetCount,
                        boundDescriptorSets.data(),
            1,
            &mvpDynamicOffset
        );
                    vkCmdBindVertexBuffers(commandBuffer, 0, 2, vertexBuffers, vertexOffsets);
                    vkCmdBindIndexBuffer(commandBuffer, indexBuffer, 0, VK_INDEX_TYPE_UINT32);

                    ChunkPushConstants pipeShadowPushConstants{};
                    pipeShadowPushConstants.chunkOffset[0] = 0.0f;
                    pipeShadowPushConstants.chunkOffset[1] = 0.0f;
                    pipeShadowPushConstants.chunkOffset[2] = 0.0f;
                    pipeShadowPushConstants.chunkOffset[3] = 0.0f;
                    pipeShadowPushConstants.cascadeData[0] = static_cast<float>(cascadeIndex);
                    pipeShadowPushConstants.cascadeData[1] = 0.0f;
                    pipeShadowPushConstants.cascadeData[2] = 0.0f;
                    pipeShadowPushConstants.cascadeData[3] = 0.0f;
                    vkCmdPushConstants(
                        commandBuffer,
                        m_pipelineLayout,
                        VK_SHADER_STAGE_VERTEX_BIT,
                        0,
                        sizeof(ChunkPushConstants),
                        &pipeShadowPushConstants
                    );
                    countDrawCalls(m_debugDrawCallsShadow, 1);
                    vkCmdDrawIndexed(commandBuffer, indexCount, instanceCount, 0, 0, 0);
                };
                drawShadowInstances(
                    m_pipeVertexBufferHandle,
                    m_pipeIndexBufferHandle,
                    m_pipeIndexCount,
                    pipeInstanceCount,
                    pipeInstanceSliceOpt
                );
                drawShadowInstances(
                    m_transportVertexBufferHandle,
                    m_transportIndexBufferHandle,
                    m_transportIndexCount,
                    transportInstanceCount,
                    transportInstanceSliceOpt
                );
            }

            if (m_grassBillboardShadowPipeline != VK_NULL_HANDLE &&
                m_grassBillboardIndexCount > 0 &&
                m_grassBillboardInstanceCount > 0 &&
                m_grassBillboardInstanceBufferHandle != kInvalidBufferHandle) {
                const VkBuffer grassVertexBuffer = m_bufferAllocator.getBuffer(m_grassBillboardVertexBufferHandle);
                const VkBuffer grassIndexBuffer = m_bufferAllocator.getBuffer(m_grassBillboardIndexBufferHandle);
                const VkBuffer grassInstanceBuffer = m_bufferAllocator.getBuffer(m_grassBillboardInstanceBufferHandle);
                if (grassVertexBuffer != VK_NULL_HANDLE &&
                    grassIndexBuffer != VK_NULL_HANDLE &&
                    grassInstanceBuffer != VK_NULL_HANDLE) {
                    const VkBuffer vertexBuffers[2] = {grassVertexBuffer, grassInstanceBuffer};
                    const VkDeviceSize vertexOffsets[2] = {0, 0};
                    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_grassBillboardShadowPipeline);
                    vkCmdBindDescriptorSets(
                        commandBuffer,
                        VK_PIPELINE_BIND_POINT_GRAPHICS,
                        m_pipelineLayout,
                        0,
                        boundDescriptorSetCount,
                        boundDescriptorSets.data(),
                        1,
                        &mvpDynamicOffset
                    );
                    vkCmdBindVertexBuffers(commandBuffer, 0, 2, vertexBuffers, vertexOffsets);
                    vkCmdBindIndexBuffer(commandBuffer, grassIndexBuffer, 0, VK_INDEX_TYPE_UINT32);

                    ChunkPushConstants grassShadowPushConstants{};
                    grassShadowPushConstants.chunkOffset[0] = 0.0f;
                    grassShadowPushConstants.chunkOffset[1] = 0.0f;
                    grassShadowPushConstants.chunkOffset[2] = 0.0f;
                    grassShadowPushConstants.chunkOffset[3] = 0.0f;
                    grassShadowPushConstants.cascadeData[0] = static_cast<float>(cascadeIndex);
                    grassShadowPushConstants.cascadeData[1] = 0.0f;
                    grassShadowPushConstants.cascadeData[2] = 0.0f;
                    grassShadowPushConstants.cascadeData[3] = 0.0f;
                    vkCmdPushConstants(
                        commandBuffer,
                        m_pipelineLayout,
                        VK_SHADER_STAGE_VERTEX_BIT,
                        0,
                        sizeof(ChunkPushConstants),
                        &grassShadowPushConstants
                    );
                    countDrawCalls(m_debugDrawCallsShadow, 1);
                    vkCmdDrawIndexed(commandBuffer, m_grassBillboardIndexCount, m_grassBillboardInstanceCount, 0, 0, 0);
                }
            }

            vkCmdEndRendering(commandBuffer);
        }
    }

    transitionImageLayout(
        commandBuffer,
        m_shadowDepthImage,
        VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
        VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
        VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
        VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
        VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
        VK_IMAGE_ASPECT_DEPTH_BIT,
        0,
        1
    );
    endDebugLabel(commandBuffer);
    writeGpuTimestampBottom(kGpuTimestampQueryShadowEnd);

    const VkExtent2D aoExtent = {
        std::max(1u, m_aoExtent.width),
        std::max(1u, m_aoExtent.height)
    };

    const bool normalDepthInitialized = m_normalDepthImageInitialized[aoFrameIndex];
    transitionImageLayout(
        commandBuffer,
        m_normalDepthImages[aoFrameIndex],
        normalDepthInitialized ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        normalDepthInitialized ? VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT : VK_PIPELINE_STAGE_2_NONE,
        normalDepthInitialized ? VK_ACCESS_2_SHADER_SAMPLED_READ_BIT : VK_ACCESS_2_NONE,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT
    );

    const bool aoDepthInitialized = m_aoDepthImageInitialized[imageIndex];
    transitionImageLayout(
        commandBuffer,
        m_aoDepthImages[imageIndex],
        aoDepthInitialized ? VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
        aoDepthInitialized
            ? (VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT)
            : VK_PIPELINE_STAGE_2_NONE,
        aoDepthInitialized ? VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT : VK_ACCESS_2_NONE,
        VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
        VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
        VK_IMAGE_ASPECT_DEPTH_BIT
    );

    const bool ssaoRawInitialized = m_ssaoRawImageInitialized[aoFrameIndex];
    transitionImageLayout(
        commandBuffer,
        m_ssaoRawImages[aoFrameIndex],
        ssaoRawInitialized ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        ssaoRawInitialized ? VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT : VK_PIPELINE_STAGE_2_NONE,
        ssaoRawInitialized ? VK_ACCESS_2_SHADER_SAMPLED_READ_BIT : VK_ACCESS_2_NONE,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT
    );

    const bool ssaoBlurInitialized = m_ssaoBlurImageInitialized[aoFrameIndex];
    transitionImageLayout(
        commandBuffer,
        m_ssaoBlurImages[aoFrameIndex],
        ssaoBlurInitialized ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        ssaoBlurInitialized ? VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT : VK_PIPELINE_STAGE_2_NONE,
        ssaoBlurInitialized ? VK_ACCESS_2_SHADER_SAMPLED_READ_BIT : VK_ACCESS_2_NONE,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT
    );

    VkViewport aoViewport{};
    aoViewport.x = 0.0f;
    aoViewport.y = 0.0f;
    aoViewport.width = static_cast<float>(aoExtent.width);
    aoViewport.height = static_cast<float>(aoExtent.height);
    aoViewport.minDepth = 0.0f;
    aoViewport.maxDepth = 1.0f;

    VkRect2D aoScissor{};
    aoScissor.offset = {0, 0};
    aoScissor.extent = aoExtent;

    VkClearValue normalDepthClearValue{};
    normalDepthClearValue.color.float32[0] = 0.5f;
    normalDepthClearValue.color.float32[1] = 0.5f;
    normalDepthClearValue.color.float32[2] = 0.5f;
    normalDepthClearValue.color.float32[3] = 0.0f;

    VkClearValue aoDepthClearValue{};
    aoDepthClearValue.depthStencil.depth = 0.0f;
    aoDepthClearValue.depthStencil.stencil = 0;

    VkRenderingAttachmentInfo normalDepthColorAttachment{};
    normalDepthColorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    normalDepthColorAttachment.imageView = m_normalDepthImageViews[aoFrameIndex];
    normalDepthColorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    normalDepthColorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    normalDepthColorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    normalDepthColorAttachment.clearValue = normalDepthClearValue;

    VkRenderingAttachmentInfo aoDepthAttachment{};
    aoDepthAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    aoDepthAttachment.imageView = m_aoDepthImageViews[imageIndex];
    aoDepthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    aoDepthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    aoDepthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    aoDepthAttachment.clearValue = aoDepthClearValue;

    VkRenderingInfo normalDepthRenderingInfo{};
    normalDepthRenderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    normalDepthRenderingInfo.renderArea.offset = {0, 0};
    normalDepthRenderingInfo.renderArea.extent = aoExtent;
    normalDepthRenderingInfo.layerCount = 1;
    normalDepthRenderingInfo.colorAttachmentCount = 1;
    normalDepthRenderingInfo.pColorAttachments = &normalDepthColorAttachment;
    normalDepthRenderingInfo.pDepthAttachment = &aoDepthAttachment;

    writeGpuTimestampTop(kGpuTimestampQueryPrepassStart);
    beginDebugLabel(commandBuffer, "Pass: Normal+Depth Prepass", 0.20f, 0.30f, 0.40f, 1.0f);
    vkCmdBeginRendering(commandBuffer, &normalDepthRenderingInfo);
    vkCmdSetViewport(commandBuffer, 0, 1, &aoViewport);
    vkCmdSetScissor(commandBuffer, 0, 1, &aoScissor);

    if (m_voxelNormalDepthPipeline != VK_NULL_HANDLE) {
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_voxelNormalDepthPipeline);
        vkCmdBindDescriptorSets(
            commandBuffer,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            m_pipelineLayout,
            0,
            boundDescriptorSetCount,
            boundDescriptorSets.data(),
            1,
            &mvpDynamicOffset
        );
        if (canDrawChunksIndirect) {
            const VkBuffer voxelVertexBuffers[2] = {chunkVertexBuffer, chunkInstanceBuffer};
            const VkDeviceSize voxelVertexOffsets[2] = {0, chunkInstanceSliceOpt->offset};
            vkCmdBindVertexBuffers(commandBuffer, 0, 2, voxelVertexBuffers, voxelVertexOffsets);
            vkCmdBindIndexBuffer(commandBuffer, chunkIndexBuffer, 0, VK_INDEX_TYPE_UINT32);

            ChunkPushConstants chunkPushConstants{};
            chunkPushConstants.chunkOffset[0] = 0.0f;
            chunkPushConstants.chunkOffset[1] = 0.0f;
            chunkPushConstants.chunkOffset[2] = 0.0f;
            chunkPushConstants.chunkOffset[3] = 0.0f;
            chunkPushConstants.cascadeData[0] = 0.0f;
            chunkPushConstants.cascadeData[1] = 0.0f;
            chunkPushConstants.cascadeData[2] = 0.0f;
            chunkPushConstants.cascadeData[3] = 0.0f;
            vkCmdPushConstants(
                commandBuffer,
                m_pipelineLayout,
                VK_SHADER_STAGE_VERTEX_BIT,
                0,
                sizeof(ChunkPushConstants),
                &chunkPushConstants
            );
            drawChunkIndirect(m_debugDrawCallsPrepass);
        }
    }

    if (m_pipeNormalDepthPipeline != VK_NULL_HANDLE) {
        auto drawNormalDepthInstances = [&](
                                            BufferHandle vertexHandle,
                                            BufferHandle indexHandle,
                                            uint32_t indexCount,
                                            uint32_t instanceCount,
                                        const std::optional<FrameArenaSlice>& instanceSlice
                                        ) {
            if (instanceCount == 0 || !instanceSlice.has_value() || indexCount == 0) {
                return;
            }
            const VkBuffer vertexBuffer = m_bufferAllocator.getBuffer(vertexHandle);
            const VkBuffer indexBuffer = m_bufferAllocator.getBuffer(indexHandle);
            const VkBuffer instanceBuffer = m_bufferAllocator.getBuffer(instanceSlice->buffer);
            if (vertexBuffer == VK_NULL_HANDLE || indexBuffer == VK_NULL_HANDLE || instanceBuffer == VK_NULL_HANDLE) {
                return;
            }
            const VkBuffer vertexBuffers[2] = {vertexBuffer, instanceBuffer};
            const VkDeviceSize vertexOffsets[2] = {0, instanceSlice->offset};
            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeNormalDepthPipeline);
            vkCmdBindDescriptorSets(
                commandBuffer,
                VK_PIPELINE_BIND_POINT_GRAPHICS,
                m_pipelineLayout,
                0,
                boundDescriptorSetCount,
                boundDescriptorSets.data(),
            1,
            &mvpDynamicOffset
        );
            vkCmdBindVertexBuffers(commandBuffer, 0, 2, vertexBuffers, vertexOffsets);
            vkCmdBindIndexBuffer(commandBuffer, indexBuffer, 0, VK_INDEX_TYPE_UINT32);
            countDrawCalls(m_debugDrawCallsPrepass, 1);
            vkCmdDrawIndexed(commandBuffer, indexCount, instanceCount, 0, 0, 0);
        };
        drawNormalDepthInstances(
            m_pipeVertexBufferHandle,
            m_pipeIndexBufferHandle,
            m_pipeIndexCount,
            pipeInstanceCount,
            pipeInstanceSliceOpt
        );
        drawNormalDepthInstances(
            m_transportVertexBufferHandle,
            m_transportIndexBufferHandle,
            m_transportIndexCount,
            transportInstanceCount,
            transportInstanceSliceOpt
        );
    }
    vkCmdEndRendering(commandBuffer);
    endDebugLabel(commandBuffer);
    writeGpuTimestampBottom(kGpuTimestampQueryPrepassEnd);

    transitionImageLayout(
        commandBuffer,
        m_normalDepthImages[aoFrameIndex],
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
        VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT
    );

    VkClearValue ssaoClearValue{};
    ssaoClearValue.color.float32[0] = 1.0f;
    ssaoClearValue.color.float32[1] = 1.0f;
    ssaoClearValue.color.float32[2] = 1.0f;
    ssaoClearValue.color.float32[3] = 1.0f;

    VkRenderingAttachmentInfo ssaoRawAttachment{};
    ssaoRawAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    ssaoRawAttachment.imageView = m_ssaoRawImageViews[aoFrameIndex];
    ssaoRawAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    ssaoRawAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    ssaoRawAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    ssaoRawAttachment.clearValue = ssaoClearValue;

    VkRenderingInfo ssaoRenderingInfo{};
    ssaoRenderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    ssaoRenderingInfo.renderArea.offset = {0, 0};
    ssaoRenderingInfo.renderArea.extent = aoExtent;
    ssaoRenderingInfo.layerCount = 1;
    ssaoRenderingInfo.colorAttachmentCount = 1;
    ssaoRenderingInfo.pColorAttachments = &ssaoRawAttachment;

    writeGpuTimestampTop(kGpuTimestampQuerySsaoStart);
    beginDebugLabel(commandBuffer, "Pass: SSAO", 0.20f, 0.36f, 0.26f, 1.0f);
    vkCmdBeginRendering(commandBuffer, &ssaoRenderingInfo);
    vkCmdSetViewport(commandBuffer, 0, 1, &aoViewport);
    vkCmdSetScissor(commandBuffer, 0, 1, &aoScissor);
    if (m_ssaoPipeline != VK_NULL_HANDLE) {
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_ssaoPipeline);
        vkCmdBindDescriptorSets(
            commandBuffer,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            m_pipelineLayout,
            0,
            boundDescriptorSetCount,
            boundDescriptorSets.data(),
            1,
            &mvpDynamicOffset
        );
        countDrawCalls(m_debugDrawCallsPrepass, 1);
        vkCmdDraw(commandBuffer, 3, 1, 0, 0);
    }
    vkCmdEndRendering(commandBuffer);
    endDebugLabel(commandBuffer);
    writeGpuTimestampBottom(kGpuTimestampQuerySsaoEnd);

    transitionImageLayout(
        commandBuffer,
        m_ssaoRawImages[aoFrameIndex],
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
        VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT
    );

    VkRenderingAttachmentInfo ssaoBlurAttachment{};
    ssaoBlurAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    ssaoBlurAttachment.imageView = m_ssaoBlurImageViews[aoFrameIndex];
    ssaoBlurAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    ssaoBlurAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    ssaoBlurAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    ssaoBlurAttachment.clearValue = ssaoClearValue;

    VkRenderingInfo ssaoBlurRenderingInfo{};
    ssaoBlurRenderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    ssaoBlurRenderingInfo.renderArea.offset = {0, 0};
    ssaoBlurRenderingInfo.renderArea.extent = aoExtent;
    ssaoBlurRenderingInfo.layerCount = 1;
    ssaoBlurRenderingInfo.colorAttachmentCount = 1;
    ssaoBlurRenderingInfo.pColorAttachments = &ssaoBlurAttachment;

    writeGpuTimestampTop(kGpuTimestampQuerySsaoBlurStart);
    beginDebugLabel(commandBuffer, "Pass: SSAO Blur", 0.22f, 0.40f, 0.30f, 1.0f);
    vkCmdBeginRendering(commandBuffer, &ssaoBlurRenderingInfo);
    vkCmdSetViewport(commandBuffer, 0, 1, &aoViewport);
    vkCmdSetScissor(commandBuffer, 0, 1, &aoScissor);
    if (m_ssaoBlurPipeline != VK_NULL_HANDLE) {
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_ssaoBlurPipeline);
        vkCmdBindDescriptorSets(
            commandBuffer,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            m_pipelineLayout,
            0,
            boundDescriptorSetCount,
            boundDescriptorSets.data(),
            1,
            &mvpDynamicOffset
        );
        countDrawCalls(m_debugDrawCallsPrepass, 1);
        vkCmdDraw(commandBuffer, 3, 1, 0, 0);
    }
    vkCmdEndRendering(commandBuffer);
    endDebugLabel(commandBuffer);
    writeGpuTimestampBottom(kGpuTimestampQuerySsaoBlurEnd);

    transitionImageLayout(
        commandBuffer,
        m_ssaoBlurImages[aoFrameIndex],
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
        VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT
    );

    m_normalDepthImageInitialized[aoFrameIndex] = true;
    m_aoDepthImageInitialized[imageIndex] = true;
    m_ssaoRawImageInitialized[aoFrameIndex] = true;
    m_ssaoBlurImageInitialized[aoFrameIndex] = true;

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(m_swapchainExtent.width);
    viewport.height = static_cast<float>(m_swapchainExtent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = m_swapchainExtent;

    if (!m_msaaColorImageInitialized[imageIndex]) {
        transitionImageLayout(
            commandBuffer,
            m_msaaColorImages[imageIndex],
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_PIPELINE_STAGE_2_NONE,
            VK_ACCESS_2_NONE,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT
        );
    }
    const bool hdrResolveInitialized = m_hdrResolveImageInitialized[aoFrameIndex];
    transitionImageLayout(
        commandBuffer,
        m_hdrResolveImages[aoFrameIndex],
        hdrResolveInitialized ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        hdrResolveInitialized ? VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT : VK_PIPELINE_STAGE_2_NONE,
        hdrResolveInitialized ? VK_ACCESS_2_SHADER_SAMPLED_READ_BIT : VK_ACCESS_2_NONE,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT
    );
    transitionImageLayout(
        commandBuffer,
        m_depthImages[imageIndex],
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
        VK_PIPELINE_STAGE_2_NONE,
        VK_ACCESS_2_NONE,
        VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
        VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
        VK_IMAGE_ASPECT_DEPTH_BIT
    );

    VkClearValue clearValue{};
    clearValue.color.float32[0] = 0.06f;
    clearValue.color.float32[1] = 0.08f;
    clearValue.color.float32[2] = 0.12f;
    clearValue.color.float32[3] = 1.0f;

    VkRenderingAttachmentInfo colorAttachment{};
    colorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    colorAttachment.imageView = m_msaaColorImageViews[imageIndex];
    colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.clearValue = clearValue;
    colorAttachment.resolveMode = VK_RESOLVE_MODE_AVERAGE_BIT;
    colorAttachment.resolveImageView = m_hdrResolveImageViews[aoFrameIndex];
    colorAttachment.resolveImageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkClearValue depthClearValue{};
    depthClearValue.depthStencil.depth = 0.0f;
    depthClearValue.depthStencil.stencil = 0;

    VkRenderingAttachmentInfo depthAttachment{};
    depthAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    depthAttachment.imageView = m_depthImageViews[imageIndex];
    depthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.clearValue = depthClearValue;

    VkRenderingInfo renderingInfo{};
    renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    renderingInfo.renderArea.offset = {0, 0};
    renderingInfo.renderArea.extent = m_swapchainExtent;
    renderingInfo.layerCount = 1;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments = &colorAttachment;
    renderingInfo.pDepthAttachment = &depthAttachment;

    writeGpuTimestampTop(kGpuTimestampQueryMainStart);
    beginDebugLabel(commandBuffer, "Pass: Main Scene", 0.20f, 0.20f, 0.45f, 1.0f);
    vkCmdBeginRendering(commandBuffer, &renderingInfo);
    vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
    vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

    if (m_skyboxPipeline != VK_NULL_HANDLE) {
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_skyboxPipeline);
        vkCmdBindDescriptorSets(
            commandBuffer,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            m_pipelineLayout,
            0,
            boundDescriptorSetCount,
            boundDescriptorSets.data(),
            1,
            &mvpDynamicOffset
        );
        countDrawCalls(m_debugDrawCallsMain, 1);
        vkCmdDraw(commandBuffer, 3, 1, 0, 0);
    }

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline);
    vkCmdBindDescriptorSets(
        commandBuffer,
        VK_PIPELINE_BIND_POINT_GRAPHICS,
        m_pipelineLayout,
        0,
        boundDescriptorSetCount,
        boundDescriptorSets.data(),
            1,
            &mvpDynamicOffset
        );
    if (canDrawChunksIndirect) {
        const VkBuffer voxelVertexBuffers[2] = {chunkVertexBuffer, chunkInstanceBuffer};
        const VkDeviceSize voxelVertexOffsets[2] = {0, chunkInstanceSliceOpt->offset};
        vkCmdBindVertexBuffers(commandBuffer, 0, 2, voxelVertexBuffers, voxelVertexOffsets);
        vkCmdBindIndexBuffer(commandBuffer, chunkIndexBuffer, 0, VK_INDEX_TYPE_UINT32);

        ChunkPushConstants chunkPushConstants{};
        chunkPushConstants.chunkOffset[0] = 0.0f;
        chunkPushConstants.chunkOffset[1] = 0.0f;
        chunkPushConstants.chunkOffset[2] = 0.0f;
        chunkPushConstants.chunkOffset[3] = 0.0f;
        chunkPushConstants.cascadeData[0] = 0.0f;
        chunkPushConstants.cascadeData[1] = 0.0f;
        chunkPushConstants.cascadeData[2] = 0.0f;
        chunkPushConstants.cascadeData[3] = 0.0f;
        vkCmdPushConstants(
            commandBuffer,
            m_pipelineLayout,
            VK_SHADER_STAGE_VERTEX_BIT,
            0,
            sizeof(ChunkPushConstants),
            &chunkPushConstants
        );
        drawChunkIndirect(m_debugDrawCallsMain);
    }

    if (m_pipePipeline != VK_NULL_HANDLE) {
        auto drawLitInstances = [&](
                                    BufferHandle vertexHandle,
                                    BufferHandle indexHandle,
                                    uint32_t indexCount,
                                    uint32_t instanceCount,
                                const std::optional<FrameArenaSlice>& instanceSlice
                                ) {
            if (instanceCount == 0 || !instanceSlice.has_value() || indexCount == 0) {
                return;
            }
            const VkBuffer vertexBuffer = m_bufferAllocator.getBuffer(vertexHandle);
            const VkBuffer indexBuffer = m_bufferAllocator.getBuffer(indexHandle);
            const VkBuffer instanceBuffer = m_bufferAllocator.getBuffer(instanceSlice->buffer);
            if (vertexBuffer == VK_NULL_HANDLE || indexBuffer == VK_NULL_HANDLE || instanceBuffer == VK_NULL_HANDLE) {
                return;
            }
            const VkBuffer vertexBuffers[2] = {vertexBuffer, instanceBuffer};
            const VkDeviceSize vertexOffsets[2] = {0, instanceSlice->offset};

            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipePipeline);
            vkCmdBindDescriptorSets(
                commandBuffer,
                VK_PIPELINE_BIND_POINT_GRAPHICS,
                m_pipelineLayout,
                0,
                boundDescriptorSetCount,
                boundDescriptorSets.data(),
            1,
            &mvpDynamicOffset
        );
            vkCmdBindVertexBuffers(commandBuffer, 0, 2, vertexBuffers, vertexOffsets);
            vkCmdBindIndexBuffer(commandBuffer, indexBuffer, 0, VK_INDEX_TYPE_UINT32);
            countDrawCalls(m_debugDrawCallsMain, 1);
            vkCmdDrawIndexed(commandBuffer, indexCount, instanceCount, 0, 0, 0);
        };
        drawLitInstances(
            m_pipeVertexBufferHandle,
            m_pipeIndexBufferHandle,
            m_pipeIndexCount,
            pipeInstanceCount,
            pipeInstanceSliceOpt
        );
        drawLitInstances(
            m_transportVertexBufferHandle,
            m_transportIndexBufferHandle,
            m_transportIndexCount,
            transportInstanceCount,
            transportInstanceSliceOpt
        );
    }

    if (m_grassBillboardPipeline != VK_NULL_HANDLE &&
        m_grassBillboardIndexCount > 0 &&
        m_grassBillboardInstanceCount > 0 &&
        m_grassBillboardInstanceBufferHandle != kInvalidBufferHandle) {
        const VkBuffer grassVertexBuffer = m_bufferAllocator.getBuffer(m_grassBillboardVertexBufferHandle);
        const VkBuffer grassIndexBuffer = m_bufferAllocator.getBuffer(m_grassBillboardIndexBufferHandle);
        const VkBuffer grassInstanceBuffer = m_bufferAllocator.getBuffer(m_grassBillboardInstanceBufferHandle);
        if (grassVertexBuffer != VK_NULL_HANDLE &&
            grassIndexBuffer != VK_NULL_HANDLE &&
            grassInstanceBuffer != VK_NULL_HANDLE) {
            const VkBuffer vertexBuffers[2] = {grassVertexBuffer, grassInstanceBuffer};
            const VkDeviceSize vertexOffsets[2] = {0, 0};
            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_grassBillboardPipeline);
            vkCmdBindDescriptorSets(
                commandBuffer,
                VK_PIPELINE_BIND_POINT_GRAPHICS,
                m_pipelineLayout,
                0,
                boundDescriptorSetCount,
                boundDescriptorSets.data(),
                1,
                &mvpDynamicOffset
            );
            vkCmdBindVertexBuffers(commandBuffer, 0, 2, vertexBuffers, vertexOffsets);
            vkCmdBindIndexBuffer(commandBuffer, grassIndexBuffer, 0, VK_INDEX_TYPE_UINT32);
            countDrawCalls(m_debugDrawCallsMain, 1);
            vkCmdDrawIndexed(commandBuffer, m_grassBillboardIndexCount, m_grassBillboardInstanceCount, 0, 0, 0);
        }
    }

    const VkPipeline activePreviewPipeline =
        (preview.mode == VoxelPreview::Mode::Remove) ? m_previewRemovePipeline : m_previewAddPipeline;
    const bool drawCubePreview = !preview.pipeStyle && preview.visible && activePreviewPipeline != VK_NULL_HANDLE;
    const bool drawFacePreview =
        !preview.pipeStyle && preview.faceVisible && preview.brushSize == 1 && m_previewRemovePipeline != VK_NULL_HANDLE;

    if (preview.pipeStyle && preview.visible && m_pipePipeline != VK_NULL_HANDLE) {
        PipeInstance previewInstance{};
        previewInstance.originLength[0] = static_cast<float>(preview.x);
        previewInstance.originLength[1] = static_cast<float>(preview.y);
        previewInstance.originLength[2] = static_cast<float>(preview.z);
        previewInstance.originLength[3] = 1.0f;
        math::Vector3 previewAxis = math::normalize(math::Vector3{preview.pipeAxisX, preview.pipeAxisY, preview.pipeAxisZ});
        if (math::lengthSquared(previewAxis) <= 0.0001f) {
            previewAxis = math::Vector3{0.0f, 1.0f, 0.0f};
        }
        previewInstance.axisRadius[0] = previewAxis.x;
        previewInstance.axisRadius[1] = previewAxis.y;
        previewInstance.axisRadius[2] = previewAxis.z;
        previewInstance.axisRadius[3] = std::clamp(preview.pipeRadius, 0.02f, 0.5f);
        if (preview.mode == VoxelPreview::Mode::Remove) {
            previewInstance.tint[0] = 1.0f;
            previewInstance.tint[1] = 0.32f;
            previewInstance.tint[2] = 0.26f;
        } else {
            previewInstance.tint[0] = 0.30f;
            previewInstance.tint[1] = 0.95f;
            previewInstance.tint[2] = 1.0f;
        }
        previewInstance.tint[3] = std::clamp(preview.pipeStyleId, 0.0f, 2.0f);
        previewInstance.extensions[0] = 0.0f;
        previewInstance.extensions[1] = 0.0f;
        previewInstance.extensions[2] = 1.0f;
        previewInstance.extensions[3] = 1.0f;
        if (preview.pipeStyleId > 0.5f && preview.pipeStyleId < 1.5f) {
            previewInstance.extensions[2] = 2.0f;
            previewInstance.extensions[3] = 0.25f;
        }
        if (preview.pipeStyleId > 1.5f) {
            previewInstance.extensions[2] = 2.0f;
            previewInstance.extensions[3] = 0.25f;
        }

        const std::optional<FrameArenaSlice> previewInstanceSlice =
            m_frameArena.allocateUpload(
                sizeof(PipeInstance),
                static_cast<VkDeviceSize>(alignof(PipeInstance)),
                FrameArenaUploadKind::PreviewData
            );
        if (previewInstanceSlice.has_value() && previewInstanceSlice->mapped != nullptr) {
            std::memcpy(previewInstanceSlice->mapped, &previewInstance, sizeof(PipeInstance));
            const bool previewUsesPipeMesh = preview.pipeStyleId < 0.5f;
            const BufferHandle previewVertexHandle =
                previewUsesPipeMesh ? m_pipeVertexBufferHandle : m_transportVertexBufferHandle;
            const BufferHandle previewIndexHandle =
                previewUsesPipeMesh ? m_pipeIndexBufferHandle : m_transportIndexBufferHandle;
            const uint32_t previewIndexCount = previewUsesPipeMesh ? m_pipeIndexCount : m_transportIndexCount;
            if (previewIndexCount == 0) {
                // No mesh data allocated for this preview style.
            }
            const VkBuffer pipeVertexBuffer = m_bufferAllocator.getBuffer(previewVertexHandle);
            const VkBuffer pipeIndexBuffer = m_bufferAllocator.getBuffer(previewIndexHandle);
            const VkBuffer pipeInstanceBuffer = m_bufferAllocator.getBuffer(previewInstanceSlice->buffer);
            if (pipeVertexBuffer != VK_NULL_HANDLE &&
                pipeIndexBuffer != VK_NULL_HANDLE &&
                pipeInstanceBuffer != VK_NULL_HANDLE &&
                previewIndexCount > 0) {
                const VkBuffer vertexBuffers[2] = {pipeVertexBuffer, pipeInstanceBuffer};
                const VkDeviceSize vertexOffsets[2] = {0, previewInstanceSlice->offset};
                vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipePipeline);
                vkCmdBindDescriptorSets(
                    commandBuffer,
                    VK_PIPELINE_BIND_POINT_GRAPHICS,
                    m_pipelineLayout,
                    0,
                    boundDescriptorSetCount,
                    boundDescriptorSets.data(),
            1,
            &mvpDynamicOffset
        );
                vkCmdBindVertexBuffers(commandBuffer, 0, 2, vertexBuffers, vertexOffsets);
                vkCmdBindIndexBuffer(commandBuffer, pipeIndexBuffer, 0, VK_INDEX_TYPE_UINT32);
                countDrawCalls(m_debugDrawCallsMain, 1);
                vkCmdDrawIndexed(commandBuffer, previewIndexCount, 1, 0, 0, 0);
            }
        }
    }

    if (drawCubePreview || drawFacePreview) {
        constexpr uint32_t kPreviewCubeIndexCount = 36u;
        constexpr uint32_t kPreviewFaceIndexCount = 6u;
        constexpr uint32_t kAddCubeFirstIndex = 0u;
        constexpr uint32_t kRemoveCubeFirstIndex = 36u;
        constexpr uint32_t kFaceFirstIndexBase = kRemoveCubeFirstIndex;
        constexpr float kChunkCoordinateScale = 1.0f;

        const VkBuffer previewVertexBuffer = m_bufferAllocator.getBuffer(m_previewVertexBufferHandle);
        const VkBuffer previewIndexBuffer = m_bufferAllocator.getBuffer(m_previewIndexBufferHandle);
        if (previewVertexBuffer != VK_NULL_HANDLE &&
            previewIndexBuffer != VK_NULL_HANDLE &&
            chunkInstanceSliceOpt.has_value() &&
            chunkInstanceBuffer != VK_NULL_HANDLE) {
            const VkBuffer previewVertexBuffers[2] = {previewVertexBuffer, chunkInstanceBuffer};
            const VkDeviceSize previewVertexOffsets[2] = {0, chunkInstanceSliceOpt->offset};
            vkCmdBindVertexBuffers(commandBuffer, 0, 2, previewVertexBuffers, previewVertexOffsets);
            vkCmdBindIndexBuffer(commandBuffer, previewIndexBuffer, 0, VK_INDEX_TYPE_UINT32);

            auto drawPreviewRange = [&](VkPipeline pipeline, uint32_t indexCount, uint32_t firstIndex, int x, int y, int z) {
                if (pipeline == VK_NULL_HANDLE || indexCount == 0) {
                    return;
                }
                ChunkPushConstants previewChunkPushConstants{};
                previewChunkPushConstants.chunkOffset[0] = static_cast<float>(x) * kChunkCoordinateScale;
                previewChunkPushConstants.chunkOffset[1] = static_cast<float>(y) * kChunkCoordinateScale;
                previewChunkPushConstants.chunkOffset[2] = static_cast<float>(z) * kChunkCoordinateScale;
                previewChunkPushConstants.chunkOffset[3] = 0.0f;
                previewChunkPushConstants.cascadeData[0] = 0.0f;
                previewChunkPushConstants.cascadeData[1] = 0.0f;
                previewChunkPushConstants.cascadeData[2] = 0.0f;
                previewChunkPushConstants.cascadeData[3] = 0.0f;

                vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
                vkCmdBindDescriptorSets(
                    commandBuffer,
                    VK_PIPELINE_BIND_POINT_GRAPHICS,
                    m_pipelineLayout,
                    0,
                    boundDescriptorSetCount,
                    boundDescriptorSets.data(),
            1,
            &mvpDynamicOffset
        );
                vkCmdPushConstants(
                    commandBuffer,
                    m_pipelineLayout,
                    VK_SHADER_STAGE_VERTEX_BIT,
                    0,
                    sizeof(ChunkPushConstants),
                    &previewChunkPushConstants
                );
                countDrawCalls(m_debugDrawCallsMain, 1);
                vkCmdDrawIndexed(commandBuffer, indexCount, 1, firstIndex, 0, 0);
            };

            if (drawCubePreview) {
                const uint32_t cubeFirstIndex =
                    (preview.mode == VoxelPreview::Mode::Add) ? kAddCubeFirstIndex : kRemoveCubeFirstIndex;
                const int brushSize = std::max(preview.brushSize, 1);
                for (int localY = 0; localY < brushSize; ++localY) {
                    for (int localZ = 0; localZ < brushSize; ++localZ) {
                        for (int localX = 0; localX < brushSize; ++localX) {
                            drawPreviewRange(
                                activePreviewPipeline,
                                kPreviewCubeIndexCount,
                                cubeFirstIndex,
                                preview.x + localX,
                                preview.y + localY,
                                preview.z + localZ
                            );
                        }
                    }
                }
            }

            if (drawFacePreview) {
                const uint32_t faceFirstIndex = kFaceFirstIndexBase + (std::min(preview.faceId, 5u) * kPreviewFaceIndexCount);
                drawPreviewRange(m_previewRemovePipeline, kPreviewFaceIndexCount, faceFirstIndex, preview.faceX, preview.faceY, preview.faceZ);
            }
        }
    }

    vkCmdEndRendering(commandBuffer);
    endDebugLabel(commandBuffer);
    writeGpuTimestampBottom(kGpuTimestampQueryMainEnd);

    transitionImageLayout(
        commandBuffer,
        m_hdrResolveImages[aoFrameIndex],
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
        VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT
    );

    transitionImageLayout(
        commandBuffer,
        m_swapchainImages[imageIndex],
        m_swapchainImageInitialized[imageIndex] ? VK_IMAGE_LAYOUT_PRESENT_SRC_KHR : VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_PIPELINE_STAGE_2_NONE,
        VK_ACCESS_2_NONE,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT
    );

    VkRenderingAttachmentInfo toneMapColorAttachment{};
    toneMapColorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    toneMapColorAttachment.imageView = m_swapchainImageViews[imageIndex];
    toneMapColorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    toneMapColorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    toneMapColorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

    VkRenderingInfo toneMapRenderingInfo{};
    toneMapRenderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    toneMapRenderingInfo.renderArea.offset = {0, 0};
    toneMapRenderingInfo.renderArea.extent = m_swapchainExtent;
    toneMapRenderingInfo.layerCount = 1;
    toneMapRenderingInfo.colorAttachmentCount = 1;
    toneMapRenderingInfo.pColorAttachments = &toneMapColorAttachment;

    writeGpuTimestampTop(kGpuTimestampQueryPostStart);
    beginDebugLabel(commandBuffer, "Pass: Tonemap + UI", 0.24f, 0.24f, 0.24f, 1.0f);
    vkCmdBeginRendering(commandBuffer, &toneMapRenderingInfo);
    vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
    vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

    if (m_tonemapPipeline != VK_NULL_HANDLE) {
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_tonemapPipeline);
        vkCmdBindDescriptorSets(
            commandBuffer,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            m_pipelineLayout,
            0,
            boundDescriptorSetCount,
            boundDescriptorSets.data(),
            1,
            &mvpDynamicOffset
        );
        countDrawCalls(m_debugDrawCallsPost, 1);
        vkCmdDraw(commandBuffer, 3, 1, 0, 0);
    }
#if defined(VOXEL_HAS_IMGUI)
    if (m_imguiInitialized) {
        ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), commandBuffer);
    }
#endif

    vkCmdEndRendering(commandBuffer);
    endDebugLabel(commandBuffer);
    writeGpuTimestampBottom(kGpuTimestampQueryPostEnd);

    transitionImageLayout(
        commandBuffer,
        m_swapchainImages[imageIndex],
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        VK_PIPELINE_STAGE_2_NONE,
        VK_ACCESS_2_NONE,
        VK_IMAGE_ASPECT_COLOR_BIT
    );
    writeGpuTimestampBottom(kGpuTimestampQueryFrameEnd);

    endDebugLabel(commandBuffer);
    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
        VOX_LOGE("render") << "vkEndCommandBuffer failed\n";
        return;
    }

    std::array<VkSemaphore, 2> waitSemaphores{};
    std::array<VkPipelineStageFlags, 2> waitStages{};
    std::array<uint64_t, 2> waitSemaphoreValues{};
    uint32_t waitSemaphoreCount = 0;

    waitSemaphores[waitSemaphoreCount] = frame.imageAvailable;
    waitStages[waitSemaphoreCount] = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    waitSemaphoreValues[waitSemaphoreCount] = 0;
    ++waitSemaphoreCount;

    if (m_pendingTransferTimelineValue > 0) {
        waitSemaphores[waitSemaphoreCount] = m_renderTimelineSemaphore;
        waitStages[waitSemaphoreCount] = VK_PIPELINE_STAGE_VERTEX_INPUT_BIT;
        waitSemaphoreValues[waitSemaphoreCount] = m_pendingTransferTimelineValue;
        ++waitSemaphoreCount;
    }

    const uint64_t signalTimelineValue = m_nextTimelineValue++;
    std::array<VkSemaphore, 2> signalSemaphores = {
        renderFinishedSemaphore,
        m_renderTimelineSemaphore
    };
    std::array<uint64_t, 2> signalSemaphoreValues = {
        0,
        signalTimelineValue
    };
    VkTimelineSemaphoreSubmitInfo timelineSubmitInfo{};
    timelineSubmitInfo.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    timelineSubmitInfo.waitSemaphoreValueCount = waitSemaphoreCount;
    timelineSubmitInfo.pWaitSemaphoreValues = waitSemaphoreValues.data();
    timelineSubmitInfo.signalSemaphoreValueCount = static_cast<uint32_t>(signalSemaphoreValues.size());
    timelineSubmitInfo.pSignalSemaphoreValues = signalSemaphoreValues.data();

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.pNext = &timelineSubmitInfo;
    submitInfo.waitSemaphoreCount = waitSemaphoreCount;
    submitInfo.pWaitSemaphores = waitSemaphores.data();
    submitInfo.pWaitDstStageMask = waitStages.data();
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    submitInfo.signalSemaphoreCount = static_cast<uint32_t>(signalSemaphores.size());
    submitInfo.pSignalSemaphores = signalSemaphores.data();

    if (vkQueueSubmit(m_graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS) {
        VOX_LOGE("render") << "vkQueueSubmit failed\n";
        return;
    }
    m_frameTimelineValues[m_currentFrame] = signalTimelineValue;
    m_swapchainImageTimelineValues[imageIndex] = signalTimelineValue;
    m_lastGraphicsTimelineValue = signalTimelineValue;

    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = &renderFinishedSemaphore;
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = &m_swapchain;
    presentInfo.pImageIndices = &imageIndex;

    const VkResult presentResult = vkQueuePresentKHR(m_graphicsQueue, &presentInfo);
    m_shadowDepthInitialized = true;
    m_swapchainImageInitialized[imageIndex] = true;
    m_msaaColorImageInitialized[imageIndex] = true;
    m_hdrResolveImageInitialized[aoFrameIndex] = true;

    if (
        acquireResult == VK_SUBOPTIMAL_KHR ||
        presentResult == VK_ERROR_OUT_OF_DATE_KHR ||
        presentResult == VK_SUBOPTIMAL_KHR
    ) {
        VOX_LOGI("render") << "swapchain needs recreate after present\n";
        recreateSwapchain();
    } else if (presentResult != VK_SUCCESS) {
        logVkFailure("vkQueuePresentKHR", presentResult);
    }

    const FrameArenaStats& frameArenaStats = m_frameArena.activeStats();
    m_debugFrameArenaUploadBytes = static_cast<std::uint64_t>(frameArenaStats.uploadBytesAllocated);
    m_debugFrameArenaUploadAllocs = frameArenaStats.uploadAllocationCount;
    m_debugFrameArenaTransientBufferBytes = static_cast<std::uint64_t>(frameArenaStats.transientBufferBytes);
    m_debugFrameArenaTransientBufferCount = frameArenaStats.transientBufferCount;
    m_debugFrameArenaTransientImageBytes = frameArenaStats.transientImageBytes;
    m_debugFrameArenaTransientImageCount = frameArenaStats.transientImageCount;
    m_debugFrameArenaAliasReuses = frameArenaStats.transientImageAliasReuses;
    const FrameArenaResidentStats& frameArenaResidentStats = m_frameArena.residentStats();
    m_debugFrameArenaResidentBufferBytes = frameArenaResidentStats.bufferBytes;
    m_debugFrameArenaResidentBufferCount = frameArenaResidentStats.bufferCount;
    m_debugFrameArenaResidentImageBytes = frameArenaResidentStats.imageBytes;
    m_debugFrameArenaResidentImageCount = frameArenaResidentStats.imageCount;
    m_debugFrameArenaResidentAliasReuses = frameArenaResidentStats.imageAliasReuses;
    m_frameArena.collectAliasedImageDebugInfo(m_debugAliasedImages);

    m_currentFrame = (m_currentFrame + 1) % kMaxFramesInFlight;
}

bool Renderer::recreateSwapchain() {
    VOX_LOGI("render") << "recreateSwapchain begin\n";
    int width = 0;
    int height = 0;
    glfwGetFramebufferSize(m_window, &width, &height);
    while ((width == 0 || height == 0) && glfwWindowShouldClose(m_window) == GLFW_FALSE) {
        // Keep swapchain recreation responsive when minimized without hard-blocking shutdown.
        glfwWaitEventsTimeout(0.05);
        glfwGetFramebufferSize(m_window, &width, &height);
    }
    if (glfwWindowShouldClose(m_window) == GLFW_TRUE) {
        return false;
    }

    vkDeviceWaitIdle(m_device);

    destroyPipeline();
    destroySwapchain();

    if (!createSwapchain()) {
        VOX_LOGE("render") << "recreateSwapchain failed: createSwapchain\n";
        return false;
    }
    if (!createGraphicsPipeline()) {
        VOX_LOGE("render") << "recreateSwapchain failed: createGraphicsPipeline\n";
        return false;
    }
    if (!createPipePipeline()) {
        VOX_LOGE("render") << "recreateSwapchain failed: createPipePipeline\n";
        return false;
    }
    if (!createAoPipelines()) {
        VOX_LOGE("render") << "recreateSwapchain failed: createAoPipelines\n";
        return false;
    }
#if defined(VOXEL_HAS_IMGUI)
    if (m_imguiInitialized) {
        ImGui_ImplVulkan_SetMinImageCount(std::max<uint32_t>(2u, static_cast<uint32_t>(m_swapchainImages.size())));
    }
#endif
    VOX_LOGI("render") << "recreateSwapchain complete\n";
    return true;
}

void Renderer::destroyHdrResolveTargets() {
    if (m_hdrResolveSampler != VK_NULL_HANDLE) {
        vkDestroySampler(m_device, m_hdrResolveSampler, nullptr);
        m_hdrResolveSampler = VK_NULL_HANDLE;
    }

    for (TransientImageHandle handle : m_hdrResolveTransientHandles) {
        if (handle != kInvalidTransientImageHandle) {
            m_frameArena.destroyTransientImage(handle);
        }
    }
    m_hdrResolveImageViews.clear();
    m_hdrResolveImages.clear();
    m_hdrResolveImageMemories.clear();
    m_hdrResolveTransientHandles.clear();
    m_hdrResolveImageInitialized.clear();
}

void Renderer::destroyMsaaColorTargets() {
    for (VkImageView imageView : m_msaaColorImageViews) {
        if (imageView != VK_NULL_HANDLE) {
            vkDestroyImageView(m_device, imageView, nullptr);
        }
    }
    m_msaaColorImageViews.clear();

    for (size_t i = 0; i < m_msaaColorImages.size(); ++i) {
        const VkImage image = m_msaaColorImages[i];
        if (image == VK_NULL_HANDLE) {
            continue;
        }
#if defined(VOXEL_HAS_VMA)
        if (m_vmaAllocator != VK_NULL_HANDLE &&
            i < m_msaaColorImageAllocations.size() &&
            m_msaaColorImageAllocations[i] != VK_NULL_HANDLE) {
            vmaDestroyImage(m_vmaAllocator, image, m_msaaColorImageAllocations[i]);
            m_msaaColorImageAllocations[i] = VK_NULL_HANDLE;
        } else {
            vkDestroyImage(m_device, image, nullptr);
        }
#else
        vkDestroyImage(m_device, image, nullptr);
#endif
    }
    m_msaaColorImages.clear();

    for (VkDeviceMemory memory : m_msaaColorImageMemories) {
        if (memory != VK_NULL_HANDLE) {
            vkFreeMemory(m_device, memory, nullptr);
        }
    }
    m_msaaColorImageMemories.clear();
#if defined(VOXEL_HAS_VMA)
    m_msaaColorImageAllocations.clear();
#endif
    m_msaaColorImageInitialized.clear();
}

void Renderer::destroyDepthTargets() {
    for (VkImageView imageView : m_depthImageViews) {
        if (imageView != VK_NULL_HANDLE) {
            vkDestroyImageView(m_device, imageView, nullptr);
        }
    }
    m_depthImageViews.clear();

    for (size_t i = 0; i < m_depthImages.size(); ++i) {
        const VkImage image = m_depthImages[i];
        if (image == VK_NULL_HANDLE) {
            continue;
        }
#if defined(VOXEL_HAS_VMA)
        if (m_vmaAllocator != VK_NULL_HANDLE &&
            i < m_depthImageAllocations.size() &&
            m_depthImageAllocations[i] != VK_NULL_HANDLE) {
            vmaDestroyImage(m_vmaAllocator, image, m_depthImageAllocations[i]);
            m_depthImageAllocations[i] = VK_NULL_HANDLE;
        } else {
            vkDestroyImage(m_device, image, nullptr);
        }
#else
        vkDestroyImage(m_device, image, nullptr);
#endif
    }
    m_depthImages.clear();

    for (VkDeviceMemory memory : m_depthImageMemories) {
        if (memory != VK_NULL_HANDLE) {
            vkFreeMemory(m_device, memory, nullptr);
        }
    }
    m_depthImageMemories.clear();
#if defined(VOXEL_HAS_VMA)
    m_depthImageAllocations.clear();
#endif
}

void Renderer::destroyAoTargets() {
    if (m_ssaoSampler != VK_NULL_HANDLE) {
        vkDestroySampler(m_device, m_ssaoSampler, nullptr);
        m_ssaoSampler = VK_NULL_HANDLE;
    }
    if (m_normalDepthSampler != VK_NULL_HANDLE) {
        vkDestroySampler(m_device, m_normalDepthSampler, nullptr);
        m_normalDepthSampler = VK_NULL_HANDLE;
    }

    for (TransientImageHandle handle : m_ssaoBlurTransientHandles) {
        if (handle != kInvalidTransientImageHandle) {
            m_frameArena.destroyTransientImage(handle);
        }
    }
    m_ssaoBlurImageViews.clear();
    m_ssaoBlurImages.clear();
    m_ssaoBlurImageMemories.clear();
    m_ssaoBlurTransientHandles.clear();
    m_ssaoBlurImageInitialized.clear();

    for (TransientImageHandle handle : m_ssaoRawTransientHandles) {
        if (handle != kInvalidTransientImageHandle) {
            m_frameArena.destroyTransientImage(handle);
        }
    }
    m_ssaoRawImageViews.clear();
    m_ssaoRawImages.clear();
    m_ssaoRawImageMemories.clear();
    m_ssaoRawTransientHandles.clear();
    m_ssaoRawImageInitialized.clear();

    for (TransientImageHandle handle : m_aoDepthTransientHandles) {
        if (handle != kInvalidTransientImageHandle) {
            m_frameArena.destroyTransientImage(handle);
        }
    }
    m_aoDepthImageViews.clear();
    m_aoDepthImages.clear();
    m_aoDepthImageMemories.clear();
    m_aoDepthTransientHandles.clear();
    m_aoDepthImageInitialized.clear();

    for (TransientImageHandle handle : m_normalDepthTransientHandles) {
        if (handle != kInvalidTransientImageHandle) {
            m_frameArena.destroyTransientImage(handle);
        }
    }
    m_normalDepthImageViews.clear();
    m_normalDepthImages.clear();
    m_normalDepthImageMemories.clear();
    m_normalDepthTransientHandles.clear();
    m_normalDepthImageInitialized.clear();
}

void Renderer::destroySwapchain() {
    destroyHdrResolveTargets();
    destroyMsaaColorTargets();
    destroyDepthTargets();
    destroyAoTargets();
    const uint32_t orphanedFrameArenaImages = m_frameArena.liveImageCount();
    if (orphanedFrameArenaImages > 0) {
        VOX_LOGI("render") << "destroySwapchain: cleaning up "
            << orphanedFrameArenaImages
            << " orphaned FrameArena image(s)\n";
        m_frameArena.destroyAllImages();
    }
    m_aoExtent = VkExtent2D{};

    for (VkSemaphore semaphore : m_renderFinishedSemaphores) {
        if (semaphore != VK_NULL_HANDLE) {
            vkDestroySemaphore(m_device, semaphore, nullptr);
        }
    }
    m_renderFinishedSemaphores.clear();

    for (VkImageView imageView : m_swapchainImageViews) {
        if (imageView != VK_NULL_HANDLE) {
            vkDestroyImageView(m_device, imageView, nullptr);
        }
    }
    m_swapchainImageViews.clear();
    m_swapchainImages.clear();
    m_swapchainImageInitialized.clear();
    m_swapchainImageTimelineValues.clear();

    if (m_swapchain != VK_NULL_HANDLE) {
        vkDestroySwapchainKHR(m_device, m_swapchain, nullptr);
        m_swapchain = VK_NULL_HANDLE;
    }
}

void Renderer::destroyFrameResources() {
    for (FrameResources& frame : m_frames) {
        if (frame.imageAvailable != VK_NULL_HANDLE) {
            vkDestroySemaphore(m_device, frame.imageAvailable, nullptr);
            frame.imageAvailable = VK_NULL_HANDLE;
        }
        if (frame.commandPool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(m_device, frame.commandPool, nullptr);
            frame.commandPool = VK_NULL_HANDLE;
        }
    }
}

void Renderer::destroyGpuTimestampResources() {
    for (VkQueryPool& queryPool : m_gpuTimestampQueryPools) {
        if (queryPool != VK_NULL_HANDLE) {
            vkDestroyQueryPool(m_device, queryPool, nullptr);
            queryPool = VK_NULL_HANDLE;
        }
    }
}

void Renderer::destroyTransferResources() {
    m_transferCommandBuffer = VK_NULL_HANDLE;
    if (m_transferCommandPool != VK_NULL_HANDLE) {
        vkDestroyCommandPool(m_device, m_transferCommandPool, nullptr);
        m_transferCommandPool = VK_NULL_HANDLE;
    }
}

void Renderer::destroyPreviewBuffers() {
    if (m_previewIndexBufferHandle != kInvalidBufferHandle) {
        m_bufferAllocator.destroyBuffer(m_previewIndexBufferHandle);
        m_previewIndexBufferHandle = kInvalidBufferHandle;
    }
    if (m_previewVertexBufferHandle != kInvalidBufferHandle) {
        m_bufferAllocator.destroyBuffer(m_previewVertexBufferHandle);
        m_previewVertexBufferHandle = kInvalidBufferHandle;
    }
    m_previewIndexCount = 0;
}

void Renderer::destroyPipeBuffers() {
    if (m_grassBillboardIndexBufferHandle != kInvalidBufferHandle) {
        m_bufferAllocator.destroyBuffer(m_grassBillboardIndexBufferHandle);
        m_grassBillboardIndexBufferHandle = kInvalidBufferHandle;
    }
    if (m_grassBillboardVertexBufferHandle != kInvalidBufferHandle) {
        m_bufferAllocator.destroyBuffer(m_grassBillboardVertexBufferHandle);
        m_grassBillboardVertexBufferHandle = kInvalidBufferHandle;
    }
    m_grassBillboardIndexCount = 0;

    if (m_transportIndexBufferHandle != kInvalidBufferHandle) {
        m_bufferAllocator.destroyBuffer(m_transportIndexBufferHandle);
        m_transportIndexBufferHandle = kInvalidBufferHandle;
    }
    if (m_transportVertexBufferHandle != kInvalidBufferHandle) {
        m_bufferAllocator.destroyBuffer(m_transportVertexBufferHandle);
        m_transportVertexBufferHandle = kInvalidBufferHandle;
    }
    m_transportIndexCount = 0;

    if (m_pipeIndexBufferHandle != kInvalidBufferHandle) {
        m_bufferAllocator.destroyBuffer(m_pipeIndexBufferHandle);
        m_pipeIndexBufferHandle = kInvalidBufferHandle;
    }
    if (m_pipeVertexBufferHandle != kInvalidBufferHandle) {
        m_bufferAllocator.destroyBuffer(m_pipeVertexBufferHandle);
        m_pipeVertexBufferHandle = kInvalidBufferHandle;
    }
    m_pipeIndexCount = 0;
}

void Renderer::destroyEnvironmentResources() {
    destroyDiffuseTextureResources();
}

void Renderer::destroyDiffuseTextureResources() {
    if (m_diffuseTextureSampler != VK_NULL_HANDLE) {
        vkDestroySampler(m_device, m_diffuseTextureSampler, nullptr);
        m_diffuseTextureSampler = VK_NULL_HANDLE;
    }
    if (m_diffuseTextureImageView != VK_NULL_HANDLE) {
        vkDestroyImageView(m_device, m_diffuseTextureImageView, nullptr);
        m_diffuseTextureImageView = VK_NULL_HANDLE;
    }
    if (m_diffuseTextureImage != VK_NULL_HANDLE) {
#if defined(VOXEL_HAS_VMA)
        if (m_vmaAllocator != VK_NULL_HANDLE && m_diffuseTextureAllocation != VK_NULL_HANDLE) {
            vmaDestroyImage(m_vmaAllocator, m_diffuseTextureImage, m_diffuseTextureAllocation);
            m_diffuseTextureAllocation = VK_NULL_HANDLE;
        } else {
            vkDestroyImage(m_device, m_diffuseTextureImage, nullptr);
        }
#else
        vkDestroyImage(m_device, m_diffuseTextureImage, nullptr);
#endif
        m_diffuseTextureImage = VK_NULL_HANDLE;
    }
    if (m_diffuseTextureMemory != VK_NULL_HANDLE) {
        vkFreeMemory(m_device, m_diffuseTextureMemory, nullptr);
        m_diffuseTextureMemory = VK_NULL_HANDLE;
    }
#if defined(VOXEL_HAS_VMA)
    m_diffuseTextureAllocation = VK_NULL_HANDLE;
#endif
}

void Renderer::destroyShadowResources() {
    if (m_shadowDepthSampler != VK_NULL_HANDLE) {
        vkDestroySampler(m_device, m_shadowDepthSampler, nullptr);
        m_shadowDepthSampler = VK_NULL_HANDLE;
    }
    if (m_shadowDepthImageView != VK_NULL_HANDLE) {
        vkDestroyImageView(m_device, m_shadowDepthImageView, nullptr);
        m_shadowDepthImageView = VK_NULL_HANDLE;
    }
    if (m_shadowDepthImage != VK_NULL_HANDLE) {
#if defined(VOXEL_HAS_VMA)
        if (m_vmaAllocator != VK_NULL_HANDLE && m_shadowDepthAllocation != VK_NULL_HANDLE) {
            vmaDestroyImage(m_vmaAllocator, m_shadowDepthImage, m_shadowDepthAllocation);
            m_shadowDepthAllocation = VK_NULL_HANDLE;
        } else {
            vkDestroyImage(m_device, m_shadowDepthImage, nullptr);
        }
#else
        vkDestroyImage(m_device, m_shadowDepthImage, nullptr);
#endif
        m_shadowDepthImage = VK_NULL_HANDLE;
    }
    if (m_shadowDepthMemory != VK_NULL_HANDLE) {
        vkFreeMemory(m_device, m_shadowDepthMemory, nullptr);
        m_shadowDepthMemory = VK_NULL_HANDLE;
    }
    m_shadowDepthInitialized = false;
}

void Renderer::destroyChunkBuffers() {
    for (ChunkDrawRange& drawRange : m_chunkDrawRanges) {
        drawRange.firstIndex = 0;
        drawRange.vertexOffset = 0;
        drawRange.indexCount = 0;
    }

    for (const DeferredBufferRelease& release : m_deferredBufferReleases) {
        if (release.handle != kInvalidBufferHandle) {
            m_bufferAllocator.destroyBuffer(release.handle);
        }
    }
    m_deferredBufferReleases.clear();

    m_chunkDrawRanges.clear();
    m_chunkLodMeshCache.clear();
    m_chunkGrassInstanceCache.clear();
    m_chunkLodMeshCacheValid = false;
    if (m_grassBillboardInstanceBufferHandle != kInvalidBufferHandle) {
        m_bufferAllocator.destroyBuffer(m_grassBillboardInstanceBufferHandle);
        m_grassBillboardInstanceBufferHandle = kInvalidBufferHandle;
    }
    m_grassBillboardInstanceCount = 0;
    m_bufferAllocator.destroyBuffer(m_chunkVertexBufferHandle);
    m_chunkVertexBufferHandle = kInvalidBufferHandle;
    m_bufferAllocator.destroyBuffer(m_chunkIndexBufferHandle);
    m_chunkIndexBufferHandle = kInvalidBufferHandle;
    m_pendingTransferTimelineValue = 0;
    m_currentChunkReadyTimelineValue = 0;
    m_transferCommandBufferInFlightValue = 0;
}

void Renderer::destroyPipeline() {
    if (m_ssaoBlurPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_device, m_ssaoBlurPipeline, nullptr);
        m_ssaoBlurPipeline = VK_NULL_HANDLE;
    }
    if (m_ssaoPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_device, m_ssaoPipeline, nullptr);
        m_ssaoPipeline = VK_NULL_HANDLE;
    }
    if (m_pipeNormalDepthPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_device, m_pipeNormalDepthPipeline, nullptr);
        m_pipeNormalDepthPipeline = VK_NULL_HANDLE;
    }
    if (m_voxelNormalDepthPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_device, m_voxelNormalDepthPipeline, nullptr);
        m_voxelNormalDepthPipeline = VK_NULL_HANDLE;
    }
    if (m_tonemapPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_device, m_tonemapPipeline, nullptr);
        m_tonemapPipeline = VK_NULL_HANDLE;
    }
    if (m_skyboxPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_device, m_skyboxPipeline, nullptr);
        m_skyboxPipeline = VK_NULL_HANDLE;
    }
    if (m_shadowPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_device, m_shadowPipeline, nullptr);
        m_shadowPipeline = VK_NULL_HANDLE;
    }
    if (m_pipeShadowPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_device, m_pipeShadowPipeline, nullptr);
        m_pipeShadowPipeline = VK_NULL_HANDLE;
    }
    if (m_grassBillboardShadowPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_device, m_grassBillboardShadowPipeline, nullptr);
        m_grassBillboardShadowPipeline = VK_NULL_HANDLE;
    }
    if (m_previewRemovePipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_device, m_previewRemovePipeline, nullptr);
        m_previewRemovePipeline = VK_NULL_HANDLE;
    }
    if (m_previewAddPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_device, m_previewAddPipeline, nullptr);
        m_previewAddPipeline = VK_NULL_HANDLE;
    }
    if (m_pipePipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_device, m_pipePipeline, nullptr);
        m_pipePipeline = VK_NULL_HANDLE;
    }
    if (m_grassBillboardPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_device, m_grassBillboardPipeline, nullptr);
        m_grassBillboardPipeline = VK_NULL_HANDLE;
    }
    if (m_pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_device, m_pipeline, nullptr);
        m_pipeline = VK_NULL_HANDLE;
    }
    if (m_pipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
        m_pipelineLayout = VK_NULL_HANDLE;
    }
}

void Renderer::shutdown() {
    VOX_LOGI("render") << "shutdown begin\n";
    if (m_device != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(m_device);
    }

    if (m_device != VK_NULL_HANDLE) {
#if defined(VOXEL_HAS_IMGUI)
        destroyImGuiResources();
#endif
        destroyFrameResources();
        destroyGpuTimestampResources();
        destroyTransferResources();
        if (m_renderTimelineSemaphore != VK_NULL_HANDLE) {
            vkDestroySemaphore(m_device, m_renderTimelineSemaphore, nullptr);
            m_renderTimelineSemaphore = VK_NULL_HANDLE;
        }
        destroyPipeBuffers();
        destroyPreviewBuffers();
        destroyEnvironmentResources();
        destroyShadowResources();
        destroyChunkBuffers();
        destroyPipeline();
        if (m_descriptorPool != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(m_device, m_descriptorPool, nullptr);
            m_descriptorPool = VK_NULL_HANDLE;
        }
        if (m_bindlessDescriptorPool != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(m_device, m_bindlessDescriptorPool, nullptr);
            m_bindlessDescriptorPool = VK_NULL_HANDLE;
        }
        if (m_descriptorSetLayout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(m_device, m_descriptorSetLayout, nullptr);
            m_descriptorSetLayout = VK_NULL_HANDLE;
        }
        if (m_bindlessDescriptorSetLayout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(m_device, m_bindlessDescriptorSetLayout, nullptr);
            m_bindlessDescriptorSetLayout = VK_NULL_HANDLE;
        }
        m_descriptorSets.fill(VK_NULL_HANDLE);
        m_bindlessDescriptorSet = VK_NULL_HANDLE;
        destroySwapchain();
        const uint32_t liveFrameArenaImagesBeforeShutdown = m_frameArena.liveImageCount();
        if (liveFrameArenaImagesBeforeShutdown > 0) {
            VOX_LOGI("render") << "shutdown: forcing cleanup of "
                << liveFrameArenaImagesBeforeShutdown
                << " remaining FrameArena image(s) before allocator shutdown\n";
            m_frameArena.destroyAllImages();
        }
        m_frameArena.shutdown(&m_bufferAllocator);
        m_bufferAllocator.shutdown();

        uint32_t rendererOwnedLiveImages = 0;
        auto logLiveImage = [&](const char* name, VkImage image) {
            if (image == VK_NULL_HANDLE) {
                return;
            }
            ++rendererOwnedLiveImages;
            VOX_LOGI("render") << "shutdown leak check: live image '" << name
                << "' handle=0x" << std::hex
                << static_cast<unsigned long long>(vkHandleToUint64(image))
                << std::dec << "\n";
        };
        logLiveImage("diffuse.albedo.image", m_diffuseTextureImage);
        logLiveImage("shadow.atlas.image", m_shadowDepthImage);
        for (uint32_t i = 0; i < static_cast<uint32_t>(m_depthImages.size()); ++i) {
            logLiveImage(("depth.msaa.image[" + std::to_string(i) + "]").c_str(), m_depthImages[i]);
        }
        for (uint32_t i = 0; i < static_cast<uint32_t>(m_msaaColorImages.size()); ++i) {
            logLiveImage(("hdr.msaaColor.image[" + std::to_string(i) + "]").c_str(), m_msaaColorImages[i]);
        }
        for (uint32_t i = 0; i < static_cast<uint32_t>(m_hdrResolveImages.size()); ++i) {
            logLiveImage(("hdr.resolve.image[" + std::to_string(i) + "]").c_str(), m_hdrResolveImages[i]);
        }
        for (uint32_t i = 0; i < static_cast<uint32_t>(m_normalDepthImages.size()); ++i) {
            logLiveImage(("ao.normalDepth.image[" + std::to_string(i) + "]").c_str(), m_normalDepthImages[i]);
        }
        for (uint32_t i = 0; i < static_cast<uint32_t>(m_aoDepthImages.size()); ++i) {
            logLiveImage(("ao.depth.image[" + std::to_string(i) + "]").c_str(), m_aoDepthImages[i]);
        }
        for (uint32_t i = 0; i < static_cast<uint32_t>(m_ssaoRawImages.size()); ++i) {
            logLiveImage(("ao.ssaoRaw.image[" + std::to_string(i) + "]").c_str(), m_ssaoRawImages[i]);
        }
        for (uint32_t i = 0; i < static_cast<uint32_t>(m_ssaoBlurImages.size()); ++i) {
            logLiveImage(("ao.ssaoBlur.image[" + std::to_string(i) + "]").c_str(), m_ssaoBlurImages[i]);
        }
        if (rendererOwnedLiveImages == 0) {
            VOX_LOGI("render") << "shutdown leak check: no renderer-owned live VkImage handles\n";
        }

#if defined(VOXEL_HAS_VMA)
        if (m_vmaAllocator != VK_NULL_HANDLE) {
            vmaDestroyAllocator(m_vmaAllocator);
            m_vmaAllocator = VK_NULL_HANDLE;
        }
#endif

        vkDestroyDevice(m_device, nullptr);
        m_device = VK_NULL_HANDLE;
    }

    if (m_surface != VK_NULL_HANDLE && m_instance != VK_NULL_HANDLE) {
        vkDestroySurfaceKHR(m_instance, m_surface, nullptr);
        m_surface = VK_NULL_HANDLE;
    }

    if (m_instance != VK_NULL_HANDLE) {
        vkDestroyInstance(m_instance, nullptr);
        m_instance = VK_NULL_HANDLE;
    }

    m_physicalDevice = VK_NULL_HANDLE;
    m_debugUtilsEnabled = false;
    m_setDebugUtilsObjectName = nullptr;
    m_cmdBeginDebugUtilsLabel = nullptr;
    m_cmdEndDebugUtilsLabel = nullptr;
    m_cmdInsertDebugUtilsLabel = nullptr;
    m_graphicsQueue = VK_NULL_HANDLE;
    m_transferQueue = VK_NULL_HANDLE;
    m_graphicsQueueFamilyIndex = 0;
    m_graphicsQueueIndex = 0;
    m_transferQueueFamilyIndex = 0;
    m_transferQueueIndex = 0;
    m_aoExtent = VkExtent2D{};
    m_depthFormat = VK_FORMAT_UNDEFINED;
    m_shadowDepthFormat = VK_FORMAT_UNDEFINED;
    m_hdrColorFormat = VK_FORMAT_UNDEFINED;
    m_normalDepthFormat = VK_FORMAT_UNDEFINED;
    m_ssaoFormat = VK_FORMAT_UNDEFINED;
    m_supportsWireframePreview = false;
    m_supportsSamplerAnisotropy = false;
    m_supportsMultiDrawIndirect = false;
    m_chunkMeshingOptions = world::MeshingOptions{};
    m_chunkMeshRebuildRequested = false;
    m_pendingChunkRemeshIndices.clear();
    m_gpuTimestampsSupported = false;
    m_gpuTimestampPeriodNs = 0.0f;
    m_gpuTimestampQueryPools.fill(VK_NULL_HANDLE);
    m_debugGpuFrameTimeMs = 0.0f;
    m_debugGpuShadowTimeMs = 0.0f;
    m_debugGpuPrepassTimeMs = 0.0f;
    m_debugGpuSsaoTimeMs = 0.0f;
    m_debugGpuSsaoBlurTimeMs = 0.0f;
    m_debugGpuMainTimeMs = 0.0f;
    m_debugGpuPostTimeMs = 0.0f;
    m_debugChunkMeshVertexCount = 0;
    m_debugChunkMeshIndexCount = 0;
    m_debugChunkLastRemeshedChunkCount = 0;
    m_debugChunkLastRemeshActiveVertexCount = 0;
    m_debugChunkLastRemeshActiveIndexCount = 0;
    m_debugChunkLastRemeshNaiveVertexCount = 0;
    m_debugChunkLastRemeshNaiveIndexCount = 0;
    m_debugChunkLastRemeshReductionPercent = 0.0f;
    m_debugChunkLastRemeshMs = 0.0f;
    m_debugChunkLastFullRemeshMs = 0.0f;
    m_debugEnableSpatialQueries = true;
    m_debugClipmapConfig = world::ClipmapConfig{};
    m_debugSpatialQueriesUsed = false;
    m_debugSpatialQueryStats = {};
    m_debugSpatialVisibleChunkCount = 0;
    m_debugCpuFrameTimingMsHistory.fill(0.0f);
    m_debugCpuFrameTimingMsHistoryWrite = 0;
    m_debugCpuFrameTimingMsHistoryCount = 0;
    m_debugGpuFrameTimingMsHistory.fill(0.0f);
    m_debugGpuFrameTimingMsHistoryWrite = 0;
    m_debugGpuFrameTimingMsHistoryCount = 0;
    m_frameTimelineValues.fill(0);
    m_pendingTransferTimelineValue = 0;
    m_currentChunkReadyTimelineValue = 0;
    m_transferCommandBufferInFlightValue = 0;
    m_lastGraphicsTimelineValue = 0;
    m_nextTimelineValue = 1;
    m_currentFrame = 0;
    m_window = nullptr;
    VOX_LOGI("render") << "shutdown complete\n";
}

} // namespace render
