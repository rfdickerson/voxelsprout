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
constexpr uint32_t kBindlessTextureIndexPlantDiffuse = 6;
constexpr uint32_t kBindlessTextureStaticCount = 7;
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
constexpr uint32_t kVoxelGiGridResolution = 64u;
constexpr uint32_t kVoxelGiWorkgroupSize = 4u;
constexpr uint32_t kVoxelGiPropagationIterations = 8u;
constexpr uint32_t kHdrResolveBloomMipCount = 6u;
constexpr uint32_t kAutoExposureHistogramBins = 64u;
constexpr uint32_t kAutoExposureWorkgroupSize = 16u;
constexpr uint32_t kSunShaftWorkgroupSize = 8u;
constexpr float kVoxelGiCellSize = 1.0f;
constexpr float kPipeTransferHalfExtent = 0.58f;
constexpr float kPipeMinRadius = 0.02f;
constexpr float kPipeMaxRadius = 0.5f;
constexpr float kPipeBranchRadiusBoost = 0.05f;
constexpr float kPipeMaxEndExtension = 0.49f;
constexpr float kBeltRadius = 0.49f;
constexpr float kTrackRadius = 0.38f;
constexpr math::Vector3 kBeltTint{0.78f, 0.62f, 0.18f};
constexpr math::Vector3 kTrackTint{0.52f, 0.54f, 0.58f};
constexpr float kBeltCargoLength = 0.30f;
constexpr float kBeltCargoRadius = 0.30f;
constexpr std::array<math::Vector3, 5> kBeltCargoTints = {
    math::Vector3{0.92f, 0.31f, 0.31f},
    math::Vector3{0.31f, 0.71f, 0.96f},
    math::Vector3{0.95f, 0.84f, 0.32f},
    math::Vector3{0.56f, 0.88f, 0.48f},
    math::Vector3{0.84f, 0.54f, 0.92f},
};
constexpr uint64_t kAcquireNextImageTimeoutNs = 100000000ull; // 100 ms
constexpr uint64_t kFrameTimelineWarnLagThreshold = 6u;
constexpr double kFrameTimelineWarnCooldownSeconds = 2.0;
constexpr float kCpuFrameEwmaAlpha = 0.08f;

void imguiCheckVkResult(VkResult result) {
    if (result != VK_SUCCESS) {
        VOX_LOGE("imgui") << "Vulkan backend error: " << static_cast<int>(result);
    }
}

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
    float skyConfig2[4];
    float skyConfig3[4];
    float skyConfig4[4];
    float skyConfig5[4];
    float colorGrading0[4];
    float colorGrading1[4];
    float colorGrading2[4];
    float colorGrading3[4];
    float voxelBaseColorPalette[16][4];
    float voxelGiGridOriginCellSize[4];
    float voxelGiGridExtentStrength[4];
};

struct alignas(16) ChunkPushConstants {
    float chunkOffset[4];
    float cascadeData[4];
};

struct alignas(16) ChunkInstanceData {
    float chunkOffset[4];
};

struct alignas(16) AutoExposureHistogramPushConstants {
    uint32_t width = 1u;
    uint32_t height = 1u;
    uint32_t totalPixels = 1u;
    uint32_t binCount = kAutoExposureHistogramBins;
    float minLogLuminance = -10.0f;
    float maxLogLuminance = 4.0f;
    float sourceMipLevel = 0.0f;
    float _pad1 = 0.0f;
};

struct alignas(16) AutoExposureUpdatePushConstants {
    uint32_t totalPixels = 1u;
    uint32_t binCount = kAutoExposureHistogramBins;
    uint32_t resetHistory = 1u;
    uint32_t _pad0 = 0u;
    float minLogLuminance = -10.0f;
    float maxLogLuminance = 4.0f;
    float lowPercentile = 0.5f;
    float highPercentile = 0.98f;
    float keyValue = 0.18f;
    float minExposure = 0.25f;
    float maxExposure = 2.2f;
    float adaptUpRate = 3.0f;
    float adaptDownRate = 1.4f;
    float deltaTimeSeconds = 0.016f;
    float _pad1 = 0.0f;
    float _pad2 = 0.0f;
};

struct alignas(16) SunShaftPushConstants {
    uint32_t width = 1u;
    uint32_t height = 1u;
    uint32_t sampleCount = 10u;
    uint32_t _pad0 = 0u;
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
            vertex.bits = world::PackedVoxelVertex::pack(x, y, z, faceId, corner, ao, material, 0u, 2u);
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

bool chunkIntersectsShadowCascadeClip(
    const world::Chunk& chunk,
    const math::Matrix4& lightViewProj,
    float clipMargin
) {
    const float chunkMinX = static_cast<float>(chunk.chunkX() * world::Chunk::kSizeX);
    const float chunkMinY = static_cast<float>(chunk.chunkY() * world::Chunk::kSizeY);
    const float chunkMinZ = static_cast<float>(chunk.chunkZ() * world::Chunk::kSizeZ);
    const float chunkMaxX = chunkMinX + static_cast<float>(world::Chunk::kSizeX);
    const float chunkMaxY = chunkMinY + static_cast<float>(world::Chunk::kSizeY);
    const float chunkMaxZ = chunkMinZ + static_cast<float>(world::Chunk::kSizeZ);

    std::array<math::Vector3, 8> corners = {
        math::Vector3{chunkMinX, chunkMinY, chunkMinZ},
        math::Vector3{chunkMaxX, chunkMinY, chunkMinZ},
        math::Vector3{chunkMinX, chunkMaxY, chunkMinZ},
        math::Vector3{chunkMaxX, chunkMaxY, chunkMinZ},
        math::Vector3{chunkMinX, chunkMinY, chunkMaxZ},
        math::Vector3{chunkMaxX, chunkMinY, chunkMaxZ},
        math::Vector3{chunkMinX, chunkMaxY, chunkMaxZ},
        math::Vector3{chunkMaxX, chunkMaxY, chunkMaxZ},
    };

    float ndcMinX = std::numeric_limits<float>::max();
    float ndcMinY = std::numeric_limits<float>::max();
    float ndcMinZ = std::numeric_limits<float>::max();
    float ndcMaxX = std::numeric_limits<float>::lowest();
    float ndcMaxY = std::numeric_limits<float>::lowest();
    float ndcMaxZ = std::numeric_limits<float>::lowest();
    for (const math::Vector3& corner : corners) {
        const math::Vector3 clip = math::transformPoint(lightViewProj, corner);
        ndcMinX = std::min(ndcMinX, clip.x);
        ndcMinY = std::min(ndcMinY, clip.y);
        ndcMinZ = std::min(ndcMinZ, clip.z);
        ndcMaxX = std::max(ndcMaxX, clip.x);
        ndcMaxY = std::max(ndcMaxY, clip.y);
        ndcMaxZ = std::max(ndcMaxZ, clip.z);
    }

    return !(ndcMaxX < (-1.0f - clipMargin) ||
             ndcMinX > (1.0f + clipMargin) ||
             ndcMaxY < (-1.0f - clipMargin) ||
             ndcMinY > (1.0f + clipMargin) ||
             ndcMaxZ < (0.0f - clipMargin) ||
             ndcMinZ > (1.0f + clipMargin));
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

struct SkyTuningSample {
    float rayleighStrength = 1.0f;
    float mieStrength = 1.0f;
    float mieAnisotropy = 0.55f;
    float skyExposure = 1.0f;
    float sunDiskIntensity = 1150.0f;
    float sunHaloIntensity = 22.0f;
    float sunDiskSize = 2.0f;
    float sunHazeFalloff = 0.35f;
};

SkyTuningSample evaluateSunriseSkyTuning(float sunElevationDegrees) {
    const float h = saturate((sunElevationDegrees + 12.0f) / 32.0f);
    const float day = smoothStep(0.15f, 0.85f, h);

    SkyTuningSample sample{};
    sample.rayleighStrength = std::lerp(1.20f, 1.00f, day);
    sample.mieStrength = std::lerp(1.85f, 0.75f, day);
    sample.mieAnisotropy = std::lerp(0.87f, 0.78f, day);
    sample.skyExposure = std::lerp(1.35f, 1.00f, day);
    sample.sunDiskIntensity = std::lerp(1450.0f, 1150.0f, day);
    sample.sunHaloIntensity = std::lerp(36.0f, 22.0f, day);
    sample.sunDiskSize = std::lerp(3.2f, 1.8f, day);
    sample.sunHazeFalloff = std::lerp(0.62f, 0.34f, day);
    return sample;
}

SkyTuningSample blendSkyTuningSample(const SkyTuningSample& base, const SkyTuningSample& target, float blend) {
    const float t = std::clamp(blend, 0.0f, 1.0f);
    SkyTuningSample result{};
    result.rayleighStrength = std::lerp(base.rayleighStrength, target.rayleighStrength, t);
    result.mieStrength = std::lerp(base.mieStrength, target.mieStrength, t);
    result.mieAnisotropy = std::lerp(base.mieAnisotropy, target.mieAnisotropy, t);
    result.skyExposure = std::lerp(base.skyExposure, target.skyExposure, t);
    result.sunDiskIntensity = std::lerp(base.sunDiskIntensity, target.sunDiskIntensity, t);
    result.sunHaloIntensity = std::lerp(base.sunHaloIntensity, target.sunHaloIntensity, t);
    result.sunDiskSize = std::lerp(base.sunDiskSize, target.sunDiskSize, t);
    result.sunHazeFalloff = std::lerp(base.sunHazeFalloff, target.sunHazeFalloff, t);
    return result;
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

void transitionBufferAccess(
    VkCommandBuffer commandBuffer,
    VkBuffer buffer,
    VkDeviceSize offset,
    VkDeviceSize size,
    VkPipelineStageFlags2 srcStageMask,
    VkAccessFlags2 srcAccessMask,
    VkPipelineStageFlags2 dstStageMask,
    VkAccessFlags2 dstAccessMask
) {
    VkBufferMemoryBarrier2 bufferBarrier{};
    bufferBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
    bufferBarrier.srcStageMask = srcStageMask;
    bufferBarrier.srcAccessMask = srcAccessMask;
    bufferBarrier.dstStageMask = dstStageMask;
    bufferBarrier.dstAccessMask = dstAccessMask;
    bufferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bufferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bufferBarrier.buffer = buffer;
    bufferBarrier.offset = offset;
    bufferBarrier.size = size;

    VkDependencyInfo dependencyInfo{};
    dependencyInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dependencyInfo.bufferMemoryBarrierCount = 1;
    dependencyInfo.pBufferMemoryBarriers = &bufferBarrier;
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

VkFormat findSupportedVoxelGiFormat(VkPhysicalDevice physicalDevice) {
    constexpr std::array<VkFormat, 2> kVoxelGiCandidates = {
        VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_FORMAT_R32G32B32A32_SFLOAT
    };

    for (VkFormat format : kVoxelGiCandidates) {
        VkFormatProperties properties{};
        vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &properties);
        const VkFormatFeatureFlags requiredFeatures =
            VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT;
        if ((properties.optimalTilingFeatures & requiredFeatures) == requiredFeatures) {
            return format;
        }
    }

    return VK_FORMAT_UNDEFINED;
}

VkFormat findSupportedVoxelGiOccupancyFormat(VkPhysicalDevice physicalDevice) {
    constexpr std::array<VkFormat, 1> kOccupancyCandidates = {
        VK_FORMAT_R8G8B8A8_UNORM
    };

    for (VkFormat format : kOccupancyCandidates) {
        VkFormatProperties properties{};
        vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &properties);
        const VkFormatFeatureFlags requiredFeatures = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
        if ((properties.optimalTilingFeatures & requiredFeatures) == requiredFeatures) {
            return format;
        }
    }

    return VK_FORMAT_UNDEFINED;
}

std::array<std::uint8_t, 3> voxelTypeAlbedoRgb(world::VoxelType type) {
    switch (type) {
    case world::VoxelType::Stone:
        return {150u, 154u, 160u};
    case world::VoxelType::Dirt:
        return {122u, 93u, 58u};
    case world::VoxelType::Grass:
        return {80u, 142u, 63u};
    case world::VoxelType::Wood:
        return {141u, 106u, 64u};
    case world::VoxelType::SolidRed:
        return {255u, 71u, 56u};
    case world::VoxelType::Empty:
    default:
        return {0u, 0u, 0u};
    }
}

std::array<std::uint8_t, 3> voxelGiAlbedoRgb(
    const world::Voxel& voxel,
    const std::array<std::uint32_t, 16>& palette
) {
    if (voxel.baseColorIndex <= 0x0Fu) {
        const std::uint32_t rgba = palette[voxel.baseColorIndex & 0x0Fu];
        return {
            static_cast<std::uint8_t>(rgba & 0xFFu),
            static_cast<std::uint8_t>((rgba >> 8u) & 0xFFu),
            static_cast<std::uint8_t>((rgba >> 16u) & 0xFFu)
        };
    }
    return voxelTypeAlbedoRgb(voxel.type);
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

int floorDiv(int value, int divisor) {
    const int q = value / divisor;
    const int r = value % divisor;
    if (r != 0 && ((r < 0) != (divisor < 0))) {
        return q - 1;
    }
    return q;
}

struct ChunkCoordKey {
    int x = 0;
    int y = 0;
    int z = 0;

    bool operator==(const ChunkCoordKey& rhs) const = default;
};

struct ChunkCoordKeyHash {
    std::size_t operator()(const ChunkCoordKey& key) const noexcept {
        std::size_t h = static_cast<std::size_t>(static_cast<std::uint32_t>(key.x));
        h ^= static_cast<std::size_t>(static_cast<std::uint32_t>(key.y)) * 0x9E3779B1u;
        h ^= static_cast<std::size_t>(static_cast<std::uint32_t>(key.z)) * 0x85EBCA77u;
        return h;
    }
};

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
        if (presentMode == VK_PRESENT_MODE_FIFO_KHR) {
            return presentMode;
        }
    }
    for (const VkPresentModeKHR presentMode : presentModes) {
        if (presentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
            return presentMode;
        }
    }
    return presentModes.front();
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

void destroyShaderModules(VkDevice device, std::span<const VkShaderModule> shaderModules) {
    for (const VkShaderModule shaderModule : shaderModules) {
        if (shaderModule != VK_NULL_HANDLE) {
            vkDestroyShaderModule(device, shaderModule, nullptr);
        }
    }
}

} // namespace
