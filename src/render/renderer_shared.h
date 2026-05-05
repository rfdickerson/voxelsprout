#pragma once

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
constexpr uint32_t kBindlessTextureIndexSkyDaylight = 7;
constexpr uint32_t kBindlessTextureIndexWaterNormal = 8;
constexpr uint32_t kBindlessTextureIndexTerrainDetail = 9;
constexpr uint32_t kBindlessTextureStaticCount = 10;
constexpr uint32_t kShadowCascadeCount = 4;
constexpr uint32_t kImportedLocalLightCapacity = 64;
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
constexpr odai::math::Vector3 kBeltTint{0.78f, 0.62f, 0.18f};
constexpr odai::math::Vector3 kTrackTint{0.52f, 0.54f, 0.58f};
constexpr float kBeltCargoLength = 0.30f;
constexpr float kBeltCargoRadius = 0.30f;
constexpr std::array<odai::math::Vector3, 5> kBeltCargoTints = {
    odai::math::Vector3{0.92f, 0.31f, 0.31f},
    odai::math::Vector3{0.31f, 0.71f, 0.96f},
    odai::math::Vector3{0.95f, 0.84f, 0.32f},
    odai::math::Vector3{0.56f, 0.88f, 0.48f},
    odai::math::Vector3{0.84f, 0.54f, 0.92f},
};
constexpr uint64_t kAcquireNextImageTimeoutNs = 2000000ull; // 2 ms
constexpr uint64_t kFrameTimelineWarnLagThreshold = 6u;
constexpr double kFrameTimelineWarnCooldownSeconds = 2.0;
constexpr float kCpuFrameEwmaAlpha = 0.08f;

struct alignas(16) CameraUniform {
    float mvp[16];
    float view[16];
    float proj[16];
    float lightViewProj[kShadowCascadeCount][16];
    float invLightViewProj[kShadowCascadeCount][16];
    float shadowCascadeSplits[4];
    float shadowAtlasUvRects[kShadowCascadeCount][4];
    float sunDirectionIntensity[4];
    float sunColorShadow[4];
    float shIrradiance[9][4];
    float shadowConfig0[4];
    float shadowConfig1[4];
    float shadowConfig2[4];
    float shadowConfig3[4];
    float shadowConfig4[4];
    float shadowVoxelGridOrigin[4];
    float shadowVoxelGridSize[4];
    float skyConfig0[4];
    float skyConfig1[4];
    float skyConfig2[4];
    float skyConfig3[4];
    float skyConfig4[4];
    float skyConfig5[4];
    float voxelGiRestirConfig0[4];
    float voxelGiRestirConfig1[4];
    float colorGrading0[4];
    float colorGrading1[4];
    float colorGrading2[4];
    float colorGrading3[4];
    float dofConfig[4];
    float dofConfig2[4];
    float waterConfig[4];
    float importedLightPositionRadius[kImportedLocalLightCapacity][4];
    float importedLightColorIntensity[kImportedLocalLightCapacity][4];
    float importedLightConfig[4];
    float morrowindGiConfig[4];
    float voxelBaseColorPalette[16][4];
    float voxelGiGridOriginCellSize[4];
    float voxelGiGridExtentStrength[4];
};

struct alignas(16) ChunkPushConstants {
    float chunkOffset[4];
    float cascadeData[4];
    std::uint32_t actorData[4];
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

struct PipeEndpointState {
    odai::math::Vector3 axis{0.0f, 1.0f, 0.0f};
    float renderedRadius = 0.45f;
    float startExtension = 0.0f;
    float endExtension = 0.0f;
};

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

void imguiCheckVkResult(VkResult result);
odai::world::ChunkMeshData buildSingleVoxelPreviewMesh(
    std::uint32_t x,
    std::uint32_t y,
    std::uint32_t z,
    std::uint32_t ao,
    std::uint32_t material
);
PipeMeshData buildTransportBoxMesh();
PipeMeshData buildPipeCylinderMesh();
odai::math::Vector3 beltDirectionAxis(odai::sim::BeltDirection direction);
odai::math::Vector3 trackDirectionAxis(odai::sim::TrackDirection direction);
std::vector<PipeEndpointState> buildPipeEndpointStates(const std::vector<odai::sim::Pipe>& pipes);
odai::math::Matrix4 transpose(const odai::math::Matrix4& matrix);
odai::math::Matrix4 perspectiveVulkan(float fovYRadians, float aspectRatio, float nearPlane, float farPlane);
odai::math::Matrix4 orthographicVulkan(
    float left,
    float right,
    float bottom,
    float top,
    float nearPlane,
    float farPlane
);
odai::math::Matrix4 lookAt(
    const odai::math::Vector3& eye,
    const odai::math::Vector3& target,
    const odai::math::Vector3& up
);
bool chunkIntersectsShadowCascadeClip(
    const odai::world::Chunk& chunk,
    const odai::math::Matrix4& lightViewProjection,
    float clipMargin
);
SkyTuningSample evaluateSunriseSkyTuning(float sunElevationDegrees);
SkyTuningSample blendSkyTuningSample(const SkyTuningSample& base, const SkyTuningSample& target, float blend);
odai::math::Vector3 computeSunColor(
    const RendererBackend::SkyDebugSettings& settings,
    const odai::math::Vector3& sunDirection
);
std::array<odai::math::Vector3, 9> computeIrradianceShCoefficients(
    const odai::math::Vector3& sunDirection,
    const odai::math::Vector3& sunColor,
    const RendererBackend::SkyDebugSettings& settings
);
uint32_t findMemoryTypeIndex(
    VkPhysicalDevice physicalDevice,
    uint32_t typeBits,
    VkMemoryPropertyFlags requiredProperties
);
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
);
void transitionBufferAccess(
    VkCommandBuffer commandBuffer,
    VkBuffer buffer,
    VkDeviceSize offset,
    VkDeviceSize size,
    VkPipelineStageFlags2 srcStageMask,
    VkAccessFlags2 srcAccessMask,
    VkPipelineStageFlags2 dstStageMask,
    VkAccessFlags2 dstAccessMask
);
VkFormat findSupportedDepthFormat(VkPhysicalDevice physicalDevice);
VkFormat findSupportedShadowDepthFormat(VkPhysicalDevice physicalDevice);
VkFormat findSupportedHdrColorFormat(VkPhysicalDevice physicalDevice);
VkFormat findSupportedNormalDepthFormat(VkPhysicalDevice physicalDevice);
VkFormat findSupportedSsaoFormat(VkPhysicalDevice physicalDevice);
VkFormat findSupportedVoxelGiFormat(VkPhysicalDevice physicalDevice);
VkFormat findSupportedVoxelGiOccupancyFormat(VkPhysicalDevice physicalDevice);
std::array<std::uint8_t, 3> voxelGiAlbedoRgb(
    const odai::world::Voxel& voxel,
    const std::array<std::uint32_t, 16>& palette
);
const char* vkResultName(VkResult result);
void logVkFailure(const char* context, VkResult result);
int floorDiv(int value, int divisor);
bool isLayerAvailable(const char* layerName);
bool isInstanceExtensionAvailable(const char* extensionName);
void appendInstanceExtensionIfMissing(std::vector<const char*>& extensions, const char* extensionName);
QueueFamilyChoice findQueueFamily(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface);
bool hasRequiredDeviceExtensions(VkPhysicalDevice physicalDevice);
SwapchainSupport querySwapchainSupport(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface);
VkSurfaceFormatKHR chooseSwapchainFormat(const std::vector<VkSurfaceFormatKHR>& formats);
VkPresentModeKHR choosePresentMode(const std::vector<VkPresentModeKHR>& presentModes);
VkExtent2D chooseExtent(GLFWwindow* window, const VkSurfaceCapabilitiesKHR& capabilities);
std::optional<std::vector<std::uint8_t>> readBinaryFile(const char* filePath);
bool createShaderModuleFromFile(
    VkDevice device,
    const char* filePath,
    const char* debugName,
    VkShaderModule& outShaderModule
);
void destroyShaderModules(VkDevice device, std::span<const VkShaderModule> shaderModules);
