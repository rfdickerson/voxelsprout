#pragma once

#include "render/backend/vulkan/buffer_helpers.h"
#include "render/backend/vulkan/descriptor_manager.h"
#include "render/frame_graph.h"
#include "render/backend/vulkan/pipeline_manager.h"
#include "render/renderer_types.h"
#include "sim/simulation.h"
#include "world/clipmap_index.h"
#include "world/chunk_grid.h"
#include "world/chunk_mesher.h"
#include "world/spatial_index.h"
#include "math/math.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

struct GLFWwindow;

// Render subsystem
// Responsible for: owning the rendering interface used by the app.
// Should NOT do: gameplay simulation, world editing rules, or graphics API specifics yet.
namespace voxelsprout::render {

class CoreFrameGraphOrderValidator;
struct CoreFrameGraphPlan;

class RendererBackend {
public:
    struct ShadowDebugSettings {
        float casterConstantBiasBase = 1.1f;
        float casterConstantBiasCascadeScale = 0.9f;
        float casterSlopeBiasBase = 1.7f;
        float casterSlopeBiasCascadeScale = 0.85f;

        float receiverNormalOffsetNear = 0.03f;
        float receiverNormalOffsetFar = 0.12f;
        float receiverBaseBiasNearTexel = 0.05f;
        float receiverBaseBiasFarTexel = 4.6f;
        float receiverSlopeBiasNearTexel = 3.8f;
        float receiverSlopeBiasFarTexel = 7.2f;

        float cascadeBlendMin = 6.0f;
        float cascadeBlendFactor = 0.30f;

        float pcfRadius = 1.0f;
        int grassShadowCascadeCount = 1;
        bool enableOccluderCulling = true;

        float ssaoRadius = 0.55f;
        float ssaoBias = 0.03f;
        float ssaoIntensity = 0.60f;
    };

    struct SkyDebugSettings {
        float sunYawDegrees = -157.5f;
        float sunPitchDegrees = -13.0f;
        float rayleighStrength = 1.0f;
        float mieStrength = 1.0f;
        float mieAnisotropy = 0.55f;
        float skyExposure = 1.0f;
        float sunDiskIntensity = 1150.0f;
        float sunHaloIntensity = 22.0f;
        float sunDiskSize = 2.0f;
        float sunHazeFalloff = 0.35f;
        float bloomThreshold = 0.75f;
        float bloomSoftKnee = 0.5f;
        float bloomBaseIntensity = 0.08f;
        float bloomSunFacingBoost = 0.28f;
        bool autoExposureEnabled = true;
        float manualExposure = 1.00f;
        float autoExposureKeyValue = 0.18f;
        float autoExposureMin = 0.25f;
        float autoExposureMax = 2.20f;
        float autoExposureAdaptUp = 3.0f;
        float autoExposureAdaptDown = 1.4f;
        float autoExposureLowPercentile = 0.50f;
        float autoExposureHighPercentile = 0.98f;
        int autoExposureUpdateIntervalFrames = 1;
        float colorGradingWhiteBalanceR = 1.02f;
        float colorGradingWhiteBalanceG = 1.00f;
        float colorGradingWhiteBalanceB = 0.98f;
        float colorGradingContrast = 1.08f;
        float colorGradingSaturation = 1.05f;
        float colorGradingVibrance = 0.10f;
        float colorGradingShadowTintR = 0.02f;
        float colorGradingShadowTintG = 0.03f;
        float colorGradingShadowTintB = 0.05f;
        float colorGradingHighlightTintR = 0.03f;
        float colorGradingHighlightTintG = 0.02f;
        float colorGradingHighlightTintB = 0.01f;
        float volumetricFogDensity = 0.0045f;
        float volumetricFogHeightFalloff = 0.075f;
        float volumetricFogBaseHeight = 6.0f;
        float volumetricSunScattering = 1.25f;
        bool autoSunriseTuning = true;
        float autoSunriseBlend = 1.0f;
        float autoSunriseAdaptSpeed = 4.0f;
        float plantQuadDirectionality = 0.34f;
    };

    struct VoxelGiDebugSettings {
        float bounceStrength = 1.45f;
        float diffusionSoftness = 0.45f;
        int visualizationMode = 0; // 0 = off, 1 = radiance, 2 = false-color luminance, 3 = radiance gray, 4 = occupancy albedo
    };

    bool init(GLFWwindow* window, const voxelsprout::world::ChunkGrid& chunkGrid);
    void clearMagicaVoxelMeshes();
    bool uploadMagicaVoxelMesh(const voxelsprout::world::ChunkMeshData& mesh, float worldOffsetX, float worldOffsetY, float worldOffsetZ);
    void setVoxelBaseColorPalette(const std::array<std::uint32_t, 16>& paletteRgba);
    bool updateChunkMesh(const voxelsprout::world::ChunkGrid& chunkGrid);
    bool updateChunkMesh(const voxelsprout::world::ChunkGrid& chunkGrid, std::size_t chunkIndex);
    bool updateChunkMesh(const voxelsprout::world::ChunkGrid& chunkGrid, std::span<const std::size_t> chunkIndices);
    bool useSpatialPartitioningQueries() const;
    voxelsprout::world::ClipmapConfig clipmapQueryConfig() const;
    void setSpatialQueryStats(bool used, const voxelsprout::world::SpatialQueryStats& stats, std::uint32_t visibleChunkCount);
    void renderFrame(
        const voxelsprout::world::ChunkGrid& chunkGrid,
        const voxelsprout::sim::Simulation& simulation,
        const voxelsprout::render::CameraPose& camera,
        const voxelsprout::render::VoxelPreview& preview,
        float simulationAlpha,
        std::span<const std::size_t> visibleChunkIndices
    );
    void setDebugUiVisible(bool visible);
    bool isDebugUiVisible() const;
    void setFrameStatsVisible(bool visible);
    bool isFrameStatsVisible() const;
    void setSunAngles(float yawDegrees, float pitchDegrees);
    float cameraFovDegrees() const;
    void shutdown();

private:
    static constexpr uint32_t kMaxFramesInFlight = 2;
    static constexpr uint32_t kShadowCascadeCount = 4;
    static constexpr uint32_t kShadowAtlasSize = 8192;
    static constexpr uint32_t kGpuTimestampQueryFrameStart = 0;
    static constexpr uint32_t kGpuTimestampQueryShadowStart = 1;
    static constexpr uint32_t kGpuTimestampQueryShadowEnd = 2;
    static constexpr uint32_t kGpuTimestampQueryGiInjectStart = 3;
    static constexpr uint32_t kGpuTimestampQueryGiInjectEnd = 4;
    static constexpr uint32_t kGpuTimestampQueryGiPropagateStart = 5;
    static constexpr uint32_t kGpuTimestampQueryGiPropagateEnd = 6;
    static constexpr uint32_t kGpuTimestampQueryAutoExposureStart = 7;
    static constexpr uint32_t kGpuTimestampQueryAutoExposureEnd = 8;
    static constexpr uint32_t kGpuTimestampQuerySunShaftStart = 9;
    static constexpr uint32_t kGpuTimestampQuerySunShaftEnd = 10;
    static constexpr uint32_t kGpuTimestampQueryPrepassStart = 11;
    static constexpr uint32_t kGpuTimestampQueryPrepassEnd = 12;
    static constexpr uint32_t kGpuTimestampQuerySsaoStart = 13;
    static constexpr uint32_t kGpuTimestampQuerySsaoEnd = 14;
    static constexpr uint32_t kGpuTimestampQuerySsaoBlurStart = 15;
    static constexpr uint32_t kGpuTimestampQuerySsaoBlurEnd = 16;
    static constexpr uint32_t kGpuTimestampQueryMainStart = 17;
    static constexpr uint32_t kGpuTimestampQueryMainEnd = 18;
    static constexpr uint32_t kGpuTimestampQueryPostStart = 19;
    static constexpr uint32_t kGpuTimestampQueryPostEnd = 20;
    static constexpr uint32_t kGpuTimestampQueryFrameEnd = 21;
    static constexpr uint32_t kGpuTimestampQueryCount = 22;
    static constexpr std::uint32_t kTimingHistorySampleCount = 240;

    struct FrameResources {
        // Per-frame command pool to allocate fresh command buffers every frame.
        // Future frame-graph systems will replace this with transient allocators.
        VkCommandPool commandPool = VK_NULL_HANDLE;
        // Signals when swapchain image acquisition is complete for this frame.
        VkSemaphore imageAvailable = VK_NULL_HANDLE;
    };

    bool createInstance();
    bool createSurface();
    bool pickPhysicalDevice();
    bool createLogicalDevice();
    bool createSwapchain();
    bool createHdrResolveTargets();
    bool createMsaaColorTargets();
    bool createDepthTargets();
    bool createAoTargets();
    bool createShadowResources();
    bool createVoxelGiResources();
    bool createAutoExposureResources();
    bool createSunShaftResources();
    bool createTimelineSemaphore();
    bool createGraphicsPipeline();
    bool createMagicaPipeline();
    bool createUploadRingBuffer();
    bool createTransferResources();
    bool createPipeBuffers();
    bool createPipePipeline();
    bool createAoPipelines();
    bool createPreviewBuffers();
    bool createEnvironmentResources();
    bool createDiffuseTextureResources();
    bool createDescriptorResources();
    using BoundDescriptorSets = DescriptorManager<kMaxFramesInFlight>::BoundDescriptorSets;
    BoundDescriptorSets updateFrameDescriptorSets(
        uint32_t aoFrameIndex,
        const VkDescriptorBufferInfo& cameraBufferInfo,
        VkBuffer autoExposureHistogramBuffer,
        VkBuffer autoExposureStateBuffer
    );
    bool createChunkBuffers(const voxelsprout::world::ChunkGrid& chunkGrid, std::span<const std::size_t> remeshChunkIndices);
    bool createFrameResources();
    bool createGpuTimestampResources();
    bool createImGuiResources();
    void destroyImGuiResources();
    void buildFrameStatsUi();
    void buildMeshingDebugUi();
    void buildShadowDebugUi();
    void buildSunDebugUi();
    void buildAimReticleUi();
    std::vector<std::uint8_t> buildShadowCandidateMask(
        std::span<const voxelsprout::world::Chunk> chunks,
        std::span<const std::size_t> visibleChunkIndices
    ) const;
    void recordVoxelGiDispatchSequence(
        VkCommandBuffer commandBuffer,
        uint32_t mvpDynamicOffset,
        VkQueryPool gpuTimestampQueryPool
    );
    bool recreateSwapchain();
    void destroySwapchain();
    void destroyHdrResolveTargets();
    void destroyMsaaColorTargets();
    void destroyDepthTargets();
    void destroyAoTargets();
    void destroyGpuTimestampResources();
    void destroyShadowResources();
    void destroyVoxelGiResources();
    void destroyAutoExposureResources();
    void destroySunShaftResources();
    void destroyFrameResources();
    void destroyChunkBuffers();
    void destroyMagicaBuffers();
    void destroyPipeBuffers();
    void destroyPreviewBuffers();
    void destroyEnvironmentResources();
    void destroyDiffuseTextureResources();
    void destroyTransferResources();
    void destroyPipeline();
    void loadDebugUtilsFunctions();
    bool allocatePerFrameDescriptorSets(
        VkDescriptorPool descriptorPool,
        VkDescriptorSetLayout descriptorSetLayout,
        std::span<VkDescriptorSet> outDescriptorSets,
        const char* failureContext,
        const char* debugNamePrefix
    );
    bool createDescriptorSetLayout(
        std::span<const VkDescriptorSetLayoutBinding> bindings,
        VkDescriptorSetLayout& outDescriptorSetLayout,
        const char* failureContext,
        const char* debugName,
        const void* pNext = nullptr
    );
    bool createDescriptorPool(
        std::span<const VkDescriptorPoolSize> poolSizes,
        uint32_t maxSets,
        VkDescriptorPool& outDescriptorPool,
        const char* failureContext,
        const char* debugName,
        VkDescriptorPoolCreateFlags flags = 0
    );
    bool createComputePipelineLayout(
        VkDescriptorSetLayout descriptorSetLayout,
        std::span<const VkPushConstantRange> pushConstantRanges,
        VkPipelineLayout& outPipelineLayout,
        const char* failureContext,
        const char* debugName
    );
    bool createComputePipeline(
        VkPipelineLayout pipelineLayout,
        VkShaderModule shaderModule,
        VkPipeline& outPipeline,
        const char* failureContext,
        const char* debugName
    );
    void setObjectName(VkObjectType objectType, uint64_t objectHandle, const char* name) const;
    void beginDebugLabel(VkCommandBuffer commandBuffer, const char* name, float r, float g, float b, float a = 1.0f) const;
    void endDebugLabel(VkCommandBuffer commandBuffer) const;
    void insertDebugLabel(VkCommandBuffer commandBuffer, const char* name, float r, float g, float b, float a = 1.0f) const;
    void readGpuTimestampResults(uint32_t frameIndex);
    void scheduleBufferRelease(BufferHandle handle, uint64_t timelineValue);
    void collectCompletedBufferReleases();
    void updateDisplayTimingStats();
    bool isTimelineValueReached(uint64_t value) const;
    struct DeferredBufferRelease {
        BufferHandle handle = kInvalidBufferHandle;
        uint64_t timelineValue = 0;
    };

    struct ChunkDrawRange {
        uint32_t firstIndex = 0;
        int32_t vertexOffset = 0;
        uint32_t indexCount = 0;
        float offsetX = 0.0f;
        float offsetY = 0.0f;
        float offsetZ = 0.0f;
    };

    struct PipeVertex {
        float position[3];
        float normal[3];
    };

    struct PipeInstance {
        float originLength[4];
        float axisRadius[4];
        float tint[4];
        float extensions[4];
    };

    struct ReadyMagicaDraw {
        VkBuffer vertexBuffer = VK_NULL_HANDLE;
        VkBuffer indexBuffer = VK_NULL_HANDLE;
        std::uint32_t indexCount = 0;
        float offsetX = 0.0f;
        float offsetY = 0.0f;
        float offsetZ = 0.0f;
    };

    struct FrameInstanceDrawData {
        uint32_t pipeInstanceCount = 0;
        std::optional<FrameArenaSlice> pipeInstanceSliceOpt = std::nullopt;
        uint32_t transportInstanceCount = 0;
        std::optional<FrameArenaSlice> transportInstanceSliceOpt = std::nullopt;
        uint32_t beltCargoInstanceCount = 0;
        std::optional<FrameArenaSlice> beltCargoInstanceSliceOpt = std::nullopt;
        std::vector<ReadyMagicaDraw> readyMagicaDraws;
    };

    struct FrameChunkDrawData {
        bool canDrawChunksIndirect = false;
        std::array<bool, kShadowCascadeCount> canDrawShadowChunksIndirectByCascade{};
        std::optional<FrameArenaSlice> chunkInstanceSliceOpt = std::nullopt;
        std::optional<FrameArenaSlice> chunkIndirectSliceOpt = std::nullopt;
        std::optional<FrameArenaSlice> shadowChunkInstanceSliceOpt = std::nullopt;
        std::array<std::optional<FrameArenaSlice>, kShadowCascadeCount> shadowCascadeIndirectSliceOpts{};
        VkBuffer chunkInstanceBuffer = VK_NULL_HANDLE;
        VkBuffer chunkIndirectBuffer = VK_NULL_HANDLE;
        VkBuffer shadowChunkInstanceBuffer = VK_NULL_HANDLE;
        std::array<VkBuffer, kShadowCascadeCount> shadowCascadeIndirectBuffers{};
        std::array<uint32_t, kShadowCascadeCount> shadowCascadeIndirectDrawCounts{};
        uint32_t chunkIndirectDrawCount = 0;
    };

    struct GrassBillboardVertex {
        float corner[2];
        float uv[2];
        float plane;
    };

    struct GrassBillboardInstance {
        float worldPosYaw[4];
        float colorTint[4];
    };

    struct MagicaMeshDraw {
        BufferHandle vertexBufferHandle = kInvalidBufferHandle;
        BufferHandle indexBufferHandle = kInvalidBufferHandle;
        std::uint32_t indexCount = 0;
        float offsetX = 0.0f;
        float offsetY = 0.0f;
        float offsetZ = 0.0f;
    };

    FrameInstanceDrawData prepareFrameInstanceDrawData(
        const voxelsprout::sim::Simulation& simulation,
        float simulationAlpha
    );

    FrameChunkDrawData prepareFrameChunkDrawData(
        const std::vector<voxelsprout::world::Chunk>& chunks,
        std::span<const std::size_t> visibleChunkIndices,
        const std::array<voxelsprout::math::Matrix4, kShadowCascadeCount>& lightViewProjMatrices,
        int cameraChunkX,
        int cameraChunkY,
        int cameraChunkZ
    );
    void drawIndirectChunkRanges(
        VkCommandBuffer commandBuffer,
        std::uint32_t& passDrawCounter,
        const FrameChunkDrawData& frameChunkDrawData
    );
    void drawIndirectShadowChunkRanges(
        VkCommandBuffer commandBuffer,
        std::uint32_t& passDrawCounter,
        std::uint32_t cascadeIndex,
        const FrameChunkDrawData& frameChunkDrawData
    );
    void recordShadowAtlasPass(
        VkCommandBuffer commandBuffer,
        VkQueryPool gpuTimestampQueryPool,
        CoreFrameGraphOrderValidator& coreFramePassOrderValidator,
        const CoreFrameGraphPlan& coreFrameGraphPlan,
        const BoundDescriptorSets& boundDescriptorSets,
        uint32_t mvpDynamicOffset,
        const FrameChunkDrawData& frameChunkDrawData,
        const std::optional<FrameArenaSlice>& chunkInstanceSliceOpt,
        const std::optional<FrameArenaSlice>& shadowChunkInstanceSliceOpt,
        VkBuffer chunkInstanceBuffer,
        VkBuffer shadowChunkInstanceBuffer,
        VkBuffer chunkVertexBuffer,
        VkBuffer chunkIndexBuffer,
        bool canDrawMagica,
        std::span<const ReadyMagicaDraw> readyMagicaDraws,
        uint32_t pipeInstanceCount,
        const std::optional<FrameArenaSlice>& pipeInstanceSliceOpt,
        uint32_t transportInstanceCount,
        const std::optional<FrameArenaSlice>& transportInstanceSliceOpt,
        uint32_t beltCargoInstanceCount,
        const std::optional<FrameArenaSlice>& beltCargoInstanceSliceOpt
    );
    void recordNormalDepthPrepass(
        VkCommandBuffer commandBuffer,
        VkQueryPool gpuTimestampQueryPool,
        CoreFrameGraphOrderValidator& coreFramePassOrderValidator,
        const CoreFrameGraphPlan& coreFrameGraphPlan,
        uint32_t aoFrameIndex,
        uint32_t imageIndex,
        VkExtent2D aoExtent,
        const VkViewport& aoViewport,
        const VkRect2D& aoScissor,
        const BoundDescriptorSets& boundDescriptorSets,
        uint32_t mvpDynamicOffset,
        const FrameChunkDrawData& frameChunkDrawData,
        const std::optional<FrameArenaSlice>& chunkInstanceSliceOpt,
        VkBuffer chunkInstanceBuffer,
        VkBuffer chunkVertexBuffer,
        VkBuffer chunkIndexBuffer,
        bool canDrawMagica,
        std::span<const ReadyMagicaDraw> readyMagicaDraws,
        uint32_t pipeInstanceCount,
        const std::optional<FrameArenaSlice>& pipeInstanceSliceOpt,
        uint32_t transportInstanceCount,
        const std::optional<FrameArenaSlice>& transportInstanceSliceOpt,
        uint32_t beltCargoInstanceCount,
        const std::optional<FrameArenaSlice>& beltCargoInstanceSliceOpt
    );

    GLFWwindow* m_window = nullptr;

    // Global Vulkan API root object.
    // Future renderer versions can add debug utils and extra instance extensions.
    VkInstance m_instance = VK_NULL_HANDLE;
    // Connection between GLFW window and Vulkan presentation.
    // Future multi-window tooling can own multiple surfaces.
    VkSurfaceKHR m_surface = VK_NULL_HANDLE;
    // Selected GPU used for rendering and present support.
    // Future device selection may become score-based for features/perf tiers.
    VkPhysicalDevice m_physicalDevice = VK_NULL_HANDLE;
    // Logical device with a graphics queue (draw+present) and a transfer queue.
    VkDevice m_device = VK_NULL_HANDLE;
    bool m_debugUtilsEnabled = false;
    PFN_vkSetDebugUtilsObjectNameEXT m_setDebugUtilsObjectName = nullptr;
    PFN_vkCmdBeginDebugUtilsLabelEXT m_cmdBeginDebugUtilsLabel = nullptr;
    PFN_vkCmdEndDebugUtilsLabelEXT m_cmdEndDebugUtilsLabel = nullptr;
    PFN_vkCmdInsertDebugUtilsLabelEXT m_cmdInsertDebugUtilsLabel = nullptr;
    uint32_t m_graphicsQueueFamilyIndex = 0;
    uint32_t m_graphicsQueueIndex = 0;
    uint32_t m_transferQueueFamilyIndex = 0;
    uint32_t m_transferQueueIndex = 0;
    VkQueue m_graphicsQueue = VK_NULL_HANDLE;
    VkQueue m_transferQueue = VK_NULL_HANDLE;

    // Presentable image chain for the window.
    // Future render-graph integration can manage this as a backend target.
    VkSwapchainKHR m_swapchain = VK_NULL_HANDLE;
    VkFormat m_swapchainFormat = VK_FORMAT_UNDEFINED;
    VkExtent2D m_swapchainExtent{};
    VkExtent2D m_aoExtent{};
    VkFormat m_depthFormat = VK_FORMAT_UNDEFINED;
    VkFormat m_shadowDepthFormat = VK_FORMAT_UNDEFINED;
    VkFormat m_normalDepthFormat = VK_FORMAT_UNDEFINED;
    VkFormat m_ssaoFormat = VK_FORMAT_UNDEFINED;
    VkFormat m_voxelGiFormat = VK_FORMAT_UNDEFINED;
    VkFormat m_voxelGiOccupancyFormat = VK_FORMAT_UNDEFINED;
    std::vector<VkImage> m_swapchainImages;
    std::vector<VkImageView> m_swapchainImageViews;
    std::vector<bool> m_swapchainImageInitialized;
    std::vector<VkImage> m_msaaColorImages;
    std::vector<VkDeviceMemory> m_msaaColorImageMemories;
    std::vector<VkImageView> m_msaaColorImageViews;
    std::vector<bool> m_msaaColorImageInitialized;
    std::vector<VmaAllocation> m_msaaColorImageAllocations;
    std::vector<VkImage> m_hdrResolveImages;
    std::vector<VkDeviceMemory> m_hdrResolveImageMemories;
    std::vector<VkImageView> m_hdrResolveImageViews;
    std::vector<VkImageView> m_hdrResolveSampleImageViews;
    std::vector<TransientImageHandle> m_hdrResolveTransientHandles;
    std::vector<bool> m_hdrResolveImageInitialized;
    uint32_t m_hdrResolveMipLevels = 1;
    VkSampler m_hdrResolveSampler = VK_NULL_HANDLE;
    std::vector<VkImage> m_depthImages;
    std::vector<VkDeviceMemory> m_depthImageMemories;
    std::vector<VkImageView> m_depthImageViews;
    std::vector<VmaAllocation> m_depthImageAllocations;
    std::vector<VkImage> m_normalDepthImages;
    std::vector<VkDeviceMemory> m_normalDepthImageMemories;
    std::vector<VkImageView> m_normalDepthImageViews;
    std::vector<TransientImageHandle> m_normalDepthTransientHandles;
    std::vector<bool> m_normalDepthImageInitialized;
    std::vector<VkImage> m_aoDepthImages;
    std::vector<VkDeviceMemory> m_aoDepthImageMemories;
    std::vector<VkImageView> m_aoDepthImageViews;
    std::vector<TransientImageHandle> m_aoDepthTransientHandles;
    std::vector<bool> m_aoDepthImageInitialized;
    std::vector<VkImage> m_ssaoRawImages;
    std::vector<VkDeviceMemory> m_ssaoRawImageMemories;
    std::vector<VkImageView> m_ssaoRawImageViews;
    std::vector<TransientImageHandle> m_ssaoRawTransientHandles;
    std::vector<bool> m_ssaoRawImageInitialized;
    std::vector<VkImage> m_ssaoBlurImages;
    std::vector<VkDeviceMemory> m_ssaoBlurImageMemories;
    std::vector<VkImageView> m_ssaoBlurImageViews;
    std::vector<TransientImageHandle> m_ssaoBlurTransientHandles;
    std::vector<bool> m_ssaoBlurImageInitialized;
    std::vector<VkImage> m_sunShaftImages;
    std::vector<VkDeviceMemory> m_sunShaftImageMemories;
    std::vector<VkImageView> m_sunShaftImageViews;
    std::vector<TransientImageHandle> m_sunShaftTransientHandles;
    std::vector<bool> m_sunShaftImageInitialized;
    VkSampler m_normalDepthSampler = VK_NULL_HANDLE;
    VkSampler m_ssaoSampler = VK_NULL_HANDLE;
    VkSampler m_sunShaftSampler = VK_NULL_HANDLE;
    VkImage m_shadowDepthImage = VK_NULL_HANDLE;
    VkImageView m_shadowDepthImageView = VK_NULL_HANDLE;
    VkSampler m_shadowDepthSampler = VK_NULL_HANDLE;
    bool m_shadowDepthInitialized = false;
    std::array<VkImage, 2> m_voxelGiImages{};
    std::array<VkImageView, 2> m_voxelGiImageViews{};
    std::array<VkDeviceMemory, 2> m_voxelGiImageMemories{};
    std::array<VkImage, 6> m_voxelGiSurfaceFaceImages{};
    std::array<VkImageView, 6> m_voxelGiSurfaceFaceImageViews{};
    std::array<VkDeviceMemory, 6> m_voxelGiSurfaceFaceMemories{};
    VkImage m_voxelGiSkyExposureImage = VK_NULL_HANDLE;
    VkImageView m_voxelGiSkyExposureImageView = VK_NULL_HANDLE;
    VkDeviceMemory m_voxelGiSkyExposureMemory = VK_NULL_HANDLE;
    VkSampler m_voxelGiSampler = VK_NULL_HANDLE;
    bool m_voxelGiInitialized = false;
    bool m_voxelGiComputeAvailable = false;
    bool m_voxelGiSkyExposureInitialized = false;
    VkImage m_voxelGiOccupancyImage = VK_NULL_HANDLE;
    VkImageView m_voxelGiOccupancyImageView = VK_NULL_HANDLE;
    VkDeviceMemory m_voxelGiOccupancyMemory = VK_NULL_HANDLE;
    VkSampler m_voxelGiOccupancySampler = VK_NULL_HANDLE;
    bool m_voxelGiOccupancyInitialized = false;
    bool m_voxelGiWorldDirty = true;
    bool m_voxelGiHasPreviousFrameState = false;
    std::array<float, 3> m_voxelGiPreviousGridOrigin{0.0f, 0.0f, 0.0f};
    std::array<float, 3> m_voxelGiPreviousSunDirection{0.0f, 0.0f, 0.0f};
    std::array<float, 3> m_voxelGiPreviousSunColor{0.0f, 0.0f, 0.0f};
    std::array<std::array<float, 3>, 9> m_voxelGiPreviousShIrradiance{};
    float m_voxelGiPreviousBounceStrength = 0.0f;
    float m_voxelGiPreviousDiffusionSoftness = 0.0f;
    BufferHandle m_autoExposureHistogramBufferHandle = kInvalidBufferHandle;
    BufferHandle m_autoExposureStateBufferHandle = kInvalidBufferHandle;
    bool m_autoExposureComputeAvailable = false;
    bool m_autoExposureHistoryValid = false;
    bool m_sunShaftComputeAvailable = false;
    bool m_sunShaftShaderAvailable = false;
    VkDescriptorSetLayout m_autoExposureDescriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool m_autoExposureDescriptorPool = VK_NULL_HANDLE;
    std::array<VkDescriptorSet, kMaxFramesInFlight> m_autoExposureDescriptorSets{};
    VkPipelineLayout m_autoExposurePipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_autoExposureHistogramPipeline = VK_NULL_HANDLE;
    VkPipeline m_autoExposureUpdatePipeline = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_sunShaftDescriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool m_sunShaftDescriptorPool = VK_NULL_HANDLE;
    std::array<VkDescriptorSet, kMaxFramesInFlight> m_sunShaftDescriptorSets{};
    VkPipelineLayout m_sunShaftPipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_sunShaftPipeline = VK_NULL_HANDLE;
    VmaAllocator m_vmaAllocator = VK_NULL_HANDLE;
    VmaAllocation m_shadowDepthAllocation = VK_NULL_HANDLE;
    VmaAllocation m_diffuseTextureAllocation = VK_NULL_HANDLE;
    std::array<VmaAllocation, 2> m_voxelGiImageAllocations{};
    std::array<VmaAllocation, 6> m_voxelGiSurfaceFaceAllocations{};
    VmaAllocation m_voxelGiSkyExposureAllocation = VK_NULL_HANDLE;
    VmaAllocation m_voxelGiOccupancyAllocation = VK_NULL_HANDLE;
    VkDeviceMemory m_shadowDepthMemory = VK_NULL_HANDLE;
    std::vector<uint64_t> m_swapchainImageTimelineValues;
    // One render-finished semaphore per swapchain image avoids reusing a semaphore
    // while presentation may still be waiting on it.
    std::vector<VkSemaphore> m_renderFinishedSemaphores;
    VkSampleCountFlagBits m_colorSampleCount = VK_SAMPLE_COUNT_4_BIT;
    VkFormat m_hdrColorFormat = VK_FORMAT_UNDEFINED;

    // Pipeline and descriptor lifetimes are owned by focused managers.
    PipelineManager m_pipelineManager{};
    DescriptorManager<kMaxFramesInFlight> m_descriptorManager{};
    FrameGraph m_frameGraph{};

    // Existing Renderer call sites use these aliases while ownership lives in managers.
    VkPipelineLayout& m_pipelineLayout = m_pipelineManager.pipelineLayout;
    VkPipeline& m_pipeline = m_pipelineManager.pipeline;
    VkPipeline& m_shadowPipeline = m_pipelineManager.shadowPipeline;
    VkPipeline& m_pipeShadowPipeline = m_pipelineManager.pipeShadowPipeline;
    VkPipeline& m_grassBillboardShadowPipeline = m_pipelineManager.grassBillboardShadowPipeline;
    VkPipeline& m_skyboxPipeline = m_pipelineManager.skyboxPipeline;
    VkPipeline& m_tonemapPipeline = m_pipelineManager.tonemapPipeline;
    VkPipeline& m_pipePipeline = m_pipelineManager.pipePipeline;
    VkPipeline& m_grassBillboardPipeline = m_pipelineManager.grassBillboardPipeline;
    VkPipeline& m_voxelNormalDepthPipeline = m_pipelineManager.voxelNormalDepthPipeline;
    VkPipeline& m_pipeNormalDepthPipeline = m_pipelineManager.pipeNormalDepthPipeline;
    VkPipeline& m_grassBillboardNormalDepthPipeline = m_pipelineManager.grassBillboardNormalDepthPipeline;
    VkPipeline& m_magicaPipeline = m_pipelineManager.magicaPipeline;
    VkPipeline& m_ssaoPipeline = m_pipelineManager.ssaoPipeline;
    VkPipeline& m_ssaoBlurPipeline = m_pipelineManager.ssaoBlurPipeline;
    VkPipeline& m_previewAddPipeline = m_pipelineManager.previewAddPipeline;
    VkPipeline& m_previewRemovePipeline = m_pipelineManager.previewRemovePipeline;
    VkPipelineLayout& m_voxelGiPipelineLayout = m_pipelineManager.voxelGiPipelineLayout;
    VkPipeline& m_voxelGiSurfacePipeline = m_pipelineManager.voxelGiSurfacePipeline;
    VkPipeline& m_voxelGiSkyExposurePipeline = m_pipelineManager.voxelGiSkyExposurePipeline;
    VkPipeline& m_voxelGiInjectPipeline = m_pipelineManager.voxelGiInjectPipeline;
    VkPipeline& m_voxelGiPropagatePipeline = m_pipelineManager.voxelGiPropagatePipeline;
    VkDescriptorSetLayout& m_descriptorSetLayout = m_descriptorManager.descriptorSetLayout;
    VkDescriptorSetLayout& m_bindlessDescriptorSetLayout = m_descriptorManager.bindlessDescriptorSetLayout;
    VkDescriptorSetLayout& m_voxelGiDescriptorSetLayout = m_descriptorManager.voxelGiDescriptorSetLayout;
    VkDescriptorPool& m_descriptorPool = m_descriptorManager.descriptorPool;
    VkDescriptorPool& m_bindlessDescriptorPool = m_descriptorManager.bindlessDescriptorPool;
    VkDescriptorPool& m_voxelGiDescriptorPool = m_descriptorManager.voxelGiDescriptorPool;
    std::array<VkDescriptorSet, kMaxFramesInFlight>& m_descriptorSets = m_descriptorManager.descriptorSets;
    std::array<VkDescriptorSet, kMaxFramesInFlight>& m_voxelGiDescriptorSets = m_descriptorManager.voxelGiDescriptorSets;
    VkDescriptorSet& m_bindlessDescriptorSet = m_descriptorManager.bindlessDescriptorSet;
    bool m_supportsWireframePreview = false;
    bool m_supportsSamplerAnisotropy = false;
    bool m_supportsMultiDrawIndirect = false;
    bool m_supportsBindlessDescriptors = false;
    bool m_supportsDisplayTiming = false;
    bool m_hasDisplayTimingExtension = false;
    bool m_enableDisplayTiming = false;
    uint32_t m_bindlessTextureCapacity = 0;
    bool m_gpuTimestampsSupported = false;
    float m_gpuTimestampPeriodNs = 0.0f;
    std::array<VkQueryPool, kMaxFramesInFlight> m_gpuTimestampQueryPools{};
    float m_maxSamplerAnisotropy = 1.0f;
    VkDeviceSize m_uniformBufferAlignment = 256;

    // Static mesh buffers per chunk draw range.
    // Future chunk streaming can replace this with sparse streaming allocations.
    BufferAllocator m_bufferAllocator;
    FrameArena m_frameArena;
    BufferHandle m_previewVertexBufferHandle = kInvalidBufferHandle;
    BufferHandle m_previewIndexBufferHandle = kInvalidBufferHandle;
    BufferHandle m_chunkVertexBufferHandle = kInvalidBufferHandle;
    BufferHandle m_chunkIndexBufferHandle = kInvalidBufferHandle;
    BufferHandle m_pipeVertexBufferHandle = kInvalidBufferHandle;
    BufferHandle m_pipeIndexBufferHandle = kInvalidBufferHandle;
    BufferHandle m_transportVertexBufferHandle = kInvalidBufferHandle;
    BufferHandle m_transportIndexBufferHandle = kInvalidBufferHandle;
    BufferHandle m_grassBillboardVertexBufferHandle = kInvalidBufferHandle;
    BufferHandle m_grassBillboardIndexBufferHandle = kInvalidBufferHandle;
    BufferHandle m_grassBillboardInstanceBufferHandle = kInvalidBufferHandle;
    std::vector<DeferredBufferRelease> m_deferredBufferReleases;
    std::vector<ChunkDrawRange> m_chunkDrawRanges;
    std::vector<voxelsprout::world::ChunkLodMeshes> m_chunkLodMeshCache;
    std::vector<std::vector<GrassBillboardInstance>> m_chunkGrassInstanceCache;
    std::vector<MagicaMeshDraw> m_magicaMeshDraws;
    bool m_chunkLodMeshCacheValid = false;
    voxelsprout::world::MeshingOptions m_chunkMeshingOptions{};
    bool m_chunkMeshRebuildRequested = false;
    std::vector<std::size_t> m_pendingChunkRemeshIndices;
    uint32_t m_previewIndexCount = 0;
    uint32_t m_pipeIndexCount = 0;
    uint32_t m_transportIndexCount = 0;
    uint32_t m_grassBillboardIndexCount = 0;
    uint32_t m_grassBillboardInstanceCount = 0;
    std::array<std::uint32_t, 16> m_voxelBaseColorPaletteRgba{};
    VkImage m_diffuseTextureImage = VK_NULL_HANDLE;
    VkDeviceMemory m_diffuseTextureMemory = VK_NULL_HANDLE;
    VkImageView m_diffuseTextureImageView = VK_NULL_HANDLE;
    VkSampler m_diffuseTextureSampler = VK_NULL_HANDLE;
    VkSampler m_diffuseTexturePlantSampler = VK_NULL_HANDLE;

    std::array<FrameResources, kMaxFramesInFlight> m_frames{};
    VkCommandPool m_transferCommandPool = VK_NULL_HANDLE;
    VkCommandBuffer m_transferCommandBuffer = VK_NULL_HANDLE;
    std::array<uint64_t, kMaxFramesInFlight> m_frameTimelineValues{};
    VkSemaphore m_renderTimelineSemaphore = VK_NULL_HANDLE;
    PFN_vkGetRefreshCycleDurationGOOGLE m_getRefreshCycleDurationGoogle = nullptr;
    PFN_vkGetPastPresentationTimingGOOGLE m_getPastPresentationTimingGoogle = nullptr;
    uint64_t m_pendingTransferTimelineValue = 0;
    uint64_t m_currentChunkReadyTimelineValue = 0;
    uint64_t m_transferCommandBufferInFlightValue = 0;
    uint64_t m_lastGraphicsTimelineValue = 0;
    uint64_t m_nextTimelineValue = 1;
    uint32_t m_nextDisplayTimingPresentId = 1;
    uint32_t m_lastSubmittedDisplayTimingPresentId = 0;
    uint32_t m_lastPresentedDisplayTimingPresentId = 0;
    uint32_t m_currentFrame = 0;
    bool m_debugUiVisible = false;
    bool m_showFrameStatsPanel = false;
    bool m_showMeshingPanel = false;
    bool m_showShadowPanel = false;
    bool m_showSunPanel = false;
    float m_debugCameraFovDegrees = 90.0f;
    bool m_debugCameraFovInitialized = false;
    bool m_debugEnableVertexAo = true;
    bool m_debugEnableSsao = true;
    bool m_debugVisualizeSsao = false;
    bool m_debugVisualizeAoNormals = false;
    ShadowDebugSettings m_shadowDebugSettings{};
    SkyDebugSettings m_skyDebugSettings{};
    VoxelGiDebugSettings m_voxelGiDebugSettings{};
    struct SkyTuningRuntimeState {
        bool initialized = false;
        float rayleighStrength = 1.0f;
        float mieStrength = 1.0f;
        float mieAnisotropy = 0.55f;
        float skyExposure = 1.0f;
        float sunDiskIntensity = 1150.0f;
        float sunHaloIntensity = 22.0f;
        float sunDiskSize = 2.0f;
        float sunHazeFalloff = 0.35f;
    } m_skyTuningRuntime{};
    bool m_imguiInitialized = false;
    VkDescriptorPool m_imguiDescriptorPool = VK_NULL_HANDLE;
    double m_lastFrameTimestampSeconds = 0.0;
    float m_debugFrameTimeMs = 0.0f;
    float m_debugGpuFrameTimeMs = 0.0f;
    float m_debugGpuShadowTimeMs = 0.0f;
    float m_debugGpuGiInjectTimeMs = 0.0f;
    float m_debugGpuGiPropagateTimeMs = 0.0f;
    float m_debugGpuAutoExposureTimeMs = 0.0f;
    float m_debugGpuSunShaftTimeMs = 0.0f;
    float m_debugGpuPrepassTimeMs = 0.0f;
    float m_debugGpuSsaoTimeMs = 0.0f;
    float m_debugGpuSsaoBlurTimeMs = 0.0f;
    float m_debugGpuMainTimeMs = 0.0f;
    float m_debugGpuPostTimeMs = 0.0f;
    float m_debugDisplayRefreshMs = 0.0f;
    float m_debugDisplayPresentMarginMs = 0.0f;
    float m_debugDisplayActualEarliestDeltaMs = 0.0f;
    std::uint32_t m_debugDisplayTimingSampleCount = 0;
    std::array<float, kTimingHistorySampleCount> m_debugCpuFrameTotalMsHistory{};
    std::array<float, kTimingHistorySampleCount> m_debugCpuFrameWorkMsHistory{};
    std::array<float, kTimingHistorySampleCount> m_debugCpuFrameEwmaMsHistory{};
    std::uint32_t m_debugCpuFrameTimingMsHistoryWrite = 0;
    std::uint32_t m_debugCpuFrameTimingMsHistoryCount = 0;
    float m_debugCpuFrameWorkMs = 0.0f;
    float m_debugCpuFrameEwmaMs = 0.0f;
    bool m_debugCpuFrameEwmaInitialized = false;
    std::array<float, kTimingHistorySampleCount> m_debugGpuFrameTimingMsHistory{};
    std::uint32_t m_debugGpuFrameTimingMsHistoryWrite = 0;
    std::uint32_t m_debugGpuFrameTimingMsHistoryCount = 0;
    float m_debugFps = 0.0f;
    std::uint32_t m_debugChunkCount = 0;
    std::uint32_t m_debugMacroCellUniformCount = 0;
    std::uint32_t m_debugMacroCellRefined4Count = 0;
    std::uint32_t m_debugMacroCellRefined1Count = 0;
    std::uint32_t m_debugDrawnLod0Ranges = 0;
    std::uint32_t m_debugDrawnLod1Ranges = 0;
    std::uint32_t m_debugDrawnLod2Ranges = 0;
    bool m_debugEnableSpatialQueries = true;
    voxelsprout::world::ClipmapConfig m_debugClipmapConfig{};
    bool m_debugSpatialQueriesUsed = false;
    voxelsprout::world::SpatialQueryStats m_debugSpatialQueryStats{};
    std::uint32_t m_debugSpatialVisibleChunkCount = 0;
    std::uint32_t m_debugChunkIndirectCommandCount = 0;
    std::uint32_t m_debugDrawCallsTotal = 0;
    std::uint32_t m_debugDrawCallsShadow = 0;
    std::uint32_t m_debugDrawCallsPrepass = 0;
    std::uint32_t m_debugDrawCallsMain = 0;
    std::uint32_t m_debugDrawCallsPost = 0;
    std::uint32_t m_debugChunkMeshVertexCount = 0;
    std::uint32_t m_debugChunkMeshIndexCount = 0;
    std::uint32_t m_debugChunkLastRemeshedChunkCount = 0;
    std::uint32_t m_debugChunkLastRemeshActiveVertexCount = 0;
    std::uint32_t m_debugChunkLastRemeshActiveIndexCount = 0;
    std::uint32_t m_debugChunkLastRemeshNaiveVertexCount = 0;
    std::uint32_t m_debugChunkLastRemeshNaiveIndexCount = 0;
    float m_debugChunkLastRemeshReductionPercent = 0.0f;
    float m_debugChunkLastRemeshMs = 0.0f;
    float m_debugChunkLastFullRemeshMs = 0.0f;
    std::uint64_t m_debugFrameArenaUploadBytes = 0;
    std::uint32_t m_debugFrameArenaUploadAllocs = 0;
    std::uint64_t m_debugFrameArenaTransientBufferBytes = 0;
    std::uint32_t m_debugFrameArenaTransientBufferCount = 0;
    std::uint64_t m_debugFrameArenaTransientImageBytes = 0;
    std::uint32_t m_debugFrameArenaTransientImageCount = 0;
    std::uint32_t m_debugFrameArenaAliasReuses = 0;
    std::uint64_t m_debugFrameArenaResidentBufferBytes = 0;
    std::uint32_t m_debugFrameArenaResidentBufferCount = 0;
    std::uint64_t m_debugFrameArenaResidentImageBytes = 0;
    std::uint32_t m_debugFrameArenaResidentImageCount = 0;
    std::uint32_t m_debugFrameArenaResidentAliasReuses = 0;
    std::vector<FrameArenaAliasedImageInfo> m_debugAliasedImages;
    // Dynamic cascade split distances in view-space units.
    // Updated per frame and consumed by shadow rendering + shading.
    std::array<float, kShadowCascadeCount> m_shadowCascadeSplits = {20.0f, 45.0f, 90.0f, 180.0f};
    std::array<float, kShadowCascadeCount> m_shadowStableCascadeRadii = {0.0f, 0.0f, 0.0f, 0.0f};
    float m_shadowStableAspectRatio = -1.0f;
    float m_shadowStableFovDegrees = -1.0f;
};

} // namespace voxelsprout::render
