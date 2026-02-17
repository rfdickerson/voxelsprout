#include "render/backend/vulkan/renderer_backend.h"

#include "core/log.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <type_traits>
#include <vector>

namespace voxelsprout::render {

namespace {

template <typename VkHandleT>
uint64_t vkHandleToUint64(VkHandleT handle) {
    if constexpr (std::is_pointer_v<VkHandleT>) {
        return reinterpret_cast<uint64_t>(handle);
    } else {
        return static_cast<uint64_t>(handle);
    }
}

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
        VOX_LOGE("render") << "missing shader file for " << debugName << ": "
                           << (filePath != nullptr ? filePath : "<null>") << "\n";
        return false;
    }
    if ((shaderFileData->size() % sizeof(std::uint32_t)) != 0) {
        VOX_LOGE("render") << "invalid SPIR-V byte size for " << debugName << ": " << filePath << "\n";
        return false;
    }

    const std::uint32_t* code = reinterpret_cast<const std::uint32_t*>(shaderFileData->data());
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = shaderFileData->size();
    createInfo.pCode = code;

    const VkResult result = vkCreateShaderModule(device, &createInfo, nullptr, &outShaderModule);
    if (result != VK_SUCCESS) {
        logVkFailure("vkCreateShaderModule(fileOrFallback)", result);
        return false;
    }
    return true;
}

struct ShaderModuleLoadSpec {
    const char* filePath = nullptr;
    const char* debugName = nullptr;
};

void destroyShaderModules(VkDevice device, std::span<const VkShaderModule> shaderModules) {
    for (const VkShaderModule shaderModule : shaderModules) {
        if (shaderModule != VK_NULL_HANDLE) {
            vkDestroyShaderModule(device, shaderModule, nullptr);
        }
    }
}

bool createShaderModulesFromFiles(
    VkDevice device,
    std::span<const ShaderModuleLoadSpec> loadSpecs,
    std::span<VkShaderModule> outShaderModules
) {
    if (loadSpecs.size() != outShaderModules.size()) {
        VOX_LOGE("render") << "createShaderModulesFromFiles argument mismatch: specs=" << loadSpecs.size()
                           << ", outputs=" << outShaderModules.size();
        return false;
    }

    std::fill(outShaderModules.begin(), outShaderModules.end(), VK_NULL_HANDLE);
    for (std::size_t i = 0; i < loadSpecs.size(); ++i) {
        const ShaderModuleLoadSpec& spec = loadSpecs[i];
        if (!createShaderModuleFromFile(device, spec.filePath, spec.debugName, outShaderModules[i])) {
            destroyShaderModules(device, outShaderModules);
            std::fill(outShaderModules.begin(), outShaderModules.end(), VK_NULL_HANDLE);
            return false;
        }
    }
    return true;
}

struct alignas(16) ChunkInstanceData {
    float chunkOffset[4];
};

struct alignas(16) ChunkPushConstants {
    float chunkOffset[4];
    float cascadeData[4];
};

} // namespace

bool RendererBackend::createMagicaPipeline() {
    if (m_pipelineLayout == VK_NULL_HANDLE) {
        return false;
    }
    if (m_depthFormat == VK_FORMAT_UNDEFINED || m_hdrColorFormat == VK_FORMAT_UNDEFINED) {
        return false;
    }

    constexpr const char* kWorldVertexShaderPath = "../src/render/shaders/voxel_packed.vert.slang.spv";
    constexpr const char* kWorldFragmentShaderPath = "../src/render/shaders/voxel_packed.frag.slang.spv";

    std::array<VkShaderModule, 2> shaderModules = {
        VK_NULL_HANDLE,
        VK_NULL_HANDLE
    };
    VkShaderModule& magicaVertShaderModule = shaderModules[0];
    VkShaderModule& magicaFragShaderModule = shaderModules[1];
    const std::array<ShaderModuleLoadSpec, 2> shaderLoadSpecs = {{
        {kWorldVertexShaderPath, "magica.voxel_packed.vert"},
        {kWorldFragmentShaderPath, "magica.voxel_packed.frag"},
    }};
    if (!createShaderModulesFromFiles(m_device, shaderLoadSpecs, shaderModules)) {
        return false;
    }

    VkPipelineShaderStageCreateInfo vertexShaderStage{};
    vertexShaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertexShaderStage.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertexShaderStage.module = magicaVertShaderModule;
    vertexShaderStage.pName = "main";

    VkPipelineShaderStageCreateInfo fragmentShaderStage{};
    fragmentShaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragmentShaderStage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragmentShaderStage.module = magicaFragShaderModule;
    fragmentShaderStage.pName = "main";
    struct WorldFragmentSpecializationData {
        std::int32_t shadowPolicyMode = 2;
        std::int32_t ambientPolicyMode = 2;
        std::int32_t forceTintOnly = 1;
    };
    const WorldFragmentSpecializationData fragmentSpecializationData{};
    const std::array<VkSpecializationMapEntry, 3> specializationMapEntries = {{
        VkSpecializationMapEntry{
            6u,
            static_cast<uint32_t>(offsetof(WorldFragmentSpecializationData, shadowPolicyMode)),
            sizeof(std::int32_t)
        },
        VkSpecializationMapEntry{
            7u,
            static_cast<uint32_t>(offsetof(WorldFragmentSpecializationData, ambientPolicyMode)),
            sizeof(std::int32_t)
        },
        VkSpecializationMapEntry{
            8u,
            static_cast<uint32_t>(offsetof(WorldFragmentSpecializationData, forceTintOnly)),
            sizeof(std::int32_t)
        }
    }};
    const VkSpecializationInfo specializationInfo{
        static_cast<uint32_t>(specializationMapEntries.size()),
        specializationMapEntries.data(),
        sizeof(fragmentSpecializationData),
        &fragmentSpecializationData
    };
    fragmentShaderStage.pSpecializationInfo = &specializationInfo;

    const std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages = {
        vertexShaderStage,
        fragmentShaderStage
    };

    VkVertexInputBindingDescription bindingDescriptions[2]{};
    bindingDescriptions[0].binding = 0;
    bindingDescriptions[0].stride = sizeof(voxelsprout::world::PackedVoxelVertex);
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
        VK_DYNAMIC_STATE_SCISSOR
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
    pipelineCreateInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
    pipelineCreateInfo.pStages = shaderStages.data();
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

    VkPipeline magicaPipeline = VK_NULL_HANDLE;
    const VkResult pipelineResult = vkCreateGraphicsPipelines(
        m_device,
        VK_NULL_HANDLE,
        1,
        &pipelineCreateInfo,
        nullptr,
        &magicaPipeline
    );

    destroyShaderModules(m_device, shaderModules);

    if (pipelineResult != VK_SUCCESS) {
        logVkFailure("vkCreateGraphicsPipelines(magica)", pipelineResult);
        return false;
    }

    if (m_magicaPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_device, m_magicaPipeline, nullptr);
    }
    m_magicaPipeline = magicaPipeline;
    setObjectName(VK_OBJECT_TYPE_PIPELINE, vkHandleToUint64(m_magicaPipeline), "pipeline.magicaVoxel");
    VOX_LOGI("render") << "pipeline config (magica): samples=" << static_cast<uint32_t>(m_colorSampleCount)
                       << ", cullMode=" << static_cast<uint32_t>(rasterizer.cullMode)
                       << ", depthCompare=" << static_cast<uint32_t>(depthStencil.depthCompareOp)
                       << "\n";
    return true;
}

bool RendererBackend::createPipePipeline() {
    if (m_pipelineLayout == VK_NULL_HANDLE) {
        return false;
    }
    if (m_depthFormat == VK_FORMAT_UNDEFINED || m_hdrColorFormat == VK_FORMAT_UNDEFINED) {
        return false;
    }

    constexpr const char* kPipeVertexShaderPath = "../src/render/shaders/pipe_instanced.vert.slang.spv";
    constexpr const char* kPipeFragmentShaderPath = "../src/render/shaders/pipe_instanced.frag.slang.spv";

    std::array<VkShaderModule, 2> pipeShaderModules = {
        VK_NULL_HANDLE,
        VK_NULL_HANDLE
    };
    VkShaderModule& pipeVertShaderModule = pipeShaderModules[0];
    VkShaderModule& pipeFragShaderModule = pipeShaderModules[1];
    const std::array<ShaderModuleLoadSpec, 2> pipeShaderLoadSpecs = {{
        {kPipeVertexShaderPath, "pipe_instanced.vert"},
        {kPipeFragmentShaderPath, "pipe_instanced.frag"},
    }};
    if (!createShaderModulesFromFiles(m_device, pipeShaderLoadSpecs, pipeShaderModules)) {
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

    destroyShaderModules(m_device, pipeShaderModules);

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
    std::array<VkShaderModule, 2> grassShaderModules = {
        VK_NULL_HANDLE,
        VK_NULL_HANDLE
    };
    VkShaderModule& grassVertShaderModule = grassShaderModules[0];
    VkShaderModule& grassFragShaderModule = grassShaderModules[1];
    const std::array<ShaderModuleLoadSpec, 2> grassShaderLoadSpecs = {{
        {kGrassBillboardVertexShaderPath, "grass_billboard.vert"},
        {kGrassBillboardFragmentShaderPath, "grass_billboard.frag"},
    }};
    if (!createShaderModulesFromFiles(m_device, grassShaderLoadSpecs, grassShaderModules)) {
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

    VkVertexInputAttributeDescription grassAttributes[5]{};
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
    grassAttributes[4].location = 4;
    grassAttributes[4].binding = 1;
    grassAttributes[4].format = VK_FORMAT_R32G32B32A32_SFLOAT;
    grassAttributes[4].offset = static_cast<uint32_t>(offsetof(GrassBillboardInstance, colorTint));
    VkPipelineVertexInputStateCreateInfo grassVertexInputInfo{};
    grassVertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    grassVertexInputInfo.vertexBindingDescriptionCount = 2;
    grassVertexInputInfo.pVertexBindingDescriptions = grassBindings;
    grassVertexInputInfo.vertexAttributeDescriptionCount = 5;
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
    destroyShaderModules(m_device, grassShaderModules);
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

bool RendererBackend::createAoPipelines() {
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
    constexpr const char* kGrassBillboardVertShaderPath = "../src/render/shaders/grass_billboard.vert.slang.spv";
    constexpr const char* kGrassBillboardNormalDepthFragShaderPath = "../src/render/shaders/grass_billboard_normaldepth.frag.slang.spv";
    constexpr const char* kFullscreenVertShaderPath = "../src/render/shaders/tone_map.vert.slang.spv";
    constexpr const char* kSsaoFragShaderPath = "../src/render/shaders/ssao.frag.slang.spv";
    constexpr const char* kSsaoBlurFragShaderPath = "../src/render/shaders/ssao_blur.frag.slang.spv";

    std::array<VkShaderModule, 9> shaderModules = {
        VK_NULL_HANDLE,
        VK_NULL_HANDLE,
        VK_NULL_HANDLE,
        VK_NULL_HANDLE,
        VK_NULL_HANDLE,
        VK_NULL_HANDLE,
        VK_NULL_HANDLE,
        VK_NULL_HANDLE,
        VK_NULL_HANDLE
    };
    VkShaderModule& voxelVertShaderModule = shaderModules[0];
    VkShaderModule& voxelNormalDepthFragShaderModule = shaderModules[1];
    VkShaderModule& pipeVertShaderModule = shaderModules[2];
    VkShaderModule& pipeNormalDepthFragShaderModule = shaderModules[3];
    VkShaderModule& grassBillboardVertShaderModule = shaderModules[4];
    VkShaderModule& grassBillboardNormalDepthFragShaderModule = shaderModules[5];
    VkShaderModule& fullscreenVertShaderModule = shaderModules[6];
    VkShaderModule& ssaoFragShaderModule = shaderModules[7];
    VkShaderModule& ssaoBlurFragShaderModule = shaderModules[8];

    const std::array<ShaderModuleLoadSpec, 9> shaderLoadSpecs = {{
        {kVoxelVertShaderPath, "voxel_packed.vert"},
        {kVoxelNormalDepthFragShaderPath, "voxel_normaldepth.frag"},
        {kPipeVertShaderPath, "pipe_instanced.vert"},
        {kPipeNormalDepthFragShaderPath, "pipe_normaldepth.frag"},
        {kGrassBillboardVertShaderPath, "grass_billboard.vert"},
        {kGrassBillboardNormalDepthFragShaderPath, "grass_billboard_normaldepth.frag"},
        {kFullscreenVertShaderPath, "tone_map.vert"},
        {kSsaoFragShaderPath, "ssao.frag"},
        {kSsaoBlurFragShaderPath, "ssao_blur.frag"},
    }};
    if (!createShaderModulesFromFiles(m_device, shaderLoadSpecs, shaderModules)) {
        return false;
    }

    VkPipeline voxelNormalDepthPipeline = VK_NULL_HANDLE;
    VkPipeline pipeNormalDepthPipeline = VK_NULL_HANDLE;
    VkPipeline grassBillboardNormalDepthPipeline = VK_NULL_HANDLE;
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
        if (grassBillboardNormalDepthPipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(m_device, grassBillboardNormalDepthPipeline, nullptr);
            grassBillboardNormalDepthPipeline = VK_NULL_HANDLE;
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
    voxelBindings[0].stride = sizeof(voxelsprout::world::PackedVoxelVertex);
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
        destroyShaderModules(m_device, shaderModules);
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
        destroyShaderModules(m_device, shaderModules);
        return false;
    }

    // Grass billboard normal-depth pipeline.
    VkPipelineShaderStageCreateInfo grassNormalDepthStageInfos[2]{};
    grassNormalDepthStageInfos[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    grassNormalDepthStageInfos[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    grassNormalDepthStageInfos[0].module = grassBillboardVertShaderModule;
    grassNormalDepthStageInfos[0].pName = "main";
    grassNormalDepthStageInfos[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    grassNormalDepthStageInfos[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    grassNormalDepthStageInfos[1].module = grassBillboardNormalDepthFragShaderModule;
    grassNormalDepthStageInfos[1].pName = "main";

    VkVertexInputBindingDescription grassBindings[2]{};
    grassBindings[0].binding = 0;
    grassBindings[0].stride = sizeof(GrassBillboardVertex);
    grassBindings[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    grassBindings[1].binding = 1;
    grassBindings[1].stride = sizeof(GrassBillboardInstance);
    grassBindings[1].inputRate = VK_VERTEX_INPUT_RATE_INSTANCE;

    VkVertexInputAttributeDescription grassAttributes[5]{};
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
    grassAttributes[4].location = 4;
    grassAttributes[4].binding = 1;
    grassAttributes[4].format = VK_FORMAT_R32G32B32A32_SFLOAT;
    grassAttributes[4].offset = static_cast<uint32_t>(offsetof(GrassBillboardInstance, colorTint));

    VkPipelineVertexInputStateCreateInfo grassVertexInputInfo{};
    grassVertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    grassVertexInputInfo.vertexBindingDescriptionCount = 2;
    grassVertexInputInfo.pVertexBindingDescriptions = grassBindings;
    grassVertexInputInfo.vertexAttributeDescriptionCount = 5;
    grassVertexInputInfo.pVertexAttributeDescriptions = grassAttributes;

    VkPipelineRasterizationStateCreateInfo grassRasterizer = rasterizer;
    grassRasterizer.cullMode = VK_CULL_MODE_NONE;

    pipelineCreateInfo.pStages = grassNormalDepthStageInfos;
    pipelineCreateInfo.pVertexInputState = &grassVertexInputInfo;
    pipelineCreateInfo.pRasterizationState = &grassRasterizer;
    const VkResult grassNormalDepthPipelineResult = vkCreateGraphicsPipelines(
        m_device,
        VK_NULL_HANDLE,
        1,
        &pipelineCreateInfo,
        nullptr,
        &grassBillboardNormalDepthPipeline
    );
    if (grassNormalDepthPipelineResult != VK_SUCCESS) {
        logVkFailure("vkCreateGraphicsPipelines(grassBillboardNormalDepth)", grassNormalDepthPipelineResult);
        destroyNewPipelines();
        destroyShaderModules(m_device, shaderModules);
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
        destroyShaderModules(m_device, shaderModules);
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
        destroyShaderModules(m_device, shaderModules);
        return false;
    }
    VOX_LOGI("render") << "pipeline config (ssaoBlur): radius=" << ssaoSpecializationData.blurRadius
              << ", sigma=" << ssaoSpecializationData.blurSigma
              << ", format=" << static_cast<int>(m_ssaoFormat)
              << "\n";

    destroyShaderModules(m_device, shaderModules);

    if (m_voxelNormalDepthPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_device, m_voxelNormalDepthPipeline, nullptr);
    }
    if (m_pipeNormalDepthPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_device, m_pipeNormalDepthPipeline, nullptr);
    }
    if (m_grassBillboardNormalDepthPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_device, m_grassBillboardNormalDepthPipeline, nullptr);
    }
    if (m_ssaoPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_device, m_ssaoPipeline, nullptr);
    }
    if (m_ssaoBlurPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_device, m_ssaoBlurPipeline, nullptr);
    }

    m_voxelNormalDepthPipeline = voxelNormalDepthPipeline;
    m_pipeNormalDepthPipeline = pipeNormalDepthPipeline;
    m_grassBillboardNormalDepthPipeline = grassBillboardNormalDepthPipeline;
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
    setObjectName(
        VK_OBJECT_TYPE_PIPELINE,
        vkHandleToUint64(m_grassBillboardNormalDepthPipeline),
        "pipeline.prepass.grassBillboardNormalDepth"
    );
    setObjectName(VK_OBJECT_TYPE_PIPELINE, vkHandleToUint64(m_ssaoPipeline), "pipeline.ssao");
    setObjectName(VK_OBJECT_TYPE_PIPELINE, vkHandleToUint64(m_ssaoBlurPipeline), "pipeline.ssaoBlur");
    return true;
}



bool RendererBackend::createGraphicsPipeline() {
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
    constexpr const char* kPipeShadowVertexShaderPath = "../src/render/shaders/pipe_shadow.vert.slang.spv";
    constexpr const char* kGrassShadowVertexShaderPath = "../src/render/shaders/grass_billboard_shadow.vert.slang.spv";
    constexpr const char* kGrassShadowFragmentShaderPath = "../src/render/shaders/grass_billboard_shadow.frag.slang.spv";

    std::array<VkShaderModule, 7> sceneShaderModules = {
        VK_NULL_HANDLE,
        VK_NULL_HANDLE,
        VK_NULL_HANDLE,
        VK_NULL_HANDLE,
        VK_NULL_HANDLE,
        VK_NULL_HANDLE,
        VK_NULL_HANDLE
    };
    VkShaderModule& worldVertShaderModule = sceneShaderModules[0];
    VkShaderModule& worldFragShaderModule = sceneShaderModules[1];
    VkShaderModule& skyboxVertShaderModule = sceneShaderModules[2];
    VkShaderModule& skyboxFragShaderModule = sceneShaderModules[3];
    VkShaderModule& toneMapVertShaderModule = sceneShaderModules[4];
    VkShaderModule& toneMapFragShaderModule = sceneShaderModules[5];
    VkShaderModule& shadowVertShaderModule = sceneShaderModules[6];

    const std::array<ShaderModuleLoadSpec, 6> sceneShaderLoadSpecs = {{
        {kWorldVertexShaderPath, "voxel_packed.vert"},
        {kWorldFragmentShaderPath, "voxel_packed.frag"},
        {kSkyboxVertexShaderPath, "skybox.vert"},
        {kSkyboxFragmentShaderPath, "skybox.frag"},
        {kToneMapVertexShaderPath, "tone_map.vert"},
        {kToneMapFragmentShaderPath, "tone_map.frag"},
    }};
    if (!createShaderModulesFromFiles(
            m_device,
            sceneShaderLoadSpecs,
            std::span<VkShaderModule>(sceneShaderModules).first(sceneShaderLoadSpecs.size())
        )) {
        return false;
    }
    auto destroySceneShaderModules = [&]() {
        destroyShaderModules(m_device, sceneShaderModules);
    };

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
        std::int32_t forceTintOnly = 0;     // 0=atlas sampling enabled, 1=tint-only shading
    };
    const WorldFragmentSpecializationData worldFragmentSpecializationData{};
    const std::array<VkSpecializationMapEntry, 3> worldFragmentSpecializationMapEntries = {{
        VkSpecializationMapEntry{
            6u,
            static_cast<uint32_t>(offsetof(WorldFragmentSpecializationData, shadowPolicyMode)),
            sizeof(std::int32_t)
        },
        VkSpecializationMapEntry{
            7u,
            static_cast<uint32_t>(offsetof(WorldFragmentSpecializationData, ambientPolicyMode)),
            sizeof(std::int32_t)
        },
        VkSpecializationMapEntry{
            8u,
            static_cast<uint32_t>(offsetof(WorldFragmentSpecializationData, forceTintOnly)),
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
    bindingDescriptions[0].stride = sizeof(voxelsprout::world::PackedVoxelVertex);
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
        destroySceneShaderModules();
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
        destroySceneShaderModules();
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
        destroySceneShaderModules();
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
    skyboxDepthStencil.depthTestEnable = VK_TRUE;
    skyboxDepthStencil.depthWriteEnable = VK_FALSE;
    skyboxDepthStencil.depthCompareOp = VK_COMPARE_OP_EQUAL;

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
        destroySceneShaderModules();
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
        destroySceneShaderModules();
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

    const std::array<ShaderModuleLoadSpec, 1> shadowShaderLoadSpecs = {{
        {kShadowVertexShaderPath, "shadow_depth.vert"},
    }};
    if (!createShaderModulesFromFiles(
            m_device,
            shadowShaderLoadSpecs,
            std::span<VkShaderModule>(sceneShaderModules).subspan(6, 1)
        )) {
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

    const std::array<VkPipelineShaderStageCreateInfo, 1> shadowShaderStages = {
        shadowVertexShaderStage
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

    destroySceneShaderModules();

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

    std::array<VkShaderModule, 1> pipeShadowShaderModules = {VK_NULL_HANDLE};
    VkShaderModule& pipeShadowVertShaderModule = pipeShadowShaderModules[0];
    const std::array<ShaderModuleLoadSpec, 1> pipeShadowShaderLoadSpecs = {{
        {kPipeShadowVertexShaderPath, "pipe_shadow.vert"},
    }};
    if (!createShaderModulesFromFiles(m_device, pipeShadowShaderLoadSpecs, pipeShadowShaderModules)) {
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

    const std::array<VkPipelineShaderStageCreateInfo, 1> pipeShadowShaderStages = {
        pipeShadowVertexShaderStage
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

    destroyShaderModules(m_device, pipeShadowShaderModules);

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

    std::array<VkShaderModule, 2> grassShadowShaderModules = {
        VK_NULL_HANDLE,
        VK_NULL_HANDLE
    };
    VkShaderModule& grassShadowVertShaderModule = grassShadowShaderModules[0];
    VkShaderModule& grassShadowFragShaderModule = grassShadowShaderModules[1];
    const std::array<ShaderModuleLoadSpec, 2> grassShadowShaderLoadSpecs = {{
        {kGrassShadowVertexShaderPath, "grass_billboard_shadow.vert"},
        {kGrassShadowFragmentShaderPath, "grass_billboard_shadow.frag"},
    }};
    if (!createShaderModulesFromFiles(m_device, grassShadowShaderLoadSpecs, grassShadowShaderModules)) {
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

    VkVertexInputAttributeDescription grassShadowAttributes[5]{};
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
    grassShadowAttributes[4].location = 4;
    grassShadowAttributes[4].binding = 1;
    grassShadowAttributes[4].format = VK_FORMAT_R32G32B32A32_SFLOAT;
    grassShadowAttributes[4].offset = static_cast<uint32_t>(offsetof(GrassBillboardInstance, colorTint));

    VkPipelineVertexInputStateCreateInfo grassShadowVertexInputInfo{};
    grassShadowVertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    grassShadowVertexInputInfo.vertexBindingDescriptionCount = 2;
    grassShadowVertexInputInfo.pVertexBindingDescriptions = grassShadowBindings;
    grassShadowVertexInputInfo.vertexAttributeDescriptionCount = 5;
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

    destroyShaderModules(m_device, grassShadowShaderModules);

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

bool RendererBackend::createSdfPipelines() {
    if (m_pipelineLayout == VK_NULL_HANDLE) {
        return false;
    }
    if (m_depthFormat == VK_FORMAT_UNDEFINED ||
        m_hdrColorFormat == VK_FORMAT_UNDEFINED ||
        m_shadowDepthFormat == VK_FORMAT_UNDEFINED ||
        m_normalDepthFormat == VK_FORMAT_UNDEFINED) {
        return false;
    }

    constexpr const char* kSdfMainVertexShaderPath = "../src/render/shaders/sdf_main.vert.slang.spv";
    constexpr const char* kSdfMainFragmentShaderPath = "../src/render/shaders/sdf_main.frag.slang.spv";
    constexpr const char* kSdfPrepassVertexShaderPath = "../src/render/shaders/sdf_prepass.vert.slang.spv";
    constexpr const char* kSdfPrepassFragmentShaderPath = "../src/render/shaders/sdf_prepass.frag.slang.spv";
    constexpr const char* kSdfShadowVertexShaderPath = "../src/render/shaders/sdf_shadow.vert.slang.spv";
    constexpr const char* kSdfShadowFragmentShaderPath = "../src/render/shaders/sdf_shadow.frag.slang.spv";

    std::array<VkShaderModule, 6> shaderModules = {
        VK_NULL_HANDLE,
        VK_NULL_HANDLE,
        VK_NULL_HANDLE,
        VK_NULL_HANDLE,
        VK_NULL_HANDLE,
        VK_NULL_HANDLE
    };
    VkShaderModule& sdfMainVertShaderModule = shaderModules[0];
    VkShaderModule& sdfMainFragShaderModule = shaderModules[1];
    VkShaderModule& sdfPrepassVertShaderModule = shaderModules[2];
    VkShaderModule& sdfPrepassFragShaderModule = shaderModules[3];
    VkShaderModule& sdfShadowVertShaderModule = shaderModules[4];
    VkShaderModule& sdfShadowFragShaderModule = shaderModules[5];
    const std::array<ShaderModuleLoadSpec, 6> shaderLoadSpecs = {{
        {kSdfMainVertexShaderPath, "sdf_main.vert"},
        {kSdfMainFragmentShaderPath, "sdf_main.frag"},
        {kSdfPrepassVertexShaderPath, "sdf_prepass.vert"},
        {kSdfPrepassFragmentShaderPath, "sdf_prepass.frag"},
        {kSdfShadowVertexShaderPath, "sdf_shadow.vert"},
        {kSdfShadowFragmentShaderPath, "sdf_shadow.frag"},
    }};
    if (!createShaderModulesFromFiles(m_device, shaderLoadSpecs, shaderModules)) {
        return false;
    }

    VkPipeline sdfMainPipeline = VK_NULL_HANDLE;
    VkPipeline sdfPrepassPipeline = VK_NULL_HANDLE;
    VkPipeline sdfShadowPipeline = VK_NULL_HANDLE;
    auto destroyNewPipelines = [&]() {
        if (sdfShadowPipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(m_device, sdfShadowPipeline, nullptr);
            sdfShadowPipeline = VK_NULL_HANDLE;
        }
        if (sdfPrepassPipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(m_device, sdfPrepassPipeline, nullptr);
            sdfPrepassPipeline = VK_NULL_HANDLE;
        }
        if (sdfMainPipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(m_device, sdfMainPipeline, nullptr);
            sdfMainPipeline = VK_NULL_HANDLE;
        }
    };

    VkPipelineVertexInputStateCreateInfo fullscreenVertexInputInfo{};
    fullscreenVertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

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
        VK_DYNAMIC_STATE_SCISSOR
    };
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
    dynamicState.pDynamicStates = dynamicStates.data();

    VkPipelineMultisampleStateCreateInfo mainMultisampling{};
    mainMultisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    mainMultisampling.rasterizationSamples = m_colorSampleCount;

    VkPipelineRenderingCreateInfo mainRenderingCreateInfo{};
    mainRenderingCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    mainRenderingCreateInfo.colorAttachmentCount = 1;
    mainRenderingCreateInfo.pColorAttachmentFormats = &m_hdrColorFormat;
    mainRenderingCreateInfo.depthAttachmentFormat = m_depthFormat;

    VkPipelineShaderStageCreateInfo mainStageInfos[2]{};
    mainStageInfos[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    mainStageInfos[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    mainStageInfos[0].module = sdfMainVertShaderModule;
    mainStageInfos[0].pName = "main";
    mainStageInfos[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    mainStageInfos[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    mainStageInfos[1].module = sdfMainFragShaderModule;
    mainStageInfos[1].pName = "main";

    VkGraphicsPipelineCreateInfo mainPipelineCreateInfo{};
    mainPipelineCreateInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    mainPipelineCreateInfo.pNext = &mainRenderingCreateInfo;
    mainPipelineCreateInfo.stageCount = 2;
    mainPipelineCreateInfo.pStages = mainStageInfos;
    mainPipelineCreateInfo.pVertexInputState = &fullscreenVertexInputInfo;
    mainPipelineCreateInfo.pInputAssemblyState = &inputAssembly;
    mainPipelineCreateInfo.pViewportState = &viewportState;
    mainPipelineCreateInfo.pRasterizationState = &rasterizer;
    mainPipelineCreateInfo.pMultisampleState = &mainMultisampling;
    mainPipelineCreateInfo.pDepthStencilState = &depthStencil;
    mainPipelineCreateInfo.pColorBlendState = &colorBlending;
    mainPipelineCreateInfo.pDynamicState = &dynamicState;
    mainPipelineCreateInfo.layout = m_pipelineLayout;
    mainPipelineCreateInfo.renderPass = VK_NULL_HANDLE;
    mainPipelineCreateInfo.subpass = 0;

    const VkResult mainPipelineResult = vkCreateGraphicsPipelines(
        m_device,
        VK_NULL_HANDLE,
        1,
        &mainPipelineCreateInfo,
        nullptr,
        &sdfMainPipeline
    );
    if (mainPipelineResult != VK_SUCCESS) {
        logVkFailure("vkCreateGraphicsPipelines(sdfMain)", mainPipelineResult);
        destroyNewPipelines();
        destroyShaderModules(m_device, shaderModules);
        return false;
    }

    VkPipelineMultisampleStateCreateInfo prepassMultisampling{};
    prepassMultisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    prepassMultisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineRenderingCreateInfo prepassRenderingCreateInfo{};
    prepassRenderingCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    prepassRenderingCreateInfo.colorAttachmentCount = 1;
    prepassRenderingCreateInfo.pColorAttachmentFormats = &m_normalDepthFormat;
    prepassRenderingCreateInfo.depthAttachmentFormat = m_depthFormat;

    VkPipelineShaderStageCreateInfo prepassStageInfos[2]{};
    prepassStageInfos[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    prepassStageInfos[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    prepassStageInfos[0].module = sdfPrepassVertShaderModule;
    prepassStageInfos[0].pName = "main";
    prepassStageInfos[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    prepassStageInfos[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    prepassStageInfos[1].module = sdfPrepassFragShaderModule;
    prepassStageInfos[1].pName = "main";

    VkGraphicsPipelineCreateInfo prepassPipelineCreateInfo = mainPipelineCreateInfo;
    prepassPipelineCreateInfo.pNext = &prepassRenderingCreateInfo;
    prepassPipelineCreateInfo.pStages = prepassStageInfos;
    prepassPipelineCreateInfo.pMultisampleState = &prepassMultisampling;

    const VkResult prepassPipelineResult = vkCreateGraphicsPipelines(
        m_device,
        VK_NULL_HANDLE,
        1,
        &prepassPipelineCreateInfo,
        nullptr,
        &sdfPrepassPipeline
    );
    if (prepassPipelineResult != VK_SUCCESS) {
        logVkFailure("vkCreateGraphicsPipelines(sdfPrepass)", prepassPipelineResult);
        destroyNewPipelines();
        destroyShaderModules(m_device, shaderModules);
        return false;
    }

    VkPipelineMultisampleStateCreateInfo shadowMultisampling{};
    shadowMultisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    shadowMultisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineRasterizationStateCreateInfo shadowRasterizer = rasterizer;
    shadowRasterizer.depthBiasEnable = VK_TRUE;

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

    VkPipelineShaderStageCreateInfo shadowStageInfos[2]{};
    shadowStageInfos[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shadowStageInfos[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    shadowStageInfos[0].module = sdfShadowVertShaderModule;
    shadowStageInfos[0].pName = "main";
    shadowStageInfos[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shadowStageInfos[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    shadowStageInfos[1].module = sdfShadowFragShaderModule;
    shadowStageInfos[1].pName = "main";

    VkGraphicsPipelineCreateInfo shadowPipelineCreateInfo = mainPipelineCreateInfo;
    shadowPipelineCreateInfo.pNext = &shadowRenderingCreateInfo;
    shadowPipelineCreateInfo.pStages = shadowStageInfos;
    shadowPipelineCreateInfo.pMultisampleState = &shadowMultisampling;
    shadowPipelineCreateInfo.pRasterizationState = &shadowRasterizer;
    shadowPipelineCreateInfo.pColorBlendState = &shadowColorBlending;
    shadowPipelineCreateInfo.pDynamicState = &shadowDynamicState;

    const VkResult shadowPipelineResult = vkCreateGraphicsPipelines(
        m_device,
        VK_NULL_HANDLE,
        1,
        &shadowPipelineCreateInfo,
        nullptr,
        &sdfShadowPipeline
    );

    destroyShaderModules(m_device, shaderModules);

    if (shadowPipelineResult != VK_SUCCESS) {
        logVkFailure("vkCreateGraphicsPipelines(sdfShadow)", shadowPipelineResult);
        destroyNewPipelines();
        return false;
    }

    if (m_sdfMainPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_device, m_sdfMainPipeline, nullptr);
    }
    if (m_sdfPrepassPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_device, m_sdfPrepassPipeline, nullptr);
    }
    if (m_sdfShadowPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_device, m_sdfShadowPipeline, nullptr);
    }
    m_sdfMainPipeline = sdfMainPipeline;
    m_sdfPrepassPipeline = sdfPrepassPipeline;
    m_sdfShadowPipeline = sdfShadowPipeline;
    setObjectName(VK_OBJECT_TYPE_PIPELINE, vkHandleToUint64(m_sdfMainPipeline), "pipeline.sdf.main");
    setObjectName(VK_OBJECT_TYPE_PIPELINE, vkHandleToUint64(m_sdfPrepassPipeline), "pipeline.sdf.prepass");
    setObjectName(VK_OBJECT_TYPE_PIPELINE, vkHandleToUint64(m_sdfShadowPipeline), "pipeline.sdf.shadow");
    VOX_LOGI("render") << "SDF pipelines ready (main + prepass + shadow)\n";
    return true;
}

} // namespace voxelsprout::render
