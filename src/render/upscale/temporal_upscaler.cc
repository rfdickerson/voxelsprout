#include "render/upscale/temporal_upscaler.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <filesystem>

namespace odai::render::upscale {

namespace {

// Mirrors the shader-side push block in taa.comp.slang / temporal_upscale.comp.slang.
struct TemporalPushConstants {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    float pad0 = 0.0f;
    float pad1 = 0.0f;
};

constexpr const char* kTaaShaderPath = "../src/render/shaders/taa.comp.slang.spv";
constexpr const char* kUpscaleShaderPath = "../src/render/shaders/temporal_upscale.comp.slang.spv";

class TemporalUpscaler final : public IUpscaler {
public:
    explicit TemporalUpscaler(const HostServices& host) : m_host(host) {}

    ~TemporalUpscaler() override { shutdown(); }

    [[nodiscard]] UpscalerBackend id() const override { return UpscalerBackend::Temporal; }

    [[nodiscard]] Capabilities capabilities() const override {
        Capabilities caps{};
        // Available whenever the resolve pipeline built. The upscale pipeline is
        // allowed to be missing -- see setup() -- and the backend then runs as a
        // same-resolution resolve rather than reporting itself unavailable.
        caps.available = m_resolvePipeline != VK_NULL_HANDLE;
        caps.unavailableReason = caps.available ? "" : "taa.comp.slang.spv failed to load";
        caps.requiresJitter = true;
        caps.requiresMotionVectors = false;  // falls back to depth reprojection
        caps.supportsSharpness = false;
        return caps;
    }

    bool setup(const SetupInfo& info) override {
        m_setup = info;
        m_upscaling = info.renderExtent.width < info.displayExtent.width ||
                      info.renderExtent.height < info.displayExtent.height;
        if (info.pipelineLayout == VK_NULL_HANDLE || !m_host.loadShader) {
            return false;
        }
        if (m_resolvePipeline == VK_NULL_HANDLE &&
            !createPipeline(kTaaShaderPath, "taa.comp", m_resolvePipeline)) {
            return false;
        }
        // Optional by design. Without it an upscaling host still gets a correct
        // -- if softer -- frame from the resolve path stretched by the tonemap,
        // which is a better failure than refusing to start.
        if (m_upscaling && m_upscalePipeline == VK_NULL_HANDLE &&
            std::filesystem::exists(kUpscaleShaderPath)) {
            createPipeline(kUpscaleShaderPath, "temporal_upscale.comp", m_upscalePipeline);
        }
        return true;
    }

    DispatchResult dispatch(const DispatchInfo& info) override;

    void shutdown() override {
        if (m_host.device == VK_NULL_HANDLE) {
            return;
        }
        for (VkPipeline* pipeline : {&m_resolvePipeline, &m_upscalePipeline}) {
            if (*pipeline != VK_NULL_HANDLE) {
                vkDestroyPipeline(m_host.device, *pipeline, nullptr);
                *pipeline = VK_NULL_HANDLE;
            }
        }
    }

private:
    bool createPipeline(const char* path, const char* debugName, VkPipeline& out) {
        VkShaderModule module = VK_NULL_HANDLE;
        if (!m_host.loadShader(path, debugName, module)) {
            return false;
        }
        VkPipelineShaderStageCreateInfo stage{};
        stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = module;
        stage.pName = "main";

        VkComputePipelineCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        createInfo.stage = stage;
        createInfo.layout = m_setup.pipelineLayout;
        createInfo.flags = VK_PIPELINE_CREATE_DESCRIPTOR_BUFFER_BIT_EXT;

        const VkResult result = vkCreateComputePipelines(
            m_host.device, VK_NULL_HANDLE, 1, &createInfo, nullptr, &out);
        vkDestroyShaderModule(m_host.device, module, nullptr);
        return result == VK_SUCCESS;
    }

    HostServices m_host;
    SetupInfo m_setup{};
    bool m_upscaling = false;
    VkPipeline m_resolvePipeline = VK_NULL_HANDLE;
    VkPipeline m_upscalePipeline = VK_NULL_HANDLE;
};

DispatchResult TemporalUpscaler::dispatch(const DispatchInfo& info) {
    DispatchResult result{};
    if (m_resolvePipeline == VK_NULL_HANDLE || info.colorInput == VK_NULL_HANDLE ||
        info.history == VK_NULL_HANDLE || info.output == VK_NULL_HANDLE) {
        return result;
    }
    const bool upscaling = m_upscaling && m_upscalePipeline != VK_NULL_HANDLE;

    if (m_host.beginDebugLabel) {
        m_host.beginDebugLabel(
            info.commandBuffer, upscaling ? "Upscale: Temporal" : "Upscale: TAA resolve",
            0.22f, 0.34f, 0.40f, 1.0f);
    }

    // Colour input: main-pass output -> compute sampled input.
    m_host.transitionImage(
        info.commandBuffer, info.colorInput, info.colorInputLayout,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);

    // This frame's output -> GENERAL for storage writes. Its previous contents
    // are history from two frames ago and are dead; UNDEFINED is both legal and
    // faster when it was never written.
    m_host.transitionImage(
        info.commandBuffer, info.output,
        info.outputInitialized ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                               : VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_GENERAL,
        info.outputInitialized ? VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
                               : VK_PIPELINE_STAGE_2_NONE,
        info.outputInitialized ? VK_ACCESS_2_SHADER_SAMPLED_READ_BIT : VK_ACCESS_2_NONE,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

    // The history image the descriptor points at has to be in SHADER_READ_ONLY
    // even on the first frame, when it holds nothing and the shader's weight is
    // zero -- validation checks descriptor layouts regardless of branches.
    if (!info.historyInitialized) {
        m_host.transitionImage(
            info.commandBuffer, info.history, VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    }

    vkCmdBindPipeline(
        info.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
        upscaling ? m_upscalePipeline : m_resolvePipeline);
    m_host.bindDescriptors(info.commandBuffer, m_setup.pipelineLayout, info.frameIndex);

    // ONE THREAD PER OUTPUT PIXEL when upscaling -- that is the grid being
    // reconstructed, and dispatching over the input extent would leave most of
    // the target unwritten.
    const Extent2D extent = upscaling ? m_setup.displayExtent : m_setup.renderExtent;
    TemporalPushConstants push{};
    push.width = extent.width;
    push.height = extent.height;
    vkCmdPushConstants(
        info.commandBuffer, m_setup.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
        sizeof(push), &push);
    vkCmdDispatch(info.commandBuffer, (extent.width + 7u) / 8u, (extent.height + 7u) / 8u, 1u);

    result.ran = true;
    result.resultInOutput = upscaling;
    if (m_host.endDebugLabel) {
        m_host.endDebugLabel(info.commandBuffer);
    }
    return result;
}

}  // namespace

std::unique_ptr<IUpscaler> createTemporalUpscaler(const HostServices& host) {
    return std::make_unique<TemporalUpscaler>(host);
}

}  // namespace odai::render::upscale
