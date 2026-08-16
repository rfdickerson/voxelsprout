#pragma once

// The plug-in seam: one dispatch interface behind which this engine's temporal
// upscaler, XeSS, FSR and DLSS are interchangeable.
//
// SHAPED AFTER FSR2's INTEGRATION MODEL, deliberately. A vendor upscaler does
// not allocate your colour, depth or motion-vector images and does not own your
// descriptor pool -- you hand it resources per dispatch and it owns only its own
// internals. FSR2 goes one step further and takes a HOST interface
// (`FfxFsr2Interface`) that the application fills in with resource and job
// scheduling. Both halves are here for the same reason: it is what lets a
// backend be swapped without the frame code around it changing, and it is what
// makes "we support DLSS" a matter of writing one class.
//
// What a backend owns:   its pipelines, its history, its algorithm.
// What the host owns:    the frame's images, their barriers, descriptor
//                        provisioning, debug labels and timestamps.
//
// The split is not arbitrary -- it is drawn where this engine's infrastructure
// is genuinely engine-specific. Descriptor BUFFERS (VK_EXT_descriptor_buffer)
// and a frame arena are not things a vendor SDK knows about, so they stay
// behind HostServices rather than leaking into every backend.

#include "render/upscale/upscale_contract.h"
#include "render/upscale/upscale_policy.h"

#include <vulkan/vulkan.h>

#include <functional>
#include <memory>

namespace odai::render::upscale {

// What a backend needs from whatever engine is hosting it. Every entry is
// something this engine already does one specific way and a different host
// would do differently.
struct HostServices {
    VkDevice device = VK_NULL_HANDLE;

    // Binds the descriptor set/buffer the host has already written for this
    // frame, against the backend's pipeline layout. The host decides whether
    // that is a descriptor buffer or a classic set; the backend does not care
    // and must not assume.
    std::function<void(VkCommandBuffer, VkPipelineLayout, std::uint32_t frameIndex)>
        bindDescriptors;

    // Layout transition + barrier. Backends express their resource needs
    // through this rather than emitting vkCmdPipelineBarrier2 themselves,
    // because the host is the only thing that knows an image's current layout.
    std::function<void(VkCommandBuffer, VkImage, VkImageLayout oldLayout,
                       VkImageLayout newLayout, VkPipelineStageFlags2 srcStage,
                       VkAccessFlags2 srcAccess, VkPipelineStageFlags2 dstStage,
                       VkAccessFlags2 dstAccess)>
        transitionImage;

    // Optional. Left unset by a host that has neither.
    std::function<void(VkCommandBuffer, const char*, float, float, float, float)> beginDebugLabel;
    std::function<void(VkCommandBuffer)> endDebugLabel;

    // Loads one SPIR-V module by path. A backend that ships its own shaders
    // (this engine's temporal one does) uses it; a vendor backend that carries
    // compiled kernels inside its SDK ignores it.
    std::function<bool(const char* path, const char* debugName, VkShaderModule& out)> loadShader;
};

// Everything fixed for as long as the swapchain is: sizes and frame invariants.
struct SetupInfo {
    Extent2D renderExtent{};
    Extent2D displayExtent{};
    // Reverse-Z. This engine uses GREATER_OR_EQUAL throughout, and every vendor
    // SDK has a flag for it because getting it wrong disables the depth-based
    // disocclusion test rather than failing.
    bool invertedDepth = true;
    bool hdrInput = true;
    // The pipeline layout the host's descriptors are written against. The
    // backend creates its pipelines with this rather than one of its own, which
    // is what keeps descriptor provisioning entirely on the host side.
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
};

// The frame's resources. Images, not descriptors: the host has already pointed
// its descriptors at these.
struct DispatchInfo {
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    std::uint32_t frameIndex = 0;

    VkImage colorInput = VK_NULL_HANDLE;
    VkImageLayout colorInputLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImage history = VK_NULL_HANDLE;
    bool historyInitialized = false;
    VkImage output = VK_NULL_HANDLE;
    bool outputInitialized = false;

    // Sub-pixel offset this frame's scene was rasterized with, in pixels. The
    // backend does not compute it -- the host must have jittered its projection
    // with the SAME value, and a backend that derived its own would be
    // describing a frame that was never rendered.
    JitterOffset jitter{};
    float deltaSeconds = 0.0f;
    bool resetHistory = false;
    float sharpness = 0.0f;
};

// What the dispatch left behind. The layouts matter to the caller: the
// non-upscaling path copies its result back over the colour input and the
// upscaling path does not, so the two leave that image in different layouts and
// a bool cannot express it.
struct DispatchResult {
    bool ran = false;
    VkImageLayout colorInputLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkPipelineStageFlags2 colorInputStage = VK_PIPELINE_STAGE_2_NONE;
    VkAccessFlags2 colorInputAccess = VK_ACCESS_2_NONE;
    // True when the result is in `output` rather than copied back into
    // `colorInput`, i.e. when the host must point its tonemap at the upscaled
    // image instead.
    bool resultInOutput = false;
};

struct Capabilities {
    bool available = false;
    // Only meaningful when `available` is false.
    const char* unavailableReason = "";
    bool requiresJitter = true;
    bool requiresMotionVectors = false;
    bool supportsSharpness = false;
};

class IUpscaler {
public:
    virtual ~IUpscaler() = default;

    [[nodiscard]] virtual UpscalerBackend id() const = 0;
    [[nodiscard]] virtual Capabilities capabilities() const = 0;

    // Called once per swapchain lifetime. False means this backend cannot run
    // at these sizes and the caller should fall back -- not that anything is
    // broken.
    virtual bool setup(const SetupInfo& info) = 0;
    virtual DispatchResult dispatch(const DispatchInfo& info) = 0;
    virtual void shutdown() = 0;
};

// Creates the backend for `id`, or nullptr when it is not compiled in. Never
// falls back on its own: resolveUpscaler() in upscale_policy.h decides what to
// run, this only builds what it decided. Keeping the two apart is what lets the
// fallback reason be reported once, at init, instead of at every dispatch.
[[nodiscard]] std::unique_ptr<IUpscaler> createUpscaler(
    UpscalerBackend id, const HostServices& host);

}  // namespace odai::render::upscale
