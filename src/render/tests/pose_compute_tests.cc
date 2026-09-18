#include "anim/gpu_pose.h"
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <vulkan/vulkan.h>
using namespace odai::anim;
using namespace odai::math;
namespace {
void check(VkResult result) {
  if (result != VK_SUCCESS)
    throw std::runtime_error("Vulkan error " + std::to_string(result));
}
struct Buffer {
  VkBuffer buffer{};
  VkDeviceMemory memory{};
  void *mapped{};
  VkDeviceSize size{};
};
struct Context {
  VkInstance instance{};
  VkPhysicalDevice physical{};
  VkDevice device{};
  VkQueue queue{};
  VkCommandPool pool{};
  VkDescriptorSetLayout layout{};
  VkDescriptorPool descriptors{};
  VkPipelineLayout pipelineLayout{};
  VkPipeline pipeline{};
  VkShaderModule shader{};
  VkQueryPool timestamps{};
  std::vector<Buffer> buffers;
  ~Context() {
    if (device) {
      vkDeviceWaitIdle(device);
      for (auto b : buffers) {
        if (b.mapped)
          vkUnmapMemory(device, b.memory);
        vkDestroyBuffer(device, b.buffer, nullptr);
        vkFreeMemory(device, b.memory, nullptr);
      }
      vkDestroyQueryPool(device, timestamps, nullptr);
      vkDestroyPipeline(device, pipeline, nullptr);
      vkDestroyShaderModule(device, shader, nullptr);
      vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
      vkDestroyDescriptorPool(device, descriptors, nullptr);
      vkDestroyDescriptorSetLayout(device, layout, nullptr);
      vkDestroyCommandPool(device, pool, nullptr);
      vkDestroyDevice(device, nullptr);
    }
    if (instance)
      vkDestroyInstance(instance, nullptr);
  }
  Buffer buffer(VkDeviceSize size, const void *data = nullptr) {
    Buffer b;
    b.size = size;
    VkBufferCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    ci.size = size;
    ci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    check(vkCreateBuffer(device, &ci, nullptr, &b.buffer));
    VkMemoryRequirements requirements;
    vkGetBufferMemoryRequirements(device, b.buffer, &requirements);
    VkPhysicalDeviceMemoryProperties memory;
    vkGetPhysicalDeviceMemoryProperties(physical, &memory);
    std::uint32_t type = memory.memoryTypeCount;
    for (std::uint32_t i = 0; i < memory.memoryTypeCount; ++i)
      if ((requirements.memoryTypeBits & (1u << i)) &&
          (memory.memoryTypes[i].propertyFlags &
           (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
              (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
               VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
        type = i;
        break;
      }
    if (type == memory.memoryTypeCount)
      throw std::runtime_error("no coherent test memory");
    VkMemoryAllocateInfo allocation{};
    allocation.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex = type;
    check(vkAllocateMemory(device, &allocation, nullptr, &b.memory));
    check(vkBindBufferMemory(device, b.buffer, b.memory, 0));
    check(vkMapMemory(device, b.memory, 0, size, 0, &b.mapped));
    if (data)
      std::memcpy(b.mapped, data, static_cast<std::size_t>(size));
    else
      std::memset(b.mapped, 0, static_cast<std::size_t>(size));
    buffers.push_back(b);
    return b;
  }
};
} // namespace
int main() {
  try {
    Context c;
    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo instance{};
    instance.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance.pApplicationInfo = &app;
    if (vkCreateInstance(&instance, nullptr, &c.instance) != VK_SUCCESS) {
      std::cout << "No Vulkan instance; skipped\n";
      return 77;
    }
    std::uint32_t count = 0;
    check(vkEnumeratePhysicalDevices(c.instance, &count, nullptr));
    if (!count)
      return 77;
    std::vector<VkPhysicalDevice> physical(count);
    check(vkEnumeratePhysicalDevices(c.instance, &count, physical.data()));
    std::uint32_t family = 0;
    bool found = false;
    for (auto p : physical) {
      VkPhysicalDeviceProperties props;
      vkGetPhysicalDeviceProperties(p, &props);
      if (props.apiVersion < VK_API_VERSION_1_3)
        continue;
      std::uint32_t n = 0;
      vkGetPhysicalDeviceQueueFamilyProperties(p, &n, nullptr);
      std::vector<VkQueueFamilyProperties> queues(n);
      vkGetPhysicalDeviceQueueFamilyProperties(p, &n, queues.data());
      for (std::uint32_t i = 0; i < n; ++i)
        if ((queues[i].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
            queues[i].timestampValidBits == 64) {
          c.physical = p;
          family = i;
          found = true;
          break;
        }
      if (found)
        break;
    }
    if (!found)
      return 77;
    const float priority = 1;
    VkDeviceQueueCreateInfo q{};
    q.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    q.queueFamilyIndex = family;
    q.queueCount = 1;
    q.pQueuePriorities = &priority;
    VkPhysicalDeviceVulkan13Features f13{};
    f13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    f13.synchronization2 = VK_TRUE;
    VkDeviceCreateInfo device{};
    device.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device.pNext = &f13;
    device.queueCreateInfoCount = 1;
    device.pQueueCreateInfos = &q;
    check(vkCreateDevice(c.physical, &device, nullptr, &c.device));
    vkGetDeviceQueue(c.device, family, 0, &c.queue);
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(c.physical, &props);
    VkQueryPoolCreateInfo query{};
    query.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    query.queryType = VK_QUERY_TYPE_TIMESTAMP;
    query.queryCount = 2;
    check(vkCreateQueryPool(c.device, &query, nullptr, &c.timestamps));
    std::cout << "GPU: " << props.deviceName << '\n';
    VkCommandPoolCreateInfo pool{};
    pool.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool.queueFamilyIndex = family;
    pool.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    check(vkCreateCommandPool(c.device, &pool, nullptr, &c.pool));
    std::array<VkDescriptorSetLayoutBinding, 6> bindings{};
    for (std::uint32_t i = 0; i < 6; ++i)
      bindings[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                     VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo layout{};
    layout.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout.bindingCount = 6;
    layout.pBindings = bindings.data();
    check(vkCreateDescriptorSetLayout(c.device, &layout, nullptr, &c.layout));
    VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, 32};
    VkPipelineLayoutCreateInfo pl{};
    pl.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pl.setLayoutCount = 1;
    pl.pSetLayouts = &c.layout;
    pl.pushConstantRangeCount = 1;
    pl.pPushConstantRanges = &range;
    check(vkCreatePipelineLayout(c.device, &pl, nullptr, &c.pipelineLayout));
    std::ifstream file(ODAI_POSE_SHADER, std::ios::binary | std::ios::ate);
    if (!file)
      throw std::runtime_error("missing built pose shader");
    auto size = file.tellg();
    file.seekg(0);
    std::vector<std::uint32_t> code(static_cast<std::size_t>(size) / 4);
    file.read(reinterpret_cast<char *>(code.data()), size);
    VkShaderModuleCreateInfo shader{};
    shader.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shader.codeSize = code.size() * 4;
    shader.pCode = code.data();
    check(vkCreateShaderModule(c.device, &shader, nullptr, &c.shader));
    VkComputePipelineCreateInfo pipeline{};
    pipeline.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline.layout = c.pipelineLayout;
    pipeline.stage = {};
    pipeline.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeline.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline.stage.module = c.shader;
    pipeline.stage.pName = "main";
    check(vkCreateComputePipelines(c.device, VK_NULL_HANDLE, 1, &pipeline,
                                   nullptr, &c.pipeline));
    VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 6};
    VkDescriptorPoolCreateInfo dp{};
    dp.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dp.maxSets = 1;
    dp.poolSizeCount = 1;
    dp.pPoolSizes = &poolSize;
    check(vkCreateDescriptorPool(c.device, &dp, nullptr, &c.descriptors));
    VkDescriptorSet set;
    VkDescriptorSetAllocateInfo allocate{};
    allocate.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocate.descriptorPool = c.descriptors;
    allocate.descriptorSetCount = 1;
    allocate.pSetLayouts = &c.layout;
    check(vkAllocateDescriptorSets(c.device, &allocate, &set));
    Skeleton skeleton;
    for (int i = 0; i < 650; ++i)
      skeleton.bones.push_back({"bone" + std::to_string(i),
                                i == 0 ? -1 : (i - 1) / 3,
                                {.1f, .2f, .3f},
                                {0, 0, 0, 1},
                                {1, 1, 1}});
    LocalPose rest;
    for (const auto &b : skeleton.bones)
      rest.push_back({b.localTranslation, b.localRotation, b.localScale});
    auto inverseBind = composePoseWorld(skeleton, rest);
    for (auto &m : inverseBind)
      m = inverse(m);
    std::vector<AnimationClip> clips;
    for (int n = 0; n < 3; ++n) {
      AnimationClip clip;
      clip.name = "clip" + std::to_string(n);
      clip.duration = 2;
      clip.loop = n != 2;
      for (int bone = 0; bone < 650; bone += 3) {
        BoneTrack track;
        track.boneIndex = bone;
        track.translationKeys = {{.2f, {.1f, .2f, .3f}},
                                 {1.5f, {.1f + n, .25f, .3f}}};
        track.rotationKeys = {
            {.2f, {0, 0, 0, 1}},
            {1.5f, normalize(Quaternion{.1f * n, .2f, .1f, 1})}};
        track.scaleKeys = {{0, {1, 1, 1}}, {2, {1.f + n * .05f, 1, 1}}};
        clip.tracks.push_back(track);
      }
      clips.push_back(std::move(clip));
    }
    PoseEvaluationPacket packet;
    for (int n = 0; n < 3; ++n) {
      PoseGraphInstruction i;
      i.clip = clips[n].name;
      i.time = n == 0 ? .05f : n == 1 ? 1.8f : 2.2f;
      packet.instructions.push_back(i);
    }
    PoseGraphInstruction blend;
    blend.kind = PoseGraphNode::Kind::Blend;
    blend.inputs = {0, 1};
    blend.weight = .3f;
    packet.instructions.push_back(blend);
    PoseGraphInstruction layer;
    layer.kind = PoseGraphNode::Kind::Layer;
    layer.inputs = {3, 2};
    layer.weight = .4f;
    layer.additive = true;
    layer.reference = "clip0";
    layer.mask.resize(650);
    for (int b = 0; b < 650; ++b)
      layer.mask[b] = b % 2 ? 1.f : .3f;
    packet.instructions.push_back(layer);
    packet.output = 4;
    auto post = std::make_shared<PosePostProcess>();
    post->offsets.resize(650);
    post->velocities.resize(650);
    post->offsetWeight = .4f;
    post->velocityWeight = .02f;
    for (int b = 0; b < 650; ++b) {
      post->offsets[b] = {{.01f, .02f, 0}, {0, .02f, 0, 0}, {.01f, 0, 0}};
      post->velocities[b] = {{.01f, 0, 0}, {0, 0, .01f, 0}, {0, .01f, 0}};
    }
    PoseModifier translate;
    translate.chain.upper = 1;
    translate.target.position = {0, -.01f, 0};
    post->modifiers.push_back(translate);
    PoseModifier limb;
    limb.kind = PoseModifier::Kind::LimbIk;
    limb.chain = {1, 4, 13};
    limb.target.position = {.4f, .6f, 1};
    limb.target.weight = .65f;
    limb.target.alignNormal = true;
    limb.target.normal = normalize(Vector3{.1f, 1, .2f});
    post->modifiers.push_back(limb);
    PoseModifier aim;
    aim.kind = PoseModifier::Kind::Aim;
    aim.chain.upper = 2;
    aim.target.position = {3, 2, 4};
    aim.target.weight = .6f;
    aim.maxRadians = .4f;
    post->modifiers.push_back(aim);
    PoseGraphInstruction procedural;
    procedural.kind = PoseGraphNode::Kind::Cache;
    procedural.inputs = {4};
    procedural.postProcess = post;
    packet.instructions.push_back(procedural);
    packet.output = 5;

    GpuPoseResource resource;
    std::string error;
    assert(packGpuPoseResource(skeleton, inverseBind, clips, resource, error));
    std::vector<std::uint32_t> frame;
    const auto actor =
        Matrix4::translation({12, 4, -8}) * Matrix4::rotationY(.6f);
    assert(packGpuPoseFrame(packet, resource, actor, frame, error));
    std::array<Buffer, 6> data{
        c.buffer(resource.words.size() * 4, resource.words.data()),
        c.buffer(frame.size() * 4, frame.data()),
        c.buffer(650 * (packet.instructions.size() * 3 + 4) * 16),
        c.buffer(650 * 64),
        c.buffer(650 * 64),
        c.buffer(650 * 64)};
    std::array<VkDescriptorBufferInfo, 6> infos{};
    std::array<VkWriteDescriptorSet, 6> writes{};
    for (std::uint32_t i = 0; i < 6; ++i) {
      infos[i] = {data[i].buffer, 0, data[i].size};
      writes[i] = {};
      writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      writes[i].dstSet = set;
      writes[i].dstBinding = i;
      writes[i].descriptorCount = 1;
      writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      writes[i].pBufferInfo = &infos[i];
    }
    vkUpdateDescriptorSets(c.device, 6, writes.data(), 0, nullptr);
    VkCommandBuffer cmd;
    VkCommandBufferAllocateInfo ca{};
    ca.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ca.commandPool = c.pool;
    ca.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ca.commandBufferCount = 1;
    check(vkAllocateCommandBuffers(c.device, &ca, &cmd));
    for (int actors : {1, 16, 48}) {
      check(vkResetCommandBuffer(cmd, 0));
      VkCommandBufferBeginInfo begin{};
      begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
      check(vkBeginCommandBuffer(cmd, &begin));
      vkCmdResetQueryPool(cmd, c.timestamps, 0, 2);
      vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                           c.timestamps, 0);
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, c.pipeline);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                              c.pipelineLayout, 0, 1, &set, 0, nullptr);
      const auto barrier = [&](VkPipelineStageFlags2 stage,
                               VkAccessFlags2 access) {
        VkMemoryBarrier2 m{};
        m.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        m.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        m.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT |
                          VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
        m.dstStageMask = stage;
        m.dstAccessMask = access;
        VkDependencyInfo d{};
        d.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        d.memoryBarrierCount = 1;
        d.pMemoryBarriers = &m;
        vkCmdPipelineBarrier2(cmd, &d);
      };
      for (int a = 0; a < actors; ++a) {
        std::array<std::uint32_t, 8> push{
            650,
            static_cast<std::uint32_t>(packet.instructions.size()),
            packet.output,
            0,
            0,
            0,
            0,
            0};
        const auto dispatch = [&]() {
          vkCmdPushConstants(cmd, c.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                             0, 32, push.data());
          vkCmdDispatch(cmd, 11, 1, 1);
        };
        barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                    VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        for (std::uint32_t node = 0; node < packet.instructions.size();
             ++node) {
          push[3] = 0;
          push[7] = node;
          barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                  VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                      VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
          dispatch();
          if (packet.instructions[node].postProcess &&
              !packet.instructions[node].postProcess->modifiers.empty()) {
            barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
            push[3] = 2;
            dispatch();
          }
        }
        push[3] = 1;
        for (std::uint32_t depth = 0; depth < resource.depthCount; ++depth) {
          barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                  VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                      VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
          push[4] = depth;
          dispatch();
        }
        barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                    VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        push[3] = 3;
        push[5] = a > 0 ? 1u : 0u;
        dispatch();
      }
      barrier(VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);
      vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
                           c.timestamps, 1);
      check(vkEndCommandBuffer(cmd));
      VkSubmitInfo submit{};
      submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
      submit.commandBufferCount = 1;
      submit.pCommandBuffers = &cmd;
      const auto start = std::chrono::steady_clock::now();
      check(vkQueueSubmit(c.queue, 1, &submit, VK_NULL_HANDLE));
      check(vkQueueWaitIdle(c.queue));
      const auto ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - start)
                          .count();
      std::array<std::uint64_t, 2> ticks{};
      check(vkGetQueryPoolResults(c.device, c.timestamps, 0, 2, sizeof(ticks),
                                  ticks.data(), sizeof(std::uint64_t),
                                  VK_QUERY_RESULT_64_BIT |
                                      VK_QUERY_RESULT_WAIT_BIT));
      const double gpuMs = (ticks[1] - ticks[0]) *
                           static_cast<double>(props.limits.timestampPeriod) /
                           1.e6;
      const auto cpuStart = std::chrono::steady_clock::now();
      LocalPose expected;
      for (int a = 0; a < actors; ++a) {
        assert(evaluatePosePacket(packet, skeleton, clips, expected, error));
        auto palette = composePoseWorld(skeleton, expected);
        for (std::size_t b = 0; b < palette.size(); ++b)
          palette[b] = actor * palette[b] * inverseBind[b];
      }
      const double cpuMs = std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - cpuStart)
                               .count();
      assert(evaluatePosePacket(packet, skeleton, clips, expected, error));
      auto world = composePoseWorld(skeleton, expected);
      const auto *actual = static_cast<const float *>(data[3].mapped);
      float maximum = 0;
      for (std::size_t b = 0; b < 650; ++b) {
        const auto palette = actor * world[b] * inverseBind[b];
        for (int r = 0; r < 4; ++r)
          for (int col = 0; col < 4; ++col) {
            const float diff =
                std::abs(palette.m[r * 4 + col] - actual[b * 16 + col * 4 + r]);
            maximum = std::max(maximum, diff);
            if (!std::isfinite(diff) || diff > .002f)
              throw std::runtime_error(
                  "CPU/GPU palette mismatch bone=" + std::to_string(b) +
                  " error=" + std::to_string(diff));
          }
      }
      const auto *gpuLocal =
          static_cast<const float *>(data[2].mapped) + packet.output * 650 * 12;
      float localError = 0, vertexError = 0;
      for (std::size_t b = 0; b < 650; ++b) {
        const auto &t = expected[b];
        const std::array<float, 12> local{
            t.translation.x, t.translation.y, t.translation.z, 0,
            t.rotation.x,    t.rotation.y,    t.rotation.z,    t.rotation.w,
            t.scale.x,       t.scale.y,       t.scale.z,       0};
        const float sign = local[4] * gpuLocal[b * 12 + 4] +
                                       local[5] * gpuLocal[b * 12 + 5] +
                                       local[6] * gpuLocal[b * 12 + 6] +
                                       local[7] * gpuLocal[b * 12 + 7] <
                                   0
                               ? -1.f
                               : 1.f;
        for (int i : {0, 1, 2, 4, 5, 6, 7, 8, 9, 10}) {
          float diff = std::abs(local[i] - gpuLocal[b * 12 + i] *
                                               (i >= 4 && i <= 7 ? sign : 1));
          if (!std::isfinite(diff) || diff > 1.e-4f)
            throw std::runtime_error("local pose mismatch");
          localError = std::max(localError, diff);
        }
        // Normalized four-influence deformation, including a synthetic morph
        // applied before the palettes. GPU palette readback is test-only.
        const Vector3 vertex = Vector3{20, -30, 40} + Vector3{.2f, .1f, -.3f};
        Vector3 cpuVertex{}, gpuVertex{};
        for (int influence = 0; influence < 4; ++influence) {
          const auto bone = (b + influence * 17) % 650;
          const float weight = (influence + 1) * .1f;
          Matrix4 gpuMatrix;
          for (int r = 0; r < 4; ++r)
            for (int col = 0; col < 4; ++col)
              gpuMatrix.m[r * 4 + col] = actual[bone * 16 + col * 4 + r];
          cpuVertex =
              cpuVertex +
              transformPoint(actor * world[bone] * inverseBind[bone], vertex) *
                  weight;
          gpuVertex = gpuVertex + transformPoint(gpuMatrix, vertex) * weight;
        }
        const float diff = length(cpuVertex - gpuVertex);
        if (!std::isfinite(diff) || diff > .002f)
          throw std::runtime_error("deformed vertex mismatch");
        vertexError = std::max(vertexError, diff);
      }
      const auto *prior = static_cast<const float *>(data[4].mapped);
      for (std::size_t i = 0; i < 650 * 16; ++i)
        if (prior[i] != actual[i])
          throw std::runtime_error("rendered palette history mismatch");
      std::cout << actors
                << " sequential actor-equivalent dispatches, 650 bones: " << ms
                << " ms including submit/wait; GPU timestamp ms " << gpuMs
                << "; CPU reference ms " << cpuMs
                << "; max local/vertex errors " << localError << "/"
                << vertexError << "; max palette error " << maximum
                << "; immutable bytes " << resource.words.size() * 4
                << ", frame bytes " << frame.size() * 4 << '\n';
    }
    std::cout << "GPU pose parity passed\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
