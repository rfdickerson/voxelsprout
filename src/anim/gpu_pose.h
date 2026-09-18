#pragma once
#include "anim/pose_graph.h"
namespace odai::anim {
// Scalar uint words form the stable CPU/Slang ABI. Floating-point entries are
// bit-cast, never numerically converted. No Vulkan types cross this boundary.
struct GpuPoseResource {
  std::vector<std::uint32_t> words;
  std::map<std::string, std::uint32_t> clips;
  std::uint32_t boneCount = 0, depthCount = 0;
};
bool packGpuPoseResource(const Skeleton &skeleton,
                         std::span<const odai::math::Matrix4> inverseBind,
                         std::span<const AnimationClip> clips,
                         GpuPoseResource &out, std::string &error);
bool packGpuPoseFrame(const PoseEvaluationPacket &packet,
                      const GpuPoseResource &resource,
                      const odai::math::Matrix4 &actorWorld,
                      std::vector<std::uint32_t> &out, std::string &error);
} // namespace odai::anim
