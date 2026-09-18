#include "anim/gpu_pose.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <stdexcept>
namespace odai::anim {
namespace {
std::uint32_t bits(float f) {
  if (!std::isfinite(f))
    throw std::runtime_error("nonfinite GPU pose value");
  return std::bit_cast<std::uint32_t>(f);
}
void matrix(std::vector<std::uint32_t> &words, const odai::math::Matrix4 &m) {
  for (float f : m.m)
    words.push_back(bits(f));
}
} // namespace
bool packGpuPoseResource(const Skeleton &skeleton,
                         std::span<const odai::math::Matrix4> inverseBind,
                         std::span<const AnimationClip> clips,
                         GpuPoseResource &out, std::string &error) {
  try {
    if (skeleton.bones.empty() || skeleton.bones.size() > 65535 ||
        inverseBind.size() != skeleton.bones.size() || clips.size() > 4096)
      throw std::runtime_error("invalid GPU rig dimensions");
    GpuPoseResource r;
    r.boneCount = static_cast<std::uint32_t>(skeleton.bones.size());
    std::vector<std::uint32_t> depth(r.boneCount);
    for (std::size_t i = 0; i < skeleton.bones.size(); ++i) {
      const auto &b = skeleton.bones[i];
      if (b.parentIndex < -1 || b.parentIndex >= static_cast<int>(i))
        throw std::runtime_error("invalid GPU hierarchy");
      depth[i] = b.parentIndex < 0 ? 0 : depth[b.parentIndex] + 1;
      r.depthCount = std::max(r.depthCount, depth[i] + 1);
      for (float f : {b.localTranslation.x, b.localTranslation.y,
                      b.localTranslation.z, 0.f, b.localRotation.x,
                      b.localRotation.y, b.localRotation.z, b.localRotation.w,
                      b.localScale.x, b.localScale.y, b.localScale.z, 0.f})
        r.words.push_back(bits(f));
      matrix(r.words, inverseBind[i]);
      r.words.push_back(static_cast<std::uint32_t>(b.parentIndex));
      r.words.push_back(depth[i]);
      r.words.push_back(0);
      r.words.push_back(0);
    }
    for (const auto &clip : clips) {
      if (clip.name.empty() || !std::isfinite(clip.duration) ||
          clip.duration <= 0 ||
          !r.clips
               .emplace(clip.name, static_cast<std::uint32_t>(r.words.size()))
               .second)
        throw std::runtime_error("invalid GPU clip");
      const auto header = r.words.size();
      r.words.insert(r.words.end(),
                     {bits(clip.duration), clip.loop ? 1u : 0u,
                      clip.additive ? 1u : 0u,
                      static_cast<std::uint32_t>(clip.extractedMotionBone)});
      const auto tracks = r.words.size();
      r.words.resize(tracks + r.boneCount * 6);
      std::vector<bool> seen(r.boneCount);
      for (const auto &t : clip.tracks) {
        if (t.boneIndex < 0 ||
            static_cast<std::size_t>(t.boneIndex) >= r.boneCount ||
            seen[t.boneIndex])
          throw std::runtime_error("invalid GPU track");
        seen[t.boneIndex] = true;
        const auto packKeys = [&](const auto &keys, std::size_t field) {
          const auto base =
              tracks + static_cast<std::size_t>(t.boneIndex) * 6 + field;
          r.words[base] = static_cast<std::uint32_t>(r.words.size());
          r.words[base + 1] = static_cast<std::uint32_t>(keys.size());
          float previous = -1;
          for (const auto &k : keys) {
            if (k.time < 0 || k.time < previous || k.time > clip.duration)
              throw std::runtime_error("invalid GPU key time");
            previous = k.time;
            r.words.insert(r.words.end(), {bits(k.time), bits(k.value.x),
                                           bits(k.value.y), bits(k.value.z)});
            if constexpr (requires { k.value.w; })
              r.words.push_back(bits(k.value.w));
            else
              r.words.push_back(0);
            r.words.insert(r.words.end(), 3, 0);
          }
        };
        packKeys(t.translationKeys, 0);
        packKeys(t.rotationKeys, 2);
        packKeys(t.scaleKeys, 4);
      }
      (void)header;
      if (r.words.size() > 64u * 1024u * 1024u)
        throw std::runtime_error("GPU clip resource exceeds 256 MiB");
    }
    out = std::move(r);
    error.clear();
    return true;
  } catch (const std::exception &e) {
    error = e.what();
    return false;
  }
}
bool packGpuPoseFrame(const PoseEvaluationPacket &packet,
                      const GpuPoseResource &resource,
                      const odai::math::Matrix4 &actorWorld,
                      std::vector<std::uint32_t> &out, std::string &error) {
  try {
    if (packet.instructions.empty() || packet.instructions.size() > 256 ||
        packet.output >= packet.instructions.size())
      throw std::runtime_error("invalid GPU packet");
    std::vector<std::uint32_t> words;
    matrix(words, actorWorld);
    words.resize(16 + packet.instructions.size() * 12);
    for (std::size_t index = 0; index < packet.instructions.size(); ++index) {
      const auto &i = packet.instructions[index];
      if (i.weight < 0 || i.weight > 1 ||
          (i.weights.empty() && i.inputs.size() > 2) ||
          (i.additive && (!i.weights.empty() || i.inputs.size() != 2)))
        throw std::runtime_error("invalid GPU instruction values");
      const auto base = 16 + index * 12;
      const auto clipOffset = [&](const std::string &name) {
        const auto found = resource.clips.find(name);
        if (found == resource.clips.end())
          throw std::runtime_error("GPU clip not resident: " + name);
        return found->second;
      };
      // 0 sample, 1 blend, 2 additive; cache and state nodes lower to blend.
      words[base] = i.kind == PoseGraphNode::Kind::Clip ? 0u
                    : i.additive                        ? 2u
                                                        : 1u;
      words[base + 1] =
          i.kind == PoseGraphNode::Kind::Clip ? clipOffset(i.clip) : 0;
      words[base + 2] = bits(i.time);
      words[base + 3] = bits(i.weight);
      words[base + 4] = static_cast<std::uint32_t>(words.size());
      words[base + 5] = static_cast<std::uint32_t>(i.inputs.size());
      if (!i.weights.empty() && i.weights.size() != i.inputs.size())
        throw std::runtime_error("invalid GPU weights");
      if (i.kind != PoseGraphNode::Kind::Clip && i.inputs.empty())
        throw std::runtime_error("empty GPU node");
      for (std::size_t n = 0; n < i.inputs.size(); ++n) {
        if (i.inputs[n] >= index)
          throw std::runtime_error("GPU dependency order");
        const float w = i.weights.empty() ? (i.inputs.size() == 1 ? 1.f
                                             : n == 0             ? 1 - i.weight
                                                                  : i.weight)
                                          : i.weights[n];
        if (w < 0 || w > 1)
          throw std::runtime_error("invalid GPU blend weight");
        words.push_back(i.inputs[n]);
        words.push_back(bits(w));
      }
      words[base + 6] =
          i.mask.empty() ? 0 : static_cast<std::uint32_t>(words.size());
      if (!i.mask.empty()) {
        if (i.mask.size() != resource.boneCount)
          throw std::runtime_error("GPU mask size");
        for (float v : i.mask) {
          if (v < 0 || v > 1)
            throw std::runtime_error("invalid GPU mask");
          words.push_back(bits(v));
        }
      }
      words[base + 7] = i.additive ? clipOffset(i.reference) : 0;
      if (i.postProcess) {
        const auto &post = *i.postProcess;
        if (post.offsets.size() != post.velocities.size() ||
            (!post.offsets.empty() &&
             post.offsets.size() != resource.boneCount) ||
            post.modifiers.size() > 16)
          throw std::runtime_error("invalid GPU pose post process");
        words[base + 9] = bits(post.offsetWeight);
        words[base + 10] = bits(post.velocityWeight);
        if (!post.offsets.empty()) {
          words[base + 8] = static_cast<std::uint32_t>(words.size());
          for (std::size_t bone = 0; bone < post.offsets.size(); ++bone)
            for (const auto *transform :
                 {&post.offsets[bone], &post.velocities[bone]})
              for (float f :
                   {transform->translation.x, transform->translation.y,
                    transform->translation.z, 0.f, transform->rotation.x,
                    transform->rotation.y, transform->rotation.z, 0.f,
                    transform->scale.x, transform->scale.y, transform->scale.z,
                    0.f})
                words.push_back(bits(f));
        }
        if (!post.modifiers.empty()) {
          words[base + 11] = static_cast<std::uint32_t>(words.size());
          words.push_back(static_cast<std::uint32_t>(post.modifiers.size()));
          for (const auto &m : post.modifiers) {
            const auto validBone = [&](int bone) {
              return bone >= 0 &&
                     static_cast<std::uint32_t>(bone) < resource.boneCount;
            };
            if (!validBone(m.chain.upper) ||
                (m.kind == PoseModifier::Kind::LimbIk &&
                 (!validBone(m.chain.lower) || !validBone(m.chain.end))))
              throw std::runtime_error("GPU modifier bone out of range");
            words.insert(words.end(),
                         {static_cast<std::uint32_t>(m.kind),
                          static_cast<std::uint32_t>(m.chain.upper),
                          static_cast<std::uint32_t>(m.chain.lower),
                          static_cast<std::uint32_t>(m.chain.end),
                          bits(m.target.position.x),
                          bits(m.target.position.y),
                          bits(m.target.position.z),
                          bits(m.target.weight),
                          bits(m.target.pole.x),
                          bits(m.target.pole.y),
                          bits(m.target.pole.z),
                          m.target.alignNormal ? 1u : 0u,
                          bits(m.target.normal.x),
                          bits(m.target.normal.y),
                          bits(m.target.normal.z),
                          bits(m.maxRadians),
                          bits(m.forward.x),
                          bits(m.forward.y),
                          bits(m.forward.z),
                          0,
                          0,
                          0,
                          0,
                          0});
          }
        }
      }
    }
    out = std::move(words);
    error.clear();
    return true;
  } catch (const std::exception &e) {
    error = e.what();
    return false;
  }
}
} // namespace odai::anim
