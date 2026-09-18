#include "anim/pose_modifiers.h"
#include <algorithm>
#include <cmath>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace odai::anim {
namespace {
using namespace odai::math;
Quaternion mul(Quaternion a, Quaternion b) {
  return normalize(Quaternion{a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
                              a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
                              a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
                              a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z});
}
Quaternion inv(Quaternion q) {
  q = normalize(q);
  return {-q.x, -q.y, -q.z, q.w};
}
Vector3 logq(Quaternion q) {
  q = normalize(q);
  if (q.w < 0)
    q = {-q.x, -q.y, -q.z, -q.w};
  const float l = length(Vector3{q.x, q.y, q.z});
  return l > 1.e-7f ? Vector3{q.x, q.y, q.z} * (2 * std::atan2(l, q.w) / l)
                    : Vector3{q.x, q.y, q.z} * 2;
}
Quaternion expq(Vector3 v) {
  const float angle = length(v);
  if (angle < 1.e-7f)
    return normalize(Quaternion{v.x * .5f, v.y * .5f, v.z * .5f, 1});
  const float s = std::sin(angle * .5f) / angle;
  return {v.x * s, v.y * s, v.z * s, std::cos(angle * .5f)};
}
Quaternion between(Vector3 a, Vector3 b) {
  a = normalize(a);
  b = normalize(b);
  const float d = std::clamp(dot(a, b), -1.f, 1.f);
  if (d < -.99999f) {
    auto axis = cross(a, Vector3{1, 0, 0});
    if (length(axis) < 1.e-5f)
      axis = cross(a, Vector3{0, 1, 0});
    axis = normalize(axis);
    return {axis.x, axis.y, axis.z, 0};
  }
  const auto c = cross(a, b);
  return normalize(Quaternion{c.x, c.y, c.z, 1 + d});
}
std::vector<Quaternion> rotations(const Skeleton &skeleton,
                                  const LocalPose &pose) {
  std::vector<Quaternion> world(pose.size());
  for (std::size_t i = 0; i < pose.size(); ++i) {
    const int p = skeleton.bones[i].parentIndex;
    world[i] = p < 0 ? pose[i].rotation : mul(world[p], pose[i].rotation);
  }
  return world;
}
bool finite(Vector3 v) {
  return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}
} // namespace
bool solveTwoBoneIk(const Skeleton &skeleton, HumanoidLimbChain chain,
                    const LimbTarget &target, LocalPose &pose) {
  using namespace odai::math;
  if (pose.size() != skeleton.bones.size() || chain.upper < 0 ||
      chain.lower < 0 || chain.end < 0 ||
      static_cast<std::size_t>(
          std::max({chain.upper, chain.lower, chain.end})) >= pose.size() ||
      !finite(target.position) || !finite(target.pole) ||
      !finite(target.normal) || !std::isfinite(target.weight))
    return false;
  // Intermediate twist bones are allowed; the chain roles must be ancestors.
  const auto ancestor = [&](int child, int parent) {
    for (int p = skeleton.bones[child].parentIndex; p >= 0;
         p = skeleton.bones[p].parentIndex)
      if (p == parent)
        return true;
    return false;
  };
  if (!ancestor(chain.lower, chain.upper) || !ancestor(chain.end, chain.lower))
    return false;
  auto world = composePoseWorld(skeleton, pose);
  auto orientation = rotations(skeleton, pose);
  const auto position = [&](int bone) {
    return transformPoint(world[bone], {});
  };
  const auto root = position(chain.upper), mid = position(chain.lower),
             end = position(chain.end);
  const float a = length(mid - root), b = length(end - mid);
  if (a < 1.e-5f || b < 1.e-5f)
    return false;
  auto direction = target.position - root;
  const float distance = length(direction);
  if (distance < 1.e-5f)
    return false;
  direction = direction / distance;
  const float d =
      std::clamp(distance, std::abs(a - b) + 1.e-4f, a + b - 1.e-4f);
  auto bend = (mid - root) - direction * dot(mid - root, direction);
  if (length(bend) < 1.e-4f)
    bend = target.pole - root - direction * dot(target.pole - root, direction);
  if (length(bend) < 1.e-4f)
    bend = cross(direction, std::abs(direction.y) < .9f ? Vector3{0, 1, 0}
                                                        : Vector3{1, 0, 0});
  bend = normalize(bend);
  const float along = (a * a + d * d - b * b) / (2 * d);
  const auto elbow = root + direction * along +
                     bend * std::sqrt(std::max(0.f, a * a - along * along));
  const auto original = pose;
  const auto turn = [&](int bone, Quaternion delta) {
    const int p = skeleton.bones[bone].parentIndex;
    const auto worldRotation = mul(delta, orientation[bone]);
    pose[bone].rotation =
        p < 0 ? worldRotation : mul(inv(orientation[p]), worldRotation);
  };
  turn(chain.upper, between(mid - root, elbow - root));
  world = composePoseWorld(skeleton, pose);
  orientation = rotations(skeleton, pose);
  const auto newMid = position(chain.lower), newEnd = position(chain.end);
  turn(chain.lower, between(newEnd - newMid, root + direction * d - newMid));
  if (target.alignNormal && length(target.normal) > 1.e-5f) {
    orientation = rotations(skeleton, pose);
    const auto up = transformPoint(toMatrix(orientation[chain.end]), {0, 1, 0});
    turn(chain.end, between(up, target.normal));
  }
  const float weight = std::clamp(target.weight, 0.f, 1.f);
  for (int bone : {chain.upper, chain.lower, chain.end})
    pose[bone].rotation =
        slerp(original[bone].rotation, pose[bone].rotation, weight);
  return true;
}
bool aimBone(const Skeleton &skeleton, int bone, odai::math::Vector3 target,
             odai::math::Vector3 forward, float maxRadians, float weight,
             LocalPose &pose) {
  using namespace odai::math;
  if (bone < 0 || static_cast<std::size_t>(bone) >= pose.size() ||
      pose.size() != skeleton.bones.size() || !finite(target) ||
      !finite(forward) || !std::isfinite(maxRadians) || !std::isfinite(weight))
    return false;
  const auto world = composePoseWorld(skeleton, pose);
  const auto q = rotations(skeleton, pose);
  const auto direction = target - transformPoint(world[bone], {});
  if (length(direction) < 1.e-5f || length(forward) < 1.e-5f)
    return false;
  auto delta =
      logq(between(transformPoint(toMatrix(q[bone]), forward), direction));
  const float angle = length(delta);
  if (angle > 0)
    delta = delta * (std::min(angle, std::clamp(maxRadians, 0.f, 3.14159265f)) /
                     angle * std::clamp(weight, 0.f, 1.f));
  const int p = skeleton.bones[bone].parentIndex;
  const auto result = mul(expq(delta), q[bone]);
  pose[bone].rotation = p < 0 ? result : mul(inv(q[p]), result);
  return true;
}
LocalPose PoseInertializer::evaluate(const LocalPose &target, bool interrupted,
                                     float duration, float delta,
                                     const LocalPose *targetPrevious) {
  using namespace odai::math;
  if (!std::isfinite(delta) || delta < 0 || !std::isfinite(duration) ||
      duration < 0)
    return target;
  if (previous_.size() != target.size()) {
    reset();
    previous_ = target;
    older_ = target;
    previousDelta_ = delta;
    return target;
  }
  if (interrupted && duration > 0) {
    elapsed_ = 0;
    duration_ = std::min(duration, 10.f);
    offsets_.resize(target.size());
    velocities_.resize(target.size());
    for (std::size_t i = 0; i < target.size(); ++i) {
      const auto r = logq(mul(previous_[i].rotation, inv(target[i].rotation)));
      offsets_[i] = {previous_[i].translation - target[i].translation,
                     {r.x, r.y, r.z, 0},
                     previous_[i].scale - target[i].scale};
      const float invDelta = previousDelta_ > 1.e-6f ? 1 / previousDelta_ : 0;
      const auto v =
          logq(mul(previous_[i].rotation, inv(older_[i].rotation))) * invDelta;
      velocities_[i] = {(previous_[i].translation - older_[i].translation) *
                            invDelta,
                        {v.x, v.y, v.z, 0},
                        (previous_[i].scale - older_[i].scale) * invDelta};
      if (targetPrevious && targetPrevious->size() == target.size() &&
          delta > 1.e-6f) {
        const auto &before = (*targetPrevious)[i];
        velocities_[i].translation =
            velocities_[i].translation -
            (target[i].translation - before.translation) / delta;
        velocities_[i].scale =
            velocities_[i].scale - (target[i].scale - before.scale) / delta;
        const auto incoming =
            logq(mul(target[i].rotation, inv(before.rotation))) / delta;
        velocities_[i].rotation.x -= incoming.x;
        velocities_[i].rotation.y -= incoming.y;
        velocities_[i].rotation.z -= incoming.z;
      }
    }
  }
  LocalPose result = target;
  offsetWeight_ = velocityWeight_ = 0;
  if (elapsed_ < duration_ && offsets_.size() == target.size()) {
    const float t = elapsed_ / duration_;
    // Cubic Hermite offset: starts at outgoing pose/velocity and reaches
    // zero offset/velocity exactly at duration, with no end-of-blend snap.
    const float h0 = 2 * t * t * t - 3 * t * t + 1,
                h1 = (t * t * t - 2 * t * t + t) * duration_;
    offsetWeight_ = h0;
    velocityWeight_ = h1;
    for (std::size_t i = 0; i < result.size(); ++i) {
      const auto &o = offsets_[i];
      const auto &v = velocities_[i];
      result[i].translation =
          result[i].translation + o.translation * h0 + v.translation * h1;
      result[i].scale = result[i].scale + o.scale * h0 + v.scale * h1;
      const Vector3 rotation =
          Vector3{o.rotation.x, o.rotation.y, o.rotation.z} * h0 +
          Vector3{v.rotation.x, v.rotation.y, v.rotation.z} * h1;
      result[i].rotation = mul(expq(rotation), result[i].rotation);
    }
    elapsed_ = std::min(duration_, elapsed_ + delta);
  }
  older_ = previous_;
  previous_ = result;
  previousDelta_ = delta;
  return result;
}
PosePostProcess PoseInertializer::evaluationParameters() const {
  PosePostProcess result;
  if (offsetWeight_ != 0 || velocityWeight_ != 0) {
    result.offsets = offsets_;
    result.velocities = velocities_;
    result.offsetWeight = offsetWeight_;
    result.velocityWeight = velocityWeight_;
  }
  return result;
}
bool applyPosePostProcess(const Skeleton &skeleton, const PosePostProcess &post,
                          LocalPose &pose) {
  using namespace odai::math;
  if (pose.size() != skeleton.bones.size() ||
      post.offsets.size() != post.velocities.size() ||
      (!post.offsets.empty() && post.offsets.size() != pose.size()) ||
      !std::isfinite(post.offsetWeight) || !std::isfinite(post.velocityWeight))
    return false;
  for (std::size_t i = 0; i < post.offsets.size(); ++i) {
    const auto &o = post.offsets[i];
    const auto &v = post.velocities[i];
    pose[i].translation = pose[i].translation +
                          o.translation * post.offsetWeight +
                          v.translation * post.velocityWeight;
    pose[i].scale = pose[i].scale + o.scale * post.offsetWeight +
                    v.scale * post.velocityWeight;
    pose[i].rotation =
        mul(expq(Vector3{o.rotation.x, o.rotation.y, o.rotation.z} *
                     post.offsetWeight +
                 Vector3{v.rotation.x, v.rotation.y, v.rotation.z} *
                     post.velocityWeight),
            pose[i].rotation);
  }
  for (const auto &operation : post.modifiers) {
    switch (operation.kind) {
    case PoseModifier::Kind::Translate:
      if (operation.chain.upper < 0 ||
          static_cast<std::size_t>(operation.chain.upper) >= pose.size() ||
          !finite(operation.target.position) ||
          !std::isfinite(operation.target.weight))
        return false;
      pose[operation.chain.upper].translation =
          pose[operation.chain.upper].translation +
          operation.target.position *
              std::clamp(operation.target.weight, 0.f, 1.f);
      break;
    case PoseModifier::Kind::LimbIk:
      if (!solveTwoBoneIk(skeleton, operation.chain, operation.target, pose))
        return false;
      break;
    case PoseModifier::Kind::Aim:
      if (!aimBone(skeleton, operation.chain.upper, operation.target.position,
                   operation.forward, operation.maxRadians,
                   operation.target.weight, pose))
        return false;
      break;
    }
  }
  return true;
}

void PoseInertializer::reset() {
  offsetWeight_ = velocityWeight_ = 0;
  previous_.clear();
  older_.clear();
  offsets_.clear();
  velocities_.clear();
  elapsed_ = duration_ = previousDelta_ = 0;
}
std::string PoseInertializer::save() const {
  if (previous_.empty())
    return {};
  using Json = nlohmann::json;
  const auto encode = [](const LocalPose &pose) {
    Json a = Json::array();
    for (const auto &p : pose)
      a.push_back({p.translation.x, p.translation.y, p.translation.z,
                   p.rotation.x, p.rotation.y, p.rotation.z, p.rotation.w,
                   p.scale.x, p.scale.y, p.scale.z});
    return a;
  };
  return Json{
      {"previous", encode(previous_)}, {"older", encode(older_)},
      {"offsets", encode(offsets_)},   {"velocities", encode(velocities_)},
      {"elapsed", elapsed_},           {"duration", duration_},
      {"delta", previousDelta_}}
      .dump();
}
bool PoseInertializer::restore(std::string_view saved, std::string &error) {
  try {
    if (saved.empty()) {
      reset();
      error.clear();
      return true;
    }
    if (saved.size() > 16 * 1024 * 1024)
      throw std::runtime_error("inertial snapshot limit");
    const auto j = nlohmann::json::parse(
        saved, [](int depth, nlohmann::json::parse_event_t, nlohmann::json &) {
          if (depth > 4)
            throw std::runtime_error("inertial nesting limit");
          return true;
        });
    const auto decode = [](const auto &a) {
      LocalPose pose;
      if (!a.is_array() || a.size() > 65535)
        throw std::runtime_error("inertial bone limit");
      for (const auto &v : a) {
        if (!v.is_array() || v.size() != 10)
          throw std::runtime_error("invalid inertial transform");
        float f[10];
        for (int i = 0; i < 10; ++i) {
          f[i] = v.at(i).template get<float>();
          if (!std::isfinite(f[i]))
            throw std::runtime_error("nonfinite inertial pose");
        }
        pose.push_back(
            {{f[0], f[1], f[2]}, {f[3], f[4], f[5], f[6]}, {f[7], f[8], f[9]}});
      }
      return pose;
    };
    PoseInertializer result;
    result.previous_ = decode(j.at("previous"));
    result.older_ = decode(j.at("older"));
    result.offsets_ = decode(j.at("offsets"));
    result.velocities_ = decode(j.at("velocities"));
    result.elapsed_ = j.at("elapsed").get<float>();
    result.duration_ = j.at("duration").get<float>();
    result.previousDelta_ = j.at("delta").get<float>();
    if (result.previous_.size() != result.older_.size() ||
        result.offsets_.size() != result.velocities_.size() ||
        (!result.offsets_.empty() &&
         result.offsets_.size() != result.previous_.size()) ||
        !std::isfinite(result.elapsed_) || result.elapsed_ < 0 ||
        !std::isfinite(result.duration_) || result.duration_ < 0 ||
        result.duration_ > 10 || result.elapsed_ > result.duration_ ||
        !std::isfinite(result.previousDelta_) || result.previousDelta_ < 0 ||
        result.previousDelta_ > .25f)
      throw std::runtime_error("invalid inertial snapshot");
    *this = std::move(result);
    error.clear();
    return true;
  } catch (const std::exception &e) {
    error = e.what();
    return false;
  }
}
} // namespace odai::anim
