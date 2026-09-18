#include "anim/pose_graph.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <nlohmann/json.hpp>
#include <set>
#include <stdexcept>

namespace odai::anim {
namespace {
using Json = nlohmann::json;
PoseParameter parameter(const Json &j) {
  if (j.is_boolean())
    return j.get<bool>();
  if (j.is_string() && j.get_ref<const std::string &>().size() <= 1024)
    return j.get<std::string>();
  if (j.is_number() && std::isfinite(j.get<double>()))
    return j.get<double>();
  throw std::runtime_error("expected finite typed parameter");
}
const AnimationClip *findClip(std::span<const AnimationClip> clips,
                              const std::string &name) {
  const auto found =
      std::find_if(clips.begin(), clips.end(),
                   [&](const auto &c) { return c.name == name; });
  return found == clips.end() ? nullptr : &*found;
}
float numeric(const std::map<std::string, PoseParameter> &params,
              const std::string &name, float fallback) {
  if (name.empty())
    return fallback;
  const auto found = params.find(name);
  if (found == params.end())
    throw std::runtime_error("missing graph parameter: " + name);
  const auto *value = std::get_if<double>(&found->second);
  if (!value || !std::isfinite(*value) ||
      std::abs(*value) > std::numeric_limits<float>::max())
    throw std::runtime_error("invalid numeric graph parameter: " + name);
  return static_cast<float>(*value);
}
} // namespace
bool compilePoseGraph(std::string_view text, PoseGraphProgram &out,
                      std::string &error) {
  try {
    if (text.size() > 4 * 1024 * 1024)
      throw std::runtime_error("graph exceeds 4 MiB");
    const auto root =
        Json::parse(text, [](int depth, Json::parse_event_t, Json &) {
          if (depth > 32)
            throw std::runtime_error("graph nesting limit");
          return true;
        });
    PoseGraphProgram result;
    result.root = root.at("root").get<std::string>();
    const auto params = root.value("parameters", Json::object());
    if (!params.is_object() || params.size() > 64)
      throw std::runtime_error("graph parameter limit");
    for (const auto &[name, v] : params.items()) {
      if (name.empty())
        throw std::runtime_error("empty parameter name");
      result.parameters[name] = parameter(v);
    }
    const auto &nodes = root.at("nodes");
    if (!nodes.is_array() || nodes.empty() || nodes.size() > 256)
      throw std::runtime_error("graph node limit");
    for (const auto &entry : nodes) {
      PoseGraphNode n;
      n.id = entry.at("id").get<std::string>();
      const auto kind = entry.at("type").get<std::string>();
      if (kind == "clip")
        n.kind = PoseGraphNode::Kind::Clip;
      else if (kind == "blend")
        n.kind = PoseGraphNode::Kind::Blend;
      else if (kind == "blend1d")
        n.kind = PoseGraphNode::Kind::Blend1D;
      else if (kind == "blend2d")
        n.kind = PoseGraphNode::Kind::Blend2D;
      else if (kind == "layer")
        n.kind = PoseGraphNode::Kind::Layer;
      else if (kind == "cache")
        n.kind = PoseGraphNode::Kind::Cache;
      else if (kind == "limb_ik" || kind == "aim" || kind == "translate")
        n.kind = PoseGraphNode::Kind::Modifier;
      else if (kind == "state_machine")
        n.kind = PoseGraphNode::Kind::StateMachine;
      else
        throw std::runtime_error("unsupported pose node: " + kind);
      n.clip = entry.value("clip", std::string{});
      n.reference = entry.value("reference_clip", std::string{});
      n.inputs = entry.value("inputs", std::vector<std::string>{});
      n.bones = entry.value("bones", std::vector<std::string>{});
      n.parameter = entry.value("parameter", std::string{});
      n.parameterY = entry.value("parameter_y", std::string{});
      n.initial = entry.value("initial", std::string{});
      n.weight = entry.value("weight", 1.f);
      n.speed = entry.value("speed", 1.f);
      n.additive = entry.value("additive", false);
      if (n.kind == PoseGraphNode::Kind::Modifier) {
        n.role = entry.at("role").get<std::string>();
        const auto target =
            entry.at("target_parameters").get<std::vector<std::string>>();
        if (target.size() != 3 || n.role.empty())
          throw std::runtime_error("invalid modifier target");
        n.targetX = target[0];
        n.targetY = target[1];
        n.targetZ = target[2];
        for (const auto &key : target)
          if (!result.parameters.contains(key) ||
              !std::holds_alternative<double>(result.parameters.at(key)))
            throw std::runtime_error("undeclared modifier target parameter: " +
                                     key);
        n.modifier.kind = kind == "limb_ik" ? PoseModifier::Kind::LimbIk
                          : kind == "aim"   ? PoseModifier::Kind::Aim
                                            : PoseModifier::Kind::Translate;
        n.modifier.maxRadians = entry.value("max_radians", .5f);
        if (!std::isfinite(n.modifier.maxRadians) ||
            n.modifier.maxRadians < 0 || n.modifier.maxRadians > 3.141593f)
          throw std::runtime_error("invalid aim limit");
      }
      if (n.id.empty() || n.id.size() > 256 || n.inputs.size() > 32 ||
          n.bones.size() > 4096 || !std::isfinite(n.weight) || n.weight < 0 ||
          n.weight > 1 || !std::isfinite(n.speed) || n.speed < 0 || n.speed > 4)
        throw std::runtime_error("invalid pose node");
      for (const auto *key : {&n.parameter, &n.parameterY})
        if (!key->empty() &&
            (!result.parameters.contains(*key) ||
             !std::holds_alternative<double>(result.parameters.at(*key))))
          throw std::runtime_error("undeclared numeric parameter: " + *key);
      if (entry.contains("samples")) {
        if (!entry.at("samples").is_array() || entry.at("samples").size() > 32)
          throw std::runtime_error("blend sample limit");
        std::set<std::pair<float, float>> coordinates;
        for (const auto &s : entry.at("samples")) {
          PoseGraphSample sample{s.at("node").get<std::string>(),
                                 s.at("x").get<float>(), s.value("y", 0.f)};
          if (!std::isfinite(sample.x) || !std::isfinite(sample.y) ||
              !coordinates.emplace(sample.x, sample.y).second)
            throw std::runtime_error("invalid blend sample");
          n.samples.push_back(std::move(sample));
        }
      }
      if (entry.contains("transitions")) {
        if (!entry.at("transitions").is_array() ||
            entry.at("transitions").size() > 128)
          throw std::runtime_error("transition limit");
        for (const auto &t : entry.at("transitions")) {
          PoseGraphTransition v{
              t.at("from").get<std::string>(), t.at("to").get<std::string>(),
              t.at("parameter").get<std::string>(), parameter(t.at("equals")),
              t.value("duration", .16f)};
          if (v.from == v.to || !result.parameters.contains(v.parameter) ||
              result.parameters.at(v.parameter).index() != v.equals.index() ||
              !std::isfinite(v.duration) || v.duration < 0 || v.duration > 10 ||
              std::find(n.inputs.begin(), n.inputs.end(), v.from) ==
                  n.inputs.end() ||
              std::find(n.inputs.begin(), n.inputs.end(), v.to) ==
                  n.inputs.end())
            throw std::runtime_error("invalid state transition");
          n.transitions.push_back(std::move(v));
        }
      }
      using K = PoseGraphNode::Kind;
      if ((n.kind == K::Clip && (n.clip.empty() || !n.inputs.empty())) ||
          ((n.kind == K::Blend || n.kind == K::Layer) &&
           n.inputs.size() != 2) ||
          ((n.kind == K::Cache || n.kind == K::Modifier) &&
           n.inputs.size() != 1) ||
          ((n.kind == K::Blend1D || n.kind == K::Blend2D) &&
           n.samples.empty()) ||
          (n.kind == K::Layer && n.additive && n.reference.empty()) ||
          (n.kind == K::StateMachine &&
           std::find(n.inputs.begin(), n.inputs.end(), n.initial) ==
               n.inputs.end()))
        throw std::runtime_error("invalid pose node arity or required field: " +
                                 n.id);
      if (n.kind == K::Blend1D) {
        std::set<float> x;
        for (const auto &sample : n.samples)
          if (!x.insert(sample.x).second)
            throw std::runtime_error("duplicate 1D coordinate");
      }
      if (!result.nodes.emplace(n.id, std::move(n)).second)
        throw std::runtime_error("duplicate pose node");
    }
    std::map<std::string, int> colors;
    std::function<void(const std::string &, int)> visit =
        [&](const std::string &id, int depth) {
          if (depth > 32 || !result.nodes.contains(id))
            throw std::runtime_error("missing node or graph depth limit: " +
                                     id);
          if (colors[id] == 1)
            throw std::runtime_error("pose dependency cycle: " + id);
          if (colors[id] == 2)
            return;
          colors[id] = 1;
          const auto &n = result.nodes.at(id);
          for (const auto &input : n.inputs)
            visit(input, depth + 1);
          for (const auto &s : n.samples)
            visit(s.node, depth + 1);
          colors[id] = 2;
        };
    visit(result.root, 0);
    for (const auto &[id, n] : result.nodes)
      visit(id, 0);
    out = std::move(result);
    error.clear();
    return true;
  } catch (const std::exception &e) {
    error = e.what();
    return false;
  }
}

std::vector<float> blendSpaceWeights(std::span<const PoseGraphSample> samples,
                                     float x, float y, bool two) {
  std::vector<float> result(samples.size());
  if (samples.empty() || !std::isfinite(x) || !std::isfinite(y))
    return result;
  // Admit triangles in stable order, rejecting crossing diagonals. The
  // explicit edge set matters for cocircular samples: empty-circle tests alone
  // admit overlapping triangles and produce a discontinuous blend surface.
  std::vector<std::pair<std::size_t, std::size_t>> edges;
  const auto side = [&](std::size_t a, std::size_t b, std::size_t c) {
    return (double(samples[b].x) - samples[a].x) *
               (double(samples[c].y) - samples[a].y) -
           (double(samples[b].y) - samples[a].y) *
               (double(samples[c].x) - samples[a].x);
  };
  if (two)
    for (std::size_t a = 0; a < samples.size(); ++a)
      for (std::size_t b = a + 1; b < samples.size(); ++b)
        for (std::size_t c = b + 1; c < samples.size(); ++c) {
          const double ax = samples[a].x, ay = samples[a].y;
          const double bx = samples[b].x - ax, by = samples[b].y - ay,
                       cx = samples[c].x - ax, cy = samples[c].y - ay;
          const double det = bx * cy - by * cx;
          if (std::abs(det) < 1.e-10)
            continue;
          const double ux =
              ((bx * bx + by * by) * cy - (cx * cx + cy * cy) * by) / (2 * det);
          const double uy =
              ((cx * cx + cy * cy) * bx - (bx * bx + by * by) * cx) / (2 * det);
          const double radius = ux * ux + uy * uy;
          bool empty = true;
          for (std::size_t d = 0; d < samples.size(); ++d)
            if (d != a && d != b && d != c) {
              const double dx = samples[d].x - ax - ux,
                           dy = samples[d].y - ay - uy;
              if (dx * dx + dy * dy < radius - 1.e-7 * std::max(1., radius)) {
                empty = false;
                break;
              }
            }
          if (!empty)
            continue;
          const std::array<std::pair<std::size_t, std::size_t>, 3> candidate{
              {{a, b}, {b, c}, {c, a}}};
          bool crossing = false;
          for (const auto &[u, v] : candidate)
            for (const auto &[p, q] : edges) {
              if (u == p || u == q || v == p || v == q)
                continue;
              if (side(u, v, p) * side(u, v, q) < 0 &&
                  side(p, q, u) * side(p, q, v) < 0)
                crossing = true;
            }
          if (crossing)
            continue;
          edges.insert(edges.end(), candidate.begin(), candidate.end());
          const double px = x - ax, py = y - ay;
          const double wb = (px * cy - py * cx) / det,
                       wc = (bx * py - by * px) / det, wa = 1 - wb - wc;
          if (wa >= -1.e-7 && wb >= -1.e-7 && wc >= -1.e-7) {
            result[a] = static_cast<float>(std::max(0., wa));
            result[b] = static_cast<float>(std::max(0., wb));
            result[c] = static_cast<float>(std::max(0., wc));
            return result;
          }
        }
  // Outside the hull (or collinear): project onto the nearest segment.
  // Interior segments cannot be closer than a hull boundary from outside.
  double best = std::numeric_limits<double>::infinity();
  for (std::size_t a = 0; a < samples.size(); ++a)
    for (std::size_t b = a; b < samples.size(); ++b) {
      const double dx = static_cast<double>(samples[b].x) - samples[a].x;
      const double dy =
          two ? static_cast<double>(samples[b].y) - samples[a].y : 0;
      const double px = static_cast<double>(x) - samples[a].x,
                   py = two ? static_cast<double>(y) - samples[a].y : 0;
      const double denom = dx * dx + dy * dy;
      const double t =
          denom > 0 ? std::clamp((px * dx + py * dy) / denom, 0., 1.) : 0.;
      const double distance =
          (px - t * dx) * (px - t * dx) + (py - t * dy) * (py - t * dy);
      // Prefer shortest spanning interval in 1D, not an arbitrary long chord.
      const double score = distance + (two ? 0 : denom * 1.e-14);
      if (score < best) {
        best = score;
        std::fill(result.begin(), result.end(), 0.f);
        result[a] += static_cast<float>(1 - t);
        result[b] += static_cast<float>(t);
      }
    }
  return result;
}
float synchronizedClipTime(const AnimationClip &leader, float time,
                           const AnimationClip &follower) {
  if (leader.duration <= 0 || follower.duration <= 0 || !std::isfinite(time))
    return 0;
  const float cycle = std::floor(time / leader.duration),
              local = time - cycle * leader.duration;
  std::vector<const AnimationAnnotation *> markers;
  for (const auto &m : leader.annotations)
    if (m.name.starts_with("sync:") && m.time >= 0 && m.time < leader.duration)
      markers.push_back(&m);
  if (markers.size() >= 2) {
    std::sort(markers.begin(), markers.end(),
              [](auto a, auto b) { return a->time < b->time; });
    for (std::size_t i = 0; i < markers.size(); ++i) {
      const auto &a = *markers[i];
      const auto &b = *markers[(i + 1) % markers.size()];
      const float end =
          b.time + (i + 1 == markers.size() ? leader.duration : 0);
      float at = local;
      if (at < a.time)
        at += leader.duration;
      if (at < a.time || at > end || end <= a.time)
        continue;
      const auto fa =
          std::find_if(follower.annotations.begin(), follower.annotations.end(),
                       [&](const auto &m) { return m.name == a.name; });
      const auto fb =
          std::find_if(follower.annotations.begin(), follower.annotations.end(),
                       [&](const auto &m) { return m.name == b.name; });
      if (fa == follower.annotations.end() || fb == follower.annotations.end())
        break;
      float targetEnd = fb->time;
      if (targetEnd <= fa->time)
        targetEnd += follower.duration;
      float mapped =
          fa->time + (at - a.time) / (end - a.time) * (targetEnd - fa->time);
      if (local < a.time)
        mapped -= follower.duration;
      return cycle * follower.duration + mapped;
    }
  }
  return time / leader.duration * follower.duration;
}

bool PoseGraphInstance::advance(
    const PoseGraphProgram &graph, const Skeleton &skeleton,
    const HumanoidRigMapping *rig,
    const std::map<std::string, PoseParameter> &inputs,
    std::span<const AnimationClip> clips, float delta,
    PoseEvaluationPacket &out, std::string &error) {
  const auto saved = clocks_;
  const auto savedEvent = lastEventClip_;
  try {
    if (!std::isfinite(delta) || delta < 0 || delta > .25f)
      throw std::runtime_error("invalid graph delta");
    auto parameters = graph.parameters;
    for (const auto &[name, value] : inputs)
      if (parameters.contains(name)) {
        if (parameters[name].index() != value.index())
          throw std::runtime_error("graph input type mismatch: " + name);
        if (const auto *number = std::get_if<double>(&value);
            number && !std::isfinite(*number))
          throw std::runtime_error("nonfinite graph input: " + name);
        parameters[name] = value;
      }
    PoseEvaluationPacket packet;
    std::map<std::string, std::uint32_t> cached;
    std::function<std::uint32_t(const std::string &)> emit =
        [&](const std::string &id) {
          if (cached.contains(id))
            return cached.at(id);
          const auto &node = graph.nodes.at(id);
          auto &clock = clocks_[id];
          PoseGraphInstruction inst;
          inst.kind = node.kind;
          inst.weight = numeric(parameters, node.parameter, node.weight);
          inst.additive = node.additive;
          inst.reference = node.reference;
          using K = PoseGraphNode::Kind;
          if (node.kind == K::Clip) {
            if (!findClip(clips, node.clip))
              throw std::runtime_error("missing graph clip: " + node.clip);
            inst.clip = node.clip;
            inst.previousTime = static_cast<float>(clock.time);
            clock.time += delta * node.speed;
            inst.time = static_cast<float>(clock.time);
          } else if (node.kind == K::StateMachine) {
            if (clock.state.empty())
              clock.state = node.initial;
            if (std::find(node.inputs.begin(), node.inputs.end(),
                          clock.state) == node.inputs.end())
              throw std::runtime_error("saved state is unavailable");
            for (const auto &transition : node.transitions)
              if (transition.from == clock.state &&
                  parameters.at(transition.parameter) == transition.equals) {
                clock.previous = clock.state;
                clock.state = transition.to;
                clock.elapsed = 0;
                clock.duration = transition.duration;
                packet.discontinuity = true;
                packet.transitionDuration = transition.duration;
                break;
              }
            if (!clock.previous.empty() && clock.elapsed < clock.duration) {
              inst.inputs.push_back(emit(clock.previous));
              inst.inputs.push_back(emit(clock.state));
              inst.weight =
                  clock.duration > 0 ? clock.elapsed / clock.duration : 1;
              clock.elapsed += delta;
            } else {
              inst.inputs.push_back(emit(clock.state));
              clock.previous.clear();
            }
          } else if (node.kind == K::Blend1D || node.kind == K::Blend2D) {
            const auto weights = blendSpaceWeights(
                node.samples, numeric(parameters, node.parameter, 0),
                numeric(parameters, node.parameterY, 0),
                node.kind == K::Blend2D);
            const float before = static_cast<float>(clock.time);
            clock.time += delta;
            const auto directClip = [&](std::uint32_t index) {
              while (packet.instructions[index].kind == K::Cache &&
                     packet.instructions[index].inputs.size() == 1)
                index = packet.instructions[index].inputs[0];
              return index;
            };
            // The first authored sample leads the sync group even at zero
            // blend weight. Re-entering samples cannot change the group clock.
            const auto leaderIndex =
                directClip(emit(node.samples.front().node));
            const auto *leader =
                findClip(clips, packet.instructions[leaderIndex].clip);
            for (std::size_t i = 0; i < weights.size(); ++i)
              if (weights[i] > 0) {
                auto index = emit(node.samples[i].node);
                const auto leaf = directClip(index);
                if (leader && packet.instructions[leaf].kind == K::Clip) {
                  auto sample = packet.instructions[leaf];
                  const auto *follower = findClip(clips, sample.clip);
                  sample.time = synchronizedClipTime(
                      *leader, static_cast<float>(clock.time), *follower);
                  sample.previousTime =
                      synchronizedClipTime(*leader, before, *follower);
                  index =
                      static_cast<std::uint32_t>(packet.instructions.size());
                  packet.instructions.push_back(std::move(sample));
                }
                inst.inputs.push_back(index);
                inst.weights.push_back(weights[i]);
              }
          } else
            for (const auto &input : node.inputs)
              inst.inputs.push_back(emit(input));
          if (node.kind == K::Modifier) {
            auto modifier = node.modifier;
            if (!rig)
              throw std::runtime_error(
                  "procedural node requires a humanoid rig");
            if (modifier.kind == PoseModifier::Kind::LimbIk) {
              const auto found = rig->limbs.find(node.role);
              if (found == rig->limbs.end())
                throw std::runtime_error("missing limb: " + node.role);
              modifier.chain = found->second;
            } else {
              const auto found = rig->roles.find(node.role);
              if (found == rig->roles.end())
                throw std::runtime_error("missing bone role: " + node.role);
              modifier.chain.upper = found->second;
            }
            modifier.target.position = {numeric(parameters, node.targetX, 0),
                                        numeric(parameters, node.targetY, 0),
                                        numeric(parameters, node.targetZ, 0)};
            modifier.target.weight = std::clamp(inst.weight, 0.f, 1.f);
            auto post = std::make_shared<PosePostProcess>();
            post->modifiers.push_back(modifier);
            inst.postProcess = std::move(post);
          }
          if (!node.bones.empty()) {
            inst.mask.resize(skeleton.bones.size());
            for (const auto &name : node.bones) {
              int bone = skeleton.findBone(name);
              if (name.starts_with("role:") && rig) {
                const auto found = rig->roles.find(name.substr(5));
                if (found != rig->roles.end())
                  bone = found->second;
              }
              if (bone < 0 ||
                  static_cast<std::size_t>(bone) >= inst.mask.size())
                throw std::runtime_error("missing graph mask bone: " + name);
              inst.mask[bone] = 1;
            }
          }
          if (packet.instructions.size() >= 256)
            throw std::runtime_error("expanded pose packet limit");
          const auto index =
              static_cast<std::uint32_t>(packet.instructions.size());
          packet.instructions.push_back(std::move(inst));
          cached[id] = index;
          return index;
        };
    packet.output = emit(graph.root);
    // Follow dominant base branches; additive layers never own events.
    std::uint32_t authority = packet.output;
    for (;;) {
      const auto &i = packet.instructions[authority];
      if (i.kind == PoseGraphNode::Kind::Clip)
        break;
      if (i.inputs.empty())
        throw std::runtime_error("empty evaluated graph node");
      std::size_t chosen = 0;
      if (!i.weights.empty())
        chosen = static_cast<std::size_t>(
            std::max_element(i.weights.begin(), i.weights.end()) -
            i.weights.begin());
      else if (i.inputs.size() == 2 && i.kind != PoseGraphNode::Kind::Layer)
        chosen = i.weight >= .5f ? 1 : 0;
      authority = i.inputs[chosen];
    }
    const auto &event = packet.instructions[authority];
    packet.eventClip = event.clip;
    packet.eventAfter = event.time;
    packet.eventBefore = event.previousTime;
    packet.discontinuity = packet.discontinuity || lastEventClip_ != event.clip;
    lastEventClip_ = event.clip;
    out = std::move(packet);
    error.clear();
    return true;
  } catch (const std::exception &e) {
    clocks_ = saved;
    lastEventClip_ = savedEvent;
    error = e.what();
    return false;
  }
}
bool evaluatePosePacket(const PoseEvaluationPacket &packet,
                        const Skeleton &skeleton,
                        std::span<const AnimationClip> clips, LocalPose &out,
                        std::string &error) {
  try {
    if (packet.instructions.empty() || packet.instructions.size() > 256 ||
        packet.output >= packet.instructions.size())
      throw std::runtime_error("invalid pose packet");
    std::vector<LocalPose> poses;
    for (const auto &inst : packet.instructions) {
      if (!std::isfinite(inst.time) || !std::isfinite(inst.weight) ||
          inst.weight < 0 || inst.weight > 1 ||
          (!inst.mask.empty() && inst.mask.size() != skeleton.bones.size()) ||
          (inst.weights.empty() && inst.inputs.size() > 2) ||
          (inst.additive && (!inst.weights.empty() || inst.inputs.size() != 2)))
        throw std::runtime_error("invalid pose instruction values");
      for (float weight : inst.mask)
        if (!std::isfinite(weight) || weight < 0 || weight > 1)
          throw std::runtime_error("invalid pose mask");
      for (float weight : inst.weights)
        if (!std::isfinite(weight) || weight < 0 || weight > 1)
          throw std::runtime_error("invalid pose weights");
      for (auto index : inst.inputs)
        if (index >= poses.size())
          throw std::runtime_error("pose packet dependency order");
      LocalPose pose;
      if (inst.kind == PoseGraphNode::Kind::Clip) {
        const auto *clip = findClip(clips, inst.clip);
        if (!clip)
          throw std::runtime_error("missing evaluated clip");
        pose = sampleLocalPose(skeleton, *clip, inst.time);
      } else {
        if (inst.inputs.empty())
          throw std::runtime_error("empty pose packet inputs");
        pose = poses[inst.inputs[0]];
        if (!inst.weights.empty()) {
          if (inst.weights.size() != inst.inputs.size())
            throw std::runtime_error("invalid blend weights");
          float total = inst.weights[0];
          for (std::size_t i = 1; i < inst.inputs.size(); ++i) {
            total += inst.weights[i];
            pose = blendLocalPoses(pose, poses[inst.inputs[i]],
                                   total > 0 ? inst.weights[i] / total : 0,
                                   inst.mask);
          }
        } else if (inst.inputs.size() == 2) {
          if (inst.additive) {
            const auto *ref = findClip(clips, inst.reference);
            if (!ref)
              throw std::runtime_error("missing additive reference");
            pose = addLocalPoses(pose, poses[inst.inputs[1]],
                                 sampleLocalPose(skeleton, *ref, 0),
                                 inst.weight, inst.mask);
          } else
            pose = blendLocalPoses(pose, poses[inst.inputs[1]], inst.weight,
                                   inst.mask);
        }
      }
      if (inst.postProcess &&
          !applyPosePostProcess(skeleton, *inst.postProcess, pose))
        throw std::runtime_error("invalid pose modifiers");
      poses.push_back(std::move(pose));
    }
    out = std::move(poses[packet.output]);
    error.clear();
    return true;
  } catch (const std::exception &e) {
    error = e.what();
    return false;
  }
}
void PoseGraphInstance::reset() {
  clocks_.clear();
  lastEventClip_.clear();
}
std::string PoseGraphInstance::save() const {
  if (clocks_.empty())
    return {};
  Json j;
  j["event"] = lastEventClip_;
  j["clocks"] = Json::object();
  for (const auto &[id, c] : clocks_)
    j["clocks"][id] = {{"time", c.time},
                       {"state", c.state},
                       {"previous", c.previous},
                       {"elapsed", c.elapsed},
                       {"duration", c.duration}};
  return j.dump();
}
bool PoseGraphInstance::restore(std::string_view text, std::string &error) {
  try {
    if (text.empty()) {
      reset();
      error.clear();
      return true;
    }
    if (text.size() > 1024 * 1024)
      throw std::runtime_error("graph snapshot limit");
    const auto j =
        Json::parse(text, [](int depth, Json::parse_event_t, Json &) {
          if (depth > 8)
            throw std::runtime_error("snapshot nesting limit");
          return true;
        });
    std::map<std::string, Clock> clocks;
    const auto &saved = j.at("clocks");
    if (!saved.is_object() || saved.size() > 256)
      throw std::runtime_error("saved graph clock limit");
    for (const auto &[id, v] : saved.items()) {
      Clock c{v.at("time").get<double>(), v.at("state").get<std::string>(),
              v.at("previous").get<std::string>(), v.at("elapsed").get<float>(),
              v.at("duration").get<float>()};
      if (id.empty() || id.size() > 256 || !std::isfinite(c.time) ||
          c.time < 0 || c.time > 1.e9 || !std::isfinite(c.elapsed) ||
          c.elapsed < 0 || c.elapsed > 10.25f || !std::isfinite(c.duration) ||
          c.duration < 0 || c.duration > 10 || c.state.size() > 256 ||
          c.previous.size() > 256)
        throw std::runtime_error("invalid saved graph clock");
      clocks.emplace(id, std::move(c));
    }
    auto event = j.value("event", std::string{});
    if (event.size() > 4096)
      throw std::runtime_error("invalid saved event clip");
    clocks_ = std::move(clocks);
    lastEventClip_ = std::move(event);
    error.clear();
    return true;
  } catch (const std::exception &e) {
    error = e.what();
    return false;
  }
}
} // namespace odai::anim
