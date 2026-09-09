#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>
namespace odai::importer {
enum class MaterialAnimatedValue : std::uint32_t {
  UOffset,
  UScale,
  VOffset,
  VScale,
  Alpha,
  EmissiveR,
  EmissiveG,
  EmissiveB,
  EmissiveMultiple,
  SpecularR,
  SpecularG,
  SpecularB,
  SpecularStrength,
  Glossiness,
  EnvironmentScale,
  DiffuseFrame,
  NormalFrame,
  GlowFrame
};
struct MaterialAnimationKey {
  float time = 0, value = 0, forward = 0, backward = 0;
  float tension = 0, bias = 0, continuity = 0;
};
struct MaterialAnimationTrack {
  MaterialAnimatedValue target = MaterialAnimatedValue::UOffset;
  std::uint32_t interpolation = 1, cycle = 0;
  float frequency = 1, phase = 0, start = 0, stop = 0;
  bool backwards = false;
  std::vector<MaterialAnimationKey> keys;
  std::vector<std::string> texturePaths;
  std::vector<std::uint32_t> textures;
};
inline float sampleMaterialAnimation(const MaterialAnimationTrack &track,
                                     float elapsed) {
  if (track.keys.empty())
    return 0;
  double time = double(elapsed) * track.frequency + track.phase;
  const double duration = double(track.stop) - track.start;
  if (duration > 0 && track.cycle < 2) {
    const double period = duration * (track.cycle == 1 ? 2 : 1);
    double local = std::fmod(time - track.start, period);
    if (local < 0)
      local += period;
    if (local > duration)
      local = period - local;
    time = track.start + local;
  } else
    time = std::clamp(time, double(track.start), double(track.stop));
  if (track.backwards)
    time = track.stop - (time - track.start);
  const auto next =
      std::upper_bound(track.keys.begin(), track.keys.end(), time,
                       [](double t, const auto &key) { return t < key.time; });
  if (next == track.keys.begin())
    return next->value;
  const auto &a = *(next - 1);
  if (next == track.keys.end() || track.interpolation == 5)
    return a.value;
  const auto &b = *next;
  const float u = float((time - a.time) / (b.time - a.time));
  if (track.interpolation == 1)
    return std::lerp(a.value, b.value, u);
  float outgoing = a.forward, incoming = b.backward;
  if (track.interpolation == 3) {
    const auto index = std::size_t((next - 1) - track.keys.begin());
    const auto tangent = [&](std::size_t i, bool forward) {
      const auto &key = track.keys[i];
      const auto &prev = track.keys[i ? i - 1 : i];
      const auto &after = track.keys[std::min(i + 1, track.keys.size() - 1)];
      const float left =
          i ? (key.value - prev.value) / (key.time - prev.time)
            : (after.value - key.value) / (after.time - key.time);
      const float right =
          i + 1 < track.keys.size()
              ? (after.value - key.value) / (after.time - key.time)
              : left;
      const float c = forward ? key.continuity : -key.continuity;
      return .5f * (1 - key.tension) *
             ((1 + c) * (1 + key.bias) * left +
              (1 - c) * (1 - key.bias) * right);
    };
    outgoing = tangent(index, true) * (b.time - a.time);
    incoming = tangent(index + 1, false) * (b.time - a.time);
  }
  const float u2 = u * u, u3 = u2 * u;
  return (2 * u3 - 3 * u2 + 1) * a.value + (u3 - 2 * u2 + u) * outgoing +
         (-2 * u3 + 3 * u2) * b.value + (u3 - u2) * incoming;
}
inline bool validMaterialAnimationTrack(const MaterialAnimationTrack &t) {
  if (std::uint32_t(t.target) >
          std::uint32_t(MaterialAnimatedValue::GlowFrame) ||
      (t.interpolation != 1 && t.interpolation != 2 && t.interpolation != 3 &&
       t.interpolation != 5) ||
      t.cycle > 2 || !std::isfinite(t.frequency) || !std::isfinite(t.phase) ||
      !std::isfinite(t.start) || !std::isfinite(t.stop) || t.stop < t.start ||
      t.keys.empty())
    return false;
  float previous = -INFINITY;
  for (const auto &k : t.keys) {
    if (!std::isfinite(k.time) || !std::isfinite(k.value) ||
        !std::isfinite(k.forward) || !std::isfinite(k.backward) ||
        !std::isfinite(k.tension) || !std::isfinite(k.bias) ||
        !std::isfinite(k.continuity) || k.time <= previous)
      return false;
    previous = k.time;
  }
  return true;
}
} // namespace odai::importer
