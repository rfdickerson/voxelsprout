#pragma once
#include "import/material_animation.h"
#include <cstring>
#include <span>
#include <unordered_set>
namespace odai::importer::fnv {
struct MaterialAnimationBlock {
  std::string_view type;
  std::span<const std::uint8_t> data;
};
struct MaterialAnimationReader {
  std::span<const std::uint8_t> bytes;
  template <class T> bool read(T &value) {
    if (bytes.size() < sizeof(T))
      return false;
    std::memcpy(&value, bytes.data(), sizeof(T));
    bytes = bytes.subspan(sizeof(T));
    return true;
  }
};
// Resolve only controllers linked from the material.
// Detached/controller-manager tracks must not start playing merely because
// their blocks occur in the file.
inline std::vector<MaterialAnimationTrack>
readNifMaterialAnimation(std::span<const MaterialAnimationBlock> blocks,
                         std::int32_t property,
                         const std::vector<std::string> &sourceTextures,
                         std::uint32_t &unsupported) {
  std::vector<MaterialAnimationTrack> result;
  auto valid = [&](int ref) {
    return ref >= 0 && std::size_t(ref) < blocks.size();
  };
  if (!valid(property))
    return result;
  const auto &prop = blocks[property];
  MaterialAnimationReader p{prop.data};
  std::uint32_t ignored, count;
  std::int32_t controller;
  if (prop.type == "BSLightingShaderProperty" && !p.read(ignored))
    return result;
  if (!p.read(ignored) || !p.read(count) || count > p.bytes.size() / 4)
    return result;
  p.bytes = p.bytes.subspan(count * 4);
  if (!p.read(controller))
    return result;
  std::unordered_set<int> visited;
  while (valid(controller) && visited.insert(controller).second) {
    const auto &block = blocks[controller];
    MaterialAnimationReader r{block.data};
    std::int32_t next, target, interp;
    std::uint16_t flags;
    MaterialAnimationTrack base;
    if (!r.read(next) || !r.read(flags) || !r.read(base.frequency) ||
        !r.read(base.phase) || !r.read(base.start) || !r.read(base.stop) ||
        !r.read(target) || !r.read(interp)) {
      ++unsupported;
      break;
    }
    controller = next;
    if (!(flags & 8))
      continue;
    if (target != property) {
      ++unsupported;
      continue;
    }
    base.cycle = (flags >> 1) & 3;
    base.backwards = (flags & 16) != 0;
    const bool effect = block.type == "BSEffectShaderPropertyFloatController";
    const bool lighting =
        block.type == "BSLightingShaderPropertyFloatController";
    const bool color = block.type == "BSEffectShaderPropertyColorController" ||
                       block.type == "BSLightingShaderPropertyColorController";
    const bool flip = block.type == "NiFlipController";
    std::uint32_t variable;
    if ((!effect && !lighting && !color && !flip) || !r.read(variable) ||
        !valid(interp)) {
      ++unsupported;
      continue;
    }
    using V = MaterialAnimatedValue;
    if (effect) {
      switch (variable) {
      case 0:
        base.target = V::EmissiveMultiple;
        break;
      case 5:
        base.target = V::Alpha;
        break;
      case 6:
        base.target = V::UOffset;
        break;
      case 7:
        base.target = V::UScale;
        break;
      case 8:
        base.target = V::VOffset;
        break;
      case 9:
        base.target = V::VScale;
        break;
      default:
        ++unsupported;
        continue;
      }
    } else if (lighting) {
      switch (variable) {
      case 8:
        base.target = V::EnvironmentScale;
        break;
      case 9:
        base.target = V::Glossiness;
        break;
      case 10:
        base.target = V::SpecularStrength;
        break;
      case 11:
        base.target = V::EmissiveMultiple;
        break;
      case 12:
        base.target = V::Alpha;
        break;
      case 20:
        base.target = V::UOffset;
        break;
      case 21:
        base.target = V::UScale;
        break;
      case 22:
        base.target = V::VOffset;
        break;
      case 23:
        base.target = V::VScale;
        break;
      default:
        ++unsupported;
        continue;
      }
    } else if (color) {
      if (block.type == "BSEffectShaderPropertyColorController") {
        if (variable != 0) {
          ++unsupported;
          continue;
        }
        base.target = V::EmissiveR;
      } else {
        if (variable > 1) {
          ++unsupported;
          continue;
        }
        base.target = variable ? V::EmissiveR : V::SpecularR;
      }
    } else {
      std::uint32_t sources;
      if ((variable != 0 && variable != 4 && variable != 6) || !r.read(sources) || sources == 0 || sources > 4096) {
        ++unsupported;
        continue;
      }
      bool good = true;
      for (std::uint32_t i = 0; i < sources; ++i) {
        std::int32_t ref;
        if (!r.read(ref) || !valid(ref) ||
            std::size_t(ref) >= sourceTextures.size() ||
            sourceTextures[ref].empty()) {
          good = false;
          break;
        }
        base.texturePaths.push_back(sourceTextures[ref]);
      }
      if (!good) {
        ++unsupported;
        continue;
      }
      base.target = variable == 4 ? V::GlowFrame : variable == 6 ? V::NormalFrame : V::DiffuseFrame;
    }
    const int channels = color ? 3 : 1;
    if (blocks[interp].type !=
        (color ? "NiPoint3Interpolator" : "NiFloatInterpolator")) {
      ++unsupported;
      continue;
    }
    MaterialAnimationReader ir{blocks[interp].data};
    float pose[3]{};
    std::int32_t data = -1;
    bool good = true;
    for (int c = 0; c < channels; ++c)
      good &= ir.read(pose[c]);
    good &= ir.read(data);
    std::vector<MaterialAnimationTrack> tracks(channels, base);
    for (int c = 0; c < channels; ++c)
      tracks[c].target = V(std::uint32_t(base.target) + c);
    if (good && data == -1) {
      for (int c = 0; c < channels; ++c) {
        good &= pose[c] > -3.4e38f; // Gamebryo invalid pose sentinel
        tracks[c].keys.push_back({base.start, pose[c], 0, 0});
      }
    } else if (good && valid(data) &&
               blocks[data].type == (color ? "NiPosData" : "NiFloatData")) {
      MaterialAnimationReader dr{blocks[data].data};
      std::uint32_t n = 0, type = 1;
      good = dr.read(n) && n > 0 && n <= 65536 && dr.read(type) &&
             (type == 1 || type == 2 || type == 3 || type == 5);
      for (auto &t : tracks)
        t.interpolation = type;
      for (std::uint32_t k = 0; good && k < n; ++k) {
        float time = 0;
        good = dr.read(time);
        MaterialAnimationKey keys[3];
        for (int c = 0; c < channels; ++c) {
          keys[c].time = time;
          good &= dr.read(keys[c].value);
        }
        if (type == 2) {
          for (int c = 0; c < channels; ++c)
            good &= dr.read(keys[c].forward);
          for (int c = 0; c < channels; ++c)
            good &= dr.read(keys[c].backward);
        }
        if (type == 3) {
          float tension = 0, bias = 0, continuity = 0;
          good &= dr.read(tension) && dr.read(bias) && dr.read(continuity);
          for (int c = 0; c < channels; ++c) {
            keys[c].tension = tension;
            keys[c].bias = bias;
            keys[c].continuity = continuity;
          }
        }
        for (int c = 0; c < channels; ++c)
          tracks[c].keys.push_back(keys[c]);
      }
    } else
      good = false;
    for (const auto &t : tracks)
      good &= validMaterialAnimationTrack(t);
    if (good)
      result.insert(result.end(), tracks.begin(), tracks.end());
    else
      ++unsupported;
  }
  if (controller != -1)
    ++unsupported;
  return result;
}
} // namespace odai::importer::fnv
