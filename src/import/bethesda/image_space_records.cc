#include "import/bethesda/image_space_records.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace odai::importer::bethesda {
namespace {
std::uint32_t u32(const std::uint8_t *p) {
  return std::uint32_t(p[0]) | (std::uint32_t(p[1]) << 8) |
         (std::uint32_t(p[2]) << 16) | (std::uint32_t(p[3]) << 24);
}
float f32(const std::uint8_t *p) {
  auto v = u32(p);
  float f;
  std::memcpy(&f, &v, 4);
  return f;
}
std::string string(const EsmSubrecordView &s) {
  const auto *end =
      static_cast<const std::uint8_t *>(std::memchr(s.data, 0, s.size));
  return {reinterpret_cast<const char *>(s.data),
          end ? std::size_t(end - s.data) : s.size};
}
bool floats(const EsmSubrecordView &s, float *dest, std::size_t count) {
  if (s.size != count * 4)
    return false;
  for (std::size_t i = 0; i < count; ++i) {
    dest[i] = f32(s.data + i * 4);
    if (!std::isfinite(dest[i]))
      return false;
  }
  return true;
}
bool curve(const EsmSubrecordView &s, std::vector<ImageSpaceKey> &out) {
  if (s.size % 8 || s.size / 8 > 65536)
    return false;
  out.clear();
  for (std::size_t i = 0; i < s.size; i += 8) {
    ImageSpaceKey k{f32(s.data + i), f32(s.data + i + 4)};
    if (!std::isfinite(k.time) || !std::isfinite(k.value) || k.time < 0 ||
        (!out.empty() && k.time < out.back().time))
      return false;
    out.push_back(k);
  }
  return true;
}
bool colors(const EsmSubrecordView &s, std::vector<ImageSpaceColorKey> &out) {
  if (s.size % 20 || s.size / 20 > 65536)
    return false;
  out.clear();
  for (std::size_t i = 0; i < s.size; i += 20) {
    ImageSpaceColorKey k;
    k.time = f32(s.data + i);
    if (!std::isfinite(k.time) || k.time < 0 ||
        (!out.empty() && k.time < out.back().time))
      return false;
    for (int j = 0; j < 4; ++j) {
      k.rgba[j] = f32(s.data + i + 4 + j * 4);
      if (!std::isfinite(k.rgba[j]))
        return false;
    }
    out.push_back(k);
  }
  return true;
}
} // namespace

bool parseImageSpace(const EsmRecordView &record, ImageSpaceRecord &out,
                     std::string &error) {
  out = {};
  error.clear();
  out.formId = record.formId;
  bool data = false;
  for (const auto &s : record.subrecords) {
    bool ok = true;
    if (s.type == "EDID")
      out.editorId = string(s);
    else if (s.type == "HNAM") {
      ok = floats(s, out.settings.hdr.data(), 9);
      data = true;
    } else if (s.type == "CNAM")
      ok = floats(s, out.settings.cinematic.data(), 3);
    else if (s.type == "TNAM")
      ok = floats(s, out.settings.tint.data(), 4);
    else if (s.type == "DNAM") {
      ok = s.size == 12 || s.size == 16;
      if (ok) {
        EsmSubrecordView prefix{s.type, s.data, 12};
        ok = floats(prefix, out.settings.dof.data(), 3);
        if (s.size == 16)
          out.settings.dofFlags =
              std::uint16_t(s.data[14]) | (std::uint16_t(s.data[15]) << 8);
      }
    } else if (s.type == "ENAM") {
      std::array<float, 14> values{};
      ok = floats(s, values.data(), 14);
      data = true;
      if (ok) {
        std::copy_n(values.begin(), 5, out.settings.hdr.begin());
        out.settings.hdr[6] = values[5];
        out.settings.hdr[7] = values[6];
        std::copy_n(values.begin() + 7, 3, out.settings.cinematic.begin());
        std::copy_n(values.begin() + 10, 4, out.settings.tint.begin());
      }
    }
    if (!ok) {
      error = "invalid IMGS " + s.type;
      return false;
    }
  }
  if (!data)
    error = "IMGS missing HDR data";
  return data;
}

bool parseImageSpaceModifier(const EsmRecordView &record,
                             ImageSpaceModifierRecord &out,
                             std::string &error) {
  out = {};
  out.formId = record.formId;
  error.clear();
  const EsmSubrecordView *header = nullptr;
  constexpr std::array<const char *, 11> effectNames{
      "BNAM", "VNAM", "RNAM", "SNAM", "UNAM", "NAM1",
      "NAM2", "WNAM", "XNAM", "YNAM", "NAM4"};
  for (const auto &s : record.subrecords) {
    bool ok = true;
    if (s.type == "EDID")
      out.editorId = string(s);
    else if (s.type == "DNAM") {
      header = &s;
      ok = s.size == 244;
      if (ok) {
        out.animatable = u32(s.data) != 0;
        out.duration = f32(s.data + 4);
        out.radialUseTarget = u32(s.data + 200) != 0;
        out.radialCenter = {f32(s.data + 204), f32(s.data + 208)};
        out.dofUseTarget = s.data[224] != 0;
        out.dofFlags = s.data[225];
        ok = std::isfinite(out.duration) && out.duration >= 0 &&
             std::isfinite(out.radialCenter[0]) &&
             std::isfinite(out.radialCenter[1]);
      }
    } else if (s.type.size() == 4 && s.type.substr(1) == "IAD") {
      const auto code = static_cast<unsigned char>(s.type[0]);
      if (code < 21)
        ok = curve(s, out.multiply[code]);
      else if (code >= 0x40 && code < 0x40 + 21)
        ok = curve(s, out.add[code - 0x40]);
    } else if (s.type == "TNAM")
      ok = colors(s, out.tint);
    else if (s.type == "NAM3")
      ok = colors(s, out.fade);
    else
      for (std::size_t i = 0; i < effectNames.size(); ++i)
        if (s.type == effectNames[i])
          ok = curve(s, out.effects[i]);
    if (!ok) {
      error = "invalid IMAD curve/header";
      return false;
    }
  }
  if (!header) {
    error = "IMAD missing DNAM";
    return false;
  }
  for (std::size_t i = 0; i < 21; ++i) {
    if (u32(header->data + 8 + i * 8) != out.multiply[i].size() ||
        u32(header->data + 12 + i * 8) != out.add[i].size()) {
      error = "IMAD curve count mismatch";
      return false;
    }
  }
  const std::array<std::size_t, 11> offsets{180, 184, 188, 192, 196, 228,
                                            232, 212, 216, 220, 240};
  for (std::size_t i = 0; i < 11; ++i)
    if (u32(header->data + offsets[i]) != out.effects[i].size()) {
      error = "IMAD effect count mismatch";
      return false;
    }
  if (u32(header->data + 176) != out.tint.size() ||
      u32(header->data + 236) != out.fade.size()) {
    error = "IMAD color count mismatch";
    return false;
  }
  return true;
}

bool buildImageSpaceTables(const FalloutLoadOrder &order, ImageSpaceTables &out,
                           std::string &error) {
  out = {};
  error.clear();
  for (std::size_t i = 0; i < order.size(); ++i) {
    EsmReader reader;
    if (!reader.open(order.entries()[i].path)) {
      error = reader.lastError();
      return false;
    }
    EsmReader::Visitor visitor;
    visitor.onGroupEnter = [](const EsmGroupView &g) {
      return g.groupType != 0 || g.rawLabel == "IMGS" || g.rawLabel == "IMAD" ||
             g.rawLabel == "WTHR" || g.rawLabel == "CELL" ||
             g.rawLabel == "WRLD";
    };
    visitor.onRecordHeader = [](const EsmRecordHeaderView &h) {
      return h.type == "IMGS" || h.type == "IMAD" || h.type == "WTHR" ||
             h.type == "CELL";
    };
    visitor.onRecord = [&](const EsmRecordView &r) {
      const auto id = order.remapFormId(i, r.formId);
      std::string why;
      const bool deleted = (r.flags & 0x20) != 0;
      if (r.type == "IMGS") {
        out.spaces.erase(id);
        ImageSpaceRecord v;
        if (!deleted && parseImageSpace(r, v, why)) {
          v.formId = id;
          out.spaces[id] = std::move(v);
        }
      } else if (r.type == "IMAD") {
        out.modifiers.erase(id);
        ImageSpaceModifierRecord v;
        if (!deleted && parseImageSpaceModifier(r, v, why)) {
          v.formId = id;
          out.modifiers[id] = std::move(v);
        }
      } else if (r.type == "WTHR") {
        out.weatherSpaces.erase(id);
        if (!deleted)
          for (const auto &s : r.subrecords)
            if (s.type == "IMSP") {
              if (s.size != 16) {
                why = "invalid IMSP";
                break;
              }
              auto &refs = out.weatherSpaces[id];
              for (int j = 0; j < 4; ++j)
                refs[j] = u32(s.data + j * 4) == 0
                              ? 0
                              : order.remapFormId(i, u32(s.data + j * 4));
            }
      } else {
        out.cells.erase(id);
        if (!deleted) {
          ImageSpaceTables::CellBinding v;
          for (const auto &s : r.subrecords) {
            if (s.type == "EDID")
              v.editorId = string(s);
            if (s.type == "XCIM") {
              if (s.size != 4) {
                why = "invalid XCIM";
                break;
              }
              v.imageSpace =
                  u32(s.data) == 0 ? 0 : order.remapFormId(i, u32(s.data));
            }
          }
          if (why.empty())
            out.cells[id] = std::move(v);
        }
      }
      if (!why.empty())
        out.diagnostics.push_back(std::to_string(id) + ": " + why);
    };
    if (!reader.walk(visitor)) {
      error = reader.lastError();
      return false;
    }
  }
  for (const auto &[id, cell] : out.cells) {
    (void)id;
    if (!cell.editorId.empty())
      out.cellImageSpaceByEditorId[cell.editorId] = cell.imageSpace;
  }
  return true;
}

float imageSpaceDofRadius(const ImageSpaceSettings &settings) {
  constexpr std::array<std::uint16_t, 8> noSky{16576, 16736, 16816, 16880,
                                               16920, 16952, 16984, 17016};
  auto it = std::find(noSky.begin(), noSky.end(), settings.dofFlags);
  if (it == noSky.end() || !std::isfinite(settings.dof[0]))
    return 0;
  return static_cast<float>(it - noSky.begin()) *
         std::clamp(settings.dof[0], 0.0f, 1.0f);
}

float sampleImageSpaceCurve(const std::vector<ImageSpaceKey> &keys, float time,
                            float fallback) {
  if (keys.empty() || !std::isfinite(time))
    return fallback;
  auto upper =
      std::upper_bound(keys.begin(), keys.end(), time,
                       [](float t, const auto &k) { return t < k.time; });
  if (upper == keys.begin())
    return upper->value;
  if (upper == keys.end())
    return keys.back().value;
  const auto &a = *(upper - 1);
  const auto &b = *upper;
  return std::lerp(a.value, b.value, (time - a.time) / (b.time - a.time));
}
ImageSpaceSettings blendImageSpaces(const ImageSpaceSettings &a,
                                    const ImageSpaceSettings &b, float t) {
  if (!std::isfinite(t))
    t = 0;
  t = std::clamp(t, 0.0f, 1.0f);
  ImageSpaceSettings r;
  for (std::size_t i = 0; i < 9; ++i)
    r.hdr[i] = std::lerp(a.hdr[i], b.hdr[i], t);
  for (std::size_t i = 0; i < 3; ++i) {
    r.cinematic[i] = std::lerp(a.cinematic[i], b.cinematic[i], t);
    r.dof[i] = std::lerp(a.dof[i], b.dof[i], t);
  }
  for (std::size_t i = 0; i < 4; ++i)
    r.tint[i] = std::lerp(a.tint[i], b.tint[i], t);
  for (std::size_t i = 0; i < 4; ++i)
    r.fade[i] = std::lerp(a.fade[i], b.fade[i], t);
  r.dofFlags = t < 0.5f ? a.dofFlags : b.dofFlags;
  return r;
}
ImageSpaceSettings sampleWeatherImageSpace(const ImageSpaceTables &tables,
                                           std::uint32_t weather, float hour,
                                           float sunrise, float sunset,
                                           bool &found) {
  found = false;
  auto it = tables.weatherSpaces.find(weather);
  if (it == tables.weatherSpaces.end() || !std::isfinite(hour))
    return {};
  if (!std::isfinite(sunrise))
    sunrise = 6;
  if (!std::isfinite(sunset))
    sunset = 18;
  sunrise = std::clamp(sunrise, 0.01f, 11.99f);
  sunset = std::clamp(sunset, 12.01f, 23.99f);
  const std::array<float, 8> times{0,
                                   sunrise,
                                   (sunrise + 12) / 2,
                                   12,
                                   (12 + sunset) / 2,
                                   sunset,
                                   (sunset + 24) / 2,
                                   24};
  const std::array<int, 8> slots{3, 0, 1, 1, 1, 2, 3, 3};
  hour = std::fmod(hour, 24);
  if (hour < 0)
    hour += 24;
  std::size_t n = 1;
  while (n < 7 && hour > times[n])
    ++n;
  auto a = tables.spaces.find(it->second[slots[n - 1]]),
       b = tables.spaces.find(it->second[slots[n]]);
  if (a == tables.spaces.end() && b == tables.spaces.end())
    return {};
  found = true;
  if (a == tables.spaces.end())
    return b->second.settings;
  if (b == tables.spaces.end())
    return a->second.settings;
  return blendImageSpaces(a->second.settings, b->second.settings,
                          (hour - times[n - 1]) / (times[n] - times[n - 1]));
}
ImageSpaceSettings applyImageSpaceModifier(const ImageSpaceSettings &base,
                                           const ImageSpaceModifierRecord &m,
                                           float elapsed, float strength) {
  if (!std::isfinite(elapsed) || !std::isfinite(strength) || elapsed < 0 ||
      (m.animatable && elapsed > m.duration))
    return base;
  auto out = base;
  const float time = m.animatable ? elapsed : 0;
  const auto apply = [&](float v, std::size_t i) {
    return v * sampleImageSpaceCurve(m.multiply[i], time, 1) +
           sampleImageSpaceCurve(m.add[i], time, 0);
  };
  for (std::size_t i = 0; i < 4; ++i)
    out.hdr[i] = apply(base.hdr[i], i);
  out.hdr[6] = apply(base.hdr[6], 6);
  out.hdr[7] = apply(base.hdr[7], 7);
  for (std::size_t i = 0; i < 3; ++i)
    out.cinematic[i] = apply(base.cinematic[i], 17 + i);
  for (std::size_t i = 0; i < 3 && !m.dofUseTarget; ++i)
    out.dof[i] = sampleImageSpaceCurve(m.effects[7 + i], time, base.dof[i]);
  if (!m.tint.empty()) {
    for (int j = 0; j < 4; ++j) {
      std::vector<ImageSpaceKey> keys;
      keys.reserve(m.tint.size());
      for (const auto &k : m.tint)
        keys.push_back({k.time, k.rgba[j]});
      out.tint[j == 3 ? 0 : j + 1] = sampleImageSpaceCurve(keys, time, 0);
    }
  }
  if (!m.fade.empty()) {
    for (int j = 0; j < 4; ++j) {
      std::vector<ImageSpaceKey> keys;
      keys.reserve(m.fade.size());
      for (const auto &k : m.fade)
        keys.push_back({k.time, k.rgba[j]});
      out.fade[j] = sampleImageSpaceCurve(keys, time, 0);
    }
  }
  return blendImageSpaces(base, out, strength);
}

void ImageSpaceCrossFade::clear() {
  from_.clear(); target_ = 0; elapsed_ = duration_ = targetElapsed_ = 0;
}
void ImageSpaceCrossFade::apply(std::uint32_t modifier, float duration) {
  const float weight = duration_ > 0 ? std::clamp(elapsed_ / duration_, 0.f, 1.f) : 1.f;
  for (auto& c : from_) c.weight *= 1 - weight;
  std::erase_if(from_, [](const auto& c) { return c.weight <= 1e-6f; });
  if (target_ && weight > 0) from_.push_back({target_, targetElapsed_, weight});
  target_ = modifier; elapsed_ = targetElapsed_ = 0;
  duration_ = std::isfinite(duration) ? std::max(duration, 0.f) : 0;
  if (duration_ == 0) from_.clear();
}
ImageSpaceSettings ImageSpaceCrossFade::sample(const ImageSpaceSettings& base,
    const ImageSpaceTables& tables, float deltaSeconds) {
  const float delta = std::isfinite(deltaSeconds) ? std::max(deltaSeconds, 0.f) : 0;
  elapsed_ += delta; targetElapsed_ += delta;
  for (auto& c : from_) c.elapsed += delta;
  const float t = duration_ > 0 ? std::clamp(elapsed_ / duration_, 0.f, 1.f) : 1.f;
  const auto value = [&](std::uint32_t id, float elapsed) {
    const auto found = tables.modifiers.find(id);
    return found == tables.modifiers.end() ? base :
        applyImageSpaceModifier(base, found->second, elapsed, 1);
  };
  float total = 0;
  for (const auto& c : from_) total += c.weight;
  float accumulated = std::max(0.f, 1 - total);
  ImageSpaceSettings previous = base;
  for (const auto& c : from_) {
    const float next = accumulated + c.weight;
    previous = blendImageSpaces(previous, value(c.id, c.elapsed), c.weight / std::max(next, 1e-6f));
    accumulated = next;
  }
  const auto result = blendImageSpaces(previous, value(target_, targetElapsed_), t);
  if (t >= 1) from_.clear();
  return result;
}
} // namespace odai::importer::bethesda
