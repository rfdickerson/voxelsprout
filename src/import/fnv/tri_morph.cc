#include "import/fnv/tri_morph.h"
#include <bit>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_set>

namespace odai::importer::fnv {
namespace {
struct Reader {
  std::span<const std::uint8_t> bytes;
  std::size_t at = 0;
  void require(std::uint64_t n) const {
    if (n > bytes.size() - at)
      throw std::runtime_error("truncated TRI payload");
  }
  std::uint32_t u32() {
    require(4);
    auto p = bytes.data() + at;
    at += 4;
    return std::uint32_t(p[0]) | (std::uint32_t(p[1]) << 8) |
           (std::uint32_t(p[2]) << 16) | (std::uint32_t(p[3]) << 24);
  }
  std::int16_t i16() {
    require(2);
    auto v = std::uint16_t(bytes[at]) | (std::uint16_t(bytes[at + 1]) << 8);
    at += 2;
    return std::bit_cast<std::int16_t>(std::uint16_t(v));
  }
  float f32() {
    float v = std::bit_cast<float>(u32());
    if (!std::isfinite(v))
      throw std::runtime_error("non-finite TRI value");
    return v;
  }
  std::string name() {
    auto n = u32();
    if (!n || n > 4096)
      throw std::runtime_error("invalid TRI name length");
    require(n);
    auto p = bytes.subspan(at, n);
    at += n;
    if (p.back() != 0)
      throw std::runtime_error("unterminated TRI name");
    std::string s(reinterpret_cast<const char *>(p.data()), n - 1);
    if (s.empty() || s.find('\0') != std::string::npos)
      throw std::runtime_error("invalid TRI name");
    return s;
  }
};
template <std::size_t N>
void faces(Reader &r, std::vector<std::array<std::uint32_t, N>> &out,
           std::uint32_t count, std::uint32_t vertices) {
  r.require(std::uint64_t(count) * N * 4);
  out.resize(count);
  for (auto &face : out)
    for (auto &index : face) {
      index = r.u32();
      if (index >= vertices)
        throw std::runtime_error("TRI face index out of bounds");
    }
}
bool finite(const TriPosition &p) {
  return std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]);
}
} // namespace
TriReadStatus readTriMorphs(std::span<const std::uint8_t> bytes,
                            TriMorphSet &output, std::string &error) {
  output = {};
  error.clear();
  if (bytes.size() < 8) {
    error = "truncated TRI signature";
    return TriReadStatus::MalformedData;
  }
  if (std::string(reinterpret_cast<const char *>(bytes.data()), 8) !=
      "FRTRI003") {
    error = "unsupported TRI signature/version (requires FRTRI003)";
    return TriReadStatus::UnsupportedFormat;
  }
  if (bytes.size() > 256u * 1024u * 1024u) {
    error = "TRI file exceeds allocation budget";
    return TriReadStatus::ResourceLimit;
  }
  try {
    Reader r{bytes, 8};
    TriMorphSet result;
    const auto nv = r.u32(), nt = r.u32(), nq = r.u32();
    result.reserved[0] = r.u32();
    result.reserved[1] = r.u32();
    const auto nu = r.u32(), hasUv = r.u32(), nm = r.u32(), nmod = r.u32(),
               nmodv = r.u32();
    for (int i = 2; i < 6; ++i)
      result.reserved[i] = r.u32();
    if (nv > 1000000 || nt > 4000000 || nq > 1000000 || nu > 1000000 ||
        nm > 4096 || nmod > 4096 || nmodv > 4000000) {
      error = "TRI counts exceed allocation limits";
      return TriReadStatus::ResourceLimit;
    }
    if (hasUv > 1)
      throw std::runtime_error("invalid TRI UV flag");
    result.hasUvFaces = hasUv != 0;
    // Check all fixed-size payload before allocating count-sized vectors.
    r.require(std::uint64_t(nv + nmodv) * 12 + std::uint64_t(nt) * 12 +
              std::uint64_t(nq) * 16 + std::uint64_t(nu) * 8 +
              hasUv * (std::uint64_t(nt) * 12 + std::uint64_t(nq) * 16) +
              std::uint64_t(nm) * (9 + std::uint64_t(nv) * 6) +
              std::uint64_t(nmod) * 9);
    result.positions.resize(nv);
    for (auto &p : result.positions)
      for (auto &v : p)
        v = r.f32();
    std::vector<TriPosition> replacements(nmodv);
    for (auto &p : replacements)
      for (auto &v : p)
        v = r.f32();
    faces(r, result.triangles, nt, nv);
    faces(r, result.quads, nq, nv);
    result.uvs.resize(nu);
    for (auto &uv : result.uvs)
      for (auto &v : uv)
        v = r.f32();
    if (hasUv) {
      faces(r, result.uvTriangles, nt, nu);
      faces(r, result.uvQuads, nq, nu);
    }
    std::unordered_set<std::string> names;
    result.morphs.reserve(nm);
    for (std::uint32_t i = 0; i < nm; ++i) {
      TriMorph m;
      m.name = r.name();
      if (!names.insert(m.name).second)
        result.ambiguousMorphNames = true;
      m.scale = r.f32();
      if (m.scale < 0)
        throw std::runtime_error("negative TRI morph scale");
      r.require(std::uint64_t(nv) * 6);
      m.deltas.resize(nv);
      for (auto &d : m.deltas)
        for (auto &v : d)
          v = r.i16();
      result.morphs.push_back(std::move(m));
    }
    names.clear();
    std::size_t offset = 0;
    for (std::uint32_t i = 0; i < nmod; ++i) {
      TriModifier m;
      m.name = r.name();
      if (!names.insert(m.name).second)
        result.ambiguousModifierNames = true;
      auto n = r.u32();
      if (n > replacements.size() - offset)
        throw std::runtime_error("TRI modifier vertex count mismatch");
      r.require(std::uint64_t(n) * 4);
      m.vertices.resize(n);
      std::unordered_set<std::uint32_t> seen;
      for (auto &v : m.vertices) {
        v = r.u32();
        if (v >= nv || !seen.insert(v).second)
          throw std::runtime_error("invalid TRI modifier vertex");
      }
      m.positions.assign(replacements.begin() + offset,
                         replacements.begin() + offset + n);
      offset += n;
      result.modifiers.push_back(std::move(m));
    }
    if (offset != replacements.size())
      throw std::runtime_error("unclaimed TRI modifier vertices");
    if (r.at != bytes.size())
      throw std::runtime_error("unexpected trailing TRI data");
    output = std::move(result);
    return TriReadStatus::Ok;
  } catch (const std::exception &e) {
    error = e.what();
    return TriReadStatus::MalformedData;
  }
}
bool evaluateTriMorphs(const TriMorphSet &set,
                       std::span<const TriPosition> neutral,
                       std::span<const TriMorphWeight> weights,
                       std::vector<TriPosition> &output, std::string &error) {
  error.clear();
  auto fail = [&](const char *message) {
    output.clear();
    error = message;
    return false;
  };
  if (neutral.size() != set.positions.size())
    return fail("TRI neutral vertex count mismatch");
  for (const auto &p : neutral)
    if (!finite(p))
      return fail("non-finite neutral vertex");
  std::unordered_set<std::uint32_t> seen;
  for (auto w : weights) {
    if (w.morph >= set.morphs.size() || !std::isfinite(w.weight) ||
        w.weight < 0 || w.weight > 1 || !seen.insert(w.morph).second)
      return fail("invalid or duplicate TRI morph weight");
    const auto &m = set.morphs[w.morph];
    if (m.deltas.size() != neutral.size() || !std::isfinite(m.scale) ||
        m.scale < 0)
      return fail("invalid TRI morph data");
  }
  // Build separately so a caller may safely pass the previous output as
  // neutral.
  std::vector<TriPosition> result(neutral.begin(), neutral.end());
  for (auto w : weights) {
    const auto &m = set.morphs[w.morph];
    for (std::size_t i = 0; i < result.size(); ++i)
      for (int k = 0; k < 3; ++k) {
        const double v =
            double(result[i][k]) + double(m.deltas[i][k]) * m.scale * w.weight;
        if (!std::isfinite(v) ||
            std::abs(v) > std::numeric_limits<float>::max())
          return fail("TRI morph result overflow");
        result[i][k] = float(v);
      }
  }
  output = std::move(result);
  return true;
}
} // namespace odai::importer::fnv
