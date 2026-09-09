#include "import/fnv/nif_particles.h"
#include "import/fnv/nif_scene.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <unordered_map>
namespace odai::importer::fnv {
namespace {
struct Reader {
  const std::uint8_t *p;
  size_t n, at = 0;
  template <class T> T get() {
    if (at + sizeof(T) > n)
      throw std::runtime_error("truncated particle data");
    T v;
    std::memcpy(&v, p + at, sizeof(v));
    at += sizeof(v);
    return v;
  }
  float f() {
    auto v = get<float>();
    if (!std::isfinite(v))
      throw std::runtime_error("nonfinite particle data");
    return v;
  }
  void skip(size_t k) {
    if (k > n - at)
      throw std::runtime_error("truncated particle data");
    at += k;
  }
  std::string str() {
    auto k = get<std::uint32_t>();
    if (k > 4096 || k > n - at)
      throw std::runtime_error("invalid particle string");
    std::string s((const char *)p + at, k);
    at += k;
    return s;
  }
};
using Mat = std::array<float, 12>;
using Vec = std::array<float, 3>;
Mat identity() { return {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0}; }
Vec vector(const Mat &m, Vec v) {
  Vec o{};
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c)
      o[r] += m[r * 4 + c] * v[c];
  return o;
}
Vec point(const Mat &m, Vec v) {
  auto o = vector(m, v);
  for (int i = 0; i < 3; ++i)
    o[i] += m[i * 4 + 3];
  return o;
}
Mat mul(const Mat &a, const Mat &b) {
  Mat o{};
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c)
      for (int k = 0; k < 3; ++k)
        o[r * 4 + c] += a[r * 4 + k] * b[k * 4 + c];
    o[r * 4 + 3] = point(a, {b[3], b[7], b[11]})[r];
  }
  return o;
}
float curve(const std::vector<MistKey> &k, float t) {
  if (k.empty())
    return 0;
  if (t <= k.front().time)
    return k.front().value;
  if (t >= k.back().time)
    return k.back().value;
  auto b = std::upper_bound(k.begin(), k.end(), t,
                            [](float x, const auto &v) { return x < v.time; });
  auto &a = *(b - 1);
  float d = b->time - a.time, u = (t - a.time) / d;
  return (2 * u * u * u - 3 * u * u + 1) * a.value +
         (u * u * u - 2 * u * u + u) * a.forward +
         (-2 * u * u * u + 3 * u * u) * b->value +
         (u * u * u - u * u) * b->backward;
}
Mat world(const NifMist &m, int id, float t, int depth = 0) {
  if (id < 0)
    return identity();
  if (size_t(id) >= m.nodes.size() || depth > 32)
    return identity();
  auto &n = m.nodes[id];
  Mat a = n.transform;
  if (!n.angles[0].empty()) {
    float q = t * n.frequency + n.phase;
    if (n.period > 0)
      q = std::fmod(std::fmod(q, n.period) + n.period, n.period);
    Mat rot = identity();
    for (int axis = 0; axis < 3; ++axis) {
      float v = curve(n.angles[axis], q), c = std::cos(v), s = std::sin(v);
      Mat r = identity();
      int j = (axis + 1) % 3, k = (axis + 2) % 3;
      r[j * 4 + j] = r[k * 4 + k] = c;
      r[j * 4 + k] = -s;
      r[k * 4 + j] = s;
      rot = mul(rot, r);
    }
    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 3; ++c)
        a[r * 4 + c] = rot[r * 4 + c];
  }
  return mul(world(m, n.parent, t, depth + 1), a);
}
float random(std::uint32_t &x) {
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  return float(x & 0xffffff) / 16777216.0f;
}
float blend(float a, float b, float x, float lo, float hi) {
  return std::lerp(a, b,
                   hi > lo ? std::clamp((x - lo) / (hi - lo), 0.f, 1.f)
                           : float(x >= hi));
}
} // namespace
bool parseNifMist(const std::vector<std::uint8_t> &bytes, NifMist &out,
                  std::string &error) {
  out = {};
  error.clear();
  NifBlockSummary s;
  if (!parseNifBlockSummary(bytes, s, error))
    return false;
  try {
    if (s.inlineNames || s.version != 0x14020007)
      throw std::runtime_error("mist requires Skyrim block layout");
    auto block = [&](int i, const char *type = nullptr) {
      if (i < 0 || size_t(i) >= s.blockStarts.size() ||
          (type && s.blockTypeNames[i] != type))
        throw std::runtime_error("missing particle block reference");
      return Reader{bytes.data() + s.blockStarts[i], s.blockSizes[i]};
    };
    int system = -1, box = -1, color = -1, scale = -1, gravity = -1,
        rotation = -1, ctl = -1;
    std::vector<int> drags;
    for (size_t i = 0; i < s.blockTypeNames.size(); ++i) {
      auto &t = s.blockTypeNames[i];
      if (t == "NiPSysMeshEmitter" || t == "BSPSysSubTexModifier" ||
          t == "NiPSysBombModifier" || t == "NiPSysColorModifier" ||
          t == "NiPSysGrowFadeModifier" ||
          t == "NiPSysTurbulenceFieldModifier" || t == "NiPSysColliderManager")
        throw std::runtime_error("unsupported particle modifier: " + std::string(t));

      if (t == "NiParticleSystem") {
        if (system != -1)
          throw std::runtime_error("multiple particle systems unsupported");
        system = int(i);
      }
      if (t == "NiPSysBoxEmitter" || t == "NiPSysCylinderEmitter" || t == "NiPSysSphereEmitter") {
        if (box >= 0) throw std::runtime_error("multiple particle emitters unsupported");
        box = int(i);
        out.emitterShape = t == "NiPSysCylinderEmitter" ? NifEmitterShape::Cylinder :
            t == "NiPSysSphereEmitter" ? NifEmitterShape::Sphere : NifEmitterShape::Box;
      }
      if (t == "BSPSysSimpleColorModifier")
        color = int(i);
      if (t == "BSPSysScaleModifier")
        scale = int(i);
      if (t == "NiPSysGravityModifier")
        gravity = int(i);
      if (t == "NiPSysRotationModifier")
        rotation = int(i);
      if (t == "NiPSysEmitterCtlr")
        ctl = int(i);
      if (t == "NiPSysDragModifier")
        drags.push_back(int(i));
    }
    if (system < 0 || box < 0 || ctl < 0)
      throw std::runtime_error("unsupported mist modifier set");
    out.nodes.resize(s.blockStarts.size());
    for (size_t i = 0; i < s.blockStarts.size(); ++i) {
      auto &t = s.blockTypeNames[i];
      if (t != "NiNode" && t != "BSFadeNode" && t != "NiParticleSystem")
        continue;
      auto r = block(i);
      r.skip(4);
      auto extras = r.get<std::uint32_t>();
      if (extras > 64)
        throw std::runtime_error("too many node extras");
      r.skip(extras * 4);
      int controller = r.get<int>();
      r.skip(4);
      Vec pos{};
      for (auto &v : pos)
        v = r.f();
      std::array<float, 9> rot{};
      for (auto &v : rot)
        v = r.f();
      float sc = r.f();
      auto &node = out.nodes[i];
      for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b)
          node.transform[a * 4 + b] = rot[a * 3 + b] * sc;
        node.transform[a * 4 + 3] = pos[a];
      }
      r.skip(4);
      if (t != "NiParticleSystem") {
        auto count = r.get<std::uint32_t>();
        if (count > 256)
          throw std::runtime_error("too many children");
        for (unsigned j = 0; j < count; ++j) {
          int child = r.get<int>();
          block(child);
          out.nodes[child].parent = int(i);
        }
      }
      if (controller >= 0 &&
          s.blockTypeNames.at(controller) == "NiTransformController") {
        auto c = block(controller);
        c.skip(6);
        node.frequency = c.f();
        node.phase = c.f();
        float begin = c.f();
        node.period = c.f();
        if (begin != 0)
          throw std::runtime_error("nonzero transform start unsupported");
        c.skip(4);
        auto ip = block(c.get<int>(), "NiTransformInterpolator");
        ip.skip(32);
        auto data = block(ip.get<int>(), "NiTransformData");
        if (data.get<unsigned>() != 1 || data.get<unsigned>() != 4)
          throw std::runtime_error("mist requires XYZ helper rotation");
        for (auto &keys : node.angles) {
          auto count = data.get<unsigned>();
          auto kind = data.get<unsigned>();
          if (count > 4096 || kind != 2)
            throw std::runtime_error("unsupported helper curve");
          for (unsigned k = 0; k < count; ++k) {
            MistKey key{data.f(), data.f(), data.f(), data.f()};
            if (!keys.empty() && key.time <= keys.back().time)
              throw std::runtime_error("unordered helper keys");
            keys.push_back(key);
          }
        }
      }
    }
    // Parent assignment survives node transform parsing. Reject cycles rather
    // than silently approximating them.
    for (size_t i = 0; i < out.nodes.size(); ++i) {
      int at = int(i);
      for (size_t n = 0; at >= 0; ++n) {
        if (n > out.nodes.size())
          throw std::runtime_error("particle node cycle");
        at = out.nodes.at(at).parent;
      }
    }
    auto modifier = [&](int i) {
      auto r = block(i);
      r.skip(8);
      if (r.get<int>() != system || r.get<std::uint8_t>() != 1)
        throw std::runtime_error("inactive/unrelated mist modifier");
      return r;
    };
    auto e = modifier(box);
    out.speed = e.f();
    out.speedVariation = e.f();
    out.declination = e.f();
    out.declinationVariation = e.f();
    out.planarAngle = e.f();
    out.planarVariation = e.f();
    std::array<float, 4> initialColor;
    for (auto& channel : initialColor) channel = e.f();
    out.radius = e.f();
    out.radiusVariation = e.f();
    out.life = e.f();
    out.lifeVariation = e.f();
    out.emitterNode = e.get<int>();
    block(out.emitterNode, "NiNode");
    if (out.emitterShape == NifEmitterShape::Box) {
      for (auto &v : out.box) v = e.f();
    } else {
      out.volumeRadius = e.f();
      if (out.emitterShape == NifEmitterShape::Cylinder) out.volumeHeight = e.f();
      if (out.volumeRadius < 0 || out.volumeHeight < 0)
        throw std::runtime_error("negative emitter dimensions");
    }
    // Modifiers are optional. Without one, retain the emitter's authored
    // initial color/size and ballistic velocity instead of rejecting the NIF.
    out.colorTimes = {0, 1, 0, 1, 0, 1};
    for (int stage = 0; stage < 3; ++stage)
      std::copy(initialColor.begin(), initialColor.end(), out.colors.begin() + stage * 4);
    if (color >= 0) {
      auto c = modifier(color);
      for (auto &v : out.colorTimes) v = c.f();
      for (auto &v : out.colors) v = c.f();
    }
    out.scales = {1, 1};
    if (scale >= 0) {
      auto z = modifier(scale);
      auto n = z.get<unsigned>();
      if (n < 2 || n > 4096) throw std::runtime_error("invalid scale curve");
      out.scales.clear();
      for (unsigned i = 0; i < n; ++i) out.scales.push_back(z.f());
    }
    out.gravityNode = out.emitterNode;
    if (gravity >= 0) {
      auto g = modifier(gravity);
      out.gravityNode = g.get<int>();
      block(out.gravityNode, "NiNode");
      for (auto &v : out.gravityAxis) v = g.f();
      if (g.f() != 0) throw std::runtime_error("gravity decay unsupported");
      out.gravity = g.f();
      if (g.get<unsigned>() != 0) throw std::runtime_error("nonplanar gravity unsupported");
    }
    if (!drags.empty() && drags.size() != 3)
      throw std::runtime_error("anisotropic drag unsupported");
    for (size_t i = 0; i < drags.size(); ++i) {
      auto d = modifier(drags[i]);
      d.skip(4);
      for (int a = 0; a < 3; ++a)
        if (d.f() != float(size_t(a) == i))
          throw std::runtime_error("rotated drag unsupported");
      float strength = d.f();
      if (i && strength != out.drag)
        throw std::runtime_error("anisotropic drag unsupported");
      out.drag = strength;
    }
    if (rotation >= 0) {
      auto r = modifier(rotation);
      out.rotationSpeed = r.f();
      out.rotationVariation = r.f();
    }
    auto control = block(ctl);
    control.skip(6);
    out.frequency = control.f();
    out.phase = control.f();
    if (control.f() != 0)
      throw std::runtime_error("emitter start unsupported");
    out.period = control.f();
    if (control.get<int>() != system)
      throw std::runtime_error("wrong emitter target");
    auto rate = block(control.get<int>(), "NiFloatInterpolator");
    out.rate = rate.f();
    if (rate.get<int>() != -1)
      throw std::runtime_error("animated birth rate unsupported");
    // Skyrim NiParticleSystem has no vertex geometry, but retains its bounds,
    // skin pointer and two shader-property references after the AVObject
    // prefix.
    auto ps = block(system);
    ps.skip(4);
    unsigned extra = ps.get<unsigned>();
    ps.skip(extra * 4 + 4 + 4 + 12 + 36 + 4 + 4 + 16 + 4);
    int shader = ps.get<int>(), alpha = ps.get<int>();
    auto ap = block(alpha, "NiAlphaProperty");
    ap.skip(12);
    if (ap.get<std::uint16_t>() != 0x10ed)
      throw std::runtime_error("mist requires source-alpha blending");
    auto mat = block(shader, "BSEffectShaderProperty");
    mat.skip(4);
    extra = mat.get<unsigned>();
    mat.skip(extra * 4 + 4 + 8 + 16);
    out.texture = mat.str();
    mat.skip(4 + 16 + 16 + 4);
    out.softDepth = mat.f();
    if (out.texture.empty())
      throw std::runtime_error("missing mist texture");
    if (out.rate <= 0 || out.rate > 256 || out.life <= 0 || out.life > 60 ||
        out.lifeVariation < 0 || out.lifeVariation >= out.life ||
        out.radius <= 0 || out.drag < 0 || out.declinationVariation < 0 ||
        out.planarVariation < 0 || std::any_of(out.box.begin(), out.box.end(), [](float v) { return v < 0; }))
      throw std::runtime_error("invalid mist emission parameters");
    return true;
  } catch (const std::exception &e) {
    error = e.what();
    out = {};
    return false;
  }
}
std::vector<MistParticle> sampleNifMist(const NifMist &m, float elapsed,
                                        std::uint32_t seed) {
  std::vector<MistParticle> result;
  if (!std::isfinite(elapsed) || elapsed < 0 || elapsed > 100000 ||
      m.rate <= 0 || m.rate > 256 || m.nodes.empty() || m.scales.size() < 2)
    return result;
  elapsed = elapsed * m.frequency + m.phase;
  if (!std::isfinite(elapsed) || elapsed < 0 || elapsed > 100000)
    return result;
  // Stable birth IDs make this independent of frame rate, cell iteration order,
  // and camera movement. Respawns receive fresh, deterministic variation.
  int last = int(std::floor(elapsed * m.rate));
  int first = std::max(
      0, int(std::floor((elapsed - m.life - m.lifeVariation) * m.rate)));
  for (int id = first; id <= last && result.size() < 512; ++id) {
    float birth = id / m.rate, age = elapsed - birth;
    std::uint32_t rng = seed ^ (std::uint32_t(id + 1) * 0x9e3779b9u);
    if (!rng)
      rng = 1;
    float life = m.life + (random(rng) * 2 - 1) * m.lifeVariation;
    if (age >= life)
      continue;
    float radius = m.radius + (random(rng) * 2 - 1) * m.radiusVariation;
    float speed = m.speed + (random(rng) * 2 - 1) * m.speedVariation;
    Vec local{};
    if (m.emitterShape == NifEmitterShape::Box) {
      for (int k = 0; k < 3; ++k) local[k] = (random(rng) - .5f) * m.box[k];
    } else {
      const float azimuth = random(rng) * 6.2831853f;
      if (m.emitterShape == NifEmitterShape::Cylinder) {
        const float r = std::sqrt(random(rng)) * m.volumeRadius;
        local = {r * std::cos(azimuth), r * std::sin(azimuth),
                 (random(rng) - .5f) * m.volumeHeight};
      } else {
        const float z = random(rng) * 2 - 1;
        const float r = std::cbrt(random(rng)) * m.volumeRadius;
        const float xy = std::sqrt(std::max(0.f, 1 - z*z));
        local = {r*xy*std::cos(azimuth), r*xy*std::sin(azimuth), r*z};
      }
    }
    Vec direction{0, 0, 1};
    if (m.declination != 0 || m.declinationVariation != 0 || m.planarAngle != 0 || m.planarVariation != 0) {
      const float polar = m.declination + (random(rng)*2-1)*m.declinationVariation;
      const float azimuth = m.planarAngle + (random(rng)*2-1)*m.planarVariation;
      direction = {std::sin(polar)*std::cos(azimuth), std::sin(polar)*std::sin(azimuth), std::cos(polar)};
    }
    for (auto& v : direction) v *= speed;
    auto em = world(m, m.emitterNode, birth);
    Vec pos = point(em, local), velocity = vector(em, direction);
    if (m.gravity == 0 && m.drag == 0) {
      for (int k = 0; k < 3; ++k) pos[k] += velocity[k] * age;
    }
    const int steps = (m.gravity == 0 && m.drag == 0) ? 0 :
        std::max(1, int(std::ceil(age * 30)));
    float dt = steps ? age / steps : 0;
    for (int step = 0; step < steps; ++step) {
      auto force = vector(world(m, m.gravityNode, birth + (step + .5f) * dt),
                          m.gravityAxis);
      for (int k = 0; k < 3; ++k) {
        velocity[k] += (force[k] * m.gravity - velocity[k] * m.drag) * dt;
        pos[k] += velocity[k] * dt;
      }
    }
    float u = age / life;
    MistParticle p;
    p.position = pos;
    float index = u * (m.scales.size() - 1);
    size_t ix = std::min(size_t(index), m.scales.size() - 2);
    p.radius = radius * std::lerp(m.scales[ix], m.scales[ix + 1], index - ix);
    p.angle =
        (m.rotationSpeed + (random(rng) * 2 - 1) * m.rotationVariation) * age;
    for (int k = 0; k < 4; ++k) {
      float v = blend(m.colors[k], m.colors[4 + k], u, m.colorTimes[2],
                      m.colorTimes[3]);
      p.color[k] =
          blend(v, m.colors[8 + k], u, m.colorTimes[4], m.colorTimes[5]);
    }
    p.color[3] *= std::min(
        m.colorTimes[0] > 0 ? std::clamp(u / m.colorTimes[0], 0.f, 1.f) : 1.f,
        m.colorTimes[1] < 1 ? std::clamp((1 - u) / (1 - m.colorTimes[1]), 0.f, 1.f) : 1.f);
    result.push_back(p);
  }
  return result;
}
// The cooked representation is explicitly field-wise and independently bounded.
std::vector<std::uint8_t> encodeNifMist(const NifMist &m) {
  std::vector<std::uint8_t> b;
  auto put = [&]<class T>(const T &v) {
    auto *p = (const std::uint8_t *)&v;
    b.insert(b.end(), p, p + sizeof(T));
  };
  auto str = [&](const std::string &s) {
    put(std::uint32_t(s.size()));
    b.insert(b.end(), s.begin(), s.end());
  };
  str(m.texture);
  put(m.emitterNode);
  put(m.gravityNode);
  for (float v :
       {m.speed, m.speedVariation, m.radius, m.radiusVariation, m.life,
        m.lifeVariation, m.rate, m.period, m.frequency, m.phase, m.gravity,
        m.drag, m.rotationSpeed, m.rotationVariation, m.softDepth})
    put(v);
  for (auto a : {m.box, m.gravityAxis})
    for (float v : a)
      put(v);
  for (float v : m.colorTimes)
    put(v);
  for (float v : m.colors)
    put(v);
  put(std::uint32_t(m.scales.size()));
  for (float v : m.scales)
    put(v);
  put(std::uint32_t(m.nodes.size()));
  for (auto &n : m.nodes) {
    put(n.parent);
    for (float v : n.transform)
      put(v);
    put(n.period);
    put(n.frequency);
    put(n.phase);
    for (auto &ks : n.angles) {
      put(std::uint32_t(ks.size()));
      for (auto &k : ks) {
        put(k.time);
        put(k.value);
        put(k.forward);
        put(k.backward);
      }
    }
  }
  // Optional versioned tail preserves existing cooked box emitters verbatim.
  put(std::uint32_t(0x3153504eu)); // NPS1
  put(std::uint8_t(m.emitterShape));
  for (float v : {m.volumeRadius, m.volumeHeight, m.declination,
                  m.declinationVariation, m.planarAngle, m.planarVariation}) put(v);
  return b;
}
bool decodeNifMist(const std::vector<std::uint8_t> &b, NifMist &m) {
  m = {};
  try {
    Reader r{b.data(), b.size()};
    m.texture = r.str();
    m.emitterNode = r.get<int>();
    m.gravityNode = r.get<int>();
    float *fs[] = {&m.speed,         &m.speedVariation,
                   &m.radius,        &m.radiusVariation,
                   &m.life,          &m.lifeVariation,
                   &m.rate,          &m.period,
                   &m.frequency,     &m.phase,
                   &m.gravity,       &m.drag,
                   &m.rotationSpeed, &m.rotationVariation,
                   &m.softDepth};
    for (auto p : fs)
      *p = r.f();
    for (auto *p : {&m.box, &m.gravityAxis})
      for (auto &v : *p)
        v = r.f();
    for (auto &v : m.colorTimes)
      v = r.f();
    for (auto &v : m.colors)
      v = r.f();
    auto count = r.get<unsigned>();
    if (count < 2 || count > 4096)
      return false;
    while (count--)
      m.scales.push_back(r.f());
    count = r.get<unsigned>();
    if (count > 4096)
      return false;
    m.nodes.resize(count);
    for (auto &n : m.nodes) {
      n.parent = r.get<int>();
      for (auto &v : n.transform)
        v = r.f();
      n.period = r.f();
      n.frequency = r.f();
      n.phase = r.f();
      for (auto &ks : n.angles) {
        auto c = r.get<unsigned>();
        if (c > 4096)
          return false;
        while (c--)
          ks.push_back({r.f(), r.f(), r.f(), r.f()});
      }
    }
    if (m.emitterNode < 0 || m.gravityNode < 0 ||
        size_t(m.emitterNode) >= m.nodes.size() ||
        size_t(m.gravityNode) >= m.nodes.size() || m.rate <= 0 ||
        m.rate > 256 || m.life <= 0 || m.life > 60 || m.lifeVariation < 0 ||
        m.lifeVariation >= m.life)
      return false;
    for (auto &n : m.nodes) {
      if (n.parent < -1 || n.parent >= int(m.nodes.size()))
        return false;
      for (const auto &keys : n.angles)
        for (size_t i = 1; i < keys.size(); ++i)
          if (keys[i].time <= keys[i - 1].time)
            return false;
    }
    for (size_t i = 0; i < m.nodes.size(); ++i) {
      int id = int(i);
      for (size_t depth = 0; id >= 0; ++depth) {
        if (depth > m.nodes.size())
          return false;
        id = m.nodes[id].parent;
      }
    }
    if (m.radius <= 0 || m.radiusVariation < 0 || m.drag < 0 || m.softDepth < 0)
      return false;
    if (r.at != r.n) {
      if (r.get<std::uint32_t>() != 0x3153504eu) return false;
      const auto shape = r.get<std::uint8_t>();
      if (shape > std::uint8_t(NifEmitterShape::Sphere)) return false;
      m.emitterShape = NifEmitterShape(shape);
      m.volumeRadius = r.f(); m.volumeHeight = r.f();
      m.declination = r.f(); m.declinationVariation = r.f();
      m.planarAngle = r.f(); m.planarVariation = r.f();
      if (m.volumeRadius < 0 || m.volumeHeight < 0 ||
          m.declinationVariation < 0 || m.planarVariation < 0) return false;
    }
    return r.at == r.n;
  } catch (...) {
    m = {};
    return false;
  }
}
} // namespace odai::importer::fnv
