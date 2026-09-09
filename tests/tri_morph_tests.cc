#include "import/fnv/asset_source.h"
#include "import/fnv/content_profile.h"
#include "import/fnv/tri_morph.h"
#include <bit>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
using namespace odai::importer::fnv;
using Bytes = std::vector<std::uint8_t>;
void u32(Bytes &b, std::uint32_t v) {
  for (int i = 0; i < 4; ++i)
    b.push_back(v >> (8 * i));
}
void f32(Bytes &b, float v) { u32(b, std::bit_cast<std::uint32_t>(v)); }
void name(Bytes &b, const std::string &s) {
  u32(b, s.size() + 1);
  b.insert(b.end(), s.begin(), s.end());
  b.push_back(0);
}
Bytes fixture(bool duplicate = false) {
  Bytes b{'F', 'R', 'T', 'R', 'I', '0', '0', '3'};
  for (auto v : {3u, 1u, 0u, 0u, 0u, 3u, 1u, 2u, 1u, 1u, 0u, 0u, 0u, 0u})
    u32(b, v);
  for (float v : {0, 0, 0, 1, 0, 0, 0, 1, 0, 2, 3, 4})
    f32(b, v);
  for (auto v : {0u, 1u, 2u})
    u32(b, v);
  for (float v : {0, 0, 1, 0, 0, 1})
    f32(b, v);
  for (auto v : {0u, 1u, 2u})
    u32(b, v);
  for (const char *n : {"BlinkLeft", duplicate ? "BlinkLeft" : "Aah"}) {
    name(b, n);
    f32(b, .25f);
    for (std::int16_t v : {std::int16_t(-4), std::int16_t(2), std::int16_t(0),
                           std::int16_t(0), std::int16_t(0), std::int16_t(0),
                           std::int16_t(0), std::int16_t(0), std::int16_t(8)}) {
      auto u = std::bit_cast<std::uint16_t>(v);
      b.push_back(u);
      b.push_back(u >> 8);
    }
  }
  name(b, "Nose");
  u32(b, 1);
  u32(b, 2);
  return b;
}
int failures = 0;
void check(bool value, const char *label) {
  if (!value) {
    std::cerr << label << '\n';
    ++failures;
  }
}
void patch(Bytes &b, std::size_t at, std::uint32_t v) {
  for (int i = 0; i < 4; ++i)
    b[at + i] = v >> (8 * i);
}
int main() {
  auto b = fixture();
  TriMorphSet s;
  std::string error;
  check(readTriMorphs(b, s, error) == TriReadStatus::Ok, error.c_str());
  check(s.positions.size() == 3 && s.morphs.size() == 2 &&
            s.morphs[0].name == "BlinkLeft",
        "named expressions preserved");
  check(s.morphs[0].deltas[0][0] == -4 && s.morphs[0].scale == .25f,
        "signed quantized deltas preserved");
  check(s.modifiers.size() == 1 && s.modifiers[0].vertices[0] == 2 &&
            s.modifiers[0].positions[0][2] == 4,
        "absolute modifiers remain separate");
  check(s.uvTriangles == s.triangles && s.uvs.size() == 3,
        "UV topology retained");
  std::vector<TriPosition> out;
  std::vector<TriMorphWeight> weights{{0, .5f}, {1, .25f}};
  check(evaluateTriMorphs(s, s.positions, weights, out, error),
        "blend succeeds");
  check(out[0][0] == -.75f && out[0][1] == .375f && out[2][2] == 1.5f,
        "multiple relative expressions add without applying absolute modifier");
  check(evaluateTriMorphs(s, s.positions, {}, out, error) && out == s.positions,
        "neutral reset has no drift");
  auto neutral = s.positions;
  neutral[0][0] = 10;
  check(evaluateTriMorphs(s, neutral, weights, out, error) &&
            out[0][0] == 9.25f,
        "actor neutral head is not replaced by TRI reference head");
  for (float bad : {-1.f, 2.f, std::numeric_limits<float>::quiet_NaN()})
    check(!evaluateTriMorphs(s, neutral, std::vector<TriMorphWeight>{{0, bad}},
                             out, error) &&
              out.empty(),
          "bad weights fail transactionally");
  check(!evaluateTriMorphs(s, neutral,
                           std::vector<TriMorphWeight>{{0, 1}, {0, 1}}, out,
                           error),
        "duplicate expression rejected");
  check(!evaluateTriMorphs(s, neutral, std::vector<TriMorphWeight>{{99, 1}},
                           out, error),
        "unknown expression rejected");
  check(!evaluateTriMorphs(s, std::span<const TriPosition>{}, weights, out,
                           error),
        "mismatched topology count rejected");
  // Every possible truncation, including names, indices and short deltas.
  for (std::size_t n = 0; n < b.size(); ++n) {
    TriMorphSet partial = s;
    check(readTriMorphs(std::span(b).first(n), partial, error) !=
                  TriReadStatus::Ok &&
              partial.positions.empty(),
          "truncation clears output");
  }
  auto bad = b;
  bad[7] = '4';
  check(readTriMorphs(bad, s, error) == TriReadStatus::UnsupportedFormat,
        "unsupported version distinguished");
  bad = b;
  patch(bad, 8, 0xffffffffu);
  check(readTriMorphs(bad, s, error) == TriReadStatus::ResourceLimit,
        "hostile count bounded");
  bad = b;
  patch(bad, 64,
        std::bit_cast<std::uint32_t>(std::numeric_limits<float>::infinity()));
  check(readTriMorphs(bad, s, error) == TriReadStatus::MalformedData,
        "nonfinite vertex rejected");
  bad = b;
  patch(bad, 112, 3);
  check(readTriMorphs(bad, s, error) == TriReadStatus::MalformedData,
        "invalid triangle index rejected");
  bad = b;
  patch(bad, bad.size() - 4, 9);
  check(readTriMorphs(bad, s, error) == TriReadStatus::MalformedData,
        "invalid absolute modifier index rejected");
  bad = b;
  bad.push_back(0);
  check(readTriMorphs(bad, s, error) == TriReadStatus::MalformedData,
        "trailing data is not silently discarded");
  Bytes empty{'F', 'R', 'T', 'R', 'I', '0', '0', '3'};
  for (int i = 0; i < 14; ++i)
    u32(empty, 0);
  check(readTriMorphs(empty, s, error) == TriReadStatus::Ok &&
            s.positions.empty(),
        "valid empty TRI");
  check(readTriMorphs(fixture(true), s, error) == TriReadStatus::Ok &&
            s.ambiguousMorphNames && s.morphs.size() == 2,
        "retail duplicate names preserve ordinal identity");
  bad = b;
  patch(bad, 148, 3);
  check(readTriMorphs(bad, s, error) == TriReadStatus::MalformedData,
        "invalid UV triangle index rejected");
  bad = b;
  bad[173] = 'X';
  check(readTriMorphs(bad, s, error) == TriReadStatus::MalformedData,
        "unterminated morph name rejected");
  namespace fs = std::filesystem;
  auto root =
      fs::temp_directory_path() /
      ("odai-tri-" +
       std::to_string(
           std::chrono::steady_clock::now().time_since_epoch().count()));
  fs::create_directories(root / "base/meshes");
  fs::create_directories(root / "mod/meshes");
  auto write = [&](const fs::path &p, const Bytes &bytes) {
    std::ofstream f(p, std::ios::binary);
    f.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
  };
  bad = b;
  bad[7] = '4';
  write(root / "base/meshes/head.tri", bad);
  write(root / "mod/meshes/head.tri", b);
  ResolvedContentProfile profile;
  profile.dataRoot = root / "base";
  ContentLayer layer;
  layer.id = "morph-override";
  layer.name = "morph override";
  layer.root = root / "mod";
  profile.layers.push_back(layer);
  FalloutAssetSource assets;
  FalloutAssetSource::ResolvedAsset resolved;
  check(
      assets.open(profile) &&
          assets.resolveAssetWithProvider("Meshes/HEAD.TRI", resolved, error) &&
          resolved.providerId == "morph-override" &&
          readTriMorphs(resolved.bytes, s, error) == TriReadStatus::Ok,
      "winning loose override and case-insensitive TRI lookup");
  check(!assets.resolveAssetWithProvider("meshes/missing.tri", resolved, error),
        "missing TRI dependency is distinct from parse failure");
  fs::remove_all(root);
  return failures ? 1 : 0;
}
