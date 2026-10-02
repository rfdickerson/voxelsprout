#include "import/bethesda/image_space_records.h"
#include <bit>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
using namespace odai::importer::bethesda;
namespace {
int failures = 0;
void check(bool ok, const char *message) {
  if (!ok) {
    std::cerr << message << '\n';
    ++failures;
  }
}
using Bytes = std::vector<std::uint8_t>;
void u32(Bytes &b, std::uint32_t v) {
  for (int i = 0; i < 4; ++i)
    b.push_back((v >> (8 * i)) & 255);
}
void f32(Bytes &b, float f) { u32(b, std::bit_cast<std::uint32_t>(f)); }
Bytes floats(std::initializer_list<float> values) {
  Bytes b;
  for (float f : values)
    f32(b, f);
  return b;
}
void set32(Bytes &b, std::size_t offset, std::uint32_t v) {
  for (int i = 0; i < 4; ++i)
    b[offset + i] = (v >> (8 * i)) & 255;
}
void sub(Bytes &b, const std::string &sig, const Bytes &payload) {
  b.insert(b.end(), sig.begin(), sig.end());
  b.push_back(payload.size() & 255);
  b.push_back((payload.size() >> 8) & 255);
  b.insert(b.end(), payload.begin(), payload.end());
}
void record(Bytes &b, const std::string &sig, std::uint32_t id,
            const Bytes &payload, std::uint32_t flags = 0) {
  b.insert(b.end(), sig.begin(), sig.end());
  u32(b, payload.size());
  u32(b, flags);
  u32(b, id);
  u32(b, 0);
  u32(b, 0);
  b.insert(b.end(), payload.begin(), payload.end());
}
Bytes header(bool patch) {
  Bytes p;
  sub(p, "HEDR", floats({1.7f, 0, 0}));
  if (patch) {
    sub(p, "MAST", Bytes{'S', 'k', 'y', 'r', 'i', 'm', '.', 'e', 's', 'm', 0});
    sub(p, "DATA", Bytes(8));
  }
  Bytes out;
  record(out, "TES4", 0, p, 1);
  return out;
}
void write(const std::filesystem::path &path, const Bytes &bytes) {
  std::ofstream f(path, std::ios::binary);
  f.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
}
} // namespace
int main(int argc, char **argv) {
  ImageSpaceTables crossTables;
  ImageSpaceModifierRecord dark, bright;
  dark.formId = 1; dark.multiply[18] = {{0, 0}};
  bright.formId = 2; bright.multiply[18] = {{0, 2}};
  crossTables.modifiers[1] = dark; crossTables.modifiers[2] = bright;
  ImageSpaceSettings neutral;
  ImageSpaceCrossFade chain;
  chain.apply(1, 2);
  auto faded = chain.sample(neutral, crossTables, 1);
  check(std::abs(faded.cinematic[1] - 0.5f) < 1e-6f, "Cross-fade duration is seconds, not strength");
  chain.apply(2, 2);
  faded = chain.sample(neutral, crossTables, 0);
  check(std::abs(faded.cinematic[1] - 0.5f) < 1e-6f, "Interrupted cross-fade starts from the current mixture");
  faded = chain.sample(neutral, crossTables, 1);
  check(std::abs(faded.cinematic[1] - 1.25f) < 1e-6f, "New target interpolates against the interrupted chain");
  faded = chain.sample(neutral, crossTables, 1);
  check(std::abs(faded.cinematic[1] - 2) < 1e-6f, "Completed chain holds its target");
  chain.apply(0, 2);
  faded = chain.sample(neutral, crossTables, 1);
  check(std::abs(faded.cinematic[1] - 1.5f) < 1e-6f, "RemoveCrossFade fades toward the current base image space");
  faded = chain.sample(neutral, crossTables, 1);
  check(std::abs(faded.cinematic[1] - 1) < 1e-6f && !chain.active(), "Fade-out clears the chain");
  chain.apply(1, 0);
  check(chain.sample(neutral, crossTables, 0).cinematic[1] == 0, "Zero duration applies immediately");
  chain.clear();
  check(!chain.active(), "Scene reset clears cross-fade state");

  std::string error;
  auto hdr = floats({0.6f, 2, 0.7f, 0.25f, 1, 1, 1, 1, 0.5f});
  auto cine = floats({0.8f, 1.1f, 1.2f});
  auto tint = floats({0.2f, 1, 0.5f, 0});
  EsmRecordView r{"IMGS",
                  0x123,
                  0,
                  {{"HNAM", hdr.data(), std::uint32_t(hdr.size())},
                   {"CNAM", cine.data(), std::uint32_t(cine.size())},
                   {"TNAM", tint.data(), std::uint32_t(tint.size())}}};
  ImageSpaceRecord space;
  check(parseImageSpace(r, space, error), "IMGS parses authored fields");
  check(space.settings.hdr[3] == 0.25f && space.settings.cinematic[0] == 0.8f &&
            space.settings.tint[2] == 0.5f,
        "IMGS preserves channels");
  r.subrecords[0].size = 4;
  check(!parseImageSpace(r, space, error), "truncated HNAM rejected");
  r.subrecords[0].size = 36;
  set32(hdr, 0, 0x7fc00000);
  check(!parseImageSpace(r, space, error), "NaN HNAM rejected");
  set32(hdr, 0, std::bit_cast<std::uint32_t>(0.6f));
  Bytes dnam(244);
  set32(dnam, 0, 1);
  set32(dnam, 4, std::bit_cast<std::uint32_t>(2.0f));
  set32(dnam, 8 + 17 * 8, 2);
  auto keys = floats({0, 1, 2, 0});
  std::string sig(1, char(17));
  sig += "IAD";
  EsmRecordView mod{
      "IMAD", 0x124, 0, {{"DNAM", dnam.data(), 244}, {sig, keys.data(), 16}}};
  ImageSpaceModifierRecord m;
  check(parseImageSpaceModifier(mod, m, error),
        "IMAD binary-signature curve parses");
  ImageSpaceSettings dof;
  dof.dof = {0.5f, 25000, 25000};
  dof.dofFlags = 16880;
  check(imageSpaceDofRadius(dof) == 1.5f,
        "authored no-sky DOF radius and strength");
  dof.dofFlags = 16848;
  check(imageSpaceDofRadius(dof) == 0, "unsupported sky blur stays neutral");
  ImageSpaceSettings base;
  check(std::abs(applyImageSpaceModifier(base, m, 1, 1).cinematic[0] - 0.5f) <
            1e-6f,
        "curve interpolates");
  check(std::abs(applyImageSpaceModifier(base, m, 1, 0.5f).cinematic[0] -
                 0.75f) < 1e-6f,
        "modifier strength blends identity");
  check(applyImageSpaceModifier(base, m, 3, 1).cinematic[0] == 1,
        "expired modifier restores base");
  check(applyImageSpaceModifier(base, m, 1, 0).cinematic[0] == 1,
        "zero strength neutral");
  auto colorKeys = floats({0, 1, 0, 0, 0, 2, 0, 0, 1, 1});
  set32(dnam, 176, 2);
  set32(dnam, 236, 2);
  mod.subrecords.push_back({"TNAM", colorKeys.data(), 40});
  mod.subrecords.push_back({"NAM3", colorKeys.data(), 40});
  check(parseImageSpaceModifier(mod, m, error),
        "IMAD tint/fade RGBA arrays parse");
  const auto colored = applyImageSpaceModifier(base, m, 1, 1);
  check(colored.tint[0] == 0.5f && colored.tint[1] == 0.5f &&
            colored.tint[3] == 0.5f,
        "tint RGBA maps to amount/RGB correctly");
  check(colored.fade[0] == 0.5f && colored.fade[2] == 0.5f &&
            colored.fade[3] == 0.5f,
        "fade retains independent color and alpha");
  mod.subrecords.resize(2);
  set32(dnam, 176, 0);
  set32(dnam, 236, 0);
  set32(dnam, 8 + 17 * 8, 3);
  check(!parseImageSpaceModifier(mod, m, error),
        "curve count mismatch rejected");
  set32(dnam, 8 + 17 * 8, 2);
  set32(keys, 8, std::bit_cast<std::uint32_t>(-1.0f));
  check(!parseImageSpaceModifier(mod, m, error),
        "negative curve time rejected");
  set32(keys, 8, std::bit_cast<std::uint32_t>(2.0f));
  const auto dir =
      std::filesystem::temp_directory_path() / "odai_image_space_test";
  std::filesystem::create_directories(dir);
  auto master = header(false);
  Bytes data;
  sub(data, "HNAM", hdr);
  sub(data, "CNAM", cine);
  record(master, "IMGS", 0x123, data);
  Bytes refs;
  for (int i = 0; i < 4; ++i)
    u32(refs, 0x123);
  Bytes weather;
  sub(weather, "IMSP", refs);
  record(master, "WTHR", 0x125, weather);
  Bytes cell;
  sub(cell, "EDID", Bytes{'T', 'e', 's', 't', 0});
  Bytes ref;
  u32(ref, 0x123);
  sub(cell, "XCIM", ref);
  record(master, "CELL", 0x126, cell);
  write(dir / "Skyrim.esm", master);
  auto patch = header(true);
  set32(hdr, 12, std::bit_cast<std::uint32_t>(0.75f));
  data.clear();
  sub(data, "HNAM", hdr);
  record(patch, "IMGS", 0x123, data);
  write(dir / "Patch.esp", patch);
  FalloutLoadOrder order;
  check(order.open(dir, {"Skyrim.esm", "Patch.esp"}, error),
        "synthetic load order opens");
  ImageSpaceTables tables;
  check(buildImageSpaceTables(order, tables, error),
        "image-space load order imports");
  check(tables.spaces.at(0x123).settings.hdr[3] == 0.75f,
        "winning override replaces base");
  check(tables.cells.at(0x126).imageSpace == 0x123,
        "cell image-space reference resolves");
  bool found = false;
  auto sampled = sampleWeatherImageSpace(tables, 0x125, 10, 6, 18, found);
  check(found && sampled.hdr[3] == 0.75f,
        "weather resolves winning image space");
  ImageSpaceTables blendTables;
  blendTables.spaces[1].settings.cinematic[0] = 0;
  blendTables.spaces[2].settings.cinematic[0] = 2;
  blendTables.weatherSpaces[3] = {1, 2, 1, 2};
  check(sampleWeatherImageSpace(blendTables, 3, 7.5f, 6, 18, found)
                .cinematic[0] == 1,
        "image spaces blend on weather dawn/day timing");
  check(sampleWeatherImageSpace(blendTables, 3, -1, 6, 18, found).cinematic ==
            sampleWeatherImageSpace(blendTables, 3, 23, 6, 18, found).cinematic,
        "weather interpolation wraps midnight");
  sampleWeatherImageSpace(tables, 0x999, 10, 6, 18, found);
  check(!found, "missing weather neutral");
  record(patch, "IMGS", 0x123, {}, 0x20);
  write(dir / "Patch.esp", patch);
  check(buildImageSpaceTables(order, tables, error) &&
            !tables.spaces.contains(0x123),
        "deleted override removes base");
  std::filesystem::remove_all(dir);
  if (argc > 1) {
    FalloutLoadOrder real;
    check(real.open(argv[1], {"Skyrim.esm", "Update.esm"}, error),
          "real load order");
    if (buildImageSpaceTables(real, tables, error)) {
      std::cout << tables.spaces.size() << " IMGS, " << tables.modifiers.size()
                << " IMAD, " << tables.diagnostics.size() << " malformed\n";
      for (const auto &d : tables.diagnostics)
        std::cout << d << '\n';
      for (const auto &[id, s] : tables.spaces) {
        std::cout << s.editorId << " hdr:";
        for (auto x : s.settings.hdr)
          std::cout << ' ' << x;
        std::cout << " cine:";
        for (auto x : s.settings.cinematic)
          std::cout << ' ' << x;
        std::cout << '\n';
      }
    } else
      check(false, error.c_str());
  }
  return failures ? 1 : 0;
}
