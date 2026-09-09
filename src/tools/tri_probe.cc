#include "tools/tri_probe.h"
#include "import/fnv/asset_source.h"
#include "import/fnv/content_profile.h"
#include "import/fnv/tri_morph.h"
#include <cmath>
#include <iostream>
#include <nlohmann/json.hpp>
int probeTriMorphs(const std::filesystem::path &source, const std::string &path,
                   bool useProfile) {
  using namespace odai::importer::fnv;
  using nlohmann::json;
  FalloutAssetSource assets;
  std::string error;
  if (useProfile) {
    ResolvedContentProfile p;
    if (!resolveContentProfile(source, {}, p, error) || p.hasErrors() ||
        !assets.open(p)) {
      std::cerr << "profile open failed: " << error << '\n';
      return 1;
    }
  } else if (!assets.open(source)) {
    std::cerr << "asset source open failed\n";
    return 1;
  }
  std::vector<std::string> paths;
  if (path == "--all") {
    for (const auto &p : assets.virtualPaths())
      if (p.ends_with(".tri"))
        paths.push_back(p);
  } else
    paths.push_back(path);
  json report = {{"format", "FRTRI003"},
                 {"runtimeConsumed", false},
                 {"assets", json::array()}};
  unsigned failures = 0;
  for (const auto &p : paths) {
    FalloutAssetSource::ResolvedAsset asset;
    TriMorphSet set;
    json item = {{"path", p}};
    if (!assets.resolveAssetWithProvider(p, asset, error)) {
      item["status"] = "missing_dependency";
      item["error"] = error;
      ++failures;
    } else {
      item["provider"] = asset.providerName;
      item["archive"] = asset.archiveName;
      item["fingerprint"] = asset.contentFingerprint;
      auto status = readTriMorphs(asset.bytes, set, error);
      item["status"] =
          status == TriReadStatus::Ok                  ? "decoded"
          : status == TriReadStatus::UnsupportedFormat ? "unsupported_format"
          : status == TriReadStatus::ResourceLimit     ? "resource_limit"
                                                       : "malformed_data";
      if (status != TriReadStatus::Ok) {
        item["error"] = error;
        ++failures;
      } else {
        item["vertices"] = set.positions.size();
        item["triangles"] = set.triangles.size();
        item["quads"] = set.quads.size();
        item["uvs"] = set.uvs.size();
        item["modifiers"] = set.modifiers.size();
        item["ambiguousMorphNames"] = set.ambiguousMorphNames;
        item["ambiguousModifierNames"] = set.ambiguousModifierNames;
        item["morphs"] = json::array();
        for (const auto &m : set.morphs) {
          double maxDelta = 0;
          for (auto d : m.deltas) {
            double sum = 0;
            for (auto v : d)
              sum += double(v) * v;
            maxDelta = std::max(maxDelta, std::sqrt(sum) * m.scale);
          }
          item["morphs"].push_back({{"name", m.name},
                                    {"scale", m.scale},
                                    {"maxDisplacement", maxDelta}});
        }
      }
    }
    report["assets"].push_back(std::move(item));
  }
  report["uniqueAssets"] = paths.size();
  report["decoded"] = paths.size() - failures;
  report["failures"] = failures;
  std::cout << report.dump(2) << '\n';
  return failures ? 1 : 0;
}
