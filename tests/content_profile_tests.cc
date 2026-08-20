#include "import/fnv/content_profile.h"
#include "import/fnv/asset_source.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace {

namespace fs = std::filesystem;
using namespace odai::importer::fnv;

int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

void write(const fs::path& path, const std::string& text = {}) {
    fs::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output << text;
}

void testJsonProfile(const fs::path& root) {
    const fs::path data = root / "json/Data";
    const fs::path layer = root / "json/mods/architecture";
    fs::create_directories(data);
    fs::create_directories(layer / "meshes");
    write(data / "Skyrim.esm");
    write(layer / "meshes/test.nif", "nif");
    const fs::path profilePath = root / "json/profile.json";
    write(profilePath, R"({
      "version": 1,
      "name": "Large Skyrim",
      "game": "skyrim-se",
      "data_root": "Data",
      "layers": [{"id":"architecture","path":"mods/architecture"}],
      "plugins": ["Patch.esp"]
    })");

    ResolvedContentProfile profile;
    std::string error;
    check(resolveContentProfile(profilePath, {}, profile, error), error.c_str());
    check(profile.game == BethesdaGame::SkyrimSpecialEdition, "JSON game is parsed");
    check(profile.plugins.size() == 2u && profile.plugins.front() == "Skyrim.esm",
          "JSON profile receives its installed base master");
    check(profile.layers.size() == 1u && profile.layers[0].root == fs::weakly_canonical(layer),
          "JSON relative layer path resolves from the manifest");
    check(profile.fingerprint.size() == 32u, "JSON profile has stable 128-bit manifest digest");

    const fs::path exported = root / "json/exported.json";
    check(writeOdaiContentProfile(exported, profile, error), error.c_str());
    ResolvedContentProfile roundTrip;
    check(resolveContentProfile(exported, {}, roundTrip, error), error.c_str());
    check(roundTrip.plugins == profile.plugins, "exported JSON preserves plugin order");
}

void testMo2Profile(const fs::path& root) {
    const fs::path instance = root / "mo2";
    const fs::path profilePath = instance / "profiles/Huge";
    const fs::path mods = instance / "mods";
    const fs::path data = root / "game/Data";
    fs::create_directories(profilePath);
    fs::create_directories(mods / "Low Priority");
    fs::create_directories(mods / "High Priority");
    fs::create_directories(mods / "Disabled");
    fs::create_directories(instance / "overwrite");
    fs::create_directories(data);
    write(data / "Skyrim.esm");
    write(data / "Skyrim - Meshes.bsa");
    write(mods / "High Priority/Skywind - Assets.bsa");
    write(mods / "Low Priority/Unrelated Resources.bsa");
    write(profilePath / "modlist.txt",
          "# highest first\n+High Priority\n-Disabled\n+Low Priority\n");
    write(profilePath / "plugins.txt", "# active\n*Skywind.esp\nInactive.esp\n");
    write(profilePath / "loadorder.txt", "Skyrim.esm\nInactive.esp\nSkywind.esp\n");

    ContentProfileResolveOptions options;
    options.dataRootOverride = data;
    ResolvedContentProfile profile;
    std::string error;
    check(resolveContentProfile(profilePath, options, profile, error), error.c_str());
    check(profile.layers.size() == 3u, "MO2 includes enabled mods plus overwrite");
    check(profile.layers[0].name == "Low Priority" && profile.layers[1].name == "High Priority",
          "MO2 reverse-written modlist becomes ascending priority");
    check(profile.layers.back().id == "overwrite", "MO2 overwrite is the highest layer");
    check(profile.plugins.size() == 2u && profile.plugins[0] == "Skyrim.esm" &&
              profile.plugins[1] == "Skywind.esp",
          "MO2 activation is independent from load-order position");
    check(profile.archives.size() == 2u,
          "MO2 infers only official and active-plugin-associated archives");
    check(std::any_of(profile.diagnostics.begin(), profile.diagnostics.end(),
              [](const ContentDiagnostic& item) {
                  return item.code == "inactive-unassociated-archive";
              }),
          "MO2 reports an unassociated archive instead of activating it alphabetically");
}

void testOpenMwProfile(const fs::path& root) {
    const fs::path cfg = root / "openmw/openmw.cfg";
    const fs::path childDir = root / "openmw/child";
    const fs::path data = root / "mw/Data Files";
    const fs::path tr = root / "mw/TR";
    const fs::path local = root / "mw/generated";
    fs::create_directories(data);
    fs::create_directories(tr);
    fs::create_directories(local);
    write(data / "Morrowind.esm");
    write(data / "Morrowind.bsa");
    write(tr / "Tamriel_Data.esm");
    write(tr / "TR_Mainland.esm");
    write(childDir / "openmw.cfg", "data=../../mw/TR\ncontent=Tamriel_Data.esm\n");
    write(cfg,
          "data=../mw/Data Files\nconfig=child\ncontent=Morrowind.esm\n"
          "content=TR_Mainland.esm\ncontent=ignored.omwscripts\n"
          "fallback-archive=Morrowind.bsa\ndata-local=../mw/generated\n");

    ResolvedContentProfile profile;
    std::string error;
    check(resolveContentProfile(cfg, {}, profile, error), error.c_str());
    check(profile.game == BethesdaGame::Morrowind, "OpenMW selects Morrowind identity");
    check(profile.dataRoot == fs::weakly_canonical(data), "OpenMW finds the base Data Files root");
    check(profile.layers.size() == 2u && profile.layers.back().root == fs::weakly_canonical(local),
          "OpenMW data-local is highest priority");
    check(profile.plugins.size() == 3u, "OpenMW content order combines included configs");
    check(profile.archives.size() == 1u && profile.archives[0].required,
          "OpenMW fallback archives are explicit and required");
    check(!profile.diagnostics.empty() && profile.diagnostics[0].code == "unsupported-script-runtime",
          "OpenMW Lua is inventoried rather than executed");

    write(childDir / "openmw.cfg", "config=..\n");
    check(!resolveContentProfile(cfg, {}, profile, error) && error.find("cycle") != std::string::npos,
          "OpenMW configuration cycles fail clearly");
}

void testPersistentContentIndex(const fs::path& root) {
    const fs::path data = root / "index/Data";
    const fs::path layer = root / "index/layer";
    const fs::path cache = root / "index/cache";
    fs::create_directories(data);
    write(data / "Skyrim.esm");
    write(layer / "Meshes/Nested/Test.NIF", "first");
    ResolvedContentProfile profile;
    profile.name = "index-test";
    profile.game = BethesdaGame::SkyrimSpecialEdition;
    profile.dataRoot = data;
    ContentLayer layerDefinition;
    layerDefinition.id = "layer";
    layerDefinition.name = "Layer";
    layerDefinition.root = layer;
    profile.layers.push_back(std::move(layerDefinition));
    profile.plugins.push_back("Skyrim.esm");
    profile.fingerprint = "index-test-fingerprint";
    profile.forceContentReindex = false;
    setenv("XDG_CACHE_HOME", cache.c_str(), 1);

    FalloutAssetSource first;
    check(first.open(profile), "profile asset source builds its persistent content index");
    const std::string firstFingerprint = first.modFingerprint();
    std::vector<std::uint8_t> bytes;
    std::string error;
    check(first.resolveAsset("meshes\\nested\\test.nif", bytes, error) &&
              std::string(bytes.begin(), bytes.end()) == "first",
          "content index resolves paths case-insensitively");

    write(layer / "Meshes/Nested/Test.NIF", "second");
    fs::last_write_time(layer / "Meshes/Nested/Test.NIF",
                        fs::last_write_time(layer / "Meshes/Nested/Test.NIF") +
                            std::chrono::seconds(2));
    FalloutAssetSource changed;
    bytes.clear();
    check(changed.open(profile) &&
              changed.resolveAsset("meshes\\nested\\test.nif", bytes, error) &&
              std::string(bytes.begin(), bytes.end()) == "second",
          "metadata changes invalidate a cached layer index");
    check(changed.modFingerprint() != firstFingerprint,
          "nested asset metadata participates in the cache namespace fingerprint");

    write(layer / "Meshes/Nested/Added.NIF", "added");
    FalloutAssetSource added;
    bytes.clear();
    check(added.open(profile) && added.resolveAsset("meshes\\nested\\added.nif", bytes, error),
          "nested additions invalidate a cached layer index");

    const fs::path cacheDirectory = cache / "odai/content-index/v1";
    for (const fs::directory_entry& entry : fs::directory_iterator(cacheDirectory)) {
        if (entry.path().extension() == ".json") write(entry.path(), "{truncated");
    }
    FalloutAssetSource recovered;
    bytes.clear();
    check(recovered.open(profile) &&
              recovered.resolveAsset("meshes\\nested\\added.nif", bytes, error),
          "a corrupt persistent content index is rebuilt automatically");
    unsetenv("XDG_CACHE_HOME");
}

void testProfileDiscovery(const fs::path& root) {
    const fs::path profile = root / "discovery/Instance/profiles/Explorer";
    write(profile / "modlist.txt", "+Content\n");
    write(profile / "plugins.txt", "*Skyrim.esm\n");
    setenv("ODAI_PROFILE_ROOTS", (root / "discovery").c_str(), 1);
    const std::vector<fs::path> discovered = discoverContentProfiles();
    check(std::find(discovered.begin(), discovered.end(), profile) != discovered.end(),
          "profile discovery finds nested MO2 profile directories");
    unsetenv("ODAI_PROFILE_ROOTS");
}

}  // namespace

int main() {
    const fs::path root = fs::temp_directory_path() / "odai_content_profile_tests";
    std::error_code error;
    fs::remove_all(root, error);
    fs::create_directories(root, error);
    testJsonProfile(root);
    testMo2Profile(root);
    testOpenMwProfile(root);
    testPersistentContentIndex(root);
    testProfileDiscovery(root);
    fs::remove_all(root, error);
    if (failures != 0) {
        std::cerr << failures << " content-profile test(s) failed\n";
        return 1;
    }
    std::cout << "content profile tests passed\n";
    return 0;
}
