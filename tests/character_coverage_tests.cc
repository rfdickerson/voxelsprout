#include "import/bethesda/character_asset_manifest.h"
#include "tools/character_coverage.h"
#include <nlohmann/json.hpp>
#include <chrono>
#include <fstream>
#include <iostream>
using namespace odai::importer::bethesda;
namespace fs = std::filesystem;
int main() {
    int failures = 0;
    auto check = [&](bool condition, const char* message) { if (!condition) { ++failures; std::cerr << message << '\n'; } };
    const auto root = fs::temp_directory_path() / ("odai-character-coverage-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    auto write = [&](const fs::path& path, const std::string& bytes) { fs::create_directories(path.parent_path()); std::ofstream out(path, std::ios::binary); out << bytes; };
    check(characterAssetPath("Animations/Walk.hkx", "meshes/actors/wolf/characters/wolf.hkx") == "meshes\\actors\\wolf\\animations\\walk.hkx", "creature relative root");
    check(characterAssetPath("wolf.hkx", "meshes/actors/wolf/characters/wolf.hkx", true) == "meshes\\actors\\wolf\\behaviors\\wolf.hkx", "behavior relative root");
    check(characterAssetPath("animations/walk.hkx", "meshes/actors/ambient/chicken/characters/chicken.hkx") == "meshes\\actors\\ambient\\chicken\\animations\\walk.hkx", "nested actor bundle");
    check(characterAssetPath("../SharedKillMoves/a.hkx", "meshes/actors/bear/behaviors/bear.hkx") == "meshes\\actors\\sharedkillmoves\\a.hkx", "bounded parent reference");
    check(characterAssetPath("animations/idle.hkx", "meshes/actors/character/_1stperson/characters/first.hkx") == "meshes\\actors\\character\\_1stperson\\animations\\idle.hkx", "first person bundle");
    check(characterAssetPath("../escape.hkx").empty() && characterAssetPath("/etc/file").empty(), "reject escaping paths");
    const std::string pack = R"({"version":1,"rules":[{"id":"a","state":"idle","variants":[{"clip":"meshes/Test.json"},{"clip":"meshes/test.json","loop":false},{"clip":"idle"},{"clip":"meshes/missing.hkx"}]}]})";
    write(root / "low/odai/animations/a.json", "broken");
    write(root / "high/odai/animations/a.json", pack);
    write(root / "high/meshes/Test.json", R"({"duration":1,"tracks":[{"bone":"Root","translationKeys":[{"time":0,"value":[0,0,0]},{"time":1,"value":[2,4,8]}]}]})");
    write(root / "high/meshes/actors/test/bad.hkx", "not HKX");
    write(root / "high/meshes/actors/test/bad.tri", "TRIP" + std::string(64, '\0'));
    FalloutAssetSource assets;
    check(assets.addModDirectory(root / "low") && assets.addModDirectory(root / "high"), "open layers");
    auto manifest = discoverCharacterAssets(assets, {{"odai/animations/a.json", {}}, {"meshes/actors/test/bad.hkx", {}}});
    check(manifest.complete && manifest.assets.size() == 4, "deduplicate dependencies and skip catalog aliases");
    check(manifest.assets.at("odai\\animations\\a.json").status == "decoded", "winning loose override");
    check(manifest.assets.at("meshes\\missing.hkx").status == "missing_dependency", "missing clip classification");
    check(manifest.assets.at("meshes\\actors\\test\\bad.hkx").status == "decode_failed_unclassified", "boolean reader failure stays unclassified");
    odai::anim::Skeleton rig; rig.bones.push_back({"Root", -1});
    odai::anim::AnimationClip source, second; odai::anim::HkxDecodedClipMetadata metadata;
    FalloutAssetSource::ResolvedAsset resolution; std::string error;
    check(loadCharacterSourceClip(assets, "meshes/test.json", rig, nullptr, false, source, metadata, resolution, error), "shared native source decode");
    auto playback = source; playback.loop = false; playback.tracks.front().translationKeys.back().value.y = 0;
    check(loadCharacterSourceClip(assets, "meshes/test.json", rig, nullptr, false, second, metadata, resolution, error), "reload immutable source");
    check(second.tracks.front().translationKeys.back().value.y == 4 && source.tracks.front().translationKeys.back().value.y == 4, "playback cannot contaminate source tracks");
    auto u32 = [](std::string& bytes, std::uint32_t n) { for (int i=0; i<4; ++i) bytes.push_back(char(n >> (8*i))); };
    auto sub = [](std::string type, std::string value) { type.push_back(char(value.size())); type.push_back(char(value.size() >> 8)); return type + value; };
    auto record = [&](std::string type, unsigned id, unsigned flags, const std::string& body) { u32(type, body.size()); u32(type, flags); u32(type, id); u32(type, 0); u32(type, 0); return type + body; };
    for (bool patch : {false, true}) {
        auto body = sub("HEDR", std::string(12, '\0'));
        if (patch) body += sub("MAST", std::string("Base.esm\0", 9)) + sub("DATA", std::string(8, '\0'));
        auto bytes = record("TES4", 0, patch ? 0 : 1, body);
        bytes += record("HDPT", 0x100, 0, sub("MODL", patch ? "actors/test/winner.nif" : "actors/test/loser.nif") + sub("NAM0", std::string("\1\0\0\0",4)) + sub("NAM1", "actors/test/bad.tri"));
        bytes += record("HDPT", 0x101, patch ? 0x20 : 0, sub("MODL", "actors/test/deleted.nif"));
        write(root / (patch ? "Patch.esp" : "Base.esm"), bytes);
    }
    FalloutLoadOrder order;
    check(order.open(root, {"Base.esm", "Patch.esp"}, error), "load test plugins");
    check(writeCharacterCoverage(assets, order, root / "report.json", error), error.c_str());
    nlohmann::json report; std::ifstream(root / "report.json") >> report;
    check(report["records"].size() == 1 && report["deletedRecordWinners"] == 1, "winning records and tombstones");
    check(report["records"][0]["assetBindings"][1]["morphRole"] == "expression_tri", "head-part morph purpose retained");
    const auto serialized = report.dump();
    check(serialized.find("winner.nif") != std::string::npos && serialized.find("loser.nif") == std::string::npos && serialized.find("deleted.nif") == std::string::npos, "only winning dependency edges");
    check(serialized.find("unsupported_format") != std::string::npos, "unsupported TRI format classified");
    fs::remove_all(root);
    return failures ? 1 : 0;
}
