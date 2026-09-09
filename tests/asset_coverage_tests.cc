#include "import/fnv/asset_source.h"
#include "import/fnv/content_record_index.h"
#include "tools/asset_coverage_report.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>

using nlohmann::json;
int main() {
    int failures = 0;
    const auto check = [&](bool ok, const char *message) {
        if (!ok) {
            std::cerr << message << '\n';
            ++failures;
        }
    };
    check(nifCoverageFailureStatus("Unsupported NIF version") == "unsupported_feature",
          "unsupported container must not be classified as malformed");
    check(nifCoverageFailureStatus("truncated header") == "malformed_data",
          "truncated container is malformed");
    check(nifCoverageFailureStatus("Could not size NIF block type") == "unassessed",
          "ambiguous sequential layout failure remains unassessed");
    const auto reference = [](unsigned id, bool disabled = false) {
        return json{{"cell", 1},
                    {"reference", id},
                    {"base", 10},
                    {"generatedGrass", false},
                    {"initiallyDisabled", disabled},
                    {"sceneInstanceEmitted", !disabled}};
    };
    json model = {
        {"path", "meshes\\a.nif"},
        {"status", "partial_support"},
        {"resolved", true},
        {"nifDecoded", true},
        {"references", {reference(1), reference(2, true)}},
        {"blocks",
         {"NiAlphaController", "NiAlphaController", "NiTransformController", "UnknownController"}},
        {"materials", json::array()}};
    model["materials"].push_back({{"lightingProperty", true},
                                  {"parametersDecoded", true},
                                  {"standardLightingRuntimeSupported", false},
                                  {"shaderType", 7},
                                  {"flags1", 1u << 12},
                                  {"unmappedFlags1", 1u << 31},
                                  {"textures",
                                   {{{"slot", 3},
                                     {"path", "textures/missing.dds"},
                                     {"resolved", false},
                                     {"runtimeSupport", "unsupported_slot"}},
                                    {{"slot", 6},
                                     {"path", "textures/bad.dds"},
                                     {"resolved", true},
                                     {"runtimeSupport", "unsupported_slot"}}}}});
    json marker = model;
    marker["path"] = "meshes\\marker.nif";
    marker["status"] = "intentional_exclusion";
    marker["references"] = {reference(3)};
    marker["references"][0]["sceneInstanceEmitted"] = false;
    json missing = {{"path", "meshes\\missing.nif"},
                    {"status", "missing_asset"},
                    {"references", {reference(4)}},
                    {"materials", json::array()}};
    missing["references"][0]["sceneInstanceEmitted"] = false;
    json malformed = missing;
    malformed["path"] = "meshes\\bad.nif";
    malformed["status"] = "malformed_container";
    malformed["resolved"] = true;
    malformed["references"] = {reference(5)};
    malformed["references"][0]["sceneInstanceEmitted"] = false;
    json mist = model;
    mist["path"] = "meshes\\mist.nif";
    mist["blocks"] = {"NiParticleSystem", "NiPSysBoxEmitter"};
    mist["particles"] = {{"typedMistDecoded", true}};
    mist["references"] = {reference(6)};
    mist["references"][0]["sceneInstanceEmitted"] = false;
    mist["references"][0]["sceneEffectEmitted"] = true;
    json report = {
        {"assets", {model, marker, missing, malformed, mist}},
        {"winningRecordTypes", {{"STAT", 1}, {"EFSH", 1}}},
        {"winningRecords",
         {{{"formId", 10}, {"type", "STAT"}, {"plugin", "override.esp"}, {"deleted", false}},
          {{"formId", 11}, {"type", "EFSH"}, {"deleted", false}},
          {{"formId", 12}, {"type", "EFSH"}, {"deleted", true}}}},
        {"riverwoodReferences", {reference(1)}},
        {"summary", {{"uniqueReferencedAssets", 5}}}};
    finalizeAssetCoverage(report);
    check(report["summary"]["referencedPlacements"] == 6, "deduplicate reference counts");
    check(report["summary"]["eligiblePlacements"] == 4, "exclude disabled and marker placements");
    check(report["summary"]["scenePreservedPlacements"] == 2,
          "count effect preservation separately from mesh decode");
    check(report["assets"][2]["stages"]["decoded"] == "not_attempted", "missing is not malformed");
    check(report["assets"][3]["stages"]["decoded"] == "failed", "malformed fails decode");
    check(report["assets"][0]["stages"]["runtimeConsumed"] == "unmeasured",
          "CPU emission is not GPU consumption");
    bool alpha = false, missingGap = false, malformedGap = false;
    for (const auto &gap : report["gaps"]) {
        if (gap["feature"] == "NiAlphaController") {
            alpha = true;
            check(gap["uniqueAssets"] == 2 && gap["affectedPlacements"] == 3 &&
                      gap["eligiblePlacements"] == 1,
                  "repeated blocks must not multiply affected placements");
        }
        if (gap["status"] == "missing_dependency")
            missingGap = true;
        if (gap["status"] == "malformed_data")
            malformedGap = true;
        check(gap["feature"] != "NiTransformController",
              "do not classify rigid animation as missing");
        check(gap["feature"] != "BSPSysLODModifier",
              "absent particle LOD block must not create an affected asset");
        check(gap["feature"] != "NiPSysBoxEmitter",
              "supported mist emitter must not be a missing feature");
    }
    check(report["assets"][0]["materials"][0]["textures"][0]["stages"]["decoded"] ==
              "not_attempted",
          "missing texture is not a decode failure");
    check(report["assets"][0]["materials"][0]["textures"][1]["stages"]["decoded"] == "failed",
          "texture decode failure is distinct from missing dependency");
    check(alpha && missingGap && malformedGap, "distinct gap statuses");
    check(report["recordFamilyCoverage"][0]["representativeRecords"].size() == 1,
          "exclude deleted record examples");
    check(assetCoverageMarkdown(report).find("GPU consumption and visible failures") !=
              std::string::npos,
          "markdown evidence caveat");
    auto grassReport = report;
    auto grass = model;
    grass["references"] = {reference(0), reference(0)};
    for (int i = 0; i < 2; ++i) {
        grass["references"][i]["generatedGrass"] = true;
        grass["references"][i]["placementKey"] = "cell:grass:" + std::to_string(i);
    }
    grassReport["assets"] = {grass};
    finalizeAssetCoverage(grassReport);
    check(grassReport["summary"]["generatedGrassCandidates"] == 2 &&
              grassReport["summary"]["referencedPlacements"] == 2,
          "generated grass with form ID zero must retain distinct placement keys");
    // Use the real resolver to ensure inventory and detailed reads agree on loose override winners.
    namespace fs = std::filesystem;
    auto root = fs::temp_directory_path() /
                ("odai-asset-coverage-test-" +
                 std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::remove_all(root);
    fs::create_directories(root / "base/meshes");
    fs::create_directories(root / "mod/meshes");
    std::ofstream(root / "base/meshes/a.nif") << "base";
    std::ofstream(root / "mod/meshes/a.nif") << "override";
    fs::create_directories(root / "base/Textures");
    std::ofstream(root / "base/Textures/Mixed.DDS") << "mixed";
    odai::importer::fnv::FalloutAssetSource assets;
    check(assets.open(root / "base") && assets.addModDirectory(root / "mod"),
          "open synthetic layers");
    const auto paths = assets.virtualPaths();
    check(paths.size() == 2, "inventory deduplicates shadowed asset");
    odai::importer::fnv::FalloutAssetSource::ResolvedAsset winner;
    std::string error;
    check(assets.resolveAssetWithProvider("meshes\\a.nif", winner, error), "winning bytes resolve");
    check(std::string(winner.bytes.begin(), winner.bytes.end()) == "override", "last layer wins");
    check(assets.resolveAssetWithProvider("textures\\mixed.dds", winner, error) &&
              std::string(winner.bytes.begin(), winner.bytes.end()) == "mixed",
          "canonical inventory resolves mixed-case base loose files");
    // Synthetic TES4 headers exercise the same winning-record index used by the report.
    const auto u32 = [](std::string &bytes, std::uint32_t n) {
        for (int i = 0; i < 4; ++i)
            bytes.push_back(char((n >> (8 * i)) & 255));
    };
    const auto record = [&](const char *type, unsigned id, unsigned flags,
                            const std::string &body) {
        std::string bytes(type, 4);
        u32(bytes, static_cast<unsigned>(body.size()));
        u32(bytes, flags);
        u32(bytes, id);
        u32(bytes, 0);
        u32(bytes, 0);
        bytes += body;
        return bytes;
    };
    const auto plugin = [&](bool patch) {
        std::string body = "HEDR";
        body.push_back(12);
        body.push_back(0);
        body.append(12, '\0');
        if (patch) {
            std::string master = "Base.esm";
            master.push_back('\0');
            body += "MAST";
            body.push_back(char(master.size()));
            body.push_back(0);
            body += master;
            body += "DATA";
            body.push_back(8);
            body.push_back(0);
            body.append(8, '\0');
        }
        auto bytes = record("TES4", 0, patch ? 0 : 1, body);
        bytes += record("STAT", 0x100, 0, {});
        bytes += record("STAT", 0x101, patch ? 0x20 : 0, {});
        std::ofstream out(root / (patch ? "Patch.esp" : "Base.esm"), std::ios::binary);
        out << bytes;
    };
    plugin(false);
    plugin(true);
    odai::importer::fnv::FalloutLoadOrder order;
    odai::importer::fnv::ContentRecordIndex index;
    check(order.open(root, {"Base.esm", "Patch.esp"}, error) && index.build(order, error),
          "synthetic record census");
    const auto *versions = index.versions(0x100);
    check(versions && versions->size() == 2 && versions->back().pluginName == "Patch.esp",
          "record census uses final override winner");
    const auto *deleted = index.versions(0x101);
    check(deleted && deleted->back().deleted, "deleted winner remains a tombstone");
    std::ofstream(root / "Patch.esp", std::ios::binary | std::ios::trunc) << "TES4";
    check(!index.build(order, error), "malformed plugin fails census with a diagnostic");
    fs::remove_all(root);
    return failures ? 1 : 0;
}
