#include "tools/asset_coverage_report.h"
#include <algorithm>
#include <cctype>
#include <map>
#include <set>
#include <sstream>

namespace {
using nlohmann::json;
struct Gap {
    std::string category, feature, status;
    std::set<std::string> paths, placements, eligible;
    json examples = json::array();
    std::size_t occurrences = 0;
};
std::string refKey(const json &ref) {
    if (ref.contains("placementKey"))
        return ref["placementKey"].get<std::string>();
    return std::to_string(ref.value("cell", 0u)) + ":" +
           std::to_string(ref.value("reference", 0u)) + ":" +
           (ref.value("generatedGrass", false) ? "grass" : "authored");
}
std::string escape(std::string value) {
    for (char &c : value)
        if (c == '|' || c == '\n' || c == '\r')
            c = ' ';
    return value;
}
} // namespace

std::string nifCoverageFailureStatus(const std::string &diagnostic) {
    std::string lower = diagnostic;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) { return char(std::tolower(c)); });
    if (lower.find("unsupported") != std::string::npos)
        return "unsupported_feature";
    // A sequential block walk can fail because of either an unknown layout or
    // corruption in its predecessor; do not label that ambiguity malformed.
    if (lower.find("could not size nif block") != std::string::npos)
        return "unassessed";
    return "malformed_data";
}

void finalizeAssetCoverage(json &report) {
    report["schemaVersion"] = 2;
    report.erase("unconsumedControllerParticleOccurrences");
    report.erase("unconsumedControllerParticleAssets");
    report.erase("unconsumedControllerParticlePlacements");
    report["stageDefinitions"] = {
        {"resolved", "Winning bytes read through FalloutAssetSource; record headers through "
                     "ContentRecordIndex."},
        {"decoded", "Format parser result, independent of geometry presence."},
        {"preserved", "Evidence in scene-builder output; not an assumption from source parsing."},
        {"runtimeConsumed",
         "unmeasured: this CPU probe does not observe GPU use or runtime activation."}};
    report["statusDefinitions"] = {
        {"intentional_exclusion",
         "Marker/hidden/inactive-only model; excluded from eligible placement counts."},
        {"missing_dependency", "Referenced asset could not be resolved."},
        {"unsupported_feature", "Known missing/partial implementation, not malformed input."},
        {"malformed_data", "Parser reports invalid data."},
        {"unassessed", "No evidence sufficient to classify support; not a failure."}};
    std::map<std::string, Gap> gaps;
    std::set<std::string> allPlacements, eligiblePlacements, emittedPlacements;
    std::map<std::string, std::size_t> statuses;
    const auto add = [&](const json &item, const std::string &category, const std::string &feature,
                         const std::string &status) {
        auto &gap = gaps[category + ":" + feature + ":" + status];
        gap.category = category;
        gap.feature = feature;
        gap.status = status;
        ++gap.occurrences;
        const auto path = item.value("path", "");
        if (gap.paths.insert(path).second && gap.examples.size() < 3)
            gap.examples.push_back(
                {{"path", path}, {"references", item.value("references", json::array())}});
        for (const auto &ref : item.value("references", json::array())) {
            const auto key = refKey(ref);
            gap.placements.insert(key);
            if (!ref.value("initiallyDisabled", false) &&
                item.value("status", "") != "intentional_exclusion")
                gap.eligible.insert(key);
        }
    };
    const std::set<std::string> rigidControllers = {"NiTransformController", "NiKeyframeController",
                                                    "NiMultiTargetTransformController"};
    for (auto &item : report["assets"]) {
        const bool excluded = item.value("status", "") == "intentional_exclusion";
        std::size_t emitted = 0, eligible = 0;
        for (const auto &ref : item["references"]) {
            const auto key = refKey(ref);
            allPlacements.insert(key);
            if (!excluded && !ref.value("initiallyDisabled", false)) {
                ++eligible;
                eligiblePlacements.insert(key);
            }
            if (ref.value("sceneInstanceEmitted", false) ||
                ref.value("sceneEffectEmitted", false)) {
                ++emitted;
                emittedPlacements.insert(key);
            }
        }
        item["eligiblePlacementCount"] = eligible;
        item["scenePreservedPlacementCount"] = emitted;
        const bool mist =
            item.contains("particles") && item["particles"].value("typedMistDecoded", false);
        item["stages"] = {{"resolved", item.value("resolved", false) ? "passed" : "failed"},
                          {"decoded", item.value("nifDecoded", false) || mist ? "passed"
                                      : item.value("resolved", false)         ? "failed"
                                                                              : "not_attempted"},
                          {"preserved", excluded  ? "intentional_exclusion"
                                        : emitted ? "scene_output"
                                                  : "not_observed"},
                          {"runtimeConsumed", "unmeasured"}};
        const auto status = item.value("status", "unassessed");
        ++statuses[status];
        if (status == "missing_asset")
            add(item, "dependency", "model", "missing_dependency");
        if (status == "malformed_container" || status == "malformed_data")
            add(item, "nif_container", "parser_failure", "malformed_data");
        if (status == "unsupported_container" || status == "unsupported_feature")
            add(item, "nif_container", "unsupported_layout", "unsupported_feature");
        item["blockCoverage"] = json::array();
        for (const auto &block : item.value("blocks", json::array())) {
            const auto type = block.get<std::string>();
            const bool particle = type.starts_with("NiPSys") || type.starts_with("BSPSys") ||
                                  type == "NiParticleSystem" || type == "BSStripParticleSystem";
            const bool controller =
                type.find("Controller") != std::string::npos || type.ends_with("Ctlr");
            std::string support = "unassessed";
            if (particle)
                support = mist && type != "BSPSysLODModifier" ? "fixture_partial" : "unsupported_feature";
            else if (controller && rigidControllers.contains(type))
                support = "rigid_animation_path";
            else if(type=="BSEffectShaderPropertyFloatController" ||
                    type=="BSEffectShaderPropertyColorController" ||
                    type=="BSLightingShaderPropertyFloatController" ||
                    type=="BSLightingShaderPropertyColorController" || type=="NiFlipController")
                support="material_animation_reader_subset";
            else if (controller &&
                     (type.starts_with("BSEffectShaderProperty") ||
                      type.starts_with("BSLightingShaderProperty") || type == "NiUVController" ||
                      type == "NiTextureTransformController" || type == "NiAlphaController" ||
                      type == "NiMaterialColorController" || type == "NiGeomMorpherController"))
                support = "unsupported_feature";
            item["blockCoverage"].push_back({{"type", type}, {"support", support}});
            if (support == "unsupported_feature")
                add(item, controller ? "controller" : "nif_block", type, support);
            // Unknown blocks are retained for audit without asserting a missing feature.
            if (support == "unassessed" && controller)
                add(item, "controller", type, "unassessed");
        }
        if (mist) {
            add(item, "nif_block", "NiPSysGravityModifier:turbulence", "unsupported_feature");
            // LOD is reported by the actual block inventory above. Do not
            // invent an affected asset when the block is absent.
        }
        for (const auto &type : item.value("unresolvedProperties", json::array()))
            add(item, "nif_property", type.get<std::string>(), "unsupported_feature");
        for (auto &material : item["materials"]) {
            if(material.value("unsupportedMaterialControllerCount",0u)>0)
                add(item,"controller","unresolved_material_track","unsupported_feature");
            if (material.value("lightingProperty", false)) {
                if (!material.value("parametersDecoded", false))
                    add(item, "material", "lighting_parameters", "unassessed");
                else if (!material.value("standardLightingRuntimeSupported", false))
                    add(item, "shader_type", std::to_string(material.value("shaderType", 0u)),
                        "unsupported_feature");
                if (material.value("flags1", 0u) & (1u << 12))
                    add(item, "shader_flag", "flags1:bit12:model_space_normals",
                        "unsupported_feature");
                for (const auto *field : {"unmappedFlags1", "unmappedFlags2"}) {
                    auto mask = material.value(field, 0u);
                    for (unsigned bit = 0; bit < 32; ++bit)
                        if (mask & (1u << bit))
                            add(item, "shader_flag",
                                std::string(field) + ":bit" + std::to_string(bit), "unassessed");
                }
            }
            for (auto &texture : material["textures"]) {
                const bool token = texture.value("referenceStatus", "") == "non_texture_token";
                const bool resolved = texture.value("resolved", false);
                const bool decoded =
                    texture.value("decoded2D", false) || texture.value("decodedCube", false);
                texture["stages"] = {{"resolved", token      ? "not_applicable"
                                                  : resolved ? "passed"
                                                             : "failed"},
                                     {"decoded", token      ? "not_applicable"
                                                 : decoded  ? "passed"
                                                 : resolved ? "failed"
                                                            : "not_attempted"},
                                     {"preserved", texture.value("preservedInScene", false)
                                                       ? "scene_texture_inventory"
                                                       : "not_observed"},
                                     {"runtimeConsumed", "unmeasured"}};
                if (token)
                    continue;
                if (!resolved)
                    add(item, "texture_dependency", texture.value("path", ""),
                        "missing_dependency");
                // DDS reader has a boolean result: unsupported codec and malformed data
                // cannot be separated honestly without a richer decoder diagnostic.
                else if (!decoded)
                    add(item, "texture_decode", texture.value("path", ""), "unassessed");
                if (texture.value("runtimeSupport", "") == "unsupported_slot")
                    add(item, "texture_slot", std::to_string(texture.value("slot", 0)),
                        "unsupported_feature");
            }
        }
    }
    report["stageCounts"] = json::object();
    for (const auto *stage : {"resolved", "decoded", "preserved", "runtimeConsumed"}) {
        std::map<std::string, std::size_t> counts;
        for (const auto &item : report["assets"])
            ++counts[item["stages"][stage].get<std::string>()];
        report["stageCounts"][stage] = counts;
    }
    std::map<std::string, std::size_t> detailedModels;
    for (std::size_t i = 0; i < report["assets"].size(); ++i)
        detailedModels.emplace(report["assets"][i]["path"].get<std::string>(), i);
    for (auto &asset : report["virtualAssets"]) {
        auto found = detailedModels.find(asset.value("path", ""));
        if (found != detailedModels.end()) {
            asset["detailedAssetIndex"] = found->second;
            asset["stages"] = report["assets"][found->second]["stages"];
            asset.erase("decoded");
            asset.erase("preserved");
            asset.erase("runtimeConsumed");
        }
    }
    report["recordFamilyCoverage"] = json::array();
    const std::set<std::string> unsupportedRecords = {"EFSH", "ARTO", "IPCT", "IPDS"};
    for (auto it = report["winningRecordTypes"].begin(); it != report["winningRecordTypes"].end();
         ++it) {
        json family = {{"type", it.key()},
                       {"activeRecords", it.value()},
                       {"status", unsupportedRecords.contains(it.key()) ? "unsupported_feature"
                                                                        : "unassessed"},
                       {"representativeRecords", json::array()},
                       {"directRiverwoodReferences", json::array()},
                       {"transitiveRiverwoodReferences", "not_assessed"}};
        std::set<std::uint32_t> ids;
        for (const auto &record : report["winningRecords"])
            if (!record.value("deleted", false) && record.value("type", "") == it.key()) {
                ids.insert(record["formId"].get<std::uint32_t>());
                if (family["representativeRecords"].size() < 3)
                    family["representativeRecords"].push_back(record);
            }
        for (const auto &ref : report["riverwoodReferences"])
            if (ids.contains(ref.value("base", 0u)))
                family["directRiverwoodReferences"].push_back(ref);
        report["recordFamilyCoverage"].push_back(std::move(family));
    }
    report["gaps"] = json::array();
    for (const auto &[key, gap] : gaps)
        report["gaps"].push_back({{"category", gap.category},
                                  {"feature", gap.feature},
                                  {"status", gap.status},
                                  {"occurrences", gap.occurrences},
                                  {"uniqueAssets", gap.paths.size()},
                                  {"affectedPlacements", gap.placements.size()},
                                  {"eligiblePlacements", gap.eligible.size()},
                                  {"examples", gap.examples}});
    std::set<std::string> grassPlacements;
    for (const auto &item : report["assets"])
        for (const auto &ref : item["references"])
            if (ref.value("generatedGrass", false))
                grassPlacements.insert(refKey(ref));
    report["summary"]["generatedGrassCandidates"] = grassPlacements.size();
    report["summary"]["authoredPlacements"] = allPlacements.size() - grassPlacements.size();
    report["summary"]["referencedPlacements"] = allPlacements.size();
    report["summary"]["eligiblePlacements"] = eligiblePlacements.size();
    report["summary"]["scenePreservedPlacements"] = emittedPlacements.size();
    report["summary"]["assetStatuses"] = statuses;
    std::size_t resolvedVirtual = 0, activeRecords = 0;
    for (const auto &asset : report["virtualAssets"])
        if (asset.value("resolved", false))
            ++resolvedVirtual;
    for (const auto &record : report["winningRecords"])
        if (!record.value("deleted", false))
            ++activeRecords;
    report["summary"]["resolvedVirtualAssets"] = resolvedVirtual;
    report["summary"]["unresolvedVirtualCandidates"] =
        report["virtualAssets"].size() - resolvedVirtual;
    report["summary"]["activeRecordWinners"] = activeRecords;
    report["summary"]["deletedRecordWinners"] = report["winningRecords"].size() - activeRecords;
    report["summary"]["visibleFailures"] = "unmeasured";
    report["limitations"] = {
        "Full winning path/record census; detailed NIF and dependency inspection is scoped to "
        "Riverwood's four cells.",
        "Eligible placements exclude initially disabled references and wholly excluded "
        "marker/hidden models; eligibility is not actual visibility.",
        "Generated grass is counted separately in reference evidence; runtime density/activation "
        "can differ.",
        "Scene output can contain fire presets: sceneEffectKind distinguishes fallbacks from "
        "authored mist, and does not imply all modifiers survived.",
        "Scene preservation is measured before upload. GPU consumption and active draw counts are "
        "unmeasured.",
        "Unknown blocks/flags/record families are unassessed, not unsupported. Support policy is "
        "conservative and must evolve with runtime changes.",
        "Texture scene inventory proves residency candidates, not per-material binding or actual "
        "sampling. DDS decode failures need codec/malformed diagnostics.",
        "Plugin family examples include direct base references only; transitive spell/effect usage "
        "is not inferred.",
        "Unused virtual assets and deleted record winners are not visible failures; gap placement "
        "counts overlap and must not be summed."};
}

std::string assetCoverageMarkdown(const json &report) {
    std::ostringstream md;
    const auto &summary = report.at("summary");
    md << "# Riverwood asset coverage\n\n"
       << report.value("virtualAssets", json::array()).size()
       << " winning virtual asset candidates; "
       << report.value("winningRecords", json::array()).size()
       << " record winners (including deleted tombstones).\n\n"
       << summary.at("uniqueReferencedAssets") << " referenced models; "
       << summary.at("referencedPlacements") << " placements; " << summary.at("eligiblePlacements")
       << " initially eligible; " << summary.at("scenePreservedPlacements")
       << " preserved as scene instances/effects.\n\n"
       << "Stages: resolved bytes → decoded format → scene preservation → runtime consumption. "
       << "GPU consumption and visible failures are **unmeasured**. Detailed inspection covers "
          "Tamriel cells (4..5, -11..-10).\n\n"
       << "| Gap | Status | Assets | Eligible / all placements | Example "
          "|\n|---|---|---:|---:|---|\n";
    for (const auto &gap : report.at("gaps")) {
        if (gap.at("status") == "unassessed")
            continue;
        const auto &example = gap.at("examples").front();
        md << "| "
           << escape(gap.at("category").get<std::string>() + ":" +
                     gap.at("feature").get<std::string>())
           << " | " << gap.at("status").get<std::string>() << " | " << gap.at("uniqueAssets")
           << " | " << gap.at("eligiblePlacements") << " / " << gap.at("affectedPlacements")
           << " | " << escape(example.at("path").get<std::string>());
        if (!example.at("references").empty()) {
            const auto &ref = example.at("references").front();
            md << " (cell " << ref.at("cell") << ", ref " << ref.at("reference") << ")";
        }
        md << " |\n";
    }
    md << "\nKnown missing visual record readers: ";
    for (const auto &family : report.at("recordFamilyCoverage"))
        if (family.at("status") == "unsupported_feature")
            md << family.at("type").get<std::string>() << " (" << family.at("activeRecords")
               << ") ";
    md << "\n\nUnknown flags/controllers and every record family remain enumerated in JSON; "
          "unknown does not mean unsupported.\n\n";
    for (const auto &limitation : report.at("limitations"))
        md << "- " << limitation.get<std::string>() << '\n';
    return md.str();
}
