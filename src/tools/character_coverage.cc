#include "tools/character_coverage.h"
#include "import/fnv/character_asset_manifest.h"
#include "import/fnv/content_record_index.h"
#include "import/fnv/nif_scene.h"
#include "import/fnv/tri_morph.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>

using namespace odai::importer::fnv;
using namespace odai::anim;
using nlohmann::json;
namespace {
bool family(std::string_view type) {
    return type == "NPC_" || type == "CREA" || type == "RACE" || type == "HDPT" ||
        type == "ARMO" || type == "ARMA" || type == "OTFT";
}
std::string text(const EsmSubrecordView& sub) {
    std::string value(reinterpret_cast<const char*>(sub.data), sub.size);
    const auto end = value.find('\0');
    if (end != std::string::npos) value.resize(end);
    return value;
}
std::string hex(std::uint32_t id) {
    std::ostringstream out; out << std::hex << std::setw(8) << std::setfill('0') << id; return out.str();
}
std::string bundle(const std::string& path) {
    return characterAssetBundleRoot(path);
}
}
bool writeCharacterCoverage(const FalloutAssetSource& assets, const FalloutLoadOrder& order,
    const std::filesystem::path& output, std::string& error) {
    ContentRecordIndex index;
    if (!index.build(order, error)) return false;
    std::map<std::string, std::set<std::string>> seeds;
    for (const auto& path : assets.virtualPaths()) {
        const bool characterMesh = path.ends_with(".nif") &&
            (path.find("\\actors\\") != std::string::npos || path.starts_with("meshes\\armor\\") || path.starts_with("meshes\\clothes\\"));
        if (characterMesh || path.ends_with(".hkx") || path.ends_with(".tri") ||
            (path.starts_with("odai\\animations\\") && path.ends_with(".json"))) seeds[path];
    }
    json records = json::array();
    std::size_t deleted = 0;
    for (const auto& [id, versions] : index.records()) if (family(versions.back().type) && versions.back().deleted) ++deleted;
    for (std::size_t plugin = 0; plugin < order.size(); ++plugin) {
        EsmReader reader;
        if (!reader.open(order.entries()[plugin].path)) { error = reader.lastError(); return false; }
        EsmReader::Visitor visitor;
        visitor.onRecordHeader = [&](const EsmRecordHeaderView& record) {
            if (!family(record.type)) return false;
            const auto* versions = index.versions(order.remapFormId(plugin, record.formId));
            return versions && versions->back().pluginIndex == plugin && !versions->back().deleted;
        };
        visitor.onRecord = [&](const EsmRecordView& record) {
            const auto id = order.remapFormId(plugin, record.formId);
            const auto label = std::string(record.type) + ":" + hex(id);
            json item = {{"id", hex(id)}, {"type", record.type}, {"plugin", order.entries()[plugin].header.fileName},
                {"dependencies", json::array()}, {"assetBindings", json::array()}, {"recordReferences", json::array()}};
            std::uint32_t morphRole = 0xffffffffu;
            for (const auto& sub : record.subrecords) {
                if (sub.type == "EDID") item["editorId"] = text(sub);
                if (record.type == "HDPT" && sub.type == "NAM0" && sub.size == 4)
                    morphRole = sub.data[0] | (std::uint32_t(sub.data[1]) << 8) |
                        (std::uint32_t(sub.data[2]) << 16) | (std::uint32_t(sub.data[3]) << 24);
                // ARMA MODL is a race FormID, not a model string. HDPT NAM1
                // contains a morph path; NAM0 identifies that path's purpose.
                const bool model = (sub.type == "MODL" && record.type != "ARMA") ||
                    sub.type == "MOD2" || sub.type == "MOD3" || sub.type == "MOD4" || sub.type == "MOD5" ||
                    (record.type == "RACE" && sub.type == "ANAM") || (record.type == "HDPT" && sub.type == "NAM1");
                if (model) {
                    const auto path = characterAssetPath(text(sub));
                    if (!path.empty() && (path.ends_with(".nif") || path.ends_with(".tri") || path.ends_with(".hkx"))) {
                        seeds[path].insert(label); item["dependencies"].push_back(path);
                        json binding = {{"field", sub.type}, {"path", path}};
                        if (record.type == "HDPT" && sub.type == "NAM1")
                            binding["morphRole"] = morphRole == 0 ? "race_morph" : morphRole == 1 ? "expression_tri" : morphRole == 2 ? "chargen_morph" : "unassessed";
                        if (record.type == "ARMA") binding["perspective"] = sub.type == "MOD4" || sub.type == "MOD5" ? "first_person" : "third_person";
                        item["assetBindings"].push_back(std::move(binding));
                    }
                }
                const bool reference = (record.type == "NPC_" && (sub.type == "RNAM" || sub.type == "WNAM" || sub.type == "PNAM" || sub.type == "DOFT" || sub.type == "SOFT" || sub.type == "TPLT")) ||
                    (record.type == "RACE" && sub.type == "WNAM") || (record.type == "HDPT" && sub.type == "HNAM") ||
                    (record.type == "ARMO" && sub.type == "MODL") ||
                    (record.type == "ARMA" && (sub.type == "RNAM" || sub.type == "MODL")) ||
                    (record.type == "OTFT" && sub.type == "INAM");
                if (reference && sub.size % 4 == 0) for (std::size_t i = 0; i < sub.size; i += 4) {
                    const auto* b = sub.data + i;
                    const std::uint32_t local = b[0] | (std::uint32_t(b[1]) << 8) | (std::uint32_t(b[2]) << 16) | (std::uint32_t(b[3]) << 24);
                    if (!local) continue;
                    const auto target = order.remapFormId(plugin, local);
                    const auto* versions = index.versions(target);
                    item["recordReferences"].push_back({{"field", sub.type}, {"id", hex(target)},
                        {"status", !versions ? "missing_record" : versions->back().deleted ? "deleted_record" : "resolved"}});
                }
            }
            // Skyrim FaceGen uses the defining plugin/local object id even when
            // another plugin overrides the actor record.
            if (record.type == "NPC_") {
                if (const auto* owner = order.ownerOf(id)) {
                    const auto localId = owner->slot.kind == FalloutPluginSlotKind::Light ? id & 0xfffu : id & 0xffffffu;
                    const auto path = characterAssetPath("meshes\\actors\\character\\facegendata\\facegeom\\" + owner->header.fileName + "\\" + hex(localId) + ".nif");
                    // Generated FaceGen is optional: absence is not a required
                    // dependency failure for templates/player/non-humanoids.
                    if (seeds.contains(path)) { seeds[path].insert(label); item["dependencies"].push_back(path); }
                }
            }
            records.push_back(std::move(item));
        };
        if (!reader.walk(visitor)) { error = reader.lastError(); return false; }
    }
    std::cerr << "Character coverage: " << records.size() << " winning records, " << seeds.size() << " asset seeds\n";
    auto manifest = discoverCharacterAssets(assets, seeds);
    // Decode skeletons once. Clip checks below use the authored animation rig;
    // this does not establish binding to a rendered actor or retargeting parity.
    std::map<std::string, HkxDecodedSkeleton> skeletons;
    for (auto& [path, entry] : manifest.assets) if (entry.kind == "animation_skeleton") {
        FalloutAssetSource::ResolvedAsset asset;
        HkxDecodedSkeleton rig;
        if (assets.resolveAssetWithProvider(path, asset, entry.error) && decodeHkxAnimationSkeleton(asset.bytes, rig, entry.error)) {
            entry.status = "decoded"; skeletons.emplace(path, std::move(rig));
        } else entry.status = "decode_failed_unclassified";
    }
    // A character definition owns its complete reachable graph/catalog closure.
    std::map<std::string, std::set<std::string>> clipRigs;
    for (const auto& [path, entry] : manifest.assets) if (entry.kind == "character_definition") {
        std::set<std::string> rigs, visited;
        for (const auto& dependency : entry.dependencies) if (skeletons.contains(dependency)) rigs.insert(dependency);
        std::vector<std::string> pending = entry.dependencies;
        while (!pending.empty()) {
            auto current = std::move(pending.back()); pending.pop_back();
            if (!visited.insert(current).second) continue;
            const auto found = manifest.assets.find(current);
            if (found == manifest.assets.end()) continue;
            if (found->second.kind == "animation_clip") clipRigs[current].insert(rigs.begin(), rigs.end());
            pending.insert(pending.end(), found->second.dependencies.begin(), found->second.dependencies.end());
        }
    }
    json report = {{"schemaVersion", 1}, {"profileFingerprint", assets.modFingerprint()},
        {"loadOrderFingerprint", order.fingerprint()}, {"complete", manifest.complete},
        {"runtimeConsumed", false}, {"deletedRecordWinners", deleted}, {"records", std::move(records)},
        {"diagnostics", manifest.diagnostics}, {"assets", json::array()}};
    std::map<std::string, std::size_t> statuses, kinds;
    std::size_t checked = 0;
    std::map<std::string, json> textureEvidence;
    for (auto& [path, entry] : manifest.assets) {
        json extra = {{"rigBinding", "unassessed"}, {"runtimeSupport", "unassessed"}};
        if (entry.status != "missing_dependency") {
            FalloutAssetSource::ResolvedAsset asset;
            if (path.ends_with(".tri") && assets.resolveAssetWithProvider(path, asset, entry.error)) {
                entry.kind = "facial_morph"; TriMorphSet morph;
                const auto status = readTriMorphs(asset.bytes, morph, entry.error);
                entry.status = status == TriReadStatus::Ok ? "decoded" : status == TriReadStatus::UnsupportedFormat ? "unsupported_format" :
                    status == TriReadStatus::ResourceLimit ? "resource_limit" : "malformed_data";
                extra["runtimeSupport"] = "actor_facial_deformation_not_integrated";
                extra["vertices"] = morph.positions.size(); extra["targets"] = morph.morphs.size();
            } else if (path.ends_with(".nif") && assets.resolveAssetWithProvider(path, asset, entry.error)) {
                entry.kind = "nif"; NifBlockSummary summary; NifSkinnedModel skin; NifModel mesh;
                if (!parseNifBlockSummary(asset.bytes, summary, entry.error)) entry.status = "decode_failed_unclassified";
                else {
                    extra["blocks"] = summary.blockTypeNames;
                    std::string skinError, meshError;
                    const bool skinOk = parseNifSkinnedMesh(asset.bytes, skin, skinError);
                    const bool meshOk = parseNifStaticMesh(asset.bytes, mesh, meshError);
                    entry.status = skinOk || meshOk ? "decoded" : "inspected";
                    extra["skinnedShapes"] = skin.shapes.size(); extra["staticShapes"] = mesh.shapes.size();
                    extra["skinDecodeError"] = skinError; extra["staticDecodeError"] = meshError;
                    std::set<std::string> textures;
                    auto collect = [&](const auto& shape) {
                        if (!shape.diffuseTexturePath.empty()) textures.insert(normalizeTexturePath(shape.diffuseTexturePath));
                        if (!shape.normalTexturePath.empty()) textures.insert(normalizeTexturePath(shape.normalTexturePath));
                        for (const auto& texture : shape.lightingMaterial.textures) if (!texture.empty() && texture.find('.') != std::string::npos)
                            textures.insert(normalizeTexturePath(texture));
                    };
                    for (const auto& shape : skin.shapes) collect(shape);
                    for (const auto& shape : mesh.shapes) collect(shape);
                    extra["textureDependencies"] = json::array();
                    for (const auto& texture : textures) {
                        const auto key = characterAssetPath(texture);
                        if (!textureEvidence.contains(key)) {
                            FalloutAssetSource::ResolvedAsset tex; std::string why;
                            const bool resolved = assets.resolveAssetWithProvider(key, tex, why);
                            textureEvidence[key] = {{"path", key}, {"status", resolved ? "resolved" : "missing_dependency"},
                                {"provider", tex.providerId}, {"fingerprint", tex.contentFingerprint}, {"decode", "unassessed"}};
                        }
                        extra["textureDependencies"].push_back(textureEvidence.at(key));
                    }
                }
            } else if (entry.kind == "animation_clip") {
                extra["bindings"] = json::array();
                auto rigs = clipRigs[path];
                bool inferred = false;
                if (rigs.empty()) {
                    // Inventory-only clips have no character reference. A unique
                    // bundle skeleton permits a clearly labeled diagnostic check.
                    for (const auto& [rig, data] : skeletons) if (!bundle(path).empty() && bundle(rig) == bundle(path)) rigs.insert(rig);
                    inferred = true;
                    if (rigs.size() != 1) rigs.clear();
                }
                bool decoded = false;
                for (const auto& rig : rigs) {
                    AnimationClip clip; HkxDecodedClipMetadata metadata; FalloutAssetSource::ResolvedAsset resolution; std::string why;
                    const auto& source = skeletons.at(rig);
                    const bool ok = loadCharacterSourceClip(assets, path, source.referenceSkeleton, &source, false, clip, metadata, resolution, why);
                    decoded |= ok;
                    extra["bindings"].push_back({{"skeleton", rig}, {"basis", inferred ? "unique_bundle_candidate" : "character_definition"},
                        {"status", ok ? "bound_to_source_skeleton" : "decode_or_binding_failed"}, {"boundTracks", metadata.boundTracks},
                        {"missingTracks", metadata.missingTracks}, {"error", why}});
                }
                if (!rigs.empty()) entry.status = decoded ? "decoded" : "decode_or_binding_failed";
                extra["rigBinding"] = rigs.empty() ? "unassessed_no_unique_skeleton" : "source_skeleton_only";
            }
        }
        if (entry.kind == "behavior_graph") extra["runtimeSupport"] = "execution_not_tested";
        const auto root = bundle(path);
        extra["actorFamily"] = root.empty() ? "record_dependency" : root.starts_with("meshes\\actors\\character\\") ? "humanoid" : "creature_or_other_rig";
        if (!root.empty() && !root.starts_with("meshes\\actors\\character\\")) extra["runtimeSupport"] = "creature_runtime_unassessed";
        extra.update({{"path", path}, {"kind", entry.kind}, {"status", entry.status}, {"provider", entry.provider},
            {"providerName", entry.providerName}, {"archive", entry.archive}, {"physicalPath", entry.physicalPath},
            {"fingerprint", entry.fingerprint}, {"bytes", entry.bytes}, {"referenced", !entry.references.empty()},
            {"references", entry.references}, {"dependencies", entry.dependencies}, {"unsupportedClasses", entry.unsupportedClasses}, {"error", entry.error}});
        ++statuses[entry.status]; ++kinds[entry.kind]; report["assets"].push_back(std::move(extra));
        if (++checked % 1000 == 0) std::cerr << "Character coverage: inspected " << checked << '/' << manifest.assets.size() << '\n';
    }
    report["statusCounts"] = statuses; report["kindCounts"] = kinds; report["uniqueAssets"] = manifest.assets.size();
    std::error_code ec;
    if (!output.parent_path().empty()) std::filesystem::create_directories(output.parent_path(), ec);
    if (ec) { error = ec.message(); return false; }
    std::ofstream jsonFile(output);
    jsonFile << report.dump(2, ' ', false, json::error_handler_t::replace) << '\n';
    if (!jsonFile) { error = "cannot write character coverage JSON"; return false; }
    auto markdown = output; markdown.replace_extension(".md");
    std::ofstream md(markdown);
    md << "# Skyrim character asset coverage\n\n" << manifest.assets.size() << " unique assets; " << report["records"].size()
       << " winning character records; " << deleted << " deleted winners excluded.\n\n"
       << "Discovery complete: " << (manifest.complete ? "yes" : "no") << ". Runtime/GPU consumption is unmeasured.\n"
       << "Clip binding checks use authored animation skeletons, not assembled render rigs.\n\n| Status | Assets |\n|---|---:|\n";
    for (const auto& [status, count] : statuses) md << "| " << status << " | " << count << " |\n";
    md << "\n## Representative unresolved assets\n\n";
    std::size_t examples = 0;
    for (const auto& item : report["assets"]) if (item["status"] != "decoded" && examples++ < 40)
        md << "- `" << item["path"].get<std::string>() << "`: " << item["status"].get<std::string>() << '\n';
    md << "\nUnreferenced inventory files are not visible failures. Boolean decoder failures remain unclassified. "
          "Facial deformation, creature playback and behavior execution are not established by this audit.\n";
    if (!md) { error = "cannot write character coverage Markdown"; return false; }
    return true;
}
