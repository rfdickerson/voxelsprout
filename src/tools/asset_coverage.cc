#include "tools/asset_coverage.h"
#include "import/dds.h"
#include "import/fnv/cell_builder.h"
#include "import/fnv/content_record_index.h"
#include "import/fnv/image_space_records.h"
#include "import/fnv/nif_scene.h"
#include "import/fnv/weather_records.h"
#include "tools/asset_coverage_report.h"
#include <algorithm>
#include <fstream>
#include <map>
#include <nlohmann/json.hpp>
#include <set>
#include <sstream>

namespace {
std::string canonical(std::string path) {
    for (char &c : path)
        c = c == '/' ? '\\' : char(std::tolower(static_cast<unsigned char>(c)));
    return path;
}
template <class Material>
std::string materialSignature(const Material &m, std::vector<std::string> paths) {
    paths.resize(9);
    for (auto &path : paths)
        if (!path.empty())
            path = canonical(odai::importer::fnv::normalizeTexturePath(path));
    return nlohmann::json{{"shaderType", m.shaderType},
                          {"flags1", m.flags1},
                          {"flags2", m.flags2},
                          {"uvOffset", m.uvOffset},
                          {"uvScale", m.uvScale},
                          {"emissive", m.emissive},
                          {"emissiveMultiplier", m.emissiveMultiplier},
                          {"specular", m.specular},
                          {"specularStrength", m.specularStrength},
                          {"glossiness", m.glossiness},
                          {"alpha", m.alpha},
                          {"environmentScale", m.environmentScale},
                          {"textureClampMode", m.textureClampMode},
                          {"refractionStrength", m.refractionStrength},
                          {"paths", paths}}
        .dump();
}

} // namespace

bool writeAssetCoverage(const odai::importer::fnv::FalloutAssetSource &assets,
                        const odai::importer::fnv::FalloutLoadOrder &order,
                        const std::filesystem::path &output, std::string &error) {
    using namespace odai::importer;
    using namespace odai::importer::fnv;
    using nlohmann::json;
    if (output.extension() != ".json") {
        error = "coverage output must end in .json";
        return false;
    }
    json report = {
        {"schemaVersion", 2},
        {"loadOrderFingerprint", order.fingerprint()},
        {"scope", "Tamriel cells x=4..5, y=-11..-10; full virtual path and winning record census"},
        {"runtimeEvidence", "scene-builder output; GPU consumption not instrumented"}};
    report["loadOrder"] = json::array();
    for (const auto &plugin : order.entries())
        report["loadOrder"].push_back(
            {{"plugin", plugin.header.fileName}, {"path", plugin.path.string()}});
    report["assetLayerFingerprint"] = assets.modFingerprint();
    const auto paths = assets.virtualPaths();
    std::map<std::string, std::size_t> extensions;
    for (const auto &path : paths) {
        const auto dot = path.find_last_of('.');
        ++extensions[dot == std::string::npos ? "<none>" : path.substr(dot)];
    }
    report["virtualPaths"] = paths;
    report["virtualAssets"] = json::array();
    for (const auto &path : paths) {
        FalloutAssetSource::ResolvedAsset winner;
        std::string diagnostic;
        const bool resolved = assets.resolveAssetWithProvider(path, winner, diagnostic);
        report["virtualAssets"].push_back({{"path", path},
                                           {"resolved", resolved},
                                           {"provider", winner.providerName},
                                           {"providerId", winner.providerId},
                                           {"archive", winner.archiveName},
                                           {"physicalPath", winner.physicalPath.string()},
                                           {"fingerprint", winner.contentFingerprint},
                                           {"bytes", winner.bytes.size()},
                                           {"diagnostic", diagnostic},
                                           {"decoded", "not_assessed"},
                                           {"preserved", "not_assessed"},
                                           {"runtimeConsumed", "unmeasured"}});
    }
    report["extensions"] = extensions;
    report["archiveWarnings"] = assets.warnings();
    ContentRecordIndex records;
    if (!records.build(order, error))
        return false;
    std::map<std::string, std::size_t> recordTypes, deletedTypes;
    report["winningRecords"] = json::array();
    std::vector<std::uint32_t> recordIds;
    for (const auto &[id, versions] : records.records())
        recordIds.push_back(id);
    std::sort(recordIds.begin(), recordIds.end());
    for (const auto id : recordIds) {
        const auto &versions = records.records().at(id);
        const auto &winner = versions.back();
        ++(winner.deleted ? deletedTypes : recordTypes)[winner.type];
        report["winningRecords"].push_back({{"formId", id},
                                            {"type", winner.type},
                                            {"plugin", winner.pluginName},
                                            {"pluginIndex", winner.pluginIndex},
                                            {"flags", winner.flags},
                                            {"deleted", winner.deleted},
                                            {"overrideCount", versions.size() - 1},
                                            {"resolved", true},
                                            {"decoded", "header_only"},
                                            {"preserved", "not_assessed"},
                                            {"runtimeConsumed", "unmeasured"}});
    }
    report["winningRecordTypes"] = recordTypes;
    report["deletedRecordTypes"] = deletedTypes;
    report["recordFamiliesWithoutVisualReader"] = json::array();
    for (const auto *type : {"EFSH", "ARTO", "IPCT", "IPDS"})
        if (recordTypes.contains(type))
            report["recordFamiliesWithoutVisualReader"].push_back(type);

    FalloutWeatherTables weatherTables;
    if (!buildFalloutWeatherTables(order, weatherTables, error))
        return false;
    report["weatherPresentation"] = {
        {"diagnostics", weatherTables.presentationDiagnostics},
        {"precipitationCount", weatherTables.precipitation.size()},
        {"consumed",
         {"SPGD rain type and density (screen-space fallback)",
          "continuous day/night fog distance"}},
        {"preservedNotConsumed",
         {"SPGD particle texture and geometry", "snow", "WTHR sky statics and aurora model",
          "RFCT visual effect reference", "wind direction/range",
          "precipitation transition thresholds", "fog maximum"}}};
    for (const auto &[id, p] : weatherTables.precipitation) {
        std::vector<std::uint8_t> bytes;
        std::string textureError;
        const bool resolved =
            !p.texture.empty() && assets.resolveTexture(p.texture, bytes, textureError);
        report["weatherPresentation"]["precipitation"].push_back(
            {{"formId", id},
             {"editorId", p.editorId},
             {"type", p.type},
             {"density", p.density},
             {"texture", p.texture},
             {"textureResolved", resolved},
             {"gravityVelocity", p.gravityVelocity},
             {"rotationVelocity", p.rotationVelocity},
             {"size", {p.sizeX, p.sizeY}},
             {"centerRange", {p.centerMin, p.centerMax}},
             {"rotationRange", p.rotationRange},
             {"subtextures", {p.subtexturesX, p.subtexturesY}},
             {"boxSize", p.boxSize}});
    }
    for (const auto &[id, w] : weatherTables.weathers)
        report["weatherPresentation"]["weathers"].push_back(
            {{"formId", id},
             {"editorId", w.editorId},
             {"precipitation", w.precipitationFormId},
             {"precipitationResolved", weatherTables.precipitation.contains(w.precipitationFormId)},
             {"rainFallbackIntensity", weatherRainIntensity(weatherTables, w, true)},
             {"visualEffect", w.visualEffectFormId},
             {"skyStatics", w.skyStatics},
             {"auroraModel", w.auroraModel}});
    ImageSpaceTables imageSpaces;
    if (!buildImageSpaceTables(order, imageSpaces, error))
        return false;
    report["imageSpaces"] = {
        {"imgsCount", imageSpaces.spaces.size()},
        {"imadCount", imageSpaces.modifiers.size()},
        {"diagnostics", imageSpaces.diagnostics},
        {"consumed",
         {"weather IMSP", "interior XCIM", "bloom scale/threshold/radius", "adaptation speed",
          "cinematic saturation/brightness/contrast", "tint", "modifier fade", "no-sky DOF",
          "Papyrus ImageSpaceModifier.Apply/Remove"}},
        {"preservedNotConsumed",
         {"HDR receive threshold/white/sunlight/sky/adaptation strength", "IMAD target luminance",
          "targeted and sky-blurring DOF", "radial/motion/double-vision blur",
          "ApplyCrossFade and target tracking"}},
        {"unsupportedReferences", {"room and underwater image-space overrides"}},
        {"spaces", json::array()},
        {"modifiers", json::array()}};
    for (const auto &[id, space] : imageSpaces.spaces)
        report["imageSpaces"]["spaces"].push_back({{"formId", id},
                                                   {"editorId", space.editorId},
                                                   {"hdr", space.settings.hdr},
                                                   {"cinematic", space.settings.cinematic},
                                                   {"tint", space.settings.tint},
                                                   {"dof", space.settings.dof},
                                                   {"dofFlags", space.settings.dofFlags}});
    for (const auto &[id, modifier] : imageSpaces.modifiers)
        report["imageSpaces"]["modifiers"].push_back({{"formId", id},
                                                      {"editorId", modifier.editorId},
                                                      {"duration", modifier.duration},
                                                      {"animatable", modifier.animatable}});

    FalloutWorldTables tables;
    FalloutCellIndex cells;
    if (!buildFalloutWorldTables(order, tables, error) ||
        !buildFalloutCellIndex(order, cells, error))
        return false;
    const auto world = tables.worldspaceFormIdsByEditorId.find("tamriel");
    if (world == tables.worldspaceFormIdsByEditorId.end()) {
        error = "coverage requires Tamriel";
        return false;
    }
    std::map<std::string, json> references;
    std::map<std::string, std::size_t> emitted;
    std::set<std::uint32_t> emittedReferences;
    std::set<std::string> emittedEffects, emittedGrass;
    std::set<std::string> sceneTextures, sceneMaterials;
    report["riverwoodReferences"] = json::array();
    std::set<std::uint32_t> landTextures;
    std::map<std::uint32_t, std::set<std::uint32_t>> landCells;
    report["cellBuildEvidence"] = json::array();
    std::set<std::string> preservedTerrainNormals;
    for (const auto &entry : cells.cells) {
        if (entry.worldspaceFormId != world->second || entry.isInterior || entry.gridX < 4 ||
            entry.gridX > 5 || entry.gridZ < -11 || entry.gridZ > -10)
            continue;
        FalloutCellRecord cell;
        if (!extractFalloutCellMerged(cells, order, entry, cell, error))
            return false;
        if (cell.land) {
            for (auto id : cell.land->quadrantBaseTextureFormId) {
                landTextures.insert(id);
                landCells[id].insert(entry.cellFormId);
            }
            for (const auto &layer : cell.land->textureLayers) {
                landTextures.insert(layer.textureFormId);
                landCells[layer.textureFormId].insert(entry.cellFormId);
            }
        }
        const auto addReference = [&](const auto &ref, bool generated) {
            if (ref.isDeleted)
                return;
            report["riverwoodReferences"].push_back(
                {{"cell", entry.cellFormId},
                 {"grid", {entry.gridX, entry.gridZ}},
                 {"reference", ref.formId},
                 {"base", ref.baseFormId},
                 {"generatedGrass", generated},
                 {"initiallyDisabled", (ref.recordFlags & 0x800u) != 0}});
            const auto model = tables.staticModelPaths.find(ref.baseFormId);
            if (model == tables.staticModelPaths.end() || model->second.empty())
                return;
            auto path = canonical(normalizeModelPath(model->second));
            if (!path.starts_with("meshes\\"))
                path = "meshes\\" + path;
            if (references[path].is_null())
                references[path] = json::array();
            json evidence = {{"cell", entry.cellFormId},
                             {"base", ref.baseFormId},
                             {"reference", ref.formId},
                             {"generatedGrass", generated},
                             {"initiallyDisabled", (ref.recordFlags & 0x800u) != 0}};
            if (generated) {
                evidence["enginePosition"] = {ref.position[0], ref.position[2], -ref.position[1]};
                evidence["placementKey"] = std::to_string(entry.cellFormId) + ":" + path + ":" +
                                           evidence["enginePosition"].dump();
            }
            references[path].push_back(std::move(evidence));
        };
        for (const auto &ref : cell.references)
            addReference(ref, false);
        for (const auto &ref : scatterSkyrimGrass(cell, tables))
            addReference(ref, true);
        CellSceneBuilder builder(assets, tables);
        builder.addCell(cell);
        ImportedScene scene;
        builder.finish(scene);
        const auto &stats = builder.stats();
        report["cellBuildEvidence"].push_back(
            {{"cell", entry.cellFormId},
             {"grassInstances", stats.grassInstances},
             {"disabledReferencesSkipped", stats.disabledReferencesSkipped},
             {"editorMarkerModelsSkipped", stats.editorMarkerModelsSkipped},
             {"effectMeshesSkipped", stats.effectMeshesSkipped},
             {"refractionShapesSkipped", stats.refractionShapesSkipped},
             {"missingBases", stats.referencesDroppedBaseNotFound},
             {"basesWithoutModels", stats.referencesDroppedBaseHasNoModel},
             {"unresolvedMeshes", stats.referencesDroppedMeshUnresolved},
             {"unreadableMeshes", stats.referencesDroppedMeshUnreadable},
             {"unresolvedTextures", stats.unresolvedTexturePaths},
             {"textureBudgetExceeded", stats.textureBudgetExceeded}});
        for (const auto &material : scene.lightingMaterials)
            sceneMaterials.insert(materialSignature(
                material, std::vector<std::string>(std::begin(material.texturePaths),
                                                   std::end(material.texturePaths))));
        for (const auto &emitter : scene.particleEmitters)
            emittedEffects.insert(emitter.sourceId);
        for (const auto &texture : scene.textures)
            sceneTextures.insert(canonical(normalizeTexturePath(texture.sourcePath)));
        for (const auto &mesh : scene.meshes) {
            for (const auto &binding : mesh.terrainNormals) {
                for (const auto index : binding.textures) {
                    if (index < scene.textures.size())
                        preservedTerrainNormals.insert(canonical(scene.textures[index].sourcePath));
                }
            }
        }
        for (const auto &instance : scene.instances) {
            auto path = canonical(normalizeModelPath(instance.modelPath));
            if (!path.starts_with("meshes\\"))
                path = "meshes\\" + path;
            ++emitted[path];
            if (instance.sourceId.starts_with("grass_"))
                emittedGrass.insert(
                    std::to_string(entry.cellFormId) + ":" + path + ":" +
                    json({instance.transform[3], instance.transform[7], instance.transform[11]})
                        .dump());
            else
                emittedReferences.insert(instance.sourceReferenceFormId);
        }
    }
    report["assets"] = json::array();
    report["landscapeMaterials"] = json::array();
    for (auto id : landTextures) {
        const auto found = tables.landTextureSlots.find(id);
        if (found == tables.landTextureSlots.end())
            continue;
        report["landscapeMaterials"].push_back(
            {{"ltex", id},
             {"slots", found->second},
             {"preservedInWorldTables", true},
             {"runtimeSupportedSlots", {0, 1}},
             {"normalPreservedInScene",
              preservedTerrainNormals.contains(canonical(found->second[1]))},
             {"normalConsumed", "unmeasured"}});
        auto &material = report["landscapeMaterials"].back();
        material["cells"] = landCells[id];
        material["dependencies"] = json::array();
        for (std::size_t slot = 0; slot < found->second.size(); ++slot) {
            const auto &path = found->second[slot];
            if (path.empty())
                continue;
            std::vector<std::uint8_t> bytes;
            std::string diagnostic;
            const bool resolved = assets.resolveTexture(path, bytes, diagnostic);
            ImportedSceneTexture texture;
            const bool decoded = resolved && loadDdsFromMemory(bytes.data(), bytes.size(), texture);
            material["dependencies"].push_back(
                {{"slot", slot},
                 {"path", path},
                 {"status", !resolved  ? "missing_dependency"
                            : !decoded ? "unassessed"
                            : slot > 1 ? "unsupported_feature"
                                       : "supported_path"},
                 {"diagnostic", diagnostic},
                 {"stages",
                  {{"resolved", resolved ? "passed" : "failed"},
                   {"decoded", decoded    ? "passed"
                               : resolved ? "failed"
                                          : "not_attempted"},
                   {"preserved", sceneTextures.contains(canonical(normalizeTexturePath(path)))
                                     ? "scene_texture_inventory"
                                     : "not_observed"},
                   {"runtimeConsumed", "unmeasured"}}}});
        }
    }
    std::map<std::string, std::size_t> blocks;
    std::size_t missing = 0, decoded = 0;
    for (const auto &[path, refs] : references) {
        auto evidencedRefs = refs;
        for (auto &ref : evidencedRefs) {
            ref["sceneInstanceEmitted"] =
                ref.value("generatedGrass", false)
                    ? emittedGrass.contains(ref.value("placementKey", ""))
                    : emittedReferences.contains(ref["reference"].get<std::uint32_t>());
            std::ostringstream sourceId;
            sourceId << "refr_" << std::hex << std::uppercase
                     << ref["reference"].get<std::uint32_t>();
            ref["sceneEffectEmitted"] = !ref.value("generatedGrass", false) &&
                                        (emittedEffects.contains(sourceId.str() + "_particles") ||
                                         emittedEffects.contains(sourceId.str() + "_mist") ||
                                         emittedEffects.contains(sourceId.str()));
            ref["sceneEffectKind"] = !ref["sceneEffectEmitted"].get<bool>() ? "none"
                                     : emittedEffects.contains(sourceId.str() + "_particles")
                                         ? "authored_particle_subset"
                                     : emittedEffects.contains(sourceId.str() + "_mist")
                                         ? "authored_mist"
                                         : "fire_preset_fallback";
        }
        json item = {{"path", path},
                     {"references", evidencedRefs},
                     {"placementCount", refs.size()},
                     {"emittedInstances", emitted[path]},
                     {"gpuConsumed", "unmeasured"},
                     {"materials", json::array()}};
        FalloutAssetSource::ResolvedAsset resolved;
        if (!assets.resolveAssetWithProvider(path, resolved, error)) {
            item["status"] = "missing_asset";
            item["error"] = error;
            ++missing;
            report["assets"].push_back(std::move(item));
            continue;
        }
        item["resolved"] = true;
        item["provider"] = resolved.providerName;
        item["archive"] = resolved.archiveName;
        item["fingerprint"] = resolved.contentFingerprint;
        NifBlockSummary summary;
        if (!parseNifBlockSummary(resolved.bytes, summary, error)) {
            item["status"] = nifCoverageFailureStatus(error);
            item["error"] = error;
            report["assets"].push_back(std::move(item));
            continue;
        }
        item["blocks"] = summary.blockTypeNames;
        if (std::find(summary.blockTypeNames.begin(), summary.blockTypeNames.end(),
                      "NiParticleSystem") != summary.blockTypeNames.end()) {
            NifMist mist;
            std::string particleError;
            const bool supported = parseNifMist(resolved.bytes, mist, particleError);
            item["particles"] = {
                {"typedMistDecoded", supported},
                {"runtimePath", "alpha billboard mist"},
                {"diagnostic", particleError},
                {"remaining",
                 {"turbulence", "LOD modifier", "full retail integration equivalence"}}};
            if (supported)
                item["particles"]["authored"] = {{"texture", mist.texture},
                                                 {"birthRate", mist.rate},
                                                 {"life", mist.life},
                                                 {"lifeVariation", mist.lifeVariation},
                                                 {"box", mist.box},
                                                 {"emitterShape", static_cast<unsigned>(mist.emitterShape)},
                                                 {"volumeRadius", mist.volumeRadius},
                                                 {"volumeHeight", mist.volumeHeight},
                                                 {"declination", mist.declination},
                                                 {"declinationVariation", mist.declinationVariation},
                                                 {"planarAngle", mist.planarAngle},
                                                 {"planarVariation", mist.planarVariation},
                                                 {"speed", mist.speed},
                                                 {"gravity", mist.gravity},
                                                 {"drag", mist.drag},
                                                 {"scaleSamples", mist.scales.size()}};
        }

        for (const auto &type : summary.blockTypeNames) {
            ++blocks[type];
        }
        NifModel model;
        const bool parsed = parseNifStaticMesh(resolved.bytes, model, error);
        const bool geometry = parsed && !model.shapes.empty();
        item["nifDecoded"] = parsed;
        item["staticGeometryDecoded"] = geometry;
        item["status"] = geometry ? "partial_support" : "no_static_geometry";
        item["intentionalExclusions"] = {
            {"editorMarkerShapes", model.editorMarkerShapeCount},
            {"hiddenShapes", model.hiddenShapeCount},
            {"inactiveSwitchSubtrees", model.inactiveSwitchSubtreeCount}};
        if (!geometry && model.skippedShapeCount == 0 &&
            (model.editorMarkerShapeCount || model.hiddenShapeCount ||
             model.inactiveSwitchSubtreeCount))
            item["status"] = "intentional_exclusion";
        if (!parsed) {
            item["diagnostic"] = error;
            item["status"] = nifCoverageFailureStatus(error);
        }
        if (geometry)
            ++decoded;
        item["unresolvedProperties"] = model.unresolvedPropertyTypes;
        item["skippedShapes"] = model.skippedShapeCount;
        item["materials"] = json::array();
        for (const auto &shape : model.shapes) {
            const auto &m = shape.lightingMaterial;
            json material = {
                {"shape", shape.name},
                {"materialAnimationTrackCount",shape.materialAnimations.size()},
                {"unsupportedMaterialControllerCount",shape.unsupportedMaterialControllers},
                {"shaderType", m.shaderType},
                {"lightingProperty", m.present && m.shaderType!=0xffffffffu},
                {"parametersDecoded", m.parametersValid},
                {"flags1", m.flags1},
                {"flags2", m.flags2},
                {"uvTransformRuntimeSupport", m.present},
                {"uvOffset", m.uvOffset},
                {"uvScale", m.uvScale},
                {"emissive", m.emissive},
                {"emissiveMultiplier", m.emissiveMultiplier},
                {"glossiness", m.glossiness},
                {"specular", m.specular},
                {"specularStrength", m.specularStrength},
                {"unmappedFlags1", m.flags1 & ~((1u << 0) | (1u << 3) | (1u << 7) | (1u << 22))},
                {"unmappedFlags2", m.flags2 & ~((1u << 4) | (1u << 5) | (1u << 6) | (1u << 29))},
                {"specularEmissiveConsumed", "unmeasured"},
                {"textures", json::array()}};
            material["sourceParametersPreservedInScene"] =
                m.present && sceneMaterials.contains(materialSignature(m, m.textures));
            material["cookedLayoutSupportsSourceParameters"] = m.present;
            material["cookedRoundTripMeasured"] = false;
            material["standardLightingRuntimeSupported"] =
                m.parametersValid && m.shaderType <= 2 && (m.flags1 & (1u << 12)) == 0;
            material["environmentReflectionRuntimeSupported"] =
                m.shaderType == 1 && (m.flags1 & 128u) != 0;
            material["environmentScale"] = m.environmentScale;
            auto textures = m.textures;
            if (textures.empty())
                textures = {shape.diffuseTexturePath, shape.normalTexturePath};
            for (std::size_t slot = 0; slot < textures.size(); ++slot) {
                if (textures[slot].empty())
                    continue;
                std::vector<std::uint8_t> bytes;
                std::string dependencyError;
                ImportedSceneTexture texture;
                const auto texturePath = canonical(textures[slot]);
                const bool textureReference = texturePath.ends_with(".dds") ||
                                              texturePath.ends_with(".tga") ||
                                              texturePath.ends_with(".bmp");
                const bool found = assets.resolveTexture(textures[slot], bytes, dependencyError);
                const bool read = found && loadDdsFromMemory(bytes.data(), bytes.size(), texture);
                material["textures"].push_back(
                    {{"slot", slot},
                     {"path", textures[slot]},
                     {"preservedInScene",
                      sceneTextures.contains(canonical(normalizeTexturePath(textures[slot])))},
                     {"diagnostic", dependencyError},
                     {"resolved", found},
                     {"decoded2D", read && texture.arrayLayers == 1},
                     {"decodedCube", read && texture.arrayLayers == 6},
                     {"layers", texture.arrayLayers},
                     {"mips", texture.mipLevelCount},
                     {"referenceStatus", !textureReference
                                             ? "non_texture_token"
                                             : (found ? "resolved" : "missing_dependency")},
                     {"runtimeSupport",
                      slot < 2 ? "static_surface"
                               : (slot == 4
                                      ? "environment_cube"
                                      : (slot == 5 ? "environment_mask"
                                                   : (slot == 2 ? "glow" : "unsupported_slot")))}});
            }
            item["materials"].push_back(std::move(material));
        }
        report["assets"].push_back(std::move(item));
    }
    report["blockOccurrences"] = blocks;
    report["summary"] = {{"uniqueReferencedAssets", references.size()},
                         {"staticGeometryDecoded", decoded},
                         {"missingAssets", missing}};
    finalizeAssetCoverage(report);
    if (output.has_parent_path())
        std::filesystem::create_directories(output.parent_path());
    std::ofstream file(output);
    if (!file) {
        error = "cannot write coverage JSON";
        return false;
    }
    // Bethesda filenames/names can contain legacy code-page bytes. Keep the
    // census writable while fingerprints identify the original source bytes.
    file << report.dump(2, ' ', false, json::error_handler_t::replace) << '\n';
    auto markdown = output;
    markdown.replace_extension(".md");
    std::ofstream md(markdown);
    md << assetCoverageMarkdown(report);
    if (!file || !md) {
        error = "coverage output write failed";
        return false;
    }
    error.clear();
    return true;
}
