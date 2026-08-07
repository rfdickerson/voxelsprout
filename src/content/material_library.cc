#include "content/material_library.h"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <unordered_set>

#include <nlohmann/json.hpp>

#include "content/content_paths.h"
#include "import/imported_material.h"

namespace odai::content {

namespace {

using nlohmann::json;

// Reads an optional float triple into a float[3], leaving it untouched when the
// key is absent or malformed. House style: defaults come from the
// already-default-constructed struct, never restated here.
void readVec3(const json& node, const char* key, float (&out)[3], std::vector<std::string>& errors,
              const std::string& where) {
    if (!node.contains(key)) {
        return;
    }
    const json& v = node[key];
    if (!v.is_array() || v.size() != 3) {
        errors.push_back(where + ": '" + key + "' must be an array of 3 numbers");
        return;
    }
    for (std::size_t i = 0; i < 3; ++i) {
        if (!v[i].is_number()) {
            errors.push_back(where + ": '" + key + "' entries must be numbers");
            return;
        }
        out[i] = v[i].get<float>();
    }
}

}  // namespace

MaterialLibraryLoadResult loadMaterialLibraryFromJson(const std::string& jsonText,
                                                      const std::string& sourceLabel) {
    MaterialLibraryLoadResult result;
    // Slot 0 is the reserved sentinel and is never authored; it is present so
    // that the index written into vertex flags and the index into this vector
    // are the same number.
    result.materials.emplace_back();

    json root;
    try {
        root = json::parse(jsonText);
    } catch (const json::exception& e) {
        result.errors.push_back(sourceLabel + ": JSON parse error: " + e.what());
        return result;
    }
    if (!root.contains("materials") || !root["materials"].is_array()) {
        result.errors.push_back(sourceLabel + ": expected a top-level 'materials' array");
        return result;
    }

    for (const json& entry : root["materials"]) {
        const std::string where =
            sourceLabel + " material #" + std::to_string(result.materials.size());
        if (!entry.is_object()) {
            result.errors.push_back(where + ": entry is not an object");
            continue;
        }
        if (result.materials.size() >= importer::kImportedSceneMaterialTableCapacity) {
            result.errors.push_back(sourceLabel + ": more than " +
                                    std::to_string(importer::kImportedSceneMaterialTableCapacity -
                                                   1u) +
                                    " materials; the rest were dropped");
            break;
        }

        importer::ImportedSceneMaterial m{};
        m.name = entry.value("name", std::string{});
        if (m.name.empty()) {
            result.errors.push_back(where + ": missing 'name'");
            continue;  // an unnamed material cannot be referenced or hot-reloaded
        }
        readVec3(entry, "baseColor", m.baseColorTint, result.errors, where);
        readVec3(entry, "emissive", m.emissive, result.errors, where);
        m.metallic = entry.value("metallic", m.metallic);
        m.roughness = entry.value("roughness", m.roughness);
        m.emissiveStrength = entry.value("emissiveStrength", m.emissiveStrength);
        // "note" is deliberately ignored — it exists so the reasoning behind a
        // coefficient (why roughness 0.35 for bronze) has somewhere to live
        // beside the number instead of in a C++ comment nobody reads.
        result.materials.push_back(std::move(m));
    }

    validateMaterialLibrary(result, sourceLabel);
    return result;
}

MaterialLibraryLoadResult loadMaterialLibraryFromFile(const std::filesystem::path& path) {
    const std::filesystem::path resolved = resolveContentPath(path);
    const std::string label = resolved.string();

    std::ifstream file(resolved);
    if (!file) {
        MaterialLibraryLoadResult result;
        result.materials.emplace_back();  // sentinel, so callers get a usable table
        result.errors.push_back(label + ": cannot open file");
        return result;
    }
    std::ostringstream buffer;
    buffer << file.rdbuf();
    return loadMaterialLibraryFromJson(buffer.str(), label);
}

void validateMaterialLibrary(MaterialLibraryLoadResult& result, const std::string& sourceLabel) {
    std::unordered_set<std::string> seen;
    for (std::size_t i = 1; i < result.materials.size(); ++i) {
        const importer::ImportedSceneMaterial& m = result.materials[i];
        const std::string where = sourceLabel + " material '" + m.name + "'";
        // Duplicate names break hot reload, which matches by name to keep
        // indices stable across an edit.
        if (!seen.insert(m.name).second) {
            result.errors.push_back(where + ": duplicate name");
        }
        const auto range = [&](float v, const char* field) {
            if (v < 0.0f || v > 1.0f) {
                result.errors.push_back(where + ": '" + field + "' is " + std::to_string(v) +
                                        ", outside [0,1]");
            }
        };
        range(m.metallic, "metallic");
        range(m.roughness, "roughness");
        if (m.emissiveStrength < 0.0f) {
            result.errors.push_back(where + ": 'emissiveStrength' is negative");
        }
    }
}

std::string materialLibraryToJson(const std::vector<importer::ImportedSceneMaterial>& materials,
                                  int indent) {
    json root;
    root["version"] = 1;
    json array = json::array();
    // Starts at 1: slot 0 is the sentinel, not authored content.
    for (std::size_t i = 1; i < materials.size(); ++i) {
        const importer::ImportedSceneMaterial& m = materials[i];
        json entry;
        entry["name"] = m.name;
        entry["baseColor"] = {m.baseColorTint[0], m.baseColorTint[1], m.baseColorTint[2]};
        entry["metallic"] = m.metallic;
        entry["roughness"] = m.roughness;
        // Emissive is omitted when unused, so the common case stays readable.
        if (m.emissiveStrength > 0.0f) {
            entry["emissive"] = {m.emissive[0], m.emissive[1], m.emissive[2]};
            entry["emissiveStrength"] = m.emissiveStrength;
        }
        array.push_back(std::move(entry));
    }
    root["materials"] = std::move(array);
    return root.dump(indent);
}

std::uint32_t findMaterialIndex(const std::vector<importer::ImportedSceneMaterial>& materials,
                                std::string_view name) {
    for (std::size_t i = 1; i < materials.size(); ++i) {
        if (materials[i].name == name) {
            return static_cast<std::uint32_t>(i);
        }
    }
    return 0u;
}

}  // namespace odai::content
