#pragma once

// Named PBR material library, authored as JSON and consumed by the renderer as
// an indexed table.
//
// A deliberate sibling of ContentDatabase rather than a member of it:
// content_database.h includes seven game/*.h 4X headers, and materials have no
// business dragging Civ types into the renderer, the procgen generators or the
// import tests.
//
// Materials load straight into importer::ImportedSceneMaterial — the same type
// the cooked .bin carries and the renderer uploads — so there is no parallel
// struct to keep in sync. nlohmann never appears in this header, matching the
// rule stated verbatim in content_loader.h and dialogue_io.h.
//
// Array order assigns the index: the first entry becomes slot 1, because slot 0
// is the reserved sentinel (see import/imported_material.h). Callers get a
// vector that is already 1-based with slot 0 filled in.

#include <filesystem>
#include <string>
#include <vector>

#include "import/imported_scene.h"

namespace odai::content {

// Non-fatal by design, mirroring dialogue_io.h: a malformed entry is skipped and
// recorded rather than aborting the load, so one bad line in a hot-reloaded file
// cannot blank a scene.
struct MaterialLibraryLoadResult {
    std::vector<importer::ImportedSceneMaterial> materials;  // 1-based; [0] is the sentinel
    std::vector<std::string> errors;

    [[nodiscard]] bool ok() const { return errors.empty(); }
};

MaterialLibraryLoadResult loadMaterialLibraryFromJson(const std::string& jsonText,
                                                      const std::string& sourceLabel = "<memory>");
MaterialLibraryLoadResult loadMaterialLibraryFromFile(const std::filesystem::path& path);

// Serializes back to the authoring format. Slot 0 is omitted — it is an
// implementation detail of the index scheme, not something to author.
std::string materialLibraryToJson(const std::vector<importer::ImportedSceneMaterial>& materials,
                                  int indent = 2);

// Cross-entry checks, run as a separate pass after parsing (precedent:
// dialogue_io.cc's validateReferences). Appends to `errors`.
void validateMaterialLibrary(MaterialLibraryLoadResult& result, const std::string& sourceLabel);

// Index of a material by name, or 0 when absent — which is exactly the "no
// library material" sentinel, so an unmatched lookup degrades to legacy shading
// rather than to a wrong material.
[[nodiscard]] std::uint32_t findMaterialIndex(
    const std::vector<importer::ImportedSceneMaterial>& materials, std::string_view name);

}  // namespace odai::content
