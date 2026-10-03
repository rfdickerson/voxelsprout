#include "import/bethesda/runtime_item_scene.h"

#include "import/bethesda/cell_builder.h"

#include <cmath>

namespace odai::importer::bethesda {

bool buildRuntimeItemScene(const FalloutAssetSource& assets,
    const std::string& modelPath, const std::string& itemId,
    const std::array<double, 3>& enginePosition, float scale,
    ImportedScene& outScene, std::string& error) {
    outScene = {};
    if (modelPath.empty() || !std::isfinite(scale) || scale <= 0.0f ||
        !std::isfinite(enginePosition[0]) || !std::isfinite(enginePosition[1]) ||
        !std::isfinite(enginePosition[2])) {
        error = "runtime item requires a model and finite placement";
        return false;
    }
    FalloutWorldTables tables;
    tables.morrowind = true;
    tables.staticModelPaths.emplace(1u, modelPath);
    tables.staticEditorIds.emplace(1u, itemId);
    tables.staticRecordTypes.emplace(1u, "MISC");
    FalloutCellRecord cell;
    FalloutPlacedReference reference;
    reference.formId = 2u;
    reference.baseFormId = 1u;
    reference.position[0] = static_cast<float>(enginePosition[0]);
    reference.position[1] = -static_cast<float>(enginePosition[2]);
    reference.position[2] = static_cast<float>(enginePosition[1]);
    reference.scale = scale;
    cell.references.push_back(reference);
    CellSceneBuilder builder(assets, tables);
    builder.addCellStatics(cell);
    builder.finish(outScene);
    if (outScene.meshes.empty() || outScene.instances.empty()) {
        error = "runtime item model produced no renderable imported geometry: " + modelPath;
        return false;
    }
    error.clear();
    return true;
}

} // namespace odai::importer::bethesda
