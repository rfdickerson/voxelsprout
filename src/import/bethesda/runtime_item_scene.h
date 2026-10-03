#pragma once

#include "import/imported_scene.h"

#include <array>
#include <string>

namespace odai::importer::bethesda {

class FalloutAssetSource;

// Builds one transient imported-scene chunk from the same TES3 NIF path and
// placement conversion used by streamed CELL references.
bool buildRuntimeItemScene(const FalloutAssetSource& assets,
    const std::string& modelPath, const std::string& itemId,
    const std::array<double, 3>& enginePosition, float scale,
    ImportedScene& outScene, std::string& error);

} // namespace odai::importer::bethesda
