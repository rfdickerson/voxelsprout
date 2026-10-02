#pragma once
#include "import/bethesda/asset_source.h"
#include "import/bethesda/plugin_load_order.h"
#include <filesystem>
bool writeCharacterCoverage(const odai::importer::bethesda::FalloutAssetSource& assets,
    const odai::importer::bethesda::FalloutLoadOrder& order,
    const std::filesystem::path& output, std::string& error);
