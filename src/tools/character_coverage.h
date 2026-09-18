#pragma once
#include "import/fnv/asset_source.h"
#include "import/fnv/plugin_load_order.h"
#include <filesystem>
bool writeCharacterCoverage(const odai::importer::fnv::FalloutAssetSource& assets,
    const odai::importer::fnv::FalloutLoadOrder& order,
    const std::filesystem::path& output, std::string& error);
