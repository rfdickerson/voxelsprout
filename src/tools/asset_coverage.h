#pragma once
#include <filesystem>
#include <string>
#include "import/fnv/asset_source.h"
#include "import/fnv/plugin_load_order.h"

bool writeAssetCoverage(const odai::importer::fnv::FalloutAssetSource& assets,
    const odai::importer::fnv::FalloutLoadOrder& order,
    const std::filesystem::path& output, std::string& error);
