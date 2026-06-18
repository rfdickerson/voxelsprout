#pragma once

#include "game/strategy_map.h"

#include <filesystem>
#include <string>

// Binary serialization for StrategyMap. Format mirrors the engine's imported-scene
// serializer style: a magic + version header followed by length-prefixed records.
namespace odai::game {

bool saveStrategyMap(const StrategyMap& map, const std::filesystem::path& outputPath);
bool loadStrategyMap(const std::filesystem::path& inputPath, StrategyMap& outMap);

// Human-readable description of the most recent save/load failure.
[[nodiscard]] const std::string& getStrategyMapLastError();

}  // namespace odai::game
