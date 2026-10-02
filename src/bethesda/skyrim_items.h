#pragma once
#include "bethesda/runtime_ids.h"
#include "import/bethesda/asset_source.h"
#include "import/bethesda/plugin_load_order.h"
#include <map>

namespace odai::bethesda {
struct SkyrimItemDefinition {
    RecordKey record;
    std::string name;
    std::string text;
    std::string model;
    float meleeDamage = 0.0f;
    float healing = 0.0f;
    std::string recordType;
    std::uint8_t weaponAnimationType = 0; // WEAP DNAM animation family
    std::uint32_t bipedSlots = 0u;
};
// Immutable winning-record metadata; no game text or assets are saved in ODAI saves.
bool loadSkyrimItems(const importer::bethesda::FalloutLoadOrder &order,
                     const importer::bethesda::FalloutAssetSource &assets,
                     std::map<RecordKey, SkyrimItemDefinition> &out, std::string &error);
} // namespace odai::bethesda
