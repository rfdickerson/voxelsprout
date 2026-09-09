#pragma once
#include "import/imported_scene.h"
#include <cstdint>

namespace odai::importer::fnv {
// Runtime-only selection. Source geometry is Bethesda Z-up; mask bits own
// four-by-four exterior cells, x first. No serialized material fields change.
struct SkyrimLodHandoff {
    std::int32_t tileX = 0, tileZ = 0;
    bool dropRegular = false;
    bool dropAll = false;
    bool clipResident = false;
    std::uint16_t residentMask = 0;
    bool operator==(const SkyrimLodHandoff&) const = default;
};
void applySkyrimLodHandoff(ImportedScene& scene, const SkyrimLodHandoff& handoff);
}
