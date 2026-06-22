#pragma once

#include "game/economy.h"

#include <string>
#include <vector>

// Religion definitions: historical faiths with per-city yield bonuses, happiness
// effects, and gated tech/building unlocks. Religions form a branching lineage
// (Canaanite -> Judaism -> Christianity -> Islam; Zoroastrianism independent).
// An empire may hold at most one state religion at a time; adopting a child
// religion replaces the parent.
namespace odai::game {

struct ReligionDef {
    std::string id;
    std::string name;
    std::string description;
    std::string parentReligion;    // "" = foundable standalone; "judaism" = reform only from Judaism
    std::string requiredTech;      // tech that unlocks the option to adopt this religion
    Yields flatBonus;              // per-city yield bonus applied each turn to every city
    int happinessBonus = 0;        // added to each city's happiness cap
    int happinessPenalty = 0;      // subtracted from each city's happiness cap
    std::vector<std::string> unlocksTechs;     // tech ids only researchable with this religion
    std::vector<std::string> uniqueBuildings;  // building ids only buildable with this religion
    std::string flavorText;
};

const std::vector<ReligionDef>& religionDefs();
const ReligionDef* findReligionDef(const std::string& id);

}  // namespace odai::game
