#pragma once

#include "game/economy.h"  // BuildingEffects, Yields

#include <cstdint>
#include <string>
#include <vector>

// Great People: a roster of famous historical figures (Euclid, Homer, Sun Tzu, ...)
// an empire can earn over a match and settle into one of its cities for a permanent,
// city-wide bonus. They are globally unique -- like world wonders, only one of each
// may ever be born in a game.
//
// This module is the catalog half of the feature: the static, data-driven list of
// who exists and what each one does. It is pure CPU data (no Vulkan / renderer / UI)
// and, like the tech tree and building catalog, is served from the active
// ContentDatabase (mods/base/data/great_people.json) so mods can add or retune
// figures without touching the engine.
//
// The earning + integration *rules* (great-person points, birth, settling a figure
// into a city, the yield bonus it then confers) live in game_sim.{h,cc} alongside
// the World/City/Empire types they operate on. See:
//   - World::greatPersonTaken / World::bornGreatPeople  (global uniqueness)
//   - Empire::greatPersonPoints / Empire::pendingGreatPeople
//   - City::greatPeople                                 (settled figures)
//   - integrateGreatPerson()                            (place a figure in a city)
namespace odai::game {

// What kind of figure this is. Drives portrait grouping, AI affinity (which class an
// empire is most likely to attract given its leader's personality), and flavor.
enum class GreatPersonClass : std::uint8_t {
    Scientist = 0,  // Great Scientist
    Writer,         // Great Writer
    Engineer,       // Great Engineer
    General,        // Great General
    Philosopher,    // Great Philosopher
    Count
};
const char* greatPersonClassName(GreatPersonClass cls);

// Parse the JSON class label ("Great Scientist", ...) into the enum. Returns
// GreatPersonClass::Count for an unrecognized label (the loader records an error).
GreatPersonClass greatPersonClassFromName(const std::string& label);

// One figure in the catalog. Static data loaded from great_people.json.
struct GreatPersonDef {
    std::string id;             // stable key, e.g. "euclid"
    std::string name;           // display name, e.g. "Euclid"
    std::string title;          // short epithet, e.g. "Father of Geometry"
    GreatPersonClass cls = GreatPersonClass::Scientist;

    // Cell in the great_people.png portrait atlas (column, row in the grid). The
    // app builds the icon-registry metadata from these so the art<->code contract is
    // this catalog, exactly like the leader/unit atlases.
    int portraitCol = 0;
    int portraitRow = 0;

    // The permanent bonus conferred on the city this figure is settled into. Always
    // city-scoped (BuildingEffects::Scope::City); reuses the very same effect block
    // buildings and wonders use, so computeCityYields() applies it with no new path.
    BuildingEffects bonus{};

    std::string bonusSummary;   // one-line, player-facing effect text (plain text)
    std::string description;    // rich-text flavor (see ui/rich_text.h markup)
};

// The full catalog (data-driven; delegates to the active ContentDatabase).
const std::vector<GreatPersonDef>& greatPeopleCatalog();

// Find a figure by id, or nullptr if none matches.
const GreatPersonDef* findGreatPerson(const std::string& id);

}  // namespace odai::game
