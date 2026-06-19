#pragma once

#include <string>
#include <vector>

// Buildable catalog for a city's production menu. Pure CPU data (no Vulkan, no
// UI types) so it can live in the game layer and be unit-tested in isolation.
namespace odai::game {

enum class BuildableKind { Unit, Building };

struct BuildableItem {
    std::string id;          // Stable key, e.g. "granary", "spearman".
    std::string name;        // Display name.
    std::string iconName;    // Resolved via ui::UiIconRegistry (e.g. "spearman", "food").
    BuildableKind kind = BuildableKind::Building;
    int productionCost = 0;  // Shields/hammers required to complete.
    std::string civpediaId;  // Article key (== id for now).
};

// Turns to finish given the production already accumulated and the per-turn
// output. Uses ceiling division; `perTurn` is clamped to >= 1 so a city with no
// production still reports a finite (worst-case) estimate. Returns 0 when the
// cost is already met.
int turnsToBuild(int productionCost, int accumulated, int perTurn);

// Default first-pass catalog (units + buildings) using icons that already exist
// in the icon registry. Replace with data-driven content later.
const std::vector<BuildableItem>& defaultBuildables();

// Returns the rich-text CivPedia article for the given item id (e.g. "granary",
// "spearman"). Returns an empty string when no article is registered.
const std::string& getPediaArticle(const std::string& id);

}  // namespace odai::game
