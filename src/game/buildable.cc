#include "game/buildable.h"

#include "content/content_database.h"

#include <algorithm>

// The buildable catalog and CivPedia articles are now data-driven: they live in
// mods/base/data/buildables.json and mods/base/data/civpedia.json and are served
// by the active ContentDatabase. turnsToBuild stays here -- it is arithmetic, not
// content.
namespace odai::game {

int turnsToBuild(int productionCost, int accumulated, int perTurn) {
    const int remaining = productionCost - accumulated;
    if (remaining <= 0) return 0;
    const int rate = std::max(1, perTurn);
    return (remaining + rate - 1) / rate;  // Ceiling division.
}

const std::vector<BuildableItem>& defaultBuildables() {
    return content::activeContent().buildables();
}

const BuildableItem* findBuildable(const std::string& id) {
    return content::activeContent().findBuildable(id);
}

const std::string& getPediaArticle(const std::string& id) {
    return content::activeContent().pediaArticle(id);
}

}  // namespace odai::game
