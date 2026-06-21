#include "game/great_people.h"

#include "content/content_database.h"

// The catalog is data-driven: the great-people list lives in JSON under
// mods/base/data and is served by the active ContentDatabase. These accessors keep
// their signatures so call sites are unchanged; they simply delegate to the loaded
// content (mirroring economy.cc's techTree()/buildingDefs()). The pure name<->enum
// helpers stay here because they are logic, not data.
namespace odai::game {

namespace {
const content::ContentDatabase& db() { return content::activeContent(); }
}  // namespace

const char* greatPersonClassName(GreatPersonClass cls) {
    switch (cls) {
        case GreatPersonClass::Scientist:   return "Great Scientist";
        case GreatPersonClass::Writer:      return "Great Writer";
        case GreatPersonClass::Engineer:    return "Great Engineer";
        case GreatPersonClass::General:     return "Great General";
        case GreatPersonClass::Philosopher: return "Great Philosopher";
        case GreatPersonClass::Count:       break;
    }
    return "Great Person";
}

GreatPersonClass greatPersonClassFromName(const std::string& label) {
    if (label == "Great Scientist")   return GreatPersonClass::Scientist;
    if (label == "Great Writer")      return GreatPersonClass::Writer;
    if (label == "Great Engineer")    return GreatPersonClass::Engineer;
    if (label == "Great General")     return GreatPersonClass::General;
    if (label == "Great Philosopher") return GreatPersonClass::Philosopher;
    return GreatPersonClass::Count;  // sentinel: unknown
}

const std::vector<GreatPersonDef>& greatPeopleCatalog() {
    return db().greatPeople();
}

const GreatPersonDef* findGreatPerson(const std::string& id) {
    return db().findGreatPerson(id);
}

}  // namespace odai::game
