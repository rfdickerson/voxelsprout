#include "game/religion.h"

#include "content/content_database.h"

namespace odai::game {

namespace {
const content::ContentDatabase& db() { return content::activeContent(); }
}  // namespace

const std::vector<ReligionDef>& religionDefs() {
    return db().religions();
}

const ReligionDef* findReligionDef(const std::string& id) {
    return db().findReligion(id);
}

}  // namespace odai::game
