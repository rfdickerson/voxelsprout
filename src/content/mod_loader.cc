#include "content/mod_loader.h"

#include "content/content_loader.h"
#include "content/content_paths.h"

#include <iostream>

namespace odai::content {

std::filesystem::path baseModDir() {
    return resolveContentPath(std::filesystem::path("mods") / "base");
}

ContentDatabase loadBaseContent() {
    ContentDatabase db;
    const std::filesystem::path dir = baseModDir();
    loadModData(db, dir);

    // The base game must define the core catalogs. If they are empty the data
    // directory was not found (e.g. wrong working directory) -- surface it loudly
    // rather than silently shipping an empty ruleset.
    if (db.techs().empty() || db.buildings().empty() || db.units().empty()) {
        const std::string msg =
            "content: base game data not found or incomplete at " + dir.string() +
            " (expected mods/base/data/*.json)";
        db.addError(msg);
        std::cerr << msg << "\n";
    }
    return db;
}

}  // namespace odai::content
