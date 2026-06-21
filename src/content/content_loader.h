#pragma once

#include <filesystem>

// JSON content loader: parses a mod's data/*.json files into a ContentDatabase.
// Kept separate from content_database.h so nlohmann/json stays out of the public
// header (and out of the many game TUs that include the database).
namespace odai::content {

class ContentDatabase;

// Load every recognized data file under modDir/data/ into db. Missing files are
// skipped silently (so partial mods are valid); files that exist but fail to
// parse record an error into db.errors(). Later calls override earlier entries
// that share an id, which is how mods layer on top of the base game.
void loadModData(ContentDatabase& db, const std::filesystem::path& modDir);

}  // namespace odai::content
