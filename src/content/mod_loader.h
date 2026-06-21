#pragma once

#include "content/content_database.h"

#include <filesystem>

// Mod discovery and load-order resolution. Phase 1 is intentionally minimal: it
// resolves and loads the base game (mods/base). Phase 3 extends this to discover
// user mods, parse manifests, and resolve dependency/override order.
namespace odai::content {

// The base game's mod directory (mods/base), resolved against the content roots.
std::filesystem::path baseModDir();

// Build a ContentDatabase from the base game. Records a diagnostic error into the
// returned database if the data could not be found or is incomplete.
ContentDatabase loadBaseContent();

}  // namespace odai::content
