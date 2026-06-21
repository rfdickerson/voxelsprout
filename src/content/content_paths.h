#pragma once

#include <filesystem>

// Locating data files at runtime for the content/mod system. Mirrors the
// resolveAssetPath logic in src/app/app.cc but lives in the (Vulkan-free,
// app-free) content layer so tools, tests, and the game core can all find
// mods/ and assets/ regardless of the working directory.
namespace odai::content {

// Resolve a relative path against the known content roots: the compiled-in
// project source dir (ODAI_PROJECT_SOURCE_DIR, when defined for this target),
// then the current working directory and a few parent levels. Returns the first
// candidate that exists, or the input unchanged if none matched (absolute paths
// are returned as-is).
std::filesystem::path resolveContentPath(const std::filesystem::path& relative);

}  // namespace odai::content
