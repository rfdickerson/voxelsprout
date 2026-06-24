#pragma once

#include "ui/ui_draw_list.h"

#include <filesystem>

// Binary cache for a tessellated vector icon (a UiGeometryBlock + its bake size).
// Mirrors the project's other binary caches (game/strategy_map_io, the font
// cache): a magic + version header followed by POD-blitted vertex/index data.
// `.odaivec` files are derived artifacts produced by odai_svg_bundler and are
// never committed; a version mismatch is rejected so callers re-tessellate.
namespace odai::ui {

bool writeVectorCache(const std::filesystem::path& path, const UiGeometryBlock& block,
                      float sizePx);

// On success fills `outBlock` (with a single SolidColor command synthesized) and
// `outSizePx`. Returns false on IO error, bad magic, or version mismatch.
bool loadVectorCache(const std::filesystem::path& path, UiGeometryBlock& outBlock,
                     float& outSizePx);

}  // namespace odai::ui
