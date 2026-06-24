#pragma once

#include "ui/ui_draw_list.h"
#include "ui/ui_types.h"

#include <filesystem>
#include <string>
#include <string_view>
#include <unordered_map>

// Runtime registry of tessellated vector icons, parallel to UiIconRegistry (which
// holds texture-atlas sprites). Each icon is a cached UiGeometryBlock baked at a
// target pixel size; widgets draw one via UiDrawList::addVectorIcon(name, dst).
//
// Unlike the texture path, vector icons need no GPU upload — the geometry streams
// through the normal SolidColor draw path.
namespace odai::ui {

struct VectorIcon {
    UiGeometryBlock geometry;  // Triangles in a [0..sizePx] box, origin top-left.
    float sizePx = 0.0f;       // The square box size the icon was baked into.
    UiRect bounds{};           // Tight bounds of the geometry within that box.
};

class VectorIconRegistry {
public:
    // Tessellate an SVG file at `sizePx` and register it under `name`. If a
    // `.odaivec` cache sits next to the SVG and is newer, it is loaded instead of
    // re-parsing. Returns false on parse/IO failure.
    bool registerFromFile(std::string_view name, const std::filesystem::path& svgPath,
                          float sizePx, float dpiScale = 1.0f);

    // Register an already-baked geometry block (from the offline bundler or a
    // cache). Takes ownership by copy.
    void registerBaked(std::string_view name, const UiGeometryBlock& block, float sizePx);

    [[nodiscard]] const VectorIcon* resolve(std::string_view name) const;

    void clear() { m_icons.clear(); }

    // Global singleton — populated at app/theme startup.
    static VectorIconRegistry& global();

private:
    std::unordered_map<std::string, VectorIcon> m_icons;
};

}  // namespace odai::ui
