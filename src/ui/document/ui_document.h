#pragma once

#include "ui/document/ui_binding.h"
#include "ui/theme/ui_theme.h"
#include "ui/widget.h"

#include <nlohmann/json.hpp>

#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

namespace odai::ui {

// Loads a .ui.json file and instantiates a live Widget tree.
//
// JSON schema excerpt:
//   {
//     "version": 1,
//     "type": "Panel",
//     "id": "city_panel",
//     "x": 0, "y": 0, "width": 420, "height": 600,
//     "background": "#2C1F0E",
//     "frame": "Panel.Default",
//     "children": [
//       { "type": "Label", "text": "{city.name}", "style": "heading",
//         "x": 12, "y": 12, "width": 396, "height": 30 },
//       { "type": "HorizontalStack", "gap": 8, "y": 50, "height": 24,
//         "children": [
//           { "type": "Label", "text": "[icon=food] {city.food:+0}" }
//         ]
//       }
//     ]
//   }
//
// Binding expressions in "text" fields are resolved via the supplied BindingContext.
// All coordinate fields support both absolute px numbers and "100%" relative strings
// (resolved against the parent rect supplied at instantiation time).
class UiDocumentLoader {
public:
    explicit UiDocumentLoader(const UiTheme& theme);

    // Register a custom widget factory for a type name. Called before load().
    using Factory = std::function<std::unique_ptr<Widget>(const nlohmann::json&,
                                                          const UiDocumentLoader&,
                                                          const BindingContext&)>;
    void registerType(std::string typeName, Factory factory);

    // Load a .ui.json file. Returns nullptr on parse or IO error.
    std::unique_ptr<Widget> load(const std::filesystem::path& path,
                                  const BindingContext& ctx = {}) const;

    // Instantiate from an already-parsed JSON node and a parent rect for
    // percentage layout resolution.
    std::unique_ptr<Widget> instantiate(const nlohmann::json& node,
                                         const BindingContext& ctx,
                                         const UiRect& parentRect) const;

    // Expose the theme for use in custom factories.
    const UiTheme& theme() const { return m_theme; }

private:
    const UiTheme& m_theme;
    std::unordered_map<std::string, Factory> m_factories;

    void registerBuiltins();

    // Resolve a coordinate value that may be a number or a "50%" string.
    static float resolveLength(const nlohmann::json& val, float parentLength);

    // Parse #RRGGBB / #RRGGBBAA color string.
    static UiColor parseColor(const std::string& hex, const UiColor& fallback);

    // Lay out the rect for a node given its parent rect.
    UiRect resolveRect(const nlohmann::json& node, const UiRect& parentRect) const;
};

}  // namespace odai::ui
