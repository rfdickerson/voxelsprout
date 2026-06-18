#include "ui/document/ui_document.h"

#include "ui/widgets/image.h"
#include "ui/widgets/label.h"
#include "ui/widgets/panel.h"
#include "ui/widgets/progress_bar.h"
#include "ui/widgets/repeater.h"
#include "ui/widgets/scroll_view.h"
#include "ui/widgets/spacer.h"
#include "ui/widgets/stack_layout.h"
#include "ui/widgets/window.h"

#include <nlohmann/json.hpp>

#include <charconv>
#include <fstream>
#include <stdexcept>

namespace odai::ui {

using json = nlohmann::json;

// --------------------------------------------------------------------------
// Helpers
// --------------------------------------------------------------------------

static UiColor parseHexColor(const std::string& hex, const UiColor& fallback) {
    const char* s = hex.c_str();
    if (*s == '#') ++s;
    const std::size_t len = hex.size() - (*hex.c_str() == '#' ? 1 : 0);
    if (len != 6 && len != 8) return fallback;

    auto hexByte = [&](const char* p) -> float {
        unsigned char hi = (p[0] >= 'a') ? (p[0] - 'a' + 10) : (p[0] >= 'A') ? (p[0] - 'A' + 10) : (p[0] - '0');
        unsigned char lo = (p[1] >= 'a') ? (p[1] - 'a' + 10) : (p[1] >= 'A') ? (p[1] - 'A' + 10) : (p[1] - '0');
        return static_cast<float>((hi << 4) | lo) / 255.0f;
    };

    const float r = hexByte(s + 0);
    const float g = hexByte(s + 2);
    const float b = hexByte(s + 4);
    const float a = (len == 8) ? hexByte(s + 6) : 1.0f;
    return UiColor{r, g, b, a};
}

// --------------------------------------------------------------------------
// UiDocumentLoader
// --------------------------------------------------------------------------

UiDocumentLoader::UiDocumentLoader(const UiTheme& theme) : m_theme(theme) {
    registerBuiltins();
}

void UiDocumentLoader::registerType(std::string typeName, Factory factory) {
    m_factories[std::move(typeName)] = std::move(factory);
}

// --------------------------------------------------------------------------
// Length resolution
// --------------------------------------------------------------------------

float UiDocumentLoader::resolveLength(const json& val, float parentLength) {
    if (val.is_number()) return val.get<float>();
    if (val.is_string()) {
        const std::string s = val.get<std::string>();
        if (!s.empty() && s.back() == '%') {
            const float pct = std::stof(s.substr(0, s.size() - 1));
            return parentLength * pct / 100.0f;
        }
        return std::stof(s);
    }
    return 0.0f;
}

UiRect UiDocumentLoader::resolveRect(const json& node, const UiRect& parent) const {
    const float pw = parent.width();
    const float ph = parent.height();

    float x = node.contains("x") ? resolveLength(node["x"], pw) : 0.0f;
    float y = node.contains("y") ? resolveLength(node["y"], ph) : 0.0f;
    float w = node.contains("width")  ? resolveLength(node["width"],  pw) : pw;
    float h = node.contains("height") ? resolveLength(node["height"], ph) : ph;

    return UiRect::fromXYWH(parent.minX + x, parent.minY + y, w, h);
}

UiColor UiDocumentLoader::parseColor(const std::string& hex, const UiColor& fallback) {
    return parseHexColor(hex, fallback);
}

// --------------------------------------------------------------------------
// Main entry points
// --------------------------------------------------------------------------

std::unique_ptr<Widget> UiDocumentLoader::load(const std::filesystem::path& path,
                                                 const BindingContext& ctx) const {
    std::ifstream f(path);
    if (!f) return nullptr;
    json doc;
    try {
        f >> doc;
    } catch (...) {
        return nullptr;
    }
    // Use screen-sized parent rect as default.
    const UiRect fullscreen{0.0f, 0.0f, 1920.0f, 1080.0f};
    return instantiate(doc, ctx, fullscreen);
}

std::unique_ptr<Widget> UiDocumentLoader::instantiate(const json& node,
                                                        const BindingContext& ctx,
                                                        const UiRect& parentRect) const {
    const std::string type = node.value("type", "Panel");
    const UiRect rect = resolveRect(node, parentRect);

    // Custom factory?
    auto it = m_factories.find(type);
    if (it != m_factories.end()) {
        auto w = it->second(node, *this, ctx);
        if (w) w->setRect(rect);
        return w;
    }
    // Unknown type → empty widget.
    auto fallback = std::make_unique<Widget>();
    fallback->setRect(rect);
    return fallback;
}

// --------------------------------------------------------------------------
// Built-in widget factories
// --------------------------------------------------------------------------

static void buildChildren(const json& node, Widget& parent,
                           const UiDocumentLoader& loader,
                           const BindingContext& ctx) {
    if (!node.contains("children")) return;
    for (const json& child : node["children"]) {
        auto w = loader.instantiate(child, ctx, parent.rect());
        if (w) parent.addChild(std::move(w));
    }
}

void UiDocumentLoader::registerBuiltins() {
    // ---- Panel ----
    registerType("Panel", [](const json& n, const UiDocumentLoader& L, const BindingContext& ctx) {
        auto w = std::make_unique<Panel>();
        if (n.contains("background")) {
            const UiColor def = w->background;
            const std::string s = n["background"].get<std::string>();
            w->background = s.starts_with('#') ? parseColor(s, def) : L.theme().color(s);
        }
        if (n.contains("borderColor")) {
            const UiColor def = w->borderColor;
            w->borderColor = parseColor(n["borderColor"].get<std::string>(), def);
        }
        if (n.contains("frame")) {
            w->nineSlice = L.theme().frame(n["frame"].get<std::string>());
        }
        buildChildren(n, *w, L, ctx);
        return w;
    });

    // ---- Label ----
    registerType("Label", [](const json& n, const UiDocumentLoader& L, const BindingContext& ctx) {
        const FontSet fonts = L.theme().bodyFontSet();
        std::string markup = n.value("text", "");
        // Resolve binding expressions.
        if (!ctx.empty()) {
            std::string out;
            out.reserve(markup.size());
            std::size_t pos = 0;
            while (pos < markup.size()) {
                const std::size_t open = markup.find('{', pos);
                if (open == std::string::npos) { out += markup.substr(pos); break; }
                out += markup.substr(pos, open - pos);
                const std::size_t close = markup.find('}', open);
                if (close == std::string::npos) { out += markup.substr(open); break; }
                const std::string expr = markup.substr(open, close - open + 1);
                out += ctx.resolve(expr);
                pos = close + 1;
            }
            markup = std::move(out);
        }
        auto w = std::make_unique<Label>(fonts, markup);
        if (n.contains("color")) {
            w->color = parseColor(n["color"].get<std::string>(), w->color);
        }
        if (n.contains("style")) {
            const std::string style = n["style"].get<std::string>();
            if (const Font* f = L.theme().font(style)) {
                // A single named font.
                w = std::make_unique<Label>(f, markup);
                if (n.contains("color")) {
                    w->color = parseColor(n["color"].get<std::string>(), w->color);
                }
            }
        }
        return w;
    });

    // ---- HorizontalStack ----
    registerType("HorizontalStack", [](const json& n, const UiDocumentLoader& L, const BindingContext& ctx) {
        auto w = std::make_unique<HorizontalStack>();
        w->gap = n.value("gap", w->gap);
        buildChildren(n, *w, L, ctx);
        return w;
    });

    // ---- VerticalStack ----
    registerType("VerticalStack", [](const json& n, const UiDocumentLoader& L, const BindingContext& ctx) {
        auto w = std::make_unique<VerticalStack>();
        w->gap = n.value("gap", w->gap);
        buildChildren(n, *w, L, ctx);
        return w;
    });

    // ---- ScrollView ----
    registerType("ScrollView", [](const json& n, const UiDocumentLoader& L, const BindingContext& ctx) {
        auto w = std::make_unique<ScrollView>();
        w->childGap = n.value("childGap", w->childGap);
        buildChildren(n, *w, L, ctx);
        return w;
    });

    // ---- Spacer ----
    registerType("Spacer", [](const json& n, const UiDocumentLoader&, const BindingContext&) {
        auto w = std::make_unique<Spacer>();
        return w;
    });

    // ---- Image ----
    registerType("Image", [](const json& n, const UiDocumentLoader&, const BindingContext&) {
        auto w = std::make_unique<Image>();
        if (n.contains("tint")) {
            w->tint = parseColor(n["tint"].get<std::string>(), w->tint);
        }
        return w;
    });

    // ---- ProgressBar ----
    registerType("ProgressBar", [](const json& n, const UiDocumentLoader& L, const BindingContext& ctx) {
        auto w = std::make_unique<ProgressBar>();
        float v = n.value("value", 0.0f);
        if (n.contains("value") && n["value"].is_string()) {
            v = ctx.resolveFloat(n["value"].get<std::string>());
        }
        w->value = v;
        if (n.contains("foreground")) w->foreground = parseColor(n["foreground"].get<std::string>(), w->foreground);
        if (n.contains("background")) w->background = parseColor(n["background"].get<std::string>(), w->background);
        if (n.contains("frame"))      w->frame = L.theme().frame(n["frame"].get<std::string>());
        return w;
    });

    // ---- Repeater ----
    // A Repeater with a static template child. In a real app the factory and items
    // are set from C++ after load(); this only wires up the JSON template.
    registerType("Repeater", [](const json& n, const UiDocumentLoader& L, const BindingContext& ctx) {
        auto w = std::make_unique<Repeater>();
        w->itemHeight = n.value("itemHeight", w->itemHeight);
        w->itemGap    = n.value("itemGap",    w->itemGap);
        // The "template" field names a child template; items are injected later.
        return w;
    });
}

}  // namespace odai::ui
