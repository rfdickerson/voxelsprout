#include "tools/ui_editor/editor_color.h"

#include <algorithm>
#include <cmath>

namespace odai::tools::ui_editor {

static float clamp01(float v) { return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v); }

float wrapHue(float degrees) {
    float h = std::fmod(degrees, 360.0f);
    if (h < 0.0f) h += 360.0f;
    return h;
}

// ─── HSV ─────────────────────────────────────────────────────────────────────

Hsv rgbToHsv(const ui::UiColor& c) {
    const float r = clamp01(c.r);
    const float g = clamp01(c.g);
    const float b = clamp01(c.b);

    const float maxC = std::max({r, g, b});
    const float minC = std::min({r, g, b});
    const float delta = maxC - minC;

    Hsv out;
    out.v = maxC;
    out.s = (maxC <= 0.0f) ? 0.0f : (delta / maxC);

    if (delta <= 0.0f) {
        out.h = 0.0f;  // achromatic: hue is undefined, report 0
        return out;
    }
    if (maxC == r) {
        out.h = 60.0f * std::fmod((g - b) / delta, 6.0f);
    } else if (maxC == g) {
        out.h = 60.0f * (((b - r) / delta) + 2.0f);
    } else {
        out.h = 60.0f * (((r - g) / delta) + 4.0f);
    }
    out.h = wrapHue(out.h);
    return out;
}

ui::UiColor hsvToRgb(const Hsv& hsv, float alpha) {
    const float h = wrapHue(hsv.h);
    const float s = clamp01(hsv.s);
    const float v = clamp01(hsv.v);

    const float c = v * s;
    const float x = c * (1.0f - std::fabs(std::fmod(h / 60.0f, 2.0f) - 1.0f));
    const float m = v - c;

    float r = 0.0f, g = 0.0f, b = 0.0f;
    if (h < 60.0f) {
        r = c; g = x;
    } else if (h < 120.0f) {
        r = x; g = c;
    } else if (h < 180.0f) {
        g = c; b = x;
    } else if (h < 240.0f) {
        g = x; b = c;
    } else if (h < 300.0f) {
        r = x; b = c;
    } else {
        r = c; b = x;
    }
    return ui::UiColor{r + m, g + m, b + m, clamp01(alpha)};
}

// ─── Harmony ─────────────────────────────────────────────────────────────────

const std::vector<Harmony>& allHarmonies() {
    static const std::vector<Harmony> kAll = {
        Harmony::Complementary, Harmony::SplitComplementary,
        Harmony::Analogous,     Harmony::Triad,
        Harmony::Tetrad,        Harmony::Square,
        Harmony::Monochromatic, Harmony::Shades,
    };
    return kAll;
}

std::string_view harmonyName(Harmony harmony) {
    switch (harmony) {
        case Harmony::Complementary:      return "Complement";
        case Harmony::SplitComplementary: return "Split";
        case Harmony::Analogous:          return "Analogous";
        case Harmony::Triad:              return "Triad";
        case Harmony::Tetrad:             return "Tetrad";
        case Harmony::Square:             return "Square";
        case Harmony::Monochromatic:      return "Mono";
        case Harmony::Shades:             return "Shades";
    }
    return "?";
}

std::vector<ui::UiColor> harmonyColors(const ui::UiColor& base, Harmony harmony) {
    const Hsv hsv = rgbToHsv(base);
    const float a = base.a;

    // Rotate the hue, keeping saturation/value/alpha — this is what makes the
    // partners read as belonging to the same palette as the base.
    auto rotated = [&](float degrees) {
        return hsvToRgb(Hsv{wrapHue(hsv.h + degrees), hsv.s, hsv.v}, a);
    };

    std::vector<ui::UiColor> out;
    switch (harmony) {
        case Harmony::Complementary:
            out = {base, rotated(180.0f)};
            break;
        case Harmony::SplitComplementary:
            out = {base, rotated(150.0f), rotated(210.0f)};
            break;
        case Harmony::Analogous:
            out = {base, rotated(-30.0f), rotated(30.0f)};
            break;
        case Harmony::Triad:
            out = {base, rotated(120.0f), rotated(240.0f)};
            break;
        case Harmony::Tetrad:
            out = {base, rotated(60.0f), rotated(180.0f), rotated(240.0f)};
            break;
        case Harmony::Square:
            out = {base, rotated(90.0f), rotated(180.0f), rotated(270.0f)};
            break;
        case Harmony::Monochromatic: {
            // A tint/shade ramp: five evenly spaced values at the base's hue and
            // saturation. The base is returned first so index 0 stays the input,
            // then the ramp runs dark to light.
            out.push_back(base);
            for (int i = 0; i < 5; ++i) {
                const float v = 0.18f + 0.19f * static_cast<float>(i);
                out.push_back(hsvToRgb(Hsv{hsv.h, hsv.s, v}, a));
            }
            break;
        }
        case Harmony::Shades: {
            // Same hue and value, saturation swept from grey to full — useful
            // for picking a muted variant of an accent.
            out.push_back(base);
            for (int i = 0; i < 5; ++i) {
                const float s = 0.05f + 0.2375f * static_cast<float>(i);
                out.push_back(hsvToRgb(Hsv{hsv.h, s, hsv.v}, a));
            }
            break;
        }
    }
    return out;
}

// ─── Contrast ────────────────────────────────────────────────────────────────

float srgbToLinearChannel(float c) {
    const float v = clamp01(c);
    return v <= 0.04045f ? (v / 12.92f) : std::pow((v + 0.055f) / 1.055f, 2.4f);
}

float relativeLuminance(const ui::UiColor& c) {
    return 0.2126f * srgbToLinearChannel(c.r) + 0.7152f * srgbToLinearChannel(c.g) +
           0.0722f * srgbToLinearChannel(c.b);
}

float contrastRatio(const ui::UiColor& a, const ui::UiColor& b) {
    const float la = relativeLuminance(a);
    const float lb = relativeLuminance(b);
    const float lighter = std::max(la, lb);
    const float darker = std::min(la, lb);
    return (lighter + 0.05f) / (darker + 0.05f);
}

ui::UiColor compositeOver(const ui::UiColor& src, const ui::UiColor& dst) {
    const float sa = clamp01(src.a);
    const float inv = 1.0f - sa;
    return ui::UiColor{
        src.r * sa + dst.r * inv,
        src.g * sa + dst.g * inv,
        src.b * sa + dst.b * inv,
        sa + dst.a * inv,
    };
}

std::string_view contrastRating(float ratio) {
    if (ratio >= 7.0f) return "AAA";
    if (ratio >= 4.5f) return "AA";
    if (ratio >= 3.0f) return "AA Large";
    return "fail";
}

}  // namespace odai::tools::ui_editor
