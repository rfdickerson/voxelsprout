#pragma once

// Sequential colour scales for data overlays — heatmaps, choropleths, any map
// that washes geometry by a scalar field.
//
// This exists because the obvious hand-rolled version is wrong in a specific,
// repeatable way. The natural first instinct is a red -> amber -> green lerp,
// which is a *diverging* scale: its lightness peaks in the middle, so a 0.5
// tile reads brighter — and therefore higher — than a 1.0 tile, and the ranking
// the whole overlay exists to communicate is destroyed. Under the two common
// forms of colour-vision deficiency the red and green ends then collapse onto
// nearly the same olive, leaving ~8% of male viewers with no signal at all.
//
// A sequential field wants a sequential scale: monotonic in perceived
// lightness, one hue family, so the ranking survives greyscale, deuteranopia
// and protanopia alike. Every scale below is verified monotonic in CIE L* by
// odai_ui_tests, with a minimum step far above the just-noticeable difference.
// That is the property worth pinning — a scale that fails it is broken no
// matter how pleasant the colours are.
//
// Vulkan-free, header-only, no dependencies beyond ui_types.

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>

#include "ui/ui_types.h"

namespace odai::ui {

// Relative luminance per WCAG 2.x (sRGB -> linear, Rec.709 weights). The
// monotonicity check a scale has to pass, and the same quantity WCAG contrast
// ratios are built from.
constexpr float srgbToLinear(float c) {
    return c <= 0.04045f ? c / 12.92f
                         : ((c + 0.055f) / 1.055f) * ((c + 0.055f) / 1.055f) *
                               ((c + 0.055f) / 1.055f);
}

inline float relativeLuminance(std::uint32_t rgbHex) {
    const float r = static_cast<float>((rgbHex >> 16) & 0xFFu) / 255.0f;
    const float g = static_cast<float>((rgbHex >> 8) & 0xFFu) / 255.0f;
    const float b = static_cast<float>(rgbHex & 0xFFu) / 255.0f;
    return 0.2126f * srgbToLinear(r) + 0.7152f * srgbToLinear(g) + 0.0722f * srgbToLinear(b);
}

// CIE L*, the *perceptual* lightness axis (0 black .. 100 white). Use this and
// not relativeLuminance() when the question is "can a viewer tell these two
// swatches apart": linear luminance is heavily compressed at the dark end, so
// two clearly-distinct dark stops differ by a hair in Y while differing by ~15
// in L*. The just-noticeable difference for a large flat field is ~2-3 L*.
inline float lstar(std::uint32_t rgbHex) {
    const float y = relativeLuminance(rgbHex);
    return y > 0.008856f ? 116.0f * std::cbrt(y) - 16.0f : y * 903.3f;
}

// A scale is just its ordered stops, darkest first. Borrowed, never owned —
// the canonical scales below are constexpr arrays with static storage.
struct ColorScale {
    std::span<const std::uint32_t> stops;

    [[nodiscard]] bool empty() const { return stops.empty(); }

    // Banded lookup: which class does t fall in? This is what a tile wash
    // should use. A continuous gradient over a tile grid is worse than
    // useless — 0.62 and 0.71 are indistinguishable side by side — whereas
    // discrete classes turn a linear falloff into visible contour rings, so
    // the viewer can see exactly where a field stops. Five to seven classes is
    // the cartographic default.
    [[nodiscard]] std::size_t classify(float t) const {
        if (stops.empty()) return 0;
        const auto n = static_cast<int>(stops.size());
        const int i = static_cast<int>(t * static_cast<float>(n));
        return static_cast<std::size_t>(i < 0 ? 0 : (i >= n ? n - 1 : i));
    }

    [[nodiscard]] std::uint32_t classHex(float t) const {
        return stops.empty() ? 0u : stops[classify(t)];
    }

    // Continuous lookup, for gradients over continuous geometry (a legend bar,
    // a line chart) where banding would be the wrong read. Prefer classify()
    // for anything drawn on a grid.
    [[nodiscard]] UiColor sample(float t) const {
        if (stops.empty()) return UiColor(0, 0, 0, 1);
        if (stops.size() == 1) return UiColor::fromRgbHex(stops[0]);
        const float clamped = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
        const float scaled = clamped * static_cast<float>(stops.size() - 1);
        const auto lo = static_cast<std::size_t>(scaled);
        const std::size_t hi = lo + 1 < stops.size() ? lo + 1 : lo;
        const float f = scaled - static_cast<float>(lo);
        const UiColor a = UiColor::fromRgbHex(stops[lo]);
        const UiColor b = UiColor::fromRgbHex(stops[hi]);
        return UiColor{a.r + (b.r - a.r) * f, a.g + (b.g - a.g) * f, a.b + (b.b - a.b) * f,
                       a.a + (b.a - a.a) * f};
    }
};

// ── Canonical scales ────────────────────────────────────────────────────────
// Five classes each, roughly 17 L* apart — well above the ~2-3 L* just-noticeable
// difference for a large flat field, so every step is unmistakable. All are
// monotonic in luminance under normal, deuteranope and protanope simulation.

namespace scales {

// Perceptually-uniform general purpose. Reach for one of these unless the data
// has a conventional colour (smog is green, heat is red) worth honouring.
inline constexpr std::uint32_t kViridis[] = {0x2C1F4E, 0x35618D, 0x21908C, 0x5EC962, 0xF9E721};
inline constexpr std::uint32_t kMagma[]   = {0x2C2542, 0x6E3170, 0xB44C60, 0xE6884E, 0xF5D69C};

// Single-hue sequentials, for when several layers share a screen and each needs
// its own identity — the hue says *which* field, the lightness says how much.
inline constexpr std::uint32_t kAmber[]   = {0x2E2418, 0x6A4318, 0xA96A1C, 0xDCA03A, 0xF5DBA2};
inline constexpr std::uint32_t kTeal[]    = {0x172A2C, 0x1C5257, 0x21847E, 0x43B8A4, 0xA6E6D4};
inline constexpr std::uint32_t kViolet[]  = {0x291F3E, 0x53407E, 0x8873BA, 0xB3A2DC, 0xDDD0F7};
inline constexpr std::uint32_t kChartreuse[] = {0x232A24, 0x45542A, 0x76832B, 0xACB13A, 0xE2E078};
inline constexpr std::uint32_t kEmber[]   = {0x24282E, 0x4C4550, 0x8A5566, 0xC9705F, 0xF2B48A};

}  // namespace scales

inline constexpr ColorScale kViridis{scales::kViridis};
inline constexpr ColorScale kMagma{scales::kMagma};
inline constexpr ColorScale kAmber{scales::kAmber};
inline constexpr ColorScale kTeal{scales::kTeal};
inline constexpr ColorScale kViolet{scales::kViolet};
inline constexpr ColorScale kChartreuse{scales::kChartreuse};
inline constexpr ColorScale kEmber{scales::kEmber};

}  // namespace odai::ui
