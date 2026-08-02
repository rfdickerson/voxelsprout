#pragma once

#include "ui/ui_types.h"

#include <cstddef>
#include <string_view>
#include <vector>

// Color-theory helpers for the UI editor's inspector: HSV, classical harmony
// schemes, and WCAG contrast scoring.
//
// COLOR SPACE — this is the part that's easy to get wrong here. A UiColor's
// channels are *sRGB-encoded* (gamma space) on the CPU: parseColorString()
// divides the hex bytes by 255 and nothing decodes them, because
// render/shaders/ui.frag.slang linearizes vertex colors on entry (see its
// srgbToLinear). So:
//   - HSV below operates directly on those gamma-space values, which is exactly
//     what a paint program's picker shows and what a designer reasons about.
//   - The luminance/contrast helpers linearize first, because WCAG defines
//     contrast on linear light. Doing HSV on linear values, or contrast on
//     gamma values, would both be wrong — in opposite directions.
namespace odai::tools::ui_editor {

struct Hsv {
    float h = 0.0f;  // hue in degrees, wrapped to [0, 360)
    float s = 0.0f;  // saturation, [0, 1]
    float v = 0.0f;  // value, [0, 1]
};

// Wraps into [0, 360). Negative inputs are handled.
float wrapHue(float degrees);

Hsv rgbToHsv(const ui::UiColor& c);
ui::UiColor hsvToRgb(const Hsv& hsv, float alpha = 1.0f);

// ─── Harmony schemes ─────────────────────────────────────────────────────────

enum class Harmony {
    Complementary,       // the base and the hue opposite it
    SplitComplementary,  // the base and the two hues flanking its opposite
    Analogous,           // the base and its two neighbours, 30 deg either side
    Triad,               // three hues 120 deg apart
    Tetrad,              // rectangle: base, +60, +180, +240
    Square,              // four hues 90 deg apart
    Monochromatic,       // one hue, stepped through value
    Shades,              // one hue, stepped through saturation
};

// All schemes, in the order the inspector lists them.
const std::vector<Harmony>& allHarmonies();

std::string_view harmonyName(Harmony harmony);

// The scheme's colors, base first — index 0 always equals `base` (alpha
// included), so a swatch row can show "what you have" next to "what goes with
// it". Hue-rotating schemes keep the base's saturation, value and alpha; the
// two ramps keep its hue and alpha.
std::vector<ui::UiColor> harmonyColors(const ui::UiColor& base, Harmony harmony);

// ─── Contrast (WCAG 2.1) ─────────────────────────────────────────────────────

// The piecewise sRGB decode, matching ui.frag.slang's srgbToLinear1 exactly.
float srgbToLinearChannel(float c);

// Relative luminance of an opaque color, on linearized sRGB.
float relativeLuminance(const ui::UiColor& c);

// Contrast ratio between two opaque colors, in [1, 21]. Alpha is ignored — a
// translucent foreground's effective contrast depends on what's behind it, so
// composite it first with compositeOver() if that matters.
float contrastRatio(const ui::UiColor& a, const ui::UiColor& b);

// Straight-alpha source-over composite, for scoring a translucent color against
// the surface it sits on.
ui::UiColor compositeOver(const ui::UiColor& src, const ui::UiColor& dst);

// WCAG rating for the ratio, as body text: "AAA" (>=7), "AA" (>=4.5),
// "AA Large" (>=3, large/bold text only), or "fail".
std::string_view contrastRating(float ratio);

}  // namespace odai::tools::ui_editor
