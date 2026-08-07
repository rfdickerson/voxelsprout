#include "games/citybuilder/citybuilder_fields.h"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace odai::games::citybuilder {

const DataLayerDesc& dataLayerDesc(DataLayer layer) {
    // Indexed by DataLayer, so the enum order and this table must stay in step.
    // Ramp derivation and the L* / colour-vision audit live on DataLayerDesc in
    // the header. In short: five classes, ~17 L* apart, one hue family each.
    static constexpr DataLayerDesc kDescs[static_cast<std::size_t>(DataLayer::Count)] = {
        {"Off", "", "", {}},
        // Plum → magenta → ember → cream. A "magma" family, chosen because it is
        // maximally distant from the unwashed map's grass green and water blue,
        // so no land-value class can be mistaken for terrain.
        {"Land Value", "poor", "prime",
         {0x2C2542, 0x6E3170, 0xB44C60, 0xE6884E, 0xF5D69C}},
        // Soot → acid. Chartreuse is the smog convention, and it keeps the
        // hazard layer clear of the fire/alert reds already in the HUD.
        {"Pollution", "clean", "choked",
         {0x232A24, 0x45542A, 0x76832B, 0xACB13A, 0xE2E078}},
        // Amber, matching the school roof (0xE0852E).
        {"Education", "none", "served",
         {0x2E2418, 0x6A4318, 0xA96A1C, 0xDCA03A, 0xF5DBA2}},
        // Teal, matching the clinic roof (0x21A89A).
        {"Health", "none", "served",
         {0x172A2C, 0x1C5257, 0x21847E, 0x43B8A4, 0xA6E6D4}},
        // Violet rather than the police blue (0x2F6BD6) it would otherwise
        // borrow: a mid-blue class is confusable with the water body colour
        // (0x2D5C8C) at a glance, and violet is not.
        {"Safety", "exposed", "covered",
         {0x291F3E, 0x53407E, 0x8873BA, 0xB3A2DC, 0xDDD0F7}},
        // Slate → hot coral. The player already had a per-tile congestion field
        // they could neither see nor causally affect; with distance-weighted
        // jobs feeding it, this layer is what closes the hypothesis-act-answer
        // loop — "that arterial is the problem", build a bypass, watch the red
        // move. Warm/neutral rather than a hue family, so it reads as a
        // *flow* measure and not another coverage ring.
        {"Traffic", "clear", "jammed",
         {0x24282E, 0x4C4550, 0x8A5566, 0xC9705F, 0xF2B48A}},
    };
    const auto i = static_cast<std::size_t>(layer);
    return kDescs[i < static_cast<std::size_t>(DataLayer::Count) ? i : 0];
}

void splatDisc(std::span<float> field, int width, int height, int c, int r, int radius,
               float peak) {
    if (radius <= 0) return;
    const int c0 = std::max(0, c - radius), c1 = std::min(width - 1, c + radius);
    const int r0 = std::max(0, r - radius), r1 = std::min(height - 1, r + radius);
    const float inv = 1.0f / static_cast<float>(radius);
    for (int rr = r0; rr <= r1; ++rr) {
        const int dr = rr - r;
        for (int cc = c0; cc <= c1; ++cc) {
            const int dc = cc - c;
            const float d = std::sqrt(static_cast<float>(dc * dc + dr * dr));
            if (d > static_cast<float>(radius)) continue;  // disc, not square
            field[static_cast<std::size_t>(rr) * static_cast<std::size_t>(width) +
                  static_cast<std::size_t>(cc)] += peak * (1.0f - d * inv);
        }
    }
}

float populationWeightedMean(std::span<const float> field, std::span<const float> weight) {
    const std::size_t n = std::min(field.size(), weight.size());
    float num = 0.0f, den = 0.0f;
    for (std::size_t i = 0; i < n; ++i) {
        num += field[i] * weight[i];
        den += weight[i];
    }
    return den > 0.0f ? num / den : 0.0f;
}

}  // namespace odai::games::citybuilder
