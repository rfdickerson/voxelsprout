#include "procgen/props.h"

#include <array>
#include <cmath>
#include <vector>

#include "procgen/primitives.h"

namespace odai::procgen {

namespace {

// Same LCG as building_generator.cc; props stay deterministic per seed.
struct Rng {
    std::uint32_t state;

    explicit Rng(std::uint32_t seed) : state(seed ? seed : 1u) {}

    std::uint32_t next() {
        state = state * 1664525u + 1013904223u;
        return state >> 8;
    }

    float uniform(float lo, float hi) {
        return lo + (hi - lo) * (static_cast<float>(next() & 0xffffu) / 65535.0f);
    }

    bool chance(float p) { return uniform(0.0f, 1.0f) < p; }
};

const Color3 kTrunkBrown = fromRgbHex(0x5C4530);
const Color3 kBareBranch = fromRgbHex(0x4A3A2E);
const Color3 kSnow = fromRgbHex(0xE8EEF2);

// Broadleaf foliage per season.
const Color3 kLeafSpring = fromRgbHex(0x6FB84A);
const Color3 kLeafSpringLight = fromRgbHex(0x93C95E);
const Color3 kBlossom = fromRgbHex(0xD8A0C0);
const Color3 kLeafDark = fromRgbHex(0x2A6F2E);
const Color3 kLeafMid = fromRgbHex(0x3C8A3C);
const Color3 kLeafLight = fromRgbHex(0x55A048);
const Color3 kLeafAutumnA = fromRgbHex(0xC87828);
const Color3 kLeafAutumnB = fromRgbHex(0xB0501E);
const Color3 kLeafAutumnLight = fromRgbHex(0xE0A83A);
// Conifers stay green; autumn shifts olive, winter dusts them with snow.
const Color3 kConifer = fromRgbHex(0x1F5C33);
const Color3 kConiferLight = fromRgbHex(0x2E7042);
const Color3 kConiferAutumn = fromRgbHex(0x3E5C2A);

std::vector<std::array<float, 2>> hexFootprint(float radius) {
    std::vector<std::array<float, 2>> points;
    points.reserve(6);
    for (int i = 0; i < 6; ++i) {
        const float angle = 2.0f * odai::math::kPi * static_cast<float>(i) / 6.0f;
        points.push_back({radius * std::cos(angle), -radius * std::sin(angle)});
    }
    return points;
}

}  // namespace

TriMesh generateTree(std::uint32_t variant, std::uint32_t seed, Season season) {
    Rng rng(seed ^ (variant + 1u) * 0x9E3779B9u);
    const float scale = rng.uniform(0.85f, 1.2f);
    CsgMesh tree;

    if ((variant & 1u) == 0u) {
        // Broadleaf: short trunk, two bulging canopy tiers.
        const float trunkH = 0.09f * scale;
        const float r0 = 0.075f * scale * rng.uniform(0.9f, 1.15f);

        if (season == Season::Winter) {
            // Bare: a taller trunk with a broom of thin risers where the
            // canopy used to be — reads as leafless branches at map scale.
            tree = makeCylinder({0.0f, 0.0f, 0.0f}, 0.020f * scale, trunkH + 0.06f * scale, 5,
                                kTrunkBrown);
            for (int i = 0; i < 3; ++i) {
                const float angle = 2.0f * odai::math::kPi * static_cast<float>(i) / 3.0f +
                                    rng.uniform(0.0f, 0.8f);
                const float bx = 0.030f * scale * std::cos(angle);
                const float bz = -0.030f * scale * std::sin(angle);
                const float h = (0.14f + rng.uniform(0.0f, 0.06f)) * scale;
                merge(tree, makeBox({bx - 0.006f, trunkH * 0.6f, bz - 0.006f},
                                    {bx + 0.006f, trunkH * 0.6f + h, bz + 0.006f}, kBareBranch));
            }
            return triangulate(tree);
        }

        Color3 leaf;
        Color3 topLeaf;
        switch (season) {
            case Season::Spring:
                leaf = rng.chance(0.25f) ? kBlossom : mix(kLeafSpring, kLeafSpringLight, rng.uniform(0.0f, 1.0f));
                topLeaf = mix(leaf, kLeafSpringLight, 0.4f);
                break;
            case Season::Autumn:
                leaf = mix(kLeafAutumnA, kLeafAutumnB, rng.uniform(0.0f, 1.0f));
                topLeaf = mix(leaf, kLeafAutumnLight, 0.5f);
                break;
            default:
                leaf = mix(kLeafDark, kLeafMid, rng.uniform(0.0f, 1.0f));
                topLeaf = mix(leaf, kLeafLight, 0.4f);
                break;
        }
        tree = makeCylinder({0.0f, 0.0f, 0.0f}, 0.020f * scale, trunkH + 0.02f, 5, kTrunkBrown);
        // Lower tier widens upward, upper tier tapers to a rounded top.
        merge(tree, makeConvexPrism(hexFootprint(r0 * 0.72f), trunkH, trunkH + 0.10f * scale, leaf,
                                    1.45f));
        merge(tree, makeConvexPrism(hexFootprint(r0), trunkH + 0.10f * scale,
                                    trunkH + 0.20f * scale, topLeaf, 0.35f));
    } else {
        // Conifer: taller trunk, stacked tapering tiers to a point.
        const float trunkH = 0.06f * scale;
        const float r0 = 0.062f * scale * rng.uniform(0.9f, 1.1f);
        tree = makeCylinder({0.0f, 0.0f, 0.0f}, 0.016f * scale, trunkH + 0.02f, 5, kTrunkBrown);
        Color3 leaf = mix(kConifer, kConiferLight, rng.uniform(0.0f, 1.0f));
        if (season == Season::Autumn) {
            leaf = mix(leaf, kConiferAutumn, 0.5f);
        }
        float y = trunkH;
        float r = r0;
        for (int tier = 0; tier < 3; ++tier) {
            const float tierH = (0.085f - 0.012f * static_cast<float>(tier)) * scale;
            Color3 tierColor = tier == 2 ? mix(leaf, kConiferLight, 0.35f) : leaf;
            if (season == Season::Winter) {
                // Snow load sits heavier on the upper boughs.
                tierColor = mix(tierColor, kSnow, 0.30f + 0.18f * static_cast<float>(tier));
            }
            merge(tree, makeConvexPrism(hexFootprint(r), y, y + tierH, tierColor, 0.25f));
            y += tierH * 0.72f;  // tiers overlap like layered boughs
            r *= 0.74f;
        }
    }
    return triangulate(tree);
}

TriMesh generateVehicle(std::uint32_t seed) {
    Rng rng(seed);
    static const Color3 kBodyColors[] = {
        fromRgbHex(0xB03A30),  // red
        fromRgbHex(0x3660A8),  // blue
        fromRgbHex(0xD8D8D2),  // white
        fromRgbHex(0x8E9498),  // silver
        fromRgbHex(0xE0B818),  // taxi yellow
        fromRgbHex(0x3E7048),  // green
        fromRgbHex(0x2E3238),  // black
    };
    const Color3 body = kBodyColors[rng.next() % (sizeof(kBodyColors) / sizeof(kBodyColors[0]))];
    const Color3 glass = fromRgbHex(0x9FC4D8);
    const Color3 tire = fromRgbHex(0x1C1E22);

    const float halfL = rng.uniform(0.070f, 0.082f);
    const float halfW = 0.040f;

    // Body slab floats just above the tire strips; cabin sits back of center.
    CsgMesh car = makeBox({-halfL, 0.016f, -halfW}, {halfL, 0.055f, halfW}, body);
    merge(car, makeBox({-halfL * 0.55f, 0.055f, -halfW * 0.8f},
                       {halfL * 0.35f, 0.085f, halfW * 0.8f}, glass));
    // Left/right tire strips read as wheels at isometric zoom.
    merge(car, makeBox({-halfL * 0.75f, 0.0f, -halfW}, {-halfL * 0.35f, 0.022f, halfW}, tire));
    merge(car, makeBox({halfL * 0.35f, 0.0f, -halfW}, {halfL * 0.75f, 0.022f, halfW}, tire));
    return triangulate(car);
}

TriMesh generatePumpkin(std::uint32_t seed) {
    Rng rng(seed);
    const Color3 rind = mix(fromRgbHex(0xD87018), fromRgbHex(0xC05A10), rng.uniform(0.0f, 1.0f));
    const Color3 stem = fromRgbHex(0x5C6B2E);
    const float r = 0.026f + rng.uniform(0.0f, 0.010f);
    const float h = r * 1.35f;

    // Squat octagonal drum, slightly narrower at the top, plus a stubby stem.
    std::vector<std::array<float, 2>> octagon;
    for (int i = 0; i < 8; ++i) {
        const float angle = 2.0f * odai::math::kPi * static_cast<float>(i) / 8.0f;
        octagon.push_back({r * std::cos(angle), -r * std::sin(angle)});
    }
    CsgMesh pumpkin = makeConvexPrism(octagon, 0.0f, h, rind, 0.72f);
    merge(pumpkin, makeCylinder({0.0f, h - 0.004f, 0.0f}, 0.006f, 0.018f, 4, stem));
    return triangulate(pumpkin);
}

TriMesh generatePowerPole(std::uint32_t seed) {
    Rng rng(seed);
    const Color3 wood = mix(fromRgbHex(0x5C4A38), fromRgbHex(0x6B5642), rng.uniform(0.0f, 1.0f));
    const Color3 insulator = fromRgbHex(0xC9C0A8);
    const float postH = 0.29f + rng.uniform(0.0f, 0.05f);

    CsgMesh pole = makeCylinder({0.0f, 0.0f, 0.0f}, 0.0105f, postH, 6, wood);
    // Cross-arm near the top, along local X, plus three insulator caps riding it.
    const float armY = postH - 0.045f;
    merge(pole, makeBox({-0.055f, armY, -0.009f}, {0.055f, armY + 0.011f, 0.009f}, wood));
    for (const float sx : {-0.042f, 0.0f, 0.042f}) {
        merge(pole, makeCylinder({sx, armY + 0.011f, 0.0f}, 0.0075f, 0.016f, 5, insulator));
    }
    return triangulate(pole);
}

TriMesh generateStreetlamp(std::uint32_t seed) {
    Rng rng(seed);
    const Color3 post = fromRgbHex(0x2A2C30);
    const Color3 glow = mix(fromRgbHex(0xF0D080), fromRgbHex(0xF5C860), rng.uniform(0.0f, 1.0f));
    const float postH = 0.19f + rng.uniform(0.0f, 0.04f);

    CsgMesh lamp = makeCylinder({0.0f, 0.0f, 0.0f}, 0.0075f, postH, 6, post);
    // Small octagonal lamp head, tapered top and bottom, sitting on the post.
    std::vector<std::array<float, 2>> octagon;
    const float r = 0.016f;
    for (int i = 0; i < 8; ++i) {
        const float angle = 2.0f * odai::math::kPi * static_cast<float>(i) / 8.0f;
        octagon.push_back({r * std::cos(angle), -r * std::sin(angle)});
    }
    CsgMesh head = makeConvexPrism(octagon, postH, postH + 0.030f, glow, 0.55f);
    merge(lamp, head);
    // Thin dark cap pinning the head to the post.
    merge(lamp, makeCylinder({0.0f, postH + 0.028f, 0.0f}, 0.008f, 0.010f, 6, post));
    return triangulate(lamp);
}

}  // namespace odai::procgen
