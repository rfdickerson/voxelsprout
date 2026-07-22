#include "procgen/props.h"

#include <array>
#include <cmath>
#include <vector>

#include "procgen/primitives.h"
#include "procgen/rng.h"

namespace odai::procgen {

namespace {

const Color3 kTrunkBrown = fromRgbHex(0x5C4530);
const Color3 kBareBranch = fromRgbHex(0x4A3A2E);
const Color3 kSnow = fromRgbHex(0xE8EEF2);
const Color3 kBirchBark = fromRgbHex(0xDEDCD2);
const Color3 kBirchLeaf = fromRgbHex(0x8FC45E);
const Color3 kBirchAutumn = fromRgbHex(0xE8C63A);
const Color3 kPoplarGreen = fromRgbHex(0x4A7E38);
const Color3 kPoplarAutumn = fromRgbHex(0xD8A832);
const Color3 kWillowGreen = fromRgbHex(0x74A050);
const Color3 kWillowLight = fromRgbHex(0x8CB464);
const Color3 kOakGreen = fromRgbHex(0x3E7A34);
const Color3 kOakAutumn = fromRgbHex(0xA85E28);
const Color3 kShrubGreen = fromRgbHex(0x4E8A40);

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

namespace {

// Bare-tree silhouette shared by every deciduous species in winter: trunk plus
// a broom of thin risers where the canopy used to be — reads as leafless
// branches at map scale.
CsgMesh bareTree(Rng& rng, float scale, float trunkH, float trunkR, const Color3& bark,
                 int branches, float branchH) {
    CsgMesh tree = makeCylinder({0.0f, 0.0f, 0.0f}, trunkR, trunkH + 0.06f * scale, 5, bark);
    for (int i = 0; i < branches; ++i) {
        const float angle = 2.0f * odai::math::kPi * static_cast<float>(i) /
                                static_cast<float>(branches) +
                            rng.uniform(0.0f, 0.8f);
        const float bx = 0.030f * scale * std::cos(angle);
        const float bz = -0.030f * scale * std::sin(angle);
        const float h = (branchH + rng.uniform(0.0f, 0.06f)) * scale;
        merge(tree, makeBox({bx - 0.006f, trunkH * 0.6f, bz - 0.006f},
                            {bx + 0.006f, trunkH * 0.6f + h, bz + 0.006f}, kBareBranch));
    }
    return tree;
}

// Seasonal canopy tint for a broadleaf-style species: base summer color, its
// own autumn color, optional spring blossom chance.
Color3 seasonLeaf(Rng& rng, Season season, const Color3& summer, const Color3& autumn,
                  float blossomChance) {
    switch (season) {
        case Season::Spring:
            if (blossomChance > 0.0f && rng.chance(blossomChance)) return kBlossom;
            return mix(mix(kLeafSpring, summer, 0.5f), kLeafSpringLight, rng.uniform(0.0f, 0.4f));
        case Season::Autumn:
            return mix(autumn, kLeafAutumnLight, rng.uniform(0.0f, 0.30f));
        default:
            return mix(summer, kLeafLight, rng.uniform(0.0f, 0.25f));
    }
}

}  // namespace

TriMesh generateTree(std::uint32_t variant, std::uint32_t seed, Season season) {
    Rng rng(seed ^ (variant + 1u) * 0x9E3779B9u);
    const float scale = rng.uniform(0.85f, 1.2f);
    CsgMesh tree;

    // Eight species keyed off the variant, each with its own silhouette and
    // seasonal response — the variety is what sells the diorama read. Callers
    // pick species contextually (willows at the waterline, shrubs in yards,
    // oaks in parks, mixed stands in the forest mask).
    switch (variant % 8u) {
        case 0u: {  // Broadleaf: short trunk, two bulging canopy tiers.
            const float trunkH = 0.09f * scale;
            const float r0 = 0.075f * scale * rng.uniform(0.9f, 1.15f);
            if (season == Season::Winter) {
                tree = bareTree(rng, scale, trunkH, 0.020f * scale, kTrunkBrown, 3, 0.14f);
                break;
            }
            const Color3 leaf = seasonLeaf(rng, season, mix(kLeafDark, kLeafMid, rng.uniform(0.0f, 1.0f)),
                                           mix(kLeafAutumnA, kLeafAutumnB, rng.uniform(0.0f, 1.0f)),
                                           0.25f);
            const Color3 topLeaf = mix(leaf, kLeafLight, 0.4f);
            tree = makeCylinder({0.0f, 0.0f, 0.0f}, 0.020f * scale, trunkH + 0.02f, 5, kTrunkBrown);
            merge(tree, makeConvexPrism(hexFootprint(r0 * 0.72f), trunkH, trunkH + 0.10f * scale,
                                        leaf, 1.45f));
            merge(tree, makeConvexPrism(hexFootprint(r0), trunkH + 0.10f * scale,
                                        trunkH + 0.20f * scale, topLeaf, 0.35f));
            break;
        }
        case 1u: {  // Conifer: stacked tapering tiers to a point.
            const float trunkH = 0.06f * scale;
            const float r0 = 0.062f * scale * rng.uniform(0.9f, 1.1f);
            tree = makeCylinder({0.0f, 0.0f, 0.0f}, 0.016f * scale, trunkH + 0.02f, 5, kTrunkBrown);
            Color3 leaf = mix(kConifer, kConiferLight, rng.uniform(0.0f, 1.0f));
            if (season == Season::Autumn) leaf = mix(leaf, kConiferAutumn, 0.5f);
            float y = trunkH;
            float r = r0;
            for (int tier = 0; tier < 3; ++tier) {
                const float tierH = (0.085f - 0.012f * static_cast<float>(tier)) * scale;
                Color3 tierColor = tier == 2 ? mix(leaf, kConiferLight, 0.35f) : leaf;
                if (season == Season::Winter) {
                    tierColor = mix(tierColor, kSnow, 0.30f + 0.18f * static_cast<float>(tier));
                }
                merge(tree, makeConvexPrism(hexFootprint(r), y, y + tierH, tierColor, 0.25f));
                y += tierH * 0.72f;  // tiers overlap like layered boughs
                r *= 0.74f;
            }
            break;
        }
        case 2u: {  // Birch: slim white trunk, small airy canopy; bright gold autumn.
            const float trunkH = 0.13f * scale;
            const float r0 = 0.050f * scale * rng.uniform(0.9f, 1.1f);
            if (season == Season::Winter) {
                tree = bareTree(rng, scale, trunkH, 0.012f * scale, kBirchBark, 3, 0.12f);
                break;
            }
            const Color3 leaf = seasonLeaf(rng, season, kBirchLeaf, kBirchAutumn, 0.0f);
            tree = makeCylinder({0.0f, 0.0f, 0.0f}, 0.012f * scale, trunkH + 0.02f, 5, kBirchBark);
            merge(tree, makeConvexPrism(hexFootprint(r0 * 0.8f), trunkH, trunkH + 0.09f * scale,
                                        leaf, 1.30f));
            merge(tree, makeConvexPrism(hexFootprint(r0), trunkH + 0.09f * scale,
                                        trunkH + 0.17f * scale, mix(leaf, kLeafLight, 0.35f),
                                        0.30f));
            break;
        }
        case 3u: {  // Columnar poplar/cypress: one tall slender flame.
            const float h = (0.30f + rng.uniform(0.0f, 0.08f)) * scale;
            const float r0 = 0.034f * scale;
            Color3 leaf = season == Season::Autumn ? kPoplarAutumn : kPoplarGreen;
            if (season == Season::Spring) leaf = mix(leaf, kLeafSpringLight, 0.3f);
            if (season == Season::Winter) leaf = mix(leaf, kSnow, 0.35f);
            tree = makeCylinder({0.0f, 0.0f, 0.0f}, 0.011f * scale, 0.05f, 5, kTrunkBrown);
            merge(tree, makeConvexPrism(hexFootprint(r0), 0.04f, 0.04f + h * 0.55f, leaf, 1.15f));
            merge(tree, makeConvexPrism(hexFootprint(r0 * 1.15f), 0.04f + h * 0.55f, 0.04f + h,
                                        mix(leaf, kLeafLight, 0.25f), 0.16f));
            break;
        }
        case 4u: {  // Willow: squat trunk, broad drooping dome with a hanging skirt.
            const float trunkH = 0.07f * scale;
            const float r0 = 0.105f * scale * rng.uniform(0.9f, 1.1f);
            if (season == Season::Winter) {
                tree = bareTree(rng, scale, trunkH, 0.024f * scale, kTrunkBrown, 5, 0.10f);
                break;
            }
            const Color3 leaf = seasonLeaf(rng, season, kWillowGreen,
                                           mix(kLeafAutumnLight, kBirchAutumn, 0.5f), 0.0f);
            tree = makeCylinder({0.0f, 0.0f, 0.0f}, 0.024f * scale, trunkH + 0.02f, 5, kTrunkBrown);
            // The skirt: a wide ring that tapers DOWN-and-out below the dome,
            // reading as trailing fronds.
            merge(tree, makeConvexPrism(hexFootprint(r0), trunkH + 0.015f, trunkH + 0.075f * scale,
                                        mix(leaf, kWillowLight, 0.25f), 1.35f));
            merge(tree, makeConvexPrism(hexFootprint(r0 * 1.35f), trunkH + 0.075f * scale,
                                        trunkH + 0.14f * scale, leaf, 0.30f));
            break;
        }
        case 5u: {  // Ornamental/blossom: small tidy globe; pink riot in spring.
            const float trunkH = 0.08f * scale;
            const float r0 = 0.055f * scale * rng.uniform(0.9f, 1.1f);
            if (season == Season::Winter) {
                tree = bareTree(rng, scale, trunkH, 0.014f * scale, kTrunkBrown, 3, 0.09f);
                break;
            }
            const Color3 leaf = seasonLeaf(rng, season, mix(kLeafMid, kLeafLight, 0.3f),
                                           fromRgbHex(0xC85A28), 0.85f);
            tree = makeCylinder({0.0f, 0.0f, 0.0f}, 0.014f * scale, trunkH + 0.02f, 5, kTrunkBrown);
            merge(tree, makeConvexPrism(hexFootprint(r0), trunkH, trunkH + 0.075f * scale, leaf,
                                        1.25f));
            merge(tree, makeConvexPrism(hexFootprint(r0 * 1.05f), trunkH + 0.075f * scale,
                                        trunkH + 0.13f * scale, mix(leaf, kLeafSpringLight, 0.25f),
                                        0.32f));
            break;
        }
        case 6u: {  // Oak: heavy trunk, broad three-tier crown — the park patriarch.
            const float trunkH = 0.10f * scale;
            const float r0 = 0.115f * scale * rng.uniform(0.92f, 1.08f);
            if (season == Season::Winter) {
                tree = bareTree(rng, scale, trunkH, 0.030f * scale, kTrunkBrown, 5, 0.16f);
                break;
            }
            const Color3 leaf = seasonLeaf(rng, season, kOakGreen, kOakAutumn, 0.0f);
            tree = makeCylinder({0.0f, 0.0f, 0.0f}, 0.030f * scale, trunkH + 0.02f, 6, kTrunkBrown);
            merge(tree, makeConvexPrism(hexFootprint(r0 * 0.70f), trunkH, trunkH + 0.09f * scale,
                                        leaf, 1.55f));
            merge(tree, makeConvexPrism(hexFootprint(r0), trunkH + 0.09f * scale,
                                        trunkH + 0.19f * scale, mix(leaf, kLeafLight, 0.20f),
                                        0.72f));
            merge(tree, makeConvexPrism(hexFootprint(r0 * 0.72f), trunkH + 0.19f * scale,
                                        trunkH + 0.26f * scale, mix(leaf, kLeafLight, 0.40f),
                                        0.30f));
            break;
        }
        default: {  // 7: yard shrub — a knee-high tuft for gardens and hedgerows.
            const float r0 = 0.036f * scale * rng.uniform(0.9f, 1.15f);
            Color3 leaf = kShrubGreen;
            if (season == Season::Autumn) leaf = mix(leaf, kLeafAutumnA, 0.45f);
            else if (season == Season::Spring) leaf = mix(leaf, kLeafSpringLight, 0.35f);
            tree = makeConvexPrism(hexFootprint(r0), 0.0f, 0.045f * scale, leaf, 1.20f);
            merge(tree, makeConvexPrism(hexFootprint(r0 * 1.1f), 0.045f * scale, 0.075f * scale,
                                        mix(leaf, kLeafLight, 0.3f), 0.35f));
            if (season == Season::Winter) {
                merge(tree, makeConvexPrism(hexFootprint(r0 * 0.95f), 0.075f * scale,
                                            0.088f * scale, kSnow, 0.5f));
            }
            break;
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

TriMesh generateBench(std::uint32_t seed) {
    Rng rng(seed);
    const Color3 wood = mix(fromRgbHex(0x8A6B3E), fromRgbHex(0x6B5230), rng.uniform(0.0f, 1.0f));
    const Color3 iron = fromRgbHex(0x2A2C30);
    const float halfW = 0.042f + rng.uniform(0.0f, 0.006f);

    // Two iron end frames, a slat seat, and a leaned back rail (facing -Z).
    CsgMesh bench = makeBox({-halfW, 0.0f, -0.012f}, {-halfW + 0.008f, 0.020f, 0.014f}, iron);
    merge(bench, makeBox({halfW - 0.008f, 0.0f, -0.012f}, {halfW, 0.020f, 0.014f}, iron));
    merge(bench, makeBox({-halfW - 0.004f, 0.020f, -0.014f}, {halfW + 0.004f, 0.028f, 0.012f}, wood));
    merge(bench, makeBox({-halfW - 0.004f, 0.028f, 0.006f}, {halfW + 0.004f, 0.052f, 0.014f}, wood));
    return triangulate(bench);
}

TriMesh generateHydrant(std::uint32_t seed) {
    Rng rng(seed);
    const Color3 paint = rng.chance(0.75f)
                             ? mix(fromRgbHex(0xC0392B), fromRgbHex(0xA82A20), rng.uniform(0.0f, 1.0f))
                             : fromRgbHex(0xE0B23A);
    const Color3 cap = mix(paint, fromRgbHex(0xFFFFFF), 0.35f);

    // Barrel, bonnet cap, and two side nozzles along local X.
    CsgMesh hydrant = makeCylinder({0.0f, 0.0f, 0.0f}, 0.011f, 0.034f, 6, paint);
    merge(hydrant, makeCylinder({0.0f, 0.034f, 0.0f}, 0.008f, 0.010f, 6, cap));
    merge(hydrant, makeBox({-0.017f, 0.018f, -0.005f}, {0.017f, 0.028f, 0.005f}, paint));
    return triangulate(hydrant);
}

TriMesh generateBillboard(std::uint32_t seed) {
    Rng rng(seed);
    const Color3 frame = fromRgbHex(0x4A4E55);
    // Two-tone "ad": a base field and a contrasting stripe band.
    const Color3 kAdColors[] = {
        fromRgbHex(0xE0C24A), fromRgbHex(0x4FC4E0), fromRgbHex(0xD64B3E), fromRgbHex(0x46C46B),
        fromRgbHex(0xE8E6DE), fromRgbHex(0x8E8CC4),
    };
    const Color3 field = kAdColors[rng.next() % 6u];
    Color3 stripe = kAdColors[rng.next() % 6u];
    if (stripe.r == field.r && stripe.g == field.g) stripe = frame;

    const float postH = 0.16f + rng.uniform(0.0f, 0.04f);
    const float panelW = 0.085f + rng.uniform(0.0f, 0.02f);
    const float panelH = 0.10f;

    CsgMesh billboard = makeBox({-0.010f, 0.0f, -0.010f}, {0.010f, postH, 0.010f}, frame);
    // Panel leans a touch back from the street (-Z) side.
    merge(billboard, makeBox({-panelW, postH, -0.016f}, {panelW, postH + panelH, -0.004f}, field));
    const float bandY = postH + rng.uniform(0.25f, 0.55f) * panelH;
    merge(billboard, makeBox({-panelW * 0.86f, bandY, -0.018f},
                             {panelW * 0.86f, bandY + panelH * 0.24f, -0.003f}, stripe));
    merge(billboard, makeBox({-panelW - 0.006f, postH - 0.010f, -0.017f},
                             {panelW + 0.006f, postH, -0.003f}, frame));
    return triangulate(billboard);
}

TriMesh generateBusStop(std::uint32_t seed) {
    Rng rng(seed);
    const Color3 frame = fromRgbHex(0x3A6B5C);
    const Color3 glassy = mix(fromRgbHex(0x9FC4D8), fromRgbHex(0x8AB4CC), rng.uniform(0.0f, 1.0f));
    const Color3 roof = fromRgbHex(0x2A2C30);
    const Color3 sign = fromRgbHex(0xE0C24A);
    const float halfW = 0.055f;

    // Back wall (glassy), flat roof, bench slat inside, open toward -Z.
    CsgMesh shelter = makeBox({-halfW, 0.0f, 0.028f}, {halfW, 0.105f, 0.036f}, glassy);
    merge(shelter, makeBox({-halfW, 0.0f, 0.020f}, {-halfW + 0.008f, 0.105f, 0.036f}, frame));
    merge(shelter, makeBox({halfW - 0.008f, 0.0f, 0.020f}, {halfW, 0.105f, 0.036f}, frame));
    merge(shelter, makeBox({-halfW - 0.006f, 0.105f, -0.020f}, {halfW + 0.006f, 0.118f, 0.042f}, roof));
    merge(shelter, makeBox({-halfW + 0.010f, 0.036f, 0.016f}, {halfW - 0.010f, 0.044f, 0.030f},
                           fromRgbHex(0x8A6B3E)));
    // Signpost beside the shelter.
    merge(shelter, makeBox({halfW + 0.012f, 0.0f, -0.004f}, {halfW + 0.020f, 0.150f, 0.004f}, frame));
    merge(shelter, makeBox({halfW + 0.004f, 0.126f, -0.006f}, {halfW + 0.028f, 0.150f, 0.006f}, sign));
    return triangulate(shelter);
}

TriMesh generateBoat(std::uint32_t variant, std::uint32_t seed) {
    Rng rng(seed);
    if ((variant & 1u) == 0u) {
        // Rowboat: tapered hull shell, bench thwart, tiny bow deck.
        const Color3 hull = mix(fromRgbHex(0x8A5C3E), fromRgbHex(0x3E8A50), rng.uniform(0.0f, 1.0f));
        const Color3 trim = fromRgbHex(0xD8C9A8);
        const float halfL = 0.065f + rng.uniform(0.0f, 0.010f);
        const float halfW = 0.024f;
        const std::vector<std::array<float, 2>> hullPlan = {
            {halfL, 0.0f}, {halfL * 0.45f, -halfW}, {-halfL, -halfW * 0.8f},
            {-halfL, halfW * 0.8f}, {halfL * 0.45f, halfW},
        };
        CsgMesh boat = makeConvexPrism(hullPlan, -0.010f, 0.020f, hull, 1.0f);
        merge(boat, makeBox({-halfL * 0.25f, 0.020f, -halfW * 0.8f},
                            {-halfL * 0.05f, 0.026f, halfW * 0.8f}, trim));
        merge(boat, makeBox({halfL * 0.45f, 0.020f, -halfW * 0.5f},
                            {halfL * 0.75f, 0.025f, halfW * 0.5f}, trim));
        return triangulate(boat);
    }
    // Barge: slab hull, cargo boxes, small wheelhouse at the stern.
    const Color3 hull = fromRgbHex(0x4A4E55);
    const Color3 cargo = mix(fromRgbHex(0xB0862E), fromRgbHex(0xA82A20), rng.uniform(0.0f, 1.0f));
    const Color3 cabin = fromRgbHex(0xE8E6DE);
    const float halfL = 0.100f + rng.uniform(0.0f, 0.012f);
    const float halfW = 0.034f;
    CsgMesh boat = makeBox({-halfL, -0.012f, -halfW}, {halfL, 0.022f, halfW}, hull);
    const int crates = rng.chance(0.5f) ? 2 : 1;
    for (int i = 0; i < crates; ++i) {
        const float cx0 = -halfL * 0.45f + static_cast<float>(i) * halfL * 0.62f;
        merge(boat, makeBox({cx0, 0.022f, -halfW * 0.7f}, {cx0 + halfL * 0.5f, 0.052f, halfW * 0.7f},
                            cargo));
    }
    merge(boat, makeBox({-halfL * 0.95f, 0.022f, -halfW * 0.55f},
                        {-halfL * 0.60f, 0.060f, halfW * 0.55f}, cabin));
    return triangulate(boat);
}

TriMesh generatePedestrian(std::uint32_t seed) {
    Rng rng(seed);
    const Color3 kShirts[] = {
        fromRgbHex(0xC0392B), fromRgbHex(0x3B90E0), fromRgbHex(0x46C46B), fromRgbHex(0xE0C24A),
        fromRgbHex(0x8E8CC4), fromRgbHex(0xE8E6DE), fromRgbHex(0xD87018), fromRgbHex(0x21A89A),
    };
    const Color3 shirt = kShirts[rng.next() % 8u];
    const Color3 trousers = mix(fromRgbHex(0x2E3238), fromRgbHex(0x4A3A2E), rng.uniform(0.0f, 1.0f));
    const Color3 skin = mix(fromRgbHex(0xE8C4A0), fromRgbHex(0x8A5C3E), rng.uniform(0.0f, 1.0f));

    const float build = rng.uniform(0.85f, 1.10f);
    const float hw = 0.0095f * build;   // torso half-width
    // Legs, torso, head — three stacked boxes read as a person at iso zoom.
    CsgMesh person = makeBox({-hw * 0.8f, 0.0f, -hw * 0.55f}, {hw * 0.8f, 0.032f, hw * 0.55f},
                             trousers);
    merge(person, makeBox({-hw, 0.032f, -hw * 0.7f}, {hw, 0.060f, hw * 0.7f}, shirt));
    merge(person, makeBox({-hw * 0.55f, 0.060f, -hw * 0.5f}, {hw * 0.55f, 0.074f, hw * 0.5f}, skin));
    return triangulate(person);
}

TriMesh generateSchoolBus(std::uint32_t seed) {
    Rng rng(seed);
    const Color3 yellow = mix(fromRgbHex(0xE8B820), fromRgbHex(0xF0C230), rng.uniform(0.0f, 1.0f));
    const Color3 roof = fromRgbHex(0xF0EEE4);
    const Color3 glass = fromRgbHex(0x9FC4D8);
    const Color3 tire = fromRgbHex(0x1C1E22);
    const Color3 bumper = fromRgbHex(0x3A3A40);

    const float halfL = 0.085f + rng.uniform(0.0f, 0.006f);
    const float halfW = 0.042f;

    // Long body slab, sloped-ish hood at the nose, window band, white roof.
    CsgMesh bus = makeBox({-halfL, 0.016f, -halfW}, {halfL, 0.075f, halfW}, yellow);
    merge(bus, makeBox({halfL - 0.006f, 0.016f, -halfW * 0.9f},
                       {halfL + 0.010f, 0.052f, halfW * 0.9f}, yellow));  // snub hood
    merge(bus, makeBox({-halfL * 0.92f, 0.048f, -halfW - 0.002f},
                       {halfL * 0.70f, 0.068f, halfW + 0.002f}, glass));  // window band
    merge(bus, makeBox({-halfL * 0.98f, 0.075f, -halfW * 0.85f},
                       {halfL * 0.85f, 0.084f, halfW * 0.85f}, roof));
    merge(bus, makeBox({-halfL - 0.004f, 0.018f, -halfW * 0.8f},
                       {-halfL, 0.042f, halfW * 0.8f}, bumper));
    // Tire strips fore and aft.
    merge(bus, makeBox({-halfL * 0.80f, 0.0f, -halfW}, {-halfL * 0.45f, 0.022f, halfW}, tire));
    merge(bus, makeBox({halfL * 0.40f, 0.0f, -halfW}, {halfL * 0.75f, 0.022f, halfW}, tire));
    return triangulate(bus);
}

TriMesh generateGarbageTruck(std::uint32_t seed) {
    Rng rng(seed);
    const Color3 cab = fromRgbHex(0x3A4A42);
    const Color3 hopper = mix(fromRgbHex(0x4E6B52), fromRgbHex(0x5A7A5E), rng.uniform(0.0f, 1.0f));
    const Color3 tire = fromRgbHex(0x1C1E22);
    const Color3 stripe = fromRgbHex(0xC9CCCE);

    const float halfL = 0.075f + rng.uniform(0.0f, 0.005f);
    const float halfW = 0.042f;

    // Cab up front (+X), tall boxy hopper behind, hazard stripe on the tail.
    CsgMesh truck = makeBox({halfL * 0.35f, 0.016f, -halfW * 0.95f},
                            {halfL, 0.062f, halfW * 0.95f}, cab);
    merge(truck, makeBox({halfL * 0.55f, 0.062f, -halfW * 0.8f},
                         {halfL * 0.95f, 0.070f, halfW * 0.8f}, cab));
    merge(truck, makeBox({-halfL, 0.016f, -halfW}, {halfL * 0.38f, 0.088f, halfW}, hopper));
    merge(truck, makeBox({-halfL - 0.004f, 0.030f, -halfW * 0.85f},
                         {-halfL + 0.006f, 0.075f, halfW * 0.85f}, stripe));
    merge(truck, makeBox({-halfL * 0.85f, 0.0f, -halfW}, {-halfL * 0.40f, 0.022f, halfW}, tire));
    merge(truck, makeBox({halfL * 0.45f, 0.0f, -halfW}, {halfL * 0.85f, 0.022f, halfW}, tire));
    return triangulate(truck);
}

TriMesh generateTrashCan(std::uint32_t seed) {
    Rng rng(seed);
    const Color3 body = rng.chance(0.7f) ? fromRgbHex(0x565A5E)
                                         : fromRgbHex(0x3E6B4A);  // the odd green bin
    const Color3 lid = mix(body, fromRgbHex(0xC9CCCE), 0.35f);

    CsgMesh can = makeCylinder({0.0f, 0.0f, 0.0f}, 0.011f, 0.026f, 6, body);
    merge(can, makeCylinder({0.0f, 0.026f, 0.0f}, 0.0125f, 0.005f, 6, lid));
    return triangulate(can);
}

}  // namespace odai::procgen
