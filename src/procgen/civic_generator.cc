#include "procgen/civic_generator.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include "procgen/primitives.h"
#include "procgen/rng.h"

// Civic buildings are assembled the same way the era generator works — a few
// seeded feature draws per kind — but each kind keeps one non-negotiable
// signature element (hose tower, colonnade, smokestacks, ...) so the service
// reads at a glance from the isometric camera. Heights stay within ~±30% of
// the old flat-box values so shadow lengths and the skyline don't jump.
namespace odai::procgen {

namespace {

const Color3 kWhiteWall = fromRgbHex(0xE8E6DE);
const Color3 kConcrete = fromRgbHex(0xC9C6BC);
const Color3 kDarkRoof = fromRgbHex(0x3A3A40);
const Color3 kGlass = fromRgbHex(0x5C88A0);
const Color3 kPoliceBlue = fromRgbHex(0x2F6BD6);
const Color3 kFireRed = fromRgbHex(0xC0392B);
const Color3 kClinicTeal = fromRgbHex(0x21A89A);
const Color3 kCrossRed = fromRgbHex(0xD64B3E);
const Color3 kSchoolBrick = fromRgbHex(0xB06A38);
const Color3 kSchoolTrim = fromRgbHex(0xE8D9B0);
const Color3 kLibraryStone = fromRgbHex(0xC9B698);
const Color3 kLibraryBrown = fromRgbHex(0x8A5C3E);
const Color3 kWood = fromRgbHex(0x8A6B3E);
const Color3 kStage = fromRgbHex(0x8A6B3E);  // boardwalk stage, not mini-golf lavender
const Color3 kSeatStone = fromRgbHex(0xAFAAA0);
const Color3 kPowerDark = fromRgbHex(0x4A4E55);
const Color3 kStackGrey = fromRgbHex(0x9AA0A8);
const Color3 kHazard = fromRgbHex(0xE0C24A);
const Color3 kWaterBlue = fromRgbHex(0x3D7BB8);

std::vector<std::array<float, 2>> rectFootprint(float x0, float z0, float x1, float z1) {
    return {{x0, z0}, {x1, z0}, {x1, z1}, {x0, z1}};
}

std::vector<std::array<float, 2>> ngonFootprint(float cx, float cz, float radius, int n) {
    std::vector<std::array<float, 2>> points;
    points.reserve(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        const float a = 2.0f * odai::math::kPi * static_cast<float>(i) / static_cast<float>(n);
        points.push_back({cx + radius * std::cos(a), cz - radius * std::sin(a)});
    }
    return points;
}

// Thin raised panel on the -Z (street) face — reads as a door/window without a
// BSP subtract. The seam is enclosed by the wall behind it, so merge is safe.
void addFrontPanel(CsgMesh& solid, float x0, float x1, float y0, float y1, float wallZ,
                   const Color3& color) {
    merge(solid, makeBox({x0, y0, wallZ - 0.022f}, {x1, y1, wallZ + 0.01f}, color));
}

// ── Police ───────────────────────────────────────────────────────────────────
CsgMesh buildPolice(float w, float d, Rng& rng) {
    const float x0 = 0.08f * w, x1 = w - 0.08f * w;
    const float z0 = 0.12f * d, z1 = d - 0.10f * d;
    const float h = rng.uniform(1.05f, 1.30f);

    CsgMesh solid = makeBox({x0, 0.0f, z0}, {x1, h, z1}, kConcrete);
    // Blue identity band under the parapet.
    merge(solid, makeBox({x0 - 0.015f, h - 0.16f, z0 - 0.015f}, {x1 + 0.015f, h - 0.08f, z1 + 0.015f},
                         kPoliceBlue));
    merge(solid, makeBox({x0 - 0.01f, h, z0 - 0.01f}, {x1 + 0.01f, h + 0.035f, z1 + 0.01f},
                         kDarkRoof));
    // Entry canopy on the street face, flanked by glass door panels.
    const float cx = 0.5f * (x0 + x1);
    const float cw = rng.uniform(0.18f, 0.26f) * w;
    merge(solid, makeBox({cx - cw, 0.30f, z0 - 0.10f}, {cx + cw, 0.36f, z0 + 0.02f}, kPoliceBlue));
    addFrontPanel(solid, cx - cw * 0.7f, cx + cw * 0.7f, 0.0f, 0.28f, z0, kGlass);
    // Antenna mast on a seeded corner.
    const bool eastMast = rng.chance(0.5f);
    const float mx = eastMast ? x1 - 0.10f * w : x0 + 0.10f * w;
    const float mz = z1 - 0.12f * d;
    merge(solid, makeBox({mx - 0.012f, h, mz - 0.012f}, {mx + 0.012f, h + rng.uniform(0.45f, 0.65f),
                                                         mz + 0.012f}, kDarkRoof));
    // Optional low annex wing.
    if (rng.chance(0.5f)) {
        const float ah = h * 0.55f;
        merge(solid, makeBox({x1 - 0.30f * w, 0.0f, z0 + 0.10f * d}, {w - 0.02f * w, ah, z1 - 0.10f * d},
                             mix(kConcrete, kWhiteWall, 0.4f)));
    }
    return solid;
}

// ── Fire ─────────────────────────────────────────────────────────────────────
CsgMesh buildFire(float w, float d, Rng& rng) {
    const float x0 = 0.08f * w, x1 = w - 0.08f * w;
    const float z0 = 0.12f * d, z1 = d - 0.10f * d;
    const float h = rng.uniform(0.95f, 1.15f);

    // Brick-red body; the fire-engine red stays on the bay doors where it
    // belongs (a 0.35 mix read as salmon, not firehouse).
    CsgMesh solid = makeBox({x0, 0.0f, z0}, {x1, h, z1}, mix(kFireRed, kWhiteWall, 0.15f));
    merge(solid, makeBox({x0 - 0.01f, h, z0 - 0.01f}, {x1 + 0.01f, h + 0.035f, z1 + 0.01f},
                         kDarkRoof));
    // Signature: 2-3 tall garage doors on the street face.
    const int doors = rng.range(2, 3);
    const float span = (x1 - x0) * 0.82f;
    const float doorW = span / static_cast<float>(doors) * 0.72f;
    const float step = span / static_cast<float>(doors);
    const float startX = 0.5f * (x0 + x1) - span * 0.5f;
    for (int i = 0; i < doors; ++i) {
        const float dx0 = startX + step * static_cast<float>(i) + (step - doorW) * 0.5f;
        addFrontPanel(solid, dx0, dx0 + doorW, 0.0f, h * 0.62f, z0, kFireRed);
        addFrontPanel(solid, dx0 + doorW * 0.1f, dx0 + doorW * 0.9f, h * 0.44f, h * 0.56f, z0 - 0.012f,
                      kWhiteWall);  // lintel stripe over each bay
    }
    // Signature: hose-drying tower on a seeded rear corner.
    const bool eastTower = rng.chance(0.5f);
    const float tx = eastTower ? x1 - 0.16f * w : x0;
    const float towerW = 0.16f * w;
    const float th = rng.uniform(1.55f, 1.85f);
    merge(solid, makeBox({tx, 0.0f, z1 - 0.22f * d}, {tx + towerW, th, z1 - 0.02f * d}, kFireRed));
    merge(solid, makeBox({tx - 0.015f, th, z1 - 0.235f * d}, {tx + towerW + 0.015f, th + 0.05f,
                                                              z1 - 0.005f * d}, kDarkRoof));
    return solid;
}

// ── Clinic ───────────────────────────────────────────────────────────────────
CsgMesh buildClinic(float w, float d, Rng& rng) {
    const float x0 = 0.08f * w, x1 = w - 0.08f * w;
    const float z0 = 0.12f * d, z1 = d - 0.10f * d;
    const float h = rng.uniform(1.15f, 1.40f);

    // White L-mass: main slab + lower street-side wing on a seeded end.
    CsgMesh solid = makeBox({x0, 0.0f, z0 + 0.28f * d}, {x1, h, z1}, kWhiteWall);
    const bool wingEast = rng.chance(0.5f);
    const float wx0 = wingEast ? 0.5f * (x0 + x1) : x0;
    const float wx1 = wingEast ? x1 : 0.5f * (x0 + x1);
    merge(solid, makeBox({wx0, 0.0f, z0}, {wx1, h * 0.55f, z0 + 0.32f * d}, kWhiteWall));
    // Teal ribbon windows on the main mass.
    merge(solid, makeBox({x0 - 0.012f, h * 0.35f, z0 + 0.27f * d}, {x1 + 0.012f, h * 0.50f, z1 + 0.012f},
                         kClinicTeal));
    merge(solid, makeBox({x0 - 0.012f, h * 0.68f, z0 + 0.27f * d}, {x1 + 0.012f, h * 0.83f, z1 + 0.012f},
                         kClinicTeal));
    // Signature: rooftop sign box with a red cross.
    const float sx = wingEast ? x0 + 0.20f * w : x1 - 0.20f * w;
    merge(solid, makeBox({sx - 0.14f, h, sx > 0.5f * w ? z1 - 0.34f * d : z0 + 0.40f * d},
                         {sx + 0.14f, h + 0.26f, (sx > 0.5f * w ? z1 - 0.34f * d : z0 + 0.40f * d) + 0.05f},
                         kWhiteWall));
    const float signZ = (sx > 0.5f * w ? z1 - 0.34f * d : z0 + 0.40f * d) - 0.012f;
    merge(solid, makeBox({sx - 0.10f, h + 0.10f, signZ}, {sx + 0.10f, h + 0.16f, signZ + 0.02f},
                         kCrossRed));
    merge(solid, makeBox({sx - 0.03f, h + 0.03f, signZ}, {sx + 0.03f, h + 0.23f, signZ + 0.02f},
                         kCrossRed));
    // Rooftop plant: 0-2 HVAC boxes.
    const int vents = rng.range(0, 2);
    for (int i = 0; i < vents; ++i) {
        const float vx = x0 + rng.uniform(0.25f, 0.75f) * (x1 - x0);
        const float vz = z0 + rng.uniform(0.45f, 0.85f) * (z1 - z0);
        merge(solid, makeBox({vx - 0.05f, h, vz - 0.05f}, {vx + 0.05f, h + 0.09f, vz + 0.05f},
                             kConcrete));
    }
    return solid;
}

// ── School ───────────────────────────────────────────────────────────────────
CsgMesh buildSchool(float w, float d, Rng& rng) {
    const float x0 = 0.08f * w, x1 = w - 0.08f * w;
    const float z0 = 0.14f * d, z1 = d - 0.10f * d;
    const float wallH = rng.uniform(0.95f, 1.15f);

    // Gabled main hall across the back of the lot.
    const float hallZ0 = z0 + 0.30f * d;
    CsgMesh solid = makeBox({x0, 0.0f, hallZ0}, {x1, wallH, z1}, kSchoolBrick);
    merge(solid, makeGablePrism(x0 - 0.01f, hallZ0 - 0.01f, x1 + 0.01f, z1 + 0.01f, wallH,
                                wallH + 0.02f, wallH + rng.uniform(0.28f, 0.40f), kDarkRoof));
    // Lower entry wing toward the street with a trim fascia.
    const float wingH = wallH * 0.55f;
    const float wx0 = x0 + 0.14f * w, wx1 = x1 - 0.14f * w;
    merge(solid, makeBox({wx0, 0.0f, z0}, {wx1, wingH, hallZ0 + 0.02f}, kSchoolBrick));
    merge(solid, makeBox({wx0 - 0.015f, wingH, z0 - 0.015f}, {wx1 + 0.015f, wingH + 0.04f,
                                                              hallZ0 + 0.02f}, kSchoolTrim));
    addFrontPanel(solid, 0.5f * (wx0 + wx1) - 0.10f, 0.5f * (wx0 + wx1) + 0.10f, 0.0f,
                  wingH * 0.75f, z0, kWood);
    // Signature: flagpole by the entrance.
    const float px = rng.chance(0.5f) ? wx0 - 0.10f : wx1 + 0.10f;
    merge(solid, makeCylinder({px, 0.0f, z0 + 0.03f}, 0.012f, rng.uniform(0.85f, 1.05f), 6,
                              kSchoolTrim));
    // 2-4 tall hall windows on the street-facing gable wall.
    const int windows = rng.range(2, 4);
    for (int i = 0; i < windows; ++i) {
        const float t = (static_cast<float>(i) + 0.5f) / static_cast<float>(windows);
        const float wx = x0 + t * (x1 - x0);
        merge(solid, makeBox({wx - 0.035f, wallH * 0.30f, hallZ0 - 0.018f},
                             {wx + 0.035f, wallH * 0.80f, hallZ0 + 0.01f}, kSchoolTrim));
    }
    // Optional gym annex on a rear corner.
    if (rng.chance(0.45f)) {
        merge(solid, makeBox({x1 - 0.26f * w, 0.0f, hallZ0 + 0.06f * d},
                             {x1 + 0.02f * w, wallH * 0.72f, z1 - 0.04f * d},
                             mix(kSchoolBrick, kSchoolTrim, 0.25f)));
    }
    return solid;
}

// ── Park ─────────────────────────────────────────────────────────────────────
// A full garden layout, not just a centerpiece: crossing gravel paths, a low
// perimeter hedge with gate gaps, flower beds or a pond in the quadrants, and
// a gazebo or fountain in the middle. Trees are placed by the app (they need
// the live season); everything else lives here.
CsgMesh buildPark(float w, float d, Rng& rng) {
    const float cx = 0.5f * w, cz = 0.5f * d;
    const Color3 kGravel = fromRgbHex(0xB8A98C);
    const Color3 kHedge = fromRgbHex(0x3E7A34);
    const Color3 kFlowerColors[4] = {fromRgbHex(0xD86A9C), fromRgbHex(0xE0C24A),
                                     fromRgbHex(0xD64B3E), fromRgbHex(0xB88CE0)};

    // Crossing paths, slightly proud of the lawn so they read at iso zoom.
    const float pathHalf = 0.055f * w;
    CsgMesh solid = makeBox({cx - pathHalf, 0.0f, 0.0f}, {cx + pathHalf, 0.012f, d}, kGravel);
    merge(solid, makeBox({0.0f, 0.0f, cz - pathHalf}, {w, 0.012f, cz + pathHalf}, kGravel));

    // Perimeter hedge: two runs per side, leaving a gate gap where each path
    // meets the edge.
    const float hedgeT = 0.035f * w, hedgeH = 0.055f;
    const float gap = pathHalf + 0.03f * w;
    const auto hedgeRun = [&](float hx0, float hz0, float hx1, float hz1) {
        if (hx1 > hx0 && hz1 > hz0) merge(solid, makeBox({hx0, 0.0f, hz0}, {hx1, hedgeH, hz1}, kHedge));
    };
    hedgeRun(0.0f, 0.0f, cx - gap, hedgeT);                 // north, west half
    hedgeRun(cx + gap, 0.0f, w, hedgeT);                    // north, east half
    hedgeRun(0.0f, d - hedgeT, cx - gap, d);                // south halves
    hedgeRun(cx + gap, d - hedgeT, w, d);
    hedgeRun(0.0f, hedgeT, hedgeT, cz - gap);               // west halves
    hedgeRun(0.0f, cz + gap, hedgeT, d - hedgeT);
    hedgeRun(w - hedgeT, hedgeT, w, cz - gap);              // east halves
    hedgeRun(w - hedgeT, cz + gap, w, d - hedgeT);

    // Quadrant dressing: flower beds (raised soil box with a bloom cap) or a
    // pond in one seeded quadrant.
    const int pondQuadrant = rng.chance(0.4f) ? rng.range(0, 3) : -1;
    for (int q = 0; q < 4; ++q) {
        const float qx = (q & 1) ? cx + (w - cx) * 0.5f : cx * 0.5f;
        const float qz = (q & 2) ? cz + (d - cz) * 0.5f : cz * 0.5f;
        if (q == pondQuadrant) {
            const float pr = 0.14f * w;
            merge(solid, makeConvexPrism(ngonFootprint(qx, qz, pr, 8), 0.0f, 0.022f, kConcrete));
            merge(solid, makeConvexPrism(ngonFootprint(qx, qz, pr * 0.82f, 8), 0.022f, 0.034f,
                                         kWaterBlue));
            continue;
        }
        if (!rng.chance(0.75f)) continue;  // some quadrants stay open lawn
        const float bw = rng.uniform(0.09f, 0.13f) * w;
        const Color3 bloom = kFlowerColors[rng.next() % 4u];
        merge(solid, makeBox({qx - bw, 0.0f, qz - bw * 0.6f}, {qx + bw, 0.020f, qz + bw * 0.6f},
                             mix(kWood, fromRgbHex(0x4A3A2E), 0.5f)));
        merge(solid, makeBox({qx - bw * 0.85f, 0.020f, qz - bw * 0.45f},
                             {qx + bw * 0.85f, 0.034f, qz + bw * 0.45f}, bloom));
    }

    if (rng.chance(0.5f)) {
        // Gazebo: plinth, posts, tapered octagonal roof with a finial.
        const float radius = std::min(w, d) * rng.uniform(0.20f, 0.24f);
        merge(solid, makeConvexPrism(ngonFootprint(cx, cz, radius, 8), 0.0f, 0.05f, kConcrete));
        for (int i = 0; i < 4; ++i) {
            const float a = odai::math::kPi * (0.25f + 0.5f * static_cast<float>(i));
            const float px = cx + radius * 0.78f * std::cos(a);
            const float pz = cz - radius * 0.78f * std::sin(a);
            merge(solid, makeBox({px - 0.018f, 0.05f, pz - 0.018f}, {px + 0.018f, 0.34f, pz + 0.018f},
                                 kWood));
        }
        merge(solid, makeConvexPrism(ngonFootprint(cx, cz, radius * 1.08f, 8), 0.34f, 0.50f,
                                     kDarkRoof, 0.18f));
        merge(solid, makeBox({cx - 0.015f, 0.50f, cz - 0.015f}, {cx + 0.015f, 0.56f, cz + 0.015f},
                             kWood));
    } else {
        // Fountain: wide basin, water disc, centre jet column.
        const float radius = std::min(w, d) * rng.uniform(0.18f, 0.22f);
        merge(solid, makeConvexPrism(ngonFootprint(cx, cz, radius, 8), 0.0f, 0.10f, kConcrete));
        merge(solid, makeConvexPrism(ngonFootprint(cx, cz, radius * 0.82f, 8), 0.10f, 0.125f,
                                     kWaterBlue));
        merge(solid, makeCylinder({cx, 0.0f, cz}, radius * 0.16f, rng.uniform(0.28f, 0.36f), 6,
                                  kConcrete));
        merge(solid, makeCylinder({cx, rng.uniform(0.28f, 0.36f), cz}, radius * 0.24f, 0.03f, 6,
                                  kWaterBlue));
    }
    return solid;
}

// ── Library ──────────────────────────────────────────────────────────────────
CsgMesh buildLibrary(float w, float d, Rng& rng) {
    const float x0 = 0.10f * w, x1 = w - 0.10f * w;
    const float z0 = 0.14f * d, z1 = d - 0.10f * d;
    const float h = rng.uniform(1.05f, 1.25f);

    // Stepped entry stairs, then the main stone mass behind a colonnade.
    CsgMesh solid = makeBox({x0 + 0.10f * w, 0.0f, z0 - 0.035f * d}, {x1 - 0.10f * w, 0.045f, z0 + 0.05f},
                            kConcrete);
    merge(solid, makeBox({x0 + 0.13f * w, 0.045f, z0 - 0.02f * d}, {x1 - 0.13f * w, 0.09f, z0 + 0.05f},
                         kConcrete));
    const float porchZ = z0 + 0.16f * d;
    merge(solid, makeBox({x0, 0.09f, porchZ}, {x1, h, z1}, kLibraryStone));
    // Signature: the colonnade — 4-5 columns carrying an entablature.
    const int cols = rng.range(4, 5);
    for (int i = 0; i < cols; ++i) {
        const float t = static_cast<float>(i) / static_cast<float>(cols - 1);
        const float px = x0 + 0.08f * w + t * (x1 - x0 - 0.16f * w);
        merge(solid, makeCylinder({px, 0.09f, z0 + 0.07f}, 0.030f, h * 0.72f, 6, kLibraryStone));
    }
    merge(solid, makeBox({x0 - 0.015f, 0.09f + h * 0.72f, z0 + 0.02f},
                         {x1 + 0.015f, 0.09f + h * 0.72f + 0.10f, porchZ + 0.05f}, kLibraryStone));
    // Low-pitch roof cap and a bronze-ish door.
    merge(solid, makeConvexPrism(rectFootprint(x0 - 0.015f, porchZ - 0.015f, x1 + 0.015f, z1 + 0.015f),
                                 h, h + 0.14f, kLibraryBrown, 0.72f));
    addFrontPanel(solid, 0.5f * w - 0.09f, 0.5f * w + 0.09f, 0.09f, 0.52f, porchZ, kLibraryBrown);
    return solid;
}

// ── Amphitheater ─────────────────────────────────────────────────────────────
CsgMesh buildAmphitheater(float w, float d, Rng& rng) {
    const float x0 = 0.06f * w, x1 = w - 0.06f * w;
    const float z0 = 0.10f * d, z1 = d - 0.08f * d;

    // Stage slab at the street side; seating tiers rise away from it.
    CsgMesh solid = makeBox({x0 + 0.16f * w, 0.0f, z0}, {x1 - 0.16f * w, 0.10f, z0 + 0.26f * d},
                            kStage);
    const int tiers = rng.range(3, 4);
    const float tierD = (z1 - z0 - 0.30f * d) / static_cast<float>(tiers);
    float inset = 0.105f * w;
    for (int i = 0; i < tiers; ++i) {
        const float tz0 = z0 + 0.30f * d + tierD * static_cast<float>(i);
        const float th = 0.10f + 0.09f * static_cast<float>(i + 1);
        merge(solid, makeBox({x0 + inset, 0.0f, tz0}, {x1 - inset, th, tz0 + tierD + 0.01f},
                             i % 2 == 0 ? kSeatStone : mix(kSeatStone, kConcrete, 0.5f)));
        // The fan splays OUTWARD as seating rises away from the stage — every
        // Greek plan and WPA band shell widens with distance, never narrows.
        inset -= 0.035f * w;
    }
    // Back wall with a seeded pair of stair notches left as raised blocks.
    const float backH = 0.10f + 0.09f * static_cast<float>(tiers) + 0.12f;
    merge(solid, makeBox({x0 + inset - 0.02f * w, 0.0f, z1 - 0.05f * d},
                         {x1 - inset + 0.02f * w, backH, z1}, kConcrete));
    if (rng.chance(0.6f)) {  // proscenium posts framing the stage
        merge(solid, makeBox({x0 + 0.14f * w, 0.0f, z0}, {x0 + 0.19f * w, 0.42f, z0 + 0.05f}, kStage));
        merge(solid, makeBox({x1 - 0.19f * w, 0.0f, z0}, {x1 - 0.14f * w, 0.42f, z0 + 0.05f}, kStage));
    }
    return solid;
}

// ── Power plant ──────────────────────────────────────────────────────────────
CsgMesh buildPowerPlant(float w, float d, Rng& rng) {
    const float x0 = 0.06f * w, x1 = w - 0.06f * w;
    const float z0 = 0.10f * d, z1 = d - 0.08f * d;
    const float hallH = rng.uniform(1.25f, 1.50f);

    // Turbine hall with a raised monitor roof strip.
    CsgMesh solid = makeBox({x0, 0.0f, z0}, {x1 - 0.26f * w, hallH, z1}, kPowerDark);
    merge(solid, makeBox({x0 + 0.10f * w, hallH, z0 + 0.14f * d},
                         {x1 - 0.36f * w, hallH + 0.16f, z1 - 0.14f * d}, mix(kPowerDark, kGlass, 0.4f)));
    // Hazard stripe over the intake doors.
    addFrontPanel(solid, x0 + 0.08f * w, x0 + 0.34f * w, 0.0f, hallH * 0.42f, z0, mix(kPowerDark, kHazard, 0.25f));
    merge(solid, makeBox({x0 + 0.06f * w, hallH * 0.46f, z0 - 0.012f},
                         {x0 + 0.36f * w, hallH * 0.52f, z0 + 0.01f}, kHazard));
    // Signature: two banded smokestacks on the open side of the lot.
    const float stackX = x1 - 0.13f * w;
    const int stacks = 2;
    for (int i = 0; i < stacks; ++i) {
        const float sz = z0 + (0.30f + 0.38f * static_cast<float>(i)) * (z1 - z0);
        const float sh = rng.uniform(2.1f, 2.5f) - 0.15f * static_cast<float>(i);
        merge(solid, makeCylinder({stackX, 0.0f, sz}, 0.085f, sh, 8, kStackGrey));
        merge(solid, makeCylinder({stackX, sh, sz}, 0.088f, 0.10f, 8, kFireRed));
    }
    // Transformer / tank cluster between hall and stacks.
    const int tanks = rng.range(1, 3);
    for (int i = 0; i < tanks; ++i) {
        const float tz = z0 + (0.18f + 0.28f * static_cast<float>(i)) * (z1 - z0);
        merge(solid, makeCylinder({x1 - 0.30f * w, 0.0f, tz}, 0.07f, 0.32f, 6, kConcrete));
    }
    return solid;
}

}  // namespace

TriMesh generateCivicBuilding(const CivicDesc& desc) {
    // Decorrelate every descriptor field, like generateBuilding does.
    std::uint32_t seed = desc.seed ? desc.seed : 1u;
    seed ^= static_cast<std::uint32_t>(desc.kind) * 0x9E3779B9u;
    seed ^= static_cast<std::uint32_t>(desc.lotWidth * 64.0f) * 0x85EBCA6Bu;
    seed ^= static_cast<std::uint32_t>(desc.lotDepth * 64.0f) * 0xC2B2AE35u;
    Rng rng(seed);

    const float w = std::max(0.4f, desc.lotWidth);
    const float d = std::max(0.4f, desc.lotDepth);

    CsgMesh solid;
    switch (desc.kind) {
        case CivicKind::Police:       solid = buildPolice(w, d, rng); break;
        case CivicKind::Fire:         solid = buildFire(w, d, rng); break;
        case CivicKind::Clinic:       solid = buildClinic(w, d, rng); break;
        case CivicKind::School:       solid = buildSchool(w, d, rng); break;
        case CivicKind::Park:         solid = buildPark(w, d, rng); break;
        case CivicKind::Library:      solid = buildLibrary(w, d, rng); break;
        case CivicKind::Amphitheater: solid = buildAmphitheater(w, d, rng); break;
        case CivicKind::PowerPlant:   solid = buildPowerPlant(w, d, rng); break;
    }
    return triangulate(solid);
}

}  // namespace odai::procgen
