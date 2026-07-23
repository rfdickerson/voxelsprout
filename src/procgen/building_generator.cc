#include "procgen/building_generator.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include "procgen/primitives.h"
#include "procgen/rng.h"

// Buildings are assembled as a series of seeded "feature draws" — massing,
// roof, crown, and facade details each picked from a per-era option pool — so
// a single style yields combinatorially many silhouettes while every draw
// stays on-palette. Same seed => same building, bit-exact.
namespace odai::procgen {

namespace {

// ── Palettes (option pools, drawn per building) ─────────────────────────────
// 1890s: brick and sandstone.
const std::array<Color3, 4> kBrickPool = {fromRgbHex(0x9C4A38), fromRgbHex(0x8A5C3E),
                                          fromRgbHex(0x6B4638), fromRgbHex(0xA85848)};
const std::array<Color3, 3> kTrimPool = {fromRgbHex(0xD8C9A8), fromRgbHex(0xC9B698),
                                         fromRgbHex(0xB8A488)};
const Color3 kDarkRoof = fromRgbHex(0x3A3A40);
const Color3 kSootBrick = fromRgbHex(0x6B4638);
const std::array<Color3, 4> kAwningPool = {fromRgbHex(0x3E8A50), fromRgbHex(0xA83A30),
                                           fromRgbHex(0x30588A), fromRgbHex(0x8A6B3E)};
// 1930s: limestone deco.
const std::array<Color3, 3> kDecoBodyPool = {fromRgbHex(0xD6C8A6), fromRgbHex(0xC4B490),
                                             fromRgbHex(0xCEBE9C)};
const std::array<Color3, 3> kDecoAccentPool = {fromRgbHex(0x3E8A80), fromRgbHex(0x8A6B3E),
                                               fromRgbHex(0x5A7A8A)};
// 1960s: concrete and curtain wall.
const Color3 kConcrete = fromRgbHex(0xE2E2DA);
const Color3 kGreyPanel = fromRgbHex(0xB8BCC0);
const std::array<Color3, 3> kGlassPool = {fromRgbHex(0x5C88A0), fromRgbHex(0x4E8A78),
                                          fromRgbHex(0x6A7E9C)};
const Color3 kMullion = fromRgbHex(0x2E3238);

// Residential wealth reads through height, like the old flat boxes did.
float tierHeightMul(int tier) {
    return tier <= 0 ? 0.55f : (tier >= 2 ? 1.35f : 1.0f);
}

std::vector<std::array<float, 2>> rectFootprint(float x0, float z0, float x1, float z1) {
    return {{x0, z0}, {x1, z0}, {x1, z1}, {x0, z1}};
}

// Gable prism with the ridge along Z: build the X-ridge version in a scratch
// frame, rotate a quarter turn, and translate into place.
CsgMesh makeGablePrismZ(float minX, float minZ, float maxX, float maxZ,
                        float y0, float eaveY, float ridgeY, const Color3& color) {
    const float w = maxZ - minZ, d = maxX - minX;
    CsgMesh g = makeGablePrism(0.0f, 0.0f, w, d, y0, eaveY, ridgeY, color);
    rotateY(g, -odai::math::kPi * 0.5f);   // (x,z) -> (-z, x)
    translate(g, {minX + d, 0.0f, minZ});
    return g;
}

// Mansard roof: steeply tapered rect prism with a small flat cap.
CsgMesh makeMansard(float x0, float z0, float x1, float z1, float y0, float h,
                    const Color3& color) {
    CsgMesh roof = makeConvexPrism(rectFootprint(x0, z0, x1, z1), y0, y0 + h, color, 0.58f);
    return roof;
}

// A thin box rotated 45 degrees about a vertical corner axis, used to shave
// art-deco chamfers off a mass's vertical edges.
CsgMesh cornerChamferCutter(float cornerX, float cornerZ, float size, float top) {
    CsgMesh cutter = makeBox({-size, -0.05f, -size}, {size, top, size}, kConcrete);
    rotateY(cutter, odai::math::kPi * 0.25f);
    translate(cutter, {cornerX, 0.0f, cornerZ});
    return cutter;
}

void addChimney(CsgMesh& solid, float cx, float cz, float baseY, float topY) {
    merge(solid, makeBox({cx - 0.028f, baseY, cz - 0.028f}, {cx + 0.028f, topY, cz + 0.028f},
                         kSootBrick));
}

// ── Windows (LOD detail pass) ────────────────────────────────────────────────
// Windows are "painted on": single outward-facing quads floated a hair off
// the wall, colored like glass — 2 triangles each instead of a 12-triangle
// box. Collected in a separate mesh and merged after all BSP ops (open quads
// must never pass through csgUnion/csgSubtract). Deliberately rng-free so
// detail 0 and detail 1 produce identical massing from the same seed.
const Color3 kSashGlass = fromRgbHex(0x39434E);      // dark Victorian panes
const Color3 kRibbonGlass = fromRgbHex(0x3E6A78);    // deco teal
const Color3 kFactoryGlass = fromRgbHex(0x4E6070);   // industrial multipane

enum class WindowStyle {
    kSash,     // 1890s: punched grid with a light sill under each window
    kRibbon,   // 1930s: continuous vertical strips between pilasters
    kMullion,  // 1960s: thin dark verticals over an already-glass skin
    kBand,     // wide horizontal glazing band per floor
};

void addFacadeQuad(CsgMesh& mesh, int face, float u0, float u1, float y0, float y1, float wall,
                   const Color3& color) {
    // face 0 = -Z, 1 = +Z, 2 = -X, 3 = +X; u runs along the face, wall is the
    // fixed coordinate already offset off the surface.
    Polygon p;
    p.color = color;
    const auto at = [&](float u, float y) -> Vector3 {
        return face < 2 ? Vector3{u, y, wall} : Vector3{wall, y, u};
    };
    p.vertices = {at(u0, y0), at(u1, y0), at(u1, y1), at(u0, y1)};
    p.plane = Plane::fromVertices(p.vertices);
    const float outward = face == 0 ? -p.plane.normal.z
                          : face == 1 ? p.plane.normal.z
                          : face == 2 ? -p.plane.normal.x
                                      : p.plane.normal.x;
    if (outward < 0.0f) p.flip();
    mesh.polygons.push_back(p);
}

// Decorate all four side faces of a box mass [x0..x1] x [y0..y1] x [z0..z1].
void addWindowsBox(CsgMesh& windows, float x0, float y0, float z0, float x1, float y1, float z1,
                   WindowStyle style, const Color3& glass, const Color3& trim) {
    constexpr float kOff = 0.006f;  // float distance off the wall
    if (y1 - y0 < 0.10f) return;
    for (int face = 0; face < 4; ++face) {
        const float fu0 = face < 2 ? x0 : z0;
        const float fu1 = face < 2 ? x1 : z1;
        const float wall = face == 0 ? z0 - kOff
                           : face == 1 ? z1 + kOff
                           : face == 2 ? x0 - kOff
                                       : x1 + kOff;
        const float width = fu1 - fu0;
        if (width < 0.12f) continue;
        switch (style) {
            case WindowStyle::kSash: {
                const int rows = std::clamp(static_cast<int>((y1 - y0) / 0.17f), 1, 4);
                const int cols = std::clamp(static_cast<int>(width / 0.145f), 1, 5);
                const float rowH = (y1 - y0) / static_cast<float>(rows);
                // Period 1/1 and 2/2 double-hung sash reads ~1:2 width:height;
                // a squat near-square upper window is the tell of a remodel.
                const float winW = 0.042f, winH = std::min(0.085f, rowH * 0.55f);
                for (int r = 0; r < rows; ++r) {
                    const float wy0 = y0 + rowH * (static_cast<float>(r) + 0.30f);
                    for (int col = 0; col < cols; ++col) {
                        const float cu = fu0 + width * (static_cast<float>(col) + 0.5f) /
                                                   static_cast<float>(cols);
                        addFacadeQuad(windows, face, cu - winW * 0.5f, cu + winW * 0.5f, wy0,
                                      wy0 + winH, wall, glass);
                        addFacadeQuad(windows, face, cu - winW * 0.5f - 0.008f,
                                      cu + winW * 0.5f + 0.008f, wy0 - 0.012f, wy0, wall, trim);
                    }
                }
                break;
            }
            case WindowStyle::kRibbon: {
                const int cols = std::clamp(static_cast<int>(width / 0.11f), 2, 6);
                const float stripW = 0.042f;
                for (int col = 0; col < cols; ++col) {
                    const float cu = fu0 + width * (static_cast<float>(col) + 0.5f) /
                                               static_cast<float>(cols);
                    addFacadeQuad(windows, face, cu - stripW * 0.5f, cu + stripW * 0.5f, y0, y1,
                                  wall, glass);
                }
                break;
            }
            case WindowStyle::kMullion: {
                const int cols = std::clamp(static_cast<int>(width / 0.058f), 2, 14);
                for (int col = 1; col < cols; ++col) {
                    const float cu = fu0 + width * static_cast<float>(col) /
                                               static_cast<float>(cols);
                    addFacadeQuad(windows, face, cu - 0.004f, cu + 0.004f, y0, y1, wall, trim);
                }
                break;
            }
            case WindowStyle::kBand: {
                const int rows = std::clamp(static_cast<int>((y1 - y0) / 0.22f), 1, 6);
                const float rowH = (y1 - y0) / static_cast<float>(rows);
                for (int r = 0; r < rows; ++r) {
                    const float wy0 = y0 + rowH * (static_cast<float>(r) + 0.35f);
                    addFacadeQuad(windows, face, fu0 + 0.02f, fu1 - 0.02f, wy0,
                                  wy0 + std::min(0.055f, rowH * 0.42f), wall, glass);
                }
                break;
            }
        }
    }
}

// ── 1890s ───────────────────────────────────────────────────────────────────

// Roof draw shared by the residential masses: gable either way, flat parapet,
// or a Second-Empire mansard.
void add1890Roof(CsgMesh& solid, Rng& rng, float x0, float z0, float x1, float z1,
                 float wallH, float heightMul, const Color3& brick, const Color3& trim,
                 bool& outFlatTop) {
    outFlatTop = false;
    // The Second Empire mansard was a holdover by 1890, not a fashion — gate
    // it to ~12% of the stock so the skyline doesn't read 1875. Gables and
    // parapets split the rest.
    const int roof = rng.chance(0.12f) ? 3 : rng.range(0, 2);
    switch (roof) {
        case 0:  // gable, ridge along X
            solid = csgUnion(solid, makeGablePrism(x0, z0, x1, z1, wallH, wallH + 0.02f,
                                                   wallH + 0.10f + 0.10f * heightMul, kDarkRoof));
            break;
        case 1:  // gable, ridge along Z
            solid = csgUnion(solid, makeGablePrismZ(x0, z0, x1, z1, wallH, wallH + 0.02f,
                                                    wallH + 0.10f + 0.10f * heightMul, kDarkRoof));
            break;
        case 2: {  // cornice + recessed parapet roof
            CsgMesh cornice = makeBox({x0 - 0.02f, wallH, z0 - 0.02f},
                                      {x1 + 0.02f, wallH + 0.035f, z1 + 0.02f}, trim);
            solid = csgUnion(solid, cornice);
            solid = csgSubtract(solid, makeBox({x0 + 0.05f, wallH + 0.015f, z0 + 0.05f},
                                               {x1 - 0.05f, wallH + 0.5f, z1 - 0.05f}, brick));
            outFlatTop = true;
            break;
        }
        default:  // mansard with a trim cap line
            solid = csgUnion(solid, makeMansard(x0, z0, x1, z1, wallH, 0.10f + 0.06f * heightMul,
                                                kDarkRoof));
            merge(solid, makeBox({x0 - 0.015f, wallH - 0.012f, z0 - 0.015f},
                                 {x1 + 0.015f, wallH + 0.012f, z1 + 0.015f}, trim));
            break;
    }
}

CsgMesh build1890Residential(float w, float d, int level, int tier, Rng& rng, bool detail) {
    const float heightMul = tierHeightMul(tier);
    const float mx = 0.10f * w, mz = 0.10f * d;
    const float x0 = mx, x1 = w - mx, z0 = mz, z1 = d - mz;
    const Color3 brick = rng.pick(kBrickPool);
    const Color3 trim = rng.pick(kTrimPool);
    const float wallH = (0.26f + 0.15f * static_cast<float>(level) + rng.uniform(0.0f, 0.08f)) * heightMul;

    // Wall rectangles the window pass decorates once the massing is settled.
    struct Mass {
        float x0, z0, x1, z1, h;
    };
    Mass masses[2];
    int numMasses = 0;

    // Massing draw: one mass, a pair of attached row-houses, or an L-shaped
    // main mass + lower wing. (Trailer-park tier stays a single low mass.)
    enum Massing { kSingle, kTwin, kMainWing };
    const Massing massing =
        tier == 0 ? kSingle : static_cast<Massing>(rng.range(0, 2));

    CsgMesh solid;
    bool flatTop = false;
    float frontH = wallH;  // wall height of the mass that owns the front door
    if (massing == kTwin) {
        // Two attached row-houses: shared party wall, offset heights and a
        // sibling brick shade. Seam faces are enclosed, so merge is safe.
        const float xm = 0.5f * (x0 + x1);
        const float hA = wallH, hB = wallH * rng.uniform(0.82f, 0.95f);
        const Color3 brickB = mix(brick, rng.pick(kBrickPool), 0.6f);
        CsgMesh a = makeBox({x0, 0.0f, z0}, {xm, hA, z1}, brick);
        bool flatA = false;
        add1890Roof(a, rng, x0, z0, xm, z1, hA, heightMul, brick, trim, flatA);
        CsgMesh b = makeBox({xm, 0.0f, z0}, {x1, hB, z1}, brickB);
        bool flatB = false;
        add1890Roof(b, rng, xm, z0, x1, z1, hB, heightMul, brickB, trim, flatB);
        solid = std::move(a);
        merge(solid, b);
        flatTop = flatA;
        masses[numMasses++] = {x0, z0, xm, z1, hA};
        masses[numMasses++] = {xm, z0, x1, z1, hB};
    } else if (massing == kMainWing) {
        // Main mass on one side, lower wing filling the rest of the lot.
        const float split = x0 + (x1 - x0) * rng.uniform(0.52f, 0.66f);
        const float wingH = wallH * rng.uniform(0.55f, 0.72f);
        const float wingZ1 = z1 - (z1 - z0) * rng.uniform(0.15f, 0.35f);
        solid = makeBox({x0, 0.0f, z0}, {split, wallH, z1}, brick);
        add1890Roof(solid, rng, x0, z0, split, z1, wallH, heightMul, brick, trim, flatTop);
        CsgMesh wing = makeBox({split, 0.0f, z0}, {x1, wingH, wingZ1}, mix(brick, trim, 0.15f));
        bool wingFlat = false;
        add1890Roof(wing, rng, split, z0, x1, wingZ1, wingH, heightMul * 0.7f, brick, trim, wingFlat);
        merge(solid, wing);
        masses[numMasses++] = {x0, z0, split, z1, wallH};
        masses[numMasses++] = {split, z0, x1, wingZ1, wingH};
    } else {
        solid = makeBox({x0, 0.0f, z0}, {x1, wallH, z1}, brick);
        add1890Roof(solid, rng, x0, z0, x1, z1, wallH, heightMul, brick, trim, flatTop);
        masses[numMasses++] = {x0, z0, x1, z1, wallH};
    }

    // Facade features, each its own draw.
    // Chimneys and bays keep to the left 40% of the frontage — that range is
    // always owned by the full-height main mass in every massing draw, so
    // nothing floats above a lower twin/wing roofline.
    if (tier > 0) {
        const int chimneys = rng.range(1, 2);
        for (int i = 0; i < chimneys; ++i) {
            const float cx = rng.uniform(x0 + 0.06f, x0 + 0.40f * (x1 - x0));
            const float cz = rng.uniform(z1 - 0.16f, z1 - 0.06f);
            addChimney(solid, cx, cz, frontH - 0.06f, frontH + (flatTop ? 0.10f : 0.22f) * heightMul);
        }
    }
    if (tier >= 1 && rng.chance(0.55f)) {
        // Bay window on the street face.
        const float bx = x0 + rng.uniform(0.08f, 0.30f) * (x1 - x0);
        merge(solid, makeBox({bx, 0.05f, z0 - 0.035f}, {bx + 0.12f * w, frontH * 0.8f, z0 + 0.02f},
                             mix(brick, trim, 0.35f)));
    }
    if (tier >= 1 && rng.chance(0.45f)) {
        // Front stoop/porch: a step and a little flat canopy on posts.
        const float px = 0.5f * (x0 + x1);
        merge(solid, makeBox({px - 0.07f, 0.0f, z0 - 0.05f}, {px + 0.07f, 0.035f, z0 + 0.02f}, trim));
        merge(solid, makeBox({px - 0.075f, 0.16f * heightMul, z0 - 0.055f},
                             {px + 0.075f, 0.18f * heightMul, z0 + 0.02f}, kDarkRoof));
    }
    // Recessed doorway on the street face.
    const float doorX = 0.5f * (x0 + x1);
    solid = csgSubtract(solid, makeBox({doorX - 0.045f, -0.01f, z0 - 0.02f},
                                       {doorX + 0.045f, 0.12f * heightMul + 0.03f, z0 + 0.04f}, brick));

    // Victorian sash windows on every mass, merged after the BSP ops.
    if (detail) {
        CsgMesh windows;
        for (int i = 0; i < numMasses; ++i) {
            const Mass& m = masses[i];
            addWindowsBox(windows, m.x0, 0.10f * heightMul, m.z0, m.x1, m.h - 0.03f, m.z1,
                          WindowStyle::kSash, kSashGlass, trim);
        }
        merge(solid, windows);
    }
    return solid;
}

CsgMesh build1890Commercial(float w, float d, int level, Rng& rng, bool detail) {
    const float mx = 0.08f * w, mz = 0.08f * d;
    const float x0 = mx, x1 = w - mx, z0 = mz, z1 = d - mz;
    const float wallH = 0.42f + 0.20f * static_cast<float>(level) + rng.uniform(0.0f, 0.12f);
    const Color3 brick = rng.pick(kBrickPool);
    const Color3 trim = rng.pick(kTrimPool);

    CsgMesh solid = makeBox({x0, 0.0f, z0}, {x1, wallH, z1}, brick);
    CsgMesh cornice = makeBox({x0 - 0.02f, wallH, z0 - 0.02f}, {x1 + 0.02f, wallH + 0.04f, z1 + 0.02f},
                              trim);
    solid = csgUnion(solid, cornice);
    solid = csgSubtract(solid, makeBox({x0 + 0.05f, wallH + 0.02f, z0 + 0.05f},
                                       {x1 - 0.05f, wallH + 0.5f, z1 - 0.05f}, brick));

    // Parapet ornament draw: plain lip, raised centre sign block, or corner posts.
    switch (rng.range(0, 2)) {
        case 1:
            merge(solid, makeBox({0.5f * (x0 + x1) - 0.10f, wallH, z0 - 0.005f},
                                 {0.5f * (x0 + x1) + 0.10f, wallH + 0.09f, z0 + 0.05f}, trim));
            break;
        case 2:
            merge(solid, makeBox({x0 - 0.01f, wallH, z0 - 0.01f}, {x0 + 0.05f, wallH + 0.07f, z0 + 0.05f},
                                 trim));
            merge(solid, makeBox({x1 - 0.05f, wallH, z0 - 0.01f}, {x1 + 0.01f, wallH + 0.07f, z0 + 0.05f},
                                 trim));
            break;
        default:
            break;
    }

    // Ground-floor storefront band + sign band above it on the street face.
    // The sign band draws from the awning pool — deep green / oxide red /
    // blue are period sign-painting colors; the deco accents are forty years
    // too modern for an 1890s frieze.
    merge(solid, makeBox({x0 + 0.02f, 0.0f, z0 - 0.012f}, {x1 - 0.02f, 0.15f, z0 + 0.03f}, trim));
    merge(solid, makeBox({x0 + 0.03f, 0.16f, z0 - 0.010f}, {x1 - 0.03f, 0.21f, z0 + 0.02f},
                         rng.pick(kAwningPool)));
    // Awnings: 0-3 colored canopies along the storefront.
    const int awnings = rng.range(0, 3);
    const Color3 awning = rng.pick(kAwningPool);
    for (int i = 0; i < awnings; ++i) {
        const float t0 = 0.08f + 0.30f * static_cast<float>(i);
        const float ax0 = x0 + t0 * (x1 - x0);
        merge(solid, makeBox({ax0, 0.115f, z0 - 0.030f}, {ax0 + 0.20f * (x1 - x0), 0.135f, z0 + 0.01f},
                             awning));
    }
    // Side alley notch on some blocks (reads as two distinct storefronts).
    if (rng.chance(0.30f)) {
        const float nx = x0 + (x1 - x0) * rng.uniform(0.55f, 0.75f);
        solid = csgSubtract(solid, makeBox({nx - 0.02f, -0.01f, z0 - 0.02f},
                                           {nx + 0.02f, wallH - 0.10f, z0 + 0.10f}, brick));
    }
    const float doorX = 0.5f * (x0 + x1);
    solid = csgSubtract(solid, makeBox({doorX - 0.05f, -0.01f, z0 - 0.03f},
                                       {doorX + 0.05f, 0.11f, z0 + 0.05f}, brick));

    // Upper-story sash windows above the storefront/sign bands.
    if (detail) {
        CsgMesh windows;
        addWindowsBox(windows, x0, 0.24f, z0, x1, wallH - 0.06f, z1, WindowStyle::kSash,
                      kSashGlass, trim);
        merge(solid, windows);
    }
    return solid;
}

CsgMesh build1890Industrial(float w, float d, int level, Rng& rng, bool detail) {
    const float mx = 0.06f * w, mz = 0.06f * d;
    const float x0 = mx, x1 = w - mx, z0 = mz, z1 = d - mz;
    const float wallH = 0.30f + 0.10f * static_cast<float>(level);
    const Color3 brick = rng.chance(0.5f) ? kSootBrick : rng.pick(kBrickPool);

    CsgMesh solid = makeBox({x0, 0.0f, z0}, {x1, wallH, z1}, brick);
    // Sawtooth roof: 2-4 parallel gables, ridge orientation drawn per mill.
    const int teeth = rng.range(2, 4);
    const bool ridgeAlongX = rng.chance(0.5f);
    if (ridgeAlongX) {
        const float strip = (z1 - z0) / static_cast<float>(teeth);
        for (int i = 0; i < teeth; ++i) {
            const float sz0 = z0 + strip * static_cast<float>(i);
            solid = csgUnion(solid, makeGablePrism(x0, sz0, x1, sz0 + strip, wallH, wallH + 0.015f,
                                                   wallH + 0.11f, kDarkRoof));
        }
    } else {
        const float strip = (x1 - x0) / static_cast<float>(teeth);
        for (int i = 0; i < teeth; ++i) {
            const float sx0 = x0 + strip * static_cast<float>(i);
            solid = csgUnion(solid, makeGablePrismZ(sx0, z0, sx0 + strip, z1, wallH, wallH + 0.015f,
                                                    wallH + 0.11f, kDarkRoof));
        }
    }
    // Smokestacks sunk through the roof.
    const int stacks = rng.range(1, 2);
    for (int i = 0; i < stacks; ++i) {
        const float sx = i == 0 ? x1 - 0.09f : x0 + 0.10f;
        const float sz = z1 - rng.uniform(0.08f, 0.18f);
        merge(solid, makeCylinder({sx, wallH - 0.05f, sz}, 0.040f,
                                  0.50f + 0.12f * static_cast<float>(level) + rng.uniform(0.0f, 0.10f),
                                  8, kSootBrick));
    }
    // Lean-to loading dock on some mills.
    if (rng.chance(0.4f)) {
        merge(solid, makeBox({x0 + 0.05f, 0.0f, z0 - 0.045f}, {x0 + 0.35f * w, wallH * 0.45f, z0 + 0.02f},
                             mix(brick, kDarkRoof, 0.3f)));
    }

    // Tall window bays between brick piers — the 19th-century mill signature
    // (full-height vertical strips, not squat domestic sash).
    if (detail) {
        CsgMesh windows;
        addWindowsBox(windows, x0, 0.07f, z0, x1, wallH - 0.05f, z1, WindowStyle::kRibbon,
                      kFactoryGlass, mix(brick, kDarkRoof, 0.4f));
        merge(solid, windows);
    }
    return solid;
}

// ── 1930s ───────────────────────────────────────────────────────────────────

CsgMesh build1930Tower(float w, float d, int level, int tier, bool commercial, Rng& rng,
                       bool detail) {
    const float heightMul = commercial ? 1.0f : tierHeightMul(tier);
    const Color3 body = rng.pick(kDecoBodyPool);
    const Color3 accent = rng.pick(kDecoAccentPool);

    const float bx0 = 0.07f * w, bx1 = w - 0.07f * w;
    const float bz0 = 0.07f * d, bz1 = d - 0.07f * d;
    float x0 = bx0, x1 = bx1, z0 = bz0, z1 = bz1;
    const int steps = std::min(4, 1 + level + (rng.chance(0.3f) ? 1 : 0));
    const float totalH = (0.55f * static_cast<float>(level) + rng.uniform(0.0f, 0.2f)) * heightMul;

    // Setback profile draw: classic centred ziggurat, front-biased "wedding
    // cake" (back edge stays flush), or a corner tower (steps hug one corner).
    enum Profile { kCentred, kFrontCake, kCornerTower };
    const Profile profile = static_cast<Profile>(rng.range(0, 2));

    CsgMesh solid;
    float y = 0.0f;
    float weightSum = 0.0f;
    for (int i = 0; i < steps; ++i) {
        weightSum += static_cast<float>(steps - i);
    }
    float baseTop = 0.0f;
    // Extents of the two lowest setback tiers, kept for the window ribbons.
    struct Tier {
        float x0, z0, x1, z1, y0, y1;
    };
    Tier tiers[2];
    int numTiers = 0;
    for (int i = 0; i < steps; ++i) {
        const float h = totalH * static_cast<float>(steps - i) / weightSum;
        CsgMesh step = makeBox({x0, y, z0}, {x1, y + h, z1}, body);
        if (i == 0) {
            solid = std::move(step);
            baseTop = h;
        } else {
            solid = csgUnion(solid, step);
        }
        if (i < 2) tiers[numTiers++] = {x0, z0, x1, z1, y, y + h};
        y += h;
        const float insetX = rng.uniform(0.12f, 0.18f) * (x1 - x0) * 0.5f;
        const float insetZ = rng.uniform(0.12f, 0.18f) * (z1 - z0) * 0.5f;
        switch (profile) {
            case kFrontCake:
                x0 += insetX;
                x1 -= insetX;
                z0 += 2.0f * insetZ;  // street edge steps back; rear stays flush
                break;
            case kCornerTower:
                x1 -= 2.0f * insetX;  // steps hug the (x0, z0) corner
                z1 -= 2.0f * insetZ;
                break;
            default:
                x0 += insetX;
                x1 -= insetX;
                z0 += insetZ;
                z1 -= insetZ;
                break;
        }
    }

    // Art-deco chamfers on two opposite corners of the base step (drawn).
    if (rng.chance(0.6f)) {
        solid = csgSubtract(solid, cornerChamferCutter(bx0, bz0, 0.045f, baseTop + 0.05f));
        solid = csgSubtract(solid, cornerChamferCutter(bx1, bz1, 0.045f, baseTop + 0.05f));
    }

    // Vertical pilaster ribs on the street face of the base step.
    const int ribs = rng.range(0, 2 + level);
    for (int i = 0; i < ribs; ++i) {
        const float t = (static_cast<float>(i) + 0.5f) / static_cast<float>(std::max(ribs, 1));
        const float rx = bx0 + 0.1f * w + t * (bx1 - bx0 - 0.2f * w);
        merge(solid, makeBox({rx - 0.011f, 0.02f, bz0 - 0.015f}, {rx + 0.011f, baseTop * 0.95f, bz0 + 0.02f},
                             accent));
    }
    // Two-tone base band on some towers.
    if (rng.chance(0.4f)) {
        merge(solid, makeBox({bx0 - 0.008f, 0.0f, bz0 - 0.008f},
                             {bx1 + 0.008f, 0.10f, bz1 + 0.008f}, mix(body, accent, 0.5f)));
    }

    // Crown draw: flat cap, stepped cap, spire, or corner finials.
    const float cx = 0.5f * (x0 + x1), cz = 0.5f * (z0 + z1);
    const float capW = 0.5f * (x1 - x0) * 0.7f, capD = 0.5f * (z1 - z0) * 0.7f;
    switch (rng.range(0, 3)) {
        case 1:
            merge(solid, makeBox({cx - capW, y - 0.01f, cz - capD}, {cx + capW, y + 0.045f, cz + capD},
                                 accent));
            merge(solid, makeBox({cx - capW * 0.55f, y + 0.045f, cz - capD * 0.55f},
                                 {cx + capW * 0.55f, y + 0.085f, cz + capD * 0.55f}, accent));
            break;
        case 2:
            merge(solid, makeBox({cx - capW, y - 0.01f, cz - capD}, {cx + capW, y + 0.04f, cz + capD},
                                 accent));
            merge(solid, makeBox({cx - 0.015f, y, cz - 0.015f},
                                 {cx + 0.015f, y + 0.20f + 0.10f * static_cast<float>(level), cz + 0.015f},
                                 accent));
            break;
        case 3:
            for (int fx = 0; fx < 2; ++fx) {
                for (int fz = 0; fz < 2; ++fz) {
                    const float px = fx == 0 ? x0 + 0.015f : x1 - 0.015f;
                    const float pz = fz == 0 ? z0 + 0.015f : z1 - 0.015f;
                    merge(solid, makeBox({px - 0.014f, y - 0.02f, pz - 0.014f},
                                         {px + 0.014f, y + 0.07f, pz + 0.014f}, accent));
                }
            }
            break;
        default:
            merge(solid, makeBox({cx - capW, y - 0.01f, cz - capD}, {cx + capW, y + 0.045f, cz + capD},
                                 accent));
            break;
    }

    // Deco window ribbons: continuous vertical strips running up the two
    // lowest setback tiers (the corner chamfers only nick the base corners,
    // clear of the ribbon margins).
    if (detail) {
        CsgMesh windows;
        for (int i = 0; i < numTiers; ++i) {
            const Tier& t = tiers[i];
            addWindowsBox(windows, t.x0, t.y0 + (i == 0 ? 0.13f : 0.04f), t.z0, t.x1,
                          t.y1 - 0.03f, t.z1, WindowStyle::kRibbon, kRibbonGlass, accent);
        }
        merge(solid, windows);
    }
    return solid;
}

CsgMesh build1930Industrial(float w, float d, int level, Rng& rng, bool detail) {
    const float mx = 0.06f * w, mz = 0.06f * d;
    const float x0 = mx, x1 = w - mx, z0 = mz, z1 = d - mz;
    const float wallH = 0.45f + 0.15f * static_cast<float>(level);
    const Color3 brick = rng.chance(0.5f) ? rng.pick(kBrickPool) : kSootBrick;

    CsgMesh solid = makeBox({x0, 0.0f, z0}, {x1, wallH, z1}, brick);
    // Roofline draw: stepped street parapet or a raised monitor roof.
    if (rng.chance(0.5f)) {
        solid = csgUnion(solid, makeBox({x0 + 0.18f * w, wallH, z0}, {x1 - 0.18f * w, wallH + 0.08f, z0 + 0.10f},
                                        brick));
    } else {
        merge(solid, makeBox({x0 + 0.12f, wallH - 0.02f, z0 + 0.30f * (z1 - z0)},
                             {x1 - 0.12f, wallH + 0.09f, z1 - 0.30f * (z1 - z0)},
                             mix(brick, kMullion, 0.4f)));
    }
    // Rooftop water tank on a pedestal (drawn), plus 1-2 smokestacks.
    if (rng.chance(0.7f)) {
        const float tx = x0 + 0.16f, tz = z1 - 0.16f;
        merge(solid, makeBox({tx - 0.05f, wallH - 0.03f, tz - 0.05f}, {tx + 0.05f, wallH + 0.07f, tz + 0.05f},
                             kMullion));
        merge(solid, makeCylinder({tx, wallH + 0.07f, tz}, 0.062f, 0.14f, 8, rng.pick(kBrickPool)));
    }
    const int stacks = rng.range(1, 2);
    for (int i = 0; i < stacks; ++i) {
        merge(solid, makeCylinder({x1 - 0.12f - 0.12f * static_cast<float>(i), wallH - 0.05f, z0 + 0.14f},
                                  0.045f, 0.60f + 0.10f * static_cast<float>(level) + rng.uniform(0.0f, 0.12f),
                                  8, kSootBrick));
    }

    // Big factory glazing: two rows of wide multipane windows.
    if (detail) {
        CsgMesh windows;
        addWindowsBox(windows, x0, 0.06f, z0, x1, wallH - 0.08f, z1, WindowStyle::kBand,
                      kFactoryGlass, kMullion);
        merge(solid, windows);
    }
    return solid;
}

// ── 1960s ───────────────────────────────────────────────────────────────────

CsgMesh build1960Tower(float w, float d, int level, int tier, bool commercial, Rng& rng,
                       bool detail) {
    const float heightMul = commercial ? 1.0f : tierHeightMul(tier);
    const float h = (1.1f + 0.5f * static_cast<float>(level) + rng.uniform(0.0f, 0.25f)) * heightMul;
    const Color3 skin = rng.chance(0.78f) ? rng.pick(kGlassPool) : kGreyPanel;
    const Color3 band = rng.chance(0.5f) ? kConcrete : kGreyPanel;

    // Massing draw: single slab, cross plan, podium + tower, or twin slabs.
    enum Massing { kSlab, kCross, kPodium, kTwinSlabs };
    const Massing massing = static_cast<Massing>(rng.range(0, 3));

    float slabW = w * 0.72f, slabD = d * 0.36f;
    float x0 = 0.5f * (w - slabW), x1 = x0 + slabW;
    float z0 = d * 0.30f, z1 = z0 + slabD;
    float towerBase = 0.0f;
    CsgMesh solid;
    switch (massing) {
        case kCross: {
            solid = makeBox({x0, 0.0f, z0}, {x1, h, z1}, skin);
            const float cx = 0.5f * w;
            const float armW = w * 0.26f;
            solid = csgUnion(solid, makeBox({cx - 0.5f * armW, 0.0f, d * 0.16f},
                                            {cx + 0.5f * armW, h * 0.92f, d - 0.10f * d}, skin));
            break;
        }
        case kPodium: {
            // Full-lot low podium with the slab rising from it.
            const float px0 = 0.06f * w, px1 = w - 0.06f * w;
            const float pz0 = 0.06f * d, pz1 = d - 0.06f * d;
            towerBase = 0.14f;
            solid = makeBox({px0, 0.0f, pz0}, {px1, towerBase, pz1}, kConcrete);
            slabW = w * rng.uniform(0.42f, 0.55f);
            x0 = 0.5f * (w - slabW);
            x1 = x0 + slabW;
            z0 = d * 0.28f;
            z1 = z0 + d * 0.40f;
            solid = csgUnion(solid, makeBox({x0, towerBase, z0}, {x1, h, z1}, skin));
            break;
        }
        case kTwinSlabs: {
            slabW = w * 0.34f;
            const float gap = w * 0.12f;
            const float ax0 = 0.5f * w - slabW - 0.5f * gap;
            const float bx0 = 0.5f * w + 0.5f * gap;
            const float hB = h * rng.uniform(0.72f, 0.9f);
            solid = makeBox({ax0, 0.0f, z0}, {ax0 + slabW, h, z1}, skin);
            merge(solid, makeBox({bx0, 0.0f, z0}, {bx0 + slabW, hB, z1}, mix(skin, kGreyPanel, 0.3f)));
            // Low link block between the pair.
            merge(solid, makeBox({ax0 + slabW - 0.01f, 0.0f, z0 + 0.25f * slabD},
                                 {bx0 + 0.01f, 0.12f, z1 - 0.25f * slabD}, kConcrete));
            x0 = ax0;
            x1 = bx0 + slabW;
            break;
        }
        default:
            if (rng.chance(0.4f)) {  // rotate the slab the other way sometimes
                slabW = w * 0.36f;
                slabD = d * 0.72f;
                x0 = w * 0.30f;
                x1 = x0 + slabW;
                z0 = 0.5f * (d - slabD);
                z1 = z0 + slabD;
            }
            solid = makeBox({x0, 0.0f, z0}, {x1, h, z1}, skin);
            break;
    }

    // Recessed pilotis lobby along the street face (skipped on podium towers,
    // whose ground floor is the podium itself).
    if (massing != kPodium) {
        solid = csgSubtract(solid, makeBox({x0 + 0.03f, -0.01f, z0 - 0.02f}, {x1 - 0.03f, 0.13f, z0 + 0.07f},
                                           skin));
        const int columns = 3;
        for (int i = 0; i < columns; ++i) {
            const float t = (static_cast<float>(i) + 0.5f) / static_cast<float>(columns);
            const float colX = x0 + 0.05f + t * (x1 - x0 - 0.10f);
            merge(solid, makeBox({colX - 0.012f, 0.0f, z0 + 0.01f}, {colX + 0.012f, 0.14f, z0 + 0.045f},
                                 kConcrete));
        }
    }

    // Horizontal spandrel bands read as curtain-wall floor lines; spacing and
    // thickness are drawn so neighbouring towers don't repeat.
    const float bandSpacing = rng.uniform(0.24f, 0.34f);
    const int bands = std::min(9, static_cast<int>(h / bandSpacing));
    const float bandThick = rng.uniform(0.022f, 0.036f);
    for (int i = 1; i <= bands; ++i) {
        const float by = towerBase + 0.16f +
                         (h - towerBase - 0.28f) * static_cast<float>(i) / static_cast<float>(bands + 1);
        merge(solid, makeBox({x0 - 0.008f, by, z0 - 0.008f}, {x1 + 0.008f, by + bandThick, z1 + 0.008f},
                             band));
    }

    // Roof plant boxes.
    const int plants = rng.range(1, 2);
    for (int i = 0; i < plants; ++i) {
        const float pw = (x1 - x0) * rng.uniform(0.18f, 0.30f);
        const float px = rng.uniform(x0 + 0.02f, x1 - pw - 0.02f);
        merge(solid, makeBox({px, h - 0.02f, z0 + (z1 - z0) * 0.2f},
                             {px + pw, h + rng.uniform(0.05f, 0.09f), z0 + (z1 - z0) * 0.8f}, kGreyPanel));
    }

    // Curtain-wall mullion grid over the glass skin. Twin slabs span a gap, so
    // they get their own grids; every other massing decorates the main slab.
    if (detail) {
        CsgMesh windows;
        const float wy0 = towerBase + 0.16f, wy1 = h - 0.05f;
        if (massing == kTwinSlabs) {
            const float slabSpan = (x1 - x0 - w * 0.12f) * 0.5f;  // recompute each slab's width
            addWindowsBox(windows, x0, wy0, z0, x0 + slabSpan, wy1, z1, WindowStyle::kMullion,
                          kMullion, kMullion);
            // The shorter slab tops out at 0.72h-0.9h; cap its grid below the
            // minimum so mullions never float past the roofline.
            addWindowsBox(windows, x1 - slabSpan, wy0, z0, x1, wy1 * 0.70f, z1,
                          WindowStyle::kMullion, kMullion, kMullion);
        } else {
            addWindowsBox(windows, x0, wy0, z0, x1, wy1, z1, WindowStyle::kMullion, kMullion,
                          kMullion);
        }
        merge(solid, windows);
    }
    return solid;
}

CsgMesh build1960Industrial(float w, float d, int level, Rng& rng, bool detail) {
    const float mx = 0.06f * w, mz = 0.10f * d;
    const float x0 = mx, x1 = w - mx, z0 = mz, z1 = d - 0.30f * d;
    const float wallH = 0.26f + 0.08f * static_cast<float>(level);

    CsgMesh solid = makeBox({x0, 0.0f, z0}, {x1, wallH, z1}, kConcrete);
    // Clerestory glass strip along the roof ridge line.
    merge(solid, makeBox({x0 + 0.06f, wallH - 0.02f, 0.5f * (z0 + z1) - 0.05f},
                         {x1 - 0.06f, wallH + 0.06f, 0.5f * (z0 + z1) + 0.05f}, rng.pick(kGlassPool)));
    // Yard draw: squat tank farm or a row of tall silos behind the shed.
    if (rng.chance(0.5f)) {
        const int tanks = rng.range(2, 4);
        for (int i = 0; i < tanks; ++i) {
            const float t = (static_cast<float>(i) + 0.5f) / static_cast<float>(tanks);
            const float cx = x0 + 0.08f + t * (x1 - x0 - 0.16f);
            merge(solid, makeCylinder({cx, 0.0f, z1 + 0.12f}, 0.048f + rng.uniform(0.0f, 0.015f),
                                      0.18f + 0.05f * static_cast<float>(level), 8, kGreyPanel));
        }
    } else {
        const int silos = rng.range(2, 3);
        for (int i = 0; i < silos; ++i) {
            const float t = (static_cast<float>(i) + 0.5f) / static_cast<float>(silos);
            const float cx = x0 + 0.10f + t * (x1 - x0 - 0.20f);
            merge(solid, makeCylinder({cx, 0.0f, z1 + 0.12f}, 0.040f,
                                      0.42f + 0.08f * static_cast<float>(level), 8, kConcrete));
        }
    }
    // Flat entrance canopy on the street side.
    if (rng.chance(0.5f)) {
        merge(solid, makeBox({x0 + 0.30f * (x1 - x0), 0.12f, z0 - 0.05f},
                             {x0 + 0.70f * (x1 - x0), 0.145f, z0 + 0.02f}, kGreyPanel));
    }

    // One clean glazing band around the shed walls.
    if (detail) {
        CsgMesh windows;
        addWindowsBox(windows, x0, wallH * 0.45f, z0, x1, wallH * 0.80f, z1, WindowStyle::kBand,
                      rng.pick(kGlassPool), kMullion);
        merge(solid, windows);
    }
    return solid;
}

}  // namespace

TriMesh generateBuilding(const BuildingDesc& desc) {
    // Mix every descriptor field into the seed so distinct descs decorrelate
    // even when the caller passes the same seed.
    std::uint32_t seed = desc.seed;
    seed ^= (static_cast<std::uint32_t>(desc.era) + 1u) * 0x9E3779B9u;
    seed ^= (static_cast<std::uint32_t>(desc.kind) + 1u) * 0x85EBCA6Bu;
    seed ^= static_cast<std::uint32_t>(desc.level) * 0xC2B2AE35u;
    seed ^= static_cast<std::uint32_t>(desc.wealthTier) * 0x27D4EB2Fu;
    Rng rng(seed);

    const int level = std::clamp(desc.level, 1, 3);
    const int tier = std::clamp(desc.wealthTier, 0, 2);
    const float w = desc.lotWidth;
    const float d = desc.lotDepth;
    // The window pass runs after every massing draw in each builder, so both
    // detail tiers produce bit-identical massing from the same seed — a LOD
    // swap never changes a building's silhouette.
    const bool detail = desc.detail != 0;

    CsgMesh solid;
    switch (desc.era) {
        case Era::E1890s:
            solid = desc.kind == BuildingKind::Residential
                        ? build1890Residential(w, d, level, tier, rng, detail)
                    : desc.kind == BuildingKind::Commercial
                        ? build1890Commercial(w, d, level, rng, detail)
                        : build1890Industrial(w, d, level, rng, detail);
            break;
        case Era::E1930s:
            solid = desc.kind == BuildingKind::Industrial
                        ? build1930Industrial(w, d, level, rng, detail)
                        : build1930Tower(w, d, level, tier,
                                         desc.kind == BuildingKind::Commercial, rng, detail);
            break;
        case Era::E1960s:
            solid = desc.kind == BuildingKind::Industrial
                        ? build1960Industrial(w, d, level, rng, detail)
                        : build1960Tower(w, d, level, tier,
                                         desc.kind == BuildingKind::Commercial, rng, detail);
            break;
    }
    return triangulate(solid);
}

}  // namespace odai::procgen
