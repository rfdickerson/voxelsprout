#include "procgen/city_terrain.h"

#include <algorithm>
#include <cmath>

#include "math/noise.h"
#include "procgen/rng.h"

namespace odai::procgen {

namespace {

// hashNoise has no seed parameter, so each noise consumer samples at a
// seed-derived domain offset instead (±1000 keeps the lattice well-behaved).
float domainOffset(std::uint32_t seed, std::uint32_t salt) {
    return (static_cast<float>(hash2d(static_cast<int>(seed), static_cast<int>(salt), salt) & 0xFFFFu) /
                65535.0f -
            0.5f) *
           2000.0f;
}

struct Grid {
    int w = 0, h = 0;
    std::vector<std::uint8_t>* water = nullptr;

    bool inBounds(int c, int r) const { return c >= 0 && c < w && r >= 0 && r < h; }
    void stamp(int c, int r) {
        if (inBounds(c, r)) (*water)[static_cast<std::size_t>(r) * w + c] = 1u;
    }
    bool wet(int c, int r) const {
        return inBounds(c, r) && (*water)[static_cast<std::size_t>(r) * w + c] != 0u;
    }
};

// Carve the river as a walk from one map edge to the opposite edge. Each step
// stamps a full-width brush on both the current and next row (or column) over
// their shared lateral range, so 4-connectivity holds no matter how sharply the
// centerline swings.
void carveRiver(Grid& g, Rng& rng, const CityTerrainParams& params, std::uint32_t seed,
                std::vector<std::pair<short, short>>& riverPath) {
    const bool vertical = rng.chance(0.5f);
    const int length = vertical ? g.h : g.w;
    const int lateralSpan = vertical ? g.w : g.h;

    const float entry = rng.uniform(lateralSpan * 0.22f, lateralSpan * 0.78f);
    const float exit = rng.uniform(lateralSpan * 0.22f, lateralSpan * 0.78f);
    const float meanderAmp = lateralSpan * 0.14f;
    const float ox = domainOffset(seed, 0xA11CEu);
    const float oy = domainOffset(seed, 0xB0B0Eu);

    std::vector<int> center(static_cast<std::size_t>(length));
    std::vector<int> half(static_cast<std::size_t>(length));
    for (int i = 0; i < length; ++i) {
        const float t = static_cast<float>(i) / static_cast<float>(length - 1);
        const float base = entry + (exit - entry) * t;
        const float meander = odai::math::fbm2(static_cast<float>(i) * 0.11f + ox, oy) * meanderAmp;
        center[static_cast<std::size_t>(i)] = static_cast<int>(std::round(base + meander));
        // Width swells and narrows smoothly along the run.
        const float wNoise =
            odai::math::valueNoise(static_cast<float>(i) * 0.07f + oy, ox) * 0.5f + 0.5f;
        half[static_cast<std::size_t>(i)] = std::clamp(
            params.riverWidthMin +
                static_cast<int>(std::round(wNoise * static_cast<float>(params.riverWidthMax -
                                                                        params.riverWidthMin))),
            1, std::max(1, params.riverWidthMax));
    }

    const auto stampSpan = [&](int along, int lo, int hi) {
        for (int c = lo; c <= hi; ++c) {
            if (vertical) g.stamp(c, along);
            else g.stamp(along, c);
        }
    };

    for (int i = 0; i < length; ++i) {
        const int c0 = center[static_cast<std::size_t>(i)];
        const int hw = half[static_cast<std::size_t>(i)];
        int lo = c0 - hw, hi = c0 + hw;
        if (i + 1 < length) {
            // Bridge to the next step's center so consecutive rows always share
            // a wet column.
            const int c1 = center[static_cast<std::size_t>(i + 1)];
            lo = std::min(lo, std::min(c0, c1));
            hi = std::max(hi, std::max(c0, c1));
        }
        stampSpan(i, lo, hi);
        const short pc = static_cast<short>(std::clamp(vertical ? c0 : i, 0, g.w - 1));
        const short pr = static_cast<short>(std::clamp(vertical ? i : c0, 0, g.h - 1));
        riverPath.emplace_back(pc, pr);
    }
}

void addLakes(Grid& g, Rng& rng, const CityTerrainParams& params, std::uint32_t seed) {
    const int count = params.lakeMax > 0 ? rng.range(0, params.lakeMax) : 0;
    for (int i = 0; i < count; ++i) {
        const float cx = rng.uniform(6.0f, static_cast<float>(g.w - 7));
        const float cy = rng.uniform(6.0f, static_cast<float>(g.h - 7));
        const float radius = rng.uniform(3.0f, 6.0f);
        const float ox = domainOffset(seed, 0x1A7Eu + static_cast<std::uint32_t>(i));
        for (int r = 0; r < g.h; ++r) {
            for (int c = 0; c < g.w; ++c) {
                const float dx = static_cast<float>(c) - cx;
                const float dy = static_cast<float>(r) - cy;
                const float dist = std::sqrt(dx * dx + dy * dy);
                const float wobble =
                    odai::math::fbm2(std::atan2(dy, dx) * 1.6f + ox, dist * 0.35f) * 0.35f;
                if (dist < radius * (0.8f + wobble)) g.stamp(c, r);
            }
        }
    }
}

void addCoast(Grid& g, Rng& rng, const CityTerrainParams& params, std::uint32_t seed) {
    if (!rng.chance(params.coastChance)) return;
    const int edge = rng.range(0, 3);  // 0=W 1=E 2=N 3=S
    const float ox = domainOffset(seed, 0xC0A57u);
    const int span = (edge <= 1) ? g.h : g.w;
    for (int i = 0; i < span; ++i) {
        const float n = odai::math::fbm2(static_cast<float>(i) * 0.13f + ox, ox * 0.5f) * 0.5f + 0.5f;
        const int depth = 2 + static_cast<int>(std::round(n * 3.0f));
        for (int d = 0; d < depth; ++d) {
            if (edge == 0) g.stamp(d, i);
            else if (edge == 1) g.stamp(g.w - 1 - d, i);
            else if (edge == 2) g.stamp(i, d);
            else g.stamp(i, g.h - 1 - d);
        }
    }
}

void buildForestMask(CityTerrain& t, const CityTerrainParams& params, std::uint32_t seed) {
    const float ox = domainOffset(seed, 0xF03E57u);
    const float oy = domainOffset(seed, 0x7EEE5u);
    t.forest.assign(static_cast<std::size_t>(t.width) * t.height, 0.0f);
    for (int r = 0; r < t.height; ++r) {
        for (int c = 0; c < t.width; ++c) {
            const float n = odai::math::fbm3(static_cast<float>(c) * params.forestFreq + ox,
                                             static_cast<float>(r) * params.forestFreq + oy);
            t.forest[static_cast<std::size_t>(r) * t.width + c] =
                std::clamp(n * 0.5f + 0.5f, 0.0f, 1.0f);
        }
    }
}

// Flood-fill helper over a predicate; returns component sizes and lets the
// caller inspect membership via the label grid (-1 = unvisited/excluded).
template <typename Pred>
std::vector<int> labelComponents(int w, int h, std::vector<int>& labels, Pred pred) {
    labels.assign(static_cast<std::size_t>(w) * h, -1);
    std::vector<int> sizes;
    std::vector<std::pair<int, int>> stack;
    for (int r = 0; r < h; ++r) {
        for (int c = 0; c < w; ++c) {
            const std::size_t idx = static_cast<std::size_t>(r) * w + c;
            if (labels[idx] != -1 || !pred(c, r)) continue;
            const int label = static_cast<int>(sizes.size());
            int size = 0;
            stack.push_back({c, r});
            labels[idx] = label;
            while (!stack.empty()) {
                const auto [cc, cr] = stack.back();
                stack.pop_back();
                ++size;
                const int nc[4] = {cc - 1, cc + 1, cc, cc};
                const int nr[4] = {cr, cr, cr - 1, cr + 1};
                for (int k = 0; k < 4; ++k) {
                    if (nc[k] < 0 || nc[k] >= w || nr[k] < 0 || nr[k] >= h) continue;
                    const std::size_t nidx = static_cast<std::size_t>(nr[k]) * w + nc[k];
                    if (labels[nidx] != -1 || !pred(nc[k], nr[k])) continue;
                    labels[nidx] = label;
                    stack.push_back({nc[k], nr[k]});
                }
            }
            sizes.push_back(size);
        }
    }
    return sizes;
}

// True if some single water component touches two opposite map edges (the
// carved river guarantees this unless a later stage broke it, which lakes and
// coast never do — they only add water).
bool waterSpansMap(const CityTerrain& t) {
    std::vector<int> labels;
    labelComponents(t.width, t.height, labels, [&](int c, int r) {
        return t.water[static_cast<std::size_t>(r) * t.width + c] != 0u;
    });
    std::vector<std::uint8_t> touchesW(64, 0), touchesE(64, 0), touchesN(64, 0), touchesS(64, 0);
    const auto mark = [&](std::vector<std::uint8_t>& v, int label) {
        if (label >= 0) {
            if (static_cast<std::size_t>(label) >= v.size()) v.resize(static_cast<std::size_t>(label) + 1, 0);
            v[static_cast<std::size_t>(label)] = 1;
        }
    };
    for (int r = 0; r < t.height; ++r) {
        mark(touchesW, labels[static_cast<std::size_t>(r) * t.width]);
        mark(touchesE, labels[static_cast<std::size_t>(r) * t.width + t.width - 1]);
    }
    for (int c = 0; c < t.width; ++c) {
        mark(touchesN, labels[static_cast<std::size_t>(c)]);
        mark(touchesS, labels[static_cast<std::size_t>(t.height - 1) * t.width + c]);
    }
    const std::size_t n = std::max({touchesW.size(), touchesE.size(), touchesN.size(), touchesS.size()});
    for (std::size_t i = 0; i < n; ++i) {
        const auto at = [&](const std::vector<std::uint8_t>& v) {
            return i < v.size() && v[i] != 0u;
        };
        if ((at(touchesW) && at(touchesE)) || (at(touchesN) && at(touchesS))) return true;
    }
    return false;
}

// Score every candidate city-site window and pick the best. A viable window is
// mostly grass, inside the largest land component, near water (waterfront
// desirability), and away from the map edge. Returns the best score, or a
// negative value if no window qualifies.
float pickSite(const CityTerrain& t, short& outC, short& outR) {
    const int winW = std::min(24, t.width);
    const int winH = std::min(16, t.height);
    std::vector<int> labels;
    const std::vector<int> sizes = labelComponents(t.width, t.height, labels, [&](int c, int r) {
        return t.water[static_cast<std::size_t>(r) * t.width + c] == 0u;
    });
    int bigLabel = -1, bigSize = 0;
    for (std::size_t i = 0; i < sizes.size(); ++i) {
        if (sizes[i] > bigSize) {
            bigSize = sizes[i];
            bigLabel = static_cast<int>(i);
        }
    }
    if (bigLabel < 0) return -1.0f;

    float bestScore = -1.0f;
    for (int r0 = 0; r0 + winH <= t.height; r0 += 2) {
        for (int c0 = 0; c0 + winW <= t.width; c0 += 2) {
            int grass = 0;
            for (int r = r0; r < r0 + winH; ++r) {
                for (int c = c0; c < c0 + winW; ++c) {
                    if (labels[static_cast<std::size_t>(r) * t.width + c] == bigLabel) ++grass;
                }
            }
            if (grass < (winW * winH * 7) / 10) continue;  // needs 70% buildable

            // Waterfront bonus: water tiles in a 2-tile ring around the window.
            int waterRing = 0;
            for (int r = r0 - 2; r < r0 + winH + 2; ++r) {
                for (int c = c0 - 2; c < c0 + winW + 2; ++c) {
                    if (c < 0 || c >= t.width || r < 0 || r >= t.height) continue;
                    if (c >= c0 && c < c0 + winW && r >= r0 && r < r0 + winH) continue;
                    if (t.water[static_cast<std::size_t>(r) * t.width + c] != 0u) ++waterRing;
                }
            }

            const float cx = static_cast<float>(c0) + winW * 0.5f;
            const float cy = static_cast<float>(r0) + winH * 0.5f;
            const float dx = cx - t.width * 0.5f;
            const float dy = cy - t.height * 0.5f;
            const float centerDist = std::sqrt(dx * dx + dy * dy);

            const float score = static_cast<float>(grass) +
                                0.6f * static_cast<float>(std::min(waterRing, 40)) -
                                2.0f * centerDist;
            if (score > bestScore) {
                bestScore = score;
                outC = static_cast<short>(c0 + winW / 2);
                outR = static_cast<short>(r0 + winH / 2);
            }
        }
    }
    return bestScore;
}

struct Attempt {
    CityTerrain terrain;
    float score = -1e9f;
    bool valid = false;
};

Attempt generateAttempt(const CityTerrainDesc& desc, std::uint32_t seed) {
    Attempt a;
    CityTerrain& t = a.terrain;
    t.width = desc.width;
    t.height = desc.height;
    t.water.assign(static_cast<std::size_t>(t.width) * t.height, 0u);

    Rng rng(seed);
    Grid g{t.width, t.height, &t.water};
    carveRiver(g, rng, desc.params, seed, t.riverPath);
    addLakes(g, rng, desc.params, seed);
    addCoast(g, rng, desc.params, seed);
    buildForestMask(t, desc.params, seed);

    // Invariants.
    int land = 0;
    for (const std::uint8_t wtr : t.water) land += (wtr == 0u) ? 1 : 0;
    const float landFrac = static_cast<float>(land) / static_cast<float>(t.water.size());

    std::vector<int> labels;
    const std::vector<int> sizes = labelComponents(t.width, t.height, labels, [&](int c, int r) {
        return t.water[static_cast<std::size_t>(r) * t.width + c] == 0u;
    });
    int bigSize = 0;
    for (const int s : sizes) bigSize = std::max(bigSize, s);
    const int minComponent = (t.width * t.height * 3) / 8;

    const float siteScore = pickSite(t, t.siteC, t.siteR);
    const bool spans = waterSpansMap(t);

    a.valid = landFrac >= desc.params.landMin && bigSize >= minComponent && siteScore > 0.0f &&
              spans;
    a.score = landFrac * 100.0f + (siteScore > 0.0f ? siteScore * 0.1f : -50.0f) +
              (spans ? 0.0f : -100.0f);
    t.valid = a.valid;
    return a;
}

}  // namespace

CityTerrain generateCityTerrain(const CityTerrainDesc& desc) {
    Attempt best;
    const std::uint32_t baseSeed = desc.seed ? desc.seed : 1u;
    for (std::uint32_t k = 0; k < 8; ++k) {
        Attempt a = generateAttempt(desc, baseSeed + k * 0x9E3779B9u);
        if (a.valid) return std::move(a.terrain);
        if (a.score > best.score) best = std::move(a);
    }
    return std::move(best.terrain);  // valid == false; game runs on an awkward map
}

}  // namespace odai::procgen
