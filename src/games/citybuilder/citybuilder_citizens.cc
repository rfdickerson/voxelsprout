#include "games/citybuilder/citybuilder_citizens.h"

#include <algorithm>
#include <unordered_set>
#include <utility>

namespace odai::games::citybuilder {

namespace {

constexpr int kMaxRoster = 96;
constexpr int kMinRoster = 6;
constexpr std::size_t kTickerCapacity = 24;
constexpr int kMaxStoriesPerMonth = 3;

void replaceAll(std::string& text, const std::string& token, const std::string& value) {
    for (std::size_t pos = text.find(token); pos != std::string::npos;
         pos = text.find(token, pos + value.size())) {
        text.replace(pos, token.size(), value);
    }
}

std::uint32_t packTile(short c, short r) {
    return (static_cast<std::uint32_t>(static_cast<std::uint16_t>(c)) << 16) |
           static_cast<std::uint16_t>(r);
}

}  // namespace

void CitizenSim::configure(odai::citybuilder::CityScriptHost* script, std::uint32_t worldSeed,
                           float storyBoost) {
    m_script = script;
    m_worldSeed = worldSeed ? worldSeed : 1u;
    m_storyBoost = storyBoost;
    m_rng = odai::procgen::Rng(m_worldSeed ^ 0xC171F0u);
}

void CitizenSim::pushTicker(std::string text, TickerKind kind, short c, short r) {
    m_ticker.push_back(TickerItem{std::move(text), kind, c, r, 0.0f});
    while (m_ticker.size() > kTickerCapacity) m_ticker.pop_front();
}

bool CitizenSim::hasTag(const Citizen& cz, const std::string& tag) const {
    if (tag == "fit") return (cz.traits & kTraitFit) != 0;
    if (tag == "parent") return (cz.traits & kTraitParent) != 0;
    if (tag == "nightowl") return (cz.traits & kTraitNightOwl) != 0;
    if (tag == "gossip") return (cz.traits & kTraitGossip) != 0;
    if (tag == "married") return cz.spouse >= 0;
    if (tag == "affair") return cz.affair >= 0;
    return false;  // unknown tag never matches — keeps modded typos harmless
}

const odai::citybuilder::StoryTemplate* CitizenSim::pickStory(const std::string& kind,
                                                              const Citizen& cz) {
    if (m_script == nullptr) return nullptr;
    float totalWeight = 0.0f;
    std::vector<const odai::citybuilder::StoryTemplate*> eligible;
    for (const odai::citybuilder::StoryTemplate& tpl : m_script->stories()) {
        if (tpl.kind != kind) continue;
        bool ok = true;
        for (const std::string& condition : tpl.conditions) {
            if (!hasTag(cz, condition)) {
                ok = false;
                break;
            }
        }
        if (!ok) continue;
        eligible.push_back(&tpl);
        totalWeight += std::max(0.01f, tpl.weight);
    }
    if (eligible.empty()) return nullptr;
    float roll = m_rng.uniform(0.0f, totalWeight);
    for (const odai::citybuilder::StoryTemplate* tpl : eligible) {
        roll -= std::max(0.01f, tpl->weight);
        if (roll <= 0.0f) return tpl;
    }
    return eligible.back();
}

std::string CitizenSim::interpolate(const std::string& tpl, const Citizen& a, const Citizen* b,
                                    const Destination* place, const std::string& street) const {
    std::string text = tpl;
    replaceAll(text, "{a}", a.fullName());
    replaceAll(text, "{b}", b != nullptr ? b->fullName() : "a mysterious stranger");
    replaceAll(text, "{family}", a.lastName);
    replaceAll(text, "{place}", place != nullptr ? place->name : "the town square");
    replaceAll(text, "{street}", street.empty() ? "Main St" : street);
    return text;
}

void CitizenSim::endWorkWeek() {
    for (Citizen& cz : m_roster) cz.atWork = false;
}

void CitizenSim::emitWeekendStory(const Destination& dest, const std::string& street) {
    if (m_roster.empty()) return;
    const Citizen& a = m_roster[m_rng.next() % static_cast<std::uint32_t>(m_roster.size())];
    if (const auto* tpl = pickStory("weekend", a)) {
        pushTicker(interpolate(tpl->text, a, nullptr, &dest, street), TickerKind::Life, dest.c,
                   dest.r);
    }
}

void CitizenSim::emitOpening(const Destination& dest, const std::string& street) {
    // Openings interpolate against a synthetic "citizen" when the roster is
    // empty (the {a} skeptic line needs someone to grumble).
    const Citizen* who = nullptr;
    if (!m_roster.empty()) {
        who = &m_roster[m_rng.next() % static_cast<std::uint32_t>(m_roster.size())];
    }
    Citizen fallback;
    fallback.firstName = "Somebody";
    fallback.lastName = "Downtown";
    const Citizen& a = who != nullptr ? *who : fallback;
    if (const auto* tpl = pickStory("opening", a)) {
        pushTicker(interpolate(tpl->text, a, nullptr, &dest, street), TickerKind::Opening, dest.c,
                   dest.r);
    }
}

void CitizenSim::reconcileMonthly(const ReconcileInput& in) {
    if (m_script == nullptr || in.homes == nullptr || in.destinations == nullptr) return;
    const std::vector<HomeSite>& homes = *in.homes;
    const std::vector<Destination>& destinations = *in.destinations;
    const auto streetOf = [&](short c, short r) {
        return in.streetName ? in.streetName(c, r) : std::string();
    };
    int emitted = 0;

    // ── Removal: homes that abandoned or were rezoned take their people. ─────
    std::unordered_set<std::uint32_t> homeSet;
    homeSet.reserve(homes.size());
    for (const HomeSite& h : homes) homeSet.insert(packTile(h.c, h.r));

    std::vector<int> remap(m_roster.size(), -1);
    {
        std::vector<Citizen> kept;
        kept.reserve(m_roster.size());
        for (std::size_t i = 0; i < m_roster.size(); ++i) {
            Citizen& cz = m_roster[i];
            if (homeSet.count(packTile(cz.homeC, cz.homeR)) != 0) {
                remap[i] = static_cast<int>(kept.size());
                kept.push_back(std::move(cz));
                continue;
            }
            if (emitted < kMaxStoriesPerMonth && m_rng.chance(0.6f)) {
                if (const auto* tpl = pickStory("departure", cz)) {
                    pushTicker(interpolate(tpl->text, cz, nullptr, nullptr,
                                           streetOf(cz.homeC, cz.homeR)),
                               TickerKind::Departure, cz.homeC, cz.homeR);
                    ++emitted;
                }
            }
        }
        m_roster = std::move(kept);
        for (Citizen& cz : m_roster) {
            cz.spouse = static_cast<signed char>(cz.spouse >= 0 ? remap[static_cast<std::size_t>(cz.spouse)] : -1);
            cz.affair = static_cast<signed char>(cz.affair >= 0 ? remap[static_cast<std::size_t>(cz.affair)] : -1);
        }
    }

    // ── Workplaces: quietly re-home anyone whose job site vanished. ──────────
    std::unordered_set<std::uint32_t> destSet;
    destSet.reserve(destinations.size());
    for (const Destination& d : destinations) destSet.insert(packTile(d.c, d.r));
    for (Citizen& cz : m_roster) {
        if (cz.workC >= 0 && destSet.count(packTile(cz.workC, cz.workR)) != 0) continue;
        if (destinations.empty()) {
            cz.workC = cz.workR = -1;
            continue;
        }
        const Destination& d =
            destinations[m_rng.next() % static_cast<std::uint32_t>(destinations.size())];
        cz.workC = d.c;
        cz.workR = d.r;
    }

    // ── Spawn toward the census-derived target. ──────────────────────────────
    const int target =
        homes.empty() ? 0 : std::clamp(in.population / 140, kMinRoster, kMaxRoster);
    while (static_cast<int>(m_roster.size()) < target &&
           static_cast<int>(m_roster.size()) < kMaxRoster) {
        // Weighted reservoir over home sites (develop = weight).
        const HomeSite* home = nullptr;
        float total = 0.0f;
        for (const HomeSite& h : homes) {
            total += std::max(0.1f, h.develop);
            if (m_rng.uniform(0.0f, total) <= std::max(0.1f, h.develop)) home = &h;
        }
        if (home == nullptr) break;

        Citizen cz;
        cz.seed = odai::procgen::hash2d(static_cast<int>(++m_citizenCounter),
                                        static_cast<int>(m_worldSeed), 0xC171E7u);
        const bool feminine = (cz.seed & 1u) != 0u;
        cz.firstName = m_script->firstName(cz.seed, feminine);
        cz.lastName = m_script->lastName(cz.seed ^ 0x5EEDFACEu);
        cz.homeC = home->c;
        cz.homeR = home->r;
        odai::procgen::Rng traitRng(cz.seed ^ 0x7124175u);
        if (traitRng.chance(0.30f)) cz.traits |= kTraitFit;
        if (traitRng.chance(0.35f)) cz.traits |= kTraitParent;
        if (traitRng.chance(0.25f)) cz.traits |= kTraitNightOwl;
        if (traitRng.chance(0.30f)) cz.traits |= kTraitGossip;
        if (!destinations.empty()) {
            const Destination& d =
                destinations[m_rng.next() % static_cast<std::uint32_t>(destinations.size())];
            cz.workC = d.c;
            cz.workR = d.r;
        }
        // Marriage: pair with an existing single citizen; couples share a name.
        if (m_rng.chance(0.4f)) {
            for (int attempt = 0; attempt < 4 && cz.spouse < 0 && !m_roster.empty(); ++attempt) {
                const std::size_t pick = m_rng.next() % m_roster.size();
                if (m_roster[pick].spouse < 0) {
                    cz.spouse = static_cast<signed char>(pick);
                    m_roster[pick].spouse = static_cast<signed char>(m_roster.size());
                    if (m_rng.chance(0.7f)) cz.lastName = m_roster[pick].lastName;
                }
            }
        }
        m_roster.push_back(std::move(cz));

        Citizen& added = m_roster.back();
        if (emitted < kMaxStoriesPerMonth && m_rng.chance(0.35f * std::min(m_storyBoost, 3.0f))) {
            if (const auto* tpl = pickStory("arrival", added)) {
                pushTicker(interpolate(tpl->text, added, nullptr, nullptr,
                                       streetOf(added.homeC, added.homeR)),
                           TickerKind::Arrival, added.homeC, added.homeR);
                ++emitted;
            }
        }
    }

    // ── Monthly story rolls. ─────────────────────────────────────────────────
    for (std::size_t i = 0; i < m_roster.size() && emitted < kMaxStoriesPerMonth; ++i) {
        Citizen& cz = m_roster[i];
        const Destination* place =
            destinations.empty()
                ? nullptr
                : &destinations[m_rng.next() % static_cast<std::uint32_t>(destinations.size())];

        // Tabloid drama for the gossip-adjacent married set (and established
        // affairs). A drama roll with no affair yet establishes one.
        const bool dramaEligible = (cz.affair >= 0) ||
                                   ((cz.traits & kTraitGossip) != 0 && cz.spouse >= 0);
        if (dramaEligible && m_rng.chance(0.012f * m_storyBoost)) {
            if (cz.affair < 0 && m_roster.size() > 2) {
                for (int attempt = 0; attempt < 4; ++attempt) {
                    const std::size_t pick = m_rng.next() % m_roster.size();
                    if (pick != i && static_cast<int>(pick) != cz.spouse) {
                        cz.affair = static_cast<signed char>(pick);
                        break;
                    }
                }
            }
            if (const auto* tpl = pickStory("drama", cz)) {
                const Citizen* partner =
                    cz.affair >= 0 ? &m_roster[static_cast<std::size_t>(cz.affair)] : nullptr;
                pushTicker(interpolate(tpl->text, cz, partner, place,
                                       streetOf(cz.homeC, cz.homeR)),
                           TickerKind::Drama, cz.homeC, cz.homeR);
                ++emitted;
                continue;
            }
        }

        if (m_rng.chance(0.04f * m_storyBoost)) {
            if (const auto* tpl = pickStory("life", cz)) {
                pushTicker(interpolate(tpl->text, cz, nullptr, place,
                                       streetOf(cz.homeC, cz.homeR)),
                           TickerKind::Life, place != nullptr ? place->c : cz.homeC,
                           place != nullptr ? place->r : cz.homeR);
                ++emitted;
            }
        }
    }
}

namespace {

// Schedule phases, in day-hours. Weekday rushes bracket the working day;
// lunch runs leave from the office; night belongs to the owls.
constexpr float kMorningRushStart = 7.0f, kMorningRushEnd = 9.5f;
constexpr float kLunchStart = 11.5f, kLunchEnd = 13.5f;
constexpr float kEveningRushStart = 17.0f, kEveningRushEnd = 19.5f;
constexpr float kNightStart = 21.0f, kNightEnd = 2.0f;  // wraps midnight
constexpr float kSoccerStart = 8.0f, kSoccerEnd = 11.0f;

bool isNightHour(float hour) { return hour >= kNightStart || hour < kNightEnd; }

}  // namespace

bool CitizenSim::rollTrip(const std::vector<Destination>& destinations, const TripContext& ctx,
                          Trip& out) {
    if (m_roster.empty() || m_script == nullptr) return false;
    const bool weekend = ctx.weekday >= 5;

    // Reservoir-pick among destinations of one category.
    const auto pickByCategory = [&](const char* category) -> const Destination* {
        const Destination* dest = nullptr;
        int seen = 0;
        for (const Destination& d : destinations) {
            if (d.category != category) continue;
            ++seen;
            if (static_cast<int>(m_rng.next() % static_cast<std::uint32_t>(seen)) == 0) dest = &d;
        }
        return dest;
    };

    // Weekday rush hours: move the workforce as a wave. Scan from a random
    // start for a citizen in the right state so the wave drains as it runs.
    if (!weekend && ctx.hour >= kMorningRushStart && ctx.hour < kMorningRushEnd) {
        const std::size_t start = m_rng.next() % m_roster.size();
        for (std::size_t k = 0; k < m_roster.size(); ++k) {
            Citizen& cz = m_roster[(start + k) % m_roster.size()];
            if (cz.atWork || cz.workC < 0 || cz.homeC < 0) continue;
            cz.atWork = true;
            out.fromC = cz.homeC;
            out.fromR = cz.homeR;
            out.toC = cz.workC;
            out.toR = cz.workR;
            return true;
        }
        // Everyone's in already — fall through to an errand.
    }
    if (!weekend && ctx.hour >= kEveningRushStart && ctx.hour < kEveningRushEnd) {
        const std::size_t start = m_rng.next() % m_roster.size();
        for (std::size_t k = 0; k < m_roster.size(); ++k) {
            Citizen& cz = m_roster[(start + k) % m_roster.size()];
            if (!cz.atWork || cz.homeC < 0) continue;
            cz.atWork = false;
            out.fromC = cz.workC;
            out.fromR = cz.workR;
            out.toC = cz.homeC;
            out.toR = cz.homeR;
            return true;
        }
    }

    const Citizen* picked = nullptr;
    const std::size_t start = m_rng.next() % m_roster.size();

    // Saturday morning: soccer practice — parents haul the team to the park.
    if (ctx.weekday == 5 && ctx.hour >= kSoccerStart && ctx.hour < kSoccerEnd &&
        m_rng.chance(0.7f)) {
        for (std::size_t k = 0; k < m_roster.size(); ++k) {
            const Citizen& cz = m_roster[(start + k) % m_roster.size()];
            if ((cz.traits & kTraitParent) == 0 || cz.homeC < 0) continue;
            if (const Destination* park = pickByCategory("park")) {
                out.fromC = cz.homeC;
                out.fromR = cz.homeR;
                out.toC = park->c;
                out.toR = park->r;
                return true;
            }
            break;  // no park in town; regular errand instead
        }
    }

    // Night: only the owls are out (bars, arcades, late diners).
    if (isNightHour(ctx.hour)) {
        for (std::size_t k = 0; k < m_roster.size(); ++k) {
            const Citizen& cz = m_roster[(start + k) % m_roster.size()];
            if ((cz.traits & kTraitNightOwl) == 0 || cz.homeC < 0) continue;
            picked = &cz;
            break;
        }
        if (picked == nullptr) return false;  // a sleepy town after dark
    } else {
        const Citizen& cz = m_roster[start];
        if (cz.homeC < 0) return false;
        picked = &cz;
    }
    const Citizen& cz = *picked;

    // Lunch runs leave from the office; everything else leaves from home.
    const bool lunchRun = !weekend && cz.atWork && cz.workC >= 0 && ctx.hour >= kLunchStart &&
                          ctx.hour < kLunchEnd;
    out.fromC = lunchRun ? cz.workC : cz.homeC;
    out.fromR = lunchRun ? cz.workR : cz.homeR;

    // Draw a need from the Lua-weighted table restricted to this citizen's
    // traits and resolve it to a named place. Night trims the table to the
    // owl categories; lunch trims it to food.
    const auto allowed = [&](const std::string& category) {
        if (lunchRun) {
            return category == "cafe" || category == "diner" || category == "grocery";
        }
        if (isNightHour(ctx.hour)) {
            return category == "bar" || category == "arcade" || category == "diner" ||
                   category == "cinema";
        }
        return true;
    };
    float totalWeight = 0.0f;
    for (const odai::citybuilder::NeedRule& need : m_script->needs()) {
        if (need.trait != "any" && !hasTag(cz, need.trait)) continue;
        if (!allowed(need.category)) continue;
        totalWeight += std::max(0.01f, need.weight);
    }
    if (totalWeight <= 0.0f) return false;
    float roll = m_rng.uniform(0.0f, totalWeight);
    const std::string* category = nullptr;
    for (const odai::citybuilder::NeedRule& need : m_script->needs()) {
        if (need.trait != "any" && !hasTag(cz, need.trait)) continue;
        if (!allowed(need.category)) continue;
        roll -= std::max(0.01f, need.weight);
        if (roll <= 0.0f) {
            category = &need.category;
            break;
        }
    }
    if (category == nullptr) return false;

    const Destination* dest = pickByCategory(category->c_str());
    if (dest == nullptr) return false;
    out.toC = dest->c;
    out.toR = dest->r;
    return true;
}

}  // namespace odai::games::citybuilder
