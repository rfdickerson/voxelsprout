#include "bethesda/tes3_progression.h"
#include "bethesda/tes3_runtime.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace odai::bethesda {
namespace {
const Tes3SubrecordData* subrecord(const Tes3NamedRecord* record, std::string_view type) {
    if (!record) return nullptr;
    for (const auto& sub : record->subrecords) if (sub.type == type) return &sub;
    return nullptr;
}
template<class T> T read(const Tes3SubrecordData& sub, std::size_t offset) {
    T result{};
    std::memcpy(&result, sub.data.data() + offset, sizeof(T));
    return result;
}
double baseStat(const Tes3DialoguePlayerState& player, std::string_view name) {
    const auto it = player.numericFilters.find(std::string(name));
    return it == player.numericFilters.end() ? 0.0 : it->second;
}
std::optional<Tes3ProgressionClass> playerClass(const Tes3Runtime& runtime) {
    if (runtime.playerState().progression.customClass)
        return validTes3ProgressionClass(*runtime.playerState().progression.customClass)
            ? runtime.playerState().progression.customClass : std::nullopt;
    return runtime.content() ? tes3ClassDefinition(*runtime.content(), runtime.playerState().actorClass)
                             : std::nullopt;
}
bool contains(const std::array<int, 5>& skills, int skill) {
    return std::find(skills.begin(), skills.end(), skill) != skills.end();
}
}

int tes3ArmorHitSlot(unsigned roll) {
    if (roll >= 100) return -1;
    if (roll < 30) return 1; // cuirass
    if (roll < 40) return 0; // helmet
    if (roll < 50) return 4; // greaves
    if (roll < 60) return 5; // boots
    if (roll < 70) return 2; // left pauldron
    if (roll < 80) return 3; // right pauldron
    if (roll < 85) return 6; // left gauntlet/bracer
    if (roll < 90) return 7; // right gauntlet/bracer
    return 8; // shield; an empty slot remains an unarmored hit
}
std::optional<Tes3ArmorDefinition> tes3ArmorDefinition(const Tes3ContentStore& content, std::string_view id) {
    const auto* data = subrecord(content.findRecord("ARMO", id), "AODT");
    if (!data || data->data.size() != 24) return std::nullopt;
    const int type = read<std::int32_t>(*data, 0);
    const double weight = read<float>(*data, 4);
    if (type < 0 || type > 10 || !std::isfinite(weight) || weight < 0) return std::nullopt;
    constexpr std::array<std::string_view, 11> settings{
        "iHelmWeight", "iCuirassWeight", "iPauldronWeight", "iPauldronWeight", "iGreavesWeight",
        "iBootsWeight", "iGauntletWeight", "iGauntletWeight", "iShieldWeight", "iGauntletWeight", "iGauntletWeight"};
    constexpr std::array<double, 11> fallback{5,30,10,10,15,20,5,5,15,5,5};
    const double reference = std::floor(tes3NumericSetting(content, settings[type], fallback[type]));
    const double light = tes3NumericSetting(content, "fLightMaxMod", .6);
    const double medium = tes3NumericSetting(content, "fMedMaxMod", .9);
    if (!std::isfinite(reference) || !std::isfinite(light) || !std::isfinite(medium) ||
        reference < 0 || light < 0 || medium < light) return std::nullopt;
    const int skill = weight <= reference * light + .0005 ? 21 : weight <= reference * medium + .0005 ? 2 : 3;
    return Tes3ArmorDefinition{type == 9 ? 6 : type == 10 ? 7 : type, skill};
}
std::optional<int> tes3WeaponSkill(const Tes3ContentStore& content, std::string_view id) {
    const auto* data = subrecord(content.findRecord("WEAP", id), "WPDT");
    if (!data || data->data.size() != 32) return std::nullopt;
    const int type = read<std::int16_t>(*data, 8);
    constexpr std::array<int, 14> skills{22,5,5,4,4,4,7,6,6,23,23,23,-1,-1};
    if (type < 0 || type >= int(skills.size()) || skills[type] < 0) return std::nullopt;
    return skills[type];
}
bool tes3AutoCalculateNpc(const Tes3ContentStore& content, Tes3ActorDefinition& actor) {
    if (!actor.autoCalculate || actor.creature) return false;
    const auto c = tes3ClassDefinition(content, actor.actorClass);
    const auto* race = subrecord(content.findRecord("RACE", actor.race), "RADT");
    if (!c || !race || race->data.size() != 140 || actor.gender < 0 || actor.gender > 1) return false;
    auto next = actor;
    const auto rounded = [](double value) {
        const double integral = std::floor(value), fraction = value - integral;
        if (fraction < .5) return integral;
        if (fraction > .5) return integral + 1;
        return std::fmod(integral, 2) == 0 ? integral : integral + 1;
    };
    std::array<float, 8> growth{};
    for (int skill = 0; skill < 27; ++skill) {
        const auto s = tes3SkillDefinition(content, skill);
        if (!s) return false;
        const bool major = contains(c->major, skill), minor = contains(c->minor, skill);
        const bool specialized = s->specialization == c->specialization;
        growth[s->attribute] += major ? 1.f : minor ? .5f : .2f;
        int raceBonus = 0;
        for (int i = 0; i < 7; ++i) if (read<std::int32_t>(*race, i * 8) == skill) {
            raceBonus = read<std::int32_t>(*race, i * 8 + 4); break;
        }
        const float value = 5 + (major ? 25 : minor ? 10 : 0) + raceBonus + (specialized ? 5 : 0) +
            (actor.level - 1.f) * ((major || minor ? 1.f : .1f) + (specialized ? .5f : 0.f));
        next.skills[std::string(tes3SkillNames[skill])] = float(std::min(100.0, rounded(value)));
    }
    for (int attribute = 0; attribute < 8; ++attribute) {
        float value = read<std::int32_t>(*race, 56 + attribute * 8 + actor.gender * 4);
        if (std::find(c->attributes.begin(), c->attributes.end(), attribute) != c->attributes.end()) value += 10;
        value = float(std::min(100.0, rounded(value + (actor.level - 1.f) * growth[attribute])));
        next.attributes[std::string(tes3AttributeNames[attribute])] = float(value);
    }
    const int healthMultiplier = 3 + (c->specialization == 0 ? 2 : c->specialization == 2 ? 1 : 0) +
        (std::find(c->attributes.begin(), c->attributes.end(), 5) != c->attributes.end() ? 1 : 0);
    next.health = float(std::floor((next.attributes["strength"] + next.attributes["endurance"]) * .5) +
        healthMultiplier * (actor.level - 1.0));
    const double magicka = next.attributes["intelligence"] * tes3NumericSetting(content, "fNPCbaseMagickaMult", 1);
    if (!std::isfinite(magicka) || magicka < 0 || magicka > std::numeric_limits<float>::max()) return false;
    next.magicka = float(magicka);
    next.fatigue = next.attributes["strength"] + next.attributes["willpower"] + next.attributes["agility"] + next.attributes["endurance"];
    actor = std::move(next);
    return true;
}
bool validTes3ProgressionClass(const Tes3ProgressionClass& c) {
    if (c.specialization < 0 || c.specialization > 2 || c.attributes[0] == c.attributes[1]) return false;
    for (int a : c.attributes) if (a < 0 || a >= 8) return false;
    std::set<int> skills;
    for (int s : c.major) if (s < 0 || s >= 27 || !skills.insert(s).second) return false;
    for (int s : c.minor) if (s < 0 || s >= 27 || !skills.insert(s).second) return false;
    return true;
}
std::optional<Tes3ProgressionClass> tes3ClassDefinition(const Tes3ContentStore& content, std::string_view id) {
    const auto* sub = subrecord(content.findRecord("CLAS", id), "CLDT");
    if (!sub || sub->data.size() != 60) return std::nullopt;
    Tes3ProgressionClass c;
    for (int i = 0; i < 2; ++i) c.attributes[i] = read<std::int32_t>(*sub, i * 4);
    c.specialization = read<std::int32_t>(*sub, 8);
    for (int i = 0; i < 5; ++i) {
        c.minor[i] = read<std::int32_t>(*sub, 12 + i * 8);
        c.major[i] = read<std::int32_t>(*sub, 16 + i * 8);
    }
    return validTes3ProgressionClass(c) ? std::optional(c) : std::nullopt;
}
std::optional<Tes3SkillDefinition> tes3SkillDefinition(const Tes3ContentStore& content, int skill) {
    if (skill < 0 || skill >= 27) return std::nullopt;
    const auto* sub = subrecord(content.findRecord("SKIL", std::to_string(skill)), "SKDT");
    if (!sub || sub->data.size() != 24) return std::nullopt;
    Tes3SkillDefinition s;
    s.attribute = read<std::int32_t>(*sub, 0);
    s.specialization = read<std::int32_t>(*sub, 4);
    if (s.attribute < 0 || s.attribute >= 8 || s.specialization < 0 || s.specialization > 2)
        return std::nullopt;
    for (int i = 0; i < 4; ++i) {
        s.useValues[i] = read<float>(*sub, 8 + i * 4);
        if (!std::isfinite(s.useValues[i]) || s.useValues[i] < 0) return std::nullopt;
    }
    return s;
}
double tes3NumericSetting(const Tes3ContentStore& content, std::string_view id, double fallback) {
    const auto* record = content.findRecord("GMST", id);
    if (!record) return fallback;
    const auto* f = subrecord(record, "FLTV");
    if (f && f->data.size() == 4) return read<float>(*f, 0);
    const auto* i = subrecord(record, "INTV");
    if (i && i->data.size() == 4) return read<std::int32_t>(*i, 0);
    return std::numeric_limits<double>::quiet_NaN();
}
void Tes3Runtime::synchronizePlayerDialogue() {
    if (m_dialogue.active) m_dialogue.player = m_playerState;
}

bool Tes3Runtime::initializePlayerProgression(std::string& error) {
    const auto c = playerClass(*this);
    const auto* race = m_content ? subrecord(m_content->findRecord("RACE", m_playerState.race), "RADT") : nullptr;
    if (!c || !race || race->data.size() != 140 || m_playerState.gender < 0 || m_playerState.gender > 1) {
        error = "Character progression requires a valid race, gender, and class"; return false;
    }
    auto next = m_playerState;
    next.progression = {};
    next.progression.customClass = m_playerState.progression.customClass;
    for (int a = 0; a < 8; ++a) {
        const int value = read<std::int32_t>(*race, 56 + a * 8 + next.gender * 4);
        if (value < 0 || value > 100) { error = "Invalid race attribute"; return false; }
        next.numericFilters[std::string(tes3AttributeNames[a])] = value;
        next.numericFilters.erase("damage:" + std::string(tes3AttributeNames[a]));
    }
    for (int a : c->attributes) next.numericFilters[std::string(tes3AttributeNames[a])] += 10;
    for (int s = 0; s < 27; ++s) {
        const auto def = tes3SkillDefinition(*m_content, s);
        if (!def) { error = "Missing or invalid TES3 skill " + std::to_string(s); return false; }
        double value = 5 + (contains(c->major, s) ? 25 : contains(c->minor, s) ? 10 : 0);
        if (def->specialization == c->specialization) value += 5;
        next.numericFilters[std::string(tes3SkillNames[s])] = value;
        next.numericFilters.erase("damage:" + std::string(tes3SkillNames[s]));
    }
    for (int i = 0; i < 7; ++i) {
        const int skill = read<std::int32_t>(*race, i * 8);
        const int bonus = read<std::int32_t>(*race, i * 8 + 4);
        if (skill == -1) continue;
        if (skill < 0 || skill >= 27 || bonus < 0 || bonus > 100) {
            error = "Invalid race skill bonus"; return false;
        }
        next.numericFilters[std::string(tes3SkillNames[skill])] += bonus;
    }
    // TES3 ability Fortify Attribute is part of the birthsign creation baseline
    // (not an equipment/spell modifier), including The Lady's Endurance.
    const auto* sign = m_content->findRecord("BSGN", next.birthsign);
    if (!sign) { error = "Missing birthsign"; return false; }
    double magickaMultiplier = tes3NumericSetting(*m_content, "fPCbaseMagickaMult", 1);
    const auto applyAbilities = [&](const Tes3NamedRecord* record) {
        if (!record) return;
        for (const auto& sub : record->subrecords) if (sub.type == "NPCS") {
            const auto end = std::find(sub.data.begin(), sub.data.end(), std::uint8_t{0});
            const std::string id(sub.data.begin(), end);
            const auto* spell = m_content->findSpell(id);
            if (!spell) continue;
            next.inventory[spell->record] = 1;
            if (spell->type != 1) continue;
            for (const auto& effect : spell->effects) {
                if (effect.effectId == 79 && effect.attribute >= 0 && effect.attribute < 8)
                    next.numericFilters[std::string(tes3AttributeNames[effect.attribute])] += effect.magnitudeMin;
                if (effect.effectId == 84) magickaMultiplier += effect.magnitudeMin * .1;
            }
        }
    };
    applyAbilities(m_content->findRecord("RACE", next.race));
    applyAbilities(sign);
    for (auto name : tes3SkillNames) next.numericFilters[std::string(name)] = std::min(100.0, baseStat(next, name));
    next.numericFilters["level"] = 1;
    next.numericFilters["maxhealth"] = std::floor((baseStat(next, "strength") + baseStat(next, "endurance")) * .5);
    next.numericFilters["health"] = next.numericFilters["maxhealth"];
    next.numericFilters["magicka_multiplier"] = magickaMultiplier;
    next.numericFilters["maxmagicka"] = baseStat(next, "intelligence") * magickaMultiplier;
    next.numericFilters["magicka"] = next.numericFilters["maxmagicka"];
    next.numericFilters["maxfatigue"] = baseStat(next, "strength") + baseStat(next, "willpower") +
        baseStat(next, "agility") + baseStat(next, "endurance");
    next.numericFilters["fatigue"] = next.numericFilters["maxfatigue"];
    if (!std::isfinite(magickaMultiplier) || magickaMultiplier < 0) {
        error = "Invalid TES3 starting magicka multiplier"; return false;
    }
    m_playerState = std::move(next);
    synchronizePlayerDialogue(); error.clear(); return true;
}

double Tes3Runtime::playerSkillRequirement(int skill) const {
    const auto c = playerClass(*this);
    const auto s = m_content ? tes3SkillDefinition(*m_content, skill) : std::nullopt;
    if (!c || !s) return 0;
    double factor = tes3NumericSetting(*m_content, contains(c->major, skill) ? "fMajorSkillBonus" :
        contains(c->minor, skill) ? "fMinorSkillBonus" : "fMiscSkillBonus",
        contains(c->major, skill) ? .75 : contains(c->minor, skill) ? 1 : 1.25);
    if (s->specialization == c->specialization)
        factor *= tes3NumericSetting(*m_content, "fSpecialSkillBonus", .8);
    const double requirement = (baseStat(m_playerState, tes3SkillNames[skill]) + 1) * factor;
    return std::isfinite(requirement) && requirement > 0 ? requirement : 0;
}
int Tes3Runtime::levelThreshold() const {
    const double value = m_content ? tes3NumericSetting(*m_content, "iLevelUpTotal", 10) : 10;
    return std::isfinite(value) && value >= 1 && value <= 1000000 && std::floor(value) == value
        ? static_cast<int>(value) : 0;
}
bool Tes3Runtime::playerLevelReady() const {
    return levelThreshold() > 0 && m_playerState.progression.levelProgress >= levelThreshold();
}
bool Tes3Runtime::advancePlayerSkill(int skill, Tes3SkillAdvanceSource source, std::string& error) {
    const auto c = playerClass(*this);
    const auto s = m_content ? tes3SkillDefinition(*m_content, skill) : std::nullopt;
    if (!c || !s || m_playerState.progression.selectionOpen || m_pendingResult) {
        error = "Skill advancement requires valid class/skill data and no open level selection"; return false;
    }
    const double base = baseStat(m_playerState, tes3SkillNames[skill]);
    const bool decrease = source == Tes3SkillAdvanceSource::Jail && skill != 18 && skill != 19;
    if (!std::isfinite(base) || base < 0 || (decrease ? base <= 0 : base >= 100)) {
        error = "Skill cannot advance beyond its base bounds"; return false;
    }
    int levelGain = 0, attributeGain = 0;
    if (!decrease) {
        const bool major = contains(c->major, skill), minor = contains(c->minor, skill);
        const double lg = tes3NumericSetting(*m_content, major ? "iLevelUpMajorMult" : "iLevelUpMinorMult", 1);
        const double ag = tes3NumericSetting(*m_content, major ? "iLevelUpMajorMultAttribute" :
            minor ? "iLevelUpMinorMultAttribute" : "iLevelupMiscMultAttriubte", 1);
        if (!std::isfinite(lg) || !std::isfinite(ag) || lg < 0 || ag < 0 || lg > 1000000 || ag > 1000000 ||
            lg != std::floor(lg) || ag != std::floor(ag) || levelThreshold() == 0) {
            error = "Invalid TES3 advancement settings"; return false;
        }
        levelGain = major || minor ? static_cast<int>(lg) : 0;
        attributeGain = s->attribute == 7 ? 0 : static_cast<int>(ag);
    }
    auto& state = m_playerState.progression;
    if (state.levelProgress > std::numeric_limits<int>::max() - levelGain ||
        state.attributeIncreases[s->attribute] > std::numeric_limits<int>::max() - attributeGain) {
        error = "Skill advancement counters overflow"; return false;
    }
    const bool wasReady = playerLevelReady();
    m_playerState.numericFilters[std::string(tes3SkillNames[skill])] = decrease ? std::max(0.0, base - 1) : std::min(100.0, base + 1);
    state.levelProgress += levelGain;
    state.attributeIncreases[s->attribute] += attributeGain;
    if (source == Tes3SkillAdvanceSource::Usage) state.skillProgress[skill] = 0;
    if (!wasReady && playerLevelReady()) state.readyNotification = true;
    synchronizePlayerDialogue(); error.clear(); return true;
}
bool Tes3Runtime::usePlayerSkill(int skill, int useType, double scale, std::string& error) {
    const auto s = m_content ? tes3SkillDefinition(*m_content, skill) : std::nullopt;
    const double requirement = playerSkillRequirement(skill);
    if (!s || requirement <= 0 || useType < 0 || useType >= 4 || !std::isfinite(scale) || scale <= 0 ||
        m_playerState.progression.selectionOpen || baseStat(m_playerState, tes3SkillNames[skill]) >= 100) {
        error = "Skill use requires valid data, scale, and an uncapped skill"; return false;
    }
    const double prior = m_playerState.progression.skillProgress[skill];
    if (!std::isfinite(prior) || prior < 0 || prior >= 1) { error = "Invalid existing skill progress"; return false; }
    const double progress = prior + s->useValues[useType] * scale / requirement;
    if (!std::isfinite(progress)) { error = "Skill use progress overflow"; return false; }
    if (progress >= 1) return advancePlayerSkill(skill, Tes3SkillAdvanceSource::Usage, error);
    m_playerState.progression.skillProgress[skill] = progress;
    synchronizePlayerDialogue(); error.clear(); return true;
}
int Tes3Runtime::playerAttributeGain(int attribute) const {
    if (!m_content || attribute < 0 || attribute >= 8) return 0;
    const double base = baseStat(m_playerState, tes3AttributeNames[attribute]);
    if (!std::isfinite(base) || base < 0 || base >= 100) return 0;
    const int count = attribute == 7 ? 0 : std::clamp(m_playerState.progression.attributeIncreases[attribute], 0, 10);
    const double gain = count == 0 ? 1 : tes3NumericSetting(*m_content,
        "iLevelUp" + std::string(count < 10 ? "0" : "") + std::to_string(count) + "Mult",
        count <= 4 ? 2 : count <= 7 ? 3 : count <= 9 ? 4 : 5);
    return std::isfinite(gain) && gain >= 1 && gain <= 100 && gain == std::floor(gain)
        ? static_cast<int>(std::min(gain, 100 - base)) : 0;
}
int Tes3Runtime::levelChoiceCount() const {
    int count = 0;
    for (auto a : tes3AttributeNames) if (baseStat(m_playerState, a) < 100) ++count;
    return std::min(3, count);
}
bool Tes3Runtime::openPlayerLevelSelection() {
    if (!playerLevelReady() || m_dialogue.active || m_pendingResult) return false;
    m_playerState.progression.selectionOpen = true;
    return true;
}
bool Tes3Runtime::confirmPlayerLevel(const std::vector<int>& attributes, std::string& error) {
    if (!m_playerState.progression.selectionOpen || !playerLevelReady() ||
        static_cast<int>(attributes.size()) != levelChoiceCount()) {
        error = "Level confirmation requires an eligible rest and all attribute choices"; return false;
    }
    for (auto name : tes3AttributeNames) {
        const double value = baseStat(m_playerState, name);
        if (!std::isfinite(value) || value < 0 || value > 1000000) {
            error = "Invalid base level attribute"; return false;
        }
    }
    auto next = m_playerState;
    std::set<int> selected;
    for (int a : attributes) {
        const int gain = playerAttributeGain(a);
        if (gain <= 0 || !selected.insert(a).second) { error = "Invalid or duplicate level attribute"; return false; }
        next.numericFilters[std::string(tes3AttributeNames[a])] += gain;
    }
    const double healthFactor = tes3NumericSetting(*m_content, "fLevelUpHealthEndMult", .1);
    const double level = baseStat(next, "level"), maximum = baseStat(next, "maxhealth");
    const double healthGain = baseStat(next, "endurance") * healthFactor;
    if (!std::isfinite(healthFactor) || healthFactor < 0 || !std::isfinite(healthGain) ||
        !std::isfinite(level) || level < 1 || level >= std::numeric_limits<int>::max() ||
        !std::isfinite(maximum + healthGain) || maximum <= 0) {
        error = "Invalid player level or health growth"; return false;
    }
    next.numericFilters["maxhealth"] = maximum + healthGain;
    next.numericFilters["health"] = std::min(maximum + healthGain, std::max(1.0, baseStat(next, "health") + healthGain));
    next.numericFilters["level"] = level + 1;
    const double magickaMultiplier = next.numericFilters.contains("magicka_multiplier") ? baseStat(next, "magicka_multiplier") : 1;
    const double magickaRatio = baseStat(next, "maxmagicka") > 0 ? baseStat(next, "magicka") / baseStat(next, "maxmagicka") : 0;
    const double fatigueRatio = baseStat(next, "maxfatigue") > 0 ? baseStat(next, "fatigue") / baseStat(next, "maxfatigue") : 0;
    next.numericFilters["maxmagicka"] = baseStat(next, "intelligence") * magickaMultiplier;
    next.numericFilters["magicka"] = next.numericFilters["maxmagicka"] * magickaRatio;
    next.numericFilters["maxfatigue"] = baseStat(next, "strength") + baseStat(next, "willpower") +
        baseStat(next, "agility") + baseStat(next, "endurance");
    next.numericFilters["fatigue"] = next.numericFilters["maxfatigue"] * fatigueRatio;
    for (std::string_view name : {"health", "maxhealth", "magicka", "maxmagicka", "fatigue", "maxfatigue"}) {
        const double value = baseStat(next, name);
        if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<float>::max()) {
            error = "Invalid derived level-up stat"; return false;
        }
    }
    next.progression.levelProgress -= levelThreshold();
    next.progression.attributeIncreases.fill(0);
    next.progression.selectionOpen = false;
    next.progression.readyNotification = false;
    m_playerState = std::move(next);
    synchronizePlayerDialogue(); error.clear(); return true;
}
} // namespace odai::bethesda
