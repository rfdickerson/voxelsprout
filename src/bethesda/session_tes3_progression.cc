#include "bethesda/bethesda_session.h"
#include "core/hash.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace odai::bethesda {
namespace {
const Tes3SubrecordData* findSub(const Tes3NamedRecord* record, std::string_view type) {
    if (record) for (const auto& sub : record->subrecords) if (sub.type == type) return &sub;
    return nullptr;
}
template<class T> T readSub(const Tes3SubrecordData& sub, std::size_t offset = 0) {
    T value{}; std::memcpy(&value, sub.data.data() + offset, sizeof(T)); return value;
}
int itemMaximumCondition(const Tes3ContentStore& content, const RecordKey& key) {
    const auto* record = content.findRecord(key.recordType, key.textId);
    const auto* data = findSub(record, key.recordType == "WEAP" ? "WPDT" : key.recordType == "ARMO" ? "AODT" : "RIDT");
    if (!data) return 0;
    if (key.recordType == "WEAP" && data->data.size() == 32) return readSub<std::uint16_t>(*data, 10);
    if (key.recordType == "ARMO" && data->data.size() == 24) return std::max(0, readSub<std::int32_t>(*data, 12));
    if (key.recordType == "REPA" && data->data.size() == 16) {
        const float quality = readSub<float>(*data, 12);
        if (std::isfinite(quality) && quality >= 0) return std::max(0, readSub<std::int32_t>(*data, 8));
    }
    return 0;
}
bool inReach(const RuntimeObject& player, const RuntimeObject& target, double distance) {
    if (player.currentSpace != target.currentSpace || !target.enabled) return false;
    const auto& a = player.transform.position; const auto& b = target.transform.position;
    return std::hypot(std::hypot(a[0] - b[0], a[1] - b[1]), a[2] - b[2]) <= distance;
}
}

double BethesdaSession::tes3MagicEffectMagnitude(ObjectId actor, int id, int skill, int attribute) const {
    double total = 0;
    std::set<RecordKey> activePassives;
    if (const auto it = m_tes3.activeSpells().find(actor); it != m_tes3.activeSpells().end()) {
        for (const auto& spell : it->second) {
            const auto* definition = m_tes3.content() ? m_tes3.content()->findSpell(spell.spell.textId) : nullptr;
            for (const auto& effect : spell.effects) {
                if (effect.expiresTick <= m_clock.tick()) continue;
                if (definition && definition->type >= 1 && definition->type <= 4) activePassives.insert(spell.spell);
                if (effect.effectId == id && (skill < 0 || effect.skill == skill) &&
                    (attribute < 0 || effect.attribute == attribute)) total += effect.magnitude;
            }
        }
    }
    const auto* object = m_world.find(actor);
    if (!object || !m_tes3.content()) return total;
    for (const auto& entry : object->inventory) {
        if (entry.count <= 0 || entry.item.recordType != "SPEL" || activePassives.contains(entry.item)) continue;
        const auto* spell = m_tes3.content()->findSpell(entry.item.textId);
        if (!spell || spell->type < 1 || spell->type > 4) continue;
        // Creation ability Fortify Attribute was incorporated into the player's
        // baseline. Abilities added later are represented by active effects.
        if (actor == m_playerObject && spell->type == 1 && id == 79 && attribute >= 0) continue;
        for (const auto& effect : spell->effects)
            if (effect.effectId == id && (skill < 0 || effect.skill == skill) &&
                (attribute < 0 || effect.attribute == attribute))
                total += std::min(effect.magnitudeMin, effect.magnitudeMax);
    }
    return total;
}
double BethesdaSession::modifiedTes3Stat(ObjectId actor, std::string_view name) const {
    double value = 0, damage = 0;
    if (actor == m_playerObject) {
        const auto& stats = m_tes3.playerState().numericFilters;
        if (const auto it = stats.find(std::string(name)); it != stats.end()) value = it->second;
        if (const auto it = stats.find("damage:" + std::string(name)); it != stats.end()) damage = it->second;
    } else if (const auto* object = m_world.find(actor); object && m_tes3.content()) {
        if (const auto* npc = m_tes3.content()->findActor(object->base.recordType, object->base.textId)) {
            if (const auto it = npc->skills.find(std::string(name)); it != npc->skills.end()) value = it->second;
            if (const auto it = npc->attributes.find(std::string(name)); it != npc->attributes.end()) value = it->second;
        }
        if (const auto it = m_tes3.referenceOverrides().find(actor); it != m_tes3.referenceOverrides().end()) {
            if (const auto stat = it->second.locals.find("stat:" + std::string(name)); stat != it->second.locals.end()) value = stat->second.number;
            if (const auto stat = it->second.locals.find("damage:" + std::string(name)); stat != it->second.locals.end()) damage = stat->second.number;
        }
    }
    for (int attribute = 0; attribute < 8; ++attribute) if (tes3AttributeNames[attribute] == name)
        value += tes3MagicEffectMagnitude(actor, 79, -1, attribute) - tes3MagicEffectMagnitude(actor, 17, -1, attribute);
    for (int skill = 0; skill < 27; ++skill) if (tes3SkillNames[skill] == name)
        value += tes3MagicEffectMagnitude(actor, 83, skill) - tes3MagicEffectMagnitude(actor, 21, skill);
    return std::max(0.0, value - damage);
}
void BethesdaSession::synchronizeTes3PlayerValues() {
    auto* player = m_world.find(m_playerObject);
    if (!player || !player->actorValues) return;
    auto& v = *player->actorValues;
    const auto& stats = m_tes3.playerState().numericFilters;
    const auto copy = [&](std::string_view key, float& target) {
        const auto it = stats.find(std::string(key));
        if (it != stats.end() && std::isfinite(it->second)) target = static_cast<float>(it->second);
    };
    copy("health", v.health); copy("maxhealth", v.maxHealth);
    copy("magicka", v.magicka); copy("maxmagicka", v.maxMagicka);
    copy("fatigue", v.stamina); copy("maxfatigue", v.maxStamina);
    m_tes3.synchronizePlayerDialogue();
}
bool BethesdaSession::initializeTes3PlayerProgression(std::string& error) {
    if (!m_tes3.initializePlayerProgression(error)) return false;
    if (auto* player = m_world.find(m_playerObject)) {
        for (const auto& [spell, count] : m_tes3.playerState().inventory) if (spell.recordType == "SPEL" && count > 0 &&
            std::none_of(player->inventory.begin(), player->inventory.end(), [&](const auto& entry) { return entry.item == spell; }))
            player->inventory.push_back({spell, 1, false});
    }
    synchronizeTes3PlayerValues(); return true;
}
bool BethesdaSession::readTes3SkillBook(const RecordKey& book, std::string& error) {
    const auto* player = m_world.find(m_playerObject);
    const auto* record = m_tes3.content() ? m_tes3.content()->findRecord("BOOK", book.textId) : nullptr;
    if (book.recordType != "BOOK" || !record || !player || !player->actorValues || player->actorValues->dead ||
        std::none_of(player->inventory.begin(), player->inventory.end(), [&](const auto& e) { return e.item == book && e.count > 0; })) {
        error = "Reading requires a living player and an owned TES3 book"; return false;
    }
    const auto* data = findSub(record, "BKDT");
    if (!data || data->data.size() != 20) { error = "Invalid TES3 book data"; return false; }
    const auto id = normalizeTes3Symbol(book.textId);
    auto& state = m_tes3.playerState().progression;
    if (state.readBooks.contains(id)) { error.clear(); return true; }
    const int skill = readSub<std::int32_t>(*data, 12);
    if (skill != -1) {
        if (skill < 0 || skill >= 27) { error = "Invalid skill-book skill"; return false; }
        const auto& stats = m_tes3.playerState().numericFilters;
        const auto it = stats.find(std::string(tes3SkillNames[skill]));
        if (it == stats.end()) { error = "Skill-book reader has no base skill"; return false; }
        if (it->second < 100 && !m_tes3.advancePlayerSkill(skill, Tes3SkillAdvanceSource::Book, error)) return false;
    }
    state.readBooks.insert(id); m_tes3.synchronizePlayerDialogue(); error.clear(); return true;
}
int BethesdaSession::tes3DerivedDisposition(ObjectId npc, bool clamp) const {
    const auto* object = m_world.find(npc);
    const auto* player = m_world.find(m_playerObject);
    const Tes3ActorDefinition* definition = nullptr;
    if (m_tes3.content()) {
        if (object) definition = m_tes3.content()->findActor(object->base.recordType, object->base.textId);
        else if (const auto ref = m_tes3.content()->references().find(npc); ref != m_tes3.content()->references().end())
            definition = m_tes3.content()->findActor(ref->second.base.recordType, ref->second.base.textId);
    }
    if (!definition || definition->creature || !player) return 0;
    const auto& content = *m_tes3.content();
    const auto& pc = m_tes3.playerState();
    const auto filter = [&](const std::string& key, double fallback = 0) {
        const auto it = pc.numericFilters.find(key);
        return it == pc.numericFilters.end() ? fallback : it->second;
    };
    const auto setting = [&](std::string_view id, double fallback) { return tes3NumericSetting(content, id, fallback); };
    float disposition = definition->disposition;
    if (const auto it = m_tes3.referenceOverrides().find(npc); it != m_tes3.referenceOverrides().end()) {
        if (const auto base = it->second.locals.find("stat:disposition"); base != it->second.locals.end())
            disposition = base->second.number;
    }
    if (normalizeTes3Symbol(definition->race) == normalizeTes3Symbol(pc.race))
        disposition += setting("fDispRaceMod", 5);
    disposition += setting("fDispPersonalityMult", .5) *
        (modifiedTes3Stat(m_playerObject, "personality") - setting("fDispPersonalityBase", 50));
    const auto factionId = normalizeTes3Symbol(definition->faction.textId);
    const auto* faction = content.findFaction(factionId);
    double reaction = 0;
    int rank = 0;
    const auto reactionTo = [&](const std::string& id) {
        double value = filter("faction_reaction:" + factionId + ":" + id);
        if (faction) if (const auto authored = faction->reactions.find(id); authored != faction->reactions.end())
            value += authored->second;
        return value;
    };
    if (const auto own = pc.factionRanks.find(factionId); own != pc.factionRanks.end()) {
        if (filter("expelled:" + factionId) == 0) { reaction = reactionTo(factionId); rank = own->second; }
    } else if (!factionId.empty()) {
        bool first = true;
        for (const auto& [id, pcRank] : pc.factionRanks) {
            if (filter("expelled:" + id) != 0) continue;
            const auto value = reactionTo(id);
            if (first || value < reaction) { first = false; reaction = value; rank = pcRank; }
        }
    }
    disposition += (setting("fDispFactionRankMult", .5) * rank + setting("fDispFactionRankBase", 1)) *
        setting("fDispFactionMod", 3) * reaction;
    const double bounty = filter("crimelevel", player->livingState ? player->livingState->bounty : 0);
    disposition -= setting("fDispCrimeMod", 0) * bounty;
    bool diseased = false;
    for (const auto& item : player->inventory) if (item.count > 0 && item.item.recordType == "SPEL") {
        const auto* spell = content.findSpell(item.item.textId);
        if (spell && (spell->type == 2 || spell->type == 3)) diseased = true;
    }
    if (const auto active = m_tes3.activeSpells().find(m_playerObject); active != m_tes3.activeSpells().end())
        for (const auto& item : active->second) {
            const auto* spell = content.findSpell(item.spell.textId);
            if (spell && (spell->type == 2 || spell->type == 3) && std::any_of(item.effects.begin(), item.effects.end(),
                [&](const auto& effect) { return effect.expiresTick > m_clock.tick(); })) diseased = true;
        }
    if (diseased) disposition += setting("fDispDiseaseMod", -10);
    if (player->equipment.drawn) disposition += setting("fDispWeaponDrawn", -5);
    disposition += tes3MagicEffectMagnitude(npc, 44); // Charm
    if (m_tes3.dialogue().active && m_tes3.dialogue().actor.object == npc)
        disposition += m_tes3.dialogue().persuasionTemporary;
    if (!std::isfinite(disposition)) return 0;
    if (clamp) return int(std::clamp(disposition, 0.0f, 100.0f));
    return int(std::clamp(double(disposition), double(std::numeric_limits<int>::min()), double(std::numeric_limits<int>::max())));
}
void BethesdaSession::finishTes3Persuasion() {
    auto& dialogue = m_tes3.dialogueForRestore();
    if (dialogue.persuasionTemporary == 0 && dialogue.persuasionPermanent == 0) return;
    const auto npc = dialogue.actor.object;
    const int permanent = dialogue.persuasionPermanent;
    dialogue.persuasionTemporary = dialogue.persuasionPermanent = 0;
    const auto* object = m_world.find(npc);
    const Tes3ActorDefinition* definition = nullptr;
    if (m_tes3.content()) {
        if (object) definition = m_tes3.content()->findActor(object->base.recordType, object->base.textId);
        else if (const auto ref = m_tes3.content()->references().find(npc); ref != m_tes3.content()->references().end())
            definition = m_tes3.content()->findActor(ref->second.base.recordType, ref->second.base.textId);
    }
    if (!definition || definition->creature) return;
    auto& locals = m_tes3.referenceOverridesForRestore()[npc].locals;
    const auto found = locals.find("stat:disposition");
    const double base = found == locals.end() ? definition->disposition : found->second.number;
    // Clamp the permanent result against the current non-Charm modifiers.
    // Temporary Charm must never become a permanent dislike when it expires.
    const int zero = int(tes3DerivedDisposition(npc, false) - base - tes3MagicEffectMagnitude(npc, 44));
    const double next = std::clamp(base + permanent, -double(zero), 100.0 - zero);
    locals["stat:disposition"] = Tes3Value::fromNumber(next);
    dialogue.actor.disposition = float(next);
}
BethesdaSession::Tes3PersuasionResult BethesdaSession::persuadeTes3Npc(Tes3PersuasionAction action, std::string& error) {
    Tes3PersuasionResult result;
    auto& dialogue = m_tes3.dialogueForRestore();
    auto* pc = m_world.find(m_playerObject);
    auto* npc = m_world.find(dialogue.actor.object);
    const auto* definition = npc && m_tes3.content() ? m_tes3.content()->findActor(npc->base.recordType, npc->base.textId) : nullptr;
    if (!dialogue.active || dialogue.goodbye || !dialogue.choices.empty() || !definition || definition->creature ||
        !pc || !pc->actorValues || pc->actorValues->dead || !npc->actorValues || npc->actorValues->dead ||
        !inReach(*pc, *npc, 256) || m_tes3.hasPendingResultTransaction() || m_tes3.playerState().progression.selectionOpen ||
        int(action) < 0 || int(action) > 5) {
        error = "Persuasion requires an available living NPC conversation"; return result;
    }
    const auto& content = *m_tes3.content();
    const auto setting = [&](std::string_view id, float fallback) { return float(tes3NumericSetting(content, id, fallback)); };
    const float personalityMod = setting("fPersonalityMod", 5), luckMod = setting("fLuckMod", 10);
    const float minimumChance = setting("iPerMinChance", 5), minimumChange = setting("iPerMinChange", 10);
    const float dieMult = setting("fPerDieRollMult", .3f), tempMult = setting("fPerTempMult", 1);
    const auto speechBase = m_tes3.playerState().numericFilters.find("speechcraft");
    const double requirement = m_tes3.playerSkillRequirement(25);
    if (!std::isfinite(personalityMod) || !std::isfinite(luckMod) || personalityMod <= 0 || luckMod <= 0 ||
        !std::isfinite(minimumChance) || !std::isfinite(minimumChange) || minimumChange < 0 ||
        !std::isfinite(dieMult) || dieMult < 0 || !std::isfinite(tempMult) || tempMult <= 0 ||
        !std::isfinite(requirement) || requirement <= 0 || speechBase == m_tes3.playerState().numericFilters.end() ||
        !std::isfinite(speechBase->second) || !tes3SkillDefinition(content, 25)) {
        error = "Invalid TES3 persuasion or advancement settings"; return result;
    }
    const auto gold = makeTes3RecordKey("MISC", "gold_001");
    const auto money = std::find_if(pc->inventory.begin(), pc->inventory.end(), [&](const auto& item) { return item.item == gold; });
    const int bribe = action == Tes3PersuasionAction::Bribe10 ? 10 : action == Tes3PersuasionAction::Bribe100 ? 100 :
        action == Tes3PersuasionAction::Bribe1000 ? 1000 : 0;
    if (bribe && (money == pc->inventory.end() || money->count < bribe)) {
        error = "Insufficient gold for the offered bribe"; return result;
    }
    const auto rating = [&](ObjectId actor, bool player) {
        const auto& values = *(player ? pc : npc)->actorValues;
        const float fatigue = setting("fFatigueBase", 1.25f) - setting("fFatigueMult", .5f) *
            (1 - (std::floor(values.maxStamina) == 0 ? 1.f : std::max(0.f, values.stamina / values.maxStamina)));
        float reputation = definition->reputation, level = definition->level;
        if (player) {
            const auto& filters = m_tes3.playerState().numericFilters;
            if (const auto it = filters.find("reputation"); it != filters.end()) reputation = float(it->second); else reputation = 0;
            if (const auto it = filters.find("level"); it != filters.end()) level = float(it->second);
        } else if (const auto it = m_tes3.referenceOverrides().find(actor); it != m_tes3.referenceOverrides().end()) {
            if (const auto v = it->second.locals.find("stat:reputation"); v != it->second.locals.end()) reputation = float(v->second.number);
            if (const auto v = it->second.locals.find("stat:level"); v != it->second.locals.end()) level = float(v->second.number);
        }
        const float rep = reputation * setting("fReputationMod", 1), lev = level * setting("fLevelMod", 5);
        const float pers = float(modifiedTes3Stat(actor, "personality")) / personalityMod;
        const float luck = float(modifiedTes3Stat(actor, "luck")) / luckMod;
        const float speech = float(modifiedTes3Stat(actor, "speechcraft")), merchant = float(modifiedTes3Stat(actor, "mercantile"));
        const float first = (rep + luck + pers + speech) * fatigue;
        return std::array<float, 3>{first, player ? first + lev : (lev + rep + luck + pers + speech) * fatigue,
            (merchant + (player ? 0 : rep) + luck + pers) * fatigue};
    };
    const auto a = rating(m_playerObject, true), b = rating(npc->id, false);
    const int disposition = tes3DerivedDisposition(npc->id);
    const float d = 1 - .02f * std::abs(disposition - 50);
    const int ratingIndex = action == Tes3PersuasionAction::Intimidate ? 1 : bribe ? 2 : 0;
    float chance = d * (a[ratingIndex] - b[ratingIndex] + 50);
    if (bribe) chance += setting(bribe == 10 ? "fBribe10Mod" : bribe == 100 ? "fBribe100Mod" : "fBribe1000Mod",
        bribe == 10 ? 35 : bribe == 100 ? 75 : 150);
    chance = std::max(minimumChance, chance);
    std::uint32_t random = m_randomState;
    random ^= random << 13; random ^= random >> 17; random ^= random << 5;
    if (!random) random = 1;
    const int roll = random % 100;
    const bool success = roll <= chance;
    float temporary = 0, permanent = 0, aiDelta = 0;
    if (action == Tes3PersuasionAction::Intimidate) {
        const float r = roll != chance ? std::floor(chance - roll) : 1;
        const float c = -std::abs(std::floor(r * dieMult));
        aiDelta = success ? std::floor(r * dieMult * tempMult) : 0;
        // Preserve vanilla's marginal-win behavior, rather than OpenMW/MCP's
        // fix which grants a temporary increase in this case.
        if (success && std::abs(c) < minimumChange) { temporary = 0; permanent = -minimumChange; }
        else { temporary = success ? -std::floor(c * tempMult) : std::floor(c * tempMult);
            permanent = success ? -std::trunc(temporary / tempMult) : c; }
    } else if (action == Tes3PersuasionAction::Taunt) {
        const float c = std::abs(std::floor(chance - roll));
        aiDelta = success ? c * dieMult * tempMult : 0;
        const float change = std::floor(-c * dieMult);
        temporary = (success ? std::min(-minimumChange, change) : change) * tempMult;
    } else {
        const float c = std::floor(dieMult * (chance - roll));
        temporary = (success ? std::max(minimumChange, c) : c) * tempMult;
    }
    if (!std::isfinite(chance) || !std::isfinite(temporary) || !std::isfinite(permanent) || !std::isfinite(aiDelta) ||
        std::abs(temporary) > 1000000 || std::abs(permanent) > 1000000 || std::abs(aiDelta) > 1000000) {
        error = "Invalid TES3 persuasion outcome"; return result;
    }
    const int temp = std::clamp(int(temporary), -disposition, 100 - disposition);
    float permValue = action == Tes3PersuasionAction::Intimidate ? permanent : std::trunc(temp / tempMult);
    if (action == Tes3PersuasionAction::Intimidate && success && temporary != 0) permValue = -std::trunc(temp / tempMult);
    if (!std::isfinite(permValue) || std::abs(permValue) > 1000000) { error = "Invalid permanent persuasion change"; return result; }
    int perm = int(permValue);
    double base = definition->disposition;
    if (const auto it = m_tes3.referenceOverrides().find(npc->id); it != m_tes3.referenceOverrides().end())
        if (const auto value = it->second.locals.find("stat:disposition"); value != it->second.locals.end()) base = value->second.number;
    if (temp > 0 && perm > 0 && base + dialogue.persuasionPermanent + perm < 0) {
        const double recovery = -(base + dialogue.persuasionPermanent);
        if (!std::isfinite(recovery) || recovery > 1000000) { error = "Invalid persuasion base disposition"; return result; }
        perm = int(recovery);
    }
    const auto sumTemp = std::int64_t(dialogue.persuasionTemporary) + temp;
    const auto sumPerm = std::int64_t(dialogue.persuasionPermanent) + perm;
    if (std::abs(sumTemp) > 1000000 || std::abs(sumPerm) > 1000000) { error = "Persuasion change limit exceeded"; return result; }
    const auto recipient = std::find_if(npc->inventory.begin(), npc->inventory.end(), [&](const auto& item) { return item.item == gold; });
    if (bribe && success && recipient != npc->inventory.end() && recipient->count > std::numeric_limits<int>::max() - bribe) {
        error = "NPC gold stack overflow"; return result;
    }
    if (speechBase->second < 100 &&
        !m_tes3.usePlayerSkill(25, success ? 0 : 1, 1, error)) return result;
    m_randomState = random;
    dialogue.persuasionTemporary = int(sumTemp); dialogue.persuasionPermanent = int(sumPerm);
    if (success && (action == Tes3PersuasionAction::Intimidate || action == Tes3PersuasionAction::Taunt)) {
        auto& locals = m_tes3.referenceOverridesForRestore()[npc->id].locals;
        const auto adjust = [&](const char* name, int baseline, double delta) {
            const auto it = locals.find(name);
            locals[name] = Tes3Value::fromNumber(std::clamp((it == locals.end() ? baseline : it->second.number) + delta, 0.0, 100.0));
        };
        const double change = std::max(minimumChange, aiDelta);
        const bool intimidate = action == Tes3PersuasionAction::Intimidate;
        adjust("stat:fight", definition->fight, intimidate ? -change : change);
        adjust("stat:flee", definition->flee, intimidate ? change : -change);
    }
    if (bribe && success) {
        money->count -= bribe;
        if (recipient == npc->inventory.end()) npc->inventory.push_back({gold, bribe, false}); else recipient->count += bribe;
        std::erase_if(pc->inventory, [](const auto& item) { return item.count <= 0; });
        const auto balance = std::find_if(npc->inventory.begin(), npc->inventory.end(), [&](const auto& item) { return item.item == gold; });
        m_tes3.referenceOverridesForRestore()[npc->id].locals["inventory:" + gold.toString()] = Tes3Value::fromNumber(balance->count);
        syncTes3PlayerInventory();
    }
    m_tes3.synchronizePlayerDialogue();
    result.accepted = true; result.success = success; result.temporaryChange = temp; result.permanentChange = perm;
    const auto name = action == Tes3PersuasionAction::Admire ? "Admire" : action == Tes3PersuasionAction::Intimidate ? "Intimidate" :
        action == Tes3PersuasionAction::Taunt ? "Taunt" : "Bribe";
    result.response = m_tes3.selectPersuasionResponse(std::string(name) + (success ? " Success" : " Fail"));
    error.clear(); return result;
}
std::vector<BethesdaSession::Tes3RepairOption> BethesdaSession::tes3RepairTools() const {
    std::vector<Tes3RepairOption> options;
    const auto* player = m_world.find(m_playerObject);
    if (!player || !m_tes3.content() || !player->enabled || !player->actorValues || player->actorValues->dead ||
        m_tes3.dialogue().active || m_tes3.hasPendingResultTransaction() || m_tes3.playerState().progression.selectionOpen) return options;
    for (std::size_t i = 0; i < player->inventory.size(); ++i) {
        const auto& entry = player->inventory[i];
        if (entry.count <= 0 || entry.item.recordType != "REPA") continue;
        const int maximum = itemMaximumCondition(*m_tes3.content(), entry.item);
        const int uses = entry.condition == -1 ? maximum : entry.condition;
        if (maximum > 0 && uses > 0 && uses <= maximum) options.push_back({i, entry.item, uses, maximum});
    }
    return options;
}
std::vector<BethesdaSession::Tes3RepairOption> BethesdaSession::tes3RepairTargets() const {
    std::vector<Tes3RepairOption> options;
    if (tes3RepairTools().empty()) return options;
    const auto* player = m_world.find(m_playerObject);
    for (std::size_t i = 0; i < player->inventory.size(); ++i) {
        const auto& entry = player->inventory[i];
        if (entry.count <= 0 || (entry.item.recordType != "WEAP" && entry.item.recordType != "ARMO")) continue;
        const int maximum = itemMaximumCondition(*m_tes3.content(), entry.item);
        if (maximum > 0 && entry.condition >= 0 && entry.condition < maximum)
            options.push_back({i, entry.item, entry.condition, maximum});
    }
    return options;
}
BethesdaSession::Tes3RepairResult BethesdaSession::repairTes3PlayerItem(
    std::size_t itemIndex, std::size_t toolIndex, std::string& error) {
    Tes3RepairResult result;
    const auto tools = tes3RepairTools(), targets = tes3RepairTargets();
    const auto tool = std::find_if(tools.begin(), tools.end(), [&](const auto& value) { return value.inventoryIndex == toolIndex; });
    const auto target = std::find_if(targets.begin(), targets.end(), [&](const auto& value) { return value.inventoryIndex == itemIndex; });
    if (tool == tools.end() || target == targets.end() || itemIndex == toolIndex) {
        error = "Repair requires an owned usable tool and damaged weapon or armor"; return result;
    }
    auto* player = m_world.find(m_playerObject);
    const auto& content = *m_tes3.content();
    const auto* data = findSub(content.findRecord("REPA", tool->item.textId), "RIDT");
    const float quality = readSub<float>(*data, 12);
    const float multiplier = float(tes3NumericSetting(content, "fRepairAmountMult", 3));
    const float fatigueBase = float(tes3NumericSetting(content, "fFatigueBase", 1.25));
    const float fatigueMult = float(tes3NumericSetting(content, "fFatigueMult", .5));
    const auto& values = *player->actorValues;
    const float fatigue = fatigueBase - fatigueMult * (1 - (std::floor(values.maxStamina) == 0 ? 1.f : std::max(0.f, values.stamina / values.maxStamina)));
    const float chance = (float(modifiedTes3Stat(m_playerObject, "strength")) * .1f +
        float(modifiedTes3Stat(m_playerObject, "luck")) * .1f + float(modifiedTes3Stat(m_playerObject, "armorer"))) * fatigue;
    const auto base = m_tes3.playerState().numericFilters.find("armorer");
    if (!std::isfinite(multiplier) || multiplier < 0 || !std::isfinite(fatigue) || !std::isfinite(chance) ||
        base == m_tes3.playerState().numericFilters.end() || !std::isfinite(base->second) ||
        !tes3SkillDefinition(content, 1) || !std::isfinite(m_tes3.playerSkillRequirement(1)) || m_tes3.playerSkillRequirement(1) <= 0) {
        error = "Invalid TES3 repair or advancement settings"; return result;
    }
    std::uint32_t random = m_randomState;
    random ^= random << 13; random ^= random >> 17; random ^= random << 5; if (!random) random = 1;
    const int roll = random % 100;
    const bool success = roll <= chance;
    const float amount = multiplier * quality * roll;
    if (!std::isfinite(amount) || amount > std::numeric_limits<int>::max() - 1.0) {
        error = "Invalid TES3 repair amount"; return result;
    }
    const int repaired = success ? std::min(target->maximum - target->condition, std::max(1, int(amount))) : 0;
    auto inventory = player->inventory;
    const auto replaceOne = [&](std::size_t index, int condition, bool consume) {
        auto instance = inventory[index];
        --inventory[index].count;
        if (instance.equipped) inventory[index].equipped = false;
        if (!consume) { instance.count = 1; instance.condition = condition; inventory.push_back(std::move(instance)); }
    };
    replaceOne(toolIndex, tool->condition - 1, tool->condition == 1);
    if (success) replaceOne(itemIndex, target->condition + repaired == target->maximum ? -1 : target->condition + repaired, false);
    std::erase_if(inventory, [](const auto& entry) { return entry.count <= 0; });
    std::vector<InventoryEntry> stacked;
    for (const auto& entry : inventory) {
        auto found = std::find_if(stacked.begin(), stacked.end(), [&](const auto& value) {
            auto candidate = value; candidate.count = entry.count; return candidate == entry;
        });
        if (found == stacked.end()) stacked.push_back(entry); else found->count += entry.count;
    }
    if (success && base->second < 100 && !m_tes3.usePlayerSkill(1, 0, 1, error)) return result;
    player->inventory = std::move(stacked); m_randomState = random;
    // Ordinary item use breaks active invisibility even on a failed attempt.
    if (auto spells = m_tes3.activeSpellsForRestore().find(m_playerObject); spells != m_tes3.activeSpellsForRestore().end()) {
        for (auto& spell : spells->second) std::erase_if(spell.effects, [](const auto& effect) { return effect.effectId == 39; });
        std::erase_if(spells->second, [](const auto& spell) { return spell.effects.empty(); });
        if (spells->second.empty()) m_tes3.activeSpellsForRestore().erase(spells);
    }
    syncTes3PlayerInventory();
    if (success) {
        if (const auto* script = findSub(content.findRecord(target->item.recordType, target->item.textId), "SCRI")) {
            const auto end = std::find(script->data.begin(), script->data.end(), 0);
            const std::string name(script->data.begin(), end);
            std::uint64_t threadId = 0;
            for (const auto& [id, thread] : m_tes3.scripts().threads())
                if (thread.program == normalizeTes3Symbol(name) && thread.owner == m_playerObject && thread.repeat) { threadId = id; break; }
            if (!threadId && !name.empty()) {
                std::string scriptError; threadId = m_tes3.scripts().start(name, m_playerObject, scriptError, true, true);
                if (!threadId) m_pendingDiagnostics.push_back("TES3 repair item script: " + scriptError);
            }
            if (threadId) m_tes3.dispatchScriptEvent(threadId, "onpcrepair", Tes3Value::fromNumber(1));
        }
    }
    result = {true, success, repaired, tool->condition == 1}; error.clear(); return result;
}
std::vector<BethesdaSession::Tes3TrainingOffer> BethesdaSession::tes3TrainingOffers(ObjectId trainer) const {
    const auto* player = m_world.find(m_playerObject);
    const auto* npc = m_world.find(trainer);
    const auto* def = npc && m_tes3.content() ? m_tes3.content()->findActor(npc->base.recordType, npc->base.textId) : nullptr;
    if (!player || !npc || !def || def->creature || !(def->serviceFlags & 0x4000) || !inReach(*player, *npc, 256) ||
        !player->actorValues || player->actorValues->dead || !npc->actorValues || npc->actorValues->dead ||
        m_tes3.playerState().progression.selectionOpen || m_tes3.hasPendingResultTransaction()) return {};
    std::vector<int> skills;
    for (int i = 0; i < 27; ++i) if (tes3SkillDefinition(*m_tes3.content(), i)) skills.push_back(i);
    std::stable_sort(skills.begin(), skills.end(), [&](int a, int b) { return modifiedTes3Stat(trainer, tes3SkillNames[a]) > modifiedTes3Stat(trainer, tes3SkillNames[b]); });
    if (skills.size() > 3) skills.resize(3);
    const double disposition = tes3DerivedDisposition(trainer);
    const auto fatigueTerm = [&](const ActorValues& av) {
        return tes3NumericSetting(*m_tes3.content(), "fFatigueBase", 1.25) -
            tes3NumericSetting(*m_tes3.content(), "fFatigueMult", .5) *
            (1 - (av.maxStamina > 0 ? std::max(0.0f, av.stamina / av.maxStamina) : 1));
    };
    const double pcTerm = (std::clamp(disposition, 0.0, 100.0) - 50 + std::min(100.0, modifiedTes3Stat(m_playerObject, "mercantile")) +
        std::min(10.0, .1 * modifiedTes3Stat(m_playerObject, "luck")) + std::min(10.0, .2 * modifiedTes3Stat(m_playerObject, "personality"))) * fatigueTerm(*player->actorValues);
    const double npcTerm = (std::min(100.0, modifiedTes3Stat(trainer, "mercantile")) + std::min(10.0, .1 * modifiedTes3Stat(trainer, "luck")) +
        std::min(10.0, .2 * modifiedTes3Stat(trainer, "personality"))) * fatigueTerm(*npc->actorValues);
    const auto goldKey = makeTes3RecordKey("MISC", "gold_001");
    int gold = 0;
    for (const auto& e : player->inventory) if (e.item == goldKey) gold = e.count;
    std::vector<Tes3TrainingOffer> result;
    const double mod = tes3NumericSetting(*m_tes3.content(), "iTrainingMod", 10);
    if (!std::isfinite(mod) || mod < 0 || !std::isfinite(pcTerm) || !std::isfinite(npcTerm)) return {};
    for (int skill : skills) {
        const auto stat = m_tes3.playerState().numericFilters.find(std::string(tes3SkillNames[skill]));
        if (stat == m_tes3.playerState().numericFilters.end()) continue;
        const double base = stat->second;
        const double rawBase = base * mod;
        if (!std::isfinite(rawBase) || rawBase < 0 || rawBase > std::numeric_limits<int>::max()) continue;
        const int basePrice = std::max(1, static_cast<int>(rawBase));
        const double price = basePrice * .01 * (100 - .5 * (pcTerm - npcTerm));
        if (!std::isfinite(price) || price > std::numeric_limits<int>::max()) continue;
        const int cost = std::max(1, static_cast<int>(std::max(0.0, price)));
        const auto s = tes3SkillDefinition(*m_tes3.content(), skill);
        result.push_back({skill, cost, base < 100 && base < modifiedTes3Stat(trainer, tes3SkillNames[skill]) &&
            base < modifiedTes3Stat(m_playerObject, tes3AttributeNames[s->attribute]) && gold >= cost});
    }
    return result;
}
bool BethesdaSession::trainTes3PlayerSkill(ObjectId trainer, int skill, std::string& error) {
    const auto offers = tes3TrainingOffers(trainer);
    const auto offer = std::find_if(offers.begin(), offers.end(), [&](const auto& o) { return o.skill == skill && o.eligible; });
    if (offer == offers.end()) { error = "Training is unavailable, unaffordable, or exceeds trainer/attribute limits"; return false; }
    if (!m_tes3.advancePlayerSkill(skill, Tes3SkillAdvanceSource::Training, error)) return false;
    auto* player = m_world.find(m_playerObject);
    for (auto& e : player->inventory) if (e.item == makeTes3RecordKey("MISC", "gold_001")) e.count -= offer->price;
    std::erase_if(player->inventory, [](const auto& e) { return e.count <= 0; });
    auto& pool = m_tes3.referenceOverridesForRestore()[trainer].locals["stat:goldpool"];
    pool = Tes3Value::fromNumber(pool.number + offer->price);
    syncTes3PlayerInventory();
    advanceTes3RestHour(false);
    advanceTes3RestHour(false);
    m_tes3.synchronizePlayerDialogue(); error.clear(); return true;
}
void BethesdaSession::advanceTes3RestHour(bool sleep) {
    const auto minute = m_livingWorld.absoluteGameMinute();
    (void)m_livingWorld.advanceGameMinutes(60, m_world);
    // Rest time still advances when routines are disabled or no cells have
    // schedules; advanceGameMinutes otherwise deliberately does no work.
    if (m_livingWorld.absoluteGameMinute() == minute)
        m_livingWorld.restoreClock(minute + 60, m_livingWorld.fractionalGameMinute());
    auto& globals = m_tes3.scripts().globals();
    const auto global = [&](const std::string& key, double fallback) {
        const auto it = globals.find(key);
        return it == globals.end() || !std::isfinite(it->second.number) ? fallback : it->second.number;
    };
    // Active TES3 spell durations use real seconds at 60 ticks/second.
    // Skip their remaining duration rather than running thousands of physics
    // or script ticks while the player is in the rest menu.
    const double timeScale = global("timescale", 30);
    const double elapsedTicks = 3600.0 * 60 / (timeScale > 0 ? timeScale : 1);
    const auto now = m_clock.tick();
    const double time = global("gamehour", 8) + 1;
    globals["gamehour"] = Tes3Value::fromNumber(std::fmod(time, 24));
    if (time >= 24) {
        globals["dayspassed"] = Tes3Value::fromNumber(global("dayspassed", 0) + 1);
        int day = int(std::clamp(global("day", 1), 1.0, 31.0)) + 1;
        int month = int(std::clamp(global("month", 0), 0.0, 11.0));
        constexpr int days[]{31,28,31,30,31,30,31,31,30,31,30,31};
        if (day > days[month]) {
            day = 1; ++month;
            if (month == 12) { month = 0; globals["year"] = Tes3Value::fromNumber(global("year", 427) + 1); }
        }
        globals["day"] = Tes3Value::fromNumber(day);
        globals["month"] = Tes3Value::fromNumber(month);
    }
    auto* player = m_world.find(m_playerObject);
    if (!player || !player->actorValues) return;
    auto& stats = m_tes3.playerState().numericFilters;
    stats["gamehour"] = globals["gamehour"].number;
    auto& av = *player->actorValues;
    if (sleep) {
        av.health = std::min(av.maxHealth, av.health + float(modifiedTes3Stat(m_playerObject, "endurance") * .1));
        double blockedFraction = 0;
        if (const auto it = m_tes3.activeSpells().find(m_playerObject); it != m_tes3.activeSpells().end())
            for (const auto& spell : it->second) for (const auto& effect : spell.effects)
                if (effect.effectId == 136 && effect.magnitude > 0 && effect.expiresTick > now)
                    blockedFraction = std::max(blockedFraction, std::min(1.0,
                        double(effect.expiresTick - now) / elapsedTicks));
        for (const auto& [id, count] : m_tes3.playerState().inventory) {
            if (id.recordType != "SPEL" || count <= 0) continue;
            const auto* spell = m_tes3.content()->findSpell(id.textId);
            if (spell && spell->type == 1) for (const auto& effect : spell->effects)
                if (effect.effectId == 136) blockedFraction = 1;
        }
        av.magicka = std::min(av.maxMagicka, av.magicka + float(modifiedTes3Stat(m_playerObject, "intelligence") *
            tes3NumericSetting(*m_tes3.content(), "fRestMagicMult", .15) * (1 - blockedFraction)));
    }
    auto& active = m_tes3.activeSpellsForRestore();
    for (auto target = active.begin(); target != active.end();) {
        auto& spells = target->second;
        std::erase_if(spells, [&](Tes3ActiveSpell& spell) {
            std::erase_if(spell.effects, [&](Tes3ActiveSpellEffect& effect) {
                if (effect.expiresTick == std::numeric_limits<std::uint64_t>::max()) return false;
                if (effect.expiresTick <= now || double(effect.expiresTick - now) <= elapsedTicks) return true;
                effect.expiresTick -= static_cast<std::uint64_t>(elapsedTicks);
                return false;
            });
            return spell.effects.empty();
        });
        if (spells.empty()) target = active.erase(target);
        else ++target;
    }
    av.stamina = std::max(av.stamina, av.maxStamina);
    stats["health"] = av.health; stats["maxhealth"] = av.maxHealth;
    stats["magicka"] = av.magicka; stats["maxmagicka"] = av.maxMagicka;
    stats["fatigue"] = av.stamina; stats["maxfatigue"] = av.maxStamina;
    m_tes3.synchronizePlayerDialogue();
}
bool BethesdaSession::restTes3Player(int hours, bool sleep, std::string& error, ObjectId bed) {
    auto* player = m_world.find(m_playerObject);
    if (!m_tes3.content() || !player || !player->actorValues || player->actorValues->dead || hours < 1 || hours > 24 ||
        m_tes3.dialogue().active || m_tes3.hasPendingResultTransaction() || m_tes3.playerState().progression.selectionOpen) {
        error = "Rest requires a living player, 1–24 hours, and no active interaction"; return false;
    }
    if (const auto it = m_tes3.playerState().numericFilters.find("control:rest"); it != m_tes3.playerState().numericFilters.end() && it->second == 0) {
        error = "Rest is disabled"; return false;
    }
    if (const auto physical = m_physics.characterState(m_playerObject); physical && !physical->grounded && !bed.valid()) {
        error = "Cannot rest in the air"; return false;
    }
    bool usingBed = false;
    if (bed.valid()) {
        const auto* object = m_world.find(bed);
        const auto ref = m_tes3.content()->references().find(bed);
        if (!object || ref == m_tes3.content()->references().end() || !inReach(*player, *object, 192)) {
            error = "Bed is not in reach"; return false;
        }
        const auto* record = m_tes3.content()->findRecord(object->base.recordType, object->base.textId);
        const auto* scriptName = findSub(record, "SCRI");
        if (scriptName) {
            const auto end = std::find(scriptName->data.begin(), scriptName->data.end(), std::uint8_t{0});
            const std::string id(scriptName->data.begin(), end);
            const auto* script = m_tes3.content()->findScript(id);
            usingBed = script && normalizeTes3Symbol(script->source).find("showrestmenu") != std::string::npos;
        }
        if (!usingBed) { error = "Rest target is not an authored bed"; return false; }
        for (const auto& sub : ref->second.subrecords) if (sub.type == "ANAM" || sub.type == "CNAM") {
            const auto end = std::find(sub.data.begin(), sub.data.end(), std::uint8_t{0});
            const std::string owner(sub.data.begin(), end);
            if (!owner.empty() && normalizeTes3Symbol(owner) != "player") {
                error = "Cannot sleep in an owned bed"; return false;
            }
        }
    }
    const auto* cell = m_tes3.content()->findRecord("CELL", player->currentSpace.cell.textId);
    const auto* data = findSub(cell, "DATA");
    if (!data || data->data.size() < 4) { error = "Rest location has no TES3 cell policy"; return false; }
    if (sleep && !usingBed && (readSub<std::uint32_t>(*data) & 4)) { error = "Only waiting is allowed here"; return false; }
    const auto* water = findSub(cell, "WHGT");
    if (water && water->data.size() == 4 && player->transform.position[1] + 64 < readSub<float>(*water)) {
        error = "Cannot rest underwater"; return false;
    }
    const auto threatened = [&]() {
        for (auto id : m_world.orderedActorIds()) {
            const auto* actor = m_world.find(id);
            if (id != m_playerObject && actor && actor->combatState && actor->combatState->combatTarget == m_playerObject &&
                (!actor->actorValues || !actor->actorValues->dead) && inReach(*player, *actor, 2048)) return true;
        }
        return false;
    };
    if (threatened()) { error = "Cannot rest with enemies nearby"; return false; }
    for (int hour = 0; hour < hours; ++hour) {
        advanceTes3RestHour(sleep);
        if (threatened()) { error = "Rest was interrupted"; return false; }
    }
    if (sleep) (void)m_tes3.openPlayerLevelSelection();
    m_tes3.synchronizePlayerDialogue(); error.clear(); return true;
}
int BethesdaSession::tes3ActorWeaponSkill(ObjectId actor) const {
    const auto* object = m_world.find(actor);
    if (!object || !m_tes3.content()) return -1;
    for (const auto& entry : object->inventory) {
        if (entry.equipped && entry.count > 0 && entry.item.recordType == "WEAP") {
            const int skill = tes3WeaponSkill(*m_tes3.content(), entry.item.textId).value_or(-1);
            return skill == 23 ? -1 : skill; // ranged weapons are not melee skill uses
        }
    }
    return 26;
}
double BethesdaSession::tes3MeleeHitChance(ObjectId attacker, ObjectId target) const {
    const auto* source = m_world.find(attacker);
    const auto* victim = m_world.find(target);
    const int skill = tes3ActorWeaponSkill(attacker);
    if (!source || !victim || !source->actorValues || !victim->actorValues || !m_tes3.content() || skill < 0) return 0;
    const auto& content = *m_tes3.content();
    const auto fatigue = [&](const RuntimeObject& actor) {
        const auto& values = *actor.actorValues;
        const double normalized = std::floor(values.maxStamina) == 0 ? 1 : std::max(0.0f, values.stamina / values.maxStamina);
        return tes3NumericSetting(content, "fFatigueBase", 1.25) -
            tes3NumericSetting(content, "fFatigueMult", .5) * (1 - normalized);
    };
    const auto magnitude = [&](ObjectId actor, int effectId) {
        return tes3MagicEffectMagnitude(actor, effectId);
    };
    double defense = 0;
    if (victim->actorValues->stamina >= 0) {
        if (magnitude(target, 45) <= 0) // Paralyze prevents evasion
            defense = (modifiedTes3Stat(target, "agility") / 5 + modifiedTes3Stat(target, "luck") / 10) * fatigue(*victim) +
                std::min(100.0, magnitude(target, 42)); // Sanctuary
        const double invisibility = tes3NumericSetting(content, "fCombatInvisoMult", .2);
        defense += std::min(100.0, invisibility * magnitude(target, 39)) +
            std::min(100.0, invisibility * magnitude(target, 40));
    }
    const auto* definition = content.findActor(source->base.recordType, source->base.textId);
    const double skillValue = definition && definition->creature
        ? modifiedTes3Stat(attacker, "combat") : modifiedTes3Stat(attacker, tes3SkillNames[skill]);
    const double attack = (std::floor(skillValue) + modifiedTes3Stat(attacker, "agility") / 5 +
        modifiedTes3Stat(attacker, "luck") / 10) * fatigue(*source) + magnitude(attacker, 117) - magnitude(attacker, 47);
    return std::round(attack - defense);
}
void BethesdaSession::awardTes3PlayerArmorHit(ObjectId attacker, std::uint64_t hitSequence) {
    const auto* player = m_world.find(m_playerObject);
    if (!player || !m_tes3.content()) return;
    const auto roll = core::mix64(m_clock.tick() ^ ObjectIdHash{}(attacker) ^ core::mix64(hitSequence)) % 100;
    const int slot = tes3ArmorHitSlot(static_cast<unsigned>(roll));
    int skill = 17;
    for (const auto& entry : player->inventory) {
        if (!entry.equipped || entry.count <= 0 || entry.item.recordType != "ARMO") continue;
        const auto armor = tes3ArmorDefinition(*m_tes3.content(), entry.item.textId);
        if (!armor) return; // malformed content must not invent an unarmored use
        if (armor->slot == slot) { skill = armor->skill; break; }
    }
    std::string ignored;
    (void)m_tes3.usePlayerSkill(skill, 0, 1, ignored);
}
void BethesdaSession::updateTes3PlayerWater() {
    if (!m_tes3.content()) return;
    const auto* player = m_world.find(m_playerObject);
    const auto* cell = player ? m_tes3.content()->findRecord("CELL", player->currentSpace.cell.textId) : nullptr;
    const auto* data = findSub(cell, "DATA");
    const auto* water = findSub(cell, "WHGT");
    std::optional<float> height;
    if (data && data->data.size() >= 4) {
        const auto flags = readSub<std::uint32_t>(*data);
        if (!(flags & 1) || (flags & 2)) height = water && water->data.size() == 4 ? readSub<float>(*water) : 0.f;
    }
    const auto scale = float(tes3NumericSetting(*m_tes3.content(), "fSwimHeightScale", .9));
    if (!std::isfinite(scale) || scale <= 0 || scale > 1 || (height && !std::isfinite(*height))) height.reset();
    (void)m_physics.setCharacterWaterLevel(m_playerObject, height,
        std::isfinite(scale) && scale > 0 && scale <= 1 ? scale : .9f);
}
void BethesdaSession::resolveTes3PlayerLanding(const PhysicsCharacterStep& physical) {
    const auto* player = m_world.find(m_playerObject);
    if (!player || !player->actorValues || player->actorValues->dead || !physical.landed || !m_tes3.content()) return;
    const auto& content = *m_tes3.content();
    const auto* cell = content.findRecord("CELL", player->currentSpace.cell.textId);
    const auto* cellData = findSub(cell, "DATA");
    const auto* water = findSub(cell, "WHGT");
    if (cellData && cellData->data.size() >= 4) {
        const auto flags = readSub<std::uint32_t>(*cellData);
        const bool hasWater = !(flags & 1) || (flags & 2);
        const double waterHeight = water && water->data.size() == 4 ? readSub<float>(*water) : 0;
        if (hasWater && physical.position.y < waterHeight) return;
    }
    double jump = 0;
    bool protectedFall = false;
    const auto effect = [&](int id, double magnitude) {
        if (id == 9) jump += magnitude;
        if ((id == 10 || id == 11) && magnitude > 0) protectedFall = true;
    };
    if (const auto it = m_tes3.activeSpells().find(m_playerObject); it != m_tes3.activeSpells().end())
        for (const auto& spell : it->second) for (const auto& e : spell.effects)
            if (e.expiresTick > m_clock.tick()) effect(e.effectId, e.magnitude);
    for (const auto& [key, count] : m_tes3.playerState().inventory) {
        if (key.recordType != "SPEL" || count <= 0) continue;
        const auto* spell = content.findSpell(key.textId);
        if (spell && spell->type == 1) for (const auto& e : spell->effects) effect(e.effectId, e.magnitudeMin);
    }
    if (protectedFall) return;
    const double minimum = tes3NumericSetting(content, "fFallDamageDistanceMin", 400);
    if (physical.fallDistanceUnits < minimum) return;
    const double acrobatics = modifiedTes3Stat(m_playerObject, "acrobatics");
    const double distance = std::max(0.0, physical.fallDistanceUnits - minimum - 1.5 * acrobatics - jump);
    const double damage = (tes3NumericSetting(content, "fFallDistanceBase", 0) +
        tes3NumericSetting(content, "fFallDistanceMult", .07) * distance) *
        (tes3NumericSetting(content, "fFallAcroBase", .25) +
         tes3NumericSetting(content, "fFallAcroMult", .01) * (100 - acrobatics));
    const auto& values = *player->actorValues;
    const double fatigue = tes3NumericSetting(content, "fFatigueBase", 1.25) -
        tes3NumericSetting(content, "fFatigueMult", .5) *
        (1 - (std::floor(values.maxStamina) == 0 ? 1 : std::max(0.0f, values.stamina / values.maxStamina)));
    const double healthLost = damage * (1 - .25 * fatigue);
    if (!std::isfinite(damage) || damage <= 0 || !std::isfinite(healthLost) || healthLost <= 0 ||
        healthLost > std::numeric_limits<float>::max()) return;
    WorldCommand hit;
    hit.type = WorldCommandType::AdjustActorValue; hit.target = m_playerObject;
    hit.actorValue = ActorValue::Health; hit.actorValueDelta = -static_cast<float>(healthLost);
    (void)m_world.queue(std::move(hit));
    if (damage <= acrobatics * fatigue) {
        std::string ignored;
        (void)m_tes3.usePlayerSkill(20, 1, 1, ignored);
    } else (void)queueActorAnimationEvent(m_playerObject, {"staggerStart", {}});
}
bool BethesdaSession::confirmTes3PlayerLevel(const std::vector<int>& attributes, std::string& error) {
    const auto* player = m_world.find(m_playerObject);
    if (!player || !player->actorValues || player->actorValues->dead) { error = "Level-up requires a living player"; return false; }
    const auto before = m_tes3.playerState();
    auto& stats = m_tes3.playerState().numericFilters;
    stats["health"] = player->actorValues->health;
    stats["maxhealth"] = player->actorValues->maxHealth;
    stats["magicka"] = player->actorValues->magicka;
    stats["maxmagicka"] = player->actorValues->maxMagicka;
    stats["fatigue"] = player->actorValues->stamina;
    stats["maxfatigue"] = player->actorValues->maxStamina;
    if (!m_tes3.confirmPlayerLevel(attributes, error)) { m_tes3.playerState() = before; return false; }
    synchronizeTes3PlayerValues(); return true;
}
} // namespace odai::bethesda
