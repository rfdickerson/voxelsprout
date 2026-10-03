#include "bethesda/bethesda_session.h"
#include "bethesda/session_text.h"
#include "core/hash.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>

namespace odai::bethesda {

namespace {

std::string asText(const Tes3Value& value) {
    if (value.type == Tes3ValueType::String) return value.string;
    if (value.type == Tes3ValueType::Object) return value.object.toString();
    return {};
}

std::int32_t asInt(const Tes3Value& value) {
    if (value.type == Tes3ValueType::Number) return static_cast<std::int32_t>(value.number);
    if (value.type == Tes3ValueType::String) {
        try { return static_cast<std::int32_t>(std::stoi(value.string)); }
        catch (...) { return 0; }
    }
    return 0;
}

double asNumber(const Tes3Value& value) {
    if (value.type == Tes3ValueType::Number) return value.number;
    if (value.type == Tes3ValueType::String) {
        try { return std::stod(value.string); }
        catch (...) { return 0.0; }
    }
    return 0.0;
}

std::string unquote(std::string value) {
    if (value.size() >= 2u && value.front() == '"' && value.back() == '"') {
        value = value.substr(1u, value.size() - 2u);
    }
    return value;
}

}  // namespace

void BethesdaSession::syncTes3PlayerInventory() {
    const RuntimeObject* player = m_world.find(m_playerObject);
    if (player == nullptr) return;

    std::map<RecordKey, std::int32_t> inventory;
    double clothingValue = 0;
    bool clothingAvailable = m_tes3.content() != nullptr;
    for (const InventoryEntry& entry : player->inventory) {
        if (entry.item.valid() && entry.count > 0) inventory[entry.item] += entry.count;
        if (!entry.equipped || entry.count <= 0 || !clothingAvailable) continue;
        const auto value = m_tes3.content()->wornItemValue(entry.item);
        if (!value.has_value()) clothingAvailable = false;
        else clothingValue += *value;
    }
    auto& filters = m_tes3.playerState().numericFilters;
    if (clothingAvailable) filters["clothingmodifier"] = clothingValue;
    else filters.erase("clothingmodifier");
    m_tes3.playerState().inventory = inventory;
    if (m_tes3.dialogue().active) {
        m_tes3.dialogueForRestore().player.inventory = std::move(inventory);
        if (clothingAvailable) m_tes3.dialogueForRestore().player.numericFilters["clothingmodifier"] = clothingValue;
        else m_tes3.dialogueForRestore().player.numericFilters.erase("clothingmodifier");
    }
}

Tes3DialogueResponse BethesdaSession::startTes3Dialogue(
    Tes3DialogueActorState actor, Tes3DialoguePlayerState player, bool strict) {
    const auto* definition = m_tes3.content() != nullptr
        ? m_tes3.content()->findActor("NPC_", actor.id) : nullptr;
    if (definition == nullptr && m_tes3.content() != nullptr)
        definition = m_tes3.content()->findActor("CREA", actor.id);
    if (definition != nullptr) {
        actor.race = definition->race;
        actor.actorClass = definition->actorClass;
        actor.faction = definition->faction.textId;
        actor.rank = static_cast<std::int8_t>(definition->rank);
        actor.gender = definition->gender;
        actor.disposition = definition->disposition;
    }
    if (const auto* object = m_world.find(actor.object); object != nullptr && object->actorValues) {
        actor.locals["actor:health"] = object->actorValues->health;
        actor.locals["actor:maxhealth"] = object->actorValues->maxHealth;
    }
    syncTes3PlayerInventory();
    player.inventory = m_tes3.playerState().inventory;
    const auto clothing = m_tes3.playerState().numericFilters.find("clothingmodifier");
    if (clothing != m_tes3.playerState().numericFilters.end()) player.numericFilters["clothingmodifier"] = clothing->second;
    else player.numericFilters.erase("clothingmodifier");
    player.deathCounts = m_tes3.playerState().deathCounts;
    const auto* worldPlayer = m_world.find(m_playerObject);
    if (worldPlayer != nullptr && worldPlayer->actorValues.has_value()) {
        const auto& values = *worldPlayer->actorValues;
        player.numericFilters["health"] = values.health;
        player.numericFilters["maxhealth"] = values.maxHealth;
        player.numericFilters["magicka"] = values.magicka;
        player.numericFilters["fatigue"] = values.stamina;
    }
    return m_tes3.startDialogue(std::move(actor), std::move(player), strict);
}

bool BethesdaSession::activateTes3Reference(
    ObjectId reference, bool fromScript, std::string& error) {
    const RuntimeObject* object = m_world.find(reference);
    const auto override = m_tes3.referenceOverrides().find(reference);
    const bool pendingDisabled = override != m_tes3.referenceOverrides().end() &&
        (override->second.deleted || override->second.enabled == false);
    if (m_tes3.content() == nullptr || object == nullptr || !object->enabled || pendingDisabled) {
        error = "TES3 activation requires an enabled resident reference";
        return false;
    }
    if (!fromScript) {
        bool scripted = false;
        for (const auto& [threadId, thread] : m_tes3.scripts().threads()) {
            (void)threadId;
            if (thread.owner == reference && thread.local &&
                thread.state != Tes3ThreadState::Failed) {
                scripted = true;
                break;
            }
        }
        if (scripted) {
            m_tes3.dispatchGameplayEvent("onactivate", reference);
            error.clear();
            return true;
        }
    }
    if (object->kind == RuntimeObjectKind::Item) {
        if (!object->base.valid() || m_world.find(m_playerObject) == nullptr) {
            error = "TES3 pickup requires an item and player";
            return false;
        }
        WorldCommand add;
        add.type = WorldCommandType::AddItem;
        add.target = m_playerObject;
        add.item = object->base;
        add.itemCount = 1;
        add.itemCondition = object->itemCondition;
        (void)m_world.queue(std::move(add));
        WorldCommand hide;
        hide.type = WorldCommandType::SetEnabled;
        hide.target = reference;
        hide.enabled = false;
        (void)m_world.queue(std::move(hide));
        m_tes3.referenceOverridesForRestore()[reference].enabled = false;
        ++m_tes3.playerState().inventory[object->base];
        if (m_tes3.dialogue().active)
            m_tes3.dialogueForRestore().player.inventory = m_tes3.playerState().inventory;
        m_tes3.dispatchGameplayEvent("onpcadd", reference);
    } else if (object->kind == RuntimeObjectKind::Activator) {
        RuntimeActivatorState state = object->activatorState.value_or(RuntimeActivatorState{});
        ++state.activationCount;
        WorldCommand activate;
        activate.type = WorldCommandType::SetActivatorState;
        activate.target = reference;
        activate.activatorState = std::move(state);
        (void)m_world.queue(std::move(activate));
    } else if (object->kind == RuntimeObjectKind::Door) {
        m_tes3DoorActivations.push_back(reference);
    }
    error.clear();
    return true;
}

Tes3NativeResult BethesdaSession::executeTes3WorldNative(const Tes3NativeCall& call) {
    Tes3NativeResult result;
    const auto resolveObject = [&](std::string authored) -> ObjectId {
        authored = unquote(std::move(authored));
        if (authored.empty()) return call.owner;
        if (normalizedEditorId(authored) == "player") return m_playerObject;
        RecordKey serialized;
        if (parseRecordKey(authored, serialized)) return ObjectId::persistent(std::move(serialized));
        const std::string wanted = makeTes3RecordKey("REFR", authored).textId;
        for (const ObjectId& residentId : m_world.orderedObjectIds()) {
            const RuntimeObject* object = m_world.find(residentId);
            if (object != nullptr &&
                ((object->id.kind == ObjectIdKind::PersistentReference &&
                  object->id.reference.textId == wanted) || object->base.textId == wanted)) {
                return object->id;
            }
        }
        if (m_tes3.content() != nullptr) {
            for (const auto& [id, reference] : m_tes3.content()->references()) {
                if (reference.base.textId == wanted ||
                    makeTes3RecordKey("REFR", reference.baseId).textId == wanted) {
                    return id;
                }
            }
        }
        return {};
    };
    const auto resolveBase = [&](const Tes3Value& value) -> RecordKey {
        if (value.type == Tes3ValueType::Object &&
            value.object.kind == ObjectIdKind::PersistentReference) return value.object.reference;
        const std::string wanted = makeTes3RecordKey("REFR", asText(value)).textId;
        if (m_tes3.content() != nullptr) {
            for (const auto& [key, record] : m_tes3.content()->namedRecords()) {
                (void)record;
                if (key.textId == wanted) return key;
            }
        }
        return {};
    };
    const std::string command = normalizedEditorId(call.command);
    const ObjectId target = resolveObject(call.target);
    const RuntimeObject* object = target.valid() ? m_world.find(target) : nullptr;
    if (command == "dontsaveobject") {
        RuntimeObject* mutableObject = target.valid() ? m_world.find(target) : nullptr;
        if (mutableObject == nullptr) {
            result.error = "DontSaveObject requires a resident scripted object";
            return result;
        }
        mutableObject->saveObject = false;
        return result;
    }
    const auto actorDefinitionForTarget = [&]() -> const Tes3ActorDefinition* {
        if (m_tes3.content() == nullptr) return nullptr;
        RecordKey base;
        if (object != nullptr) base = object->base;
        else if (target == m_playerObject) base = makeTes3RecordKey("NPC_", "player");
        else {
            const auto reference = m_tes3.content()->references().find(target);
            if (reference != m_tes3.content()->references().end()) base = reference->second.base;
        }
        const auto actor = m_tes3.content()->actors().find(base);
        return actor == m_tes3.content()->actors().end() ? nullptr : &actor->second;
    };
    const auto handleActorAndMagic = [&]() -> std::optional<Tes3NativeResult> {
    if (command == "getlevel" || command == "getrace" ||
        command == "gethealthgetratio") {
        const Tes3ActorDefinition* actor = actorDefinitionForTarget();
        if (actor == nullptr && target != m_playerObject) {
            result.error = command + " requires an authored actor"; return result;
        }
        if (command == "getlevel") result.value = Tes3Value::fromNumber(
            target == m_playerObject ? m_tes3.playerState().numericFilters["level"] : actor->level);
        else if (command == "getrace") {
            if (call.arguments.empty()) { result.error = "GetRace requires a race id"; return result; }
            const std::string race = target == m_playerObject &&
                !m_tes3.playerState().race.empty() ? m_tes3.playerState().race :
                actor != nullptr ? actor->race : std::string{};
            result.value = Tes3Value::fromNumber(normalizedEditorId(race) ==
                normalizedEditorId(asText(call.arguments[0])) ? 1.0 : 0.0);
        } else {
            double health = object != nullptr && object->actorValues.has_value()
                ? object->actorValues->health : actor->health;
            double maximum = object != nullptr && object->actorValues.has_value()
                ? object->actorValues->maxHealth : actor->health;
            const auto override = m_tes3.referenceOverrides().find(target);
            if (override != m_tes3.referenceOverrides().end()) {
                const auto savedHealth = override->second.locals.find("actor:health");
                const auto savedMaximum = override->second.locals.find("actor:maxhealth");
                if (savedHealth != override->second.locals.end()) health = savedHealth->second.number;
                if (savedMaximum != override->second.locals.end()) maximum = savedMaximum->second.number;
            }
            result.value = Tes3Value::fromNumber(maximum > 0.0 ? health / maximum : 0.0);
        }
        return result;
    }
    if (command == "getreputation" || command == "setreputation" || command == "modreputation") {
        if (!target.valid()) { result.error = command + " target does not resolve"; return result; }
        const auto* actor = actorDefinitionForTarget();
        if (target != m_playerObject && actor == nullptr) {
            result.error = command + " requires an actor"; return result;
        }
        double value = target == m_playerObject ? m_tes3.playerState().numericFilters["reputation"] : actor->reputation;
        if (target != m_playerObject) {
            const auto& locals = m_tes3.referenceOverridesForRestore()[target].locals;
            const auto found = locals.find("stat:reputation");
            if (found != locals.end()) value = found->second.number;
        }
        if (command == "getreputation") result.value = Tes3Value::fromNumber(value);
        else {
            if (call.arguments.empty()) { result.error = command + " requires a value"; return result; }
            value = command == "setreputation" ? asNumber(call.arguments[0]) : value + asNumber(call.arguments[0]);
            if (target == m_playerObject) {
                m_tes3.playerState().numericFilters["reputation"] = value;
                m_tes3.dialogueForRestore().player.numericFilters["reputation"] = value;
            } else m_tes3.referenceOverridesForRestore()[target].locals["stat:reputation"] = Tes3Value::fromNumber(value);
        }
        return result;
    }
    if (command == "getdisposition" || command == "setdisposition" || command == "moddisposition") {
        const auto* actor = actorDefinitionForTarget();
        if (!target.valid() || actor == nullptr || actor->creature) {
            result.error = command + " requires an authored NPC target";
            return result;
        }
        double value = actor->disposition;
        if (const auto it = m_tes3.referenceOverrides().find(target); it != m_tes3.referenceOverrides().end())
            if (const auto saved = it->second.locals.find("stat:disposition"); saved != it->second.locals.end()) value = saved->second.number;
        if (command == "getdisposition") result.value = Tes3Value::fromNumber(tes3DerivedDisposition(target));
        else {
            if (call.arguments.empty()) { result.error = command + " requires a value"; return result; }
            value = std::clamp(command == "setdisposition" ? asNumber(call.arguments[0]) :
                value + asNumber(call.arguments[0]), 0.0, 100.0);
            m_tes3.referenceOverridesForRestore()[target].locals["stat:disposition"] = Tes3Value::fromNumber(value);
            if (m_tes3.dialogue().active && m_tes3.dialogue().actor.object == target)
                m_tes3.dialogueForRestore().actor.disposition = static_cast<float>(value);
        }
        return result;
    }
    if (command == "forcegreeting") {
        const Tes3ActorDefinition* actor = actorDefinitionForTarget();
        if (actor == nullptr || !target.valid()) {
            result.error = "ForceGreeting requires an authored actor target";
            return result;
        }
        Tes3DialogueActorState dialogueActor;
        dialogueActor.object = target;
        dialogueActor.id = actor->id;
        dialogueActor.race = actor->race;
        dialogueActor.actorClass = actor->actorClass;
        dialogueActor.faction = actor->faction.textId;
        dialogueActor.rank = static_cast<std::int8_t>(actor->rank);
        dialogueActor.cell = object != nullptr ? object->currentSpace.cell.textId : std::string{};
        const auto state = m_tes3.referenceOverrides().find(target);
        if (state != m_tes3.referenceOverrides().end()) {
            const auto disposition = state->second.locals.find("stat:disposition");
            if (disposition != state->second.locals.end()) {
                dialogueActor.disposition = static_cast<float>(disposition->second.number);
            }
        }
        m_tes3.playerState().object = m_playerObject;
        (void)startTes3Dialogue(dialogueActor, m_tes3.playerState());
        return result;
    }
    constexpr std::array<std::string_view, 35u> tes3ActorStats = {
        "strength", "intelligence", "willpower", "agility", "speed", "endurance",
        "personality", "luck", "block", "armorer", "mediumarmor", "heavyarmor",
        "bluntweapon", "longblade", "axe", "spear", "athletics", "enchant",
        "destruction", "alteration", "illusion", "conjuration", "mysticism",
        "restoration", "alchemy", "unarmored", "security", "sneak", "acrobatics",
        "lightarmor", "shortblade", "marksman", "mercantile", "speechcraft",
        "handtohand"};
    constexpr std::array<std::string_view, 8u> tes3Attributes = {
        "strength", "intelligence", "willpower", "agility",
        "speed", "endurance", "personality", "luck"};
    constexpr std::array<std::string_view, 27u> tes3Skills = {
        "block", "armorer", "mediumarmor", "heavyarmor", "bluntweapon",
        "longblade", "axe", "spear", "athletics", "enchant", "destruction",
        "alteration", "illusion", "conjuration", "mysticism", "restoration",
        "alchemy", "unarmored", "security", "sneak", "acrobatics",
        "lightarmor", "shortblade", "marksman", "mercantile", "speechcraft",
        "handtohand"};
    const auto activeFortify = [&](const ObjectId& actor, std::string_view stat) {
        double magnitude = 0.0;
        const auto active = m_tes3.activeSpells().find(actor);
        if (active == m_tes3.activeSpells().end()) return magnitude;
        for (const Tes3ActiveSpell& spell : active->second) {
            for (const Tes3ActiveSpellEffect& effect : spell.effects) {
                if (effect.expiresTick <= call.tick) continue;
                if (effect.effectId == 79 && effect.attribute >= 0 &&
                    static_cast<std::size_t>(effect.attribute) < tes3Attributes.size() &&
                    tes3Attributes[static_cast<std::size_t>(effect.attribute)] == stat) {
                    magnitude += effect.magnitude;
                } else if (effect.effectId == 17 && effect.attribute >= 0 &&
                    static_cast<std::size_t>(effect.attribute) < tes3Attributes.size() &&
                    tes3Attributes[static_cast<std::size_t>(effect.attribute)] == stat) {
                    magnitude -= effect.magnitude;
                } else if (effect.effectId == 83 && effect.skill >= 0 &&
                    static_cast<std::size_t>(effect.skill) < tes3Skills.size() &&
                    tes3Skills[static_cast<std::size_t>(effect.skill)] == stat) {
                    magnitude += effect.magnitude;
                }
            }
        }
        return magnitude;
    };
    if (command == "getblightdisease" || command == "getcommondisease") {
        if (!target.valid()) { result.error = command + " target does not resolve"; return result; }
        const auto active = m_tes3.activeSpells().find(target);
        const std::int32_t diseaseType = command == "getblightdisease" ? 2 : 3;
        bool diseased = false;
        if (active != m_tes3.activeSpells().end() && m_tes3.content() != nullptr) {
            for (const Tes3ActiveSpell& item : active->second) {
                const auto spell = m_tes3.content()->spells().find(item.spell);
                if (spell != m_tes3.content()->spells().end() &&
                    spell->second.type == diseaseType &&
                    std::any_of(item.effects.begin(), item.effects.end(),
                        [&](const Tes3ActiveSpellEffect& effect) {
                            return effect.expiresTick > call.tick;
                        })) {
                    diseased = true;
                    break;
                }
            }
        }
        result.value = Tes3Value::fromNumber(diseased ? 1.0 : 0.0);
        return result;
    }
    if (command == "getspelleffects") {
        if (!target.valid() || call.arguments.empty()) {
            result.value = Tes3Value::fromNumber(0.0); return result;
        }
        const RecordKey spell = makeTes3RecordKey("SPEL", asText(call.arguments[0]));
        const auto active = m_tes3.activeSpells().find(target);
        const bool found = active != m_tes3.activeSpells().end() &&
            std::any_of(active->second.begin(), active->second.end(), [&](const Tes3ActiveSpell& item) {
                return item.spell == spell && std::any_of(item.effects.begin(), item.effects.end(),
                    [&](const Tes3ActiveSpellEffect& effect) { return effect.expiresTick > call.tick; });
            });
        result.value = Tes3Value::fromNumber(found ? 1.0 : 0.0);
        return result;
    }
    if (command == "cast") {
        if (m_tes3.content() == nullptr || call.arguments.empty()) {
            result.error = "Cast requires a spell"; return result;
        }
        const Tes3SpellDefinition* spell = m_tes3.content()->findSpell(asText(call.arguments[0]));
        const ObjectId recipient = call.arguments.size() >= 2u
            ? resolveObject(asText(call.arguments[1])) : target;
        if (spell == nullptr || !recipient.valid()) {
            result.error = spell == nullptr ? "Cast names an unresolved spell" :
                "Cast target does not resolve";
            return result;
        }
        Tes3ActiveSpell active;
        active.spell = spell->record;
        active.caster = call.owner;
        active.appliedTick = call.tick;
        for (const Tes3SpellEffect& authored : spell->effects) {
            if (authored.effectId != 69 && authored.effectId != 70 &&
                authored.effectId != 71 && authored.effectId != 74 &&
                authored.effectId != 79 &&
                authored.effectId != 83) {
                result.error = "Cast reaches unsupported gameplay magic effect " +
                    std::to_string(authored.effectId);
                return result;
            }
            if (authored.effectId == 74 &&
                (authored.attribute < 0 ||
                 static_cast<std::size_t>(authored.attribute) >= tes3Attributes.size())) {
                result.error = "RestoreAttribute has no authored attribute";
                return result;
            }
        }
        for (std::size_t index = 0u; index < spell->effects.size(); ++index) {
            const Tes3SpellEffect& authored = spell->effects[index];
            if (authored.effectId == 69 || authored.effectId == 70 ||
                authored.effectId == 71) {
                std::vector<std::pair<RecordKey, std::int32_t>> diseases;
                const auto collect = [&](const RecordKey& key, std::int32_t count) {
                    const auto found = m_tes3.content()->spells().find(key);
                    if (found == m_tes3.content()->spells().end() || count <= 0) return;
                    const Tes3SpellDefinition& owned = found->second;
                    const bool corprus = std::any_of(owned.effects.begin(), owned.effects.end(),
                        [](const Tes3SpellEffect& effect) { return effect.effectId == 132; });
                    if ((authored.effectId == 69 && owned.type == 3) ||
                        (authored.effectId == 70 && owned.type == 2 && !corprus) ||
                        (authored.effectId == 71 && corprus)) diseases.emplace_back(key, count);
                };
                if (recipient == m_playerObject) {
                    for (const auto& [key, count] : m_tes3.playerState().inventory) collect(key, count);
                } else if (const RuntimeObject* recipientObject = m_world.find(recipient)) {
                    for (const InventoryEntry& entry : recipientObject->inventory) {
                        collect(entry.item, entry.count);
                    }
                }
                for (const auto& [key, count] : diseases) {
                    WorldCommand remove;
                    remove.type = WorldCommandType::RemoveItem;
                    remove.target = recipient;
                    remove.item = key;
                    remove.itemCount = count;
                    (void)m_world.queue(std::move(remove));
                    if (recipient == m_playerObject) {
                        m_tes3.playerState().inventory.erase(key);
                        if (m_tes3.dialogue().active) {
                            m_tes3.dialogueForRestore().player.inventory.erase(key);
                        }
                    }
                    auto& activeSpells = m_tes3.activeSpellsForRestore()[recipient];
                    std::erase_if(activeSpells, [&](const Tes3ActiveSpell& item) {
                        return item.spell == key;
                    });
                }
                continue;
            }
            if (authored.effectId == 74) {
                if (authored.attribute < 0 ||
                    static_cast<std::size_t>(authored.attribute) >= tes3Attributes.size()) {
                    result.error = "RestoreAttribute has no authored attribute";
                    return result;
                }
                const std::string key = "damage:" +
                    std::string(tes3Attributes[static_cast<std::size_t>(authored.attribute)]);
                const double magnitude = std::max(0, authored.magnitudeMin);
                if (recipient == m_playerObject) {
                    double& damage = m_tes3.playerState().numericFilters[key];
                    damage = std::max(0.0, damage - magnitude);
                    if (m_tes3.dialogue().active)
                        m_tes3.dialogueForRestore().player.numericFilters[key] = damage;
                } else {
                    Tes3Value& damage = m_tes3.referenceOverridesForRestore()[recipient].locals[key];
                    damage = Tes3Value::fromNumber(std::max(0.0, damage.number - magnitude));
                }
                continue;
            }
            const std::int32_t minimum = std::min(authored.magnitudeMin, authored.magnitudeMax);
            const std::int32_t maximum = std::max(authored.magnitudeMin, authored.magnitudeMax);
            const std::uint64_t random = core::mix64(call.tick ^ ObjectIdHash{}(recipient) ^
                RecordKeyHash{}(spell->record) ^ static_cast<std::uint64_t>(index));
            Tes3ActiveSpellEffect effect;
            effect.effectId = authored.effectId;
            effect.skill = authored.skill;
            effect.attribute = authored.attribute;
            effect.magnitude = minimum + static_cast<double>(random %
                static_cast<std::uint64_t>(std::max(1, maximum - minimum + 1)));
            effect.expiresTick = call.tick + static_cast<std::uint64_t>(
                std::max(1, authored.duration)) * 60u;
            active.effects.push_back(effect);
        }
        if (!active.effects.empty()) {
            auto& spells = m_tes3.activeSpellsForRestore()[recipient];
            std::erase_if(spells, [&](const Tes3ActiveSpell& item) { return item.spell == spell->record; });
            spells.push_back(std::move(active));
        }
        return result;
    }
    const auto actorStatName = [&]() -> std::string {
        for (const std::string_view stat : tes3ActorStats) {
            if (command == std::string("get") + std::string(stat) ||
                command == std::string("set") + std::string(stat) ||
                command == std::string("mod") + std::string(stat)) return std::string(stat);
        }
        return {};
    };
    const std::string queriedActorStat = actorStatName();
    if (!queriedActorStat.empty()) {
        if (!target.valid()) { result.error = command + " target does not resolve"; return result; }
        const std::string storageKey = "stat:" + queriedActorStat;
        double value = 0.0;
        Tes3ReferenceOverride* override = nullptr;
        if (target != m_playerObject) {
            override = &m_tes3.referenceOverridesForRestore()[target];
        }
        const auto saved = override == nullptr ? std::map<std::string, Tes3Value>::const_iterator{} :
            override->locals.find(storageKey);
        if (override != nullptr && saved != override->locals.end()) value = saved->second.number;
        else {
            const RecordKey base = object != nullptr ? object->base :
                (target == m_playerObject ? makeTes3RecordKey("NPC_", "player") : RecordKey{});
            if (m_tes3.content() != nullptr && base.valid()) {
                const auto actor = m_tes3.content()->actors().find(base);
                if (actor != m_tes3.content()->actors().end()) {
                    const auto attribute = actor->second.attributes.find(queriedActorStat);
                    const auto skill = actor->second.skills.find(queriedActorStat);
                    if (attribute != actor->second.attributes.end()) value = attribute->second;
                    else if (skill != actor->second.skills.end()) value = skill->second;
                }
            }
            if (target == m_playerObject) {
                const auto playerValue = m_tes3.playerState().numericFilters.find(queriedActorStat);
                if (playerValue != m_tes3.playerState().numericFilters.end()) value = playerValue->second;
            }
        }
        if (command.starts_with("get")) {
            const std::string damageKey = "damage:" + queriedActorStat;
            double damage = 0.0;
            if (target == m_playerObject) {
                const auto found = m_tes3.playerState().numericFilters.find(damageKey);
                if (found != m_tes3.playerState().numericFilters.end()) damage = found->second;
            } else if (override != nullptr) {
                const auto found = override->locals.find(damageKey);
                if (found != override->locals.end()) damage = found->second.number;
            }
            result.value = Tes3Value::fromNumber(
                std::max(0.0, value + activeFortify(target, queriedActorStat) - damage));
            return result;
        }
        if (call.arguments.empty()) { result.error = command + " requires a value"; return result; }
        value = command.starts_with("set") ? asNumber(call.arguments[0])
                                             : value + asNumber(call.arguments[0]);
        if (override != nullptr) override->locals[storageKey] = Tes3Value::fromNumber(value);
        if (target == m_playerObject) {
            m_tes3.playerState().numericFilters[queriedActorStat] = value;
            if (m_tes3.dialogue().active) {
                m_tes3.dialogueForRestore().player.numericFilters[queriedActorStat] = value;
            }
        }
        return result;
    }
    if (command == "getmoving") {
        result.value = Tes3Value::fromNumber(object != nullptr && object->aiState.has_value() &&
            object->aiState->walking ? 1.0 : 0.0);
        return result;
    }
    if (command == "getwaterlevel" || command == "setwaterlevel" ||
        command == "modwaterlevel") {
        if (!target.valid()) { result.error = command + " target does not resolve"; return result; }
        double water = target == m_playerObject
            ? m_tes3.playerState().numericFilters["waterlevel"]
            : m_tes3.referenceOverridesForRestore()[target].locals["waterlevel"].number;
        if (command == "getwaterlevel") result.value = Tes3Value::fromNumber(water);
        else if (call.arguments.empty()) result.error = command + " requires a value";
        else {
            water = command == "setwaterlevel"
                ? asNumber(call.arguments[0]) : water + asNumber(call.arguments[0]);
            if (target == m_playerObject) m_tes3.playerState().numericFilters["waterlevel"] = water;
            else m_tes3.referenceOverridesForRestore()[target].locals["waterlevel"] =
                Tes3Value::fromNumber(water);
        }
        return result;
    }
        return std::nullopt;
    };
    if (auto handled = handleActorAndMagic()) return *handled;

    const auto handleMovementAndCombat = [&]() -> std::optional<Tes3NativeResult> {
    const auto currentTransform = [&]() -> std::optional<RuntimeTransform> {
        if (object != nullptr) return object->transform;
        const auto saved = m_tes3.referenceOverrides().find(target);
        if (saved != m_tes3.referenceOverrides().end() && saved->second.transform.has_value()) {
            return saved->second.transform;
        }
        if (m_tes3.content() != nullptr) {
            const auto definition = m_tes3.content()->references().find(target);
            if (definition != m_tes3.content()->references().end()) {
                RuntimeTransform transform;
                transform.position = {definition->second.position[0],
                    definition->second.position[1], definition->second.position[2]};
                transform.rotationRadians = {definition->second.rotationRadians[0],
                    definition->second.rotationRadians[1],
                    definition->second.rotationRadians[2]};
                transform.scale = definition->second.scale.value_or(1.0f);
                return transform;
            }
        }
        return std::nullopt;
    };
    if (command == "getstartingangle" || command == "setatstart") {
        if (!target.valid() || m_tes3.content() == nullptr) {
            result.error = command + " target does not resolve"; return result;
        }
        const auto definition = m_tes3.content()->references().find(target);
        if (definition == m_tes3.content()->references().end()) {
            result.error = command + " target has no authored placement"; return result;
        }
        RuntimeTransform initial;
        initial.position = {definition->second.position[0], definition->second.position[1],
            definition->second.position[2]};
        initial.rotationRadians = {definition->second.rotationRadians[0],
            definition->second.rotationRadians[1], definition->second.rotationRadians[2]};
        initial.scale = definition->second.scale.value_or(1.0f);
        if (command == "getstartingangle") {
            const std::string axis = call.arguments.empty() ? "z" :
                normalizedEditorId(asText(call.arguments[0]));
            const std::size_t index = axis == "x" ? 0u : axis == "y" ? 1u : 2u;
            result.value = Tes3Value::fromNumber(initial.rotationRadians[index] *
                (180.0 / 3.14159265358979323846));
        } else {
            m_tes3.referenceOverridesForRestore()[target].transform = initial;
            if (object != nullptr) {
                WorldCommand world;
                world.type = WorldCommandType::SetTransform;
                world.target = target;
                world.transform = initial;
                (void)m_world.queue(std::move(world));
            }
        }
        return result;
    }
    const auto storeTransform = [&](const RuntimeTransform& transform) {
        // The player is a session-owned object, not an authored CELL reference.
        // Its transform is already serialized with the world; an override here
        // cannot be resolved against the plugin when the save is loaded.
        if (target != m_playerObject) {
            m_tes3.referenceOverridesForRestore()[target].transform = transform;
        }
        if (object != nullptr) {
            WorldCommand world;
            world.type = WorldCommandType::SetTransform;
            world.target = target;
            world.transform = transform;
            (void)m_world.queue(std::move(world));
        }
    };
    if (command == "move" || command == "moveworld" || command == "rotate" ||
        command == "rotateworld" || command == "face") {
        if (!target.valid()) { result.error = command + " target does not resolve"; return result; }
        std::optional<RuntimeTransform> transform = currentTransform();
        if (!transform.has_value()) { result.error = command + " target has no transform"; return result; }
        if (command == "face") {
            if (call.arguments.size() < 2u) { result.error = "Face requires x and y"; return result; }
            const double dx = asNumber(call.arguments[0]) - transform->position[0];
            const double dy = asNumber(call.arguments[1]) - transform->position[1];
            transform->rotationRadians[2] = static_cast<float>(std::atan2(dx, dy));
        } else {
            if (call.arguments.size() < 2u) {
                result.error = command + " requires axis and rate"; return result;
            }
            const std::string axis = normalizedEditorId(asText(call.arguments[0]));
            const std::size_t component = axis == "x" ? 0u : axis == "y" ? 1u :
                axis == "z" ? 2u : 3u;
            if (component == 3u) { result.error = command + " has an invalid axis"; return result; }
            const double delta = asNumber(call.arguments[1]) / 60.0;
            if (command == "rotate" || command == "rotateworld") {
                transform->rotationRadians[component] += static_cast<float>(
                    delta * (3.14159265358979323846 / 180.0));
            } else {
                std::array<double, 3u> movement{};
                movement[component] = delta;
                if (command == "move") {
                    const double cx = std::cos(transform->rotationRadians[0]);
                    const double sx = std::sin(transform->rotationRadians[0]);
                    const double cy = std::cos(transform->rotationRadians[1]);
                    const double sy = std::sin(transform->rotationRadians[1]);
                    const double cz = std::cos(transform->rotationRadians[2]);
                    const double sz = std::sin(transform->rotationRadians[2]);
                    const double x = movement[0];
                    const double y = movement[1];
                    const double z = movement[2];
                    movement = {cz * cy * x + (cz * sy * sx - sz * cx) * y +
                            (cz * sy * cx + sz * sx) * z,
                        sz * cy * x + (sz * sy * sx + cz * cx) * y +
                            (sz * sy * cx - cz * sx) * z,
                        -sy * x + cy * sx * y + cy * cx * z};
                }
                for (std::size_t index = 0u; index < movement.size(); ++index) {
                    transform->position[index] += movement[index];
                }
            }
        }
        storeTransform(*transform);
        return result;
    }
    if (command == "getpos" || command == "getangle" || command == "getscale" ||
        command == "setpos" || command == "setangle" || command == "setscale" ||
        command == "modscale" || command == "position" || command == "positioncell") {
        if (!target.valid()) { result.error = command + " target does not resolve"; return result; }
        std::optional<RuntimeTransform> transform = currentTransform();
        if (!transform.has_value()) { result.error = command + " target has no transform"; return result; }
        const auto component = [&](const Tes3Value& value) -> std::size_t {
            const std::string axis = normalizedEditorId(asText(value));
            return axis == "y" ? 1u : axis == "z" ? 2u : 0u;
        };
        if (command == "getscale") {
            result.value = Tes3Value::fromNumber(transform->scale);
            return result;
        }
        if (command == "getpos" || command == "getangle") {
            const std::size_t axis = call.arguments.empty() ? 0u : component(call.arguments[0]);
            const double position = axis == 0u ? transform->position[0] :
                axis == 1u ? -transform->position[2] : transform->position[1];
            const double angle = axis == 0u ? transform->rotationRadians[0] :
                axis == 1u ? -transform->rotationRadians[2] :
                -transform->rotationRadians[1];
            result.value = Tes3Value::fromNumber(command == "getpos" ? position :
                angle * (180.0 / 3.14159265358979323846));
            return result;
        }
        if (command == "setpos" || command == "setangle") {
            if (call.arguments.size() < 2u) { result.error = command + " requires axis and value"; return result; }
            const std::size_t axis = component(call.arguments[0]);
            if (command == "setpos") {
                if (axis == 0u) transform->position[0] = asNumber(call.arguments[1]);
                else if (axis == 1u) transform->position[2] = -asNumber(call.arguments[1]);
                else transform->position[1] = asNumber(call.arguments[1]);
            } else {
                const float radians = static_cast<float>(
                    asNumber(call.arguments[1]) * (3.14159265358979323846 / 180.0));
                if (axis == 0u) transform->rotationRadians[0] = radians;
                else if (axis == 1u) transform->rotationRadians[2] = -radians;
                else transform->rotationRadians[1] = -radians;
            }
        } else if (command == "setscale" || command == "modscale") {
            if (call.arguments.empty()) { result.error = command + " requires a value"; return result; }
            const float value = static_cast<float>(asNumber(call.arguments[0]));
            transform->scale = command == "setscale" ? value : transform->scale + value;
        } else {
            if (call.arguments.size() < 4u) { result.error = command + " requires x y z rotation"; return result; }
            // MWScript uses Bethesda's Z-up space; RuntimeTransform is Y-up
            // with its horizontal Z axis negated.
            transform->position = {asNumber(call.arguments[0]), asNumber(call.arguments[2]),
                -asNumber(call.arguments[1])};
            transform->rotationRadians[1] = static_cast<float>(
                -asNumber(call.arguments[3]) * (3.14159265358979323846 / 180.0));
            if (command == "positioncell" && call.arguments.size() >= 5u && object != nullptr) {
                WorldCommand space;
                space.type = WorldCommandType::SetCurrentSpace;
                space.target = target;
                space.currentSpace.kind = RuntimeSpaceKind::Interior;
                space.currentSpace.cell = makeTes3RecordKey("CELL", asText(call.arguments[4]));
                (void)m_world.queue(std::move(space));
            }
        }
        storeTransform(*transform);
        return result;
    }
    if (command == "getfight" || command == "setfight" || command == "modfight") {
        if (!target.valid()) { result.error = command + " target does not resolve"; return result; }
        Tes3ReferenceOverride& state = m_tes3.referenceOverridesForRestore()[target];
        const auto* definition = actorDefinitionForTarget();
        const auto saved = state.locals.find("stat:fight");
        double fight = saved == state.locals.end() ? (definition ? definition->fight : 0) : saved->second.number;
        if (command == "getfight") result.value = Tes3Value::fromNumber(fight);
        else if (!call.arguments.empty()) {
            fight = command == "setfight" ? asNumber(call.arguments[0])
                                            : fight + asNumber(call.arguments[0]);
            state.locals["stat:fight"] = Tes3Value::fromNumber(std::clamp(fight, 0.0, 100.0));
        } else result.error = command + " requires a value";
        return result;
    }
    if (command == "raiserank" || command == "lowerrank") {
        if (!target.valid()) { result.error = command + " target does not resolve"; return result; }
        if (target == m_playerObject) {
            const std::string faction = normalizedEditorId(m_tes3.dialogue().actor.faction);
            std::int8_t& rank = m_tes3.playerState().factionRanks[faction];
            rank = static_cast<std::int8_t>(std::clamp<int>(rank +
                (command == "raiserank" ? 1 : -1), 0, 9));
            if (m_tes3.dialogue().active) m_tes3.dialogueForRestore().player = m_tes3.playerState();
            return result;
        }
        const Tes3ActorDefinition* actor = actorDefinitionForTarget();
        Tes3Value& rank = m_tes3.referenceOverridesForRestore()[target].locals["stat:rank"];
        if (rank.type == Tes3ValueType::None) {
            rank = Tes3Value::fromNumber(actor == nullptr ? -1.0 : actor->rank);
        }
        rank = Tes3Value::fromNumber(std::clamp(rank.number +
            (command == "raiserank" ? 1.0 : -1.0), -1.0, 9.0));
        if (m_tes3.dialogue().active && m_tes3.dialogue().actor.object == target) {
            m_tes3.dialogueForRestore().actor.rank = static_cast<std::int8_t>(rank.number);
        }
        return result;
    }
    const std::map<std::string, std::string> forcedMovementCommands = {
        {"forcerun", "run"}, {"clearforcerun", "run"}, {"getforcerun", "run"},
        {"forcejump", "jump"}, {"clearforcejump", "jump"},
        {"forcemovejump", "movejump"}, {"clearforcemovejump", "movejump"},
        {"getforcemovejump", "movejump"}, {"forcesneak", "sneak"},
        {"clearforcesneak", "sneak"}, {"getforcesneak", "sneak"}};
    if (const auto forcedCommand = forcedMovementCommands.find(command);
        forcedCommand != forcedMovementCommands.end()) {
        if (!target.valid()) { result.error = command + " target does not resolve"; return result; }
        const std::string key = "force:" + forcedCommand->second;
        Tes3Value* forced = nullptr;
        if (target == m_playerObject) {
            double& playerForced = m_tes3.playerState().numericFilters[key];
            if (command.starts_with("get")) {
                result.value = Tes3Value::fromNumber(playerForced != 0.0 ? 1.0 : 0.0);
            } else playerForced = command.starts_with("clear") ? 0.0 : 1.0;
            return result;
        }
        forced = &m_tes3.referenceOverridesForRestore()[target].locals[key];
        if (command.starts_with("get")) {
            result.value = Tes3Value::fromNumber(forced->truthy() ? 1.0 : 0.0);
        } else *forced = Tes3Value::fromNumber(command.starts_with("clear") ? 0.0 : 1.0);
        return result;
    }
    if (command == "getaipackagedone" || command == "getcurrentaipackage") {
        if (!target.valid()) { result.error = command + " target does not resolve"; return result; }
        const auto override = m_tes3.referenceOverrides().find(target);
        if (command == "getaipackagedone") {
            bool arrived = object != nullptr && object->aiState.has_value() &&
                object->aiState->scriptedMoveArrived;
            if (override != m_tes3.referenceOverrides().end()) {
                const auto done = override->second.locals.find("ai:done");
                if (done != override->second.locals.end()) arrived = done->second.truthy();
            }
            result.value = Tes3Value::fromNumber(arrived ? 1.0 : 0.0);
        } else {
            double package = -1.0;
            if (override != m_tes3.referenceOverrides().end()) {
                const auto found = override->second.locals.find("ai:package");
                if (found != override->second.locals.end()) package = found->second.number;
            }
            result.value = Tes3Value::fromNumber(package);
        }
        return result;
    }
    if (command == "aitravel" || command == "aiwander" || command == "aifollow" ||
        command == "aifollowcell" || command == "aiescort" || command == "aiescortcell") {
        if (object == nullptr) { result.error = command + " requires a resident actor"; return result; }
        RuntimeAiState ai = object->aiState.value_or(RuntimeAiState{});
        ai.walking = true;
        ai.scriptedMoveActive = true;
        ai.scriptedMoveArrived = false;
        ++ai.scriptedMoveRevision;
        ai.path.clear();
        ai.pathIndex = 0u;
        ai.pauseSeconds = 0.0f;
        const int package = command == "aiwander" ? 0 : command == "aitravel" ? 1 :
            command.starts_with("aiescort") ? 2 : 3;
        m_tes3.referenceOverridesForRestore()[target].locals["ai:package"] =
            Tes3Value::fromNumber(package);
        m_tes3.referenceOverridesForRestore()[target].locals["ai:done"] =
            Tes3Value::fromNumber(0.0);
        if (command == "aitravel" && call.arguments.size() >= 3u) {
            ai.wanderTarget = {static_cast<float>(asNumber(call.arguments[0])),
                static_cast<float>(asNumber(call.arguments[2])),
                -static_cast<float>(asNumber(call.arguments[1]))};
        } else if (command == "aiwander") {
            ai.wanderOrigin = {static_cast<float>(object->transform.position[0]),
                static_cast<float>(object->transform.position[1]),
                static_cast<float>(object->transform.position[2])};
            ai.wanderTarget = ai.wanderOrigin;
            ai.scriptedMoveActive = false;
            ai.walking = false;
        } else if (command == "aiescort" && call.arguments.size() >= 5u) {
            // Escort leads the player to the authored destination; it does
            // not chase the player like AIFollow.
            ai.wanderTarget = {static_cast<float>(asNumber(call.arguments[2])),
                static_cast<float>(asNumber(call.arguments[4])),
                -static_cast<float>(asNumber(call.arguments[3]))};
        } else if (!call.arguments.empty()) {
            const ObjectId destination = resolveObject(asText(call.arguments[0]));
            if (!destination.valid()) { result.error = command + " destination does not resolve"; return result; }
            WorldCommand move;
            move.type = WorldCommandType::RequestMoveTo;
            move.target = target;
            move.destination = destination;
            move.navigationRevision = ai.scriptedMoveRevision;
            (void)m_world.queue(std::move(move));
        }
        WorldCommand world;
        world.type = WorldCommandType::SetAiState;
        world.target = target;
        world.aiState = std::move(ai);
        (void)m_world.queue(std::move(world));
        return result;
    }
    if (command == "startcombat" || command == "stopcombat") {
        if (object == nullptr) { result.error = command + " requires a resident actor"; return result; }
        RuntimeCombatState combat = object->combatState.value_or(RuntimeCombatState{});
        if (command == "startcombat") {
            if (call.arguments.empty()) { result.error = "StartCombat requires a target"; return result; }
            combat.combatTarget = resolveObject(asText(call.arguments[0]));
            combat.lastTarget = combat.combatTarget;
            if (!combat.combatTarget.valid()) { result.error = "StartCombat target does not resolve"; return result; }
        } else combat.combatTarget = {};
        WorldCommand world;
        world.type = WorldCommandType::SetCombatState;
        world.target = target;
        world.combatState = combat;
        (void)m_world.queue(std::move(world));
        return result;
    }
    if (command == "gettarget") {
        if (object == nullptr || !object->combatState.has_value() || call.arguments.empty()) {
            result.value = Tes3Value::fromNumber(0.0);
            return result;
        }
        result.value = Tes3Value::fromNumber(object->combatState->combatTarget ==
            resolveObject(asText(call.arguments[0])) ? 1.0 : 0.0);
        return result;
    }
    if (command == "getdistance") {
        if (!target.valid() || call.arguments.empty()) {
            result.error = "GetDistance requires source and destination"; return result;
        }
        const ObjectId destination = resolveObject(asText(call.arguments[0]));
        const RuntimeObject* other = m_world.find(destination);
        const std::optional<RuntimeTransform> sourceTransform = currentTransform();
        if (!sourceTransform.has_value() || other == nullptr) {
            result.error = "GetDistance requires materialized transforms"; return result;
        }
        const double dx = sourceTransform->position[0] - other->transform.position[0];
        const double dy = sourceTransform->position[1] - other->transform.position[1];
        const double dz = sourceTransform->position[2] - other->transform.position[2];
        result.value = Tes3Value::fromNumber(std::sqrt(dx * dx + dy * dy + dz * dz));
        return result;
    }
    if (command == "getdetected") {
        if (object == nullptr || call.arguments.empty()) {
            result.error = "GetDetected requires a resident observer and actor id";
            return result;
        }
        const ObjectId observedId = resolveObject(asText(call.arguments[0]));
        const RuntimeObject* observed = m_world.find(observedId);
        bool detected = false;
        if (observed != nullptr && observed->enabled &&
            (!observed->actorValues.has_value() || !observed->actorValues->dead) &&
            (!object->currentSpace.cell.valid() || !observed->currentSpace.cell.valid() ||
             object->currentSpace.cell == observed->currentSpace.cell)) {
            const odai::math::Vector3 from{
                static_cast<float>(object->transform.position[0]),
                static_cast<float>(object->transform.position[1] + 64.0),
                static_cast<float>(object->transform.position[2])};
            const odai::math::Vector3 to{
                static_cast<float>(observed->transform.position[0]),
                static_cast<float>(observed->transform.position[1] + 64.0),
                static_cast<float>(observed->transform.position[2])};
            const float distance = odai::math::length(to - from);
            if (distance < 2048.0f) {
                const auto obstruction = m_physics.castRay(from, to);
                detected = !obstruction.has_value() ||
                    obstruction->distance >= distance - 1.0f;
            }
        }
        result.value = Tes3Value::fromNumber(detected ? 1.0 : 0.0);
        return result;
    }
    if (command == "getstandingpc") {
        if (!target.valid()) { result.error = "GetStandingPC target does not resolve"; return result; }
        const std::vector<PhysicsCharacterSnapshot> characters = physicsSnapshots();
        const auto player = std::find_if(characters.begin(), characters.end(),
            [&](const PhysicsCharacterSnapshot& character) {
                return character.object == m_playerObject;
            });
        result.value = Tes3Value::fromNumber(player != characters.end() &&
            player->grounded && player->supportingObject == target ? 1.0 : 0.0);
        return result;
    }
        return std::nullopt;
    };
    if (auto handled = handleMovementAndCombat()) return *handled;

    const auto handleReferenceState = [&]() -> std::optional<Tes3NativeResult> {
    if (command == "getpccell") {
        const RuntimeObject* player = m_world.find(m_playerObject);
        const std::string cell = call.arguments.empty() ? std::string{} :
            makeTes3RecordKey("CELL", asText(call.arguments[0])).textId;
        result.value = Tes3Value::fromNumber(player != nullptr &&
            player->currentSpace.cell.textId == cell ? 1.0 : 0.0);
        return result;
    }
    if (command == "getinterior") {
        const RuntimeObject* queried = object != nullptr ? object : m_world.find(m_playerObject);
        result.value = Tes3Value::fromNumber(queried != nullptr &&
            queried->currentSpace.kind == RuntimeSpaceKind::Interior ? 1.0 : 0.0);
        return result;
    }
    if (command == "getfatigue" || command == "getmagicka" ||
        command == "setfatigue" || command == "setmagicka" ||
        command == "modfatigue" || command == "modmagicka" ||
        command == "modcurrentfatigue" || command == "modcurrentmagicka") {
        if (object == nullptr || !object->actorValues.has_value()) {
            result.error = command + " requires a resident actor"; return result;
        }
        const bool fatigue = command.find("fatigue") != std::string::npos;
        if (command.starts_with("get")) {
            result.value = Tes3Value::fromNumber(
                fatigue ? object->actorValues->stamina : object->actorValues->magicka);
            return result;
        }
        if (call.arguments.empty()) { result.error = command + " requires a value"; return result; }
        const bool set = command.starts_with("set");
        const bool current = command.starts_with("modcurrent");
        WorldCommand world;
        world.type = set ? WorldCommandType::SetActorBaseValue :
            current ? WorldCommandType::AdjustActorValue : WorldCommandType::AdjustActorBaseValue;
        world.target = target;
        world.actorValue = fatigue ? ActorValue::Stamina : ActorValue::Magicka;
        if (set) world.actorValueAbsolute = static_cast<float>(asNumber(call.arguments[0]));
        else world.actorValueDelta = static_cast<float>(asNumber(call.arguments[0]));
        (void)m_world.queue(std::move(world));
        return result;
    }
    if (command == "getalarm" || command == "setalarm" || command == "modalarm" ||
        command == "getflee" || command == "setflee" || command == "modflee" ||
        command == "gethello" || command == "sethello" || command == "modhello") {
        if (!target.valid()) { result.error = command + " target does not resolve"; return result; }
        const std::string stat = command.find("alarm") != std::string::npos ? "alarm" :
            command.find("flee") != std::string::npos ? "flee" : "hello";
        auto& locals = m_tes3.referenceOverridesForRestore()[target].locals;
        const auto* definition = actorDefinitionForTarget();
        if (!locals.contains("stat:" + stat) && stat == "flee" && definition)
            locals["stat:flee"] = Tes3Value::fromNumber(definition->flee);
        Tes3Value& stored = locals["stat:" + stat];
        if (command.starts_with("get")) result.value = Tes3Value::fromNumber(stored.number);
        else if (!call.arguments.empty()) {
            stored = Tes3Value::fromNumber(command.starts_with("set")
                ? asNumber(call.arguments[0]) : stored.number + asNumber(call.arguments[0]));
        } else result.error = command + " requires a value";
        return result;
    }
    if (command == "lock" || command == "unlock" || command == "getlocked") {
        if (!target.valid()) { result.error = command + " target does not resolve"; return result; }
        Tes3Value& lock = m_tes3.referenceOverridesForRestore()[target].locals["locklevel"];
        if (command == "getlocked") result.value = Tes3Value::fromNumber(lock.number > 0.0 ? 1.0 : 0.0);
        else lock = Tes3Value::fromNumber(command == "unlock" ? 0.0 :
            (call.arguments.empty() ? 100.0 : asNumber(call.arguments[0])));
        return result;
    }
    if (command == "activate") {
        if (!activateTes3Reference(target, true, result.error)) return result;
        return result;
    }
    if (command == "getdisabled") {
        bool enabled = object != nullptr && object->enabled;
        if (object == nullptr && target.valid() && m_tes3.content() != nullptr) {
            const auto definition = m_tes3.content()->references().find(target);
            enabled = definition != m_tes3.content()->references().end() &&
                definition->second.enabled && !definition->second.deleted;
        }
        const auto override = m_tes3.referenceOverrides().find(target);
        if (override != m_tes3.referenceOverrides().end()) {
            if (override->second.enabled.has_value()) enabled = *override->second.enabled;
            if (override->second.deleted) enabled = false;
        }
        result.value = Tes3Value::fromNumber(enabled ? 0.0 : 1.0);
        return result;
    }
    if (command == "gethealth") {
        if (!target.valid()) { result.error = "GetHealth target does not resolve"; return result; }
        const auto override = m_tes3.referenceOverrides().find(target);
        if (override != m_tes3.referenceOverrides().end()) {
            const auto saved = override->second.locals.find("actor:health");
            if (saved != override->second.locals.end()) {
                result.value = saved->second;
                return result;
            }
        }
        if (object != nullptr && object->actorValues.has_value()) {
            result.value = Tes3Value::fromNumber(object->actorValues->health);
        } else if (const Tes3ActorDefinition* actor = actorDefinitionForTarget()) {
            result.value = Tes3Value::fromNumber(actor->health);
        } else result.error = "GetHealth requires an authored actor";
        return result;
    }
    if (command == "enable" || command == "disable" || command == "delete" ||
        command == "setdelete") {
        if (!target.valid()) {
            result.error = command + " target does not resolve: " + call.target;
            return result;
        }
        Tes3ReferenceOverride& override = m_tes3.referenceOverridesForRestore()[target];
        const bool deleting = command == "delete" || (command == "setdelete" &&
            (call.arguments.empty() || asNumber(call.arguments[0]) != 0.0));
        if (command == "setdelete") override.deleted = deleting;
        else if (deleting) override.deleted = true;
        else override.enabled = command == "enable";
        if (command == "setdelete" && !deleting) return result;
        if (object == nullptr) return result;
        WorldCommand world;
        world.type = deleting ? WorldCommandType::Destroy : WorldCommandType::SetEnabled;
        world.target = target;
        world.enabled = command == "enable" || (command == "setdelete" && !deleting);
        (void)m_world.queue(std::move(world));
        return result;
    }
        return std::nullopt;
    };
    if (auto handled = handleReferenceState()) return *handled;

    const auto handleInventoryAndSpawn = [&]() -> std::optional<Tes3NativeResult> {
    if (command == "additem" || command == "removeitem" || command == "getitemcount" ||
        command == "addspell" || command == "removespell" || command == "getspell") {
        if (object == nullptr || call.arguments.empty()) {
            result.error = command + " requires a resident target and item";
            return result;
        }
        const RecordKey item = resolveBase(call.arguments[0]);
        const std::int32_t count = call.arguments.size() >= 2u ? std::max(1, asInt(call.arguments[1])) : 1;
        if (!item.valid()) {
            result.error = command + " names an unresolved item";
            return result;
        }
        const bool spellCommand = command == "addspell" || command == "removespell" ||
            command == "getspell";
        const Tes3SpellDefinition* spell = nullptr;
        if (spellCommand) {
            const auto found = m_tes3.content()->spells().find(item);
            if (found == m_tes3.content()->spells().end()) {
                result.error = command + " requires a spell record";
                return result;
            }
            spell = &found->second;
        }
        if (command == "getitemcount" || command == "getspell") {
            if (target == m_playerObject) {
                const auto found = m_tes3.playerState().inventory.find(item);
                result.value = Tes3Value::fromNumber(found == m_tes3.playerState().inventory.end()
                    ? 0.0 : static_cast<double>(found->second));
                return result;
            }
            const std::string pendingKey = "inventory:" + item.toString();
            const auto override = m_tes3.referenceOverrides().find(target);
            if (override != m_tes3.referenceOverrides().end()) {
                const auto pending = override->second.locals.find(pendingKey);
                if (pending != override->second.locals.end()) {
                    result.value = Tes3Value::fromNumber(pending->second.number);
                    return result;
                }
            }
            double count = 0;
            for (const auto& entry : object->inventory) if (entry.item == item) count += entry.count;
            result.value = Tes3Value::fromNumber(count);
            return result;
        }
        if (command == "addspell" && spell->type >= 1 && spell->type <= 4) {
            for (const Tes3SpellEffect& effect : spell->effects) {
                if (effect.effectId != 3 && effect.effectId != 4 &&
                    effect.effectId != 5 && effect.effectId != 6 &&
                    effect.effectId != 17 && effect.effectId != 79 &&
                    effect.effectId != 83 && effect.effectId != 94 &&
                    effect.effectId != 95 && effect.effectId != 96 &&
                    effect.effectId != 132) {
                    result.error = "AddSpell reaches unsupported passive magic effect " +
                        std::to_string(effect.effectId);
                    return result;
                }
            }
        }
        const std::int32_t owned = target == m_playerObject
            ? (m_tes3.playerState().inventory.contains(item)
                ? m_tes3.playerState().inventory.at(item) : 0)
            : [&]() {
                const std::string pendingKey = "inventory:" + item.toString();
                const auto override = m_tes3.referenceOverrides().find(target);
                if (override != m_tes3.referenceOverrides().end()) {
                    const auto pending = override->second.locals.find(pendingKey);
                    if (pending != override->second.locals.end())
                        return static_cast<std::int32_t>(pending->second.number);
                }
                std::int64_t total = 0;
                for (const auto& entry : object->inventory) if (entry.item == item) total += entry.count;
                return static_cast<std::int32_t>(std::min<std::int64_t>(total, std::numeric_limits<std::int32_t>::max()));
            }();
        if (spellCommand && ((command == "addspell" && owned > 0) ||
            (command == "removespell" && owned == 0))) return result;
        if ((command == "additem" || command == "addspell") &&
            owned > std::numeric_limits<std::int32_t>::max() - (spellCommand ? 1 : count)) {
            result.error = "item addition would overflow inventory count"; return result;
        }
        WorldCommand world;
        world.type = command == "additem" || command == "addspell"
            ? WorldCommandType::AddItem : WorldCommandType::RemoveItem;
        world.target = target;
        world.item = item;
        world.itemCount = spellCommand ? 1 : count;
        (void)m_world.queue(std::move(world));
        if (target == m_playerObject &&
            (command == "additem" || command == "removeitem" || spellCommand)) {
            auto& inventory = m_tes3.playerState().inventory;
            const std::int32_t previous = inventory.contains(item) ? inventory.at(item) : 0;
            const std::int32_t next = command == "additem" || command == "addspell"
                ? previous + (spellCommand ? 1 : count)
                : std::max(0, previous - (spellCommand ? 1 : count));
            if (next > 0) inventory[item] = next;
            else inventory.erase(item);
            if (m_tes3.dialogue().active) {
                m_tes3.dialogueForRestore().player.inventory = inventory;
            }
        }
        if (target != m_playerObject) {
            const bool adding = command == "additem" || command == "addspell";
            const std::int32_t next = adding ? owned + (spellCommand ? 1 : count) :
                std::max(0, owned - (spellCommand ? 1 : count));
            m_tes3.referenceOverridesForRestore()[target].locals[
                "inventory:" + item.toString()] = Tes3Value::fromNumber(next);
        }
        if (spellCommand) {
            auto& active = m_tes3.activeSpellsForRestore()[target];
            if (command == "removespell") {
                std::erase_if(active, [&](const Tes3ActiveSpell& effect) {
                    return effect.spell == item;
                });
            } else if (spell->type >= 1 && spell->type <= 4) {
                Tes3ActiveSpell passive;
                passive.spell = item;
                passive.caster = target;
                passive.appliedTick = call.tick;
                for (const Tes3SpellEffect& authored : spell->effects) {
                    Tes3ActiveSpellEffect effect;
                    effect.effectId = authored.effectId;
                    effect.skill = authored.skill;
                    effect.attribute = authored.attribute;
                    effect.magnitude = std::min(authored.magnitudeMin, authored.magnitudeMax);
                    effect.expiresTick = std::numeric_limits<std::uint64_t>::max();
                    passive.effects.push_back(effect);
                }
                active.push_back(std::move(passive));
            }
        }
        return result;
    }
    if (command == "equip") {
        if (object == nullptr || call.arguments.empty()) {
            result.error = "Equip requires a resident target and item"; return result;
        }
        const RecordKey item = resolveBase(call.arguments[0]);
        if (!item.valid()) { result.error = "Equip names an unresolved item"; return result; }
        WorldCommand world;
        world.type = WorldCommandType::SetEquipped;
        world.target = target;
        world.item = item;
        world.equipped = true;
        (void)m_world.queue(std::move(world));
        return result;
    }
    if (command == "resurrect") {
        if (object == nullptr || !object->actorValues.has_value()) {
            result.error = "Resurrect requires a resident actor"; return result;
        }
        WorldCommand alive;
        alive.type = WorldCommandType::SetDead;
        alive.target = target;
        alive.actorDead = false;
        (void)m_world.queue(std::move(alive));
        WorldCommand health;
        health.type = WorldCommandType::SetActorValue;
        health.target = target;
        health.actorValue = ActorValue::Health;
        health.actorValueAbsolute = object->actorValues->maxHealth;
        (void)m_world.queue(std::move(health));
        return result;
    }
    if (command == "sethealth" || command == "modhealth" || command == "modcurrenthealth") {
        if (!target.valid() || call.arguments.empty() ||
            ((object == nullptr || !object->actorValues.has_value()) &&
             actorDefinitionForTarget() == nullptr)) {
            result.error = command + " requires an authored actor and value";
            return result;
        }
        const Tes3ActorDefinition* actor = actorDefinitionForTarget();
        Tes3ReferenceOverride& override = m_tes3.referenceOverridesForRestore()[target];
        const auto savedCurrent = override.locals.find("actor:health");
        const auto savedMaximum = override.locals.find("actor:maxhealth");
        const double current = savedCurrent != override.locals.end() ? savedCurrent->second.number :
            object != nullptr && object->actorValues.has_value()
                ? object->actorValues->health : actor->health;
        const double maximum = savedMaximum != override.locals.end() ? savedMaximum->second.number :
            object != nullptr && object->actorValues.has_value()
                ? object->actorValues->maxHealth : actor->health;
        const double value = asNumber(call.arguments[0]);
        const double nextCurrent = command == "sethealth" ? value : current + value;
        const double nextMaximum = command == "modcurrenthealth" ? maximum :
            command == "sethealth" ? value : maximum + value;
        override.locals["actor:health"] = Tes3Value::fromNumber(std::max(0.0, nextCurrent));
        override.locals["actor:maxhealth"] = Tes3Value::fromNumber(std::max(0.0, nextMaximum));
        override.locals["actor:dead"] = Tes3Value::fromNumber(nextCurrent <= 0.0 ? 1.0 : 0.0);
        if (target == m_playerObject) {
            auto& filters = m_tes3.playerState().numericFilters;
            filters["health"] = std::max(0.0, nextCurrent);
            filters["maxhealth"] = std::max(0.0, nextMaximum);
            if (m_tes3.dialogue().active)
                m_tes3.dialogueForRestore().player.numericFilters = filters;
        }
        if (object == nullptr) {
            m_tes3.recordActorDeath(target, actor->id, nextCurrent <= 0.0);
            return result;
        }
        WorldCommand world;
        world.type = command == "sethealth" ? WorldCommandType::SetActorBaseValue :
            command == "modhealth" ? WorldCommandType::AdjustActorBaseValue :
            WorldCommandType::AdjustActorValue;
        world.target = target;
        world.actorValue = ActorValue::Health;
        if (command == "sethealth") world.actorValueAbsolute = static_cast<float>(value);
        else world.actorValueDelta = static_cast<float>(value);
        (void)m_world.queue(std::move(world));
        return result;
    }
    if (command == "drop") {
        if (object == nullptr || call.arguments.empty()) {
            result.error = "Drop requires a resident target and item"; return result;
        }
        const RecordKey base = resolveBase(call.arguments[0]);
        if (!base.valid()) { result.error = "Drop names an unresolved item"; return result; }
        const std::int32_t count = call.arguments.size() >= 2u
            ? std::max(1, asInt(call.arguments[1])) : 1;
        WorldCommand remove;
        remove.type = WorldCommandType::RemoveItem;
        remove.target = target;
        remove.item = base;
        remove.itemCount = count;
        (void)m_world.queue(std::move(remove));
        for (std::int32_t index = 0; index < count; ++index) {
            RuntimeObject dropped;
            dropped.id = m_world.allocateRuntimeId();
            dropped.base = base;
            dropped.kind = RuntimeObjectKind::Item;
            dropped.transform = object->transform;
            dropped.originSpace = object->currentSpace;
            dropped.currentSpace = object->currentSpace;
            WorldCommand spawn;
            spawn.type = WorldCommandType::Spawn;
            spawn.object = std::move(dropped);
            (void)m_world.queue(std::move(spawn));
        }
        return result;
    }
    if (command == "placeitem" || command == "placeitemcell") {
        const std::size_t coordinate = command == "placeitemcell" ? 2u : 1u;
        if (call.arguments.size() < coordinate + 4u) {
            result.error = command + " requires item, position, and rotation"; return result;
        }
        const RecordKey base = resolveBase(call.arguments[0]);
        if (!base.valid()) { result.error = command + " names an unresolved item"; return result; }
        RuntimeObject placed;
        placed.id = m_world.allocateRuntimeId();
        placed.base = base;
        placed.kind = RuntimeObjectKind::Item;
        placed.transform.position = {asNumber(call.arguments[coordinate]),
            asNumber(call.arguments[coordinate + 1u]), asNumber(call.arguments[coordinate + 2u])};
        placed.transform.rotationRadians[2] = static_cast<float>(
            asNumber(call.arguments[coordinate + 3u]) * (3.14159265358979323846 / 180.0));
        if (command == "placeitemcell") {
            placed.originSpace.kind = RuntimeSpaceKind::Interior;
            placed.originSpace.cell = makeTes3RecordKey("CELL", asText(call.arguments[1]));
            placed.currentSpace = placed.originSpace;
        } else if (const RuntimeObject* player = m_world.find(m_playerObject); player != nullptr) {
            placed.originSpace = player->currentSpace;
            placed.currentSpace = player->currentSpace;
        }
        WorldCommand spawn;
        spawn.type = WorldCommandType::Spawn;
        spawn.object = std::move(placed);
        (void)m_world.queue(std::move(spawn));
        return result;
    }
    if (command == "placeatpc" || command == "placeatme") {
        if (call.arguments.empty()) { result.error = command + " requires a base record"; return result; }
        const RuntimeObject* player = command == "placeatpc"
            ? m_world.find(m_playerObject) : object;
        const RecordKey base = resolveBase(call.arguments[0]);
        if (player == nullptr || !base.valid()) {
            result.error = command + " requires a resident origin and resolved base";
            return result;
        }
        const std::int32_t count = call.arguments.size() >= 2u ? std::max(1, asInt(call.arguments[1])) : 1;
        for (std::int32_t i = 0; i < count; ++i) {
            RuntimeObject spawned;
            spawned.id = m_world.allocateRuntimeId();
            spawned.base = base;
            if (base.recordType == "NPC_" || base.recordType == "CREA") {
                spawned.kind = RuntimeObjectKind::Actor;
                spawned.actorValues.emplace();
            } else if (base.recordType == "CONT") spawned.kind = RuntimeObjectKind::Container;
            else if (base.recordType == "DOOR") spawned.kind = RuntimeObjectKind::Door;
            else if (base.recordType == "ACTI") spawned.kind = RuntimeObjectKind::Activator;
            else spawned.kind = RuntimeObjectKind::Item;
            spawned.transform = player->transform;
            spawned.originSpace = player->currentSpace;
            spawned.currentSpace = player->currentSpace;
            WorldCommand world;
            world.type = WorldCommandType::Spawn;
            world.object = std::move(spawned);
            (void)m_world.queue(std::move(world));
        }
        return result;
    }
        return std::nullopt;
    };
    if (auto handled = handleInventoryAndSpawn()) return *handled;

    result.error = "unsupported gameplay MWScript native " + command;
    return result;
}

}  // namespace odai::bethesda
