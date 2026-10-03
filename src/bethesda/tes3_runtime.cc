#include "bethesda/tes3_runtime.h"

#include "core/hash.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>

namespace odai::bethesda {
namespace {

bool sameText(std::string_view left, std::string_view right) {
    return normalizeTes3Symbol(left) == normalizeTes3Symbol(right);
}

bool compare(double left, char operation, double right) {
    if (!std::isfinite(left) || !std::isfinite(right)) return false;
    switch (operation) {
        case '0': return left == right;
        case '1': return left != right;
        case '2': return left > right;
        case '3': return left >= right;
        case '4': return left < right;
        case '5': return left <= right;
        default: return false;
    }
}

double conditionValue(const Tes3DialogueCondition& condition) {
    return std::visit([](const auto value) { return static_cast<double>(value); }, condition.value);
}

std::string argumentString(const Tes3Value& value) {
    if (value.type == Tes3ValueType::String) return value.string;
    if (value.type == Tes3ValueType::Object) return value.object.toString();
    if (value.type == Tes3ValueType::Number) return std::to_string(value.number);
    return {};
}

std::int32_t argumentInt(const Tes3Value& value) {
    if (value.type == Tes3ValueType::Number) return static_cast<std::int32_t>(value.number);
    if (value.type == Tes3ValueType::String) {
        try { return std::stoi(value.string); } catch (...) { return 0; }
    }
    return 0;
}

bool containsTopic(std::string_view response, std::string_view topic) {
    const std::string text = normalizeTes3Symbol(response);
    const std::string needle = normalizeTes3Symbol(topic);
    if (needle.empty()) return false;
    std::size_t found = text.find(needle);
    while (found != std::string::npos) {
        const auto word = [](char ch) {
            const unsigned char value = static_cast<unsigned char>(ch);
            return std::isalnum(value) != 0 || ch == '_';
        };
        const bool left = found == 0u || !word(text[found - 1u]);
        const std::size_t end = found + needle.size();
        const bool right = end == text.size() || !word(text[end]);
        if (left && right) return true;
        found = text.find(needle, found + 1u);
    }
    return false;
}

}  // namespace

bool Tes3Journal::addEntry(
    const Tes3DialogueDefinition& quest, std::int32_t index,
    std::uint64_t tick, std::string& outError) {
    if (quest.type != Tes3DialogueType::Journal) {
        outError = quest.id + " is not a journal DIAL";
        return false;
    }
    const auto info = std::find_if(quest.infos.begin(), quest.infos.end(),
        [&](const Tes3DialogueInfo& value) {
            return value.dispositionOrJournalIndex == index;
        });
    if (info == quest.infos.end()) {
        outError = "journal " + quest.id + " has no entry " + std::to_string(index);
        return false;
    }
    Tes3JournalQuestState& state = m_quests[quest.record];
    state.quest = quest.record;
    state.id = quest.id;
    if (std::find(state.visitedEntries.begin(), state.visitedEntries.end(), info->record) !=
        state.visitedEntries.end()) {
        outError.clear();
        return true;
    }
    state.currentIndex = std::max(state.currentIndex, index);
    state.hasStatusFlags = std::any_of(quest.infos.begin(), quest.infos.end(),
        [](const Tes3DialogueInfo& value) { return value.questStatus != Tes3QuestStatus::None; });
    if (state.hasStatusFlags && state.classification == Tes3JournalQuestClassification::Legacy) {
        state.classification = Tes3JournalQuestClassification::Active;
    }
    if (info->questStatus == Tes3QuestStatus::Finished) {
        state.classification = Tes3JournalQuestClassification::Completed;
    } else if (info->questStatus == Tes3QuestStatus::Name ||
               info->questStatus == Tes3QuestStatus::Restart) {
        state.classification = Tes3JournalQuestClassification::Active;
    }
    state.visitedEntries.push_back(info->record);
    m_chronology.push_back(Tes3JournalVisit{
        m_nextSequence++, tick, quest.record, info->record, index,
        info->questStatus, info->sourcePlugin});
    outError.clear();
    return true;
}

bool Tes3Journal::setIndex(
    const Tes3DialogueDefinition& quest, std::int32_t index, std::string& outError) {
    if (quest.type != Tes3DialogueType::Journal) {
        outError = quest.id + " is not a journal DIAL";
        return false;
    }
    Tes3JournalQuestState& state = m_quests[quest.record];
    state.quest = quest.record;
    state.id = quest.id;
    state.currentIndex = index;
    state.hasStatusFlags = std::any_of(quest.infos.begin(), quest.infos.end(),
        [](const Tes3DialogueInfo& value) { return value.questStatus != Tes3QuestStatus::None; });
    state.classification = state.hasStatusFlags
        ? Tes3JournalQuestClassification::Active : Tes3JournalQuestClassification::Legacy;
    outError.clear();
    return true;
}

std::int32_t Tes3Journal::index(std::string_view questId) const {
    const Tes3JournalQuestState* state = find(questId);
    return state == nullptr ? 0 : state->currentIndex;
}

const Tes3JournalQuestState* Tes3Journal::find(std::string_view questId) const {
    const auto found = m_quests.find(makeTes3RecordKey("DIAL", std::string(questId)));
    return found == m_quests.end() ? nullptr : &found->second;
}

void Tes3Journal::clear() {
    m_quests.clear();
    m_chronology.clear();
    m_nextSequence = 1u;
}

bool Tes3Runtime::configure(
    std::shared_ptr<const Tes3ContentStore> content, ObjectId player,
    std::string& outError) {
    clear();
    if (content == nullptr || !player.valid()) {
        outError = "TES3 runtime requires immutable content and an explicit player ObjectId";
        return false;
    }
    m_content = std::move(content);
    m_player = std::move(player);
    m_playerState.object = m_player;
    m_nativeRegistry = Tes3NativeRegistry::coreRuntimeRegistry();
    for (const auto& [key, global] : m_content->globals()) {
        (void)key;
        m_scripts.globals()[normalizeTes3Symbol(global.id)] = Tes3Value::fromNumber(global.value);
    }
    Tes3ScriptCompiler compiler;
    const auto validateScriptedEffects = [&](Tes3ScriptProgram& program) {
        for (const Tes3Instruction& instruction : program.instructions) {
            if (instruction.op != Tes3OpCode::Call ||
                (instruction.command != "cast" && instruction.command != "addspell") ||
                instruction.arguments.empty()) continue;
            std::string spellId = instruction.arguments.front();
            if (program.locals.contains(normalizeTes3Symbol(spellId)) ||
                m_scripts.globals().contains(normalizeTes3Symbol(spellId))) continue;
            spellId.erase(std::remove(spellId.begin(), spellId.end(), '"'), spellId.end());
            const Tes3SpellDefinition* spell = m_content->findSpell(spellId);
            if (spell == nullptr) {
                const std::string issue = instruction.command + ":unresolved:" +
                    normalizeTes3Symbol(spellId);
                m_scriptCheck.unsupportedCommands.insert(issue);
                program.unsupportedOperations.insert(issue);
                continue;
            }
            if (instruction.command == "addspell" &&
                (spell->type < 1 || spell->type > 4)) continue;
            for (const Tes3SpellEffect& effect : spell->effects) {
                const bool supported = instruction.command == "cast"
                    ? (effect.effectId == 69 || effect.effectId == 70 ||
                       effect.effectId == 71 || effect.effectId == 74 ||
                       effect.effectId == 79 ||
                       effect.effectId == 83)
                    : (effect.effectId == 3 || effect.effectId == 4 ||
                       effect.effectId == 5 || effect.effectId == 6 ||
                       effect.effectId == 17 || effect.effectId == 79 ||
                       effect.effectId == 83 || effect.effectId == 94 ||
                       effect.effectId == 95 || effect.effectId == 96 ||
                       effect.effectId == 132);
                if (!supported) {
                    const std::string issue = instruction.command + ":effect:" +
                        std::to_string(effect.effectId);
                    m_scriptCheck.unsupportedCommands.insert(issue);
                    program.unsupportedOperations.insert(issue);
                }
            }
        }
    };
    for (const auto& [key, script] : m_content->scripts()) {
        (void)key;
        ++m_scriptCheck.scripts;
        if (script.source.empty()) {
            m_scriptCheck.diagnostics.push_back(script.id +
                (script.bytecode.empty() ? ": missing source and bytecode" :
                 ": bytecode-only script requires SCDT fallback decoder"));
            continue;
        }
        Tes3CompileResult compiled = compiler.compile(script.source, script.id);
        for (const Tes3CompileDiagnostic& diagnostic : compiled.diagnostics) {
            if (diagnostic.error) m_scriptCheck.diagnostics.push_back(
                script.id + ":" + std::to_string(diagnostic.line) + ": " + diagnostic.message);
        }
        if (!compiled.success()) continue;
        validateScriptedEffects(compiled.program);
        for (const std::string& command : compiled.program.commands) {
            if (m_scripts.globals().contains(command)) continue;
            ++m_scriptCheck.commandUse[command];
            if (m_nativeRegistry.find(command) == nullptr) {
                m_scriptCheck.unsupportedCommands.insert(command);
                compiled.program.unsupportedOperations.insert(command);
            }
        }
        if (!m_scripts.registerProgram(std::move(compiled.program), outError)) return false;
        ++m_scriptCheck.compiled;
    }
    for (const auto& [topicKey, topic] : m_content->dialogues()) {
        (void)topicKey;
        for (const Tes3DialogueInfo& info : topic.infos) {
            if (info.resultScript.empty()) continue;
            ++m_scriptCheck.resultScripts;
            const std::string programId = resultProgramId(info.record);
            Tes3CompileResult compiled = compiler.compile(info.resultScript, programId);
            for (const Tes3CompileDiagnostic& diagnostic : compiled.diagnostics) {
                if (diagnostic.error) m_scriptCheck.diagnostics.push_back(
                    info.record.toString() + ":" + std::to_string(diagnostic.line) + ": " +
                    diagnostic.message);
            }
            if (!compiled.success()) continue;
            validateScriptedEffects(compiled.program);
            for (const std::string& command : compiled.program.commands) {
                if (m_scripts.globals().contains(command)) continue;
                ++m_scriptCheck.commandUse[command];
                if (m_nativeRegistry.find(command) == nullptr) {
                    m_scriptCheck.unsupportedCommands.insert(command);
                    compiled.program.unsupportedOperations.insert(command);
                }
            }
            if (!m_scripts.registerProgram(std::move(compiled.program), outError)) return false;
            ++m_scriptCheck.compiled;
        }
    }
    outError.clear();
    return true;
}

Tes3VmStepResult Tes3Runtime::step(std::uint64_t tick, std::uint32_t instructionBudget) {
    m_currentTick = tick;
    for (auto target = m_activeSpells.begin(); target != m_activeSpells.end();) {
        auto& spells = target->second;
        std::erase_if(spells, [&](Tes3ActiveSpell& spell) {
            std::erase_if(spell.effects, [&](const Tes3ActiveSpellEffect& effect) {
                return effect.expiresTick <= tick;
            });
            return spell.effects.empty();
        });
        if (spells.empty()) target = m_activeSpells.erase(target);
        else ++target;
    }
    const auto pendingThread = m_pendingResult.has_value()
        ? std::optional<std::uint64_t>(m_pendingResult->threadId) : std::nullopt;
    Tes3VmStepResult result = m_scripts.step(tick, instructionBudget,
        [this](const Tes3NativeCall& call) { return executeNative(call); }, pendingThread);
    if (pendingThread.has_value()) {
        const auto thread = m_scripts.threads().find(*pendingThread);
        if (!result.diagnostics.empty() || thread == m_scripts.threads().end() ||
            thread->second.state == Tes3ThreadState::Failed) {
            if (result.diagnostics.empty())
                result.diagnostics.push_back("dialogue result thread disappeared before commit");
            finishPendingResult(false);
        } else if (thread->second.state == Tes3ThreadState::Completed) {
            finishPendingResult(true);
        }
    }
    return result;
}

std::map<std::string, double> Tes3Runtime::dialogueActorLocals(
    const Tes3DialogueActorState& actor) const {
    auto locals = actor.locals;
    std::erase_if(locals, [](const auto& local) { return local.first.find(':') != std::string::npos; });
    const Tes3ActorDefinition* definition = m_content->findActor("NPC_", actor.id);
    if (definition == nullptr) definition = m_content->findActor("CREA", actor.id);
    if (definition != nullptr && definition->script.valid()) {
        const auto program = m_scripts.programs().find(normalizeTes3Symbol(definition->script.textId));
        if (program != m_scripts.programs().end()) {
            for (const auto& [name, type] : program->second.locals) {
                (void)type;
                locals.try_emplace(name, 0.0);
            }
        }
    }
    const auto saved = m_referenceOverrides.find(actor.object);
    if (saved != m_referenceOverrides.end()) {
        for (const auto& [name, value] : saved->second.locals) {
            if (locals.contains(name) && value.type == Tes3ValueType::Number) locals[name] = value.number;
        }
    }
    for (const auto& [id, thread] : m_scripts.threads()) {
        (void)id;
        if (!thread.local || thread.owner != actor.object) continue;
        for (const auto& [name, value] : thread.locals) {
            if (value.type == Tes3ValueType::Number) locals[name] = value.number;
        }
    }
    return locals;
}

std::optional<double> Tes3Runtime::dialogueFunctionValue(int function,
    const Tes3DialogueActorState& actor, const Tes3DialoguePlayerState& player) const {
    // Read the same named state that MWScript natives mutate. Unsupported
    // functions and unavailable named inputs stay ineligible.
    constexpr std::string_view skills[] = {
        "block", "armorer", "mediumarmor", "heavyarmor", "bluntweapon", "longblade",
        "axe", "spear", "athletics", "enchant", "destruction", "alteration", "illusion",
        "conjuration", "mysticism", "restoration", "alchemy", "unarmored", "security",
        "sneak", "acrobatics", "lightarmor", "shortblade", "marksman", "mercantile",
        "speechcraft", "handtohand"};
    constexpr std::string_view attributes[] = {
        "intelligence", "willpower", "agility", "speed", "endurance", "personality", "luck"};
    std::string key;
    if (function >= 11 && function <= 37) key = skills[function - 11];
    else if (function >= 51 && function <= 57) key = attributes[function - 51];
    else switch (function) {
        case 0: case 1: {
            if (actor.faction.empty()) return 0.0;
            const auto* faction = m_content->findFaction(actor.faction);
            if (faction == nullptr) return std::nullopt;
            double reaction = 0.0;
            for (const auto& [id, rank] : player.factionRanks) {
                (void)rank;
                if (m_content->findFaction(id) == nullptr) return std::nullopt;
                const auto authored = faction->reactions.find(id);
                double value = authored == faction->reactions.end() ? 0.0 : authored->second;
                const auto changed = player.numericFilters.find("faction_reaction:" +
                    normalizeTes3Symbol(actor.faction) + ":" + id);
                if (changed != player.numericFilters.end()) value += changed->second;
                reaction = function == 0 ? std::min(reaction, value) : std::max(reaction, value);
            }
            return reaction;
        }
        case 2: {
            if (actor.faction.empty()) return 0.0;
            const auto* faction = m_content->findFaction(actor.faction);
            if (faction == nullptr) return std::nullopt;
            const auto found = player.factionRanks.find(normalizeTes3Symbol(actor.faction));
            const int rank = found == player.factionRanks.end() ? -1 : found->second;
            if (rank >= 9) return 0.0;
            if (rank < -1) return std::nullopt;
            const auto& requirement = faction->ranks[rank + 1];
            constexpr std::string_view allAttributes[] = {"strength", "intelligence", "willpower",
                "agility", "speed", "endurance", "personality", "luck"};
            bool attributesPass = true;
            for (std::size_t i = 0; i < 2; ++i) {
                const auto index = faction->attributes[i];
                if (index < 0 || index >= 8) return std::nullopt;
                const auto value = player.numericFilters.find(std::string(allAttributes[index]));
                if (value == player.numericFilters.end()) return std::nullopt;
                attributesPass &= value->second >= (i == 0 ? requirement.attribute1 : requirement.attribute2);
            }
            std::vector<double> values;
            for (int index : faction->skills) {
                if (index == -1) continue;
                if (index < 0 || index >= 27) return std::nullopt;
                const auto value = player.numericFilters.find(std::string(skills[index]));
                if (value == player.numericFilters.end()) return std::nullopt;
                values.push_back(value->second);
            }
            std::sort(values.begin(), values.end(), std::greater<>());
            bool skillsPass = true;
            for (std::size_t i = 0; i < std::min<std::size_t>(3, values.size()); ++i)
                skillsPass &= values[i] >= (i == 0 ? requirement.primarySkill : requirement.favouredSkill);
            const auto reputation = player.numericFilters.find("faction_rep:" + normalizeTes3Symbol(actor.faction));
            const double rep = reputation == player.numericFilters.end() ? 0.0 : reputation->second;
            return (attributesPass && skillsPass ? 1.0 : 0.0) + (rep >= requirement.reputation ? 2.0 : 0.0);
        }
        case 3: case 4: {
            const auto* definition = m_content->findActor("NPC_", actor.id);
            if (definition == nullptr) definition = m_content->findActor("CREA", actor.id);
            if (definition == nullptr) return std::nullopt;
            double value = function == 3 ? definition->reputation : definition->health;
            double maximum = definition->health;
            const auto saved = m_referenceOverrides.find(actor.object);
            const std::string key = function == 3 ? "stat:reputation" : "actor:health";
            const auto resident = actor.locals.find(key);
            if (resident != actor.locals.end()) value = resident->second;
            const auto maxHealth = actor.locals.find("actor:maxhealth");
            if (maxHealth != actor.locals.end()) maximum = maxHealth->second;
            if (saved != m_referenceOverrides.end()) {
                const auto changed = saved->second.locals.find(key);
                if (changed != saved->second.locals.end()) value = changed->second.number;
                const auto changedMaximum = saved->second.locals.find("actor:maxhealth");
                if (changedMaximum != saved->second.locals.end()) maximum = changedMaximum->second.number;
            }
            if (function == 3) return value;
            if (maximum <= 0 || !std::isfinite(value) || !std::isfinite(maximum)) return std::nullopt;
            return std::trunc(100.0 * value / maximum);
        }
        case 5: key = "reputation"; break;
        case 6: key = "level"; break;
        case 7: {
            const auto health = player.numericFilters.find("health");
            const auto maximum = player.numericFilters.find("maxhealth");
            if (health != player.numericFilters.end() && maximum != player.numericFilters.end() && maximum->second > 0)
                return std::trunc(100.0 * health->second / maximum->second);
            break;
        }
        case 8: key = "magicka"; break;
        case 9: key = "fatigue"; break;
        case 10: key = "strength"; break;
        case 38: if (player.gender >= 0) return player.gender; break;
        case 39: {
            if (actor.faction.empty()) return std::nullopt;
            const auto found = player.numericFilters.find("expelled:" + normalizeTes3Symbol(actor.faction));
            return found == player.numericFilters.end() ? 0.0 : found->second;
        }
        case 40: case 41: case 58: case 60: {
            const auto active = m_activeSpells.find(player.object);
            if (active == m_activeSpells.end()) return 0.0;
            for (const auto& spell : active->second) {
                const auto definition = m_content->spells().find(spell.spell);
                for (const auto& effect : spell.effects) {
                    if (effect.expiresTick <= m_currentTick) continue;
                    if ((function == 58 && effect.effectId == 132 && effect.magnitude != 0) ||
                        (function == 60 && effect.effectId == 95 && effect.magnitude > 0) ||
                        (definition != m_content->spells().end() &&
                            ((function == 40 && definition->second.type == 3) ||
                             (function == 41 && definition->second.type == 2)))) return 1.0;
                }
            }
            return 0.0;
        }
        case 42: key = "clothingmodifier"; break;
        case 43: key = "crimelevel"; break;
        case 44: if (actor.gender >= 0 && player.gender >= 0) return actor.gender == player.gender; break;
        case 45: if (!actor.race.empty() && !player.race.empty()) return sameText(actor.race, player.race); break;
        case 46: return !actor.faction.empty() && player.factionRanks.contains(normalizeTes3Symbol(actor.faction));
        case 47: {
            const auto rank = player.factionRanks.find(normalizeTes3Symbol(actor.faction));
            if (actor.faction.empty()) return 0.0;
            if (actor.rank < 0) return std::nullopt;
            return (rank == player.factionRanks.end() ? -1 : rank->second) - actor.rank;
        }
        case 59: key = "currentweather"; break;
        case 61: {
            const auto* definition = m_content->findActor("NPC_", actor.id);
            if (definition == nullptr) definition = m_content->findActor("CREA", actor.id);
            if (definition != nullptr) return definition->level;
            break;
        }
        case 63: return actor.talkedToBefore;
        case 64: key = "health"; break;
        case 73: key = "werewolfkills"; break;
        default: break;
    }
    if (!key.empty()) {
        const auto found = player.numericFilters.find(key);
        if (found != player.numericFilters.end()) {
            double value = found->second;
            if ((function >= 10 && function <= 37) || (function >= 51 && function <= 57)) {
                const auto damage = player.numericFilters.find("damage:" + key);
                if (damage != player.numericFilters.end()) value -= damage->second;
                const auto active = m_activeSpells.find(player.object);
                if (active != m_activeSpells.end()) {
                    constexpr std::string_view allAttributes[] = {"strength", "intelligence", "willpower",
                        "agility", "speed", "endurance", "personality", "luck"};
                    for (const auto& spell : active->second) for (const auto& effect : spell.effects) {
                        if (effect.expiresTick <= m_currentTick) continue;
                        if ((effect.effectId == 79 || effect.effectId == 17) && effect.attribute >= 0 &&
                            effect.attribute < 8 && allAttributes[effect.attribute] == key)
                            value += (effect.effectId == 79 ? 1 : -1) * effect.magnitude;
                        if ((effect.effectId == 83 || effect.effectId == 21) && effect.skill >= 0 &&
                            effect.skill < 27 && skills[effect.skill] == key)
                            value += (effect.effectId == 83 ? 1 : -1) * effect.magnitude;
                    }
                }
                if (!std::isfinite(value)) return std::nullopt;
                value = std::trunc(std::max(0.0, value));
            }
            return value;
        }
    }
    return std::nullopt;
}

bool Tes3Runtime::matches(
    const Tes3DialogueInfo& info, const Tes3DialogueActorState& actor,
    const Tes3DialoguePlayerState& player, bool strict) const {
    (void)strict;
    if (!info.actor.empty() && !sameText(info.actor, actor.id)) return false;
    if (!info.race.empty() && !sameText(info.race, actor.race)) return false;
    if (!info.actorClass.empty() && !sameText(info.actorClass, actor.actorClass)) return false;
    if (info.factionless && !actor.faction.empty()) return false;
    if (!info.faction.empty() && !info.factionless && !sameText(info.faction, actor.faction)) return false;
    if (!info.cell.empty() && !normalizeTes3Symbol(actor.cell).starts_with(
            normalizeTes3Symbol(info.cell))) return false;
    double disposition = actor.disposition;
    const auto reference = m_content->references().find(actor.object);
    // Caller-supplied synthetic speaker state can deliberately differ from an
    // imported reference. Resolve gameplay disposition only for that identity.
    const bool authoredIdentity = reference == m_content->references().end() ||
        sameText(reference->second.base.textId, actor.id);
    if (m_externalNative && actor.object.valid() && authoredIdentity) {
        Tes3NativeCall call; call.command = "getdisposition"; call.owner = actor.object;
        const auto result = m_externalNative(call);
        if (result.error.empty() && result.value.type == Tes3ValueType::Number) disposition = result.value.number;
    }
    if (!std::isfinite(disposition) || disposition < info.dispositionOrJournalIndex) return false;
    if (info.rank >= 0 && actor.rank < info.rank) return false;
    if (info.gender >= 0 && actor.gender != info.gender) return false;
    if (!info.playerFaction.empty() || info.playerRank >= 0) {
        const auto rank = player.factionRanks.find(normalizeTes3Symbol(
            info.playerFaction.empty() ? actor.faction : info.playerFaction));
        if (rank == player.factionRanks.end()) return false;
        if (info.playerRank >= 0 && rank->second < info.playerRank) return false;
    }
    for (const Tes3DialogueCondition& condition : info.conditions) {
        // The legacy diagnostic flag must never turn unknown inputs into
        // eligible gameplay responses.
        if (!condition.valid) return false;
        double actual = 0.0;
        const std::string variable = normalizeTes3Symbol(condition.variable);
        const int function = static_cast<int>(condition.function);
        if (condition.function == Tes3ConditionFunction::Global) {
            const auto found = m_scripts.globals().find(variable);
            if (found == m_scripts.globals().end() || found->second.type != Tes3ValueType::Number)
                return false;
            actual = found->second.number;
        } else if (condition.function == Tes3ConditionFunction::Local ||
                   condition.function == Tes3ConditionFunction::NotLocal) {
            const auto locals = dialogueActorLocals(actor);
            const auto found = locals.find(variable);
            if (found == locals.end()) {
                // NotLocal also expresses authored absence (e.g. NoLore on
                // ordinary NPCs). Distinguish a known missing declaration
                // from a speaker/script whose input cannot be resolved.
                const auto* definition = m_content->findActor("NPC_", actor.id);
                if (definition == nullptr) definition = m_content->findActor("CREA", actor.id);
                if (condition.function != Tes3ConditionFunction::NotLocal || definition == nullptr)
                    return false;
                if (!definition->script.valid()) continue;
                const auto program = m_scripts.programs().find(normalizeTes3Symbol(definition->script.textId));
                if (program == m_scripts.programs().end() || program->second.locals.contains(variable))
                    return false;
                continue;
            }
            const bool passed = compare(found->second, condition.comparison, conditionValue(condition));
            if (condition.function == Tes3ConditionFunction::NotLocal ? passed : !passed) return false;
            continue;
        } else if (condition.function == Tes3ConditionFunction::Journal) {
            const auto* quest = m_content->findDialogue(condition.variable);
            if (quest == nullptr || quest->type != Tes3DialogueType::Journal) return false;
            actual = m_journal.index(condition.variable);
        } else if (condition.function == Tes3ConditionFunction::Item) {
            const RecordKey wanted = makeTes3RecordKey("REFR", condition.variable);
            for (const auto& [item, count] : player.inventory) {
                if (item.textId == wanted.textId) actual += count;
            }
        } else if (condition.function == Tes3ConditionFunction::Dead) {
            if (m_content->findActor("NPC_", variable) == nullptr &&
                m_content->findActor("CREA", variable) == nullptr) return false;
            const auto found = player.deathCounts.find(variable);
            actual = found == player.deathCounts.end() ? 0.0 : found->second;
        } else if (condition.function == Tes3ConditionFunction::NotId) {
            if (actor.id.empty() || sameText(actor.id, condition.variable)) return false;
            continue;
        } else if (condition.function == Tes3ConditionFunction::NotFaction) {
            if (sameText(actor.faction, condition.variable)) return false;
            continue;
        } else if (condition.function == Tes3ConditionFunction::NotClass) {
            if (actor.actorClass.empty() || sameText(actor.actorClass, condition.variable)) return false;
            continue;
        } else if (condition.function == Tes3ConditionFunction::NotRace) {
            if (actor.race.empty() || sameText(actor.race, condition.variable)) return false;
            continue;
        } else if (condition.function == Tes3ConditionFunction::NotCell) {
            if (actor.cell.empty() || normalizeTes3Symbol(actor.cell).starts_with(variable)) return false;
            continue;
        } else if (function == 50) {  // Choice
            if (m_dialogue.choice == -1) return false;
            actual = m_dialogue.choice;
        } else {
            const auto value = dialogueFunctionValue(function, actor, player);
            if (!value.has_value()) return false;
            actual = *value;
        }
        if (!compare(actual, condition.comparison, conditionValue(condition))) return false;
    }
    return true;
}

const Tes3DialogueInfo* Tes3Runtime::selectInfo(
    const Tes3DialogueDefinition& topic, bool strict) const {
    for (const Tes3DialogueInfo& info : topic.infos) {
        if (m_dialogue.choice != -1 && std::none_of(info.conditions.begin(), info.conditions.end(),
                [](const Tes3DialogueCondition& condition) { return static_cast<int>(condition.function) == 50; }))
            continue;
        if (matches(info, m_dialogue.actor, m_dialogue.player, strict)) return &info;
    }
    return nullptr;
}

Tes3DialogueResponse Tes3Runtime::startDialogue(
    Tes3DialogueActorState actor, Tes3DialoguePlayerState player, bool strict) {
    Tes3DialogueResponse response;
    if (m_pendingResult.has_value()) {
        response.diagnostics.push_back("previous dialogue result is still suspended");
        return response;
    }
    if (m_content == nullptr || !actor.object.valid() || !player.object.valid()) {
        response.diagnostics.push_back("dialogue requires configured TES3 content and participants");
        return response;
    }
    if (m_dialogueEndHook) m_dialogueEndHook();
    m_dialogue = {};
    m_dialogue.active = true;
    m_dialogue.actor = std::move(actor);
    m_dialogue.actor.talkedToBefore = !m_content->references().contains(m_dialogue.actor.object) &&
        std::any_of(m_topicResponseActors.begin(), m_topicResponseActors.end(),
            [&](const auto& response) { return sameText(response.second, m_dialogue.actor.id); });
    const auto savedActor = m_referenceOverrides.find(m_dialogue.actor.object);
    if (savedActor != m_referenceOverrides.end()) {
        const auto talked = savedActor->second.locals.find("actor:talked");
        if (talked != savedActor->second.locals.end())
            m_dialogue.actor.talkedToBefore = talked->second.number != 0;
        const auto rank = savedActor->second.locals.find("stat:rank");
        if (rank != savedActor->second.locals.end())
            m_dialogue.actor.rank = static_cast<std::int8_t>(rank->second.number);
        const auto disposition = savedActor->second.locals.find("stat:disposition");
        if (disposition != savedActor->second.locals.end())
            m_dialogue.actor.disposition = static_cast<float>(disposition->second.number);
    }
    m_playerState = std::move(player);
    m_dialogue.player = m_playerState;
    for (const auto& [key, topic] : m_content->dialogues()) {
        (void)key;
        if (topic.type != Tes3DialogueType::Greeting) continue;
        const Tes3DialogueInfo* info = selectInfo(topic, strict);
        if (info != nullptr) return activateInfo(topic, *info, strict);
    }
    m_dialogue.active = false;
    response.diagnostics.push_back("no greeting passed TES3 dialogue filters");
    return response;
}

std::vector<std::string> Tes3Runtime::availableTopics(bool strict) const {
    std::vector<std::string> result;
    if (!m_dialogue.active || m_content == nullptr) return result;
    for (const RecordKey& key : m_knownTopics) {
        const auto topic = m_content->dialogues().find(key);
        if (topic == m_content->dialogues().end() ||
            topic->second.type != Tes3DialogueType::Topic) continue;
        if (selectInfo(topic->second, strict) != nullptr) result.push_back(topic->second.id);
    }
    std::sort(result.begin(), result.end(), [](const std::string& left, const std::string& right) {
        return normalizeTes3Symbol(left) < normalizeTes3Symbol(right);
    });
    return result;
}

Tes3DialogueResponse Tes3Runtime::selectTopic(std::string_view topicId, bool strict) {
    Tes3DialogueResponse response;
    if (m_pendingResult.has_value()) {
        response.diagnostics.push_back("previous dialogue result is still suspended");
        return response;
    }
    if (!m_dialogue.active || m_content == nullptr) {
        response.diagnostics.push_back("no active TES3 conversation");
        return response;
    }
    const RecordKey key = makeTes3RecordKey("DIAL", std::string(topicId));
    if (!m_knownTopics.contains(key)) {
        response.diagnostics.push_back("topic is not known: " + std::string(topicId));
        return response;
    }
    const auto topic = m_content->dialogues().find(key);
    if (topic == m_content->dialogues().end() || topic->second.type != Tes3DialogueType::Topic) {
        response.diagnostics.push_back("unknown TES3 topic: " + std::string(topicId));
        return response;
    }
    if (!m_dialogue.choices.empty()) {
        response.diagnostics.push_back("answer the authored choice before selecting a topic");
        return response;
    }
    m_dialogue.choice = -1;
    const Tes3DialogueInfo* info = selectInfo(topic->second, strict);
    if (info == nullptr) {
        response.diagnostics.push_back("topic has no matching response");
        return response;
    }
    return activateInfo(topic->second, *info, strict);
}

Tes3DialogueResponse Tes3Runtime::selectPersuasionResponse(std::string_view responseId, bool strict) {
    Tes3DialogueResponse response;
    if (!m_dialogue.active || !m_content || m_pendingResult || !m_dialogue.choices.empty()) {
        response.diagnostics.push_back("persuasion response requires an available conversation"); return response;
    }
    const auto topic = m_content->dialogues().find(makeTes3RecordKey("DIAL", std::string(responseId)));
    if (topic == m_content->dialogues().end() || topic->second.type != Tes3DialogueType::Persuasion) {
        response.diagnostics.push_back("unknown TES3 persuasion response: " + std::string(responseId)); return response;
    }
    m_dialogue.choice = -1;
    const auto* info = selectInfo(topic->second, strict);
    if (!info) { response.diagnostics.push_back("persuasion response has no matching INFO"); return response; }
    return activateInfo(topic->second, *info, strict);
}

Tes3DialogueResponse Tes3Runtime::answerChoice(std::int32_t value, bool strict) {
    Tes3DialogueResponse response;
    const auto choice = std::find_if(m_dialogue.choices.begin(), m_dialogue.choices.end(),
        [&](const Tes3DialogueChoice& item) { return item.value == value; });
    if (choice == m_dialogue.choices.end() || m_content == nullptr) {
        response.diagnostics.push_back("dialogue choice is not currently available");
        return response;
    }
    if (m_pendingResult.has_value()) {
        const auto thread = m_scripts.threadsForRestore().find(m_pendingResult->threadId);
        if (thread == m_scripts.threadsForRestore().end() ||
            thread->second.state != Tes3ThreadState::Suspended ||
            thread->second.suspensionReason != "messagebox") {
            response.diagnostics.push_back("dialogue result is not waiting for a message choice");
            return response;
        }
        m_dialogue.choice = value;
        m_dialogue.choices.clear();
        m_dialogue.messageBoxText.clear();
        thread->second.eventVariables["buttonpressed"] = Tes3Value::fromNumber(value);
        thread->second.state = Tes3ThreadState::Running;
        thread->second.suspensionReason.clear();
        const Tes3VmStepResult script = step(m_currentTick, 10000u);
        if (!script.diagnostics.empty()) {
            response.diagnostics = script.diagnostics;
            return response;
        }
        response.accepted = true;
        response.text = m_dialogue.messageBoxText;
        response.choices = m_dialogue.choices;
        response.goodbye = !m_pendingResult.has_value() && m_dialogue.goodbye;
        if (response.goodbye) { if (m_dialogueEndHook) m_dialogueEndHook(); m_dialogue.active = false; }
        return response;
    }
    if (!m_dialogue.active) {
        m_dialogue.choice = value;
        m_dialogue.choices.clear();
        m_dialogue.messageBoxText.clear();
        for (auto& [id, thread] : m_scripts.threadsForRestore()) {
            (void)id;
            if (thread.state == Tes3ThreadState::Suspended &&
                thread.suspensionReason == "messagebox") {
                thread.eventVariables["buttonpressed"] = Tes3Value::fromNumber(value);
                thread.state = Tes3ThreadState::Running;
                thread.suspensionReason.clear();
            }
        }
        response.accepted = true;
        return response;
    }
    const Tes3DialogueState previous = m_dialogue;
    m_dialogue.choice = value;
    m_dialogue.choices.clear();
    const auto topic = m_content->dialogues().find(m_dialogue.currentTopic);
    if (topic == m_content->dialogues().end()) {
        m_dialogue = previous;
        response.diagnostics.push_back("current dialogue topic disappeared");
        return response;
    }
    const Tes3DialogueInfo* info = selectInfo(topic->second, strict);
    if (info == nullptr) {
        m_dialogue = previous;
        response.diagnostics.push_back("choice has no matching response");
        return response;
    }
    response = activateInfo(topic->second, *info, strict);
    if (!response.accepted) m_dialogue = previous;
    else m_dialogue.choice = -1;
    return response;
}

Tes3DialogueResponse Tes3Runtime::activateInfo(
    const Tes3DialogueDefinition& topic, const Tes3DialogueInfo& info, bool strict) {
    (void)strict;
    Tes3DialogueResponse response;
    if (m_pendingResult.has_value()) {
        response.diagnostics.push_back("previous dialogue result is still suspended");
        return response;
    }
    std::uint64_t resultThread = 0u;
    if (!info.resultScript.empty()) {
        PendingResult pending;
        pending.journal = m_journal;
        pending.threads = m_scripts.threads();
        pending.globals = m_scripts.globals();
        pending.nextThreadId = m_scripts.nextThreadId();
        pending.dialogue = m_dialogue;
        pending.player = m_playerState;
        pending.topics = m_knownTopics;
        pending.responseActors = m_topicResponseActors;
        pending.references = m_referenceOverrides;
        pending.spells = m_activeSpells;
        pending.sounds = m_activeSounds;
        std::string error;
        resultThread = m_scripts.start(resultProgramId(info.record),
            m_dialogue.actor.object, error);
        if (resultThread == 0u) {
            response.diagnostics.push_back(error);
            return response;
        }
        auto& resultLocals = m_scripts.threadsForRestore().at(resultThread).locals;
        for (const auto& [name, value] : dialogueActorLocals(m_dialogue.actor))
            resultLocals.try_emplace(name, Tes3Value::fromNumber(value));
        pending.threadId = resultThread;
        m_pendingResult = std::move(pending);
        if (m_beginResultTransaction) m_beginResultTransaction();
    }
    response.accepted = true;
    response.topic = topic.record;
    response.info = info.record;
    response.text = info.response;
    m_dialogue.currentTopic = topic.record;
    m_dialogue.currentInfo = info.record;
    m_topicResponseActors[info.record] = m_dialogue.actor.id;
    if (topic.type == Tes3DialogueType::Greeting &&
        m_content->references().contains(m_dialogue.actor.object))
        m_referenceOverrides[m_dialogue.actor.object].locals["actor:talked"] = Tes3Value::fromNumber(1);
    m_dialogue.exhaustedInfos.insert(info.record);
    m_dialogue.choices.clear();
    m_dialogue.goodbye = false;
    discoverTopics(info.response, response.discoveredTopics);
    if (resultThread != 0u) {
        const Tes3VmStepResult script = step(m_currentTick, 10000u);
        const auto resultState = m_scripts.threads().find(resultThread);
        const bool succeeded = script.diagnostics.empty() &&
            resultState != m_scripts.threads().end() &&
            resultState->second.state != Tes3ThreadState::Failed;
        if (!succeeded) {
            response = {};
            response.diagnostics = script.diagnostics;
            if (response.diagnostics.empty())
                response.diagnostics.push_back("dialogue result " + resultProgramId(info.record) +
                    " failed before its transition committed");
            return response;
        }
    }
    response.choices = m_dialogue.choices;
    if (!m_dialogue.messageBoxText.empty())
        response.text += "\n\n" + m_dialogue.messageBoxText;
    response.goodbye = !m_pendingResult.has_value() && m_dialogue.goodbye;
    if (response.goodbye) { if (m_dialogueEndHook) m_dialogueEndHook(); m_dialogue.active = false; }
    return response;
}

void Tes3Runtime::finishPendingResult(bool commit) {
    if (!m_pendingResult.has_value()) return;
    if (commit) {
        const auto& result = m_scripts.threads().at(m_pendingResult->threadId);
        const auto actorLocals = dialogueActorLocals(m_dialogue.actor);
        for (const auto& [name, value] : result.locals) {
            if (!actorLocals.contains(name)) continue;
            m_dialogue.actor.locals[name] = value.number;
            m_referenceOverrides[m_dialogue.actor.object].locals[name] = value;
            for (auto& [id, thread] : m_scripts.threadsForRestore()) {
                (void)id;
                if (thread.local && thread.owner == m_dialogue.actor.object && thread.locals.contains(name))
                    thread.locals[name] = value;
            }
        }
    }
    if (m_finishResultTransaction) m_finishResultTransaction(commit);
    if (!commit) {
        PendingResult& pending = *m_pendingResult;
        m_journal = std::move(pending.journal);
        m_scripts.threadsForRestore() = std::move(pending.threads);
        m_scripts.globals() = std::move(pending.globals);
        m_scripts.setNextThreadId(pending.nextThreadId);
        m_dialogue = std::move(pending.dialogue);
        m_playerState = std::move(pending.player);
        m_knownTopics = std::move(pending.topics);
        m_topicResponseActors = std::move(pending.responseActors);
        m_referenceOverrides = std::move(pending.references);
        m_activeSpells = std::move(pending.spells);
        m_activeSounds = std::move(pending.sounds);
    }
    m_pendingResult.reset();
}

void Tes3Runtime::discoverTopics(
    std::string_view response, std::vector<std::string>& outDiscovered) {
    if (m_content == nullptr) return;
    for (const auto& [key, topic] : m_content->dialogues()) {
        if (topic.type != Tes3DialogueType::Topic || m_knownTopics.contains(key)) continue;
        if (!containsTopic(response, topic.id)) continue;
        m_knownTopics.insert(key);
        outDiscovered.push_back(topic.id);
    }
}

bool Tes3Runtime::addTopic(std::string_view topicId) {
    if (m_content == nullptr) return false;
    const RecordKey key = makeTes3RecordKey("DIAL", std::string(topicId));
    const auto found = m_content->dialogues().find(key);
    if (found == m_content->dialogues().end() || found->second.type != Tes3DialogueType::Topic) {
        return false;
    }
    m_knownTopics.insert(key);
    return true;
}

Tes3NativeResult Tes3Runtime::executeNative(const Tes3NativeCall& call) {
    Tes3NativeResult result;
    const std::string command = normalizeTes3Symbol(call.command);
    if ((command == "journal" || command == "setjournalindex" ||
         command == "getjournalindex") && call.arguments.empty()) {
        result.error = command + " requires a journal id";
        return result;
    }
    if (command == "journal" || command == "setjournalindex") {
        if (call.arguments.size() < 2u || m_content == nullptr) {
            result.error = command + " requires journal id and index";
            return result;
        }
        const std::string id = argumentString(call.arguments[0]);
        const Tes3DialogueDefinition* quest = m_content->findDialogue(id);
        std::string error;
        const bool ok = quest != nullptr && (command == "journal"
            ? m_journal.addEntry(*quest, argumentInt(call.arguments[1]), call.tick, error)
            : m_journal.setIndex(*quest, argumentInt(call.arguments[1]), error));
        if (!ok) result.error = quest == nullptr ? "unknown journal " + id : error;
        return result;
    }
    if (command == "getjournalindex") {
        result.value = Tes3Value::fromNumber(m_journal.index(argumentString(call.arguments[0])));
        return result;
    }
    if (command == "addtopic") {
        if (call.arguments.empty() || !addTopic(argumentString(call.arguments[0]))) {
            result.error = "AddTopic names an unknown topic";
        }
        return result;
    }
    if (command == "choice") {
        if (call.arguments.size() < 2u || (call.arguments.size() % 2u) != 0u) {
            result.error = "Choice requires label/value pairs";
            return result;
        }
        for (std::size_t i = 0u; i < call.arguments.size(); i += 2u) {
            m_dialogue.choices.push_back(
                {argumentString(call.arguments[i]), argumentInt(call.arguments[i + 1u])});
        }
        return result;
    }
    if (command == "goodbye") {
        m_dialogue.goodbye = true;
        return result;
    }
    if (command == "clearinfoactor") {
        if (!m_dialogue.currentInfo.valid()) {
            result.error = "ClearInfoActor requires an active topic response";
            return result;
        }
        m_topicResponseActors.erase(m_dialogue.currentInfo);
        return result;
    }
    if (command == "startscript") {
        if (call.arguments.empty()) { result.error = "StartScript requires a script id"; return result; }
        std::string error;
        if (m_scripts.start(argumentString(call.arguments[0]), call.owner, error, true) == 0u) result.error = error;
        return result;
    }
    if (command == "stopscript") {
        if (call.arguments.empty()) { result.error = "StopScript requires a script id"; return result; }
        const std::string program = normalizeTes3Symbol(argumentString(call.arguments[0]));
        for (auto& [id, thread] : m_scripts.threadsForRestore()) {
            (void)id;
            if (thread.program == program && (thread.state == Tes3ThreadState::Running ||
                thread.state == Tes3ThreadState::Suspended)) {
                thread.state = Tes3ThreadState::Completed;
                thread.repeat = false;
            }
        }
        return result;
    }
    if (command == "scriptrunning") {
        const std::string program = call.arguments.empty() ? std::string{} :
            normalizeTes3Symbol(argumentString(call.arguments[0]));
        const bool running = std::any_of(m_scripts.threads().begin(), m_scripts.threads().end(),
            [&](const auto& entry) {
                return entry.second.program == program &&
                    (entry.second.state == Tes3ThreadState::Running ||
                     entry.second.state == Tes3ThreadState::Suspended);
            });
        result.value = Tes3Value::fromNumber(running ? 1.0 : 0.0);
        return result;
    }
    if (command == "random") {
        const std::uint32_t maximum = call.arguments.empty()
            ? 100u : static_cast<std::uint32_t>(std::max(1, argumentInt(call.arguments[0])));
        const std::uint64_t bits = core::mix64(call.tick ^
            static_cast<std::uint64_t>(ObjectIdHash{}(call.owner)));
        result.value = Tes3Value::fromNumber(static_cast<double>(bits % maximum));
        return result;
    }
    if (command == "getdisposition" || command == "setdisposition" ||
        command == "moddisposition") {
        if (m_externalNative) return m_externalNative(call);
        if (!call.target.empty() && !sameText(call.target, m_dialogue.actor.id)) {
            result.error = "disposition target requires a world resolver";
            return result;
        }
        if (command == "getdisposition") {
            result.value = Tes3Value::fromNumber(m_dialogue.actor.disposition);
        } else {
            const float value = call.arguments.empty() ? 0.0f :
                static_cast<float>(call.arguments[0].number);
            m_dialogue.actor.disposition = std::clamp(command == "setdisposition"
                ? value : m_dialogue.actor.disposition + value, 0.0f, 100.0f);
            m_referenceOverrides[m_dialogue.actor.object].locals["stat:disposition"] =
                Tes3Value::fromNumber(m_dialogue.actor.disposition);
        }
        return result;
    }
    const auto factionName = [&]() {
        if (!call.arguments.empty() && call.arguments.back().type == Tes3ValueType::String &&
            (command == "getpcrank" || command == "pcjoinfaction" ||
             command == "pcraiserank" || command == "pclowerrank" ||
             command == "pcexpell" || command == "pcexpelled" ||
             command == "pcclearexpelled")) {
            return normalizeTes3Symbol(call.arguments.back().string);
        }
        return normalizeTes3Symbol(m_dialogue.actor.faction);
    };
    if (command == "getpcrank" || command == "pcjoinfaction" ||
        command == "pcraiserank" || command == "pclowerrank" ||
        command == "pcexpell" || command == "pcexpelled" ||
        command == "pcclearexpelled") {
        const std::string faction = factionName();
        auto found = m_playerState.factionRanks.find(faction);
        if (command == "getpcrank") {
            result.value = Tes3Value::fromNumber(found == m_playerState.factionRanks.end()
                ? -1.0 : found->second);
        } else if (command == "pcexpelled") {
            result.value = Tes3Value::fromNumber(m_playerState.numericFilters[
                "expelled:" + faction] != 0.0 ? 1.0 : 0.0);
        } else if (command == "pcexpell") {
            m_playerState.numericFilters["expelled:" + faction] = 1.0;
        } else if (command == "pcclearexpelled") {
            m_playerState.numericFilters["expelled:" + faction] = 0.0;
        } else if (command == "pcjoinfaction") {
            m_playerState.factionRanks.try_emplace(faction, 0);
            m_playerState.numericFilters["expelled:" + faction] = 0.0;
        } else {
            std::int8_t& rank = m_playerState.factionRanks[faction];
            rank = static_cast<std::int8_t>(std::clamp<int>(rank +
                (command == "pcraiserank" ? 1 : -1), 0, 9));
        }
        m_dialogue.player = m_playerState;
        return result;
    }
    if (command == "modfactionreaction") {
        if (call.arguments.size() < 3u) {
            result.error = "ModFactionReaction requires two factions and a value";
            return result;
        }
        const std::string source = normalizeTes3Symbol(argumentString(call.arguments[0]));
        const std::string destination = normalizeTes3Symbol(argumentString(call.arguments[1]));
        m_playerState.numericFilters["faction_reaction:" + source + ":" + destination] +=
            call.arguments[2].number;
        if (m_dialogue.active) m_dialogue.player = m_playerState;
        return result;
    }
    if (command == "getreputation" || command == "setreputation" ||
        command == "modreputation") {
        if (m_externalNative) return m_externalNative(call);
        const bool playerTarget = sameText(call.target, "player") ||
            (call.target.empty() && call.owner == m_playerState.object);
        if (!playerTarget && ((!call.target.empty() && !sameText(call.target, m_dialogue.actor.id)) ||
                (call.target.empty() && call.owner != m_dialogue.actor.object))) {
            result.error = "reputation target requires a world resolver"; return result;
        }
        double reputation = playerTarget ? m_playerState.numericFilters["reputation"] :
            dialogueFunctionValue(3, m_dialogue.actor, m_playerState).value_or(0);
        if (command == "getreputation") result.value = Tes3Value::fromNumber(reputation);
        else {
            if (call.arguments.empty()) { result.error = command + " requires a value"; return result; }
            reputation = command == "setreputation" ? call.arguments[0].number : reputation + call.arguments[0].number;
            if (playerTarget) {
                m_playerState.numericFilters["reputation"] = reputation;
                m_dialogue.player = m_playerState;
            } else m_referenceOverrides[m_dialogue.actor.object].locals["stat:reputation"] = Tes3Value::fromNumber(reputation);
        }
        return result;
    }
    if (command == "modpcfacrep" || command == "setpcfacrep") {
        if (call.arguments.empty()) { result.error = command + " requires a value"; return result; }
        const std::string faction = normalizeTes3Symbol(m_dialogue.actor.faction);
        double& reputation = m_playerState.numericFilters["faction_rep:" + faction];
        reputation = command == "setpcfacrep" ? call.arguments[0].number
                                                : reputation + call.arguments[0].number;
        m_dialogue.player = m_playerState;
        return result;
    }
    if (command == "getdeadcount") {
        const std::string actor = call.arguments.empty() ? std::string{} :
            normalizeTes3Symbol(argumentString(call.arguments[0]));
        result.value = Tes3Value::fromNumber(m_playerState.deathCounts[actor]);
        return result;
    }
    if (command == "menumode") {
        result.value = Tes3Value::fromNumber(m_dialogue.active ||
            (!m_dialogue.messageBoxText.empty() && !m_dialogue.choices.empty()) ||
            m_playerState.progression.selectionOpen ||
            m_playerState.numericFilters["chargen:menu"] != 0.0 ? 1.0 : 0.0);
        return result;
    }
    constexpr std::pair<std::string_view, int> characterMenus[] = {
        {"enablenamemenu", 1}, {"enableracemenu", 2},
        {"enableclassmenu", 3}, {"enablebirthmenu", 4},
        {"enablestatreviewmenu", 5}, {"showrestmenu", 6}};
    for (const auto& [name, menu] : characterMenus) {
        if (command != name) continue;
        if (m_playerState.numericFilters["chargen:menu"] != 0.0) {
            result.error = "another character menu is already open";
            return result;
        }
        m_playerState.numericFilters["chargen:menu"] = menu;
        if (menu == 6) m_playerState.progression.restBed = call.owner == m_player ? ObjectId{} : call.owner;
        if (m_dialogue.active) m_dialogue.player = m_playerState;
        return result;
    }
    constexpr std::string_view unlockedMenus[] = {
        "enablestatsmenu", "enableinventorymenu", "enablemagicmenu", "enablemapmenu"};
    if (std::ranges::find(unlockedMenus, command) != std::end(unlockedMenus)) {
        m_playerState.numericFilters["menu:" + command.substr(6u)] = 1.0;
        if (m_dialogue.active) m_dialogue.player = m_playerState;
        return result;
    }
    if (command == "cellchanged") {
        result.value = Tes3Value::fromNumber(m_playerState.numericFilters["cellchanged"]);
        return result;
    }
    if (command == "getsecondspassed") {
        result.value = Tes3Value::fromNumber(1.0 / 60.0);
        return result;
    }
    if (command == "getsquareroot") {
        const double value = call.arguments.empty() ? 0.0 : call.arguments[0].number;
        result.value = Tes3Value::fromNumber(std::sqrt(std::max(0.0, value)));
        return result;
    }
    if (command == "getcurrenttime") {
        const auto gameHour = m_scripts.globals().find("gamehour");
        result.value = Tes3Value::fromNumber(
            gameHour == m_scripts.globals().end() ? 0.0 : gameHour->second.number);
        return result;
    }
    if (command == "getcurrentweather") {
        result.value = Tes3Value::fromNumber(m_playerState.numericFilters["currentweather"]);
        return result;
    }
    if (command == "changeweather") {
        if (call.arguments.size() < 2u) {
            result.error = "ChangeWeather requires region and weather";
            return result;
        }
        const std::string region = normalizeTes3Symbol(argumentString(call.arguments[0]));
        const double weather = call.arguments[1].number;
        m_playerState.numericFilters["weather:" + region] = weather;
        m_playerState.numericFilters["currentweather"] = weather;
        if (m_dialogue.active) m_dialogue.player = m_playerState;
        return result;
    }
    if (command == "modregion") {
        if ((call.arguments.size() != 9u && call.arguments.size() != 11u) ||
            m_content == nullptr) {
            result.error = "ModRegion requires a region and eight or ten weather chances";
            return result;
        }
        const std::string region = normalizeTes3Symbol(argumentString(call.arguments[0]));
        if (!m_content->namedRecords().contains(makeTes3RecordKey("REGN", region))) {
            result.error = "ModRegion names an unknown region " + region;
            return result;
        }
        for (std::size_t index = 1u; index < call.arguments.size(); ++index) {
            const Tes3Value& chance = call.arguments[index];
            if (chance.type != Tes3ValueType::Number || !std::isfinite(chance.number) ||
                chance.number < 0.0 || chance.number > 100.0 ||
                std::floor(chance.number) != chance.number) {
                result.error = "ModRegion weather chance " + std::to_string(index - 1u) +
                    " must be an integer from 0 to 100";
                return result;
            }
        }
        m_playerState.numericFilters["regionweather:" + region + ":count"] =
            static_cast<double>(call.arguments.size() - 1u);
        for (std::size_t index = 1u; index < call.arguments.size(); ++index) {
            m_playerState.numericFilters["regionweather:" + region + ":" +
                std::to_string(index - 1u)] = call.arguments[index].number;
        }
        if (m_dialogue.active) m_dialogue.player = m_playerState;
        return result;
    }
    if (command == "getsoundplaying") {
        const std::string sound = call.arguments.empty() ? std::string{} :
            normalizeTes3Symbol(argumentString(call.arguments[0]));
        result.value = Tes3Value::fromNumber(m_activeSounds.contains(sound) ? 1.0 : 0.0);
        return result;
    }
    if (command == "say") {
        if (call.arguments.empty()) {
            result.error = "Say requires an authored voice clip";
            return result;
        }
        // Retain the voice gate across frames and saves. Until audio duration
        // metadata is exposed here, the subtitle's reading time bounds when
        // the following authored SayDone branch may advance.
        const std::string subtitle = call.arguments.size() > 1u
            ? argumentString(call.arguments[1u]) : std::string{};
        const std::uint64_t durationTicks = static_cast<std::uint64_t>(
            std::ceil(std::max(1.0, double(subtitle.size()) / 14.0) * 60.0));
        m_playerState.numericFilters["voice:end:" + call.owner.toString()] =
            static_cast<double>(call.tick + durationTicks);
        return result;
    }
    if (command == "saydone") {
        const auto found = m_playerState.numericFilters.find(
            "voice:end:" + call.owner.toString());
        result.value = Tes3Value::fromNumber(found == m_playerState.numericFilters.end() ||
            static_cast<double>(call.tick) >= found->second ? 1.0 : 0.0);
        return result;
    }
    if (command == "playsound" || command == "playsound3d" ||
        command == "playsoundvp" || command == "playsound3dvp" ||
        command == "playloopsound3d" || command == "playloopsound3dvp") {
        if (!call.arguments.empty()) {
            m_activeSounds.insert(normalizeTes3Symbol(argumentString(call.arguments[0])));
        }
        return result;
    }
    if (command == "stopsound") {
        if (!call.arguments.empty()) {
            m_activeSounds.erase(normalizeTes3Symbol(argumentString(call.arguments[0])));
        }
        return result;
    }
    if (command == "getattacked") {
        bool attacked = false;
        for (const auto& [id, thread] : m_scripts.threads()) {
            (void)id;
            if (thread.owner != call.owner) continue;
            const auto event = thread.eventVariables.find("attacked");
            if (event != thread.eventVariables.end() && event->second.truthy()) {
                attacked = true;
                break;
            }
        }
        result.value = Tes3Value::fromNumber(attacked ? 1.0 : 0.0);
        return result;
    }
    if (command == "hitonme") {
        if (call.arguments.empty()) {
            result.error = "HitOnMe requires a weapon id";
            return result;
        }
        const std::string weapon = normalizeTes3Symbol(argumentString(call.arguments[0]));
        bool hit = false;
        for (auto& [id, thread] : m_scripts.threadsForRestore()) {
            (void)id;
            if (thread.owner != call.owner) continue;
            const auto event = thread.eventVariables.find("hitonme");
            if (event == thread.eventVariables.end() ||
                normalizeTes3Symbol(event->second.string) != weapon) continue;
            hit = true;
            thread.eventVariables.erase(event);
            break;
        }
        result.value = Tes3Value::fromNumber(hit ? 1.0 : 0.0);
        return result;
    }
    if (command == "getlocal") {
        if (m_content == nullptr || call.arguments.empty()) {
            result.error = "GetLocal requires a reference and local name";
            return result;
        }
        const std::string reference = normalizeTes3Symbol(call.target);
        const std::string name = normalizeTes3Symbol(argumentString(call.arguments[0]));
        std::set<ObjectId> matches;
        for (const auto& [id, definition] : m_content->references()) {
            if (normalizeTes3Symbol(definition.baseId) == reference)
                matches.insert(id);
        }
        RecordKey serialized;
        if (parseRecordKey(call.target, serialized))
            matches.insert(ObjectId::persistent(std::move(serialized)));
        if (matches.size() != 1u) {
            result.error = "GetLocal reference " + call.target +
                (matches.empty() ? " does not resolve" : " is ambiguous");
            return result;
        }
        for (const auto& [id, thread] : m_scripts.threads()) {
            (void)id;
            if (thread.owner != *matches.begin()) continue;
            const auto local = thread.locals.find(name);
            if (local == thread.locals.end()) continue;
            result.value = local->second;
            return result;
        }
        result.error = "GetLocal " + call.target + "." + name +
            " has no running local script";
        return result;
    }
    if (command == "onactivate") {
        bool activated = false;
        for (auto& [id, thread] : m_scripts.threadsForRestore()) {
            (void)id;
            if (thread.owner != call.owner) continue;
            if (auto event = thread.eventVariables.find("onactivate");
                event != thread.eventVariables.end()) {
                activated = activated || event->second.truthy();
                event->second = Tes3Value::fromNumber(0.0);
            }
        }
        result.value = Tes3Value::fromNumber(activated ? 1.0 : 0.0);
        return result;
    }
    if (command == "getbuttonpressed") {
        result.value = Tes3Value::fromNumber(m_dialogue.choice);
        return result;
    }
    const std::map<std::string, std::string> playerQueries = {
        {"getpcjumping", "jumping"}, {"getpcrunning", "running"},
        {"getpcsleep", "sleep"}, {"getpcsneaking", "sneaking"},
        {"getpctraveling", "traveling"}, {"getspellreadied", "spellreadied"},
        {"getweapondrawn", "weapondrawn"}, {"getwerewolfkills", "werewolfkills"}};
    if (const auto query = playerQueries.find(command); query != playerQueries.end()) {
        result.value = Tes3Value::fromNumber(m_playerState.numericFilters[query->second]);
        return result;
    }
    if (command == "messagebox") {
        m_dialogue.messageBoxText = call.arguments.empty()
            ? std::string{} : argumentString(call.arguments.front());
        m_dialogue.choices.clear();
        for (std::size_t index = 1u; index < call.arguments.size(); ++index) {
            m_dialogue.choices.push_back(
                {argumentString(call.arguments[index]), static_cast<std::int32_t>(index - 1u)});
        }
        if (!m_dialogue.choices.empty()) {
            result.suspend = true;
            result.suspensionReason = "messagebox";
        }
        return result;
    }
    if (command == "showmap") {
        if (call.arguments.empty()) { result.error = "ShowMap requires a marker id"; return result; }
        m_playerState.numericFilters["map:" + normalizeTes3Symbol(
            argumentString(call.arguments[0]))] = 1.0;
        m_dialogue.player = m_playerState;
        return result;
    }
    if (command == "setpccrimelevel" || command == "modpccrimelevel") {
        if (call.arguments.empty()) { result.error = command + " requires a value"; return result; }
        double& crime = m_playerState.numericFilters["crimelevel"];
        crime = command == "setpccrimelevel" ? call.arguments[0].number
                                               : crime + call.arguments[0].number;
        crime = std::max(0.0, crime);
        m_dialogue.player = m_playerState;
        return result;
    }
    if (command == "getpccrimelevel" || command == "getpcinjail") {
        const std::string key = command == "getpcinjail" ? "injail" : "crimelevel";
        result.value = Tes3Value::fromNumber(m_playerState.numericFilters[key]);
        return result;
    }
    if (command == "payfine" || command == "payfinethief") {
        m_playerState.numericFilters["crimelevel"] = 0.0;
        m_dialogue.player = m_playerState;
        return result;
    }
    if (command == "gotojail" || command == "wakeuppc") {
        if (command == "gotojail") m_playerState.numericFilters["injail"] = 1.0;
        else m_playerState.numericFilters["sleep"] = 0.0;
        if (m_dialogue.active) m_dialogue.player = m_playerState;
        return result;
    }
    constexpr std::string_view playerControlCommands[] = {
        "disableplayercontrols", "enableplayercontrols",
        "disableplayerfighting", "enableplayerfighting",
        "disableplayerjumping", "enableplayerjumping",
        "disableplayermagic", "enableplayermagic",
        "disableplayerviewswitch", "enableplayerviewswitch",
        "disableteleporting", "enableteleporting",
        "disablelevitation", "enablelevitation",
        "disablevanitymode", "enablevanitymode", "enablerest"};
    if (std::ranges::find(playerControlCommands, command) != std::end(playerControlCommands)) {
        const bool enabled = command.starts_with("enable");
        std::string control = command.substr(enabled ? 6u : 7u);
        if (control.empty()) control = command;
        m_playerState.numericFilters["control:" + control] = enabled ? 1.0 : 0.0;
        if (m_dialogue.active) m_dialogue.player = m_playerState;
        return result;
    }
    const Tes3NativeDefinition* definition = m_nativeRegistry.find(command);
    if (definition != nullptr && definition->disposition == Tes3NativeDisposition::PresentationOnly) {
        return result;
    }
    if (m_externalNative) return m_externalNative(call);
    result.error = "unhandled gameplay MWScript native " + command;
    return result;
}

std::string Tes3Runtime::resultProgramId(const RecordKey& info) const {
    return "dialogue_result:" + info.toString();
}

void Tes3Runtime::endDialogue() {
    finishPendingResult(false);
    if (m_dialogueEndHook) m_dialogueEndHook();
    m_dialogue = {};
}

void Tes3Runtime::recordActorDeath(ObjectId actor, std::string_view baseId, bool dead) {
    if (actor == m_player || baseId.empty()) return;
    auto& counted = m_referenceOverrides[actor].locals["actor:deathcounted"];
    if (dead && !counted.truthy()) {
        ++m_playerState.deathCounts[normalizeTes3Symbol(baseId)];
        if (m_dialogue.active) m_dialogue.player.deathCounts = m_playerState.deathCounts;
    }
    counted = Tes3Value::fromNumber(dead ? 1.0 : 0.0);
}

void Tes3Runtime::dispatchGameplayEvent(
    std::string eventName, ObjectId target, Tes3Value value) {
    eventName = normalizeTes3Symbol(eventName);
    if (eventName.empty()) return;
    if (!target.valid() || target == m_player) {
        m_playerState.numericFilters[eventName] = value.number;
        if (m_dialogue.active) m_dialogue.player = m_playerState;
    }
    for (auto& [id, thread] : m_scripts.threadsForRestore()) {
        (void)id;
        if (target.valid() && thread.owner != target) continue;
        // MWScript declares OnPCEquip/OnActivate as ordinary short locals.
        // Delivering the event only to the fallback map leaves that local at
        // zero, so the script never observes the gameplay event.
        if (auto local = thread.locals.find(eventName);
            eventName != "onactivate" && local != thread.locals.end()) {
            local->second = value;
        } else {
            thread.eventVariables.insert_or_assign(eventName, value);
        }
        if (thread.state == Tes3ThreadState::Suspended &&
            thread.suspensionReason == "event:" + eventName) {
            thread.state = Tes3ThreadState::Running;
            thread.suspensionReason.clear();
        }
    }
}

void Tes3Runtime::dispatchScriptEvent(
    std::uint64_t threadId, std::string eventName, Tes3Value value) {
    eventName = normalizeTes3Symbol(eventName);
    if (eventName.empty()) return;
    const auto found = m_scripts.threadsForRestore().find(threadId);
    if (found == m_scripts.threadsForRestore().end()) return;
    Tes3ScriptThread& thread = found->second;
    if (auto local = thread.locals.find(eventName);
        eventName != "onactivate" && local != thread.locals.end())
        local->second = value;
    else thread.eventVariables.insert_or_assign(eventName, value);
    if (thread.state == Tes3ThreadState::Suspended &&
        thread.suspensionReason == "event:" + eventName) {
        thread.state = Tes3ThreadState::Running;
        thread.suspensionReason.clear();
    }
}

void Tes3Runtime::clear() {
    finishPendingResult(false);
    m_content.reset();
    m_player = {};
    m_playerState = {};
    m_journal.clear();
    m_scripts.clear();
    m_nativeRegistry = {};
    m_scriptCheck = {};
    m_dialogue = {};
    m_knownTopics.clear();
    m_topicResponseActors.clear();
    m_referenceOverrides.clear();
    m_activeSpells.clear();
    m_activeSounds.clear();
    m_externalNative = {};
    m_dialogueEndHook = {};
    m_currentTick = 0u;
}

}  // namespace odai::bethesda
