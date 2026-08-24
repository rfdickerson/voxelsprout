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
    if (std::find(state.visitedEntries.begin(), state.visitedEntries.end(), info->record) !=
        state.visitedEntries.end()) {
        outError.clear();
        return true;
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
    const auto validateCastEffects = [&](const Tes3ScriptProgram& program) {
        for (const Tes3Instruction& instruction : program.instructions) {
            if (instruction.op != Tes3OpCode::Call || instruction.command != "cast" ||
                instruction.arguments.empty()) continue;
            std::string spellId = instruction.arguments.front();
            spellId.erase(std::remove(spellId.begin(), spellId.end(), '"'), spellId.end());
            const Tes3SpellDefinition* spell = m_content->findSpell(spellId);
            if (spell == nullptr) {
                m_scriptCheck.unsupportedCommands.insert(
                    "cast:unresolved:" + normalizeTes3Symbol(spellId));
                continue;
            }
            for (const Tes3SpellEffect& effect : spell->effects) {
                if (effect.effectId != 79 && effect.effectId != 83) {
                    m_scriptCheck.unsupportedCommands.insert(
                        "cast:effect:" + std::to_string(effect.effectId));
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
        validateCastEffects(compiled.program);
        for (const std::string& command : compiled.program.commands) {
            if (m_scripts.globals().contains(command)) continue;
            ++m_scriptCheck.commandUse[command];
            if (m_nativeRegistry.find(command) == nullptr) m_scriptCheck.unsupportedCommands.insert(command);
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
            validateCastEffects(compiled.program);
            for (const std::string& command : compiled.program.commands) {
                if (m_scripts.globals().contains(command)) continue;
                ++m_scriptCheck.commandUse[command];
                if (m_nativeRegistry.find(command) == nullptr) m_scriptCheck.unsupportedCommands.insert(command);
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
    return m_scripts.step(tick, instructionBudget,
        [this](const Tes3NativeCall& call) { return executeNative(call); });
}

bool Tes3Runtime::matches(
    const Tes3DialogueInfo& info, const Tes3DialogueActorState& actor,
    const Tes3DialoguePlayerState& player, bool strict) const {
    if (!info.actor.empty() && !sameText(info.actor, actor.id)) return false;
    if (!info.race.empty() && !sameText(info.race, actor.race)) return false;
    if (!info.actorClass.empty() && !sameText(info.actorClass, actor.actorClass)) return false;
    if (info.factionless && !actor.faction.empty()) return false;
    if (!info.faction.empty() && !info.factionless && !sameText(info.faction, actor.faction)) return false;
    if (!info.cell.empty() && !sameText(info.cell, actor.cell)) return false;
    if (info.rank >= 0 && actor.rank != info.rank) return false;
    if (info.gender >= 0 && actor.gender != info.gender) return false;
    if (!info.playerFaction.empty()) {
        const auto rank = player.factionRanks.find(normalizeTes3Symbol(info.playerFaction));
        if (rank == player.factionRanks.end()) return false;
        if (info.playerRank >= 0 && rank->second < info.playerRank) return false;
    }
    for (const Tes3DialogueCondition& condition : info.conditions) {
        if (!condition.valid) {
            if (strict) return false;
            continue;
        }
        double actual = 0.0;
        const std::string variable = normalizeTes3Symbol(condition.variable);
        const int function = static_cast<int>(condition.function);
        if (condition.function == Tes3ConditionFunction::Global) {
            const auto found = m_scripts.globals().find(variable);
            actual = found == m_scripts.globals().end() ? 0.0 : found->second.number;
        } else if (condition.function == Tes3ConditionFunction::Local ||
                   condition.function == Tes3ConditionFunction::NotLocal) {
            const auto found = actor.locals.find(variable);
            actual = found == actor.locals.end() ? 0.0 : found->second;
            if (condition.function == Tes3ConditionFunction::NotLocal) actual = actual == 0.0 ? 1.0 : 0.0;
        } else if (condition.function == Tes3ConditionFunction::Journal) {
            actual = m_journal.index(condition.variable);
        } else if (condition.function == Tes3ConditionFunction::Item) {
            const RecordKey wanted = makeTes3RecordKey("REFR", condition.variable);
            for (const auto& [item, count] : player.inventory) {
                if (item.textId == wanted.textId) actual += count;
            }
        } else if (condition.function == Tes3ConditionFunction::Dead) {
            const auto found = player.deathCounts.find(variable);
            actual = found == player.deathCounts.end() ? 0.0 : found->second;
        } else if (condition.function == Tes3ConditionFunction::NotId) {
            actual = sameText(actor.id, condition.variable) ? 0.0 : 1.0;
        } else if (condition.function == Tes3ConditionFunction::NotFaction) {
            actual = sameText(actor.faction, condition.variable) ? 0.0 : 1.0;
        } else if (condition.function == Tes3ConditionFunction::NotClass) {
            actual = sameText(actor.actorClass, condition.variable) ? 0.0 : 1.0;
        } else if (condition.function == Tes3ConditionFunction::NotRace) {
            actual = sameText(actor.race, condition.variable) ? 0.0 : 1.0;
        } else if (condition.function == Tes3ConditionFunction::NotCell) {
            actual = sameText(actor.cell, condition.variable) ? 0.0 : 1.0;
        } else if (function == 50) {  // Choice
            actual = m_dialogue.choice;
        } else {
            const auto found = player.numericFilters.find("function:" + std::to_string(function));
            if (found == player.numericFilters.end()) {
                if (strict) return false;
                continue;
            }
            actual = found->second;
        }
        if (!compare(actual, condition.comparison, conditionValue(condition))) return false;
    }
    return true;
}

const Tes3DialogueInfo* Tes3Runtime::selectInfo(
    const Tes3DialogueDefinition& topic, bool strict) const {
    for (const Tes3DialogueInfo& info : topic.infos) {
        if (m_dialogue.exhaustedInfos.contains(info.record)) continue;
        if (matches(info, m_dialogue.actor, m_dialogue.player, strict)) return &info;
    }
    return nullptr;
}

Tes3DialogueResponse Tes3Runtime::startDialogue(
    Tes3DialogueActorState actor, Tes3DialoguePlayerState player, bool strict) {
    Tes3DialogueResponse response;
    if (m_content == nullptr || !actor.object.valid() || !player.object.valid()) {
        response.diagnostics.push_back("dialogue requires configured TES3 content and participants");
        return response;
    }
    m_dialogue = {};
    m_dialogue.active = true;
    m_dialogue.actor = std::move(actor);
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
    const Tes3DialogueInfo* info = selectInfo(topic->second, strict);
    if (info == nullptr) {
        response.diagnostics.push_back("topic has no unexhausted matching response");
        return response;
    }
    m_dialogue.choice = -1;
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
    if (!m_dialogue.active) {
        m_dialogue.choice = value;
        m_dialogue.choices.clear();
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
    m_dialogue.choice = value;
    m_dialogue.choices.clear();
    const auto topic = m_content->dialogues().find(m_dialogue.currentTopic);
    if (topic == m_content->dialogues().end()) {
        response.diagnostics.push_back("current dialogue topic disappeared");
        return response;
    }
    const Tes3DialogueInfo* info = selectInfo(topic->second, strict);
    if (info == nullptr) {
        response.diagnostics.push_back("choice has no matching response");
        return response;
    }
    return activateInfo(topic->second, *info, strict);
}

Tes3DialogueResponse Tes3Runtime::activateInfo(
    const Tes3DialogueDefinition& topic, const Tes3DialogueInfo& info, bool strict) {
    (void)strict;
    Tes3DialogueResponse response;
    response.accepted = true;
    response.topic = topic.record;
    response.info = info.record;
    response.text = info.response;
    m_dialogue.currentTopic = topic.record;
    m_dialogue.currentInfo = info.record;
    m_dialogue.exhaustedInfos.insert(info.record);
    m_dialogue.choices.clear();
    m_dialogue.goodbye = false;
    discoverTopics(info.response, response.discoveredTopics);
    if (!info.resultScript.empty()) {
        std::string error;
        const std::uint64_t thread = m_scripts.start(resultProgramId(info.record),
            m_dialogue.actor.object, error);
        if (thread == 0u) {
            response.diagnostics.push_back(error);
        } else {
            const Tes3VmStepResult script = step(m_currentTick, 10000u);
            response.diagnostics.insert(response.diagnostics.end(),
                script.diagnostics.begin(), script.diagnostics.end());
        }
    }
    response.choices = m_dialogue.choices;
    response.goodbye = m_dialogue.goodbye;
    if (response.goodbye) m_dialogue.active = false;
    return response;
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
    if (command == "startscript") {
        if (call.arguments.empty()) { result.error = "StartScript requires a script id"; return result; }
        std::string error;
        if (m_scripts.start(argumentString(call.arguments[0]), call.owner, error) == 0u) result.error = error;
        return result;
    }
    if (command == "stopscript") {
        if (call.arguments.empty()) { result.error = "StopScript requires a script id"; return result; }
        const std::string program = normalizeTes3Symbol(argumentString(call.arguments[0]));
        for (auto& [id, thread] : m_scripts.threadsForRestore()) {
            (void)id;
            if (thread.program == program && (thread.state == Tes3ThreadState::Running ||
                thread.state == Tes3ThreadState::Suspended)) thread.state = Tes3ThreadState::Completed;
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
        if (command == "getdisposition") {
            result.value = Tes3Value::fromNumber(m_dialogue.actor.disposition);
        } else {
            const float value = call.arguments.empty() ? 0.0f :
                static_cast<float>(call.arguments[0].number);
            m_dialogue.actor.disposition = std::clamp(command == "setdisposition"
                ? value : m_dialogue.actor.disposition + value, 0.0f, 100.0f);
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
        double& reputation = m_playerState.numericFilters["reputation"];
        if (command == "getreputation") result.value = Tes3Value::fromNumber(reputation);
        else if (!call.arguments.empty()) reputation = command == "setreputation"
            ? call.arguments[0].number : reputation + call.arguments[0].number;
        m_dialogue.player = m_playerState;
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
        result.value = Tes3Value::fromNumber(m_dialogue.active ? 1.0 : 0.0);
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
    if (command == "getsoundplaying") {
        const std::string sound = call.arguments.empty() ? std::string{} :
            normalizeTes3Symbol(argumentString(call.arguments[0]));
        result.value = Tes3Value::fromNumber(m_activeSounds.contains(sound) ? 1.0 : 0.0);
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
    if (command == "onactivate") {
        bool activated = false;
        for (const auto& [id, thread] : m_scripts.threads()) {
            (void)id;
            if (thread.owner != call.owner) continue;
            const auto event = thread.eventVariables.find("onactivate");
            if (event != thread.eventVariables.end() && event->second.truthy()) {
                activated = true;
                break;
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
    if (m_dialogue.player.object.valid()) m_playerState = m_dialogue.player;
    m_dialogue = {};
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
        thread.eventVariables.insert_or_assign(eventName, value);
        if (thread.state == Tes3ThreadState::Suspended &&
            thread.suspensionReason == "event:" + eventName) {
            thread.state = Tes3ThreadState::Running;
            thread.suspensionReason.clear();
        }
    }
}

void Tes3Runtime::clear() {
    m_content.reset();
    m_player = {};
    m_playerState = {};
    m_journal.clear();
    m_scripts.clear();
    m_nativeRegistry = {};
    m_scriptCheck = {};
    m_dialogue = {};
    m_knownTopics.clear();
    m_referenceOverrides.clear();
    m_activeSpells.clear();
    m_activeSounds.clear();
    m_externalNative = {};
    m_currentTick = 0u;
}

}  // namespace odai::bethesda
