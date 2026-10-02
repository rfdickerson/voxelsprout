#include "bethesda/bethesda_session.h"
#include "bethesda/session_text.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>

namespace odai::bethesda {

void BethesdaSession::registerSkyrimNatives() {
    registerCoreNatives();
    registerWorldAndLocationNatives();
    registerQuestSceneAndAliasNatives();
    registerObjectReferenceNatives();
    registerActorNatives();
}

void BethesdaSession::registerCoreNatives() {
    m_papyrus.registerClassParent("ObjectReference", "Form");
    m_papyrus.registerClassParent("Actor", "ObjectReference");
    m_papyrus.registerClassParent("Quest", "Form");
    m_papyrus.registerClassParent("TopicInfo", "Form");
    m_papyrus.registerClassParent("ReferenceAlias", "Form");
    m_papyrus.registerClassParent("LocationAlias", "Form");
    m_papyrus.registerClassParent("Scene", "Form");
    m_papyrus.registerClassParent("Location", "Form");
    m_papyrus.registerClassParent("Keyword", "Form");
    m_papyrus.registerClassParent("GlobalVariable", "Form");
    m_papyrus.registerClassParent("Weather", "Form");
    m_papyrus.registerContextNative("TopicInfo.GetOwningQuest",
        [this](const PapyrusNativeContext& context,
            std::span<const PapyrusValue> arguments, BethesdaWorld&) {
            NativeCallResult result;
            if (!arguments.empty() ||
                context.self.kind != ObjectIdKind::PersistentReference) {
                result.error = "TopicInfo.GetOwningQuest expects an INFO object and no arguments";
                return result;
            }
            const auto info = m_dialogueInfos.find(context.self.reference);
            if (info == m_dialogueInfos.end()) {
                result.error = "unknown TopicInfo object " + context.self.toString();
                return result;
            }
            result.value = PapyrusValue::fromObject(
                ObjectId::persistent(info->second.quest));
            return result;
        });
    const auto noStateHandler = [](
        std::span<const PapyrusValue>, std::uint64_t, BethesdaWorld&) {
        return NativeCallResult{};
    };
    m_papyrus.registerNative("Form.OnBeginState", noStateHandler);
    m_papyrus.registerNative("Form.OnEndState", noStateHandler);
    const auto registerUpdate = [this](bool repeating, bool gameTime) {
        return [this, repeating, gameTime](const PapyrusNativeContext& context,
            std::span<const PapyrusValue> arguments, BethesdaWorld&) {
            NativeCallResult result;
            if (arguments.size() != 1u ||
                (arguments[0].type != PapyrusValueType::Float &&
                 arguments[0].type != PapyrusValueType::Integer)) {
                result.error = gameTime
                    ? "game-time update registration expects one numeric hours value"
                    : "update registration expects one numeric seconds value";
                return result;
            }
            double interval = arguments[0].type == PapyrusValueType::Float
                ? arguments[0].real : static_cast<double>(arguments[0].integer);
            if (gameTime) {
                // Skyrim's default timescale advances twenty game minutes per
                // real minute. This remains deterministic until mutable
                // timescale and world-time state become part of the session.
                constexpr double kSkyrimDefaultTimescale = 20.0;
                interval = interval * 3600.0 / kSkyrimDefaultTimescale;
            }
            if (!m_papyrus.registerForUpdate(
                    context.self, context.scriptClass, interval,
                    context.currentTick, repeating, result.error,
                    gameTime ? "OnUpdateGameTime" : "OnUpdate")) {
                return result;
            }
            return result;
        };
    };
    m_papyrus.registerContextNative("Form.RegisterForUpdate", registerUpdate(true, false));
    m_papyrus.registerContextNative("Form.RegisterForSingleUpdate", registerUpdate(false, false));
    m_papyrus.registerContextNative(
        "Form.RegisterForUpdateGameTime", registerUpdate(true, true));
    m_papyrus.registerContextNative(
        "Form.RegisterForSingleUpdateGameTime", registerUpdate(false, true));
    m_papyrus.registerContextNative("Form.UnregisterForUpdate",
        [this](const PapyrusNativeContext& context,
            std::span<const PapyrusValue> arguments, BethesdaWorld&) {
            NativeCallResult result;
            if (!arguments.empty()) {
                result.error = "UnregisterForUpdate expects no arguments";
                return result;
            }
            m_papyrus.unregisterForUpdate(
                context.self, context.scriptClass, "OnUpdate");
            return result;
        });
    m_papyrus.registerContextNative("Form.UnregisterForUpdateGameTime",
        [this](const PapyrusNativeContext& context,
            std::span<const PapyrusValue> arguments, BethesdaWorld&) {
            NativeCallResult result;
            if (!arguments.empty()) {
                result.error = "UnregisterForUpdateGameTime expects no arguments";
                return result;
            }
            m_papyrus.unregisterForUpdate(
                context.self, context.scriptClass, "OnUpdateGameTime");
            return result;
        });
}

void BethesdaSession::registerWorldAndLocationNatives() {
    m_papyrus.registerNative("Game.GetPlayer",
        [](std::span<const PapyrusValue> arguments, std::uint64_t, BethesdaWorld&) {
            NativeCallResult result;
            if (!arguments.empty()) {
                result.error = "Game.GetPlayer expects no arguments";
                return result;
            }
            result.value = PapyrusValue::fromObject(
                ObjectId::persistent(makeRecordKey("Skyrim.esm", 0x14u)));
            return result;
        });
    m_papyrus.registerNative("Game.GetForm",
        [this](std::span<const PapyrusValue> arguments, std::uint64_t, BethesdaWorld&) {
            NativeCallResult result;
            if (arguments.size() != 1u || arguments[0].type != PapyrusValueType::Integer ||
                arguments[0].integer < 0 ||
                arguments[0].integer > std::numeric_limits<std::uint32_t>::max()) {
                result.error = "Game.GetForm expects one unsigned 32-bit form ID";
                return result;
            }
            if (arguments[0].integer == 0) return result;
            if (!m_resolvedFormResolver) {
                result.error = "Game.GetForm has no active load-order resolver";
                return result;
            }
            const std::optional<ObjectId> object = m_resolvedFormResolver(
                static_cast<std::uint32_t>(arguments[0].integer));
            if (object.has_value()) result.value = PapyrusValue::fromObject(*object);
            return result;
        });
    m_papyrus.registerNative("Game.EnablePlayerControls",
        [](std::span<const PapyrusValue> arguments, std::uint64_t, BethesdaWorld&) {
            NativeCallResult result;
            if (arguments.size() != 9u ||
                !std::all_of(arguments.begin(), arguments.begin() + 8,
                    [](const PapyrusValue& value) {
                        return value.type == PapyrusValueType::Boolean;
                    }) || arguments[8].type != PapyrusValueType::Integer) {
                result.error = "Game.EnablePlayerControls expects eight booleans and one integer";
            }
            // ODAI input contexts are independently owned and already enabled
            // at this post-Helgen bootstrap boundary.
            return result;
        });
    for (const int mode : {0, 1, 2}) {
        const bool remove = mode == 1;
        const bool crossFade = mode == 2;
        m_papyrus.registerNative(crossFade ? "ImageSpaceModifier.ApplyCrossFade" : remove ? "ImageSpaceModifier.Remove" : "ImageSpaceModifier.Apply",
            [this, remove, crossFade](std::span<const PapyrusValue> arguments, std::uint64_t, BethesdaWorld&) {
                NativeCallResult result;
                if (arguments.empty() || arguments.size() > (remove ? 1u : 2u) ||
                    arguments[0].type != PapyrusValueType::Object ||
                    arguments[0].object.kind != ObjectIdKind::PersistentReference) {
                    result.error = "ImageSpaceModifier expects a record receiver and optional strength";
                    return result;
                }
                float strength = 1.0f;
                if (arguments.size() == 2u) {
                    if (arguments[1].type != PapyrusValueType::Float &&
                        arguments[1].type != PapyrusValueType::Integer) {
                        result.error = "ImageSpaceModifier strength must be numeric";
                        return result;
                    }
                    strength = static_cast<float>(arguments[1].type == PapyrusValueType::Float
                        ? arguments[1].real : arguments[1].integer);
                    if (!std::isfinite(strength)) {result.error="nonfinite image-space strength";return result;}
                }
                m_imageSpaceCommands.push_back({arguments[0].object.reference,
                    crossFade ? 1.0f : std::clamp(strength, 0.0f, 1.0f), remove, crossFade, std::max(strength, 0.0f)});
                return result;
            });
    }
    m_papyrus.registerNative("ImageSpaceModifier.RemoveCrossFade",
        [this](std::span<const PapyrusValue> arguments, std::uint64_t, BethesdaWorld&) {
            NativeCallResult result;
            float duration = 1;
            if (arguments.size() > 1 || (!arguments.empty() &&
                arguments[0].type != PapyrusValueType::Float && arguments[0].type != PapyrusValueType::Integer)) {
                result.error = "RemoveCrossFade expects an optional duration"; return result;
            }
            if (!arguments.empty()) duration = float(arguments[0].type == PapyrusValueType::Float ? arguments[0].real : arguments[0].integer);
            if (!std::isfinite(duration)) { result.error = "nonfinite cross-fade duration"; return result; }
            m_imageSpaceCommands.push_back({{}, 1, true, true, std::max(duration, 0.f)});
            return result;
        });
    m_papyrus.registerNative("Weather.ForceActive",
        [this](std::span<const PapyrusValue> arguments, std::uint64_t, BethesdaWorld&) {
            NativeCallResult result;
            if ((arguments.size() != 1u && arguments.size() != 2u) ||
                arguments[0].type != PapyrusValueType::Object ||
                arguments[0].object.kind != ObjectIdKind::PersistentReference ||
                (arguments.size() == 2u &&
                 arguments[1].type != PapyrusValueType::Boolean)) {
                result.error = "Weather.ForceActive expects weather and optional override flag";
                return result;
            }
            m_forcedWeather = arguments[0].object.reference;
            return result;
        });
    m_papyrus.registerNative("Debug.OpenUserLog",
        [this](std::span<const PapyrusValue> arguments, std::uint64_t, BethesdaWorld&) {
            NativeCallResult result;
            if (arguments.size() != 1u || arguments[0].type != PapyrusValueType::String ||
                arguments[0].string.empty()) {
                result.error = "Debug.OpenUserLog expects a non-empty log name";
                return result;
            }
            if (std::find(m_scriptDebugLogs.begin(), m_scriptDebugLogs.end(),
                    arguments[0].string) == m_scriptDebugLogs.end()) {
                m_scriptDebugLogs.push_back(arguments[0].string);
            }
            // This is deliberately an in-memory diagnostic channel. Papyrus
            // never receives filesystem access from the Skyrim API surface.
            result.value = PapyrusValue::fromBoolean(true);
            return result;
        });
    m_papyrus.registerNative("Debug.Trace",
        [this](std::span<const PapyrusValue> arguments, std::uint64_t, BethesdaWorld&) {
            NativeCallResult result;
            if (arguments.empty() || arguments.size() > 2u ||
                arguments[0].type != PapyrusValueType::String ||
                (arguments.size() == 2u &&
                 arguments[1].type != PapyrusValueType::Integer)) {
                result.error = "Debug.Trace expects a message and optional severity";
                return result;
            }
            const std::string prefix = arguments.size() == 2u
                ? "trace[" + std::to_string(arguments[1].integer) + "]: "
                : "trace: ";
            m_scriptDebugLogs.push_back(prefix + arguments[0].string);
            return result;
        });
    m_papyrus.registerNative("Game.AddAchievement",
        [this](std::span<const PapyrusValue> arguments, std::uint64_t, BethesdaWorld&) {
            NativeCallResult result;
            if (arguments.size() != 1u || arguments[0].type != PapyrusValueType::Integer) {
                result.error = "Game.AddAchievement expects an integer achievement ID";
                return result;
            }
            m_statistics["achievement:" + std::to_string(arguments[0].integer)] = 1;
            return result;
        });
    m_papyrus.registerNative("AchievementsScript.IncSideQuests",
        [this](std::span<const PapyrusValue> arguments, std::uint64_t, BethesdaWorld&) {
            NativeCallResult result;
            if (arguments.size() != 1u || arguments[0].type != PapyrusValueType::Object) {
                result.error = "AchievementsScript.IncSideQuests expects its script object";
                return result;
            }
            ++m_statistics["side_quests_completed"];
            return result;
        });
    m_papyrus.registerNative("Utility.Wait",
        [](std::span<const PapyrusValue> arguments, std::uint64_t tick, BethesdaWorld&) {
            NativeCallResult result;
            if (arguments.size() != 1u ||
                (arguments[0].type != PapyrusValueType::Float &&
                 arguments[0].type != PapyrusValueType::Integer)) {
                result.error = "Utility.Wait expects one numeric seconds value";
                return result;
            }
            const double seconds = arguments[0].type == PapyrusValueType::Float
                ? arguments[0].real : static_cast<double>(arguments[0].integer);
            if (!std::isfinite(seconds) || seconds < 0.0) {
                result.error = "Utility.Wait duration is invalid";
                return result;
            }
            const double ticks = std::ceil(seconds * 60.0);
            result.completed = false;
            result.resumeTick = tick + static_cast<std::uint64_t>(std::min<double>(
                ticks, static_cast<double>(std::numeric_limits<std::uint32_t>::max())));
            return result;
        });
    m_papyrus.registerNative("Utility.RandomInt",
        [this](std::span<const PapyrusValue> arguments, std::uint64_t, BethesdaWorld&) {
            NativeCallResult result;
            if (arguments.size() != 2u || arguments[0].type != PapyrusValueType::Integer ||
                arguments[1].type != PapyrusValueType::Integer ||
                arguments[0].integer > arguments[1].integer ||
                arguments[0].integer < std::numeric_limits<std::int32_t>::min() ||
                arguments[1].integer > std::numeric_limits<std::int32_t>::max()) {
                result.error = "Utility.RandomInt expects an ordered integer range";
                return result;
            }
            m_randomState ^= m_randomState << 13u;
            m_randomState ^= m_randomState >> 17u;
            m_randomState ^= m_randomState << 5u;
            if (m_randomState == 0u) m_randomState = 1u;
            const std::uint64_t range = static_cast<std::uint64_t>(
                arguments[1].integer - arguments[0].integer) + 1u;
            result.value = PapyrusValue::fromInteger(
                arguments[0].integer + static_cast<std::int64_t>(m_randomState % range));
            return result;
        });
    const auto globalValueNative = [this](bool set, bool integerResult) {
        return [this, set, integerResult](
            std::span<const PapyrusValue> arguments, std::uint64_t, BethesdaWorld&) {
            NativeCallResult result;
            const std::size_t expected = set ? 2u : 1u;
            if (arguments.size() != expected || arguments[0].type != PapyrusValueType::Object ||
                arguments[0].object.kind != ObjectIdKind::PersistentReference) {
                result.error = "GlobalVariable access expects a persistent global object";
                return result;
            }
            const auto found = m_globalVariables.find(arguments[0].object.reference);
            if (found == m_globalVariables.end()) {
                result.error = "GlobalVariable is not registered from winning content";
                return result;
            }
            if (set) {
                if (arguments[1].type != PapyrusValueType::Float &&
                    arguments[1].type != PapyrusValueType::Integer) {
                    result.error = "GlobalVariable.SetValue expects a numeric value";
                    return result;
                }
                const double value = arguments[1].type == PapyrusValueType::Float
                    ? arguments[1].real : static_cast<double>(arguments[1].integer);
                if (!std::isfinite(value) ||
                    value < -static_cast<double>(std::numeric_limits<float>::max()) ||
                    value > static_cast<double>(std::numeric_limits<float>::max())) {
                    result.error = "GlobalVariable.SetValue received a non-finite/out-of-range value";
                    return result;
                }
                found->second = static_cast<float>(value);
            } else if (integerResult) {
                result.value = PapyrusValue::fromInteger(
                    static_cast<std::int64_t>(found->second));
            } else {
                result.value = PapyrusValue::fromFloat(found->second);
            }
            return result;
        };
    };
    m_papyrus.registerNative("GlobalVariable.GetValue", globalValueNative(false, false));
    m_papyrus.registerNative("GlobalVariable.GetValueInt", globalValueNative(false, true));
    m_papyrus.registerNative("GlobalVariable.SetValue", globalValueNative(true, false));
    m_papyrus.registerNative("GlobalVariable.SetValueInt", globalValueNative(true, true));
    const auto locationQuery = [this](
        std::span<const PapyrusValue> arguments,
        LocationRuntimeState*& outLocation,
        RecordKey* outKeyword,
        std::string& outError) {
        if (arguments.empty() || arguments[0].type != PapyrusValueType::Object ||
            arguments[0].object.kind != ObjectIdKind::PersistentReference) {
            outError = "Location native expects a persistent location object";
            return false;
        }
        const auto location = m_locations.find(arguments[0].object.reference);
        if (location == m_locations.end()) {
            outError = "Location is not registered from winning content";
            return false;
        }
        outLocation = &location->second;
        if (outKeyword != nullptr) {
            if (arguments.size() < 2u || arguments[1].type != PapyrusValueType::Object ||
                arguments[1].object.kind != ObjectIdKind::PersistentReference) {
                outError = "Location keyword native expects a persistent keyword object";
                return false;
            }
            *outKeyword = arguments[1].object.reference;
        }
        return true;
    };
    m_papyrus.registerNative("Location.HasKeyword",
        [locationQuery](std::span<const PapyrusValue> arguments,
            std::uint64_t, BethesdaWorld&) {
            NativeCallResult result;
            LocationRuntimeState* location = nullptr;
            RecordKey keyword;
            if (!locationQuery(arguments, location, &keyword, result.error)) return result;
            result.value = PapyrusValue::fromBoolean(std::binary_search(
                location->keywords.begin(), location->keywords.end(), keyword));
            return result;
        });
    m_papyrus.registerNative("Location.GetKeywordData",
        [locationQuery](std::span<const PapyrusValue> arguments,
            std::uint64_t, BethesdaWorld&) {
            NativeCallResult result;
            LocationRuntimeState* location = nullptr;
            RecordKey keyword;
            if (!locationQuery(arguments, location, &keyword, result.error)) return result;
            const auto value = location->keywordData.find(keyword);
            result.value = PapyrusValue::fromFloat(
                value == location->keywordData.end() ? 0.0f : value->second);
            return result;
        });
    m_papyrus.registerNative("Location.SetKeywordData",
        [locationQuery](std::span<const PapyrusValue> arguments,
            std::uint64_t, BethesdaWorld&) {
            NativeCallResult result;
            LocationRuntimeState* location = nullptr;
            RecordKey keyword;
            if (arguments.size() != 3u ||
                !locationQuery(arguments, location, &keyword, result.error) ||
                (arguments[2].type != PapyrusValueType::Float &&
                 arguments[2].type != PapyrusValueType::Integer)) {
                if (result.error.empty()) {
                    result.error = "Location.SetKeywordData expects location, keyword, and value";
                }
                return result;
            }
            const double value = arguments[2].type == PapyrusValueType::Float
                ? arguments[2].real : static_cast<double>(arguments[2].integer);
            if (!std::isfinite(value)) {
                result.error = "Location.SetKeywordData received a non-finite value";
                return result;
            }
            if (!std::binary_search(location->keywords.begin(), location->keywords.end(), keyword)) {
                location->keywords.insert(
                    std::lower_bound(location->keywords.begin(), location->keywords.end(), keyword),
                    keyword);
            }
            location->keywordData.insert_or_assign(keyword, static_cast<float>(value));
            return result;
        });
    m_papyrus.registerNative("Location.IsLoaded",
        [locationQuery](std::span<const PapyrusValue> arguments,
            std::uint64_t, BethesdaWorld&) {
            NativeCallResult result;
            LocationRuntimeState* location = nullptr;
            if (arguments.size() != 1u ||
                !locationQuery(arguments, location, nullptr, result.error)) return result;
            result.value = PapyrusValue::fromBoolean(location->loaded);
            return result;
        });
    const auto locationRelationshipNative = [this](bool commonParent) {
        return [this, commonParent](std::span<const PapyrusValue> arguments,
            std::uint64_t, BethesdaWorld&) {
            NativeCallResult result;
            if (arguments.size() != 2u || arguments[0].type != PapyrusValueType::Object ||
                arguments[1].type != PapyrusValueType::Object ||
                arguments[0].object.kind != ObjectIdKind::PersistentReference ||
                arguments[1].object.kind != ObjectIdKind::PersistentReference) {
                result.error = "Location relationship query expects two location objects";
                return result;
            }
            const auto ancestry = [this](RecordKey current) {
                std::set<RecordKey> chain;
                for (std::size_t depth = 0u; depth < 64u && current.valid(); ++depth) {
                    if (!chain.insert(current).second) break;
                    const auto found = m_locations.find(current);
                    if (found == m_locations.end()) break;
                    current = found->second.parent;
                }
                return chain;
            };
            const std::set<RecordKey> left = ancestry(arguments[0].object.reference);
            if (commonParent) {
                const std::set<RecordKey> right = ancestry(arguments[1].object.reference);
                result.value = PapyrusValue::fromBoolean(std::any_of(
                    left.begin(), left.end(), [&](const RecordKey& location) {
                        return right.contains(location);
                    }));
            } else {
                result.value = PapyrusValue::fromBoolean(
                    left.contains(arguments[1].object.reference) &&
                    arguments[0].object.reference != arguments[1].object.reference);
            }
            return result;
        };
    };
    m_papyrus.registerNative("Location.IsChild", locationRelationshipNative(false));
    m_papyrus.registerNative("Location.HasCommonParent", locationRelationshipNative(true));
    m_papyrus.registerNative("Keyword.SendStoryEvent",
        [this](std::span<const PapyrusValue> arguments,
            std::uint64_t, BethesdaWorld&) {
            NativeCallResult result;
            if (arguments.empty() || arguments[0].type != PapyrusValueType::Object ||
                arguments[0].object.kind != ObjectIdKind::PersistentReference ||
                arguments.size() > 6u) {
                result.error = "Keyword.SendStoryEvent expects keyword plus up to five arguments";
                return result;
            }
            StoryEventRuntimeState event;
            event.sequence = m_nextStoryEventSequence++;
            event.keyword = arguments[0].object.reference;
            event.arguments.assign(arguments.begin() + 1, arguments.end());
            m_storyEvents.push_back(std::move(event));
            result.value = PapyrusValue::fromBoolean(true);
            return result;
        });
}

void BethesdaSession::registerQuestSceneAndAliasNatives() {
    m_papyrus.registerNative("Quest.GetStageDone",
        [this](std::span<const PapyrusValue> arguments, std::uint64_t, BethesdaWorld&) {
            NativeCallResult result;
            if (arguments.size() != 2u || arguments[0].type != PapyrusValueType::Object ||
                arguments[1].type != PapyrusValueType::Integer ||
                arguments[1].integer < std::numeric_limits<std::int32_t>::min() ||
                arguments[1].integer > std::numeric_limits<std::int32_t>::max()) {
                result.error = "Quest.GetStageDone expects quest and stage";
                return result;
            }
            const QuestRuntimeState* state = findQuest(arguments[0].object);
            if (state == nullptr) {
                result.error = "Quest.GetStageDone received an unknown quest object";
                return result;
            }
            result.value = PapyrusValue::fromBoolean(std::binary_search(
                state->completedStages.begin(), state->completedStages.end(),
                static_cast<std::int32_t>(arguments[1].integer)));
            return result;
        });
    m_papyrus.registerNative("Quest.GetStage",
        [this](std::span<const PapyrusValue> arguments, std::uint64_t, BethesdaWorld&) {
            NativeCallResult result;
            if (arguments.size() != 1u || arguments[0].type != PapyrusValueType::Object) {
                result.error = "Quest.GetStage expects a quest object";
                return result;
            }
            const QuestRuntimeState* state = findQuest(arguments[0].object);
            if (state == nullptr) result.error = "Quest.GetStage received an unknown quest object";
            else result.value = PapyrusValue::fromInteger(state->stage);
            return result;
        });
    const auto questStatusNative = [this](auto query, const char* name) {
        return [this, query, name](
            std::span<const PapyrusValue> arguments, std::uint64_t, BethesdaWorld&) {
            NativeCallResult result;
            if (arguments.size() != 1u || arguments[0].type != PapyrusValueType::Object) {
                result.error = std::string(name) + " expects a quest object";
                return result;
            }
            const QuestRuntimeState* state = findQuest(arguments[0].object);
            if (state == nullptr) result.error = std::string(name) + " received an unknown quest";
            else result.value = PapyrusValue::fromBoolean(query(*state));
            return result;
        };
    };
    m_papyrus.registerNative("Quest.IsStopped", questStatusNative(
        [](const QuestRuntimeState& state) { return !state.running; }, "Quest.IsStopped"));
    m_papyrus.registerNative("Quest.IsRunning", questStatusNative(
        [](const QuestRuntimeState& state) { return state.running; }, "Quest.IsRunning"));
    m_papyrus.registerNative("Quest.IsCompleted", questStatusNative(
        [](const QuestRuntimeState& state) { return state.completed; }, "Quest.IsCompleted"));
    m_papyrus.registerNative("Quest.SetStage",
        [this](std::span<const PapyrusValue> arguments, std::uint64_t, BethesdaWorld&) {
            NativeCallResult result;
            if (arguments.size() != 2u || arguments[0].type != PapyrusValueType::Object ||
                arguments[1].type != PapyrusValueType::Integer ||
                arguments[1].integer < std::numeric_limits<std::int32_t>::min() ||
                arguments[1].integer > std::numeric_limits<std::int32_t>::max()) {
                result.error = "Quest.SetStage expects quest and stage";
                return result;
            }
            QuestRuntimeState* state = findQuest(arguments[0].object);
            if (state == nullptr) {
                result.error = "Quest.SetStage received an unknown quest object";
                return result;
            }
            setQuestStage(state->editorId, static_cast<std::int32_t>(arguments[1].integer));
            result.value = PapyrusValue::fromBoolean(true);
            return result;
        });
    m_papyrus.registerNative("Quest.Start",
        [this](std::span<const PapyrusValue> arguments, std::uint64_t, BethesdaWorld&) {
            NativeCallResult result;
            if (arguments.size() != 1u || arguments[0].type != PapyrusValueType::Object) {
                result.error = "Quest.Start expects a quest object";
                return result;
            }
            QuestRuntimeState* state = findQuest(arguments[0].object);
            if (state == nullptr) result.error = "Quest.Start received an unknown quest object";
            else { state->running = true; result.value = PapyrusValue::fromBoolean(true); }
            return result;
        });
    m_papyrus.registerNative("Quest.Stop",
        [this](std::span<const PapyrusValue> arguments, std::uint64_t, BethesdaWorld&) {
            NativeCallResult result;
            if (arguments.size() != 1u || arguments[0].type != PapyrusValueType::Object) {
                result.error = "Quest.Stop expects a quest object";
                return result;
            }
            QuestRuntimeState* state = findQuest(arguments[0].object);
            if (state == nullptr) result.error = "Quest.Stop received an unknown quest object";
            else state->running = false;
            return result;
        });
    m_papyrus.registerNative("Quest.SetActive",
        [this](std::span<const PapyrusValue> arguments, std::uint64_t, BethesdaWorld&) {
            NativeCallResult result;
            if (arguments.size() < 2u || arguments[0].type != PapyrusValueType::Object ||
                arguments[1].type != PapyrusValueType::Boolean) {
                result.error = "Quest.SetActive expects quest and boolean";
                return result;
            }
            QuestRuntimeState* state = findQuest(arguments[0].object);
            if (state == nullptr) result.error = "Quest.SetActive received an unknown quest object";
            else state->running = arguments[1].boolean;
            return result;
        });
    const auto objectiveNative = [this](bool completed) {
        return [this, completed](
            std::span<const PapyrusValue> arguments, std::uint64_t, BethesdaWorld&) {
            NativeCallResult result;
            if (arguments.size() < 3u || arguments[0].type != PapyrusValueType::Object ||
                arguments[1].type != PapyrusValueType::Integer ||
                arguments[2].type != PapyrusValueType::Boolean) {
                result.error = "quest objective mutation expects quest, index, and boolean";
                return result;
            }
            QuestRuntimeState* state = findQuest(arguments[0].object);
            if (state == nullptr) {
                result.error = "quest objective mutation received an unknown quest object";
                return result;
            }
            const std::int32_t index = static_cast<std::int32_t>(arguments[1].integer);
            auto found = std::find_if(state->objectives.begin(), state->objectives.end(),
                [&](const QuestObjectiveState& objective) { return objective.index == index; });
            if (found == state->objectives.end()) {
                QuestObjectiveState objective;
                objective.index = index;
                state->objectives.push_back(std::move(objective));
                found = std::prev(state->objectives.end());
            }
            if (completed) found->completed = arguments[2].boolean;
            else found->displayed = arguments[2].boolean;
            result.value = PapyrusValue::fromBoolean(true);
            return result;
        };
    };
    m_papyrus.registerNative("Quest.SetObjectiveDisplayed", objectiveNative(false));
    m_papyrus.registerNative("Quest.SetObjectiveCompleted", objectiveNative(true));
    const auto objectiveQueryNative = [this](bool completed) {
        return [this, completed](
            std::span<const PapyrusValue> arguments, std::uint64_t, BethesdaWorld&) {
            NativeCallResult result;
            if (arguments.size() != 2u || arguments[0].type != PapyrusValueType::Object ||
                arguments[1].type != PapyrusValueType::Integer) {
                result.error = "quest objective query expects quest and objective index";
                return result;
            }
            const QuestRuntimeState* state = findQuest(arguments[0].object);
            if (state == nullptr) {
                result.error = "quest objective query received an unknown quest";
                return result;
            }
            const auto found = std::find_if(state->objectives.begin(), state->objectives.end(),
                [&](const QuestObjectiveState& objective) {
                    return objective.index == arguments[1].integer;
                });
            result.value = PapyrusValue::fromBoolean(found != state->objectives.end() &&
                (completed ? found->completed : found->displayed));
            return result;
        };
    };
    m_papyrus.registerNative("Quest.IsObjectiveCompleted", objectiveQueryNative(true));
    m_papyrus.registerNative("Quest.IsObjectiveDisplayed", objectiveQueryNative(false));
    m_papyrus.registerNative("Quest.SetObjectiveFailed",
        [this](std::span<const PapyrusValue> arguments, std::uint64_t, BethesdaWorld&) {
            NativeCallResult result;
            if (arguments.size() < 3u || arguments[0].type != PapyrusValueType::Object ||
                arguments[1].type != PapyrusValueType::Integer ||
                arguments[2].type != PapyrusValueType::Boolean) {
                result.error = "Quest.SetObjectiveFailed expects quest, index, and boolean";
                return result;
            }
            QuestRuntimeState* state = findQuest(arguments[0].object);
            if (state == nullptr) {
                result.error = "Quest.SetObjectiveFailed received an unknown quest";
                return result;
            }
            const std::int32_t index = static_cast<std::int32_t>(arguments[1].integer);
            auto found = std::find_if(state->objectives.begin(), state->objectives.end(),
                [&](const QuestObjectiveState& objective) { return objective.index == index; });
            if (found == state->objectives.end()) {
                QuestObjectiveState objective;
                objective.index = index;
                state->objectives.push_back(std::move(objective));
                found = std::prev(state->objectives.end());
            }
            found->failed = arguments[2].boolean;
            return result;
        });
    m_papyrus.registerNative("Quest.FailAllObjectives",
        [this](std::span<const PapyrusValue> arguments, std::uint64_t, BethesdaWorld&) {
            NativeCallResult result;
            if (arguments.size() != 1u || arguments[0].type != PapyrusValueType::Object) {
                result.error = "Quest.FailAllObjectives expects a quest object";
                return result;
            }
            QuestRuntimeState* state = findQuest(arguments[0].object);
            if (state == nullptr) result.error = "Quest.FailAllObjectives received an unknown quest object";
            else {
                state->failed = true;
                for (QuestObjectiveState& objective : state->objectives) {
                    if (!objective.completed) objective.failed = true;
                }
            }
            return result;
        });
    m_papyrus.registerNative("Quest.CompleteAllObjectives",
        [this](std::span<const PapyrusValue> arguments, std::uint64_t, BethesdaWorld&) {
            NativeCallResult result;
            if (arguments.size() != 1u || arguments[0].type != PapyrusValueType::Object) {
                result.error = "Quest.CompleteAllObjectives expects a quest object";
                return result;
            }
            QuestRuntimeState* state = findQuest(arguments[0].object);
            if (state == nullptr) result.error = "Quest.CompleteAllObjectives received an unknown quest object";
            else for (QuestObjectiveState& objective : state->objectives) {
                objective.displayed = true;
                objective.completed = true;
                objective.failed = false;
            }
            return result;
        });
    const auto scenePlayingNative = [this](bool playing) {
        return [this, playing](
            std::span<const PapyrusValue> arguments, std::uint64_t, BethesdaWorld&) {
            NativeCallResult result;
            if (arguments.size() != 1u || arguments[0].type != PapyrusValueType::Object ||
                arguments[0].object.kind != ObjectIdKind::PersistentReference) {
                result.error = "Scene mutation expects a persistent scene object";
                return result;
            }
            setScenePlaying(arguments[0].object.reference, playing);
            return result;
        };
    };
    m_papyrus.registerNative("Scene.Start", scenePlayingNative(true));
    m_papyrus.registerNative("Scene.ForceStart", [this, start = scenePlayingNative(true)](std::span<const PapyrusValue> arguments, std::uint64_t tick, BethesdaWorld& world) {
        if (arguments.size() == 1 && arguments[0].type == PapyrusValueType::Object)
            setScenePlaying(arguments[0].object.reference, false);
        return start(arguments, tick, world);
    });
    m_papyrus.registerNative("Scene.GetOwningQuest", [this](std::span<const PapyrusValue> arguments, std::uint64_t, BethesdaWorld&) {
        NativeCallResult result;
        if (!arguments.empty()) {
            const auto found = m_sceneDefinitions.find(arguments[0].object.reference);
            if (found != m_sceneDefinitions.end()) result.value = PapyrusValue::fromObject(ObjectId::persistent(found->second.quest));
        }
        return result;
    });
    m_papyrus.registerNative("Scene.Stop", scenePlayingNative(false));
    m_papyrus.registerNative("Package.GetOwningQuest", [this](std::span<const PapyrusValue> arguments, std::uint64_t, BethesdaWorld&) {
        NativeCallResult result;
        if (!arguments.empty() && arguments[0].type == PapyrusValueType::Object) {
            const auto found = m_scenePackages.find(arguments[0].object.reference);
            if (found != m_scenePackages.end() && found->second.quest.valid()) result.value = PapyrusValue::fromObject(ObjectId::persistent(found->second.quest));
        }
        return result;
    });
    m_papyrus.registerNative("Scene.IsPlaying",
        [this](std::span<const PapyrusValue> arguments, std::uint64_t, BethesdaWorld&) {
            NativeCallResult result;
            if (arguments.size() != 1u || arguments[0].type != PapyrusValueType::Object ||
                arguments[0].object.kind != ObjectIdKind::PersistentReference) {
                result.error = "Scene.IsPlaying expects a persistent scene object";
                return result;
            }
            const auto found = m_scenes.find(arguments[0].object.reference);
            result.value = PapyrusValue::fromBoolean(found != m_scenes.end() && found->second);
            return result;
        });
    const auto aliasReferenceNative = [this](
        std::span<const PapyrusValue> arguments, std::uint64_t, BethesdaWorld&) {
        NativeCallResult result;
        if (arguments.size() != 1u || arguments[0].type != PapyrusValueType::Object) {
            result.error = "ReferenceAlias lookup expects an alias object";
            return result;
        }
        const QuestAliasRuntimeState* alias = findQuestAlias(arguments[0].object);
        if (alias == nullptr) result.error = "unknown quest alias handle";
        else if (alias->target.valid()) result.value = PapyrusValue::fromObject(alias->target);
        return result;
    };
    m_papyrus.registerNative("ReferenceAlias.GetReference", aliasReferenceNative);
    m_papyrus.registerNative("ReferenceAlias.GetRef", aliasReferenceNative);
    m_papyrus.registerNative("ReferenceAlias.GetActorReference", aliasReferenceNative);
    m_papyrus.registerNative("ReferenceAlias.GetActorRef", aliasReferenceNative);
    m_papyrus.registerNative("LocationAlias.GetLocation", aliasReferenceNative);
    m_papyrus.registerNative("ReferenceAlias.GetOwningQuest",
        [this](std::span<const PapyrusValue> arguments, std::uint64_t, BethesdaWorld&) {
            NativeCallResult result;
            if (arguments.size() != 1u || arguments[0].type != PapyrusValueType::Object) {
                result.error = "ReferenceAlias.GetOwningQuest expects an alias object";
                return result;
            }
            for (const auto& [name, questState] : m_quests) {
                (void)name;
                const auto alias = std::find_if(
                    questState.aliases.begin(), questState.aliases.end(),
                    [&](const QuestAliasRuntimeState& candidate) {
                        return candidate.handle == arguments[0].object;
                    });
                if (alias != questState.aliases.end() && questState.record.valid()) {
                    result.value = PapyrusValue::fromObject(
                        ObjectId::persistent(questState.record));
                    return result;
                }
            }
            result.error = "unknown quest alias handle";
            return result;
        });
    m_papyrus.registerNative("ReferenceAlias.ForceRefTo",
        [this](std::span<const PapyrusValue> arguments, std::uint64_t, BethesdaWorld&) {
            NativeCallResult result;
            if (arguments.size() != 2u || arguments[0].type != PapyrusValueType::Object ||
                arguments[1].type != PapyrusValueType::Object || !arguments[1].object.valid()) {
                result.error = "ReferenceAlias.ForceRefTo expects alias and target objects";
                return result;
            }
            QuestAliasRuntimeState* alias = findQuestAlias(arguments[0].object);
            if (alias == nullptr) result.error = "unknown quest alias handle";
            else alias->target = arguments[1].object;
            return result;
        });
    m_papyrus.registerNative("ReferenceAlias.Clear",
        [this](std::span<const PapyrusValue> arguments, std::uint64_t, BethesdaWorld&) {
            NativeCallResult result;
            if (arguments.size() != 1u || arguments[0].type != PapyrusValueType::Object) {
                result.error = "ReferenceAlias.Clear expects an alias object";
                return result;
            }
            QuestAliasRuntimeState* alias = findQuestAlias(arguments[0].object);
            if (alias == nullptr) result.error = "unknown quest alias handle";
            else alias->target = {};
            return result;
        });
    m_papyrus.registerNative("ReferenceAlias.TryToMoveTo",
        [this](std::span<const PapyrusValue> arguments, std::uint64_t, BethesdaWorld& world) {
            NativeCallResult result;
            if (arguments.size() != 2u || arguments[0].type != PapyrusValueType::Object ||
                arguments[1].type != PapyrusValueType::Object || !arguments[1].object.valid()) {
                result.error = "ReferenceAlias.TryToMoveTo expects alias and destination objects";
                return result;
            }
            const QuestAliasRuntimeState* alias = findQuestAlias(arguments[0].object);
            if (alias == nullptr || !alias->target.valid()) {
                result.error = "ReferenceAlias.TryToMoveTo received an empty quest alias";
                return result;
            }
            WorldCommand command;
            command.type = WorldCommandType::RequestMoveTo;
            command.target = alias->target;
            command.destination = arguments[1].object;
            (void)world.queue(std::move(command));
            result.value = PapyrusValue::fromBoolean(true);
            return result;
        });
    m_papyrus.registerNative("ReferenceAlias.TryToEnable",
        [this](std::span<const PapyrusValue> arguments, std::uint64_t, BethesdaWorld& world) {
            NativeCallResult result;
            if (arguments.size() != 1u || arguments[0].type != PapyrusValueType::Object) {
                result.error = "ReferenceAlias.TryToEnable expects one alias object";
                return result;
            }
            const QuestAliasRuntimeState* alias = findQuestAlias(arguments[0].object);
            if (alias == nullptr || !alias->target.valid()) {
                result.value = PapyrusValue::fromBoolean(false);
                return result;
            }
            WorldCommand command;
            command.type = WorldCommandType::SetEnabled;
            command.target = alias->target;
            command.enabled = true;
            (void)world.queue(std::move(command));
            result.value = PapyrusValue::fromBoolean(true);
            return result;
        });
    m_papyrus.registerNative("ReferenceAlias.TryToDisable",
        [this](std::span<const PapyrusValue> arguments, std::uint64_t, BethesdaWorld& world) {
            NativeCallResult result;
            if (arguments.size() != 1u || arguments[0].type != PapyrusValueType::Object) {
                result.error = "ReferenceAlias.TryToDisable expects one alias object";
                return result;
            }
            const QuestAliasRuntimeState* alias = findQuestAlias(arguments[0].object);
            if (alias == nullptr || !alias->target.valid()) {
                result.value = PapyrusValue::fromBoolean(false);
                return result;
            }
            WorldCommand command;
            command.type = WorldCommandType::SetEnabled;
            command.target = alias->target;
            command.enabled = false;
            (void)world.queue(std::move(command));
            result.value = PapyrusValue::fromBoolean(true);
            return result;
        });
    m_papyrus.registerNative("Game.GetQuestStage",
        [this](std::span<const PapyrusValue> arguments, std::uint64_t, BethesdaWorld&) {
            NativeCallResult result;
            if (arguments.size() != 1u || arguments[0].type != PapyrusValueType::String) {
                result.error = "Game.GetQuestStage expects one quest EditorID string";
                return result;
            }
            const QuestRuntimeState* state = findQuest(arguments[0].string);
            result.value = PapyrusValue::fromInteger(state == nullptr ? 0 : state->stage);
            return result;
        });
    m_papyrus.registerNative("Game.SetQuestStage",
        [this](std::span<const PapyrusValue> arguments, std::uint64_t, BethesdaWorld&) {
            NativeCallResult result;
            if (arguments.size() != 2u || arguments[0].type != PapyrusValueType::String ||
                arguments[1].type != PapyrusValueType::Integer) {
                result.error = "Game.SetQuestStage expects quest EditorID and integer stage";
                return result;
            }
            setQuestStage(arguments[0].string, static_cast<std::int32_t>(arguments[1].integer));
            return result;
        });
}

void BethesdaSession::registerObjectReferenceNatives() {
    m_papyrus.registerNative("ObjectReference.PlayGamebryoAnimation",
        [this](std::span<const PapyrusValue> arguments, std::uint64_t, BethesdaWorld&) {
            NativeCallResult result;
            if (arguments.size() < 2 || arguments.size() > 4 || arguments[0].type != PapyrusValueType::Object ||
                arguments[0].object.kind != ObjectIdKind::PersistentReference ||
                arguments[1].type != PapyrusValueType::String || arguments[1].string.empty() ||
                (arguments.size() >= 3 && arguments[2].type != PapyrusValueType::Boolean) ||
                (arguments.size() == 4 && arguments[3].type != PapyrusValueType::Float)) {
                result.error = "PlayGamebryoAnimation expects reference, sequence, optional startOver and easeIn";
                return result;
            }
            if (arguments.size() == 4 && arguments[3].real != 0.0) {
                result.error = "Gamebryo sequence cross-fade is not supported";
                return result;
            }
            result.value = PapyrusValue::fromBoolean(m_effectAnimationPlayer &&
                m_effectAnimationPlayer({arguments[0].object.reference, arguments[1].string,
                    arguments.size() >= 3 && arguments[2].boolean}));
            return result;
        });
    m_papyrus.registerNative("ObjectReference.AddItem",
        [](std::span<const PapyrusValue> arguments, std::uint64_t, BethesdaWorld& world) {
            NativeCallResult result;
            if (arguments.size() < 3u || arguments[0].type != PapyrusValueType::Object ||
                arguments[2].type != PapyrusValueType::Integer) {
                result.error = "ObjectReference.AddItem expects object, item form, count";
                return result;
            }
            RecordKey item;
            if (arguments[1].type == PapyrusValueType::Object &&
                arguments[1].object.kind == ObjectIdKind::PersistentReference) {
                item = arguments[1].object.reference;
            } else if (arguments[1].type == PapyrusValueType::Object) {
                const RuntimeObject* instance = world.find(arguments[1].object);
                if (instance != nullptr && instance->kind == RuntimeObjectKind::Item) {
                    item = instance->base;
                }
            } else if (arguments[1].type != PapyrusValueType::String ||
                !parseRecordKey(arguments[1].string, item)) {
                result.error = "ObjectReference.AddItem received an invalid item form";
                return result;
            }
            if (!item.valid()) {
                result.error = "ObjectReference.AddItem received an invalid item form";
                return result;
            }
            if (arguments[2].integer <= 0) {
                result.error = "ObjectReference.AddItem received an invalid item/count";
                return result;
            }
            WorldCommand command;
            command.type = WorldCommandType::AddItem;
            command.target = arguments[0].object;
            command.item = std::move(item);
            command.itemCount = static_cast<std::int32_t>(arguments[2].integer);
            (void)world.queue(std::move(command));
            return result;
        });
    m_papyrus.registerNative("ObjectReference.GetItemCount",
        [](std::span<const PapyrusValue> arguments, std::uint64_t, BethesdaWorld& world) {
            NativeCallResult result;
            if (arguments.size() != 2u || arguments[0].type != PapyrusValueType::Object ||
                arguments[1].type != PapyrusValueType::Object) {
                result.error = "ObjectReference.GetItemCount expects object and item form";
                return result;
            }
            const RuntimeObject* owner = world.find(arguments[0].object);
            if (owner == nullptr) {
                result.error = "ObjectReference.GetItemCount requires a resident object";
                return result;
            }
            RecordKey item;
            if (arguments[1].object.kind == ObjectIdKind::PersistentReference) {
                item = arguments[1].object.reference;
            } else if (const RuntimeObject* instance = world.find(arguments[1].object);
                       instance != nullptr && instance->kind == RuntimeObjectKind::Item) {
                item = instance->base;
            }
            if (!item.valid()) {
                result.error = "ObjectReference.GetItemCount received an invalid item form";
                return result;
            }
            const auto found = std::find_if(owner->inventory.begin(), owner->inventory.end(),
                [&](const InventoryEntry& entry) { return entry.item == item; });
            result.value = PapyrusValue::fromInteger(
                found == owner->inventory.end() ? 0 : std::max(0, found->count));
            return result;
        });
    m_papyrus.registerNative("ObjectReference.RemoveItem",
        [](std::span<const PapyrusValue> arguments, std::uint64_t, BethesdaWorld& world) {
            NativeCallResult result;
            if (arguments.size() < 3u || arguments[0].type != PapyrusValueType::Object ||
                arguments[1].type != PapyrusValueType::Object ||
                arguments[2].type != PapyrusValueType::Integer || arguments[2].integer <= 0) {
                result.error = "ObjectReference.RemoveItem expects object, item form, positive count";
                return result;
            }
            RecordKey item;
            if (arguments[1].object.kind == ObjectIdKind::PersistentReference) {
                item = arguments[1].object.reference;
            } else if (const RuntimeObject* instance = world.find(arguments[1].object);
                       instance != nullptr && instance->kind == RuntimeObjectKind::Item) {
                item = instance->base;
            }
            if (!item.valid()) {
                result.error = "ObjectReference.RemoveItem received an invalid item form";
                return result;
            }
            WorldCommand command;
            command.type = WorldCommandType::RemoveItem;
            command.target = arguments[0].object;
            command.item = std::move(item);
            command.itemCount = static_cast<std::int32_t>(arguments[2].integer);
            (void)world.queue(std::move(command));
            return result;
        });
    const auto enabledNative = [](bool enabled) {
        return [enabled](std::span<const PapyrusValue> arguments, std::uint64_t, BethesdaWorld& world) {
            NativeCallResult result;
            if (arguments.empty() || arguments[0].type != PapyrusValueType::Object) {
                result.error = "ObjectReference enable/disable expects an object";
                return result;
            }
            WorldCommand command;
            command.type = WorldCommandType::SetEnabled;
            command.target = arguments[0].object;
            command.enabled = enabled;
            (void)world.queue(std::move(command));
            return result;
        };
    };
    m_papyrus.registerNative("ObjectReference.Enable", enabledNative(true));
    m_papyrus.registerNative("ObjectReference.Disable", enabledNative(false));
    const auto residentObject = [](const BethesdaWorld& world,
                                   const ObjectId& identity) -> std::optional<RuntimeObject> {
        if (const RuntimeObject* direct = world.find(identity)) return *direct;
        if (identity.kind != ObjectIdKind::PersistentReference) return std::nullopt;
        for (const ObjectId& residentId : world.orderedObjectIds()) {
            const RuntimeObject* object = world.find(residentId);
            if (object != nullptr && object->base == identity.reference) return *object;
        }
        return std::nullopt;
    };
    m_papyrus.registerNative("ObjectReference.Is3DLoaded",
        [residentObject](std::span<const PapyrusValue> arguments,
            std::uint64_t, BethesdaWorld& world) {
            NativeCallResult result;
            if (arguments.size() != 1u || arguments[0].type != PapyrusValueType::Object) {
                result.error = "ObjectReference.Is3DLoaded expects one object";
                return result;
            }
            result.value = PapyrusValue::fromBoolean(
                residentObject(world, arguments[0].object).has_value());
            return result;
        });
    m_papyrus.registerNative("ObjectReference.IsInInterior",
        [residentObject](std::span<const PapyrusValue> arguments,
            std::uint64_t, BethesdaWorld& world) {
            NativeCallResult result;
            if (arguments.size() != 1u || arguments[0].type != PapyrusValueType::Object) {
                result.error = "ObjectReference.IsInInterior expects one object";
                return result;
            }
            const std::optional<RuntimeObject> object =
                residentObject(world, arguments[0].object);
            result.value = PapyrusValue::fromBoolean(
                object.has_value() && object->interior);
            return result;
        });
    m_papyrus.registerNative("ObjectReference.MoveTo",
        [](std::span<const PapyrusValue> arguments,
            std::uint64_t, BethesdaWorld& world) {
            NativeCallResult result;
            if ((arguments.size() != 2u && arguments.size() != 6u) ||
                arguments[0].type != PapyrusValueType::Object ||
                arguments[1].type != PapyrusValueType::Object ||
                !arguments[1].object.valid()) {
                result.error = "ObjectReference.MoveTo expects object, destination, and optional offsets";
                return result;
            }
            WorldCommand command;
            command.type = WorldCommandType::TeleportToReference;
            command.target = arguments[0].object;
            command.destination = arguments[1].object;
            (void)world.queue(std::move(command));
            return result;
        });
    m_papyrus.registerNative("ObjectReference.AddToMap",
        [this](std::span<const PapyrusValue> arguments, std::uint64_t, BethesdaWorld&) {
            NativeCallResult result;
            if (arguments.empty() || arguments[0].type != PapyrusValueType::Object ||
                arguments[0].object.kind != ObjectIdKind::PersistentReference) {
                result.error = "ObjectReference.AddToMap expects a persistent map marker";
                return result;
            }
            const RecordKey marker = arguments[0].object.reference;
            if (std::find(m_discoveries.begin(), m_discoveries.end(), marker) == m_discoveries.end()) {
                m_discoveries.push_back(marker);
                std::sort(m_discoveries.begin(), m_discoveries.end());
            }
            return result;
        });
}

void BethesdaSession::registerActorNatives() {
    m_papyrus.registerNative("Actor.DamageActorValue",
        [](std::span<const PapyrusValue> arguments, std::uint64_t, BethesdaWorld& world) {
            NativeCallResult result;
            if (arguments.size() != 3u || arguments[0].type != PapyrusValueType::Object ||
                arguments[1].type != PapyrusValueType::String ||
                (arguments[2].type != PapyrusValueType::Float &&
                 arguments[2].type != PapyrusValueType::Integer)) {
                result.error = "Actor.DamageActorValue expects actor, value name, float amount";
                return result;
            }
            WorldCommand command;
            command.type = WorldCommandType::AdjustActorValue;
            command.target = arguments[0].object;
            const std::string value = normalizedEditorId(arguments[1].string);
            if (value == "health") command.actorValue = ActorValue::Health;
            else if (value == "stamina") command.actorValue = ActorValue::Stamina;
            else if (value == "magicka") command.actorValue = ActorValue::Magicka;
            else { result.error = "unsupported actor value " + arguments[1].string; return result; }
            const double amount = arguments[2].type == PapyrusValueType::Float
                ? arguments[2].real : static_cast<double>(arguments[2].integer);
            command.actorValueDelta = -static_cast<float>(amount);
            (void)world.queue(std::move(command));
            return result;
        });
    m_papyrus.registerNative("Actor.IsDead",
        [](std::span<const PapyrusValue> arguments, std::uint64_t, BethesdaWorld& world) {
            NativeCallResult result;
            if (arguments.size() != 1u || arguments[0].type != PapyrusValueType::Object) {
                result.error = "Actor.IsDead expects an actor object";
                return result;
            }
            const RuntimeObject* actor = world.find(arguments[0].object);
            if (actor == nullptr || actor->kind != RuntimeObjectKind::Actor) {
                result.error = "Actor.IsDead target is not resident as an actor";
            } else {
                result.value = PapyrusValue::fromBoolean(
                    actor->actorValues.has_value() && actor->actorValues->dead);
            }
            return result;
        });
    m_papyrus.registerNative("Actor.StartCombat",
        [](std::span<const PapyrusValue> arguments, std::uint64_t, BethesdaWorld& world) {
            NativeCallResult result;
            if (arguments.size() != 2u ||
                arguments[0].type != PapyrusValueType::Object ||
                arguments[1].type != PapyrusValueType::Object) {
                result.error = "Actor.StartCombat expects actor and target objects";
                return result;
            }
            const RuntimeObject* actor = world.find(arguments[0].object);
            const RuntimeObject* target = world.find(arguments[1].object);
            if (actor == nullptr || target == nullptr ||
                actor->kind != RuntimeObjectKind::Actor ||
                target->kind != RuntimeObjectKind::Actor) {
                result.error = "Actor.StartCombat requires resident actors";
                return result;
            }
            RuntimeCombatState state = actor->combatState.value_or(RuntimeCombatState{});
            state.combatTarget = arguments[1].object;
            WorldCommand command;
            command.type = WorldCommandType::SetCombatState;
            command.target = arguments[0].object;
            command.combatState = std::move(state);
            (void)world.queue(std::move(command));
            return result;
        });
    m_papyrus.registerNative("Actor.StopCombat",
        [](std::span<const PapyrusValue> arguments, std::uint64_t, BethesdaWorld& world) {
            NativeCallResult result;
            if (arguments.size() != 1u ||
                arguments[0].type != PapyrusValueType::Object) {
                result.error = "Actor.StopCombat expects one actor object";
                return result;
            }
            const RuntimeObject* actor = world.find(arguments[0].object);
            if (actor == nullptr || actor->kind != RuntimeObjectKind::Actor) {
                result.error = "Actor.StopCombat requires a resident actor";
                return result;
            }
            RuntimeCombatState state = actor->combatState.value_or(RuntimeCombatState{});
            state.combatTarget = {};
            WorldCommand command;
            command.type = WorldCommandType::SetCombatState;
            command.target = arguments[0].object;
            command.combatState = std::move(state);
            (void)world.queue(std::move(command));
            return result;
        });
    m_papyrus.registerNative("Actor.GetDistance",
        [](std::span<const PapyrusValue> arguments, std::uint64_t, BethesdaWorld& world) {
            NativeCallResult result;
            if (arguments.size() != 2u || arguments[0].type != PapyrusValueType::Object ||
                arguments[1].type != PapyrusValueType::Object) {
                result.error = "Actor.GetDistance expects two objects";
                return result;
            }
            const RuntimeObject* left = world.find(arguments[0].object);
            const RuntimeObject* right = world.find(arguments[1].object);
            if (left == nullptr || right == nullptr) {
                result.error = "Actor.GetDistance requires both objects to be resident";
                return result;
            }
            const double dx = left->transform.position[0] - right->transform.position[0];
            const double dy = left->transform.position[1] - right->transform.position[1];
            const double dz = left->transform.position[2] - right->transform.position[2];
            result.value = PapyrusValue::fromFloat(std::sqrt(dx * dx + dy * dy + dz * dz));
            return result;
        });
    m_papyrus.registerNative("Actor.GetActorValue",
        [](std::span<const PapyrusValue> arguments,
            std::uint64_t, BethesdaWorld& world) {
            NativeCallResult result;
            if (arguments.size() != 2u || arguments[0].type != PapyrusValueType::Object ||
                arguments[1].type != PapyrusValueType::String) {
                result.error = "Actor.GetActorValue expects actor and value name";
                return result;
            }
            const RuntimeObject* actor = world.find(arguments[0].object);
            if (actor == nullptr || !actor->actorValues.has_value()) {
                result.error = "Actor.GetActorValue requires a resident actor";
                return result;
            }
            const std::string value = normalizedEditorId(arguments[1].string);
            if (value == "health") result.value = PapyrusValue::fromFloat(actor->actorValues->health);
            else if (value == "stamina") result.value = PapyrusValue::fromFloat(actor->actorValues->stamina);
            else if (value == "magicka") result.value = PapyrusValue::fromFloat(actor->actorValues->magicka);
            else if (value == "aggression") result.value = PapyrusValue::fromFloat(actor->actorValues->aggression);
            else if (value == "lightarmor" || value == "heavyarmor") {
                // Skyrim's new-game skills are equal at the scenario boundary;
                // their mutable skill progression is not yet in ActorValues.
                result.value = PapyrusValue::fromFloat(15.0);
            } else {
                result.error = "unsupported actor value " + arguments[1].string;
            }
            return result;
        });
    const auto actorBooleanQuery = [](auto query, const char* name) {
        return [query, name](
            std::span<const PapyrusValue> arguments, std::uint64_t, BethesdaWorld& world) {
            NativeCallResult result;
            if (arguments.size() != 1u || arguments[0].type != PapyrusValueType::Object) {
                result.error = std::string(name) + " expects an actor object";
                return result;
            }
            const RuntimeObject* actor = world.find(arguments[0].object);
            if (actor == nullptr || actor->kind != RuntimeObjectKind::Actor) {
                result.error = std::string(name) + " target is not resident as an actor";
            } else {
                result.value = PapyrusValue::fromBoolean(query(*actor));
            }
            return result;
        };
    };
    m_papyrus.registerNative("Actor.IsInInterior", actorBooleanQuery(
        [](const RuntimeObject& actor) { return actor.interior; }, "Actor.IsInInterior"));
    m_papyrus.registerNative("Actor.IsInDialogueWithPlayer", actorBooleanQuery(
        [](const RuntimeObject& actor) { return actor.inDialogueWithPlayer; },
        "Actor.IsInDialogueWithPlayer"));
    m_papyrus.registerNative("Actor.IsInLocation",
        [](std::span<const PapyrusValue> arguments, std::uint64_t, BethesdaWorld& world) {
            NativeCallResult result;
            if (arguments.size() != 2u || arguments[0].type != PapyrusValueType::Object ||
                arguments[1].type != PapyrusValueType::Object ||
                arguments[1].object.kind != ObjectIdKind::PersistentReference) {
                result.error = "Actor.IsInLocation expects actor and location objects";
                return result;
            }
            const RuntimeObject* actor = world.find(arguments[0].object);
            if (actor == nullptr || actor->kind != RuntimeObjectKind::Actor) {
                result.error = "Actor.IsInLocation target is not resident as an actor";
            } else {
                result.value = PapyrusValue::fromBoolean(
                    actor->location == arguments[1].object.reference);
            }
            return result;
        });
    m_papyrus.registerNative("Actor.EvaluatePackage",
        [](std::span<const PapyrusValue> arguments, std::uint64_t, BethesdaWorld& world) {
            NativeCallResult result;
            if (arguments.size() != 1u || arguments[0].type != PapyrusValueType::Object) {
                result.error = "Actor.EvaluatePackage expects an actor object";
                return result;
            }
            WorldCommand command;
            command.type = WorldCommandType::EvaluatePackage;
            command.target = arguments[0].object;
            (void)world.queue(std::move(command));
            return result;
        });
    m_papyrus.registerNative("Actor.SetGhost",
        [](std::span<const PapyrusValue> arguments, std::uint64_t, BethesdaWorld& world) {
            NativeCallResult result;
            if (arguments.size() != 2u || arguments[0].type != PapyrusValueType::Object ||
                arguments[1].type != PapyrusValueType::Boolean) {
                result.error = "Actor.SetGhost expects actor and boolean";
                return result;
            }
            WorldCommand command;
            command.type = WorldCommandType::SetGhost;
            command.target = arguments[0].object;
            command.enabled = arguments[1].boolean;
            (void)world.queue(std::move(command));
            return result;
        });
    m_papyrus.registerNative("Actor.SetRelationshipRank",
        [](std::span<const PapyrusValue> arguments, std::uint64_t, BethesdaWorld& world) {
            NativeCallResult result;
            if (arguments.size() != 3u || arguments[0].type != PapyrusValueType::Object ||
                arguments[1].type != PapyrusValueType::Object ||
                arguments[2].type != PapyrusValueType::Integer) {
                result.error = "Actor.SetRelationshipRank expects actor, other actor, and rank";
                return result;
            }
            WorldCommand command;
            command.type = WorldCommandType::SetRelationshipRank;
            command.target = arguments[0].object;
            command.other = arguments[1].object;
            command.relationshipRank = static_cast<std::int32_t>(std::clamp<std::int64_t>(
                arguments[2].integer, -4, 4));
            (void)world.queue(std::move(command));
            return result;
        });
    m_papyrus.registerNative("Actor.GetRelationshipRank",
        [](std::span<const PapyrusValue> arguments, std::uint64_t, BethesdaWorld& world) {
            NativeCallResult result;
            if (arguments.size() != 2u || arguments[0].type != PapyrusValueType::Object ||
                arguments[1].type != PapyrusValueType::Object) {
                result.error = "Actor.GetRelationshipRank expects actor and other actor";
                return result;
            }
            const RuntimeObject* actor = world.find(arguments[0].object);
            if (actor == nullptr || actor->kind != RuntimeObjectKind::Actor) {
                result.error = "Actor.GetRelationshipRank requires a resident actor";
                return result;
            }
            const auto relationship = std::find_if(
                actor->relationships.begin(), actor->relationships.end(),
                [&](const RelationshipRank& value) {
                    return value.other == arguments[1].object;
                });
            result.value = PapyrusValue::fromInteger(
                relationship == actor->relationships.end() ? 0 : relationship->rank);
            return result;
        });
    const auto mutateFaction = [](WorldCommandType type, const char* name) {
        return [type, name](std::span<const PapyrusValue> arguments,
            std::uint64_t, BethesdaWorld& world) {
            NativeCallResult result;
            if (arguments.size() != 2u || arguments[0].type != PapyrusValueType::Object ||
                arguments[1].type != PapyrusValueType::Object ||
                arguments[1].object.kind != ObjectIdKind::PersistentReference) {
                result.error = std::string(name) + " expects actor and faction form";
                return result;
            }
            WorldCommand command;
            command.type = type;
            command.target = arguments[0].object;
            command.faction = arguments[1].object.reference;
            (void)world.queue(std::move(command));
            return result;
        };
    };
    m_papyrus.registerNative("Actor.AddToFaction",
        mutateFaction(WorldCommandType::AddToFaction, "Actor.AddToFaction"));
    m_papyrus.registerNative("Actor.RemoveFromFaction",
        mutateFaction(WorldCommandType::RemoveFromFaction, "Actor.RemoveFromFaction"));
    m_papyrus.registerNative("Actor.IsInFaction",
        [](std::span<const PapyrusValue> arguments, std::uint64_t, BethesdaWorld& world) {
            NativeCallResult result;
            if (arguments.size() != 2u || arguments[0].type != PapyrusValueType::Object ||
                arguments[1].type != PapyrusValueType::Object ||
                arguments[1].object.kind != ObjectIdKind::PersistentReference) {
                result.error = "Actor.IsInFaction expects actor and faction form";
                return result;
            }
            const RuntimeObject* actor = world.find(arguments[0].object);
            if (actor == nullptr || actor->kind != RuntimeObjectKind::Actor) {
                result.error = "Actor.IsInFaction requires a resident actor";
                return result;
            }
            result.value = PapyrusValue::fromBoolean(std::binary_search(
                actor->factions.begin(), actor->factions.end(),
                arguments[1].object.reference));
            return result;
        });
    m_papyrus.registerContextNative("Actor.ShowGiftMenu",
        [this](const PapyrusNativeContext& context,
            std::span<const PapyrusValue> arguments, BethesdaWorld& world) {
            NativeCallResult result;
            result.value = PapyrusValue::fromInteger(0);
            if (arguments.empty() || arguments.size() > 4u) {
                result.error =
                    "Actor.ShowGiftMenu requires giving mode and accepts at most four arguments";
                return result;
            }
            GiftMenuRequestState request;
            request.actor = context.self;
            request.player = ObjectId::persistent(
                makeRecordKey("Skyrim.esm", 0x14u));
            request.useFavorPoints = true;
            const auto booleanArgument = [&](std::size_t index, bool& value) {
                if (index >= arguments.size()) return true;
                if (arguments[index].type != PapyrusValueType::Boolean) return false;
                value = arguments[index].boolean;
                return true;
            };
            if (!booleanArgument(0u, request.playerGives) ||
                !booleanArgument(2u, request.showStolenItems) ||
                !booleanArgument(3u, request.useFavorPoints)) {
                result.error = "Actor.ShowGiftMenu boolean arguments have invalid types";
                return result;
            }
            if (arguments.size() > 1u &&
                arguments[1].type != PapyrusValueType::None) {
                if (arguments[1].type != PapyrusValueType::Object) {
                    result.error = "Actor.ShowGiftMenu filter must be a FormList or None";
                    return result;
                }
                request.filterList = arguments[1].object;
            }
            const RuntimeObject* actor = world.find(request.actor);
            const RuntimeObject* player = world.find(request.player);
            if (actor == nullptr || actor->kind != RuntimeObjectKind::Actor ||
                player == nullptr || player->kind != RuntimeObjectKind::Actor) {
                result.error = "Actor.ShowGiftMenu requires resident actor and player participants";
                return result;
            }
            const auto open = std::find_if(
                m_giftMenuRequests.begin(), m_giftMenuRequests.end(),
                [&](const GiftMenuRequestState& value) {
                    return value.actor == request.actor && value.player == request.player;
                });
            if (open == m_giftMenuRequests.end()) {
                request.sequence = m_nextGiftMenuSequence++;
                if (m_nextGiftMenuSequence == 0u) m_nextGiftMenuSequence = 1u;
                m_giftMenuRequests.push_back(std::move(request));
            }
            return result;
        });
    for (const bool equip : {true, false}) {
        m_papyrus.registerNative(equip ? "Actor.EquipItem" : "Actor.UnequipItem",
            [this, equip](std::span<const PapyrusValue> arguments, std::uint64_t, BethesdaWorld&) {
                NativeCallResult result;
                if (arguments.size() < 2 || arguments[0].type != PapyrusValueType::Object ||
                    arguments[1].type != PapyrusValueType::Object || !arguments[1].object.reference.valid()) {
                    result.error = "Equipment call requires actor and item form"; return result;
                }
                if (arguments.size() > 4 ||
                    (arguments.size() > 2 && arguments[2].type != PapyrusValueType::Boolean) ||
                    (arguments.size() > 3 && arguments[3].type != PapyrusValueType::Boolean)) {
                    result.error = "Equipment optional arguments must be booleans"; return result;
                }
                (void)equipActorItem(arguments[0].object, arguments[1].object.reference, equip, false, result.error,
                    arguments.size() > 2 && arguments[2].boolean, true);
                return result;
            });
        m_papyrus.registerNative(equip ? "Actor.DrawWeapon" : "Actor.SheatheWeapon",
            [this, equip](std::span<const PapyrusValue> arguments, std::uint64_t, BethesdaWorld&) {
                NativeCallResult result;
                if (arguments.size() != 1 || arguments[0].type != PapyrusValueType::Object) {
                    result.error = "Weapon draw call requires an actor"; return result;
                }
                (void)requestActorWeaponDraw(arguments[0].object, equip, result.error);
                return result;
            });
    }
    m_papyrus.registerNative("Actor.IsEquipped",
        [](std::span<const PapyrusValue> arguments, std::uint64_t, BethesdaWorld& world) {
            NativeCallResult result;
            if (arguments.size() != 2 || arguments[0].type != PapyrusValueType::Object ||
                arguments[1].type != PapyrusValueType::Object) {
                result.error = "IsEquipped requires actor and item form"; return result;
            }
            const auto* actor = world.find(arguments[0].object);
            result.value = PapyrusValue::fromBoolean(actor && std::any_of(actor->inventory.begin(), actor->inventory.end(),
                [&](const auto& item) { return item.item == arguments[1].object.reference && item.equipped && item.count > 0; }));
            return result;
        });
    m_papyrus.registerNative("Actor.IsWeaponDrawn",
        [](std::span<const PapyrusValue> arguments, std::uint64_t, BethesdaWorld& world) {
            NativeCallResult result;
            if (arguments.size() != 1 || arguments[0].type != PapyrusValueType::Object) {
                result.error = "IsWeaponDrawn requires actor"; return result;
            }
            const auto* actor = world.find(arguments[0].object);
            result.value = PapyrusValue::fromBoolean(actor && actor->equipment.drawn);
            return result;
        });
    m_papyrus.registerNative("Actor.GetEquippedWeapon",
        [this](std::span<const PapyrusValue> arguments, std::uint64_t, BethesdaWorld& world) {
            NativeCallResult result;
            if (arguments.empty() || arguments.size() > 2 || arguments[0].type != PapyrusValueType::Object ||
                (arguments.size() == 2 && arguments[1].type != PapyrusValueType::Boolean)) {
                result.error = "GetEquippedWeapon requires actor and optional left-hand flag"; return result;
            }
            const auto* actor = world.find(arguments[0].object);
            const auto hand = arguments.size() == 2 && arguments[1].boolean ? kEquipmentLeftHand : kEquipmentRightHand;
            if (actor) for (const auto& entry : actor->inventory) {
                const auto* item = skyrimItem(entry.item);
                if (entry.equipped && entry.count > 0 && (entry.equipmentSlots & hand) && item && item->recordType == "WEAP") {
                    result.value = PapyrusValue::fromObject(ObjectId::persistent(entry.item)); break;
                }
            }
            return result;
        });
    const auto animationRequest = [this](std::span<const PapyrusValue> arguments, std::uint64_t, BethesdaWorld&) {
        NativeCallResult result;
        if (arguments.size() != 2 || arguments[0].type != PapyrusValueType::Object ||
            arguments[1].type != PapyrusValueType::String) {
            result.error = "Animation request requires actor and event name"; return result;
        }
        const bool accepted = queueActorAnimationEvent(arguments[0].object, {arguments[1].string, {}});
        result.value = PapyrusValue::fromBoolean(accepted);
        if (!accepted) result.error = "Animation request has no resident animation instance";
        return result;
    };
    m_papyrus.registerNative("Debug.SendAnimationEvent", animationRequest);
    m_papyrus.registerNative("ObjectReference.PlayAnimation", animationRequest);
    m_papyrus.registerNative("Actor.SetOutfit",
        [](std::span<const PapyrusValue> arguments, std::uint64_t, BethesdaWorld& world) {
            NativeCallResult result;
            if ((arguments.size() != 2u && arguments.size() != 3u) ||
                arguments[0].type != PapyrusValueType::Object ||
                arguments[1].type != PapyrusValueType::Object ||
                arguments[1].object.kind != ObjectIdKind::PersistentReference ||
                (arguments.size() == 3u && arguments[2].type != PapyrusValueType::Boolean)) {
                result.error = "Actor.SetOutfit expects actor, outfit form, and optional sleep flag";
                return result;
            }
            WorldCommand command;
            command.type = WorldCommandType::SetOutfit;
            command.target = arguments[0].object;
            command.outfit = arguments[1].object.reference;
            (void)world.queue(std::move(command));
            return result;
        });
    const auto setActorValueNative = [](std::span<const PapyrusValue> arguments,
        std::uint64_t, BethesdaWorld& world) {
        NativeCallResult result;
        if (arguments.size() != 3u || arguments[0].type != PapyrusValueType::Object ||
            arguments[1].type != PapyrusValueType::String ||
            (arguments[2].type != PapyrusValueType::Float &&
             arguments[2].type != PapyrusValueType::Integer)) {
            result.error = "Actor.SetAV expects actor, value name, and numeric value";
            return result;
        }
        WorldCommand command;
        command.type = WorldCommandType::SetActorValue;
        command.target = arguments[0].object;
        const std::string value = normalizedEditorId(arguments[1].string);
        if (value == "aggression") {
            auto* actor = world.find(command.target);
            const double aggression = arguments[2].type == PapyrusValueType::Float ? arguments[2].real : double(arguments[2].integer);
            if (!actor || !std::isfinite(aggression) || aggression < 0 || aggression > 3) {
                result.error = "Actor.SetActorValue requires a resident actor and aggression in [0,3]"; return result;
            }
            if (!actor->actorValues) actor->actorValues.emplace();
            actor->actorValues->aggression = static_cast<float>(aggression);
            return result;
        }
        if (value == "health") command.actorValue = ActorValue::Health;
        else if (value == "stamina") command.actorValue = ActorValue::Stamina;
        else if (value == "magicka") command.actorValue = ActorValue::Magicka;
        else { result.error = "unsupported actor value " + arguments[1].string; return result; }
        command.actorValueAbsolute = static_cast<float>(arguments[2].type == PapyrusValueType::Float
            ? arguments[2].real : static_cast<double>(arguments[2].integer));
        (void)world.queue(std::move(command));
        return result;
    };
    m_papyrus.registerNative("Actor.SetAV", setActorValueNative);
    m_papyrus.registerNative("Actor.SetActorValue", setActorValueNative);}


}  // namespace odai::bethesda
