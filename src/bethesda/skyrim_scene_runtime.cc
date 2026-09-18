#include "bethesda/bethesda_session.h"
#include "core/log.h"
#include <algorithm>
#include <cmath>
#include <cstdio>

namespace odai::bethesda {
void BethesdaSession::registerScene(SkyrimSceneDefinition scene) {
    m_sceneDefinitions.insert_or_assign(scene.record, std::move(scene));
}
void BethesdaSession::registerScenePackage(SkyrimScenePackage package) {
    m_scenePackages.insert_or_assign(package.record, std::move(package));
}
void BethesdaSession::presentSceneSpeech(std::uint64_t sequence, double seconds) {
    for (auto& line : m_sceneSpeech) if (line.sequence == sequence && !line.presented) {
        line.presented = true;
        line.remainingSeconds = std::isfinite(seconds) && seconds > 0 ? seconds : std::max(2.0, line.text.size()/14.0);
    }
}
void BethesdaSession::advanceScenes(double seconds) {
    m_sceneWaitingActors.clear();
    advanceFlyingPackages(seconds);
    const auto waitForEscort = [&](const SkyrimScenePackage& package, const RuntimeObject& actor) {
        if (package.patrol || !package.escortTarget.valid() || package.escortWaitDistance <= 0) return false;
        const auto* target = m_world.find(package.escortTarget);
        if (!target) return false;
        double distance = 0;
        for (unsigned i=0;i<3;++i) { const auto d = target->transform.position[i]-actor.transform.position[i]; distance += d*d; }
        const bool waiting = distance > package.escortWaitDistance * package.escortWaitDistance;
        if (waiting) m_sceneWaitingActors.insert(actor.id);
        return waiting;
    };
    // Authored trigger boxes generate Papyrus events; the scripts decide which
    // actor/faction can advance chatter and whether the trigger is one-shot.
    for (auto& [id, trigger] : m_scriptTriggers) {
        const auto* volume = m_world.find(id);
        if (!volume || !volume->enabled) continue;
        std::set<ObjectId> occupants;
        for (const auto& actorId : m_world.orderedObjectIds()) {
            const auto* actor = m_world.find(actorId);
            if (!actor || actor->kind != RuntimeObjectKind::Actor || !actor->enabled || actor->interior != volume->interior) continue;
            if (actor->interior && actor->currentSpace != volume->currentSpace) continue;
            if (!actor->interior && actor->currentSpace.worldspace != volume->currentSpace.worldspace) continue;
            const double x = actor->transform.position[0] - volume->transform.position[0];
            const double y = actor->transform.position[1] - volume->transform.position[1];
            const double z = actor->transform.position[2] - volume->transform.position[2];
            const double yaw = volume->transform.rotationRadians[1];
            const double localX = std::cos(yaw)*x - std::sin(yaw)*z;
            const double localZ = std::sin(yaw)*x + std::cos(yaw)*z;
            const auto scale = volume->transform.scale;
            if (std::abs(localX) > trigger.halfExtents[0]*scale || std::abs(y) > trigger.halfExtents[2]*scale || std::abs(localZ) > trigger.halfExtents[1]*scale) continue;
            occupants.insert(actorId);
            if (!trigger.occupants.contains(actorId)) for (const auto& script : trigger.scripts) {
                std::string error;
                const std::array arguments{PapyrusValue::fromObject(actorId)};
                if (!m_papyrus.startFunctionOnObject(id, script, "OnTriggerEnter", arguments, error)) m_pendingDiagnostics.push_back(error);
            }
        }
        trigger.occupants = std::move(occupants);
    }
    const auto setLine = [&](SkyrimSceneSpeech& line, const SkyrimDialogueInfoDefinition& info) {
        const auto& response = info.responses[line.response];
        line.text = response.text;
        char key[40]; std::snprintf(key, sizeof(key), "info_%08X_%u", info.record.localFormId, response.responseNumber);
        line.voiceKey = key; line.sequence = ++m_sceneSpeechSequence;
        line.presented = false; line.remainingSeconds = 0;
    };
    for (auto it = m_sceneSpeech.begin(); it != m_sceneSpeech.end();) {
        if (!it->presented || (it->remainingSeconds -= seconds) > 0) { ++it; continue; }
        const auto info = m_dialogueInfos.find(it->responseInfo);
        if (info != m_dialogueInfos.end() && ++it->response < info->second.responses.size()) {
            setLine(*it, info->second); ++it; continue;
        }
        (void)selectDialogueInfo(it->info, it->speaker, m_playerObject, 2u, true, true);
        m_sceneProgress[it->scene].completed.insert(it->action);
        it = m_sceneSpeech.erase(it);
    }
    for (auto& [key, playing] : m_scenes) {
        if (!playing) continue;
        const auto found = m_sceneDefinitions.find(key);
        if (found == m_sceneDefinitions.end()) continue;
        const auto& scene = found->second;
        auto& state = m_sceneProgress[key];
        const auto* quest = findQuest(ObjectId::persistent(scene.quest));
        if (!quest) continue;
        const auto alias = [&](std::int32_t id) {
            for (const auto& value : quest->aliases) if (value.id == id) return value.target;
            return ObjectId{};
        };
        const auto fragment = [&](std::uint32_t phase, std::uint8_t flag) {
            for (const auto& value : scene.fragments) if (value.phase == phase && (value.flags & flag)) {
                std::string error;
                if (!m_papyrus.startFunctionOnObject(ObjectId::persistent(key), value.scriptClass, value.function, {}, error))
                    VOX_LOGW("scene") << scene.editorId << ": " << error;
            }
        };
        const auto conditions = [&](const std::vector<Condition>& values) {
            SkyrimDialogueInfoDefinition context; context.quest = scene.quest; context.conditions = values;
            const auto speaker = scene.actions.empty() ? m_playerObject : alias(scene.actions.front().actorAlias);
            return evaluateDialogueConditions(context, speaker, m_playerObject, true).matched;
        };
        if (!state.begun) { state.begun = true; fragment(0xffffffffu, 1u); }
        if (state.phase >= scene.phases.size()) { playing = false; fragment(0xffffffffu, 2u); continue; }
        const auto& phase = scene.phases[state.phase];
        if (!state.entered) {
            // A phase whose start conditions fail is skipped, as in the CK.
            if (!conditions(phase.startConditions)) { ++state.phase; continue; }
            state.entered = true; fragment(state.phase, 1u);
            VOX_LOGI("scene") << scene.editorId << " phase " << state.phase << ": " << phase.name;
        }
        bool complete = true;
        for (const auto& action : scene.actions) {
            if (action.startPhase > state.phase || action.endPhase < state.phase || state.completed.contains(action.index)) continue;
            const ObjectId speaker = alias(action.actorAlias);
            bool done = false;
            if (action.type == 2) {
                const auto [timer, inserted] = state.timers.try_emplace(action.index, action.seconds);
                timer->second -= seconds; done = timer->second <= 0;
            } else if (action.type == 0) {
                if (!action.topic.valid()) done = true;
                else if (std::none_of(m_sceneSpeech.begin(), m_sceneSpeech.end(), [&](const auto& line) { return line.scene == key && line.action == action.index; })) {
                    const auto* actor = m_world.find(speaker);
                    if (actor && actor->enabled) {
                        const auto choices = availableDialogueChoices(speaker, m_playerObject, true, std::span(&action.topic, 1));
                        if (choices.empty()) done = true;
                        else {
                            auto selected = selectDialogueInfo(choices.front().info, speaker, m_playerObject, 1u, true, true);
                            if (selected.accepted) {
                                const RecordKey response = selected.responseInfo.valid() ? selected.responseInfo : selected.info;
                                const auto info = m_dialogueInfos.find(response);
                                if (info == m_dialogueInfos.end() || info->second.responses.empty()) {
                                    (void)selectDialogueInfo(selected.info, speaker, m_playerObject, 2u, true, true);
                                    done = true;
                                }
                                else {
                                    SkyrimSceneSpeech line; line.scene = key; line.info = selected.info;
                                    line.responseInfo = response; line.speaker = speaker; line.action = action.index;
                                    setLine(line, info->second); m_sceneSpeech.push_back(std::move(line));
                                }
                            }
                        }
                    }
                }
            } else {
                const auto* actor = m_world.find(speaker);
                if (actor && !action.packages.empty()) {
                    const auto pack = m_scenePackages.find(action.packages.front());
                    if (pack != m_scenePackages.end()) {
                        (void)waitForEscort(pack->second, *actor);
                        const auto* target = m_world.find(pack->second.destination);
                        if (target) {
                            double distance = 0;
                            for (unsigned i=0;i<3;++i) { const double d = actor->transform.position[i]-target->transform.position[i]; distance += d*d; }
                            done = distance <= pack->second.radius*pack->second.radius;
                            if (!done && (!actor->navigationRequest || actor->navigationRequest->destination != target->id)) {
                                WorldCommand move; move.type = WorldCommandType::RequestMoveTo; move.target = speaker; move.destination = target->id;
                                (void)m_world.queue(std::move(move));
                            }
                        }
                    }
                }
            }
            if (done) state.completed.insert(action.index);
            else if (action.endPhase == state.phase) complete = false;
        }
        // Explicit completion conditions can finish a phase before a looping
        // or spanning action ends (notably IsSceneActionComplete).
        if (!phase.completionConditions.empty()) complete = conditions(phase.completionConditions);
        if (complete) { fragment(state.phase, 2u); ++state.phase; state.entered = false; }
        // A scene package temporarily overrides the actor's authored alias
        // stack. Resume the first eligible alias package between overrides.
        for (const auto& [actorAlias, packages] : scene.aliasPackages) {
            if (std::any_of(scene.actions.begin(), scene.actions.end(), [&](const auto& action) {
                return action.type == 1 && action.actorAlias == actorAlias && action.startPhase <= state.phase && action.endPhase >= state.phase;
            })) continue;
            const auto actorId = alias(actorAlias);
            const auto* actor = m_world.find(actorId);
            if (!actor || !actor->enabled) continue;
            for (const auto& packageId : packages) {
                const auto foundPackage = m_scenePackages.find(packageId);
                if (foundPackage == m_scenePackages.end()) continue;
                const auto& package = foundPackage->second;
                SkyrimDialogueInfoDefinition context; context.quest = scene.quest; context.conditions = package.conditions;
                if (!evaluateDialogueConditions(context, actorId, m_playerObject, true).matched) continue;
                const auto destination = package.destinationAlias >= 0 ? alias(package.destinationAlias) : package.destination;
                if (!destination.valid()) continue;
                (void)waitForEscort(package, *actor);
                if (!actor->navigationRequest || actor->navigationRequest->destination != destination) {
                    WorldCommand move; move.type = WorldCommandType::RequestMoveTo; move.target = actorId; move.destination = destination;
                    (void)m_world.queue(std::move(move));
                }
                break;
            }
        }
    }
}

void BethesdaSession::advanceFlyingPackages(double seconds) {
    for (const auto& [questId, aliases] : m_flyingAliasPackages) {
        const auto* quest = findQuest(ObjectId::persistent(questId));
        if (!quest || !quest->running) continue;
        for (const auto& [aliasId, packages] : aliases) {
            ObjectId actorId;
            for (const auto& alias : quest->aliases) if (alias.id == aliasId) actorId = alias.target;
            auto* actor = m_world.find(actorId);
            if (!actor || !actor->enabled) continue;
            for (const auto& key : packages) {
                const auto found = m_scenePackages.find(key);
                if (found == m_scenePackages.end()) continue;
                const auto& package = found->second;
                SkyrimDialogueInfoDefinition context; context.quest = questId; context.conditions = package.conditions;
                if (!evaluateDialogueConditions(context, actorId, m_playerObject, true).matched || !package.destination.valid()) continue;
                std::vector<ObjectId> path{package.destination};
                if (package.patrol) while (m_patrolLinks.contains(path.back())) {
                    const auto next = m_patrolLinks.at(path.back());
                    if (std::find(path.begin(), path.end(), next) != path.end()) break;
                    path.push_back(next);
                }
                auto destination = path.begin();
                if (actor->navigationRequest) {
                    const auto current = std::find(path.begin(), path.end(), actor->navigationRequest->destination);
                    if (current != path.end()) destination = current;
                }
                if (actor->navigationRequest && actor->navigationRequest->destination == *destination &&
                    actor->navigationRequest->status == NavigationRequestStatus::Arrived && destination+1 == path.end()) break;
                auto* target = m_world.find(*destination);
                if (!target) break;
                double distance = 0;
                for (unsigned i=0;i<3;++i) { const double d = target->transform.position[i]-actor->transform.position[i]; distance += d*d; }
                distance = std::sqrt(distance);
                // Flying actors use a coarse travel simulation along retail
                // patrol links, independent of ground NAVM residency.
                const double travel = 2000.0 * seconds;
                for (unsigned i=0;i<3;++i) actor->transform.position[i] += (target->transform.position[i]-actor->transform.position[i]) * (distance > 0 ? std::min(1.0, travel/distance) : 0);
                if (!actor->navigationRequest) actor->navigationRequest.emplace();
                auto& navigation = *actor->navigationRequest;
                navigation.destination = *destination; navigation.status = NavigationRequestStatus::Moving;
                if (distance <= travel + package.radius) {
                    if (destination+1 != path.end()) navigation.destination = *(destination+1);
                    else {
                        navigation.status = NavigationRequestStatus::Arrived;
                        std::size_t index = 0;
                        for (std::uint8_t bit : {1u, 2u, 4u}) if (package.scripts.flags & bit) {
                            const auto& fragment = package.scripts.fragments[index++];
                            if (bit != 2) continue;
                            std::string error; const std::array arguments{PapyrusValue::fromObject(actorId)};
                            (void)m_papyrus.startFunctionOnObject(ObjectId::persistent(package.record), fragment.scriptClass, fragment.function, arguments, error);
                        }
                    }
                }
                break;
            }
        }
    }
}
} // namespace odai::bethesda
