#include "bethesda/bethesda_session.h"

#include "core/hash.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <set>

namespace odai::bethesda {
namespace {

std::string normalizedEditorId(std::string value) {
    for (char& ch : value) {
        if (ch >= 'A' && ch <= 'Z') ch = static_cast<char>(ch - 'A' + 'a');
    }
    return value;
}

void hashString(std::uint64_t& hash, const std::string& text) {
    for (const unsigned char ch : text) {
        hash ^= ch;
        hash *= 1099511628211ull;
    }
    hash ^= 0xffu;
    hash *= 1099511628211ull;
}

template <typename T>
void hashScalar(std::uint64_t& hash, const T& value) {
    const auto* bytes = reinterpret_cast<const unsigned char*>(&value);
    for (std::size_t index = 0u; index < sizeof(T); ++index) {
        hash ^= bytes[index];
        hash *= 1099511628211ull;
    }
}

void hashPapyrusValue(std::uint64_t& hash, const PapyrusValue& value) {
    hashScalar(hash, value.type);
    switch (value.type) {
        case PapyrusValueType::None: break;
        case PapyrusValueType::Integer: hashScalar(hash, value.integer); break;
        case PapyrusValueType::Float: hashScalar(hash, value.real); break;
        case PapyrusValueType::Boolean: hashScalar(hash, value.boolean); break;
        case PapyrusValueType::String: hashString(hash, value.string); break;
        case PapyrusValueType::Object: hashString(hash, value.object.toString()); break;
        case PapyrusValueType::Array:
            hashScalar(hash, static_cast<std::uint64_t>(value.array.size()));
            for (const PapyrusValue& element : value.array) hashPapyrusValue(hash, element);
            break;
    }
}

void hashTes3Value(std::uint64_t& hash, const Tes3Value& value) {
    hashScalar(hash, value.type);
    switch (value.type) {
        case Tes3ValueType::None: break;
        case Tes3ValueType::Number: hashScalar(hash, value.number); break;
        case Tes3ValueType::String: hashString(hash, value.string); break;
        case Tes3ValueType::Object: hashString(hash, value.object.toString()); break;
    }
}

void hashPapyrusFrame(std::uint64_t& hash, const PapyrusCallFrameSnapshot& frame) {
    hashString(hash, frame.function);
    hashScalar(hash, static_cast<std::uint64_t>(frame.instruction));
    hashString(hash, frame.returnDestination);
    hashString(hash, frame.self.toString());
    hashString(hash, frame.scriptClass);
    std::vector<std::pair<std::string, PapyrusValue>> locals(
        frame.locals.begin(), frame.locals.end());
    std::sort(locals.begin(), locals.end(), [](const auto& left, const auto& right) {
        return left.first < right.first;
    });
    for (const auto& [name, value] : locals) {
        hashString(hash, name);
        hashPapyrusValue(hash, value);
    }
}

void hashBehaviorGraph(
    std::uint64_t& hash, const odai::anim::BehaviorGraphSnapshot& graph) {
    hashString(hash, graph.state);
    hashScalar(hash, graph.stateTime);
    hashScalar(hash, graph.fixedTick);
    hashScalar(hash, graph.wasGrounded);
    for (const odai::anim::AnimationEvent& event : graph.queuedEvents) {
        hashString(hash, event.name);
        hashString(hash, event.payload);
    }
}

}  // namespace

bool BethesdaSession::configure(BethesdaSessionConfig config, std::string& outError) {
    if (config.game == importer::fnv::BethesdaGame::Unknown) {
        outError = "Bethesda session requires a known game generation";
        return false;
    }
    if (config.contentFingerprint.empty()) {
        outError = "Bethesda session requires a content fingerprint";
        return false;
    }
    m_config = std::move(config);
    m_playerObject = m_config.playerObject;
    if (!m_playerObject.valid()) {
        if (m_config.game == importer::fnv::BethesdaGame::Morrowind) {
            m_playerObject = ObjectId::persistent(makeTes3RecordKey("NPC_", "player"));
        } else if (m_config.game == importer::fnv::BethesdaGame::SkyrimSpecialEdition) {
            m_playerObject = ObjectId::persistent(makeRecordKey("Skyrim.esm", 0x14u));
        }
    }
    m_config.playerObject = m_playerObject;
    m_randomState = m_config.randomSeed == 0u ? 1u : m_config.randomSeed;
    m_clock.reset();
    m_world.clear();
    m_physics.clear();
    m_actorAnimations.clear();
    m_pendingAnimationSnapshots.clear();
    m_pendingPhysicsSnapshots.clear();
    m_papyrus.clearRuntimeState();
    m_tes3.clear();
    m_quests.clear();
    m_questStageFragments.clear();
    m_dialogueTopics.clear();
    m_dialogueBranches.clear();
    m_dialogueInfos.clear();
    m_statistics.clear();
    m_discoveries.clear();
    m_scenes.clear();
    m_forcedWeather = {};
    m_locations.clear();
    m_globalVariables.clear();
    m_storyEvents.clear();
    m_giftMenuRequests.clear();
    m_scriptDebugLogs.clear();
    m_pendingDiagnostics.clear();
    m_pendingQuestAliasEvents.clear();
    m_resolvedFormResolver = {};
    m_nextStoryEventSequence = 1u;
    m_nextGiftMenuSequence = 1u;
    if (m_config.game == importer::fnv::BethesdaGame::SkyrimSpecialEdition) {
        if (!m_physics.initialize(outError)) return false;
        registerSkyrimNatives();
    }
    m_configured = true;
    if (!m_config.scenarioId.empty()) {
        const ScenarioDefinition* scenario = findScenario(m_config.scenarioId);
        if (scenario == nullptr || !applyScenario(*scenario, outError)) {
            m_configured = false;
            return false;
        }
    }
    outError.clear();
    return true;
}

bool BethesdaSession::configureTes3Content(
    std::shared_ptr<const Tes3ContentStore> content, std::string& outError) {
    if (!m_configured || m_config.game != importer::fnv::BethesdaGame::Morrowind) {
        outError = "TES3 content can only attach to a configured Morrowind session";
        return false;
    }
    if (!m_tes3.configure(std::move(content), m_playerObject, outError)) return false;
    m_tes3.setExternalNativeExecutor(
        [this](const Tes3NativeCall& call) { return executeTes3WorldNative(call); });
    return true;
}

bool BethesdaSession::registerActorAnimation(
    ObjectId object, std::shared_ptr<const odai::anim::AnimationView> thirdPerson,
    std::shared_ptr<const odai::anim::AnimationView> firstPerson,
    const PhysicsCharacterConfig& physicsConfig, std::string& outError) {
    if (!object.valid() || thirdPerson == nullptr) {
        outError = "actor animation registration requires ObjectId and third-person view";
        return false;
    }
    ActorAnimationRuntime runtime;
    runtime.thirdPersonView = std::move(thirdPerson);
    runtime.firstPersonView = std::move(firstPerson);
    if (!runtime.thirdPerson.bind(*runtime.thirdPersonView, outError)) return false;
    if (runtime.firstPersonView != nullptr &&
        !runtime.firstPerson.bind(*runtime.firstPersonView, outError)) return false;
    const bool hadController = m_physics.hasCharacter(object);
    if (!hadController && !registerActorController(object, physicsConfig, outError)) return false;
    const auto pending = m_pendingAnimationSnapshots.find(object);
    if (pending != m_pendingAnimationSnapshots.end()) {
        if (pending->second.firstPerson.has_value() != (runtime.firstPersonView != nullptr)) {
            outError = "pending first-person animation view does not match actor " +
                object.toString();
            if (!hadController) (void)unregisterActorController(object);
            return false;
        }
        if (!runtime.thirdPerson.restore(pending->second.thirdPerson, outError) ||
            (pending->second.firstPerson.has_value() && runtime.firstPersonView != nullptr &&
             !runtime.firstPerson.restore(*pending->second.firstPerson, outError))) {
            if (!hadController) (void)unregisterActorController(object);
            return false;
        }
        m_pendingAnimationSnapshots.erase(pending);
    }
    m_actorAnimations.insert_or_assign(object, std::move(runtime));
    outError.clear();
    return true;
}

bool BethesdaSession::registerActorController(
    ObjectId object, const PhysicsCharacterConfig& physicsConfig, std::string& outError) {
    if (m_physics.hasCharacter(object)) {
        outError = "actor controller is already registered: " + object.toString();
        return false;
    }
    if (!m_physics.addCharacter(object, physicsConfig, outError)) return false;
    const auto pending = m_pendingPhysicsSnapshots.find(object);
    if (pending != m_pendingPhysicsSnapshots.end()) {
        if (!m_physics.restoreCharacter(pending->second, outError)) {
            (void)m_physics.removeCharacter(object);
            return false;
        }
        m_pendingPhysicsSnapshots.erase(pending);
    }
    outError.clear();
    return true;
}

bool BethesdaSession::unregisterActorController(ObjectId object) {
    if (m_actorAnimations.contains(object)) return unregisterActorAnimation(object);
    const auto state = m_physics.characterState(object);
    if (!state.has_value()) return false;
    const auto snapshots = m_physics.snapshot();
    const auto saved = std::find_if(snapshots.begin(), snapshots.end(),
        [&](const PhysicsCharacterSnapshot& value) { return value.object == object; });
    if (saved != snapshots.end()) m_pendingPhysicsSnapshots.insert_or_assign(object, *saved);
    return m_physics.removeCharacter(object);
}

bool BethesdaSession::unregisterActorAnimation(ObjectId object) {
    const auto found = m_actorAnimations.find(object);
    if (found == m_actorAnimations.end()) return false;
    AnimationActorSnapshot saved{object, found->second.thirdPerson.snapshot(), std::nullopt};
    if (found->second.firstPersonView != nullptr) {
        saved.firstPerson = found->second.firstPerson.snapshot();
    }
    m_pendingAnimationSnapshots.insert_or_assign(object, std::move(saved));
    m_actorAnimations.erase(found);
    const auto physics = m_physics.snapshot();
    const auto physical = std::find_if(physics.begin(), physics.end(),
        [&](const PhysicsCharacterSnapshot& value) { return value.object == object; });
    if (physical != physics.end()) {
        m_pendingPhysicsSnapshots.insert_or_assign(object, *physical);
    }
    (void)m_physics.removeCharacter(object);
    return true;
}

bool BethesdaSession::setActorControllerInput(
    ObjectId object, const PhysicsCharacterInput& input) {
    return m_physics.setCharacterInput(object, input);
}

bool BethesdaSession::setActorAnimationInput(
    ObjectId object, odai::anim::AnimationInputState input) {
    const auto found = m_actorAnimations.find(object);
    if (found == m_actorAnimations.end()) return false;
    found->second.input = std::move(input);
    return true;
}

bool BethesdaSession::queueActorAnimationEvent(
    ObjectId object, odai::anim::AnimationEvent event) {
    const auto found = m_actorAnimations.find(object);
    if (found == m_actorAnimations.end()) return false;
    found->second.thirdPerson.queueEvent(event);
    if (found->second.firstPersonView != nullptr) found->second.firstPerson.queueEvent(std::move(event));
    return true;
}

MeleeAttackResult BethesdaSession::performMeleeAttack(
    ObjectId attacker,
    const odai::math::Vector3& forward,
    float damage,
    float rangeBethesdaUnits) {
    MeleeAttackResult result;
    RuntimeObject* source = m_world.find(attacker);
    if (source == nullptr || source->kind != RuntimeObjectKind::Actor ||
        !source->enabled || (source->actorValues.has_value() && source->actorValues->dead)) {
        result.diagnostic = "melee attacker is not a live resident actor";
        return result;
    }
    if (!m_physics.hasCharacter(attacker) || !std::isfinite(damage) || damage <= 0.0f ||
        !std::isfinite(rangeBethesdaUnits) || rangeBethesdaUnits <= 0.0f ||
        !std::isfinite(forward.x) || !std::isfinite(forward.y) ||
        !std::isfinite(forward.z) || odai::math::length(forward) <= 1.0e-5f) {
        result.diagnostic = "melee attack has invalid physical input";
        return result;
    }
    const std::uint64_t tick = m_clock.tick();
    RuntimeCombatState combat = source->combatState.value_or(RuntimeCombatState{});
    if (tick < combat.nextMeleeAttackTick) {
        result.diagnostic = "melee attack is on cooldown";
        return result;
    }
    constexpr float kStaminaCost = 10.0f;
    if (source->actorValues.has_value() && source->actorValues->stamina < kStaminaCost) {
        result.diagnostic = "melee attack requires stamina";
        return result;
    }

    result.accepted = true;
    result.damage = damage;
    ++combat.attacksStarted;
    combat.nextMeleeAttackTick = tick + 24u;  // 0.4 s at the fixed 60 Hz clock
    combat.lastTarget = {};
    for (const PhysicsMeleeCandidate& candidate :
         m_physics.meleeCandidates(attacker, forward, rangeBethesdaUnits)) {
        const RuntimeObject* target = m_world.find(candidate.object);
        if (target == nullptr || target->kind != RuntimeObjectKind::Actor ||
            !target->enabled || target->ghost ||
            (target->actorValues.has_value() && target->actorValues->dead)) {
            continue;
        }
        result.hit = true;
        result.target = candidate.object;
        combat.lastTarget = candidate.object;
        ++combat.hitsLanded;
        const float currentHealth = target->actorValues.has_value()
            ? target->actorValues->health : 100.0f;
        result.killed = currentHealth <= damage;
        WorldCommand hit;
        hit.type = WorldCommandType::AdjustActorValue;
        hit.target = candidate.object;
        hit.actorValue = ActorValue::Health;
        hit.actorValueDelta = -damage;
        (void)m_world.queue(std::move(hit));
        if (result.killed) {
            for (const auto& [questName, questState] : m_quests) {
                (void)questName;
                for (const QuestAliasRuntimeState& alias : questState.aliases) {
                    if (alias.target == candidate.object) {
                        queueQuestAliasEvent(alias.handle, "OnDeath",
                            {PapyrusValue::fromObject(attacker)});
                    }
                }
            }
        }
        break;
    }
    WorldCommand saveCombat;
    saveCombat.type = WorldCommandType::SetCombatState;
    saveCombat.target = attacker;
    saveCombat.combatState = combat;
    (void)m_world.queue(std::move(saveCombat));
    WorldCommand spendStamina;
    spendStamina.type = WorldCommandType::AdjustActorValue;
    spendStamina.target = attacker;
    spendStamina.actorValue = ActorValue::Stamina;
    spendStamina.actorValueDelta = -kStaminaCost;
    (void)m_world.queue(std::move(spendStamina));

    const auto animated = m_actorAnimations.find(attacker);
    if (animated != m_actorAnimations.end()) {
        animated->second.input.weaponDrawn = true;
        animated->second.input.attacking = true;
        animated->second.thirdPerson.queueEvent({"weaponSwing", "right"});
        if (animated->second.firstPersonView != nullptr) {
            animated->second.firstPerson.queueEvent({"weaponSwing", "right"});
        }
    }
    return result;
}

bool BethesdaSession::rotatePuzzleRing(
    ObjectId door, std::size_t ringIndex, std::string& outError) {
    const RuntimeObject* object = m_world.find(door);
    if (object == nullptr || !object->activatorState.has_value() ||
        object->activatorState->opened ||
        ringIndex >= object->activatorState->puzzleStates.size() ||
        object->activatorState->puzzleStateCount <= 0) {
        outError = "puzzle ring rotation requires a configured closed activator";
        return false;
    }
    RuntimeActivatorState state = *object->activatorState;
    state.puzzleStates[ringIndex] =
        (state.puzzleStates[ringIndex] % state.puzzleStateCount) + 1;
    WorldCommand command;
    command.type = WorldCommandType::SetActivatorState;
    command.target = door;
    command.activatorState = std::move(state);
    (void)m_world.queue(std::move(command));
    outError.clear();
    return true;
}

PuzzleDoorActivationResult BethesdaSession::activatePuzzleDoor(
    ObjectId player,
    ObjectId door,
    const RecordKey& requiredItem,
    const RecordKey& questRecord,
    std::int32_t successStage) {
    PuzzleDoorActivationResult result;
    const RuntimeObject* playerObject = m_world.find(player);
    const RuntimeObject* doorObject = m_world.find(door);
    if (playerObject == nullptr || doorObject == nullptr ||
        playerObject->kind != RuntimeObjectKind::Actor ||
        !doorObject->activatorState.has_value() || !requiredItem.valid() ||
        !questRecord.valid() || successStage < 0) {
        result.diagnostic = "puzzle activation has incomplete stable runtime data";
        return result;
    }
    result.accepted = true;
    RuntimeActivatorState state = *doorObject->activatorState;
    ++state.activationCount;
    const auto item = std::find_if(playerObject->inventory.begin(),
        playerObject->inventory.end(), [&](const InventoryEntry& entry) {
            return entry.item == requiredItem && entry.count > 0;
        });
    if (item == playerObject->inventory.end()) {
        result.missingRequiredItem = true;
    } else if (state.puzzleStates != state.puzzleSolution) {
        result.incorrectCombination = true;
    } else {
        state.opened = true;
        result.opened = true;
        QuestRuntimeState* questState = findQuest(ObjectId::persistent(questRecord));
        if (questState == nullptr) {
            result.opened = false;
            result.diagnostic = "puzzle success quest is not registered";
            return result;
        }
        setQuestStage(questState->editorId, successStage);
    }
    WorldCommand command;
    command.type = WorldCommandType::SetActivatorState;
    command.target = door;
    command.activatorState = std::move(state);
    (void)m_world.queue(std::move(command));
    return result;
}

std::size_t BethesdaSession::bindQuestInventoryForActor(
    ObjectId actor, const RecordKey& actorBase, std::string& outError) {
    RuntimeObject* actorObject = m_world.find(actor);
    if (actorObject == nullptr || actorObject->kind != RuntimeObjectKind::Actor ||
        !actorBase.valid()) {
        outError = "quest inventory binding requires a resident actor and stable base";
        return 0u;
    }
    (void)bindDynamicQuestAliasesForObject(actor, outError);
    if (!outError.empty()) return 0u;
    std::size_t materialized = 0u;
    for (auto& [questName, questState] : m_quests) {
        (void)questName;
        // Unique-actor aliases are initially resolved to the NPC_ base RecordKey.
        // Promote every matching alias to the placed runtime actor as soon as
        // that actor becomes resident; CTDA run-on QuestAlias then observes
        // live death/inventory state rather than the immutable base record.
        for (QuestAliasRuntimeState& alias : questState.aliases) {
            if (alias.target.kind == ObjectIdKind::PersistentReference &&
                alias.target.reference == actorBase) {
                alias.target = actor;
            }
        }
        std::vector<QuestAliasRuntimeState*> boundOwners;
        for (QuestAliasRuntimeState& created : questState.aliases) {
            if (!created.createdObject.valid() ||
                created.createdInAliasId < 0 ||
                created.createdObjectMaterialized) {
                continue;
            }
            const auto owner = std::find_if(
                questState.aliases.begin(), questState.aliases.end(),
                [&](const QuestAliasRuntimeState& candidate) {
                    if (candidate.id != created.createdInAliasId) return false;
                    return candidate.target == actor ||
                        (candidate.target.kind == ObjectIdKind::PersistentReference &&
                         candidate.target.reference == actorBase);
                });
            if (owner == questState.aliases.end()) continue;
            WorldCommand add;
            add.type = WorldCommandType::AddItem;
            add.target = actor;
            add.item = created.createdObject;
            add.itemCount = 1;
            (void)m_world.queue(std::move(add));
            RuntimeObject itemInstance;
            itemInstance.id = m_world.allocateRuntimeId();
            itemInstance.base = created.createdObject;
            itemInstance.kind = RuntimeObjectKind::Item;
            itemInstance.enabled = false;
            itemInstance.persistent = true;
            WorldCommand spawn;
            spawn.type = WorldCommandType::Spawn;
            spawn.object = itemInstance;
            (void)m_world.queue(std::move(spawn));
            created.target = itemInstance.id;
            created.createdObjectMaterialized = true;
            boundOwners.push_back(&*owner);
            queueQuestAliasEvent(created.handle, "OnContainerChanged",
                {PapyrusValue::fromObject(actor), PapyrusValue{}});
            ++materialized;
        }
        for (QuestAliasRuntimeState* owner : boundOwners) {
            owner->target = actor;
        }
    }
    outError.clear();
    return materialized;
}

std::size_t BethesdaSession::bindDynamicQuestAliasesForObject(
    ObjectId object, std::string& outError) {
    const RuntimeObject* candidate = m_world.find(object);
    if (candidate == nullptr) {
        outError = "dynamic quest alias binding requires a resident object";
        return 0u;
    }
    std::size_t bound = 0u;
    for (auto& [questName, questState] : m_quests) {
        (void)questName;
        for (QuestAliasRuntimeState& alias : questState.aliases) {
            if (alias.location || alias.findMatchingReferenceInAliasId < 0 ||
                !alias.referenceType.valid()) continue;
            const auto locationAlias = std::find_if(
                questState.aliases.begin(), questState.aliases.end(),
                [&](const QuestAliasRuntimeState& value) {
                    return value.id == alias.findMatchingReferenceInAliasId && value.location;
                });
            if (locationAlias == questState.aliases.end() ||
                locationAlias->target.kind != ObjectIdKind::PersistentReference ||
                candidate->location != locationAlias->target.reference ||
                std::find(candidate->referenceTypes.begin(),
                    candidate->referenceTypes.end(), alias.referenceType) ==
                    candidate->referenceTypes.end()) continue;
            if (alias.target.valid()) {
                if (alias.target != object) {
                    outError = "ambiguous dynamic quest alias " + questState.editorId + ":" +
                        std::to_string(alias.id) + " matches both " +
                        alias.target.toString() + " and " + object.toString();
                    return bound;
                }
                continue;
            }
            alias.target = object;
            ++bound;
        }
    }
    outError.clear();
    return bound;
}

LootTransferResult BethesdaSession::lootObject(ObjectId player, ObjectId source) {
    LootTransferResult result;
    const RuntimeObject* playerObject = m_world.find(player);
    const RuntimeObject* sourceObject = m_world.find(source);
    if (playerObject == nullptr || playerObject->kind != RuntimeObjectKind::Actor ||
        sourceObject == nullptr ||
        (sourceObject->kind != RuntimeObjectKind::Actor &&
         sourceObject->kind != RuntimeObjectKind::Container)) {
        result.diagnostic = "looting requires a resident player and actor/container source";
        return result;
    }
    if (sourceObject->kind == RuntimeObjectKind::Actor &&
        (!sourceObject->actorValues.has_value() || !sourceObject->actorValues->dead)) {
        result.diagnostic = "living actors cannot be looted";
        return result;
    }
    result.accepted = true;
    for (const InventoryEntry& entry : sourceObject->inventory) {
        if (!entry.item.valid() || entry.count <= 0) continue;
        result.transferred.push_back({entry.item, entry.count, false});
    }
    std::sort(result.transferred.begin(), result.transferred.end(),
        [](const InventoryEntry& left, const InventoryEntry& right) {
            return left.item < right.item;
        });
    for (const InventoryEntry& entry : result.transferred) {
        WorldCommand remove;
        remove.type = WorldCommandType::RemoveItem;
        remove.target = source;
        remove.item = entry.item;
        remove.itemCount = entry.count;
        (void)m_world.queue(std::move(remove));
        WorldCommand add;
        add.type = WorldCommandType::AddItem;
        add.target = player;
        add.item = entry.item;
        add.itemCount = entry.count;
        (void)m_world.queue(std::move(add));
        for (const auto& [questName, questState] : m_quests) {
            (void)questName;
            for (const QuestAliasRuntimeState& alias : questState.aliases) {
                if (alias.createdObject == entry.item) {
                    queueQuestAliasEvent(alias.handle, "OnContainerChanged",
                        {PapyrusValue::fromObject(player),
                         PapyrusValue::fromObject(source)});
                }
            }
        }
    }
    if (result.transferred.empty()) result.diagnostic = "Nothing to take";
    return result;
}

GiftTransferResult BethesdaSession::transferGiftMenuItem(
    std::uint64_t sequence,
    const RecordKey& item,
    std::int32_t count) {
    GiftTransferResult result;
    const auto request = std::find_if(
        m_giftMenuRequests.begin(), m_giftMenuRequests.end(),
        [&](const GiftMenuRequestState& value) { return value.sequence == sequence; });
    if (request == m_giftMenuRequests.end() || !item.valid() || count <= 0) {
        result.diagnostic = "gift transfer requires an open request, item, and positive count";
        return result;
    }
    const ObjectId source = request->playerGives ? request->player : request->actor;
    const ObjectId destination = request->playerGives ? request->actor : request->player;
    const RuntimeObject* sourceObject = m_world.find(source);
    const RuntimeObject* destinationObject = m_world.find(destination);
    if (sourceObject == nullptr || destinationObject == nullptr ||
        sourceObject->kind != RuntimeObjectKind::Actor ||
        destinationObject->kind != RuntimeObjectKind::Actor) {
        result.diagnostic = "gift transfer participants are not resident actors";
        return result;
    }
    if (request->filterList.valid()) {
        result.diagnostic =
            "gift FormList filtering is not registered for this content closure";
        return result;
    }
    const auto owned = std::find_if(
        sourceObject->inventory.begin(), sourceObject->inventory.end(),
        [&](const InventoryEntry& entry) {
            return entry.item == item && entry.count >= count;
        });
    if (owned == sourceObject->inventory.end()) {
        result.diagnostic = "gift source does not own the requested visible quantity";
        return result;
    }
    // A FormList is retained for deterministic UI filtering. Its member
    // closure is content-owned and must be registered before a filtered
    // transfer is authorized; an unresolved list never broadens the menu.
    WorldCommand remove;
    remove.type = WorldCommandType::RemoveItem;
    remove.target = source;
    remove.item = item;
    remove.itemCount = count;
    (void)m_world.queue(std::move(remove));
    WorldCommand add;
    add.type = WorldCommandType::AddItem;
    add.target = destination;
    add.item = item;
    add.itemCount = count;
    (void)m_world.queue(std::move(add));
    result.accepted = true;
    return result;
}

bool BethesdaSession::closeGiftMenu(std::uint64_t sequence, std::string& outError) {
    const auto request = std::find_if(
        m_giftMenuRequests.begin(), m_giftMenuRequests.end(),
        [&](const GiftMenuRequestState& value) { return value.sequence == sequence; });
    if (request == m_giftMenuRequests.end()) {
        outError = "gift menu request is not open";
        return false;
    }
    m_giftMenuRequests.erase(request);
    outError.clear();
    return true;
}

bool BethesdaSession::registerDialogueTopic(
    SkyrimDialogueTopicDefinition definition, std::string& outError) {
    if (!definition.record.valid()) {
        outError = "dialogue topic requires a stable RecordKey";
        return false;
    }
    const auto [found, inserted] = m_dialogueTopics.insert_or_assign(
        definition.record, std::move(definition));
    (void)found;
    (void)inserted;
    outError.clear();
    return true;
}

bool BethesdaSession::registerDialogueBranch(
    SkyrimDialogueBranchDefinition definition, std::string& outError) {
    if (!definition.record.valid() || !definition.quest.valid() ||
        !definition.startingTopic.valid()) {
        outError = "dialogue branch requires stable record, quest, and starting-topic identities";
        return false;
    }
    m_dialogueBranches.insert_or_assign(definition.record, std::move(definition));
    outError.clear();
    return true;
}

bool BethesdaSession::registerDialogueInfo(
    SkyrimDialogueInfoDefinition definition, std::string& outError) {
    if (!definition.record.valid() || !definition.topic.valid() ||
        !definition.quest.valid()) {
        outError = "dialogue INFO requires stable record, topic, and quest identities";
        return false;
    }
    if (!m_dialogueTopics.contains(definition.topic) ||
        findQuest(ObjectId::persistent(definition.quest)) == nullptr) {
        outError = "dialogue INFO names an unregistered topic or quest";
        return false;
    }
    m_dialogueInfos.insert_or_assign(definition.record, std::move(definition));
    outError.clear();
    return true;
}

ConditionEvaluation BethesdaSession::evaluateDialogueConditions(
    const SkyrimDialogueInfoDefinition& info,
    ObjectId speaker,
    ObjectId player,
    bool strict) const {
    const auto liveObject = [&](ObjectId id) -> const RuntimeObject* {
        if (const RuntimeObject* direct = m_world.find(id)) return direct;
        if (id.kind != ObjectIdKind::PersistentReference) return nullptr;
        const std::vector<RuntimeObject> objects = m_world.orderedObjects();
        const auto found = std::find_if(objects.begin(), objects.end(), [&](const auto& object) {
            return object.base == id.reference;
        });
        if (found == objects.end()) return nullptr;
        // orderedObjects is a copy; only scalar reads occur inside this call,
        // but returning a pointer to it would dangle. Resolve the real object.
        return m_world.find(found->id);
    };
    const auto sameRuntimeIdentity = [&](ObjectId actual, ObjectId expected) {
        if (actual == expected) return true;
        const RuntimeObject* object = liveObject(actual);
        return object != nullptr && expected.kind == ObjectIdKind::PersistentReference &&
            object->base == expected.reference;
    };
    const auto resolveForm = [&](std::uint32_t formId) -> std::optional<ObjectId> {
        if (formId == 0u || !m_resolvedFormResolver) return std::nullopt;
        return m_resolvedFormResolver(formId);
    };
    const auto targetForCondition = [&](const Condition& condition) -> ObjectId {
        if (condition.runOn == 0u) return speaker;
        if (condition.runOn == 2u) {
            const std::optional<ObjectId> resolved = resolveForm(condition.reference);
            if (resolved.has_value() &&
                (sameRuntimeIdentity(player, *resolved) || player == *resolved)) return player;
            return resolved.value_or(ObjectId{});
        }
        if (condition.runOn == 5u) {
            const QuestRuntimeState* questState = findQuest(ObjectId::persistent(info.quest));
            if (questState == nullptr) return {};
            const auto alias = std::find_if(
                questState->aliases.begin(), questState->aliases.end(), [&](const auto& value) {
                    return value.id == static_cast<std::int32_t>(condition.reference);
                });
            return alias == questState->aliases.end() ? ObjectId{} : alias->target;
        }
        return {};
    };
    return evaluateConditions(info.conditions, [&](const Condition& condition)
        -> std::optional<float> {
        const ObjectId target = targetForCondition(condition);
        const RuntimeObject* targetObject = liveObject(target);
        switch (condition.function) {
            case 46u:  // GetDead
                if (targetObject == nullptr || !targetObject->actorValues.has_value()) {
                    return 0.0f;
                }
                return targetObject->actorValues->dead ? 1.0f : 0.0f;
            case 47u: {  // GetItemCount
                if (targetObject == nullptr) return 0.0f;
                const std::optional<ObjectId> item = resolveForm(condition.parameter1);
                if (!item.has_value() ||
                    item->kind != ObjectIdKind::PersistentReference) return std::nullopt;
                const auto entry = std::find_if(
                    targetObject->inventory.begin(), targetObject->inventory.end(),
                    [&](const InventoryEntry& value) { return value.item == item->reference; });
                return entry == targetObject->inventory.end()
                    ? 0.0f : static_cast<float>(entry->count);
            }
            case 58u: {  // GetStage
                const std::optional<ObjectId> questObject = resolveForm(condition.parameter1);
                const QuestRuntimeState* questState = questObject.has_value()
                    ? findQuest(*questObject) : nullptr;
                return questState == nullptr
                    ? std::optional<float>{} : static_cast<float>(questState->stage);
            }
            case 59u: {  // GetStageDone
                const std::optional<ObjectId> questObject = resolveForm(condition.parameter1);
                const QuestRuntimeState* questState = questObject.has_value()
                    ? findQuest(*questObject) : nullptr;
                if (questState == nullptr) return std::nullopt;
                return std::find(questState->completedStages.begin(),
                    questState->completedStages.end(),
                    static_cast<std::int32_t>(condition.parameter2)) !=
                    questState->completedStages.end() ? 1.0f : 0.0f;
            }
            case 72u: {  // GetIsID (TES5 INFO table)
                const std::optional<ObjectId> expected = resolveForm(condition.parameter1);
                if (!expected.has_value()) return std::nullopt;
                return sameRuntimeIdentity(target, *expected) ? 1.0f : 0.0f;
            }
            case 84u: {  // GetDeadCount
                const std::optional<ObjectId> expected = resolveForm(condition.parameter1);
                if (!expected.has_value() ||
                    expected->kind != ObjectIdKind::PersistentReference) return std::nullopt;
                std::size_t dead = 0u;
                for (const RuntimeObject& object : m_world.orderedObjects()) {
                    if (object.base == expected->reference && object.actorValues.has_value() &&
                        object.actorValues->dead) ++dead;
                }
                return static_cast<float>(dead);
            }
            case 403u: {  // GetRelationshipRank
                if (targetObject == nullptr) return 0.0f;
                const std::optional<ObjectId> other = resolveForm(condition.parameter1);
                if (!other.has_value()) return std::nullopt;
                const auto rank = std::find_if(
                    targetObject->relationships.begin(), targetObject->relationships.end(),
                    [&](const RelationshipRank& relationship) {
                        return relationship.other == *other ||
                            sameRuntimeIdentity(relationship.other, *other);
                    });
                return rank == targetObject->relationships.end()
                    ? 0.0f : static_cast<float>(rank->rank);
            }
            case 566u: {  // GetIsAliasRef
                const QuestRuntimeState* questState = findQuest(
                    ObjectId::persistent(info.quest));
                if (questState == nullptr) return std::nullopt;
                const auto alias = std::find_if(
                    questState->aliases.begin(), questState->aliases.end(),
                    [&](const QuestAliasRuntimeState& value) {
                        return value.id == static_cast<std::int32_t>(condition.parameter1);
                    });
                if (alias == questState->aliases.end()) return std::nullopt;
                return sameRuntimeIdentity(target, alias->target) ? 1.0f : 0.0f;
            }
            case 629u: {  // GetVMQuestVariable
                const std::optional<ObjectId> questObject = resolveForm(condition.parameter1);
                if (!questObject.has_value() || condition.stringParameter2.empty()) {
                    return std::nullopt;
                }
                const PapyrusValue* value =
                    m_papyrus.findProperty(*questObject, condition.stringParameter2);
                if (value == nullptr) return std::nullopt;
                switch (value->type) {
                    case PapyrusValueType::Boolean: return value->boolean ? 1.0f : 0.0f;
                    case PapyrusValueType::Integer:
                        return static_cast<float>(value->integer);
                    case PapyrusValueType::Float: return static_cast<float>(value->real);
                    default: return std::nullopt;
                }
            }
            default: return std::nullopt;
        }
    }, strict);
}

std::vector<SkyrimDialogueChoice> BethesdaSession::availableDialogueChoices(
    ObjectId speaker, ObjectId player, bool strict,
    std::span<const RecordKey> eligibleTopics) const {
    std::vector<SkyrimDialogueChoice> choices;
    std::vector<RecordKey> topicRecords;
    if (!eligibleTopics.empty()) {
        topicRecords.assign(eligibleTopics.begin(), eligibleTopics.end());
    } else if (!m_dialogueBranches.empty()) {
        topicRecords.reserve(m_dialogueBranches.size());
        for (const auto& [branchRecord, branch] : m_dialogueBranches) {
            (void)branchRecord;
            if (branch.startingTopic.valid()) topicRecords.push_back(branch.startingTopic);
        }
    } else {
        topicRecords.reserve(m_dialogueTopics.size());
        for (const auto& [topicRecord, topic] : m_dialogueTopics) {
            (void)topic;
            topicRecords.push_back(topicRecord);
        }
    }
    std::sort(topicRecords.begin(), topicRecords.end());
    topicRecords.erase(std::unique(topicRecords.begin(), topicRecords.end()), topicRecords.end());
    for (const RecordKey& topicRecord : topicRecords) {
        const auto topicFound = m_dialogueTopics.find(topicRecord);
        if (topicFound == m_dialogueTopics.end()) continue;
        const SkyrimDialogueTopicDefinition& topic = topicFound->second;
        if (topic.prompt.empty()) continue;
        std::vector<const SkyrimDialogueInfoDefinition*> authoredInfos;
        for (const auto& [infoRecord, info] : m_dialogueInfos) {
            (void)infoRecord;
            if (info.topic == topicRecord) authoredInfos.push_back(&info);
        }
        std::sort(authoredInfos.begin(), authoredInfos.end(), [](const auto* left, const auto* right) {
            if (left->authoredOrder != right->authoredOrder) {
                return left->authoredOrder < right->authoredOrder;
            }
            return left->record < right->record;
        });
        for (const SkyrimDialogueInfoDefinition* candidate : authoredInfos) {
            const SkyrimDialogueInfoDefinition& info = *candidate;
            const ConditionEvaluation evaluation =
                evaluateDialogueConditions(info, speaker, player, strict);
            if (!evaluation.matched) continue;
            SkyrimDialogueChoice choice;
            choice.info = info.record;
            choice.topic = topicRecord;
            choice.quest = info.quest;
            choice.branch = topic.branch;
            choice.prompt = info.prompt.empty() ? topic.prompt : info.prompt;
            const SkyrimDialogueInfoDefinition* response = &info;
            if (response->responses.empty() && info.responseInfo.valid()) {
                const auto linked = m_dialogueInfos.find(info.responseInfo);
                if (linked != m_dialogueInfos.end()) response = &linked->second;
            }
            for (const auto& line : response->responses) {
                if (!line.text.empty()) choice.responses.push_back(line.text);
            }
            choices.push_back(std::move(choice));
            break;  // one authored winning INFO variant per player topic
        }
    }
    return choices;
}

SkyrimDialogueSelectionResult BethesdaSession::selectDialogueInfo(
    const RecordKey& infoRecord,
    ObjectId speaker,
    ObjectId player,
    std::uint8_t fragmentFlag,
    bool strict) {
    SkyrimDialogueSelectionResult result;
    result.info = infoRecord;
    const auto found = m_dialogueInfos.find(infoRecord);
    if (found == m_dialogueInfos.end()) {
        result.diagnostics.push_back("dialogue INFO is not registered: " + infoRecord.toString());
        return result;
    }
    const SkyrimDialogueInfoDefinition& selected = found->second;
    const ConditionEvaluation evaluation =
        evaluateDialogueConditions(selected, speaker, player, strict);
    result.diagnostics = evaluation.diagnostics;
    if (!evaluation.matched) {
        result.diagnostics.push_back("dialogue INFO conditions did not match");
        return result;
    }
    const auto startFragments = [&](const SkyrimDialogueInfoDefinition& info) {
        if (fragmentFlag == 0u || (fragmentFlag & (fragmentFlag - 1u)) != 0u ||
            (info.scripts.flags & fragmentFlag) == 0u) return;
        const std::size_t index = static_cast<std::size_t>(
            std::popcount(static_cast<unsigned>(info.scripts.flags & (fragmentFlag - 1u))));
        if (index >= info.scripts.fragments.size()) {
            result.diagnostics.push_back("INFO VMAD fragment flags do not match fragment list");
            return;
        }
        const VmadInfoFragment& fragment = info.scripts.fragments[index];
        std::string error;
        const std::vector<PapyrusValue> arguments{PapyrusValue::fromObject(speaker)};
        if (m_papyrus.startFunctionOnObject(
                ObjectId::persistent(info.record), fragment.scriptClass,
                fragment.function, arguments, error) == 0u) {
            result.diagnostics.push_back("could not start INFO fragment " +
                fragment.scriptClass + "." + fragment.function + ": " + error);
        }
    };
    startFragments(selected);
    const SkyrimDialogueInfoDefinition* response = &selected;
    if (selected.responses.empty() && selected.responseInfo.valid()) {
        result.responseInfo = selected.responseInfo;
        const auto linked = m_dialogueInfos.find(selected.responseInfo);
        if (linked != m_dialogueInfos.end()) {
            response = &linked->second;
            startFragments(*response);
        } else {
            result.diagnostics.push_back("dialogue DNAM response INFO is missing");
        }
    }
    for (const SkyrimDialogueResponseDefinition& line : response->responses) {
        if (!line.text.empty()) result.responses.push_back(line.text);
    }
    result.nextTopics = selected.linkedTopics;
    if (response != &selected && result.nextTopics.empty()) {
        result.nextTopics = response->linkedTopics;
    }
    WorldCommand speakerContext;
    speakerContext.type = WorldCommandType::SetActorContext;
    speakerContext.target = speaker;
    speakerContext.inDialogueWithPlayer = true;
    (void)m_world.queue(std::move(speakerContext));
    result.accepted = result.diagnostics.empty();
    return result;
}

const odai::anim::AnimationStepOutput* BethesdaSession::actorAnimationOutput(
    ObjectId object, bool firstPerson) const {
    const auto found = m_actorAnimations.find(object);
    if (found == m_actorAnimations.end()) return nullptr;
    if (firstPerson && found->second.firstPersonView != nullptr) return &found->second.firstPersonOutput;
    return &found->second.thirdPersonOutput;
}

std::vector<AnimationActorSnapshot> BethesdaSession::animationSnapshots() const {
    std::vector<AnimationActorSnapshot> result;
    result.reserve(m_actorAnimations.size() + m_pendingAnimationSnapshots.size());
    for (const auto& [object, runtime] : m_actorAnimations) {
        AnimationActorSnapshot saved{object, runtime.thirdPerson.snapshot(), std::nullopt};
        if (runtime.firstPersonView != nullptr) saved.firstPerson = runtime.firstPerson.snapshot();
        result.push_back(std::move(saved));
    }
    for (const auto& [object, saved] : m_pendingAnimationSnapshots) {
        if (!m_actorAnimations.contains(object)) result.push_back(saved);
    }
    std::sort(result.begin(), result.end(), [](const auto& left, const auto& right) {
        return left.object < right.object;
    });
    return result;
}

bool BethesdaSession::restoreAnimationSnapshots(
    std::span<const AnimationActorSnapshot> snapshots, std::string& outError) {
    std::set<ObjectId> seen;
    for (const AnimationActorSnapshot& saved : snapshots) {
        if (!saved.object.valid() || !seen.insert(saved.object).second ||
            saved.thirdPerson.stateTime < 0.0f ||
            !std::isfinite(saved.thirdPerson.stateTime) ||
            (saved.firstPerson.has_value() &&
             (saved.firstPerson->stateTime < 0.0f ||
              !std::isfinite(saved.firstPerson->stateTime)))) {
            outError = "saved animation actor is invalid or duplicated: " +
                saved.object.toString();
            return false;
        }
    }
    m_pendingAnimationSnapshots.clear();
    for (const AnimationActorSnapshot& saved : snapshots) {
        m_pendingAnimationSnapshots.emplace(saved.object, saved);
    }
    std::vector<ObjectId> remove;
    for (auto& [object, runtime] : m_actorAnimations) {
        const auto saved = m_pendingAnimationSnapshots.find(object);
        if (saved == m_pendingAnimationSnapshots.end()) {
            remove.push_back(object);
            continue;
        }
        if (saved->second.firstPerson.has_value() != (runtime.firstPersonView != nullptr) ||
            !runtime.thirdPerson.restore(saved->second.thirdPerson, outError) ||
            (saved->second.firstPerson.has_value() &&
             !runtime.firstPerson.restore(*saved->second.firstPerson, outError))) return false;
        m_pendingAnimationSnapshots.erase(saved);
    }
    for (const ObjectId& object : remove) {
        m_actorAnimations.erase(object);
        (void)m_physics.removeCharacter(object);
    }
    outError.clear();
    return true;
}

std::vector<PhysicsCharacterSnapshot> BethesdaSession::physicsSnapshots() const {
    std::vector<PhysicsCharacterSnapshot> result = m_physics.snapshot();
    result.reserve(result.size() + m_pendingPhysicsSnapshots.size());
    for (const auto& [object, saved] : m_pendingPhysicsSnapshots) {
        if (!m_physics.hasCharacter(object)) result.push_back(saved);
    }
    std::sort(result.begin(), result.end(), [](const auto& left, const auto& right) {
        return left.object < right.object;
    });
    return result;
}

bool BethesdaSession::restorePhysicsSnapshots(
    std::span<const PhysicsCharacterSnapshot> snapshots, std::string& outError) {
    std::set<ObjectId> seen;
    for (const PhysicsCharacterSnapshot& saved : snapshots) {
        if (!saved.object.valid() || !seen.insert(saved.object).second ||
            !std::isfinite(saved.position.x) || !std::isfinite(saved.position.y) ||
            !std::isfinite(saved.position.z) || !std::isfinite(saved.rotation.x) ||
            !std::isfinite(saved.rotation.y) || !std::isfinite(saved.rotation.z) ||
            !std::isfinite(saved.rotation.w) || !std::isfinite(saved.velocity.x) ||
            !std::isfinite(saved.velocity.y) || !std::isfinite(saved.velocity.z) ||
            !std::isfinite(saved.groundNormal.x) || !std::isfinite(saved.groundNormal.y) ||
            !std::isfinite(saved.groundNormal.z)) {
            outError = "saved physical actor is invalid or duplicated: " +
                saved.object.toString();
            return false;
        }
    }
    m_pendingPhysicsSnapshots.clear();
    for (const PhysicsCharacterSnapshot& saved : snapshots) {
        m_pendingPhysicsSnapshots.emplace(saved.object, saved);
    }
    const std::vector<PhysicsCharacterSnapshot> active = m_physics.snapshot();
    for (const PhysicsCharacterSnapshot& current : active) {
        const auto saved = m_pendingPhysicsSnapshots.find(current.object);
        if (saved == m_pendingPhysicsSnapshots.end()) {
            (void)m_physics.removeCharacter(current.object);
            m_actorAnimations.erase(current.object);
            continue;
        }
        if (!m_physics.restoreCharacter(saved->second, outError)) return false;
        m_pendingPhysicsSnapshots.erase(saved);
    }
    outError.clear();
    return true;
}

BethesdaSessionStep BethesdaSession::advance(
    double frameDeltaSeconds, const BeforeSimulationTick& beforeTick) {
    BethesdaSessionStep result;
    result.diagnostics = std::move(m_pendingDiagnostics);
    m_pendingDiagnostics.clear();
    if (!m_configured) {
        result.diagnostics.push_back("Bethesda session is not configured");
        return result;
    }
    result.clock = m_clock.advance(frameDeltaSeconds,
        [&](std::uint64_t tick, double stepSeconds) {
            if (beforeTick) beforeTick(tick, stepSeconds);
            simulateTick(tick, stepSeconds, result);
        });
    if (result.clock.droppedSteps != 0u) {
        result.diagnostics.push_back(
            "simulation catch-up cap dropped " + std::to_string(result.clock.droppedSteps) + " steps");
    }
    return result;
}

bool BethesdaSession::applyScenario(const ScenarioDefinition& scenario, std::string& outError) {
    if (scenario.game != m_config.game) {
        outError = "scenario " + scenario.id + " targets " +
            importer::fnv::bethesdaGameName(scenario.game) + ", not " +
            importer::fnv::bethesdaGameName(m_config.game);
        return false;
    }
    m_config.scenarioId = scenario.id;
    for (const ScenarioQuestRecord& record : scenario.questRecords) {
        quest(record.editorId).record = makeRecordKey(record.plugin, record.localFormId);
    }
    for (const ScenarioQuestSeed& seed : scenario.prerequisiteQuests) {
        setQuestStage(seed.editorId, seed.stage, seed.completed);
    }
    outError.clear();
    return true;
}

QuestRuntimeState& BethesdaSession::quest(const std::string& editorId) {
    const std::string key = normalizedEditorId(editorId);
    auto [found, inserted] = m_quests.try_emplace(key);
    if (inserted) found->second.editorId = editorId;
    return found->second;
}

const QuestRuntimeState* BethesdaSession::findQuest(const std::string& editorId) const {
    const auto found = m_quests.find(normalizedEditorId(editorId));
    return found == m_quests.end() ? nullptr : &found->second;
}

QuestRuntimeState* BethesdaSession::findQuest(const ObjectId& questObject) {
    if (questObject.kind != ObjectIdKind::PersistentReference) return nullptr;
    const auto found = std::find_if(m_quests.begin(), m_quests.end(),
        [&](auto& entry) { return entry.second.record == questObject.reference; });
    return found == m_quests.end() ? nullptr : &found->second;
}

const QuestRuntimeState* BethesdaSession::findQuest(const ObjectId& questObject) const {
    if (questObject.kind != ObjectIdKind::PersistentReference) return nullptr;
    const auto found = std::find_if(m_quests.begin(), m_quests.end(),
        [&](const auto& entry) { return entry.second.record == questObject.reference; });
    return found == m_quests.end() ? nullptr : &found->second;
}

QuestAliasRuntimeState* BethesdaSession::findQuestAlias(const ObjectId& aliasHandle) {
    for (auto& [name, questState] : m_quests) {
        (void)name;
        const auto found = std::find_if(questState.aliases.begin(), questState.aliases.end(),
            [&](const QuestAliasRuntimeState& alias) { return alias.handle == aliasHandle; });
        if (found != questState.aliases.end()) return &*found;
    }
    return nullptr;
}

const QuestAliasRuntimeState* BethesdaSession::findQuestAlias(const ObjectId& aliasHandle) const {
    for (const auto& [name, questState] : m_quests) {
        (void)name;
        const auto found = std::find_if(questState.aliases.begin(), questState.aliases.end(),
            [&](const QuestAliasRuntimeState& alias) { return alias.handle == aliasHandle; });
        if (found != questState.aliases.end()) return &*found;
    }
    return nullptr;
}

void BethesdaSession::queueQuestAliasEvent(
    ObjectId alias, std::string event, std::vector<PapyrusValue> arguments) {
    if (!alias.valid() || event.empty()) return;
    m_pendingQuestAliasEvents.push_back(
        PendingQuestAliasEvent{std::move(alias), std::move(event), std::move(arguments)});
}

void BethesdaSession::flushQuestAliasEvents() {
    for (PendingQuestAliasEvent& event : m_pendingQuestAliasEvents) {
        const std::string normalizedEvent = normalizedEditorId(event.event);
        for (const std::string& scriptClass :
             m_papyrus.scriptClassesForObject(event.alias)) {
            const std::vector<std::string> functions =
                m_papyrus.functionsForClass(scriptClass);
            const bool handled = std::any_of(
                functions.begin(), functions.end(), [&](const std::string& function) {
                    return function == scriptClass + "." + normalizedEvent ||
                        function.ends_with("." + normalizedEvent);
                });
            if (!handled) continue;
            std::string error;
            if (m_papyrus.startFunctionOnObject(
                    event.alias, scriptClass, event.event, event.arguments, error) == 0u) {
                m_pendingDiagnostics.push_back(
                    "could not post alias event " + scriptClass + "." + event.event +
                    ": " + error);
            }
        }
    }
    m_pendingQuestAliasEvents.clear();
}

bool BethesdaSession::registerQuestDefinition(
    const SkyrimQuestDefinition& definition,
    const QuestReferenceResolver& referenceResolver,
    std::string& outError) {
    if (!definition.record.valid() || definition.editorId.empty()) {
        outError = "quest definition is missing stable identity or EditorID";
        return false;
    }
    QuestRuntimeState& state = quest(definition.editorId);
    if (state.record.valid() && state.record != definition.record) {
        outError = "quest " + definition.editorId + " changed RecordKey from " +
            state.record.toString() + " to " + definition.record.toString();
        return false;
    }
    state.record = definition.record;
    std::vector<QuestStageFragmentRuntime> fragments;
    fragments.reserve(definition.stageFragments.size());
    for (const VmadQuestFragment& fragment : definition.stageFragments) {
        const auto stageDefinition = std::find_if(
            definition.stages.begin(), definition.stages.end(),
            [&](const SkyrimQuestStageDefinition& stage) {
                return stage.index == fragment.stage;
            });
        if (stageDefinition == definition.stages.end() || fragment.logEntry < 0 ||
            static_cast<std::size_t>(fragment.logEntry) >=
                stageDefinition->logEntries.size()) {
            outError = "quest fragment " + definition.editorId + "." + fragment.function +
                " names missing stage/log entry " + std::to_string(fragment.stage) + ":" +
                std::to_string(fragment.logEntry);
            return false;
        }
        fragments.push_back(QuestStageFragmentRuntime{
            fragment,
            stageDefinition->logEntries[static_cast<std::size_t>(fragment.logEntry)].conditions});
    }
    std::sort(fragments.begin(), fragments.end(),
        [](const QuestStageFragmentRuntime& left, const QuestStageFragmentRuntime& right) {
            if (left.fragment.stage != right.fragment.stage) {
                return left.fragment.stage < right.fragment.stage;
            }
            if (left.fragment.logEntry != right.fragment.logEntry) {
                return left.fragment.logEntry < right.fragment.logEntry;
            }
            if (left.fragment.scriptClass != right.fragment.scriptClass) {
                return left.fragment.scriptClass < right.fragment.scriptClass;
            }
            return left.fragment.function < right.fragment.function;
        });
    m_questStageFragments.insert_or_assign(
        normalizedEditorId(definition.editorId), std::move(fragments));
    for (const SkyrimQuestObjectiveDefinition& objective : definition.objectives) {
        auto found = std::find_if(state.objectives.begin(), state.objectives.end(),
            [&](const QuestObjectiveState& runtime) { return runtime.index == objective.index; });
        if (found == state.objectives.end()) {
            QuestObjectiveState runtime;
            runtime.index = objective.index;
            runtime.displayText = objective.displayText;
            state.objectives.push_back(std::move(runtime));
        } else if (!objective.displayText.empty()) {
            found->displayText = objective.displayText;
        }
    }
    for (const SkyrimQuestAliasDefinition& alias : definition.aliases) {
        const auto found = std::find_if(state.aliases.begin(), state.aliases.end(),
            [&](const QuestAliasRuntimeState& runtime) { return runtime.id == alias.id; });
        if (found != state.aliases.end()) continue;
        QuestAliasRuntimeState runtime;
        runtime.id = alias.id;
        runtime.name = alias.name;
        runtime.location = alias.location;
        const std::uint64_t aliasBits = core::mix64(
            static_cast<std::uint64_t>(RecordKeyHash{}(definition.record)) ^
            (static_cast<std::uint64_t>(static_cast<std::uint32_t>(alias.id)) << 1u));
        runtime.handle = ObjectId::runtime(aliasBits | (1ull << 63u));
        if (findQuestAlias(runtime.handle) != nullptr) {
            outError = "quest alias handle collision for " + definition.editorId + ":" +
                std::to_string(alias.id);
            return false;
        }
        runtime.sourceFormId = alias.forcedReferenceFormId != 0u
            ? alias.forcedReferenceFormId : alias.uniqueActorFormId;
        runtime.findMatchingReferenceInAliasId =
            alias.findMatchingReferenceInAliasId;
        if (alias.referenceTypeFormId != 0u && referenceResolver) {
            const std::optional<ObjectId> referenceType =
                referenceResolver(alias.referenceTypeFormId);
            if (!referenceType.has_value() ||
                referenceType->kind != ObjectIdKind::PersistentReference) {
                outError = "quest alias " + definition.editorId + ":" +
                    std::to_string(alias.id) + " has an unresolvable ALRT reference type";
                return false;
            }
            runtime.referenceType = referenceType->reference;
        }
        if (alias.location && alias.forcedLocationFormId != 0u && referenceResolver) {
            const std::optional<ObjectId> forcedLocation =
                referenceResolver(alias.forcedLocationFormId);
            if (!forcedLocation.has_value() ||
                forcedLocation->kind != ObjectIdKind::PersistentReference) {
                outError = "quest location alias " + definition.editorId + ":" +
                    std::to_string(alias.id) + " has an unresolvable ALFL location";
                return false;
            }
            runtime.target = *forcedLocation;
        }
        if (runtime.sourceFormId != 0u && referenceResolver) {
            const std::optional<ObjectId> target = referenceResolver(runtime.sourceFormId);
            if (target.has_value()) runtime.target = *target;
        }
        if (alias.createdObjectFormId != 0u) {
            const std::optional<ObjectId> created =
                referenceResolver(alias.createdObjectFormId);
            if (!created.has_value() ||
                created->kind != ObjectIdKind::PersistentReference) {
                outError = "quest alias " + definition.editorId + ":" +
                    std::to_string(alias.id) +
                    " has an unresolvable ALCO object";
                return false;
            }
            runtime.createdObject = created->reference;
            runtime.createdInAliasId = alias.createdInAliasId;
            runtime.createdLevel = alias.createdLevel;
        }
        state.aliases.push_back(std::move(runtime));
    }
    std::sort(state.objectives.begin(), state.objectives.end(),
        [](const auto& left, const auto& right) { return left.index < right.index; });
    std::sort(state.aliases.begin(), state.aliases.end(),
        [](const auto& left, const auto& right) { return left.id < right.id; });
    outError.clear();
    return true;
}

bool BethesdaSession::bindQuestAliasTarget(
    const ObjectId& questObject,
    std::int32_t aliasId,
    ObjectId target,
    std::string& outError) {
    QuestRuntimeState* questState = findQuest(questObject);
    if (questState == nullptr || !target.valid()) {
        outError = "quest alias binding requires a registered quest and valid target";
        return false;
    }
    const auto alias = std::find_if(
        questState->aliases.begin(), questState->aliases.end(),
        [&](const QuestAliasRuntimeState& candidate) { return candidate.id == aliasId; });
    if (alias == questState->aliases.end()) {
        outError = "quest alias " + std::to_string(aliasId) + " is not registered";
        return false;
    }
    alias->target = std::move(target);
    outError.clear();
    return true;
}

void BethesdaSession::setQuestStage(const std::string& editorId, std::int32_t stage, bool completed) {
    QuestRuntimeState& state = quest(editorId);
    const bool newlyCompleted =
        std::find(state.completedStages.begin(), state.completedStages.end(), stage) ==
        state.completedStages.end();
    state.stage = std::max(state.stage, stage);
    if (newlyCompleted) {
        state.completedStages.push_back(stage);
        std::sort(state.completedStages.begin(), state.completedStages.end());
    }
    state.running = !completed;
    state.completed = state.completed || completed;
    if (!newlyCompleted || !state.record.valid()) return;
    const auto fragments = m_questStageFragments.find(normalizedEditorId(editorId));
    if (fragments == m_questStageFragments.end()) return;
    const auto conditionValue = [&](const Condition& condition) -> std::optional<float> {
        const auto subject = [&]() -> const RuntimeObject* {
            ObjectId object;
            if (condition.runOn == 5u) {  // QuestAlias
                const auto alias = std::find_if(
                    state.aliases.begin(), state.aliases.end(),
                    [&](const QuestAliasRuntimeState& candidate) {
                        return candidate.id == static_cast<std::int32_t>(condition.reference);
                    });
                if (alias == state.aliases.end()) return nullptr;
                object = alias->target;
            } else if (condition.runOn == 2u && m_resolvedFormResolver) {  // Reference
                const std::optional<ObjectId> resolved =
                    m_resolvedFormResolver(condition.reference);
                if (!resolved.has_value()) return nullptr;
                object = *resolved;
            } else {
                return nullptr;
            }
            return m_world.find(object);
        }();
        if (subject == nullptr) return std::nullopt;
        if (condition.function == 46u) {  // GetDead
            return subject->actorValues.has_value() && subject->actorValues->dead ? 1.0f : 0.0f;
        }
        if ((condition.function == 47u || condition.function == 67u) &&
            m_resolvedFormResolver) {
            const std::optional<ObjectId> parameter =
                m_resolvedFormResolver(condition.parameter1);
            if (!parameter.has_value() ||
                parameter->kind != ObjectIdKind::PersistentReference) return std::nullopt;
            if (condition.function == 47u) {  // GetItemCount
                const auto item = std::find_if(
                    subject->inventory.begin(), subject->inventory.end(),
                    [&](const InventoryEntry& entry) {
                        return entry.item == parameter->reference;
                    });
                return item == subject->inventory.end()
                    ? 0.0f : static_cast<float>(item->count);
            }
            return subject->base == parameter->reference ? 1.0f : 0.0f;  // GetIsID
        }
        return std::nullopt;
    };
    for (const QuestStageFragmentRuntime& runtime : fragments->second) {
        const VmadQuestFragment& fragment = runtime.fragment;
        if (fragment.stage != stage) continue;
        const ConditionEvaluation conditions =
            evaluateConditions(runtime.conditions, conditionValue, true);
        for (const std::string& diagnostic : conditions.diagnostics) {
            m_pendingDiagnostics.push_back(
                state.editorId + " stage " + std::to_string(stage) + " " + diagnostic);
        }
        if (!conditions.matched) continue;
        std::string error;
        if (m_papyrus.startFunctionOnObject(
                ObjectId::persistent(state.record), fragment.scriptClass,
                fragment.function, {}, error) == 0u) {
            m_pendingDiagnostics.push_back(
                "could not dispatch " + state.editorId + " stage " +
                std::to_string(stage) + " fragment " + fragment.scriptClass + "." +
                fragment.function + ": " + error);
        }
    }
}

void BethesdaSession::setScenePlaying(const RecordKey& scene, bool playing) {
    if (scene.valid()) m_scenes.insert_or_assign(scene, playing);
}

bool BethesdaSession::registerLocation(
    RecordKey location, RecordKey parent, std::vector<RecordKey> keywords,
    std::string& outError) {
    if (!location.valid()) {
        outError = "location runtime definition has no stable RecordKey";
        return false;
    }
    if (parent == location) {
        outError = "location cannot be its own parent: " + location.toString();
        return false;
    }
    keywords.erase(std::remove_if(keywords.begin(), keywords.end(),
        [](const RecordKey& keyword) { return !keyword.valid(); }), keywords.end());
    std::sort(keywords.begin(), keywords.end());
    keywords.erase(std::unique(keywords.begin(), keywords.end()), keywords.end());
    LocationRuntimeState& state = m_locations[location];
    state.record = std::move(location);
    state.parent = std::move(parent);
    state.keywords = std::move(keywords);
    for (const RecordKey& keyword : state.keywords) {
        state.keywordData.try_emplace(keyword, 0.0f);
    }
    for (auto entry = state.keywordData.begin(); entry != state.keywordData.end();) {
        if (!std::binary_search(state.keywords.begin(), state.keywords.end(), entry->first)) {
            entry = state.keywordData.erase(entry);
        } else {
            ++entry;
        }
    }
    outError.clear();
    return true;
}

bool BethesdaSession::registerGlobalVariable(
    RecordKey variable, float initialValue, std::string& outError) {
    if (!variable.valid() || !std::isfinite(initialValue)) {
        outError = "global variable requires a stable RecordKey and finite initial value";
        return false;
    }
    m_globalVariables.try_emplace(std::move(variable), initialValue);
    outError.clear();
    return true;
}

void BethesdaSession::setLocationLoaded(const RecordKey& location, bool loaded) {
    const auto found = m_locations.find(location);
    if (found != m_locations.end()) found->second.loaded = loaded;
}

void BethesdaSession::clearLoadedLocations() {
    for (auto& [record, location] : m_locations) {
        (void)record;
        location.loaded = false;
    }
}

void BethesdaSession::registerSkyrimNatives() {
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
    m_papyrus.registerNative("Scene.Stop", scenePlayingNative(false));
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
        const std::vector<RuntimeObject> objects = world.orderedObjects();
        const auto found = std::find_if(objects.begin(), objects.end(), [&](const auto& object) {
            return object.base == identity.reference;
        });
        return found == objects.end() ? std::nullopt : std::optional<RuntimeObject>{*found};
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
    m_papyrus.registerNative("Actor.SetActorValue", setActorValueNative);
}

Tes3NativeResult BethesdaSession::executeTes3WorldNative(const Tes3NativeCall& call) {
    Tes3NativeResult result;
    const auto asText = [](const Tes3Value& value) {
        if (value.type == Tes3ValueType::String) return value.string;
        if (value.type == Tes3ValueType::Object) return value.object.toString();
        return std::string{};
    };
    const auto asInt = [](const Tes3Value& value) {
        if (value.type == Tes3ValueType::Number) return static_cast<std::int32_t>(value.number);
        if (value.type == Tes3ValueType::String) {
            try { return static_cast<std::int32_t>(std::stoi(value.string)); }
            catch (...) { return 0; }
        }
        return 0;
    };
    const auto asNumber = [](const Tes3Value& value) {
        if (value.type == Tes3ValueType::Number) return value.number;
        if (value.type == Tes3ValueType::String) {
            try { return std::stod(value.string); }
            catch (...) { return 0.0; }
        }
        return 0.0;
    };
    const auto unquote = [](std::string value) {
        if (value.size() >= 2u && value.front() == '"' && value.back() == '"') {
            value = value.substr(1u, value.size() - 2u);
        }
        return value;
    };
    const auto resolveObject = [&](std::string authored) -> ObjectId {
        authored = unquote(std::move(authored));
        if (authored.empty()) return call.owner;
        if (normalizedEditorId(authored) == "player") return m_playerObject;
        RecordKey serialized;
        if (parseRecordKey(authored, serialized)) return ObjectId::persistent(std::move(serialized));
        const std::string wanted = makeTes3RecordKey("REFR", authored).textId;
        for (const RuntimeObject& object : m_world.orderedObjects()) {
            if ((object.id.kind == ObjectIdKind::PersistentReference &&
                 object.id.reference.textId == wanted) || object.base.textId == wanted) {
                return object.id;
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
    if (command == "getlevel" || command == "getrace" ||
        command == "gethealthgetratio") {
        const Tes3ActorDefinition* actor = actorDefinitionForTarget();
        if (actor == nullptr) { result.error = command + " requires an authored actor"; return result; }
        if (command == "getlevel") result.value = Tes3Value::fromNumber(actor->level);
        else if (command == "getrace") {
            if (call.arguments.empty()) { result.error = "GetRace requires a race id"; return result; }
            result.value = Tes3Value::fromNumber(normalizedEditorId(actor->race) ==
                normalizedEditorId(asText(call.arguments[0])) ? 1.0 : 0.0);
        } else {
            const double health = object != nullptr && object->actorValues.has_value()
                ? object->actorValues->health : actor->health;
            const double maximum = object != nullptr && object->actorValues.has_value()
                ? object->actorValues->maxHealth : actor->health;
            result.value = Tes3Value::fromNumber(maximum > 0.0 ? health / maximum : 0.0);
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
        (void)m_tes3.startDialogue(dialogueActor, m_tes3.playerState(), false);
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
                } else if (effect.effectId == 83 && effect.skill >= 0 &&
                    static_cast<std::size_t>(effect.skill) < tes3Skills.size() &&
                    tes3Skills[static_cast<std::size_t>(effect.skill)] == stat) {
                    magnitude += effect.magnitude;
                }
            }
        }
        return magnitude;
    };
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
        for (std::size_t index = 0u; index < spell->effects.size(); ++index) {
            const Tes3SpellEffect& authored = spell->effects[index];
            if (authored.effectId != 79 && authored.effectId != 83) {
                result.error = "Cast reaches unsupported gameplay magic effect " +
                    std::to_string(authored.effectId);
                return result;
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
        auto& spells = m_tes3.activeSpellsForRestore()[recipient];
        std::erase_if(spells, [&](const Tes3ActiveSpell& item) { return item.spell == spell->record; });
        spells.push_back(std::move(active));
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
            result.value = Tes3Value::fromNumber(value + activeFortify(target, queriedActorStat));
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
        m_tes3.referenceOverridesForRestore()[target].transform = transform;
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
            result.value = Tes3Value::fromNumber(command == "getpos"
                ? transform->position[axis] : transform->rotationRadians[axis] *
                    (180.0 / 3.14159265358979323846));
            return result;
        }
        if (command == "setpos" || command == "setangle") {
            if (call.arguments.size() < 2u) { result.error = command + " requires axis and value"; return result; }
            const std::size_t axis = component(call.arguments[0]);
            if (command == "setpos") transform->position[axis] = asNumber(call.arguments[1]);
            else transform->rotationRadians[axis] = static_cast<float>(
                asNumber(call.arguments[1]) * (3.14159265358979323846 / 180.0));
        } else if (command == "setscale" || command == "modscale") {
            if (call.arguments.empty()) { result.error = command + " requires a value"; return result; }
            const float value = static_cast<float>(asNumber(call.arguments[0]));
            transform->scale = command == "setscale" ? value : transform->scale + value;
        } else {
            if (call.arguments.size() < 4u) { result.error = command + " requires x y z rotation"; return result; }
            transform->position = {asNumber(call.arguments[0]), asNumber(call.arguments[1]),
                asNumber(call.arguments[2])};
            transform->rotationRadians[2] = static_cast<float>(
                asNumber(call.arguments[3]) * (3.14159265358979323846 / 180.0));
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
        double fight = state.locals["stat:fight"].number;
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
        if (object == nullptr || !object->aiState.has_value()) {
            result.value = Tes3Value::fromNumber(command == "getaipackagedone" ? 1.0 : -1.0);
        } else if (command == "getaipackagedone") {
            result.value = Tes3Value::fromNumber(object->aiState->scriptedMoveArrived ? 1.0 : 0.0);
        } else {
            result.value = Tes3Value::fromNumber(object->aiState->walking ? 1.0 : -1.0);
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
        if (command == "aitravel" && call.arguments.size() >= 3u) {
            ai.wanderTarget = {static_cast<float>(asNumber(call.arguments[0])),
                static_cast<float>(asNumber(call.arguments[1])),
                static_cast<float>(asNumber(call.arguments[2]))};
        } else if (command == "aiwander") {
            ai.wanderOrigin = {static_cast<float>(object->transform.position[0]),
                static_cast<float>(object->transform.position[1]),
                static_cast<float>(object->transform.position[2])};
            ai.wanderTarget = ai.wanderOrigin;
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
        Tes3Value& stored = m_tes3.referenceOverridesForRestore()[target].locals["stat:" + stat];
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
        if (!target.valid()) { result.error = "Activate target does not resolve"; return result; }
        m_tes3.dispatchGameplayEvent("onactivate", target);
        if (object != nullptr) {
            RuntimeActivatorState activator = object->activatorState.value_or(RuntimeActivatorState{});
            ++activator.activationCount;
            WorldCommand world;
            world.type = WorldCommandType::SetActivatorState;
            world.target = target;
            world.activatorState = std::move(activator);
            (void)m_world.queue(std::move(world));
        }
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
        result.value = Tes3Value::fromNumber(object != nullptr && object->actorValues.has_value()
            ? object->actorValues->health : 0.0f);
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
        if (command == "getitemcount" || command == "getspell") {
            const auto found = std::find_if(object->inventory.begin(), object->inventory.end(),
                [&](const InventoryEntry& entry) { return entry.item == item; });
            result.value = Tes3Value::fromNumber(
                found == object->inventory.end() ? 0.0 : static_cast<double>(found->count));
            return result;
        }
        WorldCommand world;
        world.type = command == "additem" || command == "addspell"
            ? WorldCommandType::AddItem : WorldCommandType::RemoveItem;
        world.target = target;
        world.item = item;
        world.itemCount = count;
        (void)m_world.queue(std::move(world));
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
        if (object == nullptr || !object->actorValues.has_value() || call.arguments.empty()) {
            result.error = command + " requires a resident actor and value";
            return result;
        }
        WorldCommand world;
        world.type = command == "sethealth" ? WorldCommandType::SetActorBaseValue :
            command == "modhealth" ? WorldCommandType::AdjustActorBaseValue :
            WorldCommandType::AdjustActorValue;
        world.target = target;
        world.actorValue = ActorValue::Health;
        if (command == "sethealth") world.actorValueAbsolute = static_cast<float>(asInt(call.arguments[0]));
        else world.actorValueDelta = static_cast<float>(asInt(call.arguments[0]));
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
    result.error = "unsupported gameplay MWScript native " + command;
    return result;
}

void BethesdaSession::simulateTick(
    std::uint64_t tick, double stepSeconds, BethesdaSessionStep& result) {
    const float fixedDelta = static_cast<float>(stepSeconds);
    // Combat packages and Actor.StartCombat converge here. Rendering never
    // selects targets: fixed-tick AI aims from one Jolt character to the other
    // and uses the same cone/occlusion/cooldown path as player input.
    for (const RuntimeObject& actor : m_world.orderedObjects()) {
        if (!actor.combatState.has_value() ||
            !actor.combatState->combatTarget.valid() ||
            (actor.actorValues.has_value() && actor.actorValues->dead)) {
            continue;
        }
        const RuntimeObject* target = m_world.find(actor.combatState->combatTarget);
        const auto sourcePhysical = m_physics.characterState(actor.id);
        const auto targetPhysical = target == nullptr
            ? std::optional<PhysicsCharacterStep>{}
            : m_physics.characterState(target->id);
        if (target == nullptr || !target->enabled ||
            (target->actorValues.has_value() && target->actorValues->dead) ||
            !sourcePhysical.has_value() || !targetPhysical.has_value()) {
            RuntimeCombatState stopped = *actor.combatState;
            stopped.combatTarget = {};
            WorldCommand command;
            command.type = WorldCommandType::SetCombatState;
            command.target = actor.id;
            command.combatState = std::move(stopped);
            (void)m_world.queue(std::move(command));
            continue;
        }
        const odai::math::Vector3 direction =
            targetPhysical->position - sourcePhysical->position;
        if (odai::math::length(direction) <= 170.0f) {
            (void)performMeleeAttack(actor.id, direction, 10.0f, 170.0f);
        }
    }
    for (auto& [object, runtime] : m_actorAnimations) {
        if (const auto physical = m_physics.characterState(object)) {
            runtime.input.grounded = physical->grounded;
            runtime.input.groundVelocity = physical->groundVelocity;
            runtime.input.groundNormal = physical->groundNormal;
            runtime.input.verticalVelocity = physical->velocity.y;
            runtime.input.falling = physical->falling;
            runtime.input.landed = physical->landed;
            runtime.input.blocked = physical->blocked;
            runtime.input.movementSpeed = odai::math::length(odai::math::Vector3{
                physical->velocity.x, 0.0f, physical->velocity.z});
        }
        runtime.thirdPersonOutput = runtime.thirdPerson.step(runtime.input, fixedDelta);
        if (runtime.firstPersonView != nullptr) {
            runtime.firstPersonOutput = runtime.firstPerson.step(runtime.input, fixedDelta);
        }
        PhysicsCharacterInput input;
        input.desiredVelocity = runtime.input.requestedVelocity;
        input.rootMotion = runtime.thirdPersonOutput.desiredRootMotion;
        input.animationDriven = runtime.input.animationDriven;
        (void)m_physics.setCharacterInput(object, input);
        if (m_papyrus.hasFunction("OnAnimationEvent")) {
            for (const odai::anim::AnimationEvent& event : runtime.thirdPersonOutput.events) {
                const std::array arguments{PapyrusValue::fromObject(object),
                    PapyrusValue::fromString(event.name), PapyrusValue::fromString(event.payload)};
                std::string error;
                (void)m_papyrus.postEvent("OnAnimationEvent", arguments, error);
                if (!error.empty()) result.diagnostics.push_back(std::move(error));
            }
        }
        runtime.input.events.clear();
        runtime.input.attacking = false;
        runtime.input.equipping = false;
    }
    for (const auto& [object, physical] : m_physics.step(fixedDelta)) {
        const RuntimeObject* resident = m_world.find(object);
        if (resident == nullptr) continue;
        WorldCommand command;
        command.type = WorldCommandType::SetPosition;
        command.target = object;
        command.transform.position = {physical.position.x, physical.position.y, physical.position.z};
        (void)m_world.queue(std::move(command));
    }
    if (m_tes3.content() != nullptr) {
        Tes3VmStepResult tes3Vm = m_tes3.step(tick, 4096u);
        result.vmInstructions += tes3Vm.instructions;
        result.diagnostics.insert(result.diagnostics.end(),
            std::make_move_iterator(tes3Vm.diagnostics.begin()),
            std::make_move_iterator(tes3Vm.diagnostics.end()));
    }
    PapyrusAdvanceResult vm = m_papyrus.advance(tick, 4096u, m_world);
    result.vmInstructions += vm.instructions;
    result.diagnostics.insert(result.diagnostics.end(),
        std::make_move_iterator(vm.diagnostics.begin()),
        std::make_move_iterator(vm.diagnostics.end()));
    CommandApplyResult commands = m_world.applyQueuedCommands();
    result.worldCommands += commands.applied;
    result.residencyChanged = result.residencyChanged || commands.residencyChanged;
    result.renderDeltas.insert(result.renderDeltas.end(),
        std::make_move_iterator(commands.renderDeltas.begin()),
        std::make_move_iterator(commands.renderDeltas.end()));
    result.diagnostics.insert(result.diagnostics.end(),
        std::make_move_iterator(commands.diagnostics.begin()),
        std::make_move_iterator(commands.diagnostics.end()));
    // Object events observe the fully applied deterministic mutation batch.
    // They begin on the next fixed tick and are already present in VM
    // snapshots if a save occurs at this frame boundary.
    flushQuestAliasEvents();
    // xorshift32: state is part of the save and replay hash.
    m_randomState ^= m_randomState << 13u;
    m_randomState ^= m_randomState >> 17u;
    m_randomState ^= m_randomState << 5u;
    if (m_randomState == 0u) m_randomState = 1u;
}

std::uint64_t BethesdaSession::deterministicHash() const {
    std::uint64_t hash = m_world.deterministicHash();
    hashString(hash, m_config.contentFingerprint);
    hashString(hash, m_config.scenarioId);
    hashString(hash, m_playerObject.toString());
    const std::uint64_t tick = m_clock.tick();
    hash ^= core::mix64(tick);
    hash ^= core::mix64(m_randomState);
    const std::uint64_t accumulatorBits = std::bit_cast<std::uint64_t>(m_clock.accumulatorSeconds());
    hash ^= core::mix64(accumulatorBits);
    for (const auto& [key, questState] : m_quests) {
        hashString(hash, key);
        hashString(hash, questState.record.toString());
        hash ^= core::mix64(static_cast<std::uint32_t>(questState.stage));
        for (const std::int32_t completedStage : questState.completedStages) {
            hash ^= core::mix64(static_cast<std::uint32_t>(completedStage));
        }
        hashScalar(hash, questState.running);
        hash ^= questState.completed ? 0xc011ec7edull : 0u;
        hashScalar(hash, questState.failed);
        for (const QuestObjectiveState& objective : questState.objectives) {
            hash ^= core::mix64(
                static_cast<std::uint32_t>(objective.index) |
                (static_cast<std::uint64_t>(objective.displayed) << 32u) |
                (static_cast<std::uint64_t>(objective.completed) << 33u) |
                (static_cast<std::uint64_t>(objective.failed) << 34u));
        }
        for (const QuestAliasRuntimeState& alias : questState.aliases) {
            hashScalar(hash, alias.id);
            hashString(hash, alias.name);
            hashScalar(hash, alias.location);
            hashString(hash, alias.handle.toString());
            hashString(hash, alias.target.toString());
            hashScalar(hash, alias.findMatchingReferenceInAliasId);
            hashString(hash, alias.referenceType.toString());
            hashString(hash, alias.createdObject.toString());
            hashScalar(hash, alias.createdInAliasId);
            hashScalar(hash, alias.createdLevel);
            hashScalar(hash, alias.createdObjectMaterialized);
        }
    }
    for (const auto& [name, value] : m_statistics) {
        hashString(hash, name);
        hashScalar(hash, value);
    }
    for (const RecordKey& discovery : m_discoveries) {
        hashString(hash, discovery.toString());
    }
    for (const auto& [scene, playing] : m_scenes) {
        hashString(hash, scene.toString());
        hashScalar(hash, playing);
    }
    if (m_forcedWeather.valid()) hashString(hash, m_forcedWeather.toString());
    for (const auto& [record, location] : m_locations) {
        hashString(hash, record.toString());
        hashString(hash, location.parent.toString());
        hashScalar(hash, location.loaded);
        for (const RecordKey& keyword : location.keywords) {
            hashString(hash, keyword.toString());
        }
        for (const auto& [keyword, value] : location.keywordData) {
            hashString(hash, keyword.toString());
            hashScalar(hash, value);
        }
    }
    for (const auto& [record, value] : m_globalVariables) {
        hashString(hash, record.toString());
        hashScalar(hash, value);
    }
    hashScalar(hash, m_nextStoryEventSequence);
    for (const StoryEventRuntimeState& event : m_storyEvents) {
        hashScalar(hash, event.sequence);
        hashString(hash, event.keyword.toString());
        for (const PapyrusValue& argument : event.arguments) {
            hashPapyrusValue(hash, argument);
        }
    }
    hashScalar(hash, m_nextGiftMenuSequence);
    for (const GiftMenuRequestState& request : m_giftMenuRequests) {
        hashScalar(hash, request.sequence);
        hashString(hash, request.actor.toString());
        hashString(hash, request.player.toString());
        hashString(hash, request.filterList.toString());
        hashScalar(hash, request.playerGives);
        hashScalar(hash, request.showStolenItems);
        hashScalar(hash, request.useFavorPoints);
    }
    for (const std::string& log : m_scriptDebugLogs) hashString(hash, log);
    for (const AnimationActorSnapshot& animation : animationSnapshots()) {
        hashString(hash, animation.object.toString());
        hashBehaviorGraph(hash, animation.thirdPerson);
        hashScalar(hash, animation.firstPerson.has_value());
        if (animation.firstPerson.has_value()) {
            hashBehaviorGraph(hash, *animation.firstPerson);
        }
    }
    for (const PhysicsCharacterSnapshot& character : physicsSnapshots()) {
        hashString(hash, character.object.toString());
        hashScalar(hash, character.position.x); hashScalar(hash, character.position.y);
        hashScalar(hash, character.position.z);
        hashScalar(hash, character.rotation.x); hashScalar(hash, character.rotation.y);
        hashScalar(hash, character.rotation.z); hashScalar(hash, character.rotation.w);
        hashScalar(hash, character.velocity.x); hashScalar(hash, character.velocity.y);
        hashScalar(hash, character.velocity.z);
        hashScalar(hash, character.groundNormal.x); hashScalar(hash, character.groundNormal.y);
        hashScalar(hash, character.groundNormal.z);
        hashScalar(hash, character.grounded);
        hashScalar(hash, character.supportingObject.has_value());
        if (character.supportingObject.has_value()) {
            hashString(hash, character.supportingObject->toString());
        }
    }
    const PapyrusVmSnapshot vm = m_papyrus.snapshot();
    hash ^= core::mix64(vm.nextThreadId);
    std::vector<std::pair<std::string, PapyrusValue>> globals(vm.globals.begin(), vm.globals.end());
    std::sort(globals.begin(), globals.end(), [](const auto& left, const auto& right) {
        return left.first < right.first;
    });
    for (const auto& [name, value] : globals) {
        hashString(hash, name);
        hashPapyrusValue(hash, value);
    }
    std::vector<PapyrusThreadSnapshot> threads = vm.threads;
    std::sort(threads.begin(), threads.end(), [](const auto& left, const auto& right) {
        return left.id < right.id;
    });
    for (const PapyrusThreadSnapshot& thread : threads) {
        hashScalar(hash, thread.id);
        hashPapyrusFrame(hash, thread);
        hashScalar(hash, thread.resumeTick);
        hashScalar(hash, thread.failed);
        hashScalar(hash, static_cast<std::uint64_t>(thread.callStack.size()));
        for (const PapyrusCallFrameSnapshot& frame : thread.callStack) {
            hashPapyrusFrame(hash, frame);
        }
    }
    for (const PapyrusScriptInstanceSnapshot& instance : vm.instances) {
        hashString(hash, instance.object.toString());
        hashString(hash, instance.scriptClass);
        hashString(hash, instance.activeState);
        std::vector<std::pair<std::string, PapyrusValue>> properties(
            instance.properties.begin(), instance.properties.end());
        std::sort(properties.begin(), properties.end(), [](const auto& left, const auto& right) {
            return left.first < right.first;
        });
        for (const auto& [name, value] : properties) {
            hashString(hash, name);
            hashPapyrusValue(hash, value);
        }
    }
    for (const PapyrusUpdateRegistrationSnapshot& update : vm.updates) {
        hashString(hash, update.object.toString());
        hashString(hash, update.scriptClass);
        hashString(hash, update.eventFunction);
        hashScalar(hash, update.intervalTicks);
        hashScalar(hash, update.nextTick);
        hashScalar(hash, update.repeating);
    }
    if (m_tes3.content()) {
        hashScalar(hash, m_tes3.journal().nextSequence());
        for (const auto& [key, quest] : m_tes3.journal().quests()) {
            hashString(hash, key.toString());
            hashString(hash, quest.id);
            hashScalar(hash, quest.currentIndex);
            hashScalar(hash, quest.classification);
            hashScalar(hash, quest.hasStatusFlags);
            for (const RecordKey& entry : quest.visitedEntries) {
                hashString(hash, entry.toString());
            }
        }
        for (const Tes3JournalVisit& visit : m_tes3.journal().chronology()) {
            hashScalar(hash, visit.sequence);
            hashScalar(hash, visit.tick);
            hashString(hash, visit.quest.toString());
            hashString(hash, visit.info.toString());
            hashScalar(hash, visit.index);
            hashScalar(hash, visit.status);
            hashString(hash, visit.sourcePlugin);
        }
        for (const RecordKey& topic : m_tes3.knownTopics()) {
            hashString(hash, topic.toString());
        }
        hashScalar(hash, m_tes3.scripts().nextThreadId());
        for (const auto& [name, value] : m_tes3.scripts().globals()) {
            hashString(hash, name);
            hashTes3Value(hash, value);
        }
        for (const auto& [id, thread] : m_tes3.scripts().threads()) {
            hashScalar(hash, id);
            hashString(hash, thread.program);
            hashString(hash, thread.owner.toString());
            hashScalar(hash, static_cast<std::uint64_t>(thread.instruction));
            hashScalar(hash, thread.state);
            hashString(hash, thread.suspensionReason);
            hashString(hash, thread.error);
            for (const auto& [name, value] : thread.locals) {
                hashString(hash, name);
                hashTes3Value(hash, value);
            }
            for (const auto& [name, value] : thread.eventVariables) {
                hashString(hash, name);
                hashTes3Value(hash, value);
            }
        }
        for (const auto& [object, override] : m_tes3.referenceOverrides()) {
            hashString(hash, object.toString());
            hashScalar(hash, override.enabled.has_value());
            if (override.enabled.has_value()) hashScalar(hash, *override.enabled);
            hashScalar(hash, override.deleted);
            hashScalar(hash, override.transform.has_value());
            if (override.transform.has_value()) {
                for (const double component : override.transform->position) {
                    hashScalar(hash, component);
                }
                for (const float component : override.transform->rotationRadians) {
                    hashScalar(hash, component);
                }
                hashScalar(hash, override.transform->scale);
            }
            for (const auto& [name, value] : override.locals) {
                hashString(hash, name);
                hashTes3Value(hash, value);
            }
        }
        for (const std::string& sound : m_tes3.activeSounds()) hashString(hash, sound);
        for (const auto& [target, spells] : m_tes3.activeSpells()) {
            hashString(hash, target.toString());
            for (const Tes3ActiveSpell& spell : spells) {
                hashString(hash, spell.spell.toString());
                hashString(hash, spell.caster.toString());
                hashScalar(hash, spell.appliedTick);
                for (const Tes3ActiveSpellEffect& effect : spell.effects) {
                    hashScalar(hash, effect.effectId);
                    hashScalar(hash, effect.skill);
                    hashScalar(hash, effect.attribute);
                    hashScalar(hash, effect.magnitude);
                    hashScalar(hash, effect.expiresTick);
                }
            }
        }
        const Tes3DialoguePlayerState& tes3Player = m_tes3.playerState();
        hashString(hash, tes3Player.object.toString());
        for (const auto& [faction, rank] : tes3Player.factionRanks) {
            hashString(hash, faction);
            hashScalar(hash, rank);
        }
        for (const auto& [name, value] : tes3Player.numericFilters) {
            hashString(hash, name);
            hashScalar(hash, value);
        }
        for (const auto& [item, count] : tes3Player.inventory) {
            hashString(hash, item.toString());
            hashScalar(hash, count);
        }
        for (const auto& [actor, count] : tes3Player.deathCounts) {
            hashString(hash, actor);
            hashScalar(hash, count);
        }
        const Tes3DialogueState& dialogue = m_tes3.dialogue();
        hashScalar(hash, dialogue.active);
        hashString(hash, dialogue.actor.object.toString());
        hashString(hash, dialogue.player.object.toString());
        hashString(hash, dialogue.currentTopic.toString());
        hashString(hash, dialogue.currentInfo.toString());
        hashScalar(hash, dialogue.choice);
        hashScalar(hash, dialogue.goodbye);
        for (const RecordKey& info : dialogue.exhaustedInfos) {
            hashString(hash, info.toString());
        }
        for (const Tes3DialogueChoice& choice : dialogue.choices) {
            hashString(hash, choice.label);
            hashScalar(hash, choice.value);
        }
    }
    return core::mix64(hash);
}

}  // namespace odai::bethesda
