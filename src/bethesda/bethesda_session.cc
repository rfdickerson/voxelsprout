#include "bethesda/bethesda_session.h"
#include "bethesda/session_text.h"

#include "core/hash.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <set>

namespace odai::bethesda {
namespace {

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
    hashString(hash, graph.previousState);
    hashScalar(hash, graph.previousStateTime);
    hashScalar(hash, graph.transitionElapsed);
    hashScalar(hash, graph.transitionDuration);
    hashScalar(hash, graph.fixedTick);
    hashScalar(hash, graph.wasGrounded);
    hashScalar(hash, graph.wasTalking);
    hashString(hash, graph.graphFingerprint);
    hashScalar(hash, graph.executionMode);
    hashString(hash, graph.selectedRule);
    hashString(hash, graph.selectedProvider);
    hashString(hash, graph.selectedClip);
    hashString(hash, graph.previousClip);
    hashScalar(hash, graph.randomState);
    hashScalar(hash, graph.actionIdentity);
    for (const auto& [name, time] : graph.layerTimes) { hashString(hash, name); hashScalar(hash, time); }
    for (const auto& [node, state] : graph.activeStates) {
        hashScalar(hash, node); hashScalar(hash, state);
    }
    for (const auto& [name, value] : graph.variables) {
        hashString(hash, name); hashScalar(hash, value);
    }
    for (const odai::anim::AnimationEvent& event : graph.queuedEvents) {
        hashString(hash, event.name);
        hashString(hash, event.payload);
    }
}

}  // namespace

bool BethesdaSession::configure(BethesdaSessionConfig config, std::string& outError) {
    if (config.game == importer::bethesda::BethesdaGame::Unknown) {
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
        if (m_config.game == importer::bethesda::BethesdaGame::Morrowind) {
            m_playerObject = ObjectId::persistent(makeTes3RecordKey("NPC_", "player"));
        } else if (m_config.game == importer::bethesda::BethesdaGame::SkyrimSpecialEdition) {
            m_playerObject = ObjectId::persistent(makeRecordKey("Skyrim.esm", 0x14u));
        }
    }
    m_config.playerObject = m_playerObject;
    m_randomState = m_config.randomSeed == 0u ? 1u : m_config.randomSeed;
    m_clock.reset();
    m_world.clear();
    m_physics.clear();
    m_livingWorld.reset(LivingWorldConfig{
        m_config.livingWorldEnabled, m_config.gameTimeScale, 8u * 60u,
        72u * 60u, 64u});
    m_actorAnimations.clear();
    m_actorGuards.clear();
    m_meleeContacts.clear();
    m_pendingAnimationSnapshots.clear();
    m_pendingPhysicsSnapshots.clear();
    m_papyrus.clearRuntimeState();
    m_tes3.clear();
    m_quests.clear();
    m_questJournal.clear();
    m_questStageFragments.clear();
    m_dialogueTopics.clear();
    m_dialogueBranches.clear();
    m_dialogueInfos.clear();
    m_statistics.clear();
    m_discoveries.clear();
    m_scenes.clear();
    m_scriptTriggers.clear();
    m_flyingAliasPackages.clear(); m_patrolLinks.clear();
    m_sceneWaitingActors.clear();
    m_sceneDefinitions.clear(); m_scenePackages.clear(); m_sceneProgress.clear();
    m_sceneSpeech.clear(); m_sceneSpeechSequence = 0;
    m_skyrimItems.clear();
    m_forcedWeather = {};
    m_imageSpaceCommands.clear();
    m_effectAnimationPlayer = {};
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
    // Character controllers, dynamic residency and camera casts are shared
    // runtime facilities, not Skyrim script facilities. Initializing Jolt only
    // for TES5 left the TES3 player registration path operating on an inert
    // world and made cross-game presentation impossible.
    if (!m_physics.initialize(outError)) return false;
    if (m_config.game == importer::bethesda::BethesdaGame::SkyrimSpecialEdition) {
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

bool BethesdaSession::installGameplayCells(
    std::vector<GameplayCellPayload> cells, std::string& outError) {
    if (!m_configured) {
        outError = "Bethesda session is not configured";
        return false;
    }
    for (const GameplayCellPayload& cell : cells) {
        if (cell.version != kGameplayCellPayloadVersion) {
            outError = "unsupported gameplay cell payload version";
            return false;
        }
        if (cell.contentFingerprint != m_config.contentFingerprint) {
            outError = "gameplay cell payload fingerprint differs from session content";
            return false;
        }
    }
    m_livingWorld.installCells(std::move(cells));
    outError.clear();
    return true;
}

bool BethesdaSession::upsertGameplayCell(
    GameplayCellPayload cell, std::string& outError) {
    if (!m_configured) {
        outError = "Bethesda session is not configured";
        return false;
    }
    if (cell.version != kGameplayCellPayloadVersion) {
        outError = "unsupported gameplay cell payload version";
        return false;
    }
    if (cell.contentFingerprint != m_config.contentFingerprint) {
        outError = "gameplay cell payload fingerprint differs from session content";
        return false;
    }
    m_livingWorld.upsertCell(std::move(cell));
    outError.clear();
    return true;
}

bool BethesdaSession::registerDynamicBody(
    ObjectId objectId, PhysicsDynamicBodyConfig config, std::string& outError) {
    const RuntimeObject* object = m_world.find(objectId);
    if (object == nullptr) {
        outError = "dynamic body requires a resident runtime object";
        return false;
    }
    config.position = {static_cast<float>(object->transform.position[0]),
        static_cast<float>(object->transform.position[1]),
        static_cast<float>(object->transform.position[2])};
    if (!m_physics.addDynamicBody(objectId, config, outError)) return false;
    if (object->physicalState.has_value()) {
        PhysicsDynamicBodySnapshot restored;
        restored.object = objectId;
        restored.position = config.position;
        restored.rotation = {object->physicalState->rotationQuaternion[0],
            object->physicalState->rotationQuaternion[1],
            object->physicalState->rotationQuaternion[2],
            object->physicalState->rotationQuaternion[3]};
        restored.linearVelocity = {object->physicalState->linearVelocity[0],
            object->physicalState->linearVelocity[1],
            object->physicalState->linearVelocity[2]};
        restored.angularVelocity = {object->physicalState->angularVelocity[0],
            object->physicalState->angularVelocity[1],
            object->physicalState->angularVelocity[2]};
        restored.active = odai::math::length(restored.linearVelocity) > 1.0e-4f ||
            odai::math::length(restored.angularVelocity) > 1.0e-4f;
        if (!m_physics.restoreDynamicBody(restored, outError)) {
            (void)m_physics.removeDynamicBody(objectId);
            return false;
        }
    }
    outError.clear();
    return true;
}

bool BethesdaSession::configureTes3Content(
    std::shared_ptr<const Tes3ContentStore> content, std::string& outError) {
    if (!m_configured || m_config.game != importer::bethesda::BethesdaGame::Morrowind) {
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
    const PhysicsCharacterConfig& physicsConfig, std::string& outError, bool createController) {
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
    if (createController && !hadController && !registerActorController(object, physicsConfig, outError)) return false;
    const auto pending = m_pendingAnimationSnapshots.find(object);
    if (pending != m_pendingAnimationSnapshots.end()) {
        if (pending->second.firstPerson.has_value() != (runtime.firstPersonView != nullptr)) {
            outError = "pending first-person animation view does not match actor " +
                object.toString();
            if (!hadController) (void)unregisterActorController(object);
            return false;
        }
        const bool changed = (!pending->second.thirdPerson.graphFingerprint.empty() &&
            pending->second.thirdPerson.graphFingerprint != runtime.thirdPersonView->sourceFingerprint) ||
            pending->second.thirdPerson.executionMode != runtime.thirdPersonView->executionMode ||
            (pending->second.firstPerson && runtime.firstPersonView &&
                ((!pending->second.firstPerson->graphFingerprint.empty() &&
                  pending->second.firstPerson->graphFingerprint != runtime.firstPersonView->sourceFingerprint) ||
                 pending->second.firstPerson->executionMode != runtime.firstPersonView->executionMode));
        if (changed) {
            m_pendingDiagnostics.push_back("discarded incompatible animation state for " + object.toString() +
                ": resolved rig/asset fingerprint changed; equipment ownership retained");
            if (const auto* owner = m_world.find(object)) {
                WorldCommand cancel;
                cancel.type = WorldCommandType::SetEquipmentState; cancel.target = object;
                cancel.equipment = owner->equipment;
                cancel.equipment.transitioning = false;
                cancel.equipment.requestedDrawn = cancel.equipment.drawn;
                (void)m_world.queue(std::move(cancel));
                if (owner->combatState && owner->combatState->pendingMelee) {
                    WorldCommand combat;
                    combat.type = WorldCommandType::SetCombatState; combat.target = object;
                    combat.combatState = *owner->combatState;
                    combat.combatState.pendingMelee = false;
                    combat.combatState.pendingClip.clear();
                    combat.combatState.pendingDamage = 0;
                    (void)m_world.queue(std::move(combat));
                }
            }
        } else if (!runtime.thirdPerson.restore(pending->second.thirdPerson, outError) ||
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
    auto configured = physicsConfig;
    if (m_config.game == importer::bethesda::BethesdaGame::SkyrimSpecialEdition) {
        if (m_config.characterMovement.has_value()) {
            configured.movement = *m_config.characterMovement;
        }
        configured.movement.enabled = true;
    }
    if (!m_physics.addCharacter(object, configured, outError)) return false;
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
    AnimationActorSnapshot saved{object, found->second.thirdPerson.snapshot(),
        std::nullopt, found->second.bodyMorph};
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

bool BethesdaSession::addActorImpulse(
    ObjectId object, const odai::math::Vector3& velocityChange) {
    return m_physics.addCharacterImpulse(object, velocityChange);
}

bool BethesdaSession::setActorAnimationInput(
    ObjectId object, odai::anim::AnimationInputState input) {
    const auto found = m_actorAnimations.find(object);
    if (found == m_actorAnimations.end()) return false;
    input.attacking = input.attacking || found->second.input.attacking;
    input.equipping = input.equipping || found->second.input.equipping;
    input.events.insert(input.events.begin(), found->second.input.events.begin(), found->second.input.events.end());
    found->second.input = std::move(input);
    return true;
}

bool BethesdaSession::setActorBodyMorphState(
    ObjectId object, odai::anim::BodyMorphSnapshot state) {
    const auto found = m_actorAnimations.find(object);
    if (found == m_actorAnimations.end()) return false;
    for (const auto& [name, value] : state.sliders) {
        if (name.empty() || !std::isfinite(value) || value < 0.0f || value > 1.0f)
            return false;
    }
    found->second.bodyMorph = std::move(state);
    return true;
}

bool BethesdaSession::queueActorAnimationEvent(
    ObjectId object, odai::anim::AnimationEvent event) {
    if (event.name == "DrawStart" || event.name == "SheatheStart") {
        std::string error;
        return requestActorWeaponDraw(object, event.name == "DrawStart", error);
    }
    const auto found = m_actorAnimations.find(object);
    if (found == m_actorAnimations.end()) return false;
    found->second.thirdPerson.queueEvent(event);
    if (found->second.firstPersonView != nullptr) found->second.firstPerson.queueEvent(std::move(event));
    return true;
}

bool BethesdaSession::equipActorItem(ObjectId actor, const RecordKey& item,
    bool equipped, bool leftHand, std::string& error, bool equipmentPolicy, bool scriptOverride) {
    const auto* owner = m_world.find(actor);
    const auto* definition = skyrimItem(item);
    if (!owner || !definition || owner->kind != RuntimeObjectKind::Actor || !owner->enabled ||
        (owner->actorValues && owner->actorValues->dead) ||
        std::none_of(owner->inventory.begin(), owner->inventory.end(), [&](const auto& entry) {
            return entry.item == item && entry.count > 0;
        })) { error = "Equipment requires a living owner and an owned item"; return false; }
    std::uint64_t slots = 0;
    if (definition->recordType == "ARMO") slots = definition->bipedSlots;
    else if (definition->recordType == "AMMO") slots = kEquipmentAmmo;
    else if (definition->recordType == "WEAP") {
        const auto type = definition->weaponAnimationType;
        slots = (type == 5 || type == 6 || type == 7 || type == 9)
            ? kEquipmentRightHand | kEquipmentLeftHand
            : leftHand ? kEquipmentLeftHand : kEquipmentRightHand;
    }
    // Skyrim's shield biped slot also owns the left hand.
    if (definition->recordType == "ARMO" && (definition->bipedSlots & (1u << 9)))
        slots |= kEquipmentLeftHand;
    if (slots == 0) { error = "Item has no supported equipment slots"; return false; }
    WorldCommand command;
    command.type = WorldCommandType::SetEquipped;
    command.target = actor; command.item = item; command.equipped = equipped;
    command.equipmentSlots = slots;
    command.equipmentPolicy = equipmentPolicy; command.overrideEquipmentLocks = scriptOverride;
    (void)m_world.queue(std::move(command));
    error.clear(); return true;
}

bool BethesdaSession::requestActorWeaponDraw(ObjectId actor, bool drawn, std::string& error, bool combatDraw) {
    const auto* owner = m_world.find(actor);
    const auto animation = m_actorAnimations.find(actor);
    if (!owner || animation == m_actorAnimations.end() ||
        (owner->actorValues && owner->actorValues->dead)) {
        error = "Weapon draw requires a living animated actor"; return false;
    }
    if (owner->equipment.requestedDrawn == drawn &&
        (owner->equipment.transitioning || owner->equipment.drawn == drawn)) {
        error.clear(); return true;
    }
    auto state = owner->equipment;
    state.requestedDrawn = drawn; state.transitioning = true;
    state.combatDraw = combatDraw && drawn;
    WorldCommand command;
    command.type = WorldCommandType::SetEquipmentState; command.target = actor;
    command.equipment = state; (void)m_world.queue(std::move(command));
    animation->second.input.equipping = true;
    animation->second.thirdPerson.queueEvent({drawn ? "DrawStart" : "SheatheStart", {}});
    if (animation->second.firstPersonView) animation->second.firstPerson.queueEvent({drawn ? "DrawStart" : "SheatheStart", {}});
    error.clear(); return true;
}

bool BethesdaSession::useInventoryItem(ObjectId actor, const RecordKey& item, std::string& error) {
    const auto* definition = skyrimItem(item);
    const auto* owner = m_world.find(actor);
    if (!definition || !owner || !owner->actorValues || owner->actorValues->dead ||
        !owner->enabled || owner->kind != RuntimeObjectKind::Actor ||
        std::none_of(owner->inventory.begin(), owner->inventory.end(), [&](const auto& entry) {
            return entry.item == item && entry.count > 0;
        })) { error = "Item is not available to a living actor"; return false; }
    WorldCommand command;
    command.target = actor;
    command.item = item;
    if (definition->recordType == "WEAP" || definition->recordType == "ARMO" || definition->recordType == "AMMO") {
        const auto owned = std::find_if(owner->inventory.begin(), owner->inventory.end(),
            [&](const auto& entry) { return entry.item == item; });
        return equipActorItem(actor, item, !owned->equipped, false, error);
    } else if (definition->healing > 0 && std::isfinite(definition->healing)) {
        for (const auto& [name, quest] : m_quests) {
            (void)name;
            for (const auto& alias : quest.aliases) {
                if (alias.createdObject == item && quest.running && !quest.completed) {
                    error = "This item is needed by an active quest"; return false;
                }
            }
        }
        if (owner->actorValues->health >= owner->actorValues->maxHealth) {
            error = "Health is already full"; return false;
        }
        command.type = WorldCommandType::ConsumeHealingItem;
        command.actorValueDelta = definition->healing;
    } else { error = "This item's effects are not supported yet"; return false; }
    (void)m_world.queue(std::move(command));
    error.clear();return true;
}

bool BethesdaSession::actorGuarding(ObjectId actor) const {
    const auto* object = m_world.find(actor);
    return m_actorGuards.contains(actor) && object && object->enabled && object->equipment.drawn &&
        (!object->actorValues || (!object->actorValues->dead && object->actorValues->stamina >= 5)) &&
        std::any_of(object->inventory.begin(), object->inventory.end(), [](const auto& item) {
            return item.equipped && item.count > 0 && (item.equipmentSlots & (1ull << 9));
        });
}
void BethesdaSession::setActorGuard(ObjectId actor, bool held, const odai::math::Vector3& forward) {
    const float length = odai::math::length(forward);
    if (held && std::isfinite(length) && length > 1e-5f)
        m_actorGuards[actor] = forward * (1.0f / length);
    else m_actorGuards.erase(actor);
}
std::vector<MeleeContactEvent> BethesdaSession::takeMeleeContacts() {
    std::vector<MeleeContactEvent> result;
    result.swap(m_meleeContacts);
    return result;
}

MeleeAttackResult BethesdaSession::performEquippedMeleeAttack(
    ObjectId actor, const odai::math::Vector3& forward) {
    float damage = 25.0f;
    if (const auto* owner = m_world.find(actor)) {
        for (const auto& entry : owner->inventory) {
            if (!entry.equipped || entry.count <= 0) continue;
            if (const auto* definition = skyrimItem(entry.item); definition && definition->meleeDamage > 0) {
                damage = definition->meleeDamage;break;
            }
        }
    }
    return performMeleeAttack(actor, forward, damage);
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
    if (source->equipment.initialized && !source->equipment.drawn && m_actorAnimations.contains(attacker) &&
        std::any_of(source->inventory.begin(), source->inventory.end(), [&](const auto& entry) {
            const auto* item = skyrimItem(entry.item);
            return entry.equipped && entry.count > 0 && item && item->recordType == "WEAP";
        })) {
        if (!source->equipment.transitioning) {
            std::string error;
            (void)requestActorWeaponDraw(attacker, true, error);
        }
        result.diagnostic = "equipped weapon must finish drawing before melee contact";
        return result;
    }
    if (actorGuarding(attacker)) {
        result.diagnostic = "cannot attack while guarding";
        return result;
    }
    const std::uint64_t tick = m_clock.tick();
    if (const auto animated = m_actorAnimations.find(attacker);
        animated != m_actorAnimations.end() && animated->second.requestedMelee) {
        result.diagnostic = "melee action already requested this tick";
        return result;
    }
    RuntimeCombatState combat = source->combatState.value_or(RuntimeCombatState{});
    if (combat.pendingMelee || tick < combat.nextMeleeAttackTick) {
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
    const auto animation = m_actorAnimations.find(attacker);
    const odai::anim::AnimationClip* contactClip = nullptr;
    if (animation != m_actorAnimations.end()) {
        const auto& view = *animation->second.thirdPersonView;
        std::string state = "attack";
        if (!animation->second.input.weaponStyle.empty() &&
            view.stateClips.contains(state + "_" + animation->second.input.weaponStyle))
            state += "_" + animation->second.input.weaponStyle;
        const auto mapped = view.stateClips.find(state);
        const std::string name = mapped == view.stateClips.end() ? state : mapped->second;
        for (const auto& clip : view.clips)
            if (clip.name == name && !clip.loop &&
                std::any_of(clip.annotations.begin(), clip.annotations.end(), [](const auto& event) {
                    return event.name == "HitFrame" || event.name == "hitFrame";
                })) { contactClip = &clip; break; }
    }
    const bool nativeSelection = animation != m_actorAnimations.end() &&
        animation->second.thirdPersonView->executionMode == odai::anim::AnimationExecutionMode::Native &&
        animation->second.thirdPersonView->nativeProgram &&
        !animation->second.thirdPersonView->nativeProgram->rules.empty();
    if (contactClip || nativeSelection) {
        result.deferred = true;
        combat.pendingMelee = true;
        combat.pendingDamage = damage;
        combat.pendingRange = rangeBethesdaUnits;
        combat.pendingForward = {forward.x, forward.y, forward.z};
        combat.pendingClip = nativeSelection ? "" : contactClip->name;
        animation->second.requestedMelee = combat;
    } else {
        result = resolveMeleeContact(attacker, forward, damage, rangeBethesdaUnits, combat);
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
        if (!contactClip) animated->second.thirdPerson.queueEvent({"weaponSwing", "right"});
        if (!contactClip && animated->second.firstPersonView != nullptr) {
            animated->second.firstPerson.queueEvent({"weaponSwing", "right"});
        }
    }
    return result;
}

MeleeAttackResult BethesdaSession::resolveMeleeContact(ObjectId attacker,
    const odai::math::Vector3& forward, float damage, float rangeBethesdaUnits, RuntimeCombatState& combat) {
    MeleeAttackResult result;
    result.accepted = true;
    result.damage = damage;
    for (const PhysicsMeleeCandidate& candidate :
         m_physics.meleeCandidates(attacker, forward, rangeBethesdaUnits)) {
        const RuntimeObject* target = m_world.find(candidate.object);
        if (target == nullptr || target->kind != RuntimeObjectKind::Actor ||
            !target->enabled || target->ghost ||
            (target->actorValues.has_value() && target->actorValues->dead)) {
            continue;
        }
        const bool blocked = actorGuarding(candidate.object) &&
            odai::math::dot(m_actorGuards.at(candidate.object), odai::math::normalize(forward)) < -0.35f;
        if (blocked) {
            damage *= 0.3f;
            WorldCommand stamina;
            stamina.type = WorldCommandType::AdjustActorValue;
            stamina.target = candidate.object;
            stamina.actorValue = ActorValue::Stamina;
            stamina.actorValueDelta = -5.0f;
            (void)m_world.queue(std::move(stamina));
        } else (void)queueActorAnimationEvent(candidate.object, {"staggerStart", {}});
        result.damage = damage;
        if (m_meleeContacts.size() < 256) m_meleeContacts.push_back({attacker, true, blocked});
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
        // A strike changes physical momentum as well as actor values. Because
        // this lands on CharacterVirtual's external velocity channel, it can
        // carry the target beyond a ledge and gravity owns the rest of the
        // fall instead of navigation snapping them back onto the mesh.
        const odai::math::Vector3 strikeDirection = odai::math::normalize(forward);
        const float horizontalKick = std::clamp(160.0f + damage * 8.0f, 200.0f, 520.0f);
        if (!blocked) (void)m_physics.addCharacterImpulse(candidate.object,
            {strikeDirection.x * horizontalKick, 150.0f,
             strikeDirection.z * horizontalKick});
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
    if (!result.hit) {
        if (const auto source = m_physics.characterState(attacker)) {
            const auto origin = source->position + odai::math::Vector3{0, 64, 0};
            if (m_physics.castSphere(origin, origin + odai::math::normalize(forward) * rangeBethesdaUnits,
                    4.0f, attacker) && m_meleeContacts.size() < 256)
                m_meleeContacts.push_back({attacker, false, false});
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

LootTransferResult BethesdaSession::lootObject(ObjectId player, ObjectId source,
    const RecordKey& onlyItem, std::int32_t count) {
    LootTransferResult result;
    const RuntimeObject* playerObject = m_world.find(player);
    const RuntimeObject* sourceObject = m_world.find(source);
    if (playerObject == nullptr || playerObject->kind != RuntimeObjectKind::Actor ||
        sourceObject == nullptr || source == player ||
        (playerObject->actorValues && playerObject->actorValues->dead) ||
        (onlyItem.valid() && count <= 0) ||
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
        if (!entry.item.valid() || entry.count <= 0 || (onlyItem.valid() && entry.item != onlyItem)) continue;
        if (onlyItem.valid() && entry.count < count) { result.accepted = false; result.diagnostic = "Not enough items"; return result; }
        result.transferred.push_back({entry.item, onlyItem.valid() ? count : entry.count, false});
    }
    std::sort(result.transferred.begin(), result.transferred.end(),
        [](const InventoryEntry& left, const InventoryEntry& right) {
            return left.item < right.item;
        });
    for (const InventoryEntry& entry : result.transferred) {
        WorldCommand transfer;
        transfer.type = WorldCommandType::TransferItem;
        transfer.target = source;
        transfer.other = player;
        transfer.item = entry.item;
        transfer.itemCount = entry.count;
        (void)m_world.queue(std::move(transfer));
    }
    if (result.transferred.empty()) { result.diagnostic = "Nothing to take"; result.accepted = !onlyItem.valid(); }
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
        for (const ObjectId& residentId : m_world.orderedObjectIds()) {
            const RuntimeObject* object = m_world.find(residentId);
            if (object != nullptr && object->base == id.reference) return object;
        }
        return nullptr;
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
            case 1u: { // GetDistance
                auto other = resolveForm(condition.parameter1);
                if (condition.useAliases) {
                    other.reset();
                    if (const auto* quest = findQuest(ObjectId::persistent(info.quest)))
                        for (const auto& alias : quest->aliases)
                            if (alias.id == static_cast<std::int32_t>(condition.parameter1)) other = alias.target;
                }
                const auto* object = other ? liveObject(*other) : nullptr;
                if (!object || !targetObject) return std::nullopt;
                double distance = 0;
                for (unsigned i = 0; i < 3; ++i) { const double d = object->transform.position[i] - targetObject->transform.position[i]; distance += d*d; }
                return static_cast<float>(std::sqrt(distance));
            }
            case 249u: return targetObject ? (targetObject->inDialogueWithPlayer ? 1.f : 0.f) : 0.f;
            case 74u: { // GetGlobalValue
                const auto global = resolveForm(condition.parameter1);
                if (!global) return std::nullopt;
                const auto value = m_globalVariables.find(global->reference);
                return value == m_globalVariables.end() ? std::optional<float>{} : value->second;
            }
            case 71u: { // GetInFaction
                const auto faction = resolveForm(condition.parameter1);
                if (!faction || !targetObject) return std::nullopt;
                return std::find(targetObject->factions.begin(), targetObject->factions.end(), faction->reference) != targetObject->factions.end() ? 1.f : 0.f;
            }
            case 300u: return targetObject ? (targetObject->interior ? 1.f : 0.f) : std::optional<float>{};
            case 550u: { // IsSceneActionComplete
                const auto scene = resolveForm(condition.parameter1);
                if (!scene) return std::nullopt;
                const auto progress = m_sceneProgress.find(scene->reference);
                return progress != m_sceneProgress.end() && progress->second.completed.contains(condition.parameter2) ? 1.f : 0.f;
            }
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
                for (const ObjectId& actorId : m_world.orderedActorIds()) {
                    const RuntimeObject* actor = m_world.find(actorId);
                    if (actor != nullptr && actor->base == expected->reference &&
                        actor->actorValues.has_value() && actor->actorValues->dead) ++dead;
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
        // Explicit TCLT links may be NPC-only continuation topics. They
        // have no player prompt but still carry authored responses/effects.
        if (topic.prompt.empty() && eligibleTopics.empty()) continue;
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
    bool strict, bool ambient) {
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
    speakerContext.inDialogueWithPlayer = !ambient;
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

odai::anim::AnimationStepOutput BethesdaSession::interpolatedActorAnimationOutput(
    ObjectId object, float alpha, bool firstPerson) const {
    const auto found = m_actorAnimations.find(object);
    if (found == m_actorAnimations.end()) return {};
    const odai::anim::AnimationStepOutput& current =
        firstPerson && found->second.firstPersonView != nullptr
            ? found->second.firstPersonOutput : found->second.thirdPersonOutput;
    const odai::anim::AnimationStepOutput& previous =
        firstPerson && found->second.firstPersonView != nullptr
            ? found->second.previousFirstPersonOutput
            : found->second.previousThirdPersonOutput;
    if (previous.pose.empty() || current.resetHistory) return current;
    const auto& view = firstPerson && found->second.firstPersonView
        ? found->second.firstPersonView : found->second.thirdPersonView;
    if (view && view->skeleton && !previous.localPose.empty() && previous.localPose.size() == current.localPose.size()) {
        auto result = current;
        const float weight = std::isfinite(alpha) ? std::clamp(alpha,0.f,1.f) : 1.f;
        result.localPose = odai::anim::blendLocalPoses(previous.localPose,current.localPose,weight);
        odai::anim::AnimationSampler sampler;
        if (view->inverseBindMatrices.empty()) sampler.bindSkeleton(*view->skeleton);
        else sampler.bindSkeleton(*view->skeleton,view->inverseBindMatrices);
        sampler.paletteFromLocal(*view->skeleton,result.localPose,result.pose);
        const auto world = odai::anim::composePoseWorld(*view->skeleton,result.localPose);
        for (auto& [name,transform] : result.socketTransforms)
            if (const int bone = view->skeleton->findBone(name); bone >= 0) transform = world[bone];
        result.evaluationPacket.reset();
        if (previous.evaluationPacket && current.evaluationPacket &&
            previous.evaluationPacket->instructions.size()+current.evaluationPacket->instructions.size()+1 <= 256) {
            auto packet = std::make_shared<odai::anim::PoseEvaluationPacket>(*previous.evaluationPacket);
            const auto offset = static_cast<std::uint32_t>(packet->instructions.size());
            for (auto inst : current.evaluationPacket->instructions) {
                for (auto& index : inst.inputs) index += offset;
                packet->instructions.push_back(std::move(inst));
            }
            odai::anim::PoseGraphInstruction blend;
            blend.kind = odai::anim::PoseGraphNode::Kind::Blend;
            blend.inputs = {packet->output,current.evaluationPacket->output+offset};
            blend.weight = weight;
            packet->output = static_cast<std::uint32_t>(packet->instructions.size());
            packet->instructions.push_back(std::move(blend));
            result.evaluationPacket = std::move(packet);
        }
        return result;
    }
    return odai::anim::BehaviorGraphInstance::interpolate(previous, current, alpha);
}

std::vector<AnimationActorSnapshot> BethesdaSession::animationSnapshots() const {
    std::vector<AnimationActorSnapshot> result;
    result.reserve(m_actorAnimations.size() + m_pendingAnimationSnapshots.size());
    for (const auto& [object, runtime] : m_actorAnimations) {
        AnimationActorSnapshot saved{object, runtime.thirdPerson.snapshot(), std::nullopt,
            runtime.bodyMorph};
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
        runtime.requestedMelee.reset();
        if (!runtime.bodyMorph.topologyFingerprint.empty() &&
            !saved->second.bodyMorph.topologyFingerprint.empty() &&
            runtime.bodyMorph.topologyFingerprint !=
                saved->second.bodyMorph.topologyFingerprint) {
            saved->second.bodyMorph = runtime.bodyMorph;
            m_pendingDiagnostics.push_back(
                "discarded incompatible body morph state for " + object.toString());
        }
        const bool changed = (!saved->second.thirdPerson.graphFingerprint.empty() &&
                saved->second.thirdPerson.graphFingerprint != runtime.thirdPersonView->sourceFingerprint) ||
            saved->second.thirdPerson.executionMode != runtime.thirdPersonView->executionMode ||
            (saved->second.firstPerson && runtime.firstPersonView &&
                ((!saved->second.firstPerson->graphFingerprint.empty() &&
                  saved->second.firstPerson->graphFingerprint != runtime.firstPersonView->sourceFingerprint) ||
                 saved->second.firstPerson->executionMode != runtime.firstPersonView->executionMode));
        if (changed) {
            if (!runtime.thirdPerson.bind(*runtime.thirdPersonView, outError) ||
                (runtime.firstPersonView && !runtime.firstPerson.bind(*runtime.firstPersonView, outError))) return false;
            m_pendingDiagnostics.push_back("discarded incompatible animation state for " + object.toString());
            m_pendingAnimationSnapshots.erase(saved);
            continue;
        }
        if (saved->second.firstPerson.has_value() != (runtime.firstPersonView != nullptr) ||
            !runtime.thirdPerson.restore(saved->second.thirdPerson, outError) ||
            (saved->second.firstPerson.has_value() &&
             !runtime.firstPerson.restore(*saved->second.firstPerson, outError))) return false;
        runtime.bodyMorph = saved->second.bodyMorph;
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
            !std::isfinite(saved.groundNormal.z) || !validCharacterMovementState(saved.movement)) {
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

bool BethesdaSession::restoreRagdollSnapshots(
    std::span<const PhysicsRagdollSnapshot> snapshots, std::string& outError) {
    for (const PhysicsRagdollSnapshot& current : m_physics.ragdollSnapshots())
        (void)m_physics.removeRagdoll(current.object);
    std::set<ObjectId> seen;
    for (const PhysicsRagdollSnapshot& saved : snapshots) {
        if (!saved.object.valid() || !seen.insert(saved.object).second ||
            !m_physics.hasCharacter(saved.object) ||
            !m_physics.restoreRagdoll(saved, outError)) {
            if (outError.empty()) outError = "saved ragdoll is invalid or duplicated";
            return false;
        }
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
            importer::bethesda::bethesdaGameName(scenario.game) + ", not " +
            importer::bethesda::bethesdaGameName(m_config.game);
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
    m_questJournal[definition.record] = {definition.title, definition.stages};
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
        if (condition.function == 58u || condition.function == 59u) {
            if (!m_resolvedFormResolver) return std::nullopt;
            const auto id = m_resolvedFormResolver(condition.parameter1);
            const auto* target = id ? findQuest(*id) : nullptr;
            if (!target) return std::nullopt;
            if (condition.function == 58u) return static_cast<float>(target->stage);
            return std::find(target->completedStages.begin(), target->completedStages.end(),
                static_cast<std::int32_t>(condition.parameter2)) != target->completedStages.end() ? 1.f : 0.f;
        }
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

std::string BethesdaSession::questJournalTitle(const QuestRuntimeState& quest) const {
    const auto found = m_questJournal.find(quest.record);
    return found != m_questJournal.end() && !found->second.title.empty() ? found->second.title : quest.editorId;
}

std::string BethesdaSession::resolveQuestText(const QuestRuntimeState& quest, std::string text) const {
    std::size_t offset = 0;
    while ((offset = text.find("<Alias=", offset)) != std::string::npos) {
        const auto end = text.find('>', offset);
        if (end == std::string::npos) break;
        const auto name = text.substr(offset + 7, end - offset - 7);
        std::string replacement = name;
        for (const auto& alias : quest.aliases) {
            if (normalizedEditorId(alias.name) != normalizedEditorId(name)) continue;
            const auto* object = m_world.find(alias.target);
            const RecordKey key = object ? object->base : alias.target.reference;
            if (const auto* definition = skyrimItem(key); definition && !definition->name.empty()) replacement = definition->name;
            break;
        }
        text.replace(offset, end - offset + 1, replacement);
        offset += replacement.size();
    }
    return text;
}

std::string BethesdaSession::questJournalSummary(const QuestRuntimeState& quest) const {
    const auto found = m_questJournal.find(quest.record);
    if (found == m_questJournal.end()) return {};
    for (auto stage = found->second.stages.rbegin(); stage != found->second.stages.rend(); ++stage) {
        if (std::find(quest.completedStages.begin(), quest.completedStages.end(), stage->index) == quest.completedStages.end()) continue;
        for (const auto& entry : stage->logEntries) {
            if (entry.text.empty()) continue;
            SkyrimDialogueInfoDefinition info;
            info.quest = quest.record; info.conditions = entry.conditions;
            if (!evaluateDialogueConditions(info, m_playerObject, m_playerObject, true).matched) continue;
            return resolveQuestText(quest, entry.text);
        }
    }
    return {};
}

void BethesdaSession::setScenePlaying(const RecordKey& scene, bool playing) {
    if (!scene.valid()) return;
    const bool wasPlaying = m_scenes.contains(scene) && m_scenes.at(scene);
    m_scenes.insert_or_assign(scene, playing);
    if (playing && !wasPlaying) m_sceneProgress[scene] = {};
    if (!playing) std::erase_if(m_sceneSpeech, [&](const auto& line) { return line.scene == scene; });
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

void BethesdaSession::simulateTick(
    std::uint64_t tick, double stepSeconds, BethesdaSessionStep& result) {
    const float fixedDelta = static_cast<float>(stepSeconds);
    advanceLivingWorld(stepSeconds, result);
    queueCombatActions();
    advanceActorAnimations(fixedDelta, result);
    queuePhysicsTransforms(fixedDelta);
    advanceScriptsAndApplyCommands(tick, stepSeconds, result);
}

void BethesdaSession::advanceLivingWorld(double stepSeconds, BethesdaSessionStep& result) {
    LivingWorldStep living = m_livingWorld.advance(stepSeconds, m_world,
        [&](const std::array<double, 3>& from, const std::array<double, 3>& to) {
            return m_physics.hasLineOfSight(
                {static_cast<float>(from[0]), static_cast<float>(from[1]),
                 static_cast<float>(from[2])},
                {static_cast<float>(to[0]), static_cast<float>(to[1]),
                 static_cast<float>(to[2])});
        });
    result.livingWorld.actorsEvaluated += living.actorsEvaluated;
    result.livingWorld.activityChanges += living.activityChanges;
    result.livingWorld.offscreenReconciliations += living.offscreenReconciliations;
    result.livingWorld.physicalResets += living.physicalResets;
    result.livingWorld.witnesses.insert(result.livingWorld.witnesses.end(),
        std::make_move_iterator(living.witnesses.begin()),
        std::make_move_iterator(living.witnesses.end()));
    result.diagnostics.insert(result.diagnostics.end(),
        std::make_move_iterator(living.diagnostics.begin()),
        std::make_move_iterator(living.diagnostics.end()));
}

void BethesdaSession::queueCombatActions() {
    // Combat packages and Actor.StartCombat converge here. Rendering never
    // selects targets: fixed-tick AI aims from one Jolt character to the other
    // and uses the same cone/occlusion/cooldown path as player input.
    for (const ObjectId& actorId : m_world.orderedActorIds()) {
        const RuntimeObject* actorObject = m_world.find(actorId);
        if (actorObject == nullptr) continue;
        const RuntimeObject& actor = *actorObject;
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
}

void BethesdaSession::advanceActorAnimations(float fixedDelta, BethesdaSessionStep& result) {
    for (auto& [object, runtime] : m_actorAnimations) {
        auto locomotionVelocity = runtime.input.requestedVelocity;
        if (const auto physical = m_physics.characterState(object)) {
            runtime.input.teleported = odai::math::length(physical->position - runtime.input.actorPosition) > 64.f;
            runtime.input.actorPosition = physical->position;
            locomotionVelocity = physical->velocity - physical->groundVelocity;
            runtime.input.grounded = physical->grounded;
            runtime.input.groundVelocity = physical->groundVelocity;
            runtime.input.groundNormal = physical->groundNormal;
            runtime.input.verticalVelocity = physical->velocity.y;
            runtime.input.falling = physical->falling;
            runtime.input.landed = physical->landed;
            runtime.input.blocked = physical->blocked;
            static constexpr const char* phases[]{"grounded", "takeoff", "ascending", "apex", "falling", "landing"};
            static constexpr const char* impacts[]{"light", "hard", "stagger", "severe"};
            runtime.input.jumpPhase = phases[static_cast<unsigned>(physical->jumpPhase)];
            runtime.input.landingSeverity = impacts[static_cast<unsigned>(physical->landingSeverity)];
            runtime.input.landingImpactMetres = physical->landingImpactMetres;
            runtime.input.movementSpeed = odai::math::length(odai::math::Vector3{
                locomotionVelocity.x, 0.0f, locomotionVelocity.z});
        }
        if (const auto* actor = m_world.find(object)) {
            runtime.input.dead = actor->actorValues && actor->actorValues->dead;
            runtime.input.selectorContext.values["combat"] = actor->combatState && actor->combatState->combatTarget.valid();
            runtime.input.selectorContext.values["interior"] = actor->interior;
            if (actor->actorValues) {
                const auto& values = *actor->actorValues;
                runtime.input.selectorContext.values["injury"] = static_cast<double>(
                    1.f - std::clamp(values.health / std::max(1.f, values.maxHealth), 0.f, 1.f));
                runtime.input.selectorContext.values["fatigue"] = static_cast<double>(
                    1.f - std::clamp(values.stamina / std::max(1.f, values.maxStamina), 0.f, 1.f));
            }
            runtime.input.weaponStyle.clear();
            for (const auto& entry : actor->inventory) {
                if (!entry.equipped) continue;
                const auto* item = skyrimItem(entry.item);
                if (!item || item->recordType != "WEAP") continue;
                const auto type = item->weaponAnimationType;
                runtime.input.weaponStyle = type >= 1 && type <= 4 ? "1hm" :
                    type == 5 ? "2hm" : type == 6 ? "2hw" : type == 7 ? "bow" :
                    type == 8 ? "staff" : type == 9 ? "crossbow" : "h2h";
                break;
            }
            if (actor->combatState && actor->combatState->combatTarget.valid() &&
                !actor->equipment.drawn && !actor->equipment.transitioning) {
                std::string drawError;
                (void)requestActorWeaponDraw(object, true, drawError, true);
            }
            if (actor->equipment.combatDraw && actor->equipment.drawn && !actor->equipment.transitioning &&
                (!actor->combatState || !actor->combatState->combatTarget.valid())) {
                std::string sheathError;
                (void)requestActorWeaponDraw(object, false, sheathError);
            }
            runtime.input.weaponDrawn = actor->equipment.drawn;
            runtime.input.variables["bIsWeaponDrawn"] = actor->equipment.drawn ? 1.f : 0.f;
        }
        const float yaw = runtime.input.actorYawRadians;
        const float c = std::cos(yaw), s = std::sin(yaw);
        const auto velocity = locomotionVelocity;
        runtime.input.localVelocity = {c * velocity.x - s * velocity.z, velocity.y,
            s * velocity.x + c * velocity.z};
        runtime.input.ragdollActive = m_physics.ragdollSnapshot(object).has_value();
        runtime.input.footContacts = {};
        if (runtime.thirdPersonView->humanoidRig && runtime.input.grounded && !runtime.input.ragdollActive) {
            const auto& rig = *runtime.thirdPersonView->humanoidRig;
            const auto& palette = runtime.thirdPersonOutput.pose;
            const auto& inverseBind = runtime.thirdPersonView->inverseBindMatrices;
            const auto world = odai::math::Matrix4::translation(runtime.input.actorPosition) *
                odai::math::Matrix4::rotationY(yaw);
            runtime.input.footIkEnabled = true;
            for (std::size_t foot = 0; foot < 2; ++foot) {
                const auto chain = rig.limbs.find(foot == 0 ? "left_leg" : "right_leg");
                if (chain == rig.limbs.end()) continue;
                const auto bone = static_cast<std::size_t>(chain->second.end);
                if (bone >= palette.size() || bone >= inverseBind.size()) continue;
                const auto position = odai::math::transformPoint(world * palette[bone] * odai::math::inverse(inverseBind[bone]), {});
                if (const auto hit = m_physics.castDown(position + odai::math::Vector3{0,18,0}, 36.f);
                    hit && hit->normal.y >= .65f)
                    runtime.input.footContacts[foot] = {true, hit->position, hit->normal};
            }
        }
        runtime.previousThirdPersonOutput = runtime.thirdPersonOutput;
        runtime.thirdPersonOutput = runtime.thirdPerson.step(runtime.input, fixedDelta);
        if (const auto* owner = m_world.find(object); owner && owner->equipment.transitioning) {
            auto state = owner->equipment;
            const auto& output = runtime.thirdPersonOutput;
            for (const auto& event : output.clipEvents) {
                const bool attach = state.requestedDrawn
                    ? (event.name == "weaponDraw" || event.name == "WeaponDraw")
                    : (event.name == "weaponSheathe" || event.name == "WeaponSheathe");
                if (attach && state.transitioning) {
                    state.drawn = state.requestedDrawn;
                    state.transitioning = false;
                }
            }
            if (runtime.input.dead || (!output.actionActive && !runtime.input.equipping && state.transitioning)) {
                result.diagnostics.push_back("equipment attachment event missing or interrupted for " + object.toString());
                state.transitioning = false;
                state.requestedDrawn = state.drawn;
            }
            if (!(state == owner->equipment)) {
                WorldCommand command;
                command.type = WorldCommandType::SetEquipmentState; command.target = object;
                command.equipment = state; (void)m_world.queue(std::move(command));
            }
        }
        const auto* contactActor = m_world.find(object);
        if (runtime.requestedMelee || (contactActor && contactActor->combatState && contactActor->combatState->pendingMelee)) {
            RuntimeCombatState combat = runtime.requestedMelee
                ? *runtime.requestedMelee : *contactActor->combatState;
            runtime.requestedMelee.reset();
            const auto& output = runtime.thirdPersonOutput;
            // Bind a native request to the actual selected action before consuming
            // its first marker. Only the third-person authority deals damage.
            if (combat.pendingClip.empty() && output.activeState.starts_with("attack"))
                combat.pendingClip = output.activeClip;
            const bool contact = !runtime.input.dead && output.activeClip == combat.pendingClip &&
                std::any_of(output.clipEvents.begin(), output.clipEvents.end(), [](const auto& event) {
                    return event.name == "HitFrame" || event.name == "hitFrame";
                });
            const bool interrupted = runtime.input.dead || output.activeClip != combat.pendingClip || !output.actionActive;
            if (contact || interrupted) {
                if (contact) (void)resolveMeleeContact(object,
                    {combat.pendingForward[0], combat.pendingForward[1], combat.pendingForward[2]},
                    combat.pendingDamage, combat.pendingRange, combat);
                combat.pendingMelee = false;
                combat.pendingDamage = combat.pendingRange = 0;
                combat.pendingForward = {};
                combat.pendingClip.clear();
                WorldCommand command;
                command.type = WorldCommandType::SetCombatState;
                command.target = object;
                command.combatState = std::move(combat);
                (void)m_world.queue(std::move(command));
            }
        }
        if (runtime.firstPersonView != nullptr) {
            runtime.previousFirstPersonOutput = runtime.firstPersonOutput;
            auto presentationInput = runtime.input;
            presentationInput.ownsGameplayEvents = false;
            presentationInput.selectorContext.values["view"] = std::string("first_person");
            if (runtime.thirdPersonView->executionMode == odai::anim::AnimationExecutionMode::Native &&
                runtime.firstPersonView->executionMode == odai::anim::AnimationExecutionMode::Native) {
                const auto clock = runtime.thirdPerson.snapshot();
                presentationInput.sharedState = clock.state;
                presentationInput.sharedStateTime = clock.stateTime;
                presentationInput.sharedActionIdentity = clock.actionIdentity;
            }
            runtime.firstPersonOutput = runtime.firstPerson.step(presentationInput, fixedDelta);
        }
        PhysicsCharacterInput input;
        input.desiredVelocity = runtime.input.requestedVelocity;
        input.jumpRequested = runtime.input.jumpRequested;
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
}

void BethesdaSession::queuePhysicsTransforms(float fixedDelta) {
    for (const auto& [object, physical] : m_physics.step(fixedDelta)) {
        const RuntimeObject* resident = m_world.find(object);
        if (resident == nullptr) continue;
        WorldCommand command;
        command.type = WorldCommandType::SetPosition;
        command.target = object;
        command.transform.position = {physical.position.x, physical.position.y, physical.position.z};
        (void)m_world.queue(std::move(command));
    }
    for (const PhysicsDynamicBodySnapshot& dynamic : m_physics.dynamicBodySnapshots()) {
        const RuntimeObject* resident = m_world.find(dynamic.object);
        if (resident == nullptr) continue;
        WorldCommand transform;
        transform.type = WorldCommandType::SetPosition;
        transform.target = dynamic.object;
        transform.transform.position = {
            dynamic.position.x, dynamic.position.y, dynamic.position.z};
        (void)m_world.queue(std::move(transform));
        if (resident->physicalState.has_value()) {
            RuntimePhysicalState state = *resident->physicalState;
            state.rotationQuaternion = {dynamic.rotation.x, dynamic.rotation.y,
                dynamic.rotation.z, dynamic.rotation.w};
            state.linearVelocity = {dynamic.linearVelocity.x,
                dynamic.linearVelocity.y, dynamic.linearVelocity.z};
            state.angularVelocity = {dynamic.angularVelocity.x,
                dynamic.angularVelocity.y, dynamic.angularVelocity.z};
            WorldCommand physical;
            physical.type = WorldCommandType::SetPhysicalState;
            physical.target = dynamic.object;
            physical.physicalState = std::move(state);
            (void)m_world.queue(std::move(physical));
        }
    }
}

void BethesdaSession::advanceScriptsAndApplyCommands(std::uint64_t tick, double stepSeconds, BethesdaSessionStep& result) {
    if (m_tes3.content() != nullptr) {
        Tes3VmStepResult tes3Vm = m_tes3.step(tick, 4096u);
        result.vmInstructions += tes3Vm.instructions;
        result.diagnostics.insert(result.diagnostics.end(),
            std::make_move_iterator(tes3Vm.diagnostics.begin()),
            std::make_move_iterator(tes3Vm.diagnostics.end()));
    }
    advanceScenes(stepSeconds);
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
    for (const auto& transfer : commands.itemTransfers) {
        for (const auto& [name, quest] : m_quests) {
            (void)name;
            for (const auto& alias : quest.aliases) {
                if (alias.createdObject == transfer.item) {
                    queueQuestAliasEvent(alias.handle, "OnContainerChanged",
                        {PapyrusValue::fromObject(transfer.destination), PapyrusValue::fromObject(transfer.source)});
                }
            }
        }
    }
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
    hashScalar(hash, m_sceneSpeechSequence);
    for (const auto& [scene, progress] : m_sceneProgress) {
        hashString(hash, scene.toString()); hashScalar(hash, progress.phase);
        hashScalar(hash, progress.entered); hashScalar(hash, progress.begun);
        for (const auto action : progress.completed) hashScalar(hash, action);
        for (const auto& [action, seconds] : progress.timers) { hashScalar(hash, action); hashScalar(hash, seconds); }
    }
    for (const auto& line : m_sceneSpeech) {
        hashString(hash, line.scene.toString()); hashString(hash, line.info.toString());
        hashString(hash, line.responseInfo.toString()); hashString(hash, line.speaker.toString());
        hashScalar(hash, line.action); hashScalar(hash, line.response); hashScalar(hash, line.sequence);
        hashString(hash, line.text); hashString(hash, line.voiceKey);
        hashScalar(hash, line.presented); hashScalar(hash, line.remainingSeconds);
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
        hashString(hash, animation.bodyMorph.topologyFingerprint);
        for (const auto& [name, value] : animation.bodyMorph.sliders) {
            hashString(hash, name);
            hashScalar(hash, value);
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
        hashScalar(hash, character.movement.coyoteRemaining); hashScalar(hash, character.movement.bufferRemaining);
        hashScalar(hash, character.movement.airVelocityX); hashScalar(hash, character.movement.airVelocityZ);
        hashScalar(hash, character.movement.jumpHeld); hashScalar(hash, character.movement.jumpConsumed);
        hashScalar(hash, character.supportingObject.has_value());
        if (character.supportingObject.has_value()) {
            hashString(hash, character.supportingObject->toString());
        }
    }
    for (const PhysicsRagdollSnapshot& ragdoll : ragdollSnapshots()) {
        hashString(hash, ragdoll.object.toString());
        hashScalar(hash, ragdoll.active);
        for (const PhysicsRagdollJointPose& joint : ragdoll.joints) {
            hashString(hash, joint.role);
            hashScalar(hash, joint.position.x); hashScalar(hash, joint.position.y);
            hashScalar(hash, joint.position.z);
            hashScalar(hash, joint.rotation.x); hashScalar(hash, joint.rotation.y);
            hashScalar(hash, joint.rotation.z); hashScalar(hash, joint.rotation.w);
            hashScalar(hash, joint.linearVelocity.x);
            hashScalar(hash, joint.linearVelocity.y);
            hashScalar(hash, joint.linearVelocity.z);
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
