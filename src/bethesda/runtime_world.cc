#include "bethesda/runtime_world.h"

#include "core/hash.h"

#include <algorithm>
#include <bit>
#include <cstring>
#include <limits>

namespace odai::bethesda {
namespace {

void hashBytes(std::uint64_t& hash, const void* data, std::size_t size) {
    const auto* bytes = static_cast<const unsigned char*>(data);
    for (std::size_t index = 0u; index < size; ++index) {
        hash ^= bytes[index];
        hash *= 1099511628211ull;
    }
}

void hashString(std::uint64_t& hash, const std::string& text) {
    hashBytes(hash, text.data(), text.size());
    const unsigned char terminator = 0xffu;
    hashBytes(hash, &terminator, 1u);
}

InventoryEntry* inventoryEntry(RuntimeObject& object, const RecordKey& item) {
    const auto found = std::find_if(object.inventory.begin(), object.inventory.end(),
        [&](const InventoryEntry& entry) { return entry.item == item; });
    return found == object.inventory.end() ? nullptr : &*found;
}

}  // namespace

ObjectId BethesdaWorld::allocateRuntimeId() {
    while (m_nextRuntimeId != 0u && m_objects.contains(ObjectId::runtime(m_nextRuntimeId))) {
        ++m_nextRuntimeId;
    }
    return ObjectId::runtime(m_nextRuntimeId++);
}

bool BethesdaWorld::addInitialObject(RuntimeObject object, std::string& outError) {
    if (!object.id.valid()) {
        outError = "runtime object has no valid identity";
        return false;
    }
    if (!object.base.valid()) {
        outError = "runtime object " + object.id.toString() + " has no base record";
        return false;
    }
    if (m_objects.contains(object.id)) {
        outError = "duplicate runtime object " + object.id.toString();
        return false;
    }
    if (object.id.kind == ObjectIdKind::Spawned) {
        m_nextRuntimeId = std::max(m_nextRuntimeId, object.id.spawned + 1u);
    }
    m_objects.emplace(object.id, std::move(object));
    invalidateOrderedIds();
    outError.clear();
    return true;
}

std::uint64_t BethesdaWorld::queue(WorldCommand command) {
    command.sequence = m_nextCommandSequence++;
    m_commands.push_back(std::move(command));
    return m_commands.back().sequence;
}

CommandApplyResult BethesdaWorld::applyQueuedCommands() {
    CommandApplyResult result;
    std::stable_sort(m_commands.begin(), m_commands.end(),
        [](const WorldCommand& left, const WorldCommand& right) {
            return left.sequence < right.sequence;
        });
    for (WorldCommand& command : m_commands) {
        if (command.type == WorldCommandType::Spawn) {
            RuntimeObject object = std::move(command.object);
            if (!object.id.valid()) object.id = allocateRuntimeId();
            const ObjectId addedId = object.id;
            std::string error;
            if (addInitialObject(std::move(object), error)) {
                ++result.applied;
                const RuntimeObject* added = find(addedId);
                if (added != nullptr) {
                    result.residencyChanged = result.residencyChanged ||
                        added->currentSpace.kind != RuntimeSpaceKind::Unknown;
                    result.renderDeltas.push_back(RuntimeRenderDelta{
                        added->id, RuntimeRenderTransform | RuntimeRenderVisibility,
                        added->transform, added->enabled});
                }
            } else {
                result.diagnostics.push_back(std::move(error));
            }
            continue;
        }
        RuntimeObject* object = find(command.target);
        if (object == nullptr) {
            result.diagnostics.push_back("command target is not resident: " + command.target.toString());
            continue;
        }
        switch (command.type) {
            case WorldCommandType::Destroy:
                result.residencyChanged = result.residencyChanged ||
                    object->currentSpace.kind != RuntimeSpaceKind::Unknown;
                result.renderDeltas.push_back(RuntimeRenderDelta{
                    command.target, RuntimeRenderVisibility, {}, false});
                m_objects.erase(command.target);
                invalidateOrderedIds();
                ++result.applied;
                break;
            case WorldCommandType::SetTransform:
                object->transform = command.transform;
                result.renderDeltas.push_back(RuntimeRenderDelta{
                    object->id, RuntimeRenderTransform, object->transform, object->enabled});
                ++result.applied;
                break;
            case WorldCommandType::SetPosition:
                object->transform.position = command.transform.position;
                result.renderDeltas.push_back(RuntimeRenderDelta{
                    object->id, RuntimeRenderTransform, object->transform, object->enabled});
                ++result.applied;
                break;
            case WorldCommandType::SetEnabled:
                object->enabled = command.enabled;
                result.renderDeltas.push_back(RuntimeRenderDelta{
                    object->id, RuntimeRenderVisibility, object->transform, object->enabled});
                ++result.applied;
                break;
            case WorldCommandType::AdjustActorValue:
            case WorldCommandType::SetActorValue: {
                if (!object->actorValues.has_value()) object->actorValues.emplace();
                float* value = nullptr;
                switch (command.actorValue) {
                    case ActorValue::Health: value = &object->actorValues->health; break;
                    case ActorValue::Stamina: value = &object->actorValues->stamina; break;
                    case ActorValue::Magicka: value = &object->actorValues->magicka; break;
                }
                if (command.type == WorldCommandType::AdjustActorValue) *value += command.actorValueDelta;
                else *value = command.actorValueAbsolute;
                if (command.actorValue == ActorValue::Health && *value <= 0.0f) {
                    *value = 0.0f;
                    object->actorValues->dead = true;
                }
                ++result.applied;
                break;
            }
            case WorldCommandType::AdjustActorBaseValue:
            case WorldCommandType::SetActorBaseValue: {
                if (!object->actorValues.has_value()) object->actorValues.emplace();
                float* current = nullptr;
                float* maximum = nullptr;
                switch (command.actorValue) {
                    case ActorValue::Health:
                        current = &object->actorValues->health;
                        maximum = &object->actorValues->maxHealth;
                        break;
                    case ActorValue::Stamina:
                        current = &object->actorValues->stamina;
                        maximum = &object->actorValues->maxStamina;
                        break;
                    case ActorValue::Magicka:
                        current = &object->actorValues->magicka;
                        maximum = &object->actorValues->maxMagicka;
                        break;
                }
                if (command.type == WorldCommandType::SetActorBaseValue) {
                    *maximum = command.actorValueAbsolute;
                    *current = *maximum;
                } else {
                    *maximum += command.actorValueDelta;
                    *current += command.actorValueDelta;
                }
                if (command.actorValue != ActorValue::Stamina) *maximum = std::max(0.0f, *maximum);
                if (command.actorValue == ActorValue::Health && *current <= 0.0f) {
                    *current = 0.0f;
                    object->actorValues->dead = true;
                }
                ++result.applied;
                break;
            }
            case WorldCommandType::SetDead:
                if (!object->actorValues.has_value()) object->actorValues.emplace();
                object->actorValues->dead = command.actorDead;
                ++result.applied;
                break;
            case WorldCommandType::SetGhost:
                object->ghost = command.enabled;
                ++result.applied;
                break;
            case WorldCommandType::EvaluatePackage:
                ++object->packageRevision;
                ++result.applied;
                break;
            case WorldCommandType::SetRelationshipRank: {
                auto relationship = std::find_if(
                    object->relationships.begin(), object->relationships.end(),
                    [&](const RelationshipRank& rank) { return rank.other == command.other; });
                if (relationship == object->relationships.end()) {
                    object->relationships.push_back(
                        RelationshipRank{command.other, command.relationshipRank});
                } else {
                    relationship->rank = command.relationshipRank;
                }
                ++result.applied;
                break;
            }
            case WorldCommandType::AddToFaction:
            case WorldCommandType::RemoveFromFaction: {
                if (object->kind != RuntimeObjectKind::Actor || !command.faction.valid()) {
                    result.diagnostics.push_back(
                        "faction mutation requires an actor and stable faction: " +
                        command.target.toString());
                    break;
                }
                auto faction = std::lower_bound(
                    object->factions.begin(), object->factions.end(), command.faction);
                if (command.type == WorldCommandType::AddToFaction) {
                    if (faction == object->factions.end() || *faction != command.faction) {
                        object->factions.insert(faction, command.faction);
                    }
                } else if (faction != object->factions.end() && *faction == command.faction) {
                    object->factions.erase(faction);
                }
                ++result.applied;
                break;
            }
            case WorldCommandType::SetActorContext:
                object->interior = command.interior;
                object->inDialogueWithPlayer = command.inDialogueWithPlayer;
                object->location = command.location;
                ++result.applied;
                break;
            case WorldCommandType::SetOriginSpace:
                if (object->originSpace.kind != RuntimeSpaceKind::Unknown &&
                    object->originSpace != command.originSpace) {
                    result.diagnostics.push_back(
                        "origin space is immutable once established: " +
                        command.target.toString());
                    break;
                }
                if (command.originSpace.kind == RuntimeSpaceKind::Interior &&
                    !command.originSpace.cell.valid()) {
                    result.diagnostics.push_back(
                        "interior origin requires a stable cell: " +
                        command.target.toString());
                    break;
                }
                if (command.originSpace.kind == RuntimeSpaceKind::Exterior &&
                    !command.originSpace.worldspace.valid()) {
                    result.diagnostics.push_back(
                        "exterior origin requires a stable worldspace: " +
                        command.target.toString());
                    break;
                }
                object->originSpace = command.originSpace;
                ++result.applied;
                break;
            case WorldCommandType::SetCurrentSpace:
                if (command.currentSpace.kind == RuntimeSpaceKind::Interior &&
                    !command.currentSpace.cell.valid()) {
                    result.diagnostics.push_back(
                        "interior space mutation requires a stable cell: " +
                        command.target.toString());
                    break;
                }
                if (command.currentSpace.kind == RuntimeSpaceKind::Exterior &&
                    !command.currentSpace.worldspace.valid()) {
                    result.diagnostics.push_back(
                        "exterior space mutation requires a stable worldspace: " +
                        command.target.toString());
                    break;
                }
                result.residencyChanged = result.residencyChanged ||
                    object->currentSpace != command.currentSpace;
                object->currentSpace = command.currentSpace;
                object->interior =
                    command.currentSpace.kind == RuntimeSpaceKind::Interior;
                ++result.applied;
                break;
            case WorldCommandType::SetAiState:
                if (object->kind != RuntimeObjectKind::Actor) {
                    result.diagnostics.push_back(
                        "AI state requires a resident actor: " + command.target.toString());
                    break;
                }
                object->aiState = std::move(command.aiState);
                ++result.applied;
                break;
            case WorldCommandType::SetCombatState:
                if (object->kind != RuntimeObjectKind::Actor) {
                    result.diagnostics.push_back(
                        "combat state requires a resident actor: " + command.target.toString());
                    break;
                }
                object->combatState = std::move(command.combatState);
                ++result.applied;
                break;
            case WorldCommandType::SetActivatorState:
                if (object->kind != RuntimeObjectKind::Door &&
                    object->kind != RuntimeObjectKind::Activator) {
                    result.diagnostics.push_back(
                        "activator state requires a door or activator: " +
                        command.target.toString());
                    break;
                }
                object->activatorState = std::move(command.activatorState);
                ++result.applied;
                break;
            case WorldCommandType::RequestMoveTo: {
                if (object->kind != RuntimeObjectKind::Actor || !command.destination.valid()) {
                    result.diagnostics.push_back(
                        "movement request requires a resident actor and valid destination: " +
                        command.target.toString());
                    break;
                }
                const std::uint64_t revision = object->navigationRequest.has_value()
                    ? object->navigationRequest->revision + 1u : 1u;
                object->navigationRequest = RuntimeNavigationRequest{
                    command.destination, revision, NavigationRequestStatus::Pending};
                ++result.applied;
                break;
            }
            case WorldCommandType::TeleportToReference: {
                const RuntimeObject* destination = find(command.destination);
                if (destination == nullptr) {
                    result.diagnostics.push_back(
                        "teleport destination is not resident: " +
                        command.destination.toString());
                    break;
                }
                result.residencyChanged = result.residencyChanged ||
                    object->currentSpace != destination->currentSpace;
                object->transform.position = destination->transform.position;
                object->currentSpace = destination->currentSpace;
                object->interior = destination->interior;
                object->location = destination->location;
                object->navigationRequest.reset();
                result.renderDeltas.push_back(RuntimeRenderDelta{
                    object->id, RuntimeRenderTransform, object->transform, object->enabled});
                ++result.applied;
                break;
            }
            case WorldCommandType::SetNavigationStatus:
                if (!object->navigationRequest.has_value() ||
                    object->navigationRequest->revision != command.navigationRevision) {
                    result.diagnostics.push_back(
                        "stale navigation status for " + command.target.toString());
                    break;
                }
                object->navigationRequest->status = command.navigationStatus;
                ++result.applied;
                break;
            case WorldCommandType::SetOutfit:
                if (object->kind != RuntimeObjectKind::Actor || !command.outfit.valid()) {
                    result.diagnostics.push_back(
                        "outfit change requires a resident actor and stable outfit record: " +
                        command.target.toString());
                    break;
                }
                object->outfit = command.outfit;
                ++object->packageRevision;
                ++result.applied;
                break;
            case WorldCommandType::AddItem:
            case WorldCommandType::RemoveItem: {
                const std::int32_t signedCount = command.type == WorldCommandType::AddItem
                    ? command.itemCount : -command.itemCount;
                InventoryEntry* entry = inventoryEntry(*object, command.item);
                if (entry == nullptr && signedCount > 0) {
                    object->inventory.push_back(InventoryEntry{command.item, signedCount, false});
                    ++result.applied;
                } else if (entry != nullptr) {
                    entry->count = std::max<std::int32_t>(0, entry->count + signedCount);
                    if (entry->count == 0) {
                        object->inventory.erase(std::remove_if(
                            object->inventory.begin(), object->inventory.end(),
                            [&](const InventoryEntry& candidate) {
                                return candidate.item == command.item;
                            }), object->inventory.end());
                    }
                    ++result.applied;
                } else {
                    result.diagnostics.push_back(
                        "cannot remove absent item " + command.item.toString() + " from " +
                        command.target.toString());
                }
                break;
            }
            case WorldCommandType::SetEquipped: {
                InventoryEntry* entry = inventoryEntry(*object, command.item);
                if (entry == nullptr || entry->count <= 0) {
                    result.diagnostics.push_back(
                        "cannot equip absent item " + command.item.toString() + " on " +
                        command.target.toString());
                    break;
                }
                entry->equipped = command.equipped;
                ++result.applied;
                break;
            }
            case WorldCommandType::Spawn:
                break;
        }
    }
    m_commands.clear();
    return result;
}

RuntimeObject* BethesdaWorld::find(const ObjectId& id) {
    const auto found = m_objects.find(id);
    return found == m_objects.end() ? nullptr : &found->second;
}

const RuntimeObject* BethesdaWorld::find(const ObjectId& id) const {
    const auto found = m_objects.find(id);
    return found == m_objects.end() ? nullptr : &found->second;
}

void BethesdaWorld::invalidateOrderedIds() {
    m_orderedIdsDirty = true;
}

void BethesdaWorld::refreshOrderedIds() const {
    if (!m_orderedIdsDirty) return;
    m_orderedObjectIds.clear();
    m_orderedActorIds.clear();
    m_orderedObjectIds.reserve(m_objects.size());
    for (const auto& [id, object] : m_objects) {
        m_orderedObjectIds.push_back(id);
        if (object.kind == RuntimeObjectKind::Actor) {
            m_orderedActorIds.push_back(id);
        }
    }
    std::sort(m_orderedObjectIds.begin(), m_orderedObjectIds.end());
    std::sort(m_orderedActorIds.begin(), m_orderedActorIds.end());
    m_orderedIdsDirty = false;
}

std::span<const ObjectId> BethesdaWorld::orderedObjectIds() const {
    refreshOrderedIds();
    return m_orderedObjectIds;
}

std::span<const ObjectId> BethesdaWorld::orderedActorIds() const {
    refreshOrderedIds();
    return m_orderedActorIds;
}

std::vector<RuntimeObject> BethesdaWorld::orderedObjects() const {
    std::vector<RuntimeObject> objects;
    objects.reserve(m_objects.size());
    for (const ObjectId& id : orderedObjectIds()) {
        const auto found = m_objects.find(id);
        if (found != m_objects.end()) objects.push_back(found->second);
    }
    return objects;
}

std::uint64_t BethesdaWorld::deterministicHash() const {
    std::uint64_t hash = 1469598103934665603ull;
    for (RuntimeObject object : orderedObjects()) {
        const std::string id = object.id.toString();
        hashString(hash, id);
        hashString(hash, object.base.plugin);
        hashBytes(hash, &object.base.localFormId, sizeof(object.base.localFormId));
        hashBytes(hash, &object.kind, sizeof(object.kind));
        for (const double position : object.transform.position) {
            hashBytes(hash, &position, sizeof(position));
        }
        for (const float rotation : object.transform.rotationRadians) {
            hashBytes(hash, &rotation, sizeof(rotation));
        }
        hashBytes(hash, &object.transform.scale, sizeof(object.transform.scale));
        hashBytes(hash, &object.originSpace.kind, sizeof(object.originSpace.kind));
        hashString(hash, object.originSpace.cell.toString());
        hashString(hash, object.originSpace.worldspace.toString());
        hashBytes(hash, &object.originSpace.gridX, sizeof(object.originSpace.gridX));
        hashBytes(hash, &object.originSpace.gridZ, sizeof(object.originSpace.gridZ));
        hashBytes(hash, &object.currentSpace.kind, sizeof(object.currentSpace.kind));
        hashString(hash, object.currentSpace.cell.toString());
        hashString(hash, object.currentSpace.worldspace.toString());
        hashBytes(hash, &object.currentSpace.gridX, sizeof(object.currentSpace.gridX));
        hashBytes(hash, &object.currentSpace.gridZ, sizeof(object.currentSpace.gridZ));
        hashBytes(hash, &object.enabled, sizeof(object.enabled));
        hashBytes(hash, &object.ghost, sizeof(object.ghost));
        hashBytes(hash, &object.interior, sizeof(object.interior));
        hashBytes(hash, &object.inDialogueWithPlayer, sizeof(object.inDialogueWithPlayer));
        hashBytes(hash, &object.packageRevision, sizeof(object.packageRevision));
        hashString(hash, object.location.toString());
        for (const RecordKey& referenceType : object.referenceTypes) {
            hashString(hash, referenceType.toString());
        }
        std::sort(object.factions.begin(), object.factions.end());
        for (const RecordKey& faction : object.factions) {
            hashString(hash, faction.toString());
        }
        hashString(hash, object.outfit.toString());
        if (object.navigationRequest.has_value()) {
            hashString(hash, object.navigationRequest->destination.toString());
            hashBytes(hash, &object.navigationRequest->revision,
                sizeof(object.navigationRequest->revision));
            hashBytes(hash, &object.navigationRequest->status,
                sizeof(object.navigationRequest->status));
        }
        if (object.actorValues.has_value()) {
            hashBytes(hash, &object.actorValues->health, sizeof(object.actorValues->health));
            hashBytes(hash, &object.actorValues->stamina, sizeof(object.actorValues->stamina));
            hashBytes(hash, &object.actorValues->magicka, sizeof(object.actorValues->magicka));
            hashBytes(hash, &object.actorValues->dead, sizeof(object.actorValues->dead));
            hashBytes(hash, &object.actorValues->maxHealth, sizeof(object.actorValues->maxHealth));
            hashBytes(hash, &object.actorValues->maxStamina, sizeof(object.actorValues->maxStamina));
            hashBytes(hash, &object.actorValues->maxMagicka, sizeof(object.actorValues->maxMagicka));
        }
        const bool hasAiState = object.aiState.has_value();
        hashBytes(hash, &hasAiState, sizeof(hasAiState));
        if (object.aiState.has_value()) {
            const RuntimeAiState& ai = *object.aiState;
            hashBytes(hash, &ai.walking, sizeof(ai.walking));
            hashBytes(hash, &ai.projectedToNavigation, sizeof(ai.projectedToNavigation));
            hashBytes(hash, ai.wanderOrigin.data(), sizeof(ai.wanderOrigin));
            hashBytes(hash, ai.wanderTarget.data(), sizeof(ai.wanderTarget));
            const std::uint64_t pathSize = ai.path.size();
            hashBytes(hash, &pathSize, sizeof(pathSize));
            for (const RuntimePathStep& step : ai.path) {
                hashBytes(hash, &step.kind, sizeof(step.kind));
                hashBytes(hash, step.position.data(), sizeof(step.position));
                hashBytes(hash, step.arrivalPosition.data(), sizeof(step.arrivalPosition));
                hashString(hash, step.door.plugin);
                hashBytes(hash, &step.door.localFormId, sizeof(step.door.localFormId));
            }
            hashBytes(hash, &ai.pathIndex, sizeof(ai.pathIndex));
            hashBytes(hash, &ai.pauseSeconds, sizeof(ai.pauseSeconds));
            hashBytes(hash, &ai.randomState, sizeof(ai.randomState));
            hashBytes(hash, &ai.scriptedMoveActive, sizeof(ai.scriptedMoveActive));
            hashBytes(hash, &ai.scriptedMoveArrived, sizeof(ai.scriptedMoveArrived));
            hashBytes(hash, &ai.scriptedMoveRevision, sizeof(ai.scriptedMoveRevision));
        }
        const bool hasCombatState = object.combatState.has_value();
        hashBytes(hash, &hasCombatState, sizeof(hasCombatState));
        if (object.combatState.has_value()) {
            const RuntimeCombatState& combat = *object.combatState;
            hashBytes(hash, &combat.nextMeleeAttackTick,
                sizeof(combat.nextMeleeAttackTick));
            hashBytes(hash, &combat.attacksStarted, sizeof(combat.attacksStarted));
            hashBytes(hash, &combat.hitsLanded, sizeof(combat.hitsLanded));
            const auto hashObjectId = [&](const ObjectId& object) {
                const std::uint8_t kind = static_cast<std::uint8_t>(object.kind);
                hashBytes(hash, &kind, sizeof(kind));
                if (object.kind == ObjectIdKind::PersistentReference) {
                    hashString(hash, object.reference.plugin);
                    hashBytes(hash, &object.reference.localFormId,
                        sizeof(object.reference.localFormId));
                } else {
                    hashBytes(hash, &object.spawned, sizeof(object.spawned));
                }
            };
            hashObjectId(combat.combatTarget);
            hashObjectId(combat.lastTarget);
        }
        const bool hasActivatorState = object.activatorState.has_value();
        hashBytes(hash, &hasActivatorState, sizeof(hasActivatorState));
        if (object.activatorState.has_value()) {
            const RuntimeActivatorState& activator = *object.activatorState;
            hashBytes(hash, &activator.puzzleStateCount,
                sizeof(activator.puzzleStateCount));
            hashBytes(hash, &activator.activationCount,
                sizeof(activator.activationCount));
            hashBytes(hash, &activator.opened, sizeof(activator.opened));
            const std::uint64_t stateSize = activator.puzzleStates.size();
            hashBytes(hash, &stateSize, sizeof(stateSize));
            for (const std::int32_t state : activator.puzzleStates) {
                hashBytes(hash, &state, sizeof(state));
            }
            const std::uint64_t solutionSize = activator.puzzleSolution.size();
            hashBytes(hash, &solutionSize, sizeof(solutionSize));
            for (const std::int32_t state : activator.puzzleSolution) {
                hashBytes(hash, &state, sizeof(state));
            }
        }
        std::sort(object.inventory.begin(), object.inventory.end(),
            [](const InventoryEntry& left, const InventoryEntry& right) { return left.item < right.item; });
        for (const InventoryEntry& entry : object.inventory) {
            hashString(hash, entry.item.plugin);
            hashBytes(hash, &entry.item.localFormId, sizeof(entry.item.localFormId));
            hashBytes(hash, &entry.count, sizeof(entry.count));
            hashBytes(hash, &entry.equipped, sizeof(entry.equipped));
        }
        std::sort(object.relationships.begin(), object.relationships.end(),
            [](const RelationshipRank& left, const RelationshipRank& right) {
                return left.other < right.other;
            });
        for (const RelationshipRank& relationship : object.relationships) {
            hashString(hash, relationship.other.toString());
            hashBytes(hash, &relationship.rank, sizeof(relationship.rank));
        }
    }
    return core::mix64(hash);
}

void BethesdaWorld::restore(
    std::vector<RuntimeObject> objects,
    std::uint64_t nextRuntimeId,
    std::uint64_t nextCommandSequence,
    std::string& outError) {
    clear();
    m_nextRuntimeId = std::max<std::uint64_t>(1u, nextRuntimeId);
    m_nextCommandSequence = std::max<std::uint64_t>(1u, nextCommandSequence);
    for (RuntimeObject& object : objects) {
        if (!addInitialObject(std::move(object), outError)) {
            clear();
            return;
        }
    }
    outError.clear();
}

void BethesdaWorld::clear() {
    m_objects.clear();
    m_commands.clear();
    m_orderedObjectIds.clear();
    m_orderedActorIds.clear();
    m_orderedIdsDirty = true;
    m_nextRuntimeId = 1u;
    m_nextCommandSequence = 1u;
}

}  // namespace odai::bethesda
