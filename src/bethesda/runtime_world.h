#pragma once

#include "bethesda/runtime_ids.h"
#include "bethesda/runtime_render_delta.h"
#include "bethesda/runtime_transform.h"

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace odai::bethesda {

enum class RuntimeObjectKind : std::uint8_t {
    Unknown,
    Actor,
    Item,
    Container,
    Door,
    Activator,
    Projectile,
};

struct ActorValues {
    float health = 100.0f;
    float stamina = 100.0f;
    float magicka = 100.0f;
    bool dead = false;
    float maxHealth = 100.0f;
    float maxStamina = 100.0f;
    float maxMagicka = 100.0f;
    friend bool operator==(const ActorValues&, const ActorValues&) = default;
};

struct InventoryEntry {
    RecordKey item;
    std::int32_t count = 0;
    bool equipped = false;
    friend bool operator==(const InventoryEntry&, const InventoryEntry&) = default;
};

struct RelationshipRank {
    ObjectId other;
    std::int32_t rank = 0;
    friend bool operator==(const RelationshipRank&, const RelationshipRank&) = default;
};

enum class NavigationRequestStatus : std::uint8_t {
    Pending,
    Pathing,
    Moving,
    Arrived,
    Failed,
};

struct RuntimeNavigationRequest {
    ObjectId destination;
    std::uint64_t revision = 0u;
    NavigationRequestStatus status = NavigationRequestStatus::Pending;
    friend bool operator==(const RuntimeNavigationRequest&, const RuntimeNavigationRequest&) = default;
};

enum class RuntimePathStepKind : std::uint8_t {
    Walk,
    ActivateDoor,
};

struct RuntimePathStep {
    RuntimePathStepKind kind = RuntimePathStepKind::Walk;
    std::array<float, 3> position{};
    std::array<float, 3> arrivalPosition{};
    // Stable across load-order slot changes. Empty for ordinary walk steps.
    RecordKey door;
    friend bool operator==(const RuntimePathStep&, const RuntimePathStep&) = default;
};

// Deterministic package/path state owned by the Bethesda runtime rather than
// by renderer-facing actor instances. Navigation geometry itself is rebuilt
// from resident NAVM records; this is the minimal mutable cursor required to
// resume the same plan after save/load or stream eviction.
struct RuntimeAiState {
    bool walking = false;
    bool projectedToNavigation = false;
    std::array<float, 3> wanderOrigin{};
    std::array<float, 3> wanderTarget{};
    std::vector<RuntimePathStep> path;
    std::uint64_t pathIndex = 0u;
    float pauseSeconds = 0.0f;
    std::uint32_t randomState = 0u;
    bool scriptedMoveActive = false;
    bool scriptedMoveArrived = false;
    std::uint64_t scriptedMoveRevision = 0u;
    friend bool operator==(const RuntimeAiState&, const RuntimeAiState&) = default;
};

struct RuntimeCombatState {
    std::uint64_t nextMeleeAttackTick = 0u;
    std::uint64_t attacksStarted = 0u;
    std::uint64_t hitsLanded = 0u;
    ObjectId combatTarget;
    ObjectId lastTarget;
    friend bool operator==(const RuntimeCombatState&, const RuntimeCombatState&) = default;
};

struct RuntimeActivatorState {
    std::vector<std::int32_t> puzzleStates;
    std::vector<std::int32_t> puzzleSolution;
    std::int32_t puzzleStateCount = 0;
    std::uint64_t activationCount = 0u;
    bool opened = false;
    friend bool operator==(const RuntimeActivatorState&,
                           const RuntimeActivatorState&) = default;
};

enum class RuntimeSpaceKind : std::uint8_t {
    Unknown,
    Exterior,
    Interior,
};

// Static ACHR ownership is only spawn provenance. `currentSpace` is mutable
// gameplay state and is what streaming/presentation must consult after a
// package, door transition, MoveTo, or teleport crosses a cell boundary.
struct RuntimeSpaceState {
    RuntimeSpaceKind kind = RuntimeSpaceKind::Unknown;
    RecordKey cell;
    RecordKey worldspace;
    std::int32_t gridX = 0;
    std::int32_t gridZ = 0;
    friend bool operator==(const RuntimeSpaceState&,
                           const RuntimeSpaceState&) = default;
};

struct RuntimeObject {
    ObjectId id;
    RecordKey base;
    RuntimeObjectKind kind = RuntimeObjectKind::Unknown;
    RuntimeTransform transform;
    RuntimeSpaceState originSpace;
    RuntimeSpaceState currentSpace;
    bool enabled = true;
    bool persistent = false;
    bool ghost = false;
    bool interior = false;
    bool inDialogueWithPlayer = false;
    std::uint64_t packageRevision = 0u;
    RecordKey location;
    std::vector<RecordKey> referenceTypes;
    std::vector<RecordKey> factions;
    RecordKey outfit;
    std::optional<RuntimeNavigationRequest> navigationRequest;
    std::optional<RuntimeAiState> aiState;
    std::optional<RuntimeCombatState> combatState;
    std::optional<RuntimeActivatorState> activatorState;
    std::optional<ActorValues> actorValues;
    std::vector<InventoryEntry> inventory;
    std::vector<RelationshipRank> relationships;
    friend bool operator==(const RuntimeObject&, const RuntimeObject&) = default;
};

enum class ActorValue : std::uint8_t {
    Health,
    Stamina,
    Magicka,
};

enum class WorldCommandType : std::uint8_t {
    Spawn,
    Destroy,
    SetTransform,
    SetPosition,
    SetEnabled,
    AdjustActorValue,
    SetActorValue,
    AdjustActorBaseValue,
    SetActorBaseValue,
    SetDead,
    SetGhost,
    EvaluatePackage,
    SetRelationshipRank,
    AddToFaction,
    RemoveFromFaction,
    SetActorContext,
    SetOriginSpace,
    SetCurrentSpace,
    SetAiState,
    SetCombatState,
    SetActivatorState,
    RequestMoveTo,
    SetNavigationStatus,
    SetOutfit,
    AddItem,
    RemoveItem,
    SetEquipped,
    TeleportToReference,
};

// Commands are the sole cross-system mutation seam. Sequence is assigned by
// BethesdaWorld::queue and establishes a stable order even when producers are
// refactored into jobs later.
struct WorldCommand {
    std::uint64_t sequence = 0u;
    WorldCommandType type = WorldCommandType::SetEnabled;
    ObjectId target;
    RuntimeObject object;
    RuntimeTransform transform;
    bool enabled = true;
    ActorValue actorValue = ActorValue::Health;
    float actorValueDelta = 0.0f;
    float actorValueAbsolute = 0.0f;
    bool actorDead = false;
    ObjectId other;
    ObjectId destination;
    std::uint64_t navigationRevision = 0u;
    NavigationRequestStatus navigationStatus = NavigationRequestStatus::Pending;
    std::int32_t relationshipRank = 0;
    bool interior = false;
    bool inDialogueWithPlayer = false;
    RecordKey location;
    RuntimeSpaceState originSpace;
    RuntimeSpaceState currentSpace;
    RecordKey faction;
    RecordKey outfit;
    RuntimeAiState aiState;
    RuntimeCombatState combatState;
    RuntimeActivatorState activatorState;
    RecordKey item;
    std::int32_t itemCount = 0;
    bool equipped = false;
};

struct CommandApplyResult {
    std::size_t applied = 0u;
    bool residencyChanged = false;
    std::vector<std::string> diagnostics;
    RuntimeRenderDeltaBatch renderDeltas;
};

class BethesdaWorld {
public:
    [[nodiscard]] ObjectId allocateRuntimeId();
    bool addInitialObject(RuntimeObject object, std::string& outError);
    [[nodiscard]] std::uint64_t queue(WorldCommand command);
    CommandApplyResult applyQueuedCommands();

    [[nodiscard]] RuntimeObject* find(const ObjectId& id);
    [[nodiscard]] const RuntimeObject* find(const ObjectId& id) const;
    // Stable deterministic identities, cached until residency changes. Runtime
    // simulation should iterate these and resolve only the objects it needs;
    // orderedObjects() intentionally remains the owning snapshot used by saves
    // and hashing, where copying is required rather than a 60 Hz accident.
    [[nodiscard]] std::span<const ObjectId> orderedObjectIds() const;
    [[nodiscard]] std::span<const ObjectId> orderedActorIds() const;
    [[nodiscard]] std::vector<RuntimeObject> orderedObjects() const;
    [[nodiscard]] std::uint64_t deterministicHash() const;
    [[nodiscard]] std::size_t size() const { return m_objects.size(); }
    [[nodiscard]] std::uint64_t nextRuntimeId() const { return m_nextRuntimeId; }
    [[nodiscard]] std::uint64_t nextCommandSequence() const { return m_nextCommandSequence; }

    void restore(
        std::vector<RuntimeObject> objects,
        std::uint64_t nextRuntimeId,
        std::uint64_t nextCommandSequence,
        std::string& outError);
    void clear();

private:
    void invalidateOrderedIds();
    void refreshOrderedIds() const;

    std::unordered_map<ObjectId, RuntimeObject, ObjectIdHash> m_objects;
    std::vector<WorldCommand> m_commands;
    mutable std::vector<ObjectId> m_orderedObjectIds;
    mutable std::vector<ObjectId> m_orderedActorIds;
    mutable bool m_orderedIdsDirty = true;
    std::uint64_t m_nextRuntimeId = 1u;
    std::uint64_t m_nextCommandSequence = 1u;
};

}  // namespace odai::bethesda
