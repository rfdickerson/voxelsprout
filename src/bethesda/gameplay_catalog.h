#pragma once

// Immutable, rebuildable gameplay metadata compiled from Bethesda records.
//
// ImportedScene deliberately remains the renderer/collision payload.  This
// sidecar contains only simulation metadata and may be invalidated whenever a
// load-order fingerprint changes without changing ImportedScene's format.

#include "bethesda/runtime_world.h"

#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace odai::bethesda {

class Tes3ContentStore;

}  // namespace odai::bethesda

namespace odai::importer::fnv {
enum class BethesdaGame : std::uint8_t;
class FalloutLoadOrder;
struct FalloutActorScan;
struct FalloutCellRecord;
struct FalloutWorldTables;
}  // namespace odai::importer::fnv

namespace odai::bethesda {

inline constexpr std::uint32_t kGameplayCellPayloadVersion = 2u;

enum class ActivityAnchorKind : std::uint8_t {
    Idle,
    Bed,
    Workplace,
    ShopCounter,
    Meal,
    Tavern,
    Patrol,
    TravelService,
    Worship,
    Training,
};

enum class ActorRole : std::uint32_t {
    None = 0u,
    Citizen = 1u << 0u,
    Merchant = 1u << 1u,
    Guard = 1u << 2u,
    GuildMember = 1u << 3u,
    Priest = 1u << 4u,
    Traveller = 1u << 5u,
};

constexpr ActorRole operator|(ActorRole left, ActorRole right) {
    return static_cast<ActorRole>(
        static_cast<std::uint32_t>(left) | static_cast<std::uint32_t>(right));
}

constexpr bool hasRole(ActorRole roles, ActorRole role) {
    return (static_cast<std::uint32_t>(roles) & static_cast<std::uint32_t>(role)) != 0u;
}

enum class BehaviorPackageSource : std::uint8_t {
    Generated,
    AuthoredTes3,
    AuthoredTes4,
    AuthoredFallout,
    AuthoredTes5,
    QuestOrScript,
};

enum class PhysicsClassification : std::uint8_t {
    Structural,
    Dynamic,
    Breakable,
    Constrained,
};

struct ActivityAnchor {
    ObjectId object;
    ActivityAnchorKind kind = ActivityAnchorKind::Idle;
    RuntimeSpaceState space;
    std::array<double, 3> position{};
    ObjectId owner;
    RecordKey faction;
    std::uint32_t capacity = 1u;
    bool reachable = true;
    std::vector<std::string> tags;
    friend bool operator==(const ActivityAnchor&, const ActivityAnchor&) = default;
};

struct BehaviorPackage {
    std::string id;
    BehaviorPackageSource source = BehaviorPackageSource::Generated;
    RuntimeActivityKind activity = RuntimeActivityKind::Idle;
    ActivityAnchorKind anchorKind = ActivityAnchorKind::Idle;
    ObjectId explicitAnchor;
    std::uint16_t startMinute = 0u;
    std::uint16_t endMinute = 0u;
    std::int32_t priority = 0;
    bool interruptible = true;
    std::string reason;
    friend bool operator==(const BehaviorPackage&, const BehaviorPackage&) = default;
};

struct ActorArchetype {
    ObjectId actor;
    RecordKey base;
    RuntimeSpaceState homeSpace;
    std::array<double, 3> authoredPosition{};
    ActorRole roles = ActorRole::Citizen;
    ObjectId owner;
    std::vector<RecordKey> factions;
    std::vector<RecordKey> services;
    std::vector<RelationshipRank> relationships;
    std::vector<BehaviorPackage> authoredPackages;
    bool questConstrained = false;
    friend bool operator==(const ActorArchetype&, const ActorArchetype&) = default;
};

struct PhysicsPolicy {
    ObjectId object;
    PhysicsClassification classification = PhysicsClassification::Structural;
    float massKilograms = 0.0f;
    bool protectedFromDestruction = false;
    bool resettable = false;
    bool owned = false;
    bool questLinked = false;
    friend bool operator==(const PhysicsPolicy&, const PhysicsPolicy&) = default;
};

struct GameplayCellPayload {
    std::uint32_t version = kGameplayCellPayloadVersion;
    std::string contentFingerprint;
    RuntimeSpaceState space;
    std::vector<ActorArchetype> actors;
    std::vector<ActivityAnchor> anchors;
    std::vector<PhysicsPolicy> physicsPolicies;
    friend bool operator==(const GameplayCellPayload&, const GameplayCellPayload&) = default;
};

// Atomic, checksummed sidecar I/O. A mismatched expected fingerprint is a
// cache miss rather than a compatibility migration: callers rebuild from the
// winning records exactly as they do for streamed cell caches.
bool saveGameplayCellPayloadAtomic(
    const std::filesystem::path& path,
    const GameplayCellPayload& payload,
    std::string& outError);

bool loadGameplayCellPayload(
    const std::filesystem::path& path,
    std::string_view expectedFingerprint,
    GameplayCellPayload& outPayload,
    std::string& outError);

// Compiles one winning TES3 cell without city-specific data. Record identity,
// services, factions, scripts, travel destinations, furniture/activator cues,
// ownership, and AI subrecords are the only inputs.
bool compileTes3GameplayCell(
    const Tes3ContentStore& content,
    const RecordKey& cell,
    std::string contentFingerprint,
    GameplayCellPayload& outPayload,
    std::string& outError);

// Streaming-facing TES3 entry point. Named exterior cells retain their CELL
// RecordKey, but are selected using the authoritative CELL DATA grid.
bool compileTes3GameplayExteriorCell(
    const Tes3ContentStore& content,
    std::int32_t gridX,
    std::int32_t gridZ,
    std::string contentFingerprint,
    GameplayCellPayload& outPayload,
    std::string& outError);

// Common TES4/Fallout/TES5 adapter. `actors` is the winning population already
// filtered for this cell by the existing actor catalog/residency path.
bool compilePostTes3GameplayCell(
    importer::fnv::BethesdaGame game,
    const importer::fnv::FalloutLoadOrder& loadOrder,
    const importer::fnv::FalloutActorScan& actors,
    const importer::fnv::FalloutCellRecord& cell,
    const importer::fnv::FalloutWorldTables& tables,
    std::string contentFingerprint,
    GameplayCellPayload& outPayload,
    std::string& outError);

struct ScheduleEntry {
    std::string packageId;
    BehaviorPackageSource source = BehaviorPackageSource::Generated;
    RuntimeActivityKind activity = RuntimeActivityKind::Idle;
    ObjectId anchor;
    std::uint16_t startMinute = 0u;
    std::uint16_t endMinute = 0u;
    std::int32_t priority = 0;
    float confidence = 0.0f;
    std::string reason;
    friend bool operator==(const ScheduleEntry&, const ScheduleEntry&) = default;
};

struct ActorSchedule {
    ObjectId actor;
    std::vector<ScheduleEntry> entries;
    std::vector<std::string> diagnostics;
    friend bool operator==(const ActorSchedule&, const ActorSchedule&) = default;
};

struct ScheduleCompileResult {
    std::map<ObjectId, ActorSchedule> actors;
    std::vector<std::string> diagnostics;
};

class SystemicScheduleCompiler {
public:
    ScheduleCompileResult compile(
        const std::vector<GameplayCellPayload>& cells) const;

    [[nodiscard]] static const ScheduleEntry* entryAt(
        const ActorSchedule& schedule, std::uint16_t minuteOfDay);
};

}  // namespace odai::bethesda
