#include "bethesda/gameplay_catalog.h"
#include "bethesda/record_resolver.h"
#include "bethesda/tes3_content.h"
#include "import/fnv/actor_records.h"
#include "import/fnv/cell_builder.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <system_error>

#include <nlohmann/json.hpp>

namespace odai::bethesda {
namespace {

using Json = nlohmann::json;

std::string checksum(std::string_view bytes) {
    std::uint64_t hash = 1469598103934665603ull;
    for (const unsigned char byte : bytes) {
        hash ^= byte;
        hash *= 1099511628211ull;
    }
    std::ostringstream out;
    out << std::hex << std::setfill('0') << std::setw(16) << hash;
    return out.str();
}

Json recordJson(const RecordKey& key) {
    if (!key.valid()) return nullptr;
    if (key.kind == RecordKeyKind::Tes3Named) {
        return {{"kind", "tes3_named"}, {"record_type", key.recordType},
            {"text_id", key.textId}};
    }
    return {{"kind", key.kind == RecordKeyKind::Tes3Reference
            ? "tes3_reference" : "plugin_form"},
        {"plugin", key.plugin}, {"local_id", key.localFormId}};
}

bool recordFromJson(const Json& json, RecordKey& out, std::string& error) {
    if (json.is_null()) { out = {}; return true; }
    try {
        const std::string kind = json.value("kind", std::string("plugin_form"));
        if (kind == "tes3_named") {
            out = makeTes3RecordKey(json.at("record_type").get<std::string>(),
                json.at("text_id").get<std::string>());
        } else if (kind == "tes3_reference") {
            out = makeTes3ReferenceKey(json.at("plugin").get<std::string>(),
                json.at("local_id").get<std::uint32_t>());
        } else if (kind == "plugin_form") {
            out = makeRecordKey(json.at("plugin").get<std::string>(),
                json.at("local_id").get<std::uint32_t>());
        } else {
            error = "unknown gameplay sidecar RecordKey kind";
            return false;
        }
    } catch (const std::exception& exception) {
        error = std::string("invalid gameplay sidecar RecordKey: ") + exception.what();
        return false;
    }
    if (!out.valid()) { error = "empty gameplay sidecar RecordKey"; return false; }
    return true;
}

Json objectJson(const ObjectId& object) {
    if (!object.valid()) return nullptr;
    if (object.kind == ObjectIdKind::PersistentReference) {
        return {{"kind", "reference"}, {"record", recordJson(object.reference)}};
    }
    return {{"kind", "runtime"}, {"id", object.spawned}};
}

bool objectFromJson(const Json& json, ObjectId& out, std::string& error) {
    if (json.is_null()) { out = {}; return true; }
    try {
        const std::string kind = json.at("kind").get<std::string>();
        if (kind == "reference") {
            RecordKey key;
            if (!recordFromJson(json.at("record"), key, error)) return false;
            out = ObjectId::persistent(std::move(key));
        } else if (kind == "runtime") {
            out = ObjectId::runtime(json.at("id").get<std::uint64_t>());
        } else {
            error = "unknown gameplay sidecar ObjectId kind";
            return false;
        }
    } catch (const std::exception& exception) {
        error = std::string("invalid gameplay sidecar ObjectId: ") + exception.what();
        return false;
    }
    return out.valid();
}

Json spaceJson(const RuntimeSpaceState& space) {
    return {{"kind", static_cast<std::uint8_t>(space.kind)},
        {"cell", recordJson(space.cell)}, {"worldspace", recordJson(space.worldspace)},
        {"grid_x", space.gridX}, {"grid_z", space.gridZ}};
}

bool spaceFromJson(const Json& json, RuntimeSpaceState& out, std::string& error) {
    try {
        const std::uint8_t kind = json.at("kind").get<std::uint8_t>();
        if (kind > static_cast<std::uint8_t>(RuntimeSpaceKind::Interior)) {
            error = "invalid gameplay sidecar space kind";
            return false;
        }
        out.kind = static_cast<RuntimeSpaceKind>(kind);
        if (!recordFromJson(json.at("cell"), out.cell, error) ||
            !recordFromJson(json.at("worldspace"), out.worldspace, error)) return false;
        out.gridX = json.at("grid_x").get<std::int32_t>();
        out.gridZ = json.at("grid_z").get<std::int32_t>();
    } catch (const std::exception& exception) {
        error = std::string("invalid gameplay sidecar space: ") + exception.what();
        return false;
    }
    return true;
}

Json packageJson(const BehaviorPackage& package) {
    return {{"id", package.id}, {"source", static_cast<std::uint8_t>(package.source)},
        {"activity", static_cast<std::uint8_t>(package.activity)},
        {"anchor_kind", static_cast<std::uint8_t>(package.anchorKind)},
        {"explicit_anchor", objectJson(package.explicitAnchor)},
        {"start", package.startMinute}, {"end", package.endMinute},
        {"priority", package.priority}, {"interruptible", package.interruptible},
        {"reason", package.reason}};
}

bool packageFromJson(const Json& json, BehaviorPackage& out, std::string& error) {
    try {
        out.id = json.at("id").get<std::string>();
        const auto source = json.at("source").get<std::uint8_t>();
        const auto activity = json.at("activity").get<std::uint8_t>();
        const auto anchorKind = json.at("anchor_kind").get<std::uint8_t>();
        if (source > static_cast<std::uint8_t>(BehaviorPackageSource::QuestOrScript) ||
            activity > static_cast<std::uint8_t>(RuntimeActivityKind::Quest) ||
            anchorKind > static_cast<std::uint8_t>(ActivityAnchorKind::Training)) {
            error = "invalid gameplay behavior enum";
            return false;
        }
        out.source = static_cast<BehaviorPackageSource>(source);
        out.activity = static_cast<RuntimeActivityKind>(activity);
        out.anchorKind = static_cast<ActivityAnchorKind>(anchorKind);
        if (!objectFromJson(json.at("explicit_anchor"), out.explicitAnchor, error)) return false;
        out.startMinute = json.at("start").get<std::uint16_t>();
        out.endMinute = json.at("end").get<std::uint16_t>();
        out.priority = json.at("priority").get<std::int32_t>();
        out.interruptible = json.at("interruptible").get<bool>();
        out.reason = json.at("reason").get<std::string>();
    } catch (const std::exception& exception) {
        error = std::string("invalid gameplay behavior package: ") + exception.what();
        return false;
    }
    if (out.id.empty() || out.startMinute >= 1440u || out.endMinute >= 1440u) {
        error = "gameplay behavior package has invalid id or time";
        return false;
    }
    return true;
}

Json payloadJson(const GameplayCellPayload& payload) {
    Json actors = Json::array();
    for (const ActorArchetype& actor : payload.actors) {
        Json packages = Json::array();
        for (const BehaviorPackage& package : actor.authoredPackages) {
            packages.push_back(packageJson(package));
        }
        Json factions = Json::array();
        for (const RecordKey& faction : actor.factions) factions.push_back(recordJson(faction));
        Json services = Json::array();
        for (const RecordKey& service : actor.services) services.push_back(recordJson(service));
        Json relationships = Json::array();
        for (const RelationshipRank& relationship : actor.relationships) {
            relationships.push_back({{"other", objectJson(relationship.other)},
                {"rank", relationship.rank}});
        }
        actors.push_back({{"actor", objectJson(actor.actor)}, {"base", recordJson(actor.base)},
            {"home_space", spaceJson(actor.homeSpace)}, {"position", actor.authoredPosition},
            {"roles", static_cast<std::uint32_t>(actor.roles)},
            {"owner", objectJson(actor.owner)}, {"factions", std::move(factions)},
            {"services", std::move(services)}, {"relationships", std::move(relationships)},
            {"packages", std::move(packages)}, {"quest_constrained", actor.questConstrained}});
    }
    Json anchors = Json::array();
    for (const ActivityAnchor& anchor : payload.anchors) {
        anchors.push_back({{"object", objectJson(anchor.object)},
            {"kind", static_cast<std::uint8_t>(anchor.kind)},
            {"space", spaceJson(anchor.space)}, {"position", anchor.position},
            {"owner", objectJson(anchor.owner)}, {"faction", recordJson(anchor.faction)},
            {"capacity", anchor.capacity}, {"reachable", anchor.reachable},
            {"tags", anchor.tags}});
    }
    Json policies = Json::array();
    for (const PhysicsPolicy& policy : payload.physicsPolicies) {
        policies.push_back({{"object", objectJson(policy.object)},
            {"classification", static_cast<std::uint8_t>(policy.classification)},
            {"mass", policy.massKilograms}, {"protected", policy.protectedFromDestruction},
            {"resettable", policy.resettable}, {"owned", policy.owned},
            {"quest_linked", policy.questLinked}});
    }
    return {{"version", payload.version}, {"fingerprint", payload.contentFingerprint},
        {"space", spaceJson(payload.space)}, {"actors", std::move(actors)},
        {"anchors", std::move(anchors)}, {"physics", std::move(policies)}};
}

bool payloadFromJson(const Json& json, GameplayCellPayload& out, std::string& error) {
    try {
        out.version = json.at("version").get<std::uint32_t>();
        out.contentFingerprint = json.at("fingerprint").get<std::string>();
        if (!spaceFromJson(json.at("space"), out.space, error)) return false;
        for (const Json& saved : json.at("actors")) {
            ActorArchetype actor;
            if (!objectFromJson(saved.at("actor"), actor.actor, error) ||
                !recordFromJson(saved.at("base"), actor.base, error) ||
                !spaceFromJson(saved.at("home_space"), actor.homeSpace, error) ||
                !objectFromJson(saved.at("owner"), actor.owner, error)) return false;
            actor.authoredPosition = saved.at("position").get<std::array<double, 3>>();
            actor.roles = static_cast<ActorRole>(saved.at("roles").get<std::uint32_t>());
            actor.questConstrained = saved.at("quest_constrained").get<bool>();
            for (const Json& value : saved.at("factions")) {
                RecordKey key;
                if (!recordFromJson(value, key, error)) return false;
                actor.factions.push_back(std::move(key));
            }
            for (const Json& value : saved.at("services")) {
                RecordKey key;
                if (!recordFromJson(value, key, error)) return false;
                actor.services.push_back(std::move(key));
            }
            for (const Json& value : saved.at("relationships")) {
                RelationshipRank relationship;
                if (!objectFromJson(value.at("other"), relationship.other, error)) return false;
                relationship.rank = value.at("rank").get<std::int32_t>();
                actor.relationships.push_back(std::move(relationship));
            }
            for (const Json& value : saved.at("packages")) {
                BehaviorPackage package;
                if (!packageFromJson(value, package, error)) return false;
                actor.authoredPackages.push_back(std::move(package));
            }
            out.actors.push_back(std::move(actor));
        }
        for (const Json& saved : json.at("anchors")) {
            ActivityAnchor anchor;
            if (!objectFromJson(saved.at("object"), anchor.object, error) ||
                !spaceFromJson(saved.at("space"), anchor.space, error) ||
                !objectFromJson(saved.at("owner"), anchor.owner, error) ||
                !recordFromJson(saved.at("faction"), anchor.faction, error)) return false;
            const auto kind = saved.at("kind").get<std::uint8_t>();
            if (kind > static_cast<std::uint8_t>(ActivityAnchorKind::Training)) {
                error = "invalid gameplay anchor kind"; return false;
            }
            anchor.kind = static_cast<ActivityAnchorKind>(kind);
            anchor.position = saved.at("position").get<std::array<double, 3>>();
            anchor.capacity = saved.at("capacity").get<std::uint32_t>();
            anchor.reachable = saved.at("reachable").get<bool>();
            anchor.tags = saved.at("tags").get<std::vector<std::string>>();
            if (!anchor.object.valid() || anchor.capacity == 0u) {
                error = "gameplay anchor has no identity or capacity"; return false;
            }
            out.anchors.push_back(std::move(anchor));
        }
        for (const Json& saved : json.at("physics")) {
            PhysicsPolicy policy;
            if (!objectFromJson(saved.at("object"), policy.object, error)) return false;
            const auto classification = saved.at("classification").get<std::uint8_t>();
            if (classification > static_cast<std::uint8_t>(PhysicsClassification::Constrained)) {
                error = "invalid gameplay physics classification"; return false;
            }
            policy.classification = static_cast<PhysicsClassification>(classification);
            policy.massKilograms = saved.at("mass").get<float>();
            policy.protectedFromDestruction = saved.at("protected").get<bool>();
            policy.resettable = saved.at("resettable").get<bool>();
            policy.owned = saved.at("owned").get<bool>();
            policy.questLinked = saved.at("quest_linked").get<bool>();
            out.physicsPolicies.push_back(std::move(policy));
        }
    } catch (const std::exception& exception) {
        error = std::string("invalid gameplay sidecar payload: ") + exception.what();
        return false;
    }
    if (out.version != kGameplayCellPayloadVersion || out.contentFingerprint.empty()) {
        error = "unsupported gameplay sidecar version or empty fingerprint";
        return false;
    }
    return true;
}

bool activeAt(std::uint16_t start, std::uint16_t end, std::uint16_t minute) {
    if (start == end) return true;
    if (start < end) return minute >= start && minute < end;
    return minute >= start || minute < end;
}

bool overlaps(std::uint16_t aStart, std::uint16_t aEnd,
              std::uint16_t bStart, std::uint16_t bEnd) {
    for (std::uint16_t minute = 0u; minute < 1440u; ++minute) {
        if (activeAt(aStart, aEnd, minute) && activeAt(bStart, bEnd, minute)) return true;
    }
    return false;
}

struct TemplateEntry {
    RuntimeActivityKind activity;
    ActivityAnchorKind anchorKind;
    std::uint16_t start;
    std::uint16_t end;
    const char* reason;
};

std::vector<TemplateEntry> templateFor(ActorRole roles) {
    if (hasRole(roles, ActorRole::Guard)) {
        return {{RuntimeActivityKind::Patrol, ActivityAnchorKind::Patrol, 0u, 360u, "guard night watch"},
            {RuntimeActivityKind::Eat, ActivityAnchorKind::Meal, 360u, 420u, "guard meal break"},
            {RuntimeActivityKind::Patrol, ActivityAnchorKind::Patrol, 420u, 840u, "guard day watch"},
            {RuntimeActivityKind::Eat, ActivityAnchorKind::Meal, 840u, 900u, "guard meal break"},
            {RuntimeActivityKind::Patrol, ActivityAnchorKind::Patrol, 900u, 1320u, "guard evening watch"},
            {RuntimeActivityKind::Eat, ActivityAnchorKind::Meal, 1320u, 1380u, "guard meal break"},
            {RuntimeActivityKind::Patrol, ActivityAnchorKind::Patrol, 1380u, 0u, "guard night watch"}};
    }
    ActivityAnchorKind work = ActivityAnchorKind::Workplace;
    RuntimeActivityKind workActivity = RuntimeActivityKind::Work;
    if (hasRole(roles, ActorRole::Merchant)) {
        work = ActivityAnchorKind::ShopCounter;
        workActivity = RuntimeActivityKind::Shop;
    } else if (hasRole(roles, ActorRole::Priest)) {
        work = ActivityAnchorKind::Worship;
        workActivity = RuntimeActivityKind::Worship;
    } else if (hasRole(roles, ActorRole::GuildMember)) {
        work = ActivityAnchorKind::Training;
        workActivity = RuntimeActivityKind::Train;
    }
    return {{RuntimeActivityKind::Sleep, ActivityAnchorKind::Bed, 0u, 360u, "nightly rest"},
        {RuntimeActivityKind::Eat, ActivityAnchorKind::Meal, 360u, 480u, "morning meal"},
        {workActivity, work, 480u, 1080u, "role-derived daytime work"},
        {RuntimeActivityKind::Socialize, ActivityAnchorKind::Tavern, 1080u, 1320u, "evening social time"},
        {RuntimeActivityKind::Sleep, ActivityAnchorKind::Bed, 1320u, 0u, "nightly rest"}};
}

bool sameSpace(const RuntimeSpaceState& left, const RuntimeSpaceState& right) {
    return left == right;
}

bool sharesFaction(const ActorArchetype& actor, const ActivityAnchor& anchor) {
    return anchor.faction.valid() && std::binary_search(
        actor.factions.begin(), actor.factions.end(), anchor.faction);
}

std::string lowerAscii(std::string value) {
    for (char& ch : value) {
        if (ch >= 'A' && ch <= 'Z') ch = static_cast<char>(ch - 'A' + 'a');
    }
    return value;
}

bool containsAny(std::string_view text, std::initializer_list<std::string_view> needles) {
    return std::any_of(needles.begin(), needles.end(),
        [&](std::string_view needle) { return text.find(needle) != std::string_view::npos; });
}

RuntimeSpaceState tes3Space(const Tes3ReferenceDefinition& reference) {
    RuntimeSpaceState space;
    space.cell = reference.cell;
    if (reference.interior) {
        space.kind = RuntimeSpaceKind::Interior;
        return space;
    }
    space.kind = RuntimeSpaceKind::Exterior;
    space.worldspace = makeTes3RecordKey("WRLD", "vardenfell");
    if (reference.hasCellGrid) {
        space.gridX = reference.cellGridX;
        space.gridZ = reference.cellGridZ;
    } else if (reference.cell.textId.starts_with("#")) {
        const std::string_view coordinates(reference.cell.textId);
        const std::size_t comma = coordinates.find(',');
        if (comma != std::string_view::npos) {
            (void)std::from_chars(coordinates.data() + 1u,
                coordinates.data() + comma, space.gridX);
            (void)std::from_chars(coordinates.data() + comma + 1u,
                coordinates.data() + coordinates.size(), space.gridZ);
        }
    }
    return space;
}

bool subrecordPresent(const Tes3NamedRecord* record, std::string_view type) {
    return record != nullptr && std::any_of(
        record->subrecords.begin(), record->subrecords.end(),
        [&](const Tes3SubrecordData& sub) { return sub.type == type; });
}

bool referenceSubrecordPresent(
    const Tes3ReferenceDefinition& reference, std::string_view type) {
    return std::any_of(reference.subrecords.begin(), reference.subrecords.end(),
        [&](const Tes3SubrecordData& sub) {
            return sub.type == type && !sub.data.empty();
        });
}

std::optional<ActivityAnchorKind> inferTes3Anchor(
    const Tes3ReferenceDefinition& reference) {
    const std::string base = lowerAscii(reference.baseId);
    const std::string cell = lowerAscii(reference.cell.textId);
    if (containsAny(base, {"bed", "bedroll", "hammock", "cot"})) {
        return ActivityAnchorKind::Bed;
    }
    if (containsAny(base, {"counter", "cashbox", "merchant"})) {
        return ActivityAnchorKind::ShopCounter;
    }
    if (containsAny(base, {"altar", "shrine", "temple", "saint", "tribunal"})) {
        return ActivityAnchorKind::Worship;
    }
    if (containsAny(base, {"practice", "training", "target", "dummy"})) {
        return ActivityAnchorKind::Training;
    }
    if (containsAny(base, {"siltstrider", "silt_strider", "boat", "travel"})) {
        return ActivityAnchorKind::TravelService;
    }
    if (containsAny(base, {"guard", "gate", "bridge", "tower", "watch"})) {
        return ActivityAnchorKind::Patrol;
    }
    if (containsAny(base, {"chair", "stool", "bench", "table"})) {
        if (containsAny(cell, {"tavern", "cornerclub", "club", "inn", "pub"})) {
            return ActivityAnchorKind::Tavern;
        }
        return ActivityAnchorKind::Meal;
    }
    if (reference.base.recordType == "CONT") return ActivityAnchorKind::Workplace;
    return std::nullopt;
}

PhysicsClassification inferTes3Physics(std::string_view type, std::string_view id) {
    const std::string base = lowerAscii(std::string(id));
    if (type == "DOOR") return PhysicsClassification::Constrained;
    if (containsAny(base, {"glass", "bottle", "pottery", "ceramic", "crate", "barrel",
                           "basket", "urn", "wood"})) {
        return PhysicsClassification::Breakable;
    }
    if (type == "MISC" || type == "WEAP" || type == "ARMO" || type == "BOOK" ||
        type == "INGR" || type == "ALCH" || type == "CLOT" || type == "LIGH" ||
        type == "CONT" || type == "ACTI") {
        return PhysicsClassification::Dynamic;
    }
    return PhysicsClassification::Structural;
}

std::optional<ActivityAnchorKind> inferNamedAnchor(
    std::string_view type, std::string_view id, std::string_view cellName) {
    const std::string base = lowerAscii(std::string(id));
    const std::string cell = lowerAscii(std::string(cellName));
    if (containsAny(base, {"bed", "bedroll", "hammock", "cot"})) {
        return ActivityAnchorKind::Bed;
    }
    if (containsAny(base, {"counter", "cashbox", "merchant", "vendor"})) {
        return ActivityAnchorKind::ShopCounter;
    }
    if (containsAny(base, {"altar", "shrine", "temple", "chapel", "saint"})) {
        return ActivityAnchorKind::Worship;
    }
    if (containsAny(base, {"practice", "training", "target", "dummy"})) {
        return ActivityAnchorKind::Training;
    }
    if (containsAny(base, {"caravan", "vertibird", "siltstrider", "boat", "travel"})) {
        return ActivityAnchorKind::TravelService;
    }
    if (containsAny(base, {"guard", "gate", "bridge", "tower", "watch", "patrol"})) {
        return ActivityAnchorKind::Patrol;
    }
    if (type == "FURN" || containsAny(base, {"chair", "stool", "bench", "table"})) {
        return containsAny(cell, {"tavern", "cornerclub", "club", "inn", "pub", "saloon"})
            ? ActivityAnchorKind::Tavern : ActivityAnchorKind::Meal;
    }
    if (type == "CONT") return ActivityAnchorKind::Workplace;
    return std::nullopt;
}

}  // namespace

bool saveGameplayCellPayloadAtomic(
    const std::filesystem::path& path,
    const GameplayCellPayload& payload,
    std::string& outError) {
    if (payload.version != kGameplayCellPayloadVersion ||
        payload.contentFingerprint.empty()) {
        outError = "gameplay sidecar requires current version and fingerprint";
        return false;
    }
    const Json body = payloadJson(payload);
    const std::string bodyBytes = body.dump();
    const Json root{{"format", "odai-gameplay-cell"},
        {"checksum", checksum(bodyBytes)}, {"payload", body}};
    std::error_code filesystemError;
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path(), filesystemError);
        if (filesystemError) {
            outError = "could not create gameplay sidecar directory: " +
                filesystemError.message();
            return false;
        }
    }
    const std::filesystem::path temporary = path.string() + ".tmp";
    {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output) { outError = "could not open gameplay sidecar temporary file"; return false; }
        output << root.dump();
        output.flush();
        if (!output) { outError = "could not write gameplay sidecar"; return false; }
    }
    std::filesystem::rename(temporary, path, filesystemError);
    if (filesystemError) {
        std::filesystem::remove(path, filesystemError);
        filesystemError.clear();
        std::filesystem::rename(temporary, path, filesystemError);
    }
    if (filesystemError) {
        outError = "could not commit gameplay sidecar: " + filesystemError.message();
        return false;
    }
    outError.clear();
    return true;
}

bool loadGameplayCellPayload(
    const std::filesystem::path& path,
    std::string_view expectedFingerprint,
    GameplayCellPayload& outPayload,
    std::string& outError) {
    std::ifstream input(path, std::ios::binary);
    if (!input) { outError = "gameplay sidecar does not exist"; return false; }
    Json root;
    try { input >> root; }
    catch (const std::exception& exception) {
        outError = std::string("malformed gameplay sidecar: ") + exception.what();
        return false;
    }
    if (root.value("format", std::string()) != "odai-gameplay-cell" ||
        !root.contains("payload")) {
        outError = "unsupported gameplay sidecar format";
        return false;
    }
    const std::string bodyBytes = root.at("payload").dump();
    if (root.value("checksum", std::string()) != checksum(bodyBytes)) {
        outError = "gameplay sidecar checksum mismatch";
        return false;
    }
    GameplayCellPayload loaded;
    if (!payloadFromJson(root.at("payload"), loaded, outError)) return false;
    if (!expectedFingerprint.empty() && loaded.contentFingerprint != expectedFingerprint) {
        outError = "gameplay sidecar content fingerprint mismatch";
        return false;
    }
    outPayload = std::move(loaded);
    outError.clear();
    return true;
}

bool compileTes3GameplayCell(
    const Tes3ContentStore& content,
    const RecordKey& cell,
    std::string contentFingerprint,
    GameplayCellPayload& outPayload,
    std::string& outError) {
    if (!cell.valid() || cell.kind != RecordKeyKind::Tes3Named ||
        cell.recordType != "CELL" || contentFingerprint.empty()) {
        outError = "TES3 gameplay compilation requires a cell and content fingerprint";
        return false;
    }
    GameplayCellPayload payload;
    payload.contentFingerprint = std::move(contentFingerprint);
    bool foundCell = false;
    for (const auto& [id, reference] : content.references()) {
        if (reference.cell != cell || reference.deleted || !reference.enabled ||
            !reference.hasTransform) continue;
        foundCell = true;
        if (payload.space.kind == RuntimeSpaceKind::Unknown) payload.space = tes3Space(reference);
        const std::array<double, 3> position = {reference.position[0],
            reference.position[2], -reference.position[1]};
        const auto actorDefinition = content.actors().find(reference.base);
        if (actorDefinition != content.actors().end()) {
            const Tes3ActorDefinition& definition = actorDefinition->second;
            ActorArchetype actor;
            actor.actor = id;
            actor.base = definition.record;
            actor.homeSpace = tes3Space(reference);
            actor.authoredPosition = position;
            actor.roles = ActorRole::Citizen;
            const std::string searchable = lowerAscii(definition.id + " " +
                definition.name + " " + definition.actorClass + " " +
                definition.faction.textId);
            if (definition.serviceFlags != 0u) {
                actor.roles = actor.roles | ActorRole::Merchant;
                actor.services.push_back(makeTes3RecordKey(
                    "SERV", std::to_string(definition.serviceFlags)));
            }
            if (containsAny(searchable, {"guard", "ordinator", "watchman"})) {
                actor.roles = actor.roles | ActorRole::Guard;
            }
            if (definition.faction.valid()) {
                actor.roles = actor.roles | ActorRole::GuildMember;
                actor.factions.push_back(definition.faction);
            }
            if (containsAny(searchable, {"priest", "temple", "cult", "shrine"})) {
                actor.roles = actor.roles | ActorRole::Priest;
            }
            if (!definition.travelDestinations.empty()) {
                actor.roles = actor.roles | ActorRole::Traveller;
            }
            const Tes3NamedRecord* baseRecord = nullptr;
            const auto named = content.namedRecords().find(reference.base);
            if (named != content.namedRecords().end()) baseRecord = &named->second;
            const bool scripted = definition.script.valid();
            const bool directed = subrecordPresent(baseRecord, "AI_T") ||
                subrecordPresent(baseRecord, "AI_F") ||
                subrecordPresent(baseRecord, "AI_E") ||
                subrecordPresent(baseRecord, "AI_A");
            actor.questConstrained = scripted || directed;
            if (actor.questConstrained) {
                actor.authoredPackages.push_back(BehaviorPackage{
                    "tes3:scripted", BehaviorPackageSource::QuestOrScript,
                    directed ? RuntimeActivityKind::Travel : RuntimeActivityKind::Quest,
                    ActivityAnchorKind::Idle, id, 0u, 0u, 100, false,
                    directed ? "authored directed TES3 AI package" :
                        "script-constrained TES3 actor"});
            } else if (subrecordPresent(baseRecord, "AI_W") &&
                       actor.roles == ActorRole::Citizen) {
                actor.authoredPackages.push_back(BehaviorPackage{
                    "tes3:wander", BehaviorPackageSource::AuthoredTes3,
                    RuntimeActivityKind::Idle, ActivityAnchorKind::Idle,
                    id, 0u, 0u, 10, true, "authored TES3 wander package"});
            }
            payload.actors.push_back(std::move(actor));
            payload.anchors.push_back(ActivityAnchor{id, ActivityAnchorKind::Idle,
                tes3Space(reference), position, id, definition.faction, 1u, true,
                {"actor-origin"}});
            if (!definition.travelDestinations.empty()) {
                payload.anchors.push_back(ActivityAnchor{id,
                    ActivityAnchorKind::TravelService, tes3Space(reference),
                    position, id, definition.faction, 1u, true,
                    {"tes3-travel-service"}});
            }
            continue;
        }

        if (const auto kind = inferTes3Anchor(reference); kind.has_value()) {
            payload.anchors.push_back(ActivityAnchor{id, *kind,
                tes3Space(reference), position, {}, {}, 1u, true,
                {"tes3-record-inference"}});
        }
        const PhysicsClassification classification =
            inferTes3Physics(reference.base.recordType, reference.baseId);
        if (classification != PhysicsClassification::Structural) {
            const auto baseRecord = content.namedRecords().find(reference.base);
            const bool baseScripted = baseRecord != content.namedRecords().end() &&
                subrecordPresent(&baseRecord->second, "SCRI");
            const bool referenceScripted = referenceSubrecordPresent(reference, "SCRI");
            const bool owned = referenceSubrecordPresent(reference, "ANAM");
            payload.physicsPolicies.push_back(PhysicsPolicy{id, classification,
                classification == PhysicsClassification::Constrained ? 20.0f : 2.0f,
                baseScripted || referenceScripted, true, owned,
                baseScripted || referenceScripted});
        }
    }
    if (!foundCell) {
        outError = "TES3 cell has no winning placed references: " + cell.toString();
        return false;
    }
    std::sort(payload.actors.begin(), payload.actors.end(),
        [](const auto& left, const auto& right) { return left.actor < right.actor; });
    std::sort(payload.anchors.begin(), payload.anchors.end(),
        [](const auto& left, const auto& right) {
            return std::tie(left.object, left.kind) < std::tie(right.object, right.kind);
        });
    std::sort(payload.physicsPolicies.begin(), payload.physicsPolicies.end(),
        [](const auto& left, const auto& right) { return left.object < right.object; });
    outPayload = std::move(payload);
    outError.clear();
    return true;
}

bool compileTes3GameplayExteriorCell(
    const Tes3ContentStore& content,
    std::int32_t gridX,
    std::int32_t gridZ,
    std::string contentFingerprint,
    GameplayCellPayload& outPayload,
    std::string& outError) {
    const auto reference = std::find_if(
        content.references().begin(), content.references().end(),
        [&](const auto& entry) {
            const Tes3ReferenceDefinition& definition = entry.second;
            return !definition.interior && definition.hasCellGrid &&
                definition.cellGridX == gridX && definition.cellGridZ == gridZ &&
                !definition.deleted;
        });
    if (reference == content.references().end()) {
        outError = "TES3 exterior grid has no winning placed references: " +
            std::to_string(gridX) + "," + std::to_string(gridZ);
        return false;
    }
    return compileTes3GameplayCell(content, reference->second.cell,
        std::move(contentFingerprint), outPayload, outError);
}

bool compilePostTes3GameplayCell(
    importer::fnv::BethesdaGame game,
    const importer::fnv::FalloutLoadOrder& loadOrder,
    const importer::fnv::FalloutActorScan& actors,
    const importer::fnv::FalloutCellRecord& cell,
    const importer::fnv::FalloutWorldTables& tables,
    std::string contentFingerprint,
    GameplayCellPayload& outPayload,
    std::string& outError) {
    if (game == importer::fnv::BethesdaGame::Unknown ||
        game == importer::fnv::BethesdaGame::Morrowind ||
        contentFingerprint.empty() || cell.formId == 0u) {
        outError = "post-TES3 gameplay compilation requires a later game, cell, and fingerprint";
        return false;
    }
    GameplayCellPayload payload;
    payload.contentFingerprint = std::move(contentFingerprint);
    RecordKey cellKey;
    if (!stableRecordKey(loadOrder, cell.formId, cellKey, outError)) return false;
    payload.space.kind = cell.isInterior
        ? RuntimeSpaceKind::Interior : RuntimeSpaceKind::Exterior;
    if (cell.isInterior) payload.space.cell = cellKey;
    else {
        if (!stableRecordKey(loadOrder, cell.worldspaceFormId,
                payload.space.worldspace, outError)) return false;
        payload.space.gridX = cell.gridX;
        payload.space.gridZ = cell.gridZ;
    }

    std::set<std::uint32_t> actorReferences;
    for (const importer::fnv::FalloutActorPlacement& placement : actors.placements) {
        RecordKey actorKey;
        RecordKey baseKey;
        if (!stableRecordKey(loadOrder, placement.refFormId, actorKey, outError) ||
            !stableRecordKey(loadOrder, placement.baseFormId, baseKey, outError)) return false;
        const auto base = actors.bases.find(placement.baseFormId);
        if (base == actors.bases.end()) continue;
        actorReferences.insert(placement.refFormId);
        ActorArchetype actor;
        actor.actor = ObjectId::persistent(actorKey);
        actor.base = baseKey;
        actor.homeSpace = payload.space;
        actor.authoredPosition = {placement.position[0], placement.position[2],
            -placement.position[1]};
        actor.roles = ActorRole::Citizen;
        const std::string searchable = lowerAscii(
            base->second.editorId + " " + base->second.fullName);
        if (containsAny(searchable, {"merchant", "vendor", "trader", "shopkeeper"})) {
            actor.roles = actor.roles | ActorRole::Merchant;
        }
        if (containsAny(searchable, {"guard", "soldier", "security", "watchman"})) {
            actor.roles = actor.roles | ActorRole::Guard;
        }
        if (containsAny(searchable, {"priest", "chapel", "temple", "cult"})) {
            actor.roles = actor.roles | ActorRole::Priest;
        }
        if (containsAny(searchable, {"caravan", "traveller", "traveler", "ferryman"})) {
            actor.roles = actor.roles | ActorRole::Traveller;
        }
        payload.actors.push_back(std::move(actor));
        payload.anchors.push_back(ActivityAnchor{ObjectId::persistent(actorKey),
            ActivityAnchorKind::Idle, payload.space,
            {placement.position[0], placement.position[2], -placement.position[1]},
            ObjectId::persistent(actorKey), {}, 1u, true, {"actor-origin"}});
    }

    for (const importer::fnv::FalloutPlacedReference& reference : cell.references) {
        if (reference.isDeleted || (reference.recordFlags & 0x00000800u) != 0u ||
            actorReferences.contains(reference.formId)) continue;
        RecordKey objectKey;
        if (!stableRecordKey(loadOrder, reference.formId, objectKey, outError)) return false;
        const std::string type = [&] {
            const auto found = tables.staticRecordTypes.find(reference.baseFormId);
            return found == tables.staticRecordTypes.end() ? std::string() : found->second;
        }();
        const std::string editorId = [&] {
            const auto found = tables.staticEditorIds.find(reference.baseFormId);
            return found == tables.staticEditorIds.end() ? std::string() : found->second;
        }();
        const std::array<double, 3> position = {reference.position[0],
            reference.position[2], -reference.position[1]};
        if (const auto kind = inferNamedAnchor(type, editorId, cell.editorId);
            kind.has_value()) {
            payload.anchors.push_back(ActivityAnchor{ObjectId::persistent(objectKey),
                *kind, payload.space, position, {}, {}, 1u, true,
                {"post-tes3-record-inference"}});
        }
        const PhysicsClassification classification = inferTes3Physics(type, editorId);
        if (classification != PhysicsClassification::Structural) {
            const bool questLinked = !reference.vmadBytes.empty();
            payload.physicsPolicies.push_back(PhysicsPolicy{
                ObjectId::persistent(objectKey), classification,
                classification == PhysicsClassification::Constrained ? 20.0f : 2.0f,
                questLinked, true, false, questLinked});
        }
    }
    std::sort(payload.actors.begin(), payload.actors.end(),
        [](const auto& left, const auto& right) { return left.actor < right.actor; });
    std::sort(payload.anchors.begin(), payload.anchors.end(),
        [](const auto& left, const auto& right) {
            return std::tie(left.object, left.kind) < std::tie(right.object, right.kind);
        });
    std::sort(payload.physicsPolicies.begin(), payload.physicsPolicies.end(),
        [](const auto& left, const auto& right) { return left.object < right.object; });
    outPayload = std::move(payload);
    outError.clear();
    return true;
}

ScheduleCompileResult SystemicScheduleCompiler::compile(
    const std::vector<GameplayCellPayload>& cells) const {
    ScheduleCompileResult result;
    std::vector<ActivityAnchor> anchors;
    std::vector<ActorArchetype> actors;
    for (const GameplayCellPayload& cell : cells) {
        anchors.insert(anchors.end(), cell.anchors.begin(), cell.anchors.end());
        actors.insert(actors.end(), cell.actors.begin(), cell.actors.end());
    }
    std::sort(anchors.begin(), anchors.end(), [](const auto& left, const auto& right) {
        return std::tie(left.object, left.kind) < std::tie(right.object, right.kind);
    });
    anchors.erase(std::unique(anchors.begin(), anchors.end(), [](const auto& left, const auto& right) {
        return left.object == right.object && left.kind == right.kind;
    }), anchors.end());
    std::sort(actors.begin(), actors.end(), [](const auto& left, const auto& right) {
        return left.actor < right.actor;
    });
    actors.erase(std::unique(actors.begin(), actors.end(), [](const auto& left, const auto& right) {
        return left.actor == right.actor;
    }), actors.end());

    struct Reservation { std::uint16_t start; std::uint16_t end; };
    std::map<ObjectId, std::vector<Reservation>> reservations;
    const auto chooseAnchor = [&](const ActorArchetype& actor, ActivityAnchorKind kind,
                                  std::uint16_t start, std::uint16_t end) -> ObjectId {
        const ActivityAnchor* best = nullptr;
        double bestScore = -std::numeric_limits<double>::infinity();
        for (const ActivityAnchor& anchor : anchors) {
            if (!anchor.reachable || anchor.kind != kind || !anchor.object.valid()) continue;
            std::size_t concurrent = 0u;
            const auto foundReservations = reservations.find(anchor.object);
            if (foundReservations != reservations.end()) {
                concurrent = static_cast<std::size_t>(std::count_if(
                    foundReservations->second.begin(), foundReservations->second.end(),
                    [&](const Reservation& value) {
                        return overlaps(start, end, value.start, value.end);
                    }));
            }
            if (concurrent >= anchor.capacity) continue;
            double score = sameSpace(actor.homeSpace, anchor.space) ? 500.0 : 0.0;
            if (anchor.owner == actor.actor ||
                (actor.owner.valid() && anchor.owner == actor.owner)) score += 2000.0;
            if (sharesFaction(actor, anchor)) score += 1000.0;
            const double dx = actor.authoredPosition[0] - anchor.position[0];
            const double dy = actor.authoredPosition[1] - anchor.position[1];
            const double dz = actor.authoredPosition[2] - anchor.position[2];
            score -= std::sqrt(dx * dx + dy * dy + dz * dz) * 0.001;
            if (best == nullptr || score > bestScore ||
                (score == bestScore && anchor.object < best->object)) {
                best = &anchor;
                bestScore = score;
            }
        }
        if (best == nullptr) return {};
        reservations[best->object].push_back({start, end});
        return best->object;
    };

    for (ActorArchetype& actor : actors) {
        std::sort(actor.factions.begin(), actor.factions.end());
        ActorSchedule schedule;
        schedule.actor = actor.actor;
        const std::vector<TemplateEntry> generated = templateFor(actor.roles);
        for (std::size_t index = 0u; index < generated.size(); ++index) {
            const TemplateEntry& entry = generated[index];
            const ObjectId anchor = chooseAnchor(
                actor, entry.anchorKind, entry.start, entry.end);
            ScheduleEntry scheduled;
            scheduled.packageId = "generated:" + std::to_string(index);
            scheduled.startMinute = entry.start;
            scheduled.endMinute = entry.end;
            scheduled.priority = 0;
            if (anchor.valid()) {
                scheduled.activity = entry.activity;
                scheduled.anchor = anchor;
                scheduled.confidence = 0.75f;
                scheduled.reason = entry.reason;
            } else {
                scheduled.activity = RuntimeActivityKind::Idle;
                scheduled.confidence = 0.0f;
                scheduled.reason = "safe idle: no reachable " +
                    std::to_string(static_cast<std::uint8_t>(entry.anchorKind)) + " anchor";
                schedule.diagnostics.push_back(
                    actor.actor.toString() + " " + scheduled.reason);
            }
            schedule.entries.push_back(std::move(scheduled));
        }
        for (const BehaviorPackage& package : actor.authoredPackages) {
            ObjectId anchor = package.explicitAnchor;
            if (!anchor.valid()) {
                anchor = chooseAnchor(actor, package.anchorKind,
                    package.startMinute, package.endMinute);
            }
            if (!anchor.valid()) {
                schedule.diagnostics.push_back(actor.actor.toString() +
                    " authored package " + package.id + " has no reachable anchor");
                continue;
            }
            schedule.entries.push_back(ScheduleEntry{package.id, package.source,
                package.activity, anchor, package.startMinute, package.endMinute,
                package.priority, 1.0f,
                package.reason.empty() ? "authored behavior package" : package.reason});
        }
        std::stable_sort(schedule.entries.begin(), schedule.entries.end(),
            [](const ScheduleEntry& left, const ScheduleEntry& right) {
                return std::tie(left.startMinute, left.endMinute, left.packageId) <
                    std::tie(right.startMinute, right.endMinute, right.packageId);
            });
        result.diagnostics.insert(result.diagnostics.end(),
            schedule.diagnostics.begin(), schedule.diagnostics.end());
        result.actors.emplace(schedule.actor, std::move(schedule));
    }
    return result;
}

const ScheduleEntry* SystemicScheduleCompiler::entryAt(
    const ActorSchedule& schedule, std::uint16_t minuteOfDay) {
    const ScheduleEntry* best = nullptr;
    for (const ScheduleEntry& entry : schedule.entries) {
        if (!activeAt(entry.startMinute, entry.endMinute, minuteOfDay)) continue;
        if (best == nullptr || entry.priority > best->priority ||
            (entry.priority == best->priority && entry.source > best->source) ||
            (entry.priority == best->priority && entry.source == best->source &&
             entry.packageId < best->packageId)) {
            best = &entry;
        }
    }
    return best;
}

}  // namespace odai::bethesda
