#include "bethesda/save_game.h"

#include "bethesda/bethesda_session.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <system_error>

#include <nlohmann/json.hpp>

namespace odai::bethesda {
namespace {

constexpr std::uint32_t kOdaiSaveVersion = 8u;

using Json = nlohmann::json;

std::string checksum(const std::string& bytes) {
    std::uint64_t hash = 1469598103934665603ull;
    for (const unsigned char byte : bytes) {
        hash ^= byte;
        hash *= 1099511628211ull;
    }
    std::ostringstream out;
    out << std::hex << std::setfill('0') << std::setw(16) << hash;
    return out.str();
}

Json recordKeyJson(const RecordKey& key) {
    if (key.kind == RecordKeyKind::Tes3Named) {
        return {{"kind", "tes3_named"}, {"record_type", key.recordType}, {"string_id", key.textId}};
    }
    if (key.kind == RecordKeyKind::Tes3Reference) {
        return {{"kind", "tes3_reference"}, {"plugin", key.plugin},
                {"local_form_id", key.localFormId}};
    }
    return {{"plugin", key.plugin}, {"local_form_id", key.localFormId}};
}

bool recordKeyFromJson(const Json& json, RecordKey& out, std::string& error) {
    try {
        const std::string kind = json.value("kind", std::string("plugin_form"));
        if (kind == "tes3_named") {
            out = makeTes3RecordKey(json.at("record_type").get<std::string>(),
                                    json.at("string_id").get<std::string>());
        } else if (kind == "tes3_reference") {
            out = makeTes3ReferenceKey(json.at("plugin").get<std::string>(),
                                       json.at("local_form_id").get<std::uint32_t>());
        } else if (kind == "plugin_form") {
            out = makeRecordKey(json.at("plugin").get<std::string>(),
                                json.at("local_form_id").get<std::uint32_t>());
        } else {
            error = "unknown RecordKey kind " + kind;
            return false;
        }
    } catch (const std::exception& exception) {
        error = std::string("invalid RecordKey: ") + exception.what();
        return false;
    }
    if (!out.valid()) {
        error = "invalid empty RecordKey";
        return false;
    }
    return true;
}

Json objectIdJson(const ObjectId& id) {
    if (id.kind == ObjectIdKind::PersistentReference) {
        return {{"kind", "reference"}, {"record", recordKeyJson(id.reference)}};
    }
    return {{"kind", "runtime"}, {"id", id.spawned}};
}

bool objectIdFromJson(const Json& json, ObjectId& out, std::string& error) {
    try {
        const std::string kind = json.at("kind").get<std::string>();
        if (kind == "reference") {
            RecordKey key;
            if (!recordKeyFromJson(json.at("record"), key, error)) return false;
            out = ObjectId::persistent(std::move(key));
        } else if (kind == "runtime") {
            out = ObjectId::runtime(json.at("id").get<std::uint64_t>());
        } else {
            error = "unknown ObjectId kind " + kind;
            return false;
        }
    } catch (const std::exception& exception) {
        error = std::string("invalid ObjectId: ") + exception.what();
        return false;
    }
    if (!out.valid()) {
        error = "invalid ObjectId payload";
        return false;
    }
    return true;
}

Json vectorJson(const odai::math::Vector3& value) {
    return Json::array({value.x, value.y, value.z});
}

bool vectorFromJson(const Json& json, odai::math::Vector3& out, std::string& error) {
    try {
        if (!json.is_array() || json.size() != 3u) throw std::runtime_error("expected vec3 array");
        out = {json[0].get<float>(), json[1].get<float>(), json[2].get<float>()};
        if (!std::isfinite(out.x) || !std::isfinite(out.y) || !std::isfinite(out.z)) {
            throw std::runtime_error("non-finite vec3");
        }
    } catch (const std::exception& exception) {
        error = std::string("invalid physics vector: ") + exception.what();
        return false;
    }
    return true;
}

Json quaternionJson(const odai::math::Quaternion& value) {
    return Json::array({value.x, value.y, value.z, value.w});
}

bool quaternionFromJson(
    const Json& json, odai::math::Quaternion& out, std::string& error) {
    try {
        if (!json.is_array() || json.size() != 4u) {
            throw std::runtime_error("expected quaternion array");
        }
        out = {json[0].get<float>(), json[1].get<float>(),
            json[2].get<float>(), json[3].get<float>()};
        if (!std::isfinite(out.x) || !std::isfinite(out.y) ||
            !std::isfinite(out.z) || !std::isfinite(out.w)) {
            throw std::runtime_error("non-finite quaternion");
        }
    } catch (const std::exception& exception) {
        error = std::string("invalid physics quaternion: ") + exception.what();
        return false;
    }
    return true;
}

Json graphSnapshotJson(const odai::anim::BehaviorGraphSnapshot& snapshot) {
    Json events = Json::array();
    for (const odai::anim::AnimationEvent& event : snapshot.queuedEvents) {
        events.push_back({{"name", event.name}, {"payload", event.payload}});
    }
    return {{"state", snapshot.state}, {"state_time", snapshot.stateTime},
        {"fixed_tick", snapshot.fixedTick}, {"was_grounded", snapshot.wasGrounded},
        {"queued_events", std::move(events)}};
}

bool graphSnapshotFromJson(
    const Json& json, odai::anim::BehaviorGraphSnapshot& out, std::string& error) {
    try {
        out.state = json.at("state").get<std::string>();
        out.stateTime = json.at("state_time").get<float>();
        out.fixedTick = json.at("fixed_tick").get<std::uint64_t>();
        out.wasGrounded = json.at("was_grounded").get<bool>();
        for (const Json& event : json.at("queued_events")) {
            out.queuedEvents.push_back({event.at("name").get<std::string>(),
                event.at("payload").get<std::string>()});
        }
        if (out.state.empty() || out.stateTime < 0.0f || !std::isfinite(out.stateTime)) {
            throw std::runtime_error("invalid state/time");
        }
    } catch (const std::exception& exception) {
        error = std::string("invalid behavior graph snapshot: ") + exception.what();
        return false;
    }
    return true;
}

Json valueJson(const PapyrusValue& value) {
    Json json{{"type", static_cast<std::uint8_t>(value.type)}};
    switch (value.type) {
        case PapyrusValueType::None: break;
        case PapyrusValueType::Integer: json["value"] = value.integer; break;
        case PapyrusValueType::Float: json["value"] = value.real; break;
        case PapyrusValueType::Boolean: json["value"] = value.boolean; break;
        case PapyrusValueType::String: json["value"] = value.string; break;
        case PapyrusValueType::Object: json["value"] = objectIdJson(value.object); break;
        case PapyrusValueType::Array:
            json["value"] = Json::array();
            for (const PapyrusValue& element : value.array) json["value"].push_back(valueJson(element));
            break;
    }
    return json;
}

bool valueFromJson(const Json& json, PapyrusValue& out, std::string& error, std::uint32_t depth = 0u) {
    if (depth > 32u) {
        error = "Papyrus value nesting exceeds 32 levels";
        return false;
    }
    try {
        const auto rawType = json.at("type").get<std::uint8_t>();
        if (rawType > static_cast<std::uint8_t>(PapyrusValueType::Array)) {
            error = "unknown Papyrus value type " + std::to_string(rawType);
            return false;
        }
        out = {};
        out.type = static_cast<PapyrusValueType>(rawType);
        switch (out.type) {
            case PapyrusValueType::None: break;
            case PapyrusValueType::Integer: out.integer = json.at("value").get<std::int64_t>(); break;
            case PapyrusValueType::Float: out.real = json.at("value").get<double>(); break;
            case PapyrusValueType::Boolean: out.boolean = json.at("value").get<bool>(); break;
            case PapyrusValueType::String: out.string = json.at("value").get<std::string>(); break;
            case PapyrusValueType::Object:
                if (!objectIdFromJson(json.at("value"), out.object, error)) return false;
                break;
            case PapyrusValueType::Array:
                for (const Json& element : json.at("value")) {
                    PapyrusValue decoded;
                    if (!valueFromJson(element, decoded, error, depth + 1u)) return false;
                    out.array.push_back(std::move(decoded));
                }
                break;
        }
    } catch (const std::exception& exception) {
        error = std::string("invalid Papyrus value: ") + exception.what();
        return false;
    }
    return true;
}

Json transformJson(const RuntimeTransform& transform) {
    return {{"position", transform.position}, {"rotation", transform.rotationRadians},
            {"scale", transform.scale}};
}

Json spaceJson(const RuntimeSpaceState& space) {
    Json json{{"kind", static_cast<std::uint8_t>(space.kind)},
        {"grid_x", space.gridX}, {"grid_z", space.gridZ}};
    json["cell"] = space.cell.valid() ? recordKeyJson(space.cell) : Json(nullptr);
    json["worldspace"] = space.worldspace.valid()
        ? recordKeyJson(space.worldspace) : Json(nullptr);
    return json;
}

bool spaceFromJson(const Json& json, RuntimeSpaceState& out, std::string& error) {
    try {
        const std::uint8_t kind = json.at("kind").get<std::uint8_t>();
        if (kind > static_cast<std::uint8_t>(RuntimeSpaceKind::Interior)) {
            error = "invalid runtime space kind";
            return false;
        }
        out.kind = static_cast<RuntimeSpaceKind>(kind);
        out.gridX = json.at("grid_x").get<std::int32_t>();
        out.gridZ = json.at("grid_z").get<std::int32_t>();
        if (!json.at("cell").is_null() &&
            !recordKeyFromJson(json.at("cell"), out.cell, error)) return false;
        if (!json.at("worldspace").is_null() &&
            !recordKeyFromJson(json.at("worldspace"), out.worldspace, error)) return false;
    } catch (const std::exception& exception) {
        error = std::string("invalid runtime space: ") + exception.what();
        return false;
    }
    if ((out.kind == RuntimeSpaceKind::Interior && !out.cell.valid()) ||
        (out.kind == RuntimeSpaceKind::Exterior && !out.worldspace.valid())) {
        error = "runtime space is missing its stable cell/worldspace identity";
        return false;
    }
    return true;
}

bool transformFromJson(const Json& json, RuntimeTransform& out, std::string& error) {
    try {
        out.position = json.at("position").get<std::array<double, 3>>();
        out.rotationRadians = json.at("rotation").get<std::array<float, 3>>();
        out.scale = json.at("scale").get<float>();
    } catch (const std::exception& exception) {
        error = std::string("invalid runtime transform: ") + exception.what();
        return false;
    }
    return true;
}

Json objectJson(const RuntimeObject& object) {
    Json inventory = Json::array();
    for (const InventoryEntry& entry : object.inventory) {
        inventory.push_back({{"item", recordKeyJson(entry.item)}, {"count", entry.count},
                             {"equipped", entry.equipped}});
    }
    Json json{{"id", objectIdJson(object.id)}, {"base", recordKeyJson(object.base)},
              {"kind", static_cast<std::uint8_t>(object.kind)},
              {"transform", transformJson(object.transform)}, {"enabled", object.enabled},
              {"persistent", object.persistent}, {"ghost", object.ghost},
              {"interior", object.interior},
              {"in_dialogue_with_player", object.inDialogueWithPlayer},
              {"package_revision", object.packageRevision},
              {"inventory", std::move(inventory)}};
    json["origin_space"] = spaceJson(object.originSpace);
    json["current_space"] = spaceJson(object.currentSpace);
    json["location"] = object.location.valid() ? recordKeyJson(object.location) : Json(nullptr);
    json["reference_types"] = Json::array();
    for (const RecordKey& referenceType : object.referenceTypes) {
        json["reference_types"].push_back(recordKeyJson(referenceType));
    }
    json["factions"] = Json::array();
    for (const RecordKey& faction : object.factions) {
        json["factions"].push_back(recordKeyJson(faction));
    }
    json["outfit"] = object.outfit.valid() ? recordKeyJson(object.outfit) : Json(nullptr);
    if (object.navigationRequest.has_value()) {
        json["navigation_request"] = {
            {"destination", objectIdJson(object.navigationRequest->destination)},
            {"revision", object.navigationRequest->revision},
            {"status", static_cast<std::uint8_t>(object.navigationRequest->status)}};
    } else {
        json["navigation_request"] = Json(nullptr);
    }
    if (object.aiState.has_value()) {
        Json path = Json::array();
        for (const RuntimePathStep& step : object.aiState->path) {
            Json savedStep{{"kind", static_cast<std::uint8_t>(step.kind)},
                {"position", step.position},
                {"arrival_position", step.arrivalPosition}};
            savedStep["door"] = step.door.valid()
                ? recordKeyJson(step.door) : Json(nullptr);
            path.push_back(std::move(savedStep));
        }
        json["ai_state"] = {
            {"walking", object.aiState->walking},
            {"projected_to_navigation", object.aiState->projectedToNavigation},
            {"wander_origin", object.aiState->wanderOrigin},
            {"wander_target", object.aiState->wanderTarget},
            {"path", std::move(path)},
            {"path_index", object.aiState->pathIndex},
            {"pause_seconds", object.aiState->pauseSeconds},
            {"random_state", object.aiState->randomState},
            {"scripted_move_active", object.aiState->scriptedMoveActive},
            {"scripted_move_arrived", object.aiState->scriptedMoveArrived},
            {"scripted_move_revision", object.aiState->scriptedMoveRevision}};
    } else {
        json["ai_state"] = Json(nullptr);
    }
    if (object.combatState.has_value()) {
        json["combat_state"] = {
            {"next_melee_attack_tick", object.combatState->nextMeleeAttackTick},
            {"attacks_started", object.combatState->attacksStarted},
            {"hits_landed", object.combatState->hitsLanded}};
        json["combat_state"]["combat_target"] = object.combatState->combatTarget.valid()
            ? objectIdJson(object.combatState->combatTarget) : Json(nullptr);
        json["combat_state"]["last_target"] = object.combatState->lastTarget.valid()
            ? objectIdJson(object.combatState->lastTarget) : Json(nullptr);
    } else {
        json["combat_state"] = Json(nullptr);
    }
    if (object.activatorState.has_value()) {
        json["activator_state"] = {
            {"puzzle_states", object.activatorState->puzzleStates},
            {"puzzle_solution", object.activatorState->puzzleSolution},
            {"puzzle_state_count", object.activatorState->puzzleStateCount},
            {"activation_count", object.activatorState->activationCount},
            {"opened", object.activatorState->opened}};
    } else {
        json["activator_state"] = Json(nullptr);
    }
    json["relationships"] = Json::array();
    for (const RelationshipRank& relationship : object.relationships) {
        json["relationships"].push_back(
            {{"other", objectIdJson(relationship.other)}, {"rank", relationship.rank}});
    }
    if (object.actorValues.has_value()) {
        json["actor_values"] = {{"health", object.actorValues->health},
            {"stamina", object.actorValues->stamina}, {"magicka", object.actorValues->magicka},
            {"dead", object.actorValues->dead}, {"max_health", object.actorValues->maxHealth},
            {"max_stamina", object.actorValues->maxStamina},
            {"max_magicka", object.actorValues->maxMagicka}};
    }
    return json;
}

bool objectFromJson(const Json& json, RuntimeObject& out, std::string& error) {
    try {
        if (!objectIdFromJson(json.at("id"), out.id, error) ||
            !recordKeyFromJson(json.at("base"), out.base, error) ||
            !transformFromJson(json.at("transform"), out.transform, error)) return false;
        const auto kind = json.at("kind").get<std::uint8_t>();
        if (kind > static_cast<std::uint8_t>(RuntimeObjectKind::Projectile)) {
            error = "invalid runtime object kind"; return false;
        }
        out.kind = static_cast<RuntimeObjectKind>(kind);
        out.enabled = json.at("enabled").get<bool>();
        out.persistent = json.at("persistent").get<bool>();
        out.ghost = json.at("ghost").get<bool>();
        out.interior = json.at("interior").get<bool>();
        if (json.contains("origin_space") &&
            !spaceFromJson(json.at("origin_space"), out.originSpace, error)) return false;
        if (json.contains("current_space") &&
            !spaceFromJson(json.at("current_space"), out.currentSpace, error)) return false;
        if (!json.contains("current_space")) {
            out.currentSpace.kind = out.interior
                ? RuntimeSpaceKind::Interior : RuntimeSpaceKind::Unknown;
        }
        out.inDialogueWithPlayer = json.at("in_dialogue_with_player").get<bool>();
        out.packageRevision = json.at("package_revision").get<std::uint64_t>();
        if (!json.at("location").is_null() &&
            !recordKeyFromJson(json.at("location"), out.location, error)) return false;
        if (json.contains("reference_types")) {
            for (const Json& savedReferenceType : json.at("reference_types")) {
                RecordKey referenceType;
                if (!recordKeyFromJson(savedReferenceType, referenceType, error)) return false;
                out.referenceTypes.push_back(std::move(referenceType));
            }
            std::sort(out.referenceTypes.begin(), out.referenceTypes.end());
            if (std::adjacent_find(out.referenceTypes.begin(), out.referenceTypes.end()) !=
                out.referenceTypes.end()) {
                error = "runtime object has duplicate reference types";
                return false;
            }
        }
        if (json.contains("factions")) {
            for (const Json& savedFaction : json.at("factions")) {
                RecordKey faction;
                if (!recordKeyFromJson(savedFaction, faction, error)) return false;
                out.factions.push_back(std::move(faction));
            }
            std::sort(out.factions.begin(), out.factions.end());
            if (std::adjacent_find(out.factions.begin(), out.factions.end()) !=
                out.factions.end()) {
                error = "runtime actor has duplicate faction memberships";
                return false;
            }
        }
        if (json.contains("outfit") && !json.at("outfit").is_null() &&
            !recordKeyFromJson(json.at("outfit"), out.outfit, error)) return false;
        if (json.contains("navigation_request") && !json.at("navigation_request").is_null()) {
            RuntimeNavigationRequest request;
            const Json& savedRequest = json.at("navigation_request");
            if (!objectIdFromJson(savedRequest.at("destination"), request.destination, error)) {
                return false;
            }
            request.revision = savedRequest.at("revision").get<std::uint64_t>();
            const std::uint8_t status = savedRequest.at("status").get<std::uint8_t>();
            if (status > static_cast<std::uint8_t>(NavigationRequestStatus::Failed)) {
                error = "invalid saved navigation request status";
                return false;
            }
            request.status = static_cast<NavigationRequestStatus>(status);
            out.navigationRequest = std::move(request);
        }
        if (json.contains("ai_state") && !json.at("ai_state").is_null()) {
            const Json& savedAi = json.at("ai_state");
            RuntimeAiState ai;
            ai.walking = savedAi.at("walking").get<bool>();
            ai.projectedToNavigation =
                savedAi.at("projected_to_navigation").get<bool>();
            ai.wanderOrigin = savedAi.at("wander_origin").get<std::array<float, 3>>();
            ai.wanderTarget = savedAi.at("wander_target").get<std::array<float, 3>>();
            for (const Json& savedStep : savedAi.at("path")) {
                RuntimePathStep step;
                if (savedStep.is_array()) {
                    // OdaiSaveV3 stored only walk points. Their meaning is
                    // losslessly promoted to the typed V4 route format.
                    step.position = savedStep.get<std::array<float, 3>>();
                } else {
                    const std::uint8_t kind = savedStep.at("kind").get<std::uint8_t>();
                    if (kind > static_cast<std::uint8_t>(RuntimePathStepKind::ActivateDoor)) {
                        error = "invalid saved AI path step kind";
                        return false;
                    }
                    step.kind = static_cast<RuntimePathStepKind>(kind);
                    step.position =
                        savedStep.at("position").get<std::array<float, 3>>();
                    step.arrivalPosition =
                        savedStep.at("arrival_position").get<std::array<float, 3>>();
                    if (!savedStep.at("door").is_null() &&
                        !recordKeyFromJson(savedStep.at("door"), step.door, error)) {
                        return false;
                    }
                }
                ai.path.push_back(std::move(step));
            }
            ai.pathIndex = savedAi.at("path_index").get<std::uint64_t>();
            ai.pauseSeconds = savedAi.at("pause_seconds").get<float>();
            ai.randomState = savedAi.at("random_state").get<std::uint32_t>();
            ai.scriptedMoveActive = savedAi.at("scripted_move_active").get<bool>();
            ai.scriptedMoveArrived = savedAi.at("scripted_move_arrived").get<bool>();
            ai.scriptedMoveRevision =
                savedAi.at("scripted_move_revision").get<std::uint64_t>();
            const auto finitePoint = [](const std::array<float, 3>& point) {
                return std::isfinite(point[0]) && std::isfinite(point[1]) &&
                    std::isfinite(point[2]);
            };
            const bool validSteps = std::all_of(ai.path.begin(), ai.path.end(),
                [&](const RuntimePathStep& step) {
                    return finitePoint(step.position) && finitePoint(step.arrivalPosition) &&
                        (step.kind != RuntimePathStepKind::ActivateDoor || step.door.valid());
                });
            if (!finitePoint(ai.wanderOrigin) || !finitePoint(ai.wanderTarget) ||
                !validSteps ||
                ai.pathIndex > ai.path.size() || !std::isfinite(ai.pauseSeconds) ||
                ai.pauseSeconds < 0.0f) {
                error = "invalid saved AI navigation state";
                return false;
            }
            out.aiState = std::move(ai);
        }
        if (json.contains("combat_state") && !json.at("combat_state").is_null()) {
            const Json& savedCombat = json.at("combat_state");
            RuntimeCombatState combat;
            combat.nextMeleeAttackTick =
                savedCombat.at("next_melee_attack_tick").get<std::uint64_t>();
            combat.attacksStarted =
                savedCombat.at("attacks_started").get<std::uint64_t>();
            combat.hitsLanded = savedCombat.at("hits_landed").get<std::uint64_t>();
            if (combat.hitsLanded > combat.attacksStarted) {
                error = "invalid saved combat counters";
                return false;
            }
            if (savedCombat.contains("combat_target") &&
                !savedCombat.at("combat_target").is_null() &&
                !objectIdFromJson(savedCombat.at("combat_target"),
                    combat.combatTarget, error)) {
                return false;
            }
            if (!savedCombat.at("last_target").is_null() &&
                !objectIdFromJson(savedCombat.at("last_target"),
                    combat.lastTarget, error)) {
                return false;
            }
            out.combatState = std::move(combat);
        }
        if (json.contains("activator_state") && !json.at("activator_state").is_null()) {
            const Json& savedActivator = json.at("activator_state");
            RuntimeActivatorState activator;
            activator.puzzleStates =
                savedActivator.at("puzzle_states").get<std::vector<std::int32_t>>();
            activator.puzzleSolution =
                savedActivator.at("puzzle_solution").get<std::vector<std::int32_t>>();
            activator.puzzleStateCount =
                savedActivator.at("puzzle_state_count").get<std::int32_t>();
            activator.activationCount =
                savedActivator.at("activation_count").get<std::uint64_t>();
            activator.opened = savedActivator.at("opened").get<bool>();
            const auto inRange = [&](std::int32_t state) {
                return state >= 1 && state <= activator.puzzleStateCount;
            };
            if (activator.puzzleStateCount <= 0 ||
                activator.puzzleStates.empty() ||
                activator.puzzleStates.size() != activator.puzzleSolution.size() ||
                !std::all_of(activator.puzzleStates.begin(),
                    activator.puzzleStates.end(), inRange) ||
                !std::all_of(activator.puzzleSolution.begin(),
                    activator.puzzleSolution.end(), inRange)) {
                error = "invalid saved activator puzzle state";
                return false;
            }
            out.activatorState = std::move(activator);
        }
        if (json.contains("actor_values")) {
            ActorValues values;
            values.health = json.at("actor_values").at("health").get<float>();
            values.stamina = json.at("actor_values").at("stamina").get<float>();
            values.magicka = json.at("actor_values").at("magicka").get<float>();
            values.dead = json.at("actor_values").at("dead").get<bool>();
            values.maxHealth = json.at("actor_values").value("max_health", values.health);
            values.maxStamina = json.at("actor_values").value("max_stamina", values.stamina);
            values.maxMagicka = json.at("actor_values").value("max_magicka", values.magicka);
            out.actorValues = values;
        }
        for (const Json& item : json.at("inventory")) {
            InventoryEntry entry;
            if (!recordKeyFromJson(item.at("item"), entry.item, error)) return false;
            entry.count = item.at("count").get<std::int32_t>();
            entry.equipped = item.at("equipped").get<bool>();
            if (entry.count < 0) { error = "negative saved inventory count"; return false; }
            out.inventory.push_back(std::move(entry));
        }
        for (const Json& savedRelationship : json.at("relationships")) {
            RelationshipRank relationship;
            if (!objectIdFromJson(savedRelationship.at("other"), relationship.other, error)) {
                return false;
            }
            relationship.rank = savedRelationship.at("rank").get<std::int32_t>();
            out.relationships.push_back(std::move(relationship));
        }
    } catch (const std::exception& exception) {
        error = std::string("invalid runtime object: ") + exception.what();
        return false;
    }
    return true;
}

Json frameJson(const PapyrusCallFrameSnapshot& frame) {
    Json locals = Json::object();
    for (const auto& [name, value] : frame.locals) locals[name] = valueJson(value);
    Json json{{"function", frame.function}, {"instruction", frame.instruction},
        {"return_destination", frame.returnDestination}, {"script_class", frame.scriptClass},
        {"locals", std::move(locals)}};
    json["self"] = frame.self.valid() ? objectIdJson(frame.self) : Json(nullptr);
    return json;
}

bool frameFromJson(const Json& json, PapyrusCallFrameSnapshot& out, std::string& error) {
    try {
        out.function = json.at("function").get<std::string>();
        out.instruction = json.at("instruction").get<std::size_t>();
        out.returnDestination = json.at("return_destination").get<std::string>();
        out.scriptClass = json.at("script_class").get<std::string>();
        if (!json.at("self").is_null() && !objectIdFromJson(json.at("self"), out.self, error)) {
            return false;
        }
        for (const auto& [name, value] : json.at("locals").items()) {
            PapyrusValue decoded;
            if (!valueFromJson(value, decoded, error)) return false;
            out.locals.emplace(name, std::move(decoded));
        }
    } catch (const std::exception& exception) {
        error = std::string("invalid Papyrus call frame: ") + exception.what();
        return false;
    }
    return true;
}

Json vmJson(const PapyrusVmSnapshot& vm) {
    Json globals = Json::object();
    for (const auto& [name, value] : vm.globals) globals[name] = valueJson(value);
    Json threads = Json::array();
    for (const PapyrusThreadSnapshot& thread : vm.threads) {
        Json savedThread = frameJson(thread);
        savedThread["id"] = thread.id;
        savedThread["resume_tick"] = thread.resumeTick;
        savedThread["failed"] = thread.failed;
        savedThread["call_stack"] = Json::array();
        for (const PapyrusCallFrameSnapshot& frame : thread.callStack) {
            savedThread["call_stack"].push_back(frameJson(frame));
        }
        threads.push_back(std::move(savedThread));
    }
    Json instances = Json::array();
    for (const PapyrusScriptInstanceSnapshot& instance : vm.instances) {
        Json properties = Json::object();
        for (const auto& [name, value] : instance.properties) properties[name] = valueJson(value);
        instances.push_back({{"object", objectIdJson(instance.object)},
            {"script_class", instance.scriptClass}, {"active_state", instance.activeState},
            {"properties", std::move(properties)}});
    }
    Json updates = Json::array();
    for (const PapyrusUpdateRegistrationSnapshot& update : vm.updates) {
        updates.push_back({{"object", objectIdJson(update.object)},
            {"script_class", update.scriptClass}, {"event", update.eventFunction},
            {"interval_ticks", update.intervalTicks},
            {"next_tick", update.nextTick}, {"repeating", update.repeating}});
    }
    return {{"next_thread_id", vm.nextThreadId}, {"globals", std::move(globals)},
            {"threads", std::move(threads)}, {"instances", std::move(instances)},
            {"updates", std::move(updates)}};
}

bool vmFromJson(const Json& json, PapyrusVmSnapshot& out, std::string& error) {
    try {
        out.nextThreadId = json.at("next_thread_id").get<std::uint64_t>();
        for (const auto& [name, value] : json.at("globals").items()) {
            PapyrusValue decoded;
            if (!valueFromJson(value, decoded, error)) return false;
            out.globals.emplace(name, std::move(decoded));
        }
        for (const Json& savedThread : json.at("threads")) {
            PapyrusThreadSnapshot thread;
            thread.id = savedThread.at("id").get<std::uint64_t>();
            thread.resumeTick = savedThread.at("resume_tick").get<std::uint64_t>();
            thread.failed = savedThread.at("failed").get<bool>();
            if (!frameFromJson(savedThread, thread, error)) return false;
            for (const Json& savedFrame : savedThread.at("call_stack")) {
                PapyrusCallFrameSnapshot frame;
                if (!frameFromJson(savedFrame, frame, error)) return false;
                thread.callStack.push_back(std::move(frame));
            }
            out.threads.push_back(std::move(thread));
        }
        for (const Json& savedInstance : json.at("instances")) {
            PapyrusScriptInstanceSnapshot instance;
            if (!objectIdFromJson(savedInstance.at("object"), instance.object, error)) return false;
            instance.scriptClass = savedInstance.at("script_class").get<std::string>();
            instance.activeState = savedInstance.at("active_state").get<std::string>();
            for (const auto& [name, value] : savedInstance.at("properties").items()) {
                PapyrusValue decoded;
                if (!valueFromJson(value, decoded, error)) return false;
                instance.properties.emplace(name, std::move(decoded));
            }
            out.instances.push_back(std::move(instance));
        }
        for (const Json& savedUpdate : json.at("updates")) {
            PapyrusUpdateRegistrationSnapshot update;
            if (!objectIdFromJson(savedUpdate.at("object"), update.object, error)) return false;
            update.scriptClass = savedUpdate.at("script_class").get<std::string>();
            update.eventFunction = savedUpdate.value("event", "onupdate");
            update.intervalTicks = savedUpdate.at("interval_ticks").get<std::uint64_t>();
            update.nextTick = savedUpdate.at("next_tick").get<std::uint64_t>();
            update.repeating = savedUpdate.at("repeating").get<bool>();
            out.updates.push_back(std::move(update));
        }
    } catch (const std::exception& exception) {
        error = std::string("invalid Papyrus VM state: ") + exception.what();
        return false;
    }
    return true;
}

Json tes3ValueJson(const Tes3Value& value) {
    switch (value.type) {
        case Tes3ValueType::Number: return {{"type", "number"}, {"value", value.number}};
        case Tes3ValueType::String: return {{"type", "string"}, {"value", value.string}};
        case Tes3ValueType::Object: return {{"type", "object"}, {"value", objectIdJson(value.object)}};
        case Tes3ValueType::None: break;
    }
    return {{"type", "none"}};
}

bool tes3ValueFromJson(const Json& json, Tes3Value& out, std::string& error) {
    try {
        const std::string type = json.at("type").get<std::string>();
        if (type == "none") out = {};
        else if (type == "number") {
            out = Tes3Value::fromNumber(json.at("value").get<double>());
            if (!std::isfinite(out.number)) { error = "non-finite TES3 value"; return false; }
        } else if (type == "string") out = Tes3Value::fromString(json.at("value").get<std::string>());
        else if (type == "object") {
            ObjectId object;
            if (!objectIdFromJson(json.at("value"), object, error)) return false;
            out = Tes3Value::fromObject(std::move(object));
        } else { error = "unknown TES3 value type " + type; return false; }
    } catch (const std::exception& exception) {
        error = std::string("invalid TES3 value: ") + exception.what();
        return false;
    }
    return true;
}

Json tes3RuntimeJson(const Tes3Runtime& runtime) {
    if (runtime.content() == nullptr) return Json(nullptr);
    Json quests = Json::array();
    for (const auto& [key, quest] : runtime.journal().quests()) {
        (void)key;
        Json visited = Json::array();
        for (const RecordKey& entry : quest.visitedEntries) visited.push_back(recordKeyJson(entry));
        quests.push_back({{"quest", recordKeyJson(quest.quest)}, {"id", quest.id},
            {"index", quest.currentIndex}, {"visited", std::move(visited)},
            {"classification", static_cast<std::uint8_t>(quest.classification)},
            {"has_status_flags", quest.hasStatusFlags}});
    }
    Json chronology = Json::array();
    for (const Tes3JournalVisit& visit : runtime.journal().chronology()) {
        chronology.push_back({{"sequence", visit.sequence}, {"tick", visit.tick},
            {"quest", recordKeyJson(visit.quest)}, {"info", recordKeyJson(visit.info)},
            {"index", visit.index}, {"status", static_cast<std::uint8_t>(visit.status)},
            {"source_plugin", visit.sourcePlugin}});
    }
    Json topics = Json::array();
    for (const RecordKey& topic : runtime.knownTopics()) topics.push_back(recordKeyJson(topic));
    Json globals = Json::object();
    for (const auto& [name, value] : runtime.scripts().globals()) globals[name] = tes3ValueJson(value);
    Json threads = Json::array();
    for (const auto& [id, thread] : runtime.scripts().threads()) {
        (void)id;
        Json locals = Json::object();
        for (const auto& [name, value] : thread.locals) locals[name] = tes3ValueJson(value);
        Json events = Json::object();
        for (const auto& [name, value] : thread.eventVariables) events[name] = tes3ValueJson(value);
        threads.push_back({{"id", thread.id}, {"program", thread.program},
            {"owner", thread.owner.valid() ? objectIdJson(thread.owner) : Json(nullptr)},
            {"instruction", thread.instruction}, {"locals", std::move(locals)},
            {"events", std::move(events)}, {"state", static_cast<std::uint8_t>(thread.state)},
            {"suspension_reason", thread.suspensionReason}, {"error", thread.error}});
    }
    const Tes3DialogueState& dialogue = runtime.dialogue();
    Json exhausted = Json::array();
    for (const RecordKey& info : dialogue.exhaustedInfos) exhausted.push_back(recordKeyJson(info));
    Json choices = Json::array();
    for (const Tes3DialogueChoice& choice : dialogue.choices) {
        choices.push_back({{"label", choice.label}, {"value", choice.value}});
    }
    Json actorLocals = Json::object();
    for (const auto& [name, value] : dialogue.actor.locals) actorLocals[name] = value;
    Json playerFactions = Json::object();
    for (const auto& [name, rank] : dialogue.player.factionRanks) playerFactions[name] = rank;
    Json playerFilters = Json::object();
    for (const auto& [name, value] : dialogue.player.numericFilters) playerFilters[name] = value;
    Json playerInventory = Json::array();
    for (const auto& [item, count] : dialogue.player.inventory) {
        playerInventory.push_back({{"item", recordKeyJson(item)}, {"count", count}});
    }
    Json deathCounts = Json::object();
    for (const auto& [name, count] : dialogue.player.deathCounts) deathCounts[name] = count;
    Json savedDialogue{{"active", dialogue.active}, {"choice", dialogue.choice},
        {"goodbye", dialogue.goodbye}, {"exhausted", std::move(exhausted)},
        {"choices", std::move(choices)},
        {"current_topic", dialogue.currentTopic.valid() ? recordKeyJson(dialogue.currentTopic) : Json(nullptr)},
        {"current_info", dialogue.currentInfo.valid() ? recordKeyJson(dialogue.currentInfo) : Json(nullptr)},
        {"actor", {{"object", dialogue.actor.object.valid() ? objectIdJson(dialogue.actor.object) : Json(nullptr)},
            {"id", dialogue.actor.id}, {"race", dialogue.actor.race},
            {"class", dialogue.actor.actorClass}, {"faction", dialogue.actor.faction},
            {"cell", dialogue.actor.cell}, {"rank", dialogue.actor.rank},
            {"gender", dialogue.actor.gender}, {"disposition", dialogue.actor.disposition},
            {"locals", std::move(actorLocals)}}},
        {"player", {{"object", dialogue.player.object.valid() ? objectIdJson(dialogue.player.object) : Json(nullptr)},
            {"factions", std::move(playerFactions)}, {"filters", std::move(playerFilters)},
            {"inventory", std::move(playerInventory)}, {"death_counts", std::move(deathCounts)}}}};
    Json persistentFactions = Json::object();
    for (const auto& [name, rank] : runtime.playerState().factionRanks) {
        persistentFactions[name] = rank;
    }
    Json persistentFilters = Json::object();
    for (const auto& [name, value] : runtime.playerState().numericFilters) {
        persistentFilters[name] = value;
    }
    Json persistentInventory = Json::array();
    for (const auto& [item, count] : runtime.playerState().inventory) {
        persistentInventory.push_back({{"item", recordKeyJson(item)}, {"count", count}});
    }
    Json persistentDeaths = Json::object();
    for (const auto& [name, count] : runtime.playerState().deathCounts) {
        persistentDeaths[name] = count;
    }
    Json persistentPlayer{{"object", runtime.playerState().object.valid()
            ? objectIdJson(runtime.playerState().object) : Json(nullptr)},
        {"factions", std::move(persistentFactions)},
        {"filters", std::move(persistentFilters)},
        {"inventory", std::move(persistentInventory)},
        {"death_counts", std::move(persistentDeaths)}};
    Json referenceOverrides = Json::array();
    for (const auto& [id, override] : runtime.referenceOverrides()) {
        Json locals = Json::object();
        for (const auto& [name, value] : override.locals) locals[name] = tes3ValueJson(value);
        referenceOverrides.push_back({{"object", objectIdJson(id)},
            {"enabled", override.enabled.has_value() ? Json(*override.enabled) : Json(nullptr)},
            {"deleted", override.deleted},
            {"transform", override.transform.has_value()
                ? transformJson(*override.transform) : Json(nullptr)},
            {"locals", std::move(locals)}});
    }
    Json activeSounds = Json::array();
    for (const std::string& sound : runtime.activeSounds()) activeSounds.push_back(sound);
    Json activeSpells = Json::array();
    for (const auto& [target, spells] : runtime.activeSpells()) {
        for (const Tes3ActiveSpell& spell : spells) {
            Json effects = Json::array();
            for (const Tes3ActiveSpellEffect& effect : spell.effects) {
                effects.push_back({{"effect_id", effect.effectId}, {"skill", effect.skill},
                    {"attribute", effect.attribute}, {"magnitude", effect.magnitude},
                    {"expires_tick", effect.expiresTick}});
            }
            activeSpells.push_back({{"target", objectIdJson(target)},
                {"spell", recordKeyJson(spell.spell)},
                {"caster", spell.caster.valid() ? objectIdJson(spell.caster) : Json(nullptr)},
                {"applied_tick", spell.appliedTick}, {"effects", std::move(effects)}});
        }
    }
    return {{"journal", {{"quests", std::move(quests)}, {"chronology", std::move(chronology)},
                {"next_sequence", runtime.journal().nextSequence()}}},
        {"known_topics", std::move(topics)}, {"globals", std::move(globals)},
        {"threads", std::move(threads)}, {"next_thread_id", runtime.scripts().nextThreadId()},
        {"dialogue", std::move(savedDialogue)}, {"player", std::move(persistentPlayer)},
        {"reference_overrides", std::move(referenceOverrides)},
        {"active_sounds", std::move(activeSounds)},
        {"active_spells", std::move(activeSpells)}};
}

struct Tes3SavedState {
    bool present = false;
    std::map<RecordKey, Tes3JournalQuestState> quests;
    std::vector<Tes3JournalVisit> chronology;
    std::uint64_t nextJournalSequence = 1u;
    std::set<RecordKey> knownTopics;
    std::map<std::string, Tes3Value> globals;
    std::map<std::uint64_t, Tes3ScriptThread> threads;
    std::uint64_t nextThreadId = 1u;
    Tes3DialogueState dialogue;
    Tes3DialoguePlayerState playerState;
    std::map<ObjectId, Tes3ReferenceOverride> referenceOverrides;
    std::set<std::string> activeSounds;
    std::map<ObjectId, std::vector<Tes3ActiveSpell>> activeSpells;
};

bool tes3RuntimeFromJson(const Json& json, Tes3SavedState& out, std::string& error) {
    if (json.is_null()) return true;
    out.present = true;
    try {
        for (const Json& saved : json.at("journal").at("quests")) {
            Tes3JournalQuestState quest;
            if (!recordKeyFromJson(saved.at("quest"), quest.quest, error)) return false;
            quest.id = saved.at("id").get<std::string>();
            quest.currentIndex = saved.at("index").get<std::int32_t>();
            quest.classification = static_cast<Tes3JournalQuestClassification>(
                saved.at("classification").get<std::uint8_t>());
            if (static_cast<std::uint8_t>(quest.classification) >
                static_cast<std::uint8_t>(Tes3JournalQuestClassification::Completed)) {
                error = "invalid TES3 journal classification"; return false;
            }
            quest.hasStatusFlags = saved.at("has_status_flags").get<bool>();
            for (const Json& entry : saved.at("visited")) {
                RecordKey key;
                if (!recordKeyFromJson(entry, key, error)) return false;
                quest.visitedEntries.push_back(std::move(key));
            }
            if (!out.quests.emplace(quest.quest, std::move(quest)).second) {
                error = "duplicate TES3 journal quest"; return false;
            }
        }
        std::uint64_t previous = 0u;
        for (const Json& saved : json.at("journal").at("chronology")) {
            Tes3JournalVisit visit;
            visit.sequence = saved.at("sequence").get<std::uint64_t>();
            visit.tick = saved.at("tick").get<std::uint64_t>();
            if (visit.sequence == 0u || visit.sequence <= previous ||
                !recordKeyFromJson(saved.at("quest"), visit.quest, error) ||
                !recordKeyFromJson(saved.at("info"), visit.info, error)) {
                if (error.empty()) error = "TES3 journal chronology is not strictly ordered";
                return false;
            }
            previous = visit.sequence;
            visit.index = saved.at("index").get<std::int32_t>();
            visit.status = static_cast<Tes3QuestStatus>(saved.at("status").get<std::uint8_t>());
            visit.sourcePlugin = saved.at("source_plugin").get<std::string>();
            out.chronology.push_back(std::move(visit));
        }
        out.nextJournalSequence = json.at("journal").at("next_sequence").get<std::uint64_t>();
        if (out.nextJournalSequence == 0u || (!out.chronology.empty() &&
            out.nextJournalSequence <= out.chronology.back().sequence)) {
            error = "invalid TES3 journal next sequence"; return false;
        }
        for (const Json& saved : json.at("known_topics")) {
            RecordKey topic;
            if (!recordKeyFromJson(saved, topic, error)) return false;
            if (!out.knownTopics.insert(std::move(topic)).second) {
                error = "duplicate known TES3 topic"; return false;
            }
        }
        for (const auto& [name, saved] : json.at("globals").items()) {
            Tes3Value value;
            if (!tes3ValueFromJson(saved, value, error)) return false;
            out.globals.emplace(name, std::move(value));
        }
        for (const Json& saved : json.at("threads")) {
            Tes3ScriptThread thread;
            thread.id = saved.at("id").get<std::uint64_t>();
            thread.program = saved.at("program").get<std::string>();
            thread.instruction = saved.at("instruction").get<std::size_t>();
            thread.state = static_cast<Tes3ThreadState>(saved.at("state").get<std::uint8_t>());
            thread.suspensionReason = saved.at("suspension_reason").get<std::string>();
            thread.error = saved.at("error").get<std::string>();
            if (thread.id == 0u || static_cast<std::uint8_t>(thread.state) >
                    static_cast<std::uint8_t>(Tes3ThreadState::Failed) ||
                (!saved.at("owner").is_null() &&
                 !objectIdFromJson(saved.at("owner"), thread.owner, error))) return false;
            for (const auto& [name, valueJson] : saved.at("locals").items()) {
                Tes3Value value;
                if (!tes3ValueFromJson(valueJson, value, error)) return false;
                thread.locals.emplace(name, std::move(value));
            }
            for (const auto& [name, valueJson] : saved.at("events").items()) {
                Tes3Value value;
                if (!tes3ValueFromJson(valueJson, value, error)) return false;
                thread.eventVariables.emplace(name, std::move(value));
            }
            if (!out.threads.emplace(thread.id, std::move(thread)).second) {
                error = "duplicate TES3 script thread"; return false;
            }
        }
        out.nextThreadId = json.at("next_thread_id").get<std::uint64_t>();
        const Json& dialogue = json.at("dialogue");
        out.dialogue.active = dialogue.at("active").get<bool>();
        out.dialogue.choice = dialogue.at("choice").get<std::int32_t>();
        out.dialogue.goodbye = dialogue.at("goodbye").get<bool>();
        if (!dialogue.at("current_topic").is_null() &&
            !recordKeyFromJson(dialogue.at("current_topic"), out.dialogue.currentTopic, error)) return false;
        if (!dialogue.at("current_info").is_null() &&
            !recordKeyFromJson(dialogue.at("current_info"), out.dialogue.currentInfo, error)) return false;
        for (const Json& saved : dialogue.at("exhausted")) {
            RecordKey key;
            if (!recordKeyFromJson(saved, key, error)) return false;
            out.dialogue.exhaustedInfos.insert(std::move(key));
        }
        for (const Json& saved : dialogue.at("choices")) {
            out.dialogue.choices.push_back(
                {saved.at("label").get<std::string>(), saved.at("value").get<std::int32_t>()});
        }
        const Json& actor = dialogue.at("actor");
        if (!actor.at("object").is_null() &&
            !objectIdFromJson(actor.at("object"), out.dialogue.actor.object, error)) return false;
        out.dialogue.actor.id = actor.at("id").get<std::string>();
        out.dialogue.actor.race = actor.at("race").get<std::string>();
        out.dialogue.actor.actorClass = actor.at("class").get<std::string>();
        out.dialogue.actor.faction = actor.at("faction").get<std::string>();
        out.dialogue.actor.cell = actor.at("cell").get<std::string>();
        out.dialogue.actor.rank = actor.at("rank").get<std::int8_t>();
        out.dialogue.actor.gender = actor.at("gender").get<std::int8_t>();
        out.dialogue.actor.disposition = actor.at("disposition").get<float>();
        out.dialogue.actor.locals = actor.at("locals").get<std::map<std::string, double>>();
        const Json& player = dialogue.at("player");
        if (!player.at("object").is_null() &&
            !objectIdFromJson(player.at("object"), out.dialogue.player.object, error)) return false;
        out.dialogue.player.factionRanks = player.at("factions").get<std::map<std::string, std::int8_t>>();
        out.dialogue.player.numericFilters = player.at("filters").get<std::map<std::string, double>>();
        out.dialogue.player.deathCounts = player.at("death_counts").get<std::map<std::string, std::int32_t>>();
        for (const Json& saved : player.at("inventory")) {
            RecordKey item;
            if (!recordKeyFromJson(saved.at("item"), item, error)) return false;
            out.dialogue.player.inventory.emplace(std::move(item), saved.at("count").get<std::int32_t>());
        }
        const Json& persistentPlayer = json.contains("player") ? json.at("player") : player;
        if (!persistentPlayer.at("object").is_null() &&
            !objectIdFromJson(persistentPlayer.at("object"), out.playerState.object, error)) {
            return false;
        }
        out.playerState.factionRanks = persistentPlayer.at("factions").get<
            std::map<std::string, std::int8_t>>();
        out.playerState.numericFilters = persistentPlayer.at("filters").get<
            std::map<std::string, double>>();
        out.playerState.deathCounts = persistentPlayer.at("death_counts").get<
            std::map<std::string, std::int32_t>>();
        for (const Json& saved : persistentPlayer.at("inventory")) {
            RecordKey item;
            if (!recordKeyFromJson(saved.at("item"), item, error)) return false;
            out.playerState.inventory.emplace(
                std::move(item), saved.at("count").get<std::int32_t>());
        }
        for (const Json& saved : json.value("reference_overrides", Json::array())) {
            ObjectId object;
            if (!objectIdFromJson(saved.at("object"), object, error)) return false;
            Tes3ReferenceOverride override;
            if (!saved.at("enabled").is_null()) override.enabled = saved.at("enabled").get<bool>();
            override.deleted = saved.at("deleted").get<bool>();
            if (!saved.at("transform").is_null()) {
                RuntimeTransform transform;
                if (!transformFromJson(saved.at("transform"), transform, error)) return false;
                override.transform = transform;
            }
            for (const auto& [name, valueJson] : saved.at("locals").items()) {
                Tes3Value value;
                if (!tes3ValueFromJson(valueJson, value, error)) return false;
                override.locals.emplace(name, std::move(value));
            }
            if (!out.referenceOverrides.emplace(std::move(object), std::move(override)).second) {
                error = "duplicate TES3 reference override"; return false;
            }
        }
        for (const Json& saved : json.value("active_sounds", Json::array())) {
            const std::string sound = saved.get<std::string>();
            if (sound.empty() || !out.activeSounds.insert(sound).second) {
                error = "invalid or duplicate active TES3 sound"; return false;
            }
        }
        for (const Json& saved : json.value("active_spells", Json::array())) {
            ObjectId target;
            Tes3ActiveSpell spell;
            if (!objectIdFromJson(saved.at("target"), target, error) ||
                !recordKeyFromJson(saved.at("spell"), spell.spell, error) ||
                (!saved.at("caster").is_null() &&
                 !objectIdFromJson(saved.at("caster"), spell.caster, error))) return false;
            spell.appliedTick = saved.at("applied_tick").get<std::uint64_t>();
            for (const Json& savedEffect : saved.at("effects")) {
                Tes3ActiveSpellEffect effect;
                effect.effectId = savedEffect.at("effect_id").get<std::int16_t>();
                effect.skill = savedEffect.at("skill").get<std::int8_t>();
                effect.attribute = savedEffect.at("attribute").get<std::int8_t>();
                effect.magnitude = savedEffect.at("magnitude").get<double>();
                effect.expiresTick = savedEffect.at("expires_tick").get<std::uint64_t>();
                if (effect.expiresTick <= spell.appliedTick) {
                    error = "invalid TES3 active spell expiry"; return false;
                }
                spell.effects.push_back(effect);
            }
            if (!target.valid() || !spell.spell.valid() || spell.effects.empty()) {
                error = "invalid TES3 active spell"; return false;
            }
            out.activeSpells[target].push_back(std::move(spell));
        }
    } catch (const std::exception& exception) {
        error = std::string("invalid TES3 runtime state: ") + exception.what();
        return false;
    }
    return true;
}

Json sessionPayload(const BethesdaSession& session) {
    Json objects = Json::array();
    for (const RuntimeObject& object : session.world().orderedObjects()) {
        objects.push_back(objectJson(object));
    }
    Json quests = Json::array();
    for (const auto& [key, quest] : session.quests()) {
        (void)key;
        Json objectives = Json::array();
        for (const QuestObjectiveState& objective : quest.objectives) {
            objectives.push_back({{"index", objective.index}, {"displayed", objective.displayed},
                {"completed", objective.completed}, {"failed", objective.failed}});
        }
        Json aliases = Json::array();
        for (const QuestAliasRuntimeState& alias : quest.aliases) {
            Json savedAlias{{"id", alias.id}, {"name", alias.name},
                {"location", alias.location},
                {"created_in_alias_id", alias.createdInAliasId},
                {"created_level", alias.createdLevel},
                {"created_object_materialized", alias.createdObjectMaterialized}};
            savedAlias["handle"] = objectIdJson(alias.handle);
            savedAlias["target"] = alias.target.valid() ? objectIdJson(alias.target) : Json(nullptr);
            savedAlias["created_object"] = alias.createdObject.valid()
                ? recordKeyJson(alias.createdObject) : Json(nullptr);
            aliases.push_back(std::move(savedAlias));
        }
        Json savedQuest{{"editor_id", quest.editorId}, {"stage", quest.stage},
            {"completed_stages", quest.completedStages}, {"running", quest.running},
            {"completed", quest.completed}, {"failed", quest.failed},
            {"objectives", std::move(objectives)}, {"aliases", std::move(aliases)}};
        savedQuest["record"] = quest.record.valid() ? recordKeyJson(quest.record) : Json(nullptr);
        quests.push_back(std::move(savedQuest));
    }
    Json discoveries = Json::array();
    for (const RecordKey& discovery : session.discoveries()) {
        discoveries.push_back(recordKeyJson(discovery));
    }
    Json scenes = Json::array();
    for (const auto& [scene, playing] : session.scenes()) {
        scenes.push_back({{"record", recordKeyJson(scene)}, {"playing", playing}});
    }
    Json locations = Json::array();
    for (const auto& [record, location] : session.locations()) {
        (void)record;
        Json keywords = Json::array();
        for (const RecordKey& keyword : location.keywords) {
            keywords.push_back(recordKeyJson(keyword));
        }
        Json keywordData = Json::array();
        for (const auto& [keyword, value] : location.keywordData) {
            keywordData.push_back({{"keyword", recordKeyJson(keyword)}, {"value", value}});
        }
        Json saved{{"record", recordKeyJson(location.record)},
            {"keywords", std::move(keywords)}, {"keyword_data", std::move(keywordData)},
            {"loaded", location.loaded}};
        saved["parent"] = location.parent.valid()
            ? recordKeyJson(location.parent) : Json(nullptr);
        locations.push_back(std::move(saved));
    }
    Json globalVariables = Json::array();
    for (const auto& [record, value] : session.globalVariables()) {
        globalVariables.push_back({{"record", recordKeyJson(record)}, {"value", value}});
    }
    Json storyEvents = Json::array();
    for (const StoryEventRuntimeState& event : session.storyEvents()) {
        Json arguments = Json::array();
        for (const PapyrusValue& argument : event.arguments) {
            arguments.push_back(valueJson(argument));
        }
        storyEvents.push_back({{"sequence", event.sequence},
            {"keyword", recordKeyJson(event.keyword)}, {"arguments", std::move(arguments)}});
    }
    Json giftMenus = Json::array();
    for (const GiftMenuRequestState& request : session.giftMenuRequests()) {
        Json saved{{"sequence", request.sequence},
            {"actor", objectIdJson(request.actor)},
            {"player", objectIdJson(request.player)},
            {"player_gives", request.playerGives},
            {"show_stolen_items", request.showStolenItems},
            {"use_favor_points", request.useFavorPoints}};
        saved["filter_list"] = request.filterList.valid()
            ? objectIdJson(request.filterList) : Json(nullptr);
        giftMenus.push_back(std::move(saved));
    }
    Json animations = Json::array();
    for (const AnimationActorSnapshot& animation : session.animationSnapshots()) {
        Json saved{{"object", objectIdJson(animation.object)},
            {"third_person", graphSnapshotJson(animation.thirdPerson)}};
        saved["first_person"] = animation.firstPerson.has_value()
            ? graphSnapshotJson(*animation.firstPerson) : Json(nullptr);
        animations.push_back(std::move(saved));
    }
    Json physics = Json::array();
    for (const PhysicsCharacterSnapshot& character : session.physicsSnapshots()) {
        Json saved{{"object", objectIdJson(character.object)},
            {"position", vectorJson(character.position)},
            {"rotation", quaternionJson(character.rotation)},
            {"velocity", vectorJson(character.velocity)},
            {"ground_normal", vectorJson(character.groundNormal)},
            {"grounded", character.grounded}};
        saved["supporting_object"] = character.supportingObject.has_value()
            ? objectIdJson(*character.supportingObject) : Json(nullptr);
        physics.push_back(std::move(saved));
    }
    return {
        {"game", static_cast<std::uint8_t>(session.config().game)},
        {"content_fingerprint", session.config().contentFingerprint},
        {"scenario", session.config().scenarioId},
        {"tick", session.clock().tick()},
        {"accumulator_seconds", session.clock().accumulatorSeconds()},
        {"random_state", session.randomState()},
        {"world", {{"next_runtime_id", session.world().nextRuntimeId()},
                   {"next_command_sequence", session.world().nextCommandSequence()},
                   {"objects", std::move(objects)}}},
        {"quests", std::move(quests)},
        {"statistics", session.statistics()},
        {"discoveries", std::move(discoveries)},
        {"scenes", std::move(scenes)},
        {"forced_weather", session.forcedWeather().valid()
            ? recordKeyJson(session.forcedWeather()) : Json(nullptr)},
        {"locations", std::move(locations)},
        {"global_variables", std::move(globalVariables)},
        {"story_events", std::move(storyEvents)},
        {"next_story_event_sequence", session.nextStoryEventSequence()},
        {"gift_menus", std::move(giftMenus)},
        {"next_gift_menu_sequence", session.nextGiftMenuSequence()},
        {"script_debug_logs", session.scriptDebugLogs()},
        {"animations", std::move(animations)},
        {"physics", std::move(physics)},
        {"papyrus", vmJson(session.papyrus().snapshot())},
        {"tes3", tes3RuntimeJson(session.tes3())},
    };
}

void collectRecordKeys(const RuntimeObject& object, std::set<RecordKey>& keys) {
    keys.insert(object.base);
    if (object.id.kind == ObjectIdKind::PersistentReference) keys.insert(object.id.reference);
    for (const InventoryEntry& entry : object.inventory) keys.insert(entry.item);
    if (object.location.valid()) keys.insert(object.location);
    if (object.originSpace.cell.valid()) keys.insert(object.originSpace.cell);
    if (object.originSpace.worldspace.valid()) keys.insert(object.originSpace.worldspace);
    if (object.currentSpace.cell.valid()) keys.insert(object.currentSpace.cell);
    if (object.currentSpace.worldspace.valid()) keys.insert(object.currentSpace.worldspace);
    keys.insert(object.referenceTypes.begin(), object.referenceTypes.end());
    keys.insert(object.factions.begin(), object.factions.end());
    if (object.outfit.valid()) keys.insert(object.outfit);
    if (object.navigationRequest.has_value() &&
        object.navigationRequest->destination.kind == ObjectIdKind::PersistentReference) {
        keys.insert(object.navigationRequest->destination.reference);
    }
    for (const RelationshipRank& relationship : object.relationships) {
        if (relationship.other.kind == ObjectIdKind::PersistentReference) {
            keys.insert(relationship.other.reference);
        }
    }
}

void collectRecordKeys(const PapyrusValue& value, std::set<RecordKey>& keys) {
    if (value.type == PapyrusValueType::Object &&
        value.object.kind == ObjectIdKind::PersistentReference) {
        keys.insert(value.object.reference);
    } else if (value.type == PapyrusValueType::Array) {
        for (const PapyrusValue& element : value.array) collectRecordKeys(element, keys);
    }
}

void collectRecordKeys(const PapyrusCallFrameSnapshot& frame, std::set<RecordKey>& keys) {
    if (frame.self.kind == ObjectIdKind::PersistentReference) keys.insert(frame.self.reference);
    for (const auto& [name, value] : frame.locals) {
        (void)name;
        collectRecordKeys(value, keys);
    }
}

void collectRecordKeys(const PapyrusVmSnapshot& vm, std::set<RecordKey>& keys) {
    for (const auto& [name, value] : vm.globals) {
        (void)name;
        collectRecordKeys(value, keys);
    }
    for (const PapyrusThreadSnapshot& thread : vm.threads) {
        collectRecordKeys(thread, keys);
        for (const PapyrusCallFrameSnapshot& frame : thread.callStack) collectRecordKeys(frame, keys);
    }
    for (const PapyrusScriptInstanceSnapshot& instance : vm.instances) {
        if (instance.object.kind == ObjectIdKind::PersistentReference) keys.insert(instance.object.reference);
        for (const auto& [name, value] : instance.properties) {
            (void)name;
            collectRecordKeys(value, keys);
        }
    }
    for (const PapyrusUpdateRegistrationSnapshot& update : vm.updates) {
        if (update.object.kind == ObjectIdKind::PersistentReference) {
            keys.insert(update.object.reference);
        }
    }
}

}  // namespace

bool saveOdaiGameAtomic(
    const std::filesystem::path& path,
    const BethesdaSession& session,
    std::string& outError) {
    const Json payload = sessionPayload(session);
    const std::string payloadBytes = payload.dump();
    const Json root{{"format", "odai-save"}, {"version", kOdaiSaveVersion},
                    {"checksum", checksum(payloadBytes)}, {"payload", payload}};
    std::error_code error;
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path(), error);
        if (error) { outError = "could not create save directory: " + error.message(); return false; }
    }
    const std::filesystem::path temporary = path.string() + ".tmp";
    const std::filesystem::path previous = path.string() + ".previous";
    {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output) { outError = "could not create temporary save " + temporary.string(); return false; }
        output << root.dump(2) << '\n';
        output.flush();
        if (!output) {
            outError = "failed while writing temporary save " + temporary.string();
            output.close(); std::filesystem::remove(temporary, error); return false;
        }
    }
    std::filesystem::remove(previous, error);
    error.clear();
    const bool hadDestination = std::filesystem::is_regular_file(path, error) && !error;
    if (hadDestination) {
        std::filesystem::rename(path, previous, error);
        if (error) { outError = "could not stage previous save: " + error.message(); return false; }
    }
    error.clear();
    std::filesystem::rename(temporary, path, error);
    if (error) {
        if (hadDestination) {
            std::error_code restoreError;
            std::filesystem::rename(previous, path, restoreError);
        }
        outError = "could not commit save atomically: " + error.message();
        return false;
    }
    std::filesystem::remove(previous, error);
    outError.clear();
    return true;
}

bool loadOdaiGame(
    const std::filesystem::path& path,
    BethesdaSession& session,
    const SaveLoadOptions& options,
    SaveLoadReport& outReport,
    std::string& outError) {
    outReport = {};
    Json root;
    std::uint32_t saveVersion = 0u;
    try {
        std::filesystem::path readPath = path;
        std::ifstream input(readPath, std::ios::binary);
        if (!input) {
            readPath = path.string() + ".previous";
            input = std::ifstream(readPath, std::ios::binary);
            if (!input) { outError = "could not open save " + path.string(); return false; }
            outReport.recoveredPrevious = true;
            outReport.diagnostics.push_back(
                "main save was absent; recovered the previous atomic generation");
        }
        input >> root;
        saveVersion = root.at("version").get<std::uint32_t>();
        if (root.at("format").get<std::string>() != "odai-save" ||
            saveVersion == 0u || saveVersion > kOdaiSaveVersion) {
            outError = "unsupported ODAI save format/version"; return false;
        }
    } catch (const std::exception& exception) {
        outError = std::string("malformed ODAI save: ") + exception.what(); return false;
    }
    const Json& payload = root.at("payload");
    if (root.at("checksum").get<std::string>() != checksum(payload.dump())) {
        outError = "ODAI save checksum mismatch"; return false;
    }
    const std::map<std::string, QuestRuntimeState> registeredQuests = session.quests();

    std::vector<RuntimeObject> objects;
    PapyrusVmSnapshot vm;
    std::map<std::string, QuestRuntimeState> quests;
    std::map<std::string, std::int64_t> statistics;
    std::vector<RecordKey> discoveries;
    std::map<RecordKey, bool> scenes;
    RecordKey forcedWeather;
    std::map<RecordKey, LocationRuntimeState> locations;
    std::map<RecordKey, float> globalVariables;
    std::vector<StoryEventRuntimeState> storyEvents;
    std::vector<GiftMenuRequestState> giftMenuRequests;
    std::vector<std::string> scriptDebugLogs;
    std::vector<AnimationActorSnapshot> animationSnapshots;
    std::vector<PhysicsCharacterSnapshot> physicsSnapshots;
    Tes3SavedState tes3State;
    std::uint64_t nextStoryEventSequence = 1u;
    std::uint64_t nextGiftMenuSequence = 1u;
    std::uint64_t tick = 0u;
    double accumulator = 0.0;
    std::uint32_t randomState = 1u;
    std::uint64_t nextRuntimeId = 1u;
    std::uint64_t nextCommandSequence = 1u;
    std::string savedFingerprint;
    try {
        const auto savedGame = static_cast<importer::fnv::BethesdaGame>(
            payload.at("game").get<std::uint8_t>());
        if (savedGame != session.config().game) {
            outError = "save targets a different Bethesda game"; return false;
        }
        const std::string scenario = payload.at("scenario").get<std::string>();
        if (scenario != session.config().scenarioId) {
            outError = "save scenario '" + scenario + "' does not match configured scenario '" +
                session.config().scenarioId + "'"; return false;
        }
        savedFingerprint = payload.at("content_fingerprint").get<std::string>();
        tick = payload.at("tick").get<std::uint64_t>();
        accumulator = payload.at("accumulator_seconds").get<double>();
        randomState = payload.at("random_state").get<std::uint32_t>();
        nextRuntimeId = payload.at("world").at("next_runtime_id").get<std::uint64_t>();
        nextCommandSequence = payload.at("world").at("next_command_sequence").get<std::uint64_t>();
        for (const Json& savedObject : payload.at("world").at("objects")) {
            RuntimeObject object;
            if (!objectFromJson(savedObject, object, outError)) return false;
            objects.push_back(std::move(object));
        }
        for (const Json& savedQuest : payload.at("quests")) {
            QuestRuntimeState quest;
            quest.editorId = savedQuest.at("editor_id").get<std::string>();
            if (!savedQuest.at("record").is_null() &&
                !recordKeyFromJson(savedQuest.at("record"), quest.record, outError)) return false;
            quest.stage = savedQuest.at("stage").get<std::int32_t>();
            quest.completedStages = savedQuest.at("completed_stages").get<std::vector<std::int32_t>>();
            std::sort(quest.completedStages.begin(), quest.completedStages.end());
            if (std::adjacent_find(quest.completedStages.begin(), quest.completedStages.end()) !=
                quest.completedStages.end()) {
                outError = "saved quest has duplicate completed stages";
                return false;
            }
            quest.running = savedQuest.at("running").get<bool>();
            quest.completed = savedQuest.at("completed").get<bool>();
            quest.failed = savedQuest.at("failed").get<bool>();
            for (const Json& savedObjective : savedQuest.at("objectives")) {
                QuestObjectiveState objective;
                objective.index = savedObjective.at("index").get<std::int32_t>();
                objective.displayed = savedObjective.at("displayed").get<bool>();
                objective.completed = savedObjective.at("completed").get<bool>();
                objective.failed = savedObjective.at("failed").get<bool>();
                quest.objectives.push_back(std::move(objective));
            }
            for (const Json& savedAlias : savedQuest.at("aliases")) {
                QuestAliasRuntimeState alias;
                alias.id = savedAlias.at("id").get<std::int32_t>();
                alias.name = savedAlias.at("name").get<std::string>();
                alias.location = savedAlias.at("location").get<bool>();
                // Pre-V7 exposed a load-order-adjusted implementation ID.
                // It is diagnostic-only and intentionally absent from new
                // saves; stable target/created records carry persistence.
                alias.sourceFormId = savedAlias.value("source_form_id", 0u);
                if (!objectIdFromJson(savedAlias.at("handle"), alias.handle, outError)) return false;
                if (!savedAlias.at("target").is_null() &&
                    !objectIdFromJson(savedAlias.at("target"), alias.target, outError)) return false;
                if (saveVersion >= 7u) {
                    if (!savedAlias.at("created_object").is_null() &&
                        !recordKeyFromJson(savedAlias.at("created_object"),
                            alias.createdObject, outError)) return false;
                    alias.createdInAliasId =
                        savedAlias.at("created_in_alias_id").get<std::int32_t>();
                    alias.createdLevel = savedAlias.at("created_level").get<std::int32_t>();
                    alias.createdObjectMaterialized =
                        savedAlias.at("created_object_materialized").get<bool>();
                }
                quest.aliases.push_back(std::move(alias));
            }
            std::string key = quest.editorId;
            for (char& ch : key) if (ch >= 'A' && ch <= 'Z') ch = static_cast<char>(ch - 'A' + 'a');
            quests.emplace(std::move(key), std::move(quest));
        }
        statistics = payload.at("statistics").get<std::map<std::string, std::int64_t>>();
        for (const Json& savedDiscovery : payload.at("discoveries")) {
            RecordKey discovery;
            if (!recordKeyFromJson(savedDiscovery, discovery, outError)) return false;
            discoveries.push_back(std::move(discovery));
        }
        for (const Json& savedScene : payload.at("scenes")) {
            RecordKey scene;
            if (!recordKeyFromJson(savedScene.at("record"), scene, outError)) return false;
            if (!scenes.emplace(scene, savedScene.at("playing").get<bool>()).second) {
                outError = "save contains duplicate scene state";
                return false;
            }
        }
        if (payload.contains("forced_weather") &&
            !payload.at("forced_weather").is_null() &&
            !recordKeyFromJson(payload.at("forced_weather"), forcedWeather, outError)) {
            return false;
        }
        for (const Json& savedLocation : payload.at("locations")) {
            LocationRuntimeState location;
            if (!recordKeyFromJson(savedLocation.at("record"), location.record, outError)) {
                return false;
            }
            if (!savedLocation.at("parent").is_null() &&
                !recordKeyFromJson(savedLocation.at("parent"), location.parent, outError)) {
                return false;
            }
            for (const Json& savedKeyword : savedLocation.at("keywords")) {
                RecordKey keyword;
                if (!recordKeyFromJson(savedKeyword, keyword, outError)) return false;
                location.keywords.push_back(std::move(keyword));
            }
            std::sort(location.keywords.begin(), location.keywords.end());
            if (std::adjacent_find(location.keywords.begin(), location.keywords.end()) !=
                location.keywords.end()) {
                outError = "saved location contains duplicate keywords";
                return false;
            }
            for (const Json& savedKeywordData : savedLocation.at("keyword_data")) {
                RecordKey keyword;
                if (!recordKeyFromJson(savedKeywordData.at("keyword"), keyword, outError)) {
                    return false;
                }
                const float value = savedKeywordData.at("value").get<float>();
                if (!std::isfinite(value) ||
                    !location.keywordData.emplace(std::move(keyword), value).second) {
                    outError = "saved location contains invalid/duplicate keyword data";
                    return false;
                }
            }
            location.loaded = savedLocation.at("loaded").get<bool>();
            if (!locations.emplace(location.record, std::move(location)).second) {
                outError = "save contains duplicate location state";
                return false;
            }
        }
        for (const Json& savedGlobal : payload.at("global_variables")) {
            RecordKey record;
            if (!recordKeyFromJson(savedGlobal.at("record"), record, outError)) return false;
            const float value = savedGlobal.at("value").get<float>();
            if (!std::isfinite(value) || !globalVariables.emplace(record, value).second) {
                outError = "save contains invalid/duplicate global variable state";
                return false;
            }
        }
        std::uint64_t previousEventSequence = 0u;
        for (const Json& savedEvent : payload.at("story_events")) {
            StoryEventRuntimeState event;
            event.sequence = savedEvent.at("sequence").get<std::uint64_t>();
            if (event.sequence == 0u || event.sequence <= previousEventSequence ||
                !recordKeyFromJson(savedEvent.at("keyword"), event.keyword, outError)) {
                if (outError.empty()) outError = "story event sequences are not strictly ordered";
                return false;
            }
            previousEventSequence = event.sequence;
            for (const Json& savedArgument : savedEvent.at("arguments")) {
                PapyrusValue argument;
                if (!valueFromJson(savedArgument, argument, outError)) return false;
                event.arguments.push_back(std::move(argument));
            }
            storyEvents.push_back(std::move(event));
        }
        nextStoryEventSequence = payload.at("next_story_event_sequence").get<std::uint64_t>();
        if (nextStoryEventSequence == 0u ||
            (!storyEvents.empty() && nextStoryEventSequence <= storyEvents.back().sequence)) {
            outError = "invalid next story event sequence";
            return false;
        }
        if (saveVersion >= 8u) {
            std::uint64_t previousGiftSequence = 0u;
            for (const Json& savedRequest : payload.at("gift_menus")) {
                GiftMenuRequestState request;
                request.sequence = savedRequest.at("sequence").get<std::uint64_t>();
                if (request.sequence == 0u || request.sequence <= previousGiftSequence ||
                    !objectIdFromJson(savedRequest.at("actor"), request.actor, outError) ||
                    !objectIdFromJson(savedRequest.at("player"), request.player, outError)) {
                    if (outError.empty()) {
                        outError = "gift menu sequences are not strictly ordered";
                    }
                    return false;
                }
                previousGiftSequence = request.sequence;
                if (!savedRequest.at("filter_list").is_null() &&
                    !objectIdFromJson(
                        savedRequest.at("filter_list"), request.filterList, outError)) {
                    return false;
                }
                request.playerGives = savedRequest.at("player_gives").get<bool>();
                request.showStolenItems =
                    savedRequest.at("show_stolen_items").get<bool>();
                request.useFavorPoints =
                    savedRequest.at("use_favor_points").get<bool>();
                giftMenuRequests.push_back(std::move(request));
            }
            nextGiftMenuSequence =
                payload.at("next_gift_menu_sequence").get<std::uint64_t>();
            if (nextGiftMenuSequence == 0u ||
                (!giftMenuRequests.empty() &&
                 nextGiftMenuSequence <= giftMenuRequests.back().sequence)) {
                outError = "invalid next gift menu sequence";
                return false;
            }
        }
        scriptDebugLogs = payload.at("script_debug_logs").get<std::vector<std::string>>();
        if (std::any_of(scriptDebugLogs.begin(), scriptDebugLogs.end(),
                [](const std::string& log) { return log.empty(); })) {
            outError = "save contains an empty Papyrus debug-log name";
            return false;
        }
        if (saveVersion >= 2u) {
            for (const Json& savedAnimation : payload.at("animations")) {
                AnimationActorSnapshot animation;
                if (!objectIdFromJson(savedAnimation.at("object"), animation.object, outError) ||
                    !graphSnapshotFromJson(savedAnimation.at("third_person"),
                        animation.thirdPerson, outError)) return false;
                if (!savedAnimation.at("first_person").is_null()) {
                    odai::anim::BehaviorGraphSnapshot firstPerson;
                    if (!graphSnapshotFromJson(savedAnimation.at("first_person"),
                            firstPerson, outError)) return false;
                    animation.firstPerson = std::move(firstPerson);
                }
                animationSnapshots.push_back(std::move(animation));
            }
            for (const Json& savedPhysics : payload.at("physics")) {
                PhysicsCharacterSnapshot character;
                if (!objectIdFromJson(savedPhysics.at("object"), character.object, outError) ||
                    !vectorFromJson(savedPhysics.at("position"), character.position, outError) ||
                    !vectorFromJson(savedPhysics.at("velocity"), character.velocity, outError) ||
                    !vectorFromJson(savedPhysics.at("ground_normal"), character.groundNormal, outError)) {
                    return false;
                }
                if (!quaternionFromJson(
                        savedPhysics.at("rotation"), character.rotation, outError)) return false;
                character.grounded = savedPhysics.at("grounded").get<bool>();
                if (!savedPhysics.at("supporting_object").is_null()) {
                    ObjectId support;
                    if (!objectIdFromJson(savedPhysics.at("supporting_object"), support, outError)) {
                        return false;
                    }
                    character.supportingObject = std::move(support);
                }
                physicsSnapshots.push_back(std::move(character));
            }
        }
        std::sort(discoveries.begin(), discoveries.end());
        if (std::adjacent_find(discoveries.begin(), discoveries.end()) != discoveries.end()) {
            outError = "save contains duplicate map discoveries";
            return false;
        }
        if (!vmFromJson(payload.at("papyrus"), vm, outError)) return false;
        if (payload.contains("tes3") &&
            !tes3RuntimeFromJson(payload.at("tes3"), tes3State, outError)) return false;
    } catch (const std::exception& exception) {
        outError = std::string("invalid ODAI save payload: ") + exception.what(); return false;
    }

    if (saveVersion < 7u) {
        const auto inventoryContains = [&](const RecordKey& item) {
            return std::any_of(objects.begin(), objects.end(),
                [&](const RuntimeObject& object) {
                    return std::any_of(object.inventory.begin(), object.inventory.end(),
                        [&](const InventoryEntry& entry) {
                            return entry.item == item && entry.count > 0;
                        });
                });
        };
        for (auto& [questName, quest] : quests) {
            const auto registered = registeredQuests.find(questName);
            if (registered == registeredQuests.end()) continue;
            for (QuestAliasRuntimeState& alias : quest.aliases) {
                const auto definition = std::find_if(
                    registered->second.aliases.begin(), registered->second.aliases.end(),
                    [&](const QuestAliasRuntimeState& candidate) {
                        return candidate.id == alias.id;
                    });
                if (definition == registered->second.aliases.end()) continue;
                alias.createdObject = definition->createdObject;
                alias.createdInAliasId = definition->createdInAliasId;
                alias.createdLevel = definition->createdLevel;
                alias.createdObjectMaterialized = alias.createdObject.valid() &&
                    inventoryContains(alias.createdObject);
            }
        }
    }
    if (tes3State.present) {
        if (session.tes3().content() == nullptr) {
            outError = "save contains TES3 state but the configured session has no TES3 content";
            return false;
        }
        if (tes3State.playerState.object.valid() &&
            tes3State.playerState.object != session.playerObject()) {
            outError = "saved TES3 player ObjectId differs from configured player";
            return false;
        }
        for (const auto& [key, quest] : tes3State.quests) {
            const auto definition = session.tes3().content()->dialogues().find(key);
            if (definition == session.tes3().content()->dialogues().end() ||
                definition->second.type != Tes3DialogueType::Journal) {
                outError = "saved TES3 journal no longer resolves: " + key.toString();
                return false;
            }
            for (const RecordKey& info : quest.visitedEntries) {
                const bool found = std::any_of(definition->second.infos.begin(),
                    definition->second.infos.end(), [&](const Tes3DialogueInfo& value) {
                        return value.record == info;
                    });
                if (!found) {
                    outError = "saved TES3 journal INFO no longer resolves: " + info.toString();
                    return false;
                }
            }
        }
        for (const RecordKey& topic : tes3State.knownTopics) {
            if (!session.tes3().content()->dialogues().contains(topic)) {
                outError = "saved known TES3 topic no longer resolves: " + topic.toString();
                return false;
            }
        }
        for (const auto& [id, thread] : tes3State.threads) {
            (void)id;
            const auto program = session.tes3().scripts().programs().find(thread.program);
            if (program == session.tes3().scripts().programs().end() ||
                thread.instruction > program->second.instructions.size()) {
                outError = "saved TES3 script cursor no longer resolves: " + thread.program;
                return false;
            }
        }
        for (const auto& [object, override] : tes3State.referenceOverrides) {
            (void)override;
            if (!session.tes3().content()->references().contains(object)) {
                outError = "saved TES3 reference override no longer resolves: " +
                    object.toString();
                return false;
            }
        }
        for (const auto& [target, spells] : tes3State.activeSpells) {
            if (target != session.playerObject() &&
                !session.tes3().content()->references().contains(target)) {
                outError = "saved TES3 active-spell target no longer resolves: " +
                    target.toString();
                return false;
            }
            for (const Tes3ActiveSpell& spell : spells) {
                if (!session.tes3().content()->spells().contains(spell.spell)) {
                    outError = "saved TES3 active spell no longer resolves: " +
                        spell.spell.toString();
                    return false;
                }
            }
        }
    }
    // Objective wording belongs to installed content, not to the save. Merge
    // it back from the definitions registered before staging the save state.
    for (auto& [questName, quest] : quests) {
        const auto registered = registeredQuests.find(questName);
        if (registered == registeredQuests.end()) continue;
        for (QuestObjectiveState& objective : quest.objectives) {
            const auto definition = std::find_if(
                registered->second.objectives.begin(), registered->second.objectives.end(),
                [&](const QuestObjectiveState& candidate) {
                    return candidate.index == objective.index;
                });
            if (definition != registered->second.objectives.end()) {
                objective.displayText = definition->displayText;
            }
        }
        for (QuestAliasRuntimeState& alias : quest.aliases) {
            const auto definition = std::find_if(
                registered->second.aliases.begin(), registered->second.aliases.end(),
                [&](const QuestAliasRuntimeState& candidate) {
                    return candidate.id == alias.id;
                });
            if (definition == registered->second.aliases.end()) continue;
            alias.findMatchingReferenceInAliasId =
                definition->findMatchingReferenceInAliasId;
            alias.referenceType = definition->referenceType;
        }
    }
    for (const auto& [questName, quest] : quests) {
        (void)questName;
        std::int32_t previousAliasId = std::numeric_limits<std::int32_t>::min();
        for (const QuestAliasRuntimeState& alias : quest.aliases) {
            if (alias.id <= previousAliasId) {
                outError = "saved quest aliases are not strictly ordered";
                return false;
            }
            previousAliasId = alias.id;
            if (alias.createdObject.valid()) {
                const auto owner = std::find_if(
                    quest.aliases.begin(), quest.aliases.end(),
                    [&](const QuestAliasRuntimeState& candidate) {
                        return candidate.id == alias.createdInAliasId;
                    });
                if (alias.createdInAliasId < 0 || owner == quest.aliases.end()) {
                    outError = "saved created-object alias has no owner alias";
                    return false;
                }
            } else if (alias.createdInAliasId >= 0 ||
                       alias.createdObjectMaterialized) {
                outError = "saved quest alias has incomplete created-object state";
                return false;
            }
        }
    }

    if (saveVersion >= 2u) {
        std::set<ObjectId> physicalActors;
        for (const PhysicsCharacterSnapshot& character : physicsSnapshots) {
            if (!physicalActors.insert(character.object).second) {
                outError = "save contains duplicate physical actor: " +
                    character.object.toString();
                return false;
            }
        }
        std::set<ObjectId> animatedActors;
        for (const AnimationActorSnapshot& animation : animationSnapshots) {
            if (!animatedActors.insert(animation.object).second) {
                outError = "save contains duplicate animated actor: " +
                    animation.object.toString();
                return false;
            }
            if (!physicalActors.contains(animation.object)) {
                outError = "save animation has no matching physical actor: " +
                    animation.object.toString();
                return false;
            }
        }
    }

    if (savedFingerprint != session.config().contentFingerprint) {
        if (!options.recordAvailable) {
            outError = "save content fingerprint differs and no RecordKey reconciler was supplied";
            return false;
        }
        std::set<RecordKey> keys;
        for (const RuntimeObject& object : objects) collectRecordKeys(object, keys);
        collectRecordKeys(vm, keys);
        if (tes3State.present) {
            for (const auto& [key, quest] : tes3State.quests) {
                keys.insert(key);
                keys.insert(quest.visitedEntries.begin(), quest.visitedEntries.end());
            }
            for (const Tes3JournalVisit& visit : tes3State.chronology) {
                keys.insert(visit.quest);
                keys.insert(visit.info);
            }
            keys.insert(tes3State.knownTopics.begin(), tes3State.knownTopics.end());
            if (tes3State.dialogue.currentTopic.valid()) {
                keys.insert(tes3State.dialogue.currentTopic);
            }
            if (tes3State.dialogue.currentInfo.valid()) {
                keys.insert(tes3State.dialogue.currentInfo);
            }
            keys.insert(tes3State.dialogue.exhaustedInfos.begin(),
                        tes3State.dialogue.exhaustedInfos.end());
            for (const auto& [object, override] : tes3State.referenceOverrides) {
                (void)override;
                if (object.kind == ObjectIdKind::PersistentReference) {
                    keys.insert(object.reference);
                }
            }
            for (const auto& [target, spells] : tes3State.activeSpells) {
                if (target.kind == ObjectIdKind::PersistentReference) keys.insert(target.reference);
                for (const Tes3ActiveSpell& spell : spells) {
                    keys.insert(spell.spell);
                    if (spell.caster.kind == ObjectIdKind::PersistentReference) {
                        keys.insert(spell.caster.reference);
                    }
                }
            }
        }
        for (const auto& [name, quest] : quests) {
            (void)name;
            if (quest.record.valid()) keys.insert(quest.record);
            for (const QuestAliasRuntimeState& alias : quest.aliases) {
                if (alias.createdObject.valid()) keys.insert(alias.createdObject);
                if (alias.target.kind == ObjectIdKind::PersistentReference) {
                    keys.insert(alias.target.reference);
                }
            }
        }
        keys.insert(discoveries.begin(), discoveries.end());
        for (const auto& [scene, playing] : scenes) {
            (void)playing;
            keys.insert(scene);
        }
        if (forcedWeather.valid()) keys.insert(forcedWeather);
        for (const auto& [record, location] : locations) {
            keys.insert(record);
            if (location.parent.valid()) keys.insert(location.parent);
            keys.insert(location.keywords.begin(), location.keywords.end());
            for (const auto& [keyword, value] : location.keywordData) {
                (void)value;
                keys.insert(keyword);
            }
        }
        for (const auto& [record, value] : globalVariables) {
            (void)value;
            keys.insert(record);
        }
        for (const StoryEventRuntimeState& event : storyEvents) {
            keys.insert(event.keyword);
            for (const PapyrusValue& argument : event.arguments) {
                collectRecordKeys(argument, keys);
            }
        }
        for (const GiftMenuRequestState& request : giftMenuRequests) {
            if (request.actor.kind == ObjectIdKind::PersistentReference) {
                keys.insert(request.actor.reference);
            }
            if (request.player.kind == ObjectIdKind::PersistentReference) {
                keys.insert(request.player.reference);
            }
            if (request.filterList.kind == ObjectIdKind::PersistentReference) {
                keys.insert(request.filterList.reference);
            }
        }
        for (const AnimationActorSnapshot& animation : animationSnapshots) {
            if (animation.object.kind == ObjectIdKind::PersistentReference) {
                keys.insert(animation.object.reference);
            }
        }
        for (const PhysicsCharacterSnapshot& character : physicsSnapshots) {
            if (character.object.kind == ObjectIdKind::PersistentReference) {
                keys.insert(character.object.reference);
            }
            if (character.supportingObject.has_value() &&
                character.supportingObject->kind == ObjectIdKind::PersistentReference) {
                keys.insert(character.supportingObject->reference);
            }
        }
        std::vector<std::string> missing;
        for (const RecordKey& key : keys) if (!options.recordAvailable(key)) missing.push_back(key.toString());
        if (!missing.empty()) {
            outError = "changed content profile is missing " + std::to_string(missing.size()) +
                " required records; first missing: " + missing.front();
            return false;
        }
        outReport.contentReconciled = true;
        outReport.diagnostics.push_back(
            "content fingerprint changed; all saved RecordKeys reconciled successfully");
    }

    // Validate all staged state before mutating the configured session.
    PapyrusVmSnapshot oldVm = session.papyrus().snapshot();
    const std::vector<AnimationActorSnapshot> oldAnimations = session.animationSnapshots();
    const std::vector<PhysicsCharacterSnapshot> oldPhysics = session.physicsSnapshots();
    std::string restoreError;
    if (!session.papyrus().restore(vm, restoreError)) {
        outError = restoreError; return false;
    }
    BethesdaWorld stagedWorld;
    stagedWorld.restore(objects, nextRuntimeId, nextCommandSequence, restoreError);
    if (!restoreError.empty()) {
        const std::string stagedWorldError = restoreError;
        std::string rollbackError;
        (void)session.papyrus().restore(oldVm, rollbackError);
        outError = "invalid saved world: " + stagedWorldError;
        if (!rollbackError.empty()) outError += "; VM rollback failed: " + rollbackError;
        return false;
    }
    for (const GiftMenuRequestState& request : giftMenuRequests) {
        const RuntimeObject* actor = stagedWorld.find(request.actor);
        const RuntimeObject* player = stagedWorld.find(request.player);
        if (!request.actor.valid() || !request.player.valid() ||
            actor == nullptr || actor->kind != RuntimeObjectKind::Actor ||
            player == nullptr || player->kind != RuntimeObjectKind::Actor ||
            (request.filterList.valid() &&
             request.filterList.kind != ObjectIdKind::PersistentReference)) {
            outError = "save contains an invalid gift menu request";
            return false;
        }
    }
    std::vector<PhysicsCharacterSnapshot> desiredPhysics = physicsSnapshots;
    if (saveVersion >= 2u) {
        if (!session.restoreAnimationSnapshots(animationSnapshots, restoreError) ||
            !session.restorePhysicsSnapshots(desiredPhysics, restoreError)) {
            std::string ignored;
            (void)session.restoreAnimationSnapshots(oldAnimations, ignored);
            (void)session.restorePhysicsSnapshots(oldPhysics, ignored);
            (void)session.papyrus().restore(oldVm, ignored);
            outError = "invalid saved physical/animation state: " + restoreError;
            return false;
        }
    } else {
        // V1 had no graph/Jolt payload. Preserve the actor transform already
        // restored above and start each registered graph from its initial state.
        desiredPhysics = oldPhysics;
        for (PhysicsCharacterSnapshot& character : desiredPhysics) {
            if (const RuntimeObject* object = stagedWorld.find(character.object)) {
                character.position = {
                    static_cast<float>(object->transform.position[0]),
                    static_cast<float>(object->transform.position[1]),
                    static_cast<float>(object->transform.position[2])};
                character.velocity = {};
                character.grounded = false;
                character.supportingObject.reset();
            }
        }
        if (!session.restorePhysicsSnapshots(desiredPhysics, restoreError)) {
            std::string ignored;
            (void)session.papyrus().restore(oldVm, ignored);
            outError = "invalid migrated physical state: " + restoreError;
            return false;
        }
        outReport.diagnostics.push_back(
            "version-1 save initialized animation and Jolt state from actor transforms");
    }
    if (saveVersion < 3u) {
        outReport.diagnostics.push_back(
            "pre-version-3 save initialized actor AI path cursors from resident content");
    }
    if (saveVersion < 4u) {
        outReport.diagnostics.push_back(
            "pre-version-4 save promoted walk-only AI paths to typed navigation actions");
    }
    if (saveVersion < 5u) {
        outReport.diagnostics.push_back(
            "pre-version-5 save initialized deterministic melee cooldown state");
    }
    if (saveVersion < 6u) {
        outReport.diagnostics.push_back(
            "pre-version-6 save initialized VMAD-backed activator puzzle state");
    }
    if (saveVersion < 7u) {
        outReport.diagnostics.push_back(
            "pre-version-7 save reconstructed quest-created inventory aliases from installed content");
    }
    if (saveVersion < 8u) {
        outReport.diagnostics.push_back(
            "pre-version-8 save initialized actor factions, gift menus, and game-time timer metadata");
    }
    if (!tes3State.present && session.config().game == importer::fnv::BethesdaGame::Morrowind) {
        outReport.diagnostics.push_back(
            "save has no TES3 extension; journal, MWScript, topics, and dialogue initialized empty");
    }
    session.world() = std::move(stagedWorld);
    session.questsForRestore() = std::move(quests);
    session.statisticsForRestore() = std::move(statistics);
    session.discoveriesForRestore() = std::move(discoveries);
    session.scenesForRestore() = std::move(scenes);
    session.forcedWeatherForRestore() = std::move(forcedWeather);
    session.locationsForRestore() = std::move(locations);
    session.globalVariablesForRestore() = std::move(globalVariables);
    session.storyEventsForRestore() = std::move(storyEvents);
    session.giftMenuRequestsForRestore() = std::move(giftMenuRequests);
    session.scriptDebugLogsForRestore() = std::move(scriptDebugLogs);
    session.setNextStoryEventSequence(nextStoryEventSequence);
    session.setNextGiftMenuSequence(nextGiftMenuSequence);
    if (tes3State.present) {
        session.tes3().journal().questsForRestore() = std::move(tes3State.quests);
        session.tes3().journal().chronologyForRestore() = std::move(tes3State.chronology);
        session.tes3().journal().setNextSequence(tes3State.nextJournalSequence);
        session.tes3().knownTopicsForRestore() = std::move(tes3State.knownTopics);
        session.tes3().scripts().globals() = std::move(tes3State.globals);
        session.tes3().scripts().threadsForRestore() = std::move(tes3State.threads);
        session.tes3().scripts().setNextThreadId(tes3State.nextThreadId);
        session.tes3().dialogueForRestore() = std::move(tes3State.dialogue);
        session.tes3().playerState() = std::move(tes3State.playerState);
        session.tes3().referenceOverridesForRestore() =
            std::move(tes3State.referenceOverrides);
        session.tes3().activeSoundsForRestore() = std::move(tes3State.activeSounds);
        session.tes3().activeSpellsForRestore() = std::move(tes3State.activeSpells);
    }
    session.clock().reset(tick, accumulator);
    session.setRandomState(randomState);
    outError.clear();
    return true;
}

}  // namespace odai::bethesda
