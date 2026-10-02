#include "bethesda/bethesda_session.h"
#include "bethesda/runtime_ids.h"
#include "import/bethesda/morrowind_terrain.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using Json = nlohmann::json;

int fail(const std::string& message) {
    std::cerr << "headless engine: " << message << '\n';
    std::cout << Json{{"status", "fail"}, {"error", message}}.dump() << '\n';
    return 1;
}

std::array<double, 3> vector3(const Json& value, const char* name);

Json runtimeSnapshot(const odai::bethesda::BethesdaSession& session) {
    using namespace odai::bethesda;
    Json objects = Json::array();
    for (const ObjectId& id : session.world().orderedObjectIds()) {
        const RuntimeObject* object = session.world().find(id);
        if (!object) continue;
        const char* kind = object->kind == RuntimeObjectKind::Actor ? "actor" :
            object->kind == RuntimeObjectKind::Item ? "item" :
            object->kind == RuntimeObjectKind::Container ? "container" :
            object->kind == RuntimeObjectKind::Door ? "door" :
            object->kind == RuntimeObjectKind::Activator ? "activator" :
            object->kind == RuntimeObjectKind::Projectile ? "projectile" : "unknown";
        const char* spaceKind = object->currentSpace.kind == RuntimeSpaceKind::Exterior
            ? "exterior" : object->currentSpace.kind == RuntimeSpaceKind::Interior
            ? "interior" : "unknown";
        Json inventory = Json::array();
        for (const InventoryEntry& entry : object->inventory)
            inventory.push_back({{"item", entry.item.toString()}, {"count", entry.count},
                {"equipped", entry.equipped}});
        std::sort(inventory.begin(), inventory.end(), [](const Json& a, const Json& b) {
            return a.at("item").get<std::string>() < b.at("item").get<std::string>();
        });
        Json state{{"id", id.toString()}, {"base", object->base.toString()},
            {"kind", kind}, {"enabled", object->enabled},
            {"transform", {{"position", object->transform.position},
                {"rotation_radians", object->transform.rotationRadians},
                {"scale", object->transform.scale}}},
            {"space", {{"kind", spaceKind}, {"cell", object->currentSpace.cell.toString()},
                {"worldspace", object->currentSpace.worldspace.toString()},
                {"grid", {object->currentSpace.gridX, object->currentSpace.gridZ}}}},
            {"inventory", inventory}};
        if (object->actorValues) {
            const ActorValues& values = *object->actorValues;
            state["actor_values"] = {{"health", values.health},
                {"stamina", values.stamina}, {"magicka", values.magicka},
                {"max_health", values.maxHealth}, {"max_stamina", values.maxStamina},
                {"max_magicka", values.maxMagicka}, {"dead", values.dead}};
        }
        objects.push_back(std::move(state));
    }
    Json characters = Json::array();
    auto physics = session.physics().snapshot();
    std::sort(physics.begin(), physics.end(), [](const auto& a, const auto& b) {
        return a.object < b.object;
    });
    for (const auto& character : physics)
        characters.push_back({{"object", character.object.toString()},
            {"position", {character.position.x, character.position.y, character.position.z}},
            {"velocity", {character.velocity.x, character.velocity.y, character.velocity.z}},
            {"grounded", character.grounded}});
    Json quests = Json::array();
    for (const auto& [key, quest] : session.quests()) {
        Json objectives = Json::array();
        for (const auto& objective : quest.objectives)
            objectives.push_back({{"index", objective.index},
                {"displayed", objective.displayed}, {"completed", objective.completed},
                {"failed", objective.failed}});
        std::sort(objectives.begin(), objectives.end(), [](const Json& a, const Json& b) {
            return a.at("index").get<int>() < b.at("index").get<int>();
        });
        quests.push_back({{"id", key}, {"stage", quest.stage},
            {"running", quest.running}, {"completed", quest.completed},
            {"failed", quest.failed}, {"objectives", objectives}});
    }
    return {{"tick", session.clock().tick()},
        {"random_state", session.randomState()},
        {"state_hash", std::to_string(session.deterministicHash())},
        {"player_object", session.playerObject().toString()},
        {"objects", objects}, {"characters", characters}, {"quests", quests}};
}

int runStaticPhysicsFixture(const Json& fixture) {
    using namespace odai::bethesda;
    BethesdaPhysicsWorld physics;
    std::string error;
    const auto creationStart = std::chrono::steady_clock::now();
    if (!physics.initialize(error)) return fail("static physics initialization: " + error);
    const auto initialized = std::chrono::steady_clock::now();
    for (const Json& source : fixture.at("objects")) {
        RuntimeObject object;
        object.id = ObjectId::runtime(source.at("id").get<std::uint64_t>());
        object.transform.position = vector3(source.at("position"), "object position");
        const auto angles = vector3(source.at("rotation"), "object rotation");
        for (int axis = 0; axis < 3; ++axis)
            object.transform.rotationRadians[axis] = static_cast<float>(angles[axis]);
        object.transform.scale = source.at("scale").get<float>();
        std::vector<odai::math::Vector3> vertices;
        for (const Json& vertex : source.at("vertices")) {
            const auto value = vector3(vertex, "collision vertex");
            vertices.push_back({static_cast<float>(value[0]),
                static_cast<float>(value[1]), static_cast<float>(value[2])});
        }
        const auto indices = source.at("indices").get<std::vector<std::uint32_t>>();
        if (!physics.addStaticCollision(object, vertices, indices, error))
            return fail("static collision " + object.id.toString() + ": " + error);
    }
    const auto creationEnd = std::chrono::steady_clock::now();
    Json results = Json::array();
    for (const Json& query : fixture.at("queries")) {
        const auto start = vector3(query.at("from"), "query start");
        const auto end = vector3(query.at("to"), "query end");
        const odai::math::Vector3 from{static_cast<float>(start[0]),
            static_cast<float>(start[1]), static_cast<float>(start[2])};
        const odai::math::Vector3 to{static_cast<float>(end[0]),
            static_cast<float>(end[1]), static_cast<float>(end[2])};
        const auto hit = query.value("shape", std::string("ray")) == "sphere"
            ? physics.castSphere(from, to, query.at("radius").get<float>())
            : physics.castRay(from, to);
        const auto label = query.at("name").get<std::string>();
        if (hit.has_value() != query.at("hit").get<bool>())
            return fail("static query " + label + ": unexpected hit state");
        if (hit) {
            const auto expectedPosition = vector3(query.at("position"), "expected hit position");
            const auto expectedNormal = vector3(query.at("normal"), "expected hit normal");
            const float tolerance = query.value("tolerance", 0.25f);
            const auto near = [tolerance](double a, double b) {
                return std::abs(a - b) <= tolerance;
            };
            if (!hit->object || *hit->object != ObjectId::runtime(query.at("object").get<std::uint64_t>()) ||
                !near(hit->position.x, expectedPosition[0]) ||
                !near(hit->position.y, expectedPosition[1]) ||
                !near(hit->position.z, expectedPosition[2]) ||
                !near(hit->normal.x, expectedNormal[0]) ||
                !near(hit->normal.y, expectedNormal[1]) ||
                !near(hit->normal.z, expectedNormal[2]) ||
                !near(hit->distance, query.at("distance").get<double>()))
                return fail("static query " + label + ": wrong object, position, normal, or distance; got " +
                    Json{{"object", hit->object ? hit->object->toString() : "none"},
                        {"position", {hit->position.x, hit->position.y, hit->position.z}},
                        {"normal", {hit->normal.x, hit->normal.y, hit->normal.z}},
                        {"distance", hit->distance}}.dump());
            results.push_back({{"name", label}, {"from", start}, {"to", end},
                {"object", hit->object->toString()},
                {"position", {hit->position.x, hit->position.y, hit->position.z}},
                {"normal", {hit->normal.x, hit->normal.y, hit->normal.z}},
                {"distance", hit->distance}});
        } else results.push_back({{"name", label}, {"from", start}, {"to", end},
            {"hit", false}});
    }
    const auto queryEnd = std::chrono::steady_clock::now();
    const auto milliseconds = [](auto begin, auto end) {
        return std::chrono::duration<double, std::milli>(end - begin).count();
    };
    std::cout << Json{{"status", "pass"}, {"scenario", fixture.at("name")},
        {"static_objects", fixture.at("objects").size()}, {"shape", "triangle_mesh"},
        {"queries", results}, {"physics_init_ms", milliseconds(creationStart, initialized)},
        {"static_creation_ms", milliseconds(initialized, creationEnd)},
        {"query_batch_ms", milliseconds(creationEnd, queryEnd)}}.dump() << '\n';
    return 0;
}

int runTerrainFixture(const Json& fixture) {
    using namespace odai::importer::bethesda;
    odai::bethesda::BethesdaPhysicsWorld physics;
    std::string error;
    Json results = Json::array();
    const auto creationStart = std::chrono::steady_clock::now();
    for (const Json& source : fixture.at("cells")) {
        FalloutCellRecord cell;
        cell.gridX = source.at("x").get<int>();
        cell.gridZ = source.at("z").get<int>();
        cell.hasGridCoords = true;
        cell.land = std::make_unique<FalloutLandRecord>();
        cell.land->gridSize = kMorrowindLandGridSize;
        cell.land->hasHeights = true;
        cell.land->heights.resize(65u * 65u);
        const float base = source.at("base_height").get<float>();
        const float rowSlope = source.at("row_slope").get<float>();
        const float colSlope = source.at("col_slope").get<float>();
        for (int row = 0; row < 65; ++row)
            for (int col = 0; col < 65; ++col)
                cell.land->heights[static_cast<std::size_t>(row * 65 + col)] =
                    base + row * rowSlope + col * colSlope;
        if (!physics.addTerrainCell(morrowindTerrainSurface(cell), error))
            return fail("LAND terrain (" + std::to_string(cell.gridX) + "," +
                std::to_string(cell.gridZ) + "): " + error);
    }
    const auto creationEnd = std::chrono::steady_clock::now();
    const auto queryStart = creationEnd;
    for (const Json& query : fixture.at("queries")) {
        const auto origin = vector3(query.at("origin"), "terrain query origin");
        const odai::math::Vector3 from{static_cast<float>(origin[0]),
            static_cast<float>(origin[1]), static_cast<float>(origin[2])};
        const float distance = query.at("distance").get<float>();
        const auto hit = query.value("shape", std::string("ray")) == "sphere"
            ? physics.castSphere(from, {from.x, from.y - distance, from.z},
                query.at("radius").get<float>())
            : physics.castDown(from, distance);
        const bool expectedHit = query.at("hit").get<bool>();
        if (hit.has_value() != expectedHit)
            return fail("terrain query " + std::to_string(results.size()) +
                " expected hit=" + std::to_string(expectedHit) +
                " at (" + std::to_string(from.x) + "," + std::to_string(from.z) + ")");
        if (hit) {
            const auto& expectedCell = query.at("cell");
            if (!hit->terrainCell || hit->terrainCell->x != expectedCell.at(0).get<int>() ||
                hit->terrainCell->z != expectedCell.at(1).get<int>() ||
                std::abs(hit->position.y - query.at("height").get<float>()) > 0.1f ||
                std::abs(hit->distance - query.at("hit_distance").get<float>()) > 0.5f)
                return fail("terrain query " + std::to_string(results.size()) +
                    " returned wrong cell, elevation, or distance");
            results.push_back({{"cell", {hit->terrainCell->x, hit->terrainCell->z}},
                {"position", {hit->position.x, hit->position.y, hit->position.z}},
                {"normal", {hit->normal.x, hit->normal.y, hit->normal.z}},
                {"distance", hit->distance}});
        } else results.push_back(nullptr);
    }
    const auto queryEnd = std::chrono::steady_clock::now();
    const auto milliseconds = [](auto begin, auto end) {
        return std::chrono::duration<double, std::milli>(end - begin).count();
    };
    std::cout << Json{{"status", "pass"}, {"scenario", fixture.at("name")},
        {"terrain_queries", results},
        {"terrain_creation_ms", milliseconds(creationStart, creationEnd)},
        {"query_batch_ms", milliseconds(queryStart, queryEnd)},
        {"source_height_bytes", fixture.at("cells").size() * 65u * 65u * sizeof(float)}}.dump() << '\n';
    return 0;
}

std::array<double, 3> vector3(const Json& value, const char* name) {
    if (!value.is_array() || value.size() != 3u) {
        throw std::runtime_error(std::string(name) + " must contain three numbers");
    }
    std::array<double, 3> result{};
    for (std::size_t i = 0; i < result.size(); ++i) {
        result[i] = value.at(i).get<double>();
        if (!std::isfinite(result[i]) || std::abs(result[i]) > 1'000'000.0) {
            throw std::runtime_error(std::string(name) + " must contain finite engine coordinates");
        }
    }
    return result;
}

void validateScenario(const Json& fixture) {
    const auto require = [](bool condition, const std::string& path, const char* rule) {
        if (!condition) throw std::runtime_error(path + " " + rule);
    };
    const auto fields = [&](const Json& object, const std::string& path,
                            std::initializer_list<const char*> required,
                            std::initializer_list<const char*> optional) {
        require(object.is_object(), path, "must be an object");
        for (const char* key : required)
            require(object.contains(key), path + "." + key, "is required");
        for (auto it = object.begin(); it != object.end(); ++it) {
            const auto allowed = [&](const char* key) { return it.key() == key; };
            require(std::any_of(required.begin(), required.end(), allowed) ||
                std::any_of(optional.begin(), optional.end(), allowed),
                path + "." + it.key(), "is not a supported field");
        }
    };
    const auto number = [&](const Json& object, const char* key, const std::string& path) {
        require(object.at(key).is_number(), path + "." + key, "must be a number");
        require(std::isfinite(object.at(key).get<double>()), path + "." + key,
            "must be finite");
    };
    const auto textField = [&](const Json& object, const char* key, const std::string& path) {
        require(object.at(key).is_string() && !object.at(key).get<std::string>().empty(),
            path + "." + key, "must be a nonempty string");
    };
    require(fixture.is_object(), "$", "must be an object");
    if (fixture.contains("version"))
        require(fixture.at("version").is_number_integer() && fixture.at("version") == 1,
            "$.version", "must be 1");
    const std::string kind = fixture.value("kind", std::string("actor"));
    require(kind == "actor" || kind == "static_physics" || kind == "terrain_physics",
        "$.kind", "is unsupported");
    if (kind == "actor") {
        fields(fixture, "$", {"name", "seed", "step_seconds", "steps", "actor"},
            {"version", "kind", "expect", "snapshots", "inventory_actions", "inventory_expect"});
        require(fixture.at("seed").is_number_unsigned() ||
            (fixture.at("seed").is_number_integer() && fixture.at("seed").get<std::int64_t>() >= 0),
            "$.seed", "must be a nonnegative integer");
        require(fixture.at("steps").is_number_unsigned() ||
            (fixture.at("steps").is_number_integer() && fixture.at("steps").get<std::int64_t>() >= 0),
            "$.steps", "must be a nonnegative integer");
        number(fixture, "step_seconds", "$");
        const Json& actor = fixture.at("actor");
        fields(actor, "$.actor", {"id", "base", "position", "desired_velocity"}, {"inventory"});
        textField(actor, "id", "$.actor");
        textField(actor, "base", "$.actor");
        vector3(actor.at("position"), "$.actor.position");
        vector3(actor.at("desired_velocity"), "$.actor.desired_velocity");
        if (actor.contains("inventory")) {
            require(actor.at("inventory").is_array(), "$.actor.inventory", "must be an array");
            for (std::size_t i = 0; i < actor.at("inventory").size(); ++i) {
                const Json& item = actor.at("inventory").at(i);
                const std::string path = "$.actor.inventory[" + std::to_string(i) + "]";
                fields(item, path, {"id", "count"}, {"type"});
                textField(item, "id", path);
                require(item.at("count").is_number_integer() && item.at("count").get<int>() > 0,
                    path + ".count", "must be a positive integer");
                if (item.contains("type")) textField(item, "type", path);
            }
        }
        if (fixture.contains("expect")) {
            fields(fixture.at("expect"), "$.expect",
                {"x_min", "x_max", "y_max", "min_world_commands"}, {});
            for (const char* key : {"x_min", "x_max", "y_max"})
                number(fixture.at("expect"), key, "$.expect");
            require(fixture.at("expect").at("min_world_commands").is_number_integer(),
                "$.expect.min_world_commands", "must be an integer");
        }
        if (fixture.contains("inventory_actions")) {
            require(fixture.at("inventory_actions").is_array(), "$.inventory_actions", "must be an array");
            for (const Json& action : fixture.at("inventory_actions")) {
                fields(action, "$.inventory_actions[]", {"id"}, {"type"});
                textField(action, "id", "$.inventory_actions[]");
                if (action.contains("type")) textField(action, "type", "$.inventory_actions[]");
            }
        }
        if (fixture.contains("inventory_expect"))
            require(fixture.contains("inventory_actions") && fixture.at("inventory_expect").is_array(),
                "$.inventory_expect", "requires inventory_actions and must be an array");
        if (fixture.contains("snapshots")) {
            fields(fixture.at("snapshots"), "$.snapshots", {}, {"ticks"});
            if (fixture.at("snapshots").contains("ticks"))
                require(fixture.at("snapshots").at("ticks").is_array(),
                    "$.snapshots.ticks", "must be an array");
        }
    } else {
        fields(fixture, "$", {"kind", "name", kind == "static_physics" ? "objects" : "cells", "queries"},
            {"version"});
        const char* collection = kind == "static_physics" ? "objects" : "cells";
        require(fixture.at(collection).is_array(), std::string("$.") + collection, "must be an array");
        require(fixture.at("queries").is_array(), "$.queries", "must be an array");
        for (const Json& object : fixture.at(collection)) {
            if (kind == "static_physics") {
                fields(object, "$.objects[]",
                    {"id", "position", "rotation", "scale", "vertices", "indices"}, {});
                require(object.at("id").is_number_unsigned() || object.at("id").is_number_integer(),
                    "$.objects[].id", "must be an integer");
                vector3(object.at("position"), "$.objects[].position");
                vector3(object.at("rotation"), "$.objects[].rotation");
                number(object, "scale", "$.objects[]");
                require(object.at("vertices").is_array(), "$.objects[].vertices", "must be an array");
                for (const Json& vertex : object.at("vertices"))
                    vector3(vertex, "$.objects[].vertices[]");
                require(object.at("indices").is_array(), "$.objects[].indices", "must be an array");
                for (const Json& index : object.at("indices"))
                    require(index.is_number_integer(), "$.objects[].indices[]", "must be an integer");
            } else {
                fields(object, "$.cells[]",
                    {"x", "z", "base_height", "row_slope", "col_slope"}, {});
                for (const char* key : {"x", "z"})
                    require(object.at(key).is_number_integer(), std::string("$.cells[].") + key,
                        "must be an integer");
                for (const char* key : {"base_height", "row_slope", "col_slope"})
                    number(object, key, "$.cells[]");
            }
        }
        for (const Json& query : fixture.at("queries")) {
            if (kind == "static_physics") {
                fields(query, "$.queries[]", {"name", "from", "to", "hit"},
                    {"shape", "radius", "object", "position", "normal", "distance", "tolerance"});
                textField(query, "name", "$.queries[]");
                vector3(query.at("from"), "$.queries[].from");
                vector3(query.at("to"), "$.queries[].to");
                if (query.contains("position")) vector3(query.at("position"), "$.queries[].position");
                if (query.contains("normal")) vector3(query.at("normal"), "$.queries[].normal");
                if (query.contains("object"))
                    require(query.at("object").is_number_integer(), "$.queries[].object", "must be an integer");
                for (const char* key : {"distance", "tolerance", "radius"})
                    if (query.contains(key)) number(query, key, "$.queries[]");
            } else {
                fields(query, "$.queries[]", {"origin", "distance", "hit"},
                    {"shape", "radius", "cell", "height", "hit_distance"});
                vector3(query.at("origin"), "$.queries[].origin");
                number(query, "distance", "$.queries[]");
                for (const char* key : {"radius", "height", "hit_distance"})
                    if (query.contains(key)) number(query, key, "$.queries[]");
                if (query.contains("cell"))
                    require(query.at("cell").is_array() && query.at("cell").size() == 2,
                        "$.queries[].cell", "must contain two coordinates");
            }
            require(query.at("hit").is_boolean(), "$.queries[].hit", "must be a boolean");
            if (query.contains("shape"))
                require(query.at("shape").is_string() &&
                    (query.at("shape") == "ray" || query.at("shape") == "sphere"),
                    "$.queries[].shape", "must be ray or sphere");
            if (query.value("shape", std::string("ray")) == "sphere")
                require(query.contains("radius") && query.at("radius").get<double>() > 0,
                    "$.queries[].radius", "must be positive for a sphere");
        }
    }
    textField(fixture, "name", "$");
}

int run(const Json& fixture) {
    validateScenario(fixture);
    if (fixture.value("kind", std::string{}) == "static_physics")
        return runStaticPhysicsFixture(fixture);
    if (fixture.value("kind", std::string{}) == "terrain_physics")
        return runTerrainFixture(fixture);
    const std::string name = fixture.at("name").get<std::string>();
    const double stepSeconds = fixture.at("step_seconds").get<double>();
    const std::uint32_t steps = fixture.at("steps").get<std::uint32_t>();
    const std::uint32_t seed = fixture.at("seed").get<std::uint32_t>();
    if (!(stepSeconds > 0.0 && stepSeconds <= 0.25) || !std::isfinite(stepSeconds) ||
        steps == 0u || steps > 100'000u || seed == 0u) {
        return fail("step_seconds must be in (0, 0.25], steps in [1, 100000], and seed nonzero");
    }
    const bool wantsSnapshots = fixture.contains("snapshots");
    std::vector<std::uint64_t> snapshotTicks;
    if (wantsSnapshots) {
        const Json& request = fixture.at("snapshots");
        if (!request.is_object() ||
            (request.contains("ticks") && !request.at("ticks").is_array()))
            return fail("snapshots must be an object with an optional ticks array");
        const std::uint64_t actionCount = fixture.contains("inventory_actions") &&
            fixture.at("inventory_actions").is_array()
            ? fixture.at("inventory_actions").size() : 0u;
        const std::uint64_t lastTick = static_cast<std::uint64_t>(steps) + actionCount;
        if (request.contains("ticks")) for (const Json& tick : request.at("ticks")) {
            if (!tick.is_number_integer() || tick.get<std::int64_t>() < 0 ||
                tick.get<std::uint64_t>() > lastTick)
                return fail("snapshot tick must be an integer in [0, " +
                    std::to_string(lastTick) + "]");
            snapshotTicks.push_back(tick.get<std::uint64_t>());
        }
        std::sort(snapshotTicks.begin(), snapshotTicks.end());
        if (std::adjacent_find(snapshotTicks.begin(), snapshotTicks.end()) != snapshotTicks.end())
            return fail("snapshot ticks must be unique");
    }

    const Json& actorSpec = fixture.at("actor");
    const auto start = vector3(actorSpec.at("position"), "actor.position");
    const auto velocity = vector3(actorSpec.at("desired_velocity"), "actor.desired_velocity");
    odai::bethesda::RuntimeObject actor;
    actor.id = odai::bethesda::ObjectId::persistent(
        odai::bethesda::makeTes3RecordKey("NPC_", actorSpec.at("id").get<std::string>()));
    actor.base = odai::bethesda::makeTes3RecordKey(
        "NPC_", actorSpec.at("base").get<std::string>());
    actor.kind = odai::bethesda::RuntimeObjectKind::Actor;
    actor.persistent = true;
    actor.transform.position = start;
    actor.currentSpace.kind = odai::bethesda::RuntimeSpaceKind::Exterior;
    actor.actorValues.emplace();
    if (actorSpec.contains("inventory")) {
        for (const Json& seed : actorSpec.at("inventory")) {
            const auto key = odai::bethesda::makeTes3RecordKey(
                seed.value("type", "MISC"), seed.at("id").get<std::string>());
            const int count = seed.at("count").get<int>();
            if (count <= 0) return fail("inventory seed count must be positive");
            actor.inventory.push_back({key, count, false});
        }
    }
    const auto actorId = actor.id;

    odai::bethesda::BethesdaSession session;
    odai::bethesda::BethesdaSessionConfig config;
    config.game = odai::importer::bethesda::BethesdaGame::Morrowind;
    config.contentFingerprint = "headless-fixture:" + name;
    config.randomSeed = seed;
    config.playerObject = actorId;
    config.livingWorldEnabled = false;
    std::string error;
    if (!session.configure(std::move(config), error)) return fail(error);
    session.clock() = odai::bethesda::FixedStepClock(
        odai::bethesda::FixedStepConfig{stepSeconds, stepSeconds, 1u});
    if (!session.world().addInitialObject(std::move(actor), error)) return fail(error);
    odai::bethesda::PhysicsCharacterConfig controller;
    controller.position = {static_cast<float>(start[0]), static_cast<float>(start[1]),
        static_cast<float>(start[2])};
    if (!session.registerActorController(actorId, controller, error)) return fail(error);
    odai::bethesda::PhysicsCharacterInput input;
    input.desiredVelocity = {static_cast<float>(velocity[0]), static_cast<float>(velocity[1]),
        static_cast<float>(velocity[2])};
    if (!session.setActorControllerInput(actorId, input)) return fail("cannot set actor movement intent");

    Json checkpoints = Json::array();
    std::size_t nextSnapshotTick = 0u;
    const auto captureRequestedTick = [&]() {
        if (nextSnapshotTick < snapshotTicks.size() &&
            snapshotTicks[nextSnapshotTick] == session.clock().tick()) {
            checkpoints.push_back({{"tick", session.clock().tick()},
                {"state", runtimeSnapshot(session)}});
            ++nextSnapshotTick;
        }
    };
    captureRequestedTick();
    std::size_t worldCommands = 0u;
    for (std::uint32_t i = 0; i < steps; ++i) {
        const auto result = session.advance(stepSeconds);
        worldCommands += result.worldCommands;
        if (result.clock.steps != 1u || result.clock.droppedSteps != 0u) {
            return fail("simulation did not advance exactly one tick at step " + std::to_string(i));
        }
        if (!result.diagnostics.empty()) {
            return fail("simulation step " + std::to_string(i) + ": " + result.diagnostics.front());
        }
        captureRequestedTick();
    }

    Json inventorySteps = Json::array();
    if (fixture.contains("inventory_actions")) {
        const auto snapshot = [&]() {
            Json owned = Json::object();
            const auto* player = session.world().find(actorId);
            for (const auto& entry : player->inventory)
                owned[entry.item.toString()] = entry.count;
            return owned;
        };
        inventorySteps.push_back({{"inventory", snapshot()}, {"world_items", Json::array()}});
        for (const Json& action : fixture.at("inventory_actions")) {
            const auto item = odai::bethesda::makeTes3RecordKey(
                action.value("type", "MISC"), action.at("id").get<std::string>());
            if (!session.dropInventoryItem(actorId, item, error)) return fail(error);
            const auto step = session.advance(stepSeconds);
            if (!step.diagnostics.empty()) return fail(step.diagnostics.front());
            if (step.clock.steps != 1u || step.clock.droppedSteps != 0u)
                return fail("inventory action did not advance exactly one tick");
            captureRequestedTick();
            Json worldItems = Json::array();
            for (const auto& object : session.world().orderedObjects()) {
                if (object.kind != odai::bethesda::RuntimeObjectKind::Item) continue;
                const auto* player = session.world().find(actorId);
                const double dx = object.transform.position[0] - player->transform.position[0];
                const double dy = object.transform.position[1] - player->transform.position[1];
                const double dz = object.transform.position[2] - player->transform.position[2];
                if (dx * dx + dy * dy + dz * dz > 200.0 * 200.0 ||
                    object.currentSpace != player->currentSpace || !object.enabled)
                    return fail("dropped item is not active near the player in the same space");
                worldItems.push_back(object.base.toString());
            }
            inventorySteps.push_back({{"inventory", snapshot()}, {"world_items", worldItems}});
        }
        if (fixture.contains("inventory_expect") &&
            inventorySteps != fixture.at("inventory_expect"))
            return fail("inventory lifecycle differs from fixture expectation");
    }

    const auto* finalActor = session.world().find(actorId);
    if (finalActor == nullptr) return fail("actor disappeared from the runtime world");
    const auto physical = session.physics().characterState(actorId);
    if (!physical.has_value()) return fail("actor physics controller disappeared");
    const auto& position = finalActor->transform.position;
    const bool physicsMatchesWorld =
        std::abs(position[0] - physical->position.x) < 0.0001 &&
        std::abs(position[1] - physical->position.y) < 0.0001 &&
        std::abs(position[2] - physical->position.z) < 0.0001;
    const char* spaceKind = finalActor->currentSpace.kind == odai::bethesda::RuntimeSpaceKind::Exterior
        ? "exterior" : finalActor->currentSpace.kind == odai::bethesda::RuntimeSpaceKind::Interior
        ? "interior" : "unknown";
    Json output{{"scenario", name}, {"ticks", session.clock().tick()},
        {"seed", seed}, {"random_state", session.randomState()},
        {"world_commands", worldCommands},
        {"step_seconds", stepSeconds},
        {"actor", {{"id", actorId.toString()},
                   {"position", position},
                   {"space", {{"kind", spaceKind},
                              {"cell", {finalActor->currentSpace.gridX,
                                        finalActor->currentSpace.gridZ}}}}}},
        {"physics", {{"position", {physical->position.x, physical->position.y,
                                       physical->position.z}},
                     {"matches_world", physicsMatchesWorld}}},
        {"state_hash", std::to_string(session.deterministicHash())}};
    if (fixture.contains("inventory_actions")) output["inventory_steps"] = inventorySteps;
    if (wantsSnapshots)
        output["snapshots"] = {{"version", 1}, {"checkpoints", checkpoints},
            {"final", {{"tick", session.clock().tick()},
                {"state", runtimeSnapshot(session)}}}};
    if (fixture.contains("expect")) {
        const Json& expect = fixture.at("expect");
        const double xMin = expect.at("x_min").get<double>();
        const double xMax = expect.at("x_max").get<double>();
        const double yMax = expect.at("y_max").get<double>();
        const std::size_t minCommands = expect.at("min_world_commands").get<std::size_t>();
        std::string error;
        if (!std::isfinite(xMin) || !std::isfinite(xMax) || !std::isfinite(yMax) ||
            xMin > xMax) {
            error = "invalid expected bounds";
        } else if (position[0] < xMin || position[0] > xMax) {
            error = "actor x=" + std::to_string(position[0]) + " outside expected [" +
                std::to_string(xMin) + ", " + std::to_string(xMax) + "]";
        } else if (position[1] > yMax) {
            error = "actor y=" + std::to_string(position[1]) +
                " exceeds expected maximum " + std::to_string(yMax);
        } else if (worldCommands < minCommands) {
            error = "applied " + std::to_string(worldCommands) +
                " world commands; expected at least " + std::to_string(minCommands);
        } else if (!physicsMatchesWorld) {
            error = "runtime actor transform disagrees with the physics controller";
        }
        if (!error.empty()) {
            output["status"] = "fail";
            output["error"] = error;
            std::cerr << "headless engine: " << error << '\n';
            std::cout << output.dump() << '\n';
            return 1;
        }
        output["status"] = "pass";
    } else {
        output["status"] = "completed";
    }
    std::cout << output.dump() << '\n';
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) return fail("usage: odai_headless <fixture.json>");
    try {
        std::ifstream input(argv[1]);
        if (!input) return fail("cannot open fixture: " + std::string(argv[1]));
        return run(Json::parse(input));
    } catch (const std::exception& exception) {
        return fail(std::string("invalid fixture: ") + exception.what());
    }
}
