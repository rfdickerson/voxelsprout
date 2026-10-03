#include "bethesda/inventory_ui_flow.h"
#include "bethesda/runtime_ids.h"
#include "import/bethesda/asset_source.h"
#include "import/bethesda/runtime_item_scene.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Json = nlohmann::json;
using namespace odai::bethesda;

void fields(const Json& value, const std::string& path,
    std::initializer_list<const char*> required,
    std::initializer_list<const char*> optional) {
    if (!value.is_object()) throw std::runtime_error(path + " must be an object");
    for (const char* key : required)
        if (!value.contains(key)) throw std::runtime_error(path + "." + key + " is required");
    for (auto it = value.begin(); it != value.end(); ++it) {
        const auto known = [&](const char* key) { return it.key() == key; };
        if (!std::any_of(required.begin(), required.end(), known) &&
            !std::any_of(optional.begin(), optional.end(), known))
            throw std::runtime_error(path + "." + it.key() + " is not supported");
    }
}

std::string textField(const Json& value, const std::string& path) {
    if (!value.is_string() || value.get<std::string>().empty())
        throw std::runtime_error(path + " must be a nonempty string");
    return value.get<std::string>();
}

Json capture(const InventoryUiFlow& ui, const BethesdaSession& session) {
    const auto snapshot = ui.snapshot(session, [](const RecordKey& item) {
        return item.textId.empty() ? item.toString() : item.textId;
    });
    Json visible = Json::array();
    for (const auto& entry : snapshot.entries)
        visible.push_back({{"item", entry.item.toString()}, {"label", entry.label},
            {"displayed_quantity", entry.quantity}, {"selected", entry.selected},
            {"enabled", entry.enabled}});
    Json owned = Json::object();
    const RuntimeObject* player = session.world().find(session.playerObject());
    if (player) for (const InventoryEntry& entry : player->inventory)
        owned[entry.item.toString()] = entry.count;
    Json worldItems = Json::array();
    for (const ObjectId& id : session.world().orderedObjectIds()) {
        const RuntimeObject* object = session.world().find(id);
        if (!object || object->kind != RuntimeObjectKind::Item || !object->enabled) continue;
        worldItems.push_back({{"item", object->base.toString()}, {"id", id.toString()},
            {"position", object->transform.position}});
    }
    return {{"active_screen", snapshot.open ? "inventory" : "none"},
        {"selection", snapshot.selection}, {"visible", visible},
        {"inventory", owned}, {"world_items", worldItems},
        {"tick", session.clock().tick()}};
}

void check(const Json& expected, const Json& actual, const std::string& path) {
    fields(expected, path, {}, {"screen", "selected", "visible_items", "labels", "inventory", "world_items"});
    const auto equal = [&](const std::string& field, const Json& wanted, const Json& got) {
        if (wanted != got)
            throw std::runtime_error(path + "." + field + " expected " + wanted.dump() +
                ", actual " + got.dump());
    };
    if (expected.contains("screen"))
        equal("screen", expected.at("screen"), actual.at("active_screen"));
    if (expected.contains("selected")) {
        std::string selected;
        for (const Json& entry : actual.at("visible"))
            if (entry.at("selected") == true) selected = entry.at("item").get<std::string>();
        equal("selected", expected.at("selected"), selected);
    }
    if (expected.contains("visible_items")) {
        Json visible = Json::object();
        for (const Json& entry : actual.at("visible"))
            visible[entry.at("item").get<std::string>()] = entry.at("displayed_quantity");
        equal("visible_items", expected.at("visible_items"), visible);
    }
    if (expected.contains("labels")) {
        Json labels = Json::object();
        for (const Json& entry : actual.at("visible"))
            labels[entry.at("item").get<std::string>()] = entry.at("label");
        equal("labels", expected.at("labels"), labels);
    }
    if (expected.contains("inventory"))
        equal("inventory", expected.at("inventory"), actual.at("inventory"));
    if (expected.contains("world_items")) {
        Json world = Json::array();
        for (const Json& entry : actual.at("world_items")) world.push_back(entry.at("item"));
        equal("world_items", expected.at("world_items"), world);
    }
}

InventoryUiInput parseInputFrame(const Json& frame, const std::string& path) {
    fields(frame, path, {"keys"}, {"expect"});
    if (!frame.at("keys").is_array()) throw std::runtime_error(path + ".keys must be an array");
    InventoryUiInput input;
    std::set<std::string> keys;
    for (const Json& keyValue : frame.at("keys")) {
        const std::string key = textField(keyValue, path + ".keys[]");
        if (!keys.insert(key).second) throw std::runtime_error(path + ".keys duplicate " + key);
        if (key == "I") input.inventory = true;
        else if (key == "Up") input.up = true;
        else if (key == "Down") input.down = true;
        else if (key == "D") input.drop = true;
        else if (key == "Escape") input.cancel = true;
        else throw std::runtime_error(path + ".keys unsupported " + key);
    }
    if (frame.contains("expect"))
        fields(frame.at("expect"), path + ".expect", {},
            {"screen", "selected", "visible_items", "labels", "inventory", "world_items"});
    return input;
}

int run(const Json& fixture) {
    fields(fixture, "$", {"version", "kind", "name", "actor", "frames"}, {"step_seconds", "viewport"});
    if (!fixture.at("version").is_number_integer() || fixture.at("version") != 1)
        throw std::runtime_error("$.version must be 1");
    if (fixture.at("kind") != "ui_inventory")
        throw std::runtime_error("$.kind must be ui_inventory");
    const std::string name = textField(fixture.at("name"), "$.name");
    const double stepSeconds = fixture.value("step_seconds", 1.0 / 60.0);
    if (!std::isfinite(stepSeconds) || stepSeconds <= 0.0 || stepSeconds > 0.1)
        throw std::runtime_error("$.step_seconds must be in (0, 0.1]");
    if (fixture.contains("viewport")) {
        const Json& viewport = fixture.at("viewport");
        if (!viewport.is_array() || viewport.size() != 2 ||
            !viewport.at(0).is_number_integer() || !viewport.at(1).is_number_integer() ||
            viewport.at(0).get<int>() <= 0 || viewport.at(1).get<int>() <= 0)
            throw std::runtime_error("$.viewport must contain two positive integers");
    }
    if (!fixture.at("frames").is_array() || fixture.at("frames").empty())
        throw std::runtime_error("$.frames must be a nonempty array");
    std::vector<InventoryUiInput> inputs;
    inputs.reserve(fixture.at("frames").size());
    for (std::size_t index = 0; index < fixture.at("frames").size(); ++index)
        inputs.push_back(parseInputFrame(fixture.at("frames").at(index),
            "$.frames[" + std::to_string(index) + "]"));
    const Json& actorSpec = fixture.at("actor");
    fields(actorSpec, "$.actor", {"id", "base", "position", "inventory"}, {});
    if (!actorSpec.at("position").is_array() || actorSpec.at("position").size() != 3)
        throw std::runtime_error("$.actor.position must contain three coordinates");
    RuntimeObject actor;
    actor.id = ObjectId::persistent(makeTes3RecordKey("NPC_", textField(actorSpec.at("id"), "$.actor.id")));
    actor.base = makeTes3RecordKey("NPC_", textField(actorSpec.at("base"), "$.actor.base"));
    actor.kind = RuntimeObjectKind::Actor;
    actor.persistent = true;
    actor.currentSpace.kind = RuntimeSpaceKind::Exterior;
    actor.actorValues.emplace();
    for (int i = 0; i < 3; ++i) {
        if (!actorSpec.at("position").at(i).is_number())
            throw std::runtime_error("$.actor.position must contain numbers");
        actor.transform.position[i] = actorSpec.at("position").at(i).get<double>();
        if (!std::isfinite(actor.transform.position[i]))
            throw std::runtime_error("$.actor.position must be finite");
    }
    if (!actorSpec.at("inventory").is_array())
        throw std::runtime_error("$.actor.inventory must be an array");
    std::set<RecordKey> seen;
    for (const Json& item : actorSpec.at("inventory")) {
        fields(item, "$.actor.inventory[]", {"type", "id", "count"}, {});
        const RecordKey key = makeTes3RecordKey(
            textField(item.at("type"), "$.actor.inventory[].type"),
            textField(item.at("id"), "$.actor.inventory[].id"));
        if (!item.at("count").is_number_integer() || item.at("count").get<int>() <= 0 ||
            !seen.insert(key).second)
            throw std::runtime_error("$.actor.inventory[] requires a unique positive count");
        actor.inventory.push_back({key, item.at("count").get<int>(), false});
    }
    const ObjectId playerId = actor.id;
    BethesdaSession session;
    BethesdaSessionConfig config;
    config.game = odai::importer::bethesda::BethesdaGame::Morrowind;
    config.contentFingerprint = "headless-ui:" + name;
    config.playerObject = playerId;
    config.livingWorldEnabled = false;
    std::string error;
    if (!session.configure(std::move(config), error)) throw std::runtime_error(error);
    session.clock() = FixedStepClock(FixedStepConfig{stepSeconds, stepSeconds, 1u});
    if (!session.world().addInitialObject(std::move(actor), error)) throw std::runtime_error(error);

    InventoryUiFlow ui;
    Json checkpoints = Json::array();
    for (std::size_t index = 0; index < inputs.size(); ++index) {
        const Json& frame = fixture.at("frames").at(index);
        const std::string path = "$.frames[" + std::to_string(index) + "]";
        const auto outcome = ui.update(inputs[index], session);
        if (!outcome.error.empty()) throw std::runtime_error(path + ": " + outcome.error);
        if (!ui.isOpen()) {
            const auto step = session.advance(stepSeconds);
            if (step.clock.steps != 1u || !step.diagnostics.empty())
                throw std::runtime_error(path + ": simulation advance failed");
        }
        Json state = capture(ui, session);
        checkpoints.push_back({{"frame", index}, {"state", state}});
        if (frame.contains("expect")) check(frame.at("expect"), state, path + ".expect");
    }
    std::cout << Json{{"status", "pass"}, {"scenario", name},
        {"checkpoints", checkpoints}, {"final", checkpoints.back().at("state")}}.dump() << '\n';
    return 0;
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 4 && std::string(argv[1]) == "--probe-item-scene") {
            odai::importer::bethesda::FalloutAssetSource assets;
            if (!assets.open(argv[2]))
                throw std::runtime_error("cannot open local asset source");
            odai::importer::ImportedScene scene;
            std::string error;
            if (!odai::importer::bethesda::buildRuntimeItemScene(
                    assets, argv[3], "probe", {80.0, 100.0, 0.0}, 1.0f,
                    scene, error)) throw std::runtime_error(error);
            std::cout << Json{{"status", "pass"}, {"model", argv[3]},
                {"meshes", scene.meshes.size()}, {"instances", scene.instances.size()},
                {"vertices", scene.meshes.front().vertices.size()},
                {"position", {scene.instances.front().transform[3],
                    scene.instances.front().transform[7],
                    scene.instances.front().transform[11]}}}.dump() << '\n';
            return 0;
        }
        if (argc != 2) throw std::runtime_error("usage: odai_headless_ui <scenario.json>");
        std::ifstream file(argv[1]);
        if (!file) throw std::runtime_error(std::string("cannot open scenario: ") + argv[1]);
        Json fixture;
        file >> fixture;
        return run(fixture);
    } catch (const std::exception& exception) {
        std::cerr << "headless UI: " << exception.what() << '\n';
        std::cout << Json{{"status", "fail"}, {"error", exception.what()}}.dump() << '\n';
        return 1;
    }
}
