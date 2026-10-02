#include "bethesda/bethesda_session.h"
#include "bethesda/runtime_ids.h"

#include <nlohmann/json.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

using Json = nlohmann::json;

int fail(const std::string& message) {
    std::cerr << "headless engine: " << message << '\n';
    std::cout << Json{{"status", "fail"}, {"error", message}}.dump() << '\n';
    return 1;
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

int run(const Json& fixture) {
    const std::string name = fixture.at("name").get<std::string>();
    const double stepSeconds = fixture.at("step_seconds").get<double>();
    const std::uint32_t steps = fixture.at("steps").get<std::uint32_t>();
    const std::uint32_t seed = fixture.at("seed").get<std::uint32_t>();
    if (!(stepSeconds > 0.0 && stepSeconds <= 0.25) || !std::isfinite(stepSeconds) ||
        steps == 0u || steps > 100'000u || seed == 0u) {
        return fail("step_seconds must be in (0, 0.25], steps in [1, 100000], and seed nonzero");
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
