#include "app/morrowind_actor_ai.h"

#include <iostream>
#include <vector>

namespace {

int g_failures = 0;

void expectTrue(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "[actor ai test] FAIL: " << message << '\n';
        ++g_failures;
    }
}

void testScheduleStateParsing() {
    expectTrue(
        odai::app::parseMorrowindActorScheduleState("wander") ==
            odai::app::MorrowindActorScheduleState::Wander,
        "Parses wander schedule state");
    expectTrue(
        odai::app::parseMorrowindActorScheduleState("Travel") ==
            odai::app::MorrowindActorScheduleState::Travel,
        "Parses case-insensitive travel schedule state");
    expectTrue(
        odai::app::morrowindActorScheduleUsesPath(odai::app::MorrowindActorScheduleState::Travel),
        "Travel uses pathing");
    expectTrue(
        !odai::app::morrowindActorScheduleUsesPath(odai::app::MorrowindActorScheduleState::Idle),
        "Idle does not use pathing");
    expectTrue(odai::app::morrowindSeydaNeenIsNightHour(23.0f), "23:00 is night");
    expectTrue(!odai::app::morrowindSeydaNeenIsNightHour(12.0f), "12:00 is day");
}

void testPathCacheKeyBuckets() {
    const std::string a = odai::app::makeMorrowindActorPathCacheKey({10.0f, 0.0f, 20.0f}, "dock", 100.0f);
    const std::string b = odai::app::makeMorrowindActorPathCacheKey({40.0f, 0.0f, 80.0f}, "dock", 100.0f);
    const std::string c = odai::app::makeMorrowindActorPathCacheKey({140.0f, 0.0f, 80.0f}, "dock", 100.0f);
    expectTrue(a == b, "Nearby starts share path cache bucket");
    expectTrue(a != c, "Distant starts use different path cache bucket");
}

void testEndpointPathCacheKeyBuckets() {
    const std::string a = odai::app::makeMorrowindActorEndpointPathCacheKey(
        "schedule:fargoth_home",
        {10.0f, 0.0f, 20.0f},
        "target:fargoth_home",
        {420.0f, 0.0f, 900.0f},
        512.0f);
    const std::string b = odai::app::makeMorrowindActorEndpointPathCacheKey(
        "schedule:fargoth_home",
        {130.0f, 0.0f, 300.0f},
        "target:fargoth_home",
        {480.0f, 0.0f, 980.0f},
        512.0f);
    const std::string c = odai::app::makeMorrowindActorEndpointPathCacheKey(
        "schedule:fargoth_home",
        {10.0f, 0.0f, 20.0f},
        "target:dock",
        {480.0f, 0.0f, 980.0f},
        512.0f);
    expectTrue(a == b, "Endpoint cache key reuses nearby schedule endpoints");
    expectTrue(a != c, "Endpoint cache key separates different target anchors");
    expectTrue(!odai::app::isMorrowindActorPathCacheExpired(90.0f, 90.0f), "Cache survives exact lifetime");
    expectTrue(odai::app::isMorrowindActorPathCacheExpired(90.1f, 90.0f), "Cache expires after lifetime");
}

void testAvoidanceDirection() {
    const std::vector<odai::math::Vector3> positions = {
        {0.0f, 0.0f, 0.0f},
        {20.0f, 0.0f, 0.0f}
    };
    const odai::math::Vector3 direction = odai::app::computeMorrowindActorAvoidanceDirection(
        odai::app::MorrowindActorAvoidanceInput{
            positions[0],
            {0.0f, 0.0f, 1.0f},
            0u,
            80.0f,
            1.0f
        },
        positions);
    expectTrue(direction.x < 0.0f, "Avoidance steers away from nearby actor");
    expectTrue(direction.z > 0.0f, "Avoidance preserves forward progress");
}

} // namespace

int main() {
    testScheduleStateParsing();
    testPathCacheKeyBuckets();
    testEndpointPathCacheKeyBuckets();
    testAvoidanceDirection();

    if (g_failures != 0) {
        std::cerr << "[actor ai test] " << g_failures << " failures\n";
        return 1;
    }
    std::cout << "[actor ai test] all checks passed\n";
    return 0;
}
