#pragma once

#include <algorithm>
#include <cmath>

namespace odai::bethesda {

// A deterministic camera authored in the main gate's local frame. The paired
// Tamriel -> WhiterunWorld door supplies the origin and outward-facing yaw, so a
// load-order patch may move the gate without invalidating the composition.
struct WhiterunReferenceCamera {
    float position[3] = {};
    float yawDegrees = 0.0f;
    float pitchDegrees = 0.0f;
    float horizontalFovDegrees = 75.0f;
};

inline WhiterunReferenceCamera whiterunReferenceCamera(
    const float gateArrivalEye[3], float gateOutwardYawDegrees) {
    constexpr float kPi = 3.14159265358979323846f;
    constexpr float kRadiansToDegrees = 180.0f / kPi;
    // The authored arrival is inside the arch. Move east along the gate-local
    // right axis into the market plaza, then slightly inward to clear the
    // masonry return while retaining both foreground braziers.
    constexpr float kCameraInwardOffset = 100.0f;
    constexpr float kCameraArrivalRightOffset = 2384.0f;
    constexpr float kCameraHeightOffset = 45.0f;
    // Aim above and inward of the arch. This leaves the gatehouse and banner
    // in the right half of a 16:9 frame and reserves the left for the smithy.
    constexpr float kLookArrivalRightOffset = 250.0f;
    constexpr float kLookInwardOffset = 440.0f;
    constexpr float kLookHeightOffset = 350.0f;

    // XTEL preserves the target door's outward-facing rotation. The reference
    // camera lives inside the plaza, so its local forward is the opposite.
    const float yaw = (gateOutwardYawDegrees + 180.0f) * (kPi / 180.0f);
    const float inwardX = std::cos(yaw);
    const float inwardZ = std::sin(yaw);
    // XTEL yaw follows Bethesda's door convention; its screen/plaza-right
    // basis is the clockwise perpendicular of the inward vector.
    const float arrivalRightX = inwardZ;
    const float arrivalRightZ = -inwardX;

    WhiterunReferenceCamera camera;
    camera.position[0] = gateArrivalEye[0] + inwardX * kCameraInwardOffset +
        arrivalRightX * kCameraArrivalRightOffset;
    camera.position[1] = gateArrivalEye[1] + kCameraHeightOffset;
    camera.position[2] = gateArrivalEye[2] + inwardZ * kCameraInwardOffset +
        arrivalRightZ * kCameraArrivalRightOffset;

    const float targetX = gateArrivalEye[0] + arrivalRightX * kLookArrivalRightOffset +
        inwardX * kLookInwardOffset;
    const float targetY = gateArrivalEye[1] + kLookHeightOffset;
    const float targetZ = gateArrivalEye[2] + arrivalRightZ * kLookArrivalRightOffset +
        inwardZ * kLookInwardOffset;
    const float dx = targetX - camera.position[0];
    const float dy = targetY - camera.position[1];
    const float dz = targetZ - camera.position[2];
    const float horizontal = std::max(std::sqrt(dx * dx + dz * dz), 0.001f);
    camera.yawDegrees = std::atan2(dz, dx) * kRadiansToDegrees;
    camera.pitchDegrees = std::atan2(dy, horizontal) * kRadiansToDegrees;
    return camera;
}

// A second authored-gate-relative composition looking inward from the bridge
// toward the market district.  Keeping this in the paired door's frame makes
// the shot follow load-order patches that relocate the Whiterun entrance.
inline WhiterunReferenceCamera whiterunMarketReferenceCamera(
    const float gateArrivalEye[3], float gateOutwardYawDegrees) {
    constexpr float kPi = 3.14159265358979323846f;
    constexpr float kRadiansToDegrees = 180.0f / kPi;

    const float inwardYaw =
        (gateOutwardYawDegrees + 180.0f) * (kPi / 180.0f);
    const float inwardX = std::cos(inwardYaw);
    const float inwardZ = std::sin(inwardYaw);
    const float marketRoadX = inwardZ;
    const float marketRoadZ = -inwardX;

    // Sit on the gate bridge behind the first market sign/stall rather than at
    // the door's vertically stacked NAVM. This is the authored position that
    // retains the road, foreground brazier, timber façades and mountain vista.
    constexpr float kCameraInwardOffset = 100.0f;
    constexpr float kCameraRoadOffset = 484.0f;
    constexpr float kCameraHeightOffset = 45.0f;
    constexpr float kLookDistance = 5000.0f;
    constexpr float kLookHeightOffset = 175.0f;

    WhiterunReferenceCamera camera;
    camera.position[0] = gateArrivalEye[0] +
        inwardX * kCameraInwardOffset + marketRoadX * kCameraRoadOffset;
    camera.position[1] = gateArrivalEye[1] + kCameraHeightOffset;
    camera.position[2] = gateArrivalEye[2] +
        inwardZ * kCameraInwardOffset + marketRoadZ * kCameraRoadOffset;

    const float targetX = camera.position[0] + marketRoadX * kLookDistance;
    const float targetY = camera.position[1] + kLookHeightOffset;
    const float targetZ = camera.position[2] + marketRoadZ * kLookDistance;
    const float dx = targetX - camera.position[0];
    const float dy = targetY - camera.position[1];
    const float dz = targetZ - camera.position[2];
    const float horizontal = std::max(std::sqrt(dx * dx + dz * dz), 0.001f);
    camera.yawDegrees = std::atan2(dz, dx) * kRadiansToDegrees;
    camera.pitchDegrees = std::atan2(dy, horizontal) * kRadiansToDegrees;
    return camera;
}

}  // namespace odai::bethesda
