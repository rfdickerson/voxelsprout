#pragma once

namespace odai::bethesda {

// Fixed in the Imperial City Market District's authored coordinate system.
// Oblivion's child-city door links do not carry a stable district-local frame,
// so the presentation is tied to the retail worldspace coordinates instead of
// pretending an unrelated interior doorstep is an authoritative camera.
struct OblivionReferenceCamera {
    float position[3] = {};
    float yawDegrees = 0.0f;
    float pitchDegrees = 0.0f;
    float horizontalFovDegrees = 72.0f;
};

inline constexpr OblivionReferenceCamera imperialMarketReferenceCamera() {
    // The camera stands low on the open market walk and uses the paving and
    // pool curb as leading lines.  The statue is deliberately off-centre and
    // the near arcade frames the left edge; looking slightly down also keeps
    // the remote battlement walk from reading as a disconnected wall section
    // above the pavilion roof.
    return {{34500.0f, 3825.0f, -67500.0f}, -121.0f, -2.0f, 70.0f};
}

inline constexpr OblivionReferenceCamera anvilHarborReferenceCamera() {
    // Stand on the authored boardwalk just outside the Flowing Bowl.  The sign
    // placement is only ~365 units away and ~150 units above the eye, which is
    // what puts it in the upper-left instead of below the horizon.  Looking
    // along the quay keeps the castle as the vanishing point while the retail
    // ship rigging frames the right edge.
    return {{-193700.0f, 420.0f, 32500.0f}, 45.0f, -8.5f, 72.0f};
}

inline constexpr OblivionReferenceCamera greatForestReferenceCamera(
    const float weynonPrioryMarker[3]) {
    // Approach Weynon Priory through the surrounding Great Forest rather than
    // centring its buildings. The offset puts the authored road and low
    // foliage in the foreground while the priory clearing terminates the view;
    // the high tree canopy frames both sides. Marker coordinates are already
    // in engine space (X, height, -Y).
    return {{weynonPrioryMarker[0] - 2200.0f,
             weynonPrioryMarker[1] + 230.0f,
             weynonPrioryMarker[2] - 1700.0f},
            37.70f, -3.0f, 70.0f};
}

}  // namespace odai::bethesda
