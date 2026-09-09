#include "render/taa_depth_policy.h"
#include <cassert>
#include <cmath>
#include <limits>
#include <initializer_list>

int main() {
    // A single valid bilinear tap must retain accumulation at a leaf silhouette.
    // No valid taps must still reject history, and tiny slivers fade safely.
    assert(taaHistoryConfidence(0.25f) == 1.0f);
    assert(taaHistoryConfidence(1.0f) == 1.0f);
    assert(taaHistoryConfidence(0.0f) == 0.0f);
    assert(taaHistoryConfidence(0.125f) == 0.5f);
    assert(taaHistoryConfidence(std::numeric_limits<float>::quiet_NaN()) == 0.0f);
    // Infinite reverse-Z perspective projection: clip z = near, clip w = -view z.
    assert(std::abs(taaViewDepth(0.001f, 0, 1, -1, 0) - 1000.0f) < 0.01f);
    // Orthographic projection must use z, not clip.w (which is always one).
    assert(std::abs(taaViewDepth(0.75f, 0.001f, 1, 0, 1) - 250.0f) < 0.01f);
    // Camera/actor motion: compare with the predicted PREVIOUS surface depth,
    // not the current distance (the surface may have moved toward the camera).
    const float previous = taaViewDepth(1.0f / 900.0f, 0, 1, -1, 0);
    assert(taaDepthMatches(900, previous));
    assert(!taaDepthMatches(900, 500));
    // Newly revealed background must reject the old foreground, and vice versa.
    assert(!taaDepthMatches(100, 1000));
    assert(!taaDepthMatches(1000, 100));
    assert(!taaDepthMatches(0, 1000));
    assert(!taaDepthMatches(1000, 0));
    assert(!taaDepthMatches(std::numeric_limits<float>::quiet_NaN(), 1000));
    assert(!taaDepthMatches(1000, std::numeric_limits<float>::infinity()));
    assert(taaViewDepth(0, 0, 1, 0, 1) == 0);
    // Log-depth quantization must preserve history even beyond FP16's maximum
    // linear depth. Around log2(500001), FP16 steps are 1/64.
    for (float depth : {100.0f, 1000.0f, 50000.0f, 500000.0f}) {
        const float stored = std::round(std::log2(1.0f + depth) * 64.0f) / 64.0f;
        assert(taaDepthMatches(std::exp2(stored) - 1.0f, depth));
    }
}
