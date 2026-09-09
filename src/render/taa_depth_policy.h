#pragma once

// Shared by the Slang resolve and CPU regression tests. Projection coefficients
// are row 2/3's Z/W entries; works for perspective and orthographic cameras.
inline float taaViewDepth(float ndcDepth, float p22, float p23, float p32, float p33) {
    const float denominator = p22 - ndcDepth * p32;
    if (denominator > -1e-12f && denominator < 1e-12f) return 0.0f;
    return (p23 - ndcDepth * p33) / denominator;
}

inline bool taaDepthMatches(float historyDepth, float expectedDepth) {
    // Positive, bounded comparisons also reject NaNs. Two percent accommodates
    // FP16 logarithmic history depth and small subpixel surface differences.
    if (!(historyDepth > 0.0f && expectedDepth > 0.0f &&
          historyDepth < 1e12f && expectedDepth < 1e12f)) return false;
    const float tolerance = 1.0f + 0.02f * expectedDepth;
    const float difference = historyDepth - expectedDepth;
    return difference >= -tolerance && difference <= tolerance;
}

// Valid taps have already been depth-tested and normalized. A silhouette may
// leave only one bilinear tap; its footprint is not a measure of history age.
// Fade only very small footprints, where normalization amplifies sampling error.
inline float taaHistoryConfidence(float validFootprint) {
    if (!(validFootprint > 0.0f)) return 0.0f;
    return validFootprint >= 0.25f ? 1.0f : validFootprint * 4.0f;
}
