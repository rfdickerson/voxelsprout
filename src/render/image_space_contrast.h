#pragma once

// Display-space contrast with a continuous toe/shoulder. Retains the authored
// slope at middle grey without the linear operator's negative shadow values.
// This is our HDR transfer mapping, not a recovered retail shader.
inline float imageSpaceContrast(float value, float contrast) {
    if (contrast <= 0.0f) return 0.5f;
    if (value <= 0.0f) return 0.0f;
    if (value >= 1.0f) return 1.0f;
    const float x = value <= 0.5f ? value : 1.0f - value;
    const float toe = x / (contrast + (1.0f - contrast) * 2.0f * x);
    return value <= 0.5f ? toe : 1.0f - toe;
}
