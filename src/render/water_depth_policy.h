#pragma once

// Shared by the water shader and CPU regression fixtures. Missing opaque depth
// is deep/unknown water, not a shoreline; foreground depth is dry geometry.
inline float waterShoreCoverage(float waterDepth, float opaqueDepth) {
    if (!(opaqueDepth > 0.0001f)) return 1.0f;
    const float width = waterDepth * 0.001f > 2.0f ? waterDepth * 0.001f : 2.0f;
    const float distance = opaqueDepth - waterDepth;
    if (distance <= 0.0f) return 0.0f;
    if (distance >= width) return 1.0f;
    const float t = distance / width;
    return t * t * (3.0f - 2.0f * t);
}
