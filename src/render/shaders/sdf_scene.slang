module sdf_scene;

import camera_uniform;

public static const float kSdfMaxDistance = 220.0;
public static const int kSdfMaxSteps = 128;
public static const float kSdfHitEpsilon = 0.0015;

public float sdfBlendFromTime(float timeSeconds)
{
    return (sin(timeSeconds * 0.90) * 0.5) + 0.5;
}

float sdfSphere(float3 p, float radius)
{
    return length(p) - radius;
}

float sdfBox(float3 p, float3 halfExtents)
{
    const float3 q = abs(p) - halfExtents;
    return length(max(q, float3(0.0, 0.0, 0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

float smoothMin(float a, float b, float k)
{
    const float h = saturate(0.5 + (0.5 * (b - a) / max(k, 1e-4)));
    return lerp(b, a, h) - (k * h * (1.0 - h));
}

public float3 sdfObjectCenter()
{
    return float3(0.0, 5.0, 0.0);
}

public float sdfSceneDistance(float3 worldPosition, float3 objectCenter, float blend)
{
    const float3 cubeCenter = objectCenter + float3(0.0, 1.35, 0.0);
    const float sphereTravel = lerp(2.9, 0.0, blend);
    const float3 sphereCenter = cubeCenter + float3(0.0, sphereTravel, 0.0);

    const float sphereDistance = sdfSphere(worldPosition - sphereCenter, 1.30);
    const float boxDistance = sdfBox(worldPosition - cubeCenter, float3(1.25, 1.25, 1.25));
    return smoothMin(sphereDistance, boxDistance, 0.34);
}

public float3 sdfSceneNormal(float3 worldPosition, float3 objectCenter, float blend)
{
    const float e = 0.0025;
    const float3 ex = float3(e, 0.0, 0.0);
    const float3 ey = float3(0.0, e, 0.0);
    const float3 ez = float3(0.0, 0.0, e);
    const float dx =
        sdfSceneDistance(worldPosition + ex, objectCenter, blend) - sdfSceneDistance(worldPosition - ex, objectCenter, blend);
    const float dy =
        sdfSceneDistance(worldPosition + ey, objectCenter, blend) - sdfSceneDistance(worldPosition - ey, objectCenter, blend);
    const float dz =
        sdfSceneDistance(worldPosition + ez, objectCenter, blend) - sdfSceneDistance(worldPosition - ez, objectCenter, blend);
    return normalize(float3(dx, dy, dz));
}

public float3 reconstructViewRay(float2 uv)
{
    const float2 ndc = (uv * 2.0) - 1.0;
    float projX = camera.proj[0][0];
    float projY = camera.proj[1][1];
    if (abs(projX) < 1e-6) projX = (projX < 0.0) ? -1e-6 : 1e-6;
    if (abs(projY) < 1e-6) projY = (projY < 0.0) ? -1e-6 : 1e-6;
    return normalize(float3(ndc.x / projX, ndc.y / projY, -1.0));
}

public float3 viewToWorldDirection(float3 viewDirection)
{
    const float3x3 viewRotation = (float3x3)camera.view;
    return normalize(mul(transpose(viewRotation), viewDirection));
}

public bool raymarchScene(
    float3 rayOrigin,
    float3 rayDirection,
    float tMin,
    float tMax,
    float3 objectCenter,
    float blend,
    out float tHit,
    out float3 hitWorld
)
{
    tHit = tMin;
    hitWorld = rayOrigin;
    for (int i = 0; i < kSdfMaxSteps; ++i)
    {
        if (tHit > tMax || tHit > kSdfMaxDistance) return false;
        hitWorld = rayOrigin + (rayDirection * tHit);
        const float d = sdfSceneDistance(hitWorld, objectCenter, blend);
        if (d < kSdfHitEpsilon) return true;
        tHit += max(d * 0.85, 0.008);
    }
    return false;
}
