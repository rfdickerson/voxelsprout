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

public float4x4 inverse4x4(float4x4 m)
{
    const float a00 = m[0][0], a01 = m[0][1], a02 = m[0][2], a03 = m[0][3];
    const float a10 = m[1][0], a11 = m[1][1], a12 = m[1][2], a13 = m[1][3];
    const float a20 = m[2][0], a21 = m[2][1], a22 = m[2][2], a23 = m[2][3];
    const float a30 = m[3][0], a31 = m[3][1], a32 = m[3][2], a33 = m[3][3];

    const float b00 = a00 * a11 - a01 * a10;
    const float b01 = a00 * a12 - a02 * a10;
    const float b02 = a00 * a13 - a03 * a10;
    const float b03 = a01 * a12 - a02 * a11;
    const float b04 = a01 * a13 - a03 * a11;
    const float b05 = a02 * a13 - a03 * a12;
    const float b06 = a20 * a31 - a21 * a30;
    const float b07 = a20 * a32 - a22 * a30;
    const float b08 = a20 * a33 - a23 * a30;
    const float b09 = a21 * a32 - a22 * a31;
    const float b10 = a21 * a33 - a23 * a31;
    const float b11 = a22 * a33 - a23 * a32;

    const float det =
        (b00 * b11) - (b01 * b10) + (b02 * b09) +
        (b03 * b08) - (b04 * b07) + (b05 * b06);
    const float invDet = 1.0 / max(abs(det), 1e-8) * ((det < 0.0) ? -1.0 : 1.0);

    float4x4 inv;
    inv[0][0] = (a11 * b11 - a12 * b10 + a13 * b09) * invDet;
    inv[0][1] = (a02 * b10 - a01 * b11 - a03 * b09) * invDet;
    inv[0][2] = (a31 * b05 - a32 * b04 + a33 * b03) * invDet;
    inv[0][3] = (a22 * b04 - a21 * b05 - a23 * b03) * invDet;
    inv[1][0] = (a12 * b08 - a10 * b11 - a13 * b07) * invDet;
    inv[1][1] = (a00 * b11 - a02 * b08 + a03 * b07) * invDet;
    inv[1][2] = (a32 * b02 - a30 * b05 - a33 * b01) * invDet;
    inv[1][3] = (a20 * b05 - a22 * b02 + a23 * b01) * invDet;
    inv[2][0] = (a10 * b10 - a11 * b08 + a13 * b06) * invDet;
    inv[2][1] = (a01 * b08 - a00 * b10 - a03 * b06) * invDet;
    inv[2][2] = (a30 * b04 - a31 * b02 + a33 * b00) * invDet;
    inv[2][3] = (a21 * b02 - a20 * b04 - a23 * b00) * invDet;
    inv[3][0] = (a11 * b07 - a10 * b09 - a12 * b06) * invDet;
    inv[3][1] = (a00 * b09 - a01 * b07 + a02 * b06) * invDet;
    inv[3][2] = (a31 * b01 - a30 * b03 - a32 * b00) * invDet;
    inv[3][3] = (a20 * b03 - a21 * b01 + a22 * b00) * invDet;
    return inv;
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
