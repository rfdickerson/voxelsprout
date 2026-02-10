module voxel_decode;

public static const uint kShiftX = 0u;
public static const uint kShiftY = 5u;
public static const uint kShiftZ = 10u;
public static const uint kShiftFace = 15u;
public static const uint kShiftCorner = 18u;
public static const uint kShiftAo = 20u;
public static const uint kShiftMaterial = 22u;

public float3 cornerOffset(uint face, uint corner)
{
    if (face == 0u)
    {
        if (corner == 0u) return float3(1.0, 0.0, 0.0);
        if (corner == 1u) return float3(1.0, 1.0, 0.0);
        if (corner == 2u) return float3(1.0, 1.0, 1.0);
        return float3(1.0, 0.0, 1.0);
    }
    if (face == 1u)
    {
        if (corner == 0u) return float3(0.0, 0.0, 1.0);
        if (corner == 1u) return float3(0.0, 1.0, 1.0);
        if (corner == 2u) return float3(0.0, 1.0, 0.0);
        return float3(0.0, 0.0, 0.0);
    }
    if (face == 2u)
    {
        if (corner == 0u) return float3(0.0, 1.0, 0.0);
        if (corner == 1u) return float3(0.0, 1.0, 1.0);
        if (corner == 2u) return float3(1.0, 1.0, 1.0);
        return float3(1.0, 1.0, 0.0);
    }
    if (face == 3u)
    {
        if (corner == 0u) return float3(0.0, 0.0, 1.0);
        if (corner == 1u) return float3(0.0, 0.0, 0.0);
        if (corner == 2u) return float3(1.0, 0.0, 0.0);
        return float3(1.0, 0.0, 1.0);
    }
    if (face == 4u)
    {
        if (corner == 0u) return float3(1.0, 0.0, 1.0);
        if (corner == 1u) return float3(1.0, 1.0, 1.0);
        if (corner == 2u) return float3(0.0, 1.0, 1.0);
        return float3(0.0, 0.0, 1.0);
    }

    if (corner == 0u) return float3(0.0, 0.0, 0.0);
    if (corner == 1u) return float3(0.0, 1.0, 0.0);
    if (corner == 2u) return float3(1.0, 1.0, 0.0);
    return float3(1.0, 0.0, 0.0);
}
