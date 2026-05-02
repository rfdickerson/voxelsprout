module voxel_decode;

public static const uint kShiftX = 0u;
public static const uint kShiftY = 5u;
public static const uint kShiftZ = 10u;
public static const uint kShiftFace = 15u;
public static const uint kShiftCorner = 18u;
public static const uint kShiftAo = 20u;
public static const uint kShiftMaterial = 24u;
public static const uint kShiftBaseColor = 27u;
public static const uint kShiftLodLevel = 30u;
public static const uint kMaterialStone = 1u;
public static const uint kMaterialDirt = 2u;
public static const uint kMaterialGrass = 3u;

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

public float3 voxelFaceNormal(uint face)
{
    if (face == 0u) return float3(1.0, 0.0, 0.0);
    if (face == 1u) return float3(-1.0, 0.0, 0.0);
    if (face == 2u) return float3(0.0, 1.0, 0.0);
    if (face == 3u) return float3(0.0, -1.0, 0.0);
    if (face == 4u) return float3(0.0, 0.0, 1.0);
    return float3(0.0, 0.0, -1.0);
}

public bool isTerrainMaterialForMicroDisplacement(uint material)
{
    return material == kMaterialStone || material == kMaterialDirt || material == kMaterialGrass;
}

public float terrainMicroDisplacementSignal(float3 worldPosition)
{
    return
        sin(dot(worldPosition, float3(5.11, 1.37, -3.73))) * 0.46 +
        sin(dot(worldPosition, float3(-2.67, 4.29, 6.13))) * 0.34 +
        sin(dot(worldPosition, float3(8.91, -1.18, 2.45))) * 0.20;
}

public float terrainMicroDisplacementAmount(uint material, uint face, float3 worldPosition)
{
    if (!isTerrainMaterialForMicroDisplacement(material) || face == 3u)
    {
        return 0.0;
    }

    const float slope = 1.0 - abs(voxelFaceNormal(face).y);
    float strength = 0.012;
    if (material == kMaterialStone) strength = lerp(0.018, 0.034, slope);
    if (material == kMaterialDirt) strength = lerp(0.010, 0.020, slope);
    if (material == kMaterialGrass) strength = 0.008;

    return terrainMicroDisplacementSignal(worldPosition * 0.55) * strength;
}

public float3 displacedVoxelWorldPosition(uint material, uint face, float3 worldPosition)
{
    return worldPosition + (voxelFaceNormal(face) * terrainMicroDisplacementAmount(material, face, worldPosition));
}
