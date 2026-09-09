// Shared by main/depth/normal and shadow vertices. Root-preserving distance
// collapse avoids alpha disagreement between passes and needs no history buffer.
float3 fadeImportedGrass(float3 position, uint flags, uint payload, float3 eye)
{
    if ((flags & ((1u << 12u) | (1u << 2u))) == (1u << 12u))
    {
        const float rootHeight = float(payload >> 16u) * 0.25;
        const float fade = smoothstep(2800.0, 4000.0, distance(position.xz, eye.xz));
        position.y -= rootHeight * fade;
    }
    return position;
}
