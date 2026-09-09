module chunk_push_constants;

public struct ChunkPushConstants
{
    public float4 chunkOffset;
    public float4 cascadeData;
    // [0] = alpha-test threshold in 0..1 for the draw being recorded; the
    // source formats author this per surface and it is NOT always 0.5.
    // [1] > 0.5 = highlight fragments the alpha test would discard.
    // [2] > 0.5 = paint surfaces whose diffuse texture never resolved MAGENTA
    //       instead of their per-model hashed pastel. The pastel is a good
    //       default and a terrible diagnostic: a failed asset lookup comes out
    //       as plausible grey-brown rock, so nothing ever looks wrong.
    // [3] spare.
    public float4 materialParams;
    public float4 rigidAnimationTransform[3];
    // x > 0.5 enables the transform for this draw.
    public float4 rigidAnimationParams;
    public float4 lodTransition;
};

[[vk::push_constant]]
public ConstantBuffer<ChunkPushConstants> chunkPc;

// Stable in world space and identical in depth, colour, reflection and shadow
// passes. Complementary thresholds keep matching surfaces covered exactly once.
public void applyImportedLodTransition(float3 worldPosition)
{
    const float role = chunkPc.lodTransition.y;
    [branch] if (role != 0.0)
    {
        uint3 cell = uint3(int3(floor(worldPosition * 0.25)));
        uint h = cell.x * 1597334677u ^ cell.y * 3812015801u ^ cell.z * 2798796415u;
        h ^= uint(chunkPc.lodTransition.z) * 2654435761u;
        h ^= h >> 16; h *= 2246822519u; h ^= h >> 13;
        const float threshold = float(h & 65535u) / 65536.0;
        const bool incoming = threshold < saturate(chunkPc.lodTransition.x);
        if ((role > 0.0 && !incoming) || (role < 0.0 && incoming)) discard;
    }
}
