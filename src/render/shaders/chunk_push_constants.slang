module chunk_push_constants;

public struct ChunkPushConstants
{
    public float4 chunkOffset;
    public float4 cascadeData;
    // [0] = alpha-test threshold in 0..1 for the draw being recorded; the
    // source formats author this per surface and it is NOT always 0.5.
    // [1..3] spare.
    public float4 materialParams;
};

[[vk::push_constant]]
public ConstantBuffer<ChunkPushConstants> chunkPc;
