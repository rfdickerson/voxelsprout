module chunk_push_constants;

public struct ChunkPushConstants
{
    public float4 chunkOffset;
    public float4 cascadeData;
};

[[vk::push_constant]]
public ConstantBuffer<ChunkPushConstants> chunkPc;
