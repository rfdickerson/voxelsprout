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
};

[[vk::push_constant]]
public ConstantBuffer<ChunkPushConstants> chunkPc;
