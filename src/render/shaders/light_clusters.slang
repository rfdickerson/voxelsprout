module light_clusters;

import camera_uniform;

// Clustered (Forward+) local-light culling: the shared half.
//
// WHAT THIS REPLACES. Every fragment used to walk the whole uploaded light
// array and reject by distance. That is fine for a handful of lights and
// quadratic misery for a lit interior: the loop's cost is paid per pixel
// whether or not a light reaches it, and on a tile-less iGPU the dynamic loop
// also costs occupancy in the uber-shader, so the price is not merely the
// arithmetic.
//
// THE GRID. Screen is divided into kLightClusterTileSize-pixel tiles and view
// depth into kLightClusterSliceCount slices, giving a 3D grid of clusters. A
// compute pass tests every light against every cluster once per frame and
// writes a BITMASK per cluster; the fragment shader finds its own cluster from
// SV_Position and its view depth, reads two words, and iterates only the bits
// that are set.
//
// A MASK, NOT AN INDEX LIST. The usual Forward+ structure is a compacted index
// list plus per-cluster (offset, count), which needs an atomic allocator, a
// worst-case size nobody can bound, and an overflow policy. This renderer
// already caps local lights at kImportedLocalLightCapacity = 64, and 64 bits is
// exactly two words -- so the whole light set fits in a fixed-size mask with no
// atomics, no allocation and no overflow case to get wrong. If that cap ever
// rises past 64 this has to become a list; the static_assert in
// renderer_shared.h is what will say so.
//
// THE SLICING IS EXPONENTIAL, not linear. Linear slices put almost every
// cluster beyond the geometry and leave the near field -- where lights actually
// overlap -- in one or two slices. The standard mapping is
// slice = log(z) * scale + bias, which makes each slice a constant RATIO of
// depth and so a constant fraction of the projected screen area.

// Mirrored in src/render/renderer_shared.h. Change both together.
public static const uint kLightClusterTileSize = 64u;
public static const uint kLightClusterSliceCount = 24u;

// Which slice a view depth falls in. `viewDepth` is positive distance along the
// view axis, the same quantity the normal-depth target stores.
//
// Depths in front of the near plane clamp to slice 0 rather than producing a
// negative index: a fragment cannot be nearer than the near plane, but a
// floating-point epsilon at exactly the plane can make the log go negative.
public uint lightClusterSlice(float viewDepth, float sliceScale, float sliceBias)
{
    const float slice = (log2(max(viewDepth, 1e-4)) * sliceScale) + sliceBias;
    return uint(clamp(slice, 0.0, float(kLightClusterSliceCount - 1u)));
}

// Flat cluster index from a tile coordinate and a slice. Slice-major so that a
// compute thread walking consecutive indices walks one depth slice at a time,
// which is the order the AABB math is coherent in.
public uint lightClusterIndex(uint2 tile, uint slice, uint2 gridDim)
{
    const uint2 clampedTile = min(tile, gridDim - uint2(1u, 1u));
    return (slice * gridDim.x * gridDim.y) + (clampedTile.y * gridDim.x) + clampedTile.x;
}

// The cluster a fragment belongs to, from its SV_Position and view depth.
public uint lightClusterIndexForFragment(float2 fragCoord, float viewDepth)
{
    const uint2 gridDim = uint2(
        uint(camera.lightClusterConfig0.x + 0.5),
        uint(camera.lightClusterConfig0.y + 0.5));
    const uint tileSize = max(uint(camera.lightClusterConfig0.w + 0.5), 1u);
    const uint2 tile = uint2(fragCoord) / tileSize;
    const uint slice = lightClusterSlice(
        viewDepth, camera.lightClusterConfig1.x, camera.lightClusterConfig1.y);
    return lightClusterIndex(tile, slice, gridDim);
}

// True when the cluster grid is live. Zero grid dimensions mean the pass did
// not run this frame (no lights, or resources unavailable), and every consumer
// must fall back to walking the full light array rather than reading a stale or
// unwritten mask.
public bool lightClustersActive()
{
    return camera.lightClusterConfig0.x >= 0.5 && camera.lightClusterConfig0.z >= 0.5;
}
