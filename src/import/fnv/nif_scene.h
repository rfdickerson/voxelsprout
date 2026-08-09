#pragma once

// Minimal reader for Gamebryo NIF files as produced for Fallout 3 / Fallout:
// New Vegas static meshes (header version 20.2.0.7). Extracts only what a
// static-mesh cooker needs: the NiNode transform hierarchy and NiTriShape /
// NiTriShapeData triangle geometry (positions, normals, indices), plus
// materials/textures through BSShaderTextureSet, NiTexturingProperty and
// BSShaderNoLightingProperty.
//
// Skinned meshes and skeletons ARE resolved now -- see parseNifSkeleton and
// parseNifSkinnedMesh in the second half of this header. The claim that "skin
// instance refs are read but not resolved" stood here long after it stopped
// being true.
//
// What is still not read: animation. Controllers and interpolators are skipped
// wherever they appear, because FNV keeps its animation in separate .kf files
// keyed to the bone names this parser does return.
//
// Design note on robustness: NIF 20.x block headers give an explicit byte
// size per block, so a block whose internal fields this parser doesn't fully
// understand can always be *skipped* correctly by jumping to its declared
// end offset — this parser leans on that heavily. Every block type reads
// only the leading fields it actually needs, then unconditionally resumes
// at the header-declared next-block offset, so a wrong guess about a
// trailing field's layout cannot desynchronize the rest of the file. Where a
// block's own internal optional-field layout is genuinely uncertain (see
// NiTriShapeData in the .cc), the parser self-checks that its byte
// consumption lands within the block's declared size and drops that single
// shape's geometry (not the whole file) if it doesn't.
//
// Originally written against public Gamebryo/NIF format documentation with no
// Data Files to check it against; that caveat is obsolete. Every layout here
// has since been verified on retail archives with odai_newvegas_probe, and
// several of the documented layouts turned out to be wrong -- see the
// per-reader comments, which record what was measured rather than what was
// documented.

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace odai::importer::fnv {

struct NifShape {
    std::string name;               // from the NIF string table; may be empty
    std::vector<float> positions;   // xyz per vertex, world space (parent transforms applied)
    std::vector<float> normals;     // xyz per vertex, world space; empty if the source had none
    std::vector<float> uvs;         // uv per vertex (set 0); empty if the source had none
    std::vector<std::uint32_t> triangleIndices;  // 3 per triangle, indexes into positions/normals
    // Diffuse texture path as stored in the NIF, relative to Data\textures
    // and backslash-separated. Empty when the shape has no resolvable
    // BSShaderTextureSet.
    std::string diffuseTexturePath;
    // NiAlphaProperty declared alpha testing (flag 0x200) for this shape.
    bool alphaTest = false;
    // NiAlphaProperty's threshold byte, the value alphaTest compares against.
    // 128 is the neutral default (the 0.5 every renderer hardcodes) and is what
    // a shape with no alpha property keeps.
    std::uint8_t alphaThreshold = 128;
    // NiStencilProperty draw mode DRAW_BOTH. Fallout marks window glass,
    // foliage cards and awnings this way; drawn single-sided they lose
    // whichever face points away from the camera.
    bool twoSided = false;
    // NiAlphaProperty's blend bit. The imported static path draws opaque only,
    // so a blended shape (glass, an additive effect billboard) rendered through
    // it appears as a solid slab -- Goodsprings' window panes and dust effects
    // were floating white rectangles until this was read.
    bool alphaBlend = false;
};

struct NifModel {
    std::vector<NifShape> shapes;
    // Count of geometry blocks that were dropped rather than emitted as
    // possibly-corrupt geometry: either the NiTriShapeData/NiTriStripsData
    // field layout did not parse cleanly, or a NiTriShape/NiTriStrips pointed
    // at a data block that could not be read. A nonzero count here means
    // geometry is missing from the model — it is not decorative.
    std::uint32_t skippedShapeCount = 0;
    // Blocks whose type name ends in "Node" but which this parser does not
    // know how to walk. Nonzero means geometry may be missing or, worse,
    // reparented to the origin — see isNodeTypeName in nif_scene.cc.
    std::uint32_t unhandledNodeTypeCount = 0;
    // Type names of blocks referenced as a shape's property that this parser did
    // not turn into either a texture set or an alpha setting. A shape can hold a
    // shader property of a type the parser has no branch for, in which case it
    // silently ends up with no diffuse path and shades from the per-model hashed
    // colour -- grey patches on an otherwise textured model. Populated for
    // diagnostics (odai_newvegas_probe --nif) so those types can be identified
    // rather than guessed at.
    std::vector<std::string> unresolvedPropertyTypes;
};

// ---------------------------------------------------------------------------
// Skeletons and skinning (Dragon Age: Origins touchstone -- see
// docs/ROADMAP.md's Party RPG section, and the GPU skinning compute pass in
// render/backend/vulkan/skinning_resources.cc that consumes this).
//
// Fallout keeps the two halves in separate files, and the split is the thing to
// understand before reading any of it:
//
//   meshes\characters\_male\skeleton.nif   the bone hierarchy and nothing else.
//                                          65 NiNodes, no skinned geometry.
//   meshes\characters\_male\upperbody.nif  the geometry. 6 NiTriShapes, each
//                                          with a BSDismemberSkinInstance
//                                          naming the bones it binds to BY
//                                          NAME, plus 27 NiNodes that are a
//                                          partial COPY of the skeleton.
//
// So a skinned mesh does not carry bone indices into any global array -- it
// carries names. Binding is a name lookup against a skeleton parsed separately,
// which is why parseNifSkeleton and parseNifSkinnedMesh are separate entry
// points that meet in the caller rather than one function.

// One bone of a NIF skeleton, in the file's own coordinate space (Bethesda
// Z-up, centimetre-ish units -- converting to engine space is the caller's job,
// exactly as it is for static geometry).
struct NifSkeletonBone {
    std::string name;
    int parentIndex = -1;  // -1 for a root; parents always precede children
    // Bind-pose transform, RELATIVE TO THE PARENT. This is what NiNode stores
    // natively, and it is what anim::Skeleton wants, so no conversion happens
    // in between.
    float translation[3] = {0.0f, 0.0f, 0.0f};
    float rotation[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};  // row-major 3x3
    float scale = 1.0f;
};

struct NifSkeleton {
    // Topologically ordered: a bone's parentIndex is always less than its own
    // index, so a single forward pass accumulates world transforms with no
    // recursion and no second sort.
    std::vector<NifSkeletonBone> bones;
    // Nodes whose declared parent was never seen. Nonzero means the hierarchy
    // came out flatter than the file describes and some limb is attached to the
    // root instead of where it belongs -- the silent-misplacement failure this
    // parser's node handling has hit before.
    std::uint32_t orphanedBoneCount = 0;
};

// Up to four bone influences per vertex, which is what the skinning compute
// shader consumes. FNV's NiSkinData routinely gives a vertex more than four;
// the extras are dropped smallest-first and the survivors renormalized, so the
// weights always sum to 1 and a dropped influence shifts the vertex slightly
// rather than shrinking it toward the origin.
inline constexpr int kNifMaxBoneInfluences = 4;

struct NifSkinnedShape {
    std::string name;
    std::vector<float> positions;  // xyz per vertex, SKIN space (see below)
    std::vector<float> normals;
    std::vector<float> uvs;
    std::vector<std::uint32_t> triangleIndices;
    std::string diffuseTexturePath;
    bool alphaTest = false;
    std::uint8_t alphaThreshold = 128;
    bool alphaBlend = false;
    bool twoSided = false;

    // Bone names this shape binds to, in the order its own skin data uses.
    // These index nothing on their own -- resolve them against a NifSkeleton.
    std::vector<std::string> boneNames;
    // Per bone, the inverse bind transform NiSkinData stores for it, as a
    // row-major 4x4 with translation in the last column. Composed with the
    // skeleton's animated world transform, this is the skinning matrix.
    std::vector<float> inverseBindMatrices;  // 16 floats per bone

    // NiSkinData's OVERALL skin transform, the one field ahead of the per-bone
    // array. Row-major 4x4, same convention as above.
    //
    // Discarding this is a mistake worth naming, because the result looks
    // almost right: the per-bone transforms are relative to the skeleton root,
    // and this maps the shape's own geometry space into that root space. Leave
    // it out and every vertex is skinned by a matrix that is correct in
    // rotation and wrong in origin, which reads on screen as a body whose limbs
    // point the right way from the wrong place.
    //
    // Measured on characters\_male\upperbody.nif: without it, the bind pose
    // fails to round-trip by up to 112.8 units -- most of a character height.
    float skinTransform[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};

    // Per vertex, kNifMaxBoneInfluences entries each. Index is into boneNames,
    // NOT into a skeleton. Weights sum to 1 for any vertex with any influence;
    // an unused slot is weight 0 and index 0.
    std::vector<std::uint16_t> boneIndices;
    std::vector<float> boneWeights;
};

struct NifSkinnedModel {
    std::vector<NifSkinnedShape> shapes;
    // Shapes that had geometry but no resolvable skin instance. These are real
    // -- a character NIF can hold an unskinned prop -- but they cannot be posed,
    // so they are counted rather than silently emitted as if they were skinned.
    std::uint32_t unskinnedShapeCount = 0;
    // Vertices whose influence count exceeded kNifMaxBoneInfluences and were
    // truncated. Nonzero is normal for FNV bodies; a very large value means the
    // truncation is doing visible work and is worth looking at.
    std::uint32_t truncatedInfluenceVertexCount = 0;
};

// Parses the NiNode hierarchy of a skeleton NIF into a flat bone array.
// Geometry, collision (bhk*) and controller blocks are ignored -- a skeleton
// file has no geometry worth importing, and its ragdoll bodies are physics data
// this engine does not consume.
bool parseNifSkeleton(
    const std::vector<std::uint8_t>& bytes, NifSkeleton& outSkeleton, std::string& outError);

// Parses skinned geometry: NiTriShape/NiTriStrips whose skin instance resolves
// to a NiSkinData. Unskinned shapes in the same file are counted, not emitted.
bool parseNifSkinnedMesh(
    const std::vector<std::uint8_t>& bytes, NifSkinnedModel& outModel, std::string& outError);

// Raw block/string inventory of a NIF, for diagnostics only (odai_newvegas_probe
// --nifblocks). Nothing in the import path consumes this; it exists so questions
// like "what type is this shape's property, and where does its texture path
// actually live" get answered by reading the file instead of guessing at field
// offsets -- the failure mode this parser has hit repeatedly.
struct NifBlockSummary {
    std::vector<std::string> blockTypeNames;  // per block, indexed by block index
    std::vector<std::uint32_t> blockSizes;    // per block, bytes
    std::vector<std::string> strings;         // header string table
    std::vector<std::size_t> blockStarts;     // byte offset of each block
};

bool parseNifBlockSummary(
    const std::vector<std::uint8_t>& bytes, NifBlockSummary& outSummary, std::string& outError);

bool parseNifStaticMesh(const std::vector<std::uint8_t>& bytes, NifModel& outModel, std::string& outError);
bool loadNifStaticMesh(const std::filesystem::path& path, NifModel& outModel, std::string& outError);

}  // namespace odai::importer::fnv
