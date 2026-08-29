#pragma once

// Binds a Fallout: New Vegas skeleton NIF and one or more skinned body-part
// NIFs into engine-space, GPU-skinnable data: an anim::Skeleton plus the vertex
// / index / draw arrays that Renderer::uploadSkinnedMeshTemplate takes.
//
// This is the seam between nif_scene.cc's file-shaped output (bone names, skin
// spaces, Bethesda Z-up) and what the rest of the engine already speaks
// (anim::Skeleton, render::ImportedSkinnedMeshVertex, engine Y-up). Everything
// coordinate-system-shaped happens here and nowhere else.
//
// Why a separate translation unit rather than more of nif_scene.cc: nif_scene
// deliberately knows nothing about the engine's types, so it stays testable
// against raw bytes alone. This file is where engine types enter.

#include "anim/animation_clip.h"
#include "anim/skeleton.h"
#include "import/fnv/kf_animation.h"
#include "import/fnv/nif_scene.h"
#include "import/imported_scene.h"
#include "math/math.h"
#include "render/renderer_types.h"

#include <cstdint>
#include <string>
#include <vector>

namespace odai::importer::fnv {

// One skinned body part, kept separate because each has its own diffuse
// texture and the renderer draws one texture per packed draw.
struct FalloutCharacterPart {
    std::string name;
    // Asset provenance is presentation-only metadata, but retaining it here
    // makes a malformed assembled actor diagnosable one authored BODY NIF at
    // a time instead of as one anonymous merged vertex buffer.
    std::string sourcePath;
    std::string diffuseTexturePath;
    std::uint32_t firstIndex = 0;
    std::uint32_t indexCount = 0;
    bool alphaTest = false;
    std::uint8_t alphaThreshold = 128;
    bool alphaBlend = false;
    bool twoSided = false;
    // BSShaderNoLightingProperty -- see NifShape::unlit.
    bool unlit = false;
};

// A character assembled and ready to upload. vertices/indices are one merged
// buffer across all parts, because the skinning compute pass runs over a single
// contiguous vertex range per instance slot.
struct FalloutCharacter {
    anim::Skeleton skeleton;
    std::vector<odai::render::ImportedSkinnedMeshVertex> vertices;
    std::vector<std::uint32_t> indices;
    std::vector<FalloutCharacterPart> parts;

    // Per skeleton bone, the inverse bind matrix taken from NiSkinData rather
    // than derived from the skeleton's own bind pose.
    //
    // These are NOT the same thing, and preferring the file's is deliberate: a
    // skinned NIF's vertices live in "skin space", which is whatever space the
    // artist's export happened to use, and NiSkinData's per-bone transform is
    // the only record of how that space relates to each bone. Recomputing the
    // inverse bind from the skeleton assumes skin space equals the skeleton
    // root's space -- true often enough to look nearly right and be subtly
    // wrong, which is the worst failure mode available here.
    //
    // A bone no shape binds keeps identity; it is never referenced by a vertex,
    // so its value cannot matter.
    std::vector<odai::math::Matrix4> inverseBindMatrices;

    // Bones named by a skinned shape that the skeleton does not contain. A
    // triangle touching one of those influences is omitted: allowing only its
    // resolved vertices through would connect actor-local geometry to the
    // world origin and produce a strip across the scene.
    std::uint32_t unresolvedBoneCount = 0;
    std::uint32_t droppedUnresolvedBoneTriangleCount = 0;
    // Shapes whose inverse bind for a shared bone disagreed with one already
    // recorded by an earlier shape. Zero for retail body parts that share a
    // skeleton; nonzero means the parts were authored against different rigs
    // and cannot share one bone-matrix array.
    std::uint32_t conflictingInverseBindCount = 0;
};

// Converts a NIF bone hierarchy into an anim::Skeleton, changing coordinate
// systems from Bethesda Z-up to engine Y-up.
//
// The conversion is a SIMILARITY transform (M * R * transpose(M)) on each
// bone's local rotation, not the single-sided M * R that cell_builder applies
// to placed statics. The difference matters and the reason is structural:
// cell_builder's rotation is applied to vertices that are still in Bethesda
// space, so it converts and rotates in one step. A bone's local rotation is
// composed with its parent's, so it has to be a change of basis -- otherwise
// the conversion applies once per level of the hierarchy and a fingertip comes
// out rotated five times over.
bool buildFalloutSkeleton(const NifSkeleton& source, anim::Skeleton& outSkeleton);

// Binds skinned shapes to a skeleton, remapping each shape's local bone list
// onto skeleton bone indices and converting geometry to engine space.
//
// Call once per character with every body part it wears; parts appended across
// several calls share one skeleton and one bone-matrix array.
bool appendFalloutCharacterMesh(
    const NifSkinnedModel& model, FalloutCharacter& character, std::string& outError);

// Attaches a NON-skinned body part to the bone that shares its root node name.
//
// Not every body part in a creature's NIFZ list is weighted geometry. Fallout
// parents rigid props to a bone instead -- Victor's face screen is one, and its
// NIF's root node is literally called "Screen01Root", which is a bone in
// creatures\NVSecuritron\Skeleton.nif. appendFalloutCharacterMesh rejects such
// a part ("no skinned shapes") and it is simply never drawn, which is how a
// Securitron ends up with an empty black box for a head.
//
// `rootNodeName` is the part NIF's own root node, which parseNifSkeleton
// reports; `model` is the same bytes through parseNifStaticMesh.
//
// The vertices are weighted 1.0 to that bone, and baked through the INVERSE of
// the bone's stored inverse-bind so the skinning product comes back out as
// "rigidly parented to the bone" (the derivation is in the .cc). That reads the
// shared inverseBindMatrices entry without writing it, so no skinned shape's
// binding is disturbed.
//
// Emitting these unweighted instead -- which an earlier version did, on the
// grounds that nothing animated yet -- is not a neutral choice: the skinning
// shader passes an unweighted vertex through at its authored position, so the
// part misses the actor's world placement as well as its pose and draws at the
// world origin.
bool appendFalloutCharacterRigidMesh(
    const NifModel& model,
    const std::string& rootNodeName,
    FalloutCharacter& character,
    std::string& outError);

struct FalloutAnimationStats {
    std::size_t tracks = 0;
    // Tracks whose node name matched a bone in the skeleton.
    std::size_t boundTracks = 0;
    // Tracks naming a node this skeleton does not have. Normal in small
    // numbers: a .kf targets weapon and accumulation nodes ("Bip01 NonAccum",
    // "Bip01 MachineGunBarrel00") that a given creature's skeleton may omit.
    std::size_t unresolvedNodes = 0;
};

// Converts a parsed .kf into an AnimationClip posed against `skeleton`.
//
// Two jobs, and both have to be done HERE rather than in the .kf reader. The
// first is resolving node names to bone indices, which the file itself cannot
// do -- a .kf names its targets and nothing else. The second is the basis
// change: keys come out of the file in Bethesda's Z-up space, and a bone's
// animated local transform is the same kind of quantity as its bind-pose local
// transform, so it must go through exactly the conversion buildFalloutSkeleton
// applies (translation (x,y,z) -> (x,z,-y); rotation R -> M R transpose(M)).
// Converting one and not the other, or converting them differently, yields a
// character whose bind pose is right and whose every animated frame is rotated
// into the floor.
bool buildFalloutAnimationClip(
    const KfAnimation& source,
    const anim::Skeleton& skeleton,
    anim::AnimationClip& outClip,
    FalloutAnimationStats& outStats);

// Fills outMatrices with skeleton.bones.size() skinning matrices for the bind
// pose: worldBoneTransform * inverseBindMatrix, exactly the product
// anim::AnimationSampler produces for an animated pose.
//
// Useful on its own -- a character standing in bind pose is the first thing
// worth putting on screen, and it isolates "did the skin bind correctly" from
// "did the animation sample correctly".
void computeFalloutBindPose(
    const FalloutCharacter& character, std::vector<odai::math::Matrix4>& outMatrices);

}  // namespace odai::importer::fnv
