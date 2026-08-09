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

#include "anim/skeleton.h"
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
    std::string diffuseTexturePath;
    std::uint32_t firstIndex = 0;
    std::uint32_t indexCount = 0;
    bool alphaTest = false;
    std::uint8_t alphaThreshold = 128;
    bool alphaBlend = false;
    bool twoSided = false;
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

    // Bones named by a skinned shape that the skeleton does not contain. Every
    // vertex weighted to one is silently rebound to the root, so a nonzero
    // count here means the character is deformed, not merely incomplete.
    std::uint32_t unresolvedBoneCount = 0;
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
