#include "anim/humanoid_rig.h"
#include "anim/animation_sampler.h"
#include "import/bethesda/character_builder.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <fstream>
#include <iterator>

int main(int argc, char** argv) {
    using namespace odai::anim;
    odai::importer::bethesda::FalloutCharacter character;
    const std::pair<const char*, int> bones[] = {
        {"NPC Root [Root]", -1}, {"NPC Pelvis [Pelv]", 0}, {"NPC Spine [Spn0]", 0},
        {"NPC Head [Head]", 2},
        {"NPC R UpperArm [RUar]", 2}, {"NPC R Forearm [RLar]", 4}, {"NPC R Hand [RHnd]", 5},
        {"NPC L UpperArm [LUar]", 2}, {"NPC L Forearm [LLar]", 7}, {"NPC L Hand [LHnd]", 8},
        {"NPC R Thigh [RThg]", 1}, {"NPC R Calf [RClf]", 10}, {"NPC R Foot [Rft ]", 11},
        {"NPC L Thigh [LThg]", 1}, {"NPC L Calf [LClf]", 13}, {"NPC L Foot [Lft ]", 14},
        {"custom_hair", 3}, {"WeaponSword", 6}
    };
    for (const auto& [name, parent] : bones) character.skeleton.bones.push_back({name, parent, {1, 2, 3}});
    for (std::size_t i = 0; i < character.skeleton.bones.size(); ++i)
        character.inverseBindMatrices.push_back(odai::math::Matrix4::translation({float(i), 0, 0}));
    character.vertices.resize(1);
    character.vertices[0].boneIndices[0] = 6;
    character.vertices[0].boneWeights[0] = 1;
    AnimationClip clip; clip.name = "test"; clip.duration = 1;
    BoneTrack track; track.boneIndex = 6;
    track.translationKeys = {{0, {1, 2, 3}}, {1, {5, 6, 7}}};
    clip.tracks.push_back(track);
    std::vector<AnimationClip> clips{clip};
    AnimationSampler sampler;
    sampler.bindSkeleton(character.skeleton, character.inverseBindMatrices);
    std::vector<odai::math::Matrix4> before, after;
    sampler.sample(character.skeleton, clip, .5f, before);
    const auto original = character;
    std::string error;
    assert(odai::importer::bethesda::canonicalizeSkyrimCharacter(character, clips, error));
    const auto& mapping = *character.humanoidRig;
    assert(character.vertices[0].boneIndices[0] == mapping.roles.at("right_hand"));
    assert(clips[0].tracks[0].boneIndex == mapping.roles.at("right_hand"));
    assert(mapping.firstPersonMask[mapping.roles.at("right_hand")] == 1);
    assert(mapping.firstPersonMask[mapping.roles.at("head")] == 0);
    assert(character.skeleton.findBone("custom_hair") >= 0);
    sampler.bindSkeleton(character.skeleton, character.inverseBindMatrices);
    sampler.sample(character.skeleton, clips[0], .5f, after);
    for (std::size_t i = 0; i < before.size(); ++i) for (int j = 0; j < 16; ++j)
        assert(std::abs(before[i].m[j] - after[mapping.sourceToCanonical[i]].m[j]) < 0.0001f);
    HumanoidRigMapping second;
    assert(canonicalizeHumanoidRig(character.skeleton, second, error));
    for (std::size_t i = 0; i < second.sourceToCanonical.size(); ++i) assert(second.sourceToCanonical[i] == int(i));
    auto invalid = original;
    invalid.skeleton.bones[6].name = "missing_hand";
    assert(!odai::importer::bethesda::canonicalizeSkyrimCharacter(invalid, {}, error));
    assert(invalid.vertices[0].boneIndices[0] == 6 && !invalid.humanoidRig);
    invalid = original;
    invalid.vertices[0].boneIndices[0] = 999;
    assert(!odai::importer::bethesda::canonicalizeSkyrimCharacter(invalid, {}, error));
    assert(invalid.skeleton.bones[4].name == original.skeleton.bones[4].name);
    // Extended rigs must retain optional bones, deterministic identity and
    // every hierarchy level above the old small-biped sizes.
    auto extended = original.skeleton;
    extended.bones.push_back({"NPC L Toe0 [LToe]",15,{0,0,1}});
    while (extended.bones.size() < 650)
        extended.bones.push_back({"extension_" + std::to_string(extended.bones.size()),3,{0,1,0}});
    HumanoidRigMapping large;
    assert(canonicalizeHumanoidRig(extended,large,error));
    assert(large.skeleton.bones.size()==650 && large.roles.contains("left_toe"));
    assert(large.limbs.at("left_leg").end==large.roles.at("left_foot"));
    std::size_t scheduled=0;
    for (const auto& level:large.depthLevels) scheduled+=level.size();
    assert(scheduled==650 && !large.fingerprint.empty());
    HumanoidRigMapping again;
    assert(canonicalizeHumanoidRig(large.skeleton,again,error));
    assert(again.fingerprint==large.fingerprint);
    auto proportioned=extended;
    for (auto& bone:proportioned.bones) bone.localTranslation=bone.localTranslation*2.f;
    proportioned.bones[6].localRotation=odai::math::normalize(odai::math::Quaternion{0,.3f,0,1});
    ClipRigBinding binding;
    assert(bindHumanoidClipRig(extended,proportioned,TranslationScalePolicy::LimbLength,binding,error));
    assert(!binding.direct);
    AnimationClip retargeted;
    assert(retargetHumanoidClip(clip,binding,retargeted,error));
    assert(retargeted.tracks[0].boneIndex==6);
    const auto restDelta=retargeted.tracks[0].translationKeys[0].value-proportioned.bones[6].localTranslation;
    assert(odai::math::length(restDelta)<.0001f);
    assert(bindHumanoidClipRig(extended,extended,TranslationScalePolicy::LimbLength,binding,error));
    assert(binding.direct && retargetHumanoidClip(clip,binding,retargeted,error));
    assert(retargeted.tracks[0].translationKeys[1].value.x==5);
    auto incompatible=proportioned;incompatible.bones[16].parentIndex=2;
    const auto fingerprint=binding.targetFingerprint;
    assert(!bindHumanoidClipRig(extended,incompatible,TranslationScalePolicy::LimbLength,binding,error));
    assert(binding.targetFingerprint==fingerprint);
    invalid = original;
    invalid.skeleton.bones[6].parentIndex = 1;
    assert(!odai::importer::bethesda::canonicalizeSkyrimCharacter(invalid, {}, error));
    if (argc == 3 && std::string_view(argv[1]) == "--rig-file") {
        std::ifstream input(argv[2], std::ios::binary);
        const std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(input)), {});
        odai::importer::bethesda::NifSkeleton nif;
        Skeleton imported;
        HumanoidRigMapping mapped;
        if (!odai::importer::bethesda::parseNifSkeleton(bytes, nif, error) ||
            !odai::importer::bethesda::buildFalloutSkeleton(nif, imported) ||
            !canonicalizeHumanoidRig(imported, mapped, error)) {
            std::cerr << "Local rig probe failed: " << error << '\n'; return 1;
        }
        std::cout << "Local humanoid rig admitted: " << imported.bones.size() << " bones, "
            << mapped.roles.size() << " semantic roles\n";
    }
    std::cout << "Humanoid rig conversion tests passed\n";
}
