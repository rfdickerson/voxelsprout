#pragma once

#include "anim/animation_clip.h"
#include "anim/animation_sampler.h"
#include "anim/skeleton.h"
#include "engine/game_app.h"
#include "math/math.h"
#include "render/renderer_types.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

// "Legion" -- a small-party (4 legionaries vs. 4 raiders) real-time-with-pause
// skirmish. This is the Dragon Age: Origins touchstone's combat slice
// (docs/ROADMAP.md's Party RPG / Narrative section), Rome-themed, and the
// first real exerciser of the GPU skinning pipeline end to end: every
// character is an independently posed instance of the same procedural biped
// rig (anim::makeBipedSkeleton, procgen::buildHumanoidSkinnedMesh), driven
// through Renderer::uploadSkinnedMeshTemplate/setSkinnedActorPose.
//
// Deliberately NOT a mass-battle RTS: kMaxSkinnedInstances is the hard ceiling,
// matching docs/ROADMAP.md's explicit out-of-scope note on thousands-of-units
// crowd rendering. (That constant is no longer 8 -- see renderer_types.h -- but
// the point stands: every slot is independently posed and dispatched.)
namespace odai::games::legion {

enum class CombatState : std::uint8_t { Idle, Walk, Attack, Dead };

struct Character {
    std::uint32_t instanceSlot = 0;
    bool isEnemy = false;
    bool alive = true;
    float hp = 100.0f;
    float maxHp = 100.0f;
    odai::math::Vector3 position{};
    float facingYawRadians = 0.0f;
    CombatState state = CombatState::Idle;
    float animClock = 0.0f;
    float deathSinkY = 0.0f;
    int targetIndex = -1;  // index into the opposing roster, -1 = none picked yet
};

class LegionApp : public engine::GameApp {
protected:
    bool onInit() override;
    void onTick(float dt) override;
    void onRender(float dt) override;

private:
    static constexpr std::size_t kPartySize = 4;
    static constexpr std::size_t kEnemySize = 4;
    static constexpr float kEngageRange = 1.15f;
    static constexpr float kWalkSpeed = 1.6f;
    static constexpr float kAttackDamage = 12.0f;
    static constexpr float kDeathSinkSpeed = 0.5f;
    static constexpr float kDeathSinkMax = 1.3f;

    void resetBattle();
    void uploadCharacterMeshes();
    // Runs one side's AI/movement/combat for this tick. `theirs` is mutable
    // because a landed attack reduces the target's hp directly.
    void tickOneSide(std::span<Character> mine, std::span<Character> theirs, float dt);
    void updateCharacterPose(const Character& character);
    void drawHud();
    [[nodiscard]] render::CameraPose computeCameraPose() const;

    // Whichever of translationKeys/rotationKeys a clip's tracks omit falls
    // back to bind pose (AnimationSampler::sample); every clip here is
    // authored against the exact same skeleton.
    odai::anim::Skeleton m_skeleton;
    odai::anim::AnimationClip m_idleClip;
    odai::anim::AnimationClip m_walkClip;
    odai::anim::AnimationClip m_attackClip;
    odai::anim::AnimationSampler m_sampler;
    // Reused across every character/frame (AnimationSampler::sample resizes
    // in place, never shrinking capacity) so posing all 8 characters a frame
    // doesn't reallocate once warmed up.
    std::vector<odai::math::Matrix4> m_localPoseScratch;
    std::vector<odai::math::Matrix4> m_finalPoseScratch;

    std::array<Character, kPartySize> m_party{};
    std::array<Character, kEnemySize> m_enemies{};

    bool m_paused = false;
    bool m_prevSpace = false;
    bool m_prevR = false;

    float m_camYawDeg = -20.0f;
    float m_camZoom = 6.0f;

    render::CameraPose m_camera{};
};

}  // namespace odai::games::legion
