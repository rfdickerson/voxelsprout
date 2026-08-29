#pragma once

#include "anim/animation_clip.h"
#include "anim/animation_sampler.h"
#include "anim/skeleton.h"
#include "math/math.h"

#include <cstdint>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace odai::anim {

enum class AnimationDiagnosticSeverity : std::uint8_t { Info, Warning, Error };

struct AnimationDiagnostic {
    AnimationDiagnosticSeverity severity = AnimationDiagnosticSeverity::Info;
    std::string code;
    std::string message;
};

struct AnimationEvent {
    std::string name;
    std::string payload;
    friend bool operator==(const AnimationEvent&, const AnimationEvent&) = default;
};

struct AnimationInputState {
    odai::math::Vector3 requestedVelocity{};
    odai::math::Vector3 groundVelocity{};
    odai::math::Vector3 groundNormal{0.0f, 1.0f, 0.0f};
    float verticalVelocity = 0.0f;
    float movementSpeed = 0.0f;
    // Velocity after rotation into the actor's local frame. This lets a
    // third-person graph choose authored forward/back/strafe clips without
    // making the camera or renderer another movement authority.
    odai::math::Vector3 localVelocity{};
    float turnRateRadiansPerSecond = 0.0f;
    float locomotionPlaybackRate = 1.0f;
    bool grounded = true;
    bool falling = false;
    bool landed = false;
    bool blocked = false;
    bool animationDriven = false;
    bool weaponDrawn = false;
    bool attacking = false;
    bool equipping = false;
    bool sprinting = false;
    bool footIkEnabled = false;
    float leftFootIkOffset = 0.0f;   // bounded engine-space Y correction
    float rightFootIkOffset = 0.0f;
    std::vector<AnimationEvent> events;
};

struct AnimationStepOutput {
    std::vector<odai::math::Matrix4> pose;
    std::map<std::string, odai::math::Matrix4> socketTransforms;
    std::vector<AnimationEvent> events;
    odai::math::Vector3 desiredRootMotion{};
    std::vector<AnimationDiagnostic> diagnostics;
    std::string activeState;
    bool proceduralFallback = false;
};

struct RigBindingResult {
    std::vector<int> trackToBone;
    std::size_t exactMatches = 0;
    std::size_t caseInsensitiveMatches = 0;
    std::vector<std::string> missingTracks;
    std::vector<AnimationDiagnostic> diagnostics;
    [[nodiscard]] float coverage() const;
};

RigBindingResult bindTracksByName(
    std::span<const std::string> trackNames, const Skeleton& skeleton);

struct AnimationView {
    // Views are retained by BethesdaSession and may outlive the importer or a
    // rebuilt equipment mesh. Owning the immutable rig here prevents the raw
    // pointer lifetime bugs that otherwise appear on live outfit changes.
    std::shared_ptr<const Skeleton> skeleton;
    std::vector<odai::math::Matrix4> inverseBindMatrices;
    std::vector<AnimationClip> clips;
    // Gameplay state -> clip name. Missing states fall back per actor.
    std::unordered_map<std::string, std::string> stateClips;
    std::vector<std::string> socketBoneNames;
    std::string sourceFingerprint;
    std::string providerId;
    bool supportedBehaviorGraph = false;
};

struct BehaviorGraphSnapshot {
    std::string state = "idle";
    float stateTime = 0.0f;
    std::string previousState;
    float previousStateTime = 0.0f;
    float transitionElapsed = 0.0f;
    float transitionDuration = 0.0f;
    std::uint64_t fixedTick = 0;
    bool wasGrounded = true;
    std::vector<AnimationEvent> queuedEvents;
    friend bool operator==(const BehaviorGraphSnapshot&, const BehaviorGraphSnapshot&) = default;
};

// Deterministic fixed-tick graph instance. The reflected HKX graph loader can
// populate AnimationView; unsupported graphs use the same instance with the
// explicit per-actor procedural fallback flag set.
class BehaviorGraphInstance {
public:
    bool bind(const AnimationView& view, std::string& outError);
    AnimationStepOutput step(const AnimationInputState& input, float fixedDeltaSeconds);
    void queueEvent(AnimationEvent event);
    [[nodiscard]] BehaviorGraphSnapshot snapshot() const;
    bool restore(const BehaviorGraphSnapshot& snapshot, std::string& outError);

    static AnimationStepOutput interpolate(
        const AnimationStepOutput& previous, const AnimationStepOutput& current, float alpha);

private:
    const AnimationClip* clipForState(const std::string& state) const;
    std::string chooseState(const AnimationInputState& input) const;
    void refreshSockets(AnimationStepOutput& output) const;
    void applyFootIk(const AnimationInputState& input, AnimationStepOutput& output) const;

    const AnimationView* m_view = nullptr;
    AnimationSampler m_sampler;
    BehaviorGraphSnapshot m_state;
    std::vector<odai::math::Matrix4> m_bindWorld;
};

}  // namespace odai::anim
