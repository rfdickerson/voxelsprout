#pragma once

#include "anim/animation_clip.h"
#include "anim/animation_sampler.h"
#include "anim/behavior_program.h"
#include "anim/skeleton.h"
#include "anim/native_program.h"
#include "anim/humanoid_rig.h"
#include "anim/pose_modifiers.h"
#include <array>
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
    struct MotionWarpTarget { odai::math::Vector3 position{}; float yawRadians = 0; };
    std::map<std::string,MotionWarpTarget> motionWarpTargets;
    struct FootContact {
        bool valid = false;
        odai::math::Vector3 position{}, normal{0,1,0}; // world space
    };
    std::array<FootContact, 2> footContacts;
    odai::math::Vector3 actorPosition{};
    bool teleported = false, ragdollActive = false;
    std::map<std::string, LimbTarget> limbTargets; // model space, semantic chain names
    bool aimEnabled = false;
    odai::math::Vector3 aimTarget{}; // model space
    AnimationSelectorContext selectorContext;
    // A presentation view follows the authoritative action clock but samples
    // its own clips. Empty state leaves this instance as the authority.
    std::string sharedState;
    float sharedStateTime = 0;
    std::uint64_t sharedActionIdentity = 0;
    bool ownsGameplayEvents = true;
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
    bool jumpRequested = false;
    std::string jumpPhase;
    std::string landingSeverity;
    float landingImpactMetres = 0;
    bool jumpPreparing = false; // Grounded anticipation before a gameplay-owned impulse.
    bool grounded = true;
    bool falling = false;
    bool landed = false;
    bool blocked = false;
    bool animationDriven = false;
    bool weaponDrawn = false;
    bool attacking = false;
    bool equipping = false;
    bool sprinting = false;
    bool sneaking = false;
    bool swimming = false;
    bool dead = false;
    bool talking = false;
    bool running = false;
    bool blocking = false;
    std::string weaponStyle; // empty/h2h, 1hm, 2hm, 2hw, bow, crossbow, staff
    float actorYawRadians = 0.0f;
    // Named variable inputs are retained for decoded graphs; gameplay owns values.
    std::map<std::string, float> variables;
    bool footIkEnabled = false;
    float leftFootIkOffset = 0.0f;   // bounded engine-space Y correction
    float rightFootIkOffset = 0.0f;
    std::vector<AnimationEvent> events;
};

struct AnimationStepOutput {
    bool resetHistory = false;
    LocalPose localPose;
    std::shared_ptr<const PoseEvaluationPacket> evaluationPacket;
    // Authored timeline crossings only; request events cannot cause gameplay contact.
    std::vector<AnimationEvent> clipEvents;
    std::vector<odai::math::Matrix4> pose;
    std::map<std::string, odai::math::Matrix4> socketTransforms;
    std::vector<AnimationEvent> events;
    odai::math::Vector3 desiredRootMotion{};
    std::vector<AnimationDiagnostic> diagnostics;
    std::string activeState;
    bool proceduralFallback = false;
    bool authoredGraphExecuted = false;
    bool actionActive = false;
    std::string activeClip;
    AnimationExecutionMode executionMode = AnimationExecutionMode::Native;
    std::string activeRule, activeProvider, fallbackReason;
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
    AnimationExecutionMode executionMode = AnimationExecutionMode::Native;
    std::shared_ptr<const NativeAnimationProgram> nativeProgram;
    AnimationSelectorContext selectorDefaults;
    std::shared_ptr<const HumanoidRigMapping> humanoidRig;
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
    HkxCharacterAssets characterAssets;
    std::string characterDefinitionPath;
    std::string animationSkeletonPath;
    bool supportedBehaviorGraph = false;
    std::shared_ptr<const BehaviorProgram> behavior;
    std::vector<AnimationDiagnostic> diagnostics;
};

struct BehaviorGraphSnapshot {
    float warpTranslationUsed = 0;
    std::string warpWindow;
    std::string inertialState;
    std::map<std::string, std::vector<float>> footPlants;
    std::string poseGraphState;
    AnimationExecutionMode executionMode = AnimationExecutionMode::Native;
    std::string selectedRule, selectedProvider, selectedClip, previousClip;
    std::uint64_t randomState = 0, actionIdentity = 0;
    std::map<std::string, float> layerTimes;
    std::string state = "idle";
    float stateTime = 0.0f;
    std::string previousState;
    float previousStateTime = 0.0f;
    float transitionElapsed = 0.0f;
    float transitionDuration = 0.0f;
    std::uint64_t fixedTick = 0;
    bool wasGrounded = true;
    bool wasTalking = false;
    std::vector<AnimationEvent> queuedEvents;
    std::map<std::uint32_t, std::int32_t> activeStates;
    std::map<std::string, float> variables;
    std::string graphFingerprint;
    friend bool operator==(const BehaviorGraphSnapshot&, const BehaviorGraphSnapshot&) = default;
};

// Shared deterministic fixed-tick animator. Native selection is the default;
// the admitted HKX interpreter is an explicit view-level compatibility mode.
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
    const AnimationClip* clipByName(const std::string& name) const;
    const AnimationClip* clipForState(const std::string& state) const;
    std::string chooseState(const AnimationInputState& input) const;
    std::string chooseAuthoredState(std::span<const AnimationEvent> events,
        float& transitionDuration, bool& transitioned, std::vector<AnimationDiagnostic>& diagnostics);
    void refreshSockets(AnimationStepOutput& output) const;
    void applyFootIk(const AnimationInputState& input, AnimationStepOutput& output) const;

    const AnimationView* m_view = nullptr;
    AnimationSampler m_sampler;
    BehaviorGraphSnapshot m_state;
    std::vector<odai::math::Matrix4> m_bindWorld;
    PoseGraphInstance m_poseGraph;
    PoseInertializer m_inertializer;
};

}  // namespace odai::anim
