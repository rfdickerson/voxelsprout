#include "anim/skyrim_animation.h"

#include <algorithm>
#include <cctype>
#include <cmath>

namespace odai::anim {
namespace {

std::string lowerAscii(std::string text) {
    for (char& ch : text) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    return text;
}

odai::math::Matrix4 localBindMatrix(const Bone& bone) {
    return odai::math::Matrix4::translation(bone.localTranslation) *
        odai::math::toMatrix(bone.localRotation) * odai::math::Matrix4::scale(bone.localScale);
}

odai::math::Vector3 sampleRootTranslation(const AnimationClip& clip, int root, float time) {
    const BoneTrack* track = nullptr;
    for (const BoneTrack& candidate : clip.tracks) {
        if (candidate.boneIndex == root) { track = &candidate; break; }
    }
    if (track == nullptr || track->translationKeys.empty()) return {};
    float t = time;
    if (clip.duration > 0.0f && clip.loop) {
        t = std::fmod(t, clip.duration);
        if (t < 0.0f) t += clip.duration;
    } else {
        t = std::clamp(t, 0.0f, clip.duration);
    }
    const auto& keys = track->translationKeys;
    if (t <= keys.front().time) return keys.front().value;
    for (std::size_t index = 0; index + 1u < keys.size(); ++index) {
        if (t <= keys[index + 1u].time) {
            const float span = keys[index + 1u].time - keys[index].time;
            const float alpha = span > 0.0f ? (t - keys[index].time) / span : 0.0f;
            return odai::math::lerp(keys[index].value, keys[index + 1u].value, alpha);
        }
    }
    return keys.back().value;
}

}  // namespace

float RigBindingResult::coverage() const {
    return trackToBone.empty() ? 1.0f :
        static_cast<float>(exactMatches + caseInsensitiveMatches) /
            static_cast<float>(trackToBone.size());
}

RigBindingResult bindTracksByName(
    std::span<const std::string> trackNames, const Skeleton& skeleton) {
    RigBindingResult result;
    result.trackToBone.assign(trackNames.size(), -1);
    std::unordered_map<std::string, int> folded;
    for (std::size_t index = 0; index < skeleton.bones.size(); ++index) {
        folded.try_emplace(lowerAscii(skeleton.bones[index].name), static_cast<int>(index));
    }
    for (std::size_t track = 0; track < trackNames.size(); ++track) {
        const int exact = skeleton.findBone(trackNames[track]);
        if (exact >= 0) {
            result.trackToBone[track] = exact;
            ++result.exactMatches;
            continue;
        }
        const auto insensitive = folded.find(lowerAscii(trackNames[track]));
        if (insensitive != folded.end()) {
            result.trackToBone[track] = insensitive->second;
            ++result.caseInsensitiveMatches;
            result.diagnostics.push_back({AnimationDiagnosticSeverity::Warning,
                "rig.case_fallback", "case-insensitive bone match for " + trackNames[track]});
        } else {
            result.missingTracks.push_back(trackNames[track]);
        }
    }
    if (!result.missingTracks.empty()) {
        result.diagnostics.push_back({AnimationDiagnosticSeverity::Warning,
            "rig.missing_tracks", std::to_string(result.missingTracks.size()) +
                " HKX tracks have no NIF bone"});
    }
    return result;
}

bool BehaviorGraphInstance::bind(const AnimationView& view, std::string& outError) {
    outError.clear();
    if (view.skeleton == nullptr || view.skeleton->bones.empty()) {
        outError = "animation view has no skeleton";
        return false;
    }
    m_view = &view;
    m_sampler.bindSkeleton(*view.skeleton);
    m_state = BehaviorGraphSnapshot{};
    m_bindWorld.resize(view.skeleton->bones.size());
    for (std::size_t index = 0; index < view.skeleton->bones.size(); ++index) {
        const Bone& bone = view.skeleton->bones[index];
        const odai::math::Matrix4 local = localBindMatrix(bone);
        m_bindWorld[index] = bone.parentIndex >= 0 ?
            m_bindWorld[static_cast<std::size_t>(bone.parentIndex)] * local : local;
    }
    return true;
}

const AnimationClip* BehaviorGraphInstance::clipForState(const std::string& state) const {
    if (m_view == nullptr) return nullptr;
    const auto mapped = m_view->stateClips.find(state);
    const std::string wanted = mapped == m_view->stateClips.end() ? state : mapped->second;
    const auto found = std::find_if(m_view->clips.begin(), m_view->clips.end(),
        [&](const AnimationClip& clip) { return clip.name == wanted; });
    return found == m_view->clips.end() ? nullptr : &*found;
}

std::string BehaviorGraphInstance::chooseState(const AnimationInputState& input) const {
    if (input.landed) return "landing";
    if (!input.grounded || input.falling) return "fall";
    if (input.equipping) return input.weaponDrawn ? "unequip" : "equip";
    if (input.attacking) return "attack";
    if (input.weaponDrawn && input.movementSpeed > 1.0f) return "combat_locomotion";
    if (input.weaponDrawn) return "combat_idle";
    if (input.movementSpeed > 1.0f) return "locomotion";
    return "idle";
}

AnimationStepOutput BehaviorGraphInstance::step(
    const AnimationInputState& input, float fixedDeltaSeconds) {
    AnimationStepOutput output;
    if (m_view == nullptr || m_view->skeleton == nullptr) {
        output.proceduralFallback = true;
        output.diagnostics.push_back({AnimationDiagnosticSeverity::Error,
            "graph.unbound", "behavior graph instance is not bound"});
        return output;
    }
    const float delta = std::clamp(fixedDeltaSeconds, 0.0f, 0.25f);
    const std::string nextState = chooseState(input);
    if (nextState != m_state.state) {
        output.events.push_back({"state_exit", m_state.state});
        m_state.state = nextState;
        m_state.stateTime = 0.0f;
        output.events.push_back({"state_enter", m_state.state});
    }
    if (input.landed || (!m_state.wasGrounded && input.grounded)) {
        output.events.push_back({"FootLeft", "landing"});
        output.events.push_back({"FootRight", "landing"});
    }
    output.events.insert(output.events.end(), m_state.queuedEvents.begin(), m_state.queuedEvents.end());
    output.events.insert(output.events.end(), input.events.begin(), input.events.end());
    m_state.queuedEvents.clear();

    const AnimationClip* clip = clipForState(m_state.state);
    if (clip == nullptr) clip = clipForState("idle");
    if (clip != nullptr) {
        const odai::math::Vector3 before = sampleRootTranslation(*clip, 0, m_state.stateTime);
        m_state.stateTime += delta;
        const odai::math::Vector3 after = sampleRootTranslation(*clip, 0, m_state.stateTime);
        if (input.animationDriven) output.desiredRootMotion = after - before;
        m_sampler.sample(*m_view->skeleton, *clip, m_state.stateTime, output.pose);
    } else {
        output.pose.assign(m_view->skeleton->bones.size(), odai::math::Matrix4::identity());
        output.proceduralFallback = true;
        output.diagnostics.push_back({AnimationDiagnosticSeverity::Warning,
            "graph.missing_clip", "no clip for state " + m_state.state + "; bind pose fallback"});
    }
    if (!m_view->supportedBehaviorGraph) {
        output.proceduralFallback = true;
        output.diagnostics.push_back({AnimationDiagnosticSeverity::Warning,
            "graph.unsupported", "HKX graph is unsupported; deterministic state fallback is active"});
    }
    output.activeState = m_state.state;
    applyFootIk(input, output);
    refreshSockets(output);
    m_state.wasGrounded = input.grounded;
    ++m_state.fixedTick;
    return output;
}

void BehaviorGraphInstance::applyFootIk(
    const AnimationInputState& input, AnimationStepOutput& output) const {
    if (!input.footIkEnabled || m_view == nullptr || m_view->skeleton == nullptr) return;
    const auto findAny = [&](std::initializer_list<const char*> names) {
        for (const char* name : names) {
            const int bone = m_view->skeleton->findBone(name);
            if (bone >= 0) return bone;
        }
        return -1;
    };
    const int left = findAny({"NPC L Foot [Lft ]", "L Foot", "LeftFoot"});
    const int right = findAny({"NPC R Foot [Rft ]", "R Foot", "RightFoot"});
    const int pelvis = findAny({"NPC Pelvis [Pelv]", "Pelvis"});
    constexpr float kMaxAnkleCorrection = 12.0f;
    const float leftOffset = std::clamp(input.leftFootIkOffset,
        -kMaxAnkleCorrection, kMaxAnkleCorrection);
    const float rightOffset = std::clamp(input.rightFootIkOffset,
        -kMaxAnkleCorrection, kMaxAnkleCorrection);
    const auto correct = [&](int bone, float offset) {
        if (bone >= 0 && static_cast<std::size_t>(bone) < output.pose.size()) {
            output.pose[static_cast<std::size_t>(bone)] =
                odai::math::Matrix4::translation({0.0f, offset, 0.0f}) *
                output.pose[static_cast<std::size_t>(bone)];
        }
    };
    correct(left, leftOffset);
    correct(right, rightOffset);
    // Lower the pelvis only; raising it from one high foot makes the opposite
    // ankle overextend. The correction is deliberately smaller than ankles.
    correct(pelvis, std::clamp(std::min(0.0f, 0.5f * (leftOffset + rightOffset)),
        -8.0f, 0.0f));
    if (left < 0 || right < 0) {
        output.diagnostics.push_back({AnimationDiagnosticSeverity::Warning,
            "foot_ik.missing_bones", "hkbFootIkControlsModifier could not find both foot bones"});
    }
}

void BehaviorGraphInstance::refreshSockets(AnimationStepOutput& output) const {
    if (m_view == nullptr || m_view->skeleton == nullptr) return;
    for (const std::string& name : m_view->socketBoneNames) {
        const int bone = m_view->skeleton->findBone(name);
        if (bone < 0) continue;
        const std::size_t index = static_cast<std::size_t>(bone);
        output.socketTransforms[name] = index < output.pose.size() && index < m_bindWorld.size() ?
            output.pose[index] * m_bindWorld[index] : m_bindWorld[index];
    }
}

void BehaviorGraphInstance::queueEvent(AnimationEvent event) {
    m_state.queuedEvents.push_back(std::move(event));
}

BehaviorGraphSnapshot BehaviorGraphInstance::snapshot() const { return m_state; }

bool BehaviorGraphInstance::restore(const BehaviorGraphSnapshot& snapshot, std::string& outError) {
    outError.clear();
    if (snapshot.stateTime < 0.0f || !std::isfinite(snapshot.stateTime)) {
        outError = "invalid behavior graph state time";
        return false;
    }
    m_state = snapshot;
    return true;
}

AnimationStepOutput BehaviorGraphInstance::interpolate(
    const AnimationStepOutput& previous, const AnimationStepOutput& current, float alpha) {
    if (previous.pose.size() != current.pose.size()) return current;
    AnimationStepOutput result = current;
    const float t = odai::math::saturate(alpha);
    for (std::size_t matrix = 0; matrix < result.pose.size(); ++matrix) {
        for (std::size_t element = 0; element < 16u; ++element) {
            result.pose[matrix].m[element] = odai::math::lerp(
                previous.pose[matrix].m[element], current.pose[matrix].m[element], t);
        }
    }
    for (auto& [name, transform] : result.socketTransforms) {
        const auto old = previous.socketTransforms.find(name);
        if (old == previous.socketTransforms.end()) continue;
        for (std::size_t element = 0; element < 16u; ++element) {
            transform.m[element] = odai::math::lerp(old->second.m[element], transform.m[element], t);
        }
    }
    result.desiredRootMotion = odai::math::lerp(
        previous.desiredRootMotion, current.desiredRootMotion, t);
    return result;
}

}  // namespace odai::anim
