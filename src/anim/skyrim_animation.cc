#include "anim/skyrim_animation.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <functional>

namespace odai::anim {
namespace {

std::string lowerAscii(std::string text) {
  for (char &ch : text)
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  return text;
}

odai::math::Matrix4 localBindMatrix(const Bone &bone) {
  return odai::math::Matrix4::translation(bone.localTranslation) *
         odai::math::toMatrix(bone.localRotation) *
         odai::math::Matrix4::scale(bone.localScale);
}

odai::math::Vector3 sampleRootTranslation(const AnimationClip &clip, int root,
                                          float time) {
  const BoneTrack *track = nullptr;
  for (const BoneTrack &candidate : clip.tracks) {
    if (candidate.boneIndex == root) {
      track = &candidate;
      break;
    }
  }
  if (track == nullptr || track->translationKeys.empty())
    return {};
  float t = time;
  if (clip.duration > 0.0f && clip.loop) {
    t = std::fmod(t, clip.duration);
    if (t < 0.0f)
      t += clip.duration;
  } else {
    t = std::clamp(t, 0.0f, clip.duration);
  }
  const auto &keys = track->translationKeys;
  if (t <= keys.front().time)
    return keys.front().value;
  for (std::size_t index = 0; index + 1u < keys.size(); ++index) {
    if (t <= keys[index + 1u].time) {
      const float span = keys[index + 1u].time - keys[index].time;
      const float alpha = span > 0.0f ? (t - keys[index].time) / span : 0.0f;
      return odai::math::lerp(keys[index].value, keys[index + 1u].value, alpha);
    }
  }
  return keys.back().value;
}

odai::math::Vector3 rootDisplacement(const AnimationClip &clip, int root,
                                     float before, float after) {
  auto delta = sampleRootTranslation(clip, root, after) -
               sampleRootTranslation(clip, root, before);
  if (clip.loop && clip.duration > 0.0f) {
    const float loops =
        std::floor(after / clip.duration) - std::floor(before / clip.duration);
    for (const auto &track : clip.tracks) {
      if (track.boneIndex == root && !track.translationKeys.empty()) {
        delta = delta + (track.translationKeys.back().value -
                         track.translationKeys.front().value) *
                            loops;
        break;
      }
    }
  }
  return delta;
}

void appendCrossedAnnotations(const AnimationClip &clip, float before,
                              float after, std::vector<AnimationEvent> &out) {
  if (clip.annotations.empty() || !(after > before) || clip.duration <= 0.0f)
    return;
  if (!clip.loop) {
    for (const AnimationAnnotation &annotation : clip.annotations) {
      if (annotation.time > before && annotation.time <= after) {
        out.push_back({annotation.name, {}});
      }
    }
    return;
  }
  const std::int64_t firstCycle =
      static_cast<std::int64_t>(std::floor(before / clip.duration));
  const std::int64_t lastCycle =
      static_cast<std::int64_t>(std::floor(after / clip.duration));
  for (std::int64_t cycle = firstCycle; cycle <= lastCycle; ++cycle) {
    const float base = static_cast<float>(cycle) * clip.duration;
    for (const AnimationAnnotation &annotation : clip.annotations) {
      if (annotation.firstCycleOnly && cycle != 0)
        continue;
      const float eventTime = base + annotation.time;
      if (eventTime > before && eventTime <= after) {
        out.push_back({annotation.name, {}});
      }
    }
  }
}

} // namespace

float RigBindingResult::coverage() const {
  return trackToBone.empty()
             ? 1.0f
             : static_cast<float>(exactMatches + caseInsensitiveMatches) /
                   static_cast<float>(trackToBone.size());
}

RigBindingResult bindTracksByName(std::span<const std::string> trackNames,
                                  const Skeleton &skeleton) {
  RigBindingResult result;
  result.trackToBone.assign(trackNames.size(), -1);
  std::unordered_map<std::string, int> folded;
  for (std::size_t index = 0; index < skeleton.bones.size(); ++index) {
    folded.try_emplace(lowerAscii(skeleton.bones[index].name),
                       static_cast<int>(index));
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
      result.diagnostics.push_back(
          {AnimationDiagnosticSeverity::Warning, "rig.case_fallback",
           "case-insensitive bone match for " + trackNames[track]});
    } else {
      result.missingTracks.push_back(trackNames[track]);
    }
  }
  if (!result.missingTracks.empty()) {
    result.diagnostics.push_back({AnimationDiagnosticSeverity::Warning,
                                  "rig.missing_tracks",
                                  std::to_string(result.missingTracks.size()) +
                                      " HKX tracks have no NIF bone"});
  }
  return result;
}

bool BehaviorGraphInstance::bind(const AnimationView &view,
                                 std::string &outError) {
  outError.clear();
  if (view.skeleton == nullptr || view.skeleton->bones.empty()) {
    outError = "animation view has no skeleton";
    return false;
  }
  for (std::size_t index = 0; index < view.skeleton->bones.size(); ++index) {
    const int parent = view.skeleton->bones[index].parentIndex;
    if (parent < -1 || parent >= static_cast<int>(index)) {
      outError = "animation skeleton must be parent-before-child";
      return false;
    }
  }
  if (view.executionMode == AnimationExecutionMode::Havok && view.behavior &&
      view.behavior->executable()) {
    for (const auto &node : view.behavior->graph.nodes) {
      if (node.kind != HkxBehaviorNodeKind::Clip)
        continue;
      const auto clip = std::find_if(
          view.clips.begin(), view.clips.end(),
          [&](const auto &value) { return value.name == node.assetPath; });
      if (clip == view.clips.end() || clip->loop != (node.playbackMode == 1) ||
          !std::isfinite(clip->duration) || clip->duration <= 0) {
        outError =
            "authored graph clip is missing or has incompatible playback: " +
            node.assetPath;
        return false;
      }
    }
  }
  m_view = &view;
  if (view.inverseBindMatrices.empty()) {
    m_sampler.bindSkeleton(*view.skeleton);
  } else {
    m_sampler.bindSkeleton(*view.skeleton, view.inverseBindMatrices);
  }
  m_state = BehaviorGraphSnapshot{};
  m_poseGraph.reset();
  m_inertializer.reset();
  m_state.executionMode = view.executionMode;
  if (view.behavior)
    m_state.variables = view.behavior->graph.variableDefaults;
  m_state.graphFingerprint = view.sourceFingerprint;
  m_bindWorld.resize(view.skeleton->bones.size());
  for (std::size_t index = 0; index < view.skeleton->bones.size(); ++index) {
    const Bone &bone = view.skeleton->bones[index];
    const odai::math::Matrix4 local = localBindMatrix(bone);
    m_bindWorld[index] =
        bone.parentIndex >= 0
            ? m_bindWorld[static_cast<std::size_t>(bone.parentIndex)] * local
            : local;
  }
  return true;
}

const AnimationClip *
BehaviorGraphInstance::clipByName(const std::string &name) const {
  if (!m_view || name.empty())
    return nullptr;
  const auto found =
      std::find_if(m_view->clips.begin(), m_view->clips.end(),
                   [&](const auto &clip) { return clip.name == name; });
  return found == m_view->clips.end() ? nullptr : &*found;
}

const AnimationClip *
BehaviorGraphInstance::clipForState(const std::string &state) const {
  if (m_view == nullptr)
    return nullptr;
  if (state == m_state.state && !m_state.selectedClip.empty())
    if (const auto *selected = clipByName(m_state.selectedClip))
      return selected;
  const auto mapped = m_view->stateClips.find(state);
  const std::string wanted =
      mapped == m_view->stateClips.end() ? state : mapped->second;
  const auto found = std::find_if(
      m_view->clips.begin(), m_view->clips.end(),
      [&](const AnimationClip &clip) { return clip.name == wanted; });
  if (found != m_view->clips.end())
    return &*found;
  const std::string fallback =
      state.starts_with("landing_") ? "landing"
      : (state.starts_with("takeoff") || state.starts_with("ascending") ||
         state.starts_with("apex") || state.starts_with("jump_"))
          ? "jump"
      : state.starts_with("falling") ? "fall"
                                     : "";
  if (!fallback.empty()) {
    const auto base = m_view->stateClips.find(fallback);
    return clipByName(base == m_view->stateClips.end() ? fallback
                                                       : base->second);
  }
  return nullptr;
}

std::string
BehaviorGraphInstance::chooseState(const AnimationInputState &input) const {
  const auto nativeState = [&](const std::string &state) {
    return m_view->nativeProgram &&
           std::any_of(m_view->nativeProgram->rules.begin(),
                       m_view->nativeProgram->rules.end(),
                       [&](const auto &rule) { return rule.state == state; });
  };
  const auto equippedState = [&](const std::string &state) {
    const auto qualified = state + "_" + input.weaponStyle;
    return !input.weaponStyle.empty() && clipForState(qualified) ? qualified
                                                                 : state;
  };
  const auto &velocity = input.localVelocity;
  const std::string direction = std::abs(velocity.x) > std::abs(velocity.z)
                                    ? (velocity.x < 0 ? "left" : "right")
                                    : (velocity.z > 1.f ? "back" : "forward");
  const auto directionalState = [&](const std::string &base) {
    const auto state = base + "_" + direction;
    return m_view->stateClips.contains(state) || clipByName(state) ||
                   nativeState(state)
               ? state
               : base;
  };
  if (input.dead)
    return "death";
  // Request pulses start actions; clip completion or interruption ends them.
  if (const auto *active = clipForState(m_state.state);
      active && !active->loop && m_state.stateTime < active->duration &&
      !(m_state.state.starts_with("landing") &&
        (input.jumpPreparing || !input.grounded ||
         (input.movementSpeed > 1.0f && m_state.stateTime >= 0.18f))) &&
      (m_state.state.starts_with("attack") ||
       m_state.state.starts_with("equip") ||
       m_state.state.starts_with("unequip") ||
       m_state.state.starts_with("landing") || m_state.state == "stagger" ||
       m_state.state == "get_up"))
    return m_state.state;
  if (input.swimming)
    return input.movementSpeed > 1.0f ? directionalState("swim") : "swim_idle";
  if (input.jumpPreparing && input.grounded)
    return "jump";
  if (input.landed) {
    const auto impact = "landing_" + input.landingSeverity;
    return m_view->stateClips.contains(impact) || clipByName(impact) ||
                   nativeState(impact)
               ? impact
               : "landing";
  }
  if (!input.grounded && m_view->nativeProgram &&
      !m_state.selectedClip.empty() && clipByName(m_state.selectedClip)) {
    for (const auto &rule : m_view->nativeProgram->rules) {
      if (rule.id == m_state.selectedRule && rule.holdUntilLanding)
        return m_state.state;
    }
  }
  if (!input.grounded && !input.jumpPhase.empty()) {
    const auto phase = directionalState(input.jumpPhase);
    if (m_view->stateClips.contains(phase) || clipByName(phase) ||
        nativeState(phase))
      return phase;
  }
  if (!input.grounded && input.verticalVelocity > 1.0f)
    return directionalState("jump");
  if (!input.grounded || input.falling)
    return "fall";
  if (input.equipping)
    return equippedState(input.weaponDrawn ? "unequip" : "equip");
  if (input.attacking)
    return equippedState("attack");
  if (input.blocking)
    return equippedState("block");
  if (input.sneaking)
    return input.movementSpeed > 1.0f ? directionalState("sneak")
                                      : "sneak_idle";
  if (input.sprinting && input.movementSpeed > 1.0f)
    return "sprint";
  if (input.weaponDrawn && input.movementSpeed > 1.0f &&
      clipForState("combat_locomotion"))
    return "combat_locomotion";
  if (input.weaponDrawn && input.movementSpeed <= 1.0f)
    return equippedState("combat_idle");
  if (input.movementSpeed > 1.0f) {
    const std::string prefix = input.running ? "run" : "walk";
    const auto directional = prefix + "_" + direction;
    if (input.weaponDrawn &&
        clipForState(directional + "_" + input.weaponStyle))
      return directional + "_" + input.weaponStyle;
    if (clipForState(directional))
      return directional;
    if (input.running && clipForState("run_forward"))
      return "run_forward";
    return "locomotion";
  }
  if (input.talking && clipForState("talk") &&
      (!m_state.wasTalking ||
       (m_state.state == "talk" &&
        m_state.stateTime < clipForState("talk")->duration)))
    return "talk";
  if (std::abs(input.turnRateRadiansPerSecond) > 0.3f) {
    const std::string turn =
        input.turnRateRadiansPerSecond > 0 ? "turn_left" : "turn_right";
    if (clipForState(turn))
      return turn;
  }
  return "idle";
}

std::string BehaviorGraphInstance::chooseAuthoredState(
    std::span<const AnimationEvent> events, float &duration, bool &transitioned,
    std::vector<AnimationDiagnostic> &diagnostics) {
  const auto &graph = m_view->behavior->graph;
  const auto &nodes = graph.nodes;
  const auto targetNode = [&](const HkxBehaviorNode &machine,
                              std::int32_t state) {
    return *std::find_if(
        machine.children.begin(), machine.children.end(),
        [&](auto child) { return nodes[child].stateId == state; });
  };
  std::function<void(std::uint32_t)> reset = [&](std::uint32_t index) {
    m_state.activeStates.erase(index);
    for (auto child : nodes[index].children)
      reset(child);
  };
  std::function<std::string(std::uint32_t)> select =
      [&](std::uint32_t index) -> std::string {
    const auto &node = nodes[index];
    if (node.kind == HkxBehaviorNodeKind::Clip)
      return node.assetPath;
    if (node.kind == HkxBehaviorNodeKind::ManualSelector) {
      std::int32_t selected = node.selectedGeneratorIndex;
      for (const auto &binding : node.bindings) {
        const auto &name = graph.variableNames[binding.variableIndex];
        const auto variable = m_state.variables.find(name);
        if (variable == m_state.variables.end() ||
            !std::isfinite(variable->second) || variable->second < 0 ||
            variable->second >= static_cast<float>(node.children.size()) ||
            std::trunc(variable->second) != variable->second) {
          diagnostics.push_back({AnimationDiagnosticSeverity::Error,
                                 "graph.invalid_selector",
                                 node.name + ": invalid selector variable " +
                                     name + "; retaining active child"});
          const auto previous = m_state.activeStates.find(index);
          if (previous != m_state.activeStates.end())
            selected = previous->second;
        } else
          selected = static_cast<std::int32_t>(variable->second);
      }
      auto [entry, inserted] =
          m_state.activeStates.try_emplace(index, selected);
      if (!inserted && entry->second != selected) {
        reset(node.children[entry->second]);
        entry->second = selected;
        transitioned = true;
        duration = 0;
      }
      return select(node.children[selected]);
    }
    if (node.kind == HkxBehaviorNodeKind::StateMachine) {
      auto [entry, inserted] =
          m_state.activeStates.try_emplace(index, node.startStateId);
      auto current = targetNode(node, entry->second);
      const HkxBehaviorTransition *selected = nullptr;
      const auto consider = [&](std::uint32_t ownerIndex) {
        const auto &owner = nodes[ownerIndex];
        for (std::size_t ruleIndex = 0; ruleIndex < owner.transitions.size();
             ++ruleIndex) {
          const auto &rule = owner.transitions[ruleIndex];
          if (rule.flags & HkxBehaviorTransition::Disabled)
            continue;
          if (rule.toStateId == entry->second &&
              (ownerIndex != index ||
               !(rule.flags &
                 HkxBehaviorTransition::AllowWildcardSelfTransition)))
            continue;
          if (rule.eventId >= 0 &&
              std::none_of(
                  events.begin(), events.end(), [&](const auto &event) {
                    return event.name == graph.eventNames[rule.eventId];
                  }))
            continue;
          if (rule.hasCondition &&
              !(rule.flags & HkxBehaviorTransition::DisableCondition)) {
            const auto condition =
                m_view->behavior->conditions.find({ownerIndex, ruleIndex});
            if (condition == m_view->behavior->conditions.end())
              continue;
            const auto value = condition->second.evaluate(m_state.variables);
            if (!value)
              diagnostics.push_back({AnimationDiagnosticSeverity::Error,
                                     "graph.condition_evaluation",
                                     owner.name +
                                         ": missing/non-finite variable or "
                                         "invalid arithmetic in " +
                                         rule.conditionExpression});
            if (!value || *value == 0)
              continue;
          }
          // Equal priorities retain source order, local before wildcard.
          if (!selected || rule.priority > selected->priority)
            selected = &rule;
        }
      };
      consider(current);
      consider(index);
      if (selected) {
        reset(current);
        entry->second = selected->toStateId;
        current = targetNode(node, entry->second);
        transitioned = true;
        duration = selected->hasEffect
                       ? nodes[selected->effectNode].transitionDuration
                       : 0.0f;
      }
      return select(current);
    }
    return select(node.children.front());
  };
  return select(graph.rootNode);
}

AnimationStepOutput
BehaviorGraphInstance::step(const AnimationInputState &input,
                            float fixedDeltaSeconds) {
  AnimationStepOutput output;
  output.resetHistory = input.teleported;
  if (input.teleported || input.ragdollActive || input.dead || !input.grounded)
    m_state.footPlants.clear();
  auto postProcess = std::make_shared<PosePostProcess>();
  if (m_view == nullptr || m_view->skeleton == nullptr) {
    output.proceduralFallback = true;
    output.diagnostics.push_back({AnimationDiagnosticSeverity::Error,
                                  "graph.unbound",
                                  "behavior graph instance is not bound"});
    return output;
  }
  const float delta = std::isfinite(fixedDeltaSeconds)
                          ? std::clamp(fixedDeltaSeconds, 0.0f, 0.25f)
                          : 0.0f;
  std::vector<AnimationEvent> requested = m_state.queuedEvents;
  requested.insert(requested.end(), input.events.begin(), input.events.end());
  for (const auto &[name, value] : input.variables)
    if (std::isfinite(value))
      m_state.variables[name] = value;
  output.executionMode = m_view->executionMode;
  output.authoredGraphExecuted =
      m_view->executionMode == AnimationExecutionMode::Havok &&
      m_view->behavior && m_view->behavior->executable();
  float transitionDuration = 0.16f;
  bool transitioned = false;
  std::string nextState =
      output.authoredGraphExecuted
          ? chooseAuthoredState(requested, transitionDuration, transitioned,
                                output.diagnostics)
          : chooseState(input);
  if (!output.authoredGraphExecuted && !input.dead) {
    for (const auto &event : requested) {
      const std::string state = event.name == "staggerStart"  ? "stagger"
                                : event.name == "attackStart" ? "attack"
                                : event.name == "getUp"       ? "get_up"
                                                              : "";
      if (!state.empty() && clipForState(state))
        nextState = state;
      if (event.name == "DrawStart" || event.name == "SheatheStart") {
        std::string equip = event.name == "DrawStart" ? "equip" : "unequip";
        if (clipForState(equip + "_" + input.weaponStyle))
          equip += "_" + input.weaponStyle;
        if (clipForState(equip))
          nextState = equip;
      }
    }
  }
  if (!output.authoredGraphExecuted &&
      (nextState == "jump" || nextState == "fall" || nextState == "landing" ||
       m_state.state.starts_with("landing")))
    transitionDuration = 0.08f;
  if (!output.authoredGraphExecuted && !input.sharedState.empty())
    nextState = input.sharedState;
  if (!output.authoredGraphExecuted && input.grounded &&
      (nextState == "idle" || nextState == "combat_idle" ||
       nextState == "sneak_idle") &&
      (m_state.state.starts_with("walk") || m_state.state.starts_with("run") ||
       m_state.state == "locomotion" || m_state.state == "sprint" ||
       m_state.state == "sneak"))
    transitionDuration = 0.08f;
  const NativeAnimationRule *selected = nullptr;
  std::string selectedClip, selectedRule, selectedProvider;
  if (!output.authoredGraphExecuted && m_view->nativeProgram &&
      !m_view->nativeProgram->rules.empty()) {
    auto context = m_view->selectorDefaults;
    for (const auto &[key, value] : input.selectorContext.values)
      context.values[key] = value;
    context.tags.insert(input.selectorContext.tags.begin(),
                        input.selectorContext.tags.end());
    context.values["movement"] = nextState;
    context.values["speed"] = static_cast<double>(input.movementSpeed);
    context.values["vertical_speed"] =
        static_cast<double>(input.verticalVelocity);
    context.values["weapon"] = input.weaponStyle;
    context.values["grounded"] = input.grounded;
    context.values["jump_phase"] = input.jumpPhase;
    context.values["landing_severity"] = input.landingSeverity;
    context.values["impact_speed"] =
        static_cast<double>(input.landingImpactMetres);
    context.values["weapon_drawn"] = input.weaponDrawn;
    context.values["sprinting"] = input.sprinting;
    context.values["swimming"] = input.swimming;
    context.values["stance"] =
        std::string(input.sneaking ? "sneak" : "standing");
    selected = m_view->nativeProgram->select(nextState, context);
    if (!input.grounded && !input.landed && nextState == m_state.state &&
        clipByName(m_state.selectedClip)) {
      for (const auto &rule : m_view->nativeProgram->rules) {
        if (rule.id == m_state.selectedRule && rule.holdUntilLanding) {
          selected = &rule;
          break;
        }
      }
    }
    if (selected && !selected->variants.empty()) {
      selectedRule = selected->id;
      selectedProvider = selected->provider;
      const bool retain = nextState == m_state.state &&
                          selectedRule == m_state.selectedRule &&
                          selectedProvider == m_state.selectedProvider &&
                          !m_state.selectedClip.empty();
      selectedClip = retain ? m_state.selectedClip
                            : selected
                                  ->variants[chooseNativeVariant(
                                      *selected, m_state.randomState)]
                                  .clip;
      if (!clipByName(selectedClip)) {
        output.fallbackReason = "replacement clip unavailable: " + selectedClip;
      }
      transitionDuration = selected->blendSeconds;
    }
  }
  const bool selectionChanged = selectedClip != m_state.selectedClip ||
                                selectedRule != m_state.selectedRule ||
                                selectedProvider != m_state.selectedProvider;
  if (nextState != m_state.state || transitioned || selectionChanged) {
    m_poseGraph.reset();
    m_state.poseGraphState.clear();
    m_state.warpTranslationUsed = 0;
    m_state.warpWindow.clear();
    output.events.push_back({"state_exit", m_state.state});
    if (const auto *previous = clipForState(m_state.state))
      m_state.previousClip = previous->name;
    else
      m_state.previousClip.clear();
    m_state.previousState = m_state.state;
    m_state.previousStateTime = m_state.stateTime;
    m_state.transitionElapsed = 0.0f;
    // Authored graphs supply their effect duration. Recovery states use
    // a short local-TRS transition, persisted at the fixed simulation tick.
    m_state.transitionDuration = transitionDuration;
    float phaseTime = 0.0f;
    const auto gait = [](const std::string &state) {
      return state != "sneak_idle" && state != "swim_idle" &&
             (state == "locomotion" || state == "sprint" || state == "sneak" ||
              state == "swim" || state.starts_with("sneak_") ||
              state.starts_with("swim_") || state.starts_with("walk_") ||
              state.starts_with("run_"));
    };
    if (!output.authoredGraphExecuted && gait(m_state.state) &&
        gait(nextState)) {
      const auto *previousClip = clipForState(m_state.state);
      const auto mapped = m_view->stateClips.find(nextState);
      const auto *nextClip =
          !selectedClip.empty()
              ? clipByName(selectedClip)
              : clipByName(mapped == m_view->stateClips.end() ? nextState
                                                              : mapped->second);
      if (previousClip && nextClip && previousClip->loop && nextClip->loop &&
          previousClip->duration > 0 && nextClip->duration > 0)
        phaseTime = std::fmod(m_state.stateTime / previousClip->duration, 1.f) *
                    nextClip->duration;
    }
    m_state.selectedClip = selectedClip;
    m_state.selectedRule = selectedRule;
    m_state.selectedProvider = selectedProvider;
    ++m_state.actionIdentity;
    m_state.layerTimes.clear();
    m_state.state = nextState;
    m_state.stateTime = phaseTime;
    output.events.push_back({"state_enter", m_state.state});
  }
  if (input.landed || (!m_state.wasGrounded && input.grounded)) {
    output.events.push_back({"FootLeft", "landing"});
    output.events.push_back({"FootRight", "landing"});
  }
  output.events.insert(output.events.end(), m_state.queuedEvents.begin(),
                       m_state.queuedEvents.end());
  output.events.insert(output.events.end(), input.events.begin(),
                       input.events.end());
  m_state.queuedEvents.clear();

  const AnimationClip *clip = clipForState(m_state.state);
  auto adaptedPacket = std::make_shared<PoseEvaluationPacket>();
  const auto emitClip = [&](const AnimationClip &source, float time) {
    PoseGraphInstruction instruction;
    instruction.clip = source.name;
    instruction.time = time;
    const auto index =
        static_cast<std::uint32_t>(adaptedPacket->instructions.size());
    adaptedPacket->instructions.push_back(std::move(instruction));
    return index;
  };
  if (clip == nullptr) {
    output.proceduralFallback = true;
    output.fallbackReason = "state clip unavailable; using idle";
    output.diagnostics.push_back(
        {AnimationDiagnosticSeverity::Warning, "graph.missing_state_clip",
         "no clip for " + m_state.state + "; using idle"});
    clip = clipForState("idle");
  }
  if (clip != nullptr) {
    const bool locomotionState =
        m_state.state == "locomotion" || m_state.state == "sprint" ||
        m_state.state == "combat_locomotion" ||
        m_state.state.starts_with("walk_") || m_state.state.starts_with("run_");
    float playbackRate =
        locomotionState ? std::clamp(input.locomotionPlaybackRate, 0.1f, 4.0f)
                        : 1.0f;
    if (output.authoredGraphExecuted) {
      for (const auto &node : m_view->behavior->graph.nodes)
        if (node.kind == HkxBehaviorNodeKind::Clip &&
            node.assetPath == m_state.state) {
          playbackRate = node.playbackSpeed;
          break;
        }
    }
    if (!output.authoredGraphExecuted && !input.sharedState.empty() &&
        std::isfinite(input.sharedStateTime)) {
      m_state.stateTime =
          std::max(0.f, input.sharedStateTime - delta * playbackRate);
      m_state.actionIdentity = input.sharedActionIdentity;
    }
    const float beforeTime = m_state.stateTime;
    m_state.stateTime += delta * playbackRate;
    if ((nextState != m_state.previousState || transitioned) &&
        beforeTime == 0.0f && delta > 0.0f) {
      for (const auto &annotation : clip->annotations)
        if (annotation.time == 0.0f)
          output.clipEvents.push_back({annotation.name, {}});
    }
    appendCrossedAnnotations(*clip, beforeTime, m_state.stateTime,
                             output.clipEvents);
    int root = clip->extractedMotionBone >= 0
                   ? clip->extractedMotionBone
                   : m_view->skeleton->findBone("NPC Root [Root]");
    if (root < 0)
      root = 0;
    if (input.animationDriven && (output.authoredGraphExecuted || !selected ||
                                  selected->animationDriven)) {
      const auto local =
          rootDisplacement(*clip, root, beforeTime, m_state.stateTime);
      const float c = std::cos(input.actorYawRadians),
                  s = std::sin(input.actorYawRadians);
      output.desiredRootMotion = {c * local.x + s * local.z, local.y,
                                  -s * local.x + c * local.z};
    }
    output.activeClip = clip->name;
    output.actionActive = !clip->loop && m_state.stateTime < clip->duration;
    const auto *idle = clipForState("idle");
    if (clip->additive && idle && idle != clip) {
      m_sampler.sampleLayered(*m_view->skeleton, *idle,
                              static_cast<float>(m_state.fixedTick) / 60.0f,
                              *clip, m_state.stateTime, 1.0f, {}, output.pose);
    } else {
      m_sampler.sample(*m_view->skeleton, *clip, m_state.stateTime,
                       output.pose);
      adaptedPacket->output = emitClip(*clip, m_state.stateTime);
    }
    if (!m_state.previousState.empty() && m_state.transitionDuration > 0.0f &&
        m_state.transitionElapsed < m_state.transitionDuration) {
      const AnimationClip *previous = clipByName(m_state.previousClip);
      if (previous != nullptr) {
        const float linear = std::clamp(
            m_state.transitionElapsed / m_state.transitionDuration, 0.0f, 1.0f);
        const float blend = linear * linear * (3.0f - 2.0f * linear);
        m_sampler.sampleBlended(*m_view->skeleton, *previous,
                                m_state.previousStateTime +
                                    m_state.transitionElapsed,
                                *clip, m_state.stateTime, blend, output.pose);
        adaptedPacket->instructions.clear();
        PoseGraphInstruction transition;
        transition.kind = PoseGraphNode::Kind::Blend;
        transition.inputs = {emitClip(*previous, m_state.previousStateTime +
                                                     m_state.transitionElapsed),
                             emitClip(*clip, m_state.stateTime)};
        transition.weight = blend;
        adaptedPacket->output =
            static_cast<std::uint32_t>(adaptedPacket->instructions.size());
        adaptedPacket->instructions.push_back(std::move(transition));
      }
      m_state.transitionElapsed += delta;
      if (m_state.transitionElapsed >= m_state.transitionDuration) {
        m_state.previousState.clear();
        m_state.previousStateTime = 0.0f;
      }
    }
  } else {
    output.pose.assign(m_view->skeleton->bones.size(),
                       odai::math::Matrix4::identity());
    output.proceduralFallback = true;
    output.fallbackReason = "no usable clip; bind pose";
    output.diagnostics.push_back(
        {AnimationDiagnosticSeverity::Warning, "graph.missing_clip",
         "no clip for state " + m_state.state + "; bind pose fallback"});
  }
  if (selected && clip &&
      (!selected->blendSamples.empty() || !selected->layers.empty())) {
    std::vector<WeightedAnimationPose> base;
    double total = 0;
    for (const auto &point : selected->blendSamples) {
      const auto *sample = clipByName(point.clip);
      if (!sample || sample->duration <= 0)
        continue;
      const double x = static_cast<double>(input.localVelocity.x) - point.x;
      const double z = static_cast<double>(input.localVelocity.z) - point.z;
      const double weight = 1.0 / std::max(1.e-6, x * x + z * z);
      if (!std::isfinite(weight))
        continue;
      base.push_back({sample,
                      m_state.stateTime / clip->duration * sample->duration,
                      static_cast<float>(weight)});
      total += weight;
    }
    if (base.empty())
      base.push_back({clip, m_state.stateTime, 1});
    else
      for (auto &sample : base)
        sample.weight = static_cast<float>(sample.weight / total);
    if (const auto *previous = clipByName(m_state.previousClip);
        previous && !m_state.previousState.empty() &&
        m_state.transitionDuration > 0 &&
        m_state.transitionElapsed < m_state.transitionDuration) {
      const float linear =
          m_state.transitionElapsed / m_state.transitionDuration;
      const float blend = linear * linear * (3.f - 2.f * linear);
      for (auto &sample : base)
        sample.weight *= blend;
      base.insert(base.begin(),
                  {previous,
                   m_state.previousStateTime + m_state.transitionElapsed,
                   1.f - blend});
    }
    std::vector<AnimationPoseLayer> layers;
    for (const auto &configured : selected->layers) {
      auto &time = m_state.layerTimes[configured.id];
      time += delta;
      AnimationPoseLayer layer;
      layer.clip = clipByName(configured.clip);
      layer.reference = clipByName(configured.referenceClip);
      layer.time = time;
      layer.weight = configured.weight;
      layer.additive = configured.additive;
      layer.mask.resize(m_view->skeleton->bones.size());
      bool missingBone = false;
      for (const auto &name : configured.bones) {
        int index = m_view->skeleton->findBone(name);
        if (name.starts_with("role:") && m_view->humanoidRig) {
          const auto found = m_view->humanoidRig->roles.find(name.substr(5));
          if (found != m_view->humanoidRig->roles.end())
            index = found->second;
        }
        if (index >= 0 && static_cast<std::size_t>(index) < layer.mask.size())
          layer.mask[index] = 1;
        else
          missingBone = true;
      }
      if (!layer.clip || (layer.additive && !layer.reference) || missingBone) {
        output.diagnostics.push_back({AnimationDiagnosticSeverity::Warning,
                                      "native.layer_unavailable",
                                      configured.id});
        continue;
      }
      layers.push_back(std::move(layer));
    }
    m_sampler.sampleComposed(*m_view->skeleton, base, layers, output.pose);
    adaptedPacket->instructions.clear();
    PoseGraphInstruction composition;
    composition.kind = PoseGraphNode::Kind::Blend;
    for (const auto &sample : base) {
      composition.inputs.push_back(emitClip(*sample.clip, sample.time));
      composition.weights.push_back(sample.weight);
    }
    adaptedPacket->output =
        static_cast<std::uint32_t>(adaptedPacket->instructions.size());
    adaptedPacket->instructions.push_back(std::move(composition));
    for (const auto &layer : layers) {
      PoseGraphInstruction instruction;
      instruction.kind = PoseGraphNode::Kind::Layer;
      instruction.inputs = {adaptedPacket->output,
                            emitClip(*layer.clip, layer.time)};
      instruction.mask = layer.mask;
      instruction.weight = layer.weight;
      instruction.additive = layer.additive;
      if (layer.reference)
        instruction.reference = layer.reference->name;
      adaptedPacket->output =
          static_cast<std::uint32_t>(adaptedPacket->instructions.size());
      adaptedPacket->instructions.push_back(std::move(instruction));
    }
  }
  // Lower the existing selector's resolved composition into the same pose
  // packet as v2. Its weights, clocks and event authority remain unchanged.
  if ((!selected || !selected->graph) && m_view->humanoidRig &&
      m_view->executionMode == AnimationExecutionMode::Native &&
      !adaptedPacket->instructions.empty()) {
    std::string error;
    if (evaluatePosePacket(*adaptedPacket, *m_view->skeleton, m_view->clips,
                           output.localPose, error)) {
      m_sampler.paletteFromLocal(*m_view->skeleton, output.localPose,
                                 output.pose);
      output.evaluationPacket = std::move(adaptedPacket);
    }
  }
  if (selected && selected->graph && !output.authoredGraphExecuted) {
    std::map<std::string, PoseParameter> parameters =
        m_view->selectorDefaults.values;
    for (const auto &[name, value] : input.selectorContext.values)
      parameters[name] = value;
    for (const auto &[name, value] : input.variables)
      parameters[name] = static_cast<double>(value);
    parameters["speed"] = static_cast<double>(input.movementSpeed);
    parameters["velocity_x"] = static_cast<double>(input.localVelocity.x);
    parameters["velocity_z"] = static_cast<double>(input.localVelocity.z);
    parameters["grounded"] = input.grounded;
    parameters["attacking"] = input.attacking;
    parameters["weapon_drawn"] = input.weaponDrawn;
    auto packet = std::make_shared<PoseEvaluationPacket>();
    LocalPose local;
    std::string error;
    const auto beforeGraph = m_poseGraph.save();
    if (m_poseGraph.advance(*selected->graph, *m_view->skeleton,
                            m_view->humanoidRig.get(), parameters,
                            m_view->clips, delta, *packet, error) &&
        evaluatePosePacket(*packet, *m_view->skeleton, m_view->clips, local,
                           error)) {
      if (input.teleported || input.ragdollActive)
        m_inertializer.reset();
      LocalPose incomingPrevious;
      if (packet->discontinuity || selectionChanged) {
        auto previousPacket = *packet;
        for (auto &instruction : previousPacket.instructions)
          instruction.time = instruction.previousTime;
        (void)evaluatePosePacket(previousPacket, *m_view->skeleton,
                                 m_view->clips, incomingPrevious, error);
      }
      local = m_inertializer.evaluate(
          local, packet->discontinuity || selectionChanged,
          packet->transitionDuration >= 0 ? packet->transitionDuration
                                          : selected->blendSeconds,
          delta, incomingPrevious.empty() ? nullptr : &incomingPrevious);
      *postProcess = m_inertializer.evaluationParameters();
      output.localPose = local;
      m_sampler.paletteFromLocal(*m_view->skeleton, local, output.pose);
      output.clipEvents.clear();
      output.desiredRootMotion = {};
      if (const auto *authority = clipByName(packet->eventClip)) {
        appendCrossedAnnotations(*authority, packet->eventBefore,
                                 packet->eventAfter, output.clipEvents);
        if (packet->discontinuity && packet->eventBefore == 0 && delta > 0)
          for (const auto &annotation : authority->annotations)
            if (annotation.time == 0)
              output.clipEvents.push_back({annotation.name, {}});
        output.activeClip = authority->name;
        output.actionActive =
            !authority->loop && packet->eventAfter < authority->duration;
        if (input.animationDriven && selected->animationDriven) {
          const int root = authority->extractedMotionBone >= 0
                               ? authority->extractedMotionBone
                               : 0;
          const auto motion = rootDisplacement(
              *authority, root, packet->eventBefore, packet->eventAfter);
          const float c = std::cos(input.actorYawRadians),
                      s = std::sin(input.actorYawRadians);
          output.desiredRootMotion = {c * motion.x + s * motion.z, motion.y,
                                      -s * motion.x + c * motion.z};
        }
      }
      output.evaluationPacket = std::move(packet);
    } else {
      std::string ignored;
      m_poseGraph.restore(beforeGraph, ignored);
      output.diagnostics.push_back({AnimationDiagnosticSeverity::Warning,
                                    "native.graph_fallback", error});
    }
  }
  if (m_view->executionMode == AnimationExecutionMode::Havok &&
      !output.authoredGraphExecuted) {
    output.proceduralFallback = true;
    output.diagnostics.push_back(
        {AnimationDiagnosticSeverity::Warning, "graph.unsupported",
         "HKX graph is unsupported; deterministic state fallback is active"});
  }
  if (!input.ownsGameplayEvents)
    output.clipEvents.clear();
  output.events.insert(output.events.end(), output.clipEvents.begin(),
                       output.clipEvents.end());
  if (output.authoredGraphExecuted)
    m_state.queuedEvents.insert(m_state.queuedEvents.end(),
                                output.clipEvents.begin(),
                                output.clipEvents.end());
  output.activeState = m_state.state;
  output.activeRule = m_state.selectedRule;
  output.activeProvider = m_state.selectedProvider.empty()
                              ? m_view->providerId
                              : m_state.selectedProvider;
  if (!output.fallbackReason.empty())
    output.diagnostics.push_back({AnimationDiagnosticSeverity::Warning,
                                  "native.clip_unavailable",
                                  output.fallbackReason});
  if (selected && !selected->motionWarps.empty() && input.animationDriven &&
      selected->animationDriven && !input.ragdollActive && !input.dead &&
      !output.localPose.empty()) {
    const auto *authority = clipByName(output.activeClip);
    const float time = output.evaluationPacket
                           ? output.evaluationPacket->eventAfter
                           : m_state.stateTime;
    if (authority && !authority->loop)
      for (const auto &window : selected->motionWarps) {
        if (time <= window.start || time > window.end ||
            window.end > authority->duration)
          continue;
        const auto windowId = output.activeClip + ":" + window.target + ":" +
                              std::to_string(window.start);
        if (m_state.warpWindow != windowId) {
          m_state.warpWindow = windowId;
          m_state.warpTranslationUsed = 0;
        }
        const auto found = input.motionWarpTargets.find(window.target);
        if (found == input.motionWarpTargets.end())
          continue;
        const auto &target = found->second;
        if (!std::isfinite(target.position.x) ||
            !std::isfinite(target.position.y) ||
            !std::isfinite(target.position.z) ||
            !std::isfinite(target.yawRadians))
          continue;
        const int root = authority->extractedMotionBone >= 0
                             ? authority->extractedMotionBone
                             : 0;
        const auto remaining = rootDisplacement(
            *authority, root, std::max(window.start, time - delta), window.end);
        const auto rotation =
            odai::math::Matrix4::rotationY(input.actorYawRadians);
        auto correction = target.position - input.actorPosition -
                          odai::math::transformPoint(rotation, remaining);
        const float length = odai::math::length(correction);
        const float available =
            std::max(0.f, window.maxTranslation - m_state.warpTranslationUsed);
        if (length > 1.e-6f)
          correction = correction * (std::min(length, available) / length);
        const float fraction = std::clamp(
            delta / std::max(delta, window.end - time + delta), 0.f, 1.f);
        correction = correction * fraction;
        output.desiredRootMotion = output.desiredRootMotion + correction;
        m_state.warpTranslationUsed += odai::math::length(correction);
        // The actor root stays at the physics-owned position. Only the
        // bounded facing correction affects the visual in-place root.
        const float yaw =
            std::clamp(std::remainder(target.yawRadians - input.actorYawRadians,
                                      6.2831853f),
                       -window.maxYawRadians, window.maxYawRadians) *
            std::clamp((time - window.start) / (window.end - window.start), 0.f,
                       1.f);
        if (m_view->humanoidRig && root >= 0 &&
            static_cast<std::size_t>(root) < output.localPose.size()) {
          const auto facing = odai::math::transformPoint(
              odai::math::Matrix4::rotationY(yaw), {0, 0, 1000});
          if (aimBone(*m_view->skeleton, root, facing, {0, 0, 1}, std::abs(yaw),
                      1, output.localPose)) {
            PoseModifier modifier;
            modifier.kind = PoseModifier::Kind::Aim;
            modifier.chain.upper = root;
            modifier.target.position = facing;
            modifier.maxRadians = std::abs(yaw);
            postProcess->modifiers.push_back(modifier);
          }
        }
      }
  }
  if (!output.localPose.empty() && m_view->humanoidRig &&
      !input.ragdollActive && !input.dead) {
    if (input.teleported || !input.grounded)
      m_state.footPlants.clear();
    const auto &rig = *m_view->humanoidRig;
    const auto actorWorld =
        odai::math::Matrix4::translation(input.actorPosition) *
        odai::math::Matrix4::rotationY(input.actorYawRadians);
    const auto toModel = odai::math::inverse(actorWorld);
    auto world = composePoseWorld(*m_view->skeleton, output.localPose);
    float pelvisOffset = 0;
    if (input.footIkEnabled && input.grounded) {
      // A plant is a fixed world-space contact. Replacing it whenever the
      // actor has moved a small distance makes the foot skate in discrete
      // steps, which is especially visible at Skyrim walk speed. Acquire
      // close to the floor, retain through the stance, then fade while the
      // authored foot lifts. The probe normal is captured with the plant
      // so a locked foot does not swivel as later probes cross triangles.
      constexpr float kAcquireClearance = 6.f;
      constexpr float kReleaseClearance = 18.f;
      for (std::size_t foot = 0; foot < 2; ++foot) {
        const std::string name = foot == 0 ? "left_leg" : "right_leg";
        const auto chain = rig.limbs.find(name);
        const auto &contact = input.footContacts[foot];
        if (chain == rig.limbs.end() || !contact.valid) {
          m_state.footPlants.erase(name);
          continue;
        }
        const auto authored = odai::math::transformPoint(
            actorWorld * world[chain->second.end], {});
        const float clearance = authored.y - contact.position.y;
        auto plant = m_state.footPlants.find(name);
        if (clearance < -4.f || clearance >= kReleaseClearance) {
          m_state.footPlants.erase(name);
          continue;
        }
        if (plant == m_state.footPlants.end()) {
          if (clearance > kAcquireClearance)
            continue;
          plant = m_state.footPlants
                      .emplace(name,
                               std::vector<float>{
                                   contact.position.x, contact.position.y,
                                   contact.position.z, contact.normal.x,
                                   contact.normal.y, contact.normal.z})
                      .first;
        }
        const auto &p = plant->second;
        const auto hip = odai::math::transformPoint(
            actorWorld * world[chain->second.upper], {});
        const float maximumReach =
            odai::math::length(
                output.localPose[chain->second.lower].translation) +
            odai::math::length(output.localPose[chain->second.end].translation);
        if (maximumReach <= 1.e-4f ||
            odai::math::length(odai::math::Vector3{p[0], p[1], p[2]} - hip) >
                maximumReach * 1.05f) {
          m_state.footPlants.erase(name);
          continue;
        }
        const float lift =
            std::clamp((clearance - kAcquireClearance) /
                           (kReleaseClearance - kAcquireClearance),
                       0.f, 1.f);
        const float plantWeight = 1.f - lift * lift * (3.f - 2.f * lift);
        pelvisOffset =
            std::min(pelvisOffset,
                     std::clamp((p[1] - authored.y) * plantWeight, -8.f, 0.f));
      }
      if (const auto pelvis = rig.roles.find("pelvis");
          pelvis != rig.roles.end()) {
        output.localPose[pelvis->second].translation.y += pelvisOffset;
        PoseModifier modifier;
        modifier.chain.upper = pelvis->second;
        modifier.target.position = {0, pelvisOffset, 0};
        postProcess->modifiers.push_back(modifier);
      }
      for (std::size_t foot = 0; foot < 2; ++foot) {
        const std::string name = foot == 0 ? "left_leg" : "right_leg";
        const auto plant = m_state.footPlants.find(name);
        if (plant == m_state.footPlants.end())
          continue;
        const auto &p = plant->second;
        const auto authored = odai::math::transformPoint(
            actorWorld * world[rig.limbs.at(name).end], {});
        const float clearance =
            authored.y - input.footContacts[foot].position.y;
        const float lift =
            std::clamp((clearance - kAcquireClearance) /
                           (kReleaseClearance - kAcquireClearance),
                       0.f, 1.f);
        LimbTarget target;
        target.position =
            odai::math::transformPoint(toModel, {p[0], p[1], p[2]});
        const odai::math::Vector3 plantedNormal =
            p.size() >= 6 ? odai::math::Vector3{p[3], p[4], p[5]}
                          : input.footContacts[foot].normal;
        target.normal = odai::math::transformPoint(
            odai::math::Matrix4::rotationY(-input.actorYawRadians),
            plantedNormal);
        target.weight = 1.f - lift * lift * (3.f - 2.f * lift);
        target.alignNormal = true;
        if (solveTwoBoneIk(*m_view->skeleton, rig.limbs.at(name), target,
                           output.localPose)) {
          PoseModifier modifier;
          modifier.kind = PoseModifier::Kind::LimbIk;
          modifier.chain = rig.limbs.at(name);
          modifier.target = target;
          postProcess->modifiers.push_back(modifier);
        }
      }
    } else
      m_state.footPlants.clear();
    for (const auto &[name, target] : input.limbTargets)
      if (const auto chain = rig.limbs.find(name); chain != rig.limbs.end())
        if (solveTwoBoneIk(*m_view->skeleton, chain->second, target,
                           output.localPose)) {
          PoseModifier modifier;
          modifier.kind = PoseModifier::Kind::LimbIk;
          modifier.chain = chain->second;
          modifier.target = target;
          postProcess->modifiers.push_back(modifier);
        }
    if (input.aimEnabled) {
      for (const auto &[role, angle, weight] :
           {std::tuple{"spine", .35f, .35f}, std::tuple{"head", .8f, .65f}}) {
        if (aimBone(*m_view->skeleton, rig.roles.at(role), input.aimTarget,
                    {0, 0, 1}, angle, weight, output.localPose)) {
          PoseModifier modifier;
          modifier.kind = PoseModifier::Kind::Aim;
          modifier.chain.upper = rig.roles.at(role);
          modifier.target.position = input.aimTarget;
          modifier.target.weight = weight;
          modifier.maxRadians = angle;
          postProcess->modifiers.push_back(modifier);
        }
      }
    }
    m_sampler.paletteFromLocal(*m_view->skeleton, output.localPose,
                               output.pose);
  } else if (!input.ragdollActive)
    applyFootIk(input, output);
  if (output.evaluationPacket &&
      (!postProcess->offsets.empty() || !postProcess->modifiers.empty())) {
    auto packet =
        std::make_shared<PoseEvaluationPacket>(*output.evaluationPacket);
    PoseGraphInstruction instruction;
    instruction.kind = PoseGraphNode::Kind::Cache;
    instruction.inputs = {packet->output};
    instruction.postProcess = std::move(postProcess);
    packet->output = static_cast<std::uint32_t>(packet->instructions.size());
    packet->instructions.push_back(std::move(instruction));
    output.evaluationPacket = std::move(packet);
  }
  refreshSockets(output);
  m_state.wasGrounded = input.grounded;
  m_state.wasTalking = input.talking;
  ++m_state.fixedTick;
  if (std::any_of(output.diagnostics.begin(), output.diagnostics.end(),
                  [](const auto &diagnostic) {
                    return diagnostic.severity ==
                           AnimationDiagnosticSeverity::Error;
                  })) {
    output.authoredGraphExecuted = false;
    output.proceduralFallback = true;
  }
  return output;
}

void BehaviorGraphInstance::applyFootIk(const AnimationInputState &input,
                                        AnimationStepOutput &output) const {
  if (!input.footIkEnabled || !input.grounded || input.dead ||
      m_view == nullptr || m_view->skeleton == nullptr)
    return;
  const auto findAny = [&](std::initializer_list<const char *> names) {
    for (const char *name : names) {
      const int bone = m_view->skeleton->findBone(name);
      if (bone >= 0)
        return bone;
    }
    return -1;
  };
  const int left = findAny({"NPC L Foot [Lft ]", "L Foot", "LeftFoot"});
  const int right = findAny({"NPC R Foot [Rft ]", "R Foot", "RightFoot"});
  const int pelvis = findAny({"NPC Pelvis [Pelv]", "Pelvis"});
  constexpr float kMaxAnkleCorrection = 12.0f;
  const float leftOffset = std::clamp(
      input.leftFootIkOffset, -kMaxAnkleCorrection, kMaxAnkleCorrection);
  const float rightOffset = std::clamp(
      input.rightFootIkOffset, -kMaxAnkleCorrection, kMaxAnkleCorrection);
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
    output.diagnostics.push_back(
        {AnimationDiagnosticSeverity::Warning, "foot_ik.missing_bones",
         "hkbFootIkControlsModifier could not find both foot bones"});
  }
}

void BehaviorGraphInstance::refreshSockets(AnimationStepOutput &output) const {
  if (m_view == nullptr || m_view->skeleton == nullptr)
    return;
  for (const std::string &name : m_view->socketBoneNames) {
    const int bone = m_view->skeleton->findBone(name);
    if (bone < 0)
      continue;
    const std::size_t index = static_cast<std::size_t>(bone);
    output.socketTransforms[name] =
        index < output.pose.size() && index < m_bindWorld.size()
            ? output.pose[index] * m_bindWorld[index]
            : m_bindWorld[index];
  }
}

void BehaviorGraphInstance::queueEvent(AnimationEvent event) {
  m_state.queuedEvents.push_back(std::move(event));
}

BehaviorGraphSnapshot BehaviorGraphInstance::snapshot() const {
  auto result = m_state;
  result.poseGraphState = m_poseGraph.save();
  result.inertialState = m_inertializer.save();
  return result;
}

bool BehaviorGraphInstance::restore(const BehaviorGraphSnapshot &snapshot,
                                    std::string &outError) {
  outError.clear();
  if (snapshot.stateTime < 0.0f || !std::isfinite(snapshot.stateTime) ||
      snapshot.previousStateTime < 0.0f ||
      !std::isfinite(snapshot.previousStateTime) ||
      snapshot.transitionElapsed < 0.0f ||
      !std::isfinite(snapshot.transitionElapsed) ||
      snapshot.transitionDuration < 0.0f ||
      !std::isfinite(snapshot.transitionDuration)) {
    outError = "invalid behavior graph transition time";
    return false;
  }
  if (m_view && !snapshot.graphFingerprint.empty() &&
      snapshot.graphFingerprint != m_view->sourceFingerprint) {
    outError = "saved animation provider fingerprint changed";
    return false;
  }
  if (snapshot.layerTimes.size() > 16) {
    outError = "too many saved layer clocks";
    return false;
  }
  for (const auto &[name, value] : snapshot.layerTimes)
    if (name.empty() || !std::isfinite(value) || value < 0) {
      outError = "invalid saved layer clock";
      return false;
    }
  for (const auto &[name, value] : snapshot.variables) {
    if (name.empty() || !std::isfinite(value)) {
      outError = "invalid graph variable";
      return false;
    }
  }
  if (!snapshot.activeStates.empty()) {
    if (!m_view || !m_view->behavior || !m_view->behavior->executable()) {
      outError = "saved graph requires executable authored behavior";
      return false;
    }
    const auto &nodes = m_view->behavior->graph.nodes;
    for (const auto &[index, state] : snapshot.activeStates) {
      if (index < nodes.size() &&
          nodes[index].kind == HkxBehaviorNodeKind::ManualSelector &&
          state >= 0 &&
          static_cast<std::size_t>(state) < nodes[index].children.size())
        continue;
      if (index >= nodes.size() ||
          nodes[index].kind != HkxBehaviorNodeKind::StateMachine ||
          std::none_of(
              nodes[index].children.begin(), nodes[index].children.end(),
              [&](auto child) { return nodes[child].stateId == state; })) {
        outError = "saved graph has invalid active state";
        return false;
      }
    }
  }
  if (m_view && snapshot.executionMode != m_view->executionMode) {
    outError = "saved animation execution mode changed";
    return false;
  }
  if (!snapshot.selectedRule.empty()) {
    const auto *program = m_view ? m_view->nativeProgram.get() : nullptr;
    if (!program ||
        std::none_of(program->rules.begin(), program->rules.end(),
                     [&](const auto &rule) {
                       return rule.id == snapshot.selectedRule &&
                              rule.provider == snapshot.selectedProvider &&
                              rule.state == snapshot.state &&
                              std::any_of(rule.variants.begin(),
                                          rule.variants.end(),
                                          [&](const auto &variant) {
                                            return variant.clip ==
                                                   snapshot.selectedClip;
                                          });
                     })) {
      outError = "saved native animation selection is invalid";
      return false;
    }
  } else if (!snapshot.selectedClip.empty()) {
    outError = "saved native animation clip has no rule";
    return false;
  }
  PoseGraphInstance restored;
  if (!std::isfinite(snapshot.warpTranslationUsed) ||
      snapshot.warpTranslationUsed < 0 || snapshot.warpTranslationUsed > 256) {
    outError = "invalid saved warp budget";
    return false;
  }
  if (!restored.restore(snapshot.poseGraphState, outError))
    return false;
  PoseInertializer inertial;
  if (!inertial.restore(snapshot.inertialState, outError))
    return false;
  if (snapshot.footPlants.size() > 2) {
    outError = "too many saved foot plants";
    return false;
  }
  for (const auto &[name, values] : snapshot.footPlants)
    if ((name != "left_leg" && name != "right_leg") ||
        (values.size() != 3 && values.size() != 6) ||
        std::any_of(values.begin(), values.end(),
                    [](float v) { return !std::isfinite(v); })) {
      outError = "invalid saved foot plant";
      return false;
    }
  m_inertializer = std::move(inertial);
  m_poseGraph = std::move(restored);
  m_state = snapshot;
  if (m_view && m_view->behavior)
    for (const auto &[name, value] : m_view->behavior->graph.variableDefaults)
      m_state.variables.try_emplace(name, value);
  return true;
}

AnimationStepOutput
BehaviorGraphInstance::interpolate(const AnimationStepOutput &previous,
                                   const AnimationStepOutput &current,
                                   float alpha) {
  if (previous.pose.size() != current.pose.size())
    return current;
  AnimationStepOutput result = current;
  const float t = odai::math::saturate(alpha);
  for (std::size_t matrix = 0; matrix < result.pose.size(); ++matrix) {
    for (std::size_t element = 0; element < 16u; ++element) {
      result.pose[matrix].m[element] = odai::math::lerp(
          previous.pose[matrix].m[element], current.pose[matrix].m[element], t);
    }
  }
  for (auto &[name, transform] : result.socketTransforms) {
    const auto old = previous.socketTransforms.find(name);
    if (old == previous.socketTransforms.end())
      continue;
    for (std::size_t element = 0; element < 16u; ++element) {
      transform.m[element] =
          odai::math::lerp(old->second.m[element], transform.m[element], t);
    }
  }
  result.desiredRootMotion = odai::math::lerp(previous.desiredRootMotion,
                                              current.desiredRootMotion, t);
  return result;
}

} // namespace odai::anim
