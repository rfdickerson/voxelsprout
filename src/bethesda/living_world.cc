#include "bethesda/living_world.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <tuple>

namespace odai::bethesda {
namespace {

RuntimeBehaviorSource behaviorSource(BehaviorPackageSource source) {
    switch (source) {
        case BehaviorPackageSource::Generated:
            return RuntimeBehaviorSource::GeneratedSchedule;
        case BehaviorPackageSource::QuestOrScript:
            return RuntimeBehaviorSource::QuestOrScript;
        case BehaviorPackageSource::AuthoredTes3:
        case BehaviorPackageSource::AuthoredTes4:
        case BehaviorPackageSource::AuthoredFallout:
        case BehaviorPackageSource::AuthoredTes5:
            return RuntimeBehaviorSource::AuthoredPackage;
    }
    return RuntimeBehaviorSource::SafeIdle;
}

std::array<double, 3> positionOf(const RuntimeObject& object) {
    return object.transform.position;
}

double distanceSquared(
    const std::array<double, 3>& left, const std::array<double, 3>& right) {
    const double x = left[0] - right[0];
    const double y = left[1] - right[1];
    const double z = left[2] - right[2];
    return x * x + y * y + z * z;
}

bool statePresentationChanged(
    const std::optional<RuntimeLivingState>& old,
    const RuntimeLivingState& desired) {
    if (!old.has_value()) return true;
    return old->activity != desired.activity || old->source != desired.source ||
        old->tier != desired.tier || old->anchor != desired.anchor ||
        old->phaseIndex != desired.phaseIndex || old->reason != desired.reason ||
        old->travelling != desired.travelling ||
        old->nextTransitionGameMinute != desired.nextTransitionGameMinute;
}

RuntimeActivityKind reactionFor(
    const ActorArchetype* witness, const LivingWorldStimulus& stimulus) {
    if (stimulus.kind == StimulusKind::Explosion ||
        stimulus.kind == StimulusKind::BodyFound) {
        return witness != nullptr && hasRole(witness->roles, ActorRole::Guard)
            ? RuntimeActivityKind::Investigate : RuntimeActivityKind::Flee;
    }
    if (stimulus.crime != RuntimeCrimeKind::None &&
        witness != nullptr && hasRole(witness->roles, ActorRole::Guard)) {
        return stimulus.crime == RuntimeCrimeKind::Assault ||
            stimulus.crime == RuntimeCrimeKind::Murder
            ? RuntimeActivityKind::Combat : RuntimeActivityKind::Investigate;
    }
    return RuntimeActivityKind::Investigate;
}

}  // namespace

LivingWorldSimulation::LivingWorldSimulation(LivingWorldConfig config) {
    reset(config);
}

void LivingWorldSimulation::reset(LivingWorldConfig config) {
    m_config = config;
    if (!std::isfinite(m_config.timeScale) || m_config.timeScale < 0.0) {
        m_config.timeScale = 20.0;
    }
    m_cells.clear();
    m_schedules = {};
    m_archetypes.clear();
    m_anchors.clear();
    m_physicsPolicies.clear();
    m_residentSpaces.clear();
    m_pendingStimuli.clear();
    m_interruptUntilMinute.clear();
    m_absoluteGameMinute = m_config.initialGameMinute;
    m_fractionalGameMinute = 0.0;
    m_nextStimulusSequence = 1u;
    m_needsEvaluation = true;
}

void LivingWorldSimulation::installCells(std::vector<GameplayCellPayload> cells) {
    m_cells = std::move(cells);
    m_schedules = SystemicScheduleCompiler{}.compile(m_cells);
    m_archetypes.clear();
    m_anchors.clear();
    m_physicsPolicies.clear();
    for (const GameplayCellPayload& cell : m_cells) {
        for (const ActorArchetype& actor : cell.actors) {
            m_archetypes.insert_or_assign(actor.actor, actor);
        }
        for (const ActivityAnchor& anchor : cell.anchors) {
            m_anchors.insert_or_assign(anchor.object, anchor);
        }
        for (const PhysicsPolicy& policy : cell.physicsPolicies) {
            m_physicsPolicies.insert_or_assign(policy.object, policy);
        }
    }
    m_needsEvaluation = true;
}

void LivingWorldSimulation::upsertCell(GameplayCellPayload cell) {
    const auto existing = std::find_if(m_cells.begin(), m_cells.end(),
        [&](const GameplayCellPayload& current) {
            return current.space == cell.space;
        });
    if (existing == m_cells.end()) m_cells.push_back(std::move(cell));
    else *existing = std::move(cell);
    installCells(std::move(m_cells));
}

void LivingWorldSimulation::setResidentSpaces(std::vector<RuntimeSpaceState> spaces) {
    std::sort(spaces.begin(), spaces.end(), [](const auto& left, const auto& right) {
        return std::tie(left.kind, left.worldspace, left.cell, left.gridX, left.gridZ) <
            std::tie(right.kind, right.worldspace, right.cell, right.gridX, right.gridZ);
    });
    spaces.erase(std::unique(spaces.begin(), spaces.end()), spaces.end());
    if (m_residentSpaces != spaces) {
        m_residentSpaces = std::move(spaces);
        m_needsEvaluation = true;
    }
}

std::uint64_t LivingWorldSimulation::postStimulus(LivingWorldStimulus stimulus) {
    if (stimulus.sequence == 0u) stimulus.sequence = m_nextStimulusSequence++;
    else m_nextStimulusSequence = std::max(m_nextStimulusSequence, stimulus.sequence + 1u);
    m_pendingStimuli.push_back(std::move(stimulus));
    return m_pendingStimuli.back().sequence;
}

bool LivingWorldSimulation::isResident(const RuntimeSpaceState& space) const {
    return std::binary_search(m_residentSpaces.begin(), m_residentSpaces.end(), space,
        [](const auto& left, const auto& right) {
            return std::tie(left.kind, left.worldspace, left.cell, left.gridX, left.gridZ) <
                std::tie(right.kind, right.worldspace, right.cell, right.gridX, right.gridZ);
        });
}

RuntimeLivingState LivingWorldSimulation::chooseState(
    const RuntimeObject& actor, const ActorSchedule* schedule) const {
    RuntimeLivingState desired = actor.livingState.value_or(RuntimeLivingState{});
    desired.absoluteGameMinute = m_absoluteGameMinute;
    desired.tier = isResident(actor.currentSpace)
        ? RuntimeSimulationTier::Full : RuntimeSimulationTier::Abstract;

    if (actor.actorValues.has_value() && actor.actorValues->dead) {
        desired.activity = RuntimeActivityKind::Idle;
        desired.source = RuntimeBehaviorSource::Death;
        desired.reason = "dead actors do not evaluate packages";
        desired.anchor = {};
        desired.travelling = false;
        desired.nextTransitionGameMinute = std::numeric_limits<std::uint64_t>::max();
        return desired;
    }
    if (actor.combatState.has_value() && actor.combatState->combatTarget.valid()) {
        desired.activity = RuntimeActivityKind::Combat;
        desired.source = RuntimeBehaviorSource::Combat;
        desired.reason = "combat interrupts all packages";
        desired.anchor = actor.combatState->combatTarget;
        desired.travelling = false;
        desired.nextTransitionGameMinute = m_absoluteGameMinute + 1u;
        return desired;
    }
    if (actor.inDialogueWithPlayer) {
        desired.activity = RuntimeActivityKind::Dialogue;
        desired.source = RuntimeBehaviorSource::Dialogue;
        desired.reason = "player dialogue interrupts packages";
        desired.anchor = {};
        desired.travelling = false;
        desired.nextTransitionGameMinute = m_absoluteGameMinute + 1u;
        return desired;
    }
    const auto interrupted = m_interruptUntilMinute.find(actor.id);
    const bool persistedEmergency = actor.livingState.has_value() &&
        actor.livingState->source == RuntimeBehaviorSource::CrimeOrEmergency &&
        actor.livingState->nextTransitionGameMinute > m_absoluteGameMinute;
    if (persistedEmergency ||
        (interrupted != m_interruptUntilMinute.end() &&
         interrupted->second > m_absoluteGameMinute && actor.livingState.has_value() &&
         actor.livingState->source == RuntimeBehaviorSource::CrimeOrEmergency)) {
        if (interrupted != m_interruptUntilMinute.end()) {
            desired.nextTransitionGameMinute = std::max(
                desired.nextTransitionGameMinute, interrupted->second);
        }
        return desired;
    }
    if (schedule == nullptr) {
        desired.activity = RuntimeActivityKind::Idle;
        desired.source = RuntimeBehaviorSource::SafeIdle;
        desired.reason = "safe idle: actor has no compiled schedule";
        desired.anchor = {};
        desired.confidence = 0.0f;
        desired.travelling = false;
        desired.nextTransitionGameMinute = m_absoluteGameMinute + 1440u;
        return desired;
    }
    const std::uint16_t minute = static_cast<std::uint16_t>(m_absoluteGameMinute % 1440u);
    const ScheduleEntry* entry = SystemicScheduleCompiler::entryAt(*schedule, minute);
    if (entry == nullptr) {
        desired.activity = RuntimeActivityKind::Idle;
        desired.source = RuntimeBehaviorSource::SafeIdle;
        desired.reason = "safe idle: no active package at this time";
        desired.anchor = {};
        desired.confidence = 0.0f;
        desired.travelling = false;
        desired.nextTransitionGameMinute = m_absoluteGameMinute + 1u;
        return desired;
    }
    desired.activity = entry->activity;
    desired.source = behaviorSource(entry->source);
    desired.anchor = entry->anchor;
    desired.confidence = entry->confidence;
    desired.reason = entry->reason;
    desired.phaseIndex = static_cast<std::uint32_t>(
        std::distance(schedule->entries.data(), entry));
    desired.nextTransitionGameMinute = nextTransition(*schedule, entry);
    desired.travelling = desired.tier == RuntimeSimulationTier::Full &&
        desired.anchor.valid() && desired.activity != RuntimeActivityKind::Idle;
    return desired;
}

std::uint64_t LivingWorldSimulation::nextTransition(
    const ActorSchedule& schedule, const ScheduleEntry* current) const {
    for (std::uint64_t offset = 1u; offset <= 1440u; ++offset) {
        const std::uint16_t minute = static_cast<std::uint16_t>(
            (m_absoluteGameMinute + offset) % 1440u);
        if (SystemicScheduleCompiler::entryAt(schedule, minute) != current) {
            return m_absoluteGameMinute + offset;
        }
    }
    return m_absoluteGameMinute + 1440u;
}

void LivingWorldSimulation::evaluateActors(
    BethesdaWorld& world, LivingWorldStep& result) {
    for (const ObjectId& actorId : world.orderedActorIds()) {
        const RuntimeObject* actor = world.find(actorId);
        if (actor == nullptr || !actor->enabled) continue;
        ++result.actorsEvaluated;
        const auto foundSchedule = m_schedules.actors.find(actorId);
        const ActorSchedule* schedule = foundSchedule == m_schedules.actors.end()
            ? nullptr : &foundSchedule->second;
        RuntimeLivingState desired = chooseState(*actor, schedule);
        if (!statePresentationChanged(actor->livingState, desired) &&
            actor->livingState->absoluteGameMinute == m_absoluteGameMinute) continue;

        const bool activityChanged = !actor->livingState.has_value() ||
            actor->livingState->activity != desired.activity ||
            actor->livingState->anchor != desired.anchor ||
            actor->livingState->source != desired.source;
        if (activityChanged) ++result.activityChanges;

        WorldCommand living;
        living.type = WorldCommandType::SetLivingState;
        living.target = actorId;
        living.livingState = desired;
        (void)world.queue(std::move(living));

        if (!activityChanged || !desired.anchor.valid() ||
            desired.source == RuntimeBehaviorSource::Combat ||
            desired.source == RuntimeBehaviorSource::Dialogue ||
            desired.source == RuntimeBehaviorSource::Death) continue;
        const RuntimeObject* anchor = world.find(desired.anchor);
        if (desired.tier == RuntimeSimulationTier::Full) {
            if (anchor != nullptr &&
                (!actor->navigationRequest.has_value() ||
                 actor->navigationRequest->destination != desired.anchor)) {
                WorldCommand move;
                move.type = WorldCommandType::RequestMoveTo;
                move.target = actorId;
                move.destination = desired.anchor;
                (void)world.queue(std::move(move));
            }
        } else if (anchor != nullptr) {
            WorldCommand position;
            position.type = WorldCommandType::SetPosition;
            position.target = actorId;
            position.transform.position = anchor->transform.position;
            (void)world.queue(std::move(position));
            if (actor->currentSpace != anchor->currentSpace) {
                WorldCommand space;
                space.type = WorldCommandType::SetCurrentSpace;
                space.target = actorId;
                space.currentSpace = anchor->currentSpace;
                (void)world.queue(std::move(space));
            }
            ++result.offscreenReconciliations;
        }
    }
}

void LivingWorldSimulation::processStimuli(
    BethesdaWorld& world, const LineOfSight& lineOfSight,
    LivingWorldStep& result) {
    std::stable_sort(m_pendingStimuli.begin(), m_pendingStimuli.end(),
        [](const auto& left, const auto& right) { return left.sequence < right.sequence; });
    const std::size_t count = std::min(m_pendingStimuli.size(), m_config.maxStimuliPerTick);
    for (std::size_t index = 0u; index < count; ++index) {
        const LivingWorldStimulus& stimulus = m_pendingStimuli[index];
        bool crimeWitnessed = false;
        for (const ObjectId& observerId : world.orderedActorIds()) {
            if (observerId == stimulus.source || observerId == stimulus.subject) continue;
            const RuntimeObject* observer = world.find(observerId);
            if (observer == nullptr || !observer->enabled ||
                (observer->actorValues.has_value() && observer->actorValues->dead) ||
                !isResident(observer->currentSpace)) continue;
            const double distance2 = distanceSquared(positionOf(*observer), stimulus.position);
            const double hearing2 = static_cast<double>(stimulus.hearingRadius) *
                static_cast<double>(stimulus.hearingRadius);
            const double sight2 = static_cast<double>(stimulus.sightRadius) *
                static_cast<double>(stimulus.sightRadius);
            const bool heard = stimulus.hearingRadius > 0.0f && distance2 <= hearing2;
            const bool visible = stimulus.sightRadius > 0.0f && distance2 <= sight2 &&
                (!lineOfSight || lineOfSight(positionOf(*observer), stimulus.position));
            if (!heard && !visible) continue;
            const auto archetype = m_archetypes.find(observerId);
            const ActorArchetype* observerType = archetype == m_archetypes.end()
                ? nullptr : &archetype->second;
            const RuntimeActivityKind reaction = reactionFor(observerType, stimulus);
            const float distance = static_cast<float>(std::sqrt(distance2));
            const float radius = std::max(stimulus.sightRadius, stimulus.hearingRadius);
            const float confidence = std::clamp(
                stimulus.severity * (1.0f - distance / std::max(1.0f, radius)), 0.05f, 1.0f);
            RuntimeLivingState state = observer->livingState.value_or(RuntimeLivingState{});
            state.activity = reaction;
            state.source = RuntimeBehaviorSource::CrimeOrEmergency;
            state.tier = RuntimeSimulationTier::Full;
            state.anchor = stimulus.source;
            state.absoluteGameMinute = m_absoluteGameMinute;
            state.nextTransitionGameMinute = m_absoluteGameMinute +
                (reaction == RuntimeActivityKind::Flee ? 10u : 5u);
            state.confidence = confidence;
            state.reason = heard && visible ? "saw and heard stimulus" :
                heard ? "heard stimulus" : "saw stimulus";
            state.travelling = reaction == RuntimeActivityKind::Investigate;
            state.lastStimulusSequence = stimulus.sequence;
            if (stimulus.crime != RuntimeCrimeKind::None) ++state.crimesWitnessed;
            WorldCommand command;
            command.type = WorldCommandType::SetLivingState;
            command.target = observerId;
            command.livingState = std::move(state);
            (void)world.queue(std::move(command));
            m_interruptUntilMinute[observerId] = m_absoluteGameMinute +
                (reaction == RuntimeActivityKind::Flee ? 10u : 5u);
            result.witnesses.push_back(WitnessResolution{
                stimulus.sequence, observerId, stimulus.source, reaction, confidence});
            crimeWitnessed = crimeWitnessed || stimulus.crime != RuntimeCrimeKind::None;
        }
        if (crimeWitnessed && stimulus.source.valid()) {
            WorldCommand crime;
            crime.type = WorldCommandType::ReportCrime;
            crime.target = stimulus.source;
            crime.crimeKind = stimulus.crime;
            crime.crimeValue = stimulus.bounty;
            crime.stimulusSequence = stimulus.sequence;
            crime.faction = stimulus.faction;
            (void)world.queue(std::move(crime));
        }
    }
    m_pendingStimuli.erase(m_pendingStimuli.begin(),
        m_pendingStimuli.begin() + static_cast<std::ptrdiff_t>(count));
}

void LivingWorldSimulation::updatePhysicalPersistence(
    BethesdaWorld& world, LivingWorldStep& result) {
    for (const ObjectId& objectId : world.orderedObjectIds()) {
        const RuntimeObject* object = world.find(objectId);
        if (object == nullptr) continue;
        if (!object->physicalState.has_value()) {
            const auto policy = m_physicsPolicies.find(objectId);
            if (policy == m_physicsPolicies.end() ||
                policy->second.classification == PhysicsClassification::Structural) continue;
            RuntimePhysicalState initial;
            initial.authoredTransform = object->transform;
            initial.dynamic = true;
            initial.breakable =
                policy->second.classification == PhysicsClassification::Breakable;
            initial.constrained =
                policy->second.classification == PhysicsClassification::Constrained;
            initial.protectedFromDestruction = policy->second.protectedFromDestruction;
            initial.resettable = policy->second.resettable;
            initial.owned = policy->second.owned;
            initial.questLinked = policy->second.questLinked;
            initial.meaningful = initial.owned || initial.questLinked;
            WorldCommand physical;
            physical.type = WorldCommandType::SetPhysicalState;
            physical.target = objectId;
            physical.physicalState = std::move(initial);
            (void)world.queue(std::move(physical));
            continue;
        }
        if (
            !object->physicalState->resettable || object->physicalState->meaningful) continue;
        RuntimePhysicalState state = *object->physicalState;
        if (isResident(object->currentSpace)) {
            if (state.unloadedSinceGameMinute == 0u) continue;
            state.unloadedSinceGameMinute = 0u;
        } else if (state.unloadedSinceGameMinute == 0u) {
            state.unloadedSinceGameMinute = m_absoluteGameMinute;
        } else if (m_absoluteGameMinute - state.unloadedSinceGameMinute >=
                   m_config.transientResetMinutes) {
            WorldCommand transform;
            transform.type = WorldCommandType::SetTransform;
            transform.target = objectId;
            transform.transform = state.authoredTransform;
            (void)world.queue(std::move(transform));
            state.linearVelocity = {};
            state.angularVelocity = {};
            state.broken = false;
            state.unloadedSinceGameMinute = 0u;
            ++result.physicalResets;
        } else {
            continue;
        }
        WorldCommand physical;
        physical.type = WorldCommandType::SetPhysicalState;
        physical.target = objectId;
        physical.physicalState = std::move(state);
        (void)world.queue(std::move(physical));
    }
}

LivingWorldStep LivingWorldSimulation::advance(
    double fixedStepSeconds, BethesdaWorld& world,
    const LineOfSight& lineOfSight) {
    LivingWorldStep result;
    if (!m_config.enabled || m_cells.empty()) return result;
    if (!std::isfinite(fixedStepSeconds) || fixedStepSeconds < 0.0) fixedStepSeconds = 0.0;
    m_fractionalGameMinute += fixedStepSeconds * m_config.timeScale / 60.0;
    const auto elapsed = static_cast<std::uint64_t>(std::floor(m_fractionalGameMinute));
    if (elapsed != 0u) {
        m_absoluteGameMinute += elapsed;
        m_fractionalGameMinute -= static_cast<double>(elapsed);
    }
    const bool evaluate = m_needsEvaluation || elapsed != 0u || !m_pendingStimuli.empty();
    if (evaluate) evaluateActors(world, result);
    // Stimuli are applied after routine evaluation so an emergency wins this
    // tick and the interrupt window preserves it on following ticks.
    processStimuli(world, lineOfSight, result);
    if (m_needsEvaluation || elapsed != 0u) updatePhysicalPersistence(world, result);
    m_needsEvaluation = false;
    return result;
}

LivingWorldStep LivingWorldSimulation::advanceGameMinutes(
    std::uint64_t minutes, BethesdaWorld& world,
    const LineOfSight& lineOfSight) {
    LivingWorldStep result;
    if (!m_config.enabled || m_cells.empty()) return result;
    m_absoluteGameMinute += minutes;
    evaluateActors(world, result);
    processStimuli(world, lineOfSight, result);
    updatePhysicalPersistence(world, result);
    return result;
}

bool LivingWorldSimulation::markPhysicalInteraction(
    BethesdaWorld& world, ObjectId objectId,
    bool playerGrabbed, bool intentionallyPlaced,
    bool broken, std::string& outError) {
    const RuntimeObject* object = world.find(objectId);
    if (object == nullptr || !object->physicalState.has_value()) {
        outError = "physical interaction target has no runtime physical policy";
        return false;
    }
    RuntimePhysicalState state = *object->physicalState;
    if (broken && !state.breakable) {
        outError = "physical interaction target is not breakable";
        return false;
    }
    state.playerGrabbed = state.playerGrabbed || playerGrabbed;
    state.intentionallyPlaced = state.intentionallyPlaced || intentionallyPlaced;
    state.broken = state.broken || broken;
    state.lastTouchedGameMinute = m_absoluteGameMinute;
    state.meaningful = state.playerGrabbed || state.intentionallyPlaced || state.broken ||
        state.owned || state.questLinked;
    if (state.broken && state.protectedFromDestruction) {
        // Protected objects retain identity and a recoverable broken state.
        state.meaningful = true;
    }
    WorldCommand command;
    command.type = WorldCommandType::SetPhysicalState;
    command.target = objectId;
    command.physicalState = std::move(state);
    (void)world.queue(std::move(command));
    outError.clear();
    return true;
}

void LivingWorldSimulation::restoreClock(
    std::uint64_t absoluteGameMinute, double fractionalGameMinute) {
    m_absoluteGameMinute = absoluteGameMinute;
    m_fractionalGameMinute = std::isfinite(fractionalGameMinute)
        ? std::clamp(fractionalGameMinute, 0.0, std::nextafter(1.0, 0.0)) : 0.0;
}

}  // namespace odai::bethesda
