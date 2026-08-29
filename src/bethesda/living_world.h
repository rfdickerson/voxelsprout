#pragma once

#include "bethesda/gameplay_catalog.h"

#include <cstdint>
#include <functional>
#include <map>
#include <span>
#include <vector>

namespace odai::bethesda {

enum class StimulusKind : std::uint8_t {
    Combat,
    Theft,
    Assault,
    Breakage,
    Explosion,
    DisplacedOwnedObject,
    BodyFound,
    Trespass,
};

struct LivingWorldStimulus {
    std::uint64_t sequence = 0u;
    StimulusKind kind = StimulusKind::Combat;
    ObjectId source;
    ObjectId subject;
    std::array<double, 3> position{};
    float sightRadius = 2048.0f;
    float hearingRadius = 2048.0f;
    float severity = 1.0f;
    RuntimeCrimeKind crime = RuntimeCrimeKind::None;
    std::int64_t bounty = 0;
    RecordKey faction;
    friend bool operator==(const LivingWorldStimulus&,
                           const LivingWorldStimulus&) = default;
};

struct WitnessResolution {
    std::uint64_t stimulusSequence = 0u;
    ObjectId witness;
    ObjectId offender;
    RuntimeActivityKind reaction = RuntimeActivityKind::Investigate;
    float confidence = 0.0f;
    friend bool operator==(const WitnessResolution&,
                           const WitnessResolution&) = default;
};

struct LivingWorldConfig {
    bool enabled = true;
    double timeScale = 20.0;
    std::uint64_t initialGameMinute = 8u * 60u;
    std::uint64_t transientResetMinutes = 72u * 60u;
    std::size_t maxStimuliPerTick = 64u;
};

struct LivingWorldStep {
    std::size_t actorsEvaluated = 0u;
    std::size_t activityChanges = 0u;
    std::size_t offscreenReconciliations = 0u;
    std::size_t physicalResets = 0u;
    std::vector<WitnessResolution> witnesses;
    std::vector<std::string> diagnostics;
};

class LivingWorldSimulation {
public:
    using LineOfSight = std::function<bool(
        const std::array<double, 3>&, const std::array<double, 3>&)>;

    explicit LivingWorldSimulation(LivingWorldConfig config = {});

    void reset(LivingWorldConfig config = {});
    void installCells(std::vector<GameplayCellPayload> cells);
    void upsertCell(GameplayCellPayload cell);
    void setResidentSpaces(std::vector<RuntimeSpaceState> spaces);
    [[nodiscard]] std::uint64_t postStimulus(LivingWorldStimulus stimulus);

    LivingWorldStep advance(
        double fixedStepSeconds,
        BethesdaWorld& world,
        const LineOfSight& lineOfSight = {});

    // Deterministic analytical catch-up for sleep/wait/fast travel and tests.
    LivingWorldStep advanceGameMinutes(
        std::uint64_t minutes,
        BethesdaWorld& world,
        const LineOfSight& lineOfSight = {});

    bool markPhysicalInteraction(
        BethesdaWorld& world, ObjectId object,
        bool playerGrabbed, bool intentionallyPlaced,
        bool broken, std::string& outError);

    [[nodiscard]] std::uint64_t absoluteGameMinute() const {
        return m_absoluteGameMinute;
    }
    void restoreClock(std::uint64_t absoluteGameMinute, double fractionalGameMinute);
    [[nodiscard]] double fractionalGameMinute() const { return m_fractionalGameMinute; }
    [[nodiscard]] std::uint64_t nextStimulusSequence() const {
        return m_nextStimulusSequence;
    }
    void setNextStimulusSequence(std::uint64_t sequence) {
        m_nextStimulusSequence = sequence == 0u ? 1u : sequence;
    }
    [[nodiscard]] const ScheduleCompileResult& schedules() const { return m_schedules; }
    [[nodiscard]] const std::vector<GameplayCellPayload>& cells() const { return m_cells; }

private:
    [[nodiscard]] bool isResident(const RuntimeSpaceState& space) const;
    [[nodiscard]] RuntimeLivingState chooseState(
        const RuntimeObject& actor, const ActorSchedule* schedule) const;
    [[nodiscard]] std::uint64_t nextTransition(
        const ActorSchedule& schedule, const ScheduleEntry* current) const;
    void evaluateActors(BethesdaWorld& world, LivingWorldStep& result);
    void processStimuli(
        BethesdaWorld& world, const LineOfSight& lineOfSight,
        LivingWorldStep& result);
    void updatePhysicalPersistence(BethesdaWorld& world, LivingWorldStep& result);

    LivingWorldConfig m_config;
    std::vector<GameplayCellPayload> m_cells;
    ScheduleCompileResult m_schedules;
    std::map<ObjectId, ActorArchetype> m_archetypes;
    std::map<ObjectId, ActivityAnchor> m_anchors;
    std::map<ObjectId, PhysicsPolicy> m_physicsPolicies;
    std::vector<RuntimeSpaceState> m_residentSpaces;
    std::vector<LivingWorldStimulus> m_pendingStimuli;
    std::map<ObjectId, std::uint64_t> m_interruptUntilMinute;
    std::uint64_t m_absoluteGameMinute = 0u;
    double m_fractionalGameMinute = 0.0;
    std::uint64_t m_nextStimulusSequence = 1u;
    bool m_needsEvaluation = true;
};

}  // namespace odai::bethesda
