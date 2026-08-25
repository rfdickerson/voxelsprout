#pragma once

#include "bethesda/fixed_step_clock.h"
#include "bethesda/bethesda_physics_world.h"
#include "bethesda/papyrus_vm.h"
#include "bethesda/runtime_world.h"
#include "bethesda/runtime_render_delta.h"
#include "bethesda/scenario.h"
#include "bethesda/skyrim_quest.h"
#include "bethesda/skyrim_dialogue.h"
#include "bethesda/tes3_runtime.h"
#include "import/fnv/content_profile.h"
#include "anim/skyrim_animation.h"

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace odai::bethesda {

struct QuestObjectiveState {
    std::int32_t index = 0;
    bool displayed = false;
    bool completed = false;
    bool failed = false;
    // Content-owned presentation metadata. Save files persist only the four
    // mutable fields above and restore this from the registered QUST record.
    std::string displayText;
    friend bool operator==(const QuestObjectiveState&, const QuestObjectiveState&) = default;
};

struct QuestAliasRuntimeState {
    std::int32_t id = -1;
    std::string name;
    bool location = false;
    std::uint32_t sourceFormId = 0u;
    std::int32_t findMatchingReferenceInAliasId = -1;
    RecordKey referenceType;
    ObjectId handle;
    ObjectId target;
    RecordKey createdObject;
    std::int32_t createdInAliasId = -1;
    std::int32_t createdLevel = 0;
    bool createdObjectMaterialized = false;
    friend bool operator==(const QuestAliasRuntimeState&, const QuestAliasRuntimeState&) = default;
};

struct QuestRuntimeState {
    std::string editorId;
    RecordKey record;
    std::int32_t stage = 0;
    std::vector<std::int32_t> completedStages;
    bool running = false;
    bool completed = false;
    bool failed = false;
    std::vector<QuestObjectiveState> objectives;
    std::vector<QuestAliasRuntimeState> aliases;
    friend bool operator==(const QuestRuntimeState&, const QuestRuntimeState&) = default;
};

struct LocationRuntimeState {
    RecordKey record;
    RecordKey parent;
    std::vector<RecordKey> keywords;
    std::map<RecordKey, float> keywordData;
    bool loaded = false;
    friend bool operator==(const LocationRuntimeState&, const LocationRuntimeState&) = default;
};

struct StoryEventRuntimeState {
    std::uint64_t sequence = 0u;
    RecordKey keyword;
    std::vector<PapyrusValue> arguments;
    friend bool operator==(const StoryEventRuntimeState&, const StoryEventRuntimeState&) = default;
};

struct BethesdaSessionConfig {
    importer::fnv::BethesdaGame game = importer::fnv::BethesdaGame::Unknown;
    std::string contentFingerprint;
    std::string scenarioId;
    std::uint32_t randomSeed = 1u;
    ObjectId playerObject;
};

struct BethesdaSessionStep {
    FixedStepResult clock;
    std::uint32_t vmInstructions = 0u;
    std::size_t worldCommands = 0u;
    bool residencyChanged = false;
    RuntimeRenderDeltaBatch renderDeltas;
    std::vector<std::string> diagnostics;
};

struct AnimationActorSnapshot {
    ObjectId object;
    odai::anim::BehaviorGraphSnapshot thirdPerson;
    std::optional<odai::anim::BehaviorGraphSnapshot> firstPerson;
    friend bool operator==(const AnimationActorSnapshot&,
                           const AnimationActorSnapshot&) = default;
};

struct MeleeAttackResult {
    bool accepted = false;
    bool hit = false;
    bool killed = false;
    ObjectId target;
    float damage = 0.0f;
    std::string diagnostic;
};

struct PuzzleDoorActivationResult {
    bool accepted = false;
    bool opened = false;
    bool missingRequiredItem = false;
    bool incorrectCombination = false;
    std::string diagnostic;
};

struct LootTransferResult {
    bool accepted = false;
    std::vector<InventoryEntry> transferred;
    std::string diagnostic;
};

// Persistent modal interaction requested by Actor.ShowGiftMenu. The VM call
// records intent in deterministic simulation state; presentation may open and
// close the actual inventory panel without hiding the request in app-only UI
// state. A save made while the panel is open restores the same participants
// and authored filter.
struct GiftMenuRequestState {
    std::uint64_t sequence = 0u;
    ObjectId actor;
    ObjectId player;
    ObjectId filterList;
    bool playerGives = false;
    bool showStolenItems = false;
    bool useFavorPoints = false;
    friend bool operator==(const GiftMenuRequestState&,
                           const GiftMenuRequestState&) = default;
};

struct GiftTransferResult {
    bool accepted = false;
    std::string diagnostic;
};

struct SkyrimDialogueChoice {
    RecordKey info;
    RecordKey topic;
    RecordKey quest;
    RecordKey branch;
    std::string prompt;
    std::vector<std::string> responses;
};

struct SkyrimDialogueSelectionResult {
    bool accepted = false;
    RecordKey info;
    RecordKey responseInfo;
    std::vector<RecordKey> nextTopics;
    std::vector<std::string> responses;
    std::vector<std::string> diagnostics;
};

class BethesdaSession {
public:
    using QuestReferenceResolver = std::function<std::optional<ObjectId>(std::uint32_t)>;
    using ResolvedFormResolver = std::function<std::optional<ObjectId>(std::uint32_t)>;
    using BeforeSimulationTick = std::function<void(std::uint64_t, double)>;
    bool configure(BethesdaSessionConfig config, std::string& outError);
    bool configureTes3Content(
        std::shared_ptr<const Tes3ContentStore> content, std::string& outError);
    BethesdaSessionStep advance(
        double frameDeltaSeconds, const BeforeSimulationTick& beforeTick = {});
    bool applyScenario(const ScenarioDefinition& scenario, std::string& outError);
    bool registerActorAnimation(
        ObjectId object, std::shared_ptr<const odai::anim::AnimationView> thirdPerson,
        std::shared_ptr<const odai::anim::AnimationView> firstPerson,
        const PhysicsCharacterConfig& physicsConfig, std::string& outError);
    bool registerActorController(
        ObjectId object, const PhysicsCharacterConfig& physicsConfig, std::string& outError);
    bool unregisterActorController(ObjectId object);
    bool unregisterActorAnimation(ObjectId object);
    bool setActorControllerInput(ObjectId object, const PhysicsCharacterInput& input);
    bool addActorImpulse(ObjectId object, const odai::math::Vector3& velocityChange);
    bool setActorAnimationInput(ObjectId object, odai::anim::AnimationInputState input);
    bool queueActorAnimationEvent(ObjectId object, odai::anim::AnimationEvent event);
    // Must be called from BeforeSimulationTick (or another fixed-tick system).
    // Target selection uses Jolt-owned character positions and authored static
    // occlusion; state changes are queued through WorldCommand for this tick.
    [[nodiscard]] MeleeAttackResult performMeleeAttack(
        ObjectId attacker, const odai::math::Vector3& forward,
        float damage = 25.0f, float rangeBethesdaUnits = 180.0f);
    bool rotatePuzzleRing(ObjectId door, std::size_t ringIndex, std::string& outError);
    [[nodiscard]] PuzzleDoorActivationResult activatePuzzleDoor(
        ObjectId player, ObjectId door, const RecordKey& requiredItem,
        const RecordKey& quest, std::int32_t successStage);
    // Binds unique-actor quest aliases to their placed runtime actor and
    // materializes ALCO/ALCA-created inventory exactly once.
    std::size_t bindQuestInventoryForActor(
        ObjectId actor, const RecordKey& actorBase, std::string& outError);
    std::size_t bindDynamicQuestAliasesForObject(
        ObjectId object, std::string& outError);
    [[nodiscard]] LootTransferResult lootObject(ObjectId player, ObjectId source);
    [[nodiscard]] const std::vector<GiftMenuRequestState>& giftMenuRequests() const {
        return m_giftMenuRequests;
    }
    [[nodiscard]] std::vector<GiftMenuRequestState>& giftMenuRequestsForRestore() {
        return m_giftMenuRequests;
    }
    [[nodiscard]] std::uint64_t nextGiftMenuSequence() const {
        return m_nextGiftMenuSequence;
    }
    void setNextGiftMenuSequence(std::uint64_t sequence) {
        m_nextGiftMenuSequence = sequence == 0u ? 1u : sequence;
    }
    [[nodiscard]] GiftTransferResult transferGiftMenuItem(
        std::uint64_t sequence, const RecordKey& item, std::int32_t count);
    bool closeGiftMenu(std::uint64_t sequence, std::string& outError);
    bool registerDialogueTopic(
        SkyrimDialogueTopicDefinition definition, std::string& outError);
    bool registerDialogueBranch(
        SkyrimDialogueBranchDefinition definition, std::string& outError);
    bool registerDialogueInfo(
        SkyrimDialogueInfoDefinition definition, std::string& outError);
    [[nodiscard]] std::vector<SkyrimDialogueChoice> availableDialogueChoices(
        ObjectId speaker, ObjectId player, bool strict = true,
        std::span<const RecordKey> eligibleTopics = {}) const;
    // fragmentFlag is INFO VMAD bit 0 (begin/result) or bit 1 (end). The
    // default runs the authored effect associated with choosing the response.
    [[nodiscard]] SkyrimDialogueSelectionResult selectDialogueInfo(
        const RecordKey& info, ObjectId speaker, ObjectId player,
        std::uint8_t fragmentFlag = 1u, bool strict = true);
    [[nodiscard]] const std::map<RecordKey, SkyrimDialogueTopicDefinition>&
        dialogueTopics() const { return m_dialogueTopics; }
    [[nodiscard]] const std::map<RecordKey, SkyrimDialogueInfoDefinition>&
        dialogueInfos() const { return m_dialogueInfos; }
    [[nodiscard]] const std::map<RecordKey, SkyrimDialogueBranchDefinition>&
        dialogueBranches() const { return m_dialogueBranches; }
    [[nodiscard]] const odai::anim::AnimationStepOutput* actorAnimationOutput(
        ObjectId object, bool firstPerson = false) const;
    [[nodiscard]] BethesdaPhysicsWorld& physics() { return m_physics; }
    [[nodiscard]] const BethesdaPhysicsWorld& physics() const { return m_physics; }
    [[nodiscard]] std::vector<AnimationActorSnapshot> animationSnapshots() const;
    bool restoreAnimationSnapshots(
        std::span<const AnimationActorSnapshot> snapshots, std::string& outError);
    [[nodiscard]] std::vector<PhysicsCharacterSnapshot> physicsSnapshots() const;
    bool restorePhysicsSnapshots(
        std::span<const PhysicsCharacterSnapshot> snapshots, std::string& outError);

    [[nodiscard]] Tes3DialogueResponse startTes3Dialogue(
        Tes3DialogueActorState actor, Tes3DialoguePlayerState player,
        bool strict = true) {
        return m_tes3.startDialogue(std::move(actor), std::move(player), strict);
    }
    [[nodiscard]] std::vector<std::string> tes3DialogueTopics(bool strict = true) const {
        return m_tes3.availableTopics(strict);
    }
    [[nodiscard]] Tes3DialogueResponse selectTes3Topic(
        std::string_view topic, bool strict = true) {
        return m_tes3.selectTopic(topic, strict);
    }
    [[nodiscard]] Tes3DialogueResponse answerTes3Choice(
        std::int32_t choice, bool strict = true) {
        return m_tes3.answerChoice(choice, strict);
    }
    [[nodiscard]] const Tes3JournalQuestState* tes3JournalState(
        std::string_view journalId) const {
        return m_tes3.journal().find(journalId);
    }
    void dispatchTes3GameplayEvent(
        std::string eventName, ObjectId target,
        Tes3Value value = Tes3Value::fromNumber(1.0)) {
        m_tes3.dispatchGameplayEvent(std::move(eventName), std::move(target), std::move(value));
    }

    [[nodiscard]] BethesdaWorld& world() { return m_world; }
    [[nodiscard]] const BethesdaWorld& world() const { return m_world; }
    [[nodiscard]] PapyrusVm& papyrus() { return m_papyrus; }
    [[nodiscard]] const PapyrusVm& papyrus() const { return m_papyrus; }
    [[nodiscard]] FixedStepClock& clock() { return m_clock; }
    [[nodiscard]] const FixedStepClock& clock() const { return m_clock; }
    [[nodiscard]] const BethesdaSessionConfig& config() const { return m_config; }
    [[nodiscard]] ObjectId playerObject() const { return m_playerObject; }
    [[nodiscard]] Tes3Runtime& tes3() { return m_tes3; }
    [[nodiscard]] const Tes3Runtime& tes3() const { return m_tes3; }
    [[nodiscard]] const std::map<std::string, QuestRuntimeState>& quests() const { return m_quests; }
    [[nodiscard]] std::map<std::string, QuestRuntimeState>& questsForRestore() { return m_quests; }
    [[nodiscard]] std::uint32_t randomState() const { return m_randomState; }
    void setRandomState(std::uint32_t state) { m_randomState = state == 0u ? 1u : state; }
    [[nodiscard]] const std::map<std::string, std::int64_t>& statistics() const { return m_statistics; }
    [[nodiscard]] std::map<std::string, std::int64_t>& statisticsForRestore() { return m_statistics; }
    [[nodiscard]] const std::vector<RecordKey>& discoveries() const { return m_discoveries; }
    [[nodiscard]] std::vector<RecordKey>& discoveriesForRestore() { return m_discoveries; }
    [[nodiscard]] const std::map<RecordKey, bool>& scenes() const { return m_scenes; }
    [[nodiscard]] std::map<RecordKey, bool>& scenesForRestore() { return m_scenes; }
    [[nodiscard]] const RecordKey& forcedWeather() const { return m_forcedWeather; }
    RecordKey& forcedWeatherForRestore() { return m_forcedWeather; }
    [[nodiscard]] const std::map<RecordKey, LocationRuntimeState>& locations() const {
        return m_locations;
    }
    [[nodiscard]] std::map<RecordKey, LocationRuntimeState>& locationsForRestore() {
        return m_locations;
    }
    [[nodiscard]] const std::map<RecordKey, float>& globalVariables() const {
        return m_globalVariables;
    }
    [[nodiscard]] std::map<RecordKey, float>& globalVariablesForRestore() {
        return m_globalVariables;
    }
    [[nodiscard]] const std::vector<StoryEventRuntimeState>& storyEvents() const {
        return m_storyEvents;
    }
    [[nodiscard]] std::vector<StoryEventRuntimeState>& storyEventsForRestore() {
        return m_storyEvents;
    }
    [[nodiscard]] std::uint64_t nextStoryEventSequence() const {
        return m_nextStoryEventSequence;
    }
    void setNextStoryEventSequence(std::uint64_t sequence) {
        m_nextStoryEventSequence = sequence == 0u ? 1u : sequence;
    }
    [[nodiscard]] const std::vector<std::string>& scriptDebugLogs() const {
        return m_scriptDebugLogs;
    }
    [[nodiscard]] std::vector<std::string>& scriptDebugLogsForRestore() {
        return m_scriptDebugLogs;
    }
    void setScenePlaying(const RecordKey& scene, bool playing);
    void setResolvedFormResolver(ResolvedFormResolver resolver) {
        m_resolvedFormResolver = std::move(resolver);
    }
    bool registerLocation(
        RecordKey location, RecordKey parent, std::vector<RecordKey> keywords,
        std::string& outError);
    bool registerGlobalVariable(RecordKey variable, float initialValue, std::string& outError);
    void setLocationLoaded(const RecordKey& location, bool loaded);
    void clearLoadedLocations();

    QuestRuntimeState& quest(const std::string& editorId);
    [[nodiscard]] const QuestRuntimeState* findQuest(const std::string& editorId) const;
    [[nodiscard]] QuestRuntimeState* findQuest(const ObjectId& questObject);
    [[nodiscard]] const QuestRuntimeState* findQuest(const ObjectId& questObject) const;
    [[nodiscard]] QuestAliasRuntimeState* findQuestAlias(const ObjectId& aliasHandle);
    [[nodiscard]] const QuestAliasRuntimeState* findQuestAlias(const ObjectId& aliasHandle) const;
    bool bindQuestAliasTarget(
        const ObjectId& questObject, std::int32_t aliasId,
        ObjectId target, std::string& outError);
    bool registerQuestDefinition(
        const SkyrimQuestDefinition& definition,
        const QuestReferenceResolver& referenceResolver,
        std::string& outError);
    void setQuestStage(const std::string& editorId, std::int32_t stage, bool completed = false);
    [[nodiscard]] std::uint64_t deterministicHash() const;

private:
    struct QuestStageFragmentRuntime {
        VmadQuestFragment fragment;
        std::vector<Condition> conditions;
    };
    struct PendingQuestAliasEvent {
        ObjectId alias;
        std::string event;
        std::vector<PapyrusValue> arguments;
    };
    struct ActorAnimationRuntime {
        std::shared_ptr<const odai::anim::AnimationView> thirdPersonView;
        std::shared_ptr<const odai::anim::AnimationView> firstPersonView;
        odai::anim::BehaviorGraphInstance thirdPerson;
        odai::anim::BehaviorGraphInstance firstPerson;
        odai::anim::AnimationInputState input;
        odai::anim::AnimationStepOutput thirdPersonOutput;
        odai::anim::AnimationStepOutput firstPersonOutput;
    };
    void registerSkyrimNatives();
    [[nodiscard]] Tes3NativeResult executeTes3WorldNative(const Tes3NativeCall& call);
    void queueQuestAliasEvent(
        ObjectId alias, std::string event, std::vector<PapyrusValue> arguments);
    void flushQuestAliasEvents();
    void simulateTick(std::uint64_t tick, double stepSeconds, BethesdaSessionStep& result);
    [[nodiscard]] ConditionEvaluation evaluateDialogueConditions(
        const SkyrimDialogueInfoDefinition& info,
        ObjectId speaker, ObjectId player, bool strict) const;

    BethesdaSessionConfig m_config;
    FixedStepClock m_clock;
    BethesdaWorld m_world;
    BethesdaPhysicsWorld m_physics;
    PapyrusVm m_papyrus;
    Tes3Runtime m_tes3;
    ObjectId m_playerObject;
    std::map<std::string, QuestRuntimeState> m_quests;
    std::map<std::string, std::vector<QuestStageFragmentRuntime>> m_questStageFragments;
    std::map<RecordKey, SkyrimDialogueTopicDefinition> m_dialogueTopics;
    std::map<RecordKey, SkyrimDialogueBranchDefinition> m_dialogueBranches;
    std::map<RecordKey, SkyrimDialogueInfoDefinition> m_dialogueInfos;
    std::map<std::string, std::int64_t> m_statistics;
    std::vector<RecordKey> m_discoveries;
    std::map<RecordKey, bool> m_scenes;
    RecordKey m_forcedWeather;
    std::map<RecordKey, LocationRuntimeState> m_locations;
    std::map<RecordKey, float> m_globalVariables;
    std::vector<StoryEventRuntimeState> m_storyEvents;
    std::vector<GiftMenuRequestState> m_giftMenuRequests;
    std::vector<std::string> m_scriptDebugLogs;
    std::vector<std::string> m_pendingDiagnostics;
    std::vector<PendingQuestAliasEvent> m_pendingQuestAliasEvents;
    ResolvedFormResolver m_resolvedFormResolver;
    std::map<ObjectId, ActorAnimationRuntime> m_actorAnimations;
    std::map<ObjectId, AnimationActorSnapshot> m_pendingAnimationSnapshots;
    std::map<ObjectId, PhysicsCharacterSnapshot> m_pendingPhysicsSnapshots;
    std::uint64_t m_nextStoryEventSequence = 1u;
    std::uint64_t m_nextGiftMenuSequence = 1u;
    std::uint32_t m_randomState = 1u;
    bool m_configured = false;
};

}  // namespace odai::bethesda
