#include "games/newvegas/bethesda_app.h"

#include "bethesda/save_game.h"
#include "bethesda/record_resolver.h"
#include "bethesda/scenario.h"
#include "bethesda/skyrim_scenario_content.h"
#include "bethesda/vmad_reader.h"
#include "bethesda/whiterun_presentation.h"

#include "import/fnv/land_lod.h"

#include "render/upscale/upscale_policy.h"

#include "import/fnv/dialogue_records.h"

#include "import/dds.h"
#include "games/newvegas/newvegas_ogg.h"
#include "import/fnv/bsa_archive.h"

#include <fstream>
#include <sstream>
#include <random>
#include <chrono>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

#include "core/log.h"
#include "import/fnv/asset_source.h"
#include "import/fnv/nif_scene.h"
#include "ui/ui_types.h"

#include <cstdio>

#include <GLFW/glfw3.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <cctype>
#include <cstdlib>
#include <cstring>

namespace odai::games::newvegas {

namespace {

constexpr float kPi = 3.14159265358979323846f;

// Bethesda world units: 1 unit is about 1.43 cm, so ~70 units to the metre.
constexpr float kWalkUnitsPerSecond = 400.0f;
constexpr float kSprintMultiplier = 4.0f;
constexpr float kMouseSensitivity = 0.12f;
constexpr float kPitchLimitDegrees = 89.0f;
constexpr float kEyeHeightUnits = 120.0f;
// Fallout's own jump is about 1 metre at ~70 units/metre. v = sqrt(2*g*h) with a
// gravity that keeps the arc short enough to feel like a jump rather than a
// hop on the moon.
constexpr float kGravityUnitsPerSecondSq = 2600.0f;
constexpr float kJumpUnitsPerSecond = 620.0f;

// LAND posts sit on a regular 128-unit lattice (kLandPostSpacing), and the
// cooked terrain mesh preserves that, so the ground can be sampled from a plain
// 2D grid instead of a ray cast. raycastImportedScene is the obvious tool and
// the wrong one here: it is brute force over every triangle by design (see its
// header), which is fine at click rate but this scene has ~3.7M of them and the
// camera needs a ground height every frame.
constexpr float kGroundGridSpacing = 128.0f;

std::string toLowerAscii(std::string value) {
    for (char& c : value) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return value;
}

const bethesda::Tes3SubrecordData* tes3Subrecord(
    const bethesda::Tes3NamedRecord& record, std::string_view type) {
    const auto found = std::find_if(
        record.subrecords.begin(), record.subrecords.end(),
        [&](const bethesda::Tes3SubrecordData& subrecord) {
            return subrecord.type == type;
        });
    return found == record.subrecords.end() ? nullptr : &*found;
}

std::string tes3SubrecordText(
    const bethesda::Tes3NamedRecord& record, std::string_view type,
    std::string_view encoding) {
    const bethesda::Tes3SubrecordData* subrecord = tes3Subrecord(record, type);
    if (subrecord == nullptr) return {};
    std::string_view bytes(
        reinterpret_cast<const char*>(subrecord->data.data()), subrecord->data.size());
    while (!bytes.empty() && bytes.back() == '\0') bytes.remove_suffix(1u);
    return bethesda::decodeTes3Text(bytes, encoding);
}

// TES3 humanoids are assembled from BODY records rather than naming one NPC
// model. Keep that authored assembly at the presentation boundary: the content
// store remains immutable and the resulting figure still uses the same actor
// upload path as every later Bethesda game.
bool tes3ActorGeometry(
    const bethesda::Tes3ContentStore& content,
    const bethesda::Tes3ActorDefinition& actor,
    std::string& outSkeleton, std::vector<std::string>& outParts,
    std::vector<std::string>& outRigidAttachmentBones,
    std::string& outWhy) {
    outSkeleton.clear();
    outParts.clear();
    outRigidAttachmentBones.clear();
    const bethesda::Tes3NamedRecord* actorRecord = content.findRecord(
        actor.creature ? "CREA" : "NPC_", actor.id);
    if (actorRecord == nullptr) {
        outWhy = "actor definition has no retained source record";
        return false;
    }
    if (actor.creature) {
        const std::string model =
            tes3SubrecordText(*actorRecord, "MODL", content.encoding());
        if (model.empty()) {
            outWhy = "creature has no MODL";
            return false;
        }
        outSkeleton = model;
        outParts.push_back(model);
        outRigidAttachmentBones.emplace_back();
        return true;
    }

    // TES3 NPCs normally name base_anim.nif, but beast races can select a
    // compatible animation-bank variant with extra bones. Hul, for example,
    // names base_animKnA.nif; forcing her onto base_anim.nif leaves the
    // Argonian-only skin weights unresolved. Those vertices then miss the
    // actor-world transform in the GPU skinning pass and pull their triangles
    // into kilometre-long strips toward the world origin.
    const std::string authoredSkeleton =
        tes3SubrecordText(*actorRecord, "MODL", content.encoding());
    outSkeleton = authoredSkeleton.empty() ? "base_anim.nif" : authoredSkeleton;
    const auto appendBodyModel = [&](std::string_view bodyId,
                                     std::string_view rigidAttachmentBone) {
        if (bodyId.empty()) return;
        const bethesda::Tes3NamedRecord* body = content.findRecord("BODY", bodyId);
        if (body == nullptr) return;
        const std::string model = tes3SubrecordText(*body, "MODL", content.encoding());
        if (model.empty()) return;
        // A single rigid model is commonly reused for the left and right
        // equipment slots. Deduplicating on path alone deletes one limb; the
        // authored attachment pivot is part of the identity of an assembled
        // BODY piece. Fully skinned parts use an empty attachment and still
        // deduplicate exactly as before.
        for (std::size_t part = 0u; part < outParts.size(); ++part) {
            if (outParts[part] == model &&
                outRigidAttachmentBones[part] == rigidAttachmentBone) {
                return;
            }
        }
        outParts.push_back(model);
        outRigidAttachmentBones.emplace_back(rigidAttachmentBone);
    };

    // The NPC selects its authored face and hair explicitly.
    appendBodyModel(
        tes3SubrecordText(*actorRecord, "BNAM", content.encoding()), "Head");
    appendBodyModel(
        tes3SubrecordText(*actorRecord, "KNAM", content.encoding()), "Head");

    bool female = false;
    if (const bethesda::Tes3SubrecordData* flags = tes3Subrecord(*actorRecord, "FLAG");
        flags != nullptr && flags->data.size() >= sizeof(std::uint32_t)) {
        std::uint32_t value = 0u;
        std::memcpy(&value, flags->data.data(), sizeof(value));
        female = (value & 0x1u) != 0u;
    }

    // NPCO inventory is also the authored outfit in TES3. CLOT/ARMO records
    // map each covered body slot to a male BNAM and female CNAM BODY record.
    // Resolve those before bare skin so robes and uniforms replace, rather
    // than z-fight with, the body parts they cover.
    std::array<bool, 13> covered{};
    for (const auto& [itemKey, count] : actor.inventory) {
        if (count <= 0 || (itemKey.recordType != "CLOT" && itemKey.recordType != "ARMO")) {
            continue;
        }
        const auto itemIt = content.namedRecords().find(itemKey);
        if (itemIt == content.namedRecords().end()) continue;
        std::array<std::string, 32> maleParts;
        std::array<std::string, 32> femaleParts;
        std::uint32_t currentPart = 0xffffffffu;
        for (const bethesda::Tes3SubrecordData& subrecord : itemIt->second.subrecords) {
            if (subrecord.type == "INDX" && !subrecord.data.empty()) {
                currentPart = subrecord.data[0];
                continue;
            }
            if (currentPart >= maleParts.size() ||
                (subrecord.type != "BNAM" && subrecord.type != "CNAM")) continue;
            std::string_view bytes(
                reinterpret_cast<const char*>(subrecord.data.data()), subrecord.data.size());
            while (!bytes.empty() && bytes.back() == '\0') bytes.remove_suffix(1u);
            std::string& destination = subrecord.type == "BNAM"
                ? maleParts[currentPart]
                : femaleParts[currentPart];
            destination = bethesda::decodeTes3Text(bytes, content.encoding());
        }
        // Equipment distinguishes left/right slots while BODY skin records do
        // not. Collapse the equipment slot to the BODY part it replaces.
        static constexpr std::array<std::uint8_t, 25> kEquipmentToBody = {
            0, 1, 2, 3, 4, 4, 5, 5, 6, 6, 5, 7, 7,
            8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 8, 8};
        static constexpr std::array<std::string_view, 25>
            kEquipmentAttachmentBone = {
                "Head", "Head", "Neck", "Chest", "Groin", "Groin",
                "Right Hand", "Left Hand", "Right Wrist", "Left Wrist",
                "Shield", "Right Forearm", "Left Forearm",
                "Right Upper Arm", "Left Upper Arm", "Right Foot", "Left Foot",
                "Right Ankle", "Left Ankle", "Right Knee", "Left Knee",
                "Right Upper Leg", "Left Upper Leg", "Right Clavicle",
                "Left Clavicle"};
        for (std::size_t part = 0u; part < maleParts.size(); ++part) {
            const std::string& bodyId = female && !femaleParts[part].empty()
                ? femaleParts[part]
                : maleParts[part];
            if (bodyId.empty()) continue;
            const std::size_t before = outParts.size();
            appendBodyModel(
                bodyId,
                part < kEquipmentAttachmentBone.size()
                    ? kEquipmentAttachmentBone[part]
                    : std::string_view{});
            if (part < kEquipmentToBody.size()) {
                const std::size_t bodyPart = kEquipmentToBody[part];
                covered[bodyPart] = covered[bodyPart] || outParts.size() != before;
            }
        }
    }

    // One normal skin part for each authored body slot. BYDT is
    // {part, vampire, flags, type}; type 0 is skin and flag bit 0 is female.
    // Head/hair (slots 0/1) stay under the NPC's explicit BNAM/KNAM choice.
    //
    // Unlike later games, the attachment is not named by the BODY NIF. The
    // small rigid meshes are authored around the origin of the BODY slot and
    // the engine parents them to the corresponding helper node in base_anim.
    // Bilateral slots deliberately append the same mesh twice with different
    // attachment identities. Fully skinned aggregate records (notably the
    // hand/chest skin bundle) ignore the attachment in buildSkinnedActor and
    // therefore remain a single part.
    const auto appendBareBodyModel = [&](std::string_view bodyId, std::uint8_t part) {
        switch (part) {
            case 2u: appendBodyModel(bodyId, "Neck"); break;
            case 3u: appendBodyModel(bodyId, "Chest"); break;
            case 4u: appendBodyModel(bodyId, "Groin"); break;
            case 6u:
                appendBodyModel(bodyId, "Right Wrist");
                appendBodyModel(bodyId, "Left Wrist");
                break;
            case 7u:
                appendBodyModel(bodyId, "Right Forearm");
                appendBodyModel(bodyId, "Left Forearm");
                break;
            case 8u:
                appendBodyModel(bodyId, "Right Upper Arm");
                appendBodyModel(bodyId, "Left Upper Arm");
                break;
            case 9u:
                appendBodyModel(bodyId, "Right Foot");
                appendBodyModel(bodyId, "Left Foot");
                break;
            case 10u:
                appendBodyModel(bodyId, "Right Ankle");
                appendBodyModel(bodyId, "Left Ankle");
                break;
            case 11u:
                appendBodyModel(bodyId, "Right Knee");
                appendBodyModel(bodyId, "Left Knee");
                break;
            case 12u:
                appendBodyModel(bodyId, "Right Upper Leg");
                appendBodyModel(bodyId, "Left Upper Leg");
                break;
            default:
                // Some races bundle multiple weighted regions in one skin
                // BODY record. Leave those on their authored NiSkin weights.
                appendBodyModel(bodyId, {});
                break;
        }
    };
    std::array<bool, 13> selected = covered;
    for (const auto& [key, candidate] : content.namedRecords()) {
        (void)key;
        if (candidate.record.recordType != "BODY") continue;
        const bethesda::Tes3SubrecordData* bydt = tes3Subrecord(candidate, "BYDT");
        if (bydt == nullptr || bydt->data.size() < 4u) continue;
        const std::uint8_t part = bydt->data[0];
        const bool candidateFemale = (bydt->data[2] & 0x1u) != 0u;
        const std::uint8_t bodyType = bydt->data[3];
        if (part < 2u || part >= selected.size() || selected[part] ||
            candidateFemale != female || bodyType != 0u) continue;
        const std::string race = tes3SubrecordText(candidate, "FNAM", content.encoding());
        if (toLowerAscii(race) != toLowerAscii(actor.race)) continue;
        const std::size_t before = outParts.size();
        appendBareBodyModel(candidate.id, part);
        selected[part] = outParts.size() != before;
    }
    if (outParts.empty()) {
        outWhy = "NPC head, hair and race BODY records resolve no models";
        return false;
    }
    return true;
}

const bethesda::VmadScriptAttachment* findVmadScript(
    const bethesda::VmadAttachments& attachments, std::string_view className) {
    const std::string wanted = toLowerAscii(std::string(className));
    const auto found = std::find_if(
        attachments.scripts.begin(), attachments.scripts.end(),
        [&](const bethesda::VmadScriptAttachment& script) {
            return toLowerAscii(script.className) == wanted;
        });
    return found == attachments.scripts.end() ? nullptr : &*found;
}

const bethesda::VmadProperty* findVmadProperty(
    const bethesda::VmadScriptAttachment& script, std::string_view propertyName) {
    const std::string wanted = toLowerAscii(std::string(propertyName));
    const auto found = std::find_if(
        script.properties.begin(), script.properties.end(),
        [&](const bethesda::VmadProperty& property) {
            return toLowerAscii(property.name) == wanted;
        });
    return found == script.properties.end() ? nullptr : &*found;
}

bool readVmadObjectProperty(
    const bethesda::VmadScriptAttachment& script, std::string_view propertyName,
    std::uint32_t& outFormId) {
    const bethesda::VmadProperty* property = findVmadProperty(script, propertyName);
    if (property == nullptr || property->value.type != bethesda::VmadValueType::Object ||
        property->value.object.formId == 0u ||
        property->value.object.alias != 0xffffu) {
        return false;
    }
    outFormId = property->value.object.formId;
    return true;
}

bool readVmadIntegerProperty(
    const bethesda::VmadScriptAttachment& script, std::string_view propertyName,
    std::int32_t& outValue) {
    const bethesda::VmadProperty* property = findVmadProperty(script, propertyName);
    if (property == nullptr || property->value.type != bethesda::VmadValueType::Integer) {
        return false;
    }
    outValue = property->value.integer;
    return true;
}

std::string importedReferenceSourceId(std::uint32_t resolvedFormId) {
    std::ostringstream out;
    out << "refr_" << std::hex << std::uppercase << resolvedFormId;
    return out.str();
}

bool keyDown(GLFWwindow* window, int key) {
    return glfwGetKey(window, key) == GLFW_PRESS;
}

// A demo row should contain complete walking silhouettes, not every partially
// resolvable record near the spawn. A missing torso is detectable without
// knowing an actor's EditorID: complete humanoid geometry has meaningful
// vertex coverage through the middle of its standing bounds, while a detached
// head plus boots leaves that entire band empty.
bool hasHumanoidTorsoCoverage(const SkinnedActor& actor) {
    if (!actor.wanders || actor.character.vertices.size() < 100u) {
        return false;
    }
    float minY = std::numeric_limits<float>::max();
    float maxY = std::numeric_limits<float>::lowest();
    for (const render::ImportedSkinnedMeshVertex& vertex : actor.character.vertices) {
        minY = std::min(minY, vertex.position[1]);
        maxY = std::max(maxY, vertex.position[1]);
    }
    const float height = maxY - minY;
    if (!(height > 1.0f)) {
        return false;
    }
    const float torsoMin = minY + (height * 0.35f);
    const float torsoMax = minY + (height * 0.70f);
    std::size_t torsoVertices = 0u;
    for (const render::ImportedSkinnedMeshVertex& vertex : actor.character.vertices) {
        torsoVertices += vertex.position[1] >= torsoMin && vertex.position[1] <= torsoMax ? 1u : 0u;
    }
    return torsoVertices >= 100u &&
        torsoVertices * 20u >= actor.character.vertices.size();
}

float srgbChannelToLinear(float srgb) {
    return srgb <= 0.04045f ? (srgb / 12.92f)
                            : std::pow((srgb + 0.055f) / 1.055f, 2.4f);
}

std::uint64_t physicsResidencyToken(const importer::CellCoord& cell) {
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(cell.x)) << 32u) |
        static_cast<std::uint32_t>(cell.z);
}

bethesda::RuntimeAiState runtimeAiStateFor(
    const SkinnedActor& actor,
    const importer::fnv::FalloutLoadOrder& loadOrder) {
    bethesda::RuntimeAiState state;
    state.walking = actor.walking;
    state.projectedToNavigation = actor.projectedToNavigation;
    std::copy_n(actor.wanderOrigin, 3u, state.wanderOrigin.begin());
    std::copy_n(actor.wanderTarget, 3u, state.wanderTarget.begin());
    state.path.reserve(actor.wanderPath.size());
    for (const ActorNavigationStep& step : actor.wanderPath) {
        bethesda::RuntimePathStep saved;
        saved.kind = step.kind == ActorNavigationStepKind::ActivateDoor
            ? bethesda::RuntimePathStepKind::ActivateDoor
            : bethesda::RuntimePathStepKind::Walk;
        saved.position = {step.position.x, step.position.y, step.position.z};
        saved.arrivalPosition = {
            step.arrivalPosition.x, step.arrivalPosition.y, step.arrivalPosition.z};
        if (step.doorReferenceFormId != 0u) {
            std::string error;
            (void)bethesda::stableRecordKey(
                loadOrder, step.doorReferenceFormId, saved.door, error);
        }
        state.path.push_back(std::move(saved));
    }
    state.pathIndex = actor.wanderPathIndex;
    state.pauseSeconds = std::max(0.0f, actor.wanderPauseSeconds);
    state.randomState = actor.wanderRng;
    state.scriptedMoveActive = actor.scriptedMoveActive;
    state.scriptedMoveArrived = actor.scriptedMoveArrived;
    state.scriptedMoveRevision = actor.scriptedMoveRevision;
    return state;
}

void restoreRuntimeAiState(
    const bethesda::RuntimeAiState& state,
    const importer::fnv::FalloutLoadOrder& loadOrder,
    SkinnedActor& actor) {
    actor.walking = state.walking;
    actor.projectedToNavigation = state.projectedToNavigation;
    std::copy(state.wanderOrigin.begin(), state.wanderOrigin.end(), actor.wanderOrigin);
    std::copy(state.wanderTarget.begin(), state.wanderTarget.end(), actor.wanderTarget);
    actor.wanderPath.clear();
    actor.wanderPath.reserve(state.path.size());
    for (const bethesda::RuntimePathStep& saved : state.path) {
        ActorNavigationStep step;
        step.kind = saved.kind == bethesda::RuntimePathStepKind::ActivateDoor
            ? ActorNavigationStepKind::ActivateDoor
            : ActorNavigationStepKind::Walk;
        step.position = {saved.position[0], saved.position[1], saved.position[2]};
        step.arrivalPosition = {saved.arrivalPosition[0], saved.arrivalPosition[1],
            saved.arrivalPosition[2]};
        if (saved.door.valid()) {
            std::string error;
            (void)bethesda::resolvedFormId(
                loadOrder, saved.door, step.doorReferenceFormId, error);
        }
        actor.wanderPath.push_back(std::move(step));
    }
    actor.wanderPathIndex = static_cast<std::size_t>(
        std::min<std::uint64_t>(state.pathIndex, actor.wanderPath.size()));
    actor.wanderPauseSeconds = state.pauseSeconds;
    actor.wanderRng = state.randomState;
    actor.scriptedMoveActive = state.scriptedMoveActive;
    actor.scriptedMoveArrived = state.scriptedMoveArrived;
    actor.scriptedMoveRevision = state.scriptedMoveRevision;
}

}  // namespace

float BethesdaApp::verticalFovDegreesFor(float horizontalFovDegrees, float aspectRatio) {
    // Guard the degenerate frame: a zero or negative aspect would otherwise
    // divide the tangent into infinity and hand the projection a NaN, which
    // renders as an empty screen rather than as an error.
    const float safeAspect = (aspectRatio > 0.0001f) ? aspectRatio : (16.0f / 9.0f);
    const float clampedHorizontal = std::clamp(horizontalFovDegrees, 1.0f, 179.0f);
    const float halfHorizontalRadians = (clampedHorizontal * 0.5f) * (kPi / 180.0f);
    const float halfVerticalRadians = std::atan(std::tan(halfHorizontalRadians) / safeAspect);
    return halfVerticalRadians * 2.0f * (180.0f / kPi);
}

audio::AudioConfig BethesdaApp::audioConfig() const {
    audio::AudioConfig config;
    if (m_captureAudioRequested) {
        config.offlineMix = true;
        config.offlineSampleRate = 48000u;
        config.offlineChannels = 2u;
        // Fallout capture keeps licensed radio muted. The Whiterun market is
        // explicitly a Skyrim audiovisual showcase, so its retail exploration
        // score belongs in the deterministic offline mix with the rain and
        // authored city ambience.
        config.musicVolume = m_whiterunMarketReferenceShowcase ? 0.55f : 0.0f;
    }
    return config;
}

void BethesdaApp::buildGroundHeightField(const importer::ImportedScene& scene) {
    m_groundHeights.clear();
    if (scene.meshes.empty() || scene.meshes.front().name != "terrain") {
        VOX_LOGW("newvegas") << "no terrain mesh; camera will not be ground-clamped";
        return;
    }
    const importer::ImportedSceneMesh& terrain = scene.meshes.front();
    if (terrain.vertices.empty()) {
        return;
    }

    float minX = std::numeric_limits<float>::max();
    float minZ = std::numeric_limits<float>::max();
    float maxX = std::numeric_limits<float>::lowest();
    float maxZ = std::numeric_limits<float>::lowest();
    for (const importer::ImportedSceneVertex& vertex : terrain.vertices) {
        minX = std::min(minX, vertex.position[0]);
        maxX = std::max(maxX, vertex.position[0]);
        minZ = std::min(minZ, vertex.position[2]);
        maxZ = std::max(maxZ, vertex.position[2]);
    }

    m_groundOriginX = minX;
    m_groundOriginZ = minZ;
    m_groundCols = static_cast<int>(std::lround((maxX - minX) / kGroundGridSpacing)) + 1;
    m_groundRows = static_cast<int>(std::lround((maxZ - minZ) / kGroundGridSpacing)) + 1;
    if (m_groundCols <= 1 || m_groundRows <= 1) {
        m_groundCols = 0;
        m_groundRows = 0;
        return;
    }

    // Cells overlap at their shared edge posts and adjacent cells can disagree
    // slightly there, so keep the highest sample per lattice point: standing a
    // few units high reads as a step, sinking reads as falling through the world.
    m_groundHeights.assign(
        static_cast<std::size_t>(m_groundCols) * static_cast<std::size_t>(m_groundRows),
        -std::numeric_limits<float>::max());
    for (const importer::ImportedSceneVertex& vertex : terrain.vertices) {
        const int col = static_cast<int>(std::lround((vertex.position[0] - minX) / kGroundGridSpacing));
        const int row = static_cast<int>(std::lround((vertex.position[2] - minZ) / kGroundGridSpacing));
        if (col < 0 || col >= m_groundCols || row < 0 || row >= m_groundRows) {
            continue;
        }
        float& slot = m_groundHeights[(static_cast<std::size_t>(row) * m_groundCols) + col];
        slot = std::max(slot, vertex.position[1]);
    }
    VOX_LOGI("newvegas") << "ground height field: " << m_groundCols << "x" << m_groundRows
                         << " posts at " << kGroundGridSpacing << " units";
}

bool BethesdaApp::groundHeightAt(float x, float z, float& outHeight) const {
    // Streaming owns its own terrain: the whole-scene height field below is
    // built once from a loaded .bin and has nothing in it when cells arrive and
    // leave continuously.
    if (m_streamer) {
        // referenceY is the player's foot height, so ceilings and upper
        // storeys above them are not mistaken for the ground.
        return m_collision.groundHeight(x, z, m_cameraY - kEyeHeightUnits, outHeight);
    }
    if (m_groundHeights.empty()) {
        return false;
    }
    const float gridX = (x - m_groundOriginX) / kGroundGridSpacing;
    const float gridZ = (z - m_groundOriginZ) / kGroundGridSpacing;
    const int col = static_cast<int>(std::floor(gridX));
    const int row = static_cast<int>(std::floor(gridZ));
    if (col < 0 || row < 0 || col + 1 >= m_groundCols || row + 1 >= m_groundRows) {
        return false;
    }
    const float tx = gridX - static_cast<float>(col);
    const float tz = gridZ - static_cast<float>(row);
    const auto sample = [this](int c, int r) {
        return m_groundHeights[(static_cast<std::size_t>(r) * m_groundCols) + c];
    };
    const float h00 = sample(col, row);
    const float h10 = sample(col + 1, row);
    const float h01 = sample(col, row + 1);
    const float h11 = sample(col + 1, row + 1);
    // A lattice point no terrain vertex landed on is a hole, not a height of
    // -FLT_MAX; bilinear-blending one would yank the camera through the floor.
    const float unset = -std::numeric_limits<float>::max();
    if (h00 == unset || h10 == unset || h01 == unset || h11 == unset) {
        return false;
    }
    const float bottom = h00 + ((h10 - h00) * tx);
    const float top = h01 + ((h11 - h01) * tx);
    outHeight = bottom + ((top - bottom) * tz);
    return true;
}

bool BethesdaApp::loadScene(
    const std::filesystem::path& path, const float* arrivalPosition, const float* arrivalYawDegrees
) {
    // Local, not a member: uploadImportedScene deep-copies the whole scene, so
    // keeping a second copy alive for the process lifetime costs ~100 MB of
    // resident memory that nothing ever reads again.
    //
    // The full loader, NOT loadImportedSceneRuntime. The runtime one keeps only
    // the packed stream: it skips the mesh block outright and reads instances
    // just to discard them. Both are needed here and neither failure is visible
    // -- the containers come back empty rather than erroring -- so with the
    // runtime loader the ground height field is never built (camera stays in fly
    // mode) and the town centroid finds nothing (spawn falls back to the middle
    // of the map).
    importer::ImportedScene scene;
    if (!importer::loadImportedScene(path, scene)) {
        VOX_LOGE("newvegas") << "failed to load scene '" << path.string()
                             << "': " << importer::getImportedSceneLastError();
        return false;
    }
    VOX_LOGI("newvegas") << "loaded " << path.string() << " (" << scene.packedVertices.size()
                         << " vertices, " << scene.textures.size() << " textures, "
                         << scene.doors.size() << " doors)";

    // Diagnostic A/B: ODAI_FNV_AS_CHUNK routes the same scene through the
    // streaming chunk path instead of the whole-scene upload, with everything
    // else identical. Isolates "the geometry is wrong" from "the chunk path is
    // wrong" -- they look the same on screen.
    if (std::getenv("ODAI_FNV_AS_CHUNK") != nullptr) {
        if (m_renderer.addImportedSceneChunk(scene) ==
            render::Renderer::kInvalidImportedChunkIndex) {
            VOX_LOGE("newvegas") << "failed to add scene as a chunk";
            return false;
        }
        VOX_LOGI("newvegas") << "ODAI_FNV_AS_CHUNK: uploaded via addImportedSceneChunk";
    } else if (!m_renderer.uploadImportedScene(scene)) {
        VOX_LOGE("newvegas") << "failed to upload scene to the renderer";
        return false;
    }
    // ODAI_FNV_CHUNK_TEST exercises the streaming add/remove path before a real
    // cell streamer exists to drive it. It re-adds the scene just loaded as a
    // second resident chunk and then evicts it, which is the only way today to
    // check the three invariants that matter and all fail silently:
    //   * every texture is shared, so the second add must upload zero of them
    //     (the refcount table returns the resident slot instead);
    //   * the geometry arena must grow and copy, leaving chunk 0 renderable;
    //   * eviction must return the arena ranges and drop the texture refcounts
    //     back, restoring exactly the pre-test state;
    //   * eviction must release the chunk's punctual lights. These used to be
    //     appended straight into one flat list with no record of which chunk
    //     owned them, so they were never released -- invisible until the
    //     64-light budget filled up with lights from cells long since left.
    // Loading the same scene twice means the two chunks occupy the same space,
    // so the screen should look unchanged throughout -- which is the point.
    if (std::getenv("ODAI_FNV_CHUNK_TEST") != nullptr) {
        VOX_LOGI("newvegas") << "chunk test: live chunks before add = "
                             << m_renderer.liveImportedSceneChunkCount()
                             << ", lights = " << m_renderer.importedLocalLightCount();
        const std::size_t testChunk = m_renderer.addImportedSceneChunk(scene);
        if (testChunk == render::Renderer::kInvalidImportedChunkIndex) {
            VOX_LOGE("newvegas") << "chunk test: addImportedSceneChunk failed";
        } else {
            VOX_LOGI("newvegas") << "chunk test: added chunk " << testChunk
                                 << ", live chunks = " << m_renderer.liveImportedSceneChunkCount()
                                 << ", lights = " << m_renderer.importedLocalLightCount();
            m_renderer.removeImportedSceneChunk(testChunk);
            VOX_LOGI("newvegas") << "chunk test: removed chunk " << testChunk
                                 << ", live chunks = " << m_renderer.liveImportedSceneChunkCount()
                                 << ", lights = " << m_renderer.importedLocalLightCount();
        }
    }

    const bool interior = importer::importedSceneSourceTagIsInterior(scene.sourceTag);
    m_renderer.setImportedSceneInteriorMode(interior);
    m_doors = scene.doors;

    buildGroundHeightField(scene);

    // Spawn standing in Goodsprings rather than hovering over the map.
    //
    // The previous spawn put the camera at boundsMax[1] — above the highest peak
    // in the cooked region — pitched 35 degrees down. That framed the terrain but
    // left no horizon on screen at all, which is why the sky appeared to be
    // missing: the skybox draws with VK_COMPARE_OP_EQUAL against a reversed-Z
    // depth buffer, so it fills exactly the pixels no geometry covered, and from
    // up there geometry covered all of them.
    //
    // The anchor is the centroid of the town's own architecture rather than a
    // hand-entered coordinate, so it stays right if the cooked grid moves. Note
    // this lands you in the middle of Goodsprings by the houses, not on Doc
    // Mitchell's doorstep specifically -- picking out that one building needs its
    // formID from the GECK, which is not something the cooked scene records.
    float spawnX = (scene.boundsMin[0] + scene.boundsMax[0]) * 0.5f;
    float spawnZ = (scene.boundsMin[2] + scene.boundsMax[2]) * 0.5f;
    double townX = 0.0;
    double townZ = 0.0;
    std::size_t townCount = 0;
    for (const importer::ImportedSceneInstance& instance : scene.instances) {
        std::string path = instance.modelPath;
        std::transform(path.begin(), path.end(), path.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        if (path.find("goodsprings") == std::string::npos) {
            continue;
        }
        // Row-major 4x4 with translation in the last COLUMN: the cooker's
        // writeTransform puts it at 3/7/11, not the 12/13/14 a column-major
        // layout would use. Reading 12/14 gets the bottom row, which is all
        // zeroes here, so the centroid silently collapses to the origin.
        townX += static_cast<double>(instance.transform[3]);
        townZ += static_cast<double>(instance.transform[11]);
        ++townCount;
    }
    if (townCount > 0) {
        spawnX = static_cast<float>(townX / static_cast<double>(townCount));
        spawnZ = static_cast<float>(townZ / static_cast<double>(townCount));
        VOX_LOGI("newvegas") << "spawning at the centroid of " << townCount
                             << " Goodsprings placements";
    } else {
        VOX_LOGW("newvegas") << "no Goodsprings placements in this scene; spawning at scene centre";
    }
    m_cameraX = spawnX;
    m_cameraZ = spawnZ;
    float groundHeight = 0.0f;
    if (groundHeightAt(m_cameraX, m_cameraZ, groundHeight)) {
        m_cameraY = groundHeight + kEyeHeightUnits;
    } else {
        m_cameraY = scene.boundsMax[1] + kEyeHeightUnits;
        VOX_LOGW("newvegas") << "spawn point is off the terrain grid; starting in fly mode";
        m_walkMode = false;
    }
    // Level, so the horizon -- and therefore the sky -- is on screen. The
    // override has to come after this, not before: the spawn pitch is assigned
    // here, so an earlier override would be silently discarded.
    m_pitchDegrees = 0.0f;
    if (const char* pitchEnv = std::getenv("ODAI_FNV_PITCH")) {
        m_pitchDegrees = static_cast<float>(std::atof(pitchEnv));
    }


    // An arrival transform from a door wins over the spawn heuristics above.
    if (arrivalPosition != nullptr) {
        m_cameraX = arrivalPosition[0];
        m_cameraZ = arrivalPosition[2];
        float groundHeight = 0.0f;
        // Fallout's arrival Y is the floor the player stands on, so lift it to
        // eye height. Prefer the terrain lattice where there is one -- an
        // interior has no LAND, and there the authored height is all we have.
        if (groundHeightAt(m_cameraX, m_cameraZ, groundHeight)) {
            m_cameraY = groundHeight + kEyeHeightUnits;
        } else {
            m_cameraY = arrivalPosition[1] + kEyeHeightUnits;
            m_walkMode = false;
        }
        if (arrivalYawDegrees != nullptr) {
            m_yawDegrees = *arrivalYawDegrees;
        }
        m_pitchDegrees = 0.0f;
    }
    return true;
}

int BethesdaApp::findUsableDoor() const {
    // Near, and roughly in front. Both matter: a doorway you have walked past
    // should not keep offering itself, and Fallout's doors come in pairs close
    // enough that distance alone picks the wrong one.
    constexpr float kMaxDoorDistance = 260.0f;   // ~3.7 m at Bethesda scale
    constexpr float kMinFacingDot = 0.35f;
    const float yawRadians = m_yawDegrees * (kPi / 180.0f);
    const float forwardX = std::cos(yawRadians);
    const float forwardZ = std::sin(yawRadians);
    const odai::math::Vector3 query = thirdPersonPlayerShowcase()
        ? bethesdaPlayerEyePosition()
        : odai::math::Vector3{m_cameraX, m_cameraY, m_cameraZ};
    int best = -1;
    float bestDistanceSquared = kMaxDoorDistance * kMaxDoorDistance;
    for (std::size_t i = 0; i < m_doors.size(); ++i) {
        const float dx = m_doors[i].position[0] - query.x;
        const float dz = m_doors[i].position[2] - query.z;
        const float dy = m_doors[i].position[1] - query.y;
        const float distanceSquared = (dx * dx) + (dy * dy) + (dz * dz);
        if (distanceSquared > bestDistanceSquared) {
            continue;
        }
        const float horizontal = std::sqrt((dx * dx) + (dz * dz));
        if (horizontal > 1e-3f && (((dx / horizontal) * forwardX) + ((dz / horizontal) * forwardZ)) < kMinFacingDot) {
            continue;
        }
        best = static_cast<int>(i);
        bestDistanceSquared = distanceSquared;
    }
    return best;
}

int BethesdaApp::findLootableActorInReach() const {
    const odai::math::Vector3 query = thirdPersonPlayerShowcase()
        ? bethesdaPlayerEyePosition()
        : odai::math::Vector3{m_cameraX, m_cameraY, m_cameraZ};
    const float cameraPosition[3] = {query.x, query.y, query.z};
    const float yaw = m_yawDegrees * (kPi / 180.0f);
    int best = -1;
    float bestDistanceSquared = std::numeric_limits<float>::max();
    for (std::size_t index = 0u; index < m_actors.size(); ++index) {
        const SkinnedActor& actor = m_actors[index];
        if (!actor.runtimeDead ||
            !actorIsInReach(actor, cameraPosition, yaw)) {
            continue;
        }
        const float dx = actor.position[0] - query.x;
        const float dy = actor.position[1] - query.y;
        const float dz = actor.position[2] - query.z;
        const float distanceSquared = (dx * dx) + (dy * dy) + (dz * dz);
        if (distanceSquared < bestDistanceSquared) {
            best = static_cast<int>(index);
            bestDistanceSquared = distanceSquared;
        }
    }
    return best;
}

bool BethesdaApp::lootActor(int actorIndex) {
    if (!m_bethesdaSessionConfigured || actorIndex < 0 ||
        actorIndex >= static_cast<int>(m_actors.size())) {
        return false;
    }
    const SkinnedActor& actor = m_actors[static_cast<std::size_t>(actorIndex)];
    bethesda::RecordKey actorReference;
    std::string error;
    if (actor.referenceFormId == 0u ||
        !bethesda::stableRecordKey(
            m_streamLoadOrder, actor.referenceFormId, actorReference, error)) {
        m_toasts.push("Cannot search", error, "loot");
        return false;
    }
    const bethesda::ScenarioDefinition* scenario =
        bethesda::findScenario(m_scenarioId);
    if (scenario == nullptr) return false;
    const bethesda::LootTransferResult result = m_bethesdaSession.lootObject(
        bethesda::ObjectId::persistent(
            bethesda::makeRecordKey(scenario->basePlugin, 0x14u)),
        bethesda::ObjectId::persistent(std::move(actorReference)));
    if (!result.accepted) {
        m_toasts.push("Cannot search", result.diagnostic, "loot");
        return false;
    }
    if (result.transferred.empty()) {
        m_toasts.push(actor.displayName(), result.diagnostic, "loot");
        return true;
    }
    bethesda::RecordKey goldenClawItem;
    if (const bethesda::QuestRuntimeState* ms13 =
            m_bethesdaSession.findQuest("MS13")) {
        const auto clawAlias = std::find_if(
            ms13->aliases.begin(), ms13->aliases.end(),
            [](const bethesda::QuestAliasRuntimeState& alias) {
                return toLowerAscii(alias.name) == "goldenclaw";
            });
        if (clawAlias != ms13->aliases.end()) {
            goldenClawItem = clawAlias->createdObject;
        }
    }
    for (const bethesda::InventoryEntry& entry : result.transferred) {
        const bool goldenClaw = goldenClawItem.valid() && entry.item == goldenClawItem;
        const std::string itemName = goldenClaw ? "Golden Claw" : entry.item.toString();
        m_toasts.push("Item added", itemName +
            (entry.count == 1 ? std::string{} : " x" + std::to_string(entry.count)),
            "loot:" + entry.item.toString());
    }
    return true;
}

bool BethesdaApp::configureGoldenClawPuzzleForCurrentSpace(std::string& outError) {
    const bool replacingBinding = m_goldenClawPuzzle.has_value();
    if (m_goldenClawPuzzle.has_value()) {
        for (const std::uint32_t reference :
             m_goldenClawPuzzle->collisionReferenceFormIds) {
            m_disabledBethesdaCollisionReferences.erase(reference);
        }
        registerCachedBethesdaCollision();
    }
    m_goldenClawPuzzle.reset();
    if (!m_bethesdaSessionConfigured || m_streamer == nullptr ||
        !m_interiorStarted || m_currentInteriorEditorId.empty()) {
        outError.clear();
        return true;
    }
    const bethesda::QuestRuntimeState* questState =
        m_bethesdaSession.findQuest("MS13");
    if (questState == nullptr) {
        outError = "Golden Claw runtime requires the installed MS13 quest";
        return false;
    }
    const auto alias = std::find_if(
        questState->aliases.begin(), questState->aliases.end(),
        [](const bethesda::QuestAliasRuntimeState& candidate) {
            return toLowerAscii(candidate.name) == "hallofstoriesdoor";
        });
    if (alias == questState->aliases.end() ||
        alias->target.kind != bethesda::ObjectIdKind::PersistentReference) {
        outError = "MS13 HallofStoriesDoor alias has no persistent target";
        return false;
    }

    std::uint32_t keyholeReferenceFormId = 0u;
    if (!bethesda::resolvedFormId(
            m_streamLoadOrder, alias->target.reference,
            keyholeReferenceFormId, outError)) {
        return false;
    }
    if (!m_streamer->referenceBelongsToInterior(
            keyholeReferenceFormId, m_currentInteriorEditorId)) {
        // MS13 is globally registered, but its keyhole matters only while its
        // owning interior is resident.
        outError.clear();
        return true;
    }

    std::uint32_t keyholeBaseFormId = 0u;
    std::size_t keyholeSourcePluginIndex = 0u;
    std::vector<std::uint8_t> keyholeVmadBytes;
    if (!m_streamer->referenceGameplayData(
            keyholeReferenceFormId, keyholeBaseFormId, keyholeVmadBytes,
            keyholeSourcePluginIndex, outError)) {
        return false;
    }
    bethesda::VmadAttachments keyholeAttachments;
    if (!bethesda::readVmadAttachments(
            keyholeVmadBytes, keyholeAttachments, outError)) {
        outError = "Hall of Stories keyhole VMAD: " + outError;
        return false;
    }
    const bethesda::VmadScriptAttachment* keyholeScript =
        findVmadScript(keyholeAttachments, "HallofStoriesKeyholeScript");
    if (keyholeScript == nullptr) {
        outError = "Hall of Stories keyhole lacks HallofStoriesKeyholeScript";
        return false;
    }

    const auto remapKeyholeProperty = [&](std::uint32_t raw) {
        return m_streamLoadOrder.remapFormId(keyholeSourcePluginIndex, raw);
    };
    std::uint32_t requiredItemRaw = 0u;
    std::uint32_t questRaw = 0u;
    std::uint32_t doorBaseRaw = 0u;
    std::int32_t successStage = 0;
    if (!readVmadObjectProperty(*keyholeScript, "myMiscObject", requiredItemRaw) ||
        !readVmadObjectProperty(*keyholeScript, "myQuest", questRaw) ||
        !readVmadObjectProperty(*keyholeScript, "doorBase", doorBaseRaw) ||
        !readVmadIntegerProperty(
            *keyholeScript, "myQuestStageSuccess", successStage) ||
        successStage < 0) {
        outError = "HallofStoriesKeyholeScript has incomplete item, quest, door, or stage properties";
        return false;
    }

    static constexpr std::array<std::string_view, 3> kRingProperties = {
        "largeRing", "mediumRing", "smallRing"};
    std::array<std::uint32_t, 3> ringReferenceFormIds{};
    bethesda::RuntimeActivatorState activator;
    activator.puzzleStates.reserve(kRingProperties.size());
    activator.puzzleSolution.reserve(kRingProperties.size());
    for (std::size_t ring = 0u; ring < kRingProperties.size(); ++ring) {
        std::uint32_t ringRaw = 0u;
        if (!readVmadObjectProperty(*keyholeScript, kRingProperties[ring], ringRaw)) {
            outError = "HallofStoriesKeyholeScript is missing " +
                std::string(kRingProperties[ring]);
            return false;
        }
        ringReferenceFormIds[ring] = remapKeyholeProperty(ringRaw);
        std::uint32_t unusedRingBase = 0u;
        std::size_t ringSourcePluginIndex = 0u;
        std::vector<std::uint8_t> ringVmadBytes;
        if (!m_streamer->referenceGameplayData(
                ringReferenceFormIds[ring], unusedRingBase, ringVmadBytes,
                ringSourcePluginIndex, outError)) {
            outError = std::string(kRingProperties[ring]) + ": " + outError;
            return false;
        }
        bethesda::VmadAttachments ringAttachments;
        if (!bethesda::readVmadAttachments(
                ringVmadBytes, ringAttachments, outError)) {
            outError = std::string(kRingProperties[ring]) + " VMAD: " + outError;
            return false;
        }
        const bethesda::VmadScriptAttachment* ringScript =
            findVmadScript(ringAttachments, "HallofStoriesDiskScript");
        std::int32_t initialState = 0;
        std::int32_t solveState = 0;
        if (ringScript == nullptr ||
            !readVmadIntegerProperty(*ringScript, "initialState", initialState) ||
            !readVmadIntegerProperty(*ringScript, "solveState", solveState) ||
            initialState <= 0 || solveState <= 0) {
            outError = std::string(kRingProperties[ring]) +
                " lacks valid HallofStoriesDiskScript states";
            return false;
        }
        (void)ringSourcePluginIndex;
        activator.puzzleStates.push_back(initialState);
        activator.puzzleSolution.push_back(solveState);
        activator.puzzleStateCount = std::max(
            activator.puzzleStateCount, std::max(initialState, solveState));
    }

    bethesda::RecordKey keyholeBase;
    bethesda::RecordKey requiredItem;
    bethesda::RecordKey quest;
    if (!bethesda::stableRecordKey(
            m_streamLoadOrder, keyholeBaseFormId, keyholeBase, outError) ||
        !bethesda::stableRecordKey(
            m_streamLoadOrder, remapKeyholeProperty(requiredItemRaw),
            requiredItem, outError) ||
        !bethesda::stableRecordKey(
            m_streamLoadOrder, remapKeyholeProperty(questRaw), quest, outError)) {
        return false;
    }
    float keyholePosition[3] = {};
    if (!m_streamer->referencePositionEngineSpace(
            keyholeReferenceFormId, keyholePosition, outError)) {
        return false;
    }

    const bethesda::ObjectId keyholeObject = alias->target;
    const bethesda::RuntimeObject* existing =
        m_bethesdaSession.world().find(keyholeObject);
    if (existing == nullptr) {
        bethesda::RuntimeObject runtime;
        runtime.id = keyholeObject;
        runtime.base = std::move(keyholeBase);
        runtime.kind = bethesda::RuntimeObjectKind::Activator;
        runtime.persistent = true;
        runtime.interior = true;
        runtime.transform.position = {
            keyholePosition[0], keyholePosition[1], keyholePosition[2]};
        runtime.activatorState = activator;
        if (!m_bethesdaSession.world().addInitialObject(
                std::move(runtime), outError)) {
            return false;
        }
        existing = m_bethesdaSession.world().find(keyholeObject);
    }
    if (existing == nullptr ||
        existing->kind != bethesda::RuntimeObjectKind::Activator ||
        !existing->activatorState.has_value() ||
        existing->activatorState->puzzleStates.size() != kRingProperties.size() ||
        existing->activatorState->puzzleSolution.size() != kRingProperties.size()) {
        outError = "Hall of Stories keyhole save state is incompatible with installed VMAD";
        return false;
    }

    GoldenClawPuzzleBinding binding;
    binding.door = keyholeObject;
    binding.requiredItem = std::move(requiredItem);
    binding.quest = std::move(quest);
    binding.successStage = successStage;
    binding.keyholeReferenceFormId = keyholeReferenceFormId;
    std::copy_n(keyholePosition, 3u, binding.position);
    binding.collisionReferenceFormIds = {
        keyholeReferenceFormId, remapKeyholeProperty(doorBaseRaw),
        ringReferenceFormIds[0], ringReferenceFormIds[1], ringReferenceFormIds[2]};
    std::uint32_t doorFxRaw = 0u;
    if (readVmadObjectProperty(*keyholeScript, "doorFX", doorFxRaw)) {
        binding.collisionReferenceFormIds.push_back(remapKeyholeProperty(doorFxRaw));
    }
    std::sort(binding.collisionReferenceFormIds.begin(),
        binding.collisionReferenceFormIds.end());
    binding.collisionReferenceFormIds.erase(
        std::unique(binding.collisionReferenceFormIds.begin(),
            binding.collisionReferenceFormIds.end()),
        binding.collisionReferenceFormIds.end());
    m_goldenClawPuzzle = std::move(binding);

    if (existing->activatorState->opened) {
        m_disabledBethesdaCollisionReferences.insert(
            m_goldenClawPuzzle->collisionReferenceFormIds.begin(),
            m_goldenClawPuzzle->collisionReferenceFormIds.end());
    }
    registerCachedBethesdaCollision();
    if ((existing->activatorState->opened || replacingBinding) &&
        !refreshGoldenClawPresentation(outError)) {
        return false;
    }
    VOX_LOGI("scenario")
        << "configured installed Golden Claw keyhole "
        << m_goldenClawPuzzle->door.toString() << " states="
        << existing->activatorState->puzzleStates[0] << ","
        << existing->activatorState->puzzleStates[1] << ","
        << existing->activatorState->puzzleStates[2] << " solution="
        << existing->activatorState->puzzleSolution[0] << ","
        << existing->activatorState->puzzleSolution[1] << ","
        << existing->activatorState->puzzleSolution[2];
    outError.clear();
    return true;
}

bool BethesdaApp::goldenClawPuzzleInReach() const {
    if (!m_goldenClawPuzzle.has_value()) return false;
    const bethesda::RuntimeObject* keyhole =
        m_bethesdaSession.world().find(m_goldenClawPuzzle->door);
    if (keyhole == nullptr || !keyhole->activatorState.has_value() ||
        keyhole->activatorState->opened) {
        return false;
    }
    constexpr float kMaxDistance = 300.0f;
    constexpr float kMinFacingDot = 0.25f;
    const float dx = m_goldenClawPuzzle->position[0] - m_cameraX;
    const float dy = m_goldenClawPuzzle->position[1] - m_cameraY;
    const float dz = m_goldenClawPuzzle->position[2] - m_cameraZ;
    if ((dx * dx) + (dy * dy) + (dz * dz) > kMaxDistance * kMaxDistance) {
        return false;
    }
    const float horizontal = std::sqrt((dx * dx) + (dz * dz));
    if (horizontal <= 1e-3f) return true;
    const float yaw = m_yawDegrees * (kPi / 180.0f);
    return ((dx / horizontal) * std::cos(yaw)) +
        ((dz / horizontal) * std::sin(yaw)) >= kMinFacingDot;
}

bool BethesdaApp::rotateGoldenClawRing(std::size_t ringIndex) {
    if (!m_goldenClawPuzzle.has_value()) return false;
    std::string error;
    if (!m_bethesdaSession.rotatePuzzleRing(
            m_goldenClawPuzzle->door, ringIndex, error)) {
        m_toasts.push("Claw mechanism", error, "golden-claw");
        return false;
    }
    static constexpr std::array<const char*, 3> kRingNames = {
        "Large ring", "Medium ring", "Small ring"};
    m_toasts.push(kRingNames[ringIndex], "Rotated", "golden-claw");
    return true;
}

bool BethesdaApp::useGoldenClawPuzzle() {
    if (!m_goldenClawPuzzle.has_value()) return false;
    const bethesda::ScenarioDefinition* scenario =
        bethesda::findScenario(m_scenarioId);
    if (scenario == nullptr) return false;
    const bethesda::PuzzleDoorActivationResult result =
        m_bethesdaSession.activatePuzzleDoor(
            bethesda::ObjectId::persistent(
                bethesda::makeRecordKey(scenario->basePlugin, 0x14u)),
            m_goldenClawPuzzle->door,
            m_goldenClawPuzzle->requiredItem,
            m_goldenClawPuzzle->quest,
            m_goldenClawPuzzle->successStage);
    if (!result.accepted) {
        m_toasts.push("Claw mechanism", result.diagnostic, "golden-claw");
    } else if (result.missingRequiredItem) {
        m_toasts.push("Claw mechanism", "The Golden Claw is required", "golden-claw");
    } else if (result.incorrectCombination) {
        m_toasts.push("Claw mechanism", "The rings are not aligned", "golden-claw");
    } else if (result.opened) {
        m_disabledBethesdaCollisionReferences.insert(
            m_goldenClawPuzzle->collisionReferenceFormIds.begin(),
            m_goldenClawPuzzle->collisionReferenceFormIds.end());
        registerCachedBethesdaCollision();
        std::string presentationError;
        if (!refreshGoldenClawPresentation(presentationError)) {
            VOX_LOGE("render") << presentationError;
            m_toasts.push("Compatibility error", presentationError,
                "golden-claw-presentation");
        }
        m_toasts.push("Quest updated", "The Hall of Stories is open", "golden-claw");
    }
    return result.accepted;
}

bool BethesdaApp::refreshGoldenClawPresentation(std::string& outError) {
    if (!m_currentInteriorSourceScene.has_value() ||
        m_interiorChunk == render::Renderer::kInvalidImportedChunkIndex) {
        outError = "Golden Claw presentation has no resident interior source scene";
        return false;
    }
    importer::ImportedScene presentation = *m_currentInteriorSourceScene;
    std::unordered_set<std::string> hiddenSources;
    if (m_goldenClawPuzzle.has_value()) {
        if (m_disabledBethesdaCollisionReferences.contains(
                m_goldenClawPuzzle->keyholeReferenceFormId)) {
            for (const std::uint32_t reference :
                 m_goldenClawPuzzle->collisionReferenceFormIds) {
                hiddenSources.insert(importedReferenceSourceId(reference));
            }
        }
    }
    std::erase_if(presentation.instances,
        [&](const importer::ImportedSceneInstance& instance) {
            return hiddenSources.contains(instance.sourceId);
        });
    std::erase_if(presentation.particleEmitters,
        [&](const importer::ImportedSceneParticleEmitter& emitter) {
            return hiddenSources.contains(emitter.sourceId);
        });
    std::erase_if(presentation.lights,
        [&](const importer::ImportedSceneLight& light) {
            return std::any_of(hiddenSources.begin(), hiddenSources.end(),
                [&](const std::string& source) {
                    return light.sourceId == source ||
                        light.sourceId.starts_with(source + "_");
                });
        });
    importer::buildImportedScenePackedRenderData(presentation);
    const std::size_t replacement = m_renderer.addImportedSceneChunk(presentation);
    if (replacement == render::Renderer::kInvalidImportedChunkIndex) {
        outError = "renderer rejected Golden Claw presentation refresh";
        return false;
    }
    m_renderer.removeImportedSceneChunk(m_interiorChunk);
    m_interiorChunk = replacement;
    outError.clear();
    return true;
}

void BethesdaApp::beginConversation(int actorIndex) {
    if (actorIndex < 0 || actorIndex >= static_cast<int>(m_actors.size())) {
        return;
    }
    if (beginTes3Conversation(actorIndex) || beginBethesdaConversation(actorIndex)) {
        return;
    }
    // A Skyrim scenario must never fall through to the TES4/Fallout dialogue
    // importer. That format happens to produce a few structurally valid but
    // empty nodes from TES5 records, yielding a blank modal and no authored
    // effects. No matching retail INFO means the actor simply has no available
    // topic in the current quest state.
    if (m_bethesdaSessionConfigured && m_streamIsSkyrim) {
        return;
    }
    SkinnedActor& actor = m_actors[static_cast<std::size_t>(actorIndex)];
    if (!actor.canTalk()) {
        return;
    }
    m_bethesdaDialogueActive = false;
    m_bethesdaDialogueChoices.clear();
    m_bethesdaDialogueSpeaker = {};
    m_bethesdaDialoguePlayer = {};
    m_bethesdaDialoguePendingEndInfo = {};
    m_bethesdaDialogueNextTopics.clear();
    m_talkingActor = actorIndex;
    actor.talking = true;
    actor.runtime.begin(actor.tree, actor.context);
    // Reset the highlight rather than inheriting the last conversation's row,
    // which would open on a reply that belongs to somebody else's branch.
    m_dialogueChoice = 0;
    m_dialogueChoiceNodeId.clear();
    VOX_LOGI("newvegas") << "conversation: " << actor.name << " node="
                         << (actor.runtime.currentNode() != nullptr
                                 ? actor.runtime.currentNode()->id
                                 : std::string("<null>"))
                         << " finished=" << actor.runtime.isFinished();
}

bool BethesdaApp::beginTes3Conversation(int actorIndex) {
    if (!m_bethesdaSessionConfigured || !m_streamIsMorrowind ||
        actorIndex < 0 || actorIndex >= static_cast<int>(m_actors.size())) {
        return false;
    }
    SkinnedActor& actor = m_actors[static_cast<std::size_t>(actorIndex)];
    if (!actor.placed || actor.runtimeDead) return false;
    const auto& content = m_bethesdaSession.tes3().content();
    if (content == nullptr) return false;
    const bethesda::Tes3ActorDefinition* definition =
        content->findActor("NPC_", actor.name);
    if (definition == nullptr) definition = content->findActor("CREA", actor.name);
    if (definition == nullptr) return false;

    bethesda::Tes3DialogueActorState state;
    state.object = bethesda::ObjectId::persistent(definition->record);
    state.id = definition->id;
    state.race = definition->race;
    state.actorClass = definition->actorClass;
    state.faction = definition->faction.textId;
    state.rank = static_cast<std::int8_t>(std::clamp(definition->rank, -1, 127));

    // Prefer the placed FRMR identity when the presentation actor corresponds
    // to one. Result scripts then mutate the actual streamed object instead of
    // an actor-base surrogate. Synthetic diagnostic actors deliberately fall
    // back to the named base object.
    float nearestDistanceSquared = std::numeric_limits<float>::max();
    const bethesda::Tes3ReferenceDefinition* nearestReference = nullptr;
    for (const auto& [id, reference] : content->references()) {
        (void)id;
        if (reference.base != definition->record || !reference.hasTransform) continue;
        const float dx = actor.position[0] - reference.position[0];
        const float dy = actor.position[1] - reference.position[2];
        const float dz = actor.position[2] + reference.position[1];
        const float distanceSquared = (dx * dx) + (dy * dy) + (dz * dz);
        if (distanceSquared < nearestDistanceSquared) {
            nearestDistanceSquared = distanceSquared;
            nearestReference = &reference;
        }
    }
    if (nearestReference != nullptr && nearestDistanceSquared < 1600.0f * 1600.0f) {
        state.object = nearestReference->id;
        state.cell = nearestReference->cell.textId;
    }
    if (m_interiorStarted && !m_currentInteriorEditorId.empty()) {
        state.cell = m_currentInteriorEditorId;
    } else if (!m_streamSpawnInterior.empty()) {
        // A doorstep spawn names the authored interior ("Almas Thirr,
        // Temple"). TES3 exterior dialogue filters name the settlement prefix.
        const std::size_t comma = m_streamSpawnInterior.find(',');
        state.cell = m_streamSpawnInterior.substr(0u, comma);
    }

    if (m_bethesdaSession.world().find(state.object) == nullptr) {
        bethesda::RuntimeObject runtimeActor;
        runtimeActor.id = state.object;
        runtimeActor.base = definition->record;
        runtimeActor.kind = bethesda::RuntimeObjectKind::Actor;
        runtimeActor.persistent = true;
        runtimeActor.transform.position = {
            actor.position[0], actor.position[1], actor.position[2]};
        runtimeActor.actorValues.emplace();
        std::string addError;
        if (!m_bethesdaSession.world().addInitialObject(std::move(runtimeActor), addError)) {
            VOX_LOGW("tes3") << "could not bind dialogue actor " << actor.name
                              << ": " << addError;
        }
    }

    bethesda::Tes3DialoguePlayerState player =
        m_bethesdaSession.tes3().playerState();
    player.object = m_bethesdaSession.playerObject();
    const bethesda::Tes3DialogueResponse response =
        m_bethesdaSession.startTes3Dialogue(std::move(state), std::move(player), false);
    for (const std::string& diagnostic : response.diagnostics) {
        VOX_LOGW("tes3-dialogue") << diagnostic;
    }
    if (!response.accepted) return false;

    m_tes3DialogueActive = true;
    m_bethesdaDialogueActive = false;
    m_talkingActor = actorIndex;
    actor.talking = true;
    rebuildTes3ConversationTree(actor, response);
    VOX_LOGI("tes3-dialogue") << "conversation: " << actor.name
                               << " cell=\"" <<
        m_bethesdaSession.tes3().dialogue().actor.cell << "\" topics="
                               << m_bethesdaSession.tes3DialogueTopics(false).size()
                               << " faceHeight=" << conversationFaceHeight(actor)
                               << " bindFaceHeight=" << actor.headHeightUnits
                               << " standingHeight=" << actor.standingHeightUnits;
    return true;
}

void BethesdaApp::rebuildTes3ConversationTree(
    SkinnedActor& actor, const bethesda::Tes3DialogueResponse& response) {
    actor.tree = {};
    actor.tree.id = "tes3-dynamic-dialogue";
    actor.tree.startNode = "tes3-response";
    dialogue::DialogueNode node;
    node.id = actor.tree.startNode;
    node.speaker = actor.displayName();
    node.text = response.text.empty() ? "..." : response.text;
    m_tes3DialogueActions.clear();
    for (const bethesda::Tes3DialogueChoice& choice : response.choices) {
        node.choices.push_back({choice.label, node.id, {}, {}});
        m_tes3DialogueActions.push_back(
            {Tes3DialogueActionKind::Choice, {}, choice.value});
    }
    for (const std::string& topic : m_bethesdaSession.tes3DialogueTopics(false)) {
        if (m_tes3DialogueActions.size() >= 8u) break;
        node.choices.push_back({topic, node.id, {}, {}});
        m_tes3DialogueActions.push_back(
            {Tes3DialogueActionKind::Topic, topic, 0});
    }
    node.choices.push_back({"Goodbye", node.id, {}, {}});
    m_tes3DialogueActions.push_back(
        {Tes3DialogueActionKind::Goodbye, {}, 0});
    actor.tree.nodes.emplace(node.id, std::move(node));
    actor.runtime.begin(actor.tree, actor.context);
    m_dialogueChoice = 0;
    m_dialogueChoiceNodeId.clear();
}

bool BethesdaApp::beginBethesdaConversation(int actorIndex) {
    if (!m_bethesdaSessionConfigured || !m_streamIsSkyrim ||
        actorIndex < 0 || actorIndex >= static_cast<int>(m_actors.size())) {
        return false;
    }
    SkinnedActor& actor = m_actors[static_cast<std::size_t>(actorIndex)];
    if (!actor.placed || actor.runtimeDead || actor.referenceFormId == 0u) return false;
    const bethesda::ScenarioDefinition* scenario = bethesda::findScenario(m_scenarioId);
    if (scenario == nullptr) return false;
    bethesda::RecordKey speakerRecord;
    std::string error;
    if (!bethesda::stableRecordKey(
            m_streamLoadOrder, actor.referenceFormId, speakerRecord, error)) {
        VOX_LOGW("dialogue") << "could not resolve Skyrim speaker " << actor.name
                              << ": " << error;
        return false;
    }
    const bethesda::ObjectId speaker =
        bethesda::ObjectId::persistent(std::move(speakerRecord));
    const bethesda::ObjectId player = bethesda::ObjectId::persistent(
        bethesda::makeRecordKey(scenario->basePlugin, 0x14u));
    std::vector<bethesda::SkyrimDialogueChoice> choices =
        m_bethesdaSession.availableDialogueChoices(speaker, player, true);
    if (choices.empty()) return false;

    m_bethesdaDialogueActive = true;
    m_bethesdaDialogueSpeaker = speaker;
    m_bethesdaDialoguePlayer = player;
    m_bethesdaDialoguePendingEndInfo = {};
    m_bethesdaDialogueNextTopics.clear();
    rebuildBethesdaConversationTree(actor, std::move(choices));
    m_talkingActor = actorIndex;
    actor.talking = true;
    m_dialogueChoice = 0;
    m_dialogueChoiceNodeId.clear();
    VOX_LOGI("dialogue") << "Skyrim conversation: " << actor.name << " topics="
                          << m_bethesdaDialogueChoices.size();
    return true;
}

void BethesdaApp::rebuildBethesdaConversationTree(
    SkinnedActor& actor, std::vector<bethesda::SkyrimDialogueChoice> choices) {
    // The immediate-mode conversation panel supplies selection, scrolling,
    // controller input and camera framing. This tree is presentation-only;
    // branch eligibility, conditions and effects remain in BethesdaSession.
    actor.tree = {};
    actor.tree.id = "skyrim-retail-dialogue";
    actor.tree.startNode = "topics";
    dialogue::DialogueNode topics;
    topics.id = "topics";
    topics.speaker = actor.displayName();
    topics.text = " ";
    topics.choices.reserve(choices.size());
    for (std::size_t index = 0u; index < choices.size(); ++index) {
        const std::string responseId = "response_" + std::to_string(index);
        topics.choices.push_back(dialogue::DialogueChoice{
            choices[index].prompt, responseId, {}, {}});
        dialogue::DialogueNode response;
        response.id = responseId;
        response.speaker = actor.displayName();
        response.text = "[Waiting for authored response]";
        actor.tree.nodes.emplace(responseId, std::move(response));
    }
    actor.tree.nodes.emplace(topics.id, std::move(topics));
    m_bethesdaDialogueChoices = std::move(choices);
    actor.runtime.begin(actor.tree, actor.context);
    m_dialogueChoice = 0;
    m_dialogueChoiceNodeId.clear();
}

int BethesdaApp::findBethesdaDialogueActorInReach(
    const float cameraPosition[3], float cameraYawRadians) {
    if (!m_bethesdaSessionConfigured || !m_streamIsSkyrim) return -1;
    int best = -1;
    float bestDistanceSquared = 0.0f;
    for (std::size_t index = 0u; index < m_actors.size(); ++index) {
        const SkinnedActor& actor = m_actors[index];
        if (actor.runtimeDead || actor.referenceFormId == 0u ||
            !actorIsInReach(actor, cameraPosition, cameraYawRadians)) continue;
        bethesda::RecordKey speakerRecord;
        std::string error;
        if (!bethesda::stableRecordKey(
                m_streamLoadOrder, actor.referenceFormId, speakerRecord, error)) continue;
        const bethesda::ScenarioDefinition* scenario = bethesda::findScenario(m_scenarioId);
        if (scenario == nullptr) return -1;
        const bethesda::ObjectId speaker =
            bethesda::ObjectId::persistent(std::move(speakerRecord));
        const bethesda::ObjectId player = bethesda::ObjectId::persistent(
            bethesda::makeRecordKey(scenario->basePlugin, 0x14u));
        if (m_bethesdaSession.availableDialogueChoices(speaker, player, true).empty()) continue;
        const float dx = actor.position[0] - cameraPosition[0];
        const float dy = actor.position[1] - cameraPosition[1];
        const float dz = actor.position[2] - cameraPosition[2];
        const float distanceSquared = (dx * dx) + (dy * dy) + (dz * dz);
        if (best < 0 || distanceSquared < bestDistanceSquared) {
            best = static_cast<int>(index);
            bestDistanceSquared = distanceSquared;
        }
    }
    return best;
}

int BethesdaApp::findTes3DialogueActorInReach(
    const float cameraPosition[3], float cameraYawRadians) const {
    if (!m_bethesdaSessionConfigured || !m_streamIsMorrowind ||
        m_bethesdaSession.tes3().content() == nullptr) return -1;
    int best = -1;
    float bestDistanceSquared = std::numeric_limits<float>::max();
    for (std::size_t index = 0u; index < m_actors.size(); ++index) {
        const SkinnedActor& actor = m_actors[index];
        if (!actor.placed || actor.runtimeDead ||
            !actorIsInReach(actor, cameraPosition, cameraYawRadians)) continue;
        const auto& content = m_bethesdaSession.tes3().content();
        if (content->findActor("NPC_", actor.name) == nullptr &&
            content->findActor("CREA", actor.name) == nullptr) continue;
        const float dx = actor.position[0] - cameraPosition[0];
        const float dy = actor.position[1] - cameraPosition[1];
        const float dz = actor.position[2] - cameraPosition[2];
        const float distanceSquared = (dx * dx) + (dy * dy) + (dz * dz);
        if (distanceSquared < bestDistanceSquared) {
            best = static_cast<int>(index);
            bestDistanceSquared = distanceSquared;
        }
    }
    return best;
}

void BethesdaApp::chooseConversationChoice(std::size_t index) {
    SkinnedActor* actor = talkingActor();
    if (actor == nullptr) return;
    const auto visibleChoices = actor->runtime.availableChoices();
    if (index >= visibleChoices.size()) return;
    if (m_tes3DialogueActive) {
        if (index >= m_tes3DialogueActions.size()) return;
        const Tes3DialogueAction action = m_tes3DialogueActions[index];
        if (action.kind == Tes3DialogueActionKind::Goodbye) {
            endConversation();
            return;
        }
        const bethesda::Tes3DialogueResponse response =
            action.kind == Tes3DialogueActionKind::Choice
                ? m_bethesdaSession.answerTes3Choice(action.choice, false)
                : m_bethesdaSession.selectTes3Topic(action.topic, false);
        for (const std::string& diagnostic : response.diagnostics) {
            VOX_LOGW("tes3-dialogue") << diagnostic;
        }
        if (!response.accepted || response.goodbye) {
            endConversation();
            return;
        }
        rebuildTes3ConversationTree(*actor, response);
        syncTes3JournalPanel();
        return;
    }
    if (!m_bethesdaDialogueActive) {
        actor->runtime.choose(*visibleChoices[index]);
        return;
    }
    if (m_bethesdaDialoguePendingEndInfo.valid()) {
        const bethesda::RecordKey completedInfo = m_bethesdaDialoguePendingEndInfo;
        m_bethesdaDialoguePendingEndInfo = {};
        bethesda::SkyrimDialogueSelectionResult end =
            m_bethesdaSession.selectDialogueInfo(completedInfo,
                m_bethesdaDialogueSpeaker, m_bethesdaDialoguePlayer, 2u, true);
        for (const std::string& diagnostic : end.diagnostics) {
            VOX_LOGW("dialogue") << completedInfo.toString() << ": " << diagnostic;
        }
        std::vector<bethesda::SkyrimDialogueChoice> choices =
            m_bethesdaSession.availableDialogueChoices(
                m_bethesdaDialogueSpeaker, m_bethesdaDialoguePlayer, true,
                m_bethesdaDialogueNextTopics);
        m_bethesdaDialogueNextTopics.clear();
        if (choices.empty()) {
            choices = m_bethesdaSession.availableDialogueChoices(
                m_bethesdaDialogueSpeaker, m_bethesdaDialoguePlayer, true);
        }
        if (choices.empty()) {
            endConversation();
            return;
        }
        rebuildBethesdaConversationTree(*actor, std::move(choices));
        return;
    }
    if (index >= m_bethesdaDialogueChoices.size()) return;
    const bethesda::SkyrimDialogueChoice& choice = m_bethesdaDialogueChoices[index];
    bethesda::SkyrimDialogueSelectionResult begin = m_bethesdaSession.selectDialogueInfo(
        choice.info, m_bethesdaDialogueSpeaker, m_bethesdaDialoguePlayer, 1u, true);
    for (const std::string& diagnostic : begin.diagnostics) {
        VOX_LOGW("dialogue") << choice.info.toString() << ": " << diagnostic;
    }
    std::vector<std::string> responses = begin.responses;
    std::string text;
    for (const std::string& response : responses) {
        if (!text.empty()) text += "  ";
        text += response;
    }
    if (text.empty()) {
        text = "[Compatibility error: the selected INFO has no localized response]";
    }
    const std::string responseId = "response_" + std::to_string(index);
    if (auto response = actor->tree.nodes.find(responseId);
        response != actor->tree.nodes.end()) {
        response->second.text = std::move(text);
        if (begin.accepted) {
            response->second.choices.push_back(
                dialogue::DialogueChoice{"Continue", "topics", {}, {}});
        }
    }
    if (begin.accepted) {
        m_bethesdaDialoguePendingEndInfo = choice.info;
        m_bethesdaDialogueNextTopics = std::move(begin.nextTopics);
    }
    actor->runtime.choose(*visibleChoices[index]);
}

void BethesdaApp::endConversation() {
    if (m_tes3DialogueActive) {
        m_bethesdaSession.tes3().endDialogue();
    }
    if (m_bethesdaDialogueActive && m_bethesdaDialoguePendingEndInfo.valid()) {
        const bethesda::RecordKey completedInfo = m_bethesdaDialoguePendingEndInfo;
        m_bethesdaDialoguePendingEndInfo = {};
        const bethesda::SkyrimDialogueSelectionResult end =
            m_bethesdaSession.selectDialogueInfo(completedInfo,
                m_bethesdaDialogueSpeaker, m_bethesdaDialoguePlayer, 2u, true);
        for (const std::string& diagnostic : end.diagnostics) {
            VOX_LOGW("dialogue") << completedInfo.toString() << ": " << diagnostic;
        }
    }
    if (SkinnedActor* actor = talkingActor()) {
        actor->talking = false;
        // Forget which line was spoken, so returning to this actor replays the
        // greeting instead of opening on a silent node.
        actor->spokenNodeId.clear();
    }
    if (m_bethesdaDialogueActive && m_bethesdaDialogueSpeaker.valid()) {
        bethesda::WorldCommand context;
        context.type = bethesda::WorldCommandType::SetActorContext;
        context.target = m_bethesdaDialogueSpeaker;
        context.inDialogueWithPlayer = false;
        (void)m_bethesdaSession.world().queue(std::move(context));
    }
    m_talkingActor = -1;
    m_dialogueChoice = 0;
    m_dialogueChoiceNodeId.clear();
    m_bethesdaDialogueActive = false;
    m_tes3DialogueActive = false;
    m_tes3DialogueActions.clear();
    m_bethesdaDialogueSpeaker = {};
    m_bethesdaDialoguePlayer = {};
    m_bethesdaDialogueChoices.clear();
    m_bethesdaDialogueNextTopics.clear();
}

void BethesdaApp::useDoor(const importer::ImportedSceneDoor& door) {
    if (door.targetKind == importer::ImportedSceneDoorTargetKind::Exterior &&
        !m_interiorStarted && m_streamer != nullptr &&
        door.targetWorldspaceEditorId == m_streamer->currentWorldspaceEditorId()) {
        // XTEL between two exterior cells in one worldspace is just a teleport;
        // residency remains continuous and the normal streamer catches up at
        // the new position on this frame.
        m_cameraX = door.arrivalPosition[0];
        m_cameraY = door.arrivalPosition[1] + m_collision.tuning().eyeHeight;
        m_cameraZ = door.arrivalPosition[2];
        m_yawDegrees = door.arrivalYawDegrees;
        m_pitchDegrees = 0.0f;
        if (m_bethesdaPlayerControllerRegistered) {
            relocateBethesdaPlayerControllerToCamera();
        }
        reloadActorsForCurrentSpace();
        return;
    }
    if (door.targetKind != importer::ImportedSceneDoorTargetKind::CookedLegacy) {
        if (m_doorTransitionPhase == DoorTransitionPhase::None) {
            m_pendingDoor = door;
            m_doorTransitionPhase = DoorTransitionPhase::FadeOut;
            m_doorTransitionAlpha = 0.0f;
            if (m_bethesdaPlayerControllerRegistered) {
                (void)m_bethesdaSession.setActorControllerInput(
                    m_bethesdaSession.playerObject(), bethesda::PhysicsCharacterInput{});
                m_bethesdaControllerOwnsCamera = false;
            }
            if (door.locked) {
                m_toasts.push("Lock bypassed", door.targetCellEditorId.empty()
                    ? std::string("Exploration mode") : door.targetCellEditorId);
            }
        }
        return;
    }

    // An empty target cell means the exterior this interior was cooked beside;
    // both spellings go through importedSceneInteriorFileName so the cooker's
    // naming convention lives in exactly one place.
    const std::filesystem::path target = door.targetCellEditorId.empty()
        ? (m_sceneDirectory / (m_exteriorStem + ".bin"))
        : (m_sceneDirectory /
           importer::importedSceneInteriorFileName(m_exteriorStem, door.targetCellEditorId));
    if (!std::filesystem::exists(target)) {
        VOX_LOGW("newvegas") << "door leads to " << target.filename().string()
                             << ", which is not cooked; re-run the cooker with --with-interiors";
        return;
    }
    const float arrivalYaw = door.arrivalYawDegrees;
    if (!loadScene(target, door.arrivalPosition, &arrivalYaw)) {
        VOX_LOGE("newvegas") << "failed to walk through the door into " << target.filename().string();
    }
}

void BethesdaApp::rebuildStreamDoors() {
    m_doors.clear();
    for (const auto& [cell, doors] : m_streamDoorsByCell) {
        (void)cell;
        m_doors.insert(m_doors.end(), doors.begin(), doors.end());
    }
    std::vector<importer::ImportedSceneDoor> residentLinks;
    if (!m_interiorStarted) {
        for (const importer::ImportedSceneDoor& door : m_doors) {
            if (door.targetKind == importer::ImportedSceneDoorTargetKind::Exterior &&
                !door.targetWorldspaceEditorId.empty() &&
                door.targetWorldspaceEditorId == m_streamWorldspace) {
                residentLinks.push_back(door);
            }
        }
    }
    m_actorNavigation.setResidentDoors(residentLinks);
}

bool BethesdaApp::completeDoorTransition(
    const importer::ImportedSceneDoor& door, std::string& outError) {
    outError.clear();
    if (m_streamer == nullptr) {
        outError = "no active streamed session";
        return false;
    }

    if (door.targetKind == importer::ImportedSceneDoorTargetKind::Interior) {
        if (door.targetCellEditorId.empty()) {
            outError = "interior door has no destination cell identity";
            return false;
        }
        // Build first. A bad destination leaves the current world untouched.
        importer::ImportedScene scene;
        importer::fnv::CellStreamer::InteriorScene interior;
        if (!m_streamer->buildInteriorScene(
                door.targetCellEditorId, scene, interior, outError)) {
            return false;
        }

        // Upload while the old space is still intact. A renderer allocation
        // failure must leave the player in that old space, not at a black frame
        // with its residency already torn down.
        const std::size_t destinationChunk = m_renderer.addImportedSceneChunk(scene);
        if (destinationChunk == render::Renderer::kInvalidImportedChunkIndex) {
            outError = "renderer rejected the destination interior";
            return false;
        }
        const std::size_t previousInteriorChunk = m_interiorChunk;

        m_goldenClawPuzzle.reset();
        m_disabledBethesdaCollisionReferences.clear();

        m_streamer->waitIdle();
        std::string clearError;
        if (!m_streamer->selectWorldspace(
                m_streamer->currentWorldspaceEditorId(), m_renderer, clearError)) {
            m_renderer.removeImportedSceneChunk(destinationChunk);
            outError = "could not clear exterior residency: " + clearError;
            return false;
        }
        m_streamDoorsByCell.clear();
        m_doors.clear();
        m_collision.clear();
        m_actorNavigation.clear();
        m_bethesdaCollisionByCell.clear();
        if (m_bethesdaSessionConfigured) {
            m_bethesdaSession.physics().clearStreamedStaticCollision();
        }
        if (m_distantLodChunk != render::Renderer::kInvalidImportedChunkIndex) {
            m_renderer.removeImportedSceneChunk(m_distantLodChunk);
            m_distantLodChunk = render::Renderer::kInvalidImportedChunkIndex;
        }
        m_skyrimTerrainLodTileValid = false;
        m_skyrimTerrainLodWorldspace.clear();
        if (m_skyrimObjectLodChunk != render::Renderer::kInvalidImportedChunkIndex) {
            m_renderer.removeImportedSceneChunk(m_skyrimObjectLodChunk);
            m_skyrimObjectLodChunk = render::Renderer::kInvalidImportedChunkIndex;
            m_skyrimObjectLodTileValid = false;
        }
        m_skyrimObjectLodWorldspace.clear();
        for (const SkinnedActor& actor : m_actors) {
            m_renderer.setSkinnedActorVisible(actor.instanceSlot, false);
        }
        if (previousInteriorChunk != render::Renderer::kInvalidImportedChunkIndex &&
            previousInteriorChunk != destinationChunk) {
            m_renderer.removeImportedSceneChunk(previousInteriorChunk);
        }

        m_interiorChunk = destinationChunk;
        m_currentInteriorSourceScene = scene;
        const importer::CellCoord collisionCell{
            static_cast<std::int32_t>(std::floor(door.arrivalPosition[0] / 4096.0f)),
            static_cast<std::int32_t>(std::floor(door.arrivalPosition[2] / 4096.0f))};
        m_collision.addCell(collisionCell, scene);
        m_actorNavigation.addCell(collisionCell, interior.navMeshes);
        if (interior.navMeshes.empty() && (m_streamIsMorrowind || m_streamIsOblivion)) {
            m_actorNavigation.addGeneratedCell(collisionCell, scene);
        }
        m_actorNavigation.setResidentDoors({});
        if (m_bethesdaSessionConfigured) cacheBethesdaCollisionCell(collisionCell, scene);
        m_doors = scene.doors;
        m_currentInteriorEditorId = door.targetCellEditorId;
        m_interiorStarted = true;
        m_renderer.setImportedSceneInteriorMode(true);

        render::ImportedInteriorLighting lighting{};
        lighting.enabled = true;
        lighting.hasAuthoredLighting = interior.hasLighting;
        lighting.fogNear = interior.fogNear;
        lighting.fogFar = interior.fogFar;
        lighting.showSky = interior.showSky;
        lighting.useSkyLighting = interior.useSkyLighting;
        lighting.localShadowMode =
            render::ImportedInteriorLighting::LocalShadowMode::ShadowMapsWithContact;
        lighting.indirectLightingMode =
            render::ImportedInteriorLighting::IndirectLightingMode::ScreenSpaceDiffuse;
        for (int channel = 0; channel < 3; ++channel) {
            lighting.ambientColor[channel] = srgbChannelToLinear(interior.ambientColor[channel]);
            lighting.directionalColor[channel] =
                srgbChannelToLinear(interior.directionalColor[channel]);
            lighting.fogColor[channel] = srgbChannelToLinear(interior.fogColor[channel]);
            if (m_streamIsSkyrim) {
                // TES5's XCLL colors assumed the original game's brighter
                // ambient/exposure response. Lift their linear contribution,
                // while retaining the authored hue and all local-light shadow
                // contrast, instead of adding a flat unshadowed light floor.
                lighting.ambientColor[channel] =
                    std::min(lighting.ambientColor[channel] * 1.55f, 1.0f);
                lighting.directionalColor[channel] =
                    std::min(lighting.directionalColor[channel] * 1.18f, 1.0f);
            }
        }
        m_renderer.setImportedInteriorLighting(lighting);
    } else if (door.targetKind == importer::ImportedSceneDoorTargetKind::Exterior) {
        if (door.targetWorldspaceEditorId.empty()) {
            outError = "exterior door has no destination worldspace identity";
            return false;
        }
        if (!m_streamer->hasWorldspace(door.targetWorldspaceEditorId)) {
            outError = "destination worldspace is not present in the active load order: " +
                       door.targetWorldspaceEditorId;
            return false;
        }
        m_goldenClawPuzzle.reset();
        m_disabledBethesdaCollisionReferences.clear();
        m_streamer->waitIdle();
        if (m_interiorChunk != render::Renderer::kInvalidImportedChunkIndex) {
            m_renderer.removeImportedSceneChunk(m_interiorChunk);
            m_interiorChunk = render::Renderer::kInvalidImportedChunkIndex;
        }
        m_currentInteriorSourceScene.reset();
        m_collision.clear();
        m_actorNavigation.clear();
        m_bethesdaCollisionByCell.clear();
        if (m_bethesdaSessionConfigured) {
            m_bethesdaSession.physics().clearStreamedStaticCollision();
        }
        m_streamDoorsByCell.clear();
        m_doors.clear();
        if (m_distantLodChunk != render::Renderer::kInvalidImportedChunkIndex) {
            m_renderer.removeImportedSceneChunk(m_distantLodChunk);
            m_distantLodChunk = render::Renderer::kInvalidImportedChunkIndex;
        }
        m_skyrimTerrainLodTileValid = false;
        m_skyrimTerrainLodWorldspace.clear();
        if (m_skyrimObjectLodChunk != render::Renderer::kInvalidImportedChunkIndex) {
            m_renderer.removeImportedSceneChunk(m_skyrimObjectLodChunk);
            m_skyrimObjectLodChunk = render::Renderer::kInvalidImportedChunkIndex;
        }
        m_skyrimObjectLodTileValid = false;
        m_skyrimObjectLodWorldspace.clear();
        if (!m_streamer->selectWorldspace(
                door.targetWorldspaceEditorId, m_renderer, outError)) {
            return false;
        }
        m_streamWorldspace = door.targetWorldspaceEditorId;
        m_currentInteriorEditorId.clear();
        m_interiorStarted = false;
        m_renderer.setImportedSceneInteriorMode(false);
        m_renderer.setImportedInteriorLighting(render::ImportedInteriorLighting{});
        // A child/DLC worldspace may name a different climate, or inherit one
        // from its parent. Re-select automatic weather only after the worldspace
        // swap commits; an explicit --weather or restored weather remains
        // authoritative across the door.
        if (m_requestedWeatherEditorId.empty()) {
            m_activeWeatherFormId = 0u;
        }
        initWeather();
    } else {
        outError = "unsupported streamed door target";
        return false;
    }

    m_cameraX = door.arrivalPosition[0];
    m_cameraY = door.arrivalPosition[1] + m_collision.tuning().eyeHeight;
    m_cameraZ = door.arrivalPosition[2];
    m_yawDegrees = door.arrivalYawDegrees;
    m_pitchDegrees = 0.0f;
    m_walkMode = true;
    if (m_bethesdaPlayerControllerRegistered) {
        relocateBethesdaPlayerControllerToCamera();
    }

    if (!m_interiorStarted) {
        const float engine[3] = {m_cameraX, m_cameraY, m_cameraZ};
        const float zero[3] = {};
        float fallout[3] = {};
        importer::fnv::CellStreamer::engineToFallout(engine, fallout);
        m_streamer->update(m_renderer, fallout, zero);
        m_streamer->waitIdle();
        m_streamer->update(m_renderer, fallout, zero);
    }
    reloadActorsForCurrentSpace();
    VOX_LOGI("newvegas") << "door transition complete: "
                         << (m_interiorStarted ? m_currentInteriorEditorId
                                               : m_streamWorldspace);
    return true;
}

void BethesdaApp::arrangeActorParadeIfRequested() {
    const char* parade = std::getenv("ODAI_FNV_ACTORS_PARADE");
    if (parade == nullptr || m_actors.empty()) {
        return;
    }

    const char* drawMode = std::getenv("ODAI_FNV_DRAW");
    if (drawMode != nullptr && std::strcmp(drawMode, "actors") == 0) {
        const std::size_t before = m_actors.size();
        std::erase_if(m_actors, [](const SkinnedActor& actor) {
            return !hasHumanoidTorsoCoverage(actor);
        });
        VOX_LOGI("newvegas") << "actor-only parade: retained " << m_actors.size()
                              << "/" << before
                              << " complete walking humanoids";
    }
    if (m_actors.empty()) {
        VOX_LOGW("newvegas") << "actor parade: no complete walking humanoids available";
        return;
    }

    if (m_streamIsMorrowind && m_interiorStarted) {
        // Interior coordinates are local to one authored room complex. The
        // exterior diagnostic below used the camera's geometry-bounds spawn as
        // its anchor and terrain-only height sampling; in Almas Thirr that put
        // the camera between shell pieces and the actors over the void. Anchor
        // this showcase on a resident's authored foot position instead, then
        // choose the nearby direction with the most actual collision floor.
        std::erase_if(m_actors, [](const SkinnedActor& actor) {
            return actor.character.vertices.empty() || actor.draws.empty();
        });
        if (m_actors.empty()) {
            VOX_LOGW("newvegas") << "TES3 interior parade: no renderable residents";
            return;
        }

        constexpr std::size_t kDemoActors = 6u;
        constexpr std::array<float, kDemoActors> kForward = {
            150.0f, 150.0f, 150.0f, 260.0f, 260.0f, 260.0f};
        constexpr std::array<float, kDemoActors> kSide = {
            -70.0f, 0.0f, 70.0f, -70.0f, 0.0f, 70.0f};
        const float anchorX = m_actors.front().position[0];
        const float anchorZ = m_actors.front().position[2];
        float anchorY = m_actors.front().position[1];
        (void)m_collision.groundHeight(anchorX, anchorZ, anchorY, anchorY);

        struct DemoSlot {
            float x = 0.0f;
            float y = 0.0f;
            float z = 0.0f;
        };
        std::vector<DemoSlot> bestSlots;
        float bestForwardX = 1.0f;
        float bestForwardZ = 0.0f;
        bool usedAuthoredFloorPlane = false;
        for (int direction = 0; direction < 16; ++direction) {
            const float angle = static_cast<float>(direction) * (2.0f * kPi / 16.0f);
            const float forwardX = std::cos(angle);
            const float forwardZ = std::sin(angle);
            const float rightX = -forwardZ;
            const float rightZ = forwardX;
            std::vector<DemoSlot> slots;
            for (std::size_t slot = 0u; slot < kDemoActors; ++slot) {
                const float x = anchorX + forwardX * kForward[slot] + rightX * kSide[slot];
                const float z = anchorZ + forwardZ * kForward[slot] + rightZ * kSide[slot];
                float ground = anchorY;
                if (m_collision.groundHeight(x, z, anchorY, ground) &&
                    std::abs(ground - anchorY) <= 80.0f) {
                    slots.push_back({x, ground, z});
                }
            }
            if (slots.size() > bestSlots.size()) {
                bestSlots = std::move(slots);
                bestForwardX = forwardX;
                bestForwardZ = forwardZ;
            }
        }
        if (bestSlots.empty()) {
            // Some legacy interiors expose visible floor geometry only through
            // retained instances and therefore have no queryable collision
            // triangles yet. The anchor itself is an authored NPC foot point,
            // so a compact formation on that same plane is still materially
            // safer than the old geometry-bounds/camera plane. Follow the
            // resident's authored facing direction: NPCs are normally oriented
            // into their room rather than into its wall.
            const odai::math::Vector3 authoredFacing =
                actorFacing(m_actors.front().yawRadians);
            bestForwardX = authoredFacing.x;
            bestForwardZ = authoredFacing.z;
            const float rightX = -bestForwardZ;
            const float rightZ = bestForwardX;
            for (std::size_t slot = 0u; slot < kDemoActors; ++slot) {
                bestSlots.push_back({
                    anchorX + bestForwardX * kForward[slot] + rightX * kSide[slot],
                    anchorY,
                    anchorZ + bestForwardZ * kForward[slot] + rightZ * kSide[slot]});
            }
            usedAuthoredFloorPlane = true;
            VOX_LOGW("newvegas")
                << "TES3 interior parade: collision floor unavailable; using compact "
                   "authored-resident floor plane";
        }

        const std::size_t actorCount = std::min(m_actors.size(), bestSlots.size());
        m_actors.resize(actorCount);
        for (std::size_t i = 0u; i < actorCount; ++i) {
            SkinnedActor& actor = m_actors[i];
            actor.position[0] = bestSlots[i].x;
            actor.position[1] = bestSlots[i].y;
            actor.position[2] = bestSlots[i].z;
            actor.wanderOrigin[0] = actor.wanderTarget[0] = actor.position[0];
            actor.wanderOrigin[1] = actor.wanderTarget[1] = actor.position[1];
            actor.wanderOrigin[2] = actor.wanderTarget[2] = actor.position[2];
            actor.projectedToNavigation = false;
            actor.wanderPath.clear();
            actor.wanderPathIndex = 0u;
            actor.wanderPauseSeconds = 0.0f;
            actor.yawRadians = actorYawForDirection(-bestForwardX, -bestForwardZ);
        }
        m_cameraX = anchorX;
        m_cameraY = anchorY + kEyeHeightUnits;
        m_cameraZ = anchorZ;
        m_yawDegrees = std::atan2(bestForwardZ, bestForwardX) * (180.0f / kPi);
        m_pitchDegrees = 0.0f;
        VOX_LOGI("newvegas") << "TES3 interior parade: placed " << actorCount
                              << " actors on "
                              << (usedAuthoredFloorPlane
                                      ? "the authored resident floor plane"
                                      : "collision floor")
                              << " around authored anchor ("
                              << anchorX << ", " << anchorY << ", " << anchorZ << ")";
        return;
    }

    // Lay the crowd across the camera's view at a fixed distance. This is a
    // diagnostic/demo arrangement: the normal runtime preserves authored
    // placements when the environment switch is absent.
    constexpr float spacing = 130.0f;
    const float distance = std::max(200.0f, static_cast<float>(std::atof(parade)));
    const float yaw = m_yawDegrees * (kPi / 180.0f);
    const float forwardX = std::cos(yaw);
    const float forwardZ = std::sin(yaw);
    const float centreX = m_cameraX + (forwardX * distance);
    const float centreZ = m_cameraZ + (forwardZ * distance);
    for (std::size_t i = 0; i < m_actors.size(); ++i) {
        SkinnedActor& actor = m_actors[i];
        const float offset =
            (static_cast<float>(i) -
             (static_cast<float>(m_actors.size() - 1) * 0.5f)) * spacing;
        actor.position[0] = centreX + (forwardZ * offset);
        actor.position[2] = centreZ - (forwardX * offset);

        float ground = 0.0f;
        const bool onGround = m_streamer
            ? m_collision.terrainHeight(actor.position[0], actor.position[2], ground)
            : groundHeightAt(actor.position[0], actor.position[2], ground);
        actor.position[1] = onGround ? ground : (m_cameraY - kEyeHeightUnits);

        // The navigation pass may make a small local correction on its next
        // tick, but its new wander centre must be the parade position rather
        // than the actor's distant authored placement.
        actor.wanderOrigin[0] = actor.position[0];
        actor.wanderOrigin[1] = actor.position[1];
        actor.wanderOrigin[2] = actor.position[2];
        actor.wanderTarget[0] = actor.position[0];
        actor.wanderTarget[1] = actor.position[1];
        actor.wanderTarget[2] = actor.position[2];
        actor.projectedToNavigation = false;
        actor.wanderPath.clear();
        actor.wanderPathIndex = 0u;
        actor.wanderPauseSeconds = 0.0f;
        actor.yawRadians = actorYawForDirection(-forwardX, -forwardZ);
    }
    VOX_LOGI("newvegas") << "actor parade: placed " << m_actors.size()
                          << " actors " << distance << " units ahead of the camera";
}

bool BethesdaApp::ensureSkyrimActorCatalog() {
    if (m_skyrimActorCatalogReady) return true;
    if (!m_streamIsSkyrim || m_streamer == nullptr) return false;
    std::string error;
    const std::filesystem::path pluginPath =
        std::filesystem::path(m_streamDirectory) / m_streamPlugin;
    const bool loaded = !m_streamLoadOrder.empty()
        ? importer::fnv::findAllActorsAcrossOrder(
              m_streamLoadOrder, m_skyrimActorCatalog,
              m_skyrimActorVoiceFolderPlugin, error)
        : importer::fnv::findAllActors(pluginPath, m_skyrimActorCatalog, error);
    if (!loaded) {
        VOX_LOGE("runtime") << "Skyrim actor catalog failed: " << error;
        return false;
    }
    m_skyrimActorCatalogReady = true;
    VOX_LOGI("runtime") << "Skyrim actor catalog: "
                         << m_skyrimActorCatalog.placements.size()
                         << " winning placements, "
                         << m_skyrimActorCatalog.bases.size() << " bases";
    return true;
}

bool BethesdaApp::bindAndMaterializeScenarioReferences(std::string& outError) {
    outError.clear();
    if (!m_bethesdaSessionConfigured || !ensureSkyrimActorCatalog() ||
        m_streamer == nullptr || m_streamLoadOrder.empty()) {
        outError = "scenario reference materialization has no resolved Skyrim content";
        return false;
    }

    // QUST ALUA identifies a unique ACTOR BASE, while Papyrus aliases hold the
    // live placed reference. The generic form resolver cannot infer that
    // distinction and initially produces an ObjectId for the base. Resolve it
    // through the immutable winning ACHR catalog before any startup fragment
    // executes. Forced-reference aliases already name a catalog ref directly.
    std::unordered_map<std::uint32_t,
        std::vector<const importer::fnv::FalloutActorPlacement*>> placementsByBase;
    std::unordered_set<std::uint32_t> placedReferences;
    for (const auto& placement : m_skyrimActorCatalog.placements) {
        placedReferences.insert(placement.refFormId);
        placementsByBase[placement.baseFormId].push_back(&placement);
    }
    std::size_t reboundAliases = 0u;
    for (const auto& [questName, quest] : m_bethesdaSession.quests()) {
        (void)questName;
        for (const bethesda::QuestAliasRuntimeState& alias : quest.aliases) {
            if (alias.sourceFormId == 0u ||
                placedReferences.contains(alias.sourceFormId)) {
                continue;
            }
            const auto candidates = placementsByBase.find(alias.sourceFormId);
            if (candidates == placementsByBase.end() ||
                candidates->second.size() != 1u) {
                continue;
            }
            bethesda::RecordKey reference;
            std::string error;
            if (!bethesda::stableRecordKey(
                    m_streamLoadOrder, candidates->second.front()->refFormId,
                    reference, error) ||
                !m_bethesdaSession.bindQuestAliasTarget(
                    bethesda::ObjectId::persistent(quest.record), alias.id,
                    bethesda::ObjectId::persistent(std::move(reference)), error)) {
                outError = "could not bind unique actor alias " + quest.editorId + ":" +
                    std::to_string(alias.id) + ": " + error;
                return false;
            }
            ++reboundAliases;
        }
    }

    std::vector<bethesda::ObjectId> referencedObjects;
    const auto addObject = [&](const bethesda::ObjectId& object) {
        if (object.kind == bethesda::ObjectIdKind::PersistentReference) {
            referencedObjects.push_back(object);
        }
    };
    for (const auto& [questName, quest] : m_bethesdaSession.quests()) {
        (void)questName;
        for (const bethesda::QuestAliasRuntimeState& alias : quest.aliases) {
            addObject(alias.target);
        }
    }
    std::function<void(const bethesda::PapyrusValue&)> collectValue;
    collectValue = [&](const bethesda::PapyrusValue& value) {
        if (value.type == bethesda::PapyrusValueType::Object) {
            addObject(value.object);
        } else if (value.type == bethesda::PapyrusValueType::Array) {
            for (const bethesda::PapyrusValue& element : value.array) {
                collectValue(element);
            }
        }
    };
    for (const bethesda::PapyrusScriptInstanceSnapshot& instance :
         m_bethesdaSession.papyrus().snapshot().instances) {
        for (const auto& [name, value] : instance.properties) {
            (void)name;
            collectValue(value);
        }
    }
    std::sort(referencedObjects.begin(), referencedObjects.end(),
        [](const auto& left, const auto& right) {
            return left.toString() < right.toString();
        });
    referencedObjects.erase(std::unique(
        referencedObjects.begin(), referencedObjects.end()), referencedObjects.end());

    std::size_t materialized = 0u;
    for (const bethesda::ObjectId& id : referencedObjects) {
        if (m_bethesdaSession.world().find(id) != nullptr) continue;
        std::uint32_t resolvedReferenceFormId = 0u;
        std::string error;
        if (!bethesda::resolvedFormId(
                m_streamLoadOrder, id.reference,
                resolvedReferenceFormId, error)) {
            continue;
        }
        const auto actorPlacement = std::find_if(
            m_skyrimActorCatalog.placements.begin(),
            m_skyrimActorCatalog.placements.end(), [&](const auto& placement) {
                return placement.refFormId == resolvedReferenceFormId;
            });

        bethesda::RuntimeObject object;
        object.id = id;
        object.persistent = true;
        std::uint32_t baseFormId = 0u;
        if (actorPlacement != m_skyrimActorCatalog.placements.end()) {
            baseFormId = actorPlacement->baseFormId;
            object.kind = bethesda::RuntimeObjectKind::Actor;
            object.enabled = !actorPlacement->initiallyDisabled;
            object.actorValues.emplace();
            object.referenceTypes.reserve(actorPlacement->referenceTypeFormIds.size());
            for (const std::uint32_t referenceTypeFormId :
                 actorPlacement->referenceTypeFormIds) {
                bethesda::RecordKey referenceType;
                if (bethesda::stableRecordKey(
                        m_streamLoadOrder, referenceTypeFormId,
                        referenceType, error)) {
                    object.referenceTypes.push_back(std::move(referenceType));
                }
            }
            const std::vector<std::uint32_t> inventory =
                m_skyrimActorCatalog.materializeInventory(
                    baseFormId, resolvedReferenceFormId);
            for (const std::uint32_t itemFormId : inventory) {
                bethesda::RecordKey item;
                if (!bethesda::stableRecordKey(
                        m_streamLoadOrder, itemFormId, item, error)) {
                    continue;
                }
                auto entry = std::find_if(object.inventory.begin(), object.inventory.end(),
                    [&](const bethesda::InventoryEntry& value) {
                        return value.item == item;
                    });
                if (entry == object.inventory.end()) {
                    object.inventory.push_back({std::move(item), 1, false});
                } else {
                    ++entry->count;
                }
            }
            const float bethesdaPosition[3] = {
                actorPlacement->position[0], actorPlacement->position[1],
                actorPlacement->position[2]};
            float enginePosition[3] = {};
            importer::fnv::CellStreamer::falloutToEngine(
                bethesdaPosition, enginePosition);
            object.transform.position = {
                enginePosition[0], enginePosition[1], enginePosition[2]};
            object.transform.rotationRadians[1] =
                -actorPlacement->rotationRadians[2];
        } else {
            std::vector<std::uint8_t> vmad;
            std::size_t sourcePluginIndex = 0u;
            if (!m_streamer->referenceGameplayData(
                    resolvedReferenceFormId, baseFormId, vmad,
                    sourcePluginIndex, error)) {
                continue;  // Form, quest, scene, faction, or other non-reference.
            }
            (void)vmad;
            (void)sourcePluginIndex;
            object.kind = bethesda::RuntimeObjectKind::Activator;
            float enginePosition[3] = {};
            if (!m_streamer->referencePositionEngineSpace(
                    resolvedReferenceFormId, enginePosition, error)) {
                continue;
            }
            object.transform.position = {
                enginePosition[0], enginePosition[1], enginePosition[2]};
        }
        if (!bethesda::stableRecordKey(
                m_streamLoadOrder, baseFormId, object.base, error)) {
            continue;
        }
        bethesda::RuntimeSpaceState origin;
        if (runtimeOriginSpaceForReference(resolvedReferenceFormId, origin)) {
            object.originSpace = origin;
            object.currentSpace = origin;
            object.interior = origin.kind == bethesda::RuntimeSpaceKind::Interior;
        }
        if (const auto ownership =
                m_streamer->referenceCellOwnership(resolvedReferenceFormId);
            ownership.has_value() && ownership->locationFormId != 0u) {
            (void)bethesda::stableRecordKey(
                m_streamLoadOrder, ownership->locationFormId,
                object.location, error);
        }
        std::sort(object.inventory.begin(), object.inventory.end(),
            [](const auto& left, const auto& right) {
                return left.item < right.item;
            });
        std::sort(object.referenceTypes.begin(), object.referenceTypes.end());
        const bethesda::RecordKey materializedBase = object.base;
        if (!m_bethesdaSession.world().addInitialObject(std::move(object), error)) {
            outError = "could not materialize scenario reference " +
                id.toString() + ": " + error;
            return false;
        }
        if (actorPlacement != m_skyrimActorCatalog.placements.end()) {
            (void)m_bethesdaSession.bindQuestInventoryForActor(
                id, materializedBase, error);
            // The alias-created inventory path reports separately and does not
            // make the actor materialization itself invalid.
            error.clear();
        }
        ++materialized;
    }
    VOX_LOGI("scenario") << "scenario runtime closure: " << reboundAliases
                          << " unique actor aliases rebound, " << materialized
                          << " placed references materialized before VM startup";
    return true;
}

bool BethesdaApp::runtimeOriginSpaceForReference(
    std::uint32_t referenceFormId,
    bethesda::RuntimeSpaceState& outSpace) const {
    outSpace = {};
    if (m_streamer == nullptr || m_streamLoadOrder.empty()) return false;
    const auto ownership = m_streamer->referenceCellOwnership(referenceFormId);
    if (!ownership.has_value()) return false;
    std::string error;
    if (ownership->cellFormId != 0u &&
        !bethesda::stableRecordKey(
            m_streamLoadOrder, ownership->cellFormId, outSpace.cell, error)) {
        return false;
    }
    if (ownership->interior) {
        if (!outSpace.cell.valid()) return false;
        outSpace.kind = bethesda::RuntimeSpaceKind::Interior;
        return true;
    }
    if (ownership->worldspaceFormId == 0u ||
        !bethesda::stableRecordKey(
            m_streamLoadOrder, ownership->worldspaceFormId,
            outSpace.worldspace, error)) {
        return false;
    }
    outSpace.kind = bethesda::RuntimeSpaceKind::Exterior;
    outSpace.gridX = ownership->gridX;
    outSpace.gridZ = ownership->gridZ;
    return true;
}

bool BethesdaApp::runtimeSpaceForPosition(
    const float enginePosition[3],
    bethesda::RuntimeSpaceState& outSpace) const {
    outSpace = {};
    if (m_streamer == nullptr || m_streamLoadOrder.empty()) return false;
    std::string error;
    if (m_interiorStarted) {
        const std::uint32_t cellFormId =
            m_streamer->cellFormIdForInterior(m_currentInteriorEditorId);
        if (cellFormId == 0u ||
            !bethesda::stableRecordKey(
                m_streamLoadOrder, cellFormId, outSpace.cell, error)) {
            return false;
        }
        outSpace.kind = bethesda::RuntimeSpaceKind::Interior;
        return true;
    }
    const std::uint32_t worldspaceFormId = m_streamer->currentWorldspaceFormId();
    if (worldspaceFormId == 0u ||
        !bethesda::stableRecordKey(
            m_streamLoadOrder, worldspaceFormId, outSpace.worldspace, error)) {
        return false;
    }
    float fallout[3] = {};
    importer::fnv::CellStreamer::engineToFallout(enginePosition, fallout);
    const float cellSize = m_streamer->cellWorldSize();
    if (cellSize <= 0.0f) return false;
    outSpace.kind = bethesda::RuntimeSpaceKind::Exterior;
    outSpace.gridX = static_cast<std::int32_t>(std::floor(fallout[0] / cellSize));
    outSpace.gridZ = static_cast<std::int32_t>(std::floor(fallout[1] / cellSize));
    const std::uint32_t cellFormId =
        m_streamer->cellFormIdAtFallout(fallout[0], fallout[1]);
    if (cellFormId != 0u) {
        bethesda::RecordKey cell;
        if (bethesda::stableRecordKey(
                m_streamLoadOrder, cellFormId, cell, error)) {
            outSpace.cell = std::move(cell);
        }
    }
    return true;
}

bool BethesdaApp::runtimeSpaceIsResident(
    const bethesda::RuntimeSpaceState& space) const {
    if (m_streamer == nullptr || m_streamLoadOrder.empty()) return false;
    std::string error;
    if (space.kind == bethesda::RuntimeSpaceKind::Interior) {
        if (!m_interiorStarted || !space.cell.valid()) return false;
        const std::uint32_t currentCellFormId =
            m_streamer->cellFormIdForInterior(m_currentInteriorEditorId);
        bethesda::RecordKey currentCell;
        return currentCellFormId != 0u &&
            bethesda::stableRecordKey(
                m_streamLoadOrder, currentCellFormId, currentCell, error) &&
            currentCell == space.cell;
    }
    if (space.kind == bethesda::RuntimeSpaceKind::Exterior) {
        if (m_interiorStarted || !space.worldspace.valid()) return false;
        std::uint32_t worldspaceFormId = 0u;
        return bethesda::resolvedFormId(
                   m_streamLoadOrder, space.worldspace, worldspaceFormId, error) &&
            m_streamer->isExteriorCellResident(
                worldspaceFormId, space.gridX, space.gridZ);
    }
    return false;
}

void BethesdaApp::reloadActorsForCurrentSpace() {
    // Fallout/New Vegas still give Victor special startup treatment. Skyrim's
    // streamed traversal has no fixed companion and can rebuild the nearby
    // population uniformly whenever a door changes ownership space.
    if (!m_streamIsSkyrim || m_streamer == nullptr) {
        return;
    }

    endConversation();
    unregisterBethesdaActorControllers();
    for (SkinnedActor& actor : m_actors) {
        // Cell eviction is not an authored Disable. The persistent runtime
        // object's enabled bit belongs to quest/scripts and must survive this
        // presentation residency change unchanged.
        if (actor.uploaded) {
            m_renderer.setSkinnedActorVisible(actor.instanceSlot, false);
        }
    }
    m_actors.clear();
    m_victorIndex = -1;
    m_activationActor = -1;
    m_activationLootActor = -1;
    queueActorUploads();

    const float engineCentre[3] = {m_cameraX, m_cameraY, m_cameraZ};
    float bethesdaCentre[3] = {};
    importer::fnv::CellStreamer::engineToFallout(engineCentre, bethesdaCentre);
    ActorPopulationStats actorStats;
    if (!ensureSkyrimActorCatalog()) return;
    const auto runtimePlacementResolver = [this, &bethesdaCentre](
            importer::fnv::FalloutActorPlacement& placement) {
        const bethesda::RuntimeObject* runtime = nullptr;
        if (m_bethesdaSessionConfigured && placement.refFormId != 0u) {
            bethesda::RecordKey reference;
            std::string error;
            if (bethesda::stableRecordKey(
                    m_streamLoadOrder, placement.refFormId, reference, error)) {
                runtime = m_bethesdaSession.world().find(
                    bethesda::ObjectId::persistent(std::move(reference)));
            }
        }
        if (runtime != nullptr &&
            runtime->currentSpace.kind != bethesda::RuntimeSpaceKind::Unknown) {
            if (!runtimeSpaceIsResident(runtime->currentSpace)) return false;
            const float engine[3] = {
                static_cast<float>(runtime->transform.position[0]),
                static_cast<float>(runtime->transform.position[1]),
                static_cast<float>(runtime->transform.position[2])};
            importer::fnv::CellStreamer::engineToFallout(engine, placement.position);
            placement.rotationRadians[2] =
                -static_cast<float>(runtime->transform.rotationRadians[1]);
            // Authored Initially Disabled is only the initial state. A quest
            // that enabled this persistent object must be able to materialize
            // its presentation on a later residency refresh.
            placement.initiallyDisabled = !runtime->enabled;
        } else {
            const bool resident = m_interiorStarted
                ? m_streamer->referenceBelongsToInterior(
                      placement.refFormId, m_currentInteriorEditorId)
                : m_streamer->referenceBelongsToResidentExteriorCell(
                      placement.refFormId);
            if (!resident) return false;
        }
        const float dx = placement.position[0] - bethesdaCentre[0];
        const float dy = placement.position[1] - bethesdaCentre[1];
        return ((dx * dx) + (dy * dy)) <= (kActorLoadRadius * kActorLoadRadius);
    };
    const std::filesystem::path dataPath(m_streamDirectory);
    loadGoodspringsActors(
        dataPath / m_streamPlugin,
        m_streamLoadOrder.empty() ? nullptr : &m_streamLoadOrder,
        m_streamer->assets(), bethesdaCentre, kActorLoadRadius,
        kFirstCrowdSkinnedInstance,
        (thirdPersonPlayerShowcase()
             ? kPlayerAvatarSkinnedInstance
             : render::kMaxSkinnedInstances) - kFirstCrowdSkinnedInstance,
        {}, {}, m_actors, actorStats,
        &m_skyrimActorCatalog, &m_skyrimActorVoiceFolderPlugin,
        runtimePlacementResolver);

    arrangeActorParadeIfRequested();

    if (!m_actors.empty()) {
        std::string dialogueDetail;
        if (m_scenarioId.empty() || !m_streamIsSkyrim) {
            loadActorDialogue(
                dataPath / m_streamPlugin,
                m_streamLoadOrder.empty() ? nullptr : &m_streamLoadOrder,
                m_actors, dialogueDetail);
        } else {
            dialogueDetail = "retail Skyrim dialogue owned by BethesdaSession";
        }
        VOX_LOGI("newvegas") << "actor dialogue after transition: " << dialogueDetail;

        std::string voiceDetail;
        loadActorVoices(
            dataPath, m_streamPlugin, m_modDirectories, m_actors, voiceDetail);
        VOX_LOGI("newvegas") << "actor voices after transition: " << voiceDetail;
        queueActorUploads();
    }
    VOX_LOGI("newvegas")
        << "actors after transition into "
        << (m_interiorStarted ? m_currentInteriorEditorId : m_streamWorldspace)
        << ": " << actorStats.detail;
    // Existing ObjectIds restore their runtime transform before controllers
    // are registered. New catalog entries are registered from authored data.
    restoreBethesdaActorsFromSession();
    m_skyrimActorResidencyDirty = false;
    std::string puzzleError;
    if (!configureGoldenClawPuzzleForCurrentSpace(puzzleError)) {
        VOX_LOGE("scenario") << "Golden Claw compatibility error: " << puzzleError;
        m_toasts.push("Compatibility error", puzzleError, "golden-claw-compatibility");
    }
}

void BethesdaApp::queueActorUploads() {
    m_nextActorUploadIndex = 0u;
    m_actorUploadSuccessCount = 0u;
    m_actorUploadedTextureCount = 0u;
    m_actorTotalTextureCount = 0u;
    m_actorsUploadPending = std::any_of(
        m_actors.begin(), m_actors.end(), [](const SkinnedActor& actor) {
            return !actor.character.vertices.empty() && !actor.draws.empty() &&
                actor.instanceSlot != 0u;
        });
}

void BethesdaApp::realizePendingActorUploads(std::size_t maxActorUploads) {
    // Textures first, because their bindless slots are baked into the template
    // vertices. The backend now seeds skinned output buffers with the rest pose,
    // so this is safe during startup before the first skinning dispatch.
    if (m_skyrimPlayerAvatarUploadPending && m_skyrimPlayerAvatar.has_value()) {
        SkinnedActor& avatar = *m_skyrimPlayerAvatar;
        const std::vector<std::uint32_t> slots =
            m_renderer.uploadSkinnedActorTextures(avatar.instanceSlot, avatar.textures);
        remapActorTextureSlots(avatar, slots);
        render::ImportedSkinnedMeshTemplate meshTemplate{};
        meshTemplate.vertices = avatar.character.vertices;
        meshTemplate.indices = avatar.character.indices;
        meshTemplate.draws = avatar.draws;
        meshTemplate.boneCount =
            static_cast<std::uint32_t>(avatar.character.skeleton.bones.size());
        avatar.uploaded = m_renderer.uploadSkinnedMeshTemplate(
            avatar.instanceSlot, meshTemplate);
        m_skyrimPlayerAvatarUploadPending = false;
        if (!avatar.uploaded) {
            VOX_LOGE("showcase") << "Skyrim player GPU template upload failed";
        }
    }

    if (!m_actorsUploadPending || maxActorUploads == 0u) {
        return;
    }
    std::size_t submittedActors = 0u;
    while (m_nextActorUploadIndex < m_actors.size() &&
           submittedActors < maxActorUploads) {
        SkinnedActor& actor = m_actors[m_nextActorUploadIndex++];
        // TES3 keeps activation proxies even when an optional creature or
        // modded body mesh cannot be assembled. Those proxies must not upload
        // an empty template into renderer slot zero.
        if (actor.character.vertices.empty() || actor.draws.empty() ||
            actor.instanceSlot == 0u) {
            continue;
        }
        ++submittedActors;
        const std::vector<std::uint32_t> slots =
            m_renderer.uploadSkinnedActorTextures(actor.instanceSlot, actor.textures);
        remapActorTextureSlots(actor, slots);
        for (const std::uint32_t slot : slots) {
            m_actorUploadedTextureCount += (slot != 0xffffffffu) ? 1u : 0u;
        }
        m_actorTotalTextureCount += slots.size();

        render::ImportedSkinnedMeshTemplate meshTemplate{};
        meshTemplate.vertices = actor.character.vertices;
        meshTemplate.indices = actor.character.indices;
        meshTemplate.draws = actor.draws;
        meshTemplate.boneCount =
            static_cast<std::uint32_t>(actor.character.skeleton.bones.size());
        actor.uploaded = m_renderer.uploadSkinnedMeshTemplate(
            actor.instanceSlot, meshTemplate);
        m_actorUploadSuccessCount += actor.uploaded ? 1u : 0u;
    }
    if (m_nextActorUploadIndex >= m_actors.size()) {
        m_actorsUploadPending = false;
        VOX_LOGI("newvegas")
            << "actors uploaded: " << m_actorUploadSuccessCount << "/"
            << m_actors.size() << ", " << m_actorUploadedTextureCount << "/"
            << m_actorTotalTextureCount << " textures bound";
    }
}

bool BethesdaApp::prewarmSkyrimCityShowcase() {
    if (!m_streamer || m_interiorStarted) {
        VOX_LOGE("showcase") << "Skyrim city startup prewarm requires exterior streaming";
        return false;
    }

    const core::Stopwatch timer;
    const importer::CellResidencyConfig normalConfig = m_streamer->config();
    importer::CellResidencyConfig startupConfig = normalConfig;
    // During prewarm there is no visible frame to protect. Start the complete
    // local ring together and apply it as one batch; after this function the
    // normal one-chunk-per-frame streaming budget is restored.
    startupConfig.maxLoadsInFlight = 16u;
    startupConfig.maxChunkAppliesPerFrame = 16u;
    m_streamer->setConfig(startupConfig);

    const odai::math::Vector3 residencyPosition = thirdPersonPlayerShowcase()
        ? bethesdaPlayerEyePosition()
        : odai::math::Vector3{m_cameraX, m_cameraY, m_cameraZ};
    const float enginePosition[3] = {
        residencyPosition.x, residencyPosition.y, residencyPosition.z};
    float falloutPosition[3] = {};
    importer::fnv::CellStreamer::engineToFallout(enginePosition, falloutPosition);
    const float stationaryVelocity[3] = {};

    constexpr std::size_t kMaximumPrewarmPasses = 8u;
    bool ready = false;
    for (std::size_t pass = 0; pass < kMaximumPrewarmPasses; ++pass) {
        updateStreaming(0.0f);
        m_streamer->waitIdle();
        updateStreaming(0.0f);
        const auto stats = m_streamer->stats();
        if (m_streamer->isStreamingIdle() &&
            stats.residency.loadingCount == 0u) {
            ready = true;
            break;
        }
        // Keep the residency origin stable even if a callback reconstructed
        // the third-person camera while the ring was being installed.
        m_streamer->update(m_renderer, falloutPosition, stationaryVelocity);
    }
    m_streamer->setConfig(normalConfig);

    if (!ready) {
        VOX_LOGE("showcase")
            << "Skyrim city startup prewarm did not settle its initial residency ring";
        return false;
    }
    if (m_skyrimCitySpawnSettlementPending && !settleSkyrimCityShowcasePlayer()) {
        VOX_LOGE("showcase")
            << "Skyrim city startup prewarm could not settle the player on navigation";
        return false;
    }

    realizePendingActorUploads(std::numeric_limits<std::size_t>::max());
    if (!m_renderer.waitForImportedSceneUploads()) {
        VOX_LOGE("showcase") << "Skyrim city startup GPU uploads did not complete";
        return false;
    }
    if (m_whiterunReferenceShowcase) {
        const importer::fnv::CellStreamerStats stats = m_streamer->stats();
        VOX_LOGI("showcase")
            << "Whiterun reference inventory: instances="
            << stats.geometryInstancesLoaded << " banners="
            << stats.bannerInstancesLoaded << " alpha-tested-parts="
            << stats.alphaTestedPartsLoaded << " fire-emitters="
            << stats.fireEmittersLoaded << " local-lights="
            << stats.localLightsLoaded << " terrain-lod="
            << (m_skyrimTerrainLodWorldspace.empty()
                    ? "<missing>" : m_skyrimTerrainLodWorldspace)
            << " object-lod="
            << (m_skyrimObjectLodWorldspace.empty()
                    ? "<missing>" : m_skyrimObjectLodWorldspace);
        if (stats.geometryInstancesLoaded == 0u) {
            VOX_LOGE("showcase")
                << "Whiterun reference has no gate-plaza geometry instances";
            return false;
        }
        if (m_skyrimTerrainLodWorldspace.empty() ||
            m_skyrimObjectLodWorldspace.empty()) {
            VOX_LOGE("showcase")
                << "Whiterun reference is missing inherited Tamriel terrain/object LOD";
            return false;
        }
        if (stats.bannerInstancesLoaded == 0u) {
            VOX_LOGW("showcase")
                << "Whiterun reference inventory found no banner geometry";
        }
        if (stats.fireEmittersLoaded < 2u) {
            VOX_LOGW("showcase")
                << "Whiterun reference inventory found fewer than two fire emitters";
        }
        if (stats.localLightsLoaded < 2u) {
            VOX_LOGW("showcase")
                << "Whiterun reference inventory found fewer than two local lights";
        }
        if (stats.alphaTestedPartsLoaded == 0u) {
            VOX_LOGW("showcase")
                << "Whiterun reference inventory found no alpha-tested foliage/overlays";
        }
    }
    if (thirdPersonPlayerShowcase() &&
        (!m_skyrimPlayerAvatar.has_value() || !m_skyrimPlayerAvatar->uploaded)) {
        VOX_LOGE("showcase") << "Skyrim city startup player avatar is not GPU-resident";
        return false;
    }
    // A gate-cell callback may have settled the capsule while another result
    // from the same prewarm batch was still waiting to be installed. Sweep the
    // boom once more against the complete resident collision set so the first
    // visible frame cannot inherit a camera position inside that final wall.
    if (thirdPersonPlayerShowcase()) {
        reconstructPlayerCamera(1.0f / 60.0f, true);
    }

    // The visible frame loop starts from rest; no synthetic velocity sample
    // should point residency ahead because prewarm touched the tracking origin.
    m_hasPreviousCameraPosition = false;
    VOX_LOGI("showcase") << "cold-start prewarm complete in " << timer.elapsedMs()
                         << " ms: " << m_streamer->stats().residentChunks
                         << " cells, " << m_actors.size() << " actors";
    return true;
}

void BethesdaApp::updateDoorTransition(float deltaSeconds) {
    constexpr float kFadeSeconds = 0.22f;
    if (m_doorTransitionPhase == DoorTransitionPhase::None) {
        return;
    }
    if (m_doorTransitionPhase == DoorTransitionPhase::FadeOut) {
        m_doorTransitionAlpha =
            std::min(1.0f, m_doorTransitionAlpha + (deltaSeconds / kFadeSeconds));
        if (m_doorTransitionAlpha >= 1.0f && m_pendingDoor.has_value()) {
            std::string error;
            if (!completeDoorTransition(*m_pendingDoor, error)) {
                VOX_LOGE("newvegas") << "door transition failed: " << error;
                m_toasts.push("Door unavailable", error);
            }
            m_pendingDoor.reset();
            m_doorTransitionPhase = DoorTransitionPhase::FadeIn;
        }
    } else {
        m_doorTransitionAlpha =
            std::max(0.0f, m_doorTransitionAlpha - (deltaSeconds / kFadeSeconds));
        if (m_doorTransitionAlpha <= 0.0f) {
            m_doorTransitionPhase = DoorTransitionPhase::None;
        }
    }
}

namespace {

// Common install locations for Fallout: New Vegas, in the order they are tried.
// A directory only counts when it actually holds the master plugin -- an empty
// or partial directory would otherwise be "found" and then fail later with a
// much less obvious message.
std::string findFalloutDataDirectory() {
    std::vector<std::filesystem::path> candidates;

    const char* home = std::getenv("HOME");
    if (home != nullptr) {
        const std::filesystem::path homePath(home);
        candidates.push_back(homePath / ".steam/steam/steamapps/common/Fallout New Vegas/Data");
        candidates.push_back(homePath / ".local/share/Steam/steamapps/common/Fallout New Vegas/Data");
        candidates.push_back(homePath / "GOG Games/Fallout New Vegas/Data");
    }
    // WSL and dual-boot mounts of a Windows install.
    candidates.emplace_back("/mnt/c/Program Files (x86)/Steam/steamapps/common/Fallout New Vegas/Data");
    candidates.emplace_back("/mnt/c/GOG Games/Fallout New Vegas/Data");
    // Native Windows.
    candidates.emplace_back("C:/Program Files (x86)/Steam/steamapps/common/Fallout New Vegas/Data");
    candidates.emplace_back("C:/GOG Games/Fallout New Vegas/Data");

    for (const std::filesystem::path& candidate : candidates) {
        std::error_code existsError;
        if (std::filesystem::exists(candidate / "FalloutNV.esm", existsError) && !existsError) {
            return candidate.string();
        }
    }
    return {};
}

std::string findSkyrimDataDirectory() {
    std::vector<std::filesystem::path> candidates;
    if (const char* home = std::getenv("HOME")) {
        const std::filesystem::path homePath(home);
        candidates.push_back(
            homePath / ".steam/steam/steamapps/common/Skyrim Special Edition/Data");
        candidates.push_back(
            homePath / ".local/share/Steam/steamapps/common/Skyrim Special Edition/Data");
    }
    candidates.emplace_back(
        "/mnt/c/Program Files (x86)/Steam/steamapps/common/Skyrim Special Edition/Data");
    candidates.emplace_back(
        "C:/Program Files (x86)/Steam/steamapps/common/Skyrim Special Edition/Data");
    for (const std::filesystem::path& candidate : candidates) {
        std::error_code existsError;
        if (std::filesystem::exists(candidate / "Skyrim.esm", existsError) && !existsError) {
            return candidate.string();
        }
    }
    return {};
}

std::string findMorrowindDataDirectory() {
    std::vector<std::filesystem::path> candidates;
    if (const char* home = std::getenv("HOME")) {
        const std::filesystem::path homePath(home);
        candidates.push_back(homePath /
            ".steam/steam/steamapps/common/Morrowind/Data Files");
        candidates.push_back(homePath /
            ".local/share/Steam/steamapps/common/Morrowind/Data Files");
    }
    candidates.emplace_back(
        "/mnt/c/Program Files (x86)/Steam/steamapps/common/Morrowind/Data Files");
    candidates.emplace_back(
        "C:/Program Files (x86)/Steam/steamapps/common/Morrowind/Data Files");
    for (const std::filesystem::path& candidate : candidates) {
        std::error_code error;
        if (std::filesystem::is_regular_file(candidate / "Morrowind.esm", error) &&
            !error) return candidate.string();
    }
    return {};
}

}  // namespace

bool BethesdaApp::onInit() {
    if (m_balmoraSkyrimPlayerShowcase) {
        if (m_streamDirectory.empty()) m_streamDirectory = findMorrowindDataDirectory();
        if (m_skyrimAvatarDataDirectory.empty()) {
            if (const char* configured = std::getenv("ODAI_SKYRIM_DATA")) {
                m_skyrimAvatarDataDirectory = configured;
            } else {
                m_skyrimAvatarDataDirectory = findSkyrimDataDirectory();
            }
        }
        if (m_streamDirectory.empty() ||
            !std::filesystem::is_regular_file(
                std::filesystem::path(m_streamDirectory) / "Morrowind.esm")) {
            VOX_LOGE("showcase")
                << "balmora-skyrim-player requires Morrowind.esm/Vvardenfell; "
                   "pass --stream \"<Morrowind/Data Files>\"";
            return false;
        }
        if (m_skyrimAvatarDataDirectory.empty() ||
            !std::filesystem::is_regular_file(
                std::filesystem::path(m_skyrimAvatarDataDirectory) / "Skyrim.esm")) {
            VOX_LOGE("showcase")
                << "balmora-skyrim-player requires Skyrim.esm; pass --skyrim-data "
                   "\"<Skyrim Special Edition/Data>\" or set ODAI_SKYRIM_DATA";
            return false;
        }
        m_streamPlugin = "Morrowind.esm";
        m_streamWorldspace = "Vvardenfell";
        m_streamWorldspaceExplicit = true;
        m_streamSpawnInterior.clear();
        m_startInsideInterior.clear();
        m_resumeEnabled = false;
        m_walkMode = true;
        m_thirdPersonView = true;
    }
    if (m_whiterunThirdPersonShowcase) {
        if (m_streamDirectory.empty()) {
            if (const char* configured = std::getenv("ODAI_SKYRIM_DATA")) {
                m_streamDirectory = configured;
            } else {
                m_streamDirectory = findSkyrimDataDirectory();
            }
        }
        if (m_skyrimAvatarDataDirectory.empty()) {
            m_skyrimAvatarDataDirectory = m_streamDirectory;
        }
        if (m_streamDirectory.empty() ||
            !std::filesystem::is_regular_file(
                std::filesystem::path(m_streamDirectory) / "Skyrim.esm")) {
            VOX_LOGE("showcase")
                << "whiterun-third-person requires Skyrim.esm; pass --skyrim-data "
                   "\"<Skyrim Special Edition/Data>\" or set ODAI_SKYRIM_DATA";
            return false;
        }
        m_streamPlugin = "Skyrim.esm";
        m_streamWorldspace = "WhiterunWorld";
        m_streamWorldspaceExplicit = true;
        m_streamSpawnInterior.clear();
        m_startInsideInterior.clear();
        m_resumeEnabled = false;
        m_walkMode = true;
        m_thirdPersonView = true;
    }
    if (m_whiterunReferenceShowcase) {
        if (m_streamDirectory.empty()) {
            if (const char* configured = std::getenv("ODAI_SKYRIM_DATA")) {
                m_streamDirectory = configured;
            } else {
                m_streamDirectory = findSkyrimDataDirectory();
            }
        }
        if (m_streamDirectory.empty() ||
            !std::filesystem::is_regular_file(
                std::filesystem::path(m_streamDirectory) / "Skyrim.esm")) {
            VOX_LOGE("showcase")
                << "whiterun-reference requires Skyrim.esm; pass --skyrim-data "
                   "\"<Skyrim Special Edition/Data>\" or set ODAI_SKYRIM_DATA";
            return false;
        }
        m_streamPlugin = "Skyrim.esm";
        m_streamWorldspace = "WhiterunWorld";
        m_streamWorldspaceExplicit = true;
        m_streamSpawnInterior.clear();
        m_startInsideInterior.clear();
        m_resumeEnabled = false;
        m_walkMode = false;
        m_thirdPersonView = false;
        m_mouseCaptured = false;
    }
    if (m_riftenThirdPersonShowcase) {
        if (m_streamDirectory.empty()) {
            if (const char* configured = std::getenv("ODAI_SKYRIM_DATA")) {
                m_streamDirectory = configured;
            } else {
                m_streamDirectory = findSkyrimDataDirectory();
            }
        }
        if (m_skyrimAvatarDataDirectory.empty()) {
            m_skyrimAvatarDataDirectory = m_streamDirectory;
        }
        if (m_streamDirectory.empty() ||
            !std::filesystem::is_regular_file(
                std::filesystem::path(m_streamDirectory) / "Skyrim.esm")) {
            VOX_LOGE("showcase")
                << "riften-third-person requires Skyrim.esm; pass --skyrim-data "
                   "\"<Skyrim Special Edition/Data>\" or set ODAI_SKYRIM_DATA";
            return false;
        }
        m_streamPlugin = "Skyrim.esm";
        m_streamWorldspace = "RiftenWorld";
        m_streamWorldspaceExplicit = true;
        m_streamSpawnInterior.clear();
        m_startInsideInterior.clear();
        m_resumeEnabled = false;
        m_walkMode = true;
        m_thirdPersonView = true;
    }
    if (!m_scenarioId.empty() && bethesda::findScenario(m_scenarioId) == nullptr) {
        VOX_LOGE("scenario") << "unknown scenario '" << m_scenarioId
                              << "' (available: skyrim-bleak-falls, "
                                 "skyrim-whiterun-showcase, skyrim-riften-showcase)";
        return false;
    }
    // Without this the font atlas is empty, so every addText() emits zero
    // quads and GameApp::drawPerfOverlay bails outright — the HUD and F3 both
    // render nothing, silently, with no error anywhere.
    // TV-sized type. The defaults (18 px body) are a desk-monitor scale: at a
    // couch viewing distance that is roughly half the angular size a console UI
    // needs, and it is why the first pass of this HUD was unreadable on a TV.
    // 28 px body / 76 px display is the 10-foot scale, and contentScale() still
    // multiplies on top for high-DPI panels.
    //
    // Inter is the Helvetica-like grotesque already vendored here, so the
    // display face costs no new asset.
    constexpr float kTvBodySize = 28.0f;
    constexpr float kTvNumericSize = 26.0f;
    constexpr float kTvCaptionSize = 22.0f;
    constexpr float kTvDisplaySize = 76.0f;  // the discovery banner
    if (!loadFonts(
            resolveAssetPath("assets/fonts/Inter-Regular.ttf"),
            resolveAssetPath("assets/fonts/Inter-Bold.ttf"),
            resolveAssetPath("assets/fonts/Inter-Italic.ttf"),
            resolveAssetPath("assets/fonts/Inter-Regular.ttf"),
            kTvBodySize, kTvNumericSize, kTvCaptionSize, kTvDisplaySize)) {
        VOX_LOGE("newvegas") << "failed to load UI fonts";
        return false;
    }

    // Conversation type. Baked here rather than through loadFonts because
    // GameApp's four slots (body/numeric/caption/display) are a shared contract
    // every game uses, and widening it for one game's dialogue would change all
    // of them. registerUiFontAtlas is public for exactly this.
    //
    // 48/40 px against a 28 px body: a reply the player has to READ and CHOOSE
    // from across a room is not the same reading task as a status strip, and
    // the previous pass drew both at body size in a corner.
    constexpr float kDialogueLineSize = 48.0f;
    constexpr float kDialogueChoiceSize = 40.0f;
    const float uiDensity = std::clamp(contentScale(), 1.0f, 4.0f);
    std::uint32_t uiFontAtlasSize = 1024u;
    while (static_cast<float>(uiFontAtlasSize) < 1024.0f * uiDensity) {
        uiFontAtlasSize *= 2u;
    }
    const std::string regularFontPath = resolveAssetPath("assets/fonts/Inter-Regular.ttf");
    if (m_dialogueFont.loadFromFile(
            regularFontPath, kDialogueLineSize * uiDensity, uiFontAtlasSize)) {
        m_dialogueFont.setTextureId(m_renderer.registerUiFontAtlas(
            m_dialogueFont.atlasPixels().data(), m_dialogueFont.atlasWidth(),
            m_dialogueFont.atlasHeight()));
    }
    if (m_dialogueChoiceFont.loadFromFile(
            regularFontPath, kDialogueChoiceSize * uiDensity, uiFontAtlasSize)) {
        m_dialogueChoiceFont.setTextureId(m_renderer.registerUiFontAtlas(
            m_dialogueChoiceFont.atlasPixels().data(), m_dialogueChoiceFont.atlasWidth(),
            m_dialogueChoiceFont.atlasHeight()));
    }
    // Prefer a packaged serif family when one is present. Linux development
    // builds also use the metrically compatible Liberation Serif installed by
    // the base image; every face falls back independently so a missing system
    // font can never make the journal disappear.
    const auto journalFontPath = [&](std::string_view packaged, std::string_view system) {
        const std::string asset = resolveAssetPath(std::string(packaged));
        std::error_code fontError;
        if (std::filesystem::exists(asset, fontError) && !fontError) return asset;
        return std::string(system);
    };
    const auto loadJournalFace = [&](ui::Font& font, const std::string& path, float size) {
        // Load from memory because a system font directory is intentionally
        // read-only; Font::loadFromFile otherwise tries to place its optional
        // atlas cache beside the TTF and emits a misleading startup warning.
        std::ifstream input(path, std::ios::binary | std::ios::ate);
        if (!input) return false;
        const std::streamoff byteCount = input.tellg();
        if (byteCount <= 0) return false;
        input.seekg(0, std::ios::beg);
        std::vector<std::uint8_t> bytes(static_cast<std::size_t>(byteCount));
        if (!input.read(reinterpret_cast<char*>(bytes.data()), byteCount) ||
            !font.loadFromMemory(bytes.data(), bytes.size(), size * uiDensity,
                                 uiFontAtlasSize)) return false;
        font.setTextureId(m_renderer.registerUiFontAtlas(
            font.atlasPixels().data(), font.atlasWidth(), font.atlasHeight()));
        return true;
    };
    constexpr float kJournalTextSize = 34.0f;
    const bool journalRegular = loadJournalFace(
        m_tes3JournalFont,
        journalFontPath("assets/fonts/LiberationSerif-Regular.ttf",
                        "/usr/share/fonts/liberation-serif-fonts/LiberationSerif-Regular.ttf"),
        kJournalTextSize);
    const bool journalBold = loadJournalFace(
        m_tes3JournalBoldFont,
        journalFontPath("assets/fonts/LiberationSerif-Bold.ttf",
                        "/usr/share/fonts/liberation-serif-fonts/LiberationSerif-Bold.ttf"),
        kJournalTextSize);
    const bool journalItalic = loadJournalFace(
        m_tes3JournalItalicFont,
        journalFontPath("assets/fonts/LiberationSerif-Italic.ttf",
                        "/usr/share/fonts/liberation-serif-fonts/LiberationSerif-Italic.ttf"),
        kJournalTextSize);
    const ui::Font* journalRegularFace = journalRegular ? &m_tes3JournalFont : &m_uiFont;
    const ui::Font* journalBoldFace = journalBold ? &m_tes3JournalBoldFont
        : (m_uiFontBold.valid() ? &m_uiFontBold : journalRegularFace);
    const ui::Font* journalItalicFace = journalItalic ? &m_tes3JournalItalicFont
        : (m_uiFontItalic.valid() ? &m_uiFontItalic : journalRegularFace);
    m_tes3JournalPanel = std::make_unique<ui::Tes3JournalPanel>(ui::FontSet{
        journalRegularFace, journalBoldFace, journalItalicFace,
        journalBoldFace, &m_uiFontNumeric});

    if (m_streamDirectory.empty()) {
        if (const char* fromEnv = std::getenv("ODAI_FNV_STREAM_DIR")) {
            m_streamDirectory = fromEnv;
        }
    }
    if (!resolveConfiguredContentProfile()) {
        return false;
    }
    if (m_streamDirectory.empty() && m_scenePath.empty() &&
        std::getenv("ODAI_FNV_SCENE") == nullptr) {
        // Nothing specified at all: look for an installed copy of the game and
        // stream from it. Streaming needs no cooked assets, so a bare launch now
        // has a sensible thing to do -- which it did not when a cooked scene was
        // the only possible source.
        m_streamDirectory = m_scenarioId.empty()
            ? findFalloutDataDirectory()
            : findSkyrimDataDirectory();
        if (!m_streamDirectory.empty()) {
            VOX_LOGI("newvegas") << "found Bethesda data at " << m_streamDirectory;
        }
    }
    // NOTE: streaming init happens further down, AFTER the renderer pass-stack
    // configuration. Returning here instead left streaming running with ray
    // tracing, voxel GI and sun shafts all still enabled -- which showed up as a
    // BLAS/TLAS rebuild on every single streamed cell.
    // Character mode STREAMS THE WORLD TOO, and that turned out to be
    // load-bearing rather than cosmetic. With streaming off nothing calls
    // uploadImportedScene, and a frame with no imported geometry renders no
    // sky, no ground and no skinned actor -- a flat clear-colour screen that
    // looks exactly like a failed character upload. Standing the character in
    // Goodsprings costs a few seconds of streaming and makes the view both
    // correct and legible: a body at Fallout's own scale, next to Fallout's own
    // buildings, is the only way to see that the scale is right.
    const bool streamingMode = !m_streamDirectory.empty();

    if (!streamingMode && !m_characterMode && m_scenePath.empty()) {
        if (const char* fromEnv = std::getenv("ODAI_FNV_SCENE")) {
            m_scenePath = fromEnv;
        }
    }
    if (m_characterMode && m_streamDirectory.empty()) {
        VOX_LOGE("newvegas")
            << "--character needs the game's Data directory (for the skeleton and "
               "body meshes); none was found. Pass --stream \"<.../Fallout New Vegas/Data>\".";
        return false;
    }
    if (!streamingMode && !m_characterMode && m_scenePath.empty()) {
        VOX_LOGE("newvegas")
            << "no Fallout: New Vegas install found, and no scene given.\n"
               "  Stream from the game (no cooking): --stream \"<.../Fallout New Vegas/Data>\"\n"
               "  Load a cooked scene:               --scene <path.bin>\n"
               "  Or set ODAI_FNV_STREAM_DIR / ODAI_FNV_SCENE.";
        return false;
    }
    // Local, not a member: uploadImportedScene deep-copies the whole scene, so
    // keeping a second copy alive for the process lifetime costs ~100 MB of
    // resident memory that nothing ever reads again.
    //
    // The full loader, NOT loadImportedSceneRuntime. The runtime one keeps only
    // the packed stream: it skips the mesh block outright and reads instances
    // just to discard them. Both are needed here and neither failure is visible
    // -- the containers come back empty rather than erroring -- so with the
    // runtime loader the ground height field is never built (camera stays in fly
    // mode) and the town centroid finds nothing (spawn falls back to the middle
    // of the map). This costs the mesh + instance arrays for the duration of
    // onInit, which is the price of knowing where the ground and the town are.
    if (!streamingMode && !m_characterMode) {
        m_sceneDirectory = std::filesystem::path(m_scenePath).parent_path();
        m_exteriorStem = std::filesystem::path(m_scenePath).stem().string();
        if (!loadScene(std::filesystem::path(m_scenePath), nullptr, nullptr)) {
            return false;
        }
    }
    // Sun plus cascaded shadow maps, and nothing the original game didn't have.
    // Fallout: New Vegas lit its world with a directional sun, shadow maps and
    // baked ambient — no global illumination, no ray tracing, no screen-space AO,
    // no sun shafts. Matching that is both the look we want and, on an integrated
    // GPU, the difference between a playable frame rate and a driver hang-check
    // reset (VK_ERROR_DEVICE_LOST) on the very first frame.
    m_renderer.setShadowSettings(render::ShadowSettings{render::ShadowMode::Auto});

    const std::string shaderPackPreset = std::getenv("ODAI_FNV_SHADER_PACK") != nullptr
        ? std::getenv("ODAI_FNV_SHADER_PACK")
        : "";
    const bool rafaelShaderPack = shaderPackPreset == "rafael";
    if (rafaelShaderPack) {
        // Clean-room mapping of Enhanced PBR Lighting's documented defaults
        // onto this renderer's native GGX path. TES3 NIFs carry no authored
        // metallic/roughness data, so this fills only that legacy case; named
        // engine materials and newer-game PBR data remain authoritative.
        render::ImportedPbrDefaults pbrDefaults;
        pbrDefaults.enabled = true;
        if (const char* value = std::getenv("ODAI_FNV_PBR_OBJECT_ROUGHNESS")) {
            pbrDefaults.objectRoughness = static_cast<float>(std::atof(value));
        }
        if (const char* value = std::getenv("ODAI_FNV_PBR_TERRAIN_ROUGHNESS")) {
            pbrDefaults.terrainRoughness = static_cast<float>(std::atof(value));
        }
        if (const char* value = std::getenv("ODAI_FNV_PBR_METALLIC")) {
            pbrDefaults.metallic = static_cast<float>(std::atof(value));
        }
        m_renderer.setImportedPbrDefaults(pbrDefaults);
        VOX_LOGI("newvegas")
            << "shader pack preset: rafael (native GGX PBR defaults, XeGTAO/TAA, "
               "bundled/override water normal; object roughness=" << pbrDefaults.objectRoughness
            << ", terrain roughness=" << pbrDefaults.terrainRoughness
            << ", metallic=" << pbrDefaults.metallic << ")";
    } else if (!shaderPackPreset.empty()) {
        VOX_LOGW("newvegas") << "unknown ODAI_FNV_SHADER_PACK=" << shaderPackPreset
                              << "; rendering with normal settings";
    }

    // Voxel GI contributes nothing here anyway: the grid is 64 world units wide
    // and camera-following, which at Bethesda scale (~70 units/metre) is under a
    // metre across inside a scene spanning tens of thousands of units, so
    // sampleImportedVoxelGi lands outside the volume and returns black for
    // essentially every pixel. Without this the whole ReSTIR sequence — candidate,
    // temporal, spatial, resolve, all traced against the TLAS — ran every frame
    // for a contribution that was already invisible.
    // No TLAS to trace against once GI is off, so stop building acceleration
    // structures on every uploadImportedScene too. Interior shadows default to
    // the cached point-shadow atlas and need no acceleration structure. The RT
    // A/B mode below keeps the runtime alive for the same fixed tour when
    // explicitly requested.
    //
    // ODAI_FNV_RT=1 keeps the RT runtime alive anyway. GI is not the only
    // possible TLAS consumer any more: ray-traced sun shadows want one too, and
    // the shading side for those is already written and compiled
    // (sampleRayTracedDirectionalShadow / imported_static_rt.frag.slang.spv).
    // This line is the ONLY thing standing between the streaming path and a
    // TLAS -- the BLAS record block in uploadImportedSceneInternal is not gated
    // on the chunk path at all, it is gated on rayTracingRuntimeReady().
    //
    const char* interiorShadowModeEnv = std::getenv("ODAI_FNV_INTERIOR_SHADOWS");
    const bool rayTracedInteriorShadows =
        interiorShadowModeEnv != nullptr && std::strcmp(interiorShadowModeEnv, "rt") == 0;
    const bool rayTracingRequested =
        std::getenv("ODAI_FNV_RT") != nullptr || rayTracedInteriorShadows;
    m_renderer.setRayTracingEnabled(rayTracingRequested);
    if (rayTracingRequested) {
        VOX_LOGI("newvegas") << "ray tracing runtime left enabled "
                                "(acceleration structures will build per scene upload)";
    }
    // Volumetric sun shafts. sun_shafts.comp.slang is a real single-scattering
    // raymarch -- height-falloff density, Henyey-Greenstein phase, shadow-map
    // visibility sampled per step -- so this is the atmosphere pass, not a
    // radial blur.
    //
    // OFF by default, because it is currently paying for nothing: skyConfig4's
    // density/falloff/scatter are near zero for this game, so the effect is
    // invisible while the pass costs 4.7 ms of a 37.9 ms frame at 2560x1440 on
    // the LNL iGPU -- measured by toggling it, not by the GPU timer, which
    // attributes only ~3 ms to the dispatch itself. ODAI_FNV_SHAFTS=1 turns it
    // back on, which is what to do FIRST when tuning those density values: the
    // pass is worth its cost only once they are non-trivial, and this default
    // should flip back the moment they are.
    const bool sunShaftsRequested = [] {
        const char* env = std::getenv("ODAI_FNV_SHAFTS");
        return env != nullptr && env[0] != '0';
    }();
    m_renderer.setSunShaftsEnabled(sunShaftsRequested);
    // Temporal AA. This is what stops textured surfaces shimmering in motion
    // -- measured at 13x the frame-to-frame instability of flat-shaded
    // geometry before TAA existed. ODAI_TAA=0 turns it off for A/B.
    {
        const char* taaEnv = std::getenv("ODAI_TAA");
        const bool taaEnabled = taaEnv == nullptr || taaEnv[0] != '0';
        m_renderer.setTaaEnabled(taaEnabled);
    }

    // Ambient occlusion, tuned for Bethesda scale.
    //
    // The radius is NOT a taste call. The GTAO march takes six steps
    // across a screen-space radius of roughly `radius * 9297 / depth` pixels, so
    // a 7-unit radius collapses to sub-pixel steps beyond ~1500 units and the
    // estimator early-outs to "unoccluded" for the entire frame -- AO that costs
    // its full dispatch and produces nothing. 128 is the shader's own clamp
    // ceiling in ssao.comp.slang and lands at ~1.8 m, which is the scale of the
    // contact darkening this world wants.
    //
    // XeGTAO rather than GTAO, and it is cheaper AS WELL AS cleaner -- which is
    // the opposite of what "it runs three dispatches instead of one" suggests.
    //
    // GTAO marches a fixed sample count per pixel and its only smoothing is the
    // bilateral UPSAMPLE, which exists to reconstruct resolution rather than to
    // denoise; its sample pattern therefore survives into the frame as a
    // stipple, worst on terrain and on alpha-tested foliage where neighbouring
    // samples disagree about depth. XeGTAO marches a prefiltered depth pyramid
    // with adaptive sample counts and then runs a real edge-aware denoise.
    //
    // Measured on Seyda Neen at a pinned camera, as high-frequency energy in the
    // AO channel alone (ODAI_FNV_DEBUGVIEW=ao, mean |pixel - 3x3 mean|, sky
    // excluded), against interleaved A/B/A/B GPU timings:
    //
    //   GTAO            noise 3.95   ssao pass 1.33 ms   frame ~12.9 ms
    //   XeGTAO          noise 1.72   ssao pass 0.43 ms   frame ~11.6 ms
    //   HBAO            noise 2.63
    //   SSAO            noise 0.97   (least noise, crudest estimator)
    //   GTAO full-res   noise 2.70   and ~4.4 ms dearer
    //
    // So 56% less AO noise for 0.9 ms LESS on the pass. The adaptive sample
    // count is why: fewer samples where the term is already smooth.
    //
    // ODAI_XEGTAO_BLUR only affects this mode, which is worth knowing before
    // tuning it -- raising it from 4 to 16 under GTAO changes nothing at all,
    // because GTAO never reaches the XeGTAO denoise.
    //
    // Falls back to the GTAO pipeline on its own if the XeGTAO pipelines or
    // buffer sets are unavailable (see useXeGtao in frame_pass_ssao.cc).
    //
    // ODAI_FNV_AO overrides the mode (off/ssao/hbao/gtao/xegtao) for A/B.
    render::AoMode aoMode = render::AoMode::Xegtao;
    if (const char* aoEnv = std::getenv("ODAI_FNV_AO")) {
        const std::string requested = aoEnv;
        if (requested == "off") {
            aoMode = render::AoMode::Off;
        } else if (requested == "ssao") {
            aoMode = render::AoMode::Ssao;
        } else if (requested == "hbao") {
            aoMode = render::AoMode::Hbao;
        } else if (requested == "gtao") {
            aoMode = render::AoMode::Gtao;
        } else if (requested == "xegtao") {
            aoMode = render::AoMode::Xegtao;
        }
    }
    m_renderer.setSsaoEnabled(aoMode != render::AoMode::Off);
    m_renderer.setAmbientOcclusionMode(aoMode);
    // ODAI_FNV_DEBUGVIEW selects a whole-frame debug visualization by name (see
    // DebugView in renderer_types.h). It exists because a --screenshot run
    // cannot operate the ImGui combo -- F4 is the interactive way in, and this
    // is the only way to photograph a debug view from a script, which is what
    // makes an alpha or material-flags capture attributable in a bug report.
    if (const char* debugViewEnv = std::getenv("ODAI_FNV_DEBUGVIEW")) {
        const std::string requested = debugViewEnv;
        render::DebugView view = render::DebugView::Off;
        if (requested == "albedo") { view = render::DebugView::Albedo; }
        else if (requested == "normal") { view = render::DebugView::Normal; }
        else if (requested == "alpha") { view = render::DebugView::Alpha; }
        else if (requested == "flags") { view = render::DebugView::MaterialFlags; }
        else if (requested == "roughness") { view = render::DebugView::Roughness; }
        else if (requested == "metallic") { view = render::DebugView::Metallic; }
        else if (requested == "mip") { view = render::DebugView::MipLevel; }
        else if (requested == "cascade") { view = render::DebugView::CascadeIndex; }
        else if (requested == "texid") { view = render::DebugView::TextureId; }
        else if (requested == "depth") { view = render::DebugView::LinearDepth; }
        else if (requested == "shadow") { view = render::DebugView::Shadow; }
        else if (requested == "directratio") { view = render::DebugView::DirectRatio; }
        else if (requested == "terrainlayers") { view = render::DebugView::TerrainLayers; }
        else if (requested == "ao") { view = render::DebugView::AmbientOcclusion; }
        else if (requested == "ssgi") { view = render::DebugView::ScreenSpaceGi; }
        else if (requested != "off") {
            VOX_LOGW("newvegas")
                << "ODAI_FNV_DEBUGVIEW=" << requested << " is not a view name; ignoring. "
                << "Valid: albedo normal alpha flags roughness metallic mip cascade texid "
                << "depth shadow directratio terrainlayers ao ssgi\n";
        }
        m_renderer.setDebugView(view);
    }
    // ODAI_FNV_DRAW=terrain|statics|actors splits the imported draw list the way the
    // F4 panel's checkboxes do, for the same reason the debug views have an env
    // var: a --screenshot run cannot operate ImGui, and "is this artifact
    // terrain or a static" is unanswerable from a lit frame when the two draw
    // on top of each other.
    if (const char* drawEnv = std::getenv("ODAI_FNV_DRAW")) {
        const std::string requested = drawEnv;
        const bool actorsOnly = requested == "actors";
        const bool showTerrain = !actorsOnly && requested != "statics";
        const bool showStatics = !actorsOnly && requested != "terrain";
        m_renderer.setImportedSceneDebugState(
            showTerrain, showStatics, /*showTextures=*/true, /*flatShading=*/false,
            /*waterDebug=*/false);
        VOX_LOGI("newvegas") << "imported draws restricted to " << requested;
    }
    // Sweepable, because "too subtle" is a measurable claim: the A/B against
    // AO-off below is what says whether a value actually changed the image.
    //
    // NOTE the intensity is an EXPONENT: sampleSsaoAmbientFactor computes
    // pow(ssaoRaw, intensity) on a value in [0,1]. Anything below 1 pushes the
    // result toward 1, i.e. actively weakens the occlusion -- which is what the
    // inherited 0.85 was doing.
    // 300 units, about 4.3 m at Bethesda's ~70 units per metre.
    //
    // This was briefly dropped to 150 to kill a muddy cast and a dark silhouette
    // fringe on Seyda Neen's shacks, and that was treating the estimator through
    // the radius: the fringe was GTAO's sample pattern, and switching the default
    // to XeGTAO removed it at the source. Re-measured afterwards on the same
    // pinned camera, as high-frequency energy in the AO channel alone
    // (ODAI_FNV_DEBUGVIEW=ao) and as how much of the frame is meaningfully
    // occluded:
    //
    //   radius   noise   occlusion   px below 200
    //     100    1.694     17.40        5.4%
    //     150    1.722     17.76        6.3%
    //     300    1.914     21.27       10.2%
    //     450    2.164     25.51       14.2%
    //
    // 300 costs 11% more AO noise than 150 and buys 60% more occluded frame, and
    // in the LIT frame -- where AO modulates ambient, which is a fraction of the
    // lighting -- the extra noise does not survive while the extra contact
    // darkening does. For scale, GTAO at radius 150 measured 3.95 on this
    // camera: XeGTAO at 300 is half as noisy as the value this replaced.
    //
    // Note a stale claim removed with it: the comment here used to say 128 was
    // "the shader's own clamp ceiling in ssao.comp.slang", which would have made
    // every value above it identical. Both estimators clamp to [0.25, 512]
    // (ssao.comp.slang and frame_pass_ssao.cc), so the sweep above is real.
    float aoRadius = 300.0f;
    float aoBias = 40.0f;
    float aoIntensity = 1.7f;
    if (const char* env = std::getenv("ODAI_FNV_AO_RADIUS")) {
        aoRadius = static_cast<float>(std::atof(env));
    }
    if (const char* env = std::getenv("ODAI_FNV_AO_BIAS")) {
        aoBias = static_cast<float>(std::atof(env));
    }
    if (const char* env = std::getenv("ODAI_FNV_AO_INTENSITY")) {
        aoIntensity = static_cast<float>(std::atof(env));
    }
    m_renderer.setAmbientOcclusionTuning(aoRadius, aoBias, aoIntensity);

    // Multi-scale: the coarse march reaches well past contact range, the fine
    // one at ~22% of it catches where objects meet the ground. One radius
    // cannot do both -- the march has a fixed step count, so widening it just
    // spreads the same samples further apart.
    float aoFineScale = 0.22f;
    if (const char* env = std::getenv("ODAI_FNV_AO_FINE")) {
        aoFineScale = static_cast<float>(std::atof(env));
    }
    m_renderer.setAmbientOcclusionFineScale(aoFineScale);

    // Eye adaptation. Without it the renderer holds a fixed exposure, and the
    // Mojave at noon came out around 46/255 -- textured, detailed, and far too
    // dark to read, with shadowed ground going to black because AO, GI and sun
    // shafts are all deliberately off here and nothing fills them. It also gives
    // the day/night cycle somewhere to go: at a fixed exposure midnight rendered
    // pure black rather than moonlit.
    // Neutral colour grade. The post chain's defaults are a stylized look and
    // are applied with no enable gate, so this viewer inherited +8% saturation,
    // +12% vibrance, +10% contrast and an 8% blue cut on top of the tonemap.
    // Measured on a Goodsprings frame that put mean pixel saturation at 0.43
    // with a p90 of 0.80 -- a vivid image of a landscape that is meant to read
    // as dust and sun-bleached tan.
    //
    // ODAI_FNV_COLOR_LOOK=stylized restores the defaults. There was no runtime
    // knob for any of this, which is why the report of "oversaturated" had to
    // be answered by reading the shader instead of an A/B.
    std::string colorLook =
        std::getenv("ODAI_FNV_COLOR_LOOK") != nullptr ? std::getenv("ODAI_FNV_COLOR_LOOK") : "";
    if (colorLook.empty() && rafaelShaderPack) {
        // Rafael's package includes its own tonemap; use the renderer's restrained
        // cinematic equivalent unless the caller selected a grade explicitly.
        colorLook = "cinematic";
    }
    if (colorLook == "cinematic") {
        // A measured middle between the neutral grade above and the engine's
        // stylized default, for the landscape flythroughs. Every number here
        // came off a pinned West Gash frame on the Seyda Neen -> Balmora route,
        // and the two it avoids are as informative as the ones it uses:
        //
        //   neutral    sd 0.188  p1 0.135  p99 0.643  sat 0.231
        //   stylized   sd 0.260  p1 0.000  p99 0.638  sat 0.397
        //
        // Stylized buys its contrast by CRUSHING THE SHADOWS TO ZERO, and on a
        // scene whose depth is carried by aerial perspective that deletes the
        // depth cue -- the far ridge and the near rock end up the same black.
        // So shadowDensity stays at 1.0 here and the contrast comes from the
        // midtones and from the white point instead, which lifts the top of the
        // histogram rather than pushing down the bottom.
        //
        // Vibrance is kept well under the stylized 0.12 because it targets the
        // LEAST saturated pixels, and in fog that is most of the frame.
        render::ColorGradingSettings grade;
        grade.midtoneContrast = 1.12f;
        grade.saturation = 1.10f;
        grade.vibrance = 0.05f;
        grade.shadowDensity = 1.0f;
        // Cool the shadows and warm the highlights very slightly: the classic
        // teal/amber split, at a fraction of the usual strength because
        // Morrowind's own palette is already blue-green.
        grade.shadowTint[2] = 0.03f;
        grade.highlightTint[0] = 0.03f;
        m_renderer.setColorGrading(grade);

        // The look INCLUDES its white point, because the grade alone does not
        // fix what is wrong with these frames. Grading on its own measured
        // sd 0.1845 / p99 0.634 -- i.e. it moved saturation and essentially
        // nothing else, since there was no highlight range for contrast to act
        // on. The white point is what supplies that range; the two together are
        // the look. ODAI_FNV_WHITEPOINT still overrides, in applyTonemapSettings.
        render::TonemapSettings tonemap = m_renderer.tonemapSettings();
        // Rafael's own defaults call for a 1.0 linear white and a 0.45 exterior
        // shoulder. The controls are not mathematically identical across the
        // two tone curves, but preserving their anchors avoids the clipped,
        // chalk-white roofs a generic cinematic white point produced here.
        tonemap.whitePoint = rafaelShaderPack ? 1.0f : 0.8f;
        tonemap.highlightShoulder = rafaelShaderPack ? 0.45f : 1.0f;
        m_renderer.setTonemapSettings(tonemap);
    } else if (colorLook != "stylized") {
        m_renderer.setNeutralColorGrading();
    }

    m_renderer.setAutoExposureEnabled(true);
    if (const char* exposureKey = std::getenv("ODAI_FNV_EXPOSURE_KEY")) {
        m_renderer.setAutoExposureKeyValue(
            static_cast<float>(std::atof(exposureKey)));
    }

    // Diagnostic A/B: ODAI_FNV_NOTEX forces every imported surface to shade from
    // its vertex colour instead of its texture. Comparing a capture with and
    // without it answers a question that is otherwise guesswork -- whether a
    // washed-out surface is showing a pale TEXTURE or is falling back to vertex
    // colour and being blown out by lighting. The two look identical on screen.
    if (const char* noTextures = std::getenv("ODAI_FNV_NOTEX")) {
        if (noTextures[0] != '\0' && noTextures[0] != '0') {
            m_renderer.setImportedSceneDebugState(true, true, false, false, false);
            VOX_LOGI("newvegas") << "ODAI_FNV_NOTEX: imported textures disabled (vertex colour only)";
        }
    }

    // Start hour override. Lighting bugs and "it's just a dim hour" look the
    // same in a single capture; being able to shoot the same view at several
    // times of day separates them.
    if (!m_timeOfDayExplicit) {
      if (const char* hourEnv = std::getenv("ODAI_FNV_HOUR")) {
        const float hour = static_cast<float>(std::atof(hourEnv));
        if (hour >= 0.0f && hour < 24.0f) {
            m_timeOfDayHours = hour;
        }
      }
    }
    applyTimeOfDay();

    // Last, so the streamer inherits the pass-stack configuration above rather
    // than paying for ray tracing and voxel GI on every streamed cell.
    if (streamingMode && !initStreaming()) {
        return false;
    }
    if ((!m_scenarioId.empty() || m_streamIsMorrowind) && !initBethesdaSession()) {
        return false;
    }
    if (skyrimCityShowcase() && !prewarmSkyrimCityShowcase()) {
        return false;
    }
    // Skyrim's exterior art was authored around a restrained, contrasty
    // display curve. Leaving it on the shared neutral viewer grade makes the
    // pale WTHR horizon occupy most of the midtone range, so stone, timber and
    // distant architecture all converge on the same blue-grey value. Apply a
    // small Skyrim default only after initStreaming() has identified the file
    // format. An explicit color-look request remains authoritative.
    if (m_streamIsSkyrim && std::getenv("ODAI_FNV_COLOR_LOOK") == nullptr) {
        render::ColorGradingSettings grade;
        grade.whiteBalance[0] = 1.02f;
        grade.whiteBalance[2] = 0.94f;
        grade.contrast = 1.10f;
        grade.midtoneContrast = 1.22f;
        grade.saturation = 0.92f;
        grade.vibrance = 0.02f;
        grade.shadowDensity = 0.92f;
        m_renderer.setColorGrading(grade);
        VOX_LOGI("newvegas") << "color look: Skyrim restrained";
    }
    if (m_characterMode && !initCharacter(m_streamDirectory)) {
        return false;
    }

    // After streaming init, so a failed stream never leaves weather half-set,
    // and after applyTimeOfDay so the first push uses the real hour.
    if (const char* pluginsEnv = std::getenv("ODAI_FNV_PLUGINS")) {
        const std::string plugins = pluginsEnv;
        std::size_t start = 0;
        while (start <= plugins.size()) {
            const std::size_t end = plugins.find(',', start);
            const std::string entry =
                plugins.substr(start, end == std::string::npos ? std::string::npos : end - start);
            if (!entry.empty()) {
                m_extraPlugins.push_back(entry);
            }
            if (end == std::string::npos) {
                break;
            }
            start = end + 1;
        }
    }
    if (const char* weatherEnv = std::getenv("ODAI_FNV_WEATHER")) {
        m_requestedWeatherEditorId = weatherEnv;
    }
    // An interior publishes its own XCLL and CELL sky policy during
    // initStreaming. Applying the exterior climate afterwards used to replace
    // both with SkyrimCloudy, which is why Dragonsreach looked sunlit and blue
    // through openings despite authoring black fog and zero directional light.
    if (!m_interiorStarted) {
        initWeather();
    }
    applyTonemapSettings();

    // Pip-Boy palette for notifications, matching the HUD chrome.
    ui::ToastStyle toastStyle{};
    toastStyle.widthPx = 300.0f;
    m_toasts.setStyle(toastStyle);
    ui::ToastTiming toastTiming{};
    toastTiming.holdSeconds = 4.5f;
    m_toasts.setTiming(toastTiming);
    m_toasts.setMaxVisible(3);

    // Discovery banner: centred, chrome-free, slow fade.
    m_banner.setStyle(ui::makeBannerStyle());
    m_banner.setTiming(ui::makeBannerTiming());
    m_banner.setMaxVisible(1);

    // ODAI_FNV_UI_DEMO=1 opens the menu and stacks sample toasts at startup.
    // The screenshot path cannot press buttons, so without this the menu and a
    // multi-toast stack are the two things that can only be checked by a human
    // with a controller in hand -- which is to say, not checked.
    if (std::getenv("ODAI_FNV_UI_DEMO") != nullptr) {
        // ODAI_FNV_UI_DEMO=menu opens the pause menu; anything else shows the
        // discovery banner. They are mutually exclusive at runtime (the banner
        // holds while the menu is up), so the demo cannot show both either.
        const std::string demoMode = std::getenv("ODAI_FNV_UI_DEMO");
        m_navDriving = true;  // show the focus highlight and the controller labels
        // "weather" opens the picker sub-page. It needs its own value because it
        // is two keypresses deep and a screenshot run cannot press either.
        m_menuOpen = (demoMode == "menu" || demoMode == "weather");
        if (demoMode == "weather") {
            openWeatherPicker();
        }
        if (!m_menuOpen && demoMode != "journal") {
            m_banner.push("Goodsprings", "Location discovered", "region:Goodsprings");
        }
        if (demoMode != "journal") {
            m_toasts.push("Stimpak", "Added to inventory");
            m_toasts.push("Quest updated", "Back in the Saddle");
        }
    }

    setMouseCaptured(true);
    return true;
}

bool BethesdaApp::initCharacter(const std::filesystem::path& dataFilesPath) {
    importer::fnv::FalloutAssetSource assets;
    if (!assets.open(dataFilesPath)) {
        VOX_LOGE("newvegas") << "could not index archives under " << dataFilesPath;
        return false;
    }

    std::string error;
    std::vector<std::uint8_t> bytes;
    if (!assets.resolveMesh(m_characterSkeletonPath, bytes, error)) {
        VOX_LOGE("newvegas") << "skeleton not found: " << m_characterSkeletonPath << " (" << error << ")";
        return false;
    }
    importer::fnv::NifSkeleton nifSkeleton;
    if (!importer::fnv::parseNifSkeleton(bytes, nifSkeleton, error)) {
        VOX_LOGE("newvegas") << "skeleton parse failed: " << error;
        return false;
    }
    if (!importer::fnv::buildFalloutSkeleton(nifSkeleton, m_character.skeleton)) {
        VOX_LOGE("newvegas") << "skeleton conversion failed";
        return false;
    }

    for (const std::string& partPath : m_characterPartPaths) {
        if (!assets.resolveMesh(partPath, bytes, error)) {
            VOX_LOGW("newvegas") << "body part not found: " << partPath << " (" << error << ")";
            continue;
        }
        importer::fnv::NifSkinnedModel model;
        if (!importer::fnv::parseNifSkinnedMesh(bytes, model, error)) {
            VOX_LOGW("newvegas") << "body part parse failed: " << partPath << " (" << error << ")";
            continue;
        }
        if (!importer::fnv::appendFalloutCharacterMesh(model, m_character, error)) {
            VOX_LOGW("newvegas") << "body part bind failed: " << partPath << " (" << error << ")";
        }
    }
    if (m_character.vertices.empty()) {
        VOX_LOGE("newvegas") << "no skinned geometry loaded";
        return false;
    }

    // One draw per part -- minus the gore caps. The draws index the merged
    // buffer, which is why appendFalloutCharacterMesh records
    // firstIndex/indexCount rather than leaving each part with its own arrays.
    //
    // A body NIF ships dismemberment geometry alongside the body: on
    // characters\_male\upperbody.nif, 3 of the 6 shapes ("limbcaps",
    // "meatneck01", "meathead01") are meat caps the game reveals only when a
    // limb comes off. They are skinned and they bind correctly -- their
    // measured bind-pose bounds are simply not on the standing body
    // ("limbcaps" sits at y -102..-18, well below the feet at 0.78) because
    // nothing positions them until a limb is severed.
    //
    // Drawing them makes an otherwise correct character look broken, and it was
    // the reason the first framed capture of this view looked like it had
    // failed. Excluded by texture because the proper discriminator --
    // BSDismemberSkinInstance's per-partition body-part IDs -- is the one part
    // of that block this importer deliberately does not read. All three use
    // textures\gore\MeatCapGore01.dds and no non-gore part does.
    // ODAI_FNV_CHAR_ALL=1 keeps the caps, for diagnosing which parts reach the
    // screen at all.
    const bool keepAllParts = std::getenv("ODAI_FNV_CHAR_ALL") != nullptr;
    const auto isGoreCap = [keepAllParts](const std::string& texturePath) {
        if (keepAllParts) {
            return false;
        }
        std::string lowered = texturePath;
        for (char& ch : lowered) {
            ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        }
        return lowered.find("\\gore\\") != std::string::npos;
    };
    m_characterDraws.clear();
    m_characterDraws.reserve(m_character.parts.size());
    std::vector<const importer::fnv::FalloutCharacterPart*> drawnParts;
    for (const auto& part : m_character.parts) {
        if (isGoreCap(part.diffuseTexturePath)) {
            VOX_LOGI("newvegas") << "  skipping gore cap \"" << part.name << "\"";
            continue;
        }
        importer::ImportedScenePackedDraw draw{};
        draw.firstIndex = part.firstIndex;
        draw.indexCount = part.indexCount;
        draw.alphaThreshold = part.alphaThreshold;
        m_characterDraws.push_back(draw);
        drawnParts.push_back(&part);
    }
    // ODAI_FNV_CHAR_NODRAW=1 uploads the template (so the skinning dispatch
    // still runs) but issues no draws. Paired with ODAI_FNV_SKIN_BYPASS, which
    // does the opposite, it separates "the compute pass corrupts the frame"
    // from "the geometry it produces does".
    if (std::getenv("ODAI_FNV_CHAR_NODRAW") != nullptr) {
        VOX_LOGW("newvegas") << "character draws SUPPRESSED: dispatch only";
        m_characterDraws.clear();
    } else if (m_characterDraws.empty()) {
        VOX_LOGE("newvegas") << "every part was filtered out; nothing to draw";
        return false;
    }

    importer::fnv::computeFalloutBindPose(m_character, m_characterBindPose);

    render::ImportedSkinnedMeshTemplate meshTemplate{};
    meshTemplate.vertices = m_character.vertices;
    meshTemplate.indices = m_character.indices;
    meshTemplate.draws = m_characterDraws;
    meshTemplate.boneCount = static_cast<std::uint32_t>(m_character.skeleton.bones.size());
    // ODAI_FNV_SKIN_BYPASS=1 skips the skinning dispatch, leaving the output
    // buffer at the rest pose the upload seeded it with. It is the one
    // diagnostic that separates "the vertex data I handed over is wrong" from
    // "the compute pass is not doing what I think": a clean figure under bypass
    // and an exploded one without it puts the fault squarely in the dispatch.
    if (std::getenv("ODAI_FNV_SKIN_BYPASS") != nullptr) {
        VOX_LOGW("newvegas") << "skinning dispatch BYPASSED: showing the rest pose";
        m_renderer.setSkinningDebugBypass(true);
    }
    // ODAI_FNV_CHAR_NOUPLOAD=1 does everything except hand the mesh to the GPU.
    // It answers the one question the other toggles cannot: whether the frame
    // breaks because of the skinned instance at all, or because of something
    // else this mode does.
    if (std::getenv("ODAI_FNV_CHAR_NOUPLOAD") != nullptr) {
        VOX_LOGW("newvegas") << "character GPU upload SKIPPED";
        m_characterBindPose.clear();
        return true;
    }
    if (!m_renderer.uploadSkinnedMeshTemplate(0u, meshTemplate)) {
        VOX_LOGE("newvegas") << "uploadSkinnedMeshTemplate failed";
        return false;
    }

    VOX_LOGI("newvegas") << "character: " << m_character.skeleton.bones.size() << " bones, "
                         << m_character.vertices.size() << " vertices, "
                         << (m_character.indices.size() / 3u) << " triangles, "
                         << m_character.parts.size() << " parts, "
                         << m_character.unresolvedBoneCount << " unresolved bones";

    // Frame the camera on the skinned bind-pose bounds rather than on a guessed
    // height. The character's extent is the only thing in the scene, and
    // guessing it wrong means an empty screen that looks exactly like a failed
    // upload -- which is the outcome this whole mode exists to rule out.
    float boundsMin[3] = {1e30f, 1e30f, 1e30f};
    float boundsMax[3] = {-1e30f, -1e30f, -1e30f};
    // Over the DRAWN parts only. Including the filtered gore caps here would
    // frame the camera on a body twice its real height and push the character
    // itself into the top half of the screen.
    for (const importer::fnv::FalloutCharacterPart* part : drawnParts) {
    for (std::uint32_t idx = part->firstIndex; idx < part->firstIndex + part->indexCount; ++idx) {
        const auto& vertex = m_character.vertices[m_character.indices[idx]];
        odai::math::Vector3 skinned{0.0f, 0.0f, 0.0f};
        const odai::math::Vector3 rest{vertex.position[0], vertex.position[1], vertex.position[2]};
        for (int k = 0; k < importer::fnv::kNifMaxBoneInfluences; ++k) {
            const float weight = vertex.boneWeights[k];
            if (weight <= 0.0f) {
                continue;
            }
            const std::size_t bone = vertex.boneIndices[k];
            if (bone >= m_characterBindPose.size()) {
                continue;
            }
            const odai::math::Vector3 contribution =
                odai::math::transformPoint(m_characterBindPose[bone], rest);
            skinned.x += contribution.x * weight;
            skinned.y += contribution.y * weight;
            skinned.z += contribution.z * weight;
        }
        const float values[3] = {skinned.x, skinned.y, skinned.z};
        for (int a = 0; a < 3; ++a) {
            boundsMin[a] = std::min(boundsMin[a], values[a]);
            boundsMax[a] = std::max(boundsMax[a], values[a]);
        }
    }
    }
    const float centreX = (boundsMin[0] + boundsMax[0]) * 0.5f;
    const float centreZ = (boundsMin[2] + boundsMax[2]) * 0.5f;
    const float height = std::max(1.0f, boundsMax[1] - boundsMin[1]);
    VOX_LOGI("newvegas") << "character bind-pose bounds"
                         << " x " << boundsMin[0] << ".." << boundsMax[0]
                         << " y " << boundsMin[1] << ".." << boundsMax[1]
                         << " z " << boundsMin[2] << ".." << boundsMax[2]
                         << " (" << height << " units tall)";

    // Stand the character in front of wherever streaming spawned the camera,
    // rather than moving the camera to the character. The spawn is on the
    // ground in Goodsprings and the camera is at eye height there; dragging it
    // to a bare bounding box would give up the one thing this view is for,
    // which is seeing the body at the same scale as the world around it.
    //
    // The offsets fold into the bone matrices in updateCharacterPose: a skinned
    // actor has no separate instance transform.
    const float yawRadians = m_yawDegrees * (kPi / 180.0f);
    const float forwardX = std::cos(yawRadians);
    const float forwardZ = std::sin(yawRadians);
    // Far enough that the whole figure fits a 75-degree vertical FOV with
    // margin (tan(37.5 deg) ~= 0.767), and no closer -- near clip aside, a body
    // filling the frame hides exactly the scale comparison being made.
    const float standoff = height * 1.1f / 0.767f;
    m_characterWorldX = m_cameraX + (forwardX * standoff) - centreX;
    m_characterWorldZ = m_cameraZ + (forwardZ * standoff) - centreZ;
    // The bind pose already stands on y = 0 (measured: feet at 0.78), so the
    // ground height goes in unmodified.
    float groundY = m_cameraY;
    if (groundHeightAt(m_characterWorldX, m_characterWorldZ, groundY)) {
        m_characterWorldY = groundY;
    } else {
        // No collision data yet (the cell may still be streaming): drop the
        // character to the camera's own foot height, which the spawn put on the
        // ground.
        m_characterWorldY = m_cameraY - kEyeHeightUnits;
    }
    VOX_LOGI("newvegas") << "character placed at " << m_characterWorldX << ", " << m_characterWorldY
                         << ", " << m_characterWorldZ << " (camera at " << m_cameraX << ", "
                         << m_cameraY << ", " << m_cameraZ << ")";
    return true;
}

void BethesdaApp::updateCharacterPose() {
    if (m_characterBindPose.empty()) {
        return;
    }
    // ODAI_FNV_CHAR_NOPOSE=1 never submits a pose. With the output buffer now
    // seeded at upload time, the actor should still draw -- in rest pose, at
    // the origin -- which isolates the per-frame pose upload from everything
    // else the skinned path does.
    if (std::getenv("ODAI_FNV_CHAR_NOPOSE") != nullptr) {
        return;
    }
    // ODAI_FNV_CHAR_IDENTITY=1 submits a pose of the right SHAPE but with no
    // data in it: as many identity matrices as the slot expects. It separates
    // the two ways this path can fail, which look identical on screen -- a
    // mechanical fault in the per-frame upload (wrong buffer, missing barrier)
    // still corrupts the frame with identity matrices, whereas a fault in the
    // bind-pose MATRICES themselves cannot, and the actor merely collapses to
    // the origin.
    if (std::getenv("ODAI_FNV_CHAR_IDENTITY") != nullptr) {
        m_characterPoseScratch.assign(m_characterBindPose.size(), odai::math::Matrix4::identity());
        render::ImportedSkinnedActorFrameData identityPose{};
        identityPose.boneMatrices = m_characterPoseScratch;
        m_renderer.setSkinnedActorPose(0u, identityPose);
        return;
    }
    // World placement rides on the bone matrices, pre-multiplied: the skinning
    // pass consumes bone matrices and nothing else, so there is no separate
    // instance transform to put it in.
    const odai::math::Matrix4 actorWorld = odai::math::Matrix4::translation(
        odai::math::Vector3{m_characterWorldX, m_characterWorldY, m_characterWorldZ});
    m_characterPoseScratch.resize(m_characterBindPose.size());
    for (std::size_t i = 0; i < m_characterBindPose.size(); ++i) {
        m_characterPoseScratch[i] = actorWorld * m_characterBindPose[i];
    }
    static bool loggedPose = false;
    if (!loggedPose && !m_characterPoseScratch.empty()) {
        loggedPose = true;
        const odai::math::Matrix4& b = m_characterBindPose[0];
        const odai::math::Matrix4& f = m_characterPoseScratch[0];
        VOX_LOGI("newvegas") << "pose[0] bind translation (" << b(0, 3) << "," << b(1, 3) << ","
                             << b(2, 3) << ") final (" << f(0, 3) << "," << f(1, 3) << ","
                             << f(2, 3) << ") bones=" << m_characterPoseScratch.size();
    }
    render::ImportedSkinnedActorFrameData pose{};
    pose.boneMatrices = m_characterPoseScratch;
    m_renderer.setSkinnedActorPose(0u, pose);
}

void BethesdaApp::applyTimeOfDay() {
    // Map 0..24h onto a sun that rises in the east and sets in the west.
    //
    // Sign convention, which is easy to get backwards: setSunAngles takes the
    // direction the light TRAVELS, not the direction to the sun. frame_run.cc
    // computes toSun = -sunDirection, so the sun is above the horizon only
    // while pitch is NEGATIVE — hence the debug slider's -89..+5 range.
    // Getting this backwards puts the
    // sun under the map at every hour and the whole world renders in ambient
    // only, which reads as "everything is super dark".
    //
    //   midnight -> +75 (below horizon)   dawn/dusk -> 0 (on the horizon)
    //   noon     -> -75 (high overhead)
    const float dayFraction = m_timeOfDayHours / 24.0f;
    const float pitchDegrees = std::cos(dayFraction * 2.0f * kPi) * 75.0f;
    const float yawDegrees = 90.0f + (dayFraction * 360.0f);
    m_renderer.setSunAngles(yawDegrees, pitchDegrees);
    applyWeather();
}

void BethesdaApp::initWeather() {
    if (m_streamDirectory.empty()) {
        return;  // a cooked scene has no plugin to read weather from
    }
    if (m_streamIsMorrowind) {
        VOX_LOGI("newvegas") << "TES3 load order: keeping the procedural Morrowind sky";
        return;
    }
    // This used to skip the read entirely unless a weather mod or an explicit
    // --weather gave it something to select. The pause menu's weather picker is
    // what made that premise false: vanilla's own records ARE selectable now, so
    // gating on a mod being present left the picker permanently empty on a stock
    // install. The read is a top-level group-header walk (see
    // buildFalloutWeatherTables) rather than a scan of the file, so paying it
    // unconditionally is close to free: measured at 0.49 ms for vanilla's 63
    // WTHR and 31 CLMT, against a ~2.0 s startup. The log line below carries the
    // number so a regression here cannot hide.

    std::vector<std::string> requested;
    requested.push_back(m_streamPlugin);
    requested.insert(requested.end(), m_extraPlugins.begin(), m_extraPlugins.end());

    // Timed because this now runs on every launch rather than only when a
    // weather mod is loaded, and because a plugin scan that forgets to filter on
    // the record header is the classic way to add seconds to startup here
    // without anything looking wrong. See CLAUDE.md on onRecordHeader.
    const core::Stopwatch weatherTimer;

    importer::fnv::FalloutLoadOrder fallbackOrder;
    const importer::fnv::FalloutLoadOrder* weatherOrder = &m_streamLoadOrder;
    std::string error;
    if (weatherOrder->empty()) {
        // Single-plugin Fallout/Oblivion sessions do not keep a streaming load
        // order. Reconstruct only in that case. Skyrim must use the exact
        // authoritative profile already opened by the streamer so DLC and ESL
        // WRLD/CLMT/WTHR overrides cannot disappear from the sky subsystem.
        for (const std::string& modDirectory : m_modDirectories) {
            fallbackOrder.addSearchRoot(std::filesystem::path(modDirectory));
        }
        if (!fallbackOrder.open(
                std::filesystem::path(m_streamDirectory), requested, error)) {
            VOX_LOGW("newvegas") << "weather disabled: " << error;
            return;
        }
        weatherOrder = &fallbackOrder;
    }
    if (!buildFalloutWeatherTables(*weatherOrder, m_weatherTables, error)) {
        VOX_LOGW("newvegas") << "weather disabled: " << error;
        return;
    }

    std::string loadOrderText;
    for (const auto& entry : weatherOrder->entries()) {
        if (!loadOrderText.empty()) {
            loadOrderText += " -> ";
        }
        loadOrderText += entry.header.fileName;
    }
    VOX_LOGI("newvegas") << "load order: " << loadOrderText;
    VOX_LOGI("newvegas") << "weather: " << m_weatherTables.weathers.size() << " WTHR, "
                         << m_weatherTables.climates.size() << " CLMT in "
                         << weatherTimer.elapsedMs() << " ms";

    if (!m_requestedWeatherEditorId.empty()) {
        const importer::fnv::FalloutWeatherRecord* weather =
            m_weatherTables.findWeatherByEditorId(m_requestedWeatherEditorId);
        if (weather == nullptr) {
            VOX_LOGW("newvegas") << "no weather named \"" << m_requestedWeatherEditorId
                                 << "\"; falling back to the climate";
        } else {
            m_activeWeatherFormId = weather->formId;
            VOX_LOGI("newvegas") << "weather forced to " << weather->editorId << " (0x"
                                 << std::hex << weather->formId << std::dec << ")";
        }
    }

    if (m_activeWeatherFormId == 0u) {
        // Fall back to the climate of the worldspace WE ARE STREAMING: whichever
        // of its weathers has the highest chance is the closest thing to "what
        // you would normally see here" without running the mod's selection
        // scripts.
        //
        // This used to walk climateByWorldspaceFormId and take the first entry
        // that had any weathers, discarding the worldspace key entirely. With
        // FalloutNV.esm alone that is survivable -- there is not much to pick
        // wrongly from. Add a plugin that pulls the DLCs in as masters, which
        // any "base game + all DLC" patch does, and the first entry in an
        // unordered_map became Lonesome Road's NVDLC04NukedClimate: the Mojave
        // rendered under the Divide's irradiated sky, green from horizon to
        // zenith. The map is keyed by worldspace precisely so it can be asked
        // about ONE worldspace, and the fix is to ask.
        const importer::fnv::FalloutClimateRecord* bestClimate = nullptr;
        const auto worldspaceIt = m_weatherTables.worldspaceFormIdByEditorId.find(
            toLowerAscii(m_streamWorldspace));
        if (worldspaceIt == m_weatherTables.worldspaceFormIdByEditorId.end()) {
            VOX_LOGW("newvegas") << "weather: no worldspace record named \"" << m_streamWorldspace
                                 << "\"; leaving the procedural sky alone";
        } else {
            // A WALLED CITY INHERITS ITS PARENT'S CLIMATE. Skyrim's
            // WhiterunWorld record is an EDID and a WNAM and nothing else, so
            // asking it for a climate finds none and the city renders under the
            // bare procedural sky -- no authored gradient, and no cloud layer at
            // all. Tamriel, one hop up, carries the climate for the whole
            // province. Bounded rather than recursive; the chain is one link in
            // practice and a cycle must not hang startup.
            auto climateIt = m_weatherTables.climateByWorldspaceFormId.find(worldspaceIt->second);
            std::uint32_t inheritedFrom = worldspaceIt->second;
            for (int hop = 0; hop < 8 && climateIt == m_weatherTables.climateByWorldspaceFormId.end();
                 ++hop) {
                const auto parentIt = m_weatherTables.parentWorldspaceFormId.find(inheritedFrom);
                if (parentIt == m_weatherTables.parentWorldspaceFormId.end()) {
                    break;
                }
                inheritedFrom = parentIt->second;
                climateIt = m_weatherTables.climateByWorldspaceFormId.find(inheritedFrom);
            }
            if (climateIt != m_weatherTables.climateByWorldspaceFormId.end() &&
                inheritedFrom != worldspaceIt->second) {
                VOX_LOGI("newvegas") << "weather: " << m_streamWorldspace
                                     << " names no climate; inherited from parent worldspace 0x"
                                     << std::hex << inheritedFrom << std::dec;
            }
            if (climateIt == m_weatherTables.climateByWorldspaceFormId.end()) {
                VOX_LOGW("newvegas") << "weather: worldspace " << m_streamWorldspace
                                     << " names no climate; leaving the procedural sky alone";
            } else {
                const auto found = m_weatherTables.climates.find(climateIt->second);
                if (found == m_weatherTables.climates.end() || found->second.weathers.empty()) {
                    VOX_LOGW("newvegas")
                        << "weather: climate for " << m_streamWorldspace
                        << " is missing or lists no weathers; leaving the procedural sky alone";
                } else {
                    bestClimate = &found->second;
                }
            }
        }
        if (bestClimate != nullptr) {
            // TNAM, in 10-minute units, giving the START and END of each
            // transition; the samplers want the single hour at which the
            // Sunrise and Sunset slots PEAK, which is the midpoint.
            // SkyrimClimate authors 5:30-10:00 and 16:00-20:30, so its dawn
            // peaks at 7:45 rather than the 6:00 default -- close to two hours
            // out, which is a whole slot's worth of colour.
            const auto hoursFromTnam = [](std::uint8_t begin, std::uint8_t end) {
                return ((static_cast<float>(begin) + static_cast<float>(end)) * 0.5f) / 6.0f;
            };
            const float sunrise = hoursFromTnam(bestClimate->sunriseBegin, bestClimate->sunriseEnd);
            const float sunset = hoursFromTnam(bestClimate->sunsetBegin, bestClimate->sunsetEnd);
            // A climate with no TNAM reads as 0 and 0, which would put dusk
            // before dawn and collapse the whole day curve onto one slot.
            if (sunrise > 0.5f && sunset > sunrise + 1.0f && sunset < 23.5f) {
                m_sunriseHour = sunrise;
                m_sunsetHour = sunset;
            }
            VOX_LOGI("newvegas") << "climate " << bestClimate->editorId << ": sunrise peaks "
                                 << m_sunriseHour << "h, sunset " << m_sunsetHour << "h";
            const auto best = std::max_element(
                bestClimate->weathers.begin(), bestClimate->weathers.end(),
                [](const auto& a, const auto& b) { return a.chance < b.chance; });
            m_activeWeatherFormId = best->weatherFormId;
            const importer::fnv::FalloutWeatherRecord* weather =
                m_weatherTables.findWeather(m_activeWeatherFormId);
            VOX_LOGI("newvegas") << "weather from climate " << bestClimate->editorId << ": "
                                 << (weather != nullptr ? weather->editorId : "<unresolved>");
        }
    }

    // Everything from here -- clouds, sky colours, audio, tonemap -- is the same
    // work a runtime weather change does, so it is one call rather than a copy.
    selectWeather(m_activeWeatherFormId);
}

// The post-processing curve. Deliberately NOT part of selectWeather, where it
// used to live: initWeather returns early on any plugin with no TES4 record, so
// every Morrowind run silently ignored ODAI_FNV_TONEMAP entirely. A renderer
// setting that has nothing to do with weather must not be reachable only
// through the weather path -- the symptom was an env var that measured
// byte-identical to unset and looked like a broken tonemap rather than a
// missing call.
void BethesdaApp::applyTonemapSettings() {
    // ODAI_FNV_WHITEPOINT=<scene linear>[,<shoulder>] pins a scene value to
    // display white on the ACES path. Off by default, because it is a look
    // change and every other game shares this curve.
    //
    // What it is FOR: measured across a Seyda-Neen-to-Balmora flight, the 99th
    // percentile of frame luma sat between 0.64 and 0.70 and moved by under
    // 0.02 under every other knob in the chain -- fog distance, the ENB curve,
    // the stylized colour look. The frame never reached white, and no amount of
    // grading fixes a range the tonemap did not produce.
    if (const char* whiteEnv = std::getenv("ODAI_FNV_WHITEPOINT")) {
        render::TonemapSettings tonemap = m_renderer.tonemapSettings();
        tonemap.whitePoint = std::strtof(whiteEnv, nullptr);
        if (const char* comma = std::strchr(whiteEnv, ',')) {
            tonemap.highlightShoulder = std::strtof(comma + 1, nullptr);
        }
        m_renderer.setTonemapSettings(tonemap);
        VOX_LOGI("newvegas") << "tonemap white point " << tonemap.whitePoint << ", shoulder "
                             << tonemap.highlightShoulder;
    }

    // ODAI_FNV_TONEMAP=enb switches the post pass to the curve Enhanced Shaders
    // uses, with its own tuned Fallout values. Off by default: it is a distinct
    // look, not a strict improvement, and every other game keeps the ACES fit
    // regardless because the setting is per-renderer and only this game sets it.
    if (const char* tonemapEnv = std::getenv("ODAI_FNV_TONEMAP")) {
        std::string mode = tonemapEnv;
        for (char& c : mode) {
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
        if (mode == "enb" || mode == "1") {
            render::TonemapSettings tonemap;
            tonemap.mode = render::TonemapMode::Enb;
            // Enhanced Shaders retunes these by time of day (contrast 1.35 day /
            // 1.25 night, saturation 1.25 / 0.9, curve 8.0 / 10.0). Interpolate
            // on the same day/night axis rather than picking one and calling it
            // done -- the night values exist because the day ones look wrong
            // after dark.
            const bool night = m_timeOfDayHours < 6.0f || m_timeOfDayHours >= 19.0f;
            tonemap.contrast = night ? 1.25f : 1.35f;
            tonemap.saturation = night ? 0.90f : 1.25f;
            tonemap.curve = night ? 10.0f : 8.0f;
            tonemap.overbrightDampening = night ? 50.0f : 75.0f;
            m_renderer.setTonemapSettings(tonemap);
            VOX_LOGI("newvegas") << "tonemap: ENB (Enhanced Shaders values, "
                                 << (night ? "night" : "day") << ")";
        }
    }
}

namespace {

// Turns Skyrim's semantic cloud band into the slice of sky it covers and the
// projection its art was drawn for. The record says WHERE a layer belongs (see
// FalloutCloudBand); this says what that means to the renderer, which is the
// game's business rather than the importer's.
//
// The windows overlap on purpose: a deck of cloud and the bank under it are not
// separated by a line in the sky. The numbers are dir.y, so 0 is the horizon
// and 1 the zenith, and the horizon band's 0.30 ceiling is about 17 degrees --
// roughly where a real cloud bank stops reading as "on the skyline".
void applySkyrimCloudBand(importer::fnv::FalloutCloudBand band,
                          render::WeatherCloudLayer& target) {
    using Band = importer::fnv::FalloutCloudBand;
    switch (band) {
        case Band::Upper:
            target.mapping = render::WeatherCloudMapping::TilingPlane;
            target.scale = 2.6f;
            target.bandLow = 0.20f;
            target.bandHigh = 1.0f;
            break;
        case Band::Lower:
            target.mapping = render::WeatherCloudMapping::TilingPlane;
            target.scale = 1.5f;
            target.bandLow = 0.10f;
            target.bandHigh = 1.0f;
            break;
        case Band::Horizon:
            target.mapping = render::WeatherCloudMapping::Cylindrical;
            // Bearing repeats: the art is one bank, not a panorama of the whole
            // compass, so it has to go round more than once.
            target.scale = 4.0f;
            target.bandLow = 0.0f;
            target.bandHigh = 0.30f;
            break;
        case Band::Fill:
        case Band::WholeSky:
            target.mapping = render::WeatherCloudMapping::TilingPlane;
            target.scale = 2.0f;
            break;
    }
}

bool buildSkyrimWeatherCloudMesh(
    const importer::fnv::FalloutAssetSource& assets,
    const importer::fnv::FalloutWeatherRecord& weather,
    const int layerSources[render::kWeatherCloudLayerCount],
    render::WeatherCloudMesh& outMesh,
    std::string& outError) {
    outMesh = render::WeatherCloudMesh{};
    std::vector<std::uint8_t> bytes;
    if (!assets.resolveMesh("Sky\\Clouds.nif", bytes, outError)) {
        outError = "authored Skyrim weather dome is missing: " + outError;
        return false;
    }
    importer::fnv::NifModel model;
    if (!importer::fnv::parseNifStaticMesh(bytes, model, outError)) {
        outError = "authored Skyrim weather dome could not be parsed: " + outError;
        return false;
    }
    // clouds.nif contains one surface per WTHR texture layer, stored in reverse
    // layer order (28..0). This is the geometry that owns the atlas UVs; a
    // fullscreen projection cannot reconstruct them from the DDS alone.
    constexpr std::size_t kSkyrimWeatherLayerCount = 29u;
    if (model.shapes.size() != kSkyrimWeatherLayerCount) {
        outError = "authored Skyrim weather dome expected 29 surfaces, found " +
            std::to_string(model.shapes.size());
        return false;
    }

    constexpr float kSkyRadiusScale = 1000.0f;
    for (std::uint32_t slot = 0u; slot < render::kWeatherCloudLayerCount; ++slot) {
        const int source = layerSources[slot];
        if (source < 0 || static_cast<std::size_t>(source) >= weather.cloudLayers.size()) {
            continue;
        }
        const int authoredLayer = weather.cloudLayers[static_cast<std::size_t>(source)].index;
        if (authoredLayer < 0 || authoredLayer >= static_cast<int>(kSkyrimWeatherLayerCount)) {
            outError = "Skyrim weather cloud layer index is outside clouds.nif: " +
                std::to_string(authoredLayer);
            return false;
        }
        const importer::fnv::NifShape& shape =
            model.shapes[(kSkyrimWeatherLayerCount - 1u) -
                         static_cast<std::size_t>(authoredLayer)];
        float maximumVertexAlpha = 1.0f;
        if (!shape.colors.empty()) {
            maximumVertexAlpha = 0.0f;
            for (std::size_t color = 3u; color < shape.colors.size(); color += 4u) {
                maximumVertexAlpha = std::max(maximumVertexAlpha, shape.colors[color]);
            }
        }
        VOX_LOGI("newvegas") << "Skyrim cloud slot " << slot
                              << " WTHR layer " << authoredLayer
                              << " -> clouds.nif surface \"" << shape.name
                              << "\" (" << (shape.positions.size() / 3u)
                              << " vertices, max alpha " << maximumVertexAlpha << ")";
        const std::uint32_t baseVertex = static_cast<std::uint32_t>(outMesh.vertices.size());
        const std::size_t vertexCount = shape.positions.size() / 3u;
        outMesh.vertices.reserve(outMesh.vertices.size() + vertexCount);
        for (std::size_t vertexIndex = 0u; vertexIndex < vertexCount; ++vertexIndex) {
            render::WeatherCloudMeshVertex vertex;
            // NIF model space is Z-up; the renderer is Y-up. The shader adds
            // the current camera position, so these remain camera-centred and
            // cannot drift through the city as ordinary world geometry.
            vertex.position[0] = shape.positions[vertexIndex * 3u] * kSkyRadiusScale;
            vertex.position[1] = shape.positions[(vertexIndex * 3u) + 2u] * kSkyRadiusScale;
            vertex.position[2] = -shape.positions[(vertexIndex * 3u) + 1u] * kSkyRadiusScale;
            if ((vertexIndex * 2u) + 1u < shape.uvs.size()) {
                vertex.uv[0] = shape.uvs[vertexIndex * 2u];
                vertex.uv[1] = shape.uvs[(vertexIndex * 2u) + 1u];
            }
            if ((vertexIndex * 4u) + 3u < shape.colors.size()) {
                std::copy_n(&shape.colors[vertexIndex * 4u], 4u, vertex.color);
            }
            vertex.layer = slot;
            outMesh.vertices.push_back(vertex);
        }
        for (const std::uint32_t index : shape.triangleIndices) {
            if (index >= vertexCount) {
                outError = "authored Skyrim weather dome has an out-of-range index";
                return false;
            }
            outMesh.indices.push_back(baseVertex + index);
        }
    }
    if (outMesh.indices.empty()) {
        outError = "authored Skyrim weather dome selected no active surfaces";
        return false;
    }
    return true;
}

}  // namespace

// Everything that has to happen when the active weather changes: cloud layers
// re-uploaded, sky colours re-decoded, audio re-picked.
//
// Split out of initWeather so the pause menu can change weather at runtime.
// Doing it by hand at the call site was never going to stay correct -- setting
// m_activeWeatherFormId alone leaves the previous weather's cloud textures on
// the GPU and its rain still playing, which reads as "the picker only half
// works" rather than as a missing call.
void BethesdaApp::selectWeather(std::uint32_t weatherFormId) {
    m_activeWeatherFormId = weatherFormId;

    // Cloud layers. These come out of the mod's own BSA, which the streamer's
    // asset source already indexes -- reusing it means no second search path and
    // no second copy of the loose-beats-archive precedence rules.
    const importer::fnv::FalloutWeatherRecord* active =
        m_activeWeatherFormId != 0u ? m_weatherTables.findWeather(m_activeWeatherFormId) : nullptr;
    if (active != nullptr && m_streamer != nullptr) {
        const importer::fnv::FalloutAssetSource& assets = m_streamer->assets();
        render::WeatherCloudTextures clouds;
        const bool tiling =
            active->cloudMapping == importer::fnv::FalloutCloudMapping::TilingPlane;
        for (int& source : m_cloudLayerSource) {
            source = -1;
        }
        int loadedLayers = 0;

        // The record has already dropped the layers it disables and paired each
        // survivor with its own tint; this only has to fill the renderer's four
        // slots with the ones whose textures actually resolve, REMEMBERING WHICH
        // LAYER EACH SLOT CAME FROM so applyWeather tints it from the same one.
        for (std::size_t index = 0;
             index < active->cloudLayers.size() &&
             static_cast<std::size_t>(loadedLayers) < render::kWeatherCloudLayerCount;
             ++index) {
            const importer::fnv::FalloutWeatherCloudLayer& layer = active->cloudLayers[index];
            const std::size_t slot = static_cast<std::size_t>(loadedLayers);
            std::vector<std::uint8_t> bytes;
            std::string assetError;
            if (!assets.resolveTexture(layer.texture, bytes, assetError)) {
                // Not a warning for Skyrim: a retail record names layers whose
                // textures do not ship (SkyrimCloudy's disabled ones are dead
                // leftovers from the Oblivion and Fallout records it was copied
                // from), and NAM1 has already thrown those away. One that gets
                // this far is worth hearing about.
                VOX_LOGW("newvegas") << "cloud layer " << layer.index << " (" << layer.texture
                                     << ") unresolved: " << assetError;
                continue;
            }
            render::WeatherCloudLayer& target = clouds.layers[slot];
            if (!importer::loadDdsFromMemory(bytes.data(), bytes.size(), target.texture)) {
                VOX_LOGW("newvegas") << "cloud layer " << layer.index << " (" << layer.texture
                                     << ") failed to decode";
                target.texture = importer::ImportedSceneTexture{};
                continue;
            }
            target.texture.sourcePath = layer.texture;
            if (tiling) {
                // Texture units per second. Skyrim's bytes are a rate, not a
                // velocity in any unit this renderer shares, so the scale is
                // chosen for the look: a sheet crosses the sky in a couple of
                // minutes at the speeds SkyrimCloudy authors.
                target.scrollU = layer.driftX * 0.010f;
                target.scrollV = layer.driftY * 0.010f;
                applySkyrimCloudBand(layer.band, target);
            } else {
                // Radians per second about the zenith -- a dome map rotates, it
                // does not translate.
                target.mapping = render::WeatherCloudMapping::DomeFisheye;
                target.scrollU = layer.driftX * 0.0035f;
                // Dome scale: 1.0 puts the horizon exactly on the texture's
                // inscribed circle, which is how these fisheye sky maps are
                // drawn. Slightly under 1 for the upper layers pulls their rim
                // inside the horizon so they read as higher and further away.
                target.scale = (layer.index < 2) ? 1.0f : 0.92f;
            }
            m_cloudLayerSource[slot] = static_cast<int>(index);
            ++loadedLayers;
        }
        VOX_LOGI("newvegas") << "cloud layers: " << loadedLayers << " of "
                             << active->cloudLayers.size() << " authored in use ("
                             << (tiling ? "tiling sheets" : "dome fisheye") << ")";
        if (tiling) {
            render::WeatherCloudMesh cloudMesh;
            std::string cloudMeshError;
            if (buildSkyrimWeatherCloudMesh(
                    assets, *active, m_cloudLayerSource, cloudMesh, cloudMeshError)) {
                m_renderer.setWeatherCloudMesh(cloudMesh);
            } else {
                m_renderer.setWeatherCloudMesh(render::WeatherCloudMesh{});
                VOX_LOGE("newvegas") << cloudMeshError;
            }
        } else {
            m_renderer.setWeatherCloudMesh(render::WeatherCloudMesh{});
        }
        m_renderer.setWeatherClouds(clouds);
    }

    applyWeather();
    initWeatherAudio();


}

namespace {

// Writes already-resolved sound bytes to a playable file and returns its path.
//
// Two reasons this exists rather than handing bytes straight to the audio
// facade. It loads by std::filesystem::path only, and Fallout's ambient loops
// are Ogg Vorbis, which miniaudio cannot decode at all -- so the .ogg is
// converted to .wav here (see newvegas_ogg.cc). Cached by name, so a sound
// costs one conversion per install rather than one per run.
std::filesystem::path cacheWeatherSound(
    const std::string& virtualPath,
    const std::vector<std::uint8_t>& bytes,
    const std::filesystem::path& cacheDirectory) {
    if (cacheDirectory.empty() || bytes.empty()) {
        return {};
    }
    std::string leaf = virtualPath;
    const std::size_t lastSeparator = leaf.find_last_of("\\/");
    if (lastSeparator != std::string::npos) {
        leaf = leaf.substr(lastSeparator + 1u);
    }
    const std::filesystem::path raw = cacheDirectory / leaf;
    std::filesystem::path playable = raw;
    std::string extension = raw.extension().string();
    for (char& c : extension) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    const bool needsOggConversion = extension == ".ogg";
    const bool needsXwmConversion = extension == ".xwm";
    const bool needsConversion = needsOggConversion || needsXwmConversion;
    if (needsConversion) {
        playable.replace_extension(".wav");
    }

    std::error_code existsError;
    if (std::filesystem::exists(playable, existsError) && !existsError) {
        return playable;
    }

    std::error_code createError;
    std::filesystem::create_directories(cacheDirectory, createError);
    {
        std::ofstream out(raw, std::ios::binary | std::ios::trunc);
        if (!out) {
            return {};
        }
        out.write(
            reinterpret_cast<const char*>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
    }
    if (needsConversion) {
        const bool converted = needsXwmConversion
            ? decodeXwmToWav(raw, playable)
            : decodeOggToWav(raw, playable);
        if (!converted) {
            return {};
        }
    }
    return playable;
}

}  // namespace

void BethesdaApp::initWeatherAudio() {
    const importer::fnv::FalloutWeatherRecord* weather =
        m_activeWeatherFormId != 0u ? m_weatherTables.findWeather(m_activeWeatherFormId) : nullptr;
    if (weather == nullptr || m_streamDirectory.empty() || m_streamer == nullptr) {
        return;
    }

    const importer::fnv::FalloutAssetSource& assets = m_streamer->assets();
    const std::filesystem::path dataFilesPath(m_streamDirectory);
    const std::filesystem::path audioCache = m_streamCacheDirectory.empty()
        ? std::filesystem::path{}
        : std::filesystem::path(m_streamCacheDirectory) / "audio";

    // A runtime weather change must retire the previous global beds before
    // starting the next ones. Positional and regional ambience is managed
    // independently by updateSkyrimAmbience().
    if (m_rainAmbient.valid()) {
        m_audio.stopAmbient(m_rainAmbient, 1.0f);
    }
    if (m_windAmbient.valid()) {
        m_audio.stopAmbient(m_windAmbient, 1.0f);
    }
    m_rainAmbient = {};
    m_windAmbient = {};
    m_rainLoop = {};
    m_windLoop = {};

    // Resolves the first candidate that exists, through the ordinary asset
    // precedence, and caches it as a playable .wav.
    //
    // Order matters: a weather mod ships sounds authored for its own weathers
    // inside its BSA, which is indexed as a mod archive, so it wins. Reaching
    // past that into the base game's archives is how this first shipped, and it
    // picked "emt_raintoggle_lp" -- a MONO 6-second object-emitter loop from Old
    // World Blues, meant to play from a dripping pipe, not a global rain bed.
    const auto loadFirst = [&](std::initializer_list<const char*> candidates,
                               const char* what) -> audio::SoundHandle {
        for (const char* candidate : candidates) {
            std::vector<std::uint8_t> bytes;
            std::string assetError;
            if (!assets.resolveAsset(candidate, bytes, assetError) || bytes.empty()) {
                continue;
            }
            const std::filesystem::path cached =
                cacheWeatherSound(candidate, bytes, audioCache);
            if (cached.empty()) {
                continue;
            }
            const audio::SoundHandle handle =
                m_audio.loadSound(cached, audio::SoundCategory::Ambient);
            if (handle.id != 0u) {
                VOX_LOGI("newvegas") << what << ": " << candidate;
                return handle;
            }
        }
        VOX_LOGW("newvegas") << "no " << what << " found in the loaded archives";
        return {};
    };

    if (weather->hasPrecipitation()) {
        // WTHR has no rain-intensity field -- classification only says "rainy" --
        // so intensity comes from the editor ID, which is a heuristic and named
        // as one. The fallbacks walk down to whatever exists.
        std::string lowered = weather->editorId;
        for (char& c : lowered) {
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
        const bool heavy = lowered.find("heavy") != std::string::npos ||
            lowered.find("storm") != std::string::npos;
        m_rainLoop = heavy
            ? loadFirst({"sound\\fx\\amb\\weather\\rain\\amb_weather_rain_heavy_lp.wav",
                         "sound\\fx\\amb\\weather\\rain\\amb_weather_rain_medium_lp.wav",
                         "sound\\fx\\weather\\amb_weather_rain_heavy_lp.wav",
                         "sound\\fx\\weather\\amb_rainstorm_lp.wav",
                         "sound\\fx\\weather\\amb_weather_rain_medium_lp.wav",
                         "sound\\fx\\weather\\nvdlc02_rain-amb.wav"},
                        "rain")
            : loadFirst({"sound\\fx\\amb\\weather\\rain\\amb_weather_rain_medium_lp.wav",
                         "sound\\fx\\amb\\weather\\rain\\amb_weather_rain_light_lp.wav",
                         "sound\\fx\\amb\\weather\\rain\\amb_weather_rain_drizzle_lp.wav",
                         "sound\\fx\\weather\\amb_weather_rain_medium_lp.wav",
                         "sound\\fx\\weather\\amb_weather_rain_light_lp.wav",
                         "sound\\fx\\weather\\amb_rain_lp.wav",
                         "sound\\fx\\weather\\nvdlc02_rain-amb.wav"},
                        "rain");
        if (m_rainLoop.id != 0u) {
            m_rainAmbient = m_audio.startAmbient(m_rainLoop, 2.5f);
        }
    }

    if (weather->windSpeed > 40u) {
        const bool strongWind = weather->windSpeed > 80u;
        m_windLoop = loadFirst(
            strongWind
                ? std::initializer_list<const char*>{
                      "sound\\fx\\weather\\amb_windheavy_lp.wav",
                      "sound\\fx\\weather\\amb_windlight_lp.wav"}
                : std::initializer_list<const char*>{
                      "sound\\fx\\weather\\amb_windlight_lp.wav",
                      "sound\\fx\\weather\\amb_windheavy_lp.wav"},
            "wind");
        if (m_windLoop.id != 0u) {
            m_windAmbient = m_audio.startAmbient(m_windLoop, 3.0f);
        }
    }

    // Skyrim ambience comes from authored regional and placed descriptors.
    // Its score is xWMA in the Sounds BSA; extract and transcode one stable
    // daytime exploration track into the cache that miniaudio can stream.
    if (m_streamIsSkyrim) {
        if (!m_musicTrack.valid()) {
            const char* requested = std::getenv("ODAI_SKYRIM_MUSIC");
            const std::string virtualPath = requested != nullptr && requested[0] != '\0'
                ? requested
                : "music\\explore\\mus_explore_day_01.xwm";
            std::vector<std::uint8_t> bytes;
            std::string assetError;
            if (assets.resolveAsset(virtualPath, bytes, assetError) && !bytes.empty()) {
                const std::filesystem::path playable =
                    cacheWeatherSound(virtualPath, bytes, audioCache);
                if (!playable.empty()) {
                    m_musicTrack = m_audio.loadMusic(playable);
                }
            }
            if (m_musicTrack.valid()) {
                m_audio.playMusic(m_musicTrack, 4.0f, true);
                VOX_LOGI("newvegas") << "Skyrim music: " << virtualPath;
            } else {
                VOX_LOGW("newvegas") << "Skyrim music unavailable: " << virtualPath;
            }
        }
        return;
    }

    // Radio, not score. Fallout keeps two separate sets of loose music: the
    // orchestral exploration beds under Data\Music, and the 48 licensed radio
    // songs under Data\Sound\songs\radionv -- Big Iron, Blue Moon, Johnny
    // Guitar. The radio station is the one that sounds like Fallout, and it is
    // what this plays.
    //
    // ODAI_FNV_MUSIC takes either a full path or a song name ("Big_Iron",
    // "MUS_Big_Iron", "MUS_Big_Iron.mp3"); with nothing set, a track is picked
    // from the station at random, like tuning in.
    std::filesystem::path musicPath;
    const std::filesystem::path stationDir = dataFilesPath / "Sound" / "songs" / "radionv";
    if (const char* musicEnv = std::getenv("ODAI_FNV_MUSIC")) {
        const std::string request = musicEnv;
        std::error_code existsError;
        if (std::filesystem::exists(request, existsError) && !existsError) {
            musicPath = request;
        } else {
            // Match by name, case-insensitively, with or without the MUS_ prefix
            // and the extension.
            std::string wanted = request;
            for (char& c : wanted) {
                c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            }
            std::error_code iterError;
            std::filesystem::directory_iterator iterator(stationDir, iterError);
            if (!iterError) {
                for (const auto& entry : iterator) {
                    std::string name = entry.path().filename().string();
                    for (char& c : name) {
                        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
                    }
                    if (name.find(wanted) != std::string::npos) {
                        musicPath = entry.path();
                        break;
                    }
                }
            }
            if (musicPath.empty()) {
                VOX_LOGW("newvegas") << "no song matching \"" << request << "\" in "
                                     << stationDir.string();
            }
        }
    }
    if (musicPath.empty()) {
        std::vector<std::filesystem::path> station;
        std::error_code iterError;
        std::filesystem::directory_iterator iterator(stationDir, iterError);
        if (!iterError) {
            for (const auto& entry : iterator) {
                if (entry.path().extension() == ".mp3") {
                    station.push_back(entry.path());
                }
            }
        }
        if (!station.empty()) {
            // Sorted first so the pick depends only on the seed, not on readdir
            // order, which differs between machines.
            std::sort(station.begin(), station.end());
            std::mt19937 rng(static_cast<std::uint32_t>(
                std::chrono::steady_clock::now().time_since_epoch().count()));
            musicPath = station[rng() % station.size()];
        }
    }

    std::error_code musicError;
    if (!musicPath.empty() && std::filesystem::exists(musicPath, musicError) && !musicError) {
        m_musicTrack = m_audio.loadMusic(musicPath);
        if (m_musicTrack.id != 0u) {
            m_audio.playMusic(m_musicTrack, 4.0f, true);
            VOX_LOGI("newvegas") << "radio: " << musicPath.stem().string();
        }
    } else {
        VOX_LOGW("newvegas") << "no radio songs found under " << stationDir.string();
    }
}

audio::SoundHandle BethesdaApp::loadAmbientDescriptor(std::uint32_t descriptorFormId) {
    if (descriptorFormId == 0u || m_streamer == nullptr) {
        return {};
    }
    if (const auto cached = m_ambientSounds.find(descriptorFormId);
        cached != m_ambientSounds.end()) {
        return cached->second;
    }
    const importer::fnv::FalloutSoundDescriptorRecord* descriptor =
        m_streamer->soundDescriptor(descriptorFormId);
    if (descriptor == nullptr || descriptor->filePaths.empty()) {
        m_ambientSounds.emplace(descriptorFormId, audio::SoundHandle{});
        return {};
    }

    // A descriptor can author several equivalent takes. Pick once from stable
    // IDs and the capture seed, never directory or load timing order.
    const std::size_t variant = static_cast<std::size_t>(
        (descriptorFormId ^ m_ambienceRandomState) % descriptor->filePaths.size());
    std::string virtualPath = descriptor->filePaths[variant];
    std::replace(virtualPath.begin(), virtualPath.end(), '/', '\\');
    std::string loweredPath = virtualPath;
    for (char& c : loweredPath) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    // Some Skyrim descriptors retain an export-root prefix such as
    // Data\Sound\FX\... while others start at FX\. Cut to the actual virtual
    // Sound root before applying the ordinary prefix.
    const std::size_t soundRoot = loweredPath.find("sound\\");
    if (soundRoot != std::string::npos) {
        virtualPath = virtualPath.substr(soundRoot);
        loweredPath = loweredPath.substr(soundRoot);
    }
    if (loweredPath.rfind("sound\\", 0u) != 0u) {
        virtualPath = "sound\\" + virtualPath;
    }
    std::filesystem::path extension(virtualPath);
    std::string suffix = extension.extension().string();
    for (char& c : suffix) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    if (suffix != ".wav" && suffix != ".ogg") {
        VOX_LOGW("newvegas") << "unsupported ambient audio format: " << virtualPath;
        m_ambientSounds.emplace(descriptorFormId, audio::SoundHandle{});
        return {};
    }

    std::vector<std::uint8_t> bytes;
    std::string error;
    if (!m_streamer->assets().resolveAsset(virtualPath, bytes, error) || bytes.empty()) {
        VOX_LOGW("newvegas") << "ambient asset missing: " << virtualPath;
        m_ambientSounds.emplace(descriptorFormId, audio::SoundHandle{});
        return {};
    }
    const std::filesystem::path cacheDirectory = m_streamCacheDirectory.empty()
        ? std::filesystem::path{}
        : std::filesystem::path(m_streamCacheDirectory) / "audio";
    const std::filesystem::path playable =
        cacheWeatherSound(virtualPath, bytes, cacheDirectory);
    const audio::SoundHandle sound = playable.empty()
        ? audio::SoundHandle{}
        : m_audio.loadSound(playable, audio::SoundCategory::Ambient);
    m_ambientSounds.emplace(descriptorFormId, sound);
    if (sound.valid()) {
        VOX_LOGI("newvegas") << "ambient descriptor " << descriptor->editorId
                             << ": " << virtualPath;
    }
    return sound;
}

void BethesdaApp::clearSkyrimAmbience() {
    for (const auto& [reference, active] : m_activePlacedAmbients) {
        (void)reference;
        m_audio.stopAmbient(active.handle, 0.75f);
    }
    for (const auto& [descriptor, handle] : m_activeRegionAmbients) {
        (void)descriptor;
        m_audio.stopAmbient(handle, 1.5f);
    }
    m_activePlacedAmbients.clear();
    m_activeRegionAmbients.clear();
}

void BethesdaApp::updateSkyrimAmbience(float deltaSeconds) {
    if (m_streamer == nullptr || m_interiorStarted) {
        if (!m_activePlacedAmbients.empty() || !m_activeRegionAmbients.empty()) {
            clearSkyrimAmbience();
        }
        return;
    }

    struct Candidate {
        importer::fnv::FalloutSoundEmitterRecord emitter;
        float distanceSquared = 0.0f;
    };
    std::unordered_map<std::uint32_t, Candidate> nearestByDescriptor;
    for (const auto& [cell, emitters] : m_streamAmbientEmittersByCell) {
        (void)cell;
        for (const importer::fnv::FalloutSoundEmitterRecord& emitter : emitters) {
            const float dx = emitter.position[0] - m_cameraX;
            const float dy = emitter.position[1] - m_cameraY;
            const float dz = emitter.position[2] - m_cameraZ;
            const float distanceSquared = (dx * dx) + (dy * dy) + (dz * dz);
            const importer::fnv::FalloutSoundDescriptorRecord* descriptor =
                m_streamer->soundDescriptor(emitter.descriptorFormId);
            float maxDistance = 4000.0f;
            if (descriptor != nullptr) {
                if (const auto* output = m_streamer->soundOutputModel(
                        descriptor->outputModelFormId)) {
                    maxDistance = std::max(output->maxDistance, 1.0f);
                }
            }
            if (distanceSquared > maxDistance * maxDistance) {
                continue;
            }
            auto [it, inserted] = nearestByDescriptor.emplace(
                emitter.descriptorFormId, Candidate{emitter, distanceSquared});
            if (!inserted && distanceSquared < it->second.distanceSquared) {
                it->second = Candidate{emitter, distanceSquared};
            }
        }
    }
    std::vector<Candidate> candidates;
    candidates.reserve(nearestByDescriptor.size());
    for (const auto& entry : nearestByDescriptor) {
        candidates.push_back(entry.second);
    }
    std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
        return a.distanceSquared < b.distanceSquared;
    });
    if (candidates.size() > 4u) {
        candidates.resize(4u);
    }

    std::unordered_set<std::uint32_t> wantedReferences;
    for (const Candidate& candidate : candidates) {
        const auto& emitter = candidate.emitter;
        wantedReferences.insert(emitter.referenceFormId);
        auto active = m_activePlacedAmbients.find(emitter.referenceFormId);
        if (active != m_activePlacedAmbients.end()) {
            m_audio.setAmbientPosition(
                active->second.handle,
                {emitter.position[0], emitter.position[1], emitter.position[2]});
            continue;
        }
        const audio::SoundHandle sound = loadAmbientDescriptor(emitter.descriptorFormId);
        const importer::fnv::FalloutSoundDescriptorRecord* descriptor =
            m_streamer->soundDescriptor(emitter.descriptorFormId);
        audio::AttenuationParams attenuation{150.0f, 4000.0f, 1.0f};
        if (descriptor != nullptr) {
            if (const auto* output = m_streamer->soundOutputModel(
                    descriptor->outputModelFormId)) {
                attenuation.minDistance = output->minDistance;
                attenuation.maxDistance = output->maxDistance;
            }
        }
        const audio::AmbientHandle handle = m_audio.startAmbientAt(
            sound, {emitter.position[0], emitter.position[1], emitter.position[2]},
            attenuation, 0.75f);
        if (handle.valid()) {
            m_activePlacedAmbients.emplace(
                emitter.referenceFormId,
                ActivePlacedAmbient{emitter.descriptorFormId, handle});
        }
    }
    for (auto it = m_activePlacedAmbients.begin(); it != m_activePlacedAmbients.end();) {
        if (!wantedReferences.contains(it->first)) {
            m_audio.stopAmbient(it->second.handle, 0.75f);
            it = m_activePlacedAmbients.erase(it);
        } else {
            ++it;
        }
    }

    m_regionAmbiencePollSeconds += deltaSeconds;
    if (m_regionAmbiencePollSeconds < 5.0f) {
        return;
    }
    m_regionAmbiencePollSeconds = 0.0f;
    m_ambienceRandomState ^= m_ambienceRandomState << 13u;
    m_ambienceRandomState ^= m_ambienceRandomState >> 17u;
    m_ambienceRandomState ^= m_ambienceRandomState << 5u;
    const std::uint8_t weatherFlags = [&]() {
        const auto* weather = m_weatherTables.findWeather(m_activeWeatherFormId);
        return weather != nullptr && weather->classification != 0u
            ? weather->classification
            : static_cast<std::uint8_t>(1u);
    }();
    const float position[3] = {m_cameraX, m_cameraY, m_cameraZ};
    std::unordered_set<std::uint32_t> wantedRegionLoops;
    for (const importer::fnv::FalloutRegionRecord::Sound& regionSound :
         m_streamer->regionSoundsAtEngineSpace(position)) {
        if (regionSound.weatherFlags != 0u &&
            (regionSound.weatherFlags & weatherFlags) == 0u) {
            continue;
        }
        const auto* descriptor = m_streamer->soundDescriptor(regionSound.descriptorFormId);
        if (descriptor == nullptr) {
            continue;
        }
        const audio::SoundHandle sound = loadAmbientDescriptor(regionSound.descriptorFormId);
        if (!sound.valid()) {
            continue;
        }
        if (descriptor->looping && wantedRegionLoops.size() < 2u) {
            wantedRegionLoops.insert(regionSound.descriptorFormId);
            if (!m_activeRegionAmbients.contains(regionSound.descriptorFormId)) {
                const audio::AmbientHandle handle = m_audio.startAmbient(sound, 1.5f);
                if (handle.valid()) {
                    m_activeRegionAmbients.emplace(regionSound.descriptorFormId, handle);
                }
            }
        } else if (!descriptor->looping) {
            const float roll = static_cast<float>(m_ambienceRandomState % 10000u) / 100.0f;
            if (roll < std::clamp(regionSound.chance, 0.0f, 100.0f)) {
                m_audio.playSound(sound);
            }
        }
    }
    for (auto it = m_activeRegionAmbients.begin(); it != m_activeRegionAmbients.end();) {
        if (!wantedRegionLoops.contains(it->first)) {
            m_audio.stopAmbient(it->second, 1.5f);
            it = m_activeRegionAmbients.erase(it);
        } else {
            ++it;
        }
    }
}

void BethesdaApp::applyWeather() {
    if (m_streamIsMorrowind) {
        // TES3 does not have Fallout-style WTHR records, but publishing no
        // atmosphere at all is not neutral in the imported renderer.  It makes
        // the skybox use its procedural horizon while aerial perspective and
        // volumetric fog use a separate clear-sky fallback palette.  The two
        // meet as grey haze over a nearly black strip immediately below y=0.
        //
        // Publish one coherent TES3 atmosphere instead.  These are linear HDR
        // radiance stops, not display-referred Morrowind.ini bytes.  The sun
        // disk and scattering remain procedural; only the gradient, horizon,
        // and the fog it fades into are tied together here.
        const float hour = std::fmod(std::max(m_timeOfDayHours, 0.0f), 24.0f);
        const float sunHeight = std::sin(((hour - 6.0f) / 12.0f) * kPi);
        const auto smoothUnit = [](float value) {
            const float t = std::clamp(value, 0.0f, 1.0f);
            return t * t * (3.0f - (2.0f * t));
        };
        const float daylight = smoothUnit((sunHeight + 0.08f) / 0.58f);
        const float twilight =
            1.0f - smoothUnit(std::fabs(sunHeight) / 0.34f);

        constexpr float nightUpper[3] = {0.012f, 0.024f, 0.065f};
        constexpr float nightLower[3] = {0.025f, 0.045f, 0.105f};
        constexpr float nightHorizon[3] = {0.055f, 0.070f, 0.120f};
        constexpr float nightFog[3] = {0.040f, 0.055f, 0.090f};
        constexpr float dayUpper[3] = {0.105f, 0.300f, 0.680f};
        constexpr float dayLower[3] = {0.430f, 0.700f, 1.050f};
        constexpr float dayHorizon[3] = {0.720f, 0.835f, 0.960f};
        constexpr float dayFog[3] = {0.480f, 0.625f, 0.760f};
        constexpr float duskUpper[3] = {0.150f, 0.175f, 0.360f};
        constexpr float duskLower[3] = {0.600f, 0.350f, 0.400f};
        constexpr float duskHorizon[3] = {0.920f, 0.500f, 0.300f};
        constexpr float duskFog[3] = {0.500f, 0.365f, 0.380f};

        render::WeatherSkyParams params;
        params.weight = 1.0f;
        for (int channel = 0; channel < 3; ++channel) {
            const float baseUpper = std::lerp(nightUpper[channel], dayUpper[channel], daylight);
            const float baseLower = std::lerp(nightLower[channel], dayLower[channel], daylight);
            const float baseHorizon = std::lerp(nightHorizon[channel], dayHorizon[channel], daylight);
            const float baseFog = std::lerp(nightFog[channel], dayFog[channel], daylight);
            const float twilightWeight = twilight * 0.78f;
            params.skyUpper[channel] = std::lerp(baseUpper, duskUpper[channel], twilightWeight);
            params.skyLower[channel] = std::lerp(baseLower, duskLower[channel], twilightWeight);
            params.horizon[channel] = std::lerp(baseHorizon, duskHorizon[channel], twilightWeight);
            params.fogColor[channel] = std::lerp(baseFog, duskFog[channel], twilightWeight);
        }
        params.fogFarDistance = std::lerp(100000.0f, 160000.0f, daylight);
        params.sunGlare = std::lerp(0.55f, 1.0f, daylight);
        m_renderer.setWeatherSky(params);
        return;
    }

    const importer::fnv::FalloutWeatherRecord* weather =
        m_activeWeatherFormId != 0u ? m_weatherTables.findWeather(m_activeWeatherFormId) : nullptr;
    if (weather == nullptr) {
        // No climate names a weather -- every Oblivion worldspace, and any FNV
        // one before the first weather resolves. The sky is left procedural, but
        // the aerial-perspective distance still has to be published or the
        // shader falls back to 15000 units (~214 m) and a city vista renders
        // behind a wall of milk. Weight stays 0, so only the distance is taken.
        static const float s_fogFar = []() {
            const char* env = std::getenv("ODAI_FNV_FOGFAR");
            const float value = env != nullptr ? static_cast<float>(std::atof(env)) : 160000.0f;
            return value > 1.0f ? value : 160000.0f;
        }();
        render::WeatherSkyParams clear;
        clear.weight = 0.0f;
        clear.fogFarDistance = s_fogFar;
        m_renderer.setWeatherSky(clear);
        return;
    }

    // WTHR colours are authored as sRGB bytes for a renderer that displayed them
    // directly. This one is HDR: the frame goes through an ACES curve and auto
    // exposure keyed to a sunlit desert sitting around 0.3 linear.
    //
    // Decoding to linear and stopping there is not enough, and the failure is
    // silent. A heavy-overcast sky is authored sRGB 23,27,30 -- linear 0.0086 --
    // which ACES maps to ~0.002 and the exposure scale then buries. The sky
    // rendered PURE BLACK while the terrain looked correctly lit, because the
    // values are display-referred and were being read as radiance.
    //
    // A flat gain cannot fix this. Measured on two real weathers: the gain that
    // makes a heavy overcast readable (~10) washes a clear zenith from deep blue
    // to pale haze, and the gain that keeps the blue (~3) puts the overcast back
    // at pure black. The pipeline's response below ~0.05 linear is far steeper
    // than the rest of its range, so darks need lifting MORE than brights.
    //
    // pow(linear, contrast) does exactly that, and one exponent covers both
    // cases where no single multiplier does. This is a display-referred fudge,
    // not physics; the principled fix is to invert the tonemap on the GPU, where
    // the auto-exposure scale is actually known. Both knobs are env-tunable.
    static const float s_skyContrast = []() {
        const char* env = std::getenv("ODAI_FNV_SKY_CONTRAST");
        const float value = env != nullptr ? static_cast<float>(std::atof(env)) : 0.60f;
        return value > 0.0f ? value : 0.60f;
    }();
    static const float s_skyGain = []() {
        const char* env = std::getenv("ODAI_FNV_SKY_GAIN");
        const float value = env != nullptr ? static_cast<float>(std::atof(env)) : 1.6f;
        return value > 0.0f ? value : 1.6f;
    }();
    // Enhanced Shaders runs 1.25 by day and 0.9 at night against these same
    // weather records; 1.15 is a compromise for a single global value.
    static const float s_skySaturation = []() {
        const char* env = std::getenv("ODAI_FNV_SKY_SATURATION");
        const float value = env != nullptr ? static_cast<float>(std::atof(env)) : 1.15f;
        return value > 0.0f ? value : 1.15f;
    }();
    const auto decode = [&](const importer::fnv::FalloutColorRgb& color, float* out) {
        const auto channel = [](std::uint8_t value) {
            const float srgb = static_cast<float>(value) / 255.0f;
            return srgb <= 0.04045f ? (srgb / 12.92f)
                                    : std::pow((srgb + 0.055f) / 1.055f, 2.4f);
        };
        // Shape MAGNITUDE, keep HUE. Applying pow() per channel (which this
        // did first) pulls the channels toward each other, so an exponent
        // below 1 desaturates: it lifted the overcast greys correctly and
        // simultaneously washed a clear zenith from deep blue to pale haze.
        //
        // Splitting magnitude from direction is how ENB's tonemap does it
        // (Enhanced Shaders' enbeffect.fx: contrast on `color/normalize(color)`,
        // saturation on `normalize(color)`), and it fixes the hue shift for the
        // same reason -- the direction vector is left alone unless saturation
        // explicitly touches it.
        const float linear[3] = {channel(color.r), channel(color.g), channel(color.b)};
        const float magnitude =
            std::sqrt((linear[0] * linear[0]) + (linear[1] * linear[1]) + (linear[2] * linear[2]));
        if (magnitude <= 1e-6f) {
            out[0] = out[1] = out[2] = 0.0f;
            return;
        }
        const float shaped = std::pow(magnitude, s_skyContrast) * s_skyGain;
        for (int i = 0; i < 3; ++i) {
            // pow on the unit direction is ENB's saturation control; above 1
            // pushes the dominant channel further ahead of the others.
            out[i] = std::pow(linear[i] / magnitude, s_skySaturation) * shaped;
        }
    };

    using importer::fnv::FalloutWeatherColor;
    const float hour = m_timeOfDayHours;
    // The climate's own dawn and dusk, not the sampler's 6/19 defaults; see
    // m_sunriseHour. Threaded through EVERY sample below, because two samples
    // taken against different day curves disagree about which slot it is --
    // the sky would reach its sunset colour while the clouds were still on Day.
    const float dawn = m_sunriseHour;
    const float dusk = m_sunsetHour;
    const auto skyColor = [&](FalloutWeatherColor channel) {
        return sampleFalloutWeatherColor(*weather, channel, hour, dawn, dusk);
    };
    render::WeatherSkyParams params;
    params.weight = 1.0f;
    decode(skyColor(FalloutWeatherColor::SkyUpper), params.skyUpper);
    decode(skyColor(FalloutWeatherColor::SkyLower), params.skyLower);
    decode(skyColor(FalloutWeatherColor::Horizon), params.horizon);
    // Aerial perspective is the FAR haze, so a record that separates the two
    // (Skyrim) must hand over the far colour; channel 1 is its near fog and is
    // a saturated tint meant for the first few metres.
    if (weather->hasFogFarColor) {
        decode(sampleFalloutWeatherColorRow(weather->fogFarColors, hour, dawn, dusk),
               params.fogColor);
    } else {
        decode(skyColor(FalloutWeatherColor::Fog), params.fogColor);
    }
    if (m_streamIsSkyrim) {
        // `decode` lifts display-referred WTHR bytes enough for an emissive HDR
        // sky to survive exposure and ACES. Fog is not emissive: feeding that
        // same lifted value into both aerial perspective and volumetric fog
        // lights distant geometry twice, and Skyrim's pale blue far-fog turns
        // Whiterun into a cyan silhouette. Keep the authored hue, but compress
        // and neutralise the atmospheric stops before they become incident
        // radiance. Clouds retain their separate authored tints below.
        const auto restrainAtmosphere = [](float* color, float neutralAmount, float gain) {
            const float luma =
                (color[0] * 0.2126f) + (color[1] * 0.7152f) + (color[2] * 0.0722f);
            for (int channel = 0; channel < 3; ++channel) {
                color[channel] = std::lerp(color[channel], luma, neutralAmount) * gain;
            }
        };
        restrainAtmosphere(params.skyUpper, 0.45f, 0.85f);
        restrainAtmosphere(params.skyLower, 0.50f, 0.82f);
        restrainAtmosphere(params.horizon, 0.55f, 0.80f);
        restrainAtmosphere(params.fogColor, 0.58f, 0.72f);
    }
    // Day fog until dusk, night fog after; the record authors the two
    // separately and there is no third value to interpolate toward.
    const bool daytime = hour >= dawn && hour < dusk;
    params.fogFarDistance = daytime ? weather->fogDayFar : weather->fogNightFar;
    if ((weather->classification & 0x04u) != 0u) {
        const std::string weatherName = toLowerAscii(weather->editorId);
        const bool heavy = weatherName.find("heavy") != std::string::npos ||
            weatherName.find("storm") != std::string::npos ||
            weatherName.find("overcast") != std::string::npos;
        params.precipitationIntensity = heavy ? 1.0f : 0.68f;
    }

    // Sunlight and Ambient light the GROUND. These two channels were read out
    // of every record and then dropped, so a storm rendered as a dark sky over
    // sunlit terrain -- the sky was the only thing the weather touched.
    //
    // Decoded plainly to linear rather than through decode() above: that
    // function's gain and contrast exist to make an EMISSIVE sky readable
    // through an ACES curve, and applying them here would push a light source
    // through a display-referred fudge twice. The renderer takes hue from these
    // and bounds the intensity itself; see WeatherSkyParams::lightingWeight.
    const auto decodeLinear = [](const importer::fnv::FalloutColorRgb& color, float* out) {
        out[0] = srgbChannelToLinear(static_cast<float>(color.r) / 255.0f);
        out[1] = srgbChannelToLinear(static_cast<float>(color.g) / 255.0f);
        out[2] = srgbChannelToLinear(static_cast<float>(color.b) / 255.0f);
    };
    decodeLinear(skyColor(FalloutWeatherColor::Sunlight), params.sunlightColor);
    decodeLinear(skyColor(FalloutWeatherColor::Ambient), params.ambientColor);
    // ODAI_FNV_LIGHT_WEIGHT=0 is the A/B control, and the only way to attribute
    // a brightness change to the weather rather than to the sky gradient.
    static const float s_lightingWeight = []() {
        const char* env = std::getenv("ODAI_FNV_LIGHT_WEIGHT");
        return env != nullptr ? std::clamp(static_cast<float>(std::atof(env)), 0.0f, 1.0f) : 1.0f;
    }();
    params.lightingWeight = s_lightingWeight;
    // DATA's Sun Glare byte, which is the one field in that block the sky can
    // use directly. SkyrimCloudy authors 153 of 255; a fog weather authors far
    // less, and the difference is a sun with a halo against one that is a bare
    // disc in soup.
    params.sunGlare = static_cast<float>(weather->sunGlare) / 255.0f;

    // Cloud tints and opacities come from the layer the slot is DRAWING, not
    // from the slot number -- see m_cloudLayerSource. Both track time of day
    // and are sampled the same way the sky colours are.
    for (int slot = 0; slot < render::kWeatherCloudLayerCount; ++slot) {
        // ODAI_FNV_NOCLOUDS isolates the sky gradient from the cloud layers.
        // Worth keeping: "the sky is black" has two very different causes
        // (an authored-dark gradient vs. total cloud cover) and they are
        // indistinguishable on screen.
        static const bool s_noClouds = std::getenv("ODAI_FNV_NOCLOUDS") != nullptr;
        const int source = m_cloudLayerSource[slot];
        if (source < 0 || static_cast<std::size_t>(source) >= weather->cloudLayers.size() ||
            s_noClouds) {
            params.cloudOpacity[slot] = 0.0f;
            continue;
        }
        const importer::fnv::FalloutWeatherCloudLayer& layer =
            weather->cloudLayers[static_cast<std::size_t>(source)];
        decode(sampleFalloutWeatherCloudTint(layer, hour, dawn, dusk), params.cloudTint[slot]);
        if (m_streamIsSkyrim) {
            // Keep cloud whites neutral under overcast records. The source tint
            // still controls brightness; only the exaggerated HDR chroma is
            // pulled back here.
            float* tint = params.cloudTint[slot];
            const float luma =
                (tint[0] * 0.2126f) + (tint[1] * 0.7152f) + (tint[2] * 0.0722f);
            for (int channel = 0; channel < 3; ++channel) {
                tint[channel] = std::lerp(tint[channel], luma, 0.45f);
            }
        }
        // JNAM, where the record authors one. Skyrim holds its fully-opaque
        // 32x32 fill swatch at 0.4-0.5 here; drawn at 1.0 that layer is a coat
        // of paint over the whole sky.
        params.cloudOpacity[slot] = sampleFalloutWeatherCloudAlpha(layer, hour, dawn, dusk);
    }
    // One line per weather change, not per frame: "the sky is black" is
    // otherwise indistinguishable from "the sky is not being set at all".
    static std::uint32_t s_loggedWeather = 0;
    if (s_loggedWeather != m_activeWeatherFormId) {
        s_loggedWeather = m_activeWeatherFormId;
        VOX_LOGI("newvegas") << "sky linear rgb: upper(" << params.skyUpper[0] << ","
                             << params.skyUpper[1] << "," << params.skyUpper[2] << ") horizon("
                             << params.horizon[0] << "," << params.horizon[1] << ","
                             << params.horizon[2] << ") fog(" << params.fogColor[0] << ","
                             << params.fogColor[1] << "," << params.fogColor[2]
                             << ") fogFar=" << params.fogFarDistance
                             << " weight=" << params.weight;
    }
    m_renderer.setWeatherSky(params);
}

namespace {

// The tour, in engine space: where the camera is and what it is pointed at.
//
// THESE ARE THE TOWN'S OWN LANDMARKS, not invented viewpoints. Each look-at is
// the doorstep of a named interior cell, read out of the plugin by spawning at
// it (--spawn GSProspectorSaloonInterior and friends print the position), so
// the tour is aimed at the buildings Goodsprings actually has rather than at
// coordinates that looked good once and drift the moment anything moves.
//
//   Prospector Saloon      (-67452, 8472, -4900)
//   General Store          (-69319, 8501, -3528)
//   Gas Station            (-75169, 8880, -4076)
//   Doc Mitchell's House   (-73163, 8806, -1312)
//   Schoolhouse            (-74482, 8354,  4780)
//   Victor's Shack         (-72319, 8440,  5928)
//
// Camera heights are absolute rather than ground-relative: the tour is a drone
// shot over a valley whose floor moves 500 units under it, and clamping to the
// terrain would make the camera bob over every rise it crossed.
struct FlyWaypoint {
    float position[3];
    float lookAt[3];
};

// A tour loaded from disk, when --tour-file names one. Empty means the built-in
// Goodsprings list below.
//
// A file rather than more hardcoded arrays because framing a flythrough is
// iterative -- every waypoint is a guess until you watch it -- and a rebuild per
// guess makes that loop useless. It is also what lets one binary tour three
// different games.
std::vector<FlyWaypoint> g_runtimeTour;

constexpr FlyWaypoint kGoodspringsTour[] = {
    // High and south, the whole town in frame.
    {{-70600.0f, 10600.0f, -10200.0f}, {-70600.0f, 8700.0f, -4400.0f}},
    // Down toward the saloon, the first building anyone sees.
    {{-69100.0f,  9500.0f,  -7200.0f}, {-67452.0f, 8620.0f, -4900.0f}},
    // Low across the saloon front, turning onto Easy Pete's spot beside it.
    {{-67500.0f,  8830.0f,  -5300.0f}, {-67845.0f, 8480.0f, -3334.0f}},
    // The general store, on the corner of the main road.
    {{-68400.0f,  8980.0f,  -4100.0f}, {-69319.0f, 8620.0f, -3528.0f}},
    // Along the road toward Doc Mitchell's, Victor parked outside it.
    {{-71400.0f,  9080.0f,  -2600.0f}, {-72943.0f, 8780.0f, -1092.0f}},
    // Over the spawn, then north up the rise to the schoolhouse.
    {{-73200.0f,  9100.0f,   -200.0f}, {-74482.0f, 8560.0f,  4780.0f}},
    {{-73900.0f,  9050.0f,   2900.0f}, {-74482.0f, 8500.0f,  4780.0f}},
    // Victor's shack at the north end, then the turn back south.
    {{-73000.0f,  8900.0f,   4900.0f}, {-72319.0f, 8560.0f,  5928.0f}},
    {{-71200.0f,  8980.0f,   4400.0f}, {-68000.0f, 8500.0f,  2200.0f}},
    // Down to head height over the east end of town, which is where the people
    // are: the five Powder Gangers stand around (-66131, 1645), with a settler
    // and the bighorners between there and the road.
    {{-68900.0f,  8760.0f,   2300.0f}, {-66991.0f, 8300.0f,  1981.0f}},
    {{-67400.0f,  8600.0f,   2350.0f}, {-66500.0f, 8350.0f,  1900.0f}},
    {{-65700.0f,  8700.0f,   2050.0f}, {-65300.0f, 8420.0f,  1500.0f}},
};

// Where the tour stops aiming at coordinates and starts aiming at whoever is
// there. The last waypoints are over the east end of town, and the people
// standing there WANDER -- up to 950 units from where the plugin put them --
// so a fixed look-at is a coin flip that lands on empty dirt about as often as
// it lands on a person. See updateFlythrough.
constexpr float kTourActorTrackStart = 0.72f;
constexpr float kTourActorTrackFull = 0.86f;
constexpr int kBuiltinTourCount =
    static_cast<int>(sizeof(kGoodspringsTour) / sizeof(kGoodspringsTour[0]));

// True once --tour-file has replaced the built-in path. The camera treats an
// authored tour as authoritative: the actor hand-off below is a flourish
// written for Goodsprings, where the tour ends among the townspeople on
// purpose, and it silently overrides whatever a tour file aimed at. In Megaton
// that meant a pan across the shanties turned into a top-down stare at a
// settler standing on the crater floor.
bool tourIsAuthored() {
    return !g_runtimeTour.empty();
}

int tourCount() {
    return g_runtimeTour.empty() ? kBuiltinTourCount
                                 : static_cast<int>(g_runtimeTour.size());
}

// Critically damped smoothing toward a target, with a ceiling on rate of
// change. The standard spring-damper solution with the exponential replaced by
// its Pade approximation, so it is stable at any timestep -- an explicit spring
// integrated with a 40 ms frame overshoots and rings.
//
// `smoothSeconds` is roughly how long it takes to cover the distance, not a
// half-life. `velocity` is carried by the caller because that state is what
// makes it critically damped rather than exponential.
float smoothDampAngle(float current, float target, float& velocity, float smoothSeconds,
                      float maxRatePerSecond, float deltaSeconds) {
    if (deltaSeconds <= 0.0f) {
        return current;
    }
    const float omega = 2.0f / std::max(smoothSeconds, 1e-4f);
    const float x = omega * deltaSeconds;
    const float decay = 1.0f / (1.0f + x + (0.48f * x * x) + (0.235f * x * x * x));
    // Clamping the DISTANCE rather than the step is what makes the rate ceiling
    // behave: the filter then eases toward a target it is allowed to reach,
    // instead of being clipped every frame and arriving with a corner.
    const float maxDistance = maxRatePerSecond * smoothSeconds;
    const float change = std::clamp(current - target, -maxDistance, maxDistance);
    const float clampedTarget = current - change;
    const float temp = (velocity + (omega * change)) * deltaSeconds;
    velocity = (velocity - (omega * temp)) * decay;
    float result = clampedTarget + ((change + temp) * decay);
    // Do not overshoot past the target from the wrong side.
    if (((target - current) > 0.0f) == (result > target)) {
        result = target;
        velocity = (result - target) / deltaSeconds;
    }
    return result;
}

// Duplicate the ends rather than wrapping: this is a path, not a loop, and
// wrapping would curve the first segment toward the last landmark.
const FlyWaypoint& tourWaypoint(int index) {
    const int clamped = std::clamp(index, 0, tourCount() - 1);
    return g_runtimeTour.empty() ? kGoodspringsTour[clamped] : g_runtimeTour[clamped];
}

// CENTRIPETAL Catmull-Rom (Barry-Goldman form, alpha = 0.5), not the uniform
// one this used to use.
//
// Uniform Catmull-Rom takes the tangent at p1 as (p2 - p0)/2 regardless of how
// far apart those points actually are. These waypoints are not evenly spaced --
// the legs run from ~1500 to ~3400 units -- so the curve arrives at a knot with
// one speed and leaves it with another, and every waypoint is a visible kink.
// Uneven spacing also lets the uniform form overshoot and, at a tight corner,
// cusp: the camera briefly reverses. Parameterizing the knots by sqrt(chord
// length) is the standard fix and is guaranteed cusp- and self-intersection-free.
//
// Knot spacings are clamped away from zero because the ends are duplicated, so
// the first and last spans have zero chord length and would divide by it.
void centripetalKnots(const float p0[3], const float p1[3], const float p2[3],
                      const float p3[3], float outKnots[4]) {
    const auto span = [](const float a[3], const float b[3]) {
        const float dx = b[0] - a[0];
        const float dy = b[1] - a[1];
        const float dz = b[2] - a[2];
        return std::max(std::sqrt(std::sqrt((dx * dx) + (dy * dy) + (dz * dz))), 1e-4f);
    };
    outKnots[0] = 0.0f;
    outKnots[1] = outKnots[0] + span(p0, p1);
    outKnots[2] = outKnots[1] + span(p1, p2);
    outKnots[3] = outKnots[2] + span(p2, p3);
}

// One de-Boor-style pyramid step: linear blend of a and b over [ta, tb] at t.
void knotLerp(const float a[3], const float b[3], float ta, float tb, float t, float out[3]) {
    const float denominator = (tb - ta);
    const float w = (std::abs(denominator) < 1e-6f) ? 0.0f : ((t - ta) / denominator);
    for (int axis = 0; axis < 3; ++axis) {
        out[axis] = a[axis] + ((b[axis] - a[axis]) * w);
    }
}

// Evaluate the curve through p1..p2 at local parameter s in [0,1], using a knot
// vector supplied by the caller. The look-at spline is deliberately evaluated
// against the POSITION knots rather than its own: the two must stay paired
// frame for frame, and a look-at sequence has repeated entries (waypoints 5 and
// 6 share one) whose own centripetal knots would advance at a different rate.
void evaluateCentripetal(const float p0[3], const float p1[3], const float p2[3],
                         const float p3[3], const float knots[4], float s, float out[3]) {
    const float t = knots[1] + ((knots[2] - knots[1]) * s);
    float a1[3];
    float a2[3];
    float a3[3];
    knotLerp(p0, p1, knots[0], knots[1], t, a1);
    knotLerp(p1, p2, knots[1], knots[2], t, a2);
    knotLerp(p2, p3, knots[2], knots[3], t, a3);
    float b1[3];
    float b2[3];
    knotLerp(a1, a2, knots[0], knots[2], t, b1);
    knotLerp(a2, a3, knots[1], knots[3], t, b2);
    knotLerp(b1, b2, knots[1], knots[2], t, out);
}

// Sample at a parameter measured in SEGMENTS, i.e. u in [0,1] spans the whole
// waypoint list with each leg getting an equal slice of u regardless of length.
void sampleTourByParameter(float u, float outPosition[3], float outLookAt[3]) {
    const float span = static_cast<float>(tourCount() - 1);
    const float scaled = std::clamp(u, 0.0f, 1.0f) * span;
    const int segment = std::min(static_cast<int>(scaled), tourCount() - 2);
    const float s = scaled - static_cast<float>(segment);
    const FlyWaypoint& w0 = tourWaypoint(segment - 1);
    const FlyWaypoint& w1 = tourWaypoint(segment);
    const FlyWaypoint& w2 = tourWaypoint(segment + 1);
    const FlyWaypoint& w3 = tourWaypoint(segment + 2);
    float knots[4];
    centripetalKnots(w0.position, w1.position, w2.position, w3.position, knots);
    evaluateCentripetal(w0.position, w1.position, w2.position, w3.position, knots, s, outPosition);
    evaluateCentripetal(w0.lookAt, w1.lookAt, w2.lookAt, w3.lookAt, knots, s, outLookAt);
}

// Arc-length reparameterization.
//
// Equal u per leg means equal TIME per leg, and the legs differ in length by
// better than 2x -- so the camera visibly speeds up over the long run to Doc
// Mitchell's and crawls across the short hops at the east end. Constant ground
// speed is what reads as a smooth dolly, so the eased parameter below is a
// distance along the path and this table converts it back to a curve parameter.
constexpr int kTourArcSamples = 1024;

// Rebuilt whenever the loaded tour changes rather than cached forever.
//
// This was a function-local `static const` built on first call, which is
// correct only because --tour-file happens to be parsed before the first frame.
// Any runtime tour swap would keep the OLD path's arc-length table and silently
// reparameterize the new curve by the old one's distances -- a camera that
// speeds up and slows down at the previous tour's waypoints, which is a very
// hard symptom to attribute back to here. Keying on the waypoint count and
// first waypoint costs one comparison per frame and removes the trap.
const std::vector<float>& tourArcLengths() {
    static std::vector<float> table;
    static int builtForWaypointCount = -1;
    static FlyWaypoint builtForFirstWaypoint{};
    const int waypointCount = tourCount();
    const FlyWaypoint firstWaypoint = tourWaypoint(0);
    const bool stale =
        builtForWaypointCount != waypointCount ||
        builtForFirstWaypoint.position[0] != firstWaypoint.position[0] ||
        builtForFirstWaypoint.position[1] != firstWaypoint.position[1] ||
        builtForFirstWaypoint.position[2] != firstWaypoint.position[2];
    if (!stale) {
        return table;
    }
    builtForWaypointCount = waypointCount;
    builtForFirstWaypoint = firstWaypoint;
    table = [&]() {
        std::vector<float> lengths(kTourArcSamples + 1, 0.0f);
        float previous[3] = {};
        float ignoredLookAt[3] = {};
        sampleTourByParameter(0.0f, previous, ignoredLookAt);
        for (int i = 1; i <= kTourArcSamples; ++i) {
            float current[3] = {};
            sampleTourByParameter(static_cast<float>(i) / static_cast<float>(kTourArcSamples),
                                  current, ignoredLookAt);
            const float dx = current[0] - previous[0];
            const float dy = current[1] - previous[1];
            const float dz = current[2] - previous[2];
            lengths[i] = lengths[i - 1] + std::sqrt((dx * dx) + (dy * dy) + (dz * dz));
            previous[0] = current[0];
            previous[1] = current[1];
            previous[2] = current[2];
        }
        const float total = lengths.back();
        if (total > 1e-3f) {
            for (float& entry : lengths) {
                entry /= total;
            }
        }
        return lengths;
    }();
    return table;
}

// distance in [0,1] along the path -> curve parameter in [0,1].
float tourParameterAtDistance(float distance) {
    const std::vector<float>& lengths = tourArcLengths();
    const float target = std::clamp(distance, 0.0f, 1.0f);
    const auto upper = std::upper_bound(lengths.begin(), lengths.end(), target);
    if (upper == lengths.begin()) {
        return 0.0f;
    }
    if (upper == lengths.end()) {
        return 1.0f;
    }
    const auto lower = upper - 1;
    const float lowerValue = *lower;
    const float upperValue = *upper;
    const float denominator = upperValue - lowerValue;
    const float fraction = (denominator > 1e-9f) ? ((target - lowerValue) / denominator) : 0.0f;
    const float index = static_cast<float>(lower - lengths.begin()) + fraction;
    return index / static_cast<float>(kTourArcSamples);
}

void sampleTour(float distance, float outPosition[3], float outLookAt[3]) {
    sampleTourByParameter(tourParameterAtDistance(distance), outPosition, outLookAt);
}

}  // namespace

// "px py pz  lx ly lz" per line; '#' comments and blank lines ignored. Returns
// how many waypoints were loaded so a typo in the path is not silently a
// built-in Goodsprings tour of a Fallout 3 worldspace.
int loadTourFile(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        return 0;
    }
    std::vector<FlyWaypoint> loaded;
    std::string line;
    while (std::getline(input, line)) {
        const std::size_t hash = line.find('#');
        if (hash != std::string::npos) {
            line.resize(hash);
        }
        FlyWaypoint waypoint{};
        std::istringstream stream(line);
        if (stream >> waypoint.position[0] >> waypoint.position[1] >> waypoint.position[2] >>
            waypoint.lookAt[0] >> waypoint.lookAt[1] >> waypoint.lookAt[2]) {
            loaded.push_back(waypoint);
        }
    }
    // Catmull-Rom needs four control points; fewer than four cannot describe a
    // curve and would index past the ends.
    if (loaded.size() < 4u) {
        return 0;
    }
    g_runtimeTour = std::move(loaded);
    return static_cast<int>(g_runtimeTour.size());
}

bool BethesdaApp::updateFlythrough(float deltaSeconds) {
    m_flythroughTime += deltaSeconds;
    const float raw = std::clamp(m_flythroughTime / m_flythroughSeconds, 0.0f, 1.0f);
    // Ease only the ENDS, at constant speed in between.
    //
    // A smoothstep over the whole path is the obvious thing and it is wrong for
    // a tour: it makes the middle -- where all the landmarks are -- rush past
    // at nearly double speed while the first and last waypoints get a third of
    // the running time each. This is the integral of a speed profile that ramps
    // up over the first `kEase` of the run, holds, and ramps back down.
    //
    // THE RAMP-DOWN BRANCH WAS DIVIDING THE WRONG NUMERATOR. The profile's area
    // up to `raw` is `total - remaining^2 / (2 * kEase)`, and it was written as
    // `1 - remaining^2 / (2 * kEase)` -- larger by exactly kEase. So at
    // raw = 1 - kEase the eased parameter jumped from 0.962 straight to 1.0 and
    // the std::min pinned it there: the camera SNAPPED forward over the last
    // stretch of path and then sat frozen for the final 7% of the run, while
    // the actor tracking below went on swinging the aim around. That is the
    // "smooth until the end, then jittery" this had.
    constexpr float kEase = 0.07f;
    const float total = 1.0f - kEase;  // area under that profile
    float eased = raw;
    if (raw < kEase) {
        eased = ((raw * raw) / (2.0f * kEase)) / total;
    } else if (raw > 1.0f - kEase) {
        const float remaining = 1.0f - raw;
        eased = (total - ((remaining * remaining) / (2.0f * kEase))) / total;
    } else {
        eased = ((kEase * 0.5f) + (raw - kEase)) / total;
    }
    eased = std::clamp(eased, 0.0f, 1.0f);

    float position[3] = {};
    float lookAt[3] = {};
    sampleTour(eased, position, lookAt);
    m_cameraX = position[0];
    m_cameraY = position[1];
    m_cameraZ = position[2];

    // Hand the aim over to a real inhabitant for the last stretch. Nearest
    // walker to the camera, aimed at the chest rather than the feet, blended in
    // so the camera drifts onto them instead of snapping.
    if (eased > kTourActorTrackStart && !tourIsAuthored()) {
        // LATCHED, not re-chosen every frame. Picking the nearest actor afresh
        // each frame makes the aim jump the moment two of them swap places in
        // the ordering, and again whenever one crosses the near or far cutoff --
        // the target teleports across the width of the town between one frame
        // and the next. Once someone is chosen the tour stays with them, which
        // is also what a real camera operator would do.
        if (m_tourTrackedActor >= 0 &&
            (static_cast<std::size_t>(m_tourTrackedActor) >= m_actors.size() ||
             !m_actors[static_cast<std::size_t>(m_tourTrackedActor)].placed)) {
            m_tourTrackedActor = -1;
        }
        if (m_tourTrackedActor < 0) {
            float bestDistanceSq = 4000.0f * 4000.0f;
            for (std::size_t i = 0; i < m_actors.size(); ++i) {
                const SkinnedActor& actor = m_actors[i];
                if (!actor.placed || !actor.wanders) {
                    continue;
                }
                const float dx = actor.position[0] - m_cameraX;
                const float dz = actor.position[2] - m_cameraZ;
                const float distanceSq = (dx * dx) + (dz * dz);
                // Not the one under the camera's nose: at a few hundred units an
                // actor fills the frame and the town behind them is gone.
                //
                // 1200, not the 300 this had. The tour flies ~500 units above
                // the townspeople, so 300 units of GROUND distance is a 59
                // degree downward aim -- and since the camera is still moving
                // toward them, it then passes very nearly overhead. Aiming
                // through the nadir makes yaw ill-conditioned: the measured
                // trace swung 140 degrees of yaw in 0.3 s at raw 0.79 while the
                // target itself barely moved. 1200 units keeps the aim under
                // ~23 degrees down and the pass off to one side.
                if (distanceSq < 1200.0f * 1200.0f || distanceSq >= bestDistanceSq) {
                    continue;
                }
                bestDistanceSq = distanceSq;
                m_tourTrackedActor = static_cast<int>(i);
            }
        }
        if (m_tourTrackedActor >= 0) {
            const SkinnedActor& tracked = m_actors[static_cast<std::size_t>(m_tourTrackedActor)];
            const float weight = std::clamp(
                (eased - kTourActorTrackStart) / (kTourActorTrackFull - kTourActorTrackStart),
                0.0f, 1.0f);
            const float smooth = weight * weight * (3.0f - (2.0f * weight));
            const float target[3] = {
                tracked.position[0],
                tracked.position[1] + conversationFaceHeight(tracked),
                tracked.position[2]};
            // Low-pass the aim point before it reaches the camera.
            //
            // An actor is re-settled onto the terrain EVERY frame (see
            // updateActorWandering) and slid out of walls on top of that, so
            // their position carries per-frame steps that a fixed look-at
            // never had. Pointing straight at it hands that noise to the
            // camera's pitch and yaw, which is the shake at the end of the
            // tour. Time-constant form so it behaves the same at any frame rate.
            constexpr float kAimTimeConstantSeconds = 0.35f;
            const float alpha =
                1.0f - std::exp(-std::max(deltaSeconds, 0.0f) / kAimTimeConstantSeconds);
            if (!m_tourAimValid) {
                m_tourAim[0] = target[0];
                m_tourAim[1] = target[1];
                m_tourAim[2] = target[2];
                m_tourAimValid = true;
            }
            for (int axis = 0; axis < 3; ++axis) {
                m_tourAim[axis] += (target[axis] - m_tourAim[axis]) * alpha;
                lookAt[axis] += (m_tourAim[axis] - lookAt[axis]) * smooth;
            }
        }
    }

    const float dx = lookAt[0] - m_cameraX;
    const float dy = lookAt[1] - m_cameraY;
    const float dz = lookAt[2] - m_cameraZ;
    const float horizontal = std::sqrt((dx * dx) + (dz * dz));
    if (horizontal > 1e-3f) {
        const float desiredYaw = std::atan2(dz, dx) * (180.0f / kPi);
        const float desiredPitch = std::clamp(
            std::atan2(dy, horizontal) * (180.0f / kPi), -kPitchLimitDegrees, kPitchLimitDegrees);

        // Snapping the camera onto the aim direction makes the ANGLES only as
        // smooth as the geometry that produced them, and near the nadir that is
        // not smooth at all: a look-at a few hundred units away and several
        // hundred below turns yaw into a badly conditioned function of the
        // target's horizontal position. Smoothing the aim POINT does not help
        // there, because the point is barely moving -- the angle is what
        // explodes.
        //
        // So the last step of the tour camera is a critically damped filter on
        // the angles themselves, with a hard ceiling on turn rate. Critically
        // damped rather than exponential because it is C1: it eases out of a
        // turn instead of arriving with a corner in angular velocity. The rate
        // ceiling is what bounds a whip if the geometry ever goes bad again --
        // the camera lags the target for a moment and catches up, which reads
        // as a camera operator, not as a glitch.
        constexpr float kAngleSmoothSeconds = 0.25f;
        constexpr float kMaxYawRateDegreesPerSecond = 90.0f;
        constexpr float kMaxPitchRateDegreesPerSecond = 60.0f;
        if (!m_tourAnglesValid) {
            m_yawDegrees = desiredYaw;
            m_pitchDegrees = desiredPitch;
            m_tourYawVelocity = 0.0f;
            m_tourPitchVelocity = 0.0f;
            m_tourAnglesValid = true;
        } else {
            // Shortest arc: the tour crosses +/-180 and chasing the long way
            // round would be a full spin.
            float yawTarget = desiredYaw;
            while (yawTarget - m_yawDegrees > 180.0f) {
                yawTarget -= 360.0f;
            }
            while (yawTarget - m_yawDegrees < -180.0f) {
                yawTarget += 360.0f;
            }
            m_yawDegrees = smoothDampAngle(
                m_yawDegrees, yawTarget, m_tourYawVelocity, kAngleSmoothSeconds,
                kMaxYawRateDegreesPerSecond, deltaSeconds);
            m_pitchDegrees = std::clamp(
                smoothDampAngle(m_pitchDegrees, desiredPitch, m_tourPitchVelocity,
                                kAngleSmoothSeconds, kMaxPitchRateDegreesPerSecond, deltaSeconds),
                -kPitchLimitDegrees, kPitchLimitDegrees);
        }
    }

    // ODAI_FNV_TOUR_TRACE=<path> writes one CSV row per tour frame.
    //
    // "Is the camera smooth" cannot be answered from a screenshot and is only
    // half-answerable by watching -- a kink at a waypoint and a shake in the
    // aim look alike at speed and have different causes. The trace makes both
    // measurable: differentiate position for speed, and yaw/pitch twice for the
    // angular jerk that reads as jitter.
    if (const char* tracePath = std::getenv("ODAI_FNV_TOUR_TRACE")) {
        static std::ofstream s_trace(tracePath);
        static bool s_header = false;
        if (!s_header) {
            s_trace << "raw,eased,x,y,z,yaw,pitch\n";
            s_header = true;
        }
        s_trace << raw << ',' << eased << ',' << m_cameraX << ',' << m_cameraY << ','
                << m_cameraZ << ',' << m_yawDegrees << ',' << m_pitchDegrees << '\n';
    }
    return raw < 1.0f;
}

void BethesdaApp::updateCamera(float deltaSeconds) {
    // The scripted tour owns the camera outright -- no input, no collision, no
    // ground clamp. It flies over rooftops on purpose.
    if (m_flythroughSeconds > 0.0f) {
        // Stream, TAA and exposure warm up on the exact first authored pose.
        // Advancing here before frame capture begins used to consume the start
        // of the tour and leave an equally long frozen tail at the endpoint.
        const bool captureWaiting =
            (!m_captureVideoPath.empty() || !m_captureDirectory.empty()) && !m_captureStarted;
        updateFlythrough(captureWaiting ? 0.0f : deltaSeconds);
        return;
    }

    // ODAI_FNV_BENCH=1 walks the camera forward on a slow turn instead of
    // reading input. "It is jittery when I move" is not reproducible from a
    // standing start, and a hand-driven walk is not comparable between runs --
    // this makes the motion identical every time so a frame-time change is
    // attributable to the code rather than to how the tester moved.
    static const bool s_bench = std::getenv("ODAI_FNV_BENCH") != nullptr;
    if (s_bench) {
        // ODAI_FNV_BENCH_FIXED_DT=1 advances by a FIXED step instead of real
        // elapsed time, which makes frame N land at exactly the same camera
        // position on every run. That is what lets two captures taken one frame
        // apart be compared: without it the walk depends on how fast the
        // machine happened to render, and any diff is dominated by the camera
        // having moved a different distance.
        static const bool s_fixedDt = std::getenv("ODAI_FNV_BENCH_FIXED_DT") != nullptr;
        // ODAI_FNV_BENCH_SPEED overrides the walk speed. A very low value is
        // what isolates temporal shimmer: with the camera barely moving between
        // two frames, anything that still differs is the renderer being
        // unstable rather than the world going past. Turn rate scales with it
        // so a slow walk is also a slow turn.
        static const float s_benchSpeed = []() {
            const char* env = std::getenv("ODAI_FNV_BENCH_SPEED");
            return env != nullptr ? static_cast<float>(std::atof(env)) : 400.0f;
        }();
        // ODAI_FNV_BENCH_TURN is the turn rate in degrees/second at the
        // default speed; 0 walks a straight line.
        //
        // The default 6 deg/s is a CIRCLE of radius speed/turn -- about 3820
        // units at 400 u/s, which is smaller than one 4096-unit exterior cell.
        // That is fine for measuring steady-state rendering (the point it was
        // written for) and useless for measuring streaming: the walk never
        // leaves the cells it started resident in, so no cell is ever loaded or
        // evicted mid-run. Testing traversal means setting this to 0.
        static const float s_benchTurn = []() {
            const char* env = std::getenv("ODAI_FNV_BENCH_TURN");
            return env != nullptr ? static_cast<float>(std::atof(env)) : 6.0f;
        }();
        // ODAI_FNV_BENCH_HEADING picks the initial compass direction, applied
        // once, so a straight-line run can be aimed at a specific neighbour
        // region instead of wherever spawn happened to face.
        static const float s_benchHeading = []() {
            const char* env = std::getenv("ODAI_FNV_BENCH_HEADING");
            return env != nullptr ? static_cast<float>(std::atof(env))
                                  : std::numeric_limits<float>::quiet_NaN();
        }();
        if (!std::isnan(s_benchHeading) && !m_benchHeadingApplied) {
            m_yawDegrees = s_benchHeading;
            m_benchHeadingApplied = true;
        }
        const float step = s_fixedDt ? (1.0f / 60.0f) : deltaSeconds;
        m_yawDegrees += (s_benchSpeed / 400.0f) * s_benchTurn * step;
        const float yawRadians = m_yawDegrees * (kPi / 180.0f);
        const float kBenchSpeed = s_benchSpeed;  // default ~5.7 m/s, a fast jog
        m_cameraX += std::cos(yawRadians) * kBenchSpeed * step;
        m_cameraZ += std::sin(yawRadians) * kBenchSpeed * step;
        float groundHeight = 0.0f;
        if (groundHeightAt(m_cameraX, m_cameraZ, groundHeight)) {
            m_cameraY = groundHeight + kEyeHeightUnits;
        }
        return;
    }

    // Mouselook from raw cursor deltas; GameApp has put the cursor in
    // GLFW_CURSOR_DISABLED mode so it reports unbounded relative motion.
    double cursorX = 0.0;
    double cursorY = 0.0;
    glfwGetCursorPos(m_window, &cursorX, &cursorY);
    // A screenshot run must not mouselook. The cursor sits wherever the desktop
    // left it, so 700 warm-up frames of deltas rotate the camera by an
    // arbitrary amount -- which silently defeats ODAI_FNV_YAW/PITCH and makes
    // two captures of "the same view" incomparable. That cost a bogus A/B.
    const bool suppressMouseLook = !m_screenshotPath.empty();
    // A conversation is MODAL, the way Skyrim's is: while it is up the player
    // neither walks nor looks around, and the camera turns onto the speaker.
    //
    // Mouselook is suppressed rather than the cursor being re-captured, and
    // m_lastCursorX/Y keep being written below either way. That is what makes
    // leaving a conversation seamless: the else-branch clears m_hasCursorSample,
    // so the first frame after the card closes applies no delta at all instead
    // of one worth however far the mouse travelled while it was open.
    const bool inConversation = talkingActor() != nullptr;
    bool playerDead = false;
    if (m_bethesdaSessionConfigured) {
        const bethesda::RuntimeObject* player =
            m_bethesdaSession.world().find(m_bethesdaSession.playerObject());
        playerDead = player != nullptr && player->actorValues.has_value() &&
            player->actorValues->dead;
    }
    const bool giftMenuOpen = m_bethesdaSessionConfigured &&
        !m_bethesdaSession.giftMenuRequests().empty();
    const bool canControlPlayer =
        !inConversation && !m_menuOpen && !giftMenuOpen && !playerDead;
    if (m_mouseCaptured && !suppressMouseLook && !inConversation && !giftMenuOpen) {
        if (m_hasCursorSample) {
            m_yawDegrees += static_cast<float>(cursorX - m_lastCursorX) * kMouseSensitivity;
            m_pitchDegrees -= static_cast<float>(cursorY - m_lastCursorY) * kMouseSensitivity;
            m_pitchDegrees = std::clamp(m_pitchDegrees, -kPitchLimitDegrees, kPitchLimitDegrees);
        }
        m_hasCursorSample = true;
    } else {
        m_hasCursorSample = false;
    }
    m_lastCursorX = cursorX;
    m_lastCursorY = cursorY;

    if (thirdPersonPlayerShowcase()) {
        const bool viewDown = keyDown(m_window, GLFW_KEY_V);
        if (canControlPlayer && viewDown && !m_viewToggleLatch) {
            m_thirdPersonView = !m_thirdPersonView;
            reconstructPlayerCamera(deltaSeconds, true);
        }
        m_viewToggleLatch = viewDown;
        if (canControlPlayer && m_thirdPersonView && m_uiInput.scrollDelta != 0.0f) {
            m_cameraBoomRequested = std::clamp(
                m_cameraBoomRequested - m_uiInput.scrollDelta * 24.0f,
                90.0f, 520.0f);
        }
    }

    // Turn to the speaker's face and hold there.
    //
    // Aiming at his ORIGIN would point the camera at his wheel: he stands ~187
    // units tall and his feet are the placement. The face screen is the thing a
    // conversation is about, so that is what gets centred.
    //
    // Eased rather than snapped, and re-aimed every frame rather than once on
    // open: a hard cut to a new orientation is disorienting, and he is animated,
    // so a one-shot aim would drift off him as the idle moves him.
    // The dolly. Eased on the way in AND on the way out, so leaving a
    // conversation widens back rather than snapping.
    {
        constexpr float kFovTauSeconds = 0.22f;
        // The constants are horizontal; the eased value is vertical. Converting
        // the TARGET each frame (rather than easing in horizontal and
        // converting after) also means a window resize retargets smoothly
        // instead of stepping.
        int fovFramebufferWidth = 0;
        int fovFramebufferHeight = 0;
        framebufferSize(fovFramebufferWidth, fovFramebufferHeight);
        const float aspectRatio =
            (fovFramebufferWidth > 0 && fovFramebufferHeight > 0)
                ? (static_cast<float>(fovFramebufferWidth) /
                   static_cast<float>(fovFramebufferHeight))
                : (16.0f / 9.0f);
        const float targetFov = verticalFovDegreesFor(
            inConversation ? kConversationHorizontalFovDegrees : kDefaultHorizontalFovDegrees,
            aspectRatio);
        const float blend = 1.0f - std::exp(-deltaSeconds / kFovTauSeconds);
        m_cameraFovDegrees += (targetFov - m_cameraFovDegrees) * blend;
    }

    if (const SkinnedActor* speakingActor = inConversation ? talkingActor() : nullptr) {
        // Aim at a fraction of the actor's OWN height rather than a constant.
        // A placement is at the FEET, so aiming at the origin points the camera
        // at the ground; and a bighorner, a settler and a Securitron are not
        // the same height, so one constant cannot frame all three.
        const float faceHeightUnits = conversationFaceHeight(*speakingActor);
        // Time constant, not a per-frame fraction: a fixed fraction converges
        // at whatever rate the machine happens to render at, so the turn would
        // be visibly faster on a fast GPU.
        constexpr float kAimTauSeconds = 0.12f;
        const float dx = speakingActor->position[0] - m_cameraX;
        const float dy = (speakingActor->position[1] + faceHeightUnits) - m_cameraY;
        const float dz = speakingActor->position[2] - m_cameraZ;
        const float horizontal = std::sqrt((dx * dx) + (dz * dz));
        if (horizontal > 1e-3f) {
            const float desiredYaw = std::atan2(dz, dx) * (180.0f / kPi);

            // Aiming AT his face centres it -- directly behind the card, which
            // is the one thing a conversation must not hide. Skyrim keeps the
            // speaker in frame and puts the words under them, so the camera
            // aims low enough that his face rises to just above the card's top
            // edge.
            //
            // The offset is derived from the projection rather than dialled in
            // by eye: a point f half-heights above centre subtends
            // atan(f * tan(fovY/2)), so the pitch has to come DOWN by that much
            // to lift the face there. A hardcoded degree count would drift the
            // moment the FOV changed.
            int framebufferWidth = 0;
            int framebufferHeight = 0;
            framebufferSize(framebufferWidth, framebufferHeight);
            float pitchOffsetDegrees = 0.0f;
            if (framebufferHeight > 0) {
                const auto heightPx = static_cast<float>(framebufferHeight);
                // Before the card has ever been drawn there is no measured top
                // edge; 0.30 is where a typical four-reply card starts.
                const float panelTopPx =
                    m_dialoguePanelTopPx > 1.0f ? m_dialoguePanelTopPx : (heightPx * 0.30f);
                // This is the FACE CENTRE, not the top of the skull. Putting
                // the centre at 10% left no room for the forehead once the
                // portrait lens narrowed, so the torso/crotch filled the shot
                // while the face was technically aimed just beyond the frame.
                // Reserve a fifth of the image above the centre, while still
                // preferring the clear strip immediately over the dialogue
                // card when that strip is taller.
                const float faceTargetPx =
                    std::max(heightPx * 0.20f, panelTopPx - (heightPx * 0.07f));
                const float halfHeights =
                    std::clamp(((heightPx * 0.5f) - faceTargetPx) / (heightPx * 0.5f), 0.0f, 0.9f);
                // The LIVE fov, not the default: it is easing while this runs,
                // and the offset that lands his face above the card is a
                // function of it. Using the constant here would slide the
                // framing down over the length of the zoom.
                const float halfFovTangent =
                    std::tan((m_cameraFovDegrees * 0.5f) * (kPi / 180.0f));
                pitchOffsetDegrees =
                    std::atan(halfHeights * halfFovTangent) * (180.0f / kPi);
            }
            const float desiredPitch = std::clamp(
                (std::atan2(dy, horizontal) * (180.0f / kPi)) - pitchOffsetDegrees,
                -kPitchLimitDegrees, kPitchLimitDegrees);
            // Shortest way round: without the wrap, turning from 350 to 10
            // degrees takes the camera the long way, a full spin past the
            // world, which reads as the view being thrown rather than turned.
            float yawDelta = std::fmod((desiredYaw - m_yawDegrees) + 540.0f, 360.0f) - 180.0f;
            const float blend = 1.0f - std::exp(-deltaSeconds / kAimTauSeconds);
            m_yawDegrees += yawDelta * blend;
            m_pitchDegrees += (desiredPitch - m_pitchDegrees) * blend;
        }
    }

    // Shallow focus on the speaker, arriving with the dolly.
    //
    // What an 80 mm portrait lens actually does is throw everything off the
    // subject plane out, and that is the half a narrower FOV cannot fake: at
    // 55 degrees the background is merely smaller, not separated. Focus rides
    // the measured distance to Victor's face -- the same point the aim uses --
    // so it stays locked on him rather than on a fixed distance he happens to
    // stand at.
    //
    // The focus RANGE is not the physical depth of field. A real 80 mm at f/2.8
    // on a subject 4.4 m away holds about 24 cm sharp, which here is ~17 units
    // and would blur Victor's own body along with the town. This is the
    // distance over which blur ramps to full BEYOND him, so it is set to keep
    // the robot sharp and take everything past him: ~3 m.
    {
        constexpr float kDofTauSeconds = 0.28f;
        constexpr float kFocusRangeUnits = 220.0f;
        constexpr float kMaxBlurRadiusPixels = 12.0f;
        // Well below the 1.25 diorama default, which stretches the near ramp to
        // ~400 units. Victor is a solid object roughly 100 units deep standing
        // ON the focal plane, so a near ramp as short as the far one blurs his
        // own front along with the ground -- measured: his edge detail dropped
        // 25% at 1.25 and holds at this. Long enough to still take the fence
        // and the dirt the camera is standing over.
        constexpr float kNearBlurScale = 0.55f;

        // ODAI_FNV_DIALOGUE_NODOF=1 keeps the conversation framing -- the aim,
        // the dolly, the modal lock -- and only drops the lens blur. It is the
        // control for measuring the DoF: with the camera pointed anywhere else
        // the same screen crop is not the same content, so a no-conversation
        // capture cannot be the baseline.
        static const bool s_noDialogueDof = std::getenv("ODAI_FNV_DIALOGUE_NODOF") != nullptr;
        const SkinnedActor* speakingActor = talkingActor();
        const bool wantDof = speakingActor != nullptr && !s_noDialogueDof;
        const float easeBlend = 1.0f - std::exp(-deltaSeconds / kDofTauSeconds);
        m_dialogueDofBlend += ((wantDof ? 1.0f : 0.0f) - m_dialogueDofBlend) * easeBlend;

        // Focus on whoever is speaking; with nobody speaking the blend is
        // easing to zero and the distance no longer matters.
        const float faceHeightUnits =
            speakingActor != nullptr ? conversationFaceHeight(*speakingActor) : 0.0f;
        const float dx = (speakingActor != nullptr ? speakingActor->position[0] : m_cameraX) - m_cameraX;
        const float dy =
            ((speakingActor != nullptr ? speakingActor->position[1] : m_cameraY) + faceHeightUnits) -
            m_cameraY;
        const float dz = (speakingActor != nullptr ? speakingActor->position[2] : m_cameraZ) - m_cameraZ;
        const float focusDistance =
            std::max(std::sqrt((dx * dx) + (dy * dy) + (dz * dz)), 1.0f);

        // The renderer's radius is in PIXELS, which is the honest contract for a
        // post-process kernel but means a fixed number is a different-sized
        // blur on every display -- the same shot reads as a strong lens at
        // 1080p and a mild one at 4K. Scale it so the effect is a constant
        // fraction of the image instead, which is what "an 80 mm lens" means
        // to anyone looking at it.
        int dofWidth = 0;
        int dofHeight = 0;
        framebufferSize(dofWidth, dofHeight);
        const float resolutionScale =
            dofHeight > 0 ? (static_cast<float>(dofHeight) / 1080.0f) : 1.0f;

        if (m_dialogueDofBlend > 0.002f) {
            m_renderer.setDepthOfField(
                true, focusDistance, kFocusRangeUnits,
                kMaxBlurRadiusPixels * resolutionScale * m_dialogueDofBlend, kNearBlurScale);
            m_dialogueDofActive = true;
        } else if (m_dialogueDofActive) {
            // Hand it back once, flipping only the enable so anything dialled
            // into the debug sliders survives.
            m_renderer.setDepthOfField(false, focusDistance, kFocusRangeUnits, 0.0f, kNearBlurScale);
            m_dialogueDofActive = false;
        }
    }

    // Must match the renderer's own camera basis exactly, or WASD walks off at
    // an angle to where you are looking. computeCameraForward
    // (render/backend/vulkan/frame_math.h) is:
    //     forward = (cos(yaw)*cos(pitch), sin(pitch), sin(yaw)*cos(pitch))
    // so in the XZ plane forward is (cos(yaw), sin(yaw)) — NOT (sin, -cos),
    // which is that basis rotated 90 degrees and was what this used.
    // Right is forward advanced a quarter turn: (cos(yaw+90), sin(yaw+90)).
    const float yawRadians = m_yawDegrees * (kPi / 180.0f);
    const float forwardX = std::cos(yawRadians);
    const float forwardZ = std::sin(yawRadians);
    const float rightX = -std::sin(yawRadians);
    const float rightZ = std::cos(yawRadians);

    float moveX = 0.0f;
    float moveY = 0.0f;
    float moveZ = 0.0f;
    // Rooted for the conversation. The keys are not merely ignored further
    // down -- they are never read -- so a held W does not accumulate anywhere
    // and release the moment the card closes. Everything below this point
    // still runs: gravity, the terrain pin and the collision push-out all keep
    // working, so a conversation opened while stepping off a kerb still settles
    // the player onto the ground rather than freezing them mid-air.
    if (canControlPlayer) {
        if (keyDown(m_window, GLFW_KEY_W)) { moveX += forwardX; moveZ += forwardZ; }
        if (keyDown(m_window, GLFW_KEY_S)) { moveX -= forwardX; moveZ -= forwardZ; }
        if (keyDown(m_window, GLFW_KEY_D)) { moveX += rightX;   moveZ += rightZ; }
        if (keyDown(m_window, GLFW_KEY_A)) { moveX -= rightX;   moveZ -= rightZ; }
        if (keyDown(m_window, GLFW_KEY_SPACE)) { moveY += 1.0f; }
        if (keyDown(m_window, GLFW_KEY_LEFT_CONTROL)) { moveY -= 1.0f; }
    }

    const float lengthSquared = (moveX * moveX) + (moveZ * moveZ);
    if (lengthSquared > 1e-6f) {
        const float inverseLength = 1.0f / std::sqrt(lengthSquared);
        moveX *= inverseLength;
        moveZ *= inverseLength;
    }

    float speed = kWalkUnitsPerSecond;
    const bool sprinting = keyDown(m_window, GLFW_KEY_LEFT_SHIFT);
    if (sprinting) {
        speed *= kSprintMultiplier;
    }
    if (m_bethesdaPlayerControllerRegistered && m_walkMode) {
        if (!m_bethesdaControllerOwnsCamera) {
            if (thirdPersonPlayerShowcase()) {
                m_bethesdaControllerOwnsCamera = true;
                reconstructPlayerCamera(deltaSeconds, true);
            } else {
                relocateBethesdaPlayerControllerToCamera();
            }
        }
        bethesda::PhysicsCharacterInput input;
        input.desiredVelocity = {moveX * speed, 0.0f, moveZ * speed};
        const bethesda::ObjectId playerId = m_bethesdaSession.playerObject();
        const auto physical = m_bethesdaSession.physics().characterState(playerId);
        if (canControlPlayer && keyDown(m_window, GLFW_KEY_SPACE) &&
            physical.has_value() && physical->grounded) {
            input.desiredVelocity.y = kJumpUnitsPerSecond;
        }
        if (thirdPersonPlayerShowcase() && lengthSquared > 1.0e-6f) {
            const float wanted = actorYawForDirection(moveX, moveZ);
            float turn = std::fmod(wanted - m_playerYawRadians + 3.0f * kPi,
                2.0f * kPi) - kPi;
            constexpr float kMaximumTurnRate = 8.0f;
            turn = std::clamp(turn, -kMaximumTurnRate * deltaSeconds,
                kMaximumTurnRate * deltaSeconds);
            m_playerPreviousYawRadians = m_playerYawRadians;
            m_playerYawRadians += turn;
        }
        if (thirdPersonPlayerShowcase()) {
            anim::AnimationInputState animationInput;
            animationInput.requestedVelocity = input.desiredVelocity;
            animationInput.movementSpeed = std::sqrt(
                input.desiredVelocity.x * input.desiredVelocity.x +
                input.desiredVelocity.z * input.desiredVelocity.z);
            const odai::math::Vector3 facing = actorFacing(m_playerYawRadians);
            const odai::math::Vector3 right{-facing.z, 0.0f, facing.x};
            animationInput.localVelocity = {
                odai::math::dot(input.desiredVelocity, right),
                input.desiredVelocity.y,
                odai::math::dot(input.desiredVelocity, facing)};
            animationInput.turnRateRadiansPerSecond = deltaSeconds > 0.0f
                ? (m_playerYawRadians - m_playerPreviousYawRadians) / deltaSeconds
                : 0.0f;
            animationInput.sprinting = sprinting &&
                animationInput.movementSpeed > 1.0f;
            if (physical.has_value()) {
                animationInput.groundVelocity = physical->groundVelocity;
                animationInput.groundNormal = physical->groundNormal;
                animationInput.verticalVelocity = physical->velocity.y;
                animationInput.grounded = physical->grounded;
                animationInput.falling = physical->falling;
                animationInput.landed = physical->landed;
                animationInput.blocked = physical->blocked;
            }
            if (m_skyrimPlayerAvatar.has_value() &&
                m_skyrimPlayerAvatar->walkSpeedUnitsPerSecond > 1.0f) {
                animationInput.locomotionPlaybackRate = std::clamp(
                    animationInput.movementSpeed /
                        m_skyrimPlayerAvatar->walkSpeedUnitsPerSecond,
                    0.25f, 4.0f);
            }
            (void)m_bethesdaSession.setActorAnimationInput(
                playerId, std::move(animationInput));
        }
        (void)m_bethesdaSession.setActorControllerInput(playerId, input);
        m_bethesdaControllerOwnsCamera = true;
        if (thirdPersonPlayerShowcase()) reconstructPlayerCamera(deltaSeconds);
        return;
    }
    if (m_bethesdaPlayerControllerRegistered) {
        (void)m_bethesdaSession.setActorControllerInput(
            m_bethesdaSession.playerObject(), bethesda::PhysicsCharacterInput{});
        m_bethesdaControllerOwnsCamera = false;
    }
    m_cameraX += moveX * speed * deltaSeconds;
    m_cameraZ += moveZ * speed * deltaSeconds;

    // Push back out of anything solid the move ended inside. Walk mode only:
    // fly mode is the diagnostic camera and deliberately passes through
    // everything, which is what makes it useful for looking at geometry.
    if (m_walkMode && m_streamer) {
        m_collision.resolveHorizontal(m_cameraX, m_cameraY, m_cameraZ);
    }

    // Walk mode pins the eye to the terrain; fly mode (F) keeps the old free
    // movement, which is still the only way to inspect the scene from above or
    // to get back if you walk off the edge of the cooked grid.
    float groundHeight = 0.0f;
    if (m_walkMode && groundHeightAt(m_cameraX, m_cameraZ, groundHeight)) {
        const float standingHeight = groundHeight + kEyeHeightUnits;
        if (m_airborne) {
            m_verticalVelocity -= kGravityUnitsPerSecondSq * deltaSeconds;
            m_cameraY += m_verticalVelocity * deltaSeconds;
            // Land when the ground catches up. Tested against the ground under
            // the CURRENT position, so walking off a ledge mid-jump lands on
            // whatever is actually below rather than the height jumped from.
            if (m_cameraY <= standingHeight) {
                m_cameraY = standingHeight;
                m_verticalVelocity = 0.0f;
                m_airborne = false;
            }
        } else {
            m_cameraY = standingHeight;
            if (keyDown(m_window, GLFW_KEY_SPACE)) {
                m_verticalVelocity = kJumpUnitsPerSecond;
                m_airborne = true;
            }
        }
    } else {
        // Fly mode (or off the terrain grid): Space/Ctrl move straight up and
        // down, and there is nothing to fall onto.
        m_cameraY += moveY * speed * deltaSeconds;
        m_airborne = false;
        m_verticalVelocity = 0.0f;
    }
}

bool BethesdaApp::resolveConfiguredContentProfile() {
    if (m_contentProfilePath.empty()) {
        return true;
    }
    importer::fnv::ContentProfileResolveOptions options;
    if (!m_streamDirectory.empty()) {
        options.dataRootOverride = std::filesystem::path(m_streamDirectory);
    }
    if (!m_contentProfileModsRoot.empty()) {
        options.modsRoot = std::filesystem::path(m_contentProfileModsRoot);
    }
    for (const std::string& root : m_modDirectories) {
        options.extraLayers.emplace_back(root);
    }
    options.forceContentReindex = m_forceContentReindex;
    if (const char* modsEnv = std::getenv("ODAI_FNV_MODS")) {
        const std::string mods = modsEnv;
        std::size_t start = 0u;
        while (start <= mods.size()) {
            const std::size_t end = mods.find(':', start);
            const std::string entry = mods.substr(
                start, end == std::string::npos ? std::string::npos : end - start);
            if (!entry.empty()) options.extraLayers.emplace_back(entry);
            if (end == std::string::npos) break;
            start = end + 1u;
        }
    }

    importer::fnv::ResolvedContentProfile resolved;
    std::string error;
    if (!importer::fnv::resolveContentProfile(
            std::filesystem::path(m_contentProfilePath), options, resolved, error)) {
        if (!m_compatibilityReportPath.empty()) {
            std::string reportError;
            if (!importer::fnv::writeContentCompatibilityReport(
                    std::filesystem::path(m_compatibilityReportPath), resolved, reportError)) {
                VOX_LOGW("mods") << reportError;
            }
        }
        VOX_LOGE("newvegas") << "content profile failed: " << error;
        for (const importer::fnv::ContentDiagnostic& diagnostic : resolved.diagnostics) {
            VOX_LOGE("mods") << diagnostic.code << ": " << diagnostic.message;
        }
        return false;
    }
    if (resolved.plugins.empty()) {
        VOX_LOGE("newvegas") << "content profile has no active Bethesda plugins";
        return false;
    }

    const std::vector<std::string> explicitPlugins = std::move(m_extraPlugins);
    m_streamDirectory = resolved.dataRoot.string();
    m_streamPlugin = resolved.plugins.front();
    m_extraPlugins.assign(resolved.plugins.begin() + 1, resolved.plugins.end());
    for (const std::string& plugin : explicitPlugins) {
        const auto duplicate = std::find_if(
            m_extraPlugins.begin(), m_extraPlugins.end(), [&](const std::string& existing) {
                return toLowerAscii(existing) == toLowerAscii(plugin);
            });
        if (duplicate == m_extraPlugins.end() &&
            toLowerAscii(plugin) != toLowerAscii(m_streamPlugin)) {
            m_extraPlugins.push_back(plugin);
        }
    }
    resolved.plugins.clear();
    resolved.plugins.push_back(m_streamPlugin);
    resolved.plugins.insert(resolved.plugins.end(), m_extraPlugins.begin(), m_extraPlugins.end());
    m_modDirectories.clear();
    for (const importer::fnv::ContentLayer& layer : resolved.layers) {
        if (layer.enabled) m_modDirectories.push_back(layer.root.string());
    }
    m_loadOrderFingerprint = resolved.fingerprint;
    m_streamIsMorrowind = resolved.game == importer::fnv::BethesdaGame::Morrowind;
    m_streamIsOblivion = resolved.game == importer::fnv::BethesdaGame::Oblivion;
    m_streamIsSkyrim =
        resolved.game == importer::fnv::BethesdaGame::SkyrimSpecialEdition;

    for (const importer::fnv::ContentDiagnostic& diagnostic : resolved.diagnostics) {
        if (diagnostic.severity == importer::fnv::ContentDiagnosticSeverity::Warning) {
            VOX_LOGW("mods") << diagnostic.code << ": " << diagnostic.message
                              << (diagnostic.source.empty() ? std::string{} :
                                  " (" + diagnostic.source.string() + ")");
        } else {
            VOX_LOGI("mods") << diagnostic.code << ": " << diagnostic.message;
        }
    }
    VOX_LOGI("mods") << "profile " << resolved.name << " ("
                      << importer::fnv::bethesdaGameName(resolved.game) << "): "
                      << resolved.layers.size() << " layers, " << resolved.plugins.size()
                      << " plugins, fingerprint " << resolved.fingerprint;
    if (!m_compatibilityReportPath.empty()) {
        std::string reportError;
        if (!importer::fnv::writeContentCompatibilityReport(
                std::filesystem::path(m_compatibilityReportPath), resolved, reportError)) {
            VOX_LOGW("mods") << reportError;
        }
    }
    m_contentProfile = std::move(resolved);
    return true;
}

bool BethesdaApp::initStreaming() {
    std::string requestedWorldspaceBeforeResume = m_streamWorldspace;
    // One worker per core minus the main thread and a little headroom, floored
    // at 2. Streaming is latency-sensitive rather than throughput-bound, so
    // oversubscribing here would just contend with the render thread.
    const unsigned hardwareThreads = std::max(4u, std::thread::hardware_concurrency());
    unsigned streamThreads = std::max(2u, hardwareThreads - 2u);
    if (const char* env = std::getenv("ODAI_FNV_STREAM_THREADS")) {
        streamThreads = std::max(1u, static_cast<unsigned>(std::atoi(env)));
    }
    // A spawn interior nobody asked for is a New Vegas default, so do not hunt
    // for it in somebody else's plugin -- Fallout 3 warned about a missing
    // GSDocMitchellHouse on every launch.
    if (!m_streamSpawnInteriorExplicit && toLowerAscii(m_streamPlugin) != "falloutnv.esm") {
        m_streamSpawnInterior.clear();
    }
    VOX_LOGI("newvegas") << "streaming workers: " << streamThreads;
    m_streamJobs = std::make_unique<core::JobSystem>(streamThreads);
    m_streamer = std::make_unique<importer::fnv::CellStreamer>();
    if (m_contentProfile.has_value()) {
        m_streamer->setContentProfile(*m_contentProfile);
    }

    // ODAI_FNV_MODS is ':'-separated, appended after any --mod so the flag
    // keeps the lower priority position it was given on the command line and
    // the env can layer on top.
    if (!m_contentProfile.has_value()) if (const char* modsEnv = std::getenv("ODAI_FNV_MODS")) {
        const std::string mods = modsEnv;
        std::size_t start = 0;
        while (start <= mods.size()) {
            const std::size_t end = mods.find(':', start);
            const std::string entry =
                mods.substr(start, end == std::string::npos ? std::string::npos : end - start);
            if (!entry.empty()) {
                m_modDirectories.push_back(entry);
            }
            if (end == std::string::npos) {
                break;
            }
            start = end + 1;
        }
    }
    for (const std::string& modDirectory : m_modDirectories) {
        VOX_LOGI("newvegas") << "mod directory: " << modDirectory;
        if (!m_contentProfile.has_value()) {
            m_streamer->addModDirectory(std::filesystem::path(modDirectory));
        }
    }

    // Needed even on the single-plugin path: actor/dialogue and TES4 weather
    // readers accept enough TES3 bytes to fail confusingly rather than being a
    // meaningful part of a world-rendering-only Morrowind session.
    {
        importer::fnv::FalloutPluginHeader header;
        std::string headerError;
        if (importer::fnv::readFalloutPluginHeader(
                std::filesystem::path(m_streamDirectory) / m_streamPlugin,
                header, headerError)) {
            m_streamIsMorrowind = m_streamIsMorrowind ||
                header.format == importer::fnv::EsmPluginFormat::kMorrowind;
            m_streamIsOblivion = m_streamIsOblivion ||
                header.format == importer::fnv::EsmPluginFormat::kOblivion;
            // Skyrim is the only supported generation whose base master uses
            // localized string tables. That is a format property, unlike a
            // filename check, so a renamed or total-conversion master keeps
            // the generated-object LOD path.
            m_streamIsSkyrim = m_streamIsSkyrim || header.isLocalized;
            // WastelandNV is the historical runtime default, not a sensible
            // implicit worldspace for a Skyrim interior/start request. Keep an
            // explicit --worldspace authoritative, but otherwise establish
            // Skyrim's parent exterior before opening the streamer or applying
            // an optional saved-space override.
            if (m_streamIsSkyrim && !m_streamWorldspaceExplicit) {
                m_streamWorldspace = "Tamriel";
                requestedWorldspaceBeforeResume = m_streamWorldspace;
            }
        }
    }

    // ODAI_FNV_TEX_SIZE is the mip-drop ceiling. The 512 default is what makes
    // the base game fit; a high-resolution texture pack is invisible without
    // raising it, because its art gets dropped straight back down. Memory goes
    // as the square, so this is the knob to reach for first when the GPU starts
    // complaining.
    if (const char* texSizeEnv = std::getenv("ODAI_FNV_TEX_SIZE")) {
        const int requested = std::atoi(texSizeEnv);
        if (requested >= 0) {
            m_streamer->setMaxTextureSize(static_cast<std::uint32_t>(requested));
            VOX_LOGI("newvegas") << "texture ceiling: "
                                 << (requested == 0 ? "unclamped" : std::to_string(requested) + " px");
        }
    }

    if (m_streamCacheEnabled) {
        if (m_streamCacheDirectory.empty()) {
            if (const char* fromEnv = std::getenv("ODAI_FNV_CACHE_DIR")) {
                m_streamCacheDirectory = fromEnv;
            }
        }
        if (m_streamCacheDirectory.empty()) {
            // XDG cache location, falling back to the home directory. Built
            // cells are derived data: safe to lose, expensive to recompute.
            if (const char* xdgCache = std::getenv("XDG_CACHE_HOME")) {
                m_streamCacheDirectory = (std::filesystem::path(xdgCache) / "odai" / "fnv").string();
            } else if (const char* home = std::getenv("HOME")) {
                m_streamCacheDirectory =
                    (std::filesystem::path(home) / ".cache" / "odai" / "fnv").string();
            }
        }
        if (!m_streamCacheDirectory.empty()) {
            m_streamer->setCacheDirectory(std::filesystem::path(m_streamCacheDirectory));
        }
    }

    std::string error;
    // Stream across the whole load order when extra plugins are loaded, so a
    // patch's record fixes -- moved statics, corrected models, replaced terrain
    // -- actually reach the scene. YUP alone ships 18871 reference overrides,
    // none of which did anything while cells came from one plugin.
    //
    // Deliberately skipped when nothing was added: re-indexing seven plugins
    // costs startup time for no override, and the single-plugin path is the one
    // every measurement in this project was taken on.
    if (m_contentProfile.has_value() || m_streamIsSkyrim || !m_extraPlugins.empty()) {
        std::vector<std::string> requestedPlugins;
        std::filesystem::path loadOrderSource;
        if (m_contentProfile.has_value()) {
            requestedPlugins = m_contentProfile->plugins;
        } else if (m_streamIsSkyrim) {
            std::optional<std::filesystem::path> explicitLoadOrder;
            if (!m_loadOrderPath.empty()) {
                explicitLoadOrder = std::filesystem::path(m_loadOrderPath);
            } else if (const char* fromEnv = std::getenv("ODAI_FNV_LOAD_ORDER")) {
                if (*fromEnv != '\0') {
                    explicitLoadOrder = std::filesystem::path(fromEnv);
                }
            }
            if (!importer::fnv::resolveInstalledSkyrimPluginList(
                    std::filesystem::path(m_streamDirectory), explicitLoadOrder,
                    requestedPlugins, loadOrderSource, error)) {
                VOX_LOGE("newvegas") << "Skyrim load order failed: " << error;
                return false;
            }
        } else {
            requestedPlugins.push_back(m_streamPlugin);
        }
        if (!m_contentProfile.has_value()) {
            requestedPlugins.insert(
                requestedPlugins.end(), m_extraPlugins.begin(), m_extraPlugins.end());
        }
        importer::fnv::FalloutLoadOrder streamOrder;
        std::string orderError;
        const bool loadOrderOpened = m_contentProfile.has_value()
            ? streamOrder.open(*m_contentProfile, orderError)
            : ([&]() {
                for (const std::string& modDirectory : m_modDirectories) {
                    streamOrder.addSearchRoot(std::filesystem::path(modDirectory));
                }
                return streamOrder.open(
                    std::filesystem::path(m_streamDirectory), requestedPlugins, orderError);
            })();
        if (!loadOrderOpened) {
            if (m_contentProfile.has_value() || m_streamIsSkyrim) {
                VOX_LOGE("newvegas") << "authoritative load order failed: " << orderError;
                return false;
            }
            VOX_LOGW("newvegas") << "streaming one plugin only: " << orderError;
        } else {
            std::string loadOrderText;
            for (const auto& entry : streamOrder.entries()) {
                if (!loadOrderText.empty()) {
                    loadOrderText += " -> ";
                }
                loadOrderText += entry.header.fileName;
            }
            VOX_LOGI("newvegas") << "streaming across " << streamOrder.size()
                                 << " plugins (record overrides active): " << loadOrderText;
            if (m_streamIsSkyrim) {
                VOX_LOGI("newvegas") << "Skyrim load-order source: "
                                     << (loadOrderSource.empty()
                                             ? std::string("installed official content")
                                             : loadOrderSource.string());
            }
            // Kept on the app too: actor discovery needs the same order, and a
            // companion mod's NPC/placement/race/armour all live in its plugin.
            m_streamLoadOrder = streamOrder;
            m_streamIsMorrowind = !streamOrder.entries().empty() &&
                streamOrder.entries().front().header.format ==
                    importer::fnv::EsmPluginFormat::kMorrowind;
            m_streamIsOblivion = !streamOrder.entries().empty() &&
                streamOrder.entries().front().header.format ==
                    importer::fnv::EsmPluginFormat::kOblivion;
            m_streamIsSkyrim = !streamOrder.entries().empty() &&
                streamOrder.entries().front().header.isLocalized;
            m_loadOrderFingerprint = m_contentProfile.has_value()
                ? m_contentProfile->fingerprint + "-" + streamOrder.fingerprint()
                : streamOrder.fingerprint();
            m_streamer->setLoadOrder(std::move(streamOrder));
        }
    }
    if (m_traversalStatePath.empty()) {
        m_traversalStatePath = defaultTraversalStatePath();
    }
    if (m_resumeEnabled && !m_explicitStart) {
        std::error_code stateExistsError;
        if (std::filesystem::is_regular_file(m_traversalStatePath, stateExistsError) &&
            !stateExistsError) {
            TraversalState state;
            std::string stateError;
            if (loadTraversalState(m_traversalStatePath, state, stateError)) {
                if (!state.worldspaceEditorId.empty()) {
                    m_streamWorldspace = state.worldspaceEditorId;
                }
                if (state.interior && !state.interiorEditorId.empty()) {
                    m_startInsideInterior = state.interiorEditorId;
                }
                m_timeOfDayHours = state.timeOfDayHours;
                if (!state.weatherEditorId.empty()) {
                    m_requestedWeatherEditorId = state.weatherEditorId;
                }
                m_resumeState = std::move(state);
                VOX_LOGI("newvegas") << "resuming traversal from "
                                     << m_traversalStatePath.string();
            } else {
                VOX_LOGW("newvegas") << "ignoring traversal state: " << stateError;
            }
        }
    }
    if (!m_streamer->open(
            std::filesystem::path(m_streamDirectory), std::filesystem::path(m_streamPlugin),
            m_streamWorldspace, *m_streamJobs, error)) {
        if (m_resumeState.has_value()) {
            VOX_LOGW("newvegas") << "saved space is unavailable (" << error
                                 << "); falling back to " << requestedWorldspaceBeforeResume;
            m_resumeState.reset();
            m_startInsideInterior.clear();
            m_streamWorldspace = requestedWorldspaceBeforeResume;
            if (!m_streamer->open(
                    std::filesystem::path(m_streamDirectory),
                    std::filesystem::path(m_streamPlugin), m_streamWorldspace,
                    *m_streamJobs, error)) {
                VOX_LOGE("newvegas") << "streaming fallback failed: " << error;
                return false;
            }
        } else {
            VOX_LOGE("newvegas") << "streaming init failed: " << error;
            return false;
        }
    }
    if (m_resumeState.has_value() && m_resumeState->interior &&
        !m_streamer->hasInterior(m_startInsideInterior)) {
        VOX_LOGW("newvegas") << "saved interior " << m_startInsideInterior
                             << " is unavailable; using exterior spawn";
        m_startInsideInterior.clear();
        m_resumeState.reset();
    }
    if (m_resumeState.has_value()) {
        const bool fingerprintChanged =
            !m_resumeState->loadOrderFingerprint.empty() &&
            m_resumeState->loadOrderFingerprint != m_loadOrderFingerprint;
        for (const TraversalDiscovery& saved : m_resumeState->discoveries) {
            const auto marker = std::find_if(
                m_streamer->mapMarkers().begin(), m_streamer->mapMarkers().end(),
                [&](const importer::fnv::FalloutMapMarkerRecord& current) {
                    return fingerprintChanged
                        ? current.name == saved.name
                        : current.referenceFormId == saved.sourceReferenceFormId;
                });
            if (marker == m_streamer->mapMarkers().end()) {
                continue;
            }
            m_discoveredMarkerIds.insert(marker->referenceFormId);
            m_discoveredLocations.push_back(
                TraversalDiscovery{marker->referenceFormId, saved.worldspaceEditorId,
                                   marker->name});
        }
        if (fingerprintChanged) {
            VOX_LOGI("newvegas")
                << "load order changed; re-resolved saved discoveries by world/name";
        }
    }
    if (m_streamer->availableCellCount() == 0u) {
        VOX_LOGE("newvegas") << "worldspace " << m_streamWorldspace << " has no streamable cells in "
                             << m_streamDirectory;
        return false;
    }

    // Mirror the resident set into collision. Registered before the first
    // update() so no cell can become resident without collision knowing.
    m_collision.clear();
    m_actorNavigation.clear();
    m_bethesdaCollisionByCell.clear();
    m_bethesdaGameplayResidentCells.clear();
    if (m_whiterunReferenceShowcase) {
        m_streamer->setScenePresentationOverride(
            [](importer::ImportedScene& scene) {
                std::unordered_set<std::uint32_t> hiddenDoorReferences;
                std::unordered_set<std::uint32_t> mainGateShellMeshes;
                for (importer::ImportedSceneInstance& instance : scene.instances) {
                    const std::string modelPath = toLowerAscii(instance.modelPath);
                    if (modelPath.find("wrwallmaingate01.nif") != std::string::npos) {
                        mainGateShellMeshes.insert(instance.meshIndex);
                    }
                    if (modelPath.find("wrdoormaingate01.nif") ==
                            std::string::npos) {
                        continue;
                    }
                    instance.initiallyVisible = false;
                    if (instance.sourceReferenceFormId != 0u) {
                        hiddenDoorReferences.insert(instance.sourceReferenceFormId);
                    }
                }
                if (hiddenDoorReferences.empty() && mainGateShellMeshes.empty()) {
                    return;
                }
                std::size_t hiddenDoorMeshes = 0u;
                for (importer::ImportedSceneMesh& mesh : scene.meshes) {
                    const bool usesMainGateDoorTexture = std::any_of(
                        mesh.parts.begin(), mesh.parts.end(),
                        [&](const importer::ImportedSceneMeshPart& part) {
                            if (part.textureIndex >= scene.textures.size()) {
                                return false;
                            }
                            const std::string path =
                                toLowerAscii(scene.textures[part.textureIndex].sourcePath);
                            return path.find("wrdoormaingate01.dds") != std::string::npos ||
                                path.ends_with("\\wrdoor01.dds");
                        });
                    if (!usesMainGateDoorTexture) {
                        continue;
                    }
                    mesh.vertices.clear();
                    mesh.indices.clear();
                    mesh.parts.clear();
                    ++hiddenDoorMeshes;
                }
                std::size_t hiddenEmbeddedDoorParts = 0u;
                for (const std::uint32_t meshIndex : mainGateShellMeshes) {
                    if (meshIndex >= scene.meshes.size()) {
                        continue;
                    }
                    importer::ImportedSceneMesh& mesh = scene.meshes[meshIndex];
                    hiddenEmbeddedDoorParts += std::erase_if(
                        mesh.parts,
                        [&](const importer::ImportedSceneMeshPart& part) {
                            return part.textureIndex < scene.textures.size() &&
                                toLowerAscii(scene.textures[part.textureIndex].sourcePath)
                                    .ends_with("\\wrwoodplanks01.dds");
                        });
                }
                std::erase_if(
                    scene.collisionTriangles,
                    [&](const importer::ImportedSceneCollisionTriangle& triangle) {
                        return hiddenDoorReferences.contains(triangle.sourceReferenceFormId);
                    });
                importer::buildImportedScenePackedRenderData(scene);
                importer::buildImportedScenePageRanges(scene);
                VOX_LOGI("showcase")
                    << "Whiterun reference opened " << hiddenDoorReferences.size()
                    << " authored main-gate door reference(s) across "
                    << hiddenDoorMeshes << " retail mesh(es), plus "
                    << hiddenEmbeddedDoorParts
                    << " embedded closed-door part(s), before presentation";
            });
    }
    m_streamer->setCellCallbacks(
        [this](
            const importer::CellCoord& cell,
            const importer::ImportedScene& scene,
            const std::vector<importer::fnv::FalloutNavMeshRecord>& navMeshes) {
            if (m_streamIsMorrowind) {
                m_bethesdaGameplayResidentCells.insert(cell);
            }
            // Cache first so initially hidden references are known to both
            // collision implementations. Otherwise the lightweight capsule
            // resolver can stand on an invisible door/FX hull that Jolt has
            // correctly omitted from the same streamed cell.
            if (!m_scenarioId.empty() || m_streamIsMorrowind) {
                cacheBethesdaCollisionCell(cell, scene);
            }
            m_collision.addCell(cell, scene, m_disabledBethesdaCollisionReferences);
            m_actorNavigation.addCell(cell, navMeshes);
            if (navMeshes.empty() && (m_streamIsMorrowind || m_streamIsOblivion)) {
                m_actorNavigation.addGeneratedCell(cell, scene);
            }
            m_streamDoorsByCell[cell] = scene.doors;
            rebuildStreamDoors();
            if (m_streamIsSkyrim) m_skyrimActorResidencyDirty = true;
            if (m_whiterunReferenceShowcase) {
                for (const importer::ImportedSceneInstance& instance : scene.instances) {
                    const std::string path = toLowerAscii(instance.modelPath);
                    if (path.find("wrwallmaingate") == std::string::npos &&
                        path.find("wrbrazier") == std::string::npos &&
                        path.find("banner") == std::string::npos) {
                        continue;
                    }
                    VOX_LOGI("showcase")
                        << "Whiterun reference anchor " << instance.modelPath
                        << " at (" << instance.transform[3] << ", "
                        << instance.transform[7] << ", " << instance.transform[11]
                        << ") in cell " << cell.x << "," << cell.z;
                }
            }
            if (std::getenv("ODAI_FNV_LOG_NAVIGATION") != nullptr) {
                VOX_LOGI("newvegas") << "navigation cell " << cell.x << "," << cell.z
                                     << ": " << navMeshes.size() << " meshes, resident total "
                                     << m_actorNavigation.meshCount() << " meshes / "
                                     << m_actorNavigation.triangleCount() << " triangles / "
                                     << m_actorNavigation.generatedNodeCount()
                                     << " generated nodes";
            }
        },
        [this](const importer::CellCoord& cell) {
            m_bethesdaGameplayResidentCells.erase(cell);
            m_collision.removeCell(cell);
            m_actorNavigation.removeCell(cell);
            removeBethesdaCollisionCell(cell);
            m_streamDoorsByCell.erase(cell);
            rebuildStreamDoors();
            if (m_streamIsSkyrim) m_skyrimActorResidencyDirty = true;
        });
    m_streamAmbientEmittersByCell.clear();
    m_streamer->setAmbientEmitterCallbacks(
        [this](
            const importer::CellCoord& cell,
            const std::vector<importer::fnv::FalloutSoundEmitterRecord>& emitters) {
            m_streamAmbientEmittersByCell.insert_or_assign(cell, emitters);
        },
        [this](const importer::CellCoord& cell) {
            m_streamAmbientEmittersByCell.erase(cell);
        });
    m_ambienceRandomState = m_captureSeed != 0u ? m_captureSeed : 0x4f444149u;

    importer::CellResidencyConfig config;
    // From the plugin, not a constant. Fallout and Oblivion exterior cells are
    // 4096 units square (33 height posts at 128-unit spacing); Morrowind's are
    // 8192 (65 posts at the same spacing). Everything about residency is
    // expressed in cells, so a grid built on the wrong figure loads a quarter
    // of the world it believes it is loading.
    config.cellSize = m_streamer->cellWorldSize();
    // TES3 cells are twice as wide as the later games. A radius of four was
    // therefore loading 81 detailed cells (roughly a kilometre-square region),
    // including collision/nav/gameplay for references that cannot affect the
    // player. Five by five still covers several city blocks and cuts detailed
    // residency to 25 cells; explicit compatibility env overrides keep their
    // historical spelling.
    if (m_streamIsMorrowind) {
        config.loadRadius = 2;
        config.unloadRadius = 4;
    } else if (skyrimCityShowcase()) {
        // Skyrim's walled cities store much of their always-visible kit in
        // persistent cell 0,0 even though the main-gate arrival is several
        // grid cells away. Keep that cell explicitly resident and stream only
        // the local 3x3 ring instead of a 9x9 radius-four square.
        config.loadRadius = 1;
        config.unloadRadius = 3;
        config.maxLoadsInFlight = 2u;
        m_streamer->setPinnedCells({importer::CellCoord{0, 0}});
    }
    if (const char* radiusEnv = std::getenv("ODAI_FNV_LOAD_RADIUS")) {
        config.loadRadius = std::max(0, std::atoi(radiusEnv));
        config.unloadRadius = config.loadRadius + 2;
    }
    m_streamer->setConfig(config);
    if (skyrimCityShowcase()) {
        // The initial 3x3 ring plus a city's persistent cell fits this modest
        // reserve in the retail masters. Allocate it once so the cold load
        // does not repeatedly recreate and copy device-local arenas as worker
        // results land. A changed load order may exceed it and still uses
        // normal growth.
        constexpr std::uint64_t kInitialVertexReserve = 1'000'000u;
        constexpr std::uint64_t kInitialIndexReserve = 2'500'000u;
        if (!m_renderer.reserveImportedSceneGeometry(
                kInitialVertexReserve, kInitialIndexReserve)) {
            VOX_LOGW("showcase")
                << "initial geometry reservation failed; falling back to streamed growth";
        }
    }

    // A fixed-step video capture advances the tour in simulation time, not in
    // wall time. That can cross the entire route while the worker threads are
    // still extracting its first cells, leaving a visibly half-built town in
    // the recording even though every mesh eventually loads. Preload and pin
    // the whole corridor before its first captured frame instead.
    const bool isVideoTourCapture = m_flythroughSeconds > 0.0f &&
        (!m_captureVideoPath.empty() || !m_captureDirectory.empty());
    if (isVideoTourCapture) {
        constexpr int kTourPreloadSamples = 96;
        std::unordered_set<importer::CellCoord, importer::CellCoordHash> pinnedCells;
        const std::int32_t radius = std::max(0, config.loadRadius);
        const std::int32_t lodTileCells = importer::fnv::kLandLodBlockCells;
        std::int32_t minLodTileX = std::numeric_limits<std::int32_t>::max();
        std::int32_t minLodTileZ = std::numeric_limits<std::int32_t>::max();
        std::int32_t maxLodTileX = std::numeric_limits<std::int32_t>::min();
        std::int32_t maxLodTileZ = std::numeric_limits<std::int32_t>::min();
        for (int sample = 0; sample <= kTourPreloadSamples; ++sample) {
            float enginePosition[3] = {};
            float ignoredLookAt[3] = {};
            sampleTour(static_cast<float>(sample) / static_cast<float>(kTourPreloadSamples),
                       enginePosition, ignoredLookAt);
            float falloutPosition[3] = {};
            importer::fnv::CellStreamer::engineToFallout(enginePosition, falloutPosition);
            const importer::CellCoord centre =
                m_streamer->config().cellSize > 0.0f
                ? importer::CellCoord{
                    static_cast<std::int32_t>(std::floor(falloutPosition[0] / config.cellSize)),
                    static_cast<std::int32_t>(std::floor(falloutPosition[1] / config.cellSize))}
                : importer::CellCoord{};
            const std::int32_t lodTileX =
                importer::fnv::landLodTileOrigin(centre.x, lodTileCells);
            const std::int32_t lodTileZ =
                importer::fnv::landLodTileOrigin(centre.z, lodTileCells);
            minLodTileX = std::min(minLodTileX, lodTileX);
            minLodTileZ = std::min(minLodTileZ, lodTileZ);
            maxLodTileX = std::max(maxLodTileX, lodTileX);
            maxLodTileZ = std::max(maxLodTileZ, lodTileZ);
            for (std::int32_t dz = -radius; dz <= radius; ++dz) {
                for (std::int32_t dx = -radius; dx <= radius; ++dx) {
                    pinnedCells.insert(importer::CellCoord{centre.x + dx, centre.z + dz});
                }
            }
        }
        std::vector<importer::CellCoord> corridor(pinnedCells.begin(), pinnedCells.end());
        m_streamer->setPinnedCells(corridor);
        m_captureRoutePreloadActive = !corridor.empty();
        m_capturePinnedCells = std::move(pinnedCells);
        m_captureSkyrimLodBoundsValid = m_streamIsSkyrim && !corridor.empty();
        if (m_captureSkyrimLodBoundsValid) {
            m_captureSkyrimLodMinTileX = minLodTileX;
            m_captureSkyrimLodMinTileZ = minLodTileZ;
            m_captureSkyrimLodMaxTileX = maxLodTileX;
            m_captureSkyrimLodMaxTileZ = maxLodTileZ;
        }
        VOX_LOGI("newvegas") << "capture route preload: pinned " << corridor.size()
                             << " exterior cells before recording"
                             << (m_captureSkyrimLodBoundsValid
                                     ? (", fixed Skyrim LOD route tiles x[" +
                                        std::to_string(minLodTileX) + "," +
                                        std::to_string(maxLodTileX) + "] z[" +
                                        std::to_string(minLodTileZ) + "," +
                                        std::to_string(maxLodTileZ) + "]")
                                     : std::string());
    }

    // Spawn at the centre of the available cells so the first ring has content
    // on every side; the world origin is often outside a cooked region entirely.
    // ENGINE space (Y-up), not Fallout space (Z-up). Assigning a Fallout grid
    // coordinate to m_cameraY put the camera tens of thousands of units below
    // the terrain -- the streamed world rendered correctly and the player was
    // simply underneath it.
    // START INSIDE A ROOM, which is where New Vegas itself begins -- you wake up
    // on Doc Mitchell's table, not on his porch.
    //
    // Built and uploaded here rather than streamed: an interior is ONE room, not
    // a grid, so there is nothing for the residency planner to plan. It goes in
    // as an ordinary scene chunk and into collision as an ordinary cell, so
    // walls, floors and the ground clamp all work with no interior-specific
    // code anywhere downstream.
    if (!m_startInsideInterior.empty()) {
        importer::ImportedScene interiorScene;
        importer::fnv::CellStreamer::InteriorScene interior;
        std::string interiorError;
        if (!m_streamer->buildInteriorScene(
                m_startInsideInterior, interiorScene, interior, interiorError)) {
            VOX_LOGE("newvegas") << "cannot start inside " << m_startInsideInterior << ": "
                                 << interiorError;
            return false;
        }
        m_interiorChunk = m_renderer.addImportedSceneChunk(interiorScene);
        if (m_interiorChunk == render::Renderer::kInvalidImportedChunkIndex) {
            VOX_LOGE("newvegas") << "failed to upload interior " << m_startInsideInterior;
            return false;
        }
        m_currentInteriorSourceScene = interiorScene;
        // An interior has its own coordinate space with no grid, so it gets one
        // synthetic cell of its own. The coordinate only has to be consistent
        // and not collide with a streamed exterior cell, and an interior sits
        // nowhere near the worldspace grid to begin with.
        const importer::CellCoord interiorCell{
            static_cast<std::int32_t>(std::floor(interior.spawnPosition[0] / 4096.0f)),
            static_cast<std::int32_t>(std::floor(interior.spawnPosition[2] / 4096.0f))};
        m_collision.addCell(interiorCell, interiorScene);
        // Direct interior starts are built before BethesdaSession exists. Keep
        // their authored collision mesh so initBethesdaSession() can register
        // it with Jolt before the player character controller is created.
        // Door transitions already call this path after the session exists;
        // omitting it here gave the player gravity but no floor body and made
        // them fall through cells such as WhiterunDragonsreach.
        if (!m_scenarioId.empty() || m_streamIsMorrowind) {
            cacheBethesdaCollisionCell(interiorCell, interiorScene);
        }
        m_actorNavigation.addCell(interiorCell, interior.navMeshes);
        if (interior.navMeshes.empty() && (m_streamIsMorrowind || m_streamIsOblivion)) {
            m_actorNavigation.addGeneratedCell(interiorCell, interiorScene);
        }
        m_actorNavigation.setResidentDoors({});
        m_doors = interiorScene.doors;
        m_currentInteriorEditorId = m_startInsideInterior;

        if (interior.hasSpawn) {
            m_cameraX = interior.spawnPosition[0];
            m_cameraY = interior.spawnPosition[1] + m_collision.tuning().eyeHeight;
            m_cameraZ = interior.spawnPosition[2];
            m_yawDegrees = interior.spawnYawDegrees;
            m_pitchDegrees = 0.0f;
        }
        // THE ROOM HAS TO BE TOLD IT IS A ROOM. Interior mode was set only on
        // the cooked --scene path (from the scene's source tag), so a STREAMED
        // interior was shaded by the full outdoor rig: sun at 0.95 instead of
        // 0.24, sky irradiance at 0.58 instead of 0.22, no sky-visibility term
        // and no voxel GI. Dragonsreach rendered as a sunlit pavilion -- flat,
        // bright, and with none of the contact darkening the interior path gets
        // from lerp(0.28, 1.0, skyVisibility). It reads as "AO is broken"
        // rather than as "this scene never said it was indoors".
        render::ImportedInteriorLighting lighting{};
        lighting.enabled = true;
        lighting.hasAuthoredLighting = interior.hasLighting;
        lighting.fogNear = interior.fogNear;
        lighting.fogFar = interior.fogFar;
        lighting.showSky = interior.showSky;
        lighting.useSkyLighting = interior.useSkyLighting;
        lighting.localShadowMode =
            render::ImportedInteriorLighting::LocalShadowMode::ShadowMapsWithContact;
        lighting.indirectLightingMode =
            render::ImportedInteriorLighting::IndirectLightingMode::ScreenSpaceDiffuse;
        if (const char* shadowMode = std::getenv("ODAI_FNV_INTERIOR_SHADOWS")) {
            if (std::strcmp(shadowMode, "rt") == 0) {
                lighting.localShadowMode =
                    render::ImportedInteriorLighting::LocalShadowMode::RayTraced;
            } else if (std::strcmp(shadowMode, "maps") == 0) {
                lighting.localShadowMode =
                    render::ImportedInteriorLighting::LocalShadowMode::ShadowMaps;
            } else if (std::strcmp(shadowMode, "contact") == 0) {
                lighting.localShadowMode =
                    render::ImportedInteriorLighting::LocalShadowMode::ShadowMapsWithContact;
            } else if (std::strcmp(shadowMode, "off") == 0) {
                lighting.localShadowMode = render::ImportedInteriorLighting::LocalShadowMode::Off;
            } else {
                VOX_LOGW("newvegas")
                    << "unknown ODAI_FNV_INTERIOR_SHADOWS='" << shadowMode
                    << "'; using contact";
            }
        }
        if (const char* giMode = std::getenv("ODAI_FNV_INTERIOR_GI")) {
            if (std::strcmp(giMode, "ssgi") == 0) {
                lighting.indirectLightingMode =
                    render::ImportedInteriorLighting::IndirectLightingMode::ScreenSpaceDiffuse;
            } else if (std::strcmp(giMode, "off") == 0) {
                lighting.indirectLightingMode =
                    render::ImportedInteriorLighting::IndirectLightingMode::Off;
            } else {
                VOX_LOGW("newvegas")
                    << "unknown ODAI_FNV_INTERIOR_GI='" << giMode
                    << "'; using ssgi";
            }
        }
        for (int channel = 0; channel < 3; ++channel) {
            lighting.ambientColor[channel] = srgbChannelToLinear(interior.ambientColor[channel]);
            lighting.directionalColor[channel] = srgbChannelToLinear(interior.directionalColor[channel]);
            lighting.fogColor[channel] = srgbChannelToLinear(interior.fogColor[channel]);
            if (m_streamIsSkyrim) {
                lighting.ambientColor[channel] =
                    std::min(lighting.ambientColor[channel] * 1.55f, 1.0f);
                lighting.directionalColor[channel] =
                    std::min(lighting.directionalColor[channel] * 1.18f, 1.0f);
            }
        }
        m_renderer.setImportedInteriorLighting(lighting);
        VOX_LOGI("newvegas") << "started inside " << m_startInsideInterior
                             << (interior.hasLighting
                                     ? " (linear XCLL applied)"
                                     : " (no XCLL lighting on this cell)");
        m_interiorStarted = true;
    }

    float spawn[3] = {0.0f, 0.0f, 0.0f};
    // Doc Mitchell's doorstep first -- that is where New Vegas actually begins.
    // Fall back to the middle of the worldspace if the cell is missing, so a
    // different plugin or a trimmed install still starts somewhere sensible.
    bool spawnedAtScenarioMarker = false;
    bool spawnedAtShowcase = false;
    if (!m_interiorStarted && m_balmoraSkyrimPlayerShowcase) {
        constexpr odai::math::Vector3 kSouthCanalApproach{
            -19920.0f, 300.0f, 12960.0f};
        odai::math::Vector3 projected = kSouthCanalApproach;
        if (!m_actorNavigation.projectPoint(
                projected.x, projected.y, projected.z, 384.0f, 640.0f,
                projected)) {
            float ground = projected.y;
            if (m_collision.groundHeight(
                    projected.x, projected.z, projected.y, ground)) {
                projected.y = ground;
            } else {
                VOX_LOGE("showcase")
                    << "Balmora south-canal anchor is not navigation/ground reachable";
                return false;
            }
        }
        spawn[0] = projected.x;
        spawn[1] = projected.y + kEyeHeightUnits;
        spawn[2] = projected.z;
        spawnedAtShowcase = true;
        m_yawDegrees = -90.0f;
        m_pitchDegrees = -8.0f;
        VOX_LOGI("showcase") << "Balmora player anchor projected to ("
                              << projected.x << ", " << projected.y << ", "
                              << projected.z << ")";
    } else if (!m_interiorStarted && skyrimCityShowcase()) {
        const bool whiterun = m_whiterunThirdPersonShowcase ||
            m_whiterunReferenceShowcase;
        const char* childWorldspace = whiterun ? "WhiterunWorld" : "RiftenWorld";
        const char* cityName = whiterun ? "Whiterun" : "Riften";
        float gateYawDegrees = 0.0f;
        if (!m_streamer->spawnAtWorldspaceEntranceEngineSpace(
                "Tamriel", childWorldspace, spawn, gateYawDegrees)) {
            VOX_LOGE("showcase")
                << cityName
                << " main-gate arrival could not be resolved from retail XTEL pairs";
            return false;
        }
        spawnedAtShowcase = true;
        if (m_whiterunReferenceShowcase) {
            const bethesda::WhiterunReferenceCamera camera =
                m_whiterunMarketReferenceShowcase
                    ? bethesda::whiterunMarketReferenceCamera(spawn, gateYawDegrees)
                    : bethesda::whiterunReferenceCamera(spawn, gateYawDegrees);
            std::copy_n(camera.position, 3u, spawn);
            m_yawDegrees = camera.yawDegrees;
            m_pitchDegrees = camera.pitchDegrees;
            VOX_LOGI("showcase")
                << (m_whiterunMarketReferenceShowcase
                        ? "Whiterun market reference camera from authored gate yaw "
                        : "Whiterun reference camera from authored gate yaw ")
                << gateYawDegrees << ": x=" << spawn[0] << " y=" << spawn[1]
                << " z=" << spawn[2] << " yaw=" << m_yawDegrees
                << " pitch=" << m_pitchDegrees;
        } else {
            m_yawDegrees = gateYawDegrees;
            m_pitchDegrees = -8.0f;
        }
    } else if (!m_interiorStarted && !m_scenarioStartMarker.empty()) {
        const std::string wantedMarker = toLowerAscii(m_scenarioStartMarker);
        const auto marker = std::find_if(
            m_streamer->mapMarkers().begin(), m_streamer->mapMarkers().end(),
            [&](const importer::fnv::FalloutMapMarkerRecord& candidate) {
                return !candidate.deleted && !candidate.initiallyDisabled &&
                    candidate.worldspaceFormId == m_streamer->currentWorldspaceFormId() &&
                    toLowerAscii(candidate.name) == wantedMarker;
            });
        if (marker != m_streamer->mapMarkers().end()) {
            spawn[0] = marker->position[0];
            spawn[1] = marker->position[2] + kEyeHeightUnits;
            spawn[2] = -marker->position[1];
            spawnedAtScenarioMarker = true;
            VOX_LOGI("scenario") << "spawn marker " << marker->name << " resolved at ref 0x"
                                  << std::hex << marker->referenceFormId << std::dec;
        } else {
            VOX_LOGE("scenario") << "required start marker '" << m_scenarioStartMarker
                                  << "' is unavailable in " << m_streamWorldspace;
            return false;
        }
    }
    const bool spawnedAtDoorstep =
        !m_interiorStarted && !m_streamSpawnInterior.empty() &&
        m_streamer->spawnAtInteriorDoorEngineSpace(m_streamSpawnInterior, spawn);
    const bool haveSpawn =
        !m_interiorStarted && (spawnedAtShowcase || spawnedAtScenarioMarker || spawnedAtDoorstep ||
                               m_streamer->suggestedSpawnEngineSpace(spawn));
    if (haveSpawn) {
        bool spawnedAtExplicitPosition = false;
        m_cameraX = spawn[0];
        m_cameraY = spawn[1];  // height
        m_cameraZ = spawn[2];
        // A doorstep spawn is at eye height on the ground, so look at the
        // horizon; the worldspace-centre fallback is well above the terrain and
        // wants to look down at it.
        if (!spawnedAtShowcase) {
            m_pitchDegrees = (spawnedAtScenarioMarker || spawnedAtDoorstep) ? 0.0f : -20.0f;
        }
        // Same diagnostic override the cooked-scene path has: being able to
        // look straight down separates "above the ground" from "inside it".
        if (const char* pitchEnv = std::getenv("ODAI_FNV_PITCH")) {
            m_pitchDegrees = static_cast<float>(std::atof(pitchEnv));
        }
        // Pinning yaw as well is what makes two captures comparable: the mouse
        // position at startup otherwise rotates the camera differently per run,
        // so an A/B of a rendering change compares two different views.
        if (const char* yawEnv = std::getenv("ODAI_FNV_YAW")) {
            m_yawDegrees = static_cast<float>(std::atof(yawEnv));
        }
        if (const char* heightEnv = std::getenv("ODAI_FNV_SPAWN_HEIGHT")) {
            m_cameraY += static_cast<float>(std::atof(heightEnv));
        }
        // Full engine-space spawn override, for pinning the camera somewhere
        // known-good while diagnosing.
        if (const char* posEnv = std::getenv("ODAI_FNV_SPAWN_POS")) {
            float px = 0.0f;
            float py = 0.0f;
            float pz = 0.0f;
            if (std::sscanf(posEnv, "%f,%f,%f", &px, &py, &pz) == 3) {
                m_cameraX = px;
                m_cameraY = py;
                m_cameraZ = pz;
                spawnedAtExplicitPosition = true;
            }
        }
        if (skyrimCityThirdPersonShowcase()) {
            const float engineSpawn[3] = {m_cameraX, m_cameraY, m_cameraZ};
            float bethesdaSpawn[3] = {};
            importer::fnv::CellStreamer::engineToFallout(
                engineSpawn, bethesdaSpawn);
            const float cellSize = m_streamer->cellWorldSize();
            m_skyrimCitySpawnCell = {
                static_cast<std::int32_t>(std::floor(bethesdaSpawn[0] / cellSize)),
                static_cast<std::int32_t>(std::floor(bethesdaSpawn[1] / cellSize))};
            m_skyrimCityAuthoredSpawnFeet = {
                m_cameraX, m_cameraY - kEyeHeightUnits, m_cameraZ};
            m_skyrimCitySpawnSettlementPending = true;
        }
        VOX_LOGI("newvegas") << "spawn (engine space): x=" << m_cameraX
                             << " y=" << m_cameraY << " (height) z=" << m_cameraZ;
        // Walk on arrival at a doorstep: collision now supplies terrain height
        // from the streamed cells, so there is ground to stand on. The
        // worldspace-centre fallback still starts in fly mode, because it aims
        // the camera from well above the terrain.
        m_walkMode = spawnedAtShowcase || spawnedAtScenarioMarker || spawnedAtDoorstep ||
            spawnedAtExplicitPosition;
        // Stand Victor beside wherever the player starts, rather than at his
        // ACRE reference ~7400 units away. Talking to him is the thing being
        // built; a hike across Goodsprings before every test is friction with
        // no upside. ODAI_FNV_VICTOR_HOME=1 puts him back at his real spot.
        if (std::getenv("ODAI_FNV_VICTOR_HOME") == nullptr) {
            const float offsetX = 220.0f;
            const float offsetZ = 220.0f;
            m_victorSpawnPosition[0] = m_cameraX + offsetX;
            m_victorSpawnPosition[2] = m_cameraZ + offsetZ;
            float ground = 0.0f;
            m_victorSpawnPosition[1] =
                groundHeightAt(m_victorSpawnPosition[0], m_victorSpawnPosition[2], ground)
                    ? ground
                    : (m_cameraY - kEyeHeightUnits);
        }
        if (!m_streamIsMorrowind && !m_streamIsSkyrim) {
            const std::filesystem::path dataPath(m_streamDirectory);
            // Victor is loaded into a local and appended to m_actors AFTER the
            // town, because loadGoodspringsActors clears the list -- and it has
            // to run second anyway, since excluding his base from the generic
            // scan needs the formID his own placement lookup finds.
            SkinnedActor victor;
            const bool victorLoaded =
                loadVictor(dataPath / m_streamPlugin, m_streamer->assets(), victor,
                           m_victorSpawnPosition[1] != 0.0f ? m_victorSpawnPosition : nullptr);
            if (victorLoaded) {
                // Turn him to face wherever the player starts. His authored
                // ACRE rotation is not used: standing him beside the spawn
                // already overrode his authored POSITION, and a robot facing
                // the direction he faces in a different part of town reads as
                // broken rather than as fidelity.
                victor.yawRadians = std::atan2(
                    m_cameraZ - victor.position[2], m_cameraX - victor.position[0]);
                victor.instanceSlot = kVictorSkinnedInstance;
            }
            VOX_LOGI("newvegas") << "Victor: " << victor.status;
            // The rest of the town, discovered from the plugin around wherever
            // the player actually is rather than from a hardcoded list.
            {
                const float engineCentre[3] = {m_cameraX, m_cameraY, m_cameraZ};
                float bethesdaCentre[3] = {};
                importer::fnv::CellStreamer::engineToFallout(engineCentre, bethesdaCentre);
                const float centreXY[3] = {
                    bethesdaCentre[0], bethesdaCentre[1], bethesdaCentre[2]};
                ActorPopulationStats actorStats;
                loadGoodspringsActors(
                    dataPath / m_streamPlugin,
                    m_streamLoadOrder.empty() ? nullptr : &m_streamLoadOrder,
                    m_streamer->assets(), centreXY, kActorLoadRadius,
                    kFirstCrowdSkinnedInstance,
                    render::kMaxSkinnedInstances - kFirstCrowdSkinnedInstance,
                    {victor.baseFormId},
                    [this](std::uint32_t referenceFormId) {
                        return m_streamer != nullptr &&
                               m_streamer->referenceBelongsToCurrentWorldspace(
                                   referenceFormId);
                    },
                    m_actors, actorStats);
                if (victorLoaded) {
                    m_victorIndex = static_cast<int>(m_actors.size());
                    m_actors.push_back(std::move(victor));
                }
                queueActorUploads();
                // Dialogue for everybody who has any, in one plugin walk. Runs
                // after Victor joins the list so his own tree is left alone and
                // his base is not asked for a second time.
                {
                    std::string dialogueDetail;
                    if (m_scenarioId.empty() || !m_streamIsSkyrim) {
                        loadActorDialogue(
                            dataPath / m_streamPlugin,
                            m_streamLoadOrder.empty() ? nullptr : &m_streamLoadOrder, m_actors,
                            dialogueDetail);
                    } else {
                        dialogueDetail = "retail Skyrim dialogue owned by BethesdaSession";
                    }
                    VOX_LOGI("newvegas") << "actor dialogue: " << dialogueDetail;
                    // AFTER the dialogue: an actor with nothing to say needs no
                    // voice index, and skipping those is most of the town.
                    std::string voiceDetail;
                    loadActorVoices(
                        dataPath, m_streamPlugin, m_modDirectories, m_actors, voiceDetail);
                    VOX_LOGI("newvegas") << "actor voices: " << voiceDetail;
                }
                arrangeActorParadeIfRequested();
                VOX_LOGI("newvegas") << "actors: " << actorStats.detail;
                for (const SkinnedActor& actor : m_actors) {
                    VOX_LOGI("newvegas")
                        << "  actor " << actor.name << " slot=" << actor.instanceSlot << " at ("
                        << actor.position[0] << ", " << actor.position[1] << ", "
                        << actor.position[2] << ") verts=" << actor.character.vertices.size()
                        << " parts=" << actor.character.parts.size()
                        << " unresolvedBones=" << actor.character.unresolvedBoneCount
                        << " bindConflicts=" << actor.character.conflictingInverseBindCount
                        << " clip=" << (actor.idleClip.tracks.empty() ? "none" : "idle")
                        << (actor.canTalk()
                                ? (" topics=" + std::to_string(actor.tree.nodes.size()))
                                : std::string());
                }
            }
            if (m_victorIndex >= 0) {
                const SkinnedActor& loaded = m_actors[static_cast<std::size_t>(m_victorIndex)];
                VOX_LOGI("newvegas") << "Victor animation: " << loaded.animationStatus;
                VOX_LOGI("newvegas") << "Victor load: " << loaded.timing;
                VOX_LOGI("newvegas") << "Victor voice: " << loaded.voice.status;
            }
        }

        // AFTER the doorstep decision, which would otherwise overwrite it ten
        // lines later -- the first attempt at this set walk mode above and was
        // silently undone here, so every headless fly capture stayed ON FOOT.
        // F toggles this interactively; these are the same switch for a
        // headless run, which cannot press a key.
        if (std::getenv("ODAI_FNV_SPAWN_POS") != nullptr &&
            std::getenv("ODAI_FNV_WALK") == nullptr) {
            // Placing the camera explicitly implies flying it: walk mode
            // re-snaps Y to the ground every frame, so an authored height
            // survived exactly one frame.
            m_walkMode = false;
        }
        if (const char* flyEnv = std::getenv("ODAI_FNV_FLY")) {
            if (flyEnv[0] != '\0' && flyEnv[0] != '0') {
                m_walkMode = false;
            }
        }
    }
    if (m_resumeState.has_value()) {
        m_cameraX = m_resumeState->position[0];
        m_cameraY = m_resumeState->position[1];
        m_cameraZ = m_resumeState->position[2];
        m_yawDegrees = m_resumeState->yawDegrees;
        m_pitchDegrees = m_resumeState->pitchDegrees;
        m_walkMode = true;
        VOX_LOGI("newvegas") << "resumed camera (" << m_cameraX << ", " << m_cameraY
                             << ", " << m_cameraZ << ")";
    }
    // Skyrim has no special startup companion. Populate only after resume has
    // supplied the final camera and space identity, otherwise a saved Bannered
    // Mare session scans the default Tamriel spawn and carries those actors
    // through the interior load.
    if (m_streamIsSkyrim) {
        reloadActorsForCurrentSpace();
    }
    if (m_interiorStarted) {
        VOX_LOGI("newvegas")
            << "interior-only residency active: exterior cells=0, distant LOD=off, water=0";
        return true;
    }
    VOX_LOGI("newvegas") << "streaming " << m_streamer->availableCellCount()
                         << " cells from " << m_streamDirectory
                         << " (load radius " << config.loadRadius
                         << ", unload " << config.unloadRadius << ")";
    loadDistantLandLod();
    return true;
}

// Distant landscape, from the game's own LOD pyramid.
//
// OFF BY DEFAULT, because the two obvious placements are both measured wrong
// and the right one is not built yet. Set ODAI_FNV_LOD_TIER=4|8|16|32 to load
// one tier across the whole worldspace, which is how the numbers below were
// taken and is still the fastest way to look at a tier.
//
// What a whole-world single tier does, on the Mojave:
//
//   level16  64 tiles, 112576 triangles, 64 textures, 40 ms
//   level4   1024 tiles, 1976092 triangles, 1020 textures, 1130 ms
//
// and neither is usable:
//
//  * A COARSE TIER SITS ABOVE THE DETAILED TERRAIN. level16 resamples 16 cells
//    per tile, which averages a valley away, so Goodsprings renders drowned in
//    a smooth tan surface with the road and rooftops poking through it. The
//    error is thousands of units, not tens, so the sink below cannot reach it
//    -- sinking that far would bury the distant mountains it exists to draw.
//  * A FINE TIER EXHAUSTS THE TEXTURE TABLE. Terrain LOD names one diffuse per
//    tile, and the bindless table holds kBindlessTargetTextureCapacity = 1024
//    total. level4's 1020 leaves nothing for the world itself, so EVERY surface
//    in the frame loses its texture and falls back to the hashed pastel that
//    stands in for one. It does not look like a texture-budget failure; it
//    looks like the renderer broke.
//
// So the design this needs is per-tile chunks with tier RINGS -- fine tiles
// just outside the loaded cells, coarser further out, tiles overlapping the
// loaded square excluded -- which bounds both the triangle count and, more
// importantly, the texture count. That is the next step, and the two numbers
// above are the budget it has to fit inside.
//
// Morrowind ships no distant land whatsoever, and Oblivion's is a different
// naming scheme with a single 32-cell tier, so this currently covers FNV and
// Fallout 3 only. An absent tier is not an error here -- it logs and leaves the
// horizon as it was.
void BethesdaApp::loadDistantLandLod() {
    if (m_streamer == nullptr) {
        return;
    }
    // 0 disables, and is the default. Tiers are cell widths: 4, 8, 16, 32.
    int tierCells = 0;
    if (const char* env = std::getenv("ODAI_FNV_LOD_TIER")) {
        tierCells = std::atoi(env);
    }
    if (tierCells <= 0) {
        return;
    }
    if (!importer::fnv::landLodTierExists(importer::fnv::LandLodSet::Terrain, tierCells)) {
        VOX_LOGW("newvegas") << "ODAI_FNV_LOD_TIER=" << tierCells
                             << " is not one of 4, 8, 16, 32; distant LOD disabled";
        return;
    }
    std::int32_t minX = 0;
    std::int32_t minZ = 0;
    std::int32_t maxX = 0;
    std::int32_t maxZ = 0;
    if (!m_streamer->cellGridBounds(minX, minZ, maxX, maxZ)) {
        return;
    }
    // Enough to put the LOD surface under the detailed terrain everywhere the
    // two overlap. Small against a cell (4096) and invisible at the distances
    // this geometry is seen from.
    float sinkUnits = 96.0f;
    if (const char* env = std::getenv("ODAI_FNV_LOD_SINK")) {
        sinkUnits = static_cast<float>(std::atof(env));
    }

    const importer::fnv::FalloutAssetSource& assets = m_streamer->assets();
    importer::ImportedScene scene;
    scene.sourceTag = "fnv_lod";
    importer::fnv::LandLodTierStats stats;
    std::string error;
    const auto start = std::chrono::steady_clock::now();
    const bool ok = importer::fnv::appendLandLodTier(
        [&](const std::string& path, std::vector<std::uint8_t>& bytes) {
            return assets.resolveMesh(path, bytes, error);
        },
        [&](const std::string& path, std::vector<std::uint8_t>& bytes) {
            return assets.resolveTexture(path, bytes, error);
        },
        m_streamWorldspace, importer::fnv::LandLodSet::Terrain, tierCells,
        minX, minZ, maxX, maxZ, sinkUnits, scene, stats, error);
    if (!ok) {
        VOX_LOGI("newvegas") << "no distant LOD for " << m_streamWorldspace << ": " << error;
        return;
    }
    importer::buildImportedScenePackedRenderData(scene);
    importer::buildImportedScenePageRanges(scene);
    const std::size_t chunk = m_renderer.addImportedSceneChunk(scene);
    const double ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    if (chunk == render::Renderer::kInvalidImportedChunkIndex) {
        VOX_LOGW("newvegas") << "distant LOD upload failed";
        return;
    }
    m_distantLodChunk = chunk;
    VOX_LOGI("newvegas") << "distant LOD level" << tierCells << ": " << stats.tilesParsed
                         << " tiles, " << stats.triangles << " triangles, " << stats.textures
                         << " textures, sink " << sinkUnits << " units, in " << ms << " ms";
}

void BethesdaApp::updateSkyrimTerrainLod(const float bethesdaPosition[3]) {
    // A walking-character demo deliberately has no world subject. Avoid
    // parsing and uploading the 49-tile BTR ring merely to hide it again in
    // the renderer; that work dominated both startup and steady-state memory.
    const char* drawMode = std::getenv("ODAI_FNV_DRAW");
    if (drawMode != nullptr && std::strcmp(drawMode, "actors") == 0) {
        return;
    }
    if (!m_streamIsSkyrim || m_streamer == nullptr) {
        return;
    }

    constexpr std::int32_t kTileCells = importer::fnv::kLandLodBlockCells;
    constexpr std::int32_t kTileRadius = 3;
    const float cellSize = m_streamer->cellWorldSize();
    if (cellSize <= 0.0f) {
        return;
    }
    const auto cellX = static_cast<std::int32_t>(std::floor(bethesdaPosition[0] / cellSize));
    const auto cellZ = static_cast<std::int32_t>(std::floor(bethesdaPosition[1] / cellSize));
    const auto tileX = importer::fnv::landLodTileOrigin(cellX, kTileCells);
    const auto tileZ = importer::fnv::landLodTileOrigin(cellZ, kTileCells);
    const bool fixedCaptureLod =
        m_captureRoutePreloadActive && m_captureSkyrimLodBoundsValid;
    if ((fixedCaptureLod && m_captureSkyrimTerrainLodFrozen) ||
        (!fixedCaptureLod && m_skyrimTerrainLodTileValid &&
         tileX == m_skyrimTerrainLodTileX && tileZ == m_skyrimTerrainLodTileZ)) {
        return;
    }

    const std::int32_t firstX =
        (fixedCaptureLod ? m_captureSkyrimLodMinTileX : tileX) -
        (kTileRadius * kTileCells);
    const std::int32_t firstZ =
        (fixedCaptureLod ? m_captureSkyrimLodMinTileZ : tileZ) -
        (kTileRadius * kTileCells);
    const std::int32_t lastX =
        (fixedCaptureLod ? m_captureSkyrimLodMaxTileX : tileX) +
        (kTileRadius * kTileCells);
    const std::int32_t lastZ =
        (fixedCaptureLod ? m_captureSkyrimLodMaxTileZ : tileZ) +
        (kTileRadius * kTileCells);
    importer::ImportedScene scene;
    importer::fnv::LandLodTierStats stats;
    std::string error;
    const auto start = std::chrono::steady_clock::now();
    const importer::fnv::FalloutAssetSource& assets = m_streamer->assets();
    std::vector<std::string> lodWorldspaces =
        m_streamer->currentWorldspaceEditorIdAncestry();
    if (!m_skyrimTerrainLodWorldspace.empty()) {
        const auto cached = std::find(
            lodWorldspaces.begin(), lodWorldspaces.end(), m_skyrimTerrainLodWorldspace);
        if (cached != lodWorldspaces.end()) {
            std::rotate(lodWorldspaces.begin(), cached, cached + 1);
        }
    }
    bool built = false;
    std::string attempts;
    for (const std::string& worldspace : lodWorldspaces) {
        importer::ImportedScene candidate;
        importer::fnv::LandLodTierStats candidateStats;
        std::string candidateError;
        if (importer::fnv::appendLandLodTier(
                [&](const std::string& path, std::vector<std::uint8_t>& bytes) {
                    return assets.resolveMesh(path, bytes, candidateError);
                },
                [&](const std::string& path, std::vector<std::uint8_t>& bytes) {
                    return assets.resolveTexture(path, bytes, candidateError);
                },
                worldspace, importer::fnv::LandLodSet::SkyrimTerrain, kTileCells,
                firstX, firstZ, lastX, lastZ, 96.0f, candidate, candidateStats,
                candidateError)) {
            scene = std::move(candidate);
            scene.sourceTag = "skyrim_terrain_lod:" + worldspace;
            stats = candidateStats;
            built = true;
            if (m_skyrimTerrainLodWorldspace != worldspace) {
                VOX_LOGI("newvegas") << "Skyrim terrain LOD for "
                                     << m_streamWorldspace << " resolved from "
                                     << worldspace;
                m_skyrimTerrainLodWorldspace = worldspace;
            }
            break;
        }
        if (!attempts.empty()) attempts += "; ";
        attempts += worldspace + ": " + candidateError;
    }
    if (!built) error = attempts;

    std::size_t replacement = render::Renderer::kInvalidImportedChunkIndex;
    if (built) {
        // Remove only the coarse triangles fully covered by detailed LAND.
        // Removing an entire four-cell BTR tile when one edge cell overlaps
        // creates a three-cell void (the mountain bug); keeping every triangle
        // paints coarse square height/texture patches over Whiterun's detailed
        // approach. Boundary-crossing triangles stay as a narrow overlap seam,
        // where the 96-unit sink gives detailed LAND depth precedence.
        const std::int32_t detailedRadius = std::max(0, m_streamer->config().loadRadius);
        const float detailMinX = static_cast<float>(cellX - detailedRadius) * cellSize;
        const float detailMaxX = static_cast<float>(cellX + detailedRadius + 1) * cellSize;
        const float detailMinZ = static_cast<float>(cellZ - detailedRadius) * cellSize;
        const float detailMaxZ = static_cast<float>(cellZ + detailedRadius + 1) * cellSize;
        std::size_t trimmedTriangles = 0u;
        for (const importer::ImportedSceneInstance& instance : scene.instances) {
            if (instance.meshIndex >= scene.meshes.size()) {
                continue;
            }
            importer::ImportedSceneMesh& mesh = scene.meshes[instance.meshIndex];
            const float tileOffsetX = instance.transform[3];
            const float tileOffsetZ = -instance.transform[11];
            std::vector<std::uint32_t> keptIndices;
            std::vector<importer::ImportedSceneMeshPart> keptParts;
            keptIndices.reserve(mesh.indices.size());
            keptParts.reserve(mesh.parts.size());
            for (const importer::ImportedSceneMeshPart& sourcePart : mesh.parts) {
                importer::ImportedSceneMeshPart part = sourcePart;
                part.firstIndex = static_cast<std::uint32_t>(keptIndices.size());
                const std::size_t first = sourcePart.firstIndex;
                const std::size_t end = std::min<std::size_t>(
                    mesh.indices.size(), first + sourcePart.indexCount);
                for (std::size_t index = first; index + 2u < end; index += 3u) {
                    bool fullyCoveredByDetailedLand = true;
                    for (std::size_t corner = 0; corner < 3u; ++corner) {
                        const std::uint32_t vertexIndex = mesh.indices[index + corner];
                        if (vertexIndex >= mesh.vertices.size()) {
                            fullyCoveredByDetailedLand = false;
                            break;
                        }
                        const importer::ImportedSceneVertex& vertex = mesh.vertices[vertexIndex];
                        const float worldX = tileOffsetX + vertex.position[0];
                        const float worldZ = tileOffsetZ + vertex.position[1];
                        if (fixedCaptureLod) {
                            const importer::CellCoord detailedCell{
                                static_cast<std::int32_t>(std::floor(worldX / cellSize)),
                                static_cast<std::int32_t>(std::floor(worldZ / cellSize))};
                            if (!m_capturePinnedCells.contains(detailedCell)) {
                                fullyCoveredByDetailedLand = false;
                                break;
                            }
                        } else if (worldX <= detailMinX || worldX >= detailMaxX ||
                                   worldZ <= detailMinZ || worldZ >= detailMaxZ) {
                            fullyCoveredByDetailedLand = false;
                            break;
                        }
                    }
                    if (fullyCoveredByDetailedLand) {
                        ++trimmedTriangles;
                        continue;
                    }
                    keptIndices.push_back(mesh.indices[index]);
                    keptIndices.push_back(mesh.indices[index + 1u]);
                    keptIndices.push_back(mesh.indices[index + 2u]);
                }
                part.indexCount = static_cast<std::uint32_t>(keptIndices.size()) - part.firstIndex;
                if (part.indexCount != 0u) {
                    keptParts.push_back(part);
                }
            }
            mesh.indices = std::move(keptIndices);
            mesh.parts = std::move(keptParts);
        }
        importer::buildImportedScenePackedRenderData(scene);
        importer::buildImportedScenePageRanges(scene);
        replacement = m_renderer.addImportedSceneChunk(scene);
        VOX_LOGI("newvegas") << "Skyrim terrain LOD handoff trimmed "
                             << trimmedTriangles << " covered BTR triangles";
    }
    if (m_distantLodChunk != render::Renderer::kInvalidImportedChunkIndex) {
        m_renderer.removeImportedSceneChunk(m_distantLodChunk);
    }
    m_distantLodChunk = replacement;
    m_skyrimTerrainLodTileX = tileX;
    m_skyrimTerrainLodTileZ = tileZ;
    m_skyrimTerrainLodTileValid = true;
    m_captureSkyrimTerrainLodFrozen = fixedCaptureLod;

    if (!built) {
        VOX_LOGI("newvegas") << "no Skyrim terrain LOD around tile " << tileX << ","
                             << tileZ << ": " << error;
        return;
    }
    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    VOX_LOGI("newvegas") << (fixedCaptureLod ? "fixed capture Skyrim terrain LOD" :
                                                "Skyrim terrain LOD around tile")
                         << " " << tileX << "," << tileZ
                         << ": " << stats.tilesParsed << " BTR tiles, " << stats.triangles
                         << " triangles, " << stats.textures << " textures, in " << ms << " ms";
}

void BethesdaApp::updateSkyrimObjectLod(const float bethesdaPosition[3]) {
    // See updateSkyrimTerrainLod: actor-only runs should not build the 49-tile
    // BTO skyline behind a deliberately hidden world.
    const char* drawMode = std::getenv("ODAI_FNV_DRAW");
    if (drawMode != nullptr && std::strcmp(drawMode, "actors") == 0) {
        return;
    }
    if (!m_streamIsSkyrim || m_streamer == nullptr) {
        return;
    }
    // The fixed Whiterun presentation has no visible loading phase. Build its
    // parent proxy once the child ring is final, so the handoff below can trim
    // against the complete detailed residency set instead of rebuilding the
    // 7x7 BTO window once per arriving cell.
    if (m_whiterunReferenceShowcase && !m_streamer->isStreamingIdle()) {
        return;
    }

    constexpr std::int32_t kTileCells = importer::fnv::kLandLodBlockCells;
    // A 3x3 BTO window only reaches roughly 20-31k units from the camera,
    // depending on where the camera sits inside its current tile. Skyrim's
    // mountains routinely cross that boundary while still clearly visible,
    // producing a hard partial-mountain silhouette even though every triangle
    // in the resident tiles is submitted. Seven tiles per axis keeps complete
    // object-LOD features resident to at least ~49k units in every direction;
    // the existing imported page culling keeps off-screen tiles out of draws.
    constexpr std::int32_t kTileRadius = 3;
    const float cellSize = m_streamer->cellWorldSize();
    if (cellSize <= 0.0f) {
        return;
    }
    const auto cellX = static_cast<std::int32_t>(std::floor(bethesdaPosition[0] / cellSize));
    const auto cellZ = static_cast<std::int32_t>(std::floor(bethesdaPosition[1] / cellSize));
    const auto tileX = importer::fnv::landLodTileOrigin(cellX, kTileCells);
    const auto tileZ = importer::fnv::landLodTileOrigin(cellZ, kTileCells);
    const bool fixedCaptureLod =
        m_captureRoutePreloadActive && m_captureSkyrimLodBoundsValid;
    if ((fixedCaptureLod && m_captureSkyrimObjectLodFrozen) ||
        (!fixedCaptureLod && m_skyrimObjectLodTileValid &&
         tileX == m_skyrimObjectLodTileX && tileZ == m_skyrimObjectLodTileZ)) {
        return;
    }

    // BTO geometry is already in world space and all nine tiles share one
    // atlas, so build the window as ONE ImportedScene. Besides avoiding nine
    // atlas decodes, this gives the renderer one replaceable chunk when the
    // camera crosses a four-cell boundary.
    importer::ImportedScene scene;
    importer::fnv::LandLodTierStats stats;
    std::string error;
    const auto start = std::chrono::steady_clock::now();
    const importer::fnv::FalloutAssetSource& assets = m_streamer->assets();
    const std::int32_t firstX =
        (fixedCaptureLod ? m_captureSkyrimLodMinTileX : tileX) -
        (kTileRadius * kTileCells);
    const std::int32_t firstZ =
        (fixedCaptureLod ? m_captureSkyrimLodMinTileZ : tileZ) -
        (kTileRadius * kTileCells);
    const std::int32_t lastX =
        (fixedCaptureLod ? m_captureSkyrimLodMaxTileX : tileX) +
        (kTileRadius * kTileCells);
    const std::int32_t lastZ =
        (fixedCaptureLod ? m_captureSkyrimLodMaxTileZ : tileZ) +
        (kTileRadius * kTileCells);
    std::vector<std::string> lodWorldspaces =
        m_streamer->currentWorldspaceEditorIdAncestry();
    if (!m_skyrimObjectLodWorldspace.empty()) {
        const auto cached = std::find(
            lodWorldspaces.begin(), lodWorldspaces.end(), m_skyrimObjectLodWorldspace);
        if (cached != lodWorldspaces.end()) {
            std::rotate(lodWorldspaces.begin(), cached, cached + 1);
        }
    }
    bool built = false;
    std::string attempts;
    for (const std::string& worldspace : lodWorldspaces) {
        importer::ImportedScene candidate;
        importer::fnv::LandLodTierStats candidateStats;
        std::string candidateError;
        if (importer::fnv::appendLandLodTier(
                [&](const std::string& path, std::vector<std::uint8_t>& bytes) {
                    return assets.resolveMesh(path, bytes, candidateError);
                },
                [&](const std::string& path, std::vector<std::uint8_t>& bytes) {
                    return assets.resolveTexture(path, bytes, candidateError);
                },
                worldspace, importer::fnv::LandLodSet::SkyrimObjects, kTileCells,
                firstX, firstZ, lastX, lastZ, 0.0f, candidate, candidateStats,
                candidateError)) {
            scene = std::move(candidate);
            scene.sourceTag = "skyrim_object_lod:" + worldspace;
            stats = candidateStats;
            built = true;
            if (m_skyrimObjectLodWorldspace != worldspace) {
                VOX_LOGI("newvegas") << "Skyrim object LOD for "
                                     << m_streamWorldspace << " resolved from "
                                     << worldspace;
                m_skyrimObjectLodWorldspace = worldspace;
            }
            break;
        }
        if (!attempts.empty()) attempts += "; ";
        attempts += worldspace + ": " + candidateError;
    }
    if (!built) error = attempts;

    std::size_t replacement = render::Renderer::kInvalidImportedChunkIndex;
    if (built) {
        // A BTO is a coarse stand-in for objects covered by four detailed
        // cells. It must disappear before the camera reaches the tile, not
        // only after crossing its boundary: otherwise Whiterun's whole coarse
        // city pops back in when the player is a few hundred units across the
        // neighbouring tile edge. Keep a one-cell hand-off band around every
        // BTO tile. At the pinned southern approach Whiterun is ~6100 units
        // beyond that band and remains complete; on the near approach its BTO
        // yields to the authored city shells before the low-detail rocks and
        // roofs become visible.
        const auto distanceToInterval = [](float value, float low, float high) {
            if (value < low) {
                return low - value;
            }
            return value > high ? value - high : 0.0f;
        };
        const bool keepAllBtoShapes =
            std::getenv("ODAI_DEBUG_KEEP_SKYRIM_BTO") != nullptr;
        std::vector<std::size_t> nearMeshIndices;
        if (!keepAllBtoShapes) {
            for (std::int32_t tz = firstZ; tz <= lastZ; tz += kTileCells) {
                for (std::int32_t tx = firstX; tx <= lastX; tx += kTileCells) {
                    const float minX = static_cast<float>(tx) * cellSize;
                    const float maxX = static_cast<float>(tx + kTileCells) * cellSize;
                    const float minZ = static_cast<float>(tz) * cellSize;
                    const float maxZ = static_cast<float>(tz + kTileCells) * cellSize;
                    if (fixedCaptureLod) {
                        // Whiterun is a child worldspace whose visible 3x3
                        // detail ring occupies only part of Tamriel's 4x4 BTO
                        // tile. Requiring all sixteen parent cells to be pinned
                        // therefore retained the LargeRef city proxy directly
                        // over the resident gate—the blurry "door" in the
                        // reference view. The fixed camera has already proved
                        // its complete visible ring resident, so its containing
                        // parent tile yields as one authored LOD unit.
                        if (m_whiterunReferenceShowcase) {
                            if (tx != tileX || tz != tileZ) {
                                continue;
                            }
                        } else {
                            bool detailedTileResident = true;
                            for (std::int32_t cz = tz;
                                 cz < tz + kTileCells && detailedTileResident; ++cz) {
                                for (std::int32_t cx = tx; cx < tx + kTileCells; ++cx) {
                                    if (!m_capturePinnedCells.contains(
                                            importer::CellCoord{cx, cz})) {
                                        detailedTileResident = false;
                                        break;
                                    }
                                }
                            }
                            if (!detailedTileResident) {
                                continue;
                            }
                        }
                    } else {
                        const float dx = distanceToInterval(bethesdaPosition[0], minX, maxX);
                        const float dz = distanceToInterval(bethesdaPosition[1], minZ, maxZ);
                        if ((dx * dx) + (dz * dz) >= cellSize * cellSize) {
                            continue;
                        }
                    }
                    const std::string meshNamePrefix =
                        "lod" + std::to_string(kTileCells) + "_" + std::to_string(tx) + "_" +
                        std::to_string(tz);
                    const bool childCityTile = m_whiterunReferenceShowcase ||
                        m_streamer->hasChildWorldspaceCellInRange(
                            tx, tz, tx + kTileCells - 1, tz + kTileCells - 1);
                    for (std::size_t meshIndex = 0; meshIndex < scene.meshes.size(); ++meshIndex) {
                        importer::ImportedSceneMesh& mesh = scene.meshes[meshIndex];
                        const std::string& meshName = mesh.name;
                        const bool belongsToTile =
                            meshName == meshNamePrefix ||
                            meshName.starts_with(meshNamePrefix + "_");
                        if (!belongsToTile) {
                            continue;
                        }
                        if (childCityTile) {
                            // The fixed reference prewarms every detailed cell
                            // that can contribute to its view. Its containing
                            // parent tile can therefore yield completely; this
                            // also avoids retaining a coarse gate leaf whose
                            // triangles straddle a detailed-cell boundary.
                            if (m_whiterunReferenceShowcase) {
                                nearMeshIndices.push_back(meshIndex);
                                continue;
                            }
                            // A parent BTO may merge the whole walled city into
                            // one mesh, so dropping the mesh creates holes when
                            // only part of the child is resident. Clip at the
                            // triangle centroid instead: detailed child cells
                            // replace their exact footprint while the proxy
                            // remains everywhere else.
                            std::vector<std::uint32_t> keptIndices;
                            std::vector<importer::ImportedSceneMeshPart> keptParts;
                            keptIndices.reserve(mesh.indices.size());
                            keptParts.reserve(mesh.parts.size());
                            for (const importer::ImportedSceneMeshPart& sourcePart : mesh.parts) {
                                importer::ImportedSceneMeshPart part = sourcePart;
                                part.firstIndex = static_cast<std::uint32_t>(keptIndices.size());
                                const std::size_t begin = sourcePart.firstIndex;
                                const std::size_t end = std::min(
                                    begin + static_cast<std::size_t>(sourcePart.indexCount),
                                    mesh.indices.size());
                                for (std::size_t i = begin; i + 2u < end; i += 3u) {
                                    const std::uint32_t ia = mesh.indices[i];
                                    const std::uint32_t ib = mesh.indices[i + 1u];
                                    const std::uint32_t ic = mesh.indices[i + 2u];
                                    if (ia >= mesh.vertices.size() || ib >= mesh.vertices.size() ||
                                        ic >= mesh.vertices.size()) {
                                        continue;
                                    }
                                    const float centreX =
                                        (mesh.vertices[ia].position[0] +
                                         mesh.vertices[ib].position[0] +
                                         mesh.vertices[ic].position[0]) / 3.0f;
                                    const float centreEngineZ =
                                        (mesh.vertices[ia].position[2] +
                                         mesh.vertices[ib].position[2] +
                                         mesh.vertices[ic].position[2]) / 3.0f;
                                    const std::int32_t cellX = static_cast<std::int32_t>(
                                        std::floor(centreX / cellSize));
                                    const std::int32_t cellZ = static_cast<std::int32_t>(
                                        std::floor(-centreEngineZ / cellSize));
                                    if (m_streamer->isExteriorCellResident(
                                            m_streamer->currentWorldspaceFormId(), cellX, cellZ)) {
                                        continue;
                                    }
                                    keptIndices.push_back(ia);
                                    keptIndices.push_back(ib);
                                    keptIndices.push_back(ic);
                                }
                                part.indexCount = static_cast<std::uint32_t>(keptIndices.size()) -
                                    part.firstIndex;
                                if (part.indexCount != 0u) {
                                    keptParts.push_back(part);
                                }
                            }
                            mesh.indices = std::move(keptIndices);
                            mesh.parts = std::move(keptParts);
                            continue;
                        }
                        const bool largeReference = meshName.ends_with("_largeref");
                        if (!largeReference) {
                            nearMeshIndices.push_back(meshIndex);
                        }
                    }
                }
            }
        } else {
            VOX_LOGW("newvegas")
                << "ODAI_DEBUG_KEEP_SKYRIM_BTO active: near-tile object LOD handoff disabled";
        }
        scene.instances.erase(
            std::remove_if(
                scene.instances.begin(), scene.instances.end(),
                [&](const importer::ImportedSceneInstance& instance) {
                    return std::find(
                               nearMeshIndices.begin(), nearMeshIndices.end(),
                               static_cast<std::size_t>(instance.meshIndex)) !=
                           nearMeshIndices.end();
                }),
            scene.instances.end());
        if (m_whiterunReferenceShowcase) {
            VOX_LOGI("showcase")
                << "Whiterun reference object-LOD handoff removed "
                << nearMeshIndices.size()
                << " parent BTO mesh(es) from the detailed gate tile";
        }
        importer::buildImportedScenePackedRenderData(scene);
        importer::buildImportedScenePageRanges(scene);
        replacement = m_renderer.addImportedSceneChunk(scene);
    }
    if (m_skyrimObjectLodChunk != render::Renderer::kInvalidImportedChunkIndex) {
        m_renderer.removeImportedSceneChunk(m_skyrimObjectLodChunk);
    }
    m_skyrimObjectLodChunk = replacement;
    m_skyrimObjectLodTileX = tileX;
    m_skyrimObjectLodTileZ = tileZ;
    m_skyrimObjectLodTileValid = true;
    m_captureSkyrimObjectLodFrozen = fixedCaptureLod;

    if (!built) {
        VOX_LOGI("newvegas") << "no Skyrim object LOD around tile " << tileX << ","
                             << tileZ << ": " << error;
        return;
    }
    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    VOX_LOGI("newvegas") << (fixedCaptureLod ? "fixed capture Skyrim object LOD" :
                                                "Skyrim object LOD around tile")
                         << " " << tileX << "," << tileZ
                         << ": " << stats.tilesParsed << " BTO tiles, " << stats.triangles
                         << " triangles, " << stats.textures << " atlas texture(s), in "
                         << ms << " ms";
}

// Headless check that collision is actually doing its job, because walking
// around by hand is not a repeatable test and "it felt solid" is not a result.
//
// Two properties, both of which fail silently: terrain has to be sampleable
// everywhere the player can stand (otherwise they fall through), and a point
// placed inside an obstacle has to come back out (otherwise buildings are
// scenery).
void BethesdaApp::runCollisionSelfTest() {
    const float step = 256.0f;
    int sampled = 0;
    int grounded = 0;
    float minClearance = 1e30f;
    float maxClearance = -1e30f;
    for (int dz = -6; dz <= 6; ++dz) {
        for (int dx = -6; dx <= 6; ++dx) {
            const float x = m_cameraX + (static_cast<float>(dx) * step);
            const float z = m_cameraZ + (static_cast<float>(dz) * step);
            float height = 0.0f;
            ++sampled;
            // Terrain only here: this samples coverage, and mixing in geometry
            // would report a rooftop as "the ground".
            if (!m_collision.terrainHeight(x, z, height)) {
                continue;
            }
            ++grounded;
            minClearance = std::min(minClearance, height);
            maxClearance = std::max(maxClearance, height);
        }
    }

    // Walk a straight line and confirm the player actually travels. The failure
    // this catches is collision pinning them in place a few steps in, which is
    // exactly what a single box per mesh did and what "it looked solid" cannot
    // distinguish from working.
    const float walkStep = 40.0f;
    float px = m_cameraX;
    float pz = m_cameraZ;
    float py = m_cameraY;
    float travelled = 0.0f;
    int blockedSteps = 0;
    for (int step = 0; step < 200; ++step) {
        const float beforeX = px;
        const float beforeZ = pz;
        pz -= walkStep;  // due north in engine space
        m_collision.resolveHorizontal(px, py, pz);
        float ground = 0.0f;
        if (m_collision.groundHeight(px, pz, py - m_collision.tuning().eyeHeight, ground)) {
            py = ground + m_collision.tuning().eyeHeight;
        }
        const float moved =
            std::sqrt(((px - beforeX) * (px - beforeX)) + ((pz - beforeZ) * (pz - beforeZ)));
        travelled += moved;
        if (moved < walkStep * 0.25f) {
            ++blockedSteps;
        }
    }

    // The opposite failure: collision so permissive that nothing blocks. Probe
    // each wall triangle's centroid and require the player to be pushed out of
    // it. Walking freely is only good news if walls still stop you.
    int wallProbes = 0;
    int wallBlocks = 0;
    m_collision.forEachNearbyTriangle(
        m_cameraX, m_cameraZ, [&](const CollisionWorld::Triangle& triangle) {
            if (triangle.normalY >= m_collision.tuning().minWalkableNormalY) {
                return;
            }
            const float minY = std::min({triangle.v[1], triangle.v[4], triangle.v[7]});
            const float maxY = std::max({triangle.v[1], triangle.v[4], triangle.v[7]});
            if ((maxY - minY) < m_collision.tuning().eyeHeight) {
                return;  // too short to be a wall the player walks into
            }
            const float cx = (triangle.v[0] + triangle.v[3] + triangle.v[6]) / 3.0f;
            const float cy = (triangle.v[1] + triangle.v[4] + triangle.v[7]) / 3.0f;
            const float cz = (triangle.v[2] + triangle.v[5] + triangle.v[8]) / 3.0f;
            float wx = cx;
            float wz = cz;
            ++wallProbes;
            m_collision.resolveHorizontal(wx, cy + m_collision.tuning().eyeHeight * 0.5f, wz);
            if (std::sqrt(((wx - cx) * (wx - cx)) + ((wz - cz) * (wz - cz))) > 1.0f) {
                ++wallBlocks;
            }
        });

    VOX_LOGI("newvegas") << "collision self-test: walls " << wallBlocks << "/" << wallProbes
                         << " pushed a probe off their surface";
    VOX_LOGI("newvegas") << "collision self-test: terrain " << grounded << "/" << sampled
                         << " sample points grounded (heights " << minClearance << ".."
                         << maxClearance << "); walked " << travelled << " of "
                         << (walkStep * 200.0f) << " units due north, " << blockedSteps
                         << "/200 steps blocked; " << m_collision.triangleCount()
                         << " collision triangles across " << m_collision.residentCellCount()
                         << " cells";
}

void BethesdaApp::updateStreaming(float deltaSeconds) {
    if (!m_streamer) {
        return;
    }
    if (m_interiorStarted) {
        if (m_bethesdaCollisionBroadPhaseDirty && m_bethesdaSessionConfigured) {
            m_bethesdaSession.physics().optimizeBroadPhase();
            m_bethesdaCollisionBroadPhaseDirty = false;
        }
        return;
    }

    const odai::math::Vector3 residencyPosition = thirdPersonPlayerShowcase()
        ? bethesdaPlayerEyePosition()
        : odai::math::Vector3{m_cameraX, m_cameraY, m_cameraZ};
    // Velocity by differencing the gameplay-residency origin rather than
    // reading the movement code's own. In third person this is the player,
    // never the displaced camera boom.
    // code's own: that one is zeroed by collision and jumping, which would make
    // the planner think a walking player had stopped.
    float velocity[3] = {0.0f, 0.0f, 0.0f};
    if (m_hasPreviousCameraPosition && deltaSeconds > 0.0f) {
        velocity[0] = (residencyPosition.x - m_previousCameraX) / deltaSeconds;
        velocity[1] = (residencyPosition.y - m_previousCameraY) / deltaSeconds;
        velocity[2] = (residencyPosition.z - m_previousCameraZ) / deltaSeconds;
    }
    m_previousCameraX = residencyPosition.x;
    m_previousCameraY = residencyPosition.y;
    m_previousCameraZ = residencyPosition.z;
    m_hasPreviousCameraPosition = true;

    // The planner ranks cells in FALLOUT space; the camera moves in engine
    // space. Converting is not optional -- feeding engine coordinates straight
    // in makes the grid's second axis the player's altitude, so streaming
    // follows how high they are rather than where they are.
    const float enginePosition[3] = {
        residencyPosition.x, residencyPosition.y, residencyPosition.z};
    float falloutPosition[3] = {0.0f, 0.0f, 0.0f};
    float falloutVelocity[3] = {0.0f, 0.0f, 0.0f};
    importer::fnv::CellStreamer::engineToFallout(enginePosition, falloutPosition);
    importer::fnv::CellStreamer::engineToFallout(velocity, falloutVelocity);
    m_streamer->update(m_renderer, falloutPosition, falloutVelocity);
    // Mesh bodies are usable immediately; the global Jolt broad-phase rebuild
    // is only an acceleration pass. Batch it after the load ring settles so a
    // 25-cell arrival performs one rebuild rather than one per rendered frame.
    if (m_bethesdaCollisionBroadPhaseDirty &&
        m_streamer->isStreamingIdle() && m_bethesdaSessionConfigured) {
        m_bethesdaSession.physics().optimizeBroadPhase();
        m_bethesdaCollisionBroadPhaseDirty = false;
    }
    updateSkyrimTerrainLod(falloutPosition);
    updateSkyrimObjectLod(falloutPosition);

    // Actor placements and quest aliases are gameplay residency, not renderer
    // payload. Rebuild the nearby Skyrim population only after the asynchronous
    // cell ring settles; syncBethesdaActors then publishes stable RecordKeys,
    // XLCN and XLRT to BethesdaSession before any alias can materialize loot.
    // Persistent runtime state remains in BethesdaWorld when an actor leaves;
    // reloadActorsForCurrentSpace disables presentation/physics and a later
    // revisit re-enables the same ObjectId instead of creating a duplicate.
    if (m_streamIsSkyrim && m_skyrimActorResidencyDirty &&
        m_streamer->stats().residency.loadingCount == 0u &&
        m_doorTransitionPhase == DoorTransitionPhase::None &&
        talkingActor() == nullptr && m_bethesdaSession.giftMenuRequests().empty()) {
        reloadActorsForCurrentSpace();
    }

    // Once, after the first ring has settled.
    if (!m_collisionSelfTestDone && std::getenv("ODAI_FNV_COLLISION_TEST") != nullptr &&
        m_streamer->stats().residency.loadingCount == 0u &&
        m_streamer->stats().residentChunks > 0u) {
        m_collisionSelfTestDone = true;
        runCollisionSelfTest();
    }

    m_streamStatsLogTimer += deltaSeconds;
    if (m_streamStatsLogTimer >= 2.0f) {
        m_streamStatsLogTimer = 0.0f;
        const importer::fnv::CellStreamerStats stats = m_streamer->stats();
        VOX_LOGI("streamer") << "resident=" << stats.residentChunks
                             << " loading=" << stats.residency.loadingCount
                             << " loaded=" << stats.scenesLoaded
                             << " evicted=" << stats.residency.evictions
                             << " wasted=" << stats.residency.wastedLoads
                             << " missing=" << stats.residency.unavailableCells
                             << " applyMs(last/worst)=" << stats.lastApplyMs
                             << "/" << stats.worstApplyMs
                             << " buildMs(last/worst)=" << stats.lastBuildMs
                             << "/" << stats.worstBuildMs
                             << " cache(hit/miss)=" << stats.cacheHits << "/" << stats.cacheMisses
                             << " cacheLoadMs=" << stats.lastCacheLoadMs
                             << " fxSkipped=" << stats.effectMeshesSkipped
                             << " nodeParseFails=" << stats.nodeParseFailures
                             << " droppedLayers=" << stats.droppedTerrainLayers
                             << " waterCells=" << stats.waterPatchesLoaded
                             << " blendedDraws=" << stats.blendedPartsLoaded;
    }
}

void BethesdaApp::onTick(float deltaSeconds) {
    // A recording runs on its own clock. Everything downstream of here -- the
    // tour, the wander, the animation, the day cycle -- takes this dt, so the
    // world advances one authored frame per rendered frame however long the
    // rendering took. See setCaptureSequence.
    if (m_captureFixedDt > 0.0f) {
        deltaSeconds = m_captureFixedDt;
    }
    if (m_bethesdaSessionConfigured) {
        syncBethesdaPlayerState(false);
        const bethesda::BethesdaSessionStep sessionStep =
            m_bethesdaSession.advance(static_cast<double>(deltaSeconds),
                [this](std::uint64_t, double fixedStepSeconds) {
                    stepBethesdaActorControllers(
                        static_cast<float>(fixedStepSeconds));
                    if (m_meleeAttackPending) {
                        const float yaw = m_yawDegrees * (kPi / 180.0f);
                        const float pitch = m_pitchDegrees * (kPi / 180.0f);
                        const float horizontal = std::cos(pitch);
                        const bethesda::MeleeAttackResult attack =
                            m_bethesdaSession.performMeleeAttack(
                                m_bethesdaSession.playerObject(),
                                {std::cos(yaw) * horizontal, std::sin(pitch),
                                    std::sin(yaw) * horizontal});
                        if (attack.accepted) {
                            m_toasts.push(
                                attack.hit ? (attack.killed ? "Enemy defeated" : "Hit")
                                           : "Attack missed",
                                attack.hit ? attack.target.toString() : std::string{},
                                "melee-attack");
                        }
                        m_meleeAttackPending = false;
                    }
                });
        m_sessionInterpolationAlpha =
            static_cast<float>(sessionStep.clock.interpolationAlpha);
        if (m_skyrimCitySpawnSettlementPending) {
            (void)settleSkyrimCityShowcasePlayer();
        }
        if (m_bethesdaControllerOwnsCamera) pullBethesdaPlayerControllerState();
        pullBethesdaActorControllerStates();
        // Runtime simulation-level changes do not alter the streamer's placed
        // actor set. Marking the catalog dirty here caused a full retail actor
        // rescan/rebuild after its own synchronization, creating a self-fed
        // multi-second hitch loop. Stream cell callbacks above are the sole
        // presentation-residency authority.
        (void)m_renderer.applyRuntimeRenderDeltas(sessionStep.renderDeltas);
        for (const std::string& diagnostic : sessionStep.diagnostics) {
            VOX_LOGW("runtime") << diagnostic;
        }
        const bool saveDown = keyDown(m_window, GLFW_KEY_F5);
        const bool loadDown = keyDown(m_window, GLFW_KEY_F9);
        if (saveDown && !m_gameplaySaveKeyLatch) (void)saveGameplayState();
        if (loadDown && !m_gameplayLoadKeyLatch) (void)loadGameplayState();
        m_gameplaySaveKeyLatch = saveDown;
        m_gameplayLoadKeyLatch = loadDown;
    }
    // Keep renderer-side water and rigid machinery on the tour's fixed clock.
    // During capture pre-roll, hold phase zero while streaming, TAA, and
    // exposure warm up. Frame zero then starts camera and machinery together,
    // independent of how long extraction took on this machine.
    const bool capturePreroll = m_captureFixedDt > 0.0f && !m_captureStarted;
    if (!capturePreroll) {
        m_visualTimeSeconds += deltaSeconds;
    }
    m_renderer.setVisualTimeSeconds(m_visualTimeSeconds);
    // Before anything reads input: the menu toggle decided here gates whether
    // camera movement runs at all this frame.
    pollNavInput(deltaSeconds);
    updateTes3JournalInput();
    updateGiftMenu();
    const bool giftMenuOpen = m_bethesdaSessionConfigured &&
        !m_bethesdaSession.giftMenuRequests().empty();
    const bool meleeDown =
        glfwGetMouseButton(m_window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
    if (meleeDown && !m_meleeAttackButtonLatch && m_bethesdaSessionConfigured &&
        !m_menuOpen && !m_tes3JournalOpen && !giftMenuOpen && m_talkingActor < 0 &&
        m_doorTransitionPhase == DoorTransitionPhase::None) {
        m_meleeAttackPending = true;
    }
    m_meleeAttackButtonLatch = meleeDown;
    m_toasts.update(deltaSeconds);
    // The banner is a WORLD event, so it pauses with the world. Letting it run
    // under an open menu means a discovery fades in and out behind a modal
    // panel and the player never sees the one thing it existed to tell them --
    // and while it lasted, two pieces of large centred type sat on top of each
    // other. Held here, it plays the moment the menu closes.
    //
    // A conversation counts for the same reason, and now more literally: the
    // dialogue card is centred large type, and "Goodsprings / Location
    // discovered" landed straight across Victor's first two replies.
    if (!m_menuOpen && !m_tes3JournalOpen && !giftMenuOpen && m_talkingActor < 0) {
        m_banner.update(deltaSeconds);
    }
    // Region lookup walks the cell index, so it is polled a few times a second
    // rather than every frame. A player cannot cross a 4096-unit cell in less
    // than that even sprinting, so nothing is missed.
    m_regionPollSeconds += deltaSeconds;
    if (m_regionPollSeconds >= 0.25f) {
        m_regionPollSeconds = 0.0f;
        updateRegionDiscovery();
    }

    // Normal traversal realizes one actor per visible frame so a newly entered
    // 40-person cell cannot starve event polling. The Whiterun cold-start path
    // calls the same helper with an unlimited budget before the frame loop.
    realizePendingActorUploads(1u);

    updateSkyrimPlayerAvatar(deltaSeconds);

    // Pose every actor every frame, whether or not it is being talked to -- the
    // idle clip is what makes an actor read as someone standing there rather
    // than a statue of them.
    if (!m_actors.empty()) {
        // The renderer's skinned path was built for a party, so submitting a
        // whole worldspace's actors without culling repeats every body part in
        // the skinning, velocity, depth, main and shadow passes. Keep a wide
        // margin around the camera and view cone: actors near enough to notice
        // never pop when the player turns, while guards across Whiterun cost no
        // GPU work until they can contribute to the frame.
        static const float actorDrawDistance = [] {
            constexpr float kDefaultActorDrawDistance = 6000.0f;
            const char* value = std::getenv("ODAI_FNV_ACTOR_DRAW_DISTANCE");
            if (value == nullptr) {
                return kDefaultActorDrawDistance;
            }
            const float parsed = static_cast<float>(std::atof(value));
            return parsed > 0.0f ? parsed : kDefaultActorDrawDistance;
        }();
        constexpr float kAlwaysVisibleDistance = 1200.0f;
        // cos(100 degrees): retain actors slightly behind the view plane for
        // wide displays, shadows and a pop-free turn.
        constexpr float kViewDotMargin = -0.17364818f;
        const float yawRadians = m_yawDegrees * (kPi / 180.0f);
        const float forwardX = std::cos(yawRadians);
        const float forwardZ = std::sin(yawRadians);
        const float maxDistanceSquared = actorDrawDistance * actorDrawDistance;
        const float alwaysVisibleDistanceSquared =
            kAlwaysVisibleDistance * kAlwaysVisibleDistance;
        for (std::size_t actorIndex = 0; actorIndex < m_actors.size(); ++actorIndex) {
            SkinnedActor& actor = m_actors[actorIndex];
            const float dx = actor.position[0] - m_cameraX;
            const float dz = actor.position[2] - m_cameraZ;
            const float distanceSquared = (dx * dx) + (dz * dz);
            bool visible = distanceSquared <= maxDistanceSquared;
            if (visible && distanceSquared > alwaysVisibleDistanceSquared) {
                const float inverseDistance = 1.0f / std::sqrt(distanceSquared);
                const float viewDot = ((dx * forwardX) + (dz * forwardZ)) * inverseDistance;
                visible = viewDot >= kViewDotMargin;
            }
            visible = visible || static_cast<int>(actorIndex) == m_talkingActor;
            actor.renderVisible = visible;
            if (actor.uploaded) {
                m_renderer.setSkinnedActorVisible(actor.instanceSlot, visible);
            }
        }

        // Move before posing: the pose folds in world placement, so wandering
        // afterwards would draw everyone one frame behind where they are.
        // Do not bind an actor to the first NAVM cell that happens to finish.
        // Streaming workers complete out of order; until the initial resident
        // set is idle, the actor's own/nearest mesh may simply not have arrived.
        if (!m_bethesdaSessionConfigured) {
            const ActorNavigationWorld* actorNavigation =
                (m_streamer && m_streamer->isStreamingIdle()) ? &m_actorNavigation : nullptr;
            updateActorWandering(
                m_actors, deltaSeconds, actorNavigation,
                [this](float x, float z, float referenceY, float& outHeight) {
                    // The ACTOR's own foot height is the reference, not the
                    // camera's. groundHeight uses it to reject ceilings and to
                    // raise onto walkable geometry, so someone on a porch stays
                    // on the porch instead of sinking to the terrain under it.
                    return m_streamer
                        ? m_collision.groundHeight(x, z, referenceY, outHeight)
                        : false;
                },
                [this](float& x, float& z, float feetY, float headY, float radius) {
                    if (m_streamer) {
                        m_collision.resolveHorizontalFor(
                            x, z, feetY, headY, radius, m_collision.tuning().stepHeight);
                    }
                },
                m_talkingActor);
        }

        updateActorPoses(m_actors, deltaSeconds);
        for (const SkinnedActor& actor : m_actors) {
            if (!actor.uploaded || !actor.renderVisible) {
                continue;
            }
            render::ImportedSkinnedActorFrameData pose{};
            pose.boneMatrices = actor.poseScratch;
            m_renderer.setSkinnedActorPose(actor.instanceSlot, pose);
        }
    }

    // ODAI_FNV_TALK opens a conversation on the first tick, so the dialogue UI
    // can be checked from a --screenshot run, which cannot press E. With no
    // value it picks Victor; with an EditorID it picks that actor, which is the
    // only way to photograph anyone else's conversation.
    // ODAI_FNV_VICTOR_TALK is the old spelling and still works.
    {
        const char* autoTalkEnv = std::getenv("ODAI_FNV_TALK");
        if (autoTalkEnv == nullptr) {
            autoTalkEnv = std::getenv("ODAI_FNV_VICTOR_TALK");
        }
        static bool autoTalked = false;
        if (autoTalkEnv != nullptr && !autoTalked && m_talkingActor < 0 &&
            !m_tes3JournalOpen && !m_actors.empty() &&
            (m_streamer == nullptr || m_streamer->isStreamingIdle())) {
            autoTalked = true;
            const std::string wanted = toLowerAscii(autoTalkEnv);
            // "1" is the historical value of ODAI_FNV_VICTOR_TALK and names
            // nobody, so it means "whoever the default speaker is".
            const bool wantsDefault = wanted.empty() || wanted == "1";
            for (std::size_t i = 0; i < m_actors.size(); ++i) {
                const bool mayHaveBethesdaTopics = m_bethesdaSessionConfigured &&
                    (m_streamIsSkyrim || m_streamIsMorrowind) &&
                    m_actors[i].placed && !m_actors[i].runtimeDead;
                if (!m_actors[i].canTalk() && !mayHaveBethesdaTopics) {
                    continue;
                }
                if (wantsDefault ? (static_cast<int>(i) == m_victorIndex)
                                 : (toLowerAscii(m_actors[i].name) == wanted)) {
                    beginConversation(static_cast<int>(i));
                    break;
                }
            }
            if (m_talkingActor < 0) {
                VOX_LOGW("newvegas") << "auto-talk: no actor with dialogue matching \""
                                     << autoTalkEnv << "\"";
            }
        }
    }

    SkinnedActor* speaker = talkingActor();

    // Talking to Victor. Held keys are edge-detected by keyDown(), so a choice
    // is taken once per press rather than once per frame.
    // Edge-latched per slot. keyDown() is level-triggered, so an unlatched read
    // takes one choice PER FRAME: a normal ~100 ms press on "1" consumed six
    // choices, ran off the end of the branch and closed the conversation before
    // it could be read, which looked exactly like Victor refusing to talk.
    for (int slot = 0; slot < 9; ++slot) {
        const bool pressed = keyDown(m_window, GLFW_KEY_1 + slot);
        const bool edge = pressed && !m_choiceKeyLatch[slot];
        m_choiceKeyLatch[slot] = pressed;
        if (!edge || speaker == nullptr) {
            continue;
        }
        const auto choices = speaker->runtime.availableChoices();
        if (static_cast<std::size_t>(slot) < choices.size()) {
            chooseConversationChoice(static_cast<std::size_t>(slot));
        }
    }
    // Highlight-and-confirm, alongside the number keys rather than instead of
    // them. The numbers are the fast path for someone at a keyboard; up/down
    // and Accept are the only ones that work from a couch, and they come from
    // UiNavInput so a gamepad drives them identically (pollNavInput already
    // folds the d-pad, the left stick and the arrow keys into the same
    // actions, with auto-repeat, so a held direction scrolls instead of
    // jumping one row per frame).
    if (speaker != nullptr) {
        const auto choices = speaker->runtime.availableChoices();
        const auto choiceCount = static_cast<int>(choices.size());
        // A new node means a new set of replies; leaving the old index in
        // place would highlight an unrelated line, or one that no longer
        // exists.
        const dialogue::DialogueNode* currentNode = speaker->runtime.currentNode();
        const std::string currentNodeId = currentNode != nullptr ? currentNode->id : std::string();
        if (currentNodeId != m_dialogueChoiceNodeId) {
            m_dialogueChoiceNodeId = currentNodeId;
            m_dialogueChoice = 0;
            // ODAI_FNV_DIALOGUE_SELECT=<n> starts on the nth reply (0-based).
            // The highlight is the whole point of this panel and a --screenshot
            // run cannot press a key, so without this the only row that can
            // ever be photographed is the first one -- and "the highlight is
            // drawn" and "the highlight tracks the selection" are different
            // claims.
            if (const char* fromEnv = std::getenv("ODAI_FNV_DIALOGUE_SELECT")) {
                m_dialogueChoice = std::atoi(fromEnv);
            }
        }
        if (choiceCount > 0) {
            if (m_nav.pressed(ui::UiNavAction::Up)) {
                // Wrapping, not clamping: a four-item list on a controller is
                // faster to reach the end of by going up once.
                m_dialogueChoice = (m_dialogueChoice + choiceCount - 1) % choiceCount;
            }
            if (m_nav.pressed(ui::UiNavAction::Down)) {
                m_dialogueChoice = (m_dialogueChoice + 1) % choiceCount;
            }
            m_dialogueChoice = std::clamp(m_dialogueChoice, 0, choiceCount - 1);
            if (m_nav.pressed(ui::UiNavAction::Accept)) {
                chooseConversationChoice(static_cast<std::size_t>(m_dialogueChoice));
                m_dialogueChoice = 0;
            }
        } else {
            m_dialogueChoice = 0;
        }
    }
    if (speaker != nullptr &&
        (speaker->runtime.isFinished() || speaker->runtime.currentNode() == nullptr)) {
        endConversation();
        speaker = nullptr;
    }
    // One call site rather than one per way a conversation can advance (a
    // choice, opening it, the auto-talk hook). It is a no-op once the current
    // node has been spoken, so polling it costs a map lookup and cannot start a
    // line twice.
    if (speaker != nullptr && !m_streamCacheDirectory.empty()) {
        speakActorLine(
            *speaker, std::filesystem::path(m_streamCacheDirectory) / "voice", m_audio);
    }

    // ESCAPE LEAVES A CONVERSATION AND NOTHING ELSE. It used to quit the game
    // when no one was speaking, and keyDown is a LEVEL read -- a single press
    // spans ~10 frames at 60 fps. So closing a dialogue box with Escape ended
    // the conversation on the first of those frames and then, on the second
    // frame of the same press, found no speaker and quit. Not an edge case:
    // pressing Escape to back out of a conversation quit the game every time.
    //
    // Edge-latched now, and the quit is gone from here entirely -- backing out
    // of something must never be the same keystroke as leaving the game. With
    // no speaker, Escape falls through to the pause menu, which is what the
    // HUD hint has always claimed it does.
    const bool escapeDown = keyDown(m_window, GLFW_KEY_ESCAPE);
    const bool escapePressed = escapeDown && !m_escapeLatch;
    m_escapeLatch = escapeDown;
    if (escapePressed && speaker != nullptr) {
        endConversation();
        return;
    }

    updateDoorTransition(deltaSeconds);
    if (!m_tes3JournalOpen && m_doorTransitionPhase == DoorTransitionPhase::None) {
        if (!m_whiterunReferenceShowcase) {
            updateCamera(deltaSeconds);
        }
        updateStreaming(deltaSeconds);
    }
    updateSkyrimAmbience(deltaSeconds);
    m_stateSaveSeconds += deltaSeconds;
    if (!m_whiterunReferenceShowcase && m_stateSaveSeconds >= 5.0f) {
        m_stateSaveSeconds = 0.0f;
        saveTraversalState(false);
    }

    // Time-of-day controls. Edge-latched so a held key steps once.
    const bool bracketLeft = keyDown(m_window, GLFW_KEY_LEFT_BRACKET);
    if (bracketLeft && !m_bracketLeftLatch) {
        m_timeOfDayHours = std::fmod(m_timeOfDayHours - 1.0f + 24.0f, 24.0f);
    }
    m_bracketLeftLatch = bracketLeft;

    const bool bracketRight = keyDown(m_window, GLFW_KEY_RIGHT_BRACKET);
    if (bracketRight && !m_bracketRightLatch) {
        m_timeOfDayHours = std::fmod(m_timeOfDayHours + 1.0f, 24.0f);
    }
    m_bracketRightLatch = bracketRight;

    // P QUITS. A deliberate, single-purpose key, because the alternative was
    // Escape doing double duty as "close this" and "leave the game" -- and a key
    // that both dismisses a panel and exits has no safe way to be pressed.
    //
    // P was the day-cycle pause toggle, which is still on the pause menu's
    // "Day cycle" row; the [ and ] keys still step time directly.
    const bool quitPressed = keyDown(m_window, GLFW_KEY_P);
    if (quitPressed && !m_quitKeyLatch) {
        glfwSetWindowShouldClose(m_window, GLFW_TRUE);
    }
    m_quitKeyLatch = quitPressed;

    // Edge-latched: holding E must not re-trigger on the door you arrive next
    // to, which is always within range of the one you just came through.
    // ONE activation target per frame, decided here rather than by which `if`
    // happens to run first. An actor wins over a door at equal reach: Victor
    // stands a step from Doc Mitchell's porch, and a player pressing E while
    // facing him means to talk, not to go inside.
    const odai::math::Vector3 activationOrigin = thirdPersonPlayerShowcase()
        ? bethesdaPlayerEyePosition()
        : odai::math::Vector3{m_cameraX, m_cameraY, m_cameraZ};
    const float cameraPosition[3] = {
        activationOrigin.x, activationOrigin.y, activationOrigin.z};
    m_activationLootActor = (m_talkingActor >= 0 || m_tes3JournalOpen)
        ? -1 : findLootableActorInReach();
    m_activationActor = -1;
    if (!m_tes3JournalOpen && m_talkingActor < 0 && m_activationLootActor < 0) {
        m_activationActor = (m_bethesdaSessionConfigured && m_streamIsMorrowind)
            ? findTes3DialogueActorInReach(
                  cameraPosition, m_yawDegrees * (kPi / 180.0f))
            : (m_bethesdaSessionConfigured && m_streamIsSkyrim)
            ? findBethesdaDialogueActorInReach(
                  cameraPosition, m_yawDegrees * (kPi / 180.0f))
            : findActorInReach(
                  m_actors, cameraPosition, m_yawDegrees * (kPi / 180.0f));
    }
    const bool clawPuzzleInReach = m_activationLootActor < 0 &&
        m_activationActor < 0 && !m_menuOpen &&
        m_talkingActor < 0 && goldenClawPuzzleInReach();
    static constexpr std::array<int, 3> kClawRingKeys = {
        GLFW_KEY_1, GLFW_KEY_2, GLFW_KEY_3};
    bool rotatedClawRing = false;
    for (std::size_t ring = 0u; ring < kClawRingKeys.size(); ++ring) {
        const bool ringDown = keyDown(m_window, kClawRingKeys[ring]);
        const bool ringEdge = ringDown && !m_goldenClawRingKeyLatch[ring];
        m_goldenClawRingKeyLatch[ring] = ringDown;
        if (clawPuzzleInReach && ringEdge && !rotatedClawRing) {
            rotatedClawRing = rotateGoldenClawRing(ring);
        }
    }
    // Latch BEFORE the branch below. It used to be updated after an early
    // return that the Victor path took, so the latch stayed false while E was
    // held: the next frame saw a fresh "press" and walked the player through
    // Doc Mitchell's door -- which is a step from where Victor stands -- so the
    // conversation opened and an interior load closed it in the same keypress.
    const bool doorPressed = keyDown(m_window, GLFW_KEY_E);
    const bool doorEdge = doorPressed && !m_doorKeyLatch;
    m_doorKeyLatch = doorPressed;
    if (!m_tes3JournalOpen && doorEdge && m_activationLootActor >= 0) {
        (void)lootActor(m_activationLootActor);
        return;
    }
    if (!m_tes3JournalOpen && doorEdge && m_activationActor >= 0) {
        beginConversation(m_activationActor);
        // The line itself is started by the single speakActorLine poll above,
        // on the next tick.
        return;  // E opened a conversation; do not also walk through a door
    }
    if (!m_tes3JournalOpen && doorEdge && clawPuzzleInReach) {
        (void)useGoldenClawPuzzle();
    } else if (!m_tes3JournalOpen && doorEdge) {
        const int doorIndex = findUsableDoor();
        if (doorIndex >= 0) {
            useDoor(m_doors[static_cast<std::size_t>(doorIndex)]);
        }
    }

    const bool walkPressed = keyDown(m_window, GLFW_KEY_F);
    if (!m_tes3JournalOpen && walkPressed && !m_walkModeLatch) {
        m_walkMode = !m_walkMode;
    }
    m_walkModeLatch = walkPressed;

    const bool tabPressed = keyDown(m_window, GLFW_KEY_TAB);
    if (!m_tes3JournalOpen && tabPressed && !m_tabLatch) {
        m_mouseCaptured = !m_mouseCaptured;
        setMouseCaptured(m_mouseCaptured);
    }
    m_tabLatch = tabPressed;

    if (!m_dayCyclePaused) {
        m_timeOfDayHours = std::fmod(m_timeOfDayHours + (m_dayCycleHoursPerSecond * deltaSeconds), 24.0f);
    }
    applyTimeOfDay();
}


// ---------------------------------------------------------------------------
// Console-friendly UI: nav input, region-discovery toasts, Pip-Boy HUD.

namespace {

// One palette, used by every piece of chrome below, so the HUD reads as one
// instrument rather than a pile of independently styled boxes. Fallout keeps
// its Pip-Boy phosphor; TES3 switches these four values to warm parchment when
// its runtime is configured.
ui::UiColor kPipGreen{0.42f, 1.00f, 0.52f, 1.00f};
ui::UiColor kPipGreenDim{0.26f, 0.66f, 0.32f, 1.00f};

// Greedy word wrap against a baked font's own metrics.
//
// The HUD's addText draws one unwrapped run, which is fine for a status strip
// and wrong for a paragraph: Victor's longer lines ran off the side of the
// screen. Written here rather than reached for through rich_text because that
// path parses <b>/<color=...> markup, and this text comes out of a 1998 game's
// dialogue records -- a stray '<' in a line is content, not a tag.
//
// A single word longer than maxWidth is emitted on its own over-long line
// rather than split mid-word: it cannot be made to fit, and breaking it is
// less readable than letting one line run.
std::vector<std::string> wrapTextToWidth(
    const ui::Font& font, const std::string& text, float maxWidth
) {
    std::vector<std::string> lines;
    if (text.empty()) {
        return lines;
    }
    if (maxWidth <= 0.0f) {
        lines.push_back(text);
        return lines;
    }
    std::string line;
    std::size_t wordStart = 0;
    while (wordStart <= text.size()) {
        std::size_t wordEnd = text.find(' ', wordStart);
        if (wordEnd == std::string::npos) {
            wordEnd = text.size();
        }
        const std::string word = text.substr(wordStart, wordEnd - wordStart);
        if (!word.empty()) {
            const std::string candidate = line.empty() ? word : (line + " " + word);
            if (!line.empty() && font.measureText(candidate) > maxWidth) {
                lines.push_back(line);
                line = word;
            } else {
                line = candidate;
            }
        }
        if (wordEnd == text.size()) {
            break;
        }
        wordStart = wordEnd + 1;
    }
    if (!line.empty()) {
        lines.push_back(line);
    }
    return lines;
}
ui::UiColor kPipPanel{0.02f, 0.07f, 0.03f, 0.82f};
ui::UiColor kPipPanelSolid{0.02f, 0.07f, 0.03f, 0.95f};

void useMorrowindUiPalette() {
    kPipGreen = {0.88f, 0.78f, 0.60f, 1.00f};
    kPipGreenDim = {0.64f, 0.59f, 0.50f, 1.00f};
    kPipPanel = {0.10f, 0.09f, 0.075f, 0.86f};
    kPipPanelSolid = {0.10f, 0.09f, 0.075f, 0.96f};
}

void useSkyrimUiPalette() {
    // Skyrim's menus are restrained and nearly monochrome. Keep the shared HUD
    // widgets, but replace Fallout's phosphor green with cool silver over a
    // charcoal translucent panel.
    kPipGreen = {0.88f, 0.90f, 0.94f, 1.00f};
    kPipGreenDim = {0.58f, 0.62f, 0.69f, 1.00f};
    kPipPanel = {0.035f, 0.040f, 0.050f, 0.86f};
    kPipPanelSolid = {0.035f, 0.040f, 0.050f, 0.96f};
}

float deadzone(float value, float threshold) {
    if (value > -threshold && value < threshold) {
        return 0.0f;
    }
    return value;
}

}  // namespace

void BethesdaApp::pollNavInput(float deltaSeconds) {
    m_nav.beginFrame();

    const bool keyUp = keyDown(m_window, GLFW_KEY_UP);
    const bool keyDownArrow = keyDown(m_window, GLFW_KEY_DOWN);
    const bool keyLeft = keyDown(m_window, GLFW_KEY_LEFT);
    const bool keyRight = keyDown(m_window, GLFW_KEY_RIGHT);
    bool accept = keyDown(m_window, GLFW_KEY_ENTER);
    bool cancel = keyDown(m_window, GLFW_KEY_ESCAPE);
    bool menu = cancel;

    // Gamepad, when one is present. GLFW's gamepad mapping gives the same
    // button indices for every recognized pad, so this needs no per-controller
    // handling -- an unmapped joystick simply reports false here rather than
    // producing garbage input.
    float stickX = 0.0f;
    float stickY = 0.0f;
    GLFWgamepadstate pad{};
    const bool hasPad = glfwJoystickIsGamepad(GLFW_JOYSTICK_1) == GLFW_TRUE &&
        glfwGetGamepadState(GLFW_JOYSTICK_1, &pad) == GLFW_TRUE;
    if (hasPad) {
        constexpr float kStickDeadzone = 0.25f;
        stickX = deadzone(pad.axes[GLFW_GAMEPAD_AXIS_LEFT_X], kStickDeadzone);
        stickY = deadzone(pad.axes[GLFW_GAMEPAD_AXIS_LEFT_Y], kStickDeadzone);
        accept = accept || pad.buttons[GLFW_GAMEPAD_BUTTON_A] == GLFW_PRESS;
        cancel = cancel || pad.buttons[GLFW_GAMEPAD_BUTTON_B] == GLFW_PRESS;
        menu = menu || pad.buttons[GLFW_GAMEPAD_BUTTON_START] == GLFW_PRESS;
        m_nav.setAction(ui::UiNavAction::PrevTab,
                        pad.buttons[GLFW_GAMEPAD_BUTTON_LEFT_BUMPER] == GLFW_PRESS);
        m_nav.setAction(ui::UiNavAction::NextTab,
                        pad.buttons[GLFW_GAMEPAD_BUTTON_RIGHT_BUMPER] == GLFW_PRESS);
    }

    // All three sources for a direction -- stick, d-pad, arrow key -- combined
    // into ONE level, then set once.
    //
    // This used to call m_navStick.apply() and then fold the digital sources in
    // with `if (key) setAction(action, true)`. Two setAction calls per action
    // per frame, and the second one saw the first's `false` as the previous
    // frame's state, so every frame an arrow key was held produced a fresh
    // press edge. The dialogue list scrolled at frame rate -- about 100 items a
    // second on this machine -- which is what "the selection moves too fast"
    // actually was. The auto-repeat timing was never involved.
    int stickDirectionX = 0;
    int stickDirectionY = 0;
    m_navStick.resolveDirection(stickX, stickY, stickDirectionX, stickDirectionY);
    const bool padUp = hasPad && pad.buttons[GLFW_GAMEPAD_BUTTON_DPAD_UP] == GLFW_PRESS;
    const bool padDown = hasPad && pad.buttons[GLFW_GAMEPAD_BUTTON_DPAD_DOWN] == GLFW_PRESS;
    const bool padLeft = hasPad && pad.buttons[GLFW_GAMEPAD_BUTTON_DPAD_LEFT] == GLFW_PRESS;
    const bool padRight = hasPad && pad.buttons[GLFW_GAMEPAD_BUTTON_DPAD_RIGHT] == GLFW_PRESS;
    m_nav.setAction(ui::UiNavAction::Up, keyUp || padUp || stickDirectionY < 0);
    m_nav.setAction(ui::UiNavAction::Down, keyDownArrow || padDown || stickDirectionY > 0);
    m_nav.setAction(ui::UiNavAction::Left, keyLeft || padLeft || stickDirectionX < 0);
    m_nav.setAction(ui::UiNavAction::Right, keyRight || padRight || stickDirectionX > 0);
    m_nav.setAction(ui::UiNavAction::Accept, accept);
    m_nav.setAction(ui::UiNavAction::Cancel, cancel);
    m_nav.setAction(ui::UiNavAction::Menu, menu);

    m_navRepeat.update(m_nav, deltaSeconds);

    // Any directional or accept input means a controller is driving. The mouse
    // takes it back in updateCamera, which is where mouse motion is already
    // being read.
    m_navDriving = m_navDriving || m_nav.active;
    m_nav.active = false;

    // Not while a conversation is up: Escape is both Menu and Cancel, and this
    // runs at the top of the tick, before the dialogue's own Escape handling.
    // Backing out of a conversation therefore closed it AND opened the menu in
    // one press -- two modal states toggled by one key, which reads as the menu
    // appearing for no reason.
    const bool giftMenuOpen = m_bethesdaSessionConfigured &&
        !m_bethesdaSession.giftMenuRequests().empty();
    if (m_nav.pressed(ui::UiNavAction::Menu) &&
        m_talkingActor < 0 && !giftMenuOpen && !m_tes3JournalOpen) {
        // The weather picker is a sub-page of the menu, so Escape backs out of
        // it one level rather than closing everything. Closing straight to the
        // world would make the picker feel like a separate mode the player had
        // fallen into, and there would be no way to change your mind about one
        // weather without leaving the menu entirely.
        if (m_menuOpen && m_weatherPickerOpen) {
            m_weatherPickerOpen = false;
        } else if (m_menuOpen && m_compatibilityPanelOpen) {
            m_compatibilityPanelOpen = false;
        } else {
            m_menuOpen = !m_menuOpen;
            // Releasing the mouse with the menu up is what makes it usable on
            // PC; on a controller it costs nothing.
            setMouseCaptured(!m_menuOpen);
        }
    }
}

void BethesdaApp::syncTes3JournalPanel() {
    if (!m_streamIsMorrowind || !m_bethesdaSessionConfigured ||
        m_tes3JournalPanel == nullptr ||
        m_bethesdaSession.tes3().content() == nullptr) return;
    const auto& runtime = m_bethesdaSession.tes3();
    const auto& content = runtime.content();
    std::vector<ui::Tes3JournalPanel::Entry> entries;
    entries.reserve(runtime.journal().chronology().size());
    for (const bethesda::Tes3JournalVisit& visit : runtime.journal().chronology()) {
        const auto questIt = content->dialogues().find(visit.quest);
        if (questIt == content->dialogues().end()) continue;
        const bethesda::Tes3DialogueDefinition& quest = questIt->second;
        const auto info = std::find_if(quest.infos.begin(), quest.infos.end(),
            [&](const bethesda::Tes3DialogueInfo& item) { return item.record == visit.info; });
        const bethesda::Tes3JournalQuestState* state = runtime.journal().find(quest.id);
        ui::Tes3JournalPanel::Entry entry;
        entry.sequence = visit.sequence;
        entry.questId = quest.id;
        entry.title = bethesda::normalizeTes3Symbol(quest.id) == "tr_m3_tt_bloodstone"
            ? "Bloodstone Pilgrimage" : quest.id;
        entry.text = info != quest.infos.end() ? info->response : std::string{};
        entry.index = visit.index;
        if (state != nullptr) {
            entry.hasStatusFlags = state->hasStatusFlags;
            if (state->classification == bethesda::Tes3JournalQuestClassification::Active) {
                entry.state = ui::Tes3JournalPanel::QuestState::Active;
            } else if (state->classification ==
                       bethesda::Tes3JournalQuestClassification::Completed) {
                entry.state = ui::Tes3JournalPanel::QuestState::Completed;
            }
        }
        entries.push_back(std::move(entry));
    }
    std::vector<std::string> topics;
    topics.reserve(runtime.knownTopics().size());
    for (const bethesda::RecordKey& topic : runtime.knownTopics()) {
        topics.push_back(topic.textId);
    }
    m_tes3JournalPanel->setJournal(std::move(entries), std::move(topics));
    m_tes3JournalSyncedVisits = runtime.journal().chronology().size();
    if (!m_tes3PinnedQuest.empty()) {
        (void)m_tes3JournalPanel->pinQuest(m_tes3PinnedQuest);
    }
}

void BethesdaApp::updateTes3JournalInput() {
    bool journalDown = keyDown(m_window, GLFW_KEY_J);
    GLFWgamepadstate journalPad{};
    if (glfwJoystickIsGamepad(GLFW_JOYSTICK_1) == GLFW_TRUE &&
        glfwGetGamepadState(GLFW_JOYSTICK_1, &journalPad) == GLFW_TRUE) {
        // The controller's View/Back button is the natural journal shortcut;
        // Start remains the pause menu and B remains modal cancel.
        const bool controllerJournal =
            journalPad.buttons[GLFW_GAMEPAD_BUTTON_BACK] == GLFW_PRESS;
        journalDown = journalDown || controllerJournal;
        m_navDriving = m_navDriving || controllerJournal;
    }
    const bool journalEdge = journalDown && !m_tes3JournalKeyLatch;
    m_tes3JournalKeyLatch = journalDown;
    if (journalEdge && m_streamIsMorrowind && m_bethesdaSessionConfigured &&
        m_talkingActor < 0 && !m_menuOpen) {
        m_tes3JournalOpen = !m_tes3JournalOpen;
        if (m_tes3JournalOpen) syncTes3JournalPanel();
        setMouseCaptured(!m_tes3JournalOpen);
    }
    if (!m_tes3JournalOpen || m_tes3JournalPanel == nullptr) return;
    if (m_nav.pressed(ui::UiNavAction::Up)) m_tes3JournalPanel->moveSelection(-1);
    if (m_nav.pressed(ui::UiNavAction::Down)) m_tes3JournalPanel->moveSelection(1);
    if (m_nav.pressed(ui::UiNavAction::PrevTab) ||
        m_nav.pressed(ui::UiNavAction::Left)) {
        const int current = static_cast<int>(m_tes3JournalPanel->view());
        m_tes3JournalPanel->setView(
            static_cast<ui::Tes3JournalPanel::View>((current + 3) % 4));
    }
    if (m_nav.pressed(ui::UiNavAction::NextTab) ||
        m_nav.pressed(ui::UiNavAction::Right)) {
        const int current = static_cast<int>(m_tes3JournalPanel->view());
        m_tes3JournalPanel->setView(
            static_cast<ui::Tes3JournalPanel::View>((current + 1) % 4));
    }
    if (m_nav.pressed(ui::UiNavAction::Accept)) {
        m_tes3JournalPanel->pinSelected();
        m_tes3PinnedQuest = m_tes3JournalPanel->pinnedQuest();
    }
    if (m_nav.pressed(ui::UiNavAction::Cancel)) {
        m_tes3JournalOpen = false;
        setMouseCaptured(true);
    }
}

void BethesdaApp::updateGiftMenu() {
    if (!m_bethesdaSessionConfigured ||
        m_bethesdaSession.giftMenuRequests().empty()) {
        m_presentedGiftMenuSequence = 0u;
        m_giftMenuChoice = 0;
        return;
    }
    const bethesda::GiftMenuRequestState request =
        m_bethesdaSession.giftMenuRequests().front();
    if (m_presentedGiftMenuSequence != request.sequence) {
        m_presentedGiftMenuSequence = request.sequence;
        m_giftMenuChoice = 0;
        endConversation();
        setMouseCaptured(false);
    }
    const bethesda::ObjectId sourceId =
        request.playerGives ? request.player : request.actor;
    const bethesda::RuntimeObject* source =
        m_bethesdaSession.world().find(sourceId);
    const int itemCount = source == nullptr
        ? 0 : static_cast<int>(source->inventory.size());
    m_giftMenuChoice = std::clamp(m_giftMenuChoice, 0, std::max(0, itemCount - 1));
    if (itemCount > 0 && m_nav.pressed(ui::UiNavAction::Up)) {
        m_giftMenuChoice = (m_giftMenuChoice + itemCount - 1) % itemCount;
    }
    if (itemCount > 0 && m_nav.pressed(ui::UiNavAction::Down)) {
        m_giftMenuChoice = (m_giftMenuChoice + 1) % itemCount;
    }
    if (itemCount > 0 && m_nav.pressed(ui::UiNavAction::Accept)) {
        const bethesda::InventoryEntry entry =
            source->inventory[static_cast<std::size_t>(m_giftMenuChoice)];
        const bethesda::GiftTransferResult transfer =
            m_bethesdaSession.transferGiftMenuItem(request.sequence, entry.item, 1);
        if (transfer.accepted) {
            m_toasts.push(
                request.playerGives ? "Item given" : "Item received",
                entry.item.toString(), "gift-transfer:" + entry.item.toString());
        } else {
            m_toasts.push("Gift transfer unavailable", transfer.diagnostic, "gift-transfer-error");
        }
    }
    if (m_nav.pressed(ui::UiNavAction::Cancel)) {
        std::string error;
        if (!m_bethesdaSession.closeGiftMenu(request.sequence, error)) {
            m_toasts.push("Could not close gift menu", error, "gift-menu-close");
        } else {
            m_presentedGiftMenuSequence = 0u;
            m_giftMenuChoice = 0;
            setMouseCaptured(!m_menuOpen && m_talkingActor < 0);
        }
    }
}

void BethesdaApp::updateRegionDiscovery() {
    if (!m_streamer) {
        return;
    }
    const odai::math::Vector3 discoveryOrigin = thirdPersonPlayerShowcase()
        ? bethesdaPlayerEyePosition()
        : odai::math::Vector3{m_cameraX, m_cameraY, m_cameraZ};
    const float position[3] = {
        discoveryOrigin.x, discoveryOrigin.y, discoveryOrigin.z};
    // ODAI_FNV_LOG_REGION=1 traces the cell and region under the camera every
    // poll. "No banner fired" and "the walk never left the region" look the
    // same from the outside, and a traversal test has to tell them apart.
    static const bool s_logRegion = std::getenv("ODAI_FNV_LOG_REGION") != nullptr;
    if (s_logRegion) {
        float fallout[3] = {};
        importer::fnv::CellStreamer::engineToFallout(position, fallout);
        std::string names;
        for (const std::string& name : m_streamer->regionNamesAtEngineSpace(position)) {
            names += names.empty() ? name : (", " + name);
        }
        VOX_LOGI("newvegas") << "cell ("
                             << static_cast<int>(std::floor(fallout[0] / 4096.0f)) << ","
                             << static_cast<int>(std::floor(fallout[1] / 4096.0f))
                             << ") regions: " << (names.empty() ? "<none>" : names);
    }
    for (const std::string& name : m_streamer->regionNamesAtEngineSpace(position)) {
        // insert() reports whether it was new, so the "have I seen this?" check
        // and the record of having seen it are one operation -- there is no
        // window where a second call in the same frame announces it twice.
        if (!m_discoveredRegions.insert(name).second) {
            continue;
        }
        VOX_LOGI("newvegas") << "discovered region: " << name;
        // Keyed on the region so a player standing on a cell boundary, where
        // the streamer flips between two cells, refreshes one announcement
        // instead of queueing a run of identical ones.
        m_banner.push(name, "Location discovered", "region:" + name);
    }

    if (m_interiorStarted) {
        return;
    }
    constexpr float kMarkerDiscoveryRadius = 4096.0f;
    constexpr float kMarkerDiscoveryRadiusSquared =
        kMarkerDiscoveryRadius * kMarkerDiscoveryRadius;
    for (const importer::fnv::FalloutMapMarkerRecord& marker : m_streamer->mapMarkers()) {
        if (marker.deleted || marker.initiallyDisabled || marker.name.empty() ||
            marker.worldspaceFormId != m_streamer->currentWorldspaceFormId() ||
            m_discoveredMarkerIds.contains(marker.referenceFormId)) {
            continue;
        }
        const float markerEngine[3] = {marker.position[0], marker.position[2], -marker.position[1]};
        const float dx = markerEngine[0] - discoveryOrigin.x;
        const float dz = markerEngine[2] - discoveryOrigin.z;
        if ((dx * dx) + (dz * dz) > kMarkerDiscoveryRadiusSquared) {
            continue;
        }
        m_discoveredMarkerIds.insert(marker.referenceFormId);
        m_discoveredLocations.push_back(
            TraversalDiscovery{marker.referenceFormId,
                               m_streamer->currentWorldspaceEditorId(), marker.name});
        VOX_LOGI("newvegas") << "discovered location: " << marker.name;
        m_banner.push(marker.name, "Location discovered", "marker:" + marker.name);
    }
}

void BethesdaApp::saveTraversalState(bool force) {
    // Scenario sessions are persisted exclusively through the versioned ODAI
    // save. Keeping the legacy camera-only traversal JSON beside it would
    // create two competing authorities for player position and world time.
    if (!m_scenarioId.empty() || !m_streamer || m_traversalStatePath.empty() ||
        m_doorTransitionPhase != DoorTransitionPhase::None) {
        return;
    }
    const odai::math::Vector3 savedEye = thirdPersonPlayerShowcase()
        ? bethesdaPlayerEyePosition()
        : odai::math::Vector3{m_cameraX, m_cameraY, m_cameraZ};
    if (!force) {
        float ground = 0.0f;
        if (!m_collision.groundHeight(
                savedEye.x, savedEye.z,
                savedEye.y - m_collision.tuning().eyeHeight, ground) ||
            std::abs((savedEye.y - m_collision.tuning().eyeHeight) - ground) > 24.0f) {
            return;
        }
    }
    TraversalState state;
    state.interior = m_interiorStarted;
    state.worldspaceEditorId = m_streamer->currentWorldspaceEditorId();
    state.interiorEditorId = m_currentInteriorEditorId;
    state.position[0] = savedEye.x;
    state.position[1] = savedEye.y;
    state.position[2] = savedEye.z;
    state.yawDegrees = m_yawDegrees;
    state.pitchDegrees = m_pitchDegrees;
    state.timeOfDayHours = m_timeOfDayHours;
    state.loadOrderFingerprint = m_loadOrderFingerprint;
    if (const importer::fnv::FalloutWeatherRecord* weather =
            m_weatherTables.findWeather(m_activeWeatherFormId)) {
        state.weatherEditorId = weather->editorId;
    }
    state.discoveries = m_discoveredLocations;
    std::string error;
    if (!saveTraversalStateAtomic(m_traversalStatePath, state, error)) {
        VOX_LOGW("newvegas") << "could not save traversal state: " << error;
    }
}

void BethesdaApp::onShutdown() {
    if (std::getenv("ODAI_FNV_ACTORS_LIST") != nullptr) {
        for (const SkinnedActor& actor : m_actors) {
            const float dx = actor.position[0] - actor.wanderOrigin[0];
            const float dy = actor.position[1] - actor.wanderOrigin[1];
            const float dz = actor.position[2] - actor.wanderOrigin[2];
            VOX_LOGI("newvegas")
                << "  actor final " << actor.name << " at ("
                << actor.position[0] << ", " << actor.position[1] << ", "
                << actor.position[2] << ") moved="
                << std::sqrt((dx * dx) + (dy * dy) + (dz * dz))
                << " walking=" << (actor.walking ? "yes" : "no")
                << " nav=" << (actor.projectedToNavigation ? "projected" : "fallback");
        }
    }
    if (!m_whiterunReferenceShowcase &&
        m_bethesdaSessionConfigured && !m_gameplaySavePath.empty()) {
        (void)saveGameplayState();
    }
    if (!m_whiterunReferenceShowcase) {
        saveTraversalState(true);
    }
    clearSkyrimAmbience();
    if (m_streamer) {
        m_streamer->waitIdle();
    }
}

void BethesdaApp::setScenario(std::string id) {
    const bethesda::ScenarioDefinition* scenario = bethesda::findScenario(id);
    if (scenario == nullptr) {
        m_scenarioId = std::move(id);
        return;
    }
    m_scenarioId = scenario->id;
    m_scenarioStartMarker = scenario->startMarker;
    m_streamPlugin = scenario->basePlugin;
    m_streamWorldspace = scenario->worldspace;
    m_streamWorldspaceExplicit = true;
    m_resumeEnabled = false;
    m_explicitStart = true;
}

void BethesdaApp::cacheBethesdaCollisionCell(
    const importer::CellCoord& cell, const importer::ImportedScene& scene) {
    BethesdaCollisionMesh mesh;
    for (const importer::ImportedSceneInstance& instance : scene.instances) {
        if (!instance.initiallyVisible && instance.sourceReferenceFormId != 0u) {
            m_disabledBethesdaCollisionReferences.insert(instance.sourceReferenceFormId);
        }
        if (!m_bethesdaSessionConfigured || !m_streamIsMorrowind ||
            instance.sourceReferenceIdentity.empty()) continue;
        bethesda::RecordKey referenceKey;
        if (!bethesda::parseRecordKey(instance.sourceReferenceIdentity, referenceKey) ||
            referenceKey.kind != bethesda::RecordKeyKind::Tes3Reference) continue;
        const bethesda::ObjectId id = bethesda::ObjectId::persistent(referenceKey);
        if (m_bethesdaSession.world().find(id) != nullptr) continue;
        const auto definition = m_bethesdaSession.tes3().content()->references().find(id);
        if (definition == m_bethesdaSession.tes3().content()->references().end()) continue;
        // Visible architecture and terrain belong to rendering/collision, not
        // the mutable gameplay world. Register only placed records that can be
        // activated, carried, scripted, or simulated. Treating every STAT as
        // an Item produced ~13k heavyweight RuntimeObjects around Balmora and
        // made every fixed tick copy/sort inventories for decorative walls.
        const std::string& type = definition->second.base.recordType;
        const bool gameplayReference =
            type == "NPC_" || type == "CREA" || type == "CONT" ||
            type == "DOOR" || type == "ACTI" || type == "ALCH" ||
            type == "APPA" || type == "ARMO" || type == "BOOK" ||
            type == "CLOT" || type == "INGR" || type == "LIGH" ||
            type == "LOCK" || type == "MISC" || type == "PROB" ||
            type == "REPA" || type == "WEAP";
        if (!gameplayReference) continue;
        bethesda::RuntimeObject object;
        object.id = id;
        object.base = definition->second.base;
        object.persistent = true;
        object.enabled = instance.initiallyVisible && definition->second.enabled;
        object.transform.position = {
            instance.transform[3], instance.transform[7], instance.transform[11]};
        object.originSpace.cell = definition->second.cell;
        object.originSpace.kind = definition->second.interior
            ? bethesda::RuntimeSpaceKind::Interior
            : bethesda::RuntimeSpaceKind::Exterior;
        if (!definition->second.interior) {
            object.originSpace.worldspace =
                bethesda::makeTes3RecordKey("WRLD", "vardenfell");
            if (definition->second.hasCellGrid) {
                object.originSpace.gridX = definition->second.cellGridX;
                object.originSpace.gridZ = definition->second.cellGridZ;
            }
        }
        object.currentSpace = object.originSpace;
        object.interior = definition->second.interior;
        const float scaleX = std::sqrt(
            (instance.transform[0] * instance.transform[0]) +
            (instance.transform[4] * instance.transform[4]) +
            (instance.transform[8] * instance.transform[8]));
        object.transform.scale = scaleX;
        const auto savedOverride = m_bethesdaSession.tes3().referenceOverrides().find(id);
        if (savedOverride != m_bethesdaSession.tes3().referenceOverrides().end()) {
            if (savedOverride->second.deleted) continue;
            if (savedOverride->second.enabled.has_value()) {
                object.enabled = *savedOverride->second.enabled;
            }
            if (savedOverride->second.transform.has_value()) {
                object.transform = *savedOverride->second.transform;
            }
        }
        const bethesda::Tes3ActorDefinition* actorDefinition = nullptr;
        if (type == "NPC_" || type == "CREA") {
            object.kind = bethesda::RuntimeObjectKind::Actor;
            object.actorValues.emplace();
            const auto actor = m_bethesdaSession.tes3().content()->actors().find(object.base);
            if (actor != m_bethesdaSession.tes3().content()->actors().end()) {
                actorDefinition = &actor->second;
                object.actorValues->health = actorDefinition->health;
                object.actorValues->magicka = actorDefinition->magicka;
                object.actorValues->stamina = actorDefinition->fatigue;
                object.actorValues->maxHealth = actorDefinition->health;
                object.actorValues->maxMagicka = actorDefinition->magicka;
                object.actorValues->maxStamina = actorDefinition->fatigue;
                if (actorDefinition->faction.valid()) {
                    object.factions.push_back(actorDefinition->faction);
                }
                for (const auto& [item, count] : actorDefinition->inventory) {
                    object.inventory.push_back({item, count, false});
                }
            }
        } else if (type == "CONT") object.kind = bethesda::RuntimeObjectKind::Container;
        else if (type == "DOOR") object.kind = bethesda::RuntimeObjectKind::Door;
        else if (type == "ACTI") object.kind = bethesda::RuntimeObjectKind::Activator;
        else object.kind = bethesda::RuntimeObjectKind::Item;
        std::string bindError;
        if (!m_bethesdaSession.world().addInitialObject(std::move(object), bindError)) {
            VOX_LOGW("tes3") << "could not bind streamed reference "
                               << instance.sourceReferenceIdentity << ": " << bindError;
        } else if (actorDefinition != nullptr && actorDefinition->script.valid()) {
            std::string scriptError;
            const std::uint64_t threadId = m_bethesdaSession.tes3().scripts().start(
                actorDefinition->script.textId, id, scriptError);
            if (threadId == 0u) {
                VOX_LOGW("tes3") << "could not start local script "
                    << actorDefinition->script.toString() << " for "
                    << instance.sourceReferenceIdentity << ": " << scriptError;
            } else if (savedOverride != m_bethesdaSession.tes3().referenceOverrides().end()) {
                auto thread = m_bethesdaSession.tes3().scripts().threadsForRestore().find(threadId);
                if (thread != m_bethesdaSession.tes3().scripts().threadsForRestore().end()) {
                    for (const auto& [name, value] : savedOverride->second.locals) {
                        thread->second.locals.insert_or_assign(name, value);
                    }
                }
            }
        }
    }
    if (m_streamIsMorrowind &&
        m_bethesdaGameplayResidentCells.contains(cell)) {
        upsertMorrowindGameplayCell(cell);
        refreshBethesdaGameplayResidency();
    }
    const auto appendTriangle = [&](const float* vertices, std::uint32_t sourceReferenceFormId) {
        if (mesh.vertices.size() >
            static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()) - 3u) return;
        const std::uint32_t first = static_cast<std::uint32_t>(mesh.vertices.size());
        mesh.vertices.push_back({vertices[0], vertices[1], vertices[2]});
        mesh.vertices.push_back({vertices[3], vertices[4], vertices[5]});
        mesh.vertices.push_back({vertices[6], vertices[7], vertices[8]});
        mesh.indices.insert(mesh.indices.end(), {first, first + 1u, first + 2u});
        mesh.triangleSourceReferenceFormIds.push_back(sourceReferenceFormId);
    };
    for (const importer::ImportedSceneCollisionTriangle& triangle :
         scene.collisionTriangles) {
        appendTriangle(triangle.vertices, triangle.sourceReferenceFormId);
    }
    if (!scene.meshes.empty() && scene.meshes.front().name == "terrain") {
        const importer::ImportedSceneMesh& terrain = scene.meshes.front();
        for (std::size_t offset = 0u; offset + 2u < terrain.indices.size(); offset += 3u) {
            const std::uint32_t a = terrain.indices[offset];
            const std::uint32_t b = terrain.indices[offset + 1u];
            const std::uint32_t c = terrain.indices[offset + 2u];
            if (a >= terrain.vertices.size() || b >= terrain.vertices.size() ||
                c >= terrain.vertices.size()) continue;
            float vertices[9] = {};
            std::copy_n(terrain.vertices[a].position, 3u, vertices);
            std::copy_n(terrain.vertices[b].position, 3u, vertices + 3u);
            std::copy_n(terrain.vertices[c].position, 3u, vertices + 6u);
            appendTriangle(vertices, 0u);
        }
    }
    if (mesh.indices.empty()) {
        m_bethesdaCollisionByCell.erase(cell);
        return;
    }
    m_bethesdaCollisionByCell.insert_or_assign(cell, std::move(mesh));
    if (!m_bethesdaSessionConfigured) return;
    registerBethesdaCollisionCell(cell);
}

void BethesdaApp::removeBethesdaCollisionCell(const importer::CellCoord& cell) {
    m_bethesdaCollisionByCell.erase(cell);
    if (m_bethesdaSessionConfigured) {
        (void)m_bethesdaSession.physics().removeStreamedStaticCollision(
            physicsResidencyToken(cell));
        refreshBethesdaGameplayResidency();
    }
}

void BethesdaApp::registerBethesdaCollisionCell(const importer::CellCoord& cell) {
    if (!m_bethesdaSessionConfigured) return;
    const auto found = m_bethesdaCollisionByCell.find(cell);
    if (found == m_bethesdaCollisionByCell.end()) return;
    const BethesdaCollisionMesh& mesh = found->second;
    std::vector<odai::math::Vector3> filteredVertices;
    std::vector<std::uint32_t> filteredIndices;
    filteredVertices.reserve(mesh.vertices.size());
    filteredIndices.reserve(mesh.indices.size());
    const std::size_t triangleCount = mesh.indices.size() / 3u;
    for (std::size_t triangle = 0u; triangle < triangleCount; ++triangle) {
        const std::uint32_t source =
            triangle < mesh.triangleSourceReferenceFormIds.size()
                ? mesh.triangleSourceReferenceFormIds[triangle]
                : 0u;
        if (source != 0u && m_disabledBethesdaCollisionReferences.contains(source)) {
            continue;
        }
        const std::size_t indexOffset = triangle * 3u;
        const std::uint32_t a = mesh.indices[indexOffset];
        const std::uint32_t b = mesh.indices[indexOffset + 1u];
        const std::uint32_t c = mesh.indices[indexOffset + 2u];
        if (a >= mesh.vertices.size() || b >= mesh.vertices.size() ||
            c >= mesh.vertices.size()) {
            continue;
        }
        const std::uint32_t first = static_cast<std::uint32_t>(filteredVertices.size());
        filteredVertices.push_back(mesh.vertices[a]);
        filteredVertices.push_back(mesh.vertices[b]);
        filteredVertices.push_back(mesh.vertices[c]);
        filteredIndices.insert(
            filteredIndices.end(), {first, first + 1u, first + 2u});
    }
    const std::uint64_t token = physicsResidencyToken(cell);
    if (filteredIndices.empty()) {
        (void)m_bethesdaSession.physics().removeStreamedStaticCollision(token);
        return;
    }
    std::string error;
    if (!m_bethesdaSession.physics().addStreamedStaticCollision(
            token, filteredVertices, filteredIndices, error)) {
        VOX_LOGW("physics") << "could not restore collision cell " << cell.x << ","
                            << cell.z << ": " << error;
    } else {
        m_bethesdaCollisionBroadPhaseDirty = true;
        // The controller can be created before asynchronous exterior collision
        // reaches residency. If this cell introduced a floor through the
        // capsule, repair it immediately; waiting for CharacterVirtual::Update
        // leaves contradictory contacts that reject every movement direction.
        // City showcases have a stronger authored-arrival settlement below;
        // applying this generic visual-triangle recovery afterwards can lift
        // them off the Jolt floor selected by that settlement.
        if (!skyrimCityThirdPersonShowcase()) {
            (void)recoverBethesdaPlayerControllerFromIntersectingFloor();
        }
        if (m_skyrimCitySpawnSettlementPending && cell == m_skyrimCitySpawnCell) {
            (void)settleSkyrimCityShowcasePlayer();
        }
    }
}

void BethesdaApp::registerCachedBethesdaCollision() {
    if (!m_bethesdaSessionConfigured) return;
    std::vector<importer::CellCoord> cells;
    cells.reserve(m_bethesdaCollisionByCell.size());
    for (const auto& [cell, mesh] : m_bethesdaCollisionByCell) {
        (void)mesh;
        cells.push_back(cell);
    }
    std::sort(cells.begin(), cells.end(), [](const auto& left, const auto& right) {
        return std::tie(left.x, left.z) < std::tie(right.x, right.z);
    });
    for (const importer::CellCoord& cell : cells) {
        registerBethesdaCollisionCell(cell);
        if (m_streamIsMorrowind &&
            m_bethesdaGameplayResidentCells.contains(cell)) {
            upsertMorrowindGameplayCell(cell);
        }
    }
    refreshBethesdaGameplayResidency();
}

void BethesdaApp::upsertMorrowindGameplayCell(
    const importer::CellCoord& cell) {
    if (!m_bethesdaSessionConfigured || !m_streamIsMorrowind ||
        m_bethesdaSession.tes3().content() == nullptr) return;

    std::string fingerprint = m_loadOrderFingerprint;
    if (fingerprint.empty() && m_contentProfile.has_value()) {
        fingerprint = m_contentProfile->fingerprint;
    }
    if (fingerprint.empty()) {
        fingerprint = "unfingerprinted:" + toLowerAscii(m_streamPlugin);
    }

    bethesda::GameplayCellPayload payload;
    std::string error;
    std::filesystem::path sidecar;
    bool loaded = false;
    if (m_streamCacheEnabled && !m_streamCacheDirectory.empty()) {
        sidecar = std::filesystem::path(m_streamCacheDirectory) / "gameplay" /
            ("tes3_" + std::to_string(cell.x) + "_" +
             std::to_string(cell.z) + ".json");
        loaded = bethesda::loadGameplayCellPayload(
            sidecar, fingerprint, payload, error);
    }
    if (!loaded) {
        if (!bethesda::compileTes3GameplayExteriorCell(
                *m_bethesdaSession.tes3().content(), cell.x, cell.z,
                fingerprint, payload, error)) {
            VOX_LOGW("tes3") << "gameplay sidecar skipped for " << cell.x << ","
                               << cell.z << ": " << error;
            return;
        }
        if (!sidecar.empty()) {
            std::string cacheError;
            if (!bethesda::saveGameplayCellPayloadAtomic(
                    sidecar, payload, cacheError)) {
                VOX_LOGW("tes3") << "could not cache gameplay sidecar "
                                   << sidecar.string() << ": " << cacheError;
            }
        }
    }
    // Some valid anchors (beds, benches, counters, signs) are rendering-only
    // STAT references and are intentionally excluded from the heavyweight
    // activation catalog. Materialize only the sidecar-selected anchors as
    // lightweight runtime targets so navigation and offscreen reconciliation
    // address the real placed reference instead of inventing city markers.
    for (const bethesda::ActivityAnchor& anchor : payload.anchors) {
        if (m_bethesdaSession.world().find(anchor.object) != nullptr) continue;
        const auto definition = m_bethesdaSession.tes3().content()->references().find(
            anchor.object);
        if (definition == m_bethesdaSession.tes3().content()->references().end()) continue;
        bethesda::RuntimeObject target;
        target.id = anchor.object;
        target.base = definition->second.base;
        target.kind = bethesda::RuntimeObjectKind::Activator;
        target.transform.position = anchor.position;
        target.transform.scale = definition->second.scale.value_or(1.0f);
        target.originSpace = anchor.space;
        target.currentSpace = anchor.space;
        target.enabled = definition->second.enabled && !definition->second.deleted;
        target.persistent = true;
        target.ghost = true;
        target.interior = anchor.space.kind == bethesda::RuntimeSpaceKind::Interior;
        std::string targetError;
        if (!m_bethesdaSession.world().addInitialObject(
                std::move(target), targetError)) {
            VOX_LOGW("tes3") << "could not materialize gameplay anchor "
                               << anchor.object.toString() << ": " << targetError;
        }
    }
    if (!m_bethesdaSession.upsertGameplayCell(std::move(payload), error)) {
        VOX_LOGW("tes3") << "could not install gameplay sidecar for "
                           << cell.x << "," << cell.z << ": " << error;
    }
}

void BethesdaApp::refreshBethesdaGameplayResidency() {
    if (!m_bethesdaSessionConfigured) return;
    std::vector<bethesda::RuntimeSpaceState> spaces;
    spaces.reserve(m_bethesdaGameplayResidentCells.size());
    for (const bethesda::GameplayCellPayload& payload :
         m_bethesdaSession.livingWorld().cells()) {
        if (payload.space.kind != bethesda::RuntimeSpaceKind::Exterior) continue;
        const importer::CellCoord cell{payload.space.gridX, payload.space.gridZ};
        if (m_bethesdaGameplayResidentCells.contains(cell)) {
            spaces.push_back(payload.space);
        }
    }
    m_bethesdaSession.setGameplayResidentSpaces(std::move(spaces));
}

bool BethesdaApp::loadScenarioQuestDefinitions(const bethesda::ScenarioDefinition& scenario) {
    bethesda::SkyrimScenarioContentReport report;
    std::string error;
    if (!bethesda::loadSkyrimScenarioContent(
            scenario, m_streamLoadOrder, m_streamer->assets(),
            m_bethesdaSession, report, error)) {
        VOX_LOGE("scenario") << error;
        return false;
    }
    for (const std::string& diagnostic : report.diagnostics) {
        VOX_LOGW("scenario") << diagnostic;
    }
    for (const std::string& blocker : report.runtimeBlockers) {
        VOX_LOGW("scenario") << "runtime blocker: " << blocker;
    }
    for (const bethesda::ScenarioQuestLoadDetail& quest : report.quests) {
        VOX_LOGI("scenario") << "quest " << quest.editorId << " registered: "
                              << quest.stages << " stages, "
                              << quest.objectives << " objectives, "
                              << quest.aliases << " aliases, "
                              << quest.referencedRecords << " referenced records, "
                              << quest.unresolvedCalls << " unresolved call bindings";
    }
    return true;
}
bool BethesdaApp::initBethesdaSession() {
    m_bethesdaPlayerControllerRegistered = false;
    m_bethesdaControllerOwnsCamera = false;
    std::string fingerprint = m_loadOrderFingerprint;
    if (fingerprint.empty() && m_contentProfile.has_value()) fingerprint = m_contentProfile->fingerprint;
    if (fingerprint.empty()) fingerprint = "unfingerprinted:" + toLowerAscii(m_streamPlugin);
    std::string error;
    if (m_streamIsMorrowind) {
        useMorrowindUiPalette();
        if (!m_bethesdaSession.configure(
                bethesda::BethesdaSessionConfig{
                    importer::fnv::BethesdaGame::Morrowind, fingerprint, {},
                    m_captureSeed == 0u ? 1u : m_captureSeed}, error)) {
            VOX_LOGE("tes3") << "runtime session failed: " << error;
            return false;
        }
        auto content = std::make_shared<bethesda::Tes3ContentStore>();
        const std::string encoding = m_contentProfile.has_value()
            ? m_contentProfile->encoding : std::string("windows-1252");
        if (!content->load(m_streamLoadOrder, encoding, error) ||
            !m_bethesdaSession.configureTes3Content(content, error)) {
            VOX_LOGE("tes3") << "content runtime failed: " << error;
            return false;
        }
        bethesda::RuntimeObject player;
        player.id = m_bethesdaSession.playerObject();
        player.base = bethesda::makeTes3RecordKey("NPC_", "player");
        player.kind = bethesda::RuntimeObjectKind::Actor;
        player.persistent = true;
        player.actorValues.emplace();
        if (const bethesda::Tes3ActorDefinition* authoredPlayer =
                content->findActor("NPC_", "player"); authoredPlayer != nullptr) {
            player.actorValues->health = authoredPlayer->health;
            player.actorValues->magicka = authoredPlayer->magicka;
            player.actorValues->stamina = authoredPlayer->fatigue;
            player.actorValues->maxHealth = authoredPlayer->health;
            player.actorValues->maxMagicka = authoredPlayer->magicka;
            player.actorValues->maxStamina = authoredPlayer->fatigue;
            if (authoredPlayer->faction.valid()) player.factions.push_back(authoredPlayer->faction);
            for (const auto& [item, count] : authoredPlayer->inventory) {
                player.inventory.push_back({item, count, false});
                m_bethesdaSession.tes3().playerState().inventory[item] += count;
            }
            for (const auto& [name, value] : authoredPlayer->attributes) {
                m_bethesdaSession.tes3().playerState().numericFilters[name] = value;
            }
            for (const auto& [name, value] : authoredPlayer->skills) {
                m_bethesdaSession.tes3().playerState().numericFilters[name] = value;
            }
            m_bethesdaSession.tes3().playerState().numericFilters["level"] = authoredPlayer->level;
            m_bethesdaSession.tes3().playerState().numericFilters["reputation"] = 0.0;
        }
        player.transform.position = {m_cameraX, m_cameraY - kEyeHeightUnits, m_cameraZ};
        const float feet[3] = {m_cameraX, m_cameraY - kEyeHeightUnits, m_cameraZ};
        bethesda::RuntimeSpaceState playerSpace;
        if (runtimeSpaceForPosition(feet, playerSpace)) {
            player.originSpace = playerSpace;
            player.currentSpace = std::move(playerSpace);
        }
        if (!m_bethesdaSession.world().addInitialObject(std::move(player), error)) {
            VOX_LOGE("tes3") << "could not create player runtime object: " << error;
            return false;
        }
        m_bethesdaSessionConfigured = true;
        if (m_balmoraSkyrimPlayerShowcase && m_gameplaySavePath.empty()) {
            if (const char* xdg = std::getenv("XDG_DATA_HOME")) {
                m_gameplaySavePath = std::filesystem::path(xdg) /
                    "odai/saves/balmora-skyrim-player.odai.json";
            } else if (const char* home = std::getenv("HOME")) {
                m_gameplaySavePath = std::filesystem::path(home) /
                    ".local/share/odai/saves/balmora-skyrim-player.odai.json";
            }
        }
        const bethesda::Tes3ScriptCheckReport& report =
            m_bethesdaSession.tes3().scriptCheckReport();
        VOX_LOGI("tes3") << "runtime active: " << content->dialogues().size()
                          << " dialogues, " << content->scripts().size() << " scripts, "
                          << report.unsupportedCommands.size() << " unsupported commands";
        for (const std::string& diagnostic : report.diagnostics) {
            VOX_LOGW("tes3") << diagnostic;
        }
        if (!m_tes3StartQuest.empty()) {
            const bethesda::Tes3DialogueDefinition* quest =
                content->findDialogue(m_tes3StartQuest);
            std::string journalError;
            if (quest == nullptr || quest->type != bethesda::Tes3DialogueType::Journal) {
                VOX_LOGE("tes3") << "start quest not found: " << m_tes3StartQuest;
                return false;
            }
            if (!m_bethesdaSession.tes3().journal().addEntry(
                    *quest, m_tes3StartQuestIndex, 0u, journalError)) {
                VOX_LOGE("tes3") << "could not start quest " << m_tes3StartQuest
                                   << ": " << journalError;
                return false;
            }
            m_tes3PinnedQuest = quest->id;
            if (bethesda::normalizeTes3Symbol(quest->id) == "tr_m3_tt_bloodstone") {
                (void)m_bethesdaSession.tes3().addTopic("Bloodstone Pilgrimage");
            }
            syncTes3JournalPanel();
            m_toasts.push("Quest started", quest->id, "tes3-quest-start:" + quest->id);
            VOX_LOGI("tes3") << "quest started: " << quest->id
                              << " index=" << m_tes3StartQuestIndex;
            if (const char* demo = std::getenv("ODAI_FNV_UI_DEMO");
                demo != nullptr && std::string_view(demo) == "journal") {
                m_tes3JournalOpen = true;
                m_navDriving = true;
                setMouseCaptured(false);
            }
        }

        // TES3 NPCs/creatures are FRMR attachments inside CELL records rather
        // than the ACHR/ACRE records used by the later games. Publish actors at
        // those exact transforms so activation and dynamic dialogue address the
        // visible resident, not a second synthetic scan-order identity.
        m_actors.clear();
        constexpr float kTes3ActorProxyRadius = 7000.0f;
        const std::string currentInteriorKey =
            bethesda::normalizeTes3Symbol(m_currentInteriorEditorId);
        std::size_t scannedReferences = 0u;
        std::size_t actorReferences = 0u;
        std::size_t exteriorActorReferences = 0u;
        float nearestActorDistanceSquared = std::numeric_limits<float>::max();
        std::string nearestActorId;
        for (const auto& [referenceId, reference] : content->references()) {
            (void)referenceId;
            // Content construction owns tens of thousands of references and
            // runs before GameApp::run starts. Service the platform queue while
            // selecting presentation actors so a close request during startup
            // is not forced to wait for the whole catalog walk.
            if ((++scannedReferences & 1023u) == 0u) {
                glfwPollEvents();
                if (m_window != nullptr && glfwWindowShouldClose(m_window)) {
                    VOX_LOGI("tes3") << "actor population canceled by window close";
                    return false;
                }
            }
            if (!reference.enabled || reference.deleted || !reference.hasTransform) continue;
            const auto definition = content->actors().find(reference.base);
            if (definition == content->actors().end()) continue;
            ++actorReferences;
            const std::string& cellId = reference.cell.textId;
            const bool exteriorCell = !reference.interior;
            if (m_interiorStarted) {
                // Interior transforms are cell-local. Comparing them to the
                // player's local coordinates without first matching the CELL
                // made actors from every interior in Morrowind/TR look nearby:
                // Almas Thirr alone queued 12,605 activation proxies.
                if (currentInteriorKey.empty() ||
                    bethesda::normalizeTes3Symbol(cellId) != currentInteriorKey) {
                    continue;
                }
            } else if (!exteriorCell) {
                // Exterior transforms are world-space and may legitimately
                // cross a neighboring cell within the radius; interior-local
                // coordinates must never enter that distance test.
                continue;
            }
            ++exteriorActorReferences;
            const float engineX = reference.position[0];
            const float engineY = reference.position[2];
            const float engineZ = -reference.position[1];
            const float dx = engineX - m_cameraX;
            const float dz = engineZ - m_cameraZ;
            const float distanceSquared = (dx * dx) + (dz * dz);
            if (distanceSquared < nearestActorDistanceSquared) {
                nearestActorDistanceSquared = distanceSquared;
                nearestActorId = definition->second.id;
            }
            if (distanceSquared > kTes3ActorProxyRadius * kTes3ActorProxyRadius) continue;
            SkinnedActor actor;
            actor.name = definition->second.id;
            actor.fullName = definition->second.name;
            actor.runtimeObjectId = referenceId;
            actor.position[0] = engineX;
            actor.position[1] = engineY;
            actor.position[2] = engineZ;
            actor.yawRadians = -reference.rotationRadians[2];
            actor.standingHeightUnits = 120.0f;
            actor.humanoid = !definition->second.creature;
            actor.placed = true;
            actor.renderVisible = true;
            if (m_bethesdaSession.world().find(referenceId) == nullptr) {
                bethesda::RuntimeObject object;
                object.id = referenceId;
                object.base = definition->second.record;
                object.kind = bethesda::RuntimeObjectKind::Actor;
                object.transform.position = {engineX, engineY, engineZ};
                object.transform.rotationRadians[1] = actor.yawRadians;
                object.enabled = true;
                object.persistent = true;
                object.interior = reference.interior;
                object.actorValues.emplace();
                object.actorValues->health = definition->second.health;
                object.actorValues->magicka = definition->second.magicka;
                object.actorValues->stamina = definition->second.fatigue;
                object.actorValues->maxHealth = definition->second.health;
                object.actorValues->maxMagicka = definition->second.magicka;
                object.actorValues->maxStamina = definition->second.fatigue;
                for (const auto& [item, count] : definition->second.inventory) {
                    object.inventory.push_back({item, count, false});
                }
                const float enginePosition[3] = {engineX, engineY, engineZ};
                bethesda::RuntimeSpaceState space;
                if (runtimeSpaceForPosition(enginePosition, space)) {
                    object.originSpace = space;
                    object.currentSpace = std::move(space);
                }
                std::string runtimeError;
                if (!m_bethesdaSession.world().addInitialObject(
                        std::move(object), runtimeError)) {
                    VOX_LOGW("tes3") << "could not register placed actor "
                                      << actor.name << ": " << runtimeError;
                    actor.runtimeObjectId = {};
                }
            }
            m_actors.push_back(std::move(actor));
        }
        std::sort(m_actors.begin(), m_actors.end(), [&](const SkinnedActor& left,
                                                        const SkinnedActor& right) {
            const float ldx = left.position[0] - m_cameraX;
            const float ldz = left.position[2] - m_cameraZ;
            const float rdx = right.position[0] - m_cameraX;
            const float rdz = right.position[2] - m_cameraZ;
            return (ldx * ldx) + (ldz * ldz) < (rdx * rdx) + (rdz * rdz);
        });
        std::uint32_t nextSlot = kFirstCrowdSkinnedInstance;
        std::size_t builtActors = 0u;
        std::unordered_map<std::string, anim::AnimationClip> idleBySkeleton;
        std::unordered_set<std::string> unavailableIdleSkeletons;
        struct CachedWalk {
            anim::AnimationClip clip;
            float speedUnitsPerSecond = 0.0f;
        };
        std::unordered_map<std::string, CachedWalk> walkBySkeleton;
        std::unordered_set<std::string> unavailableWalkSkeletons;
        std::size_t actorBuildCount = 0u;
        for (SkinnedActor& actor : m_actors) {
            if ((actorBuildCount++ & 3u) == 0u) {
                glfwPollEvents();
                if (m_window != nullptr && glfwWindowShouldClose(m_window)) {
                    VOX_LOGI("tes3") << "actor assembly canceled by window close";
                    return false;
                }
            }
            const std::uint32_t actorSlotLimit = thirdPersonPlayerShowcase()
                ? kPlayerAvatarSkinnedInstance : render::kMaxSkinnedInstances;
            if (nextSlot >= actorSlotLimit) break;
            const bethesda::Tes3ActorDefinition* definition =
                content->findActor("NPC_", actor.name);
            if (definition == nullptr) definition = content->findActor("CREA", actor.name);
            if (definition == nullptr) continue;
            std::string skeleton;
            std::vector<std::string> parts;
            std::vector<std::string> rigidAttachmentBones;
            std::string why;
            if (!tes3ActorGeometry(
                    *content, *definition, skeleton, parts,
                    rigidAttachmentBones, why)) {
                VOX_LOGW("tes3") << "actor " << actor.name
                                  << " remains activation-only: " << why;
                continue;
            }
            const char* requestedTrace = std::getenv("ODAI_TES3_ACTOR_TRACE");
            const bool traceActor = requestedTrace != nullptr &&
                toLowerAscii(requestedTrace) == toLowerAscii(actor.name);
            if (traceActor) {
                const bethesda::Tes3NamedRecord* source = content->findRecord(
                    definition->creature ? "CREA" : "NPC_", definition->id);
                VOX_LOGI("tes3") << "actor trace " << actor.name
                                  << ": position=(" << actor.position[0] << ", "
                                  << actor.position[1] << ", " << actor.position[2]
                                  << ") skeleton=" << skeleton << " explicitModel="
                                  << (source != nullptr
                                          ? tes3SubrecordText(
                                                *source, "MODL", content->encoding())
                                          : std::string());
                for (std::size_t part = 0u; part < parts.size(); ++part) {
                    VOX_LOGI("tes3") << "actor trace " << actor.name << " part["
                                      << part << "]=" << parts[part];
                }
                if (const char* requestedPart =
                        std::getenv("ODAI_TES3_ACTOR_PART_FILTER")) {
                    const std::string wantedPart = toLowerAscii(requestedPart);
                    // Rebuild both arrays together: their indices are the
                    // authored BODY-slot association.
                    std::vector<std::string> filteredParts;
                    std::vector<std::string> filteredAttachments;
                    for (std::size_t part = 0u; part < parts.size(); ++part) {
                        if (toLowerAscii(parts[part]).find(wantedPart) ==
                            std::string::npos) {
                            continue;
                        }
                        filteredParts.push_back(parts[part]);
                        filteredAttachments.push_back(rigidAttachmentBones[part]);
                    }
                    parts = std::move(filteredParts);
                    rigidAttachmentBones = std::move(filteredAttachments);
                    VOX_LOGI("tes3") << "actor trace " << actor.name
                                      << ": retained " << parts.size()
                                      << " matching part(s) for diagnostic capture";
                }
            }
            if (!buildSkinnedActor(
                    m_streamer->assets(), skeleton, parts, actor.character,
                    actor.textures, actor.draws, why, &rigidAttachmentBones)) {
                VOX_LOGW("tes3") << "actor " << actor.name
                                  << " remains activation-only: " << why;
                continue;
            }
            if (traceActor && std::getenv("ODAI_TES3_ACTOR_VERBOSE") != nullptr) {
                std::vector<odai::math::Matrix4> bindPose;
                importer::fnv::computeFalloutBindPose(actor.character, bindPose);
                for (std::size_t partIndex = 0u;
                     partIndex < actor.character.parts.size(); ++partIndex) {
                    const importer::fnv::FalloutCharacterPart& part =
                        actor.character.parts[partIndex];
                    float minimum[3] = {
                        std::numeric_limits<float>::max(),
                        std::numeric_limits<float>::max(),
                        std::numeric_limits<float>::max()};
                    float maximum[3] = {
                        std::numeric_limits<float>::lowest(),
                        std::numeric_limits<float>::lowest(),
                        std::numeric_limits<float>::lowest()};
                    for (std::uint32_t offset = 0u; offset < part.indexCount; ++offset) {
                        const std::size_t indexOffset =
                            static_cast<std::size_t>(part.firstIndex) + offset;
                        if (indexOffset >= actor.character.indices.size()) break;
                        const std::uint32_t vertexIndex = actor.character.indices[indexOffset];
                        if (vertexIndex >= actor.character.vertices.size()) continue;
                        const auto& vertex = actor.character.vertices[vertexIndex];
                        odai::math::Vector4 posed{0.0f, 0.0f, 0.0f, 0.0f};
                        float totalWeight = 0.0f;
                        for (int influence = 0; influence < 4; ++influence) {
                            const float weight = vertex.boneWeights[influence];
                            const std::size_t bone = vertex.boneIndices[influence];
                            if (weight <= 0.0f || bone >= bindPose.size()) continue;
                            const odai::math::Vector4 transformed = bindPose[bone] *
                                odai::math::Vector4{
                                    vertex.position[0], vertex.position[1],
                                    vertex.position[2], 1.0f};
                            posed.x += transformed.x * weight;
                            posed.y += transformed.y * weight;
                            posed.z += transformed.z * weight;
                            totalWeight += weight;
                        }
                        if (totalWeight <= 1e-6f) {
                            posed = {vertex.position[0], vertex.position[1],
                                     vertex.position[2], 1.0f};
                        }
                        minimum[0] = std::min(minimum[0], posed.x);
                        minimum[1] = std::min(minimum[1], posed.y);
                        minimum[2] = std::min(minimum[2], posed.z);
                        maximum[0] = std::max(maximum[0], posed.x);
                        maximum[1] = std::max(maximum[1], posed.y);
                        maximum[2] = std::max(maximum[2], posed.z);
                    }
                    VOX_LOGI("tes3")
                        << "actor trace " << actor.name << " boundPart[" << partIndex
                        << "] source=" << part.sourcePath << " shape=" << part.name
                        << " bounds=(" << minimum[0] << ".." << maximum[0] << ", "
                        << minimum[1] << ".." << maximum[1] << ", "
                        << minimum[2] << ".." << maximum[2] << ") texture="
                        << part.diffuseTexturePath;
                }
            }
            actor.instanceSlot = nextSlot++;
            actor.standingHeightUnits = actorStandingHeight(actor.character);
            findActorHeadAnchor(
                actor.character, actor.headAnchorBone, actor.headAnchorLocal,
                actor.headHeightUnits);
            actor.sampler.bindSkeleton(
                actor.character.skeleton, actor.character.inverseBindMatrices);
            std::string idleWhy;
            const std::string idleKey = toLowerAscii(skeleton);
            if (const auto idle = idleBySkeleton.find(idleKey);
                idle != idleBySkeleton.end()) {
                actor.idleClip = idle->second;
            } else if (!unavailableIdleSkeletons.contains(idleKey)) {
                if (loadActorIdleClip(
                        m_streamer->assets(), skeleton, actor.character.skeleton,
                        builtActors, actor.idleClip, idleWhy)) {
                    idleBySkeleton.emplace(idleKey, actor.idleClip);
                } else {
                    unavailableIdleSkeletons.insert(idleKey);
                }
            }
            if (const auto walk = walkBySkeleton.find(idleKey);
                walk != walkBySkeleton.end()) {
                actor.walkClip = walk->second.clip;
                actor.walkSpeedUnitsPerSecond = walk->second.speedUnitsPerSecond;
            } else if (!unavailableWalkSkeletons.contains(idleKey)) {
                CachedWalk walk;
                std::string walkWhy;
                if (loadActorWalkClip(
                        m_streamer->assets(), skeleton, actor.character.skeleton,
                        false, walk.clip, walk.speedUnitsPerSecond, walkWhy)) {
                    actor.walkClip = walk.clip;
                    actor.walkSpeedUnitsPerSecond = walk.speedUnitsPerSecond;
                    walkBySkeleton.emplace(idleKey, std::move(walk));
                } else {
                    unavailableWalkSkeletons.insert(idleKey);
                }
            }
            actor.wanders = !actor.walkClip.tracks.empty() &&
                actor.walkSpeedUnitsPerSecond > 1.0f;
            for (int axis = 0; axis < 3; ++axis) {
                actor.wanderOrigin[axis] = actor.position[axis];
                actor.wanderTarget[axis] = actor.position[axis];
            }
            actor.wanderRng = 0x9e3779b9u ^
                (static_cast<std::uint32_t>(builtActors + 1u) * 2654435761u);
            actor.wanderPauseSeconds =
                static_cast<float>(builtActors % 7u) * 0.9f;
            if (traceActor && std::getenv("ODAI_TES3_ACTOR_VERBOSE") != nullptr &&
                !actor.idleClip.tracks.empty()) {
                for (const float sampleTime : {0.0f, 0.65f, 1.3f, 1.95f}) {
                    std::vector<odai::math::Matrix4> sampledPose;
                    actor.sampler.sample(
                        actor.character.skeleton, actor.idleClip, sampleTime, sampledPose);
                    float minimum[3] = {
                        std::numeric_limits<float>::max(),
                        std::numeric_limits<float>::max(),
                        std::numeric_limits<float>::max()};
                    float maximum[3] = {
                        std::numeric_limits<float>::lowest(),
                        std::numeric_limits<float>::lowest(),
                        std::numeric_limits<float>::lowest()};
                    for (const auto& vertex : actor.character.vertices) {
                        odai::math::Vector4 posed{0.0f, 0.0f, 0.0f, 0.0f};
                        float totalWeight = 0.0f;
                        for (int influence = 0; influence < 4; ++influence) {
                            const float weight = vertex.boneWeights[influence];
                            const std::size_t bone = vertex.boneIndices[influence];
                            if (weight <= 0.0f || bone >= sampledPose.size()) continue;
                            const odai::math::Vector4 transformed = sampledPose[bone] *
                                odai::math::Vector4{
                                    vertex.position[0], vertex.position[1],
                                    vertex.position[2], 1.0f};
                            posed.x += transformed.x * weight;
                            posed.y += transformed.y * weight;
                            posed.z += transformed.z * weight;
                            totalWeight += weight;
                        }
                        if (totalWeight <= 1e-6f) continue;
                        minimum[0] = std::min(minimum[0], posed.x);
                        minimum[1] = std::min(minimum[1], posed.y);
                        minimum[2] = std::min(minimum[2], posed.z);
                        maximum[0] = std::max(maximum[0], posed.x);
                        maximum[1] = std::max(maximum[1], posed.y);
                        maximum[2] = std::max(maximum[2], posed.z);
                    }
                    VOX_LOGI("tes3") << "actor trace " << actor.name
                                      << " idle t=" << sampleTime << " bounds=("
                                      << minimum[0] << ".." << maximum[0] << ", "
                                      << minimum[1] << ".." << maximum[1] << ", "
                                      << minimum[2] << ".." << maximum[2] << ")";
                }
                for (const char* boneName : {
                         "Bip01 Pelvis", "Bip01 Spine1", "Bip01 Head",
                         "Bip01 L UpperArm", "Bip01 L Forearm", "Bip01 L Hand"}) {
                    const int boneIndex = actor.character.skeleton.findBone(boneName);
                    if (boneIndex < 0) continue;
                    const auto& bone = actor.character.skeleton.bones[
                        static_cast<std::size_t>(boneIndex)];
                    const auto track = std::find_if(
                        actor.idleClip.tracks.begin(), actor.idleClip.tracks.end(),
                        [boneIndex](const anim::BoneTrack& candidate) {
                            return candidate.boneIndex == boneIndex;
                        });
                    VOX_LOGI("tes3") << "actor trace " << actor.name << " bone="
                                      << boneName << " bindT=(" << bone.localTranslation.x
                                      << ", " << bone.localTranslation.y << ", "
                                      << bone.localTranslation.z << ") bindQ=("
                                      << bone.localRotation.x << ", " << bone.localRotation.y
                                      << ", " << bone.localRotation.z << ", "
                                      << bone.localRotation.w << ") keys="
                                      << (track != actor.idleClip.tracks.end()
                                              ? track->rotationKeys.size()
                                              : 0u);
                    if (track != actor.idleClip.tracks.end() &&
                        !track->rotationKeys.empty()) {
                        const auto& key = track->rotationKeys.front();
                        VOX_LOGI("tes3") << "actor trace " << actor.name << " bone="
                                          << boneName << " firstQ=(" << key.value.x << ", "
                                          << key.value.y << ", " << key.value.z << ", "
                                          << key.value.w << ")";
                    }
                }
            }
            ++builtActors;
            VOX_LOGI("tes3") << "  actor " << actor.name << " slot="
                              << actor.instanceSlot << " verts="
                              << actor.character.vertices.size() << " parts="
                              << actor.character.parts.size();
        }
        if (!m_balmoraSkyrimPlayerShowcase &&
            !addSkyrimGuardsToBalmora(nextSlot)) return false;
        arrangeActorParadeIfRequested();
        queueActorUploads();
        VOX_LOGI("tes3") << "bound " << m_actors.size()
                          << " nearby FRMR actor(s) for activation/dialogue; built "
                          << builtActors << " visible actor(s); actor refs="
                          << actorReferences << " exterior=" << exteriorActorReferences
                          << " nearest=" << (nearestActorId.empty() ? "<none>" : nearestActorId)
                          << " distance="
                          << (nearestActorId.empty()
                                  ? -1.0f
                                  : std::sqrt(nearestActorDistanceSquared));
        registerCachedBethesdaCollision();
        if (!registerBethesdaPlayerController()) return false;
        if (thirdPersonPlayerShowcase() && !initSkyrimPlayerAvatar()) return false;
        if (thirdPersonPlayerShowcase() && !m_gameplayLoadPath.empty() &&
            !loadGameplayState()) return false;
        return true;
    }
    const bethesda::ScenarioDefinition* scenario = bethesda::findScenario(m_scenarioId);
    if (scenario == nullptr) {
        VOX_LOGE("scenario") << "unknown scenario '" << m_scenarioId << "'";
        return false;
    }
    if (!m_streamIsSkyrim) {
        VOX_LOGE("scenario") << scenario->id << " requires Skyrim Special Edition content";
        return false;
    }
    useSkyrimUiPalette();
    if (!m_bethesdaSession.configure(
            bethesda::BethesdaSessionConfig{scenario->game, fingerprint, scenario->id,
                                            m_captureSeed == 0u ? 1u : m_captureSeed}, error)) {
        VOX_LOGE("scenario") << "runtime session failed: " << error;
        return false;
    }
    m_bethesdaSessionConfigured = true;
    if (!loadScenarioQuestDefinitions(*scenario)) {
        m_bethesdaSessionConfigured = false;
        return false;
    }
    registerCachedBethesdaCollision();
    if (m_gameplaySavePath.empty()) {
        if (const char* xdg = std::getenv("XDG_DATA_HOME")) {
            m_gameplaySavePath = std::filesystem::path(xdg) / "odai/saves/" /
                (scenario->id + ".odai.json");
        } else if (const char* home = std::getenv("HOME")) {
            m_gameplaySavePath = std::filesystem::path(home) / ".local/share/odai/saves/" /
                (scenario->id + ".odai.json");
        }
    }
    if (!m_gameplayLoadPath.empty()) {
        if (!loadGameplayState()) return false;
    } else {
        bethesda::RuntimeObject player;
        player.id = m_bethesdaSession.playerObject();
        player.base = bethesda::makeRecordKey(scenario->basePlugin, 0x7u);
        player.kind = bethesda::RuntimeObjectKind::Actor;
        player.persistent = true;
        player.actorValues.emplace();
        player.transform.position = {m_cameraX, m_cameraY - kEyeHeightUnits, m_cameraZ};
        const float playerFeet[3] = {
            m_cameraX, m_cameraY - kEyeHeightUnits, m_cameraZ};
        bethesda::RuntimeSpaceState playerSpace;
        if (runtimeSpaceForPosition(playerFeet, playerSpace)) {
            player.originSpace = playerSpace;
            player.currentSpace = std::move(playerSpace);
            player.interior = m_interiorStarted;
        }
        if (!m_bethesdaSession.world().addInitialObject(std::move(player), error)) {
            VOX_LOGE("scenario") << "could not create scenario player: " << error;
            return false;
        }
        syncBethesdaActors(true, true);
    }
    if (!configureGoldenClawPuzzleForCurrentSpace(error)) {
        VOX_LOGE("scenario") << "Golden Claw compatibility error: " << error;
        return false;
    }
    if (!registerBethesdaPlayerController()) return false;
    if (thirdPersonPlayerShowcase() && !initSkyrimPlayerAvatar()) return false;
    VOX_LOGI("scenario") << scenario->id << " active; F5 saves, F9 loads";
    return true;
}

bool BethesdaApp::addSkyrimGuardsToBalmora(std::uint32_t firstInstanceSlot) {
    if (!m_streamIsMorrowind || m_interiorStarted) return true;
    // Keep the deliberately cross-game population scoped to Balmora rather
    // than silently adding Skyrim guards to every Vvardenfell/TR exterior.
    constexpr float kBalmoraCentreX = -20000.0f;
    constexpr float kBalmoraCentreZ = 14000.0f;
    constexpr float kBalmoraShowcaseRadius = 7000.0f;
    const float balmoraDx = m_cameraX - kBalmoraCentreX;
    const float balmoraDz = m_cameraZ - kBalmoraCentreZ;
    if ((balmoraDx * balmoraDx) + (balmoraDz * balmoraDz) >
        kBalmoraShowcaseRadius * kBalmoraShowcaseRadius) {
        return true;
    }

    std::vector<std::filesystem::path> candidates;
    if (const char* configured = std::getenv("ODAI_SKYRIM_DATA")) {
        candidates.emplace_back(configured);
    }
    if (const char* home = std::getenv("HOME")) {
        const std::filesystem::path homePath(home);
        candidates.push_back(homePath /
            ".local/share/Steam/steamapps/common/Skyrim Special Edition/Data");
        candidates.push_back(homePath /
            ".steam/steam/steamapps/common/Skyrim Special Edition/Data");
    }
    const auto available = std::find_if(candidates.begin(), candidates.end(),
        [](const std::filesystem::path& candidate) {
            return std::filesystem::is_regular_file(candidate / "Skyrim.esm");
        });
    if (available == candidates.end()) {
        VOX_LOGW("tes3")
            << "Balmora Skyrim guards skipped: set ODAI_SKYRIM_DATA to a Skyrim SE Data directory";
        return true;
    }

    // These are the authored approaches to Balmora's two low canal bridges,
    // derived from the bridge/canal kit references in exterior cell (-3,-2).
    // Fixed city anchors keep a camera launched inside a wall or over the water
    // from dragging the whole showcase there. Each point sits just beyond a
    // bridge end; navmesh projection below performs the final height settle.
    constexpr std::array<std::array<float, 3>, 4> kGuardAnchors{{
        {{-19920.0f, 300.0f, 12960.0f}},
        {{-18672.0f, 300.0f, 12960.0f}},
        {{-19920.0f, 300.0f, 14896.0f}},
        {{-18672.0f, 300.0f, 14896.0f}},
    }};
    constexpr std::size_t kGuardCount = kGuardAnchors.size();
    const float centre[3] = {
        kBalmoraCentreX, kGuardAnchors.front()[1], kBalmoraCentreZ};
    std::vector<SkinnedActor> guards;
    std::string detail;
    if (!loadSkyrimGuardShowcase(
            *available, centre, firstInstanceSlot, kGuardCount, guards, detail)) {
        VOX_LOGW("tes3") << "Balmora Skyrim guards skipped: " << detail;
        return true;
    }

    std::size_t added = 0u;
    for (std::size_t guardIndex = 0u; guardIndex < guards.size(); ++guardIndex) {
        SkinnedActor& guard = guards[guardIndex];
        const auto& anchor = kGuardAnchors[guardIndex % kGuardAnchors.size()];
        std::copy(anchor.begin(), anchor.end(), guard.position);
        // Face inward across the bridge until the navmesh planner supplies the
        // first walking target. actorYawForDirection uses asset-forward -Z.
        guard.yawRadians = actorYawForDirection(
            guardIndex % 2u == 0u ? 1.0f : -1.0f, 0.0f);
        odai::math::Vector3 projected;
        if (m_actorNavigation.projectPoint(
                guard.position[0], guard.position[1], guard.position[2],
                320.0f, 500.0f, projected)) {
            guard.position[0] = projected.x;
            guard.position[1] = projected.y;
            guard.position[2] = projected.z;
            guard.projectedToNavigation = true;
        } else {
            float ground = guard.position[1];
            if (m_collision.groundHeight(
                    guard.position[0], guard.position[2], guard.position[1], ground)) {
                guard.position[1] = ground;
            }
        }
        std::copy_n(guard.position, 3u, guard.wanderOrigin);
        std::copy_n(guard.position, 3u, guard.wanderTarget);

        bethesda::RuntimeObject* existing =
            m_bethesdaSession.world().find(guard.runtimeObjectId);
        if (existing == nullptr) {
            bethesda::RuntimeObject object;
            object.id = guard.runtimeObjectId;
            object.base = bethesda::makeRecordKey(
                "Skyrim.esm", guard.baseFormId & 0x00ffffffu);
            object.kind = bethesda::RuntimeObjectKind::Actor;
            object.transform.position = {
                guard.position[0], guard.position[1], guard.position[2]};
            object.transform.rotationRadians[1] = guard.yawRadians;
            object.enabled = true;
            object.persistent = true;
            object.actorValues.emplace();
            object.aiState = runtimeAiStateFor(guard, m_streamLoadOrder);
            const float enginePosition[3] = {
                guard.position[0], guard.position[1], guard.position[2]};
            bethesda::RuntimeSpaceState space;
            if (runtimeSpaceForPosition(enginePosition, space)) {
                object.originSpace = space;
                object.currentSpace = std::move(space);
            }
            std::string error;
            if (!m_bethesdaSession.world().addInitialObject(
                    std::move(object), error)) {
                VOX_LOGW("tes3") << "could not register " << guard.displayName()
                                  << " in Balmora: " << error;
                continue;
            }
        } else {
            guard.position[0] = static_cast<float>(existing->transform.position[0]);
            guard.position[1] = static_cast<float>(existing->transform.position[1]);
            guard.position[2] = static_cast<float>(existing->transform.position[2]);
            std::copy_n(guard.position, 3u, guard.wanderOrigin);
            std::copy_n(guard.position, 3u, guard.wanderTarget);
        }
        m_actors.push_back(std::move(guard));
        ++added;
    }
    VOX_LOGI("tes3") << "Balmora cross-game guard showcase: " << added
                      << " registered; " << detail;
    return true;
}

bool BethesdaApp::registerBethesdaPlayerController() {
    if (!m_bethesdaSessionConfigured) return false;
    const bethesda::ObjectId playerId = m_bethesdaSession.playerObject();
    const bethesda::RuntimeObject* player = m_bethesdaSession.world().find(playerId);
    if (player == nullptr) {
        VOX_LOGE("physics") << "player is missing from the runtime world";
        return false;
    }
    bethesda::PhysicsCharacterConfig config;
    config.position = {
        static_cast<float>(player->transform.position[0]),
        static_cast<float>(player->transform.position[1]),
        static_cast<float>(player->transform.position[2])};
    std::string error;
    if (!m_bethesdaSession.registerActorController(playerId, config, error)) {
        VOX_LOGE("physics") << "could not register player controller: " << error;
        return false;
    }
    m_bethesdaPlayerControllerRegistered = true;
    m_bethesdaControllerOwnsCamera = m_walkMode;
    if (m_skyrimCitySpawnSettlementPending) {
        (void)settleSkyrimCityShowcasePlayer();
    }
    if (!skyrimCityThirdPersonShowcase()) {
        (void)recoverBethesdaPlayerControllerFromIntersectingFloor();
    }
    if (m_bethesdaControllerOwnsCamera) pullBethesdaPlayerControllerState();
    return true;
}

bool BethesdaApp::settleSkyrimCityShowcasePlayer() {
    if (!m_skyrimCitySpawnSettlementPending ||
        !m_bethesdaPlayerControllerRegistered || !m_bethesdaSessionConfigured) {
        return !m_skyrimCitySpawnSettlementPending;
    }
    const bethesda::ObjectId playerId = m_bethesdaSession.playerObject();
    const auto physical = m_bethesdaSession.physics().characterState(playerId);
    if (!physical.has_value()) return false;

    odai::math::Vector3 feet = m_skyrimCityAuthoredSpawnFeet;
    const bool gateCollisionResident =
        m_bethesdaCollisionByCell.contains(m_skyrimCitySpawnCell);
    // The gate cell can arrive before the rest of the startup ring. Wait for
    // the complete collision/navigation batch so a projection cannot choose a
    // point whose adjacent wall has not been registered yet.
    if (!gateCollisionResident || !m_streamer || !m_streamer->isStreamingIdle()) {
        return false;
    }
    const bethesda::PhysicsCharacterConfig capsule;
    const float capsuleRadius = std::max(
        capsule.boundsHalfExtents.x, capsule.boundsHalfExtents.z);
    // Imported Havok vertices and NAVM/visual collision are quantized through
    // different retail encodings. Leave a two-unit numerical margin around the
    // controller's authored step while still rejecting the gate threshold's
    // much larger overlapping-floor discrepancies.
    constexpr float kGroundAgreementSlack = 2.0f;
    constexpr float kMinimumWalkableGroundNormalY = 0.64f;
    const auto tryCandidate = [&](const odai::math::Vector3& requested,
                                  float navigationRadius,
                                  odai::math::Vector3& outFeet) {
        odai::math::Vector3 candidate;
        if (!m_actorNavigation.projectPoint(
                requested.x, requested.y, requested.z,
                navigationRadius, 512.0f, candidate)) return false;

        const auto groundCandidate = [&](odai::math::Vector3& point) {
            // NAVM supplies a legal horizontal location, not an exact rendered
            // floor height. Require the imported visual collision and Jolt's
            // supporting static body to agree within one authored step; this
            // prevents choosing a gate ledge whose physics floor is underneath
            // the visible street.
            float visualGround = point.y;
            if (!m_collision.groundHeight(
                    point.x, point.z, point.y, visualGround)) return false;
            const float castOriginY = visualGround + capsule.stepHeight +
                kGroundAgreementSlack + 4.0f;
            const auto physicsGround = m_bethesdaSession.physics().castDown(
                {point.x, castOriginY, point.z}, 1024.0f);
            if (!physicsGround.has_value() ||
                std::fabs(physicsGround->normal.y) < kMinimumWalkableGroundNormalY ||
                std::fabs(physicsGround->position.y - visualGround) >
                    capsule.stepHeight + kGroundAgreementSlack) {
                return false;
            }
            point.y = physicsGround->position.y + 0.1f;
            return true;
        };
        if (!groundCandidate(candidate)) return false;

        const float beforeX = candidate.x;
        const float beforeZ = candidate.z;
        m_collision.resolveHorizontalFor(
            candidate.x, candidate.z, candidate.y,
            candidate.y + capsule.boundsHalfExtents.y * 2.0f,
            capsuleRadius, capsule.stepHeight);
        if ((candidate.x != beforeX || candidate.z != beforeZ) &&
            !groundCandidate(candidate)) return false;

        // Settlement is also the first-frame camera contract. Reject a floor
        // beside a gate wall when the same collision-aware boom sweep used by
        // the visible loop cannot retain most of the showcase framing.
        constexpr float kPivotHeight = 105.0f;
        constexpr float kCameraRadius = 12.0f;
        const float yaw = m_yawDegrees * (kPi / 180.0f);
        const float pitch = m_pitchDegrees * (kPi / 180.0f);
        const float horizontal = std::cos(pitch);
        const odai::math::Vector3 forward{
            std::cos(yaw) * horizontal, std::sin(pitch),
            std::sin(yaw) * horizontal};
        const odai::math::Vector3 pivot = candidate +
            odai::math::Vector3{0.0f, kPivotHeight, 0.0f};
        const odai::math::Vector3 requestedCamera =
            pivot - forward * m_cameraBoomRequested;
        if (const auto hit = m_bethesdaSession.physics().castSphere(
                pivot, requestedCamera, kCameraRadius, playerId)) {
            constexpr float kMinimumBoomFraction = 0.8f;
            if (hit->distance - 2.0f <
                m_cameraBoomRequested * kMinimumBoomFraction) return false;
        }
        outFeet = candidate;
        if (candidate.x != beforeX || candidate.z != beforeZ) {
            VOX_LOGI("showcase")
                << "pushed authored city spawn clear of wall by ("
                << (candidate.x - beforeX) << ", "
                << (candidate.z - beforeZ) << ")";
        }
        return true;
    };

    bool grounded = tryCandidate(feet, 256.0f, feet);
    if (!grounded) {
        // A paired door can place the authored point on the gate threshold,
        // where decorative ledges overlap the playable street. Search only the
        // immediate entrance apron, prioritizing the XTEL facing direction,
        // and keep the first nav/visual/Jolt-consistent capsule pose.
        static constexpr std::array<float, 8> kDirectionOffsets{
            0.0f, 45.0f, -45.0f, 90.0f, -90.0f, 135.0f, -135.0f, 180.0f};
        // The first 256 units are the door/arch collision band. A point can
        // have a valid floor there while the full capsule is still wedged
        // against the gate jamb, so begin on the open inner apron.
        for (float distance = 320.0f; distance <= 1024.0f && !grounded;
             distance += 64.0f) {
            for (const float offsetDegrees : kDirectionOffsets) {
                const float angle = (m_yawDegrees + offsetDegrees) * (kPi / 180.0f);
                odai::math::Vector3 requested = m_skyrimCityAuthoredSpawnFeet;
                requested.x += std::cos(angle) * distance;
                requested.z += std::sin(angle) * distance;
                if (tryCandidate(requested, 48.0f, feet)) {
                    grounded = true;
                    VOX_LOGI("showcase")
                        << "moved city arrival " << distance
                        << " units onto a playable gate-apron surface";
                    break;
                }
            }
        }
    }

    // Until the asynchronous gate cell arrives, reset the controller to the
    // resolved gate-apron feet pose every frame. It can no longer accumulate
    // several seconds of gravity in an empty Jolt world and end up below the
    // city.
    bethesda::PhysicsCharacterSnapshot settled;
    settled.object = playerId;
    settled.position = feet;
    settled.rotation = physical->rotation;
    settled.velocity = {};
    settled.groundNormal = {0.0f, 1.0f, 0.0f};
    settled.grounded = grounded;
    std::string error;
    if (!m_bethesdaSession.physics().restoreCharacter(settled, error)) {
        VOX_LOGW("showcase") << "could not hold Skyrim city gate spawn: " << error;
        return false;
    }
    if (!grounded) return false;

    m_skyrimCitySpawnSettlementPending = false;
    m_skyrimCityAuthoredSpawnFeet = feet;
    VOX_LOGI("showcase") << "Skyrim city player settled on gate navigation at feet ("
                          << feet.x << ", " << feet.y << ", " << feet.z << ")";
    syncBethesdaPlayerState(true);
    reconstructPlayerCamera(1.0f / 60.0f, true);
    return true;
}

bool BethesdaApp::recoverBethesdaPlayerControllerFromIntersectingFloor() {
    if (!m_walkMode || !m_bethesdaPlayerControllerRegistered ||
        !m_bethesdaSessionConfigured) return false;
    const bethesda::ObjectId playerId = m_bethesdaSession.playerObject();
    const auto physical = m_bethesdaSession.physics().characterState(playerId);
    if (!physical.has_value()) return false;
    float recoveredFeetY = 0.0f;
    const float headY = physical->position.y + kEyeHeightUnits;
    if (!m_collision.recoverFeetAboveIntersectingFloor(
            physical->position.x, physical->position.z, physical->position.y, headY,
            bethesda::PhysicsCharacterConfig{}.stepHeight, recoveredFeetY)) return false;

    bethesda::PhysicsCharacterSnapshot recovered;
    recovered.object = playerId;
    recovered.position = physical->position;
    recovered.position.y = recoveredFeetY + 0.1f;
    recovered.rotation = physical->rotation;
    recovered.groundNormal = {0.0f, 1.0f, 0.0f};
    recovered.grounded = true;
    recovered.supportingObject = physical->supportingObject;
    std::string error;
    if (!m_bethesdaSession.physics().restoreCharacter(recovered, error)) {
        VOX_LOGW("physics") << "could not recover intersecting player capsule: " << error;
        return false;
    }
    VOX_LOGW("physics") << "recovered player capsule from intersecting floor: feet y="
                         << physical->position.y << " -> " << recovered.position.y;
    m_bethesdaControllerOwnsCamera = true;
    pullBethesdaPlayerControllerState();
    syncBethesdaPlayerState(true);
    return true;
}

void BethesdaApp::pullBethesdaPlayerControllerState() {
    if (!m_bethesdaPlayerControllerRegistered) return;
    const bethesda::ObjectId playerId = m_bethesdaSession.playerObject();
    const auto physical = m_bethesdaSession.physics().characterState(playerId);
    if (!physical.has_value()) return;
    if (thirdPersonPlayerShowcase()) {
        reconstructPlayerCamera(1.0f / 60.0f);
    } else {
        m_cameraX = physical->position.x;
        m_cameraY = physical->position.y + kEyeHeightUnits;
        m_cameraZ = physical->position.z;
    }
}

odai::math::Vector3 BethesdaApp::bethesdaPlayerFeetPosition() const {
    if (m_bethesdaPlayerControllerRegistered && m_bethesdaSessionConfigured) {
        const auto physical = m_bethesdaSession.physics().characterState(
            m_bethesdaSession.playerObject());
        if (physical.has_value()) return physical->position;
    }
    return {m_cameraX, m_cameraY - kEyeHeightUnits, m_cameraZ};
}

odai::math::Vector3 BethesdaApp::bethesdaPlayerEyePosition() const {
    return bethesdaPlayerFeetPosition() + odai::math::Vector3{0.0f, kEyeHeightUnits, 0.0f};
}

void BethesdaApp::reconstructPlayerCamera(float deltaSeconds, bool snapInward) {
    if (!thirdPersonPlayerShowcase() || !m_bethesdaPlayerControllerRegistered) return;
    const odai::math::Vector3 feet = bethesdaPlayerFeetPosition();
    if (!m_thirdPersonView) {
        const odai::math::Vector3 eye = feet +
            odai::math::Vector3{0.0f, kEyeHeightUnits, 0.0f};
        m_cameraX = eye.x;
        m_cameraY = eye.y;
        m_cameraZ = eye.z;
        return;
    }
    constexpr float kPivotHeight = 105.0f;
    constexpr float kCameraRadius = 12.0f;
    const odai::math::Vector3 pivot = feet +
        odai::math::Vector3{0.0f, kPivotHeight, 0.0f};
    const float yaw = m_yawDegrees * (kPi / 180.0f);
    const float pitch = m_pitchDegrees * (kPi / 180.0f);
    const float horizontal = std::cos(pitch);
    const odai::math::Vector3 forward{
        std::cos(yaw) * horizontal, std::sin(pitch),
        std::sin(yaw) * horizontal};
    const odai::math::Vector3 requested = pivot - forward * m_cameraBoomRequested;
    float unobstructedDistance = m_cameraBoomRequested;
    if (const auto hit = m_bethesdaSession.physics().castSphere(
            pivot, requested, kCameraRadius, m_bethesdaSession.playerObject())) {
        unobstructedDistance = std::clamp(hit->distance - 2.0f, 0.0f,
            m_cameraBoomRequested);
    }
    if (snapInward || unobstructedDistance < m_cameraBoomActual) {
        m_cameraBoomActual = unobstructedDistance;
    } else {
        constexpr float kRecoveryTauSeconds = 0.20f;
        const float blend = 1.0f - std::exp(
            -std::max(deltaSeconds, 0.0f) / kRecoveryTauSeconds);
        m_cameraBoomActual += (unobstructedDistance - m_cameraBoomActual) * blend;
    }
    const odai::math::Vector3 camera = pivot - forward * m_cameraBoomActual;
    m_cameraX = camera.x;
    m_cameraY = camera.y;
    m_cameraZ = camera.z;
}

bool BethesdaApp::initSkyrimPlayerAvatar() {
    SkinnedActor avatar;
    std::string detail;
    if (!loadSkyrimPlayerAvatar(
            std::filesystem::path(m_skyrimAvatarDataDirectory),
            m_skyrimPlayerOutfitEditorId, kPlayerAvatarSkinnedInstance,
            avatar, detail)) {
        VOX_LOGE("showcase") << detail;
        return false;
    }
    const odai::math::Vector3 feet = bethesdaPlayerFeetPosition();
    avatar.position[0] = feet.x;
    avatar.position[1] = feet.y;
    avatar.position[2] = feet.z;
    avatar.yawRadians = m_playerYawRadians;
    avatar.renderVisible = m_thirdPersonView;
    m_skyrimPlayerEquippedSignature = 1469598103934665603ull;
    for (const std::uint32_t item : avatar.inventoryFormIds) {
        m_skyrimPlayerEquippedSignature ^= item;
        m_skyrimPlayerEquippedSignature *= 1099511628211ull;
    }
    auto animationView = std::make_shared<anim::AnimationView>();
    animationView->skeleton =
        std::make_shared<const anim::Skeleton>(avatar.character.skeleton);
    animationView->inverseBindMatrices = avatar.character.inverseBindMatrices;
    animationView->clips = {avatar.idleClip, avatar.walkClip};
    animationView->clips.insert(animationView->clips.end(),
        avatar.authoredLocomotionClips.begin(),
        avatar.authoredLocomotionClips.end());
    animationView->stateClips = {
        {"idle", avatar.idleClip.name},
        {"locomotion", avatar.walkClip.name},
        {"sprint", "Skyrim male sprint forward"},
        {"jump", "Skyrim jump"},
        {"fall", "Skyrim fall"},
        {"landing", "Skyrim landing"}};
    animationView->providerId = "skyrim-avatar:" +
        m_skyrimPlayerOutfitEditorId;
    animationView->supportedBehaviorGraph =
        !avatar.idleClip.name.starts_with("procedural") &&
        !avatar.walkClip.name.starts_with("procedural");
    bethesda::PhysicsCharacterConfig alreadyRegistered;
    alreadyRegistered.position = feet;
    std::string animationError;
    if (!m_bethesdaSession.registerActorAnimation(
            m_bethesdaSession.playerObject(), std::move(animationView), nullptr,
            alreadyRegistered, animationError)) {
        VOX_LOGE("showcase")
            << "could not register fixed-tick Skyrim player graph: "
            << animationError;
        return false;
    }
    m_skyrimPlayerAvatar = std::move(avatar);
    m_skyrimPlayerAvatarUploadPending = true;
    m_cameraBoomActual = m_cameraBoomRequested;
    reconstructPlayerCamera(1.0f / 60.0f, true);
    VOX_LOGI("showcase") << "Skyrim player avatar ready: " << detail;
    return true;
}

void BethesdaApp::updateSkyrimPlayerAvatar(float deltaSeconds) {
    if (!m_skyrimPlayerAvatar.has_value()) return;
    SkinnedActor& avatar = *m_skyrimPlayerAvatar;
    const odai::math::Vector3 feet = bethesdaPlayerFeetPosition();
    avatar.position[0] = feet.x;
    avatar.position[1] = feet.y;
    avatar.position[2] = feet.z;
    avatar.yawRadians = m_playerYawRadians;
    avatar.renderVisible = m_thirdPersonView;
    if (const auto physical = m_bethesdaSession.physics().characterState(
            m_bethesdaSession.playerObject())) {
        const float horizontalSpeed = std::sqrt(
            physical->velocity.x * physical->velocity.x +
            physical->velocity.z * physical->velocity.z);
        avatar.walking = horizontalSpeed > 1.0f;
    }
    const bool freezeAtBindPose = std::getenv("ODAI_FNV_NOANIM") != nullptr ||
        std::getenv("ODAI_FNV_VICTOR_NOANIM") != nullptr;
    const anim::AnimationStepOutput graphPose = freezeAtBindPose
        ? anim::AnimationStepOutput{}
        : m_bethesdaSession.interpolatedActorAnimationOutput(
              m_bethesdaSession.playerObject(), m_sessionInterpolationAlpha);
    if (freezeAtBindPose) {
        const odai::math::Matrix4 world =
            odai::math::Matrix4::translation(feet) *
            odai::math::Matrix4::rotationY(avatar.yawRadians);
        avatar.poseScratch.assign(avatar.character.skeleton.bones.size(), world);
    } else if (!graphPose.pose.empty()) {
        avatar.poseScratch = graphPose.pose;
        const odai::math::Matrix4 world =
            odai::math::Matrix4::translation(feet) *
            odai::math::Matrix4::rotationY(avatar.yawRadians);
        for (odai::math::Matrix4& bone : avatar.poseScratch) bone = world * bone;
    } else {
        updateActorPoses(std::span<SkinnedActor>(&avatar, 1u), deltaSeconds);
    }
    if (avatar.uploaded) {
        m_renderer.setSkinnedActorVisible(avatar.instanceSlot, avatar.renderVisible);
        if (avatar.renderVisible) {
            render::ImportedSkinnedActorFrameData pose{};
            pose.boneMatrices = avatar.poseScratch;
            m_renderer.setSkinnedActorPose(avatar.instanceSlot, pose);
        }
    }
}

void BethesdaApp::relocateBethesdaPlayerControllerToCamera() {
    if (!m_bethesdaPlayerControllerRegistered) return;
    const bethesda::ObjectId playerId = m_bethesdaSession.playerObject();
    const auto physical = m_bethesdaSession.physics().characterState(playerId);
    if (!physical.has_value()) return;
    bethesda::PhysicsCharacterSnapshot saved;
    saved.object = playerId;
    saved.position = {m_cameraX, m_cameraY - kEyeHeightUnits, m_cameraZ};
    saved.rotation = physical->rotation;
    saved.groundNormal = {0.0f, 1.0f, 0.0f};
    std::string error;
    if (!m_bethesdaSession.physics().restoreCharacter(saved, error)) {
        VOX_LOGW("physics") << "could not relocate scenario player controller: " << error;
        return;
    }
    m_bethesdaControllerOwnsCamera = true;
    syncBethesdaPlayerState(true);
    if (thirdPersonPlayerShowcase()) {
        const float viewYaw = m_yawDegrees * (kPi / 180.0f);
        m_playerPreviousYawRadians = m_playerYawRadians;
        m_playerYawRadians = actorYawForDirection(
            std::cos(viewYaw), std::sin(viewYaw));
        reconstructPlayerCamera(1.0f / 60.0f, true);
    }
}

std::optional<bethesda::ObjectId> BethesdaApp::runtimeObjectIdForActor(
    const SkinnedActor& actor) const {
    if (actor.runtimeObjectId.valid()) return actor.runtimeObjectId;
    if (actor.referenceFormId == 0u) return std::nullopt;
    bethesda::RecordKey reference;
    std::string error;
    if (!bethesda::stableRecordKey(
            m_streamLoadOrder, actor.referenceFormId, reference, error)) {
        return std::nullopt;
    }
    return bethesda::ObjectId::persistent(std::move(reference));
}

void BethesdaApp::unregisterBethesdaActorControllers() {
    if (!m_bethesdaSessionConfigured) return;
    for (SkinnedActor& actor : m_actors) {
        actor.runtimeRequestedVelocity = {};
        const std::optional<bethesda::ObjectId> resolved = runtimeObjectIdForActor(actor);
        if (!resolved.has_value()) continue;
        const bethesda::ObjectId& id = *resolved;
        if (m_bethesdaSession.physics().hasCharacter(id)) {
            (void)m_bethesdaSession.unregisterActorController(id);
        }
        actor.runtimeControllerOwned = false;
        actor.runtimeControllerNeedsRelocation = false;
        actor.runtimeControllerBlocked = false;
    }
}

void BethesdaApp::pullBethesdaActorControllerStates() {
    if (!m_bethesdaSessionConfigured) return;
    for (SkinnedActor& actor : m_actors) {
        const std::optional<bethesda::ObjectId> resolved = runtimeObjectIdForActor(actor);
        if (!resolved.has_value()) continue;
        const bethesda::ObjectId& id = *resolved;
        const auto physical = m_bethesdaSession.physics().characterState(id);
        if (!physical.has_value()) {
            actor.runtimeControllerOwned = false;
            continue;
        }
        actor.runtimeControllerOwned = true;
        actor.position[0] = physical->position.x;
        actor.position[1] = physical->position.y;
        actor.position[2] = physical->position.z;
        actor.runtimeControllerBlocked = physical->blocked;
    }
}

void BethesdaApp::stepBethesdaActorControllers(float fixedDeltaSeconds) {
    if (!m_bethesdaSessionConfigured) return;
    pullBethesdaActorControllerStates();
    for (SkinnedActor& actor : m_actors) {
        const std::optional<bethesda::ObjectId> id = runtimeObjectIdForActor(actor);
        if (!id.has_value()) continue;
        const bethesda::RuntimeObject* runtime = m_bethesdaSession.world().find(*id);
        actor.runtimeDead = runtime != nullptr && runtime->actorValues.has_value() &&
            runtime->actorValues->dead;
        if (actor.runtimeDead) {
            actor.walking = false;
            actor.runtimeRequestedVelocity = {};
        }
    }
    const ActorNavigationWorld* navigation =
        (m_streamer && m_streamer->isStreamingIdle()) ? &m_actorNavigation : nullptr;
    updateActorWandering(
        m_actors, fixedDeltaSeconds, navigation,
        [this](float x, float z, float referenceY, float& outHeight) {
            return m_streamer
                ? m_collision.groundHeight(x, z, referenceY, outHeight)
                : false;
        },
        [this](float& x, float& z, float feetY, float headY, float radius) {
            if (m_streamer) {
                m_collision.resolveHorizontalFor(
                    x, z, feetY, headY, radius, m_collision.tuning().stepHeight);
            }
        },
        m_talkingActor);
    submitBethesdaActorControllerIntents();
    syncBethesdaActors(false, false);
}

void BethesdaApp::submitBethesdaActorControllerIntents() {
    if (!m_bethesdaSessionConfigured) return;
    // Jolt's CharacterVsCharacterCollisionSimple deliberately checks every
    // registered virtual character against every other one. Keep that exact
    // collision set local to the player while distant city residents continue
    // on the navmesh. Separate enter/exit radii prevent controller churn at
    // the boundary and keep hundreds of visible actors practical.
    constexpr float kPhysicsEnterRadius = 2400.0f;
    constexpr float kPhysicsExitRadius = 2800.0f;
    odai::math::Vector3 playerPosition{
        m_cameraX, m_cameraY - kEyeHeightUnits, m_cameraZ};
    if (const auto player = m_bethesdaSession.physics().characterState(
            m_bethesdaSession.playerObject())) {
        playerPosition = player->position;
    }
    for (SkinnedActor& actor : m_actors) {
        const std::optional<bethesda::ObjectId> resolved = runtimeObjectIdForActor(actor);
        if (!resolved.has_value()) continue;
        const bethesda::ObjectId& id = *resolved;
        if (m_bethesdaSession.world().find(id) == nullptr) continue;

        const float dx = actor.position[0] - playerPosition.x;
        const float dz = actor.position[2] - playerPosition.z;
        const float distanceSquared = (dx * dx) + (dz * dz);
        const bool hasController = m_bethesdaSession.physics().hasCharacter(id);
        const float activeRadius = hasController
            ? kPhysicsExitRadius : kPhysicsEnterRadius;
        if (distanceSquared > activeRadius * activeRadius) {
            if (hasController) {
                // This is proximity residency, not save-state residency. The
                // runtime object's synchronized transform is authoritative;
                // re-entry builds a fresh controller at the navmesh position.
                (void)m_bethesdaSession.physics().removeCharacter(id);
            }
            actor.runtimeControllerOwned = false;
            actor.runtimeControllerBlocked = false;
            continue;
        }

        if (!hasController) {
            const float height = std::clamp(
                actor.standingHeightUnits > 1.0f ? actor.standingHeightUnits : 128.0f,
                48.0f, 256.0f);
            const float radius = std::clamp(height * 0.28f, 12.0f, 48.0f);
            bethesda::PhysicsCharacterConfig config;
            config.position = {actor.position[0], actor.position[1], actor.position[2]};
            config.boundsHalfExtents = {radius, height * 0.5f, radius};
            std::string error;
            if (!m_bethesdaSession.registerActorController(id, config, error)) {
                VOX_LOGW("physics") << "could not register actor controller "
                                     << actor.name << ": " << error;
                actor.runtimeControllerOwned = false;
                continue;
            }
            actor.runtimeControllerOwned = true;
            if (const auto restored = m_bethesdaSession.physics().characterState(id)) {
                actor.position[0] = restored->position.x;
                actor.position[1] = restored->position.y;
                actor.position[2] = restored->position.z;
                actor.runtimeControllerBlocked = restored->blocked;
            }
        }

        if (actor.runtimeControllerNeedsRelocation) {
            if (const auto physical = m_bethesdaSession.physics().characterState(id)) {
                bethesda::PhysicsCharacterSnapshot relocated;
                relocated.object = id;
                relocated.position = {
                    actor.position[0], actor.position[1], actor.position[2]};
                relocated.rotation = physical->rotation;
                relocated.groundNormal = {0.0f, 1.0f, 0.0f};
                std::string error;
                if (!m_bethesdaSession.physics().restoreCharacter(relocated, error)) {
                    VOX_LOGW("physics") << "could not project actor controller "
                                         << actor.name << ": " << error;
                }
            }
            actor.runtimeControllerNeedsRelocation = false;
        }

        bethesda::PhysicsCharacterInput input;
        input.desiredVelocity = actor.runtimeRequestedVelocity;
        (void)m_bethesdaSession.setActorControllerInput(id, input);
    }
}

void BethesdaApp::syncBethesdaPlayerState(bool applyNow) {
    if (!m_bethesdaSessionConfigured) return;
    const bethesda::ObjectId playerId = m_bethesdaSession.playerObject();
    const bethesda::RuntimeObject* player = m_bethesdaSession.world().find(playerId);
    if (player == nullptr) return;
    bethesda::WorldCommand command;
    command.type = bethesda::WorldCommandType::SetTransform;
    command.target = playerId;
    float feetPosition[3] = {};
    if (const auto physical = m_bethesdaSession.physics().characterState(playerId)) {
        command.transform.position = {
            physical->position.x, physical->position.y, physical->position.z};
        feetPosition[0] = physical->position.x;
        feetPosition[1] = physical->position.y;
        feetPosition[2] = physical->position.z;
    } else {
        command.transform.position = {m_cameraX, m_cameraY - kEyeHeightUnits, m_cameraZ};
        feetPosition[0] = m_cameraX;
        feetPosition[1] = m_cameraY - kEyeHeightUnits;
        feetPosition[2] = m_cameraZ;
    }
    command.transform.rotationRadians = thirdPersonPlayerShowcase()
        ? std::array<float, 3>{0.0f, m_playerYawRadians, 0.0f}
        : std::array<float, 3>{
            m_pitchDegrees * (kPi / 180.0f),
            m_yawDegrees * (kPi / 180.0f), 0.0f};
    (void)m_bethesdaSession.world().queue(std::move(command));
    bethesda::RuntimeSpaceState space;
    if (runtimeSpaceForPosition(feetPosition, space) &&
        player->currentSpace != space) {
        bethesda::WorldCommand spaceCommand;
        spaceCommand.type = bethesda::WorldCommandType::SetCurrentSpace;
        spaceCommand.target = playerId;
        spaceCommand.currentSpace = std::move(space);
        (void)m_bethesdaSession.world().queue(std::move(spaceCommand));
    }
    if (applyNow) (void)m_bethesdaSession.world().applyQueuedCommands();
}

void BethesdaApp::syncBethesdaActors(bool addMissing, bool applyNow) {
    if (!m_bethesdaSessionConfigured || m_streamLoadOrder.empty()) return;
    m_bethesdaSession.clearLoadedLocations();
    std::vector<std::uint32_t> loadedLocationFormIds =
        m_streamer->residentLocationFormIds();
    if (m_interiorStarted) {
        const std::uint32_t interiorLocation =
            m_streamer->locationFormIdForInterior(m_currentInteriorEditorId);
        if (interiorLocation != 0u) loadedLocationFormIds.push_back(interiorLocation);
    }
    for (const std::uint32_t locationFormId : loadedLocationFormIds) {
        bethesda::RecordKey location;
        std::string locationError;
        if (bethesda::stableRecordKey(
                m_streamLoadOrder, locationFormId, location, locationError)) {
            m_bethesdaSession.setLocationLoaded(location, true);
        }
    }
    for (SkinnedActor& actor : m_actors) {
        if (actor.referenceFormId == 0u || actor.baseFormId == 0u) continue;
        bethesda::RecordKey reference;
        bethesda::RecordKey base;
        std::string error;
        if (!bethesda::stableRecordKey(m_streamLoadOrder, actor.referenceFormId, reference, error) ||
            !bethesda::stableRecordKey(m_streamLoadOrder, actor.baseFormId, base, error)) {
            VOX_LOGW("runtime") << "actor identity unavailable for " << actor.name << ": " << error;
            continue;
        }
        const bethesda::ObjectId id = bethesda::ObjectId::persistent(std::move(reference));
        bethesda::RuntimeTransform transform;
        transform.position = {actor.position[0], actor.position[1], actor.position[2]};
        transform.rotationRadians[1] = actor.yawRadians;
        const float enginePosition[3] = {
            actor.position[0], actor.position[1], actor.position[2]};
        float falloutPosition[3] = {};
        importer::fnv::CellStreamer::engineToFallout(enginePosition, falloutPosition);
        const std::uint32_t locationFormId = m_interiorStarted
            ? m_streamer->locationFormIdForInterior(m_currentInteriorEditorId)
            : m_streamer->locationFormIdAtFallout(falloutPosition[0], falloutPosition[1]);
        bethesda::RecordKey location;
        if (locationFormId != 0u &&
            !bethesda::stableRecordKey(m_streamLoadOrder, locationFormId, location, error)) {
            VOX_LOGW("runtime") << "location identity unavailable for " << actor.name
                                 << ": " << error;
            location = {};
        }
        bethesda::RuntimeSpaceState originSpace;
        const bool hasOriginSpace =
            runtimeOriginSpaceForReference(actor.referenceFormId, originSpace);
        bethesda::RuntimeSpaceState currentSpace;
        const bool hasCurrentSpace =
            runtimeSpaceForPosition(enginePosition, currentSpace);
        const bethesda::RuntimeObject* existing = m_bethesdaSession.world().find(id);
        if (existing == nullptr) {
            if (!addMissing) continue;
            bethesda::RuntimeObject object;
            object.id = id;
            object.base = base;
            object.kind = bethesda::RuntimeObjectKind::Actor;
            object.transform = transform;
            if (hasOriginSpace) object.originSpace = originSpace;
            if (hasCurrentSpace) object.currentSpace = currentSpace;
            object.enabled = actor.renderVisible;
            object.persistent = true;
            object.interior = m_interiorStarted;
            object.inDialogueWithPlayer = actor.talking;
            object.location = location;
            for (const std::uint32_t itemFormId : actor.inventoryFormIds) {
                bethesda::RecordKey item;
                if (!bethesda::stableRecordKey(
                        m_streamLoadOrder, itemFormId, item, error)) {
                    VOX_LOGW("runtime") << "inventory identity unavailable for "
                                         << actor.name << ": " << error;
                    continue;
                }
                const auto existingItem = std::find_if(
                    object.inventory.begin(), object.inventory.end(),
                    [&](const bethesda::InventoryEntry& entry) {
                        return entry.item == item;
                    });
                if (existingItem == object.inventory.end()) {
                    object.inventory.push_back({std::move(item), 1, false});
                } else {
                    ++existingItem->count;
                }
            }
            std::sort(object.inventory.begin(), object.inventory.end(),
                [](const bethesda::InventoryEntry& left,
                   const bethesda::InventoryEntry& right) {
                    return left.item < right.item;
                });
            for (const std::uint32_t referenceTypeFormId : actor.referenceTypeFormIds) {
                bethesda::RecordKey referenceType;
                if (!bethesda::stableRecordKey(
                        m_streamLoadOrder, referenceTypeFormId, referenceType, error)) {
                    VOX_LOGW("runtime") << "reference type identity unavailable for "
                                         << actor.name << ": " << error;
                    continue;
                }
                object.referenceTypes.push_back(std::move(referenceType));
            }
            std::sort(object.referenceTypes.begin(), object.referenceTypes.end());
            object.referenceTypes.erase(std::unique(
                object.referenceTypes.begin(), object.referenceTypes.end()),
                object.referenceTypes.end());
            object.aiState = runtimeAiStateFor(actor, m_streamLoadOrder);
            object.actorValues.emplace();
            if (!m_bethesdaSession.world().addInitialObject(std::move(object), error)) {
                VOX_LOGW("runtime") << "could not register actor " << actor.name << ": " << error;
            } else {
                const std::size_t materialized =
                    m_bethesdaSession.bindQuestInventoryForActor(id, base, error);
                if (!error.empty()) {
                    VOX_LOGW("runtime") << "could not bind quest inventory for "
                                         << actor.name << ": " << error;
                } else if (materialized != 0u) {
                    VOX_LOGI("runtime") << "materialized " << materialized
                                         << " quest-created item(s) on " << actor.name;
                }
            }
            continue;
        }
        (void)m_bethesdaSession.bindQuestInventoryForActor(id, base, error);
        actor.renderVisible = existing->enabled;
        if (actor.scriptedMoveArrived && existing->navigationRequest.has_value() &&
            existing->navigationRequest->revision == actor.scriptedMoveRevision) {
            bethesda::WorldCommand arrived;
            arrived.type = bethesda::WorldCommandType::SetNavigationStatus;
            arrived.target = id;
            arrived.navigationRevision = actor.scriptedMoveRevision;
            arrived.navigationStatus = bethesda::NavigationRequestStatus::Arrived;
            (void)m_bethesdaSession.world().queue(std::move(arrived));
            actor.scriptedMoveArrived = false;
        }
        if (existing->navigationRequest.has_value() &&
            existing->navigationRequest->status != bethesda::NavigationRequestStatus::Arrived &&
            existing->navigationRequest->status != bethesda::NavigationRequestStatus::Failed &&
            actor.scriptedMoveRevision != existing->navigationRequest->revision &&
            existing->navigationRequest->destination.kind ==
                bethesda::ObjectIdKind::PersistentReference) {
            std::uint32_t destinationFormId = 0u;
            float destination[3] = {};
            std::string navigationError;
            if (bethesda::resolvedFormId(
                    m_streamLoadOrder, existing->navigationRequest->destination.reference,
                    destinationFormId, navigationError) &&
                m_streamer->referencePositionEngineSpace(
                    destinationFormId, destination, navigationError)) {
                std::vector<ActorNavigationStep> path;
                if (m_actorNavigation.buildPath(
                        odai::math::Vector3{
                            actor.position[0], actor.position[1], actor.position[2]},
                        odai::math::Vector3{destination[0], destination[1], destination[2]}, path) &&
                    !path.empty()) {
                    actor.wanderPath = std::move(path);
                    actor.wanderPathIndex = 1u;
                    actor.wanderTarget[0] = actor.wanderPath.front().position.x;
                    actor.wanderTarget[1] = actor.wanderPath.front().position.y;
                    actor.wanderTarget[2] = actor.wanderPath.front().position.z;
                    actor.wanderPauseSeconds = 0.0f;
                    actor.wanders = true;
                    actor.scriptedMoveActive = true;
                    actor.scriptedMoveArrived = false;
                    actor.scriptedMoveRevision = existing->navigationRequest->revision;
                    bethesda::WorldCommand moving;
                    moving.type = bethesda::WorldCommandType::SetNavigationStatus;
                    moving.target = id;
                    moving.navigationRevision = actor.scriptedMoveRevision;
                    moving.navigationStatus = bethesda::NavigationRequestStatus::Moving;
                    (void)m_bethesdaSession.world().queue(std::move(moving));
                }
            }
        }
        if (existing->transform != transform) {
            bethesda::WorldCommand move;
            move.type = bethesda::WorldCommandType::SetTransform;
            move.target = id;
            move.transform = transform;
            (void)m_bethesdaSession.world().queue(std::move(move));
        }
        if (existing->originSpace.kind == bethesda::RuntimeSpaceKind::Unknown &&
            hasOriginSpace) {
            bethesda::WorldCommand origin;
            origin.type = bethesda::WorldCommandType::SetOriginSpace;
            origin.target = id;
            origin.originSpace = originSpace;
            (void)m_bethesdaSession.world().queue(std::move(origin));
        }
        if (hasCurrentSpace && existing->currentSpace != currentSpace) {
            bethesda::WorldCommand space;
            space.type = bethesda::WorldCommandType::SetCurrentSpace;
            space.target = id;
            space.currentSpace = currentSpace;
            (void)m_bethesdaSession.world().queue(std::move(space));
        }
        if (existing->interior != m_interiorStarted ||
            existing->inDialogueWithPlayer != actor.talking ||
            existing->location != location) {
            bethesda::WorldCommand context;
            context.type = bethesda::WorldCommandType::SetActorContext;
            context.target = id;
            context.interior = m_interiorStarted;
            context.inDialogueWithPlayer = actor.talking;
            context.location = location;
            (void)m_bethesdaSession.world().queue(std::move(context));
        }
        const bethesda::RuntimeAiState aiState =
            runtimeAiStateFor(actor, m_streamLoadOrder);
        if (!existing->aiState.has_value() || *existing->aiState != aiState) {
            bethesda::WorldCommand ai;
            ai.type = bethesda::WorldCommandType::SetAiState;
            ai.target = id;
            ai.aiState = aiState;
            (void)m_bethesdaSession.world().queue(std::move(ai));
        }
    }
    if (applyNow) {
        const bethesda::CommandApplyResult applied =
            m_bethesdaSession.world().applyQueuedCommands();
        for (const std::string& diagnostic : applied.diagnostics) {
            VOX_LOGW("runtime") << diagnostic;
        }
        (void)m_renderer.applyRuntimeRenderDeltas(applied.renderDeltas);
    }
}

void BethesdaApp::restoreBethesdaActorsFromSession() {
    if (!m_bethesdaSessionConfigured || m_streamLoadOrder.empty()) return;
    for (SkinnedActor& actor : m_actors) {
        const std::optional<bethesda::ObjectId> id = runtimeObjectIdForActor(actor);
        if (!id.has_value()) continue;
        const bethesda::RuntimeObject* object = m_bethesdaSession.world().find(*id);
        if (object == nullptr) continue;
        actor.position[0] = static_cast<float>(object->transform.position[0]);
        actor.position[1] = static_cast<float>(object->transform.position[1]);
        actor.position[2] = static_cast<float>(object->transform.position[2]);
        actor.yawRadians = object->transform.rotationRadians[1];
        actor.renderVisible = object->enabled;
        if (object->aiState.has_value()) {
            restoreRuntimeAiState(*object->aiState, m_streamLoadOrder, actor);
        }
    }
    syncBethesdaActors(true, true);
    submitBethesdaActorControllerIntents();
    pullBethesdaActorControllerStates();
}

bool BethesdaApp::saveGameplayState() {
    if (!m_bethesdaSessionConfigured || m_gameplaySavePath.empty()) return false;
    syncBethesdaPlayerState(true);
    std::string error;
    if (!bethesda::saveOdaiGameAtomic(m_gameplaySavePath, m_bethesdaSession, error)) {
        VOX_LOGE("save") << error;
        return false;
    }
    VOX_LOGI("save") << "saved ODAI gameplay state to " << m_gameplaySavePath.string();
    m_toasts.push("Game saved", m_gameplaySavePath.filename().string(), "gameplay-save");
    return true;
}

bool BethesdaApp::loadGameplayState() {
    const std::filesystem::path path = m_gameplayLoadPath.empty()
        ? m_gameplaySavePath : m_gameplayLoadPath;
    if (!m_bethesdaSessionConfigured || path.empty()) return false;
    const bool controllerWasRegistered = m_bethesdaPlayerControllerRegistered;
    bethesda::SaveLoadReport report;
    std::string error;
    bethesda::SaveLoadOptions options;
    options.recordAvailable = [this](const bethesda::RecordKey& key) {
        if (m_streamLoadOrder.empty()) return false;
        std::uint32_t formId = 0u;
        std::string resolutionError;
        return bethesda::resolvedFormId(m_streamLoadOrder, key, formId, resolutionError);
    };
    if (!bethesda::loadOdaiGame(path, m_bethesdaSession, options, report, error)) {
        VOX_LOGE("save") << error;
        return false;
    }
    const bethesda::ObjectId playerId = m_bethesdaSession.playerObject();
    if (playerId.valid()) {
        const bethesda::RuntimeObject* player = m_bethesdaSession.world().find(playerId);
        if (player != nullptr) {
            m_cameraX = static_cast<float>(player->transform.position[0]);
            m_cameraY = static_cast<float>(player->transform.position[1]);
            m_cameraZ = static_cast<float>(player->transform.position[2]);
            if (thirdPersonPlayerShowcase()) {
                m_playerPreviousYawRadians = m_playerYawRadians;
                m_playerYawRadians = player->transform.rotationRadians[1];
            } else {
                m_pitchDegrees = player->transform.rotationRadians[0] * (180.0f / kPi);
                m_yawDegrees = player->transform.rotationRadians[1] * (180.0f / kPi);
            }
        }
        const std::vector<bethesda::PhysicsCharacterSnapshot> physicalActors =
            m_bethesdaSession.physicsSnapshots();
        const auto savedPhysical = std::find_if(
            physicalActors.begin(), physicalActors.end(),
            [&](const bethesda::PhysicsCharacterSnapshot& value) {
                return value.object == playerId;
            });
        if (savedPhysical != physicalActors.end()) {
            m_cameraX = savedPhysical->position.x;
            m_cameraY = savedPhysical->position.y + kEyeHeightUnits;
            m_cameraZ = savedPhysical->position.z;
        } else if (player != nullptr) {
            // Controller-less V2 saves predate feet-origin ownership: their
            // player transform was the eye camera. Normalize it once before a
            // controller is registered and all future saves carry physics.
            syncBethesdaPlayerState(true);
            report.diagnostics.push_back(
                "normalized legacy controller-less player transform to feet origin");
        }
        if (controllerWasRegistered &&
            !m_bethesdaSession.physics().hasCharacter(playerId)) {
            m_bethesdaPlayerControllerRegistered = false;
            if (!registerBethesdaPlayerController()) return false;
        }
        m_bethesdaControllerOwnsCamera = m_walkMode;
        if (m_bethesdaControllerOwnsCamera) pullBethesdaPlayerControllerState();
    }
    restoreBethesdaActorsFromSession();
    std::string puzzleError;
    if (!configureGoldenClawPuzzleForCurrentSpace(puzzleError)) {
        VOX_LOGE("save") << "Golden Claw compatibility error after load: "
                          << puzzleError;
        return false;
    }
    for (const std::string& diagnostic : report.diagnostics) {
        VOX_LOGW("save") << diagnostic;
    }
    VOX_LOGI("save") << "loaded ODAI gameplay state from " << path.string();
    const std::string loadTitle = report.recoveredPrevious
        ? "Recovered previous save"
        : (report.contentReconciled ? "Game loaded with content changes" : "Game loaded");
    m_toasts.push(loadTitle, path.filename().string(), "gameplay-load");
    return true;
}

void BethesdaApp::drawPipBoyHud() {
    const float scale = contentScale();
    int screenWidth = 0;
    int screenHeight = 0;
    framebufferSize(screenWidth, screenHeight);
    const float margin = 16.0f * scale;

    // Skyrim's best dialogue decision is modal visual focus: once a
    // conversation begins, the compass, status meters, interaction prompts and
    // quest tracker stop competing with the person speaking. Preserve that
    // hierarchy for the remastered TES3 presentation while retaining
    // Morrowind's warm serif language inside the dialogue surface itself.
    if (m_streamIsMorrowind) {
        if (const SkinnedActor* speaker = talkingActor()) {
            if (const dialogue::DialogueNode* node = speaker->runtime.currentNode()) {
                drawDialoguePanel(*node, screenWidth, screenHeight, scale);
            }
            return;
        }
    }

    // Status strip, bottom-left: the readouts that belong on screen all the
    // time. Kept to one line so it never competes with the world.
    const int hours = static_cast<int>(m_timeOfDayHours);
    const int minutes = static_cast<int>((m_timeOfDayHours - static_cast<float>(hours)) * 60.0f);
    char status[192];
    const std::size_t locationCount = m_discoveredLocations.size();
    std::snprintf(
        status, sizeof(status), "%02d:%02d%s   %s   %zu location%s",
        hours, minutes, m_dayCyclePaused ? " (paused)" : "",
        m_walkMode ? "ON FOOT" : "FLY", locationCount, locationCount == 1u ? "" : "s");

    const float statusWidth = m_uiFont.measureText(status) + (margin * 1.5f);
    const float statusHeight = m_uiFont.lineHeightPx() + (10.0f * scale);
    ui::UiRect statusRect{};
    statusRect.minX = margin;
    statusRect.maxX = margin + statusWidth;
    statusRect.maxY = static_cast<float>(screenHeight) - margin;
    statusRect.minY = statusRect.maxY - statusHeight;
    m_uiDrawList.addRoundRectFilled(statusRect, kPipPanel, 3.0f * scale);
    m_uiDrawList.addRoundRect(statusRect, kPipGreenDim, 3.0f * scale, 1.0f * scale);
    m_uiDrawList.addText(
        m_uiFont, status,
        ui::UiVec2{statusRect.minX + (margin * 0.75f), statusRect.minY + (5.0f * scale)},
        kPipGreen);

    // Interaction prompt, centred low -- where an action prompt belongs, and
    // labelled for whichever device is driving.
    const bool clawPuzzleInReach = !m_menuOpen && m_activationLootActor < 0 &&
        m_activationActor < 0 &&
        goldenClawPuzzleInReach();
    if (clawPuzzleInReach) {
        const bethesda::RuntimeObject* keyhole =
            m_bethesdaSession.world().find(m_goldenClawPuzzle->door);
        const std::vector<std::int32_t>& states =
            keyhole->activatorState->puzzleStates;
        char prompt[192];
        std::snprintf(prompt, sizeof(prompt),
            "[1] Large %d   [2] Medium %d   [3] Small %d   [E] Use Golden Claw",
            states[0], states[1], states[2]);
        const float promptWidth = m_uiFont.measureText(prompt);
        m_uiDrawList.addText(
            m_uiFont, prompt,
            ui::UiVec2{(static_cast<float>(screenWidth) - promptWidth) * 0.5f,
                       static_cast<float>(screenHeight) - (96.0f * scale)},
            kPipGreen);
    } else if (const int usableDoor = findUsableDoor();
               usableDoor >= 0 && !m_menuOpen &&
               m_activationLootActor < 0 && m_activationActor < 0) {
        const importer::ImportedSceneDoor& door = m_doors[static_cast<std::size_t>(usableDoor)];
        char prompt[192];
        std::snprintf(
            prompt, sizeof(prompt), "%s  %s%s", m_navDriving ? "(A)" : "[E]",
            door.targetCellEditorId.empty() ? "Exit" : door.targetCellEditorId.c_str(),
            door.locked ? "  [LOCKED - BYPASS]" : "");
        const float promptWidth = m_uiFont.measureText(prompt);
        ui::UiVec2 promptPosition{};
        promptPosition.x = (static_cast<float>(screenWidth) - promptWidth) * 0.5f;
        promptPosition.y = static_cast<float>(screenHeight) - (96.0f * scale);
        m_uiDrawList.addText(m_uiFont, prompt, promptPosition, kPipGreen);
    }

    // Compass strip, top-centre, with a marker for Victor.
    //
    // Bearing convention: the camera's forward in XZ is (cos(yaw), sin(yaw)),
    // and Fallout's north is +Y in its own space, which this engine stores as
    // -Z. So north sits at yaw 270 and compass degrees are (yaw + 90) mod 360 --
    // worth writing down because getting it wrong gives a compass that is
    // plausibly wrong by 90 degrees, which is worse than none.
    {
        const auto compassDegrees = [](float yawDegrees) {
            float d = std::fmod(yawDegrees + 90.0f, 360.0f);
            return d < 0.0f ? d + 360.0f : d;
        };
        const float heading = compassDegrees(m_yawDegrees);
        static const char* kPoints[8] = {"N", "NE", "E", "SE", "S", "SW", "W", "NW"};
        const char* cardinal = kPoints[static_cast<int>((heading + 22.5f) / 45.0f) % 8];

        char headingText[64] = {};
        std::snprintf(headingText, sizeof(headingText), "%s  %3d°", cardinal,
                      static_cast<int>(heading + 0.5f) % 360);
        const float headingWidth = m_uiFont.measureText(headingText);
        m_uiDrawList.addText(
            m_uiFont, headingText,
            ui::UiVec2{(static_cast<float>(screenWidth) - headingWidth) * 0.5f, 12.0f * scale},
            kPipGreen);

        // Where Victor is from here, so "I cannot find him" becomes a bearing
        // and a distance rather than a search.
        if (m_victorIndex >= 0 && m_talkingActor < 0) {
            const SkinnedActor& victor = m_actors[static_cast<std::size_t>(m_victorIndex)];
            const float dx = victor.position[0] - m_cameraX;
            const float dz = victor.position[2] - m_cameraZ;
            const float distance = std::sqrt((dx * dx) + (dz * dz));
            const float toVictor =
                compassDegrees(std::atan2(dz, dx) * (180.0f / kPi));
            // Signed turn, so the hint says which way to turn rather than
            // leaving the player to subtract two bearings in their head.
            float turn = std::fmod((toVictor - heading) + 540.0f, 360.0f) - 180.0f;
            char victorText[96] = {};
            std::snprintf(victorText, sizeof(victorText), "Victor  %4d u  %s %d°",
                          static_cast<int>(distance),
                          turn >= 0.0f ? "turn right" : "turn left",
                          static_cast<int>(std::fabs(turn) + 0.5f));
            const float victorWidth = m_uiFont.measureText(victorText);
            m_uiDrawList.addText(
                m_uiFont, victorText,
                ui::UiVec2{(static_cast<float>(screenWidth) - victorWidth) * 0.5f, 34.0f * scale},
                std::fabs(turn) < 12.0f ? kPipGreen : kPipGreenDim);
        }

        if (m_streamer && !m_interiorStarted) {
            const importer::fnv::FalloutMapMarkerRecord* nearest = nullptr;
            float nearestDistanceSquared = 20000.0f * 20000.0f;
            for (const auto& marker : m_streamer->mapMarkers()) {
                if (marker.deleted || marker.initiallyDisabled || marker.name.empty() ||
                    marker.worldspaceFormId != m_streamer->currentWorldspaceFormId()) {
                    continue;
                }
                const float dx = marker.position[0] - m_cameraX;
                const float dz = -marker.position[1] - m_cameraZ;
                const float distanceSquared = (dx * dx) + (dz * dz);
                if (distanceSquared < nearestDistanceSquared) {
                    nearest = &marker;
                    nearestDistanceSquared = distanceSquared;
                }
            }
            if (nearest != nullptr) {
                const float dx = nearest->position[0] - m_cameraX;
                const float dz = -nearest->position[1] - m_cameraZ;
                const float markerBearing =
                    compassDegrees(std::atan2(dz, dx) * (180.0f / kPi));
                const float turn =
                    std::fmod((markerBearing - heading) + 540.0f, 360.0f) - 180.0f;
                char markerText[192] = {};
                std::snprintf(
                    markerText, sizeof(markerText), "%s  %s %d°  %d u",
                    nearest->name.c_str(), turn >= 0.0f ? "right" : "left",
                    static_cast<int>(std::fabs(turn) + 0.5f),
                    static_cast<int>(std::sqrt(nearestDistanceSquared)));
                const float width = m_uiFont.measureText(markerText);
                m_uiDrawList.addText(
                    m_uiFont, markerText,
                    ui::UiVec2{(static_cast<float>(screenWidth) - width) * 0.5f,
                               56.0f * scale},
                    m_discoveredMarkerIds.contains(nearest->referenceFormId)
                        ? kPipGreen : kPipGreenDim);
            }
        }
    }

    // TES3 does not invent objective markers. The tracker is the latest
    // authored journal entry from the quest the player pinned in the journal.
    if (m_streamIsMorrowind && m_bethesdaSessionConfigured &&
        !m_tes3JournalOpen && m_tes3JournalPanel != nullptr) {
        if (m_bethesdaSession.tes3().journal().chronology().size() !=
            m_tes3JournalSyncedVisits) {
            syncTes3JournalPanel();
        }
        if (const auto tracked = m_tes3JournalPanel->latestPinnedEntry();
            tracked.has_value()) {
            const float width = std::min(620.0f * scale,
                static_cast<float>(screenWidth) * 0.36f);
            const float padding = 16.0f * scale;
            const std::vector<std::string> lines =
                wrapTextToWidth(m_uiFont, tracked->text, width - (padding * 2.0f));
            const std::size_t shown = std::min<std::size_t>(lines.size(), 4u);
            const float lineHeight = m_uiFont.lineHeightPx() * 1.12f;
            ui::UiRect tracker;
            tracker.maxX = static_cast<float>(screenWidth) - margin;
            tracker.minX = tracker.maxX - width;
            tracker.minY = 88.0f * scale;
            tracker.maxY = tracker.minY + padding * 2.0f +
                lineHeight * static_cast<float>(shown + 1u);
            m_uiDrawList.addRoundRectFilled(tracker, kPipPanel, 4.0f * scale);
            m_uiDrawList.addRoundRect(tracker, kPipGreenDim, 4.0f * scale, 1.0f * scale);
            m_uiDrawList.addText(m_uiFontBold.valid() ? m_uiFontBold : m_uiFont,
                tracked->title.c_str(),
                {tracker.minX + padding, tracker.minY + padding * 0.65f}, kPipGreen);
            for (std::size_t line = 0u; line < shown; ++line) {
                m_uiDrawList.addText(m_uiFont, lines[line].c_str(),
                    {tracker.minX + padding,
                     tracker.minY + padding + lineHeight * static_cast<float>(line + 1u)},
                    kPipGreenDim);
            }
        }
    }

    // Runtime quest objectives are deliberately separate from imported scene
    // data. Skyrim wording is resolved from the owning plugin's localized
    // STRINGS table; stable identities remain a visible fallback for mods with
    // absent or malformed localization data.
    if (m_bethesdaSessionConfigured && !m_menuOpen) {
        std::vector<std::string> objectiveLines;
        for (const auto& [editorId, quest] : m_bethesdaSession.quests()) {
            for (const bethesda::QuestObjectiveState& objective : quest.objectives) {
                if (!objective.displayed || objective.completed) continue;
                const std::string label = objective.displayText.empty()
                    ? editorId + " objective " + std::to_string(objective.index)
                    : objective.displayText;
                objectiveLines.push_back(
                    std::string(objective.failed ? "[!] " : "[ ] ") + label);
            }
        }
        if (!objectiveLines.empty()) {
            constexpr std::size_t kMaxVisibleObjectives = 6u;
            if (objectiveLines.size() > kMaxVisibleObjectives) {
                objectiveLines.resize(kMaxVisibleObjectives);
            }
            float width = m_uiFont.measureText("ACTIVE OBJECTIVES");
            for (const std::string& line : objectiveLines) {
                width = std::max(width, m_uiFont.measureText(line));
            }
            const float padding = 12.0f * scale;
            const float lineHeight = m_uiFont.lineHeightPx() + (4.0f * scale);
            ui::UiRect panel;
            panel.maxX = static_cast<float>(screenWidth) - margin;
            panel.minX = panel.maxX - width - (padding * 2.0f);
            panel.minY = 88.0f * scale;
            panel.maxY = panel.minY +
                lineHeight * static_cast<float>(objectiveLines.size() + 1u) + padding;
            m_uiDrawList.addRoundRectFilled(panel, kPipPanel, 3.0f * scale);
            m_uiDrawList.addRoundRect(panel, kPipGreenDim, 3.0f * scale, 1.0f * scale);
            m_uiDrawList.addText(m_uiFont, "ACTIVE OBJECTIVES",
                ui::UiVec2{panel.minX + padding, panel.minY + (6.0f * scale)}, kPipGreen);
            for (std::size_t index = 0u; index < objectiveLines.size(); ++index) {
                m_uiDrawList.addText(m_uiFont, objectiveLines[index].c_str(),
                    ui::UiVec2{panel.minX + padding,
                        panel.minY + lineHeight * static_cast<float>(index + 1u) +
                            (6.0f * scale)},
                    objectiveLines[index].starts_with("[!]") ? kPipGreenDim : kPipGreen);
            }
        }
    }

    // Victor. The conversation is drawn straight onto the HUD draw list rather
    // than through DialoguePanel: the panel wants a widget tree, and this app's
    // HUD is immediate-mode text, so one path is simpler than bridging two.
    if (const SkinnedActor* speaker = talkingActor()) {
        if (const dialogue::DialogueNode* node = speaker->runtime.currentNode()) {
            drawDialoguePanel(*node, screenWidth, screenHeight, scale);
        }
    } else if (m_activationLootActor >= 0 &&
               m_activationLootActor < static_cast<int>(m_actors.size())) {
        const std::string prompt =
            "E  search " +
            m_actors[static_cast<std::size_t>(m_activationLootActor)].displayName();
        m_uiDrawList.addText(m_uiFont, prompt,
            ui::UiVec2{64.0f * scale,
                static_cast<float>(screenHeight) - (132.0f * scale)},
            kPipGreen);
    } else if (m_activationActor >= 0 &&
               m_activationActor < static_cast<int>(m_actors.size())) {
        const std::string prompt =
            "E  talk to " + m_actors[static_cast<std::size_t>(m_activationActor)].displayName();
        m_uiDrawList.addText(m_uiFont, prompt,
                             ui::UiVec2{64.0f * scale, static_cast<float>(screenHeight) - (132.0f * scale)},
                             kPipGreen);
    }

    // Hint line, top-left. Names the buttons of whichever device is in use --
    // showing "Tab" to someone holding a controller is worse than showing
    // nothing.
    const char* hint = m_navDriving
        ? "(Start) menu   (LS) move   (A) use   J journal"
        : "J journal   Esc menu   [ ] time   P quit   Tab cursor";
    m_uiDrawList.addText(m_uiFont, hint, ui::UiVec2{margin, margin}, kPipGreenDim);
}

void BethesdaApp::buildWeatherChoices() {
    if (!m_weatherChoices.empty()) {
        return;
    }
    // Every weather the load order defines, not just the ones this worldspace's
    // climate runs. Scoping to the climate was the first attempt and it is too
    // narrow to be useful: NVDefaultClimate names exactly TWO, so the picker
    // offered clear-day and clear-night out of the 63 vanilla ships. The point
    // of the picker is looking at skies, and the list scrolls and pages, so
    // there is nothing to be gained by hiding most of them.
    m_weatherChoices.reserve(m_weatherTables.weathers.size());
    for (const auto& [formId, record] : m_weatherTables.weathers) {
        m_weatherChoices.push_back(formId);
    }
    // By name, because the player is reading names. formID order is load-order
    // order, which shuffles when a plugin is added and is meaningless on screen.
    std::sort(
        m_weatherChoices.begin(), m_weatherChoices.end(),
        [this](std::uint32_t a, std::uint32_t b) {
            const auto* ra = m_weatherTables.findWeather(a);
            const auto* rb = m_weatherTables.findWeather(b);
            const std::string& na = ra != nullptr ? ra->editorId : std::string{};
            const std::string& nb = rb != nullptr ? rb->editorId : std::string{};
            return na < nb;
        });
    m_weatherChoices.erase(
        std::unique(m_weatherChoices.begin(), m_weatherChoices.end()), m_weatherChoices.end());
    VOX_LOGI("newvegas") << "weather picker: " << m_weatherChoices.size() << " choices";
}

void BethesdaApp::openWeatherPicker() {
    m_weatherPickerOpen = true;
    buildWeatherChoices();
    // Open ON the active weather rather than at the top. The list is sorted by
    // name, so row 1 is alphabetical happenstance -- with vanilla's 63 that is a
    // Pitt DLC weather, nowhere near whatever is currently over the Mojave --
    // and scrolling back to where you already were is the first thing the player
    // would otherwise have to do.
    const auto found =
        std::find(m_weatherChoices.begin(), m_weatherChoices.end(), m_activeWeatherFormId);
    const int activeIndex = (found != m_weatherChoices.end())
        ? static_cast<int>(found - m_weatherChoices.begin())
        : 0;
    // A few rows of context above it rather than pinned to the top edge.
    m_weatherScrollTop = std::max(0, activeIndex - 4);
    m_weatherFocus.setFocus(std::min(activeIndex, 4));
}

bool BethesdaApp::drawWeatherPicker(const ui::UiRect& panelArea, float scale) {
    buildWeatherChoices();
    if (m_weatherChoices.empty()) {
        m_weatherPickerOpen = false;
        return false;
    }

    const int choiceCount = static_cast<int>(m_weatherChoices.size());
    // A fixed window of rows, sized to what fits comfortably rather than to the
    // list -- the list can be 473 long.
    constexpr int kVisibleRows = 10;
    const int visibleRows = std::min(kVisibleRows, choiceCount);
    const float lineHeight = m_uiFont.lineHeightPx();
    const float rowHeight = lineHeight + (10.0f * scale);
    const float headerBand = lineHeight + (28.0f * scale);
    const float footerBand = lineHeight + (22.0f * scale);

    float contentWidth = m_uiFont.measureText("WEATHER");
    for (int i = 0; i < choiceCount; ++i) {
        const auto* record = m_weatherTables.findWeather(m_weatherChoices[i]);
        if (record != nullptr) {
            contentWidth = std::max(contentWidth, m_uiFont.measureText(record->editorId.c_str()));
        }
    }
    const float panelWidth = std::max(560.0f * scale, contentWidth + (96.0f * scale));
    const float panelHeight =
        headerBand + (rowHeight * static_cast<float>(visibleRows)) + footerBand;
    ui::UiRect panel{};
    panel.minX = ((panelArea.minX + panelArea.maxX) - panelWidth) * 0.5f;
    panel.maxX = panel.minX + panelWidth;
    panel.minY = ((panelArea.minY + panelArea.maxY) - panelHeight) * 0.5f;
    panel.maxY = panel.minY + panelHeight;
    m_uiDrawList.addRoundRectFilled(panel, kPipPanelSolid, 4.0f * scale);
    m_uiDrawList.addRoundRect(panel, kPipGreen, 4.0f * scale, 1.5f * scale);

    char header[64];
    std::snprintf(header, sizeof(header), "WEATHER  (%d)", choiceCount);
    m_uiDrawList.addText(
        m_uiFont, header,
        ui::UiVec2{panel.minX + (24.0f * scale), panel.minY + (14.0f * scale)}, kPipGreen);

    // The focus ring holds only the VISIBLE rows, so navigating off either end
    // has to scroll the window rather than move focus. Registering all 473 would
    // let focus land on a row that is not drawn, and the highlight would simply
    // vanish.
    m_weatherFocus.beginFrame();
    std::vector<ui::UiRect> rows(static_cast<std::size_t>(visibleRows));
    for (int i = 0; i < visibleRows; ++i) {
        ui::UiRect row{};
        row.minX = panel.minX + (16.0f * scale);
        row.maxX = panel.maxX - (16.0f * scale);
        row.minY = panel.minY + headerBand + (static_cast<float>(i) * rowHeight);
        row.maxY = row.minY + rowHeight - (3.0f * scale);
        rows[static_cast<std::size_t>(i)] = row;
        m_weatherFocus.addItem(row);
    }
    if (!m_navDriving) {
        double cursorX = 0.0;
        double cursorY = 0.0;
        glfwGetCursorPos(m_window, &cursorX, &cursorY);
        m_weatherFocus.focusHovered(
            ui::UiVec2{static_cast<float>(cursorX), static_cast<float>(cursorY)});
    }

    // Scroll BEFORE navigating, so a press at the edge of the window moves the
    // list by one instead of being swallowed by the focus ring's own clamp.
    const int maxScroll = std::max(0, choiceCount - visibleRows);
    const int focusedRow = std::max(0, m_weatherFocus.focused());
    if (m_nav.pressed(ui::UiNavAction::Down) && focusedRow == visibleRows - 1 &&
        m_weatherScrollTop < maxScroll) {
        ++m_weatherScrollTop;
    } else if (m_nav.pressed(ui::UiNavAction::Up) && focusedRow == 0 && m_weatherScrollTop > 0) {
        --m_weatherScrollTop;
    } else {
        m_weatherFocus.applyNavigation(m_nav);
    }
    // Shoulder buttons page. 473 entries one row at a time is a minute of
    // holding a stick; this makes the far end of the list reachable.
    if (m_nav.pressed(ui::UiNavAction::NextTab)) {
        m_weatherScrollTop = std::min(maxScroll, m_weatherScrollTop + visibleRows);
    }
    if (m_nav.pressed(ui::UiNavAction::PrevTab)) {
        m_weatherScrollTop = std::max(0, m_weatherScrollTop - visibleRows);
    }
    m_weatherScrollTop = std::clamp(m_weatherScrollTop, 0, maxScroll);

    for (int i = 0; i < visibleRows; ++i) {
        const int choiceIndex = m_weatherScrollTop + i;
        if (choiceIndex >= choiceCount) {
            break;
        }
        const std::uint32_t formId = m_weatherChoices[static_cast<std::size_t>(choiceIndex)];
        const auto* record = m_weatherTables.findWeather(formId);
        const bool focused = m_weatherFocus.isFocused(i);
        const bool isActive = (formId == m_activeWeatherFormId);
        if (focused) {
            m_uiDrawList.addRoundRectFilled(
                rows[static_cast<std::size_t>(i)], ui::UiColor{0.16f, 0.42f, 0.20f, 0.85f},
                3.0f * scale);
        }
        // The active weather is marked as well as highlighted: focus and
        // "currently applied" are different things, and on this palette a single
        // green cue for both is unreadable.
        const std::string label =
            std::string{isActive ? "> " : "  "} + (record != nullptr ? record->editorId : "<?>");
        m_uiDrawList.addText(
            m_uiFont, label.c_str(),
            ui::UiVec2{
                rows[static_cast<std::size_t>(i)].minX + (16.0f * scale),
                rows[static_cast<std::size_t>(i)].minY + (5.0f * scale)},
            focused ? kPipGreen : kPipGreenDim);
    }

    if (m_nav.pressed(ui::UiNavAction::Accept)) {
        const int choiceIndex = m_weatherScrollTop + focusedRow;
        if (choiceIndex >= 0 && choiceIndex < choiceCount) {
            selectWeather(m_weatherChoices[static_cast<std::size_t>(choiceIndex)]);
        }
    }

    char footer[96];
    std::snprintf(
        footer, sizeof(footer), "%s    %d-%d of %d",
        m_navDriving ? "(A) apply   (LB/RB) page   (B) back"
                     : "Enter apply   Q/E page   Esc back",
        m_weatherScrollTop + 1, std::min(choiceCount, m_weatherScrollTop + visibleRows),
        choiceCount);
    m_uiDrawList.addText(
        m_uiFont, footer,
        ui::UiVec2{panel.minX + (24.0f * scale), panel.maxY - footerBand + (12.0f * scale)},
        kPipGreenDim);
    return true;
}

bool BethesdaApp::drawCompatibilityPanel(const ui::UiRect& panelArea, float scale) {
    if (!m_contentProfile.has_value()) {
        m_compatibilityPanelOpen = false;
        return false;
    }
    const importer::fnv::ResolvedContentProfile& profile = *m_contentProfile;
    const float lineHeight = m_uiFont.lineHeightPx();
    const std::size_t visibleDiagnostics = std::min<std::size_t>(profile.diagnostics.size(), 8u);
    const float panelWidth = std::min(900.0f * scale, panelArea.maxX - 48.0f * scale);
    const float panelHeight = (180.0f + 42.0f * static_cast<float>(visibleDiagnostics)) * scale;
    ui::UiRect panel{};
    panel.minX = ((panelArea.minX + panelArea.maxX) - panelWidth) * 0.5f;
    panel.maxX = panel.minX + panelWidth;
    panel.minY = ((panelArea.minY + panelArea.maxY) - panelHeight) * 0.5f;
    panel.maxY = panel.minY + panelHeight;
    m_uiDrawList.addRoundRectFilled(panel, kPipPanelSolid, 4.0f * scale);
    m_uiDrawList.addRoundRect(panel, kPipGreen, 4.0f * scale, 1.5f * scale);
    float y = panel.minY + 16.0f * scale;
    m_uiDrawList.addText(m_uiFont, "MOD COMPATIBILITY",
        ui::UiVec2{panel.minX + 24.0f * scale, y}, kPipGreen);
    y += lineHeight + 10.0f * scale;
    const std::string identity = profile.name + "  [" +
        importer::fnv::bethesdaGameName(profile.game) + "]";
    m_uiDrawList.addText(m_uiFont, identity.c_str(),
        ui::UiVec2{panel.minX + 24.0f * scale, y}, kPipGreenDim);
    y += lineHeight + 6.0f * scale;
    const std::string counts = std::to_string(profile.layers.size()) + " layers   " +
        std::to_string(profile.plugins.size()) + " plugins   " +
        std::to_string(profile.archives.size()) + " explicit archives";
    m_uiDrawList.addText(m_uiFont, counts.c_str(),
        ui::UiVec2{panel.minX + 24.0f * scale, y}, kPipGreenDim);
    y += lineHeight + 12.0f * scale;
    if (visibleDiagnostics == 0u) {
        m_uiDrawList.addText(m_uiFont, "No profile compatibility warnings.",
            ui::UiVec2{panel.minX + 24.0f * scale, y}, kPipGreen);
    } else {
        for (std::size_t i = 0; i < visibleDiagnostics; ++i) {
            const importer::fnv::ContentDiagnostic& item = profile.diagnostics[i];
            const std::string text = (item.severity ==
                importer::fnv::ContentDiagnosticSeverity::Error ? "ERROR " : "WARN  ") +
                item.code + ": " + item.message;
            m_uiDrawList.addText(m_uiFont, text.c_str(),
                ui::UiVec2{panel.minX + 24.0f * scale, y},
                item.severity == importer::fnv::ContentDiagnosticSeverity::Error
                    ? ui::UiColor{1.0f, 0.45f, 0.3f, 1.0f} : kPipGreenDim);
            y += lineHeight + 5.0f * scale;
        }
    }
    const char* footer = m_navDriving ? "(B) back" : "Esc back";
    m_uiDrawList.addText(m_uiFont, footer,
        ui::UiVec2{panel.minX + 24.0f * scale, panel.maxY - lineHeight - 12.0f * scale},
        kPipGreenDim);
    return true;
}

void BethesdaApp::drawPauseMenu() {
    if (!m_menuOpen) {
        // Keep the ring empty so a stale focus index cannot survive a close and
        // reopen and act on the wrong entry.
        m_menuFocus.beginFrame();
        m_weatherFocus.beginFrame();
        m_weatherPickerOpen = false;
        m_compatibilityPanelOpen = false;
        return;
    }
    const float scale = contentScale();
    int screenWidth = 0;
    int screenHeight = 0;
    framebufferSize(screenWidth, screenHeight);

    // Dim the world so the menu is unambiguously modal.
    ui::UiRect full{0.0f, 0.0f, static_cast<float>(screenWidth), static_cast<float>(screenHeight)};
    m_uiDrawList.addRectFilled(full, ui::UiColor{0.0f, 0.02f, 0.0f, 0.55f});

    // The picker REPLACES the menu rather than layering over it: two focus rings
    // reading the same nav input would both move, and backing out would land on
    // whichever row the hidden one had drifted to.
    if (m_weatherPickerOpen) {
        m_menuFocus.beginFrame();
        if (drawWeatherPicker(full, scale)) {
            return;
        }
    }
    if (m_compatibilityPanelOpen) {
        m_menuFocus.beginFrame();
        if (drawCompatibilityPanel(full, scale)) {
            return;
        }
    }

    struct Entry {
        const char* label;
        const char* value;
    };
    char timeValue[32];
    std::snprintf(timeValue, sizeof(timeValue), "%s", m_dayCyclePaused ? "Paused" : "Running");
    char regionValue[32];
    std::snprintf(regionValue, sizeof(regionValue), "%zu", m_discoveredRegions.size());
    const importer::fnv::FalloutWeatherRecord* activeWeather =
        m_activeWeatherFormId != 0u ? m_weatherTables.findWeather(m_activeWeatherFormId) : nullptr;
    const std::string weatherValue =
        activeWeather != nullptr ? activeWeather->editorId : std::string{"<none>"};
    char modValue[48];
    std::snprintf(
        modValue, sizeof(modValue), "%zu layers / %zu warnings",
        m_contentProfile.has_value() ? m_contentProfile->layers.size() : 0u,
        m_contentProfile.has_value() ? m_contentProfile->diagnostics.size() : 0u);
    const Entry entries[] = {
        {m_walkMode ? "Movement: On Foot" : "Movement: Fly", ""},
        {"Day cycle", timeValue},
        {"Weather", weatherValue.c_str()},
        {"Regions discovered", regionValue},
        {"Mod compatibility", modValue},
        {"Resume", ""},
    };
    constexpr std::size_t kEntryCount = sizeof(entries) / sizeof(entries[0]);

    // Panel metrics as three explicit bands -- header, rows, footer -- rather
    // than one fudged total. The first version folded the footer into a single
    // padding constant and the footer text landed on top of the last row: the
    // arithmetic has to close, and it only closes if each band is named.
    // Every band is derived from the font's line height rather than being a
    // fixed pixel count. The type scale moved once already (to a TV size) and
    // the fixed values silently stopped fitting -- the header ran into the
    // first row. Derived bands cannot drift out of step with the type.
    const float lineHeight = m_uiFont.lineHeightPx();
    const float rowHeight = lineHeight + (16.0f * scale);
    const float headerBand = lineHeight + (28.0f * scale);
    const float footerBand = lineHeight + (22.0f * scale);
    // Wide enough for the widest row, so a longer label cannot overrun the
    // panel it is drawn inside.
    float contentWidth = m_uiFont.measureText("PIP-BOY 3000");
    for (const Entry& entry : entries) {
        const float rowWidth = m_uiFont.measureText(entry.label) +
            (entry.value[0] != '\0' ? m_uiFont.measureText(entry.value) + (48.0f * scale) : 0.0f);
        contentWidth = std::max(contentWidth, rowWidth);
    }
    const float panelWidth = std::max(460.0f * scale, contentWidth + (64.0f * scale));
    const float panelHeight =
        headerBand + (rowHeight * static_cast<float>(kEntryCount)) + footerBand;
    ui::UiRect panel{};
    panel.minX = (static_cast<float>(screenWidth) - panelWidth) * 0.5f;
    panel.maxX = panel.minX + panelWidth;
    panel.minY = (static_cast<float>(screenHeight) - panelHeight) * 0.5f;
    panel.maxY = panel.minY + panelHeight;
    m_uiDrawList.addRoundRectFilled(panel, kPipPanelSolid, 4.0f * scale);
    m_uiDrawList.addRoundRect(panel, kPipGreen, 4.0f * scale, 1.5f * scale);
    m_uiDrawList.addText(
        m_uiFont, "PIP-BOY 3000",
        ui::UiVec2{panel.minX + (24.0f * scale), panel.minY + (14.0f * scale)}, kPipGreen);

    // Register every row with the focus ring, THEN navigate. Navigating with a
    // partial list would let the first row absorb every move.
    m_menuFocus.beginFrame();
    ui::UiRect rows[kEntryCount];
    for (std::size_t i = 0; i < kEntryCount; ++i) {
        ui::UiRect row{};
        row.minX = panel.minX + (16.0f * scale);
        row.maxX = panel.maxX - (16.0f * scale);
        row.minY = panel.minY + headerBand + (static_cast<float>(i) * rowHeight);
        row.maxY = row.minY + rowHeight - (4.0f * scale);
        rows[i] = row;
        m_menuFocus.addItem(row);
    }
    if (!m_navDriving) {
        double cursorX = 0.0;
        double cursorY = 0.0;
        glfwGetCursorPos(m_window, &cursorX, &cursorY);
        m_menuFocus.focusHovered(
            ui::UiVec2{static_cast<float>(cursorX), static_cast<float>(cursorY)});
    }
    m_menuFocus.applyNavigation(m_nav);

    for (std::size_t i = 0; i < kEntryCount; ++i) {
        const bool focused = m_menuFocus.isFocused(static_cast<int>(i));
        if (focused) {
            m_uiDrawList.addRoundRectFilled(
                rows[i], ui::UiColor{0.16f, 0.42f, 0.20f, 0.85f}, 3.0f * scale);
        }
        m_uiDrawList.addText(
            m_uiFont, entries[i].label,
            ui::UiVec2{rows[i].minX + (16.0f * scale), rows[i].minY + (8.0f * scale)},
            focused ? kPipGreen : kPipGreenDim);
        if (entries[i].value[0] != '\0') {
            const float valueWidth = m_uiFont.measureText(entries[i].value);
            m_uiDrawList.addText(
                m_uiFont, entries[i].value,
                ui::UiVec2{rows[i].maxX - valueWidth - (16.0f * scale), rows[i].minY + (8.0f * scale)},
                focused ? kPipGreen : kPipGreenDim);
        }
    }

    if (m_nav.pressed(ui::UiNavAction::Accept)) {
        switch (m_menuFocus.focused()) {
            case 0: m_walkMode = !m_walkMode; break;
            case 1: m_dayCyclePaused = !m_dayCyclePaused; break;
            case 2: openWeatherPicker(); break;
            case 3: break;  // a readout, not an action
            case 4:
                if (m_contentProfile.has_value()) m_compatibilityPanelOpen = true;
                break;
            case 5: m_menuOpen = false; setMouseCaptured(true); break;
            default: break;
        }
    }

    const char* footer = m_navDriving ? "(A) select    (B) back" : "Enter select    Esc back";
    m_uiDrawList.addText(
        m_uiFont, footer,
        ui::UiVec2{panel.minX + (24.0f * scale), panel.maxY - footerBand + (12.0f * scale)},
        kPipGreenDim);

    if (!m_discoveredLocations.empty()) {
        constexpr std::size_t kVisibleLocations = 8u;
        const std::size_t first = m_discoveredLocations.size() > kVisibleLocations
            ? m_discoveredLocations.size() - kVisibleLocations : 0u;
        const float listWidth = 360.0f * scale;
        const float listHeight =
            (m_uiFont.lineHeightPx() + 20.0f * scale) +
            static_cast<float>(m_discoveredLocations.size() - first) *
                (m_uiFont.lineHeightPx() + 6.0f * scale);
        ui::UiRect locations{
            std::min(panel.maxX + 18.0f * scale,
                     static_cast<float>(screenWidth) - listWidth - 16.0f * scale),
            panel.minY, 0.0f, panel.minY + listHeight};
        locations.maxX = locations.minX + listWidth;
        m_uiDrawList.addRoundRectFilled(locations, kPipPanelSolid, 4.0f * scale);
        m_uiDrawList.addRoundRect(locations, kPipGreenDim, 4.0f * scale, 1.0f * scale);
        m_uiDrawList.addText(
            m_uiFont, "DISCOVERED LOCATIONS",
            ui::UiVec2{locations.minX + 16.0f * scale,
                       locations.minY + 10.0f * scale}, kPipGreen);
        float y = locations.minY + m_uiFont.lineHeightPx() + 18.0f * scale;
        for (std::size_t i = first; i < m_discoveredLocations.size(); ++i) {
            m_uiDrawList.addText(
                m_uiFont, m_discoveredLocations[i].name.c_str(),
                ui::UiVec2{locations.minX + 16.0f * scale, y}, kPipGreenDim);
            y += m_uiFont.lineHeightPx() + 6.0f * scale;
        }
    }
}

void BethesdaApp::drawDialoguePanel(
    const dialogue::DialogueNode& node, int screenWidth, int screenHeight, float scale
) {
    const bool compactTes3 = m_streamIsMorrowind;
    // Fall back to the body face when a dialogue bake failed; the layout below
    // measures whatever font it is handed, so it stays correct either way.
    const ui::Font& lineFont = compactTes3 && m_tes3JournalFont.valid()
        ? m_tes3JournalFont
        : (m_dialogueFont.valid() ? m_dialogueFont : m_uiFont);
    const ui::Font& choiceFont = compactTes3 && m_tes3JournalFont.valid()
        ? m_tes3JournalFont
        : (m_dialogueChoiceFont.valid() ? m_dialogueChoiceFont : m_uiFont);
    const ui::Font& speakerFont = compactTes3 && m_tes3JournalBoldFont.valid()
        ? m_tes3JournalBoldFont
        : (m_uiFontBold.valid() ? m_uiFontBold : m_uiFont);
    const ui::Font& footerFont = compactTes3 && m_tes3JournalFont.valid()
        ? m_tes3JournalFont : m_uiFont;

    const auto width = static_cast<float>(screenWidth);
    const auto height = static_cast<float>(screenHeight);
    const ui::UiColor dialogueText = compactTes3
        ? ui::UiColor{0.91f, 0.83f, 0.68f, 1.0f} : kPipGreen;
    const ui::UiColor dialogueMuted = compactTes3
        ? ui::UiColor{0.68f, 0.61f, 0.50f, 1.0f} : kPipGreenDim;
    const ui::UiColor dialogueBorder = compactTes3
        ? ui::UiColor{0.58f, 0.48f, 0.34f, 0.96f} : kPipGreenDim;
    const ui::UiColor dialoguePanel = compactTes3
        ? ui::UiColor{0.075f, 0.065f, 0.052f, 0.91f} : kPipPanelSolid;

    // Width is capped in *scaled* units as well as as a fraction of the screen.
    // A line of text that spans an entire 4K width is unreadable no matter how
    // big the glyphs are -- the eye loses the start of the next line -- so the
    // card stops growing once it is wide enough for a comfortable measure.
    const float panelWidth = compactTes3
        ? std::min(width * 0.68f, 1280.0f * scale)
        : std::min(width * 0.74f, 1500.0f * scale);
    const float padding = (compactTes3 ? 22.0f : 40.0f) * scale;
    const float innerWidth = panelWidth - (padding * 2.0f);

    const std::vector<std::string> spokenLines =
        wrapTextToWidth(lineFont, node.text, innerWidth);
    const float spokenLineHeight =
        lineFont.lineHeightPx() * (compactTes3 ? 1.04f : 1.18f);

    // Replies are indented past their number, and the wrap has to account for
    // that or a long reply overruns the card it is measured against.
    const float choiceIndent = (compactTes3 ? 44.0f : 56.0f) * scale;
    const float choiceRowPadding = (compactTes3 ? 8.0f : 14.0f) * scale;
    const float choiceLineHeight =
        choiceFont.lineHeightPx() * (compactTes3 ? 1.05f : 1.15f);
    const SkinnedActor* speakingActor = talkingActor();
    const auto choices = speakingActor != nullptr
        ? speakingActor->runtime.availableChoices()
        : decltype(speakingActor->runtime.availableChoices()){};
    const std::size_t choiceCount = std::min<std::size_t>(choices.size(), 9u);
    std::vector<std::vector<std::string>> choiceLines;
    std::vector<float> choiceHeights;
    choiceLines.reserve(choiceCount);
    choiceHeights.reserve(choiceCount);
    for (std::size_t i = 0; i < choiceCount; ++i) {
        choiceLines.push_back(
            wrapTextToWidth(choiceFont, choices[i]->text, innerWidth - choiceIndent));
        const float rows = static_cast<float>(std::max<std::size_t>(choiceLines.back().size(), 1u));
        choiceHeights.push_back((rows * choiceLineHeight) + (choiceRowPadding * 2.0f));
    }

    const float speakerHeight = speakerFont.lineHeightPx();
    // Weighted toward the replies: the rule belongs to the block above it, and
    // an equal gap on both sides made the first reply's highlight border read
    // as if it were touching the rule.
    const float ruleGapAbove = (compactTes3 ? 10.0f : 22.0f) * scale;
    const float ruleGapBelow = (compactTes3 ? 12.0f : 30.0f) * scale;
    const float ruleGap = ruleGapAbove + ruleGapBelow;
    const float footerHeight =
        footerFont.lineHeightPx() + ((compactTes3 ? 8.0f : 18.0f) * scale);
    const float speakerGap = (compactTes3 ? 5.0f : 12.0f) * scale;
    const float spokenHeight =
        static_cast<float>(std::max<std::size_t>(spokenLines.size(), 1u)) * spokenLineHeight;

    // THE CARD IS CAPPED, AND THE REPLIES SCROLL INSIDE IT.
    //
    // It used to grow with the reply count and stay centred, so Easy Pete's nine
    // replies made a card spanning nearly the whole screen -- and the camera,
    // which frames the speaker's face just above the card's top edge, had
    // nowhere left to put him. He was completely hidden behind his own
    // dialogue. A conversation must not hide the person talking, which is the
    // same rule the pitch offset exists to serve; the offset simply cannot
    // honour it once the card has eaten the frame.
    //
    // 0.62 leaves the top third of the screen clear, which at this card's width
    // is enough for a head and shoulders at conversation distance.
    const float fixedHeight = (padding * 2.0f) + speakerHeight + speakerGap + spokenHeight +
                              ruleGap + footerHeight;
    const float maxPanelHeight = height * (compactTes3 ? 0.43f : 0.62f);
    const float choiceBudget = std::max(0.0f, maxPanelHeight - fixedHeight);

    // The window of replies that fits, slid just far enough to keep the
    // highlighted one inside it. Sliding by one rather than paging keeps the
    // list still under the cursor for as long as possible.
    const auto fitFrom = [&](std::size_t start) {
        float used = 0.0f;
        std::size_t shown = 0;
        for (std::size_t i = start; i < choiceCount; ++i) {
            if (shown > 0u && (used + choiceHeights[i]) > choiceBudget) {
                break;
            }
            used += choiceHeights[i];
            ++shown;
        }
        return shown;
    };
    std::size_t firstChoice = 0;
    std::size_t visibleChoices = fitFrom(0);
    const auto selected = static_cast<std::size_t>(std::max(m_dialogueChoice, 0));
    while (selected >= (firstChoice + visibleChoices) && (firstChoice + visibleChoices) < choiceCount) {
        ++firstChoice;
        visibleChoices = fitFrom(firstChoice);
    }
    float choicesHeight = 0.0f;
    for (std::size_t i = firstChoice; i < firstChoice + visibleChoices; ++i) {
        choicesHeight += choiceHeights[i];
    }
    const bool choicesClipped = visibleChoices < choiceCount;

    const float panelHeight = (padding * 2.0f) + speakerHeight + speakerGap + spokenHeight +
                              ruleGap + choicesHeight + footerHeight;

    const float panelX = (width - panelWidth) * 0.5f;
    // Keep the portrait area above the text, especially at native HiDPI where
    // the correctly density-scaled type makes this card physically tall.
    // Centring a 62%-high card left only ~19% of the frame for the speaker;
    // even a correct face aim then cropped the head and presented the torso.
    // A small bottom safe area preserves the HUD separation while giving the
    // camera an honest head-and-shoulders viewport.
    const float panelY = std::max(
        0.0f, height - panelHeight - (height * (compactTes3 ? 0.02f : 0.035f)));
    const ui::UiRect panel{panelX, panelY, panelX + panelWidth, panelY + panelHeight};
    const float corner = (compactTes3 ? 4.0f : 10.0f) * scale;
    // Published for updateCamera, which frames Victor's face above this edge.
    m_dialoguePanelTopPx = panelY;

    // The card sits over the world, so it needs to separate from whatever is
    // behind it: a shadow to lift it off the terrain, a near-opaque fill so
    // text never competes with a bright sky, and a phosphor edge to tie it to
    // the rest of the HUD.
    m_uiDrawList.addDropShadow(panel, ui::UiColor{0.0f, 0.0f, 0.0f, 0.55f}, 18.0f * scale, corner);
    m_uiDrawList.addRoundRectFilled(panel, dialoguePanel, corner);
    m_uiDrawList.addRoundRect(panel, dialogueBorder, corner, 2.0f * scale);
    if (compactTes3) {
        // A restrained inset rule gives the window the layered carved-frame
        // read of Morrowind's menus without spending portrait space on a large
        // ornamental bitmap or introducing another texture dependency.
        const float inset = 5.0f * scale;
        m_uiDrawList.addRoundRect(
            ui::UiRect{panel.minX + inset, panel.minY + inset,
                       panel.maxX - inset, panel.maxY - inset},
            ui::UiColor{dialogueBorder.r, dialogueBorder.g, dialogueBorder.b, 0.38f},
            corner * 0.65f, 1.0f * scale);
    }

    float y = panel.minY + padding;

    // Speaker, centred over the line, in caps -- a name label, not prose.
    std::string speaker = node.speaker;
    if (speaker.empty()) {
        speaker = speakingActor != nullptr ? speakingActor->displayName() : std::string("?");
    }
    for (char& c : speaker) {
        c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    }
    m_uiDrawList.addText(
        speakerFont, speaker,
        ui::UiVec2{panel.minX + ((panelWidth - speakerFont.measureText(speaker)) * 0.5f), y},
        dialogueMuted);
    y += speakerHeight + speakerGap;

    // What Victor says: centred, because it is one short block and centring it
    // under the name reads as a single unit.
    for (const std::string& text : spokenLines) {
        m_uiDrawList.addText(
            lineFont, text,
            ui::UiVec2{panel.minX + ((panelWidth - lineFont.measureText(text)) * 0.5f), y},
            dialogueText);
        y += spokenLineHeight;
    }

    y += ruleGapAbove;
    m_uiDrawList.addRectFilled(
        ui::UiRect{panel.minX + padding, y, panel.maxX - padding, y + (1.0f * scale)},
        ui::UiColor{dialogueMuted.r, dialogueMuted.g, dialogueMuted.b, 0.45f});
    y += ruleGapBelow;

    // The replies: LEFT aligned, unlike the block above. They are a list to be
    // scanned down, and centring a list makes every row start in a different
    // place, which is exactly what the eye uses to track position.
    for (std::size_t i = firstChoice; i < firstChoice + visibleChoices; ++i) {
        const float rowHeight = choiceHeights[i];
        const bool selected = static_cast<int>(i) == m_dialogueChoice;
        const ui::UiRect row{panel.minX + (padding * 0.5f), y,
                             panel.maxX - (padding * 0.5f), y + rowHeight};

        // Selection is stated three ways at once, because any one of them alone
        // fails somewhere: a fill is invisible to the colour-blind against a
        // green-on-green palette, a colour change alone is easy to miss across
        // a room, and a caret alone is small. Together they are unmistakable at
        // TV distance.
        if (selected) {
            m_uiDrawList.addRoundRectFilled(
                row, ui::UiColor{dialogueText.r, dialogueText.g, dialogueText.b,
                                 compactTes3 ? 0.13f : 0.20f}, corner * 0.6f);
            m_uiDrawList.addRoundRect(
                row, dialogueText, corner * 0.6f,
                (compactTes3 ? 1.25f : 2.0f) * scale);
            m_uiDrawList.addText(
                choiceFont, ">",
                ui::UiVec2{row.minX + (16.0f * scale), y + choiceRowPadding}, dialogueText);
        }

        const std::string number = std::to_string(i + 1) + ".";
        m_uiDrawList.addText(
            choiceFont, number,
            ui::UiVec2{row.minX + (40.0f * scale), y + choiceRowPadding},
            selected ? dialogueText : dialogueMuted);

        float choiceY = y + choiceRowPadding;
        for (const std::string& text : choiceLines[i]) {
            m_uiDrawList.addText(
                choiceFont, text,
                ui::UiVec2{row.minX + choiceIndent + (30.0f * scale), choiceY},
                selected ? dialogueText : dialogueMuted);
            choiceY += choiceLineHeight;
        }
        y += rowHeight;
    }

    // The count is stated when the list is clipped, because a reply the player
    // cannot see is a reply they do not know to scroll to -- and the numbers on
    // the visible rows are the TRUE indices, so "7." appearing first is only
    // legible next to "of 9".
    std::string footer;
    if (m_navDriving) {
        footer = choiceCount == 0
            ? std::string("B  end conversation")
            : std::string("D-pad  select     A  choose     B  leave");
    } else {
        footer = choiceCount == 0
            ? std::string("Esc  end conversation")
            : std::string("Up/Down  select     Enter  choose     Esc  leave");
    }
    if (choicesClipped) {
        const std::string range = " (" + std::to_string(firstChoice + 1u) + "-" +
            std::to_string(firstChoice + visibleChoices) + " of " +
            std::to_string(choiceCount) + ")";
        footer = m_navDriving
            ? "D-pad  select" + range + "     A  choose     B  leave"
            : "Up/Down  select" + range + "     Enter  choose     Esc  leave";
    }
    m_uiDrawList.addText(
        footerFont, footer,
        ui::UiVec2{panel.minX + ((panelWidth - footerFont.measureText(footer)) * 0.5f),
                   panel.maxY - padding - footerFont.lineHeightPx() + (10.0f * scale)},
        dialogueMuted);
}

void BethesdaApp::drawHud() {
    // ODAI_FNV_NOHUD=1 draws the world and nothing else. A screenshot meant to
    // show the RENDERER has its own subject, and the Pip-Boy chrome, the key
    // hints and the debug readouts all sit on top of it.
    static const bool s_noHud = [] {
        const char* env = std::getenv("ODAI_FNV_NOHUD");
        return env != nullptr && env[0] != '0';
    }();
    if (s_noHud) {
        // The software cursor is not part of the HUD -- GameApp draws it after
        // this returns -- and it is drawn wherever the desktop happened to
        // leave the pointer, so it lands in the corner of the capture.
        setCursorVisible(false);
        if (m_doorTransitionAlpha > 0.0f) {
            int width = 0;
            int height = 0;
            framebufferSize(width, height);
            m_uiDrawList.addRectFilled(
                ui::UiRect{0.0f, 0.0f, static_cast<float>(width), static_cast<float>(height)},
                ui::UiColor{0.0f, 0.0f, 0.0f, m_doorTransitionAlpha});
        }
        return;
    }
    drawPipBoyHud();
    drawPauseMenu();
    drawGiftMenu();
    drawTes3Journal();

    // Toasts last so they sit above the menu: a discovery that fires while the
    // menu is open must not be hidden behind it.
    int screenWidth = 0;
    int screenHeight = 0;
    framebufferSize(screenWidth, screenHeight);
    const ui::UiRect screen{
        0.0f, 0.0f, static_cast<float>(screenWidth), static_cast<float>(screenHeight)};
    // Journal entries are read at length. Hold transient world notifications
    // until it closes rather than covering the tabs or the authored prose.
    if (!m_tes3JournalOpen) {
        m_toasts.draw(m_uiDrawList, m_uiFont, screen, contentScale());
    }
    // The banner draws its title in the display face and its subtitle in the
    // body face -- the size jump between them is what makes the location name
    // read as the headline rather than as another line of HUD text. Falls back
    // to the body face if loadFonts was not given a display size.
    if (!m_menuOpen && !m_tes3JournalOpen && m_talkingActor < 0) {
        const ui::Font& bannerFont = m_uiFontDisplay.valid() ? m_uiFontDisplay : m_uiFont;
        m_banner.draw(m_uiDrawList, bannerFont, m_uiFont, screen, contentScale());
    }
    if (m_doorTransitionAlpha > 0.0f) {
        m_uiDrawList.addRectFilled(
            screen, ui::UiColor{0.0f, 0.0f, 0.0f, m_doorTransitionAlpha});
    }
}

void BethesdaApp::drawTes3Journal() {
    if (!m_tes3JournalOpen || m_tes3JournalPanel == nullptr) return;
    if (m_bethesdaSession.tes3().journal().chronology().size() !=
        m_tes3JournalSyncedVisits) {
        syncTes3JournalPanel();
    }
    int screenWidth = 0;
    int screenHeight = 0;
    framebufferSize(screenWidth, screenHeight);
    const float scale = contentScale();
    const ui::UiRect screen{
        0.0f, 0.0f, static_cast<float>(screenWidth), static_cast<float>(screenHeight)};
    m_uiDrawList.addRectFilled(screen, {0.0f, 0.0f, 0.0f, 0.68f});
    const float marginX = std::max(48.0f * scale, screen.width() * 0.12f);
    const float marginY = std::max(44.0f * scale, screen.height() * 0.09f);
    const ui::UiRect panel{
        marginX, marginY, screen.maxX - marginX, screen.maxY - marginY};
    m_tes3JournalPanel->rebuild(panel, scale);
    m_tes3JournalPanel->draw(m_uiDrawList);
    const char* footer = m_navDriving
        ? "D-pad  Browse     LB / RB  Pages     A  Pin quest     B  Close"
        : "Up / Down  Browse     Left / Right  Pages     Enter  Pin     J / Esc  Close";
    const ui::Font& footerFont = m_tes3JournalFont.valid()
        ? m_tes3JournalFont : m_uiFont;
    m_uiDrawList.addText(footerFont, footer,
        {panel.minX + (42.0f * scale), panel.maxY - (54.0f * scale)},
        ui::UiColor{0.20f, 0.14f, 0.09f, 0.92f});
}

void BethesdaApp::drawGiftMenu() {
    if (!m_bethesdaSessionConfigured ||
        m_bethesdaSession.giftMenuRequests().empty()) {
        return;
    }
    const bethesda::GiftMenuRequestState& request =
        m_bethesdaSession.giftMenuRequests().front();
    const bethesda::ObjectId sourceId =
        request.playerGives ? request.player : request.actor;
    const bethesda::RuntimeObject* source =
        m_bethesdaSession.world().find(sourceId);

    int screenWidth = 0;
    int screenHeight = 0;
    framebufferSize(screenWidth, screenHeight);
    const float scale = contentScale();
    const float width = std::min(720.0f * scale, static_cast<float>(screenWidth) * 0.72f);
    const float padding = 28.0f * scale;
    const float lineHeight = m_uiFont.lineHeightPx() + (10.0f * scale);
    const std::size_t itemCount = source == nullptr ? 0u : source->inventory.size();
    const std::size_t visibleCount = std::min<std::size_t>(itemCount, 10u);
    const float height = padding * 2.0f + lineHeight *
        static_cast<float>(visibleCount + 3u);
    const ui::UiRect screen{0.0f, 0.0f,
        static_cast<float>(screenWidth), static_cast<float>(screenHeight)};
    const ui::UiRect panel{
        (screen.maxX - width) * 0.5f,
        (screen.maxY - height) * 0.5f,
        (screen.maxX + width) * 0.5f,
        (screen.maxY + height) * 0.5f};
    m_uiDrawList.addRectFilled(screen, ui::UiColor{0.0f, 0.0f, 0.0f, 0.62f});
    m_uiDrawList.addDropShadow(
        panel, ui::UiColor{0.0f, 0.0f, 0.0f, 0.65f}, 18.0f * scale, 8.0f * scale);
    m_uiDrawList.addRoundRectFilled(panel, kPipPanelSolid, 8.0f * scale);
    m_uiDrawList.addRoundRect(panel, kPipGreen, 8.0f * scale, 2.0f * scale);

    const std::string title = request.playerGives ? "GIVE ITEMS" : "TAKE SUPPLIES";
    m_uiDrawList.addText(m_uiFontBold.valid() ? m_uiFontBold : m_uiFont, title,
        ui::UiVec2{panel.minX + padding, panel.minY + padding}, kPipGreen);
    float y = panel.minY + padding + lineHeight;
    if (source == nullptr || source->inventory.empty()) {
        m_uiDrawList.addText(m_uiFont, "No available items",
            ui::UiVec2{panel.minX + padding, y}, kPipGreenDim);
        y += lineHeight;
    } else {
        std::size_t first = 0u;
        if (source->inventory.size() > visibleCount &&
            m_giftMenuChoice >= static_cast<int>(visibleCount)) {
            first = static_cast<std::size_t>(m_giftMenuChoice) - visibleCount + 1u;
        }
        for (std::size_t row = 0u; row < visibleCount; ++row) {
            const std::size_t index = first + row;
            const bethesda::InventoryEntry& entry = source->inventory[index];
            const bool selected = static_cast<int>(index) == m_giftMenuChoice;
            const ui::UiRect rowRect{
                panel.minX + (padding * 0.5f), y,
                panel.maxX - (padding * 0.5f), y + lineHeight};
            if (selected) {
                m_uiDrawList.addRoundRectFilled(
                    rowRect,
                    ui::UiColor{kPipGreen.r, kPipGreen.g, kPipGreen.b, 0.20f},
                    3.0f * scale);
            }
            const std::string label = (selected ? "> " : "  ") +
                entry.item.toString() + "  x" + std::to_string(entry.count);
            m_uiDrawList.addText(m_uiFont, label,
                ui::UiVec2{panel.minX + padding, y + (4.0f * scale)},
                selected ? kPipGreen : kPipGreenDim);
            y += lineHeight;
        }
    }
    if (request.filterList.valid()) {
        m_uiDrawList.addText(m_uiFont,
            "Compatibility: this menu requires FLST filtering before transfer",
            ui::UiVec2{panel.minX + padding, panel.maxY - padding - lineHeight},
            kPipGreenDim);
    }
    const std::string footer =
        "Up/Down  select     Enter  transfer one     Esc  done";
    m_uiDrawList.addText(m_uiFont, footer,
        ui::UiVec2{panel.minX + padding, panel.maxY - padding}, kPipGreenDim);
}

void BethesdaApp::updateDebugStats() {
    // Gated on the panel actually being up: this formats a couple of dozen
    // strings and it would otherwise do that every frame for nobody.
    if (!m_renderer.isDebugUiVisible()) {
        return;
    }
    auto number = [](auto value, int decimals = 0) {
        char text[48];
        if (decimals > 0) {
            std::snprintf(text, sizeof(text), "%.*f", decimals, static_cast<double>(value));
        } else {
            std::snprintf(text, sizeof(text), "%lld", static_cast<long long>(value));
        }
        return std::string{text};
    };

    std::vector<render::DebugStatGroup> groups;

    if (m_streamer != nullptr) {
        const importer::fnv::CellStreamerStats stats = m_streamer->stats();
        render::DebugStatGroup streaming{"Cell Streaming", {}};
        streaming.rows.push_back({"Resident cells", number(stats.residentChunks)});
        streaming.rows.push_back({"Loading", number(stats.residency.loadingCount)});
        streaming.rows.push_back({"Loaded / evicted",
            number(stats.scenesLoaded) + " / " + number(stats.residency.evictions)});
        // Wasted loads are the honest cost of prediction: a cell read to
        // completion and then thrown away because the player turned. A number
        // that climbs with distance travelled means the lead time is too long.
        streaming.rows.push_back({"Wasted / unavailable",
            number(stats.residency.wastedLoads) + " / " + number(stats.residency.unavailableCells)});
        streaming.rows.push_back({"Failed loads", number(stats.loadFailures)});
        streaming.rows.push_back({"Empty cells", number(stats.emptyScenes)});
        streaming.rows.push_back({"", ""});
        // Apply is main-thread time; build is worker time. Only the first is
        // felt as a hitch, which is why they are reported separately.
        streaming.rows.push_back({"Apply ms (last / worst)",
            number(stats.lastApplyMs, 2) + " / " + number(stats.worstApplyMs, 2)});
        streaming.rows.push_back({"Build ms (last / worst)",
            number(stats.lastBuildMs, 2) + " / " + number(stats.worstBuildMs, 2)});
        streaming.rows.push_back({"Cell cache hit / miss",
            number(stats.cacheHits) + " / " + number(stats.cacheMisses)});
        streaming.rows.push_back({"Cache load ms", number(stats.lastCacheLoadMs, 2)});
        if (stats.cacheWriteFailures > 0) {
            streaming.rows.push_back({"Cache write failures", number(stats.cacheWriteFailures)});
        }
        groups.push_back(std::move(streaming));
    }

    render::DebugStatGroup world{"World", {}};
    world.rows.push_back({"Camera",
        number(m_cameraX, 0) + ", " + number(m_cameraY, 0) + ", " + number(m_cameraZ, 0)});
    world.rows.push_back({"Cell",
        number(std::floor(m_cameraX / 4096.0f)) + ", " + number(std::floor(m_cameraZ / 4096.0f))});
    world.rows.push_back({"Hour", number(m_timeOfDayHours, 2)});
    const importer::fnv::FalloutWeatherRecord* weather =
        m_activeWeatherFormId != 0u ? m_weatherTables.findWeather(m_activeWeatherFormId) : nullptr;
    world.rows.push_back({"Weather", weather != nullptr ? weather->editorId : "<none>"});
    world.rows.push_back({"Actors", number(m_actors.size())});
    if (m_bethesdaSessionConfigured) {
        world.rows.push_back({"Runtime objects", number(m_bethesdaSession.world().size())});
        world.rows.push_back({"Runtime actors",
            number(m_bethesdaSession.world().orderedActorIds().size())});
    }
    world.rows.push_back({"Regions discovered", number(m_discoveredRegions.size())});
    groups.push_back(std::move(world));

    m_renderer.setDebugStatGroups(std::move(groups));
}

bool BethesdaApp::captureWarmupComplete() {
    if (m_framesRendered <= m_captureWarmupFrames) {
        return false;
    }
    if (!m_captureRoutePreloadActive && m_framesRendered >= m_captureWarmupFrameCeiling) {
        return true;
    }
    // Streaming is wall-clock work on other threads, so a frame count cannot
    // stand in for it. This mattered the moment frame capture got 28x faster:
    // the same 60 warm-up frames went from over a minute of wall time to about
    // a second, and the opening of every capture became a half-built town.
    if (m_streamer != nullptr && !m_streamer->isStreamingIdle()) {
        return false;
    }
    if (m_captureSkyrimLodBoundsValid &&
        (!m_captureSkyrimTerrainLodFrozen || !m_captureSkyrimObjectLodFrozen)) {
        return false;
    }
    if (!m_captureUploadsReady) {
        m_captureUploadsReady = m_renderer.waitForImportedSceneUploads();
    }
    return m_captureUploadsReady;
}

bool BethesdaApp::beginCaptureAudio() {
    if (!m_captureAudioRequested || m_captureAudio.isOpen()) {
        return true;
    }
    if (!m_audio.offlineMixActive() || m_audio.mixSampleRate() != 48000u ||
        m_audio.mixChannels() != 2u) {
        VOX_LOGE("newvegas")
            << "--capture-audio requires the miniaudio offline mixer at 48 kHz stereo";
        return false;
    }

    m_captureTemporaryVideoPath = m_captureVideoPath + ".video.tmp.mp4";
    m_captureTemporaryAudioPath = m_captureVideoPath + ".audio.tmp.wav";
    if (!m_captureAudio.open(m_captureTemporaryAudioPath, m_audio.mixSampleRate(),
                             static_cast<std::uint16_t>(m_audio.mixChannels()))) {
        return false;
    }

    // The weather beds use fades up to three seconds. Advance the device-free
    // graph while discarding four seconds so frame zero starts from the stable
    // authored ambience instead of fading up from silence.
    constexpr std::uint64_t kPrimeFrames = 4u * 48000u;
    constexpr std::uint64_t kPrimeChunkFrames = 4096u;
    m_capturePcm.resize(kPrimeChunkFrames * m_audio.mixChannels());
    std::uint64_t remaining = kPrimeFrames;
    while (remaining > 0u) {
        const std::uint64_t frames = std::min(remaining, kPrimeChunkFrames);
        std::span<float> chunk(m_capturePcm.data(), frames * m_audio.mixChannels());
        if (!m_audio.renderOfflineFrames(chunk, frames)) {
            VOX_LOGE("newvegas") << "offline audio prime failed";
            return false;
        }
        remaining -= frames;
    }
    m_captureAudioPrimed = true;
    m_captureAudioFramesWritten = 0u;
    VOX_LOGI("newvegas") << "offline capture audio primed for 4 seconds";
    return true;
}

bool BethesdaApp::writeCaptureAudioFrame() {
    if (!m_captureAudioRequested) {
        return true;
    }
    if (!m_captureAudioPrimed || !m_captureAudio.isOpen()) {
        return false;
    }
    const std::uint64_t fps = static_cast<std::uint64_t>(m_captureVideoFps + 0.5f);
    if (fps == 0u) {
        return false;
    }
    // Match the integer frame rate handed to ffmpeg. Taking the difference
    // between cumulative rational targets also works when sampleRate/fps is not
    // integral, without drifting by a rounded sample every video frame.
    const std::uint64_t targetFrames =
        (static_cast<std::uint64_t>(m_captureWritten + 1) * m_audio.mixSampleRate()) / fps;
    const std::uint64_t frames = targetFrames - m_captureAudioFramesWritten;
    m_capturePcm.resize(frames * m_audio.mixChannels());
    if (!m_audio.renderOfflineFrames(m_capturePcm, frames) ||
        !m_captureAudio.write(m_capturePcm)) {
        return false;
    }
    m_captureAudioFramesWritten = targetFrames;
    return true;
}

bool BethesdaApp::finishCaptureVideo() {
    const bool videoOk = m_captureVideo.close();
    if (!m_captureAudioRequested) {
        return videoOk;
    }
    const bool audioOk = m_captureAudio.close();
    if (!videoOk || !audioOk) {
        return false;
    }
    if (!render::muxVideoAndAudio(m_captureTemporaryVideoPath.string(),
                                  m_captureTemporaryAudioPath.string(), m_captureVideoPath)) {
        VOX_LOGE("newvegas") << "capture mux failed; retained temporary video and WAV";
        return false;
    }
    std::error_code removeError;
    std::filesystem::remove(m_captureTemporaryVideoPath, removeError);
    removeError.clear();
    std::filesystem::remove(m_captureTemporaryAudioPath, removeError);
    return true;
}

void BethesdaApp::onRender(float /*deltaSeconds*/) {
    // Before beginFrameDraw: the backend consumes the pending pose while
    // recording this frame, so setting it afterwards would always be a frame
    // late -- invisible on a still bind pose, and a lag on an animated one.
    if (m_characterMode) {
        updateCharacterPose();
    }
    updateDebugStats();
    beginFrameDraw();
    drawHud();

    render::CameraPose camera{};
    camera.x = m_cameraX;
    camera.y = m_cameraY;
    camera.z = m_cameraZ;
    camera.yawDegrees = m_yawDegrees;
    camera.pitchDegrees = m_pitchDegrees;
    camera.fovDegrees = m_cameraFovDegrees;

    const float yawRadians = m_yawDegrees * (kPi / 180.0f);
    const float pitchRadians = m_pitchDegrees * (kPi / 180.0f);
    const float cosPitch = std::cos(pitchRadians);
    m_audio.setListenerTransform(audio::ListenerTransform{
        {m_cameraX, m_cameraY, m_cameraZ},
        {std::cos(yawRadians) * cosPitch, std::sin(pitchRadians),
         std::sin(yawRadians) * cosPitch},
        {0.0f, 1.0f, 0.0f}});
    submitFrame(camera);

    // Capture AFTER submitFrame: the capture reads the last presented image, so
    // running it before there is one gets nothing. The warm-up frames matter
    // too -- auto-exposure adapts over several frames, so a capture on frame 0
    // shows a mid-adaptation image rather than what the scene settles at.
    if (!m_screenshotPath.empty()) {
        ++m_framesRendered;
        // A FRAME COUNT IS NOT A STREAMING WAIT, and this path was still
        // counting frames alone long after the video path stopped. On an empty
        // scene a frame costs almost nothing, so several hundred of them elapse
        // in about a second while the cells are still arriving -- and the shot
        // comes out as bare sky over the skybox's ground colour, which reads as
        // "the geometry stopped rendering" rather than "the shot was early".
        // Two hours went into the wrong half of that sentence.
        //
        // Same ceiling as the video path, so a worldspace that never settles
        // still produces a file instead of hanging.
        const bool streamReady = m_streamer == nullptr || m_streamer->isStreamingIdle();
        const bool lodReady = !m_streamIsSkyrim || m_streamer == nullptr ||
            (m_skyrimTerrainLodTileValid && m_skyrimObjectLodTileValid);
        if (streamReady && lodReady && !m_captureUploadsReady) {
            m_captureUploadsReady = m_renderer.waitForImportedSceneUploads();
        }
        const bool settled = m_framesRendered >= m_captureWarmupFrameCeiling ||
            (streamReady && lodReady && m_captureUploadsReady);
        if (m_framesRendered >= m_screenshotWarmupFrames && settled) {
            if (!m_renderer.captureFrameToFile(m_screenshotPath)) {
                VOX_LOGE("newvegas") << "screenshot capture failed";
            }
            glfwSetWindowShouldClose(m_window, GLFW_TRUE);
        }
    }

    // Frame-sequence recording, straight into an encoder. Same cadence and same
    // warm-up as the stills path below, but the frames never land on disk.
    if (!m_captureVideoPath.empty() && m_captureWritten < m_captureFrames) {
        ++m_framesRendered;
        if (captureWarmupComplete()) {
            m_captureStarted = true;
            if (!beginCaptureAudio()) {
                glfwSetWindowShouldClose(m_window, GLFW_TRUE);
                return;
            }
            std::uint32_t width = 0;
            std::uint32_t height = 0;
            bool ok = m_renderer.captureFrameRgb(m_captureRgb, width, height);
            if (ok && !m_captureVideo.isOpen()) {
                // Opened on the FIRST captured frame rather than up front,
                // because the swapchain extent is what it is: ODAI_WINDOW_SIZE
                // is a request the window manager is free to ignore, and ffmpeg
                // needs the real number baked into its input description.
                const std::string writerPath = m_captureAudioRequested
                    ? m_captureTemporaryVideoPath.string()
                    : m_captureVideoPath;
                ok = m_captureVideo.open(writerPath, width, height,
                                         static_cast<int>(m_captureVideoFps + 0.5f));
            }
            if (ok) {
                ok = m_captureVideo.writeFrame(m_captureRgb);
            }
            if (ok) {
                ok = writeCaptureAudioFrame();
            }
            if (!ok) {
                VOX_LOGE("newvegas") << "video capture failed at frame " << m_captureWritten;
                m_captureVideo.close();
                m_captureAudio.close();
                glfwSetWindowShouldClose(m_window, GLFW_TRUE);
                return;
            }
            ++m_captureWritten;
            if ((m_captureWritten % 120) == 0) {
                VOX_LOGI("newvegas")
                    << "captured " << m_captureWritten << "/" << m_captureFrames << " frames";
            }
            if (m_captureWritten >= m_captureFrames) {
                // Closed HERE, not in the destructor: pclose is where a failed
                // encode reports itself, and an hour of rendering should not
                // discover that during teardown.
                if (!finishCaptureVideo()) {
                    VOX_LOGE("newvegas") << "video capture finalization failed";
                }
                glfwSetWindowShouldClose(m_window, GLFW_TRUE);
            }
        }
    }

    // The older stills path. One PPM per frame, numbered, for ffmpeg to stitch
    // afterwards. Kept because a still sequence is what a frame-by-frame
    // comparison wants -- but for a recording, prefer --capture-video above.
    if (!m_captureDirectory.empty() && m_captureWritten < m_captureFrames) {
        ++m_framesRendered;
        if (captureWarmupComplete()) {
            m_captureStarted = true;
            char leaf[32] = {};
            std::snprintf(leaf, sizeof(leaf), "/frame_%05d.ppm", m_captureWritten);
            if (!m_renderer.captureFrameToFile(m_captureDirectory + leaf)) {
                VOX_LOGE("newvegas") << "sequence capture failed at frame " << m_captureWritten;
                glfwSetWindowShouldClose(m_window, GLFW_TRUE);
            }
            ++m_captureWritten;
            // Progress, because a 900-frame capture is minutes of silence
            // otherwise and a stalled one looks identical to a slow one.
            if ((m_captureWritten % 60) == 0) {
                VOX_LOGI("newvegas")
                    << "captured " << m_captureWritten << "/" << m_captureFrames << " frames";
            }
            if (m_captureWritten >= m_captureFrames) {
                glfwSetWindowShouldClose(m_window, GLFW_TRUE);
            }
        }
    }
}

}  // namespace odai::games::newvegas
