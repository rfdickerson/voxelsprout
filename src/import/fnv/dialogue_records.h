#pragma once

// Builds a branching dialogue tree for ONE speaker out of a Fallout plugin's
// DIAL/INFO records, in the engine's own dialogue::DialogueTree form so the
// existing runtime (dialogue/dialogue_runtime.h) and panel
// (ui/widgets/dialogue_panel.h) drive it with no Fallout-specific code.
//
// HOW A LINE IS ATTRIBUTED TO A SPEAKER, which is the non-obvious part: an INFO
// record carries no speaker field. Fallout attributes it with a CTDA condition
// -- function index 72, GetIsID, whose first parameter is the actor's formID.
// That was derived rather than looked up: odai_newvegas_probe --dialogue
// histograms every CTDA function index whose parameter matches the wanted
// actor, and across Victor's records function 72 accounts for 179 of 181 hits.
//
// Structure of the graph, measured on Victor (CREA 0x103DFD, 173 INFOs / 184
// responses / 137 links):
//
//   DIAL  is a TOPIC. Its FULL is the text the PLAYER says to raise it.
//   INFO  is a RESPONSE, living in its topic's child group. NAM1 holds the
//         spoken line; a record may carry several, separated by NEXT.
//   TCLT  on an INFO lists the topics that become available after that line --
//         i.e. the player's choices.
//
// So a node is an INFO, its choices are its TCLT topics, and following a choice
// means finding the INFO for this speaker under the chosen topic.
//
// CONDITIONS ARE NOT EVALUATED. Every INFO is gated by CTDA conditions on quest
// stage, faction, karma and prior topics, and this engine models none of that.
// They are read only far enough to identify the speaker; the rest are ignored,
// which means the tree offers lines that the real game would gate. That is a
// deliberate limitation of importing dialogue into an engine with no quest
// system, not an oversight -- see kDialogueConditionsIgnored.

#include "dialogue/dialogue_types.h"

#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace odai::importer::fnv {

// CTDA function index whose param1 is an actor formID and which Fallout uses to
// bind an INFO to its speaker. Derived by measurement, not documentation.
inline constexpr std::uint32_t kCtdaFunctionGetIsId = 72u;

// Stated in one place so callers can surface it rather than quietly imply the
// conversation is faithful.
inline constexpr bool kDialogueConditionsIgnored = true;

struct DialogueImportStats {
    std::uint32_t topicsSeen = 0;
    std::uint32_t infosForSpeaker = 0;
    std::uint32_t responsesConcatenated = 0;
    std::uint32_t choiceLinks = 0;
    std::uint32_t danglingLinks = 0;  // TCLT topics with no INFO for this speaker
};

// Where a speaker stands in the world, from its ACRE (creature) or ACHR (NPC)
// reference. Bethesda coordinates, straight out of the reference's DATA -- the
// caller converts to engine space the same way cell_builder does.
//
// Separate from the cell pipeline on purpose: that path places STAT records and
// would need creature base records, model resolution and an opt-in filter to
// avoid populating the whole Mojave with frozen actors (there is no .kf reader,
// so every one of them would stand in bind pose). One named speaker does not
// need any of that.
struct SpeakerPlacement {
    std::uint32_t referenceFormId = 0;
    // The actor BASE this reference instances. A caller that also populates the
    // area generically needs it to avoid rendering this actor twice, once from
    // each system -- excluding by editor-ID string would work until two mods
    // disagreed about the name.
    std::uint32_t baseFormId = 0;
    std::uint32_t cellFormId = 0;
    float position[3] = {};
    float rotationRadians[3] = {};
    bool found = false;
    // A creature's MODL names its SKELETON, not a mesh -- the geometry is the
    // NIFZ list of body parts, which are skinned to that skeleton and stored
    // beside it. Guessing a mesh filename instead gets you
    // securitron_static.nif, which is a GROUP prop of several deactivated
    // robots (one shape spanning 401x367 units) rather than one actor.
    std::string skeletonPath;                 // e.g. creatures\NVSecuritron\Skeleton.nif
    std::vector<std::string> bodyPartPaths;   // resolved beside the skeleton
};

bool findSpeakerPlacement(
    const std::filesystem::path& pluginPath,
    const std::string& speakerEditorId,
    SpeakerPlacement& outPlacement,
    std::string& outError);

// Finds the CREA/NPC_ whose EDID equals `speakerEditorId` (case-insensitive)
// and builds its tree. Returns false when the speaker or any dialogue for it
// cannot be found; `outError` says which.
//
// `outTree.startNode` is a GREETING response when one exists, because that is
// what the game opens a conversation with.
//
// TWO plugin walks: one to turn the EditorID into a formID, one to read the
// dialogue. Prefer buildSpeakerDialogueTrees when the caller already knows the
// formIDs, which anything populating a settlement does.
bool buildSpeakerDialogueTree(
    const std::filesystem::path& pluginPath,
    const std::string& speakerEditorId,
    odai::dialogue::DialogueTree& outTree,
    DialogueImportStats& outStats,
    std::string& outError);

struct SpeakerDialogueRequest {
    // The actor BASE, which is what a CTDA names. A placement's own formID
    // matches nothing.
    std::uint32_t baseFormId = 0;
    // Shown as the node's speaker and used as the tree's id. Usually the
    // actor's FULL, falling back to its EditorID.
    std::string displayName;
};

// Builds a tree per speaker in ONE walk over the plugin's DIAL/INFO records.
//
// The reason this exists rather than a loop over the single-speaker version:
// that one walks the whole plugin per speaker, and the dialogue pass alone is
// ~75 ms against FalloutNV.esm. A town of ten actors would spend most of a
// second re-reading the same records to answer the same question ten times.
// Attribution is by formID, which every caller populating a settlement already
// has, so nothing is lost by skipping the EditorID lookup.
//
// Speakers with no dialogue are simply absent from `outTrees` -- most actors in
// a cell have nothing to say, and that is not an error.
bool buildSpeakerDialogueTrees(
    const std::filesystem::path& pluginPath,
    const std::vector<SpeakerDialogueRequest>& speakers,
    std::unordered_map<std::uint32_t, odai::dialogue::DialogueTree>& outTrees,
    std::unordered_map<std::uint32_t, DialogueImportStats>& outStats,
    std::string& outError);

}  // namespace odai::importer::fnv
