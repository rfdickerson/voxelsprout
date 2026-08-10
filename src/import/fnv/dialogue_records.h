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

// Finds the CREA/NPC_ whose EDID equals `speakerEditorId` (case-insensitive)
// and builds its tree. Returns false when the speaker or any dialogue for it
// cannot be found; `outError` says which.
//
// `outTree.startNode` is a GREETING response when one exists, because that is
// what the game opens a conversation with.
bool buildSpeakerDialogueTree(
    const std::filesystem::path& pluginPath,
    const std::string& speakerEditorId,
    odai::dialogue::DialogueTree& outTree,
    DialogueImportStats& outStats,
    std::string& outError);

}  // namespace odai::importer::fnv
