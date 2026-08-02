#pragma once

#include "dialogue/dialogue_context.h"

#include <filesystem>
#include <string>

// Persistence for the one piece of dialogue state that outlives a single
// conversation: MapDialogueContext's flags and per-companion approval. This
// is the "no persistence" gap docs/ROADMAP.md flags under the Dragon Age
// touchstone (Party RPG / Narrative, Tier 4 step 3) -- everything else about
// a conversation (current node, available choices) is transient and lives in
// DialogueRuntime instead. JSON, mirroring dialogue_io.cc's format rather
// than strategy_map_io.cc's binary one: this data is small, hand-inspectable
// save-slot content, not a bulk asset.
namespace odai::dialogue {

bool saveDialogueState(const MapDialogueContext& ctx, const std::filesystem::path& outputPath);
bool loadDialogueState(const std::filesystem::path& inputPath, MapDialogueContext& ctx);

// Same operations against an in-memory JSON string, for embedding dialogue
// state as one field inside a larger save-game blob instead of its own file.
std::string saveDialogueStateToJson(const MapDialogueContext& ctx);
bool loadDialogueStateFromJson(const std::string& jsonText, MapDialogueContext& ctx);

// Human-readable description of the most recent save/load failure.
[[nodiscard]] const std::string& getDialogueStateLastError();

}  // namespace odai::dialogue
