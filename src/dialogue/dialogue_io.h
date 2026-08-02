#pragma once

#include "dialogue/dialogue_types.h"

#include <filesystem>
#include <string>
#include <vector>

// JSON loading for DialogueTree (dialogue_types.h). Kept separate from the
// type/runtime headers so nlohmann::json never leaks into them, mirroring
// content/content_database.h's split from content/content_loader.cc.
namespace odai::dialogue {

struct DialogueLoadResult {
    DialogueTree tree;
    // Non-fatal: a malformed node/choice/effect is skipped and recorded here
    // rather than aborting the whole load. A dangling target/autoNext id is
    // also recorded here so broken authoring surfaces at load time.
    std::vector<std::string> errors;
    [[nodiscard]] bool ok() const { return errors.empty(); }
};

DialogueLoadResult loadDialogueTreeFromJson(const std::string& jsonText,
                                             const std::string& sourceLabel = "<memory>");
DialogueLoadResult loadDialogueTreeFromFile(const std::filesystem::path& path);

}  // namespace odai::dialogue
