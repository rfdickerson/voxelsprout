#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace odai::render {

struct DialogueFontConfig {
    bool enabled = false;
    std::filesystem::path fontPath;
    float sizePixels = 18.0f;
    std::string sourceLabel;
};

struct DialogueFontResolveInput {
    std::string requestedFont;
    std::optional<float> requestedSizePixels;
    std::vector<std::filesystem::path> fontDirectories;
};

[[nodiscard]] DialogueFontConfig resolveDialogueFontConfig(const DialogueFontResolveInput& input);

} // namespace odai::render
