#pragma once

#include "anim/animation_clip.h"
#include "anim/skeleton.h"

#include <filesystem>
#include <string>
#include <vector>

// JSON loading for Skeleton and AnimationClip. Kept separate from the
// type/sampler headers so nlohmann::json never leaks into them, mirroring
// dialogue/dialogue_io.h's split from dialogue_types.h/dialogue_runtime.h.
namespace odai::anim {

struct SkeletonLoadResult {
    Skeleton skeleton;
    // Non-fatal: a malformed bone entry is skipped and recorded here; a bone
    // whose parent name isn't found (including forward references — bones
    // must be declared parent-before-child) is also recorded here.
    std::vector<std::string> errors;
    [[nodiscard]] bool ok() const { return errors.empty(); }
};

SkeletonLoadResult loadSkeletonFromJson(const std::string& jsonText,
                                        const std::string& sourceLabel = "<memory>");
SkeletonLoadResult loadSkeletonFromFile(const std::filesystem::path& path);

struct AnimationClipLoadResult {
    AnimationClip clip;
    std::vector<std::string> errors;
    [[nodiscard]] bool ok() const { return errors.empty(); }
};

// skeleton is used only to resolve each track's bone name into a bone index;
// the loaded clip itself has no further dependency on it.
AnimationClipLoadResult loadAnimationClipFromJson(const std::string& jsonText, const Skeleton& skeleton,
                                                   const std::string& sourceLabel = "<memory>");
AnimationClipLoadResult loadAnimationClipFromFile(const std::filesystem::path& path, const Skeleton& skeleton);

}  // namespace odai::anim
