#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

namespace render {

enum class FrameArenaPass : uint8_t {
    Unknown = 0,
    Ssao = 1,
    Shadow = 2,
    Main = 3,
    Post = 4,
    Ui = 5
};

struct FrameArenaPassRange {
    FrameArenaPass first = FrameArenaPass::Unknown;
    FrameArenaPass last = FrameArenaPass::Unknown;
};

inline int frameArenaPassIndex(FrameArenaPass pass) {
    return static_cast<int>(pass);
}

inline bool isValidFrameArenaPassRange(const FrameArenaPassRange& range) {
    if (range.first == FrameArenaPass::Unknown || range.last == FrameArenaPass::Unknown) {
        return false;
    }
    return frameArenaPassIndex(range.first) <= frameArenaPassIndex(range.last);
}

inline bool frameArenaPassRangesOverlap(const FrameArenaPassRange& lhs, const FrameArenaPassRange& rhs) {
    return frameArenaPassIndex(lhs.first) <= frameArenaPassIndex(rhs.last) &&
           frameArenaPassIndex(rhs.first) <= frameArenaPassIndex(lhs.last);
}

inline bool canAliasWithPassRanges(
    const std::vector<FrameArenaPassRange>& existingRanges,
    const FrameArenaPassRange& candidateRange
) {
    if (!isValidFrameArenaPassRange(candidateRange)) {
        return false;
    }
    for (const FrameArenaPassRange& range : existingRanges) {
        if (frameArenaPassRangesOverlap(range, candidateRange)) {
            return false;
        }
    }
    return true;
}

inline void addAliasPassRange(
    std::vector<FrameArenaPassRange>& ranges,
    const FrameArenaPassRange& range
) {
    if (isValidFrameArenaPassRange(range)) {
        ranges.push_back(range);
    }
}

inline void acquireAliasBlockRef(uint32_t& refCount) {
    ++refCount;
}

inline bool releaseAliasBlockRef(uint32_t& refCount) {
    if (refCount == 0) {
        return true;
    }
    --refCount;
    return refCount == 0;
}

inline const char* frameArenaPassName(FrameArenaPass pass) {
    switch (pass) {
    case FrameArenaPass::Ssao:
        return "SSAO";
    case FrameArenaPass::Shadow:
        return "Shadow";
    case FrameArenaPass::Main:
        return "Main";
    case FrameArenaPass::Post:
        return "Post";
    case FrameArenaPass::Ui:
        return "UI";
    case FrameArenaPass::Unknown:
    default:
        return "Unknown";
    }
}

} // namespace render
