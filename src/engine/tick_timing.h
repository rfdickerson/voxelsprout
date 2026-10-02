#pragma once

namespace odai::engine {

// Nested wall-clock timings for one game update. The streamer and cell fields
// are subsets of streamingMs; do not sum every field to reconstruct tickMs.
struct TickTiming {
    float sessionMs = 0.0f;
    float actorsMs = 0.0f;
    float streamingMs = 0.0f;
    float streamerUpdateMs = 0.0f;
    float streamerApplyMs = 0.0f;
    float streamerUploadMs = 0.0f;
    float streamerCallbacksMs = 0.0f;
    float streamerEvictionMs = 0.0f;
    float streamerRendererRemoveMs = 0.0f;
    float streamerEvictionCallbacksMs = 0.0f;
    float evictionCollisionMs = 0.0f;
    float evictionNavigationMs = 0.0f;
    float evictionDoorsMs = 0.0f;
    float cellRegistrationMs = 0.0f;
    float runtimeObjectsMs = 0.0f;
    float gameplayCellMs = 0.0f;
    float gameplayIoMs = 0.0f;
    float gameplayCompileMs = 0.0f;
    float gameplayPublishMs = 0.0f;
    float gameplayAnchorMs = 0.0f;
    float gameplayUpsertMs = 0.0f;
    float physicsInstallMs = 0.0f;
    float collisionWorldMs = 0.0f;
    float navigationMs = 0.0f;
    float broadPhaseMs = 0.0f;
    unsigned residentCells = 0;
};

} // namespace odai::engine
