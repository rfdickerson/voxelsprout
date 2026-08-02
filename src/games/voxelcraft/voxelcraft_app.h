#pragma once

#include "audio/audio_types.h"
#include "engine/game_app.h"
#include "games/voxelcraft/voxelcraft_player.h"
#include "games/voxelcraft/voxelcraft_streaming.h"
#include "render/renderer_types.h"
#include "world/clipmap_index.h"
#include "world/world.h"

#include <cstdint>
#include <filesystem>
#include <vector>

// VoxelCraft: a single-player first-person voxel sandbox (walk, break/place blocks,
// hitch-free chunk streaming). See docs at the plan this game was built from --
// src/games/voxelcraft/voxelcraft_player.h/.cc and voxelcraft_streaming.h/.cc are kept free
// of GameApp/UI/GLFW coupling so they're reusable from a future headless multiplayer server
// unchanged.
namespace odai::games::voxelcraft {

class VoxelCraftApp : public engine::GameApp {
protected:
    bool onInit() override;
    void onTick(float dt) override;
    void onRender(float dt) override;
    void onShutdown() override;

private:
    void sampleInput();
    void updateChunkStreaming();
    void refreshStreamingWindow(bool forceRendererUpload);
    void updateHotbarAndInventoryInput();
    void updateBlockInteraction();
    void tickAutosave(float dt);
    void tickFootstepAudio(float dxHorizontal, float dzHorizontal);
    void saveWorld();
    void drawHud();

    [[nodiscard]] render::CameraPose interpolatedCameraPose(float alpha) const;

    PlayerState m_player{};
    PlayerState m_playerPrevious{};
    PlayerInputSnapshot m_pendingInput{};
    VoxelBreakProgress m_breakProgress{};

    world::World m_world;
    world::ChunkClipmapIndex m_clipmapIndex;
    world::ClipmapConfig m_appliedClipmapConfig{};
    bool m_hasAppliedClipmapConfig = false;
    ChunkStreamingPipeline m_streamingPipeline;
    std::vector<world::World::ChunkKey> m_lastDesiredChunkKeys;

    render::GameplayUiState m_gameplayUiState{};
    std::vector<std::size_t> m_visibleChunkIndices;

    double m_simAccumulatorSeconds = 0.0;

    std::filesystem::path m_worldPath{"voxelcraft_world.vxw"};
    bool m_worldDirty = false;
    float m_worldAutosaveElapsedSeconds = 0.0f;

    bool m_mouseCaptured = true;
    bool m_wasEscapeDown = false;
    bool m_wasInventoryKeyDown = false;
    bool m_wasSaveKeyDown = false;
    std::uint32_t m_hotbarKeyPrevMask = 0;

    // Ambient loops, started once in onInit() at fixed world positions -- see
    // docs/GAME_API.md §4 for the Audio facade this pulls from.
    odai::audio::SoundHandle   m_ambientTorchSound;
    odai::audio::SoundHandle   m_ambientRiverSound;
    odai::audio::SoundHandle   m_ambientWindSound;
    odai::audio::AmbientHandle m_ambientTorchHandle;
    odai::audio::AmbientHandle m_ambientRiverHandle;
    odai::audio::AmbientHandle m_ambientWindHandle;

    odai::audio::SoundHandle m_sfxFootstep;
    odai::audio::SoundHandle m_sfxBlockBreak;
    float m_footstepDistanceAccumulator = 0.0f;  // horizontal voxels moved since the last footstep
};

} // namespace odai::games::voxelcraft
