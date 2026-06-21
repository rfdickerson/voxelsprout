#pragma once

#include <cstdint>

// Lightweight, dependency-free value types shared by the audio facade and its
// backends. No miniaudio, no platform headers — safe to include anywhere.
namespace odai::audio {

// Mixing buses. Master scales everything; the other three are per-category
// volume groups so the UI can offer separate sliders later.
enum class SoundCategory : std::uint8_t {
    Master = 0,
    Music = 1,
    Ambient = 2,
    Ui = 3,
};
inline constexpr int kSoundCategoryCount = 4;

// Opaque handle to a loaded one-shot / ambient clip. id 0 is always invalid.
struct SoundHandle {
    std::uint32_t id = 0;
    [[nodiscard]] bool valid() const { return id != 0; }
};

// Opaque handle to a loaded (streamed) music track. id 0 is always invalid.
struct MusicHandle {
    std::uint32_t id = 0;
    [[nodiscard]] bool valid() const { return id != 0; }
};

// Initial volumes (0..1) and mute state, typically sourced from the app config.
struct AudioConfig {
    float masterVolume = 1.0f;
    float musicVolume = 0.6f;
    float ambientVolume = 0.5f;
    float uiVolume = 0.8f;
    bool muted = false;
};

}  // namespace odai::audio
