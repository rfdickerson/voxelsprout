#pragma once

#include "math/math.h"

#include <cstdint>

// Lightweight value types shared by the audio facade and its backends. No
// miniaudio, no platform headers — safe to include anywhere. Depends on
// math/math.h for Vector3 (positional audio), same as render/renderer_types.h
// depends on it for camera/geometry types.
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
    // Deterministic silent mode for headless tools and tests. Normal runtime
    // configuration leaves this false and uses miniaudio when available.
    bool forceNullBackend = false;
};

// World-space transform used both to position the listener each frame and to
// spatialize positional sounds against it. forward/up let the caller hand
// over a ready-made basis (games already compute this from yaw/pitch for the
// camera) instead of the backend re-deriving it.
struct ListenerTransform {
    odai::math::Vector3 position{};
    odai::math::Vector3 forward{0.0f, 0.0f, -1.0f};
    odai::math::Vector3 up{0.0f, 1.0f, 0.0f};
};

// Linear-rolloff distance attenuation for one positional sound (one-shot or
// ambient loop): full volume inside minDistance, silent past maxDistance.
struct AttenuationParams {
    float minDistance = 1.0f;
    float maxDistance = 50.0f;
    float rolloff = 1.0f;
};

// Fixed capacity for concurrently active ambient loop slots (torch + river +
// wind bed, etc.). A hard cap, not a growable container, so startAmbient/
// startAmbientAt never allocate on a hot path and capacity exhaustion is an
// explicit, observable failure rather than unbounded growth.
inline constexpr int kMaxAmbientSlots = 8;

// Opaque handle to one active ambient loop *slot* (not the loaded clip data —
// that's still SoundHandle, reusable across many ambient slots/one-shots).
// id 0 is always invalid.
struct AmbientHandle {
    std::uint32_t id = 0;
    [[nodiscard]] bool valid() const { return id != 0; }
};

}  // namespace odai::audio
