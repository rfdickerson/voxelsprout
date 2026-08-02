#pragma once

#include "audio/audio_types.h"

#include <filesystem>
#include <memory>

// Public audio facade. Mirrors odai::render::Renderer: a narrow, move-only
// class hiding the real backend behind a PIMPL pointer so no third-party audio
// header (miniaudio) leaks across this boundary.
//
// Threading: every method is expected to be called from the main thread only.
// The backend's mixing runs on its own device-callback thread internally; the
// underlying engine API is thread-safe for play/stop/volume against that thread,
// so the facade adds no locking.
//
// Degradation: init() never hard-fails. If no audio device is available (or the
// audio library was built without a backend), it falls back to a silent backend
// and every call becomes a safe no-op. Calls on invalid handles are no-ops too.
namespace odai::audio {

class AudioBackend;  // defined only in the .cc files

class Audio {
public:
    Audio();
    ~Audio();
    Audio(Audio&&) noexcept;
    Audio& operator=(Audio&&) noexcept;
    Audio(const Audio&) = delete;
    Audio& operator=(const Audio&) = delete;

    // Brings up the backend. Returns true even when running silent.
    bool init(const AudioConfig& cfg);
    // Per-frame pump (pass the same clamped dt the app loop computes). Light
    // bookkeeping only — the backend mixes/fades on its own thread.
    void update(float dt);
    // Tears down the backend deterministically (joins the device thread). Safe
    // to call more than once; further calls become no-ops.
    void shutdown();

    // Fully decoded into memory; for short SFX and ambient loops. Invalid on failure.
    [[nodiscard]] SoundHandle loadSound(const std::filesystem::path& file, SoundCategory category);
    // Streamed from disk; for long music tracks. Invalid on failure.
    [[nodiscard]] MusicHandle loadMusic(const std::filesystem::path& file);

    void playSound(SoundHandle clip);                                            // fire-and-forget one-shot
    void playSoundAt(SoundHandle clip, const odai::math::Vector3& position,       // spatialized one-shot
                     const AttenuationParams& attenuation = {});

    // Ambient loops occupy one of a fixed set of concurrent slots (see kMaxAmbientSlots
    // in audio_types.h) — a torch, a river, and a wind bed can all run at once. startAmbient
    // starts a global (unpositioned) bed; startAmbientAt starts one distance-attenuated
    // against the listener. Both crossfade in on their own; stopAmbient crossfades one out.
    [[nodiscard]] AmbientHandle startAmbient(SoundHandle loop, float fadeSeconds = 1.0f);
    [[nodiscard]] AmbientHandle startAmbientAt(SoundHandle loop, const odai::math::Vector3& position,
                                               const AttenuationParams& attenuation = {},
                                               float fadeSeconds = 1.0f);
    void stopAmbient(AmbientHandle handle, float fadeSeconds = 1.0f);
    void setAmbientPosition(AmbientHandle handle, const odai::math::Vector3& position);

    // Updates the 3D listener (typically the player/camera) that playSoundAt and positional
    // ambient loops are attenuated against. Call once per frame from game code.
    void setListenerTransform(const ListenerTransform& listener);

    void playMusic(MusicHandle track, float fadeSeconds = 2.0f, bool loop = true);  // crossfades
    void stopMusic(float fadeSeconds = 2.0f);

    void setMasterVolume(float v);                       // 0..1 (clamped)
    void setCategoryVolume(SoundCategory c, float v);    // 0..1 (clamped)
    [[nodiscard]] float categoryVolume(SoundCategory c) const;
    void setMuted(bool muted);
    [[nodiscard]] bool muted() const;
    [[nodiscard]] bool deviceActive() const;             // false when running silent

private:
    std::unique_ptr<AudioBackend> m_backend;
};

}  // namespace odai::audio
