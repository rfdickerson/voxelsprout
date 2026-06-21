#pragma once

#include "audio/audio_types.h"

#include <filesystem>
#include <memory>

// Internal backend interface behind the Audio facade. Included only by the
// audio .cc files — never by a public header — so concrete backends can pull in
// miniaudio without leaking it. Two implementations exist:
//   - NullBackend     : always compiled; silent no-op; deviceActive() == false.
//   - MiniaudioBackend: compiled only when ODAI_AUDIO_HAVE_MINIAUDIO is defined.
namespace odai::audio {

class AudioBackend {
public:
    virtual ~AudioBackend() = default;

    virtual void update(float dt) = 0;

    virtual SoundHandle loadSound(const std::filesystem::path& file, SoundCategory category) = 0;
    virtual MusicHandle loadMusic(const std::filesystem::path& file) = 0;

    virtual void playSound(SoundHandle clip) = 0;
    virtual void startAmbient(SoundHandle loop, float fadeSeconds) = 0;
    virtual void stopAmbient(float fadeSeconds) = 0;
    virtual void playMusic(MusicHandle track, float fadeSeconds, bool loop) = 0;
    virtual void stopMusic(float fadeSeconds) = 0;

    virtual void setMasterVolume(float v) = 0;
    virtual void setCategoryVolume(SoundCategory c, float v) = 0;
    [[nodiscard]] virtual float categoryVolume(SoundCategory c) const = 0;
    virtual void setMuted(bool muted) = 0;
    [[nodiscard]] virtual bool muted() const = 0;
    [[nodiscard]] virtual bool deviceActive() const = 0;
};

// Always available. Silent; loaders return invalid handles. Seeded with cfg so
// volume/mute state round-trips through config persistence even with no device.
std::unique_ptr<AudioBackend> createNullBackend(const AudioConfig& cfg);

#ifdef ODAI_AUDIO_HAVE_MINIAUDIO
// Real backend. Returns nullptr if no device is present / engine init fails, so
// the caller can fall back to the null backend.
std::unique_ptr<AudioBackend> createMiniaudioBackend(const AudioConfig& cfg);
#endif

}  // namespace odai::audio
