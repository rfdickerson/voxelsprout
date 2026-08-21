#include "audio/audio_backend.h"

#include <array>

// Silent backend. Always compiled, with no third-party dependency, so every
// environment (no device, CI, WSL, audio lib built without miniaudio) still
// gets a valid Audio. It stores volume/mute state so those values round-trip
// through config persistence even though nothing is ever played.
namespace odai::audio {
namespace {

// SoundCategory values map directly to volume slots: Master=0, Music=1,
// Ambient=2, Ui=3.
int volumeIndex(SoundCategory c) { return static_cast<int>(c); }

class NullBackend final : public AudioBackend {
public:
    explicit NullBackend(const AudioConfig& cfg)
        : m_volumes{cfg.masterVolume, cfg.musicVolume, cfg.ambientVolume, cfg.uiVolume},
          m_muted(cfg.muted) {}

    void update(float /*dt*/) override {}

    SoundHandle loadSound(const std::filesystem::path& /*file*/, SoundCategory /*category*/) override {
        return {};
    }
    MusicHandle loadMusic(const std::filesystem::path& /*file*/) override { return {}; }

    void playSound(SoundHandle /*clip*/) override {}
    void playSoundAt(SoundHandle /*clip*/, const odai::math::Vector3& /*position*/,
                     const AttenuationParams& /*attenuation*/) override {}

    AmbientHandle startAmbient(SoundHandle /*loop*/, float /*fadeSeconds*/) override { return {}; }
    AmbientHandle startAmbientAt(SoundHandle /*loop*/, const odai::math::Vector3& /*position*/,
                                 const AttenuationParams& /*attenuation*/, float /*fadeSeconds*/) override {
        return {};
    }
    void stopAmbient(AmbientHandle /*handle*/, float /*fadeSeconds*/) override {}
    void setAmbientPosition(AmbientHandle /*handle*/, const odai::math::Vector3& /*position*/) override {}

    void setListenerTransform(const ListenerTransform& /*listener*/) override {}

    void playMusic(MusicHandle /*track*/, float /*fadeSeconds*/, bool /*loop*/) override {}
    void stopMusic(float /*fadeSeconds*/) override {}

    void setMasterVolume(float v) override { m_volumes[0] = v; }
    void setCategoryVolume(SoundCategory c, float v) override { m_volumes[volumeIndex(c)] = v; }
    float categoryVolume(SoundCategory c) const override { return m_volumes[volumeIndex(c)]; }
    void setMuted(bool muted) override { m_muted = muted; }
    bool muted() const override { return m_muted; }
    bool deviceActive() const override { return false; }
    bool offlineMixActive() const override { return false; }
    std::uint32_t mixSampleRate() const override { return 0u; }
    std::uint32_t mixChannels() const override { return 0u; }
    bool renderOfflineFrames(std::span<float>, std::uint64_t) override { return false; }

private:
    std::array<float, kSoundCategoryCount> m_volumes{1.0f, 0.6f, 0.5f, 0.8f};
    bool m_muted = false;
};

}  // namespace

std::unique_ptr<AudioBackend> createNullBackend(const AudioConfig& cfg) {
    return std::make_unique<NullBackend>(cfg);
}

}  // namespace odai::audio
