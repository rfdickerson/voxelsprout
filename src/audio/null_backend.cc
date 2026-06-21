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
    void startAmbient(SoundHandle /*loop*/, float /*fadeSeconds*/) override {}
    void stopAmbient(float /*fadeSeconds*/) override {}
    void playMusic(MusicHandle /*track*/, float /*fadeSeconds*/, bool /*loop*/) override {}
    void stopMusic(float /*fadeSeconds*/) override {}

    void setMasterVolume(float v) override { m_volumes[0] = v; }
    void setCategoryVolume(SoundCategory c, float v) override { m_volumes[volumeIndex(c)] = v; }
    float categoryVolume(SoundCategory c) const override { return m_volumes[volumeIndex(c)]; }
    void setMuted(bool muted) override { m_muted = muted; }
    bool muted() const override { return m_muted; }
    bool deviceActive() const override { return false; }

private:
    std::array<float, kSoundCategoryCount> m_volumes{1.0f, 0.6f, 0.5f, 0.8f};
    bool m_muted = false;
};

}  // namespace

std::unique_ptr<AudioBackend> createNullBackend(const AudioConfig& cfg) {
    return std::make_unique<NullBackend>(cfg);
}

}  // namespace odai::audio
