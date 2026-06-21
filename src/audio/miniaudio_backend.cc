#include "audio/audio_backend.h"

#include "core/log.h"

#include <miniaudio.h>

#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

// Real backend: a thin adapter over miniaudio's high-level engine API.
//   - one ma_sound_group per category bus (Music / Ambient / Ui); Master is the
//     engine endpoint volume.
//   - playSound  -> ma_engine_play_sound (engine owns + reaps each instance, so
//     overlapping one-shots like rapid UI clicks just work).
//   - startAmbient/playMusic -> persistent ma_sound with a fade-in; the previous
//     one is faded out and retired (crossfade). Music streams from disk.
// miniaudio mixes and applies fades on its own device thread; update(dt) only
// reaps faded-out sounds we own so they don't leak.
namespace odai::audio {
namespace {

int volumeIndex(SoundCategory c) { return static_cast<int>(c); }

ma_uint64 fadeMs(float seconds) {
    if (seconds <= 0.0f) return 0;
    return static_cast<ma_uint64>(seconds * 1000.0f);
}

class MiniaudioBackend final : public AudioBackend {
public:
    static std::unique_ptr<AudioBackend> create(const AudioConfig& cfg);
    ~MiniaudioBackend() override;

    void update(float dt) override;

    SoundHandle loadSound(const std::filesystem::path& file, SoundCategory category) override;
    MusicHandle loadMusic(const std::filesystem::path& file) override;

    void playSound(SoundHandle clip) override;
    void startAmbient(SoundHandle loop, float fadeSeconds) override;
    void stopAmbient(float fadeSeconds) override;
    void playMusic(MusicHandle track, float fadeSeconds, bool loop) override;
    void stopMusic(float fadeSeconds) override;

    void setMasterVolume(float v) override;
    void setCategoryVolume(SoundCategory c, float v) override;
    float categoryVolume(SoundCategory c) const override;
    void setMuted(bool muted) override;
    bool muted() const override;
    bool deviceActive() const override { return m_active; }

private:
    MiniaudioBackend() = default;

    void applyVolumes();
    ma_sound_group* groupFor(SoundCategory c);
    std::unique_ptr<ma_sound> createSound(const std::string& path, SoundCategory category,
                                          ma_uint32 flags, bool loop, float fadeSeconds);
    void retire(std::unique_ptr<ma_sound> sound, float fadeSeconds);

    struct SoundDef {
        std::string path;
        SoundCategory category = SoundCategory::Ui;
    };
    struct FadingSound {
        std::unique_ptr<ma_sound> sound;
        float remaining = 0.0f;
    };

    ma_engine m_engine{};
    bool m_engineInitialized = false;
    bool m_active = false;

    std::array<ma_sound_group, 3> m_groups{};                 // [0]=Music [1]=Ambient [2]=Ui
    std::array<bool, 3> m_groupInit{false, false, false};

    std::array<float, kSoundCategoryCount> m_volumes{1.0f, 0.6f, 0.5f, 0.8f};
    bool m_muted = false;

    std::vector<SoundDef> m_sounds;  // SFX + ambient clips, addressed by SoundHandle
    std::vector<SoundDef> m_music;   // streamed tracks, addressed by MusicHandle

    std::unique_ptr<ma_sound> m_ambient;
    std::unique_ptr<ma_sound> m_musicCurrent;
    std::vector<FadingSound> m_fading;  // sounds fading out, reaped in update()
};

std::unique_ptr<AudioBackend> MiniaudioBackend::create(const AudioConfig& cfg) {
    std::unique_ptr<MiniaudioBackend> backend(new MiniaudioBackend());

    ma_engine_config engineConfig = ma_engine_config_init();
    if (ma_engine_init(&engineConfig, &backend->m_engine) != MA_SUCCESS) {
        VOX_LOGW("audio") << "ma_engine_init failed; audio will run silent";
        return nullptr;  // caller falls back to the null backend
    }
    backend->m_engineInitialized = true;

    for (int i = 0; i < 3; ++i) {
        if (ma_sound_group_init(&backend->m_engine, 0, nullptr, &backend->m_groups[i]) == MA_SUCCESS) {
            backend->m_groupInit[i] = true;
        } else {
            VOX_LOGW("audio") << "sound group " << i << " init failed; routing to master";
        }
    }

    backend->m_volumes = {cfg.masterVolume, cfg.musicVolume, cfg.ambientVolume, cfg.uiVolume};
    backend->m_muted = cfg.muted;
    backend->applyVolumes();
    backend->m_active = true;

    VOX_LOGI("audio") << "miniaudio engine initialized ("
                      << ma_engine_get_sample_rate(&backend->m_engine) << " Hz)";
    return backend;
}

MiniaudioBackend::~MiniaudioBackend() {
    // Uninit sounds before groups before the engine so nothing references a
    // destroyed parent while the device thread is still being torn down.
    if (m_ambient) ma_sound_uninit(m_ambient.get());
    if (m_musicCurrent) ma_sound_uninit(m_musicCurrent.get());
    for (FadingSound& f : m_fading) {
        if (f.sound) ma_sound_uninit(f.sound.get());
    }
    m_fading.clear();
    for (int i = 0; i < 3; ++i) {
        if (m_groupInit[i]) ma_sound_group_uninit(&m_groups[i]);
    }
    if (m_engineInitialized) ma_engine_uninit(&m_engine);
}

void MiniaudioBackend::applyVolumes() {
    ma_engine_set_volume(&m_engine, m_muted ? 0.0f : m_volumes[0]);
    for (int i = 0; i < 3; ++i) {
        if (m_groupInit[i]) ma_sound_group_set_volume(&m_groups[i], m_volumes[i + 1]);
    }
}

ma_sound_group* MiniaudioBackend::groupFor(SoundCategory c) {
    if (c == SoundCategory::Master) return nullptr;  // engine endpoint
    const int idx = static_cast<int>(c) - 1;         // Music=0, Ambient=1, Ui=2
    if (idx < 0 || idx >= 3 || !m_groupInit[idx]) return nullptr;
    return &m_groups[idx];
}

std::unique_ptr<ma_sound> MiniaudioBackend::createSound(const std::string& path, SoundCategory category,
                                                        ma_uint32 flags, bool loop, float fadeSeconds) {
    auto sound = std::make_unique<ma_sound>();
    if (ma_sound_init_from_file(&m_engine, path.c_str(), flags, groupFor(category), nullptr, sound.get()) !=
        MA_SUCCESS) {
        VOX_LOGW("audio") << "failed to init sound: " << path;
        return nullptr;
    }
    if (loop) ma_sound_set_looping(sound.get(), MA_TRUE);
    if (fadeSeconds > 0.0f) {
        ma_sound_set_fade_in_milliseconds(sound.get(), 0.0f, 1.0f, fadeMs(fadeSeconds));
    }
    ma_sound_start(sound.get());
    return sound;
}

void MiniaudioBackend::retire(std::unique_ptr<ma_sound> sound, float fadeSeconds) {
    if (!sound) return;
    if (fadeSeconds > 0.0f) {
        // -1 start volume means "fade from the current volume".
        ma_sound_set_fade_in_milliseconds(sound.get(), -1.0f, 0.0f, fadeMs(fadeSeconds));
        m_fading.push_back(FadingSound{std::move(sound), fadeSeconds});
    } else {
        ma_sound_uninit(sound.get());
    }
}

void MiniaudioBackend::update(float dt) {
    for (auto it = m_fading.begin(); it != m_fading.end();) {
        it->remaining -= dt;
        if (it->remaining <= 0.0f) {
            ma_sound_uninit(it->sound.get());
            it = m_fading.erase(it);
        } else {
            ++it;
        }
    }
}

SoundHandle MiniaudioBackend::loadSound(const std::filesystem::path& file, SoundCategory category) {
    std::error_code ec;
    if (!std::filesystem::exists(file, ec)) {
        VOX_LOGW("audio") << "sound file missing: " << file.string();
        return {};
    }
    m_sounds.push_back(SoundDef{file.string(), category});
    return SoundHandle{static_cast<std::uint32_t>(m_sounds.size())};
}

MusicHandle MiniaudioBackend::loadMusic(const std::filesystem::path& file) {
    std::error_code ec;
    if (!std::filesystem::exists(file, ec)) {
        VOX_LOGW("audio") << "music file missing: " << file.string();
        return {};
    }
    m_music.push_back(SoundDef{file.string(), SoundCategory::Music});
    return MusicHandle{static_cast<std::uint32_t>(m_music.size())};
}

void MiniaudioBackend::playSound(SoundHandle clip) {
    if (!clip.valid() || clip.id > m_sounds.size()) return;
    const SoundDef& def = m_sounds[clip.id - 1];
    // The engine's resource manager caches the decoded data by path, so repeated
    // plays don't re-decode; it also owns and reaps the inline sound instance.
    ma_engine_play_sound(&m_engine, def.path.c_str(), groupFor(def.category));
}

void MiniaudioBackend::startAmbient(SoundHandle loop, float fadeSeconds) {
    if (!loop.valid() || loop.id > m_sounds.size()) return;
    if (m_ambient) retire(std::move(m_ambient), fadeSeconds);
    const SoundDef& def = m_sounds[loop.id - 1];
    m_ambient = createSound(def.path, SoundCategory::Ambient, MA_SOUND_FLAG_DECODE, /*loop=*/true, fadeSeconds);
}

void MiniaudioBackend::stopAmbient(float fadeSeconds) {
    if (m_ambient) retire(std::move(m_ambient), fadeSeconds);
}

void MiniaudioBackend::playMusic(MusicHandle track, float fadeSeconds, bool loop) {
    if (!track.valid() || track.id > m_music.size()) return;
    if (m_musicCurrent) retire(std::move(m_musicCurrent), fadeSeconds);
    const SoundDef& def = m_music[track.id - 1];
    m_musicCurrent = createSound(def.path, SoundCategory::Music, MA_SOUND_FLAG_STREAM, loop, fadeSeconds);
}

void MiniaudioBackend::stopMusic(float fadeSeconds) {
    if (m_musicCurrent) retire(std::move(m_musicCurrent), fadeSeconds);
}

void MiniaudioBackend::setMasterVolume(float v) {
    m_volumes[0] = v;
    if (!m_muted) ma_engine_set_volume(&m_engine, v);
}

void MiniaudioBackend::setCategoryVolume(SoundCategory c, float v) {
    if (c == SoundCategory::Master) {
        setMasterVolume(v);
        return;
    }
    m_volumes[volumeIndex(c)] = v;
    const int idx = static_cast<int>(c) - 1;
    if (idx >= 0 && idx < 3 && m_groupInit[idx]) ma_sound_group_set_volume(&m_groups[idx], v);
}

float MiniaudioBackend::categoryVolume(SoundCategory c) const {
    return m_volumes[volumeIndex(c)];
}

void MiniaudioBackend::setMuted(bool muted) {
    m_muted = muted;
    ma_engine_set_volume(&m_engine, muted ? 0.0f : m_volumes[0]);
}

bool MiniaudioBackend::muted() const {
    return m_muted;
}

}  // namespace

std::unique_ptr<AudioBackend> createMiniaudioBackend(const AudioConfig& cfg) {
    return MiniaudioBackend::create(cfg);
}

}  // namespace odai::audio
