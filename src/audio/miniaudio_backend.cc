#include "audio/audio_backend.h"

#include "core/log.h"
#include "math/math.h"

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
//   - playSoundAt -> a transient owned ma_sound with 3D spatialization enabled,
//     reaped once ma_sound_at_end() reports it finished.
//   - startAmbient/startAmbientAt -> a persistent ma_sound with a fade-in in one
//     of a fixed set of ambient slots (torch + river + wind bed, etc. can all be
//     active at once); startAmbientAt additionally enables spatialization/
//     distance attenuation. playMusic uses the same fade-in pattern for a single
//     current track. In every case the previous occupant of a slot is faded out
//     and retired (crossfade), never hard-stopped. Music streams from disk.
// miniaudio mixes/spatializes and applies fades on its own device thread;
// update(dt) only reaps faded-out/finished sounds we own so they don't leak.
namespace odai::audio {
namespace {

int volumeIndex(SoundCategory c) { return static_cast<int>(c); }

ma_uint64 fadeMs(float seconds) {
    if (seconds <= 0.0f) return 0;
    return static_cast<ma_uint64>(seconds * 1000.0f);
}

// Packs a slot index (0-based) and a generation counter into one handle id so a
// handle from a since-reused slot is detected as stale rather than silently
// touching the wrong sound. id 0 (slotIndex+1 == 0) is reserved for "invalid".
std::uint32_t packAmbientHandle(int slotIndex, std::uint32_t generation) {
    return (generation << 16) | static_cast<std::uint32_t>(slotIndex + 1);
}
int ambientHandleSlotIndex(AmbientHandle handle) {
    return static_cast<int>(handle.id & 0xFFFFu) - 1;
}
std::uint32_t ambientHandleGeneration(AmbientHandle handle) {
    return handle.id >> 16;
}

class MiniaudioBackend final : public AudioBackend {
public:
    static std::unique_ptr<AudioBackend> create(const AudioConfig& cfg);
    ~MiniaudioBackend() override;

    void update(float dt) override;

    SoundHandle loadSound(const std::filesystem::path& file, SoundCategory category) override;
    MusicHandle loadMusic(const std::filesystem::path& file) override;

    void playSound(SoundHandle clip) override;
    void playSoundAt(SoundHandle clip, const odai::math::Vector3& position,
                     const AttenuationParams& attenuation) override;

    AmbientHandle startAmbient(SoundHandle loop, float fadeSeconds) override;
    AmbientHandle startAmbientAt(SoundHandle loop, const odai::math::Vector3& position,
                                 const AttenuationParams& attenuation, float fadeSeconds) override;
    void stopAmbient(AmbientHandle handle, float fadeSeconds) override;
    void setAmbientPosition(AmbientHandle handle, const odai::math::Vector3& position) override;

    void setListenerTransform(const ListenerTransform& listener) override;

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
    void applyPositional(ma_sound& sound, const odai::math::Vector3& position,
                         const AttenuationParams& attenuation);
    AmbientHandle startAmbientSlot(SoundHandle loop, bool positional, const odai::math::Vector3& position,
                                   const AttenuationParams& attenuation, float fadeSeconds);
    // Returns nullptr if the handle is invalid, stale (slot reused since issue), or inactive.
    struct AmbientSlot;
    AmbientSlot* resolveAmbientSlot(AmbientHandle handle);

    struct SoundDef {
        std::string path;
        SoundCategory category = SoundCategory::Ui;
    };
    struct FadingSound {
        std::unique_ptr<ma_sound> sound;
        float remaining = 0.0f;
    };
    struct AmbientSlot {
        std::unique_ptr<ma_sound> sound;
        bool positional = false;
        bool active = false;
        std::uint32_t generation = 0;
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

    std::array<AmbientSlot, kMaxAmbientSlots> m_ambientSlots{};   // torch + river + wind bed, etc.
    std::vector<std::unique_ptr<ma_sound>> m_positionalOneShots;  // playSoundAt instances, reaped in update()
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
    for (AmbientSlot& slot : m_ambientSlots) {
        if (slot.sound) ma_sound_uninit(slot.sound.get());
    }
    for (std::unique_ptr<ma_sound>& sound : m_positionalOneShots) {
        if (sound) ma_sound_uninit(sound.get());
    }
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
    for (auto it = m_positionalOneShots.begin(); it != m_positionalOneShots.end();) {
        if (*it && ma_sound_at_end(it->get())) {
            ma_sound_uninit(it->get());
            it = m_positionalOneShots.erase(it);
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

void MiniaudioBackend::applyPositional(ma_sound& sound, const odai::math::Vector3& position,
                                       const AttenuationParams& attenuation) {
    ma_sound_set_spatialization_enabled(&sound, MA_TRUE);
    ma_sound_set_position(&sound, position.x, position.y, position.z);
    ma_sound_set_attenuation_model(&sound, ma_attenuation_model_linear);
    ma_sound_set_min_distance(&sound, attenuation.minDistance);
    ma_sound_set_max_distance(&sound, attenuation.maxDistance);
    ma_sound_set_rolloff(&sound, attenuation.rolloff);
}

void MiniaudioBackend::playSoundAt(SoundHandle clip, const odai::math::Vector3& position,
                                   const AttenuationParams& attenuation) {
    if (!clip.valid() || clip.id > m_sounds.size()) return;
    const SoundDef& def = m_sounds[clip.id - 1];
    std::unique_ptr<ma_sound> sound =
        createSound(def.path, def.category, MA_SOUND_FLAG_DECODE, /*loop=*/false, /*fadeSeconds=*/0.0f);
    if (!sound) return;
    applyPositional(*sound, position, attenuation);
    m_positionalOneShots.push_back(std::move(sound));
}

MiniaudioBackend::AmbientSlot* MiniaudioBackend::resolveAmbientSlot(AmbientHandle handle) {
    if (!handle.valid()) return nullptr;
    const int index = ambientHandleSlotIndex(handle);
    if (index < 0 || index >= kMaxAmbientSlots) return nullptr;
    AmbientSlot& slot = m_ambientSlots[static_cast<std::size_t>(index)];
    if (!slot.active || slot.generation != ambientHandleGeneration(handle)) return nullptr;
    return &slot;
}

AmbientHandle MiniaudioBackend::startAmbientSlot(SoundHandle loop, bool positional,
                                                 const odai::math::Vector3& position,
                                                 const AttenuationParams& attenuation, float fadeSeconds) {
    if (!loop.valid() || loop.id > m_sounds.size()) return {};

    int freeIndex = -1;
    for (int i = 0; i < kMaxAmbientSlots; ++i) {
        if (!m_ambientSlots[static_cast<std::size_t>(i)].active) {
            freeIndex = i;
            break;
        }
    }
    if (freeIndex < 0) {
        VOX_LOGW("audio") << "no free ambient slot (max " << kMaxAmbientSlots << "); dropping start request";
        return {};
    }

    const SoundDef& def = m_sounds[loop.id - 1];
    std::unique_ptr<ma_sound> sound =
        createSound(def.path, def.category, MA_SOUND_FLAG_DECODE, /*loop=*/true, fadeSeconds);
    if (!sound) return {};
    if (positional) {
        applyPositional(*sound, position, attenuation);
    } else {
        ma_sound_set_spatialization_enabled(sound.get(), MA_FALSE);
    }

    AmbientSlot& slot = m_ambientSlots[static_cast<std::size_t>(freeIndex)];
    slot.sound = std::move(sound);
    slot.positional = positional;
    slot.active = true;
    ++slot.generation;
    return AmbientHandle{packAmbientHandle(freeIndex, slot.generation)};
}

AmbientHandle MiniaudioBackend::startAmbient(SoundHandle loop, float fadeSeconds) {
    return startAmbientSlot(loop, /*positional=*/false, odai::math::Vector3{}, AttenuationParams{}, fadeSeconds);
}

AmbientHandle MiniaudioBackend::startAmbientAt(SoundHandle loop, const odai::math::Vector3& position,
                                               const AttenuationParams& attenuation, float fadeSeconds) {
    return startAmbientSlot(loop, /*positional=*/true, position, attenuation, fadeSeconds);
}

void MiniaudioBackend::stopAmbient(AmbientHandle handle, float fadeSeconds) {
    AmbientSlot* slot = resolveAmbientSlot(handle);
    if (!slot) return;
    slot->active = false;
    retire(std::move(slot->sound), fadeSeconds);
}

void MiniaudioBackend::setAmbientPosition(AmbientHandle handle, const odai::math::Vector3& position) {
    AmbientSlot* slot = resolveAmbientSlot(handle);
    if (!slot || !slot->positional || !slot->sound) return;
    ma_sound_set_position(slot->sound.get(), position.x, position.y, position.z);
}

void MiniaudioBackend::setListenerTransform(const ListenerTransform& listener) {
    ma_engine_listener_set_position(&m_engine, 0, listener.position.x, listener.position.y, listener.position.z);
    ma_engine_listener_set_direction(&m_engine, 0, listener.forward.x, listener.forward.y, listener.forward.z);
    ma_engine_listener_set_world_up(&m_engine, 0, listener.up.x, listener.up.y, listener.up.z);
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
