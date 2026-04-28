#pragma once

#include <filesystem>
#include <memory>

namespace odai::audio {

struct MusicSettings {
    bool enabled = true;
    float volume = 0.35f;
};

class SoundEngine {
public:
    SoundEngine();
    ~SoundEngine();

    SoundEngine(const SoundEngine&) = delete;
    SoundEngine& operator=(const SoundEngine&) = delete;
    SoundEngine(SoundEngine&&) = delete;
    SoundEngine& operator=(SoundEngine&&) = delete;

    [[nodiscard]] bool init(const MusicSettings& settings);
    void shutdown();
    void update();

    void setMusicSettings(const MusicSettings& settings);
    [[nodiscard]] MusicSettings musicSettings() const;

    [[nodiscard]] bool playMorrowindMusic(const std::filesystem::path& dataFilesPath);
    [[nodiscard]] bool playMusicDirectory(const std::filesystem::path& musicDirectory);
    void stopMusic();

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace odai::audio
