#include "audio/sound_engine.h"

#include "core/log.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <mfapi.h>
#include <mfidl.h>
#include <mfreadwrite.h>
#include <mmreg.h>
#include <wrl/client.h>
#include <xaudio2.h>
#include <windows.h>
#endif

namespace odai::audio {
namespace {

float clampVolume(float volume) {
    return std::clamp(volume, 0.0f, 1.0f);
}

bool isSupportedMusicFile(const std::filesystem::path& path) {
    std::string extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return extension == ".mp3" || extension == ".wav" || extension == ".wma";
}

} // namespace

class SoundEngine::Impl {
public:
    [[nodiscard]] bool init(const MusicSettings& settings);
    void shutdown();
    void update();
    void setMusicSettings(const MusicSettings& settings);
    [[nodiscard]] MusicSettings musicSettings() const { return m_settings; }
    [[nodiscard]] bool playMorrowindMusic(const std::filesystem::path& dataFilesPath);
    [[nodiscard]] bool playMusicDirectory(const std::filesystem::path& musicDirectory);
    void stopMusic();

private:
    MusicSettings m_settings{};
    std::vector<std::filesystem::path> m_playlist;
    std::size_t m_nextTrackIndex = 0;
    bool m_initialized = false;
    bool m_musicPlaying = false;

#ifdef _WIN32
    struct DecodedAudio {
        [[nodiscard]] WAVEFORMATEX* format() {
            return reinterpret_cast<WAVEFORMATEX*>(formatBytes.data());
        }
        [[nodiscard]] const WAVEFORMATEX* format() const {
            return reinterpret_cast<const WAVEFORMATEX*>(formatBytes.data());
        }
        std::vector<std::uint8_t> formatBytes;
        std::vector<std::uint8_t> pcm;
    };

    [[nodiscard]] bool decodeAudioFile(const std::filesystem::path& path, DecodedAudio& outAudio) const;
    [[nodiscard]] bool playDecodedAudio(DecodedAudio audio);
    void playNextTrack();
    [[nodiscard]] static std::wstring widePath(const std::filesystem::path& path);

    Microsoft::WRL::ComPtr<IXAudio2> m_xaudio;
    IXAudio2MasteringVoice* m_masterVoice = nullptr;
    IXAudio2SourceVoice* m_musicVoice = nullptr;
    DecodedAudio m_currentMusic;
    bool m_comInitialized = false;
    bool m_mediaFoundationStarted = false;
#else
    void playNextTrack() {}
#endif
};

bool SoundEngine::Impl::init(const MusicSettings& settings) {
    m_settings = settings;
    m_settings.volume = clampVolume(m_settings.volume);
    if (!m_settings.enabled) {
        VOX_LOGI("audio") << "music disabled";
        m_initialized = true;
        return true;
    }

#ifdef _WIN32
    const HRESULT coResult = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    if (SUCCEEDED(coResult)) {
        m_comInitialized = true;
    } else if (coResult != RPC_E_CHANGED_MODE) {
        VOX_LOGW("audio") << "CoInitializeEx failed: 0x" << std::hex << static_cast<unsigned long>(coResult);
        return false;
    }

    HRESULT result = MFStartup(MF_VERSION);
    if (FAILED(result)) {
        VOX_LOGW("audio") << "MFStartup failed: 0x" << std::hex << static_cast<unsigned long>(result);
        shutdown();
        return false;
    }
    m_mediaFoundationStarted = true;

    result = XAudio2Create(m_xaudio.GetAddressOf(), 0, XAUDIO2_DEFAULT_PROCESSOR);
    if (FAILED(result)) {
        VOX_LOGW("audio") << "XAudio2Create failed: 0x" << std::hex << static_cast<unsigned long>(result);
        shutdown();
        return false;
    }

    result = m_xaudio->CreateMasteringVoice(&m_masterVoice);
    if (FAILED(result)) {
        VOX_LOGW("audio") << "CreateMasteringVoice failed: 0x" << std::hex << static_cast<unsigned long>(result);
        shutdown();
        return false;
    }

    m_initialized = true;
    VOX_LOGI("audio") << "sound engine initialized";
    return true;
#else
    VOX_LOGW("audio") << "audio playback is not implemented on this platform";
    m_initialized = true;
    return true;
#endif
}

void SoundEngine::Impl::shutdown() {
    stopMusic();
#ifdef _WIN32
    if (m_masterVoice != nullptr) {
        m_masterVoice->DestroyVoice();
        m_masterVoice = nullptr;
    }
    m_xaudio.Reset();
    if (m_mediaFoundationStarted) {
        MFShutdown();
        m_mediaFoundationStarted = false;
    }
    if (m_comInitialized) {
        CoUninitialize();
        m_comInitialized = false;
    }
#endif
    m_playlist.clear();
    m_initialized = false;
}

void SoundEngine::Impl::update() {
    if (!m_initialized || !m_settings.enabled || m_playlist.empty() || !m_musicPlaying) {
        return;
    }
#ifdef _WIN32
    if (m_musicVoice == nullptr) {
        playNextTrack();
        return;
    }

    XAUDIO2_VOICE_STATE state{};
    m_musicVoice->GetState(&state);
    if (state.BuffersQueued == 0) {
        playNextTrack();
    }
#endif
}

void SoundEngine::Impl::setMusicSettings(const MusicSettings& settings) {
    m_settings = settings;
    m_settings.volume = clampVolume(m_settings.volume);
#ifdef _WIN32
    if (m_musicVoice != nullptr) {
        m_musicVoice->SetVolume(m_settings.enabled ? m_settings.volume : 0.0f);
    }
#endif
    if (!m_settings.enabled) {
        stopMusic();
    }
}

bool SoundEngine::Impl::playMorrowindMusic(const std::filesystem::path& dataFilesPath) {
    return playMusicDirectory(dataFilesPath / "Music");
}

bool SoundEngine::Impl::playMusicDirectory(const std::filesystem::path& musicDirectory) {
    if (!m_initialized || !m_settings.enabled) {
        return false;
    }
    if (!std::filesystem::exists(musicDirectory)) {
        VOX_LOGW("audio") << "music directory missing: " << musicDirectory.string();
        return false;
    }

    std::vector<std::filesystem::path> playlist;
    std::error_code error;
    for (const std::filesystem::directory_entry& entry :
         std::filesystem::recursive_directory_iterator(musicDirectory, error)) {
        if (error) {
            VOX_LOGW("audio") << "music directory scan stopped: " << error.message();
            break;
        }
        if (entry.is_regular_file() && isSupportedMusicFile(entry.path())) {
            playlist.push_back(entry.path());
        }
    }
    std::sort(playlist.begin(), playlist.end());
    if (playlist.empty()) {
        VOX_LOGW("audio") << "no supported music files found in " << musicDirectory.string();
        return false;
    }

    std::mt19937 randomEngine{std::random_device{}()};
    std::shuffle(playlist.begin(), playlist.end(), randomEngine);

    stopMusic();
    m_playlist = std::move(playlist);
    m_nextTrackIndex = 0;
    m_musicPlaying = true;
    VOX_LOGI("audio") << "queued " << m_playlist.size() << " music tracks from " << musicDirectory.string();
    playNextTrack();
    return true;
}

void SoundEngine::Impl::stopMusic() {
    m_musicPlaying = false;
#ifdef _WIN32
    if (m_musicVoice != nullptr) {
        m_musicVoice->Stop(0);
        m_musicVoice->FlushSourceBuffers();
        m_musicVoice->DestroyVoice();
        m_musicVoice = nullptr;
    }
    m_currentMusic = {};
#endif
}

#ifdef _WIN32
std::wstring SoundEngine::Impl::widePath(const std::filesystem::path& path) {
    return path.wstring();
}

bool SoundEngine::Impl::decodeAudioFile(const std::filesystem::path& path, DecodedAudio& outAudio) const {
    Microsoft::WRL::ComPtr<IMFSourceReader> reader;
    HRESULT result = MFCreateSourceReaderFromURL(widePath(path).c_str(), nullptr, reader.GetAddressOf());
    if (FAILED(result)) {
        VOX_LOGW("audio") << "failed to open music file: " << path.string();
        return false;
    }

    Microsoft::WRL::ComPtr<IMFMediaType> requestedType;
    result = MFCreateMediaType(requestedType.GetAddressOf());
    if (FAILED(result)) {
        return false;
    }
    requestedType->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Audio);
    requestedType->SetGUID(MF_MT_SUBTYPE, MFAudioFormat_PCM);
    result = reader->SetCurrentMediaType(MF_SOURCE_READER_FIRST_AUDIO_STREAM, nullptr, requestedType.Get());
    if (FAILED(result)) {
        VOX_LOGW("audio") << "music decoder does not support PCM output: " << path.string();
        return false;
    }

    Microsoft::WRL::ComPtr<IMFMediaType> actualType;
    result = reader->GetCurrentMediaType(MF_SOURCE_READER_FIRST_AUDIO_STREAM, actualType.GetAddressOf());
    if (FAILED(result)) {
        return false;
    }

    WAVEFORMATEX* waveFormat = nullptr;
    UINT32 waveFormatSize = 0;
    result = MFCreateWaveFormatExFromMFMediaType(actualType.Get(), &waveFormat, &waveFormatSize);
    if (FAILED(result) || waveFormat == nullptr || waveFormatSize < sizeof(WAVEFORMATEX)) {
        if (waveFormat != nullptr) {
            CoTaskMemFree(waveFormat);
        }
        return false;
    }
    const std::uint8_t* waveFormatBytes = reinterpret_cast<const std::uint8_t*>(waveFormat);
    outAudio.formatBytes.assign(waveFormatBytes, waveFormatBytes + waveFormatSize);
    CoTaskMemFree(waveFormat);

    outAudio.pcm.clear();
    for (;;) {
        DWORD streamFlags = 0;
        Microsoft::WRL::ComPtr<IMFSample> sample;
        result = reader->ReadSample(
            MF_SOURCE_READER_FIRST_AUDIO_STREAM,
            0,
            nullptr,
            &streamFlags,
            nullptr,
            sample.GetAddressOf());
        if (FAILED(result)) {
            VOX_LOGW("audio") << "music decode failed: " << path.string();
            outAudio = {};
            return false;
        }
        if ((streamFlags & MF_SOURCE_READERF_ENDOFSTREAM) != 0) {
            break;
        }
        if (sample == nullptr) {
            continue;
        }

        Microsoft::WRL::ComPtr<IMFMediaBuffer> buffer;
        result = sample->ConvertToContiguousBuffer(buffer.GetAddressOf());
        if (FAILED(result)) {
            continue;
        }

        BYTE* data = nullptr;
        DWORD maxLength = 0;
        DWORD currentLength = 0;
        result = buffer->Lock(&data, &maxLength, &currentLength);
        if (FAILED(result)) {
            continue;
        }
        const std::size_t oldSize = outAudio.pcm.size();
        outAudio.pcm.resize(oldSize + currentLength);
        std::copy(data, data + currentLength, outAudio.pcm.begin() + static_cast<std::ptrdiff_t>(oldSize));
        buffer->Unlock();
    }

    return !outAudio.pcm.empty();
}

bool SoundEngine::Impl::playDecodedAudio(DecodedAudio audio) {
    if (audio.pcm.empty() || audio.formatBytes.empty() || m_xaudio == nullptr) {
        return false;
    }

    if (m_musicVoice != nullptr) {
        m_musicVoice->Stop(0);
        m_musicVoice->FlushSourceBuffers();
        m_musicVoice->DestroyVoice();
        m_musicVoice = nullptr;
    }

    HRESULT result = m_xaudio->CreateSourceVoice(&m_musicVoice, audio.format());
    if (FAILED(result)) {
        VOX_LOGW("audio") << "CreateSourceVoice failed: 0x" << std::hex << static_cast<unsigned long>(result);
        return false;
    }

    m_currentMusic = std::move(audio);
    XAUDIO2_BUFFER buffer{};
    buffer.AudioBytes = static_cast<UINT32>(m_currentMusic.pcm.size());
    buffer.pAudioData = m_currentMusic.pcm.data();
    buffer.Flags = XAUDIO2_END_OF_STREAM;
    result = m_musicVoice->SubmitSourceBuffer(&buffer);
    if (FAILED(result)) {
        VOX_LOGW("audio") << "SubmitSourceBuffer failed: 0x" << std::hex << static_cast<unsigned long>(result);
        m_musicVoice->DestroyVoice();
        m_musicVoice = nullptr;
        m_currentMusic = {};
        return false;
    }

    m_musicVoice->SetVolume(m_settings.enabled ? m_settings.volume : 0.0f);
    result = m_musicVoice->Start(0);
    if (FAILED(result)) {
        VOX_LOGW("audio") << "music voice start failed: 0x" << std::hex << static_cast<unsigned long>(result);
        m_musicVoice->DestroyVoice();
        m_musicVoice = nullptr;
        m_currentMusic = {};
        return false;
    }
    return true;
}

void SoundEngine::Impl::playNextTrack() {
    if (m_playlist.empty() || !m_settings.enabled) {
        stopMusic();
        return;
    }

    for (std::size_t attempt = 0; attempt < m_playlist.size(); ++attempt) {
        const std::filesystem::path path = m_playlist[m_nextTrackIndex];
        m_nextTrackIndex = (m_nextTrackIndex + 1u) % m_playlist.size();

        DecodedAudio audio;
        if (!decodeAudioFile(path, audio)) {
            continue;
        }
        if (playDecodedAudio(std::move(audio))) {
            VOX_LOGI("audio") << "playing music: " << path.filename().string();
            return;
        }
    }

    VOX_LOGW("audio") << "failed to decode any queued music track";
    stopMusic();
}
#endif

SoundEngine::SoundEngine() : m_impl(std::make_unique<Impl>()) {}
SoundEngine::~SoundEngine() = default;

bool SoundEngine::init(const MusicSettings& settings) {
    return m_impl->init(settings);
}

void SoundEngine::shutdown() {
    m_impl->shutdown();
}

void SoundEngine::update() {
    m_impl->update();
}

void SoundEngine::setMusicSettings(const MusicSettings& settings) {
    m_impl->setMusicSettings(settings);
}

MusicSettings SoundEngine::musicSettings() const {
    return m_impl->musicSettings();
}

bool SoundEngine::playMorrowindMusic(const std::filesystem::path& dataFilesPath) {
    return m_impl->playMorrowindMusic(dataFilesPath);
}

bool SoundEngine::playMusicDirectory(const std::filesystem::path& musicDirectory) {
    return m_impl->playMusicDirectory(musicDirectory);
}

void SoundEngine::stopMusic() {
    m_impl->stopMusic();
}

} // namespace odai::audio
