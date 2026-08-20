#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>

#include "audio/audio.h"
#include "audio/wav_writer.h"
#include "math/math.h"

// These tests explicitly select the retained null backend so they run with no
// audio device on any platform/CI. They verify the graceful-degrade contract: init never fails,
// missing files yield invalid handles, every play call is a safe no-op, and
// volume/mute state round-trips (so config persistence works without a device).

namespace {

int g_failures = 0;

odai::audio::AudioConfig silentConfig() {
    odai::audio::AudioConfig config;
    config.forceNullBackend = true;
    return config;
}

void expectTrue(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "[audio test] FAIL: " << message << '\n';
        ++g_failures;
    }
}

void expectNear(float actual, float expected, const char* message) {
    if (std::fabs(actual - expected) > 1e-4f) {
        std::cerr << "[audio test] FAIL: " << message << " (expected " << expected
                  << ", got " << actual << ")\n";
        ++g_failures;
    }
}

void testInitRunsSilent() {
    using namespace odai::audio;
    Audio audio;
    expectTrue(audio.init(silentConfig()), "init returns true even with no device");
    expectTrue(!audio.deviceActive(), "null backend reports deviceActive() == false");
}

void testMissingFilesYieldInvalidHandles() {
    using namespace odai::audio;
    Audio audio;
    audio.init(silentConfig());
    const SoundHandle sound = audio.loadSound("does/not/exist.wav", SoundCategory::Ui);
    const MusicHandle music = audio.loadMusic("does/not/exist.mp3");
    expectTrue(!sound.valid(), "loadSound on a missing file is invalid");
    expectTrue(!music.valid(), "loadMusic on a missing file is invalid");
}

void testPlayCallsAreNoOps() {
    using namespace odai::audio;
    Audio audio;
    audio.init(silentConfig());
    // Neither invalid nor (synthetic) valid handles may crash on the null backend.
    audio.playSound(SoundHandle{});
    audio.playSound(SoundHandle{42});
    const AmbientHandle a1 = audio.startAmbient(SoundHandle{}, 1.0f);
    const AmbientHandle a2 = audio.startAmbient(SoundHandle{7}, 0.5f);
    audio.stopAmbient(a1, 1.0f);
    audio.stopAmbient(a2, 1.0f);
    audio.playMusic(MusicHandle{}, 2.0f, true);
    audio.playMusic(MusicHandle{3}, 2.0f, false);
    audio.stopMusic(0.0f);
    audio.update(0.016f);
    expectTrue(true, "play/stop/update calls do not crash on the null backend");
}

void testVolumeRoundTripAndClamp() {
    using namespace odai::audio;
    Audio audio;
    audio.init(silentConfig());

    audio.setCategoryVolume(SoundCategory::Music, 0.25f);
    expectNear(audio.categoryVolume(SoundCategory::Music), 0.25f, "category volume round-trips");

    audio.setMasterVolume(0.5f);
    expectNear(audio.categoryVolume(SoundCategory::Master), 0.5f, "master volume round-trips");

    audio.setCategoryVolume(SoundCategory::Ui, 1.5f);
    expectNear(audio.categoryVolume(SoundCategory::Ui), 1.0f, "volume above 1 clamps to 1");
    audio.setCategoryVolume(SoundCategory::Ambient, -0.5f);
    expectNear(audio.categoryVolume(SoundCategory::Ambient), 0.0f, "volume below 0 clamps to 0");
}

void testMuteToggles() {
    using namespace odai::audio;
    Audio audio;
    audio.init(silentConfig());
    expectTrue(!audio.muted(), "starts unmuted by default");
    audio.setMuted(true);
    expectTrue(audio.muted(), "setMuted(true) reports muted");
    audio.setMuted(false);
    expectTrue(!audio.muted(), "setMuted(false) reports unmuted");
}

void testConfigSeedsState() {
    using namespace odai::audio;
    Audio audio;
    AudioConfig cfg = silentConfig();
    cfg.masterVolume = 0.7f;
    cfg.musicVolume = 0.3f;
    cfg.muted = true;
    audio.init(cfg);
    expectNear(audio.categoryVolume(SoundCategory::Master), 0.7f, "init seeds master volume");
    expectNear(audio.categoryVolume(SoundCategory::Music), 0.3f, "init seeds music volume");
    expectTrue(audio.muted(), "init seeds muted state");
}

void testListenerTransformDoesNotCrash() {
    using namespace odai::audio;
    Audio audio;
    audio.init(silentConfig());
    audio.setListenerTransform(ListenerTransform{});  // default (zero position, -Z forward)
    audio.setListenerTransform(ListenerTransform{
        odai::math::Vector3{12.0f, 5.0f, -3.0f},
        odai::math::Vector3{1.0f, 0.0f, 0.0f},
        odai::math::Vector3{0.0f, 1.0f, 0.0f}});
    expectTrue(true, "setListenerTransform does not crash on the null backend");
}

void testPlaySoundAtIsNoOp() {
    using namespace odai::audio;
    Audio audio;
    audio.init(silentConfig());
    audio.playSoundAt(SoundHandle{}, odai::math::Vector3{}, AttenuationParams{});
    audio.playSoundAt(SoundHandle{9}, odai::math::Vector3{1.0f, 2.0f, 3.0f}, AttenuationParams{2.0f, 20.0f, 1.5f});
    expectTrue(true, "playSoundAt does not crash on the null backend");
}

void testAmbientSlotsAlwaysInvalidOnNullBackend() {
    using namespace odai::audio;
    Audio audio;
    audio.init(silentConfig());
    const AmbientHandle global = audio.startAmbient(SoundHandle{5}, 1.0f);
    const AmbientHandle positional = audio.startAmbientAt(
        SoundHandle{6}, odai::math::Vector3{1.0f, 1.0f, 1.0f}, AttenuationParams{}, 1.0f);
    expectTrue(!global.valid(), "startAmbient on the null backend always yields an invalid handle");
    expectTrue(!positional.valid(), "startAmbientAt on the null backend always yields an invalid handle");
}

void testAmbientStopAndRepositionAreNoOps() {
    using namespace odai::audio;
    Audio audio;
    audio.init(silentConfig());
    audio.stopAmbient(AmbientHandle{}, 1.0f);
    audio.stopAmbient(AmbientHandle{3}, 0.5f);  // synthetic, non-issued handle
    audio.setAmbientPosition(AmbientHandle{}, odai::math::Vector3{});
    audio.setAmbientPosition(AmbientHandle{3}, odai::math::Vector3{4.0f, 4.0f, 4.0f});
    expectTrue(true, "stopAmbient/setAmbientPosition do not crash on invalid or synthetic handles");
}

void testManyConcurrentAmbientStartsStayIndependent() {
    using namespace odai::audio;
    Audio audio;
    audio.init(silentConfig());
    // Exercise well past kMaxAmbientSlots worth of concurrent starts; the null backend does
    // no real slot bookkeeping, so every one must independently no-op.
    for (int i = 0; i < kMaxAmbientSlots + 2; ++i) {
        const AmbientHandle handle = audio.startAmbientAt(
            SoundHandle{static_cast<std::uint32_t>(i + 1)},
            odai::math::Vector3{static_cast<float>(i), 0.0f, 0.0f}, AttenuationParams{}, 0.5f);
        expectTrue(!handle.valid(), "each concurrent ambient start on the null backend is independently invalid");
    }
}

void testCallsAfterShutdownAreSafe() {
    using namespace odai::audio;
    Audio audio;
    audio.init(AudioConfig{});
    audio.shutdown();
    audio.shutdown();  // idempotent
    audio.update(0.016f);
    audio.playSound(SoundHandle{1});
    expectTrue(!audio.deviceActive(), "deviceActive() is false after shutdown");
    expectTrue(!audio.loadSound("x.wav", SoundCategory::Ui).valid(),
               "loadSound after shutdown is invalid, not a crash");
}

std::filesystem::path writeToneFixture() {
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "odai_offline_audio_test.wav";
    std::vector<float> samples(480u * 2u);
    for (std::size_t frame = 0; frame < 480u; ++frame) {
        const float sample = 0.25f * std::sin(
            static_cast<float>(frame) * 2.0f * 3.1415926535f * 440.0f / 48000.0f);
        samples[frame * 2u] = sample;
        samples[frame * 2u + 1u] = sample;
    }
    odai::audio::WavWriter writer;
    expectTrue(writer.open(path, 48000u, 2u), "float WAV fixture opens");
    expectTrue(writer.write(samples), "float WAV fixture writes exact frame count");
    expectTrue(writer.framesWritten() == 480u, "WAV writer reports exact frames");
    expectTrue(writer.close(), "float WAV fixture closes and patches header");
    return path;
}

void testWavHeaderAndOfflineDeterminism() {
    using namespace odai::audio;
    const std::filesystem::path path = writeToneFixture();
    std::ifstream file(path, std::ios::binary);
    std::vector<std::uint8_t> bytes{
        std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
    expectTrue(bytes.size() == 44u + (480u * 2u * sizeof(float)),
               "WAV byte length matches 48 kHz stereo float payload");
    expectTrue(bytes.size() >= 44u && bytes[0] == 'R' && bytes[1] == 'I' &&
                   bytes[2] == 'F' && bytes[3] == 'F' && bytes[8] == 'W' &&
                   bytes[9] == 'A' && bytes[10] == 'V' && bytes[11] == 'E',
               "WAV header has RIFF/WAVE signatures");

    const auto render = [&]() {
        Audio audio;
        AudioConfig config;
        config.offlineMix = true;
        config.offlineSampleRate = 48000u;
        config.offlineChannels = 2u;
        expectTrue(audio.init(config), "offline mixer initializes without a device");
        expectTrue(audio.offlineMixActive(), "offline mixer reports active");
        expectTrue(!audio.deviceActive(), "offline mixer does not claim a playback device");
        expectTrue(audio.mixSampleRate() == 48000u && audio.mixChannels() == 2u,
                   "offline mixer exposes 48 kHz stereo format");
        const SoundHandle tone = audio.loadSound(path, SoundCategory::Ambient);
        expectTrue(tone.valid(), "offline mixer loads a WAV asset");
        const AmbientHandle loop = audio.startAmbient(tone, 0.0f);
        expectTrue(loop.valid(), "offline mixer starts an ambient loop");
        std::vector<float> pcm(1600u * 2u);
        expectTrue(audio.renderOfflineFrames(pcm, 1600u),
                   "offline mixer renders the requested exact frame count");
        return pcm;
    };
    const std::vector<float> first = render();
    const std::vector<float> second = render();
    expectTrue(first == second, "identical offline graphs render deterministic PCM");
    expectTrue(std::any_of(first.begin(), first.end(), [](float sample) {
                   return std::fabs(sample) > 1e-5f;
               }),
               "offline render contains mixed audio rather than silence");

    std::error_code removeError;
    std::filesystem::remove(path, removeError);
}

void testNullBackendRejectsOfflineRendering() {
    using namespace odai::audio;
    Audio audio;
    audio.init(silentConfig());
    std::vector<float> pcm(32u, 1.0f);
    expectTrue(!audio.offlineMixActive(), "null backend reports no offline mixer");
    expectTrue(!audio.renderOfflineFrames(pcm, 16u),
               "null backend reports offline rendering failure");
}

}  // namespace

int main() {
    testInitRunsSilent();
    testMissingFilesYieldInvalidHandles();
    testPlayCallsAreNoOps();
    testVolumeRoundTripAndClamp();
    testMuteToggles();
    testConfigSeedsState();
    testCallsAfterShutdownAreSafe();
    testListenerTransformDoesNotCrash();
    testPlaySoundAtIsNoOp();
    testAmbientSlotsAlwaysInvalidOnNullBackend();
    testAmbientStopAndRepositionAreNoOps();
    testManyConcurrentAmbientStartsStayIndependent();
    testWavHeaderAndOfflineDeterminism();
    testNullBackendRejectsOfflineRendering();

    if (g_failures != 0) {
        std::cerr << "[audio test] " << g_failures << " failures\n";
        return 1;
    }
    std::cout << "[audio test] all checks passed\n";
    return 0;
}
