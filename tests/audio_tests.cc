#include <cmath>
#include <iostream>

#include "audio/audio.h"

// These tests link only the facade + NullBackend (ODAI_AUDIO_HAVE_MINIAUDIO is
// not defined for this target), so they run with no audio device on any
// platform/CI. They verify the graceful-degrade contract: init never fails,
// missing files yield invalid handles, every play call is a safe no-op, and
// volume/mute state round-trips (so config persistence works without a device).

namespace {

int g_failures = 0;

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
    expectTrue(audio.init(AudioConfig{}), "init returns true even with no device");
    expectTrue(!audio.deviceActive(), "null backend reports deviceActive() == false");
}

void testMissingFilesYieldInvalidHandles() {
    using namespace odai::audio;
    Audio audio;
    audio.init(AudioConfig{});
    const SoundHandle sound = audio.loadSound("does/not/exist.wav", SoundCategory::Ui);
    const MusicHandle music = audio.loadMusic("does/not/exist.mp3");
    expectTrue(!sound.valid(), "loadSound on a missing file is invalid");
    expectTrue(!music.valid(), "loadMusic on a missing file is invalid");
}

void testPlayCallsAreNoOps() {
    using namespace odai::audio;
    Audio audio;
    audio.init(AudioConfig{});
    // Neither invalid nor (synthetic) valid handles may crash on the null backend.
    audio.playSound(SoundHandle{});
    audio.playSound(SoundHandle{42});
    audio.startAmbient(SoundHandle{}, 1.0f);
    audio.startAmbient(SoundHandle{7}, 0.5f);
    audio.stopAmbient(1.0f);
    audio.playMusic(MusicHandle{}, 2.0f, true);
    audio.playMusic(MusicHandle{3}, 2.0f, false);
    audio.stopMusic(0.0f);
    audio.update(0.016f);
    expectTrue(true, "play/stop/update calls do not crash on the null backend");
}

void testVolumeRoundTripAndClamp() {
    using namespace odai::audio;
    Audio audio;
    audio.init(AudioConfig{});

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
    audio.init(AudioConfig{});
    expectTrue(!audio.muted(), "starts unmuted by default");
    audio.setMuted(true);
    expectTrue(audio.muted(), "setMuted(true) reports muted");
    audio.setMuted(false);
    expectTrue(!audio.muted(), "setMuted(false) reports unmuted");
}

void testConfigSeedsState() {
    using namespace odai::audio;
    Audio audio;
    AudioConfig cfg;
    cfg.masterVolume = 0.7f;
    cfg.musicVolume = 0.3f;
    cfg.muted = true;
    audio.init(cfg);
    expectNear(audio.categoryVolume(SoundCategory::Master), 0.7f, "init seeds master volume");
    expectNear(audio.categoryVolume(SoundCategory::Music), 0.3f, "init seeds music volume");
    expectTrue(audio.muted(), "init seeds muted state");
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

}  // namespace

int main() {
    testInitRunsSilent();
    testMissingFilesYieldInvalidHandles();
    testPlayCallsAreNoOps();
    testVolumeRoundTripAndClamp();
    testMuteToggles();
    testConfigSeedsState();
    testCallsAfterShutdownAreSafe();

    if (g_failures != 0) {
        std::cerr << "[audio test] " << g_failures << " failures\n";
        return 1;
    }
    std::cout << "[audio test] all checks passed\n";
    return 0;
}
