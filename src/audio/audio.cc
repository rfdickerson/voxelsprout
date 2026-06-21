#include "audio/audio.h"

#include "audio/audio_backend.h"

namespace odai::audio {
namespace {
float clamp01(float v) {
    if (v < 0.0f) return 0.0f;
    if (v > 1.0f) return 1.0f;
    return v;
}
}  // namespace

// Defined here (not in the header) so the inline-generated special members see a
// complete AudioBackend type — same pattern as odai::render::Renderer.
Audio::Audio() = default;
Audio::~Audio() = default;
Audio::Audio(Audio&&) noexcept = default;
Audio& Audio::operator=(Audio&&) noexcept = default;

bool Audio::init(const AudioConfig& cfg) {
#ifdef ODAI_AUDIO_HAVE_MINIAUDIO
    m_backend = createMiniaudioBackend(cfg);
#endif
    if (!m_backend) {
        m_backend = createNullBackend(cfg);
    }
    return true;
}

void Audio::update(float dt) {
    if (m_backend) m_backend->update(dt);
}

void Audio::shutdown() {
    m_backend.reset();
}

SoundHandle Audio::loadSound(const std::filesystem::path& file, SoundCategory category) {
    return m_backend ? m_backend->loadSound(file, category) : SoundHandle{};
}

MusicHandle Audio::loadMusic(const std::filesystem::path& file) {
    return m_backend ? m_backend->loadMusic(file) : MusicHandle{};
}

void Audio::playSound(SoundHandle clip) {
    if (m_backend) m_backend->playSound(clip);
}

void Audio::startAmbient(SoundHandle loop, float fadeSeconds) {
    if (m_backend) m_backend->startAmbient(loop, fadeSeconds);
}

void Audio::stopAmbient(float fadeSeconds) {
    if (m_backend) m_backend->stopAmbient(fadeSeconds);
}

void Audio::playMusic(MusicHandle track, float fadeSeconds, bool loop) {
    if (m_backend) m_backend->playMusic(track, fadeSeconds, loop);
}

void Audio::stopMusic(float fadeSeconds) {
    if (m_backend) m_backend->stopMusic(fadeSeconds);
}

void Audio::setMasterVolume(float v) {
    if (m_backend) m_backend->setMasterVolume(clamp01(v));
}

void Audio::setCategoryVolume(SoundCategory c, float v) {
    if (m_backend) m_backend->setCategoryVolume(c, clamp01(v));
}

float Audio::categoryVolume(SoundCategory c) const {
    return m_backend ? m_backend->categoryVolume(c) : 0.0f;
}

void Audio::setMuted(bool muted) {
    if (m_backend) m_backend->setMuted(muted);
}

bool Audio::muted() const {
    return m_backend && m_backend->muted();
}

bool Audio::deviceActive() const {
    return m_backend && m_backend->deviceActive();
}

}  // namespace odai::audio
