#include "render/video_writer.h"

#include "core/log.h"

#include <cstdlib>
#include <sstream>

namespace odai::render {

const std::string& preferredVideoEncoder() {
    static const std::string encoder = []() -> std::string {
        if (const char* env = std::getenv("ODAI_CAPTURE_ENCODER")) {
            return env;
        }
        return "libopenh264";
    }();
    return encoder;
}

VideoWriter::~VideoWriter() {
    close();
}

bool VideoWriter::open(const std::string& outputPath,
                       std::uint32_t width,
                       std::uint32_t height,
                       int fps) {
    close();
    if (width == 0u || height == 0u || fps <= 0) {
        VOX_LOGE("render") << "video capture: bad geometry " << width << "x" << height << " @"
                           << fps;
        return false;
    }

    const std::string& encoder = preferredVideoEncoder();
    // openh264 has no -crf: it is bitrate-controlled, and passing a CRF is
    // silently ignored rather than rejected, which reads as "the capture came
    // out soft" with nothing to point at. 16M is generous for 1080p and holds
    // up at the 4K the swapchain often opens at.
    const std::string quality = "-b:v 16M -maxrate 20M";

    std::ostringstream command;
    command << "ffmpeg -y -hide_banner -loglevel error"
            << " -f rawvideo -pix_fmt rgb24"
            << " -s " << width << "x" << height
            << " -r " << fps
            << " -i -"
            // Odd dimensions are real: the swapchain follows the window, and
            // yuv420p cannot represent an odd width or height. Round down.
            << " -vf \"scale=trunc(iw/2)*2:trunc(ih/2)*2\""
            << " -c:v " << encoder << " " << quality
            << " -pix_fmt yuv420p -movflags +faststart"
            << " \"" << outputPath << "\"";

    m_pipe = popen(command.str().c_str(), "w");
    if (m_pipe == nullptr) {
        VOX_LOGE("render") << "video capture: cannot start ffmpeg for " << outputPath;
        return false;
    }
    m_path = outputPath;
    m_frameBytes = static_cast<std::size_t>(width) * height * 3u;
    m_framesWritten = 0;
    VOX_LOGI("render") << "video capture: " << outputPath << " " << width << "x" << height << " @"
                       << fps << " fps via " << encoder;
    return true;
}

bool VideoWriter::writeFrame(const std::vector<std::uint8_t>& rgb) {
    if (m_pipe == nullptr) {
        return false;
    }
    if (rgb.size() != m_frameBytes) {
        VOX_LOGE("render") << "video capture: frame is " << rgb.size() << " bytes, expected "
                           << m_frameBytes << " -- the swapchain resized mid-capture";
        return false;
    }
    const std::size_t written = std::fwrite(rgb.data(), 1, rgb.size(), m_pipe);
    if (written != rgb.size()) {
        VOX_LOGE("render") << "video capture: short write to encoder (" << written << "/"
                           << rgb.size() << ")";
        return false;
    }
    ++m_framesWritten;
    return true;
}

bool VideoWriter::close() {
    if (m_pipe == nullptr) {
        return true;
    }
    const int status = pclose(m_pipe);
    m_pipe = nullptr;
    const bool ok = status == 0;
    if (ok) {
        VOX_LOGI("render") << "video capture written: " << m_path << " (" << m_framesWritten
                           << " frames)";
    } else {
        VOX_LOGE("render") << "video capture: ffmpeg exited " << status << " for " << m_path;
    }
    return ok;
}

bool muxVideoAndAudio(const std::string& videoPath, const std::string& wavPath,
                      const std::string& outputPath) {
    std::ostringstream command;
    command << "ffmpeg -y -hide_banner -loglevel error"
            << " -i \"" << videoPath << "\""
            << " -i \"" << wavPath << "\""
            << " -map 0:v:0 -map 1:a:0 -c:v copy"
            << " -c:a aac -b:a 192k -ar 48000 -ac 2"
            << " -shortest -movflags +faststart"
            << " \"" << outputPath << "\"";
    const int status = std::system(command.str().c_str());
    if (status != 0) {
        VOX_LOGE("render") << "audio mux: ffmpeg exited " << status << " for " << outputPath;
        return false;
    }
    VOX_LOGI("render") << "audio mux written: " << outputPath;
    return true;
}

}  // namespace odai::render
