#include "render/video_writer.h"

#include "core/log.h"

#include <array>
#include <cstdlib>
#include <sstream>

namespace odai::render {

namespace {

bool ffmpegHasEncoder(const char* name) {
    std::string command = "ffmpeg -hide_banner -loglevel error -encoders 2>/dev/null";
    std::FILE* pipe = popen(command.c_str(), "r");
    if (pipe == nullptr) {
        return false;
    }
    std::array<char, 512> line{};
    bool found = false;
    while (std::fgets(line.data(), static_cast<int>(line.size()), pipe) != nullptr) {
        if (std::string(line.data()).find(name) != std::string::npos) {
            found = true;
            break;
        }
    }
    pclose(pipe);
    return found;
}

}  // namespace

const std::string& preferredVideoEncoder() {
    static const std::string encoder = []() -> std::string {
        if (const char* env = std::getenv("ODAI_CAPTURE_ENCODER")) {
            return env;
        }
        if (ffmpegHasEncoder("libx264")) {
            return "libx264";
        }
        if (ffmpegHasEncoder("libopenh264")) {
            return "libopenh264";
        }
        // Neither present: still name one, so the failure surfaces as ffmpeg's
        // own "Unknown encoder" rather than as a silently truncated file.
        return "libx264";
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
    // Two rate-control dialects: libx264 takes -crf, libopenh264 ignores it and
    // needs a bitrate. Getting this wrong is not an error, it is a soft-looking
    // file, so pick per encoder rather than passing both.
    const std::string quality = (encoder == "libx264")
        ? "-preset slow -crf 18"
        : "-b:v 16M -maxrate 20M";

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

}  // namespace odai::render
