#pragma once

// Streams captured frames straight into an encoder instead of writing stills.
//
// The stills path (`--capture-seq`) writes one PPM per frame, and at the sizes
// this renderer actually opens -- 2133x1200 is 7.7 MB uncompressed -- a
// three-location capture put ~7 GB on disk and exhausted the filesystem quota
// mid-run. Nothing about a video capture needs those files to exist: the frames
// are produced in order, consumed in order, and never revisited.
//
// The encoder is an ffmpeg child process fed raw rgb24 on its stdin. Linking a
// codec library instead would put an encoder dependency on every target that
// compiles the Vulkan backend, for a diagnostic/authoring feature -- the same
// reasoning that keeps frame_capture.cc emitting PPM rather than PNG.

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace odai::render {

class VideoWriter {
public:
    VideoWriter() = default;
    ~VideoWriter();

    VideoWriter(const VideoWriter&) = delete;
    VideoWriter& operator=(const VideoWriter&) = delete;

    // Spawns the encoder. Frame size is fixed for the life of the writer,
    // because it is baked into ffmpeg's input description.
    bool open(const std::string& outputPath, std::uint32_t width, std::uint32_t height, int fps);

    // One frame of tightly packed rgb24, exactly width*height*3 bytes.
    bool writeFrame(const std::vector<std::uint8_t>& rgb);

    // Closes stdin and waits for the encoder to finish writing the container.
    // Called by the destructor, but worth calling explicitly: this is where a
    // failed encode is reported, and a destructor is the wrong place to learn
    // that an hour of rendering produced nothing.
    bool close();

    bool isOpen() const { return m_pipe != nullptr; }
    const std::string& path() const { return m_path; }
    std::uint64_t framesWritten() const { return m_framesWritten; }

private:
    std::FILE* m_pipe = nullptr;
    std::string m_path;
    std::size_t m_frameBytes = 0;
    std::uint64_t m_framesWritten = 0;
};

// The H.264 encoder to ask ffmpeg for, resolved once against what this ffmpeg
// build actually has. libx264 is the usual answer and is what most builds ship,
// but a distribution ffmpeg without it is common enough to matter -- Fedora's
// is one -- and discovering that only when the child process dies wastes the
// whole render. $ODAI_CAPTURE_ENCODER overrides.
const std::string& preferredVideoEncoder();

}  // namespace odai::render
