#pragma once

// Decodes an Ogg Vorbis file to a 16-bit PCM .wav. See the .cc for why this
// exists: miniaudio has no Vorbis decoder, and every Fallout ambient loop is
// .ogg. Returns false if the file cannot be decoded or written.

#include <filesystem>

namespace odai::games::newvegas {

bool decodeOggToWav(const std::filesystem::path& oggPath, const std::filesystem::path& wavPath);

}  // namespace odai::games::newvegas
