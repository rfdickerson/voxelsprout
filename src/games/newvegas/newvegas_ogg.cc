// Ogg Vorbis -> WAV, so Fallout's ambient loops can be played at all.
//
// miniaudio ships decoders for WAV, FLAC and MP3 only; Vorbis needs a decoding
// backend bolted on. Fallout's music is .mp3 and plays directly, but every
// ambient and weather loop is .ogg, so without this the rain is silent.
//
// Decoding to a .wav beside the extracted .ogg is the smaller change than
// registering a custom ma_decoding_backend_vtable through the audio facade's
// PIMPL boundary, and it happens once per sound per install rather than per
// run. The cost is disk: these loops are a few seconds, so a handful of MB.
//
// stb_vorbis.c is compiled here, in its own translation unit, because it is a
// C file with its own static state and macro surface that should not be pulled
// into the game's own TU.

#include "games/newvegas/newvegas_ogg.h"

#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

// stb_vorbis defines these itself; silence the warnings its C sources produce
// under the project's C++ warning flags.
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#pragma GCC diagnostic ignored "-Wunused-value"
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wsign-compare"
#endif
#include <stb_vorbis.c>
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

namespace odai::games::newvegas {

namespace {

std::string shellQuote(const std::filesystem::path& path) {
    std::string quoted = "'";
    for (const char c : path.string()) {
        if (c == '\'') {
            quoted += "'\\''";
        } else {
            quoted.push_back(c);
        }
    }
    quoted.push_back('\'');
    return quoted;
}

void appendU32(std::vector<std::uint8_t>& out, std::uint32_t value) {
    for (int shift = 0; shift < 32; shift += 8) {
        out.push_back(static_cast<std::uint8_t>((value >> shift) & 0xFFu));
    }
}

void appendU16(std::vector<std::uint8_t>& out, std::uint16_t value) {
    out.push_back(static_cast<std::uint8_t>(value & 0xFFu));
    out.push_back(static_cast<std::uint8_t>((value >> 8) & 0xFFu));
}

}  // namespace

bool decodeOggToWav(
    const std::filesystem::path& oggPath, const std::filesystem::path& wavPath) {
    int channels = 0;
    int sampleRate = 0;
    short* samples = nullptr;
    const int frameCount =
        stb_vorbis_decode_filename(oggPath.string().c_str(), &channels, &sampleRate, &samples);
    if (frameCount <= 0 || samples == nullptr || channels <= 0 || sampleRate <= 0) {
        std::free(samples);
        return false;
    }

    const std::uint32_t dataBytes =
        static_cast<std::uint32_t>(frameCount) * static_cast<std::uint32_t>(channels) * 2u;
    const std::uint16_t blockAlign = static_cast<std::uint16_t>(channels * 2);

    std::vector<std::uint8_t> header;
    header.reserve(44);
    const char* riff = "RIFF";
    header.insert(header.end(), riff, riff + 4);
    appendU32(header, 36u + dataBytes);  // whole file after this field
    const char* waveFmt = "WAVEfmt ";
    header.insert(header.end(), waveFmt, waveFmt + 8);
    appendU32(header, 16u);  // PCM fmt chunk size
    appendU16(header, 1u);   // PCM
    appendU16(header, static_cast<std::uint16_t>(channels));
    appendU32(header, static_cast<std::uint32_t>(sampleRate));
    appendU32(header, static_cast<std::uint32_t>(sampleRate) * blockAlign);  // byte rate
    appendU16(header, blockAlign);
    appendU16(header, 16u);  // bits per sample
    const char* dataTag = "data";
    header.insert(header.end(), dataTag, dataTag + 4);
    appendU32(header, dataBytes);

    std::ofstream out(wavPath, std::ios::binary | std::ios::trunc);
    if (!out) {
        std::free(samples);
        return false;
    }
    out.write(reinterpret_cast<const char*>(header.data()),
              static_cast<std::streamsize>(header.size()));
    out.write(reinterpret_cast<const char*>(samples), static_cast<std::streamsize>(dataBytes));
    const bool ok = static_cast<bool>(out);
    std::free(samples);
    return ok;
}

bool decodeXwmToWav(
    const std::filesystem::path& xwmPath, const std::filesystem::path& wavPath) {
    if (xwmPath.empty() || wavPath.empty()) {
        return false;
    }
    std::ostringstream command;
    command << "ffmpeg -y -hide_banner -loglevel error -i "
            << shellQuote(xwmPath)
            << " -vn -acodec pcm_s16le " << shellQuote(wavPath);
    if (std::system(command.str().c_str()) != 0) {
        return false;
    }
    std::error_code error;
    return std::filesystem::exists(wavPath, error) && !error &&
        std::filesystem::file_size(wavPath, error) > 44u && !error;
}

}  // namespace odai::games::newvegas
