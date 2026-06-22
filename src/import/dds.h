#pragma once

#include "import/imported_scene.h"

#include <cstdint>
#include <filesystem>

namespace odai::importer {

// Load a DDS file into an ImportedSceneTexture.
// Handles classic DDS (DXT1/DXT5/ATI1/ATI2) and the DX10 extension (BC7, etc.).
// On success fills tex.width/height/mipLevelCount/format/rgba8 and returns true.
// On failure returns false; tex is unchanged.
bool loadDds(const std::filesystem::path& path, ImportedSceneTexture& tex);

// Write a block-compressed DDS file with the given packed mip chain.
// format must be BC1–BC7 (not RGBA8). mipData must contain all mip levels
// packed largest-first. Returns false if the file could not be written.
bool writeDds(const std::filesystem::path& path,
              std::uint32_t width, std::uint32_t height,
              std::uint32_t mipLevelCount, TextureFormat format,
              const std::uint8_t* mipData, std::size_t mipDataSize);

// Bytes per compressed 4×4 block for format; returns 0 for RGBA8.
std::uint32_t ddsBlockBytes(TextureFormat format);

} // namespace odai::importer
