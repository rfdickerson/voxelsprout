#pragma once
#include <cstdint>

namespace odai::importer {
// Authored SRC_ALPHA / ONE effects must not attenuate the background.
enum class ImportedBlendMode : std::uint8_t { Alpha = 0, Additive = 1 };
}
