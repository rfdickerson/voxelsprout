#pragma once

#include "import/bethesda/esm_reader.h"
#include "import/bethesda/plugin_load_order.h"
#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace odai::importer::bethesda {

// Skyrim IMGS HNAM/CNAM/TNAM/DNAM, in authored units. Kept separate from
// renderer tuning: unsupported channels remain inspectable instead of
// repurposed.
struct ImageSpaceSettings {
  std::array<float, 9> hdr{1, 1, 1, 0, 1, 1, 1, 1, 1};
  std::array<float, 3> cinematic{1, 1, 1}; // saturation, brightness, contrast
  std::array<float, 4> tint{};             // amount, red, green, blue
  std::array<float, 4> fade{};             // transient modifier RGB, alpha
  std::array<float, 3> dof{};              // strength, distance, range
  std::uint16_t dofFlags = 0;
};
struct ImageSpaceRecord {
  std::uint32_t formId = 0;
  std::string editorId;
  ImageSpaceSettings settings;
};
struct ImageSpaceKey {
  float time = 0, value = 0;
};
struct ImageSpaceColorKey {
  float time = 0;
  std::array<float, 4> rgba{};
};
struct ImageSpaceModifierRecord {
  std::uint32_t formId = 0;
  std::string editorId;
  bool animatable = false;
  float duration = 0;
  std::array<std::vector<ImageSpaceKey>, 21> multiply, add;
  // Blur, double vision, radial strength/up/start/down/down-start,
  // DoF strength/distance/range, motion blur.
  std::array<std::vector<ImageSpaceKey>, 11> effects;
  std::vector<ImageSpaceColorKey> tint, fade;
  bool radialUseTarget = false, dofUseTarget = false;
  std::array<float, 2> radialCenter{};
  std::uint8_t dofFlags = 0;
};
struct ImageSpaceTables {
  std::unordered_map<std::uint32_t, ImageSpaceRecord> spaces;
  std::unordered_map<std::uint32_t, ImageSpaceModifierRecord> modifiers;
  std::unordered_map<std::uint32_t, std::array<std::uint32_t, 4>> weatherSpaces;
  struct CellBinding {
    std::string editorId;
    std::uint32_t imageSpace = 0;
  };
  std::unordered_map<std::uint32_t, CellBinding> cells;
  std::unordered_map<std::string, std::uint32_t> cellImageSpaceByEditorId;
  std::vector<std::string> diagnostics;
};

// Separate Papyrus cross-fade chain; ordinary Apply/Remove modifiers remain independent.
class ImageSpaceCrossFade {
public:
  void clear();
  void apply(std::uint32_t modifier, float duration);
  ImageSpaceSettings sample(const ImageSpaceSettings&, const ImageSpaceTables&, float deltaSeconds);
  bool active() const { return target_ != 0 || !from_.empty(); }
private:
  struct Contribution { std::uint32_t id; float elapsed, weight; };
  std::vector<Contribution> from_;
  std::uint32_t target_ = 0;
  float elapsed_ = 0, duration_ = 0, targetElapsed_ = 0;
};

bool parseImageSpace(const EsmRecordView &, ImageSpaceRecord &,
                     std::string &error);
bool parseImageSpaceModifier(const EsmRecordView &, ImageSpaceModifierRecord &,
                             std::string &error);
bool buildImageSpaceTables(const FalloutLoadOrder &, ImageSpaceTables &,
                           std::string &error);
// Decode the authored no-sky radius enum. Sky-blurring variants have no
// equivalent in the current depth buffer and return zero rather than guessing.
float imageSpaceDofRadius(const ImageSpaceSettings &);
float sampleImageSpaceCurve(const std::vector<ImageSpaceKey> &, float time,
                            float fallback);
ImageSpaceSettings blendImageSpaces(const ImageSpaceSettings &,
                                    const ImageSpaceSettings &, float weight);
ImageSpaceSettings sampleWeatherImageSpace(const ImageSpaceTables &,
                                           std::uint32_t weather, float hour,
                                           float sunrise, float sunset,
                                           bool &found);
// Returns base unchanged after an animated modifier expires. Nonanimated
// modifiers persist until removed by their owner. Strength interpolates
// identity.
ImageSpaceSettings applyImageSpaceModifier(const ImageSpaceSettings &,
                                           const ImageSpaceModifierRecord &,
                                           float elapsedSeconds,
                                           float strength);
} // namespace odai::importer::bethesda
