#pragma once
#include "import/fnv/nif_material_animation.h"

namespace odai::importer::fnv {
// Skyrim's manager sequences bind a concrete interpolator to a controller
// whose live endpoint is a blend interpolator. Resolve that binding rather
// than accidentally playing all of the manager's sequences simultaneously.
inline std::vector<std::vector<MaterialAnimationTrack>> readNifEffectSequences(
    std::span<const MaterialAnimationBlock> blocks,
    const std::vector<std::string>& strings) {
    std::vector<std::vector<MaterialAnimationTrack>> result(blocks.size());
    const auto valid = [&](int i) { return i >= 0 && std::size_t(i) < blocks.size(); };
    for (const auto& block : blocks) {
        if (block.type != "NiControllerSequence") continue;
        MaterialAnimationReader r{block.data};
        std::uint32_t name, count, grow;
        if (!r.read(name) || name >= strings.size() || !r.read(count) || count > 4096 ||
            !r.read(grow) || r.bytes.size() < count * 29u + 28u) continue;
        auto bindings = r.bytes.first(count * 29u);
        r.bytes = r.bytes.subspan(count * 29u);
        float weight, frequency, start, stop;
        std::int32_t text, manager;
        std::uint32_t cycle;
        if (!r.read(weight) || !r.read(text) || !r.read(cycle) || !r.read(frequency) ||
            !r.read(start) || !r.read(stop) || !r.read(manager) || !valid(manager) ||
            blocks[manager].type != "NiControllerManager" || cycle > 2) continue;
        for (std::uint32_t i = 0; i < count; ++i) {
            MaterialAnimationReader binding{bindings.subspan(i * 29u, 29u)};
            std::int32_t interp, controller;
            if (!binding.read(interp) || !binding.read(controller) || !valid(interp) ||
                !valid(controller) || blocks[controller].data.size() < 30) continue;
            const auto& ctrl = blocks[controller];
            std::int32_t target;
            std::memcpy(&target, ctrl.data.data() + 22, 4);
            if (!valid(target)) continue;
            std::vector<MaterialAnimationTrack> tracks;
            if (ctrl.type == "NiVisController" && blocks[interp].type == "NiBoolInterpolator") {
                MaterialAnimationReader ir{blocks[interp].data};
                std::uint8_t pose;
                std::int32_t data;
                MaterialAnimationTrack t;
                t.target = MaterialAnimatedValue::Visibility;
                t.interpolation = 5;
                if (!ir.read(pose) || !ir.read(data)) continue;
                if (data == -1 && pose <= 1) t.keys.push_back({start, float(pose)});
                else if (valid(data) && blocks[data].type == "NiBoolData") {
                    MaterialAnimationReader dr{blocks[data].data};
                    std::uint32_t n, interpolation;
                    if (!dr.read(n) || n > 65536 || !dr.read(interpolation) || interpolation != 5) continue;
                    bool good = true;
                    for (std::uint32_t k = 0; k < n; ++k) {
                        float time;
                        std::uint8_t value;
                        if (!dr.read(time) || !dr.read(value) || value > 1) { good = false; break; }
                        t.keys.push_back({time, float(value)});
                    }
                    if (!good) continue;
                }
                tracks.push_back(std::move(t));
            } else {
                // Reuse the typed scalar/color parser with this sequence's
                // concrete endpoint and no next-controller chain.
                auto patchedBlocks = std::vector<MaterialAnimationBlock>(blocks.begin(), blocks.end());
                std::vector<std::uint8_t> controllerBytes(ctrl.data.begin(), ctrl.data.end());
                std::vector<std::uint8_t> propertyBytes(blocks[target].data.begin(), blocks[target].data.end());
                const int offset = blocks[target].type == "BSLightingShaderProperty" ? 4 : 0;
                if (propertyBytes.size() < std::size_t(offset + 12)) continue;
                std::uint32_t extras;
                std::memcpy(&extras, propertyBytes.data() + offset + 4, 4);
                if (extras > 1024 || propertyBytes.size() < offset + 12u + extras * 4u) continue;
                std::memcpy(propertyBytes.data() + offset + 8 + extras * 4u, &controller, 4);
                const std::int32_t end = -1;
                std::memcpy(controllerBytes.data(), &end, 4);
                std::memcpy(controllerBytes.data() + 26, &interp, 4);
                patchedBlocks[controller].data = controllerBytes;
                patchedBlocks[target].data = propertyBytes;
                std::uint32_t unsupported = 0;
                tracks = readNifMaterialAnimation(patchedBlocks, target, {}, unsupported);
            }
            for (auto& t : tracks) {
                t.sequence = strings[name];
                t.start = start; t.stop = stop; t.frequency = frequency;
                t.phase = 0; t.cycle = cycle; t.backwards = false;
                if (validMaterialAnimationTrack(t)) result[target].push_back(std::move(t));
            }
        }
    }
    return result;
}
} // namespace odai::importer::fnv
