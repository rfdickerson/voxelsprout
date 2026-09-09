#pragma once
#include "import/imported_scene.h"
#include <algorithm>
#include <span>

namespace odai::importer {
// Approximate the authored Blinn exponent in the existing GGX renderer.
// Keep source glossiness intact in the cooked record, rather than quantizing it.
inline GpuImportedMaterial makeImportedNifGpuMaterial(
    const ImportedNifLightingMaterial& source, std::span<const std::uint32_t> textureSlots) {
    GpuImportedMaterial gpu;
    gpu.sourceFlags[0] = source.flags1;
    gpu.sourceFlags[1] = source.flags2;
    gpu.sourceFlags[2] = source.shaderType;
    gpu.sourceFlags[3] = source.valid && source.shaderType <= 2 && (source.flags1 & (1u << 12)) == 0;
    gpu.emissiveRoughness[3] = std::clamp(std::pow(2.0f / (std::max(source.glossiness,0.0f) + 2.0f), 0.25f), 0.04f, 1.0f);
    for (int c = 0; c < 3; ++c) {
        gpu.specularStrength[c] = std::max(source.specular[c],0.0f);
        gpu.emissiveRoughness[c] = (source.flags1 & (1u << 22)) != 0 || (source.flags2 & (1u << 6)) != 0
                ? std::max(source.emissive[c] * source.emissiveMultiplier,0.0f) : 0.0f;
    }
    gpu.specularStrength[3] = (source.flags1 & 1u) != 0 ? std::max(source.specularStrength,0.0f) : 0.0f;
    gpu.environment[0] = source.environmentScale;
    const std::uint32_t roles[4] = {1,2,4,5};
    for (int role = 0; role < 4; ++role) {
        const auto texture = source.textures[roles[role]];
        gpu.textures[role] = texture < textureSlots.size() ? textureSlots[texture] : 0xffffffffu;
    }
    return gpu;
}
// Texture frame indices in these runtime tracks have already been remapped to
// resident bindless slots. Missing frames retain the static diffuse binding.
inline GpuImportedMaterial sampleImportedMaterial(const ImportedNifLightingMaterial& source,
    GpuImportedMaterial gpu, float elapsed) {
    if(source.animations.empty()) return gpu;
    float values[16]={source.uvOffset[0],source.uvScale[0],source.uvOffset[1],source.uvScale[1],
        source.alpha,source.emissive[0],source.emissive[1],source.emissive[2],source.emissiveMultiplier,
        source.specular[0],source.specular[1],source.specular[2],source.specularStrength,
        source.glossiness,source.environmentScale,0};
    for(const auto& track:source.animations) {
        const float value=sampleMaterialAnimation(track,elapsed);
        if(!std::isfinite(value)) continue;
        if (std::uint32_t(track.target) < 16u) values[std::uint32_t(track.target)]=value;
        if(std::uint32_t(track.target) >= std::uint32_t(MaterialAnimatedValue::DiffuseFrame) && !track.textures.empty()) {
            const auto frame=std::size_t(std::clamp(std::floor(double(value)),0.0,double(track.textures.size()-1)));
            const auto texture = track.textures[frame];
            if (track.target == MaterialAnimatedValue::DiffuseFrame) gpu.animationState[0] = texture;
            else if (texture != 0xffffffffu) gpu.textures[track.target == MaterialAnimatedValue::NormalFrame ? 0 : 1] = texture;
        }
    }
    gpu.animationUv[0]=values[0];gpu.animationUv[1]=values[2];
    gpu.animationUv[2]=values[1];gpu.animationUv[3]=values[3];
    gpu.animationColor[3]=std::max(values[4],0.f);
    gpu.animationState[1]=1;
    gpu.animationPalette[0]=values[5];gpu.animationPalette[1]=std::max(values[8],0.f);
    const bool effect=source.shaderType==0xffffffffu;
    for(int c=0;c<3;++c) {
        if(effect) gpu.animationColor[c]=std::max(values[5+c]*values[8],0.f);
        else {
            gpu.specularStrength[c]=std::max(values[9+c],0.f);
            if((source.flags1&(1u<<22)) || (source.flags2&(1u<<6)))
                gpu.emissiveRoughness[c]=std::max(values[5+c]*values[8],0.f);
        }
    }
    gpu.specularStrength[3]=(source.flags1&1)?std::max(values[12],0.f):0.f;
    gpu.emissiveRoughness[3]=std::clamp(std::pow(2.f/(std::max(values[13],0.f)+2.f),.25f),.04f,1.f);
    gpu.environment[0]=std::max(values[14],0.f);
    return gpu;
}
} // namespace odai::importer
