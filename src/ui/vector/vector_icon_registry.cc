#include "ui/vector/vector_icon_registry.h"

#include "ui/vector/svg_document.h"
#include "ui/vector/vector_cache_io.h"

#include <limits>
#include <string>
#include <system_error>

namespace odai::ui {

namespace {

UiRect computeBounds(const UiGeometryBlock& block) {
    if (block.vertices.empty()) {
        return UiRect{};
    }
    float minX = std::numeric_limits<float>::max();
    float minY = std::numeric_limits<float>::max();
    float maxX = std::numeric_limits<float>::lowest();
    float maxY = std::numeric_limits<float>::lowest();
    for (const UiVertex& v : block.vertices) {
        minX = std::min(minX, v.posPx[0]);
        minY = std::min(minY, v.posPx[1]);
        maxX = std::max(maxX, v.posPx[0]);
        maxY = std::max(maxY, v.posPx[1]);
    }
    return UiRect{minX, minY, maxX, maxY};
}

// True if `cache` exists and is at least as new as `source`.
bool cacheIsFresh(const std::filesystem::path& cache, const std::filesystem::path& source) {
    std::error_code ec;
    if (!std::filesystem::exists(cache, ec)) {
        return false;
    }
    const auto cacheTime = std::filesystem::last_write_time(cache, ec);
    if (ec) {
        return false;
    }
    const auto sourceTime = std::filesystem::last_write_time(source, ec);
    if (ec) {
        return true;  // Source missing/unreadable but cache exists — use the cache.
    }
    return cacheTime >= sourceTime;
}

}  // namespace

bool VectorIconRegistry::registerFromFile(std::string_view name,
                                          const std::filesystem::path& svgPath, float sizePx,
                                          float dpiScale) {
    if (sizePx <= 0.0f) {
        return false;
    }

    UiGeometryBlock block;
    float bakedSize = sizePx;

    // Fast path: a fresh sidecar cache baked at the requested size.
    std::filesystem::path cachePath = svgPath;
    cachePath.replace_extension(".odaivec");
    bool loaded = false;
    if (cacheIsFresh(cachePath, svgPath)) {
        float cachedSize = 0.0f;
        if (loadVectorCache(cachePath, block, cachedSize) && cachedSize == sizePx) {
            bakedSize = cachedSize;
            loaded = true;
        } else {
            block = UiGeometryBlock{};  // Reset on size mismatch / bad cache.
        }
    }

    if (!loaded) {
        SvgTessellateOptions opts;
        opts.targetSizePx = sizePx;
        opts.dpiScale = dpiScale;
        if (!tessellateSvgFile(svgPath, opts, block)) {
            return false;
        }
        bakedSize = sizePx;
    }

    VectorIcon icon;
    icon.geometry = std::move(block);
    icon.sizePx = bakedSize;
    icon.bounds = computeBounds(icon.geometry);
    m_icons[std::string(name)] = std::move(icon);
    return true;
}

void VectorIconRegistry::registerBaked(std::string_view name, const UiGeometryBlock& block,
                                       float sizePx) {
    VectorIcon icon;
    icon.geometry = block;
    icon.sizePx = sizePx;
    icon.bounds = computeBounds(icon.geometry);
    m_icons[std::string(name)] = std::move(icon);
}

const VectorIcon* VectorIconRegistry::resolve(std::string_view name) const {
    const auto it = m_icons.find(std::string(name));
    return it == m_icons.end() ? nullptr : &it->second;
}

VectorIconRegistry& VectorIconRegistry::global() {
    static VectorIconRegistry instance;
    return instance;
}

}  // namespace odai::ui
