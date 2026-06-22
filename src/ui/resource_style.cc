#include "ui/resource_style.h"

#include <cstdio>
#include <mutex>
#include <unordered_map>

namespace odai::ui {
namespace {

std::unordered_map<std::string, ResourceStyle>& registry() {
    static std::unordered_map<std::string, ResourceStyle> s_registry;
    return s_registry;
}

std::mutex& registryMutex() {
    static std::mutex s_mutex;
    return s_mutex;
}

}  // namespace

void registerResourceStyle(std::string_view key, const ResourceStyle& style) {
    std::lock_guard lock(registryMutex());
    registry()[std::string(key)] = style;
}

const ResourceStyle* resourceStyle(std::string_view key) {
    std::lock_guard lock(registryMutex());
    auto& reg = registry();
    auto it = reg.find(std::string(key));
    return (it != reg.end()) ? &it->second : nullptr;
}

std::string resourceText(int value, bool signedDelta) {
    if (value == 0) return "\xE2\x80\x94";  // em dash
    if (signedDelta && value > 0) return "+" + std::to_string(value);
    return std::to_string(value);
}

std::string resourceTextFloat(float value, bool signedDelta, int decimals) {
    if (value == 0.0f) return "\xE2\x80\x94";
    char buf[32];
    const char* fmt = decimals <= 0 ? (signedDelta ? "%+.0f" : "%.0f")
                    : decimals == 1 ? (signedDelta ? "%+.1f" : "%.1f")
                                    : (signedDelta ? "%+.2f" : "%.2f");
    std::snprintf(buf, sizeof(buf), fmt, static_cast<double>(value));
    return buf;
}

}  // namespace odai::ui
