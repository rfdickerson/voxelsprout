#include "anim/native_program.h"

#include <nlohmann/json.hpp>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <limits>

namespace odai::anim {
namespace {
using Json = nlohmann::json;
int integer(const Json& object, const char* name, int fallback) {
    if (!object.contains(name)) return fallback;
    const auto& field = object.at(name);
    if (!field.is_number_integer() || field.get<double>() < std::numeric_limits<int>::min() ||
        field.get<double>() > std::numeric_limits<int>::max()) throw std::runtime_error(std::string("invalid integer: ") + name);
    return field.get<int>();
}
AnimationContextValue value(const Json& j) {
    if (j.is_boolean()) return j.get<bool>();
    if (j.is_string()) return j.get<std::string>();
    if (j.is_number() && std::isfinite(j.get<double>())) return j.get<double>();
    throw std::runtime_error("condition values must be strings, booleans or finite numbers");
}
bool matches(const NativeAnimationCondition& condition, const AnimationSelectorContext& context) {
    if (condition.op == NativeAnimationCondition::Op::Tag)
        return context.tags.contains(std::get<std::string>(condition.value));
    const auto found = context.values.find(condition.field);
    if (found == context.values.end()) return false;
    if (condition.op == NativeAnimationCondition::Op::Equal) return found->second == condition.value;
    const auto* actual = std::get_if<double>(&found->second);
    const auto* expected = std::get_if<double>(&condition.value);
    return actual && expected && std::isfinite(*actual) &&
        (condition.op == NativeAnimationCondition::Op::Minimum ? *actual >= *expected : *actual <= *expected);
}
}

const NativeAnimationRule* NativeAnimationProgram::select(std::string_view state,
    const AnimationSelectorContext& context) const {
    const NativeAnimationRule* best = nullptr;
    for (const auto& rule : rules) {
        if (rule.state != state) continue;
        bool match = true;
        for (const auto& condition : rule.conditions) if (!matches(condition, context)) { match = false; break; }
        if (!match) continue;
        if (!best || rule.priority > best->priority ||
            (rule.priority == best->priority && (rule.layer > best->layer ||
                (rule.layer == best->layer && rule.id < best->id)))) best = &rule;
    }
    return best;
}

bool compileNativeAnimationPack(std::string_view text, std::string_view provider,
    int layer, NativeAnimationProgram& out, std::string& error) {
    error.clear();
    try {
        if (text.size() > 4u * 1024u * 1024u) throw std::runtime_error("pack exceeds 4 MiB");
        const auto root = Json::parse(text, [](int depth, Json::parse_event_t, Json&) {
            if (depth > 16) throw std::runtime_error("pack nesting exceeds 16 levels");
            return true;
        });
        const int version = integer(root, "version", 0);
        if (!root.is_object() || (version != 1 && version != 2) || !root.at("rules").is_array() ||
            root.at("rules").size() > 4096) throw std::runtime_error("expected version 1 and at most 4096 rules");
        NativeAnimationProgram compiled;
        std::set<std::string> ids;
        for (const auto& entry : root.at("rules")) {
            NativeAnimationRule rule;
            rule.id = entry.at("id").get<std::string>();
            rule.state = entry.at("state").get<std::string>();
            rule.provider = provider;
            rule.layer = layer;
            rule.priority = integer(entry, "priority", 0);
            rule.blendSeconds = entry.value("blend_time", 0.16f);
            rule.animationDriven = entry.value("animation_driven", false);
            rule.holdUntilLanding = entry.value("hold_until_landing", false);
            if (entry.contains("motion_warps")) {
                const auto& windows = entry.at("motion_warps");
                if (version != 2 || !windows.is_array() || windows.size() > 16)
                    throw std::runtime_error("motion warps require version 2, at most 16 windows");
                float previousEnd = -1;
                for (const auto& w : windows) {
                    NativeMotionWarpWindow window{w.at("target").get<std::string>(),w.at("start").get<float>(),
                        w.at("end").get<float>(),w.value("max_translation",32.f),w.value("max_yaw",.5f)};
                    if (window.target.empty() || !std::isfinite(window.start) || !std::isfinite(window.end) ||
                        window.start < 0 || window.start < previousEnd || window.end <= window.start ||
                        !std::isfinite(window.maxTranslation) || window.maxTranslation < 0 || window.maxTranslation > 256 ||
                        !std::isfinite(window.maxYawRadians) || window.maxYawRadians < 0 || window.maxYawRadians > 3.14159265f)
                        throw std::runtime_error("invalid or overlapping motion warp window");
                    previousEnd = window.end;
                    rule.motionWarps.push_back(std::move(window));
                }
            }
            if (version == 2 && entry.contains("graph")) {
                auto graph = std::make_shared<PoseGraphProgram>();
                std::string graphError;
                if (!compilePoseGraph(entry.at("graph").dump(), *graph, graphError))
                    throw std::runtime_error("invalid pose graph: " + graphError);
                rule.graph = std::move(graph);
            }
            if (rule.id.empty() || rule.state.empty() || !ids.insert(rule.id).second ||
                !std::isfinite(rule.blendSeconds) || rule.blendSeconds < 0 || rule.blendSeconds > 10)
                throw std::runtime_error("invalid/duplicate rule ID, state or blend time");
            const auto conditions = entry.value("conditions", Json::object());
            if (!conditions.is_object() || conditions.size() > 64) throw std::runtime_error("invalid conditions");
            for (const auto& [field, test] : conditions.items()) {
                if (field.empty()) throw std::runtime_error("empty condition field");
                if (field == "tags") {
                    if (!test.is_array() || test.size() > 64) throw std::runtime_error("tags must be an array");
                    for (const auto& tag : test)
                        rule.conditions.push_back({field, tag.get<std::string>(), NativeAnimationCondition::Op::Tag});
                } else if (test.is_object()) {
                    if (test.empty()) throw std::runtime_error("empty range");
                    for (const auto& [op, bound] : test.items()) {
                        if ((op != "min" && op != "max") || !bound.is_number()) throw std::runtime_error("invalid range");
                        rule.conditions.push_back({field, value(bound), op == "min" ?
                            NativeAnimationCondition::Op::Minimum : NativeAnimationCondition::Op::Maximum});
                    }
                } else rule.conditions.push_back({field, value(test)});
            }
            const auto& variants = entry.at("variants");
            if (!variants.is_array() || variants.empty() || variants.size() > 128) throw std::runtime_error("invalid variants");
            double total = 0;
            for (const auto& item : variants) {
                NativeAnimationVariant variant{item.at("clip").get<std::string>(),
                    item.value("weight", 1.0), item.value("loop", true)};
                total += variant.weight;
                if (variant.clip.empty() || !std::isfinite(variant.weight) || variant.weight <= 0 || !std::isfinite(total))
                    throw std::runtime_error("invalid clip or variant weight");
                rule.variants.push_back(std::move(variant));
            }
            if (entry.contains("blend_samples")) {
                const auto& samples = entry.at("blend_samples");
                if (!samples.is_array() || samples.empty() || samples.size() > 32) throw std::runtime_error("invalid blend samples");
                std::set<std::pair<float, float>> points;
                for (const auto& item : samples) {
                    NativeBlendSample sample{item.at("clip").get<std::string>(), item.at("x").get<float>(), item.at("z").get<float>()};
                    if (sample.clip.empty() || !std::isfinite(sample.x) || !std::isfinite(sample.z) ||
                        !points.emplace(sample.x, sample.z).second) throw std::runtime_error("invalid/duplicate blend point");
                    rule.blendSamples.push_back(std::move(sample));
                }
            }
            if (entry.contains("layers")) {
                const auto& layers = entry.at("layers");
                if (!layers.is_array() || layers.size() > 16) throw std::runtime_error("invalid layers");
                std::set<std::string> layerIds;
                for (const auto& item : layers) {
                    NativeAnimationLayer overlay;
                    overlay.id = item.at("id").get<std::string>();
                    overlay.clip = item.at("clip").get<std::string>();
                    overlay.referenceClip = item.value("reference_clip", std::string{});
                    overlay.order = integer(item, "order", 0);
                    overlay.weight = item.value("weight", 1.f);
                    overlay.additive = item.value("additive", false);
                    overlay.bones = item.at("bones").get<std::vector<std::string>>();
                    if (overlay.id.empty() || overlay.clip.empty() || !layerIds.insert(overlay.id).second ||
                        !std::isfinite(overlay.weight) || overlay.weight < 0 || overlay.weight > 1 ||
                        overlay.bones.empty() || overlay.bones.size() > 512 ||
                        (overlay.additive && overlay.referenceClip.empty())) throw std::runtime_error("invalid layer or additive reference");
                    rule.layers.push_back(std::move(overlay));
                }
                std::sort(rule.layers.begin(), rule.layers.end(), [](const auto& a, const auto& b) {
                    if (a.additive != b.additive) return !a.additive;
                    return a.order != b.order ? a.order < b.order : a.id < b.id;
                });
            }
            compiled.rules.push_back(std::move(rule));
        }
        out = std::move(compiled);
        return true;
    } catch (const std::exception& exception) {
        error = std::string("invalid native animation pack: ") + exception.what();
        return false;
    }
}

std::size_t chooseNativeVariant(const NativeAnimationRule& rule, std::uint64_t& state) {
    // SplitMix64: explicitly specified arithmetic, independent of standard-library RNGs.
    state += 0x9e3779b97f4a7c15ull;
    auto bits = state;
    bits = (bits ^ (bits >> 30)) * 0xbf58476d1ce4e5b9ull;
    bits = (bits ^ (bits >> 27)) * 0x94d049bb133111ebull;
    bits ^= bits >> 31;
    double total = 0;
    for (const auto& variant : rule.variants) total += variant.weight;
    double choice = static_cast<double>(bits >> 11) * 0x1.0p-53 * total;
    for (std::size_t i = 0; i < rule.variants.size(); ++i) {
        choice -= rule.variants[i].weight;
        if (choice < 0) return i;
    }
    return rule.variants.empty() ? 0 : rule.variants.size() - 1;
}
} // namespace odai::anim
