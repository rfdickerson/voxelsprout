#include "ui/document/ui_binding.h"

#include "core/log.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <fstream>
#include <sstream>

namespace odai::ui {

// ---------------------------------------------------------------------------
// JsonDataNode implementation
// ---------------------------------------------------------------------------

struct JsonDataNode::Impl {
    nlohmann::json value;
};

static std::shared_ptr<JsonDataNode> makeNode(nlohmann::json v) {
    auto n = std::make_shared<JsonDataNode>();
    n->m_impl = std::make_shared<JsonDataNode::Impl>();
    n->m_impl->value = std::move(v);
    return n;
}

std::shared_ptr<JsonDataNode> JsonDataNode::fromString(std::string_view jsonText) {
    try {
        return makeNode(nlohmann::json::parse(jsonText));
    } catch (const nlohmann::json::exception& e) {
        VOX_LOGE("ui") << "JsonDataNode::fromString parse error: " << e.what() << "\n";
        return nullptr;
    }
}

std::shared_ptr<JsonDataNode> JsonDataNode::fromFile(const char* path) {
    if (!path) return nullptr;
    std::ifstream f(path);
    if (!f) {
        VOX_LOGE("ui") << "JsonDataNode::fromFile: cannot open " << path << "\n";
        return nullptr;
    }
    try {
        nlohmann::json j;
        f >> j;
        return makeNode(std::move(j));
    } catch (const nlohmann::json::exception& e) {
        VOX_LOGE("ui") << "JsonDataNode::fromFile parse error in " << path
                       << ": " << e.what() << "\n";
        return nullptr;
    }
}

std::string JsonDataNode::getString(std::string_view key) const {
    if (!m_impl) return {};
    const auto& v = m_impl->value;
    if (v.is_object() && v.contains(key)) {
        const auto& child = v[std::string(key)];
        if (child.is_string()) return child.get<std::string>();
        if (child.is_number()) return std::to_string(child.get<double>());
        if (child.is_boolean()) return child.get<bool>() ? "true" : "false";
    }
    return {};
}

float JsonDataNode::getFloat(std::string_view key) const {
    if (!m_impl) return 0.0f;
    const auto& v = m_impl->value;
    if (v.is_object() && v.contains(key)) {
        const auto& child = v[std::string(key)];
        if (child.is_number()) return child.get<float>();
        if (child.is_string()) {
            try { return std::stof(child.get<std::string>()); } catch (...) {}
        }
    }
    return 0.0f;
}

std::shared_ptr<DataNode> JsonDataNode::getChild(std::string_view key) const {
    if (!m_impl) return nullptr;
    const auto& v = m_impl->value;
    if (v.is_object() && v.contains(key)) {
        return makeNode(v[std::string(key)]);
    }
    return nullptr;
}

std::vector<std::shared_ptr<DataNode>> JsonDataNode::getList(std::string_view key) const {
    std::vector<std::shared_ptr<DataNode>> result;
    if (!m_impl) return result;
    const auto& v = m_impl->value;
    const nlohmann::json* arr = nullptr;
    if (v.is_array()) {
        arr = &v;
    } else if (v.is_object() && v.contains(key) && v[std::string(key)].is_array()) {
        arr = &v[std::string(key)];
    }
    if (arr) {
        result.reserve(arr->size());
        for (const auto& item : *arr) {
            result.push_back(makeNode(item));
        }
    }
    return result;
}

std::string JsonDataNode::selfString() const {
    if (!m_impl) return {};
    const auto& v = m_impl->value;
    if (v.is_string()) return v.get<std::string>();
    if (v.is_number()) return std::to_string(v.get<double>());
    if (v.is_boolean()) return v.get<bool>() ? "true" : "false";
    return {};
}

float JsonDataNode::selfFloat() const {
    if (!m_impl) return 0.0f;
    const auto& v = m_impl->value;
    if (v.is_number()) return v.get<float>();
    if (v.is_string()) {
        try { return std::stof(v.get<std::string>()); } catch (...) {}
    }
    return 0.0f;
}

// ---------------------------------------------------------------------------
// BindingContext
// ---------------------------------------------------------------------------

void BindingContext::bind(std::string rootKey, std::shared_ptr<DataNode> node) {
    m_roots[std::move(rootKey)] = std::move(node);
}

// Resolve a dotted path like "city.name" against registered roots.
// Returns nullptr if any segment is missing.
std::shared_ptr<DataNode> BindingContext::resolveNode(std::string_view path) const {
    const std::size_t dot = path.find('.');
    const std::string_view root = path.substr(0, dot);
    const auto it = m_roots.find(std::string(root));
    if (it == m_roots.end()) return nullptr;

    std::shared_ptr<DataNode> node = it->second;
    if (dot == std::string_view::npos) return node;

    std::string_view rest = path.substr(dot + 1);
    while (!rest.empty() && node != nullptr) {
        const std::size_t nextDot = rest.find('.');
        const std::string_view seg = rest.substr(0, nextDot);
        node = node->getChild(seg);
        if (nextDot == std::string_view::npos) break;
        rest = rest.substr(nextDot + 1);
    }
    return node;
}

std::string BindingContext::applyFormat(float value, std::string_view spec) {
    if (spec.empty()) {
        // Default: strip trailing zeros for floats, show integer if whole.
        if (std::fmod(std::abs(value), 1.0f) < 0.001f) {
            return std::to_string(static_cast<int>(value));
        }
        std::ostringstream ss;
        ss.precision(1);
        ss << std::fixed << value;
        return ss.str();
    }
    if (spec == "+0") {
        const int iv = static_cast<int>(std::round(value));
        return (iv >= 0 ? "+" : "") + std::to_string(iv);
    }
    if (spec == "+0.0") {
        std::ostringstream ss;
        ss.precision(1);
        ss << std::fixed << (value >= 0.0f ? "+" : "") << value;
        return ss.str();
    }
    if (spec == ",0") {
        const long long iv = static_cast<long long>(std::round(value));
        std::string s = std::to_string(std::abs(iv));
        for (int i = static_cast<int>(s.size()) - 3; i > 0; i -= 3) {
            s.insert(static_cast<std::size_t>(i), ",");
        }
        return (iv < 0 ? "-" : "") + s;
    }
    if (spec == "%") {
        return std::to_string(static_cast<int>(std::round(value * 100.0f))) + "%";
    }
    // Unknown spec: just convert.
    return std::to_string(static_cast<int>(std::round(value)));
}

std::string BindingContext::resolve(std::string_view expr) const {
    // Strip {…}
    if (expr.size() >= 2 && expr.front() == '{' && expr.back() == '}') {
        expr = expr.substr(1, expr.size() - 2);
    }

    // Split off format spec.
    std::string_view path = expr;
    std::string_view spec;
    const std::size_t colon = expr.find(':');
    if (colon != std::string_view::npos) {
        path = expr.substr(0, colon);
        spec = expr.substr(colon + 1);
    }

    std::shared_ptr<DataNode> node = resolveNode(path);
    if (!node) {
        // Return raw expression so artists can see the missing binding.
        return std::string("{") + std::string(expr) + "}";
    }

    if (!spec.empty()) {
        return applyFormat(node->selfFloat(), spec);
    }
    return node->selfString();
}

float BindingContext::resolveFloat(std::string_view expr) const {
    if (expr.size() >= 2 && expr.front() == '{' && expr.back() == '}') {
        expr = expr.substr(1, expr.size() - 2);
    }
    const std::size_t colon = expr.find(':');
    const std::string_view path = (colon != std::string_view::npos) ? expr.substr(0, colon) : expr;
    std::shared_ptr<DataNode> node = resolveNode(path);
    return node ? node->selfFloat() : 0.0f;
}

}  // namespace odai::ui
