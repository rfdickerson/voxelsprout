#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace odai::ui {

// A polymorphic node in the data tree. Game code subclasses this to expose
// live simulation data; mock data loads it from JSON.
//
// All path components below the root are accessed via getChild(key).
// Lists are accessed via getList(key).
class DataNode {
public:
    virtual ~DataNode() = default;

    virtual std::string            getString(std::string_view key) const = 0;
    virtual float                  getFloat(std::string_view key)  const = 0;
    virtual std::shared_ptr<DataNode> getChild(std::string_view key) const = 0;
    virtual std::vector<std::shared_ptr<DataNode>> getList(std::string_view key) const = 0;

    // Convenience: return this node's own string value (used when the node IS
    // the value, e.g. list items that are plain strings).
    virtual std::string selfString() const { return {}; }
    virtual float       selfFloat()  const { return 0.0f; }
};

// A DataNode backed by an nlohmann::json value. Loaded from mock JSON files.
// Forward-declared here; implemented in ui_binding.cc to keep nlohmann out of
// headers.
class JsonDataNode : public DataNode {
public:
    // Factory: parse a JSON string and return the root DataNode.
    // Returns nullptr on parse failure.
    static std::shared_ptr<JsonDataNode> fromString(std::string_view jsonText);
    static std::shared_ptr<JsonDataNode> fromFile(const char* path);

    std::string getString(std::string_view key)                          const override;
    float       getFloat(std::string_view key)                           const override;
    std::shared_ptr<DataNode> getChild(std::string_view key)             const override;
    std::vector<std::shared_ptr<DataNode>> getList(std::string_view key) const override;
    std::string selfString() const override;
    float       selfFloat()  const override;

    // Opaque implementation; defined only in ui_binding.cc.
    struct Impl;
    std::shared_ptr<Impl> m_impl;
};

// The binding context holds named root DataNodes and resolves binding
// expressions of the form "{root.path.to.value:formatSpec}".
//
// Supported format specs (after ':'):
//   +0        signed integer   (+23, -4)
//   +0.0      signed float     (+4.2, -1.8)
//   ,0        thousands-sep integer  (1,439)
//   %         multiply × 100, append "%" (0.23 → "23%")
//   0/max     fraction         (2/4)  — max is another binding expression
//
// If no spec: value converted to string as-is.
class BindingContext {
public:
    void bind(std::string rootKey, std::shared_ptr<DataNode> node);

    // Resolve a binding expression. Returns the raw expression if no binding
    // is registered for the root key (makes missing bindings visible).
    std::string resolve(std::string_view expr) const;
    float       resolveFloat(std::string_view expr) const;

    bool empty() const { return m_roots.empty(); }

private:
    std::unordered_map<std::string, std::shared_ptr<DataNode>> m_roots;

    std::shared_ptr<DataNode> resolveNode(std::string_view path) const;
    static std::string applyFormat(float value, std::string_view spec);
};

}  // namespace odai::ui
