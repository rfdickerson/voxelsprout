#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace odai::bethesda {

enum class RecordKeyKind : std::uint8_t {
    PluginForm,
    Tes3Reference,
    Tes3Named,
};

// Stable record identity for runtime state and saves. The load-order-adjusted
// FormID is deliberately not stored: regular and light plugin slots can move
// whenever a profile changes, while (plugin, local FormID) remains stable.
// TES3 named records use their case-insensitive (record type, string ID)
// identity instead; placed FRMR references retain the defining plugin and
// plugin-local FRMR number.
struct RecordKey {
    // Keep the original members first so existing aggregate initializers remain
    // source-compatible. Named TES3 keys leave both of these empty/zero.
    std::string plugin;
    std::uint32_t localFormId = 0u;
    RecordKeyKind kind = RecordKeyKind::PluginForm;
    std::string recordType;
    std::string textId;

    [[nodiscard]] bool valid() const;
    [[nodiscard]] bool isNumeric() const { return kind != RecordKeyKind::Tes3Named; }
    [[nodiscard]] std::string toString() const;

    friend bool operator==(const RecordKey&, const RecordKey&) = default;
    friend bool operator<(const RecordKey& left, const RecordKey& right);
};

// Lowercases ASCII plugin names and normalizes path separators away. Bethesda
// plugin names are case-insensitive even on a case-sensitive host filesystem.
[[nodiscard]] RecordKey makeRecordKey(std::string plugin, std::uint32_t localFormId);
[[nodiscard]] RecordKey makeTes3ReferenceKey(std::string plugin, std::uint32_t localFrmr);
[[nodiscard]] RecordKey makeTes3RecordKey(std::string recordType, std::string textId);
[[nodiscard]] bool parseRecordKey(std::string_view text, RecordKey& out);

enum class ObjectIdKind : std::uint8_t {
    Invalid,
    PersistentReference,
    Spawned,
};

struct ObjectId {
    ObjectIdKind kind = ObjectIdKind::Invalid;
    RecordKey reference;
    std::uint64_t spawned = 0u;

    [[nodiscard]] static ObjectId persistent(RecordKey referenceKey);
    [[nodiscard]] static ObjectId runtime(std::uint64_t runtimeId);
    [[nodiscard]] bool valid() const;
    [[nodiscard]] std::string toString() const;

    friend bool operator==(const ObjectId&, const ObjectId&) = default;
    friend bool operator<(const ObjectId& left, const ObjectId& right) {
        if (left.kind != right.kind) return left.kind < right.kind;
        if (left.kind == ObjectIdKind::PersistentReference) return left.reference < right.reference;
        return left.spawned < right.spawned;
    }
};

struct RecordKeyHash {
    [[nodiscard]] std::size_t operator()(const RecordKey& key) const noexcept;
};

struct ObjectIdHash {
    [[nodiscard]] std::size_t operator()(const ObjectId& id) const noexcept;
};

}  // namespace odai::bethesda
