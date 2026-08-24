#include "bethesda/runtime_ids.h"

#include "core/hash.h"

#include <algorithm>
#include <charconv>
#include <cctype>
#include <iomanip>
#include <sstream>
#include <tuple>

namespace odai::bethesda {
namespace {

std::string normalizePlugin(std::string plugin) {
    std::replace(plugin.begin(), plugin.end(), '\\', '/');
    const std::size_t slash = plugin.find_last_of('/');
    if (slash != std::string::npos) plugin.erase(0u, slash + 1u);
    for (char& ch : plugin) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return plugin;
}

std::string normalizeTes3Type(std::string type) {
    for (char& ch : type) {
        ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
    }
    return type;
}

std::string normalizeTes3Id(std::string id) {
    for (char& ch : id) {
        if (static_cast<unsigned char>(ch) < 0x80u) {
            ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        }
    }
    return id;
}

bool isUnreserved(unsigned char ch) {
    return std::isalnum(ch) != 0 || ch == '-' || ch == '_' || ch == '.' || ch == '~' || ch == ' ';
}

std::string escapeIdentity(std::string_view value) {
    static constexpr char digits[] = "0123456789ABCDEF";
    std::string result;
    result.reserve(value.size());
    for (const unsigned char ch : value) {
        if (isUnreserved(ch)) {
            result.push_back(static_cast<char>(ch));
        } else {
            result.push_back('%');
            result.push_back(digits[ch >> 4u]);
            result.push_back(digits[ch & 0x0fu]);
        }
    }
    return result;
}

int hexDigit(char ch) {
    if (ch >= '0' && ch <= '9') return ch - '0';
    if (ch >= 'a' && ch <= 'f') return ch - 'a' + 10;
    if (ch >= 'A' && ch <= 'F') return ch - 'A' + 10;
    return -1;
}

bool unescapeIdentity(std::string_view value, std::string& out) {
    out.clear();
    out.reserve(value.size());
    for (std::size_t i = 0u; i < value.size(); ++i) {
        if (value[i] != '%') {
            out.push_back(value[i]);
            continue;
        }
        if (i + 2u >= value.size()) return false;
        const int high = hexDigit(value[i + 1u]);
        const int low = hexDigit(value[i + 2u]);
        if (high < 0 || low < 0) return false;
        out.push_back(static_cast<char>((high << 4) | low));
        i += 2u;
    }
    return true;
}

std::uint64_t hashText(std::string_view text) {
    std::uint64_t hash = 1469598103934665603ull;
    for (const unsigned char ch : text) {
        hash ^= ch;
        hash *= 1099511628211ull;
    }
    return hash;
}

}  // namespace

bool RecordKey::valid() const {
    if (kind == RecordKeyKind::Tes3Named) {
        return plugin.empty() && localFormId == 0u && !recordType.empty() && !textId.empty();
    }
    return !plugin.empty() && recordType.empty() && textId.empty();
}

std::string RecordKey::toString() const {
    if (kind == RecordKeyKind::Tes3Named) {
        return "tes3:" + recordType + ":" + escapeIdentity(textId);
    }
    std::ostringstream out;
    if (kind == RecordKeyKind::Tes3Reference) out << "frmr:";
    out << plugin << ":0x" << std::hex << std::setfill('0') << std::setw(8) << localFormId;
    return out.str();
}

bool operator<(const RecordKey& left, const RecordKey& right) {
    return std::tie(left.kind, left.plugin, left.localFormId, left.recordType, left.textId) <
        std::tie(right.kind, right.plugin, right.localFormId, right.recordType, right.textId);
}

RecordKey makeRecordKey(std::string plugin, std::uint32_t localFormId) {
    return RecordKey{normalizePlugin(std::move(plugin)), localFormId,
                     RecordKeyKind::PluginForm, {}, {}};
}

RecordKey makeTes3ReferenceKey(std::string plugin, std::uint32_t localFrmr) {
    RecordKey key{normalizePlugin(std::move(plugin)), localFrmr,
                  RecordKeyKind::Tes3Reference, {}, {}};
    return key;
}

RecordKey makeTes3RecordKey(std::string recordType, std::string textId) {
    RecordKey key;
    key.kind = RecordKeyKind::Tes3Named;
    key.recordType = normalizeTes3Type(std::move(recordType));
    key.textId = normalizeTes3Id(std::move(textId));
    return key;
}

bool parseRecordKey(std::string_view text, RecordKey& out) {
    if (text.starts_with("tes3:")) {
        const std::size_t separator = text.find(':', 5u);
        if (separator == std::string_view::npos || separator == 5u || separator + 1u >= text.size()) {
            return false;
        }
        std::string id;
        if (!unescapeIdentity(text.substr(separator + 1u), id)) return false;
        out = makeTes3RecordKey(std::string(text.substr(5u, separator - 5u)), std::move(id));
        return out.valid();
    }
    bool tes3Reference = false;
    if (text.starts_with("frmr:")) {
        tes3Reference = true;
        text.remove_prefix(5u);
    }
    const std::size_t separator = text.rfind(':');
    if (separator == std::string_view::npos || separator == 0u || separator + 1u >= text.size()) {
        return false;
    }
    std::string_view number = text.substr(separator + 1u);
    if (number.starts_with("0x") || number.starts_with("0X")) number.remove_prefix(2u);
    std::uint32_t localFormId = 0u;
    const auto parsed = std::from_chars(number.data(), number.data() + number.size(), localFormId, 16);
    if (parsed.ec != std::errc{} || parsed.ptr != number.data() + number.size()) return false;
    out = tes3Reference
        ? makeTes3ReferenceKey(std::string(text.substr(0u, separator)), localFormId)
        : makeRecordKey(std::string(text.substr(0u, separator)), localFormId);
    return out.valid();
}

ObjectId ObjectId::persistent(RecordKey referenceKey) {
    return ObjectId{ObjectIdKind::PersistentReference, std::move(referenceKey), 0u};
}

ObjectId ObjectId::runtime(std::uint64_t runtimeId) {
    return ObjectId{ObjectIdKind::Spawned, {}, runtimeId};
}

bool ObjectId::valid() const {
    return (kind == ObjectIdKind::PersistentReference && reference.valid()) ||
        (kind == ObjectIdKind::Spawned && spawned != 0u);
}

std::string ObjectId::toString() const {
    if (kind == ObjectIdKind::PersistentReference) return "ref:" + reference.toString();
    if (kind == ObjectIdKind::Spawned) return "runtime:" + std::to_string(spawned);
    return "invalid";
}

std::size_t RecordKeyHash::operator()(const RecordKey& key) const noexcept {
    std::uint64_t hash = hashText(key.plugin) ^ key.localFormId;
    hash ^= core::mix64(hashText(key.recordType));
    hash ^= core::mix64(hashText(key.textId));
    hash ^= static_cast<std::uint64_t>(key.kind) << 61u;
    return static_cast<std::size_t>(core::mix64(hash));
}

std::size_t ObjectIdHash::operator()(const ObjectId& id) const noexcept {
    if (id.kind == ObjectIdKind::PersistentReference) {
        return core::mix64(RecordKeyHash{}(id.reference) ^ 0x726566ull);
    }
    return core::mix64(id.spawned ^ (static_cast<std::uint64_t>(id.kind) << 60u));
}

}  // namespace odai::bethesda
