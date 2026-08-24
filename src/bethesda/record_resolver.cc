#include "bethesda/record_resolver.h"

#include <algorithm>

namespace odai::bethesda {
namespace {

std::string lower(std::string value) {
    for (char& ch : value) if (ch >= 'A' && ch <= 'Z') ch = static_cast<char>(ch - 'A' + 'a');
    return value;
}

}  // namespace

bool stableRecordKey(
    const importer::fnv::FalloutLoadOrder& loadOrder,
    std::uint32_t resolvedFormId,
    RecordKey& out,
    std::string& outError) {
    const importer::fnv::FalloutLoadOrderEntry* owner = loadOrder.ownerOf(resolvedFormId);
    if (owner == nullptr) {
        outError = "resolved form ID 0x" + std::to_string(resolvedFormId) +
            " has no load-order owner";
        return false;
    }
    const std::uint32_t local = owner->slot.kind == importer::fnv::FalloutPluginSlotKind::Light
        ? (resolvedFormId & 0x00000fffu) : (resolvedFormId & 0x00ffffffu);
    out = makeRecordKey(owner->header.fileName, local);
    if (!out.valid()) { outError = "resolved form ID maps to an invalid RecordKey"; return false; }
    outError.clear(); return true;
}

bool resolvedFormId(
    const importer::fnv::FalloutLoadOrder& loadOrder,
    const RecordKey& key,
    std::uint32_t& out,
    std::string& outError) {
    const std::string wanted = lower(key.plugin);
    const auto found = std::find_if(loadOrder.entries().begin(), loadOrder.entries().end(),
        [&](const importer::fnv::FalloutLoadOrderEntry& entry) {
            return lower(entry.header.fileName) == wanted;
        });
    if (found == loadOrder.entries().end()) {
        outError = "RecordKey requires missing plugin " + key.plugin; return false;
    }
    if (found->slot.kind == importer::fnv::FalloutPluginSlotKind::Light &&
        key.localFormId > 0x00000fffu) {
        outError = "light-plugin RecordKey exceeds the 12-bit local form range"; return false;
    }
    if (found->slot.kind == importer::fnv::FalloutPluginSlotKind::Regular &&
        key.localFormId > 0x00ffffffu) {
        outError = "RecordKey exceeds the 24-bit local form range"; return false;
    }
    out = found->slot.encode(key.localFormId);
    outError.clear(); return true;
}

}  // namespace odai::bethesda
