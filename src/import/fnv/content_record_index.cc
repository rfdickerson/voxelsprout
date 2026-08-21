#include "import/fnv/content_record_index.h"

#include "import/fnv/esm_reader.h"

namespace odai::importer::fnv {

bool ContentRecordIndex::build(const FalloutLoadOrder& order, std::string& outError) {
    outError.clear();
    m_versions.clear();
    m_overrideCount = 0u;
    m_deletionCount = 0u;
    for (std::size_t pluginIndex = 0; pluginIndex < order.entries().size(); ++pluginIndex) {
        const FalloutLoadOrderEntry& entry = order.entries()[pluginIndex];
        // TES3 identifies most records by string IDs rather than numeric form
        // IDs. Its world merge remains handled by FalloutCellIndex; recording
        // thousands of form ID zero entries here would invent conflicts.
        if (entry.header.format == EsmPluginFormat::kMorrowind) continue;
        EsmReader reader;
        if (!reader.open(entry.path)) {
            outError = reader.lastError();
            m_versions.clear();
            return false;
        }
        EsmReader::Visitor visitor;
        visitor.onRecordHeader = [&](const EsmRecordHeaderView& header) {
            if (header.formId == 0u || header.type == "TES4") return false;
            const std::uint32_t global = order.remapFormId(pluginIndex, header.formId);
            auto& chain = m_versions[global];
            if (!chain.empty()) ++m_overrideCount;
            const bool deleted = (header.flags & 0x00000020u) != 0u;
            if (deleted) ++m_deletionCount;
            chain.push_back(ContentRecordVersion{
                global, std::string(header.type), pluginIndex, entry.header.fileName,
                entry.path, header.flags, deleted});
            return false;
        };
        if (!reader.walk(visitor)) {
            outError = reader.lastError();
            m_versions.clear();
            return false;
        }
    }
    return true;
}

const std::vector<ContentRecordVersion>* ContentRecordIndex::versions(
    std::uint32_t globalFormId) const {
    const auto found = m_versions.find(globalFormId);
    return found == m_versions.end() ? nullptr : &found->second;
}

}  // namespace odai::importer::fnv
