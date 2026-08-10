#include "import/fnv/plugin_load_order.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <fstream>
#include <system_error>
#include <unordered_map>

namespace odai::importer::fnv {

namespace {

constexpr std::uint32_t kTes4MasterFlag = 0x00000001u;
// A record header is 24 bytes: type[4], dataSize[4], flags[4], formId[4],
// versionControlInfo[4], formVersion[2], unknown[2].
constexpr std::size_t kRecordHeaderSize = 24u;

std::string toLowerAsciiCopy(std::string text) {
    for (char& c : text) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return text;
}

std::uint32_t readU32(const std::uint8_t* bytes) {
    std::uint32_t value = 0;
    std::memcpy(&value, bytes, sizeof(value));
    return value;
}

std::uint16_t readU16(const std::uint8_t* bytes) {
    std::uint16_t value = 0;
    std::memcpy(&value, bytes, sizeof(value));
    return value;
}

// Masters name a file, and on a case-sensitive filesystem the case a plugin
// declares need not match the case on disk -- "FalloutNV.esm" against
// "falloutnv.esm" is the same file everywhere the game ever ran. Resolve
// through a lowercased listing of the directory rather than trusting either
// spelling.
class DirectoryListing {
public:
    // `directories` in ASCENDING priority: each one's entries overwrite the
    // previous, so the caller passes the Data directory first and mod roots
    // after it, in load order.
    explicit DirectoryListing(const std::vector<std::filesystem::path>& directories) {
        for (const std::filesystem::path& directory : directories) {
            std::error_code iterError;
            std::filesystem::directory_iterator iterator(directory, iterError);
            if (iterError) {
                continue;
            }
            for (const auto& entry : iterator) {
                std::error_code typeError;
                if (!entry.is_regular_file(typeError) || typeError) {
                    continue;
                }
                // insert_or_assign, not emplace: a later directory is higher
                // priority and must replace what an earlier one offered.
                m_pathsByLowerName.insert_or_assign(
                    toLowerAsciiCopy(entry.path().filename().string()), entry.path());
            }
        }
    }

    // Returns the path and the on-disk spelling, or false if nothing matches.
    bool find(const std::string& fileName, std::filesystem::path& outPath) const {
        const auto found = m_pathsByLowerName.find(toLowerAsciiCopy(fileName));
        if (found == m_pathsByLowerName.end()) {
            return false;
        }
        outPath = found->second;
        return true;
    }

private:
    std::unordered_map<std::string, std::filesystem::path> m_pathsByLowerName;
};

}  // namespace

bool readFalloutPluginHeader(
    const std::filesystem::path& path, FalloutPluginHeader& outHeader, std::string& outError) {
    outError.clear();
    outHeader = FalloutPluginHeader{};
    outHeader.fileName = path.filename().string();

    std::ifstream input(path, std::ios::binary);
    if (!input) {
        outError = "cannot open " + path.string();
        return false;
    }

    std::uint8_t header[kRecordHeaderSize];
    if (!input.read(reinterpret_cast<char*>(header), kRecordHeaderSize)) {
        outError = "truncated before the TES4 header: " + path.string();
        return false;
    }
    if (std::memcmp(header, "TES4", 4) != 0) {
        outError = "not a Fallout plugin (no TES4 record): " + path.string();
        return false;
    }

    const std::uint32_t dataSize = readU32(header + 4);
    const std::uint32_t flags = readU32(header + 8);
    outHeader.isMaster = (flags & kTes4MasterFlag) != 0u;

    // The TES4 record is never compressed and is a few hundred bytes; reading
    // it is the whole cost of placing a plugin in a load order.
    std::vector<std::uint8_t> body(dataSize);
    if (dataSize != 0u &&
        !input.read(reinterpret_cast<char*>(body.data()), static_cast<std::streamsize>(dataSize))) {
        outError = "truncated TES4 record: " + path.string();
        return false;
    }

    std::size_t offset = 0;
    while (offset + 6u <= body.size()) {
        const char* type = reinterpret_cast<const char*>(body.data() + offset);
        const std::uint16_t size = readU16(body.data() + offset + 4);
        const std::size_t dataOffset = offset + 6u;
        if (dataOffset + size > body.size()) {
            break;
        }
        if (std::memcmp(type, "HEDR", 4) == 0 && size >= 8u) {
            outHeader.recordCount = readU32(body.data() + dataOffset + 4);
        } else if (std::memcmp(type, "MAST", 4) == 0) {
            std::string master(
                reinterpret_cast<const char*>(body.data() + dataOffset), size);
            // Bethesda strings are zero-terminated inside their declared size.
            const std::size_t terminator = master.find('\0');
            if (terminator != std::string::npos) {
                master.resize(terminator);
            }
            if (!master.empty()) {
                outHeader.masters.push_back(std::move(master));
            }
        }
        offset = dataOffset + size;
    }
    return true;
}

bool FalloutLoadOrder::open(
    const std::filesystem::path& dataFilesPath,
    const std::vector<std::string>& requestedFileNames,
    std::string& outError) {
    outError.clear();
    m_entries.clear();

    // Data first, then mod roots in load order, so a mod's own copy of a plugin
    // wins over the game's.
    std::vector<std::filesystem::path> searchDirectories;
    searchDirectories.push_back(dataFilesPath);
    searchDirectories.insert(searchDirectories.end(), m_searchRoots.begin(), m_searchRoots.end());
    const DirectoryListing listing(searchDirectories);

    // Position in m_entries by lowercased file name, so a plugin reached twice
    // (as a master of two different mods, say) is placed exactly once.
    std::unordered_map<std::string, std::size_t> placedByLowerName;

    // Depth-first over the master graph: a plugin is appended only after every
    // master it declares is already in the list. Cycles cannot occur in a
    // well-formed set, but a malformed one must not recurse forever, so a
    // plugin is marked in-progress before its masters are walked.
    std::unordered_map<std::string, bool> inProgress;

    struct Placer {
        const DirectoryListing& listing;
        std::vector<FalloutLoadOrderEntry>& entries;
        std::unordered_map<std::string, std::size_t>& placedByLowerName;
        std::unordered_map<std::string, bool>& inProgress;
        std::string error;

        bool place(const std::string& fileName) {
            const std::string key = toLowerAsciiCopy(fileName);
            if (placedByLowerName.count(key) != 0u) {
                return true;  // already positioned; never move it
            }
            if (inProgress[key]) {
                // A master cycle. Treat it as placed to break the recursion --
                // the formIDs will be wrong, but the alternative is a hang.
                return true;
            }
            inProgress[key] = true;

            std::filesystem::path path;
            if (!listing.find(fileName, path)) {
                if (error.empty()) {
                    error = "plugin not found: " + fileName;
                }
                inProgress[key] = false;
                return false;
            }

            FalloutPluginHeader header;
            std::string headerError;
            if (!readFalloutPluginHeader(path, header, headerError)) {
                if (error.empty()) {
                    error = headerError;
                }
                inProgress[key] = false;
                return false;
            }

            // Masters first, and in the order this plugin declares them.
            for (const std::string& master : header.masters) {
                if (!place(master)) {
                    inProgress[key] = false;
                    return false;
                }
            }

            inProgress[key] = false;
            FalloutLoadOrderEntry entry;
            entry.path = path;
            entry.header = std::move(header);
            placedByLowerName.emplace(key, entries.size());
            entries.push_back(std::move(entry));
            return true;
        }
    };

    Placer placer{listing, m_entries, placedByLowerName, inProgress, {}};
    for (const std::string& fileName : requestedFileNames) {
        if (!placer.place(fileName)) {
            outError = placer.error.empty() ? ("cannot place plugin " + fileName) : placer.error;
            m_entries.clear();
            return false;
        }
    }

    if (m_entries.size() > 256u) {
        outError = "load order has " + std::to_string(m_entries.size()) +
            " plugins; a formID mod index is one byte, so 256 is the hard ceiling";
        m_entries.clear();
        return false;
    }

    // Now that every plugin has a position, build each one's local -> global
    // map. Local index i names masters[i] for i < masters.size(), and the
    // plugin itself at i == masters.size().
    for (std::size_t i = 0; i < m_entries.size(); ++i) {
        FalloutLoadOrderEntry& entry = m_entries[i];
        entry.globalIndex = static_cast<std::uint8_t>(i);
        entry.localToGlobal.clear();
        entry.localToGlobal.reserve(entry.header.masters.size() + 1u);
        for (const std::string& master : entry.header.masters) {
            const auto found = placedByLowerName.find(toLowerAsciiCopy(master));
            if (found == placedByLowerName.end()) {
                outError = "master " + master + " of " + entry.header.fileName +
                    " is not in the load order";
                m_entries.clear();
                return false;
            }
            entry.localToGlobal.push_back(static_cast<std::uint8_t>(found->second));
        }
        entry.localToGlobal.push_back(entry.globalIndex);
    }
    return true;
}

std::uint32_t FalloutLoadOrder::remapFormId(
    std::size_t pluginIndex, std::uint32_t localFormId) const {
    if (pluginIndex >= m_entries.size()) {
        return localFormId;
    }
    const FalloutLoadOrderEntry& entry = m_entries[pluginIndex];
    const std::size_t localIndex = static_cast<std::size_t>(localFormId >> 24u);
    if (localIndex >= entry.localToGlobal.size()) {
        // Not an index this plugin declares. Leaving it alone keeps the bad
        // reference identifiable instead of aliasing it onto a real record.
        return localFormId;
    }
    return (localFormId & 0x00FFFFFFu) |
        (static_cast<std::uint32_t>(entry.localToGlobal[localIndex]) << 24u);
}

std::string FalloutLoadOrder::fingerprint() const {
    std::string result;
    for (const FalloutLoadOrderEntry& entry : m_entries) {
        if (!result.empty()) {
            result += "-";
        }
        for (char c : toLowerAsciiCopy(entry.header.fileName)) {
            const bool safe = (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9');
            result.push_back(safe ? c : '_');
        }
    }
    return result;
}

}  // namespace odai::importer::fnv
