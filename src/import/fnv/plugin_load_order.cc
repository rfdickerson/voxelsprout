#include "import/fnv/plugin_load_order.h"

#include "import/fnv/content_profile.h"

#include "import/fnv/esm_reader.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string_view>
#include <system_error>
#include <unordered_map>

namespace odai::importer::fnv {

namespace {

constexpr std::uint32_t kTes4MasterFlag = 0x00000001u;
constexpr std::uint32_t kTes4LocalizedFlag = 0x00000080u;
constexpr std::uint32_t kTes4LightFlag = 0x00000200u;
// The largest record header of any supported generation: type[4], dataSize[4],
// flags[4], formId[4], versionControlInfo[4], formVersion[2], unknown[2]. Read
// this many bytes to sniff the file, then rewind to the header size the sniff
// reports — Oblivion's is 20, and reading 24 would swallow the first four bytes
// of HEDR and silently find no masters.
constexpr std::size_t kMaxRecordHeaderSize = 24u;

std::string toLowerAsciiCopy(std::string text) {
    for (char& c : text) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return text;
}

std::string trimAscii(std::string text) {
    const auto isSpace = [](unsigned char c) { return std::isspace(c) != 0; };
    while (!text.empty() && isSpace(static_cast<unsigned char>(text.front()))) {
        text.erase(text.begin());
    }
    while (!text.empty() && isSpace(static_cast<unsigned char>(text.back()))) {
        text.pop_back();
    }
    return text;
}

void appendUniquePlugin(std::vector<std::string>& plugins, const std::string& name) {
    const std::string lowered = toLowerAsciiCopy(name);
    const auto found = std::find_if(
        plugins.begin(), plugins.end(), [&](const std::string& existing) {
            return toLowerAsciiCopy(existing) == lowered;
        });
    if (found == plugins.end()) {
        plugins.push_back(name);
    }
}

bool findCaseInsensitiveFile(
    const std::filesystem::path& directory,
    const std::string& name,
    std::filesystem::path& outPath) {
    std::error_code error;
    for (std::filesystem::directory_iterator it(directory, error), end; !error && it != end;
         it.increment(error)) {
        std::error_code typeError;
        if (!it->is_regular_file(typeError) || typeError) {
            continue;
        }
        if (toLowerAsciiCopy(it->path().filename().string()) == toLowerAsciiCopy(name)) {
            outPath = it->path();
            return true;
        }
    }
    return false;
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

    // Read the widest header plus the four bytes the first subrecord type
    // occupies past it, which is what detectEsmPluginFormat compares. A short
    // read is tolerated down to the widest header: a file with fewer bytes than
    // that past its header has no subrecords to find, and failing it here would
    // reject plugins this function used to accept.
    std::uint8_t header[kMaxRecordHeaderSize + 4u] = {};
    input.read(reinterpret_cast<char*>(header), static_cast<std::streamsize>(sizeof(header)));
    const auto headerBytesRead = static_cast<std::size_t>(input.gcount());
    if (headerBytesRead < kMaxRecordHeaderSize) {
        outError = "truncated before the TES3/TES4 header: " + path.string();
        return false;
    }
    // A short read sets eofbit/failbit, and the seekg below needs a clear stream.
    input.clear();
    const EsmPluginFormat format = detectEsmPluginFormat(header, headerBytesRead);
    outHeader.format = format;
    const bool isTes3 = format == EsmPluginFormat::kMorrowind;
    if ((!isTes3 && std::memcmp(header, "TES4", 4) != 0) ||
        (isTes3 && std::memcmp(header, "TES3", 4) != 0)) {
        outError = "not a supported Bethesda plugin (no TES3/TES4 record): " + path.string();
        return false;
    }

    const std::uint32_t dataSize = readU32(header + 4);
    if (!isTes3) {
        const std::uint32_t flags = readU32(header + 8);
        outHeader.isMaster = (flags & kTes4MasterFlag) != 0u;
        outHeader.isLocalized = (flags & kTes4LocalizedFlag) != 0u;
        outHeader.isLight = (flags & kTes4LightFlag) != 0u;
    }

    // Seek to where the record body actually starts, because the read above
    // deliberately overshot. Oblivion's header is 20 bytes, so continuing
    // sequentially from 24 would swallow the first four bytes of HEDR and walk
    // the body one subrecord out of phase — which finds no MAST at all and so
    // reports a mod with masters as having none.
    const std::size_t headerSize = esmRecordHeaderSize(format);
    input.seekg(static_cast<std::streamoff>(headerSize), std::ios::beg);

    // Bound the declared size against the file BEFORE sizing anything from it.
    // --plugin-add takes an arbitrary user-named path, so a truncated download
    // or any garbage file whose first four bytes read "TES4" can declare up to
    // 0xFFFFFFFF here; sizing the vector from that asks for ~4 GB and dies on
    // bad_alloc instead of reporting "not a Fallout plugin".
    std::error_code fileSizeError;
    const auto fileSize = std::filesystem::file_size(path, fileSizeError);
    if (!fileSizeError && static_cast<std::uintmax_t>(headerSize) + dataSize > fileSize) {
        outError = std::string(isTes3 ? "TES3" : "TES4") +
            " record claims more bytes than the file holds: " + path.string();
        return false;
    }
    // The TES4 record is never compressed and is a few hundred bytes; reading
    // it is the whole cost of placing a plugin in a load order.
    std::vector<std::uint8_t> body(dataSize);
    if (dataSize != 0u &&
        !input.read(reinterpret_cast<char*>(body.data()), static_cast<std::streamsize>(dataSize))) {
        outError = std::string("truncated ") + (isTes3 ? "TES3" : "TES4") +
            " record: " + path.string();
        return false;
    }

    const std::size_t subrecordHeaderSize = isTes3 ? 8u : 6u;
    std::size_t offset = 0;
    while (offset + subrecordHeaderSize <= body.size()) {
        const char* type = reinterpret_cast<const char*>(body.data() + offset);
        const std::uint32_t size = isTes3
            ? readU32(body.data() + offset + 4)
            : static_cast<std::uint32_t>(readU16(body.data() + offset + 4));
        const std::size_t dataOffset = offset + subrecordHeaderSize;
        if (dataOffset + size > body.size()) {
            break;
        }
        if (std::memcmp(type, "HEDR", 4) == 0) {
            if (isTes3 && size >= 300u) {
                // version, file type, company[32], description[256], records.
                outHeader.isMaster = readU32(body.data() + dataOffset + 4) == 1u;
                outHeader.recordCount = readU32(body.data() + dataOffset + 296u);
            } else if (!isTes3 && size >= 8u) {
                outHeader.recordCount = readU32(body.data() + dataOffset + 4);
            }
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

bool readBethesdaPluginList(
    const std::filesystem::path& path,
    std::vector<std::string>& outActivePlugins,
    std::string& outError) {
    outActivePlugins.clear();
    outError.clear();
    std::ifstream input(path);
    if (!input) {
        outError = "cannot open load order " + path.string();
        return false;
    }
    std::string line;
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        line = trimAscii(std::move(line));
        if (line.empty() || line[0] == '#' || line[0] == ';') {
            continue;
        }
        // Skyrim SE plugins.txt keeps inactive entries without a star. Treating
        // them as active is especially dangerous for half-installed mod sets.
        if (line[0] != '*') {
            continue;
        }
        line = trimAscii(line.substr(1));
        if (!line.empty()) {
            appendUniquePlugin(outActivePlugins, line);
        }
    }
    if (!input.eof() && input.fail()) {
        outError = "failed while reading load order " + path.string();
        outActivePlugins.clear();
        return false;
    }
    return true;
}

bool resolveInstalledSkyrimPluginList(
    const std::filesystem::path& dataFilesPath,
    const std::optional<std::filesystem::path>& explicitList,
    std::vector<std::string>& outPlugins,
    std::filesystem::path& outSource,
    std::string& outError) {
    outPlugins.clear();
    outSource.clear();
    outError.clear();

    std::vector<std::filesystem::path> candidates;
    if (explicitList.has_value()) {
        candidates.push_back(*explicitList);
    } else {
        // A Steam install gives an exact Proton prefix without guessing which
        // of several user profiles belongs to this Data directory.
        const std::filesystem::path gameRoot = dataFilesPath.parent_path();
        const std::filesystem::path common = gameRoot.parent_path();
        if (toLowerAsciiCopy(common.filename().string()) == "common") {
            const std::filesystem::path steamApps = common.parent_path();
            candidates.push_back(
                steamApps / "compatdata" / "489830" / "pfx" / "drive_c" / "users" /
                "steamuser" / "AppData" / "Local" / "Skyrim Special Edition" /
                "plugins.txt");
        }
        const char* home = std::getenv("HOME");
        if (home != nullptr) {
            const std::filesystem::path homePath(home);
            for (const std::filesystem::path& steamRoot : {
                     homePath / ".local" / "share" / "Steam",
                     homePath / ".steam" / "steam"}) {
                candidates.push_back(
                    steamRoot / "steamapps" / "compatdata" / "489830" / "pfx" /
                    "drive_c" / "users" / "steamuser" / "AppData" / "Local" /
                    "Skyrim Special Edition" / "plugins.txt");
            }
            candidates.push_back(
                homePath / ".local" / "share" / "Skyrim Special Edition" / "plugins.txt");
        }
    }

    std::filesystem::path selected;
    if (explicitList.has_value()) {
        selected = candidates.front();
        std::error_code existsError;
        if (!std::filesystem::is_regular_file(selected, existsError) || existsError) {
            outError = "load order does not exist: " + selected.string();
            return false;
        }
    } else {
        // If more than one valid profile exists, the newest is the one a mod
        // manager or the game's launcher most recently wrote.
        std::filesystem::file_time_type newest{};
        bool haveNewest = false;
        for (const std::filesystem::path& candidate : candidates) {
            std::error_code stampError;
            if (!std::filesystem::is_regular_file(candidate, stampError) || stampError) {
                continue;
            }
            const auto stamp = std::filesystem::last_write_time(candidate, stampError);
            if (!stampError && (!haveNewest || stamp > newest)) {
                selected = candidate;
                newest = stamp;
                haveNewest = true;
            }
        }
    }

    // Base masters are implicit in plugins.txt. Add only what is actually
    // installed, except Skyrim.esm itself which is required.
    const std::vector<std::string> implicit = {
        "Skyrim.esm", "Update.esm", "Dawnguard.esm", "HearthFires.esm", "Dragonborn.esm"};
    for (const std::string& name : implicit) {
        std::filesystem::path found;
        if (findCaseInsensitiveFile(dataFilesPath, name, found)) {
            appendUniquePlugin(outPlugins, found.filename().string());
        } else if (name == "Skyrim.esm") {
            outError = "Skyrim.esm not found in " + dataFilesPath.string();
            return false;
        }
    }

    if (!selected.empty()) {
        std::vector<std::string> active;
        if (!readBethesdaPluginList(selected, active, outError)) {
            outPlugins.clear();
            return false;
        }
        for (const std::string& name : active) {
            appendUniquePlugin(outPlugins, name);
        }
        outSource = selected;
        return true;
    }

    // A stock Proton install often has no profile until the Bethesda launcher
    // runs. Skyrim.ccc is the authoritative Creation Club order, but it may
    // list content not installed, so include only present files.
    const std::filesystem::path cccPath = dataFilesPath.parent_path() / "Skyrim.ccc";
    std::ifstream ccc(cccPath);
    if (ccc) {
        std::string line;
        while (std::getline(ccc, line)) {
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            line = trimAscii(std::move(line));
            if (line.empty() || line[0] == '#' || line[0] == ';') {
                continue;
            }
            std::filesystem::path found;
            if (findCaseInsensitiveFile(dataFilesPath, line, found)) {
                appendUniquePlugin(outPlugins, found.filename().string());
            }
        }
        outSource = cccPath;
    } else {
        outSource = "official installed masters";
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

    std::size_t regularCount = 0u;
    std::size_t lightCount = 0u;
    for (FalloutLoadOrderEntry& entry : m_entries) {
        if (entry.header.isLight) {
            if (lightCount >= 4096u) {
                outError = "load order has more than 4096 light plugins";
                m_entries.clear();
                return false;
            }
            entry.slot = FalloutPluginSlot{
                FalloutPluginSlotKind::Light, static_cast<std::uint16_t>(lightCount++)};
        } else {
            // 0xFE is reserved for the light namespace and 0xFF for transient
            // runtime forms, leaving regular slots 0x00..0xFD.
            if (regularCount >= 254u) {
                outError = "load order has more than 254 regular plugins";
                m_entries.clear();
                return false;
            }
            entry.slot = FalloutPluginSlot{
                FalloutPluginSlotKind::Regular, static_cast<std::uint16_t>(regularCount++)};
        }
    }

    // Now that every plugin has a position, build each one's local -> global
    // map. TES4 addresses masters at 0..N-1 and self at N; TES3 addresses self
    // at zero and masters at 1..N.
    for (std::size_t i = 0; i < m_entries.size(); ++i) {
        FalloutLoadOrderEntry& entry = m_entries[i];
        entry.localToGlobal.clear();
        entry.localToGlobal.reserve(entry.header.masters.size() + 1u);
        if (entry.header.format == EsmPluginFormat::kMorrowind) {
            entry.localToGlobal.push_back(entry.slot);
        }
        for (const std::string& master : entry.header.masters) {
            const auto found = placedByLowerName.find(toLowerAsciiCopy(master));
            if (found == placedByLowerName.end()) {
                outError = "master " + master + " of " + entry.header.fileName +
                    " is not in the load order";
                m_entries.clear();
                return false;
            }
            entry.localToGlobal.push_back(m_entries[found->second].slot);
        }
        if (entry.header.format != EsmPluginFormat::kMorrowind) {
            entry.localToGlobal.push_back(entry.slot);
        }
    }
    return true;
}

bool FalloutLoadOrder::open(
    const ResolvedContentProfile& profile, std::string& outError) {
    m_searchRoots.clear();
    for (const ContentLayer& layer : profile.layers) {
        if (layer.enabled) m_searchRoots.push_back(layer.root);
    }
    return open(profile.dataRoot, profile.plugins, outError);
}

std::uint32_t FalloutLoadOrder::remapFormId(
    std::size_t pluginIndex, std::uint32_t localFormId) const {
    if (pluginIndex >= m_entries.size()) {
        return localFormId;
    }
    const FalloutLoadOrderEntry& entry = m_entries[pluginIndex];
    const std::size_t localIndex = static_cast<std::size_t>(localFormId >> 24u);
    if (localIndex >= entry.localToGlobal.size()) {
        if (entry.header.format == EsmPluginFormat::kMorrowind) {
            // TES3 treats an out-of-range content-file byte as belonging to the
            // current file. Preserve the low 24 bits and make that ownership
            // explicit in global space.
            return entry.slot.encode(localFormId & 0x00FFFFFFu);
        }
        // Not an index this plugin declares. Leaving it alone keeps the bad
        // reference identifiable instead of aliasing it onto a real record.
        return localFormId;
    }
    const FalloutPluginSlot target = entry.localToGlobal[localIndex];
    const std::uint32_t objectId = localFormId & 0x00FFFFFFu;
    if (target.kind == FalloutPluginSlotKind::Light && objectId > 0xFFFu) {
        // A light plugin cannot own that object ID. Preserve the malformed raw
        // value instead of aliasing it onto an unrelated valid record.
        return localFormId;
    }
    return target.encode(objectId);
}

const FalloutLoadOrderEntry* FalloutLoadOrder::ownerOf(std::uint32_t globalFormId) const {
    const bool light = (globalFormId >> 24u) == 0xFEu;
    const std::uint16_t slot = light
        ? static_cast<std::uint16_t>((globalFormId >> 12u) & 0x0FFFu)
        : static_cast<std::uint16_t>(globalFormId >> 24u);
    const FalloutPluginSlotKind kind =
        light ? FalloutPluginSlotKind::Light : FalloutPluginSlotKind::Regular;
    const auto found = std::find_if(
        m_entries.begin(), m_entries.end(), [&](const FalloutLoadOrderEntry& entry) {
            return entry.slot.kind == kind && entry.slot.index == slot;
        });
    return found == m_entries.end() ? nullptr : &*found;
}

std::string FalloutLoadOrder::fingerprint() const {
    // A directory component must stay short even for a six-plugin Morrowind
    // chain. Hash ordered names plus content stamps rather than concatenating
    // them; changing a file in place then selects a fresh derived-cell cache.
    std::uint64_t hash = 1469598103934665603ull;
    const auto mix = [&hash](std::string_view value) {
        for (const unsigned char c : value) {
            hash ^= c;
            hash *= 1099511628211ull;
        }
        hash ^= 0xffu;
        hash *= 1099511628211ull;
    };
    for (const FalloutLoadOrderEntry& entry : m_entries) {
        mix(toLowerAsciiCopy(entry.header.fileName));
        mix(entry.slot.kind == FalloutPluginSlotKind::Light ? "light" : "regular");
        mix(std::to_string(entry.slot.index));
        std::error_code error;
        const auto size = std::filesystem::file_size(entry.path, error);
        mix(error ? std::string("?") : std::to_string(size));
        error.clear();
        const auto stamp = std::filesystem::last_write_time(entry.path, error);
        mix(error ? std::string("?") : std::to_string(stamp.time_since_epoch().count()));
    }
    std::ostringstream result;
    result << m_entries.size() << "p_" << std::hex << std::setw(16) << std::setfill('0') << hash;
    return result.str();
}

}  // namespace odai::importer::fnv
