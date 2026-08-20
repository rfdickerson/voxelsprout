#pragma once

// Load order for Fallout ESM/ESP plugins, and the formID remapping that makes
// more than one of them loadable at once.
//
// THE PROBLEM THIS SOLVES. Every formID's high byte is a mod index, but the
// index stored inside a plugin file is LOCAL to that plugin: it indexes that
// plugin's own TES4 master list, and the value one past the last master means
// "a record I define myself". The same byte means different things in different
// files. NevadaSkies.esp declares five masters (FalloutNV + the four DLC), so
// inside that file 0x00 is FalloutNV, 0x04 is LonesomeRoad, and 0x05 is Nevada
// Skies' own 375 new weather records. Load it after a different set of plugins
// and every one of those bytes has to mean something else.
//
// So a formID is only meaningful once remapped into the global load order.
// Skipping this does not fail loudly -- it silently resolves references to the
// wrong records, which is far worse than an error.
//
// Everything here is pure CPU and touches only each plugin's TES4 header (a few
// hundred bytes at the front of the file), never its contents. Reading
// FalloutNV.esm's masters does not map its 234 MB.

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "import/fnv/esm_reader.h"

namespace odai::importer::fnv {

struct ResolvedContentProfile;

// A plugin's TES3/TES4 header, which is all that is needed to place it in a
// load order.
struct FalloutPluginHeader {
    std::string fileName;  // "NevadaSkies.esp", as the masters of other plugins spell it
    std::vector<std::string> masters;  // MAST subrecords, in declared order
    std::uint32_t recordCount = 0;     // HEDR
    bool isMaster = false;             // TES4 flag 0x1: an .esm-style master
    // TES4 flag 0x200. Skyrim light plugins share the 0xFE global namespace:
    // twelve bits select the plugin and twelve bits select one of its records.
    bool isLight = false;
    // TES4 flag 0x80: the plugin's player-facing text lives in side files under
    // `Strings\` and every lstring subrecord (FULL, RDMP, DESC, NAM1) holds a
    // four-byte string ID instead of the text. Set for Skyrim.esm and clear for
    // every Fallout 3 / New Vegas / Oblivion plugin. See strings_table.h --
    // reading one of those IDs as a zstring succeeds and returns a plausible
    // one-character name, so nothing downstream can detect this on its own.
    bool isLocalized = false;
    // TES3 and TES4 use different master-index conventions. Keeping the
    // container generation on the header makes remapping an explicit property
    // of the file rather than an inference from its extension.
    EsmPluginFormat format = EsmPluginFormat::kFallout3;
};

enum class FalloutPluginSlotKind : std::uint8_t {
    Regular,
    Light,
};

struct FalloutPluginSlot {
    FalloutPluginSlotKind kind = FalloutPluginSlotKind::Regular;
    std::uint16_t index = 0;

    [[nodiscard]] std::uint32_t encode(std::uint32_t objectId) const {
        if (kind == FalloutPluginSlotKind::Light) {
            return 0xFE000000u | (static_cast<std::uint32_t>(index) << 12u) |
                (objectId & 0x00000FFFu);
        }
        return (static_cast<std::uint32_t>(index) << 24u) | (objectId & 0x00FFFFFFu);
    }
};

// Reads just the TES3/TES4 record at the front of `path`. Returns false if the
// file cannot be read or does not begin with a supported header record.
bool readFalloutPluginHeader(
    const std::filesystem::path& path, FalloutPluginHeader& outHeader, std::string& outError);

// One plugin, placed.
struct FalloutLoadOrderEntry {
    std::filesystem::path path;
    FalloutPluginHeader header;
    // This plugin's global regular/light slot. Load-order position is kept by
    // the vector itself and is deliberately distinct from the encoded slot:
    // light plugins do not consume one of the 254 regular slots.
    FalloutPluginSlot slot;
    // Local mod index -> global slot. TES4 stores masters first and self
    // last; TES3 stores self at zero and masters at 1..N. The vector is laid
    // out exactly as the source format addresses it.
    std::vector<FalloutPluginSlot> localToGlobal;
};

// Reads a Skyrim Special Edition plugins.txt. Only '*' lines are active;
// blank lines and comments are ignored. The caller supplies implicit base/DLC
// masters separately.
bool readBethesdaPluginList(
    const std::filesystem::path& path,
    std::vector<std::string>& outActivePlugins,
    std::string& outError);

// Resolves the active Skyrim plugin names without loading their records.
// `explicitList` wins. Otherwise standard native/Proton profile locations are
// tried, followed by the installed base/DLC files and present Skyrim.ccc
// entries. `outSource` names the selected profile or Skyrim.ccc fallback.
bool resolveInstalledSkyrimPluginList(
    const std::filesystem::path& dataFilesPath,
    const std::optional<std::filesystem::path>& explicitList,
    std::vector<std::string>& outPlugins,
    std::filesystem::path& outSource,
    std::string& outError);

class FalloutLoadOrder {
public:
    // Extra directories searched for plugins, ahead of the Data directory and in
    // the order added (later wins), matching how mod roots resolve assets.
    //
    // This is what lets a plugin live in the same mod directory as the BSA it
    // ships with, instead of having to be copied into the game install or into a
    // symlink farm mirroring it. Masters still resolve from wherever they are, so
    // a mod directory holding nothing but the .esp works.
    //
    // Call before open(); open() does not clear the list.
    void addSearchRoot(std::filesystem::path directory) {
        m_searchRoots.push_back(std::move(directory));
    }

    // Resolves `requestedFileNames` (in the order the user wants them loaded)
    // against `dataFilesPath`, pulling in every master each one declares.
    //
    // Masters are inserted ahead of the plugin that needs them, so asking for
    // just "NevadaSkies.esp" yields FalloutNV.esm, the four DLC masters, then
    // NevadaSkies.esp -- which is what the game would load and what makes the
    // plugin's formIDs resolvable at all. A plugin already present keeps its
    // first position rather than being moved, so an explicit order the caller
    // gave is never silently rearranged.
    //
    // Returns false if a requested plugin or a declared master is missing;
    // outError names the first one, because "load order is wrong" is useless
    // and "DeadMoney.esm not found in <dir>" is actionable.
    bool open(
        const std::filesystem::path& dataFilesPath,
        const std::vector<std::string>& requestedFileNames,
        std::string& outError);

    // Places exactly the plugins from an immutable resolved content graph and
    // uses its ordered enabled layers as plugin search roots.
    bool open(const ResolvedContentProfile& profile, std::string& outError);

    [[nodiscard]] const std::vector<FalloutLoadOrderEntry>& entries() const { return m_entries; }
    [[nodiscard]] std::size_t size() const { return m_entries.size(); }
    [[nodiscard]] bool empty() const { return m_entries.empty(); }

    // Rewrites `localFormId`'s mod index from plugin `pluginIndex`'s local
    // space into the global one. Returns the formID unchanged when the local
    // index is not one this plugin declares -- a malformed reference is not
    // worth inventing a target for.
    [[nodiscard]] std::uint32_t remapFormId(std::size_t pluginIndex, std::uint32_t localFormId) const;

    // Finds which plugin owns a remapped global formID. Used for localized
    // string tables, whose IDs remain local to their source plugin.
    [[nodiscard]] const FalloutLoadOrderEntry* ownerOf(std::uint32_t globalFormId) const;

    // A stable identifier for the whole load order, for folding into a cache
    // key. Changing which plugins are loaded, or their order, changes this.
    [[nodiscard]] std::string fingerprint() const;

private:
    std::vector<std::filesystem::path> m_searchRoots;  // ahead of Data; later wins
    std::vector<FalloutLoadOrderEntry> m_entries;
};

}  // namespace odai::importer::fnv
