#pragma once

// Localized string tables: the side files a LOCALIZED plugin keeps its player-
// facing text in.
//
// WHY THIS EXISTS. Fallout 3 and New Vegas store a region's map name, an NPC's
// FULL name and a dialogue response as literal text inside the plugin. Skyrim
// does not. Skyrim.esm sets TES4 flag 0x80 ("localized") and every one of those
// subrecords becomes a FOUR-BYTE STRING ID indexing a side file under
// `Strings\`, which for Skyrim SE lives inside `Skyrim - Interface.bsa`.
//
// Reading one of those as a zstring does not fail -- it returns the low bytes of
// the ID as characters, stopping at whatever byte happens to be zero. Whiterun's
// weather region stores RDMP = `68 10 01 00`, so the banner announced "h". That
// is the whole failure mode: a plausible-looking short string, never an error,
// and it looks like a truncation bug rather than a format difference.
//
// Three files carry the text, split by how long the strings are:
//
//   .STRINGS    null-terminated, no length prefix -- FULL, RDMP, SHRT, the
//               short player-facing names. This is the one anything here needs.
//   .DLSTRINGS  length-prefixed -- DESC, book text.
//   .ILSTRINGS  length-prefixed -- INFO NAM1 dialogue responses.
//
// All three share one directory layout, so `lengthPrefixed` is the only
// difference in the parse.

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>

namespace odai::importer::fnv {

class FalloutAssetSource;

// Which of the three files a blob came from. The directory is identical; only
// whether the payload carries a uint32 byte count in front of it differs.
enum class FalloutStringFileKind : std::uint8_t {
    Strings,    // .STRINGS   -- null-terminated payloads
    DlStrings,  // .DLSTRINGS -- uint32 length, then the bytes
    IlStrings,  // .ILSTRINGS -- uint32 length, then the bytes
};

// String ID -> text, for one plugin. IDs are LOCAL TO THE PLUGIN that stored
// them, exactly like the mod index in a formID, so one table per plugin.
class FalloutStringTable {
public:
    // Parses a whole .STRINGS/.DLSTRINGS/.ILSTRINGS blob. Returns false and
    // writes outError when the directory does not fit the blob; entries whose
    // individual offsets are out of range are skipped rather than failing the
    // file, because one bad row should not cost every other name.
    bool loadFromBytes(
        const std::uint8_t* bytes,
        std::size_t size,
        FalloutStringFileKind kind,
        std::string& outError);

    // Null when the ID is not in this table. A miss is ordinary: a plugin that
    // overrides one record still indexes only its own strings.
    [[nodiscard]] const std::string* find(std::uint32_t stringId) const;

    [[nodiscard]] bool empty() const { return m_stringsById.empty(); }
    [[nodiscard]] std::size_t size() const { return m_stringsById.size(); }

private:
    std::unordered_map<std::uint32_t, std::string> m_stringsById;
};

// Loads `strings\<plugin base name>_<language>.strings` (or the DL/IL variant)
// through `assets`, which resolves loose files and BSAs with the usual mod
// precedence.
//
// `pluginFileName` is the plugin as the load order spells it ("Skyrim.esm");
// the extension is dropped to form the base name. `language` is lowercase
// ("english"). Returns false with outError set when the file cannot be found or
// does not parse -- callers treat that as "leave the raw text alone" rather than
// as fatal, since a plugin may legitimately ship no table for a language.
bool loadFalloutStringTable(
    const FalloutAssetSource& assets,
    const std::string& pluginFileName,
    const std::string& language,
    FalloutStringFileKind kind,
    FalloutStringTable& outTable,
    std::string& outError);

// The language to load, from ODAI_FNV_LANGUAGE, lowercased; "english" when
// unset. Read once.
[[nodiscard]] const std::string& falloutStringLanguage();

}  // namespace odai::importer::fnv
