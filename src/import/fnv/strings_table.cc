#include "import/fnv/strings_table.h"

#include <cctype>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "core/log.h"
#include "import/fnv/asset_source.h"

namespace odai::importer::fnv {
namespace {

std::uint32_t readU32(const std::uint8_t* data) {
    std::uint32_t value = 0;
    std::memcpy(&value, data, sizeof(value));
    return value;
}

std::string toLowerAsciiCopy(std::string text) {
    for (char& c : text) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return text;
}

}  // namespace

bool FalloutStringTable::loadFromBytes(
    const std::uint8_t* bytes,
    std::size_t size,
    FalloutStringFileKind kind,
    std::string& outError) {
    m_stringsById.clear();
    // count[4], dataSize[4], then count directory rows of {id[4], offset[4]},
    // then dataSize bytes of payload. Offsets are relative to the START of the
    // payload block, not to the file.
    constexpr std::size_t kHeaderSize = 8u;
    constexpr std::size_t kDirectoryRowSize = 8u;
    if (bytes == nullptr || size < kHeaderSize) {
        outError = "string table is shorter than its 8-byte header";
        return false;
    }
    const std::uint32_t count = readU32(bytes);
    const std::uint32_t dataSize = readU32(bytes + 4);
    // Sized from the file's own declaration, so bound it against the file
    // before allocating anything: this is read from an arbitrary BSA entry.
    const std::size_t directoryBytes = static_cast<std::size_t>(count) * kDirectoryRowSize;
    if (directoryBytes / kDirectoryRowSize != count) {
        outError = "string table directory size overflows";
        return false;
    }
    const std::size_t payloadStart = kHeaderSize + directoryBytes;
    if (payloadStart > size || dataSize > size - payloadStart) {
        outError = "string table declares more bytes than the file holds";
        return false;
    }
    const std::uint8_t* payload = bytes + payloadStart;
    const bool lengthPrefixed = (kind != FalloutStringFileKind::Strings);

    m_stringsById.reserve(count);
    for (std::uint32_t row = 0; row < count; ++row) {
        const std::uint8_t* entry = bytes + kHeaderSize + (static_cast<std::size_t>(row) * kDirectoryRowSize);
        const std::uint32_t stringId = readU32(entry);
        const std::uint32_t offset = readU32(entry + 4);
        if (offset >= dataSize) {
            continue;  // one bad row must not cost every other name
        }
        const std::uint8_t* cursor = payload + offset;
        std::size_t available = dataSize - offset;
        if (lengthPrefixed) {
            if (available < sizeof(std::uint32_t)) {
                continue;
            }
            const std::uint32_t declared = readU32(cursor);
            cursor += sizeof(std::uint32_t);
            available -= sizeof(std::uint32_t);
            if (declared > available) {
                continue;
            }
            available = declared;
        }
        // Both kinds terminate at the first NUL: a length-prefixed payload
        // counts the terminator in its declared size, so trimming at it is
        // correct for either.
        const auto* terminator =
            static_cast<const std::uint8_t*>(std::memchr(cursor, 0, available));
        const std::size_t length =
            (terminator == nullptr) ? available : static_cast<std::size_t>(terminator - cursor);
        m_stringsById.emplace(stringId, std::string(reinterpret_cast<const char*>(cursor), length));
    }
    return true;
}

const std::string* FalloutStringTable::find(std::uint32_t stringId) const {
    const auto found = m_stringsById.find(stringId);
    return found == m_stringsById.end() ? nullptr : &found->second;
}

bool loadFalloutStringTable(
    const FalloutAssetSource& assets,
    const std::string& pluginFileName,
    const std::string& language,
    FalloutStringFileKind kind,
    FalloutStringTable& outTable,
    std::string& outError) {
    std::string base = toLowerAsciiCopy(pluginFileName);
    const std::size_t dot = base.find_last_of('.');
    if (dot != std::string::npos) {
        base.erase(dot);
    }
    const char* extension = ".strings";
    switch (kind) {
        case FalloutStringFileKind::DlStrings: extension = ".dlstrings"; break;
        case FalloutStringFileKind::IlStrings: extension = ".ilstrings"; break;
        case FalloutStringFileKind::Strings: break;
    }
    const std::string virtualPath = "strings\\" + base + "_" + language + extension;
    std::vector<std::uint8_t> bytes;
    if (!assets.resolveAsset(virtualPath, bytes, outError)) {
        return false;
    }
    return outTable.loadFromBytes(bytes.data(), bytes.size(), kind, outError);
}

const std::string& falloutStringLanguage() {
    static const std::string s_language = []() {
        const char* env = std::getenv("ODAI_FNV_LANGUAGE");
        if (env == nullptr || env[0] == '\0') {
            return std::string("english");
        }
        return toLowerAsciiCopy(std::string(env));
    }();
    return s_language;
}

}  // namespace odai::importer::fnv
