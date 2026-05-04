#include "render/dialogue_font_config.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <exception>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace odai::render {
namespace {

constexpr float kDefaultDialogueFontSize = 18.0f;
constexpr float kMinDialogueFontSize = 8.0f;
constexpr float kMaxDialogueFontSize = 48.0f;

std::string trim(std::string_view value) {
    std::size_t begin = 0;
    while (begin < value.size() && std::isspace(static_cast<unsigned char>(value[begin])) != 0) {
        ++begin;
    }
    std::size_t end = value.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(value[end - 1])) != 0) {
        --end;
    }
    return std::string(value.substr(begin, end - begin));
}

std::string lowerCopy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

std::string normalizedOmwSectionName(std::string value) {
    value = lowerCopy(trim(value));
    std::replace(value.begin(), value.end(), '_', ' ');
    return value;
}

bool parseFloat(std::string_view value, float& outValue) {
    try {
        std::size_t parsedLength = 0;
        const float parsed = std::stof(std::string(value), &parsedLength);
        if (parsedLength != value.size() || !std::isfinite(parsed)) {
            return false;
        }
        outValue = parsed;
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

float clampFontSize(float value) {
    if (!std::isfinite(value)) {
        return kDefaultDialogueFontSize;
    }
    return std::clamp(value, kMinDialogueFontSize, kMaxDialogueFontSize);
}

bool isFontFileExtension(const std::filesystem::path& path) {
    const std::string ext = lowerCopy(path.extension().string());
    return ext == ".ttf" || ext == ".otf";
}

bool isOmwFontExtension(const std::filesystem::path& path) {
    return lowerCopy(path.extension().string()) == ".omwfont";
}

std::optional<std::filesystem::path> findCaseInsensitiveChild(
    const std::filesystem::path& directory,
    const std::filesystem::path& filename
) {
    if (directory.empty()) {
        return std::nullopt;
    }
    const std::filesystem::path exact = directory / filename;
    if (std::filesystem::exists(exact)) {
        return exact;
    }
    if (!std::filesystem::exists(directory) || !std::filesystem::is_directory(directory) || filename.has_parent_path()) {
        return std::nullopt;
    }

    const std::string wanted = lowerCopy(filename.filename().string());
    for (const std::filesystem::directory_entry& entry : std::filesystem::directory_iterator(directory)) {
        if (lowerCopy(entry.path().filename().string()) == wanted) {
            return entry.path();
        }
    }
    return std::nullopt;
}

std::optional<std::filesystem::path> resolveFontFilePath(
    const std::filesystem::path& value,
    const std::vector<std::filesystem::path>& fontDirectories
) {
    if (value.empty()) {
        return std::nullopt;
    }
    if (value.is_absolute()) {
        if (std::filesystem::exists(value)) {
            return value;
        }
        return std::nullopt;
    }
    if (std::filesystem::exists(value)) {
        return value;
    }
    for (const std::filesystem::path& directory : fontDirectories) {
        if (std::optional<std::filesystem::path> exact = findCaseInsensitiveChild(directory, value);
            exact.has_value() && std::filesystem::exists(*exact)) {
            return exact;
        }
    }
    return std::nullopt;
}

std::vector<std::filesystem::path> candidateFontFilenames(std::string_view requestedFont) {
    std::vector<std::filesystem::path> candidates;
    const std::filesystem::path requestedPath(trim(requestedFont));
    if (requestedPath.empty()) {
        return candidates;
    }
    if (requestedPath.has_extension()) {
        candidates.push_back(requestedPath);
        return candidates;
    }
    candidates.emplace_back(std::string(requestedFont) + ".ttf");
    candidates.emplace_back(std::string(requestedFont) + ".otf");
    return candidates;
}

struct OmwFontSection {
    std::string trueTypeFont;
    std::string type;
    std::optional<float> sizePixels;
};

using OmwFontSections = std::unordered_map<std::string, OmwFontSection>;

OmwFontSections parseOmwFontFile(const std::filesystem::path& path) {
    OmwFontSections sections;
    std::ifstream file(path);
    if (!file) {
        return sections;
    }

    std::string currentSection;
    std::string line;
    while (std::getline(file, line)) {
        const std::size_t hashComment = line.find('#');
        const std::size_t semicolonComment = line.find(';');
        const std::size_t commentPos = std::min(
            hashComment == std::string::npos ? line.size() : hashComment,
            semicolonComment == std::string::npos ? line.size() : semicolonComment);
        line.erase(commentPos);
        const std::string trimmed = trim(line);
        if (trimmed.empty()) {
            continue;
        }
        if (trimmed.front() == '[' && trimmed.back() == ']' && trimmed.size() >= 2u) {
            currentSection = normalizedOmwSectionName(trimmed.substr(1, trimmed.size() - 2u));
            sections.try_emplace(currentSection);
            continue;
        }
        if (currentSection.empty()) {
            continue;
        }
        const std::size_t equals = trimmed.find('=');
        if (equals == std::string::npos) {
            continue;
        }
        const std::string key = lowerCopy(trim(std::string_view(trimmed).substr(0, equals)));
        const std::string value = trim(std::string_view(trimmed).substr(equals + 1u));
        OmwFontSection& section = sections[currentSection];
        if (key == "truetype font") {
            section.trueTypeFont = value;
            continue;
        }
        if (key == "type") {
            section.type = lowerCopy(value);
            continue;
        }
        if (key == "size") {
            float parsed = 0.0f;
            if (parseFloat(value, parsed)) {
                section.sizePixels = parsed;
            }
            continue;
        }
    }
    return sections;
}

std::vector<std::filesystem::path> findOmwFontFiles(const std::vector<std::filesystem::path>& fontDirectories) {
    std::vector<std::filesystem::path> files;
    for (const std::filesystem::path& directory : fontDirectories) {
        if (!std::filesystem::exists(directory) || !std::filesystem::is_directory(directory)) {
            continue;
        }
        for (const std::filesystem::directory_entry& entry : std::filesystem::directory_iterator(directory)) {
            if (entry.is_regular_file() && isOmwFontExtension(entry.path())) {
                files.push_back(entry.path());
            }
        }
    }
    return files;
}

std::vector<std::string> sectionFallbacks(const std::string& requestedFont) {
    std::vector<std::string> sections;
    if (!requestedFont.empty()) {
        sections.push_back(normalizedOmwSectionName(requestedFont));
    }
    sections.emplace_back("journalbook");
    sections.emplace_back("default");
    return sections;
}

bool shouldUseSystemMorrowindFallback(const std::string& requestedFont) {
    if (requestedFont.empty()) {
        return true;
    }
    const std::string section = normalizedOmwSectionName(requestedFont);
    return section == "default" ||
           section == "journalbook" ||
           section == "bookletter" ||
           section == "magic cards" ||
           section == "century gothic";
}

std::optional<DialogueFontConfig> resolveFromOmwFontFile(
    const std::filesystem::path& omwFontPath,
    const std::vector<std::string>& sections,
    const std::optional<float> requestedSizePixels
) {
    const OmwFontSections parsed = parseOmwFontFile(omwFontPath);
    for (const std::string& sectionName : sections) {
        const auto found = parsed.find(sectionName);
        if (found == parsed.end()) {
            continue;
        }
        const OmwFontSection& section = found->second;
        if (section.trueTypeFont.empty()) {
            continue;
        }
        if (!section.type.empty() && section.type != "freetype") {
            continue;
        }
        const std::optional<std::filesystem::path> fontPath =
            findCaseInsensitiveChild(omwFontPath.parent_path(), section.trueTypeFont);
        if (!fontPath.has_value() || !std::filesystem::exists(*fontPath) || !isFontFileExtension(*fontPath)) {
            continue;
        }
        DialogueFontConfig config{};
        config.enabled = true;
        config.fontPath = *fontPath;
        config.sizePixels = clampFontSize(
            requestedSizePixels.value_or(section.sizePixels.value_or(kDefaultDialogueFontSize)));
        config.sourceLabel = omwFontPath.string() + "[" + sectionName + "]";
        return config;
    }
    return std::nullopt;
}

std::optional<DialogueFontConfig> resolveSystemMorrowindFontFallback(const std::optional<float> requestedSizePixels) {
    constexpr std::array<const char*, 5> kCandidatePaths = {
        "C:/Windows/Fonts/GOTHIC.TTF",
        "C:/Windows/Fonts/GOTHICB.TTF",
        "C:/Windows/Fonts/CENTURY.TTF",
        "/mnt/c/Windows/Fonts/GOTHIC.TTF",
        "/mnt/c/Windows/Fonts/CENTURY.TTF"
    };
    for (const char* candidate : kCandidatePaths) {
        const std::filesystem::path path(candidate);
        if (!std::filesystem::exists(path)) {
            continue;
        }
        DialogueFontConfig config{};
        config.enabled = true;
        config.fontPath = path;
        config.sizePixels = clampFontSize(requestedSizePixels.value_or(19.0f));
        config.sourceLabel = "system Morrowind-style fallback";
        return config;
    }
    return std::nullopt;
}

} // namespace

DialogueFontConfig resolveDialogueFontConfig(const DialogueFontResolveInput& input) {
    DialogueFontConfig fallback{};
    const std::string requestedFont = trim(input.requestedFont);
    const std::optional<float> requestedSize =
        input.requestedSizePixels.has_value()
            ? std::optional<float>(clampFontSize(*input.requestedSizePixels))
            : std::nullopt;

    if (!requestedFont.empty()) {
        const std::filesystem::path requestedPath(requestedFont);
        if (isFontFileExtension(requestedPath)) {
            if (std::optional<std::filesystem::path> resolved =
                    resolveFontFilePath(requestedPath, input.fontDirectories);
                resolved.has_value()) {
                DialogueFontConfig config{};
                config.enabled = true;
                config.fontPath = *resolved;
                config.sizePixels = requestedSize.value_or(kDefaultDialogueFontSize);
                config.sourceLabel = requestedFont;
                return config;
            }
        }

        if (isOmwFontExtension(requestedPath)) {
            if (std::optional<std::filesystem::path> resolved =
                    resolveFontFilePath(requestedPath, input.fontDirectories);
                resolved.has_value()) {
                if (std::optional<DialogueFontConfig> config =
                        resolveFromOmwFontFile(
                            *resolved,
                            sectionFallbacks(std::string{}),
                            requestedSize);
                    config.has_value()) {
                    return *config;
                }
            }
        }

        for (const std::filesystem::path& candidate : candidateFontFilenames(requestedFont)) {
            if (std::optional<std::filesystem::path> resolved =
                    resolveFontFilePath(candidate, input.fontDirectories);
                resolved.has_value() && isFontFileExtension(*resolved)) {
                DialogueFontConfig config{};
                config.enabled = true;
                config.fontPath = *resolved;
                config.sizePixels = requestedSize.value_or(kDefaultDialogueFontSize);
                config.sourceLabel = requestedFont;
                return config;
            }
        }
    }

    const std::vector<std::string> sections = sectionFallbacks(requestedFont);
    for (const std::filesystem::path& omwFontPath : findOmwFontFiles(input.fontDirectories)) {
        if (std::optional<DialogueFontConfig> config =
                resolveFromOmwFontFile(omwFontPath, sections, requestedSize);
            config.has_value()) {
            return *config;
        }
    }

    if (shouldUseSystemMorrowindFallback(requestedFont)) {
        if (std::optional<DialogueFontConfig> config = resolveSystemMorrowindFontFallback(requestedSize);
            config.has_value()) {
            return *config;
        }
    }

    return fallback;
}

} // namespace odai::render
