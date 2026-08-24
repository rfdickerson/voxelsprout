#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace odai::anim {

struct HkxSectionView {
    std::string name;
    std::uint32_t dataStart = 0;
    std::uint32_t localFixups = 0;
    std::uint32_t globalFixups = 0;
    std::uint32_t virtualFixups = 0;
    std::uint32_t exports = 0;
    std::uint32_t imports = 0;
    std::uint32_t end = 0;
};

struct HkxObjectReference {
    std::uint32_t sourceSection = 0;
    std::uint32_t sourceOffset = 0;
    std::uint32_t targetSection = 0;
    std::uint32_t targetOffset = 0;
};

struct HkxObjectRecord {
    std::uint32_t section = 0;
    std::uint32_t offset = 0;
    std::string className;
};

enum class HkxGeneratorIdentity : std::uint8_t {
    Unknown,
    Vanilla,
    Fnis,
    Nemesis,
};

struct HkxPackfileSummary {
    std::string contentsVersion;
    std::uint8_t pointerSize = 0;
    bool littleEndian = false;
    std::vector<HkxSectionView> sections;
    std::vector<std::string> classNames;
    std::vector<std::string> unsupportedBehaviorClasses;
    std::vector<HkxObjectReference> objectReferences;
    std::vector<HkxObjectRecord> objects;
    std::size_t localFixupCount = 0;
    std::size_t globalFixupCount = 0;
    std::size_t virtualFixupCount = 0;
    bool containsSkeleton = false;
    bool containsAnimation = false;
    bool containsBehaviorGraph = false;
    bool containsPhysicsMetadata = false;
    HkxGeneratorIdentity generator = HkxGeneratorIdentity::Unknown;
};

struct HkxReadLimits {
    std::size_t maxFileBytes = 256u * 1024u * 1024u;
    std::size_t maxSections = 32u;
    std::size_t maxClassNames = 16384u;
    std::size_t maxStringBytes = 512u;
};

// Clean-room structural reader for the legacy x64 little-endian Havok
// packfiles shipped by Skyrim SE. It validates section and fixup ranges and
// inventories reflected class names, but never constructs a Havok object.
bool inspectHkxPackfile(
    std::span<const std::uint8_t> bytes, HkxPackfileSummary& out,
    std::string& outError, const HkxReadLimits& limits = {});

const char* hkxGeneratorName(HkxGeneratorIdentity generator);

}  // namespace odai::anim
