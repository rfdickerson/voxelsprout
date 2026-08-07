// Named material library: JSON parsing, the error contract, and the index
// scheme that ties an authored name to the number stored in vertex flags.
//
// The point of most of these is the hot-reload path: it matches by NAME and
// must keep indices stable, because the index is baked into vertex bits that
// cannot be updated without re-extruding the scene. A silent renumber would
// repaint every surface in the city.

#include <cstdint>
#include <iostream>
#include <string>

#include "content/material_library.h"
#include "import/imported_material.h"

namespace {

int g_failures = 0;

void check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "[material library test] FAIL: " << message << '\n';
        ++g_failures;
    }
}

void checkNear(float actual, float expected, float epsilon, const std::string& message) {
    const float diff = actual > expected ? actual - expected : expected - actual;
    if (diff > epsilon) {
        std::cerr << "[material library test] FAIL: " << message << " (expected " << expected
                  << ", got " << actual << ")\n";
        ++g_failures;
    }
}

using odai::content::loadMaterialLibraryFromJson;
using odai::content::MaterialLibraryLoadResult;

void testParsesAndIndexes() {
    std::printf("parse + index assignment\n");
    const MaterialLibraryLoadResult r = loadMaterialLibraryFromJson(R"({
        "version": 1,
        "materials": [
            {"name": "brick_1890", "baseColor": [0.62, 0.36, 0.29], "roughness": 0.82},
            {"name": "mullion", "metallic": 0.9, "roughness": 0.34,
             "note": "ignored by the loader on purpose"},
            {"name": "neon", "emissive": [1.0, 0.35, 0.55], "emissiveStrength": 4.0}
        ]
    })");
    check(r.ok(), "a well-formed library parses without errors");
    // Slot 0 is the reserved sentinel, so three authored materials give four entries.
    check(r.materials.size() == 4, "sentinel plus three authored materials");
    check(r.materials[0].name.empty(), "slot 0 is the unnamed sentinel");
    check(r.materials[1].name == "brick_1890", "array order assigns index 1 to the first entry");
    checkNear(r.materials[1].roughness, 0.82f, 1e-6f, "roughness parsed");
    checkNear(r.materials[1].baseColorTint[0], 0.62f, 1e-6f, "baseColor parsed");
    // Unspecified fields keep the struct's defaults rather than being zeroed.
    checkNear(r.materials[1].metallic, 0.0f, 1e-6f, "unspecified metallic keeps its default");
    checkNear(r.materials[2].metallic, 0.9f, 1e-6f, "second entry parsed");
    checkNear(r.materials[3].emissiveStrength, 4.0f, 1e-6f, "emissive strength parsed");

    check(odai::content::findMaterialIndex(r.materials, "mullion") == 2u, "lookup by name");
    // A miss returns 0, which is the "no library material" sentinel -- so an
    // unresolved name degrades to legacy shading rather than a wrong material.
    check(odai::content::findMaterialIndex(r.materials, "nope") == 0u,
          "an unknown name resolves to the sentinel, not to a wrong slot");
}

void testMalformedInputIsNonFatal() {
    std::printf("malformed input\n");
    // Broken JSON: errors recorded, and the caller still gets a usable table.
    const MaterialLibraryLoadResult bad = loadMaterialLibraryFromJson("{not json");
    check(!bad.ok(), "unparseable JSON reports an error");
    check(bad.materials.size() == 1, "unparseable JSON still yields the sentinel table");

    const MaterialLibraryLoadResult noArray = loadMaterialLibraryFromJson(R"({"version": 1})");
    check(!noArray.ok(), "a missing 'materials' array reports an error");

    // One bad entry is skipped; the good ones around it still load.
    const MaterialLibraryLoadResult mixed = loadMaterialLibraryFromJson(R"({
        "materials": [
            {"name": "good_one", "roughness": 0.5},
            {"roughness": 0.2},
            {"name": "good_two", "metallic": 0.3}
        ]
    })");
    check(!mixed.ok(), "an unnamed entry reports an error");
    check(mixed.materials.size() == 3, "the unnamed entry is skipped, the others survive");
    check(mixed.materials[1].name == "good_one" && mixed.materials[2].name == "good_two",
          "surviving entries keep their order");
}

void testValidation() {
    std::printf("validation pass\n");
    const MaterialLibraryLoadResult dup = loadMaterialLibraryFromJson(R"({
        "materials": [{"name": "same"}, {"name": "same"}]
    })");
    check(!dup.ok(), "duplicate names are reported -- hot reload matches by name");

    const MaterialLibraryLoadResult range = loadMaterialLibraryFromJson(R"({
        "materials": [{"name": "wild", "metallic": 4.0, "roughness": -1.0}]
    })");
    check(!range.ok(), "out-of-range coefficients are reported");
    check(range.errors.size() >= 2, "both offending fields are named");
}

void testCapacity() {
    std::printf("capacity\n");
    std::string json = R"({"materials": [)";
    // One more than the authorable count (capacity minus the sentinel).
    for (std::uint32_t i = 0; i < odai::importer::kImportedSceneMaterialTableCapacity; ++i) {
        if (i != 0) json += ",";
        json += R"({"name": "m)" + std::to_string(i) + R"("})";
    }
    json += "]}";
    const MaterialLibraryLoadResult r = loadMaterialLibraryFromJson(json);
    check(!r.ok(), "overflowing the table reports an error rather than silently dropping");
    check(r.materials.size() == odai::importer::kImportedSceneMaterialTableCapacity,
          "the table is capped at capacity, sentinel included");
}

void testJsonRoundTrip() {
    std::printf("save round trip\n");
    const MaterialLibraryLoadResult first = loadMaterialLibraryFromJson(R"({
        "materials": [
            {"name": "a", "baseColor": [0.1, 0.2, 0.3], "metallic": 0.4, "roughness": 0.6},
            {"name": "b", "emissive": [1.0, 0.5, 0.25], "emissiveStrength": 2.5}
        ]
    })");
    check(first.ok(), "fixture parses");
    const std::string text = odai::content::materialLibraryToJson(first.materials);
    const MaterialLibraryLoadResult second = loadMaterialLibraryFromJson(text);
    check(second.ok(), "serialized output parses back");
    check(second.materials.size() == first.materials.size(), "entry count survives a round trip");
    check(second.materials[1].name == "a" && second.materials[2].name == "b",
          "names and therefore indices survive a round trip");
    checkNear(second.materials[1].metallic, 0.4f, 1e-6f, "metallic survives");
    checkNear(second.materials[2].emissiveStrength, 2.5f, 1e-6f, "emissive strength survives");
}

}  // namespace

int main() {
    testParsesAndIndexes();
    testMalformedInputIsNonFatal();
    testValidation();
    testCapacity();
    testJsonRoundTrip();

    if (g_failures != 0) {
        std::cerr << "[material library test] " << g_failures << " failures\n";
        return 1;
    }
    std::cout << "[material library test] all checks passed\n";
    return 0;
}
