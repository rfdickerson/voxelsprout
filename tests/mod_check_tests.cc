#include "import/fnv/mod_check.h"
#include "import/fnv/plugin_load_order.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <nlohmann/json.hpp>
#include <zlib.h>

namespace {
namespace fs = std::filesystem;
using namespace odai::importer::fnv;
int failures = 0;
void check(bool ok, const std::string& name) { if (!ok) { std::cerr << "FAIL: " << name << '\n'; ++failures; } }
void u32(std::string& bytes, std::uint32_t n) { for (unsigned i = 0; i < 4; ++i) bytes += static_cast<char>(n >> (8u * i)); }
std::string sub(const std::string& type, const std::string& data) {
    std::string bytes = type;
    bytes += static_cast<char>(data.size()); bytes += static_cast<char>(data.size() >> 8u);
    return bytes + data;
}
std::string record(const std::string& type, std::uint32_t id, const std::string& body = {}, std::uint32_t flags = 0) {
    std::string bytes = type; u32(bytes, static_cast<std::uint32_t>(body.size())); u32(bytes, flags); u32(bytes, id); u32(bytes, 0); u32(bytes, 0);
    return bytes + body;
}
std::string plugin(const std::vector<std::string>& masters = {}, std::uint32_t flags = 0, const std::string& records = {}) {
    auto body = sub("HEDR", std::string(12, '\0'));
    for (const auto& master : masters) body += sub("MAST", master + '\0') + sub("DATA", std::string(8, '\0'));
    return record("TES4", 0, body, flags) + records;
}
void write(const fs::path& path, const std::string& data) { fs::create_directories(path.parent_path()); std::ofstream(path, std::ios::binary) << data; }
std::string read(const fs::path& path) { std::ifstream f(path, std::ios::binary); return {std::istreambuf_iterator<char>(f), {}}; }
bool has(const ModCheckReport& r, const std::string& code) { for (const auto& d : r.diagnostics) if (d.code == code) return true; return false; }
ResolvedContentProfile profile(const fs::path& root, std::vector<std::string> names) {
    ResolvedContentProfile p; p.game = BethesdaGame::SkyrimSpecialEdition; p.dataRoot = root; p.plugins = std::move(names); return p;
}
void testOrders(const fs::path& root) {
    write(root / "Base.esm", plugin());
    write(root / "Patch.esp", plugin({"Base.esm"}));
    write(root / "Independent.esp", plugin());
    auto p = profile(root, {"patch.ESP", "Independent.esp", "BASE.esm"});
    const auto r = checkSkyrimMods(p, false);
    check(!r.hasErrors(), "valid graph");
    check(r.resolvedOrder == std::vector<std::string>({"Independent.esp", "Base.esm", "Patch.esp"}), "stable available-node priority");
    check(r.resolvedOrder == checkSkyrimMods(p, false).resolvedOrder, "deterministic sort");
    p.plugins = {"Patch.esp"};
    check(checkSkyrimMods(p, false).changes.front().find("insert Base.esm") == 0, "inserted master reported");
    p.plugins = {"Base.esm", "Independent.esp", "Patch.esp"};
    check(checkSkyrimMods(p, false).changes.empty(), "valid explicit order preserved");
    write(root / "Missing.esp", plugin({"Absent.esm"}));
    p.plugins = {"Missing.esp"};
    check(has(checkSkyrimMods(p), "missing-plugin"), "missing master");
    write(root / "A.esp", plugin({"B.esp"})); write(root / "B.esp", plugin({"A.esp"}));
    p.plugins = {"A.esp"};
    auto cycle = checkSkyrimMods(p);
    check(has(cycle, "dependency-cycle") && cycle.diagnostics[0].message.find("A.esp -> B.esp -> A.esp") != std::string::npos, "full cycle chain");
    FalloutLoadOrder runtime; std::string error;
    check(!runtime.open(p, error) && runtime.empty(), "runtime rejects same cycle");
    write(root / "A.esp", plugin({"A.esp"}));
    check(has(checkSkyrimMods(p), "dependency-cycle"), "self cycle");
    write(root / "low/Patch.esp", plugin({"Absent.esm"}));
    write(root / "high/Patch.esp", plugin({"Base.esm"}));
    p.plugins = {"Patch.esp"};
    ContentLayer layer; layer.root = root / "low"; p.layers.push_back(layer);
    layer.root = root / "high"; p.layers.push_back(layer);
    check(!checkSkyrimMods(p).hasErrors(), "later search root wins");
    p.layers.back().enabled = false;
    check(has(checkSkyrimMods(p), "missing-plugin"), "disabled root ignored");
    auto oblivion = plugin(); oblivion.erase(20, 4); write(root / "Old.esp", oblivion);
    check(has(checkSkyrimMods(profile(root, {"Old.esp"})), "incompatible-layout"), "Skyrim rejects Oblivion layout");
    check(!runtime.open(root, {"Base.esm", "Old.esp"}, error), "mixed layouts rejected");
}
void testIntegrity(const fs::path& root) {
    auto run = [&](const std::string& bytes) { write(root / "Test.esp", bytes); return checkSkyrimMods(profile(root, {"Test.esp"})); };
    std::string extendedSize; u32(extendedSize, 70000);
    const auto extendedHeader = sub("HEDR", std::string(12, 0)) + sub("XXXX", extendedSize) + std::string("ONAM\0\0", 6) + std::string(70000, 0);
    check(!run(record("TES4", 0, extendedHeader)).hasErrors(), "extended header subrecords supported");
    check(has(run("TES4"), "malformed-header"), "truncated plugin header");
    check(has(run(record("TES4", 0, sub("HEDR", "x"))), "malformed-header"), "short HEDR");
    check(has(run(record("TES4", 0, sub("HEDR", std::string(12, 0)) + "x")), "malformed-header"), "trailing header bytes");
    check(has(run(plugin({}, 0, "GRUP")), "record-integrity"), "truncated group");
    std::string group = "GRUP"; u32(group, 1000); group += std::string(16, 0);
    check(has(run(plugin({}, 0, group)), "record-integrity"), "group exceeds file");
    auto truncated = record("STAT", 1, "abc"); truncated.pop_back();
    check(has(run(plugin({}, 0, truncated)), "record-integrity"), "record exceeds file");
    auto partial = run(plugin({}, 0, record("STAT", 1, "x")));
    check(has(partial, "record-integrity"), "partial subrecord rejected");
    check(partial.diagnostics[0].fileOffset == plugin().size() && partial.diagnostics[0].formId == 1, "record error location");
    check(has(run(plugin({}, 0, record("STAT", 1, sub("XXXX", std::string(4, 0))))), "record-integrity"), "dangling XXXX rejected");
    std::string broken; u32(broken, 10); broken += "garbage";
    check(has(run(plugin({}, 0, record("STAT", 1, broken, 0x40000))), "record-integrity"), "broken compression");
    auto duplicate = run(plugin({}, 0, record("STAT", 1) + record("STAT", 1)));
    check(has(duplicate, "duplicate-record"), "duplicate IDs");
    check(has(run(plugin({}, 0, record("STAT", 0x01000001))), "invalid-master-index"), "undeclared local index");
    check(has(run(plugin({}, 0x200, record("STAT", 0x1000))), "invalid-light-id"), "light object overflow");
    auto deleted = run(plugin({}, 0, record("REFR", 1, {}, 0x20) + record("NAVM", 2, {}, 0x20)));
    check(!deleted.hasErrors() && has(deleted, "deleted-reference") && has(deleted, "deleted-navmesh"), "deleted records are warnings");
    const std::string payload = sub("EDID", "example");
    uLongf size = compressBound(payload.size()); std::string compressed(size, '\0');
    compress(reinterpret_cast<Bytef*>(compressed.data()), &size, reinterpret_cast<const Bytef*>(payload.data()), payload.size());
    compressed.resize(size); std::string prefix; u32(prefix, static_cast<std::uint32_t>(payload.size()));
    check(!run(plugin({}, 0, record("STAT", 1, prefix + compressed, 0x40000))).hasErrors(), "valid compression");
    const auto missingTrailer = compressed.substr(0, compressed.size() - 4);
    check(has(run(plugin({}, 0, record("STAT", 1, prefix + missingTrailer, 0x40000))), "record-integrity"), "truncated stream rejected despite complete payload");
    std::string hugePrefix; u32(hugePrefix, 0xFFFFFFFFu);
    check(has(run(plugin({}, 0, record("STAT", 1, hugePrefix + "garbage", 0x40000))), "record-integrity"), "corrupt huge prefix does not allocate declared size");
    compressed.back() ^= 1;
    check(has(run(plugin({}, 0, record("STAT", 1, prefix + compressed, 0x40000))), "compression-checksum"), "tolerated checksum is reported");
    write(root / "Base.esm", plugin({}, 0, record("STAT", 1)));
    write(root / "Override.esp", plugin({"Base.esm"}, 0, record("STAT", 1)));
    auto overrides = checkSkyrimMods(profile(root, {"Override.esp"}));
    check(!overrides.hasErrors() && overrides.overrideCount == 1, "normal override counted");
}
void testLimits(const fs::path& root) {
    auto p = profile(root, {});
    for (unsigned i = 0; i < 255; ++i) {
        auto name = "R" + std::to_string(i) + ".esp"; write(root / name, plugin()); p.plugins.push_back(name);
    }
    check(has(checkSkyrimMods(p, false), "slot-limit"), "regular slot overflow");
    p.plugins.pop_back(); check(!checkSkyrimMods(p, false).hasErrors(), "254 regular slots accepted");
    p.plugins.clear();
    for (unsigned i = 0; i < 4097; ++i) {
        auto name = "L" + std::to_string(i) + ".esp"; write(root / name, plugin({}, 0x200)); p.plugins.push_back(name);
    }
    check(has(checkSkyrimMods(p, false), "slot-limit"), "light slot overflow");
    p.plugins.pop_back();
    // Open directly to exercise the boundary without a quadratic sort fixture.
    FalloutLoadOrder order; std::string error;
    check(order.open(p, error), "4096 light slots accepted");
}
void testCommand(const fs::path& root) {
    write(root / "Base.esm", plugin({}, 0, record("STAT", 1)));
    write(root / "Patch.esp", plugin({"Base.esm"}, 0, record("REFR", 0x01000001, {}, 0x20)));
    auto p = profile(root, {"Patch.esp"});
    ContentLayer layer; layer.id = "assets"; layer.name = "Assets"; layer.root = root / "assets"; fs::create_directories(layer.root); p.layers.push_back(layer);
    const auto input = root / "input.json", output = root / "output.json";
    std::string error; check(writeOdaiContentProfile(input, p, error), "write input fixture");
    const auto original = read(input), originalPlugin = read(root / "Patch.esp");
    auto command = [&](std::vector<std::string> args, std::string& result) {
        std::vector<const char*> argv; for (const auto& arg : args) argv.push_back(arg.c_str());
        std::ostringstream out, err;
        int code = runModCheckCommand(static_cast<int>(argv.size()), argv.data(), out, err); result = out.str(); return code;
    };
    std::string result;
    check(command({"--profile", input.string(), "--json", "--export-profile", output.string()}, result) == 0, "warning-only export succeeds");
    auto json = nlohmann::json::parse(result);
    check(json["ok"] == true && json["resolved_order"].size() == 2, "JSON report");
    ResolvedContentProfile reloaded;
    check(resolveContentProfile(output, {}, reloaded, error), "reload export");
    check(reloaded.plugins == std::vector<std::string>({"Base.esm", "Patch.esp"}) && reloaded.layers.size() == 1 && reloaded.layers[0].root == layer.root && !reloaded.fingerprint.empty(), "export preserves assets and fingerprints");
    const auto originalExport = read(output);
    check(command({"--profile", input.string(), "--export-profile", output.string()}, result) == 2 && read(output) == originalExport, "existing export refused");
    check(command({"--profile", input.string(), "--export-profile", input.string()}, result) == 2, "source profile export refused");
    check(command({"--profile", input.string()}, result) == 0 && result.find("Resolved order:") != std::string::npos, "human output");
    check(command({}, result) == 2 && command({"--profile"}, result) == 2 && command({"--unknown"}, result) == 2, "usage statuses");
    check(read(input) == original && read(root / "Patch.esp") == originalPlugin, "sources unchanged");
    write(root / "Base.esm", "bad");
    check(command({"--profile", input.string(), "--export-profile", (root / "bad.json").string()}, result) == 1 && !fs::exists(root / "bad.json"), "invalid data prevents export");
}
}
int main() {
    const auto root = fs::temp_directory_path() / ("odai-mod-check-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    testOrders(root / "orders"); testIntegrity(root / "integrity"); testLimits(root / "limits"); testCommand(root / "command");
    fs::remove_all(root);
    return failures == 0 ? 0 : 1;
}
