#include "bethesda/skyrim_items.h"
#include "bethesda/skyrim_persistent_references.h"
#include <bit>
#include <cassert>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>

using namespace odai::bethesda;
namespace {
using Bytes = std::vector<std::uint8_t>;
void u16(Bytes &b, std::uint16_t n) {
    b.push_back(n);
    b.push_back(n >> 8u);
}
void u32(Bytes &b, std::uint32_t n) {
    for (int i = 0; i < 4; ++i)
        b.push_back(n >> (8 * i));
}
void append(Bytes &b, const Bytes &x) { b.insert(b.end(), x.begin(), x.end()); }
void tag(Bytes &b, const char *t) { b.insert(b.end(), t, t + 4); }
Bytes word(std::uint32_t n) {
    Bytes b;
    u32(b, n);
    return b;
}
void sub(Bytes &b, const char *t, const Bytes &x) {
    tag(b, t);
    u16(b, x.size());
    append(b, x);
}
Bytes record(const char *t, std::uint32_t id, Bytes body, std::uint32_t flags = 0) {
    Bytes b;
    tag(b, t);
    u32(b, body.size());
    u32(b, flags);
    u32(b, id);
    u32(b, 0);
    u32(b, 44);
    append(b, body);
    return b;
}
Bytes group(std::uint32_t label, std::int32_t type, const Bytes &body) {
    Bytes b;
    tag(b, "GRUP");
    u32(b, body.size() + 24);
    u32(b, label);
    u32(b, type);
    u32(b, 0);
    u32(b, 0);
    append(b, body);
    return b;
}
Bytes header(bool override = false) {
    Bytes body, hedr;
    u32(hedr, std::bit_cast<std::uint32_t>(1.7f));
    u32(hedr, 5);
    u32(hedr, 0x800);
    sub(body, "HEDR", hedr);
    if (override) {
        sub(body, "MAST", Bytes{'F', 'i', 'x', 't', 'u', 'r', 'e', '.', 'e', 's', 'm', 0});
        sub(body, "DATA", Bytes(8));
    }
    return record("TES4", 0, body, 1);
}
Bytes actor(std::uint32_t flags = 0, float x = 10) {
    Bytes b;
    sub(b, "NAME", word(0x100));
    Bytes pose;
    for (float f : {x, 20.f, 30.f, 0.f, 0.f, 0.f})
        u32(pose, std::bit_cast<std::uint32_t>(f));
    sub(b, "DATA", pose);
    sub(b, "XLRT", word(0x400));
    return record("ACHR", 0x200, b, flags);
}
void write(const std::filesystem::path &p, const Bytes &b) {
    std::ofstream f(p, std::ios::binary);
    f.write(reinterpret_cast<const char *>(b.data()), b.size());
    assert(f.good());
}
} // namespace
int main() {
    const auto dir = std::filesystem::temp_directory_path() /
                     ("odai-reference-test-" +
                      std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(dir);
    Bytes plugin = header();
    Bytes npc;
    sub(npc, "EDID", Bytes{'A', 'c', 't', 'o', 'r', 0});
    Bytes inventory = word(0x500);
    append(inventory, word(2));
    sub(npc, "CNTO", inventory);
    append(plugin, record("NPC_", 0x100, npc));
    Bytes book;
    sub(book, "FULL", Bytes{'J', 'o', 'u', 'r', 'n', 'a', 'l', 0});
    sub(book, "DESC", Bytes{'R', 'e', 'a', 'd', ' ', 'm', 'e', 0});
    append(plugin, record("BOOK", 0x500, book));
    Bytes weapon, damage = word(10);
    append(damage, word(std::bit_cast<std::uint32_t>(2.f)));
    u16(damage, 12);
    sub(weapon, "DATA", damage);
    sub(weapon, "DNAM", Bytes{1});
    append(plugin, record("WEAP", 0x501, weapon));
    Bytes effectData(72);
    auto av = word(24);
    std::copy(av.begin(), av.end(), effectData.begin() + 68);
    Bytes effect;
    sub(effect, "DATA", effectData);
    append(plugin, record("MGEF", 0x502, effect));
    Bytes potion;
    sub(potion, "EFID", word(0x502));
    Bytes magnitude = word(std::bit_cast<std::uint32_t>(25.f));
    append(magnitude, word(0));
    append(magnitude, word(0));
    sub(potion, "EFIT", magnitude);
    append(plugin, record("ALCH", 0x503, potion));
    Bytes cell;
    sub(cell, "DATA", Bytes{1, 0});
    sub(cell, "XLCN", word(0x300));
    append(plugin, record("CELL", 0x90, cell));
    append(plugin, group(0x90, 6, actor()));
    write(dir / "Fixture.esm", plugin);
    odai::importer::bethesda::FalloutLoadOrder order;
    std::string error;
    assert(order.open(dir, {"Fixture.esm"}, error));
    odai::importer::bethesda::FalloutAssetSource assets;
    assert(assets.open(dir));
    std::map<RecordKey, SkyrimItemDefinition> itemCatalog;
    assert(loadSkyrimItems(order, assets, itemCatalog, error));
    assert(itemCatalog.at(makeRecordKey("Fixture.esm", 0x500)).name == "Journal");
    assert(itemCatalog.at(makeRecordKey("Fixture.esm", 0x500)).text == "Read me");
    assert(itemCatalog.at(makeRecordKey("Fixture.esm", 0x501)).meleeDamage == 12);
    assert(itemCatalog.at(makeRecordKey("Fixture.esm", 0x501)).recordType == "WEAP");
    assert(itemCatalog.at(makeRecordKey("Fixture.esm", 0x500)).recordType == "BOOK");
    assert(itemCatalog.at(makeRecordKey("Fixture.esm", 0x503)).healing == 25);
    BethesdaWorld world;
    assert(materializeSkyrimPersistentReferences(order, {0x200, 0x500}, world, error));
    const auto id = ObjectId::persistent(makeRecordKey("Fixture.esm", 0x200));
    const auto *object = world.find(id);
    assert(object);
    assert(object->kind == RuntimeObjectKind::Actor);
    assert(object->transform.position[0] == 10 && object->transform.position[1] == 30 &&
           object->transform.position[2] == -20);
    assert(object->currentSpace.kind == RuntimeSpaceKind::Interior);
    assert(object->currentSpace.cell == makeRecordKey("Fixture.esm", 0x90));
    assert(object->location == makeRecordKey("Fixture.esm", 0x300));
    assert(object->referenceTypes == std::vector<RecordKey>{makeRecordKey("Fixture.esm", 0x400)});
    assert(object->inventory.size() == 1 && object->inventory[0].count == 2);
    // Startup effects apply without a character controller or rendered actor.
    WorldCommand command;
    command.type = WorldCommandType::SetGhost;
    command.target = id;
    command.enabled = true;
    (void)world.queue(command);
    assert(world.applyQueuedCommands().diagnostics.empty());
    assert(world.find(id)->ghost);
    assert(materializeSkyrimPersistentReferences(order, {0x200}, world, error));
    assert(world.find(id)->ghost && world.find(id)->inventory[0].count == 2);
    // A winning override owns placement and enabled state, but cannot reset a live save.
    Bytes patch = header(true);
    append(patch, group(0x90, 6, actor(0x800, 50)));
    write(dir / "Patch.esp", patch);
    assert(order.open(dir, {"Fixture.esm", "Patch.esp"}, error));
    BethesdaWorld overridden;
    assert(materializeSkyrimPersistentReferences(order, {0x200}, overridden, error));
    assert(!overridden.find(id)->enabled && overridden.find(id)->transform.position[0] == 50);
    Bytes deleted = header(true);
    append(deleted, group(0x90, 6, actor(0x20)));
    write(dir / "Patch.esp", deleted);
    assert(order.open(dir, {"Fixture.esm", "Patch.esp"}, error));
    BethesdaWorld absent;
    assert(materializeSkyrimPersistentReferences(order, {0x200}, absent, error));
    assert(!absent.find(id));
    std::filesystem::remove_all(dir);
    std::cout << "Skyrim persistent reference tests passed\n";
}
