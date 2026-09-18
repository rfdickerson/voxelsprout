#include "bethesda/skyrim_items.h"
#include "bethesda/record_resolver.h"
#include "import/fnv/strings_table.h"
#include <cmath>
#include <cstring>

namespace odai::bethesda {
namespace {
template <class T> T read(const std::uint8_t *p) {
    T v{};
    std::memcpy(&v, p, sizeof(v));
    return v;
}
std::string text(const importer::fnv::EsmSubrecordView &sub) {
    std::string value(reinterpret_cast<const char *>(sub.data), sub.size);
    const auto nul = value.find('\0');
    if (nul != std::string::npos)
        value.resize(nul);
    return value;
}
struct Effect {
    std::uint32_t form = 0;
    float magnitude = 0;
    bool immediate = false;
};
struct Item {
    SkyrimItemDefinition definition;
    std::vector<Effect> effects;
    bool potion = false;
    bool unsupported = false;
};
} // namespace
bool loadSkyrimItems(const importer::fnv::FalloutLoadOrder &order,
                     const importer::fnv::FalloutAssetSource &assets,
                     std::map<RecordKey, SkyrimItemDefinition> &out, std::string &error) {
    std::map<std::uint32_t, Item> items;
    std::map<std::uint32_t, bool> healingEffects;
    for (std::size_t plugin = 0; plugin < order.size(); ++plugin) {
        const auto &source = order.entries()[plugin];
        importer::fnv::EsmReader reader;
        if (!reader.open(source.path)) {
            error = reader.lastError();
            return false;
        }
        importer::fnv::FalloutStringTable names, descriptions;
        std::string localizationError;
        if (source.header.isLocalized) {
            (void)importer::fnv::loadFalloutStringTable(
                assets, source.header.fileName, importer::fnv::falloutStringLanguage(),
                importer::fnv::FalloutStringFileKind::Strings, names, localizationError);
            (void)importer::fnv::loadFalloutStringTable(
                assets, source.header.fileName, importer::fnv::falloutStringLanguage(),
                importer::fnv::FalloutStringFileKind::DlStrings, descriptions, localizationError);
        }
        const auto localized = [&](const auto &sub, const auto &table) {
            if (!source.header.isLocalized)
                return text(sub);
            if (sub.size == 4u)
                if (const auto *value = table.find(read<std::uint32_t>(sub.data)))
                    return *value;
            return std::string{};
        };
        importer::fnv::EsmReader::Visitor visitor;
        visitor.onRecordHeader = [](const auto &r) {
            return r.type == "NPC_" || r.type == "CELL" || r.type == "LCTN" || r.type == "MGEF" || r.type == "MISC" || r.type == "BOOK" || r.type == "WEAP" ||
                   r.type == "ARMO" || r.type == "ALCH" || r.type == "INGR" || r.type == "KEYM" ||
                   r.type == "AMMO" || r.type == "SLGM" || r.type == "SCRL";
        };
        bool failed = false;
        visitor.onRecord = [&](const importer::fnv::EsmRecordView &r) {
            const auto id = order.remapFormId(plugin, r.formId);
            if (r.type == "MGEF") {
                bool healing = false, conditioned = false;
                if ((r.flags & 0x20u) == 0u)
                    for (const auto &sub : r.subrecords) {
                        if (sub.type == "CTDA" || sub.type == "VMAD")
                            conditioned = true;
                        // TES5 MGEF DATA: flags, archetype (64), primary AV (68).
                        if (sub.type == "DATA" && sub.size >= 72u)
                            healing = (read<std::uint32_t>(sub.data) & 5u) == 0u &&
                                      read<std::uint32_t>(sub.data + 64u) == 0u &&
                                      read<std::uint32_t>(sub.data + 68u) == 24u;
                    }
                healingEffects[id] = healing && !conditioned;
                return;
            }
            items.erase(id);
            if ((r.flags & 0x20u) != 0u)
                return;
            Item item;
            item.potion = r.type == "ALCH";
            item.definition.recordType = r.type;
            if (!stableRecordKey(order, id, item.definition.record, error)) {
                failed = true;
                return;
            }
            float damage = 0;
            bool melee = false, enchanted = false;
            for (const auto &sub : r.subrecords) {
                if (sub.type == "FULL")
                    item.definition.name = localized(sub, names);
                if (sub.type == "DESC" && r.type == "BOOK")
                    item.definition.text = localized(sub, descriptions);
                if ((sub.type == "MODL" && r.type != "ARMO") ||
                    (sub.type == "MOD2" && r.type == "ARMO"))
                    item.definition.model = text(sub);
                if ((sub.type == "BOD2" || sub.type == "BODT") && r.type == "ARMO" && sub.size >= 4u)
                    item.definition.bipedSlots = read<std::uint32_t>(sub.data);
                if (sub.type == "ENAM")
                    enchanted = true;
                if (sub.type == "DATA" && r.type == "WEAP" && sub.size >= 10u)
                    damage = read<std::uint16_t>(sub.data + 8u);
                if (sub.type == "DNAM" && r.type == "WEAP" && sub.size >= 1u) {
                    melee = sub.data[0] >= 1u && sub.data[0] <= 6u;
                    item.definition.weaponAnimationType = sub.data[0];
                }
                if (sub.type == "CTDA" || sub.type == "VMAD")
                    item.unsupported = true;
                if (sub.type == "EFID" && sub.size == 4u)
                    item.effects.push_back(
                        {order.remapFormId(plugin, read<std::uint32_t>(sub.data))});
                if (sub.type == "EFIT" && !item.effects.empty() && sub.size == 12u) {
                    auto &effect = item.effects.back();
                    effect.magnitude = read<float>(sub.data);
                    effect.immediate = read<std::uint32_t>(sub.data + 4u) == 0u &&
                                       read<std::uint32_t>(sub.data + 8u) == 0u;
                }
            }
            item.definition.meleeDamage = melee && !enchanted ? damage : 0.f;
            items[id] = std::move(item);
        };
        if (!reader.walk(visitor) || failed) {
            if (error.empty())
                error = reader.lastError();
            return false;
        }
    }
    std::map<RecordKey, SkyrimItemDefinition> result;
    for (auto &[id, item] : items) {
        (void)id;
        if (item.potion && !item.unsupported && item.effects.size() == 1u) {
            const auto &effect = item.effects.front();
            if (effect.immediate && std::isfinite(effect.magnitude) && effect.magnitude > 0 &&
                healingEffects[effect.form])
                item.definition.healing = effect.magnitude;
        }
        result.emplace(item.definition.record, std::move(item.definition));
    }
    out = std::move(result);
    error.clear();
    return true;
}
} // namespace odai::bethesda
