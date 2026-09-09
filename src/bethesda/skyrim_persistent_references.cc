#include "bethesda/skyrim_persistent_references.h"
#include "bethesda/record_resolver.h"
#include "import/fnv/actor_records.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>

namespace odai::bethesda {
namespace {
template <class T> T read(const std::uint8_t *data) {
    T value{};
    std::memcpy(&value, data, sizeof(T));
    return value;
}
struct Cell {
    RuntimeSpaceState space;
    RecordKey location;
};
struct Placement {
    RuntimeObject object;
    std::uint32_t base = 0u;
    std::uint32_t cell = 0u;
};
} // namespace

bool materializeSkyrimPersistentReferences(const importer::fnv::FalloutLoadOrder &order,
                                           const std::set<std::uint32_t> &references,
                                           BethesdaWorld &world, std::string &error) {
    std::map<std::uint32_t, Placement> placements;
    std::map<std::uint32_t, Cell> cells;
    for (std::size_t plugin = 0; plugin < order.size(); ++plugin) {
        importer::fnv::EsmReader reader;
        if (!reader.open(order.entries()[plugin].path)) {
            error = reader.lastError();
            return false;
        }
        std::uint32_t currentCell = 0u, currentWorld = 0u;
        std::vector<std::pair<std::uint32_t, std::uint32_t>> parents;
        importer::fnv::EsmReader::Visitor visitor;
        visitor.onGroupEnter = [&](const importer::fnv::EsmGroupView &group) {
            parents.emplace_back(currentCell, currentWorld);
            if (group.groupType == 1)
                currentWorld = order.remapFormId(
                    plugin, read<std::uint32_t>(
                                reinterpret_cast<const std::uint8_t *>(group.rawLabel.data())));
            if (group.groupType == 6)
                currentCell = order.remapFormId(
                    plugin, read<std::uint32_t>(
                                reinterpret_cast<const std::uint8_t *>(group.rawLabel.data())));
            return true;
        };
        visitor.onGroupExit = [&](const importer::fnv::EsmGroupView &) {
            currentCell = parents.back().first;
            currentWorld = parents.back().second;
            parents.pop_back();
        };
        visitor.onRecordHeader = [&](const importer::fnv::EsmRecordHeaderView &record) {
            return record.type == "CELL" ||
                   ((record.type == "ACHR" || record.type == "REFR") &&
                    references.contains(order.remapFormId(plugin, record.formId)));
        };
        bool failed = false;
        visitor.onRecord = [&](const importer::fnv::EsmRecordView &record) {
            if (failed)
                return;
            const auto form = order.remapFormId(plugin, record.formId);
            const auto key = [&](std::uint32_t id, RecordKey &destination) {
                if (id == 0u) {
                    destination = {};
                    return true;
                }
                if (!stableRecordKey(order, id, destination, error)) {
                    failed = true;
                    return false;
                }
                return true;
            };
            if (record.type == "CELL") {
                if ((record.flags & 0x20u) != 0u) {
                    cells.erase(form);
                    return;
                }
                Cell cell;
                key(form, cell.space.cell);
                key(currentWorld, cell.space.worldspace);
                cell.space.kind = RuntimeSpaceKind::Interior;
                for (const auto &sub : record.subrecords) {
                    if (sub.type == "DATA" && sub.size >= 1u)
                        cell.space.kind = (sub.data[0] & 1u) ? RuntimeSpaceKind::Interior
                                                             : RuntimeSpaceKind::Exterior;
                    if (sub.type == "XCLC" && sub.size >= 8u) {
                        cell.space.gridX = read<std::int32_t>(sub.data);
                        cell.space.gridZ = read<std::int32_t>(sub.data + 4u);
                    }
                    if (sub.type == "XLCN" && sub.size == 4u)
                        key(order.remapFormId(plugin, read<std::uint32_t>(sub.data)),
                            cell.location);
                }
                if (cell.space.kind == RuntimeSpaceKind::Interior)
                    cell.space.worldspace = {};
                // A CELL override may occur outside its original WRLD group.
                if (!cell.space.worldspace.valid() &&
                    cell.space.kind == RuntimeSpaceKind::Exterior && cells.contains(form))
                    cell.space.worldspace = cells.at(form).space.worldspace;
                cells[form] = std::move(cell);
                return;
            }
            if ((record.flags & 0x20u) != 0u) {
                placements.erase(form);
                return;
            }
            Placement placement;
            placement.cell = currentCell;
            if (placement.cell == 0u && placements.contains(form))
                placement.cell = placements.at(form).cell;
            RecordKey reference;
            key(form, reference);
            auto &object = placement.object;
            object.id = ObjectId::persistent(std::move(reference));
            object.kind =
                record.type == "ACHR" ? RuntimeObjectKind::Actor : RuntimeObjectKind::Activator;
            object.persistent = true;
            object.enabled = (record.flags & 0x800u) == 0u;
            bool hasPosition = false;
            for (const auto &sub : record.subrecords) {
                if (sub.type == "NAME" && sub.size == 4u) {
                    placement.base = order.remapFormId(plugin, read<std::uint32_t>(sub.data));
                    key(placement.base, object.base);
                }
                if (sub.type == "DATA" && sub.size == 24u) {
                    const float x = read<float>(sub.data), y = read<float>(sub.data + 4u),
                                z = read<float>(sub.data + 8u);
                    object.transform.position = {x, z, -y};
                    object.transform.rotationRadians = {read<float>(sub.data + 12u),
                                                        -read<float>(sub.data + 20u),
                                                        -read<float>(sub.data + 16u)};
                    hasPosition = std::isfinite(x) && std::isfinite(y) && std::isfinite(z);
                }
                if (sub.type == "XLRT" && sub.size % 4u == 0u) {
                    for (std::size_t offset = 0; offset < sub.size; offset += 4u) {
                        RecordKey type;
                        key(order.remapFormId(plugin, read<std::uint32_t>(sub.data + offset)),
                            type);
                        if (type.valid())
                            object.referenceTypes.push_back(std::move(type));
                    }
                }
            }
            if (!object.base.valid() || !hasPosition || placement.cell == 0u) {
                failed = true;
                error = "quest reference has incomplete placed data: " + object.id.toString();
                return;
            }
            placements[form] = std::move(placement);
        };
        if (!reader.walk(visitor) || failed) {
            if (error.empty())
                error = reader.lastError();
            return false;
        }
    }
    importer::fnv::FalloutActorScan actors;
    std::unordered_map<std::uint32_t, std::string> voiceOwners;
    if (std::any_of(placements.begin(), placements.end(),
                    [](const auto &entry) {
                        return entry.second.object.kind == RuntimeObjectKind::Actor;
                    }) &&
        !importer::fnv::findAllActorsAcrossOrder(order, actors, voiceOwners, error))
        return false;
    // Validate everything before committing any objects.
    for (auto &[form, placement] : placements) {
        (void)form;
        auto &object = placement.object;
        const auto cell = cells.find(placement.cell);
        if (cell == cells.end()) {
            error = "quest reference has no winning CELL: " + object.id.toString();
            return false;
        }
        object.originSpace = object.currentSpace = cell->second.space;
        object.interior = object.currentSpace.kind == RuntimeSpaceKind::Interior;
        object.location = cell->second.location;
        if (object.kind == RuntimeObjectKind::Actor) {
            object.actorValues.emplace();
            for (const auto &[formId, count] :
                 actors.materializeInventoryStacks(placement.base, form)) {
                RecordKey item;
                if (!stableRecordKey(order, formId, item, error))
                    return false;
                const auto found =
                    std::find_if(object.inventory.begin(), object.inventory.end(),
                                 [&](const InventoryEntry &entry) { return entry.item == item; });
                if (found == object.inventory.end())
                    object.inventory.push_back({std::move(item), count, false});
                else
                    found->count += count;
            }
            std::sort(object.inventory.begin(), object.inventory.end(),
                      [](const auto &a, const auto &b) { return a.item < b.item; });
        }
    }
    for (auto &[form, placement] : placements) {
        (void)form;
        if (world.find(placement.object.id) == nullptr &&
            !world.addInitialObject(std::move(placement.object), error))
            return false;
    }
    error.clear();
    return true;
}
} // namespace odai::bethesda
