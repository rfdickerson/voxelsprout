#include "bethesda/skyrim_scenario_content.h"

#include "bethesda/record_resolver.h"
#include "bethesda/skyrim_runtime_records.h"
#include "import/fnv/esm_reader.h"
#include "import/fnv/strings_table.h"

#include <algorithm>
#include <cstring>
#include <functional>
#include <optional>
#include <set>
#include <unordered_map>

namespace odai::bethesda {
namespace {

std::string lowerAscii(std::string value) {
    for (char& ch : value) {
        if (ch >= 'A' && ch <= 'Z') ch = static_cast<char>(ch - 'A' + 'a');
    }
    return value;
}

}  // namespace

bool loadSkyrimScenarioContent(
    const ScenarioDefinition& scenario,
    const importer::fnv::FalloutLoadOrder& loadOrder,
    const importer::fnv::FalloutAssetSource& assets,
    BethesdaSession& session,
    SkyrimScenarioContentReport& outReport,
    std::string& outError) {
    outReport = {};
    if (scenario.game != importer::fnv::BethesdaGame::SkyrimSpecialEdition || loadOrder.empty()) {
        outError = "Skyrim scenario content requires a Skyrim scenario and active load order";
        return false;
    }

    std::map<std::uint32_t, const ScenarioQuestRecord*> wantedQuestForms;
    for (const ScenarioQuestRecord& quest : scenario.questRecords) {
        const auto source = std::find_if(
            loadOrder.entries().begin(), loadOrder.entries().end(), [&](const auto& entry) {
                return lowerAscii(entry.header.fileName) == lowerAscii(quest.plugin);
            });
        if (source == loadOrder.entries().end()) {
            outError = "required quest master is absent: " + quest.plugin;
            return false;
        }
        const std::size_t sourceIndex = static_cast<std::size_t>(
            std::distance(loadOrder.entries().begin(), source));
        wantedQuestForms.emplace(
            loadOrder.remapFormId(sourceIndex, quest.localFormId), &quest);
    }
    std::map<std::uint32_t, SkyrimQuestDefinition> winningDefinitions;
    std::map<std::uint32_t, std::size_t> winningDefinitionSources;
    std::set<std::uint32_t> deletedRequiredQuests;
    for (std::size_t pluginIndex = 0u; pluginIndex < loadOrder.entries().size(); ++pluginIndex) {
        const auto& entry = loadOrder.entries()[pluginIndex];
        importer::fnv::EsmReader reader;
        if (!reader.open(entry.path)) {
            outError = "could not open quest source " + entry.path.string() + ": " + reader.lastError();
            return false;
        }
        importer::fnv::EsmReader::Visitor visitor;
        visitor.onRecordHeader = [&](const importer::fnv::EsmRecordHeaderView& header) {
            return header.type == "QUST" && wantedQuestForms.contains(
                loadOrder.remapFormId(pluginIndex, header.formId));
        };
        bool parseFailed = false;
        std::string parseError;
        visitor.onRecord = [&](const importer::fnv::EsmRecordView& record) {
            const std::uint32_t resolved = loadOrder.remapFormId(pluginIndex, record.formId);
            const auto match = wantedQuestForms.find(resolved);
            if (match == wantedQuestForms.end()) return;
            if ((record.flags & 0x20u) != 0u) {
                winningDefinitions.erase(resolved);
                winningDefinitionSources.erase(resolved);
                deletedRequiredQuests.insert(resolved);
                return;
            }
            SkyrimQuestDefinition definition;
            if (!readSkyrimQuest(record,
                    makeRecordKey(match->second->plugin, match->second->localFormId),
                    definition, parseError)) {
                parseFailed = true;
                return;
            }
            if (lowerAscii(definition.editorId) != lowerAscii(match->second->editorId)) {
                parseError = "winning QUST override changed required EditorID " +
                    match->second->editorId + " to " + definition.editorId;
                parseFailed = true;
                return;
            }
            winningDefinitions.insert_or_assign(resolved, std::move(definition));
            winningDefinitionSources.insert_or_assign(resolved, pluginIndex);
            deletedRequiredQuests.erase(resolved);
        };
        if (!reader.walk(visitor) || parseFailed) {
            outError = "could not parse quest closure from " + entry.path.string() + ": " +
                (parseFailed ? parseError : reader.lastError());
            return false;
        }
    }
    if (!deletedRequiredQuests.empty()) {
        outError = "winning load-order record deletes required scenario QUST " +
            wantedQuestForms.at(*deletedRequiredQuests.begin())->editorId;
        return false;
    }
    if (winningDefinitions.size() != scenario.questRecords.size()) {
        outError = "resolved " + std::to_string(winningDefinitions.size()) + " of " +
            std::to_string(scenario.questRecords.size()) + " required quest records";
        return false;
    }
    std::vector<SkyrimQuestDefinition> definitions;
    std::map<RecordKey, std::size_t> definitionSourcePluginIndices;
    definitions.reserve(winningDefinitions.size());
    for (auto& [resolved, definition] : winningDefinitions) {
        definitionSourcePluginIndices.emplace(
            definition.record, winningDefinitionSources.at(resolved));
        definitions.push_back(std::move(definition));
    }

    std::unordered_map<std::string, importer::fnv::FalloutStringTable> stringTables;
    std::set<std::string> unavailableStringTables;
    const auto resolveObjectiveText = [&](SkyrimQuestDefinition& definition,
                                          std::size_t sourcePluginIndex) {
        const std::string sourcePlugin =
            loadOrder.entries()[sourcePluginIndex].header.fileName;
        const std::string plugin = lowerAscii(sourcePlugin);
        if (unavailableStringTables.contains(plugin)) return;
        auto table = stringTables.find(plugin);
        if (table == stringTables.end()) {
            importer::fnv::FalloutStringTable loaded;
            std::string error;
            if (!importer::fnv::loadFalloutStringTable(
                    assets, sourcePlugin,
                    importer::fnv::falloutStringLanguage(),
                    importer::fnv::FalloutStringFileKind::Strings,
                    loaded, error)) {
                unavailableStringTables.insert(plugin);
                outReport.diagnostics.push_back(
                    sourcePlugin + " has no readable quest objective string table: " +
                    error);
                return;
            }
            table = stringTables.emplace(plugin, std::move(loaded)).first;
        }
        for (SkyrimQuestObjectiveDefinition& objective : definition.objectives) {
            if (const std::string* text = table->second.find(objective.displayTextId)) {
                objective.displayText = *text;
            } else if (objective.displayTextId != 0u) {
                outReport.diagnostics.push_back(
                    definition.editorId + " objective " + std::to_string(objective.index) +
                    " has unresolved localized string " +
                    std::to_string(objective.displayTextId));
            }
        }
    };
    for (SkyrimQuestDefinition& definition : definitions) {
        resolveObjectiveText(definition, definitionSourcePluginIndices.at(definition.record));
    }

    const auto* loadOrderPointer = &loadOrder;
    const BethesdaSession::ResolvedFormResolver referenceResolver =
        [loadOrderPointer](std::uint32_t formId) -> std::optional<ObjectId> {
        RecordKey stable;
        std::string error;
        if (!stableRecordKey(*loadOrderPointer, formId, stable, error)) return std::nullopt;
        return ObjectId::persistent(std::move(stable));
    };
    session.setResolvedFormResolver(referenceResolver);
    const auto remapQuestAliasForms = [&](SkyrimQuestDefinition& definition,
                                          std::size_t sourcePluginIndex) {
        for (SkyrimQuestAliasDefinition& alias : definition.aliases) {
            const auto remap = [&](std::uint32_t& formId) {
                if (formId != 0u) {
                    formId = loadOrder.remapFormId(sourcePluginIndex, formId);
                }
            };
            remap(alias.forcedReferenceFormId);
            remap(alias.uniqueActorFormId);
            remap(alias.createdObjectFormId);
            remap(alias.forcedLocationFormId);
            remap(alias.referenceTypeFormId);
        }
        for (SkyrimQuestStageDefinition& stage : definition.stages) {
            for (SkyrimQuestLogEntryDefinition& entry : stage.logEntries) {
                for (Condition& condition : entry.conditions) {
                    if (condition.function == 47u || condition.function == 67u) {
                        condition.parameter1 = loadOrder.remapFormId(
                            sourcePluginIndex, condition.parameter1);
                    }
                    if (condition.runOn == 2u && condition.reference != 0u) {
                        condition.reference = loadOrder.remapFormId(
                            sourcePluginIndex, condition.reference);
                    }
                }
            }
        }
    };
    std::unordered_map<std::string, PexScript> loadedPexScripts;
    std::function<bool(std::uint32_t, const std::string&, std::string&)>
        ensureRuntimeRecord;
    std::function<bool(std::uint32_t, std::string&)> ensureQuestDefinition;
    ensureQuestDefinition = [&](std::uint32_t resolvedQuestFormId, std::string& error) {
        const std::optional<ObjectId> questObject = referenceResolver(resolvedQuestFormId);
        if (!questObject.has_value()) {
            error = "cross-quest alias has an unresolvable quest form";
            return false;
        }
        if (session.findQuest(*questObject) != nullptr) return true;
        RecordKey record = questObject->reference;
        std::optional<SkyrimQuestDefinition> winning;
        std::size_t winningPluginIndex = 0u;
        bool winningDeleted = false;
        for (std::size_t pluginIndex = 0u;
             pluginIndex < loadOrder.entries().size(); ++pluginIndex) {
            importer::fnv::EsmReader reader;
            if (!reader.open(loadOrder.entries()[pluginIndex].path)) continue;
            importer::fnv::EsmReader::Visitor visitor;
            visitor.onRecordHeader = [&](const importer::fnv::EsmRecordHeaderView& header) {
                return header.type == "QUST" &&
                    loadOrder.remapFormId(pluginIndex, header.formId) == resolvedQuestFormId;
            };
            std::string parseError;
            visitor.onRecord = [&](const importer::fnv::EsmRecordView& questRecord) {
                if ((questRecord.flags & 0x20u) != 0u) {
                    winning.reset();
                    winningDeleted = true;
                    return;
                }
                SkyrimQuestDefinition definition;
                if (!readSkyrimQuest(questRecord, record, definition, parseError)) return;
                winning = std::move(definition);
                winningPluginIndex = pluginIndex;
                winningDeleted = false;
            };
            if (!reader.walk(visitor) || !parseError.empty()) {
                error = parseError.empty() ? reader.lastError() : parseError;
                return false;
            }
        }
        if (!winning.has_value()) {
            error = winningDeleted
                ? "winning load-order record deletes required cross-quest QUST " +
                    record.toString()
                : "cross-quest alias owner is not a QUST record";
            return false;
        }
        resolveObjectiveText(*winning, winningPluginIndex);
        remapQuestAliasForms(*winning, winningPluginIndex);
        if (!session.registerQuestDefinition(*winning, referenceResolver, error)) return false;

        // A Quest-typed property can transition into another quest in the
        // same authored fragment (MQ103 -> MQ104 is the release-route case).
        // Register its common QF attachment as part of the same content
        // closure so SetStage never targets a definition with no executable
        // fragment owner.
        for (const VmadScriptAttachment& attachment : winning->scripts.scripts) {
            std::vector<std::uint8_t> pexBytes;
            if (!assets.resolveAsset(
                    "scripts\\" + attachment.className + ".pex", pexBytes, error)) {
                error = "missing cross-quest script " + attachment.className + ": " + error;
                return false;
            }
            PexScript pex;
            if (!readPexScript(pexBytes, pex, error)) {
                error = "malformed cross-quest script " + attachment.className + ": " + error;
                return false;
            }
            if (!session.papyrus().hasScriptClass(attachment.className)) {
                PexCompatibilityReport compatibility;
                if (!session.papyrus().loadPexScript(pex, false, compatibility, error)) return false;
            }
            loadedPexScripts.insert_or_assign(lowerAscii(attachment.className), pex);
            const auto objectInfo = std::find_if(
                pex.objectInfo.begin(), pex.objectInfo.end(), [&](const PexObjectInfo& object) {
                    return lowerAscii(object.name) == lowerAscii(attachment.className);
                });
            if (objectInfo != pex.objectInfo.end()) {
                for (const PexPropertyInfo& propertyInfo : objectInfo->properties) {
                    const auto property = std::find_if(
                        attachment.properties.begin(), attachment.properties.end(),
                        [&](const VmadProperty& candidate) {
                            return lowerAscii(candidate.name) == lowerAscii(propertyInfo.name);
                        });
                    if (property == attachment.properties.end()) continue;
                    std::string propertyType = lowerAscii(propertyInfo.type);
                    if (propertyType.ends_with("[]")) propertyType.resize(propertyType.size() - 2u);
                    std::function<bool(const VmadValue&)> ensureValue;
                    ensureValue = [&](const VmadValue& value) {
                        if (value.type == VmadValueType::Object &&
                            value.object.formId != 0u && value.object.alias == 0xffffu &&
                            (propertyType == "location" ||
                             propertyType == "globalvariable")) {
                            return ensureRuntimeRecord(
                                loadOrder.remapFormId(
                                    winningPluginIndex, value.object.formId),
                                propertyType, error);
                        }
                        for (const VmadValue& element : value.array) {
                            if (!ensureValue(element)) return false;
                        }
                        return true;
                    };
                    if (!ensureValue(property->value)) return false;
                }
            }
            std::function<bool(const VmadValue&, PapyrusValue&)> convert;
            convert = [&](const VmadValue& value, PapyrusValue& converted) {
                if (value.type == VmadValueType::Object) {
                    if (value.object.formId == 0u) { converted = {}; return true; }
                    const std::uint32_t resolved = loadOrder.remapFormId(
                        winningPluginIndex, value.object.formId);
                    if (value.object.alias != 0xffffu) {
                        const std::optional<ObjectId> ownerObject = referenceResolver(resolved);
                        const QuestRuntimeState* owner = ownerObject.has_value()
                            ? session.findQuest(*ownerObject) : nullptr;
                        if (owner == nullptr) {
                            // The attached QF class can contain properties for
                            // later, unrelated stages. Preserve them as None
                            // until that quest enters the reachable closure;
                            // an actual access then fails visibly in the VM.
                            converted = {};
                            return true;
                        }
                        const auto alias = std::find_if(
                            owner->aliases.begin(), owner->aliases.end(),
                            [&](const QuestAliasRuntimeState& candidate) {
                                return candidate.id == value.object.alias;
                            });
                        if (alias == owner->aliases.end()) return false;
                        converted = PapyrusValue::fromObject(alias->handle);
                        return true;
                    }
                    const std::optional<ObjectId> object = referenceResolver(resolved);
                    if (!object.has_value()) return false;
                    converted = PapyrusValue::fromObject(*object);
                    return true;
                }
                if (value.type == VmadValueType::String) {
                    converted = PapyrusValue::fromString(value.string); return true;
                }
                if (value.type == VmadValueType::Integer) {
                    converted = PapyrusValue::fromInteger(value.integer); return true;
                }
                if (value.type == VmadValueType::Float) {
                    converted = PapyrusValue::fromFloat(value.real); return true;
                }
                if (value.type == VmadValueType::Boolean) {
                    converted = PapyrusValue::fromBoolean(value.boolean); return true;
                }
                std::vector<PapyrusValue> values;
                for (const VmadValue& element : value.array) {
                    PapyrusValue convertedElement;
                    if (!convert(element, convertedElement)) return false;
                    values.push_back(std::move(convertedElement));
                }
                converted = PapyrusValue::fromArray(std::move(values));
                return true;
            };
            std::unordered_map<std::string, PapyrusValue> properties;
            for (const VmadProperty& property : attachment.properties) {
                PapyrusValue value;
                if (!convert(property.value, value)) {
                    error = "could not resolve cross-quest property " +
                        attachment.className + "." + property.name;
                    return false;
                }
                properties.emplace(property.name, std::move(value));
            }
            if (!session.papyrus().attachScript(
                    *questObject, attachment.className, std::move(properties), error)) {
                return false;
            }
        }
        return true;
    };
    ensureRuntimeRecord = [&](std::uint32_t resolvedFormId,
        const std::string& requestedType, std::string& error) {
        const std::string type = lowerAscii(requestedType);
        if (type == "quest") return ensureQuestDefinition(resolvedFormId, error);
        if (type != "location" && type != "globalvariable") return true;
        RecordKey stable;
        if (!stableRecordKey(loadOrder, resolvedFormId, stable, error)) return false;
        if (type == "location" && session.locations().contains(stable)) return true;
        if (type == "globalvariable" && session.globalVariables().contains(stable)) return true;
        std::optional<SkyrimLocationDefinition> winningLocation;
        std::optional<SkyrimGlobalVariableDefinition> winningGlobal;
        std::size_t winningPluginIndex = 0u;
        for (std::size_t pluginIndex = 0u;
             pluginIndex < loadOrder.entries().size(); ++pluginIndex) {
            importer::fnv::EsmReader reader;
            if (!reader.open(loadOrder.entries()[pluginIndex].path)) continue;
            importer::fnv::EsmReader::Visitor visitor;
            visitor.onRecordHeader = [&](const importer::fnv::EsmRecordHeaderView& header) {
                const bool expected = type == "location"
                    ? header.type == "LCTN" : header.type == "GLOB";
                return expected &&
                    loadOrder.remapFormId(pluginIndex, header.formId) == resolvedFormId;
            };
            std::string parseError;
            visitor.onRecord = [&](const importer::fnv::EsmRecordView& record) {
                if (type == "location") {
                    SkyrimLocationDefinition definition;
                    if (!readSkyrimLocation(record, stable, definition, parseError)) return;
                    winningLocation = std::move(definition);
                } else {
                    SkyrimGlobalVariableDefinition definition;
                    if (!readSkyrimGlobalVariable(record, stable, definition, parseError)) return;
                    winningGlobal = std::move(definition);
                }
                winningPluginIndex = pluginIndex;
            };
            if (!reader.walk(visitor) || !parseError.empty()) {
                error = parseError.empty() ? reader.lastError() : parseError;
                return false;
            }
        }
        if (type == "location") {
            if (!winningLocation.has_value()) {
                error = "required Location property does not resolve to an LCTN record";
                return false;
            }
            RecordKey parent;
            if (winningLocation->parentFormId != 0u) {
                const std::uint32_t resolvedParent = loadOrder.remapFormId(
                    winningPluginIndex, winningLocation->parentFormId);
                const std::optional<ObjectId> parentObject = referenceResolver(resolvedParent);
                if (!parentObject.has_value()) {
                    error = "LCTN parent cannot be resolved through the active load order";
                    return false;
                }
                parent = parentObject->reference;
            }
            std::vector<RecordKey> keywords;
            for (const std::uint32_t keywordFormId : winningLocation->keywordFormIds) {
                const std::uint32_t resolvedKeyword = loadOrder.remapFormId(
                    winningPluginIndex, keywordFormId);
                const std::optional<ObjectId> keyword = referenceResolver(resolvedKeyword);
                if (!keyword.has_value()) {
                    error = "LCTN keyword cannot be resolved through the active load order";
                    return false;
                }
                keywords.push_back(keyword->reference);
            }
            return session.registerLocation(stable, std::move(parent),
                std::move(keywords), error);
        }
        if (!winningGlobal.has_value()) {
            error = "required GlobalVariable property does not resolve to a GLOB record";
            return false;
        }
        return session.registerGlobalVariable(
            stable, winningGlobal->initialValue, error);
    };
    const auto registerAttachmentRuntimeRecords = [&](const PexScript& pex,
        const VmadScriptAttachment& attachment, std::size_t sourcePluginIndex,
        std::string& error) {
        const auto objectInfo = std::find_if(
            pex.objectInfo.begin(), pex.objectInfo.end(), [&](const PexObjectInfo& object) {
                return lowerAscii(object.name) == lowerAscii(attachment.className);
            });
        if (objectInfo == pex.objectInfo.end()) return true;
        for (const PexPropertyInfo& propertyInfo : objectInfo->properties) {
            const auto property = std::find_if(
                attachment.properties.begin(), attachment.properties.end(),
                [&](const VmadProperty& candidate) {
                    return lowerAscii(candidate.name) == lowerAscii(propertyInfo.name);
                });
            if (property == attachment.properties.end()) continue;
            std::string propertyType = lowerAscii(propertyInfo.type);
            if (propertyType.ends_with("[]")) propertyType.resize(propertyType.size() - 2u);
            std::function<bool(const VmadValue&)> visit;
            visit = [&](const VmadValue& value) {
                if (value.type == VmadValueType::Object && value.object.formId != 0u &&
                    value.object.alias == 0xffffu) {
                    const std::uint32_t resolved = loadOrder.remapFormId(
                        sourcePluginIndex, value.object.formId);
                    return ensureRuntimeRecord(resolved, propertyType, error);
                }
                if (value.type == VmadValueType::ObjectArray) {
                    for (const VmadValue& element : value.array) {
                        if (!visit(element)) return false;
                    }
                }
                return true;
            };
            if (!visit(property->value)) return false;
        }
        return true;
    };
    std::unordered_map<std::string, std::vector<std::uint32_t>> propertyObjectOwners;

    // Register every quest and alias handle before attaching any VMAD. The
    // winning records are stored by resolved form id, which is not a dependency
    // order: MQ102B sorts before MQ102 even though its QF properties reference
    // aliases owned by MQ102. A two-phase load makes cross-quest alias values
    // independent of plugin record ordering.
    for (const SkyrimQuestDefinition& definition : definitions) {
        const auto sourcePlugin = definitionSourcePluginIndices.find(definition.record);
        if (sourcePlugin == definitionSourcePluginIndices.end()) {
            outError = "winning quest source plugin was not retained";
            return false;
        }
        SkyrimQuestDefinition runtimeDefinition = definition;
        remapQuestAliasForms(runtimeDefinition, sourcePlugin->second);
        std::string error;
        if (!session.registerQuestDefinition(runtimeDefinition, referenceResolver, error)) {
            outError = "could not register " + definition.editorId + ": " + error;
            return false;
        }
    }

    for (const SkyrimQuestDefinition& definition : definitions) {
        std::string error;
        const auto sourcePlugin = definitionSourcePluginIndices.find(definition.record);
        if (sourcePlugin == definitionSourcePluginIndices.end()) {
            outError = "winning quest source plugin was not retained";
            return false;
        }
        const std::size_t sourcePluginIndex = sourcePlugin->second;
        const ObjectId questObject = ObjectId::persistent(definition.record);
        const QuestRuntimeState* questState = session.findQuest(questObject);
        if (questState == nullptr) {
            outError = "registered quest " + definition.editorId + " has no runtime state";
            return false;
        }
        ScenarioQuestLoadDetail detail;
        detail.editorId = definition.editorId;
        detail.stages = definition.stages.size();
        detail.objectives = definition.objectives.size();
        detail.aliases = definition.aliases.size();
        detail.stageFragments = definition.stageFragments.size();
        detail.aliasScriptAttachments = definition.aliasScripts.size();
        detail.referencedRecords = definition.referencedFormIds.size();
        const auto scenarioQuest = std::find_if(
            scenario.questRecords.begin(), scenario.questRecords.end(),
            [&](const ScenarioQuestRecord& record) {
                return lowerAscii(record.editorId) == lowerAscii(definition.editorId);
            });
        const bool scriptsRequired = scenarioQuest == scenario.questRecords.end() ||
            scenarioQuest->scriptsRequired;
        std::size_t questScriptCount = definition.scripts.scripts.size();
        for (const VmadQuestAliasAttachment& alias : definition.aliasScripts) {
            questScriptCount += alias.scripts.size();
        }
        detail.scripts = scriptsRequired ? questScriptCount : 0u;

        if (!scriptsRequired) {
            outReport.diagnostics.push_back(
                definition.editorId + " scripts intentionally skipped after scenario bootstrap");
        }
        std::function<bool(const VmadValue&, PapyrusValue&)> convert;
        convert = [&](const VmadValue& value, PapyrusValue& converted) {
            switch (value.type) {
                case VmadValueType::Object:
                    if (value.object.formId == 0u) {
                        converted = {};
                    } else if (value.object.alias != 0xffffu) {
                        // Object-format-2 VMAD values carry both the owning
                        // quest form and its alias id. Quest fragments commonly
                        // hold cross-quest aliases (MQ102A/B point at MQ102's
                        // RiverwoodFriend and FactionFriend); looking only on
                        // the script's current quest silently made those legal
                        // properties unresolvable.
                        const std::uint32_t resolvedOwner = loadOrder.remapFormId(
                            sourcePluginIndex, value.object.formId);
                        const std::optional<ObjectId> ownerObject =
                            referenceResolver(resolvedOwner);
                        const QuestRuntimeState* ownerQuest = ownerObject.has_value()
                            ? session.findQuest(*ownerObject) : nullptr;
                        if (ownerQuest == nullptr) return false;
                        const auto alias = std::find_if(
                            ownerQuest->aliases.begin(), ownerQuest->aliases.end(),
                            [&](const QuestAliasRuntimeState& runtime) {
                                return runtime.id == value.object.alias;
                            });
                        if (alias == ownerQuest->aliases.end()) return false;
                        converted = PapyrusValue::fromObject(alias->handle);
                    } else {
                        const std::uint32_t resolved = loadOrder.remapFormId(
                            sourcePluginIndex, value.object.formId);
                        const std::optional<ObjectId> object = referenceResolver(resolved);
                        if (!object.has_value()) return false;
                        converted = PapyrusValue::fromObject(*object);
                    }
                    return true;
                case VmadValueType::String:
                    converted = PapyrusValue::fromString(value.string); return true;
                case VmadValueType::Integer:
                    converted = PapyrusValue::fromInteger(value.integer); return true;
                case VmadValueType::Float:
                    converted = PapyrusValue::fromFloat(value.real); return true;
                case VmadValueType::Boolean:
                    converted = PapyrusValue::fromBoolean(value.boolean); return true;
                case VmadValueType::ObjectArray:
                case VmadValueType::StringArray:
                case VmadValueType::IntegerArray:
                case VmadValueType::FloatArray:
                case VmadValueType::BooleanArray: {
                    std::vector<PapyrusValue> elements;
                    elements.reserve(value.array.size());
                    for (const VmadValue& element : value.array) {
                        PapyrusValue convertedElement;
                        if (!convert(element, convertedElement)) return false;
                        elements.push_back(std::move(convertedElement));
                    }
                    converted = PapyrusValue::fromArray(std::move(elements));
                    return true;
                }
            }
            return false;
        };
        const auto loadAndAttach = [&](const VmadScriptAttachment& attachment,
                                       ObjectId target,
                                       const std::string& targetName) {
            detail.scriptClasses.push_back(attachment.className);
            std::vector<std::uint8_t> pexBytes;
            const std::string pexPath = "scripts\\" + attachment.className + ".pex";
            if (!assets.resolveAsset(pexPath, pexBytes, error)) {
                outError = "missing required script " + pexPath + ": " + error;
                return false;
            }
            PexScript pex;
            if (!readPexScript(pexBytes, pex, error)) {
                outError = "malformed required script " + pexPath + ": " + error;
                return false;
            }
            if (!session.papyrus().hasScriptClass(attachment.className)) {
                PexCompatibilityReport report;
                if (!session.papyrus().loadPexScript(pex, false, report, error)) {
                    outError = "could not load required script " + pexPath + ": " + error;
                    return false;
                }
            }
            loadedPexScripts.insert_or_assign(lowerAscii(attachment.className), pex);
            if (!registerAttachmentRuntimeRecords(
                    pex, attachment, sourcePluginIndex, error)) {
                outError = "could not register runtime records for " +
                    attachment.className + ": " + error;
                return false;
            }
            const auto objectInfo = std::find_if(
                pex.objectInfo.begin(), pex.objectInfo.end(), [&](const PexObjectInfo& object) {
                    return lowerAscii(object.name) == lowerAscii(attachment.className);
                });
            if (objectInfo != pex.objectInfo.end()) {
                for (const PexPropertyInfo& propertyInfo : objectInfo->properties) {
                    const auto property = std::find_if(
                        attachment.properties.begin(), attachment.properties.end(),
                        [&](const VmadProperty& candidate) {
                            return lowerAscii(candidate.name) == lowerAscii(propertyInfo.name);
                        });
                    if (property != attachment.properties.end() &&
                        property->value.type == VmadValueType::Object &&
                        property->value.object.formId != 0u &&
                        property->value.object.alias == 0xffffu) {
                        propertyObjectOwners[lowerAscii(propertyInfo.type)].push_back(
                            loadOrder.remapFormId(
                                sourcePluginIndex, property->value.object.formId));
                    }
                }
            }

            std::unordered_map<std::string, PapyrusValue> properties;
            for (const VmadProperty& property : attachment.properties) {
                PapyrusValue value;
                if (!convert(property.value, value)) {
                    outError = "could not resolve VMAD property " + attachment.className + "." +
                        property.name;
                    return false;
                }
                properties.emplace(property.name, std::move(value));
            }
            if (!session.papyrus().attachScript(
                    target, attachment.className, std::move(properties), error)) {
                outError = "could not attach " + attachment.className + " to " +
                    targetName + ": " + error;
                return false;
            }
            return true;
        };
        if (scriptsRequired) {
            for (const VmadScriptAttachment& attachment : definition.scripts.scripts) {
                if (!loadAndAttach(attachment, questObject, definition.editorId)) return false;
            }
            for (const VmadQuestAliasAttachment& aliasAttachment : definition.aliasScripts) {
                const auto alias = std::find_if(
                    questState->aliases.begin(), questState->aliases.end(),
                    [&](const QuestAliasRuntimeState& runtime) {
                        return runtime.id == aliasAttachment.object.alias;
                    });
                if (alias == questState->aliases.end()) {
                    outError = definition.editorId + " VMAD names missing alias " +
                        std::to_string(aliasAttachment.object.alias);
                    return false;
                }
                const std::uint32_t owner = loadOrder.remapFormId(
                    sourcePluginIndex, aliasAttachment.object.formId);
                const std::optional<ObjectId> ownerObject = referenceResolver(owner);
                if (!ownerObject.has_value() || ownerObject->reference != definition.record) {
                    outError = definition.editorId + " VMAD alias attachment has a foreign owner";
                    return false;
                }
                for (const VmadScriptAttachment& attachment : aliasAttachment.scripts) {
                    if (!loadAndAttach(attachment, alias->handle,
                            definition.editorId + " alias " + alias->name)) return false;
                }
            }
        }
        outReport.quests.push_back(std::move(detail));
    }

    // Resolve the retail DIAL/INFO closure owned by the scenario quests. TES5
    // stores quest ownership on DIAL.QNAM and INFO ownership in the topic's
    // type-7 child GRUP, not in an INFO QSTI field as Fallout does.
    struct WinningDialogueTopic {
        SkyrimDialogueTopicDefinition definition;
        std::size_t sourcePluginIndex = 0u;
    };
    struct WinningDialogueBranch {
        SkyrimDialogueBranchDefinition definition;
        std::size_t sourcePluginIndex = 0u;
    };
    struct WinningDialogueInfo {
        SkyrimDialogueInfoDefinition definition;
        std::size_t sourcePluginIndex = 0u;
    };
    std::set<std::uint32_t> scenarioQuestFormIds;
    for (const ScenarioQuestRecord& quest : scenario.questRecords) {
        // MQ101 is consumed only to seed post-Helgen prerequisites. Its large
        // dialogue/scene closure is intentionally outside the playable route.
        if (!quest.scriptsRequired) continue;
        const auto source = std::find_if(
            loadOrder.entries().begin(), loadOrder.entries().end(), [&](const auto& entry) {
                return lowerAscii(entry.header.fileName) == lowerAscii(quest.plugin);
            });
        if (source == loadOrder.entries().end()) continue;
        scenarioQuestFormIds.insert(loadOrder.remapFormId(
            static_cast<std::size_t>(std::distance(loadOrder.entries().begin(), source)),
            quest.localFormId));
    }
    std::map<std::uint32_t, WinningDialogueTopic> winningTopics;
    std::map<std::uint32_t, WinningDialogueBranch> winningBranches;
    std::map<std::uint32_t, WinningDialogueInfo> winningInfos;
    std::uint64_t authoredInfoOrder = 0u;
    for (std::size_t pluginIndex = 0u; pluginIndex < loadOrder.entries().size(); ++pluginIndex) {
        importer::fnv::EsmReader reader;
        if (!reader.open(loadOrder.entries()[pluginIndex].path)) {
            outError = "could not open dialogue source " +
                loadOrder.entries()[pluginIndex].path.string() + ": " + reader.lastError();
            return false;
        }
        std::uint32_t currentTopicFormId = 0u;
        importer::fnv::EsmReader::Visitor visitor;
        visitor.onGroupEnter = [&](const importer::fnv::EsmGroupView& group) {
            if (group.groupType == 7 && group.rawLabel.size() == 4u) {
                std::uint32_t raw = 0u;
                std::memcpy(&raw, group.rawLabel.data(), sizeof(raw));
                currentTopicFormId = loadOrder.remapFormId(pluginIndex, raw);
            }
            return true;
        };
        visitor.onRecordHeader = [](const importer::fnv::EsmRecordHeaderView& header) {
            return header.type == "DLBR" || header.type == "DIAL" ||
                header.type == "INFO";
        };
        bool parseFailed = false;
        std::string parseError;
        visitor.onRecord = [&](const importer::fnv::EsmRecordView& record) {
            const std::uint32_t resolvedRecord =
                loadOrder.remapFormId(pluginIndex, record.formId);
            if ((record.flags & 0x20u) != 0u) {
                if (record.type == "DLBR") winningBranches.erase(resolvedRecord);
                else if (record.type == "DIAL") {
                    winningTopics.erase(resolvedRecord);
                } else if (record.type == "INFO") {
                    winningInfos.erase(resolvedRecord);
                }
                return;
            }
            RecordKey stable;
            if (!stableRecordKey(loadOrder, resolvedRecord, stable, parseError)) {
                parseFailed = true;
                return;
            }
            if (record.type == "DLBR") {
                SkyrimDialogueBranchDefinition branch;
                if (!readSkyrimDialogueBranch(record, stable, branch, parseError)) {
                    parseFailed = true;
                    return;
                }
                const std::uint32_t questForm =
                    loadOrder.remapFormId(pluginIndex, branch.rawQuestFormId);
                if (!scenarioQuestFormIds.contains(questForm)) {
                    winningBranches.erase(resolvedRecord);
                    return;
                }
                winningBranches.insert_or_assign(
                    resolvedRecord, WinningDialogueBranch{std::move(branch), pluginIndex});
                return;
            }
            if (record.type == "DIAL") {
                SkyrimDialogueTopicDefinition topic;
                if (!readSkyrimDialogueTopic(record, stable, topic, parseError)) {
                    parseFailed = true;
                    return;
                }
                const std::uint32_t questForm =
                    loadOrder.remapFormId(pluginIndex, topic.rawQuestFormId);
                if (!scenarioQuestFormIds.contains(questForm)) {
                    winningTopics.erase(resolvedRecord);
                    return;
                }
                winningTopics.insert_or_assign(
                    resolvedRecord, WinningDialogueTopic{std::move(topic), pluginIndex});
                return;
            }
            const auto topic = winningTopics.find(currentTopicFormId);
            if (topic == winningTopics.end()) {
                winningInfos.erase(resolvedRecord);
                return;
            }
            const std::uint32_t questForm = loadOrder.remapFormId(
                topic->second.sourcePluginIndex,
                topic->second.definition.rawQuestFormId);
            const std::optional<ObjectId> questObject = referenceResolver(questForm);
            if (!questObject.has_value() ||
                questObject->kind != ObjectIdKind::PersistentReference) {
                parseError = "dialogue topic quest owner has no stable identity";
                parseFailed = true;
                return;
            }
            SkyrimDialogueInfoDefinition info;
            if (!readSkyrimDialogueInfo(record, stable,
                    topic->second.definition.record, questObject->reference,
                    info, parseError)) {
                parseFailed = true;
                return;
            }
            info.authoredOrder = authoredInfoOrder++;
            winningInfos.insert_or_assign(
                resolvedRecord, WinningDialogueInfo{std::move(info), pluginIndex});
        };
        if (!reader.walk(visitor) || parseFailed) {
            outError = "could not parse Skyrim dialogue closure from " +
                loadOrder.entries()[pluginIndex].path.string() + ": " +
                (parseFailed ? parseError : reader.lastError());
            return false;
        }
    }

    std::set<RecordKey> winningTopicKeys;
    for (const auto& [resolved, topic] : winningTopics) {
        (void)resolved;
        winningTopicKeys.insert(topic.definition.record);
    }
    for (auto info = winningInfos.begin(); info != winningInfos.end();) {
        if (!winningTopicKeys.contains(info->second.definition.topic)) {
            info = winningInfos.erase(info);
        } else {
            ++info;
        }
    }

    std::map<std::string, importer::fnv::FalloutStringTable> dialogueStrings;
    std::map<std::string, importer::fnv::FalloutStringTable> dialogueIlStrings;
    std::set<std::string> missingDialogueStringTables;
    const auto dialogueTable = [&](std::size_t pluginIndex,
                                   importer::fnv::FalloutStringFileKind kind)
        -> const importer::fnv::FalloutStringTable* {
        const std::string pluginName = loadOrder.entries()[pluginIndex].header.fileName;
        const std::string key = lowerAscii(pluginName);
        auto& tables = kind == importer::fnv::FalloutStringFileKind::Strings
            ? dialogueStrings : dialogueIlStrings;
        const std::string missingKey = key + ":" +
            std::to_string(static_cast<unsigned>(kind));
        if (missingDialogueStringTables.contains(missingKey)) return nullptr;
        auto table = tables.find(key);
        if (table != tables.end()) return &table->second;
        importer::fnv::FalloutStringTable loaded;
        std::string error;
        if (!importer::fnv::loadFalloutStringTable(
                assets, pluginName, importer::fnv::falloutStringLanguage(),
                kind, loaded, error)) {
            missingDialogueStringTables.insert(missingKey);
            outReport.diagnostics.push_back(
                pluginName + " has no readable dialogue string table: " + error);
            return nullptr;
        }
        return &tables.emplace(key, std::move(loaded)).first->second;
    };

    for (auto& [resolvedBranch, winning] : winningBranches) {
        (void)resolvedBranch;
        const std::uint32_t questForm = loadOrder.remapFormId(
            winning.sourcePluginIndex, winning.definition.rawQuestFormId);
        const std::uint32_t startTopicForm = loadOrder.remapFormId(
            winning.sourcePluginIndex, winning.definition.rawStartingTopicFormId);
        const std::optional<ObjectId> questObject = referenceResolver(questForm);
        if (!questObject.has_value() ||
            questObject->kind != ObjectIdKind::PersistentReference ||
            !stableRecordKey(loadOrder, startTopicForm,
                winning.definition.startingTopic, outError)) {
            if (outError.empty()) outError = "dialogue branch quest owner has no stable identity";
            return false;
        }
        winning.definition.quest = questObject->reference;
        if (!session.registerDialogueBranch(winning.definition, outError)) return false;
        ++outReport.dialogueBranchesRegistered;
    }

    for (auto& [resolvedTopic, winning] : winningTopics) {
        (void)resolvedTopic;
        const std::uint32_t questForm = loadOrder.remapFormId(
            winning.sourcePluginIndex, winning.definition.rawQuestFormId);
        const std::optional<ObjectId> questObject = referenceResolver(questForm);
        if (!questObject.has_value() ||
            questObject->kind != ObjectIdKind::PersistentReference) {
            outError = "dialogue topic quest owner has no stable identity";
            return false;
        }
        winning.definition.quest = questObject->reference;
        if (winning.definition.rawBranchFormId != 0u &&
            !stableRecordKey(loadOrder,
                loadOrder.remapFormId(
                    winning.sourcePluginIndex, winning.definition.rawBranchFormId),
                winning.definition.branch, outError)) return false;
        if (winning.definition.promptStringId != 0u) {
            const auto* table = dialogueTable(
                winning.sourcePluginIndex,
                importer::fnv::FalloutStringFileKind::Strings);
            if (table != nullptr) {
                if (const std::string* prompt = table->find(
                        winning.definition.promptStringId)) {
                    winning.definition.prompt = *prompt;
                }
            }
        }
        if (!session.registerDialogueTopic(winning.definition, outError)) return false;
        ++outReport.dialogueTopicsRegistered;
    }

    std::vector<std::string> dialogueRootFunctions;
    std::set<std::string> attachedDialogueScripts;
    for (auto& [resolvedInfo, winning] : winningInfos) {
        SkyrimDialogueInfoDefinition& info = winning.definition;
        const std::size_t sourcePluginIndex = winning.sourcePluginIndex;
        const auto remapConditionForm = [&](std::uint32_t& formId) {
            if (formId != 0u) formId = loadOrder.remapFormId(sourcePluginIndex, formId);
        };
        for (Condition& condition : info.conditions) {
            switch (condition.function) {
                case 47u: case 58u: case 59u: case 71u: case 72u:
                case 84u: case 161u: case 359u: case 403u: case 629u:
                    remapConditionForm(condition.parameter1);
                    break;
                default: break;
            }
            if (condition.runOn == 2u) remapConditionForm(condition.reference);
        }
        const auto resolveLinked = [&](std::uint32_t raw, RecordKey& stable) {
            if (raw == 0u) return true;
            return stableRecordKey(loadOrder,
                loadOrder.remapFormId(sourcePluginIndex, raw), stable, outError);
        };
        if (!resolveLinked(info.rawResponseInfoFormId, info.responseInfo) ||
            !resolveLinked(info.rawPreviousInfoFormId, info.previousInfo)) return false;
        for (const std::uint32_t raw : info.rawLinkedTopicFormIds) {
            RecordKey linked;
            if (!resolveLinked(raw, linked)) return false;
            info.linkedTopics.push_back(std::move(linked));
        }
        const auto* table = dialogueTable(sourcePluginIndex,
            importer::fnv::FalloutStringFileKind::IlStrings);
        if (info.promptStringId != 0u) {
            const auto* promptTable = dialogueTable(sourcePluginIndex,
                importer::fnv::FalloutStringFileKind::Strings);
            if (promptTable != nullptr) {
                if (const std::string* prompt = promptTable->find(info.promptStringId)) {
                    info.prompt = *prompt;
                }
            }
        }
        for (SkyrimDialogueResponseDefinition& response : info.responses) {
            if (table != nullptr && response.textStringId != 0u) {
                if (const std::string* text = table->find(response.textStringId)) {
                    response.text = *text;
                }
            }
        }
        if (!session.registerDialogueInfo(info, outError)) return false;
        ++outReport.dialogueInfosRegistered;

        std::map<std::string, VmadScriptAttachment> attachments;
        for (const VmadScriptAttachment& attachment : info.scripts.common.scripts) {
            attachments.insert_or_assign(lowerAscii(attachment.className), attachment);
        }
        for (const VmadInfoFragment& fragment : info.scripts.fragments) {
            dialogueRootFunctions.push_back(
                lowerAscii(fragment.scriptClass + "." + fragment.function));
            attachments.try_emplace(lowerAscii(fragment.scriptClass),
                VmadScriptAttachment{fragment.scriptClass, 0u, {}});
            ++outReport.dialogueFragmentsLoaded;
        }
        for (const auto& [normalizedClass, attachment] : attachments) {
            std::vector<std::uint8_t> pexBytes;
            std::string error;
            if (!assets.resolveAsset(
                    "scripts\\" + attachment.className + ".pex", pexBytes, error)) {
                outError = "missing dialogue script " + attachment.className + ": " + error;
                return false;
            }
            PexScript pex;
            if (!readPexScript(pexBytes, pex, error)) {
                outError = "malformed dialogue script " + attachment.className + ": " + error;
                return false;
            }
            if (!session.papyrus().hasScriptClass(attachment.className)) {
                PexCompatibilityReport compatibility;
                if (!session.papyrus().loadPexScript(pex, false, compatibility, error)) {
                    outError = "could not load dialogue script " +
                        attachment.className + ": " + error;
                    return false;
                }
                loadedPexScripts.insert_or_assign(normalizedClass, pex);
            }
            std::unordered_map<std::string, PapyrusValue> properties;
            std::function<bool(const VmadValue&, PapyrusValue&)> convert;
            convert = [&](const VmadValue& value, PapyrusValue& converted) {
                if (value.type == VmadValueType::Object) {
                    if (value.object.formId == 0u) { converted = {}; return true; }
                    const std::uint32_t resolved = loadOrder.remapFormId(
                        sourcePluginIndex, value.object.formId);
                    if (value.object.alias != 0xffffu) {
                        const std::optional<ObjectId> questObject = referenceResolver(resolved);
                        const QuestRuntimeState* questState = questObject.has_value()
                            ? session.findQuest(*questObject) : nullptr;
                        if (questState == nullptr) return false;
                        const auto alias = std::find_if(
                            questState->aliases.begin(), questState->aliases.end(),
                            [&](const QuestAliasRuntimeState& candidate) {
                                return candidate.id == value.object.alias;
                            });
                        if (alias == questState->aliases.end()) return false;
                        converted = PapyrusValue::fromObject(alias->handle);
                        return true;
                    }
                    const std::optional<ObjectId> object = referenceResolver(resolved);
                    if (!object.has_value()) return false;
                    converted = PapyrusValue::fromObject(*object);
                    return true;
                }
                if (value.type == VmadValueType::String) {
                    converted = PapyrusValue::fromString(value.string); return true;
                }
                if (value.type == VmadValueType::Integer) {
                    converted = PapyrusValue::fromInteger(value.integer); return true;
                }
                if (value.type == VmadValueType::Float) {
                    converted = PapyrusValue::fromFloat(value.real); return true;
                }
                if (value.type == VmadValueType::Boolean) {
                    converted = PapyrusValue::fromBoolean(value.boolean); return true;
                }
                std::vector<PapyrusValue> values;
                for (const VmadValue& element : value.array) {
                    PapyrusValue convertedElement;
                    if (!convert(element, convertedElement)) return false;
                    values.push_back(std::move(convertedElement));
                }
                converted = PapyrusValue::fromArray(std::move(values));
                return true;
            };
            for (const VmadProperty& property : attachment.properties) {
                PapyrusValue value;
                if (!convert(property.value, value)) {
                    outError = "could not resolve dialogue VMAD property " +
                        attachment.className + "." + property.name;
                    return false;
                }
                properties.emplace(property.name, std::move(value));
            }
            const std::string instanceKey = info.record.toString() + ":" + normalizedClass;
            if (attachedDialogueScripts.insert(instanceKey).second &&
                !session.papyrus().attachScript(ObjectId::persistent(info.record),
                    attachment.className, std::move(properties), outError)) return false;
        }
    }

    // applyScenario establishes seed identities before retail definitions and
    // VMAD instances exist. Replay only script-backed startup stages now so
    // their authored fragments/objectives run exactly once against the fully
    // registered quest. MQ101 is intentionally excluded by scriptsRequired.
    for (const ScenarioQuestSeed& seed : scenario.prerequisiteQuests) {
        const auto questRecord = std::find_if(
            scenario.questRecords.begin(), scenario.questRecords.end(),
            [&](const ScenarioQuestRecord& value) {
                return lowerAscii(value.editorId) == lowerAscii(seed.editorId);
            });
        if (questRecord == scenario.questRecords.end() || !questRecord->scriptsRequired) continue;
        QuestRuntimeState* state = session.findQuest(ObjectId::persistent(
            makeRecordKey(questRecord->plugin, questRecord->localFormId)));
        if (state == nullptr) {
            outError = "scenario startup quest is not registered: " + seed.editorId;
            return false;
        }
        state->completedStages.erase(std::remove(
            state->completedStages.begin(), state->completedStages.end(), seed.stage),
            state->completedStages.end());
        session.setQuestStage(seed.editorId, seed.stage, seed.completed);
        outReport.diagnostics.push_back(
            "replayed authored startup stage " + seed.editorId + ":" +
            std::to_string(seed.stage));
    }
    outReport.diagnostics.push_back(
        "registered " + std::to_string(outReport.dialogueBranchesRegistered) +
        " Skyrim DLBR branches, " +
        std::to_string(outReport.dialogueTopicsRegistered) + " DIAL topics, " +
        std::to_string(outReport.dialogueInfosRegistered) +
        " INFO variants, and " + std::to_string(outReport.dialogueFragmentsLoaded) +
        " authored INFO fragments");

    std::vector<std::string> rootFunctions;
    for (const ScenarioQuestLoadDetail& detail : outReport.quests) {
        for (const std::string& scriptClass : detail.scriptClasses) {
            std::vector<std::string> functions = session.papyrus().functionsForClass(scriptClass);
            rootFunctions.insert(rootFunctions.end(),
                std::make_move_iterator(functions.begin()),
                std::make_move_iterator(functions.end()));
        }
    }
    rootFunctions.insert(rootFunctions.end(), dialogueRootFunctions.begin(),
        dialogueRootFunctions.end());
    std::sort(rootFunctions.begin(), rootFunctions.end());
    rootFunctions.erase(std::unique(rootFunctions.begin(), rootFunctions.end()),
        rootFunctions.end());

    // Resolve the compiled script-class closure reached by quest fragments.
    // Engine-native classes simply have no PEX asset and remain candidates for
    // the native registry; game-authored classes such as CWScript are loaded
    // here so strict diagnostics report their actual downstream requirements.
    std::set<std::string> attemptedScriptClasses;
    constexpr std::size_t kMaximumTransitiveClasses = 256u;
    for (;;) {
        const std::vector<std::string> bindings =
            session.papyrus().unresolvedCallBindings(rootFunctions);
        bool loadedAny = false;
        for (const std::string& binding : bindings) {
            const std::size_t arrow = binding.find(" -> ");
            if (arrow == std::string::npos) continue;
            const std::string target = binding.substr(arrow + 4u);
            const std::size_t separator = target.find('.');
            if (separator == std::string::npos || separator == 0u) continue;
            const std::string scriptClass = target.substr(0u, separator);
            const std::string normalizedClass = lowerAscii(scriptClass);
            if (!attemptedScriptClasses.insert(normalizedClass).second) continue;
            if (session.papyrus().hasScriptClass(scriptClass)) continue;
            if (attemptedScriptClasses.size() > kMaximumTransitiveClasses) {
                outError = "scenario script closure exceeds " +
                    std::to_string(kMaximumTransitiveClasses) + " classes";
                return false;
            }
            std::vector<std::uint8_t> pexBytes;
            std::string error;
            if (!assets.resolveAsset("scripts\\" + scriptClass + ".pex", pexBytes, error)) {
                continue;
            }
            PexScript pex;
            if (!readPexScript(pexBytes, pex, error)) {
                outError = "malformed transitive script scripts\\" + scriptClass +
                    ".pex: " + error;
                return false;
            }
            const auto targetFunction = std::find_if(
                pex.functions.begin(), pex.functions.end(), [&](const PexFunctionInfo& function) {
                    return lowerAscii(function.qualifiedName()) == lowerAscii(target);
                });
            if (targetFunction != pex.functions.end() && targetFunction->native()) {
                continue;
            }
            PexCompatibilityReport compatibility;
            if (!session.papyrus().loadPexScript(pex, false, compatibility, error)) {
                outError = "could not load transitive script scripts\\" + scriptClass +
                    ".pex: " + error;
                return false;
            }
            loadedPexScripts.insert_or_assign(normalizedClass, pex);
            outReport.transitiveScriptClasses.push_back(scriptClass);
            loadedAny = true;
        }
        if (!loadedAny) break;
    }
    std::sort(outReport.transitiveScriptClasses.begin(),
        outReport.transitiveScriptClasses.end());
    if (!outReport.transitiveScriptClasses.empty()) {
        outReport.diagnostics.push_back(
            "loaded " + std::to_string(outReport.transitiveScriptClasses.size()) +
            " transitive game-authored script classes");
    }

    std::set<std::string> transitiveClasses;
    for (const std::string& scriptClass : outReport.transitiveScriptClasses) {
        transitiveClasses.insert(lowerAscii(scriptClass));
    }
    std::vector<std::uint32_t> candidateOwnerFormIds;
    for (const auto& [propertyType, ownerFormIds] : propertyObjectOwners) {
        (void)propertyType;
        candidateOwnerFormIds.insert(candidateOwnerFormIds.end(),
            ownerFormIds.begin(), ownerFormIds.end());
    }
    std::sort(candidateOwnerFormIds.begin(), candidateOwnerFormIds.end());
    candidateOwnerFormIds.erase(
        std::unique(candidateOwnerFormIds.begin(), candidateOwnerFormIds.end()),
        candidateOwnerFormIds.end());
    for (const std::string& scriptClass : transitiveClasses) {
        std::vector<std::uint32_t> ownerFormIds = candidateOwnerFormIds;
        std::sort(ownerFormIds.begin(), ownerFormIds.end());
        ownerFormIds.erase(std::unique(ownerFormIds.begin(), ownerFormIds.end()),
            ownerFormIds.end());
        for (const std::uint32_t ownerFormId : ownerFormIds) {
            std::optional<VmadScriptAttachment> winningAttachment;
            std::optional<SkyrimQuestDefinition> winningQuestDefinition;
            std::size_t winningPluginIndex = 0u;
            std::size_t winningQuestPluginIndex = 0u;
            RecordKey ownerRecord;
            std::string ownerError;
            if (!stableRecordKey(loadOrder, ownerFormId, ownerRecord, ownerError)) continue;
            for (std::size_t pluginIndex = 0u;
                 pluginIndex < loadOrder.entries().size(); ++pluginIndex) {
                importer::fnv::EsmReader reader;
                if (!reader.open(loadOrder.entries()[pluginIndex].path)) continue;
                importer::fnv::EsmReader::Visitor visitor;
                visitor.onRecordHeader = [&](const importer::fnv::EsmRecordHeaderView& header) {
                    return loadOrder.remapFormId(pluginIndex, header.formId) == ownerFormId;
                };
                std::string vmadError;
                visitor.onRecord = [&](const importer::fnv::EsmRecordView& record) {
                    if (record.type == "QUST") {
                        SkyrimQuestDefinition definition;
                        if (!readSkyrimQuest(record, ownerRecord, definition, vmadError)) return;
                        winningQuestDefinition = std::move(definition);
                        winningQuestPluginIndex = pluginIndex;
                    }
                    for (const importer::fnv::EsmSubrecordView& subrecord : record.subrecords) {
                        if (subrecord.type != "VMAD") continue;
                        VmadAttachments attachments;
                        if (!readVmadAttachments(
                                std::span<const std::uint8_t>(subrecord.data, subrecord.size),
                                attachments, vmadError)) {
                            return;
                        }
                        const auto attached = std::find_if(
                            attachments.scripts.begin(), attachments.scripts.end(),
                            [&](const VmadScriptAttachment& candidate) {
                                return lowerAscii(candidate.className) == scriptClass;
                            });
                        if (attached != attachments.scripts.end()) {
                            winningAttachment = *attached;
                            winningPluginIndex = pluginIndex;
                        }
                    }
                };
                if (!reader.walk(visitor) || !vmadError.empty()) {
                    outError = vmadError.empty() ? reader.lastError() : vmadError;
                    return false;
                }
            }
            if (!winningAttachment.has_value()) continue;

            if (winningQuestDefinition.has_value() &&
                session.findQuest(ObjectId::persistent(ownerRecord)) == nullptr) {
                remapQuestAliasForms(*winningQuestDefinition, winningQuestPluginIndex);
                if (!session.registerQuestDefinition(
                        *winningQuestDefinition, referenceResolver, outError)) {
                    return false;
                }
            }
            const auto transitivePex = loadedPexScripts.find(scriptClass);
            if (transitivePex == loadedPexScripts.end() ||
                !registerAttachmentRuntimeRecords(
                    transitivePex->second, *winningAttachment,
                    winningPluginIndex, outError)) {
                if (outError.empty()) {
                    outError = "transitive script PEX metadata is unavailable";
                }
                return false;
            }

            std::unordered_map<std::string, PapyrusValue> properties;
            std::function<bool(const VmadValue&, PapyrusValue&)> convert;
            convert = [&](const VmadValue& value, PapyrusValue& converted) {
                switch (value.type) {
                    case VmadValueType::Object: {
                        if (value.object.formId == 0u) { converted = {}; return true; }
                        const std::uint32_t resolved = loadOrder.remapFormId(
                            winningPluginIndex, value.object.formId);
                        if (value.object.alias != 0xffffu) {
                            std::string aliasError;
                            if (!ensureQuestDefinition(resolved, aliasError)) return false;
                            const std::optional<ObjectId> aliasQuestObject =
                                referenceResolver(resolved);
                            if (!aliasQuestObject.has_value()) return false;
                            const QuestRuntimeState* ownerQuest =
                                session.findQuest(*aliasQuestObject);
                            if (ownerQuest == nullptr) return false;
                            const auto alias = std::find_if(
                                ownerQuest->aliases.begin(), ownerQuest->aliases.end(),
                                [&](const QuestAliasRuntimeState& candidate) {
                                    return candidate.id == value.object.alias;
                                });
                            if (alias == ownerQuest->aliases.end()) return false;
                            converted = PapyrusValue::fromObject(alias->handle);
                            return true;
                        }
                        const std::optional<ObjectId> object = referenceResolver(resolved);
                        if (!object.has_value()) return false;
                        converted = PapyrusValue::fromObject(*object);
                        return true;
                    }
                    case VmadValueType::String:
                        converted = PapyrusValue::fromString(value.string); return true;
                    case VmadValueType::Integer:
                        converted = PapyrusValue::fromInteger(value.integer); return true;
                    case VmadValueType::Float:
                        converted = PapyrusValue::fromFloat(value.real); return true;
                    case VmadValueType::Boolean:
                        converted = PapyrusValue::fromBoolean(value.boolean); return true;
                    case VmadValueType::ObjectArray:
                    case VmadValueType::StringArray:
                    case VmadValueType::IntegerArray:
                    case VmadValueType::FloatArray:
                    case VmadValueType::BooleanArray: {
                        std::vector<PapyrusValue> elements;
                        for (const VmadValue& element : value.array) {
                            PapyrusValue convertedElement;
                            if (!convert(element, convertedElement)) return false;
                            elements.push_back(std::move(convertedElement));
                        }
                        converted = PapyrusValue::fromArray(std::move(elements));
                        return true;
                    }
                }
                return false;
            };
            for (const VmadProperty& property : winningAttachment->properties) {
                PapyrusValue converted;
                if (!convert(property.value, converted)) {
                    outError = "could not resolve transitive VMAD property " +
                        winningAttachment->className + "." + property.name +
                        " (type " + std::to_string(static_cast<unsigned>(property.value.type)) +
                        ", form " + std::to_string(property.value.object.formId) +
                        ", alias " + std::to_string(property.value.object.alias) + ")";
                    return false;
                }
                properties.emplace(property.name, std::move(converted));
            }
            const std::optional<ObjectId> owner = referenceResolver(ownerFormId);
            if (!owner.has_value() || !session.papyrus().attachScript(
                    *owner, winningAttachment->className, std::move(properties), outError)) {
                if (outError.empty()) outError = "could not resolve transitive script owner";
                return false;
            }
            ++outReport.transitiveScriptInstances;
        }
    }
    if (outReport.transitiveScriptInstances != 0u) {
        outReport.diagnostics.push_back(
            "attached " + std::to_string(outReport.transitiveScriptInstances) +
            " transitive winning-record VMAD script instances");
    }
    outReport.unresolvedCallBindings =
        session.papyrus().unresolvedCallBindings(rootFunctions);
    for (ScenarioQuestLoadDetail& detail : outReport.quests) {
        detail.unresolvedCalls = 0u;
        for (const std::string& binding : outReport.unresolvedCallBindings) {
            const std::string loweredBinding = lowerAscii(binding);
            const bool owned = std::any_of(
                detail.scriptClasses.begin(), detail.scriptClasses.end(),
                [&](const std::string& scriptClass) {
                    return loweredBinding.starts_with(lowerAscii(scriptClass) + ".");
                });
            if (owned) ++detail.unresolvedCalls;
        }
    }
    if (!outReport.unresolvedCallBindings.empty()) {
        outReport.diagnostics.push_back(
            "scenario VM requires " + std::to_string(outReport.unresolvedCallBindings.size()) +
            " unresolved call bindings after native and script resolution");
    }
    outReport.locationsRegistered = session.locations().size();
    outReport.globalVariablesRegistered = session.globalVariables().size();
    outReport.diagnostics.push_back(
        "registered " + std::to_string(outReport.locationsRegistered) +
        " LCTN locations and " + std::to_string(outReport.globalVariablesRegistered) +
        " GLOB values for reachable native state");
    if (!outReport.transitiveScriptClasses.empty() &&
        outReport.transitiveScriptInstances < outReport.transitiveScriptClasses.size()) {
        outReport.runtimeBlockers.push_back(
            "one or more reachable game-authored scripts lack a winning-record VMAD instance");
    }
    outReport.runtimeBlockers.insert(outReport.runtimeBlockers.end(), {
        "resident NAVM borders stitch geometrically and same-space teleport portals execute "
        "as typed actions; cross-space package transitions and Skyrim explicit preferred "
        "links are incomplete",
        "Actor.ShowGiftMenu persists and presents deterministic actor inventory, but Papyrus "
        "latent resumption and favor-point value budgets are incomplete",
        "Jolt-owned combat and VMAD-derived persistent claw-puzzle state are available, but "
        "retail faction hostility, death/ragdoll, authored claw-door HKX motion/audio, general "
        "container/leveled loot and equipment, and full HKX graph binding are incomplete"});
    outError.clear();
    return true;
}

}  // namespace odai::bethesda
