#include "bethesda/skyrim_scene.h"
#include <cstring>
#include <cmath>

namespace odai::bethesda {
bool readSkyrimScene(const importer::bethesda::EsmRecordView& record,
    RecordKey stable, SkyrimSceneDefinition& out, std::string& error) {
    out = {}; out.record = std::move(stable);
    SkyrimScenePhase* phase = nullptr;
    SkyrimSceneAction* action = nullptr;
    unsigned conditionSection = 0;
    bool endPhaseSeen = false;
    for (const auto& sub : record.subrecords) {
        const auto u32 = [&]() { std::uint32_t value = 0; if (sub.size >= 4) std::memcpy(&value, sub.data, 4); return value; };
        const auto str = [&]() { return std::string(reinterpret_cast<const char*>(sub.data), sub.size && sub.data[sub.size-1] == 0 ? sub.size-1 : sub.size); };
        if (sub.type == "EDID") out.editorId = str();
        else if (sub.type == "VMAD") {
            if (!readVmadSceneFragments({sub.data, sub.size}, out.scripts, out.fragments, error)) return false;
        } else if (sub.type == "HNAM") {
            if (phase) phase = nullptr;
            else { out.phases.emplace_back(); phase = &out.phases.back(); conditionSection = 0; }
        } else if (sub.type == "ANAM") {
            if (!sub.size) action = nullptr;
            else if (sub.size == 2) {
                out.actions.emplace_back(); action = &out.actions.back();
                action->type = sub.data[0] | (sub.data[1] << 8); endPhaseSeen = false;
                if (action->type > 2) { error = "unsupported SCEN action type"; return false; }
            } else { error = "invalid SCEN ANAM size"; return false; }
        } else if (phase) {
            auto& conditions = conditionSection == 0 ? phase->startConditions : phase->completionConditions;
            if (sub.type == "NAM0") phase->name = str();
            else if (sub.type == "NEXT") ++conditionSection;
            else if (sub.type == "CTDA") {
                Condition condition;
                if (!readCondition({sub.data, sub.size}, condition, error)) return false;
                // Alias run-on stores its alias index in the final CTDA word.
                if (condition.runOn == 5 && sub.size >= 32) std::memcpy(&condition.reference, sub.data+28, 4);
                conditions.push_back(std::move(condition));
            } else if ((sub.type == "CIS1" || sub.type == "CIS2") && !conditions.empty()) {
                (sub.type == "CIS1" ? conditions.back().stringParameter1 : conditions.back().stringParameter2) = str();
            }
        } else if (action) {
            if (sub.type == "ALID") action->actorAlias = static_cast<std::int32_t>(u32());
            else if (sub.type == "INAM") action->index = u32();
            else if (sub.type == "FNAM") action->flags = u32();
            else if (sub.type == "SNAM") {
                if (endPhaseSeen && action->type == 2) {
                    if (sub.size != 4) { error = "invalid SCEN timer"; return false; }
                    std::memcpy(&action->seconds, sub.data, 4);
                    if (!std::isfinite(action->seconds) || action->seconds < 0) { error = "invalid SCEN duration"; return false; }
                } else action->startPhase = u32();
            } else if (sub.type == "ENAM") { action->endPhase = u32(); endPhaseSeen = true; }
            else if (sub.type == "DATA") action->rawTopic = u32();
            else if (sub.type == "PNAM") action->rawPackages.push_back(u32());
        } else if (sub.type == "PNAM") out.rawQuest = u32();
        else if (sub.type == "FNAM") out.flags = u32();
    }
    if (phase || action || !out.rawQuest) { error = "incomplete SCEN record"; return false; }
    for (const auto& value : out.actions) if (value.startPhase > value.endPhase || value.endPhase >= out.phases.size()) {
        error = "SCEN action phase out of range"; return false;
    }
    error.clear(); return true;
}
} // namespace odai::bethesda
