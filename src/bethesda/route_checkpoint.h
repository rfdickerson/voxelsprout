#pragma once
#include "bethesda/bethesda_session.h"
#include <nlohmann/json.hpp>

namespace odai::bethesda {
// Read-only observations, never a certificate of ordinary-input quest completion.
nlohmann::json bleakFallsRouteContract();
nlohmann::json scenarioStartContract(const std::string& id);
nlohmann::json assessScenarioStart(const BethesdaSession& session);
nlohmann::json routeProfileMetadata(const importer::bethesda::ResolvedContentProfile& profile);
nlohmann::json routeCheckpoint(const BethesdaSession& session);
nlohmann::json assessBleakFallsStart(const BethesdaSession& session);
}
