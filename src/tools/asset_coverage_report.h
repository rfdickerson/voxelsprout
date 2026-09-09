#pragma once
#include <nlohmann/json.hpp>
#include <string>

// Reporting policy operates on measured probe evidence; it never infers GPU use
// from decoding, or counts a block occurrence as a visible failure.
std::string nifCoverageFailureStatus(const std::string &diagnostic);
void finalizeAssetCoverage(nlohmann::json &report);
std::string assetCoverageMarkdown(const nlohmann::json &report);
