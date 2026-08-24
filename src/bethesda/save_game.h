#pragma once

#include "bethesda/runtime_ids.h"

#include <filesystem>
#include <functional>
#include <string>
#include <vector>

namespace odai::bethesda {

class BethesdaSession;

struct SaveLoadOptions {
    // Required to reconcile a save whose content fingerprint differs from the
    // configured session. Every RecordKey referenced by the save must resolve;
    // otherwise loading fails without mutating the session.
    std::function<bool(const RecordKey&)> recordAvailable;
};

struct SaveLoadReport {
    bool contentReconciled = false;
    bool recoveredPrevious = false;
    std::vector<std::string> diagnostics;
};

bool saveOdaiGameAtomic(
    const std::filesystem::path& path,
    const BethesdaSession& session,
    std::string& outError);

bool loadOdaiGame(
    const std::filesystem::path& path,
    BethesdaSession& session,
    const SaveLoadOptions& options,
    SaveLoadReport& outReport,
    std::string& outError);

}  // namespace odai::bethesda
