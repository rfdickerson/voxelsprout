#pragma once

#include "bethesda/runtime_world.h"
#include "import/bethesda/plugin_load_order.h"
#include <set>

namespace odai::bethesda {

// Materialize only installed placed references reached by quest/script bindings.
// This creates persistent gameplay state, not rendering or physics residency.
// Existing state is authoritative; revisiting or loading content cannot reset it.
bool materializeSkyrimPersistentReferences(const importer::bethesda::FalloutLoadOrder &order,
                                           const std::set<std::uint32_t> &references,
                                           BethesdaWorld &world, std::string &error);

} // namespace odai::bethesda
