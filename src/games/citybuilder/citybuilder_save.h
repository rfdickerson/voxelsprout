#pragma once

// Versioned binary save/load for the city builder.
//
// Modelled on src/game/strategy_map_io.cc, the tree's existing versioned binary
// writer — not JSON, because a 56x56 tile grid plus a citizen roster is bulk
// data, not configuration.
//
// The one rule that matters here: Tile is written FIELD BY FIELD, never as a
// raw struct blob. Tile has padding and it has grown repeatedly (trafficLoad,
// charTicks and zoneAge are all recent additions), so an fwrite of the struct
// would bake this week's layout and this compiler's padding into every save
// file. Field-by-field costs a few hundred lines of nothing and buys forward
// compatibility.
//
// Deliberately NOT saved, and regenerated instead:
//   * street / block / business names — pure functions of m_worldSeed and
//     position, so regenerating them on load is a free determinism check: if
//     they come back different, the seed did not round-trip.
//   * every ambient agent (cars, pedestrians, boats, sims, fire trucks, drops,
//     fx, rise animations) — decoration that respawns within a second.
//   * the parcel layout — recomputeParcels() derives it from the zone map.
//
// The citizen roster IS saved, RNG state and counter included: skip those and a
// reloaded city desyncs from the story stream, which is exactly the sort of bug
// that looks like "the tabloid went weird" three sessions later.

#include <string>

namespace odai::games::citybuilder {

class CityBuilderApp;

// Both return false and leave the target untouched on any failure. The reason
// is available from lastSaveError() rather than thrown — a failed save must
// never take the running game down with it.
bool saveCity(const CityBuilderApp& app, const std::string& path);
bool loadCity(CityBuilderApp& app, const std::string& path);

[[nodiscard]] const std::string& lastSaveError();

}  // namespace odai::games::citybuilder
