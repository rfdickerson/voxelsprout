#pragma once

#include "game/game_sim.h"
#include "game/units.h"

// AI military decision layer. Bridges the economy layer (World / Empire / City)
// with the tactical layer (GameState / Unit) to give AI empires military behavior
// tuned to their leader's personality archetype.
//
// Call once per turn, after stepTurn() has run (which fills World::pendingUnits
// and refreshes city yields) and after those pending units have been spawned into
// the GameState. Then call advanceTurn() so the paths issued here execute.
namespace odai::game {

// Issue military production orders and movement/attack orders for all AI-managed
// empires. Production orders override idle city queues; movement orders set
// Unit::path so advanceTurn() advances units along them next turn.
void stepAiUnits(World& world, GameState& gs, std::uint8_t playerOwner);

}  // namespace odai::game
