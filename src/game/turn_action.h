#pragma once

// The single action the player must take next to make progress. The smart turn
// button surfaces the highest-priority unmet requirement so "what do I do next"
// is always answered by one button. Pure logic — no engine/UI dependencies, so
// the priority ordering can be unit-tested directly.
namespace odai::game {

enum class TurnAction {
    ChooseResearch,    // no research target picked yet
    ChooseProduction,  // a player city has an empty production queue
    UnitNeedsOrders,   // a player unit still has moves and no orders
    NextTurn,          // nothing blocks — advance the turn
};

// First unmet requirement wins, in the order above. `researchChosen` is false
// when the player has not selected a research target; `idleCities` / `idleUnits`
// count player cities with empty production and units still awaiting orders.
inline TurnAction nextTurnAction(bool researchChosen, int idleCities, int idleUnits) {
    if (!researchChosen) return TurnAction::ChooseResearch;
    if (idleCities > 0)  return TurnAction::ChooseProduction;
    if (idleUnits > 0)   return TurnAction::UnitNeedsOrders;
    return TurnAction::NextTurn;
}

}  // namespace odai::game
