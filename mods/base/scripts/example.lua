-- Example mod script for the base game.
--
-- This file demonstrates the scripting API without changing the base balance:
-- it only logs notable events. Copy these patterns into your own mod's
-- mods/<id>/scripts/*.lua to react to gameplay or author custom effects.
--
-- Available globals (all sandboxed -- no os/io/filesystem access):
--   Events.on(event, fn)      subscribe to: turn_start, turn_end, city_yields,
--                             building_built, wonder_built, tech_researched,
--                             city_founded
--   Effects.register(id, fn)  per-building/wonder yield callback (see bottom)
--   Game.turn() / Game.empire(i) / Game.city(i) / Game.num_empires() / num_cities()
--   Rng.int(lo, hi) / Rng.number()   deterministic, seeded from the world seed
--   Log.info(msg)             diagnostic logging (stderr; never the event log)

-- Greet once at the start of the game.
Events.on("turn_start", function(world)
  if world.turn == 0 then
    Log.info("base mod loaded -- " .. world:num_empires() .. " empires on turn 0")
  end
end)

-- Announce wonders as they are completed.
Events.on("wonder_built", function(empire, wonderId)
  Log.info(empire.name .. " completed wonder '" .. wonderId .. "'")
end)

-- Announce the player's (empire 1) discoveries.
Events.on("tech_researched", function(empire, techId)
  if empire.id == 1 then
    Log.info(empire.leader_name .. " of " .. empire.name .. " researched '" .. techId .. "'")
  end
end)

-- ---------------------------------------------------------------------------
-- Effects.register example (commented out so the base game stays vanilla).
-- Uncomment to give every city of an empire that owns the Great Library an
-- extra +1 science on top of the declarative effect in buildings.json:
--
-- Effects.register("great_library", function(ctx)
--   ctx.science = ctx.science + 1
-- end)
--
-- The callback receives a YieldContext (ctx) whose numeric fields you may adjust
-- before the city's percentage multipliers apply: ctx.food/production/gold/
-- science/culture, ctx.prodPct/goldPct/sciencePct, ctx.happy, ctx.growBonus.
-- Read-only context: ctx.city, ctx.empire.
-- ---------------------------------------------------------------------------
