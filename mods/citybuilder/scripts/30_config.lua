-- Tuning config, read once at load. Every key is optional — omitted keys keep
-- the engine's compiled-in defaults (shown here). Values are numbers only.

Config.terrain{
  land_min        = 0.55,  -- reject maps with less buildable land than this
  river_width_min = 2,     -- carved river half-width varies in this range
  river_width_max = 3,
  lake_max        = 2,     -- 0..N seeded lakes
  coast_chance    = 0.25,  -- chance one map edge floods into a coastline
  forest_freq     = 0.09,  -- fbm frequency of the tree-density mask
}

Config.scatter{
  hydrant_per_mille  = 120,  -- per-mille chance per eligible road tile
  bench_per_mille    = 260,  -- near parks / high-desirability commercial
  billboard_per_mille = 180, -- rear corners of developed level-2+ C/I plots
  bus_stop_per_mille = 140,  -- road tiles fronting dense commercial
}
