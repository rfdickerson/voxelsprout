-- Citizen story templates. The C++ event engine rolls these monthly and
-- interpolates the placeholders from the live roster and city:
--   {a}       the citizen's full name
--   {b}       another citizen's full name (spouse/affair partner when relevant)
--   {family}  the citizen's last name
--   {place}   a named business or civic destination
--   {street}  a named street
--
-- kind: "opening" | "arrival" | "departure" | "life" | "drama"
-- weight: relative pick probability within its kind
-- requires: tags that must all hold — traits ("fit","parent","gossip",
--   "nightowl") or state ("married","affair")
-- Tone contract: PG-13 tabloid. Wink, don't leer.

-- Business openings (fired when a commercial lot first develops).
Stories.register{ id = "open_doors", kind = "opening", weight = 3,
  text = "{place} opens its doors on {street}" }
Stories.register{ id = "open_ribbon", kind = "opening", weight = 2,
  text = "Ribbon cut at {place} — half the town showed up for the free cake" }
Stories.register{ id = "open_skeptic", kind = "opening", weight = 1,
  text = "New in town: {place}. {a} gives it a month" }

-- Arrivals and departures (roster churn).
Stories.register{ id = "arrive_family", kind = "arrival", weight = 3,
  text = "The {family} family moves in on {street}" }
Stories.register{ id = "arrive_single", kind = "arrival", weight = 2,
  text = "{a} arrives in town with two suitcases and big plans" }
Stories.register{ id = "depart_quiet", kind = "departure", weight = 3,
  text = "{a} packed up and left town overnight" }
Stories.register{ id = "depart_bitter", kind = "departure", weight = 1,
  text = "{a} leaves town, citing 'everything about this place'" }

-- Everyday life.
Stories.register{ id = "life_highscore", kind = "life", weight = 2,
  text = "{a} finally beat the high score at {place}" }
Stories.register{ id = "life_dog", kind = "life", weight = 2,
  text = "{a} adopted a scruffy terrier from behind {place}" }
Stories.register{ id = "life_chili", kind = "life", weight = 2,
  text = "{a} won the {place} chili cook-off. Again" }
Stories.register{ id = "life_band", kind = "life", weight = 1,
  text = "{a} started a band. Neighbors on {street} have opinions" }
Stories.register{ id = "life_garden", kind = "life", weight = 1,
  text = "{a}'s prize pumpkin mysteriously vanished from {street}" }
Stories.register{ id = "life_yoga", kind = "life", weight = 2, requires = { "fit" },
  text = "{a} hasn't missed a sunrise class at {place} all month" }
Stories.register{ id = "life_marathon", kind = "life", weight = 1, requires = { "fit" },
  text = "{a} is 'training for something big', tells everyone at {place}" }
Stories.register{ id = "life_daycare", kind = "life", weight = 2, requires = { "parent" },
  text = "{a} late for pickup at {place} — again" }
Stories.register{ id = "life_bakesale", kind = "life", weight = 1, requires = { "parent" },
  text = "{a}'s bake-sale brownies cleared out {place} in minutes" }
Stories.register{ id = "life_closer", kind = "life", weight = 1, requires = { "nightowl" },
  text = "{a} closed down {place} again last night" }

-- Weekend beats (fired Saturday mornings, anchored to a park or other spot).
Stories.register{ id = "weekend_soccer", kind = "weekend", weight = 3,
  text = "Saturday soccer at {place} — {a}'s squad wins on a golden goal" }
Stories.register{ id = "weekend_soccer_loss", kind = "weekend", weight = 1,
  text = "Tough morning at {place}: {a}'s team loses 7-nil, spirits undimmed" }
Stories.register{ id = "weekend_market", kind = "weekend", weight = 2,
  text = "The farmers market takes over {street} until noon" }
Stories.register{ id = "weekend_picnic", kind = "weekend", weight = 2,
  text = "The {family} family picnic claims the best spot at {place}" }
Stories.register{ id = "weekend_kite", kind = "weekend", weight = 1,
  text = "{a}'s kite is in a tree at {place}. Again" }

-- Tabloid drama. The engine only rolls these for citizens whose tags match, and
-- an "affair" template both reports and establishes the link.
Stories.register{ id = "drama_seen", kind = "drama", weight = 2, requires = { "gossip", "married" },
  text = "{a} seen leaving {place} with {b} — again" }
Stories.register{ id = "drama_friends", kind = "drama", weight = 1, requires = { "affair" },
  text = "{a} and {b} suddenly 'just friends', sources say" }
Stories.register{ id = "drama_voices", kind = "drama", weight = 1, requires = { "married" },
  text = "Neighbors report raised voices at the {family} house on {street}" }
Stories.register{ id = "drama_flowers", kind = "drama", weight = 1, requires = { "affair" },
  text = "A very large flower delivery arrived at the {family} house. Card unsigned" }
