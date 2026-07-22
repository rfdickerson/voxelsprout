-- Need schedules: which destination categories a citizen visits, weighted by
-- trait. trait "any" applies to every citizen; other traits only to citizens
-- who have them. Categories must match what the business name generator in
-- 00_names.lua can return (plus the civic categories "park" and "school").
-- Commuting to work is built into the engine and always dominant.

-- Everyone.
Needs.register{ trait = "any", category = "cafe",    weight = 2.0 }
Needs.register{ trait = "any", category = "grocery", weight = 2.0 }
Needs.register{ trait = "any", category = "diner",   weight = 1.2 }
Needs.register{ trait = "any", category = "park",    weight = 1.5 }
Needs.register{ trait = "any", category = "salon",   weight = 0.6 }
Needs.register{ trait = "any", category = "bookshop", weight = 0.5 }
Needs.register{ trait = "any", category = "cinema",  weight = 0.8 }
Needs.register{ trait = "any", category = "laundry", weight = 0.7 }
Needs.register{ trait = "any", category = "repair",  weight = 0.4 }
Needs.register{ trait = "any", category = "boutique", weight = 0.5 }
Needs.register{ trait = "any", category = "petstore", weight = 0.4 }

-- The fit crowd.
Needs.register{ trait = "fit", category = "gym",  weight = 2.5 }
Needs.register{ trait = "fit", category = "yoga", weight = 2.5 }

-- Parents.
Needs.register{ trait = "parent", category = "daycare", weight = 3.5 }
Needs.register{ trait = "parent", category = "school",  weight = 1.5 }
Needs.register{ trait = "parent", category = "arcade",  weight = 1.0 }

-- Night owls.
Needs.register{ trait = "nightowl", category = "bar",    weight = 2.5 }
Needs.register{ trait = "nightowl", category = "arcade", weight = 1.2 }
Needs.register{ trait = "nightowl", category = "diner",  weight = 1.0 }
