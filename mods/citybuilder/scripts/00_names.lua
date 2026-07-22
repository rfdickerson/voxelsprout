-- Name generators for the city builder. Registered functions receive a seeded
-- rng handle (rng:int(lo,hi), rng:number(), rng:chance(p), rng:pick(table)).
-- Same seed => same name; do NOT retain the rng past the call.
--
-- Names.register accepts a partial table; keys you omit keep the previous (or
-- built-in fallback) generator. business must return {name=..., category=...}
-- where category is one of the destination categories the citizen sim routes
-- to (cafe, yoga, daycare, ... — see 20_needs.lua).

local cityPre = {
  "Cedar", "Marrow", "Ash", "Birch", "Harlow", "Wren", "Alder", "Fall",
  "Stone", "Bright", "Copper", "Gull", "Iron", "Maple", "Norwood", "Pell",
  "Quill", "Rook", "Sorrel", "Tarn", "Bram", "Clay", "Dun", "Ever",
}
local citySuf = {
  "field", "brook", "haven", "ford", "mont", "dale", "port", "ton", "bury",
  "view", " Falls", " Springs", " Heights", " Grove", " Hollow", " Junction",
}

local streetNames = {
  "Maple", "Oak", "Elm", "Birch", "Juniper", "Willow", "Chestnut", "Larch",
  "Main", "Harbor", "Mill", "Depot", "Foundry", "Orchard", "Prospect",
  "Garfield", "Sumner", "Whitmore", "Calloway", "Endicott", "Hollis",
}
local streetTypes = { " St", " St", " Ave", " Ave", " Rd", " Lane", " Blvd", " Way" }

local firstF = {
  "Marla", "June", "Opal", "Cora", "Hazel", "Ruth", "Vera", "Ida", "Nell",
  "Pearl", "Etta", "Mae", "Sylvie", "Dot", "Flo", "Greta", "Lois", "Wilma",
}
local firstM = {
  "Ed", "Gus", "Ray", "Cal", "Ned", "Amos", "Frank", "Walt", "Hank", "Roy",
  "Vern", "Earl", "Otis", "Chet", "Mort", "Silas", "Buck", "Ira",
}
local lastNames = {
  "Voss", "Krane", "Mercer", "Holt", "Bram", "Tully", "Ashford", "Pike",
  "Snell", "Dunmore", "Fairley", "Ostrander", "Quimby", "Rourke", "Slade",
  "Thorne", "Umber", "Wexley", "Yardley", "Zoller", "Cobb", "Draper",
}

-- Business name grammars per category. {L} = a last name, {C} = a city-ish
-- word. Tier/era nudge which grammar fires so an 1890s mill and a 1960s yoga
-- studio read differently.
local commercial = {
  cafe     = { "{L}'s Cafe", "The {C} Bean", "Sunrise Cafe", "{L} & Daughters Coffee" },
  diner    = { "{L}'s Diner", "The {C} Grill", "Blue Plate Diner", "Starlite Diner" },
  gym      = { "Iron {C} Fitness", "{L}'s Gym", "Powerhouse Gym", "The Sweatbox" },
  yoga     = { "Sunrise Yoga", "{C} Flow Yoga", "Still Waters Yoga", "Lotus & {C}" },
  daycare  = { "Little Sprouts Daycare", "{C} Tots", "Sunny Days Daycare", "Wee Care" },
  salon    = { "{L}'s Salon", "The {C} Clip", "Shear Genius", "Curl Up & Dye" },
  arcade   = { "Pixel Palace", "The {C} Arcade", "Token Alley", "Joystick Junction" },
  bookshop = { "{L}'s Books", "The Dusty Jacket", "{C} Pages", "Chapter & Verse" },
  grocery  = { "{L}'s Grocery", "{C} Market", "Corner Greens", "The Pantry" },
  bar      = { "The {C} Tap", "{L}'s Tavern", "Moonlight Lanes", "The Rusty Anchor" },
  cinema   = { "The {C} Bijou", "Roxy Theater", "{C} Picture House", "The Marquee" },
  petstore = { "{C} Paws", "{L}'s Pets", "Feather & Fin", "The Barking Lot" },
  laundry  = { "{C} Suds", "{L}'s Laundry", "The Wash House", "Spin City" },
  repair   = { "{L} & Sons Repair", "{C} Fix-It", "The Tinker Shop", "Ace Repair" },
  boutique = { "Maison {L}", "{C} Threads", "The Gilded Hem", "{L}'s Finery" },
}
local industrial = {
  mill    = { "{C} Mill", "{L} Milling Co.", "Old {C} Gristworks" },
  foundry = { "{L} Foundry", "{C} Ironworks", "Consolidated {C} Steel" },
  depot   = { "{C} Freight Depot", "{L} Haulage", "Interstate Depot 9" },
  plant   = { "{C} Works", "{L} Manufacturing", "Apex {C} Plant" },
  yard    = { "{L} Lumber Yard", "{C} Salvage", "Bayside Scrapyard" },
}

local function expand(rng, template)
  local out = template:gsub("{L}", function() return rng:pick(lastNames) end)
  out = out:gsub("{C}", function() return rng:pick(cityPre) end)
  return out
end

local blockByTier = {
  [0] = { "{L} Trailer Court", "{C} Flats", "Lowside Rows" },
  [1] = { "{L} Rowhouses", "{C} Terrace", "{C} Commons" },
  [2] = { "{L} Estates", "{C} Crest", "The {C} Gardens" },
}

Names.register{
  city = function(rng)
    return rng:pick(cityPre) .. rng:pick(citySuf)
  end,

  street = function(rng)
    return rng:pick(streetNames) .. rng:pick(streetTypes)
  end,

  first = function(rng, feminine)
    if feminine then return rng:pick(firstF) end
    return rng:pick(firstM)
  end,

  last = function(rng)
    return rng:pick(lastNames)
  end,

  block = function(rng, tier)
    local pool = blockByTier[tier] or blockByTier[1]
    return expand(rng, rng:pick(pool))
  end,

  -- kind is "commercial" or "industrial"; tier 0..2; era 0 (1890s), 1 (1930s),
  -- 2 (1960s). Returns {name=..., category=...}.
  business = function(rng, kind, tier, era)
    local pool = (kind == "industrial") and industrial or commercial
    -- Draw the category first (deterministic per seed), then a grammar for it.
    local cats = {}
    for cat in pairs(pool) do cats[#cats + 1] = cat end
    table.sort(cats)  -- pairs() order is undefined; sort for determinism
    local cat = rng:pick(cats)
    local name = expand(rng, rng:pick(pool[cat]))
    -- Old-money flourish for wealthy 1890s storefronts.
    if kind == "commercial" and tier >= 2 and era == 0 and rng:chance(0.3) then
      name = name .. " Est. 18" .. rng:int(55, 90)
    end
    return { name = name, category = cat }
  end,
}
