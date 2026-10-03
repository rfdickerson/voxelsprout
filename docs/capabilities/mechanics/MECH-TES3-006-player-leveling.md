# MECH-TES3-006: Morrowind Player Leveling

Status: Partial

## Goal

The player earns and confirms character levels through TES3 skill advancement,
resting, and attribute choices. A character created through MECH-TES3-001 can
reach level 3 through ordinary gameplay and satisfy Caius's level prerequisite
in MECH-TES3-002 without a stat override or permissive dialogue condition.

## Dependencies

- MECH-TES3-001: persistent race, class, specialization, attributes, and skills
- TES3 CLAS, SKIL, GMST, book, trainer, and player-stat data
- Gameplay skill-use events, trainer transactions, and skill-book reading
- Legal sleeping/resting, time advancement, and the retained RPG UI
- Gameplay save/load and HARNESS-002, HARNESS-003, and HARNESS-006

MECH-TES3-002 consumes the resulting level; it is not a prerequisite for the
progression calculation. Missing skill-use producers or rest/service behavior
are implementation dependencies, not grounds to certify the complete capability.

## Required Behavior

### Skill advancement

- Class major/minor membership, specialization, and each skill's governing
  attribute come from imported content, including custom classes. Character
  creation grants establish the baseline and do not count as earned increases.
- Successful skill use accumulates fractional progress using the TES3 SKIL use
  values and applicable GMST factors. The advancement requirement is
  `(base skill + 1) * class factor * specialization factor`, where the class
  factor is `fMajorSkillBonus`, `fMinorSkillBonus`, or `fMiscSkillBonus`, and
  `fSpecialSkillBonus` applies only to the matching specialization. Verify
  threshold crossing, remainder handling, and use-event eligibility against
  TES3 behavior; do not grant progress merely for pressing an action button.
- Earned whole-point increases through use, purchased training, and skill books
  feed one progression operation. Training applies TES3 eligibility, price, and
  time rules and charges once. A skill book awards its increase only on its
  first qualifying reading. All 27 skills have their applicable use producers;
  unsupported producers must be listed as remaining work while status is Partial.
- Normal advancement stops at base skill 100. Fortify, drain, damage, restoration,
  equipment changes, and direct MWScript stat assignments do not masquerade as
  earned increases. Temporary modified values do not determine class membership
  or permanently rewrite the progression baseline. TES3 prison skill-loss and
  subsequent recovery must follow verified counter behavior rather than resetting
  the character to its starting class values.

### Level eligibility and attribute bonuses

- Each earned major or minor skill point adds one to character-level progress;
  miscellaneous increases do not. Vanilla eligibility is 10 combined increases,
  read from `iLevelUpTotal`. No combat XP, quest XP, or automatic character-level
  grant is introduced.
- Every earned skill point, including miscellaneous, training, and books, counts
  toward its governing attribute's bonus since the last confirmed level-up.
  Luck governs no skill and receives only +1 in the vanilla rules.
- Vanilla bonuses follow this table. Read the imported `iLevelUp01Mult` through
  `iLevelUp10Mult` settings rather than hardcoding vanilla multipliers; zero
  increases give +1 and counts above ten use the ten-increase entry.

  | Governing skill increases | Attribute gain |
  | --- | --- |
  | 0 | +1 |
  | 1–4 | +2 |
  | 5–7 | +3 |
  | 8–9 | +4 |
  | 10 or more | +5 |

- Reaching eligibility presents the TES3 rest notification and persistent ready
  indicator without immediately changing level. The stats UI shows actual
  progress, including overflow, and individual skill progress.
- Skill increases after eligibility continue contributing to the next confirmed
  level's attribute bonuses. Confirmation consumes one level's threshold and
  clears **all** attribute increase counters, including unselected attributes.
  Excess major/minor progress remains, but its consumed attribute bonuses are
  not carried into a second level-up. For example, 15/10 becomes 5/10.

### Rest and level-up UI

- A legal sleep/rest of at least one game hour opens level-up when eligible.
  Waiting does not substitute for sleeping. Prohibited or interrupted rest does
  not grant a level; bed ownership, nearby threats, and rest-location restrictions
  retain TES3 behavior. Verify the exact interruption and opening order locally.
- Display the new level, available attribute gains, and level description. The
  player chooses three distinct attributes; selecting one twice cannot spend two
  choices on it. If fewer than three base attributes are below 100, require only
  that many choices. With none available, confirmation must still be possible.
- Attributes at base 100 are unavailable; gains approaching the cap stop at 100.
  Modified attribute values do not make an otherwise eligible base attribute
  unavailable. Selecting or revising choices previews the result without mutating
  gameplay stats. Invalid or incomplete confirmation changes nothing.
- Valid confirmation applies choices and advances exactly one level atomically.
  Double input, reopening the panel, or loading a save cannot apply it twice.
  Banked eligibility remains available for a later qualifying rest, rather than
  automatically spending another level in the same confirmation.

### Health and integration

- Each confirmed level adds post-selection base Endurance times
  `fLevelUpHealthEndMult` to maximum health (vanilla 0.1). Retain fractional
  health; Endurance 45 yields 4.5, not 4. Gains are not retroactive. Temporary
  Fortify/Drain/Damage Endurance does not alter permanent health growth. Verify
  birthsign/ability treatment against retail TES3 rather than assuming all
  visible Endurance modifiers are equivalent.
- Current health adjustment and derived-stat recalculation follow TES3 behavior
  and are tested separately from ordinary rest restoration. Do not recalculate
  accumulated maximum health from the character-creation formula each level.
- Dialogue, scripts, stats UI, and gameplay read the same committed level and
  attributes. A pending level does not satisfy a higher-level dialogue gate.
- Save/load preserves fractional skill progress, base stats, character progress,
  attribute counters, book-read history, fractional health, and pending level-up
  state. Save during selection either restores the uncommitted choices or safely
  reopens selection from unchanged stats; it never consumes eligibility. Older
  saves initialize missing counters without inventing historical skill gains.
  Preserve cooked-scene and streamed-chunk layouts; progression belongs to the
  versioned gameplay save state.

## Implementation Boundaries

Extend the existing Bethesda session/runtime player state and TES3 native stat
access. Keep skill-use calculation, earned skill advancement, and confirmed
level-up as focused, deterministic operations with explicit advancement sources.
The application and UI submit gameplay intents and render session state; neither
keeps a second progression ledger. No new generic plugin or content system is
needed. OpenMW is a read-only behavioral reference, not an architecture template.

Implement in coherent stages: progression state and synthetic rules; real skill
use/training/books; rest and UI; persistence and normal-input retail verification.
An arithmetic-only implementation remains Partial.

## Verification

- Add hand-rolled CTest fixtures for major/minor/misc membership, specialization,
  fractional use thresholds, skill caps, all advancement sources, and excluded
  temporary/script changes. Exercise imported custom class/SKIL/GMST values.
- Test every multiplier boundary (0, 1, 4, 5, 7, 8, 9, 10, and above 10), Luck,
  mixed skills sharing an attribute, and mixed major/minor thresholds.
- Test 9/10, 10/10, 15/10, and at least 20/10; gains after eligibility; discarded
  unselected bonuses; and the next rest after banking multiple levels.
- Use headless UI intents for waiting versus sleeping, illegal/interrupted rest,
  selection/reselection, duplicates, incomplete confirmation, attributes at
  98–100, fewer than three available attributes, and all attributes capped.
- Test post-selection Endurance, fractional health across repeated levels,
  modifier/ability treatment, and derived/current stat behavior.
- Round-trip saves before eligibility, while eligible, during selection, after
  confirmation, and with banked levels. Test older saves and repeated input/load
  for duplicate training charges, book gains, and level commits.
- With local base `Morrowind.esm`, start through character generation, earn level
  3 through ordinary skill use and legal rest, and pass Caius's authored level
  gate. Also verify real training and a skill book, overflow bonuses, attribute
  caps, and an interrupted rest. No console stat grants, forced dialogue,
  journal overrides, or test-only skill awards may establish retail acceptance.
  Keep game data and capture evidence local; record reproducible actions and
  assertions without redistributing assets.
- Run relevant progression, TES3 runtime/script/content, save, dialogue, and
  headless UI targets, then the complete CTest suite before changing status in
  `docs/PARITY.md`. Review the final implementation diff.

## Definition of Done

All required behavior passes synthetic and normal-input local TES3 verification.
A persistent player-created character earns level 3, makes valid attribute
choices after resting, retains correct health and overflow across save/load, and
unlocks Caius's level gate through the shared committed player state. All skill
advancement sources and applicable skill-use producers are covered; passing only
the arithmetic tests or a debug-driven level gate is insufficient.

## References and Current Status

- Requested rules reference: [UESP — Morrowind:Level](https://en.uesp.net/wiki/Morrowind:Level).
  Direct retrieval returned HTTP 403 during specification drafting.
- Primary overview: [Morrowind GOTY PC manual](https://steamcdn-a.akamaihd.net/steam/apps/22320/manuals/mwgoty_pcmanual.pdf),
  character progression and leveling sections. Search-indexed excerpts confirm
  attribute bonuses and Endurance-based health growth.
- Local behavioral cross-check: `references/openmw/apps/openmw/mwmechanics/npcstats.cpp`
  for skill thresholds, counter consumption, GMST multipliers, and health;
  `references/openmw/apps/openmw/mwgui/levelupdialog.cpp` for selection and caps.
  These are reference observations, not proof of retail acceptance. Resolve
  retail edge cases during implementation without weakening the criteria.
- MECH-TES3-002's existing opening replay stops at Caius's level-3 prerequisite
  because normal player progression was absent when that replay was recorded.
  The implementation and verification below do not yet certify an ordinary-input
  continuation of that retail route.


## Implementation and Verification (2026-10-03)

The runtime imports numeric SKIL identities and reads governing attributes,
specializations, use values, class membership, and progression GMST overrides.
`Tes3ProgressionState` is part of the shared player state; base stats continue to
use the existing native/dialogue stat map. A custom-class definition is supported
by the progression backend. Character review initializes race/gender/class and
birthsign baselines without awarding earned counters, including ability attribute
and maximum-magicka bonuses and starting spells.

The shared advancement operation handles use, training, skill books, and TES3
jail loss/recovery accounting. Use progress is normalized; successful use-earned
advancement discards excess progress, while training/books retain the existing
fraction. Skill and attribute caps, the complete vanilla multiplier table, GMST
overrides, overflow level progress, and counter reset on confirmation are covered.
Direct native stat changes remain separate from earned advancement. `GetLevel`
now reads the committed player level rather than the imported NPC template.

Actual running relative to physical support advances Athletics; a requested and
successful physical takeoff advances Acrobatics once. Resolved melee contact
advances the equipped melee weapon skill or Hand-to-hand, and a successful player
guard advances Block. Walking, stationary run intent, airborne camera movement,
attack requests, and missed contacts do not award those uses. This does not
certify TES3 combat hit-chance or armor-location mechanics.

Owned skill-book reading awards once per case-insensitive book identity, with
read history persisted. Trainers offer their three highest available imported
skills, enforce player/trainer/attribute and gold limits, apply the implemented
TES3 barter terms, transfer gold once, and advance two game hours. Full derived
disposition, automatically calculated trainer stats, and all service modifiers
still require retail parity verification.

The retained UI exposes `T` for rest, `F1` for level and skill progress, and `R`
for training during an eligible conversation. Rest offers sleep/wait and 1–24
hours. Session checks enforce the available cell no-sleep flag, physical support,
water height, authored bed identity/ownership, and nearby active combat targets.
Waiting cannot open level selection. Legal sleep opens a preview-only selection
of distinct eligible attributes; confirmation atomically grants one level and
post-selection base-Endurance health, preserves fractional health and overflow,
and resets all attribute counters. Fewer than three available attributes and
fully capped attributes are handled. Loading safely restarts the selection
preview without consuming eligibility.

Gameplay saves carry a versioned progression section, normalized skill progress,
attribute counters, banked levels, ready/selection state, bed identity, read-book
history, and custom-class data. Older saves preserve existing stats and initialize
missing progression to zero. Malformed progression is rejected without mutating
the session. The deterministic state hash includes the ledger. Cooked-scene and
streamed-chunk formats are unchanged.

Verification completed:

- `cmake --preset linux-vcpkg` and the full Debug build passed.
- All **65 registered CTest targets passed**, including the new
  `odai_tes3_progression_tests` and retained runtime/script/content/save/UI tests.
- Synthetic verification covers all 27 skill record indices, class and
  specialization thresholds, use overflow, training/book progress retention,
  multiplier boundaries, 9/10 and 10/10 readiness, 15/10 and 20/10 banking,
  attribute caps, duplicate/incomplete/repeated confirmation, base-Endurance
  fractional health despite temporary fortification, jail loss/recovery,
  actual movement and melee outcomes, hostile rest rejection, headless selection
  intents, pending-selection save/load, older saves, and malformed saves.
- The optional local check
  `build-linux/odai_tes3_progression_tests --retail "<Morrowind Data Files>"`
  passes for all 27 retail SKIL definitions, imported classes, and a Breton male
  Warrior with Lady's Favor. It checks the initial ability-adjusted attributes,
  health and magicka, and empty earned counters. This is a record/initialization
  check, not a normal-input character-generation or leveling playthrough.

Remaining acceptance work:

- Complete the remaining use producers: Armorer's imported damage/combat-durability
  path; Enchant; all six magic schools;
  Alchemy; Security; Sneak; Marksman;
  Mercantile. Speechcraft persuasion is wired below but still needs authored
  AI and retail parity verification. Authored responses have synthetic coverage
  below. Athletics swimming is connected below and still needs retail parity
  verification. The shared calculation supports their imported use values, but an API
  call or synthetic award is not proof of a real gameplay producer.
- Verify ordinary-input custom-class creation locally, complete prison service integration,
  trainer/disposition parity, and the rest pipeline's authored sleep events,
  random/scripted interruptions, and ownership cases. Retail verification of
  timed-effect expiration during rest remains required.
  Live temporary modifiers on derived magicka/fatigue need full parity checks.
- Verify the native-DPI UI visually, including level description presentation,
  and complete the specified rest/selection/cap scenarios through normal input.
- Complete character generation, earn level 3, and pass Caius's authored level
  gate through the retail gameplay path without stat or quest overrides. Extend
  the existing main-quest replay only after that evidence exists.

The Definition of Done remains unchanged and has not passed. Status is Partial.

### Rest-effect follow-up (2026-10-03)

Rest now shortens active TES3 spell durations by the equivalent real duration
of each game hour using the imported `timescale` global (30 when absent, one
when zero). It expires individual effects and removes empty spells/actor entries
without advancing physics ticks, preserves permanent effects, and applies to
other actors as well as the player. Sleep restores magicka only for the portion
of the hour remaining after temporary Stunted Magicka expires. Waiting ages
effects without restoring magicka; illegal rest leaves effect state unchanged.
The duration conversion and Stunted Magicka treatment were cross-checked against
the read-only OpenMW `Actors::rest` and `Actors::restoreDynamicStats` reference.
This is synthetic/reference evidence, not retail acceptance.

Progression regressions cover expiration boundaries, multi-hour remaining
duration, permanent effects, zero time scale, rejected rest, and partial-hour
magicka restoration. The full Debug build and all 65 CTest targets pass.

### Custom-class input follow-up (2026-10-03)

The retained class menu now offers Create custom class. A shared semantic input
flow previews specialization, two distinct favored attributes, five major skills,
and five minor skills; chosen skills are excluded from subsequent choices in
both groups. Back revises previous choices or returns to the imported-class
list. Final review commits the class definition; the existing character-review
operation then initializes its baseline without earned counters. Selecting an
imported class clears a previous custom definition. Loading restarts an unfinished
custom-class preview without changing the committed class or baseline.

Progression tests exercise the same input flow, including distinct membership,
preview isolation, revision, confirmation, repeated confirmation, custom starting
stats, earned major/minor versus miscellaneous counters, and save/load. The
application builds with this menu connected. Native-DPI visual verification and
an ordinary-input retail character-generation run remain required.
The full Debug build, all 65 CTest targets, and navigation-map validation pass.
Character-menu Back also consumes Escape without opening the pause menu.

### Physical fall-use follow-up (2026-10-03)

The character controller accumulates descending distance in Bethesda units and
delivers it with a physical landing. TES3 player landings now calculate health
damage using imported `fFallDamageDistanceMin`, `fFallDistanceBase`,
`fFallDistanceMult`, `fFallAcroBase`, and `fFallAcroMult`, modified Acrobatics,
Jump magnitude, and the fatigue term. A damaging landing at or below the
Acrobatics/fatigue knockdown threshold awards SKIL use index 1 exactly once.
Harmless landings, landings above that threshold, water landings, and active
Slow Fall/Levitate protection do not award that use. Severe landings request
stagger presentation; full TES3 knockdown and magical movement behavior still
need parity work and retail verification.

Measured in-progress fall distance is part of physical gameplay snapshots,
save/load, and deterministic hashing. Older saves initialize a missing value
to zero; malformed negative values fail before session mutation. Recovery onto
solid ground clears the accumulated distance. Cooked/streamed layouts are
unchanged.

Progression regressions drive actual physics for harmless, damaging, severe,
Slow Fall, Jump, and underwater landings. They verify damage arithmetic, the
imported fall-use value, no duplicate award on subsequent grounded ticks,
mid-fall save/load outcomes, older saves, and malformed saves. The rules were
cross-checked against read-only OpenMW fall handling; the vanilla numeric
fallbacks were verified from the local base Morrowind.esm records. This does not
establish ordinary-input retail acceptance. The full Debug build, all 65 CTest
targets, and navigation validation pass. Capability status remains Partial.

### Defensive contact-use follow-up (2026-10-03)

Resolved incoming melee contacts with positive implemented health damage now
advance one of Light Armor, Medium Armor, Heavy Armor, or Unarmored. Each contact
selects one body slot with the TES3 distribution: 30% cuirass, 10% each helmet,
greaves, boots, pauldrons and shield, and 5% each hand. An empty selected slot
counts as Unarmored. The deterministic roll uses the contact's attacker, tick,
and existing hit sequence; no second progression ledger was introduced.

ARMO classification imports item weight and the corresponding reference-weight
GMST, `fLightMaxMod`, and `fMedMaxMod`, including the TES3 boundary tolerance.
Bracers occupy the same slots as their corresponding gauntlets. Loading migrates
the earlier separate bracer slot masks while preserving items and policy locks.
Malformed armor definitions do not invent defensive skill progress.

Tests drive missed and successful physics-backed contacts for all four skill
categories, isolate the category awarded by each hit, verify no later duplicate
award, and round-trip the resulting progress. Additional fixtures cover all armor
types, slot distribution, classification boundaries, imported GMST overrides,
bracer replacement, and legacy bracer saves. The full Debug build, all 65 CTest
targets, and navigation validation pass.

This verifies these producers on the implemented melee contact path. Full TES3
hit-chance, armor rating/durability, health-versus-fatigue damage, ranged contacts,
and ordinary-input retail defensive-use acceptance remain unverified. The
Definition of Done has not passed; status remains Partial.

### Weapon identity and hit-eligibility follow-up (2026-10-03)

WEAP numeric types now map correctly to Short Blade, Long Blade, Blunt Weapon,
Spear, Axe, or Marksman. The earlier range-based mapping awarded incorrect skills
for several types, including one-handed long blades. Ammunition is rejected as
a weapon skill source, and equipped bows/crossbows/thrown weapons cannot award
Marksman from a melee request.

TES3 geometric contact now passes a stat-based hit-chance check before damage,
Block/armor use, or weapon skill use. It uses modified weapon skill, Agility,
Luck, both actors' fatigue terms, Fortify Attack, Blind, Sanctuary, paralysis,
Invisibility/Chameleon, and imported `fCombatInvisoMult`, rounding the resulting
percentage. Deterministic rolls use existing saved session state and attack
sequence. Creature attributes and combat/magic/stealth skills are now imported;
creature hit chance uses the authored combat skill.

Regression fixtures cover every weapon type, successful contacts for each melee
category, ranged equipment exclusion, fatigue and defensive magic, failed rolls
with unchanged damage/progression, and save/load hit-outcome replay. Incoming
failed rolls also leave all four defensive skill progress values unchanged.
The full Debug build, all 65 CTest targets, and navigation validation pass.

This is synthetic/reference verification of the implemented stat-based path.
Awareness/sneak and knockdown state, complete combat damage
and equipment condition, and retail behavior still require work. Remaining skill
producers and the normal-input level-3/Caius acceptance run are still required.
Capability status remains Partial; the Definition of Done has not passed.

### Passive trainer and gameplay stat modifiers

NPC/creature authored spell grants and NPC racial spell grants are imported as
owned spells, with duplicate grants collapsed. Modified session skill/attribute
values now include owned abilities, diseases and curses as well as unexpired
active effects. Training limits/prices and melee hit chance use this shared
calculation. An owned passive with an active representation contributes once;
the player's creation ability attributes remain part of the established base,
while later timed attributes remain modifiers. No earned progression counters
or authored base skills change when these effects appear or expire.

Fixtures cover racial/authored duplicate grants, modified trainer eligibility
and prices, timed Drain Skill expiry, passive/active overlap, player creation
attributes versus timed fortification, and gameplay save/load. The full Debug
build, all 65 CTest targets, navigation validation, and local retail record and
initialization checks pass. These checks do not certify normal-input retail
training. Derived disposition, NPC generated spells and dynamic-stat modifiers,
remaining skill-use producers, rest/service parity, and normal-input level-3/Caius
acceptance remain outstanding. Status remains Partial; the Definition of Done
has not passed.

### Automatically calculated trainer baselines

Imported automatic NPCs now derive their base attributes, all 27 skills, health,
magicka, and fatigue from race, gender, class, level, SKIL records, and the NPC
magicka setting. Skill and attribute growth use TES3 ties-to-even rounding and
the base-100 cap, including level-zero records. Trainers therefore rank their
offers and enforce skill limits against calculated stats; runtime stat overrides
still apply over that baseline. Missing class/race/skill records leave the actor
unchanged rather than installing a partial calculation.

Fixtures cover gender, major/minor/misc and specialization growth, racial bonuses,
halfway rounding, levels zero and 200, derived stats, an actual training
transaction, and a runtime trainer override. The local retail record probe checks
complete attribute and skill maps on every imported automatic NPC. This verifies
record initialization, not normal-input retail training. NPC ability modifiers,
generated spells, full disposition/service parity, remaining skill-use producers,
and the level-3/Caius acceptance run remain outstanding.
The full Debug build, all 65 CTest targets, and navigation validation pass.
Capability status remains Partial; the Definition of Done has not passed.

### Shared derived disposition for training and dialogue

Trainer prices, session-backed `GetDisposition`, and authored dialogue filters
now use one derived disposition calculation. It applies base disposition,
shared race, modified Personality, faction/self reactions and associated rank,
expulsion, script reaction changes, bounty, common/blight disease, drawn weapons,
and Charm using imported disposition GMSTs. Float intermediates, truncation and
the 0–100 clamp retain TES3 arithmetic. Synthetic caller-supplied dialogue actors
whose IDs differ from their imported reference keep their explicit fixture state.

`SetDisposition`/`ModDisposition` change the stored base, while native reads do
not create or overwrite it. Tests cover positive/negative/self faction reactions,
expulsion and script changes, disease, weapons, Charm expiration, native/dialogue
agreement, training price changes, base-versus-derived script operations, GMST
overrides, clamping and save/load. Crime-witness penalties, persuasion's temporary
changes, the remaining service/rest and skill-use paths, and normal-input retail
acceptance still need work. Capability status remains Partial.
The full Debug build, all 65 CTest targets, navigation validation, and local
retail record/initialization checks pass. Retail normal-input acceptance has
not passed; the Definition of Done remains unfulfilled.

### Speechcraft persuasion gameplay path

The conversation's `H` menu submits Admire, Intimidate, Taunt and the three
bribe amounts to the session. Outcomes use modified Speechcraft/Mercantile,
Personality, Luck, reputation, level, fatigue, current derived disposition,
imported persuasion GMSTs and the saved session random stream. Accepted attempts
award SKIL success or failure use index; capped Speechcraft still permits the
gameplay action. Unavailable conversations, authored choices, level selection,
unaffordable bribes and invalid data reject before consuming a roll or award.
Successful bribes transfer gold once to the NPC; failed bribes transfer none.
Intimidate/Taunt modify imported Fight/Flee settings and persist their overrides.

Temporary and permanent disposition changes are separate conversation state.
Ending/replacing a conversation or an authored Goodbye commits the permanent
portion once, drops the temporary portion, and excludes Charm from the permanent
clamp. The vanilla marginal Intimidate case remains distinct from OpenMW/MCP's
fix. Save/load preserves active changes and random state; older saves initialize
missing changes to zero, and malformed counters reject without mutation.

These tests also exposed and fixed restored dialogue's incomplete player snapshot:
active dialogue now derives from the saved committed character, and closing it
does not overwrite race/class or the progression ledger. Fixtures cover all six
actions, failed use, gold transfers, class/attribute counters at a threshold,
caps, modified skills, amplified temporary GMSTs, chance equality, marginal
intimidation, pending saves, older/malformed saves and repeated closing. The full
Debug build, 65 CTest targets, navigation validation and local retail record and
initialization checks pass.

Authored persuasion response text/results now have synthetic coverage below.
Full Taunt/Intimidate AI reactions, NPC reputation parity and ordinary-input
retail persuasion remain unverified.
The remaining skill-use/rest/service paths and normal-input level-3/Caius
acceptance are still required. Status remains Partial; the Definition of Done
has not passed.


### Authored persuasion replies

Accepted persuasion attempts select the imported Persuasion DIAL for their
success/failure outcome and use the existing INFO filters and result-script
transaction path. Replies do not need to be known topics, and they remain
unavailable through ordinary topic selection. The conversation UI displays the
selected response and its choices; authored Goodbye closes it normally.

Gameplay outcomes, gold transfer and Speechcraft use resolve before the reply.
A MessageBox suspends only the authored result, blocks further attempts, and
continues once without rerolling or awarding another use. A failing result rolls
back its script effects while preserving the already resolved attempt. Closing
a suspended reply cancels its script effects and commits the prior permanent
persuasion change once.

Synthetic fixtures cover all six successful action replies, failed Admire and
Bribe replies, filtered result selection, MessageBox continuation/repeated input,
script failure/recovery, suspended cancellation, Goodbye and topic isolation.
The full Debug build, all 65 CTest targets, navigation validation, and local
retail record/initialization check pass. This does not establish normal-input
retail persuasion or leveling acceptance. The remaining producers, AI/service/
rest parity and ordinary-input level-3/Caius run remain required; status is
Partial and the Definition of Done has not passed.


### Faction NPC reputation initialization

Faction NPC reputation now initializes as
`iAutoRepFacMod * (rank + 1) + iAutoRepLevMod * (level - 1)` for both explicit
and automatic NPC records. Unaffiliated NPCs retain authored reputation, and
creatures do not use the NPC rule. The local base Morrowind.esm supplies factors
2 and 0; these are also the absent-setting fallbacks. Signed rank/level arithmetic
is retained, while invalid or overflowing imported factors leave the authored
value intact. This follows the read-only OpenMW NPC initialization reference.

Fixtures cover explicit/automatic faction members, imported factor overrides,
unaffiliated NPCs, rank -1 and level 0, native Get/ModReputation, persisted
instance overrides, and deterministic persuasion outcomes with baseline versus
modified NPC reputation. Existing content/probe assertions now check initialized
faction reputation while their raw authored record values remain unchanged.
The local retail initialization probe checks every faction NPC against imported
factors. The full Debug build, all 65 CTest targets and navigation validation
pass. This is initialization/reference evidence, not ordinary-input retail
persuasion acceptance. Remaining skill producers, AI/service/rest parity and
the normal-input level-3/Caius verification still prevent Definition of Done;
capability status remains Partial.


### Swimming Athletics producer

The player character controller now derives swimming from the imported cell's
water height (exterior default zero) and `fSwimHeightScale` times physical actor
height. The local retail setting is 0.9. While submerged, controller movement
retains collision resolution, permits vertical swimming, suppresses gravity and
jump launches, clears fall distance, and limits upward motion at the swim surface.
A small surface tolerance prevents floating-point drift from toggling gravity.
The application submits pitch-directed swim movement, Space ascent and Ctrl
descent through the existing player controller.

Resolved swimming movement awards Athletics SKIL use index 1, scaled by elapsed
simulation seconds, independently of the run key. It excludes stationary and
blocked input, run use, and Acrobatics jump use. Water state is derived again
from saved cell/physical state rather than introducing another saved ledger.
Synthetic physical tests cover idle flotation, actual movement and imported use
value, vertical ascent/surface limits, blocked movement, leaving a water cell,
and save/load. Incoming armor-contact fixtures now have a supporting floor,
so they do not fall indefinitely through exterior water during their cooldowns.

The full Debug build, all 65 CTest targets, navigation validation and local
retail initialization check pass. Ordinary-input retail swimming, full TES3
swimming-speed/fatigue rules, water walking/levitation, rendered actor-height
parity and shoreline transitions still need verification or implementation.
Remaining skill producers and rest/service parity, together with the retail
level-3/Caius acceptance run, keep this capability Partial. Its Definition of
Done has not passed.


### Item-condition prerequisite for Armorer

Inventory stacks and dropped world items now retain an optional condition/use
count, with -1 representing the imported full value. Different conditions of the
same base record stay separate. Additions, aggregate removals, mixed-stack
transfers, selected-condition drops and ordinary TES3 pickups preserve the state
and untouched copies. Inventory drop input submits the selected stack's condition.
NPC native item counts sum all stacks, matching the existing player count map.
Aggregate additions/transfers reject integer overflow before mutation.

Gameplay saves preserve inventory and world conditions; older saves initialize
missing values to -1 without inventing damage. Negative and oversized condition
values reject before applying the save. Deterministic hashing includes condition,
TES3 inventory identity and canonical stack ordering. A dropped item picked up
through ordinary activation exposed a prior save resolver restriction: enabled
overrides for runtime-created items now resolve only when the saved object and
its imported TES3 base record exist. Unresolved reference overrides still reject.
Cooked-scene and streamed-chunk formats are unchanged.

Fixtures cover mixed-condition stacks, aggregate transfer/removal, selected drop,
actual activation pickup, save/load, older saves, malformed conditions and
aggregate count overflow. The full Debug build, all 65 CTest targets, navigation
validation and local retail record/initialization check pass. This establishes
the persistent state required for repair; it does not award Armorer or certify
a repair producer. Repair attempts, tool-use consumption, repair UI, authored
OnPCRepair handling, combat durability and retail repair verification remain
required, alongside the other outstanding skill/rest/service and level-3/Caius
acceptance work. Capability status remains Partial; Definition of Done has
not passed.


### Repair attempts and Armorer progression

The retained F2 Repair menu selects an owned REPA tool and damaged weapon or
armor. Its available rows show imported maximum condition/tool uses and current
instance state. The session validates eligibility before consuming a saved random
roll, then applies the modified Armorer/Strength/Luck and fatigue success rule.
Repair amount uses imported REPA quality and `fRepairAmountMult` (verified local
retail value 3), truncation and the one-point minimum, capped by missing condition.

Each accepted attempt consumes one use from one tool, splitting stacked tools as
needed; an exhausted tool is removed. Success repairs one item copy, restacks
identical fully repaired copies, and awards Armorer SKIL use index 0 through the
shared progression operation. Failure consumes the tool use but awards no skill
progress. Base skill 100 still permits repair without advancement. Ordinary
repair use breaks active invisibility. Successful attempts deliver OnPCRepair
through the existing owned-item script path. Invalid/unavailable selections reject
before a roll, condition change, or skill award.

Fixtures exercise success and failure, chance equality, imported tool quality,
stack splitting, full-condition cap/restacking, exhaustion, repeated invalid
input, skill cap, a real repair crossing a minor-skill/attribute threshold,
invisibility, authored repair notification, blocked level selection and save/load
of condition, remaining tools, progress, script state and random state. The full
Debug build, all 65 CTest targets, navigation validation and local retail
record/initialization check pass.

This is a connected repair action and UI, with synthetic outcome verification.
Imported reference condition and combat durability must still supply damaged
items during ordinary retail play; repair-menu visual/input acceptance and retail
repair verification remain outstanding. Owned-item scripts also retain the
existing player-owned execution context rather than full per-item instance
script identity. Other skill producers and rest/service parity, plus the
normal-input level-3/Caius acceptance run, still prevent Definition of Done.
Capability status remains Partial.
