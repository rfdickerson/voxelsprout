# MECH-TES3-002: Main Quest Dialogue and Journal Conditions

Status: Partial

## Goal

The TES3 runtime selects main-quest dialogue and advances journal entries from the imported game's conditions and result scripts, so the standard MECH-002 route can progress through conversation.

## Dependencies

- TES3 DIAL/INFO import, dialogue UI, journal state, and MWScript result execution
- Player and actor state, inventory, factions, and save/load
- HARNESS-007 probe artifact/record assertions and scoped dialogue replays

## Required Behavior

- Dialogue filters used by the standard main quest evaluate their authored actor, player, cell, faction, disposition, global, local, inventory, death-count, and journal inputs. An unavailable input does not silently make a guarded response eligible.
- Topics, choices, greetings, and result scripts are presented and executed in the order allowed by imported content. A conversation observes changes made by an earlier response, including the item-state behavior already added for MECH-002.
- Journal entries are recorded once when their authored transition occurs. The journal shows current and historical progress, including the informants, lost prophecies, and seven House and Ashlander recognitions. Independent recognitions can progress in any allowed order.
- Dialogue and journal state survives cell transitions and save/load. Reopening a conversation does not duplicate one-time quest rewards or advance a quest without its conditions.

## Verification

- Add synthetic dialogue fixtures covering each condition type needed by the route, both eligible and ineligible responses, choice/result ordering, independent recognition progress, and save/load.
- Trace the local `Morrowind.esm` route at representative Caius, Urshilaku, Great House, Ashlander, Archcanon, and Vivec conversations. Assert that the expected INFO and journal index are reached by normal interactions.
- Run the relevant CTest targets and existing suite before changing status in `docs/PARITY.md`.

## Definition of Done

Every dialogue and journal gate on MECH-002's standard route is reachable through its authored conditions, with no permissive fallback or externally set quest stage required.


## Implementation and verification (2026-10-03)

The runtime rejects malformed and unavailable conditions even through the legacy
`strict=false` API. It evaluates minimum actor/player ranks, disposition, cell
prefixes, globals, declared speaker locals, journal indices, inventory, authored
actor death counts, negative identity filters, and choices. `NotLocal` negates
its authored comparison; a known absent declaration is distinct from an
unresolved speaker/script. Named player stat, faction, reputation, health,
disease, and conversation inputs now feed supported native numeric filters.
Unimplemented numeric inputs continue to reject responses.

Result scripts inherit the speaker's declared locals and commit their changes
into reference state and resident local-script threads. Authored conditions
control repeat eligibility; displaying an INFO no longer suppresses it for the
rest of the conversation. Outstanding choices cannot be bypassed by selecting
another topic, and a failed choice preserves the prompt. Repeated journal
entries do not change chronology or reopen a completed quest. World deaths feed
persistent, per-reference death counts once per death.

Synthetic binary DIAL/INFO fixtures in `odai_tes3_runtime_tests` exercise both
eligible and ineligible filters, unavailable inputs in both API modes, choices,
result ordering, local rewards within and across conversations, seven independent
recognitions in mixed order, duplicate entries, cell transitions, and save/load.
The full Debug build and all 63 registered CTest targets passed.

`odai_bethesda_probe --tes3-dialogue-trace` supports `--actor` to disambiguate
actor IDs from topic names, ordered `--topic` / `--choice` interactions,
`--expect-info <id>`, and `--expect-journal <id> <index>` assertions. Actor probes
use authored placement and actor data with session-backed result execution.
There is no stage, inventory, global, or condition override option.

Local base `Morrowind.esm` checks passed for the following fresh-state responses.
Journal assertions remained at zero; these checks establish initial eligibility
and absence of premature quest advancement, not completion of the later gates.
Game data and probe output remain local under `/tmp/mech-tes3-002-traces`.

| Speaker | Interaction | Expected INFO |
| --- | --- | --- |
| Caius Cosades | Greeting, then Report to Caius Cosades | 15425327421904220202 |
| Sul-Matuul | Greeting | 248414305611922759 |
| Athyn Sarethi | Greeting, then House Redoran | 26999160593007611872 |
| Kaushad | Greeting | 3129422684258721217 |
| Tholer Saryoni | Greeting | 78232102502120445 |
| Vivec (`vivec_god`) | Greeting | 166552843338625147 |

### Follow-up verification (2026-10-03)

Imported FACT rank requirements and reactions now feed dialogue filters.
Equipped CLOT/ARMO values feed clothing gates. Player stat filters include
active fortification, drain, and damage; health percentages use TES3 integer
truncation. Actor health/reputation and missing player faction rank differences
are evaluated separately from player values. Qualified disposition/reputation
calls resolve their actual authored actor, including unloaded references.
Completed dialogue choices clear the choice filter, and TalkedToPc snapshots
prior per-reference conversation history and persists across saves.

Integer GLOB conversion now follows TES3's float-to-long/short rules. This fixes
retail short globals containing denormal or NaN float representations, which
previously produced invalid saves. Non-finite filter operands remain ineligible.

`tests/fixtures/harness/tes3_mainquest_dialogue_opening.json` records 35 authored
headless interaction intents and expected INFO/journal checkpoints. The local
base-game replay passes Census release, delivery to Caius, Hasphat's puzzle-box
hand-in, Sharn's skull hand-in, their informant reports, and save/reload. No
inventory grant, journal/global override, or forced INFO selection is supplied.
The probe places the player at authored interaction locations; this evidence
does not certify navigation or UI input. All 64 registered CTest targets pass.

The replay reaches Caius's normal level-3 prerequisite: INFO
`7418104811174331576` retains `A1_4_MuzgobInformant=25` rather than offering the
Vivec informant assignment. Normal skill growth/level-up is not implemented in
the runtime, so that prerequisite cannot currently be earned by this replay.
The level gate is preserved.

Remaining acceptance work: provide normal player progression, then earn the
remaining prerequisites and assert every retail main-quest gate, including the
Vivec informants, lost prophecies, all seven recognitions, Archcanon, and Vivec.
The opening replay and synthetic recognitions do not satisfy the complete
route verification or the Definition of Done. Status remains Partial.
