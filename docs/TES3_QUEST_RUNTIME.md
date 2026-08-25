# TES3 quest runtime

ODAI has a clean-room TES3 gameplay path for Morrowind content. OpenMW's public
ESM3 and dialogue behavior are compatibility references; ODAI does not copy or
link OpenMW implementation code.

The runtime currently provides:

- tagged, case-insensitive TES3 record identities and stable FRMR identities;
- immutable later-wins content merging for named records, DIAL/INFO ordering,
  SCPT, GLOB, actors, cells, and placed references;
- dynamic greeting/topic selection, SCVR filters, known topics, authored
  choices, synchronous result scripts, and Tribunal quest status flags;
- a deterministic MWScript source frontend and VM with saved instruction
  cursors, globals, locals, event variables, modal choices, and fixed-tick
  scheduling;
- chronological/active/completed journal presentation without invented
  Skyrim-style objectives, plus a latest-entry pinned quest tracker model;
- sparse unloaded-reference overrides and ODAI save-v8 persistence;
- typed SPEL/ENAM definitions plus deterministic, saved active-spell state for
  the Fortify Attribute/Skill effects used by the Bloodstone pilgrimage; and
- profile-driven script, dialogue, journal, structural quest-suite, and
  deterministic virtual-player probes.

Run the installed-content compiler gate with an OpenMW profile:

```bash
./build-linux/odai_bethesda_probe \
  --tes3-scriptcheck /path/to/openmw.cfg --strict
```

The strict command exits unsuccessfully while any source/bytecode block is
unhandled or any discovered gameplay native or resolved `Cast` effect is
unsupported. The quest-suite
probe likewise reports `release_gate_passed: false` until the authored,
event-driven transition explorer covers every branch; structural journal
enumeration is not treated as quest compatibility.

```bash
./build-linux/odai_bethesda_probe \
  --tes3-dialogue-trace /path/to/openmw.cfg <actor-or-topic>
./build-linux/odai_bethesda_probe \
  --tes3-quest-trace /path/to/openmw.cfg <journal-id>
./build-linux/odai_bethesda_probe \
  --tes3-quest-suite /path/to/openmw.cfg --quest <journal-id>
```

The virtual player walks authored load-door edges from an exterior cell into
every Fighters Guild and back out, including chained interiors such as Vivec's
Foreign Quarter Plaza. It also exercises every placed caravaner destination as
a silt-strider trip and every placed guild-guide destination as a teleport.
It never fabricates a transition: malformed cells, locks, missing doors,
unplaced service actors, and unresolved arrival cells are emitted in `gaps`.

```bash
./build-linux/odai_bethesda_probe \
  --tes3-virtual-player /path/to/openmw.cfg \
  --strict --report /tmp/tes3-virtual-player.json
```

Normal mode is diagnostic and writes a report even when gaps exist. `--strict`
makes any unreachable guild, unresolved required route, cell extraction error,
or missing strider/guild-guide network fail the command for continuous testing.

OpenMW Lua, MWSE, native extenders, and dynamic engine plugins remain disabled.
The repository policy may admit only sandboxed OpenMW Lua v129 after the TR and
original-campaign MWScript gates are complete; that later phase is not a
general-purpose Lua plugin system.
