# Skyrim mod checking

`odai mod-check` checks a Skyrim Special Edition content profile without opening
windows or initializing Vulkan. It reads source plugins and profiles without
modifying them. It supports the same ODAI JSON and MO2 profile inputs as the runtime.

```sh
odai mod-check --profile /path/to/profile.json
odai mod-check --profile /path/to/MO2/profiles/MyProfile --data /path/to/Data --mods-root /path/to/MO2/mods --json
odai mod-check --profile /path/to/profile.json --export-profile /path/to/new-profile.json
```

Tools-only builds expose the same command as
`odai_bethesda_probe --mod-check --profile /path/to/profile.json`.
Use `--help` after either subcommand for options.

The checker validates headers, master dependencies, cycles, container layouts,
plugin slot limits, record/group bounds, compressed data and subrecord structure.
It also checks duplicate record identities within a plugin, undeclared local
master indices, and object IDs outside the light namespace. Deleted references
and navmeshes are warnings requiring review. Normal overrides are counted using
the importer’s existing provenance index; they are not automatically errors.
A checksum exception retained for retail compatibility is reported as a warning.

The report includes the requested and suggested orders, inserted masters and
moved plugins. Dependency sorting is deterministic: among plugins whose masters
are ready, explicit plugins retain their requested priority; implicit masters
use their discovery order after ready explicit plugins. This is dependency-only
sorting, without LOOT community metadata or inferred conflict-resolution rules.
Plugin search roots retain the existing profile precedence (later enabled layers
win). Runtime startup shares header/dependency validation and rejects invalid
Skyrim orders; it does not perform the full record scan or automatically apply
the checker’s suggested order.

`--export-profile` writes the suggested order into a separate ODAI profile only
when checking succeeds. The destination must not exist, and its parent directory
must exist. The export preserves asset layers and archives and is reloaded and
validated before publication. Source plugins, source profiles and existing
exports are never overwritten.

Exit statuses are `0` for valid content (including warnings), `1` for validation
failure, and `2` for usage or output failure. JSON reports contain `version`, `ok`,
`requested_order`, `resolved_order`, `changes`, `override_count`, and `diagnostics`.
Diagnostics carry `severity`, `code`, `message`, `source`, and, when known,
`form_id` (the plugin-local integer ID) and `file_offset` (byte offset).
Graph validation stops at the first blocking error; full scans report findings
across plugins, stopping a malformed plugin at its first structural error.

This milestone does not repair files, remove identical-to-master records, inspect
arbitrary reference fields, or replace LOOT/xEdit. Container checks cannot identify
every wrong-game plugin sharing the same TES4 layout. Use LOOT for metadata-based
ordering and xEdit for contextual inspection and cleaning.
