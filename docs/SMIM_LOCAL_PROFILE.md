# Local SMIM showcase profile

SMIM SE 2.08 from the user's Downloads archive is installed locally using the
FOMOD's `Skyrim 2016 Special Edition: Everything` mapping, in declared order.
The selected payload contains 1,110 NIFs, 315 DDS textures, the merged SE ESP and
one inert Windows shortcut. No installer executable was run. Textures retain their
full authored size; their formats are 170 BC1 and 145 BC3, not BC6H.

Local profile: `captures/smim-showcase/profile.json`. It retains the discovered
Skyrim.ccc active content and adds `SMIM-SE-Merged-All.esp` last (11 plugins total),
with 22 base/DLC archives explicitly enabled. The SMIM loose asset layer overrides
base archives. The installed game Data directory is unchanged. Profile fingerprints
separate generated import caches from the previous unmodded configuration.

Launch from the repository root:

```sh
python scripts/capture_riverwood.py --profile small-fast \
  --content-profile captures/smim-showcase/profile.json \
  --view street --look day --play --name smim-showcase/play
```

The small window remains 768×432 logical points and renders at native display DPI
(1536×864 on the current display). `--profile` selects graphics settings;
`--content-profile` selects assets and active plugins. It does not change the global
installation or automatically enable SMIM in unrelated launches.

Local capture: `captures/smim-showcase/riverwood-street.{png,json,log}`. The final
600-frame capture completed with Vulkan synchronization validation enabled, zero
validation errors and a clean renderer image shutdown. Known unsupported ambient
particle/marker warnings remain. This is not exhaustive compatibility or performance
certification of SMIM. Mod assets, the installer selection manifest and captures
are Git-ignored and stay local.

The probe's `--why` winner annotation now uses the runtime asset resolver rather
than provider enumeration order, which incorrectly put listed base archives above
loose mod assets. The capture script accepts an explicit content profile and reads
non-UTF-8 mod log text without aborting a successful capture.
