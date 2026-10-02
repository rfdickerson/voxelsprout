# HARNESS-004: Visual regression testing

Status: Implemented

## Goal

Compare runtime PPM screenshots with reviewed baselines in a repeatable,
machine-readable way, with a clear failing exit status and inspectable diff.

## Contract

`scripts/visual_regression.py` reads a version 1 JSON manifest containing
`name`, `baseline`, `channel_tolerance` (0–255), and
`max_changed_fraction` (0–1). The baseline path is relative to the manifest.
Pass `--actual capture.ppm` to compare an existing screenshot, or set a
`command` string array containing `{output}` and pass `--output capture.ppm`
to run a deterministic `odai --screenshot` command before comparing.
`--diff diff.ppm` writes a red channel image of maximum channel delta per
pixel. Only pixel deltas above `channel_tolerance` count as changed. The
changed pixel fraction must not exceed `max_changed_fraction`.

The comparator requires binary P6 PPM, 8 bit channels, and matching dimensions.
It emits one JSON report with dimensions, changed pixel count and fraction,
largest channel delta, and verdict. Exit codes are 0 for pass, 1 for a visual
regression, and 2 for invalid input or capture failure. It does not silently
resize images. Review and commit baseline changes deliberately; retail game
assets and local captures stay outside the repository.

Example manifest:

```json
{
  "version": 1,
  "name": "example-view",
  "baseline": "baseline.ppm",
  "channel_tolerance": 2,
  "max_changed_fraction": 0.01,
  "command": ["./build-linux/odai", "--scene", "scene.bin", "--screenshot", "{output}", "30"]
}
```

## Verification and Definition of Done

1. CTest synthetic images cover exact match, tolerated channel noise,
   threshold boundary, failed regression, diff output, and dimension mismatch.
2. The comparison CLI can capture through the production screenshot command.
3. The full CTest suite passes before status is marked Implemented.

The synthetic CI test needs no GPU or game data. Production visual baselines
are scene and hardware specific and are added per rendering capability.
