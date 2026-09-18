# Linux installation

The package is experimental; the continuous Skyrim route is not release-qualified.
Use Linux x86-64 with a Vulkan 1.4 GPU driver. The current Jolt dependency
build uses AVX2/FMA-class CPU instructions; older CPUs are not qualified. Python 3 and Tk are
required for the setup screen (`sudo apt install python3 python3-tk` on Ubuntu,
or `sudo dnf install python3-tkinter` on Fedora).
`vulkan-tools` is optional and supplies GPU information for diagnostic exports.

Extract the archive and run `bin/odai-launcher`. Keep `bin`, `lib` and `share`
together. You can launch from any working directory; the SDK and source checkout
are not needed. The Vulkan loader, GPU drivers and glibc come from your system.
The archive is built against the CI Linux distribution; older glibc versions are
not claimed compatible. Windows packages are not part of this release.

## Required installed content

Use Skyrim Special Edition with Skyrim, Update, Dawnguard, HearthFires,
Dragonborn, Fishing, Survival Mode, Rare Curios, Saints & Seducers, and
`_ResourcePack`. The English interface/voice archive is the initial baseline.
The exact plugin and archive filenames are in `share/odai/skyrim-slice.json`
([source manifest](../packaging/skyrim-slice.json)). Other languages and content
revisions need separate validation.

Install SMIM SE **2.08**, using its **Everything** selection, into a separate
Data-style root containing `meshes`, `textures`, and `SMIM-SE-Merged-All.esp`.
Install **JK’s Skyrim 1.7** into a separate root containing `JKs Skyrim.esp`
and its two BSA archives. Obtain mods from their authors; nothing downloads or
runs installers automatically. File presence checks do not prove mod versions:
confirm the versions in your mod manager before selecting those roots.

In the launcher, select the three roots and click Check content. Missing files
are listed explicitly. The generated profile loads the official content first,
then SMIM, then JK’s Skyrim. Extra mods are outside the supported slice baseline;
advanced CLI profiles remain available.

## Saves and troubleshooting

Configuration: `$XDG_CONFIG_HOME/odai` (default `~/.config/odai`).
Launcher saves and logs: `$XDG_DATA_HOME/odai` (default `~/.local/share/odai`).
New Game creates a new slot; Continue selects a slot. F5 writes the active slot,
retaining `.previous`. Recover previous loads that generation into a new slot.
A corrupt main save is reported rather than silently rolling progress backward.
The engine validates save checksums and content compatibility when loading.

Export diagnostics includes build version, content fingerprint, compatibility
codes and available GPU/driver information. It excludes raw logs and local paths.
A report before first launch may lack a fingerprint. Local `runtime.log` contains
more details and may contain personal paths; review it before sharing.

## Build prerequisites

On Ubuntu, install `build-essential cmake ninja-build pkg-config git curl zip
unzip tar xorg-dev libglu1-mesa-dev libvulkan-dev libasound2-dev libpulse-dev
libwayland-dev wayland-protocols libxkbcommon-dev python3-tk`. Install vcpkg and
set `VCPKG_ROOT` to its checkout. Install Slang's `slangc` on PATH (CI pins its
version in the workflow), or pass `-DSLANGC_EXECUTABLE=/path/to/slangc`.

Use the README presets. Shader outputs go into the build directory. Install and
package outputs contain an explicit engine-asset allowlist, never local game data.
