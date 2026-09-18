# ODAI

ODAI is an MIT-licensed Vulkan runtime for Bethesda worlds. The first release
being prepared is a **Linux Skyrim SE gameplay slice**: Riverwood → Bleak Falls
Barrow → Whiterun, using **JK’s Skyrim 1.7 + SMIM SE 2.08 Everything**.

**Experimental: the continuous playable route has not passed release acceptance.**
Morrowind, Oblivion, Fallout 3 and New Vegas currently have experimental import
support; this is not a promise that those games are playable to completion.

## Install and play

Follow the [installation guide](docs/INSTALLATION.md), then run `odai-launcher`.
Select your Skyrim SE Data directory and the two extracted mod roots, check the
content, and choose New Game or Continue. Game and mod assets must be supplied
by you. No models, textures, scripts, voices or game fonts are distributed.

The launcher keeps configuration and saves in your user directories. Saves use
ODAI’s own format; Skyrim `.ess` saves and SKSE are unsupported. The previous
save generation can be opened with Recover previous without replacing the slot.

Keyboard: WASD/mouse move and look; E interacts; primary mouse attacks;
I opens inventory; J journal; M map; Escape pauses; F5 saves; F9 loads.
Inventory/dialogue navigation supports keyboard and controller. Full controls
and advanced CLI usage are in the [runtime reference](docs/RUNTIME_REFERENCE.md).

## Build

Install a C++20 compiler, CMake 3.21+, Ninja, vcpkg, Vulkan development files,
and `slangc`. Set `VCPKG_ROOT`; see the installation guide for Linux packages.

```sh
cmake --preset linux-vcpkg
cmake --build --preset linux-vcpkg -j
ctest --test-dir build-linux --output-on-failure
```

Use `linux-vcpkg-relwithdebinfo` or `linux-vcpkg-release` for performance work.
`cmake --install build-linux-release --prefix /path/to/odai` stages the runtime;
`cpack --config build-linux-release/CPackConfig.cmake` creates an experimental archive
and SHA-256 checksum. Packaging is not evidence that gameplay release gates pass.

## Status and contribution

- [Compatibility](docs/COMPATIBILITY.md) · [Release gates](docs/RELEASE_CHECKLIST.md)
- [Runtime implementation status](docs/SKYRIM_FIRST_RUNTIME.md) · [Documentation](docs/index.md)
- [Contributing](CONTRIBUTING.md) · [Security reporting](SECURITY.md)
- [Changelog](CHANGELOG.md) · [Third-party notices](THIRD_PARTY_NOTICES.md)

ODAI is an independent project and is not affiliated with Bethesda or OpenMW.
