# Morrowind and Tamriel Rebuilt

The Bethesda viewer streams Morrowind, Tamriel Data, and Tamriel Rebuilt from
their original TES3 plugins. The archives are extracted once; they are not
copied into the repository or into the Steam installation.

## One-time extraction

```bash
MODS=/home/rfdickerson/.local/share/odai/morrowind
mkdir -p "$MODS/Tamriel_Data_25.05" "$MODS/Tamriel_Rebuilt_25.08.12"

7z x "/home/rfdickerson/Downloads/Tamriel_Data_25.05_HD.7z" \
  -o"$MODS/Tamriel_Data_25.05"
7z x "/home/rfdickerson/Downloads/Tamriel Rebuilt 25.08.12-42145-25-08-12-1755040619.7z" \
  -o"$MODS/Tamriel_Rebuilt_25.08.12"
```

Keep the BAIN directory names. The runtime uses these three roots:

```bash
MW="/home/rfdickerson/.local/share/Steam/steamapps/common/Morrowind/Data Files"
TD="/home/rfdickerson/.local/share/odai/morrowind/Tamriel_Data_25.05/00 Data Files"
TR="/home/rfdickerson/.local/share/odai/morrowind/Tamriel_Rebuilt_25.08.12"
```

## Stream Tamriel Rebuilt

Run the viewer from its build directory so its compiled shader paths resolve:

```bash
cd /home/rfdickerson/projects/voxelsprout/cmake-build-release
./odai_game_newvegas \
  --stream "$MW" --plugin Morrowind.esm --worldspace Vvardenfell \
  --mod "$TD" --mod "$TR/00 Core" --mod "$TR/01 Faction Integration" \
  --plugin-add TR_Factions.esp
```

`TR_Factions.esp` resolves the complete chain automatically:

```text
Morrowind.esm -> Tribunal.esm -> Bloodmoon.esm -> Tamriel_Data.esm ->
TR_Mainland.esm -> TR_Factions.esp
```

Later mod roots win for loose assets. TES3 exterior cells merge by grid,
interiors by case-insensitive name, and references by remapped FRMR identity.

## Rafael / Enhanced PBR renderer preset

These downloads target OpenMW's GLSL and `.omwfx` interfaces, which are not
binary-compatible with this renderer's Slang/Vulkan pass graph. Keep their
GPLv3 shader sources outside the MIT repository. The `rafael` preset maps the
packs' reusable intent onto native features instead:

- Enhanced PBR's documented legacy defaults: object roughness `0.84`, terrain
  roughness `0.92`, metallic `0.0`;
- Enhanced PBR's supplied `water_nm.png` in the renderer's water-normal slot;
- Rafael's linear-white `1.0` and exterior-shoulder `0.45` tonemap anchors;
- the existing native GGX BRDF, XeGTAO, TAA, cinematic grade, water, fog, and
  tonemap implementations.

Extract both user-provided archives once:

```bash
PACKS=/home/rfdickerson/.local/share/odai/morrowind/shader-packs
mkdir -p "$PACKS/rafael-2.0e" "$PACKS/enhanced-pbr-2.0e"

7z x "/home/rfdickerson/Downloads/Rafael's Shader Pack 2.0e 53667 2.0e 2026-08-12T12-13Z M4op4PJH.7z" \
  -o"$PACKS/rafael-2.0e"
7z x "/home/rfdickerson/Downloads/Enhanced PBR Lighting for OpenMW 0.49-0.52 53667 2.0e 2026-08-12T12-14Z p6TN64HF.7z" \
  -o"$PACKS/enhanced-pbr-2.0e"
```

Then add one argument to any Morrowind/Tamriel Rebuilt launch:

```bash
./odai_game_newvegas \
  --stream "$MW" --plugin Morrowind.esm --worldspace Vvardenfell \
  --mod "$TD" --mod "$TR/00 Core" --mod "$TR/01 Faction Integration" \
  --plugin-add TR_Factions.esp --shader-pack rafael
```

The preset uses the standard XDG data location above. A custom installation
can select a PNG or DDS directly with `ODAI_WATER_NORMAL=/path/to/water_nm.png`.
`ODAI_FNV_PBR_OBJECT_ROUGHNESS`, `ODAI_FNV_PBR_TERRAIN_ROUGHNESS`, and
`ODAI_FNV_PBR_METALLIC` override the defaults for A/B tuning. Authored native
PBR materials always take precedence over the preset.

## Almas Thirr checks

The city spans exterior cells `(5,-28)` and `(6,-28)`. Start directly inside a
representative interior with:

```bash
cd /home/rfdickerson/projects/voxelsprout/cmake-build-release
./odai_game_newvegas \
  --stream "$MW" --plugin Morrowind.esm --worldspace Vvardenfell \
  --mod "$TD" --mod "$TR/00 Core" --mod "$TR/01 Faction Integration" \
  --plugin-add TR_Factions.esp --interior "Almas Thirr, Canalworks"
```

For a clean river-approach still:

```bash
cd /home/rfdickerson/projects/voxelsprout/cmake-build-release
ODAI_WINDOW_SIZE=1920x1080 ODAI_FNV_HOUR=17.5 ODAI_FNV_NOHUD=1 \
ODAI_FNV_COLOR_LOOK=cinematic ODAI_FNV_LOAD_RADIUS=2 \
ODAI_FNV_SPAWN_POS=50500,220,243000 ODAI_FNV_YAW=-90.6 ODAI_FNV_PITCH=4.4 \
./odai_game_newvegas \
  --stream "$MW" --plugin Morrowind.esm --worldspace Vvardenfell \
  --mod "$TD" --mod "$TR/00 Core" --mod "$TR/01 Faction Integration" \
  --plugin-add TR_Factions.esp \
  --screenshot ../almas_thirr_sunset.ppm 120
```

## 30-second showcase capture

```bash
cd /home/rfdickerson/projects/voxelsprout/cmake-build-release
ODAI_WINDOW_SIZE=1920x1080 ODAI_FNV_HOUR=17.5 ODAI_FNV_NOHUD=1 \
ODAI_FNV_COLOR_LOOK=cinematic ODAI_FNV_LOAD_RADIUS=2 \
ODAI_FNV_SPAWN_POS=50500,220,243000 ODAI_CAPTURE_ENCODER=libopenh264 \
./odai_game_newvegas \
  --stream "$MW" --plugin Morrowind.esm --worldspace Vvardenfell \
  --mod "$TD" --mod "$TR/00 Core" --mod "$TR/01 Faction Integration" \
  --plugin-add TR_Factions.esp \
  --tour-file ../assets/tours/almas_thirr_river.txt --flythrough 30 \
  --capture-video ../almas_thirr_showcase.mp4 60 30
```

The capture path waits for streaming, TAA, and exposure while holding the first
tour pose, then advances at a fixed `1/60` second per recorded frame.

Verify the finished container with:

```bash
ffprobe -v error \
  -show_entries stream=index,codec_type,codec_name,width,height,r_frame_rate,nb_frames:format=duration \
  -of default=noprint_wrappers=1 ../almas_thirr_showcase.mp4
```

The expected video values are `h264`, `1920`, `1080`, `60/1`, and `1800`;
format duration is `30.000000`, and no `codec_type=audio` entry should appear.
