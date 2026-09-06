# Skyrim audio

Streaming Skyrim sessions load music and ambience from the installed game's
assets, using the same loose-file and archive precedence as the scene importer.
No game audio is bundled with the runtime.

Solitude uses `mus_explore_day_02.xwm` ("From Past to Present"). Elsewhere, the
initial exploration score uses `mus_explore_day_01.xwm` from 06:00 to 20:00 and
`mus_explore_night_01.xwm` otherwise. It fades in over four seconds and loops.
Solitude also uses a quieter ambient bus and excludes wilderness insect and
cricket descriptors that leak in from the surrounding Reach region. Authored
city emitters such as taverns and the windmill remain positional and audible.
Set `ODAI_SKYRIM_MUSIC` to an archive-relative music path to choose another track,
for example `music\explore\mus_explore_day_03.xwm`.
This is a looping score bed; combat playlists and live time-of-day transitions
are not implemented.

Weather adds rain and wind beds. Exterior regional descriptors provide loops
and chance-based events; nearby placed descriptors supply positional sounds,
including taverns and the Solitude windmill. Regional events draw independently.
Interior regional/placed ambience is not yet implemented.

XWM decoding requires `ffmpeg` on PATH. Decoded audio is cached under the stream
cache's `audio` directory, keyed by the authored virtual path and source bytes so
same-named sounds in different directories and changed mod assets remain distinct.
`--no-cache` disables cooked-cell caching while keeping this audio extraction
directory available.

Interactive playback and `--capture-audio` both include music and ambience.
Use `--capture-video <output.mp4> <fps> <seconds> --capture-audio` for a recording
with the engine mix. `--capture-seed <u32>` fixes regional random choices.
