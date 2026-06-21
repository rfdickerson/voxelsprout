// Single translation unit that compiles the miniaudio implementation. Kept apart
// from miniaudio_backend.cc (which only uses the API) so the large generated
// body lives in its own TU with default compiler flags — mirrors the project's
// other vendored single-header impls (vma_usage.cc, the stb impl TUs).
#define MA_IMPLEMENTATION
#define MA_NO_ENCODING  // playback only — we never write audio files
#include <miniaudio.h>
