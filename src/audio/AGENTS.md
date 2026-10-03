# src/audio

- Responsibility: Audio service, null/miniaudio backends and WAV evidence.
- Public interfaces: `audio.h`, `audio_backend.h`, `audio_types.h`, `wav_writer.h`.
- Invariants: Keep null backend usable without device availability; preserve service/backend separation.
- Dependencies: odai_audio links core; miniaudio is optional.
- Extension points: Backend implementations and audio service commands.
- Tests: `odai_audio_tests`.
- Validation: from repository root, `python3 tools/ai/nav.py tests <changed-file>`;
  `python3 tools/ai/nav.py run-tests <changed-file>` builds and runs associated tests.
  Use `check <target>` for a compile check; read parent guidance for full validation.
- Traps: miniaudio_impl.cc is the implementation translation unit; decoding integration lives in games/bethesda/audio_decode.cc.
