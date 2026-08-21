#!/usr/bin/env python3
"""Render and assemble the deterministic 90-second Skyrim showcase."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "assets" / "tours" / "skyrim_showcase" / "showcase.json"
DEFAULT_OUTPUT = ROOT / "captures" / "skyrim_showcase_1080p60.mp4"


class ShowcaseError(RuntimeError):
    pass


def run(
    command: list[str], *, env: dict[str, str] | None = None, cwd: Path | None = None
) -> None:
    print("+", " ".join(command), flush=True)
    completed = subprocess.run(command, env=env, cwd=cwd, check=False)
    if completed.returncode:
        raise ShowcaseError(f"command failed with exit code {completed.returncode}: {command[0]}")


def audit_tour(
    probe_executable: Path,
    data: Path,
    worldspace: str,
    tour: Path,
    report: Path,
    load_order: Path | None,
) -> None:
    command = [
        str(probe_executable), str(data), "--tourcheck", worldspace, str(tour),
    ]
    if load_order is not None:
        command.append(str(load_order))
    print("+", " ".join(command), flush=True)
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    report.write_text(completed.stdout, encoding="utf-8")
    if completed.stderr:
        (report.parent / f"{report.stem}.stderr.log").write_text(
            completed.stderr, encoding="utf-8")
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise ShowcaseError(
            f"tour audit returned malformed JSON for {tour}; preserved {report}: {error}"
        ) from error
    if completed.returncode or result.get("ok") is not True:
        raise ShowcaseError(f"tour audit rejected {tour}; see {report}")


def probe(path: Path, *, count_frames: bool = False) -> dict[str, Any]:
    command = [
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_streams", "-show_format",
    ]
    if count_frames:
        command.append("-count_frames")
    command.append(str(path))
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode:
        raise ShowcaseError(f"ffprobe rejected {path}: {completed.stderr.strip()}")
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise ShowcaseError(f"ffprobe returned malformed JSON for {path}: {error}") from error


def stream(metadata: dict[str, Any], codec_type: str) -> dict[str, Any]:
    for candidate in metadata.get("streams", []):
        if candidate.get("codec_type") == codec_type:
            return candidate
    raise ShowcaseError(f"missing {codec_type} stream")


def rational(value: str) -> float:
    numerator, separator, denominator = value.partition("/")
    if not separator:
        return float(value)
    return float(numerator) / float(denominator)


def validate_clip(path: Path, width: int, height: int, fps: int, duration: float) -> None:
    metadata = probe(path)
    video = stream(metadata, "video")
    audio = stream(metadata, "audio")
    if int(video.get("width", 0)) != width or int(video.get("height", 0)) != height:
        raise ShowcaseError(f"{path} is not {width}x{height}")
    if abs(rational(video.get("avg_frame_rate", "0")) - fps) > 0.001:
        raise ShowcaseError(f"{path} is not {fps} fps")
    if int(audio.get("sample_rate", 0)) != 48000 or int(audio.get("channels", 0)) != 2:
        raise ShowcaseError(f"{path} does not contain 48 kHz stereo audio")
    actual_duration = float(metadata.get("format", {}).get("duration", 0.0))
    if abs(actual_duration - duration) > (1.5 / fps):
        raise ShowcaseError(
            f"{path} duration is {actual_duration:.4f}s, expected {duration:.4f}s")


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ShowcaseError(f"cannot read manifest {path}: {error}") from error
    if manifest.get("version") != 1:
        raise ShowcaseError("showcase manifest version must be 1")
    shots = manifest.get("shots")
    if not isinstance(shots, list) or len(shots) != 8:
        raise ShowcaseError("showcase manifest must contain exactly eight shots")
    required = {"id", "tour", "worldspace", "weather", "hour", "duration", "seed"}
    total = 0.0
    seen: set[str] = set()
    for index, shot in enumerate(shots):
        if not isinstance(shot, dict) or not required.issubset(shot):
            raise ShowcaseError(f"shot {index} is missing required fields")
        shot_id = str(shot["id"])
        if not shot_id or shot_id in seen:
            raise ShowcaseError(f"shot {index} has an empty or duplicate id")
        seen.add(shot_id)
        if not str(shot["worldspace"]) or not str(shot["weather"]):
            raise ShowcaseError(f"shot {shot_id} must name a worldspace and weather")
        hour = float(shot["hour"])
        duration = float(shot["duration"])
        if not 0.0 <= hour < 24.0 or duration <= 1.0:
            raise ShowcaseError(f"shot {shot_id} has an invalid hour or duration")
        total += duration
        tour = path.parent / str(shot["tour"])
        try:
            rows = []
            for line in tour.read_text(encoding="utf-8").splitlines():
                values = line.split("#", 1)[0].split()
                if values:
                    rows.append([float(value) for value in values])
        except (OSError, ValueError) as error:
            raise ShowcaseError(f"malformed tour {tour}: {error}") from error
        if len(rows) < 4 or any(len(row) != 6 for row in rows):
            raise ShowcaseError(f"tour {tour} needs at least four six-number rows")
    transition = float(manifest.get("transition_seconds", 1.0))
    expected = float(manifest.get("master_duration_seconds", 90.0))
    assembled = total - transition * (len(shots) - 1)
    if abs(assembled - expected) > 1e-6:
        raise ShowcaseError(
            f"source durations/crossfades produce {assembled}s, expected {expected}s")
    return manifest


def select_h264_encoder() -> str:
    completed = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"],
        text=True, capture_output=True, check=False)
    if completed.returncode:
        raise ShowcaseError("ffmpeg could not enumerate its video encoders")
    available = {
        fields[1]
        for line in completed.stdout.splitlines()
        if len(fields := line.split()) >= 2 and fields[0].startswith("V")
    }
    requested = os.environ.get("ODAI_SHOWCASE_ENCODER")
    if requested:
        if requested not in available:
            raise ShowcaseError(f"requested H.264 encoder is unavailable: {requested}")
        return requested
    for candidate in ("libx264", "libopenh264"):
        if candidate in available:
            return candidate
    raise ShowcaseError("ffmpeg has no supported H.264 encoder (libx264 or libopenh264)")


def assemble(clips: list[Path], shots: list[dict[str, Any]], manifest: dict[str, Any], output: Path) -> None:
    transition = float(manifest["transition_seconds"])
    total_duration = float(manifest["master_duration_seconds"])
    command = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-y"]
    for clip in clips:
        command.extend(["-i", str(clip)])

    filters: list[str] = []
    combined_duration = float(shots[0]["duration"])
    video_label = "0:v"
    audio_label = "0:a"
    for index in range(1, len(clips)):
        offset = combined_duration - transition
        next_video = f"vx{index}"
        next_audio = f"ax{index}"
        filters.append(
            f"[{video_label}][{index}:v]xfade=transition=fade:duration={transition}:"
            f"offset={offset:.6f}[{next_video}]")
        filters.append(
            f"[{audio_label}][{index}:a]acrossfade=d={transition}:c1=tri:c2=tri[{next_audio}]")
        video_label = next_video
        audio_label = next_audio
        combined_duration += float(shots[index]["duration"]) - transition
    filters.append(
        f"[{video_label}]fade=t=in:st=0:d=0.5,"
        f"fade=t=out:st={total_duration - 0.5}:d=0.5,format=yuv420p[vout]")
    filters.append(
        f"[{audio_label}]afade=t=in:st=0:d=0.5,"
        f"afade=t=out:st={total_duration - 0.5}:d=0.5,"
        "loudnorm=I=-18:TP=-1.5:LRA=11[aout]")
    encoder = select_h264_encoder()
    video_options = (
        ["-preset", "slow", "-crf", "18"]
        if encoder == "libx264"
        else ["-b:v", "24000000", "-maxrate", "30000000", "-bufsize", "60000000"]
    )
    command.extend([
        "-filter_complex", ";".join(filters),
        "-map", "[vout]", "-map", "[aout]",
        "-r", str(int(manifest["fps"])), "-t", f"{total_duration:.6f}",
        "-c:v", encoder,
        *video_options, "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
        "-movflags", "+faststart", str(output),
    ])
    run(command)


def validate_master(path: Path, manifest: dict[str, Any]) -> None:
    metadata = probe(path, count_frames=True)
    video = stream(metadata, "video")
    audio = stream(metadata, "audio")
    width = int(manifest["width"])
    height = int(manifest["height"])
    fps = int(manifest["fps"])
    expected_frames = int(round(float(manifest["master_duration_seconds"]) * fps))
    if (int(video.get("width", 0)), int(video.get("height", 0))) != (width, height):
        raise ShowcaseError("master has the wrong dimensions")
    if abs(rational(video.get("avg_frame_rate", "0")) - fps) > 0.001:
        raise ShowcaseError("master has the wrong frame rate")
    counted = int(video.get("nb_read_frames", video.get("nb_frames", 0)) or 0)
    if counted != expected_frames:
        raise ShowcaseError(f"master contains {counted} frames, expected {expected_frames}")
    if video.get("codec_name") != "h264" or video.get("pix_fmt") != "yuv420p":
        raise ShowcaseError("master is not H.264/YUV420p")
    if audio.get("codec_name") != "aac" or int(audio.get("channels", 0)) != 2:
        raise ShowcaseError("master is not stereo AAC")
    if int(audio.get("sample_rate", 0)) != 48000:
        raise ShowcaseError("master audio is not 48 kHz")


def generate_anchor_contact_sheet(
    clips: list[Path], shots: list[dict[str, Any]], work_dir: Path
) -> Path:
    anchor_dir = work_dir / "anchors"
    anchor_dir.mkdir(parents=True, exist_ok=True)
    anchor_index = 0
    for clip, shot in zip(clips, shots):
        duration = float(shot["duration"])
        for time_seconds in (0.5, duration * 0.5, duration - 0.5):
            anchor = anchor_dir / f"anchor_{anchor_index:02d}.png"
            run([
                "ffmpeg", "-hide_banner", "-loglevel", "warning", "-y",
                "-ss", f"{time_seconds:.6f}", "-i", str(clip),
                "-frames:v", "1", "-vf", "scale=480:270", str(anchor),
            ])
            anchor_index += 1
    contact_sheet = work_dir / "skyrim_showcase_contact_sheet.png"
    run([
        "ffmpeg", "-hide_banner", "-loglevel", "warning", "-y",
        "-framerate", "1", "-start_number", "0",
        "-i", str(anchor_dir / "anchor_%02d.png"),
        "-frames:v", "1", "-vf", "tile=3x8:padding=2:margin=2",
        str(contact_sheet),
    ])
    return contact_sheet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--odai", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--load-order", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--keep-clips", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--audit-only", action="store_true",
        help="run all real-data tour checks and preserve their JSON reports without capturing",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest = load_manifest(args.manifest.resolve())
        output = args.output.resolve()
        if output.exists() and not args.force and not args.audit_only:
            raise ShowcaseError(f"refusing to overwrite existing capture: {output}")
        if args.dry_run:
            print("validated eight tours, source durations, crossfades, and 90-second master")
            return 0
        if not args.odai.is_file() or not os.access(args.odai, os.X_OK):
            raise ShowcaseError(f"odai executable is unavailable: {args.odai}")
        bethesda_probe = args.odai.resolve().with_name("odai_bethesda_probe")
        if not bethesda_probe.is_file() or not os.access(bethesda_probe, os.X_OK):
            raise ShowcaseError(f"odai_bethesda_probe is unavailable: {bethesda_probe}")
        if not (args.data / "Skyrim.esm").is_file():
            raise ShowcaseError(f"Skyrim.esm is unavailable under: {args.data}")
        if args.load_order is not None and not args.load_order.is_file():
            raise ShowcaseError(f"load order is unavailable: {args.load_order}")
        work_dir = (args.work_dir or output.with_suffix(".work")).resolve()
        clips = [work_dir / f"{index + 1:02d}_{shot['id']}.mp4"
                 for index, shot in enumerate(manifest["shots"])]
        output.parent.mkdir(parents=True, exist_ok=True)
        work_dir.mkdir(parents=True, exist_ok=True)
        if output.exists() and not args.audit_only:
            output.unlink()

        environment = os.environ.copy()
        environment.update({
            "ODAI_WINDOW_SIZE": f"{manifest['width']}x{manifest['height']}",
            "ODAI_RENDER_SIZE": f"{manifest['width']}x{manifest['height']}",
            "ODAI_FNV_NOHUD": "1",
            "ODAI_FNV_LOAD_RADIUS": "4",
            "ODAI_TERRAIN_TESS": "1",
        })
        for shot in manifest["shots"]:
            tour = (args.manifest.parent / shot["tour"]).resolve()
            audit_tour(
                bethesda_probe,
                args.data.resolve(),
                str(shot["worldspace"]),
                tour,
                work_dir / f"{shot['id']}.tourcheck.json",
                args.load_order.resolve() if args.load_order is not None else None,
            )
        if args.audit_only:
            print(f"all eight Skyrim tours passed; reports preserved under {work_dir}")
            return 0
        if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
            raise ShowcaseError("ffmpeg and ffprobe must be available on PATH")
        for index, (shot, clip) in enumerate(zip(manifest["shots"], clips)):
            if clip.exists() and not args.force:
                validate_clip(
                    clip, int(manifest["width"]), int(manifest["height"]),
                    int(manifest["fps"]), float(shot["duration"]))
                print(f"reusing validated clip {clip}")
                continue
            if clip.exists():
                clip.unlink()
            state = work_dir / f"{shot['id']}.state.json"
            command = [
                str(args.odai.resolve()), "--stream", str(args.data.resolve()),
                "--plugin", "Skyrim.esm", "--worldspace", str(shot["worldspace"]),
                "--weather", str(shot["weather"]), "--hour", str(shot["hour"]),
                "--tour-file", str((args.manifest.parent / shot["tour"]).resolve()),
                "--flythrough", str(shot["duration"]),
                "--capture-video", str(clip), str(manifest["fps"]), str(shot["duration"]),
                "--capture-audio", "--capture-seed", str(shot["seed"]),
                "--state", str(state), "--no-resume",
                "--upscaler", "temporal", "--upscaler-quality", "native",
            ]
            if args.load_order is not None:
                command.extend(["--load-order", str(args.load_order.resolve())])
            # The runtime's shader search still includes paths relative to the
            # build directory. Make the pipeline independent of the shell's
            # current working directory while that compatibility path exists.
            run(command, env=environment, cwd=args.odai.resolve().parent)
            validate_clip(
                clip, int(manifest["width"]), int(manifest["height"]),
                int(manifest["fps"]), float(shot["duration"]))
            print(f"captured shot {index + 1}/{len(clips)}: {shot['id']}")

        contact_sheet = generate_anchor_contact_sheet(clips, manifest["shots"], work_dir)
        print(f"wrote visual-review contact sheet {contact_sheet}")
        assemble(clips, manifest["shots"], manifest, output)
        validate_master(output, manifest)
        if not args.keep_clips:
            for clip in clips:
                clip.unlink(missing_ok=True)
            for state in work_dir.glob("*.state.json"):
                state.unlink(missing_ok=True)
            try:
                work_dir.rmdir()
            except OSError:
                pass
        print(f"wrote {output}: 1920x1080, 60 fps, 5400 frames, 90 seconds")
        return 0
    except ShowcaseError as error:
        print(f"error: {error}", file=sys.stderr)
        print("work files were preserved for recovery", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
