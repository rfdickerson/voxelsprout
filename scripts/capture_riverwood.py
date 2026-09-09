#!/usr/bin/env python3
"""Capture deterministic Skyrim SE Riverwood lighting/material comparisons."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
VIEWS = {
    # Engine coordinates. Each view exercises a distinct acceptance area.
    "street": ("22300,-60,44400", "130", "-12"),
    "mill": ("21300,500,42300", "137", "-3"),
    "shade": ("20800,520,43800", "126", "-5"),
    "forest": ("18600,800,44900", "111", "-2"),
}
WEATHERS = {
    "day": ("SkyrimClearFF", 10.5),
    "overcast": ("SkyrimCloudyTU", 13.0),
    "dusk": ("SkyrimClearFF", 19.25),
    "night": ("SkyrimClearFF", 0.5),
}


def digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def capture_settings(size, profile, position, yaw, pitch):
    settings = {
        "XDG_CACHE_HOME": str(ROOT / "captures/riverwood-work/cache"),
        "ODAI_WINDOW_HIDPI": "0",
        "ODAI_WINDOW_SIZE": size, "ODAI_RENDER_SIZE": size,
        "ODAI_RENDER_SCALE": "1.0", "ODAI_FNV_TEX_SIZE": "2048",
        "ODAI_TEXTURE_ANISOTROPY": "16", "ODAI_UPSCALE_MIPBIAS": "-0.35",
        "ODAI_TAA": "1", "ODAI_FNV_NOHUD": "1", "ODAI_FNV_LOAD_RADIUS": "4",
        "ODAI_SKYRIM_LOD_RADIUS": "9", "ODAI_RENDER_FAR": "180000",
        "ODAI_FNV_AO": "xegtao", "ODAI_XEGTAO_BLUR": "2",
        "ODAI_FNV_SPAWN_POS": position, "ODAI_FNV_YAW": yaw, "ODAI_FNV_PITCH": pitch,
        "ODAI_GPU_TIMINGS": "1", "ODAI_DRAW_COUNTS": "1",
    }
    if profile == "small-fast":
        settings.update({"ODAI_FNV_LOAD_RADIUS": "1", "ODAI_SKYRIM_LOD_RADIUS": "5",
                         "ODAI_SHADOW_DISTANCE": "3000", "ODAI_TERRAIN_TESS": "0",
                         "ODAI_SHADOW_FAR_RESOLUTION": "1024",
                         "ODAI_WATER_REFLECTION_DIVISOR": "4",
                         "ODAI_UPSCALE_MIPBIAS": "0"})
    return settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", choices=VIEWS, default="street")
    parser.add_argument("--look", choices=WEATHERS, default="day")
    parser.add_argument("--weather", help="Override the look with an authored weather EditorID")
    parser.add_argument("--variant", choices=("lit", "ao", "gi", "effects-off"), default="lit")
    parser.add_argument("--profile", choices=("fidelity", "small-fast"), default="fidelity",
                        help="small-fast uses a 768x432 window and shorter detail/shadow ranges; play honors display DPI")
    parser.add_argument("--content-profile", type=Path,
                        help="ODAI/MO2 content profile selecting mod assets and active plugins")
    parser.add_argument("--play", action="store_true", help="Leave Riverwood interactive instead of capturing")
    parser.add_argument("--size", help="Window size in logical points for play, output pixels for captures")
    parser.add_argument("--frames", type=int, default=360)
    parser.add_argument("--name")
    parser.add_argument("--binary", type=Path, default=ROOT / "build-linux-relwithdebinfo/odai")
    parser.add_argument("--data", type=Path, default=Path.home() /
                        ".local/share/Steam/steamapps/common/Skyrim Special Edition/Data")
    parser.add_argument("--validation", action="store_true")
    parser.add_argument("--env", action="append", default=[], metavar="ODAI_KEY=VALUE")
    args = parser.parse_args()
    args.size = args.size or ("768x432" if args.profile == "small-fast" else "1920x1080")
    width, height = map(int, args.size.split("x"))
    if width <= 0 or height <= 0 or not 60 <= args.frames < 900:
        parser.error("use a positive size and 60..899 warmup frames")

    position, yaw, pitch = VIEWS[args.view]
    weather, hour = WEATHERS[args.look]
    weather = args.weather or weather
    stem = args.name or f"riverwood-{args.view}-{args.look}-{args.variant}"
    destination = ROOT / "captures" / stem
    destination.parent.mkdir(parents=True, exist_ok=True)
    binary = args.binary.resolve()
    settings = capture_settings(args.size, args.profile, position, yaw, pitch)
    if args.play:
        settings.pop("ODAI_FNV_NOHUD")
        # Interactive size is in desktop logical points. Let GLFW allocate the
        # native HiDPI framebuffer and render at that extent, including resizes.
        # Captures retain their explicit output-pixel contract.
        settings.pop("ODAI_RENDER_SIZE")
        settings["ODAI_WINDOW_HIDPI"] = "1"
    if args.variant in ("ao", "gi"):
        settings["ODAI_FNV_DEBUGVIEW"] = "ao" if args.variant == "ao" else "ssgi"
    elif args.variant == "effects-off":
        settings["ODAI_FNV_AO"] = "off"
        settings["ODAI_FNV_EXTERIOR_GI"] = "off"
    if args.validation:
        settings.update({"VK_INSTANCE_LAYERS": "VK_LAYER_KHRONOS_validation",
                         "VK_LAYER_VALIDATE_SYNC": "1"})
    for override in args.env:
        key, separator, value = override.partition("=")
        if not separator or not key.startswith("ODAI_"):
            parser.error("--env requires ODAI_KEY=VALUE")
        settings[key] = value

    environment = {key: value for key, value in os.environ.items() if not key.startswith("ODAI_")}
    environment.update(settings)
    ppm = destination.with_suffix(".ppm")
    png = destination.with_suffix(".png")
    log = destination.with_suffix(".log")
    command = [str(binary), "--stream", str(args.data.resolve()), "--plugin", "Skyrim.esm",
               "--worldspace", "Tamriel", "--no-resume", "--weather", weather,
               "--hour", str(hour), "--upscaler", "temporal", "--upscaler-quality", "native"]
    if args.content_profile:
        command.extend(["--profile", str(args.content_profile.resolve())])
    if args.play:
        print(f"Playing Riverwood at {args.size}; log: {log}", flush=True)
        with log.open("w") as output:
            result = subprocess.run(command, cwd=binary.parent, env=environment,
                                    stdout=output, stderr=subprocess.STDOUT)
        raise SystemExit(result.returncode)
    command.extend(["--screenshot", str(ppm), str(args.frames)])
    ppm.unlink(missing_ok=True)
    png.unlink(missing_ok=True)
    with log.open("w") as output:
        result = subprocess.run(command, cwd=binary.parent, env=environment,
                                stdout=output, stderr=subprocess.STDOUT, timeout=300)
    if result.returncode:
        raise RuntimeError(f"odai failed ({result.returncode}); inspect {log}")
    logs = log.read_text(encoding="utf-8", errors="replace")
    required = ("frame capture written:", "shutdown leak check: no renderer-owned live VkImage handles")
    if any(item not in logs for item in required):
        raise RuntimeError(f"capture did not complete cleanly; inspect {log}")
    if args.validation and ("validation=on" not in logs or "WITHOUT validation" in logs):
        raise RuntimeError(f"validation layer did not load; inspect {log}")
    if args.validation and "Validation Error:" in logs:
        raise RuntimeError(f"Vulkan validation errors; inspect {log}")
    with Image.open(ppm) as source:
        source.load()
        if source.size != (width, height) or source.mode != "RGB":
            raise RuntimeError(f"unexpected capture: {source.size}, {source.mode}")
        source.save(png, format="PNG")
        pixels = source.tobytes()
    record = {
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "command": command, "environment_overrides": settings,
        "executable_sha256": digest(binary), "rgb_sha256": hashlib.sha256(pixels).hexdigest(),
        "image": {"width": width, "height": height, "mode": "RGB"},
        "timings": [line for line in logs.splitlines() if "GPU ms:" in line or "frame stats over" in line],
    }
    destination.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
    print(png)


if __name__ == "__main__":
    main()
