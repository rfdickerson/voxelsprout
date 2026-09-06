#!/usr/bin/env python3
"""Reproduce the Solitude still directly from odai; PNG encoding preserves RGB."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name")
    parser.add_argument("--view", choices=("market", "waterfront", "sunset", "dragonsreach"), default="market")
    parser.add_argument("--size", default="3840x2160")
    parser.add_argument("--frames", type=int, default=240)
    parser.add_argument("--weather", default="SkyrimClear")
    parser.add_argument("--hour", type=float)
    parser.add_argument("--worldspace")
    parser.add_argument("--spawn")
    parser.add_argument("--binary", type=Path, default=ROOT / "build-linux-relwithdebinfo/odai")
    parser.add_argument("--validation", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--data", type=Path, default=Path.home() /
                        ".local/share/Steam/steamapps/common/Skyrim Special Edition/Data")
    parser.add_argument("--env", action="append", default=[], metavar="ODAI_KEY=VALUE")
    args = parser.parse_args()
    if args.hour is None:
        args.hour = 17.6 if args.view == "sunset" else (13.0 if args.view == "dragonsreach" else 14.0)
    width, height = map(int, args.size.split("x"))
    if width <= 0 or height <= 0 or not 1 <= args.frames < 900:
        parser.error("use a positive image size and 1..899 warmup frames")
    if not 0.0 <= args.hour < 24.0:
        parser.error("--hour must be in [0, 24)")
    destination = ROOT / "captures" / (args.name or f"solitude-{args.view}-4k")
    destination.parent.mkdir(parents=True, exist_ok=True)
    binary = args.binary.resolve()
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith("ODAI_")}
    settings = {
        "XDG_CACHE_HOME": str(ROOT / "captures/solitude-work/cache"),
        "ODAI_WINDOW_SIZE": args.size,
        "ODAI_RENDER_SIZE": args.size,
        "ODAI_RENDER_SCALE": "1.0",
        "ODAI_FNV_TEX_SIZE": "2048",
        "ODAI_TEXTURE_ANISOTROPY": "16",
        "ODAI_UPSCALE_MIPBIAS": "-0.35",
        "ODAI_TAA": "1",
        "ODAI_FNV_NOHUD": "1",
        "ODAI_FNV_AO": "xegtao",
        "ODAI_FNV_AO_RADIUS": "180",
        "ODAI_FNV_AO_INTENSITY": "1.4",
        "ODAI_FNV_AO_FINE": "0.25",
        "ODAI_XEGTAO_BLUR": "2",
        "ODAI_FNV_LOAD_RADIUS": "4",
    }
    views = {
        "market": ("SolitudeWorld", "-64530,-8260,-105551", "0", "-2", "0.95,0.95"),
        "waterfront": ("Tamriel", "-55000,-13000,-90000", "249", "5", "0.7,0.7"),
        "sunset": ("SolitudeWorld", "-63500,-8200,-105600", "145", "4", "0.85,0.85"),
        "dragonsreach": ("WhiterunWorld", "22500,700,11000", "291", "-5", "0.8,0.8"),
    }
    world, position, yaw, pitch, exposure = views[args.view]
    settings.update({"ODAI_FNV_SPAWN_POS": position, "ODAI_FNV_YAW": yaw,
                     "ODAI_FNV_PITCH": pitch, "ODAI_FNV_EXPOSURE_RANGE": exposure})
    if args.view == "waterfront":
        settings["ODAI_FOG_DENSITY"] = "0.00002"
    if args.view == "dragonsreach":
        settings.update({
            "ODAI_FNV_AO_RADIUS": "360", "ODAI_FNV_AO_INTENSITY": "2.2",
            "ODAI_FNV_LOAD_RADIUS": "5", "ODAI_FOG_DENSITY": "0.000008",
            "ODAI_SKYRIM_LOD_RADIUS": "9", "ODAI_RENDER_FAR": "180000",
            "ODAI_FNV_DAYTIME_LOCAL_LIGHT_SCALE": "0.08",
            "ODAI_FNV_DIFFUSE_WRAP": "0.08", "ODAI_FNV_AMBIENT_SCALE": "0.65",
        })
    if args.view == "sunset":
        settings.update({
            "ODAI_FNV_SHAFTS": "1", "ODAI_FOG_BASE": "-8300",
            "ODAI_FOG_DENSITY": "0.00012", "ODAI_FOG_FALLOFF": "0.0005",
            "ODAI_FOG_SCATTER": "3", "ODAI_FNV_AO_RADIUS": "360",
            "ODAI_FNV_AO_INTENSITY": "2.8",
        })
    if args.validation:
        sdk = Path.home() / "vulkan/1.4.357.0/x86_64"
        settings.update({
            "VK_LAYER_PATH": str(sdk / "share/vulkan/explicit_layer.d"),
            "VK_INSTANCE_LAYERS": "VK_LAYER_KHRONOS_validation",
            "VK_LAYER_ENABLES": "VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT",
            "LD_LIBRARY_PATH": str(sdk / "lib") + ":" + environment.get("LD_LIBRARY_PATH", ""),
        })
    for entry in args.env:
        key, separator, value = entry.partition("=")
        if not separator or not key.startswith("ODAI_"):
            parser.error("--env requires ODAI_KEY=VALUE")
        settings[key] = value
    environment.update(settings)
    ppm = destination.with_suffix(".ppm")
    png = destination.with_suffix(".png")
    log = destination.with_suffix(".log")
    command = [str(binary), "--stream", str(args.data.resolve()),
               "--plugin", "Skyrim.esm", "--worldspace", args.worldspace or world,
               "--no-resume", "--state", str(destination.with_suffix(".state.json")),
               "--weather", args.weather, "--hour", str(args.hour),
               "--upscaler", "temporal", "--upscaler-quality", "native",
               "--screenshot", str(ppm), str(args.frames)]
    if args.no_cache:
        command.append("--no-cache")
    if args.spawn:
        command.extend(["--spawn", args.spawn])
    diff = subprocess.check_output(["git", "diff", "HEAD"], cwd=ROOT)
    destination.with_suffix(".patch").write_bytes(diff)
    record = {
        "git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "executable_sha256": sha256(binary),
        "shader_sha256": {path.name: sha256(path) for path in sorted(
            (ROOT / "src/render/shaders").glob("*.spv"))},
        "cwd": str(binary.parent), "command": command,
        "environment_overrides": settings,
    }
    # Never mistake an old file for a successful new capture.
    ppm.unlink(missing_ok=True)
    png.unlink(missing_ok=True)
    with log.open("w") as output:
        result = subprocess.run(command, cwd=binary.parent, env=environment,
                                stdout=output, stderr=subprocess.STDOUT, timeout=300)
    record["exit_code"] = result.returncode
    destination.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
    if result.returncode:
        raise RuntimeError(f"odai failed ({result.returncode}); inspect {log}")
    logs = log.read_text()
    required = [
                "frame capture written:", "shutdown leak check: no renderer-owned live VkImage handles"]
    if any(text not in logs for text in required) or any(
            error in logs for error in ("[error]", "Validation Error", "SYNC-HAZARD")):
        raise RuntimeError(f"capture did not pass startup/render checks; inspect {log}")
    record["scene_log"] = [line for line in logs.splitlines() if any(
        term in line for term in ("load order:", "load-order source:", "inventory:",
                                 "spawn (engine space)", "resident=",
                                 "water normal texture ready", "GPU ms:", "createInstance (",
                                 "prewarm complete", "weather:", "selected GPU:",
                                 "framebuffer", "render resolution"))]
    with Image.open(ppm) as source:
        source.load()
        if source.size != (width, height) or source.mode != "RGB":
            raise RuntimeError(f"unexpected capture: {source.size}, {source.mode}")
        source.save(png, format="PNG")
        with Image.open(png) as encoded:
            if encoded.size != source.size or encoded.tobytes() != source.tobytes():
                raise RuntimeError("PNG conversion changed pixels")
        record["image"] = {"width": width, "height": height, "mode": "RGB",
                           "rgb_sha256": hashlib.sha256(source.tobytes()).hexdigest(),
                           "ppm_sha256": sha256(ppm), "png_sha256": sha256(png),
                           "lossless_conversion_verified": True}
    destination.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
    print(png)


if __name__ == "__main__":
    main()
