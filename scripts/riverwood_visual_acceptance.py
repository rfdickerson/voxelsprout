#!/usr/bin/env python3
"""Record local, lossless Riverwood visual acceptance sequences (not FPS tests)."""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess

from PIL import Image, ImageDraw, ImageChops, ImageStat
from capture_riverwood import ROOT, WEATHERS, capture_settings, digest

FIXTURES = {
    "forest-approach": ("13980,46,48800", "-60", "-2"),
    "water": ("21300,500,42300", "165", "-18"),
    "foliage": ("21300,500,42300", "137", "-3"),
    "road": ("22300,-60,44400", "130", "-12"),
    "handoff": ("20200,800,44800", "-45", "-2"),
    "village-route": ("20300,800,46000", "-26", "-2"),
}


def write_index(output):
    cases = []
    for path in sorted(output.glob("*/manifest.json")):
        record = json.loads(path.read_text())
        if "frame_count" in record:
            cases.append({"name": path.parent.name, "label": path.parent.name.replace("forest-route", "village-route"), "frames": record["frame_count"],
                          "pose": record["environment_overrides"]["ODAI_FNV_SPAWN_POS"],
                          "extent": record["extent"], "status": record["status"]})
    data = json.dumps(cases).replace("</", "<\\/")
    html = """<!doctype html><meta charset="utf-8"><title>Riverwood visual acceptance</title>
<style>body{background:#171b20;color:#edf1f5;font:16px system-ui;margin:24px}header{max-width:1100px}
button,select,input{font:inherit;margin:6px;padding:6px}img{display:block;max-width:100%;height:auto;background:black}
#frame{width:min(600px,60vw)}a{color:#9bcbff}small{color:#b9c2cc}</style>
<header><h1>Riverwood visual acceptance</h1><p>Lossless native frames, fixed 60 Hz simulation. Capture readback is not an FPS benchmark.
No matched retail reference is attached; recordings are not automatic visual approvals.</p>
<select id="case"></select><button id="play">Play / pause</button><button id="native">Native pixels / fit</button>
<input id="frame" type="range" min="0" value="0"><span id="number"></span><p id="info"></p>
<p><a id="manifest">Capture settings and hashes</a> · <a id="log">Runtime log</a></p></header>
<img id="image" alt="Recorded Riverwood frame"><script>
const cases=__DATA__, picker=document.getElementById('case'), slider=document.getElementById('frame'), picture=document.getElementById('image');
let playing=false, last=0, nativeSize=false;
for(const c of cases){let option=document.createElement('option');option.textContent=c.label;picker.append(option)}
function show(){if(!cases.length)return;const c=cases[picker.selectedIndex];slider.max=c.frames-1;
slider.value=Math.min(Number(slider.value),c.frames-1);picture.src=c.name+'/frame_'+String(slider.value).padStart(5,'0')+'.png';
document.getElementById('number').textContent=slider.value+' / '+(c.frames-1);
document.getElementById('info').textContent=c.extent.join(' × ')+' pixels; initial engine position '+c.pose+'; '+c.status;
document.getElementById('manifest').href=c.name+'/manifest.json';document.getElementById('log').href=c.name+'/runtime.log'}
picker.onchange=()=>{slider.value=0;show()};slider.oninput=show;
document.getElementById('play').onclick=()=>{playing=!playing};
document.getElementById('native').onclick=()=>{nativeSize=!nativeSize;picture.style.maxWidth=nativeSize?'none':'100%'};
function tick(t){if(playing&&t-last>=1000/60){slider.value=(Number(slider.value)+1)%cases[picker.selectedIndex].frames;show();last=t}requestAnimationFrame(tick)}
show();requestAnimationFrame(tick);
</script>""".replace("__DATA__", data)
    (output / "index.html").write_text(html)


def convert_frame(frame, extent):
    with Image.open(frame) as source:
        source.load()
        if source.size != extent or source.mode != "RGB":
            raise RuntimeError("capture is not native requested RGB extent")
        pixels = source.copy()
        rgb_hash = hashlib.sha256(source.tobytes()).hexdigest()
        source.save(frame.with_suffix(".png"), compress_level=1)
    frame.unlink()  # PNG retains exactly the original pixels.
    return pixels, rgb_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", nargs="+", choices=FIXTURES, default=list(FIXTURES))
    parser.add_argument("--look", nargs="+", choices=WEATHERS, default=list(WEATHERS))
    parser.add_argument("--variant", choices=("lit", "ao", "ao-off", "taa-off", "albedo"), default="lit")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--tour", type=Path, help="Use an authored camera route with real cell streaming")
    parser.add_argument("--size", default="1536x864")
    parser.add_argument("--window-size", help="Logical window size; use native DPI and verify --size output pixels")
    parser.add_argument("--validation", action="store_true")
    parser.add_argument("--index-only", action="store_true", help="Rebuild the local frame viewer without capturing")
    parser.add_argument("--output", type=Path, default=ROOT / "captures/visual-acceptance")
    parser.add_argument("--binary", type=Path, default=ROOT / "build-linux-relwithdebinfo/odai")
    args = parser.parse_args()
    if not 1 <= args.frames <= 3600:
        parser.error("frames must be 1..3600")
    width, height = map(int, args.size.split("x"))
    if min(width, height) < 1:
        parser.error("size must be positive")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.index_only:
        write_index(args.output)
        return
    shader_hashes = {p.name: digest(p) for p in sorted((ROOT / "src/render/shaders").glob("*.spv"))}
    for fixture in args.fixture:
        for look in args.look:
            target = args.output / f"{fixture}-{look}-{args.variant}"
            target.mkdir()  # Evidence is immutable: choose a new output directory to rerun.
            settings = capture_settings(args.size, "small-fast", *FIXTURES[fixture])
            if args.window_size:
                settings["ODAI_WINDOW_SIZE"] = args.window_size
                for key in ("ODAI_WINDOW_HIDPI", "ODAI_RENDER_SIZE", "ODAI_RENDER_SCALE"):
                    settings.pop(key, None)
            if args.variant == "ao-off": settings["ODAI_FNV_AO"] = "off"
            if args.variant == "taa-off": settings["ODAI_TAA"] = "0"
            if args.variant in ("ao", "albedo"): settings["ODAI_FNV_DEBUGVIEW"] = args.variant
            if fixture in ("handoff", "village-route", "forest-approach"):
                settings.update({"ODAI_FNV_BENCH": "1", "ODAI_FNV_BENCH_FIXED_DT": "1",
                                 "ODAI_FNV_BENCH_SPEED": "250", "ODAI_FNV_BENCH_TURN": "0",
                                 "ODAI_FNV_BENCH_HEADING": FIXTURES[fixture][1]})
            if args.validation:
                settings.update({"VK_INSTANCE_LAYERS": "VK_LAYER_KHRONOS_validation", "VK_LAYER_VALIDATE_SYNC": "1"})
            environment = {k: v for k, v in os.environ.items() if not k.startswith("ODAI_")}
            environment.update(settings)
            weather, hour = WEATHERS[look]
            command = [str(args.binary.resolve()), "--stream", str(Path.home() / ".local/share/Steam/steamapps/common/Skyrim Special Edition/Data"),
                       "--plugin", "Skyrim.esm", "--worldspace", "Tamriel", "--no-resume", "--weather", weather,
                       "--hour", str(hour), "--upscaler", "temporal", "--upscaler-quality", "native",
                       "--capture-seq", str(target), "60", str((args.frames + 0.01) / 60)]
            if args.tour:
                settings["ODAI_TOUR_STREAMING"] = "1"
                environment["ODAI_TOUR_STREAMING"] = "1"
                settings["ODAI_FNV_TOUR_TRACE"] = str(target / "camera.csv")
                environment["ODAI_FNV_TOUR_TRACE"] = settings["ODAI_FNV_TOUR_TRACE"]
                command += ["--tour-file", str(args.tour.resolve()), "--flythrough", str(args.frames / 60)]
            record = {"command": command, "environment_overrides": settings, "fixed_dt": 1 / 60,
                      "executable_sha256": digest(args.binary), "shaders_sha256": shader_hashes,
                      "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                      "retail_reference": None, "status": "recording", "performance_measurement": False}
            if args.tour:
                record.update({"tour_sha256": digest(args.tour), "tour_contents": args.tour.read_text()})
            manifest = target / "manifest.json"
            manifest.write_text(json.dumps(record, indent=2) + "\n")
            print(f"Recording {target.name}", flush=True)
            with (target / "runtime.log").open("w") as log:
                subprocess.run(command, cwd=args.binary.parent, env=environment, stdout=log,
                               stderr=subprocess.STDOUT, timeout=600, check=True)
            logs = (target / "runtime.log").read_text()
            if "shutdown leak check: no renderer-owned live VkImage handles" not in logs:
                raise RuntimeError(f"unclean shutdown: {target}")
            if "Validation Error:" in logs or "SYNC-HAZARD" in logs:
                raise RuntimeError(f"validation failure: {target}")
            if args.validation and "validation=on" not in logs:
                raise RuntimeError("validation layer not loaded")
            frames = sorted(target.glob("frame_*.ppm"))
            if len(frames) != args.frames:
                raise RuntimeError(f"expected {args.frames} frames, found {len(frames)}")
            previous = None
            changes = []
            hashes = []
            # Bound decoded images in flight while encoding independently.
            # Serial comparison still follows frame order, not worker order.
            with ThreadPoolExecutor(max_workers=4) as encoder:
                remaining = iter(frames)
                pending = deque()
                for frame in remaining:
                    pending.append(encoder.submit(convert_frame, frame, (width, height)))
                    if len(pending) == 8:
                        break
                while pending:
                    pixels, rgb_hash = pending.popleft().result()
                    hashes.append(rgb_hash)
                    if previous is not None:
                        changes.append(sum(ImageStat.Stat(ImageChops.difference(pixels, previous)).mean) / 3)
                    previous = pixels
                    frame = next(remaining, None)
                    if frame is not None:
                        pending.append(encoder.submit(convert_frame, frame, (width, height)))
            record.update({"status": "captured-not-visually-approved", "frame_count": len(frames),
                           "frame_rgb_sha256": hashes, "extent": [width, height],
                           "mean_adjacent_rgb_change_0_255": sum(changes) / len(changes) if changes else 0,
                           "note": "Pixel change includes intentional wind, water, camera motion and TAA; it is not a shimmer score."})
            manifest.write_text(json.dumps(record, indent=2) + "\n")
            sheet = Image.new("RGB", (768 * 2, 432 * 2 + 24), "#161616")
            ImageDraw.Draw(sheet).text((8, 5), target.name + " — time samples, display downsampled", fill="white")
            for i, index in enumerate([0, len(frames)//3, 2*len(frames)//3, len(frames)-1]):
                with Image.open(frames[index].with_suffix(".png")) as source:
                    sheet.paste(source.resize((768, 432)), ((i%2)*768, 24+(i//2)*432))
            sheet.save(target / "contact.png")
            write_index(args.output)
            print(f"Completed {target.name}: {len(frames)} lossless native frames", flush=True)


if __name__ == "__main__":
    main()
