#!/usr/bin/env python3
"""Native-DPI traversal timing; no per-frame readback or validation overhead."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
from PIL import Image
from capture_riverwood import ROOT, WEATHERS, capture_settings, digest


def summarize(rows):
    values = sorted(float(r['interval_ms']) for r in rows)
    def percentile(p): return values[int((len(values)-1)*p)]
    return {'frames': len(values), 'mean_ms': sum(values)/len(values),
            'fps': 1000*len(values)/sum(values), 'p50_ms': percentile(.5),
            'p95_ms': percentile(.95), 'p99_ms': percentile(.99), 'max_ms': values[-1],
            'over_16_67_ms': sum(v > 1000/60 for v in values),
            'over_33_33_ms': sum(v > 1000/30 for v in values),
            'worst_frames': sorted(rows, key=lambda r: float(r['interval_ms']), reverse=True)[:20]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--look', nargs='+', choices=WEATHERS, default=list(WEATHERS))
    parser.add_argument('--frames', type=int, default=1800)
    parser.add_argument('--warmup', type=int, default=240)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--position', default='13980,46,48800')
    parser.add_argument('--heading', default='-60')
    parser.add_argument('--tour', type=Path, default=ROOT/'scripts/fixtures/riverwood-forest-road.tour')
    args = parser.parse_args()
    if args.frames < 60 or args.warmup < 0: parser.error('invalid frame counts')
    binary = ROOT/'build-linux-relwithdebinfo/odai'
    for look in args.look:
        target = args.output.resolve()/look
        target.mkdir(parents=True) # retain evidence; never overwrite a case
        settings = capture_settings('768x432', 'small-fast', args.position, args.heading, '-2')
        for key in ['ODAI_WINDOW_HIDPI', 'ODAI_RENDER_SIZE', 'ODAI_RENDER_SCALE']:
            settings.pop(key, None)
        settings.update({'ODAI_FNV_BENCH':'1', 'ODAI_FNV_BENCH_FIXED_DT':'1',
            'ODAI_FNV_BENCH_SPEED':'250', 'ODAI_FNV_BENCH_TURN':'0',
            'ODAI_FNV_BENCH_HEADING':args.heading, 'ODAI_FNV_BENCH_WARMUP_FRAMES':str(args.warmup),
            'ODAI_FRAME_STATS':'300', 'ODAI_FRAME_STATS_CSV':str(target/'frames.csv'),
            'ODAI_DEBUG_CHUNK_TIMING':'1', 'ODAI_TOUR_STREAMING':'1',
            'ODAI_FNV_TOUR_TRACE':str(target/'camera.csv'),
            'ODAI_TOUR_FIXED_DT':'1', 'ODAI_TOUR_WARMUP_FRAMES':str(args.warmup)})
        env = {k:v for k,v in os.environ.items() if not k.startswith(('ODAI_', 'VK_LAYER')) and k != 'VK_INSTANCE_LAYERS'}
        env.update(settings)
        weather, hour = WEATHERS[look]
        command = [str(binary), '--stream', str(Path.home()/'.local/share/Steam/steamapps/common/Skyrim Special Edition/Data'),
            '--plugin','Skyrim.esm','--worldspace','Tamriel','--no-resume','--weather',weather,'--hour',str(hour),
            '--tour-file',str(args.tour.resolve()),'--flythrough',str(args.frames/60),
            '--screenshot',str(target/'endpoint.ppm'),str(args.warmup+args.frames)]
        record = {'command':command,'environment_overrides':settings,'binary_sha256':digest(binary),
            'shaders_sha256':{p.name:digest(p) for p in (ROOT/'src/render/shaders').glob('*.spv')},
            'tour_sha256':digest(args.tour), 'tour_contents':args.tour.read_text(),
            'warmup_frames':args.warmup,'simulation_dt':1/60,'validation':False,
            'measurement':'wall-clock frame intervals after stationary warmup; final still only'}
        (target/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')
        print('Benchmarking '+look, flush=True)
        with (target/'runtime.log').open('w') as log:
            subprocess.run(command, cwd=binary.parent, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=300)
        if 'shutdown leak check: no renderer-owned live VkImage handles' not in (target/'runtime.log').read_text():
            raise RuntimeError('unclean renderer shutdown')
        rows = list(csv.DictReader((target/'frames.csv').open()))
        # The interval attributed to frame N covers the work in frame N-1.
        measured = [r for r in rows if int(r['frame']) > args.warmup]
        if len(measured) < args.frames-3: raise RuntimeError('route ended early')
        record['timings'] = summarize(measured)
        camera = list(csv.DictReader((target/'camera.csv').open()))
        steps = [math.dist([float(a[k]) for k in ('x','y','z')],
                           [float(b[k]) for k in ('x','y','z')])
                 for a,b in zip(camera,camera[1:])]
        record['camera'] = {'samples':len(camera), 'max_step_units':max(steps,default=0),
            'first':camera[0], 'last':camera[-1],
            'note':'Rounded diagnostic trace; asynchronous residency timing is not deterministic.'}
        with Image.open(target/'endpoint.ppm') as im:
            record['extent'] = im.size
            if im.size != (1536,864): raise RuntimeError('native 2x DPI extent changed')
            im.save(target/'endpoint.png')
        (target/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps({k:v for k,v in record['timings'].items() if k!='worst_frames'}),flush=True)

if __name__ == '__main__': main()
